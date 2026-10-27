#!/usr/bin/env python3
"""Does GENESIS recover a condition's biology from 25 cells, or reproduce their sampling error?

The n_cells_percond25 run trained on exactly 25 perturbed cells per combination (all controls kept; ~12 per plate
when it spans plates): training depth is fixed, so a deep truth costs no shallow conditions. The yardstick is n TRUE
cells, drawn again and again: where GENESIS lands on that curve is its cells-equivalent (~25: it reproduced its
sample; >> 25: it corrected it). Energy is a level only: measured cells share the truth's distribution.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cohort import ARM, KEYS, OUT, REPO, RUN, SEED, SPLIT_PLATES, eligible  # noqa: E402
from fixed_truth import TestCells, test_table  # noqa: E402
from score import EVAL_CACHE, Cells, score_vs, tile  # noqa: E402

from src.eval.reference.pairs import (  # noqa: E402
    SHORT, EvalNpzReader, control_keys, perturbed_keys, read_control, read_perturbed, replicate_design,
    set_dataset_median_umi, to_t,
)
from src.eval.reference.generation import generate_condition  # noqa: E402
from src.paths import expand_path  # noqa: E402

if ARM != "pc25":
    raise SystemExit(f"percond.py scores the 25-cells-per-combination run: set SHALLOW_ARM=pc25 (got {ARM})")
PC = OUT / "percond"
NS = (10, 25, 50, 100, 200, 300)
NS_DEEP = (500, 700, 1000)   # cross_deep: plate A's whole held-out set is free, so its curve runs on to the repeat
N_SPLITS, R = 3, 2          # same_plate/test: 3 splits x 2 draws = 6 draws per n; cross: R draws per n
MIN_LEFT = 600              # same_plate/test: a pool of 300 reaches the curve's last n
SIZES = (600, 1200, 2400, 10**9)
PER_STRATUM = 350           # x (one plate / several) x 3 sizes -> ~2,100 conditions
N_TEST = 1000
# extra arms scored on an existing cohort's items (same seeds, same truth): cohort -> the cohort it extends
EXTRA = {"cross_pooled": "cross", "cross_deep": "cross", "same_plate_pool": "same_plate", "test_pool": "test"}


# 1. The cohorts

def build(pairs_data: Path) -> None:
    """items.parquet, the items of each cohort; the EXTRA cohorts score other arms on their base cohort's items.

      same_plate  sampled training conditions (PER_STRATUM per plates x size stratum): `split_half`
      test        N_TEST test conditions, same protocol: the model saw none of their cells (zero-shot)
      cross       both directions of every eligible plate-6/14 combination, all held-out cells: `cross`
      test_cross  Figure 3a's 530 held-out plate pairs, both directions: `cross` + `cross_deep` on test conditions
    """
    PC.mkdir(parents=True, exist_ok=True)
    cond = pd.read_parquet(OUT / "conditions.parquet").rename(columns={"cond_id": "cid"})
    n_plates = cond.groupby(["cell_name", "drug", "dose"]).plate.transform("size")
    cond["plates"] = np.where(n_plates > 1, "several", "one")
    cond["size"] = pd.cut(cond.n_left, SIZES, right=False, labels=[f"{a}-{b}" for a, b in zip(SIZES, SIZES[1:])])
    ok = cond[(cond.n_drawn >= 4) & (cond.n_left >= MIN_LEFT)]
    picks = []
    for k, ((pl, sz), g) in enumerate(ok.groupby(["plates", "size"], observed=True, sort=True)):
        take = np.sort(np.random.default_rng([SEED, 25, k]).choice(len(g), min(PER_STRATUM, len(g)), replace=False))
        picks.append(g.iloc[take])
    same = pd.concat(picks).assign(cohort="same_plate", cid_y=-1)

    tc, rows = test_table()
    tc.to_parquet(PC / "test_conditions.parquet", index=False)
    np.save(PC / "test_rows.npy", rows)
    big = tc[tc.n_tot >= MIN_LEFT]
    # seed tags 101/102 sit off the strata's 0..5
    test = big.iloc[np.sort(np.random.default_rng([SEED, 25, 101]).choice(len(big), min(N_TEST, len(big)), replace=False))]
    test = test.assign(cohort="test", cid_y=-1, plates="", size="")

    cross = plate614(cond)   # every eligible plate-6/14 combination, both directions

    # test_cross: Figure 3a's own held-out plate pairs (scripts.figures.figure3.score.scaling.load_pairs: 530 pairs), both directions
    from scripts.figures.figure3.score.scaling import load_pairs   # Figure 2a's pairs: pairs_data/plate_pairs/
    key = {(r.cell_name, r.drug, r.dose, r.plate): r.cid for r in tc.itertuples()}
    tx = []
    for c, dr, ds, i, j in load_pairs(pairs_data).index:
        a, b = (key.get((c, dr, str(float(ds)), f"plate{q}")) for q in (i, j))
        if a is None or b is None:
            raise SystemExit(f"Figure 3a pair {(c, dr, ds, i, j)} is not among this run's test conditions")
        tx += [(a, b), (b, a)]
    tx = pd.DataFrame(tx, columns=["cid", "cid_y"])
    test_cross = tc.set_index("cid").loc[tx.cid].reset_index().assign(cohort="test_cross", cid_y=tx.cid_y.to_numpy(), plates="", size="")

    cols = ["cohort", "cid", "cid_y", "plates", "size", *KEYS, "n_drawn", "n_left"]
    items = pd.concat([same[cols], test[cols], cross[cols], test_cross[cols]], ignore_index=True)
    items.to_parquet(PC / "items.parquet", index=False)
    print(items.groupby(["cohort", "plates", "size"], observed=True, sort=False)
          .agg(n=("cid", "size"), n_drawn=("n_drawn", "median"), n_left=("n_left", "median")))


def plate614(cond: pd.DataFrame) -> pd.DataFrame:
    """Both directions of every combination eligible (the repeat gate: >= 4 drawn, >= 50 held out) on
    plate 6 and on plate 14."""
    ok = cond[eligible(cond) & cond.plate.isin(SPLIT_PLATES)]
    both = ok[ok.plate == "plate6"].merge(ok[ok.plate == "plate14"], on=["cell_name", "drug", "dose"], suffixes=("_6", "_14"))

    def side(src: str, dst: str) -> pd.DataFrame:
        return pd.DataFrame({
            "cohort": "cross", "cid": both[f"cid_{src}"].astype(np.int64), "cid_y": both[f"cid_{dst}"].astype(np.int64),
            "plates": both[f"plates_{src}"], "size": both[f"size_{src}"].astype(object),
            "cell_name": both.cell_name, "drug": both.drug, "dose": both.dose, "plate": both[f"plate_{src}"],
            "n_drawn": both[f"n_drawn_{src}"].astype(np.int64), "n_left": both[f"n_left_{src}"].astype(np.int64),
        })

    out = pd.concat([side("6", "14"), side("14", "6")], ignore_index=True)
    return out.sort_values(["cid", "cid_y"], kind="mergesort").reset_index(drop=True)


def already_scored(cohort: str, scores: Path) -> set[tuple[int, int]]:
    """Directions already in this cohort's shard files. Smoke files (``__limit``) are not scores."""
    files = [p for p in scores.glob(f"{cohort}__shard*of*.parquet")
             if "limit" not in p.name and "backup" not in str(p) and "superseded" not in str(p)]
    if not files:
        return set()
    df = pd.concat([pd.read_parquet(p, columns=["cid", "cid_y"]) for p in files], ignore_index=True)
    return set(zip(df.cid.astype(np.int64), df.cid_y.astype(np.int64)))


# 2. The arms against one truth

def measured(pool: np.ndarray, n_truth: int, read, rng: np.random.Generator, draws: int, split: int, ctrl,
             ns: tuple = NS) -> dict:
    """n true cells drawn from the pool, as measured and tiled to the truth size (the DEG comparison)."""
    arms = {}
    for n in ns:
        if n > len(pool):
            continue
        for k in range(draws):
            x = read(np.sort(rng.choice(pool, n, replace=False)))
            r = split * draws + k
            arms[f"m{n}_r{r}"] = (x, "raw", ctrl)
            arms[f"m{n}_r{r}_tiled"] = (tile(x, n_truth), "raw", ctrl)
    return arms


def halves(left: np.ndarray, cid: int, cohort: str, split: int):
    """Seeded truth/pool cut of the held-out rows. Only `cohort == "test"` enters the seed,
    so a later cohort on the same cid reproduces the same_plate halves. The generator is
    returned mid-stream: `measured` draws from it, and that stream is part of the stored scores."""
    rng = np.random.default_rng([SEED, 25, cohort == "test", int(cid), int(split)])
    perm = rng.permutation(left)
    return np.sort(perm[: len(perm) // 2]), perm[len(perm) // 2:], rng


def split_half(src, it, gen_fn, ctrl: torch.Tensor, tag: dict, pool_only: bool = False) -> list[dict]:
    """same_plate and test: N_SPLITS seeded halvings of the held-out cells; every arm against the truth half
    (the split-half reference, truth half vs pool half, is the curve's top end):

      genesis       as many cells as the truth half
      drawn / copy  this plate's training cells, as measured / tiled to the truth size
      m{n}_r{k}     n cells from the pool (R draws per n per split), and *_tiled to the truth size (DEG's
                    matched depth: its FDR gate rewards cell count)

    pool_only (*_pool cohorts): the same halvings, one arm -- the WHOLE pool, the curve's endpoint (split-half)."""
    left, drawn = src.split(it.cid)
    out = []
    for s in range(N_SPLITS):
        truth, pool, rng = halves(left, it.cid, it.cohort, s)
        h = len(truth)
        if pool_only:   # the pool is the other half (h or h + 1 cells): no tiling needed
            out += score_vs(src.read(truth), ctrl, {"pool_all": (src.read(np.sort(pool)), "raw", ctrl)}, {**tag, "split": s})
            continue
        arms = {"genesis": (gen_fn(h), "log1p", ctrl)}
        if len(drawn):
            x = src.read(drawn)
            arms |= {"drawn": (x, "raw", ctrl), "copy": (tile(x, h), "raw", ctrl)}
        arms |= measured(pool, h, src.read, rng, R, s, ctrl)
        out += score_vs(src.read(truth), ctrl, arms, {**tag, "split": s})
    return out


def cross(cells: Cells, it, gen_fn, ctrl_a: torch.Tensor, ctrl_b: torch.Tensor, tag: dict,
          energy_all_cells: bool = False) -> list[dict]:
    """Plate A's arms, with A's controls, against all of plate B's held-out cells (no split: the truth is on the
    other plate). cells: training conditions (Cells) or test conditions (TestCells: no drawn / copy).

      repeat        all of A's held-out cells (Figure 2a's full-depth repeat: the curve's endpoint)
      genesis       as many cells as A holds out (the repeat's depth)
      drawn / copy  A's training cells, as measured / tiled to that depth
      m{n}_r{k}     n of A's held-out cells, and *_tiled
    """
    (left_a, drawn_a), (left_b, _) = cells.split(it.cid), cells.split(it.cid_y)
    m = len(left_a)
    arms = {"repeat": (cells.read(left_a), "raw", ctrl_a), "genesis": (gen_fn(m), "log1p", ctrl_a)}
    if len(drawn_a):
        x = cells.read(drawn_a)
        arms |= {"drawn": (x, "raw", ctrl_a), "copy": (tile(x, m), "raw", ctrl_a)}
    arms |= measured(left_a, m, cells.read, np.random.default_rng([SEED, 25, it.cid, it.cid_y]), R, 0, ctrl_a)
    return score_vs(cells.read(left_b), ctrl_b, arms, {**tag, "split": 0}, energy_all_cells)


def cross_deep(cells: Cells, it, ctrl_a: torch.Tensor, ctrl_b: torch.Tensor, tag: dict,
               energy_all_cells: bool = False) -> list[dict]:
    """`cross`'s curve continued past 300 (NS_DEEP, where plate A holds that many), same truth: it then runs
    from 300 cells to the repeat, which is the curve at n = all of A's held-out cells."""
    (left_a, _), (left_b, _) = cells.split(it.cid), cells.split(it.cid_y)
    arms = measured(left_a, len(left_a), cells.read, np.random.default_rng([SEED, 25, it.cid, it.cid_y, 1]), R, 0, ctrl_a, NS_DEEP)
    return score_vs(cells.read(left_b), ctrl_b, arms, {**tag, "split": 0}, energy_all_cells) if arms else []


def cross_pooled(cells: Cells, it, ctrl_a: torch.Tensor, ctrl_b: torch.Tensor, tag: dict) -> list[dict]:
    """Everything GENESIS saw of the combination: the 25 training cells of BOTH plates (A's ~12 and B's
    ~13; the cap is per combination), against the same truth as `cross` (B's held-out cells). B's cells
    share the truth's plate, so this is the strongest 'what it saw' arm; their controls are both pools."""
    (left_a, drawn_a), (left_b, drawn_b) = cells.split(it.cid), cells.split(it.cid_y)
    x, ctrl = cells.read(np.sort(np.concatenate([drawn_a, drawn_b]))), torch.cat([ctrl_a, ctrl_b])
    arms = {"pooled": (x, "raw", ctrl), "pooled_tiled": (tile(x, len(left_a)), "raw", ctrl)}
    return score_vs(cells.read(left_b), ctrl_b, arms, {**tag, "split": 0, "n_drawn_b": len(drawn_b)})


# 3. The loop

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cohort-only", action="store_true")
    ap.add_argument("--pairs-data", type=Path, help="--cohort-only: the dir holding Figure 2a's plate_pairs/ "
                    "(core_deliverables/data/figure2/main), whose 530 pairs make the test_cross cohort")
    ap.add_argument("--cohort", choices=("same_plate", "test", "cross", "test_cross", *EXTRA))
    ap.add_argument("--mem-gb", type=float, default=0, help="cap this process on a shared GPU; 0 leaves it uncapped")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verify", action="store_true", help="assert eval cache == our rows per condition")
    ap.add_argument("--resume", action="store_true", help="keep this shard's saved rows and skip their items")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--energy-all-cells", action="store_true",
                    help="energy distance on every cell, not a 2,048-cell subsample (cross, test_cross: a whole "
                         "other plate is the truth); writes scores_energy_all_cells/")
    a = ap.parse_args()
    if a.cohort_only:
        if a.pairs_data is None:
            raise SystemExit("--cohort-only needs --pairs-data")
        build(a.pairs_data)
        return 0
    if a.cohort is None:
        raise SystemExit("--cohort same_plate|test|cross (or --cohort-only)")
    base = EXTRA.get(a.cohort, a.cohort)   # the cohort whose items (and seeds) an extra cohort reuses
    if a.energy_all_cells and a.cohort not in ("cross", "test_cross"):
        raise SystemExit(f"--energy-all-cells scores cross and test_cross, not {a.cohort}")
    scores = PC / ("scores_energy_all_cells" if a.energy_all_cells else "scores")

    from src.model.inference_output import PertGenInferenceBundle
    device = torch.device(a.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", 0)
    if a.mem_gb:
        if device.type != "cuda":
            raise SystemExit("--mem-gb is a CUDA cap")
        frac = a.mem_gb * (1024 ** 3) / torch.cuda.get_device_properties(device).total_memory
        if not 0 < frac < 1:
            raise SystemExit(f"--mem-gb {a.mem_gb} is {frac:.2f} of {device}")
        torch.cuda.set_per_process_memory_fraction(frac, device.index)
        print(f"capping this process at {a.mem_gb:g} GB on {device}", flush=True)
    d = replicate_design("tahoe")
    set_dataset_median_umi(d.dataset)
    cells = Cells(device)
    src = TestCells(cells, PC) if base.startswith("test") else cells
    reader = EvalNpzReader(Path(expand_path(EVAL_CACHE)))
    bundle = PertGenInferenceBundle.from_run(RUN, checkpoint="last", device=str(device), use_compile=False)
    if list(bundle.gene_names) != list(cells.cache.var_names):
        raise SystemExit("model genes != training-cache genes (order or content)")
    # Shard the full list, THEN drop what is scored: filtering first made the split depend on what other
    # shards had saved by the time this one started, so concurrent or chained shards missed directions.
    items = pd.read_parquet(PC / "items.parquet").query("cohort == @base").iloc[a.shard::a.n_shards]
    if a.cohort in ("cross", "cross_deep"):
        done = already_scored(a.cohort, scores)
        keep = [(int(c), int(y)) not in done for c, y in zip(items.cid, items.cid_y)]
        print(f"{a.cohort}: {len(items) - sum(keep):,} of this shard's directions already scored, "
              f"{sum(keep):,} left", flush=True)
        items = items[keep]
    items = items.iloc[: a.limit or None]
    dest = scores / f"{a.cohort}__shard{a.shard}of{a.n_shards}{f'__limit{a.limit}' if a.limit else ''}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if a.cohort in ("cross", "cross_deep") and dest.is_file() and not a.resume:
        a.resume = True   # earlier directions already live in this file; overwriting would drop them
        print(f"{dest.name} exists; keeping its rows", flush=True)
    print(f"{a.cohort}: {len(items):,} items (shard {a.shard}/{a.n_shards}) -> {dest}", flush=True)

    def ctrl_of(keys: dict) -> torch.Tensor:
        return to_t(read_control(reader, control_keys(d, keys, keys["plate"]), "plate"), device)

    rows, t0 = [], time.time()
    if a.resume and dest.is_file():   # saved rows hold whole items (written between items), so skip by item
        rows = pd.read_parquet(dest).to_dict("records")
        done = {(r["cid"], r["cid_y"]) for r in rows}
        items = items[[(c, y) not in done for c, y in zip(items.cid, items.cid_y)]]
        print(f"resume: {len(done):,} items kept, {len(items):,} to go", flush=True)
    for k, it in enumerate(items.itertuples(), start=1):
        ka = {c: str(getattr(it, c)) for c in KEYS}
        if a.verify:
            ev = np.asarray(read_perturbed(reader, perturbed_keys(d, ka, ka["plate"]), "plate"), dtype=np.float32)
            tr = np.vstack([src.read(r).cpu().numpy() for r in src.split(it.cid) if len(r)])
            if ev.shape != tr.shape or not np.array_equal(ev[np.lexsort(ev.T)], tr[np.lexsort(tr.T)]):
                raise SystemExit(f"{ka}: eval cache {ev.shape} != our rows {tr.shape}")
        g = torch.Generator(device=bundle.device).manual_seed(SEED + 25 + it.cid + (10**7 if base.startswith("test") else 0))

        def gen_fn(n: int) -> torch.Tensor:
            return to_t(generate_condition(bundle, ka, n_cells=n, batch_size=4096, generator=g), device)

        tag = {"cohort": a.cohort, "cid": int(it.cid), "cid_y": int(it.cid_y), "plates": it.plates, "size": it.size,
               **ka, "n_drawn": int(it.n_drawn), "n_left": int(it.n_left)}
        if a.cohort.startswith("cross") or a.cohort == "test_cross":
            kb = {c: str(src.cond.at[it.cid_y, c]) for c in KEYS}
            if any(kb[c] != ka[c] for c in ("cell_name", "drug", "dose")) or kb["plate"] == ka["plate"]:
                raise RuntimeError(f"{ka} -> {kb}: not the same combination on another plate")
            e = a.energy_all_cells
            fn = {"cross": lambda *x: cross(cells, it, gen_fn, *x, energy_all_cells=e), "cross_pooled": lambda *x: cross_pooled(cells, it, *x),
                  "cross_deep": lambda *x: cross_deep(cells, it, *x),
                  "test_cross": lambda *x: (cross(src, it, gen_fn, *x, energy_all_cells=e)
                                            + cross_deep(src, it, *x, energy_all_cells=e))}[a.cohort]
            rows += fn(ctrl_of(ka), ctrl_of(kb), {**tag, "plate_y": kb["plate"]})
        else:
            rows += split_half(src, it, gen_fn, ctrl_of(ka), tag, pool_only=a.cohort.endswith("_pool"))
        if k == 1 or k % 10 == 0 or k == len(items):
            print(f"[{k}/{len(items)}] {ka} | {time.time() - t0:.0f}s", flush=True)
            if rows:   # rewritten as it grows (a killed run keeps its rows); via rename, since concurrent
                tmp = dest.with_name(dest.name + ".tmp")   # shards read every file at startup
                pd.DataFrame(rows).to_parquet(tmp, index=False)
                tmp.replace(dest)
    df = pd.DataFrame(rows)
    if df[SHORT].isna().all(axis=1).any():
        raise SystemExit("some rows have every metric NaN")
    print(f"wrote {dest} ({len(df):,} rows) in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
