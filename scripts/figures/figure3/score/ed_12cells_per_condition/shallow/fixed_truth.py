#!/usr/bin/env python3
"""Does GENESIS rescue a shallowly sampled TRAINING condition, or inherit its sampling noise?

The 5% cells arm trained on ~7.4% of every training condition, so training depth is tied to condition size, and
so to the depth of any truth built from its held-out cells. Here the truth is FIXED at T = 300 held-out cells (the
largest T that still reaches ~10-25 training cells): training depth varies across conditions, the truth does not.
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
from cohort import KEYS, MIN_DRAWN, OUT, RUN, SEED, obs_keys  # noqa: E402
from score import EVAL_CACHE, Cells, score_vs, tile  # noqa: E402

from src.eval.reference.pairs import (  # noqa: E402
    SHORT, EvalNpzReader, control_keys, perturbed_keys, read_control, read_perturbed, replicate_design,
    set_dataset_median_umi, to_t,
)
from src.eval.reference.generation import generate_condition  # noqa: E402
from src.paths import expand_path  # noqa: E402

T = 300
NS = (10, 25, 50, 100, 200)
# training-depth strata: every shallow condition, a seeded sample of the deep ones (runtime)
STRATA = ((4, 10, None), (11, 25, None), (26, 50, 2000), (51, 100, 1000), (101, 200, 1000), (201, 10**9, 1000))
N_TEST = 1000
FT = OUT / "fixed_truth"
COMBO = ["cell_name", "drug", "dose"]


# 1. A cohort: which conditions, and a plate-B partner for each

def pick_partner(tab: pd.DataFrame, n_col: str) -> pd.Series:
    """Per condition, one other plate of its combination with >= T usable cells (seeded by the
    combination and plate, so the pick never moves with the cohort), else -1."""
    ok = tab[tab[n_col] >= T]
    out = pd.Series(-1, index=tab.cid.to_numpy())
    for key, g in ok.groupby(COMBO, observed=True, sort=True):
        if len(g) < 2:
            continue
        ids, plates = g.cid.to_numpy(), g.plate.to_numpy()
        for i, p in zip(ids, plates):
            others = ids[ids != i]
            rng = np.random.default_rng([SEED, T, *(ord(ch) for ch in "|".join(map(str, (*key, p))))])
            out[i] = int(others[rng.integers(len(others))])
    return out


def test_table() -> tuple[pd.DataFrame, np.ndarray]:
    """The run's test conditions (every row unseen), grouped like cohort.py groups training rows:
    (one row per condition with its offset into rows, every test row in condition order)."""
    obs = obs_keys()
    te = np.load(RUN / "split_cache" / "test_idx.npy")
    if np.intersect1d(te, np.load(RUN / "split_cache" / "pert_train_idx_pre_hbss.npy")).size:
        raise SystemExit("test rows overlap the training pool")
    df = obs.iloc[te].reset_index(drop=True).assign(row=te).sort_values([*KEYS, "row"], kind="stable")
    tc = df.groupby(KEYS, observed=True, sort=False).size().rename("n_tot").reset_index()
    tc["offset"] = np.concatenate([[0], np.cumsum(tc.n_tot.to_numpy())[:-1]])
    tc = tc.astype({k: str for k in KEYS}).assign(cid=np.arange(len(tc)), n_drawn=0, n_left=lambda x: x.n_tot)
    return tc, df.row.to_numpy(np.int64)


def build() -> None:
    """items.parquet: `train`, training conditions with >= T held-out cells stratified by training depth
    (STRATA); `test`, the run's test conditions, whose cells the model never saw (the zero-shot line)."""
    FT.mkdir(parents=True, exist_ok=True)
    cond = pd.read_parquet(OUT / "conditions.parquet").rename(columns={"cond_id": "cid"})
    ok = cond[(cond.n_drawn >= MIN_DRAWN) & (cond.n_left >= T)].copy()
    ok["partner"] = pick_partner(ok, "n_left").loc[ok.cid].to_numpy()
    picks = []
    for lo, hi, cap in STRATA:
        s = ok[ok.n_drawn.between(lo, hi)]
        if cap is not None and len(s) > cap:
            s = s.iloc[np.sort(np.random.default_rng([SEED, T, lo]).choice(len(s), cap, replace=False))]
        picks.append(s.assign(stratum=f"{lo}-{hi}" if hi < 10**9 else f">{lo - 1}"))
    train = pd.concat(picks).assign(cohort="train")

    tc, rows = test_table()
    tc["partner"] = pick_partner(tc, "n_tot").loc[tc.cid].to_numpy()
    big = tc[tc.n_tot >= T]
    test = big.iloc[np.sort(np.random.default_rng([SEED, T, 0]).choice(len(big), min(N_TEST, len(big)), replace=False))]
    tc.to_parquet(FT / "test_conditions.parquet", index=False)
    np.save(FT / "test_rows.npy", rows)
    items = pd.concat([train, test.assign(cohort="test", stratum="test")], ignore_index=True)
    items[["cohort", "cid", "partner", "stratum", *KEYS, "n_drawn", "n_left"]].to_parquet(FT / "items.parquet", index=False)
    print(items.groupby(["cohort", "stratum"], sort=False).agg(n=("cid", "size"), with_plate_b=("partner", lambda p: int((p >= 0).sum()))))


# 2. The rows of a condition: training conditions via Cells, test conditions from their own table

class TestCells:
    """Test-condition rows (all unseen) in Cells' interface: split() -> (rows, no drawn rows)."""

    def __init__(self, cells: Cells, root: Path = FT):
        self.cond = pd.read_parquet(root / "test_conditions.parquet")
        self.rows = np.load(root / "test_rows.npy")
        self.read = cells.read

    def split(self, cid: int) -> tuple[np.ndarray, np.ndarray]:
        c = self.cond.iloc[cid]
        if int(c.cid) != cid:
            raise RuntimeError(f"test_conditions.parquet row {cid} holds cid {c.cid}")
        return self.rows[int(c.offset): int(c.offset + c.n_tot)], np.empty(0, np.int64)


def truth_and_pool(left: np.ndarray, cid: int, cohort: str) -> tuple[np.ndarray, np.ndarray]:
    """A seeded permutation of the held-out rows: the first T are the truth, the rest the pool that
    measured_n draws from (nested: the 25 cells contain the 10)."""
    perm = np.random.default_rng([SEED, T, cohort == "test", cid]).permutation(left)
    return np.sort(perm[:T]), perm[T:]


def score_item(src, it, gen_fn, ctrl_a: torch.Tensor, ctrl_b: torch.Tensor | None, tag: dict) -> list[dict]:
    """Every arm against the same T held-out cells of plate A, each with its own plate's control pool:

      genesis      T cells generated for A
      drawn        A's training cells as measured (energy is a U-statistic: a small set is noisier, not penalized)
      copy         those cells tiled to T (the matched-depth null for DEG, whose FDR gate rewards depth)
      measured_n   n further held-out cells of A, outside the truth (n in NS): the sampling-noise curve
      plate_b      T held-out cells of the same combination on another plate B: the experimental repeat

    Test conditions have no drawn / copy arm."""
    left, drawn = src.split(it.cid)
    truth, pool = truth_and_pool(left, it.cid, it.cohort)
    if np.intersect1d(truth, drawn).size or np.intersect1d(truth, pool).size:
        raise RuntimeError(f"{tag}: truth rows overlap the training or pool rows")
    arms = {"genesis": (gen_fn(T), "log1p", ctrl_a)}
    if len(drawn):
        x = src.read(drawn)
        arms |= {"drawn": (x, "raw", ctrl_a), "copy": (tile(x, T), "raw", ctrl_a)}
    arms |= {f"measured_{n}": (src.read(np.sort(pool[:n])), "raw", ctrl_a) for n in NS if n <= len(pool)}
    if ctrl_b is not None:
        left_b, _ = src.split(it.partner)
        arms["plate_b"] = (src.read(truth_and_pool(left_b, it.partner, it.cohort)[0]), "raw", ctrl_b)
    return score_vs(src.read(truth), ctrl_a, arms, tag)


# 3. The loop

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cohort-only", action="store_true")
    ap.add_argument("--cohort", choices=("train", "test"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verify", action="store_true", help="assert truth cache == training cache per condition")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.cohort_only:
        build()
        return 0
    if a.cohort is None:
        raise SystemExit("--cohort train|test (or --cohort-only)")

    from src.model.inference_output import PertGenInferenceBundle
    device = torch.device(a.device)
    d = replicate_design("tahoe")
    set_dataset_median_umi(d.dataset)
    cells = Cells(device)
    src = cells if a.cohort == "train" else TestCells(cells)
    cond = src.cond.rename(columns={"cond_id": "cid"})
    reader = EvalNpzReader(Path(expand_path(EVAL_CACHE)))
    bundle = PertGenInferenceBundle.from_run(RUN, checkpoint="last", device=a.device, use_compile=False)
    if list(bundle.gene_names) != list(cells.cache.var_names):
        raise SystemExit("model genes != training-cache genes (order or content)")
    items = pd.read_parquet(FT / "items.parquet").query("cohort == @a.cohort")
    items = items.iloc[a.shard::a.n_shards].iloc[: a.limit or None]
    dest = FT / "scores" / f"{a.cohort}__shard{a.shard}of{a.n_shards}{f'__limit{a.limit}' if a.limit else ''}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"{a.cohort}: {len(items):,} conditions (shard {a.shard}/{a.n_shards}), T={T} -> {dest}", flush=True)

    rows, t0 = [], time.time()
    for k, it in enumerate(items.itertuples(), start=1):
        ka = {c: str(getattr(it, c)) for c in KEYS}
        if a.verify:   # the eval cache holds this condition's cells: the same multiset as our rows
            ev = np.asarray(read_perturbed(reader, perturbed_keys(d, ka, ka["plate"]), "plate"), dtype=np.float32)
            tr = np.vstack([src.read(r).cpu().numpy() for r in src.split(it.cid) if len(r)])
            if ev.shape != tr.shape or not np.array_equal(ev[np.lexsort(ev.T)], tr[np.lexsort(tr.T)]):
                raise SystemExit(f"{ka}: eval cache {ev.shape} != our rows {tr.shape}")
        ctrl_a = to_t(read_control(reader, control_keys(d, ka, ka["plate"]), "plate"), device)
        ctrl_b, plate_b = None, ""
        if it.partner >= 0:
            plate_b = str(cond.at[it.partner, "plate"])
            if any(str(cond.at[it.partner, c]) != ka[c] for c in COMBO):
                raise RuntimeError(f"{ka}: partner {it.partner} is another combination")
            ctrl_b = to_t(read_control(reader, control_keys(d, {**ka, "plate": plate_b}, plate_b), "plate"), device)
        g = torch.Generator(device=bundle.device).manual_seed(SEED + T + it.cid + (10**7 if a.cohort == "test" else 0))

        def gen_fn(n: int) -> torch.Tensor:
            return to_t(generate_condition(bundle, ka, n_cells=n, batch_size=4096, generator=g), device)

        tag = {"cohort": a.cohort, "cid": int(it.cid), "stratum": it.stratum, **ka, "plate_b": plate_b,
               "n_drawn": int(it.n_drawn), "n_left": int(it.n_left)}
        rows += score_item(src, it, gen_fn, ctrl_a, ctrl_b, tag)
        if k == 1 or k % 100 == 0 or k == len(items):
            print(f"[{k}/{len(items)}] {ka} | {time.time() - t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_parquet(dest, index=False)   # rewritten as it grows: a killed run keeps its rows
    df = pd.DataFrame(rows)
    if df[SHORT].isna().all(axis=1).any():
        raise SystemExit("some rows have every metric NaN")
    print(f"wrote {dest} ({len(df):,} rows) in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
