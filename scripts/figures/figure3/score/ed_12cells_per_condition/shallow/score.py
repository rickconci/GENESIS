#!/usr/bin/env python3
"""Step 2 of the shallow-sampling analysis: every arm of one cohort, scored against a size-matched truth.

Every reference is size-matched to the arm it judges, since DEG overlap depends on the number of cells. Cohorts:
`same_plate` (Figure 2's split-half), `cross` (the evaluation: Figure 2a's cross comparison) and, on cross's
directions, `cross_additive`, `cross_shrinkage`, `full_repeat`. Kernels are Figure 2's (`score_true_gen`); one
row per (comparison, redraw, half, arm).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cohort import HVG_CACHE, KEYS, OUT, RUN, SEED, cross_pairs, same_plate_sample  # noqa: E402

from src.data.cache.raw_hvg_cache import RawHvgShardCache  # noqa: E402
from src.eval.reference.pairs import (  # noqa: E402
    CTRL_FLOORS, SHORT, EvalNpzReader, control_keys, metric_cols, perturbed_keys, read_control, read_perturbed,
    replicate_design, resolve_ctrl_pair, score_true_gen, set_dataset_median_umi, to_t,
)
from src.eval.reference.generation import generate_condition  # noqa: E402
from src.eval.metrics.eval_metrics import score_pair  # noqa: E402
from src.eval.metrics.reporting import materialize_metric_tensor_batch  # noqa: E402
from src.eval.reference.split_half import SplitHalfSampler  # noqa: E402
from src.baselines.artifacts import Baseline  # noqa: E402
from src.baselines.roles import ArmKey  # noqa: E402
from src.eval.arms.baseline import _stable_seed, emit_per_cell_delta, normalize_plate_for_eval  # noqa: E402
from src.model.inference_output import ModelOutputSpace  # noqa: E402
from src.eval.orchestration.model_eval import _stable_cond_seed  # noqa: E402
from src.paths import expand_path  # noqa: E402

EVAL_CACHE = "{data:tahoe}/eval_cache/normalization_raw_counts/geneset_c0d6420ffc0fa861__n_2000"
N_REDRAWS = 3   # split-half redraws per condition


# 1. Cells: a condition's drawn and left-out rows, read as raw counts

class Cells:
    """Global rows per condition (from cohort.py) -> raw-count tensors from the training cache."""

    def __init__(self, device: torch.device):
        self.cond = pd.read_parquet(OUT / "conditions.parquet")
        z = np.load(OUT / "rows.npz")
        self.rows, self.drawn = z["rows"], z["drawn"]
        self.cache = RawHvgShardCache(expand_path(HVG_CACHE), eager_obs=False)
        self.device = device

    def split(self, cid: int) -> tuple[np.ndarray, np.ndarray]:
        """(left rows, drawn rows); cohort.py stores each condition's left rows first."""
        c = self.cond.iloc[cid]
        if int(c.cond_id) != cid:
            raise RuntimeError(f"conditions.parquet row {cid} holds cond_id {c.cond_id}")
        sl = slice(int(c.offset), int(c.offset + c.n_tot))
        rows, drawn = self.rows[sl], self.drawn[sl]
        if drawn[: c.n_left].any() or not drawn[c.n_left:].all():
            raise RuntimeError(f"condition {cid}: rows.npz is not ordered left-then-drawn")
        return rows[: c.n_left], rows[c.n_left:]

    def read(self, rows: np.ndarray) -> torch.Tensor:
        return to_t(self.cache.gene_expression(rows).toarray(), self.device)


def tile(x: torch.Tensor, n: int) -> torch.Tensor:
    """The copy-paste null: x's rows repeated in order until there are n. With n or more rows
    already, x is returned whole: duplication never discards a measured cell."""
    if x.shape[0] >= n:
        return x
    return x[torch.arange(n, device=x.device) % x.shape[0]]


def both_ways(truth: torch.Tensor, ctrl: torch.Tensor, rep: int, seed: int) -> list[tuple]:
    """One SplitHalfSampler partition, oriented both ways: [(truth half, ctrl, other ctrl, other rows)].
    The sampler returns only the second half's rows; the first half's come from its own seeded
    permutation (its _seed_rep then one randperm), checked against what it returned."""
    a, _b, ca, cb, b_idx = SplitHalfSampler().sample(truth, ctrl, rep=rep, seed=seed)
    torch.manual_seed(int(seed) + int(rep))
    a_idx = torch.randperm(int(truth.shape[0]), device=truth.device)[: truth.shape[0] // 2]
    if not torch.equal(truth[a_idx], a):
        raise RuntimeError("could not recover SplitHalfSampler's first half: did its draw order change?")
    return [(a, ca, cb, b_idx), (truth[b_idx], cb, ca, a_idx)]


# 2. One truth and the arms that face it

def score_vs(truth: torch.Tensor, ctrl_t: torch.Tensor, arms: dict[str, tuple[torch.Tensor, str, torch.Tensor]],
             tag: dict, energy_all_cells: bool = False) -> list[dict]:
    """Every arm (cells, 'raw' | 'log1p', its plate's controls) against one truth with controls ctrl_t.
    energy_all_cells: energy on every cell (src.eval.metrics.arm.compute_arm_metrics); rows get energy_cells=all."""
    out = []
    for arm, (cells, space, ctrl_a) in arms.items():
        c_t, c_a = resolve_ctrl_pair(ctrl_t, ctrl_a, mode="full", seed=0)   # one plate: the pool twice (cloned)
        torch.manual_seed(_stable_cond_seed(SEED, {**tag, "arm": arm}))   # energy/MMD subsample their inputs
        out.append({**tag, "arm": arm, "n_truth": int(truth.shape[0]), "n_cells": int(cells.shape[0]),
                    **({"energy_cells": "all"} if energy_all_cells else {}),   # after the seed: it must not move
                    **score_true_gen(truth, cells, c_t, c_a, space, energy_all_cells=energy_all_cells)})
    return out


def subsample(rows: np.ndarray, m: int, cid: int) -> np.ndarray:
    """m of a condition's rows, seeded by the condition alone (the same m cells in both directions)."""
    return np.sort(np.random.default_rng([SEED, cid]).permutation(rows)[:m])


def same_plate(cells: Cells, cid: int, gen_fn, ctrl: torch.Tensor, out_space: ModelOutputSpace,
               tag: dict) -> list[dict]:
    """Figure 2's split-half on a training condition's n held-out cells (never seen by the model).

    Each redraw is one src/eval SplitHalfSampler partition: the held-out cells AND the plate's control
    pool are split into disjoint halves (H1, ctrl1), (H2, ctrl2). Each half is the truth once
    (N_REDRAWS redraws x 2 halves), its response taken against its own control half:

      split_half   the other true half, against the other control half: the reference
                   (src/eval/reference split_half: raw vs raw)
      genesis      n/2 generated cells in the other half's place, same control halves
                   (src/eval/orchestration/model_eval: one fresh draw of n cells per redraw, split on
                   the same rows as the truth, so G1 faces H1 and G2 faces H2)
    """
    truth = cells.read(cells.split(cid)[0])
    seed = _stable_cond_seed(SEED, {c: tag[f"{c}_x"] for c in KEYS})
    cols = metric_cols()
    bundles, meta = [], []
    for r in range(N_REDRAWS):
        gen = gen_fn(int(truth.shape[0]))   # fresh cells per redraw, split on the truth's rows
        for half, (t, c_t, c_o, other) in enumerate(both_ways(truth, ctrl, r, seed)):
            for arm, cand, space in (("split_half", truth[other], "raw"), ("genesis", gen[other], out_space)):
                torch.manual_seed(_stable_cond_seed(SEED, {**tag, "redraw": r, "half": half, "arm": arm}))
                bundles.append(score_pair(t, cand, c_t, c_o, a_space="raw", b_space=space,
                                          ctrl_det_floors=CTRL_FLOORS))
                meta.append({**tag, "redraw": r, "half": half, "arm": arm, "n_truth": int(t.shape[0]),
                             "n_cells": int(cand.shape[0]), "n_ctrl_truth": int(c_t.shape[0])})
    flat = materialize_metric_tensor_batch(bundles)
    return [{**m, **{s: float(f[cols[s]]) for s in SHORT}} for m, f in zip(meta, flat)]


def cross(cells: Cells, cx: int, cy: int, gen_fn, ctrl_x: torch.Tensor, ctrl_y: torch.Tensor,
          tag: dict) -> list[dict]:
    """Every arm from source plate X (= A) against the target plate Y's (= B) held-out cells, at one depth m.

    Truth = m held-out cells of B, m = min(held-out on A, held-out on B). Every arm comes from A, with A's
    controls, and crosses the batch:

      repeat       m held-out cells of A: the experimental repeat, the reference (Figure 2a's
                   true_true kernel; checked to 1e-6)
      genesis      m cells generated for plate A
      drawn        A's training cells, as measured (the null for energy, where tiling is penalized)
      copy         A's training cells tiled to m (the copy-paste null at GENESIS's depth, for DEG)

    Each combination appears in both directions (A <-> B), as Figure 2a averages them. GENESIS trained
    on the training cells of BOTH plates, never on a held-out cell.
    """
    (left_x, drawn_x), (left_y, _) = cells.split(cx), cells.split(cy)
    if np.intersect1d(drawn_x, left_x).size:   # split() guarantees it; a regression here would leak
        raise RuntimeError(f"condition {cx}: training and held-out rows overlap")
    m = min(len(left_x), len(left_y))
    x = cells.read(drawn_x)
    arms = {"repeat": (cells.read(subsample(left_x, m, cx)), "raw", ctrl_x),
            "genesis": (gen_fn(m), "log1p", ctrl_x),
            "drawn": (x, "raw", ctrl_x), "copy": (tile(x, m), "raw", ctrl_x)}
    # one truth tensor faces every arm: the target cells are identical across arms by construction
    return score_vs(cells.read(subsample(left_y, m, cy)), ctrl_y, arms, {**tag, "m": m})


def write_manifest(dest: Path, cond: pd.DataFrame, items: list[tuple[int, int]]) -> None:
    """What was scored, from what: the checkpoint's checksum, the draw's real counts, every direction."""
    ckpt = RUN / "checkpoints" / "last.ckpt"
    h = hashlib.sha256()
    with open(ckpt, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    meta = json.loads((RUN / "split_cache" / "meta.json").read_text())
    directions = pd.DataFrame(items, columns=["cond_id_x", "cond_id_y"])
    for side in ("x", "y"):
        c = cond.set_index("cond_id").loc[directions[f"cond_id_{side}"]]
        for k in (*KEYS, "n_drawn", "n_left"):
            directions[f"{k}_{side}"] = c[k].to_numpy()
    directions["m"] = directions[["n_left_x", "n_left_y"]].min(axis=1)
    directions.to_parquet(dest.with_suffix(".directions.parquet"), index=False)
    dest.with_suffix(".manifest.json").write_text(json.dumps({
        "run": str(RUN), "checkpoint": str(ckpt), "checkpoint_sha256": h.hexdigest(),
        "split_meta": meta, "train_fraction_of_rows": meta["n_train"] / meta["cells_summary"]["n_before"],
        "n_directions": len(directions), "n_combinations": len(directions) // 2,
        "plate_pairs": directions.assign(pair=["/".join(sorted(t)) for t in zip(directions.plate_x, directions.plate_y)])
                                 .iloc[::2].pair.value_counts().to_dict(),
    }, indent=2))


def cross_additive(cells: Cells, cx: int, cy: int, additive, ctrl_x: torch.Tensor, ctrl_y: torch.Tensor,
                   tag: dict, kappa: float | None = None) -> list[dict]:
    """The Additive baseline refit on the arm's draw, as a cross arm: plate A's real
    control cells shifted by its predicted delta (Figure 2's emission), m of them, against the same
    plate-B truth as `cross`. With kappa the arm is the SHRINKAGE baseline:
    A's own training-cell mean pulled toward Additive, w = n / (n + kappa)."""
    (left_x, drawn_x), (left_y, _) = cells.split(cx), cells.split(cy)
    m = min(len(left_x), len(left_y))
    pool = ctrl_x.cpu().numpy()
    mean = pool.mean(axis=0)
    cell, drug, dose = tag["cell_name_x"], tag["drug_x"], float(tag["dose_x"])
    key = ArmKey(held=(cell,), kept=(), stratum=(normalize_plate_for_eval(tag["plate_x"]),), pert=(drug, round(dose, 6)))
    vec = additive.predict(key, mean)
    name = "additive"
    if kappa is not None:
        w = len(drawn_x) / (len(drawn_x) + kappa)
        vec = w * cells.read(drawn_x).double().mean(0).cpu().numpy() + (1 - w) * vec
        name = "shrinkage"
    rows = emit_per_cell_delta(pred_vec_raw=vec, ctrl_cells_raw=pool, ctrl_mean_raw=mean, n_cells=m,
                               seed=_stable_seed((cell, drug, dose, normalize_plate_for_eval(tag["plate_x"])), SEED))
    if not np.isfinite(rows).all():
        raise RuntimeError(f"{tag}: non-finite Additive emission")
    arms = {name: (to_t(rows, ctrl_x.device), "raw", ctrl_x)}
    return score_vs(cells.read(subsample(left_y, m, cy)), ctrl_y, arms, {**tag, "m": m})


def full_repeat(cells: Cells, cx: int, cy: int, gen_fn, ctrl_x: torch.Tensor, ctrl_y: torch.Tensor,
                tag: dict) -> list[dict]:
    """Figure 2a's repeat at full depth: every cell of plate X (trained + held-out) vs every cell of Y.
    No model is scored, so training cells may enter: it measures what depth m costs the repeat."""
    rx, ry = (np.concatenate(cells.split(c)) for c in (cx, cy))
    return score_vs(cells.read(ry), ctrl_y, {"repeat_full": (cells.read(rx), "raw", ctrl_x)}, {**tag, "m": len(ry)})


# 3. The loop

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cohort", choices=("cross", "cross_additive", "cross_shrinkage", "full_repeat", "same_plate"),
                    required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N comparisons of this shard (smoke test)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--items", default="", help="comma-separated indices into the cohort (smoke cases)")
    ap.add_argument("--plate-pair", default="", help="only this plate pair, e.g. plate14/plate6 (order-free)")
    ap.add_argument("--verify", action="store_true", help="assert truth cache == training cache per condition")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=OUT / "scores")
    a = ap.parse_args()

    from src.model.inference_output import PertGenInferenceBundle
    device = torch.device(a.device)
    d = replicate_design("tahoe")
    set_dataset_median_umi(d.dataset)
    cells = Cells(device)
    reader = EvalNpzReader(Path(expand_path(EVAL_CACHE)))
    bundle = PertGenInferenceBundle.from_run(RUN, checkpoint="last", device=a.device, use_compile=False)
    if list(bundle.gene_names) != list(cells.cache.var_names):
        raise SystemExit("model genes != training-cache genes (order or content)")

    additive, kappa = None, None
    if a.cohort in ("cross_additive", "cross_shrinkage"):
        additive = Baseline.load(OUT / "additive_refit.npz")
    if a.cohort == "cross_shrinkage":
        k = json.loads((OUT / "shrinkage_kappa.json").read_text())["kappa"]
        kappa = float("inf") if k == "inf" else float(k)
        print(f"shrinkage kappa = {kappa}", flush=True)
    cond = cells.cond
    if a.cohort == "same_plate":
        items = [(int(c), int(c)) for c in same_plate_sample(cond)]
    else:
        p = cross_pairs(cond)
        items = list(zip(p.cond_id_x, p.cond_id_y))
    if a.plate_pair:   # the cross-batch result is read on plates 6/14 only
        want = sorted(a.plate_pair.split("/"))
        items = [(x, y) for x, y in items if sorted((cond.at[x, "plate"], cond.at[y, "plate"])) == want]
        if not items:
            raise SystemExit(f"no directions on plate pair {a.plate_pair}")
    a.out.mkdir(parents=True, exist_ok=True)
    if a.cohort == "cross" and a.shard == 0 and not (a.limit or a.items or a.plate_pair):
        write_manifest(a.out / "cross", cond, items)
    if a.items:
        items, tag_run = [items[int(i)] for i in a.items.split(",")], "__items"
    else:
        items, tag_run = items[a.shard::a.n_shards][: a.limit or None], f"__limit{a.limit}" if a.limit else ""
    tag_run += f"__{a.plate_pair.replace('/', '-')}" if a.plate_pair else ""
    dest = a.out / f"{a.cohort}__shard{a.shard}of{a.n_shards}{tag_run}.parquet"
    print(f"{a.cohort}: {len(items):,} comparisons (shard {a.shard}/{a.n_shards}) -> {dest}", flush=True)

    rows, t0 = [], time.time()
    for k, (cx, cy) in enumerate(items, start=1):
        kx, ky = ({c: str(cond.at[i, c]) for c in KEYS} for i in (cx, cy))
        if a.verify:   # the truth cache holds this condition's cells: the same multiset as the training rows
            ev = np.asarray(read_perturbed(reader, perturbed_keys(d, ky, ky["plate"]), "plate"), dtype=np.float32)
            tr = np.vstack([cells.read(r).cpu().numpy() for r in cells.split(cy)])
            if ev.shape != tr.shape or not np.array_equal(ev[np.lexsort(ev.T)], tr[np.lexsort(tr.T)]):
                raise SystemExit(f"{ky}: truth cache {ev.shape} != training cache {tr.shape} (cells or gene order)")
        ctrl = {kk["plate"]: to_t(read_control(reader, control_keys(d, kk, kk["plate"]), "plate"), device)
                for kk in (ky, kx)}
        g = torch.Generator(device=bundle.device).manual_seed(SEED + cx)   # GENESIS cells for X, seeded by X

        def gen_fn(n: int) -> torch.Tensor:
            return to_t(generate_condition(bundle, kx, n_cells=n, batch_size=4096, generator=g), device)

        tag = {"cohort": a.cohort, "cond_id_x": cx, "cond_id_y": cy, **{f"{c}_x": kx[c] for c in KEYS},
               "plate_y": ky["plate"], "n_drawn": int(cond.at[cx, "n_drawn"]),
               "n_left_x": int(cond.at[cx, "n_left"]), "n_left_y": int(cond.at[cy, "n_left"])}
        if cx == cy:
            rows += same_plate(cells, cx, gen_fn, ctrl[kx["plate"]], bundle.output_space, tag)
        else:
            if a.cohort in ("cross_additive", "cross_shrinkage"):
                rows += cross_additive(cells, cx, cy, additive, ctrl[kx["plate"]], ctrl[ky["plate"]], tag, kappa)
            else:
                pair_fn = {"cross": cross, "full_repeat": full_repeat}[a.cohort]
                rows += pair_fn(cells, cx, cy, gen_fn, ctrl[kx["plate"]], ctrl[ky["plate"]], tag)
        if k == 1 or k % 100 == 0 or k == len(items):
            print(f"[{k}/{len(items)}] {kx} -> {ky['plate']} | {time.time() - t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_parquet(dest, index=False)   # rewritten as it grows: a killed run keeps its rows
    df = pd.DataFrame(rows)
    if df[SHORT].isna().all(axis=1).any():
        raise SystemExit("some rows have every metric NaN")
    print(f"wrote {dest} ({len(df):,} rows) in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
