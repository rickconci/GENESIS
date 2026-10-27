#!/usr/bin/env python3
"""Reference scores: how close two real measurements of the same condition are.

  split_half       one condition's cells split into disjoint halves, scored against each other
                   (the dashed line on every metric-bar panel) -> noise_ceiling.json
  replicate_pairs  a model against the experimental repeat: one row per pair of replicates of a
                   condition (true_true, same_*, cross_*, gen_gen) -> pair table (Fig 2a/b, 3a)
  ladder           split-half vs the repeat at matched cells per arm (Fig 1d) -> reps.parquet
  replicate_pairs_baselines  the four linear baselines vs the repeat, on the GENESIS pair list
                   (reads model_vs_batch_gap__genesis.parquet in --results, writes beside it)
  control_swap     one condition's cells against two different control wells, real and
                   shuffled (PBMC PBS wells) -> pairs.csv + summary.json
  well_draws       the same cells against equal-size control samples drawn within one well
                   and between two wells -> rows.csv + energy.csv + strata.csv + summary.json

  python scripts/eval/reference.py split_half --dataset tahoe \\
      --split-config core_util_data/splits/tahoe/bio/tahoe_legacy_bio_holdout.yaml \\
      --eval-cache '{data:tahoe}/eval_cache/normalization_raw_counts/geneset_c0d6420ffc0fa861__n_2000' \\
      --gene-names core_util_data/tahoe/gene_subsets/STATE_HVG.txt
  python scripts/eval/reference.py replicate_pairs --dataset tahoe --eval-cache <ns> \\
      --conditions multi_plate_test_conditions.tsv --model genesis --fmt npy --pred-dir <run>/predictions --out p.parquet
  python scripts/eval/reference.py ladder --dataset tahoe --eval-cache <ns> --cohort cohort.csv --out <dir>
  python scripts/eval/reference.py control_swap --dataset parse --eval-cache <ns> \\
      --conditions-from <genesis metrics_summary.json> --unit-cols donor cell_type \\
      --swap-col bc1_well --swap-values H7 H8 H9 H10 H11 H12 --out-prefix <dir>/cytokine_deg_pbs_well_swap

Condition, replicate, control and control-match columns come from the dataset registry
(core_util_data/datasets/<dataset>.yaml). Every random draw is seeded per condition, so a
rerun is bit for bit and a subset scores as it does in the full run.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import zlib
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.vocab import to_categorical_str  # noqa: E402
from src.eval.reference.control_wells import (  # noqa: E402
    DEG_KEY, PEARSON_KEY, half_size, pseudo_wells, summarize, swap_pairs, well_draws, within_between_tests,
)
from src.eval.reference.pairs import (  # noqa: E402
    DEFAULT_REPORT_SPACE, REPORT_SPACES, EvalNpzReader, control_keys, perturbed_keys,
    read_control, read_perturbed, replicate_design, _replicate_label, set_dataset_median_umi, to_t,
)
from src.eval.reference.ladder import LADDER, STRATEGIES, score_condition  # noqa: E402
from src.eval.reference.cli import (  # noqa: E402
    add_replicate_pairs_parser, add_split_half_parser, cmd_replicate_pairs, cmd_split_half,
    device_for, namespace,
)

log = logging.getLogger("reference")


# split_half and replicate_pairs live in src/eval/reference/cli.py.

# 1. replicate_pairs_baselines: the four linear baselines on the pairs GENESIS already scored

def cmd_replicate_pairs_baselines(a: argparse.Namespace) -> int:
    """The baselines vs the repeat on the GENESIS pair list (src/eval/reference/baselines.py), written beside it."""
    from src.eval.reference.baselines import run  # heavy imports only on this path
    return run(a.device, a.overwrite, a.check_only, a.results, a.report_space, a.baseline_root, a.check_pred_root,
               a.energy_all_cells)


# 2. ladder: split-half vs the repeat as a function of depth, one rung at a time

def cmd_ladder(a: argparse.Namespace) -> int:
    """Both strategies at every rung of LADDER for each cohort pair -> reps.parquet, points.parquet, manifest.json."""
    d = replicate_design(a.dataset)
    rep, max_n = d.replicate_col, max(LADDER)
    log.info("dataset=%s L_ref=%s ladder=%s", d.dataset, set_dataset_median_umi(d.dataset), LADDER)
    ns = namespace(a.eval_cache)
    cohort = pd.read_csv(a.cohort)  # the published cohort: its row index seeds each condition
    cohort = cohort.head(a.limit) if a.limit else cohort
    reader, device = EvalNpzReader(ns), device_for(a.device)
    a.out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for i, r in cohort.iterrows():
        cond = {c: to_categorical_str(r[c]) for c in d.condition_cols}
        reps = (r[f"{rep}_i"], r[f"{rep}_j"])
        truth, ctrl = {}, {}
        for p in reps:
            # max_cells subsamples with the reader's seed; both pools must clear the deepest rung twice
            tn = read_perturbed(reader, perturbed_keys(d, cond, _replicate_label(d, str(p))), rep, max_cells=2 * max_n,
                                seed=a.seed + int(i))
            cn = read_control(reader, control_keys(d, cond, _replicate_label(d, str(p))), rep, max_cells=2 * max_n,
                              seed=a.seed + 10_000 + int(i))
            if min(tn.shape[0], cn.shape[0]) < 2 * max_n:
                raise SystemExit(f"{cond} {rep} {p}: {tn.shape[0]} perturbed / {cn.shape[0]} control cells, "
                                 f"need {2 * max_n} of each; this cohort does not clear the ladder")
            truth[p], ctrl[p] = to_t(tn, device), to_t(cn, device)
        for cr in score_condition(truth=truth, ctrl=ctrl, plate_i=reps[0], plate_j=reps[1],
                                  n_repeats=a.n_repeats, seed=a.seed + 1000 * int(i)):
            rows.append({**cr, **{c: r[c] for c in d.condition_cols}, f"{rep}_i": reps[0], f"{rep}_j": reps[1]})
        log.info("[%d/%d] %s %s %s+%s | %.0fs", i + 1, len(cohort), cond, rep, *reps, time.time() - t0)
    reps_df = pd.DataFrame(rows)
    reps_df.to_parquet(a.out / "reps.parquet", index=False)
    ids = [*d.condition_cols, f"{rep}_i", f"{rep}_j", "strategy", "n_cells"]
    metrics = [c for c in reps_df.columns if c not in [*ids, "rep", "sh_plate"]]
    points = reps_df.groupby(ids)[metrics].agg(["mean", "std"]).reset_index()
    points.columns = ["__".join(c).rstrip("_") for c in points.columns]
    points.to_parquet(a.out / "points.parquet", index=False)
    (a.out / "manifest.json").write_text(json.dumps({
        "ladder_cells_per_arm": list(LADDER), "strategies": list(STRATEGIES), "cohort": str(a.cohort),
        "n_conditions_scored": len(cohort), "n_repeats": a.n_repeats, "seed": a.seed, "namespace": str(ns),
        "control_cells_per_arm": "n, identical across strategies",
        "cross_plate_controls": "replicate-matched (each arm keeps its own replicate's pool)",
        "split_half_plate": "drawn at random per repeat from the pair", "n_metrics": len(metrics),
    }, indent=2))
    log.info("wrote %s (%d rows, %d metrics)", a.out, len(reps_df), len(metrics))
    return 0


# 3. control_swap: perturbed cells fixed, only the control well changes

def cmd_control_swap(a: argparse.Namespace) -> int:
    """Each condition against every pair of its unit's control wells, real and shuffled -> pairs.csv, summary.json."""
    summary_in = json.loads(a.conditions_from.read_text())
    # the run's own column order, not condition_keys' (sorted): the bootstrap resamples rows
    # in that order, so another order gives other CIs
    if not (cols := summary_in.get("condition_variables")):
        raise SystemExit(f"{a.conditions_from.name}: no condition_variables recorded")
    conds = pd.DataFrame([r["condition_keys"] for r in summary_in["results"]])[list(cols)]
    if missing := sorted({*a.unit_cols, a.swap_col} - set(conds.columns)):
        raise SystemExit(f"{a.conditions_from.name}: condition_keys lack {missing}")
    if conds.duplicated().any():
        raise SystemExit(f"{a.conditions_from.name}: duplicate conditions")
    wells, per_cond = list(a.swap_values), comb(len(a.swap_values), 2)
    tag = f"_shard{a.shard}of{a.n_shards}" if a.n_shards > 1 else (f"_limit{a.limit}" if a.limit else "")
    out = {k: Path(f"{a.out_prefix}_{k}{tag}.{'csv' if k == 'pairs' else 'json'}") for k in ("pairs", "summary", "skipped")}
    if a.merge:
        return merge_control_swap(a, conds, per_cond)
    log.info("L_ref=%s", set_dataset_median_umi(a.dataset))
    reader, device = EvalNpzReader(namespace(a.eval_cache)), device_for(a.device)
    conds = conds.head(a.limit) if a.limit else conds
    rows, skipped, t0 = [], {"control_missing": 0, "control_small": 0, "perturbed_small": 0}, time.time()
    for s, (unit, sub) in enumerate(conds.groupby(list(a.unit_cols), sort=True)):
        if s % a.n_shards != a.shard:
            continue
        unit = dict(zip(a.unit_cols, unit))
        try:
            real = {w: reader.control({**unit, a.swap_col: w}) for w in wells}
        except FileNotFoundError as e:  # no NPZ = zero cells of this unit in that well
            skipped["control_missing"] += len(sub)
            log.warning("skip %s | a control well has no cells: %s", unit, e)
            continue
        if min(x.shape[0] for x in real.values()) < a.min_cells:
            skipped["control_small"] += len(sub)
            log.warning("skip %s | a control well has < %d cells", unit, a.min_cells)
            continue
        ctrls = {"real": {w: to_t(x, device) for w, x in real.items()},
                 "shuffled": {w: to_t(x, device) for w, x in pseudo_wells(real, wells, a.seed + s).items()}}
        for c in sub.to_dict("records"):
            x = reader.perturbed(c)
            if x.shape[0] < a.min_cells:
                skipped["perturbed_small"] += 1
                continue
            pert = to_t(x, device)
            for arm, ctrl in ctrls.items():
                rows += [{**c, "arm": arm, "well_i": wi, "well_j": wj, "n_perturbed": int(x.shape[0]),
                          "n_i": int(ctrl[wi].shape[0]), "n_j": int(ctrl[wj].shape[0]), "deg": deg,
                          "pearson_delta": pr} for wi, wj, deg, pr in swap_pairs(pert, ctrl, wells)]
        log.info("unit %d %s | conditions=%d rows=%d | %.0fs", s + 1, unit, len(sub), len(rows), time.time() - t0)
    if not rows:
        raise SystemExit(f"nothing scored (skipped={skipped})")
    out["pairs"].parent.mkdir(parents=True, exist_ok=True)
    pairs = pd.DataFrame(rows)
    pairs.to_csv(out["pairs"], index=False)
    if a.n_shards > 1:
        out["skipped"].write_text(json.dumps(skipped))
        log.info("wrote %s | run --merge when all %d shards are done", out["pairs"], a.n_shards)
        return 0
    summary = {"min_cells": a.min_cells, "skipped": skipped, "deg_key": DEG_KEY, "pearson_key": PEARSON_KEY,
               **summarize(pairs, list(conds.columns), a.n_boot, a.seed)}
    out["summary"].write_text(json.dumps(summary, indent=2))
    log.info("wrote %s and %s", out["pairs"], out["summary"])
    return 0


def merge_control_swap(a: argparse.Namespace, conds: pd.DataFrame, per_cond: int) -> int:
    """Concatenate the shards; refuse unless every condition is scored or counted as skipped."""
    shard = [f"_shard{i}of{a.n_shards}" for i in range(a.n_shards)]
    paths = [Path(f"{a.out_prefix}_pairs{t}.csv") for t in shard]
    if missing := [p.name for p in paths if not p.is_file()]:
        raise SystemExit(f"missing shards: {missing}")
    pairs = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    skipped: dict[str, int] = {}
    for t in shard:
        for k, v in json.loads(Path(f"{a.out_prefix}_skipped{t}.json").read_text()).items():
            skipped[k] = skipped.get(k, 0) + v
    keys = list(conds.columns)
    if (n := pairs.groupby(keys).ngroups) + sum(skipped.values()) != len(conds):
        raise SystemExit(f"scored {n} + skipped {skipped} != {len(conds)} conditions")
    if (pairs.groupby([*keys, "arm"]).size() != per_cond).any():
        raise SystemExit(f"a condition does not have all {per_cond} well pairs per arm")
    pairs.to_csv(f"{a.out_prefix}_pairs.csv", index=False)
    summary = {"min_cells": a.min_cells, "n_conditions_total": len(conds), "skipped": skipped,
               "deg_key": DEG_KEY, "pearson_key": PEARSON_KEY, **summarize(pairs, keys, a.n_boot, a.seed)}
    Path(f"{a.out_prefix}_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("merged %d shards | %d conditions", a.n_shards, n)
    return 0


# 4. well_draws: equal-size control samples within one well vs between two wells

def cmd_well_draws(a: argparse.Namespace) -> int:
    """Within- vs between-well control samples of equal size s_a per (donor, cell type) stratum."""
    from src.eval.reference.pairs import score_raw_pair
    from src.eval.metrics.energy import energy_distance
    from src.eval.metrics.spaces import _project_raw_to_spaces

    summary_in = json.loads(a.conditions_from.read_text())
    if not (cols := summary_in.get("condition_variables")):
        raise SystemExit(f"{a.conditions_from.name}: no condition_variables recorded")
    conds = pd.DataFrame([r["condition_keys"] for r in summary_in["results"]])[list(cols)]
    if conds.duplicated().any():
        raise SystemExit(f"{a.conditions_from.name}: duplicate conditions")
    tag = f"_shard{a.shard}of{a.n_shards}" if a.n_shards > 1 else (f"_limit{a.limit}" if a.limit else "")
    names = ("rows", "energy", "strata")
    if a.merge:
        return merge_well_draws(a, conds, names)
    log.info("L_ref=%s", set_dataset_median_umi(a.dataset))
    reader, device = EvalNpzReader(namespace(a.eval_cache)), device_for(a.device)
    pearson = "log1p_slice_med_rgs/pearson_r_diff"                     # the depth-scaled log space (Methods)
    rows, energy, strata, t0 = [], [], [], time.time()
    for s, (unit, sub) in enumerate(conds.groupby(list(a.unit_cols), sort=True)):
        if s % a.n_shards != a.shard:
            continue
        unit = dict(zip(a.unit_cols, unit))
        real = {}
        for w in a.swap_values:
            try:
                real[w] = reader.control({**unit, a.swap_col: w})
            except FileNotFoundError:                                  # no NPZ = zero cells in that well
                real[w] = np.zeros((0, 1), dtype=np.float32)
        sizes = {w: int(x.shape[0]) for w, x in real.items()}
        kept = half_size(sizes, a.min_half)
        strata.append({**unit, **{f"n_{w}": n for w, n in sizes.items()}, "n_conditions": len(sub),
                       "wells": "|".join(kept[0]) if kept else "", "s_a": kept[1] if kept else 0})
        if kept is None:
            log.info("drop %s | wells %s: s_a < %d", unit, sizes, a.min_half)
            continue
        wells, s_a = kept
        seed = a.seed + zlib.crc32("|".join(str(unit[c]) for c in a.unit_cols).encode())  # per stratum, stable
        draws = well_draws(sizes, wells, s_a, a.n_draws, seed)
        ctrl = {w: to_t(real[w], device) for w in wells}
        pairs = [(arm, wi, wj, d, ctrl[wi][torch.as_tensor(ia, device=device)],
                  ctrl[wj][torch.as_tensor(ib, device=device)]) for arm, wi, wj, d, ia, ib in draws]
        for arm, wi, wj, d, ca, cb in pairs:                           # control vs control: no condition
            xa, xb = (_project_raw_to_spaces(c)["log1p_slice_med_rgs"] for c in (ca, cb))
            energy.append({**unit, "arm": arm, "well_i": wi, "well_j": wj, "draw": d, "s_a": s_a,
                           "energy": float(energy_distance(xa, xb, max_samples=None))})
        for c in sub.to_dict("records")[: a.limit or None]:
            x = reader.perturbed(c)
            if x.shape[0] < a.min_cells:
                continue
            pert = to_t(x, device)
            for arm, wi, wj, d, ca, cb in pairs:
                m = score_raw_pair(pert, pert, ca, cb, dist=False)
                rows.append({**c, "arm": arm, "well_i": wi, "well_j": wj, "draw": d, "s_a": s_a,
                             "n_perturbed": int(x.shape[0]), "deg": m[DEG_KEY], "pearson_delta": m[pearson]})
        log.info("stratum %d %s | s_a=%d wells=%d conditions=%d rows=%d | %.0fs", s + 1, unit, s_a, len(wells),
                 len(sub), len(rows), time.time() - t0)
    if not rows:
        raise SystemExit("nothing scored")
    for name, data in zip(names, (rows, energy, strata)):
        path = Path(f"{a.out_prefix}_{name}{tag}.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(data).to_csv(path, index=False)
    if a.n_shards > 1:
        log.info("wrote shard %d/%d | run --merge when all are done", a.shard, a.n_shards)
        return 0
    return summarize_well_draws(a, *(pd.DataFrame(d) for d in (rows, energy, strata)), conds)


def merge_well_draws(a: argparse.Namespace, conds: pd.DataFrame, names: tuple[str, ...]) -> int:
    """Concatenate the shards; refuse unless every stratum appears exactly once."""
    tabs = []
    for name in names:
        paths = [Path(f"{a.out_prefix}_{name}_shard{i}of{a.n_shards}.csv") for i in range(a.n_shards)]
        if missing := [p.name for p in paths if not p.is_file()]:
            raise SystemExit(f"missing shards: {missing}")
        tabs.append(pd.concat([pd.read_csv(p) for p in paths], ignore_index=True))
        tabs[-1].to_csv(f"{a.out_prefix}_{name}.csv", index=False)
    if (n := len(tabs[2])) != conds.groupby(list(a.unit_cols)).ngroups or tabs[2].duplicated(list(a.unit_cols)).any():
        raise SystemExit(f"strata table has {n} rows, expected each of {conds.groupby(list(a.unit_cols)).ngroups} once")
    return summarize_well_draws(a, *tabs, conds)


def summarize_well_draws(a: argparse.Namespace, rows: pd.DataFrame, energy: pd.DataFrame,
                         strata: pd.DataFrame, conds: pd.DataFrame) -> int:
    keys, unit = list(conds.columns), list(a.unit_cols)
    kept = strata[strata["s_a"] > 0]
    summary = {"min_half": a.min_half, "n_draws": a.n_draws, "min_cells": a.min_cells,
               "deg_key": DEG_KEY, "pearson_key": "log1p_slice_med_rgs/pearson_r_diff",
               "energy": "control sample vs control sample, log1p_slice_med_rgs, no subsampling",
               "strata": {"total": len(strata), "kept": len(kept), "five_wells": int((kept["wells"].str.count(r"\|") == 4).sum())},
               "conditions": {"total": len(conds), "scored": int(rows.groupby(keys).ngroups)},
               "s_a": {"min": int(kept["s_a"].min()), "median": float(kept["s_a"].median()), "max": int(kept["s_a"].max())},
               "cluster": unit,
               **within_between_tests(rows, keys, unit, a.n_boot, a.seed),
               "energy_per_stratum": within_between_tests(energy, unit, unit, a.n_boot, a.seed)["energy"]}
    Path(f"{a.out_prefix}_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("wrote %s_summary.json | %d strata, %d conditions", a.out_prefix, len(kept), summary["conditions"]["scored"])
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="kind", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", required=True)
    common.add_argument("--eval-cache", required=True, help="truth namespace; {data:<dataset>} templates allowed")

    add_split_half_parser(sub, common=common)
    add_replicate_pairs_parser(sub, common=common)

    bp = sub.add_parser("replicate_pairs_baselines", help="linear baselines vs the repeat (no --dataset: Tahoe-bio)")
    bp.add_argument("--device", default="cuda:0")
    bp.add_argument("--overwrite", action="store_true")
    bp.add_argument("--check-only", action="store_true")
    bp.add_argument("--results", type=Path, default=None,
                    help="reads model_vs_batch_gap__genesis.parquet here and writes beside it "
                         "(default: results/tahoe/plate_pairs/slice_med_seeded)")
    bp.add_argument("--report-space", choices=REPORT_SPACES, default=DEFAULT_REPORT_SPACE)
    bp.add_argument("--baseline-root", type=Path, required=True,
                    help="dir of <kind>.npz artifacts, fit on the split the GENESIS pairs were scored on")
    bp.add_argument("--check-pred-root", type=Path, default=None,
                    help="GENESIS per-plate npy behind the genesis parquet, for the energy self-check "
                         "(default: Fig 2a's bio run)")
    bp.add_argument("--energy-all-cells", action="store_true",
                    help="energy distance on every cell, not a 2,048-cell draw; the genesis parquet's must be too")

    ld = sub.add_parser("ladder", parents=[common], help="split-half vs the repeat at matched depth")
    ld.add_argument("--cohort", type=Path, required=True,
                    help="CSV: condition columns + <replicate>_i/_j (Figure 1's data/replication_ladder_equaln/cohort.csv)")
    ld.add_argument("--n-repeats", type=int, default=20)
    ld.add_argument("--seed", type=int, default=17)
    ld.add_argument("--limit", type=int, default=0, help="first N cohort rows (gates, smoke tests)")
    ld.add_argument("--device", default="cuda")
    ld.add_argument("--out", type=Path, required=True, help="directory")

    c = sub.add_parser("control_swap", parents=[common], help="fixed perturbed cells, two control wells")
    c.add_argument("--conditions-from", type=Path, required=True,
                   help="a metrics_summary.json whose conditions to score (e.g. the panel's GENESIS arm)")
    c.add_argument("--unit-cols", nargs="+", required=True, help="columns that own one set of control wells")
    c.add_argument("--swap-col", required=True, help="the control-well column, e.g. bc1_well")
    c.add_argument("--swap-values", nargs="+", required=True, help="the control wells, in order")
    c.add_argument("--min-cells", type=int, default=10, help=">= this many cells per pseudobulk (the DEG rule)")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--n-boot", type=int, default=10_000)
    c.add_argument("--limit", type=int, default=0, help="first N conditions (timing runs, gates)")
    c.add_argument("--shard", type=int, default=0)
    c.add_argument("--n-shards", type=int, default=1)
    c.add_argument("--merge", action="store_true", help="concatenate the shards and summarize")
    c.add_argument("--device", default="cuda")
    c.add_argument("--out-prefix", required=True, help="writes <prefix>_pairs.csv and <prefix>_summary.json")

    wd = sub.add_parser("well_draws", parents=[common], help="equal-size control samples within vs between wells")
    wd.add_argument("--conditions-from", type=Path, required=True, help="a metrics_summary.json whose conditions to score")
    wd.add_argument("--unit-cols", nargs="+", required=True, help="columns that own one set of control wells (a stratum)")
    wd.add_argument("--swap-col", required=True, help="the control-well column, e.g. bc1_well")
    wd.add_argument("--swap-values", nargs="+", required=True, help="the control wells, in order")
    wd.add_argument("--min-half", type=int, default=50, help="keep a stratum only if s_a reaches this")
    wd.add_argument("--n-draws", type=int, default=10, help="draws per well (within) and per well pair (between)")
    wd.add_argument("--min-cells", type=int, default=10, help=">= this many perturbed cells (the DEG rule)")
    wd.add_argument("--seed", type=int, default=0)
    wd.add_argument("--n-boot", type=int, default=10_000)
    wd.add_argument("--limit", type=int, default=0, help="first N conditions per stratum (timing runs)")
    wd.add_argument("--shard", type=int, default=0)
    wd.add_argument("--n-shards", type=int, default=1)
    wd.add_argument("--merge", action="store_true", help="concatenate the shards and test")
    wd.add_argument("--device", default="cuda")
    wd.add_argument("--out-prefix", required=True, help="writes <prefix>_{rows,energy,strata}.csv and _summary.json")

    a = p.parse_args()
    logging.basicConfig(level=a.log_level, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    cmds = {"split_half": cmd_split_half, "replicate_pairs": cmd_replicate_pairs, "ladder": cmd_ladder,
            "control_swap": cmd_control_swap, "well_draws": cmd_well_draws}
    cmds["replicate_pairs_baselines"] = cmd_replicate_pairs_baselines
    return cmds[a.kind](a)


if __name__ == "__main__":
    raise SystemExit(main())
