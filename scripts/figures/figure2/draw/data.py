"""Figure 2's numbers: which plate pairs and conditions are plotted, their 95% CIs, and the tests of each arm
against the experimental repeat on exactly those pairs. The perturbation-split Extended Data page draws the same
panels from its own files, so every loader takes its paths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.figures import metrics
from src.figures.palette import ARMS
from src.stats.equivalence import NO_GAP_TESTS, mark_no_gap, repeat_tests

# Plate-pair file stem -> arm. The repeat is read off GENESIS's file: energy subsamples cells, so files differ ~1e-3
PAIR_ARMS = {"genesis": "GENESIS", "state": "STATE", "perturbdiff": "PerturbDiff", "cellflow": "CellFlow",
             "perturbation_mean": "PertMean", "linear_ridge": "LinRidge", "additive": "Additive",
             "context_mean": "CtxMean", "copy_closest": "Nearest-neighbor"}
KEYS = ["cell_name", "drug", "dose", "plate_i", "plate_j"]
REPEAT = "Experimental repeat"
# (metric, title, higher is better): the plate-pair scorer's names (a, b), then the per-condition tables' (c)
PAIR_METRICS = [("deg_adaptivek_truth", "Top-100 DEG overlap", True), ("pearson_r_diff", "Pearson Δ", True),
                ("energy_distance", "Energy distance", False)]
COND_METRICS = [(metrics.DEG, "Top-100 DEG overlap", True), (metrics.PEARSON, "Pearson Δ", True),
                (metrics.ENERGY, "Energy distance (log)", False)]
R_TEST = NO_GAP_TESTS[0]                         # the row of a test table that carries an arm's star
MIN_CELLS = 50                                   # per plate: a thinner plate is too noisy to score (both splits)


def pair_key(c, d, x, i, j) -> tuple:
    """A plate pair, comparable across writers: plates as str, dose rounded."""
    return c, d, round(float(x), 6), str(i), str(j)


def read_arms(pair_dir: Path) -> dict[str, pd.DataFrame]:
    """Every arm's plate-pair scores, indexed by pair_key. The panels draw every arm, so each file must exist."""
    frames = {}
    for stem, arm in PAIR_ARMS.items():
        df = pd.read_parquet(pair_dir / f"pair_metrics__{stem}.parquet")
        keys = [pair_key(*k) for k in df[KEYS].itertuples(index=False)]
        if len(set(keys)) != len(keys):
            raise ValueError(f"{stem}: duplicate plate-pair keys")
        frames[arm] = df.set_index(pd.MultiIndex.from_tuples(keys, names=KEYS))
    return frames


def wide(frames: dict[str, pd.DataFrame], keep: list[tuple], m: str) -> pd.DataFrame:
    """One metric on ``keep``: the repeat, each arm's mean over both directions, and each direction alone
    (m(A, B') and m(B, A'), on which the win rate votes)."""
    cols = {f"{REPEAT}__{m}": frames["GENESIS"].loc[keep, f"true_true__{m}"]}
    for arm, df in frames.items():
        cols[f"{arm}__{m}"] = df.loc[keep, f"cross_mean__{m}"]
        for d in ("ij", "ji"):
            cols[f"{arm}__{m}__{d}"] = df.loc[keep, f"cross_{d}__{m}"]
    return pd.DataFrame(cols)


# ----------------------------------------------------------------------------
# Read the plate pairs: Figure 2 (one set for all metrics) and the perturbation split (one set per metric)
# ----------------------------------------------------------------------------

def pairs(pair_dir: Path) -> pd.DataFrame:
    """Figure 2a-b's pairs: shared by every arm, finite on all three metrics, minus a cell line x compound
    trained at another concentration (not held out) and plates under MIN_CELLS. Both lists are the scorer's;
    a row matching no pair is a naming mismatch, so it raises."""
    frames = read_arms(pair_dir)
    common = sorted(set.intersection(*(set(f.index) for f in frames.values())))
    seen = pd.read_csv(pair_dir / "seen_in_training.csv")
    seen = set(zip(seen.cell_name, seen.drug))
    small = {pair_key(*r) for r in pd.read_csv(pair_dir / "too_few_cells.csv")[KEYS].itertuples(index=False)}
    if unmatched := sorted(seen - {k[:2] for k in common}) + sorted(small - set(common)):
        raise ValueError(f"seen_in_training.csv / too_few_cells.csv rows match no shared pair: {unmatched}")
    held_out = [k for k in common if k[:2] not in seen]
    keep = [k for k in held_out if k not in small]
    t = pd.concat([wide(frames, keep, m) for m, _t, _hb in PAIR_METRICS], axis=1)
    means = [f"{arm}__{m}" for m, _t, _hb in PAIR_METRICS for arm in [REPEAT, *frames]]
    finite = np.isfinite(t[means].to_numpy(float)).all(axis=1)
    if not np.isfinite(t[finite].to_numpy(float)).all():
        raise ValueError("a kept pair has a finite cross_mean but a non-finite direction")
    print(f"plate pairs: {len(common)} shared, {len(common) - len(held_out)} seen in training, "
          f"{len(held_out) - len(keep)} too few cells, {int((~finite).sum())} non-finite, {int(finite.sum())} kept")
    return t[finite].reset_index()


def pert_pairs(pair_dir: Path, bench_csv: Path) -> pd.DataFrame:
    """The perturbation split's pairs: the benchmark's, asserted rather than filtered (every benchmark pair scored
    with >= MIN_CELLS per plate, every extra pair under it). A pair counts for a metric when the repeat and every
    arm define it; elsewhere that metric is blank, so each metric's bars and curves share one set."""
    bench = {pair_key(*k) for k in pd.read_csv(bench_csv, dtype={"plate_i": str, "plate_j": str})[KEYS]
             .itertuples(index=False)}
    frames = read_arms(pair_dir)
    for arm, df in frames.items():
        if set(df.report_space) != {metrics.SPACE}:
            raise ValueError(f"{arm}: report space {set(df.report_space)}, Figure 2a uses {metrics.SPACE}")
        small = df[["n_truth_i", "n_truth_j"]].min(axis=1).lt(MIN_CELLS).to_numpy()
        missing, extra, thin = bench - set(df.index), set(df.index[~small]) - bench, set(df.index[small]) & bench
        if missing or extra or thin:
            raise ValueError(f"{arm}: {len(missing)} benchmark pairs missing, {len(extra)} extra with >= {MIN_CELLS} "
                             f"cells, {len(thin)} benchmark pairs under {MIN_CELLS}")
    keep = sorted(bench)
    for arm, df in frames.items():              # the repeat is truth only: every writer must agree on it
        for m, _t, _hb in PAIR_METRICS:
            if not np.allclose(frames["GENESIS"].loc[keep, f"true_true__{m}"], df.loc[keep, f"true_true__{m}"],
                               atol=1e-9, equal_nan=True):
                raise ValueError(f"{arm}: its experimental repeat ({m}) differs from GENESIS's file")
    blocks, n = [], {}
    for m, _t, _hb in PAIR_METRICS:
        block = wide(frames, keep, m)
        ok = np.isfinite(block.to_numpy(float)).all(axis=1)
        blocks.append(block.where(pd.Series(ok, index=block.index), np.nan, axis=0))
        n[m] = int(ok.sum())
    print(f"plate pairs: {len(keep)} in the benchmark; per metric {n}")
    return pd.concat(blocks, axis=1).reset_index()


def metric_pairs(pairs: pd.DataFrame, m: str) -> pd.DataFrame:
    """The pairs metric ``m`` keeps (all of Figure 2's; the perturbation split blanks some), renumbered."""
    return pairs[pairs[f"{REPEAT}__{m}"].notna()].reset_index(drop=True)


def tests_vs_repeat(pairs: pd.DataFrame) -> pd.DataFrame:
    """Every arm x metric on exactly the pairs its bar averages: does it differ from the repeat at all?
    Both tests have the null of NO gap, two-sided, clustered on the condition (3-plate conditions give 3
    dependent pairs):
      R = mean(arm) / mean(repeat); H0 R = 1
      win rate against the repeat, the two directions' votes averaged; H0 50% (replicate level)
    An arm is marked (a star) when NEITHER finds a gap: no difference detected, not equivalence."""
    rows = []
    for m, _t, hb in PAIR_METRICS:
        p = metric_pairs(pairs, m)
        cluster = p[["cell_name", "drug", "dose"]].astype(str).agg("|".join, axis=1).to_numpy()
        ref = p[f"{REPEAT}__{m}"].to_numpy()
        for arm in ARMS:
            ab, ba = (p[f"{arm}__{m}__{d}"].to_numpy() for d in ("ij", "ji"))
            rows += [dict(metric=m, arm=arm, **r) for r in repeat_tests(ref, p[f"{arm}__{m}"].to_numpy(), ab, ba,
                                                                        cluster, hb)]
    t = pd.DataFrame(rows)
    t["mark"] = mark_no_gap(t)
    return t


# ----------------------------------------------------------------------------
# Then held-out conditions (c), and the CIs every bar carries
# ----------------------------------------------------------------------------

def conditions(per_condition: list[Path], ceilings: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-condition scores, and the split-half reference (mean, mean repeat s.d.) per panel and metric."""
    pc = pd.concat([pd.read_parquet(f) for f in per_condition], ignore_index=True)
    pc["model"] = pc.model.replace(metrics.ARM_NAME)
    if unknown := set(pc.model) - set(ARMS):
        raise KeyError(f"arms with no style in Figure 2: {sorted(unknown)}")
    ceil = pd.concat([pd.read_csv(f) for f in ceilings], ignore_index=True)
    return pc, ceil[ceil.report_space.eq(metrics.SPACE)].set_index(["panel", "metric_key"])


def cis(path: Path) -> pd.DataFrame:
    """The drawn 95% CIs, keyed (panel, metric, arm); panel is 'a' or 'c_<dataset>'. The panels refuse a CI
    whose mean is not the plotted one (src.figures.bars.ci_row)."""
    ci = pd.read_csv(path)
    drawn = ci[ci.drawn.astype(bool)]
    if drawn.duplicated(["panel", "metric", "arm"]).any():
        raise ValueError(f"{path.name}: two drawn CIs for one bar")
    return drawn.set_index(["panel", "metric", "arm"])
