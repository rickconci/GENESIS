"""Figure 2's data rules: which plate pairs and conditions are plotted, read from a bundle's data/, and the
tests of each arm against the experimental repeat on exactly those pairs. Shared with its Extended Data
twins (ED02, ED14) and Figure 3a, which draw the same panels on other splits.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.figures.palette import ARMS
from src.stats.equivalence import NO_GAP_TESTS, mark_no_gap, repeat_tests

# Plate-pair parquet stem -> label. The repeat is read off GENESIS's file, as the
# published plotter does: energy subsamples cells, so files differ by ~1e-3.
PAIR_ARMS = {"genesis": "GENESIS", "state": "STATE", "perturbdiff": "PerturbDiff",
             "cellflow": "CellFlow", "perturbation_mean": "PertMean",
             "linear_ridge": "LinRidge", "additive": "Additive",
             "context_mean": "CtxMean", "copy_closest": "Nearest-neighbor"}
KEYS = ["cell_name", "drug", "dose", "plate_i", "plate_j"]
# (metric, title, higher is better). The plate-pair scorer and the split-half
# scorer name the same three metrics differently.
PAIR_METRICS = [("deg_adaptivek_truth", "Top-100 DEG overlap", True),
                ("pearson_r_diff", "Pearson Δ", True),
                ("energy_distance", "Energy distance", False)]
# Panel c reports in the same space as a and b (slice_med); DEG is always raw_int_rgs.
COND_SPACE = "log1p_slice_med_rgs"
COND_METRICS = [
    ("deg_raw_int_rgs/deg_overlap_top_p_ranked_k100_fdr0p001_lfc0p1_adaptivek_truth_detfloor10",
     "Top-100 DEG overlap", True),
    (f"{COND_SPACE}/pearson_r_diff", "Pearson Δ", True),
    (f"{COND_SPACE}/energy_distance", "Energy distance (log)", False),
]
COND_PANELS = [("tahoe_bio", "Tahoe-100M"), ("parse_bio", "PBMC cytokine atlas")]
COND_NAME = {"copy_closest": "Nearest-neighbor"}  # numbers.csv spells one arm differently
MATRIX_LINE, MATRIX_ARMS = "HepG2_C3A", ["truth", "genesis", "state", "perturbdiff", "cellflow"]
# The npz `params` name the space the matrices were built in; the colourbar says it in words.
# A file without a `space` key is log1p_rgs.
MATRIX_SPACE = {"log1p_rgs": "log1p", "log1p_slice_med_rgs": "log1p depth-scaled"}
MATRIX_CONTROL = "batch_matched_mean_subtracted"  # what "control-subtracted" names


# ----------------------------------------------------------------------------
# 1. Load the data (every loader takes the bundle's data/ directory)
# ----------------------------------------------------------------------------

def load_pairs(data: Path) -> pd.DataFrame:
    """One row per plate pair, finite for every arm on all three metrics.

    Equivalent to the published trio's rule (per-metric finite intersection, then
    the intersection across metrics), so a and b share one N.
    """
    frames = {}
    for stem, label in PAIR_ARMS.items():
        df = pd.read_parquet(data / "plate_pairs" / f"pair_metrics__{stem}.parquet").set_index(KEYS)
        if df.index.has_duplicates:
            raise ValueError(f"{stem}: duplicate plate-pair keys")
        frames[label] = df
    common = sorted(set.intersection(*(set(f.index) for f in frames.values())))
    # A cell line x compound trained at another concentration is not held out: its pairs
    # leave the set. The list is written by Figure 2's extract, with src/eval/reference/replicate.py's
    # drop_seen_in_training; a row matching no pair means a naming mismatch, so it raises.
    seen = pd.read_csv(data / "plate_pairs" / "seen_in_training.csv")
    seen = set(zip(seen.cell_name, seen.drug))
    # A plate with < 50 truth cells is too thin to score; listed by
    # replicate.py's drop_small_plates. Keys normalised: plates as str, dose rounded.
    key = lambda c, d, x, i, j: (c, d, round(float(x), 6), str(i), str(j))  # noqa: E731
    small = {key(*r) for r in pd.read_csv(data / "plate_pairs" / "too_few_cells.csv")[KEYS].itertuples(index=False)}
    if unmatched := sorted(seen - {k[:2] for k in common}) + sorted(small - {key(*k) for k in common}):
        raise ValueError(f"seen_in_training.csv / too_few_cells.csv rows match no shared pair: {unmatched}")
    n_shared = len(common)
    common = [k for k in common if k[:2] not in seen]
    n_seen = n_shared - len(common)
    common = [k for k in common if key(*k) not in small]
    wide = pd.DataFrame(index=pd.MultiIndex.from_tuples(common, names=KEYS))
    for m, _t, _hb in PAIR_METRICS:
        wide[f"Experimental repeat__{m}"] = frames["GENESIS"].loc[common, f"true_true__{m}"]
        for label, df in frames.items():
            wide[f"{label}__{m}"] = df.loc[common, f"cross_mean__{m}"]
    finite = np.isfinite(wide.to_numpy(dtype=float)).all(axis=1)
    # each direction alone, m(A, B') and m(B, A'): the averaged win rate votes on these
    for m, _t, _hb in PAIR_METRICS:
        for label, df in frames.items():
            for d in ("ij", "ji"):
                wide[f"{label}__{m}__{d}"] = df.loc[common, f"cross_{d}__{m}"]
    if not np.isfinite(wide[finite].to_numpy(dtype=float)).all():
        raise ValueError("a kept pair has a finite cross_mean but a non-finite direction")
    print(f"plate pairs: {n_shared} shared, {n_seen} seen in training, "
          f"{n_shared - n_seen - len(common)} too few cells, {int((~finite).sum())} non-finite, {int(finite.sum())} kept")
    return wide[finite].reset_index()


def load_conditions(data: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-condition scores (written by Figure 2's extract) and the
    split-half reference (mean and mean repeat s.d.) per panel and metric."""
    pc = pd.read_parquet(data / "model_comparison" / "per_condition.parquet")
    pc["model"] = pc["model"].replace(COND_NAME)
    unknown = set(pc["model"]) - set(ARMS)
    if unknown:
        raise KeyError(f"per_condition.parquet has arms with no style: {sorted(unknown)}")
    ceil = pd.read_csv(data / "model_comparison" / "ceilings.csv")
    ceil = ceil[ceil.report_space.eq(COND_SPACE)].set_index(["panel", "metric_key"])
    return pc, ceil


def load_cis(data: Path) -> pd.DataFrame:
    """The drawn 95% CIs (confidence_intervals.csv), keyed (panel, metric, arm); panel is 'a' or 'c_<dataset>'."""
    ci = pd.read_csv(data / "confidence_intervals.csv")
    return ci[ci.drawn].set_index(["panel", "metric", "arm"])


def load_matrices(data: Path) -> dict[str, np.ndarray]:
    """Each arm's condition x condition energy matrix on the conditions all arms share,
    plus the space they were built in, as words for the colourbar."""
    raw, spaces = {}, set()
    for arm in MATRIX_ARMS:
        with np.load(data / "energy_matrices" / f"{MATRIX_LINE}__{arm}.npz", allow_pickle=True) as z:
            raw[arm] = (z["matrix"].astype(np.float64), [str(c) for c in z["conditions"]])
            params = json.loads(str(z["params"])) if "params" in z.files else {}
        if params.get("control", MATRIX_CONTROL) != MATRIX_CONTROL:
            raise ValueError(f"{arm}: control {params['control']!r}, but the colourbar says control-subtracted")
        spaces.add(params.get("space", "log1p_rgs"))
    if len(spaces) != 1 or not spaces <= MATRIX_SPACE.keys():
        raise ValueError(f"energy matrices built in {sorted(spaces)}: need one space of {sorted(MATRIX_SPACE)}")
    # Arms can lose conditions independently (a missing prediction plate).
    common = set.intersection(*(set(c) for _m, c in raw.values()))
    keep = [c for c in raw["truth"][1] if c in common]
    out = {}
    for arm, (mat, conds) in raw.items():
        pos = {c: i for i, c in enumerate(conds)}
        sel = np.array([pos[c] for c in keep])
        out[arm] = mat[np.ix_(sel, sel)]
    out["conditions"], out["space"] = np.array(keep), MATRIX_SPACE[spaces.pop()]
    print(f"{MATRIX_LINE}: {len(keep)} conditions shared by {len(MATRIX_ARMS)} arms, space {out['space']!r}")
    return out


# ----------------------------------------------------------------------------
# 2. Test each arm: is it as close to the truth as the repeat is?
# ----------------------------------------------------------------------------

# Two tests, one question: does the arm differ from the repeat at all? Both have the null
# of NO gap, two-sided:
#   R = mean(arm) / mean(repeat), pair-weighted (exactly the means the bars draw); H0 R = 1
#   win rate against the repeat: the share of pair comparisons the arm wins (ties 1/2), the
#   two directions' votes averaged; H0 50%, replicate level (exact for a perfect generator,
#   where each single vote is a fair coin). The table stores its complement, P(arm worse),
#   as the older winrate_equivalence rows do.
# A bar gets a bold star when NEITHER test finds a gap (both P >= 0.05). That is "no difference
# detected", not "equivalent": only the CI can turn it into "close", and the Appendix draws
# it. No star means at least one test found a gap. Requiring both is the stricter rule (two
# chances to find a gap). The condition is the unit (3-plate conditions give 3 dependent
# pairs); P is the cluster-robust t-test's and the bootstrap verdict is checked against it.
# The table also keeps the non-inferiority margins (5%, 10%; one-sided, 90% CIs) for reference.
R_TEST = NO_GAP_TESTS[0]


def tests_vs_repeat(pairs: pd.DataFrame, metrics: list = PAIR_METRICS) -> pd.DataFrame:
    """Every arm x metric, on exactly the pairs the bars average (canonical draw). ``metrics``: a subset of
    PAIR_METRICS, for figures that test each metric on its own pair set (ED02)."""
    cluster = pairs[["cell_name", "drug", "dose"]].astype(str).agg("|".join, axis=1).to_numpy()
    rows = []
    for m, _t, hb in metrics:
        ref = pairs[f"Experimental repeat__{m}"].to_numpy()
        for lab in ARMS:
            ab, ba = (pairs[f"{lab}__{m}__{d}"].to_numpy() for d in ("ij", "ji"))
            rows += [dict(metric=m, arm=lab, **r)
                     for r in repeat_tests(ref, pairs[f"{lab}__{m}"].to_numpy(), ab, ba, cluster, hb)]
    t = pd.DataFrame(rows)
    t["mark"] = mark_no_gap(t)                                         # one mark, on the R row
    return t
