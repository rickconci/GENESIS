"""Figure 1d's numbers: how well two arms of one condition agree, by cells per arm.

reps.parquet holds one row per condition, strategy (split-half or experimental repeat), cells-per-arm rung
and draw. Panel d averages the draws within each condition first, so every condition weighs the same, then
pairs the two strategies so both curves run over the same conditions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# (reps.parquet column, y label, higher-is-better)
METRICS = [
    ("deg_raw_int_rgs/deg_overlap_top_p_ranked_k100_fdr0p001_lfc0p1_adaptivek_truth_detfloor10",
     "Top-100 DEG overlap", True),
    # depth-scaled space (each cell to a common panel library), as in Fig 2a-c
    ("log1p_slice_med_rgs/pearson_r_diff", "Pearson Δ", True),
    ("log1p_slice_med_rgs/energy_distance", "Energy distance", False),
]
COND_KEYS = ["cell_name", "drug", "dose"]
# Adaptive-k DEG overlap is undefined when the reference arm has no significant
# DEGs. Leaving those draws out averages only strong responders at shallow depth,
# which makes the curve fall from 50 to 100 cells. They score 0: nothing recovered.
DEG, N_SIG_REF = METRICS[0][0], "deg_raw_int_rgs/n_sig_degs_a_fdr0p001_lfc0p1_detfloor10"


def load(points: Path) -> pd.DataFrame:
    """One row per (condition, strategy, n): the 20 repeats averaged first, so every
    condition weighs the same in the mean over conditions."""
    df = pd.read_parquet(points)
    no_ref = df[N_SIG_REF].eq(0)
    if (other := df[DEG].isna() & ~no_ref).any():
        raise ValueError(f"{int(other.sum())} DEG draws undefined with reference DEGs present")
    df.loc[no_ref, DEG] = 0.0
    cols, keys = [m for m, _, _ in METRICS], COND_KEYS + ["strategy", "n_cells"]
    # mean() skips NaN: a metric undefined on some of a condition's draws would silently average fewer draws.
    # Undefined on all of them is an undefined arm, which paired() drops.
    na = df[cols].isna().groupby([df[k] for k in keys])
    if (partial := na.any() & ~na.all()).to_numpy().any():
        raise ValueError(f"{int(partial.to_numpy().sum())} (condition, metric) cells undefined on some draws only")
    return df.groupby(keys, as_index=False)[cols].mean()


def paired(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per (condition, n): both strategies side by side. Dropping a condition that
    lacks either strategy keeps the two curves over the same conditions."""
    w = df.pivot_table(index=COND_KEYS + ["n_cells"], columns="strategy",
                       values=metric).reset_index()
    return w.dropna(subset=["split_half", "cross_plate"])
