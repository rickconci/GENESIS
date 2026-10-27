"""Shared plotting constants."""

from __future__ import annotations

import seaborn as sns

sns.set_style("whitegrid")

LOWER_IS_BETTER_METRICS = frozenset({
    "mse", "mae", "mmd_rbf", "mmd_linear", "energy_distance",
})

RAW_NORM_IDS: frozenset[str] = frozenset({"raw", "raw_counts", "raw_softplus"})
LOG1P_NORM_IDS: frozenset[str] = frozenset({
    "log1p", "log1p_softplus", "log1p_relu", "log1p_norm", "log1p_cpm",
})

NORM_DISPLAY_LABELS: dict[str, str] = {
    "raw": "Raw Counts",
    "raw_int": "Raw int (pseudobulk)",
    "raw_int_rgs": "Raw int (rgs)",
    "log1p": "log1p",
    "log1p_rgs": "log1p (rgs)",
    "log1p_norm": "log1p (median-norm)",
    "log1p_cpm": "log1p CPM",
    "log1p_slice_med_rgs": "log1p slice-med (rgs)",
    "raw_counts": "Raw Counts",
    "raw_softplus": "Raw Counts",
    "log1p_softplus": "log1p",
    "log1p_relu": "log1p",
}

METRICS_SUMMARY_FILENAME = "metrics_summary.json"
BOOTSTRAP_CEILING_FILENAME = "bootstrap_noise_ceiling_summary.json"

GENE_SUMMARY_SPACES: tuple[str, ...] = (
    "raw_int_rgs", "log1p_rgs", "log1p_slice_med_rgs",
)

PLOT_TYPES: tuple[str, ...] = (
    "scatter_mean_var",
    "gene_var_hist",
)

TRUE_COLOR = "#4C72B0"
PRED_COLOR = "#E24A33"
ACCENT_COLOR = "#55A868"

_MODEL_LEGEND_LABEL_BY_RUN_DIR_BASENAME: dict[str, str] = {
    "tahoe_eval_state_v2": "STATE",
}

_MODEL_LEGEND_LABEL_BY_EVAL_DIR_PREFIX: dict[str, str] = {
    "from_scratch_tahoe100m": "PerturbDiff",
    "STATE_ST_Tahoe_preprint": "STATE",
}

_PERTGEN_EVAL_DIR_PREFIXES: tuple[str, ...] = (
    "20260331_003903_abl_relu",
)

# Minimum visible characters for model names in cross-condition plot legends.
CROSS_PLOT_MODEL_LABEL_MIN_CHARS = 25
