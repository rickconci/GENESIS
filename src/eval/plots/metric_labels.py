"""Metric naming, sorting, and protocol label helpers."""

from __future__ import annotations

from collections.abc import Mapping

from .constants import LOWER_IS_BETTER_METRICS, NORM_DISPLAY_LABELS


def norm_display_label(normalization_id: str) -> str:
    """Public accessor for :data:`NORM_DISPLAY_LABELS` with fallback."""
    return NORM_DISPLAY_LABELS.get(normalization_id, normalization_id)


def is_lower_better(metric: str) -> bool:
    """Return True if lower metric values indicate better performance."""
    base = metric.rsplit("/", 1)[-1]
    for prefix in LOWER_IS_BETTER_METRICS:
        if base.startswith(prefix) or base == prefix:
            return True
    return False


def pretty_metric_name(metric: str) -> str:
    """Human-readable metric name for plot titles."""
    mapping = {
        "pearson_r": "Pearson r",
        "pearson_r_diff": "Pearson r (delta)",
        "spearman_r_diff": "Spearman rho (delta)",
        "gene_det_rate_spearman": "Gene det-rate Spearman",
        "mse": "MSE",
        "mae": "MAE",
        "r2": "R\u00b2",
        "r2_diff": "R\u00b2 (delta)",
        "mmd_rbf": "MMD (RBF)",
        "mmd_linear": "MMD (Linear)",
    }
    prefix = ""
    tail = metric
    if "/" in metric:
        prefix, tail = metric.split("/", 1)
    if tail in mapping:
        pretty_tail = mapping[tail]
    else:
        pretty_tail = tail.replace("_", " ").title()
    return f"{prefix}/{pretty_tail}" if prefix else pretty_tail


_DELTA_METRIC_NAMES = frozenset({
    "r2_diff",
    "pearson_r_diff",
    "spearman_r_diff",
    "delta_l2_sq_norm_ratio",
})

_DEFAULT_DEG_SUFFIX = "fdr0p001_lfc0p1_detfloor10"


def metric_eval_protocol_label(metric: str, *, strategy: str) -> str:
    """One-line subtitle: full-cell vs split-half scoring for this metric key."""
    space, _, tail = metric.partition("/")
    split_half_metric = (
        strategy == "split_half"
        and (space.startswith("deg_") or tail in _DELTA_METRIC_NAMES)
    )
    if split_half_metric:
        return (
            "Cell protocol: split-half "
            "(half-sized truth/pred arms; disjoint control halves)"
        )
    return "Cell protocol: full cells (full truth vs full prediction per rep)"


def default_model_vs_ceiling_metric_keys() -> list[str]:
    """Curated cross-plot keys: slice_med (arm/dist), raw_int_rgs (delta), deg_raw_int_rgs."""
    arm = ("pearson_r", "mse", "mae", "r2", "mmd_rbf", "energy_distance")
    delta = tuple(sorted(_DELTA_METRIC_NAMES))
    deg = (
        f"jaccard_sig_{_DEFAULT_DEG_SUFFIX}",
        f"spearman_lfc_sig_{_DEFAULT_DEG_SUFFIX}",
        f"pearson_lfc_de_intersection_{_DEFAULT_DEG_SUFFIX}",
        f"roc_auc_{_DEFAULT_DEG_SUFFIX}",
        f"pr_auc_dir_match_{_DEFAULT_DEG_SUFFIX}",
    )
    return (
        [f"log1p_slice_med_rgs/{m}" for m in arm]
        + [f"raw_int_rgs/{m}" for m in delta]
        + [f"deg_raw_int_rgs/{m}" for m in deg]
    )


def is_default_cross_plot_metric(metric: str) -> bool:
    """Default cross-plot filter: no pf_log_pf; raw_int_rgs for delta/DEG; slice_med else."""
    if "pf_log_pf" in metric:
        return False
    space, _, tail = metric.partition("/")
    if not space:
        return False
    if space == "deg_raw_int_rgs":
        return True
    if space == "raw_int_rgs":
        return tail in _DELTA_METRIC_NAMES
    if space == "log1p_slice_med_rgs":
        return tail not in _DELTA_METRIC_NAMES
    return False


def apply_default_cross_plot_metric_filter(metrics: list[str]) -> list[str]:
    """Keep only default cross-plot metrics, preserving sort order."""
    return sorted(
        (m for m in metrics if is_default_cross_plot_metric(m)),
        key=_metric_plot_sort_key,
    )


def _metric_plot_sort_key(metric: str) -> tuple[int, str]:
    """Stable plot order: DEG → pseudobulk arm → delta → distributional."""
    tail = metric.rsplit("/", 1)[-1]
    if metric.startswith("deg_"):
        group = 0
    elif tail in _DELTA_METRIC_NAMES:
        group = 2
    elif tail in ("energy_distance", "mmd_rbf", "mmd_linear"):
        group = 3
    else:
        group = 1
    return (group, metric)


def overlap_plot_metric_keys(
    nc_mean: Mapping[str, Mapping[str, float]],
    model_data: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    apply_default_filter: bool = True,
) -> list[str]:
    """Metric keys shared by ceiling and at least one model summary."""
    all_nc = {m for mm in nc_mean.values() for m in mm}
    all_model = {m for md in model_data.values() for mm in md.values() for m in mm}
    shared = sorted(all_nc & all_model, key=_metric_plot_sort_key)
    if apply_default_filter:
        return apply_default_cross_plot_metric_filter(shared)
    return shared


def overlap_model_metric_keys(
    model_data: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    apply_default_filter: bool = True,
) -> list[str]:
    """Metric keys present in every model summary (model-only cross plots)."""
    per_model = [
        {m for mm in md.values() for m in mm}
        for md in model_data.values()
    ]
    if not per_model:
        return []
    shared = sorted(set.intersection(*per_model), key=_metric_plot_sort_key)
    if apply_default_filter:
        return apply_default_cross_plot_metric_filter(shared)
    return shared
