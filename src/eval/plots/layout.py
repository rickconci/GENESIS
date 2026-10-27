"""Figure layout helpers for cross-condition plots."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

# Per-metric y bands (always within [0, 1]); checked before unit-interval defaults.
_CROSS_CONDITION_NARROW_YLIM: dict[str, tuple[float, float]] = {
    "r2": (0.75, 1.0),
    "pearson_r": (0.75, 1.0),
    # pearson_r_diff / spearman_r_diff: full [0, 1] — models often sit well below 0.75
    "r2_cov": (0.5, 1.0),
    "r2_diff": (0.5, 1.0),
    "r2_var": (0.8, 1.0),
    "mae": (0.0, 0.1),
    "mse": (0.0, 0.05),
    "mmd_rbf": (0.0, 0.05),
    "mmd_linear": (0.0, 0.05),
}

# DEG / overlap metrics scored on [0, 1] (prefix match on metric tail).
_UNIT_INTERVAL_METRIC_PREFIXES: tuple[str, ...] = (
    "jaccard_sig_",
    "roc_auc_",
    "pr_auc_",
    "deg_overlap_top_p_ranked_k",
)

# Count metrics can exceed 1; only pin ymin=0 and let matplotlib autoscale ymax.
_COUNT_METRIC_PREFIXES: tuple[str, ...] = (
    "n_sig_degs_",
    "n_reproducible_degs",
)

_DEFAULT_CROSS_CONDITION_YLIM: tuple[float, float] = (0.0, 1.0)


def _plot_bottom_margin(
    *,
    n_legend: int,
    n_conditions: int,
    annotate_xticks: bool,
    max_xtick_labels: int,
) -> float:
    """Figure fraction reserved below axes: ticks → xlabel → gap → legend."""
    tick_frac = 0.17 if annotate_xticks and n_conditions <= max_xtick_labels else 0.05
    xlabel_frac = 0.030
    legend_frac = 0.028 * max(n_legend, 1) + 0.022
    gap = 0.012
    return min(tick_frac + xlabel_frac + gap + legend_frac + 0.02, 0.46)


def _cross_condition_figsize(
    n_conditions: int,
    figsize: tuple[float, float],
) -> tuple[float, float]:
    """Tall figure with modest width growth per condition."""
    width = min(14.0, max(7.5, 0.55 * n_conditions + 2.8))
    height = max(figsize[1], 9.0)
    return (width, height)


def _rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean with proper edge handling (no zero-padding artifacts)."""
    n = len(y)
    out = np.empty(n, dtype=np.float64)
    cs = np.concatenate([[0.0], np.cumsum(y.astype(np.float64))])
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = (cs[hi] - cs[lo]) / (hi - lo)
    return out


def _apply_cross_condition_metric_ylim(ax: Axes, metric: str) -> None:
    """Y-axis limits for model-vs-ceiling and multi-strategy cross-condition plots.

    Outer bound is always ``[0, 1]`` unless the metric is a raw count (``n_sig_degs_*``,
    ``n_reproducible_degs*``).  Many metrics use narrower fixed bands tuned for the
    usual high-signal range (see ``_CROSS_CONDITION_NARROW_YLIM``).
    """
    base = metric.rsplit("/", 1)[-1]

    if base.startswith(_COUNT_METRIC_PREFIXES):
        ax.set_ylim(bottom=0.0)
        return

    narrow = _CROSS_CONDITION_NARROW_YLIM.get(base)
    if narrow is not None:
        ax.set_ylim(*narrow)
        return

    if base.startswith(_UNIT_INTERVAL_METRIC_PREFIXES):
        ax.set_ylim(*_DEFAULT_CROSS_CONDITION_YLIM)
        return

    ax.set_ylim(*_DEFAULT_CROSS_CONDITION_YLIM)
