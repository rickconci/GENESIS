"""Synthesized per-condition boxplot drawing helpers."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

from .constants import CROSS_PLOT_MODEL_LABEL_MIN_CHARS
from .layout import (
    _apply_cross_condition_metric_ylim,
    _cross_condition_figsize,
    _plot_bottom_margin,
    _rolling_mean,
)
from .metric_labels import is_lower_better, pretty_metric_name

logger = logging.getLogger(__name__)

# Above this many x positions, per-condition boxes are sub-pixel — draw trend lines only.
CROSS_PLOT_BOX_MAX_CONDITIONS = 256


def use_per_condition_boxes(n_conditions: int) -> bool:
    """Whether to draw synthesized K-rep boxes at each condition index."""
    return int(n_conditions) <= CROSS_PLOT_BOX_MAX_CONDITIONS


def _finite_metric_values_for_conditions(
    metrics_by_cond: Mapping[str, Mapping[str, float]],
    metric: str,
    cond_keys: Sequence[str] | None = None,
) -> np.ndarray:
    """Collect finite metric values across condition keys."""
    keys = cond_keys if cond_keys is not None else metrics_by_cond.keys()
    out: list[float] = []
    for ck in keys:
        block = metrics_by_cond.get(ck)
        if not isinstance(block, Mapping):
            continue
        if metric not in block:
            continue
        v = float(block[metric])
        if math.isfinite(v):
            out.append(v)
    return np.asarray(out, dtype=np.float64)


def _comparison_boxplot_rows(
    *,
    metric: str,
    nc_mean: dict[str, dict[str, float]],
    model_data: dict[str, dict[str, dict[str, float]]],
    model_order: Sequence[str],
    model_id_labels: dict[str, str],
    cond_keys: Sequence[str],
    ceiling_label: str = "Noise Ceiling",
) -> list[dict[str, Any]]:
    """Build ranked boxplot rows (worst -> best median) for ceiling + models."""
    lower_better = is_lower_better(metric)
    rows: list[dict[str, Any]] = []
    nc_vals = _finite_metric_values_for_conditions(nc_mean, metric, cond_keys)
    if nc_vals.size:
        rows.append({
            "label": ceiling_label,
            "values": nc_vals,
            "median": float(np.median(nc_vals)),
            "kind": "ceiling",
        })
    for mid in model_order:
        vals = _finite_metric_values_for_conditions(
            model_data.get(mid, {}), metric, cond_keys,
        )
        if vals.size == 0:
            continue
        rows.append({
            "label": model_id_labels.get(mid, mid),
            "values": vals,
            "median": float(np.median(vals)),
            "kind": "model",
            "model_id": mid,
        })
    rows.sort(key=lambda r: float(r["median"]), reverse=lower_better)
    return rows


def _draw_comparison_boxplot_on_ax(
    ax,
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    facecolors: Sequence[str],
) -> None:
    """Draw one ranked boxplot panel (shared by model-vs-ceiling and multi-strategy)."""
    if not rows:
        return
    data = [np.asarray(r["values"], dtype=np.float64) for r in rows]
    labels = [str(r["label"]) for r in rows]
    bp = ax.boxplot(
        data,
        labels=labels,
        showfliers=True,
        patch_artist=True,
    )
    for patch, color in zip(bp["boxes"], facecolors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    vertical_labels = ["\n".join(list(lab)) for lab in labels]
    ax.set_xticklabels(vertical_labels, rotation=0, fontsize=8)
    _apply_cross_condition_metric_ylim(ax, metric)
    ax.grid(True, axis="y", alpha=0.3)


def _truncate_middle_label(text: str, max_len: int = 12) -> str:
    """Truncate long labels to ``max_len`` chars with ``...`` in the middle."""
    t = str(text)
    if len(t) <= max_len:
        return t
    inner = max_len - 3
    head = inner // 2
    tail = inner - head
    return f"{t[:head]}...{t[-tail:]}"


def format_cross_plot_model_label(
    label: str,
    summary_path: Path,
    *,
    min_chars: int = CROSS_PLOT_MODEL_LABEL_MIN_CHARS,
    explicit: bool = False,
) -> str:
    """Build a model legend label showing at least ``min_chars`` when possible.

    Short aliases (e.g. from ``--labels``) are replaced by the run directory
    name when that name is longer.  Names longer than ``min_chars`` are
    middle-truncated to ``min_chars``.

    When ``explicit=True`` (``--labels`` from compare-jsons), the label is
    returned as-is — e.g. ``run_name_suffix`` like ``mlp_short_seed3``.
    """
    primary = str(label).strip()
    if explicit:
        return primary
    run_name = summary_path.parent.name
    candidate = primary if len(primary) >= min_chars else (
        run_name if len(run_name) >= len(primary) else primary
    )
    if len(candidate) < min_chars and len(run_name) > len(candidate):
        candidate = run_name
    return _truncate_middle_label(candidate, max_len=min_chars)


def _synth_bxp_stat(mean: float, std: float) -> dict[str, Any]:
    """Synthesize one matplotlib ``bxp`` stat dict from K-rep summary.

    Per-rep raw values are not persisted to JSON (only ``mean`` and ``std``
    over K reps), so each box is drawn under a Gaussian-style convention:
    median tick at ``mean``, IQR ≈ ``[mean - std, mean + std]`` (~ ±1σ),
    whiskers at ``[mean - 2σ, mean + 2σ]``.  This makes the box a faithful
    visual of ``mean ± std`` rather than a true sample distribution.
    """
    m = float(mean)
    s = float(std) if math.isfinite(float(std)) else 0.0
    return {
        "med": m,
        "mean": m,
        "q1": m - s,
        "q3": m + s,
        "whislo": m - 2.0 * s,
        "whishi": m + 2.0 * s,
        "fliers": [],
    }


def _condition_short_label(ck: str) -> str:
    """Compact x-axis label: drop ``key=`` prefixes, join with ``|``."""
    parts = [p.split("=", 1)[-1] for p in ck.split("__")]
    return " | ".join(parts) if parts else ck


def _series_per_condition_stats(
    *,
    mean_by_cond: Mapping[str, Mapping[str, float]],
    std_by_cond: Mapping[str, Mapping[str, float]],
    metric: str,
    cond_order: Sequence[str],
) -> tuple[list[int], list[dict[str, Any]]]:
    """Pure: extract (condition_index, bxp stat dict) for finite mean values."""
    idx: list[int] = []
    stats: list[dict[str, Any]] = []
    for i, ck in enumerate(cond_order):
        m_block = mean_by_cond.get(ck) or {}
        if metric not in m_block:
            continue
        mu = float(m_block[metric])
        if not math.isfinite(mu):
            continue
        s_block = std_by_cond.get(ck) or {}
        sd = float(s_block.get(metric, 0.0))
        idx.append(i)
        stats.append(_synth_bxp_stat(mu, sd))
    return idx, stats


def _draw_series_boxes_on_ax(
    ax,
    *,
    idx: Sequence[int],
    stats: Sequence[Mapping[str, Any]],
    width: float,
    color: str,
    box_alpha: float,
    zorder: int = 2,
) -> None:
    """Draw one series of synthesized boxes centered on each condition index."""
    if not stats:
        return
    positions = [float(i) for i in idx]
    bp = ax.bxp(
        list(stats),
        positions=positions,
        widths=float(width),
        showfliers=False,
        patch_artist=True,
        manage_ticks=False,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(box_alpha)
        patch.set_zorder(zorder)
    for line in bp["medians"]:
        line.set_color(color)
        line.set_linewidth(1.2)
        line.set_alpha(min(1.0, box_alpha + 0.12))
        line.set_zorder(zorder + 1)
    for key in ("whiskers", "caps"):
        for line in bp[key]:
            line.set_color(color)
            line.set_alpha(min(1.0, box_alpha + 0.12))
            line.set_linewidth(0.8)
            line.set_zorder(zorder)


def _rolling_smooth_window(n: int) -> int:
    """Adaptive neighbourhood size for cross-condition rolling mean."""
    return max(3, n // 40)


def _draw_series_smooth_line_on_ax(
    ax,
    *,
    idx: Sequence[int],
    stats: Sequence[Mapping[str, Any]],
    color: str,
    zorder: int = 10,
    linestyle: str = "-",
    linewidth: float = 2.0,
) -> None:
    """Overlay rolling-mean trajectory through per-condition K-rep means.

    Uses :func:`layout._rolling_mean` with edge-aware windows: each x position
    averages neighbouring condition means rather than zero-padding at the boundaries.
    """
    if len(idx) < 2:
        return
    pairs = sorted(zip(idx, stats), key=lambda t: t[0])
    sx = np.asarray([p[0] for p in pairs], dtype=np.float64)
    sy = np.asarray([float(p[1]["mean"]) for p in pairs], dtype=np.float64)
    win = _rolling_smooth_window(len(sy))
    smoothed = _rolling_mean(sy, win)
    ax.plot(
        sx,
        smoothed,
        color=color,
        linewidth=float(linewidth),
        linestyle=linestyle,
        alpha=1.0,
        zorder=zorder,
        solid_capstyle="round",
    )


def _plot_single_metric(
    *,
    metric: str,
    nc_mean: dict[str, dict[str, float]],
    nc_std: dict[str, dict[str, float]],
    model_data: dict[str, dict[str, dict[str, float]]],
    model_stds: dict[str, dict[str, dict[str, float]]],
    model_order: Sequence[str],
    model_id_labels: dict[str, str],
    out_dir: Path,
    figsize: tuple[float, float],
    dpi: int,
    ceiling_color: str,
    model_colors: list[str],
    annotate_xticks: bool,
    max_xtick_labels: int,
    skip_existing: bool = False,
    protocol_subtitle: str | None = None,
    model_linestyles: Mapping[str, str] | None = None,
    legend_ncol: int | None = None,
    truncate_legend_labels: bool = True,
) -> Optional[Path]:
    """Per-condition synthesized boxplots: ceiling + each model, ranked by ceiling.

    X axis = conditions sorted by the noise-ceiling mean (worst → best,
    left → right).  For every condition, ceiling and model boxes are drawn
    at the **same** x position (overlaid) using the stored K-rep ``(mean, std)``.
    A rolling-mean line is drawn on top of each series to show the local
    trajectory across conditions.
    """
    from matplotlib.lines import Line2D

    out_dir = Path(out_dir)
    safe_metric = metric.replace("/", "-")
    out_path = out_dir / f"model_vs_ceiling__{safe_metric}.png"
    if skip_existing and out_path.is_file():
        logger.info("Skipping existing cross-condition plot: %s", out_path)
        return out_path

    lower_better = is_lower_better(metric)
    cond_keys = [
        ck for ck, md in nc_mean.items()
        if metric in md and math.isfinite(float(md[metric]))
    ]
    if not cond_keys:
        pool: dict[str, list[float]] = {}
        for mid in model_order:
            for ck, md in model_data.get(mid, {}).items():
                if metric not in md:
                    continue
                v = float(md[metric])
                if math.isfinite(v):
                    pool.setdefault(ck, []).append(v)
        cond_keys = list(pool.keys())
        if not cond_keys:
            logger.warning("Metric %s not found in any model series.", metric)
            return None
        cond_order = sorted(
            cond_keys,
            key=lambda ck: float(np.nanmean(pool[ck])),
            reverse=lower_better,
        )
    else:
        cond_order = sorted(
            cond_keys,
            key=lambda ck: float(nc_mean[ck][metric]),
            reverse=lower_better,
        )

    series: list[dict[str, Any]] = [{
        "label": "Noise Ceiling",
        "color": ceiling_color,
        "linestyle": "-",
        "mean_by_cond": nc_mean,
        "std_by_cond": nc_std,
    }]
    model_color_idx = 0
    ls_map = model_linestyles or {}
    for mid in model_order:
        if mid not in model_data:
            continue
        series.append({
            "label": model_id_labels.get(mid, mid),
            "color": model_colors[model_color_idx % len(model_colors)],
            "linestyle": str(ls_map.get(mid, "-")),
            "mean_by_cond": model_data[mid],
            "std_by_cond": model_stds.get(mid, {}),
        })
        model_color_idx += 1

    rendered = 0
    box_width = 0.65
    draw_boxes = use_per_condition_boxes(len(cond_order))
    fig_w, fig_h = _cross_condition_figsize(len(cond_order), figsize)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    for s_idx, s in enumerate(series):
        idx, stats = _series_per_condition_stats(
            mean_by_cond=s["mean_by_cond"],
            std_by_cond=s["std_by_cond"],
            metric=metric,
            cond_order=cond_order,
        )
        if not stats:
            continue
        is_ceiling = s_idx == 0
        if draw_boxes:
            _draw_series_boxes_on_ax(
                ax,
                idx=idx,
                stats=stats,
                width=box_width,
                color=str(s["color"]),
                box_alpha=0.18 if is_ceiling else 0.30,
                zorder=s_idx + 1,
            )
        _draw_series_smooth_line_on_ax(
            ax,
            idx=idx,
            stats=stats,
            color=str(s["color"]),
            linestyle=str(s.get("linestyle", "-")),
            linewidth=2.4 if is_ceiling else 1.6,
            zorder=20 + s_idx,
        )
        rendered += 1

    if rendered == 0:
        plt.close(fig)
        logger.warning("Metric %s: no finite values for any series.", metric)
        return None

    centers = np.arange(len(cond_order))
    ax.set_xticks(centers)
    if annotate_xticks and len(cond_order) <= max_xtick_labels:
        ax.set_xticklabels(
            [_condition_short_label(ck) for ck in cond_order],
            rotation=90,
            ha="center",
            va="top",
            fontsize=7,
        )
    else:
        step = max(1, len(cond_order) // 20)
        ax.set_xticks(centers[::step])
        ax.set_xticklabels([str(i) for i in centers[::step]], fontsize=8)
    ax.set_xlim(-0.5, len(cond_order) - 0.5)
    ax.tick_params(axis="x", pad=2)
    _apply_cross_condition_metric_ylim(ax, metric)
    ax.grid(True, axis="y", alpha=0.3)
    ax.margins(x=0.02)

    proxies = [
        Line2D(
            [0], [0],
            color=str(s["color"]),
            linestyle=str(s.get("linestyle", "-")),
            linewidth=2.0,
        )
        for s in series
    ]
    legend_labels = [
        (
            str(s["label"])
            if (s_idx == 0 or not truncate_legend_labels)
            else _truncate_middle_label(
                str(s["label"]), max_len=CROSS_PLOT_MODEL_LABEL_MIN_CHARS,
            )
        )
        for s_idx, s in enumerate(series)
    ]
    n_legend = len(series)
    ncol = int(legend_ncol) if legend_ncol is not None else (
        4 if n_legend > 8 else 1
    )

    title_lines = [f"{pretty_metric_name(metric)} — Model vs Noise Ceiling"]
    if protocol_subtitle:
        title_lines.append(protocol_subtitle)
    if draw_boxes:
        detail = (
            f"Per-condition box = K-rep mean ± std + rolling-mean trend "
            f"({len(cond_order)} conditions, sorted by ceiling: worst → best, left → right)"
        )
    else:
        detail = (
            f"Rolling-mean trend only ({len(cond_order)} conditions; "
            f"per-condition boxes omitted above {CROSS_PLOT_BOX_MAX_CONDITIONS} conditions)"
        )
    title_lines.append(detail)
    ax.set_title("\n".join(title_lines), fontsize=13, fontweight="bold")
    ax.set_ylabel(pretty_metric_name(metric), fontsize=12)
    ax.set_xlabel("Condition (sorted by noise ceiling)", fontsize=11, labelpad=4)

    bottom = _plot_bottom_margin(
        n_legend=n_legend,
        n_conditions=len(cond_order),
        annotate_xticks=annotate_xticks,
        max_xtick_labels=max_xtick_labels,
    )
    # Extra room when the legend is multi-column / many series
    if ncol > 1 or n_legend > 6:
        bottom = max(bottom, 0.18 + 0.03 * ((n_legend + ncol - 1) // ncol))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.86, bottom=bottom)
    legend_y = max(0.015, bottom * 0.22)
    fig.legend(
        proxies,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=ncol,
        frameon=False,
        fontsize=8 if n_legend > 6 else 9,
    )

    fig.savefig(out_path, dpi=dpi, pad_inches=0.12)
    plt.close(fig)
    logger.info("Saved per-condition boxplot: %s", out_path)
    return out_path
