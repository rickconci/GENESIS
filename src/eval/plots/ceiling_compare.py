"""Noise-ceiling strategy comparison and pairwise ceiling plots."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy import stats as sp_stats

from .boxplots import (
    _draw_comparison_boxplot_on_ax,
    _finite_metric_values_for_conditions,
)
from .bundles import SummaryBundle, load_bootstrap_ceiling
from .conditions import (
    aggregated_scalar_field,
    bundle_plot_drop_variables,
    entry_condition_key,
    safe_filename_part,
    _filter_metrics_for_plot,
)
from .constants import BOOTSTRAP_CEILING_FILENAME
from .metric_labels import is_lower_better, pretty_metric_name

logger = logging.getLogger(__name__)


def _condition_field_from_ref_results(
    ref_bundle: SummaryBundle,
    field: str,
) -> dict[str, float]:
    """Map condition key -> numeric field from the reference summary ``results``."""
    out: dict[str, float] = {}
    for entry in ref_bundle.results:
        ck = entry_condition_key(entry, ref_bundle.condition_variables)
        if field not in entry:
            out[ck] = float("nan")
        else:
            out[ck] = float(entry[field])
    return out


def _order_conditions_by_field(
    conds: Sequence[str],
    sort_values: dict[str, float],
    *,
    descending: bool,
) -> list[str]:
    """Stable order: by ``sort_values`` (NaN / missing last)."""

    def _key(ck: str) -> tuple[int, float, str]:
        v = sort_values.get(ck, float("nan"))
        if math.isnan(v):
            return (1, 0.0, ck)
        v_sort = -v if descending else v
        return (0, v_sort, ck)

    return sorted(conds, key=_key)


def _plot_ceilings_only_cross(
    *,
    metric: str,
    cond_order: list[str],
    means: list[dict[str, dict[str, float]]],
    stds: list[dict[str, dict[str, float]]],
    labels: list[str],
    out_path: Path,
    figsize: tuple[float, float],
    dpi: int,
    sort_by: str,
    sort_descending: bool,
    strategy_colors: list[str],
    annotate_xticks: bool,
    max_xtick_labels: int,
    draw_errorbars: bool,
) -> None:
    """Cross-condition panel: one boxplot per noise-ceiling strategy."""
    if not cond_order:
        return

    lower_better = is_lower_better(metric)
    rows: list[dict[str, Any]] = []
    for i, label in enumerate(labels):
        vals = _finite_metric_values_for_conditions(means[i], metric, cond_order)
        if vals.size == 0:
            continue
        rows.append({
            "label": label,
            "values": vals,
            "median": float(np.median(vals)),
            "kind": "ceiling",
            "color_idx": i,
        })
    if not rows:
        return
    rows.sort(key=lambda r: float(r["median"]), reverse=lower_better)
    facecolors = [
        strategy_colors[int(r["color_idx"]) % len(strategy_colors)] for r in rows
    ]

    fig_w = max(8.0, 0.6 * len(rows) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, 6.0))
    _draw_comparison_boxplot_on_ax(
        ax, rows, metric=metric, facecolors=facecolors,
    )
    ax.set_title(
        f"{pretty_metric_name(metric)} — all noise-ceiling strategies\n"
        f"Distribution across {len(cond_order)} conditions (ranked worst → best by median)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel(pretty_metric_name(metric), fontsize=12)
    ax.set_xlabel("Strategy rank (worst → best)", fontsize=12)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ranked cross-condition ceiling boxplot: %s", out_path)


def plot_noise_ceilings_cross_condition(
    ceiling_paths: Sequence[Path | str],
    *,
    labels: Sequence[str] | None = None,
    sort_by: str = "n_truth",
    sort_descending: bool = False,
    sort_from_index: int = 0,
    metrics: list[str] | None = None,
    out_dir: Path | None = None,
    figsize_per_metric: tuple[float, float] = (10, 9),
    dpi: int = 200,
    strategy_colors: list[str] | None = None,
    annotate_xticks: bool = True,
    max_xtick_labels: int = 80,
    with_repeat_errorbars: bool = True,
    allowed_condition_keys: set[tuple[str, float]] | None = None,
    allowed_cell_names: set[str] | None = None,
) -> list[Path]:
    """Cross-condition traces for multiple noise ceilings (like ``cross`` for models).

    Each strategy is a coloured offset scatter + rolling-mean line; repeat
    std is drawn as optional vertical error bars.  Shared x order comes from
    the reference summary's ``results`` field (e.g. ``n_truth``), not from
    the metric's own value (unlike :func:`plot_model_vs_ceiling_across_conditions`).

    Args:
        ceiling_paths: At least one ``bootstrap_noise_ceiling_summary.json`` path
            (typically one per ``--strategy`` run).
        labels: One legend label per path (defaults from
            :func:`_default_noise_ceiling_pair_label`).  Must match the number
            of paths when given.
        sort_by: Name of a per-condition field in ``results`` (e.g. ``n_truth``,
            ``n_control``, ``n_plates``) used to order the x axis.
        sort_descending: If True, larger ``sort_by`` values appear on the
            **left** (e.g. most ``n_truth`` first).  If False (default for
            ``n_truth``), smaller values are on the left—so **fewer cells on the
            left, more cells on the right**.
        sort_from_index: Which of ``ceiling_paths`` to read ``results`` from
            for the sort field (all runs should list the same conditions).
        metrics: Subset of metric keys, or ``None`` for the intersection
            across all strategies' ``mean_metrics``.
        out_dir: Output directory (default:
            ``<first path>.parent / "ceiling_strategies_ranked__<sort_by>"``).
        figsize_per_metric: Base figure size (height grows with the legend).
        dpi: Output resolution.
        strategy_colors: Per-strategy colors (Wong defaults if omitted).
        annotate_xticks: Per-condition x tick labels.
        max_xtick_labels: Cap before switching to subsampled index ticks.
        with_repeat_errorbars: Draw y-error bars (std over repeats) per point.
        allowed_condition_keys: Optional holdout (drug, dose) filter.
        allowed_cell_names: Optional cell-line filter.

    Returns:
        List of one PNG per metric.
    """
    paths = [Path(p) for p in ceiling_paths]
    n_run = len(paths)
    if n_run < 1:
        return []

    if labels is not None and len(labels) != n_run:
        raise ValueError(
            f"labels length {len(labels)} != ceiling_paths length {n_run}"
        )
    if not (0 <= sort_from_index < n_run):
        raise ValueError(
            f"sort_from_index {sort_from_index} out of range for {n_run} paths"
        )
    resolved_labels: list[str] = [
        (labels[i] if labels is not None else _default_noise_ceiling_pair_label(paths[i]))
        for i in range(n_run)
    ]

    bundles = [load_bootstrap_ceiling(p) for p in paths]
    means: list[dict[str, dict[str, float]]] = []
    stds: list[dict[str, dict[str, float]]] = []
    for b in bundles:
        m_f, s_f = _filter_metrics_for_plot(
            b,
            allowed_condition_keys=allowed_condition_keys,
            allowed_cell_names=allowed_cell_names,
        )
        means.append(m_f)
        stds.append(s_f)

    shared: set[str] = set(means[0].keys())
    for m_f in means[1:]:
        shared &= set(m_f.keys())
    shared_conds: list[str] = sorted(shared)
    if not shared_conds:
        logger.warning("No shared conditions after filters for ranked ceiling plot.")
        return []

    ref = bundles[sort_from_index]
    drop = bundle_plot_drop_variables(ref)
    if drop:
        sort_vals = aggregated_scalar_field(
            ref.results, ref.condition_variables, sort_by, drop_variables=drop,
        )
    else:
        sort_vals = _condition_field_from_ref_results(ref, sort_by)
    cond_order = _order_conditions_by_field(
        shared_conds, sort_vals, descending=sort_descending
    )
    logger.info(
        "Ranked cross-condition: %d conditions, order by %s from path index %d.",
        len(cond_order), sort_by, sort_from_index,
    )

    if metrics is not None:
        inter = _intersection_metric_keys_across_ceilings(means, cond_order)
        plot_metrics = [m for m in metrics if m in inter]
    else:
        plot_metrics = sorted(
            _intersection_metric_keys_across_ceilings(means, cond_order)
        )
    if not plot_metrics:
        logger.warning("No metrics to plot for ranked cross-condition ceilings.")
        return []

    if out_dir is None:
        safe_sort = safe_filename_part(sort_by)
        out_dir = paths[0].parent / f"ceiling_strategies_ranked__{safe_sort}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    default_palette = [
        "#E69F00", "#56B4E9", "#009E73", "#CC79A7",
        "#0072B2", "#D55E00", "#F0E442", "#000000",
    ]
    if strategy_colors is None:
        strategy_colors = default_palette

    saved: list[Path] = []
    for metric in plot_metrics:
        out_m = out_dir / (
            f"ceiling_strategies_ranked__{safe_filename_part(sort_by)}__"
            f"{safe_filename_part(metric.replace('/', '-'))}.png"
        )
        _plot_ceilings_only_cross(
            metric=metric,
            cond_order=cond_order,
            means=means,
            stds=stds,
            labels=resolved_labels,
            out_path=out_m,
            figsize=figsize_per_metric,
            dpi=dpi,
            sort_by=sort_by,
            sort_descending=sort_descending,
            strategy_colors=strategy_colors,
            annotate_xticks=annotate_xticks,
            max_xtick_labels=max_xtick_labels,
            draw_errorbars=with_repeat_errorbars,
        )
        saved.append(out_m)

    logger.info("Ranked cross-condition ceiling plots: %d -> %s", len(saved), out_dir)
    return saved


def _default_noise_ceiling_pair_label(summary_path: Path) -> str:
    """Short label for a ceiling JSON (prefer sampling strategy directory name)."""
    p = Path(summary_path)
    if p.parent.parent.name and p.name == BOOTSTRAP_CEILING_FILENAME:
        return p.parent.parent.name
    return p.parent.name or p.stem


def _union_metric_keys_for_conditions(
    metrics_by_condition: dict[str, dict[str, float]],
    condition_keys: Iterable[str],
) -> set[str]:
    """All metric keys that appear in any of the given conditions."""
    out: set[str] = set()
    for k in condition_keys:
        out |= set(metrics_by_condition.get(k, {}).keys())
    return out


def _intersection_metric_keys_across_ceilings(
    means: Sequence[dict[str, dict[str, float]]],
    shared_conds: Sequence[str],
) -> set[str]:
    """Metric keys that appear in every ceiling file for all shared conditions."""
    if not means:
        return set()
    key_sets: list[set[str]] = []
    for m_dict in means:
        key_sets.append(_union_metric_keys_for_conditions(m_dict, shared_conds))
    out = key_sets[0].copy()
    for ks in key_sets[1:]:
        out &= ks
    return out


def _unordered_pair_indices(n: int) -> list[tuple[int, int]]:
    """All (i, j) with 0 <= i < j < n."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _ceiling_pair_scatter_on_ax(
    ax: Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    x_err: np.ndarray | None,
    y_err: np.ndarray | None,
    x_label: str,
    y_label: str,
    with_error_bars: bool,
    xlabel_fontsize: int = 9,
    ylabel_fontsize: int = 9,
) -> tuple[int, float | None]:
    """Draw y vs x + identity line; axis labels. Returns (n_points, pearson_r or None)."""
    n = int(xs.size)
    if n == 0:
        return 0, None
    use_err = (
        with_error_bars
        and x_err is not None
        and y_err is not None
        and len(x_err) == n
        and len(y_err) == n
        and (np.any(x_err > 0) or np.any(y_err > 0))
    )
    if use_err:
        ax.errorbar(
            xs,
            ys,
            xerr=x_err,
            yerr=y_err,
            fmt="none",
            ecolor="#888888",
            elinewidth=0.6,
            capsize=0,
            alpha=0.25,
            zorder=1,
        )
    ax.scatter(xs, ys, s=8.0, alpha=0.5, c="#2C7FB8", edgecolors="none", zorder=2)
    lo = float(min(float(xs.min()), float(ys.min())))
    hi = float(max(float(xs.max()), float(ys.max())))
    span = hi - lo
    if span <= 0:
        pad = 1e-9 if lo == 0 else abs(lo) * 1e-6
        lo -= pad
        hi += pad
    else:
        pad = 0.02 * span
        lo -= pad
        hi += pad
    ax.plot([lo, hi], [lo, hi], color="#444444", linestyle="--", linewidth=1.2, alpha=0.75, zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{x_label} (mean over repeats)", fontsize=xlabel_fontsize)
    ax.set_ylabel(f"{y_label} (mean over repeats)", fontsize=ylabel_fontsize)
    ax.grid(True, alpha=0.3)
    r_val: float | None
    if n >= 2:
        r_val = float(sp_stats.pearsonr(xs.astype(np.float64), ys.astype(np.float64))[0])
    else:
        r_val = None
    return n, r_val


def _plot_single_metric_ceiling_pair(
    *,
    metric: str,
    x_label: str,
    y_label: str,
    xs: np.ndarray,
    ys: np.ndarray,
    x_err: np.ndarray | None,
    y_err: np.ndarray | None,
    out_path: Path,
    figsize: tuple[float, float],
    dpi: int,
    with_error_bars: bool,
) -> None:
    """One scatter: ceiling B vs ceiling A for a single metric (paired conditions)."""
    n = int(xs.size)
    if n == 0:
        return
    fig, ax = plt.subplots(figsize=figsize)
    n_pts, r_val = _ceiling_pair_scatter_on_ax(
        ax,
        xs,
        ys,
        x_err=x_err,
        y_err=y_err,
        x_label=x_label,
        y_label=y_label,
        with_error_bars=with_error_bars,
        xlabel_fontsize=12,
        ylabel_fontsize=12,
    )
    r_txt = f"Pearson r = {r_val:.4f} (n = {n_pts})" if r_val is not None else f"n = {n_pts}"
    ax.set_title(
        f"{pretty_metric_name(metric)} — {y_label} vs {x_label}\n{r_txt}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ceiling-pair plot: %s", out_path)


def plot_noise_ceiling_vs_noise_ceiling(
    ceiling_a_path: Path,
    ceiling_b_path: Path,
    *,
    label_a: str | None = None,
    label_b: str | None = None,
    metrics: list[str] | None = None,
    out_dir: Path | None = None,
    figsize: tuple[float, float] = (7.0, 7.0),
    dpi: int = 200,
    with_error_bars: bool = False,
    allowed_condition_keys: set[tuple[str, float]] | None = None,
    allowed_cell_names: set[str] | None = None,
) -> list[Path]:
    """Per-metric scatter: noise ceiling B vs A for all paired conditions.

    Each point is one (cell, drug, dose) present in both summaries; x/y are
    :func:`load_bootstrap_ceiling` point estimates (``mean_metrics``). Optional
    error bars use the repeat-std already stored in the JSON.

    Args:
        ceiling_a_path: Path to first ``bootstrap_noise_ceiling_summary.json``
            (x-axis).
        ceiling_b_path: Path to second summary (y-axis).
        label_a: Legend / axis label for the first file.  Default: parent of
            the strategy folder (e.g. ``bootstrap``) when using the standard
            ``results/<dataset>/<strategy>/noise_ceiling/geneset_<hash>__n_<N>__norm_<id>/...`` layout.
        label_b: Label for the second file (same default rule).
        metrics: Subset of metric keys to plot.  ``None`` = intersection of
            all keys appearing in either summary over conditions that pass
            filters.
        out_dir: Output directory (default:
            ``<ceiling_b_path>.parent / "ceiling_vs_ceiling_plots__<a>__<b>"``).
        figsize: Figure size per metric (square is recommended).
        dpi: Output resolution.
        with_error_bars: If True, draw light repeat-std error bars (can be
            busy for many conditions).
        allowed_condition_keys: Optional ``(drug, rounded_dose)`` set from
            a holdout YAML (same convention as
            :func:`plot_model_vs_ceiling_across_conditions`).
        allowed_cell_names: Optional cell lines to keep.

    Returns:
        List of saved PNG paths (one per metric with at least one paired point).
    """
    a_path = Path(ceiling_a_path)
    b_path = Path(ceiling_b_path)
    ceiling_a = load_bootstrap_ceiling(a_path)
    ceiling_b = load_bootstrap_ceiling(b_path)

    a_mean, a_std = _filter_metrics_for_plot(
        ceiling_a,
        allowed_condition_keys=allowed_condition_keys,
        allowed_cell_names=allowed_cell_names,
    )
    b_mean, b_std = _filter_metrics_for_plot(
        ceiling_b,
        allowed_condition_keys=allowed_condition_keys,
        allowed_cell_names=allowed_cell_names,
    )

    shared_conds = sorted(set(a_mean.keys()) & set(b_mean.keys()))
    logger.info(
        "Ceiling pair: %d shared conditions (A had %d, B had %d after filters).",
        len(shared_conds),
        len(a_mean),
        len(b_mean),
    )
    if not shared_conds:
        logger.warning("No overlapping conditions after filters; no plots written.")
        return []

    x_lab = label_a or _default_noise_ceiling_pair_label(a_path)
    y_lab = label_b or _default_noise_ceiling_pair_label(b_path)

    keys_a = _union_metric_keys_for_conditions(a_mean, shared_conds)
    keys_b = _union_metric_keys_for_conditions(b_mean, shared_conds)
    if metrics is not None:
        plot_metrics = [m for m in metrics if m in keys_a and m in keys_b]
    else:
        plot_metrics = sorted(keys_a & keys_b)

    if not plot_metrics:
        logger.warning("No metrics to plot (empty intersection or filter).")
        return []

    if out_dir is None:
        safe_x = safe_filename_part(x_lab)
        safe_y = safe_filename_part(y_lab)
        out_dir = b_path.parent / f"ceiling_vs_ceiling_plots__{safe_x}__{safe_y}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for metric in plot_metrics:
        xs_list: list[float] = []
        ys_list: list[float] = []
        xe_list: list[float] = []
        ye_list: list[float] = []
        for ck in shared_conds:
            ma = a_mean[ck]
            mb = b_mean[ck]
            if metric not in ma or metric not in mb:
                continue
            xs_list.append(float(ma[metric]))
            ys_list.append(float(mb[metric]))
            if with_error_bars:
                xe_list.append(float(a_std[ck].get(metric, 0.0)))
                ye_list.append(float(b_std[ck].get(metric, 0.0)))
        if not xs_list:
            logger.debug("Metric %s: no paired values; skip.", metric)
            continue
        xs = np.array(xs_list, dtype=np.float64)
        ys = np.array(ys_list, dtype=np.float64)
        x_err: np.ndarray | None
        y_err: np.ndarray | None
        if with_error_bars and xe_list and ye_list:
            x_err = np.array(xe_list, dtype=np.float64)
            y_err = np.array(ye_list, dtype=np.float64)
        else:
            x_err = None
            y_err = None

        safe_m = safe_filename_part(metric.replace("/", "-"))
        out_png = out_dir / f"ceiling_pair__{safe_filename_part(x_lab)}__{safe_filename_part(y_lab)}__{safe_m}.png"
        _plot_single_metric_ceiling_pair(
            metric=metric,
            x_label=x_lab,
            y_label=y_lab,
            xs=xs,
            ys=ys,
            x_err=x_err,
            y_err=y_err,
            out_path=out_png,
            figsize=figsize,
            dpi=dpi,
            with_error_bars=with_error_bars,
        )
        saved.append(out_png)

    logger.info("Ceiling-vs-ceiling scatter plots: %d metrics -> %s", len(saved), out_dir)
    return saved


def plot_noise_ceilings_all_pairs(
    ceiling_paths: Sequence[Path | str],
    *,
    labels: Sequence[str] | None = None,
    metrics: list[str] | None = None,
    out_dir: Path | None = None,
    panel_size: tuple[float, float] = (3.6, 3.3),
    dpi: int = 200,
    with_error_bars: bool = False,
    ncols: int = 3,
    allowed_condition_keys: set[tuple[str, float]] | None = None,
    allowed_cell_names: set[str] | None = None,
) -> list[Path]:
    """One PNG per metric: all unordered pairs of K noise-ceiling runs in a grid.

    For *K* summaries (e.g. each ``--strategy``), draws :math:`\\binom{K}{2}`
    scatter panels (y vs x, per-condition means, y = x line) on a single
    figure, plus Pearson r in each panel title.

    Args:
        ceiling_paths: At least two paths to ``bootstrap_noise_ceiling_summary.json``
            (same gene set / split is assumed; conditions are intersected).
        labels: One label per path (defaults from
            :func:`_default_noise_ceiling_pair_label`).  Must match length of
            ``ceiling_paths`` when provided.
        metrics: Subset of metric keys.  ``None`` = keys present in *every*
            run for *every* shared condition.
        out_dir: Output directory (default:
            ``<first_path>.parent / "ceiling_all_pairs_plots"``).
        panel_size: (width, height) in inches of each subplot.
        dpi: Output resolution.
        with_error_bars: Light repeat-std x/y error bars in each panel.
        ncols: Number of columns in the pairwise grid.
        allowed_condition_keys: Optional ``(drug, dose)`` filter (holdout YAML).
        allowed_cell_names: Optional cell-line filter.

    Returns:
        List of saved PNG paths (one per metric with at least one non-empty
        pair panel).
    """
    paths = [Path(p) for p in ceiling_paths]
    n_runs = len(paths)
    if n_runs < 2:
        logger.warning("Need at least two noise-ceiling JSON paths; got %d.", n_runs)
        return []

    if labels is not None and len(labels) != n_runs:
        raise ValueError(f"labels length {len(labels)} != ceiling_paths length {n_runs}")

    resolved_labels: list[str] = [
        (labels[i] if labels is not None else _default_noise_ceiling_pair_label(paths[i]))
        for i in range(n_runs)
    ]

    bundles = [load_bootstrap_ceiling(p) for p in paths]
    means: list[dict[str, dict[str, float]]] = []
    stds: list[dict[str, dict[str, float]]] = []
    for b in bundles:
        m_f, s_f = _filter_metrics_for_plot(
            b,
            allowed_condition_keys=allowed_condition_keys,
            allowed_cell_names=allowed_cell_names,
        )
        means.append(m_f)
        stds.append(s_f)

    shared_conds: set[str] = set(means[0].keys())
    for m_f in means[1:]:
        shared_conds &= set(m_f.keys())
    shared_conds_list = sorted(shared_conds)
    logger.info(
        "All-pairs noise ceiling: %d shared conditions across %d runs.",
        len(shared_conds_list),
        n_runs,
    )
    if not shared_conds_list:
        logger.warning("No overlapping conditions after filters; no plots written.")
        return []

    if metrics is not None:
        base = _intersection_metric_keys_across_ceilings(means, shared_conds_list)
        plot_metrics = [m for m in metrics if m in base]
    else:
        plot_metrics = sorted(_intersection_metric_keys_across_ceilings(means, shared_conds_list))

    if not plot_metrics:
        logger.warning("No metrics to plot (empty key intersection or filter).")
        return []

    if out_dir is None:
        out_dir = paths[0].parent / "ceiling_all_pairs_plots"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = _unordered_pair_indices(n_runs)
    n_pairs = len(pairs)
    ncols_eff = min(max(1, ncols), n_pairs)
    nrows = int(math.ceil(n_pairs / ncols_eff))
    fig_w = ncols_eff * panel_size[0]
    fig_h = nrows * panel_size[1]

    saved: list[Path] = []
    for metric in plot_metrics:
        fig, axes_2d = plt.subplots(
            nrows, ncols_eff, figsize=(fig_w, fig_h), squeeze=False,
        )
        axes_flat: list[Axes] = list(axes_2d.ravel())
        for ax in axes_flat[n_pairs:]:
            ax.set_visible(False)

        any_data = False
        for slot, (i, j) in enumerate(pairs):
            ax = axes_flat[slot]
            xs_list: list[float] = []
            ys_list: list[float] = []
            xe_list: list[float] = []
            ye_list: list[float] = []
            for ck in shared_conds_list:
                mi = means[i].get(ck, {})
                mj = means[j].get(ck, {})
                if metric not in mi or metric not in mj:
                    continue
                xs_list.append(float(mi[metric]))
                ys_list.append(float(mj[metric]))
                if with_error_bars:
                    xe_list.append(float(stds[i][ck].get(metric, 0.0)))
                    ye_list.append(float(stds[j][ck].get(metric, 0.0)))
            if not xs_list:
                ax.set_visible(False)
                continue
            any_data = True
            xs = np.array(xs_list, dtype=np.float64)
            ys = np.array(ys_list, dtype=np.float64)
            if with_error_bars and xe_list and ye_list:
                x_e = np.array(xe_list, dtype=np.float64)
                y_e = np.array(ye_list, dtype=np.float64)
            else:
                x_e = None
                y_e = None
            li, lj = resolved_labels[i], resolved_labels[j]
            n_pts, r_val = _ceiling_pair_scatter_on_ax(
                ax,
                xs,
                ys,
                x_err=x_e,
                y_err=y_e,
                x_label=li,
                y_label=lj,
                with_error_bars=with_error_bars,
            )
            r_txt = f"r = {r_val:.4f} (n = {n_pts})" if r_val is not None else f"n = {n_pts}"
            ax.set_title(f"{lj} vs {li}\n{r_txt}", fontsize=9, fontweight="bold")

        if not any_data:
            plt.close(fig)
            logger.debug("Metric %s: no pairwise data; skip.", metric)
            continue

        fig.suptitle(
            f"{pretty_metric_name(metric)} — all pairwise noise ceilings",
            fontsize=14,
            fontweight="bold",
            y=1.01,
        )
        fig.tight_layout()
        safe_m = safe_filename_part(metric.replace("/", "-"))
        out_png = out_dir / f"ceiling_all_pairs__{safe_m}.png"
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_png)
        logger.info("Saved all-pairs ceiling plot: %s", out_png)

    logger.info("All-pair ceiling plots: %d metrics -> %s", len(saved), out_dir)
    return saved

