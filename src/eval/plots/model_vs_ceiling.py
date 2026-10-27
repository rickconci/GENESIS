"""Model vs noise-ceiling cross-condition plots."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .boxplots import _plot_single_metric, format_cross_plot_model_label
from .bundles import SummaryBundle, _model_legend_label, load_bootstrap_ceiling, load_metrics_summary
from .conditions import _filter_metrics_for_plot, normalize_bundle_condition_keys
from .metric_labels import (
    metric_eval_protocol_label,
    overlap_model_metric_keys,
    overlap_plot_metric_keys,
)
from .summary_artifacts import (
    _save_model_metric_summary_artifacts,
    _save_model_winrate_artifacts,
)

logger = logging.getLogger(__name__)


def series_ids(paths: Sequence[Path]) -> list[str]:
    """Plot id per summary. The parent folder name when that name is unique.

    5ep and 50ep both write ``eval/after_fit/test/metrics_summary.json``.
    Keying only on ``test`` would keep the second file and drop the first line.
    """
    resolved = [Path(p).resolve() for p in paths]
    bases = [p.parent.name for p in resolved]
    ids: list[str] = []
    for path, base in zip(resolved, bases):
        if bases.count(base) == 1:
            ids.append(base)
            continue
        # parents: test, after_fit, eval, <run dir>
        run = path.parents[3].name if len(path.parents) > 3 else path.parent.as_posix()
        ids.append(f"{run}__{base}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"plot series ids still collide: {ids}")
    return ids


def plot_model_vs_ceiling_across_conditions(
    ceiling_summary_path: Path | None,
    model_summary_paths: Sequence[Path],
    *,
    include_ceiling: bool = True,
    metrics: list[str] | None = None,
    eval_strategy: str = "split_half",
    model_labels: dict[str, str] | None = None,
    out_dir: Path | None = None,
    figsize_per_metric: tuple[float, float] = (10, 9),
    dpi: int = 200,
    ceiling_color: str = "#888888",
    model_colors: list[str] | None = None,
    annotate_xticks: bool = True,
    max_xtick_labels: int = 80,
    allowed_condition_keys: set[tuple[str, float]] | None = None,
    allowed_cell_names: set[str] | None = None,
    required_plot_condition_keys: set[str] | None = None,
    drop_variables: Sequence[str] | None = (),
    skip_existing: bool = False,
    summary_plots: bool = False,
    jobs: int = 1,
) -> list[Path]:
    """Generate per-metric cross-condition boxplots (ceiling + models).

    Reads directly from the bespoke summary JSONs produced by the live
    eval pipelines -- no intermediary schema classes are involved.

    Each figure shows one box per series (noise ceiling + each model): values
    are the per-condition metric scores, ranked worst → best by median.

    Args:
        ceiling_summary_path: Path to ``noise_ceiling.json`` (bootstrap summary).
            ``None`` when ``include_ceiling=False``.
        include_ceiling: When False, plot models only (no noise-ceiling box).
        model_summary_paths: One or more paths to ``metrics_summary.json``
            files (typically one per model).
        metrics: Which metric keys to plot (e.g.
            ``["log1p_slice_med_rgs/pearson_r", "raw_int_rgs/mse"]``).  ``None`` = overlapping
            metrics after the default filter (log1p_cpm arm/dist, raw_int
            pseudobulk delta, deg_raw_int; no pf_log_pf).
        eval_strategy: Strategy label appended to plot subtitles.
        model_labels: Override display labels keyed by the model summary
            path's parent dir name.  ``None`` = auto via the built-in
            heuristic.
        out_dir: Output directory (default:
            ``<ceiling_summary_path>.parent / "model_vs_ceiling_plots"``).
        figsize_per_metric: Unused (width scales with number of series).
        dpi: Resolution.
        ceiling_color: Fill color for the noise-ceiling box.
        model_colors: Per-model box colors (Wong colorblind-safe palette by
            default).
        annotate_xticks: Unused (kept for API compatibility).
        max_xtick_labels: Unused (kept for API compatibility).
        allowed_condition_keys: Optional set of ``(drug, rounded_dose)`` to
            filter conditions (e.g. from a holdout YAML).
        allowed_cell_names: Optional set of cell-line names to include.
        required_plot_condition_keys: Optional set of plot-level condition-key
            strings; when set, only these keys are plotted (e.g. cross-model
            intersection from ``compare-jsons --intersect-condition-keys``).
        drop_variables: Plot aggregation axes to drop (e.g. ``("plate",)``).
            Default ``()`` → one point per scored row (per-plate when plate is
            scored), ignoring ``plot_aggregate_over`` in summary JSON headers.
            Pass ``None`` to use each file's ``plot_aggregate_over`` header
            (e.g. after eval with ``plot_aggregate_over: plate`` in config).
        skip_existing: If True, do not re-render ``model_vs_ceiling__<metric>.png``
            when the file already exists; return that path and log a skip.
        summary_plots: When True and there are multiple models, also write
            ``summary_plots/`` win-rate and model-metric summary artifacts.
        jobs: Parallel metric renders (thread pool; default 1 = sequential).

    Returns:
        List of saved PNG paths, one per plotted metric.
    """
    if include_ceiling:
        if ceiling_summary_path is None:
            raise ValueError(
                "ceiling_summary_path is required when include_ceiling=True.",
            )
        ceiling = normalize_bundle_condition_keys(
            load_bootstrap_ceiling(Path(ceiling_summary_path)),
        )
        nc_mean, nc_std = _filter_metrics_for_plot(
            ceiling,
            drop_variables=drop_variables,
            allowed_condition_keys=allowed_condition_keys,
            allowed_cell_names=allowed_cell_names,
        )
        if required_plot_condition_keys is not None:
            nc_mean = {
                k: v for k, v in nc_mean.items() if k in required_plot_condition_keys
            }
            nc_std = {
                k: v for k, v in nc_std.items() if k in required_plot_condition_keys
            }
        logger.info(
            "Ceiling: %d plot-level conditions (from %d scored rows).",
            len(nc_mean), len(ceiling.results),
        )
    else:
        nc_mean, nc_std = {}, {}
        logger.info("Ceiling omitted (models-only cross plots).")

    models: list[tuple[Path, SummaryBundle]] = []
    for p in model_summary_paths:
        p = Path(p)
        models.append((p, normalize_bundle_condition_keys(load_metrics_summary(p))))

    model_data: dict[str, dict[str, dict[str, float]]] = {}
    model_stds: dict[str, dict[str, dict[str, float]]] = {}
    model_ids: list[str] = []
    model_id_labels: dict[str, str] = {}
    for (summary_path, bundle), mid in zip(models, series_ids([p for p, _ in models])):
        if mid in model_id_labels:
            raise RuntimeError(f"duplicate plot series id {mid}")
        model_ids.append(mid)
        label = None
        if model_labels is not None:
            label = model_labels.get(mid)
        raw_label = label or _model_legend_label(summary_path, mid)
        model_id_labels[mid] = format_cross_plot_model_label(
            raw_label,
            summary_path,
            explicit=label is not None,
        )
        filtered_m, filtered_s = _filter_metrics_for_plot(
            bundle,
            drop_variables=drop_variables,
            allowed_condition_keys=allowed_condition_keys,
            allowed_cell_names=allowed_cell_names,
        )
        if required_plot_condition_keys is not None:
            filtered_m = {
                k: v for k, v in filtered_m.items() if k in required_plot_condition_keys
            }
            filtered_s = {
                k: v for k, v in filtered_s.items() if k in required_plot_condition_keys
            }
        model_data[mid] = filtered_m
        model_stds[mid] = filtered_s
        logger.info(
            "Model %s: %d plot-level conditions (from %d scored rows).",
            mid, len(filtered_m), len(bundle.results),
        )

    all_nc_metrics = {m for mm in nc_mean.values() for m in mm.keys()}
    all_model_metrics = {
        m for md in model_data.values() for mm in md.values() for m in mm
    }

    if metrics is not None:
        allowed = all_nc_metrics if include_ceiling else all_model_metrics
        plot_metrics = [m for m in metrics if m in allowed]
    elif include_ceiling:
        plot_metrics = overlap_plot_metric_keys(nc_mean, model_data)
        logger.info(
            "Plotting %d overlapping metrics between ceiling and model(s) "
            "(default filter: log1p_cpm + raw_int delta + deg_raw_int; no pf_log_pf).",
            len(plot_metrics),
        )
    else:
        plot_metrics = overlap_model_metric_keys(model_data)
        logger.info(
            "Plotting %d metrics shared across model(s) "
            "(default filter: log1p_cpm + raw_int delta + deg_raw_int; no pf_log_pf).",
            len(plot_metrics),
        )

    if not plot_metrics:
        logger.warning(
            "No plot metrics (%s).",
            "ceiling vs models" if include_ceiling else "models only",
        )
        return []

    if out_dir is None:
        if include_ceiling and ceiling_summary_path is not None:
            out_dir = Path(ceiling_summary_path).parent / "model_vs_ceiling_plots"
        else:
            out_dir = Path(model_summary_paths[0]).parent / "model_vs_ceiling_plots"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Wong (2011) colorblind-safe palette (Nature Methods recommendation)
    default_palette = [
        "#E69F00", "#56B4E9", "#009E73", "#CC79A7",
        "#0072B2", "#D55E00", "#F0E442", "#000000",
    ]
    if model_colors is None:
        model_colors = default_palette

    saved: list[Path] = []
    write_summary_plots = summary_plots and len(model_ids) > 1
    summary_dir = out_dir / "summary_plots"
    n_jobs = max(1, int(jobs))

    def _render_metric(metric: str) -> list[Path]:
        paths: list[Path] = []
        fig_path = _plot_single_metric(
            metric=metric,
            nc_mean=nc_mean,
            nc_std=nc_std,
            model_data=model_data,
            model_stds=model_stds,
            model_order=model_ids,
            model_id_labels=model_id_labels,
            out_dir=out_dir,
            figsize=figsize_per_metric,
            dpi=dpi,
            ceiling_color=ceiling_color,
            model_colors=model_colors,
            annotate_xticks=annotate_xticks,
            max_xtick_labels=max_xtick_labels,
            skip_existing=skip_existing,
            protocol_subtitle=metric_eval_protocol_label(metric, strategy=eval_strategy),
            # format_cross_plot_model_label already capped auto names.
            # A second cap here hid explicit --labels longer than 25 chars.
            truncate_legend_labels=False,
        )
        if fig_path is not None:
            paths.append(fig_path)
        if write_summary_plots:
            paths.extend(
                _save_model_metric_summary_artifacts(
                    metric=metric,
                    nc_mean=nc_mean,
                    model_order=model_ids,
                    model_id_labels=model_id_labels,
                    model_data=model_data,
                    out_dir=summary_dir,
                )
            )
            paths.extend(
                _save_model_winrate_artifacts(
                    metric=metric,
                    model_order=model_ids,
                    model_id_labels=model_id_labels,
                    model_data=model_data,
                    out_dir=summary_dir,
                    dpi=dpi,
                )
            )
        return paths

    if n_jobs == 1:
        for metric in plot_metrics:
            saved.extend(_render_metric(metric))
    else:
        logger.info("Rendering %d metrics with jobs=%d", len(plot_metrics), n_jobs)
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            futures = [pool.submit(_render_metric, metric) for metric in plot_metrics]
            for fut in as_completed(futures):
                saved.extend(fut.result())

    logger.info(
        "Cross-condition plots saved: %d metrics -> %s", len(saved), out_dir,
    )
    return saved


def plot_model_vs_ceiling(
    model_summary_path: Path,
    ceiling_summary_path: Path,
    *,
    out_dir: Path | None = None,
    metrics: list[str] | None = None,
    model_label: str | None = None,
    dpi: int = 200,
    skip_existing: bool = False,
) -> list[Path]:
    """Plot one model eval summary against one noise-ceiling summary.

    Convenience wrapper around :func:`plot_model_vs_ceiling_across_conditions`
    for the common case: two JSON paths, default metric set, no Tahoe-only
    cell-line / holdout filters.

    Args:
        model_summary_path: ``metrics_summary.json`` from model eval.
        ceiling_summary_path: ``noise_ceiling.json`` from the ceiling run.
        out_dir: Where to write PNGs (default:
            ``<model_summary_path.parent>/model_vs_ceiling_plots``).
        metrics: Metric keys to plot (default: all keys shared by ceiling
            and model summaries).
        model_label: Optional legend label for the model trace.
        dpi: Figure resolution.
        skip_existing: Skip PNGs that already exist on disk.

    Returns:
        List of saved plot paths (cross-condition PNGs only for single-model runs).
    """
    import json

    model_summary_path = Path(model_summary_path)
    ceiling_summary_path = Path(ceiling_summary_path)
    if out_dir is None:
        out_dir = model_summary_path.parent / "model_vs_ceiling_plots"

    strategy = "split_half"
    try:
        header = json.loads(ceiling_summary_path.read_text())
        strategy = str(header.get("strategy") or strategy)
    except (OSError, json.JSONDecodeError, TypeError):
        logger.warning(
            "Could not read strategy from %s; using %r.",
            ceiling_summary_path, strategy,
        )

    model_labels = None
    if model_label is not None:
        model_labels = {series_ids([model_summary_path])[0]: model_label}

    return plot_model_vs_ceiling_across_conditions(
        ceiling_summary_path=ceiling_summary_path,
        model_summary_paths=[model_summary_path],
        metrics=metrics,
        eval_strategy=strategy,
        model_labels=model_labels,
        out_dir=Path(out_dir),
        dpi=dpi,
        allowed_condition_keys=None,
        allowed_cell_names=None,
        skip_existing=skip_existing,
    )
