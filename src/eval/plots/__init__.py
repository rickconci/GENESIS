"""Plotting utilities for model evaluation vs bootstrap noise ceiling.

Submodules:

* :mod:`constants` — shared colours, filenames, norm labels.
* :mod:`bundles` — :class:`SummaryBundle` loaders for model/ceiling JSON.
* :mod:`condition_plots` — per-condition gene scatter / histogram.
* :mod:`model_vs_ceiling` — cross-condition model vs ceiling boxplots.
* :mod:`ceiling_compare` — multi-strategy / pairwise ceiling plots.
"""

from __future__ import annotations

from .bundles import (
    DropoutArrays,
    GeneLevelArrays,
    SummaryBundle,
    load_bootstrap_ceiling,
    load_metrics_summary,
)
from .ceiling_compare import (
    plot_noise_ceiling_vs_noise_ceiling,
    plot_noise_ceilings_all_pairs,
    plot_noise_ceilings_cross_condition,
)
from .condition_plots import (
    plot_condition_all,
    plot_condition_scatter_mean_var,
    plot_gene_variance_histogram,
    resolve_plot_set,
)
from .conditions import condition_short_label, entry_condition_key, safe_filename_part
from .constants import (
    ACCENT_COLOR,
    BOOTSTRAP_CEILING_FILENAME,
    GENE_SUMMARY_SPACES,
    LOG1P_NORM_IDS,
    LOWER_IS_BETTER_METRICS,
    METRICS_SUMMARY_FILENAME,
    NORM_DISPLAY_LABELS,
    PLOT_TYPES,
    PRED_COLOR,
    RAW_NORM_IDS,
    TRUE_COLOR,
)
from .metric_labels import (
    default_model_vs_ceiling_metric_keys,
    is_lower_better,
    metric_eval_protocol_label,
    norm_display_label,
    overlap_plot_metric_keys,
    pretty_metric_name,
)
from .model_vs_ceiling import (
    plot_model_vs_ceiling,
    plot_model_vs_ceiling_across_conditions,
)
from .scatter import draw_scatter_panel

__all__ = [
    "BOOTSTRAP_CEILING_FILENAME",
    "GENE_SUMMARY_SPACES",
    "GeneLevelArrays",
    "LOG1P_NORM_IDS",
    "LOWER_IS_BETTER_METRICS",
    "METRICS_SUMMARY_FILENAME",
    "NORM_DISPLAY_LABELS",
    "PLOT_TYPES",
    "RAW_NORM_IDS",
    "SummaryBundle",
    "TRUE_COLOR",
    "PRED_COLOR",
    "ACCENT_COLOR",
    "condition_short_label",
    "entry_condition_key",
    "draw_scatter_panel",
    "is_lower_better",
    "load_bootstrap_ceiling",
    "load_metrics_summary",
    "norm_display_label",
    "plot_condition_all",
    "plot_condition_scatter_mean_var",
    "plot_gene_variance_histogram",
    "plot_model_vs_ceiling",
    "plot_model_vs_ceiling_across_conditions",
    "plot_noise_ceiling_vs_noise_ceiling",
    "plot_noise_ceilings_all_pairs",
    "plot_noise_ceilings_cross_condition",
    "pretty_metric_name",
    "default_model_vs_ceiling_metric_keys",
    "overlap_plot_metric_keys",
    "metric_eval_protocol_label",
    "resolve_plot_set",
    "safe_filename_part",
]
