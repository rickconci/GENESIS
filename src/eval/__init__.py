"""Evaluation package exports.

Producers and readers speak the unified JSON schema built by
:func:`src.eval.orchestration.model_eval.build_result_row`: every per-condition row
in both ``metrics_summary.json`` (model eval, written by
``scripts/eval/score.py genesis``) and
``noise_ceiling.json`` (noise ceiling, written by
``scripts/eval/reference.py split_half``) carries
``condition_keys``, ``k``, ``metrics`` (clustered into the four
metric groups) and ``metrics_std``.
"""

from .metrics.arm import compute_arm_metrics
from .metrics.deg import compute_deg_detail, compute_deg_overlap_metrics
from .metrics.delta import compute_delta_metrics
from .metrics.eval_metrics import compute_arm_dist_bundle, compute_delta_deg_bundle
from .metrics.reporting import materialize_metric_tensor_batch, materialize_metric_tensors
from .metrics.gene_summary import compute_gene_summary
from .metrics.energy import energy_distance
from .metrics.mmd import mmd_rbf
from .orchestration.model_eval import (
    ModelEvalResult,
    run_model_eval,
)
from .reference.split_half import (
    NoiseCeilingError,
    SplitHalfSampler,
    run_noise_ceiling,
    run_split_half_reference,
)
from .cache.eval_cache_io import (
    ConditionBundle,
    ConditionKey,
    ConditionPayload,
    ConditionPrefetcher,
    ControlCache,
    QuadrupletRow,
    control_npz_path,
    encode_rows,
    gene_set_hash,
    group_by_condition,
    load_work_queue,
    load_work_queue_or_none,
    perturbed_npz_path,
    resolve_cache_namespace,
    safe_filename_part,
)
from .plots import (
    BOOTSTRAP_CEILING_FILENAME,
    GENE_SUMMARY_SPACES,
    GeneLevelArrays,
    LOG1P_NORM_IDS,
    METRICS_SUMMARY_FILENAME,
    NORM_DISPLAY_LABELS,
    PLOT_TYPES,
    RAW_NORM_IDS,
    SummaryBundle,
    draw_scatter_panel,
    load_bootstrap_ceiling,
    load_metrics_summary,
    plot_condition_all,
    plot_condition_scatter_mean_var,
    plot_model_vs_ceiling,
    plot_model_vs_ceiling_across_conditions,
    plot_noise_ceiling_vs_noise_ceiling,
    plot_noise_ceilings_all_pairs,
    plot_noise_ceilings_cross_condition,
    resolve_plot_set,
)

__all__ = [
    # eval_metrics
    "compute_arm_dist_bundle",
    "compute_arm_metrics",
    "compute_delta_metrics",
    "compute_delta_deg_bundle",
    "compute_deg_overlap_metrics",
    "compute_deg_detail",
    "compute_gene_summary",
    "materialize_metric_tensors",
    "materialize_metric_tensor_batch",
    "energy_distance",
    "mmd_rbf",
    # noise_ceiling_eval
    "NoiseCeilingError",
    "run_noise_ceiling",
    "run_split_half_reference",
    # noise_ceiling_samplers
    "SplitHalfSampler",
    # model_eval
    "ModelEvalResult",
    "run_model_eval",
    # pregeneration
    # eval_cache_io
    "ConditionBundle",
    "ConditionKey",
    "ConditionPayload",
    "ConditionPrefetcher",
    "ControlCache",
    "QuadrupletRow",
    "control_npz_path",
    "encode_rows",
    "gene_set_hash",
    "group_by_condition",
    "load_work_queue",
    "load_work_queue_or_none",
    "perturbed_npz_path",
    "resolve_cache_namespace",
    "safe_filename_part",
    # plots
    "BOOTSTRAP_CEILING_FILENAME",
    "GENE_SUMMARY_SPACES",
    "GeneLevelArrays",
    "LOG1P_NORM_IDS",
    "METRICS_SUMMARY_FILENAME",
    "NORM_DISPLAY_LABELS",
    "PLOT_TYPES",
    "RAW_NORM_IDS",
    "SummaryBundle",
    "draw_scatter_panel",
    "load_bootstrap_ceiling",
    "load_metrics_summary",
    "plot_condition_all",
    "plot_condition_scatter_mean_var",
    "plot_model_vs_ceiling",
    "plot_model_vs_ceiling_across_conditions",
    "plot_noise_ceiling_vs_noise_ceiling",
    "plot_noise_ceilings_all_pairs",
    "plot_noise_ceilings_cross_condition",
    "resolve_plot_set",
]
