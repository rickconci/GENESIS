"""PertGen configuration dataclass."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union


@dataclass
class DatasetPaths:
    """Path bundle for one dataset preset (YAML blocks like ``tahoe:``, ``parse:``).

    When ``PertGenConfig.dataset`` matches the preset key (e.g. ``dataset: tahoe``),
    :func:`src.config.loader.load_config` fills unset paths from that block before
    path resolution.

    Attributes:
        data_dir: Plate h5ad root or cache-backed data root for this preset.
        hvg_genes_path: HVG gene list (``.txt`` one name per line, or ``.pkl`` sequence).
        cache_dir: Optional mmap CSR **training** cache root
            (``<data_dir>/hvg_cache/geneset_*``). Independent of eval truth cache.
        noise_ceiling_dir: Noise-ceiling JSON/parquet output root. When unset,
            defaults to ``results/<dataset>/split_half/noise_ceiling/``
            under the repo, with ``geneset_<hash>__n_<N>__norm_<id>/`` subdirs.
            Independent of ``cache_dir`` and training ``workdir``.
        split_config: Split YAML relative to repo root unless absolute.
        files_glob: Glob pattern for matching data files.
        model_conditioning_variables: List of column names to condition on.
        metadata_joins: Explicit left-join steps applied before split /
            HBSS predicates. Each entry is a dict with ``left_on`` (obs
            column), ``parquet`` (metadata file), and ``columns`` (fields
            to bring in). See ``core_util_data/datasets/tahoe.yaml``.
        control_match_keys: Per-dataset list of obs columns that define the
            control-matching grouping used by the Δ-metric (e.g.
            ``["plate", "cell_name"]`` for Tahoe, ``["donor"]`` for Parse).
            Each name must be in ``model_conditioning_variables``.
        plot_aggregate_over: Conditioning variable to average metrics over in
            cross-condition plots (e.g. ``plate`` for Tahoe). ``null`` / omitted
            → each scored row is an independent plot point (full
            ``model_conditioning_variables`` in ``metrics_summary``).
        build_cache: Build mmap CSR cache when ``cache_dir`` is missing
            (see ``src.data.cache.raw_hvg_cache_build``).
        eval_cache_root: Root for eval NPZ namespaces (default ``<data_dir>/eval_cache``).
        eval_cache_dir: Optional full eval NPZ namespace
            (``.../normalization_<id>/geneset_<hash>__n_<N>``). Overrides
            ``eval_cache_root`` resolution when set.
        build_eval_cache: Build per-condition NPZ truth under ``eval_cache/``
            (see ``src.data.cache.eval_cache_build``). Preferred for eval scoring.
        build_noise_ceiling: After the eval cache is populated, run the
            split-half reference on it (see
            ``src.eval.reference.split_half``). Results are written under
            ``results/<dataset>/split_half/noise_ceiling/`` (or ``noise_ceiling_dir``)
            in model-independent subdirs keyed by gene set, normalization, split
            YAML, etc. — shared across training runs.
        noise_ceiling_n_repeats: Split-half repeats per condition for the
            reference (default 20).
        overwrite_cache: Force rebuild of mmap CSR shards.
        overwrite_noise_ceiling: Force rebuild of an existing
            ``noise_ceiling.json`` even if present on disk.
    """

    data_dir: Optional[str] = None
    hvg_genes_path: Optional[str] = None
    cache_dir: Optional[str] = None
    eval_cache_root: Optional[str] = None
    eval_cache_dir: Optional[str] = None
    noise_ceiling_dir: Optional[str] = None
    split_config: Optional[str] = None
    files_glob: Optional[str] = None
    model_conditioning_variables: Optional[List[str]] = None
    control_condition: Optional[Dict[str, Any]] = None
    metadata_joins: Optional[List[Any]] = None
    control_match_keys: Optional[List[str]] = None
    plot_aggregate_over: Optional[str] = None
    build_cache: Optional[bool] = None
    build_eval_cache: Optional[bool] = None
    build_noise_ceiling: Optional[bool] = None
    noise_ceiling_n_repeats: Optional[int] = None
    overwrite_cache: Optional[bool] = None
    overwrite_eval_cache: Optional[bool] = None
    overwrite_noise_ceiling: Optional[bool] = None
    registry_key: Optional[str] = None  # hydrated from core_util_data/datasets/*.yaml


@dataclass
class PertGenConfig:
    """Full configuration for the PertGen perturbation-response model.

    Architecture: condition embedding → condition MLP → gene outputs.

    Data-dependent fields (num_output_genes, n_drugs, n_cell_names, n_doses)
    are set by main after DataModule.setup() where needed.
    """

    # --- Model architecture ---
    # Condition MLP: flatten [cell, drug, dose, optional plate, noise] and map
    # that vector through Linear→GELU blocks and a ReLU head to gene outputs.
    # One dense vector of width ``condition_mlp_tokdim`` per modality.
    backbone_type: str = "condition_mlp"  # the only backbone; kept so other values raise at load
    condition_mlp_hidden_dims: List[int] = field(default_factory=lambda: [2048, 3072])
    condition_mlp_tokdim: int = 256

    # --- Data-driven sizes (set by DataModule.setup) ---
    # num_output_genes: HVG list length with hvg_genes_path, else all genes in the dataset.
    num_output_genes: int = 0
    n_categories: Dict[str, int] = field(default_factory=dict)

    # --- Output / loss mode ---
    n_repeats: int = 16
    variogram_loss_weight: float = 0.0  # Weight for variogram score term (0 = disabled).
    variogram_p: float = 0.5  # Power parameter for variogram score.
    variogram_n_pairs: int = 2000  # Gene pairs per step (permutation-matched, scatter-free backward).
    variogram_batch_subsample: int = 0  # Score the variogram on this many conditions per step (0 = all). Unbiased MC estimate of the same mean; the dominant cost is the gather backward, which scales linearly in B.
    variogram_repeat_subsample: int = 0  # Score the variogram on this many repeats (0 = all). Sound because the split-half estimator is debiased; the cheapest way to buy a larger n_pairs.
    # --- set-level RBF kernel score (ED Fig. 9a ablation) ---
    # When True, training draws ONE pool of `n_repeats` samples per distinct
    # condition in the batch (not per cell) and scores it against all that
    # condition's cells. Forward passes become C * n_repeats and stop depending on
    # cells-per-condition. `n_repeats` is therefore read as samples-per-CONDITION (R').
    # Requires train_loss=rbf_kernel_score and cell_set_budget_from_train_indices.
    group_by_condition: bool = False

    # --- Optimiser ---
    lr: float = 5e-4
    lr_init: float = 0.0  # Absolute LR at step 0; the warmup ramps linearly from here to `lr`.
    weight_decay: float = 0.05
    warmup_steps_min: int = 500  # Minimum warmup steps (floor); actual warmup = max(this, 5% of total steps).
    warmup_steps_max: Optional[int] = None  # Optional ceiling on warmup steps; actual warmup = min(max(warmup_steps_min, 5% of total), this). Cap takes precedence over floor if max < min. Warmup ramps from lr_init, so a short ramp from 0 (< 20 steps) may be unstable.
    # Fraction of *all* steps treated as WSD "stable" (Hu et al.). Decay occupies
    # whatever remains after warmup+stable. 1.0 → no WSD decay.
    wsd_stable_frac: float = 0.85
    # Per-epoch peak-LR steps on top of WSD: each milestone's ``lr_mul`` holds from its
    # epoch until the next (published runs: x1 -> x0.1 -> x0.01). The milestones'
    # n_repeats / batch_size must equal the top-level values (checked at load).
    # Unset -> x1 throughout.
    between_epoch_curriculum_schedule: Optional[List[Any]] = None
    # Hydrated by src.main before pl.Trainer(...) — kept on cfg so lr_lambda is
    # pure arithmetic and does not trigger trainer.estimated_stepping_batches
    # (which would build the DataLoader during configure_optimizers).
    total_training_steps: int = 0
    max_epochs: int = 100
    grad_clip: float = 0.0  # Keep 0 with fused AdamW (AMP gradient clipping incompatible).
    batch_size: int = 64
    accumulate_grad_batches: int = 1  # Accumulate N micro-batches before optimizer step; amortises optimizer overhead.

    # --- Precision / memory ---
    precision: str = "bf16-mixed"
    checkpoint_data_fraction: float = 0.1  # Save a checkpoint every this fraction of an epoch by data volume (e.g. 0.1 = every 10%). Batch-size-invariant.
    compile_model: bool = False  # torch.compile for speed (PyTorch 2.0+); off by default.
    compile_mode: str = "max-autotune"  # "default" (triton fusion), "reduce-overhead", or "max-autotune" (uses CUDA graphs — incompatible with accumulate_grad_batches > 1).

    # --- Eval callback ---
    # When True, :class:`src.training.callbacks.ModelEvalCallback` runs the
    # streaming model-eval pipeline (``src.eval.orchestration.model_eval``)
    # after selected training epochs and logs headline metrics to wandb.
    use_eval_callback: bool = True
    # Run model eval every N completed training epochs (1 = every epoch).
    eval_callback_every_n_epochs: int = 1
    # When True, ``max_epochs`` in YAML is the hbss100% budget; lower hbss
    # fractions run longer so total train-cell exposure stays ~matched.
    hbss_equalize_training_budget: bool = False
    # If set, spread eval callbacks across the (possibly scaled) run instead
    # of firing every ``eval_callback_every_n_epochs`` when the budget grows.
    eval_callback_points: Optional[int] = None
    # Unset → ``<data_dir>/hvg_cache/geneset_*``, ``noise_ceiling_results/``, etc.
    cache_dir: Optional[str] = None
    eval_cache_root: Optional[str] = None
    eval_cache_dir: Optional[str] = None
    noise_ceiling_dir: Optional[str] = None
    workdir: Optional[str] = None
    build_cache: bool = False
    build_eval_cache: bool = True
    # After ``trainer.fit``, score noise ceiling on the eval truth cache when True.
    build_noise_ceiling: bool = False
    noise_ceiling_n_repeats: int = 20
    overwrite_cache: bool = False
    overwrite_eval_cache: bool = False
    overwrite_noise_ceiling: bool = False
    # After ``trainer.fit``, run full model eval (forward + metrics) on these
    # splits when the eval truth cache exists (see ``src.eval.model_eval``).
    run_model_eval_after_fit: bool = True
    plot_model_vs_ceiling_after_fit: bool = True
    model_eval_splits: Optional[List[str]] = None  # eval callback; default validation+test
    model_eval_after_fit_splits: Optional[List[str]] = None  # post-fit; default validation+test
    overwrite_model_eval: bool = False
    model_eval_batch_size: int = 16384
    # How many per-condition GPU prediction buffers are allowed simultaneously
    # in the streaming pipeline.  3 is enough for gen + scoring overlap with
    # one condition in-flight on each stream plus a prefetched one.  Bump only
    # if you observe metric-stream stalls in profiling.
    model_eval_pipeline_depth: int = 3
    # Truth/ctrl prefetcher threads.  mmap reads are cheap so 1 is usually
    # plenty; >1 only helps when many small conditions have to be loaded.
    model_eval_metrics_workers: int = 1
    model_eval_device: str = "cuda"
    model_eval_compile: bool = False
    # When True, each completed condition's rep-major GPU buffer is also
    # async-D2H'd and written to ``<out>/predictions/<cond>.npy`` so it can
    # be reloaded for downstream analyses.  Off by default to keep model eval
    # ephemeral and fast.
    model_eval_persist_predictions: bool = False
    # K split-half repeats per condition in model eval. Decoupled from
    # ``noise_ceiling_n_repeats`` so model eval can use a small K (default 5)
    # while the reference stays at a larger K.
    model_eval_n_repeats: int = 5
    # Named key set from configs/eval/metrics.yaml, for model eval and the reference.
    # ``core`` = the three headline metrics; ``extended`` = every key (the published files).
    eval_metric_set: str = "core"
    # Per-gene arrays (means, DEG detail, dropout) in a parquet next to the summary; only
    # the per-gene figures read them, and they cost a second pass over every condition.
    eval_gene_sidecars: bool = False
    eval_min_cells: int = 4

    # --- Control matching (hydrated from the active dataset preset) ---
    control_match_keys: Optional[List[str]] = None
    # Cross-condition plots: nanmean metrics over this axis (hydrated from preset).
    # null → plot every scored row; e.g. ``plate`` for Tahoe.
    plot_aggregate_over: Optional[str] = None

    # --- Early stopping ---
    use_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_tolerance: float = 0.0001
    early_stopping_mode: str = "min"
    early_stopping_monitor: str = "val/loss"

    # --- Data ---
    # Optional preset selector matching a nested YAML block name (see ``DatasetPaths``).
    # Example YAML: ``tahoe: { data_dir: ... }`` plus ``dataset: tahoe``.
    # When False, training ignores any on-disk mmap CSR cache and reads h5ad files
    # directly.  The cache itself is never deleted or overwritten; it is simply skipped.
    use_train_cache: bool = True
    dataset: Optional[str] = None
    tahoe: Optional[DatasetPaths] = None
    parse: Optional[DatasetPaths] = None
    data_dir: Optional[str] = None
    # Train mmap cache (manifest.json + shard_###/). When set, DataModule reads CSR
    # HVG subsets instead of h5ads.
    hvg_genes_path: Optional[str] = None  # Hydrated from active dataset preset (``tahoe`` / ``parse``).
    # GENESIS trains in log1p: the loader yields raw counts and the training module
    # takes log1p on the GPU. Must be true; the training module raises otherwise.
    apply_log1p: bool = False
    num_workers: int = 8
    fetch_factor: int = 128
    # DataLoader prefetch_factor when num_workers > 0. Independent of
    # fetch_factor: ``fetch_factor + 1`` over-buffers workers and amplifies RSS
    # thrash on huge CSR shards. Default 2.
    prefetch_factor: int = 2
    block_size: int = 8
    # Train loader mode:
    #   "per_cell" — default scDataset + BlockShuffling (one cell per row).
    #   "cell_set" — C conditions × m_c cells from eval_cache (``dm.eval_manifest``).
    #     Sets are ragged; batches carry ``cell_counts`` instead of per-row codes.
    #     Requires group_by_condition=true.
    train_loader_mode: str = "per_cell"
    # UPPER BOUND on cells per condition, never a target. A condition with fewer
    # cells contributes all of them; short pools are NOT resampled to fill the
    # cap, because duplicated rows would claim evidence term 1 does not have.
    use_cell_set: int = 256
    # Minimum cells per train condition (from condition summary + split filter).
    cell_set_min_cells: int = 1
    # Restrict each cell_set unit to the split's TRAIN rows: a fixed, seeded subset of exactly
    # its train-row count (e.g. under a cell budget), and weight its loss term by that count.
    # Units with no train rows are dropped. Without it, units read every cell in the eval cache.
    cell_set_budget_from_train_indices: bool = False
    # Verbose stdout logging for cell_set loader (condition keys, shapes, sampling).
    cell_set_debug: bool = False
    # Train objective:
    #   "energy_score" — per-cell energy score (default; needs n_repeats > 1).
    #   "mse" — per-cell MSE of the condition mean (n_repeats=1, condition noise
    #           zeroed so the model is exactly deterministic).
    #   "rbf_kernel_score" — the energy score with its distance kernel swapped for a
    #           multi-scale RBF: per-cell MMD in expectation (rbf_bandwidth_* below).
    train_loss: str = "energy_score"
    # train_loss=rbf_kernel_score: the RBF bandwidths are the median distance between
    # training cells (log1p) times these scales. The median is estimated once at fit start
    # from rbf_bandwidth_n_rows cells and saved to <run>/rbf_bandwidth.json, unless set here.
    rbf_bandwidth_scales: List[float] = field(default_factory=lambda: [0.25, 0.5, 1.0, 2.0, 4.0])
    rbf_bandwidth_median: Optional[float] = None
    rbf_bandwidth_n_rows: int = 4096

    # --- Splits ---
    # Path to a unified split YAML (see core_util_data/splits/ and
    # src/data/splits/data_class.py). The YAML names a dataset registry entry under
    # core_util_data/datasets/<name>.yaml, declares predicate-based test /
    # validation blocks, and chooses the train sampling strategy
    # (`sample: all` or `sample: hbss` + `fraction`). Required at runtime;
    # paths resolve under the repo root.
    split_config: Optional[str] = None
    save_split_coverage: bool = True  # Write eval/split_coverage.png + f_min JSON during dm.setup().
    split_coverage_max_pert_rows: Optional[int] = None  # Optional cap for split_coverage_gene_labels.png; null = index plot only.
    raise_on_axis_coverage_failure: bool = True  # Block setup if marginal axis coverage or holdout leakage fails.
    files_glob: Optional[str] = None
    model_conditioning_variables: List[str] = field(default_factory=list)
    # Subset of ``model_conditioning_variables`` the generator actually embeds.
    # Eval / cache / metrics stay on the full MCV identity. Unset or [] → all MCV.
    condition_input_variables: Optional[List[str]] = None
    control_condition: Optional[Dict[str, Any]] = None
    metadata_joins: Optional[List[Any]] = None
    seed: int = 42

    # --- Device ---
    # ``devices`` 1 or "auto" is the one training device. ``strategy`` stays "auto".
    strategy: str = "auto"
    devices: Union[int, str] = 1
    accelerator: str = "auto"  # "auto", "gpu", "cpu".
    num_nodes: int = 1

    # --- Profiler / smoke runs ---
    use_profiler: bool = False  # Enable PyTorchProfiler; traces in workdir/perf_logs (Chrome trace).
    profiler_steps: int = 200  # Active (recorded) profiler steps per cycle.
    profiler_warmup_steps: int = 1  # Warmup steps before active profiling (discarded from trace).
    profiler_skip_steps: int = 10  # Steps to skip before the profiler schedule starts.
    limit_train_batches: Optional[Union[int, float]] = None  # If set, run only this many train batches per epoch (e.g. 100 for profiling).
    # --- Logging ---
    use_wandb: bool = True
    log_every_n_steps: int = 50
    # Warmstart the final Linear's bias so predictions at step 0 match the
    # marginal per-gene log1p mean, killing the ~E[||Y||] "predict-zero" loss
    # floor. μ ≥ 0, so bias = μ gives ReLU(bias) = μ exactly. Every published run sets it.
    head_bias_init_from_train_mean: bool = False
    head_bias_init_n_batches: int = 100  # Batches of the train dataloader consumed for the running mean.
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    run_name_suffix: Optional[str] = None  # Appended to auto-generated W&B run name (see utils._build_run_name).
    runs_split_slug: Optional[str] = None  # Override runs/<dataset>/<split>/… split segment (e.g. graded_holdout/ofat).

    # --- Resume / weights ---
    resume_ckpt: Optional[str] = None  # Full Lightning resume (epoch, optimizer, weights).
    # Weights only: load model.state_dict, fresh optimizer/epoch. Mutually exclusive with resume_ckpt.
    weights_ckpt: Optional[str] = None


# When adding another ``Optional[DatasetPaths]`` sibling, extend this set (used by
# :func:`sanitize_config_mapping`).
_DATASET_PATH_PRESETS = frozenset({"tahoe", "parse"})

# Options with one behaviour, built into the model. Every YAML and checkpoint
# carries them, so they are dropped like any stale key,
# but only at this value: anything else asks for a model this code cannot build,
# and since the state-dict keys match it would otherwise load and run silently.
_HARDWIRED = {
    "condition_mlp_norm": "none",
    "condition_mlp_activation": "gelu",
    "condition_mlp_dropout": 0.0,
    "condition_mlp_residual": False,
    "output_activation": "relu",
    "predict_delta": False,
    "variogram_fast": True,
    "variogram_debias": True,
    "variogram_balanced_zeros": False,
    "variogram_mean_weight": False,
    "total_log1p_mass_weight": 0.0,
    "raw_mass_huber_weight": 0.0,
    "output_l1_weight": 0.0,
    "output_l2_weight": 0.0,
    "cpm_energy_loss_weight": 0.0,
    "per_gene_crps_weight": 0.0,
    "mmd_sigma": None,
    "apply_zscore": False,
    "energy_zscore": False,
    "use_pca_energy_loss": False,
    "noise_distribution": "uniform",
    "target_transform": "auto",
}


def sanitize_config_mapping(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Turn a dumped ``config`` dict into ``PertGenConfig(**…)`` kwargs.

    Stale checkpoint keys removed from ``PertGenConfig`` are dropped, except a
    hardwired option set to another value, which raises. Nested presets
    (``tahoe`` / ``parse``) are reconstructed from plain ``dict`` when needed.
    """
    valid = {f.name for f in fields(PertGenConfig)}
    dp_keys = {f.name for f in fields(DatasetPaths)}
    out: Dict[str, Any] = {}
    for key, val in raw.items():
        if key in _HARDWIRED and val != _HARDWIRED[key]:
            raise ValueError(f"{key}={val!r} is not supported; this code always uses {_HARDWIRED[key]!r}")
        if key not in valid:
            continue
        if key in _DATASET_PATH_PRESETS:
            if val is None or isinstance(val, DatasetPaths):
                out[key] = val
            elif isinstance(val, dict):
                sub = {nk: nv for nk, nv in val.items() if nk in dp_keys}
                out[key] = DatasetPaths(**sub) if sub else DatasetPaths()
            else:
                out[key] = None
        else:
            out[key] = val
    return out


def resolve_condition_input_variables(config: PertGenConfig) -> List[str]:
    """Labels the backbone embeds. Unset/empty → all ``model_conditioning_variables``.

    Eval-cache keys, work-queue rows, and metrics stay on the full MCV. This
    list only drops modalities from the generator. A var not in MCV is a
    config error (would silently change the scored identity).
    """
    mcv = list(getattr(config, "model_conditioning_variables", None) or [])
    raw = getattr(config, "condition_input_variables", None)
    if not raw:
        return mcv
    inputs = [str(v) for v in raw]
    if len(inputs) != len(set(inputs)):
        raise ValueError(f"condition_input_variables has duplicates: {inputs}")
    missing = [v for v in inputs if v not in mcv]
    if missing:
        raise ValueError(
            f"condition_input_variables {missing} not in model_conditioning_variables "
            f"{mcv}. Keep MCV as the full eval identity; this field only drops "
            "labels from the generator input."
        )
    return inputs


def epoch_lr_milestones(config: PertGenConfig) -> List[Tuple[int, float]]:
    """Sorted ``(epoch, lr_mul)`` from ``between_epoch_curriculum_schedule``; ``[(0, 1.0)]`` when unset.

    A milestone that still pins ``n_repeats`` or ``batch_size`` must match the top-level
    value: those per-epoch changes are gone, and ignoring one would silently train a
    different model than the YAML describes.
    """
    sched = config.between_epoch_curriculum_schedule
    if not sched:
        return [(0, 1.0)]
    out = []
    for i, m in enumerate(sched):
        where = f"between_epoch_curriculum_schedule[{i}]"
        if not isinstance(m, Mapping) or "epoch" not in m or "lr_mul" not in m:
            raise ValueError(f"{where} needs 'epoch' and 'lr_mul', got {m!r}")
        if extra := set(m) - {"epoch", "lr_mul", "n_repeats", "batch_size"}:
            raise ValueError(f"{where}: unknown keys {sorted(extra)}")
        for k in ("n_repeats", "batch_size"):
            if k in m and int(m[k]) != int(getattr(config, k)):
                raise ValueError(
                    f"{where}.{k}={m[k]} but {k}={getattr(config, k)}: per-epoch {k} changes are not supported"
                )
        epoch, mul = int(m["epoch"]), float(m["lr_mul"])
        if epoch < 0 or not (math.isfinite(mul) and mul > 0):
            raise ValueError(f"{where}: need epoch >= 0 and a finite lr_mul > 0, got {dict(m)}")
        out.append((epoch, mul))
    out.sort()
    epochs = [e for e, _ in out]
    if epochs[0] != 0 or len(set(epochs)) != len(epochs):
        raise ValueError(f"between_epoch_curriculum_schedule needs one milestone at epoch 0 and unique epochs, got {epochs}")
    return out


def _clean_path(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, str) and not val.strip():
        return None
    return str(val)


def blank_cache_paths_to_none(cfg: PertGenConfig) -> None:
    """Treat empty YAML strings as unset for cache path fields."""
    cfg.cache_dir = _clean_path(getattr(cfg, "cache_dir", None))
    cfg.eval_cache_root = _clean_path(getattr(cfg, "eval_cache_root", None))
    cfg.eval_cache_dir = _clean_path(getattr(cfg, "eval_cache_dir", None))
    cfg.noise_ceiling_dir = _clean_path(getattr(cfg, "noise_ceiling_dir", None))
    cfg.workdir = _clean_path(getattr(cfg, "workdir", None))


def migrate_legacy_cache_keys(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Map deprecated train/eval cache YAML keys to unified ``cache_dir`` fields."""
    out = dict(raw)
    alias_pairs = (
        ("train_cache_dir", "cache_dir"),
        ("build_train_cache", "build_cache"),
        ("overwrite_train_cache", "overwrite_cache"),
    )
    for old, new in alias_pairs:
        if old in out and out.get(new) is None:
            out[new] = out[old]
        out.pop(old, None)
    for preset in _DATASET_PATH_PRESETS:
        block = out.get(preset)
        if isinstance(block, dict):
            out[preset] = migrate_legacy_cache_keys(block)
    return out


_DEFAULT_H5AD_GLOB = "*.h5ad"


def resolve_files_glob(config: PertGenConfig) -> str:
    """Glob for plate/batch files; default h5ad pattern when unset."""
    glob = getattr(config, "files_glob", None)
    if glob:
        return str(glob)
    return _DEFAULT_H5AD_GLOB


def resolve_cache_dir(config: PertGenConfig) -> Optional[Path]:
    """Default ``<data_dir>/hvg_cache/geneset_<hash>__n_<N>`` when ``data_dir`` is set."""
    from pathlib import Path

    raw = _clean_path(getattr(config, "cache_dir", None))
    if raw:
        return _under_data_dir(config, "cache_dir", gene_set_cache_dirname(
            gene_names_from_hvg_config(config),
        ))
    if not config.data_dir:
        return None
    leaf = gene_set_cache_dirname(gene_names_from_hvg_config(config))
    return (Path(config.data_dir).expanduser() / "hvg_cache" / leaf).resolve()


def resolve_eval_cache_root(config: PertGenConfig) -> Path:
    """Root directory for per-condition eval NPZ caches (``<data_dir>/eval_cache``)."""
    from pathlib import Path

    raw = _clean_path(getattr(config, "eval_cache_root", None))
    if raw:
        p = Path(raw).expanduser()
        if p.is_absolute():
            return p.resolve()
        if not config.data_dir:
            raise ValueError("eval_cache_root is relative but data_dir is None.")
        return (Path(config.data_dir).expanduser() / p).resolve()
    if not config.data_dir:
        raise ValueError("data_dir required for eval cache root.")
    return (Path(config.data_dir).expanduser() / "eval_cache").resolve()


def resolve_eval_cache_dir(config: PertGenConfig, *, normalization_id: str = "raw_counts") -> Path:
    """Per-condition NPZ eval truth namespace under ``eval_cache/``.

    Keep ``eval_cache_dir`` only if its basename hash matches ``hvg_genes_path``;
    otherwise derive from the training gene set (mismatched overrides lose).
    """
    from pathlib import Path

    from src.data.data_utils import parse_geneset_dirname, resolve_cache_namespace
    from src.eval.cache.eval_cache_io import gene_set_hash

    gene_names = gene_names_from_hvg_config(config)
    gsh, n_genes = gene_set_hash(gene_names), len(gene_names)

    raw_ns = _clean_path(getattr(config, "eval_cache_dir", None))
    if raw_ns:
        p = Path(raw_ns).expanduser()
        if p.is_absolute():
            resolved = p.resolve()
        else:
            if not config.data_dir:
                raise ValueError("eval_cache_dir is relative but data_dir is None.")
            resolved = (Path(config.data_dir).expanduser() / p).resolve()
        parsed = parse_geneset_dirname(resolved.name)
        if parsed is not None and parsed[0] == gsh:
            return resolved
        import logging
        logging.getLogger(__name__).warning(
            "Ignoring eval_cache_dir=%s; using geneset_%s__n_%d from hvg_genes_path",
            resolved, gsh, n_genes,
        )

    return resolve_cache_namespace(
        resolve_eval_cache_root(config),
        normalization_id=normalization_id,
        gene_set_hash_value=gsh,
        n_genes=n_genes,
        create=False,
    )


def gene_names_from_hvg_config(config: PertGenConfig) -> List[str]:
    """Gene list from ``hvg_genes_path`` (for cache path naming before ``dm.setup()``)."""
    from src.data.data_utils import load_hvg_gene_names_from_file

    path = _clean_path(getattr(config, "hvg_genes_path", None))
    if not path:
        raise ValueError("hvg_genes_path required for gene-set cache paths.")
    return load_hvg_gene_names_from_file(path)


def gene_set_cache_dirname(gene_names: Sequence[str]) -> str:
    """Same leaf as eval truth cache: ``geneset_<hash>__n_<N>``."""
    from src.eval.cache.eval_cache_io import gene_set_hash

    return f"geneset_{gene_set_hash(gene_names)}__n_{len(gene_names)}"


def _under_data_dir(config: PertGenConfig, attr: str, default: str) -> Path:
    """Unset or relative path → ``<data_dir>/<default>`` or ``<data_dir>/<path>``."""
    from pathlib import Path

    raw = _clean_path(getattr(config, attr, None))
    base = Path(config.data_dir).expanduser() if config.data_dir else None
    if not raw:
        if base is None:
            raise ValueError(f"{attr} unset and data_dir is None.")
        return (base / default).resolve()
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    if base is None:
        return (Path.cwd() / p).resolve()
    return (base / p).resolve()


def resolve_noise_ceiling_dir(config: PertGenConfig) -> Path:
    """Noise-ceiling root (shared across model training runs).

    Default: ``results/<dataset>/split_half/noise_ceiling/`` under the repo.
    Override with ``noise_ceiling_dir`` (absolute or relative to repo root).
    """
    from pathlib import Path

    from src.paths import PERTGEN_REPO_ROOT, noise_ceiling_results_root

    raw = _clean_path(config.noise_ceiling_dir)
    if raw:
        p = Path(raw).expanduser()
        if p.is_absolute():
            return p.resolve()
        return (PERTGEN_REPO_ROOT / p).resolve()
    return noise_ceiling_results_root(config.dataset)


def _sanitize_run_path_token(raw: str) -> str:
    """Filesystem-safe path segment (keep ``/`` for nested split slugs)."""
    import re

    text = str(raw).strip()
    if not text:
        return "_empty"
    parts = []
    for part in text.replace("\\", "/").split("/"):
        token = re.sub(r"[^A-Za-z0-9._+-]+", "_", part.strip())
        parts.append(token or "_")
    return "/".join(parts)


def _resolve_split_config_path(split_config: str | Path | None, *, repo_root: Path) -> Path | None:
    from pathlib import Path

    raw = _clean_path(split_config)
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve() if path.is_file() else path.resolve()
    candidate = (repo_root / path).resolve()
    if candidate.is_file():
        return candidate
    return path.resolve()


def _infer_dataset_from_paths(config: PertGenConfig) -> str | None:
    """Guess dataset from legacy YAML paths when ``dataset`` was never written."""
    markers = (
        ("tahoe", "tahoe"),
        ("parse", "parse"),
        ("pbmc", "pbmc"),
    )
    raw = _clean_path(getattr(config, "data_dir", None))
    if not raw:
        return None
    lower = raw.lower()
    for needle, dataset in markers:
        if needle in lower:
            return dataset
    return None


def _infer_split_slug_from_legacy_fields(config: PertGenConfig) -> str | None:
    """Guess split folder for pre-``split_config`` resolved YAML."""
    split_type = _clean_path(getattr(config, "split_type", None))
    if split_type == "perturbation_holdout":
        return "legacy_perturbation_holdout"
    run_name = _clean_path(getattr(config, "wandb_run_name", None)) or ""
    if "perthold" in run_name:
        return "legacy_perturbation_holdout"
    if "graded" in run_name:
        return "graded_holdout/seed0"
    return None


def dataset_slug_from_config(config: PertGenConfig, *, repo_root: Path | None = None) -> str:
    """Top-level runs folder: ``config.dataset``, else infer from ``split_config``."""
    from pathlib import Path

    from src.paths import PERTGEN_REPO_ROOT

    repo = Path(repo_root).resolve() if repo_root is not None else PERTGEN_REPO_ROOT
    dataset = _clean_path(getattr(config, "dataset", None))
    if dataset:
        return _sanitize_run_path_token(dataset)

    split_path = _resolve_split_config_path(getattr(config, "split_config", None), repo_root=repo)
    if split_path is not None:
        parts = split_path.parts
        if "graded_holdout" in parts:
            idx = parts.index("graded_holdout")
            if idx + 1 < len(parts):
                return _sanitize_run_path_token(parts[idx + 1])
        if "heatmap" in parts:
            idx = parts.index("heatmap")
            if idx + 1 < len(parts):
                return _sanitize_run_path_token(parts[idx + 1])

    legacy = _infer_dataset_from_paths(config)
    if legacy:
        return _sanitize_run_path_token(legacy)
    return "_unknown"


def split_slug_from_config(config: PertGenConfig, *, repo_root: Path | None = None) -> str:
    """Second-level runs folder derived from ``split_config`` path."""
    import re
    from pathlib import Path

    from src.paths import PERTGEN_REPO_ROOT

    repo = Path(repo_root).resolve() if repo_root is not None else PERTGEN_REPO_ROOT
    override = _clean_path(getattr(config, "runs_split_slug", None))
    if override:
        return _sanitize_run_path_token(override)

    split_path = _resolve_split_config_path(getattr(config, "split_config", None), repo_root=repo)
    if split_path is None:
        legacy = _infer_split_slug_from_legacy_fields(config)
        if legacy:
            return _sanitize_run_path_token(legacy)
        return "_no_split"

    posix = split_path.as_posix()
    graded = re.search(r"graded_holdout/[^/]+/seed(\d+)/([^/]+)$", posix)
    if graded:
        seed = graded.group(1)
        stem = Path(graded.group(2)).stem
        base = f"graded_holdout/seed{seed}"
        dataset = _clean_path(getattr(config, "dataset", None)) or ""
        default_stem = f"{dataset}_graded_holdout" if dataset else "graded_holdout"
        if stem in {default_stem, "graded_holdout"}:
            return _sanitize_run_path_token(base)
        return _sanitize_run_path_token(f"{base}/{stem}")

    parts = split_path.parts
    if "splits" in parts:
        idx = parts.index("splits")
        rel = parts[idx + 1 : -1]
        if rel:
            return _sanitize_run_path_token("/".join(rel) + "/" + split_path.stem)
    return _sanitize_run_path_token(split_path.stem)


def model_slug_from_config(config: PertGenConfig) -> str:
    """Third-level runs folder: explicit ``run_name_suffix``, else backbone type."""
    suffix = _clean_path(getattr(config, "run_name_suffix", None))
    if suffix:
        return _sanitize_run_path_token(suffix)
    backbone = _clean_path(getattr(config, "backbone_type", None))
    if backbone:
        return _sanitize_run_path_token(backbone)
    return "_auto"


def resolve_run_group_path(
    config: PertGenConfig,
    *,
    project_root: str | Path | None = None,
) -> Path:
    """``<dataset>/<split>/<model>`` under ``runs/`` (each segment sanitized)."""
    from pathlib import Path

    from src.paths import PERTGEN_REPO_ROOT

    repo = Path(project_root).resolve() if project_root is not None else PERTGEN_REPO_ROOT
    dataset = dataset_slug_from_config(config, repo_root=repo)
    split = split_slug_from_config(config, repo_root=repo)
    model = model_slug_from_config(config)
    return Path(dataset) / Path(split) / Path(model)


def run_location_from_config_path(
    config: PertGenConfig,
    config_path: str | Path | None,
    *,
    repo_root: Path,
) -> tuple[str, str] | None:
    """``(dataset, split slug)`` read off a YAML living at ``configs/<dataset>/<role path>/x.yaml``.

    The configs/ tree IS the runs/ tree, so a new run lands where its config sits and the two
    cannot drift. Only a first folder that names a registered dataset (core_util_data/datasets/*.yaml)
    opts in; every other location returns None and keeps the legacy slug logic.
    """
    from pathlib import Path

    if config_path is None:
        return None
    configs_root = (repo_root / "configs").resolve()
    try:
        rel = Path(config_path).resolve().relative_to(configs_root)
    except ValueError:
        return None
    folders = rel.parts[:-1]
    datasets = {p.stem for p in (repo_root / "core_util_data" / "datasets").glob("*.yaml")}
    if not folders or folders[0] not in datasets:
        return None

    dataset, where = folders[0], f"configs/{rel.as_posix()}"
    if len(folders) < 2:
        raise ValueError(f"{where}: put the config in a split/role folder, e.g. configs/{dataset}/bio/main/")
    declared = _clean_path(getattr(config, "dataset", None))
    if declared and declared != dataset:
        raise ValueError(f"{where}: dataset={declared!r} but the config sits under configs/{dataset}/")
    slug = _sanitize_run_path_token("/".join(folders[1:]))
    explicit = _clean_path(getattr(config, "runs_split_slug", None))
    # A slug left over from before a move must not silently send the run somewhere else.
    if explicit and _sanitize_run_path_token(explicit) != slug:
        raise ValueError(f"{where}: runs_split_slug={explicit!r} disagrees with its location {slug!r}; drop the key")
    return dataset, slug


def resolve_workdir(
    config: PertGenConfig,
    *,
    run_name: str,
    fallback_runs_dir: str | Path = "runs",
    project_root: str | Path | None = None,
    config_path: str | Path | None = None,
) -> Path:
    """Default ``runs/<dataset>/<split>/<model>/<run_name>``; explicit ``workdir`` wins.

    A config inside the role tree (see ``run_location_from_config_path``) takes ``<dataset>/<split>``
    from its own location instead of from ``dataset`` / ``runs_split_slug``.
    """
    from pathlib import Path

    from src.paths import PERTGEN_REPO_ROOT

    repo = Path(project_root).resolve() if project_root is not None else PERTGEN_REPO_ROOT
    runs_parent = Path(fallback_runs_dir).expanduser()
    if not runs_parent.is_absolute():
        runs_parent = (repo / runs_parent).resolve()

    raw = _clean_path(getattr(config, "workdir", None))
    if raw:
        p = Path(raw).expanduser()
        if p.is_absolute():
            return p.resolve()
        return (repo / p).resolve()

    location = run_location_from_config_path(config, config_path, repo_root=repo)
    if location is not None:
        dataset, split = location
        return (runs_parent / dataset / split / model_slug_from_config(config) / run_name).resolve()
    group = resolve_run_group_path(config, project_root=repo)
    return (runs_parent / group / run_name).resolve()

