"""PertGen model evaluation: streaming forward + scoring on one GPU.

Used by :mod:`src.main` after training (when ``run_model_eval_after_fit`` is
set and the eval truth cache exists) and by
``scripts/eval/score.py genesis`` (thin CLI shim).

Unified K-rep protocol
----------------------
Every per-condition metric (pseudobulk arm, pseudobulk delta, distributional
``mmd_rbf`` / ``energy_distance``, DEG overlap) is computed K times with the
same :class:`~src.eval.reference.split_half.SplitHalfSampler` as the
split-half reference.  Each rep draws ``(truth_arm, _, ctrl_a, ctrl_b,
pred_cell_idx)`` from the sampler; model predictions replace the reference's
other truth half at ``pred_rep[pred_cell_idx]``, so both arms are half-``N``.
Reps are averaged to produce ``<metric>`` (mean) in ``metrics`` plus
``metrics_std`` companions and a single top-level ``k`` per condition.

Streaming pipeline
------------------
Predictions never leave the GPU.  The forward loop scatters batched outputs
directly into per-condition rep-major buffers (``(K * N_c, n_genes)``) on a
generation stream; as each condition's buffer fills, a ``cuda.Event`` is
recorded and the condition is handed off to the consumer.  Scoring runs on
its own metrics stream (scoring.GenesisSource), waits on the gen event, and frees the buffer
once the metric kernels return.  Throughput ≈ ``max(forward_time,
metrics_time)`` instead of ``forward_time + metrics_time``; RAM stays
constant in the number of conditions (no host-side ``pred_chunks`` dict).

Structure
---------
* Setup / IO — :func:`_resolve_eval_paths`, :func:`_check_existing_metrics`,
  :func:`_load_inference_bundle`, :func:`_load_work_queue_or_none`,
  :func:`_write_empty_split_summary`.
* K-rep plan & inflation — :func:`_make_model_eval_plan` + immutable
  :class:`ModelEvalPlan`, and :func:`_inflate_work_queue` returning an
  :class:`InflationStats` record.
* Per-condition scoring — :func:`_load_condition_tensors`,
  :func:`_score_condition_kreps` (one unified K-rep loop covering every
  metric), :func:`_compute_condition_extras` (returns :class:`CondExtras`),
  :func:`build_result_row`.
* Output — :func:`_genesis_fields`; the loop and the summary writer are
  :func:`src.eval.scoring.score_conditions` / ``write_summary`` (GENESIS is a ``GenesisSource``).

"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

from src.config.config import PertGenConfig
from src.data.data_utils import EvalManifest, resolve_eval_group_keys
from src.data.cache.eval_cache_build import EVAL_NPZ_MODE
from src.eval.cache.eval_cache_io import (
    ConditionPayload,
    ControlCache,
    QuadrupletRow,
    group_by_condition,
    resolve_eval_cache_mode,
    safe_filename_part,
)
from src.eval.metrics.deg import compute_deg_detail
from src.eval.metrics.eval_metrics import score_pair
from src.eval.metrics.reporting import (
    check_metric_keys,
    cluster_flat_metrics,
    load_metric_set,
    materialize_metric_tensor_batch,
)
from src.eval.metrics.gene_summary import compute_dropout_summary, compute_gene_summary
from src.eval.reference.split_half import SplitHalfSampler
from src.eval.plots.gene_arrays import (
    METRICS_GENE_ARRAYS_FILENAME,
)
from src.eval.orchestration.log_utils import (
    EvalStartupProfiler,
    ensure_eval_phase_logging,
    headline_metrics,
    log_detail,
    log_step,
)
from src.eval.orchestration.pregeneration import (
    StreamingPregenStats,
)
from src.model.inference_output import PertGenInferenceBundle, to_raw_integer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_EVAL_SPLITS: tuple[str, ...] = ("validation", "test")

# Headline metric keys logged after the K-rep loop for the first condition.
_KREP_HEADLINE_KEYS: tuple[str, ...] = (
    "log1p_slice_med_rgs/r2_diff",
    "log1p_slice_med_rgs/pearson_r_diff",
    "log1p_slice_med_rgs/spearman_r_diff",
    "raw_int_rgs/r2_diff",
    "raw_int_rgs/pearson_r_diff",
    "raw_int_rgs/spearman_r_diff",
    "raw_int_rgs/gene_det_rate_spearman",
    "log1p_slice_med_rgs/r2",
    "log1p_slice_med_rgs/energy_distance",
    "deg_raw_int_rgs/deg_overlap_top_p_ranked_k100_fdr0p001_lfc0p1_adaptivek_truth_detfloor10",
    "deg_raw_int_rgs/jaccard_sig_fdr0p001_lfc0p1_detfloor10",
)


# ===========================================================================
# Small pure helpers
# ===========================================================================


def _stable_cond_seed(base_seed: int, key_dict: Mapping[str, Any]) -> int:
    """Reproducible 31-bit seed for a condition.

    Python's built-in ``hash()`` of strings is salted per-process unless
    ``PYTHONHASHSEED=0``, so any seed derived from it is **not** stable
    across runs.  We hash a deterministic ``repr(sorted(items))`` with
    ``base_seed`` mixed in so changing ``--seed`` still produces different
    draws, but the same ``(seed, condition)`` always agrees.
    """
    payload = f"{int(base_seed)}|{sorted(key_dict.items())!r}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFF_FFFF


def _gib(n_bytes: float) -> float:
    return float(n_bytes) / (1024.0 ** 3)


def _gpu_mem_snapshot(dev: torch.device) -> tuple[float, float, float, float]:
    """``(free, total, alloc, reserved)`` in GiB for ``dev`` (CUDA only)."""
    free_b, total_b = torch.cuda.mem_get_info(dev)
    alloc_b = torch.cuda.memory_allocated(dev)
    reserved_b = torch.cuda.memory_reserved(dev)
    return _gib(free_b), _gib(total_b), _gib(alloc_b), _gib(reserved_b)


def _key_label(key_dict: Mapping[str, Any]) -> str:
    """Stable ``"k1=v1|k2=v2"`` label for log lines."""
    return "|".join(f"{k}={v}" for k, v in sorted(key_dict.items()))


def _maybe_empty_cuda_cache(dev: torch.device) -> None:
    """``torch.cuda.empty_cache()`` guarded on CUDA availability."""
    if torch.cuda.is_available() and dev.type == "cuda":
        torch.cuda.empty_cache()


# ===========================================================================
# Public dataclasses
# ===========================================================================


@dataclass(frozen=True)
class ModelEvalResult:
    """Paths and counts from one completed model-eval run."""

    eval_split: str
    out_dir: Path
    metrics_path: Path
    n_conditions: int
    n_skipped: int
    forward_seconds: float


# ===========================================================================
# Internal records (immutable, pure-data)
# ===========================================================================


@dataclass(frozen=True)
class ModelEvalPlan:
    """Resolved K-rep eval protocol for one model-eval call."""

    active: bool          # when False, skip K-rep inflation and eligibility gates
    k: int                # model_eval_n_repeats
    min_truth_cells: int = 4
    min_ctrl_cells: int = 4
    metrics: Optional[frozenset[str]] = None  # load_metric_set(...); None = every key
    gene_sidecars: bool = False               # per-gene parquet next to the summary
    # Ctrl detection-fraction floor for the calibrated DEG family (genes below it are
    # masked before DEG bookkeeping).  ``(0.10,)`` is the detection-fraction
    # calibrated default; the floor is part of the one DEG definition, so it is
    # always applied.
    ctrl_det_floors: tuple[float, ...] = (0.10,)


@dataclass(frozen=True)
class InflationStats:
    """Summary statistics for the K-rep work-queue inflation."""

    n_conditions: int
    n_enrolled: int
    n_skipped_small: int
    cells_orig: int
    cells_inflated: int
    per_cond_extra: tuple[int, ...]


@dataclass(frozen=True)
class KRepResult:
    """Unified per-key statistics from the K-rep loop for one condition.

    ``means[k]`` is the mean across reps; ``stds[k]`` is the population
    standard deviation across reps; ``n_reps_per_key[k]`` records how
    many reps had a finite value (NaN-skipped per key).
    """

    means: dict[str, float]
    stds: dict[str, float]
    n_reps_per_key: dict[str, int]
    k_effective: int
    elapsed_seconds: float


@dataclass(frozen=True)
class CondExtras:
    """Per-condition diagnostic bundles (gene-level, DEG, dropout)."""

    gene_summary: dict
    deg_detail: dict
    dropout_summary: dict


# ===========================================================================
# Setup / IO helpers
# ===========================================================================


def _checkpoint_stem(checkpoint: Path) -> str:
    """Filesystem-safe checkpoint basename (no ``.ckpt`` suffix)."""
    return checkpoint.stem.replace(" ", "_").replace("/", "-").replace("\\", "-")


def default_model_eval_out_dir(
    workdir: Path,
    checkpoint: Path,
    eval_split: str,
    *,
    config: PertGenConfig,
    results_subdir: Optional[str] = None,
) -> Path:
    """``results/<dataset>/split_half/pertgen/<run_name>__<ckpt_stem>__<split>`` under the repo."""
    from src.paths import pertgen_model_eval_results_root

    leaf = f"{Path(workdir).name}__{_checkpoint_stem(checkpoint)}__{eval_split}"
    root = pertgen_model_eval_results_root(config.dataset)
    if results_subdir:
        return (root / results_subdir / leaf).resolve()
    return (root / leaf).resolve()


def _resolve_eval_paths(
    run_dir: Path,
    out_dir: Optional[Path],
    checkpoint: Path,
    eval_split: str,
    config: PertGenConfig,
) -> tuple[Path, Path]:
    """Resolve ``(out_dir, metrics_path)`` and ensure ``out_dir`` exists."""
    out = (
        Path(out_dir).resolve() if out_dir is not None
        else default_model_eval_out_dir(run_dir, checkpoint, eval_split, config=config)
    )
    out.mkdir(parents=True, exist_ok=True)
    return out, out / "metrics_summary.json"


def _check_existing_metrics(
    metrics_path: Path,
    out_dir: Path,
    eval_split: str,
    overwrite: bool,
) -> Optional[ModelEvalResult]:
    """Return a cached :class:`ModelEvalResult` when metrics already exist."""
    if not metrics_path.is_file() or overwrite:
        return None
    log_step(
        logger, "skip",
        "existing metrics at %s (set overwrite_model_eval to rerun)", metrics_path,
    )
    with metrics_path.open() as f:
        existing = json.load(f)
    return ModelEvalResult(
        eval_split=eval_split,
        out_dir=out_dir,
        metrics_path=metrics_path,
        n_conditions=int(existing.get("n_conditions", 0)),
        n_skipped=0,
        forward_seconds=float(existing.get("forward_seconds", 0.0)),
    )


def _load_inference_bundle(
    run_dir: Path,
    *,
    checkpoint: Optional[Path],
    device: str,
    use_compile: bool,
) -> PertGenInferenceBundle:
    """Build the inference bundle for this run."""
    return PertGenInferenceBundle.from_run(
        run_dir,
        checkpoint=checkpoint,
        device=device,
        use_compile=use_compile,
    )


def _load_work_queue_or_none(
    *,
    manifest: EvalManifest,
    config: PertGenConfig,
    bundle: PertGenInferenceBundle,
    eval_split: str,
    min_cells: int,
    conditions_query: Optional[str] = None,
    max_conditions: Optional[int] = None,
    conditions_seed: int = 0,
    drop_control_rows: bool = True,
) -> Optional[list[QuadrupletRow]]:
    """Build the work queue; return ``None`` when the split has 0 conditions."""
    from ..cache.eval_cache_io import load_work_queue_or_none

    mcv = list(manifest.group_cols or config.model_conditioning_variables or [])
    if not mcv:
        raise ValueError("model_conditioning_variables empty; cannot build work queue.")
    split_yaml = manifest.split_config or config.split_config
    if not split_yaml:
        raise ValueError("split_config required for model eval work queue.")
    return load_work_queue_or_none(
        condition_summary_parquet=manifest.condition_summary,
        split_yaml=split_yaml,
        eval_split=eval_split,
        condition_variables=mcv,
        control_condition=dict(config.control_condition or {}),
        metadata_joins=list(config.metadata_joins or []),
        bundle=bundle,
        min_cells=min_cells,
        conditions_query=conditions_query,
        max_conditions=max_conditions,
        conditions_seed=conditions_seed,
        control_match_keys=list(config.control_match_keys or []) or None,
        drop_control_rows=drop_control_rows,
    )


def _write_empty_split_summary(
    metrics_path: Path,
    out_dir: Path,
    eval_split: str,
) -> ModelEvalResult:
    """Persist a marker summary and return the corresponding result."""
    summary = {
        "eval_split": eval_split,
        "n_conditions": 0,
        "skipped_empty_split": True,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    metrics_path.write_text(json.dumps(summary, indent=2))
    return ModelEvalResult(
        eval_split=eval_split,
        out_dir=out_dir,
        metrics_path=metrics_path,
        n_conditions=0,
        n_skipped=0,
        forward_seconds=0.0,
    )


# ===========================================================================
# K-rep eval plan + inflation (pure)
# ===========================================================================


def _make_model_eval_plan(
    config: PertGenConfig,
    n_conditions: int,
    *,
    ctrl_det_floors: tuple[float, ...] = (0.10,),
) -> ModelEvalPlan:
    """Resolve K for one model-eval run."""
    plan = ModelEvalPlan(
        active=True,
        k=config.model_eval_n_repeats,
        min_truth_cells=SplitHalfSampler().min_cells(),
        metrics=load_metric_set(config.eval_metric_set),
        gene_sidecars=config.eval_gene_sidecars,
        ctrl_det_floors=(
            tuple(float(t) for t in ctrl_det_floors)
            if ctrl_det_floors else (0.10,)
        ),
    )
    log_step(
        logger, "krep:plan",
        "K=%d (model_eval_n_repeats) | split_half | min_truth=%d | n_conditions=%d | "
        "metric_set=%s | gene_sidecars=%s",
        plan.k, plan.min_truth_cells, n_conditions,
        config.eval_metric_set, plan.gene_sidecars,
    )
    return plan


def _inflate_work_queue(
    work_queue: Sequence[QuadrupletRow],
    plan: ModelEvalPlan,
    *,
    eval_group_keys: Sequence[str],
) -> tuple[list[QuadrupletRow], dict[int, int], InflationStats]:
    """Inflate per-condition pregen budget for the unified K-rep loop.

    Returns ``(inflated_queue, id(qrow) -> orig_n map, InflationStats)``.
    Pure: no IO or GPU.

    The model emits **K predictions per truth cell** so each rep gets a
    fresh full-N prediction slice.  Per condition with ``N`` truth cells::

        extras_per_cond = (K - 1) * N
        total_per_cond  = N + extras_per_cond = K * N

    Extras are distributed across the condition's qrows proportionally
    to each qrow's ``num_cells``; the rounding remainder is absorbed by
    the last qrow.  ``qrow`` here means a row of the eval work queue
    keyed by the full set of ``model_conditioning_variables`` -- on
    Tahoe a qrow is a unique ``(cell_name, drug, dose, plate)`` tuple
    within a condition; on Parse it is a unique ``(cytokine, donor)``
    tuple; the K-rep code never assumes a specific dataset's variables.
    Conditions with ``N < min_truth_cells`` are not inflated (they
    will be skipped at scoring time).
    """
    inflated: list[QuadrupletRow] = []
    orig_n_by_id: dict[int, int] = {}
    bundles = group_by_condition(list(work_queue), eval_group_keys=eval_group_keys)
    n_conditions = len(bundles)

    if not plan.active:
        for qrow in work_queue:
            inflated.append(qrow)
            orig_n_by_id[id(qrow)] = qrow.num_cells
        stats = InflationStats(
            n_conditions=n_conditions, n_enrolled=0, n_skipped_small=0,
            cells_orig=sum(orig_n_by_id.values()),
            cells_inflated=sum(r.num_cells for r in inflated),
            per_cond_extra=(),
        )
        return inflated, orig_n_by_id, stats

    n_enrolled = 0
    n_skipped_small = 0
    per_cond_extra: list[int] = []
    for cb in bundles:
        total_orig = sum(qrow.num_cells for qrow in cb.rows)
        if total_orig < plan.min_truth_cells:
            n_skipped_small += 1
            log_detail(
                logger, "krep:inflate_skip",
                "%s | total_orig=%d < %d → no extras (will be skipped at scoring)",
                _key_label(cb.key.as_dict()), total_orig, plan.min_truth_cells,
            )
            for qrow in cb.rows:
                inflated.append(qrow)
                orig_n_by_id[id(qrow)] = qrow.num_cells
            continue
        n_enrolled += 1
        # Total budget per condition = K * N; ``orig`` already covers the
        # first slice (rep 0) so we only need (K - 1) * N extras.
        total_extra = max(0, (plan.k - 1) * total_orig)
        per_cond_extra.append(total_extra)
        remaining = total_extra
        per_qrow_alloc: list[int] = []
        for i, qrow in enumerate(cb.rows):
            if i == len(cb.rows) - 1:
                alloc = remaining
            else:
                alloc = min(
                    remaining,
                    int(round(total_extra * (qrow.num_cells / total_orig))),
                )
            per_qrow_alloc.append(alloc)
            inflated_row = replace(qrow, num_cells=qrow.num_cells + alloc)
            inflated.append(inflated_row)
            orig_n_by_id[id(inflated_row)] = qrow.num_cells
            remaining -= alloc
        log_detail(
            logger, "krep:inflate",
            "%s | N=%d K=%d → extras=%d (qrows=%s) | total=%d=K*N",
            _key_label(cb.key.as_dict()), total_orig, plan.k,
            total_extra, per_qrow_alloc, total_orig + total_extra,
        )

    cells_orig = sum(orig_n_by_id.values())
    cells_inflated = sum(r.num_cells for r in inflated)
    stats = InflationStats(
        n_conditions=n_conditions, n_enrolled=n_enrolled,
        n_skipped_small=n_skipped_small,
        cells_orig=cells_orig, cells_inflated=cells_inflated,
        per_cond_extra=tuple(per_cond_extra),
    )
    budget = per_cond_extra or [0]
    log_step(
        logger, "krep:budget",
        "K=%d | conds_enrolled=%d skipped_small=%d | "
        "cells %d → %d (+%d, %.2fx) | per-cond extras min/median/max=%d/%d/%d",
        plan.k, stats.n_enrolled, stats.n_skipped_small,
        stats.cells_orig, stats.cells_inflated,
        stats.cells_inflated - stats.cells_orig,
        stats.cells_inflated / max(stats.cells_orig, 1),
        min(budget), int(np.median(budget)), max(budget),
    )
    return inflated, orig_n_by_id, stats


# ===========================================================================
# Per-condition helpers (pure given tensors)
# ===========================================================================


def persist_generated_without_control(
    persist_dumper: Optional["_PredictionDumper"],
    *,
    pred_t: torch.Tensor,
    cond_key: Mapping[str, Any],
) -> bool:
    """Dump an already-generated cloud when the train-control NPZ is missing.

    Donor-holdout PBS-test donors have a per-well truth file and no pooled
    control, so DEG/delta cannot run. Well×well energy only needs the npy.
    Returns False when persist is off, and the caller re-raises.
    """
    if persist_dumper is None:
        return False
    persist_dumper.submit(
        cond_key=cond_key,
        pred_t=pred_t,
        gen_event=None,
    )
    return True


def _load_condition_tensors(
    payload: ConditionPayload,
    *,
    control_cache: ControlCache,
    dev: torch.device,
    log_headroom: bool,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Move truth to ``dev`` and stack the GPU-resident control pool.

    Returns ``None`` when the payload lacks truth/ctrl.  In the streaming
    pipeline, predictions are produced by
    :func:`src.eval.orchestration.pregeneration.run_streaming_eval` directly in rep-major
    ``(K * N, n_genes)`` GPU layout (no host round-trip), so the K-rep
    loop reads the prediction tensor straight from the
    :class:`CompletedCondition` handoff.
    """
    if payload.truth is None or not payload.ctrl_keys:
        return None
    truth_t = torch.from_numpy(payload.truth).to(dev, dtype=torch.float32, non_blocking=True)
    ctrl_t = control_cache.stack(payload.ctrl_keys)
    if log_headroom and torch.cuda.is_available() and dev.type == "cuda":
        free_g, total_g, _, _ = _gpu_mem_snapshot(dev)
        log_step(
            logger, "metrics:headroom",
            "free=%.2f GiB / total=%.2f GiB | ctrl=%s truth=%s",
            free_g, total_g, tuple(ctrl_t.shape), tuple(truth_t.shape),
        )
    return truth_t, ctrl_t


def _condition_eligible_for_kreps(
    plan: ModelEvalPlan,
    truth_t: torch.Tensor,
    ctrl_t: torch.Tensor,
) -> bool:
    """True when this condition has enough truth + ctrl cells to K-rep."""
    return (
        plan.active
        and int(truth_t.shape[0]) >= plan.min_truth_cells
        and int(ctrl_t.shape[0]) >= plan.min_ctrl_cells
    )


def aggregate_rep_metrics(
    per_rep: Sequence[Mapping[str, float]],
) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
    """Per-key (mean, std, n_reps) over K reps, NaN-skipping per key.

    Pure: takes already-materialized float dicts, returns three flat
    dicts keyed by metric name. 

    Shared by model eval and noise ceiling so both JSON schemas agree.
    """
    if not per_rep:
        return {}, {}, {}
    keys = sorted({k for r in per_rep for k in r.keys()})
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    n_reps_per_key: dict[str, int] = {}
    for k in keys:
        vals = [r[k] for r in per_rep if k in r and not np.isnan(r[k])]
        if not vals:
            continue
        means[k] = float(np.mean(vals))
        stds[k] = float(np.std(vals, ddof=0))
        n_reps_per_key[k] = len(vals)
    return means, stds, n_reps_per_key


def _pred_control_half(pred_ctrl_t: torch.Tensor, *, seed: int, key_dict: Mapping[str, Any],
                       rep: int) -> torch.Tensor:
    """Half of the prediction's own control pool, the split-half size of ``ctrl_b``.

    A private generator: energy and MMD subsample through torch's global RNG, so drawing
    here must not move it (the copied cells' energy stays the held-out-control arm's).
    """
    m = int(pred_ctrl_t.shape[0])
    if m // 2 < 2:
        raise ValueError(f"{_key_label(key_dict)}: prediction controls have {m} cells; need >= 4 to halve")
    g = torch.Generator(device=pred_ctrl_t.device)
    g.manual_seed(_stable_cond_seed(seed, {**key_dict, "arm": "neighbor_ctrl"}) + rep)
    return pred_ctrl_t[torch.randperm(m, generator=g, device=pred_ctrl_t.device)[: m // 2]]


def _score_condition_kreps(
    *,
    plan: ModelEvalPlan,
    truth_t: torch.Tensor,
    ctrl_t: torch.Tensor,
    pred_t_all: Optional[torch.Tensor],
    k_eff: int,
    output_space: Any,
    seed: int,
    key_dict: Mapping[str, Any],
    first_condition_verbose: bool,
    pred_ctrl_t: Optional[torch.Tensor] = None,
) -> KRepResult:
    """Unified K-rep loop scoring all four metric groups for one condition.

    Per rep ``r ∈ [0, K_eff)``:

    * :class:`~src.eval.reference.split_half.SplitHalfSampler` draws
      ``(truth_arm, truth_b, ctrl_a, ctrl_b, pred_cell_idx)`` with
      ``torch.manual_seed(cond_seed + rep)``.
    * ``pred_rep = pred_t_all[r * N : (r + 1) * N]`` (rep-major layout).
    * ``pred_arm = pred_rep[pred_cell_idx]`` stands where the split-half reference
      puts ``truth_b``; ``pred_t_all=None`` IS that reference (raw counts, no projection).
    * Arm + delta + DEG bundles all score ``(truth_arm, pred_arm)`` paired
      with ``(ctrl_a, ctrl_b)`` — identical sizing to noise ceiling.
    * ``pred_ctrl_t`` (opt-in, copy arms only): ``ctrl_b`` becomes a half of
      these controls instead (:func:`_pred_control_half`); truth, ``ctrl_a``,
      the energy draw and the DEG detection floor (still the drawn ``ctrl_b``,
      so truth's DEGs are every arm's) are untouched.

    The K rep dicts share a single D2H sync via
    :func:`materialize_metric_tensor_batch`; per-key aggregation
    (mean / std / n_reps) is done in :func:`aggregate_rep_metrics` on
    the host.
    """
    n_truth = int(truth_t.shape[0])
    sampler = SplitHalfSampler()
    dev = truth_t.device

    if pred_t_all is not None and torch.cuda.is_available() and dev.type == "cuda":
        free_g, _, alloc_g, _ = _gpu_mem_snapshot(dev)
        log_detail(
            logger, "krep:rep_setup",
            "%s | n_truth=%d K_eff=%d | "
            "pred_all=%s (%.2f GiB) | gpu alloc=%.2f free=%.2f GiB",
            _key_label(key_dict), n_truth, k_eff,
            tuple(pred_t_all.shape),
            _gib(pred_t_all.element_size() * pred_t_all.nelement()),
            alloc_g, free_g,
        )

    # a model's cells go through its output space; the reference's truth half is already raw
    project_b = pred_t_all is not None
    cond_seed = _stable_cond_seed(seed, key_dict)
    rep_tensor_dicts: list[dict[str, torch.Tensor]] = []
    t_kloop = time.perf_counter()
    for rep in range(k_eff):
        truth_arm, truth_b, ctrl_a, ctrl_b, pred_cell_idx = sampler.sample(
            truth_t, ctrl_t, rep=rep, seed=cond_seed,
        )
        pred_arm = (pred_t_all[rep * n_truth : (rep + 1) * n_truth][pred_cell_idx]
                    if project_b else truth_b)
        det_ctrl_b = None
        if pred_ctrl_t is not None:
            det_ctrl_b, ctrl_b = ctrl_b, _pred_control_half(pred_ctrl_t, seed=seed, key_dict=key_dict, rep=rep)
        # the reference's truth half is already counts; a model's cells name their output space
        rep_metrics = score_pair(
            truth_arm, pred_arm, ctrl_a, ctrl_b,
            a_space="raw", b_space=output_space if project_b else "raw",
            det_ctrl_b=det_ctrl_b, metrics=plan.metrics,
            ctrl_det_floors=plan.ctrl_det_floors or (0.10,),
        )
        check_metric_keys(rep_metrics, plan.metrics)
        rep_tensor_dicts.append(rep_metrics)
        if first_condition_verbose:
            log_detail(
                logger, "krep:rep",
                "%s rep=%d/%d | truth_arm=%s pred_arm=%s ctrl_a=%s ctrl_b=%s",
                _key_label(key_dict), rep + 1, k_eff,
                tuple(truth_arm.shape), tuple(pred_arm.shape),
                tuple(ctrl_a.shape), tuple(ctrl_b.shape),
            )

    per_rep = materialize_metric_tensor_batch(rep_tensor_dicts)
    means, stds, n_reps_per_key = aggregate_rep_metrics(per_rep)
    elapsed = time.perf_counter() - t_kloop

    headline_summary = ", ".join(
        f"{k.split('/', 1)[-1]}={means[k]:.3g}±{stds[k]:.3g}(n={n_reps_per_key[k]})"
        for k in _KREP_HEADLINE_KEYS if k in means
    ) or "no headline keys present"
    log_step(
        logger, "krep:result",
        "%s | K_eff=%d keys=%d | %.1fs | %s",
        _key_label(key_dict), k_eff, len(means), elapsed, headline_summary,
    )
    return KRepResult(
        means=means, stds=stds, n_reps_per_key=n_reps_per_key,
        k_effective=k_eff, elapsed_seconds=elapsed,
    )


def _structure_krep_metrics(
    rep_result: KRepResult,
) -> tuple[dict[str, float], dict[str, float]]:
    """Split K-rep aggregates into ``(metrics, metrics_std)`` dicts.

    Pure: produces the JSON schema written into ``results[i]`` so plot
    consumers can read mean ± std without per-metric ``_n_reps`` keys.
    """
    return (
        dict(rep_result.means),
        dict(rep_result.stds),
    )


@dataclass(frozen=True)
class PregeneratedScoreResult:
    """Metrics + parquet extras for one pre-generated prediction matrix."""

    metrics: dict[str, float]
    metrics_std: dict[str, float]
    k_effective: int
    extras: Optional[CondExtras]  # None unless plan.gene_sidecars


def pregenerated_eval_plan(
    *,
    ctrl_det_floors: tuple[float, ...] = (0.10,),
    metrics: Optional[frozenset[str]] = None,
    gene_sidecars: bool = False,
) -> ModelEvalPlan:
    """Fixed K=1 split-half plan for external NPZ predictions.

    ``ctrl_det_floors`` mirrors the pertgen CLI default so competitor JSONs
    carry the same calibrated DEG family as PertGen runs (cross-plots pick it up
    via metric-key intersection). ``metrics=None`` scores every key.
    """
    return ModelEvalPlan(
        active=True, k=1, metrics=metrics, gene_sidecars=gene_sidecars,
        ctrl_det_floors=(
            tuple(float(t) for t in ctrl_det_floors)
            if ctrl_det_floors else (0.10,)
        ),
    )


def score_pregenerated_condition(
    *,
    truth_t: torch.Tensor,
    ctrl_t: torch.Tensor,
    pred_t: torch.Tensor,
    output_space: Any,
    seed: int,
    key_dict: Mapping[str, Any],
    plan: Optional[ModelEvalPlan] = None,
    pred_ctrl_t: Optional[torch.Tensor] = None,
) -> Optional[PregeneratedScoreResult]:
    """Score one condition from a single pre-generated prediction matrix (K=1).

    Uses the same K-rep loop as :func:`run_model_eval` with ``k_eff=1``:
    :class:`~src.eval.reference.split_half.SplitHalfSampler` draws for
    delta/DEG and arm metrics.  Returns ``None`` when the condition fails
    :func:`_condition_eligible_for_kreps`.

    ``pred_ctrl_t``: None (default, every published arm) scores the prediction against the
    held-out condition's own controls. A copy arm may pass its neighbour's controls instead,
    so the copied cells' Δ and DEGs are against the line they were measured in; the gene
    arrays (extras) stay on ``ctrl_t``.
    """
    plan = plan or pregenerated_eval_plan()
    if int(plan.k) != 1:
        raise ValueError(
            f"score_pregenerated_condition requires K=1 (one NPZ matrix per "
            f"condition), got K={plan.k}",
        )
    if not _condition_eligible_for_kreps(plan, truth_t, ctrl_t):
        return None

    n_truth = int(truth_t.shape[0])
    rep_result = _score_condition_kreps(
        plan=plan,
        truth_t=truth_t,
        ctrl_t=ctrl_t,
        pred_t_all=pred_t,
        k_eff=1,
        output_space=output_space,
        seed=int(seed),
        key_dict=key_dict,
        first_condition_verbose=False,
        pred_ctrl_t=pred_ctrl_t,
    )
    metrics, metrics_std = _structure_krep_metrics(rep_result)
    extras = _compute_condition_extras(
        pred_t=pred_t[:n_truth],
        truth_t=truth_t,
        ctrl_t=ctrl_t,
        output_space=output_space,
    ) if plan.gene_sidecars else None
    return PregeneratedScoreResult(
        metrics=metrics,
        metrics_std=metrics_std,
        k_effective=int(rep_result.k_effective),
        extras=extras,
    )


def _compute_condition_extras(
    *,
    pred_t: torch.Tensor,
    truth_t: torch.Tensor,
    ctrl_t: torch.Tensor,
    output_space: Any,
    pred_label: str = "pred",
) -> CondExtras:
    """Compute per-condition gene_summary / deg_detail / dropout_summary."""
    pred_raw_int = to_raw_integer(pred_t, output_space)
    gene_summary = compute_gene_summary(
        truth_raw=truth_t,
        pred_raw_integer=pred_raw_int,
        control_raw=ctrl_t,
        pred_label=pred_label,
    )
    deg_detail = compute_deg_detail(
        control_raw=ctrl_t,
        truth_raw=truth_t,
        predicted_model_space=pred_t,
        model_output_space=output_space,
        fdr_threshold=0.01,
    )
    dropout_summary = compute_dropout_summary(
        control_raw=ctrl_t,
        truth_raw=truth_t,
        predicted_model_space=pred_t,
        model_output_space=output_space,
    )
    return CondExtras(
        gene_summary=gene_summary,
        deg_detail=deg_detail,
        dropout_summary=dropout_summary,
    )


def build_result_row(
    *,
    key_dict: Mapping[str, str],
    n_pred: int,
    n_truth: int,
    n_ctrl: int,
    metrics: Mapping[str, float],
    metrics_std: Mapping[str, float],
    k: int,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build the per-condition JSON row written to ``results[]``.

    Shared by model eval and noise ceiling: same schema, same clustering
    (``deg``, ``pseudobulk_arm``, ``pseudobulk_delta``, ``distributional``)
    so plots read both files via :func:`_load_result_metrics`.

    Bulk gene arrays live in the parquet sidecar; JSON carries scalars only.
    ``k`` is the number of K-rep draws that contributed to ``metrics`` /
    ``metrics_std`` for this condition.

    Args:
        extra: Optional extra scalar fields appended after the schema
            keys (e.g. ``n_control_groups`` for noise ceiling).
    """
    flat_m = {name: float(v) for name, v in metrics.items()}
    flat_s = {name: float(v) for name, v in metrics_std.items()}
    row: dict[str, Any] = {
        "condition_keys": dict(key_dict),
        "n_predicted": int(n_pred),
        "n_truth": int(n_truth),
        "n_control": int(n_ctrl),
        "k": int(k),
        "metrics": cluster_flat_metrics(flat_m),
        "metrics_std": cluster_flat_metrics(flat_s),
    }
    if extra:
        row.update({str(name): value for name, value in extra.items()})
    return row


# ===========================================================================
# Output assembly
# ===========================================================================


def _genesis_fields(
    *,
    bundle: PertGenInferenceBundle,
    pg_stats: StreamingPregenStats,
    eval_split: str,
    mcv: list[str],
    eval_group_keys: Sequence[str],
    plot_aggregate_over: Optional[str],
    truth_dir: Path,
    k_reps: int,
    metric_set: str,
    gene_sidecars: bool,
    conditions_query: Optional[str] = None,
    max_conditions: Optional[int] = None,
    conditions_seed: int = 0,
) -> dict[str, Any]:
    """GENESIS's own ``metrics_summary.json`` fields; scoring.write_summary adds the shared ones."""
    return {
        "model_id": bundle.model_id,
        "gene_set_id": bundle.gene_set_hash,
        "n_genes": len(bundle.gene_names),
        "run_dir": str(bundle.run_dir),
        "checkpoint": str(bundle.checkpoint_path),
        "eval_split": eval_split,
        "model_output_space": bundle.output_space.value,
        "gene_summary_pred_label": "pred",
        "gene_arrays_parquet": METRICS_GENE_ARRAYS_FILENAME if gene_sidecars else None,
        "metric_set": metric_set,
        "model_eval_n_repeats": int(k_reps),
        # per_condition_rep: noise seeded per (condition, repeat), so a subset draws what the full run draws
        "generation_noise_seeding": "per_condition_rep",
        "noise_ceiling_strategy": "split_half",  # figure readers check it
        "condition_variables": mcv,
        "eval_group_keys": list(eval_group_keys),
        "plot_aggregate_over": plot_aggregate_over,
        "eval_cache_namespace": str(truth_dir),
        # Subset provenance: non-null means these numbers cover a subset of the
        # split, not the whole split -- do not compare them to a full-split run
        # without checking these three fields.
        "conditions_query": conditions_query,
        "max_conditions": int(max_conditions) if max_conditions is not None else None,
        "conditions_seed": int(conditions_seed),
        "forward_seconds": pg_stats.elapsed_seconds,
        "forward_slots_per_second": pg_stats.slots_per_second,
        "forward_n_batches": pg_stats.n_batches,
        "forward_n_total_cells": pg_stats.n_total_cells,
        "gene_names": list(bundle.gene_names),
    }


def summary_overall_stats(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    """Top-level ``mean_metrics`` / ``median_metrics`` / ``mean_metrics_std``.

    Aggregates per-condition rows produced by :func:`build_result_row`
    into three flat dicts.  Shared by the model-eval summary writer and
    the noise-ceiling summary writer so both JSON files expose the same
    overall-summary keys (and so a side-by-side legend/headline reads
    the same fields from either path).
    """
    if not results:
        return {}
    from src.eval.metrics.reporting import flatten_clustered_metrics

    def _flat_metrics(row: Mapping[str, Any]) -> dict[str, float]:
        return flatten_clustered_metrics(row.get("metrics", {}))

    def _flat_stds(row: Mapping[str, Any]) -> dict[str, float]:
        return flatten_clustered_metrics(row.get("metrics_std", {}))

    metric_keys = sorted(_flat_metrics(results[0]).keys())
    out: dict[str, dict[str, float]] = {
        "mean_metrics": {
            k: float(np.mean([_flat_metrics(r)[k] for r in results if k in _flat_metrics(r)]))
            for k in metric_keys
        },
        "median_metrics": {
            k: float(np.median([_flat_metrics(r)[k] for r in results if k in _flat_metrics(r)]))
            for k in metric_keys
        },
    }
    std_keys = sorted(_flat_stds(results[0]).keys())
    if std_keys:
        out["mean_metrics_std"] = {
            k: float(np.mean([_flat_stds(r)[k] for r in results if k in _flat_stds(r)]))
            for k in std_keys
        }
    return out


# ===========================================================================
# Optional persistent prediction cache
# ===========================================================================


class _PredictionDumper:
    """Async D2H + disk dump of per-condition rep-major prediction buffers.

    A dedicated CUDA stream owns the GPU→pinned-CPU copy (it waits on the
    producer's ``gen_event`` so the copy never races generation), and a
    background daemon thread consumes a bounded queue of pinned tensors,
    writing each to ``<out_root>/<cond_key_label>.npy`` after the D2H
    completes.  This never gates the GPU pipeline: ``submit()`` returns
    immediately after queueing the copy.

    On CPU (test harness), the copy and write run synchronously inside
    ``submit`` so the dumper still works end-to-end without CUDA.
    """

    def __init__(
        self,
        *,
        out_root: Path,
        condition_variables: Sequence[str],
        device: torch.device,
        queue_max: int = 8,
    ) -> None:
        import queue
        import threading

        self._root = Path(out_root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._cond_vars = list(condition_variables)
        self._device = device
        self._is_cuda = (device.type == "cuda")
        self._stream = torch.cuda.Stream(device=device) if self._is_cuda else None
        self._queue: "queue.Queue" = queue.Queue(maxsize=int(queue_max))
        self._sentinel: object = object()
        self._n_submitted = 0
        self._n_written = 0
        self._failed: list[tuple[str, Exception]] = []
        self._thread = threading.Thread(
            target=self._writer_loop, name="pertgen-persist-dump", daemon=True,
        )
        self._thread.start()
        log_step(
            logger, "persist",
            "predictions will be written to %s | queue_max=%d",
            self._root, int(queue_max),
        )

    def _cond_label(self, cond_key: Mapping[str, Any]) -> str:
        """Build a deterministic filesystem-safe label from the condition keys."""
        parts = []
        for k in self._cond_vars:
            if k in cond_key:
                parts.append(f"{k}={safe_filename_part(str(cond_key[k]))}")
        if not parts:
            parts.append("cond")
        return "__".join(parts)

    def submit(
        self,
        *,
        cond_key: Mapping[str, Any],
        pred_t: torch.Tensor,
        gen_event: Optional[torch.cuda.Event],
    ) -> None:
        """Queue one condition's GPU buffer for async D2H + disk write."""
        self._n_submitted += 1
        dst_path = self._root / f"{self._cond_label(cond_key)}.npy"

        if self._is_cuda and self._stream is not None:
            # Stage the copy on a dedicated D2H stream gated by gen_event so
            # producer kernels finish before we touch pred_t.
            cpu_buf = torch.empty(
                pred_t.shape, dtype=pred_t.dtype, pin_memory=True,
            )
            if gen_event is not None:
                self._stream.wait_event(gen_event)
            # The dump stream is not the stream pred_t was created on, so we
            # must tell the caching allocator that this stream is in use --
            # otherwise the storage could be reclaimed before the D2H copy
            # has actually run on self._stream.
            pred_t.record_stream(self._stream)
            with torch.cuda.stream(self._stream):
                cpu_buf.copy_(pred_t, non_blocking=True)
            ready_event = torch.cuda.Event()
            ready_event.record(self._stream)
        else:
            # CPU fallback: synchronous copy + no event needed.
            cpu_buf = pred_t.detach().clone().cpu()
            ready_event = None
        self._queue.put((cpu_buf, dst_path, ready_event))

    def _writer_loop(self) -> None:
        """Daemon thread: pop pinned buffers, sync events, persist .npy files."""
        while True:
            item = self._queue.get()
            if item is self._sentinel:
                self._queue.task_done()
                return
            cpu_buf, dst_path, ready_event = item
            try:
                if ready_event is not None:
                    ready_event.synchronize()
                # NumPy view borrows the pinned tensor's storage; np.save
                # writes a fresh copy via the .npy format so it is safe to
                # release the tensor afterwards.
                np.save(dst_path, cpu_buf.numpy())
                self._n_written += 1
            except Exception as exc:  # noqa: BLE001  -- fail loud per principle 4
                self._failed.append((str(dst_path), exc))
                logger.error("persist | failed to write %s: %s", dst_path, exc)
            finally:
                del cpu_buf
                self._queue.task_done()

    def close(self) -> None:
        """Flush queue, join the writer thread, raise on any prior failure."""
        self._queue.put(self._sentinel)
        self._thread.join()
        log_step(
            logger, "persist",
            "wrote %d/%d prediction tensors to %s",
            self._n_written, self._n_submitted, self._root,
        )
        if self._failed:
            raise RuntimeError(
                f"prediction dumper had {len(self._failed)} write failures; "
                f"first: {self._failed[0][0]} -> {self._failed[0][1]!r}",
            )


# ===========================================================================
# Orchestrator
# ===========================================================================


def run_model_eval(
    *,
    run_dir: Path,
    eval_split: str,
    manifest: EvalManifest,
    config: PertGenConfig,
    checkpoint: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    device: str = "cuda",
    batch_size: int = 16384,
    seed: int = 42,
    min_cells: int = 4,
    overwrite: bool = False,
    pipeline_depth: int = 3,
    metrics_workers: int = 4,
    sync_cuda_streams: bool = False,
    pad_last_batch: bool = False,
    use_compile: bool = False,
    persist_predictions: bool = False,
    ctrl_det_floors: tuple[float, ...] = (0.10,),
    startup_profiler: Optional[EvalStartupProfiler] = None,
    conditions_query: Optional[str] = None,
    max_conditions: Optional[int] = None,
    conditions_seed: int = 0,
    drop_control_rows: bool = True,
) -> ModelEvalResult:
    """Stream-generate predictions and score them against eval NPZ truth.

    Predictions never leave the GPU: the forward loop fills per-condition
    rep-major buffers on ``stream_gen`` while scoring overlaps on
    ``stream_metrics`` for previously-completed conditions.  No prediction
    accumulates on the host, so RAM is constant in the number of conditions
    and the wall time approaches ``max(forward_time, metrics_time)``.

    Requires ``manifest.eval_truth_dir`` on disk (per-condition NPZ under ``eval_cache/``).

    Args:
        pipeline_depth: Max number of conditions whose GPU rep-major
            buffers are allocated simultaneously.  Default 3 covers
            typical (gen, score, prefetch) overlap; bump to 4-5 only if
            you see metric-stream stalls.
        persist_predictions: When True, each completed condition's GPU
            buffer is also async-D2H'd and written to
            ``<out>/predictions/<cond_key>.npy`` for later reuse.  Adds
            disk I/O but does not slow down the GPU pipeline.
        conditions_query / max_conditions / conditions_seed: Optional subset
            of the split's conditions -- a ``DataFrame.query`` expression and
            a seeded cap.  Needed for ``eval_split="train"`` (tens of
            thousands of conditions) and for carving a composition-matched
            train subset to compare against test.
    """
    run_dir = Path(run_dir).resolve()
    ensure_eval_phase_logging()
    # Bind composition L_ref for log1p_slice_med_rgs (n_genes/20k · median UMI).
    from ..metrics.spaces import set_dataset_median_umi
    if config.dataset is None:
        raise ValueError(
            "config.dataset is required to set composition L_ref for "
            "log1p_slice_med_rgs; got None."
        )
    l_ref = set_dataset_median_umi(str(config.dataset))
    logger.info("composition L_ref=%.1f (dataset=%s)", l_ref, config.dataset)
    profiler = startup_profiler if startup_profiler is not None else EvalStartupProfiler(
        label="startup",
    )
    truth_dir = manifest.eval_truth_dir
    if not truth_dir.is_dir():
        raise FileNotFoundError(
            "Eval truth cache missing on manifest. "
            "Set build_eval_cache=true and run setup, or scripts/data/build_cache.py eval.",
        )
    if resolve_eval_cache_mode(truth_dir) != EVAL_NPZ_MODE:
        raise ValueError(
            f"Model eval requires NPZ eval truth at {truth_dir} "
            f"(mode={resolve_eval_cache_mode(truth_dir)!r}). "
            "Rebuild with scripts/data/build_cache.py eval.",
        )

    with profiler.phase("bundle_load"):
        bundle = _load_inference_bundle(
            run_dir, checkpoint=checkpoint, device=device, use_compile=use_compile,
        )
    out, metrics_path = _resolve_eval_paths(run_dir, out_dir, bundle.checkpoint_path, eval_split, config)
    cached = _check_existing_metrics(metrics_path, out, eval_split, overwrite)
    if cached is not None:
        return cached

    with profiler.phase("work_queue"):
        work_queue = _load_work_queue_or_none(
            manifest=manifest, config=config, bundle=bundle,
            eval_split=eval_split, min_cells=min_cells,
            conditions_query=conditions_query,
            max_conditions=max_conditions,
            conditions_seed=conditions_seed,
            drop_control_rows=drop_control_rows,
        )
    if work_queue is None:
        return _write_empty_split_summary(metrics_path, out, eval_split)

    mcv = list(manifest.group_cols or config.model_conditioning_variables or [])
    eval_group_keys = resolve_eval_group_keys(mcv, config.control_condition, config.control_match_keys)
    n_conditions = len(
        group_by_condition(work_queue, eval_group_keys=eval_group_keys),
    )
    log_step(
        logger, "setup",
        "split=%s | %d conditions, %d quadruplets | eval_group=%s | ckpt=%s | out=%s",
        eval_split, n_conditions, len(work_queue), eval_group_keys,
        bundle.checkpoint_path.name, out,
    )
    log_step(logger, "cache", "eval truth %s", truth_dir)

    plan = _make_model_eval_plan(config, n_conditions, ctrl_det_floors=ctrl_det_floors)
    inflated_queue, orig_n_by_id, _ = _inflate_work_queue(
        work_queue, plan, eval_group_keys=eval_group_keys,
    )

    from src.eval.scoring import GenesisSource, score_conditions, write_summary  # scoring imports this module

    persist_dumper = (_PredictionDumper(out_root=out / "predictions", condition_variables=mcv, device=bundle.device)
                      if persist_predictions else None)
    source = GenesisSource(bundle=bundle, inflated_queue=inflated_queue, orig_n_by_id=orig_n_by_id, k=plan.k,
                           batch_size=batch_size, seed=seed, pad_last_batch=pad_last_batch,
                           pipeline_depth=pipeline_depth, eval_split=eval_split,
                           sync_cuda_streams=sync_cuda_streams, persist_dumper=persist_dumper)
    profiler.log_summary(logger)
    script_t0 = time.perf_counter()
    try:
        results, skips = score_conditions(
            source, cache_ns=truth_dir, bundles=group_by_condition(inflated_queue, eval_group_keys=eval_group_keys),
            condition_variables=mcv, plan=plan, seed=seed, device=bundle.device, metrics_workers=metrics_workers,
            gene_arrays_path=out / METRICS_GENE_ARRAYS_FILENAME, desc=f"Model eval [{eval_split}]",
        )
    finally:
        if persist_dumper is not None:
            persist_dumper.close()
    pg_stats = source.stats
    log_step(logger, "metrics", "forward=%.1fs (%.0f cells/s, %d batches, %d conditions emitted)",
             pg_stats.elapsed_seconds, pg_stats.slots_per_second, pg_stats.n_batches, pg_stats.n_conditions_yielded)

    fields = _genesis_fields(
        bundle=bundle, pg_stats=pg_stats, eval_split=eval_split, mcv=mcv, eval_group_keys=eval_group_keys,
        plot_aggregate_over=(config.plot_aggregate_over or "").strip() or None,
        truth_dir=truth_dir, k_reps=plan.k, metric_set=config.eval_metric_set, gene_sidecars=plan.gene_sidecars,
        conditions_query=conditions_query, max_conditions=max_conditions, conditions_seed=conditions_seed,
    )
    summary = write_summary(metrics_path, fields=fields, results=results, skips=skips)
    skipped = int(sum(skips.values()))
    mean_line = (
        " | mean " + headline_metrics(summary.get("mean_metrics", {}))
        if results else ""
    )
    log_step(
        logger, "done",
        "split=%s | %d conditions | wall=%.1fs | %s%s",
        eval_split, len(results), time.perf_counter() - script_t0, metrics_path, mean_line,
    )
    return ModelEvalResult(
        eval_split=eval_split,
        out_dir=out,
        metrics_path=metrics_path,
        n_conditions=len(results),
        n_skipped=skipped,
        forward_seconds=float(pg_stats.elapsed_seconds),
    )
