"""Dataset-agnostic eval artifacts: train cache, eval NPZ cache, noise ceiling.

Post-training eval orchestration lives in ``main.run`` (eval cache → model eval
→ noise ceiling → model-vs-ceiling plots).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional, Sequence

from src.config.config import (
    PertGenConfig,
    resolve_cache_dir,
    resolve_files_glob,
    resolve_noise_ceiling_dir,
)
from src.data.data_utils import (
    EvalManifest,
    resolve_control_match_keys,
    resolve_eval_group_keys,
)
from src.eval.cache.eval_cache_io import gene_set_hash

logger = logging.getLogger(__name__)

EVAL_SPLITS = ("validation", "test")


################################################################################
#### cache
################################################################################


def raw_hvg_cache_ready(cache_dir: Path) -> bool:
    return (Path(cache_dir) / "manifest.json").is_file()


def maybe_build_cache(config: PertGenConfig, *, rank0: bool = True) -> None:
    """Build mmap CSR cache when missing under ``resolve_cache_dir``."""
    if not config.build_cache:
        return
    cache_path = resolve_cache_dir(config)
    if cache_path is None:
        return
    force = config.overwrite_cache
    if raw_hvg_cache_ready(cache_path) and not force:
        print(f"  [train_cache] skip | cache complete at {cache_path}", flush=True)
        return
    if not rank0:
        return
    if not config.data_dir or not config.hvg_genes_path:
        raise ValueError("Cache build requires data_dir and hvg_genes_path.")

    print(f"  [train_cache] BUILDING mmap CSR at {cache_path}", flush=True)

    from src.data.cache.raw_hvg_cache_build import (
        obs_keys_for_raw_hvg_cache,
        run_raw_hvg_cache_build,
    )

    mcv = list(config.model_conditioning_variables)
    ctrl = dict(config.control_condition or {})
    match_keys = resolve_control_match_keys(mcv, ctrl, config.control_match_keys)

    run_raw_hvg_cache_build(
        data_dir=config.data_dir,
        hvg_genes_path=config.hvg_genes_path,
        cache_dir=cache_path,
        obs_keys=obs_keys_for_raw_hvg_cache(mcv),
        files_glob=config.files_glob,
        n_workers=8,
        force=force,
        ranges_per_shard=4,
        condition_variables=mcv,
        control_condition=ctrl,
        control_match_keys=match_keys,
    )


def maybe_build_eval_cache(config: PertGenConfig, dm: Any) -> dict[str, Any]:
    """Ensure the per-condition eval NPZ cache exists."""
    from src.data.cache.eval_cache_build import run_eval_cache_build

    manifest = dm.eval_manifest
    mcv = list(config.model_conditioning_variables or ())
    ctrl = dict(config.control_condition or {})
    match_keys = resolve_control_match_keys(mcv, ctrl, config.control_match_keys)
    allow_build = config.build_eval_cache
    out_dir = manifest.eval_cache_dir

    return run_eval_cache_build(
        data_dir=Path(config.data_dir),
        h5ad_paths=sorted(Path(config.data_dir).glob(resolve_files_glob(config))),
        output_dir=out_dir,
        hvg_indices=dm._hvg_indices,
        hvg_genes_path=Path(dm.hvg_genes_path),
        condition_variables=mcv,
        control_condition=ctrl,
        match_keys=match_keys,
        gene_set_hash_value=manifest.gene_set_hash,
        n_genes=manifest.n_genes,
        normalization_id=str(manifest.normalization_id or "raw_counts"),
        split_config=manifest.split_config,
        metadata_joins=list(config.metadata_joins or ()),
        condition_summary_path=manifest.condition_summary,
        min_cells=config.eval_min_cells,
        allow_build=allow_build,
    )


def _eval_config_keys(config: PertGenConfig) -> tuple[list[str], dict, list[str]]:
    mcv = list(config.model_conditioning_variables or [])
    ctrl = dict(config.control_condition or {})
    match_keys = resolve_control_match_keys(mcv, ctrl, config.control_match_keys)
    return mcv, ctrl, match_keys


################################################################################
#### noise_ceiling
################################################################################


def _split_config_hash(split_config: str | Path | None) -> str:
    """Stable short hash of the split YAML (or ``none`` when unset); raises when the file is missing."""
    from src.data.cache.eval_cache_build import split_config_hash  # lazy: heavy module

    return split_config_hash(split_config)


def noise_ceiling_gene_norm_dir_name(
    *,
    gene_set_hash_value: str,
    n_genes: int,
    normalization_id: str,
) -> str:
    """Gene-set + normalization bucket under ``.../noise_ceiling/``."""
    return (
        f"geneset_{gene_set_hash_value}__n_{int(n_genes)}__norm_{normalization_id}"
    )


def noise_ceiling_run_dir_name(
    *,
    n_repeats: int,
    split_config_hash: str,
    min_cells: int,
    eval_split: str,
) -> str:
    """Leaf directory for one split under a gene-set + norm bucket."""
    return f"nr_{int(n_repeats)}__splitcfg_{split_config_hash}__mincells_{int(min_cells)}__{eval_split}"


def resolve_noise_ceiling_paths(
    config: PertGenConfig,
    gene_names: Sequence[str],
    eval_split: str,
    *,
    normalization_id: str = "raw_counts",
) -> tuple[Path, Path]:
    """Return ``(out_dir, out_json)`` for one split under the shared cache."""
    gsh = gene_set_hash(gene_names)
    gene_norm_dir = noise_ceiling_gene_norm_dir_name(
        gene_set_hash_value=gsh,
        n_genes=len(gene_names),
        normalization_id=normalization_id,
    )
    run_dir = noise_ceiling_run_dir_name(
        n_repeats=config.noise_ceiling_n_repeats,
        split_config_hash=_split_config_hash(config.split_config),
        min_cells=config.eval_min_cells,
        eval_split=eval_split,
    )
    out_dir = resolve_noise_ceiling_dir(config) / gene_norm_dir / run_dir
    return out_dir, out_dir / "noise_ceiling.json"


def resolve_noise_ceiling_json_path(
    config: PertGenConfig,
    gene_names: Sequence[str],
    eval_split: str,
    *,
    normalization_id: str | None = None,
) -> Path:
    """Path to shared ``noise_ceiling.json`` for one split (model-independent)."""
    _, out_json = resolve_noise_ceiling_paths(
        config,
        gene_names,
        eval_split,
        normalization_id=str(normalization_id or "raw_counts"),
    )
    return out_json


def _noise_ceiling_device(_config: PertGenConfig) -> str:
    """Pick torch device for noise-ceiling scoring (overridable in tests)."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def compute_noise_ceiling(
    config: PertGenConfig,
    gene_names: Sequence[str],
    manifest: EvalManifest,
    *,
    eval_splits: Sequence[str] = EVAL_SPLITS,
    ctrl_det_floors: tuple[float, ...] = (0.10,),
    drop_control_rows: bool = True,
    max_conditions: Optional[int] = None,
) -> dict[str, Path]:
    """Score noise ceiling on the eval NPZ truth cache; write per-split JSON summaries.

    Reads perturbed/control NPZs from ``manifest.eval_truth_dir`` (the eval cache
    namespace). No model checkpoint or forward pass is involved.

    ``ctrl_det_floors`` (default ``(0.10,)``) is the detection-fraction floor of the
    one calibrated DEG family, so the ceiling JSON carries the same
    ``deg_*_detfloor10`` metric keys as the model metrics_summary.json
    (cross-plots pick them up via metric-key intersection).

    ``max_conditions`` scores a seeded subset (gates, smoke tests); each condition draws
    from its own stable seed, so its values equal those of the full run.
    """
    truth_dir = manifest.eval_truth_dir
    if not any(truth_dir.glob("perturbed_cache__*.npz")):
        raise RuntimeError(
            f"Eval NPZ cache empty ({truth_dir}). "
            "Run scripts/data/build_cache.py eval first.",
        )

    mcv, ctrl, match_keys = _eval_config_keys(config)
    gene_names = list(gene_names)
    gsh = gene_set_hash(gene_names)

    from src.eval.cache.eval_cache_io import load_work_queue_or_none
    from src.eval.metrics.reporting import load_metric_set
    from src.eval.plots.gene_arrays import BOOTSTRAP_GENE_ARRAYS_FILENAME
    from src.eval.reference.split_half import SplitHalfSampler, run_split_half_reference as run_noise_ceiling
    from src.eval.scoring import write_summary

    try:
        device = _noise_ceiling_device(config)
    except Exception:
        device = "cpu"

    n_repeats = config.noise_ceiling_n_repeats
    overwrite = config.overwrite_noise_ceiling
    seed = config.seed
    sampler = SplitHalfSampler()
    metrics = load_metric_set(config.eval_metric_set)
    if manifest.gene_set_hash and gsh != manifest.gene_set_hash:
        raise RuntimeError(
            f"Gene set hash mismatch (dm={gsh!r}, manifest={manifest.gene_set_hash!r})."
        )
    mcv = list(config.model_conditioning_variables or [])
    control_condition = dict(config.control_condition or {})
    metadata_joins = list(config.metadata_joins or [])
    min_cells = config.eval_min_cells
    norm_id = str(manifest.normalization_id or "raw_counts")
    split_yaml = manifest.split_config or config.split_config
    if not split_yaml:
        raise ValueError("split_config required for noise-ceiling work queue.")
    split_cfg_hash = _split_config_hash(split_yaml)
    eval_group_keys = resolve_eval_group_keys(mcv, control_condition, config.control_match_keys)

    out_base = resolve_noise_ceiling_dir(config)
    out_base.mkdir(parents=True, exist_ok=True)
    logger.info("Noise-ceiling output root (model-independent): %s", out_base)
    written: dict[str, Path] = {}

    for split in eval_splits:
        out_dir, out_json = resolve_noise_ceiling_paths(
            config, gene_names, split, normalization_id=norm_id,
        )
        if out_json.is_file() and not overwrite:
            # The path does not encode the metric set, so a cached core-only reference must not
            # stand in for a wider request. A file without the field holds every key ("extended").
            cached_set = json.loads(out_json.read_text()).get("metric_set", "extended")
            cached = load_metric_set(cached_set)
            if cached is not None and (metrics is None or not metrics <= cached):
                raise RuntimeError(
                    f"{out_json} holds metric set {cached_set!r}, which lacks keys of "
                    f"{config.eval_metric_set!r}; set overwrite_noise_ceiling to rescore it."
                )
            logger.info(
                "Noise-ceiling cache hit; skip %s (overwrite_noise_ceiling to rerun).",
                out_json,
            )
            written[split] = out_json
            continue

        work_queue = load_work_queue_or_none(
            condition_summary_parquet=manifest.condition_summary,
            split_yaml=split_yaml,
            eval_split=split,
            condition_variables=mcv,
            control_condition=control_condition,
            metadata_joins=metadata_joins,
            bundle=None,
            min_cells=min_cells,
            control_match_keys=list(config.control_match_keys or []) or None,
            drop_control_rows=drop_control_rows,
            max_conditions=max_conditions,
        )
        if work_queue is None:
            logger.info(
                "Noise-ceiling skipped for split=%r (0 eligible conditions).",
                split,
            )
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        results, skips = run_noise_ceiling(
            truth_dir=truth_dir,
            work_queue=work_queue,
            n_repeats=n_repeats,
            seed=seed,
            device=device,
            condition_variables=mcv,
            gene_arrays_parquet=out_dir / BOOTSTRAP_GENE_ARRAYS_FILENAME,
            eval_group_keys=eval_group_keys,
            ctrl_det_floors=(
                tuple(float(t) for t in ctrl_det_floors)
                if ctrl_det_floors else (0.10,)
            ),
            # log1p_slice_med_rgs needs the dataset's L_ref; run_noise_ceiling raises without it
            dataset=config.dataset or None,
            metrics=metrics,
            gene_sidecars=config.eval_gene_sidecars,
        )
        fields = {
            "eval_split": split,
            "strategy": sampler.name,
            "strategy_params": dict(sampler.strategy_params),
            "n_repeats": n_repeats,
            "seed": seed,
            "normalization_id": norm_id,
            "split_config_hash": split_cfg_hash,
            "split_config": str(config.split_config) if config.split_config else None,
            "eval_min_cells": min_cells,
            "condition_variables": mcv,
            "plot_aggregate_over": config.plot_aggregate_over,
            "control_condition": control_condition,
            "gene_set_hash": gsh,
            "n_genes": len(gene_names),
            "eval_cache_namespace": str(truth_dir),
            "gene_arrays_parquet": BOOTSTRAP_GENE_ARRAYS_FILENAME if config.eval_gene_sidecars else None,
            "metric_set": config.eval_metric_set,
            "gene_summary_pred_label": sampler.pred_label,
            "ctrl_det_floors": (
                list(float(t) for t in ctrl_det_floors)
                if ctrl_det_floors else [0.10]
            ),
        }
        write_summary(out_json, fields=fields, results=results, skips=skips)
        written[split] = out_json
        logger.info(
            "Noise-ceiling | split=%s strategy=%s conditions=%d -> %s",
            split, sampler.name, len(results), out_json,
        )
    return written
