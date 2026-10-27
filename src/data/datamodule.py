"""Tahoe-100M dataloader using scDataset and Lightning DataModule.

Each batch yields dicts of NumPy arrays:
  - ``<var>_idx`` int64 for every generator input var
    (``condition_input_variables``, default all ``model_conditioning_variables``)
  - lib_size float32
  - gene_expression float32 dense (or SciPy CSR with ``keep_sparse``)
The LightningModule converts to torch tensors and tiles by ``n_repeats``.

The rest of this package, read in the order a row travels:
  * ``splits/``           who is train, validation, test
  * ``sampling/``         which conditions and which cells the main loader sees
  * ``ablations/``        cell-set train loader (C conditions × ragged cells)
  * ``cache/``            mmap CSR of the HVG matrix
  * ``vocab.py``          dose and plate vocabulary
  * ``data_utils.py``     eval-cache paths, manifests, tensor helpers

This module focuses on ``PertGenDataModule`` and ``setup()`` orchestration.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import yaml
from anndata.experimental import AnnCollection
from scdataset import BlockShuffling, MultiIndexable, scDataset
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from ..config.config import PertGenConfig, resolve_cache_dir, resolve_eval_cache_dir
from .vocab import (
    aggregate_condition_summary,
    concentrations_from_obs as _concentrations_from_obs,
    obs_values_to_categorical,
)
from .splits.data_class import HBSSSpec, SplitConfig
from .data_utils import (
    EvalManifest,
    _ensure_obs_columns_categorical,
    _fast_fit_label_encoder,
    _numpy_batch_to_tensor,
    enrich_obs_with_metadata,
    gene_set_hash,
    load_hvg_gene_names_from_file,
    load_hvg_indices_from_genes_file,
    log_split_setup_summary,
    write_eval_manifest,
)
from .sampling.hbss_sampling import (
    HBSSResult,
    encode_hbss_obs,
    run_hbss,
    selected_combined_keys,
)
from .cache.raw_hvg_cache import (
    TRANSFORM_TAG,
    HvgCacheCollection,
    RawHvgShardCache,
    normalization_id_from_transform,
)
from .splits.predicates import (
    apply_split_predicate,
    build_predicate_column_views,
    split_membership_masks,
)
from .splits.coverage import write_split_coverage_artifacts

logger = logging.getLogger(__name__)


__all__ = [
    "FetchTransform",
    "PertGenDataModule",
    "apply_split_predicate",
    "build_predicate_column_views",
    "enrich_obs_with_metadata",
    "make_fetch_transform",
    "split_membership_masks",
]

# Workdir subdir for train/val/test indices (+ HBSS sidecars). Rank 0 writes;
# other DDP ranks wait and load so predicates/HBSS run once per workdir.
_SPLIT_CACHE_DIRNAME = "split_cache"
_SPLIT_CACHE_READY = "READY"
_SPLIT_CACHE_WAIT_S = 3_600.0
_SPLIT_CACHE_POLL_S = 2.0


def _ddp_world_and_rank() -> Tuple[int, int]:
    """``(world_size, rank)`` from env (works before ``init_process_group``)."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    return world, rank


def _split_cache_fingerprint(
    *,
    split_config_path: Optional[str],
    split_cfg: SplitConfig,
    control_condition: Optional[Dict[str, Any]],
    n_obs: int,
) -> str:
    """Stable id for a split+HBSS+cell-budget result over a fixed collection size."""
    path = Path(split_config_path) if split_config_path else None
    if path is not None and not path.is_file():
        raise FileNotFoundError(f"split_config {split_config_path!r} does not exist (moved? see core_util_data/README.md)")
    if path is not None:
        split_sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        split_path = str(path.resolve())
    else:
        split_sha = "missing"
        split_path = ""
    payload = {
        "split_config_path": split_path,
        "split_config_sha": split_sha,
        "n_obs": int(n_obs),
        "control_condition": control_condition,
        "train_sample": split_cfg.train.sample,
        "train_fraction": split_cfg.train.fraction,
        "hbss": (
            None
            if split_cfg.train.hbss is None
            else {
                "balance_axes": list(split_cfg.train.hbss.balance_axes),
                "candidate_axis": split_cfg.train.hbss.candidate_axis,
                "diversity_axes": dict(split_cfg.train.hbss.diversity_axes),
                "lam": float(split_cfg.train.hbss.lam),
                "seed": int(split_cfg.train.hbss.seed),
                "max_iter": int(split_cfg.train.hbss.max_iter),
                "tol": float(split_cfg.train.hbss.tol),
                "ambiguous_diversity_to_singleton": bool(
                    split_cfg.train.hbss.ambiguous_diversity_to_singleton
                ),
            }
        ),
        "cells_target_n": split_cfg.train.cells.target_n,
        "cells_seed": int(split_cfg.train.cells.seed),
        "controls_to_train": bool(split_cfg.controls_to_train),
    }
    if split_cfg.train.cells.keep_controls:   # only when set, so shipped split caches keep their ids
        payload["cells_keep_controls"] = True
    if split_cfg.train.hbss is not None and split_cfg.train.hbss.tie_break != "legacy":   # same: legacy keeps shipped ids
        payload["hbss_tie_break"] = split_cfg.train.hbss.tie_break
    if split_cfg.train.cells.per_condition is not None:  # same: absent key, shipped fingerprints
        payload["cells_per_condition"] = int(split_cfg.train.cells.per_condition)
        payload["cells_group_by"] = list(split_cfg.train.cells.group_by)
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:32]


def _split_cache_dir(workdir: Path) -> Path:
    return Path(workdir).resolve() / _SPLIT_CACHE_DIRNAME


def _try_load_split_cache(
    cache_dir: Path,
    fingerprint: str,
) -> Optional[Dict[str, Any]]:
    """Return cached split payload if READY + fingerprint match; else None."""
    ready = cache_dir / _SPLIT_CACHE_READY
    meta_path = cache_dir / "meta.json"
    if not ready.is_file() or not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text())
    if str(meta.get("fingerprint", "")) != fingerprint:
        return None
    train_p = cache_dir / "train_idx.npy"
    val_p = cache_dir / "val_idx.npy"
    test_p = cache_dir / "test_idx.npy"
    if not (train_p.is_file() and val_p.is_file() and test_p.is_file()):
        return None
    out: Dict[str, Any] = {
        "train_idx": np.load(train_p).astype(np.int64, copy=False),
        "val_idx": np.load(val_p).astype(np.int64, copy=False),
        "test_idx": np.load(test_p).astype(np.int64, copy=False),
        "meta": meta,
        "hbss_summary": meta.get("hbss_summary"),
        "cells_summary": meta.get("cells_summary"),
    }
    pert_p = cache_dir / "pert_train_idx_pre_hbss.npy"
    if pert_p.is_file():
        out["pert_train_idx_pre_hbss"] = np.load(pert_p).astype(np.int64, copy=False)
    selected_p = cache_dir / "hbss_selected.parquet"
    if selected_p.is_file():
        out["hbss_selected"] = pd.read_parquet(selected_p)
    hbss_meta_p = cache_dir / "hbss_meta.json"
    if hbss_meta_p.is_file():
        out["hbss_meta"] = json.loads(hbss_meta_p.read_text())
    report_p = cache_dir / "hbss_balance_report.csv"
    if report_p.is_file():
        out["hbss_balance_report"] = pd.read_csv(report_p)
    return out


def _wait_for_split_cache(cache_dir: Path, fingerprint: str) -> Dict[str, Any]:
    """Poll until rank 0 finishes writing the split cache (file-based barrier)."""
    print(
        f"  Data: split — waiting for rank-0 cache at {cache_dir} "
        f"(fingerprint={fingerprint[:8]}…)",
        flush=True,
    )
    t0 = time.monotonic()
    while time.monotonic() - t0 < _SPLIT_CACHE_WAIT_S:
        loaded = _try_load_split_cache(cache_dir, fingerprint)
        if loaded is not None:
            print(
                f"  Data: split — loaded rank-0 cache "
                f"[{time.monotonic() - t0:.1f}s wait]",
                flush=True,
            )
            return loaded
        time.sleep(_SPLIT_CACHE_POLL_S)
    raise TimeoutError(
        f"Timed out after {_SPLIT_CACHE_WAIT_S:.0f}s waiting for split cache "
        f"at {cache_dir} (fingerprint={fingerprint}). "
        "Rank 0 may have failed during predicates/HBSS."
    )


def _write_split_cache(
    cache_dir: Path,
    *,
    fingerprint: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    pert_train_idx_pre_hbss: Optional[np.ndarray],
    hbss_summary: Optional[Dict[str, Any]],
    cells_summary: Optional[Dict[str, Any]],
    hbss_selected: Optional[pd.DataFrame],
    hbss_meta: Optional[Dict[str, Any]],
    hbss_balance_report: Optional[pd.DataFrame],
) -> None:
    """Atomically publish split indices (READY marker written last)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    ready = cache_dir / _SPLIT_CACHE_READY
    if ready.exists():
        ready.unlink()

    np.save(cache_dir / "train_idx.npy", np.asarray(train_idx, dtype=np.int64))
    np.save(cache_dir / "val_idx.npy", np.asarray(val_idx, dtype=np.int64))
    np.save(cache_dir / "test_idx.npy", np.asarray(test_idx, dtype=np.int64))
    if pert_train_idx_pre_hbss is not None:
        np.save(
            cache_dir / "pert_train_idx_pre_hbss.npy",
            np.asarray(pert_train_idx_pre_hbss, dtype=np.int64),
        )
    if hbss_selected is not None:
        hbss_selected.to_parquet(cache_dir / "hbss_selected.parquet", index=False)
    if hbss_meta is not None:
        (cache_dir / "hbss_meta.json").write_text(
            json.dumps(hbss_meta, indent=2, default=str),
        )
    if hbss_balance_report is not None:
        hbss_balance_report.to_csv(
            cache_dir / "hbss_balance_report.csv", index=False,
        )
    meta = {
        "fingerprint": fingerprint,
        "n_train": int(train_idx.size),
        "n_val": int(val_idx.size),
        "n_test": int(test_idx.size),
        "hbss_summary": hbss_summary,
        "cells_summary": cells_summary,
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    ready.write_text(fingerprint + "\n")
    print(
        f"  Data: split — wrote cache → {cache_dir} "
        f"(train={train_idx.size:,}, fingerprint={fingerprint[:8]}…)",
        flush=True,
    )


def _hbss_kept_perturbed_indices(
    pert_train_idx: np.ndarray,
    obs_eval: pd.DataFrame,
    hbss_spec: HBSSSpec,
    fraction: float,
) -> Tuple[np.ndarray, HBSSResult]:
    """Run HBSS on perturbed-train rows and return surviving indices + diagnostics.

    Reserved separately from the predicate engine so the control rows are kept
    verbatim and only the perturbed rows are stratified (see split docs §2.3).
    """
    if pert_train_idx.size == 0:
        raise ValueError("HBSS: pert_train_idx is empty; nothing to sample.")

    t0 = time.monotonic()
    print(
        f"  HBSS: encoding {pert_train_idx.size:,} perturbed train rows...",
        flush=True,
    )
    encoding = encode_hbss_obs(
        obs_eval,
        hbss_spec.balance_axes,
        hbss_spec.diversity_axes,
        row_indices=pert_train_idx,
    )
    t_encode = time.monotonic() - t0
    print(
        f"  HBSS: unique triplets = {len(encoding.triplets):,} ({t_encode:.1f}s)",
        flush=True,
    )

    # run_hbss needs a DataFrame for API compat; encoding is precomputed.
    sub_obs = obs_eval.iloc[pert_train_idx]
    result = run_hbss(
        sub_obs,
        hbss_spec,
        fraction=fraction,
        encoding=encoding,
    )

    t_filter0 = time.monotonic()
    allowed = selected_combined_keys(result.selected, encoding)
    keep_mask = np.isin(encoding.combined, allowed)
    kept = pert_train_idx[keep_mask].astype(np.int64, copy=False)
    t_filter = time.monotonic() - t_filter0
    print(
        f"  HBSS: filtering {pert_train_idx.size:,} rows -> "
        f"{kept.size:,} kept ({t_filter:.1f}s)",
        flush=True,
    )

    if result.meta["n_selected"] > 0 and kept.size == 0:
        raise ValueError(
            f"HBSS selected {result.meta['n_selected']} unique tuple(s) but zero "
            "perturbed train rows survived the tuple intersection. Likely a "
            "balance_axes key-policy mismatch with "
            "vocab.obs_column_to_canonical_codes.",
        )
    n_pert = int(pert_train_idx.size)
    result.meta.update({
        "n_train_rows_pert_before": n_pert,
        "n_train_rows_pert_after": int(kept.size),
        "row_coverage_pert": (float(kept.size) / n_pert) if n_pert else 0.0,
        "wall_seconds": {
            **result.meta.get("wall_seconds", {}),
            "row_filter": round(float(t_filter), 4),
        },
    })
    return kept, result


# One split pipeline. Training and scripts/plot/split_qc.py both call it, so a
# coverage plot can never describe a different train set than the one the model saw.

def compute_train_split(
    obs: pd.DataFrame,
    split_cfg: SplitConfig,
    *,
    metadata_joins: Any = (),
    control_condition: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Row indices of one split, from the full ``obs`` and its split YAML.

    1. Enrich ``obs`` with metadata parquets so predicate columns (Organ, moa-fine) exist.
    2. Apply val/test predicates; controls return to train when ``controls_to_train`` is set.
    3. ``train.sample == "hbss"``: controls are kept verbatim; only perturbed train rows
       are stratified.
    4. ``train.cells``: either a uniform ``target_n`` cap, or ``per_condition`` cells from
       each ``group_by`` condition. ``keep_controls`` leaves every control row out of the cap.

    Returns ``obs_eval``, ``train_idx``, ``val_idx``, ``test_idx``, ``pert_train_idx_pre_hbss``,
    ``hbss_result`` (None without HBSS), ``hbss_summary`` and ``cells_summary``.
    """
    from types import SimpleNamespace

    t = time.monotonic()
    obs_eval = enrich_obs_with_metadata(obs, SimpleNamespace(metadata_joins=list(metadata_joins or ())))
    print(f"  Data: split — metadata enriched (n_obs={len(obs_eval):,}, cols={len(obs_eval.columns)}) "
          f"[{time.monotonic() - t:.1f}s]", flush=True)

    t = time.monotonic()
    is_train, is_val, is_test, control_mask = split_membership_masks(obs_eval, split_cfg, control_condition)
    train_idx, val_idx, test_idx = (np.where(m)[0].astype(np.int64) for m in (is_train, is_val, is_test))
    print(f"  Data: split — predicates applied (train={train_idx.size:,}, val={val_idx.size:,}, "
          f"test={test_idx.size:,}) [{time.monotonic() - t:.1f}s]", flush=True)
    if train_idx.size == 0:
        raise ValueError("Split produced an empty train set. Check validation/test "
                         "predicates and controls_to_train in the split config.")
    pert_train_idx_pre_hbss = train_idx[~control_mask[train_idx]].astype(np.int64, copy=False)

    hbss_result, hbss_summary = None, None
    if split_cfg.train.sample == "hbss":
        if split_cfg.train.hbss is None or split_cfg.train.fraction is None:
            raise ValueError("train.sample='hbss' requires train.fraction and a train.hbss block.")
        ctrl_in_train = control_mask[train_idx]
        pert_idx, ctrl_idx = train_idx[~ctrl_in_train], train_idx[ctrl_in_train]
        kept_pert, hbss_result = _hbss_kept_perturbed_indices(
            pert_idx, obs_eval, split_cfg.train.hbss, float(split_cfg.train.fraction),
        )
        train_idx = np.sort(np.concatenate([kept_pert, ctrl_idx])).astype(np.int64, copy=False)
        m = hbss_result.meta
        hbss_summary = {
            "n_triplets_total": m["n_total_triplets"], "n_triplets_kept": m["n_selected"],
            "n_train_rows_pert_before": m["n_train_rows_pert_before"],
            "n_train_rows_pert_after": m["n_train_rows_pert_after"],
            "n_train_rows_ctrl_kept": int(ctrl_idx.size),
            "ipf_converged": m["ipf_converged"], "ipf_iters": m["ipf_iters"],
        }

    cells_summary = None
    cells = split_cfg.train.cells
    keep = control_mask[train_idx] if cells.keep_controls else None
    if cells.target_n is not None:
        from src.data.sampling.cell_subsampling import subsample_train_cells

        train_idx, cells_summary = subsample_train_cells(
            train_idx, target_n=cells.target_n, seed=cells.seed, keep_mask=keep,
        )
    elif cells.per_condition is not None:
        from src.data.sampling.cell_subsampling import subsample_per_condition

        missing = [c for c in cells.group_by if c not in obs_eval.columns]
        if missing:
            raise ValueError(f"train.cells.group_by columns missing from obs: {missing}")
        # One id per condition, in train-row order. Controls are masked out inside the draw.
        grouped = obs_eval.iloc[train_idx].groupby(
            list(cells.group_by), sort=False, observed=True).ngroup().to_numpy()
        train_idx, cells_summary = subsample_per_condition(
            train_idx, grouped, n_per=cells.per_condition, seed=cells.seed, keep_mask=keep,
        )
    return {
        "obs_eval": obs_eval, "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
        "pert_train_idx_pre_hbss": pert_train_idx_pre_hbss, "hbss_result": hbss_result,
        "hbss_summary": hbss_summary, "cells_summary": cells_summary,
    }


def _prefetch_factor_for_loader(*, num_workers: int, prefetch_factor: int) -> Optional[int]:
    """Resolve DataLoader ``prefetch_factor`` from config (independent of fetch_factor).

    PyTorch requires ``prefetch_factor is None`` when ``num_workers == 0``.
    """
    if int(num_workers) <= 0:
        return None
    pf = int(prefetch_factor)
    if pf < 1:
        raise ValueError(f"prefetch_factor must be >= 1 when num_workers>0, got {pf}")
    return pf


# ---------------------------------------------------------------------------
# Batch transform
# ---------------------------------------------------------------------------


def _is_identity_indices(indices: np.ndarray, n_cols: int) -> bool:
    """Return True iff ``indices`` is ``arange(n_cols)`` (no permutation, no drop).

    Used in the fetch transform to skip column subsetting when the cache is
    already HVG-aligned with the requested HVG file. ``np.array_equal``
    short-circuits on the shape mismatch before doing any element compare;
    for the matching case it is ~5 µs on a 4K-element array — negligible
    against the milliseconds of CSR work per fetch.
    """
    if indices.shape[0] != n_cols:
        return False
    return bool(
        np.array_equal(indices, np.arange(n_cols, dtype=indices.dtype)),
    )


# Per-process fetch counter for debug instrumentation (FetchTransform is frozen).


@dataclass(frozen=True)
class FetchTransform:
    """Picklable per-fetch transform used by ``scDataset``.

    Runs once per fetched chunk (``batch_size × fetch_factor`` cells), so
    encoding / sparse-densify / HVG subset / log1p are amortised over the
    whole chunk rather than redone per mini-batch. ``hvg_indices`` slices
    columns *before* densifying so we never allocate the full
    ``(chunk × n_vars)`` dense block.

    Module-level (not a closure) so it round-trips through ``pickle`` — that
    in turn lets the DataLoader use ``multiprocessing_context="spawn"`` for
    backends that don't need fork (e.g. the mmap CSR cache). The semantics
    are identical to the previous ``make_fetch_transform`` closure.

    Attributes:
        encoders: Per-conditioning-variable :class:`sklearn.preprocessing.LabelEncoder`.
        hvg_indices: Optional column indices to subset to before densifying.
        apply_log1p: Apply ``log1p`` *somewhere* in the pipeline.
        keep_sparse: Yield a CSR ``gene_expression`` instead of dense float32.
        defer_log1p_to_gpu: When True, skip the CPU log1p so the Lightning
            module can apply ``torch.log1p`` on device.
    """

    encoders: Mapping[str, LabelEncoder]
    hvg_indices: Optional[np.ndarray] = None
    apply_log1p: bool = False
    keep_sparse: bool = False
    defer_log1p_to_gpu: bool = False

    @property
    def cpu_log1p(self) -> bool:
        # CPU log1p only when log1p was requested and we are *not* deferring
        # it to the GPU pre-loss.
        return bool(self.apply_log1p) and not self.defer_log1p_to_gpu

    def __call__(self, chunk: Any) -> Any:
        # Materialise the chunk view as an in-memory AnnData; .X may be CSR
        # (cache path) or dense (h5ad path after backed read).
        adata = chunk.to_adata()
        X, obs = adata.X, adata.obs

        # Encoders only fire for variables that exist on this chunk's obs;
        # missing variables are skipped silently so backends with subset obs
        # don't crash here.
        data: Dict[str, Any] = {
            f"{var}_idx": enc.transform(
                np.asarray(obs[var].astype(str).values),
            ).astype(np.int64)
            for var, enc in self.encoders.items()
            if var in obs.columns
        }

        # When the dataset is *already* HVG-only (mmap CSR caches always are)
        # the resolved ``hvg_indices`` is just ``arange(n_vars)``. Skipping
        # this no-op slice is both a perf win (no scipy fancy-index pass over
        # a ~35M-nnz uint16 CSR per fetch) and dodges the scipy
        # ``_minor_index_fancy`` crash that fires on large sliced CSRs.
        needs_hvg_slice = self.hvg_indices is not None and not _is_identity_indices(
            self.hvg_indices, X.shape[1],
        )

        if self.keep_sparse and sparse.issparse(X):
            gene_expr = (X[:, self.hvg_indices] if needs_hvg_slice else X).tocsr()
            lib_size = np.asarray(gene_expr.sum(axis=1)).ravel().astype(np.float32)
        else:
            if sparse.issparse(X):
                X_sub = X[:, self.hvg_indices] if needs_hvg_slice else X
                # Densify FIRST (single scipy C loop into a uint16/float32
                # dense buffer) then cast the dense buffer to float32 if
                # needed. The reverse order — ``astype(float32).toarray()``
                # — allocates a float32-data CSR intermediate (~140 MB on a
                # 35M-nnz fetch) and takes ~10× longer than the
                # contiguous dense cast (~2 s vs ~160 ms in benchmarks).
                dense = X_sub.toarray()
                gene_expr = (
                    dense if dense.dtype == np.float32
                    else dense.astype(np.float32, copy=False)
                )
            else:
                X = np.asarray(X, dtype=np.float32)
                gene_expr = (
                    X[:, self.hvg_indices]
                    if self.hvg_indices is not None
                    else X.copy()
                )
            lib_size = gene_expr.sum(axis=-1).astype(np.float32)
            if self.cpu_log1p:
                # in-place log1p avoids a second dense allocation.
                np.log1p(gene_expr, out=gene_expr)

        data["gene_expression"] = gene_expr
        data["lib_size"] = lib_size  # sum of the HVG counts in this row
        return MultiIndexable(data)


def make_fetch_transform(
    encoders: Dict[str, LabelEncoder],
    hvg_indices: Optional[np.ndarray] = None,
    apply_log1p: bool = False,
    keep_sparse: bool = False,
    defer_log1p_to_gpu: bool = False,
) -> FetchTransform:
    """Back-compat shim returning a :class:`FetchTransform`.

    Kept so external callers / tests that imported the closure factory keep
    working; new code should construct :class:`FetchTransform` directly.
    """
    return FetchTransform(
        encoders=encoders,
        hvg_indices=hvg_indices,
        apply_log1p=apply_log1p,
        keep_sparse=keep_sparse,
        defer_log1p_to_gpu=defer_log1p_to_gpu,
    )


# ---------------------------------------------------------------------------
# Lightning DataModule
# ---------------------------------------------------------------------------


class PertGenDataModule(pl.LightningDataModule):
    """Lightning DataModule for Tahoe-100M via scDataset.

    Loads h5ad files (or the mmap CSR cache when present), runs splits through
    :func:`split_membership_masks`, fits per-variable label encoders, derives
    HVG indices, builds a chunk-level fetch transform and persists the eval
    manifest (condition summary + cache namespace) under ``<workdir>/eval/``.

    Attributes populated by ``setup()`` for the model: ``encoders``,
    ``n_categories``, ``_hvg_indices``, ``_gene_names``, ``_n_genes``,
    ``_train_idx`` / ``_val_idx`` / ``_test_idx``, ``_eval_manifest``.
    """

    def __init__(self, config: PertGenConfig, workdir: Union[str, Path]) -> None:
        super().__init__()
        self.config = config
        # ``batch_size`` is a thin pass-through to ``config.batch_size``;
        # see the property below, so a write to ``dm.batch_size`` reaches
        # ``train_dataloader``.
        self._workdir = Path(workdir).resolve()
        self.split_config: SplitConfig = self.load_split_config(config.split_config)
        self.dataset = config.dataset
        self.data_dir = Path(config.data_dir) if config.data_dir is not None else None
        self.cache_dir = resolve_cache_dir(config)
        # ``cache_dir`` still tracks where the *eval cache* lives so the eval
        # manifest can point at it; ``_use_train_cache`` toggles whether the
        # training DataLoader pulls from the cache or from raw h5ad files.
        self._use_train_cache = bool(getattr(config, "use_train_cache", True))
        if not self._use_train_cache and self.cache_dir is not None:
            print(
                f"  [use_train_cache=false] Bypassing train cache at {self.cache_dir}; "
                "loading training data from h5ad files directly "
                "(cache_dir retained for eval manifest).",
                flush=True,
            )
        self.hvg_genes_path = (
            Path(config.hvg_genes_path) if config.hvg_genes_path is not None else None
        )
        self.model_conditioning_variables = config.model_conditioning_variables
        self._collection = None
        self._hvg_cache: Optional[RawHvgShardCache] = None
        # Populated when ``train.sample == "hbss"``; persisted at end of setup().
        self._hbss_balance_report: Optional[pd.DataFrame] = None
        self._hbss_meta: Optional[Dict[str, Any]] = None
        self._hbss_selected: Optional[pd.DataFrame] = None
        self._condition_summary: Optional[pd.DataFrame] = None
        self._condition_summary_path: Optional[Path] = None
        self._obs_eval: Optional[pd.DataFrame] = None
        self._pert_train_idx_pre_hbss: Optional[np.ndarray] = None
        self._eval_manifest: Optional[EvalManifest] = None
        self._eval_cache_normalization_id: Optional[str] = None
        # Populated when ``train_loader_mode == "cell_set"``.
        self._cell_set_dataset = None

    # --- Public properties ---

    @property
    def condition_summary_path(self) -> Optional[Path]:
        return self._condition_summary_path

    @property
    def eval_manifest(self) -> EvalManifest:
        if self._eval_manifest is None:
            raise RuntimeError(
                "eval_manifest is set in setup() after eval namespace resolve + manifest write.",
            )
        return self._eval_manifest

    @property
    def train_indices(self) -> np.ndarray:
        return self._train_idx

    @property
    def val_indices(self) -> np.ndarray:
        return self._val_idx

    @property
    def test_indices(self) -> np.ndarray:
        return self._test_idx

    def setup(self, stage: Optional[str] = None) -> None:
        if self._collection is not None:
            return

        t0 = time.monotonic()
        self._load_collection()

        print(f"  Data: computing splits (n_obs={self._collection.n_obs:,})...", flush=True)
        self._train_idx, self._val_idx, self._test_idx, _ = self._compute_splits(
            self._collection.obs,
        )

        # --- encoders (one LabelEncoder per generator INPUT var) ---
        # Eval identity stays on full MCV (condition summary, NPZ keys). Only
        # ``condition_input_variables`` become embeddings / *_idx in the batch.
        from ..config import resolve_condition_input_variables

        input_vars = resolve_condition_input_variables(self.config)
        mcv = list(self.config.model_conditioning_variables or [])
        if input_vars != mcv:
            print(
                f"  Data: condition INPUT {input_vars} "
                f"(eval identity still {mcv})",
                flush=True,
            )
        print(
            f"  Data: fitting encoders ({len(input_vars)} vars)...",
            flush=True,
        )
        t_enc = time.monotonic()
        self.encoders, self.n_categories = {}, {}
        for var in input_vars:
            if var not in self._collection.obs.columns:
                print(f" Warning: {var} not found in collection. Skipping encoder.", flush=True)
                continue
            le = _fast_fit_label_encoder(self._collection.obs[var])
            self.encoders[var] = le
            self.n_categories[var] = len(le.classes_)
        stats = " ".join(f"{k}={v}" for k, v in self.n_categories.items())
        print(f" Data: encoders fitted ({stats}) [{time.monotonic() - t_enc:.2f}s]", flush=True)

        # --- HVG resolution: optional gene list file → indices into var_names ---
        print("  Data: resolving HVG gene indices...", flush=True)
        t_hvg = time.monotonic()
        if self.hvg_genes_path is not None and self.hvg_genes_path.is_file():
            var_names = list(self._collection.var_names)
            self._hvg_indices, self._gene_names = load_hvg_indices_from_genes_file(
                self.hvg_genes_path, var_names,
            )
            source_msg = f"loaded from {self.hvg_genes_path.name}"
        else:
            self._hvg_indices = np.arange(self._collection.n_vars, dtype=np.int64)
            self._gene_names = [str(x) for x in self._collection.var_names]
            source_msg = "using all genes (no file)"
        self._n_genes = len(self._hvg_indices)
        self._n_genes_full = int(self._collection.n_vars)
        print(
            f"  HVG: {source_msg} -> {self._n_genes} genes "
            f"({time.monotonic() - t_hvg:.1f}s)",
            flush=True,
        )

        # --- fetch transform (per-chunk, encoders+HVG+log1p amortised) ---
        print("  Data: building fetch transform...", flush=True)
        t_fetch = time.monotonic()
        # log1p runs on the GPU — cheaper than a CPU pass over the dense chunk.
        self._fetch_transform = make_fetch_transform(
            self.encoders,
            hvg_indices=self._hvg_indices,
            apply_log1p=self.config.apply_log1p,
            keep_sparse=False,
            defer_log1p_to_gpu=bool(self.config.apply_log1p),
        )
        self._batch_transform = None
        print(f"  Data: fetch transform ready [{time.monotonic() - t_fetch:.1f}s]", flush=True)

        print("  Data: building condition summary...", flush=True)
        self._build_and_save_condition_summary()

        self.config.num_output_genes = int(self._n_genes)
        self.config.n_categories = dict(self.n_categories)

        # --- encoder + gene-name sidecar (consumed by eval scripts) ---
        enc_dir = self._workdir
        out_genes = (
            [str(x) for x in self._collection.var_names]
            if self.hvg_genes_path is None
            else [str(self._collection.var_names[int(i)]) for i in self._hvg_indices]
        )
        print("  Data: saving encoder mappings...", flush=True)
        t_save = time.monotonic()
        enc_dir.mkdir(parents=True, exist_ok=True)
        (enc_dir / "encoder_idx_to_name.json").write_text(json.dumps(
            {
                var: [str(x) for x in le.classes_]
                for var, le in self.encoders.items()
            },
            indent=2,
        ))
        (enc_dir / "output_gene_names.txt").write_text(
            "\n".join(out_genes) + ("\n" if out_genes else ""),
        )
        print(
            f"  Data: encoder mappings + {len(out_genes)} output genes saved "
            f"(dir={enc_dir}) [{time.monotonic() - t_save:.1f}s]",
            flush=True,
        )

        if self._hbss_balance_report is not None:
            self._hbss_balance_report.to_csv(
                enc_dir / "hbss_balance_report.csv", index=False,
            )
            (enc_dir / "hbss_meta.json").write_text(
                json.dumps(self._hbss_meta, indent=2, default=str),
            )
            self._hbss_selected.to_parquet(
                enc_dir / "hbss_selected_triplets.parquet",
            )
            print(f"  Data: HBSS report saved (dir={enc_dir})", flush=True)

        if self.cache_dir is None:
            raise RuntimeError("Eval manifest requires cache_dir (set data_dir + hvg_genes_path).")

        # normalization id determines which eval cache namespace the manifest points at
        if self._hvg_cache is not None:
            transform = str(self._hvg_cache.manifest.get("transform", TRANSFORM_TAG))
            norm_id = normalization_id_from_transform(transform)
            print(
                f"  Data: normalization id={norm_id} "
                f"(from cache manifest transform={transform!r})",
                flush=True,
            )
        else:
            # h5ad AnnCollection stores raw counts; log1p runs on the GPU.
            norm_id = "raw_counts"
            print(f"  Data: normalization id={norm_id} (h5ad path, no cache manifest)", flush=True)
        self._eval_cache_normalization_id = norm_id

        self._eval_manifest = EvalManifest(
            condition_summary=self._condition_summary_path,
            eval_cache_dir=resolve_eval_cache_dir(self.config, normalization_id=str(norm_id)),
            group_cols=tuple(self.config.model_conditioning_variables or ()),
            gene_set_hash=gene_set_hash(self._gene_names),
            n_genes=len(self._gene_names),
            split_config=str(self.config.split_config) if self.config.split_config else None,
            normalization_id=norm_id,
        )
        print("  Data: writing eval manifest...", flush=True)
        t_manifest = time.monotonic()
        np.save(
            self._condition_summary_path.parent / "hvg_indices.npy",
            np.asarray(self._hvg_indices, dtype=np.int64),
        )
        write_eval_manifest(self._workdir, self._eval_manifest)
        print(
            f"  Data: eval manifest written [{time.monotonic() - t_manifest:.1f}s]",
            flush=True,
        )

        if getattr(self.config, "save_split_coverage", True) and self._obs_eval is not None:
            from ..config.dataset_registry import load_dataset_registry

            print("  Data: writing split coverage artifacts...", flush=True)
            t_cov = time.monotonic()
            registry = load_dataset_registry(self.config.dataset)
            png_path, json_path = write_split_coverage_artifacts(
                self._workdir,
                full_obs=self._collection.obs,
                obs_eval=self._obs_eval,
                train_idx=self._train_idx,
                pert_train_idx_pre_hbss=self._pert_train_idx_pre_hbss,
                split_cfg=self.split_config,
                bio_columns=registry.biological_context_columns,
                pert_columns=registry.perturbation_columns,
                control_pert=registry.control_pert,
                dataset_name=str(self.config.dataset),
                split_config_path=str(self.config.split_config or ""),
                max_pert_rows=getattr(self.config, "split_coverage_max_pert_rows", None),
                raise_on_axis_failure=getattr(self.config, "raise_on_axis_coverage_failure", True),
            )
            print(
                f"  Data: split coverage saved ({png_path.name}, {json_path.name}) "
                f"[{time.monotonic() - t_cov:.1f}s]",
                flush=True,
            )
            self._obs_eval = None
            self._pert_train_idx_pre_hbss = None

        # Drop obs columns workers don't need (~1.6 GB of int64 columns) so
        # the pickled cache shipped to each spawn worker stays small. We keep
        # the conditioning columns the FetchTransform looks up. Disable via
        # ``config.skip_obs_trim=true`` if any consumer needs the full obs.
        if self._hvg_cache is not None and not getattr(
            self.config, "skip_obs_trim", False,
        ):
            t_trim = time.monotonic()
            before_bytes = int(self._hvg_cache.obs.memory_usage(deep=True).sum())
            self._hvg_cache.trim_obs_columns(
                tuple(self.config.model_conditioning_variables or ()),
            )
            after_bytes = int(self._hvg_cache.obs.memory_usage(deep=True).sum())
            print(
                f"  Data: trimmed obs columns "
                f"({before_bytes / 2**20:.0f} MB -> {after_bytes / 2**20:.0f} MB) "
                f"[{time.monotonic() - t_trim:.1f}s]",
                flush=True,
            )

        print(f"  Data: setup total ({time.monotonic() - t0:.1f}s)", flush=True)

    def _uses_spawn_dataloader(self) -> bool:
        return self._hvg_cache is not None

    def load_split_config(self, split_config_path: str) -> SplitConfig:
        """Parse + validate the split-config YAML."""
        return SplitConfig.from_dict(yaml.safe_load(Path(split_config_path).read_text()) or {})

    def _load_collection(self) -> None:
        """Open the mmap CSR cache, or the h5ad ``AnnCollection`` when it is absent."""
        if self._collection is not None:
            return

        if self.cache_dir is not None and self._use_train_cache:
            manifest = Path(self.cache_dir) / "manifest.json"
            if manifest.is_file():
                t0 = time.monotonic()
                print(
                    f"  Data: opening RawHvgShardCache at {self.cache_dir} (eager_obs=True)...",
                    flush=True,
                )
                cache = RawHvgShardCache(self.cache_dir, eager_obs=True)
                self._hvg_cache = cache
                self._collection = HvgCacheCollection(cache)
                self._n_genes = self._collection.n_vars

                # ``dose`` is derived from drugname_drugconc — not persisted in the
                # obs sidecar — so reconstruct it for encoder fitting + predicates.
                if (
                    "dose" not in cache.obs.columns
                    and "drugname_drugconc" in cache.obs.columns
                ):
                    t_dose = time.monotonic()
                    cache.obs["dose"] = _concentrations_from_obs(
                        cache.obs, key="drugname_drugconc",
                    )
                    unique = np.unique(cache.obs["dose"].to_numpy())
                    print(
                        f"  HvgCache: derived 'dose' from 'drugname_drugconc' "
                        f"({len(unique)} unique values: {unique.tolist()}) "
                        f"[{time.monotonic() - t_dose:.1f}s]",
                        flush=True,
                    )

                if self.hvg_genes_path and self.hvg_genes_path.is_file():
                    req = load_hvg_gene_names_from_file(self.hvg_genes_path)
                    cache_genes = set(cache.var_names)
                    missing = [g for g in req if g not in cache_genes]
                    if missing:
                        logger.warning(
                            "HVG genes file %s: %d/%d not in cache %s (%d vars); examples: %s",
                            self.hvg_genes_path, len(missing), len(req),
                            self.cache_dir, len(cache_genes), sorted(missing)[:10],
                        )
                    if not any(g in cache_genes for g in req):
                        raise ValueError(
                            f"HVG gene list {self.hvg_genes_path} lists no genes "
                            f"present in cache {self.cache_dir} ({len(cache_genes)} vars). "
                            f"Rebuild the cache or fix hvg_genes_path.",
                        )

                n_shards = len(getattr(cache, "shards", []) or [])
                print(
                    f"  Data: cache loaded from {self.cache_dir} "
                    f"(n_shards={n_shards}, n_obs={cache.n_obs:,}, "
                    f"n_hvg={self._collection.n_vars}) ({time.monotonic() - t0:.1f}s)",
                    flush=True,
                )
                return

        # H5AD fallback (no cache): mirror the cache's obs shape (categoricals
        # for conditioning columns + derived ``dose``) so downstream paths are
        # dtype-stable across the two backends.
        adatas: List[ad.AnnData] = []
        target_cols = self.config.model_conditioning_variables
        for p in Path(self.data_dir).glob(self.config.files_glob or "*.h5ad"):
            t_start = time.monotonic()
            a = ad.read_h5ad(p, backed="r")
            if "drugname_drugconc" in a.obs.columns:
                if not isinstance(a.obs["drugname_drugconc"].dtype, pd.CategoricalDtype):
                    a.obs["drugname_drugconc"] = a.obs["drugname_drugconc"].astype("category")
                a.obs["dose"] = _concentrations_from_obs(a.obs, key="drugname_drugconc")
            valid_cols = [c for c in target_cols if c in a.obs.columns]
            if "dose" in a.obs.columns and "dose" not in valid_cols:
                valid_cols = [*valid_cols, "dose"]
            a.obs = _ensure_obs_columns_categorical(a.obs[valid_cols].copy(), valid_cols)
            adatas.append(a)
            print(f" Loaded {p.name} (n={a.n_obs:,}) [{time.monotonic() - t_start:.1f}s]")

        self._collection = AnnCollection(adatas)
        self._n_genes = self._collection.n_vars

    def _apply_loaded_split_cache(
        self, loaded: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Dict[str, Any]]]:
        """Hydrate datamodule fields from a split-cache payload."""
        self._obs_eval = None  # coverage already written by the producer rank
        self._pert_train_idx_pre_hbss = loaded.get("pert_train_idx_pre_hbss")
        self._hbss_selected = loaded.get("hbss_selected")
        self._hbss_meta = loaded.get("hbss_meta")
        self._hbss_balance_report = loaded.get("hbss_balance_report")
        train_idx = loaded["train_idx"]
        val_idx = loaded["val_idx"]
        test_idx = loaded["test_idx"]
        hbss_summary = loaded.get("hbss_summary")
        cells_summary = loaded.get("cells_summary")
        log_split_setup_summary(
            split_s=0.0,
            split_cfg=self.split_config,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            hbss_summary=hbss_summary,
            cells_summary=cells_summary,
            hbss_balance_report=self._hbss_balance_report,
        )
        print(
            f"  Data: split — cache hit "
            f"(train={train_idx.size:,}, val={val_idx.size:,}, test={test_idx.size:,})",
            flush=True,
        )
        return train_idx, val_idx, test_idx, hbss_summary

    def _compute_splits(
        self, obs: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Dict[str, Any]]]:
        """Compute (train, val, test) indices via :func:`compute_train_split`.

        Tries the workdir ``split_cache/`` first (rank 0 writes; other DDP ranks wait).
        Returns ``(train_idx, val_idx, test_idx, hbss_summary)``;
        ``hbss_summary`` is ``None`` when ``train.sample == "all"``.
        """
        cfg = self.split_config
        cache_dir = _split_cache_dir(self._workdir)
        fingerprint = _split_cache_fingerprint(
            split_config_path=getattr(self.config, "split_config", None),
            split_cfg=cfg,
            control_condition=getattr(self.config, "control_condition", None),
            n_obs=int(len(obs)),
        )
        world, rank = _ddp_world_and_rank()

        cached = _try_load_split_cache(cache_dir, fingerprint)
        if cached is not None:
            return self._apply_loaded_split_cache(cached)

        # Non-zero ranks never recompute — wait for rank 0's cache publish.
        if world > 1 and rank != 0:
            cached = _wait_for_split_cache(cache_dir, fingerprint)
            return self._apply_loaded_split_cache(cached)

        t_split = time.monotonic()
        split = compute_train_split(
            obs, cfg,
            metadata_joins=getattr(self.config, "metadata_joins", None),
            control_condition=getattr(self.config, "control_condition", None),
        )
        self._obs_eval = split["obs_eval"]
        self._pert_train_idx_pre_hbss = split["pert_train_idx_pre_hbss"]
        if split["hbss_result"] is not None:
            r = split["hbss_result"]
            self._hbss_balance_report, self._hbss_meta, self._hbss_selected = r.balance_report, r.meta, r.selected
        train_idx, val_idx, test_idx = split["train_idx"], split["val_idx"], split["test_idx"]
        hbss_summary, cells_summary = split["hbss_summary"], split["cells_summary"]

        _write_split_cache(
            cache_dir,
            fingerprint=fingerprint,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            pert_train_idx_pre_hbss=self._pert_train_idx_pre_hbss,
            hbss_summary=hbss_summary,
            cells_summary=cells_summary,
            hbss_selected=self._hbss_selected,
            hbss_meta=self._hbss_meta,
            hbss_balance_report=self._hbss_balance_report,
        )

        log_split_setup_summary(
            split_s=time.monotonic() - t_split,
            split_cfg=self.split_config,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            hbss_summary=hbss_summary,
            cells_summary=cells_summary,
            hbss_balance_report=self._hbss_balance_report,
        )
        return train_idx, val_idx, test_idx, hbss_summary

    def _build_and_save_condition_summary(self) -> None:
        """Build per-group cell counts and write ``eval/condition_summary.parquet``."""
        t0 = time.monotonic()
        cols = list(self.config.model_conditioning_variables)
        missing = [c for c in cols if c not in self._collection.obs.columns]
        if missing:
            raise KeyError(
                f"Conditioning columns {missing} not in obs. "
                "Derive dose via concentrations_from_obs before summarising.",
            )
        # aggregate_condition_summary counts raveled int codes; ~2s for 100M
        # rows, vs minutes for a structured-array np.unique.
        out = enrich_obs_with_metadata(
            aggregate_condition_summary(self._collection.obs, cols), self.config,
        )
        path = self._workdir / "eval" / "condition_summary.parquet"
        self._condition_summary = out
        self._condition_summary_path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path, index=False)
        print(
            f"  Data: condition summary: {path} ({len(out)} groups) "
            f"[{time.monotonic() - t0:.1f}s]",
            flush=True,
        )

    # --- Batch size ---

    @property
    def batch_size(self) -> int:
        """Batch size used by the train ``DataLoader``. Fixed for the run."""
        return int(self.config.batch_size)

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        # Coerce + validate so callers that pass np.int64 / floats etc. don't
        # poison config.batch_size with a non-int.
        v = int(value)
        if v <= 0:
            raise ValueError(f"batch_size must be positive, got {value!r}")
        self.config.batch_size = v

    # --- Dataloaders ---

    @staticmethod
    def _worker_init_fn(worker_id: int) -> None:
        """Per-worker setup: thread clamp only.

        Clamp PyTorch threads to 1 so ``num_workers × omp_threads`` does not
        oversubscribe the CPU; OMP/BLAS env vars must be set before
        fork/spawn, ``torch.set_num_threads`` is the one knob safe post-fork.

        No ``setrlimit(RLIMIT_AS, ...)``: it caps *virtual* address space, which
        the cache loader's legitimate ~175 GiB of lazy file-backed mmaps exceed
        (OSError ENOMEM). The kernel OOM killer is the hard limit.
        """
        torch.set_num_threads(1)

    def train_dataloader(self) -> DataLoader:
        mode = str(getattr(self.config, "train_loader_mode", "per_cell")).strip().lower()
        if mode == "cell_set":
            from .ablations.cell_set_dataloader import cell_set_train_dataloader

            return cell_set_train_dataloader(self)
        if mode != "per_cell":
            raise ValueError(
                f"Unknown train_loader_mode={self.config.train_loader_mode!r}; "
                "expected 'per_cell' or 'cell_set'.",
            )

        # Make num_workers visible to _worker_init_fn (which is a staticmethod
        # so it can pickle cleanly under spawn). Env is inherited by both fork
        # and spawn workers, unlike attributes captured via closure.
        import os as _os
        _os.environ["PERTGEN_NUM_WORKERS"] = str(int(self.config.num_workers))

        strategy = BlockShuffling(
            block_size=max(1, self.config.block_size),
            indices=self._train_idx,
            drop_last=True,
        )
        ds_kw = dict(
            data_collection=self._collection,
            strategy=strategy,
            fetch_factor=self.config.fetch_factor,
            fetch_transform=self._fetch_transform,
            batch_transform=self._batch_transform,
            drop_last=True,
            seed=self.config.seed,
        )
        dataset = scDataset(batch_size=self.config.batch_size, **ds_kw)
        num_workers = int(self.config.num_workers)
        use_workers = num_workers > 0
        ctx: Optional[str] = (
            ("spawn" if self._uses_spawn_dataloader() else "fork") if use_workers else None
        )

        # Forking a process that already holds an initialized CUDA context gives
        # the child an unusable one. Spawn re-executes and is safe, so only the
        # *forking* backends have to give up their workers under distributed —
        # the cache and parquet backends spawn and keep theirs. (This replaces a
        # blanket "num_workers=0 under DDP" override, which also zeroed the
        # spawn-safe cache backend and left the loader single-threaded.)
        if use_workers and ctx == "fork" and dist.is_available() and dist.is_initialized():
            print(
                f"  [DDP] h5ad backend forks its workers; overriding "
                f"num_workers={num_workers} -> 0 (fork + initialized CUDA is unsafe)",
                flush=True,
            )
            num_workers = 0
            use_workers = False
            ctx = None
            # Keep the env the workers/monitor read in step with the real count.
            _os.environ["PERTGEN_NUM_WORKERS"] = "0"

        prefetch = _prefetch_factor_for_loader(
            num_workers=num_workers,
            prefetch_factor=int(self.config.prefetch_factor),
        )
        backend = "cache" if self._hvg_cache else "h5ad"
        print(
            f"  Data: DataLoader mp_start_method={ctx or 'n/a'} ({backend} backend; "
            f"num_workers={num_workers}; prefetch_factor={prefetch})",
            flush=True,
        )
        # A start-of-fit scan (the head-bias warm-start, the RBF bandwidth estimate) opens
        # the parent's shard mmaps. Drop them so the spawn pickle cannot ship that state
        # even if __getstate__ is bypassed.
        # Workers re-open on first gather. The parent does not gather train rows
        # after this point; eval uses a separate NPZ reader.
        if self._hvg_cache is not None:
            n = self._hvg_cache.drop_lazy_mmap_state()
            if n:
                print(f"  Data: dropped lazy mmap LRU before spawn pickle ({n} open shards)", flush=True)
        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=_numpy_batch_to_tensor,
            persistent_workers=use_workers,
            prefetch_factor=prefetch,
            multiprocessing_context=ctx,
            worker_init_fn=self._worker_init_fn if use_workers else None,
        )
