"""Per-condition eval truth NPZ cache — build path from plate h5ads.

Each perturbed condition and each control pool is written as a compressed NPZ
holding a sparse CSR HVG matrix under
``<data_dir>/eval_cache/normalization_<id>/geneset_<hash>__n_<N>/``.
Build is embarrassingly parallel across h5ad files (one worker per plate).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import resource
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from ..vocab import concentrations_from_obs, obs_values_to_categorical, to_categorical_str
from ..data_utils import (
    control_npz_path,
    filter_condition_summary_for_eval,
    perturbed_npz_path,
    save_json,
)

logger = logging.getLogger(__name__)

_GIB = 1024**3
_MIB = 1024**2
_MEM_LOG_INTERVAL_S = 45.0


def _term(msg: str) -> None:
    """Stdout progress. ``logger.info`` is silent under ``src.main`` unless logging is configured."""
    print(f"  [eval_cache] {msg}", flush=True)


def _read_proc_meminfo_kb() -> dict[str, int]:
    """System memory counters from /proc/meminfo (values in kB)."""
    out: dict[str, int] = {}
    with open("/proc/meminfo", encoding="ascii") as fh:
        for line in fh:
            key, rest = line.split(":", 1)
            out[key] = int(rest.strip().split()[0])
    return out


def _process_rss_bytes() -> int | None:
    """Current process RSS from /proc/self/status."""
    try:
        with open("/proc/self/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError as exc:
        logger.warning("MEM | could not read process RSS: %s", exc)
    return None


def _peak_rss_mib() -> float | None:
    """Peak RSS for this process (Linux ru_maxrss is KiB)."""
    try:
        peak_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak_kib / 1024.0
    except OSError as exc:
        logger.warning("MEM | could not read peak RSS: %s", exc)
        return None


def _format_mem_line(
    *,
    workers: int | None = None,
    in_flight: Sequence[str] | None = None,
) -> str:
    """Structured memory snapshot for parent or worker logs."""
    parts = ["MEM"]
    try:
        mem = _read_proc_meminfo_kb()
        avail_kib = mem.get("MemAvailable", mem.get("MemFree", 0))
        total_kib = mem["MemTotal"]
        avail_gib = avail_kib * 1024 / _GIB
        used_pct = 100.0 * (1.0 - avail_kib / total_kib) if total_kib else 0.0
        parts.append(f"avail={avail_gib:.2f}GiB")
        parts.append(f"used={used_pct:.1f}%")
    except OSError as exc:
        parts.append(f"sys=unavailable({exc})")

    rss = _process_rss_bytes()
    if rss is not None:
        parts.append(f"rss={rss / _MIB:.0f}MiB")

    if workers is not None:
        parts.append(f"workers={workers}")
    if in_flight:
        names = list(in_flight)
        if len(names) <= 6:
            parts.append(f"in_flight={names}")
        else:
            parts.append(
                f"in_flight={len(names)}:[{', '.join(names[:3])}, ..., {names[-1]}]",
            )
    return " | ".join(parts)


def _log_pool_failure_context(
    *,
    exc: BaseException,
    done: int,
    n_total: int,
    failed_plate: str | None,
    futures: Mapping[Any, Path],
    n_workers: int,
    in_flight: Sequence[str],
    elapsed_s: float,
) -> None:
    """Emit actionable diagnostics when a worker dies or the pool breaks."""
    pending = sorted(p.name for fut, p in futures.items() if not fut.done())
    logger.error(
        "Eval NPZ build | POOL FAILURE %s after %d/%d h5ads (%.1fs) | "
        "failed_plate=%s pending=%s in_flight=%s workers=%d | %s",
        type(exc).__name__,
        done,
        n_total,
        elapsed_s,
        failed_plate or "?",
        pending,
        sorted(in_flight),
        n_workers,
        _format_mem_line(workers=n_workers, in_flight=in_flight),
        exc_info=exc if not isinstance(exc, BrokenProcessPool) else False,
    )


EVAL_NPZ_MODE = "npz_v1"
EVAL_NPZ_SCOPE = "all_conditions"
EVAL_NPZ_SCOPE_SPLIT = "split_aware"
EVAL_NPZ_VERSION = 2
EVAL_NPZ_STORAGE = "sparse_csr_v1"
EVAL_NPZ_STORAGE_DENSE = "dense_v0"
DEFAULT_EVAL_SPLITS = ("validation", "test")
_UINT16_MAX = int(np.iinfo(np.uint16).max)

# Rows per backed h5ad read in build_one_h5ad's streaming pass.
# Caps peak per-worker memory at ~chunk_rows × n_genes_total sparse during
# the read, instead of |union_rows| × n_genes_total. With ~6 parallel
# workers and 60k-gene plates this keeps each worker around ~10–15 GiB
# peak instead of >100 GiB. Tunable per-call via build_one_h5ad(read_chunk_rows=...).
EVAL_NPZ_READ_CHUNK_ROWS = 500_000

PERTURBED_ARRAY_KEY = "condition_data"
CONTROL_ARRAY_KEY = "control_pool_data"
_oob_sanitize_files = 0


def eval_npz_storage_dtype(normalization_id: str) -> np.dtype:
    """On-disk CSR ``data`` dtype: uint16 for raw counts, float32 for log1p."""
    norm = str(normalization_id).strip().lower()
    if norm == "raw_counts":
        return np.dtype(np.uint16)
    if norm == "log1p":
        return np.dtype(np.float32)
    raise ValueError(
        f"Unknown normalization_id {normalization_id!r}; "
        "expected 'raw_counts' or 'log1p'.",
    )


def _csr_component_keys(array_key: str) -> tuple[str, str, str, str]:
    return (
        f"{array_key}_data",
        f"{array_key}_indices",
        f"{array_key}_indptr",
        f"{array_key}_shape",
    )


def library_full_key(array_key: str) -> str:
    """NPZ key holding the per-cell FULL-transcriptome library for a group."""
    return f"{array_key}_library_full"


def save_eval_sparse_npz(
    path: Path,
    array_key: str,
    mat: sparse.csr_matrix,
    *,
    normalization_id: str,
    library_full: Optional[np.ndarray] = None,
) -> None:
    """Write one eval group as compressed CSR components inside an NPZ.

    ``library_full`` is the per-cell sum over **all** genes in the source h5ad,
    captured before the HVG slice. The cached matrix keeps only the selected genes
    (~9% of a tahoe cell's counts, ~13% on parse), so this divisor cannot be
    recovered from the cache afterwards — and it is the divisor
    ``scanpy.pp.normalize_total`` uses. Without it, a consumer wanting the
    authors'-style ``log1p(T * x / L_full)`` space can only divide by the
    selected-gene library, which is a **different transform**: the ratio
    ``L_selected / L_full`` varies per cell (measured on parse: after normalising
    to a constant 3320 over all genes, the 2,000-gene sums still carry CV 0.64).

    Optional and additive: older caches simply lack the key, and
    :meth:`EvalNpzReader.library_full` returns ``None`` for them.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    storage_dtype = eval_npz_storage_dtype(normalization_id)
    csr = sparse.csr_matrix(mat, copy=True)
    if storage_dtype == np.uint16:
        if csr.data.size and int(csr.data.max()) > _UINT16_MAX:
            raise OverflowError(
                f"Eval NPZ {path}: max raw count {int(csr.data.max())} exceeds "
                f"uint16 max ({_UINT16_MAX}).",
            )
    csr = csr.astype(storage_dtype, copy=False)
    data_k, idx_k, ptr_k, shape_k = _csr_component_keys(array_key)
    arrays: dict[str, np.ndarray] = {
        data_k: csr.data,
        idx_k: csr.indices.astype(np.int32, copy=False),
        ptr_k: csr.indptr.astype(np.int32, copy=False),
        shape_k: np.asarray(csr.shape, dtype=np.int64),
    }
    if library_full is not None:
        lib = np.asarray(library_full, dtype=np.float32).ravel()
        if lib.size != csr.shape[0]:
            raise ValueError(
                f"Eval NPZ {path}: library_full has {lib.size} entries for "
                f"{csr.shape[0]} rows.",
            )
        arrays[library_full_key(array_key)] = lib
    np.savez_compressed(path, **arrays)


def _sanitize_csr_oob_columns(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    shape: tuple[int, int],
    *,
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Drop nnz whose column index is out of range for ``shape[1]``.

    A cache can carry ``indices.max() >= shape[1]`` when the on-disk width
    drifts from the HVG column map used at gather time.
    SciPy ``toarray()`` on such CSRs corrupts the heap (segfault /
    ``free(): invalid next size``). Dropping the stray nnz is lossless for
    the declared column width; rebuild the cache to fix upstream.
    """
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if indices.size == 0 or int(indices.max()) < n_cols:
        return data, indices, indptr, (n_rows, n_cols)

    dropped = 0
    new_data: list[np.ndarray] = []
    new_indices: list[np.ndarray] = []
    new_indptr = [0]
    for r in range(n_rows):
        s, e = int(indptr[r]), int(indptr[r + 1])
        row_idx = indices[s:e]
        row_data = data[s:e]
        keep = row_idx < n_cols
        dropped += int((~keep).sum())
        new_indices.append(row_idx[keep])
        new_data.append(row_data[keep])
        new_indptr.append(new_indptr[-1] + int(keep.sum()))

    global _oob_sanitize_files
    _oob_sanitize_files += 1
    if _oob_sanitize_files == 1:
        logger.info(
            "Eval NPZ CSR load: sanitizing out-of-range column indices "
            "(first: %s dropped %d/%d nnz, n_cols=%d max_index=%d). "
            "Further files sanitized silently.",
            path.name,
            dropped,
            int(indices.size),
            n_cols,
            int(indices.max()),
        )
    else:
        logger.debug(
            "Eval NPZ %s: dropped %d/%d OOB nnz (n_cols=%d)",
            path.name,
            dropped,
            int(indices.size),
            n_cols,
        )
    out_data = np.concatenate(new_data) if new_data else np.empty(0, dtype=data.dtype)
    out_idx = np.concatenate(new_indices) if new_indices else np.empty(0, dtype=indices.dtype)
    out_ptr = np.asarray(new_indptr, dtype=indptr.dtype)
    return out_data, out_idx, out_ptr, (n_rows, n_cols)


def load_eval_npz_matrix(
    path: Path,
    array_key: str,
) -> Union[sparse.csr_matrix, np.ndarray]:
    """Load one eval group preserving sparse layout (CSR) when present.

    Sparse CSR layouts are kept compressed so callers can row-slice cheaply
    BEFORE densification — for ~10k cell × 2k gene conditions, full
    ``toarray()`` costs ~1.5 s while ``csr[idx_1024].toarray()`` costs ~13 ms.
    Legacy dense layouts are returned as float32 ndarrays unchanged.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Eval NPZ missing: {path}")
    with np.load(path) as z:
        if array_key in z.files:
            return np.asarray(z[array_key], dtype=np.float32)
        data_k, idx_k, ptr_k, shape_k = _csr_component_keys(array_key)
        if data_k not in z.files:
            raise KeyError(
                f"Eval NPZ {path} missing {array_key!r} or CSR components; "
                f"files={list(z.files)}",
            )
        data = np.array(z[data_k])
        indices = np.array(z[idx_k])
        indptr = np.array(z[ptr_k])
        shape = tuple(int(v) for v in z[shape_k])
        data, indices, indptr, shape = _sanitize_csr_oob_columns(
            data, indices, indptr, shape, path=path,
        )
        return sparse.csr_matrix((data, indices, indptr), shape=shape)


def load_eval_npz_dense(path: Path, array_key: str) -> np.ndarray:
    """Load one eval group as a dense float32 ndarray (back-compat shim)."""
    mat = load_eval_npz_matrix(path, array_key)
    if sparse.issparse(mat):
        return np.asarray(mat.toarray(), dtype=np.float32)
    return mat


def read_eval_npz_manifest(namespace: Path) -> dict[str, Any]:
    """Read NPZ eval manifest, or synthesize one for legacy caches without manifest.json."""
    ns = Path(namespace)
    path = ns / "manifest.json"
    if path.is_file():
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Invalid eval NPZ manifest: {path}") from exc
    if any(ns.glob("perturbed_cache__*.npz")):
        return {
            "mode": EVAL_NPZ_MODE,
            "legacy_without_manifest": True,
            "n_conditions": len(list(ns.glob("perturbed_cache__*.npz"))),
            "n_control_pools": len(list(ns.glob("control_cache__*.npz"))),
        }
    raise FileNotFoundError(
        f"Eval NPZ manifest missing: {path}. Run scripts/data/build_cache.py eval.",
    )


def _normalize_control_condition(
    control_condition: Optional[Mapping[str, Any]],
) -> dict[str, str]:
    """Stringify control condition values for manifest comparison."""
    return {k: to_categorical_str(v) for k, v in dict(control_condition or {}).items()}


def _prepare_obs_columns(
    obs: pd.DataFrame,
    *,
    condition_variables: Sequence[str],
    control_condition: Optional[Mapping[str, Any]],
    match_keys: Sequence[str],
) -> pd.DataFrame:
    """Return obs with derived ``dose`` and canonical string columns for grouping."""
    out = obs.copy()
    needed = sorted(
        set(condition_variables)
        | set((control_condition or {}).keys())
        | set(match_keys),
    )
    if "dose" in needed and "dose" not in out.columns and "drugname_drugconc" in out.columns:
        out["dose"] = concentrations_from_obs(out, key="drugname_drugconc")

    missing = [c for c in needed if c not in out.columns]
    if missing:
        raise KeyError(
            f"obs missing columns required for eval cache build {missing}; "
            f"available={list(out.columns)}",
        )

    str_frame = pd.DataFrame(index=out.index)
    for col in needed:
        str_frame[col] = obs_values_to_categorical(out[col])
    str_frame["_row"] = np.arange(len(out), dtype=np.int64)
    return str_frame


def _control_row_mask(
    obs_str: pd.DataFrame,
    control_condition: Mapping[str, str],
) -> np.ndarray:
    """Boolean mask of rows matching every control-condition key."""
    if not control_condition:
        return np.zeros(len(obs_str), dtype=bool)
    mask = np.ones(len(obs_str), dtype=bool)
    for key, val in control_condition.items():
        mask &= np.asarray(obs_str[key], dtype=str) == str(val)
    return mask


def _group_row_indices(
    obs_str: pd.DataFrame,
    group_cols: Sequence[str],
    row_mask: Optional[np.ndarray] = None,
) -> dict[tuple[str, ...], np.ndarray]:
    """Group positional row indices (0..n-1) by string key tuple."""
    cols = list(group_cols)
    if not cols:
        raise ValueError("_group_row_indices requires non-empty group_cols.")
    sub = obs_str.loc[row_mask] if row_mask is not None else obs_str
    if sub.empty:
        return {}

    out: dict[tuple[str, ...], np.ndarray] = {}
    for key, group in sub.groupby(cols, sort=False):
        key_t = key if isinstance(key, tuple) else (str(key),)
        out[key_t] = np.sort(group["_row"].to_numpy(dtype=np.int64))
    return out


def _split_row_masks(
    *,
    adata: Any,
    condition_variables: Sequence[str],
    control_condition: Mapping[str, Any],
    match_keys: Sequence[str],
    split_config: Union[str, Path],
    metadata_joins: Optional[Sequence[Mapping[str, Any]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(is_train, is_val, is_test) per obs row, through the training split engine."""
    from ..data_utils import (
        _coerce_split_config,
        enrich_obs_with_metadata,
        prepare_cache_obs_for_indexing,
    )
    from ...data.splits.predicates import split_membership_masks

    joins = list(metadata_joins or ())

    class _JoinConfig:
        metadata_joins = joins

    obs_eval = enrich_obs_with_metadata(
        prepare_cache_obs_for_indexing(
            adata.obs,
            condition_variables=list(condition_variables),
            control_condition=control_condition,
            match_keys=list(match_keys),
        ),
        _JoinConfig(),
    )
    is_train, is_val, is_test, _ = split_membership_masks(
        obs_eval, _coerce_split_config(resolve_split_config_path(split_config)), control_condition,
    )
    return np.asarray(is_train), np.asarray(is_val), np.asarray(is_test)


def _category_order(values: pd.Series) -> list[str]:
    """The column's own category order (e.g. wells H1..H12), else sorted values."""
    if isinstance(values.dtype, pd.CategoricalDtype):
        return [str(c) for c in values.cat.categories]
    return sorted({str(v) for v in values})


def pool_unit_controls(
    namespace: Path,
    *,
    match_keys: Sequence[str],
    unit_column: str,
    control_units: Sequence[tuple[Mapping[str, str], bool]],
    unit_order: Sequence[str],
    normalization_id: str,
    overwrite: bool,
) -> dict[str, int]:
    """Write each match_keys control pool as its train units stacked in unit order.

    A pool with no train unit gets no file: its control cells are all held out,
    so serving any of them as a control would leak test cells into scoring.
    """
    rank = {u: i for i, u in enumerate(unit_order)}
    groups: dict[tuple, list[tuple[int, Mapping[str, str]]]] = {}
    for keys, is_train in control_units:
        pooled = tuple((k, keys[k]) for k in match_keys)
        groups.setdefault(pooled, [])
        if is_train:
            if keys[unit_column] not in rank:
                raise KeyError(f"unit {keys[unit_column]!r} missing from {unit_column} categories")
            groups[pooled].append((rank[keys[unit_column]], keys))
    n_written = n_dropped = 0
    for pooled, units in sorted(groups.items()):
        dest = control_npz_path(namespace, dict(pooled))
        if dest.exists() and not overwrite:
            raise FileExistsError(f"{dest.name} exists; a stale pooled control could leak "
                                  "held-out cells. Rebuild with overwrite.")
        if not units:
            dest.unlink(missing_ok=True)
            n_dropped += 1
            continue
        paths = [control_npz_path(namespace, keys) for _, keys in sorted(units, key=lambda t: t[0])]
        mats = [load_eval_npz_matrix(p, CONTROL_ARRAY_KEY) for p in paths]
        libs = []
        for p in paths:
            with np.load(p) as z:
                libs.append(np.asarray(z[library_full_key(CONTROL_ARRAY_KEY)], dtype=np.float32).ravel())
        save_eval_sparse_npz(dest, CONTROL_ARRAY_KEY, sparse.vstack(mats, format="csr"),
                             normalization_id=normalization_id, library_full=np.concatenate(libs))
        n_written += 1
    return {"n_pooled_written": n_written, "n_pooled_without_train_units": n_dropped}


def probe_h5ad(h5ad_path: Path, hvg_genes_path: Path,
               normalization_id: Optional[str] = None) -> tuple[np.ndarray, list[str], str]:
    """(HVG column indices, gene names, normalization id) for a dataset's first h5ad.

    The normalization id is probed from the data unless given: raw counts and
    log-normalised inputs need different storage dtypes.
    """
    from ..data_utils import load_hvg_indices_from_genes_file, probe_normalization_id

    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        hvg_idx, gene_names = load_hvg_indices_from_genes_file(Path(hvg_genes_path), list(adata.var_names))

        def row(i: int) -> np.ndarray:
            x = adata.X[int(i)]
            return np.asarray(x.toarray() if sparse.issparse(x) else x, dtype=np.float64).ravel()

        norm_id = normalization_id or probe_normalization_id(adata.n_obs, row, seed=0)
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()
    return np.asarray(hvg_idx, dtype=np.int64), list(gene_names), str(norm_id)


def split_config_hash(split_config: Union[str, Path, None]) -> str:
    """Stable short hash of the split YAML's CONTENTS (``none`` when unset).

    Contents, not path, so a moved split keeps its cache key. A missing file raises: hashing
    the path string instead would silently mint a new key for a stale path.
    """
    if not split_config:
        return "none"
    return hashlib.sha256(resolve_split_config_path(split_config).read_bytes()).hexdigest()[:16]


def resolve_split_config_path(
    split_config: Union[str, Path],
    *,
    repo_root: Optional[Path] = None,
) -> Path:
    """Resolve split YAML to an existing file (absolute or relative to repo root)."""
    path = Path(split_config).expanduser()
    if path.is_file():
        return path.resolve()
    # parents[2] of this file is PertGen/src, not the repo: relative splits only resolved from a PertGen cwd.
    from src.paths import PERTGEN_REPO_ROOT

    root = repo_root or PERTGEN_REPO_ROOT
    candidate = path if path.is_absolute() else (root / path)
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(
        f"Split config not found: {candidate}. "
        "Pass --split-config to build_eval_cache.py or fix config.split_config "
        f"(see core_util_data/splits/ under {root}).",
    )


def _control_id_keys_from_perturbed(
    condition_keys: Mapping[str, str],
    *,
    match_keys: Sequence[str],
) -> dict[str, str]:
    """Control pool keys referenced by one perturbed condition."""
    return {k: str(condition_keys[k]) for k in match_keys if k in condition_keys}


def _review_from_plan(plan: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Attach cache-dir and totals to a plan dict."""
    n_existing = int(plan["n_existing"])
    n_to_build = int(plan["n_to_build"])
    return {
        "npz_dir": Path(output_dir),
        "eval_splits": list(plan.get("eval_splits") or DEFAULT_EVAL_SPLITS),
        "n_existing": n_existing,
        "n_to_build": n_to_build,
        "n_total": n_existing + n_to_build,
        "n_perturbed_targets": int(plan["n_perturbed_targets"]),
        "n_control_targets": int(plan["n_control_targets"]),
        "missing_perturbed": list(plan.get("missing_perturbed") or ()),
        "missing_control": list(plan.get("missing_control") or ()),
    }


def _plan_eval_npz_targets(
    output_dir: Path,
    condition_summary_path: Path,
    *,
    condition_variables: Sequence[str],
    control_condition: Mapping[str, Any],
    match_keys: Sequence[str],
    split_config: Optional[Union[str, Path]] = None,
    min_cells: int = 0,
) -> dict[str, Any]:
    """Enumerate expected NPZ targets from condition_summary; report what's missing."""
    ns = Path(output_dir)
    mcv = list(condition_variables)
    mk = list(match_keys)
    ctrl = _normalize_control_condition(control_condition)
    summary = pd.read_parquet(condition_summary_path)
    if split_config is not None:
        filt = filter_condition_summary_for_eval(
            summary,
            split_config=Path(split_config),
            eval_splits=DEFAULT_EVAL_SPLITS,
            control_condition=ctrl,
            drop_control_rows=True,
            min_cells=min_cells if min_cells > 0 else None,
        )
    else:
        from ..data_utils import drop_control_condition_rows

        filt = drop_control_condition_rows(summary, ctrl)
        if min_cells > 0:
            filt = filt[
                pd.to_numeric(filt["num_cells"], errors="coerce").fillna(0)
                >= int(min_cells)
            ].copy()
    pert_paths: list[Path] = []
    ctrl_paths: list[Path] = []
    for row in filt.itertuples(index=False):
        keys = {var: str(getattr(row, var)) for var in mcv}
        pert_paths.append(perturbed_npz_path(ns, keys))
        ctrl_paths.append(
            control_npz_path(
                ns,
                _control_id_keys_from_perturbed(keys, match_keys=mk),
            ),
        )
    ctrl_paths = list(dict.fromkeys(ctrl_paths))
    miss_p = [p for p in pert_paths if not p.is_file()]
    miss_c = [p for p in ctrl_paths if not p.is_file()]
    return {
        "n_perturbed_targets": len(pert_paths),
        "n_control_targets": len(ctrl_paths),
        "n_existing": len(pert_paths) + len(ctrl_paths) - len(miss_p) - len(miss_c),
        "n_to_build": len(miss_p) + len(miss_c),
        "missing_perturbed": [p.name for p in miss_p],
        "missing_control": [p.name for p in miss_c],
        "eval_splits": list(DEFAULT_EVAL_SPLITS),
    }


def build_one_h5ad(
    h5ad_path: Path,
    *,
    output_dir: Path,
    hvg_indices: np.ndarray,
    condition_variables: Sequence[str],
    control_condition: Mapping[str, Any],
    match_keys: Sequence[str],
    normalization_id: str = "raw_counts",
    overwrite: bool = False,
    split_config: Optional[Union[str, Path]] = None,
    metadata_joins: Optional[Sequence[Mapping[str, Any]]] = None,
    read_chunk_rows: int = EVAL_NPZ_READ_CHUNK_ROWS,
    control_unit_column: Optional[str] = None,
    control_split_config: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    """Build perturbed + control NPZ files reachable from one plate h5ad.

    ``read_chunk_rows`` bounds the number of rows materialized per backed-h5ad
    read; smaller values lower peak RSS, larger values cut h5py setup overhead.

    ``control_split_config`` makes controls follow a split: control rows the split
    holds out (val/test) are written as perturbed truth keyed by
    ``condition_variables``, and only its train control rows feed eval controls.
    ``control_unit_column`` (e.g. a well) writes one control pool per
    (match_keys, unit); the match_keys pool is then stacked from the train units
    by :func:`pool_unit_controls`. Both default off.
    """
    h5ad_path = Path(h5ad_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mcv = list(condition_variables)
    mk = list(match_keys)
    pool_keys = mk + [control_unit_column] if control_unit_column else mk
    ctrl_str = _normalize_control_condition(control_condition)

    t0 = time.perf_counter()
    _term(f"{h5ad_path.name} | opening h5ad, scanning obs...")
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        # Stage 1: obs-only work — masks + groupings, zero X access.
        obs_str = _prepare_obs_columns(
            adata.obs,
            condition_variables=mcv,
            control_condition=ctrl_str,
            match_keys=pool_keys,
        )
        unit_order = (_category_order(adata.obs[control_unit_column])
                      if control_unit_column else [])
        ctrl_mask = _control_row_mask(obs_str, ctrl_str)
        masks = dict(adata=adata, condition_variables=mcv, control_condition=control_condition,
                     match_keys=pool_keys, metadata_joins=metadata_joins)
        split_mask = None
        if split_config is not None:
            _, is_val, is_test = _split_row_masks(split_config=split_config, **masks)
            split_mask = is_val | is_test
        # Held-out controls become truth; only train controls may serve as eval controls.
        ctrl_heldout = np.zeros(len(obs_str), dtype=bool)
        ctrl_train = ctrl_mask
        if control_split_config is not None:
            c_train, c_val, c_test = _split_row_masks(split_config=control_split_config, **masks)
            ctrl_heldout, ctrl_train = ctrl_mask & (c_val | c_test), ctrl_mask & c_train
        pert_row_mask = (~ctrl_mask) | ctrl_heldout
        if split_mask is not None:
            pert_row_mask &= split_mask
        pert_groups = _group_row_indices(obs_str, mcv, row_mask=pert_row_mask)
        # Unit pools keep every control row (references need held-out units too);
        # without units the pool is the train controls directly.
        ctrl_groups = _group_row_indices(
            obs_str, pool_keys, row_mask=ctrl_mask if control_unit_column else ctrl_train,
        )
        control_units = []
        if control_unit_column:
            for key_t, rows in ctrl_groups.items():
                train = ctrl_train[rows]
                if train.any() and not train.all():
                    raise ValueError(f"control unit {dict(zip(pool_keys, key_t))} straddles the "
                                     "train/held-out boundary of the control split")
                control_units.append(({c: str(v) for c, v in zip(pool_keys, key_t)}, bool(train.all())))

        # Which control pools are actually referenced by a perturbed condition.
        required_ctrl_keys = {
            tuple(sorted(
                _control_id_keys_from_perturbed(
                    {c: str(v) for c, v in zip(mcv, key_t)}, match_keys=mk,
                ).items()
            ))
            for key_t in pert_groups
        }

        # Stage 2: decide jobs without touching X (skip already-built NPZs).
        pert_jobs: list[tuple[Path, np.ndarray]] = []
        n_pert_skipped = 0
        for key_t, rows in pert_groups.items():
            out_path = perturbed_npz_path(
                output_dir, {c: str(v) for c, v in zip(mcv, key_t)},
            )
            if out_path.is_file() and not overwrite:
                n_pert_skipped += 1
                continue
            pert_jobs.append((out_path, rows))

        ctrl_jobs: list[tuple[Path, np.ndarray]] = []
        n_ctrl_skipped = 0
        for key_t, rows in ctrl_groups.items():
            keys = {c: str(v) for c, v in zip(pool_keys, key_t)}
            pooled = tuple(sorted((k, v) for k, v in keys.items() if k in mk))
            if split_config is not None and pooled not in required_ctrl_keys:
                continue
            out_path = control_npz_path(output_dir, keys)
            if out_path.is_file() and not overwrite:
                n_ctrl_skipped += 1
                continue
            ctrl_jobs.append((out_path, rows))

        # Stage 3: STREAMING forward pass through X.
        # Read the union of relevant rows in row-chunks (sequential h5py reads,
        # since union_rows is sorted), HVG-slice each chunk immediately to drop
        # the ~30× wider full-gene matrix before it can pile up, then route the
        # HVG rows into per-job partials. Each job is flushed to disk as soon as
        # its last row has been seen, so RAM never holds the full HVG block —
        # only the current chunk plus tiny partials for jobs straddling
        # chunk boundaries. Peak per worker ≈ chunk_rows × n_genes_total sparse.
        n_pert_written = 0
        n_ctrl_written = 0
        if pert_jobs or ctrl_jobs:
            union_rows = np.unique(np.concatenate(
                [rows for _, rows in pert_jobs] + [rows for _, rows in ctrl_jobs]
            ))
            n_union = int(union_rows.size)
            hvg_idx = np.asarray(hvg_indices, dtype=np.int64)
            chunk_rows = max(1, int(read_chunk_rows))
            n_chunks = (n_union + chunk_rows - 1) // chunk_rows
            stream_msg = (
                f"{h5ad_path.name} | streaming {n_union:,} rows in {n_chunks} chunk(s) "
                f"of {chunk_rows:,} | {len(pert_jobs)} pert + {len(ctrl_jobs)} ctrl "
                f"(hvg={int(hvg_idx.size)}/{int(adata.shape[1])})"
            )
            _term(stream_msg)
            logger.info(
                "Eval NPZ | %s | streaming read: %d rows in %d chunk(s) of %d "
                "for %d pert + %d ctrl jobs (hvg=%d/%d)",
                h5ad_path.name, n_union, n_chunks, chunk_rows,
                len(pert_jobs), len(ctrl_jobs),
                int(hvg_idx.size), int(adata.shape[1]),
            )

            # Per-job state: (out_path, sorted local indices into union_rows, array_key).
            # gi = searchsorted(union_rows, rows) maps each job's plate rows
            # to positions in the union; gi is sorted since rows is sorted.
            jobs: list[tuple[Path, np.ndarray, str]] = []
            for out_path, rows in pert_jobs:
                jobs.append((out_path, np.searchsorted(union_rows, rows), PERTURBED_ARRAY_KEY))
            for out_path, rows in ctrl_jobs:
                jobs.append((out_path, np.searchsorted(union_rows, rows), CONTROL_ARRAY_KEY))
            job_total = np.array([gi.size for _, gi, _ in jobs], dtype=np.int64)
            partials: list[list[sparse.csr_matrix]] = [[] for _ in jobs]
            # Per-cell FULL-gene library, captured before the HVG slice discards
            # ~90% of each cell's counts. This is normalize_total's divisor and is
            # unrecoverable from the cached matrix. See save_eval_sparse_npz.
            lib_partials: list[list[np.ndarray]] = [[] for _ in jobs]
            seen = np.zeros(len(jobs), dtype=np.int64)

            for ci, start in enumerate(range(0, n_union, chunk_rows), start=1):
                stop = min(start + chunk_rows, n_union)
                _term(
                    f"{h5ad_path.name} | chunk {ci}/{n_chunks} "
                    f"reading rows {start:,}–{stop:,} of {n_union:,} "
                    f"({time.perf_counter() - t0:.0f}s so far)..."
                )
                raw = adata[union_rows[start:stop]].X
                csr = (
                    sparse.csr_matrix(raw) if sparse.issparse(raw)
                    else sparse.csr_matrix(np.asarray(raw))
                )
                del raw
                # Full-gene per-cell library BEFORE the slice — cheap (one pass
                # over the chunk's CSR data) and impossible to recover after.
                chunk_lib = np.asarray(csr.sum(axis=1), dtype=np.float64).ravel()
                chunk_hvg = csr[:, hvg_idx]
                del csr  # release the wide (n_genes_total) intermediate ASAP

                # Route chunk rows into each still-open job; flush when complete.
                for ji, (out_path, gi, array_key) in enumerate(jobs):
                    remaining = int(job_total[ji] - seen[ji])
                    if remaining == 0:
                        continue
                    in_chunk = (gi >= start) & (gi < stop)
                    n_here = int(in_chunk.sum())
                    if n_here == 0:
                        continue
                    local = gi[in_chunk] - start
                    partials[ji].append(chunk_hvg[local, :])
                    lib_partials[ji].append(chunk_lib[local])
                    seen[ji] += n_here
                    if seen[ji] == job_total[ji]:
                        mat = (
                            partials[ji][0] if len(partials[ji]) == 1
                            else sparse.vstack(partials[ji], format="csr")
                        )
                        lib = (
                            lib_partials[ji][0] if len(lib_partials[ji]) == 1
                            else np.concatenate(lib_partials[ji])
                        )
                        save_eval_sparse_npz(
                            out_path, array_key, mat,
                            normalization_id=normalization_id,
                            library_full=lib,
                        )
                        partials[ji] = []  # drop partials reference for GC
                        lib_partials[ji] = []
                        if array_key == PERTURBED_ARRAY_KEY:
                            n_pert_written += 1
                        else:
                            n_ctrl_written += 1
                del chunk_hvg, chunk_lib

            # Defensive: every job must have received exactly all its rows.
            # If anything diverges we want a loud failure, not a silent
            # half-written cache (per project rule: never fail silently).
            unflushed = [
                (jobs[ji][0].name, int(seen[ji]), int(job_total[ji]))
                for ji in range(len(jobs))
                if seen[ji] != job_total[ji]
            ]
            if unflushed:
                raise RuntimeError(
                    f"Eval NPZ {h5ad_path.name}: {len(unflushed)} job(s) did not "
                    f"receive all expected rows during streaming write. First few: "
                    f"{unflushed[:3]}. Likely an index/union mismatch.",
                )
        else:
            _term(
                f"{h5ad_path.name} | nothing to write "
                f"(skipped pert={n_pert_skipped} ctrl={n_ctrl_skipped})"
            )
    finally:
        if hasattr(adata, "file") and adata.file is not None:
            adata.file.close()

    elapsed = float(time.perf_counter() - t0)
    stats = {
        "h5ad_path": str(h5ad_path),
        "n_obs": int(len(obs_str)),
        "n_perturbed_written": n_pert_written,
        "n_perturbed_skipped": n_pert_skipped,
        "n_control_written": n_ctrl_written,
        "n_control_skipped": n_ctrl_skipped,
        "elapsed_seconds": elapsed,
        "control_units": control_units,
        "unit_order": unit_order,
    }
    logger.info(
        "Eval NPZ | %s | pert=%d+%d skip ctrl=%d+%d skip | %.1fs",
        h5ad_path.name,
        n_pert_written, n_pert_skipped,
        n_ctrl_written, n_ctrl_skipped,
        elapsed,
    )
    _term(
        f"{h5ad_path.name} | done pert={n_pert_written}+{n_pert_skipped}skip "
        f"ctrl={n_ctrl_written}+{n_ctrl_skipped}skip | {elapsed:.1f}s"
    )
    return stats


def _build_one_h5ad_worker(
    h5ad_path: str,
    build_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """ProcessPool entry: log plate start/end, peak RSS, and re-raise worker failures."""
    path = Path(h5ad_path)
    plate = path.name
    pid = os.getpid()
    t0 = time.perf_counter()
    _term(f"worker START plate={plate} pid={pid}")
    logger.info(
        "Eval NPZ worker | START plate=%s pid=%d | %s",
        plate,
        pid,
        _format_mem_line(),
    )
    try:
        stats = build_one_h5ad(path, **build_kwargs)
    except Exception:
        peak = _peak_rss_mib()
        logger.exception(
            "Eval NPZ worker | FAIL plate=%s pid=%d elapsed=%.1fs peak_rss=%sMiB | %s",
            plate,
            pid,
            time.perf_counter() - t0,
            f"{peak:.0f}" if peak is not None else "?",
            _format_mem_line(),
        )
        raise
    peak = _peak_rss_mib()
    stats = dict(stats)
    if peak is not None:
        stats["peak_rss_mib"] = peak
    logger.info(
        "Eval NPZ worker | DONE plate=%s pid=%d elapsed=%.1fs peak_rss=%sMiB | %s",
        plate,
        pid,
        time.perf_counter() - t0,
        f"{peak:.0f}" if peak is not None else "?",
        _format_mem_line(),
    )
    return stats


def _build_eval_npz_h5ads(
    *,
    data_dir: Path,
    h5ad_paths: Sequence[Path],
    output_dir: Path,
    hvg_indices: np.ndarray,
    hvg_genes_path: Path,
    condition_variables: Sequence[str],
    control_condition: Mapping[str, Any],
    match_keys: Sequence[str],
    gene_set_hash_value: str,
    n_genes: int,
    normalization_id: str,
    overwrite: bool = False,
    max_workers: Optional[int] = None,
    split_config: Optional[Union[str, Path]] = None,
    metadata_joins: Optional[Sequence[Mapping[str, Any]]] = None,
    control_unit_column: Optional[str] = None,
    control_split_config: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    """Parallel h5ad → per-condition NPZ build (no planning gate)."""
    ns = Path(output_dir)
    mcv = list(condition_variables)
    mk = list(match_keys)
    ctrl = dict(control_condition or {})
    split_path = str(Path(split_config).resolve()) if split_config else None
    split_hash = split_config_hash(split_config)
    use_split_scope = split_config is not None

    ns.mkdir(parents=True, exist_ok=True)
    build_kwargs = {
        "output_dir": ns,
        "hvg_indices": np.asarray(hvg_indices, dtype=np.int64),
        "condition_variables": mcv,
        "control_condition": ctrl,
        "match_keys": mk,
        "normalization_id": str(normalization_id),
        "overwrite": bool(overwrite),
        "split_config": split_path if use_split_scope else None,
        "metadata_joins": metadata_joins,
        "control_unit_column": control_unit_column,
        "control_split_config": str(Path(control_split_config).resolve()) if control_split_config else None,
    }
    h5ad_list = [Path(p) for p in h5ad_paths]
    n_workers = max(1, min(len(h5ad_list) or 1, max_workers or os.cpu_count() or 4))
    _term(
        f"BUILDING | {len(h5ad_list)} plates, {n_workers} workers, out={ns}"
    )
    logger.info(
        "Eval NPZ build | data=%s h5ads=%d workers=%d out=%s splits=%s force=%s | %s",
        data_dir, len(h5ad_list), n_workers, ns, list(DEFAULT_EVAL_SPLITS), overwrite,
        _format_mem_line(workers=n_workers),
    )

    t0 = time.perf_counter()
    last_mem_log = t0
    if not h5ad_list:
        part_stats: list[dict[str, Any]] = []
    elif n_workers == 1:
        part_stats = []
        for i, p in enumerate(h5ad_list, start=1):
            part_stats.append(_build_one_h5ad_worker(str(p), build_kwargs))
            now = time.perf_counter()
            remain = now - t0
            eta = remain / i * (len(h5ad_list) - i)
            _term(
                f"plates {i}/{len(h5ad_list)} done, last={p.name} "
                f"({remain:.0f}s elapsed, ~{eta:.0f}s remaining)"
            )
            if i == len(h5ad_list) or (now - last_mem_log) >= _MEM_LOG_INTERVAL_S:
                logger.info(
                    "Eval NPZ build | h5ads_done=%d/%d (%.1fs) | %s",
                    i, len(h5ad_list), now - t0,
                    _format_mem_line(workers=1),
                )
                last_mem_log = now
    else:
        in_flight = {p.name for p in h5ad_list}
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_build_one_h5ad_worker, str(p), build_kwargs): p
                for p in h5ad_list
            }
            done = 0
            part_stats = []
            try:
                for fut in as_completed(futures):
                    plate_path = futures[fut]
                    plate = plate_path.name
                    try:
                        part_stats.append(fut.result())
                    except BrokenProcessPool as exc:
                        _log_pool_failure_context(
                            exc=exc,
                            done=done,
                            n_total=len(futures),
                            failed_plate=plate,
                            futures=futures,
                            n_workers=n_workers,
                            in_flight=sorted(in_flight),
                            elapsed_s=time.perf_counter() - t0,
                        )
                        raise RuntimeError(
                            f"Eval NPZ build worker pool broke while processing {plate!r}; "
                            f"{len(in_flight)} plates still in flight. "
                            "Likely OOM — reduce --n-workers or free system memory.",
                        ) from exc
                    except Exception as exc:
                        _log_pool_failure_context(
                            exc=exc,
                            done=done,
                            n_total=len(futures),
                            failed_plate=plate,
                            futures=futures,
                            n_workers=n_workers,
                            in_flight=sorted(in_flight),
                            elapsed_s=time.perf_counter() - t0,
                        )
                        raise
                    in_flight.discard(plate)
                    done += 1
                    now = time.perf_counter()
                    eta = (now - t0) / done * (len(futures) - done)
                    _term(
                        f"plates {done}/{len(futures)} done, last={plate} "
                        f"({now - t0:.0f}s elapsed, ~{eta:.0f}s remaining) "
                        f"| in_flight={len(in_flight)}"
                    )
                    if (
                        done == len(futures)
                        or done % max(1, len(futures) // 4) == 0
                        or (now - last_mem_log) >= _MEM_LOG_INTERVAL_S
                    ):
                        logger.info(
                            "Eval NPZ build | h5ads_done=%d/%d in_flight=%d (%.1fs) | %s",
                            done, len(futures), len(in_flight), now - t0,
                            _format_mem_line(workers=n_workers, in_flight=sorted(in_flight)),
                        )
                        last_mem_log = now
            except BrokenProcessPool as exc:
                _log_pool_failure_context(
                    exc=exc,
                    done=done,
                    n_total=len(futures),
                    failed_plate=None,
                    futures=futures,
                    n_workers=n_workers,
                    in_flight=sorted(in_flight),
                    elapsed_s=time.perf_counter() - t0,
                )
                raise RuntimeError(
                    f"Eval NPZ build worker pool broke with {len(in_flight)} plates "
                    "still in flight. Likely OOM — reduce --n-workers or free system memory.",
                ) from exc

    pooling = None
    if control_unit_column:
        orders = {tuple(s["unit_order"]) for s in part_stats}
        if len(orders) > 1:
            raise ValueError(f"h5ads disagree on {control_unit_column} category order: {orders}")
        pooling = pool_unit_controls(
            ns, match_keys=mk, unit_column=control_unit_column,
            control_units=[u for s in part_stats for u in s["control_units"]],
            unit_order=list(orders.pop()) if orders else [], normalization_id=str(normalization_id),
            overwrite=True,  # pooled pools are derived from this build's unit files
        )
        _term(f"pooled unit controls | {pooling}")
    n_pert = len(list(ns.glob("perturbed_cache__*.npz")))
    n_ctrl = len(list(ns.glob("control_cache__*.npz")))
    index_scope = EVAL_NPZ_SCOPE_SPLIT if use_split_scope else EVAL_NPZ_SCOPE
    storage_dtype = eval_npz_storage_dtype(normalization_id)
    manifest = {
        "mode": EVAL_NPZ_MODE,
        "version": EVAL_NPZ_VERSION,
        "storage": EVAL_NPZ_STORAGE,
        "storage_dtype": str(storage_dtype),
        "index_scope": index_scope,
        "normalization_id": normalization_id,
        "gene_set_hash": gene_set_hash_value,
        "n_genes": int(n_genes),
        "condition_variables": mcv,
        "control_condition": _normalize_control_condition(ctrl),
        "match_keys": mk,
        "n_conditions": n_pert,
        "n_control_pools": n_ctrl,
        "data_dir": str(Path(data_dir).resolve()),
        "hvg_genes_path": str(Path(hvg_genes_path).resolve()),
        "n_h5ad_files": len(h5ad_list),
    }
    if use_split_scope:
        manifest["split_config"] = split_path
        manifest["split_config_hash"] = split_hash
        manifest["eval_splits_built"] = list(DEFAULT_EVAL_SPLITS)
    if control_split_config:
        manifest["control_split_config"] = build_kwargs["control_split_config"]
        manifest["control_split_config_hash"] = split_config_hash(control_split_config)
    if control_unit_column:
        manifest["control_unit_column"] = control_unit_column
        manifest["control_pooling"] = pooling
    save_json(ns / "manifest.json", manifest)
    elapsed = float(time.perf_counter() - t0)

    n_built = sum(
        s.get("n_perturbed_written", 0) + s.get("n_control_written", 0)
        for s in part_stats
    )
    n_skipped = sum(
        s.get("n_perturbed_skipped", 0) + s.get("n_control_skipped", 0)
        for s in part_stats
    )
    logger.info(
        "Eval NPZ build complete in %.1fs | built=%d skipped=%d conditions=%d control_pools=%d | %s | %s",
        elapsed, n_built, n_skipped, n_pert, n_ctrl, ns,
        _format_mem_line(workers=n_workers),
    )
    _term(
        f"complete in {elapsed:.1f}s | built={n_built} skipped={n_skipped} "
        f"conditions={n_pert} control_pools={n_ctrl}"
    )
    return {
        "output_dir": str(ns),
        "mode": EVAL_NPZ_MODE,
        "n_built": n_built,
        "n_skipped": n_skipped,
        "n_conditions": n_pert,
        "n_control_pools": n_ctrl,
        "eval_splits_built": list(DEFAULT_EVAL_SPLITS) if use_split_scope else None,
        "split_config_hash": split_hash if use_split_scope else None,
        "elapsed_seconds": elapsed,
        "n_h5ad_files": len(h5ad_list),
        "n_workers": n_workers,
        "part_stats": part_stats,
    }


def run_eval_cache_build(
    *,
    data_dir: Path,
    h5ad_paths: Sequence[Path],
    output_dir: Path,
    hvg_indices: np.ndarray,
    hvg_genes_path: Path,
    condition_variables: Sequence[str],
    control_condition: Mapping[str, Any],
    match_keys: Sequence[str],
    gene_set_hash_value: str,
    n_genes: int,
    normalization_id: str,
    overwrite: bool = False,
    max_workers: Optional[int] = None,
    split_config: Optional[Union[str, Path]] = None,
    metadata_joins: Optional[Sequence[Mapping[str, Any]]] = None,
    condition_summary_path: Optional[Path] = None,
    min_cells: int = 0,
    dry_run: bool = False,
    allow_build: bool = True,
    control_unit_column: Optional[str] = None,
    control_split_config: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    """Plan from condition_summary, optionally build missing NPZs, replan, return stats.

    When ``condition_summary_path`` is set, enumerates expected perturbed + control
    NPZ files for validation + test, builds only when files are missing (unless
    ``overwrite``), and verifies completeness after build.

    Without a condition summary, skips planning and runs the h5ad build directly
    (per-plate skip of existing NPZs still applies).
    """
    ns = Path(output_dir)
    summary_path = Path(condition_summary_path) if condition_summary_path else None
    plan: dict[str, Any] | None = None
    if summary_path is not None and (control_unit_column or control_split_config):
        raise ValueError("condition-summary planning does not know about control units or a "
                         "control split; build without --condition-summary")

    if summary_path is not None and summary_path.is_file():
        plan = _plan_eval_npz_targets(
            ns,
            summary_path,
            condition_variables=condition_variables,
            control_condition=control_condition,
            match_keys=match_keys,
            split_config=split_config,
            min_cells=min_cells,
        )
        review = _review_from_plan(plan, ns)
        review["did_build"] = False
        _term(
            f"review | {review['n_existing']}/{review['n_total']} present "
            f"({review['n_to_build']} missing) splits={review['eval_splits']} dir={ns}"
        )
        logger.info(
            "Eval NPZ cache review | %s/%s present (%s missing) splits=%s dir=%s",
            review["n_existing"],
            review["n_total"],
            review["n_to_build"],
            review["eval_splits"],
            ns,
        )
        if dry_run:
            return review
        if not overwrite and review["n_to_build"] == 0:
            _term("skip | cache complete, not rebuilding")
            return review
    elif dry_run:
        raise FileNotFoundError(
            f"condition_summary not found at {summary_path}; "
            "required for dry_run. Run dm.setup() or pass --condition-summary.",
        )
    elif summary_path is not None:
        _term(f"review skipped (no condition_summary at {summary_path}); will skip existing NPZs per plate")
        logger.info(
            "Eval NPZ cache review skipped (no condition_summary at %s); "
            "build_one_h5ad will skip existing NPZ files per plate.",
            summary_path,
        )

    needs_build = plan is None or overwrite or int(plan["n_to_build"]) > 0
    if needs_build and not allow_build:
        if plan is not None:
            review = _review_from_plan(plan, ns)
            raise FileNotFoundError(
                f"Eval NPZ cache incomplete under {ns}: "
                f"{review['n_existing']}/{review['n_total']} files present, "
                f"{review['n_to_build']} missing for splits {review['eval_splits']}. "
                "Set build_eval_cache=true or run scripts/data/build_cache.py eval first.",
            )
        raise FileNotFoundError(
            f"Eval NPZ build not allowed (allow_build=false) under {ns}.",
        )

    if not needs_build:
        assert plan is not None
        review = _review_from_plan(plan, ns)
        review["did_build"] = False
        return review

    logger.info(
        "Building eval NPZ cache from h5ads | missing=%s overwrite=%s",
        plan["n_to_build"] if plan is not None else "?",
        overwrite,
    )
    _term(
        f"BUILDING from h5ads | missing={plan['n_to_build'] if plan is not None else '?'} "
        f"overwrite={overwrite}"
    )
    build_stats = _build_eval_npz_h5ads(
        data_dir=data_dir,
        h5ad_paths=h5ad_paths,
        output_dir=ns,
        hvg_indices=hvg_indices,
        hvg_genes_path=hvg_genes_path,
        condition_variables=condition_variables,
        control_condition=control_condition,
        match_keys=match_keys,
        gene_set_hash_value=gene_set_hash_value,
        n_genes=n_genes,
        normalization_id=normalization_id,
        overwrite=bool(overwrite),
        max_workers=max_workers,
        split_config=split_config,
        metadata_joins=metadata_joins,
        control_unit_column=control_unit_column,
        control_split_config=control_split_config,
    )

    if plan is None:
        n_pert = int(build_stats.get("n_conditions", 0))
        n_ctrl = int(build_stats.get("n_control_pools", 0))
        n_existing = n_pert + n_ctrl
        return {
            "npz_dir": ns,
            "eval_splits": list(DEFAULT_EVAL_SPLITS),
            "n_existing": n_existing,
            "n_to_build": 0,
            "n_total": n_existing,
            "n_built": int(build_stats.get("n_built", 0)),
            "n_conditions": n_pert,
            "n_control_pools": n_ctrl,
            "did_build": True,
            **build_stats,
        }

    plan = _plan_eval_npz_targets(
        ns,
        summary_path,
        condition_variables=condition_variables,
        control_condition=control_condition,
        match_keys=match_keys,
        split_config=split_config,
        min_cells=min_cells,
    )
    review = _review_from_plan(plan, ns)
    review["n_built"] = int(build_stats.get("n_built", 0))
    review["did_build"] = True
    if review["n_to_build"] > 0:
        raise RuntimeError(
            f"Eval NPZ cache build finished but {review['n_to_build']} files still "
            f"missing under {ns} (splits={review['eval_splits']}).",
        )
    logger.info(
        "Eval NPZ cache built | %s/%s present splits=%s",
        review["n_existing"],
        review["n_total"],
        review["eval_splits"],
    )
    _term(
        f"built | {review['n_existing']}/{review['n_total']} present "
        f"splits={review['eval_splits']}"
    )
    return review
