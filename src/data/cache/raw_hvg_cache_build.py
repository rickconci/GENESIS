"""Parallel HVG-subset raw-counts CSR mmap cache builder.

Programmatic entry point: :func:`run_raw_hvg_cache_build`.

Builds the on-disk layout read by :mod:`src.data.cache.raw_hvg_cache` (training mmap CSR).
Eval truth is built separately via :mod:`src.data.cache.eval_cache_build` (NPZ under ``eval_cache/``).

Build pipeline (single-pass, row-range-parallel):

1. Probe each h5ad for ``n_obs`` via :func:`probe_n_obs`.
2. Resolve the HVG column set once into a bool keep-mask + HVG-space remap.
3. Split each shard into ``ranges_per_shard`` row ranges; each range is one
   ``ProcessPoolExecutor`` job streaming via h5py.
4. Master finalises each shard: indptr, memmap blit, obs sidecar, meta.json.
5. Top-level ``manifest.json`` after every shard finalises.

Usage::

    python scripts/data/build_cache.py train --config <cfg> --n-workers 32 --ranges-per-shard 4

"""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing as mp
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import anndata as ad
import h5py
import numpy as np

from src.data.data_utils import load_hvg_indices_from_genes_file
from src.data.cache.raw_hvg_cache import (
    CACHE_VERSION,
    OBS_SIDECAR_DIR,
    OBS_SIDECAR_NPZ,
    TRANSFORM_TAG,
)


# ---------------------------------------------------------------------------
# h5py row-range workers (single-pass build path)
# ---------------------------------------------------------------------------

from dataclasses import dataclass


# Inner row-chunk inside one row-range worker. Sized so peak slab memory is
# ~600 MB per worker (~32k rows × ~1500 nnz × 12 B uncompressed).
DEFAULT_INNER_CHUNK_ROWS = 32_768

# Streaming chunk for tmp -> memmap blit (uint16). 256 MB stays comfortably in
# CPU caches and bounds peak RAM during assembly regardless of total nnz.
_BLIT_CHUNK_BYTES = 1 << 28

# Sentinel for the cap on raw-count values; the on-disk dtype is uint16.
_UINT16_MAX = int(np.iinfo(np.uint16).max)


# --- Result dataclasses (picklable across the worker pool) ----------------


@dataclass(frozen=True)
class RangeResult:
    """One range worker's output: per-row HVG nnz + paths to uint16 temp binaries.

    The ``per_row_nnz`` array is concatenated across ranges by
    :func:`assemble_shard` to build the final ``indptr``. ``tmp_data_path``
    and ``tmp_idx_path`` hold the filtered ``uint16`` data / column indices in
    row-major order; assembly blits them sequentially into the memmap.
    """

    shard_id: int
    range_idx: int
    row_start: int
    row_end: int
    per_row_nnz: np.ndarray  # int64, shape (row_end - row_start,)
    tmp_data_path: Path
    tmp_idx_path: Path
    nnz: int
    max_count_observed: int


@dataclass(frozen=True)
class AssembleResult:
    """Summary returned by :func:`assemble_shard` after final memmap is on disk."""

    n_obs: int
    nnz: int
    max_count_observed: int
    max_col: int  # max(indices); -1 if nnz==0


# --- Pure helpers ---------------------------------------------------------


def compute_hvg_col_mask(
    n_var: int,
    hvg_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a ``bool[n_var]`` keep-mask and ``int32[n_var]`` HVG-space remap.

    Source CSR column indices live in ``[0, n_var)``. After filtering with
    ``keep_mask[col]``, surviving column ids are remapped to HVG-space
    ``[0, n_hvg)`` via ``remap[col]`` (entries where ``keep_mask`` is False
    are left at ``-1`` to make accidental reads explode loudly).

    Args:
        n_var: Total number of variables (genes) in the source dataset.
        hvg_indices: int64 column indices into the source dataset to keep.

    Returns:
        ``(keep_mask, remap)`` — the mask is ``bool[n_var]``, the remap is
        ``int32[n_var]`` with HVG-space indices on kept columns and ``-1``
        elsewhere.
    """
    if n_var <= 0:
        raise ValueError(f"n_var must be positive, got {n_var}")
    hvg = np.asarray(hvg_indices, dtype=np.int64)
    if hvg.size == 0:
        raise ValueError("hvg_indices is empty")
    if hvg.min() < 0 or hvg.max() >= n_var:
        raise ValueError(
            f"HVG indices out of bounds: min={hvg.min()} max={hvg.max()} "
            f"n_var={n_var}"
        )
    if np.unique(hvg).size != hvg.size:
        raise ValueError("HVG indices contain duplicates.")
    if hvg.size > _UINT16_MAX:
        raise ValueError(
            f"HVG count {hvg.size} > uint16 max ({_UINT16_MAX}); "
            "indices dtype must be widened.",
        )

    keep_mask = np.zeros(n_var, dtype=bool)
    keep_mask[hvg] = True

    remap = np.full(n_var, -1, dtype=np.int32)
    remap[hvg] = np.arange(hvg.size, dtype=np.int32)

    return keep_mask, remap


def plan_row_ranges(n_obs: int, target_ranges: int) -> List[Tuple[int, int]]:
    """Split ``[0, n_obs)`` into ~equal half-open ``[start, stop)`` ranges.

    ``target_ranges <= 1`` collapses to a single range. When the request is
    bigger than ``n_obs``, we cap at ``n_obs`` (one row per range), since
    issuing empty ranges would later trip the contiguity check in
    :func:`assemble_shard`.
    """
    if n_obs < 0:
        raise ValueError(f"n_obs must be non-negative, got {n_obs}")
    if n_obs == 0:
        return []
    target = max(1, min(int(target_ranges), n_obs))
    if target == 1:
        return [(0, n_obs)]

    base, rem = divmod(n_obs, target)
    out: List[Tuple[int, int]] = []
    cursor = 0
    for i in range(target):
        size = base + (1 if i < rem else 0)
        out.append((cursor, cursor + size))
        cursor += size
    if cursor != n_obs:  # pragma: no cover — defensive
        raise RuntimeError(f"plan_row_ranges miscounted: {cursor} != {n_obs}")
    return out


def probe_n_obs(h5ad_path: Path) -> int:
    """Cheap n_obs probe via h5py — ``len(X/indptr) - 1`` for CSR-encoded X."""
    with h5py.File(str(h5ad_path), "r") as f:
        if "X" not in f:
            raise KeyError(f"{h5ad_path}: missing /X group")
        x = f["X"]
        if "indptr" not in x:
            raise KeyError(
                f"{h5ad_path}: /X has no 'indptr' dataset; only sparse CSR "
                "h5ads are supported by the fast builder.",
            )
        return int(x["indptr"].shape[0]) - 1


def _open_x_csr(handle: h5py.File) -> Tuple[h5py.Dataset, h5py.Dataset, h5py.Dataset]:
    """Return ``(data, indices, indptr)`` h5py datasets from sparse CSR ``/X``.

    AnnData stores sparse X as a group with ``encoding-type`` attribute set
    to ``csr_matrix`` or ``csc_matrix``. We only support CSR; CSC would
    require a different filtering strategy (column-wise nnz buckets).
    """
    if "X" not in handle:
        raise KeyError("h5ad missing /X group")
    x = handle["X"]
    enc = x.attrs.get("encoding-type") or x.attrs.get("h5sparse_format")
    if enc is not None:
        enc_str = enc.decode() if isinstance(enc, bytes) else str(enc)
        if "csr" not in enc_str.lower():
            raise NotImplementedError(
                f"h5ad /X encoding-type={enc_str!r}; fast builder only "
                "supports csr_matrix.",
            )
    elif not all(k in x for k in ("data", "indices", "indptr")):
        # No attr and not the CSR triple shape — bail loudly.
        raise RuntimeError(
            "h5ad /X has no encoding-type and is not the CSR triple. "
            "Dense X is not supported by the fast builder.",
        )
    return x["data"], x["indices"], x["indptr"]


# --- Worker: HVG-filter one row range in a single pass ---------------------


def process_row_range(
    *,
    shard_id: int,
    range_idx: int,
    h5ad_path: str,
    row_start: int,
    row_end: int,
    hvg_col_mask: np.ndarray,
    hvg_index_map: np.ndarray,
    tmp_dir: str,
    inner_chunk_rows: int = DEFAULT_INNER_CHUNK_ROWS,
) -> RangeResult:
    """Stream ``[row_start, row_end)`` once, HVG-filter, write tmp uint16 binaries.

    Each gzip chunk of ``X/data`` and ``X/indices`` covering this range is
    decompressed exactly once. Filtered output is appended row-major to the
    per-range temp files, and the per-row HVG nnz is accumulated for the
    master to fold into ``indptr`` later.

    Designed to be called as a ``ProcessPoolExecutor`` job. Picklable args
    only; the h5py handle is opened *inside* the worker after fork.
    """
    if row_end <= row_start:
        raise ValueError(f"empty range [{row_start}, {row_end})")
    n_rows = row_end - row_start
    inner = max(1, int(inner_chunk_rows))

    tmp_dir_p = Path(tmp_dir)
    tmp_dir_p.mkdir(parents=True, exist_ok=True)
    tmp_data_path = tmp_dir_p / f"shard{shard_id:03d}_r{range_idx:03d}_data.u16"
    tmp_idx_path = tmp_dir_p / f"shard{shard_id:03d}_r{range_idx:03d}_idx.u16"

    per_row_nnz = np.zeros(n_rows, dtype=np.int64)
    range_nnz = 0
    max_count = 0

    try:
        with h5py.File(h5ad_path, "r") as f:
            xdata, xindices, xindptr = _open_x_csr(f)
            local_indptr = np.asarray(
                xindptr[row_start : row_end + 1], dtype=np.int64,
            )
            if local_indptr.shape[0] != n_rows + 1:  # pragma: no cover — defensive
                raise RuntimeError(
                    f"shard {shard_id} range {range_idx}: indptr slice has "
                    f"{local_indptr.shape[0]} entries, expected {n_rows + 1}",
                )

            with open(tmp_data_path, "wb", buffering=1 << 22) as fd_data, open(
                tmp_idx_path, "wb", buffering=1 << 22,
            ) as fd_idx:
                for r0 in range(0, n_rows, inner):
                    r1 = min(r0 + inner, n_rows)
                    a = int(local_indptr[r0])
                    b = int(local_indptr[r1])
                    if b == a:
                        # All-empty rows in this sub-chunk; per_row_nnz stays 0.
                        continue

                    data_slab = xdata[a:b]
                    idx_slab = xindices[a:b]

                    # HVG-keep mask: bool over the slab's nnz.
                    keep = hvg_col_mask[idx_slab]

                    # Per-row HVG nnz: distribute keep flags onto rows via repeat.
                    row_lengths = np.diff(local_indptr[r0 : r1 + 1])
                    if int(row_lengths.sum()) != (b - a):  # pragma: no cover — defensive
                        raise RuntimeError(
                            f"shard {shard_id} range {range_idx}: "
                            f"row_lengths.sum()={int(row_lengths.sum())} "
                            f"!= b-a={b - a} (corrupted indptr?)",
                        )
                    row_id = np.repeat(
                        np.arange(r1 - r0, dtype=np.int32), row_lengths,
                    )
                    per_row_nnz[r0:r1] = np.bincount(
                        row_id[keep], minlength=r1 - r0,
                    )

                    # Filter and remap; write to tmp binaries.
                    if keep.any():
                        data_kept = data_slab[keep]
                        idx_kept = hvg_index_map[idx_slab[keep]]

                        if data_kept.size:
                            m = int(data_kept.max())
                            if m > _UINT16_MAX:
                                raise OverflowError(
                                    f"shard {shard_id} range {range_idx}: "
                                    f"max raw count {m} exceeds uint16 max "
                                    f"({_UINT16_MAX}); rebuild with int32.",
                                )
                            if m > max_count:
                                max_count = m

                        data_u16 = data_kept.astype(np.uint16, copy=False)
                        idx_u16 = idx_kept.astype(np.uint16, copy=False)
                        fd_data.write(data_u16.tobytes())
                        fd_idx.write(idx_u16.tobytes())
                        range_nnz += int(data_u16.size)
    except BaseException:
        # Never leave half-written tmp files behind.
        for p in (tmp_data_path, tmp_idx_path):
            try:
                if p.is_file():
                    p.unlink()
            except OSError:
                pass
        raise

    return RangeResult(
        shard_id=shard_id,
        range_idx=range_idx,
        row_start=row_start,
        row_end=row_end,
        per_row_nnz=per_row_nnz,
        tmp_data_path=tmp_data_path,
        tmp_idx_path=tmp_idx_path,
        nnz=range_nnz,
        max_count_observed=max_count,
    )


# --- Master-side: assemble all ranges into the final memmap ----------------


def _blit_uint16_into_memmap(
    src: Path,
    dst_mm: np.ndarray,
    dst_offset: int,
    total_n: int,
) -> None:
    """Copy ``total_n`` uint16 elements from ``src`` into ``dst_mm[dst_offset:]``.

    Streamed in ``_BLIT_CHUNK_BYTES`` chunks so peak RAM stays bounded
    regardless of per-range nnz.     ``dst_mm.dtype`` may differ from uint16; numpy's slice assignment will
    upcast safely since uint16 values fit in any wider integer dtype.
    wider signed integer dtype we use here.
    """
    if total_n <= 0:
        return
    chunk_n = max(1, _BLIT_CHUNK_BYTES // 2)
    pos = 0
    with open(src, "rb") as f:
        while pos < total_n:
            want = min(chunk_n, total_n - pos)
            buf = f.read(want * 2)
            n_read = len(buf) // 2
            if n_read == 0:
                raise EOFError(
                    f"{src}: expected {total_n} u16 but EOF at {pos}",
                )
            arr = np.frombuffer(buf, dtype=np.uint16)
            dst_mm[dst_offset + pos : dst_offset + pos + n_read] = arr
            pos += n_read


def assemble_shard(
    shard_dir: Path,
    range_results: Sequence[RangeResult],
    *,
    keep_tmp: bool = False,
) -> AssembleResult:
    """Concat per-row nnz → indptr; sequential blit of tmp files → memmap.

    Validates that ranges are contiguous and exactly cover ``[0, n_obs)``;
    enforces ``sum(range.nnz) == sum(per_row_nnz)`` so a mid-range crash
    cannot silently produce a truncated cache.
    """
    if not range_results:
        raise ValueError("assemble_shard called with no range results.")

    sorted_results = sorted(range_results, key=lambda r: r.row_start)
    cursor = 0
    for r in sorted_results:
        if r.row_start != cursor:
            raise ValueError(
                f"non-contiguous ranges: expected start={cursor}, "
                f"got [{r.row_start}, {r.row_end})",
            )
        if r.row_end <= r.row_start:
            raise ValueError(f"empty range result [{r.row_start}, {r.row_end})")
        cursor = r.row_end
    n_obs = cursor

    all_per_row_nnz = np.concatenate([r.per_row_nnz for r in sorted_results])
    if all_per_row_nnz.shape[0] != n_obs:  # pragma: no cover — defensive
        raise RuntimeError(
            f"per_row_nnz total {all_per_row_nnz.shape[0]} != n_obs {n_obs}",
        )

    indptr = np.zeros(n_obs + 1, dtype=np.int64)
    np.cumsum(all_per_row_nnz, out=indptr[1:])
    nnz_total = int(indptr[-1])
    max_count = max(int(r.max_count_observed) for r in sorted_results)

    range_nnz_sum = sum(int(r.nnz) for r in sorted_results)
    if range_nnz_sum != nnz_total:
        raise RuntimeError(
            f"nnz mismatch: sum(range.nnz)={range_nnz_sum} but "
            f"sum(per_row_nnz)={nnz_total}",
        )

    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    data_path = shard_dir / "data.npy"
    idx_path = shard_dir / "indices.npy"
    indptr_path = shard_dir / "indptr.npy"

    data_mm = np.lib.format.open_memmap(
        data_path, mode="w+", dtype=np.uint16, shape=(nnz_total,),
    )
    idx_mm = np.lib.format.open_memmap(
        idx_path, mode="w+", dtype=np.uint16, shape=(nnz_total,),
    )

    offset = 0
    try:
        for r in sorted_results:
            if r.nnz == 0:
                continue
            _blit_uint16_into_memmap(r.tmp_data_path, data_mm, offset, r.nnz)
            _blit_uint16_into_memmap(r.tmp_idx_path, idx_mm, offset, r.nnz)
            offset += r.nnz
        if offset != nnz_total:  # pragma: no cover — defensive
            raise RuntimeError(
                f"blit cursor {offset} != nnz_total {nnz_total}",
            )
        # Persist once so opens skip a full mmap np.max.
        max_col = int(np.max(idx_mm)) if nnz_total else -1
    finally:
        data_mm.flush()
        idx_mm.flush()
        del data_mm, idx_mm

    np.save(indptr_path, indptr)

    if not keep_tmp:
        for r in sorted_results:
            for p in (r.tmp_data_path, r.tmp_idx_path):
                try:
                    Path(p).unlink()
                except OSError:
                    logger.warning("Failed to unlink tmp %s", p)

    return AssembleResult(
        n_obs=n_obs,
        nnz=nnz_total,
        max_count_observed=max_count,
        max_col=max_col,
    )


def cleanup_range_tmps(results: Sequence[RangeResult]) -> None:
    """Best-effort unlink of per-range temp files (use on failure paths)."""
    for r in results:
        for p in (r.tmp_data_path, r.tmp_idx_path):
            try:
                Path(p).unlink()
            except OSError:
                pass



logger = logging.getLogger(__name__)


def _term(msg: str) -> None:
    """Stdout progress. ``logger.info`` is silent under ``src.main`` unless logging is configured."""
    print(f"  [train_cache] {msg}", flush=True)


REPO_ROOT = Path(__file__).resolve().parents[3]  # src/data/cache/ -> PertGen (parents[2] is src/)

PLATE_GLOB = "plate*_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"
PLATE_RE = re.compile(
    r"^plate(\d+)_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab\.h5ad$"
)

DEFAULT_HVG_PATH = REPO_ROOT / "core_util_data" / "shared" / "gene_subsets" / "total_hallmark_genes.pkl"

# Sidecar layout for obs columns:
#   <key>__codes      : (n_obs,) int{8|16|32}   pd.Categorical.cat.codes (smallest fit)
#   <key>__categories : (n_unique,) object       unique labels in code order
# Plus the always-present integer columns local_obs_index (int64) and shard (int64).
# We deliberately do NOT save obs_names — nothing downstream reads it and it is
# the dominant pickle cost on read.

DEFAULT_CHUNK_ROWS = DEFAULT_INNER_CHUNK_ROWS

# Default number of row-range workers per shard.
DEFAULT_RANGES_PER_SHARD = 4


# ---------------------------------------------------------------------------
# Discovery & hashing helpers
# ---------------------------------------------------------------------------


def discover_shard_paths(data_dir: Path, files_glob: str) -> List[Tuple[int, Path]]:
    """Return ``(shard_id, path)`` sorted by shard_id.

    Attempts to extract an integer shard ID from filenames like ``plate_01.h5ad``
    or ``shard_12.h5ad``. If no integer is found, shards are numbered sequentially.
    """
    found: List[Tuple[int, Path]] = []
    
    import re
    # Match "plate_01", "shard01", "plate1", etc.
    SHARD_RE = re.compile(r"(?i)(?:plate|shard)[^\d]*(\d+)")
    
    for f in data_dir.glob(files_glob):
        m = SHARD_RE.search(f.name)
        if m:
            found.append((int(m.group(1)), f))
        else:
            found.append((-1, f))
            
    # Sort first by id (if present), then by filename
    found.sort(key=lambda x: (x[0] if x[0] != -1 else 0, x[1].name))
    
    final_found = []
    for i, (old_id, f) in enumerate(found, start=1):
        if old_id != -1:
            final_found.append((old_id, f))
        else:
            final_found.append((i, f))
            
    logger.debug(
        "Discovered %d shard(s) under %s (glob=%s)",
        len(final_found),
        data_dir,
        files_glob,
    )
    if final_found:
        logger.debug(
            "Shard id range: %d … %d",
            final_found[0][0],
            final_found[-1][0],
        )
    return final_found


def sha256_file(path: Path) -> str:
    """SHA-256 hex digest of file contents (1 MB chunks)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Per-shard worker
# ---------------------------------------------------------------------------


def _meta_matches(
    shard_dir: Path,
    n_obs: int,
    n_hvg: int,
    h5ad_mtime: float,
    hvg_hash: str,
) -> bool:
    """Cache-hit check: per-shard artefacts are present and meta matches."""
    meta_path = shard_dir / "meta.json"
    needed = (
        shard_dir / "data.npy",
        shard_dir / "indices.npy",
        shard_dir / "indptr.npy",
        meta_path,
    )
    sidecar_dir = shard_dir / OBS_SIDECAR_DIR
    sidecar_npz = shard_dir / OBS_SIDECAR_NPZ
    has_sidecar = (
        sidecar_dir.is_dir() and any(sidecar_dir.glob("*.npy"))
    ) or sidecar_npz.is_file()
    if not all(p.is_file() for p in needed) or not has_sidecar:
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        int(meta.get("n_obs", -1)) == n_obs
        and int(meta.get("n_hvg", -1)) == n_hvg
        and abs(float(meta.get("h5ad_mtime", 0.0)) - float(h5ad_mtime)) < 1e-6
        and meta.get("hvg_file_sha256") == hvg_hash
        and meta.get("transform") == TRANSFORM_TAG
    )


def _save_obs_sidecar(sidecar: Dict[str, np.ndarray], shard_dir: Path) -> None:
    """Write mmap-friendly flat ``.npy`` arrays under ``obs_sidecar/``."""
    out_dir = shard_dir / OBS_SIDECAR_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    legacy_npz = shard_dir / OBS_SIDECAR_NPZ
    if legacy_npz.is_file():
        legacy_npz.unlink()
    for key, arr in sidecar.items():
        np.save(out_dir / f"{key}.npy", arr)


def _build_obs_sidecar(adata: ad.AnnData, shard_id: int, obs_keys: List[str]) -> Dict[str, np.ndarray]:
    """Extract obs columns as numpy arrays; row order matches adata.obs row order.

    String-valued columns in `obs_keys` are stored as
    ``<key>__codes`` (smallest-fit pandas Categorical codes) and
    ``<key>__categories`` (object array of the unique labels). On read this
    avoids pickling 100M-row object arrays — the categories array has only as
    many entries as unique labels, so its pickle cost is negligible.
    """
    n_obs = adata.n_obs
    out: Dict[str, np.ndarray] = {
        "local_obs_index": np.arange(n_obs, dtype=np.int64),
        "shard": np.full(n_obs, shard_id, dtype=np.int64),
    }
    for key in obs_keys:
        if key not in adata.obs.columns:
            raise KeyError(
                f"Missing obs column {key!r} in shard {shard_id}. "
                f"Available: {list(adata.obs.columns)}"
            )
        # ``astype("category")`` is idempotent for already-categorical columns.
        # cat.codes uses the smallest int dtype that fits len(categories) (int8/16/32),
        # so storage stays compact even for high-cardinality fields like drugname_drugconc.
        s = adata.obs[key].astype("category")
        out[f"{key}__codes"] = s.cat.codes.to_numpy()
        out[f"{key}__categories"] = np.asarray(
            list(s.cat.categories.astype(str)), dtype=object
        )
    return out


def _resolve_hvg_indices(
    hvg_path: Path,
    reference_h5ad: Path,
) -> Tuple[np.ndarray, List[str]]:
    """Map HVG file → column indices once (shared across shards with same var_names)."""
    adata = ad.read_h5ad(reference_h5ad, backed="r")
    try:
        var_names = [str(x) for x in adata.var_names]
        hvg_indices, genes = load_hvg_indices_from_genes_file(hvg_path, var_names)
        present_names = [var_names[int(i)] for i in hvg_indices]
        missing_n = len(genes) - len(present_names)
        logger.info(
            "HVG mapping (%s): %d genes in file → %d columns used (%d missing)",
            reference_h5ad.name,
            len(genes),
            len(present_names),
            missing_n,
        )
        return hvg_indices, present_names
    finally:
        adata.file.close()


class _ShardProgress:
    """Parent-process shard completion tracker (ETA + one line per shard)."""

    def __init__(self, discovered: Sequence[Tuple[int, Path]]) -> None:
        self.total = len(discovered)
        self.done = 0
        self.built = 0
        self.skipped = 0
        self.t0 = time.monotonic()
        self._file_names = {sid: path.name for sid, path in discovered}

    def log_plan(self) -> None:
        ids = ", ".join(f"{sid:03d}" for sid, _ in sorted(self._file_names.items()))
        logger.info("Shard queue (%d): %s", self.total, ids)
        _term(f"shard queue ({self.total}): {ids}")

    def on_done(self, rep: Dict[str, Any]) -> None:
        self.done += 1
        if rep.get("skipped"):
            self.skipped += 1
        else:
            self.built += 1
        sid = int(rep["shard_id"])
        elapsed = time.monotonic() - self.t0
        eta = (elapsed / self.done) * (self.total - self.done) if self.done else 0.0
        tag = "SKIP " if rep.get("skipped") else "BUILD"
        n_obs = int(rep.get("n_obs", 0))
        n_hvg = int(rep.get("n_hvg", 0))
        nnz = int(rep.get("nnz", 0))
        sparsity = 100.0 * nnz / max(1, n_obs * n_hvg) if nnz else 0.0
        shard_wall = rep.get("wall_seconds")
        shard_s = f" shard={shard_wall:.0f}s" if shard_wall is not None else ""
        fname = self._file_names.get(sid, "?")
        logger.info(
            "[%d/%d] %s shard %03d | %s | n_obs=%s n_hvg=%d nnz=%s sparsity=%.1f%%%s | "
            "elapsed=%.0fs ETA~%.0fs",
            self.done,
            self.total,
            tag,
            sid,
            fname,
            f"{n_obs:,}",
            n_hvg,
            f"{nnz:,}" if nnz else "—",
            sparsity,
            shard_s,
            elapsed,
            eta,
        )
        _term(
            f"plates {self.done}/{self.total} {tag.strip()} shard {sid:03d} | {fname} "
            f"| n_obs={n_obs:,} ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)"
        )


class _RangeJobProgress:
    """Track row-range worker completion (logs before shard finalisation)."""

    def __init__(self, total_jobs: int) -> None:
        self.total = int(total_jobs)
        self.done = 0
        self.t0 = time.monotonic()

    def on_range_done(
        self,
        *,
        shard_id: int,
        range_idx: int,
        res: RangeResult,
        pending_ranges_in_shard: int,
    ) -> None:
        self.done += 1
        elapsed = time.monotonic() - self.t0
        eta = (elapsed / self.done) * (self.total - self.done) if self.done else 0.0
        n_rows = int(res.row_end - res.row_start)
        logger.info(
            "Range [%d/%d] shard=%03d range=%d rows=%s nnz=%s | "
            "pending_in_shard=%d | elapsed=%.0fs ETA~%.0fs",
            self.done,
            self.total,
            shard_id,
            range_idx,
            f"{n_rows:,}",
            f"{res.nnz:,}",
            pending_ranges_in_shard,
            elapsed,
            eta,
        )
        _term(
            f"range {self.done}/{self.total} | shard {shard_id:03d} range={range_idx} "
            f"rows={n_rows:,} nnz={res.nnz:,} | pending_in_shard={pending_ranges_in_shard} "
            f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)"
        )


def _resolve_shard_dir(out_root: Path, shard_id: int) -> Path:
    """Convention: ``<out_root>/shard_<NNN>/``. Centralised so callers can't drift."""
    return out_root / f"shard_{shard_id:03d}"


def _resolve_tmp_dir(out_root: Path) -> Path:
    """Per-cache scratch space for per-range temp binaries.

    Sibling of the shard subdirs so it shares the same disk (same write
    bandwidth) and is trivially cleaned up by deleting the cache root.
    """
    return out_root / "_tmp"


def _make_skipped_report(
    shard_id: int, h5ad_path: Path, n_obs: int, n_hvg: int, shard_dir: Path,
) -> Dict[str, Any]:
    """Cache-hit report — skip path, no work done."""
    return {
        "shard_id": shard_id,
        "h5ad_path": str(h5ad_path),
        "n_obs": n_obs,
        "n_hvg": n_hvg,
        "skipped": True,
        "path": str(shard_dir),
    }


def _finalize_shard(
    *,
    shard_id: int,
    h5ad_path: Path,
    out_root: Path,
    range_results: Sequence[RangeResult],
    obs_keys: List[str],
    hvg_indices: np.ndarray,
    hvg_hash: str,
    chunk_rows: int,
    started_at: float,
) -> Dict[str, Any]:
    """Master-side per-shard finalisation: assemble + obs sidecar + meta.json.

    Runs once per shard, *after* all of that shard's range workers have
    returned. Builds the final ``data.npy`` / ``indices.npy`` / ``indptr.npy``
    by blitting the per-range temp binaries; opens AnnData once for the obs
    sidecar; writes ``meta.json``.
    """
    h5ad_p = Path(h5ad_path)
    out_root_p = Path(out_root)
    shard_dir = _resolve_shard_dir(out_root_p, shard_id)
    shard_dir.mkdir(parents=True, exist_ok=True)

    n_hvg = int(hvg_indices.shape[0])
    h5ad_mtime = h5ad_p.stat().st_mtime

    # 1) Assemble: per-row nnz -> indptr; sequential blit of tmp -> memmap.
    t_assemble = time.monotonic()
    asm = assemble_shard(shard_dir, range_results)
    assemble_dt = time.monotonic() - t_assemble
    logger.debug(
        "Shard %d: assemble done n_obs=%s nnz=%s (%.1fs)",
        shard_id, f"{asm.n_obs:,}", f"{asm.nnz:,}", assemble_dt,
    )

    # 2) Obs sidecar: open AnnData backed once, build categorical codes, save.
    t_obs = time.monotonic()
    adata = ad.read_h5ad(h5ad_p, backed="r")
    try:
        if int(adata.n_obs) != asm.n_obs:
            raise RuntimeError(
                f"Shard {shard_id}: AnnData n_obs={adata.n_obs} disagrees with "
                f"range-aggregated n_obs={asm.n_obs}; h5ad changed under us?",
            )
        sidecar = _build_obs_sidecar(adata, shard_id, obs_keys)
        var_names = [str(x) for x in adata.var_names]
    finally:
        adata.file.close()
    _save_obs_sidecar(sidecar, shard_dir)
    obs_dt = time.monotonic() - t_obs

    gene_names = [var_names[int(i)] for i in hvg_indices]

    # 3) meta.json — the schema the readers expect.
    meta = {
        "shard_id": shard_id,
        "h5ad_path": str(h5ad_p.resolve()),
        "h5ad_mtime": h5ad_mtime,
        "n_obs": asm.n_obs,
        "n_hvg": n_hvg,
        "nnz": asm.nnz,
        "max_col": asm.max_col,
        "hvg_genes_path": "",  # filled in by caller (run_raw_hvg_cache_build)
        "hvg_file_sha256": hvg_hash,
        "data_dtype": "uint16",
        "indices_dtype": "uint16",
        "indptr_dtype": "int64",
        "max_count_observed": asm.max_count_observed,
        "chunk_rows": chunk_rows,
        "transform": TRANSFORM_TAG,
        "version": CACHE_VERSION,
    }
    # NOTE: the caller patches `hvg_genes_path` because the path is a runtime
    # decision (not a per-shard concern); the writer below picks it up.
    return {
        "shard_id": shard_id,
        "h5ad_path": str(h5ad_p),
        "n_obs": asm.n_obs,
        "n_hvg": n_hvg,
        "nnz": asm.nnz,
        "max_count_observed": asm.max_count_observed,
        "skipped": False,
        "path": str(shard_dir),
        "gene_names": gene_names,
        "wall_seconds": time.monotonic() - started_at,
        "assemble_seconds": assemble_dt,
        "obs_sidecar_seconds": obs_dt,
        "meta_partial": meta,
    }


def _write_shard_meta(
    shard_dir: Path, meta_partial: Dict[str, Any], hvg_path: Path,
) -> None:
    """Stamp the runtime-resolved ``hvg_genes_path`` and write ``meta.json``."""
    meta = dict(meta_partial)
    meta["hvg_genes_path"] = str(hvg_path.resolve())
    (shard_dir / "meta.json").write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _write_manifest(
    out_root: Path,
    data_dir: Path,
    hvg_path: Path,
    hvg_hash: str,
    chunk_rows: int,
    shard_reports: List[Dict[str, Any]],
    gene_names: List[str],
    *,
    condition_variables: Optional[Sequence[str]] = None,
    obs_sidecar_keys: Optional[Sequence[str]] = None,
) -> None:
    """Write root ``manifest.json`` (AnnCollection-style global obs offsets)."""
    offset = 0
    shards_out: List[Dict[str, Any]] = []
    for rep in sorted(shard_reports, key=lambda r: r["shard_id"]):
        n = int(rep["n_obs"])
        shards_out.append(
            {
                "shard_id": int(rep["shard_id"]),
                "n_obs": n,
                "global_obs_offset": offset,
                "cache_subdir": f"shard_{int(rep['shard_id']):03d}",
                "nnz": int(rep.get("nnz", 0)),
                "max_count_observed": int(rep.get("max_count_observed", 0)),
            }
        )
        offset += n

    n_hvg = int(shard_reports[0]["n_hvg"]) if shard_reports else 0

    manifest = {
        "version": CACHE_VERSION,
        "data_dir": str(data_dir.resolve()),
        "hvg_genes_path": str(hvg_path.resolve()),
        "hvg_file_sha256": hvg_hash,
        "chunk_rows": chunk_rows,
        "total_obs": offset,
        "n_shards": len(shards_out),
        "n_hvg": n_hvg,
        "data_dtype": "uint16",
        "indices_dtype": "uint16",
        "indptr_dtype": "int64",
        "transform": TRANSFORM_TAG,
        "shards": shards_out,
        "data_rel_path": "data.npy",
        "indices_rel_path": "indices.npy",
        "indptr_rel_path": "indptr.npy",
        "obs_sidecar_rel_path": f"{OBS_SIDECAR_DIR}/",
        "obs_sidecar_format": "npy_dir",
        "obs_sidecar_legacy_npz": OBS_SIDECAR_NPZ,
    }
    if condition_variables is not None:
        manifest["condition_variables"] = list(condition_variables)
    if obs_sidecar_keys is not None:
        manifest["obs_sidecar_keys"] = list(obs_sidecar_keys)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Wrote %s", out_root / "manifest.json")

    # Persist HVG gene names at the cache root (matches per-shard column order).
    if gene_names:
        (out_root / "hvg_gene_names.txt").write_text(
            "\n".join(gene_names) + "\n"
        )
        logger.info(
            "Wrote %s (%d genes)",
            out_root / "hvg_gene_names.txt", len(gene_names),
        )


def obs_keys_for_raw_hvg_cache(obs_keys: Sequence[str]) -> List[str]:
    """Obs columns to persist in the sidecar (matches :meth:`PertGenDataModule._load_collection`).

    ``dose`` is derived at cache read time from ``drugname_drugconc``, so the
    builder stores the source column instead of ``dose`` when needed.
    """
    keys = list(obs_keys)
    if "dose" in keys and "drugname_drugconc" not in keys:
        keys = [k for k in keys if k != "dose"] + ["drugname_drugconc"]
    return keys


def _probe_shard_for_build(
    *,
    shard_id: int,
    h5ad_path: Path,
    out_root: Path,
    n_hvg: int,
    hvg_hash: str,
    force: bool,
) -> Tuple[bool, int, Optional[Dict[str, Any]]]:
    """Cheap pre-flight: ``(needs_build, n_obs, skipped_report_or_None)``.

    Probes ``n_obs`` via h5py (no AnnData open) and consults
    :func:`_meta_matches`. Skipped shards still need a report so the manifest
    includes them.
    """
    n_obs = probe_n_obs(h5ad_path)
    h5ad_mtime = h5ad_path.stat().st_mtime
    shard_dir = _resolve_shard_dir(out_root, shard_id)

    if not force and _meta_matches(shard_dir, n_obs, n_hvg, h5ad_mtime, hvg_hash):
        logger.debug(
            "Shard %d: skip (cache hit) n_obs=%s n_hvg=%d",
            shard_id, f"{n_obs:,}", n_hvg,
        )
        return False, n_obs, _make_skipped_report(
            shard_id, h5ad_path, n_obs, n_hvg, shard_dir,
        )
    return True, n_obs, None


def cache_has_complete_csr(cache_dir: Union[str, Path]) -> bool:
    """True when ``manifest.json`` and ``hvg_gene_names.txt`` exist under ``cache_dir``."""
    root = Path(cache_dir)
    return (root / "manifest.json").is_file() and (root / "hvg_gene_names.txt").is_file()


def load_gene_names_from_cache(cache_dir: Union[str, Path]) -> List[str]:
    """Read HVG gene names from an on-disk cache (column order for the mmap CSR)."""
    path = Path(cache_dir) / "hvg_gene_names.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing hvg_gene_names.txt under {cache_dir}")
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def run_raw_hvg_cache_build(
    *,
    data_dir: Union[str, Path],
    hvg_genes_path: Union[str, Path],
    cache_dir: Union[str, Path],
    obs_keys: Sequence[str],
    files_glob: str = "*.h5ad",
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    n_workers: int = 8,
    shards_filter: Optional[set[int]] = None,
    force: bool = False,
    ranges_per_shard: int = DEFAULT_RANGES_PER_SHARD,
    condition_variables: Optional[Sequence[str]] = None,
    control_condition: Optional[Mapping[str, Any]] = None,
    control_match_keys: Optional[Sequence[str]] = None,
    normalization_id: str = "raw_counts",
) -> dict[str, Any]:
    """Build mmap CSR HVG cache under ``cache_dir``.

    When the CSR cache already exists and ``force`` is false, returns without
    rebuilding shards.
    """
    try:
        mp.set_start_method("fork", force=False)
    except RuntimeError:
        pass

    data_dir = Path(data_dir).resolve()
    hvg_path = Path(hvg_genes_path).resolve()
    out_root = Path(cache_dir).resolve()
    sidecar_keys = obs_keys_for_raw_hvg_cache(obs_keys)
    mcv = list(condition_variables) if condition_variables is not None else list(obs_keys)

    if not force and cache_has_complete_csr(out_root):
        _term(f"skip | cache complete at {out_root}")
        return {
            "cache_dir": str(out_root),
            "mode": "existing_cache",
            "csr_rebuild": False,
        }

    if not data_dir.is_dir():
        raise FileNotFoundError(f"data_dir is not a directory: {data_dir}")
    if not hvg_path.is_file():
        raise FileNotFoundError(f"HVG genes file not found: {hvg_path}")

    discovered = discover_shard_paths(data_dir, files_glob)
    if shards_filter is not None:
        before_filter = len(discovered)
        discovered = [(i, p) for i, p in discovered if i in shards_filter]
        logger.info(
            "Shard filter: %s → %d shard(s) (from %d discovered)",
            sorted(shards_filter), len(discovered), before_filter,
        )
        if not discovered:
            raise FileNotFoundError(
                f"No shards matched filter {sorted(shards_filter)} under {data_dir}",
            )
    if not discovered:
        raise FileNotFoundError(
            f"No files matching {files_glob!r} under {data_dir}",
        )

    out_root.mkdir(parents=True, exist_ok=True)
    hvg_hash = sha256_file(hvg_path)

    # Resolve HVG once (shared across shards with identical var_names).
    hvg_indices, gene_names = _resolve_hvg_indices(hvg_path, discovered[0][1])
    n_hvg = int(hvg_indices.shape[0])
    if n_hvg >= np.iinfo(np.uint16).max:
        raise ValueError(
            f"n_hvg={n_hvg} >= 65535; uint16 indices dtype no longer fits. "
            "Update the cache builder dtype.",
        )

    # n_var probe (one-shot, AnnData backed) + HVG col mask + remap. Mask and
    # remap are dense over the full `n_var` so workers can index them with
    # raw column ids straight out of `X/indices`.
    ref_adata = ad.read_h5ad(discovered[0][1], backed="r")
    try:
        n_var = int(ref_adata.n_vars)
    finally:
        ref_adata.file.close()
    hvg_col_mask, hvg_index_map = compute_hvg_col_mask(n_var, hvg_indices)

    # Pre-flight: classify each shard as build-or-skip; collect skipped reports.
    progress = _ShardProgress(discovered)
    progress.log_plan()

    reports: List[Dict[str, Any]] = []
    to_build: List[Tuple[int, Path, int]] = []  # (shard_id, h5ad_path, n_obs)
    for shard_id, h5ad_path in discovered:
        needs_build, n_obs, skip_rep = _probe_shard_for_build(
            shard_id=shard_id, h5ad_path=h5ad_path, out_root=out_root,
            n_hvg=n_hvg, hvg_hash=hvg_hash, force=force,
        )
        if not needs_build:
            reports.append(skip_rep)  # type: ignore[arg-type]
            progress.on_done(skip_rep)  # type: ignore[arg-type]
        else:
            to_build.append((shard_id, h5ad_path, n_obs))

    if not to_build:
        # Nothing to do — write manifest from the skipped reports and bail.
        _write_manifest(
            out_root,
            data_dir,
            hvg_path,
            hvg_hash,
            chunk_rows,
            reports,
            gene_names,
            condition_variables=mcv,
            obs_sidecar_keys=sidecar_keys,
        )
        return _build_summary_dict(
            out_root, data_dir, reports, gene_names, elapsed=0.0,
        )

    # Plan row ranges per shard; flatten to a single job list.
    job_specs: List[Tuple[int, int, Path, int, int]] = []  # (shard_id, range_idx, path, r_start, r_end)
    n_ranges_by_shard: Dict[int, int] = {}
    h5ad_by_shard: Dict[int, Path] = {}
    started_by_shard: Dict[int, float] = {}
    for shard_id, h5ad_path, n_obs in to_build:
        ranges = plan_row_ranges(n_obs, ranges_per_shard)
        if not ranges:
            raise RuntimeError(
                f"Shard {shard_id}: plan_row_ranges returned empty (n_obs={n_obs})",
            )
        n_ranges_by_shard[shard_id] = len(ranges)
        h5ad_by_shard[shard_id] = h5ad_path
        for ri, (a, b) in enumerate(ranges):
            job_specs.append((shard_id, ri, h5ad_path, a, b))

    requested_workers = int(n_workers)
    n_workers_eff = max(1, min(requested_workers, len(job_specs)))
    if requested_workers > len(job_specs):
        logger.warning(
            "Requested %d workers but only %d range job(s); using %d",
            requested_workers, len(job_specs), n_workers_eff,
        )

    tmp_dir = _resolve_tmp_dir(out_root)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Raw HVG cache build | data_dir=%s cache_dir=%s shards_to_build=%d "
        "ranges_per_shard=%d total_jobs=%d workers=%d chunk_rows=%d force=%s",
        data_dir, out_root, len(to_build), ranges_per_shard,
        len(job_specs), n_workers_eff, chunk_rows, force,
    )
    logger.info(
        "Progress: one INFO line per range job (%d total), then one per shard "
        "finalised (%d total)",
        len(job_specs),
        len(to_build),
    )
    _term(
        f"BUILDING | {len(to_build)} plates, {len(job_specs)} range jobs, "
        f"{n_workers_eff} workers, out={out_root}"
    )

    t0 = time.monotonic()
    range_progress = _RangeJobProgress(len(job_specs))
    pending_by_shard: Dict[int, int] = dict(n_ranges_by_shard)
    range_results_by_shard: Dict[int, List[RangeResult]] = {
        sid: [] for sid in n_ranges_by_shard
    }
    for sid in n_ranges_by_shard:
        started_by_shard[sid] = t0

    ctx = mp.get_context("fork")
    try:
        with ProcessPoolExecutor(max_workers=n_workers_eff, mp_context=ctx) as ex:
            future_to_meta = {
                ex.submit(
                    process_row_range,
                    shard_id=sid,
                    range_idx=ri,
                    h5ad_path=str(path),
                    row_start=a,
                    row_end=b,
                    hvg_col_mask=hvg_col_mask,
                    hvg_index_map=hvg_index_map,
                    tmp_dir=str(tmp_dir),
                    inner_chunk_rows=chunk_rows,
                ): (sid, ri)
                for (sid, ri, path, a, b) in job_specs
            }
            for fut in as_completed(future_to_meta):
                sid, ri = future_to_meta[fut]
                try:
                    res = fut.result()
                except Exception:
                    logger.exception("Range %d:%d FAILED", sid, ri)
                    raise
                range_results_by_shard[sid].append(res)
                pending_by_shard[sid] -= 1
                range_progress.on_range_done(
                    shard_id=sid,
                    range_idx=ri,
                    res=res,
                    pending_ranges_in_shard=pending_by_shard[sid],
                )
                if pending_by_shard[sid] == 0:
                    logger.info(
                        "Shard %03d: all ranges done — assembling + obs sidecar",
                        sid,
                    )
                    _term(f"shard {sid:03d} | all ranges done, assembling + obs sidecar")
                    rep = _finalize_shard(
                        shard_id=sid,
                        h5ad_path=h5ad_by_shard[sid],
                        out_root=out_root,
                        range_results=range_results_by_shard[sid],
                        obs_keys=sidecar_keys,
                        hvg_indices=hvg_indices,
                        hvg_hash=hvg_hash,
                        chunk_rows=chunk_rows,
                        started_at=started_by_shard[sid],
                    )
                    _write_shard_meta(
                        _resolve_shard_dir(out_root, sid),
                        rep.pop("meta_partial"),
                        hvg_path,
                    )
                    reports.append(rep)
                    progress.on_done(rep)
                    # Drop tmp results to free their per_row_nnz arrays early.
                    range_results_by_shard[sid] = []
    except BaseException:
        # On any failure, scrub tmp files we know about so the next run
        # doesn't pick up stale state.
        for results in range_results_by_shard.values():
            cleanup_range_tmps(results)
        raise

    # Best-effort: remove the tmp dir if it's empty.
    try:
        next(tmp_dir.iterdir())
    except StopIteration:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except OSError:
            pass

    _write_manifest(
        out_root,
        data_dir,
        hvg_path,
        hvg_hash,
        chunk_rows,
        reports,
        gene_names,
        condition_variables=mcv,
        obs_sidecar_keys=sidecar_keys,
    )

    elapsed = time.monotonic() - t0
    return _build_summary_dict(
        out_root, data_dir, reports, gene_names, elapsed=elapsed,
    )


def _build_summary_dict(
    out_root: Path,
    data_dir: Path,
    reports: Sequence[Dict[str, Any]],
    gene_names: List[str],
    *,
    elapsed: float,
) -> Dict[str, Any]:
    """Fold per-shard reports into the final summary + log a one-liner."""
    n_skipped = sum(1 for r in reports if r.get("skipped"))
    total_obs = sum(int(r.get("n_obs", 0)) for r in reports)
    total_nnz = sum(int(r.get("nnz", 0)) for r in reports)
    summary = {
        "cache_dir": str(out_root),
        "data_dir": str(data_dir),
        "n_shards": len(reports),
        "n_built": len(reports) - n_skipped,
        "n_skipped": n_skipped,
        "elapsed_seconds": float(elapsed),
        "gene_names": gene_names,
    }
    logger.info(
        "Raw HVG cache DONE | shards=%d built=%d skipped=%d total_obs=%s "
        "total_nnz=%s wall=%.1fs",
        summary["n_shards"], summary["n_built"], summary["n_skipped"],
        f"{total_obs:,}", f"{total_nnz:,}", elapsed,
    )
    _term(
        f"complete in {elapsed:.1f}s | shards={summary['n_shards']} "
        f"built={summary['n_built']} skipped={summary['n_skipped']} "
        f"total_obs={total_obs:,}"
    )
    return summary

__all__ = [
    "AssembleResult",
    "DEFAULT_CHUNK_ROWS",
    "DEFAULT_INNER_CHUNK_ROWS",
    "DEFAULT_RANGES_PER_SHARD",
    "RangeResult",
    "assemble_shard",
    "cleanup_range_tmps",
    "compute_hvg_col_mask",
    "cache_has_complete_csr",
    "discover_shard_paths",
    "load_gene_names_from_cache",
    "obs_keys_for_raw_hvg_cache",
    "plan_row_ranges",
    "probe_n_obs",
    "process_row_range",
    "run_raw_hvg_cache_build",
    "sha256_file",
]


def resolve_force_csr(*, force_flag: bool, explicit_cache_dir: bool, config_overwrite_cache: bool) -> bool:
    """Whether to rebuild CSR shards. ``--force`` wins; the config's
    ``overwrite_cache`` counts only when no ``--cache-dir`` was given, because
    pointing at a specific cache means "use this one as-is"."""
    if force_flag:
        return True
    return False if explicit_cache_dir else bool(config_overwrite_cache)

