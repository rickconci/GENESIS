"""The HVG training cache: raw counts as one mmap CSR per plate, read a batch at a time.

Layout, written by :mod:`src.data.cache.raw_hvg_cache_build`::

    <cache_root>/
      manifest.json             shard list, global row offsets, n_hvg, transform tag
      hvg_gene_names.txt        column order
      shard_<NNN>/
        data.npy  indices.npy   (nnz,) uint16 counts and HVG column ids
        indptr.npy              (n_obs + 1,) int64
        obs_sidecar/            <key>__codes.npy + <key>__categories.npy per obs column,
                                plus shard.npy and local_obs_index.npy
        meta.json               includes max_col, the largest column id on disk

Older caches (the STATE 2k Tahoe and Parse caches among them) hold the same obs
arrays in a single ``obs_sidecar.npz``; both are read.

Why a shard is never a ``scipy.sparse.csr_matrix``. A Tahoe plate holds 1-6 billion
nonzeros. SciPy cannot index uint16 columns, so wrapping a plate casts its indices
to int64: nnz × 8 bytes, 38 GiB for the largest one, in every DataLoader worker. A
shard therefore stays three mmaps, and a fetch copies out only the nonzeros of the
rows it asks for. Memory scales with the batch, not the plate.

Obs is stored as categorical codes plus the label list, so the 100M-row frame is
rebuilt without ever materialising 100M Python strings.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

logger = logging.getLogger(__name__)


# The on-disk schema, shared with the builder

CACHE_VERSION = 2
TRANSFORM_TAG = "raw_counts_hvg_subset_csr_obs_codes_v1"
OBS_SIDECAR_NPZ = "obs_sidecar.npz"
OBS_SIDECAR_DIR = "obs_sidecar"


def normalization_id_from_transform(transform: str) -> str:
    """Map cache ``manifest.transform`` to eval-cache ``normalization_id``."""
    tag = str(transform or "").strip()
    if not tag or tag == TRANSFORM_TAG or "raw_counts" in tag:
        return "raw_counts"
    if "log1p" in tag:
        return "log1p"
    raise ValueError(
        f"Unknown cache transform tag {transform!r}; expected "
        f"{TRANSFORM_TAG!r} or a tag containing 'raw_counts'/'log1p'.",
    )


# A shard: three mmaps, opened without reading a byte

@dataclass(frozen=True)
class MmapCsrTriple:
    """One shard's CSR arrays as mmaps; pages load only when a gather touches them."""

    data: np.ndarray     # (nnz,) uint16
    indices: np.ndarray  # (nnz,) uint16
    indptr: np.ndarray   # (n_obs + 1,) int64
    n_obs: int
    n_hvg: int

    @property
    def nnz(self) -> int:
        return int(self.indptr[-1])

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.n_obs, self.n_hvg)

    @property
    def dtype(self) -> Any:
        return self.data.dtype


def open_triple(shard_dir: Path, n_obs: int, n_hvg: int) -> MmapCsrTriple:
    """Mmap a shard and refuse it if a column id could write past the dense buffer.

    ``toarray()`` trusts ``indices < n_hvg``; a corrupt id is a heap overwrite, not an
    error. The builder records ``max_col`` in ``meta.json`` so this check is O(1). A
    cache without it pays one full scan of the indices, which on a 5B-nnz plate pages
    ~10 GiB through every worker, so that path warns.
    """
    data, indices, indptr = (np.load(shard_dir / f"{n}.npy", mmap_mode="r") for n in ("data", "indices", "indptr"))
    triple = MmapCsrTriple(data, indices, indptr, int(n_obs), int(n_hvg))
    if triple.indptr.shape[0] != n_obs + 1:
        raise ValueError(f"{shard_dir}: indptr has {triple.indptr.shape[0]} entries, expected n_obs+1={n_obs + 1}")
    if triple.data.shape != triple.indices.shape:
        raise ValueError(f"{shard_dir}: data.shape={triple.data.shape} != indices.shape={triple.indices.shape}")
    if triple.indices.size == 0:
        return triple
    meta_path = shard_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    if "max_col" in meta:
        max_col = int(meta["max_col"])
    else:
        logger.warning("%s: meta.json has no max_col; scanning all %d column ids", shard_dir, triple.nnz)
        max_col = int(np.max(triple.indices))
    if max_col >= n_hvg:
        raise ValueError(f"Corrupt CSR cache ({shard_dir}): max column id {max_col} >= n_hvg={n_hvg}. Rebuild the cache.")
    return triple


# The gather: copy out the nonzeros of the requested rows

def _gather_sorted(triple: MmapCsrTriple, rows: np.ndarray) -> sparse.csr_matrix:
    """CSR of ``rows`` (ascending, in bounds), in that order.

    Ascending rows make the mmap reads walk forward through the file, which is what
    the page cache rewards. The nonzeros of row ``r`` sit at ``indptr[r]:indptr[r+1]``;
    one flat ``src`` index of all those positions turns the copy into a single fancy
    index instead of a Python loop over rows.
    """
    starts = triple.indptr[rows].astype(np.int64, copy=False)
    counts = triple.indptr[rows + 1].astype(np.int64, copy=False) - starts
    indptr = np.empty(rows.size + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    total = int(indptr[-1])
    # Output slot k belongs to run r = run[k]; its source is starts[r] + (k - indptr[r]).
    run = np.repeat(np.arange(rows.size, dtype=np.int64), counts)
    src = starts[run] + (np.arange(total, dtype=np.int64) - indptr[run])
    # SciPy's C kernels assume indices and indptr share one dtype; a mismatch makes
    # the later column slice in FetchTransform compute a negative nnz.
    idx_dtype = np.int32 if total < (1 << 31) else np.int64
    return sparse.csr_matrix(
        (np.asarray(triple.data[src]), triple.indices[src].astype(idx_dtype), indptr.astype(idx_dtype, copy=False)),
        shape=(rows.size, triple.n_hvg),
    )


def _inverse(order: np.ndarray) -> np.ndarray:
    """``inv`` with ``sorted[inv] == original`` for ``sorted = original[order]``."""
    inv = np.empty(order.size, dtype=np.int64)
    inv[order] = np.arange(order.size, dtype=np.int64)
    return inv


def gather_rows_from_triple(triple: MmapCsrTriple, local_rows: Sequence[int] | np.ndarray) -> sparse.csr_matrix:
    """CSR of the requested shard rows, in the caller's order."""
    local = np.asarray(local_rows, dtype=np.int64).ravel()
    if local.size == 0:
        return sparse.csr_matrix((0, triple.n_hvg), dtype=triple.dtype)
    if local.min() < 0 or local.max() >= triple.n_obs:
        raise IndexError(f"local_rows out of bounds: min={local.min()} max={local.max()} n_obs={triple.n_obs}")
    order = np.argsort(local, kind="mergesort")
    return _gather_sorted(triple, local[order])[_inverse(order)]


# The cache: every shard as one row space, in AnnCollection order

@dataclass(frozen=True)
class _ShardEntry:
    shard_id: int
    n_obs: int
    n_hvg: int
    global_offset: int
    cache_subdir: Path


class RawHvgShardCache:
    """All shards of one cache as a single ``(n_obs, n_hvg)`` row space.

    Global row ``i`` is row ``i - offset`` of the shard whose ``[offset, offset + n_obs)``
    contains it; shards are ordered by ``shard_id``, the order ``AnnCollection`` uses
    over the source h5ads, so row numbers agree with the h5ad path.

    Shards are mmapped on first touch, not at open: SciPy-free as they are, a cold open
    still reads ``meta.json`` and maps three files per plate. State: ``_triples`` holds
    the open mmaps of this process.
    """

    def __init__(self, cache_root: Union[str, Path], *, eager_obs: bool = True) -> None:
        self.cache_root = Path(cache_root)
        manifest_path = self.cache_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Cache manifest not found: {manifest_path}. Run scripts/data/build_cache.py train first.")
        self.manifest: Dict[str, Any] = json.loads(manifest_path.read_text())
        if "shards" not in self.manifest:
            raise KeyError(f"{manifest_path} has no 'shards'. Rebuild with scripts/data/build_cache.py train.")

        genes_path = self.cache_root / "hvg_gene_names.txt"
        if not genes_path.is_file():
            raise FileNotFoundError(f"HVG gene names file missing: {genes_path}")
        self.var_names: List[str] = [ln.strip() for ln in genes_path.read_text().splitlines() if ln.strip()]
        self.n_vars: int = int(self.manifest["n_hvg"])
        if len(self.var_names) != self.n_vars:
            raise ValueError(f"hvg_gene_names.txt has {len(self.var_names)} entries but manifest reports n_hvg={self.n_vars}.")

        self._shards: List[_ShardEntry] = []
        offset = 0
        for s in sorted(self.manifest["shards"], key=lambda s: int(s["shard_id"])):
            self._shards.append(_ShardEntry(int(s["shard_id"]), int(s["n_obs"]), self.n_vars, offset, self.cache_root / s["cache_subdir"]))
            offset += int(s["n_obs"])
        self.n_obs: int = offset
        self.shard_offsets = np.asarray([e.global_offset for e in self._shards] + [self.n_obs], dtype=np.int64)
        self._triples: Dict[int, MmapCsrTriple] = {}
        self.obs: pd.DataFrame = self._load_obs_concat() if eager_obs else pd.DataFrame()

    def _get_shard_triple(self, entry: _ShardEntry) -> MmapCsrTriple:
        triple = self._triples.get(entry.shard_id)
        if triple is None:
            triple = self._triples[entry.shard_id] = open_triple(entry.cache_subdir, entry.n_obs, entry.n_hvg)
            logger.info("RawHvgShardCache: opened shard %03d | rows=%d nnz=%d", entry.shard_id, entry.n_obs, triple.nnz)
        return triple

    def gene_expression(self, idx: np.ndarray) -> sparse.csr_matrix:
        """CSR of global rows ``idx``, in that order, uint16 counts.

        One global sort serves every shard: sorted global rows are sorted within each
        shard and contiguous across shards, so each shard gathers one slice in file
        order and a single permutation restores the caller's order at the end.
        """
        idx = np.asarray(idx, dtype=np.int64).ravel()
        if idx.size == 0:
            return sparse.csr_matrix((0, self.n_vars), dtype=np.uint16)
        if idx.min() < 0 or idx.max() >= self.n_obs:
            raise IndexError(f"idx out of bounds: min={idx.min()} max={idx.max()} n_obs={self.n_obs}")
        order = np.argsort(idx, kind="mergesort")
        rows = idx[order]
        cuts = np.searchsorted(rows, self.shard_offsets)
        pieces = [
            _gather_sorted(self._get_shard_triple(e), rows[a:b] - e.global_offset)
            for e, a, b in zip(self._shards, cuts[:-1], cuts[1:]) if b > a
        ]
        stacked = pieces[0] if len(pieces) == 1 else sparse.vstack(pieces, format="csr")
        return stacked[_inverse(order)]

    def trim_obs_columns(self, keep: Sequence[str]) -> None:
        """Keep only ``keep`` in ``obs``; missing names are skipped.

        Workers need the conditioning columns and nothing else once setup has drawn
        splits and fitted encoders in the parent. The int64 ``local_obs_index`` and
        ``shard`` columns alone are ~1.6 GiB over 100M rows, and ``obs`` is pickled
        once per spawn worker.
        """
        present = [c for c in keep if c in self.obs.columns]
        if not present:
            raise ValueError(f"trim_obs_columns: none of {list(keep)!r} are in obs (cols={list(self.obs.columns)!r})")
        dropped = sorted(set(self.obs.columns) - set(present))
        self.obs = self.obs[present].copy()  # a view would keep the full frame alive
        logger.info("RawHvgShardCache: trimmed obs to %s (dropped=%s)", present, dropped)

    def drop_lazy_mmap_state(self) -> int:
        """Forget open shards (re-mmapped on next gather); returns how many were open."""
        n = len(self._triples)
        self._triples = {}
        return n

    def __getstate__(self) -> Dict[str, Any]:
        """Pickle without the open shards.

        ``DataLoader(multiprocessing_context="spawn")`` pickles the cache into every
        worker. An ``np.memmap`` pickles as a full copy of the mapped bytes, not as a
        file reference: with four Tahoe shards open that is 23.5 GiB per worker,
        written serially by the parent. Without them the pickle is ``obs`` only
        (~400 MiB), and each worker re-maps the same files, sharing their page cache.
        """
        state = self.__dict__.copy()
        state["_triples"] = {}
        return state

    def __len__(self) -> int:
        return self.n_obs

    @property
    def shards(self) -> List[Dict[str, Any]]:
        return [
            {"shard_id": e.shard_id, "n_obs": e.n_obs, "n_hvg": e.n_hvg,
             "global_obs_offset": e.global_offset, "cache_subdir": str(e.cache_subdir)}
            for e in self._shards
        ]

    def _load_obs_concat(self) -> pd.DataFrame:
        """One obs frame over all shards, row ``i`` = global row ``i``.

        Shards load in parallel (NFS latency, not CPU, is the cost). Each column is a
        per-shard ``Categorical`` from codes, joined by ``union_categoricals``, which
        remaps codes against the merged label set without an object-array detour.
        """
        if not self._shards:
            return pd.DataFrame()
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=len(self._shards)) as pool:
            parts = list(pool.map(lambda e: _read_obs_sidecar(e.cache_subdir), self._shards))
        for e, p in zip(self._shards, parts):
            if len(p["shard"]) != e.n_obs:
                raise ValueError(f"{e.cache_subdir}: obs sidecar has {len(p['shard'])} rows, manifest says {e.n_obs}")
        keys = sorted(k[: -len("__codes")] for k in parts[0] if k.endswith("__codes"))
        if not keys:
            raise RuntimeError(f"{self._shards[0].cache_subdir}: obs sidecar has no <key>__codes arrays. Rebuild the cache.")
        cols: Dict[str, Any] = {c: np.concatenate([np.asarray(p[c], dtype=np.int64) for p in parts]) for c in ("local_obs_index", "shard")}
        for k in keys:
            cols[k] = pd.api.types.union_categoricals([
                pd.Categorical.from_codes(np.asarray(p[f"{k}__codes"]), categories=pd.Index(np.asarray(p[f"{k}__categories"], dtype=object)))
                for p in parts
            ])
        out = pd.DataFrame(cols)
        print(f"  HvgCache: obs for {len(parts)} shards, {len(out):,} rows x {len(out.columns)} cols ({time.monotonic() - t0:.1f}s)", flush=True)
        return out


def _read_obs_sidecar(shard_dir: Path) -> Dict[str, np.ndarray]:
    """One shard's obs arrays, from the ``.npy`` directory or the older single ``.npz``."""
    d = shard_dir / OBS_SIDECAR_DIR
    files = sorted(d.glob("*.npy")) if d.is_dir() else []
    if files:
        return {p.stem: np.load(p, mmap_mode="r" if p.stem.endswith("__codes") else None, allow_pickle=True) for p in files}
    npz = shard_dir / OBS_SIDECAR_NPZ
    if npz.is_file():
        with np.load(npz, allow_pickle=True) as z:
            return {k: np.asarray(z[k]) for k in z.files}
    raise FileNotFoundError(f"Missing obs sidecar under {shard_dir}; expected {OBS_SIDECAR_DIR}/ or {OBS_SIDECAR_NPZ}.")


# The AnnCollection face that scDataset indexes

class _CacheChunkView:
    """What ``collection[idx]`` returns: the chunk's CSR and obs, materialised on ``to_adata``."""

    __slots__ = ("X", "obs", "var_names")

    def __init__(self, X: sparse.csr_matrix, obs: pd.DataFrame, var_names: List[str]) -> None:
        self.X, self.obs, self.var_names = X, obs, var_names

    def to_adata(self) -> ad.AnnData:
        return ad.AnnData(X=self.X, obs=self.obs, var=pd.DataFrame(index=pd.Index(self.var_names, name="gene")))


class HvgCacheCollection:
    """The slice of the ``AnnCollection`` interface that ``PertGenDataModule`` and scDataset use."""

    def __init__(self, cache: RawHvgShardCache) -> None:
        if len(cache.obs) != cache.n_obs:
            raise ValueError(f"obs has {len(cache.obs)} rows but cache.n_obs={cache.n_obs}")
        self.cache = cache

    n_obs = property(lambda self: self.cache.n_obs)
    n_vars = property(lambda self: self.cache.n_vars)
    obs = property(lambda self: self.cache.obs)
    var_names = property(lambda self: self.cache.var_names)
    shards = property(lambda self: self.cache.shards)

    def __len__(self) -> int:
        return self.cache.n_obs

    def __getitem__(self, idx: Any) -> _CacheChunkView:
        rows = np.arange(*idx.indices(self.cache.n_obs), dtype=np.int64) if isinstance(idx, slice) else np.asarray(idx, dtype=np.int64).ravel()
        return _CacheChunkView(self.cache.gene_expression(rows), self.cache.obs.iloc[rows], self.cache.var_names)


def open_cache(cache_root: Union[str, Path], *, eager_obs: bool = True) -> Tuple[RawHvgShardCache, HvgCacheCollection]:
    """``(cache, collection)``: the reader and the face the data module hands scDataset."""
    cache = RawHvgShardCache(cache_root, eager_obs=eager_obs)
    return cache, HvgCacheCollection(cache)


__all__ = [
    "CACHE_VERSION",
    "TRANSFORM_TAG",
    "OBS_SIDECAR_NPZ",
    "OBS_SIDECAR_DIR",
    "MmapCsrTriple",
    "open_triple",
    "gather_rows_from_triple",
    "normalization_id_from_transform",
    "RawHvgShardCache",
    "HvgCacheCollection",
    "open_cache",
]
