"""Per-condition sufficient statistics of log(1 + count), from one sequential pass over the HVG shards.

Why: an arm's response mean should use every cell of the arm, and reading one condition at a
time costs one NFS open per condition (~3 h cold for Tahoe's 65k). The mmap shards hold the
same cells as the eval NPZs (bit-identical, same gene order), so one sequential read of each
shard sums every condition at once.

Stored per condition and per half (a seeded coin per cell): the cell count ``n`` and the mean of
log1p(count) per gene. The halves make the noise of anything built from the means measurable
(split-half agreement); the whole-data mean is their cell-weighted mix (``pooled_mean``). Every
condition and control in the cache is in the table, so it does not depend on a split: a split
only selects rows.

On disk, next to the shards:  <cache_root>/condition_stats/stats__<key12>.npz  (+ .key.json)
"""
from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.data.vocab import concentrations_from_obs, to_categorical_str
from src.data.cache.raw_hvg_cache import _read_obs_sidecar

log = logging.getLogger(__name__)

BUILDER_VERSION = 1      # bump when a change here alters the numbers on disk
CACHE_DIRNAME = "condition_stats"
CHUNK_ROWS = 131_072     # ~16M nnz per bincount: the nnz work, not the per-chunk target, dominates


# The log1p sums: every condition of one CSR shard at once

def log1p_sums(data: np.ndarray, indices: np.ndarray, indptr: np.ndarray, labels: np.ndarray, *,
               n_labels: int, n_genes: int, chunk_rows: int = CHUNK_ROWS) -> np.ndarray:
    """``(n_labels, n_genes)`` float64 sums of log1p(count) over the rows carrying each label.

    Counts are uint16, so a 65,536-entry table replaces log1p on ~10^10 values with a gather.
    Each chunk scatters onto its own labels only, so the bincount target stays small however
    many labels the shard has.
    """
    n_rows = len(indptr) - 1
    labels = np.asarray(labels)
    if labels.shape != (n_rows,):
        raise ValueError(f"{labels.shape[0]} labels for {n_rows} rows")
    if n_rows and (labels.min() < 0 or labels.max() >= n_labels):
        raise ValueError(f"a label outside [0, {n_labels}): every row needs a condition")
    if data.dtype != np.uint16:
        raise TypeError(f"counts are {data.dtype}; the log1p table needs uint16")
    lut = np.log1p(np.arange(2**16, dtype=np.float64))
    sums = np.zeros((n_labels, n_genes), dtype=np.float64)
    for a in range(0, n_rows, chunk_rows):
        b = min(a + chunk_rows, n_rows)
        lo, hi = int(indptr[a]), int(indptr[b])
        genes = np.asarray(indices[lo:hi], dtype=np.int64)
        if genes.size and genes.max() >= n_genes:     # would land silently in the next label's block
            raise ValueError(f"a gene index >= n_genes ({n_genes}) in rows {a}:{b}")
        local, inverse = np.unique(labels[a:b], return_inverse=True)
        row_label = np.repeat(inverse.astype(np.int64), np.diff(np.asarray(indptr[a:b + 1])))
        part = np.bincount(row_label * n_genes + genes, weights=lut[np.asarray(data[lo:hi])],
                           minlength=local.size * n_genes)
        sums[local] += part.reshape(local.size, n_genes)
    return sums


# The labels: which condition each row of a shard belongs to

def _column(sidecar: Mapping[str, np.ndarray], col: str) -> tuple[np.ndarray, np.ndarray]:
    """(per-row codes, canonical string per category) of one key column. Tahoe stores no
    ``dose``: it is parsed once per ``drugname_drugconc`` category, as the eval index does."""
    src = "drugname_drugconc" if col == "dose" and "dose__codes" not in sidecar else col
    if f"{src}__codes" not in sidecar:
        raise KeyError(f"no {col!r} in the obs sidecar (has {sorted(k[:-7] for k in sidecar if k.endswith('__codes'))})")
    codes = np.asarray(sidecar[f"{src}__codes"], dtype=np.int64)
    cats = np.asarray(sidecar[f"{src}__categories"], dtype=object)
    if codes.size and codes.min() < 0:
        raise ValueError(f"{int((codes < 0).sum())} rows have no {col!r}")
    if src != col:
        cats = concentrations_from_obs(pd.DataFrame({src: cats.astype(str)}), key=src).astype(float)
    return codes, np.array([to_categorical_str(c) for c in cats], dtype=object)


def shard_condition_labels(sidecar: Mapping[str, np.ndarray], columns: Sequence[str]) -> tuple[list[tuple], np.ndarray]:
    """Sorted condition keys of one shard, and each row's index into them.

    Rows group on category codes (cheap); the code tuples then map to canonical strings and
    merge, so two spellings of one value (``0.05`` and its float32 twin) are one condition.
    """
    cols = [_column(sidecar, c) for c in columns]
    dims = tuple(len(names) for _, names in cols)
    uniq, row_code = np.unique(np.ravel_multi_index(tuple(codes for codes, _ in cols), dims), return_inverse=True)
    code_keys = [tuple(str(names[i]) for (_, names), i in zip(cols, idx))
                 for idx in zip(*np.unravel_index(uniq, dims))]
    keys = sorted(set(code_keys))
    pos = {k: i for i, k in enumerate(keys)}
    return keys, np.array([pos[k] for k in code_keys], dtype=np.int64)[row_code]


def _shard_stats(entry: dict, *, root: Path, columns: Sequence[str], seed: int, n_genes: int):
    """One shard: its condition keys, cells per (key, half), and log1p sums per (key, half)."""
    d = root / entry["cache_subdir"]
    keys, cond = shard_condition_labels(_read_obs_sidecar(d), columns)   # PBMC: the .npy dir (its .npz is corrupt)
    indptr = np.load(d / "indptr.npy", mmap_mode="r")
    if not len(indptr) - 1 == len(cond) == int(entry["n_obs"]):
        raise ValueError(f"{d}: {len(indptr) - 1} CSR rows, {len(cond)} sidecar rows, manifest {entry['n_obs']}")
    # The coin is a function of (seed, shard) alone: the halves never depend on chunking or workers.
    labels = 2 * cond + np.random.default_rng([seed, int(entry["shard_id"])]).integers(0, 2, size=len(cond))
    t0 = time.time()
    sums = log1p_sums(np.load(d / "data.npy", mmap_mode="r"), np.load(d / "indices.npy", mmap_mode="r"), indptr,
                      labels, n_labels=2 * len(keys), n_genes=n_genes)
    log.info("%s: %d cells, %d conditions [%.0fs]", d.name, len(cond), len(keys), time.time() - t0)
    return keys, np.bincount(labels, minlength=2 * len(keys)).reshape(-1, 2), sums.reshape(len(keys), 2, n_genes)


# Build and load the table

def _stats_path(root: Path, columns: Sequence[str], seed: int) -> tuple[Path, dict]:
    """Everything that changes the numbers is in the key; a table for other columns is another file."""
    key = {"builder_version": BUILDER_VERSION, "columns": list(columns), "seed": int(seed),
           "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
           "genes_sha256": hashlib.sha256((root / "hvg_gene_names.txt").read_bytes()).hexdigest()}
    h = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    return root / CACHE_DIRNAME / f"stats__{h}.npz", key


def build_condition_stats(cache_root: Path, columns: Sequence[str], *, seed: int = 0, workers: int = 4) -> Path:
    """Sum every shard (``workers`` processes: NFS throughput grows with concurrent readers)."""
    root = Path(cache_root)
    manifest = json.loads((root / "manifest.json").read_text())
    if "raw_counts" not in str(manifest.get("transform", "")):
        raise ValueError(f"{root}: transform {manifest.get('transform')!r} is not raw counts")
    genes = (root / "hvg_gene_names.txt").read_text().split()
    if len(genes) != int(manifest["n_hvg"]):
        raise ValueError(f"{root}: {len(genes)} gene names for n_hvg {manifest['n_hvg']}")
    path, key = _stats_path(root, columns, seed)
    one = partial(_shard_stats, root=root, columns=tuple(columns), seed=seed, n_genes=len(genes))
    t0 = time.time()
    if workers > 1:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
            parts = list(ex.map(one, manifest["shards"]))
    else:
        parts = [one(e) for e in manifest["shards"]]
    keys = sorted({k for ks, _, _ in parts for k in ks})
    pos = {k: i for i, k in enumerate(keys)}
    n, sums = np.zeros((len(keys), 2), np.int64), np.zeros((len(keys), 2, len(genes)))
    for ks, n_s, s_s in parts:                              # a condition on several shards (plates) merges here
        rows = np.array([pos[k] for k in ks], dtype=np.int64)
        n[rows] += n_s
        sums[rows] += s_s
    if int(n.sum()) != int(manifest["total_obs"]):
        raise ValueError(f"summed {int(n.sum())} cells, manifest total_obs {manifest['total_obs']}")
    # A half with no cells keeps mean 0 and n 0; pooled_mean weights by n and refuses an empty pool.
    mean = np.divide(sums, n[:, :, None], out=np.zeros_like(sums), where=n[:, :, None] > 0).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, keys=np.array(keys, dtype=str).reshape(len(keys), len(columns)), columns=np.array(columns),
             genes=np.array(genes), n=n, mean=mean)
    tmp.replace(path)
    stats = {"n_conditions": len(keys), "n_cells": int(n.sum()), "build_seconds": round(time.time() - t0, 1)}
    path.with_suffix(".key.json").write_text(json.dumps({"key": key, "stats": stats}, indent=2) + "\n")
    log.info("condition stats %s -> %s", stats, path)
    return path


@dataclass(frozen=True)
class ConditionStats:
    """``n[c, h]`` cells and ``mean[c, h]`` log1p means of condition ``keys[c]``, half ``h``."""
    columns: tuple
    keys: list
    genes: list
    n: np.ndarray
    mean: np.ndarray
    index: dict = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", {k: i for i, k in enumerate(self.keys)})


def load_condition_stats(cache_root: Path, columns: Sequence[str], *, seed: int = 0,
                         genes: Optional[Sequence[str]] = None) -> ConditionStats:
    """``genes``: the order the caller's arrays use; a table in another order is refused."""
    path, _ = _stats_path(Path(cache_root), columns, seed)
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing; build it with build_condition_stats (scripts/data/build_cache.py "
                                f"stats --cache-dir {cache_root} --columns {' '.join(columns)} --seed {seed})")
    z = np.load(path)
    st = ConditionStats(tuple(z["columns"].tolist()), [tuple(k) for k in z["keys"].tolist()],
                        z["genes"].tolist(), z["n"], z["mean"])
    if genes is not None and list(genes) != st.genes:
        raise ValueError(f"{path}: gene order differs from the caller's {len(genes)} genes")
    log.info("condition stats %s from %s", st.mean.shape, path)
    return st


def pooled_mean(stats: ConditionStats, rows: Sequence[int], half: Optional[int] = None) -> tuple[np.ndarray, int]:
    """Cell-weighted log1p mean over conditions ``rows`` (both halves, or one), and its cell count."""
    rows = np.asarray(rows, dtype=np.int64)
    halves = slice(None) if half is None else slice(half, half + 1)
    n = stats.n[rows, halves].astype(np.float64)
    total = int(n.sum())
    if total == 0:
        raise ValueError(f"no cells in conditions {rows.tolist()} (half {half})")
    return np.einsum("rh,rhg->g", n, stats.mean[rows, halves].astype(np.float64)) / total, total
