"""External-model arm: one prediction NPZ per work-queue row (STATE, CellFlow, PerturbDiff, ...).

NPZ contract::

    predictions   (n_cells, n_genes) float32, log1p space
    gene_names    (n_genes,)
    <index fields> scalars matching ``index_variables`` (plate may be stored as plate_norm)

The index maps each NPZ to a condition key; the getter returns that file's cells
in the canonical gene order. Predictions are scored as ModelOutputSpace.LOG1P.
"""
from __future__ import annotations

import logging
import re
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from src.eval.cache.eval_cache_io import QuadrupletRow

log = logging.getLogger(__name__)

_PLATE_LOOSE_RE = re.compile(r"(?i)^plate(\d+)(?:[_].*)?$")


@dataclass(frozen=True)
class _PredNpzEntry:
    path: Path
    key: tuple[Any, ...]


def _normalize_plate(raw: str) -> Optional[str]:
    m = _PLATE_LOOSE_RE.match(str(raw).strip())
    return m.group(1) if m else None


def _read_npz_index_fields(path: Path, index_variables: Sequence[str]) -> Optional[tuple[Any, ...]]:
    """Read index key fields from one prediction NPZ header."""
    try:
        with np.load(path, allow_pickle=True) as z:
            files = set(z.files)
            parts: list[Any] = []
            for var in index_variables:
                if var == "plate_norm":
                    if "plate_norm" in files:
                        val = z["plate_norm"]
                        parts.append(str(val).strip() if val.ndim == 0 else str(val[0]).strip())
                    elif "plate" in files:
                        norm = _normalize_plate(str(z["plate"]))
                        if norm is None:
                            log.warning("Cannot normalise plate in %s; skipping.", path.name)
                            return None
                        parts.append(norm)
                    else:
                        log.warning("No plate/plate_norm in %s; skipping.", path.name)
                        return None
                    continue
                if var not in files:
                    log.warning("Missing field %r in %s; skipping.", var, path.name)
                    return None
                val = z[var]
                if var == "dose":
                    parts.append(round(float(val), 6))
                else:
                    parts.append(str(val).strip() if val.ndim == 0 else str(val[0]).strip())
            return tuple(parts)
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to read %s: %s", path.name, exc)
        return None


def build_prediction_index(
    root: Path,
    index_variables: Sequence[str],
    *,
    max_workers: int = 16,
) -> dict[tuple[Any, ...], _PredNpzEntry]:
    files = sorted(root.glob("*.npz"))
    if not files:
        raise SystemExit(f"No .npz files found under {root}")

    index: dict[tuple[Any, ...], _PredNpzEntry] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
        for path, key in zip(files, pool.map(
            lambda p: _read_npz_index_fields(p, index_variables), files,
        )):
            if key is None:
                continue
            if key in index:
                log.warning(
                    "Duplicate prediction NPZ for %s (keeping first): %s vs %s",
                    key, index[key].path.name, path.name,
                )
                continue
            index[key] = _PredNpzEntry(path=path, key=key)
    log.info(
        "Indexed prediction NPZs | root=%s files=%d indexed=%d vars=%s",
        root, len(files), len(index), list(index_variables),
    )
    return index


def qrow_lookup_key(
    qrow: QuadrupletRow, index_variables: Sequence[str],
) -> Optional[tuple[Any, ...]]:
    """Map a work-queue row to the prediction-index key."""
    parts: list[Any] = []
    for var in index_variables:
        if var == "plate_norm":
            plate_raw = str(qrow.condition_keys.get("plate", ""))
            norm = _normalize_plate(plate_raw)
            if norm is None:
                return None
            parts.append(norm)
        elif var == "dose":
            parts.append(round(float(qrow.condition_keys[var]), 6))
        else:
            if var not in qrow.condition_keys:
                return None
            # .strip() to match _read_npz_index_fields, which strips the NPZ side.
            # Without it, any condition variable carrying leading/trailing
            # whitespace can never be looked up: the tahoe summary has the drugs
            # 'Erdafitinib ' and 'Selinexor ', which index as 'Erdafitinib' and
            # would be silently dropped from the mean (7 of 192 tahoe-bio
            # validation conditions, 3.6%).
            parts.append(str(qrow.condition_keys[var]).strip())
    return tuple(parts)


class NpzPredGetter:
    """``pred_getter`` for :class:`ConditionPrefetcher`."""

    def __init__(
        self,
        index: dict[tuple[Any, ...], _PredNpzEntry],
        index_variables: Sequence[str],
        *,
        gene_names: Sequence[str],
        gene_reindex: Optional[np.ndarray],
        match_rows: bool = False,
        min_cells: int = 4,
    ) -> None:
        """``match_rows``: resample each cloud to the truth's cell count. The K=1 scorer
        indexes predictions with a truth-length split-half index, so a cloud of another
        size (e.g. one cell type's share of a classified mixed cloud) must be matched;
        a cloud under ``min_cells`` is a miss, not a resampled handful."""
        self._match_rows, self._min_cells = bool(match_rows), int(min_cells)
        self._index = index
        self._index_variables = tuple(index_variables)
        self._n_genes = len(gene_names)
        self._gene_reindex = gene_reindex
        self._miss_count = 0

    def __call__(self, qrow: QuadrupletRow) -> Optional[np.ndarray]:
        key = qrow_lookup_key(qrow, self._index_variables)
        if key is None:
            self._miss_count += 1
            return None
        entry = self._index.get(key)
        if entry is None:
            self._miss_count += 1
            return None
        with np.load(entry.path, allow_pickle=True) as z:
            arr = np.asarray(z["predictions"], dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self._n_genes:
            log.warning(
                "Shape mismatch in %s: got %s expected (*, %d)",
                entry.path.name, arr.shape, self._n_genes,
            )
            return None
        if self._gene_reindex is not None:
            arr = np.ascontiguousarray(arr[:, self._gene_reindex])
        if self._match_rows:
            n, have = int(qrow.num_cells), int(arr.shape[0])
            if have < self._min_cells:
                log.warning("%s: %d predicted cells < min_cells=%d; skipping.", entry.path.name, have, self._min_cells)
                self._miss_count += 1
                return None
            if have != n:   # seed = adler32 of the index key: the published classified-arm rule
                seed = zlib.adler32("|".join(str(x) for x in key).encode()) & 0x7FFFFFFF
                arr = np.ascontiguousarray(arr[np.random.default_rng(seed).choice(have, size=n, replace=have < n)])
        return arr

    @property
    def miss_count(self) -> int:
        return self._miss_count


def resolve_gene_reindex(
    npz_gene_names: Sequence[str], canonical: Sequence[str],
) -> Optional[np.ndarray]:
    if list(npz_gene_names) == list(canonical):
        return None
    npz_set = set(map(str, npz_gene_names))
    canon_set = set(map(str, canonical))
    if npz_set != canon_set:
        missing_in_npz = canon_set - npz_set
        extra_in_npz = npz_set - canon_set
        raise SystemExit(
            "Gene-set mismatch between prediction NPZ and canonical list:\n"
            f"  missing in NPZ (n={len(missing_in_npz)}): {sorted(missing_in_npz)[:10]} ...\n"
            f"  extra  in NPZ (n={len(extra_in_npz)}): {sorted(extra_in_npz)[:10]} ...",
        )
    order_in_npz = {g: i for i, g in enumerate(map(str, npz_gene_names))}
    return np.asarray([order_in_npz[str(g)] for g in canonical], dtype=np.int64)


def default_index_variables(condition_variables: Sequence[str]) -> tuple[str, ...]:
    """Competitor NPZ headers store the normalised plate id."""
    return tuple("plate_norm" if v == "plate" else v for v in condition_variables)


def npz_gene_names(index: dict) -> list[str]:
    with np.load(next(iter(index.values())).path, allow_pickle=True) as z:
        return [str(g) for g in z["gene_names"]]
