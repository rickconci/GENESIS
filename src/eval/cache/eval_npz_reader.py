"""Runtime reader for per-condition eval truth NPZ caches.

``EvalNpzReader(namespace)`` reads one ``geneset_<hash>__n_<N>`` namespace (built by
src/data/cache/eval_cache_build.py); ``.perturbed(keys)`` and ``.control(keys)`` return
a condition's or a control pool's cells as dense float32 raw counts, (n_cells, n_genes).
"""

from __future__ import annotations

import logging
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Mapping, Union

import numpy as np
from scipy import sparse

from ...data.vocab import to_categorical_str
from ...data.data_utils import control_npz_path, perturbed_npz_path, safe_filename_part
from src.data.cache.eval_cache_build import (
    CONTROL_ARRAY_KEY,
    EVAL_NPZ_MODE,
    PERTURBED_ARRAY_KEY,
    library_full_key,
    load_eval_npz_matrix,
    read_eval_npz_manifest,
)

logger = logging.getLogger(__name__)


def subsample_and_densify(
    mat: Union[sparse.csr_matrix, np.ndarray],
    max_cells: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Row-subsample (if requested) then return as dense float32 ndarray.

    For CSR matrices we slice rows on the *sparse* layout BEFORE densifying;
    this keeps the densification work proportional to ``max_cells`` rather
    than the full condition's cell count (the dominant cost when sampling a
    small fraction, e.g. 1024 of ~9000 cells).
    """
    n = mat.shape[0]
    take_all = max_cells is None or max_cells <= 0 or n <= max_cells
    if sparse.issparse(mat):
        sliced = mat if take_all else mat[rng.choice(n, size=int(max_cells), replace=False)]
        return np.asarray(sliced.toarray(), dtype=np.float32)
    if take_all:
        return mat
    return mat[rng.choice(n, size=int(max_cells), replace=False)]


# The same function under its private name, which callers import.
_subsample_and_densify = subsample_and_densify


def _keys_seed(keys: Mapping[str, str]) -> int:
    """Deterministic 31-bit seed from sorted condition keys."""
    parts = tuple(sorted((str(k), str(v)) for k, v in keys.items()))
    return hash(parts) % (2**31)


# Tahoe noise-ceiling caches name NPZs with short keys (cell/plate/dose) while
# the work queue carries parquet column names (cell_name + full plate str).
_PLATE_TAIL_RE = re.compile(r"(?i)^plate(\d+)(?:[_].*)?$")


def work_queue_keys_to_canonical_npz_keys(
    keys: Mapping[str, str],
) -> dict[str, str]:
    """Map work-queue keys to current eval-cache NPZ filename keys.

    Whole-dataset Tahoe builds keep ``cell_name`` and full ``plate`` strings but
    format ``dose`` with :func:`to_categorical_str` (``0.05`` not float32 noise).
    """
    out = {str(k): str(v) for k, v in keys.items()}
    if "dose" in keys:
        dose = keys["dose"]
        if isinstance(dose, (float, np.floating)):
            out["dose"] = to_categorical_str(dose)
        else:
            try:
                out["dose"] = to_categorical_str(float(dose))
            except (TypeError, ValueError):
                out["dose"] = str(dose)
    return out


def work_queue_keys_to_npz_filename_keys(
    keys: Mapping[str, str],
) -> dict[str, str]:
    """Map work-queue conditioning keys to on-disk Tahoe NPZ filename keys.

    Tahoe cache filenames use ``cell`` (not ``cell_name``), normalized plate
    numbers (``4`` not ``plate4_...``), and doses formatted with six decimal
    places (``0.050000`` not ``0.05``).  PBMC/Parse caches
    pass through unchanged when ``cell_name`` is absent.
    """
    out = {str(k): str(v) for k, v in keys.items()}
    if "cell_name" not in out:
        return out
    out["cell"] = out.pop("cell_name")
    if "dose" in out:
        out["dose"] = f"{float(out['dose']):.6f}"
    if "plate" in out:
        plate_raw = str(out["plate"]).strip()
        m = _PLATE_TAIL_RE.match(plate_raw)
        out["plate"] = m.group(1) if m else plate_raw
    return out


_TAHOE_NPZ_KEY_ORDER: tuple[str, ...] = ("cell", "drug", "dose", "plate")


def _tahoe_ordered_npz_path(
    namespace: Path,
    prefix: str,
    keys: Mapping[str, str],
) -> Path:
    """Tahoe cache filenames in ``cell, drug, dose, plate`` order (not sorted)."""
    parts = [prefix]
    for key in _TAHOE_NPZ_KEY_ORDER:
        if key in keys:
            parts.append(f"{key}_{safe_filename_part(keys[key])}")
    return Path(namespace) / ("__".join(parts) + ".npz")


def _resolve_existing_npz_path(
    namespace: Path,
    *,
    prefix: str,
    keys: Mapping[str, str],
) -> Path:
    """Return the first existing NPZ path for ``keys`` (direct, canonical, Tahoe short-key)."""
    path_fn = perturbed_npz_path if prefix == "perturbed_cache" else control_npz_path
    candidates: list[Path] = [path_fn(namespace, keys)]
    canonical = work_queue_keys_to_canonical_npz_keys(keys)
    if canonical != dict(keys):
        candidates.append(path_fn(namespace, canonical))
    aliased = work_queue_keys_to_npz_filename_keys(keys)
    if aliased != dict(keys):
        candidates.append(path_fn(namespace, aliased))
        candidates.append(_tahoe_ordered_npz_path(namespace, prefix, aliased))
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path
    return candidates[0]


class EvalNpzReader:
    """Load dense float32 truth from sparse or dense per-condition NPZ files."""

    def __init__(
        self,
        namespace: Path,
        *,
        cache_size: int = 32,
    ) -> None:
        from ..orchestration.log_utils import log_step

        ns = Path(namespace)
        manifest = read_eval_npz_manifest(ns)
        if manifest.get("mode") != EVAL_NPZ_MODE:
            raise ValueError(
                f"Eval namespace {ns} has mode={manifest.get('mode')!r}; "
                f"expected {EVAL_NPZ_MODE!r}. Rebuild with scripts/data/build_cache.py eval.",
            )
        self._ns = ns
        # Sidecar-only enrich namespaces point matrix NPZs at the live SRC.
        matrix_ns = manifest.get("matrix_namespace")
        self._matrix_ns = Path(matrix_ns) if matrix_ns else ns
        if matrix_ns and not self._matrix_ns.is_dir():
            raise FileNotFoundError(
                f"manifest matrix_namespace missing: {self._matrix_ns} (from {ns})",
            )
        self._cache_size = max(1, int(cache_size))
        # Cache the on-disk layout (CSR for sparse files, a dense ndarray for
        # dense ones). Keeping CSR avoids the 10-100x densification cost when
        # callers only need a row-subsample.
        self._cache: OrderedDict[str, Union[sparse.csr_matrix, np.ndarray]] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._packed_lib: dict[str, object] | None = None
        self._packed_lock = threading.Lock()
        log_step(
            logger, "index:read",
            "eval NPZ reader ready in %.1fs | %s | matrix_ns=%s | conditions=%s control_pools=%s storage=%s",
            0.0,
            ns,
            self._matrix_ns,
            manifest.get("n_conditions"),
            manifest.get("n_control_pools"),
            manifest.get("storage", "dense_v0"),
        )

    def __getstate__(self) -> dict:
        """Drop locks and lazily-rebuildable cache state so ``spawn`` can pickle us.

        WHY: the cell_set train loader puts this reader inside a Dataset, and
        ``DataLoader(multiprocessing_context="spawn")`` pickles that Dataset to
        every worker. Two things break:

        * ``threading.Lock`` is unpicklable outright — ``TypeError: cannot pickle
          '_thread.lock' object``, which is a hard crash at worker start;
        * ``_packed_lib`` holds ``np.load(..., mmap_mode="r")`` arrays, and
          ``np.memmap`` pickles by VALUE (it inherits ``ndarray.__reduce__``), so
          a warm reader would copy the mapped bytes down the pipe per worker —
          the same trap documented on ``RawHvgShardCache.__getstate__``.

        Both are rebuilt on demand: ``_load_npz_matrix`` refills ``_cache`` and
        ``_load_packed_library_full`` re-mmaps the sidecar, giving workers shared
        file-backed pages instead of per-worker anonymous copies.

        Must NOT mutate ``self``: pickling is not allowed to empty the parent's
        live LRU or drop the lock it is still using.
        """
        state = self.__dict__.copy()
        state.pop("_cache_lock", None)
        state.pop("_packed_lock", None)
        state["_cache"] = OrderedDict()
        state["_packed_lib"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore, then give this process its OWN locks and empty caches."""
        self.__dict__.update(state)
        # Locks are per-process by construction; a pickle never carries them.
        self._cache_lock = threading.Lock()
        self._packed_lock = threading.Lock()
        # Defensive: a pickle could arrive with populated caches whose
        # mmaps were materialised. Drop them so this process re-opens real mmaps.
        self._cache = OrderedDict()
        self._packed_lib = None

    @property
    def namespace(self) -> Path:
        return self._ns

    def _load_npz_matrix(
        self,
        path: Path,
        array_key: str,
    ) -> Union[sparse.csr_matrix, np.ndarray]:
        # Returns the cached on-disk layout (CSR or dense). Densification is
        # deferred to the caller via ``_subsample_and_densify`` so we never
        # materialize cells we will immediately discard.
        cache_key = str(path)
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
                return cached

        # Load outside the lock so ThreadPool fill workers can overlap NFS IO.
        mat = load_eval_npz_matrix(path, array_key)
        with self._cache_lock:
            # Another worker may have won the race; keep the first.
            existing = self._cache.get(cache_key)
            if existing is not None:
                self._cache.move_to_end(cache_key)
                return existing
            self._cache[cache_key] = mat
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
            return mat

    def _load_packed_library_full(self, stem: str) -> np.ndarray | None:
        """Look up ``stem`` in packed ``.library_full/{stems,indptr,flat}``."""
        with self._packed_lock:
            if self._packed_lib is None:
                side = self._ns / ".library_full"
                stems_path = side / "stems.txt"
                if not stems_path.is_file():
                    self._packed_lib = {}
                    return None
                stems = [ln.strip() for ln in stems_path.read_text().splitlines() if ln.strip()]
                indptr = np.load(side / "indptr.npy", mmap_mode="r")
                flat = np.load(side / "flat.npy", mmap_mode="r")
                if int(indptr.shape[0]) != len(stems) + 1:
                    raise ValueError(
                        f"packed library_full indptr len {indptr.shape[0]} "
                        f"!= n_stems+1 ({len(stems)+1}) under {side}",
                    )
                index = {s: i for i, s in enumerate(stems)}
                self._packed_lib = {"index": index, "indptr": indptr, "flat": flat}
            store = self._packed_lib
        if not store:
            return None
        idx = store["index"].get(stem)  # type: ignore[index]
        if idx is None:
            return None
        indptr = store["indptr"]  # type: ignore[index]
        flat = store["flat"]  # type: ignore[index]
        a, b = int(indptr[idx]), int(indptr[idx + 1])
        return np.asarray(flat[a:b], dtype=np.float64)

    def _load_library_full(self, path: Path, array_key: str) -> np.ndarray | None:
        """Per-cell FULL-transcriptome library, or ``None`` when the cache has none.

        The cached matrix holds only the selected genes (~9% of a tahoe cell's
        counts), so this is the only record of ``normalize_total``'s divisor. Row
        order matches the cached matrix. Returns ``None`` rather than raising, so
        callers can detect a cache built without it and rebuild instead of silently
        substituting the selected-gene library, which is a different transform.

        Lookup order (packed first: with_libfull namespaces always miss the
        in-NPZ key, and opening the NPZ just to discover that dominated train IO):
          1. packed store ``stems.txt`` + ``indptr.npy`` + ``flat.npy``
          2. per-file sidecar ``<ns>/.library_full/<stem>.npy``
          3. in-NPZ ``*_library_full`` key
        """
        packed = self._load_packed_library_full(path.stem)
        if packed is not None:
            return packed
        for side in (
            self._ns / ".library_full" / f"{path.stem}.npy",
            path.parent / ".library_full" / f"{path.stem}.npy",
        ):
            if side.is_file():
                return np.asarray(np.load(side), dtype=np.float64)
        key = library_full_key(array_key)
        with np.load(path, allow_pickle=False) as npz:
            if key in npz.files:
                return np.asarray(npz[key], dtype=np.float64)
        return None
    def perturbed_library_full(
        self, condition_keys: Mapping[str, str],
    ) -> np.ndarray | None:
        """Full-gene library per perturbed cell (see :meth:`_load_library_full`)."""
        path = _resolve_existing_npz_path(
            self._matrix_ns, prefix="perturbed_cache", keys=condition_keys,
        )
        return self._load_library_full(path, PERTURBED_ARRAY_KEY)

    def control_library_full(
        self, control_id_keys: Mapping[str, str],
    ) -> np.ndarray | None:
        """Full-gene library per control cell (see :meth:`_load_library_full`)."""
        path = _resolve_existing_npz_path(
            self._matrix_ns, prefix="control_cache", keys=control_id_keys,
        )
        return self._load_library_full(path, CONTROL_ARRAY_KEY)

    def perturbed(
        self,
        condition_keys: Mapping[str, str],
        *,
        max_cells: int | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """Return dense ``(n_cells, G)`` perturbed truth for one condition."""
        path = _resolve_existing_npz_path(
            self._matrix_ns, prefix="perturbed_cache", keys=condition_keys,
        )
        mat = self._load_npz_matrix(path, PERTURBED_ARRAY_KEY)
        rng_seed = seed if seed is not None else _keys_seed(condition_keys)
        return subsample_and_densify(mat, max_cells, np.random.default_rng(rng_seed))

    def control(
        self,
        control_id_keys: Mapping[str, str],
        *,
        max_cells: int | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """Return dense ``(n_cells, G)`` control truth for one pool."""
        path = _resolve_existing_npz_path(
            self._matrix_ns, prefix="control_cache", keys=control_id_keys,
        )
        mat = self._load_npz_matrix(path, CONTROL_ARRAY_KEY)
        rng_seed = seed if seed is not None else _keys_seed(control_id_keys)
        return subsample_and_densify(mat, max_cells, np.random.default_rng(rng_seed))

    def perturbed_matrix(
        self, condition_keys: Mapping[str, str],
    ) -> Union[sparse.csr_matrix, np.ndarray]:
        """Cached on-disk layout (CSR or dense) — no densify."""
        path = _resolve_existing_npz_path(
            self._matrix_ns, prefix="perturbed_cache", keys=condition_keys,
        )
        return self._load_npz_matrix(path, PERTURBED_ARRAY_KEY)

    def control_matrix(
        self, control_id_keys: Mapping[str, str],
    ) -> Union[sparse.csr_matrix, np.ndarray]:
        """Cached on-disk layout (CSR or dense) — no densify."""
        path = _resolve_existing_npz_path(
            self._matrix_ns, prefix="control_cache", keys=control_id_keys,
        )
        return self._load_npz_matrix(path, CONTROL_ARRAY_KEY)

    def perturbed_indices(self, condition_keys: Mapping[str, str]) -> np.ndarray:
        """NPZ mode has no row-index layer; raise to catch misuse."""
        raise NotImplementedError(
            "EvalNpzReader stores per-condition matrices only; "
            "perturbed_indices is unavailable.",
        )

    def control_indices(self, control_id_keys: Mapping[str, str]) -> np.ndarray:
        raise NotImplementedError(
            "EvalNpzReader stores per-condition matrices only; "
            "control_indices is unavailable.",
        )
