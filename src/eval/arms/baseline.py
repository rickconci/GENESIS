"""Baseline arm: a fitted perturbation_mean / context_mean / linear_ridge / additive artifact.

A baseline predicts one expression vector per condition, and real control cells carry
it: each emitted cell is a control cell of this condition shifted by the baseline's
delta, so it keeps real dropout and library size. The artifact's fit space
(src/baselines/space.py) picks the shift:
  raw               ``emit_per_cell_delta``: the delta in log1p_cpm; rows are raw counts (RAW).
  log1p_cp10k_full  ``emit_in_fit_space``: the delta in log1p(1e4 x / L_full), denormalised with
                    each cell's own L_full as the STATE arm is; rows are log1p(counts) (LOG1P).
Each getter's ``output_space`` says which, and the scorer must read its rows as that.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import zlib
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from src.baselines.artifacts import Baseline
from src.baselines.roles import BaselineConditionRoles, normalize_plate_for_eval
from src.baselines.space import RAW, FitSpaceReader, from_fit_space, library_namespace
from src.eval.cache.eval_cache_io import QuadrupletRow
from src.eval.cache.eval_npz_reader import EvalNpzReader
from src.model.inference_output import ModelOutputSpace

log = logging.getLogger(__name__)

EMISSION_TILE_MEAN = "tile_mean_to_num_cells"
EMISSION_PER_CELL = "per_cell_ctrl_plus_delta_log1p_cpm"
_CPM_SCALE = 1e4  # matches inference_output.to_log1p_cpm / spaces._to_log1p_cpm_torch default


def _check_axes(fit: Baseline, roles: BaselineConditionRoles) -> None:
    """Scoring with other roles than the fit's would look up keys the fit never made."""
    if fit.axes != roles.axes:
        raise ValueError(f"{fit.kind} was fit with roles {fit.axes}, scoring uses {roles.axes}; "
                         "pass the fit's --condition-variables / --control-match-keys")


def _stable_seed(obj: object, salt: int) -> int:
    """Process-independent seed (``hash()`` is PYTHONHASHSEED-salted)."""
    h = hashlib.blake2b(repr(obj).encode(), digest_size=8).digest()
    return (int.from_bytes(h, "big") ^ int(salt)) % (2**32)


def _log1p_cpm_rows(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(log1p(raw/lib*1e4), lib)`` for a 2D raw-count array, lib floored at 1."""
    lib = np.maximum(raw.sum(axis=1, keepdims=True), 1.0)
    return np.log1p(raw / lib * _CPM_SCALE), lib


def emit_per_cell_delta(
    *,
    pred_vec_raw: np.ndarray,
    ctrl_cells_raw: np.ndarray,
    ctrl_mean_raw: np.ndarray,
    n_cells: int,
    seed: int,
) -> np.ndarray:
    """Real control cells shifted by the baseline delta in ``log1p_cpm``.

    A baseline predicts one vector. Tiling it to ``n_truth`` identical rows makes the
    scorer's predicted pseudobulk a *log of a mean* while truth/control are *means of
    logs*; the Jensen gap is expression-dependent, so it does not cancel in the delta
    and swamps it (measured: a zero-error oracle scores ``pearson_r_diff`` ~0 with
    ``delta_l2_sq_norm_ratio`` 234-350). Identical rows also give zero within-condition
    variance, which makes energy/MMD meaningless.

    Instead: take the delta this baseline actually claims, expressed in the reporting
    space, and add it to real control cells so each emitted cell keeps real dropout and
    library size::

        delta_log = log1p_cpm(pred_vec) - log1p_cpm(ctrl_mean)
        row_i     = expm1(clip(log1p_cpm(ctrl_cell_i) + delta_log, 0)) * lib_i / 1e4

    Returned rows are RAW counts, so the caller's ``ModelOutputSpace.RAW`` contract is
    unchanged: the scorer re-derives ``log1p_cpm`` from them itself (up to one
    per-cell renormalisation, which the measured oracle score already includes).

    No leakage: these are the same control cells every model is conditioned on, and the
    perturbed truth cells are untouched.
    """
    # Clamp at 0 exactly as the scorer does (``to_raw(x, RAW)`` is ``_clamp_nonneg``).
    # mu_ctrl + delta_p goes negative for strongly down-regulated genes, and
    # log1p(negative) is NaN -- which silently drops pearson/r2/energy downstream.
    pred_vec_raw = np.clip(
        np.asarray(pred_vec_raw, dtype=np.float32).reshape(1, -1), 0.0, None,
    )
    ctrl_mean_raw = np.clip(
        np.asarray(ctrl_mean_raw, dtype=np.float32).reshape(1, -1), 0.0, None,
    )
    tgt_log, _ = _log1p_cpm_rows(pred_vec_raw)
    ref_log, _ = _log1p_cpm_rows(ctrl_mean_raw)
    delta_log = tgt_log - ref_log  # (1, G) in the reporting space

    n_pool = int(ctrl_cells_raw.shape[0])
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_pool, size=int(n_cells), replace=n_pool < int(n_cells))
    cells = np.clip(ctrl_cells_raw[idx].astype(np.float32, copy=False), 0.0, None)

    cell_log, lib = _log1p_cpm_rows(cells)
    new_log = np.clip(cell_log + delta_log, 0.0, None)
    return (np.expm1(new_log) * lib / _CPM_SCALE).astype(np.float32, copy=False)


def emit_in_fit_space(*, delta: np.ndarray, ctrl_cells: np.ndarray, ctrl_lib: np.ndarray, n_cells: int,
                      seed: int) -> np.ndarray:
    """A log1p_cp10k_full fit, emitted as the STATE arm is: ``n_cells`` matched control cells (drawn
    with replacement only when the pool is smaller), each moved by the fitted delta in
    log1p(1e4 x / L_full) and denormalised with its own L_full. Returns log1p(counts), so the scorer
    snaps it at the geometric cuts every LOG1P arm gets.
    """
    n_pool = int(ctrl_cells.shape[0])
    idx = np.random.default_rng(seed).choice(n_pool, size=int(n_cells), replace=n_pool < int(n_cells))
    moved = ctrl_cells[idx] + np.asarray(delta, dtype=np.float32).reshape(1, -1)
    return np.log1p(from_fit_space(moved, ctrl_lib[idx])).astype(np.float32)


def _space_reader(fit: Baseline, npz_reader: EvalNpzReader) -> Optional[FitSpaceReader]:
    """The fit space's control reader (cells with their L_full), or None for a raw fit."""
    space = fit.params.get("fit_space", RAW)
    return None if space == RAW else FitSpaceReader(EvalNpzReader(library_namespace(npz_reader.namespace)))


class BaselinePredGetter:
    """``pred_getter`` wrapping a fitted baseline with eval-time control lookup."""

    def __init__(
        self,
        *,
        fit: Baseline,
        n_genes: int,
        npz_reader: EvalNpzReader,
        roles: BaselineConditionRoles,
        emission: str = EMISSION_TILE_MEAN,
        ctrl_pool_cells: int = 2048,
        pool_cache_size: int = 8,
        seed: int = 42,
    ) -> None:
        _check_axes(fit, roles)
        self._fit = fit
        self._roles = roles
        self._n_genes = int(n_genes)
        self._reader = npz_reader
        self._ctrl_mean_cache: dict[tuple[tuple[str, str], ...], np.ndarray] = {}
        self._ctrl_missing: set[tuple[tuple[str, str], ...]] = set()
        self._lock = threading.Lock()
        self._miss_count = 0
        self._emission = str(emission)
        self._ctrl_pool_cells = int(ctrl_pool_cells)
        self._pool_cache_size = int(pool_cache_size)
        self._seed = int(seed)
        # Control MEANS are tiny and cached without bound; control CELL POOLS are
        # ~pool x G floats each, so they get a small LRU (tahoe has >1000 strata).
        self._ctrl_pool_cache: "OrderedDict[tuple[tuple[str, str], ...], np.ndarray]" = (
            OrderedDict()
        )
        # A fit-space fit moves the WHOLE matched pool's cells, as the STATE arm draws them.
        self._space = _space_reader(fit, npz_reader)
        self._space_pools: OrderedDict = OrderedDict()   # control key -> (cells, L_full, mean) in the fit space
        self.output_space = ModelOutputSpace.RAW if self._space is None else ModelOutputSpace.LOG1P
        if self._space is not None and self._emission != EMISSION_PER_CELL:
            raise ValueError(f"a {fit.params['fit_space']} fit is emitted per cell only, not {self._emission}")

    def _space_pool(self, qrow: QuadrupletRow) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(this row's control cells in the fit space, their L_full, their mean), LRU-cached."""
        key = self._control_id_tuple(qrow.control_id_keys)
        with self._lock:
            if key in self._space_pools:
                self._space_pools.move_to_end(key)
                return self._space_pools[key]
        cells, lib = self._space.control_with_library(qrow.control_id_keys)
        pool = (cells, lib, cells.astype(np.float64).mean(axis=0))
        with self._lock:
            self._space_pools[key] = pool
            while len(self._space_pools) > max(1, self._pool_cache_size):
                self._space_pools.popitem(last=False)
        return pool

    def _control_id_tuple(
        self, control_id_keys: dict[str, str],
    ) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((str(k), str(v)) for k, v in control_id_keys.items()))

    def _plate_ctrl_pool(self, qrow: QuadrupletRow) -> Optional[np.ndarray]:
        """Return a subsampled matrix of this row's control CELLS (LRU cached).

        Only used by the per-cell emission. Reading the pool also fills the mean
        cache (computed on the FULL pool, so the tile_mean path is unaffected).
        """
        ctrl_key = self._control_id_tuple(qrow.control_id_keys)
        with self._lock:
            pool = self._ctrl_pool_cache.get(ctrl_key)
            if pool is not None:
                self._ctrl_pool_cache.move_to_end(ctrl_key)
                return pool
            if ctrl_key in self._ctrl_missing:
                return None
        if self._plate_ctrl_mean(qrow, _want_pool=True) is None:
            return None
        with self._lock:
            pool = self._ctrl_pool_cache.get(ctrl_key)
            if pool is not None:
                self._ctrl_pool_cache.move_to_end(ctrl_key)
            return pool

    def _plate_ctrl_mean(
        self, qrow: QuadrupletRow, *, _want_pool: bool = False,
    ) -> Optional[np.ndarray]:
        """Return row mean of this row's control pool (cached, thread-safe)."""
        ctrl_key = self._control_id_tuple(qrow.control_id_keys)
        cached = self._ctrl_mean_cache.get(ctrl_key)
        if cached is not None and not _want_pool:
            return cached
        with self._lock:
            cached = self._ctrl_mean_cache.get(ctrl_key)
            if cached is not None and not _want_pool:
                return cached
            if ctrl_key in self._ctrl_missing:
                return None
            try:
                arr = self._reader.control(qrow.control_id_keys)
            except FileNotFoundError:
                log.warning(
                    "Control NPZ missing for %s (cell=%s plate=%s)",
                    ctrl_key, qrow.cell_name, qrow.plate,
                )
                self._ctrl_missing.add(ctrl_key)
                return None
            if arr.ndim != 2 or arr.shape[1] != self._n_genes:
                log.warning(
                    "Control NPZ shape %s; expected (*, %d) for %s",
                    arr.shape, self._n_genes, ctrl_key,
                )
                self._ctrl_missing.add(ctrl_key)
                return None
            mu = arr.astype(np.float64, copy=False).mean(axis=0)
            self._ctrl_mean_cache[ctrl_key] = mu
            if _want_pool:
                n_pool = int(arr.shape[0])
                if 0 < self._ctrl_pool_cells < n_pool:
                    # Deterministic per-stratum subsample: bounds the LRU footprint.
                    sub = np.random.default_rng(
                        _stable_seed(ctrl_key, self._seed),
                    ).choice(n_pool, size=self._ctrl_pool_cells, replace=False)
                    pool = arr[np.sort(sub)].astype(np.float32, copy=False)
                else:
                    pool = arr.astype(np.float32, copy=False)
                self._ctrl_pool_cache[ctrl_key] = pool
                while len(self._ctrl_pool_cache) > max(1, self._pool_cache_size):
                    self._ctrl_pool_cache.popitem(last=False)
            return mu

    def __call__(self, qrow: QuadrupletRow) -> Optional[np.ndarray]:
        n_cells = int(qrow.num_cells)
        if n_cells <= 0:
            self._miss_count += 1
            return None

        # The emission seed hashes the raw (cell, pert, dose, plate_norm), as the published bars did.
        roles = self._roles
        keys = qrow.condition_keys
        cell = str(keys[roles.cell_var])
        pert = str(keys[roles.pert_var])
        dose = roles.dose_of(keys)
        plate_norm = normalize_plate_for_eval(roles.plate_of(keys))
        if self._space is not None:
            try:
                cells, lib, mu = self._space_pool(qrow)
                delta = np.asarray(self._fit.predict(roles.arm_key(keys), mu), dtype=np.float64) - mu
            except (FileNotFoundError, KeyError) as exc:
                log.warning("%s: no prediction for %s | %s", self._fit.kind, dict(keys), exc)
                self._miss_count += 1
                return None
            rows = emit_in_fit_space(delta=delta, ctrl_cells=cells, ctrl_lib=lib, n_cells=n_cells,
                                     seed=_stable_seed((cell, pert, dose, plate_norm), self._seed))
            if not np.isfinite(rows).all():
                log.warning("%s: non-finite fit-space emission for %s; skipping.", self._fit.kind, dict(keys))
                self._miss_count += 1
                return None
            return rows
        matched_ctrl = self._plate_ctrl_mean(qrow)
        if matched_ctrl is None:
            self._miss_count += 1
            return None
        try:
            vec = self._fit.predict(roles.arm_key(keys), matched_ctrl)
        except KeyError as exc:
            log.warning("%s: no prediction for %s | %s", self._fit.kind, dict(keys), exc)
            self._miss_count += 1
            return None

        vec = np.asarray(vec, dtype=np.float32).ravel()
        if vec.shape[0] != self._n_genes:
            log.warning(
                "Prediction length %d != n_genes %d for %s|%s|%s|plate=%s; skipping.",
                vec.shape[0], self._n_genes, cell, pert, dose, plate_norm,
            )
            self._miss_count += 1
            return None

        if self._emission == EMISSION_TILE_MEAN:
            return np.broadcast_to(vec, (n_cells, self._n_genes))

        pool = self._plate_ctrl_pool(qrow)
        if pool is None:
            log.warning(
                "No control pool for %s|%s|%s|plate=%s; cannot emit per-cell.",
                cell, pert, dose, plate_norm,
            )
            self._miss_count += 1
            return None
        rows = emit_per_cell_delta(
            pred_vec_raw=vec,
            ctrl_cells_raw=pool,
            ctrl_mean_raw=matched_ctrl,
            n_cells=n_cells,
            seed=_stable_seed((cell, pert, dose, plate_norm), self._seed),
        )
        # A non-finite row silently drops pearson/r2/energy (rank metrics survive),
        # which reads as "the baseline is bad" rather than "the emission is broken".
        if not np.isfinite(rows).all():
            log.warning(
                "Non-finite per-cell emission for %s|%s|%s|plate=%s; skipping.",
                cell, pert, dose, plate_norm,
            )
            self._miss_count += 1
            return None
        return rows

    @property
    def miss_count(self) -> int:
        return self._miss_count

    @property
    def n_ctrl_means_cached(self) -> int:
        return len(self._ctrl_mean_cache)

    @property
    def n_ctrl_means_missing(self) -> int:
        return len(self._ctrl_missing)


class ThreeFactorPredGetter:
    """``pred_getter`` for a baseline fit on two context axes (PBMC donor x cell type).

    The control is this condition's whole pooled control (no subsample), and each
    condition's emission seed is adler32 of "rep|pert|group": both as in the published
    PBMC baseline bars, so they rescore identically. A covariate missing from the fit
    is a logged miss, never a zero-filled prediction (that would read as a confident null).
    """

    def __init__(self, *, fit: Baseline, roles: BaselineConditionRoles, npz_reader: EvalNpzReader,
                 n_genes: int) -> None:
        if len(roles.context_vars) != 2:
            raise ValueError(f"three-factor fit needs two context axes, got {roles.context_vars}")
        _check_axes(fit, roles)
        self._kind, self._fit, self._roles = fit.kind, fit, roles
        self._reader, self._n_genes = npz_reader, int(n_genes)
        self._rep, self._grp = roles.context_vars
        self._pert = roles.pert_var
        self._ctrl: dict[tuple, tuple] = {}   # control key -> (cells, mean, L_full or None), in the fit's space
        self._lock = threading.Lock()
        self._miss_count = 0
        self._space = _space_reader(fit, npz_reader)
        self.output_space = ModelOutputSpace.RAW if self._space is None else ModelOutputSpace.LOG1P

    def _control(self, qrow: QuadrupletRow) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        key = tuple(sorted((str(k), str(v)) for k, v in qrow.control_id_keys.items()))
        with self._lock:
            if key not in self._ctrl:
                if self._space is None:
                    cells = np.asarray(self._reader.control(qrow.control_id_keys))
                    self._ctrl[key] = (cells.astype(np.float32), cells.astype(np.float64).mean(axis=0), None)
                else:
                    cells, lib = self._space.control_with_library(qrow.control_id_keys)
                    self._ctrl[key] = (cells, cells.astype(np.float64).mean(axis=0), lib)
            return self._ctrl[key]

    def __call__(self, qrow: QuadrupletRow) -> Optional[np.ndarray]:
        k = qrow.condition_keys
        rep, grp, pert = str(k[self._rep]), str(k[self._grp]), str(k[self._pert])
        try:
            cells, mean, lib = self._control(qrow)
            vec = self._fit.predict(self._roles.arm_key(k), mean)
        except (FileNotFoundError, KeyError) as exc:
            log.warning("%s: no prediction for %s | %s", self._kind, dict(k), exc)
            self._miss_count += 1
            return None
        seed = zlib.adler32(f"{rep}|{pert}|{grp}".encode()) & 0x7FFFFFFF
        if lib is None:
            rows = emit_per_cell_delta(pred_vec_raw=vec, ctrl_cells_raw=cells, ctrl_mean_raw=mean,
                                       n_cells=int(qrow.num_cells), seed=seed)
        else:
            rows = emit_in_fit_space(delta=np.asarray(vec, dtype=np.float64) - mean, ctrl_cells=cells,
                                     ctrl_lib=lib, n_cells=int(qrow.num_cells), seed=seed)
        if not np.isfinite(rows).all():
            log.warning("%s: non-finite emission for %s; skipping.", self._kind, dict(k))
            self._miss_count += 1
            return None
        return rows

    @property
    def miss_count(self) -> int:
        return self._miss_count


def validate_gene_agreement(
    artifact_gene_names: Sequence[str],
    canonical: Sequence[str],
    artifact_path: Path,
) -> None:
    if list(artifact_gene_names) == list(canonical):
        return
    raise SystemExit(
        "Gene names in baseline artifact differ from the --gene-names file. "
        "Both the baseline fit and the truth cache must use the same ordered "
        "gene list.\n"
        f"  artifact : {artifact_path} (n={len(artifact_gene_names)})\n"
        f"  canonical: n={len(canonical)} "
        f"first-mismatch idx="
        f"{next((i for i, (a, b) in enumerate(zip(artifact_gene_names, canonical)) if a != b), -1)}",
    )
