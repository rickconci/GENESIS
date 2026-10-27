"""Truth and controls, one load per unit of a context group.

A context group is one condition (the columns a design shares) measured on several
units (Tahoe: plates). Split-half on Tahoe puts the plate inside the condition key,
so a group there has one unit. A dataset that pools plates passes every plate as a
unit and scores the pool. Either way each unit's arrays are read once.

The readers try both replicate spellings, because the 2k cache stores ``6`` and the
5,750 cache stores ``plate6``.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterator, Mapping, Protocol, Sequence

import numpy as np


class _TruthReader(Protocol):
    def perturbed(self, keys: dict[str, str], **kw) -> np.ndarray: ...
    def control(self, keys: dict[str, str], **kw) -> np.ndarray: ...


class _GroupSpec(Protocol):
    condition_cols: Sequence[str]
    replicate_col: str
    control_cols: Sequence[str]


@dataclass(frozen=True)
class UnitTruth:
    """One unit (one plate) of a context group."""
    unit: str
    keys: dict[str, str]
    truth: np.ndarray
    control: np.ndarray


@dataclass(frozen=True)
class GroupRef:
    """A group before anything is read: enough for a design to say which clouds it needs.

    ``values`` keeps the caller's types (a dose stays a float), because the replicate seeds
    hash those types. ``unit_keys`` are the cache-key spelling of each unit's condition.
    ``n_cells`` is each unit's truth size; loading checks the cache agrees.
    """
    values: Mapping[str, object]
    units: tuple[str, ...]
    unit_keys: tuple[dict[str, str], ...]
    n_cells: tuple[int, ...]

    @property
    def condition(self) -> dict[str, str]:
        return {k: str(v) for k, v in self.values.items()}


def group_ref(spec: _GroupSpec, cond: Mapping[str, object], units: Sequence[str],
              n_cells: Sequence[int]) -> GroupRef:
    if not units:
        raise ValueError(f"context group {dict(cond)} has no units")
    if len(set(map(str, units))) != len(units):
        raise ValueError(f"context group {dict(cond)} lists a unit twice: {list(units)}")
    if len(n_cells) != len(units) or any(int(n) < 1 for n in n_cells):
        raise ValueError(f"context group {dict(cond)}: n_cells {list(n_cells)} for units {list(units)}")
    values = {c: cond[c] for c in spec.condition_cols}
    keys = tuple({**{c: str(v) for c, v in values.items()}, spec.replicate_col: str(u)} for u in units)
    return GroupRef(values, tuple(map(str, units)), keys, tuple(map(int, n_cells)))


@dataclass(frozen=True)
class ContextGroup:
    ref: GroupRef
    units: tuple[UnitTruth, ...]


def _replicate_variants(keys: dict[str, str], col: str) -> list[dict[str, str]]:
    """Caches disagree on the replicate id: the 2k cache stores ``6``, the 5,750 one ``plate6``."""
    if col not in keys:
        return [keys]
    v = str(keys[col])
    alt = v[len(col):] if v.startswith(col) else f"{col}{v}"
    return [keys, {**keys, col: alt}]


def _read_any(fn, keys: dict[str, str], col: str, **kw) -> np.ndarray:
    last: FileNotFoundError | None = None
    for variant in _replicate_variants(keys, col):
        try:
            return fn(variant, **kw)
        except FileNotFoundError as exc:
            last = exc
    if last is None:  # _replicate_variants always yields at least the keys themselves
        raise FileNotFoundError(keys)
    raise last


def read_perturbed(reader: _TruthReader, keys: dict[str, str], col: str = "plate", **kw) -> np.ndarray:
    return _read_any(reader.perturbed, keys, col, **kw)


def read_control(reader: _TruthReader, keys: dict[str, str], col: str = "plate", **kw) -> np.ndarray:
    return _read_any(reader.control, keys, col, **kw)


def read_control_with_library(reader: _TruthReader, keys: dict[str, str],
                              col: str = "plate") -> tuple[np.ndarray, np.ndarray]:
    """A control pool's cells and each cell's full-transcriptome library, from ONE file (a ``__with_libfull``
    reader). A Tahoe pool can sit under two file names (``plate_14`` and ``plate_plate14``) and the packed sidecar
    knows only one, so this takes the spelling that has a library: rows and libraries then come from the same
    file. Raises when no spelling has one."""
    for variant in _replicate_variants(keys, col):
        try:
            lib = reader.control_library_full(variant)
        except FileNotFoundError:
            continue
        if lib is not None:
            return reader.control(variant), lib
    raise FileNotFoundError(f"no full-transcriptome library for the control pool {dict(keys)}")


def control_library(reader: _TruthReader, keys: dict[str, str], cells: np.ndarray, col: str = "plate") -> np.ndarray:
    """The L_full of ``cells``, a control pool read elsewhere, row for row: the file that carries the library
    must hold exactly these cells, or the libraries would land on the wrong ones."""
    found, lib = read_control_with_library(reader, keys, col)
    if not np.array_equal(found, cells):
        raise ValueError(f"control pool {dict(keys)}: its two files differ; its libraries cannot be aligned")
    return lib


def _as_cells(name: str, keys: Mapping[str, str], arr: np.ndarray, n_genes: int | None) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(f"{name} for {dict(keys)} is empty or not a cell matrix: {arr.shape}")
    if n_genes is not None and arr.shape[1] != n_genes:
        raise ValueError(f"{name} for {dict(keys)} has {arr.shape[1]} genes, the group's truth has {n_genes}")
    return arr


def load_group(reader: _TruthReader, spec: _GroupSpec, ref: GroupRef) -> ContextGroup:
    """Each unit's truth and controls, read once, in the ref's unit order."""
    loaded: list[UnitTruth] = []
    n_genes: int | None = None
    for unit, keys, n in zip(ref.units, ref.unit_keys, ref.n_cells):
        truth = _as_cells("truth", keys, read_perturbed(reader, keys, spec.replicate_col), n_genes)
        if truth.shape[0] != n:  # the clouds were asked for at n cells; scoring would misalign them
            raise ValueError(f"truth for {keys} has {truth.shape[0]} cells, the group declared {n}")
        n_genes = int(truth.shape[1])
        ctrl_keys = {c: keys[c] for c in spec.control_cols if c in keys}
        missing = [c for c in spec.control_cols if c not in keys]
        if missing:
            raise KeyError(f"control keys {missing} are not in {keys}")
        control = _as_cells("control", ctrl_keys, read_control(reader, ctrl_keys, spec.replicate_col), n_genes)
        loaded.append(UnitTruth(unit, keys, truth, control))
    return ContextGroup(ref, tuple(loaded))


class GroupPrefetcher:
    """One group ahead, same shape as ConditionPrefetcher: group i scores while i+1 is read."""

    def __init__(self, reader: _TruthReader, spec: _GroupSpec, *, max_workers: int = 2) -> None:
        self._reader, self._spec, self._workers = reader, spec, int(max_workers)

    def _load(self, ref: GroupRef) -> ContextGroup:
        return load_group(self._reader, self._spec, ref)

    def iter_groups(self, groups: Sequence[GroupRef]) -> Iterator[ContextGroup]:
        n = len(groups)
        if n == 0:
            return
        if self._workers <= 0:
            for item in groups:
                yield self._load(item)
            return
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            futures = {0: pool.submit(self._load, groups[0])}
            for i in range(n):
                if i + 1 < n:
                    futures[i + 1] = pool.submit(self._load, groups[i + 1])
                yield futures.pop(i).result()
