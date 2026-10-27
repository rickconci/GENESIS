"""Which condition column plays which role for the expression baselines, and the arm table they fit.

Every train condition is an *arm*: one point on five roles, read from the dataset registry
(``core_util_data/datasets/*.yaml``), so a new dataset is a registry entry, not new code:

* held axis ``h``   -- ``biological_context_columns`` (Tahoe ``cell_name``, PBMC ``donor``): what
  a test condition is held out along, so what the transfer baselines average over.
* kept axes ``k``   -- context columns a run adds to the control match keys beyond ``h``
  (PBMC per-well cache: ``cell_type``; Tahoe: none). A perturbation's shift is per ``k``.
* stratum ``r``     -- ``plate_column`` (Tahoe ``plate``; PBMC none): controls are matched
  within it, and every (h, k, r) arm is one vote.
* perturbation ``p`` -- ``perturbation_columns`` (Tahoe ``drug, dose``; PBMC ``cytokine``).
* control unit ``u`` -- a condition column that is none of the above (PBMC ``bc1_well``):
  its units are mixed, weighted by cell count, into one control per (h, k, r).

A run that conditions on more than the registry default (PBMC cell type + well) passes the
run's ``condition_variables`` / ``control_match_keys``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple, Optional, Sequence

import numpy as np

from src.data.vocab import normalize_plate

# Stand-ins for a dataset without the column; only the scoring seed still reads them.
NO_DOSE = 0.0
NO_PLATE = "_"

_PLATE_LOOSE_RE = re.compile(r"(?i)^plate(\d+)(?:[_].*)?$")


def normalize_plate_for_eval(raw: str) -> str:
    """Map work-queue plate strings to artifact / cache plate_norm keys."""
    s = str(raw).strip()
    m = _PLATE_LOOSE_RE.match(s)
    if m:
        return m.group(1)
    return normalize_plate(s)


class ArmKey(NamedTuple):
    """One arm's canonical values per role. Doses round to 6 d.p. and plates normalise, so a
    fit-time key and a predict-time key built from differently rendered strings still agree."""
    held: tuple
    kept: tuple
    stratum: tuple
    pert: tuple


@dataclass(frozen=True)
class BaselineConditionRoles:
    """Mapping from baseline roles to a dataset's condition-variable names."""

    dataset_alias: str
    cell_var: str
    pert_var: str
    dose_var: Optional[str]
    plate_var: Optional[str]
    condition_variables: tuple[str, ...]
    control_condition: dict[str, Any]
    control_match_keys: tuple[str, ...]
    context_vars: tuple[str, ...] = ()
    unit_var: Optional[str] = None
    held_vars: tuple[str, ...] = ()
    kept_vars: tuple[str, ...] = ()

    @property
    def control_pert_value(self) -> str:
        """The control label for :attr:`pert_var` (``control_drug`` in the baselines)."""
        return str(self.control_condition[self.pert_var])

    @property
    def pert_vars(self) -> tuple[str, ...]:
        return (self.pert_var,) + ((self.dose_var,) if self.dose_var else ())

    @property
    def stratum_vars(self) -> tuple[str, ...]:
        return (self.plate_var,) if self.plate_var else ()

    @property
    def axes(self) -> dict[str, list[str]]:
        """Column names per role, recorded in every artifact and checked at scoring."""
        return {"held": list(self.held_vars), "kept": list(self.kept_vars),
                "stratum": list(self.stratum_vars), "pert": list(self.pert_vars)}

    def canon(self, column: str, value: Any) -> Any:
        if column == self.dose_var:
            return round(float(value), 6)
        if column == self.plate_var:
            return normalize_plate_for_eval(str(value))
        return str(value)

    def arm_key(self, condition_keys: Mapping[str, Any]) -> ArmKey:
        def part(cols: Sequence[str]) -> tuple:
            return tuple(self.canon(c, condition_keys[c]) for c in cols)
        return ArmKey(part(self.held_vars), part(self.kept_vars), part(self.stratum_vars), part(self.pert_vars))

    def dose_of(self, condition_keys: Mapping[str, Any]) -> float:
        """Dose for this condition, or :data:`NO_DOSE` when the dataset has none."""
        if self.dose_var is None:
            return NO_DOSE
        return float(condition_keys[self.dose_var])

    def plate_of(self, condition_keys: Mapping[str, Any]) -> str:
        """Raw (un-normalised) plate id, or :data:`NO_PLATE` when there are no plates."""
        if self.plate_var is None:
            return NO_PLATE
        return str(condition_keys[self.plate_var])


def baseline_condition_roles(
    dataset_alias: str,
    *,
    condition_variables: Optional[Sequence[str]] = None,
    control_match_keys: Optional[Sequence[str]] = None,
) -> BaselineConditionRoles:
    """Build :class:`BaselineConditionRoles` for a dataset alias (``tahoe``/``parse``).

    Args:
        dataset_alias: Registry alias, as used in ``results/<dataset>/``.
        condition_variables: Override the registry's condition columns.
        control_match_keys: Override the registry's control match keys.

    Raises:
        ValueError: If there is no context or perturbation column, the held axis is not
            matched on, or the condition columns leave more than one control unit.
    """
    from src.config import DatasetConditionSpec

    spec = DatasetConditionSpec.from_alias(dataset_alias)
    if not spec.context_vars or not spec.perturbation_vars:
        raise ValueError(
            f"Dataset {dataset_alias!r} needs at least one biological_context_columns "
            f"and one perturbation_columns entry to fit expression baselines.",
        )
    plate = str(spec.plate_column) if spec.plate_column else None
    cond = tuple(str(v) for v in (condition_variables or spec.conditioning_vars))
    match = tuple(str(v) for v in (control_match_keys or spec.control_match_keys))
    perts = tuple(str(v) for v in spec.perturbation_vars)
    context = tuple(v for v in match if v != plate)
    held = tuple(str(v) for v in spec.context_vars)
    units = [v for v in cond if v not in context and v not in perts and v != plate]
    if not context or len(units) > 1 or not set(held) <= set(context):
        raise ValueError(
            f"{dataset_alias}: condition columns {cond} with match keys {match} give context "
            f"{context} and control units {units}; need >= 1 context axis including the held "
            f"axis {held}, and <= 1 unit.",
        )
    return BaselineConditionRoles(
        dataset_alias=str(spec.dataset_alias),
        cell_var=context[0],
        pert_var=perts[0],
        dose_var=perts[1] if len(perts) > 1 else None,
        plate_var=plate,
        condition_variables=cond,
        control_condition={str(k): v for k, v in spec.control_condition.items()},
        control_match_keys=match,
        context_vars=context,
        unit_var=units[0] if units else None,
        held_vars=held,
        kept_vars=tuple(v for v in context if v not in held),
    )


# The arm table: what every baseline fits on, in the caller's order

@dataclass(frozen=True)
class Arms:
    """Train arms in fit order. ``mean`` is each arm's mean expression as read; ``delta`` is it
    minus its stratum's control, in float64. Order matters: group means sum rows in it."""
    keys: tuple[ArmKey, ...]
    mean: np.ndarray
    delta: np.ndarray

    @classmethod
    def build(cls, keys: Sequence[ArmKey], mean: np.ndarray, control: np.ndarray) -> "Arms":
        """``control`` is row-aligned with ``mean``: each arm's own stratum control."""
        keys = tuple(keys)
        if mean.ndim != 2 or mean.shape != control.shape or mean.shape[0] != len(keys) or not keys:
            raise ValueError(f"{len(keys)} keys, means {mean.shape}, controls {control.shape}")
        seen: set[ArmKey] = set()
        for k in keys:
            if k in seen:
                raise ValueError(f"two train arms share the key {k}: a control unit is not mixed out")
            seen.add(k)
        if not (np.isfinite(mean).all() and np.isfinite(control).all()):
            raise ValueError("non-finite arm or control means")
        return cls(keys, mean, mean.astype(np.float64) - control.astype(np.float64))


def group_mean(values: np.ndarray, labels: Sequence[tuple]) -> dict[tuple, np.ndarray]:
    """Mean of the rows sharing a label, one vote per row, rows summed in arm order."""
    rows: dict[tuple, list[int]] = {}
    for i, label in enumerate(labels):
        rows.setdefault(label, []).append(i)
    return {label: np.asarray(values[idx], dtype=np.float64).mean(axis=0) for label, idx in rows.items()}


def _as_vec(v: np.ndarray, n_genes: int, where: str) -> np.ndarray:
    vec = np.asarray(v, dtype=np.float64).ravel()
    if vec.shape != (n_genes,):
        raise ValueError(f"{where}: expected ({n_genes},), got {vec.shape}")
    if not np.isfinite(vec).all():
        raise ValueError(f"{where}: non-finite values")
    return vec
