"""A fitted baseline on disk: one ``<kind>.npz`` of plain arrays plus a JSON header.

Every kind is a few named tables of (G,) vectors keyed by role tuples (src/baselines/roles.py):

    perturbation_mean, linear_ridge  shift[kept + pert]           predict = control + shift
    context_mean                     context[held + kept + stratum] predict = the value itself
    additive                         mu[()], then one effect table per axis (additive.terms)

The header holds the kind, the gene order, the role columns, the fit parameters, and each
table's keys; each table's vectors are one (n_keys, G) float64 array. Nothing is pickled, so
moving code never orphans an artifact; it loads with ``allow_pickle=False``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.baselines import additive
from src.baselines.roles import ArmKey, _as_vec

logger = logging.getLogger(__name__)

KINDS = ("perturbation_mean", "context_mean", "linear_ridge", "additive")


@dataclass(frozen=True)
class Baseline:
    """One fitted kind. A covariate missing from a table is a KeyError (the caller's miss),
    never a zero-filled prediction, which would read as a confident null; the additive
    model is the exception, see src/baselines/additive.py."""
    kind: str
    genes: tuple[str, ...]
    axes: dict[str, list[str]]
    tables: dict[str, dict[tuple, np.ndarray]]
    params: dict = field(default_factory=dict)

    def predict(self, key: ArmKey, control: np.ndarray) -> np.ndarray:
        """(G,) prediction for one condition, given its matched control mean."""
        if self.kind == "context_mean":
            return _row(self.tables["context"], key.held + key.kept + key.stratum, "context").copy()
        control = _as_vec(control, len(self.genes), "control")
        if self.kind in ("perturbation_mean", "linear_ridge"):
            return control + _row(self.tables["shift"], key.kept + key.pert, "shift")
        shift = self.tables["mu"][()]
        for name, level in zip(additive.term_names(self.axes["kept"]), additive.terms(key)):
            effect = self.tables[name].get(level)
            if effect is None:
                logger.warning("additive: %s level %s not in train; its effect is zero (the average)", name, level)
                continue
            shift = shift + effect
        return control + shift

    def save(self, path: Path) -> None:
        header = {"kind": self.kind, "genes": list(self.genes), "axes": self.axes, "params": self.params,
                  "keys": {name: [list(k) for k in table] for name, table in self.tables.items()}}
        arrays = {name: np.stack(list(table.values())).astype(np.float64) for name, table in self.tables.items()}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, header=np.array(json.dumps(header)), **arrays)

    @classmethod
    def load(cls, path: Path) -> "Baseline":
        with np.load(path, allow_pickle=False) as z:
            if "header" not in z.files:
                raise ValueError(f"{path} has no header: not a baseline artifact (or a pickled one from "
                                 "before the plain-array format; refit it with scripts/baselines/fit.py)")
            header = json.loads(str(z["header"]))
            tables = {name: dict(zip((tuple(k) for k in keys), z[name])) for name, keys in header["keys"].items()}
        if header["kind"] not in KINDS:
            raise ValueError(f"{path}: unknown baseline kind {header['kind']!r}")
        return cls(header["kind"], tuple(header["genes"]), header["axes"], tables, header["params"])


def _row(table: dict[tuple, np.ndarray], key: tuple, what: str) -> np.ndarray:
    if key not in table:
        raise KeyError(f"{what} {key!r} was not in the fit")
    return table[key]
