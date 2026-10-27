"""What a condition is, and which of its units form a scored pair.

The replicate column and the condition columns come from the dataset registry.
Filters drop pairs that are not a real repeat measurement (seen in training, or
a plate with too few cells).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.dataset_registry import load_dataset_registry


@dataclass(frozen=True)
class ReplicateDesign:
    dataset: str
    condition_cols: tuple[str, ...]  # biological context + perturbation: one condition
    replicate_col: str               # independent runs of that condition
    control_cols: tuple[str, ...]    # control pools are matched on these
    file_cols: tuple[str, ...]       # prediction file-name order (the eval cache key)

    @property
    def context_col(self) -> str:
        return self.condition_cols[0]


def replicate_design(dataset: str) -> ReplicateDesign:
    reg = load_dataset_registry(dataset)
    rep = reg.experimental_replicate_column
    if rep is None:
        raise ValueError(f"dataset {reg.name!r} has no experimental_replicate_column: nothing is "
                         f"measured twice, so there is no experimental-replicate reference")
    cond = tuple(c for c in (*reg.biological_context_columns, *reg.perturbation_columns) if c != rep)
    return ReplicateDesign(reg.name, cond, rep, reg.derive_control_match_keys(),
                           reg.derive_conditioning_vars())


def _replicate_label(d: ReplicateDesign, rep: str) -> str:
    """Eval-cache form of a replicate id: ``6`` -> ``plate6``."""
    rep = str(rep)
    return rep if rep.startswith(d.replicate_col) else f"{d.replicate_col}{rep}"


def enumerate_pairs(d: ReplicateDesign, conditions: pd.DataFrame,
                    replicates_col: str) -> list[tuple[dict[str, Any], str, str]]:
    """Every (condition, replicate i, replicate j) from a table listing each condition's replicates."""
    missing = [c for c in (*d.condition_cols, replicates_col) if c not in conditions.columns]
    if missing:
        raise KeyError(f"conditions table is missing {missing}")
    out = []
    for r in conditions.to_dict("records"):
        cond = {c: r[c] for c in d.condition_cols}
        out += [(cond, str(i), str(j)) for i, j in combinations(str(r[replicates_col]).split(","), 2)]
    return out


# A repeat pair tests generalization only if the model never saw its context with its
# perturbation. Splits that hold out only some concentrations of a compound in a line leave
# that line x compound in training at the others; such pairs are dropped.
def trained_combinations(dataset: str, split_yaml: Path,
                         summary: Path | None = None) -> set[tuple[str, str]]:
    """(context, perturbation) with at least one TRAINING condition, at any dose or plate.

    TRAIN = every non-control condition minus the split's test and validation queues, exactly
    as the baselines are fit (src/baselines/fit.py). Raises if the split cannot be read
    or holds out nothing.
    """
    from src.data.splits.generation import load_split_policy
    from src.baselines.roles import baseline_condition_roles
    from src.baselines.fit import held_out, train_rows  # lazy: fit imports the eval stack
    split_yaml = Path(split_yaml)
    if not split_yaml.is_file():
        raise FileNotFoundError(f"split YAML not found: {split_yaml}")
    summary = Path(summary or load_split_policy(dataset).condition_summary)
    roles = baseline_condition_roles(dataset)
    held, _sizes = held_out(summary, split_yaml, roles, min_cells=1)
    return {(r[roles.cell_var], r[roles.pert_var]) for r in train_rows(summary, roles, held, 1, None, 0)}


def drop_seen_in_training(pairs: pd.DataFrame, d: ReplicateDesign,
                          trained: set[tuple[str, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a pair table into (kept, dropped): dropped pairs' (context, perturbation) was trained."""
    ctx, pert = d.condition_cols[0], d.condition_cols[1]
    seen = np.fromiter(((str(c), str(p)) in trained for c, p in zip(pairs[ctx], pairs[pert])),
                       dtype=bool, count=len(pairs))
    return pairs[~seen], pairs[seen]


# A plate holding a handful of a condition's cells is a stray population, not a repeat
# measurement: both plates need >= 50 (three Tahoe pairs have 8-24 on one plate).
MIN_CELLS_PER_PLATE = 50


def drop_small_plates(pairs: pd.DataFrame,
                      min_cells: int = MIN_CELLS_PER_PLATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a pair table into (kept, dropped): dropped pairs observe < min_cells on either plate."""
    missing = [c for c in ("n_truth_i", "n_truth_j") if c not in pairs.columns]
    if missing:
        raise KeyError(f"pair table has no observed cell counts: missing {missing}")
    n = pairs[["n_truth_i", "n_truth_j"]].to_numpy(dtype=float)
    if np.isnan(n).any():
        raise ValueError(f"{int(np.isnan(n).any(axis=1).sum())} pairs have no observed cell count")
    small = n.min(axis=1) < min_cells
    return pairs[~small], pairs[small]
