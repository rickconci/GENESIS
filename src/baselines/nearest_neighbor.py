"""Nearest neighbour: copy the closest train context's cells for the test perturbation.

"Closest" is the train context whose responses to the OTHER perturbations look most like
the test context's: mean (1 - cosine) over the perturbations both have, on per-arm mean
displacements in a shared PCA-100 log1p space (``deltas.npz``). The held-out perturbation
never enters the distance: the test arm has no delta. The prediction is the neighbour's raw
cells for the test perturbation, subsampled to the truth's cell count -- an empirical floor
that any model claiming to generalise across contexts must beat.

Axes come from roles.py. Context = control-match keys minus the plate (Tahoe ``cell_name``;
PBMC ``donor, cell_type``). A neighbour shares every context axis after the first and differs
on the first: a CD14 Mono test copies another donor's CD14 Mono, never a T cell.

  1. ``build_deltas``: per train arm, mean log1p(arm) - mean log1p(matched control) over
     every cell (condition_stats.py), projected on the top PCA-100 axes of those
     responses. Cached as deltas.npz; rebuilding re-picks neighbours.
  2. ``distance_table``: the neighbour of every context. Copying its cells at scoring time
     is src/eval/arms/copy_neighbor.py.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd
import torch

from src.baselines.condition_stats import ConditionStats, pooled_mean
from src.baselines.roles import BaselineConditionRoles
from src.data.vocab import to_categorical_str
from src.data.data_utils import _coerce_split_config
from src.data.splits.predicates import apply_split_predicate

log = logging.getLogger(__name__)


# The tags: (context, perturbation), comparable between summary, queue and cache

PCA_DIM = 100


def canon_pert(values: tuple, roles: BaselineConditionRoles) -> tuple:
    """Perturbation key; the dose rounds to 5 d.p. so float32 0.05 and the string '0.05' match."""
    return (str(values[0]),) + ((round(float(values[1]), 5),) if roles.dose_var else ())


def tag_of(keys: Mapping, roles: BaselineConditionRoles) -> tuple:
    pert = (keys[roles.pert_var],) + ((keys[roles.dose_var],) if roles.dose_var else ())
    return tuple(str(keys[v]) for v in roles.context_vars), canon_pert(pert, roles)


@dataclass(frozen=True)
class ArmSummaries:
    """One response delta per train arm: ``deltas[i]`` belongs to ``tags[i]`` = (context, perturbation)."""
    tags: list
    deltas: np.ndarray

    def __post_init__(self) -> None:
        if len(self.tags) != len(self.deltas):
            raise ValueError(f"{len(self.tags)} tags for {len(self.deltas)} deltas")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError(f"{len(self.tags) - len(set(self.tags))} duplicate arm tags")


def load_deltas(path: Path, roles: BaselineConditionRoles) -> ArmSummaries:
    z = np.load(path, allow_pickle=True)
    tags = [(tuple(str(x) for x in t[0]), canon_pert(tuple(t[1]), roles)) for t in z["tags"]]
    log.info("deltas %s from %s", z["deltas"].shape, path)
    return ArmSummaries(tags, z["deltas"])


def _train_rows(summary: Path, split_yaml: Path) -> pd.DataFrame:
    """Summary rows outside the split's test and validation predicates."""
    frame = pd.read_parquet(summary)
    cfg = _coerce_split_config(split_yaml)
    held = pd.Series(False, index=frame.index)
    for name in ("test", "validation"):
        block = getattr(cfg, name, None)
        if block is not None:
            held |= apply_split_predicate(frame, block)
    return frame.loc[~held]


def train_arms(summary: Path, split_yaml: Path, roles: BaselineConditionRoles, min_cells: int) -> pd.DataFrame:
    """Non-control train rows with at least ``min_cells`` cells, one per condition (plate, well)."""
    train = _train_rows(summary, split_yaml)
    return train.loc[(train[roles.pert_var].astype(str) != roles.control_pert_value) & (train["num_cells"] >= min_cells)]


def train_controls(summary: Path, split_yaml: Path, roles: BaselineConditionRoles) -> pd.DataFrame:
    """Control train rows (PBMC: the PBS wells a split does not hold out)."""
    train = _train_rows(summary, split_yaml)
    return train.loc[train[roles.pert_var].astype(str) == roles.control_pert_value]


def _row_keys(train: pd.DataFrame, cols) -> list[dict]:
    # Iterating a Series boxes float32 doses to Python floats, exactly as the published
    # caches were built (the reader normalises the dose string itself).
    return [dict(zip(cols, (str(v) for v in vals))) for vals in zip(*(train[c] for c in cols))]


def held_contexts(summary: Path, split_yaml: Path, roles: BaselineConditionRoles) -> list[tuple]:
    """Contexts with any condition in the split's test block, sorted."""
    frame = pd.read_parquet(summary)
    test = frame.loc[apply_split_predicate(frame, _coerce_split_config(split_yaml).test)]
    return sorted({tuple(str(r[v]) for v in roles.context_vars) for r in test.to_dict("records")})


# Build the deltas: every cell, from one pass over the shards (condition_stats.py)

def response_deltas(stats: ConditionStats, train: pd.DataFrame, controls: pd.DataFrame,
                    roles: BaselineConditionRoles, half: Optional[int] = None) -> ArmSummaries:
    """Gene-space response of every train arm: mean log1p(arm) - mean log1p(control), each over
    all cells (or one half). An arm measured in several control strata (Tahoe plates) gets the
    cell-weighted mix of its stratum-matched responses; PBMC wells pool within one stratum.

    The control of a stratum is its train control cells; a stratum whose controls are all held
    out (PBMC donors with every PBS well in test) uses all of them, and one with none at all
    drops its arms. Both are logged, as the per-file builder this replaces did.
    """
    if tuple(stats.columns) != tuple(roles.condition_variables):
        raise ValueError(f"stats columns {stats.columns} != condition columns {roles.condition_variables}")
    halves = slice(None) if half is None else slice(half, half + 1)

    def row(keys: Mapping) -> int:
        k = tuple(to_categorical_str(float(keys[c])) if c == roles.dose_var else str(keys[c]) for c in stats.columns)
        if k not in stats.index:
            raise KeyError(f"no cells in the condition stats for {dict(zip(stats.columns, k))}")
        return stats.index[k]

    def stratum(keys: Mapping) -> tuple:
        return tuple(str(keys[c]) for c in roles.control_match_keys)

    train_ctrl: dict[tuple, list[int]] = {}
    for keys in _row_keys(controls, roles.condition_variables):
        train_ctrl.setdefault(stratum(keys), []).append(row(keys))
    all_ctrl: dict[tuple, list[int]] = {}
    for i, k in enumerate(stats.keys):
        keys = dict(zip(stats.columns, k))
        if keys[roles.pert_var] == roles.control_pert_value:
            all_ctrl.setdefault(stratum(keys), []).append(i)
    arms: dict[tuple, dict[tuple, list[int]]] = {}
    for keys in _row_keys(train, roles.condition_variables):
        arms.setdefault(tag_of(keys, roles), {}).setdefault(stratum(keys), []).append(row(keys))

    tags, deltas, fallback, no_ctrl, empty = [], [], set(), set(), 0
    for tag in sorted(arms):
        total, n_all = 0.0, 0
        for g, rows in arms[tag].items():
            ctrl = train_ctrl.get(g) or all_ctrl.get(g)
            if not ctrl:
                no_ctrl.add(g)
                continue
            if g not in train_ctrl:
                fallback.add(g)
            if stats.n[rows, halves].sum() == 0 or stats.n[ctrl, halves].sum() == 0:
                empty += 1                                   # only in one half: an arm of a few cells
                continue
            m, n = pooled_mean(stats, rows, half)
            total = total + n * (m - pooled_mean(stats, ctrl, half)[0])
            n_all += n
        if n_all:
            tags.append(tag)
            deltas.append(total / n_all)
    if fallback:
        log.warning("%d control strata have no train control cells; using all of theirs: %s", len(fallback), sorted(fallback))
    if no_ctrl:
        log.warning("%d control strata have no control cells at all; their arms are dropped: %s", len(no_ctrl), sorted(no_ctrl))
    if empty:
        log.warning("half %s: %d (arm, stratum) pairs have no cells in this half; skipped", half, empty)
    log.info("responses: %d of %d train arms", len(tags), len(arms))
    return ArmSummaries(tags, np.array(deltas, dtype=np.float64).reshape(len(tags), len(stats.genes)))


def _response_axes(responses: np.ndarray, k: int) -> np.ndarray:
    """Top ``k`` principal axes ``(k, genes)`` of the responses: float64 on the CPU, no sampling.

    Rows >= genes: eigenvectors of the Gram matrix C.T @ C, the thin SVD's basis at ~1/100 the
    cost for 2000 genes. Float64 is load-bearing: a float32 Gram rotated the top 100 axes enough
    to change a close neighbour. Each axis is signed so its largest-magnitude entry is positive,
    then rounded to float32, as the published deltas.npz were built.
    """
    x = np.asarray(responses, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError(f"responses have non-finite values; shape {x.shape}")
    c = torch.as_tensor(x - x.mean(axis=0), dtype=torch.float64)
    if c.shape[0] >= c.shape[1]:
        comps = torch.linalg.eigh(c.T @ c)[1][:, -k:].flip(1)      # eigh ascends: largest axis first
    else:
        comps = torch.linalg.svd(c, full_matrices=False)[2][:k].T
    sign = comps.gather(0, comps.abs().argmax(dim=0, keepdim=True)).sign()
    sign[sign == 0] = 1
    return (comps * sign).T.numpy().astype(np.float32)


def build_deltas(stats: ConditionStats, train: pd.DataFrame, controls: pd.DataFrame, roles: BaselineConditionRoles,
                 out_path: Path, half: Optional[int] = None) -> ArmSummaries:
    """``response_deltas`` on the top ``PCA_DIM`` principal axes of those same responses, saved.

    The axes are fit on every train response, not on single cells (whose top axes are line
    identity and depth). The projection skips the fit's mean, so a zero response stays exactly
    zero, as PCA(arm) - PCA(control) always did.
    """
    gene = response_deltas(stats, train, controls, roles, half)
    axes = _response_axes(gene.deltas, min(PCA_DIM, *gene.deltas.shape))
    z = (gene.deltas @ axes.T.astype(np.float64)).astype(np.float32)
    tags = np.empty((len(gene.tags), 2), dtype=object)
    for i, (ctx, pert) in enumerate(gene.tags):
        tags[i, 0], tags[i, 1] = ctx, pert
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, deltas=z, tags=tags)
    log.info("deltas %s (PCA-%d of %d responses) -> %s", z.shape, len(axes), len(z), out_path)
    return ArmSummaries(gene.tags, z)


def assert_no_held_arms(summaries: ArmSummaries, held: Iterable[tuple]) -> None:
    """A deltas file built on another split can hold this split's test arms; ranking on them leaks."""
    leaked = set(summaries.tags) & set(held)
    if leaked:
        raise ValueError(f"{len(leaked)} held-out arms have deltas (built on another split?), e.g. {sorted(leaked)[:3]}")


# Find the neighbour: the train context whose responses point the same way

def profile_index(summaries: ArmSummaries) -> dict[tuple, dict[tuple, int]]:
    """context -> {perturbation -> row of ``summaries.deltas``}: each context's response profile."""
    out: dict[tuple, dict[tuple, int]] = {}
    for i, (ctx, pert) in enumerate(summaries.tags):
        out.setdefault(ctx, {})[pert] = i
    return out


def mean_cosine(summaries: ArmSummaries, own: Mapping, mates: Mapping) -> Optional[tuple[float, int]]:
    """(mean 1 - cosine, n shared) between two contexts' deltas over the perturbations both
    have; rows with a zero delta are skipped. None when nothing is shared."""
    shared = sorted(own.keys() & mates.keys())
    if not shared:
        return None
    a = summaries.deltas[[mates[m] for m in shared]].astype(np.float64)
    b = summaries.deltas[[own[m] for m in shared]].astype(np.float64)
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    ok = (na > 0) & (nb > 0)
    if not ok.any():
        return None
    return float(np.mean(1.0 - np.sum(a[ok] * b[ok], axis=1) / (na[ok] * nb[ok]))), len(shared)


def distance_table(summaries: ArmSummaries) -> dict[tuple, dict[tuple, tuple[float, int]]]:
    """context -> {candidate -> (mean 1 - cosine, n shared)} over every candidate of its group.

    A candidate shares every context axis after the first and differs on the first. The
    distance is a function of the two contexts alone (the test perturbation has no delta), so
    it is computed once here instead of once per test condition. Pairs sharing no non-zero
    delta are absent.
    """
    profiles = profile_index(summaries)
    groups: dict[tuple, list[tuple]] = {}
    for ctx in profiles:
        groups.setdefault(ctx[1:], []).append(ctx)
    table = {}
    for ctx, own in profiles.items():
        pairs = ((c, mean_cosine(summaries, own, profiles[c])) for c in groups[ctx[1:]] if c != ctx)
        table[ctx] = {c: got for c, got in pairs if got is not None}
    return table


def line_neighbors(summaries: ArmSummaries, held: list[tuple], plotted: Optional[list[tuple]] = None) -> pd.DataFrame:
    """ONE closest train context per held-out context, for every perturbation at once.

    The per-condition rule (``closest_by_cosine``) re-picks the neighbour for each test
    perturbation; a figure that copies one line for every drug needs one neighbour per line,
    over all the perturbations both have in training. Held-out contexts are never candidates.
    ``plotted``: the perturbations a figure draws; ``frac_plotted_drugs_in_neighbor_train`` is
    the share the neighbour has in training (a copy of an unmeasured arm is no copy).
    """
    profiles, table = profile_index(summaries), distance_table(summaries)
    held_set = set(held)
    rows = []
    for ctx in held:
        own = profiles.get(ctx, {})
        cands = sorted((d, n, c) for c, (d, n) in table.get(ctx, {}).items() if c not in held_set)
        if len(cands) < 2:
            raise ValueError(f"{ctx}: {len(cands)} candidate contexts share a perturbation; need a neighbour and a runner-up")
        (d1, n1, c1), (d2, _n2, c2) = cands[:2]
        row = {"context": ctx, "neighbor": c1[0], "distance": round(d1, 4), "n_shared_arms": n1,
               "runner_up": c2[0], "runner_up_distance": round(d2, 4), "n_own_train_arms": len(own)}
        if plotted is not None:
            row["frac_plotted_drugs_in_neighbor_train"] = round(sum(p in profiles[c1] for p in plotted) / len(plotted), 3)
        rows.append(row)
    return pd.DataFrame(rows)
