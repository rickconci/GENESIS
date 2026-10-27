"""Nearest-neighbour arm: the neighbour context's raw cells for the test perturbation.

Which train context is the neighbour is the baseline itself (src/baselines/nearest_neighbor.py:
``deltas.npz`` and ``distance_table``). This arm only copies: per queue row it looks up the
neighbour, takes its cells for the same perturbation (on the truth's plate when
``plate_matched``), subsamples to the truth's cell count, and records the neighbour,
runner-up and margin of every condition.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from src.baselines.nearest_neighbor import ArmSummaries, _row_keys, distance_table, tag_of
from src.baselines.roles import BaselineConditionRoles
from src.eval.cache.eval_cache_io import QuadrupletRow
from src.eval.cache.eval_npz_reader import EvalNpzReader

log = logging.getLogger(__name__)

METRICS = ("cosine",)


# The copy: where a train arm's cells are, and which ones to take

def _str_keys(keys: Mapping) -> tuple:
    """Condition keys as sorted string pairs: one lookup key for the queue row and the scored bundle."""
    return tuple(sorted((str(k), str(v)) for k, v in keys.items()))


class CopySource:
    """Where an arm's cells are: the cache keys of its train rows, in summary order.

    The copy is the first such row, or with ``plate`` the row on that plate: copying a
    neighbour from another plate than the truth's mixes in batch effect.
    """

    def __init__(self, train: pd.DataFrame, roles: BaselineConditionRoles) -> None:
        self._roles, self._first, self._by_plate = roles, {}, {}
        for keys in _row_keys(train, roles.condition_variables):
            tag = tag_of(keys, roles)
            self._first.setdefault(tag, keys)
            if roles.plate_var:
                self._by_plate.setdefault((*tag, keys[roles.plate_var]), keys)

    def keys(self, context: tuple, pert: tuple, plate: Optional[str] = None) -> Optional[dict]:
        return self._first.get((context, pert)) if plate is None else self._by_plate.get((context, pert, plate))


class CopyNeighborPredGetter:
    """``pred_getter``: the neighbour context's raw cells for the test perturbation.

    Controls (the published default, ``own_controls=False``): the copied cells are scored
    against the HELD-OUT context's controls, like every model. log2(nb_pert / heldout_ctrl)
    = log2(nb_pert / nb_ctrl) + log2(nb_ctrl / heldout_ctrl): the second term is a
    drug-independent basal offset between the two contexts, so this copies the neighbour's
    cells, not its response. ``own_controls=True`` scores them against the neighbour's own
    controls (same plate, same cell type) and copies the response: on Tahoe bio it moves
    Fig. 2c's Pearson Δ 0.21 -> 0.38 and DEG 0.17 -> 0.26, energy unchanged. The paper uses
    the default.
    """

    def __init__(self, metric: str, *, summaries: ArmSummaries, roles: BaselineConditionRoles,
                 source: CopySource, npz_reader: EvalNpzReader, plate_matched: bool,
                 own_controls: bool = False) -> None:
        if metric not in METRICS:
            raise ValueError(f"metric {metric!r} not in {METRICS}")
        if plate_matched and not roles.plate_var:
            raise ValueError(f"{roles.dataset_alias} has no plate column to match on")
        if own_controls and roles.plate_var and not plate_matched:
            raise ValueError("own_controls reads the neighbour's controls on the truth's plate: needs plate_matched")
        self._metric, self._roles = metric, roles
        self._source, self._reader, self._plate_matched = source, npz_reader, plate_matched
        self.own_controls = own_controls
        self._neighbor_of: dict[tuple, tuple] = {}     # str condition keys -> neighbour context
        self._table = distance_table(summaries)
        self._holders: dict[tuple, list[tuple]] = {}   # (group, perturbation) -> contexts with that train arm, sorted
        for ctx, pert in sorted(summaries.tags):
            self._holders.setdefault((ctx[1:], pert), []).append(ctx)
        self.choices: dict[tuple, dict] = {}            # condition -> pick() + copied keys
        self.misses: dict[str, int] = {}
        self._lock = threading.Lock()

    def _miss(self, why: str, keys: Mapping) -> None:
        log.warning("copy %s: %s for %s", self._metric, why, dict(keys))
        with self._lock:
            self.misses[why] = self.misses.get(why, 0) + 1

    def pick(self, tag) -> Optional[dict]:
        """The closest context entitled to the test perturbation (it has that arm in training),
        its runner-up and the margin between them. Ties go to the first context in sorted order."""
        ctx, pert = tag
        row = self._table.get(ctx, {})
        ranked = sorted((row[c][0], i, c) for i, c in enumerate(self._holders.get((ctx[1:], pert), []))
                        if c != ctx and c in row)
        if not ranked:
            return None
        (d1, _, nb), second = ranked[0], ranked[1] if len(ranked) > 1 else None
        return {"neighbor": nb, "distance": d1, "n_shared": row[nb][1],
                "runner_up": second[2] if second else None, "margin": second[0] - d1 if second else float("nan")}

    def __call__(self, qrow: QuadrupletRow) -> Optional[np.ndarray]:
        keys, n = qrow.condition_keys, int(qrow.num_cells)
        tag = tag_of(keys, self._roles)
        got = self.pick(tag)
        if got is None:
            self._miss("no_neighbor", keys)
            return None
        nb = got["neighbor"]
        if nb == tag[0] or nb[1:] != tag[0][1:]:
            raise RuntimeError(f"{self._metric} neighbour {nb} breaks the context rule for {tag[0]}")
        plate = str(keys[self._roles.plate_var]) if self._plate_matched else None
        src = self._source.keys(nb, tag[1], plate)
        if src is None:
            self._miss("no_arm" if plate is None else "no_arm_on_plate", keys)
            return None
        pred = np.asarray(self._reader.perturbed(src, max_cells=n, seed=0), dtype=np.float32)
        if pred.shape[0] == 0:
            self._miss("empty_arm", keys)
            return None
        if pred.shape[0] < n:   # the scorer reads pred[:n_truth]; resample rather than read past the end
            pred = pred[np.random.default_rng(0).choice(pred.shape[0], size=n, replace=True)]
        if not np.isfinite(pred).all():
            raise RuntimeError(f"non-finite copied cells {src} for {dict(keys)}")
        with self._lock:
            self.choices[tuple(sorted(keys.items()))] = {**got, "copied": src}
            self._neighbor_of[_str_keys(keys)] = nb
        return pred

    def neighbor_control_keys(self, keys: Mapping, ctrl_keys: list[dict]) -> list[dict]:
        """The held-out condition's control keys with its context swapped for the neighbour's
        (``own_controls``). Raises when nothing was copied for ``keys``: no silent fallback."""
        nb = self._neighbor_of.get(_str_keys(keys))
        if nb is None:
            raise KeyError(f"no copied neighbour recorded for {dict(keys)}")
        swap = dict(zip(self._roles.context_vars, nb))
        return [{**k, **swap} for k in ctrl_keys]

    @property
    def miss_count(self) -> int:
        return sum(self.misses.values())

    def choices_frame(self) -> pd.DataFrame:
        ctx_vars = self._roles.context_vars
        rows = [{**dict(k), **{f"neighbor_{v}": x for v, x in zip(ctx_vars, c["neighbor"])},
                 "distance": c["distance"], **{f"copied_{k2}": v2 for k2, v2 in c["copied"].items()},
                 **{f"runner_up_{v}": x for v, x in zip(ctx_vars, c["runner_up"] or (None,) * len(ctx_vars))},
                 "margin": c["margin"], "n_shared": c["n_shared"]}
                for k, c in self.choices.items()]
        return pd.DataFrame(rows).sort_values(list(self._roles.condition_variables), kind="stable")


def check_expected_neighbors(choices: pd.DataFrame, expected_tsv: Path, roles: BaselineConditionRoles) -> None:
    """Every scored context must copy the one neighbour a figure cites (TSV: <context>, neighbor).
    Guards against bars and figures quietly copying different contexts."""
    if len(roles.context_vars) != 1:
        raise SystemExit("--expect-neighbors supports one context axis")
    ctx = roles.context_vars[0]
    cited = pd.read_csv(expected_tsv, sep="\t").set_index(ctx)["neighbor"].astype(str).to_dict()
    got = choices.groupby(ctx)[f"neighbor_{ctx}"].agg(lambda s: sorted(set(s)))
    bad = {c: (nbs, cited.get(c)) for c, nbs in got.items() if nbs != [cited.get(c)]}
    if bad:
        raise SystemExit(f"neighbours differ from {expected_tsv}: {bad}")
    unseen = sorted(set(cited) - set(got.index))
    log.info("neighbours match %s on %d contexts%s", expected_tsv.name, len(got),
             f" ({len(unseen)} cited contexts not scored here)" if unseen else "")
