"""What to generate, and how to score it.

Generation is the same for every design: one cloud of ``n_truth`` cells per
(condition, unit, rep), with that unit's conditioning. Designs differ only in
scoring. A cloud request carries its own seed; two designs asking for the same
(row, rep, seed) share one cloud, and two seed schemes for one plate stay two clouds.

    SplitHalf       halves truth and controls; rep r's cloud, indexed by the B half,
                    against truth half A (arms N/2). No clouds = the split-half reference.
    ReplicatePairs  splits nothing: per pair of units, true_true A vs B, same A vs A',
                    cross A vs B', gen_gen A' vs B' (the published pair-table row).

``requests`` sees only the :class:`GroupRef` (cell counts, no arrays), so a live source
can lay every cloud of a run into one batched stream before the first group is read.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

import torch

from src.eval.reference.truth_io import ContextGroup, GroupRef


@dataclass(frozen=True)
class CloudRequest:
    row: Mapping[str, str]
    n_cells: int
    rep: int
    seed: int

    @property
    def key(self) -> tuple:
        """Dedup key: the row (any key order), the rep and the seed."""
        return (tuple(sorted((str(k), str(v)) for k, v in self.row.items())), int(self.rep), int(self.seed))


def dedupe(requests: Sequence[CloudRequest]) -> list[CloudRequest]:
    seen: dict[tuple, CloudRequest] = {}
    for req in requests:
        if seen.setdefault(req.key, req).n_cells != req.n_cells:
            raise ValueError(f"two requests for one cloud disagree on its size: {req.key}")
    return list(seen.values())


def _on(x: Any, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


class SplitHalf:
    """Seeds as model eval: the draw is ``manual_seed(stable_cond_seed + rep)`` inside the
    sampler; the cloud is ``condition_noise_seed(seed, keys, rep)``."""

    name = "split_half"

    def __init__(self, *, plan: Any, seed: int, output_space: Any, device: torch.device,
                 live: bool = True) -> None:
        if int(plan.k) < 1:
            raise ValueError(f"split-half repeats must be >= 1; got {plan.k}")
        self.plan, self.seed, self.output_space, self.device = plan, int(seed), output_space, device
        self.live, self.skips = bool(live), Counter()

    def _request(self, ref: GroupRef, i: int, rep: int) -> CloudRequest:
        from src.eval.orchestration.pregeneration import condition_noise_seed
        keys = ref.unit_keys[i]
        return CloudRequest(keys, ref.n_cells[i], rep, condition_noise_seed(self.seed, keys, rep))

    def requests(self, ref: GroupRef) -> list[CloudRequest]:
        if not self.live:
            return []
        return [self._request(ref, i, rep) for i in range(len(ref.units)) for rep in range(self.plan.k)]

    def score(self, group: ContextGroup, clouds: Mapping[tuple, Any]) -> list[dict]:
        """One metrics row per unit, the schema of metrics_summary.json / noise_ceiling.json."""
        from src.eval.orchestration.model_eval import (
            _condition_eligible_for_kreps, _score_condition_kreps, build_result_row)

        rows = []
        for i, unit in enumerate(group.units):
            truth, ctrl = _on(unit.truth, self.device), _on(unit.control, self.device)
            if not _condition_eligible_for_kreps(self.plan, truth, ctrl):
                self.skips["insufficient_cells"] += 1
                continue
            pred = (torch.cat([_on(clouds[self._request(group.ref, i, r).key], self.device)
                               for r in range(self.plan.k)]) if self.live else None)
            rep = _score_condition_kreps(
                plan=self.plan, truth_t=truth, ctrl_t=ctrl, pred_t_all=pred, k_eff=self.plan.k,
                output_space=self.output_space, seed=self.seed, key_dict=dict(unit.keys),
                first_condition_verbose=False)
            n_truth = int(truth.shape[0])
            rows.append(build_result_row(
                key_dict=dict(unit.keys), n_pred=n_truth if pred is None else int(pred.shape[0]),
                n_truth=n_truth, n_ctrl=int(ctrl.shape[0]), metrics=rep.means, metrics_std=rep.stds,
                k=rep.k_effective))
        return rows


class ReplicatePairs:
    """One full-plate cloud per unit, seeded ``replicate_k_seed`` (draw 0 = generate.py's files)."""

    name = "replicate_pairs"

    def __init__(self, design: Any, *, model: str, report_space: str, device: torch.device,
                 ref: Any = None, draw: int = 0) -> None:
        self.design, self.model, self.report_space = design, model, report_space
        self.device, self.ref, self.draw, self.skips = device, ref, int(draw), Counter()

    def _request(self, ref: GroupRef, i: int) -> CloudRequest:
        from src.eval.reference.generation import replicate_k_seed
        seed = replicate_k_seed(self.design, ref.values, ref.units[i], self.draw)
        return CloudRequest(ref.unit_keys[i], ref.n_cells[i], self.draw, seed)

    def requests(self, ref: GroupRef) -> list[CloudRequest]:
        return [self._request(ref, i) for i in range(len(ref.units))] if len(ref.units) > 1 else []

    def score(self, group: ContextGroup, clouds: Mapping[tuple, Any]) -> list[dict]:
        """One pair-table row per unordered pair of units, in the units' order."""
        from src.eval.reference.pairs import score_loaded_pair

        if len(group.units) < 2:
            self.skips["single_unit"] += 1
            return []
        truth = {u.unit: _on(u.truth, self.device) for u in group.units}
        ctrl = {u.unit: _on(u.control, self.device) for u in group.units}
        gen = {u.unit: _on(clouds[self._request(group.ref, i).key], self.device)
               for i, u in enumerate(group.units)}
        return [score_loaded_pair(self.design, dict(group.ref.values), a.unit, b.unit, truth, ctrl, gen,
                                  model=self.model, fmt="npy", report_space=self.report_space, ref=self.ref)
                for a, b in combinations(group.units, 2)]
