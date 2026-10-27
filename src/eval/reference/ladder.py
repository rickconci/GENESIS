"""Split-half against the experimental repeat, as a function of cells per arm (Figure 1d).

Two agreement curves for one condition at the same depth:

  split_half   both arms are disjoint halves of one replicate's cells: sampling noise only,
               the split-half reference as the paper defines it
  cross_plate  one arm from each of two replicates of the condition: sampling noise plus
               whatever differs between two runs of the experiment

At matched depth split-half agrees more; the gap between the curves is experimental noise.
Both curves call one kernel (``score_raw_pair``), which is what makes them comparable. Each
arm is n perturbed and n control cells, controls matched to their replicate; which replicate
supplies the split-half arm is drawn per repeat, so neither biases the curve.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.eval.reference.pairs import score_raw_pair

LADDER = (50, 100, 200, 400, 800, 1600)  # cells per arm
STRATEGIES = ("split_half", "cross_plate")


def _shuffled(x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    return x[torch.as_tensor(rng.permutation(int(x.shape[0])), device=x.device, dtype=torch.long)]


def score_condition(*, truth: dict[Any, torch.Tensor], ctrl: dict[Any, torch.Tensor], plate_i: Any,
                    plate_j: Any, n_repeats: int, seed: int) -> list[dict[str, Any]]:
    """Both strategies at every rung, ``n_repeats`` draws each; every metric is kept."""
    rng = np.random.default_rng(seed)
    out: list[dict[str, Any]] = []
    for rep in range(n_repeats):
        # fresh draws per repeat; nested prefixes within one, so the ladder is monotone in one sample
        t = {p: _shuffled(truth[p], rng) for p in (plate_i, plate_j)}
        c = {p: _shuffled(ctrl[p], rng) for p in (plate_i, plate_j)}
        sh = plate_i if rng.random() < 0.5 else plate_j
        for n in LADDER:
            res = {"split_half": score_raw_pair(t[sh][:n], t[sh][n:2 * n], c[sh][:n], c[sh][n:2 * n]),
                   "cross_plate": score_raw_pair(t[plate_i][:n], t[plate_j][:n], c[plate_i][:n], c[plate_j][:n])}
            out += [{"strategy": k, "n_cells": n, "rep": rep, "sh_plate": sh, **v} for k, v in res.items()]
    return out
