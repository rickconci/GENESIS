"""Perturbation mean: the matched control plus the perturbation's mean shift over held contexts.

For a train arm a = (h, k, r, p), delta_a = mean_a - control(h, k, r). The shift of (k, p) is
the mean of delta_a over every train arm with that (k, p), one vote per arm whatever its cell
count, so deep-sequenced arms do not dominate. Tahoe (no kept axis) averages over cell lines
and plates; PBMC (kept: cell type) over donors. A held-out (h, k, r, p) is predicted as its
own control + shift(k, p): the model knows the perturbation, never the interaction.
"""
from __future__ import annotations

import numpy as np

from src.baselines.roles import Arms, group_mean


def fit(arms: Arms) -> dict[tuple, np.ndarray]:
    """Shift per (kept..., perturbation...) key."""
    return group_mean(arms.delta, [a.kept + a.pert for a in arms.keys])
