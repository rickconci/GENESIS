"""Context mean: the context's average expression over its train perturbations.

The value of (h, k, r) is the equal-weight mean of the train arm means in that context and
stratum. Equal weight, not cell-count weight: it matches how the perturbation mean pools arms
and keeps deep-sequenced arms from dominating the null. It is absolute expression and ignores
the perturbation: how far a model gets from "this context, any perturbation". Tahoe's value is
per (cell line, plate); PBMC's per (donor, cell type).

Formulation: bioRxiv 10.1101/2025.06.26.661135 (page 37).
"""
from __future__ import annotations

import numpy as np

from src.baselines.roles import Arms, group_mean


def fit(arms: Arms) -> dict[tuple, np.ndarray]:
    """Mean expression per (held..., kept..., stratum...) key."""
    return group_mean(arms.mean, [a.held + a.kept + a.stratum for a in arms.keys])
