"""Additive: the matched control plus one main effect per axis, no interactions.

delta_hat = mu + e_held(h) + e_k1(k1) + ... + e_pert(p), fit as sequential means on per-arm
deltas in that order, each on the residual of the ones before: the gene-space cousin of
hallmark C3. Context goes before the perturbation because a held-out context still has other
perturbations in train, while the held-out perturbation is only seen in other contexts.
Tahoe: mu + alpha(cell line) + beta(drug, dose). PBMC: mu + alpha(donor) + gamma(cell type)
+ beta(cytokine). Each arm has weight 1, whatever its cell count.

An effect averages to zero over the train arms, so a level never seen in train contributes
zero: the average level. Figure 4 relies on this for cell lines held out of every training
drug (prediction mu + beta). The prediction is one (G,) mean; the per-cell cloud is the
shared ``emit_per_cell_delta`` shift of real control cells (src/eval/arms/baseline.py).
"""
from __future__ import annotations

import numpy as np

from src.baselines.roles import ArmKey, Arms, group_mean


def terms(key: ArmKey) -> list[tuple]:
    """The level of each axis, in fit order: held, each kept column, perturbation."""
    return [key.held, *((v,) for v in key.kept), key.pert]


def term_names(kept_vars: list[str]) -> list[str]:
    return ["held", *(f"kept__{c}" for c in kept_vars), "pert"]


def fit(arms: Arms) -> tuple[np.ndarray, list[dict[tuple, np.ndarray]]]:
    """mu, then one effect table per term of :func:`terms`."""
    mu = arms.delta.mean(axis=0)
    resid = arms.delta - mu
    levels = list(zip(*(terms(k) for k in arms.keys)))
    effects = []
    for i, labels in enumerate(levels):
        effect = group_mean(resid, labels)
        effects.append(effect)
        if i < len(levels) - 1:
            resid = resid - np.stack([effect[label] for label in labels])
    return mu, effects
