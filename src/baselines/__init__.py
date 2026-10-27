"""Baselines: the simple models GENESIS is compared against, and how each is built.

One file per baseline, each defining it for both axis layouts (two-factor Tahoe, three-factor PBMC):

  perturbation_mean.py  matched control + the perturbation's mean shift over contexts
  context_mean.py       the context's mean expression over its train perturbations
  linear_ridge.py       the perturbation-mean table, ridge-fit in a gene PCA basis
  additive.py           matched control + one main effect per context axis + the perturbation's
  nearest_neighbor.py   the closest train context's cells (neighbour from condition_stats.py)

roles.py says which dataset column plays which role (from core_util_data/datasets/*.yaml),
fit.py fits the four linear baselines for one split, artifacts.py saves and loads them.
Entry point: scripts/baselines/fit.py. Scoring is not here: src/eval/arms/ turns an
artifact (or the neighbour) into cells, and src/eval/scoring.py scores them like any model.
"""
