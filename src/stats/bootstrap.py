"""95% cluster-bootstrap confidence intervals of a plotted mean, paired across arms.

The CI is the percentile interval of the PLOTTED mean: the plain mean over the panel's units,
exactly what the bar or line draws. Each resample draws clusters with replacement, and a
cluster's units always come together, so the plate pairs or conditions of one cell line x
compound (or one donor x cytokine) count as the one sample they are. Within a panel and metric
every arm shares every draw: the intervals are paired.

Clusters, by dataset:
  tahoe  cell line x drug, doses pooled (drawn); cell line x drug x dose (sensitivity, not drawn)
  parse  donor x cytokine, so one sample's cell types and wells move together

A figure's driver chooses its units (plate pairs, conditions). The table drivers (Fig. 2, ED02, ED14)
call ``ci_rows`` once per panel, metric and cluster unit, and the figure only draws the table those
rows make; the others (Fig. 3a, Fig. 4, ED12) call ``cluster_bootstrap_ci`` with their own labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

B, SEED, LEVEL = 10_000, 20261014, 0.95
CHUNK = 500                                    # resamples per matrix product, to bound memory
# unit name -> condition keys that make one cluster
UNITS = {"line_x_drug": ["cell_name", "drug"], "line_x_drug_x_dose": ["cell_name", "drug", "dose"],
         "donor_x_cytokine": ["donor", "cytokine"]}
PRIMARY = {"tahoe": "line_x_drug", "parse": "donor_x_cytokine"}   # the unit the figures draw
DATASET_UNITS = {"tahoe": ["line_x_drug", "line_x_drug_x_dose"], "parse": ["donor_x_cytokine"]}  # CSV order


# One estimator: resample whole clusters, read the mean's percentiles
def cluster_bootstrap_ci(values: np.ndarray, clusters: np.ndarray, *, b: int = B, seed: int = SEED,
                         level: float = LEVEL) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Percentile CI of the unit mean, resampling whole clusters; paired across columns.

    values: (n_units,) or (n_units, k). Each column is one quantity on the same units
        (e.g. the repeat and GENESIS); every resample uses one cluster draw for all columns.
    clusters: (n_units,) labels. A drawn cluster brings all of its units.
    Returns (mean, lo, hi), each (k,). mean is the plain mean over units (the plotted
    number); a resample's mean is its drawn units' sum over their count, so clusters of
    different sizes weigh as their units do.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[:, None] if v.ndim == 1 else v
    if len(v) != len(clusters) or len(v) == 0:
        raise ValueError(f"{len(v)} units but {len(clusters)} cluster labels")
    if not np.isfinite(v).all():
        raise ValueError("non-finite values: the plotted set should already exclude them")
    _codes, inv = np.unique(np.asarray(clusters), return_inverse=True)
    n_c = int(inv.max()) + 1
    if n_c < 2:   # every resample would be the sample itself: a zero-width interval that reads as certainty
        raise ValueError(f"{n_c} cluster: a CI needs at least 2 clusters")
    sums = np.zeros((n_c, v.shape[1]))
    np.add.at(sums, inv, v)
    counts = np.bincount(inv, minlength=n_c).astype(np.float64)
    rng, boot = np.random.default_rng(seed), np.empty((b, v.shape[1]))
    for s in range(0, b, CHUNK):
        draws = rng.integers(0, n_c, size=(min(CHUNK, b - s), n_c))
        w = np.stack([np.bincount(d, minlength=n_c) for d in draws]).astype(np.float64)
        boot[s:s + len(draws)] = (w @ sums) / (w @ counts)[:, None]
    tail = 100 * (1 - level) / 2
    lo, hi = np.percentile(boot, [tail, 100 - tail], axis=0)
    return v.mean(axis=0), lo, hi


def cluster_draws(n_clusters: int, *, b: int = B, seed: int = SEED) -> np.ndarray:
    """(b, n_clusters): how often each cluster is drawn in each resample, the very draws cluster_bootstrap_ci makes
    for that many clusters. For a plotted statistic that is not the plain unit mean (a mean of per-line means, a
    correlation recomputed from the drawn units), so it is resampled exactly as Figure 2's means are; its interval
    is then np.percentile over the resamples, as there."""
    if n_clusters < 2:
        raise ValueError(f"{n_clusters} cluster: a CI needs at least 2 clusters")
    rng, out = np.random.default_rng(seed), np.empty((b, n_clusters))
    for s in range(0, b, CHUNK):
        draws = rng.integers(0, n_clusters, size=(min(CHUNK, b - s), n_clusters))
        out[s:s + len(draws)] = np.stack([np.bincount(d, minlength=n_clusters) for d in draws])
    return out


def condition_keys(conditions: pd.Series) -> pd.DataFrame:
    """'k=v|k=v' condition labels (scripts/figures/figure2/main/figure2_extract.py's format) -> one column per key, same index."""
    rows = [dict(kv.split("=", 1) for kv in c.split("|")) for c in conditions]
    if any(len(r) != c.count("|") + 1 for r, c in zip(rows, conditions)):
        raise ValueError("a condition label does not parse as k=v|k=v")
    return pd.DataFrame(rows, index=conditions.index)


# One CI table: a row per (panel, metric, arm, cluster unit), the columns figures read
def ci_rows(block: pd.DataFrame, keys: pd.DataFrame, unit: str, *, panel: str, metric: str,
            dataset: str) -> list[dict]:
    """One panel's CIs for one metric and cluster unit: a row per arm (block's columns, one shared draw).

    block: one row per plotted unit (a plate pair, a condition), one column per arm.
    keys: the same rows, holding UNITS[unit]'s columns; a cluster is their str values joined by '|'.
    """
    if len(keys) != len(block):
        raise ValueError(f"{panel} {metric}: {len(keys)} key rows for {len(block)} plotted rows")
    if not keys.index.equals(block.index):   # rows pair by label: a reordered frame would move every CI
        raise ValueError(f"{panel} {metric}: keys and plotted rows have a different index (order or labels)")
    if missing := [c for c in UNITS[unit] if c not in keys.columns]:
        raise ValueError(f"{panel} {metric}: keys have no {missing} for {unit}")
    if gaps := [c for c in UNITS[unit] if keys[c].isna().any()]:   # str(NaN) would be a cluster named "nan"
        raise ValueError(f"{panel} {metric}: {gaps} missing on some rows for {unit}")
    cl = keys[UNITS[unit]].astype(str).agg("|".join, axis=1).to_numpy()
    try:
        mean, lo, hi = cluster_bootstrap_ci(block.to_numpy(dtype=float), cl)
    except ValueError as e:                  # say which bar: a rebuild has dozens of these
        raise ValueError(f"{panel} {metric} {unit}: {e}") from e
    return [dict(panel=panel, metric=metric, arm=arm, unit=unit, drawn=unit == PRIMARY[dataset],
                 mean=m, lo=a, hi=z, n_units=len(set(cl)), n_obs=len(block), B=B, seed=SEED, level=LEVEL)
            for arm, m, a, z in zip(block.columns, mean, lo, hi)]
