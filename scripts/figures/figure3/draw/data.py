"""Figure 3's numbers, read from core_deliverables/data/figure3/main/.

Panel a scores every downsampled GENESIS run on exactly Figure 2c's Tahoe-100M conditions; a point is the mean of
its 3 runs, and its 95% CI is Figure 2's cluster bootstrap. The arms Figure 2c also drew must reproduce Figure 2's
published CIs, so a stale copy of Figure 2's table cannot draw without a word. Panel b is a held-out cell line
that saw 0-164 perturbations in training, 3 entity seeds per point.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.figures import metrics
from src.stats.bootstrap import PRIMARY, UNITS, cluster_bootstrap_ci, condition_keys

# Short name (the scaling table's) -> (Figure 2's metric key, axis title, higher is better)
METRICS = {"deg_adaptivek_truth": (metrics.DEG, "Top-100 DEG overlap ↑", True),
           "pearson_r_diff": (metrics.PEARSON, "Pearson Δ ↑", True),
           "energy_distance": (metrics.ENERGY, "Energy distance ↓", False)}
BIO_LEVELS = (0, 15, 30, 80, 164)            # perturbations the held-out line saw; 0 = its controls only


# ----------------------------------------------------------------------------
# 1. Build panel a's tables: Figure 2c's conditions x (its arms + every downsampled point)
# ----------------------------------------------------------------------------

def load_runs(data: Path) -> pd.DataFrame:
    """One row per training run: its axis (conds or cells), % kept, stem and training-set size; 3 per point."""
    runs = pd.read_csv(data / "scaling_runs.csv")
    if (n := runs.groupby(["axis", "pct_kept"]).size()).ne(3).any():
        raise ValueError(f"expected 3 runs per point:\n{n}")
    if runs.n_ctrl.nunique() != 1:
        raise ValueError(f"runs train on different control sets: {sorted(runs.n_ctrl.unique())}")
    return runs


def pooled(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per point, stem pool_<axis><pct>: its training-set size, the mean over its runs."""
    arms = (runs.assign(n_pert=runs.n_train - runs.n_ctrl).groupby(["axis", "pct_kept"], sort=False)
            [["n_train", "n_pert", "n_ctrl", "n_conds"]].mean().reset_index())
    return arms.assign(stem="pool_" + arms.axis + arms.pct_kept.astype(str))


def on_figure2c(data: Path, runs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per metric, conditions x arms: Figure 2c's arms, then each point's mean over its runs. Figure 2's seed-64
    run is both its GENESIS column and the 100% point's first run. A hole anywhere raises."""
    pc = pd.read_parquet(data / "figure2" / "per_condition.parquet")
    pc = pc[pc.panel.eq("tahoe_bio")].assign(model=lambda d: d.model.replace(metrics.ARM_NAME))
    scaling = pd.read_parquet(data / "split_half" / "scaling_runs.parquet")
    column = {s: "GENESIS" if s == "genesis" else s for s in runs.stem}
    blocks = {}
    for short, (key, _title, _hb) in METRICS.items():
        block = pc[pc.metric_key.eq(key)].pivot(index="condition", columns="model", values="value")
        if block.isna().any().any():
            raise RuntimeError(f"{short}: Figure 2c's table has a hole")
        got = scaling[scaling.metric.eq(short)].pivot(index="condition", columns="stem", values="value")
        block = block.join(got.reindex(block.index))
        if missing := [c for c in column.values() if c not in block or block[c].isna().any()]:
            raise RuntimeError(f"{short}: {missing} do not cover Figure 2c's {len(block)} conditions")
        for (axis, pct), g in runs.groupby(["axis", "pct_kept"]):
            block[f"pool_{axis}{pct}"] = block[[column[s] for s in g.stem]].to_numpy(float).mean(axis=1)
        blocks[short] = block
    return blocks


def cis(block: pd.DataFrame, cols: list[str], key: str, published: pd.DataFrame | None) -> dict[str, tuple]:
    """(mean, lo, hi) per column: Figure 2's cluster bootstrap over conditions (cell line x drug), one paired
    draw. Every column Figure 2c drew must equal Figure 2's confidence_intervals.csv, or these are not its
    conditions; None only for a block that is not Figure 2c's full condition set."""
    cl = condition_keys(block.index.to_series())[UNITS[PRIMARY["tahoe"]]].agg("|".join, axis=1).to_numpy()
    mean, lo, hi = cluster_bootstrap_ci(block[cols].to_numpy(float), cl)
    out = {c: (float(m), float(a), float(b)) for c, m, a, b in zip(cols, mean, lo, hi)}
    if published is None:
        return out
    pub = published[published.panel.eq("c_tahoe_bio") & published.metric.eq(key) & published.drawn].set_index("arm")
    for c in (c for c in cols if c in pub.index):
        if not np.allclose(out[c], pub.loc[c, ["mean", "lo", "hi"]].to_numpy(float), rtol=0, atol=1e-9):
            raise RuntimeError(f"{key} {c}: CI {out[c]} != Figure 2c {tuple(pub.loc[c, ['mean', 'lo', 'hi']])}")
    return out


# ----------------------------------------------------------------------------
# 2. Build panel b's table: 5 held-out lines x 5 overlap levels x 3 entity seeds
# ----------------------------------------------------------------------------

def held_out_lines(data: Path, key: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Seed mean and s.d. per (held-out line, overlap level): arrays of 5 x 5, and the line names."""
    t = pd.read_csv(data / "overlap_bio" / "cell_seed_overlap_strict_drug.csv")
    t = t[t.metric_key.eq(key)]
    if len(t) != 5 * len(BIO_LEVELS) * 3 or t.metric_mean.isna().any():
        raise ValueError(f"{key}: expected a full 5 x {len(BIO_LEVELS)} x 3 table, got {len(t)} rows")
    g = t.groupby(["cell_line", "overlap"])["metric_mean"]
    mean, sd = g.mean().unstack()[list(BIO_LEVELS)], g.std(ddof=1).unstack()[list(BIO_LEVELS)]
    return mean.to_numpy(), sd.to_numpy(), list(mean.index)
