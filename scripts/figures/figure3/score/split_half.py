"""Figure 3a on the split-half: each downsampled arm scored on exactly Figure 2c's Tahoe-100M conditions, and
checked against Figure 2c's published numbers, so the variants cannot drift from that figure. Shared by the
split-half variant, the uniform-draw variant and the per-run statistics. Every loader takes the bundle's
data/ (Figure 2c's inputs and the split-half scores are copied there by Figure 3's extract).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.figures.figure3.score.repeat_pairs import COND_NAME, COND_SPACE
from src.stats.bootstrap import PRIMARY, UNITS, cluster_bootstrap_ci, condition_keys

# short name -> (Figure 2c's metric key, panel title, higher is better)
SPLIT_METRICS = {
    "deg_adaptivek_truth": ("deg_raw_int_rgs/deg_overlap_top_p_ranked_k100_fdr0p001_lfc0p1_adaptivek_truth_detfloor10",
                            "Top-100 DEG overlap ↑", True),
    "pearson_r_diff": (f"{COND_SPACE}/pearson_r_diff", "Pearson Δ ↑", True),
    "energy_distance": (f"{COND_SPACE}/energy_distance", "Energy distance ↓", False),
}
SHIFT_MAX = 0.005      # K=3 may stand in for K=1 only if it moves an arm's condition-mean by less than this


def load_figure2c(data: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Figure 2c's Tahoe-100M per-condition scores, its ceiling, and the means this panel must reproduce."""
    pc = pd.read_parquet(data / "figure2" / "per_condition.parquet")
    pc = pc[pc.panel.eq("tahoe_bio")].copy()
    pc["model"] = pc["model"].replace(COND_NAME)
    ceil = pd.read_csv(data / "figure2" / "ceilings.csv")
    ceil = ceil[ceil.panel.eq("tahoe_bio") & ceil.report_space.eq(COND_SPACE)]
    drawn = pd.read_csv(data / "figure2" / "figure2c_source_data_plotted.csv")
    drawn = drawn[drawn.panel.eq("tahoe_bio")]
    return pc, ceil, drawn


def _figure2c_block(pc: pd.DataFrame, scaling: pd.DataFrame, drawn: pd.DataFrame, short: str) -> pd.DataFrame:
    """One metric: Figure 2c's arms (condition x arm) joined with the scaling arms on Figure 2c's conditions.
    Figure 2c's GENESIS mean and n must come out as published."""
    key = SPLIT_METRICS[short][0]
    wide = pc[pc.metric_key.eq(key)].pivot(index="condition", columns="model", values="value")
    if wide.isna().any().any():
        raise RuntimeError(f"{short}: Figure 2c's table has a hole")
    got = scaling[scaling.metric.eq(short)].pivot(index="condition", columns="stem", values="value").reindex(wide.index)
    block = wide.join(got)
    pub = drawn[drawn.metric_key.eq(key)].set_index("arm")
    if abs(block["GENESIS"].mean() - float(pub.loc["GENESIS", "mean"])) > 1e-5:
        raise RuntimeError(f"{short}: GENESIS mean {block['GENESIS'].mean()} != Figure 2c {pub.loc['GENESIS', 'mean']}")
    if len(block) != int(pub.loc["GENESIS", "n_conditions"]):
        raise RuntimeError(f"{short}: n={len(block)} != Figure 2c {int(pub.loc['GENESIS', 'n_conditions'])}")
    return block


def on_figure2c(pc: pd.DataFrame, scaling: pd.DataFrame, drawn: pd.DataFrame, ceil: pd.DataFrame,
                stems: list[str]) -> dict[str, pd.DataFrame]:
    """Each metric on exactly Figure 2c's conditions. A scaling arm missing any of them raises."""
    out = {}
    for short, (key, _title, _hb) in SPLIT_METRICS.items():
        block = _figure2c_block(pc, scaling, drawn, short)
        if missing := [s for s in stems if s not in block.columns or block[s].isna().any()]:
            raise RuntimeError(f"{short}: {missing} do not cover Figure 2c's {len(block)} conditions")
        pub = drawn[drawn.metric_key.eq(key)].set_index("arm")
        ceil_mean = float(ceil.set_index("metric_key").loc[key, "split_half_reference_mean"])
        if abs(ceil_mean - float(pub.loc["GENESIS", "split_half_reference_mean"])) > 1e-5:
            raise RuntimeError(f"{short}: ceiling {ceil_mean} != Figure 2c {pub.loc['GENESIS', 'split_half_reference_mean']}")
        out[short] = block
    return out


def on_figure2c_dropping(pc: pd.DataFrame, scaling: pd.DataFrame, drawn: pd.DataFrame, stems: list[str]
                         ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """(full Figure 2c block, plotted block) per metric. The plotted block drops a condition
    only when a scaling arm has no score for it, and only if that barely moves GENESIS."""
    full, plotted = {}, {}
    for short in SPLIT_METRICS:
        block = _figure2c_block(pc, scaling, drawn, short)
        if absent := [s for s in stems if s not in block.columns]:
            raise RuntimeError(f"{short}: {absent} have no column")
        keep = block.index[block[stems].notna().all(axis=1)]
        if len(keep) < len(block) - 5:
            raise RuntimeError(f"{short}: {len(block) - len(keep)} conditions missing from a scaling arm")
        sub = block.loc[keep]
        shift = float(sub["GENESIS"].mean()) - float(block["GENESIS"].mean())
        if abs(shift) >= 0.001:
            raise RuntimeError(f"{short}: dropping {len(block) - len(keep)} conditions moves GENESIS by {shift:+.4f}")
        if len(keep) != len(block):
            holes = {s: int(block[s].isna().sum()) for s in stems if block[s].isna().any()}
            print(f"{short}: plotting {len(keep)} of Figure 2c's {len(block)} ({holes}); GENESIS mean shifts {shift:+.6f}")
        full[short], plotted[short] = block, sub
    return full, plotted


def split_cis(block: pd.DataFrame, cols: list[str], key: str, published: pd.DataFrame | None = None
              ) -> dict[str, tuple[float, float, float]]:
    """(mean, lo, hi) per column: Figure 2's cluster bootstrap over the conditions (line x drug), paired
    across the columns. ``published`` (Figure 2's confidence_intervals.csv): the columns Figure 2c drew
    must match it, or the conditions are not Figure 2c's."""
    cl = condition_keys(block.index.to_series())[UNITS[PRIMARY["tahoe"]]].agg("|".join, axis=1).to_numpy()
    mean, lo, hi = cluster_bootstrap_ci(block[cols].to_numpy(float), cl)
    out = {c: (float(m), float(a), float(b)) for c, m, a, b in zip(cols, mean, lo, hi)}
    if published is not None:
        pub = published[published.panel.eq("c_tahoe_bio") & published.metric.eq(key) & published.drawn].set_index("arm")
        for c in (c for c in cols if c in pub.index):
            if not np.allclose(out[c], pub.loc[c, ["mean", "lo", "hi"]].to_numpy(float), rtol=0, atol=1e-9):
                raise RuntimeError(f"{key} {c}: CI {out[c]} != Figure 2c {tuple(pub.loc[c, ['mean', 'lo', 'hi']])}")
    return out


def same_checkpoint(blocks: dict[str, pd.DataFrame], k3: pd.DataFrame, stem: str) -> None:
    """K=3 may stand in for K=1 only if, on the arm scored both ways (``stem``; ``k3`` its K=3 scores),
    the condition-mean barely moves on every plotted metric."""
    for short, block in blocks.items():
        both = k3[k3.metric.eq(short)].set_index("condition").value.reindex(block.index)
        if both.isna().any():
            raise RuntimeError(f"{short}: the K=3 eval misses {int(both.isna().sum())} of the plotted conditions")
        shift = abs(float(both.mean()) - float(block[stem].mean()))
        print(f"25% cells {short}: K=1 {block[stem].mean():.4f}  K=3 {both.mean():.4f}  shift {shift:.4f}")
        if shift >= SHIFT_MAX:
            raise RuntimeError(f"{short}: K=3 moves the 25% cells mean by {shift:.4f} (>= {SHIFT_MAX}); not plotting K=3 arms")
