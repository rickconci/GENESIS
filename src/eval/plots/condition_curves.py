"""Per-condition curves: every condition ranked by a reference, every arm as a running mean.

Bars average conditions away; this keeps them. Conditions (or plate pairs) go left
to right by increasing reference reproducibility. The reference is drawn raw, each
arm as a centred running mean over ~1/40 of the points, so an arm that tracks the
reference shows where it holds up and where it does not.

Two references, one plot:
  split_half       per-condition split-half score (noise_ceiling.json results) vs
                   each arm's per-condition score (metrics_summary.json)
  replicate_pairs  per plate pair: true_true (plate A vs plate B) vs each arm's
                   cross_mean (pair_metrics__<arm>.parquet)
Data functions are pure (paths in, DataFrame out); ``draw`` only draws.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.eval.plots.style import COL, GRID, INK, MUTED, SPLIT_HALF_C  # noqa: E402
from src.eval.results import load_percond  # noqa: E402

REPEAT_C = "#1f5f68"   # experimental repeat teal, as Figure 2
REF = "reference"


def rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    """Centred running mean that shrinks at the ends instead of padding."""
    n, half = len(y), window // 2
    i = np.arange(n)
    lo, hi = np.maximum(0, i - half), np.minimum(n, i + half + 1)
    cs = np.r_[0.0, np.cumsum(np.asarray(y, dtype=np.float64))]
    return (cs[hi] - cs[lo]) / (hi - lo)


# 1. Build the two wide frames: one row per unit, f"{arm}__{metric}" columns

def pair_frame(pair_files: Mapping[str, Path], keys: Sequence[str], metrics: Sequence[str],
               reference_arm: str, directions: bool = False) -> pd.DataFrame:
    """Plate pairs shared by every arm, finite for every arm on every metric.

    The repeat (true_true) is read from `reference_arm`'s file: energy subsamples
    cells, so each file's copy of it differs by ~1e-3. `directions` adds each arm's two
    directions, m(A, B') and m(B, A') (`<arm>__<metric>__ij` / `__ji`), which the averaged
    win rate votes on.
    """
    if reference_arm not in pair_files:
        raise KeyError(f"reference arm {reference_arm!r} not among {list(pair_files)}")
    frames = {}
    for arm, path in pair_files.items():
        df = pd.read_parquet(path).set_index(list(keys))
        if df.index.has_duplicates:
            raise ValueError(f"{arm}: duplicate plate-pair keys in {path}")
        frames[arm] = df
    common = sorted(set.intersection(*(set(f.index) for f in frames.values())))
    wide = pd.DataFrame(index=pd.MultiIndex.from_tuples(common, names=list(keys)))
    for m in metrics:
        wide[f"{REF}__{m}"] = frames[reference_arm].loc[common, f"true_true__{m}"]
        for arm, df in frames.items():
            wide[f"{arm}__{m}"] = df.loc[common, f"cross_mean__{m}"]
    finite = np.isfinite(wide.to_numpy(dtype=float)).all(axis=1)
    print(f"plate pairs: {len(common)} shared, {int((~finite).sum())} non-finite dropped, {int(finite.sum())} kept")
    kept = wide[finite].copy()
    if directions:
        for m in metrics:
            for arm, df in frames.items():
                for d in ("ij", "ji"):
                    kept[f"{arm}__{m}__{d}"] = df.loc[kept.index, f"cross_{d}__{m}"]
        if not np.isfinite(kept.to_numpy(dtype=float)).all():
            raise ValueError("a kept pair has a finite cross_mean but a non-finite direction")
    return kept.reset_index()


def condition_frame(arm_summaries: Mapping[str, Path], ceiling_json: Path, metrics: Sequence[tuple],
                    key: str) -> pd.DataFrame:
    """Conditions where the split-half reference and every arm have metric `key`."""
    ref = load_percond(ceiling_json, metrics)
    arms = {a: load_percond(p, metrics) for a, p in arm_summaries.items()}
    keep = sorted(c for c in ref if key in ref[c] and all(key in arms[a].get(c, {}) for a in arms))
    if not keep:
        raise ValueError(f"no condition has {key} for the reference and every arm")
    out = pd.DataFrame({"condition": ["|".join(c) for c in keep], f"{REF}__{key}": [ref[c][key] for c in keep]})
    for a in arms:
        out[f"{a}__{key}"] = [arms[a][c][key] for c in keep]
    return out


def ranked(frame: pd.DataFrame, arms: Sequence[str], metric: str, higher_is_better: bool) -> pd.DataFrame:
    """Units sorted by the reference (worst -> best, stable), with each arm's running mean."""
    ref = frame[f"{REF}__{metric}"].to_numpy(dtype=float)
    order = np.argsort(ref if higher_is_better else -ref, kind="mergesort")
    win = max(3, len(order) // 40)
    id_cols = [c for c in frame.columns if "__" not in c]
    out = frame.iloc[order][id_cols].reset_index(drop=True).assign(metric=metric, rank=np.arange(len(order)),
                                                                   **{REF: ref[order]})
    for a in arms:
        y = frame[f"{a}__{metric}"].to_numpy(dtype=float)[order]
        out[a], out[f"{a}__running_mean"] = y, rolling_mean(y, win)
    return out


# 2. Draw the figure: one row per metric

def draw(tables: Sequence[tuple[str, bool, bool, pd.DataFrame]], arms: Sequence[str], colors: Mapping[str, str],
         reference_label: str, unit_label: str, title: str, out_png: Path) -> list[Path]:
    """tables: (y label, higher_is_better, log_y, ranked table) per metric row."""
    matplotlib.rc_file_defaults()   # src.eval.plots applies a seaborn theme on import
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42, "svg.fonttype": "none"})
    ref_c = REPEAT_C if reference_label.startswith("Experimental") else SPLIT_HALF_C
    fig, axes = plt.subplots(len(tables), 1, figsize=(6.4, 2.3 * len(tables) + 0.6), squeeze=False)
    for ax, (ylabel, hb, logy, t) in zip(axes[:, 0], tables):
        x = t["rank"].to_numpy()
        if logy:
            ax.set_yscale("log")
        for i, a in enumerate(arms):
            hero = i == 0
            if hero:   # the lead arm's raw cloud under its trend; other arms' clouds only add noise
                ax.scatter(x, t[a], s=0.6, color=colors[a], alpha=0.25, linewidths=0, zorder=2, rasterized=True)
            ax.plot(x, t[f"{a}__running_mean"], color=colors[a], lw=1.1 if hero else 0.7, zorder=8 if hero else 4,
                    label=a)
        ax.plot(x, t[REF], color=ref_c, lw=1.1, zorder=6, label=reference_label)
        ax.set_ylabel(f"{ylabel} {'↑' if hb else '↓'}", color=INK)
        ax.set_xlabel(f"{unit_label} ranked by {reference_label.lower()} (n = {len(t)})", color=INK, fontsize=7)
        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.yaxis.grid(True, color=GRID, lw=0.4, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, frameon=False, fontsize=6.5, loc="lower center")
    fig.suptitle(title, fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("png", "pdf", "svg"):
        path = out_png.with_suffix(f".{ext}")
        fig.savefig(path, dpi=200, facecolor="white")
        written.append(path)
    plt.close(fig)
    return written


def arm_colors(arms: Sequence[str], color_keys: Mapping[str, str]) -> dict[str, str]:
    """Registry palette where an arm has a key; an unknown arm raises rather than guessing."""
    out = {}
    for a in arms:
        key = color_keys.get(a, a)
        if key not in COL:
            raise KeyError(f"arm {a!r}: no palette entry for {key!r}; pass NAME:COLORKEY=PATH")
        out[a] = COL[key]
    return out
