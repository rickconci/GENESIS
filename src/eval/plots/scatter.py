"""Low-level scatter panel primitive."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

from .constants import TRUE_COLOR


def draw_scatter_panel(
    ax: plt.Axes,
    true_arr: np.ndarray,
    pred_arr: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    alpha: float = 0.25,
    s: float = 4.0,
) -> None:
    """Draw one scatter panel with identity line and Pearson r/R^2 annotation.

    Filters out non-finite entries before computing the correlation, so plots
    stay stable even when ``log`` transforms introduce ``-inf`` on exactly
    zero entries.
    """
    mask = np.isfinite(true_arr) & np.isfinite(pred_arr)
    x, y = np.asarray(true_arr)[mask], np.asarray(pred_arr)[mask]
    if x.size == 0:
        ax.text(
            0.5, 0.5, "no finite points",
            transform=ax.transAxes, ha="center", va="center", fontsize=10,
        )
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        return

    ax.scatter(x, y, alpha=alpha, s=s, edgecolors="none", c=TRUE_COLOR)

    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    margin = (hi - lo) * 0.02 if hi > lo else 1.0
    ax.plot(
        [lo - margin, hi + margin], [lo - margin, hi + margin],
        ls="--", color="gray", alpha=0.7, lw=1, zorder=0,
    )

    r, _ = sp_stats.pearsonr(x, y)
    r2 = r * r
    ax.text(
        0.05, 0.95,
        f"Pearson r = {r:.4f}\nR\u00b2 = {r2:.4f}\nn_genes = {len(x):,}",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
