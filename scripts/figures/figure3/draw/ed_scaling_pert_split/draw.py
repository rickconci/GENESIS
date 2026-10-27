"""Extended Data: data efficiency for held-out perturbations, with Figure 3's schematics.

a, Tahoe-100M: a held-out compound seen in training in k of 20 pool cell lines; 12 of 40 compounds and GENESIS's
   mean over all 40 (± s.e.m.).
b, PBMC cytokine atlas: GENESIS trained on fewer cells or fewer conditions (± s.e.m.), against itself on all the
   data; n.s. marks a point where neither of Figure 2a's no-gap tests found a difference.

    python -m scripts.figures.figure3.draw.ed_scaling_pert_split.draw
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from scripts.figures import CORE  # noqa: E402
from scripts.figures.figure3.draw import panels as p  # noqa: E402
from src.figures import ed_page, metrics, page  # noqa: E402
from src.figures.palette import C_SPLIT, FILL, INK  # noqa: E402
from src.stats.equivalence import NO_GAP_MARK as NS, NO_GAP_TESTS  # noqa: E402

H, ROW_A, ROW_B = 116.0, 0.0, 54.0
LEVELS = (1, 5, 15, 20)                         # pool lines that saw the held-out compound
DRUG_LINE = "#b5b9be"
# (overlap table's key, the PBMC tables' key, title)
METRICS = (("deg", metrics.DEG, "Top-100 DEG overlap ↑"), ("pd", metrics.PEARSON, "Pearson r Δ ↑"),
           ("ed", metrics.ENERGY, "Energy distance ↓"))
QUESTION = "How many\ncell lines\nfor a given\nperturbation\ndoes GENESIS\nneed to see to\nperform well?"


def span(values) -> tuple[float, float]:
    lo, hi = float(np.min(values)), float(np.max(values))
    return lo - 0.12 * (hi - lo), hi + 0.12 * (hi - lo)


def legend(fig, handles: list, y0: float) -> None:
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(118 / ed_page.W, 1 - (y0 + 6.0) / H),
               ncol=len(handles), frameon=False, handlelength=2.6)


# ----------------------------------------------------------------------------
# Draw panel a: Figure 3b's question transposed, a held-out compound seen in k pool lines
# ----------------------------------------------------------------------------

def held_out_compound(rng):
    """Figure 3b's grid with the roles swapped: the test compounds (right) are seen in only a few pool lines
    (lower rows), never in the test lines (top rows)."""
    def face(r, c):
        if p.is_test_cell(r, c):
            return "white"
        if c >= 3:
            return p.GREYS(0.8) if (r + c) % 2 == 0 else "white"
        return p.GREYS(0.65 + 0.28 * rng.random())
    return face


def compound_curves(t: pd.DataFrame, key: str) -> np.ndarray:
    """(40 compounds x 4 k): the mean over 3 seeds (which pool lines saw it)."""
    t = t[t.metric.eq(key)]
    n = t.groupby(["drug", "overlap", "entity_seed"]).size()
    if (n != 1).any() or len(n) != 40 * len(LEVELS) * 3:
        raise RuntimeError(f"{key}: expected one row per compound x k x seed")
    return t.groupby(["drug", "overlap"]).metric_mean.mean().unstack("overlap")[list(LEVELS)].to_numpy()


def panel_a(fig, t: pd.DataFrame, y0: float) -> None:
    page.letter(fig, "a", 1, y0)
    ed_page.banner_line(fig, y0, "Tahoe-100M", "perturbation generalization", "Split-half reference protocol", C_SPLIT)
    p.schematic_b(page.axes_mm(fig, 2, y0 + 10, 46, 30), QUESTION, face=held_out_compound(np.random.default_rng(4)))
    for (key, _pbmc, title), x in zip(METRICS, (56.0, 97.5, 139.0)):    # Figure 3b's metric columns
        v = compound_curves(t, key)
        mean, sem = v.mean(axis=0), v.std(axis=0, ddof=1) / np.sqrt(len(v))
        order = np.argsort(v[:, 0])                  # 12 compounds spread over the k = 1 score
        sample = v[order[np.round(np.linspace(0, len(order) - 1, 12)).astype(int)]]
        ax = page.axes_mm(fig, x, y0 + 13, 32, 22)
        for y in sample:
            ax.plot(LEVELS, y, color=DRUG_LINE, lw=0.5)
        ax.errorbar(LEVELS, mean, yerr=sem, fmt="-o", color=FILL["GENESIS"], lw=0.8, ms=1.6,
                    capsize=1.2, elinewidth=0.5, capthick=0.5, zorder=3)      # Figure 3b's error bars
        ax.set_ylim(*span(np.r_[sample.ravel(), mean]))
        ax.set_xlim(-1, 22)
        ax.set_xticks(LEVELS, [str(k) for k in LEVELS], fontsize=5)
        ax.set_xlabel("Cell lines in which the test\nperturbation was seen in training")
        ax.set_title(title, pad=2)
        ax.grid(axis="y", color=p.GRID, lw=0.4)
    legend(fig, [Line2D([0], [0], color=DRUG_LINE, lw=0.5, label="12 of 40 compounds"),
                 Line2D([0], [0], color=FILL["GENESIS"], marker="o", lw=0.8, ms=1.6,
                        label="GENESIS, mean over 40 compounds")], y0)


# ----------------------------------------------------------------------------
# Then b: Figure 3a's design on PBMCs; no repeat, so the full model is the reference
# ----------------------------------------------------------------------------

def ns_bracket(ax, x: float, y: float, ref: float, color: str, side: int) -> None:
    """Figure 2a's n.s. bracket from a point to the reference, the two values the tests compare; ``side`` -1
    draws it left of the point, +1 right, the mark on that outer side."""
    tick = -0.035 * side
    ax.plot([x + tick, x, x, x + tick], [y, y, ref, ref], color=color, lw=0.5, zorder=6, solid_joinstyle="miter")
    ax.annotate(NS, (x, (y + ref) / 2), xytext=(1.2 * side, 0), textcoords="offset points",
                ha="left" if side > 0 else "right", va="center", fontsize=6, color=color, zorder=7)


def panel_b(fig, plotted: pd.DataFrame, stats: pd.DataFrame, y0: float) -> None:
    page.letter(fig, "b", 1, y0)
    ed_page.banner_line(fig, y0, "PBMC cytokine atlas", "biological context generalization",
                        "reference: GENESIS on all training data", FILL["GENESIS"])
    p.schematic_a(page.axes_mm(fig, 2, y0 + 10, 48, 50))
    marked = stats[stats.test.eq(NO_GAP_TESTS[0]) & stats["mark"].eq(NS)]
    ns, xpos = set(zip(marked.metric, marked.arm)), {5: 0, 25: 1, 100: 2}
    arm = {("fewer cells", 5): "n_cells_005", ("fewer cells", 25): "n_cells_025", ("fewer conditions", 25): "n_conds_025"}
    series = {"fewer conditions": (p.CONDITIONS, p.CONDITIONS, "o", p.CONDITIONS, "-", -1),
              "fewer cells": (p.CELLS, p.CELLS_INK, "s", "white", "--", +1)}     # colour, text, marker, face, dash, side
    for i, (_k, key, title) in enumerate(METRICS):
        t = plotted[plotted.metric_key.eq(key)]
        full = t[t.pct_kept.eq(100)]["mean"].unique()
        if len(full) != 1:
            raise RuntimeError(f"{key}: the two series end on different full models")
        ax = page.axes_mm(fig, 61 + 42 * i, y0 + 17, 30, 30)        # Figure 3a's metric columns
        lo, hi = span(np.r_[t["mean"] - t["sem"], t["mean"] + t["sem"]])
        for name, (color, ink, marker, face, ls, side) in series.items():
            q = t[t.series.eq(name)].sort_values("pct_kept")
            x, mean, sem = q.pct_kept.map(xpos).to_numpy(), q["mean"].to_numpy(), q["sem"].to_numpy()
            ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.15, linewidth=0)
            ax.plot(x, mean, ls=ls, color=color, lw=1.0, zorder=3)
            ax.plot(x[:-1], mean[:-1], ls="none", marker=marker, color=color, ms=3.2, mfc=face, mew=0.8, zorder=4)
            for xx, yy, pct in zip(x[:-1], mean[:-1], q.pct_kept.to_numpy()[:-1]):
                if (key, arm[name, pct]) in ns:
                    ns_bracket(ax, xx + 0.13 * side, yy, full[0], ink, side)
        ax.plot(2, full[0], "o", color=FILL["GENESIS"], ms=3.2, zorder=5)
        ax.axhline(full[0], ls=":", lw=0.8, color=FILL["GENESIS"])
        ax.set_ylim(lo, hi)
        ax.set_xlim(-0.15, 2.15)
        # Two-count ticks: each series' training cells in its colour (the series differ in cells here)
        cells = {n: t[t.series.eq(n)].set_index("pct_kept").n_train for n in series}
        ax.set_xticks([0, 1, 2], ["5%", "25%", "100%"])
        for xx, pct in zip((0, 1, 2), (5, 25, 100)):
            lines = [(cells["fewer cells"][100], INK)] if pct == 100 else \
                [(cells[n][pct], series[n][1]) for n in ("fewer conditions", "fewer cells") if pct in cells[n].index]
            for j, (c, color) in enumerate(lines):
                ax.text(xx, -0.13 - 0.085 * j, f"{p.fmt_count(c)} cells", transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=5, color=color)
        ax.set_title(title, pad=2)
        ax.grid(axis="y", color=p.GRID, lw=0.4)
    legend(fig, [Line2D([0], [0], color=p.CONDITIONS, marker="o", lw=1.0, ms=3.2, label="GENESIS, fewer conditions"),
                 Line2D([0], [0], color=p.CELLS, marker="s", mfc="white", mew=0.8, lw=1.0, ls="--", ms=3.2,
                        label="GENESIS, fewer cells"),
                 # all three meet at 100%, so the reference is named here, not on the axes
                 Line2D([0], [0], color=FILL["GENESIS"], marker="o", ls=":", lw=0.8, ms=3.2,
                        label="GENESIS, all training data")], y0)


def main(data: Path = CORE / "data" / "figure3", out: Path = CORE / "figure3") -> None:
    data = data / "ED_scaling_pert_split"
    ed_page.main_figure_style(page.style)
    fig = ed_page.figure(H)
    panel_a(fig, pd.read_csv(data / "drug_seed_overlap.csv"), ROW_A)
    panel_b(fig, pd.read_csv(data / "parse_bio_cells_conds_source_data_plotted.csv"),
            pd.read_csv(data / "parse_bio_cells_conds_source_data_statistics.csv"), ROW_B)
    ed_page.save(fig, out / "ed" / "ED_scaling_pert_split")
    plt.close(fig)


if __name__ == "__main__":
    main()
