"""Figure 3: how much data GENESIS needs.

a, GENESIS trained on fewer conditions or fewer cells, scored on Figure 2c's Tahoe-100M conditions against the two
   closest other arms (trained on everything). Each point is the mean of 3 runs; whiskers and bands are 95% CIs.
b, a held-out cell line that saw 0-164 perturbations in training.

    python -m scripts.figures.figure3.draw.main.draw       # core_deliverables/data/figure3 -> core_deliverables/figure3
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
from scripts.figures.figure3.draw import data as d, panels as p  # noqa: E402
from src.figures import page  # noqa: E402
from src.figures.bars import whisker  # noqa: E402
from src.figures.palette import FILL  # noqa: E402

MS = 1.6        # half the usual 3.2 pt: on ~4,000 conditions most CIs (+/-0.006) would hide under a full marker
YLIM_B = {"deg_adaptivek_truth": (0.20, 0.65), "pearson_r_diff": (0.40, 0.85), "energy_distance": (0.00, 0.15)}
QUESTION_B = "How many\nperturbations\nfor a given\ncell line does\nGENESIS need\nto see to\nperform well?"


def legend(fig, handles: list, y: float, **kw) -> None:
    """One key in one row under a panel's plots, centred on the metric columns, its top y mm down."""
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(118 / page.W, 1 - y / (fig.get_figheight() * 25.4)), **kw)


def panel_a(fig, blocks: dict, arms: pd.DataFrame, published: pd.DataFrame) -> None:
    stems = {(r.axis, int(r.pct_kept)): r.stem for r in arms.itertuples()}
    stems["cells", 100] = stems["conds", 100]                 # both series end on the same full model
    levels = sorted({pct for _axis, pct in stems})            # 5, 25, 100: x = 0, 1, 2
    by = arms.set_index("stem").n_pert
    n_pert = [by[stems["conds", pct]] for pct in levels]
    if n_pert != [by[stems["cells", pct]] for pct in levels]:   # each cells run draws its conds twin's perturbed cells
        raise ValueError(f"the two series differ in perturbed cells: {by.to_dict()}")
    for col, (short, (key, title, higher)) in zip(p.COLS, d.METRICS.items()):
        ax, block = page.axes_mm(fig, col, 8, p.COL_W, 32), blocks[short]
        others = {a: float(block[a].mean()) for a in FILL if a != "GENESIS"}
        best = sorted(others, key=others.get, reverse=higher)[:2]                  # the two closest other arms
        ci = d.cis(block, ["GENESIS", *arms.stem, *best], key, published)         # GENESIS: checked, not drawn
        curves = {axis: (np.arange(len(levels)), np.array([ci[stems[axis, pct]][0] for pct in levels]))
                  for axis in ("conds", "cells")}
        shown = [*stems.values(), *best]
        lo, hi = min(ci[s][1] for s in shown), max(ci[s][2] for s in shown)
        pad = 0.12 * (hi - lo)
        p.scaling_axis(ax, curves, p.comparators(others, best), (lo - pad, hi + pad), MS)
        p.pert_ticks(ax, list(range(len(levels))), n_pert)
        for a in best:              # a comparator is a line across the axis, so its CI is a band along it
            ax.axhspan(ci[a][1], ci[a][2], color=p.MUTED if a in p.BASELINES else FILL[a], alpha=0.18, lw=0, zorder=1)
        full = len(levels) - 1
        whiskers = [(x, stems[axis, pct]) for axis in ("conds", "cells") for x, pct in enumerate(levels[:-1])]
        for x, s in whiskers + [(full, s) for s in (stems["conds", 100], *best)]:
            whisker(ax, x, *ci[s], zorder=7)                  # above the markers, so the part inside stays visible
        ax.set_title(title, pad=2)
    p.schematic_a(page.axes_mm(fig, 2, 4, 48, 50))
    legend(fig, [Line2D([0], [0], color=p.CONDITIONS, marker="o", lw=1.0, ms=MS, label="GENESIS, fewer conditions"),
                 Line2D([0], [0], color=p.CELLS, marker="s", lw=1.0, ms=MS, label="GENESIS, fewer cells")],
           50.0, handlelength=2.6)


def panel_b(fig, data: Path) -> None:
    p.schematic_b(page.axes_mm(fig, 2, 62.5, 46, 30), QUESTION_B)    # grid left half, thought cloud right half
    for col, (short, (key, title, _higher)) in zip(p.COLS, d.METRICS.items()):
        mean, sd, lines = d.held_out_lines(data, key)
        p.overlap_axis(page.axes_mm(fig, col, 65, p.COL_W, 25), mean, sd, lines, title, YLIM_B[short])
    legend(fig, [Line2D([0], [0], color=p.LINE_COLORS[ln], marker="o", lw=0.8, ms=1.6, label=ln) for ln in lines],
           99.5, handlelength=1.8, columnspacing=1.4)


def main(data: Path = CORE / "data" / "figure3", out: Path = CORE / "figure3") -> None:
    page.style()
    runs = d.load_runs(data / "main")
    published = pd.read_csv(data / "main" / "figure2" / "confidence_intervals.csv")
    fig = plt.figure(figsize=(page.W / 25.4, 104.5 / 25.4), facecolor="white")
    page.letter(fig, "a", 1, 1)
    panel_a(fig, d.on_figure2c(data / "main", runs), d.pooled(runs), published)
    page.letter(fig, "b", 1, 61.5)
    panel_b(fig, data / "main")
    page.save(fig, out / "figure3")
    plt.close(fig)


if __name__ == "__main__":
    main()
