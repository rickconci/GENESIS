"""Figure 1: a-c three schematics (architecture.py, energy.py, replication.py; no data), d agreement between
two arms of one condition vs cells per arm (main/reps.parquet; its rules are data.py).

    python -m scripts.figures.figure1.draw.main.draw
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

from scripts.figures import CORE  # noqa: E402
from scripts.figures.figure1.draw.main import architecture, energy, replication  # noqa: E402
from scripts.figures.figure1.draw.main.data import METRICS, load, paired  # noqa: E402
from src.figures import page  # noqa: E402
from src.figures.schematic import MM  # noqa: E402

INK = "#1b1d1f"
RULE = "#dcdad4"

# Panel d: its colours and layout. It has no title: the selection rule (exactly two plates,
# random, >=3,200 cells) is in the legend.
SERIES = [("split_half", "Split-half", "#9c4d63"),
          ("cross_plate", "Experimental repeat", "#1f5f68")]


def style() -> None:
    """Figure 1's own sizes (not page.style's), from a fresh process's settings."""
    mpl.rc_file_defaults()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],                    # page.style's: the same pixels on any OS
        "font.size": 6.0, "axes.labelsize": 6.0, "axes.titlesize": 6.6,
        "xtick.labelsize": 5.4, "ytick.labelsize": 5.4, "legend.fontsize": 5.6,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.0, "ytick.major.size": 2.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 300,           # the schematics measure text at this resolution
        "pdf.fonttype": 42,          # text stays editable (TrueType, never outlined)
    })

# mm. Artwork sizes are the schematics' own canvases (one source: their P["canvas"]). A band above
# each row holds the panel letter so it sits at the top-left without covering the drawing.
GAP_H, GAP_V = 3.5, 2.2
MARGIN = 1.4
LETTER_BAND = 3.6
FW_W, FW_H = (architecture.P["canvas"][k] for k in ("wmm", "hmm"))
EN_W, EN_H = (energy.P["canvas"][k] for k in ("wmm", "hmm"))
RP_W, RP_H = (replication.P["canvas"][k] for k in ("wmm", "hmm"))


def _rect(x, y, w, h, fig_w, fig_h):
    return [x / fig_w, y / fig_h, w / fig_w, h / fig_h]


def _draw_panel(module, x, y, fig, fig_w, fig_h):
    """Draw one schematic into an axes whose box is its own canvas: 1 unit = 1 pt, y runs down."""
    c = module.P["canvas"]
    ax = fig.add_axes(_rect(x, y, c["wmm"], c["hmm"], fig_w, fig_h))
    ax.set_xlim(0, c["wmm"] * MM)
    ax.set_ylim(c["hmm"] * MM, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_facecolor(c["bg"])
    module.draw(ax)


def _letter(fig, lab, x, y_top, fig_w, fig_h):
    fig.text(
        x / fig_w, 1.0 - y_top / fig_h, lab,
        fontsize=8, fontweight="bold", color=INK,
        ha="left", va="top", zorder=20,
    )


def _draw_ladder(fig, x, y, w, h, fig_w, fig_h, df):
    """Three metrics, one shared legend, inside the panel-d artwork box."""
    # Fractions of the artwork box; top leaves room for the legend under the letter band.
    left, right, top, bottom, wspace = 0.085, 0.995, 0.77, 0.20, 0.42
    n = len(METRICS)
    box = _rect(x, y, w, h, fig_w, fig_h)
    inner_w = box[2] * (right - left)
    aw = inner_w / (n + (n - 1) * wspace)
    gap = wspace * aw
    x0 = box[0] + box[2] * left
    y0 = box[1] + box[3] * bottom
    ah = box[3] * (top - bottom)

    ns = sorted(df["n_cells"].unique())
    axes = []
    for i, (metric, label, higher_better) in enumerate(METRICS):
        ax = fig.add_axes([x0 + i * (aw + gap), y0, aw, ah])
        axes.append(ax)
        pw = paired(df, metric)
        for strat, name, hue in SERIES:
            g = pw.groupby("n_cells")[strat]
            m, se = g.mean(), g.sem()
            ax.plot(m.index, m.values, "-o", color=hue, lw=1.1, ms=2.6,
                    mfc="white", mew=0.9, label=name, zorder=3)
            lo, hi = m - se, m + se
            if not higher_better:
                # log axis: a band edge at or below zero would spike to the
                # floor, so clip a decade under the smallest mean.
                floor = float(np.nanmin(m.values)) / 10.0
                lo = lo.clip(lower=floor)
            ax.fill_between(m.index, lo, hi, color=hue, alpha=0.16, lw=0, zorder=2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel("Cells per arm")
        # the metric is named above its axis (clearer and narrower than a y label)
        ax.set_title(f"{label} {'↑' if higher_better else '↓'}", pad=3)
        if higher_better:
            # both are bounded by 1: a fixed 0-1 axis shows the distance to
            # perfect agreement; the small margin keeps markers near 0 unclipped
            ax.set_ylim(-0.03, 1.03)
            ax.set_yticks(np.linspace(0, 1, 6))
        else:
            ax.set_yscale("log")
            # plain decimals: 10^x superscripts render under the 5 pt floor.
            # Ticks are decades, so -log10 is the number of decimals needed.
            ax.yaxis.set_major_formatter(FuncFormatter(
                lambda v, _: f"{v:.{max(0, -round(np.log10(v)))}f}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="major", color=RULE, lw=0.4, alpha=0.8)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    # Centred over panel d, just above the axis titles, so it reads with the plots.
    leg_y = y + h * top + 4.4
    fig.legend(
        handles, labels, frameon=False, ncol=2,
        loc="lower center",
        bbox_to_anchor=((x + w / 2) / fig_w, leg_y / fig_h),
        handlelength=1.5, columnspacing=1.4, borderpad=0.0, handletextpad=0.4,
        fontsize=5.6,
    )


def build(df: pd.DataFrame) -> plt.Figure:
    style()

    row1_art = max(FW_H, EN_H)
    row2_art = RP_H
    fig_w = MARGIN + FW_W + GAP_H + EN_W + MARGIN
    fig_h = MARGIN + LETTER_BAND + row1_art + GAP_V + LETTER_BAND + row2_art + MARGIN
    ladder_w = fig_w - 2 * MARGIN - RP_W - GAP_H

    fig = plt.figure(figsize=(fig_w / 25.4, fig_h / 25.4))
    fig.patch.set_facecolor("white")
    # Text measurement in the panel scripts needs a renderer.
    fig.canvas.draw()

    # y from the top of the page; artwork sits under its letter band. matplotlib y is from the bottom.
    top1 = MARGIN
    art1 = top1 + LETTER_BAND
    y1_fw = fig_h - art1 - FW_H
    y1_en = fig_h - art1 - EN_H
    x_fw = MARGIN
    x_en = MARGIN + FW_W + GAP_H

    top2 = art1 + row1_art + GAP_V
    art2 = top2 + LETTER_BAND
    y2 = fig_h - art2 - row2_art
    x_rp = MARGIN
    x_ld = MARGIN + RP_W + GAP_H

    _letter(fig, "a", x_fw, top1, fig_w, fig_h)
    _letter(fig, "b", x_en, top1, fig_w, fig_h)
    _letter(fig, "c", x_rp, top2, fig_w, fig_h)
    _letter(fig, "d", x_ld, top2, fig_w, fig_h)

    _draw_panel(architecture, x_fw, y1_fw, fig, fig_w, fig_h)
    _draw_panel(energy, x_en, y1_en, fig, fig_w, fig_h)
    _draw_panel(replication, x_rp, y2, fig, fig_w, fig_h)
    _draw_ladder(fig, x_ld, y2, ladder_w, row2_art, fig_w, fig_h, df)
    return fig


def main(data: Path = CORE / "data" / "figure1", out: Path = CORE / "figure1") -> None:
    fig = build(load(data / "main" / "reps.parquet"))
    page.save(fig, out / "figure1", dpi=450)     # 450, not 600: the published composite is 450 dpi
    plt.close(fig)


if __name__ == "__main__":
    main()
