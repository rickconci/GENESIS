"""Figure 3's look, shared with its Extended Data pages: the two downsampled series' colours, the train/test
grid schematics, and the metric axes of panels a and b.

Positions are mm from the top left (src.figures.page); page.W is read at call time, so the same panels draw on
the 183 mm main page and the 180 mm Extended Data page.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.ticker  # noqa: F401  (mpl.ticker.MaxNLocator)
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse, Polygon, Rectangle

from scripts.figures.figure3.draw.data import BIO_LEVELS
from src.figures.palette import FILL, GRID, INK

# Both downsampled series are GENESIS, so both are GENESIS blues, told apart by lightness and marker;
# light blue is too pale for 5-6 pt text, so its text is darker
CONDITIONS, CELLS, CELLS_INK = "#1a3a6b", "#6fa8ee", "#3a73c0"
MUTED = "#6f6e69"                            # a baseline's comparator line: darker than palette.MUTED
BASELINES = ("PertMean", "LinRidge", "Additive", "CtxMean", "Nearest-neighbor")   # achromatic paper-wide
EDGE = "#c7d0d3"
GREYS = LinearSegmentedColormap.from_list("cells", ["white", "black"])           # a grid cell's cell count
# Panel b's held-out lines, clear of every model hue
LINE_COLORS = {"C32": "#c0392b", "HOP62": "#d4a000", "HepG2/C3A": "#4d9221",
               "Hs 766T": "#de77ae", "PANC-1": "#54616e"}
COLS, COL_W = (61, 103, 145), 30             # the metric axes' left edges and width (mm): both rows align


def fmt_count(n: float) -> str:
    v = n / 1e6
    return f"{v:.1f}M" if v < 10 else f"{v:.0f}M"


# ----------------------------------------------------------------------------
# Draw the schematics: a 6 x 6 grid of biological contexts x perturbations, the test block top right
# ----------------------------------------------------------------------------

def is_test_cell(r: int, c: int) -> bool:
    """The test block: the top 2 contexts x the right 3 of 6 perturbations."""
    return r <= 1 and c >= 3


def grid_matrix(ax, x0: float, y0: float, w: float, face, lw: float = 0.3, outline: float = 0.6) -> float:
    """A 6 x 6 grid of width w (data units) with SQUARE cells, ``face(r, c)`` colouring each; returns its height.
    The axes are not square in print, so the height comes from their physical aspect and data limits."""
    fw, fh = ax.figure.get_size_inches()
    pos, xl, yl = ax.get_position(), ax.get_xlim(), ax.get_ylim()
    h = w * (pos.width * fw) / (pos.height * fh) * (yl[1] - yl[0]) / (xl[1] - xl[0])
    cw, ch = w / 6, h / 6
    for r in range(6):
        for c in range(6):
            ax.add_patch(Rectangle((x0 + c * cw, y0 + (5 - r) * ch), cw, ch,
                                   facecolor=face(r, c), edgecolor=EDGE, linewidth=lw))
    ax.add_patch(Rectangle((x0 + 3 * cw, y0 + 4 * ch), 3 * cw, 2 * ch, fill=False, edgecolor=INK, linewidth=outline))
    return h


def schematic_a(ax) -> None:
    """The full training grid fanned out into its two subsampled grids, #cells and #conditions. Drawn in mm (one
    data unit = 1 mm both ways), so the cells are square and each label runs along its arrow."""
    pos, (fw, fh) = ax.get_position(), ax.figure.get_size_inches()
    ax.set_axis_off()
    ax.set_xlim(0, pos.width * fw * 25.4)
    ax.set_ylim(0, ah := pos.height * fh * 25.4)
    rng = np.random.default_rng(4)
    mid, big, small, x_small = ah - 22, 17.0, 11.5, 35.0    # full grid's centre; widths; left of small grids
    grid_matrix(ax, 4.5, mid - big / 2, big,
                lambda r, c: "white" if is_test_cell(r, c) else GREYS(0.65 + 0.28 * rng.random()))
    ax.text(4.5 + big / 2, mid - big / 2 - 1.0, "Perturbation", ha="center", va="top")
    ax.text(1.5, mid, "Biological context", ha="center", va="center", rotation=90)
    ax.add_patch(Rectangle((4.5, 8.0), 2.3, 1.8, facecolor="white", edgecolor=INK, linewidth=0.5))
    ax.text(7.6, 8.9, "Test set", va="center")
    ax.imshow(np.tile(np.linspace(0, 1, 120), (8, 1)), extent=(4.5, 27.0, 4.0, 6.0),
              origin="lower", aspect="auto", cmap=GREYS)
    ax.add_patch(Rectangle((4.5, 4.0), 22.5, 2.0, fill=False, edgecolor=EDGE, linewidth=0.3))  # white end needs an edge
    ax.text(4.5, 3.5, "No cells", ha="left", va="top", fontsize=5.5)
    ax.text(27.0, 3.5, "More cells", ha="right", va="top", fontsize=5.5)
    # The two small grids sit mirror-symmetric about the full grid's centre, so the two arrows are one arrow
    # reflected; each takes its series' colour
    keep = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0],
                     [0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]])
    faces = {"#cells": lambda r, c: "white" if is_test_cell(r, c) else GREYS(0.18 + 0.34 * rng.random()),
             "#conditions": lambda r, c: "white" if is_test_cell(r, c) else (GREYS(0.8) if keep[r, c] else "white")}
    for side, word, color, ink in ((+1, "#cells", CELLS, CELLS_INK), (-1, "#conditions", CONDITIONS, CONDITIONS)):
        yc = mid + side * 12.5                              # the small grid's centre
        grid_matrix(ax, x_small, yc - small / 2, small, faces[word], outline=0.5)
        ax.text(x_small + small / 2, yc + small / 2 + 0.7, word, ha="center", va="bottom",
                fontsize=5.5, fontweight="bold", color=ink)
        (x0, y0), (x1, y1) = (4.5 + big + 1.0, mid + side * 2.0), (x_small - 1.0, yc)
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color=color, mutation_scale=7))
        dx, dy = x1 - x0, y1 - y0                           # the label rides the arrow's outer side
        nx, ny = side * -dy / np.hypot(dx, dy), side * dx / np.hypot(dx, dy)
        ax.text((x0 + x1) / 2 + 1.5 * nx, (y0 + y1) / 2 + 1.5 * ny, "subsampling", fontsize=5, color=ink,
                rotation=np.degrees(np.arctan2(dy, dx)), rotation_mode="anchor", ha="center", va="center")


def schematic_b(ax, question: str, face=None) -> None:
    """The train/test grid beside a thought cloud asking ``question``; by default the held-out lines (top rows)
    see only some perturbations (``face(r, c)`` colours a cell otherwise)."""
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    rng = np.random.default_rng(4)

    def held_out_line(r, c):
        if is_test_cell(r, c):
            return "white"
        if r <= 1:                                         # the test lines see only k perturbations
            return GREYS(0.8) if (r + c) % 2 == 0 else "white"
        return GREYS(0.65 + 0.28 * rng.random())
    h = grid_matrix(ax, 0.13, 0.24, 0.36, face or held_out_line)
    ax.text(0.31, 0.20, "Perturbation", ha="center", va="top")
    ax.text(0.07, 0.24 + h / 2, "Biological context", ha="center", va="center", rotation=90)
    # A cartoon thought cloud: round bumps on an ellipse, trailing puffs instead of a tail. Drawn in mm (the axes
    # are not square). The outline is the union of the bumps, traced by a ray from the centre to the farthest
    # bump it crosses, so neighbours meet in sharp notches.
    pos, (fw, fh) = ax.get_position(), ax.figure.get_size_inches()
    mmx, mmy = pos.width * fw * 25.4, pos.height * fh * 25.4
    cx, cy = 0.78, 0.53                                    # clear of the grid; bumps may enter the margin
    a, b, r, n = 7.4, 8.6, 3.1, 11                         # ring radii, bump radius (mm), bump count
    phi = 2 * np.pi * (np.arange(n) + 0.5) / n
    bumps = np.c_[a * np.cos(phi), b * np.sin(phi)]
    theta = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    d = np.c_[np.cos(theta), np.sin(theta)]
    extent = 1 / np.sqrt((d[:, 0] / a) ** 2 + (d[:, 1] / b) ** 2)    # the ring itself fills the middle
    for p in bumps:
        proj = d @ p
        disc = proj ** 2 - (p @ p - r * r)
        hit = disc >= 0
        extent[hit] = np.maximum(extent[hit], proj[hit] + np.sqrt(disc[hit]))
    ink = dict(facecolor="#f2f2f2", edgecolor=MUTED, linewidth=0.6, clip_on=False)
    ax.add_patch(Polygon(np.c_[cx + extent * d[:, 0] / mmx, cy + extent * d[:, 1] / mmy], closed=True, **ink))
    for px, py, pr in ((-8.6, -11.8, 1.4), (-10.6, -14.6, 0.85)):   # puffs, shrinking away down-left
        ax.add_patch(Ellipse((cx + px / mmx, cy + py / mmy), 2 * pr / mmx, 2 * pr / mmy, **ink))
    ax.text(cx, cy, question, ha="center", va="center", multialignment="center", fontsize=5.5, style="italic",
            linespacing=1.15)


# ----------------------------------------------------------------------------
# Then the metric axes
# ----------------------------------------------------------------------------

def pert_ticks(ax, xs: list[float], n_pert: list[float], label: str = "Perturbed training cells") -> None:
    """x ticks in training cells and, smaller under each, its share of the full model's (the last of xs).
    The offsets are fixed points below the axis, so every panel sets its label at the same height."""
    ax.set_xticks(xs, [fmt_count(n) for n in n_pert])
    for x, n in zip(xs, n_pert):
        pct = 100 * n / n_pert[-1]
        ax.annotate("(100%)" if pct > 99.95 else f"({pct:.1f}%)", (x, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -13.5), textcoords="offset points", ha="center", va="top", fontsize=5.0, color=INK)
    ax.annotate(label, (0.5, 0), xycoords="axes fraction", xytext=(0, -22.5),
                textcoords="offset points", ha="center", va="top", fontsize=6.0, color=INK)


def comparators(others: dict[str, float], best: list[str]) -> list[tuple]:
    """(level, name, colour, dash) of the two closest other arms: a deep model dotted in its hue, two
    achromatic baselines told apart by dash."""
    dashes = iter([":", (0, (3, 1.5))])
    return [(others[a], a, MUTED, next(dashes)) if a in BASELINES else (others[a], a, FILL[a], ":") for a in best]


def scaling_axis(ax, curves: dict, comps: list[tuple], ylim: tuple[float, float], ms: float) -> None:
    """Panel a's metric: the two series ({axis: (x, mean)}) meeting at the full model (Figure 2's GENESIS, in its
    colour), and the two closest other arms as lines across, each named at its end."""
    full = curves["conds"][0][-1]                          # both series end on the shared 100% point
    span = ylim[1] - ylim[0]
    for axis, color, marker in (("conds", CONDITIONS, "o"), ("cells", CELLS, "s")):
        x, mean = curves[axis]
        ax.plot(x, mean, color=color, lw=1.0, zorder=3)
        ax.plot(x[:-1], mean[:-1], ls="none", marker=marker, color=color, ms=ms, mfc=color, mew=0.8 * ms / 3.2,
                zorder=4)
    ax.plot(full, curves["conds"][1][-1], "o", color=FILL["GENESIS"], ms=ms, zorder=5)
    lo = min(y for y, *_ in comps)
    for y, name, color, ls in comps:  # two lines can sit ~0.002 apart: the lower one is named under its line
        under = y == lo
        ax.axhline(y, ls=ls, lw=0.8, color=color)
        ax.plot(full, y, "o", color=color, ms=0.75 * ms, zorder=2)   # these arms trained on 100% of the data
        ax.text(full - 0.1, y + (-0.02 if under else 0.015) * span, name, color=color, ha="right",
                va="top" if under else "bottom", fontsize=5.5)
    ax.set_ylim(*ylim)
    ax.set_yticks([t for t in mpl.ticker.MaxNLocator(nbins=5).tick_values(*ylim) if ylim[0] <= t <= ylim[1]])
    ax.set_xlim(-0.15, full + 0.15)
    ax.grid(axis="y", color=GRID, lw=0.4)


def overlap_axis(ax, mean: np.ndarray, sd: np.ndarray, lines: list[str], title: str, ylim: tuple) -> None:
    """Panel b's metric: one curve per held-out line (lines differ in difficulty, so they are not pooled)."""
    k = np.array([1 if v == 0 else v for v in BIO_LEVELS])  # 0 (controls only) drawn at 1
    if (mean - sd).min() < ylim[0] or (mean + sd).max() > ylim[1]:   # fixed limits; the data must fit them
        raise ValueError(f"{title}: data span [{(mean - sd).min():.3f}, {(mean + sd).max():.3f}] leaves ylim {ylim}")
    for y, e, name in zip(mean, sd, lines):
        ax.errorbar(k, y, yerr=e, fmt="-o", color=LINE_COLORS[name], lw=0.8, ms=1.6,
                    capsize=1.2, elinewidth=0.5, capthick=0.5)
    ax.set_title(title, pad=2)
    ax.set_ylim(*ylim)
    ax.set_xlim(-4, k[-1] + 8)
    ax.set_xticks(k, [str(v) for v in k], fontsize=5)      # 5 pt: 15 and 30 sit 2.6 mm apart
    ax.set_xlabel("Perturbations seen in training\nfor the test cell line")
    ax.grid(axis="y", color=GRID, lw=0.4)
