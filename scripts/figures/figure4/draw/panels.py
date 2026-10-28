"""Figure 4's heatmap at print size: rows in named blocks (measured, then each model) x drug columns
in their classes, the PROGENy score as colour. Layout is in inches from the bottom left; the page width is
src.figures.page.W, read at call time (page_in()), so the Extended Data build's 180 mm reaches every panel.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd

from src.figures import page

INK, NA_C = "#0b0b0b", "#d0d0d0"
SHADE_C = "#efefef"                             # every other drug class; well under NA_C and the lattice
TICK_PT, VMAX = 5.0, 1.5                        # Nature's minimum; one colour, one log2FC
ARM_LABEL = {"truth": "Experimental readout", "genesis": "GENESIS", "additive": "Additive",
             "copy_closest_15": "Nearest-neighbor line", "state": "State"}
ARM_COLOUR = {"genesis": "#2a78d6", "additive": "#6f6d66", "copy_closest_15": "#a9a69c",
              "state": "#1baf7a"}   # State's published hue (Figs 2-3)

# Layout is in inches and saved without "tight": a tight crop would change the page and
# with it the printed type size. page.save() refuses anything drawn off the page.
LINE_RIGHT_IN = 0.62                            # group brackets
GRID_TOP_IN, GRID_BOTTOM_IN = 0.50, 1.20        # title + colour bar; vertical drug names
# The row-group brackets sit in a 0.62 in right margin: PROGENy's long group names need short forms.
GROUP_SHORT = {"Growth factor": "Growth", "Inflammation": "Inflam.", "Hormone / dev.": "Horm."}


def page_in() -> float:
    """The page width in inches (src.figures.page.W is mm): 183 mm for Figure 4, 180 on an Extended Data page."""
    return page.W / 25.4


def style() -> None:
    """Figure 4's rcParams, from a fresh process's settings, on the main figures' 183 mm page."""
    mpl.rc_file_defaults()
    page.W = 183.0
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],    # page.style's: same pixels on any OS
        "font.size": 6.0, "axes.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "lines.markeredgewidth": 0.4,
        # A light lattice; heatmap() moves it onto the cell edges (see there).
        "axes.grid": True, "grid.color": ".8", "grid.linewidth": 0.8, "axes.axisbelow": True,
        "pdf.fonttype": 42,
    })


# ----------------------------------------------------------------------------
# 1. Order the rows and columns: the pathways in their groups, the drugs in their classes
# ----------------------------------------------------------------------------

def pathway_rows(progs: pd.DataFrame) -> tuple[list[str], list[tuple[str, int, int]]]:
    """The drawn pathways, top down, and each biological group's (short name, first row, last row).
    WNT is the last Growth row. The programmes table still files it under Hormone / dev."""
    chunks: list[tuple[str, list[str]]] = []
    for grp, g in progs[progs.set == "grid"].groupby("group", sort=False):
        chunks.append((GROUP_SHORT.get(grp, grp), list(dict.fromkeys(g.programme))))
    donors = [(i, ps) for i, (name, ps) in enumerate(chunks) if name != "Growth" and "WNT" in ps]
    if len(donors) != 1:
        raise RuntimeError(f"WNT should sit in one non-growth group, found {len(donors)}")
    donors[0][1].remove("WNT")
    growth = [ps for name, ps in chunks if name == "Growth"]
    if len(growth) != 1:
        raise RuntimeError(f"expected one Growth group, found {len(growth)}")
    growth[0].append("WNT")
    names, groups = [], []
    for name, ps in chunks:
        if ps:
            groups.append((name, len(names), len(names) + len(ps) - 1))
            names += ps
    return names, groups


def column_order(S: dict, t: pd.DataFrame) -> tuple[list[str], list]:
    """Columns, ``S['cond_order']``, and their class brackets, ``S['cond_groups']``. The order must be exactly the
    drawn drugs, once each: a class order that drops or invents one would silently hide a column."""
    drawn = set(t.loc[t.arm != "truth", "drug"].unique())
    order = S["cond_order"]
    if set(order) != drawn or len(order) != len(drawn):
        missing, extra = sorted(drawn - set(order)), sorted(set(order) - drawn)
        raise RuntimeError(f"cond_order is not the drawn drugs: missing {missing}, extra {extra}")
    partition(S["cond_groups"], len(order), "cond_groups")
    return list(order), S["cond_groups"]


def partition(spans: list, n: int, what: str) -> None:
    """``spans`` = [(name, first, last)] must cover 0..n-1 once, in order, with no gaps."""
    cursor = 0
    for name, a, b in spans:
        if a != cursor or b < a:
            raise RuntimeError(f"{what}: {name} [{a}, {b}] does not continue from {cursor}")
        cursor = b + 1
    if cursor != n:
        raise RuntimeError(f"{what}: groups end at {cursor}, not {n}")


def column_brackets(ax: plt.Axes, fig: plt.Figure, groups: list, n_cols: int, depth_in: float,
                    names_in: float, vlines: list[tuple[float, float]] | None) -> None:
    """One bracket per class, under the rotated drug names, and a rule at each boundary. ``vlines``: data-y spans
    of that rule (the core figure breaks it between cell lines); None draws it the full height of the axes.
    Every other class is shaded from the grid down ``depth_in`` inches, behind its names. The names already
    occupy ``names_in`` below the axis; the bracket sits at the bottom of that margin. A group named "long|short"
    draws the first label that fits its columns; a label wider than its columns would be read as covering the
    next class, so if none fits that raises."""
    partition(groups, n_cols, "column groups")
    col_in = ax.get_position().width * fig.get_size_inches()[0] / n_cols
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    labels = []
    for name, c0, c1 in groups:
        room, fits = (c1 - c0 + 1) * col_in, None
        for option in name.split("|"):
            probe = fig.text(0, 0, option, fontsize=TICK_PT)
            width = probe.get_window_extent(rend).width / fig.dpi
            probe.remove()
            if width <= room + 0.002:       # past the column edge it is read as the next class
                fits = option
                break
        if fits is None:
            raise RuntimeError(f"class label {name!r}: no option fits its {c1 - c0 + 1} columns ({room:.3f} in)")
        labels.append(fits)
    groups = [(lab, c0, c1) for lab, (_n, c0, c1) in zip(labels, groups)]
    depth = depth_in / (ax.get_position().height * fig.get_size_inches()[1])   # in axes heights
    for _name, c0, c1 in groups[1::2]:      # zorder under the axis (0.5 with axisbelow), so names sit on top
        ax.add_patch(plt.Rectangle((c0 - 0.5, -depth), c1 - c0 + 1, depth, transform=ax.get_xaxis_transform(),
                                   facecolor=SHADE_C, edgecolor="none", zorder=0.1, clip_on=False))
    tr = mtransforms.offset_copy(ax.get_xaxis_transform(), fig=fig, y=-(names_in - 0.02) * 72, units="points")
    for i, (name, c0, c1) in enumerate(groups):
        ax.plot([c0 - 0.4, c1 + 0.4], [0, 0], transform=tr, color=INK, lw=0.6, clip_on=False)
        ax.annotate(name, ((c0 + c1) / 2, 0), xycoords=tr, xytext=(0, -1.5),
                    textcoords="offset points", ha="center", va="top", fontsize=TICK_PT,
                    color=INK, annotation_clip=False)
        if i:
            x = c0 - 0.5
            if vlines is None:
                ax.axvline(x, color=INK, lw=0.6, zorder=4)
            else:
                for y0, y1 in vlines:
                    ax.plot([x, x], [y0, y1], color=INK, lw=0.6, zorder=4, solid_capstyle="butt")


def blocks_of(arms: list[str], t: pd.DataFrame, rows: list[tuple[str, str]], cols: list[str]) -> list:
    """Measured, then each arm: [(block name, [(row label, values over cols)])]; ``rows`` = [(label, pathway)]."""
    v = t.set_index(["programme", "drug", "arm"]).v
    return [(ARM_LABEL[arm], [(lab, [v.get((r, c, arm), np.nan) for c in cols]) for lab, r in rows])
            for arm in ["truth", *arms]]


# ----------------------------------------------------------------------------
# 2. Draw a heatmap: rows in named blocks x drug columns, PROGENy score (weighted mean log2FC)
# ----------------------------------------------------------------------------

# Over 40 lines: its blocks are the row layout and gaps, one image per block, the gap shading, the lattice,
# the column names and class brackets, the block names, the row-group brackets, the colour bar and grey key.
def heatmap(blocks: list, cols: list[str], *, left_in: float, row_in: float, groups: list, col_groups: list,
            title: str = "", names_in: float = GRID_BOTTOM_IN, shade_gaps: bool = True,
            gaps: list[float] | None = None, open_gaps: set[int] | None = None) -> plt.Figure:
    """Blocks = [(name, [(row label, values over cols)])], the first the measured readout. ``groups`` =
    [(name, first, last)] rows within a block, bracketed right of every block; ``col_groups`` the same for
    columns, bracketed under the drug names. ``names_in``: the margin the drug names hang in (a panel that bolds
    names needs more). ``shade_gaps`` paints the class shade through the white rows between blocks. ``gaps``:
    the white between consecutive blocks, in rows (default one each). ``open_gaps``: indices into ``gaps`` where
    the class rules stop, so that white stays empty."""
    gaps = [1.0] * (len(blocks) - 1) if gaps is None else list(gaps)
    if len(gaps) != len(blocks) - 1 or min(gaps, default=1.0) <= 0:
        raise ValueError(f"gaps {gaps}: want {len(blocks) - 1} positive heights, one between each pair of blocks")
    # Each block is its own image at its own height (y, in rows), so a gap can be any height; n = the
    # grid's height in rows, gaps included.
    rows, y, spans, n = [], [], [], 0.0
    for (name, rs), gap in zip(blocks, gaps + [0.0]):
        spans.append((name, n, n + len(rs) - 1))
        rows, y, n = rows + rs, y + list(n + np.arange(len(rs))), n + len(rs) + gap
    M = np.array([r[1] for r in rows], dtype=float)
    # Class names sit under the drug names: 0.13 in holds a 5 pt line plus a small pad.
    bottom = names_in + 0.13
    H, W = GRID_TOP_IN + row_in * n + bottom, page_in()
    fig = plt.figure(figsize=(W, H), facecolor="white")
    ax = fig.add_axes([left_in / W, bottom / H, (W - left_in - LINE_RIGHT_IN) / W, row_in * n / H])
    i = 0
    for (_name, y0, y1), (_n, rs) in zip(spans, blocks):
        im = ax.imshow(np.ma.masked_invalid(M[i:i + len(rs)]), aspect="auto", interpolation="nearest",
                       extent=(-0.5, len(cols) - 0.5, y1 + 0.5, y0 - 0.5),
                       cmap=mpl.colormaps["RdBu_r"].with_extremes(bad=NA_C), vmin=-VMAX, vmax=VMAX)
        i += len(rs)
    for (_name, _y0, y1), gap in zip(spans, gaps):           # a gap is space, not "not detected"
        ax.add_patch(plt.Rectangle((-0.5, y1 + 0.5), len(cols), gap, facecolor="white", edgecolor="none", zorder=3))
        if shade_gaps:                           # otherwise the gap stays white; the shade is behind the names
            for _g, c0, c1 in col_groups[1::2]:
                ax.add_patch(plt.Rectangle((c0 - 0.5, y1 + 0.5), c1 - c0 + 1, gap, facecolor=SHADE_C,
                                           edgecolor="none", zorder=3))
    # The lattice frames every cell (minor ticks on the cell edges), so one cell can be
    # followed across blocks without a line running through its middle.
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks([e for _name, y0, y1 in spans for e in np.arange(y0 - 0.5, y1 + 1)], minor=True)
    ax.grid(False)
    ax.grid(True, which="minor")
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlim(-0.5, len(cols) - 0.5), ax.set_ylim(n - 0.5, -0.5)
    # Salt forms are distinct compounds, so they are abbreviated, never dropped.
    names = [c.replace("hydrochloride", "HCl") for c in cols]
    if len(set(names)) != len(names):
        raise RuntimeError("abbreviating 'hydrochloride' merged two names")
    ax.set_xticks(range(len(cols)), names, rotation=90, fontsize=TICK_PT)
    # A class rule is one line per run of blocks. An open gap (the break between cell lines) is
    # left out, so the line does not cross it; every other gap stays part of the line.
    rules = None
    if open_gaps:
        start, end, rules = spans[0][1] - 0.5, spans[0][2] + 0.5, []
        for i in range(len(gaps)):
            if i in open_gaps:
                rules.append((start, end))
                start = spans[i + 1][1] - 0.5
            end = spans[i + 1][2] + 0.5
        rules.append((start, end))
    column_brackets(ax, fig, col_groups, len(cols), depth_in=bottom - 0.04, names_in=names_in, vlines=rules)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=TICK_PT)
    ax.tick_params(length=0, pad=1.0)
    off = (left_in - 0.24) * 72                  # block names, rotated, left of the row labels
    tr = mtransforms.offset_copy(ax.get_yaxis_transform(), fig=fig, x=-off, units="points")
    for name, y0, y1 in spans:
        ax.plot([0, 0], [y0 - 0.4, y1 + 0.4], color=INK, lw=0.6, transform=tr, clip_on=False)
        # 5 pt bold ~0.042 in / character. Break a too-long name on its first space, then on a hyphen.
        span_in = (y1 - y0 + 1) * row_in
        if max(len(line) for line in name.split("\n")) * 0.042 > span_in:
            name = name.replace(" ", "\n", 1)
        if "-" in name and max(len(line) for line in name.split("\n")) * 0.042 > span_in:
            name = name.replace("-", "-\n", 1)
        ax.annotate(name, (0, (y0 + y1) / 2), xycoords=ax.get_yaxis_transform(), xytext=(-off - 2, 0),
                    textcoords="offset points", rotation=90, ha="right", va="center", fontsize=TICK_PT,
                    fontweight="bold", color=INK, annotation_clip=False, linespacing=1.0)
    trr = mtransforms.offset_copy(ax.get_yaxis_transform(), fig=fig, x=4, units="points")
    for _name, b0, _b1 in spans:
        for gname, g0, g1 in groups:
            ax.plot([1, 1], [b0 + g0 - 0.4, b0 + g1 + 0.4], color=INK, lw=0.5, transform=trr, clip_on=False)
            ax.annotate(gname, (1, b0 + (g0 + g1) / 2), xycoords=ax.get_yaxis_transform(),
                        xytext=(7, 0), textcoords="offset points", ha="left", va="center",
                        fontsize=TICK_PT, color=INK, annotation_clip=False)
    cb_left, cb_w = W - 1.75, 1.6                # the score bar top right, the grey swatch under it
    cb = fig.colorbar(im, cax=fig.add_axes([cb_left / W, (H - 0.30) / H, cb_w / W, 0.07 / H]),
                      orientation="horizontal", extend="both", ticks=[-VMAX, 0, VMAX])
    # Σ w·log2FC / Σ|w| over the footprint genes, a weighted mean, so still in log2FC
    cb.set_label("PROGENy score (weighted mean log$_2$FC)", fontsize=TICK_PT, labelpad=2)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=TICK_PT, length=1.5, width=0.4, pad=1.0)
    cb.outline.set_linewidth(0.4)
    if np.isnan(M).any():                        # a key only for a cell that is grey
        kax = fig.add_axes([(cb_left + cb_w / 2 - 0.27) / W, (H - 0.49) / H, 0.70 / W, 0.09 / H])
        kax.set_xlim(0, 7), kax.set_ylim(0, 2), kax.axis("off")
        kax.add_patch(plt.Rectangle((0, 0.25), 0.7, 1.5, facecolor=NA_C, edgecolor="none"))
        kax.text(1.05, 1.0, "not detected", va="center", fontsize=TICK_PT, color=INK)
    fig.text(0.06 / W, 1 - 0.04 / H, title, fontsize=7, fontweight="bold", ha="left", va="top", color=INK)
    return fig
