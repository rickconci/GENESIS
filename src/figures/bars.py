"""Bars that say what they mean: touching bars with value labels, their 95% CIs as whiskers,
a squeezed band when one outlier would flatten the rest, and a key that names every arm.

Every label sits at a fixed offset on paper, so axis limits are solved from the labels' measured extents
(label_room), never tuned to today's numbers.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.patheffects  # noqa: F401  (mpl.patheffects, the whiskers' halo)
import matplotlib.ticker  # noqa: F401  (mpl.ticker, the break's ticks)
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.figures import page
from src.figures.palette import ARMS, GRID, INK, hatch_ink

BREAK_GAP = 0.04                              # the squeezed band, as a share of [0, a]


# ----------------------------------------------------------------------------
# 1. The CIs a bar carries (each figure's extract.py writes them; figures only draw them)
# ----------------------------------------------------------------------------

def ci_row(ci: pd.DataFrame, key: tuple, mean: float) -> pd.Series:
    """One CI, whose point estimate must be the plotted mean."""
    r = ci.loc[key]
    if abs(r["mean"] - mean) > 1e-12:          # one float ulp of summation order, no more
        raise ValueError(f"CI {key} is for mean {r['mean']!r}, the figure plots {mean!r}: rerun bootstrap_ci.py")
    return r


def whisker(ax, x, y, lo, hi, *, horizontal: bool = False, color: str = INK, capsize: float = 1.2,
            zorder: float = 5) -> None:
    """A thin CI whisker at (x, y) from lo to hi, along y (along x if ``horizontal``); scalars, or arrays for one
    errorbar over many points. A white halo keeps the half inside a dark bar or band visible."""
    m = x if horizontal else y
    err = np.vstack([np.asarray(m, float) - np.asarray(lo, float), np.asarray(hi, float) - np.asarray(m, float)])
    halo = [mpl.patheffects.withStroke(linewidth=1.3, foreground="white")]
    for art in ax.errorbar(x, y, **{"xerr" if horizontal else "yerr": err}, fmt="none", ecolor=color, elinewidth=0.5,
                           capsize=capsize, capthick=0.5, zorder=zorder).lines[1:]:
        for a in art:
            a.set_path_effects(halo)


def ci_whiskers(ax, ci: pd.DataFrame, panel: str, metric: str, labels: list[str], x: np.ndarray,
                means: np.ndarray) -> dict:
    """Every bar's CI as a whisker; returns each bar's CI row."""
    out = {}
    for k, xi, m in zip(labels, x, means):
        out[k] = r = ci_row(ci, (panel, metric, k), m)
        whisker(ax, xi, m, r.lo, r.hi)
    return out


def label_tops(labels: list[str], means: np.ndarray, cis: dict) -> tuple[np.ndarray, np.ndarray]:
    """(lowest, highest) drawn point of each bar: its whisker's ends, or the bar top. The value
    label sits above the highest, and every axis limit is fitted to both."""
    lows = np.array([min(m, cis[k].lo) if k in cis else m for k, m in zip(labels, means)])
    return lows, np.array([max(m, cis[k].hi) if k in cis else m for k, m in zip(labels, means)])


def ci_columns(r: pd.Series | None) -> dict:
    """A plotted row's CI fields for the source data (blank for arms without a CI; callers that
    draw no CIs, e.g. the Extended Data's vendored use, add no columns)."""
    keys = ["lo", "hi", "unit", "n_units", "B", "seed"]
    return {f"ci95_{k}": (r[k] if r is not None else None) for k in keys}


# ----------------------------------------------------------------------------
# 2. Draw the bars, their labels, and the axis break
# ----------------------------------------------------------------------------

def y_break(ax, a: float, b: float, top: float, *, cut: bool, upper: float = 0.29 / 1.33) -> None:
    """Squeeze the empty band (a, b) so one outlier arm cannot flatten the rest.

    Piecewise-linear y: [0, a] keeps most of the height, (a, b) a sliver, [b, top]
    the share ``upper`` of the axis. ``cut`` whites out the band so a bar visibly breaks;
    a curve keeps what passes through it, compressed but not hidden.
    """
    if not (0 < a < b < top and 0 < upper < 1):
        raise ValueError(f"need 0 < a < b < top and 0 < upper < 1, got {a}, {b}, {top}, {upper}")
    g = BREAK_GAP * a
    u = upper / (1 - upper) * (a + g)
    fwd = lambda y: np.where(y <= a, y, np.where(y < b, a + (y - a) / (b - a) * g,  # noqa: E731
                                                a + g + (y - b) / (top - b) * u))
    inv = lambda z: np.where(z <= a, z, np.where(z < a + g, a + (z - a) / g * (b - a),  # noqa: E731
                                                 b + (z - a - g) / u * (top - b)))
    ax.set_yscale("function", functions=(lambda y: fwd(np.asarray(y, float)),
                                         lambda z: inv(np.asarray(z, float))))
    ax.set_ylim(0, top)
    lo = [t for t in mpl.ticker.MaxNLocator(4).tick_values(0, a) if 0 <= t <= a]
    # an upper tick hugging the break would sit on the lower segment's top tick: keep clear of b
    hi = [t for t in mpl.ticker.MaxNLocator(2).tick_values(b, top) if b + 0.15 * (top - b) <= t <= top]
    ax.set_yticks(lo + hi)
    if cut:
        ax.axhspan(a, b, color="white", zorder=4, lw=0)
    ya, yb = (float(fwd(np.asarray(v, float)) / fwd(np.asarray(top, float))) for v in (a, b))
    ax.plot([0, 0], [ya, yb], transform=ax.transAxes, color="white", lw=1.6,
            clip_on=False, zorder=8, solid_capstyle="butt")
    for y in (ya, yb):
        ax.plot([-0.02, 0.02], [y - 0.012, y + 0.012], transform=ax.transAxes,
                color=INK, lw=0.5, clip_on=False, zorder=9)


def bars(ax, labels: list[str], means: np.ndarray, colors: list[str], hatches: list[str]) -> np.ndarray:
    """Touching bars with a rotated value label on each (their CIs are drawn by ci_whiskers)."""
    x = np.arange(len(labels), dtype=float)
    edges = [hatch_ink(c) if h else "white" for c, h in zip(colors, hatches)]
    ax.bar(x, means, width=1.0, color=colors, hatch=hatches, edgecolor=edges,
           linewidth=0.3, zorder=3)
    ax.set_xlim(-0.7, len(x) - 0.3)
    ax.yaxis.grid(True, color=GRID, lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    return x


def bar_legend(fig, x: float, y: float, ncol: int, before: tuple = (), after: tuple = (),
               columnspacing: float = 1.0) -> None:
    """Fill/hatch key for the bar panels: every arm, between optional reference handles.
    Bar order changes per metric, so the key, not the position, names a bar."""
    handles = [*before, *(Patch(facecolor=f, hatch=h, edgecolor=hatch_ink(f) if h else "none",
                                lw=0.3, label=k) for k, (f, h, _d) in ARMS.items()), *after]
    fig.legend(handles=handles, ncol=ncol, frameon=False, loc="upper left", handlelength=1.6,
               columnspacing=columnspacing, handletextpad=0.4, labelspacing=0.35,
               bbox_to_anchor=(x / page.W, 1 - y / (fig.get_figheight() * 25.4)))


def value_labels(ax, x: np.ndarray, means: np.ndarray, tops: np.ndarray | None = None) -> list:
    """A rotated value 1 pt above each bar (or above its CI whisker, `tops`): offset on paper, so
    the same on any y scale. Two significant figures: the bar is a summary, the tables keep full precision."""
    tops = means if tops is None else tops
    # 1 pt above a bar; 2 pt above a whisker, so its cap (and halo) does not read as part of the number
    return [ax.annotate(f"{m:.2g}", (xi, y), xytext=(0, 1.0 if y == m else 2.0), textcoords="offset points",
                        ha="center", va="bottom", rotation=90, fontsize=6, color=INK, zorder=7)
            for xi, m, y in zip(x, means, tops)]


def label_room(ax, tops: np.ndarray, stacks: list[list]) -> np.ndarray:
    """Share of the axis height each bar's stacked labels take above its top, plus 1 pt.

    The labels sit at fixed offsets on paper, so the share does not depend on the y
    limits: each caller solves for the top that fits every bar on its own scale, and no
    headroom is tuned to today's numbers."""
    ax.figure.canvas.draw()                   # the labels' extents need a laid-out figure
    pad, h = ax.figure.dpi / 72, ax.get_window_extent().height
    room = np.array([(max(t.get_window_extent().y1 for t in stack) + pad
                      - ax.transData.transform((0, y))[1]) / h for y, stack in zip(tops, stacks)])
    if (room > 0.6).any():
        raise RuntimeError(f"bar labels need {room.max():.0%} of the axis height: make the axis taller")
    return room


def metric_title(ax, title: str, higher_is_better: bool, sub: str = "", **kw) -> None:
    # a and c name the metric above each axis (clearer, and narrower than a y label).
    ax.set_title(f"{title} {'↑' if higher_is_better else '↓'}" + (f"\n{sub}" if sub else ""), **kw)
