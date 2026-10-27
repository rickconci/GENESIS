"""Figure 2's panels, drawn at print size; the perturbation-split Extended Data page draws the same three.

a, each model against the experimental repeat on the same plate pairs
b, those pairs ranked by the repeat
c, the split-half reference next to each model on held-out contexts

Positions are mm from the top left (src.figures.page). a and c share the left area and its three metric columns,
so each metric reads down one column; b runs down the right beside both.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.ticker  # noqa: F401  (mpl.ticker, c's log ticks)
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from scripts.figures.figure2.draw.data import COND_METRICS, PAIR_METRICS, R_TEST, REPEAT, metric_pairs
from src.figures import page
from src.figures.bars import (BREAK_GAP, bar_legend, bars, ci_row, ci_whiskers, label_room, label_tops, metric_title,
                              value_labels, y_break)
from src.figures.page import axes_mm, banner
from src.figures.palette import ARMS, C_REPEAT, C_SPLIT, GRID, INK, REF_DASH, curve_color

LEFT_X, LEFT_W, COL_PAD = 4.0, 123.0, 8.5    # COL_PAD: a column's room for its y ticks ("0.001" in c)
B_LETTER_X, B_X, B_W = 128.5, 132.0, 48.0   # 3 mm right margin: b's lines must not touch the edge
BAR_H, B_AX_H, C_BLOCK_H = 32.0, 38.0, 50.0
STUB = 0.06                                   # share of the axis an outlier bar shows above a break


def metric_columns(x0: float, w: float) -> list[tuple[float, float]]:
    """(x, width) of each metric's axis; a and c both call this, so their columns align."""
    cw = w / 3
    return [(x0 + i * cw + COL_PAD, cw - COL_PAD) for i in range(3)]


def rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    """Centred running mean that shrinks at the ends instead of padding."""
    n, half = len(y), window // 2
    i = np.arange(n)
    lo, hi = np.maximum(0, i - half), np.minimum(n, i + half + 1)
    cs = np.r_[0.0, np.cumsum(y.astype(np.float64))]
    return (cs[hi] - cs[lo]) / (hi - lo)


# ----------------------------------------------------------------------------
# Panel a: each model against the experimental repeat, on the same plate pairs
# ----------------------------------------------------------------------------

def panel_a(fig, pairs: pd.DataFrame, tests: pd.DataFrame, ci: pd.DataFrame, x0: float, y0: float, w: float) -> None:
    banner(fig, x0, y0, w, 4.6, [[("Tahoe-100M", INK, "bold"), ("  ·  ", INK, "normal"),
                                  ("Experimental reproducibility", C_REPEAT, "bold")]])
    ax_y = 8.0                                 # under the one-line banner: the title only
    for (m, title, hb), (ax_x, aw) in zip(PAIR_METRICS, metric_columns(x0, w)):
        ax = axes_mm(fig, ax_x, y0 + ax_y, aw, BAR_H)
        rep = float(pairs[f"{REPEAT}__{m}"].mean())
        model = {arm: float(pairs[f"{arm}__{m}"].mean()) for arm in ARMS}
        order = sorted(model, key=model.get, reverse=hb)
        means = np.array([model[k] for k in order])
        x = bars(ax, order, means, [ARMS[k][0] for k in order], [ARMS[k][1] for k in order])
        # The repeat is the reference every arm is tested against: a line across in its CI band, not a bar
        ax.axhline(rep, color=C_REPEAT, lw=0.8, ls=REF_DASH, zorder=6)
        cis = ci_whiskers(ax, ci, "a", m, order, x, means)
        r = ci_row(ci, ("a", m, REPEAT), rep)
        ax.axhspan(r.lo, r.hi, color=C_REPEAT, alpha=0.12, lw=0, zorder=1)
        cis[REPEAT], ref_top = r, max(rep, r.hi)
        lows, tops = label_tops(order, means, cis)
        marked = tests[tests.metric.eq(m) & tests.test.eq(R_TEST)].set_index("arm").mark
        # A star on a marked arm's value label (DejaVu Sans: Helvetica and Arial have no U+2605)
        stacks = [[t] + ([ax.annotate("★", (0.5, 1.0), xycoords=t, xytext=(0, 0.6), textcoords="offset points",
                                      ha="center", va="bottom", fontsize=7, fontfamily="DejaVu Sans", color=INK,
                                      zorder=7)] if marked[k] else [])
                  for k, t in zip(order, value_labels(ax, x, means, tops))]
        room = label_room(ax, tops, stacks)    # labels sit at fixed paper offsets above `tops`
        # The top one or two arms, if every one of them is >3x every other arm: break the axis
        top_k = next((j for j in (1, 2) if np.sort(means)[-j] > 3 * np.sort(means)[:-j].max()), 0)
        if top_k:
            hi = np.argsort(means)[-top_k:]
            rest, rest_room = np.delete(tops, hi), np.delete(room, hi)
            # Above the break: a stub per outlier bar (its whole whisker), then its labels. Below: every other
            # bar with its whisker and labels, and the repeat's line and band
            up = float(room[hi].max()) + STUB * top_k
            a = max(float(np.max(rest / (1 - (1 + BREAK_GAP) * rest_room / (1 - up)))), 1.05 * ref_top)
            b = 0.9 * float(lows[hi].min())
            top = max(b + (tops[i] - b) / (1 - room[i] / up) for i in hi)
            y_break(ax, a, b, top, cut=True, upper=up)
        else:
            ax.set_ylim(0, max(float(np.max(tops / (1 - room))), 1.05 * ref_top))
        metric_title(ax, title, hb, pad=3)
        ax.set_xticks([])
    # Six columns keep the key at two rows, inside the left block; the two reference keys share the first column
    bar_legend(fig, x0, y0 + ax_y + BAR_H + 2.0, ncol=6, columnspacing=0.5,
               before=(Line2D([], [], color=C_REPEAT, lw=0.8, ls=REF_DASH, label="Experimental reproducibility"),
                       Line2D([], [], ls="none", marker="*", ms=5, color=INK, label="Not statistically different\nthan repeat")))


# ----------------------------------------------------------------------------
# Then b: the same pairs ranked by the repeat, one axis per metric
# ----------------------------------------------------------------------------

def curve(ax, pairs: pd.DataFrame, m: str, hb: bool) -> dict[str, np.ndarray]:
    """The repeat drawn raw, left to right by increasing reproducibility; each arm as a running mean over ~1/40 of
    the pairs (a faint cloud under the deep models). Returns each arm's running mean."""
    rep = pairs[f"{REPEAT}__{m}"].to_numpy()
    order = np.argsort(rep if hb else -rep, kind="mergesort")
    x, win = np.arange(len(order)), max(3, len(order) // 40)
    trend = {}
    for arm, (fill, _h, dash) in ARMS.items():
        y = pairs[f"{arm}__{m}"].to_numpy()[order]
        hero, neural = arm == "GENESIS", dash == "solid"
        if neural:
            ax.scatter(x, y, s=0.6, color=fill, alpha=0.30 if hero else 0.08, linewidths=0, zorder=2, rasterized=True)
        trend[arm] = rolling_mean(y, win)
        ax.plot(x, trend[arm], color=fill if neural else curve_color(fill), lw=1.1 if hero else 0.6, ls=dash,
                zorder=8 if hero else 4)
    ax.plot(x, rep[order], color=C_REPEAT, lw=1.1, zorder=6)
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.yaxis.grid(True, color=GRID, lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    return trend


def panel_b(fig, pairs: pd.DataFrame, x0: float, y0: float, w: float, ax_h: float) -> None:
    """One axis per metric, stacked, each ranking the pairs that metric keeps by the repeat on it."""
    banner(fig, x0, y0, w, 8.0, [[("Tahoe-100M", INK, "bold")], [("Experimental reproducibility", C_REPEAT, "bold")]])
    ax_x, ax_w, gap = x0 + 7.0, w - 7.0, 6.0   # gap: room for each axis's title
    axes = [axes_mm(fig, ax_x, y0 + 12.0 + i * (ax_h + gap), ax_w, ax_h) for i in range(len(PAIR_METRICS))]
    trends = [curve(ax, metric_pairs(pairs, m), m, hb) for ax, (m, _t, hb) in zip(axes, PAIR_METRICS)]
    for ax, (_m, title, hb) in zip(axes, PAIR_METRICS):
        metric_title(ax, title, hb, pad=2)
        ax.tick_params(labelbottom=ax is axes[-1])   # one rank axis, labelled once at the foot
    axes[0].set_ylim(0, 1)
    # Energy: an arm far above the rest goes above a break, by a's rule (the top one or two arms by mean, if each
    # is >3x every other arm); in Figure 2b, Nearest-neighbor and CellFlow
    m, en = PAIR_METRICS[-1][0], axes[-1]
    p = metric_pairs(pairs, m)
    mean = {k: float(p[f"{k}__{m}"].mean()) for k in ARMS}
    srt = sorted(mean, key=mean.get)
    top_k = next((j for j in (1, 2) if mean[srt[-j]] > 3 * max(mean[k] for k in srt[:-j])), 0)
    if top_k:
        out = np.column_stack([trends[-1][k] for k in srt[-top_k:]])
        rest = p[[f"{REPEAT}__{m}"] + [f"{k}__{m}" for k in ARMS if k not in srt[-top_k:]]].to_numpy()
        y_break(en, float(np.nanpercentile(rest, 98)) * 1.12, 0.95 * np.nanmin(out), 1.05 * np.nanmax(out), cut=False)
    en.set_xlabel("Plate pairs ranked by\nexperimental reproducibility")       # one line overruns the axis
    handles = [Line2D([], [], color=C_REPEAT, lw=1.1, label="Experimental reproducibility")] + [
        Line2D([], [], color=f if d == "solid" else curve_color(f), ls=d, lw=1.1 if k == "GENESIS" else 0.8, label=k)
        for k, (f, _h, d) in ARMS.items()]
    foot = y0 + 12.0 + len(axes) * (ax_h + gap) + 5.0
    # Short handles, shifted 11 mm left: the key stays inside the 180 mm Extended Data page too
    fig.legend(handles=handles, ncol=2, frameon=False, loc="upper left", handlelength=1.4, columnspacing=0.45,
               handletextpad=0.3, labelspacing=0.35,
               bbox_to_anchor=((x0 - 11.0) / page.W, 1 - foot / (fig.get_figheight() * 25.4)))


# ----------------------------------------------------------------------------
# Then c: the split-half reference next to each model, on held-out contexts
# ----------------------------------------------------------------------------

def panel_c(fig, pc: pd.DataFrame, ceil: pd.DataFrame, ci: pd.DataFrame, panels: list[tuple[str, str]],
            x0: float, y0: float, w: float) -> None:
    """One row of bars per (panel, dataset name); the band is the mean per-condition s.d. of the split-half
    reference over its repeats."""
    head_h, ax_y = 4.6, 8.0                     # as in a
    for r, (panel, name) in enumerate(panels):
        top, sub = y0 + r * C_BLOCK_H, pc[pc.panel.eq(panel)]
        banner(fig, x0, top, w, head_h, [[(name, INK, "bold"), ("  ·  ", INK, "normal"),
                                          ("Split-half reproducibility", C_SPLIT, "bold")]])
        for (key, title, hb), (ax_x, aw) in zip(COND_METRICS, metric_columns(x0, w)):
            ax = axes_mm(fig, ax_x, top + ax_y, aw, BAR_H)
            stat = sub[sub.metric_key.eq(key)].groupby("model")["value"].agg(["mean", "count"])
            stat = stat.sort_values("mean", ascending=not hb)
            # Condition sets are per metric (DEG is undefined when truth calls no DEGs); every arm must be scored
            # on that one set, or the bars compare different conditions
            if stat["count"].nunique() != 1:
                raise ValueError(f"{panel} {key}: arms cover different condition counts {dict(stat['count'])}")
            metric_title(ax, title, hb, pad=3)
            arms, means = list(stat.index), stat["mean"].to_numpy()
            x = bars(ax, arms, means, [ARMS[k][0] for k in arms], [ARMS[k][1] for k in arms])
            ref, sd = ceil.loc[(panel, key), ["split_half_reference_mean", "split_half_reference_repeat_std"]]
            lows, tops = label_tops(arms, means, ci_whiskers(ax, ci, f"c_{panel}", key, arms, x, means))
            if not hb:
                ax.set_yscale("log")
                ax.set_ylim(min(lows.min(), ref) * 0.4, tops.max())
                # Plain decimals: 10^x superscripts set at 70% fall under Nature's 5 pt floor
                ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _p: f"{v:g}"))
            room = label_room(ax, tops, [[t] for t in value_labels(ax, x, means, tops)])
            if hb:
                ax.set_ylim(0, max(float(np.max(tops / (1 - room))), 1.05 * (ref + sd)))
            else:                               # the same fit, in decades
                lo = np.log10(ax.get_ylim()[0])
                ax.set_ylim(top=10 ** float(np.max(lo + (np.log10(tops) - lo) / (1 - room))))
            ax.axhspan(max(ref - sd, ax.get_ylim()[0]), ref + sd, color=C_SPLIT, alpha=0.12, lw=0, zorder=1)
            ax.axhline(ref, color=C_SPLIT, lw=0.8, ls=REF_DASH, zorder=6)
            ax.set_xticks([])
    bar_legend(fig, x0, y0 + (len(panels) - 1) * C_BLOCK_H + ax_y + BAR_H + 2.0, ncol=5,
               after=(Line2D([], [], color=C_SPLIT, lw=0.8, ls=REF_DASH, label="Split-half reproducibility (± repeat s.d.)"),))
