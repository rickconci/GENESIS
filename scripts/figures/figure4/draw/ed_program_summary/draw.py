"""Extended Data: pathway-level agreement in the held-out lines, every model conditioned on one batch and scored
against the other, beside experimental reproducibility (batch A scored against batch B).

    a  Pearson r of the PROGENy scores along three axes: across pathways, across compounds, across cell lines
    b  a's across-pathways r taken apart: per held-out line, per compound, and GENESIS against experimental
       reproducibility per compound

Every number and interval is computed upstream (Figure 2's cluster bootstrap); this draws them. No explanatory
text on the page: the caption says it. The panels here are also ED_tahoe_hallmark_gsea's.

    python -m scripts.figures.figure4.draw.ed_program_summary.draw
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from scripts.figures import CORE  # noqa: E402
from scripts.figures.figure4.draw import panels as F  # noqa: E402
from src.figures import ed_page  # noqa: E402
from src.figures.bars import metric_title  # noqa: E402
from src.figures.palette import C_REPEAT, INK, MUTED  # noqa: E402

# The one nearest neighbor is Figure 2a's (nn_own): the closest training line's measured response, its cells
# against its OWN plate controls, so the two lines' baseline difference does not ride along. Experimental
# reproducibility is drawn in Figure 2's teal.
REPEAT = "replicate"                   # the tables' name for batch A scored against batch B
COLOUR = F.ARM_COLOUR | {"nn_own": F.ARM_COLOUR["copy_closest_15"], REPEAT: C_REPEAT}
ARM_NAME = F.ARM_LABEL | {"nn_own": "Nearest neighbor", REPEAT: "Experimental reproducibility"}
ARMS = ["genesis", "additive", "nn_own", "state"]
H, PT = 86.0, 5.5
HALO = [pe.withStroke(linewidth=1.5, foreground="white")]   # legible over dots and rules
AXES = [("profile", "Across pathways"), ("compound", "Across compounds"), ("line", "Across cell lines")]
# b's cell lines in Figure 4's row order, spelled as the figure spells them
B_LINES = {"PANC-1": "PANC-1", "HOP62": "HOP62", "Hs_766T": "Hs 766T", "HepG2_C3A": "HepG2/C3A", "C32": "C32"}


def vbars(ax, values: dict, ci: dict, ref: float, ref_ci: tuple, arms: list[str], ylim: tuple, yticks: list,
          fmt: str, ylabel: str = "") -> None:
    """One vertical bar per model, its 95% CI as a whisker and its value above it; experimental reproducibility as a
    solid line in Figure 2's teal over its CI band, its value at the line's right end. Bars start at the axis floor."""
    span = ylim[1] - ylim[0]
    ax.axhspan(*ref_ci, color=C_REPEAT, alpha=0.12, lw=0, zorder=0)
    for i, m in enumerate(arms):
        v, (lo, hi) = values[m], ci[m]
        ax.bar(i, v, width=0.62, color=COLOUR[m], lw=0, zorder=2)
        ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK, elinewidth=0.5, capsize=1.2,
                    capthick=0.5, zorder=5)
        ax.text(i, max(hi, ylim[0]) + 0.02 * span, fmt.format(v), ha="center", va="bottom", fontsize=PT, zorder=6,
                path_effects=HALO)                 # legible where it crosses the reproducibility line
    ax.axhline(ref, color=C_REPEAT, lw=0.8, zorder=4)
    ax.text(len(arms) - 0.42, ref, fmt.format(ref), ha="left", va="center", fontsize=PT, color=C_REPEAT,
            clip_on=False)
    ax.set_xlim(-0.6, len(arms) - 0.4), ax.set_ylim(*ylim), ax.set_yticks(yticks)
    ax.set_xticks([])
    ax.set_ylabel(ylabel, fontsize=PT, labelpad=1.5)
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def row_axes(fig, t: pd.DataFrame, arms: list[str], y: float, lab: str = "a", ylabel: str = "Pearson r") -> None:
    """a: three panels, one per axis, on one scale from 0 (negative r is the axis floor)."""
    ed_page.letter(fig, lab, 0.0, y)
    for k, (axis, title) in enumerate(AXES):
        a = t[t.axis.eq(axis)].set_index("arm")
        ax = ed_page.axes_mm(fig, 12.0 + 58.0 * k, y + 5.0, 44.0, 26.0)
        vbars(ax, a.r.to_dict(), {m: (a.ci95_lo[m], a.ci95_hi[m]) for m in arms}, a.r[REPEAT],
              (a.ci95_lo[REPEAT], a.ci95_hi[REPEAT]), arms, (0.0, 0.8), [0.0, 0.2, 0.4, 0.6, 0.8], "{:.2f}",
              ylabel if k == 0 else "")
        metric_title(ax, title, True, fontsize=6.0, pad=4.0)


# Over 40 lines: its three panels (by line, by compound, GENESIS against reproducibility) share one row.
def row_marginals(fig, m: pd.DataFrame, arms: list[str], y: float, lab: str = "b",
                  ylabel: str = "Pearson r across pathways") -> None:
    """b: a's across-pathways r taken apart (``m``: the marginals table; ``arms`` drawn as given). Left, per cell
    line in B_LINES' order (keys in either spelling): each model's bar with its 95% CI, experimental
    reproducibility as a's line over its band. Middle, per compound: every compound's r per model, its median as
    a bar. Right, per compound: GENESIS against experimental reproducibility."""
    ed_page.letter(fig, lab, 0.0, y)
    line, comp = m[m.level.eq("line")].set_index(["key", "arm"]), m[m.level.eq("compound")]
    keys = set(line.index.get_level_values("key"))
    order = [(k if k in keys else shown, shown) for k, shown in B_LINES.items()]
    if missing := [k for k, _s in order if k not in keys]:
        raise RuntimeError(f"no per-line rows for {missing}")
    top = max(0.8, float(np.ceil(line.ci95_hi.max() * 5) / 5))    # a's scale unless a whisker passes it
    ax = ed_page.axes_mm(fig, 12.0, y + 5.0, 74.0, 26.0)
    w = 0.8 / len(arms)
    for i, (ln, _shown) in enumerate(order):
        rep = line.loc[(ln, REPEAT)]
        ax.fill_between([i - 0.45, i + 0.45], rep.ci95_lo, rep.ci95_hi, color=C_REPEAT, alpha=0.12, lw=0, zorder=0)
        ax.plot([i - 0.45, i + 0.45], [rep.r] * 2, color=C_REPEAT, lw=0.8, zorder=4)
        for k, arm in enumerate(arms):
            r, x = line.loc[(ln, arm)], i + (k - (len(arms) - 1) / 2) * w
            ax.bar(x, r.r, width=0.9 * w, color=COLOUR[arm], lw=0, zorder=2)
            ax.errorbar(x, r.r, yerr=[[r.r - r.ci95_lo], [r.ci95_hi - r.r]], fmt="none", ecolor=INK, elinewidth=0.5,
                        capsize=0.8, capthick=0.5, zorder=5)
            if r.r < 0:                                     # a bar below the floor would vanish: its value instead
                ax.text(x, 0.01, f"{r.r:.2f}", ha="center", va="bottom", rotation=90, fontsize=PT - 0.5,
                        path_effects=HALO, zorder=6)
    ax.set_xlim(-0.6, len(order) - 0.4), ax.set_ylim(0.0, top), ax.set_yticks(np.arange(0.0, top + 1e-9, 0.2))
    ax.set_xticks(range(len(order)), [shown for _k, shown in order], fontsize=PT)
    ax.tick_params(axis="x", length=0, pad=2.0)
    ax.set_ylabel(ylabel, fontsize=PT, labelpad=1.5)
    metric_title(ax, "By cell line", True, fontsize=6.0, pad=4.0)
    bx = ed_page.axes_mm(fig, 98.0, y + 5.0, 44.0, 26.0)
    rng = np.random.default_rng(0)                          # the jitter only: fixed so redraws are identical
    for k, arm in enumerate([REPEAT, *arms]):
        v = comp[comp.arm.eq(arm)].r.to_numpy()
        bx.scatter(k + rng.uniform(-0.2, 0.2, len(v)), v, s=1.5, color=COLOUR[arm], alpha=0.7, lw=0, zorder=3)
        bx.plot([k - 0.3, k + 0.3], [np.median(v)] * 2, color=INK, lw=0.9, zorder=4)
    bx.axhline(0, color=INK, lw=0.4, zorder=1)
    bx.set_xlim(-0.6, len(arms) + 0.6), bx.set_ylim(-0.4, 1.0), bx.set_yticks([-0.4, 0.0, 0.4, 0.8])
    bx.set_xticks([])
    metric_title(bx, "By compound", True, fontsize=6.0, pad=4.0)
    cx = ed_page.axes_mm(fig, 152.0, y + 5.0, 26.0, 26.0)
    wide = comp.pivot(index="key", columns="arm", values="r")
    cx.plot([-0.4, 1.0], [-0.4, 1.0], color=MUTED, lw=0.5, ls=(0, (3, 2)), zorder=1)
    cx.scatter(wide[REPEAT], wide["genesis"], s=2.0, color=COLOUR["genesis"], alpha=0.8, lw=0, zorder=3)
    cx.set_xlim(-0.4, 1.0), cx.set_ylim(-0.4, 1.0), cx.set_xticks([0.0, 0.5, 1.0]), cx.set_yticks([0.0, 0.5, 1.0])
    cx.set_xlabel("Experimental reproducibility", fontsize=PT, labelpad=1.5)
    cx.set_ylabel("GENESIS", fontsize=PT, labelpad=1.5)
    metric_title(cx, "Per compound", True, fontsize=6.0, pad=4.0)
    for a in (ax, bx, cx):
        a.grid(False)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)


def key(fig, arms: list[str], x: float, y: float) -> None:
    """One key for the page, one row at its top left: experimental reproducibility first, then each model."""
    handles = [Patch(color=COLOUR[m], lw=0, label=ARM_NAME[m]) for m in [REPEAT, *arms]]
    fig.legend(handles=handles, ncol=len(handles), frameon=False, fontsize=PT, loc="upper left", handlelength=1.2,
               columnspacing=1.2, borderaxespad=0, borderpad=0,
               bbox_to_anchor=(x / ed_page.W, 1 - y / (fig.get_figheight() * 25.4)))


def main(data: Path = CORE / "data" / "figure4", out: Path = CORE / "figure4") -> None:
    d = data / "ED_program_summary"
    a, m = pd.read_csv(d / "axis_ci.csv"), pd.read_csv(d / "profile_marginals.csv")
    if missing := {*ARMS, REPEAT} - set(a.arm) | {*ARMS, REPEAT} - set(m.arm):
        raise RuntimeError(f"{d} lacks {sorted(missing)}")
    ed_page.main_figure_style(F.style)
    fig = ed_page.figure(H)
    key(fig, ARMS, 6.0, 0.5)
    row_axes(fig, a, ARMS, 6.0)
    row_marginals(fig, m, ARMS, 44.0)
    ed_page.save(fig, out / "ed" / "ED_program_summary")


if __name__ == "__main__":
    main()
