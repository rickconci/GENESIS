"""Extended Data: the PBMC cytokine atlas has no unconfounded repeat, and its within-plate well effect is large.

a, the design as run: one donor per plate, one fixed layout, so a cytokine's only other measurement is another donor
b, two equal-size control samples, from two control wells (with the well effect) or the halves of one (without)
c, how far apart those two samples are, per metric, with stratum-clustered 95% CIs (each mean checked against the
   published summary)
Control wells are green, one shade per well and one green per plate, so b's wells are a's plate-1 wells 1 and 2.
Orange = includes the well effect, teal = does not.

    python -m scripts.figures.figure2.draw.ed_pbmc_wells.draw
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from scripts.figures import CORE  # noqa: E402
from scripts.figures.figure2.draw.data import PAIR_METRICS  # noqa: E402
from src.figures import ed_page, page  # noqa: E402
from src.figures.bars import bars, label_room, metric_title, value_labels, whisker  # noqa: E402
from src.figures.palette import INK  # noqa: E402

H = 53.0
WITH, WITHOUT, C_CYTO, GREY = "#c8673e", "#1f5f68", "#e8a33d", "#6f6c64"   # well effect present, absent
C_WELL = "#d9d7d0"
# The six control wells, one shade each; one green per plate (grass, yellow-green), kept off WITHOUT's teal and
# darker than the grey cytokine wells. Dark and light alternate, so neighbouring wells (b's 1 and 2) differ
GREENS = (("#14532d", "#4cb563", "#1f7a3a", "#7acc85", "#2f9a4a", "#a8dfa8"),
          ("#4d6b12", "#a1c03a", "#68891a", "#bdd65c", "#84a626", "#d6e98a"))
ROWS = "ABCDEFGH"
PITCH, R_WELL = 1.95, 0.72                     # a's plate: well pitch and radius (mm)
R_W = 2.6                                      # b's wells (mm); each holds 8 cells at these offsets
CELL_XY = [(-1.2, -1.1), (0.0, -1.3), (1.2, -1.1), (-1.4, 0.1), (-0.15, 0.05), (1.3, 0.15), (-0.6, 1.25), (0.75, 1.2)]
N_BOOT, SEED = 10_000, 0


def mm_axes(fig, x, y, w, h):
    """Axes whose data units are millimetres, origin top-left, y down: schematics drawn to size."""
    ax = ed_page.axes_mm(fig, x, y, w, h)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_axis_off()
    return ax


# ----------------------------------------------------------------------------
# Panel a: the design as run (one donor per plate, fixed layout, controls on the edge row)
# ----------------------------------------------------------------------------

def plate_box(ax, x0, y0, z=1):
    ax.add_patch(FancyBboxPatch((x0, y0), 12 * PITCH + 0.8, 8 * PITCH + 0.8, boxstyle="round,pad=0,rounding_size=0.7",
                                fc="#fafaf8", ec=GREY, lw=0.4, zorder=z))


def well_xy(x0, y0, row: int, col: int):
    return x0 + 0.4 + PITCH * (col + 0.5), y0 + 0.4 + PITCH * (row + 0.5)


def as_run(ax, layout: pd.DataFrame, pick: str) -> None:
    """Plates 1 and 2 in full, side by side: the same layout, so the same cytokine sits in the same well (no
    randomization) and its only other measurement is another donor (no replication, no blocking). Plates 3-12,
    identical again, stack behind plate 2. Controls: plate p's k-th well (by column) is GREENS[p][k]."""
    ctrl = sorted(layout.loc[layout.cytokine == "PBS", "col"])
    if len(ctrl) != len(GREENS[0]):
        raise ValueError(f"{len(ctrl)} control wells in plate_layout.csv; GREENS has {len(GREENS[0])} shades")

    def wells(x0, y0, z, greens):
        for _, w in layout.iterrows():
            fc = greens[ctrl.index(w.col)] if w.cytokine == "PBS" else C_CYTO if w.cytokine == pick else C_WELL
            ax.add_patch(Circle(well_xy(x0, y0, ROWS.index(w.row), w.col - 1), R_WELL, fc=fc, ec="none", zorder=z))
    x1, x2, y0, dx, dy = 1.0, 31.0, 14.0, 2.0, 0.95
    for k in range(12, 2, -1):                     # plates 12 ... 3, back to front, behind plate 2
        plate_box(ax, x2 + (k - 2) * dx, y0 - (k - 2) * dy, z=20 - k)
    plate_box(ax, x2, y0, z=20)
    wells(x2, y0, 21, GREENS[1])
    plate_box(ax, x1, y0, z=20)
    wells(x1, y0, 21, GREENS[0])
    pw = layout[layout.cytokine == pick].iloc[0]
    a, b = (well_xy(x, y0, ROWS.index(pw.row), pw.col - 1) for x in (x1, x2))
    ax.plot([a[0], b[0]], [a[1], b[1]], color=C_CYTO, lw=0.6, ls=(0, (2, 1.5)), zorder=25)
    below = y0 + 8 * PITCH + 1.2
    for k in (0, 1):                               # plate 1's control wells 1 and 2: b's two wells
        ax.text(well_xy(x1, y0, 7, ctrl[k] - 1)[0], below, str(k + 1), ha="center", va="top", fontsize=5)
    below += 2.3
    ax.text(x1 + 6 * PITCH + 0.4, below, "Donor 1", ha="center", va="top", fontsize=5.5)
    ax.text(x2 + 6 * PITCH + 0.4, below, "Donor 2", ha="center", va="top", fontsize=5.5)
    ax.text(x2 + 10 * dx + 12 * PITCH + 1.8, y0 - 10 * dy + 0.4, "… Donor 12", va="top", fontsize=5.5)


def panel_a(fig, layout: pd.DataFrame) -> None:
    ed_page.letter(fig, "a", 0.0, 0.0)
    page.banner(fig, 4.0, 0.0, 84.0, 5.0, [[("Can donor effects be told apart from plate effects?", INK, "bold")]])
    pick = "IFN-gamma" if "IFN-gamma" in set(layout.cytokine) else layout.cytokine.iloc[0]
    fig.text(6.0 / ed_page.W, 1 - 7.5 / H, "PBMC cytokine atlas dataset design", fontsize=6.0, fontweight="bold",
             va="top")
    as_run(mm_axes(fig, 6.0, 9.0, 84.0, 36.0), layout, pick)
    for kx, c in [(8.0, C_CYTO)] + [(22.0 + 2.2 * i, c) for i, c in enumerate(GREENS[0][:2] + GREENS[1][:2])]:
        mm_axes(fig, kx, 47.0, 2.0, 2.0).add_patch(Circle((1.0, 1.0), 0.8, fc=c, ec="none"))
    for kx, label in ((10.6, "Cytokine"), (31.0, "Controls")):
        fig.text(kx / ed_page.W, 1 - 48.0 / H, label, fontsize=5.5, va="center")


# ----------------------------------------------------------------------------
# Then b: one cytokine well against two control samples of 8 cells each (matched size is drawn, not asserted)
# ----------------------------------------------------------------------------

def arrow(ax, a, b, colour, shrink=2.2):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=4.5, shrinkA=shrink, shrinkB=shrink,
                                 connectionstyle="arc3,rad=0.0", color=colour, lw=0.6, zorder=6))


def cells_well(ax, c, colours, r=R_W, split=False):
    """One well at centre c holding len(colours) cells (8, or 16 when split into two halves of 8)."""
    ax.add_patch(Circle(c, r, fc="#fafaf8", ec=GREY, lw=0.5, zorder=2))
    if split:                                      # two halves side by side, a dashed cut between
        ax.plot([c[0], c[0]], [c[1] - r + 0.3, c[1] + r - 0.3], color=GREY, lw=0.45, ls=(0, (2, 1.5)), zorder=4)
        for j, half in enumerate((colours[:8], colours[8:])):
            for (dx_, dy_), col in zip(CELL_XY, half):
                ax.add_patch(Circle((c[0] + (j - 0.5) * 2.1 + 0.55 * dx_, c[1] + 0.95 * dy_), 0.32, fc=col, ec="none",
                                    zorder=3))
        return
    for (dx_, dy_), col in zip(CELL_XY, colours):
        ax.add_patch(Circle((c[0] + dx_, c[1] + dy_), 0.42, fc=col, ec="none", zorder=3))


def schematic_b(ax) -> None:
    """The cytokine well (right, mid-height) against two control wells (top: with the well effect) and against the
    two halves of one (bottom: without); the wells are a's plate-1 control wells 1 and 2, in their greens."""
    w1, w2 = GREENS[0][:2]
    xc, (y_top, y_bot) = 24.0, (8.5, 25.5)
    ymid = (y_top + y_bot) / 2
    cells_well(ax, (xc, ymid), [C_CYTO] * 8)
    ax.text(xc, ymid + R_W + 0.6, "Cytokine\nwell", ha="center", va="top", fontsize=5, color=GREY)
    for j, (cols, label) in enumerate((([w1] * 8, "1"), ([w2] * 8, "2"))):
        cx = 3.5 + R_W + j * (2 * R_W + 2.4)
        cells_well(ax, (cx, y_top), cols)
        arrow(ax, (cx, y_top + R_W), (xc, ymid), WITH, shrink=2.6)
        ax.text(cx, y_top - R_W - 0.4, label, ha="center", va="bottom", fontsize=5, color=GREY)
    ax.text(2.0, y_top - R_W - 3.4, "Between two control wells", fontsize=5.5, color=WITH, fontweight="bold",
            va="bottom")
    r, cx = R_W + 0.9, 3.5 + R_W + 2.0
    cells_well(ax, (cx, y_bot), [w1] * 16, r=r, split=True)
    for dx_ in (-1.1, 1.1):
        arrow(ax, (cx + dx_, y_bot - r + 0.4), (xc, ymid), WITHOUT, shrink=2.6)
    ax.text(cx, y_bot + r + 0.4, "1, two halves", ha="center", va="top", fontsize=5, color=GREY)
    ax.text(2.0, y_bot + R_W + 3.4, "Within one control well (split-half)", fontsize=5.5, color=WITHOUT,
            fontweight="bold", va="top")
    ax.add_patch(FancyArrowPatch((0.9, y_top), (0.9, y_bot), arrowstyle="<|-|>", mutation_scale=3.5, color=GREY,
                                 lw=0.5, shrinkA=0, shrinkB=0))
    ax.text(0.4, ymid, "Matched size", rotation=90, ha="right", va="center", fontsize=5, color=GREY)


# ----------------------------------------------------------------------------
# Then c: between vs within, per metric, with the published stratum-clustered CIs
# ----------------------------------------------------------------------------

def cluster_ci(values: np.ndarray, cluster: np.ndarray) -> tuple[float, float]:
    """95% percentile CI of a mean, resampling whole clusters (multinomial weights, fixed seed). Not
    src.stats.bootstrap's draw (integer draws, its own seed): the published CIs are this one's."""
    if len(values) != len(cluster) or not np.isfinite(values).all():
        raise ValueError(f"{len(values)} values ({int((~np.isfinite(values)).sum())} non-finite), {len(cluster)} labels")
    ids, inv = np.unique(cluster, return_inverse=True)
    if len(ids) < 2:            # every resample would be the sample itself: a zero-width CI that reads as certainty
        raise ValueError(f"{len(ids)} cluster: a CI needs at least 2")
    s, n = np.bincount(inv, weights=values), np.bincount(inv).astype(float)
    w = np.random.default_rng(SEED).multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=N_BOOT)
    lo, hi = np.percentile((w @ s) / (w @ n), [2.5, 97.5])
    return float(lo), float(hi)


def between_within(summary: dict, per: pd.DataFrame, energy: pd.DataFrame) -> list[list[tuple]]:
    """Per metric, [(mean, lo, hi) between, within], clustered on donor x cell type; means must equal the summary's."""
    strata = (per.donor + "|" + per.cell_type).to_numpy()
    e = energy.groupby(["donor", "cell_type", "arm"])["energy"].mean().unstack()
    out = []
    for metric in ("deg", "pearson_delta", "energy"):
        if metric == "energy":
            vals, clu, sm = e, (e.index.get_level_values(0) + "|" + e.index.get_level_values(1)).to_numpy(), \
                summary["energy_per_stratum"]
        else:
            ok = per[[f"{metric}__within", f"{metric}__between"]].notna().all(axis=1).to_numpy()
            vals = per.loc[ok, [f"{metric}__within", f"{metric}__between"]].set_axis(["within", "between"], axis=1)
            clu, sm = strata[ok], summary[metric]
        pair = []
        for arm in ("between", "within"):
            v = vals[arm].to_numpy(float)
            if not np.isclose(v.mean(), sm[f"{arm}_mean"]):
                raise RuntimeError(f"{metric} {arm}: {v.mean()} != summary {sm[f'{arm}_mean']}")
            pair.append((float(v.mean()), *cluster_ci(v, clu)))
        out.append(pair)
    return out


def metric_axis(ax, pair: list, title: str, higher_is_better: bool) -> None:
    """One metric as Figure 2 draws it: touching bars (between, within), 95% CI whiskers, rotated values above them."""
    means, tops = (np.array([p[i] for p in pair]) for i in (0, 2))
    x = bars(ax, ["between", "within"], means, [WITH, WITHOUT], ["", ""])
    for xi, (m, lo, hi) in zip(x, pair):
        whisker(ax, xi, m, lo, hi)
    room = label_room(ax, tops, [[t] for t in value_labels(ax, x, means, tops)])
    ax.set_ylim(0, float(np.max(tops / (1 - room))))
    metric_title(ax, title, higher_is_better, pad=3)
    ax.set_xticks([])


def main(data: Path = CORE / "data" / "figure2", out: Path = CORE / "figure2") -> None:
    data = data / "ED_pbmc_wells"
    layout = pd.read_csv(data / "plate_layout.csv")
    summary = json.loads((data / "well_draws_summary.json").read_text())
    ed_page.main_figure_style(lambda: page.style(label=6.5, tick=6.0))
    fig = ed_page.figure(H)
    panel_a(fig, layout)
    page.banner(fig, 92.0, 0.0, 84.0, 5.0, [[("How large is the well effect within a plate?", INK, "bold")]])
    ed_page.letter(fig, "b", 88.0, 7.0)
    schematic_b(mm_axes(fig, 96.0, 7.0, 32.0, 34.5))
    stats = between_within(summary, pd.read_csv(data / "well_draws_per_condition.csv"),
                           pd.read_csv(data / "well_draws_energy.csv"))
    ed_page.letter(fig, "c", 126.0, 7.0)
    titles = ["Top-100 DEG\noverlap", "Pearson Δ", "Energy\ndistance"]     # Figure 2's, wrapped to fit 3 columns
    if [t.replace("\n", " ") for t in titles] != [t for _, t, _ in PAIR_METRICS]:
        raise RuntimeError("c's metric titles drifted from Figure 2's")
    for i, (pair, title, (_, _, hb)) in enumerate(zip(stats, titles, PAIR_METRICS)):
        metric_axis(ed_page.axes_mm(fig, 137.0 + 15.0 * i, 14.0, 8.0, 24.0), pair, title, hb)
    for i, (c, label) in enumerate([(WITH, "Between two control wells"), (WITHOUT, "Within one control well")]):
        kax = ed_page.axes_mm(fig, 137.0, 42.0 + 3.2 * i, 2.4, 1.8)
        kax.set_axis_off()
        kax.add_patch(Rectangle((0, 0), 1, 1, fc=c, ec=INK, lw=0.35))
        fig.text((137.0 + 3.2) / ed_page.W, 1 - (42.0 + 3.2 * i + 0.9) / H, label, fontsize=5.5, va="center")
    ed_page.save(fig, out / "ed" / "ED_pbmc_wells")
    plt.close(fig)


if __name__ == "__main__":
    main()
