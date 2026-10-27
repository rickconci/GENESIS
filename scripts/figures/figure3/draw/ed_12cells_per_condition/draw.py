"""Extended Data: how many cells a condition needs, on Tahoe-100M.

a, training conditions (plates 6 and 14): GENESIS on all cells vs on 25 cells per combination (~12 per plate), and
   that model's own ~12 training cells, drawn or copy-pasted as the prediction; against the experimental repeat.
b, Figure 3a's 530 held-out plate pairs (500 combinations): the same two models against the repeat.
c, Figure 3a's uniform cell draw, 1% to 100% of training cells (its published table; nothing is rescored).
95% CIs are Figure 2's cluster bootstrap over cell line x drug, one paired draw per metric.

    python -m scripts.figures.figure3.draw.ed_12cells_per_condition.draw
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from scripts.figures import CORE  # noqa: E402
from scripts.figures.figure3.draw import panels as p  # noqa: E402
from src.figures import ed_page, page  # noqa: E402
from src.figures.bars import whisker  # noqa: E402
from src.figures.palette import C_REPEAT, C_SPLIT, FILL  # noqa: E402
from src.stats.bootstrap import cluster_bootstrap_ci  # noqa: E402

H, AX_Y, AX_H = 129.0, 9.0, 24.0                   # page height; a-b's axes below their banner's top (mm)
METRICS = (("deg_adaptivek_truth", "Top-100 DEG overlap ↑"), ("pearson_r_diff", "Pearson Δ ↑"),
           ("energy_distance", "Energy distance ↓"))
C_DRAWN, C_COPY = "#8c510a", "#e6ab02"             # the training-cell arms: off the model palette
# a's four bars; b draws the first two. The 25-cell model is named by its per-plate depth, as its comparators are
ARMS = (("full", "GENESIS,\nall cells", FILL["GENESIS"]), ("pc25", "GENESIS,\n~12 cells", p.CELLS),
        ("drawn", "its ~12\ncells", C_DRAWN), ("copy", "copy-\npasted", C_COPY))
REPEAT = "Experimental reproducibility"


# ----------------------------------------------------------------------------
# Build the tables: one per metric, rows x arms, keyed "cell line \t drug \t dose[ \t plates]"
# ----------------------------------------------------------------------------

def both_directions(raw: pd.DataFrame, cohort: str, metric: str, pairs: bool = False) -> pd.DataFrame:
    """One row per direction, kept only when both directions of the unit are finite. Training: the unit is the
    combination (plates 6 and 14, so two directions). Test: the plate pair, Figure 3a's; a combination on three
    plates is three pairs, so requiring two directions per combination would drop it."""
    need = ["genesis", "repeat"] + (["drawn", "copy"] if cohort == "cross" else [])
    w = raw[raw.cohort.eq(cohort) & raw.arm.isin(need)].pivot_table(
        index=["cid", "cell_name", "drug", "dose", "plate", "plate_y"], columns="arm", values=metric,
        aggfunc="mean").dropna(subset=need).reset_index()
    w["combo"] = w.cell_name + "\t" + w.drug + "\t" + w.dose.astype(str)
    if not pairs:
        return w[w.groupby("combo").cid.transform("size").eq(2)]
    a, b = (w[c].astype(str).str.removeprefix("plate").astype(int) for c in ("plate", "plate_y"))
    w["plate_i"], w["plate_j"] = np.minimum(a, b), np.maximum(a, b)
    w["pair"] = w.combo + "\t" + w.plate_i.astype(str) + "\t" + w.plate_j.astype(str)
    return w[w.groupby("pair").cid.transform("size").eq(2)]


def training(data: Path, raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """a: both models and the repeat on plates 6/14 (all of A vs all of B), joined with the 25-cell model's own
    training cells (both directions averaged). Each table keeps the combinations every arm has."""
    rep = pd.read_parquet(data / "train614_repeat.parquet")
    own = raw[raw.cohort.eq("cross") & raw.arm.isin(["drawn", "copy"])]
    rep, own = (d.assign(combo=d.cell_name + "\t" + d.drug + "\t" + d.dose.astype(float).map("{:g}".format))
                for d in (rep, own))
    out = {}
    for metric, _title in METRICS:
        w = rep.pivot_table(index="combo", columns="model", values=f"cross_mean__{metric}")
        tt = rep.pivot_table(index="combo", columns="model", values=f"true_true__{metric}")
        if not np.allclose(tt["full"], tt["pc25"], equal_nan=True):
            raise RuntimeError(f"train614 {metric}: the two models' repeat (real vs real) differ")
        g = own.groupby(["combo", "arm"])[metric]
        cells = g.mean().where(g.count().eq(2)).unstack("arm")          # both directions finite
        out[metric] = w.join(tt["full"].rename("repeat")).join(cells[["drawn", "copy"]]).dropna()
    return out


def test(data: Path, raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """b: Figure 3a's 530 plate pairs (500 combinations): the 25-cell model and the repeat, joined on line,
    compound, dose and both plates with Figure 3a's all-cells GENESIS. Pearson's repeat must equal Figure 3a's;
    DEG's is one direction there, so the bars' both-directions repeat stays."""
    f3a = pd.read_csv(data / "figure3a_source_data_per_pair.csv")
    f3a = f3a.assign(dose_f=f3a.dose.astype(float), lo=f3a[["plate_i", "plate_j"]].min(axis=1),
                     hi=f3a[["plate_i", "plate_j"]].max(axis=1))
    out = {}
    for metric, _title in METRICS:
        w = both_directions(raw, "test_cross", metric, pairs=True)
        out[metric] = w.groupby("pair", sort=False)[["genesis", "repeat"]].mean().rename(columns={"genesis": "pc25"})
        if len(out[metric]) != 530 or (n := w[["cell_name", "drug", "dose"]].drop_duplicates().shape[0]) != 500:
            raise RuntimeError(f"test {metric}: {len(out[metric])} pairs from {n} combinations; Figure 3a is 530 from 500")
    w = both_directions(raw, "test_cross", "pearson_r_diff", pairs=True)
    pair = (w.assign(dose_f=w.dose.astype(float)).groupby("pair", sort=False)
            [["cell_name", "drug", "dose_f", "plate_i", "plate_j"]].first().reset_index()
            .rename(columns={"plate_i": "lo", "plate_j": "hi"}))
    m = pair.merge(f3a, on=["cell_name", "drug", "dose_f", "lo", "hi"], how="left", indicator=True).set_index("pair")
    if len(pair) != len(f3a) or len(m) != len(pair) or not m._merge.eq("both").all() or m.index.has_duplicates:
        raise RuntimeError(f"Figure 3a join: {int(m._merge.eq('both').sum())} of {len(pair)} pairs, it has {len(f3a)}")
    ours = out["pearson_r_diff"]["repeat"]
    if (m["Experimental repeat__pearson_r_diff"].reindex(ours.index) - ours).abs().max() > 1e-3:
        raise RuntimeError("test Pearson repeat disagrees with Figure 3a on the joined pairs")
    for metric, c in out.items():
        c["full"] = m[f"GENESIS__{metric}"].reindex(c.index)
        if c["full"].isna().any():
            raise RuntimeError(f"test {metric}: {int(c['full'].isna().sum())} pairs lack Figure 3a's GENESIS")
    return out


def cis(c: pd.DataFrame, cols: list[str]) -> dict[str, tuple[float, float, float]]:
    """{column: (mean, lo, hi)} from one paired draw, clustered on the key's cell line and drug."""
    cl = c.index.str.split("\t").str[:2].str.join("|").to_numpy()
    mean, lo, hi = cluster_bootstrap_ci(c[cols].to_numpy(dtype=float), cl)
    if np.abs(mean - c[cols].mean().to_numpy()).max() > 1e-12:
        raise RuntimeError("bootstrap mean is not the plotted mean")
    return {k: (float(m), float(a), float(b)) for k, m, a, b in zip(cols, mean, lo, hi)}


# ----------------------------------------------------------------------------
# Then a and b: a row of bars per setting, the repeat dashed in its CI band
# ----------------------------------------------------------------------------

def bars(ax, arms: tuple, est: dict, name_ref: bool) -> None:
    """Means with 95% whiskers on an axis as wide as a's four bars; the repeat named in a row's first axes."""
    rep = est["repeat"]
    top = max([est[k][2] for k, *_ in arms] + [rep[2]]) * 1.4
    for i, (k, _label, color) in enumerate(arms):
        v, lo, hi = est[k]
        ax.bar(i, v, color=color, width=0.62, linewidth=0, zorder=2)
        whisker(ax, i, v, lo, hi, capsize=1.4, zorder=7)
        if v > 0.15 * top:          # gold and light blue are too light for white type
            ax.text(i, v * 0.45, f"{v:.2f}", ha="center", va="center", fontsize=5.5, zorder=3,
                    color="#1b1d1f" if color in (C_COPY, p.CELLS) else "#ffffff")
        else:
            ax.text(i, hi + 0.03 * top, f"{v:.2f}", ha="center", va="bottom", fontsize=5.5, zorder=3)
    ax.axhspan(rep[1], rep[2], color=C_REPEAT, alpha=0.14, lw=0, zorder=1)
    ax.axhline(rep[0], color=C_REPEAT, lw=0.8, ls=(0, (4, 2.5)), zorder=4)
    if name_ref:
        ax.legend(handles=[Line2D([0], [0], color=C_REPEAT, lw=0.8, ls=(0, (4, 2.5)), label=REPEAT)],
                  loc="upper right", frameon=False, fontsize=5.5, handlelength=2.4, borderaxespad=0.2)
    ax.set_xticks(range(len(arms)), [label for _k, label, _c in arms], fontsize=5.5)
    pad = 0.5 + (len(ARMS) - len(arms)) / 2
    ax.set_xlim(-0.5 - pad, len(arms) - 0.5 + pad)
    ax.set_ylim(0, top)
    ax.yaxis.grid(True, color=p.GRID, lw=0.4)
    ax.set_axisbelow(True)


def bar_row(fig, letter: str, y0: float, setting: str, tables: dict, arms: tuple) -> None:
    page.letter(fig, letter, 1, y0)
    ed_page.banner_line(fig, y0, "Tahoe-100M", setting, REPEAT, C_REPEAT, x=5.0, w=174.0)
    for j, (metric, title) in enumerate(METRICS):
        ax = page.axes_mm(fig, 12 + 58 * j, y0 + AX_Y, 50, AX_H)
        bars(ax, arms, cis(tables[metric], [k for k, *_ in arms] + ["repeat"]), name_ref=j == 0)
        ax.set_title(title, pad=2)


# ----------------------------------------------------------------------------
# Then c: the cells series of Figure 3a's uniform-draw variant, the two closest other arms across
# ----------------------------------------------------------------------------

def uniform_metric(ax, t: pd.DataFrame, title: str) -> None:
    """One metric from the variant's published table; refused unless it is 1, 5, 25, 100% with intervals
    around their means, and two other arms."""
    cells = t[t.series.eq("fewer cells")].sort_values("pct_kept")
    comps = t[~t.series.isin(["fewer cells", "fewer conditions", "Split-half"])]
    mean, lo, hi = (cells[c].to_numpy(float) for c in ("mean", "ci95_lo", "ci95_hi"))
    if cells.pct_kept.tolist() != [1, 5, 25, 100] or not (lo <= mean).all() or not (mean <= hi).all() or len(comps) != 2:
        raise RuntimeError(f"{title}: not 1/5/25/100% cells with intervals and two other arms")
    x = np.arange(len(cells))
    ys = np.r_[lo, hi, comps.ci95_lo, comps.ci95_hi]
    pad = 0.12 * (ys.max() - ys.min())
    ylim = (float(ys.min() - pad), float(ys.max() + pad))
    dashes, span = iter([":", (0, (3, 1.5))]), ylim[1] - ylim[0]
    for r in comps.itertuples():
        color = p.MUTED if r.series in p.BASELINES else FILL[r.series]
        ax.axhspan(r.ci95_lo, r.ci95_hi, color=color, alpha=0.18, lw=0, zorder=1)
        ax.axhline(r.mean, ls=next(dashes) if r.series in p.BASELINES else ":", lw=0.8, color=color, zorder=2)
        ax.plot(x[-1], r.mean, "o", color=color, ms=0.75 * 1.6, zorder=2)
        under = r.mean == float(comps["mean"].min())
        ax.text(x[-1] - 0.15, r.mean + (-0.022 if under else 0.018) * span, r.series, color=color,
                ha="right", va="top" if under else "bottom", fontsize=5.5, clip_on=False)
    ax.plot(x, mean, color=p.CELLS, lw=1.0, zorder=3)
    ax.plot(x, mean, ls="none", marker="s", color=p.CELLS, ms=1.6, mew=0, zorder=4)
    above = mean[-1] > float(comps["mean"].max())     # named on the side facing the other arms
    ax.text(x[-1] - 0.15, lo[-1] - 0.03 * span if above else hi[-1] + 0.03 * span, "GENESIS", color=p.CELLS,
            ha="right", va="top" if above else "bottom", fontsize=5.5)
    whisker(ax, x, mean, lo, hi, zorder=7)
    whisker(ax, np.full(len(comps), x[-1]), comps["mean"], comps.ci95_lo, comps.ci95_hi, zorder=7)
    ax.set_xlim(-0.45, x[-1] + 0.28)
    ax.set_ylim(*ylim)
    ax.set_yticks([t for t in mticker.MaxNLocator(nbins=4).tick_values(*ylim) if ylim[0] <= t <= ylim[1]])
    ax.grid(axis="y", color=p.GRID, lw=0.4)
    ax.set_title(title, pad=3)
    p.pert_ticks(ax, list(x), cells.n_train.tolist(), label="Training cells")   # controls are drawn too


def main(data: Path = CORE / "data" / "figure3", out: Path = CORE / "figure3") -> None:
    data = data / "ED_12cells_per_condition"
    ed_page.main_figure_style(page.style)
    fig = ed_page.figure(H)
    pc = data / "ed_cells_per_combination"
    raw = pd.read_parquet(pc / "percond25_pc25.parquet", columns=["cohort", "cid", "cell_name", "drug", "dose",
                                                                  "plate", "plate_y", "arm", *dict(METRICS)])
    bar_row(fig, "a", 0.0, "Training conditions", training(pc, raw), ARMS)
    bar_row(fig, "b", 41.0, "Test conditions", test(pc, raw), ARMS[:2])
    page.letter(fig, "c", 1, 82.0)
    ed_page.banner_line(fig, 82.0, "Tahoe-100M", "Fewer training cells", "Split-half reproducibility", C_SPLIT,
                        x=5.0, w=174.0)
    uniform = pd.read_csv(data / "ed_cell_sampling" / "figure3a_uniform_cells_split_source_data_plotted.csv")
    for x, (metric, title) in zip((16.0, 74.0, 132.0), METRICS):     # no schematic: the metrics take the page
        uniform_metric(page.axes_mm(fig, x, 82.0 + 9, 40.0, 27.0), uniform[uniform.metric.eq(metric)], title)
    ed_page.save(fig, out / "ed" / "ED_12cells_per_condition")
    plt.close(fig)


if __name__ == "__main__":
    main()
