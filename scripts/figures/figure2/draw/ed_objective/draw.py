"""Extended Data: the training objective. GENESIS's energy score against set-level MMD (multi-scale Gaussian kernel),
both retraining Figure 3a's 25%-cells model with only the loss changed; Tahoe-100M held-out cell lines (the split-half
conditions). Four metrics in Figure 2c's style; under each, the share of conditions GENESIS wins (clustered P).

    python -m scripts.figures.figure2.draw.ed_objective.draw
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from scripts.figures import CORE  # noqa: E402
from scripts.figures.figure2.draw.panels import BAR_H  # noqa: E402
from src.figures import ed_page, page  # noqa: E402
from src.figures.bars import bars, label_room, metric_title, value_labels  # noqa: E402
from src.figures.palette import C_SPLIT, FILL, INK, REF_DASH  # noqa: E402

H, X0, W_ROW, PAD = 62.0, 4.0, 172.0, 7.0      # page height; banner's left edge and width; a column's y-tick room
REF, G, MMD = "Split-half reference", "GENESIS 25% cells", "Set-level MMD 25% cells"
# arm -> (key label, short name, fill): MMD in GENESIS's hue, lightened (same network, another loss)
ARMS = {G: ("GENESIS (energy score)", "GENESIS", FILL["GENESIS"]), MMD: ("MMD, set-level", "MMD set", "#7fa9dc")}
METRICS = [("deg_adaptivek_truth", "Top-100 DEG overlap", True, False), ("pearson_r_diff", "Pearson r Δ", True, False),
           ("energy_distance", "Energy distance (log)", False, True), ("mmd_rbf", "MMD, Gaussian (log)", False, True)]


def bar_axis(ax, means: np.ndarray, ref: float, log: bool) -> None:
    """Figure 2c's bar axis: touching bars, rotated values, the split-half reference dashed."""
    x = bars(ax, list(ARMS), means, [f for _k, _s, f in ARMS.values()], ["", ""])
    if log:
        ax.set_yscale("log")
        ax.set_ylim(min(means.min(), ref) * 0.4, means.max())
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _p: f"{v:g}"))
    room = label_room(ax, means, [[t] for t in value_labels(ax, x, means)])
    if log:
        lo = np.log10(ax.get_ylim()[0])
        ax.set_ylim(top=10 ** float(np.max(lo + (np.log10(means) - lo) / (1 - room))))
    else:
        ax.set_ylim(0, max(float(np.max(means / (1 - room))), 1.05 * ref))
    ax.axhline(ref, color=C_SPLIT, lw=0.8, ls=REF_DASH, zorder=6)
    ax.set_xticks([])


def main(data: Path = CORE / "data" / "figure2", out: Path = CORE / "figure2") -> None:
    data = data / "ED_objective"
    plotted = pd.read_csv(data / "loss_ablation_mmd_source_data_plotted.csv")
    h2h = pd.read_csv(data / "loss_ablation_mmd_source_data_genesis_vs_mmd.csv")
    p, w = plotted[plotted.panel.eq("b")], h2h[h2h.panel.eq("b")]      # the tables' panel b: split-half conditions
    ed_page.main_figure_style(lambda: page.style(label=6.5, tick=6.0))
    fig = ed_page.figure(H)
    ed_page.letter(fig, "a", 0.0, 0.0)
    page.banner(fig, X0, 0.0, W_ROW, 8.0, [[("Tahoe-100M", INK, "bold"), ("  ·  biological context generalization  ·  "
                                                                        "GENESIS on 25% of training cells", INK, "normal")],
                                           [(REF, C_SPLIT, "bold")]])
    cw = W_ROW / 4
    for i, (m, title, hb, log) in enumerate(METRICS):
        q = p[p.metric.eq(m)].set_index("arm")
        r = w[w.metric.eq(m) & w.vs.eq(MMD)]
        if q.n_units.nunique() != 1 or len(r) != 1 or r.n_units.iloc[0] != q.n_units.iloc[0]:
            raise RuntimeError(f"{m}: arms or the head-to-head are on different condition sets")
        n, r = int(q.n_units.iloc[0]), r.iloc[0]
        ax = page.axes_mm(fig, X0 + i * cw + PAD, 14.0, cw - PAD, BAR_H)
        metric_title(ax, title, hb, sub=f"n = {n:,}", pad=2)
        bar_axis(ax, q.loc[list(ARMS), "mean"].to_numpy(), float(q.loc[REF, "mean"]), log)
        pval = "P < 10⁻³⁰⁰" if r.p < 1e-300 else f"P = {ed_page.p_text(r.p)}"
        ax.text(0.5, -0.06, f"GENESIS better than {ARMS[MMD][1]}:\n{r.share:.0%} of conditions, {pval}",
                transform=ax.transAxes, ha="center", va="top", fontsize=5.5, linespacing=1.25)
    # The bars carry no x labels, so the key names them
    handles = [Patch(facecolor=f, edgecolor="none", lw=0.3, label=k) for k, _s, f in ARMS.values()]
    handles.append(Line2D([], [], color=C_SPLIT, lw=0.8, ls=REF_DASH, label=REF))
    fig.legend(handles=handles, ncol=len(handles), frameon=False, loc="upper left", handlelength=1.6,
               columnspacing=1.0, handletextpad=0.4, bbox_to_anchor=(X0 / ed_page.W, 1 - (H - 6.0) / H))
    ed_page.save(fig, out / "ed" / "ED_objective")
    plt.close(fig)


if __name__ == "__main__":
    main()
