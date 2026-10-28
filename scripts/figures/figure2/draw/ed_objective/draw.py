"""Extended Data: the training objective. GENESIS's energy score against set-level MMD (multi-scale Gaussian kernel),
both retraining Figure 3a's 25%-cells model with only the loss changed; Tahoe-100M held-out cell lines, drawn as
Figure 2c (the split-half reference and its repeat s.d.).

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
from src.figures import ed_page, page  # noqa: E402
from src.figures.bars import bars, label_room, metric_title, value_labels  # noqa: E402
from src.figures.palette import C_SPLIT, FILL, INK, REF_DASH  # noqa: E402

H, X0 = 52.0, 4.0                          # one header, the bars, the key; same page height as ED conditioning
# 9.5 mm bars, between this panel's old 7 mm (a 104 mm row) and ED conditioning's old 11 mm
# (a 172 mm row). Two touching bars occupy 2.4 data units, so an axis is 22.8 mm; "0.0001" needs the pad.
BAR_H, PAD, AX_W = 30.0, 8.5, 22.8
COL = PAD + AX_W                              # 31.3; four columns span 125 mm
REF, G, MMD = "Split-half reference", "GENESIS 25% cells", "Set-level MMD 25% cells"
# arm -> (key label, short name, fill): MMD in GENESIS's hue, lightened (same network, another loss)
ARMS = {G: ("GENESIS (energy score)", "GENESIS", FILL["GENESIS"]), MMD: ("MMD, set-level", "MMD set", "#7fa9dc")}
METRICS = [("deg_adaptivek_truth", "Top-100 DEG overlap", True, False), ("pearson_r_diff", "Pearson Δ", True, False),
           ("energy_distance", "Energy distance", False, True), ("mmd_rbf", "MMD, Gaussian", False, True)]


def bar_axis(ax, means: np.ndarray, ref: float, sd: float, log: bool) -> None:
    """Figure 2c's bar axis: touching bars, the split-half reference dashed inside its repeat-s.d. band."""
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
        ax.set_ylim(0, max(float(np.max(means / (1 - room))), 1.05 * (ref + sd)))
    ax.axhspan(max(ref - sd, ax.get_ylim()[0]), ref + sd, color=C_SPLIT, alpha=0.12, lw=0, zorder=1)
    ax.axhline(ref, color=C_SPLIT, lw=0.8, ls=REF_DASH, zorder=6)
    ax.set_xticks([])


def main(data: Path = CORE / "data" / "figure2", out: Path = CORE / "figure2") -> None:
    data = data / "ED_objective"
    plotted = pd.read_csv(data / "loss_ablation_mmd_source_data_plotted.csv")
    p = plotted[plotted.panel.eq("b")]                             # the table's panel b: split-half conditions
    ed_page.main_figure_style(lambda: page.style(label=6.5, tick=6.0))
    fig = ed_page.figure(H)
    ed_page.letter(fig, "a", 0.0, 0.0)
    # Figure 2c's one-line header: the banner ends on the last axis, as that panel's does
    page.banner(fig, X0, 0.0, 3 * COL + PAD + AX_W, 4.6, [[("Tahoe-100M", INK, "bold"), ("  ·  ", INK, "normal"),
                                                          ("Split-half reproducibility", C_SPLIT, "bold")]])
    for i, (m, title, hb, log) in enumerate(METRICS):
        q = p[p.metric.eq(m)].set_index("arm")
        if q.n_units.nunique() != 1:
            raise RuntimeError(f"{m}: arms cover different condition counts {dict(q.n_units)}")
        sd = float(q.loc[REF, "split_half_repeat_std"])
        if not np.isfinite(sd):
            raise RuntimeError(f"{m}: split-half repeat s.d. missing")
        ax = page.axes_mm(fig, X0 + i * COL + PAD, 8.0, AX_W, BAR_H)
        metric_title(ax, title, hb, pad=3)
        bar_axis(ax, q.loc[list(ARMS), "mean"].to_numpy(), float(q.loc[REF, "mean"]), sd, log)
    # The bars carry no x labels, so the key names them
    handles = [Patch(facecolor=f, edgecolor="none", lw=0.3, label=k) for k, _s, f in ARMS.values()]
    handles.append(Line2D([], [], color=C_SPLIT, lw=0.8, ls=REF_DASH, label="Split-half reproducibility (± repeat s.d.)"))
    fig.legend(handles=handles, ncol=len(handles), frameon=False, loc="upper left", handlelength=1.6,
               columnspacing=1.0, handletextpad=0.4, bbox_to_anchor=(X0 / ed_page.W, 1 - (8.0 + BAR_H + 2.0) / H))
    ed_page.save(fig, out / "ed" / "ED_objective")
    plt.close(fig)


if __name__ == "__main__":
    main()
