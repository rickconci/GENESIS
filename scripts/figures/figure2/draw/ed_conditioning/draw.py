"""Extended Data: GENESIS uses every covariate it is given; it does not just learn batch effects.

Tahoe-100M, held-out cell lines: GENESIS given the plate alone (batch only), the plate and cell line, the plate,
compound and concentration, or all of them. Bars are condition means ranked best first (no error bars, as
Figure 2c), beside the split-half reference.

    python -m scripts.figures.figure2.draw.ed_conditioning.draw
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
from src.figures.palette import C_SPLIT, GRID, INK, REF_DASH, hatch_ink  # noqa: E402

H = 52.0                                          # one row: Fig. 2c's one-line header, bars, key
# 9.5 mm bars and a 30 mm bar, the same as ED objective, between this row's old 11 mm bars on
# 172 mm and that panel's old 7 mm bars on 104 mm. Four touching bars occupy 4.4 data units.
BAR_H, PAD, AX_W = 30.0, 8.5, 41.8
ROW = 3 * AX_W + 3 * PAD                      # left pad and two gaps: 151 mm
TITLES = {"deg": "Top-100 DEG overlap", "pearson_r_diff": "Pearson r Δ", "energy_distance": "Energy distance"}
# (letter, top mm, folder, arm -> display name, key's top mm)
ROWS = (("a", 0.0, "tahoe", {"plate": "Plate only (batch)", "plate + bio": "Plate + cell line",
                             "plate + pert": "Plate + compound, concentration", "plate + bio + pert": "All covariates"},
         42.0),)


def load(d: Path, rename: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(per-condition scores, split-half reference per metric, arm styles). Bars only."""
    arms = pd.read_csv(d / "arms.csv", keep_default_na=False).replace({"arm": rename})
    arms = arms[arms.role.eq("bar")]
    if missing := set(rename.values()) - set(arms.arm):
        raise RuntimeError(f"{d.name}: arms not found after renaming: {sorted(missing)}")
    pc = pd.read_parquet(d / "per_condition.parquet").replace({"arm": rename})
    pc = pc[pc.arm.isin(arms.arm)]
    if pc.space.nunique() > 1:
        raise RuntimeError(f"{d.name}: more than one scoring space; this figure draws one")
    return pc, pd.read_csv(d / "reference.csv"), arms


def metric_row(fig, pc: pd.DataFrame, ref: pd.DataFrame, arms: pd.DataFrame, y0: float) -> None:
    """Fig. 2c's header, then one axis per metric: bars ranked best first, the split-half reference dashed in its
    repeat-s.d. band. Condition sets still have to match across arms (DEG is undefined where truth calls none)."""
    style = arms.set_index("arm")
    # Fig. 2c: one 4.6 mm line, dataset · reference, axes 8 mm down.
    page.banner(fig, 4.0, y0, ROW, 4.6, [[("Tahoe-100M", INK, "bold"), ("  ·  ", INK, "normal"),
                                          ("Split-half reproducibility", C_SPLIT, "bold")]])
    for i, m in enumerate(ref.itertuples()):
        ax = page.axes_mm(fig, 4.0 + PAD + i * (AX_W + PAD), y0 + 8.0, AX_W, BAR_H)
        stat = pc[pc.metric_key.eq(m.metric_key)].groupby("arm", sort=False)["value"].agg(["mean", "count"])
        stat = stat.sort_values("mean", ascending=not m.higher_is_better)
        if stat["count"].nunique() != 1:
            raise ValueError(f"{m.metric_key}: arms cover different condition counts {dict(stat['count'])}")
        fills, hatches = list(style.loc[stat.index, "fill"]), list(style.loc[stat.index, "hatch"])
        x = np.arange(len(stat), dtype=float)
        ax.bar(x, stat["mean"], width=1.0, color=fills, hatch=hatches, linewidth=0.3, zorder=3,
               edgecolor=[hatch_ink(f) if h else "white" for f, h in zip(fills, hatches)])
        ax.set_xlim(-0.7, len(x) - 0.3)
        ax.yaxis.grid(True, color=GRID, lw=0.4, zorder=0)
        ax.set_axisbelow(True)
        r, sd = m.split_half_reference_mean, m.split_half_reference_repeat_sd
        span = [*stat["mean"], r, r + sd]
        if m.log_axis:
            ax.set_yscale("log")
            ax.set_ylim(min(span) * 0.4, max(span) * 3.0)
            # Plain decimals: 10^x superscripts set at 70% fall under Nature's 5 pt floor
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _p: f"{v:g}"))
        else:
            ax.set_ylim(0, max(span) * 1.18)
        ax.axhspan(max(r - sd, ax.get_ylim()[0]), r + sd, color=C_SPLIT, alpha=0.12, lw=0, zorder=1)
        ax.axhline(r, color=C_SPLIT, lw=0.8, ls=REF_DASH, zorder=6)
        lo, hi = ax.get_ylim()
        for xi, v in zip(x, stat["mean"]):
            ax.text(xi, v * 1.15 if m.log_axis else v + 0.015 * (hi - lo), f"{v:.3g}", ha="center",
                    va="bottom", rotation=90, fontsize=5, color=INK, zorder=7)
        ax.set_xticks([])
        leaf = "deg" if m.metric_key.startswith("deg") else m.metric_key.split("/")[-1]
        ax.set_title(f"{TITLES[leaf]} {'↑' if m.higher_is_better else '↓'}", pad=3)


def legend(fig, arms: pd.DataFrame, y: float) -> None:
    """Bar order changes per metric, so the key, not the position, names a bar."""
    handles = [Patch(facecolor=a.fill, hatch=a.hatch, edgecolor=hatch_ink(a.fill) if a.hatch else "none", lw=0.3,
                     label=a.arm) for a in arms.itertuples()]
    handles.append(Line2D([], [], color=C_SPLIT, lw=0.8, ls=REF_DASH,
                          label="Split-half reproducibility (± repeat s.d.)"))
    fig.legend(handles=handles, ncol=len(handles), frameon=False, loc="upper left", handlelength=1.6,
               columnspacing=1.0, handletextpad=0.4, labelspacing=0.35,
               bbox_to_anchor=(6.0 / page.W, 1 - y / (fig.get_figheight() * 25.4)))


def main(data: Path = CORE / "data" / "figure2", out: Path = CORE / "figure2") -> None:
    ed_page.main_figure_style(page.style)
    fig = ed_page.figure(H)
    for letter, y0, sub, rename, key_y in ROWS:
        pc, ref, arms = load(data / "ED_conditioning" / sub, rename)
        if ref.split_half_reference_mean.isna().any():
            raise ValueError(f"{sub}: a metric has no split-half reference")
        ed_page.letter(fig, letter, 0.0, y0)
        metric_row(fig, pc, ref, arms, y0)
        legend(fig, arms, key_y)
    ed_page.save(fig, out / "ed" / "ED_conditioning")
    plt.close(fig)


if __name__ == "__main__":
    main()
