"""Extended Data: Hallmark GSEA agreement with an independent batch in the held-out cell lines.

ED_program_summary's page on Hallmark NES instead of PROGENy scores (its key and panels): every condition on two
plates at 5 uM, each model scored against the other batch.
    a  Pearson r of Hallmark NES along three axes (across pathways, across compounds, across cell lines)
    b  a's across-pathways r broken down: per held-out line, per compound, and GENESIS against experimental
       reproducibility per compound
Models: the 5,750-gene GENESIS trained on 15% of conditions, Additive on the same split, the nearest-neighbour line.
Spearman is in the same tables and not drawn.

    python -m scripts.figures.figure4.draw.ed_tahoe_hallmark_gsea.draw
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.figures import CORE
from scripts.figures.figure4.draw import panels as F
from scripts.figures.figure4.draw.ed_program_summary.draw import key, row_axes, row_marginals
from src.figures import ed_page

H = 86.0
ARMS = ["genesis", "additive", "copy_closest_15"]


def main(data: Path = CORE / "data" / "figure4", out: Path = CORE / "figure4") -> None:
    d = data / "ED_tahoe_hallmark_gsea"
    t, m = pd.read_csv(d / "axis_ci.csv"), pd.read_csv(d / "margins.csv")
    t, m = t[t.method.eq("pearson")].copy(), m[m.method.eq("pearson")].copy()
    if t.empty or m.empty:
        raise RuntimeError(f"{d} has no pearson rows")
    ed_page.main_figure_style(F.style)
    fig = ed_page.figure(H)
    key(fig, ARMS, 6.0, 0.5)
    row_axes(fig, t, ARMS, 6.0, lab="a", ylabel="Pearson r")
    row_marginals(fig, m, ARMS, 44.0, lab="b", ylabel="Pearson r across pathways")
    ed_page.save(fig, out / "ed" / "ED_tahoe_hallmark_gsea")


if __name__ == "__main__":
    main()
