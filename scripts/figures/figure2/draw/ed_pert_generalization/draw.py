"""Extended Data: GENESIS on held-out perturbations, Figure 2a-c on the perturbation splits, drawn by Figure 2's
own panels in its layout. What differs is in the data, not the drawing:
  - plate-pair sets are per metric: a pair counts for a metric when the repeat and every arm define it, so DEG
    overlap has fewer pairs, and the tests run per metric;
  - PBMC CellFlow (cell types by the authors' 1-NN label transfer) joins c's PBMC bars, so its per-metric sets
    drop IL-18Ra and Megalin, which CellFlow cannot take as input (no ESM2 embedding).

    python -m scripts.figures.figure2.draw.ed_pert_generalization.draw
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.figures import CORE  # noqa: E402
from scripts.figures.figure2.draw import data as d, panels as p  # noqa: E402
from src.figures import ed_page, page  # noqa: E402

H, Y_C = 168.0, 58.0
B_W = 45.0                                      # Figure 2b's column, narrowed to the 180 mm page
PANELS = [("tahoe_pert_fixed", "Tahoe-100M"), ("parse_pert_fixed", "PBMC cytokine atlas")]


def main(data: Path = CORE / "data" / "figure2", out: Path = CORE / "figure2") -> None:
    data = data / "ED_pert_generalization"
    cond = data / "ed_pert_generalization"
    pairs = d.pert_pairs(data / "ed_pert_vs_repeat", data / "tahoe_pert_overlap" / "cross_batch" / "benchmark_pairs.csv")
    pc, ceil = d.conditions([cond / f"{k}__per_condition.parquet" for k, _ in PANELS],
                            [cond / f"{k}__ceilings.csv" for k, _ in PANELS])
    ed_page.main_figure_style(lambda: page.style(label=6.5, tick=6.0))
    ci = d.cis(cond / "confidence_intervals.csv")
    fig = ed_page.figure(H)
    ed_page.letter(fig, "a", 0.0, 0.0)
    p.panel_a(fig, pairs, d.tests_vs_repeat(pairs), ci, p.LEFT_X, 0.0, p.LEFT_W)
    ed_page.letter(fig, "b", p.B_LETTER_X, 0.0)
    p.panel_b(fig, pairs, p.B_X, 0.0, B_W, p.B_AX_H)
    ed_page.letter(fig, "c", 0.0, Y_C)
    p.panel_c(fig, pc, ceil, ci, PANELS, p.LEFT_X, Y_C, p.LEFT_W)
    ed_page.save(fig, out / "ed" / "ED_pert_generalization")
    plt.close(fig)


if __name__ == "__main__":
    main()
