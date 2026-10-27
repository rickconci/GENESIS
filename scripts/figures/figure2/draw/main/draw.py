"""Figure 2: GENESIS against the experiment's own repeat, and against every competitor.

a, Tahoe-100M plate pairs: each model against the experimental repeat (dashed), 3 metrics; a star marks an arm
   where neither no-gap test finds a difference from the repeat
b, the same pairs ranked by the repeat
c, held-out cell lines (Tahoe-100M) and donors (PBMC): the split-half reference next to each model
Every bar and the repeat carry 95% cluster-bootstrap CIs (whiskers; a band around the repeat in a).

    python -m scripts.figures.figure2.draw.main.draw       # core_deliverables/data/figure2 -> core_deliverables/figure2
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.figures import CORE  # noqa: E402
from scripts.figures.figure2.draw import data as d, panels as p  # noqa: E402
from src.figures import page  # noqa: E402

H, Y_C = 168.0, 58.0                            # page height; panel c's top (mm)
PANELS = [("tahoe_bio", "Tahoe-100M"), ("parse_bio", "PBMC cytokine atlas")]


def main(data: Path = CORE / "data" / "figure2", out: Path = CORE / "figure2") -> None:
    data = data / "main"
    page.style(label=6.5, tick=6.0)
    pairs = d.pairs(data / "plate_pairs")
    pc, ceil = d.conditions([data / "model_comparison" / "per_condition.parquet"],
                            [data / "model_comparison" / "ceilings.csv"])
    ci = d.cis(data / "confidence_intervals.csv")
    fig = plt.figure(figsize=(page.W / 25.4, H / 25.4), facecolor="white")
    page.letter(fig, "a", 0.0, 0.0)             # a top left, c below it on a's columns, b down the right
    p.panel_a(fig, pairs, d.tests_vs_repeat(pairs), ci, p.LEFT_X, 0.0, p.LEFT_W)
    page.letter(fig, "b", p.B_LETTER_X, 0.0)
    p.panel_b(fig, pairs, p.B_X, 0.0, p.B_W, p.B_AX_H)
    page.letter(fig, "c", 0.0, Y_C)
    p.panel_c(fig, pc, ceil, ci, PANELS, p.LEFT_X, Y_C, p.LEFT_W)
    lo, hi = page.check_text(fig)
    print(f"figure2: text {lo:g}-{hi:g} pt, no label collides")
    page.save(fig, out / "figure2")
    plt.close(fig)


if __name__ == "__main__":
    main()
