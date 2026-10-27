"""Figure 4: two held-out Tahoe lines (the core pair, main/stories/core_pair.tsv), measured over GENESIS, on the
depth-adjusted PROGENy pathways x the axis drugs, the curated responses boxed where the test scores confirm them.

    python -m scripts.figures.figure4.draw.main.draw
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.figures import CORE  # noqa: E402
from scripts.figures.figure4.draw.main import layout as C  # noqa: E402
from scripts.figures.figure4.draw import data as D  # noqa: E402
from scripts.figures.figure4.draw import panels as F  # noqa: E402
from src.figures import page  # noqa: E402


def main(data: Path = CORE / "data" / "figure4", out: Path = CORE / "figure4") -> None:
    d = data / "main"
    F.style()
    axis = D.axis_table(d / "drug_axis/drug_axis.tsv")
    S = D.dataset(d / "tahoe_progeny_depth", axis, ["genesis"])
    pathways, _groups = F.pathway_rows(S["progs"])
    curated = C.curated_table(d / "stories/curated_picks.tsv", set(S["cond_order"]), pathways)
    fixed = C.core_pair(d / "stories/core_pair.tsv", curated)
    safe = {v: k for k, v in S["disp"].items()}
    lines = (safe[fixed.line_a.iat[0]], safe[fixed.line_b.iat[0]])
    if lines not in combinations(S["held"], 2):
        raise RuntimeError(f"core pair {lines}: not a pair of held-out lines, in this order")
    # Each axis drug's drawn class (the brackets), which the candidates carry
    drugs = {d: name for name, a, b in S["cond_groups"] for d in S["cond_order"][a:b + 1]}
    cand = C.candidates(pd.read_parquet(d / "context_analysis/test_scores.parquet"),
                        D.drawn_values(S).loc["truth"], tuple(S["disp"][c] for c in lines), drugs, pathways)
    fig = C.core(S, lines, C.fixed_rule(cand, fixed), D.multi_plate(axis))
    page.save(fig, out / "figure4", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
