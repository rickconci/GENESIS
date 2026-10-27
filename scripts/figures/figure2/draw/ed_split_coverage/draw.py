"""Extended Data: the four benchmark splits, as transposes of one design.

a, Tahoe-100M, biological-context generalization (State's split): 5 evaluation cell lines
b, Tahoe-100M, perturbation generalization: 40 evaluation compounds
c, PBMC cytokine atlas, biological-context generalization (State's split): 4 evaluation donors
d, PBMC cytokine atlas, perturbation generalization: 30 evaluation cytokines

Each panel is the split's grid: contexts are rows, perturbations columns (a Tahoe column is one compound at one
concentration, so no column mixes sets). In a and c the evaluation contexts are the top rows, and the columns run
from the perturbations they train on to those they are tested on. In b and d the evaluation perturbations are the
right-hand columns: tested in the top rows, trained in k of the bottom rows. There the bottom block is sorted per
column, training at the bottom, so it reads as a staircase over k; its rows are therefore not one context each.
The membership table is written with the code training uses.

    python -m scripts.figures.figure2.draw.ed_split_coverage.draw
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
from src.figures import ed_page  # noqa: E402
from src.figures.palette import INK  # noqa: E402

H = 170.0
L, R = 33.0, 40.0                             # row-label margins left and right (mm)
TOP = {"bio": 10.0, "pert": 7.0}              # header above each grid: bio bracket labels take two lines
# (letter, split, dataset, task, kind, contexts, perturbations, top y, grid height mm)
PANELS = [("a", "tahoe_bio", "Tahoe-100M", "Biological-context generalization", "bio", "cell lines", "compounds", 0, 36),
          ("b", "tahoe_pert", "Tahoe-100M", "Perturbation generalization", "pert", "cell lines", "compounds", 52, 36),
          ("c", "parse_bio", "PBMC cytokine atlas", "Biological-context generalization", "bio", "donors", "cytokines", 101, 20),
          ("d", "parse_pert", "PBMC cytokine atlas", "Perturbation generalization", "pert", "donors", "cytokines", 137, 19)]
# One colour per set, none of them a model's. Validation is neither trained on (splits.py drops it from training)
# nor scored (every model is taken at its last checkpoint). The control is train data, kept as its own column; the
# 21 unmeasured Tahoe cells (four columns) are drawn as train, with no key
ORDER = ["control", "train", "validation", "test"]   # bio column order, by set in the evaluation contexts
COLOR = {"train": "#d9d7d0", "test": "#a3203a", "validation": "#ffffff"}
COLOR["control"] = COLOR["not measured"] = COLOR["train"]
LEGEND = {"Train": COLOR["train"], "Test": COLOR["test"], "Validation": COLOR["validation"]}
DOTTED = dict(color=INK, alpha=0.6, lw=0.4, ls=(0, (1, 1.5)))


def cap(t: str) -> str:
    """Sentence case, leaving the rest as written."""
    return t[:1].upper() + t[1:]


def line(fig, xs: list[float], ys: list[float]) -> None:
    fig.add_artist(Line2D([x / ed_page.W for x in xs], [1 - y / H for y in ys], transform=fig.transFigure,
                          color=INK, lw=0.4))


def text(fig, x: float, y: float, t: str, **kw):
    return fig.text(x / ed_page.W, 1 - y / H, t, fontsize=5.5, **kw)


# ----------------------------------------------------------------------------
# The grid: contexts x (perturbation, concentration), coloured by set
# ----------------------------------------------------------------------------

def grid(c: pd.DataFrame, kind: str) -> tuple[pd.DataFrame, int, pd.Series]:
    """The split's sets as a context x column table, ordered as in the sketch, with the number of evaluation-context
    rows on top and each column's group (bio: its set in the evaluation contexts; pert: whether it is an evaluation
    perturbation), indexed by perturbation."""
    c = c.assign(col=c.perturbation + "|" + c.dose.astype(str))
    pert_of = c.drop_duplicates("col").set_index("col")["perturbation"]
    top_ctx = sorted(set(c.context[c.set == "test"]), key=str.casefold)
    rows = top_ctx + sorted(set(c.context) - set(top_ctx), key=str.casefold)
    st = c.pivot(index="context", columns="col", values="set").reindex(rows).fillna("not measured")
    ctrl = sorted(set(c.col[c.control]))                 # one column per split, in train for every context
    st[ctrl] = st[ctrl].where(st[ctrl] != "train", "control")
    top = st.loc[top_ctx]
    if kind == "bio":
        seen = {q: set(top[q]) - {"not measured"} for q in st.columns}
        if any(len(v) != 1 for v in seen.values()):
            raise ValueError("the evaluation contexts do not share one set per column")
        group = pd.Series({q: v.pop() for q, v in seen.items()})
        cols = sorted(st.columns, key=lambda q: (ORDER.index(group[q]), pert_of[q].casefold(), q))
    else:
        group = top.eq("test").any().map({True: "evaluation", False: "training"})
        group[ctrl] = "control"
        k = st.eq("train").sum()
        cols = sorted(st.columns, key=lambda q: (group[q] == "evaluation", group[q] != "control", -k[q],
                                                 pert_of[q].casefold(), q))
    st, group = st[cols], group[cols].rename(index=pert_of)
    if kind == "pert":                      # each evaluation column's training contexts at the bottom: a staircase
        for j in np.flatnonzero(group.eq("evaluation").to_numpy()):
            st.iloc[len(top_ctx):, j] = sorted(st.iloc[len(top_ctx):, j], key=lambda v: v == "train")
    return st, len(top_ctx), group


def runs(keys) -> list[tuple]:
    """(key, first, last) for each run of equal consecutive keys."""
    keys, out, start = list(keys), [], 0
    for i in range(1, len(keys) + 1):
        if i == len(keys) or keys[i] != keys[start]:
            out.append((keys[start], start, i - 1))
            start = i
    return out


# ----------------------------------------------------------------------------
# Then a panel: the grid, brackets naming its groups, region names, and axis labels
# ----------------------------------------------------------------------------

def panel(fig, c: pd.DataFrame, kind: str, y: float, ht: float, head: tuple, ctxs: str, perts: str) -> None:
    st, n_top, group = grid(c, kind)
    nr, nc = st.shape
    hx, hy, hw = L, y + TOP[kind], ed_page.W - L - R
    X, Y = hx + np.arange(nc + 1) / nc * hw, hy + np.arange(nr + 1) / nr * ht   # column and row edges (mm)
    ax = ed_page.axes_mm(fig, hx, hy, hw, ht)
    ax.imshow(np.array([[mpl.colors.to_rgb(COLOR[v]) for v in r] for r in st.values]),
              aspect="auto", interpolation="nearest")
    ax.set(xticks=[], yticks=[])
    for side in ax.spines.values():
        side.set(visible=True, lw=0.4, color=INK)
    # Brackets: column groups above the grid, row groups beside it (left in a and c, right in b and d)
    col_runs = runs(group)
    unit = "compound–concentration pairs" if c.dose.notna().any() else perts
    for key, a, b in col_runs:
        if key in ("validation", "control"):            # one column each in a and c: named by the key
            continue
        if kind == "bio":
            label = f"{perts} in {key} for evaluation {ctxs}" + (f"\n({b - a + 1:,} {unit})" if unit != perts
                                                                   else f" ({b - a + 1})")
        else:
            m = group.index[a:b + 1].nunique()
            label = f"{key} {perts} ({m}" + (", 3 concentrations each)" if unit != perts else ")")
        line(fig, [X[a] + 0.2, X[a] + 0.2, X[b + 1] - 0.2, X[b + 1] - 0.2], [hy - 0.6, hy - 1.4, hy - 1.4, hy - 0.6])
        text(fig, (X[a] + X[b + 1]) / 2, hy - 1.9, cap(label), ha="center", va="bottom", linespacing=1.0)
    side = "left" if kind == "bio" else "right"
    bx, tick = (hx - 0.8, -0.6) if side == "left" else (hx + hw + 0.8, 0.6)
    for is_top, a, b in runs([i < n_top for i in range(nr)]):
        n = b - a + 1
        if kind == "bio":
            label = f"evaluation {ctxs} ({n})" if is_top else f"training {ctxs} ({n})"
        else:
            label = f"{ctxs} in {'test' if is_top else 'train'} for\nevaluation {perts} ({n})"
        line(fig, [bx - tick, bx, bx, bx - tick], [Y[a] + 0.2, Y[a] + 0.2, Y[b + 1] - 0.2, Y[b + 1] - 0.2])
        text(fig, bx + 2 * tick, (Y[a] + Y[b + 1]) / 2, cap(label), ha="right" if side == "left" else "left",
             va="center", linespacing=1.0)
    # Region names. The dotted line extends the test block's edge across the training block: on its far side
    # everything is trained (training contexts in a and c, training perturbations in b and d)
    test = [(a, b) for key, a, b in col_runs if key in ("test", "evaluation")][0]
    text(fig, (X[test[0]] + X[nc]) / 2, (Y[0] + Y[n_top]) / 2, "TEST", ha="center", va="center", color="white",
         fontweight="bold")
    if kind == "bio":
        text(fig, (X[0] + X[nc]) / 2, (Y[n_top] + Y[nr]) / 2, "TRAIN", ha="center", va="center", color=INK)
        ax.axhline(n_top - 0.5, **DOTTED)
    else:
        text(fig, (X[0] + X[test[0]]) / 2, (Y[0] + Y[nr]) / 2, "TRAIN", ha="center", va="center", color=INK)
        ax.axvline(test[0] - 0.5, **DOTTED)
    # Axis labels: x below the grid, y on the side free of brackets
    n = nc - len(group[group == "control"])
    text(fig, (X[0] + X[nc]) / 2, Y[nr] + 1.0,
         cap(f"{perts} × concentration ({n:,})" if unit != perts else f"{perts} ({n})") + ", control first",
         ha="center", va="top")
    text(fig, hx + hw + 1.0 if side == "left" else hx - 1.0, (Y[0] + Y[nr]) / 2, cap(f"{ctxs} ({nr})"), rotation=90,
         ha="left" if side == "left" else "right", va="center")
    t = fig.text(4.5 / ed_page.W, 1 - (y + 0.3) / H, head[0], fontsize=6.5, fontweight="bold", va="top")
    fig.canvas.draw()                         # the dataset's width places the task after it
    fig.text(t.get_window_extent(fig.canvas.get_renderer()).x1 / fig.bbox.width, 1 - (y + 0.3) / H,
             f"  ·  {head[1]}", fontsize=6.5, va="top")
    ed_page.letter(fig, head[2], 0.0, y)


def main(data: Path = CORE / "data" / "figure2", out: Path = CORE / "figure2") -> None:
    ed_page.style()
    combos = pd.read_parquet(data / "ED_split_coverage" / "split_combinations.parquet")
    if missing := {p[1] for p in PANELS} - set(combos.split):
        raise RuntimeError(f"splits missing from split_combinations.parquet: {sorted(missing)}")
    fig = ed_page.figure(H)
    for letter, split, dataset, task, kind, ctxs, perts, y, ht in PANELS:
        panel(fig, combos[combos.split == split], kind, y, ht, (dataset, task, letter), ctxs, perts)
    fig.legend(handles=[Patch(facecolor=v, edgecolor=INK, lw=0.4, label=k) for k, v in LEGEND.items()],
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, handlelength=1.4,
               handleheight=0.9, columnspacing=1.4, borderaxespad=0.0, fontsize=5.5)
    ed_page.save(fig, out / "ed" / "ED_split_coverage")
    plt.close(fig)


if __name__ == "__main__":
    main()
