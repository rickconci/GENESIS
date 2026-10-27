#!/usr/bin/env python3
"""Figure 4's pathway responses, cross-batch: Pearson r along three axes, each model against the experimental
replicate (plate A vs plate B).

Every held-out test condition on two or more plates: Figure 2a's 530 plate pairs, 500 conditions = 100
(compound, dose) x 5 held-out lines, 14 PROGENy pathways, depth-adjusted, each plate alone with the gene gate
intersected over the pair's plates. A condition is (compound, dose). Each
model is scored against the plate it was NOT conditioned on, both directions averaged
(scripts.figures.figure4.extract.crossbatch). The nearest neighbour enters three times: NN own, the paper's (Figure
2a's v2 neighbour, its log2FC against its own controls); NN response (the same on the hbss015
neighbour table); and NN cells (Figure 2a's neighbour's cells against the held-out line's controls).

  profile    r over the 14 pathways of one (line, condition)   -> mean over conditions per line, then over lines
  compound   r over the 100 conditions of one (line, pathway)  -> mean over pathways per line, then over lines
  line       r over the 5 lines of one (pathway, condition)        -> mean over every (pathway, condition)
             (conditions every line has; the plain mean, not the double-centred pooled r)
The per-condition and per-pathway means are written too. Afatinib is on three plates (three pairs): every
axis is scored once per pair ("layer"; two-plate conditions are the same in each) and r averaged over layers,
which on the profile axis is exactly its three pairs averaged within the condition.

Nothing is written unless, on the axis's replicated compounds and the gate Figure 4's context panel used (two
plates' detection averaged), the double-centred pooled r reproduces that panel's table (context_panel_source_data.csv).

    python scripts/figures/figure4/score/ed_program_summary/axis_correlation.py --bundle <bundle>
      -> <bundle>/tables/supporting/tahoe_progeny_depth/axis_correlation/{profile,compound,line}_{per_group,summary}.csv
      -> <bundle>/figures/supporting/tahoe_progeny_depth/axis_correlation/axis_correlation.{svg,pdf,png}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from src.eval.plots.style import COL, HATCH, hatch_ink  # noqa: E402
from scripts.figures.figure4.extract.crossbatch import REPEAT, cross_batch_r, double_centre  # noqa: E402

SPLIT, VALUES = "tahoe_progeny_depth", "data/axis_correlation/pair_values__depth.parquet"
LABEL = {"genesis": "GENESIS", "state": "STATE", "additive": "Additive",
         "nn_own": "NN own", "nn_response": "NN response", "nn_cells": "NN cells"}
# Every nearest neighbour in Figure 2a's brown and cross-hatch; NN response and NN cells lighter.
COL = COL | {"NN own": COL["copy_closest"], "NN response": "#b08a62", "NN cells": "#c9a27a"}
HATCH = HATCH | {k: HATCH["copy_closest"] for k in ("NN own", "NN response", "NN cells")}
# (axis, the r groups: (line, the other key), or None for the line axis, whose groups hold every line)
AXES = [("profile", ["cell_name", "drug"]), ("compound", ["cell_name", "programme"]), ("line", None)]
C_REPEAT, REF_DASH, INK = "#1f5f68", (0, (4, 2.5)), "#1b1d1f"   # Figure 2a's experimental replicate
NO_DATE = {"svg": {"Date": None}, "pdf": {"CreationDate": None}, "png": {}}
TOL = 1e-6


# ----------------------------------------------------------------------------
# The tables: (programme, condition, line) x (arm, plate), one per pair layer, every cell present
# ----------------------------------------------------------------------------

def layers(bundle: Path, gate: str, arms: list[str], drugs5: set[str] | None = None) -> list[pd.DataFrame]:
    """Layer k takes each condition's k-th pair, or its only pair. ``drugs5``: keep those compounds at 5 uM."""
    v = pd.read_parquet(bundle / VALUES)
    v = v[(v.gate == gate) & v.arm.isin(["truth", *arms])]
    if drugs5 is not None:
        v = v[v.drug.isin(drugs5) & np.isclose(v.dose, 5.0)]
    v = v.assign(drug=v.drug + " @ " + v.dose.map("{:g} uM".format))   # the condition
    n = v.groupby(["cell_name", "drug"]).pair.transform("nunique")
    if not set(n.unique()) <= {1, int(n.max())}:
        raise SystemExit(f"conditions carry {sorted(n.unique())} pairs: layers need one count besides 1")
    # A pathway or line absent for EVERY arm leaves no NaN behind, so each level is held to its own list.
    want = dict(drug=set(v.drug), cell_name=set(pd.read_csv(bundle / f"data/{SPLIT}/heldout_lines.tsv", sep="\t").cell_name),
                programme=set(pd.read_csv(bundle / f"data/{SPLIT}/programmes.tsv", sep="\t").query("set == 'grid'").programme))
    out = []
    for k in range(int(n.max())):
        sel = v[v.pair == np.where(n > 1, k, 0)]
        w = sel.set_index(["programme", "drug", "cell_name", "arm", "slot"]).v.unstack(["arm", "slot"])  # raises on a duplicate
        have = {key: set(w.index.get_level_values(key)) for key in want}
        if any(have[key] != want[key] for key in want) or w["truth"].isna().any().any():
            raise SystemExit(f"layer {k} incomplete: absent {[sorted(want[x] - have[x]) for x in want]}, "
                             f"{int(w['truth'].isna().sum().sum())} truth cells missing")
        out.append(w)
    print(f"  {gate} gate: {len(want['drug'])} conditions x {len(want['cell_name'])} lines x "
          f"{len(want['programme'])} pathways, {len(out)} pair layer(s)")
    for a in arms:
        if gone := {k for w in out for k, m in w[a].isna().any(axis=1).groupby(level=["drug", "cell_name"]).any().items() if m}:
            print(f"  ! {a} is absent from {len(gone)} (condition, line)s, e.g. {sorted(gone)[:2]}: scored alone on the rest")
    return out


# ----------------------------------------------------------------------------
# Three axes: r per group and layer, layers averaged, then averaged (or pooled) per level
# ----------------------------------------------------------------------------

def layered(ws: list[pd.DataFrame], f) -> tuple[pd.DataFrame, int]:
    """f(w) -> (rows, dropped) on every layer; r averaged over layers, dropped groups summed."""
    parts = [f(w) for w in ws]
    per = pd.concat([p for p, _ in parts])
    keys = [c for c in per.columns if c != "r"]
    out = per.groupby(keys, sort=False).r.mean().reset_index()[parts[0][0].columns]
    if len(out) != len(parts[0][0]):
        raise SystemExit("layers disagree on the groups or their sizes: they would average unlike things")
    return out, sum(d for _, d in parts)


def full_grid(w: pd.DataFrame) -> pd.DataFrame:
    """The conditions present in every line x pathway: every across-lines r then runs over all five lines."""
    k = w.groupby(level="drug").size()
    full = w.index.get_level_values("cell_name").nunique() * w.index.get_level_values("programme").nunique()
    return w[w.index.get_level_values("drug").isin(k.index[k == full])]


def score(ws: list[pd.DataFrame], arms: list[str]) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Every arm present everywhere is scored together, on every condition. An arm absent from some
    (condition, line)s (NN response: its neighbour has < 50 cells on a plate) is scored alone on the rest,
    as Figures 2a and 4 skip a pair for that arm only; its replicate rows are the full set's. A
    (condition, line) it lacks in any pair layer leaves every layer, so the layers stay alike."""
    whole = [a for a in arms if all(w[a].notna().all().all() for w in ws)]
    parts = []
    for s in [whole] + [[a] for a in arms if a not in whole]:
        if s is not whole:
            print(f"  {s[0]} alone, on the (condition, line)s it has:")
        cl = lambda w: w.index.droplevel("programme")  # noqa: E731
        gone = {k for w in ws for k in cl(w)[w[s].isna().any(axis=1).to_numpy()]}
        res = score_arms([w.loc[~cl(w).isin(gone), ["truth", *s]] for w in ws], s)
        parts.append(res if s is whole else {ax: tuple(f[f.arm != REPEAT] for f in fs) for ax, fs in res.items()})
    return {ax: tuple(pd.concat([p[ax][j] for p in parts], ignore_index=True) for j in (0, 1)) for ax in parts[0]}


def score_arms(ws: list[pd.DataFrame], arms: list[str]) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """{axis: (per-group r, summary)}. Summary rows: level ('overall' or a key), key, arm, r, n, n_dropped;
    n = what was averaged (lines for 'overall' on the first two axes, groups otherwise).
    The condition is written as 'compound' ('<compound> @ <dose> uM')."""
    mean = lambda f, lv: f.groupby([*lv, "arm"], sort=False).r.agg(r="mean", n="size").reset_index()  # noqa: E731
    out = {}
    for axis, by in AXES:
        if by is not None:
            per, dropped = layered(ws, lambda w: cross_batch_r(w, by, arms))
            line = mean(per, ["cell_name"])
            # overall = mean over lines of each line's mean: every line weighs the same even if a group drops
            rows = [([], mean(line, []), dropped), (["cell_name"], line, dropped), (by[1:], mean(per, by[1:]), dropped)]
        else:
            per, dropped = layered([full_grid(w) for w in ws], lambda w: cross_batch_r(w, ["programme", "drug"], arms))
            rows = [(lv, mean(per, lv), dropped) for lv in ([], ["programme"], ["drug"])]
        s = pd.concat([f.assign(level=lv[0] if lv else "overall", key=f[lv[0]] if lv else "", n_dropped=d)
                       for lv, f, d in rows])[["level", "key", "arm", "r", "n", "n_dropped"]].reset_index(drop=True)
        out[axis] = per.rename(columns={"drug": "compound"}), s.replace({"level": {"drug": "compound"}})
        o = s[s.level == "overall"].set_index("arm")
        print(f"  {axis:8s} " + "  ".join(f"{a} {o.r[a]:.3f}" for a in [REPEAT, *arms])
              + f"   (dropped {int(o.n_dropped.iloc[0])})")
    return out


def check_context(bundle: Path, arms: list[str]) -> None:
    """On Figure 4's context set (27 compounds at 5 uM, averaged gate) the double-centred pooled r IS
    the context panel's r (context_panel_source_data.csv): refuse anything else, so these values stand on the
    published ones."""
    ax = pd.read_csv(bundle / "data/drug_axis/drug_axis.tsv", sep="\t")
    (w,) = layers(bundle, "mean", arms, set(ax.drug[ax.on_axis & ax.replicated]))
    whole = [a for a in arms if w[a].notna().all().all()]
    o = cross_batch_r(double_centre(w[["truth", *whole]]), [], whole)[0].set_index("arm").r
    t = bundle / f"tables/supporting/{SPLIT}"
    for f in (t / "context/context_panel_source_data.csv", t / "state/context/context_panel_source_data.csv"):
        ref = pd.read_csv(f).query("programme == 'all'").set_index("arm").r
        if len(shared := ref.index.intersection(o.index).drop(REPEAT, errors="ignore")) == 0:
            raise SystemExit(f"{f.relative_to(bundle)} shares no model with {list(o.index)}: nothing checked")
        for a in [REPEAT, *shared]:
            if not abs(ref[a] - o[a]) <= TOL:      # 'not <=' so a NaN fails too
                raise SystemExit(f"line axis {a} r {o[a]:.6f} != {f.relative_to(bundle)} {ref[a]:.6f}")
            print(f"  check: line axis {a} {o[a]:.6f} == context table {ref[a]:.6f}")


# ----------------------------------------------------------------------------
# The panel: one bar per model, the experimental replicate dashed, the per-level values as dots
# ----------------------------------------------------------------------------

def panel(res: dict, arms: list[str]) -> plt.Figure:
    # importing src.eval.plots runs sns.set_style("whitegrid") (grid on, grey frame): start from matplotlib's own
    plt.style.use("default")
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 6.0, "axes.linewidth": 0.5, "hatch.linewidth": 0.35, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "svg.hashsalt": "axis_correlation"})
    # (title: what r runs across, the level drawn as points, its key, filled marker?)
    spec = {"profile": ("Across pathways", "cell_name", "one cell line", True),
            "compound": ("Across compounds", "cell_name", "one cell line", True),
            "line": ("Across cell lines", "programme", "one pathway", False)}
    halo = [pe.withStroke(linewidth=1.5, foreground="white")]                # keeps a number legible over the dots
    # pathways hollow, lines filled: the right panel's points are a different unit
    dot = lambda filled: dict(s=3 if filled else 4, color=INK if filled else "none", edgecolors=INK,  # noqa: E731
                              alpha=0.5, lw=0 if filled else 0.4)
    fig, axs = plt.subplots(1, 3, figsize=((30 * len(arms) + 30) / 25.4, 42 / 25.4), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.2, top=0.8, wspace=0.12)
    for ax, (axis, (_per, s)) in zip(axs, res.items()):
        o = s[s.level == "overall"].set_index("arm").r
        for i, a in enumerate(arms):
            fill = COL[LABEL[a]]
            ax.bar(i, o[a], 0.7, color=fill, hatch=HATCH.get(LABEL[a], ""), edgecolor=hatch_ink(fill), lw=0)
            d = s[(s.level == spec[axis][1]) & (s.arm == a)].r
            ax.scatter(i + np.linspace(-0.2, 0.2, len(d)), d, zorder=3, **dot(spec[axis][3]))
            inside = o[a] > 0.15     # just under the mean, clear of the replicate line; a stub bar takes it above
            ax.text(i, o[a] - 0.02 if inside else max(o[a], 0) + 0.02, f"{o[a]:.2f}", ha="center",
                    va="top" if inside else "bottom", fontsize=5.5, zorder=5, color="white" if inside else INK,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground=fill)] if inside else halo)
        ax.axhline(o[REPEAT], color=C_REPEAT, lw=0.8, ls=REF_DASH, zorder=4)
        ax.text(len(arms) - 0.5, o[REPEAT] + 0.02, f"experimental reproducibility {o[REPEAT]:.2f}", ha="right",
                va="bottom", fontsize=5.5, color=C_REPEAT, zorder=5, path_effects=halo)
        ax.axhline(0, color=INK, lw=0.4)
        ax.set_xticks(range(len(arms)), [LABEL[a].replace(" ", "\n") for a in arms], fontsize=5.5)
        ax.set_title(spec[axis][0], fontsize=6.5)
        ax.set_ylim(min(-0.2, float(s.r.min()) - 0.05), 1.05)
        ax.scatter([0.04], [0.93], transform=ax.transAxes, **dot(spec[axis][3]))   # the points' key, in-panel
        ax.text(0.07, 0.93, spec[axis][2], transform=ax.transAxes, va="center", fontsize=5.5, color=INK)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel("Pearson r")
    return fig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", default=list(LABEL), choices=list(LABEL))
    a = ap.parse_args()
    check_context(a.bundle, a.arms)
    res = score(layers(a.bundle, "replicate", a.arms), a.arms)
    tab = a.bundle / f"tables/supporting/{SPLIT}/axis_correlation"
    tab.mkdir(parents=True, exist_ok=True)
    for axis, (per, s) in res.items():
        per.to_csv(tab / f"{axis}_per_group.csv", index=False, float_format="%.6g")
        s.to_csv(tab / f"{axis}_summary.csv", index=False, float_format="%.6g")
    fig, out = panel(res, a.arms), a.bundle / f"figures/supporting/{SPLIT}/axis_correlation/axis_correlation"
    bb, size = fig.get_tightbbox(), fig.get_size_inches()   # savefig would clip a label drawn off the page
    if bb.x0 < -0.01 or bb.y0 < -0.01 or bb.x1 > size[0] + 0.01 or bb.y1 > size[1] + 0.01:
        raise SystemExit(f"panel drawing {bb.bounds} leaves the {size} in canvas")
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=600 if ext == "png" else 450, facecolor="white", metadata=NO_DATE[ext])
    print(f"wrote {tab}/ and {out}.{{svg,pdf,png}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
