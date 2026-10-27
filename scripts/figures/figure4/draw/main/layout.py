"""Figure 4's core figure: two held-out lines stacked, measured over GENESIS, the 14 pathways x the
axis drugs, the responses they share (amber) and those of one line (green) boxed where the data bear them out.
The rules (candidates -> judge -> story_rule / fixed_rule) are pure functions of the context analysis's test
scores; core() draws the pair through panels.heatmap.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.figures.figure4.draw import panels as F
from src.figures import page

# ----------------------------------------------------------------------------
# Draw the core figure: two held-out lines, measured over GENESIS, with the
# responses they share and the ones that belong to one line marked
# ----------------------------------------------------------------------------

# One box colour per call, for the bold name and its boxes in both lines; how often GENESIS
# agrees is a tally beside the key, not a colour. Amber and a dark bluish green: off the RdBu scale and GENESIS
# blue, 3.4:1 and 5.3:1 on white, CIE76 dE 59 / 45 / 77 apart under simulated deutan / protan / tritan vision
# (Machado 2009). No colour stands out from every cell, so each box carries a white halo. A drug's name is
# green if any of its boxes is unique (line-specific), else amber.
SHARED_C, UNIQUE_C = "#c27c00", "#007a5a"
BOX_COLOURS = {"shared": SHARED_C, "unique": UNIQUE_C}
HALO = [pe.withStroke(linewidth=2.0, foreground="white")]
REPEAT = "•"                # before the name of a drug measured on more than one 5 uM plate
NAMES_IN = 1.32             # drug-name margin: a bold name runs ~10% past the plain 1.20 in (core() checks)
# Rows of white: a line's measured over its GENESIS, then the next line. The between-line gap is the
# panel break (the letter and the line's name sit on its lower edge); class rules stop there.
GAP_WITHIN, GAP_BETWEEN = 0.4, 5.0
LINE_PT = 7.0                        # each line's name, centred over its two blocks, title-sized
KEY_PT = 7.0                         # the key and the colour bar's text: Nature's 5 pt floor is too small to read
BLOCK_NAMES = {"Experimental readout": "Experimental data"}   # this figure's wording
EXCEPTION = "measured_apart_min"     # curated_picks' optional column: a five-line grid rule, not used by pairs


def mark_repeats(ax: plt.Axes, cols: list[str], reps: set[str]) -> None:
    """Prefix REPEAT to the name of every column in ``reps`` (data.multi_plate): the axis mixes them with
    drugs on one plate."""
    names = [t.get_text() for t in ax.get_xticklabels()]
    if len(names) != len(cols):
        raise RuntimeError(f"{len(names)} column names for {len(cols)} columns")
    ax.set_xticks(range(len(cols)), [f"{REPEAT} {n}" if c in reps else n for c, n in zip(cols, names)],
                  rotation=90, fontsize=F.TICK_PT)


def box(ax: plt.Axes, x: float, y: float, colour: str, h: int = 1) -> None:
    """Frame rows y .. y+h-1 of column x, with the white halo. y increases down the grid."""
    ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, h, fill=False, edgecolor=colour, lw=1.0, zorder=6,
                               clip_on=False, path_effects=HALO))


# The core pair, drawn as frames rather than one cell per call. EGFR and MAPK are one
# shared response: PROGENy's EGFR footprint is MAPK targets. JAK-STAT, NFkB and TNFa are one line-specific
# frame; the call inside it is TNFa for trametinib and JAK-STAT for RMC-6236. Homoharringtonine's NFkB and
# TNFa are one shared frame; its Hypoxia cell stays its own. Every other column, and every other pair,
# stays one cell per call.
_RAS = (("shared", ("EGFR", "MAPK")), ("unique", ("JAK-STAT", "NFkB", "TNFa")))
COLUMN_FRAMES = {"Trametinib": _RAS, "RMC-6236": _RAS, "Homoharringtonine": (("shared", ("NFkB", "TNFa")),)}
CORE_LINES = ("HOP62", "Hs_766T")


def cell_frames(lines: tuple[str, str], marks: pd.DataFrame, names: list[str]) -> list[tuple[str, str, int, int]]:
    """(drug, call, first row, height) for the boxes core() draws. ``marks`` are the drawn rows, one per
    (drug, pathway). On the core pair a frame replaces the cells it covers, and every drawn mark inside it
    must carry that frame's call. Raises if a frame's pathways are not drawn or not contiguous."""
    covered: set[tuple[str, str]] = set()
    out: list[tuple[str, str, int, int]] = []
    if lines == CORE_LINES:
        for drug, frames in COLUMN_FRAMES.items():
            for call, paths in frames:
                missing = [p for p in paths if p not in names]
                ys = [names.index(p) for p in paths if p in names]
                if missing or ys != list(range(ys[0], ys[0] + len(ys))):
                    raise RuntimeError(f"{drug} {call} frame {paths}: not a contiguous run of drawn rows")
                inside = marks[(marks.drug == drug) & marks.pathway.isin(paths)]
                if inside.empty or not inside.call.eq(call).all():
                    raise RuntimeError(f"{drug} {call} frame covers calls "
                                       f"{sorted(zip(inside.pathway, inside.call))}")
                covered |= {(drug, p) for p in paths}
                out.append((drug, call, ys[0], len(paths)))
    for r in marks.itertuples():
        if (r.drug, r.pathway) not in covered:
            out.append((r.drug, r.call, names.index(r.pathway), 1))
    return out


def bold_names(ax: plt.Axes, cols: list[str], marks: pd.DataFrame) -> None:
    """Each marked drug's name bold in its call's colour: green if any of its marks is unique."""
    for d, calls in marks.groupby("drug").call:
        ax.get_xticklabels()[cols.index(d)].set(fontweight="bold",
                                                color=UNIQUE_C if (calls == "unique").any() else SHARED_C)


def panel_letters(fig: plt.Figure, ax: plt.Axes, owner: list[str], tops: np.ndarray) -> None:
    """a, b, ... at the top left of each cell line: the 8 pt bold letter the other main figures use,
    its top level with that line's name."""
    fig.canvas.draw()
    for lab, name in zip("abcdefghijklmnopqrstuvwxyz", dict.fromkeys(owner)):
        y = ax.transData.transform((0, tops[owner.index(name)] - 0.5))[1] / fig.dpi
        top = (fig.get_figheight() - y) * 25.4 - (2 + LINE_PT) * 25.4 / 72   # name: 2 pt up, va bottom
        if top < 0:
            raise RuntimeError(f"{name}: panel letter {lab} falls above the canvas")
        page.letter(fig, lab, 0.0, top)


def box_key(fig: plt.Figure) -> None:
    """The mark legend, top left, centred on the colour bar's row (the bar spans 0.23-0.30 in from the top;
    the figure has no title): shared, unique, and the two-plate dot, at KEY_PT. The call rule and GENESIS's
    tally are caption. Raises if the line reaches the colour bar (top right, from W - 1.75 in)."""
    w_in, h_in = fig.get_size_inches()
    rend, x, y, side = fig.canvas.get_renderer(), 0.06, 1 - 0.265 / h_in, 0.075
    for tok in ("Shared", SHARED_C, "Unique*", UNIQUE_C, f"   {REPEAT} Measured on two or more repeats"):
        if tok in (SHARED_C, UNIQUE_C):
            fig.add_artist(plt.Rectangle((x / w_in, y - side / 2 / h_in), side / w_in, side / h_in,
                                         transform=fig.transFigure, fill=False, edgecolor=tok, lw=1.0,
                                         path_effects=HALO))
            x += side + 0.08
        else:
            txt = fig.text(x / w_in, y, tok, fontsize=KEY_PT, ha="left", va="center", color=F.INK)
            x += txt.get_window_extent(rend).width / fig.dpi + 0.05
    if x > F.page_in() - 1.75:
        raise RuntimeError(f"the legend ends at {x:.2f} in, under the colour bar")


# Over 40 lines: its blocks are the four blocks and their gaps, heatmap(), the key and the resized colour bar,
# the repeat dots and bold names with the bracket check, the line names, the frames, the drawn-direction check.
def core(S: dict, lines: tuple[str, str], marks: pd.DataFrame, reps: set[str]) -> plt.Figure:
    """Blocks: line 1 measured, line 1 GENESIS, line 2 measured, line 2 GENESIS; the 14 pathways in
    their groups; the axis drugs in their classes. Each ``drawn`` pick (a rule's: drug, pathway, call,
    truth_A/B, genesis_right) boxes its cell in all four blocks in its call's colour, except the core
    pair's trametinib, RMC-6236 and homoharringtonine columns, which are frames (cell_frames). The header is
    the swatches. No error bars. ``reps``: the multi-plate drugs."""
    marks = marks[marks.drawn].drop_duplicates(["drug", "pathway"])
    t = S["mv"][S["mv"].set == "grid"]
    cols, col_groups = F.column_order(S, t)
    names, groups = F.pathway_rows(S["progs"])
    blocks, gaps, owner = [], [], []         # the blocks carry the arm only; the line's name is drawn above them
    for c in lines:
        rows = [(f"{p}{'*' if S['cov'].get((p, c), 0) < 3 else ''}", p) for p in names]
        mine = [(BLOCK_NAMES.get(name, name), rs) for name, rs in F.blocks_of(S["arms"], t[t.cell_name == c], rows, cols)]
        gaps += [GAP_BETWEEN] * bool(blocks) + [GAP_WITHIN] * (len(mine) - 1)
        blocks, owner = blocks + mine, owner + [S["disp"][c]] * len(mine)
    # 0.14 in is the pad past the longest pathway name (Androgen, 0.34 in at 5 pt). The 0.035*len
    # term underestimates it; heatmap() parks the block brackets in the 0.24 in beyond that.
    fig = F.heatmap(blocks, cols, left_in=0.14 + 0.035 * max(len(n) for n in names) + 0.24, row_in=0.08,
                    groups=groups, col_groups=col_groups, names_in=NAMES_IN, shade_gaps=False, gaps=gaps,
                    open_gaps={i for i, g in enumerate(gaps) if g == GAP_BETWEEN})
    box_key(fig)
    if len(bar := [a for a in fig.axes if a.get_xlabel().startswith("PROGENy score")]) != 1:
        raise RuntimeError(f"{lines}: expected one colour bar, found {len(bar)}")
    bar[0].xaxis.label.set_size(KEY_PT), bar[0].tick_params(labelsize=KEY_PT)   # heatmap() draws it at TICK_PT
    fig.canvas.draw()                                          # its label is now wider than the bar: pull it in
    w_in, over = fig.get_figwidth(), bar[0].xaxis.label.get_window_extent().x1 / fig.dpi - (fig.get_figwidth() - 0.06)
    if over > 0:
        pos = bar[0].get_position()
        bar[0].set_position([pos.x0 - over / w_in, pos.y0, pos.width, pos.height])
    ax = fig.axes[0]
    mark_repeats(ax, cols, reps)
    bold_names(ax, cols, marks)
    fig.canvas.draw()                                          # no name, bold or not, may reach the class bracket
    bracket_in = ax.get_position().y0 * fig.get_size_inches()[1] - (NAMES_IN - 0.02)
    if (low := [n.get_text() for n in ax.get_xticklabels() if n.get_window_extent().y0 / fig.dpi < bracket_in + 0.01]):
        raise RuntimeError(f"{lines}: {low} reach the class bracket; widen NAMES_IN")
    tops = ax.get_yticks()[::len(names)]                      # each block's first row, as drawn
    for name in dict.fromkeys(owner):
        ax.annotate(name, ((len(cols) - 1) / 2, tops[owner.index(name)] - 0.5), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom", fontsize=LINE_PT, fontweight="bold",
                    color=F.INK, annotation_clip=False, zorder=7,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
    panel_letters(fig, ax, owner, tops)
    for drug, call, y, h in cell_frames(lines, marks, names):
        for top in tops:
            box(ax, cols.index(drug), top + y, BOX_COLOURS[call], h)
    v = t[t.arm == "truth"].set_index(["cell_name", "drug", "programme"]).v      # re-read: what is drawn
    if (bad := [r.drug for r in marks.itertuples() if not right_way(r.call, (r.clipped_A, r.clipped_B),
                                                                     [v[(c, r.drug, r.pathway)] for c in lines])]):
        raise RuntimeError(f"{lines}: {bad} boxed in a direction their drawn cells do not show")
    return fig


# ----------------------------------------------------------------------------
# Find the picks in the scores
# ----------------------------------------------------------------------------

# The pair rule, by effect size, no noise floors. A candidate is an axis drug at 5 uM
# on any pathway but Androgen (its floor, 0.06, is too small to mean anything), each value the mean over the
# plates where that line is clean. The call is made on log2 sizes: 'unique' when the lines are UNIQUE apart,
# else 'shared' when both move the same way by SHARED_MIN; the drawn depth values must make the same call,
# and still show it where the colour scale saturates (clipped to +-VMAX: +3.0 vs +1.6 is one colour),
# and its direction must replicate on each of two 5 uM plates or at another dose. GENESIS is right when it
# has truth's sign (the box colour); it compresses sizes, so whether it also shows at least half
# the effect is a column of its own (genesis_shows_half).
DOSE, OTHER_DOSES, UNMARKED = 5.0, (0.5, 0.05), "Androgen"
UNIQUE, SHARED_MIN = 0.5, 0.5                  # log2FC
CAND_COLS = ["drug", "drug_class", "pathway", "truth_A", "truth_B", "genesis_A", "genesis_B", "drawn_A",
             "drawn_B", "n_plates", "plates_A", "plates_B",
             *(f"truth_{a}_{x:g}uM" for x in OTHER_DOSES for a in "AB")]
JUDGED_COLS = ["call", "call_measured", "clipped_A", "clipped_B", "genesis_right", "genesis_shows_half",
               "replicated_by", "strength", "why_not"]
# The curated stories: the eligible (drug, pathway) are curated; the data only decide which hold in a pair. The
# core figure is one pair whose boxes and calls are fixed (core_pair.tsv); the data must confirm each or the
# build stops.
STORY_COLS = ["story_id", "pathway", "drug", "expected", "explanation"]
FIXED_COLS = ["line_a", "line_b", "call", "pathway", "drug", "story_id"]


def candidates(t: pd.DataFrame, drawn: pd.Series, lines: tuple[str, str], drugs: dict[str, str],
               pathways) -> pd.DataFrame:
    """Every (axis drug, pathway) with a clean 5 uM truth e in both ``lines`` (display spelling; A =
    lines[0]), with all a pick rule may read. Each value is a line's mean over the plates where it is
    clean (recovery > 0.5, no zero-mean gene; truth and GENESIS each by their own rows): truth and GENESIS
    at 5 uM, truth at each other dose (NaN where unmeasured); plates_A/B: truth on the 5 uM plates clean
    in both lines. ``drugs``: axis drug -> its drawn class. Raises unless the lines share their 5 uM plates."""
    A, B = lines
    if (odd := set(t.dose) - {DOSE, *OTHER_DOSES}):
        raise RuntimeError(f"doses {sorted(odd)}: neither 5 uM nor a dose a call replicates on")
    t = t[t.cell_name.isin(lines) & t.drug.isin(drugs) & t.readout.isin(pathways) & t.is_test]
    five = (t[(t.dose == DOSE) & (t.arm == "truth")].groupby(["drug", "readout", "cell_name"])
            .plate.apply(frozenset))
    if (absent := sorted(set(drugs) - set(five.index.get_level_values("drug")))):
        raise RuntimeError(f"no 5 uM score for {absent}")
    if (apart := [k for k, s in five.groupby(level=["drug", "readout"]) if len(s) != 2 or s.iloc[0] != s.iloc[1]]):
        raise RuntimeError(f"lines on different 5 uM plates: {apart[:5]}")
    ok = t[(t.recovery > 0.5) & (t.floor_hits == 0)]                        # clean, row by row
    e = (ok.set_index(["arm", "readout", "drug", "dose", "plate", "cell_name"]).e    # unstack raises on a duplicate
         .unstack("cell_name").reindex(columns=list(lines)))
    m = e.groupby(level=["arm", "readout", "drug", "dose"]).mean()           # NaN-skipping: each line's clean plates
    nan = pd.Series({A: np.nan, B: np.nan})
    rows = []
    for (p, d), r in m.xs(("truth", DOSE), level=("arm", "dose")).dropna().iterrows():
        g = m.loc[("genesis", p, d, DOSE)] if ("genesis", p, d, DOSE) in m.index else nan
        pl = e.loc[("truth", p, d, DOSE)].dropna()                         # the plates clean in both lines
        row = dict(drug=d, drug_class=drugs[d], pathway=p, truth_A=r[A], truth_B=r[B], genesis_A=g[A],
                   genesis_B=g[B], drawn_A=drawn[(A, d, p)], drawn_B=drawn[(B, d, p)], n_plates=len(pl),
                   plates_A=tuple(pl[A]), plates_B=tuple(pl[B]))
        for x in OTHER_DOSES:
            o = m.loc[("truth", p, d, x)] if ("truth", p, d, x) in m.index else nan
            row |= {f"truth_A_{x:g}uM": o[A], f"truth_B_{x:g}uM": o[B]}
        rows.append(row)
    return pd.DataFrame(rows, columns=CAND_COLS)


def size_call(a: float, b: float) -> str:
    """'unique': the lines at least UNIQUE apart; else 'shared': same sign, each at least SHARED_MIN; else ''."""
    if abs(a - b) >= UNIQUE:
        return "unique"
    if np.sign(a) == np.sign(b) and min(abs(a), abs(b)) >= SHARED_MIN:
        return "shared"
    return ""


def both_call(e: tuple[float, float], c: tuple[float, float]) -> str:
    """The call the measured ``e`` and the clipped drawn ``c`` both make, the same way:
    'unique' when each pair is UNIQUE apart in one direction, else 'shared' when all four values move one
    way by SHARED_MIN; else ''. A saturated pair (both past VMAX) can be shared though e is far apart."""
    def apart(x):
        return abs(x[0] - x[1]) >= UNIQUE

    def together(x):
        return np.sign(x[0]) == np.sign(x[1]) and min(map(abs, x)) >= SHARED_MIN

    if apart(e) and apart(c) and np.sign(e[0] - e[1]) == np.sign(c[0] - c[1]):
        return "unique"
    if together(e) and together(c) and np.sign(e[0]) == np.sign(c[0]):
        return "shared"
    return ""


def right_way(kind: str, truth: tuple[float, float], model: Sequence[float]) -> bool:
    """``model`` points the way ``truth``'s call does: 'shared', every value in ``model`` has truth's sign
    (any number of lines); otherwise (unique, line-specific) the sign of a - b. NaN never does."""
    if kind == "shared":
        return all(np.sign(m) == np.sign(truth[0]) for m in model)
    return np.sign(model[0] - model[1]) == np.sign(truth[0] - truth[1])


def judge(r: dict) -> dict:
    """One candidate row: its call, or '' and why_not. The call must hold on the measured e (plate background
    removed) and on the drawn depth values clipped to the colour scale (+-VMAX: +3.0 against +1.6 is one
    colour, so not unique), the same way (both_call); and replicate in that direction on each of exactly
    two 5 uM plates clean in both lines, else at the first other dose. GENESIS is right when it points the
    call's way; genesis_shows_half when it also shows half the threshold. call_measured is size_call on e
    alone, for the record."""
    e, gm = (r["truth_A"], r["truth_B"]), (r["genesis_A"], r["genesis_B"])
    c = tuple(float(x) for x in np.clip((r["drawn_A"], r["drawn_B"]), -F.VMAX, F.VMAX))
    out = dict(call="", call_measured=size_call(*e), clipped_A=c[0], clipped_B=c[1], genesis_right=None,
               genesis_shows_half=None, replicated_by="", strength=np.nan)
    if r["pathway"] == UNMARKED:
        return out | dict(why_not="Androgen: its floor is too small to mean anything")
    if not (kind := both_call(e, c)):
        return out | dict(why_not="no call as drawn" if not size_call(*c) else "the measured e does not make it")
    two = r["n_plates"] == 2 and all(right_way(kind, c, ab) for ab in zip(r["plates_A"], r["plates_B"]))
    doses = [f"dose {x:g}" for x in OTHER_DOSES if right_way(kind, c, (r[f"truth_A_{x:g}uM"], r[f"truth_B_{x:g}uM"]))]
    if not (how := "two_plates" if two else next(iter(doses), "")):
        return out | dict(why_not="not replicated")
    right = bool(right_way(kind, c, gm))
    half = abs(gm[0] - gm[1]) >= UNIQUE / 2 if kind == "unique" else min(map(abs, gm)) >= SHARED_MIN / 2
    strength = abs(c[0] - c[1]) if kind == "unique" else min(map(abs, c))
    return out | dict(call=kind, genesis_right=right, genesis_shows_half=right and bool(half), replicated_by=how,
                      strength=strength, why_not="")


def curated_table(path: Path, axis: set[str], pathways: list[str]) -> pd.DataFrame:
    """The curated stories; raises unless the file has exactly STORY_COLS, every drug is on the axis and every
    pathway is drawn: a story naming a column or row the figure lacks could never be boxed."""
    cur = pd.read_csv(path, sep="\t", comment="#", dtype=str, keep_default_na=False)
    if list(cur.columns) not in (STORY_COLS, STORY_COLS + [EXCEPTION]):
        raise RuntimeError(f"{path.name}: columns {list(cur.columns)}, expected {STORY_COLS} (+ {EXCEPTION})")
    off, unknown = sorted(set(cur.drug) - axis), sorted(set(cur.pathway) - set(pathways))
    if off or unknown:
        raise RuntimeError(f"{path.name}: drugs off the axis {off}, pathways not drawn {unknown}")
    return cur[STORY_COLS]


def story_rule(cand: pd.DataFrame, curated: pd.DataFrame) -> pd.DataFrame:
    """The curated stories in one pair, a pure function of ``candidates`` and the curated table: every curated row,
    judged (call '' and why_not where it does not hold here), boxed (``drawn``) when it calls."""
    by = {(r["drug"], r["pathway"]): r for r in cand.to_dict("records")}
    rows = [s | (r | judge(r) if (r := by.get((s["drug"], s["pathway"]))) else
                 dict(call="", why_not="not measured cleanly at 5 uM in both lines"))
            for s in curated.to_dict("records")]
    cols = [*curated.columns, *[c for c in CAND_COLS if c not in curated.columns], *JUDGED_COLS]
    marks = pd.DataFrame(rows, columns=cols)
    return marks.assign(drawn=marks.call != "")


def core_pair(path: Path, curated: pd.DataFrame) -> pd.DataFrame:
    """The core figure's boxes: one pair of lines, each row a curated story with the call it
    must get (fixed_call), its story's text joined. Raises on another layout, two pairs, or a row that is
    not one of the curated stories."""
    fixed = pd.read_csv(path, sep="\t", comment="#", dtype=str, keep_default_na=False)
    if list(fixed.columns) != FIXED_COLS or len(fixed[["line_a", "line_b"]].drop_duplicates()) != 1:
        raise RuntimeError(f"{path.name}: columns {list(fixed.columns)} (expected {FIXED_COLS}), one pair of lines")
    rows = fixed.rename(columns={"call": "fixed_call"}).merge(curated, on=["story_id", "pathway", "drug"],
                                                              how="left", validate="one_to_one", indicator=True)
    if (stray := rows[rows._merge != "both"]).size:
        raise RuntimeError(f"{path.name}: not curated stories: "
                           f"{stray[['story_id', 'drug', 'pathway']].values.tolist()}")
    return rows.drop(columns="_merge")


def fixed_rule(cand: pd.DataFrame, fixed: pd.DataFrame) -> pd.DataFrame:
    """The core figure, a pure function of ``candidates`` and core_pair's rows: exactly those boxes, each
    judged; raises unless every one gets its fixed call."""
    marks = story_rule(cand, fixed)
    if len(bad := marks[marks.call != marks.fixed_call]):
        evidence = ["drug", "pathway", "fixed_call", "call", "why_not", "truth_A", "truth_B", "clipped_A", "clipped_B"]
        raise RuntimeError(f"the data do not confirm the core figure's calls:\n{bad[evidence].to_string()}")
    return marks
