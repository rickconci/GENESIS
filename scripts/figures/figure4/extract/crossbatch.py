"""Cross-batch agreement: does a response measured on one Tahoe plate come back on another plate?

Some test conditions of the five held-out lines ran on two plates (Afatinib on three). For those we hold a cube of
pathway responses, (pathway, condition, line), measured on plate A and on plate B, and each method's prediction
conditioned on plate A and on plate B. Each side is scored against the MEASURED responses:

  replicate   plate A's measurement against plate B's: what two measurements of the same biology share across a
              batch, the ceiling for any method
  a method    GENESIS and Additive; State and the nearest neighbour (NN, nn_own) on PROGENy; copy_closest_15 on
              Hallmark. Its prediction conditioned on one plate against the OTHER plate's measurement, both
              directions averaged, so reproducing the conditioning plate's own batch effect earns nothing

Agreement is a Pearson (or Spearman) r along one axis of the cube, the other two held fixed:

  profile    across pathways, per (condition, line): is the shape of a response right?
  compound   across conditions, per (line, pathway): are the drugs ordered right within a line?
  line       across the five lines, per (pathway, condition): is it the right line that responds more?

Each axis's r are averaged into one number per side, and its 95% CI is Figure 2's cluster bootstrap
(src.stats.bootstrap): line x drug clusters (the drug on the line axis, whose r runs over every line) are redrawn
and the statistic recomputed per draw. One draw of every cluster must give the analysis's own published number,
or the interval would be around some other statistic. Two EDs read this module: ED_program_summary (PROGENy scores
on Figure 2a's plate pairs, depth-adjusted) and ED_tahoe_hallmark_gsea (Hallmark NES at 5 uM, Pearson and Spearman).
The statistic itself, cross_batch_r and double_centre, is the first section; the level-3 scorer
(score/ed_program_summary/axis_correlation.py) imports it from here.
"""

from __future__ import annotations

import itertools
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.stats.bootstrap import B, LEVEL, SEED, cluster_draws

TAIL = 100 * (1 - LEVEL) / 2


# ----------------------------------------------------------------------------
# The cross-batch r: per group, the repeat's r and each arm's both-directions r, on the same groups
# ----------------------------------------------------------------------------

REPEAT = "replicate"   # the experimental replicate, as the published context tables name it
# A vector flatter than this (log2FC units) has no direction to correlate: centring a constant
# leaves ~1e-16 of round-off, which np.corrcoef would happily turn into an r of +-1. Absolute, not
# relative: inputs are float64 log2FC with |v| < ~15, where round-off stays ~1e-14.
FLAT = 1e-12


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r, NaN where either vector is flat."""
    if np.ptp(x) < FLAT or np.ptp(y) < FLAT:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def cross_batch_r(w: pd.DataFrame, by: list[str], arms: list[str]) -> tuple[pd.DataFrame, int]:
    """Rows (*by, arm, r, n), the repeat first, then ``arms``; and the number of groups dropped.

    A group where any arm's r is undefined (a flat vector) is dropped for EVERY arm, so all arms
    are averaged over the same groups. Raises on a missing (arm, slot), a non-finite value, an
    empty table, an arm named like the repeat or truth, or every group dropped: each would
    otherwise shrink, bias or relabel the average without a word."""
    if len(set(arms)) != len(arms) or {"truth", REPEAT} & set(arms):
        raise ValueError(f"arms must be distinct and not 'truth' / '{REPEAT}': {arms}")
    need = [(a, s) for a in ["truth", *arms] for s in (0, 1)]
    if missing := [c for c in need if c not in w.columns]:
        raise ValueError(f"columns missing: {missing}")
    if w.empty:
        raise ValueError("empty table")
    x = w[need]
    if (bad := int((~np.isfinite(x.to_numpy())).sum())):
        raise ValueError(f"{bad} values not finite")
    rows, dropped = [], 0
    groups = x.groupby(level=by if len(by) > 1 else by[0]) if by else [((), x)]   # one level: a scalar key
    for key, g in groups:
        v = {c: g[c].to_numpy() for c in need}
        r = {REPEAT: _pearson(v["truth", 0], v["truth", 1])}
        r |= {a: 0.5 * (_pearson(v[a, 1], v["truth", 0]) + _pearson(v[a, 0], v["truth", 1])) for a in arms}
        if any(np.isnan(list(r.values()))):
            dropped += 1
            continue
        k = dict(zip(by, key if len(by) > 1 else (key,)))
        rows += [k | dict(arm=a, r=ra, n=len(g)) for a, ra in r.items()]
    if not rows:
        raise ValueError(f"every one of {dropped} groups has a flat vector: r undefined everywhere")
    return pd.DataFrame(rows, columns=[*by, "arm", "r", "n"]), dropped


def double_centre(w: pd.DataFrame) -> pd.DataFrame:
    """Remove each (programme, drug)'s mean over lines, then each (programme, line)'s mean over
    drugs, column by column. What is left is the line-specific response: drug effect + line shift
    (what Additive predicts) goes to 0. Two passes are exact only on a full, finite grid (a NaN
    would shift its neighbours' means), so it raises on anything less."""
    lv = [w.index.get_level_values(k).nunique() for k in ("programme", "drug", "cell_name")]
    if not w.index.is_unique or len(w) != int(np.prod(lv)):
        raise ValueError(f"not a full grid: {len(w)} rows for programme x drug x line = {lv}")
    if (bad := int((~np.isfinite(w.to_numpy())).sum())):
        raise ValueError(f"{bad} values not finite")
    w = w - w.groupby(level=["programme", "drug"]).transform("mean")
    return w - w.groupby(level=["programme", "cell_name"]).transform("mean")


# ----------------------------------------------------------------------------
# The cross-batch cube: (pathway, condition, line) per (arm, plate), and r along each of its axes
# ----------------------------------------------------------------------------
# A model conditioned on one plate is scored against the other plate, both ways averaged; the replicate is plate A
# against plate B. A three-plate drug (Afatinib) is scored once per plate pair ("layer") and r averaged over layers.

def cube(ws: list[pd.DataFrame], arms: list[str]) -> tuple[list[dict], list[str], list[str], dict]:
    """(per layer {(arm, slot): (P, C, L) array}, conditions, lines, {side: (C, L) mask}). A side's mask is the
    (condition, line)s it has in every layer and pathway, as axis-corr drops a unit from all layers."""
    if not ws or any(w.empty for w in ws):
        raise ValueError("no layer, or an empty one: there is nothing to correlate")
    idx = ws[0].index
    progs, conds, lines = (sorted(idx.get_level_values(k).unique()) for k in ("programme", "drug", "cell_name"))
    full, shape = pd.MultiIndex.from_product([progs, conds, lines]), (len(progs), len(conds), len(lines))
    layers = [{c: w[c].reindex(full).to_numpy(float).reshape(shape) for c in w.columns} for w in ws]
    if any(np.isnan(lay["truth", s]).any() for lay in layers for s in (0, 1)):
        raise RuntimeError("a measured value is missing: the grid is not full")
    has = {REPEAT: np.ones(shape[1:], bool)} | {a: np.logical_and.reduce(
        [np.isfinite(lay[a, s]).all(axis=0) for lay in layers for s in (0, 1)]) for a in arms}
    return layers, conds, lines, has


def pairs(side: str) -> list[tuple]:
    """The (prediction, truth) columns whose r is averaged into a side's number."""
    return [(("truth", 0), ("truth", 1))] if side == REPEAT else [((side, 1), ("truth", 0)), ((side, 0), ("truth", 1))]


# Two weighted Pearson formulas, each the one its ED was published with: they agree to round-off, not to the bit.
def weighted_r(w: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson r over units weighted by their draw counts: w (B, n), x and y (n, k) -> (B, k)."""
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("a non-finite value: the side's mask should have dropped its unit")
    s, sx, sy = w.sum(1, keepdims=True), w @ x, w @ y
    cov = s * (w @ (x * y)) - sx * sy
    return cov / np.sqrt((s * (w @ (x * x)) - sx ** 2) * (s * (w @ (y * y)) - sy ** 2))


def wpearson(w: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The same, centred first, along the last axis (the Hallmark ED's Spearman)."""
    s = w.sum(-1, keepdims=True)
    da, db = a - (w * a).sum(-1, keepdims=True) / s, b - (w * b).sum(-1, keepdims=True) / s
    return (w * da * db).sum(-1) / np.sqrt((w * da * da).sum(-1) * (w * db * db).sum(-1))


def r_over(x: np.ndarray, y: np.ndarray, ax: int) -> np.ndarray:
    """Pearson r along axis ``ax`` of two equal arrays (unweighted)."""
    xc, yc = x - x.mean(ax, keepdims=True), y - y.mean(ax, keepdims=True)
    return (xc * yc).sum(ax) / np.sqrt((xc ** 2).sum(ax) * (yc ** 2).sum(ax))


# Each statistic maps (layer, side, weights per unit (B, C, L) or per drug (B, C), the side's mask) to a value per
# draw: one draw of ones is the published number.
def profile(lay: dict, side: str, wu: np.ndarray, has: np.ndarray) -> np.ndarray:
    """r over the pathways of each (condition, line), each line's weighted mean over its conditions, then the mean
    over lines."""
    v, wm = np.mean([r_over(lay[p], lay[t], 0) for p, t in pairs(side)], axis=0), wu * has
    if np.isnan(v[has]).any():           # nan_to_num below may only zero the units outside the mask
        raise RuntimeError(f"{side}: a flat pathway profile, its r is undefined")
    if (wm.sum(1) == 0).any():
        raise RuntimeError("a resample drew no unit of some line")
    return ((wm * np.nan_to_num(v)).sum(1) / wm.sum(1)).mean(1)


def compound(lay: dict, side: str, wu: np.ndarray, has: np.ndarray, rank: bool = False) -> np.ndarray:
    """r over the drawn conditions of each (line, pathway), the mean over pathways, then over lines. ``rank``:
    Spearman, each unit ranked among the drawn ones (mid-ranks of its weight)."""
    out = []
    for li in range(has.shape[1]):
        m, wl = has[:, li], wu[:, has[:, li], li]
        if rank:
            out.append(np.mean([np.mean([wpearson(wl, midranks(wl, lay[p][k, m, li]), midranks(wl, lay[t][k, m, li]))
                                         for p, t in pairs(side)], axis=0) for k in range(lay["truth", 0].shape[0])],
                               axis=0))
        else:
            rs = [weighted_r(wl, lay[p][:, m, li].T, lay[t][:, m, li].T) for p, t in pairs(side)]
            out.append(np.mean(rs, axis=0).mean(1))
    return np.mean(out, axis=0)


def line(lay: dict, side: str, wd: np.ndarray, has: np.ndarray) -> np.ndarray:
    """r over the five lines of each (pathway, condition), its weighted mean (not the double-centred
    pooled r); only conditions every line has. A group's r is fixed under a resample of conditions."""
    keep = has.all(axis=1)
    r = np.mean([r_over(lay[p][:, keep], lay[t][:, keep], 2) for p, t in pairs(side)], axis=0).mean(0)
    return wd[:, keep] @ r / wd[:, keep].sum(1)


def midranks(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Mid-ranks of x (n,) among the drawn units, weights w (B, n): a unit's rank is the weight below it plus (its
    weight + 1) / 2, tie groups sharing their mean rank."""
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("no unit, or a non-finite value: argsort would rank a NaN above every number")
    o = np.argsort(x, kind="stable")
    starts = np.flatnonzero(np.r_[True, np.diff(x[o]) != 0])
    sums = np.add.reduceat(w[:, o], starts, axis=1)                           # weight per tie group
    grp = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(x)]))
    out = np.empty_like(w)
    out[:, o] = (np.cumsum(sums, 1) - sums + (sums + 1) / 2)[:, grp]
    return out


def ranked(layers: list[dict], ax: int) -> list[dict]:
    """Every column ranked along ``ax``: Spearman where a resample cannot change the ranks."""
    return [{c: rankdata(v, axis=ax) for c, v in lay.items()} for lay in layers]


# ----------------------------------------------------------------------------
# The CIs: Figure 2's draws (src.stats.bootstrap), the statistic recomputed per draw
# ----------------------------------------------------------------------------

def axis_ci(layers: dict, conds: list[str], lines: list[str], has: dict, sides: list[str], stats: dict,
            ref: dict, atol: float) -> pd.DataFrame:
    """Per (method, axis, side): r and the percentile 95% CI over cluster_draws, every side sharing each draw.
    Clusters: line x drug (doses pooled), or the drug on the line axis (each r runs over every line). ``layers``:
    {method: its layers}; ``stats``: {(method, axis): statistic}. Refused unless one draw of every cluster gives
    ``ref`` (the analysis's own table) for every side."""
    drug = np.array([c.split(" @ ")[0] for c in conds])
    k_lxd, inv_lxd = np.unique(np.array([[f"{ln}|{d}" for ln in lines] for d in drug]), return_inverse=True)
    k_d, inv_d = np.unique(drug, return_inverse=True)
    lxd = (len(k_lxd), inv_lxd.reshape(drug.size, len(lines)), "cell line x drug")
    units = {"profile": lxd, "compound": lxd, "line": (len(k_d), inv_d, "drug")}
    rows = []
    for (method, ax), f in stats.items():
        n_c, inv, unit = units[ax]
        draws = [d[:, inv] for d in (np.ones((1, n_c)), cluster_draws(n_c))]
        for side in sides:
            point, boot = (np.mean([f(lay, side, w, has[side]) for lay in layers[method]], axis=0) for w in draws)
            if not np.isclose(point[0], ref[method, ax, side], rtol=0, atol=atol) or not np.isfinite(boot).all():
                raise RuntimeError(f"{method}/{ax}/{side}: one draw of every cluster gives {point[0]:.6f}, the "
                                   f"analysis {ref[method, ax, side]:.6f}; or a resample's r is undefined")
            lo, hi = np.percentile(boot, [TAIL, 100 - TAIL])
            rows.append(dict(method=method, axis=ax, arm=side, r=point[0], ci95_lo=lo, ci95_hi=hi, cluster=unit,
                             n_clusters=n_c, ci95_B=B, ci95_seed=SEED))
        print(f"  {method} {ax}: " + "  ".join(f"{x['arm']} {x['r']:.3f}" for x in rows[-len(sides):]), flush=True)
    return pd.DataFrame(rows)


def profile_marginals(layers: list[dict], conds: list[str], lines: list[str], has: dict, arms: list[str]
                      ) -> pd.DataFrame:
    """The across-pathways r per held-out line (the mean over its conditions; CI over the line x drug draws) and per
    compound (the mean over its lines and doses; no CI), every pair layer and both directions averaged."""
    drug = np.array([c.split(" @ ")[0] for c in conds])
    k, inv = np.unique([[f"{ln}|{d}" for ln in lines] for d in drug], return_inverse=True)
    inv, draws, rows = inv.reshape(len(conds), len(lines)), cluster_draws(len(k)), []
    for side in [REPEAT, *arms]:
        h = has[side]
        v = np.where(h, np.mean([np.mean([r_over(lay[p], lay[t], 0) for p, t in pairs(side)], axis=0)
                                 for lay in layers], axis=0), 0.0)                     # (C, L)
        means = []
        for wts in (np.ones((1, len(k))), draws):
            w = wts[:, inv] * h                                                       # (B, C, L)
            if (w.sum(1) == 0).any():
                raise RuntimeError("a resample drew no unit of some line")
            means.append((w * v).sum(1) / w.sum(1))                                  # (B, L)
        lo, hi = np.percentile(means[1], [TAIL, 100 - TAIL], axis=0)
        rows += [dict(level="line", key=ln, arm=side, r=float(means[0][0, li]), ci95_lo=lo[li], ci95_hi=hi[li],
                      n_units=int(h[:, li].sum()), ci95_B=B, ci95_seed=SEED) for li, ln in enumerate(lines)]
        for d in sorted(set(drug), key=str.casefold):
            if not (m := (drug == d)[:, None] & h).any():
                raise RuntimeError(f"{d}/{side}: no unit left to average")
            rows.append(dict(level="compound", key=d, arm=side, r=float(v[m].mean()), n_units=int(m.sum())))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# ED_program_summary: the PROGENy cube of Figure 2a's plate pairs, against axis-corr's own tables
# ----------------------------------------------------------------------------

ED05_ARMS = ["genesis", "state", "additive", "nn_own"]


@cache
def progeny_cube(ac: Path, heldout: frozenset[str], programmes: frozenset[str]) -> tuple:
    """Every held-out test condition on two or more plates, the replicate gene gate (both plates' controls detect
    the gene), one layer per pair, from ``ac``/pair_values__depth.parquet. Raises unless each layer holds every
    condition x ``heldout`` line x ``programmes`` pathway of truth."""
    v = pd.read_parquet(ac / "pair_values__depth.parquet")
    v = v[(v.gate == "replicate") & v.arm.isin(["truth", *ED05_ARMS])]
    v = v.assign(drug=v.drug + " @ " + v.dose.map("{:g} uM".format))   # the condition
    n = v.groupby(["cell_name", "drug"]).pair.transform("nunique")
    if not set(n.unique()) <= {1, int(n.max())}:
        raise RuntimeError(f"conditions carry {sorted(n.unique())} pairs: layers need one count besides 1")
    want = dict(drug=set(v.drug), cell_name=set(heldout), programme=set(programmes))
    ws = []
    for k in range(int(n.max())):
        w = v[v.pair == np.where(n > 1, k, 0)].set_index(["programme", "drug", "cell_name", "arm", "slot"]).v \
            .unstack(["arm", "slot"])                                                  # raises on a duplicate
        if any(set(w.index.get_level_values(x)) != want[x] for x in want) or w["truth"].isna().any().any():
            raise RuntimeError(f"layer {k} is not every condition x line x pathway of truth")
        ws.append(w)
    return cube(ws, ED05_ARMS)


def ed05_axis_ci(ac: Path, heldout: frozenset[str], programmes: frozenset[str]) -> pd.DataFrame:
    layers, conds, lines, has = progeny_cube(ac, heldout, programmes)
    ref = {("pearson", ax, r.arm): r.r for ax in ("profile", "compound", "line")
           for r in pd.read_csv(ac / f"{ax}_summary.csv").query("level == 'overall'").itertuples()}
    t = axis_ci({"pearson": layers}, conds, lines, has, [REPEAT, *ED05_ARMS],
                {("pearson", "profile"): profile, ("pearson", "compound"): compound, ("pearson", "line"): line},
                ref, 2e-6)
    return t[["axis", "arm", "r", "cluster", "n_clusters", "ci95_lo", "ci95_hi", "ci95_B", "ci95_seed"]]


def ed05_marginals(ac: Path, heldout: frozenset[str], programmes: frozenset[str]) -> pd.DataFrame:
    """Refused unless the per-line r are axis-corr's per-line means and the per-compound r its per-unit table's."""
    layers, conds, lines, has = progeny_cube(ac, heldout, programmes)
    m = profile_marginals(layers, conds, lines, has, ED05_ARMS)
    pub = pd.read_csv(ac / "profile_summary.csv").query("level == 'cell_name'").set_index(["key", "arm"]).r
    unit = pd.read_csv(ac / "profile_per_group.csv")
    unit = unit.assign(key=unit.compound.str.split(" @ ").str[0]).groupby(["key", "arm"]).r.mean()
    want = pd.concat([pub, unit]).reindex(list(zip(m.key, m.arm))).to_numpy()
    if not (np.abs(m.r.to_numpy() - want) <= 2e-6).all():
        raise RuntimeError("a per-line or per-compound r is not axis-corr's (or axis-corr lacks it)")
    return m


# ----------------------------------------------------------------------------
# ED_tahoe_hallmark_gsea: the Hallmark NES cube at one dose, against hallmark_gsea_axes' own statistics
# ----------------------------------------------------------------------------

GSEA_ARMS = ["genesis", "additive", "copy"]          # hallmark_gsea_agreement's sides
GSEA_NAMES = {"copy": "copy_closest_15"}             # its response copy takes Figure 4's name
GSEA_AXES = {"profile": ["drug", "cell_name"], "compound": ["programme", "cell_name"], "line": ["programme", "drug"]}


def gsea_layer(rows: pd.DataFrame, N: pd.DataFrame, k: int, terms: list[str]) -> pd.DataFrame:
    """hallmark_gsea_axes.cube, cross-plate: the k-th plate pair of every condition on two or more plates."""
    s = []
    for (cell, drug, dose), g in rows.groupby(["cell_name", "drug", "dose"], sort=False):
        if len(g := g.assign(p=g.plate.str.removeprefix("plate").astype(int)).sort_values("p")) >= 2:
            pp = list(itertools.combinations(g.row.astype(int), 2))
            s.append(dict(cell_name=cell, drug=f"{drug} @ {dose:g} uM", t=pp[k % len(pp)]))
    s = pd.DataFrame(s)
    w = pd.DataFrame({(a, i): N.loc[[(t[i], a) for t in s.t], terms].to_numpy().T.ravel()
                      for a in ["truth", *GSEA_ARMS] for i in (0, 1)},
                     index=pd.MultiIndex.from_arrays([np.repeat(terms, len(s)), np.tile(s.drug, len(terms)),
                                                      np.tile(s.cell_name, len(terms))],
                                                     names=["programme", "drug", "cell_name"]))
    w.columns = pd.MultiIndex.from_tuples(w.columns, names=["arm", "slot"])
    return w


@cache
def gsea_cube(folder: Path, dose: float) -> tuple:
    """(cube, ref): the ``dose`` layers (5 uM: the lower doses barely replicate on Hallmark) of the
    pathways finite everywhere, and each (method, axis, side)'s r, averaged over Afatinib's three
    pairs: the Hallmark reference every CI's point must reproduce."""
    rows = pd.read_csv(folder / "plates.tsv", sep="\t", dtype={"plate": str})
    N = pd.read_parquet(folder / "nes.parquet", columns=["row", "side", "term", "nes"]).pivot_table(
        index=["row", "side"], columns="term", values="nes", dropna=False)
    rows = rows[np.isclose(rows.dose, dose)]
    w0 = gsea_layer(rows, N, 0, list(N.columns))
    terms = [t for t in N.columns if np.isfinite(w0.xs(t, level="programme").to_numpy()).all()]
    if len(terms) != len(N.columns):
        print(f"  Hallmark sets not finite in every cross-plate cell, dropped: {sorted(set(N.columns) - set(terms))}")
    ws, ref = [gsea_layer(rows, N, k, terms) for k in range(3)], {}
    for m, (ax, by) in itertools.product(("pearson", "spearman"), GSEA_AXES.items()):
        per = [cross_batch_r(w.groupby(level=by).rank() if m == "spearman" else w, by, GSEA_ARMS)[0]
               .groupby("arm").r.mean() for w in ws]
        ref |= {(m, ax, s): float(np.mean([p[s] for p in per])) for s in [REPEAT, *GSEA_ARMS]}
    return cube(ws, GSEA_ARMS), ref


@cache
def gsea_axis_ci(folder: Path, dose: float) -> pd.DataFrame:
    """Pearson, and Spearman: ranks within a condition on the pathway and line axes (fixed under a resample of
    conditions), ranks among the drawn conditions on the compound axis."""
    (layers, conds, lines, has), ref = gsea_cube(folder, dose)
    rank = lambda ax, f: lambda lay, s, w, h: f(ranked([lay], ax)[0], s, w, h)  # noqa: E731
    stats = {("pearson", "profile"): profile, ("pearson", "compound"): compound, ("pearson", "line"): line,
             ("spearman", "profile"): rank(0, profile),
             ("spearman", "compound"): lambda lay, s, w, h: compound(lay, s, w, h, rank=True),
             ("spearman", "line"): rank(2, line)}
    t = axis_ci({"pearson": layers, "spearman": layers}, conds, lines, has, [REPEAT, *GSEA_ARMS], stats, ref, 1e-9)
    return t.replace({"arm": GSEA_NAMES})


def gsea_margins(folder: Path, dose: float) -> pd.DataFrame:
    """ED_program_summary's marginals on the Hallmark layers (ranked within a condition for Spearman); both the
    per-line and the per-compound r must average back to the across-pathways bar."""
    (layers, conds, lines, has), _ = gsea_cube(folder, dose)
    bars, out = gsea_axis_ci(folder, dose).query("axis == 'profile'").set_index(["method", "arm"]).r, []
    for method, lay in (("pearson", layers), ("spearman", ranked(layers, 0))):
        m = profile_marginals(lay, conds, lines, has, GSEA_ARMS).assign(method=method)
        got = m.replace({"arm": GSEA_NAMES}).groupby(["level", "arm"]).r.mean()
        if not all(np.isclose(got[lv, a], bars[method, a], rtol=0, atol=1e-9) for lv, a in got.index):
            raise RuntimeError(f"{method}: a per-line or per-compound mean is not the across-pathways bar")
        out.append(m)
    return pd.concat(out, ignore_index=True).replace({"arm": GSEA_NAMES})
