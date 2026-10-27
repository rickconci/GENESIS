"""The context analysis's shared arithmetic: per-plate gates, the pathway and gene-set score, the depth
and plate-background corrections, the shared / line-specific split, and the DepMap slices. One recipe for both
arms, so they compare on the drugs and lines they share: the 50-line atlas arm (truth only, figure4_ctx_atlas.py)
and the test arm (5 held-out lines, measured and GENESIS, figure4_ctx_scores.py).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.figures.figure4.extract.progeny import FLOOR, progeny

MIN_DRUGS = 15      # a plate background estimated on fewer drugs is not a background
DEPMAP = Path("results/external/depmap_23q4")    # under the repo: the raw DepMap 23Q4 download
C3A, HEPG2 = "ACH-001021", "ACH-000739"  # C3A is absent from 23Q4 expression and CRISPR: parental is the proxy


# ----------------------------------------------------------------------------
# 1. Gate the genes: a gene is readable on a plate if that plate's controls detect it
# ----------------------------------------------------------------------------
#   gate        a gene counts on a plate where that plate's controls detect it (>= FLOOR of cells)
#   score       v = sum(w x) / sum|w| over the gated members; NaN (never 0) where none is gated
#   depth       a constant per-gene shift d moves a score by d * sum(w)/sum|w|, so it is removed exactly
#   background  e = v - the median v of the plate's drugs not annotated to that readout

def plate_gate(det: pd.DataFrame, plate: str, floor: float = FLOOR) -> np.ndarray:
    """Per gene, True iff `plate`'s control pool (of one line) detects it in >= `floor` of cells.
    Per plate, not per line: requiring every plate let one weak control pool veto a gene
    on all 14 plates of a line (C32 plate 3 alone removed 9 PROGENy genes). Every condition on a plate, and
    that plate's background, share this one gate, so the background still cancels control noise exactly."""
    rows = det.xs(plate, level=-1) if isinstance(det.index, pd.MultiIndex) else det.loc[[plate]]
    return (rows.to_numpy() >= floor).all(axis=0)


# ----------------------------------------------------------------------------
# 2. Score the readouts: the bundle's v = sum(w x) / sum|w| over gated members
# ----------------------------------------------------------------------------

def readouts(progeny_tsv: Path, gene_sets: Path, panel: set) -> pd.DataFrame:
    """(readout, gene, weight) for the 14 PROGENy pathways (``progeny_tsv``: the bundle's OmniPath download)
    plus every scored gene set (``gene_sets``: apriori/gene_sets.tsv), on one panel. Gene sets are unweighted
    signed, so a set and a pathway are the same kind of number."""
    prog = progeny(progeny_tsv, panel).rename(columns={"programme": "readout"})[["readout", "gene", "weight"]]
    gs = pd.read_csv(gene_sets, sep="\t", comment="#")
    gs = gs[gs.status.isin(("candidate", "calibration")) & gs.gene.isin(panel)]
    gs = gs.rename(columns={"set": "readout", "sign": "weight"})[["readout", "gene", "weight"]]
    return pd.concat([prog, gs.astype({"weight": float})], ignore_index=True)


def score(lfc: np.ndarray, gate: np.ndarray, weights: pd.DataFrame,
          genes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(v, n_genes) for every row of `lfc` against every readout in `weights`. Ungated genes contribute
    nothing to numerator or denominator; a readout with no gated member is NaN, never 0 — "not measured
    here" and "measured, no change" must not be the same number."""
    gi = {g: i for i, g in enumerate(genes)}
    reads = list(dict.fromkeys(weights.readout))
    W = np.zeros((len(genes), len(reads)))
    for j, r in enumerate(reads):
        g = weights[weights.readout == r]
        W[[gi[x] for x in g.gene], j] = g.weight
    Wg = W * gate[:, None]
    num, den = np.where(gate, lfc, 0.0) @ Wg, np.abs(Wg).sum(0)
    n = np.broadcast_to((Wg != 0).sum(0), (len(lfc), len(reads)))
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan), n


def weight_ratio(weights: pd.DataFrame, reads: list[str]) -> np.ndarray:
    """r = sum(w)/sum|w| per readout: how much a uniform per-gene shift moves the score. 1 for an
    all-positive set (the most depth-exposed configuration), 0 for a balanced one."""
    g = weights.groupby("readout").weight.apply(lambda w: w.sum() / w.abs().sum())
    return g.reindex(reads).to_numpy()


def adjust_depth(v: np.ndarray, depth: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Remove the library-depth component. Subtracting a constant delta from every gene's log2FC moves a
    score by delta*r exactly, so this equals a per-gene depth subtraction without touching the genes."""
    return v - depth[:, None] * r[None, :]


# ----------------------------------------------------------------------------
# 3. Build a reference: the plate's own drugs, minus the ones aimed at this pathway
# ----------------------------------------------------------------------------

def background(df: pd.DataFrame, exclude: set, min_drugs: int = MIN_DRUGS,
               max_drop: float = 0.05) -> tuple[pd.DataFrame, int]:
    """(b, n_dropped). b per (cell_name, plate, dose, readout) is the median v over that plate's drugs
    that are NOT annotated to that readout; excluding the annotated ones matters because plates 1-3 carry
    ~27% MAPK-axis inhibitors, which would drag the MAPK background toward the response being measured.

    Dose is nearly determined by plate (each plate ran one dose, with a few off-dose stragglers), so a
    small tail of groups holds 3-10 drugs. Those conditions get no background and are dropped rather than
    scored against noise — 0.8% of the atlas. A larger loss means the grouping is wrong, and raises."""
    keep = [(d_, r_) not in exclude for d_, r_ in zip(df.drug, df.readout)]
    d = df[pd.Series(keep, index=df.index)]
    g = d.groupby(["cell_name", "plate", "dose", "readout"], sort=False)
    out = g.v.agg(b="median", n_bg="size").reset_index()
    thin = out[out.n_bg < min_drugs]
    lost = int(thin.n_bg.sum())
    if lost > max_drop * max(len(d), 1):
        raise SystemExit(f"{len(thin)} background groups have too few drugs (< {min_drugs}), "
                         f"covering {lost} of {len(d)} rows (> {max_drop:.0%}), e.g.\n{thin.head()}")
    return out[out.n_bg >= min_drugs].reset_index(drop=True), lost


# ----------------------------------------------------------------------------
# 4. Split the effect: what every line shares, and what one line does alone
# ----------------------------------------------------------------------------

def decompose(e: pd.DataFrame) -> pd.DataFrame:
    """Per (drug, dose, readout): beta = the median effect over lines (what is shared) and gamma = each
    line's departure from it. Reported as two numbers, never combined into one index: a drug can be
    strong-and-uniform, weak-and-specific, or both, and one score cannot say which."""
    g = e.groupby(["drug", "dose", "readout"], sort=False)
    beta = g.e.median().rename("beta")
    j = e.join(beta, on=["drug", "dose", "readout"])
    j["gamma"] = j.e - j.beta
    k = j.groupby(["drug", "dose", "readout"], sort=False)
    out = k.agg(beta=("beta", "first"), n_lines=("e", "size"),
                gamma_spread=("gamma", lambda x: x.max() - x.min())).reset_index()
    idx = k.gamma.apply(lambda x: x.abs().idxmax())
    out["gamma_max_line"] = j.loc[idx.to_numpy(), "cell_name"].to_numpy()
    out["gamma_max"] = j.loc[idx.to_numpy(), "gamma"].to_numpy()
    return out


# ----------------------------------------------------------------------------
# 5. Read DepMap: slices of its matrices, and a manifest of what was frozen
# ----------------------------------------------------------------------------

def depmap_slice(csv: Path, rows: list[str], genes: list[str]) -> pd.DataFrame:
    """Read only `genes` (symbol, matching DepMap's 'SYM (entrez)' headers) for `rows` (ModelIDs) from a
    400+ MB DepMap matrix. A requested gene DepMap lacks raises: it would otherwise vanish from a score."""
    header = pd.read_csv(csv, nrows=0).columns
    col = {h.split(" (")[0]: h for h in header[1:]}
    if (missing := [g for g in genes if g not in col]):
        raise SystemExit(f"{csv.name}: genes absent from DepMap: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    m = pd.read_csv(csv, usecols=[header[0]] + [col[g] for g in genes], index_col=0)
    # usecols returns FILE order, not request order: select by header name before renaming, or every
    # gene silently takes its neighbour's values (this once put BRAF under 'MDM2').
    m = m[[col[g] for g in genes]]
    m.columns = genes
    return m.reindex(rows)


def write_manifest(apriori: Path) -> None:
    """apriori/MANIFEST.tsv: the sha256 of every a priori table, so a report can prove what was frozen."""
    rows = [dict(file=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            for p in sorted(apriori.glob("*.tsv")) if p.name != "MANIFEST.tsv"]
    pd.DataFrame(rows).to_csv(apriori / "MANIFEST.tsv", sep="\t", index=False)
