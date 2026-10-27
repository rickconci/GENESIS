#!/usr/bin/env python3
"""Step 0 of Figure 4's context analysis: check the hand-curated a priori tables (context_analysis/data/apriori/)
against the data they make claims about and freeze them, with the derived tables and DepMap slices the scoring
steps read.
    python scripts/figures/figure4/score/main/ctx_apriori.py --bundle <bundle>
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
from scripts.figures.figure4.extract.context import C3A, DEPMAP, HEPG2, depmap_slice, write_manifest  # noqa: E402
from scripts.figures.figure4.extract.progeny import ATLAS, N_REAL, progeny, safe  # noqa: E402

PANEL3072 = REPO / "core_util_data/tahoe/gene_subsets/STATE_HVG_plus_progeny_pad3072.txt"
FLOOR = 5             # members on the panel for a set to be scored (leverage, not variance: see the plan)
MDM2_DEPENDENT = -0.5  # DepMap's conventional Chronos dependency cut; the test lines sit far from it


def read_tsv(path: Path) -> pd.DataFrame:
    """The a priori tables carry '#' header comments that explain themselves; pandas skips them."""
    return pd.read_csv(path, sep="\t", comment="#", dtype=str).fillna("")


# ----------------------------------------------------------------------------
# Expectations: mechanism class -> pathway, with hand overrides
# ----------------------------------------------------------------------------

def apply_moa_map(meta: pd.DataFrame, moa: pd.DataFrame) -> pd.DataFrame:
    """One row per drug. Precedence drug > target > moa: a hand drug row is the most specific claim and
    Tahoe's `moa-fine` the least (139 of the 268 test drugs are 'unclear'). A drug matching no rule gets
    no expectation and an empty `exclude_from`, so it stays IN every pathway's background."""
    unknown = set(moa.loc[moa.match_on == "drug", "key"]) - set(meta.drug)
    if unknown:
        raise SystemExit(f"moa_pathway_map drug rows name drugs not in Tahoe: {sorted(unknown)}")
    by = {kind: moa[moa.match_on == kind].set_index("key") for kind in ("drug", "target", "moa")}
    rows = []
    for r in meta.rename(columns={"moa-fine": "moa_fine"}).itertuples():   # 'moa-fine' is no identifier
        targets = [t.strip() for t in str(r.targets or "").split(",") if t.strip() and t.strip() != "None"]
        hit, rule = None, ""
        if r.drug in by["drug"].index:
            hit, rule = by["drug"].loc[r.drug], "drug"
        elif (t := next((t for t in targets if t in by["target"].index), None)):
            hit, rule = by["target"].loc[t], "target"
        elif r.moa_fine in by["moa"].index:
            hit, rule = by["moa"].loc[r.moa_fine], "moa"
        rows.append(dict(drug=r.drug, moa_fine=r.moa_fine, targets=", ".join(targets), rule=rule,
                         **{c: (hit[c] if hit is not None else "") for c in
                            ("expect_pathway", "expect_sign", "exclude_from", "context", "tier")}))
    out = pd.DataFrame(rows)
    out.loc[out.expect_sign == "", "expect_sign"] = "none"
    return out


# ----------------------------------------------------------------------------
# Gene sets: measurable on the panel, and how much of PROGENy they already are
# ----------------------------------------------------------------------------

def check_gene_sets(sets: pd.DataFrame, panel: set, progeny: pd.DataFrame, floor: int = FLOOR) -> pd.DataFrame:
    """Per set: members on `panel`, and the largest fraction of those members inside ONE PROGENy pathway
    (membership only picks candidates; the realized correlation in Step 2b sets the final status).
    Raises when the hand `status` contradicts the panel, because then the frozen table is wrong."""
    members = progeny.groupby("programme").gene.apply(set)
    rows = []
    for name, g in sets.groupby("set", sort=False):
        on = [x for x in g.gene if x in panel]
        frac = {p: sum(x in m for x in on) / max(len(on), 1) for p, m in members.items()}
        top = max(frac, key=frac.get)
        status = g.status.iloc[0]
        if status in ("candidate", "calibration") and len(on) < floor:
            raise SystemExit(f"gene set {name!r} is '{status}' but has {len(on)} on the panel (< {floor})")
        if status == "not_measurable" and len(on) >= floor:
            raise SystemExit(f"gene set {name!r} is 'not_measurable' but {len(on)} members are on the panel")
        rows.append(dict(set=name, status=status, n_listed=len(g), n_on_panel=len(on),
                         top_pathway=top, top_frac=frac[top], mean_sign=float(pd.to_numeric(g.sign).mean())))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Compounds: salt twins share one, enantiomers never do
# ----------------------------------------------------------------------------

def check_twins(twins: pd.DataFrame, drugs: set) -> dict[str, str]:
    """drug -> compound for every twin. Raises on a drug Tahoe lacks, a drug in two compounds, or a
    `not_twin` drug placed in a compound (an enantiomer merged into its racemate leaks a different drug)."""
    if (missing := set(twins.drug) - drugs):
        raise SystemExit(f"compound_twins names drugs not in Tahoe: {sorted(missing)}")
    real = twins[twins.kind != "not_twin"]
    if (dup := real.drug[real.drug.duplicated()].tolist()):
        raise SystemExit(f"drugs listed in two compounds: {dup}")
    if (bad := set(real.drug) & set(twins.drug[twins.kind == "not_twin"])):
        raise SystemExit(f"not_twin drugs placed in a compound: {sorted(bad)}")
    return dict(zip(real.drug, real.compound))


# ----------------------------------------------------------------------------
# Line features: drivers, p53 competence, baseline pathway activity
# ----------------------------------------------------------------------------

def p53_competence(mdm2: pd.Series, thr: float = MDM2_DEPENDENT) -> pd.Series:
    """Competent iff the line depends on MDM2 (Chronos < thr): a cell needs MDM2 only to restrain live p53.
    Chosen over the TP53 gene effect, which does not separate (PANC-1, a known R273H LoF line, scores
    below Hs 766T on it). A missing line raises: absence from CRISPR must never read as 'not competent'."""
    if mdm2.isna().any():
        raise SystemExit(f"no MDM2 CRISPR value for: {list(mdm2.index[mdm2.isna()])}")
    return mdm2 < thr


def baseline_progeny(expr: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """PROGENy's intended use: z-score each gene across lines, then v = Σw·z / Σ|w| over present members.
    A gene constant across lines has no z; it is set to 0 so it neither helps nor poisons the score."""
    if len(expr) < 2:
        raise SystemExit(f"baseline PROGENy needs at least 2 lines to z-score across, got {len(expr)}")
    sd = expr.std(ddof=0)
    z = (expr - expr.mean()).div(sd.where(sd > 0)).fillna(0.0)
    out = {}
    for p, g in weights.groupby("programme"):
        g = g[g.gene.isin(z.columns)]
        out[p] = z[g.gene.tolist()].to_numpy() @ g.weight.to_numpy() / np.abs(g.weight).sum()
    return pd.DataFrame(out, index=expr.index)


# ----------------------------------------------------------------------------
# PROGENy's own training: which of its agents we also have, and in which of our lines it trained
# ----------------------------------------------------------------------------

def lines_in_text(text: str, aliases: dict[str, list[str]]) -> set[str]:
    """Our lines named in a free-text `cells` field, matched on whole tokens and their 2-3 token joins:
    'Hs 766T' and 'Hs766T' both hit, 'Panc-10.05' does not hit PANC-1, 'SH4' does not hit H4."""
    tok = [t for t in "".join(c if c.isalnum() else " " for c in str(text).upper()).split()]
    grams = {"".join(tok[i:i + n]) for n in (1, 2, 3) for i in range(len(tok) - n + 1)}
    alnum = lambda s: "".join(c for c in s.upper() if c.isalnum())
    return {line for line, names in aliases.items() if any(alnum(a) in grams for a in names)}


def match_progeny_agents(train: pd.DataFrame, agent_class: pd.DataFrame, exp: pd.DataFrame,
                         blocked: set[str] | None = None) -> pd.DataFrame:
    """One row per PROGENy (pathway, effect, agent): the Tahoe drugs that are the same compound ('exact',
    by salt-stripped name) or, failing that, the same class ('class', via agent_class: a Tahoe moa-fine
    string, or 'drug:A|B' where Tahoe's annotation is too thin to carry the class), else 'none'.

    `blocked` agents never match by name. PROGENy's 'doxycycline' induces an shBRAF transgene and has
    nothing to do with Tahoe's antibiotic: the shared string would invent an overlap that is not there."""
    strip = lambda s: re.sub(r"\s*\(.*\)\s*$", "", str(s)).strip().lower()
    by_name = exp.groupby(exp.drug.map(strip)).drug.apply(lambda d: sorted(d))
    cls = agent_class.set_index("agent").tahoe_class
    rows = []
    for (pw, eff, ag), g in train.groupby(["pathway", "effect", "agent"], sort=False):
        drugs, match = ([] if ag in (blocked or set()) else by_name.get(strip(ag), [])), "exact"
        if not drugs and ag in cls.index:
            c, match = cls[ag], "class"
            drugs = c[5:].split("|") if c.startswith("drug:") else sorted(exp.drug[exp.moa_fine == c])
        rows.append(dict(pathway=pw, effect=eff, agent=ag, match=match if drugs else "none",
                         tahoe_drugs="; ".join(sorted(drugs)), n_experiments=len(g),
                         cells=" | ".join(sorted(set(g.cells.astype(str))))))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# The build: real inputs, every table checked against the others
# ----------------------------------------------------------------------------

DRIVERS = ["BRAF", "KRAS", "NRAS", "TP53", "PTEN", "SMAD4", "CDKN2A", "PIK3CA", "APC", "CTNNB1",
           "NF1", "STK11", "ARID1A"]
CRISPR_GENES = ["MDM2", "USP7", "PPM1D", "TP53", "BRAF", "KRAS", "NRAS", "MITF", "CTNNB1"]
# PROGENy agents whose name coincides with a Tahoe drug but means something else entirely.
BLOCKED_AGENTS = {"doxycycline"}   # a transgene inducer in PROGENy, the antibiotic in Tahoe
# Our test lines as PROGENy's free-text `cells` field spells them.
LINE_ALIASES = {"C32": ["C32"], "HOP62": ["HOP62", "HOP-62"], "HepG2_C3A": ["HepG2", "Hep G2", "C3A"],
                "Hs_766T": ["Hs 766T", "Hs766T"], "PANC-1": ["PANC-1", "PANC1", "Panc-1"]}


def check_crispr_against_tp53_table(tp53_by_model: pd.Series, tp53_table: pd.DataFrame, tol: float = 1e-3) -> None:
    """The TP53 Chronos value read from CRISPRGeneEffect must equal the one DepMap's TP53 table carries, on
    every model in both. Two sources for one number: a mislabelled column cannot agree with both, which is
    the check the coincidence-prone competence comparison failed to be."""
    ref = tp53_table.dropna(subset=["ModelID", "tp53_crispr_gene_effect"]).set_index("ModelID").tp53_crispr_gene_effect
    both = tp53_by_model.dropna().index.intersection(ref.index)
    if len(both) == 0:
        raise SystemExit("no model shared by the CRISPR slice and the TP53 table: cannot cross-check")
    d = (tp53_by_model[both] - ref[both]).abs()
    if (d > tol).any():
        raise SystemExit(f"CRISPR TP53 disagrees with the TP53 table:\n{d[d > tol]}")
    print(f"  ✓ CRISPR TP53 matches the independent TP53 table on {len(both)} models (max |Δ| {d.max():.1e})")


def line_ids(lines: list[str], up: Path) -> pd.DataFrame:
    """Atlas line -> DepMap model actually used. Joined through the bundle's safe() spelling (the atlas says
    'HepG2_C3A', the metadata 'HepG2/C3A'). C3A is absent from 23Q4 expression and CRISPR, so parental
    Hep G2 stands in, and `model_used` says so on the row. ``up``: context_analysis/data/upstream/."""
    cl = read_tsv_or_parquet(up / "cell_line_metadata.parquet")
    cl["cell_name_safe"] = cl.cell_name.map(safe)
    first = cl.drop_duplicates("cell_name_safe").set_index("cell_name_safe")
    out = pd.DataFrame({"cell_name": lines})
    out["depmap_id"] = out.cell_name.map(first.Cell_ID_DepMap)
    out["organ"] = out.cell_name.map(first.Organ)
    out["model_used"] = out.depmap_id.replace({C3A: HEPG2})
    for g in DRIVERS:
        mutant = set(cl.loc[cl.Driver_Gene_Symbol == g, "cell_name_safe"])
        out[f"mut_{g}"] = out.cell_name.isin(mutant)
    return out


def read_tsv_or_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else read_tsv(path)


def build_line_features(progeny_all, up: Path, atlas: Path, depmap: Path) -> pd.DataFrame:
    """One row per atlas line: drivers, organ, MDM2-based p53 competence (+ the raw Chronos columns behind
    it), and baseline PROGENy on DepMap basal expression. Lines DepMap lacks are printed, never dropped.
    Writes the two DepMap slices it read into ``up``."""
    lines = sorted(pd.read_parquet(atlas / "atlas_keys.parquet").cell_name.unique())
    feat = line_ids(lines, up)
    have = feat.dropna(subset=["model_used"])
    if (absent := sorted(set(feat.cell_name) - set(have.cell_name))):
        print(f"  ! no DepMap model for {absent}: their CRISPR and expression features are NA")
    cr = depmap_slice(depmap / "CRISPRGeneEffect.csv", have.model_used.tolist(), CRISPR_GENES)
    check_crispr_against_tp53_table(cr.TP53, pd.read_csv(up / "tahoe_cellline_tp53_status.csv"))
    cr.to_csv(up / "depmap23q4_crispr_slice.tsv", sep="\t")
    cr.index = have.cell_name.tolist()
    if (nocrispr := cr.index[cr.MDM2.isna()].tolist()):
        print(f"  ! no CRISPR screen for {nocrispr}: p53 competence is NA for them")
    feat = feat.merge(cr.add_prefix("chronos_"), left_on="cell_name", right_index=True, how="left")
    comp = p53_competence(cr.MDM2.dropna())
    feat["p53_competent"] = feat.cell_name.map(comp)
    w = progeny_all(set(pd.read_csv(depmap / "OmicsExpressionProteinCodingGenesTPMLogp1.csv", nrows=0)
                        .columns[1:].str.split(" \\(").str[0]))
    ex = depmap_slice(depmap / "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
                      have.model_used.tolist(), sorted(set(w.gene)))
    ex.to_csv(up / "depmap23q4_expr_progeny_slice.tsv", sep="\t")
    ex.index = have.cell_name.tolist()
    if (noexpr := ex.index[ex.isna().all(axis=1)].tolist()):
        print(f"  ! no DepMap expression for {noexpr}: their baseline PROGENy is NA")
    base = baseline_progeny(ex.dropna(how="all"), w)
    return feat.merge(base.add_prefix("baseline_"), left_on="cell_name", right_index=True, how="left")


def check_against_ordering(feat: pd.DataFrame, apriori: Path) -> None:
    """The ordering table states each test line's competence by hand; line_features computes it. They agree
    or nothing is written (this is where a wrong DepMap join, e.g. PANC 10.05 for PANC-1, would surface)."""
    order = read_tsv(apriori / "cell_line_ordering.tsv").set_index("cell_name")
    got = feat.set_index("cell_name").loc[order.index, "p53_competent"]
    want = order.p53_competent.eq("yes")
    if not got.astype(bool).equals(want):
        raise SystemExit(f"p53 competence disagrees with cell_line_ordering.tsv:\n{pd.DataFrame({'got': got, 'want': want})}")
    print(f"  ✓ test-line p53 competence matches cell_line_ordering.tsv: {dict(got)}")


def main() -> None:
    """Writes drug_expectations, gene_sets_checked, compound_map, line_features and MANIFEST (sha256 of every a
    priori file) into apriori/, and the DepMap slices into data/upstream/. Nothing is written unless every check
    passes; a re-run overwrites only the derived tables."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, required=True, help="reads and writes BUNDLE/context_analysis/data")
    bundle = ap.parse_args().bundle.resolve()
    AP, UP = bundle / "context_analysis/data/apriori", bundle / "context_analysis/data/upstream"
    atlas = ATLAS
    tsv = bundle / "data/upstream/progeny_omnipath_academic.tsv"   # the bundle's PROGENy: top-100 per pathway
    meta = pd.read_parquet(UP / "drug_metadata.parquet")
    exp = apply_moa_map(meta, read_tsv(AP / "moa_pathway_map.tsv"))
    print(f"  expectations: {(exp.rule != '').sum()} of {len(exp)} drugs matched a rule "
          f"({exp.rule.value_counts().to_dict()})")
    panel3072 = set(PANEL3072.read_text().split()[:N_REAL])
    sets = read_tsv(AP / "gene_sets.tsv")
    gs = check_gene_sets(sets, panel3072, progeny(tsv, panel3072))
    panel5750 = set((atlas / "atlas_genes.txt").read_text().split())   # the supporting arm's panel
    gs["n_on_panel_5750"] = [m.gene.isin(panel5750).sum() for _, m in sets.groupby("set", sort=False)]
    cmap = check_twins(read_tsv(AP / "compound_twins.tsv"), set(meta.drug))
    feat = build_line_features(lambda genes: progeny(tsv, genes), UP, atlas, REPO / DEPMAP)
    check_against_ordering(feat, AP)
    train = pd.read_csv(UP / "progeny_training_experiments.tsv", sep="\t")
    pm = match_progeny_agents(train, read_tsv(AP / "progeny_agent_class.tsv"), exp, BLOCKED_AGENTS)
    pm["overlap_lines"] = [sorted(set().union(*(lines_in_text(c, LINE_ALIASES) for c in r.cells.split(" | "))))
                           for r in pm.itertuples()]
    pm["overlap_lines"] = pm.overlap_lines.str.join("; ")
    hit = pm[pm.match != "none"]
    print(f"  PROGENy: {len(pm)} agents, {len(hit)} with a Tahoe counterpart "
          f"({(pm.match == 'exact').sum()} exact); {(pm.overlap_lines != '').sum()} agents trained in one of our lines")
    pm.to_csv(AP / "progeny_match.tsv", sep="\t", index=False)
    exp.to_csv(AP / "drug_expectations.tsv", sep="\t", index=False)
    gs.to_csv(AP / "gene_sets_checked.tsv", sep="\t", index=False)
    pd.DataFrame({"drug": meta.drug, "compound": meta.drug.map(cmap).fillna(meta.drug)}).to_csv(
        AP / "compound_map.tsv", sep="\t", index=False)
    feat.to_csv(AP / "line_features.tsv", sep="\t", index=False)
    write_manifest(AP)
    print(f"  wrote drug_expectations, gene_sets_checked, compound_map, line_features ({len(feat)} lines), MANIFEST")


if __name__ == "__main__":
    main()
