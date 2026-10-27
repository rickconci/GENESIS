"""Figure 4's data: every file under core_deliverables/data/figure4/, rebuilt from the upstream copies in
core_results/figure4/, then checked against that folder's MANIFEST.tsv.

    python -m scripts.figures.figure4.extract.extract            # rebuild in memory and compare; writes nothing
    python -m scripts.figures.figure4.extract.extract --write    # overwrite what differs, and its MANIFEST rows

Rebuilt here: the drug axis (on the held-out split, derived from the GENESIS run's split YAML), the depth-adjusted
PROGENy values (truth from the held-out lines' cells in the n_3072 eval cache), the context analysis's test scores
(from each test condition's mean counts), and the r and 95% CIs of the two cross-batch EDs (crossbatch.py). Frozen,
read as they are, because each needs hours of predictions or cell sums (core_results/MANIFEST.tsv's made_by names
the command): the GENESIS and State mean counts, the context analysis's per-condition means (ctx_pass), the
baselines, the per-plate pathway values the PROGENy ED scores, and the Hallmark NES. The curated stories are
hand-written.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from functools import cache

import numpy as np
import pandas as pd
import yaml

from scripts.figures import CORE
from scripts.figures.figure4.extract import crossbatch as X
from scripts.figures.figure4.extract import context as C
from scripts.figures.figure4.draw.data import AXIS_MOA, MOA_GROUPS, columns
from scripts.figures.figure4.score.main import ctx_scores as S
from src.config.dataset_registry import load_dataset_registry
from src.data.splits.data_class import SplitConfig
from src.data.splits.generation import load_condition_summary, load_split_policy
from src.data.splits.predicates import split_membership_masks
from src.data.splits.schemes import Axes, canon_dose, predicate_pairs
from src.eval.cache.eval_npz_reader import EvalNpzReader
from src.eval.reference.truth_io import read_control, read_perturbed

REPO = CORE.parent
RAW = REPO / "core_results" / "figure4"
OUT = CORE / "data" / "figure4"
PANEL = REPO / "core_util_data/tahoe/gene_subsets/STATE_HVG_plus_progeny_pad3072.txt"   # its last 6 genes are pad
N_REAL, TOP = 3066, 100      # TOP: PROGENy genes per pathway, the PROGENy paper's and R package's default
FLOOR, DOSE, EPS = 0.10, 5.0, 1e-9   # a gene counts where its controls detect it; log2((m + EPS) / (c + EPS))
MIX = -1                     # slot of a drug's plate mixture; 0 / 1 = the lower / higher plate of a replicated pair
ARMS = ["genesis", "state"]
# Grouped a priori by what drives each pathway, not by the data.
GROUPS = [("Growth factor", ["EGFR", "MAPK", "PI3K", "VEGF"]), ("Stress", ["Hypoxia", "p53", "Trail"]),
          ("Inflammation", ["JAK-STAT", "NFkB", "TNFa", "TGFb"]), ("Hormone / dev.", ["Androgen", "Estrogen", "WNT"])]
# The split the Figure 4 GENESIS run trained on: split_config in its config_resolved.yaml (runs/tahoe/bio/biology/
# progeny3072_cells005_keepctrl/*/261016_181841_*)
SPLIT_YAML = REPO / ("core_util_data/splits/tahoe/bio/tahoe_scaling_bio_keepctrl/entity_seed0/"
                     "tahoe_scaling_bio_celln4579207_keepctrl_mhbss005_es0.yaml")


def safe(s: object) -> str:
    """src.data.data_utils.safe_filename_part: the caches spell 'HepG2/C3A' 'HepG2_C3A'."""
    return re.sub(r"[^\w.\-]+", "_", str(s).strip()).strip("_")[:180] or "unknown"


# ----------------------------------------------------------------------------
# The drug axis: one rule on the Tahoe annotation, the plates the test predictions stand on
# ----------------------------------------------------------------------------

@cache
def gated_pairs() -> tuple[dict[str, tuple[str, str]], list[str]]:
    """{drug: (lower plate, higher plate)} at 5 uM where the plates replicate (truth vs truth DEG >= 0.2), and the
    five held-out lines. The pair is PER DRUG: Trametinib is 3/12, not 6/14."""
    p = pd.read_parquet(RAW / "plate_pairs/pair_metrics__genesis.parquet")
    p = p[(p.true_true__deg_adaptivek_truth >= 0.2) & np.isclose(p.dose, DOSE)]
    pairs = {}
    for drug, g in p.groupby("drug"):
        if len(ps := sorted({str(x) for r in g.itertuples() for x in (r.plate_i, r.plate_j)}, key=int)) != 2:
            raise RuntimeError(f"{drug}: {len(ps)} gated plates {ps}, expected a pair")
        pairs[drug] = (ps[0], ps[1])
    return pairs, sorted(p.cell_name.unique())


@cache
def heldout_split() -> pd.DataFrame:
    """(drug, status, doses_test, doses_train) for every drug the condition summary has in the five held-out lines:
    the doses the run's split YAML tests there, and the doses its predicate trains on (the control trains too:
    controls_to_train). status 'test' iff a dose is tested; a drug only in validation reads 'train' with no dose,
    as the context analysis's table (heldout_drug_split.tsv) has it. Raises unless the five lines share one
    drug list and one train pool."""
    policy, registry, held = load_split_policy("tahoe"), load_dataset_registry("tahoe"), gated_pairs()[1]
    split, s = yaml.safe_load(SPLIT_YAML.read_text()), load_condition_summary(policy, registry)
    s = s.assign(train=split_membership_masks(s, SplitConfig.from_dict(split), registry.control_pert)[0],
                 dose=[canon_dose(x, (registry.control_pert["dose"], *policy.doses)) for x in s.dose])
    s = s[s.cell_name.isin(held)]
    pool = {c: frozenset(zip(g.drug, g.dose)) for c, g in s[s.train].groupby("cell_name")}
    drugs = {c: frozenset(g.drug) for c, g in s.groupby("cell_name")}
    if set(pool) != set(held) or len(set(pool.values())) != 1 or len(set(drugs.values())) != 1:
        raise RuntimeError("the five held-out lines do not share one drug list and one train pool")
    test = predicate_pairs(split["test"], Axes.from_registry(registry), policy.doses, held)  # raises on other lines
    doses = lambda d, ps: ";".join(str(x) for x in sorted(x for p, x in ps if p == d))  # noqa: E731
    t = pd.DataFrame([dict(drug=d, doses_test=doses(d, test), doses_train=doses(d, pool[held[0]]))
                      for d in drugs[held[0]]])
    t.insert(1, "status", np.where(t.doses_test != "", "test", "train"))
    return t.sort_values(["status", "drug"], ignore_index=True)


def plates_at_dose(drugs: set[str], genes: list[str]) -> dict[str, tuple[str, ...]]:
    """{drug: its 5 uM plates, ascending}, from the GENESIS test persist's scored conditions (exactly its 4,211
    prediction files). Raises unless all five lines carry the same plates, or if the persist's gene axis is not the
    panel's: its means (arm_means) are positional."""
    blob = json.loads((RAW / "drug_axis/genesis_test_k1_metrics_summary.json").read_text())
    if blob["gene_names"] != genes or blob["n_skipped"] != 0:
        raise RuntimeError("the GENESIS persist's gene axis is not the panel's, or it skipped conditions")
    by_file = {safe(d): d for d in drugs}
    if len(by_file) != len(drugs):
        raise RuntimeError("two drug names share one file spelling")
    seen: dict[str, dict[str, set]] = {}
    for c in (r["condition_keys"] for r in blob["results"]):
        if float(c["dose"]) == DOSE and safe(c["drug"]) in by_file:
            seen.setdefault(by_file[safe(c["drug"])], {}).setdefault(c["cell_name"], set()).add(
                str(c["plate"]).removeprefix("plate"))
    out = {}
    for d in sorted(drugs):
        if len(seen.get(d, {})) != 5 or len(per_line := {frozenset(p) for p in seen[d].values()}) != 1:
            raise RuntimeError(f"{d}: 5 uM plates per line {seen.get(d)}; expected the same plates in all 5 lines")
        out[d] = tuple(sorted(per_line.pop(), key=int))
    return out


@cache
def axis() -> pd.DataFrame:
    """drug_axis.tsv: every drug scored (the axis rule's and the replicated pairs'), its moa-fine, drawn group,
    5 uM plates, replicated, on_axis. The rule: test at 5 uM in the held-out lines and a moa-fine
    in AXIS_MOA; nothing added or removed by hand."""
    pairs, _ = gated_pairs()
    meta = pd.read_parquet(RAW / "drug_axis/drug_metadata.parquet")
    if meta.drug.duplicated().any():
        raise RuntimeError(f"drug_metadata: a drug listed twice: {sorted(meta.drug[meta.drug.duplicated()])}")
    moa = dict(zip(meta.drug, meta["moa-fine"]))
    split = heldout_split()
    keep = {d for d, s in zip(split.drug, split.doses_test) if str(DOSE) in s.split(";") and moa.get(d) in AXIS_MOA}
    if absent := sorted((set(pairs) | keep) - set(moa)):
        raise RuntimeError(f"no Tahoe annotation for {absent}")
    drugs = set(pairs) | keep
    plates = plates_at_dose(drugs, panel())
    if moved := {d: (p, plates[d]) for d, p in pairs.items() if tuple(p) != plates[d]}:
        raise RuntimeError(f"a replicated pair's plates differ from the predictions': {moved}")
    group = {c: g.split("|")[0] for g, steps in MOA_GROUPS for step in steps for c in step}
    order = columns(keep, moa)[0] + sorted(drugs - keep, key=str.casefold)    # drawn order, then the rest
    return pd.DataFrame(dict(drug=order, moa=[moa[d] for d in order],
                             drug_class=[group[moa[d]] if d in keep else "" for d in order],
                             plates=[";".join(plates[d]) for d in order],
                             replicated=[d in pairs for d in order], on_axis=[d in keep for d in order]))


def conditions() -> dict[str, tuple[str, ...]]:
    """{drug: its 5 uM plates}, alphabetical, so no file's row order follows how the drugs are grouped for drawing."""
    t = axis()
    return {d: tuple(p.split(";")) for d, p in sorted(zip(t.drug, t.plates), key=lambda x: x[0].casefold())}


# ----------------------------------------------------------------------------
# The truth: the held-out lines' cells in the n_3072 eval cache, as the atlas builder summarises them
# ----------------------------------------------------------------------------

@cache
def panel() -> list[str]:
    """The 3,072 genes in cache-column order, as deg_readout.resolve_panel reads them."""
    genes = [g.strip() for g in PANEL.read_text().splitlines() if g.strip()]
    if len(genes) != N_REAL + 6:
        raise RuntimeError(f"expected the 3,072 panel, got {len(genes)} genes")
    return genes


def col_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(mean over cells, fraction of cells with a non-zero count) per gene, float64, from raw counts."""
    if x.shape[0] == 0 or not np.array_equal(x, np.round(x)):
        raise ValueError("empty, or not raw integer counts")
    return x.mean(0, dtype=np.float64), (x != 0).mean(0, dtype=np.float64)


@cache
def truth() -> dict[str, dict]:
    """cmean / D: (line, plate) -> control mean count / detection; pmean: (line, drug, plate) -> perturbed mean
    count, at 5 uM, for every scored drug. Keys use the safe line spelling and 'plateN'."""
    ns, genes = RAW / "progeny/eval_cache_n3072", panel()
    # The cache stores no gene names: positional columns are right only if it was built from this very gene file
    if not json.loads((ns / "manifest.json").read_text())["hvg_genes_path"].endswith(str(PANEL.relative_to(REPO))):
        raise RuntimeError("the eval cache was not built from the 3,072 panel's gene list")
    reader, t = EvalNpzReader(ns, cache_size=4), {"cmean": {}, "D": {}, "pmean": {}}

    def read(f, keys: dict) -> np.ndarray:                  # deg_readout.load_control / load_perturbed
        if (x := np.asarray(f(reader, keys, "plate"), np.float32)).shape[1] != len(genes):
            raise RuntimeError(f"{keys}: {x.shape[1]} genes, the panel has {len(genes)}")
        return x
    for cell in gated_pairs()[1]:
        for drug, plates in conditions().items():
            for plate in (f"plate{p}" for p in plates):
                if (safe(cell), plate) not in t["cmean"]:
                    x = read(read_control, {"cell_name": cell, "plate": plate})
                    t["cmean"][safe(cell), plate], t["D"][safe(cell), plate] = col_stats(x)
                x = read(read_perturbed, {"cell_name": cell, "drug": drug, "dose": str(DOSE), "plate": plate})
                t["pmean"][safe(cell), drug, plate] = col_stats(x)[0]
    return t


# ----------------------------------------------------------------------------
# PROGENy and the depth-adjusted pathway score, every sample x every pathway at once
# ----------------------------------------------------------------------------

@cache
def programmes() -> pd.DataFrame:
    """(set, group, programme, gene, weight): the top TOP genes per pathway by p-value over the whole human model
    (the OmniPath download), then restricted to the panel's real genes (not re-ranked on it)."""
    t = (pd.read_csv(RAW / "progeny/progeny_omnipath_academic.tsv", sep="\t", dtype={"value": str})
         .pivot(index=["record_id", "genesymbol"], columns="label", values="value").reset_index()
         .astype({"weight": float, "p_value": float}))
    # One gene under two UniProt IDs is listed twice with identical values: one copy is the gene.
    t = t.drop_duplicates(["pathway", "genesymbol", "weight", "p_value"]).sort_values(["pathway", "p_value"],
                                                                                      kind="stable")
    if (tie := t.groupby("pathway").p_value.agg(lambda p: p.iloc[TOP - 1] == p.iloc[TOP])).any():
        raise RuntimeError(f"p-value tie at rank {TOP}, the cut is arbitrary: {list(tie[tie].index)}")
    t = t.groupby("pathway").head(TOP)
    if (dup := t[t.duplicated(["pathway", "genesymbol"], keep=False)]).size:
        raise RuntimeError(f"conflicting weights for one gene in a top-{TOP} set:\n{dup}")
    if sorted(p for _g, ps in GROUPS for p in ps) != sorted(t.pathway.unique()):
        raise RuntimeError(f"GROUPS must name each of the {t.pathway.nunique()} pathways once")
    t = t[t.genesymbol.isin(set(panel()[:N_REAL]))]
    return pd.DataFrame([dict(set="grid", group=g, programme=p, gene=r.genesymbol, weight=r.weight)
                         for g, ps in GROUPS for p in ps for r in t[t.pathway == p].itertuples()])


def responses(means: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """(keys, log2FC, gate) per held-out line x drug x arm (truth, then ARMS): the mixture of the drug's plates
    (slot MIX) and, for a replicated pair, each plate alone. The gate is the truth controls' detection averaged
    over the plates, the same for every arm; log2FC is 0 outside it."""
    t, rep, keys, lfc, gate = truth(), set(gated_pairs()[0]), [], [], []
    for c in sorted({k[0] for k in t["pmean"]}):
        for drug, plates in conditions().items():
            ps = [f"plate{p}" for p in plates]
            ctl, eps = [t["cmean"][c, p] for p in ps], len(ps) * EPS
            ok = np.mean([t["D"][c, p] for p in ps], axis=0) >= FLOOR
            for arm, ms in [("truth", [t["pmean"][c, drug, p] for p in ps]),
                            *[(m, [means[c, drug, m, int(p)] for p in plates]) for m in ARMS]]:
                # sum(): this summation order reproduces the published table byte for byte
                rows = [(MIX, np.log2((sum(ms) + eps) / (sum(ctl) + eps)))]
                rows += [(s, np.log2((m_ + EPS) / (c_ + EPS))) for s, (m_, c_) in enumerate(zip(ms, ctl))] \
                    if drug in rep else []
                for slot, x in rows:
                    if not np.isfinite(x[ok]).all():
                        raise RuntimeError(f"{c}/{drug}/{arm}/slot {slot}: a gated log2FC is not finite")
                    keys.append((c, drug, arm, slot)), lfc.append(np.where(ok, x, 0.0)), gate.append(ok)
    keys = pd.DataFrame(keys, columns=["cell_name", "drug", "arm", "slot"])
    # one truth mixture per (line, drug), a single-plate pair per replicated one, every arm on truth's rows
    tr, gate, n_lines = keys[keys.arm == "truth"], np.array(gate), len(gated_pairs()[1])
    rows = {a: set(zip(g.cell_name, g.drug, g.slot)) for a, g in keys.groupby("arm")}
    if (tr.slot == MIX).sum() != n_lines * len(conditions()) or (tr.slot != MIX).sum() != 2 * n_lines * len(rep) \
            or any(rows[m] != rows["truth"] for m in ARMS):
        raise RuntimeError("responses are not one mixture per line x drug, two plates per replicated pair, every arm")
    if n := int(gate[:, N_REAL:].sum()):         # the pad must never enter a score
        raise RuntimeError(f"{n} gated pad-gene values")
    return keys, np.array(lfc), gate


def depth(keys: pd.DataFrame) -> np.ndarray:
    """Per response row: log2(mean transcripts per cell, perturbed / same-plate controls), from the h5ad's tscp_count
    over every gene. A mixture pools its plates as its log2FC does. Measured, so every arm's row gets truth's."""
    lib = pd.read_csv(RAW / "progeny/library_size__tahoe.tsv", sep="\t")
    lib = lib[np.isclose(lib.dose, DOSE)].assign(cell_name=lambda x: x.cell_name.map(safe))
    lib, conds, out = lib.set_index(["cell_name", "drug", "plate"])[["mean_tscp", "ctrl_mean_tscp"]], conditions(), {}
    for c, d, slot in keys[["cell_name", "drug", "slot"]].drop_duplicates().itertuples(index=False):
        ps = [f"plate{p}" for p in conds[d]]
        r = lib.loc[[(c, d, p) for p in (ps if slot == MIX else [ps[slot]])]]   # KeyError: no tscp for it
        out[c, d, slot] = np.log2(r.mean_tscp.sum() / r.ctrl_mean_tscp.sum())
    return np.array([out[k] for k in zip(keys.cell_name, keys.drug, keys.slot)])


def score(keys: pd.DataFrame, lfc: np.ndarray, gate: np.ndarray, shift: np.ndarray) -> pd.DataFrame:
    """Per (sample, programme): v = sum(w x) / sum(|w|) over the gated members, x = log2FC minus the sample's depth
    shift; n_genes, and (truth mixture only) the gated members by name. No gated member: no row."""
    sets, gi = programmes(), {g: i for i, g in enumerate(panel())}
    progs = list(dict.fromkeys(sets.programme))
    W = np.zeros((len(gi), len(progs)))
    for j, p in enumerate(progs):
        W[[gi[x] for x in sets.gene[sets.programme == p]], j] = sets.weight[sets.programme == p]
    x = np.where(gate, lfc - shift[:, None], 0.0)
    num, den, n = x @ W, gate @ np.abs(W), gate.astype(int) @ (W != 0)
    tm, out = ((keys.arm == "truth") & (keys.slot == MIX)).to_numpy(), []
    for j, p in enumerate(progs):
        members = np.array(sets.gene[sets.programme == p])
        t = keys.assign(programme=p, v=num[:, j] / np.where(n[:, j] > 0, den[:, j], 1.0), n_genes=n[:, j], genes="")
        t.loc[tm, "genes"] = [",".join(members[gate[i, [gi[m] for m in members]]]) for i in np.flatnonzero(tm)]
        out.append(t[t.n_genes > 0])
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------------
# Figure 4 and ED_celllines_per_model: the depth split's values, coverage and the two lists they need
# ----------------------------------------------------------------------------

def heldout_lines() -> pd.DataFrame:
    """The five held-out lines, safe and display spelling."""
    test = gated_pairs()[1]
    return pd.DataFrame(dict(cell_name=[safe(c) for c in test], display=test))


@cache
def module_values() -> pd.DataFrame:
    """Truth, GENESIS 5% and State 5% on every scored drug, the plate mixture, depth-adjusted. The model means are
    frozen (arm_means__progeny3072.npz, a superset of these conditions)."""
    z = np.load(RAW / "progeny/arm_means__progeny3072.npz")
    M = z["means"]                       # each z[...] decompresses the whole array anew: read it once
    means = {(c, d, m, int(p)): M[i] for i, (c, d, m, p) in enumerate(zip(z["cell_name"], z["drug"], z["model"],
                                                                           z["pred_plate"]))}
    keys, lfc, gate = responses(means)
    t = score(keys, lfc, gate, depth(keys)).assign(set="grid")
    mv = t[t.slot == MIX].assign(held_out=lambda x: x.cell_name.isin(set(heldout_lines().cell_name)))
    return mv[["set", "programme", "cell_name", "drug", "arm", "v", "n_genes", "genes", "held_out"]]


def coverage() -> pd.DataFrame:
    """Per programme x held-out line: members the controls detect (D >= FLOOR), averaged over the line's (drug,
    plate)s. The figure's '*' marks a mean under 3."""
    t, gi, rows = truth(), {g: i for i, g in enumerate(panel())}, []
    for p, g in programmes().groupby("programme", sort=False):
        cols, per = [gi[m] for m in g.gene], {}
        for c, _drug, plate in t["pmean"]:
            per.setdefault(c, []).append(int((t["D"][c, plate][cols] >= FLOOR).sum()))
        rows += [dict(programme=p, cell_name=c, mean_members=float(np.mean(v))) for c, v in per.items()]
    return pd.DataFrame(rows).assign(held_out=lambda x: x.cell_name.isin(set(heldout_lines().cell_name)))


def additive_values() -> bytes:
    """compare_baselines' Additive and copy_closest_15, depth mixtures (frozen). Refused unless their mean |error|
    per (pathway, line, arm) against the rebuilt truth is the sidecar's (summary_metrics__depth.tsv), row for row."""
    path, mv = RAW / "compare_baselines/depth_module_values.parquet", module_values()
    pred, k = pd.read_parquet(path), ["programme", "cell_name", "arm"]
    ref = mv.loc[mv.arm.eq("truth"), ["programme", "cell_name", "drug", "v"]].rename(columns={"v": "truth"})
    m = pred.merge(ref, on=["programme", "cell_name", "drug"], how="left")
    if pred.v.isna().any() or m.truth.isna().any():
        raise RuntimeError("a baseline value is NaN, or has no measured value")
    got = m.assign(mae=(m.v - m.truth).abs()).groupby(k, sort=False).agg(n=("mae", "size"), mae=("mae", "mean"))
    pub = pd.read_csv(RAW / "compare_baselines/summary_metrics__depth.tsv", sep="\t").set_index(k)
    if not got.index.symmetric_difference(pub.index).empty or not np.array_equal(got.reindex(pub.index).n, pub.n) \
            or not np.allclose(got.reindex(pub.index).mae, pub.mae, rtol=0, atol=1e-6):
        raise RuntimeError("the baselines' mean |error| is not summary_metrics__depth.tsv's")
    return path.read_bytes()


def nn_values() -> bytes:
    """Fig. 2a's nearest neighbour (nn_own), depth mixtures (frozen): one arm, every value finite."""
    path = RAW / "compare_baselines/depth_module_values__nn_own.parquet"
    if set((t := pd.read_parquet(path)).arm) != {"nn_own"} or t.v.isna().any():
        raise RuntimeError(f"{path.name}: arms {sorted(set(t.arm))}, or a NaN value")
    return path.read_bytes()


# ----------------------------------------------------------------------------
# The context analysis's test scores: figure4_ctx_scores.py's arithmetic on the staged means
# ----------------------------------------------------------------------------
# Each test condition's mean counts, both arms, are frozen (ctx_pass: figure4_ctx_means.py read ~95 GiB of
# predictions for them). From the means on, everything is rebuilt.

CTX = RAW / "context_analysis"
KEYS = ["cell_name", "drug", "dose", "plate"]          # a test condition


def test_scores() -> pd.DataFrame:
    """figure4_ctx_scores.main, step for step, on core_results: each plate's gene gate and log2FC, the PROGENy and
    gene-set scores, the measured tscp depth taken off BOTH arms, each arm's own plate background, recovery and
    is_test."""
    keys, means = S.load_pass(CTX / "ctx_pass")
    reader, genes = EvalNpzReader(RAW / "progeny/eval_cache_n3072", cache_size=4), panel()
    ctrl = {(c, p): col_stats(np.asarray(read_control(reader, {"cell_name": c, "plate": p}, "plate"), np.float32))
            for c, p in keys[["cell_name", "plate"]].drop_duplicates().itertuples(index=False)}
    w = C.readouts(RAW / "progeny/progeny_omnipath_academic.tsv", CTX / "gene_sets.tsv", set(genes[:N_REAL]))
    reads = list(dict.fromkeys(w.readout))
    lib = pd.read_csv(RAW / "progeny/library_size__tahoe.tsv", sep="\t").set_index(KEYS).depth_log2fc
    if np.isnan(depth := lib.reindex(pd.MultiIndex.from_frame(keys[KEYS])).to_numpy()).any():
        raise RuntimeError(f"{int(np.isnan(depth).sum())} test conditions have no tscp depth")
    exp = pd.read_csv(CTX / "drug_expectations.tsv", sep="\t").fillna("")
    exclude = {(e.drug, p) for e in exp.itertuples() for p in str(e.exclude_from).split(",") if p}
    tables = []
    for arm in ("truth", "genesis"):
        v, fh = np.full((len(keys), len(reads)), np.nan), np.zeros(len(keys), int)
        for (c, plate), idx in keys.groupby(["cell_name", "plate"]).groups.items():
            rows, (cm, det) = np.asarray(idx), ctrl[c, plate]
            gate = det >= FLOOR
            v[rows] = C.score(S.lfc_from_means(means[arm][rows], cm, gate), gate, w, genes)[0]
            fh[rows] = S.floor_hits(means[arm][rows], gate)
        v = C.adjust_depth(v, depth, C.weight_ratio(w, reads))
        t = keys.assign(arm=arm, floor_hits=fh, **{x: v[:, j] for j, x in enumerate(reads)})
        t = t.melt(id_vars=["arm", *KEYS, "n_cells", "floor_hits"], value_vars=reads, var_name="readout",
                   value_name="v").dropna(subset=["v"])
        t = t[t.plate != "plate12"]           # 5 conditions in the cache: no background can be formed
        bg, lost = C.background(t, exclude)
        print(f"  {arm}: background dropped {lost} rows in thin groups")
        t = t.merge(bg, on=["cell_name", "plate", "dose", "readout"], how="inner")
        tables.append(t.assign(e=t.v - t.b))
    out = pd.concat(tables, ignore_index=True)
    out = out.join(keys.groupby(["cell_name", "plate"]).n_cells.median().rename("plate_median_cells"),
                   on=["cell_name", "plate"])
    out["recovery"] = out.n_cells / out.plate_median_cells
    split = heldout_split()
    test = {(d, float(x)) for d, s in zip(split.drug, split.doses_test) for x in s.split(";") if x}
    return out.assign(is_test=[(d, x) in test for d, x in zip(out.drug, out.dose)])


# ----------------------------------------------------------------------------
# Then every file, compared with MANIFEST.tsv
# ----------------------------------------------------------------------------

DEPTH, ED = "tahoe_progeny_depth", "ED_celllines_per_model"
SPLIT = {"coverage.tsv": coverage, "heldout_lines.tsv": heldout_lines, "module_values.parquet": module_values,
         "programmes.tsv": programmes}
# axis-corr's tables, and the held-out lines and pathways every layer of its cube must hold
ED05 = lambda: (RAW / "axis_correlation", frozenset(heldout_lines().cell_name),  # noqa: E731
                frozenset(programmes().programme))
TABLES = {
    "main/context_analysis/test_scores.parquet": test_scores,
    "main/drug_axis/drug_axis.tsv": axis,
    "main/stories/core_pair.tsv": lambda: (RAW / "stories/core_pair.tsv").read_bytes(),
    "main/stories/curated_picks.tsv": lambda: (RAW / "stories/curated_picks.tsv").read_bytes(),
    **{f"main/{DEPTH}/{f}": build for f, build in SPLIT.items()},
    "ED_program_summary/axis_ci.csv": lambda: X.ed05_axis_ci(*ED05()),
    "ED_program_summary/profile_marginals.csv": lambda: X.ed05_marginals(*ED05()),
    "ED_tahoe_hallmark_gsea/axis_ci.csv": lambda: X.gsea_axis_ci(RAW / "hallmark_gsea", DOSE),
    "ED_tahoe_hallmark_gsea/margins.csv": lambda: X.gsea_margins(RAW / "hallmark_gsea", DOSE),
    f"{ED}/drug_axis/drug_axis.tsv": axis,                                          # the ED draws the same axis
    **{f"{ED}/{DEPTH}/{f}": build for f, build in SPLIT.items()},
    f"{ED}/compare_baselines/depth_module_values.parquet": additive_values,
    f"{ED}/compare_baselines/depth_module_values__nn_own.parquet": nn_values,
    f"{ED}/compare_baselines/summary_metrics__depth.tsv":
        lambda: (RAW / "compare_baselines/summary_metrics__depth.tsv").read_bytes(),
}
# The CI tables were written at 6 significant digits; their counts as integers, even where a NaN makes them float
SIX_DIGITS = {"ED_program_summary/axis_ci.csv", "ED_program_summary/profile_marginals.csv",
              "ED_tahoe_hallmark_gsea/axis_ci.csv", "ED_tahoe_hallmark_gsea/margins.csv"}
COUNTS = ["n_units", "n_clusters", "ci95_B", "ci95_seed"]


def serialize(rel: str, t: pd.DataFrame | bytes) -> bytes:
    if isinstance(t, bytes):
        return t
    if rel.endswith(".parquet"):
        t.reset_index(drop=True).to_parquet(buf := io.BytesIO(), index=False)
        return buf.getvalue()
    if rel in SIX_DIGITS:
        return t.astype({c: "Int64" for c in COUNTS if c in t}).to_csv(index=False, float_format="%.6g").encode()
    return t.to_csv(index=False, sep="\t" if rel.endswith(".tsv") else ",").encode()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="overwrite the files that differ, and their MANIFEST rows")
    write = ap.parse_args().write
    manifest = pd.read_csv(OUT / "MANIFEST.tsv", sep="\t", dtype=str, keep_default_na=False).set_index("path")
    if odd := set(manifest.index) ^ set(TABLES):
        raise RuntimeError(f"MANIFEST.tsv and this script list different files: {sorted(odd)}")
    differ = []
    for rel, build in TABLES.items():
        blob = serialize(rel, build())
        sha, want = hashlib.sha256(blob).hexdigest(), manifest.at[rel, "sha256"]
        state = "same bytes" if sha == want else "DIFFERS"
        # a parquet's bytes carry the writer's library versions; its values are what the figure reads
        if state == "DIFFERS" and rel.endswith(".parquet") and hashlib.sha256((OUT / rel).read_bytes()).hexdigest() == want \
                and pd.read_parquet(io.BytesIO(blob)).equals(pd.read_parquet(OUT / rel)):
            state = "same values"
        print(f"{state:12s} {rel}", flush=True)
        if state == "DIFFERS":
            differ.append(rel)
            if write:
                (OUT / rel).parent.mkdir(parents=True, exist_ok=True)
                (OUT / rel).write_bytes(blob)
                manifest.loc[rel, ["bytes", "sha256"]] = [str(len(blob)), sha]
    if write and differ:
        manifest.reset_index().to_csv(OUT / "MANIFEST.tsv", sep="\t", index=False)
        print(f"wrote {len(differ)} files and their MANIFEST.tsv rows")
    elif differ:
        raise SystemExit(f"{len(differ)} of {len(TABLES)} files differ from MANIFEST.tsv (--write to replace them)")
    else:
        print(f"all {len(TABLES)} files match MANIFEST.tsv")


if __name__ == "__main__":
    main()
