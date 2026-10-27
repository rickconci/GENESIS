"""Figure 4's arithmetic: the gates, the PROGENy pathway members and the pathway score that every
Figure 4 extract shares. Pure functions of arrays and a bundle's own files: nothing reads results/ (ATLAS only
names it), and only fetch_progeny() the network; the extracts read the predictions and caches, then call these.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

# The shipped setting: control-only detection gate, conditions of >= 50 cells, GENESIS trained on 15% of
# conditions (HBSS), plates A+B mixed.
#   gene gate  a gene counts in a sample where the controls detect it: mean detection over the plates >= FLOOR
#   pathway    v = sum(w * log2FC) / sum(|w|) over the gated members (w = 1: the curated module mean)
#   error      mean |arm - truth| over the conditions both have (mae_table); predict-0 = mean |truth|
FLOOR, DOSE, MIN_CELLS = 0.10, 5.0, 50
COPY_TABLE = "copy_closest_line__cosine_pca100__hbss015.tsv"
MIX = -1                 # slot of the plate A+B mixture; 0 / 1 = plate A / B alone
TOP = 100                # PROGENy genes per pathway: the PROGENy paper's and R package's default
PROGENY_URL = "https://omnipathdb.org/annotations?resources=PROGENy&license=academic&format=tsv"
# The frozen 5,750-gene response atlas (truth log2FC per Tahoe test condition), a level-3 input read from
# results/; it is not part of the release, and the producer that reads it says so when it is absent.
ATLAS = Path(__file__).resolve().parents[4] / "results/tahoe/union5750/response_atlas"
# Grouped a priori by what drives each pathway, not by the data.
GROUPS = [("Growth factor", ["EGFR", "MAPK", "PI3K", "VEGF"]),
          ("Stress", ["Hypoxia", "p53", "Trail"]),
          ("Inflammation", ["JAK-STAT", "NFkB", "TNFa", "TGFb"]),
          ("Hormone / dev.", ["Androgen", "Estrogen", "WNT"])]
N_REAL = 3066            # the 3,072 panel's last 6 genes are pad


def safe(s: object) -> str:
    """src.data.data_utils.safe_filename_part: the atlas spells 'HepG2/C3A' 'HepG2_C3A'."""
    return re.sub(r"[^\w.\-]+", "_", str(s).strip()).strip("_")[:180] or "unknown"


def gated_pairs(up: Path) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """{drug: (lower plate, higher plate)} at 5 uM where the plates replicate (DEG >= 0.2), and the 5
    held-out lines. ``up``: a bundle's data/upstream/. The pair is PER DRUG: Trametinib is 3/12, not 6/14."""
    p = pd.read_parquet(up / "tahoe_bio/pair_metrics__genesis.parquet")
    p = p[(p.true_true__deg_adaptivek_truth >= 0.2) & np.isclose(p.dose, DOSE)]
    pairs = {}
    for drug, g in p.groupby("drug"):
        ps = sorted({str(x) for r in g.itertuples() for x in (r.plate_i, r.plate_j)}, key=int)
        if len(ps) != 2:
            raise SystemExit(f"{drug}: {len(ps)} gated plates {ps}, expected a pair")
        pairs[drug] = (ps[0], ps[1])
    return pairs, sorted(p.cell_name.unique())


# ----------------------------------------------------------------------------
# 1. Read PROGENy: the database fixes both the rows and their genes
# ----------------------------------------------------------------------------

def fetch_progeny(path: Path) -> None:
    """Download the PROGENy human model from OmniPath (licence academic: the default commercial view returns
    under a fifth of the model). Network: only an extract calls this, and only when ``path`` is missing."""
    req = Request(PROGENY_URL, headers={"User-Agent": "PertGen-figure4-progeny/1.0"})
    path.write_bytes(urlopen(req, timeout=600).read())


def progeny(path: Path, panel: set) -> pd.DataFrame:
    """(set, group, programme, gene, weight): the top TOP genes per pathway by p-value over the
    whole human model (``path``: the OmniPath download), then restricted to the panel (not re-ranked on it)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: an extract fetches it once (progeny.fetch_progeny)")
    t = (pd.read_csv(path, sep="\t", dtype={"value": str})
         .pivot(index=["record_id", "genesymbol"], columns="label", values="value").reset_index()
         .astype({"weight": float, "p_value": float}))
    # One gene under two UniProt IDs is listed twice with identical values: one copy is the gene.
    t = t.drop_duplicates(["pathway", "genesymbol", "weight", "p_value"]).sort_values(
        ["pathway", "p_value"], kind="stable")
    if (tie := t.groupby("pathway").p_value.agg(lambda p: p.iloc[TOP - 1] == p.iloc[TOP])).any():
        raise SystemExit(f"p-value tie at rank {TOP}, the cut is arbitrary: {list(tie[tie].index)}")
    t = t.groupby("pathway").head(TOP)
    if (dup := t[t.duplicated(["pathway", "genesymbol"], keep=False)]).size:
        raise SystemExit(f"conflicting weights for one gene in a top-{TOP} set:\n{dup}")
    if sorted(p for _g, ps in GROUPS for p in ps) != sorted(t.pathway.unique()):
        raise SystemExit(f"GROUPS must name each of the {t.pathway.nunique()} pathways once")
    on = t.genesymbol.isin(panel)
    print(f"  PROGENy: {len(t)} members, {on.sum()} on the panel")
    t = t[on]
    return pd.DataFrame([dict(set="grid", group=g, programme=p, gene=r.genesymbol, weight=r.weight)
                         for g, ps in GROUPS for p in ps for r in t[t.pathway == p].itertuples()])


# ----------------------------------------------------------------------------
# 2. Score the pathways: every sample x every pathway at once
# ----------------------------------------------------------------------------

def score(keys: pd.DataFrame, lfc: np.ndarray, gate: np.ndarray, sets: pd.DataFrame, genes: list[str],
          centre: bool, shift: np.ndarray | None = None) -> pd.DataFrame:
    """Per (sample, programme): v = sum(w * x) / sum(|w|) over the gated members, n_genes, and
    (truth mixture only) the gated members by name. x = log2FC, or with ``centre`` log2FC minus
    the sample's mean over all its gated genes, or with ``shift`` log2FC minus that sample's number
    (the depth split). A programme with no gated member in a sample has no row there."""
    gi = {g: i for i, g in enumerate(genes)}
    progs = list(dict.fromkeys(sets.programme))
    W = np.zeros((len(genes), len(progs)))
    for j, p in enumerate(progs):
        g = sets[sets.programme == p]
        W[[gi[x] for x in g.gene], j] = g.weight
    if centre and shift is not None:
        raise ValueError("centring already removes any per-sample shift: pass one or the other")
    x = np.where(gate, lfc - (lfc.sum(1) / gate.sum(1))[:, None], 0.0) if centre else lfc
    x = np.where(gate, x - shift[:, None], 0.0) if shift is not None else x
    num, den, n = x @ W, gate @ np.abs(W), gate.astype(int) @ (W != 0)
    out = []
    for j, p in enumerate(progs):
        members = np.array(sets.gene[sets.programme == p])
        cols = np.array([gi[m] for m in members])
        t = keys.assign(programme=p, v=num[:, j] / np.where(n[:, j] > 0, den[:, j], 1.0), n_genes=n[:, j])
        t["genes"] = ""
        tm = ((t.arm == "truth") & (t.slot == MIX)).to_numpy()
        t.loc[tm, "genes"] = [",".join(members[gate[i, cols]]) for i in np.flatnonzero(tm)]
        out.append(t[t.n_genes > 0])
    return pd.concat(out, ignore_index=True)


def depth(keys: pd.DataFrame, conds: dict, library: Path) -> np.ndarray:
    """Per response row: log2(mean transcripts per cell, perturbed / same-plate controls), from the h5ad's
    tscp_count over every gene, not the panel (``library``: library_size__tahoe.tsv). A mixture pools its
    plates as its log2FC does: the sum of the plates' means over the sum of their controls'. Measured, so
    every arm's row gets truth's number."""
    lib = pd.read_csv(library, sep="\t")
    lib = lib[np.isclose(lib.dose, DOSE)].assign(cell_name=lambda x: x.cell_name.map(safe))
    lib = lib.set_index(["cell_name", "drug", "plate"])[["mean_tscp", "ctrl_mean_tscp"]]
    out = {}
    for c, d, slot in keys[["cell_name", "drug", "slot"]].drop_duplicates().itertuples(index=False):
        ps = [f"plate{p}" for p in conds[d]]
        r = lib.loc[[(c, d, p) for p in (ps if slot == MIX else [ps[slot]])]]   # KeyError: no tscp for it
        out[(c, d, slot)] = np.log2(r.mean_tscp.sum() / r.ctrl_mean_tscp.sum())
    return np.array([out[k] for k in zip(keys.cell_name, keys.drug, keys.slot)])


def assert_pad_ungated(gate: np.ndarray, n_real: int = N_REAL) -> None:
    """The panel's last genes are pad: none may pass the gate, or it would enter coverage and the centred mean.
    The data keep them out (pad control detection <= 0.0042 < 0.10); this says so instead of trusting it."""
    if (n := int(gate[:, n_real:].sum())):
        raise SystemExit(f"{n} gated pad-gene values (columns >= {n_real}): the pad must never be scored")


# ----------------------------------------------------------------------------
# 3. Tabulate the errors and write the files
# ----------------------------------------------------------------------------

def mae_table(mv: pd.DataFrame, ctx: str, cond: str, held: list[str], arms: list[str]) -> pd.DataFrame:
    """Per (programme, held-out context, arm): mean |arm - truth| over the conditions both
    have; predict-0 = mean |truth|. The statistic the summaries draw. Every (programme,
    context, arm) gets a row: a context that measures nothing reads NaN with n = 0."""
    w = mv[mv[ctx].isin(held)].pivot_table(index=["programme", ctx, cond], columns="arm",
                                           values="v", dropna=False)
    out = []
    for p in mv.programme.unique():
        for c in held:
            g = w.loc[(p, c)] if (p, c) in w.index.droplevel(2) else pd.DataFrame(columns=["truth"])
            tr = g.get("truth", pd.Series(dtype=float))
            out.append(dict(programme=p, cell_name=c, arm="predict0", n=int(tr.notna().sum()),
                            mae=float(tr.abs().mean()) if tr.notna().any() else np.nan))
            for m in arms:
                e = (g[m] - tr).abs() if m in g else pd.Series(dtype=float)
                out.append(dict(programme=p, cell_name=c, arm=m, n=int(e.notna().sum()),
                                mae=float(e.mean()) if e.notna().any() else np.nan))
    return pd.DataFrame(out)


def write(data: Path, split: str, files: dict[str, pd.DataFrame], dry: bool) -> None:
    """Write ``data``/<split>/ (``data``: a bundle's data/); an existing file that differs is refused, never
    overwritten."""
    out = data / split
    for name, df in files.items():
        p = out / name
        if p.exists():
            old = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, sep="\t")
            new = df.reset_index(drop=True)
            if p.suffix != ".parquet":                  # compare as the TSV reads back
                new = pd.read_csv(pd.io.common.StringIO(new.to_csv(sep="\t", index=False)), sep="\t")
            pd.testing.assert_frame_equal(old, new, check_dtype=False)
            print(f"  = data/{split}/{name} unchanged")
        elif dry:
            print(f"  would write data/{split}/{name} ({len(df)} rows)")
        else:
            out.mkdir(parents=True, exist_ok=True)
            if p.suffix == ".parquet":
                df.reset_index(drop=True).to_parquet(p, index=False)
            else:
                df.to_csv(p, sep="\t", index=False)
            print(f"  wrote data/{split}/{name} ({len(df)} rows)")
