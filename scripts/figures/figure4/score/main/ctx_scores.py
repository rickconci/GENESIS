#!/usr/bin/env python3
"""Test-arm pathway scores of Figure 4's context analysis: measured and GENESIS, five held-out lines, every test
condition, with the 50-line arm's arithmetic -> context_analysis/data/test_scores.parquet.

    python scripts/figures/figure4/score/main/ctx_scores.py --bundle <bundle>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
import src.eval.readout.deg_readout as R  # noqa: E402  the eval cache's control pools
from scripts.figures.figure4.extract import context as C  # noqa: E402
from scripts.figures.figure4.extract.progeny import FLOOR, N_REAL  # noqa: E402

PSEUDO = 1e-9      # the bundle's: log2((m + 1e-9) / (c + 1e-9))


# ----------------------------------------------------------------------------
# Fold changes: the bundle's arithmetic, gated per plate
# ----------------------------------------------------------------------------

def lfc_from_means(means: np.ndarray, ctrl_mean: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """log2((m + 1e-9) / (c + 1e-9)) per gene, 0 where the plate's controls do not detect the gene."""
    return np.where(gate, np.log2((means + PSEUDO) / (ctrl_mean + PSEUDO)), 0.0)


def floor_hits(means: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """Per condition, gated genes whose mean is exactly 0 -- each one reads ~ -30 log2 and can swamp a
    pathway score. Reported, not hidden; the recovery gate removes most of them."""
    return ((means == 0) & gate).sum(1)


# ----------------------------------------------------------------------------
# The build
# ----------------------------------------------------------------------------

def load_pass(shards: Path) -> tuple[pd.DataFrame, dict]:
    """(condition keys, per-arm mean counts) from ctx_means.py's shards, one pair of files per line."""
    keys, means = [], {"truth": [], "genesis": []}
    for f in sorted(shards.glob("*__genesis.npz")):
        g, t = np.load(f), np.load(str(f).replace("__genesis", "__truth"))
        keys.append(pd.DataFrame({c: g[c] for c in ("cell_name", "drug", "dose", "plate")}).assign(n_cells=g["n_cells"]))
        means["genesis"].append(g["means"]), means["truth"].append(t["means"])
    k = pd.concat(keys, ignore_index=True)
    k["dose"] = k.dose.astype(float)
    return k, {a: np.concatenate(v) for a, v in means.items()}


def controls(keys: pd.DataFrame, panel) -> dict:
    """(line, plate) -> (control mean count, control detection) on the 3,072 panel, from the eval cache."""
    out, G = {}, len(panel.genes)
    for line, plate in keys[["cell_name", "plate"]].drop_duplicates().itertuples(index=False):
        x = np.asarray(R.load_control(panel.cache, cell_name=line, plate=plate, n_genes=G), dtype=np.float64)
        out[(line, plate)] = (x.mean(0), (x != 0).mean(0), len(x))
    return out


# Over 40 lines: numerical: the per-arm scoring pass, each step on the last one's arrays, kept verbatim.
def main() -> None:
    """Reads the pass's per-condition means (ctx_means.py -> context_analysis/data/upstream/ctx_pass/) and the
    plates' control pools (the n_3072 eval cache), and applies exactly the arithmetic the 50-line arm uses
    (scripts/figures/figure4/extract/context.py): per-plate gene gate, the bundle's log2FC, PROGENy + gene-set scores, the tscp
    depth subtraction, and a plate background over the drugs not annotated to that pathway. Two differences from
    the atlas arm, both because there is a model:

      - the depth subtraction is the measured tscp ratio, applied to BOTH arms, so truth - GENESIS is unchanged;
      - each arm gets its own background (GENESIS's from its own predictions on that plate), so a model that
        is right about drugs but off by a plate-wide constant is not penalised for the constant.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, required=True, help="reads and writes BUNDLE/context_analysis/data")
    bundle = ap.parse_args().bundle.resolve()
    data = bundle / "context_analysis/data"
    panel = R.resolve_panel("progeny3072")
    keys, means = load_pass(data / "upstream/ctx_pass")
    ctrl = controls(keys, panel)
    genes = list(panel.genes)
    w = C.readouts(bundle / "data/upstream/progeny_omnipath_academic.tsv", data / "apriori/gene_sets.tsv",
                   set(panel.genes[:N_REAL]))
    reads = list(dict.fromkeys(w.readout))
    r = C.weight_ratio(w, reads)
    # depth: the measured tscp ratio of the TRUTH condition, the same number subtracted from both arms
    lib = pd.read_csv(data / "upstream/library_size__tahoe.tsv", sep="\t")
    dep = lib.set_index(["cell_name", "drug", "dose", "plate"]).depth_log2fc
    depth = dep.reindex(pd.MultiIndex.from_frame(keys[["cell_name", "drug", "dose", "plate"]])).to_numpy()
    if np.isnan(depth).any():
        raise SystemExit(f"{int(np.isnan(depth).sum())} test conditions have no tscp depth")
    exp = pd.read_csv(data / "apriori/drug_expectations.tsv", sep="\t").fillna("")
    exclude = {(e.drug, p) for e in exp.itertuples() for p in str(e.exclude_from).split(",") if p}
    tables = []
    for arm in ("truth", "genesis"):
        v = np.full((len(keys), len(reads)), np.nan)
        ng = np.zeros((len(keys), len(reads)), int)
        fh = np.zeros(len(keys), int)
        for (line, plate), idx in keys.groupby(["cell_name", "plate"]).groups.items():
            rows = np.asarray(idx)
            cm, det, _n = ctrl[(line, plate)]
            gate = det >= FLOOR
            lfc = lfc_from_means(means[arm][rows], cm, gate)
            v[rows], ng[rows] = C.score(lfc, gate, w, genes)
            fh[rows] = floor_hits(means[arm][rows], gate)
        v = C.adjust_depth(v, depth, r)
        t = keys.assign(arm=arm, floor_hits=fh, **{x: v[:, j] for j, x in enumerate(reads)})
        t = t.melt(id_vars=["arm", "cell_name", "drug", "dose", "plate", "n_cells", "floor_hits"],
                   value_vars=reads, var_name="readout", value_name="v").dropna(subset=["v"])
        t = t[t.plate != "plate12"]           # 5 conditions in the cache: no background can be formed
        bg, lost = C.background(t, exclude)
        print(f"  {arm}: background dropped {lost} rows in thin groups")
        t = t.merge(bg, on=["cell_name", "plate", "dose", "readout"], how="inner")
        t["e"] = t.v - t.b
        tables.append(t)
    out = pd.concat(tables, ignore_index=True)
    pmed = keys.groupby(["cell_name", "plate"]).n_cells.median().rename("plate_median_cells")
    out = out.join(pmed, on=["cell_name", "plate"])
    out["recovery"] = out.n_cells / out.plate_median_cells
    split = pd.read_csv(data / "apriori/heldout_drug_split.tsv", sep="\t", comment="#").fillna("")
    test = {(d, float(x)) for d, s in zip(split.drug, split.doses_test) for x in s.split(";") if x}
    out["is_test"] = [(d, x) in test for d, x in zip(out.drug, out.dose)]
    out.to_parquet(data / "test_scores.parquet", index=False)
    print(f"  wrote test_scores.parquet: {len(out):,} rows, {out.drug.nunique()} drugs, "
          f"{out.cell_name.nunique()} lines, arms {sorted(out.arm.unique())}")


if __name__ == "__main__":
    main()
