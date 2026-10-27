#!/usr/bin/env python3
"""The expensive pass of Figure 4's context analysis: every test condition's mean counts AND its per-cell pathway
scores, both arms, into context_analysis/data/upstream/ctx_pass/. NEVER-AUTO: it reads ~95 GiB of predictions.

    python scripts/figures/figure4/score/main/ctx_means.py --bundle <bundle>   # 5 workers, one a line
"""
from __future__ import annotations

import argparse
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
import src.eval.readout.deg_readout as R  # noqa: E402  the cache and prediction loaders
from scripts.figures.figure4.extract.context import readouts  # noqa: E402
from scripts.figures.figure4.extract.progeny import N_REAL, safe  # noqa: E402
from src.model.inference_output import to_raw_integer  # noqa: E402  the scorer's snap to the count lattice

RUNS = REPO / "runs/tahoe/bio/biology/progeny3072_cells005_keepctrl"
THREADS = 6                                 # per worker; 5 lines x 6 = 30 of the node's 48 cores
TARGET = 1e4                                # counts per cell before log1p, for the per-cell score
LINES = ["C32", "HOP62", "HepG2/C3A", "Hs 766T", "PANC-1"]


# ----------------------------------------------------------------------------
# A per-cell score: each cell against its own plate's controls
# ----------------------------------------------------------------------------

def _cpm_log(x: np.ndarray) -> np.ndarray:
    """Depth-normalise each cell to TARGET counts, then log1p. Without this a per-cell score would
    mostly re-measure how deeply that cell was sequenced."""
    tot = x.sum(1, keepdims=True)
    return np.log1p(np.divide(x, np.where(tot > 0, tot, 1.0)) * TARGET)


def control_reference(ctrl: np.ndarray) -> dict:
    """Per-gene mean and sd of the plate's control cells in the normalised space. A gene with no spread
    keeps sd 0 here and is dropped at scoring time: patching it to a small number would let one flat
    gene dominate every score."""
    z = _cpm_log(np.asarray(ctrl, dtype=np.float64))
    return {"mean": z.mean(0), "sd": z.std(0, ddof=0)}


def per_cell_scores(cells: np.ndarray, ref: dict, weights: pd.DataFrame, genes: list[str]) -> np.ndarray:
    """[n_cells x n_readouts]: z each gene against the control reference, then the weighted mean over each
    readout's members, Sum(w z)/Sum|w| -- PROGENy's own per-cell use. Genes with no control spread are
    excluded from both numerator and denominator."""
    gi = {g: i for i, g in enumerate(genes)}
    ok = ref["sd"] > 0
    z = np.where(ok, (_cpm_log(np.asarray(cells, dtype=np.float64)) - ref["mean"]) /
                 np.where(ok, ref["sd"], 1.0), 0.0)
    reads = list(dict.fromkeys(weights.readout))
    W = np.zeros((len(genes), len(reads)))
    for j, r in enumerate(reads):
        g = weights[weights.readout == r]
        W[[gi[x] for x in g.gene if x in gi], j] = [w for x, w in zip(g.gene, g.weight) if x in gi]
    W = W * ok[:, None]
    den = np.abs(W).sum(0)
    return (z @ W) / np.where(den > 0, den, 1.0)


# ----------------------------------------------------------------------------
# The pass: one worker per held-out line, one shard per line, resumable
# ----------------------------------------------------------------------------

def test_conditions(panel, genesis: Path) -> pd.DataFrame:
    """Every (line, drug, dose, plate) with a GENESIS test prediction, in display spelling (what the
    loaders take). Built from the prediction files themselves, so nothing absent from disk is listed."""
    rx = re.compile(r"cell_name=(.+?)__drug=(.+?)__dose=(.+?)__plate=plate(\d+)\.npy")
    disk = {rx.match(f.name).groups() for f in (genesis / "predictions").iterdir()}
    run = next(RUNS.glob("*/261016_181841*"))
    summ = pd.read_parquet(run / "eval/condition_summary.parquet")
    summ["key"] = [(safe(c), safe(d), str(float(x)), p.replace("plate", ""))
                   for c, d, x, p in zip(summ.cell_name, summ.drug, summ.dose, summ.plate)]
    t = summ[summ.key.isin(disk)][["cell_name", "drug", "dose", "plate", "num_cells"]].reset_index(drop=True)
    if len(t) != len(disk):
        raise SystemExit(f"{len(disk)} prediction files but {len(t)} matched conditions: names do not round-trip")
    return t


def readout_weights(bundle: Path, panel) -> pd.DataFrame:
    """The 14 PROGENy pathways and the scored gene sets on the panel's real genes (the last 6 are pad)."""
    return readouts(bundle / "data/upstream/progeny_omnipath_academic.tsv",
                    bundle / "context_analysis/data/apriori/gene_sets.tsv", set(panel.genes[:N_REAL]))


def score_line(cell: str, bundle: Path, genesis: Path) -> str:
    """All test conditions of one held-out line -> one shard per arm (means + per-cell scores).
    A process of its own: the five lines run side by side, each reading only its own files."""
    torch.set_num_threads(THREADS)
    panel, shards, tag = R.resolve_panel("progeny3072"), bundle / "context_analysis/data/upstream/ctx_pass", safe(cell)
    if all((shards / f"{tag}__{a}.npz").exists() for a in ("truth", "genesis")):
        return f"{cell}: shards exist, skipped"
    conds = test_conditions(panel, genesis)
    conds = conds[conds.cell_name == cell].reset_index(drop=True)
    w, genes, G = readout_weights(bundle, panel), list(panel.genes), len(panel.genes)
    root, fmt = genesis / "predictions", panel.arms["genesis"][1]
    refs, keep = {}, {a: {"means": [], "cells": [], "cond": [], "n": []} for a in ("truth", "genesis")}
    for i, r in conds.iterrows():
        if r.plate not in refs:
            refs[r.plate] = control_reference(R.load_control(panel.cache, cell_name=cell, plate=r.plate, n_genes=G))
        t = R.load_perturbed(panel.cache, cell_name=cell, drug=r.drug, dose=float(r.dose), plate=r.plate, n_genes=G)
        p = R.load_pred(R.pred_path(root, fmt, r.drug, cell, float(r.dose), int(r.plate.replace("plate", ""))), genes)
        if p.shape != t.shape:
            raise SystemExit(f"{cell}/{r.drug}/{r.dose}/{r.plate}: prediction {p.shape} vs measured {t.shape}")
        pc = to_raw_integer(torch.as_tensor(p, dtype=torch.float32), R.MODEL_SPACE).to(torch.float64).numpy()
        for arm, x in (("truth", np.asarray(t, dtype=np.float64)), ("genesis", pc)):
            keep[arm]["means"].append(x.mean(0))
            keep[arm]["cells"].append(per_cell_scores(x, refs[r.plate], w, genes).astype(np.float32))
            keep[arm]["cond"].append(np.full(len(x), i, dtype=np.int32))
            keep[arm]["n"].append(len(x))
        if i % 100 == 0:
            print(f"    {cell}: {i}/{len(conds)}", flush=True)
    shards.mkdir(parents=True, exist_ok=True)
    reads = np.array(list(dict.fromkeys(w.readout)))
    for arm, k in keep.items():
        np.savez(shards / f"{tag}__{arm}.npz", means=np.array(k["means"]), percell=np.concatenate(k["cells"]),
                 cond=np.concatenate(k["cond"]), n_cells=np.array(k["n"]), readouts=reads,
                 # fixed-width unicode, never object arrays: a shard must load without pickle
                 **{c: np.asarray(conds[c].astype(str).tolist()) for c in ("cell_name", "drug", "dose", "plate")})
    return f"{cell}: {len(conds)} conditions, {sum(keep['truth']['n']):,} cells"


def main() -> None:
    """GENESIS's predictions are stored per cell -- 4,211 test conditions, ~8.3M cells x 3,072 genes, about
    95 GiB on NFS. This reads them once and keeps two things:

      means__{truth,genesis}.npz    per-condition mean counts [n_cond x 3072] -- reproduces the bundle exactly
      percell__{truth,genesis}.npz  per-cell pathway scores [n_cells x n_readouts], ~0.6 GiB an arm

    The per-cell layer is cheap next to the read. It is a DIFFERENT statistic from the figure's log2FC-of-means,
    not a replacement: it answers whether a context effect is every cell shifting a little or a subpopulation
    responding a lot -- the selection confound the one-timepoint design otherwise cannot address. Refuses to
    overwrite.
    """
    from scripts.figures.figure4.score.main import model_means as S   # the GENESIS gene-axis guard (a sibling script)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    panel = R.resolve_panel("progeny3072")
    ap.add_argument("--bundle", type=Path, required=True, help="writes BUNDLE/context_analysis/data/upstream/ctx_pass")
    ap.add_argument("--genesis", type=Path, default=Path(panel.arms["genesis"][0]).parent,
                    help="the GENESIS K=1 test persist (predictions/ + metrics_summary.json); default: the progeny3072 panel's")
    args = ap.parse_args()
    bundle, genesis = args.bundle.resolve(), args.genesis.resolve()
    S.assert_genesis_axis(genesis / "metrics_summary.json", list(panel.genes))
    print(f"  {len(test_conditions(panel, genesis))} test conditions over {len(LINES)} lines; "
          f"shards -> {bundle / 'context_analysis/data/upstream/ctx_pass'}", flush=True)
    with ProcessPoolExecutor(len(LINES), mp_context=get_context("spawn")) as ex:
        for msg in ex.map(score_line, LINES, [bundle] * len(LINES), [genesis] * len(LINES)):
            print(f"  {msg}", flush=True)


if __name__ == "__main__":
    main()
