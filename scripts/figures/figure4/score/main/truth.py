"""Truth for the held-out lines from the n_3072 eval cache, on the 3,072 panel, with the atlas check: a module the
3,072 extracts import (progeny_tables.py, compare_baselines.py); no entrypoint.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
import src.eval.readout.deg_readout as R  # noqa: E402  the cache readers (extract only: src.eval restyles matplotlib)
from scripts.figures.figure4.extract.progeny import ATLAS, DOSE, safe  # noqa: E402  (ATLAS: the extract reads it as T.ATLAS)

PSEUDO = 1e-9   # the atlas's: log2((m + 1e-9) / (c + 1e-9))


def col_stats(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(mean over cells, fraction of cells with a non-zero count) per gene, float64, from raw counts."""
    x = np.asarray(counts)
    if x.shape[0] == 0:
        raise ValueError("empty count matrix")
    if not np.array_equal(x, np.round(x)):
        raise ValueError("counts must be raw integers")
    return x.mean(0, dtype=np.float64), (x != 0).mean(0, dtype=np.float64)


def assert_cache_genes(cache: Path, genes: list[str]) -> None:
    """The cache stores no gene names and its namespace hash ignores order: its manifest's gene list must be
    the panel's, in order, or positional columns would mislabel genes (the atlas check covers only shared genes)."""
    listed = Path(json.loads((Path(cache) / "manifest.json").read_text())["hvg_genes_path"]).read_text().split()
    if listed != list(genes):
        raise SystemExit(f"{cache}: manifest gene list ({len(listed)}) is not the panel's gene list ({len(genes)}) in order")


@dataclass(frozen=True)
class Truth:
    """Keys use the safe line spelling ('HepG2_C3A') and 'plateN', as the atlas does."""
    genes: list[str]
    cmean: dict   # (cell, plate) -> control mean count [G]
    D: dict       # (cell, plate) -> control detection [G]
    pmean: dict   # (cell, drug, plate) -> perturbed mean count [G], at 5 uM
    n_cells: dict  # (cell, drug, plate) -> perturbed cells


def _load_line(cell: str, conds: dict) -> tuple[dict, dict, dict, dict]:
    """One held-out line (display spelling): the axis's conditions and the controls of their plates."""
    panel = R.resolve_panel("progeny3072")
    assert_cache_genes(panel.cache, panel.genes)
    G, cs, cmean, D, pmean, n = len(panel.genes), safe(cell), {}, {}, {}, {}
    for drug, plates in conds.items():
        for p in plates:
            plate = f"plate{p}"
            if (cs, plate) not in cmean:
                cmean[(cs, plate)], D[(cs, plate)] = col_stats(R.load_control(panel.cache, cell_name=cell, plate=plate, n_genes=G))
            x = R.load_perturbed(panel.cache, cell_name=cell, drug=drug, dose=DOSE, plate=plate, n_genes=G)
            pmean[(cs, drug, plate)], n[(cs, drug, plate)] = col_stats(x)[0], x.shape[0]
    print(f"  truth {cell}: {len(pmean)} conditions, {len(cmean)} control pools", flush=True)
    return cmean, D, pmean, n


def load_truth(lines: list[str], conds: dict) -> Truth:
    """Truth for the held-out ``lines`` (display spelling) at ``conds`` (drug_axis.conditions: drug -> its
    5 uM plates), one process per line.

    Per (line, plate): the control pool's mean count and detection (fraction of cells with count != 0);
    per (line, drug, plate) of the drug axis at 5 uM: the perturbed mean count and its cell count. Raw integer
    counts, float64 means: the response atlas's definitions, on the 3,072 panel, which holds every
    PROGENy member the atlas's 5,750 panel lacked. The cache holds only the five held-out lines,
    which is all this figure draws.
    """
    with ProcessPoolExecutor(len(lines), mp_context=get_context("spawn")) as ex:
        parts = list(ex.map(_load_line, lines, [conds] * len(lines)))
    merged = [{k: v for part in parts for k, v in part[i].items()} for i in range(4)]
    return Truth(R.resolve_panel("progeny3072").genes, *merged)


def check_against_atlas(t: Truth, atlas: Path) -> None:
    """Raise unless, on the genes the 3,072 and 5,750 panels share, control mean and detection
    and every condition's log2FC equal the atlas's (float32 storage tolerance). ``atlas``: the response
    atlas folder (progeny.ATLAS): the same cells read twice, so the two paths must agree."""
    ag = (atlas / "atlas_genes.txt").read_text().split()
    shared = sorted(set(ag) & set(t.genes))
    ia, it = np.array([ag.index(g) for g in shared]), np.array([t.genes.index(g) for g in shared])
    ck = pd.read_parquet(atlas / "atlas_ctrl_keys.parquet")
    drow = {(c, p): i for i, (c, p) in enumerate(zip(ck.cell_name, ck.plate))}
    cm, cd = np.load(atlas / "atlas_ctrl_mean.npy", mmap_mode="r"), np.load(atlas / "atlas_ctrl_det.npy", mmap_mode="r")
    for key in t.cmean:
        r = drow[key]   # KeyError: a pool the atlas never measured
        if not (np.allclose(t.cmean[key][it], cm[r, ia], rtol=1e-6, atol=1e-7)
                and np.allclose(t.D[key][it], cd[r, ia], rtol=0, atol=1e-6)):
            raise SystemExit(f"{key}: control mean/detection differ from the atlas on shared genes")
    K = pd.read_parquet(atlas / "atlas_keys.parquet")
    K = K[np.isclose(K.dose, DOSE)]
    krow = {(c, d, p): i for i, c, d, p in zip(K.index, K.cell_name, K.drug, K.plate)}
    L = np.load(atlas / "atlas_lfc.npy", mmap_mode="r")
    worst = 0.0
    for (c, d, p), m in t.pmean.items():
        lfc = np.log2((m[it] + PSEUDO) / (t.cmean[(c, p)][it] + PSEUDO))
        worst = max(worst, float(np.abs(lfc - L[krow[(c, d, p)], ia]).max()))
    if worst > 1e-5:
        raise SystemExit(f"log2FC differs from the atlas on shared genes: max |diff| {worst:.3g}")
    print(f"  ✓ atlas reproduced on {len(shared)} shared genes: {len(t.cmean)} control pools, "
          f"{len(t.pmean)} conditions (max |Δlog2FC| {worst:.1e})")
