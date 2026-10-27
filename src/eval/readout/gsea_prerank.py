"""Preranked GSEA of one response vector: an NES per gene set, reproducible between samples.

gseapy.prerank with the choices that make two samples' NES comparable written down:
  ties   broken by gene name, with offsets smaller than the gap between any two distinct scores, so
         prerank never orders tied genes arbitrarily (which would differ between plates for no reason);
  seed   fixed, so every sample is normalised against the same permutation draw;
  sizes  no set-size filter: a set keeps every member present in the ranking, however few. A set with no
         member in the ranking gets a row with NaN, so every sample reports every set.
"""
from __future__ import annotations

import json
from pathlib import Path

import gseapy as gp
import numpy as np
import pandas as pd

N_PERM, SEED = 1000, 123
COLUMNS = ["term", "nes", "nom_p", "fdr", "n_genes", "lead_genes"]


def load_msigdb(path: Path) -> dict[str, list[str]]:
    """An MSigDB JSON collection as {set name: member symbols}."""
    sets = {name: list(entry["geneSymbols"]) for name, entry in json.loads(Path(path).read_text()).items()}
    if not sets or any(not g for g in sets.values()):
        raise ValueError(f"{path}: empty collection or an empty set")
    return sets


def untie(score: pd.Series) -> pd.Series:
    """Scores that sort as (score, gene name) without reordering distinct scores.

    Walk the (score, gene) order and lift every value that does not exceed its predecessor to the next
    representable float above it. A fixed offset (gap / n) fails when two distinct scores are a few ULPs
    apart: the offset then rounds away. Values move by at most a few ULPs per tie.
    """
    if score.isna().any() or not np.isfinite(score.to_numpy(float)).all():
        raise ValueError("scores must be finite: drop ungated genes before ranking")
    if score.index.duplicated().any():
        raise ValueError("a gene appears twice in the ranking")
    z = score.to_numpy(float)
    if np.ptp(z) == 0:
        raise ValueError("every gene has the same score; there is no ranking")
    order = np.lexsort((score.index.to_numpy(str), z))
    v = z[order].copy()
    for k in range(1, len(v)):   # a lifted value can meet the next one, so walk the whole order
        if v[k] <= v[k - 1]:
            v[k] = np.nextafter(v[k - 1], np.inf)
    out = np.empty(len(z))
    out[order] = v
    out = pd.Series(out, index=score.index)
    if out.duplicated().any() or (np.argsort(out.to_numpy(), kind="stable") != order).any():
        raise AssertionError("tie-break failed; prerank would order ties arbitrarily")
    return out


def prerank_nes(score: pd.Series, gene_sets: dict[str, list[str]], *, n_perm: int = N_PERM,
                seed: int = SEED) -> pd.DataFrame:
    """One row per set in ``gene_sets``: NES, nominal p, FDR q, members in the ranking, leading edge."""
    ranked = untie(score)
    present = {k: v for k, v in gene_sets.items() if set(v) & set(ranked.index)}
    rows = pd.DataFrame(columns=COLUMNS)
    if present:
        res = gp.prerank(rnk=ranked, gene_sets=present, permutation_num=n_perm, min_size=1,
                         max_size=len(ranked), threads=1, outdir=None, no_plot=True, seed=seed).res2d
        rows = pd.DataFrame({"term": res.Term, "nes": res.NES.astype(float), "nom_p": res["NOM p-val"].astype(float),
                             "fdr": res["FDR q-val"].astype(float),
                             "n_genes": res["Tag %"].str.split("/").str[1].astype(int), "lead_genes": res.Lead_genes})
    missing = [k for k in gene_sets if k not in set(rows.term)]
    blank = pd.DataFrame({"term": missing, "nes": np.nan, "nom_p": np.nan, "fdr": np.nan, "n_genes": 0, "lead_genes": ""})
    out = pd.concat([rows, blank], ignore_index=True).set_index("term").loc[list(gene_sets)].reset_index()
    return out.astype({"n_genes": int})
