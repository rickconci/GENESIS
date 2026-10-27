#!/usr/bin/env python3
"""Hallmark GSEA agreement between plates, and with GENESIS, Additive and Copy closest, on tahoe-bio test conditions.

    python scripts/figures/figure4/score/ed_tahoe_hallmark_gsea/hallmark_gsea.py --stage all --workers 12

A condition is (held-out line, drug, dose), measured on 1 to 3 plates, each with >= 50 perturbed and 50
control cells. For every plate and side (truth, GENESIS, Additive, Copy closest): the depth-adjusted
pseudobulk log2FC on the condition's gated genes (src/eval/readout/pseudobulk_response.py), then preranked
GSEA on the 50 MSigDB Hallmark sets (src/eval/readout/gsea_prerank.py): 50 NES per plate and side.

Agreement, Pearson and Spearman, of NES:
  per condition, across the 50 pathways
    replicate  true i vs true j, mean over the condition's plate pairs              (two or more plates)
    cross      true i vs arm j, mean over ordered pairs i != j                      (two or more plates)
    self       true p vs arm p, mean over plates: every condition. Arm p and true p share plate p's
               control mean, so self is optimistic; on multi-plate conditions it sits beside cross to
               measure by how much.
  per (line, dose, pathway), across compounds: the same three; summaries average over pathways and lines.

Sides (all from the 15 % HBSS training split of union5750_hbss015_last_test_k1, Figure 4's 5,750-gene GENESIS):
  GENESIS   the persisted predictions, snapped as eval snaps them, against plate p's measured controls;
  Additive  deg_readout's union5750_hbss015 panel artifact, emitted as every baseline is (baseline_pred, in its fit
            space: a log1p_cp10k_full fit also reads plate p's control libraries), against plate p's controls;
  Copy      Figure 2a's rule (src/eval/arms/copy_neighbor.py, v2 deltas) on GENESIS-15 %'s training arms: the
            neighbour line's measured cells for the drug on plate p, against the neighbour's plate-p controls.
Every side is gated on truth's genes and given truth's depth; Copy is also gated on the neighbour's detection.

Stages (each cached under results/tahoe/hallmark_gsea_agreement/; delete a file to recompute it):
  neighbors  the published neighbour table (neighbors.parquet); this tree does not rebuild it
  responses  per plate: every side's pseudobulk means, checked against the eval run's stored truth and GENESIS
             arrays, Figure 4's stored Additive means and the response atlas (Copy)
  gsea       NES per plate and side
  metrics    per_condition.tsv, per_pathway.tsv, summary.tsv
  split      the same-plate reference: each plate's perturbed cells AND its control cells split
             into two random halves; truth half k is scored against control half k, a model compared with truth half
             k against the OTHER control half, so no comparison shares a control cell -> nes_split.parquet
"""
from __future__ import annotations

import argparse
import itertools
import logging
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
from src.data.vocab import to_categorical_str  # noqa: E402
from src.eval.readout.gsea_prerank import load_msigdb, prerank_nes  # noqa: E402
from src.eval.readout.pseudobulk_response import (depth_of, detection_gate, load_depth, log2fc,  # noqa: E402
                                                  response)
from src.paths import expand_path  # noqa: E402

log = logging.getLogger("hallmark_gsea")
H5AD = Path(expand_path("{data:tahoe}"))   # this machine's Tahoe h5ad folder (src/paths.py)
CACHE = H5AD / "eval_cache/normalization_raw_counts/geneset_e639c3a3c608d54a__n_5750"
GENES_TXT = REPO / "core_util_data/tahoe/gene_subsets/STATE_HVG_plus_hallmark_5750.txt"
HALLMARK = REPO / "core_util_data/shared/msigdb_v2026.1.Hs_json_files_to_download_locally/h.all.v2026.1.Hs.json"
LIBRARY = None    # --library: library_size__tahoe.tsv, the measured depth every side is given
GENESIS = REPO / "results/tahoe/split_half/pertgen/union5750_hbss015_last_test_k1"   # --genesis
ADDITIVE_PANEL = "union5750_hbss015"
ADDITIVE_ROOT = None    # --additive-root; main sets it, by default to the panel's registered baseline_root
# Workers are forked, never spawned: they read OUT, GENESIS, ADDITIVE_ROOT and LIBRARY as main set them
FORK = get_context("fork")
FIG4_MEANS = REPO / "results/tahoe/union5750/arm_means__union5750_hbss015.npz"
ATLAS = REPO / "results/tahoe/union5750/response_atlas"
OUT = REPO / "results/tahoe/hallmark_gsea_agreement"

MIN_CELLS = 50
SIDES = ("truth", "genesis", "additive", "copy")
ARMS = SIDES[1:]
MEANS = ("m_truth", "c_truth", "det_truth", "m_genesis", "m_additive", "m_copy", "c_copy", "det_copy")
KEY = ["cell_name", "drug", "dose"]   # a condition; cell_name is the dataset's (display) name


def genes() -> list[str]:
    g = [x.strip() for x in GENES_TXT.read_text().splitlines() if x.strip()]
    if len(g) != 5750:
        raise ValueError(f"{GENES_TXT.name}: {len(g)} genes, expected 5750")
    return g


def slug(cell: str) -> str:
    return cell.replace("/", "_").replace(" ", "_")


# ----------------------------------------------------------------------------
# Conditions: every test plate GENESIS was scored on, gated at 50 cells
# ----------------------------------------------------------------------------

def plate_rows() -> pd.DataFrame:
    """One row per test plate, with the summary's cell counts and Figure 4's >= 50-cell gate."""
    from src.data.splits.generation import load_split_policy
    k = pq.read_table(GENESIS / "metrics_gene_arrays.parquet",
                      columns=["ck__cell_name", "ck__drug", "ck__dose", "ck__plate"]).to_pandas()
    k.columns = ["cell_name", "drug", "dose", "plate"]
    k["dose"] = k.dose.astype(float)
    s = pd.read_parquet(load_split_policy("tahoe").condition_summary)
    s = s.assign(d5=s.dose.astype(float).round(5))
    pert = s.set_index(["cell_name", "drug", "d5", "plate"]).num_cells
    ctrl = s[s.drug == "DMSO_TF"].groupby(["cell_name", "plate"]).num_cells.sum()
    k["n_pert"] = [int(pert[(c, d, round(z, 5), p)]) for c, d, z, p in zip(k.cell_name, k.drug, k.dose, k.plate)]
    k["n_ctrl"] = [int(ctrl[(c, p)]) for c, p in zip(k.cell_name, k.plate)]
    k = k[(k.n_pert >= MIN_CELLS) & (k.n_ctrl >= MIN_CELLS)].sort_values(KEY + ["plate"]).reset_index(drop=True)
    k["n_plates"] = k.groupby(KEY).plate.transform("size")
    k["row"] = np.arange(len(k))
    log.info("%d gated test plates in %d conditions; plates per condition %s", len(k), k.groupby(KEY).ngroups,
             k.groupby(KEY).size().value_counts().sort_index().to_dict())
    return k


# ----------------------------------------------------------------------------
# The neighbour: the published table
# ----------------------------------------------------------------------------

def stage_neighbors(rows: pd.DataFrame) -> pd.DataFrame:
    """The published neighbour table. Rebuilding it picked neighbours, so this tree only reads the cache."""
    path = OUT / "neighbors.parquet"
    if not path.is_file():
        raise SystemExit(f"{path} missing; the copy-neighbour table is a published cache, not rebuilt here")
    return pd.read_parquet(path)


# ----------------------------------------------------------------------------
# Pseudobulks: every side's mean counts per plate
# ----------------------------------------------------------------------------

def _unit(task: tuple) -> dict:
    """One (held-out line, plate): controls once, then every test row on that plate, every side."""
    from src.baselines.space import library_namespace
    from src.eval.cache.eval_npz_reader import EvalNpzReader
    from src.eval.readout import deg_readout as S
    from src.eval.reference.truth_io import control_library
    from src.model.inference_output import to_raw_integer
    cell, plate, rows, copied = task
    reader = EvalNpzReader(CACHE, cache_size=2)
    ctrl = np.asarray(reader.control({"cell_name": cell, "plate": plate}), np.float32)
    gen = S.load_baselines(("additive",), genes=genes(), root=ADDITIVE_ROOT)["additive"]
    space = S.prediction_space(gen)
    lib = (None if space == "raw" else
           control_library(EvalNpzReader(library_namespace(CACHE), cache_size=2), {"cell_name": cell, "plate": plate}, ctrl))
    out = {m: [] for m in MEANS} | {"row": [], "n_loaded": []}
    nb_ctrl: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for r, keys in zip(rows, copied):
        k = {"cell_name": cell, "drug": r["drug"], "dose": to_categorical_str(r["dose"]), "plate": plate}
        truth = np.asarray(reader.perturbed(k), np.float32)
        pred = np.load(S.pred_path(GENESIS / "predictions", "npy", r["drug"], cell, r["dose"], plate.removeprefix("plate")))
        if pred.shape != truth.shape:
            raise ValueError(f"{k}: GENESIS {pred.shape} vs truth {truth.shape}")
        add = S.baseline_pred(gen, "additive", drug=r["drug"], cell=cell, dose=r["dose"], plate=plate.removeprefix("plate"),
                              ctrl_raw=ctrl, n_cells=len(truth), ctrl_lib=lib)
        out["row"].append(r["row"]); out["n_loaded"].append(len(truth))
        out["m_truth"].append(truth.mean(0)); out["c_truth"].append(ctrl.mean(0)); out["det_truth"].append((ctrl > 0).mean(0))
        out["m_genesis"].append(to_raw_integer(pred, "log1p").mean(0)); out["m_additive"].append(to_raw_integer(add, space).mean(0))
        nan = np.full(truth.shape[1], np.nan, np.float32)
        if keys is None:
            out["m_copy"].append(nan); out["c_copy"].append(nan); out["det_copy"].append(nan)
            continue
        if keys["cell_name"] not in nb_ctrl:   # neighbours repeat across drugs on a plate: their controls once
            pool = np.asarray(reader.control({"cell_name": keys["cell_name"], "plate": plate}), np.float32)
            nb_ctrl[keys["cell_name"]] = (pool.mean(0), (pool > 0).mean(0))
        out["m_copy"].append(np.asarray(reader.perturbed(keys), np.float32).mean(0))
        out["c_copy"].append(nb_ctrl[keys["cell_name"]][0]); out["det_copy"].append(nb_ctrl[keys["cell_name"]][1])
    return {k: np.asarray(v, np.float32) if k in MEANS else np.asarray(v) for k, v in out.items()}


def stage_responses(rows: pd.DataFrame, picks: pd.DataFrame, workers: int) -> dict[str, np.ndarray]:
    d = OUT / "responses"
    if all((d / f"{m}.npy").is_file() for m in MEANS):
        return {m: np.load(d / f"{m}.npy", mmap_mode="r") for m in MEANS}
    import ast
    copied = dict(zip(picks.row, picks.copied.map(lambda s: None if s is None else ast.literal_eval(s))))
    tasks = [(c, p, g[["row", "drug", "dose"]].to_dict("records"), [copied[i] for i in g.row])
             for (c, p), g in rows.groupby(["cell_name", "plate"])]
    full = {m: np.full((len(rows), 5750), np.nan, np.float32) for m in MEANS}
    with ProcessPoolExecutor(workers, mp_context=FORK) as pool:
        for i, part in enumerate(pool.map(_unit, tasks), 1):
            if (part["n_loaded"] != rows.n_pert.to_numpy()[part["row"]]).any():
                raise AssertionError("a loaded perturbed pool differs from the summary's cell count")
            for m in MEANS:
                full[m][part["row"]] = part[m]
            log.info("responses %d/%d units", i, len(tasks))
    check_means(rows, picks, full)
    d.mkdir(parents=True, exist_ok=True)
    for m in MEANS:
        np.save(d / f"{m}.npy", full[m])
    return full


def check_means(rows: pd.DataFrame, picks: pd.DataFrame, full: dict) -> None:
    """Truth and GENESIS against the eval run's own arrays (every plate); Additive against Figure 4's stored
    means; Copy's log2FC against the response atlas. Any difference stops the run."""
    cols = ["ck__cell_name", "ck__drug", "ck__dose", "ck__plate", "gs__raw_int_rgs__true_mean",
            "gs__raw_int_rgs__pred_mean", "gs__raw_int_rgs__ctrl_mean", "dropout__raw_int_rgs__ctrl_zero_frac"]
    ev = pq.read_table(GENESIS / "metrics_gene_arrays.parquet", columns=cols).to_pandas()
    ev.index = list(zip(ev.ck__cell_name, ev.ck__drug, ev.ck__dose.astype(float).round(5), ev.ck__plate))
    for r in rows.itertuples():
        e = ev.loc[[(r.cell_name, r.drug, round(r.dose, 5), r.plate)]].iloc[0]
        for ours, theirs in (("m_truth", "true_mean"), ("m_genesis", "pred_mean"), ("c_truth", "ctrl_mean")):
            if not np.allclose(full[ours][r.row], np.asarray(e[f"gs__raw_int_rgs__{theirs}"], float), rtol=1e-4, atol=1e-5):
                raise AssertionError(f"{r.cell_name}/{r.drug}/{r.dose}/{r.plate}: {ours} differs from the eval run")
        if not np.allclose(full["det_truth"][r.row], 1 - np.asarray(e.dropout__raw_int_rgs__ctrl_zero_frac, float), atol=1e-5):
            raise AssertionError(f"{r.cell_name}/{r.drug}/{r.plate}: control detection differs from the eval run")
    log.info("truth, control and GENESIS means equal the eval run's on all %d plates", len(rows))
    check_additive(rows, full)
    check_copy(rows, picks, full)


def check_additive(rows: pd.DataFrame, full: dict) -> None:
    from src.eval.readout import deg_readout as S
    gen = S.load_baselines(("additive",), genes=genes(), root=ADDITIVE_ROOT)["additive"]
    if S.prediction_space(gen) != "raw":    # the stored means are the raw fit's emission: they check only that fit
        log.warning("Additive is a %s fit; %s holds the raw fit's means, so it cannot check this one (NOT checked)",
                    gen.params["fit_space"], FIG4_MEANS.name)
        return
    z = np.load(FIG4_MEANS, allow_pickle=True)
    ref = {(c, d, str(p)): i for i, (c, d, m, p) in enumerate(zip(z["cell_name"], z["drug"], z["model"], z["pred_plate"]))
           if m == "additive"}
    hit = 0
    for r in rows[np.isclose(rows.dose, 5.0)].itertuples():
        i = ref.get((slug(r.cell_name), r.drug, r.plate.removeprefix("plate")))
        if i is None:
            continue
        if np.abs(full["m_additive"][r.row] - z["means"][i]).max() > 1e-4:
            raise AssertionError(f"{r.cell_name}/{r.drug}/{r.plate}: Additive mean differs from Figure 4's")
        hit += 1
    if hit == 0:
        raise AssertionError("no plate overlaps Figure 4's stored Additive means: nothing was checked")
    log.info("Additive means equal Figure 4's on the %d plates it stored", hit)


def check_copy(rows: pd.DataFrame, picks: pd.DataFrame, full: dict) -> None:
    import ast
    if not ATLAS.is_dir():  # a reference outside the release: without it only this check is lost, never a side
        log.warning("reference atlas absent: copy-arm check skipped (%s)", ATLAS)
        return
    K = pd.read_parquet(ATLAS / "atlas_keys.parquet")
    at = {(c, d, round(float(z), 4), p): i for i, (c, d, z, p) in enumerate(zip(K.cell_name, K.drug, K.dose, K.plate))}
    L, hit = np.load(ATLAS / "atlas_lfc.npy", mmap_mode="r"), 0
    for row, s in zip(picks.row, picks.copied):
        if s is None:
            continue
        k = ast.literal_eval(s)
        i = at.get((slug(k["cell_name"]), k["drug"], round(float(k["dose"]), 4), k["plate"]))
        if i is None:
            continue
        ours = log2fc(full["m_copy"][row], full["c_copy"][row])
        if not np.allclose(ours, L[i], rtol=1e-4, atol=1e-4):
            raise AssertionError(f"Copy {k}: log2FC differs from the response atlas")
        hit += 1
    if hit < 0.9 * picks.copied.notna().sum():
        raise AssertionError(f"only {hit} copied arms are in the response atlas: too few to trust the check")
    log.info("Copy log2FC equals the response atlas on %d of %d copied plates", hit, picks.copied.notna().sum())


# ----------------------------------------------------------------------------
# NES: depth-adjusted responses, gated, ranked, one GSEA per plate and side
# ----------------------------------------------------------------------------

_W: dict = {}   # per worker: the means (memory-mapped), the gene axis, the Hallmark sets, the depth table


def _init_gsea() -> None:
    d = OUT / "responses"
    _W.update(means={m: np.load(d / f"{m}.npy", mmap_mode="r") for m in MEANS},
              split={m: np.load(d / f"{m}.npy", mmap_mode="r") for m in SPLIT if (d / f"{m}.npy").is_file()},
              genes=np.array(genes()), sets=load_msigdb(HALLMARK), depth=load_depth(LIBRARY))


def _gsea_condition(task: tuple) -> pd.DataFrame:
    """Every plate and side of one condition: the gate is shared by its plates, the depth is truth's."""
    plates, (cell, drug, dose) = task   # plates: [(row, "plateN"), ...], plain tuples so they pickle
    M, g = _W["means"], _W["genes"]
    gate = detection_gate([M["det_truth"][i] for i, _ in plates])
    has_copy = all(np.isfinite(M["det_copy"][i]).all() for i, _ in plates)
    gate_copy = gate & detection_gate([M["det_copy"][i] for i, _ in plates]) if has_copy else None
    out = []
    for i, plate in plates:
        d = depth_of(_W["depth"], cell, drug, dose, plate)
        for side in SIDES:
            if side == "copy" and gate_copy is None:
                continue
            ctrl, gt = (M["c_copy"][i], gate_copy) if side == "copy" else (M["c_truth"][i], gate)
            x = response(M[f"m_{side}"][i], ctrl, gt, d)
            score = pd.Series(x[gt], index=g[gt])
            out.append(prerank_nes(score, _W["sets"]).assign(row=i, side=side, n_ranked=int(gt.sum()),
                                                             n_extreme=int((np.abs(score) > 10).sum())))
    return pd.concat(out, ignore_index=True)


def stage_gsea(rows: pd.DataFrame, workers: int) -> pd.DataFrame:
    path = OUT / "nes.parquet"
    if path.is_file():
        return pd.read_parquet(path)
    tasks = [(list(zip(g.row.tolist(), g.plate.tolist())), k) for k, g in rows.groupby(KEY, sort=False)]
    parts = []
    with ProcessPoolExecutor(workers, mp_context=FORK, initializer=_init_gsea) as pool:
        for i, part in enumerate(pool.map(_gsea_condition, tasks, chunksize=4), 1):
            parts.append(part)
            if i % 200 == 0 or i == len(tasks):
                log.info("gsea %d/%d conditions", i, len(tasks))
    nes = pd.concat(parts, ignore_index=True)
    ext = nes.drop_duplicates(["row", "side"]).groupby("side").n_extreme.agg(["sum", lambda s: int((s > 0).sum())])
    log.info("genes with |depth-adjusted log2FC| > 10 entering a ranking (total, plates with any):\n%s", ext.to_string())
    nes.drop(columns="lead_genes").to_parquet(path, index=False)
    return nes


# ----------------------------------------------------------------------------
# Split halves: a same-plate reference, with no cell shared between the two sides
# ----------------------------------------------------------------------------

SPLIT = ("m_h1", "m_h2", "c_h1", "c_h2")


def halves(n: int, key: str) -> tuple[np.ndarray, np.ndarray]:
    """A random split of n cells into two halves, fixed by ``key`` (crc32), so a rerun draws the same cells."""
    perm = np.random.default_rng(zlib.crc32(key.encode())).permutation(n)
    return np.sort(perm[: n // 2]), np.sort(perm[n // 2:])


def _split_unit(task: tuple) -> dict:
    """One (held-out line, plate): its control halves once, then every test row's perturbed halves."""
    from src.eval.cache.eval_npz_reader import EvalNpzReader
    cell, plate, rows = task
    reader = EvalNpzReader(CACHE, cache_size=2)
    ctrl = np.asarray(reader.control({"cell_name": cell, "plate": plate}), np.float32)
    c1, c2 = halves(len(ctrl), f"ctrl|{cell}|{plate}")
    c_h1, c_h2 = ctrl[c1].mean(0), ctrl[c2].mean(0)
    out = {m: [] for m in SPLIT} | {"row": []}
    for r in rows:
        k = {"cell_name": cell, "drug": r["drug"], "dose": to_categorical_str(r["dose"]), "plate": plate}
        truth = np.asarray(reader.perturbed(k), np.float32)
        h1, h2 = halves(len(truth), f"pert|{cell}|{r['drug']}|{r['dose']}|{plate}")
        out["row"].append(r["row"]); out["m_h1"].append(truth[h1].mean(0)); out["m_h2"].append(truth[h2].mean(0))
        out["c_h1"].append(c_h1); out["c_h2"].append(c_h2)
    return {k: np.asarray(v, np.float32) if k in SPLIT else np.asarray(v) for k, v in out.items()}


def stage_split_responses(rows: pd.DataFrame, workers: int) -> None:
    d = OUT / "responses"
    if all((d / f"{m}.npy").is_file() for m in SPLIT):
        return
    tasks = [(c, p, g[["row", "drug", "dose"]].to_dict("records")) for (c, p), g in rows.groupby(["cell_name", "plate"])]
    full = {m: np.full((len(rows), 5750), np.nan, np.float32) for m in SPLIT}
    with ProcessPoolExecutor(workers, mp_context=FORK) as pool:
        for i, part in enumerate(pool.map(_split_unit, tasks), 1):
            for m in SPLIT:
                full[m][part["row"]] = part[m]
            log.info("split halves %d/%d units", i, len(tasks))
    whole = np.load(d / "m_truth.npy", mmap_mode="r")
    n = rows.n_pert.to_numpy()[:, None]
    mix = (np.floor(n / 2) * full["m_h1"] + np.ceil(n / 2) * full["m_h2"]) / n   # the halves must rebuild the whole
    if not np.allclose(mix, whole, rtol=1e-4, atol=1e-5):
        raise AssertionError("the two perturbed halves do not average back to the plate's mean")
    for m in SPLIT:
        np.save(d / f"{m}.npy", full[m])
    log.info("split halves rebuild every plate's perturbed mean")


SPLIT_SIDES = {"truth_h1": ("m_h1", "c_h1"), "truth_h2": ("m_h2", "c_h2"),
               "genesis_c1": ("m_genesis", "c_h1"), "genesis_c2": ("m_genesis", "c_h2"),
               "additive_c1": ("m_additive", "c_h1"), "additive_c2": ("m_additive", "c_h2")}


def _gsea_split(task: tuple) -> pd.DataFrame:
    """Every plate of one condition: both truth halves and each model against either control half."""
    plates, (cell, drug, dose) = task
    M, S, g = _W["means"], _W["split"], _W["genes"]
    gate = detection_gate([M["det_truth"][i] for i, _ in plates])
    out = []
    for i, plate in plates:
        d = depth_of(_W["depth"], cell, drug, dose, plate)
        for side, (m, c) in SPLIT_SIDES.items():
            pert = S[m][i] if m in S else M[m][i]
            x = response(pert, S[c][i], gate, d)
            out.append(prerank_nes(pd.Series(x[gate], index=g[gate]), _W["sets"]).assign(row=i, side=side))
    return pd.concat(out, ignore_index=True)


def stage_split_gsea(rows: pd.DataFrame, workers: int) -> None:
    path = OUT / "nes_split.parquet"
    if path.is_file():
        return
    tasks = [(list(zip(g.row.tolist(), g.plate.tolist())), k) for k, g in rows.groupby(KEY, sort=False)]
    parts = []
    with ProcessPoolExecutor(workers, mp_context=FORK, initializer=_init_gsea) as pool:
        for i, part in enumerate(pool.map(_gsea_split, tasks, chunksize=4), 1):
            parts.append(part)
            if i % 200 == 0 or i == len(tasks):
                log.info("split gsea %d/%d conditions", i, len(tasks))
    pd.concat(parts, ignore_index=True).drop(columns="lead_genes").to_parquet(path, index=False)


# ----------------------------------------------------------------------------
# Agreement: per condition across pathways, per pathway across compounds
# ----------------------------------------------------------------------------

METHODS = {"pearson": lambda x, y: pearsonr(x, y)[0], "spearman": lambda x, y: spearmanr(x, y)[0]}
MIN_POINTS = 3


def corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < MIN_POINTS or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return np.nan
    return float(METHODS[method](x[ok], y[ok]))


def per_condition(rows: pd.DataFrame, V: dict, method: str) -> pd.DataFrame:
    """V[(row, side)] = 50 NES. Each condition: replicate and cross (two or more plates), self (always)."""
    out = []
    for (cell, drug, dose), g in rows.groupby(KEY, sort=False):
        P = list(g.row)
        rec = dict(cell_name=cell, drug=drug, dose=dose, n_plates=len(P), method=method)
        if len(P) > 1:
            rec["replicate"] = np.nanmean([corr(V[(i, "truth")], V[(j, "truth")], method)
                                           for i, j in itertools.combinations(P, 2)])
        for a in ARMS:
            if any((p, a) not in V for p in P):
                continue
            rec[f"self_{a}"] = np.nanmean([corr(V[(p, "truth")], V[(p, a)], method) for p in P])
            if len(P) > 1:
                rec[f"cross_{a}"] = np.nanmean([corr(V[(i, "truth")], V[(j, a)], method)
                                                for i, j in itertools.permutations(P, 2)])
        out.append(rec)
    return pd.DataFrame(out)


def per_pathway(rows: pd.DataFrame, nes: pd.DataFrame, method: str) -> pd.DataFrame:
    """Per (line, dose, pathway): r across compounds. Replicate and cross use plates A < B of each multi-plate
    drug; a three-plate drug's pair is enumerated and r averaged over the choices. Self pools every plate."""
    W = nes.pivot_table(index=["row", "side"], columns="term", values="nes")
    out = []
    for (cell, dose), g in rows.groupby(["cell_name", "dose"]):
        multi = {d: list(s.row) for d, s in g[g.n_plates > 1].groupby("drug")}
        choices = list(itertools.product(*[list(itertools.combinations(p, 2)) for p in multi.values()]))
        if len(choices) > 27:
            raise ValueError(f"{cell}/{dose}: {len(choices)} pair assignments; too many three-plate drugs")
        for term in W.columns:
            col = W[term]
            v = lambda r, s: col.get((r, s), np.nan)  # noqa: E731
            rec = dict(cell_name=cell, dose=dose, term=term, method=method, n_multi=len(multi), n_plates_all=len(g))
            if len(multi) >= MIN_POINTS:
                rec["replicate"] = np.nanmean([corr(np.array([v(a, "truth") for a, _ in c]),
                                                    np.array([v(b, "truth") for _, b in c]), method) for c in choices])
                for a_ in ARMS:
                    rec[f"cross_{a_}"] = np.nanmean([0.5 * (corr(np.array([v(a, "truth") for a, _ in c]),
                                                                 np.array([v(b, a_) for _, b in c]), method)
                                                            + corr(np.array([v(b, "truth") for _, b in c]),
                                                                   np.array([v(a, a_) for a, _ in c]), method))
                                                     for c in choices])
                    mr = [r for p in multi.values() for r in p]
                    rec[f"self_multi_{a_}"] = corr(np.array([v(r, "truth") for r in mr]), np.array([v(r, a_) for r in mr]), method)
            for a_ in ARMS:
                rec[f"self_{a_}"] = corr(np.array([v(r, "truth") for r in g.row]), np.array([v(r, a_) for r in g.row]), method)
            out.append(rec)
    return pd.DataFrame(out)


def summarize(cond: pd.DataFrame, path_: pd.DataFrame) -> pd.DataFrame:
    """Means, medians and (per condition) a cluster bootstrap 95 % CI over (line, drug), per dose and overall."""
    from src.stats.bootstrap import cluster_bootstrap_ci
    out = []
    for view, t, unit in (("per_condition", cond, "condition"), ("per_pathway", path_, "line x pathway")):
        cols = [c for c in t.columns if c.startswith(("replicate", "cross_", "self_"))]
        for (method, dose), g in list(t.groupby(["method", "dose"])) + [((m, "all"), t[t.method == m]) for m in METHODS]:
            for c in cols:
                ok = g[np.isfinite(g[c])] if c in g else g.iloc[:0]
                rec = dict(view=view, method=method, dose=dose, quantity=c, unit=unit, n=len(ok),
                           mean=ok[c].mean(), median=ok[c].median())
                if view == "per_condition" and len(ok) and ok[["cell_name", "drug"]].drop_duplicates().shape[0] > 1:
                    _m, lo, hi = cluster_bootstrap_ci(ok[c].to_numpy(), (ok.cell_name + "|" + ok.drug).to_numpy())
                    rec.update(ci_lo=float(lo[0]), ci_hi=float(hi[0]))
                out.append(rec)
    return pd.DataFrame(out)


def stage_metrics(rows: pd.DataFrame, nes: pd.DataFrame) -> None:
    V = {k: g.set_index("term").nes.reindex(list(load_msigdb(HALLMARK))).to_numpy(float)
         for k, g in nes.groupby(["row", "side"])}
    cond = pd.concat([per_condition(rows, V, m) for m in METHODS], ignore_index=True)
    path_ = pd.concat([per_pathway(rows, nes, m) for m in METHODS], ignore_index=True)
    summ = summarize(cond, path_)
    cond.to_csv(OUT / "per_condition.tsv", sep="\t", index=False)
    path_.to_csv(OUT / "per_pathway.tsv", sep="\t", index=False)
    summ.to_csv(OUT / "summary.tsv", sep="\t", index=False)
    pd.set_option("display.width", 220)
    show = summ[(summ.dose == "all")].pivot_table(index=["view", "quantity"], columns="method", values="mean")
    log.info("means over all doses:\n%s", show.round(3).to_string())


# ----------------------------------------------------------------------------
# A run
# ----------------------------------------------------------------------------

def main() -> int:
    global OUT, GENESIS, ADDITIVE_ROOT, LIBRARY      # re-pointed before any worker forks
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", choices=["neighbors", "responses", "gsea", "metrics", "split", "all"], default="all")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", type=Path, default=OUT, help="the stage cache (needs neighbors.parquet, a published input)")
    ap.add_argument("--genesis", type=Path, default=GENESIS, help="the GENESIS K=1 test persist (predictions/ + "
                    "metrics_gene_arrays.parquet)")
    ap.add_argument("--library", type=Path, required=True, help="library_size__tahoe.tsv (tscp per condition)")
    ap.add_argument("--additive-root", type=Path, help="the Additive's fit dir (additive.npz); default: deg_readout's "
                    + ADDITIVE_PANEL + " baseline_root")
    args = ap.parse_args()
    if args.stage not in ("neighbors", "responses") and not HALLMARK.is_file():  # stop before the pseudobulks
        raise FileNotFoundError(f"{HALLMARK}: the MSigDB Hallmark gene sets (collection h.all, MSigDB v2026.1.Hs, JSON) "
                                "are not in the repository; download them to this path (see README)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    from src.eval.readout import deg_readout as S
    OUT, GENESIS, LIBRARY = args.out, args.genesis, args.library
    ADDITIVE_ROOT = args.additive_root or S.resolve_panel(ADDITIVE_PANEL).baseline_root
    OUT.mkdir(parents=True, exist_ok=True)
    rows = plate_rows()
    rows.to_csv(OUT / "plates.tsv", sep="\t", index=False)
    picks = stage_neighbors(rows)
    if args.stage == "neighbors":
        return 0
    stage_responses(rows, picks, args.workers)
    if args.stage == "responses":
        return 0
    nes = stage_gsea(rows, args.workers)
    if args.stage == "gsea":
        return 0
    if args.stage == "split":
        stage_split_responses(rows, args.workers)
        stage_split_gsea(rows, args.workers)
        return 0
    stage_metrics(rows, nes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
