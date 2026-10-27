#!/usr/bin/env python3
"""Figure 4's pathway responses on every held-out test condition measured on two or more plates, each plate alone:
what the cross-batch axis correlations score (scripts/figures/figure4/score/ed_program_summary/axis_correlation.py -> Extended Data program summary).

The conditions are Figure 2a's plate pairs (scripts/figures/figure2/extract/extract.py figure2a_pairs): 530 pairs, 500
conditions = 100 (compound, dose) x the 5 held-out lines, >= 50 cells per plate, none seen in training at another
dose. Afatinib is on three plates at every dose, so its conditions carry three pairs. Per plate:

  truth     the plate's perturbed mean count, and its control pool's mean and detection (n_3072 cache)
  genesis   the mean of its persisted per-plate prediction, snapped as model_means does
  state     the same
  additive  compare_baselines' fit (the bundle's cached sums, in its fit space), emitted onto the
            line's own plate controls at the pair's dose, as compare_baselines emits it at 5 uM
  nearest neighbour, three ways, each a training line's measured cells on the same plate:
  nn_own       Figure 2a's neighbour (v2 table), its log2FC against ITS OWN plate controls, a gene kept only where
               the neighbour's controls detect it too: it copies the response.
  nn_response  the same on the hbss015 neighbour table (Figure 4's copy_closest_15).
  nn_cells     Figure 2a's neighbour against the HELD-OUT line's plate controls: it copies the cells, so the two
               lines' basal difference rides along.
  Each uses the neighbour's whole plate mean (a mean needs no draw) and needs >= 50 neighbour cells on the plate;
  a pair short of one is written without that arm.

Each plate's log2FC is against its own controls; the pathway score is progeny.score on the depth split (minus
truth's measured depth of that plate). Two gene gates, both written:
  replicate  a gene counts only where BOTH plates' controls detect it (>= 0.10): per plate, then intersected, so a
             difference in gene sets is never read as plate noise. The one the correlations use.
  mean       the two plates' detection averaged, Figure 4's own rule. On the replicated 5 uM compounds it must
             equal the bundle's plate_values (truth, GENESIS, State) and compare_baselines' summary_plates__depth
             (Additive, copy_closest_15), or nothing is written.

The measured and generated means (~1,030 plates; ~60 GB of predictions) are cached in <bundle>/data/axis_correlation/
pair_means.npz, the neighbours' (one pass over the hvg cache) in neighbor_means.npz. The Additive is never cached: it
is emitted from the bundle's fit on every run, so the values always carry the fit compare_baselines scored.

    python scripts/figures/figure4/score/ed_program_summary/pair_values.py --bundle <bundle>
      -> <bundle>/data/axis_correlation/{pair_means.npz, neighbor_means.npz, pair_values__depth.parquet}
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
from scripts.figures.figure4.score.main import compare_baselines as CB  # noqa: E402  the Additive fit and its train-row index
from scripts.figures.figure4.score.main import progeny_tables as E  # noqa: E402  the library-size table
from scripts.figures.figure4.score.main import model_means as SM  # noqa: E402  the prediction readout
from scripts.figures.figure4.score.main import truth as T  # noqa: E402  raw-count column stats
import src.eval.readout.deg_readout as R  # noqa: E402
from scripts.figures.figure2.extract.extract import figure2a_pairs  # noqa: E402
from scripts.figures.figure4.score import drug_axis  # noqa: E402
from scripts.figures.figure4.extract.progeny import FLOOR, assert_pad_ungated, safe, score  # noqa: E402
from src.baselines.space import library_namespace  # noqa: E402
from src.eval.arms.baseline import normalize_plate_for_eval  # noqa: E402
from src.eval.cache.eval_npz_reader import EvalNpzReader  # noqa: E402
from src.eval.reference.truth_io import control_library  # noqa: E402
from src.model.inference_output import to_raw_integer  # noqa: E402

AXIS_DIR = Path("data/axis_correlation")          # in the bundle: the caches and the values
MIN_CELLS = 50          # Figure 2a's floor, re-checked on the n_3072 cache; Figure 4's for a neighbour plate
ARMS = ["truth", "genesis", "state", "additive"]
# arm -> (neighbour table: held-out line -> the training line copied, whose controls its log2FC is against)
V2 = REPO / "core_util_data/tahoe/eval_inputs/copy_closest_line__cosine_v2.tsv"   # frozen input
NN = {"nn_response": (CB.NEIGHBORS, "own"), "nn_cells": (V2, "heldout"), "nn_own": (V2, "own")}


# ----------------------------------------------------------------------------
# The pairs: Figure 2a's, each condition's pairs numbered by plate
# ----------------------------------------------------------------------------

def read_pairs() -> pd.DataFrame:
    """(cell = display, cell_name = safe, drug, dose, plate_i < plate_j, pair = 0.. within the condition)."""
    p = figure2a_pairs()[["cell_name", "drug", "dose", "plate_i", "plate_j"]].astype({"plate_i": int, "plate_j": int})
    if (p.plate_i >= p.plate_j).any() or p.duplicated().any():
        raise SystemExit("Figure 2a's pairs: a pair is unordered or repeated")
    p = p.rename(columns={"cell_name": "cell"}).assign(cell_name=lambda x: x.cell.map(safe))
    p = p.sort_values(["cell_name", "drug", "dose", "plate_i", "plate_j"], kind="stable")
    return p.assign(pair=p.groupby(["cell_name", "drug", "dose"]).cumcount()).reset_index(drop=True)


def plates_of(g: pd.DataFrame) -> list[tuple[str, float, int]]:
    return sorted({(r.drug, r.dose, p) for r in g.itertuples() for p in (r.plate_i, r.plate_j)})


# ----------------------------------------------------------------------------
# Means: truth, GENESIS and State per plate (one worker per line), then the Additive
# ----------------------------------------------------------------------------

def line_means(cell: str, plates: list[tuple[str, float, int]], genesis: Path) -> dict:
    """{(kind, drug, dose, plate): vector} for one line (display spelling); kind in ctrl_mean / ctrl_det
    (drug '', dose 0) and the measured or generated arms, plus ('n', ...) the measured cell count."""
    torch.set_num_threads(SM.THREADS)
    G, out = len(SM.PANEL.genes), {}
    for drug, dose, p in plates:
        if ("ctrl_mean", "", 0.0, p) not in out:
            c = R.load_control(SM.PANEL.cache, cell_name=cell, plate=f"plate{p}", n_genes=G)
            out[("ctrl_mean", "", 0.0, p)], out[("ctrl_det", "", 0.0, p)] = T.col_stats(c)
        x = R.load_perturbed(SM.PANEL.cache, cell_name=cell, drug=drug, dose=dose, plate=f"plate{p}", n_genes=G)
        if x.shape[0] < MIN_CELLS:
            raise SystemExit(f"{cell} / {drug} / {dose} / plate{p}: {x.shape[0]} cells in the n_3072 cache (< {MIN_CELLS})")
        out[("truth", drug, dose, p)], out[("n", drug, dose, p)] = T.col_stats(x)[0], np.array([x.shape[0]], float)
        for arm in SM.ARMS:
            root, fmt = (genesis / "predictions", SM.PANEL.arms[arm][1]) if arm == "genesis" else SM.PANEL.arms[arm]
            y = R.load_pred(R.pred_path(root, fmt, drug, cell, dose, p), SM.PANEL.genes)
            if y.shape != x.shape:
                raise SystemExit(f"{cell} / {drug} / {dose} / plate{p} / {arm}: prediction {y.shape}, measured {x.shape}")
            out[(arm, drug, dose, p)] = SM.mean_count(y)
    print(f"  {cell}: {len(plates)} plates", flush=True)
    return out


def additive_fit(bundle: Path, conditions: set[tuple[str, str, float]]):
    """compare_baselines' Additive, from the bundle's cached sums; refuses if any test condition trained it."""
    man = json.loads((CB.CACHE / "manifest.json").read_text())
    nb = pd.read_csv(CB.NEIGHBORS, sep="\t")
    drugs = set(drug_axis.conditions(drug_axis.table(bundle / "data/drug_axis/drug_axis.tsv")))
    idx = CB.index_rows(man, np.sort(np.load(CB.train_idx())), set(nb.neighbor), drugs)   # compare_baselines' exact index
    lab, n = idx["add"][2], np.bincount(idx["add"][1])
    partial = bundle / CB.PARTIAL
    if (shape := np.load(partial / "add_sum.npy", mmap_mode="r").shape) != (len(lab["cell"]), CB.G):
        raise SystemExit(f"add_sum.npy is {shape}, this index has {len(lab['cell'])} groups: not the cached fit")
    hit = [(c, d, z) for c, d, z in zip(lab["cell"], lab["drug"], lab["dose"]) if (c, d, round(float(z), 6)) in conditions]
    if hit:
        raise SystemExit(f"additive trained on {len(hit)} test conditions, e.g. {hit[:3]}")
    return CB.fit_additive(lab, n, partial)


def additive_means(model, pairs: pd.DataFrame, n_cells: dict) -> dict:
    """{(cell, drug, dose, plate): mean count}: emitted onto the line's own plate controls in the fit's space (their
    libraries from the eval cache's L_full sidecar), snapped in that space, then averaged."""
    space = R.prediction_space(model)
    libs = None if space == "raw" else EvalNpzReader(library_namespace(Path(SM.PANEL.cache)))
    out = {}
    for cell, g in pairs.groupby("cell"):
        plates = plates_of(g)
        ctrl = {p: R.load_control(SM.PANEL.cache, cell_name=cell, plate=f"plate{p}", n_genes=CB.G)
                for p in {p for *_, p in plates}}
        lib = {p: None if libs is None else control_library(libs, {"cell_name": cell, "plate": f"plate{p}"}, c)
               for p, c in ctrl.items()}
        for drug, dose, p in plates:
            if (drug, round(float(dose), 6)) not in model.tables["pert"]:
                raise SystemExit(f"additive has no effect for {drug} at {dose} uM: it was not in training")
            cells = R.baseline_pred(model, "additive", drug=drug, cell=cell, dose=dose, plate=p, ctrl_raw=ctrl[p],
                                    n_cells=int(n_cells[(cell, drug, dose, p)]), ctrl_lib=lib[p])
            out[(cell, drug, dose, p)] = to_raw_integer(
                torch.as_tensor(cells, dtype=torch.float32), space).to(torch.float64).mean(0).numpy()
        print(f"  additive {cell}: {len(plates)} plates", flush=True)
    return out


def load(path: Path) -> dict:
    print(f"  reading cached {path} (delete it to re-read the cells)", flush=True)
    z = np.load(path)
    v = z["v"]   # each z["v"] decompresses the whole array again
    return {(k, c, d, float(z_), int(p)): v[i]
            for i, (k, c, d, z_, p) in enumerate(zip(z["kind"], z["cell"], z["drug"], z["dose"], z["plate"]))}


def save(path: Path, out: dict) -> None:
    """One (G,) row per key; a count is padded to G and read back as [0]."""
    if not all(np.isfinite(v).all() for v in out.values()):
        raise SystemExit("a non-finite mean")
    keys = list(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, v=np.array([np.resize(out[k], CB.G) for k in keys]),
                        **{f: np.asarray([k[i] for k in keys]) for i, f in enumerate(["kind", "cell", "drug", "dose", "plate"])})
    print(f"  wrote {path}: {len(keys)} vectors", flush=True)


def means(bundle: Path, pairs: pd.DataFrame, genesis: Path) -> dict:
    """{(kind, cell display, drug, dose, plate): vector}: the measured and generated means from pair_means.npz
    (computed and cached when it is missing), the Additive emitted from the bundle's fit."""
    cache = bundle / AXIS_DIR / "pair_means.npz"
    if cache.exists():
        out = {k: v for k, v in load(cache).items() if k[0] != "additive"}   # an older cache may hold one: not used
    else:
        per_line = {c: plates_of(g) for c, g in pairs.groupby("cell")}
        with ProcessPoolExecutor(len(per_line), mp_context=get_context("spawn")) as ex:
            parts = dict(zip(per_line, ex.map(line_means, per_line, per_line.values(), [genesis] * len(per_line))))
        out = {(k, c, d, z, p): v for c, part in parts.items() for (k, d, z, p), v in part.items()}
        save(cache, out)
    n_cells = {(c, d, z, p): v[0] for (k, c, d, z, p), v in out.items() if k == "n"}
    model = additive_fit(bundle, {(r.cell, r.drug, round(float(r.dose), 6)) for r in pairs.itertuples()})
    return out | {("additive", *k): v for k, v in additive_means(model, pairs, n_cells).items()}


def neighbors(pairs: pd.DataFrame) -> dict[str, dict[str, str]]:
    """{arm: {held-out line: neighbour}}, display spelling. A neighbour that is itself held out is a leak."""
    nb = {a: pd.read_csv(f, sep="\t").set_index("cell_name").neighbor.to_dict() for a, (f, _) in NN.items()}
    held = set(pairs.cell)
    for a, t in nb.items():
        if set(t) != held or held & set(t.values()):
            raise SystemExit(f"{a}: table lines {sorted(t)} vs held out {sorted(held)}, neighbours {sorted(t.values())}")
    return nb


def row_sums(csr: tuple[np.ndarray, np.ndarray, np.ndarray], rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-gene count sum and nonzero count over ``rows`` of one shard's (data, indices, indptr) mmaps.
    Reads only those rows: wrapping the shard as scipy CSR first scans its billion-entry index."""
    data, indices, ptr = csr
    lo, n = ptr[rows].astype(np.int64), (ptr[rows + 1] - ptr[rows]).astype(np.int64)
    pos = np.repeat(lo - np.cumsum(n) + n, n) + np.arange(n.sum())   # every stored entry of every row
    col, val = indices[pos].astype(np.int64), data[pos].astype(np.float64)
    return np.bincount(col, val, CB.G), np.bincount(col[val > 0], minlength=CB.G).astype(np.float64)


def shard_neighbors(subdir: str, want: set[tuple]) -> dict:
    """One shard (= one plate): the neighbours' control pools and the wanted (line, drug, dose) groups."""
    shard = CB.CACHE / subdir
    csr = tuple(np.load(shard / f"{f}.npy", mmap_mode="r") for f in ("data", "indices", "indptr"))
    if len(plates := CB._cats(shard, "plate")) != 1:
        raise SystemExit(f"{shard.name}: expected one plate, got {plates}")
    p = int(normalize_plate_for_eval(plates[0]))
    cid = {n: i for i, n in enumerate(CB._cats(shard, "cell_name"))}
    did = {n: i for i, n in enumerate(CB._cats(shard, "drug"))}
    # drug from the drug code, dose from the concentration code: the two category lists are not aligned
    zval = np.array([round(float(ast.literal_eval(z)[0][1]), 6) for z in CB._cats(shard, "drugname_drugconc")])
    c, d, z = CB._codes(shard, "cell_name"), CB._codes(shard, "drug"), CB._codes(shard, "drugname_drugconc")
    out = {}
    for ln in sorted({k[0] for k in want}):
        rows = np.flatnonzero(c == cid[ln]) if ln in cid else np.array([], int)
        groups = {("nb_ctrl", ln, "", 0.0, p): rows[d[rows] == did[CB.CONTROL]]}
        groups |= {("nb", ln, drug, dose, p): rows[(d[rows] == did.get(drug, -1)) & (zval[z[rows]] == dose)]
                   for l_, drug, dose, pp in want if l_ == ln and pp == p}
        for key, r in groups.items():
            if not len(r):
                continue
            total, nnz = row_sums(csr, r)
            out[key] = total / len(r)
            out[("nb_det" if key[0] == "nb_ctrl" else "nb_n", *key[1:])] = (
                nnz / len(r) if key[0] == "nb_ctrl" else np.array([len(r)], float))
    print(f"  {subdir} (plate {p}): {len(out)} neighbour vectors", flush=True)
    return out


def neighbor_means(bundle: Path, pairs: pd.DataFrame, nb: dict) -> dict:
    """{(kind, neighbour, drug, dose, plate): vector}: nb = perturbed mean count, nb_n = cells ([0]),
    nb_ctrl / nb_det = control mean and detection (drug '', dose 0). Every cell on the plate, none drawn."""
    cache = bundle / AXIS_DIR / "neighbor_means.npz"
    if cache.exists():
        return load(cache)
    want = {(t[r.cell], r.drug, round(float(r.dose), 6), int(p))
            for t in nb.values() for r in pairs.itertuples() for p in (r.plate_i, r.plate_j)}
    man = json.loads((CB.CACHE / "manifest.json").read_text())
    if (CB.CACHE / "hvg_gene_names.txt").read_text().split() != list(SM.PANEL.genes) or man["n_hvg"] != CB.G:
        raise SystemExit("the hvg cache gene axis is not the panel's: positional means would mislabel genes")
    if not man["transform"].startswith("raw_counts"):
        raise SystemExit(f"the hvg cache holds {man['transform']}, not raw counts")
    subdirs = [s["cache_subdir"] for s in man["shards"]]
    # random mmap reads over NFS are latency-bound: one process per shard
    with ProcessPoolExecutor(len(subdirs), mp_context=get_context("spawn")) as ex:
        out = {k: v for part in ex.map(shard_neighbors, subdirs, [want] * len(subdirs)) for k, v in part.items()}
    if miss := sorted({(ln, p) for ln, *_, p in want} - {(k[1], k[4]) for k in out if k[0] == "nb_ctrl"}):
        raise SystemExit(f"a neighbour has no controls on a plate it is copied from: {miss}")
    save(cache, out)
    return out


# ----------------------------------------------------------------------------
# Pathway values: each plate's log2FC against its own controls, under each gate
# ----------------------------------------------------------------------------

# Over 40 lines: one loop per gate, kept whole so every arm's log2FC, gate and depth shift are built side by side.
def values(bundle: Path, pairs: pd.DataFrame, m: dict, nbm: dict, nb: dict, sets: pd.DataFrame) -> pd.DataFrame:
    lib = pd.read_csv(bundle / E.LIBRARY, sep="\t").assign(dose=lambda x: x.dose.round(6))
    tscp = lib.set_index(["cell_name", "drug", "dose", "plate"])
    out = []
    for gate in ("replicate", "mean"):
        keys, lfc, ok_rows, shift, short = [], [], [], [], dict.fromkeys(NN, 0)
        both = lambda det: ((det[0] >= FLOOR) & (det[1] >= FLOOR) if gate == "replicate"  # noqa: E731
                            else (det[0] + det[1]) / 2 >= FLOOR)
        for r in pairs.itertuples():
            ps, dose = (r.plate_i, r.plate_j), round(float(r.dose), 6)
            ctrl = [m[("ctrl_mean", r.cell, "", 0.0, p)] for p in ps]
            ok = both([m[("ctrl_det", r.cell, "", 0.0, p)] for p in ps])
            # (arm, perturbed mean per plate, the controls its log2FC is against, its gene gate)
            arms = [(a, [m[(a, r.cell, r.drug, float(r.dose), p)] for p in ps], ctrl, ok) for a in ARMS]
            for a, t in nb.items():
                ln = t[r.cell]
                if any(nbm.get(("nb_n", ln, r.drug, dose, p), [0])[0] < MIN_CELLS for p in ps):
                    short[a] += 1
                    continue
                pert = [nbm[("nb", ln, r.drug, dose, p)] for p in ps]
                if NN[a][1] == "heldout":
                    arms.append((a, pert, ctrl, ok))
                else:
                    own = [nbm[("nb_ctrl", ln, "", 0.0, p)] for p in ps]
                    arms.append((a, pert, own, ok & both([nbm[("nb_det", ln, "", 0.0, p)] for p in ps])))
            for slot, p in enumerate(ps):
                t = tscp.loc[(r.cell, r.drug, dose, f"plate{p}")]   # KeyError: no library size
                for arm, pert, cs, g in arms:
                    keys.append((r.cell_name, r.drug, float(r.dose), r.pair, arm, slot))
                    lfc.append(np.where(g, np.log2((pert[slot] + 1e-9) / (cs[slot] + 1e-9)), 0.0))
                    ok_rows.append(g)
                    shift.append(np.log2(t.mean_tscp / t.ctrl_mean_tscp))   # truth's depth, given to every arm
        if gate == "replicate":
            print(f"  pairs without a neighbour plate of >= {MIN_CELLS} cells (arm left out there): {short}", flush=True)
        gate_m = np.array(ok_rows)
        assert_pad_ungated(gate_m)
        k = pd.DataFrame(keys, columns=["cell_name", "drug", "dose", "pair", "arm", "slot"])
        v = score(k, np.array(lfc), gate_m, sets, SM.PANEL.genes, centre=False, shift=np.array(shift))
        out.append(v.drop(columns="genes").assign(gate=gate))
    return pd.concat(out, ignore_index=True)


def check_published(bundle: Path, v: pd.DataFrame) -> None:
    """The mean gate on the replicated 5 uM compounds IS the bundle's per-plate values, every arm but Figure 2a's
    neighbour (no published pathway values); nn_response is copy_closest_15."""
    ax = drug_axis.table(bundle / "data/drug_axis/drug_axis.tsv")
    rep = set(ax.drug[ax.replicated])
    got = v[(v.gate == "mean") & np.isclose(v.dose, 5.0) & v.drug.isin(rep) & ~v.arm.isin(["nn_cells", "nn_own"])]
    got = got.assign(arm=got.arm.replace({"nn_response": "copy_closest_15"}))
    pub = pd.concat([pd.read_parquet(bundle / "data/tahoe_progeny_depth/plate_values.parquet"),
                     pd.read_parquet(bundle / CB.PARTIAL / "summary_plates__depth.parquet")])
    key = ["programme", "cell_name", "drug", "arm", "slot"]
    mm = pub[key + ["v"]].merge(got[key + ["v"]], on=key, how="outer", indicator=True)
    worst = float(np.nanmax(np.abs(mm.v_x - mm.v_y)))
    if (mm._merge != "both").any() or not np.allclose(mm.v_x, mm.v_y, rtol=1e-9, atol=1e-12):
        raise SystemExit(f"mean-gate values are not the published ones: {int((mm._merge != 'both').sum())} "
                         f"unmatched rows, max |diff| {worst:.3g}")
    print(f"  published per-plate values reproduced: {len(mm)} ({mm.arm.nunique()} arms, max |diff| {worst:.1e})",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, required=True,
                    help="reads BUNDLE/data and its compare_baselines fit, writes BUNDLE/" + str(AXIS_DIR))
    ap.add_argument("--genesis", type=Path, default=SM.PANEL.arms["genesis"][0].parent,
                    help="the GENESIS K=1 test persist (predictions/ + metrics_summary.json); default: the progeny3072 panel's")
    args = ap.parse_args()
    bundle, genesis = args.bundle.resolve(), args.genesis.resolve()
    SM.assert_genesis_axis(genesis / "metrics_summary.json", SM.PANEL.genes)
    out = bundle / AXIS_DIR / "pair_values__depth.parquet"
    if out.exists():
        raise SystemExit(f"{out} exists; delete it to rescore")
    pairs = read_pairs()
    print(f"{len(pairs)} pairs, {pairs.groupby(['cell_name', 'drug', 'dose']).ngroups} conditions", flush=True)
    m, nb = means(bundle, pairs, genesis), neighbors(pairs)
    nbm = neighbor_means(bundle, pairs, nb)
    sets = pd.read_csv(bundle / "data/tahoe_progeny_depth/programmes.tsv", sep="\t")
    v = values(bundle, pairs, m, nbm, nb, sets)
    check_published(bundle, v)
    v.to_parquet(out, index=False)
    print(f"wrote {out}: {len(v)} rows", flush=True)


if __name__ == "__main__":
    main()
