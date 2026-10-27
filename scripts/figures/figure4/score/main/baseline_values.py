#!/usr/bin/env python3
"""ED_celllines_per_model's two baselines as values, no drawing: Additive and Fig. 2a's nearest neighbor (NN),
depth-adjusted PROGENy on the drug axis, into <bundle>/data/upstream/compare_baselines/.

Additive is the baseline compare_baselines.py scored and did not draw, refit from its cached cell sums and
refused unless the per-pathway mean |error| matches that script's depth sidecar.

NN is Fig. 2a's nearest neighbor (nn_own, the ED program summary's): the v2 neighbor line's measured
5 µM cells on the drug's own plates, against the NEIGHBOR's own plate controls, a gene kept only where both
lines' controls detect it: the neighbor's response, without the two lines' basal difference. --nn-controls heldout
scores the same cells against the held-out line's controls (nn_cells). Each plate of a replicated compound must
equal the axis correlations' single-plate value of the same arm (pair_values.py's pair_values__depth.parquet).

    python -m scripts.figures.figure4.score.main.baseline_values additive --bundle <bundle>   # -> depth_module_values.parquet
    python -m scripts.figures.figure4.score.main.baseline_values nn --bundle <bundle>         # -> depth_module_values__nn_own.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
from scripts.figures.figure4.extract.progeny import MIX, safe  # noqa: E402

PARTIAL = Path("data/upstream/compare_baselines")
SIDECAR = "summary_metrics__depth.tsv"
VALUES = "depth_module_values.parquet"          # mixture pathway scores, written once beside the sums
NN_READS = "nn_v2_neighbor_reads.npz"           # the neighbors' plate means: both control choices score from it
NN_TABLE = REPO / "core_util_data/tahoe/eval_inputs/copy_closest_line__cosine_v2.tsv"   # Fig. 2a's neighbors
AXIS_CORR = Path("data/axis_correlation/pair_values__depth.parquet")   # in the bundle: pair_values.py's values
PAIRS = REPO / "core_util_data/tahoe/eval_inputs/benchmark_pairs.csv"   # --pairs: Fig. 2a's 530 plate pairs
# The NN's controls -> its arm id (the axis correlations'): own is the published figure
NN_ARM = {"own": "nn_own", "heldout": "nn_cells"}


# ----------------------------------------------------------------------------
# The two baselines: the same depth score as the heatmap, checked against the sidecar
# ----------------------------------------------------------------------------

def mae_like_sidecar(pred: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Per (pathway, line, arm): how many drugs were scored, and the mean |error| over those drugs.

    A drug the arm did not score is absent, not a zero. A scored drug with no truth, or a NaN score,
    would change that mean without a trace, so both raise."""
    if pred.v.isna().any():
        raise RuntimeError(f"{int(pred.v.isna().sum())} scored pathway values are NaN")
    ref = truth.loc[truth.arm.eq("truth"), ["programme", "cell_name", "drug", "v"]].rename(columns={"v": "truth"})
    m = pred.merge(ref, on=["programme", "cell_name", "drug"], how="left")
    if m.truth.isna().any():
        raise RuntimeError(f"{int(m.truth.isna().sum())} scored rows have no measured pathway value")
    return (m.assign(mae=(m.v - m.truth).abs())
            .groupby(["programme", "cell_name", "arm"], sort=False)
            .agg(n=("mae", "size"), mae=("mae", "mean")).reset_index())


def assert_sidecar(pred: pd.DataFrame, truth: pd.DataFrame, published: pd.DataFrame, atol: float = 1e-6) -> None:
    """Refuse unless ``pred`` reproduces ``published`` (the depth MAE sidecar) row for row."""
    got = mae_like_sidecar(pred, truth).set_index(["programme", "cell_name", "arm"])
    pub = published.set_index(["programme", "cell_name", "arm"])
    if not got.index.symmetric_difference(pub.index).empty:
        raise RuntimeError(f"MAE rows differ from the sidecar: "
                           f"missing {list(pub.index.difference(got.index)[:3])}, "
                           f"extra {list(got.index.difference(pub.index)[:3])}")
    got = got.reindex(pub.index)
    if not np.array_equal(got.n.to_numpy(), pub.n.to_numpy()):
        raise RuntimeError("a pathway was scored on a different number of drugs than the sidecar")
    if not np.allclose(got.mae.to_numpy(), pub.mae.to_numpy(), rtol=0, atol=atol):
        i = int(np.abs(got.mae.to_numpy() - pub.mae.to_numpy()).argmax())
        raise RuntimeError(f"MAE differs from the sidecar at {pub.index[i]}: "
                           f"{got.mae.iloc[i]:.6g} vs {pub.mae.iloc[i]:.6g}")


def score_depth_baselines(bundle: Path) -> pd.DataFrame:
    """Additive and the nearest-neighbor line, depth-adjusted mixture, on the drug axis.

    The cell sums already exist (compare_baselines, hours). This refits from them, scores only the
    depth split, and does not rewrite the sidecar, the plate table, or comparisons.tsv."""
    from scripts.figures.figure4.score.main import compare_baselines as B
    from scripts.figures.figure4.score import drug_axis
    from scripts.figures.figure4.score.main import truth as T
    from scripts.figures.figure4.extract.progeny import assert_pad_ungated, depth, gated_pairs, safe, score  # noqa: E402

    partial = bundle / PARTIAL
    man = json.loads((B.CACHE / "manifest.json").read_text())
    ax = drug_axis.table(bundle / "data/drug_axis/drug_axis.tsv")
    conds, axis, replicated = drug_axis.conditions(ax), set(ax.drug[ax.on_axis]), set(ax.drug[ax.replicated])
    _, test = gated_pairs(bundle / "data/upstream")
    display = dict(zip([safe(c) for c in test], test))
    copy_of = {safe(r.cell_name): r.neighbor for r in pd.read_csv(B.NEIGHBORS, sep="\t").itertuples()}
    if set(copy_of) != set(display):
        raise RuntimeError(f"neighbor lines {sorted(copy_of)} are not the held-out lines {sorted(display)}")
    truth = T.load_truth(test, conds)
    sets = pd.read_csv(bundle / "data/tahoe_progeny_raw/programmes.tsv", sep="\t")
    B.check_truth(truth, conds, sets, bundle)
    train = np.sort(np.load(B.train_idx()))
    idx = B.index_rows(man, train, set(copy_of.values()), set(conds))
    B.assert_no_leak(idx["add"][2], set(display.values()), axis | replicated)
    rows, ga, gp, gc = B.merged_rows(idx)
    B.accumulate(rows, np.column_stack([ga, gp, gc]), {
        "add_sum": (len(idx["add"][2]["cell"]), B.G),
        "pert_sum": (len(idx["pert"][2]["cell"]), B.G),
        "ctrl_sum": (len(idx["ctrl"][2]["cell"]), B.G),
        "ctrl_nnz": (len(idx["ctrl"][2]["cell"]), B.G)}, partial)
    model = B.fit_additive(idx["add"][2], np.bincount(idx["add"][1]), partial)
    scored = {d: conds[d] for d in axis | replicated}
    add = B.additive_means(model, truth, scored, display)
    pert, ctrl, det = B.neighbor_stats(idx["pert"][2], np.bincount(idx["pert"][1]),
                                       idx["ctrl"][2], np.bincount(idx["ctrl"][1]), partial)
    keys, lfc, gate = B.responses(truth, scored, add, pert, ctrl, det, copy_of, axis, replicated)
    assert_pad_ungated(gate)
    dep = depth(keys, scored, bundle / "context_analysis/data/upstream/library_size__tahoe.tsv")
    got = score(keys, lfc, gate, sets, truth.genes, False, dep)
    mix = got[got.slot == MIX].assign(arm=lambda x: x.arm.replace({"nearest_neighbor": "copy_closest_15"}))
    return mix[["programme", "cell_name", "drug", "arm", "v", "n_genes"]].reset_index(drop=True)


def additive(bundle: Path) -> pd.DataFrame:
    """The cached depth scores, or a fresh scoring. Either way the sidecar must agree. Only Additive is
    drawn: the cache also holds compare_baselines' copy_closest_15, checked with it and then dropped."""
    path = bundle / PARTIAL / VALUES
    published = pd.read_csv(bundle / PARTIAL / SIDECAR, sep="\t")
    truth = pd.read_parquet(bundle / "data/tahoe_progeny_depth/module_values.parquet")
    if path.exists():
        pred = pd.read_parquet(path)
        print(f"  baselines: {path.relative_to(bundle)} ({len(pred)} rows)", flush=True)
    else:
        print("  baselines: scoring Additive and NN from the cached cell sums", flush=True)
        pred = score_depth_baselines(bundle)
    try:
        assert_sidecar(pred, truth, published)
    except RuntimeError as e:
        raise RuntimeError(f"{path}: {e}") from e
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pred.to_parquet(path, index=False)
        print(f"  wrote {path.relative_to(bundle)}", flush=True)
    print(f"  ✓ MAE matches {SIDECAR}", flush=True)
    return pred[pred.arm == "additive"]


# ----------------------------------------------------------------------------
# Fig. 2a's nearest neighbor: its cells, the held-out line's controls
# ----------------------------------------------------------------------------

def neighbor_plates(want: set[tuple], genes: list[str], path: Path, bundle: Path) -> dict:
    """{(kind, line, drug, dose, plate): vector} over every cell of the neighbors' plates, none drawn:
    nb = perturbed mean count, nb_n = cells ([0]), nb_ctrl / nb_det = control mean and detection.
    pair_values' reader (one process per shard), cached at ``path``; refused unless the plates
    it already read for Fig. 2a's pairs come back identical."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    from scripts.figures.figure4.score.ed_program_summary import pair_values as ET
    nb_means = bundle / ET.AXIS_DIR / "neighbor_means.npz"
    if (ET.CB.CACHE / "hvg_gene_names.txt").read_text().split() != list(genes):
        raise RuntimeError("the hvg cache gene axis is not truth's: positional means would mislabel genes")
    if path.exists():
        out = ET.load(path)   # a plate with no cells is absent either way, so the lines are what can go stale
        if (got := {k[1] for k in out if k[0] == "nb_ctrl"}) != (lines := {w[0] for w in want}):
            raise RuntimeError(f"{path.name} holds lines {sorted(got)}, the table wants {sorted(lines)}: delete it")
    else:
        subdirs = [s["cache_subdir"] for s in json.loads((ET.CB.CACHE / "manifest.json").read_text())["shards"]]
        with ProcessPoolExecutor(len(subdirs), mp_context=get_context("spawn")) as ex:
            out = {k: v for part in ex.map(ET.shard_neighbors, subdirs, [want] * len(subdirs)) for k, v in part.items()}
        ET.save(path, out)
    cached = ET.load(nb_means)
    shared = [k for k in out if k in cached]
    if not shared or any(not np.array_equal(np.resize(out[k], len(genes)), cached[k]) for k in shared):
        raise RuntimeError(f"{len(shared)} re-read neighbor vectors differ from {nb_means.name} (or none overlap)")
    print(f"  neighbors: {sum(k[0] == 'nb' for k in out)} plates read, {len(shared)} vectors equal the cached ones",
          flush=True)
    return out


def score_nn(bundle: Path, controls: str) -> pd.DataFrame:
    """Depth-adjusted pathway scores of the neighbor's cells, the mixture on the drug axis, against the
    ``controls`` of NN_ARM. Each plate of a replicated compound is also scored alone and checked against
    the axis correlations' same arm, then dropped. own differs from heldout only in the controls and the
    gate (neighbor_gate, compare_baselines').

    A (line, drug) whose neighbor has < 50 cells on any of the drug's plates is not scored: grey."""
    from scripts.figures.figure4.score.main import compare_baselines as B
    from scripts.figures.figure4.score import drug_axis
    from scripts.figures.figure4.score.main import truth as T
    from scripts.figures.figure4.extract.progeny import DOSE, FLOOR, assert_pad_ungated, depth, gated_pairs, score  # noqa: E402

    ax = drug_axis.table(bundle / "data/drug_axis/drug_axis.tsv")
    conds, axis, replicated = drug_axis.conditions(ax), set(ax.drug[ax.on_axis]), set(ax.drug[ax.replicated])
    _, test = gated_pairs(bundle / "data/upstream")
    copy_of = {safe(r.cell_name): r.neighbor for r in pd.read_csv(NN_TABLE, sep="\t").itertuples()}
    if set(copy_of) != {safe(c) for c in test} or set(copy_of.values()) & set(test):
        raise RuntimeError(f"{NN_TABLE.name}: lines {sorted(copy_of)}, neighbors {sorted(copy_of.values())}")
    truth = T.load_truth(test, conds)
    sets = pd.read_csv(bundle / "data/tahoe_progeny_raw/programmes.tsv", sep="\t")
    B.check_truth(truth, conds, sets, bundle)
    dose, drugs = round(float(DOSE), 6), sorted(axis | replicated)
    nbm = neighbor_plates({(nb, d, dose, int(p)) for nb in copy_of.values() for d in drugs for p in conds[d]},
                          truth.genes, bundle / PARTIAL / NN_READS, bundle)
    keys, lfc, gate, short = [], [], [], []
    for c, nb in sorted(copy_of.items()):
        for d in drugs:
            ps = conds[d]
            if min(nbm.get(("nb_n", nb, d, dose, int(p)), [0])[0] for p in ps) < B.MIN_MEASURED:
                short.append((c, d))
                continue
            ms = [nbm[("nb", nb, d, dose, int(p))] for p in ps]
            held = [truth.cmean[(c, f"plate{p}")] for p in ps]   # the held-out line's controls, as every model
            ok = np.mean([truth.D[(c, f"plate{p}")] for p in ps], axis=0) >= FLOOR
            own = [nbm[("nb_ctrl", nb, "", 0.0, int(p))] for p in ps]   # KeyError: the neighbor has no controls there
            ok_own = B.neighbor_gate(ok, np.stack([nbm[("nb_det", nb, "", 0.0, int(p))] for p in ps]))
            cs, g = (own, ok_own) if controls == "own" else (held, ok)
            for s in [*(range(len(ps)) if d in replicated else []), *([MIX] if d in axis else [])]:
                keys.append((c, d, NN_ARM[controls], s))
                lfc.append(np.where(g, B.plate_mixture(ms, cs) if s == MIX else B.plate_mixture([ms[s]], [cs[s]]), 0.0))
                gate.append(g)
    print(f"  NN not scored (< {B.MIN_MEASURED} neighbor cells on a plate): {len(short)} (line, drug)s, "
          f"e.g. {short[:3]}", flush=True)
    keys = pd.DataFrame(keys, columns=["cell_name", "drug", "arm", "slot"])
    gate = np.array(gate)
    assert_pad_ungated(gate)
    dep = depth(keys, conds, bundle / "context_analysis/data/upstream/library_size__tahoe.tsv")
    got = score(keys, np.array(lfc), gate, sets, truth.genes, False, dep)
    assert_matches_axis_corr(got[got.slot != MIX], conds, NN_ARM[controls], bundle)
    mix = got[got.slot == MIX]
    return mix[["programme", "cell_name", "drug", "arm", "v", "n_genes"]].reset_index(drop=True)


def assert_matches_axis_corr(plates: pd.DataFrame, conds: dict, arm: str, bundle: Path, atol: float = 1e-6) -> None:
    """Each plate of a replicated compound is a single-plate ``arm`` value of the axis correlations
    (pair_values, 'mean' gate = this mixture's gate): the same neighbor, cells, controls, gate
    and depth, or the NN drawn here is not the ED program summary's. Compared where Fig. 2a's pair is
    exactly the compound's plates, slot for slot (pair_values numbers a condition's pairs this way)."""
    p = pd.read_csv(PAIRS).sort_values(["cell_name", "drug", "dose", "plate_i", "plate_j"], kind="stable")
    p = p.assign(pair=p.groupby(["cell_name", "drug", "dose"]).cumcount(), cell_name=p.cell_name.map(safe))
    p = p[np.isclose(p.dose, 5.0) & [(str(i), str(j)) == tuple(conds.get(d, ())) for d, i, j in
                                     zip(p.drug, p.plate_i, p.plate_j)]]
    ref = pd.read_parquet(bundle / AXIS_CORR).query("arm == @arm and gate == 'mean'")
    ref = ref[np.isclose(ref.dose, 5.0)].merge(p[["cell_name", "drug", "pair"]], on=["cell_name", "drug", "pair"])
    k = ["cell_name", "drug", "slot", "programme"]
    m = plates.assign(slot=plates.slot.astype(int)).merge(ref.assign(slot=ref.slot.astype(int)), on=k,
                                                           suffixes=("", "_ref"))
    if m.empty:
        raise RuntimeError("no replicated NN plate overlaps the axis correlations: nothing was checked")
    if not (np.allclose(m.v, m.v_ref, rtol=0, atol=atol) and (m.n_genes == m.n_genes_ref).all()):
        i = int(np.abs(m.v - m.v_ref).argmax())
        raise RuntimeError(f"NN differs from the axis correlations' {arm} at {m.loc[i, k].to_dict()}: "
                           f"{m.v[i]:.6g} vs {m.v_ref[i]:.6g}")
    print(f"  ✓ NN equals the axis correlations' {arm} on {len(m)} single-plate pathway values", flush=True)


def nn(bundle: Path, controls: str) -> pd.DataFrame:
    """The cached NN scores, or a fresh scoring (reads the neighbors' plates once: minutes)."""
    path = bundle / PARTIAL / f"depth_module_values__{NN_ARM[controls]}.parquet"
    if path.exists():
        print(f"  NN: {path.relative_to(bundle)}", flush=True)
        return pd.read_parquet(path)
    print(f"  NN: scoring Fig. 2a's neighbor from its measured cells, {controls} controls", flush=True)
    got = score_nn(bundle, controls)
    if got.v.isna().any():
        raise RuntimeError(f"{int(got.v.isna().sum())} NN pathway values are NaN")
    got.to_parquet(path, index=False)
    print(f"  wrote {path.relative_to(bundle)}", flush=True)
    return got




if __name__ == "__main__":   # the scoring spawns workers, which re-import this file
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("values", choices=("additive", "nn"))
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, default=PAIRS, help="Fig. 2a's plate pairs (the NN check's slots)")
    ap.add_argument("--nn-controls", choices=tuple(NN_ARM), default="own",
                    help="NN's log2FC against the neighbor's own controls (published) or the held-out line's")
    a = ap.parse_args()
    PAIRS = a.pairs   # read in this process only (assert_matches_axis_corr), never by a spawned worker
    additive(a.bundle.resolve()) if a.values == "additive" else nn(a.bundle.resolve(), a.nn_controls)
