#!/usr/bin/env python3
"""State, Additive and the nearest-neighbor line, scored like Figure 4 and not drawn: the depth summary's extra arms
and tables/comparisons.tsv. Fits on 4.6M cells of the hvg cache (hours): never run automatically.

    python scripts/figures/figure4/score/main/compare_baselines.py --bundle <bundle>
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
from scripts.figures.figure4.score.main import truth as T  # noqa: E402
import src.eval.readout.deg_readout as R  # noqa: E402
from src.data.cache.raw_hvg_cache import RawHvgShardCache  # noqa: E402
from src.eval.arms.baseline import normalize_plate_for_eval  # noqa: E402
from src.baselines.artifacts import Baseline  # noqa: E402
from src.baselines.fit import fit_kinds  # noqa: E402
from src.baselines.roles import Arms, baseline_condition_roles  # noqa: E402
from src.baselines.space import CP10K, LOG1P_CP10K_FULL, library_namespace  # noqa: E402
from src.eval.cache.eval_npz_reader import EvalNpzReader  # noqa: E402
from src.eval.reference.truth_io import control_library  # noqa: E402
from scripts.figures.figure4.score import drug_axis  # noqa: E402
from scripts.figures.figure4.extract.progeny import (COPY_TABLE, DOSE, FLOOR, MIX, assert_pad_ungated, depth, gated_pairs,  # noqa: E402
                                      safe, score)
from src.model.inference_output import to_raw_integer  # noqa: E402
from src.paths import expand_path  # noqa: E402

CACHE = Path(expand_path("{data:tahoe}/hvg_cache/geneset_12aead7e0e3f60fa__n_3072"))
LIBRARY = Path(expand_path("{data:tahoe}/library_full"))   # plate<p>_tscp_count.npy: each h5ad row's L_full
FIT_SPACE = LOG1P_CP10K_FULL   # the Additive's fit space, as every baseline in the paper (src/baselines/space.py)
RUN_DIR = REPO / "runs/tahoe/bio/biology/progeny3072_cells005_keepctrl"   # the GENESIS run whose train rows fit the Additive
NEIGHBORS = REPO / "core_util_data/tahoe/eval_inputs" / COPY_TABLE   # frozen input
PARTIAL = Path("data/upstream/compare_baselines")    # in the bundle: resumable sums and the summary's sidecars
CONTROL = "DMSO_TF"
DOSES = {0.0: 0, 0.05: 1, 0.5: 2, 5.0: 3}     # every dose the cache stores; anything else is a new encoding
MIN_DRAWN = 4          # fit_baselines' min_cells: a perturbed mean from fewer cells is not a mean
MIN_MEASURED = 50      # the response atlas's floor: a neighbor plate below it is not a measurement
CHUNK = 100_000
G = 3072


def train_idx() -> Path:
    """The run's split_cache/train_idx.npy, looked up when first read, so importing this module needs no runs/."""
    hit = next(RUN_DIR.glob("*/261016_181841*/split_cache/train_idx.npy"), None)
    if hit is None:
        raise FileNotFoundError(f"no */261016_181841*/split_cache/train_idx.npy under {RUN_DIR}: "
                                "download the GENESIS checkpoints (README: Setup)")
    return hit


# ----------------------------------------------------------------------------
# The mixture: n plates, equal weights, the pair formula when n is 2
# ----------------------------------------------------------------------------

def plate_mixture(pert: list[np.ndarray], ctrl: list[np.ndarray]) -> np.ndarray:
    """log2((sum_p m_p + P*1e-9) / (sum_p c_p + P*1e-9)). One plate is the single-plate formula."""
    if not pert or len(pert) != len(ctrl):
        raise ValueError(f"need the same non-empty plates of perturbed and control, got {len(pert)} and {len(ctrl)}")
    eps = len(pert) * 1e-9
    return np.log2((sum(pert) + eps) / (sum(ctrl) + eps))


def neighbor_gate(held_ok: np.ndarray, det: np.ndarray) -> np.ndarray:
    """A gene stays where the held-out controls detect it and the neighbor's plates do too, on average."""
    if det.ndim != 2 or det.shape[1] != held_ok.shape[0] or len(det) == 0:
        raise ValueError(f"detection {det.shape} does not match the held-out gate {held_ok.shape}")
    return held_ok & (det.mean(0) >= FLOOR)


def headline(mae: pd.DataFrame) -> pd.Series:
    """Per line the mean over pathways, then the mean over lines. The number the figure's bar is."""
    return mae.groupby(["arm", "cell_name"]).mae.mean().groupby("arm").mean()


# ----------------------------------------------------------------------------
# The rows: who was trained on, and who is a neighbor's measurement
# ----------------------------------------------------------------------------

def _cats(shard: Path, name: str) -> list[str]:
    return [str(x) for x in np.load(shard / "obs_sidecar" / f"{name}__categories.npy", allow_pickle=True)]


def _codes(shard: Path, name: str) -> np.ndarray:
    return np.load(shard / "obs_sidecar" / f"{name}__codes.npy")


def _pack(rows: np.ndarray, keys: np.ndarray, cells: list[str], drugs: list[str]) -> tuple:
    """(rows, group id per row, per-group string labels). ``keys`` is a recarray of integer codes."""
    if len(rows) != len(np.unique(rows)):
        raise SystemExit("a row was indexed twice")
    _, first, gid = np.unique(keys, return_index=True, return_inverse=True)
    t = keys[first]
    lab = {}
    if "c" in t.dtype.names:
        lab["cell"] = np.asarray(cells)[t["c"]]
    if "d" in t.dtype.names:
        lab["drug"] = np.asarray(drugs)[t["d"]]
    if "z" in t.dtype.names:
        lab["dose"] = np.array(sorted(DOSES))[t["z"]]
    lab["plate"] = t["p"].astype(np.int16)
    return rows, gid.astype(np.int32), lab


# Over 40 lines: NEVER-AUTO numerical code (hours to run, never run here), kept verbatim.
def index_rows(man: dict, train: np.ndarray, neighbors: set[str], drugs: set[str]) -> dict:
    """Row ids and group ids. Codes are per shard, so they become one global id first.
    A row may be both a training cell and a neighbor measurement: the 5% draw kept controls."""
    shards = [(CACHE / s["cache_subdir"], int(s["global_obs_offset"]), int(s["n_obs"])) for s in man["shards"]]
    drug_names = sorted({n for shard, _, _ in shards for n in _cats(shard, "drug")})
    cell_names = sorted({n for shard, _, _ in shards for n in _cats(shard, "cell_name")})
    for what, want, have in (("neighbor lines", neighbors, cell_names), ("axis drugs", drugs, drug_names)):
        if missing := sorted(set(want) - set(have)):
            raise SystemExit(f"{what} are not in the cache: {missing}")
    drug_id = {n: i for i, n in enumerate(drug_names)}
    cell_id = {n: i for i, n in enumerate(cell_names)}
    add_rows, add_keys, pert_rows, pert_keys, ctrl_rows, ctrl_keys = [], [], [], [], [], []
    for shard, lo, n in shards:
        dcat, zcat = _cats(shard, "drug"), _cats(shard, "drugname_drugconc")
        # The two category lists are not index-aligned (salt forms swap slots). A row's own
        # codes are: drug from the drug code, dose from the concentration code.
        zdose = []
        for z in zcat:
            rec = ast.literal_eval(z)[0]
            if (dose := round(float(rec[1]), 6)) not in DOSES:
                raise SystemExit(f"{shard.name}: dose {dose} is not one of {sorted(DOSES)}")
            zdose.append(DOSES[dose])
        plates = _cats(shard, "plate")
        if len(plates) != 1:
            raise SystemExit(f"{shard.name}: expected one plate, got {plates}")
        plate = np.int16(normalize_plate_for_eval(plates[0]))
        ccat = _cats(shard, "cell_name")
        did = np.array([drug_id[x] for x in dcat], dtype=np.int32)[_codes(shard, "drug")]
        cid = np.array([cell_id[x] for x in ccat], dtype=np.int32)[_codes(shard, "cell_name")]
        dose = np.array(zdose, dtype=np.int8)[_codes(shard, "drugname_drugconc")]
        i0, i1 = int(np.searchsorted(train, lo)), int(np.searchsorted(train, lo + n))
        local = train[i0:i1] - lo
        if len(local):
            add_rows.append(train[i0:i1])
            add_keys.append(np.rec.fromarrays(
                [cid[local], did[local], dose[local], np.full(len(local), plate, np.int16)], names="c,d,z,p"))
        nb = np.isin(cid, [cell_id[x] for x in neighbors])
        want = nb & np.isin(did, [drug_id[x] for x in drugs]) & (dose == DOSES[5.0])
        ctrl = nb & (did == drug_id[CONTROL])
        if int(want.sum()):
            pert_rows.append(np.flatnonzero(want) + lo)
            pert_keys.append(np.rec.fromarrays(
                [cid[want], did[want], np.full(int(want.sum()), plate, np.int16)], names="c,d,p"))
        if int(ctrl.sum()):
            ctrl_rows.append(np.flatnonzero(ctrl) + lo)
            ctrl_keys.append(np.rec.fromarrays(
                [cid[ctrl], np.full(int(ctrl.sum()), plate, np.int16)], names="c,p"))
        print(f"  {shard.name}: {len(local):,} train rows, {int(want.sum()):,} neighbor measurements", flush=True)
    if not add_rows or not pert_rows or not ctrl_rows:
        raise SystemExit(f"empty index: train groups {len(add_rows)}, neighbor plates {len(pert_rows)}, "
                         f"neighbor controls {len(ctrl_rows)}")
    add = _pack(np.concatenate(add_rows), np.concatenate(add_keys), cell_names, drug_names)
    pert = _pack(np.concatenate(pert_rows), np.concatenate(pert_keys), cell_names, drug_names)
    ctrl = _pack(np.concatenate(ctrl_rows), np.concatenate(ctrl_keys), cell_names, drug_names)
    return {"add": add, "pert": pert, "ctrl": ctrl}


def merged_rows(idx: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One sorted row list, with a group id per accumulator (-1 where that row does not contribute)."""
    parts = []
    for name in ("add", "pert", "ctrl"):
        rows, gid, _ = idx[name]
        tag = np.full((len(rows), 3), -1, np.int32)
        tag[:, ("add", "pert", "ctrl").index(name)] = gid
        parts.append(np.rec.fromarrays([rows, tag[:, 0], tag[:, 1], tag[:, 2]], names="row,add,pert,ctrl"))
    both = np.concatenate(parts)
    both.sort(order="row")
    # A row in two accumulators is two records: fold them, and refuse two groups in one accumulator.
    # concatenate returns a structured ndarray, so fields are both["row"], not attributes.
    uniq, start = np.unique(both["row"], return_index=True)
    start = np.append(start, len(both))
    out = np.full((len(uniq), 3), -1, np.int32)
    for i, (a, b) in enumerate(zip(start[:-1], start[1:])):
        block = both[a:b]
        for k, col in enumerate(("add", "pert", "ctrl")):
            got = block[col]
            got = got[got >= 0]
            if len(set(got.tolist())) > 1:
                raise SystemExit(f"row {int(uniq[i])} is in two {col} groups")
            if len(got):
                out[i, k] = got[0]
    return uniq, out[:, 0], out[:, 1], out[:, 2]


def scatter(sums: np.ndarray, gid: np.ndarray, mat: sparse.spmatrix) -> None:
    """sums[gid] += the rows of ``mat``. ``gid`` may be any ids; they are compacted for the product."""
    if len(gid) == 0:
        return
    uniq, inv = np.unique(gid, return_inverse=True)
    ind = sparse.csr_matrix((np.ones(len(inv), np.float64), (inv, np.arange(len(inv)))),
                            shape=(len(uniq), len(inv)))
    sums[uniq] += (ind @ mat).toarray()


def shard_libraries(man: dict) -> np.ndarray:
    """Every cache row's L_full, in cache order: a shard is one plate's h5ad, its obs sidecar maps a shard row to
    that h5ad's row, and that row's tscp_count (the full X row sum on Tahoe) is the library."""
    lib = np.zeros(int(man["total_obs"]), np.float32)
    for s in man["shards"]:
        d = CACHE / s["cache_subdir"]
        plate = re.search(r"/plate(\d+)_", json.loads((d / "meta.json").read_text())["h5ad_path"]).group(1)
        lo, n = int(s["global_obs_offset"]), int(s["n_obs"])
        lib[lo:lo + n] = np.load(LIBRARY / f"plate{plate}_tscp_count.npy", mmap_mode="r")[
            np.load(d / "obs_sidecar" / "local_obs_index.npy")]
    if not (lib > 0).all():
        raise SystemExit(f"{int((lib <= 0).sum())} cache rows have no L_full")
    return lib


def to_fit_space_rows(mat: sparse.csr_matrix, lib: np.ndarray) -> sparse.csr_matrix:
    """log1p(1e4 x / L_full) on the stored entries: a zero stays zero, so the sum stays sparse."""
    out = sparse.csr_matrix(mat.multiply((CP10K / lib.astype(np.float64))[:, None]))
    out.data = np.log1p(out.data)
    return out


# Over 40 lines: NEVER-AUTO numerical code (hours to run, never run here).
def accumulate(rows: np.ndarray, gids: np.ndarray, shapes: dict[str, tuple], partial: Path) -> None:
    """Per-group sums over ``rows``, into ``partial``: the Additive's in its fit space (log1p(1e4 x / L_full) per
    cell), the neighbor's raw counts (and, for its controls, nonzero counts). One shard file at a time, so a killed
    run resumes from the shards already flushed; a cache of Additive sums in another fit space is refused."""
    partial.mkdir(parents=True, exist_ok=True)
    marker = partial / "fit_space.json"
    if (partial / "add_sum.npy").exists() and (not marker.exists() or json.loads(marker.read_text())["add_sum"] != FIT_SPACE):
        raise SystemExit(f"{partial} holds Additive sums of another fit space (no {marker.name} naming {FIT_SPACE}): "
                         "use a fresh --bundle")
    marker.write_text(json.dumps({"add_sum": FIT_SPACE}))
    man = json.loads((CACHE / "manifest.json").read_text())
    lib = shard_libraries(man)
    done_path = partial / "done.json"
    done = set(json.loads(done_path.read_text())) if done_path.exists() else set()
    maps = {}
    for name, shape in shapes.items():
        path = partial / f"{name}.npy"
        if path.exists():
            arr = np.load(path, mmap_mode="r+")
            if arr.shape != shape:
                raise SystemExit(f"{path.name} is {arr.shape}, this index needs {shape}: delete {partial}")
        else:
            arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=shape)
            arr[:] = 0
        maps[name] = arr
    cache = RawHvgShardCache(CACHE, eager_obs=False)
    if cache.var_names != R.resolve_panel("progeny3072").genes:
        raise SystemExit("the hvg cache gene axis is not the panel's: positional means would mislabel genes")
    checked = False
    t0 = time.time()
    for s in man["shards"]:
        if s["cache_subdir"] in done:
            print(f"  {s['cache_subdir']}: already summed", flush=True)
            continue
        lo, hi = int(s["global_obs_offset"]), int(s["global_obs_offset"]) + int(s["n_obs"])
        a, b = int(np.searchsorted(rows, lo)), int(np.searchsorted(rows, hi))
        sl, sg = rows[a:b], gids[a:b]
        for c0 in range(0, len(sl), CHUNK):
            r, g = sl[c0:c0 + CHUNK], sg[c0:c0 + CHUNK]
            mat = cache.gene_expression(r).astype(np.float64)
            lib_r = lib[r]
            if (np.asarray(mat.sum(axis=1)).ravel() > lib_r).any():
                raise SystemExit("a row holds more counts than its L_full: the libraries are not this cache's rows")
            if not checked:
                if (mat.data == 0).any() or int((mat[:100].toarray() != 0).sum()) != int(mat[:100].nnz):
                    raise SystemExit("the CSR stores explicit zeros, or drops real counts: detection would be wrong")
                checked = True
            for k, col in enumerate(("add", "pert", "ctrl")):
                m = g[:, k] >= 0
                if m.any():
                    scatter(maps[f"{col}_sum"], g[m, k], to_fit_space_rows(mat[m], lib_r[m]) if col == "add" else mat[m])
            m = g[:, 2] >= 0
            if m.any():
                ones = mat[m].copy()
                ones.data = np.ones(ones.nnz, np.float64)
                scatter(maps["ctrl_nnz"], g[m, 2], ones)
        for arr in maps.values():
            arr.flush()
        done.add(s["cache_subdir"])
        done_path.write_text(json.dumps(sorted(done)))
        print(f"  {s['cache_subdir']}: {b - a:,} rows | {time.time() - t0:.0f}s", flush=True)
    if done != {s["cache_subdir"] for s in man["shards"]}:
        raise SystemExit(f"summed {sorted(done)}, the cache has {[s['cache_subdir'] for s in man['shards']]}")


# ----------------------------------------------------------------------------
# The fit and the two predictions
# ----------------------------------------------------------------------------

def fit_additive(lab: dict, n: np.ndarray, partial: Path) -> Baseline:
    """Sufficient statistics from the drawn cells. A held-out line's own test drugs must not be in them."""
    marker = partial / "fit_space.json"
    if not marker.exists() or json.loads(marker.read_text())["add_sum"] != FIT_SPACE:
        raise SystemExit(f"{partial}/add_sum.npy is not marked as {FIT_SPACE} sums: rerun the accumulation")
    sums = np.load(partial / "add_sum.npy", mmap_mode="r")
    mean = np.asarray(sums) / np.maximum(n, 1)[:, None]
    ctrl = lab["drug"] == CONTROL
    # One control pool per (line, plate). DMSO carries a single dose; a second label is
    # the same pool split, so the pools are combined by cell count rather than left as two.
    acc, nn = {}, {}
    for i in np.flatnonzero(ctrl):
        key = (str(lab["cell"][i]), str(int(lab["plate"][i])))
        acc[key] = acc.get(key, 0.0) + n[i] * mean[i]
        nn[key] = nn.get(key, 0) + int(n[i])
    mu_ctrl = {k: acc[k] / nn[k] for k in acc}
    keep = np.flatnonzero(~ctrl & (n >= MIN_DRAWN))
    strata = [(str(lab["cell"][i]), str(int(lab["plate"][i]))) for i in keep]
    if missing := sorted(set(strata) - set(mu_ctrl)):
        raise SystemExit(f"{len(missing)} (line, plate) strata have perturbed draws but no drawn control, "
                         f"e.g. {missing[:3]}")
    roles = baseline_condition_roles("tahoe")
    keys = [roles.arm_key({"cell_name": c, "drug": str(lab["drug"][i]), "dose": float(lab["dose"][i]),
                           "plate": f"plate{p}"}) for i, (c, p) in zip(keep, strata)]
    arms = Arms.build(keys, mean[keep], np.stack([mu_ctrl[s] for s in strata]))
    print(f"  additive: {len(keep):,} perturbed groups (>= {MIN_DRAWN} drawn cells), "
          f"{len(mu_ctrl):,} control strata; {int(ctrl.size - ctrl.sum() - len(keep)):,} groups below the minimum "
          f"dropped", flush=True)
    return fit_kinds(arms, roles, list(R.resolve_panel("progeny3072").genes), {"additive"}, fit_space=FIT_SPACE)["additive"]


def assert_no_leak(lab: dict, held: set[str], drugs: set[str]) -> None:
    """The 5 lines' on-axis drugs at 5 uM are the test. They must not have trained the fit."""
    hit = [f"{c} / {d}" for c, d, z in zip(lab["cell"], lab["drug"], lab["dose"])
           if c in held and d in drugs and np.isclose(z, DOSE)]
    if hit:
        raise SystemExit(f"additive saw {len(hit)} held-out conditions, e.g. {hit[:3]}")


def additive_means(model: Baseline, truth: T.Truth, conds: dict, display: dict) -> dict:
    """(safe line, drug, plate) -> mean count. Emitted onto the line's own controls in the fit's space (their
    libraries from the eval cache's L_full sidecar), then snapped in that space."""
    panel = R.resolve_panel("progeny3072")
    space = R.prediction_space(model)
    libs = None if space == "raw" else EvalNpzReader(library_namespace(Path(panel.cache)))
    out = {}
    for cs, disp in display.items():
        if (disp,) not in model.tables["held"]:
            print(f"  additive: {disp} has no cell effect (held out of every training drug); prediction is mu + beta",
                  flush=True)
        plates = sorted({p for ps in conds.values() for p in ps})
        ctrl = {p: R.load_control(panel.cache, cell_name=disp, plate=f"plate{p}", n_genes=G) for p in plates}
        lib = {p: None if libs is None else control_library(libs, {"cell_name": disp, "plate": f"plate{p}"}, ctrl[p])
               for p in plates}
        for drug, ps in conds.items():
            if (drug, round(float(DOSE), 6)) not in model.tables["pert"]:
                raise SystemExit(f"additive has no drug effect for {drug} at {DOSE} uM: it was not in training")
            for p in ps:
                cells = R.baseline_pred(model, "additive", drug=drug, cell=disp, dose=DOSE, plate=p,
                                        ctrl_raw=ctrl[p], n_cells=truth.n_cells[(cs, drug, f"plate{p}")], ctrl_lib=lib[p])
                out[(cs, drug, int(p))] = to_raw_integer(
                    torch.as_tensor(cells, dtype=torch.float32), space).to(torch.float64).mean(0).numpy()
        print(f"  additive predicted {disp}", flush=True)
    return out


def neighbor_stats(lab_pert: dict, n_pert: np.ndarray, lab_ctrl: dict, n_ctrl: np.ndarray,
                   partial: Path) -> tuple[dict, dict, dict]:
    """(cell, drug, plate) -> perturbed mean; (cell, plate) -> control mean and detection. Plates under 50 cells dropped."""
    psum = np.load(partial / "pert_sum.npy", mmap_mode="r")
    csum, cnnz = np.load(partial / "ctrl_sum.npy", mmap_mode="r"), np.load(partial / "ctrl_nnz.npy", mmap_mode="r")
    pert, small = {}, 0
    for i, (c, d, p, n) in enumerate(zip(lab_pert["cell"], lab_pert["drug"], lab_pert["plate"], n_pert)):
        if n < MIN_MEASURED:
            small += 1
            continue
        pert[(str(c), str(d), int(p))] = psum[i] / n
    ctrl, det = {}, {}
    for i, (c, p, n) in enumerate(zip(lab_ctrl["cell"], lab_ctrl["plate"], n_ctrl)):
        if n < MIN_MEASURED:
            raise SystemExit(f"neighbor {c} plate {p} has {n} control cells (< {MIN_MEASURED})")
        ctrl[(str(c), int(p))] = csum[i] / n
        det[(str(c), int(p))] = cnnz[i] / n
    print(f"  nearest neighbor: {len(pert):,} measured plates, {small} below {MIN_MEASURED} cells dropped", flush=True)
    return pert, ctrl, det


# ----------------------------------------------------------------------------
# The pathway error
# ----------------------------------------------------------------------------

# Over 40 lines: NEVER-AUTO numerical code (hours to run, never run here), kept verbatim.
def responses(truth: T.Truth, conds: dict, add: dict, pert: dict, ctrl: dict, det: dict,
              copy_of: dict, axis: set[str], replicated: set[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Mixture log2FC on the on-axis drugs, and each plate of a replicated pair.

    The mixture is the mean-|error| comparison. The single plates are the summary's
    cross-batch panel. One gate per (line, drug, arm), the same gate the mixture uses,
    applied to every plate: the figure scores a replicated pair that way."""
    keys, lfc, gate = [], [], []
    skipped = []

    def push(arm: str, ms: list, cs: list, ok_arm: np.ndarray) -> None:
        if drug in axis:
            keys.append((c, drug, arm, MIX))
            lfc.append(np.where(ok_arm, plate_mixture(ms, cs), 0.0))
            gate.append(ok_arm)
        if drug not in replicated:
            return
        if len(ms) != 2:
            raise SystemExit(f"{drug} is replicated on {len(ms)} plates, not a pair")
        for slot, (m_, c_) in enumerate(zip(ms, cs)):
            keys.append((c, drug, arm, slot))
            lfc.append(np.where(ok_arm, plate_mixture([m_], [c_]), 0.0))
            gate.append(ok_arm)

    for c in sorted({k[0] for k in truth.pmean}):
        nb = copy_of[c]
        for drug, plates in conds.items():
            ps = [f"plate{p}" for p in plates]
            if any((c, drug, p) not in truth.pmean for p in ps):
                raise SystemExit(f"{c} / {drug}: truth has no plate {ps}")
            ctl = [truth.cmean[(c, p)] for p in ps]
            ok = np.mean([truth.D[(c, p)] for p in ps], axis=0) >= FLOOR
            push("additive", [add[(c, drug, int(p))] for p in plates], ctl, ok)
            have = [(nb, drug, int(p)) in pert and (nb, int(p)) in ctrl for p in plates]
            if not all(have):
                skipped.append((c, drug, [p for p, h in zip(plates, have) if not h]))
                continue
            ms = [pert[(nb, drug, int(p))] for p in plates]
            cs = [ctrl[(nb, int(p))] for p in plates]
            push("nearest_neighbor", ms, cs, neighbor_gate(ok, np.stack([det[(nb, int(p))] for p in plates])))
    if skipped:
        print(f"  nearest neighbor missing {len(skipped)} (line, drug)s, e.g. {skipped[:3]}", flush=True)
    if len(skipped) > 0.2 * len(conds) * len({k[0] for k in truth.pmean}):
        raise SystemExit("nearest neighbor is missing more than a fifth of the conditions: the plates do not match")
    return pd.DataFrame(keys, columns=["cell_name", "drug", "arm", "slot"]), np.array(lfc), np.array(gate)


def check_truth(truth: T.Truth, conds: dict, sets: pd.DataFrame, bundle: Path) -> None:
    """Raise unless this mixture, scored, reproduces the pathway values the figure was drawn from."""
    keys, lfc, gate = [], [], []
    for c in sorted({k[0] for k in truth.pmean}):
        for drug, plates in conds.items():
            ps = [f"plate{p}" for p in plates]
            if any((c, drug, p) not in truth.pmean for p in ps):
                continue
            ctl = [truth.cmean[(c, p)] for p in ps]
            ok = np.mean([truth.D[(c, p)] for p in ps], axis=0) >= FLOOR
            ms = [truth.pmean[(c, drug, p)] for p in ps]
            keys.append((c, drug, "truth", MIX))
            lfc.append(np.where(ok, plate_mixture(ms, ctl), 0.0))
            gate.append(ok)
    assert_pad_ungated(np.array(gate))
    got = score(pd.DataFrame(keys, columns=["cell_name", "drug", "arm", "slot"]),
                  np.array(lfc), np.array(gate), sets, truth.genes, False)
    ref = pd.read_parquet(bundle / "data/tahoe_progeny_raw/module_values.parquet")
    ref = ref[ref.arm == "truth"][["programme", "cell_name", "drug", "v"]]
    m = ref.merge(got, on=["programme", "cell_name", "drug"], how="outer", indicator=True)
    if (m._merge != "both").any() or not np.allclose(m.v_x, m.v_y, rtol=1e-9, atol=1e-12):
        raise SystemExit(f"truth pathway values are not the figure's: "
                         f"{int((m._merge != 'both').sum())} unmatched, "
                         f"max |diff| {np.nanmax(np.abs(m.v_x - m.v_y)):.3g}")
    print(f"  ✓ truth: {len(m)} pathway values reproduce the figure", flush=True)


# Over 40 lines: NEVER-AUTO numerical code (hours to run, never run here), kept verbatim.
def main() -> None:
    """The figure is measured versus GENESIS. These three are the comparisons, on the same
    on-axis drugs (drug_axis.tsv on_axis) and the same pathway score (scripts.figures.figure4.extract.progeny.score):

      state              already in metrics.tsv (read, not refit)
      additive           fit on the cells GENESIS trained on: every control, plus the
                         4,579,207 perturbed cells of the Fig 3a 5% draw, in log1p(1e4 x / L_full)
                         as every baseline is (FIT_SPACE). A held-out condition is predicted the
                         way every baseline is: the fitted delta added to that line's own control
                         cells in that space, denormalised with each cell's L_full, snapped to
                         integer counts, then averaged.
      nearest_neighbor   the published cosine-PCA neighbor's measured 5 uM response on
                         the drug's own plates, equal plate weights, its own controls,
                         and a gene kept only where the neighbor's controls detect it too.

    Mean |error| is the figure's: per line the mean over the 14 pathways, then the mean
    over the 5 lines. Raw, centred, and depth-adjusted. Depth subtracts the measured
    library-size shift from every arm, the same number the depth panels use. Where an
    arm shares truth's gene gate that shift cancels in the error, so Additive's depth
    error equals its raw error. The neighbor's gate is stricter, so its depth error
    does not.

    The depth summary draws these arms. It reads two sidecars written here, under the
    figure's arm name (nearest_neighbor is copy_closest_15): summary_metrics__depth.tsv
    and summary_plates__depth.parquet. The plates are the replicated pairs, each plate
    alone, so the summary's cross-batch panel can score them. The published metrics
    and plate_values are left as the figure's own arms. Also writes tables/comparisons.tsv.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, required=True, help="reads BUNDLE/data, writes BUNDLE/" + str(PARTIAL)
                    + " and BUNDLE/tables/comparisons.tsv")
    bundle = ap.parse_args().bundle.resolve()
    partial = bundle / PARTIAL
    man = json.loads((CACHE / "manifest.json").read_text())
    if [ln.strip() for ln in (CACHE / "hvg_gene_names.txt").read_text().splitlines() if ln.strip()] != \
            R.resolve_panel("progeny3072").genes:
        raise SystemExit("cache genes != the panel")
    ax = drug_axis.table(bundle / "data/drug_axis/drug_axis.tsv")
    conds, axis, replicated = drug_axis.conditions(ax), set(ax.drug[ax.on_axis]), set(ax.drug[ax.replicated])
    _, test = gated_pairs(bundle / "data/upstream")
    display = dict(zip([safe(c) for c in test], test))
    nb = pd.read_csv(NEIGHBORS, sep="\t")
    copy_of = {safe(r.cell_name): r.neighbor for r in nb.itertuples()}
    if set(copy_of) != set(display):
        raise SystemExit(f"neighbor table lines {sorted(copy_of)} are not the held-out lines {sorted(display)}")
    print("truth, and that this scoring is the figure's:", flush=True)
    truth = T.load_truth(test, conds)
    sets = pd.read_csv(bundle / "data/tahoe_progeny_raw/programmes.tsv", sep="\t")
    check_truth(truth, conds, sets, bundle)
    print("indexing train rows and neighbor measurements:", flush=True)
    train = np.sort(np.load(train_idx()))
    if train.max() >= man["total_obs"]:
        raise SystemExit(f"train row {train.max()} is past the cache ({man['total_obs']} rows)")
    idx = index_rows(man, train, set(copy_of.values()), set(conds))
    # Every drug the summary scores on a held-out line is a test condition, the cross-batch
    # pairs included, so none of them at 5 uM may have trained the fit.
    assert_no_leak(idx["add"][2], set(display.values()), axis | replicated)
    rows, ga, gp, gc = merged_rows(idx)
    print(f"  {len(rows):,} rows, {len(idx['add'][2]['cell']):,} additive groups", flush=True)
    accumulate(rows, np.column_stack([ga, gp, gc]), {
        "add_sum": (len(idx["add"][2]["cell"]), G),
        "pert_sum": (len(idx["pert"][2]["cell"]), G),
        "ctrl_sum": (len(idx["ctrl"][2]["cell"]), G),
        "ctrl_nnz": (len(idx["ctrl"][2]["cell"]), G)}, partial)
    print("fitting additive and reading the neighbor:", flush=True)
    model = fit_additive(idx["add"][2], np.bincount(idx["add"][1]), partial)
    scored = {d: conds[d] for d in axis | replicated}
    add = additive_means(model, truth, scored, display)
    n_pert, n_ctrl = np.bincount(idx["pert"][1]), np.bincount(idx["ctrl"][1])
    pert, ctrl, det = neighbor_stats(idx["pert"][2], n_pert, idx["ctrl"][2], n_ctrl, partial)
    keys, lfc, gate = responses(truth, scored, add, pert, ctrl, det, copy_of, axis, replicated)
    assert_pad_ungated(gate)
    # Truth's measured depth, one number per (line, drug, slot), given to every arm. The depth
    # panels subtract this before the pathway score. A single plate uses that plate's own shift.
    dep = depth(keys, scored, bundle / "context_analysis/data/upstream/library_size__tahoe.tsv")
    rows_out = []
    for split, centre, shift in (("raw", False, None), ("centred", True, None), ("depth", False, dep)):
        got = score(keys, lfc, gate, sets, truth.genes, centre, shift)
        if split == "depth":
            # The summary's arm id for the neighbor is Fig. 4's. Published plate_values stay untouched:
            # a hole in an extra arm would drop compounds from every arm's cross-batch statistic.
            plates = got.loc[got.slot != MIX, ["programme", "cell_name", "drug", "arm", "slot", "v"]]
            plates = plates.assign(set="grid", arm=plates.arm.replace({"nearest_neighbor": "copy_closest_15"}))
            plates.to_parquet(partial / "summary_plates__depth.parquet", index=False)
            print(f"  depth plates: {len(plates)} single-plate pathway values", flush=True)
        ref = pd.read_parquet(bundle / f"data/tahoe_progeny_{split}/module_values.parquet")
        ref = ref[(ref.arm == "truth") & ref.drug.isin(axis)][
            ["programme", "cell_name", "drug", "v"]].rename(columns={"v": "truth"})
        m = got[got.slot == MIX].merge(ref, on=["programme", "cell_name", "drug"], how="left")
        if m.truth.isna().any():
            raise SystemExit(f"{split}: a scored row has no truth")
        mae = (m.assign(mae=(m.v - m.truth).abs())
               .groupby(["programme", "cell_name", "arm"], sort=False)
               .agg(n=("mae", "size"), mae=("mae", "mean")).reset_index())
        if split == "depth":
            side = mae.replace({"arm": {"nearest_neighbor": "copy_closest_15"}})
            side.to_csv(partial / "summary_metrics__depth.tsv", sep="\t", index=False, float_format="%.8g")
        published = pd.read_csv(bundle / f"data/tahoe_progeny_{split}/metrics.tsv", sep="\t")
        published = published[published.arm.isin(["genesis", "state", "predict0"])]
        both = pd.concat([published, mae], ignore_index=True)
        h = headline(both)
        print(f"{split}: " + "  ".join(f"{a} {h[a]:.3f}" for a in
                                       ["genesis", "state", "additive", "nearest_neighbor", "predict0"]), flush=True)
        rows_out.append(both.assign(split=split))
    out = pd.concat(rows_out, ignore_index=True)
    path = bundle / "tables/comparisons.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, sep="\t", index=False, float_format="%.8g")
    print(f"wrote {path.relative_to(bundle)}", flush=True)


if __name__ == "__main__":
    main()
