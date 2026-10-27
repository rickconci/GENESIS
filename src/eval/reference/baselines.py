"""Plate-pair energy for the four linear baselines, same contract as the neural arms.

The gated energy bars compare ``cross_mean`` of energy distance in ``--report-space``
(default slice_med, the published tables; log1p_rgs for the Appendix) on the pairs
``reference.py replicate_pairs`` already scored in that same space. Those parquets are
log1p neural models. The baselines (``--baseline-root``, fit on the split the GENESIS
pairs come from) are emitted as the split-half scorer emits them, in the space their
fit calls for (``deg_readout.prediction_space``): raw counts for a raw fit, log1p counts
for a log1p_cp10k_full fit, which also reads each control cell's L_full from the
``__with_libfull`` sidecar. Every projection below takes that space.

One pass over the GENESIS pair list (556). Each direction is truth on one plate
against cells emitted from the *other* plate's DMSO pool, ``n_cells`` equal to
that plate's truth count — the same ``n_gen == n_truth`` the neural parquets
have. ``true_true__*`` is copied off the GENESIS parquet: it is the real batch
gap, and recomputing it would only invite a second draw of the energy subsample.

Top-100 DEG and Pearson r Δ go through ``score_pair`` exactly as
``score_true_gen`` does for the neural arms, prediction in RAW: truth against its
own plate's DMSO, the emission against the DMSO of the plate it was emitted from.
Both are deterministic, so no seed is pinned for them.

Run it as ``python scripts/eval/reference.py replicate_pairs_baselines``; ``run`` is the entry.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[3]

from src.eval.reference.pairs import (  # noqa: E402
    CTRL_FLOORS,
    DEFAULT_REPORT_SPACE,
    REPORT_SPACES,
    check_report_space,
    metric_cols,
    load_pred,
    prediction_path,
    read_control,
    read_perturbed,
    replicate_design,
    to_t,
)
from src.eval.reference.truth_io import control_library  # noqa: E402
from src.baselines.space import library_namespace  # noqa: E402

from src.paths import expand_path  # noqa: E402
EVAL_CACHE = Path(expand_path("{data:tahoe}/eval_cache/normalization_raw_counts/geneset_c0d6420ffc0fa861__n_2000"))
from src.eval.readout.deg_readout import (  # noqa: E402
    BASELINES,
    GENES,
    baseline_pred,
    load_baselines,
    prediction_space,
)
from src.eval.cache.eval_npz_reader import EvalNpzReader  # noqa: E402
from src.eval.metrics.energy import energy_distance  # noqa: E402
from src.eval.metrics.eval_metrics import score_pair
from src.eval.metrics.reporting import materialize_metric_tensors
from src.eval.metrics.spaces import (
    _project_predictions_to_spaces,
    _project_raw_to_spaces,
    set_dataset_median_umi,
)

RESULTS = REPO / "results/tahoe/plate_pairs/slice_med_seeded"
GENESIS_PARQUET = RESULTS / "model_vs_batch_gap__genesis.parquet"
REPORT_SPACE = DEFAULT_REPORT_SPACE  # set once by --report-space
BASELINE_ROOT: Path | None = None  # --baseline-root, set once by run()
ENERGY_ALL_CELLS = False  # --energy-all-cells, set once by run(): energy on every cell, as the neural arms' uncapped tables
# --check-pred-root: the GENESIS predictions behind model_vs_batch_gap__genesis.parquet (default: Fig 2a's bio run)
CHECK_PRED_ROOT = (
    REPO / "results/tahoe/split_half/pertgen"
    / "260916_114938_cur-between_epoch-tp458752-bs1792xnrm256_nre256_lr0.0009_wd0.0001_"
    "biohold_condmlp-D256-h4096x8192_cond-plate_mlpout-relu_vg0.5p0.5np6k_log1p_s64_"
    "h4096x8192_nrs256_vg05rs256_5ep_state2k_tahoe_bio_100cond_seed0__last__test"
    / "predictions"
)
# Same cap compute_arm_metrics uses. Pinned so a rebuild does not redraw the
# subsample: the neural parquets are already one draw, frozen on disk.
SUBSAMPLE_CAP = 2048
DELTA_METRICS = ("deg_adaptivek_truth", "pearson_r_diff")

LABEL = {
    "perturbation_mean": "PertMean",
    "linear_ridge": "LinRidge",
    "additive": "Additive",
    "context_mean": "CtxMean",
}


def _seed(*parts: object) -> int:
    h = hashlib.sha256(repr(parts).encode()).digest()
    return int.from_bytes(h[:8], "little") % (2**31)


def report_energy(
    pred_raw_or_log: torch.Tensor,
    truth_raw: torch.Tensor,
    *,
    model_output_space: str,
    seed: int,
) -> float:
    """Energy in the space the batch-gap parquet stores, one seeded subsample."""
    pred_sp = _project_predictions_to_spaces(pred_raw_or_log, model_output_space)
    truth_sp = _project_raw_to_spaces(truth_raw)
    g = torch.Generator(device=pred_raw_or_log.device)
    g.manual_seed(seed)
    e = energy_distance(
        pred_sp[REPORT_SPACE], truth_sp[REPORT_SPACE], max_samples=None if ENERGY_ALL_CELLS else SUBSAMPLE_CAP, generator=g,
    )
    val = float(e.detach().cpu())
    if not np.isfinite(val):
        raise RuntimeError(f"non-finite energy (space={model_output_space})")
    return val


def delta_scores(
    truth_raw: torch.Tensor,
    pred: torch.Tensor,
    ctrl_truth: torch.Tensor,
    ctrl_pred: torch.Tensor,
    pred_space: str = "raw",
) -> dict[str, float]:
    """DEG overlap and Pearson Δ, the delta half of score_true_gen, the prediction read in ``pred_space``.

    NaN is kept, not raised: undefined DEG overlap is NaN for the neural arms too,
    and the plotter drops non-finite pairs from every arm together.
    """
    bundle = score_pair(truth_raw, pred, ctrl_truth, ctrl_pred, a_space="raw", b_space=pred_space,
                        arm=False, ctrl_det_floors=CTRL_FLOORS)
    flat = materialize_metric_tensors(bundle)
    cols = metric_cols(REPORT_SPACE)
    return {s: float(flat[cols[s]]) for s in DELTA_METRICS}


def _pairs() -> pd.DataFrame:
    if not GENESIS_PARQUET.is_file():
        raise FileNotFoundError(GENESIS_PARQUET)
    df = pd.read_parquet(GENESIS_PARQUET)
    check_report_space(df, str(GENESIS_PARQUET), REPORT_SPACE)
    if df.duplicated(["cell_name", "drug", "dose", "plate_i", "plate_j"]).any():
        raise ValueError("GENESIS parquet has duplicate plate pairs")
    # Its true_true energy is copied into every baseline table: it must be scored on the same cells as ours.
    uncapped = "energy_cells" in df.columns and set(df.energy_cells) == {"all"}
    if uncapped != ENERGY_ALL_CELLS:
        raise ValueError(f"{GENESIS_PARQUET}: energy {'on all cells' if uncapped else 'capped'}, this run "
                         f"{'on all cells' if ENERGY_ALL_CELLS else 'capped'}: --energy-all-cells must match it")
    return df


def check_against_genesis(device: torch.device) -> None:
    """One pair with both sides ≤ the subsample cap must reproduce the parquet.

    Above the cap the stored number is an unseeded draw and cannot be matched.
    A miss here means the projection or the argument order drifted.
    """
    pred_root = CHECK_PRED_ROOT
    df = _pairs()
    ok = df[(df.n_truth_i <= SUBSAMPLE_CAP) & (df.n_gen_j <= SUBSAMPLE_CAP)]
    if ok.empty:
        raise RuntimeError("no pair under the energy subsample cap to check")
    r = ok.iloc[0]
    reader = EvalNpzReader(EVAL_CACHE, cache_size=4)
    dose_s = str(float(r.dose))
    truth_i = to_t(read_perturbed(
        reader, {"cell_name": r.cell_name, "drug": r.drug, "dose": dose_s, "plate": str(r.plate_i)},
    ), device)
    cond = {"cell_name": r.cell_name, "drug": r.drug, "dose": float(r.dose)}
    gen_j = to_t(load_pred(prediction_path(
        pred_root, replicate_design("tahoe"), cond, str(r.plate_j), "npy",
    )), device)
    got = report_energy(gen_j, truth_i, model_output_space="log1p", seed=0)
    # n ≤ cap, so the seed is never consumed. The stored cross_ij is deterministic.
    ref = float(r.cross_ij__energy_distance)
    if abs(got - ref) > 1e-5:
        raise RuntimeError(
            f"energy check failed for {r.cell_name} {r.drug} {r.dose} "
            f"plates {r.plate_i}/{r.plate_j}: got {got:.8f} parquet {ref:.8f}"
        )
    print(f"energy check ok  {got:.6f} vs parquet {ref:.6f}")


def _out(kind: str) -> Path:
    return RESULTS / f"model_vs_batch_gap__{kind}.parquet"


def _partial(kind: str) -> Path:
    return RESULTS / f"model_vs_batch_gap__{kind}.partial.parquet"


def score(device: torch.device, overwrite: bool) -> None:
    df = _pairs()
    need = ["true_true__energy_distance", "true_true__deg_adaptivek_truth",
            "true_true__pearson_r_diff"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"GENESIS parquet missing {missing}")

    pending = list(BASELINES)
    if not overwrite:
        pending = []
        for kind in BASELINES:
            path = _out(kind)
            if path.is_file() and len(pd.read_parquet(path)) == len(df):
                print(f"exists ({len(df)} pairs), skipping {path.name}")
            else:
                pending.append(kind)
    if not pending:
        return

    gens = load_baselines(tuple(pending), root=BASELINE_ROOT)
    for kind, gen in gens.items():
        if list(gen.genes) != GENES:
            raise ValueError(f"{kind}: gene axis is not STATE_HVG")
    spaces = {kind: prediction_space(gen) for kind, gen in gens.items()}
    print(f"prediction spaces {spaces}")

    reader = EvalNpzReader(EVAL_CACHE, cache_size=128)
    libs = (EvalNpzReader(library_namespace(EVAL_CACHE), cache_size=128)
            if any(s != "raw" for s in spaces.values()) else None)   # a fit-space emission needs each cell's L_full
    rows: dict[str, list[dict]] = {k: [] for k in pending}
    done: set[tuple] = set()
    # Resume a killed sweep from the partials rather than rescoring an hour.
    for kind in pending:
        part = _partial(kind)
        if part.is_file() and not overwrite:
            prev = pd.read_parquet(part)
            rows[kind] = prev.to_dict("records")
            print(f"resume {kind}: {len(rows[kind])} pairs")
    if rows[pending[0]]:
        done = {
            (r["cell_name"], r["drug"], float(r["dose"]), r["plate_i"], r["plate_j"])
            for r in rows[pending[0]]
        }
        for kind in pending[1:]:
            keys = {
                (r["cell_name"], r["drug"], float(r["dose"]), r["plate_i"], r["plate_j"])
                for r in rows[kind]
            }
            if keys != done:
                raise RuntimeError(f"{kind} partial keys disagree with {pending[0]}")

    t0 = time.perf_counter()
    n_new = 0
    for r in tqdm(df.itertuples(index=False), total=len(df), desc="linear gap"):
        key = (r.cell_name, r.drug, float(r.dose), str(r.plate_i), str(r.plate_j))
        if key in done:
            continue
        cell, drug, dose, pi, pj = key
        dose_s = str(dose)
        truth_i = to_t(read_perturbed(
            reader, {"cell_name": cell, "drug": drug, "dose": dose_s, "plate": pi},
        ), device)
        truth_j = to_t(read_perturbed(
            reader, {"cell_name": cell, "drug": drug, "dose": dose_s, "plate": pj},
        ), device)
        ctrl_i = read_control(reader, {"cell_name": cell, "plate": pi})
        ctrl_j = read_control(reader, {"cell_name": cell, "plate": pj})
        if truth_i.shape[1] != len(GENES) or truth_j.shape[1] != len(GENES):
            raise ValueError(
                f"gene axis {truth_i.shape[1]}/{truth_j.shape[1]} != {len(GENES)} "
                f"for {cell} {drug} {dose}"
            )
        if ctrl_i.shape[1] != len(GENES) or ctrl_j.shape[1] != len(GENES):
            raise ValueError(f"control gene axis mismatch for {cell} plates {pi}/{pj}")
        ctrl_i_t, ctrl_j_t = to_t(ctrl_i, device), to_t(ctrl_j, device)
        lib_i = control_library(libs, {"cell_name": cell, "plate": pi}, ctrl_i) if libs else None
        lib_j = control_library(libs, {"cell_name": cell, "plate": pj}, ctrl_j) if libs else None

        for kind in pending:
            gen_i = baseline_pred(
                gens[kind], kind, drug=drug, cell=cell, dose=dose, plate=pi,
                ctrl_raw=ctrl_i, n_cells=int(truth_i.shape[0]), ctrl_lib=lib_i,
            )
            gen_j = baseline_pred(
                gens[kind], kind, drug=drug, cell=cell, dose=dose, plate=pj,
                ctrl_raw=ctrl_j, n_cells=int(truth_j.shape[0]), ctrl_lib=lib_j,
            )
            pred_i = to_t(gen_i, device)
            pred_j = to_t(gen_j, device)
            e_ij = report_energy(
                pred_j, truth_i, model_output_space=spaces[kind],
                seed=_seed(kind, cell, drug, dose, pi, pj, "ij"),
            )
            e_ji = report_energy(
                pred_i, truth_j, model_output_space=spaces[kind],
                seed=_seed(kind, cell, drug, dose, pi, pj, "ji"),
            )
            # Cross: truth on plate i vs the emission from plate j's DMSO, and back.
            d_ij = delta_scores(truth_i, pred_j, ctrl_i_t, ctrl_j_t, spaces[kind])
            d_ji = delta_scores(truth_j, pred_i, ctrl_j_t, ctrl_i_t, spaces[kind])
            delta_cols = {}
            for s in DELTA_METRICS:
                delta_cols[f"true_true__{s}"] = float(getattr(r, f"true_true__{s}"))
                delta_cols[f"cross_ij__{s}"] = d_ij[s]
                delta_cols[f"cross_ji__{s}"] = d_ji[s]
                delta_cols[f"cross_mean__{s}"] = 0.5 * (d_ij[s] + d_ji[s])
            rows[kind].append({
                "model": kind,
                "ctrl_mode": "full",
                "report_space": REPORT_SPACE,
                **({"energy_cells": "all"} if ENERGY_ALL_CELLS else {}),
                "cell_name": cell,
                "drug": drug,
                "dose": dose,
                "plate_i": pi,
                "plate_j": pj,
                "n_truth_i": int(truth_i.shape[0]),
                "n_truth_j": int(truth_j.shape[0]),
                "n_gen_i": int(pred_i.shape[0]),
                "n_gen_j": int(pred_j.shape[0]),
                "true_true__energy_distance": float(r.true_true__energy_distance),
                "cross_ij__energy_distance": e_ij,
                "cross_ji__energy_distance": e_ji,
                "cross_mean__energy_distance": 0.5 * (e_ij + e_ji),
                **delta_cols,
            })
        n_new += 1
        if n_new % 25 == 0:
            for kind in pending:
                pd.DataFrame(rows[kind]).to_parquet(_partial(kind), index=False)

    if any(len(rows[k]) != len(df) for k in pending):
        raise RuntimeError(
            "pair count mismatch: " + ", ".join(f"{k}={len(rows[k])}" for k in pending)
        )
    for kind in pending:
        out = _out(kind)
        pd.DataFrame(rows[kind]).to_parquet(out, index=False)
        part = _partial(kind)
        if part.is_file():
            part.unlink()
        print(f"wrote {len(df)} pairs -> {out}")
    print(f"scored {n_new} new pairs in {time.perf_counter() - t0:.1f}s")


def run(
    device: str = "cuda:0",
    overwrite: bool = False,
    check_only: bool = False,
    results: Path | None = None,
    report_space: str = DEFAULT_REPORT_SPACE,
    baseline_root: Path | None = None,
    check_pred_root: Path | None = None,
    energy_all_cells: bool = False,
) -> int:
    """Self-check one pair against the GENESIS parquet, then score every baseline (skipped on --check-only)."""
    global RESULTS, GENESIS_PARQUET, REPORT_SPACE, BASELINE_ROOT, CHECK_PRED_ROOT, ENERGY_ALL_CELLS  # set once per run
    ENERGY_ALL_CELLS = energy_all_cells
    if report_space not in REPORT_SPACES:
        raise ValueError(f"report_space {report_space!r} not in {REPORT_SPACES}")
    results = results or RESULTS  # None = the module default (published slice_med tables)
    BASELINE_ROOT, CHECK_PRED_ROOT = baseline_root, check_pred_root or CHECK_PRED_ROOT
    RESULTS, GENESIS_PARQUET = results, results / "model_vs_batch_gap__genesis.parquet"
    REPORT_SPACE = report_space
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    if dev.type == "cpu":
        raise RuntimeError("energy check and scoring need a GPU; cuda is not available")
    l_ref = set_dataset_median_umi("tahoe")
    print(f"device={dev}  L_ref={l_ref}")
    check_against_genesis(dev)
    if check_only:
        return 0
    score(dev, overwrite)
    return 0
