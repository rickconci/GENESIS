"""Score a model against an experimental replicate: the same condition run twice.

When a dataset measures one condition on two replicates A and B (Tahoe: two plates),
the real gap d(A, B) is the experiment's own reproducibility, and a model is judged
against it.  Four quantities per replicate pair, each scored with the same metric
kernels as the main eval (control pools travel with their replicate):

    true_true   d(A, B)                       the experimental-replicate reference
    same_mean   mean of d(A, A'), d(B, B')    model vs the replicate it was conditioned on
    cross_mean  mean of d(A, B'), d(B, A')    model vs the OTHER replicate -- the headline
    gen_gen     d(A', B')                     the model's own replicate-to-replicate spread

Which column is the replicate, and which columns make a condition, come from the
dataset registry (``core_util_data/datasets/<name>.yaml``), so nothing here names
Tahoe's columns.  A dataset whose ``experimental_replicate_column`` is null has no
such reference, and asking for one raises.

Non-DEG metrics are read in one report space: ``log1p_slice_med_rgs`` (the default, and the
published Figure 2a/2b/3a tables) or ``log1p_rgs`` (Figure 2c's space; the Appendix
plate-pair bundle). Tables carry a ``report_space`` column so one is never read as the
other. A pseudobulk-only arm (an MSE model: one log1p vector per condition, an estimate of
the ``log1p_rgs`` pseudobulk) scores through ``score_true_pseudobulk``, in log1p_rgs only.

Same-replicate comparisons keep the whole control pool (passed twice, cloned, so the
DEG bundle's ``ctrl_a is ctrl_b`` guard holds without dropping cells). The published
tables are this mode; halving the pool is not offered.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from src.data.vocab import to_categorical_str
from src.data.data_utils import safe_filename_part
from src.eval.cache.eval_npz_reader import EvalNpzReader
from src.eval.reference.replicate import (
    MIN_CELLS_PER_PLATE, ReplicateDesign, drop_seen_in_training, drop_small_plates,
    enumerate_pairs, replicate_design, trained_combinations, _replicate_label,
)
from src.eval.reference.truth_io import read_control, read_perturbed
from src.eval.metrics.delta import compute_delta_metrics
from src.eval.metrics.eval_metrics import score_pair
from src.eval.metrics.reporting import materialize_metric_tensors
from src.eval.metrics.spaces import set_dataset_median_umi
from src.eval.orchestration.model_eval import _stable_cond_seed

__all__ = [
    "CTRL_FLOORS", "DEFAULT_REPORT_SPACE", "METRIC_COLS", "MODEL_SPACE", "PSEUDOBULK_SHORT",
    "REPORT_SPACES", "SHORT", "EvalNpzReader", "metric_cols",
    "ReplicateDesign", "check_report_space", "control_keys",
    "MIN_CELLS_PER_PLATE", "drop_seen_in_training", "drop_small_plates", "enumerate_pairs",
    "load_copy", "load_neighbors", "load_pred", "load_pseudobulk", "materialize_metric_tensors",
    "neighbour_context",
    "perturbed_keys", "prediction_path", "read_control", "read_perturbed", "replicate_design",
    "resolve_ctrl_pair", "score_gen_gen", "score_raw_pair", "score_replicate_pair", "score_true_gen",
    "score_true_pseudobulk", "score_true_true_cross", "set_dataset_median_umi", "to_t",
    "trained_combinations",
]

MODEL_SPACE = "log1p"
CTRL_FLOORS = (0.10,)
# Headline keys. DEG lives on the count lattice; the rest in the chosen report space.
REPORT_SPACES = ("log1p_slice_med_rgs", "log1p_rgs")
DEFAULT_REPORT_SPACE = "log1p_slice_med_rgs"  # the published plate-pair tables


def metric_cols(space: str = DEFAULT_REPORT_SPACE) -> dict[str, str]:
    """Short key -> eval-bundle key, with every non-DEG metric read in ``space``."""
    if space not in REPORT_SPACES:
        raise ValueError(f"unknown report space {space!r}; expected one of {REPORT_SPACES}")
    deg = "deg_raw_int_rgs/deg_overlap_top_p_ranked_k100_fdr0p001_lfc0p1"
    return {"deg_adaptivek_truth": f"{deg}_adaptivek_truth_detfloor10",
            **{m: f"{space}/{m}" for m in ("pearson_r", "pearson_r_diff", "energy_distance", "mmd_rbf")}}


METRIC_COLS = metric_cols()
PSEUDOBULK_SHORT = ("pearson_r", "pearson_r_diff")  # all a cell-less arm can be scored on
SHORT = list(METRIC_COLS)
# ASCII-only, as the competitor NPZ writers named their files.  It drops non-ASCII drug
# names (e.g. γ-Oryzanol) from npz arms: a coverage gap, kept so scores match those files.
_NPZ_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


# 1. Where predictions live, and how truth and controls are read
def prediction_path(root: Path, d: ReplicateDesign, cond: Mapping[str, Any], rep: str, fmt: str) -> Path:
    """``npy``: the eval pipeline's ``k=v__...`` naming. ``npz``: the competitor writers' naming."""
    vals = {**cond, d.replicate_col: _replicate_label(d, rep)}
    if fmt == "npy":
        return root / ("__".join(f"{k}={safe_filename_part(to_categorical_str(vals[k]))}"
                                 for k in d.file_cols) + ".npy")
    if fmt == "npz":
        def part(k: str) -> str:
            v = vals[k]
            if k == d.replicate_col:
                return v
            return f"{float(v):.6f}" if isinstance(v, (float, np.floating)) else _NPZ_SAFE.sub("_", str(v)).strip("_")
        return root / ("__".join(part(k) for k in d.file_cols) + ".npz")
    raise ValueError(f"unknown prediction format {fmt!r}")


def perturbed_keys(d: ReplicateDesign, cond: Mapping[str, Any], rep: str) -> dict[str, str]:
    return {**{c: str(cond[c]) for c in d.condition_cols}, d.replicate_col: str(rep)}


def control_keys(d: ReplicateDesign, cond: Mapping[str, Any], rep: str) -> dict[str, str]:
    full = perturbed_keys(d, cond, rep)
    return {c: full[c] for c in d.control_cols}


def load_pred(path: Path) -> np.ndarray:
    """Load predictions and assert log1p scale.

    STATE / CellFlow / GENESIS label ``log1p``.  PerturbDiff NPZs can claim
    ``normalization=raw_counts`` while holding log1p(CPM), so the label is not trusted;
    raw counts on HVGs routinely exceed ~50, log1p sits in ~0-15.
    """
    if path.suffix == ".npy":
        arr = np.load(path)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as z:
            if "predictions" not in z.files:
                raise KeyError(f"{path}: missing 'predictions'; keys={z.files}")
            arr = z["predictions"]
    else:
        raise ValueError(f"unsupported prediction suffix: {path}")
    arr = np.asarray(arr, dtype=np.float32)
    if float(np.nanmax(arr)) > 30.0:
        raise ValueError(f"{path}: max={float(np.nanmax(arr)):.2f} looks like raw counts, "
                         f"but the scorer expects {MODEL_SPACE}")
    return arr


def load_pseudobulk(path: Path) -> np.ndarray:
    """A pseudobulk-only prediction: one ``(1, G)`` log1p vector, returned as ``(G,)``."""
    arr = load_pred(path)
    if arr.ndim != 2 or arr.shape[0] != 1:
        raise ValueError(f"{path}: a pseudobulk file holds one row; got shape {arr.shape}")
    return arr[0]


def check_report_space(df: pd.DataFrame, name: str, expected: str) -> None:
    """Refuse a pair table not scored in ``expected``.

    A table without the column is unlabelled (and slice_med): refused too, rather than assumed.
    """
    if "report_space" not in df.columns:
        raise ValueError(f"{name}: no report_space column; an unlabelled table is refused, never assumed")
    if spaces := sorted(set(df["report_space"]) - {expected}):
        raise ValueError(f"{name}: report_space {spaces}, expected {expected!r}")


def to_t(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(arr, dtype=np.float32), device=device)


def load_copy(reader: EvalNpzReader, keys: dict[str, str], n_cells: int, col: str = "plate") -> np.ndarray:
    """A neighbour line's measured cells as the prediction, ``n_cells`` rows.

    ``keys`` already name the neighbour.  Same draw as the model-comparison copy
    (seed 0, resampled with replacement when the neighbour has fewer cells).  Raises
    FileNotFoundError when the neighbour was never measured on that replicate.
    """
    cells = np.asarray(read_perturbed(reader, keys, col, max_cells=n_cells, seed=0), dtype=np.float32)
    if cells.shape[0] == 0:
        raise ValueError(f"empty copy cloud for {keys}")
    if cells.shape[0] < n_cells:
        cells = cells[np.random.default_rng(0).choice(cells.shape[0], size=n_cells, replace=True)]
    return cells


# 2. The control pools for each comparison
def resolve_ctrl_pair(ctrl_for_a: torch.Tensor, ctrl_for_b: torch.Tensor, *, mode: str,
                      seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(ctrl_a, ctrl_b) for the DEG bundle. ``full`` keeps every control cell.

    ``seed`` is unused: the published comparisons draw no control split. It is accepted so that
    a call asking for another mode fails on the mode, not on its arguments.
    """
    del seed
    if mode != "full":
        raise ValueError(f"unknown ctrl mode {mode!r}; the published comparisons use 'full'")
    if ctrl_for_a is not ctrl_for_b:  # cross-replicate: two different pools already
        return ctrl_for_a, ctrl_for_b
    return ctrl_for_a, ctrl_for_a.clone()


# 3. The scorers: truth vs a prediction, truth vs truth, prediction vs prediction
def score_true_gen(truth: torch.Tensor, pred: torch.Tensor, ctrl_a: torch.Tensor,
                   ctrl_b: torch.Tensor, pred_space: str,
                   report_space: str = DEFAULT_REPORT_SPACE,
                   det_ctrl_b: torch.Tensor | None = None, energy_all_cells: bool = False) -> dict[str, float]:
    """Truth (raw) vs a prediction in ``pred_space`` (log1p for models, raw for copy).
    ``det_ctrl_b``: the detection floor's control (a copy scored against its own controls)."""
    flat = materialize_metric_tensors(score_pair(
        truth, pred, ctrl_a, ctrl_b, a_space="raw", b_space=pred_space,
        det_ctrl_b=det_ctrl_b, ctrl_det_floors=CTRL_FLOORS, energy_all_cells=energy_all_cells))
    cols = metric_cols(report_space)
    return {s: float(flat[cols[s]]) for s in SHORT}


def score_true_pseudobulk(truth: torch.Tensor, vec: torch.Tensor, ctrl_a: torch.Tensor,
                          ctrl_b: torch.Tensor) -> dict[str, float]:
    """Truth (raw) vs a predicted ``log1p_rgs`` pseudobulk ``vec`` (G,): what an MSE model emits.

    The vector is that pseudobulk directly -- never snapped, since RGS on a mean zeroes every
    gene below log1p 0.35. Argument order and kernels are ``score_true_gen``'s (ctrl_a goes
    with truth, ctrl_b with the prediction), so a vector equal to the mean of log1p integer
    cells scores exactly as those cells. DEG, energy and MMD need cells: NaN.
    """
    if vec.ndim != 1 or vec.shape[0] != truth.shape[1]:
        raise ValueError(f"pseudobulk must be ({truth.shape[1]},); got {tuple(vec.shape)}")
    if not bool(torch.isfinite(vec).all()):
        raise ValueError("pseudobulk has non-finite entries")
    if float(vec.max()) > 30.0:
        raise ValueError(f"pseudobulk max={float(vec.max()):.2f} looks like raw counts, not log1p")
    t, ca, cb = (torch.log1p(x.to(torch.float32)) for x in (truth, ctrl_a, ctrl_b))  # log1p_rgs
    pred = vec.to(t)[None]
    out = {s: float("nan") for s in SHORT}
    out["pearson_r_diff"] = float(compute_delta_metrics(pred, t, cb, ca)["pearson_r_diff"])
    out["pearson_r"] = float(torch.corrcoef(torch.stack([pred.mean(0), t.mean(0)]))[0, 1])
    return out


def score_raw_pair(a: torch.Tensor, b: torch.Tensor, ctrl_a: torch.Tensor, ctrl_b: torch.Tensor, *,
                   dist: bool = True, energy_all_cells: bool = False) -> dict[str, float]:
    """Every metric of two measured (raw) cell sets, each against its own control pool.

    ``dist=False`` skips the distribution bundle (energy, MMD, pseudobulk arm metrics) when
    only delta and DEG are wanted, e.g. one perturbed set against two control wells.
    """
    out = score_pair(a, b, ctrl_a, ctrl_b, a_space="raw", b_space="raw", arm=dist, ctrl_det_floors=CTRL_FLOORS,
                     energy_all_cells=energy_all_cells)
    return {k: float(v) for k, v in materialize_metric_tensors(out).items()}


def score_true_true_cross(truth_i: torch.Tensor, truth_j: torch.Tensor, ctrl_i: torch.Tensor,
                          ctrl_j: torch.Tensor, report_space: str = DEFAULT_REPORT_SPACE,
                          energy_all_cells: bool = False) -> dict[str, float]:
    """The experimental-replicate reference d(A, B): both arms raw, each with its own controls."""
    flat = score_raw_pair(truth_i, truth_j, ctrl_i, ctrl_j, energy_all_cells=energy_all_cells)
    cols = metric_cols(report_space)
    return {s: flat[cols[s]] for s in SHORT}


def score_gen_gen(gen_a: torch.Tensor, gen_b: torch.Tensor, ctrl_a: torch.Tensor,
                  ctrl_b: torch.Tensor, pred_space: str,
                  report_space: str = DEFAULT_REPORT_SPACE, energy_all_cells: bool = False) -> dict[str, float]:
    """d(A', B'): both arms are predictions, snapped from ``pred_space``."""
    flat = materialize_metric_tensors(score_pair(
        gen_a, gen_b, ctrl_a, ctrl_b, a_space=pred_space, b_space=pred_space, ctrl_det_floors=CTRL_FLOORS,
        energy_all_cells=energy_all_cells))
    cols = metric_cols(report_space)
    return {s: float(flat[cols[s]]) for s in SHORT}


# 4. One replicate pair -> one row of the pair table
def _key(v: Any) -> Any:
    """Condition value as a lookup key: numbers rounded (doses come back as float32 noise)."""
    try:
        return round(float(v), 6)
    except (TypeError, ValueError):
        return str(v)


# The copy arm stands a measured line in for the test one. On the bio split one line stands in
# for each test line; on the pert split the pick is per condition and plate, because each held-out
# line x compound copies the line nearest to it for that compound (score.py copy_neighbor).
def load_neighbors(path: Path | None, d: ReplicateDesign) -> dict[tuple, str]:
    """Copy arm: (context,) or (*condition, replicate) -> the context whose cells stand in.

    A table with a ``neighbor`` column is per context; one with ``neighbor_<context>`` plus
    the condition and replicate columns (score.py copy_neighbor's neighbors.tsv) is per condition.
    """
    if path is None:
        raise SystemExit("--fmt copy needs --neighbor-tsv")
    t, ctx = pd.read_csv(path, sep="\t"), d.context_col
    if "neighbor" in t.columns:
        nb = dict(zip(t[ctx].astype(str), t["neighbor"].astype(str)))
        if any(n in nb for n in nb.values()):
            raise ValueError(f"a copy neighbour is itself a test line: {nb}")
        return {(k,): v for k, v in nb.items()}
    col = f"neighbor_{ctx}"
    missing = [c for c in (*d.condition_cols, d.replicate_col, col) if c not in t.columns]
    if missing:
        raise KeyError(f"{path}: no 'neighbor' column and missing {missing}")
    if (t[col].astype(str) == t[ctx].astype(str)).any():
        raise ValueError(f"{path}: a condition copies its own {ctx}")
    keys = [(*(_key(r[c]) for c in d.condition_cols), _replicate_label(d, str(r[d.replicate_col])))
            for r in t.to_dict("records")]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path}: a condition and replicate has two neighbours")
    return dict(zip(keys, t[col].astype(str)))


def neighbour_context(neighbors: Mapping[tuple, str], d: ReplicateDesign,
                      cond: Mapping[str, Any], rep: str) -> str:
    """The context copied for ``cond`` on replicate ``rep``; raises if the table has none."""
    full = (*(_key(cond[c]) for c in d.condition_cols), _replicate_label(d, rep))
    for k in (full, (str(cond[d.context_col]),)):
        if k in neighbors:
            return neighbors[k]
    raise KeyError(f"no copy neighbour for {cond} on {rep}")


def load_reference(path: Path | None, d: ReplicateDesign, space: str) -> pd.DataFrame | None:
    """Precomputed true_true__* on the same gene set, keyed by pair; None = compute each."""
    if path is None:
        return None
    ref = pd.read_parquet(path)
    check_report_space(ref, str(path), space)
    # a missing column fails up front: it must never read as "pair not found"
    if missing := [f"true_true__{s}" for s in SHORT if f"true_true__{s}" not in ref.columns]:
        raise KeyError(f"{path} lacks {missing}: not a reference in the current metric namespace")
    return ref.set_index([*d.condition_cols, f"{d.replicate_col}_i", f"{d.replicate_col}_j"])


def _reference_row(ref: pd.DataFrame | None, cond: Mapping[str, Any], i: str, j: str) -> dict[str, float] | None:
    if ref is None or (key := (*cond.values(), i, j)) not in ref.index:
        return None
    row = ref.loc[key]
    row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
    return {s: float(row[f"true_true__{s}"]) for s in SHORT}


def score_replicate_pair(d: ReplicateDesign, reader: EvalNpzReader, cond: Mapping[str, Any], i: str, j: str, *,
                         model: str, fmt: str, pred_dir: Path | None, report_space: str, ctrl_mode: str,
                         neighbors: Mapping[tuple, str], ref: pd.DataFrame | None,
                         device: torch.device, copy_ctrl: str = "heldout",
                         energy_all_cells: bool = False) -> dict[str, Any] | None:
    """One row: true_true, same_*, cross_* and gen_gen for replicates i, j of ``cond``.

    ``fmt``: npy / npz (a model's cells), copy (a neighbour context's measured cells),
    pseudobulk (one log1p vector: Pearson r and Pearson delta only). None when a copy
    neighbour was not measured on both replicates.

    ``copy_ctrl`` (fmt copy only): "heldout" (published) pairs the copied cells with the held-out
    context's controls, as every model; "own" with the neighbour's controls on that replicate,
    which drops the basal offset log2(nb_ctrl / heldout_ctrl) from the copy's Δ and DEGs
    (CopyNeighborPredGetter has the split-half twin and the numbers). Energy ignores controls.
    """
    if copy_ctrl not in ("heldout", "own"):
        raise ValueError(f"unknown copy_ctrl {copy_ctrl!r}; expected 'heldout' or 'own'")
    if copy_ctrl == "own" and fmt != "copy":
        raise ValueError(f"copy_ctrl='own' is for --fmt copy; got fmt={fmt!r}")
    rep = d.replicate_col
    truth = {r: to_t(read_perturbed(reader, perturbed_keys(d, cond, r), rep), device) for r in (i, j)}
    ctrl = {r: to_t(read_control(reader, control_keys(d, cond, r), rep), device) for r in (i, j)}
    pred_ctrl = ctrl  # the controls each prediction is scored against
    if fmt == "copy":
        nb = {r: neighbour_context(neighbors, d, cond, r) for r in (i, j)}
        try:
            gen = {r: to_t(load_copy(reader, perturbed_keys(d, {**cond, d.context_col: nb[r]}, r),
                                     truth[r].shape[0], rep), device) for r in (i, j)}
        except FileNotFoundError:
            print(f"\nSKIP {cond} {rep}s {i}/{j}: neighbour not measured on both")
            return None
        if copy_ctrl == "own":  # outside the try: copied cells without their controls must raise
            pred_ctrl = {r: to_t(read_control(reader, control_keys(d, {**cond, d.context_col: nb[r]}, r), rep),
                                 device) for r in (i, j)}
    elif fmt == "pseudobulk":
        gen = {r: to_t(load_pseudobulk(prediction_path(pred_dir, d, cond, r, "npy")), device) for r in (i, j)}
    else:
        gen = {r: to_t(load_pred(prediction_path(pred_dir, d, cond, r, fmt)), device) for r in (i, j)}
    return score_loaded_pair(d, cond, i, j, truth, ctrl, gen, pred_ctrl, model=model, fmt=fmt,
                             report_space=report_space, ctrl_mode=ctrl_mode, ref=ref, copy_ctrl=copy_ctrl,
                             energy_all_cells=energy_all_cells)


def score_loaded_pair(d: ReplicateDesign, cond: Mapping[str, Any], i: str, j: str,
                      truth: Mapping[str, torch.Tensor], ctrl: Mapping[str, torch.Tensor],
                      gen: Mapping[str, torch.Tensor], pred_ctrl: Mapping[str, torch.Tensor] | None = None, *,
                      model: str, fmt: str, report_space: str, ref: pd.DataFrame | None,
                      ctrl_mode: str = "full", copy_ctrl: str = "heldout",
                      energy_all_cells: bool = False) -> dict[str, Any]:
    """The pair-table row from cells already on the device; ``score_replicate_pair`` is load + this.

    ``pred_ctrl`` (default ``ctrl``): the controls each prediction is scored against. ``gen`` holds
    each replicate's cloud in the model's space, ``raw`` for a copy, or a log1p vector (pseudobulk).
    ``energy_all_cells``: energy distance on every cell of a replicate (each set is a whole plate's
    condition, so the 2,048-cell cap binds far more often than on split halves); MMD keeps its cap.
    """
    if energy_all_cells and ref is not None:
        raise ValueError("--reference rows carry capped true_true energy; score all-cells energy without one")
    rep = d.replicate_col
    pred_ctrl = ctrl if pred_ctrl is None else pred_ctrl
    for r in (i, j):
        if gen[r].shape[-1] != truth[r].shape[1]:
            raise ValueError(f"gene axis mismatch for {cond} {rep} {r}: pred {tuple(gen[r].shape)} "
                             f"truth {tuple(truth[r].shape)}")
    space = "raw" if fmt == "copy" else MODEL_SPACE

    def seed(comparison: str) -> int:
        """Energy and MMD subsample large cell sets through torch's global RNG: seed it per
        (condition, pair, comparison), so a pair scores the same in any run and any subset."""
        return _stable_cond_seed(0, {**cond, f"{rep}_i": i, f"{rep}_j": j, "comparison": comparison})

    def vs(name: str, t: torch.Tensor, g: torch.Tensor, c_a: torch.Tensor, c_b: torch.Tensor,
           det_r: str) -> dict[str, float]:
        """``det_r``: the replicate whose held-out controls stand in for c_b in the DEG floor when the copy
        brings its own, so truth's DEGs (and which pairs are defined) are every arm's."""
        torch.manual_seed(seed(name))
        if fmt == "pseudobulk":
            return score_true_pseudobulk(t, g, c_a, c_b)
        return score_true_gen(t, g, c_a, c_b, space, report_space,
                              det_ctrl_b=ctrl[det_r] if copy_ctrl == "own" else None,
                              energy_all_cells=energy_all_cells)

    same = {}
    for r in (i, j):
        c_a, c_b = resolve_ctrl_pair(ctrl[r], pred_ctrl[r], mode=ctrl_mode, seed=seed(f"same_ctrl_{r}"))
        same[r] = vs(f"same_{r}", truth[r], gen[r], c_a, c_b, r)
    cross_ij = vs("cross_ij", truth[i], gen[j], ctrl[i], pred_ctrl[j], j)  # controls travel with their replicate
    cross_ji = vs("cross_ji", truth[j], gen[i], ctrl[j], pred_ctrl[i], i)
    if fmt == "pseudobulk":
        gen_gen = {s: float("nan") for s in SHORT}  # no cells, no spread
    else:
        torch.manual_seed(seed("gen_gen"))
        gen_gen = score_gen_gen(gen[i], gen[j], pred_ctrl[i], pred_ctrl[j], space, report_space,
                                energy_all_cells=energy_all_cells)
    tt = _reference_row(ref, cond, i, j)
    if tt is None:
        torch.manual_seed(seed("true_true"))
        tt = score_true_true_cross(truth[i], truth[j], ctrl[i], ctrl[j], report_space,
                                   energy_all_cells=energy_all_cells)
    n_gen = {r: 0 if fmt == "pseudobulk" else int(gen[r].shape[0]) for r in (i, j)}
    # only the opt-ins are labelled, so a default rerun writes the published schema byte for byte
    row = {"model": model, "ctrl_mode": ctrl_mode, **({"copy_ctrl": "own"} if copy_ctrl == "own" else {}),
           **({"energy_cells": "all"} if energy_all_cells else {}),
           "report_space": report_space, **cond,
           f"{rep}_i": i, f"{rep}_j": j, "n_truth_i": int(truth[i].shape[0]),
           "n_truth_j": int(truth[j].shape[0]), "n_gen_i": n_gen[i], "n_gen_j": n_gen[j]}
    for s in SHORT:
        row.update({f"true_true__{s}": tt[s], f"same_i__{s}": same[i][s], f"same_j__{s}": same[j][s],
                    f"same_mean__{s}": 0.5 * (same[i][s] + same[j][s]),
                    f"cross_ij__{s}": cross_ij[s], f"cross_ji__{s}": cross_ji[s],
                    f"cross_mean__{s}": 0.5 * (cross_ij[s] + cross_ji[s]), f"gen_gen__{s}": gen_gen[s]})
    return row
