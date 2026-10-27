"""Score one comparison: snap both arms to counts, project, compose the bundles.

:func:`score_pair` is the kernel every design calls. It composes two bundles:

* :func:`compute_arm_dist_bundle` — arm + distributional metrics
  (:mod:`.arm`), no controls.
* :func:`compute_delta_deg_bundle` — delta (:mod:`.delta`) and DEG overlap
  (:mod:`.deg`) metrics, given *disjoint* paired controls.

Both take ``metrics``, a key set from ``configs/eval/metrics.yaml``
(:func:`.reporting.load_metric_set`; ``None`` = every key), and skip the groups
nobody asked for. Report spaces live in :mod:`.spaces`; metric sets, JSON
grouping and the host transfer in :mod:`.reporting`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Union

import torch

if TYPE_CHECKING:
    from ...model.inference_output import ModelOutputSpace

from ..orchestration.log_utils import (
    log_detail,
    log_metric_bundle,
    shapes_line,
)
from .arm import compute_arm_metrics
from .deg import compute_deg_overlap_metrics
from .delta import _spearman_correlation_1d, compute_delta_metrics
from .reporting import _DEG_HEADLINE, _PSEUDOBULK_HEADLINE
from .spaces import (
    DEG_SPACES,
    PSEUDOBULK_DELTA_METRIC_SPACES,
    PSEUDOBULK_METRIC_SPACES,
    _project_raw_to_spaces,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Score a comparison: snap both arms to counts, then two bundles
# ---------------------------------------------------------------------------


@torch.no_grad()
def score_pair(
    a: torch.Tensor,
    b: torch.Tensor,
    ctrl_a: torch.Tensor,
    ctrl_b: torch.Tensor,
    *,
    a_space: Union[str, "ModelOutputSpace"],
    b_space: Union[str, "ModelOutputSpace"],
    det_ctrl_b: Optional[torch.Tensor] = None,
    metrics: Optional[frozenset[str]] = None,
    ctrl_det_floors: tuple[float, ...] = (0.10,),
    arm: bool = True,
    energy_all_cells: bool = False,
) -> dict[str, torch.Tensor]:
    """Arm, delta and DEG for one comparison, each arm naming the space it lives in.

    Both arms are snapped onto integer counts (:func:`to_raw_integer`: ``"raw"``
    rounds, ``"log1p"`` takes the log1p-nearest count), so every prediction is
    scored on truth's lattice. Measured counts are fixed points of the snap.
    Controls are measured counts and pass through. ``a`` is the reference: truth
    when one arm is truth. ``energy_all_cells``: see :func:`compute_arm_metrics`.
    """
    from ...model.inference_output import to_raw_integer

    a, b = to_raw_integer(a, a_space), to_raw_integer(b, b_space)
    out: dict[str, torch.Tensor] = {}
    if arm:  # distributional metrics draw the global RNG; delta and DEG do not
        out.update(compute_arm_dist_bundle(a, b, metrics=metrics, energy_all_cells=energy_all_cells))
    out.update(compute_delta_deg_bundle(
        a, b, ctrl_a, ctrl_b, ctrl_det_floors=ctrl_det_floors, det_ctrl_b=det_ctrl_b, metrics=metrics))
    return out


@torch.no_grad()
def compute_arm_dist_bundle(
    arm_a: torch.Tensor,
    arm_b: torch.Tensor,
    *,
    metrics: Optional[frozenset[str]] = None,
    energy_all_cells: bool = False,
) -> dict[str, torch.Tensor]:
    """Arm + distributional metrics of two count matrices, keyed ``<space>/<metric>``.

    No control is involved, so there is no shared-baseline bias to guard against.
    ``metrics``: flat keys to return (``None`` = all). Every space still runs, so
    skipped metrics replay their RNG draws in order.
    """
    arm_a_spaces = _project_raw_to_spaces(arm_a)
    arm_b_spaces = _project_raw_to_spaces(arm_b)
    log_detail(logger, "arm_dist", "%s", shapes_line(arm_a=arm_a, arm_b=arm_b))

    out: dict[str, torch.Tensor] = {}
    for space in PSEUDOBULK_METRIC_SPACES:
        # (group_a, group_b) = (B, A): the arm metrics take group_b as the reference
        # distribution (its variance is the R² denominator), and A is truth.
        names = None if metrics is None else frozenset(
            k.split("/", 1)[1] for k in metrics if k.startswith(f"{space}/"))
        bundle = compute_arm_metrics(arm_b_spaces[space], arm_a_spaces[space], names=names,
                                     energy_all_cells=energy_all_cells)
        log_metric_bundle(logger, space, bundle, prefer=_PSEUDOBULK_HEADLINE)
        out.update({f"{space}/{k}": v for k, v in bundle.items()})
    # Per-gene detection profile P(count > 0), on the same K-rep slice as MMD / energy.
    if metrics is None or "raw_int_rgs/gene_det_rate_spearman" in metrics:
        out["raw_int_rgs/gene_det_rate_spearman"] = compute_gene_detection_rate_spearman(
            truth_raw_int=arm_a_spaces["raw_int_rgs"],
            pred_raw_int=arm_b_spaces["raw_int_rgs"],
        )
    return out


@torch.no_grad()
def compute_gene_detection_rate_spearman(
    *,
    truth_raw_int: torch.Tensor,
    pred_raw_int: torch.Tensor,
) -> torch.Tensor:
    """Spearman ρ of per-gene detection rates on ``raw_int_rgs`` matrices.

    For each gene, ``det_rate = mean_cells(count > 0)``. Cells need not be
    paired (any-to-any generative eval): only the gene-level detection
    *profile* is compared. Constant rate vectors → NaN (same contract as
    :func:`_spearman_correlation_1d` / ``spearman_r_diff``), so one
    degenerate condition does not abort a multi-thousand-condition sweep.

    Args:
        truth_raw_int: ``(n_truth, n_genes)`` integer counts (rgs truth).
        pred_raw_int: ``(n_pred, n_genes)`` integer counts (rgs preds).
            Same gene axis / device as ``truth_raw_int`` (eval-cache contract).

    Returns:
        0-d float32 tensor on ``truth_raw_int.device``.

    Raises:
        ValueError: Rank/shape/device errors (empty cells, gene-width or device mismatch).
    """
    if truth_raw_int.ndim != 2 or pred_raw_int.ndim != 2:
        raise ValueError(
            f"gene_det_rate_spearman expects 2D (n_cells, n_genes); got "
            f"truth={tuple(truth_raw_int.shape)} pred={tuple(pred_raw_int.shape)}"
        )
    n_truth, n_genes_t = truth_raw_int.shape
    n_pred, n_genes_p = pred_raw_int.shape
    if n_truth == 0 or n_pred == 0:
        raise ValueError(
            f"gene_det_rate_spearman needs ≥1 cell per arm; got "
            f"n_truth={int(n_truth)} n_pred={int(n_pred)}"
        )
    if n_genes_t != n_genes_p:
        raise ValueError(
            f"gene_det_rate_spearman n_genes mismatch: "
            f"truth={int(n_genes_t)} pred={int(n_genes_p)}"
        )
    if truth_raw_int.device != pred_raw_int.device:
        raise ValueError(
            f"gene_det_rate_spearman device mismatch: "
            f"truth={truth_raw_int.device} pred={pred_raw_int.device}"
        )
    return _spearman_correlation_1d(
        (truth_raw_int > 0).mean(dim=0, dtype=torch.float64),
        (pred_raw_int > 0).mean(dim=0, dtype=torch.float64),
    ).float()


@torch.no_grad()
def compute_delta_deg_bundle(
    delta_a: torch.Tensor,
    delta_b: torch.Tensor,
    ctrl_a: torch.Tensor,
    ctrl_b: torch.Tensor,
    *,
    ctrl_det_floors: tuple[float, ...] = (0.10,),
    det_ctrl_b: Optional[torch.Tensor] = None,
    metrics: Optional[frozenset[str]] = None,
) -> dict[str, torch.Tensor]:
    """Delta + DEG metrics of two count matrices, each against its own control.

    Keys are ``<space>/<metric>`` for delta and ``deg_<space>/<metric>`` for DEG
    overlap. ``ctrl_a`` and ``ctrl_b`` MUST be disjoint cells: a shared baseline
    collapses the ``r2_diff`` denominator and makes unrelated arms agree, so
    passing one tensor twice raises.

    Args:
        delta_a: ``(n_a, n_genes)`` counts paired with ``ctrl_a``.
        delta_b: ``(n_b, n_genes)`` counts paired with ``ctrl_b``.
        ctrl_a / ctrl_b: raw integer counts; must be disjoint.
        ctrl_det_floors: Detection-fraction thresholds forwarded to
            :func:`compute_deg_overlap_metrics` (default ``(0.10,)``).
        det_ctrl_b: Stands in for ``ctrl_b`` in the detection floor only (default ``ctrl_b``).
            A prediction scored against its own controls (copy arms) passes the truth context's
            ``ctrl_b`` here: the floor picks the genes truth's DEGs are called on, and truth alone
            decides those (and so which conditions are defined), never the arm's controls.
        metrics: Flat keys to return (``None`` = all). Wilcoxon runs only when a
            ``deg_*`` key is asked for; nothing here draws random numbers.

    Raises:
        ValueError: When ``ctrl_a is ctrl_b``, or ``det_ctrl_b`` has another gene axis.
    """
    if det_ctrl_b is not None and det_ctrl_b.shape[1:] != ctrl_a.shape[1:]:
        raise ValueError(f"compute_delta_deg_bundle: det_ctrl_b genes {tuple(det_ctrl_b.shape)} "
                         f"do not match ctrl_a {tuple(ctrl_a.shape)}")
    if ctrl_a is ctrl_b:
        raise ValueError(
            "compute_delta_deg_bundle: ctrl_a and ctrl_b must be disjoint "
            "cells (e.g. two halves of a random permutation of one pool). "
            "A shared baseline biases r2_diff and every DEG overlap metric.",
        )

    delta_a_spaces, delta_b_spaces, ctrl_a_spaces, ctrl_b_spaces = (
        _project_raw_to_spaces(x) for x in (delta_a, delta_b, ctrl_a, ctrl_b))
    log_detail(logger, "delta_deg", "%s",
               shapes_line(delta_a=delta_a, delta_b=delta_b, ctrl_a=ctrl_a, ctrl_b=ctrl_b))

    out: dict[str, torch.Tensor] = {}
    for space in PSEUDOBULK_DELTA_METRIC_SPACES:
        if metrics is not None and not any(k.startswith(f"{space}/") for k in metrics):
            continue
        delta = compute_delta_metrics(
            delta_b_spaces[space], delta_a_spaces[space],
            ctrl_b_spaces[space],
            ctrl_a_spaces[space],
        )
        log_metric_bundle(logger, space, delta, prefer=_PSEUDOBULK_HEADLINE)
        out.update({f"{space}/{k}": v for k, v in delta.items()})

    # Pooled control detection fraction over both halves — S-free, rep-stable.
    det_b = ctrl_b if det_ctrl_b is None else det_ctrl_b
    n_a = float(ctrl_a.shape[0])
    n_b = float(det_b.shape[0])
    ctrl_det_frac = (
        (ctrl_a > 0).to(torch.float64).sum(dim=0)
        + (det_b > 0).to(torch.float64).sum(dim=0)
    ) / max(n_a + n_b, 1.0)

    for space in DEG_SPACES:
        asked = None if metrics is None else [k for k in metrics if k.startswith(f"deg_{space}/")]
        if asked == []:
            continue
        deg = compute_deg_overlap_metrics(
            ctrl_a_spaces[space], delta_a_spaces[space],
            ctrl_b_spaces[space], delta_b_spaces[space],
            ctrl_det_floors=ctrl_det_floors,
            ctrl_det_frac=ctrl_det_frac,
            include_non_topk=asked is None or any("/deg_overlap_top_" not in k for k in asked),
        )
        log_metric_bundle(logger, f"deg_{space}", deg, prefer=_DEG_HEADLINE)
        out.update({f"deg_{space}/{k}": v for k, v in deg.items()})
    return out if metrics is None else {k: v for k, v in out.items() if k in metrics}
