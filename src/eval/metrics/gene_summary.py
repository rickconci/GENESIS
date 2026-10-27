"""Per-gene sidecars for plots: mean / variance and exact-zero (drop-out) fractions.

Not metrics: nothing here is scored or averaged over reps. The arrays go to
the gene-array parquet sidecars (:mod:`src.eval.plots.gene_arrays`), beside
:func:`src.eval.metrics.deg.compute_deg_detail`'s per-gene Wilcoxon arrays.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch

if TYPE_CHECKING:
    from ...model.inference_output import ModelOutputSpace

from ..orchestration.log_utils import log_detail
from .spaces import DEG_SPACES, _project_predictions_to_spaces, _project_raw_to_spaces

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-gene mean / variance in every report space
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_gene_summary(
    truth_raw: torch.Tensor,
    pred_raw_integer: torch.Tensor,
    *,
    control_raw: Optional[torch.Tensor] = None,
    pred_label: str = "pred",
) -> dict[str, dict[str, list[float]]]:
    """Per-gene mean / variance (ddof=0) of every arm in each ``*_rgs`` report space.

    All inputs are integer counts (the prediction already snapped). With
    ``control_raw`` the plots can take ``perturbed - control`` straight from JSON::

        {
          "raw_int_rgs": {
            "true_mean": [...], "true_var": [...],
            "<pred_label>_mean": [...], "<pred_label>_var": [...],
            "ctrl_mean": [...], "ctrl_var": [...],  # if control_raw is given
          },
          "log1p_rgs":           {...},
          "log1p_slice_med_rgs": {...},
        }

    ``pred_label`` names the second arm's keys, so plot readers can tell producers apart.
    Every statistic is stacked on device and moved to host in one transfer.
    """
    if pred_label in ("true", "ctrl"):
        raise ValueError(f"pred_label={pred_label!r} would overwrite the truth/control keys")
    arms = {"true": truth_raw, pred_label: pred_raw_integer}
    if control_raw is not None:
        arms["ctrl"] = control_raw
    projected = {name: _project_raw_to_spaces(x) for name, x in arms.items()}
    spaces = list(projected["true"])
    slots = [(space, name) for space in spaces for name in arms]
    stats = torch.stack([
        s
        for space, name in slots
        for s in (projected[name][space].mean(dim=0, dtype=torch.float64),
                  projected[name][space].var(dim=0, unbiased=False).double())
    ]).cpu().numpy()

    out: dict[str, dict[str, list[float]]] = {space: {} for space in spaces}
    for i, (space, name) in enumerate(slots):
        out[space][f"{name}_mean"] = stats[2 * i].tolist()
        out[space][f"{name}_var"] = stats[2 * i + 1].tolist()
    log_detail(
        logger, "gene_summary",
        "n_genes=%d n_truth=%d n_pred=%d",
        truth_raw.shape[1], truth_raw.shape[0], pred_raw_integer.shape[0],
    )
    return out


# ---------------------------------------------------------------------------
# Per-gene dropout (zero-fraction) summary
# ---------------------------------------------------------------------------


@torch.no_grad()
def _exact_zero_frac_gpu(x: torch.Tensor) -> torch.Tensor:
    """Per-gene fraction of cells with exact zero entry, as a GPU tensor.

    Uses ``torch.eq(x, 0)`` rather than ``abs(x) < eps`` because the point
    of this diagnostic is to catch *exact* drop-out predictions (ReLU
    heads, clipped negatives) vs sub-rounding-threshold floats.  The two
    are genuinely different signals; an ``eps`` comparison would conflate
    them.

    Returns:
        ``(n_genes,)`` float32 tensor on ``x.device``.  Empty-input guard
        returns a zero tensor of the right shape without introducing a
        D2H sync.
    """
    n = x.shape[0]
    n_genes = x.shape[1]
    if n == 0:
        return torch.zeros(n_genes, device=x.device, dtype=torch.float32)
    return (x == 0).to(torch.float32).mean(dim=0)


@torch.no_grad()
def compute_dropout_summary(
    *,
    control_raw: torch.Tensor,
    truth_raw: torch.Tensor,
    predicted_model_space: torch.Tensor,
    model_output_space: Union[str, "ModelOutputSpace"],
) -> dict[str, dict[str, list[float]]]:
    """Per-gene exact-zero fraction (drop-out rate) for every DEG target space.

    For each entry in :data:`DEG_SPACES`, converts (control, truth,
    predicted) into that space (via :func:`_project_predictions_to_spaces`
    / :func:`_project_raw_to_spaces`) and returns the fraction of cells
    where each gene's value is **exactly** zero.  Intended as a drop-out
    calibration diagnostic complementary to :func:`compute_gene_summary`
    and :func:`compute_deg_detail`.

    Truth and control are integer counts, so their zero set is the cells'
    own. The prediction's zero set is taken *after* the count-lattice snap:
    it counts genuine zero predictions and sub-threshold positives that
    :func:`src.model.inference_output.to_raw_integer` rounded to 0 alike.

    Args:
        control_raw: ``(n_ctrl, n_genes)`` raw integer counts.
        truth_raw: ``(n_truth, n_genes)`` raw integer counts.
        predicted_model_space: ``(n_pred, n_genes)`` in
            ``model_output_space`` (same tensor used by the K-rep loop).
        model_output_space: Canonical model output space id.

    Returns:
        Nested dict of Python lists (JSON-friendly), keyed by DEG
        target space::

            {
              "raw_int_rgs": {
                "ctrl_zero_frac":  [n_genes floats],
                "true_zero_frac":  [n_genes floats],
                "pred_zero_frac":  [n_genes floats],
              },
            }
    """
    ctrl_spaces = _project_raw_to_spaces(control_raw)
    truth_spaces = _project_raw_to_spaces(truth_raw)
    pred_spaces = _project_predictions_to_spaces(
        predicted_model_space, model_output_space,
    )
    # Every per-gene zero fraction is stacked on device: one host transfer.
    slot_keys: list[tuple[str, str]] = []
    zero_tensors: list[torch.Tensor] = []
    for space in DEG_SPACES:
        zero_tensors.append(_exact_zero_frac_gpu(ctrl_spaces[space]))
        slot_keys.append((space, "ctrl_zero_frac"))
        zero_tensors.append(_exact_zero_frac_gpu(truth_spaces[space]))
        slot_keys.append((space, "true_zero_frac"))
        zero_tensors.append(_exact_zero_frac_gpu(pred_spaces[space]))
        slot_keys.append((space, "pred_zero_frac"))

    batch_host = torch.stack(zero_tensors, dim=0).detach().cpu().numpy()

    out: dict[str, dict[str, list[float]]] = {space: {} for space in DEG_SPACES}
    for i, (space, key) in enumerate(slot_keys):
        out[space][key] = batch_host[i].tolist()
    for space in DEG_SPACES:
        log_detail(
            logger, f"dropout_{space}",
            "mean_pred_zero=%.4g mean_true_zero=%.4g mean_ctrl_zero=%.4g",
            float(np.mean(out[space]["pred_zero_frac"])),
            float(np.mean(out[space]["true_zero_frac"])),
            float(np.mean(out[space]["ctrl_zero_frac"])),
        )
    return out


