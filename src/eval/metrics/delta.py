"""Delta metrics: compare ``(a - ctrl_a)`` with ``(b - ctrl_b)``.

The two controls must be disjoint cells. With one shared control the control
means cancel in the residual ``d_a - d_b``, the ``r2_diff`` denominator shares
a baseline, and unrelated arms look correlated by construction. Callers split
one control pool into two random halves per rep instead.
"""

from __future__ import annotations

import torch

from .arm import _nan_tensor
from .deg import _rank_1d_avg


def _pearson_correlation_1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pearson :math:`r` between two 1D tensors (NaN when degenerate)."""
    if x.numel() < 2:
        return _nan_tensor(x.device)
    corr = torch.corrcoef(torch.stack([x, y]))[0, 1]
    degenerate = (torch.var(x) == 0) | (torch.var(y) == 0)
    return torch.where(degenerate, _nan_tensor(x.device), corr)


def _spearman_correlation_1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Spearman :math:`\\rho`: Pearson on average ranks (constant input has constant ranks -> NaN)."""
    if x.numel() < 2:
        return _nan_tensor(x.device)
    return _pearson_correlation_1d(_rank_1d_avg(x), _rank_1d_avg(y))


@torch.no_grad()
def compute_delta_metrics(
    group_a: torch.Tensor,
    group_b: torch.Tensor,
    ctrl_a: torch.Tensor,
    ctrl_b: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Delta metrics on ``(group_a − ctrl_a)`` vs ``(group_b − ctrl_b)``.

    ``ctrl_a`` and ``ctrl_b`` must be disjoint cells (see module docstring).

    Bundle:

    * ``r2_diff`` — R² on the per-gene perturbation delta.
    * ``pearson_r_diff`` — Pearson correlation of per-gene deltas.
    * ``spearman_r_diff`` — Spearman correlation of per-gene deltas.
    * ``delta_l2_sq_norm_ratio`` — :math:`\\|d_a\\|^2 / \\|d_b\\|^2`.
      NaN when ``\\|d_b\\|^2 = 0``.

    Args:
        group_a: ``(n_a, n_genes)`` arm A cells.
        group_b: ``(n_b, n_genes)`` arm B cells.
        ctrl_a: ``(n_ctrl_a, n_genes)`` control cells paired with arm A.
        ctrl_b: ``(n_ctrl_b, n_genes)`` control cells paired with arm B.

    Returns:
        Dict of 0-d GPU tensors with ``r2_diff``, ``pearson_r_diff``,
        ``spearman_r_diff``, and ``delta_l2_sq_norm_ratio``.

    Raises:
        ValueError: If ``ctrl_a is ctrl_b`` (caller passed the same tensor
            twice; that path biases the deltas).
    """
    if ctrl_a is ctrl_b:
        raise ValueError(
            "compute_delta_metrics requires DISJOINT controls; "
            "got ctrl_a is ctrl_b. Split a single control pool into two "
            "random halves.",
        )
    ma, mb = group_a.mean(0), group_b.mean(0)
    d_a = ma - ctrl_a.mean(0)
    d_b = mb - ctrl_b.mean(0)

    ss_res_delta = torch.sum((d_a - d_b) ** 2)
    ss_tot_delta = torch.var(d_b) * (d_b.shape[0] - 1)
    r2_diff = 1.0 - (ss_res_delta / (ss_tot_delta + 1e-12))

    num = (d_a.double() ** 2).sum()
    den = (d_b.double() ** 2).sum()
    ratio = torch.where(
        den > 0,
        (num / den).to(torch.float32),
        _nan_tensor(group_a.device),
    )
    return {
        "r2_diff": r2_diff.to(ma.dtype),
        "pearson_r_diff": _pearson_correlation_1d(d_a, d_b).to(ma.dtype),
        "spearman_r_diff": _spearman_correlation_1d(d_a, d_b).to(ma.dtype),
        "delta_l2_sq_norm_ratio": ratio,
    }
