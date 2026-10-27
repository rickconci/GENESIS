"""Arm metrics: compare two cell sets directly, no control involved.

``pearson_r`` / ``mse`` / ``mae`` / ``r2`` on the pseudobulk means, ``r2_var``
and ``r2_cov`` on dispersion, plus the distributional
pair (``mmd_rbf``, ``energy_distance``). Deterministic given the inputs apart
from the distributional subsampling, which draws the global RNG.
"""

from __future__ import annotations

from typing import Optional

import torch

from .energy import DEFAULT_MAX_SAMPLES, energy_distance
from .mmd import mmd_rbf


# ``r2_cov`` materialises a (G, G) covariance per arm (~36 MB at 3000 genes in float32).
# Above the cap it is NaN, not absent, so every summary keeps the same keys.
_DEFAULT_R2_COV_MAX_GENES: int = 3000


def _nan_tensor(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """0-d NaN sentinel on ``device`` (used for absent / undefined metrics)."""
    return torch.tensor(float("nan"), device=device, dtype=dtype)


@torch.no_grad()
def compute_arm_metrics(
    group_a: torch.Tensor,
    group_b: torch.Tensor,
    *,
    r2_cov_max_genes: int = _DEFAULT_R2_COV_MAX_GENES,
    names: Optional[frozenset[str]] = None,
    energy_all_cells: bool = False,
) -> dict[str, torch.Tensor]:
    """Pseudobulk metrics that depend **only** on ``(group_a, group_b)``.

    Returns 0-d GPU tensors so K reps accumulate without per-scalar host
    syncs; finalise with :func:`materialize_metric_tensors`.

    Bundle:

    * ``pearson_r`` / ``mse`` / ``mae`` / ``r2`` — profile mean comparison.
    * ``r2_var`` — R² on per-gene variance; flags collapsed/inflated dispersion.
    * ``r2_cov`` — R² on the upper-triangular gene-gene covariance block.
      NaN when ``n_genes > r2_cov_max_genes`` (quadratic materialisation).
    * ``mmd_rbf`` — unbiased RBF-kernel MMD² (median bandwidth on
      ``group_a ∪ group_b``).
    * ``energy_distance`` — energy statistic between the two empirical
      cell distributions (:mod:`src.eval.metrics.energy`; same subsample
      cap and formula as the permutation test).

    Args:
        group_a: ``(n_a, n_genes)`` cell matrix (e.g. predicted arm).
        group_b: ``(n_b, n_genes)`` cell matrix (e.g. truth arm).
        r2_cov_max_genes: Gene-count cap for ``r2_cov`` (defaults to
            :data:`_DEFAULT_R2_COV_MAX_GENES`).
        names: Metric names to return (``None`` = all). The costly ones
            (``r2_cov``, ``mmd_rbf``, ``energy_distance``) are skipped when absent.
        energy_all_cells: Score ``energy_distance`` on every cell instead of the
            :data:`DEFAULT_MAX_SAMPLES` subsample. MMD keeps its cap.

    Returns:
        Dict of 0-d GPU tensors keyed by metric name.
    """
    def want(name: str) -> bool:
        return names is None or name in names

    device = group_a.device
    ma, mb = group_a.mean(0), group_b.mean(0)
    diff = ma - mb
    ss_tot_mean = torch.var(mb) * (mb.shape[0] - 1)

    va = group_a.var(dim=0, unbiased=False).double()
    vb = group_b.var(dim=0, unbiased=False).double()
    diff_v = va - vb
    ss_tot_var = torch.var(vb) * (vb.shape[0] - 1)
    r2_var = 1.0 - (torch.sum(diff_v**2) / (ss_tot_var + 1e-12))

    metrics: dict[str, torch.Tensor] = {
        "pearson_r": torch.corrcoef(torch.stack([ma, mb]))[0, 1],
        "mse": torch.mean(diff**2),
        "mae": torch.mean(torch.abs(diff)),
        "r2": 1.0 - (torch.sum(diff**2) / (ss_tot_mean + 1e-12)),
        "r2_var": r2_var.to(torch.float32),
    }

    # ``r2_cov``: upper-triangular block of the gene-gene covariance (G x G, the costly one).
    n_genes = int(group_a.shape[1])
    if want("r2_cov") and n_genes <= int(r2_cov_max_genes):
        ca = torch.cov(group_a.T.double())
        cb = torch.cov(group_b.T.double())
        triu_mask = torch.triu(
            torch.ones(n_genes, n_genes, device=ca.device, dtype=torch.bool),
        )
        ca_flat, cb_flat = ca[triu_mask], cb[triu_mask]
        ss_tot_cov = torch.var(cb_flat) * (cb_flat.shape[0] - 1)
        r2_cov = 1.0 - (torch.sum((ca_flat - cb_flat) ** 2) / (ss_tot_cov + 1e-12))
        metrics["r2_cov"] = r2_cov.to(torch.float32)
    else:
        metrics["r2_cov"] = _nan_tensor(device)

    # Both subsample arms above DEFAULT_MAX_SAMPLES cells off the global RNG, MMD first. A skipped
    # one replays its randperm draws so energy_distance sees the same cells under every metric set.
    # All-cells energy draws nothing, so it replays too: MMD, and every later space, then see exactly
    # the cells they see at the default cap.
    for name, fn in (("mmd_rbf", mmd_rbf), ("energy_distance", energy_distance)):
        all_cells = energy_all_cells and name == "energy_distance"
        if want(name):
            metrics[name] = fn(group_a, group_b, max_samples=None) if all_cells else fn(group_a, group_b)
        if not want(name) or all_cells:
            for n in (int(group_a.shape[0]), int(group_b.shape[0])):
                if n > DEFAULT_MAX_SAMPLES:
                    torch.randperm(n, device=device)
    return metrics if names is None else {k: v for k, v in metrics.items() if k in names}
