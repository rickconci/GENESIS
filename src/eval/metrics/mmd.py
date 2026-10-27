"""Distribution-comparison metric: unbiased RBF-kernel MMD².

Kernel buffers are :math:`O((n + m)^2)`, so each arm is capped at
:data:`DEFAULT_MAX_SAMPLES` rows (shared with :mod:`src.eval.metrics.energy`).
Bandwidth and kernel stay on-device: no host sync inside the metric loop.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from .energy import DEFAULT_MAX_SAMPLES, _subsample_rows_torch


def rbf_kernel(
    X: torch.Tensor,
    Y: torch.Tensor,
    sigma: Union[float, torch.Tensor],
) -> torch.Tensor:
    """Pairwise RBF kernel ``k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))``.

    Args:
        X: ``(n, d)`` tensor.
        Y: ``(m, d)`` tensor.
        sigma: Bandwidth as a Python scalar or a 0-d tensor on ``X.device``.

    Returns:
        ``(n, m)`` kernel values; all zeros when ``sigma <= 0``.
    """
    sigma_t: torch.Tensor = (
        sigma
        if isinstance(sigma, torch.Tensor)
        else torch.as_tensor(sigma, device=X.device, dtype=X.dtype)
    )
    XX = (X * X).sum(dim=1, keepdim=True)
    YY = (Y * Y).sum(dim=1, keepdim=True)
    sq_dist = (XX + YY.T - 2.0 * (X @ Y.T)).clamp(min=0.0)
    # The floor keeps exp() finite at sigma=0; torch.where then zeroes that case without a host sync.
    denom = 2.0 * sigma_t.pow(2) + 1e-30
    k = torch.exp(-sq_dist / denom)
    return torch.where(sigma_t > 0, k, torch.zeros_like(k))


def median_heuristic(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Bandwidth: median positive pairwise distance over ``X ∪ Y``, as a 0-d tensor (>= 1e-6)."""
    with torch.no_grad():
        XX = (X * X).sum(dim=1, keepdim=True)
        YY = (Y * Y).sum(dim=1, keepdim=True)
        d_xy = (XX + YY.T - 2.0 * (X @ Y.T)).clamp(min=0.0).sqrt()
        if X.size(0) > 1:
            d_xx = (XX + XX.T - 2.0 * (X @ X.T)).clamp(min=0.0).sqrt()
            triu = torch.triu_indices(d_xx.size(0), d_xx.size(1), offset=1, device=X.device)
            vals_xx = d_xx[triu[0], triu[1]]
        else:
            vals_xx = d_xy.new_zeros(0)
        if Y.size(0) > 1:
            d_yy = (YY + YY.T - 2.0 * (Y @ Y.T)).clamp(min=0.0).sqrt()
            triu = torch.triu_indices(d_yy.size(0), d_yy.size(1), offset=1, device=Y.device)
            vals_yy = d_yy[triu[0], triu[1]]
        else:
            vals_yy = d_xy.new_zeros(0)
        all_d = torch.cat([d_xy.flatten(), vals_xx, vals_yy])
        # Non-finite / zero distances become +inf so they never win the median, with no
        # data-dependent shape. If none is positive (constant inputs) the bandwidth falls back to 1.
        valid = torch.isfinite(all_d) & (all_d > 0)
        masked = torch.where(valid, all_d, torch.full_like(all_d, float("inf")))
        fallback = torch.ones((), device=X.device, dtype=masked.dtype)
        sigma = torch.where(valid.any(), torch.median(masked), fallback)
        return sigma.clamp(min=1e-6)


def mmd_rbf(
    X: torch.Tensor,
    Y: torch.Tensor,
    sigma: Union[float, torch.Tensor, None] = None,
    *,
    max_samples: Optional[int] = DEFAULT_MAX_SAMPLES,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Unbiased MMD² with an RBF kernel, computed on-device.

    .. math::

        \\operatorname{MMD}^2(P, Q) =
            \\mathbb{E}_{x, x' \\sim P}[k(x, x')]
            + \\mathbb{E}_{y, y' \\sim Q}[k(y, y')]
            - 2 \\cdot \\mathbb{E}_{x \\sim P, y \\sim Q}[k(x, y)]

    Within-arm terms are U-statistics: ``(K.sum() - K.diagonal().sum()) / (n * (n - 1))``.

    Args:
        X: ``(n, d)`` samples from the first distribution.
        Y: ``(m, d)`` samples from the second distribution.
        sigma: RBF bandwidth; ``None`` uses :func:`median_heuristic`.
        max_samples: Per-side row cap (Tahoe conditions exceed 30k cells, and the
            full kernels would need >30 GiB); ``None`` or ``<= 0`` disables it.
        generator: Optional ``torch.Generator`` for subsample indices; ``None``
            draws the global RNG.

    Returns:
        0-d float32 tensor with the clamped non-negative MMD²; ``0`` when either
        arm has ``< 2`` cells after subsampling.
    """
    # float32: bf16 inputs give NaNs in exp / median.
    X = X.float()
    Y = Y.float()
    X = _subsample_rows_torch(X, max_samples, generator)
    Y = _subsample_rows_torch(Y, max_samples, generator)

    n, m = X.size(0), Y.size(0)
    if n < 2 or m < 2:
        return torch.tensor(0.0, device=X.device, dtype=torch.float32)

    if sigma is None:
        sigma = median_heuristic(X, Y)

    K_xx = rbf_kernel(X, X, sigma)
    K_yy = rbf_kernel(Y, Y, sigma)
    K_xy = rbf_kernel(X, Y, sigma)

    term_xx = (K_xx.sum() - K_xx.diagonal().sum()) / float(n * (n - 1))
    term_yy = (K_yy.sum() - K_yy.diagonal().sum()) / float(m * (m - 1))
    term_xy = K_xy.mean()

    mmd_sq = term_xx + term_yy - 2.0 * term_xy
    return mmd_sq.clamp(min=0.0).to(torch.float32)
