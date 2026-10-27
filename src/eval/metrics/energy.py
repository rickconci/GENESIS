"""Energy distance between two cell clouds (Székely / scPerturb / pertpy compatible).

    E(P, Q) = 2 * E[||X - Y||] - E[||X - X'||] - E[||Y - Y'||]

Within-arm expectations use the U-statistic (``i != j`` pairs, divide by
``n * (n - 1)``).  The result is clamped to ``>= 0``.  This is the energy
**statistic** (often written :math:`\\mathcal{E}` or :math:`\\mathcal{D}^2` in
Székely–Rizzo); :math:`\\sqrt{E}` is the metric distance.

Pairwise buffers are ``O(n^2 + m^2 + n m)``, so each arm is capped at
:data:`DEFAULT_MAX_SAMPLES` rows (the same cap as MMD).
"""

from __future__ import annotations

from typing import Optional

import torch

# Shared subsample cap for energy distance and MMD (see mmd.py).
DEFAULT_MAX_SAMPLES: int = 2048


def _subsample_rows_torch(
    X: torch.Tensor,
    max_samples: Optional[int],
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    """Return ``X`` subsampled along dim 0 to at most ``max_samples`` rows."""
    n = X.size(0)
    if max_samples is None or max_samples <= 0 or n <= max_samples:
        return X
    if generator is not None:
        idx = torch.randperm(n, generator=generator, device=X.device)[:max_samples]
    else:
        idx = torch.randperm(n, device=X.device)[:max_samples]
    return X.index_select(0, idx)


def _pairwise_euclidean_torch(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """``(n, m)`` Euclidean distance matrix on ``X.device``."""
    XX = (X * X).sum(dim=1, keepdim=True)
    YY = (Y * Y).sum(dim=1, keepdim=True)
    return (XX + YY.T - 2.0 * (X @ Y.T)).clamp(min=0.0).sqrt()


def energy_distance(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    max_samples: Optional[int] = DEFAULT_MAX_SAMPLES,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Energy distance between empirical cell distributions (GPU).

    Args:
        X: ``(n, d)`` samples from the first distribution.
        Y: ``(m, d)`` samples from the second distribution.
        max_samples: Per-side row cap; ``None`` or ``<= 0`` disables subsampling.
        generator: Optional ``torch.Generator`` for subsample indices.

    Returns:
        0-d float32 tensor on ``X.device`` with the clamped non-negative energy
        statistic; ``0`` when either arm has ``< 2`` cells after subsampling.
    """
    X = X.float()
    Y = Y.float()
    X = _subsample_rows_torch(X, max_samples, generator)
    Y = _subsample_rows_torch(Y, max_samples, generator)

    n, m = X.size(0), Y.size(0)
    if n < 2 or m < 2:
        return torch.tensor(0.0, device=X.device, dtype=torch.float32)

    D_xy = _pairwise_euclidean_torch(X, Y)
    D_xx = _pairwise_euclidean_torch(X, X)
    D_yy = _pairwise_euclidean_torch(Y, Y)

    e_xy = D_xy.mean()
    e_xx = (D_xx.sum() - D_xx.diagonal().sum()) / float(n * (n - 1))
    e_yy = (D_yy.sum() - D_yy.diagonal().sum()) / float(m * (m - 1))

    energy = 2.0 * e_xy - e_xx - e_yy
    return energy.clamp(min=0.0).to(torch.float32)
