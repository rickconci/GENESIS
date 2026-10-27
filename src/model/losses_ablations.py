"""Ablation objectives (Extended Data Fig. 9): the RBF kernel score, per cell and per set."""

from __future__ import annotations

import math
from typing import Sequence

import torch


def rbf_kernel_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    bandwidths: Sequence[float],
) -> dict[str, torch.Tensor]:
    """The energy score with its distance kernel swapped for a multi-scale Gaussian.

    ``S = ½·E k(X, X′) − E k(X, y)``, ``k(a, b) = mean_s exp(−‖a − b‖² / 2σ_s²)``.

    The energy score is this same kernel score with ``k = −‖a − b‖``. Over y ~ Q the
    expectation is ``½·MMD²(P, Q) − ½·E k(Y, Y′)``: minimising it per cell minimises
    the RBF-MMD to the condition's distribution, and it is strictly proper because a
    Gaussian (and a mixture of Gaussians) is a characteristic kernel. Bandwidths are
    fixed numbers, never re-estimated from the batch, so the objective cannot move.

    Distances are squared, never square-rooted, so coinciding samples give finite
    gradients. The sample-sample term uses the per-cell centred Gram expansion, exact
    by translation invariance (see :func:`~src.model.losses.energy_score` for why
    centring is needed), and everything is computed in fp32.

    Args:
        predictions: ``(B, n_repeats, d)`` samples, ``n_repeats >= 2``.
        targets: ``(B, d)`` one observed cell per row.
        bandwidths: the σ_s, each finite and > 0.
    """
    sig = _check_bandwidths(bandwidths)
    dt = torch.promote_types(predictions.dtype, torch.float32)       # at least fp32
    x, y = predictions.to(dt), targets.to(dt)
    n = x.size(1)
    if n < 2:
        raise ValueError(f"rbf_kernel_score needs >= 2 samples per cell; got {n}")
    centered = x - x.mean(dim=1, keepdim=True)
    sq = (centered * centered).sum(-1)
    d_xx = (sq.unsqueeze(2) + sq.unsqueeze(1) - 2.0 * torch.bmm(centered, centered.transpose(1, 2))).clamp_min(0.0)
    diff = x - y.unsqueeze(1)
    d_xy = (diff * diff).sum(-1)                                            # (B, n)
    k_xx, k_xy = _rbf(d_xx, sig), _rbf(d_xy, sig)                             # (B, n, n), (B, n)
    eye = torch.eye(n, device=x.device, dtype=torch.bool).unsqueeze(0)
    term_xx = k_xx.masked_fill(eye, 0.0).sum(dim=(1, 2)) / (n * (n - 1))
    term_xy = k_xy.mean(dim=1)
    score = (0.5 * term_xx - term_xy).mean()
    # The keys training logs, so the caller returns this dict as-is.
    return {"rbf_kernel_score": score, "rbf_term_xx": term_xx.mean(), "rbf_term_xy": term_xy.mean(),
            "loss": score}


def _check_bandwidths(bandwidths: Sequence[float]) -> list[float]:
    sig = [float(s) for s in bandwidths]
    if not sig or any(not math.isfinite(s) or s <= 0.0 for s in sig):
        raise ValueError(f"RBF kernel score needs finite bandwidths > 0; got {list(bandwidths)}")
    return sig


def _rbf(d2: torch.Tensor, sig: Sequence[float]) -> torch.Tensor:
    """mean_s exp(−d² / 2σ_s²) on squared distances: no sqrt, so no NaN gradient at d = 0."""
    return sum(torch.exp(-d2 / (2.0 * s * s)) for s in sig) / len(sig)


def grouped_rbf_kernel_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    bandwidths: Sequence[float],
    weights: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """The set-level estimator of :func:`rbf_kernel_score`: one pool against all a condition's cells.

    ``S_c = ½·mean_{i≠j} k(X_i, X_j) − mean_{i, a} k(X_i, Y_a)`` over the condition's pool X
    (R samples) and its real cells Y (``mask``; ragged, never topped up). Conditions are
    weighted by ``weights`` -- their training-cell budget -- so the objective is the
    per-cell score's (every cell one term); with one cell per set and equal weights it IS
    the     per-cell score. ``rbf_term_yy`` (real-real, needs >= 2 cells) carries no gradient and
    is computed for the log only: ``½·(term_xx + term_yy) − term_xy`` is an unbiased ½·MMD².

    Gram expansions, centred on the pool mean (pool-pool) and on the DETACHED masked
    target mean (pool-real): the expansion subtracts terms of size ‖x‖² to get ‖x − y‖²,
    so it loses precision at convergence unless both sides are first shifted to a common
    centre (exact by translation invariance). Squared distances only, at least fp32.

    Args:
        predictions: ``(C, R, G)`` generated pool per condition, ``R >= 2``.
        targets: ``(C, m_max, G)`` real cells, right-padded.
        mask: ``(C, m_max)`` bool, True at real cells; every condition needs one.
        bandwidths: the fixed σ_s.
        weights: ``(C,)`` > 0, each condition's training-cell budget.
    """
    sig = _check_bandwidths(bandwidths)
    if predictions.dim() != 3 or targets.dim() != 3 or predictions.shape[0] != targets.shape[0]:
        raise ValueError(f"need (C, R, G) and (C, m, G); got {tuple(predictions.shape)}, {tuple(targets.shape)}")
    C, R, _ = predictions.shape
    if mask.shape != targets.shape[:2]:
        raise ValueError(f"mask {tuple(mask.shape)} must be (C, m) = {tuple(targets.shape[:2])}")
    if R < 2:
        raise ValueError(f"need >= 2 generated samples per condition; got {R}")
    if weights.shape != (C,) or not bool(torch.isfinite(weights).all()) or not bool((weights > 0).all()):
        raise ValueError(f"weights must be (C,) = ({C},), finite and > 0; got {weights.tolist()}")
    n = mask.sum(dim=1)
    if bool((n < 1).any()):
        raise ValueError("a condition has no real cells in its set")
    dt = torch.promote_types(predictions.dtype, torch.float32)
    x, y = predictions.to(dt), targets.to(dt)
    wm = mask.to(dt).unsqueeze(-1)                                         # (C, m, 1)
    n_f = n.to(dt)

    # pool-pool, off-diagonal
    p_c = x - x.mean(dim=1, keepdim=True)
    p_sq = (p_c * p_c).sum(-1)
    d_xx = (p_sq.unsqueeze(2) + p_sq.unsqueeze(1) - 2.0 * torch.bmm(p_c, p_c.transpose(1, 2))).clamp_min(0.0)
    eye = torch.eye(R, device=x.device, dtype=torch.bool).unsqueeze(0)
    term_xx = _rbf(d_xx, sig).masked_fill(eye, 0.0).sum(dim=(1, 2)) / (R * (R - 1))      # (C,)

    # pool-real; padded rows collapse onto the centre and are masked out of the sum
    centre = ((y * wm).sum(dim=1, keepdim=True) / n_f.view(C, 1, 1)).detach()
    x_c, y_c = x - centre, (y - centre) * wm
    x_sq, y_sq = (x_c * x_c).sum(-1), (y_c * y_c).sum(-1)
    d_xy = (y_sq.unsqueeze(2) + x_sq.unsqueeze(1) - 2.0 * torch.bmm(y_c, x_c.transpose(1, 2))).clamp_min(0.0)
    term_xy = (_rbf(d_xy, sig) * wm).sum(dim=(1, 2)) / (n_f * R)                          # (C,)

    # real-real, for the log: off-diagonal, masked, conditions with >= 2 cells
    with torch.no_grad():
        d_yy = (y_sq.unsqueeze(2) + y_sq.unsqueeze(1) - 2.0 * torch.bmm(y_c, y_c.transpose(1, 2))).clamp_min(0.0)
        pair = (wm * wm.transpose(1, 2)).masked_fill(torch.eye(y.shape[1], device=y.device, dtype=torch.bool), 0.0)
        many = n >= 2
        term_yy = (_rbf(d_yy, sig) * pair).sum(dim=(1, 2)) / (n_f * (n_f - 1)).clamp_min(1.0)
        term_yy_mean = term_yy[many].mean() if bool(many.any()) else torch.tensor(float("nan"), dtype=dt)

    w = weights.to(dt)
    wsum = w.sum()
    score = (w * (0.5 * term_xx - term_xy)).sum() / wsum
    return {"rbf_kernel_score": score, "rbf_term_xx": (w * term_xx).sum() / wsum,
            "rbf_term_xy": (w * term_xy).sum() / wsum, "rbf_term_yy": term_yy_mean, "loss": score}


def median_pairwise_distance(x: torch.Tensor) -> float:
    """Median Euclidean distance over all distinct pairs of rows: the RBF bandwidth's scale."""
    if x.dim() != 2 or x.size(0) < 2:
        raise ValueError(f"need >= 2 rows of shape (n, d); got {tuple(x.shape)}")
    d = torch.cdist(x.float(), x.float())
    iu = torch.triu_indices(x.size(0), x.size(0), offset=1, device=d.device)
    med = float(d[iu[0], iu[1]].median())
    if not med > 0.0:
        raise ValueError("median pairwise distance is 0: the rows are (mostly) identical")
    return med
