"""How GENESIS learns to generate cells: the training loss.

GENESIS is a generator. For each cell in a batch it draws ``n_repeats`` samples from the
distribution it predicts for that cell's condition, and we observe one real cell ``y``. The loss
has to judge a whole predicted distribution against that single real cell. Proper scoring rules
do exactly this: their expected value is lowest only when the predicted distribution is the true
one, so minimising them over many cells teaches the model the true distribution.

The loss adds two of them:

1. The energy score (Gneiting & Raftery, 2007) compares whole expression profiles: samples
   should sit close to the real cell and be as spread out as real cells are.
2. The variogram score (Scheuerer & Hamill, 2015) compares how genes vary together: for pairs
   of genes, the gap between them in the samples should match the gap in the real cell. The
   energy score is only weakly sensitive to this structure in high dimension, which is what
   the variogram score was introduced for.

``pertgen_loss`` at the bottom puts them together. Shapes: the model emits flat
``(B * n_repeats, G)`` rows; the scores read them as ``(B, n_repeats, G)``, one pool of samples
per real cell ``(B, G)``. The ablation objectives live in losses_ablations.py.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

from .losses_ablations import rbf_kernel_score


# ---------------------------------------------------------------------------
# 1. The energy score: samples near the real cell, spread like real cells
# ---------------------------------------------------------------------------


def energy_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """``ES = E||X - y|| - 0.5 * E||X - X'||``, averaged over the batch.

    The first term pulls every sample ``X`` toward the real cell ``y``. On its own it would
    collapse all samples onto one point; the second term, the mean distance between two samples,
    rewards spread. Their balance is lowest, in expectation, exactly when the samples come from
    the distribution ``y`` was drawn from.

    Args:
        predictions: Samples ``(B, n_repeats, d)``.
        targets: One real cell per row ``(B, d)``.
        eps: Added under the square roots: ``||v||`` has no gradient at ``v = 0``, and ``eps``
            keeps it finite when two points coincide.

    Returns:
        Batch means, ready to log: ``energy_score``, ``energy_term1`` (``E||X-y||``)
        and ``energy_term2`` (``E||X-X'||`` over distinct pairs).
    """
    diff_xy = predictions - targets.unsqueeze(1)  # (B, n, d)
    term1 = torch.sqrt((diff_xy * diff_xy).sum(-1) + eps).mean(dim=1)  # (B,)

    # Term 2 needs the distance between every pair of samples. Subtracting them pairwise would
    # build a (B, n, n, d) tensor, ~268 GB at production shape, so we use
    # ||a - b||² = ||a||² + ||b||² - 2a·b, which needs only the (B, n, n) Gram matrix. That
    # identity subtracts large numbers to get a small one, so it is accurate when ||x||² is on
    # the scale of the distances themselves. Centring each pool on its own mean gives exactly
    # that and leaves every distance unchanged, so the result stays accurate in bf16 at any
    # spread. The clamp at 0 absorbs the last rounding.
    centered = predictions - predictions.mean(dim=1, keepdim=True)  # (B, n, d)
    x_sq = (centered * centered).sum(-1)  # (B, n)
    gram = torch.bmm(centered, centered.transpose(1, 2))  # (B, n, n)
    dist_xx = torch.sqrt(torch.clamp(x_sq.unsqueeze(2) + x_sq.unsqueeze(1) - 2.0 * gram, min=0.0) + eps)

    n = predictions.size(1)
    eye = torch.eye(n, device=predictions.device, dtype=torch.bool).unsqueeze(0)
    term2 = dist_xx.masked_fill(eye, 0.0).sum(dim=(1, 2)) / (n * (n - 1))  # (B,), distinct pairs only

    term1, term2 = term1.mean(), term2.mean()
    return {"energy_score": term1 - 0.5 * term2, "energy_term1": term1, "energy_term2": term2}


# ---------------------------------------------------------------------------
# 2. The variogram score: gene pairs vary in the samples as in the real cell
# ---------------------------------------------------------------------------
#
# For a pair of genes (i, j), |y_i - y_j|^p measures how far apart the two genes sit in the
# real cell, and a good generator reproduces that across its samples. Scoring all G²/2 pairs
# is too costly, so each step draws a random subset. Gathering those pairs out of the
# (B, n_repeats, G) samples is the variogram's main cost, so the pieces below build up to a
# cheap gather: pairs drawn as permutations of the genes, a permutation gather whose backward
# costs as little as its forward, and then the score itself.


def _permutation_pair_indices(
    n_genes: int,
    n_pairs: int,
    *,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random gene pairs, drawn as perfect matchings: shuffle the genes, pair them off two by two.

    One shuffle of ``n_genes`` gives ``n_genes // 2`` pairs that use every gene exactly once, so
    the block ``[idx_i, idx_j]`` is a permutation of the gene axis, which ``permute_genes``
    gathers cheaply. No gene repeats within a block, which also lowers the variance of the
    estimate compared with drawing pairs independently. Shuffles are stacked until ``n_pairs``
    and the last one is truncated.
    """
    if n_genes < 2:
        raise ValueError(f"Need n_genes >= 2 for pair sampling, got {n_genes}")
    if n_pairs < 1:
        raise ValueError(f"Need n_pairs >= 1, got {n_pairs}")

    block = n_genes // 2
    n_perms = -(-n_pairs // block)  # ceil
    perms = torch.cat([torch.randperm(n_genes, generator=generator)[: 2 * block] for _ in range(n_perms)])
    pairs = perms.view(-1, 2)[:n_pairs].t().contiguous()  # (2, n_pairs): row 0 is idx_i, row 1 idx_j
    if device is not None:
        # Pinned host memory lets the copy run asynchronously on the GPU stream, so the host never
        # waits on it. The indices come from the same CPU generator either way.
        cuda = torch.device(device).type == "cuda"
        pairs = (pairs.pin_memory() if cuda else pairs).to(device, non_blocking=cuda)
    return pairs[0], pairs[1]


class _PermuteGenes(torch.autograd.Function):
    """``x[..., perm]`` for a permutation ``perm``, with a backward as cheap as its forward.

    Every input gene lands in exactly one output position, so its gradient is the one value at
    that position: the backward is a gather by the inverse permutation. Autograd's default
    backward for indexing allows for repeated indices and so accumulates with atomic adds,
    which profiling put at 250 ms per gather; this backward takes 6.6 ms.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(perm)
        return x.index_select(-1, perm)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (perm,) = ctx.saved_tensors
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.numel(), device=perm.device)
        return grad_out.index_select(-1, inv), None


def permute_genes(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Permute the gene (last) axis of ``x``, with an inverse-gather backward.

    ``perm`` holds every index of ``range(x.shape[-1])`` exactly once; that is what makes the
    inverse gather the full gradient.
    """
    return _PermuteGenes.apply(x, perm)


def variogram_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    p: float = 0.5,
    n_pairs: int = 2000,
    batch_subsample: int = 0,
    repeat_subsample: int = 0,
    eps: float = 1e-4,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Debiased variogram score: do gene pairs differ in the samples as they do in the real cell?

    For each pair ``(i, j)``, ``obs = |y_i - y_j|^p`` in the real cell, and ``E1, E2`` are the
    mean of ``|X_i - X_j|^p`` over the first and the second half of the samples. The score is
    ``mean_{pairs} (obs - E1)(obs - E2)``. The two halves are independent, so the product is an
    unbiased estimate of ``(obs - E|X_i - X_j|^p)^2``: the samples' own noise averages out, and
    the score measures the gene-pair structure alone, whatever the model's spread and however
    many samples it sees.

    ``batch_subsample`` and ``repeat_subsample`` score a random contiguous window of cells and of
    samples (0 = all). The score is a mean of unbiased terms, so a window estimates the same
    quantity at a fraction of the cost. ``p = 0.5`` tempers the weight of large gaps. ``|x|^p``
    has an infinite slope at 0 for ``p < 1``, so gaps are floored at ``eps`` first, which caps
    the slope (at about 50 with the defaults).

    The random draws happen in a fixed order, so one seed reproduces one training step: the cell
    window (global RNG), then the gene pairs and the sample window (``generator``).

    Args:
        predictions: ``(B, n_repeats, G)``.
        targets: ``(B, G)``, one real cell per row.
    """
    B, n_repeats, G = predictions.shape
    if targets.shape[0] != B:
        raise ValueError(f"variogram targets need one row per cell, B={B}; got {targets.shape[0]}")

    # A contiguous slice of cells is cheap to differentiate (a zero-fill plus a strided copy),
    # and the loader has already shuffled the cells, so the slice is a random subset.
    if 0 < batch_subsample < B:
        start = int(torch.randint(0, B - batch_subsample + 1, (1,), device="cpu"))
        predictions = predictions[start : start + batch_subsample]
        targets = targets[start : start + batch_subsample]
    idx_i, idx_j = _permutation_pair_indices(G, n_pairs, generator=generator, device=predictions.device)

    if 0 < repeat_subsample < n_repeats:
        offset = int(torch.randint(0, n_repeats - repeat_subsample + 1, (1,), generator=generator))
        # Copy the sample window once into contiguous memory: every pair block below gathers from
        # it, and a gene-axis gather takes 2.0 ms there against 23.7 ms on the strided slice
        # (at (2048, 64, 2000)).
        predictions = predictions[:, offset : offset + repeat_subsample].contiguous()
        n_repeats = repeat_subsample
    if n_repeats < 2:
        raise ValueError(f"the split-half estimator needs n_repeats >= 2, got {n_repeats}")

    obs = (targets[:, idx_i] - targets[:, idx_j]).abs().clamp(min=eps).pow(p)  # (B, P), the real cell's gaps

    # The samples' gaps, one block of G // 2 pairs at a time. A full block uses every gene once and
    # goes through permute_genes; a partial last block (and every block when G is odd) uses plain
    # indexing.
    block = G // 2
    chunks = []
    for start in range(0, idx_i.numel(), block):
        ii, jj = idx_i[start : start + block], idx_j[start : start + block]
        if 2 * ii.numel() == G:
            xp = permute_genes(predictions, torch.cat([ii, jj]))
            x_i, x_j = xp[..., : ii.numel()], xp[..., ii.numel() :]
        else:
            x_i, x_j = predictions[..., ii], predictions[..., jj]
        chunks.append((x_i - x_j).abs().clamp(min=eps).pow(p))
    pred_var = torch.cat(chunks, dim=-1) if len(chunks) > 1 else chunks[0]  # (B, n_repeats, P)

    half = n_repeats // 2
    e1, e2 = pred_var[:, :half].mean(dim=1), pred_var[:, half : 2 * half].mean(dim=1)
    # (obs - e1)(obs - e2), multiplied out as the published runs computed it, so their numbers
    # reproduce bit for bit.
    return (obs * obs - obs * (e1 + e2) + e1 * e2).mean()


# ---------------------------------------------------------------------------
# 3. The training loss: energy + weighted variogram, or one ablation
# ---------------------------------------------------------------------------


def pertgen_loss(
    gene_expression: torch.Tensor,
    predictions: torch.Tensor,
    config: Any,
    rbf_bandwidths: Optional[Sequence[float]] = None,
) -> dict[str, torch.Tensor]:
    """The loss GENESIS trains on: ``energy_score + config.variogram_loss_weight * variogram_score``.

    ``config.train_loss`` picks the objective. ``"energy_score"`` is the main model. The two
    ablations (Extended Data Fig. 9) replace the energy score and keep everything else, so they
    take ``variogram_loss_weight = 0``: ``"mse"`` regresses each condition's mean from one
    noise-free prediction per cell, and ``"rbf_kernel_score"`` keeps the energy score's form with
    a Gaussian kernel at the fixed ``rbf_bandwidths``. The name is matched exactly, the way
    ``is_deterministic`` matches it when deciding whether the model gets noise, so the loss and
    the noise always agree. The variogram settings are read off ``config`` too.

    Args:
        gene_expression: Real cells, one row per cell ``(B, G)``.
        predictions: Samples ``(B * n_repeats, G)``, flat, as the model emits them.
        rbf_bandwidths: The fixed σ_s for ``train_loss='rbf_kernel_score'``.

    Returns:
        Dict with ``"loss"`` and the terms that make it up, ready to log.
    """
    loss_name = config.train_loss
    if loss_name not in ("energy_score", "mse", "rbf_kernel_score"):
        raise ValueError(
            f"Unknown train_loss={loss_name!r}; expected 'energy_score', 'mse' or 'rbf_kernel_score'.",
        )
    if loss_name != "energy_score" and config.variogram_loss_weight != 0.0:
        raise ValueError(f"train_loss={loss_name} scores no auxiliary term; set variogram_loss_weight to 0")

    # MSE: one deterministic prediction per cell, regressed onto the real cells of its condition.
    if loss_name == "mse":
        if config.n_repeats != 1:
            raise ValueError(f"train_loss=mse needs n_repeats == 1; got {config.n_repeats}")
        mse = torch.nn.functional.mse_loss(predictions, gene_expression)
        return {"mse": mse, "loss": mse}

    # The distributional scores: group the flat samples into one pool per real cell.
    if config.n_repeats < 2:
        raise ValueError(f"train_loss={loss_name} needs n_repeats >= 2; got {config.n_repeats}")
    if loss_name == "rbf_kernel_score" and rbf_bandwidths is None:
        raise ValueError("train_loss=rbf_kernel_score needs rbf_bandwidths (fixed, > 0)")
    B = predictions.shape[0] // config.n_repeats
    if gene_expression.shape[0] != B:
        raise ValueError(f"targets need one row per cell, B={B}; got {gene_expression.shape[0]} rows")
    pools = predictions.reshape(B, config.n_repeats, -1)

    # The RBF ablation: the energy score's form with a Gaussian kernel (see rbf_kernel_score).
    if loss_name == "rbf_kernel_score":
        return rbf_kernel_score(pools, gene_expression, rbf_bandwidths)

    metrics = energy_score(pools, gene_expression)
    loss = metrics["energy_score"]
    if config.variogram_loss_weight > 0.0:
        # The variogram reads its own reshape of the predictions, so each score adds its gradient
        # to the flat predictions separately, in the order the published runs used: the gradient
        # then matches theirs bit for bit.
        metrics["variogram_score"] = variogram_score(
            predictions.reshape(B, config.n_repeats, -1), gene_expression,
            p=config.variogram_p, n_pairs=config.variogram_n_pairs,
            batch_subsample=config.variogram_batch_subsample, repeat_subsample=config.variogram_repeat_subsample,
        )
        loss = loss + config.variogram_loss_weight * metrics["variogram_score"]
    metrics["loss"] = loss
    return metrics
