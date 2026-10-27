"""DEG overlap: two independent Wilcoxon + BH analyses, then how well they agree.

:func:`compute_deg_overlap_metrics(ctrl_a, target_a, ctrl_b, target_b)` calls
DEGs in each arm against its own control, then scores the overlap: Jaccard and
top-k p-ranked overlap, ``spearman_lfc_sig_<suffix>`` (rank correlation of
log2FC on truth-DE genes), ``pearson_lfc_de_intersection_<suffix>`` (Pearson on
the intersection-DE log2FC), and ``roc_auc_<suffix>`` / ``pr_auc_<suffix>``
with ``sig_a`` as the label and ``-log10(q_b)`` as the score (STATE paper eqs.
44-46). ``pr_auc_dir_match_<suffix>`` signs that score by
``sign(z_a) * sign(z_b)`` so confidently wrong-direction calls rank below
non-DE genes. ``spearman_z_dir_truth_top_k<k>_<suffix>`` asks what the top-K
overlaps cannot: *within* the truth's strongest k DEGs, is their ORDER right?

Every primitive returns 0-d GPU tensors with no host sync of its own.
:func:`compute_deg_detail` exports the per-gene z / q / LFC arrays for plots.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch

if TYPE_CHECKING:
    from ...model.inference_output import ModelOutputSpace

from ..orchestration.log_utils import log_detail
from .spaces import DEG_SPACES, _project_predictions_to_spaces, _project_raw_to_spaces

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Average ranks (scipy ``rankdata`` ties), shared with Spearman in delta.py
# ---------------------------------------------------------------------------

def _torch_rankdata_average(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n_samples, n_genes = X.shape
    device = X.device
    sorted_vals, sort_idx = torch.sort(X, dim=0)
    ordinal = torch.arange(
        1, n_samples + 1, device=device, dtype=torch.float64,
    ).unsqueeze(1).expand(n_samples, n_genes)
    new_group = torch.ones(n_samples, n_genes, dtype=torch.bool, device=device)
    new_group[1:] = sorted_vals[1:] != sorted_vals[:-1]
    group_ids = torch.cumsum(new_group, dim=0)
    # ``group_ids`` lies in ``[1, n_samples]`` by construction (one id per
    # unique value, at most one per sample).  Allocating ``n_samples + 1``
    # rows up-front avoids a D2H sync on ``group_ids.max()`` to size the
    # scatter buffers; unused rows stay zero and are
    # inert under both the ``gather`` and the ``(rank_cnt**3 - rank_cnt).sum``
    # tie-correction (``0^3 - 0 == 0``).
    max_gid_bound = n_samples + 1
    gid = group_ids.long()
    ones = torch.ones(n_samples, n_genes, device=device, dtype=torch.float64)
    rank_sum = torch.zeros(max_gid_bound, n_genes, device=device, dtype=torch.float64)
    rank_cnt = torch.zeros(max_gid_bound, n_genes, device=device, dtype=torch.float64)
    rank_sum.scatter_add_(0, gid, ordinal)
    rank_cnt.scatter_add_(0, gid, ones)
    avg_per_group = rank_sum / rank_cnt.clamp(min=1)
    sorted_avg = avg_per_group.gather(0, gid)
    ranks = torch.zeros(n_samples, n_genes, device=device, dtype=torch.float64)
    ranks.scatter_(0, sort_idx, sorted_avg)
    tie_term = (rank_cnt.pow(3) - rank_cnt).sum(dim=0)
    return ranks, tie_term


# ---------------------------------------------------------------------------
# Wilcoxon rank-sum per gene + Benjamini-Hochberg
# ---------------------------------------------------------------------------

# Wilcoxon rank-pass uses ~8 float64 ``(n_combined+1, chunk)`` buffers
# concurrently (sort_idx, group_ids, rank_sum, rank_cnt, avg_per_group,
# sorted_avg, ranks, ordinal) so the per-chunk peak is ``8 * chunk_bytes``.
# We size each chunk to consume only the *free* GPU memory minus a
# configurable headroom (default 5 GiB) so the run does not steal memory
# from concurrent kernels / other processes on a shared GPU.
_WILCOXON_BUFFERS_PER_CHUNK = 8
_WILCOXON_HEADROOM_BYTES = int(
    os.environ.get("PERTGEN_WILCOXON_HEADROOM_BYTES", str(5 * 1024**3))
)
# Per-buffer budget on CPU, or when CUDA cannot report free memory.
_WILCOXON_CPU_FALLBACK_BYTES = 128 * 1024 * 1024


def _pick_wilcoxon_gene_chunk(
    n_combined: int,
    n_genes: int,
    device: torch.device,
    *,
    headroom_bytes: int = _WILCOXON_HEADROOM_BYTES,
) -> int:
    """Largest gene-chunk that fits in ``(free_gpu - headroom)`` per pass.

    The chunk-time peak is ``~8 * (n_combined+1) * chunk * 8`` bytes (eight
    fp64 buffers); we solve for ``chunk`` given a per-buffer byte budget
    of ``(free_gpu - headroom_bytes) / 8``.  On CPU (or when CUDA cannot
    report free memory) we fall back to a 128 MiB per-buffer budget so
    callers without a GPU still chunk sensibly.

    Args:
        n_combined: ``n_control + n_target`` rows entering the rank pass.
        n_genes: Total gene count to chunk over.
        device: Tensor device — used to query free CUDA memory.
        headroom_bytes: Bytes to leave free on the GPU (default 5 GiB).

    Returns:
        ``min(chunk, n_genes)`` so small panels pass through unchunked.
    """
    per_gene_bytes = (n_combined + 1) * 8
    if device.type == "cuda" and torch.cuda.is_available():
        free_bytes, _ = torch.cuda.mem_get_info(device)
        budget = max(0, int(free_bytes) - int(headroom_bytes))
        per_buffer_bytes = budget // _WILCOXON_BUFFERS_PER_CHUNK
    else:
        per_buffer_bytes = _WILCOXON_CPU_FALLBACK_BYTES
    chunk = max(1, per_buffer_bytes // per_gene_bytes)
    return int(min(chunk, n_genes))


def _compute_wilcoxon_vectorized(
    control_X: torch.Tensor,
    target_X: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-gene Mann-Whitney U, matching ``scipy.stats.mannwhitneyu`` (asymptotic, continuity).

    Inputs are integer counts (``raw_int_rgs``), so LFC is
    ``log2((mean_t + eps) / (mean_c + eps))`` on count means — the standard
    scRNA-seq convention. The U test itself is rank-invariant; only LFC would
    change if the inputs lived in another space.

    Args:
        control_X: Control cells, shape ``(n_control, n_genes)``.
        target_X: Target cells, shape ``(n_target, n_genes)``.

    The rank pass is performed in **gene chunks** sized by
    :func:`_pick_wilcoxon_gene_chunk` (free GPU memory minus a 5 GiB
    headroom by default).  All ops in :func:`_torch_rankdata_average` are
    independent across the gene axis (``torch.sort``, ``cumsum``, and the
    dim-0 ``scatter_add``), so the chunked result is bit-for-bit identical
    to the unchunked version.
    """
    n_control = control_X.shape[0]
    n_target = target_X.shape[0]
    n_genes = control_X.shape[1]
    N = n_control + n_target
    device = control_X.device

    R_target = torch.empty(n_genes, dtype=torch.float64, device=device)
    tie_term = torch.empty(n_genes, dtype=torch.float64, device=device)
    mean_control = torch.empty(n_genes, dtype=torch.float64, device=device)
    mean_target = torch.empty(n_genes, dtype=torch.float64, device=device)

    chunk = _pick_wilcoxon_gene_chunk(N, n_genes, device)
    for s in range(0, n_genes, chunk):
        e = min(s + chunk, n_genes)
        c_slice = control_X[:, s:e]
        t_slice = target_X[:, s:e]
        X_combined = torch.cat([c_slice, t_slice], dim=0)
        ranks_chunk, tie_chunk = _torch_rankdata_average(X_combined)
        R_target[s:e] = ranks_chunk[n_control:].sum(dim=0)
        tie_term[s:e] = tie_chunk
        mean_control[s:e] = c_slice.double().mean(dim=0)
        mean_target[s:e] = t_slice.double().mean(dim=0)
        del X_combined, ranks_chunk, tie_chunk

    U = R_target - n_target * (n_target + 1.0) / 2.0
    mu = n_target * n_control / 2.0
    sigma_sq = (n_target * n_control / 12.0) * ((N + 1.0) - tie_term / (N * (N - 1.0)))
    sigma = torch.sqrt(sigma_sq)
    d = U - mu
    z = (d - 0.5 * torch.sign(d)) / sigma
    z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    p_values = 2.0 * torch.special.ndtr(-torch.abs(z))
    log2_fc = torch.log2((mean_target + 1e-9) / (mean_control + 1e-9))
    return z.float(), p_values.float(), log2_fc.float()


def _benjamini_hochberg_torch(p_values: torch.Tensor) -> torch.Tensor:
    n = len(p_values)
    sorted_pvals, sort_indices = torch.sort(p_values)
    correction_factors = torch.arange(1, n + 1, dtype=torch.float32, device=p_values.device)
    corrected = sorted_pvals * n / correction_factors
    corrected = corrected.flip(0).cummin(0).values.flip(0).clamp(max=1.0)
    adjusted = torch.zeros_like(p_values)
    adjusted[sort_indices] = corrected
    return adjusted


# ---------------------------------------------------------------------------
# Sync-free 1D rank / Spearman / ROC-AUC / PR-AUC helpers
# ---------------------------------------------------------------------------
#
# No op below has a data-dependent shape, so one host sync at the end of the
# bundle materialises every 0-d scalar.

def _rank_1d_avg(x: torch.Tensor) -> torch.Tensor:
    """1D average ranks (1-indexed, ties averaged, float64)."""
    ranks_2d, _ = _torch_rankdata_average(x.unsqueeze(1))
    return ranks_2d.squeeze(1)


def _pearson_correlation_subset(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Pearson :math:`r` on the genes where ``mask`` is True, as mask-weighted sums.

    Weighting instead of ``x[mask]`` keeps every shape fixed (no host sync).
    NaN when ``mask.sum() < 2`` or either side is constant on the subset.
    """
    w = mask.bool().to(torch.float64)
    W = w.sum()
    safe_W = W.clamp(min=1.0)
    x64 = x.to(torch.float64)
    y64 = y.to(torch.float64)
    dx = x64 - (w * x64).sum() / safe_W
    dy = y64 - (w * y64).sum() / safe_W
    cov = (w * dx * dy).sum()
    var_x = (w * dx * dx).sum()
    var_y = (w * dy * dy).sum()
    corr = cov / torch.sqrt(var_x * var_y).clamp(min=1e-30)
    nan_t = torch.tensor(float("nan"), device=x.device, dtype=torch.float64)
    degenerate = (W < 2.0) | (var_x == 0) | (var_y == 0)
    return torch.where(degenerate, nan_t, corr).to(torch.float32)


def _spearman_correlation_subset(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """``scipy.stats.spearmanr(x[mask], y[mask])``, shape-stable.

    Non-mask entries become ``+inf`` before ranking, so the subset holds ranks
    ``1..n_sig`` (ties averaged within it, as spearmanr would) and the tied
    ``+inf`` block gets zero weight in the Pearson. Assumes finite ``x``/``y``
    (LFCs carry a 1e-9 pseudocount), so ``+inf`` only comes from the sentinel.
    """
    mask_b = mask.bool()
    inf_x = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
    inf_y = torch.tensor(float("inf"), device=y.device, dtype=y.dtype)
    rx = _rank_1d_avg(torch.where(mask_b, x, inf_x))
    ry = _rank_1d_avg(torch.where(mask_b, y, inf_y))
    return _pearson_correlation_subset(rx, ry, mask_b)


def _spearman_z_rank_truth_topk(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    sig_a: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Spearman ρ of DE *evidence rank* within the truth's strongest ``k`` DEGs.

    Answers "inside the top-k, does the ORDER match?".  The top-K overlap
    metrics score set membership only, so a model that recovers every
    member of the top-k but inverts their ranking scores a perfect 1.0
    there and can still score ≈ −1 here.

    Selection is truth-only (``min(k, n_sig_a)`` genes), so the gene set is
    a property of the condition and identical for every model — the same
    argument that makes ``adaptivek_truth`` preferable to a pred-side
    denominator.  A missed DEG is not dropped: it keeps the model's own
    (weak) evidence rank and drags the correlation down.

    Two deliberate choices in the ranking statistic:

    * Rank on ``|z|``, not ``q``.  Within one arm BH is monotone in ``p``
      and ``p`` is monotone in ``|z|``, so the two orderings agree — except
      that ``p = 2Φ(−|z|)`` underflows to 0.0 in float32 near ``|z| > 13``
      (routine at ~1e4 cells), tying hundreds of genes at the float floor
      and destroying precisely the ordering being measured.  ``z`` never
      saturates.
    * Score the model by ``sign(z_a) · z_b``, not ``|z_b|``.  Under ``|z_b|``
      a gene the model drives hard in the WRONG direction reads as a top
      hit; signing it against the truth's direction sinks it below every
      agreeing gene (same guard as ``pr_auc_dir_match``).

    Args:
        z_a: ``(n_genes,)`` truth Wilcoxon z (detfloor-masked by caller).
        z_b: ``(n_genes,)`` prediction Wilcoxon z.
        sig_a: ``(n_genes,)`` boolean truth DE calls.
        k: Top-k depth; capped at ``n_genes`` and at ``n_sig_a``.

    Returns:
        0-d float32 ρ, or NaN when the truth calls fewer than 2 DEGs.
    """
    n_genes = int(z_a.shape[0])
    k_slice = min(int(k), n_genes)
    neg_inf = torch.tensor(-float("inf"), device=z_a.device, dtype=z_a.dtype)
    top_idx = torch.topk(
        torch.where(sig_a, z_a.abs(), neg_inf), k=k_slice, largest=True,
    ).indices
    # Re-gather ``sig_a`` so padding slots (when ``n_sig_a < k_slice`` every
    # remaining candidate is the −inf sentinel) drop out rather than seeding
    # the subset with non-DE genes.
    mask = torch.zeros(n_genes, dtype=torch.bool, device=z_a.device)
    mask.scatter_(0, top_idx, sig_a[top_idx])
    return _spearman_correlation_subset(z_a.abs(), torch.sign(z_a) * z_b, mask)


def _roc_auc_binary(
    labels: torch.Tensor,
    scores: torch.Tensor,
) -> torch.Tensor:
    """Sync-free ROC-AUC via the Mann-Whitney U identity.

    Uses ``AUC = (R_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)``
    with average-rank tie handling -- equivalent to
    ``sklearn.metrics.roc_auc_score``.  All intermediate tensors are
    shape-stable so the primitive has no host sync of its own.

    Args:
        labels: 1D boolean / 0-1 tensor (ground-truth labels).
        scores: 1D tensor of confidence scores (higher = more positive).
            ROC-AUC is rank-invariant, so ``-log10(q)``, ``-q``, and
            ``1 - q`` all give the same value.

    Returns:
        0-d float32 tensor with ROC-AUC in ``[0, 1]``, or NaN when one
        class is empty.
    """
    y = labels.to(torch.float64)
    s = scores.to(torch.float64)
    r = _rank_1d_avg(s)
    n = torch.tensor(float(y.shape[0]), device=y.device, dtype=torch.float64)
    n_pos = y.sum()
    n_neg = n - n_pos
    r_pos = (y * r).sum()
    u = r_pos - n_pos * (n_pos + 1.0) / 2.0
    denom = (n_pos * n_neg).clamp(min=1.0)
    auc = u / denom
    nan_t = torch.tensor(float("nan"), device=y.device, dtype=torch.float64)
    degenerate = (n_pos == 0) | (n_neg == 0)
    return torch.where(degenerate, nan_t, auc).to(torch.float32)


def _pr_auc_binary(
    labels: torch.Tensor,
    scores: torch.Tensor,
) -> torch.Tensor:
    """Sync-free PR-AUC using sklearn's average-precision formulation.

    Sorts by score descending (stable), then accumulates
    ``AP = Σ_i y_sorted_i * tp_i / (i+1) / n_pos``.  This matches
    ``sklearn.metrics.average_precision_score`` except for the way
    tied scores are pooled: sklearn groups tied thresholds and contributes
    the precision at the grouped boundary, while the stable-argsort
    version here evaluates them in sort order.  The difference is
    typically negligible for ``-log10(q)``-style continuous confidences.

    Args:
        labels: 1D boolean / 0-1 tensor.
        scores: 1D tensor of confidence scores (higher = more positive).

    Returns:
        0-d float32 tensor with PR-AUC / average precision in ``[0, 1]``,
        or NaN when the positive class is empty.
    """
    y = labels.to(torch.float64)
    s = scores.to(torch.float64)
    n = y.shape[0]
    order = torch.argsort(s, descending=True, stable=True)
    y_sorted = y[order]
    tp = torch.cumsum(y_sorted, dim=0)
    rank = torch.arange(1, n + 1, device=y.device, dtype=torch.float64)
    precision = tp / rank
    n_pos = y.sum()
    ap = (y_sorted * precision).sum() / n_pos.clamp(min=1.0)
    nan_t = torch.tensor(float("nan"), device=y.device, dtype=torch.float64)
    return torch.where(n_pos == 0, nan_t, ap).to(torch.float32)


# ---------------------------------------------------------------------------
# DEG overlap metrics
# ---------------------------------------------------------------------------

# One DEG call: BH q < 1e-3 AND |log2FC| >= 0.1, on genes that pass the control
# detection floor P(count > 0) >= t (t=0.10 -> infix ``detfloor10``). Near-zero control
# genes have tiny variance, so Wilcoxon overcalls on them (Bourgon et al. 2010; Squair
# et al. 2021); detection rate filters them without depending on the report-space scale.
_DEG_FDR: float = 0.001
_DEG_MIN_ABS_LFC: float = 0.1
_DEG_SIG: str = "fdr0p001_lfc0p1"


def detfloor_variant_infix(t: float) -> str:
    """DEG-family metric infix for detection-fraction floor ``t``.

    Encodes percent: ``0.10`` → ``"detfloor10"``, ``0.05`` → ``"detfloor05"``.
    ``round`` guards float edge cases (``0.1 * 100 == 9.999...``).
    """
    pct = int(round(float(t) * 100.0))
    if pct < 0 or pct > 100:
        raise ValueError(
            f"detfloor_variant_infix: t must be in [0, 1]; got {t!r} → {pct}%"
        )
    return f"detfloor{pct:02d}"


def _apply_deg_floor_mask(
    q_a: torch.Tensor, q_b: torch.Tensor,
    z_a: torch.Tensor, z_b: torch.Tensor,
    keep: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Neuter dropped genes: q → 1.0 (sig=False, -log10(q)=0), z → 0.

    Keeps LFC intact; Spearman/Pearson subsets already respect the ``sig``
    mask. ``keep`` is ``(n_genes,)`` boolean from the caller.
    """
    ones = torch.ones((), dtype=q_a.dtype, device=q_a.device)
    zeros_z = torch.zeros((), dtype=z_a.dtype, device=z_a.device)
    q_a_m = torch.where(keep, q_a, ones)
    q_b_m = torch.where(keep, q_b, ones)
    z_a_m = torch.where(keep, z_a, zeros_z)
    z_b_m = torch.where(keep, z_b, zeros_z)
    return q_a_m, q_b_m, z_a_m, z_b_m


def _deg_family_from_stats(
    q_a: torch.Tensor, q_b: torch.Tensor,
    z_a: torch.Tensor, z_b: torch.Tensor,
    lfc_a: torch.Tensor, lfc_b: torch.Tensor,
    *,
    k_list: list[int],
    infix: str,
    include_non_topk: bool = True,
) -> dict[str, torch.Tensor]:
    """One floor's DEG family from per-gene stats already masked by that floor.

    Keys end in ``{_DEG_SIG}_{infix}``, the top-K overlaps in
    ``{_DEG_SIG}_adaptivek_truth_{infix}``. ``include_non_topk`` gates the
    set, LFC and AUC metrics; the top-K overlaps are always emitted.
    """
    device = q_a.device
    n_genes = q_a.shape[0]
    sig_a = (q_a < _DEG_FDR) & (torch.abs(lfc_a) >= _DEG_MIN_ABS_LFC)
    sig_b = (q_b < _DEG_FDR) & (torch.abs(lfc_b) >= _DEG_MIN_ABS_LFC)
    n_sig_a = sig_a.sum()
    direction_match = torch.sign(z_a) == torch.sign(z_b)
    sfx = f"{_DEG_SIG}_{infix}"

    out: dict[str, torch.Tensor] = {}
    if include_non_topk:
        scores_b = -torch.log10(q_b.clamp(min=1e-300))
        union = (sig_a | sig_b).sum()
        inter = (sig_a & sig_b).sum()
        out[f"n_sig_degs_a_{sfx}"] = n_sig_a
        out[f"n_sig_degs_b_{sfx}"] = sig_b.sum()
        out[f"n_reproducible_degs_{sfx}"] = (sig_a & sig_b & direction_match).sum()
        # inter ⊆ union, so union == 0 gives 0/1 == 0 without a host-side branch.
        out[f"jaccard_sig_{sfx}"] = inter.to(torch.float32) / union.to(torch.float32).clamp(min=1.0)
        out[f"spearman_lfc_sig_{sfx}"] = _spearman_correlation_subset(lfc_a, lfc_b, sig_a)
        r_int = _pearson_correlation_subset(lfc_a, lfc_b, sig_a & sig_b)
        out[f"pearson_lfc_de_intersection_{sfx}"] = r_int
        # A pred that cannot be scored against a truth of >= 3 DEGs (shares < 2 of them, or is flat
        # on them) has failed, not abstained: it scores 0, so a K-rep mean averages the 0 in rather
        # than skipping the rep, as a model's single rep would. Truth < 3 DEGs: undefined for all.
        out[f"pearson_lfc_de_intersection_miss0_{sfx}"] = torch.where(
            n_sig_a >= 3, torch.nan_to_num(r_int, nan=0.0), torch.full_like(r_int, float("nan")),
        )
        out[f"roc_auc_{sfx}"] = _roc_auc_binary(sig_a, scores_b)
        out[f"pr_auc_{sfx}"] = _pr_auc_binary(sig_a, scores_b)
        out[f"pr_auc_dir_match_{sfx}"] = _pr_auc_binary(sig_a, torch.sign(z_a) * torch.sign(z_b) * scores_b)
        # Order agreement INSIDE the truth's top-k DEGs; the top-K overlap below scores membership only.
        for k in k_list:
            out[f"spearman_z_dir_truth_top_k{k}_{sfx}"] = _spearman_z_rank_truth_topk(z_a, z_b, sig_a, k)

    # Top-K overlap as R-precision: rank each arm's significant genes by q, keep the first
    # k_eff = min(k, n_sig_a) of each, count the sign-matching shared genes over k_eff. k_eff
    # is the truth's, a property of the condition and the same for every model, so over- and
    # under-calling cost alike (a fixed k penalises conditions with < k true DEGs; the pred's
    # count would reward under-calling). No truth DEGs -> NaN.
    inf_p = torch.tensor(float("inf"), device=device, dtype=q_a.dtype)
    nan_t = torch.tensor(float("nan"), device=device, dtype=torch.float32)
    p_a_sig = torch.where(sig_a, q_a, inf_p)
    p_b_sig = torch.where(sig_b, q_b, inf_p)
    for k in k_list:
        k_slice = min(k, n_genes)
        k_eff = torch.clamp(n_sig_a.to(torch.int64), max=int(k_slice))
        keep_slot = torch.arange(k_slice, device=device, dtype=torch.int64) < k_eff
        masks = []
        for p_sig, sig in ((p_a_sig, sig_a), (p_b_sig, sig_b)):
            # sig[idx] drops padding slots when an arm has fewer than k_slice significant genes
            idx = torch.topk(p_sig, k=k_slice, largest=False).indices
            masks.append(torch.zeros(n_genes, dtype=torch.bool, device=device).scatter_(
                0, idx, sig[idx] & keep_slot))
        match_count = (masks[0] & masks[1] & direction_match).sum().to(torch.float32)
        value = match_count / k_eff.to(torch.float32).clamp(min=1.0)
        out[f"deg_overlap_top_p_ranked_k{k}_{_DEG_SIG}_adaptivek_truth_{infix}"] = (
            torch.where(k_eff == 0, nan_t, value))
    return out


@torch.no_grad()
def compute_deg_overlap_metrics(
    ctrl_a: torch.Tensor,
    target_a: torch.Tensor,
    ctrl_b: torch.Tensor,
    target_b: torch.Tensor,
    k_list: Optional[list[int]] = None,
    *,
    ctrl_det_floors: tuple[float, ...] = (0.10,),
    ctrl_det_frac: Optional[torch.Tensor] = None,
    include_non_topk: bool = True,
) -> dict[str, torch.Tensor]:
    """DEG-level comparison: overlap between two independent DE analyses.

    Calls ``Wilcoxon(ctrl_a, target_a)`` -> DEGs_a and ``Wilcoxon(ctrl_b, target_b)``
    -> DEGs_b, then scores their agreement once per control-detection floor in
    ``ctrl_det_floors`` (genes with pooled-control ``ctrl_det_frac < t`` drop out).
    The floor is part of the DEG definition, so ``ctrl_det_frac`` is required.
    ``include_non_topk=False`` skips the Jaccard / LFC / AUC family.

    Returns 0-d GPU tensors; finalise with
    :func:`materialize_metric_tensors` / :func:`materialize_metric_tensor_batch`.
    """
    if k_list is None:
        k_list = [20, 50, 100, 500]

    if ctrl_det_frac is None or len(ctrl_det_floors) == 0:
        raise ValueError(
            "compute_deg_overlap_metrics: the detection floor is part of the "
            "DEG definition -- ctrl_det_frac must be set and "
            f"ctrl_det_floors non-empty; got floors={ctrl_det_floors!r}, "
            f"det_frac is {'set' if ctrl_det_frac is not None else 'None'}.",
        )
    if ctrl_det_frac.ndim != 1 or int(ctrl_det_frac.shape[0]) != int(ctrl_a.shape[1]):
        raise ValueError(
            "compute_deg_overlap_metrics: ctrl_det_frac must be (n_genes,); "
            f"got shape={tuple(ctrl_det_frac.shape)}, n_genes={ctrl_a.shape[1]}.",
        )
    seen: set[str] = set()
    for t in ctrl_det_floors:
        if not (0.0 <= float(t) <= 1.0):
            raise ValueError(
                f"compute_deg_overlap_metrics: ctrl_det_floors entries must "
                f"lie in [0, 1]; got {t!r}.",
            )
        infix = detfloor_variant_infix(t)
        if infix in seen:
            raise ValueError(
                "compute_deg_overlap_metrics: ctrl_det_floors produced "
                f"duplicate metric infix {infix!r} for threshold {t!r}; use "
                "thresholds that differ after rounding to percent.",
            )
        seen.add(infix)

    z_a, p_a, lfc_a = _compute_wilcoxon_vectorized(ctrl_a, target_a)
    z_b, p_b, lfc_b = _compute_wilcoxon_vectorized(ctrl_b, target_b)
    q_a = _benjamini_hochberg_torch(p_a)
    q_b = _benjamini_hochberg_torch(p_b)

    metrics: dict[str, torch.Tensor] = {}
    for t in ctrl_det_floors:
        q_a_m, q_b_m, z_a_m, z_b_m = _apply_deg_floor_mask(q_a, q_b, z_a, z_b, ctrl_det_frac >= float(t))
        metrics.update(_deg_family_from_stats(
            q_a_m, q_b_m, z_a_m, z_b_m, lfc_a, lfc_b,
            k_list=k_list, infix=detfloor_variant_infix(t), include_non_topk=include_non_topk,
        ))
    return metrics


# ---------------------------------------------------------------------------
# Per-gene DEG detail (Wilcoxon q/z/LFC arrays, for JSON plot consumers)
# ---------------------------------------------------------------------------


_DEG_DETAIL_ARRAY_KEYS: tuple[str, ...] = (
    "z_truth_vs_ctrl",
    "q_truth_vs_ctrl",
    "lfc_truth_vs_ctrl",
    "z_pred_vs_ctrl",
    "q_pred_vs_ctrl",
    "lfc_pred_vs_ctrl",
)


@torch.no_grad()
def _compute_deg_detail_arrays_gpu(
    ctrl: torch.Tensor,
    truth: torch.Tensor,
    pred: torch.Tensor,
) -> list[torch.Tensor]:
    """Per-gene Wilcoxon / BH outputs for one space, kept on-device.

    Returns the six ``(n_genes,)`` arrays ordered to match
    :data:`_DEG_DETAIL_ARRAY_KEYS`, so the caller can stack them across
    spaces and pull all of them with a single D2H transfer.
    """
    z_a, p_a, lfc_a = _compute_wilcoxon_vectorized(ctrl, truth)
    z_b, p_b, lfc_b = _compute_wilcoxon_vectorized(ctrl, pred)
    q_a = _benjamini_hochberg_torch(p_a)
    q_b = _benjamini_hochberg_torch(p_b)
    return [z_a, q_a, lfc_a, z_b, q_b, lfc_b]


@torch.no_grad()
def compute_deg_detail(
    *,
    control_raw: torch.Tensor,
    truth_raw: torch.Tensor,
    predicted_model_space: torch.Tensor,
    model_output_space: Union[str, "ModelOutputSpace"],
    fdr_threshold: float = 0.01,
) -> dict[str, dict[str, Any]]:
    """Per-gene Wilcoxon detail for every DEG target space in :data:`DEG_SPACES`.

    For each entry in :data:`DEG_SPACES`, converts (control, truth,
    predicted) into that space and runs the Wilcoxon + BH primitive on
    ``(control, truth)`` and ``(control, predicted)``.  Returns a
    JSON-friendly dict keyed by the space id::

        {
          "raw_int_rgs": {
              "fdr_threshold": 0.01, "log2_fc_space": "raw",
              "z_truth_vs_ctrl":  [...], "q_truth_vs_ctrl":  [...],
              "lfc_truth_vs_ctrl":[...],
              "z_pred_vs_ctrl":   [...], "q_pred_vs_ctrl":   [...],
              "lfc_pred_vs_ctrl": [...],
          },
        }

    Downstream plot code builds match / false-positive / missed gene masks
    per space with::

        block = detail["raw_int_rgs"]
        q_a = np.asarray(block["q_truth_vs_ctrl"])
        q_b = np.asarray(block["q_pred_vs_ctrl"])
        z_a = np.asarray(block["z_truth_vs_ctrl"])
        z_b = np.asarray(block["z_pred_vs_ctrl"])
        sig_a = q_a < block["fdr_threshold"]
        sig_b = q_b < block["fdr_threshold"]
        dir_match = np.sign(z_a) == np.sign(z_b)
        match     = sig_a & sig_b & dir_match
        false_pos = sig_b & ~match
        missed    = sig_a & ~sig_b

    Args:
        control_raw: ``(n_ctrl, n_genes)`` raw integer counts.
        truth_raw: ``(n_truth, n_genes)`` raw integer counts.
        predicted_model_space: ``(n_pred, n_genes)`` in ``model_output_space``
            (exactly the same tensor used by the K-rep loop).
        model_output_space: Canonical model output space id.
        fdr_threshold: FDR threshold recorded alongside the arrays so plot
            readers know which cutoff the producer had in mind.

    Returns:
        JSON-friendly nested dict keyed by DEG target space.
    """
    ctrl_spaces = _project_raw_to_spaces(control_raw)
    truth_spaces = _project_raw_to_spaces(truth_raw)
    pred_spaces = _project_predictions_to_spaces(
        predicted_model_space, model_output_space,
    )

    # Stack every space's six arrays into one (n_spaces * 6, n_genes) tensor: one host transfer.
    stacked_list: list[torch.Tensor] = []
    for space in DEG_SPACES:
        arrs = _compute_deg_detail_arrays_gpu(
            ctrl_spaces[space], truth_spaces[space], pred_spaces[space],
        )
        stacked_list.extend(a.to(torch.float32) for a in arrs)

    batch_host = torch.stack(stacked_list, dim=0).detach().cpu().numpy()

    out: dict[str, dict[str, Any]] = {}
    n_arr = len(_DEG_DETAIL_ARRAY_KEYS)
    for s_idx, space in enumerate(DEG_SPACES):
        block: dict[str, Any] = {
            "fdr_threshold": float(fdr_threshold),
            "log2_fc_space": "raw",  # parquet schema column; LFC is always on count means
        }
        base = s_idx * n_arr
        for a_idx, key in enumerate(_DEG_DETAIL_ARRAY_KEYS):
            block[key] = batch_host[base + a_idx].tolist()
        out[space] = block
        q_t = np.asarray(block["q_truth_vs_ctrl"])
        q_p = np.asarray(block["q_pred_vs_ctrl"])
        log_detail(
            logger, f"deg_detail_{space}",
            "n_sig_truth=%d n_sig_pred=%d",
            int((q_t < fdr_threshold).sum()),
            int((q_p < fdr_threshold).sum()),
        )
    return out
