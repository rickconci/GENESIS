"""Linear ridge: the perturbation-mean table, ridge-fit in a gene PCA basis.

Y[:, j] = shift_j, the perturbation-mean shift of column j = (kept..., perturbation...), with
b the row mean of Y; solve Y ~ G K P^T + b 1^T for K by ridge and predict control + G K P_j + b.
Tahoe's columns are (drug, dose); PBMC's are (cell type, cytokine).

Gene basis
----------
With identity factors (G = I, P = I) the ridge shrinks every perturbation-specific delta
by exactly 1/(1+lambda)^2 (4x at lambda = 1, measured 4.00 on every arm): a handicapped
copy of perturbation_mean. The reference implementation (Ahlmann-Eltze, Huber & Anders,
Nature Methods 2025, run_linear_pretrained_model.R) has the same ``solve_y_axb`` but passes
G = top PCs of the train pseudobulk with lambda = 0.1, as ``gene_pca_basis`` does here, so
the prediction is a genuine rank-``pca_dim`` linear model. P stays one-hot: their
perturbation-as-gene embedding needs CRISPR targets, and our splits hold out combinations,
so every test perturbation is in train.
"""
from __future__ import annotations

import logging
import time
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from src.baselines.roles import Arms

logger = logging.getLogger(__name__)


# 1. The ridge solve and the gene basis it runs in

def _solve_ridge_factors(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    ridge_lambda: float,
) -> NDArray[np.float64]:
    """Return ``X`` solving ``(a.T @ a + λ I) X = b`` (``a`` is ``(n, d)``)."""
    reg = float(ridge_lambda)
    ata = a.T @ a
    d = ata.shape[0]
    lhs = ata + reg * np.eye(d, dtype=np.float64)
    return np.linalg.solve(lhs, b)


def _ridge_inv_ptp(p_mat: NDArray[np.float64], ridge_lambda: float) -> NDArray[np.float64]:
    """Return ``(P^T P + λ I)^{-1}`` for ``P`` shaped ``(n_p, d_p)``."""
    reg = float(ridge_lambda)
    d_p = p_mat.shape[1]
    lhs = p_mat.T @ p_mat + reg * np.eye(d_p, dtype=np.float64)
    return np.linalg.solve(lhs, np.eye(d_p, dtype=np.float64))


def gene_pca_basis(means: Sequence[np.ndarray], dim: int) -> np.ndarray:
    """Top-``dim`` PC scores of the (genes x train-conditions) pseudobulk matrix.

    Mirrors ``prcomp_irlba(as.matrix(assay(train_data, "X")), n = dim)$x`` from the
    reference benchmark: observations are GENES, variables are train conditions, and
    variables are mean-centred (R's ``center = TRUE``, ``scale. = FALSE``).

    Algebra: for column-centred ``X`` (g x c), gene scores are ``U S`` from
    ``X = U S V^T``. When ``g >> c`` (10k-gene Tahoe), forming ``X X^T`` (g x g) and
    ``eigh`` is the wrong side -- use the thin Gram ``X^T X`` (c x c) and recover
    scores as ``X V`` (same eigenvectors up to sign). Prefer torch CUDA when
    available; fall back to numpy on CPU, loudly.

    Returns ``(n_genes, d)`` float64, where ``d = min(dim, rank)``.
    """
    if not len(means):
        raise ValueError("cannot build a PCA gene basis: no train conditions were read")
    # (n_genes, n_conditions)
    mat = np.stack(list(means), axis=1).astype(np.float64, copy=False)
    mat = mat - mat.mean(axis=0, keepdims=True)  # centre each condition (variable)
    g, c = mat.shape
    k = max(int(dim), 1)

    t0 = time.time()
    use_thin = c <= g
    device_s = "numpy"
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_s = str(device)
        xt = torch.as_tensor(mat, dtype=torch.float64, device=device)
        if use_thin:
            # (c, c) = V S^2 V^T; scores = X V (= U S)
            gram = xt.T @ xt
            evals_t, evecs_t = torch.linalg.eigh(gram)
            order = torch.argsort(evals_t, descending=True)[:k]
            evals_d = torch.clamp(evals_t[order], min=0.0)
            scores_t = xt @ evecs_t[:, order]
            var_total = float(torch.clamp(evals_t, min=0.0).sum().item())
            evals_keep = evals_d
        else:
            # g < c: gene Gram is the thin side
            gram = xt @ xt.T
            evals_t, evecs_t = torch.linalg.eigh(gram)
            order = torch.argsort(evals_t, descending=True)[:k]
            evals_d = torch.clamp(evals_t[order], min=0.0)
            scores_t = evecs_t[:, order] * torch.sqrt(evals_d)[None, :]
            var_total = float(torch.clamp(evals_t, min=0.0).sum().item())
            evals_keep = evals_d
        keep = evals_keep > (evals_keep[0] * 1e-12 if float(evals_keep[0]) > 0 else 0.0)
        if not bool(torch.all(keep)):
            logger.warning("gene PCA | requested dim=%d but only %d components are non-degenerate",
                        dim, int(keep.sum().item()))
            scores_t = scores_t[:, keep]
            evals_keep = evals_keep[keep]
        scores = scores_t.detach().cpu().numpy()
        explained = float(evals_keep.sum().item() / var_total) if var_total > 0 else float("nan")
    except Exception as exc:  # noqa: BLE001 - fall back rather than abort a long fit
        logger.warning("gene PCA | torch path failed (%s); falling back to numpy thin eigh", exc)
        device_s = "numpy"
        if use_thin:
            gram = mat.T @ mat
            evals, evecs = np.linalg.eigh(gram)
            order = np.argsort(evals)[::-1][:k]
            evals_d = np.clip(evals[order], 0.0, None)
            scores = mat @ evecs[:, order]
            var_total = float(np.clip(evals, 0.0, None).sum())
        else:
            gram = mat @ mat.T
            evals, evecs = np.linalg.eigh(gram)
            order = np.argsort(evals)[::-1][:k]
            evals_d = np.clip(evals[order], 0.0, None)
            scores = evecs[:, order] * np.sqrt(evals_d)[np.newaxis, :]
            var_total = float(np.clip(evals, 0.0, None).sum())
        keep = evals_d > (evals_d[0] * 1e-12 if evals_d[0] > 0 else 0.0)
        if not keep.all():
            logger.warning("gene PCA | requested dim=%d but only %d components are non-degenerate",
                        dim, int(keep.sum()))
            scores = scores[:, keep]
            evals_d = evals_d[keep]
        explained = float(evals_d.sum() / var_total) if var_total > 0 else float("nan")

    logger.info(
        "gene PCA | basis=(%d, %d) from %d train conditions | variance explained=%.3f | "
        "device=%s thin=%s in %.2fs",
        scores.shape[0], scores.shape[1], c, explained, device_s, use_thin, time.time() - t0,
    )
    return np.ascontiguousarray(scores, dtype=np.float64)


# 2. Fit the ridge: one fitted shift per column, so prediction is a lookup

def fit(arms: Arms, shift: dict[tuple, np.ndarray], *, pca_dim: int, ridge_lambda: float) -> dict[tuple, np.ndarray]:
    """Ridge of the perturbation-mean ``shift`` table, G from the train arm means."""
    cols = sorted(shift)
    g_mat = gene_pca_basis(list(arms.mean), pca_dim)
    y = np.zeros((g_mat.shape[0], len(cols)), dtype=np.float64)
    for j, col in enumerate(cols):
        y[:, j] = np.asarray(shift[col], dtype=np.float64).ravel()
    b = y.mean(axis=1)
    y_c = y - b[:, np.newaxis]
    p_mat = np.eye(len(cols), dtype=np.float64)
    k_mat = _solve_ridge_factors(g_mat, g_mat.T @ y_c @ p_mat, ridge_lambda) @ _ridge_inv_ptp(p_mat, ridge_lambda)
    fitted = {col: g_mat @ (k_mat @ p_mat[j]) + b for j, col in enumerate(cols)}
    # The point of the PCA basis is that the perturbation-specific shift survives the ridge:
    # log how much does, so a degenerate refit is obvious.
    kept = np.mean([np.linalg.norm(fitted[c] - b) for c in cols])
    full = np.mean(np.linalg.norm(y_c, axis=0))
    logger.info("linear_ridge | d_g=%d lambda=%g columns=%d | retained fraction of the "
                "perturbation-specific shift %.3f", g_mat.shape[1], ridge_lambda, len(cols), kept / max(full, 1e-12))
    return fitted
