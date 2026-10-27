"""Uniform random subsample of train row indices to a fixed cell budget."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


def subsample_train_cells(
    train_idx: np.ndarray,
    *,
    target_n: int,
    seed: int = 0,
    keep_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Subsample ``train_idx`` to at most ``target_n`` rows without replacement.

    ``keep_mask`` (bool, aligned with ``train_idx``) marks rows kept unconditionally, as HBSS
    keeps controls; ``target_n`` then budgets only the unmarked rows. Without it the draw is the
    one the shipped splits were made with, bit for bit.

    Returns ``(out_idx, summary)``. When ``target_n`` covers the budgeted rows, returns
    every row with ``applied=False``.
    """
    idx = np.asarray(train_idx, dtype=np.int64)
    fixed = idx[:0]
    if keep_mask is not None:
        keep_mask = np.asarray(keep_mask, dtype=bool)
        if keep_mask.shape != idx.shape:
            raise ValueError(f"keep_mask shape {keep_mask.shape} != train_idx shape {idx.shape}.")
        fixed, idx = idx[keep_mask], idx[~keep_mask]
    n_before = int(idx.size)
    target = int(target_n)
    if target <= 0:
        raise ValueError(f"target_n must be positive; got {target_n!r}.")
    applied = target < n_before
    chosen = np.random.default_rng(int(seed)).choice(idx, size=target, replace=False) if applied else idx
    out = np.sort(np.concatenate([chosen.astype(np.int64, copy=False), fixed]))
    summary = {
        "target_n": target,
        "n_before": n_before,
        "n_after": int(chosen.size),
        "seed": int(seed),
        "applied": applied,
    }
    if keep_mask is not None:
        summary["n_kept_fixed"] = int(fixed.size)
    return out, summary


def allocate_cell_budget(weights: np.ndarray, target_n: int, seed: int) -> np.ndarray:
    """One multivariate-hypergeometric draw of ``target_n`` cells across groups.

    ``weights[i]`` is the number of cells in group ``i``. The result is what
    ``subsample_train_cells`` does to cell rows, summed back to those groups:
    each count is at most its weight, and the counts sum to
    ``min(target_n, sum(weights))``. Group order is the caller's; the seed
    walks groups left to right.
    """
    w = np.asarray(weights, dtype=np.int64)
    if w.ndim != 1:
        raise ValueError(f"weights must be 1-d; got shape {w.shape}")
    if np.any(w < 0):
        raise ValueError("cell weights must be non-negative")
    total = int(w.sum())
    target = int(target_n)
    if target <= 0:
        raise ValueError(f"target_n must be positive; got {target_n!r}.")
    if target >= total:
        return w.copy()
    rng = np.random.default_rng(int(seed))
    out = np.zeros(w.size, dtype=np.int64)
    remaining_pop = total
    remaining_draw = target
    for i, wi in enumerate(w.tolist()):
        if remaining_draw == 0 or wi == 0:
            remaining_pop -= wi
            continue
        drawn = int(rng.hypergeometric(
            ngood=wi, nbad=remaining_pop - wi, nsample=remaining_draw,
        ))
        out[i] = drawn
        remaining_pop -= wi
        remaining_draw -= drawn
    if int(out.sum()) != target:
        raise RuntimeError(f"cell budget drew {int(out.sum())}, wanted {target}")
    return out


def subsample_per_condition(
    train_idx: np.ndarray,
    groups: np.ndarray,
    *,
    n_per: int,
    seed: int = 0,
    keep_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Keep ``n_per`` cells from each group. A shorter group is kept whole.

    ``groups`` is aligned with ``train_idx``. ``keep_mask`` rows (controls) are
    not grouped and are all returned. Groups are walked in id order, so the
    seed does not depend on row order beyond the order inside a group.
    """
    idx = np.asarray(train_idx, dtype=np.int64)
    groups = np.asarray(groups)
    if groups.shape != idx.shape:
        raise ValueError(f"groups shape {groups.shape} != train_idx shape {idx.shape}.")
    n_per = int(n_per)
    if n_per <= 0:
        raise ValueError(f"n_per must be positive; got {n_per!r}.")
    fixed = idx[:0]
    if keep_mask is not None:
        keep_mask = np.asarray(keep_mask, dtype=bool)
        if keep_mask.shape != idx.shape:
            raise ValueError(f"keep_mask shape {keep_mask.shape} != train_idx shape {idx.shape}.")
        fixed, idx, groups = idx[keep_mask], idx[~keep_mask], groups[~keep_mask]
    n_before = int(idx.size)
    if n_before:
        order = np.argsort(groups, kind="mergesort")
        idx, groups = idx[order], groups[order]
        bounds = np.r_[0, np.flatnonzero(np.diff(groups)) + 1, n_before]
        rng = np.random.default_rng(int(seed))
        parts, n_short = [], 0
        for s, e in zip(bounds[:-1], bounds[1:]):
            n = int(e - s)
            if n <= n_per:
                parts.append(idx[s:e])
                n_short += int(n < n_per)
            else:
                parts.append(idx[s + rng.choice(n, size=n_per, replace=False)])
        drawn = np.concatenate(parts)
        n_groups = int(bounds.size - 1)
    else:
        drawn, n_short, n_groups = idx, 0, 0
    out = np.sort(np.concatenate([drawn.astype(np.int64, copy=False), fixed]))
    summary = {
        "per_condition": n_per, "n_groups": n_groups, "n_groups_short": int(n_short),
        "n_before": n_before, "n_after": int(drawn.size), "seed": int(seed),
        "applied": bool(n_before and int(drawn.size) < n_before),
    }
    if keep_mask is not None:
        summary["n_kept_fixed"] = int(fixed.size)
    return out, summary
