"""Balanced equal-contribution subsample across plates, without replacement.

A+B (k=2): ``N = 2 * (min(n_a, n_b) // 2)``.
14-plate:  ``N = 14 * (min_p n_p // 14)``.
Do not cap N for the paper protocol. ``max_share`` is a debug throttle.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def half_count(n_a: int, n_b: int, max_half: int | None = None) -> int:
    """Cells to draw from *each* plate. Raises if a plate has <2 cells."""
    n_a, n_b = int(n_a), int(n_b)
    if n_a < 0 or n_b < 0:
        raise ValueError(f"cell counts must be >=0; got n_a={n_a}, n_b={n_b}")
    h = min(n_a, n_b) // 2
    if max_half is not None:
        cap = int(max_half)
        if cap < 1:
            raise ValueError(f"max_half must be >=1, got {max_half}")
        h = min(h, cap)
    if h < 1:
        raise ValueError(
            f"need >=2 generated cells on each plate to draw N/2; got n_a={n_a}, n_b={n_b}"
        )
    return h


def draw_half_half_indices(
    n_a: int, n_b: int, rng: np.random.Generator, max_half: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Without-replacement indices into plate A and plate B, length half_count each."""
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be np.random.Generator, got {type(rng)!r}")
    h = half_count(n_a, n_b, max_half=max_half)
    ia = rng.choice(n_a, size=h, replace=False)
    ib = rng.choice(n_b, size=h, replace=False)
    return ia.astype(np.int64, copy=False), ib.astype(np.int64, copy=False)


def share_count(
    counts: Sequence[int], n_plates: int, max_share: int | None = None
) -> int:
    """Cells to draw from *each* plate. Raises if any plate has too few."""
    ns = [int(n) for n in counts]
    if n_plates < 2:
        raise ValueError(f"n_plates must be >=2, got {n_plates}")
    if len(ns) != n_plates:
        raise ValueError(f"need {n_plates} counts, got {len(ns)}: {ns}")
    if any(n < 0 for n in ns):
        raise ValueError(f"cell counts must be >=0; got {ns}")
    h = min(ns) // n_plates
    if max_share is not None:
        cap = int(max_share)
        if cap < 1:
            raise ValueError(f"max_share must be >=1, got {max_share}")
        h = min(h, cap)
    if h < 1:
        raise ValueError(
            f"need >={n_plates} generated cells on each plate to draw "
            f"N/{n_plates}; got {ns}"
        )
    return h


def draw_balanced_indices(
    counts: Sequence[int],
    rng: np.random.Generator,
    max_share: int | None = None,
) -> list[np.ndarray]:
    """Without-replacement indices into each plate, length share_count each."""
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be np.random.Generator, got {type(rng)!r}")
    ns = [int(n) for n in counts]
    h = share_count(ns, len(ns), max_share=max_share)
    out = [rng.choice(n, size=h, replace=False).astype(np.int64, copy=False) for n in ns]
    return out
