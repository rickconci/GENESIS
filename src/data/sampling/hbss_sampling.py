"""Generic Hierarchical Balanced Stratified Sampling (HBSS).

1. **N-dimensional IPF** computes a fractional selection weight tensor
   ``Q`` of shape ``(|A_1|, ..., |A_k|)`` over the balance axes
   ``A_1, ..., A_k`` (e.g. cell_name × drug × dose for Tahoe, donor ×
   cytokine for Parse).
2. **Largest-remainder rounding** turns the per-slice fractional
   marginals (slice = product of all balance axes except the candidate
   axis) into integer quotas summing exactly to the target count.
3. **Diversity-aware greedy** picks the candidate-axis entries inside
   each slice, penalising repetition of parent-group labels supplied
   via :attr:`HBSSSpec.diversity_axes`.

Public entry point: :func:`run_hbss`, which consumes a deduplicated
triplet DataFrame and an :class:`HBSSSpec` and returns the selected
balance-axis tuples plus diagnostics. No disk I/O.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..vocab import obs_column_to_canonical_codes, obs_values_to_categorical
from ..splits.data_class import HBSSSpec

logger = logging.getLogger(__name__)


# Substring sentinels in a diversity parent column that should not collapse
# into a single bucket (otherwise the diversity penalty would aggressively
# avoid all of them). When the parent label matches one of these (or is
# NaN), we replace it with a per-candidate singleton bucket — see
# ``HBSSSpec.ambiguous_diversity_to_singleton``.
_AMBIGUOUS_DIVERSITY_LABELS: frozenset[str] = frozenset(
    {"unclear", "nan", "none", ""}
)


# =============================================================================
# Diagnostic + result containers
# =============================================================================


@dataclass
class IPFResult:
    """Output of :func:`ipf_nd`.

    Attributes:
        Q: Float tensor of shape ``levels_per_axis`` with selection weights
            in ``[0, 1]``. Zero outside the observed support.
        axes: Balance-axis column names in tensor-axis order.
        levels: For each axis, the sorted list of unique observed values
            (so ``Q[i_1, ..., i_k]`` corresponds to
            ``(levels[0][i_1], ..., levels[k-1][i_k])``).
        n_iter: Number of IPF passes that were actually run.
        max_delta: Final ``max|Q^(t+1) - Q^(t)|`` value at convergence/stop.
        converged: Whether ``max_delta < tol`` was reached before
            ``max_iter``.
    """

    Q: np.ndarray
    axes: List[str]
    levels: List[List[Any]]
    n_iter: int
    max_delta: float
    converged: bool


@dataclass
class HBSSResult:
    """Output of :func:`run_hbss`.

    Attributes:
        selected: One row per kept balance-axis tuple. Columns are exactly
            ``cfg.balance_axes`` plus a ``prob`` column with the IPF
            weight of the kept tuple. Numeric columns are rounded to 6
            decimals so they match :func:`build_tuple_filter`'s output.
        balance_report: Per-axis CV / chi-square diagnostics over both
            balance and diversity axes (see :func:`balance_report`).
        meta: Run metadata (target / selected counts, IPF convergence,
            wall-clock breakdown, axis dimensions).
    """

    selected: pd.DataFrame
    balance_report: pd.DataFrame
    meta: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Helpers
# =============================================================================


def _balance_axis_keys(series: pd.Series) -> np.ndarray:
    """Stable categorical keys for one balance-axis column."""
    return obs_values_to_categorical(series)


def _build_diversity_groups(
    levels: Sequence[Any],
    label_lookup: Dict[Any, Any],
    *,
    ambiguous_to_singleton: bool,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Encode parent labels along one axis into integer group ids.

    Args:
        levels: Sorted unique candidate labels along the axis.
        label_lookup: Mapping from candidate label -> parent label (may
            contain NaN / ambiguous values).
        ambiguous_to_singleton: If True, NaN/ambiguous parents are
            replaced by a per-candidate singleton so each such candidate
            occupies its own bucket. If False, all ambiguous parents
            collapse into one ``__AMBIGUOUS__`` bucket.

    Returns:
        ``(group_ids, label_lut)`` where ``group_ids[i]`` is the integer
        bucket id of ``levels[i]`` and ``label_lut`` maps canonical
        bucket label -> integer (for diagnostics).
    """
    label_lut: Dict[str, int] = {}
    group_ids = np.empty(len(levels), dtype=np.int64)
    for i, candidate in enumerate(levels):
        raw = label_lookup.get(candidate, None)
        is_ambiguous = (
            raw is None
            or (isinstance(raw, float) and np.isnan(raw))
            or (isinstance(raw, str) and raw.strip().lower() in _AMBIGUOUS_DIVERSITY_LABELS)
        )
        if is_ambiguous and ambiguous_to_singleton:
            canonical = f"__SINGLETON__::{candidate}"
        elif is_ambiguous:
            canonical = "__AMBIGUOUS__"
        else:
            canonical = str(raw)
        if canonical not in label_lut:
            label_lut[canonical] = len(label_lut)
        group_ids[i] = label_lut[canonical]
    return group_ids, label_lut


# =============================================================================
# Phase 1: N-dimensional IPF
# =============================================================================


def ipf_nd(
    triplets: pd.DataFrame,
    n_target: int,
    *,
    balance_axes: Sequence[str],
    max_iter: int = 300,
    tol: float = 1e-4,
) -> IPFResult:
    """N-dimensional Iterative Proportional Fitting with capacity clipping.

    Builds a non-negative tensor ``Q`` whose marginals approximate
    ``n_target / |A_a|`` along each balance axis ``A_a`` and whose
    support is contained in the observed combinations of
    ``triplets[balance_axes]``. Each entry is clipped to ``<= 1``.

    Reduces to a 3D IPF for Tahoe's three balance axes and to a 2D IPF for
    Parse-style ``[donor, cytokine]`` axes.

    Args:
        triplets: One row per observed unique combination over
            ``balance_axes``. Duplicates are tolerated (only the support
            pattern is used).
        n_target: Target number of triplets to select. Must be in
            ``(0, len(triplets.drop_duplicates(subset=balance_axes))]``.
        balance_axes: Column names. The tensor axis order matches the
            sequence order.
        max_iter: Maximum number of IPF sweeps.
        tol: Convergence threshold on the max element-wise change.

    Returns:
        :class:`IPFResult`.

    Raises:
        ValueError: For invalid ``n_target`` or missing columns.
    """
    missing = [c for c in balance_axes if c not in triplets.columns]
    if missing:
        raise ValueError(
            f"ipf_nd: balance_axes {missing!r} are not present in triplets."
        )
    unique = triplets.drop_duplicates(subset=list(balance_axes))
    n_obs = len(unique)
    if n_target <= 0:
        raise ValueError(f"ipf_nd: n_target must be > 0, got {n_target}.")
    if n_target > n_obs:
        raise ValueError(
            f"ipf_nd: n_target={n_target} exceeds number of observed unique "
            f"tuples ({n_obs}) over {list(balance_axes)!r}."
        )

    # Stable sorted level lists per axis; numeric columns sort by value,
    # strings sort lexicographically. Build a coordinate per row.
    levels: List[List[Any]] = []
    idx_maps: List[Dict[Any, int]] = []
    for col in balance_axes:
        keys = _balance_axis_keys(unique[col])
        vals = sorted(dict.fromkeys(keys))
        levels.append(vals)
        idx_maps.append({v: i for i, v in enumerate(vals)})

    shape = tuple(len(lvl) for lvl in levels)
    capacity = np.zeros(shape, dtype=np.float64)

    coords: List[np.ndarray] = []
    for col, idx_map in zip(balance_axes, idx_maps):
        keys = _balance_axis_keys(unique[col])
        coords.append(np.fromiter((idx_map[k] for k in keys), dtype=np.int64, count=len(keys)))
    capacity[tuple(coords)] = 1.0

    n_obs_total = float(capacity.sum())
    Q = capacity * (n_target / n_obs_total)
    targets = [n_target / s for s in shape]
    ndim = Q.ndim

    last_delta = float("inf")
    for it in range(max_iter):
        Q_prev = Q.copy()
        for axis in range(ndim):
            other_axes = tuple(a for a in range(ndim) if a != axis)
            sums = Q.sum(axis=other_axes)
            factors = np.where(
                sums > 0, targets[axis] / np.maximum(sums, 1e-30), 0.0
            )
            broadcast_shape = [1] * ndim
            broadcast_shape[axis] = shape[axis]
            Q = np.minimum(Q * factors.reshape(broadcast_shape), capacity)

        last_delta = float(np.abs(Q - Q_prev).max())
        if last_delta < tol:
            logger.info(
                "IPF converged after %d iters (max_delta=%.2e < tol=%.0e)",
                it + 1, last_delta, tol,
            )
            return IPFResult(Q, list(balance_axes), levels, it + 1, last_delta, True)

    logger.warning(
        "IPF did NOT converge in %d iters (max_delta=%.2e, tol=%.0e).",
        max_iter, last_delta, tol,
    )
    return IPFResult(Q, list(balance_axes), levels, max_iter, last_delta, False)


# =============================================================================
# Phase 2: largest-remainder rounding
# =============================================================================


def largest_remainder_round(float_vals: np.ndarray, target_sum: int, kind: str = "stable") -> np.ndarray:
    """Round real-valued quotas to integers summing exactly to ``target_sum``.

    Hamilton / largest-remainder: floor everything, then add (subtract)
    one to the entries with the largest (smallest) fractional remainders
    until the integer sum matches ``target_sum``.
    """
    floors = np.floor(float_vals).astype(np.int64)
    remainder = float_vals - floors
    deficit = int(target_sum - floors.sum())
    # Exact ties are common (IPF repeats remainders). kind="stable" gives them to the lowest index;
    # "quicksort" (HBSSSpec.tie_break="legacy") orders them per CPU, AVX-512 vs AVX2.
    if deficit > 0:
        order = np.argsort(-remainder, kind=kind)[:deficit]
        floors[order] += 1
    elif deficit < 0:
        eligible = np.where(floors > 0)[0]
        if len(eligible) < -deficit:
            raise ValueError(
                "largest_remainder_round: cannot reduce floors to match "
                "target_sum without going negative."
            )
        order = eligible[np.argsort(remainder[eligible], kind=kind)][:(-deficit)]
        floors[order] -= 1
    return floors


# =============================================================================
# Phase 3: diversity-aware greedy
# =============================================================================


def diversity_greedy_select(
    probs: np.ndarray,
    group_ids: np.ndarray,
    quota: int,
    *,
    lam: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pick ``quota`` indices, trading IPF score against group diversity.

    At each step the alive-candidate score is

        score(i) = max(probs[i] - lam * group_counts[group_ids[i]], 0)

    and one alive index is sampled with probability proportional to
    ``score``. When all alive scores collapse to zero we fall back to a
    uniform draw — see ``docs/HBSS_sampling_algorithm.md`` §3.3 for the
    rationale.
    """
    n = len(probs)
    if quota >= n:
        return np.arange(n, dtype=np.int64)
    if quota <= 0:
        return np.empty(0, dtype=np.int64)

    n_groups = int(group_ids.max()) + 1
    group_counts = np.zeros(n_groups, dtype=np.int64)
    alive = np.ones(n, dtype=bool)
    selected = np.empty(quota, dtype=np.int64)

    for k in range(quota):
        score = probs - lam * group_counts[group_ids]
        weight = np.where(alive, np.maximum(score, 0.0), 0.0)
        total = float(weight.sum())
        if total > 0.0:
            best = int(rng.choice(n, p=weight / total))
        else:
            alive_idx = np.where(alive)[0]
            best = int(rng.choice(alive_idx))
        selected[k] = best
        group_counts[group_ids[best]] += 1
        alive[best] = False

    return selected


# =============================================================================
# Top-level orchestration
# =============================================================================


@dataclass
class HBSSEncoding:
    """Integer-code view of obs rows for fast HBSS triplet build + row filter.

    ``combined[i]`` is the raveled balance-axis tuple code for encoded row ``i``.
    ``triplets`` holds one row per observed unique tuple.
    """

    balance_axes: List[str]
    shapes: Tuple[int, ...]
    labels: List[np.ndarray]
    combined: np.ndarray
    triplets: pd.DataFrame


def encode_hbss_obs(
    obs: pd.DataFrame,
    balance_axes: Sequence[str],
    diversity_axes: Dict[str, str],
    *,
    row_indices: np.ndarray | None = None,
) -> HBSSEncoding:
    """Encode balance axes as int64 codes; build unique triplet table.

    When ``row_indices`` is set, only those rows are encoded (column views,
    no full multi-column ``DataFrame.copy()``).
    """
    missing_balance = [c for c in balance_axes if c not in obs.columns]
    if missing_balance:
        raise ValueError(
            f"HBSS: balance_axes {missing_balance!r} are not present in obs. "
            f"Available columns: {list(obs.columns)!r}"
        )
    parent_cols = sorted({p for p in diversity_axes.values()})
    missing_parents = [p for p in parent_cols if p not in obs.columns]
    if missing_parents:
        raise ValueError(
            f"HBSS: diversity parent columns {missing_parents!r} are not "
            f"present in obs. Check metadata joins."
        )

    code_arrays: List[np.ndarray] = []
    label_arrays: List[np.ndarray] = []
    shapes: List[int] = []
    for col in balance_axes:
        series = obs[col].iloc[row_indices] if row_indices is not None else obs[col]
        codes, labels = obs_column_to_canonical_codes(series)
        code_arrays.append(codes)
        label_arrays.append(labels)
        shapes.append(int(len(labels)))

    combined = np.ravel_multi_index(tuple(code_arrays), tuple(shapes))
    uniq, first_idx = np.unique(combined, return_index=True)
    axis_codes = np.unravel_index(uniq, tuple(shapes))

    triplet_data: Dict[str, Any] = {
        ax: label_arrays[i][axis_codes[i]] for i, ax in enumerate(balance_axes)
    }
    extra_parents = [p for p in parent_cols if p not in balance_axes]
    if extra_parents:
        if row_indices is not None:
            source_rows = row_indices[first_idx]
        else:
            source_rows = first_idx
        for p in extra_parents:
            triplet_data[p] = obs[p].iloc[source_rows].to_numpy()

    triplets = pd.DataFrame(triplet_data)[
        list(balance_axes) + extra_parents
    ].reset_index(drop=True)

    return HBSSEncoding(
        balance_axes=list(balance_axes),
        shapes=tuple(shapes),
        labels=label_arrays,
        combined=combined,
        triplets=triplets,
    )


def selected_combined_keys(
    selected: pd.DataFrame,
    encoding: HBSSEncoding,
) -> np.ndarray:
    """Ravel selected balance-axis tuples under ``encoding`` label maps."""
    sel_codes: List[np.ndarray] = []
    for i, ax in enumerate(encoding.balance_axes):
        canon = obs_values_to_categorical(selected[ax])
        lut = {encoding.labels[i][j]: j for j in range(len(encoding.labels[i]))}
        try:
            sel_codes.append(
                np.fromiter(
                    (lut[k] for k in canon), dtype=np.int64, count=len(selected),
                )
            )
        except KeyError as exc:
            raise ValueError(
                f"selected_combined_keys: value {exc.args[0]!r} on axis {ax!r} "
                "is absent from the encoding label map — balance_axes key-policy "
                "mismatch between selection and encode_hbss_obs.",
            ) from exc
    return np.ravel_multi_index(tuple(sel_codes), encoding.shapes)


def _build_triplet_table(
    obs: pd.DataFrame,
    balance_axes: Sequence[str],
    diversity_axes: Dict[str, str],
) -> pd.DataFrame:
    """Deduplicate ``obs`` over ``balance_axes`` and carry parent columns."""
    return encode_hbss_obs(obs, balance_axes, diversity_axes).triplets


def balance_report(
    selected: pd.DataFrame,
    *,
    balance_axes: Sequence[str],
    diversity_axes: Dict[str, str],
) -> pd.DataFrame:
    """Per-axis CV / chi-square diagnostics over the selected triplets.

    The ``role`` column distinguishes balance axes (where HBSS targets
    uniform marginals) from diversity parent axes (which HBSS only
    spreads within-slice — non-uniformity here is expected, not a
    defect).
    """
    from scipy.stats import chisquare

    rows: List[Dict[str, Any]] = []
    diversity_parents = sorted({p for p in diversity_axes.values()})
    audited_axes = [(c, "balance") for c in balance_axes] + [
        (c, "diversity") for c in diversity_parents if c not in balance_axes
    ]

    for col, role in audited_axes:
        if col not in selected.columns:
            continue
        counts = selected[col].value_counts(dropna=False)
        if len(counts) == 0 or counts.sum() == 0:
            continue
        try:
            stat, p = chisquare(counts.values)
        except ValueError:
            stat, p = float("nan"), float("nan")
        rows.append(
            {
                "axis": col,
                "role": role,
                "n_categories": int(len(counts)),
                "mean_count": round(float(counts.mean()), 2),
                "min_count": int(counts.min()),
                "max_count": int(counts.max()),
                "cv": round(float(counts.std(ddof=0) / max(counts.mean(), 1e-12)), 4),
                "chi2": round(float(stat), 2) if np.isfinite(stat) else stat,
                "p_uniform": round(float(p), 4) if np.isfinite(p) else p,
            }
        )
    return pd.DataFrame(rows)


def run_hbss(
    obs: pd.DataFrame,
    cfg: HBSSSpec,
    *,
    fraction: float,
    encoding: HBSSEncoding | None = None,
) -> HBSSResult:
    """Run HBSS end-to-end against the given observation table.

    Args:
        obs: Pre-enriched observation table. Must contain every column
            referenced by ``cfg.balance_axes`` and by the parent values of
            ``cfg.diversity_axes``. Rows can be repeated (one row per
            cell, etc.) — the algorithm deduplicates internally over the
            balance axes.
        cfg: Validated :class:`HBSSSpec` (see ``__post_init__`` checks).
        fraction: Fraction in ``(0, 1)`` of unique tuples to keep.
        encoding: Optional precomputed :class:`HBSSEncoding` (avoids a
            second encode pass when the caller also filters rows).

    Returns:
        :class:`HBSSResult`.

    Raises:
        ValueError: If columns are missing, ``fraction`` is out of range,
            ``round(fraction * n_unique_tuples) <= 0``, IPF fails to
            converge, quota assignment does not sum to ``n_target``, or
            the final selection count differs from ``n_target``.
    """
    if not (0.0 < float(fraction) < 1.0):
        raise ValueError(f"HBSS: fraction must be in (0, 1); got {fraction!r}.")

    t_encode0 = time.perf_counter()
    if encoding is None:
        print("  HBSS: encoding axes...", flush=True)
        encoding = encode_hbss_obs(obs, cfg.balance_axes, cfg.diversity_axes)
        t_encode = time.perf_counter() - t_encode0
        print(
            f"  HBSS: unique triplets = {len(encoding.triplets):,} ({t_encode:.1f}s)",
            flush=True,
        )
    else:
        t_encode = 0.0

    triplets = encoding.triplets
    n_total = len(triplets)
    n_target = int(round(float(fraction) * n_total))
    if n_target <= 0:
        raise ValueError(
            f"HBSS: round(fraction * n_unique_tuples) = 0 (fraction={fraction}, "
            f"n_unique_tuples={n_total}); nothing to sample."
        )
    rng = np.random.default_rng(cfg.seed)

    logger.info(
        "HBSS: fraction=%.4f -> n_target=%d / N=%d (axes=%s, candidate=%s, "
        "lam=%.2f, seed=%d)",
        fraction, n_target, n_total, list(cfg.balance_axes),
        cfg.candidate_axis, cfg.lam, cfg.seed,
    )

    # --- Phase 1 ---
    t0 = time.perf_counter()
    ipf = ipf_nd(
        triplets,
        n_target=n_target,
        balance_axes=cfg.balance_axes,
        max_iter=cfg.max_iter,
        tol=cfg.tol,
    )
    t_ipf = time.perf_counter() - t0
    print(f"  HBSS: IPF done ({t_ipf:.1f}s)", flush=True)
    if not ipf.converged:
        raise ValueError(
            f"HBSS IPF did not converge in {ipf.n_iter} iteration(s) "
            f"(max_delta={ipf.max_delta:.2e}, tol={cfg.tol:.0e}). "
            "Increase train.hbss.max_iter, relax train.hbss.tol, or check "
            "that balance_axes form a feasible support."
        )
    Q = ipf.Q
    candidate_axis_idx = list(cfg.balance_axes).index(cfg.candidate_axis)
    candidate_levels = ipf.levels[candidate_axis_idx]
    other_axes = [
        a for a in range(Q.ndim) if a != candidate_axis_idx
    ]
    other_shape = tuple(Q.shape[a] for a in other_axes)
    other_axis_names = [cfg.balance_axes[a] for a in other_axes]

    # Diversity penalty groups along the candidate axis only.
    parent_col = cfg.diversity_axes.get(cfg.candidate_axis)
    if parent_col is not None:
        candidate_to_parent = (
            triplets.drop_duplicates(subset=[cfg.candidate_axis])
            .set_index(cfg.candidate_axis)[parent_col]
            .to_dict()
        )
        group_ids, group_lut = _build_diversity_groups(
            candidate_levels,
            candidate_to_parent,
            ambiguous_to_singleton=cfg.ambiguous_diversity_to_singleton,
        )
    else:
        # No diversity penalty: every candidate is its own group ->
        # diversity_greedy_select reduces to pure IPF-weighted multinomial.
        group_ids = np.arange(len(candidate_levels), dtype=np.int64)
        group_lut = {f"__SINGLETON__::{c}": i for i, c in enumerate(candidate_levels)}

    # --- Phase 2: per-slice integer quotas ---
    t0 = time.perf_counter()
    r_slice = Q.sum(axis=candidate_axis_idx).reshape(-1)
    sort_kind = "stable" if cfg.tie_break == "stable" else "quicksort"
    if cfg.tie_break == "legacy":
        print("  HBSS: tie_break=legacy: tied quotas follow this CPU's sort (AVX-512 and AVX2 hosts "
              "draw different conditions); set train.conditions.hbss.tie_break: stable for new splits", flush=True)
    q_slice_flat = largest_remainder_round(r_slice, n_target, kind=sort_kind)
    slice_capacity_flat = (Q > 0).sum(axis=candidate_axis_idx).reshape(-1)
    over = q_slice_flat > slice_capacity_flat
    if over.any():
        q_slice_flat = np.minimum(q_slice_flat, slice_capacity_flat)
        residual = n_target - int(q_slice_flat.sum())
        if residual > 0:
            slack = slice_capacity_flat - q_slice_flat
            score = np.where(
                slack > 0,
                r_slice - q_slice_flat.astype(np.float64),
                -np.inf,
            )
            for ix in np.argsort(-score, kind=sort_kind):
                if residual <= 0:
                    break
                if slack[ix] <= 0:
                    continue
                bump = int(min(int(slack[ix]), residual))
                q_slice_flat[ix] += bump
                residual -= bump
        if int(q_slice_flat.sum()) != n_target:
            raise ValueError(
                f"HBSS quota rounding: after capacity capping, slice quotas "
                f"sum to {int(q_slice_flat.sum())} != n_target={n_target}."
            )
    elif int(q_slice_flat.sum()) != n_target:
        raise ValueError(
            f"HBSS quota rounding: slice quotas sum to "
            f"{int(q_slice_flat.sum())} != n_target={n_target}."
        )
    q_slice = q_slice_flat.reshape(other_shape) if other_shape else q_slice_flat.reshape(())
    t_round = time.perf_counter() - t0

    # --- Phase 3: diversity-aware greedy per slice ---
    t0 = time.perf_counter()
    selected_keys: List[Tuple[Any, ...]] = []
    selected_probs: List[float] = []
    n_slices_used = 0

    slice_indices = (
        list(np.ndindex(*other_shape)) if other_shape else [()]
    )
    for slice_coord in slice_indices:
        # Quota for this slice (scalar).
        quota = int(q_slice[slice_coord]) if other_shape else int(q_slice)
        if quota <= 0:
            continue
        # Slice through Q along the candidate axis.
        full_index: List[Any] = [slice(None)] * Q.ndim
        for axis, val in zip(other_axes, slice_coord):
            full_index[axis] = int(val)
        slab = Q[tuple(full_index)]
        cand_mask = slab > 0
        cand_pos = np.where(cand_mask)[0]
        if quota > 0 and cand_pos.size == 0:
            raise ValueError(
                f"HBSS slice {slice_coord} has quota={quota} but no observed "
                "candidates along the candidate axis (Q>0 support is empty)."
            )
        if cand_pos.size == 0:
            continue
        n_slices_used += 1
        picks = diversity_greedy_select(
            probs=slab[cand_pos],
            group_ids=group_ids[cand_pos],
            quota=quota,
            lam=cfg.lam,
            rng=rng,
        )
        for p in picks:
            ci = int(cand_pos[p])
            key: List[Any] = [None] * Q.ndim
            key[candidate_axis_idx] = candidate_levels[ci]
            for axis, val, name in zip(other_axes, slice_coord, other_axis_names):
                key[axis] = ipf.levels[axis][int(val)]
            selected_keys.append(tuple(key))
            selected_probs.append(float(slab[ci]))
    t_greedy = time.perf_counter() - t0

    if len(selected_keys) != n_target:
        raise ValueError(
            f"HBSS selected {len(selected_keys)} tuple(s) != n_target={n_target} "
            f"(residual={n_target - len(selected_keys)})."
        )

    # Build the selected DataFrame in balance-axis column order, then
    # carry parent columns for the diagnostic report.
    selected = pd.DataFrame(selected_keys, columns=list(cfg.balance_axes))
    selected["prob"] = selected_probs
    parent_cols = sorted({p for p in cfg.diversity_axes.values() if p not in cfg.balance_axes})
    if parent_cols:
        join_subset = triplets.drop_duplicates(subset=list(cfg.balance_axes))[
            list(cfg.balance_axes) + parent_cols
        ]
        selected = selected.merge(join_subset, on=list(cfg.balance_axes), how="left")

    rep = balance_report(
        selected, balance_axes=cfg.balance_axes, diversity_axes=cfg.diversity_axes
    )

    meta: Dict[str, Any] = {
        "fraction": float(fraction),
        "lam": float(cfg.lam),
        "seed": int(cfg.seed),
        "max_iter": int(cfg.max_iter),
        "tol": float(cfg.tol),
        "tie_break": cfg.tie_break,
        "balance_axes": list(cfg.balance_axes),
        "candidate_axis": cfg.candidate_axis,
        "diversity_axes": dict(cfg.diversity_axes),
        "n_total_triplets": int(n_total),
        "n_target": int(n_target),
        "n_selected": int(len(selected)),
        "axis_dims": {ax: len(lvl) for ax, lvl in zip(ipf.axes, ipf.levels)},
        "n_diversity_groups": int(len(group_lut)),
        "ipf_iters": int(ipf.n_iter),
        "ipf_max_delta": float(ipf.max_delta),
        "ipf_converged": bool(ipf.converged),
        "n_slices_used": int(n_slices_used),
        "wall_seconds": {
            "encode": round(float(t_encode), 4),
            "ipf": round(float(t_ipf), 4),
            "rounding": round(float(t_round), 4),
            "greedy": round(float(t_greedy), 4),
        },
    }
    return HBSSResult(selected=selected, balance_report=rep, meta=meta)


def build_tuple_filter(
    selected: pd.DataFrame,
    balance_axes: Sequence[str],
) -> pd.MultiIndex:
    """Multi-index of kept balance-axis tuples for ``.isin`` filtering."""
    arrays = [_balance_axis_keys(selected[col]) for col in balance_axes]
    return pd.MultiIndex.from_arrays(arrays, names=list(balance_axes))


__all__ = [
    "HBSSResult",
    "HBSSEncoding",
    "IPFResult",
    "balance_report",
    "build_tuple_filter",
    "diversity_greedy_select",
    "encode_hbss_obs",
    "ipf_nd",
    "largest_remainder_round",
    "run_hbss",
    "selected_combined_keys",
]
