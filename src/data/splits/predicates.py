"""Train/val/test split predicate engine.

Predicate grammar (a "split predicate" in the split YAML):
    * ``dict``  → AND over (column, value(s)); ``"any"`` is a wildcard,
                  ``"false"`` forces the block to match zero rows.
    * ``list``  → OR over AND blocks above.
    * ``None``  → matches every row.

The fast path fuses pure multi-column AND blocks into a single
``np.isin`` pass over a per-row combined int64 key (see
``np.ravel_multi_index``). For 100M Tahoe rows this is ~10× cheaper than
running one ``np.isin`` per column-target pair.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..vocab import to_categorical_str
from .data_class import SplitConfig

ColumnView = Tuple[str, Any, Any]  # ("categorical", codes, categories) | ("float", values, None)
RowTupleCache = Dict[Tuple[str, ...], Tuple[np.ndarray, Tuple[int, ...]]]


def _predicate_columns(
    predicate: Union[Dict[str, Any], List[Dict[str, Any]], None],
) -> set[str]:
    """Union of column names referenced by ``predicate`` (dict or list of dicts)."""
    if not predicate:
        return set()
    blocks = predicate if isinstance(predicate, list) else [predicate]
    return {col for block in blocks for col in block.keys()}


def build_predicate_column_views(
    obs: pd.DataFrame,
    columns: set[str],
) -> Dict[str, ColumnView]:
    """Per-column view for fast predicate evaluation; numeric → float values,
    everything else → (codes, categories). Cast non-categorical strings once."""
    views: Dict[str, ColumnView] = {}
    for col in columns:
        if col not in obs.columns:
            continue
        s = obs[col]
        if isinstance(s.dtype, pd.CategoricalDtype):
            views[col] = ("categorical", s.cat.codes.to_numpy(), s.cat.categories)
        elif pd.api.types.is_float_dtype(s):
            views[col] = ("float", s.fillna(0.0).to_numpy(dtype=np.float64), None)
        else:
            # ``pd.Categorical(...).codes`` is already a numpy array — no
            # ``.to_numpy()``.
            cats = pd.Categorical(s.astype(str))
            views[col] = ("categorical", np.asarray(cats.codes), cats.categories)
    return views


def _present_uniq_indices(uniq: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Indices into sorted ``uniq`` of the float ``targets`` actually present in obs.

    Presence is a TOLERANCE match, never exact float equality. Tahoe's dose column
    is stored float32; widening it to float64 preserves the float32 representation
    error, so an obs 0.05 is 0.05000000074505806 while the YAML target is exactly
    0.05. Powers of two (0.5, 5.0) *are* exactly representable, so exact equality
    would drop ONLY dose=0.05 -- 34% of tahoe-bio test conditions (4211 -> 2786) and
    28% of tahoe-pert (2115 -> 1515) -- with no error and a plausible-looking
    n_conditions.

    rtol=1e-6 sits far above float32's ~6e-8 relative error and far below the 10x
    spacing of real dose levels, so a genuinely absent dose is still dropped. Both
    the fast and slow predicate paths route through here, so they compare doses the
    same way.
    """
    if uniq.size == 0 or targets.size == 0:
        return np.asarray([], dtype=np.int64)
    idx = np.searchsorted(uniq, targets)
    lo = np.clip(idx - 1, 0, uniq.size - 1)
    hi = np.clip(idx, 0, uniq.size - 1)
    nearest = np.where(np.abs(uniq[lo] - targets) <= np.abs(uniq[hi] - targets), lo, hi)
    present = np.isclose(uniq[nearest], targets, rtol=1e-6, atol=1e-12)
    return nearest[present].astype(np.int64, copy=False)


def _target_codes(col: str, val: Any, views: Dict[str, ColumnView]) -> np.ndarray:
    """Allowed literal codes for one predicate column (float values → searchsorted
    bin indices against the column's unique values; categorical → canonical-string
    lookup against ``cat.categories``).

    Values absent from ``obs`` are DROPPED, in both branches. This matters when the
    caller evaluates a predicate against a shard rather than the whole dataset: the
    eval-cache builder goes plate by plate and Tahoe doses are plate-specific, so a
    split asking for ``dose: [0.05, 0.5, 5.0]`` meets plates that hold only
    ``{0.0, 0.05}``. A bare ``searchsorted`` would return the insertion index for
    the missing values — out of range (``ravel_multi_index`` then raises "invalid
    entry in coordinates array") or, worse, the interior code of a *different* dose.
    """
    values = val if isinstance(val, list) else [val]
    kind, obs_vals, categories = views[col]
    if kind == "float":
        targets = np.asarray([float(v) for v in values], dtype=np.float64)
        if targets.size == 0:
            return targets.astype(np.int64)
        return _present_uniq_indices(np.unique(obs_vals), targets)
    lookup = {to_categorical_str(cat): i for i, cat in enumerate(categories)}
    return np.asarray(
        [lookup[to_categorical_str(v)] for v in values if to_categorical_str(v) in lookup],
        dtype=np.int64,
    )


def _row_codes_and_shape(view: ColumnView) -> Tuple[np.ndarray, int]:
    """Per-row integer codes + axis size for ``np.ravel_multi_index``. Mirrors the
    code derivation in ``_target_codes`` so target and row keys land in the
    same coordinate system; -1 categorical codes are remapped to an extra slot.
    """
    kind, obs_vals, categories = view
    if kind == "float":
        uniq = np.unique(obs_vals)
        return np.searchsorted(uniq, obs_vals).astype(np.int64, copy=False), int(len(uniq))
    row_codes = np.asarray(obs_vals, dtype=np.int64)
    n_codes = int(len(categories))
    if np.any(row_codes < 0):
        row_codes = np.where(row_codes < 0, n_codes, row_codes)
        n_codes += 1
    return row_codes, n_codes


def _block_eligible_for_combined(
    block: Dict[str, Any], views: Dict[str, ColumnView],
) -> bool:
    """True iff every (col, val) is a literal/list literal with the column resolved."""
    if not block:
        return False
    return all(
        str(v).lower() not in ("any", "false") and c in views
        for c, v in block.items()
    )


def _precompute_row_tuple_codes(
    views: Dict[str, ColumnView], column_sets: set[Tuple[str, ...]],
) -> RowTupleCache:
    """Cache per-row combined codes + per-axis shapes for each multi-column set
    that will be reused across predicate blocks (val/test/control share columns)."""
    cache: RowTupleCache = {}
    for cols in column_sets:
        per_col = [_row_codes_and_shape(views[c]) for c in cols]
        code_cols = [rc for rc, _ in per_col]
        shapes = tuple(n for _, n in per_col)
        cache[cols] = (np.ravel_multi_index(tuple(code_cols), shapes), shapes)
    return cache


def _eligible_column_sets(
    predicate: Union[Dict[str, Any], List[Dict[str, Any]], None],
    views: Dict[str, ColumnView],
) -> set[Tuple[str, ...]]:
    """Unique sorted column tuples (size ≥ 2) usable by the combined fast path."""
    if not predicate:
        return set()
    blocks = predicate if isinstance(predicate, list) else [predicate]
    return {
        tuple(sorted(b.keys()))
        for b in blocks
        if len(b) >= 2 and _block_eligible_for_combined(b, views)
    }


def _apply_block_per_column(
    obs: pd.DataFrame,
    block: Dict[str, Any],
    views: Dict[str, ColumnView],
    n_obs: int,
) -> np.ndarray:
    """Slow path: one ``np.isin`` (or float ``isin``) per (column, target) pair."""
    mask = np.ones(n_obs, dtype=bool)
    for col, val in block.items():
        sval = str(val).lower()
        if sval == "any":
            continue
        if sval == "false":
            return np.zeros(n_obs, dtype=bool)
        if col not in obs.columns:
            raise ValueError(f"Column {col!r} not found in obs")
        values = val if isinstance(val, list) else [val]
        kind, obs_vals, categories = views[col]
        if kind == "float":
            # Tolerance-match via _present_uniq_indices, then compare exactly
            # against the obs-side values it selected (safe: they come from obs).
            # A bare np.isin(obs_vals, targets) is the float32-vs-float64 trap.
            targets = np.asarray([float(v) for v in values], dtype=np.float64)
            uniq = np.unique(obs_vals)
            allowed = uniq[_present_uniq_indices(uniq, targets)]
            if allowed.size == 0:
                return np.zeros(n_obs, dtype=bool)
            mask &= np.isin(obs_vals, allowed)
            continue
        target_codes = _target_codes(col, val, views)
        if target_codes.size == 0:
            return np.zeros(n_obs, dtype=bool)
        mask &= np.isin(obs_vals, target_codes)
    return mask


def _apply_block_combined(
    block: Dict[str, Any],
    views: Dict[str, ColumnView],
    n_obs: int,
    *,
    row_tuple_cache: Optional[RowTupleCache] = None,
) -> np.ndarray:
    """Fast path: one ``np.isin`` against the Cartesian product of allowed codes,
    after collapsing each row's (col_1, ..., col_k) tuple into a single int64."""
    cols = tuple(sorted(block.keys()))
    target_cols: List[np.ndarray] = []
    for col in cols:
        codes = _target_codes(col, block[col], views)
        if codes.size == 0:
            return np.zeros(n_obs, dtype=bool)
        target_cols.append(codes)

    if row_tuple_cache and cols in row_tuple_cache:
        row_combined, shape = row_tuple_cache[cols]
    else:
        per_col = [_row_codes_and_shape(views[c]) for c in cols]
        shape = tuple(n for _, n in per_col)
        row_combined = np.ravel_multi_index(tuple(rc for rc, _ in per_col), shape)

    grids = np.meshgrid(*target_cols, indexing="ij")
    target_combined = np.ravel_multi_index(tuple(g.ravel() for g in grids), shape)
    return np.isin(row_combined, target_combined)


def apply_split_predicate(
    obs: pd.DataFrame,
    predicate: Union[Dict[str, Any], List[Dict[str, Any]]],
    *,
    column_views: Optional[Dict[str, ColumnView]] = None,
    row_tuple_cache: Optional[RowTupleCache] = None,
    use_combined: bool = True,
) -> np.ndarray:
    """Boolean row mask for ``predicate`` (dict = AND, list = OR-of-ANDs).

    Special values per key: ``"any"`` (no-op), ``"false"`` (force-empty block).
    Categorical columns match on ``cat.codes``; float columns match on cleaned
    numeric values, so ``dose`` predicates never stringify 100M rows.
    """
    if not predicate:
        return np.ones(len(obs), dtype=bool)
    blocks = [predicate] if isinstance(predicate, dict) else predicate
    if column_views is None:
        column_views = build_predicate_column_views(obs, _predicate_columns(blocks))

    n_obs = len(obs)
    final = np.zeros(n_obs, dtype=bool)
    for block in blocks:
        if use_combined and _block_eligible_for_combined(block, column_views):
            final |= _apply_block_combined(
                block, column_views, n_obs, row_tuple_cache=row_tuple_cache,
            )
        else:
            final |= _apply_block_per_column(obs, block, column_views, n_obs)
    return final


def split_membership_masks(
    obs_eval: pd.DataFrame,
    cfg: SplitConfig,
    control_condition: Optional[Mapping[str, Any]] = None,
    *,
    column_views: Optional[Dict[str, ColumnView]] = None,
    use_combined: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(is_train, is_val, is_test, control_mask)`` row masks for one
    :class:`SplitConfig`. Builds + reuses the column views and the combined
    row-tuple cache across val/test/control so we never re-derive row codes."""
    if column_views is None:
        needed = (
            _predicate_columns(cfg.test)
            | _predicate_columns(cfg.validation)
            | _predicate_columns(control_condition)
        )
        column_views = build_predicate_column_views(obs_eval, needed)

    row_tuple_cache: Optional[RowTupleCache] = None
    if use_combined:
        sets = (
            _eligible_column_sets(cfg.test, column_views)
            | _eligible_column_sets(cfg.validation, column_views)
            | _eligible_column_sets(control_condition, column_views)
        )
        if sets:
            row_tuple_cache = _precompute_row_tuple_codes(column_views, sets)

    def _mask(pred: Any) -> np.ndarray:
        return apply_split_predicate(
            obs_eval, pred,
            column_views=column_views,
            row_tuple_cache=row_tuple_cache,
            use_combined=use_combined,
        )

    is_test = _mask(cfg.test)
    is_val = _mask(cfg.validation)
    control_mask = _mask(control_condition)
    if cfg.controls_to_train:
        is_val &= ~control_mask
        is_test &= ~control_mask
    is_val &= ~is_test
    is_train = ~(is_val | is_test)
    if cfg.train_exclusions:
        is_train &= ~_mask(cfg.train_exclusions)
    return is_train, is_val, is_test, control_mask
