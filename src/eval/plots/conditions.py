"""Condition key resolution, aggregation, and plot filtering."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np

from src.data.data_utils import resolve_plot_aggregate_drop

from src.eval.metrics.reporting import flatten_clustered_metrics
from src.eval.plots.gene_arrays import condition_key_from_mapping


def entry_condition_key(
    entry: Mapping[str, Any],
    variables: Sequence[str],
) -> str:
    """Resolve the canonical condition key for one JSON result row."""
    ck = entry.get("condition_keys")
    if not isinstance(ck, dict):
        raise KeyError("Result row missing condition_keys dict.")
    return condition_key_from_mapping(ck, variables)


def plot_variables(
    variables: Sequence[str],
    *,
    drop_variables: Sequence[str],
) -> tuple[str, ...]:
    """Condition axes kept after dropping replicate/batch dims for plots."""
    drop = set(drop_variables)
    return tuple(v for v in variables if v not in drop)


def mean_metrics_across_groups(
    results: Sequence[Mapping[str, Any]],
    variables: Sequence[str],
    *,
    drop_variables: Sequence[str] = (),
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Average per-row metrics over ``drop_variables`` (e.g. plate).

    Each ``results[]`` row must carry a ``condition_keys`` dict with the full
    scored conditioning tuple.  Returns plot-level keys (all variables except
    ``drop_variables``) mapped to the nanmean of ``metrics`` / ``metrics_std``
    across contributing rows.
    """
    plot_vars = plot_variables(variables, drop_variables=drop_variables)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, entry in enumerate(results):
        ck = entry.get("condition_keys")
        if not isinstance(ck, dict):
            raise KeyError("Result row missing condition_keys dict.")
        pk = condition_key_from_mapping(ck, plot_vars)
        groups[pk].append(i)

    metrics_by: dict[str, dict[str, float]] = {}
    stds_by: dict[str, dict[str, float]] = {}
    for pk, idxs in groups.items():
        m_acc: dict[str, list[float]] = defaultdict(list)
        s_acc: dict[str, list[float]] = defaultdict(list)
        for i in idxs:
            for k, v in flatten_clustered_metrics(
                results[i].get("metrics", {}),
            ).items():
                fv = float(v)
                if math.isfinite(fv):
                    m_acc[k].append(fv)
            for k, v in flatten_clustered_metrics(
                results[i].get("metrics_std", {}),
            ).items():
                fv = float(v)
                if math.isfinite(fv):
                    s_acc[k].append(fv)
        metrics_by[pk] = {k: float(np.nanmean(vs)) for k, vs in m_acc.items()}
        stds_by[pk] = {k: float(np.nanmean(vs)) for k, vs in s_acc.items()}
    return metrics_by, stds_by


def bundle_plot_drop_variables(
    bundle: "SummaryBundle",
    *,
    drop_variables: Optional[Sequence[str]] = (),
) -> tuple[str, ...]:
    """Resolve plot aggregation axes from arg or summary JSON header.

    Default ``drop_variables=()`` → one plot point per scored row (per-plate
    when plate is in ``condition_variables``), ignoring any
    ``plot_aggregate_over: plate`` in the JSON header.

    Pass ``drop_variables=None`` to read ``plot_aggregate_over`` from the bundle
    (per-file JSON / eval config).
    """
    if drop_variables is not None:
        return tuple(drop_variables)
    return resolve_plot_aggregate_drop(
        bundle.condition_variables,
        bundle.plot_aggregate_over,
    )


def bundle_metrics_for_plot(
    bundle: "SummaryBundle",
    *,
    drop_variables: Optional[Sequence[str]] = (),
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Plot-level metrics: aggregate across ``drop_variables`` when set."""
    drop = bundle_plot_drop_variables(bundle, drop_variables=drop_variables)
    if not drop:
        return bundle.metrics_by_condition, bundle.metrics_std_by_condition
    return mean_metrics_across_groups(
        bundle.results,
        bundle.condition_variables,
        drop_variables=drop,
    )


def _allow_plot_condition(
    condition_keys: Mapping[str, Any],
    *,
    allowed_condition_keys: Optional[set[tuple[str, float]]],
    allowed_cell_names: Optional[set[str]],
) -> bool:
    """Holdout filters using ``condition_keys``, not string key parsing."""
    if allowed_cell_names is not None:
        if str(condition_keys.get("cell_name", "")) not in allowed_cell_names:
            return False
    if allowed_condition_keys is not None:
        raw_dose = condition_keys.get("dose", 0.0)
        dose = round(float(raw_dose), 4) if raw_dose != "" else 0.0
        drug = str(condition_keys.get("drug", ""))
        if (drug, dose) not in allowed_condition_keys:
            return False
    return True


# Filter is dominated by iterating all rows + nanmean over metric groups.
# `_validate_compare_json_summaries` and `plot_model_vs_ceiling_across_conditions`
# call this on the same bundle with the same args, so we memoize by
# (bundle source, args) and return the same result tuple both times.
_FilterCacheKey = tuple[
    str,                                 # bundle.source (str)
    tuple[str, ...] | None,              # drop_variables (or ("__default__",))
    tuple[tuple[str, float], ...] | None,  # sorted allowed_condition_keys
    tuple[str, ...] | None,              # sorted allowed_cell_names
]
_FILTER_CACHE: dict[
    _FilterCacheKey,
    tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]],
] = {}


def clear_filter_cache() -> None:
    """Drop the module-level filter memo (tests / long-running procs)."""
    _FILTER_CACHE.clear()


def _filter_cache_key(
    bundle: "SummaryBundle",
    drop_variables: Optional[Sequence[str]],
    allowed_condition_keys: Optional[set[tuple[str, float]]],
    allowed_cell_names: Optional[set[str]],
) -> _FilterCacheKey:
    """Hashable key derived from bundle source + filter args."""
    drop_tuple = (
        None if drop_variables is None else tuple(drop_variables)
    )
    keys_tuple = (
        None
        if allowed_condition_keys is None
        else tuple(sorted(allowed_condition_keys))
    )
    cells_tuple = (
        None if allowed_cell_names is None else tuple(sorted(allowed_cell_names))
    )
    return (str(bundle.source), drop_tuple, keys_tuple, cells_tuple)


def _filter_metrics_for_plot(
    bundle: "SummaryBundle",
    *,
    drop_variables: Optional[Sequence[str]] = (),
    allowed_condition_keys: Optional[set[tuple[str, float]]] = None,
    allowed_cell_names: Optional[set[str]] = None,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Aggregate (if needed), then apply holdout / cell-line filters."""
    cache_key = _filter_cache_key(
        bundle, drop_variables, allowed_condition_keys, allowed_cell_names,
    )
    cached = _FILTER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    drop = bundle_plot_drop_variables(bundle, drop_variables=drop_variables)
    plot_vars = plot_variables(bundle.condition_variables, drop_variables=drop)
    filtered_results: list[dict[str, Any]] = []
    for entry in bundle.results:
        ck = entry.get("condition_keys")
        if not isinstance(ck, dict):
            continue
        plot_ck = {v: ck[v] for v in plot_vars if v in ck}
        if not _allow_plot_condition(
            plot_ck,
            allowed_condition_keys=allowed_condition_keys,
            allowed_cell_names=allowed_cell_names,
        ):
            continue
        filtered_results.append(entry)
    result = mean_metrics_across_groups(
        filtered_results,
        bundle.condition_variables,
        drop_variables=drop,
    )
    _FILTER_CACHE[cache_key] = result
    return result


def aggregated_scalar_field(
    results: Sequence[Mapping[str, Any]],
    variables: Sequence[str],
    field: str,
    *,
    drop_variables: Sequence[str] = (),
) -> dict[str, float]:
    """Nanmean of a scalar ``results[]`` field grouped for plotting."""
    plot_vars = plot_variables(variables, drop_variables=drop_variables)
    acc: dict[str, list[float]] = defaultdict(list)
    for entry in results:
        ck = entry.get("condition_keys")
        if not isinstance(ck, dict):
            continue
        if field not in entry:
            continue
        pk = condition_key_from_mapping(ck, plot_vars)
        fv = float(entry[field])
        if math.isfinite(fv):
            acc[pk].append(fv)
    return {pk: float(np.nanmean(vs)) for pk, vs in acc.items()}


def condition_short_label(key: str) -> str:
    parts = key.split("__")
    if len(parts) == 3:
        cell, drug, dose = parts
        dose_short = dose.rstrip("0").rstrip(".")
        return f"{cell} | {drug} | {dose_short}"
    return key


def safe_filename_part(s: str) -> str:
    return str(s).replace("/", "-").replace(" ", "_")


def canonical_condition_variables(variables: Sequence[str]) -> tuple[str, ...]:
    """Stable cross-file condition-variable order (alphabetical)."""
    return tuple(sorted(variables))


def normalize_bundle_condition_keys(bundle: "SummaryBundle") -> "SummaryBundle":
    """Re-key a summary bundle so condition ids match across JSON producers.

    Some eval exports list the same ``condition_variables`` in different
    orders (e.g. ``donor,cytokine`` vs ``cytokine,donor``).  Plotting joins
    series by the string condition key, so we rebuild keys from ``results[]``
    using alphabetically sorted variable names.
    """
    from dataclasses import replace

    canon = canonical_condition_variables(bundle.condition_variables)
    if canon == bundle.condition_variables:
        return bundle
    drop = bundle_plot_drop_variables(bundle)
    metrics_by, stds_by = mean_metrics_across_groups(
        bundle.results,
        canon,
        drop_variables=drop,
    )
    return replace(
        bundle,
        condition_variables=canon,
        metrics_by_condition=metrics_by,
        metrics_std_by_condition=stds_by,
    )
