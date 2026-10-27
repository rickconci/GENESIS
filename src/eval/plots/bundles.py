"""Summary JSON loaders and typed bundle views."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.eval.metrics.reporting import flatten_clustered_metrics
from src.eval.plots.gene_arrays import (
    DEG_ARRAY_KEYS,
    DEG_SPACES,
    condition_key_from_mapping,
    deg_column,
    deg_scalar_column,
    dropout_column,
    gs_column,
    load_gene_arrays_dataframe,
    parquet_row_condition_keys,
    resolve_sidecar_path,
)

from .conditions import entry_condition_key
from .constants import (
    GENE_SUMMARY_SPACES,
    _MODEL_LEGEND_LABEL_BY_EVAL_DIR_PREFIX,
    _MODEL_LEGEND_LABEL_BY_RUN_DIR_BASENAME,
    _PERTGEN_EVAL_DIR_PREFIXES,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeneLevelArrays:
    """Per-condition arrays extracted from the gene-arrays parquet or JSON.

    The arm naming (``pred`` vs ``bootstrap``) follows the producer's
    ``gene_summary_pred_label`` field: model-eval runs use ``pred_mean`` /
    ``pred_var``; bootstrap ceiling runs use ``bootstrap_mean`` /
    ``bootstrap_var``.  The consumer-facing fields are normalised to
    ``pred_*`` regardless of origin to keep plotting code uniform.

    ``ctrl_mean`` / ``ctrl_var`` are populated when the producer embedded
    the control arm in ``gene_summary`` (every producer in src/eval does);
    a JSON without it leaves them ``None``.
    """

    condition_keys: dict[str, str]
    condition_variables: tuple[str, ...]
    space: str
    true_mean: np.ndarray
    true_var: np.ndarray
    pred_mean: np.ndarray
    pred_var: np.ndarray
    pred_label: str
    ctrl_mean: Optional[np.ndarray] = None
    ctrl_var: Optional[np.ndarray] = None

    @property
    def cell_name(self) -> str:
        return str(self.condition_keys.get("cell_name", ""))

    @property
    def drug(self) -> str:
        return str(self.condition_keys.get("drug", ""))

    @property
    def dose(self) -> float:
        raw = self.condition_keys.get("dose", 0.0)
        return float(raw) if raw != "" else 0.0

    @property
    def condition_label(self) -> str:
        parts = [
            f"{k}={self.condition_keys[k]}"
            for k in self.condition_variables
            if k in self.condition_keys
        ]
        return " | ".join(parts) if parts else condition_key_from_mapping(
            self.condition_keys, self.condition_variables,
        )

    @property
    def condition_key(self) -> str:
        return condition_key_from_mapping(
            self.condition_keys, self.condition_variables,
        )

    @property
    def has_control(self) -> bool:
        return self.ctrl_mean is not None and self.ctrl_var is not None


@dataclass(frozen=True)
class DropoutArrays:
    """Per-condition per-gene exact-zero fractions for one DEG target space.

    Emitted by :func:`src.eval.metrics.gene_summary.compute_dropout_summary` and
    stored in the sidecar parquet under ``dropout__<space>__*`` columns.

    Each array has length ``n_genes`` with values in ``[0, 1]``; entry
    ``g`` is the fraction of cells in that arm with *exactly* zero
    expression for gene ``g``.  Truth and control zero-fractions are
    identical across spaces by construction (raw integer counts whose
    zero set is preserved by monotonic element-wise and per-cell CPM
    transforms); ``pred`` differs across spaces only between ``raw_int``
    and ``raw`` / ``log1p_cpm`` (the gap is the fraction of cells where
    ``round(clip(expm1(pred_log1p)))`` collapsed a sub-1 positive float
    to 0).
    """

    condition_keys: dict[str, str]
    condition_variables: tuple[str, ...]
    space: str
    ctrl_zero_frac: Optional[np.ndarray]
    true_zero_frac: np.ndarray
    pred_zero_frac: np.ndarray
    pred_label: str

    @property
    def cell_name(self) -> str:
        return str(self.condition_keys.get("cell_name", ""))

    @property
    def drug(self) -> str:
        return str(self.condition_keys.get("drug", ""))

    @property
    def dose(self) -> float:
        raw = self.condition_keys.get("dose", 0.0)
        return float(raw) if raw != "" else 0.0

    @property
    def condition_label(self) -> str:
        parts = [
            f"{k}={self.condition_keys[k]}"
            for k in self.condition_variables
            if k in self.condition_keys
        ]
        return " | ".join(parts) if parts else condition_key_from_mapping(
            self.condition_keys, self.condition_variables,
        )

    @property
    def condition_key(self) -> str:
        return condition_key_from_mapping(
            self.condition_keys, self.condition_variables,
        )


@dataclass
class SummaryBundle:
    """In-memory view of a ``metrics_summary.json`` / bootstrap ceiling JSON.

    ``metrics_by_condition`` maps a canonical condition-key string (see
    :func:`entry_condition_key`) to a flat metric dict — one entry per scored
    row (full ``condition_variables``).  Cross-condition plots may nanmean
    over ``plot_aggregate_over`` (see :func:`_filter_metrics_for_plot`).  For bootstrap
    ceiling files we additionally expose ``metrics_std_by_condition`` (std
    across repeats) so plots can draw error bars.

    Per-gene ``gene_summary`` / ``deg_detail`` arrays are read from the
    sidecar parquet (``metrics_gene_arrays.parquet`` /
    ``bootstrap_gene_arrays.parquet``).  The parquet is loaded lazily on first
    access and cached on the bundle.
    """

    source: Path
    kind: str  # "model" | "ceiling"
    gene_set_id: Optional[str]
    gene_names: list[str]
    gene_summary_pred_label: str
    results: list[dict[str, Any]]
    metrics_by_condition: dict[str, dict[str, float]]
    metrics_std_by_condition: dict[str, dict[str, float]]
    condition_variables: tuple[str, ...] = ()
    plot_aggregate_over: Optional[str] = None
    strategy: Optional[str] = None
    gene_arrays_parquet: Optional[Path] = None
    # Lazily populated on first call to gene_arrays / deg_detail_by_condition.
    _arrays_df: Optional[pd.DataFrame] = field(default=None, repr=False, compare=False)

    # -- lazy parquet access -------------------------------------------------

    def _load_arrays_df(self) -> Optional[pd.DataFrame]:
        """Return (and cache) the gene-arrays sidecar DataFrame, if any."""
        if self.gene_arrays_parquet is None:
            return None
        if self._arrays_df is None:
            self._arrays_df = load_gene_arrays_dataframe(self.gene_arrays_parquet)
        return self._arrays_df

    def gene_arrays(self, space: str) -> list[GeneLevelArrays]:
        """Materialise per-condition gene-level arrays for ``space``.

        Reads the gene-array sidecar parquet written alongside the summary
        JSON.

        Args:
            space: One of :data:`GENE_SUMMARY_SPACES`.

        Returns:
            List of :class:`GeneLevelArrays`, one per parquet row.
        """
        if space not in GENE_SUMMARY_SPACES:
            raise ValueError(
                f"Unknown gene_summary space: {space!r} "
                f"(valid: {GENE_SUMMARY_SPACES})"
            )
        df = self._load_arrays_df()
        if df is None:
            raise FileNotFoundError(
                f"Gene-array sidecar missing for {self.source} "
                f"(expected metrics_gene_arrays.parquet next to the summary JSON).",
            )
        return self._gene_arrays_from_parquet(df, space)

    def deg_detail_by_condition(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Canonical condition key -> ``deg_space`` -> ``deg_detail`` dict.

        The outer key is :func:`entry_condition_key`.  The inner key is the
        DEG target space (:data:`DEG_SPACES`).  Values mirror the on-disk
        ``deg_detail`` shape: ``fdr_threshold`` / ``log2_fc_space`` scalars
        plus per-gene ``z_*`` / ``q_*`` / ``lfc_*`` arrays.
        """
        df = self._load_arrays_df()
        if df is None:
            raise FileNotFoundError(
                f"Gene-array sidecar missing for {self.source} "
                f"(expected metrics_gene_arrays.parquet next to the summary JSON).",
            )
        return self._deg_detail_from_parquet(df)

    def deg_detail_by_condition_for_space(
        self, deg_space: str,
    ) -> dict[str, dict[str, Any]]:
        """Flat condition key -> ``deg_detail`` dict for a single space.

        Convenience view for plot scripts that iterate one DEG space at a
        time; silently drops conditions for which the space is missing.

        Args:
            deg_space: Target space (:data:`DEG_SPACES`).

        Returns:
            ``{condition_key: block}`` dict; empty if no row has the
            requested space.
        """
        nested = self.deg_detail_by_condition()
        out: dict[str, dict[str, Any]] = {}
        for key, by_space in nested.items():
            block = by_space.get(deg_space)
            if block is not None:
                out[key] = block
        return out

    def dropout_summary(self, deg_space: str) -> list[DropoutArrays]:
        """Per-condition drop-out (exact-zero fraction) arrays for ``deg_space``.

        Reads ``dropout__<space>__*`` columns from the gene-array sidecar.
        Rows with ``None`` dropout arrays for the requested space are skipped.

        Args:
            deg_space: One of :data:`DEG_SPACES`.

        Returns:
            List of :class:`DropoutArrays`, one per condition with a
            dropout block for ``deg_space``.
        """
        if deg_space not in DEG_SPACES:
            raise ValueError(
                f"Unknown DEG space: {deg_space!r} (valid: {DEG_SPACES})"
            )
        df = self._load_arrays_df()
        if df is None:
            raise FileNotFoundError(
                f"Gene-array sidecar missing for {self.source} "
                f"(expected metrics_gene_arrays.parquet next to the summary JSON).",
            )
        return self._dropout_from_parquet(df, deg_space)

    # -- parquet path --------------------------------------------------------

    def _gene_arrays_from_parquet(
        self, df: pd.DataFrame, space: str,
    ) -> list[GeneLevelArrays]:
        pred_label = self.gene_summary_pred_label  # display-only
        # Canonical parquet columns (produced by GeneArraysParquetWriter)
        # always use the ``pred_*`` arm, regardless of the producer's
        # ``gene_summary_pred_label``; the display label is attached at
        # the GeneLevelArrays boundary.
        col_tm = gs_column(space, "true_mean")
        col_tv = gs_column(space, "true_var")
        col_pm = gs_column(space, "pred_mean")
        col_pv = gs_column(space, "pred_var")
        col_cm = gs_column(space, "ctrl_mean")
        col_cv = gs_column(space, "ctrl_var")
        array_cols = [col_tm, col_tv, col_pm, col_pv]
        if col_cm in df.columns:
            array_cols.append(col_cm)
        if col_cv in df.columns:
            array_cols.append(col_cv)
        out: list[GeneLevelArrays] = []
        variables = self.condition_variables
        for rd in df.to_dict(orient="records"):
            keys = parquet_row_condition_keys(rd)
            cm = rd.get(col_cm)
            cv = rd.get(col_cv)
            out.append(
                GeneLevelArrays(
                    condition_keys=keys,
                    condition_variables=variables,
                    space=space,
                    true_mean=np.asarray(rd[col_tm], dtype=np.float64),
                    true_var=np.asarray(rd[col_tv], dtype=np.float64),
                    pred_mean=np.asarray(rd[col_pm], dtype=np.float64),
                    pred_var=np.asarray(rd[col_pv], dtype=np.float64),
                    pred_label=pred_label,
                    ctrl_mean=np.asarray(cm, dtype=np.float64) if cm is not None else None,
                    ctrl_var=np.asarray(cv, dtype=np.float64) if cv is not None else None,
                )
            )
        return out

    def _deg_detail_from_parquet(
        self, df: pd.DataFrame,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Read the per-space ``deg__<space>__*`` columns."""
        out: dict[str, dict[str, dict[str, Any]]] = {}
        variables = self.condition_variables
        needed: list[str] = []
        per_space_cols: dict[str, dict[str, str]] = {}
        for space in DEG_SPACES:
            fdr_col = deg_scalar_column(space, "fdr_threshold")
            lfs_col = deg_scalar_column(space, "log2_fc_space")
            arm_cols = {arm: deg_column(space, arm) for arm in DEG_ARRAY_KEYS}
            per_space_cols[space] = {"fdr": fdr_col, "lfs": lfs_col, **arm_cols}
            for c in (fdr_col, lfs_col, *arm_cols.values()):
                if c in df.columns:
                    needed.append(c)
        present = [c for c in needed if c in df.columns]
        for rd in df[present].to_dict(orient="records"):
            keys = parquet_row_condition_keys(rd)
            key = condition_key_from_mapping(keys, variables)
            by_space: dict[str, dict[str, Any]] = {}
            for space in DEG_SPACES:
                cols = per_space_cols[space]
                if not any(cols[arm] in df.columns for arm in DEG_ARRAY_KEYS):
                    continue
                block: dict[str, Any] = {
                    "fdr_threshold": float(rd.get(cols["fdr"], float("nan"))),
                    "log2_fc_space": str(rd.get(cols["lfs"], "")),
                }
                for arm in DEG_ARRAY_KEYS:
                    val = rd.get(cols[arm])
                    block[arm] = np.asarray(val) if val is not None else None
                by_space[space] = block
            out[key] = by_space
        return out

    def _dropout_from_parquet(
        self, df: pd.DataFrame, deg_space: str,
    ) -> list[DropoutArrays]:
        pred_label = self.gene_summary_pred_label
        col_c = dropout_column(deg_space, "ctrl_zero_frac")
        col_t = dropout_column(deg_space, "true_zero_frac")
        col_p = dropout_column(deg_space, "pred_zero_frac")
        if col_t not in df.columns or col_p not in df.columns:
            return []
        out: list[DropoutArrays] = []
        variables = self.condition_variables
        for rd in df.to_dict(orient="records"):
            t = rd.get(col_t)
            p = rd.get(col_p)
            if t is None or p is None:
                continue
            c = rd.get(col_c)
            keys = parquet_row_condition_keys(rd)
            out.append(
                DropoutArrays(
                    condition_keys=keys,
                    condition_variables=variables,
                    space=deg_space,
                    ctrl_zero_frac=(
                        np.asarray(c, dtype=np.float64) if c is not None else None
                    ),
                    true_zero_frac=np.asarray(t, dtype=np.float64),
                    pred_zero_frac=np.asarray(p, dtype=np.float64),
                    pred_label=pred_label,
                )
            )
        return out

def _load_result_metrics(
    entry: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    """Read ``(metrics, metrics_std)`` from one ``results[]`` row (flat)."""
    metrics = flatten_clustered_metrics(entry.get("metrics", {}))
    stds = flatten_clustered_metrics(entry.get("metrics_std", {}))
    return metrics, stds


# Path-keyed load memo. Same (abs_path, mtime, size) → same parsed bundle. Cheap
# to store (bundles are already alive elsewhere) and catches
# `_validate_compare_json_summaries` re-loading before
# `plot_model_vs_ceiling_across_conditions` re-loads the identical files.
_LOAD_CACHE: dict[tuple[str, str, int, int], "SummaryBundle"] = {}


def clear_load_cache() -> None:
    """Drop the module-level bundle load cache (tests / long-running procs)."""
    _LOAD_CACHE.clear()


def _load_cache_key(path: Path, kind: str) -> tuple[str, str, int, int]:
    """Cache key from absolute path + mtime_ns + size — invalidates on write."""
    st = path.stat()
    return (kind, str(path.resolve()), int(st.st_mtime_ns), int(st.st_size))


def load_metrics_summary(path: Path) -> SummaryBundle:
    """Load a GENESIS ``metrics_summary.json`` (``scripts/eval/score.py genesis``).

    The ``results[i]`` block carries ``k``, ``metrics`` (means), and
    ``metrics_std`` (population std across K-rep draws).  Metric keys use
    ``<space>/<metric>`` (e.g. ``"log1p_cpm/pearson_r"``,
    ``"deg_raw_int/jaccard_sig_fdr0p001_lfc0p1_detfloor10"``).

    Per-gene ``gene_summary`` / ``deg_detail`` arrays are read from the
    parquet sidecar pointed to by ``data["gene_arrays_parquet"]`` (or the
    canonical filename next to the JSON).

    Repeated loads of the same file (as long as mtime+size are unchanged)
    return the cached bundle — this dominates cross-plot warm-up when the
    same JSON is opened by multiple plotting helpers per run.
    """
    path = Path(path)
    cache_key = _load_cache_key(path, kind="model")
    cached = _LOAD_CACHE.get(cache_key)
    if cached is not None:
        return cached
    with path.open("r") as f:
        data = json.load(f)
    results = data.get("results", [])
    variables = tuple(data.get("condition_variables") or ())
    plot_agg = data.get("plot_aggregate_over")
    if plot_agg is not None:
        plot_agg = str(plot_agg).strip() or None
    metrics_by_cond: dict[str, dict[str, float]] = {}
    stds_by_cond: dict[str, dict[str, float]] = {}
    for entry in results:
        cond_key = entry_condition_key(entry, variables)
        metrics, stds = _load_result_metrics(entry)
        metrics_by_cond[cond_key] = metrics
        stds_by_cond[cond_key] = stds
    sidecar = resolve_sidecar_path(path, data)
    bundle = SummaryBundle(
        source=path,
        kind="model",
        gene_set_id=data.get("gene_set_id"),
        gene_names=list(data.get("gene_names", [])),
        gene_summary_pred_label=str(data.get("gene_summary_pred_label", "pred")),
        condition_variables=variables,
        plot_aggregate_over=plot_agg,
        strategy=(str(data["strategy"]) if data.get("strategy") else None),
        results=list(results),
        metrics_by_condition=metrics_by_cond,
        metrics_std_by_condition=stds_by_cond,
        gene_arrays_parquet=sidecar,
    )
    _LOAD_CACHE[cache_key] = bundle
    return bundle


def load_bootstrap_ceiling(path: Path) -> SummaryBundle:
    """Load a noise-ceiling ``noise_ceiling.json`` (unified schema).

    The ceiling JSON shares the model-eval schema: each ``results[]``
    row carries ``k``, ``metrics`` (means, clustered into the four metric
    groups), and ``metrics_std`` (population std across K-rep draws).
    This loader is the strict twin of :func:`load_metrics_summary` with
    ``kind="ceiling"``.
    """
    path = Path(path)
    cache_key = _load_cache_key(path, kind="ceiling")
    cached = _LOAD_CACHE.get(cache_key)
    if cached is not None:
        return cached
    with path.open("r") as f:
        data = json.load(f)
    results = data.get("results", [])
    variables = tuple(data.get("condition_variables") or ())
    plot_agg = data.get("plot_aggregate_over")
    if plot_agg is not None:
        plot_agg = str(plot_agg).strip() or None
    metrics_by_cond: dict[str, dict[str, float]] = {}
    stds_by_cond: dict[str, dict[str, float]] = {}
    for entry in results:
        cond_key = entry_condition_key(entry, variables)
        metrics, stds = _load_result_metrics(entry)
        metrics_by_cond[cond_key] = metrics
        stds_by_cond[cond_key] = stds
    sidecar = resolve_sidecar_path(path, data)
    bundle = SummaryBundle(
        source=path,
        kind="ceiling",
        gene_set_id=data.get("gene_set_id"),
        gene_names=list(data.get("gene_names", [])),
        gene_summary_pred_label=str(data.get("gene_summary_pred_label", "bootstrap")),
        condition_variables=variables,
        plot_aggregate_over=plot_agg,
        strategy=(str(data["strategy"]) if data.get("strategy") else None),
        results=list(results),
        metrics_by_condition=metrics_by_cond,
        metrics_std_by_condition=stds_by_cond,
        gene_arrays_parquet=sidecar,
    )
    _LOAD_CACHE[cache_key] = bundle
    return bundle


# ---------------------------------------------------------------------------
# Model legend label heuristic (for cross-condition plots)
# ---------------------------------------------------------------------------


def _model_legend_label(summary_path: Path, bundle_source_name: str) -> str:
    """Short name for cross-condition legend entries.

    A ``metrics_summary.json`` lives inside a per-model output dir, so
    we look at the *parent* directory name (and a couple of ancestors) to
    pick up paper-facing labels.  Fallback is the parent directory name.
    """
    eval_dir = summary_path.parent
    dir_name = eval_dir.name
    if dir_name in _MODEL_LEGEND_LABEL_BY_EVAL_DIR_PREFIX:
        return _MODEL_LEGEND_LABEL_BY_EVAL_DIR_PREFIX[dir_name]
    for prefix in _PERTGEN_EVAL_DIR_PREFIXES:
        if dir_name.startswith(prefix):
            return "PertGen"
    if bundle_source_name:
        bn = Path(str(bundle_source_name)).name
        if bn in _MODEL_LEGEND_LABEL_BY_RUN_DIR_BASENAME:
            return _MODEL_LEGEND_LABEL_BY_RUN_DIR_BASENAME[bn]
    return dir_name
