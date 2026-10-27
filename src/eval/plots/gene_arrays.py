"""Sidecar parquet I/O for per-condition gene-level arrays.

The ``gene_summary`` (per-gene mean / variance in raw / log1p / log1p_cpm
space) and ``deg_detail`` (per-gene Wilcoxon z / q / log2fc arrays) blocks
live in a compact columnar parquet sidecar written alongside the JSON, not
inside ``metrics_summary.json``: for a test-set run with ~2300 conditions ×
2000 genes, each Python ``float`` serialises to ~16 chars of JSON, so those
blocks would inflate a single summary to ~3 GB and load slowly.

The JSON keeps scalar metrics + a
``gene_arrays_parquet`` pointer; plot readers go through the helpers
defined here.

Schema (one row per condition):

* Key columns: ``ck__<var>`` (str) for each entry in the writer's
  ``condition_variables`` (dataset-agnostic).
* Per DEG-target-space scalar metadata:
  ``deg__<deg_space>__fdr_threshold`` (float64),
  ``deg__<deg_space>__log2_fc_space`` (str).
* Gene-summary arrays (``list<float32>``, length == ``n_genes``):
  ``gs__<space>__<arm>`` for ``space`` ∈ {raw, log1p, log1p_cpm} and
  ``arm`` ∈ {true_mean, true_var, pred_mean, pred_var, ctrl_mean,
  ctrl_var}.
* DEG-detail arrays (``list<float32>``, length == ``n_genes``):
  ``deg__<deg_space>__{z|q|lfc}_{truth|pred}_vs_ctrl`` for each
  ``deg_space`` ∈ :data:`DEG_SPACES`.
* Drop-out (per-gene exact-zero fraction) arrays (``list<float32>``,
  length == ``n_genes``):
  ``dropout__<deg_space>__{ctrl|true|pred}_zero_frac`` for each
  ``deg_space`` ∈ :data:`DEG_SPACES`.  Computed by
  :func:`src.eval.metrics.gene_summary.compute_dropout_summary`; columns are
  written as ``None`` by producers that don't compute this summary.

The predicted-arm column name is normalised to ``pred_*`` regardless of
which producer wrote the row (the JSON header's
``gene_summary_pred_label`` is honoured by the reader for display
labelling).

Writer:
    The writer is streaming: it buffers rows in Python and flushes a parquet
    row group every ``batch_size`` conditions, so peak RAM stays
    ``O(batch_size * n_genes)`` rather than ``O(n_conditions * n_genes)``.

Reader:
    :func:`load_gene_arrays_dataframe` returns a pandas ``DataFrame`` (one
    row per condition; each array column is ``object`` dtype with ``np.ndarray``
    cells) and caches the result process-wide by absolute path so repeated
    calls from the plot helpers don't re-parse the file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..metrics.spaces import DEG_SPACES

logger = logging.getLogger(__name__)

# Canonical sidecar filenames so producers and readers agree without
# having to negotiate through the JSON header.
METRICS_GENE_ARRAYS_FILENAME: str = "metrics_gene_arrays.parquet"
BOOTSTRAP_GENE_ARRAYS_FILENAME: str = "bootstrap_gene_arrays.parquet"

CK_PREFIX: str = "ck__"

GENE_SUMMARY_SPACES: tuple[str, ...] = (
    "raw_int_rgs", "log1p_rgs", "log1p_slice_med_rgs",
)

# Arm names as they appear inside the producer's ``gene_summary`` dict once
# the producer-specific predicted label ("pred" / "bootstrap") has been
# normalised to the canonical "pred_*" form used in the parquet columns.
GENE_SUMMARY_ARMS: tuple[str, ...] = (
    "true_mean", "true_var",
    "pred_mean", "pred_var",
    "ctrl_mean", "ctrl_var",
)

DEG_ARRAY_KEYS: tuple[str, ...] = (
    "z_truth_vs_ctrl", "q_truth_vs_ctrl", "lfc_truth_vs_ctrl",
    "z_pred_vs_ctrl", "q_pred_vs_ctrl", "lfc_pred_vs_ctrl",
)

# Per-gene drop-out (exact-zero fraction) arms, one set per entry in
# :data:`DEG_SPACES`.  Written by :func:`src.eval.metrics.gene_summary.compute_dropout_summary`.
DROPOUT_ARRAY_KEYS: tuple[str, ...] = (
    "ctrl_zero_frac", "true_zero_frac", "pred_zero_frac",
)


def ck_column(var: str) -> str:
    """Parquet column for one conditioning variable: ``ck__<var>``."""
    return f"{CK_PREFIX}{var}"


def condition_key_from_mapping(
    keys: Mapping[str, Any],
    variables: Sequence[str],
) -> str:
    """Canonical condition id shared by JSON metrics and parquet rows.

    Joins present ``variables`` values with ``__`` in config order
    (skipping keys absent from ``keys``, e.g. grouped eval).
    """
    parts: list[str] = []
    for var in variables:
        if var not in keys:
            continue
        val = keys[var]
        if var == "dose":
            parts.append(f"{float(val):.6f}")
        else:
            parts.append(str(val))
    if not parts:
        raise ValueError(
            f"Cannot build condition key from variables={list(variables)!r} "
            f"and keys={dict(keys)!r}.",
        )
    return "__".join(parts)


def parquet_row_condition_keys(row: Mapping[str, Any]) -> dict[str, str]:
    """Read conditioning keys from one parquet row (``ck__*`` columns)."""
    ck_cols = [c for c in row if isinstance(c, str) and c.startswith(CK_PREFIX)]
    if not ck_cols:
        raise KeyError("Parquet row has no ck__* condition-key columns.")
    return {c[len(CK_PREFIX):]: str(row[c]) for c in sorted(ck_cols)}


def gs_column(space: str, arm: str) -> str:
    """Canonical column name for a gene-summary arm: ``gs__<space>__<arm>``."""
    return f"gs__{space}__{arm}"


def deg_column(deg_space: str, arm: str) -> str:
    """Canonical column name for a deg-detail arm: ``deg__<deg_space>__<arm>``."""
    return f"deg__{deg_space}__{arm}"


def dropout_column(deg_space: str, arm: str) -> str:
    """Canonical column name for a dropout arm: ``dropout__<deg_space>__<arm>``."""
    return f"dropout__{deg_space}__{arm}"


def deg_scalar_column(deg_space: str, field: str) -> str:
    """Canonical column name for a per-space DEG scalar (``fdr_threshold``, ``log2_fc_space``).

    Example: ``deg__log1p_cpm__fdr_threshold``.
    """
    return f"deg__{deg_space}__{field}"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def _arrow_schema(
    condition_variables: Sequence[str],
    *,
    include_ctrl: bool = True,
) -> pa.Schema:
    """Build the pyarrow schema for the gene-arrays parquet.

    Args:
        condition_variables: ``model_conditioning_variables`` for the run.
        include_ctrl: If ``True`` (default), include the ``ctrl_mean`` /
            ``ctrl_var`` gene-summary columns.  Current producers always
            supply the control arm, so the default matches production.

    Returns:
        A pyarrow ``Schema`` with ``list<float32>`` array columns and
        string / float scalar key columns.
    """
    list_type = pa.list_(pa.float32())
    fields: list[pa.Field] = [
        pa.field(ck_column(var), pa.string(), nullable=True)
        for var in sorted(set(condition_variables))
    ]
    for deg_space in DEG_SPACES:
        fields.append(pa.field(deg_scalar_column(deg_space, "fdr_threshold"), pa.float64()))
        fields.append(pa.field(deg_scalar_column(deg_space, "log2_fc_space"), pa.string()))
    arms = GENE_SUMMARY_ARMS if include_ctrl else GENE_SUMMARY_ARMS[:4]
    for space in GENE_SUMMARY_SPACES:
        for arm in arms:
            fields.append(pa.field(gs_column(space, arm), list_type))
    for deg_space in DEG_SPACES:
        for arm in DEG_ARRAY_KEYS:
            fields.append(pa.field(deg_column(deg_space, arm), list_type))
    for deg_space in DEG_SPACES:
        for arm in DROPOUT_ARRAY_KEYS:
            fields.append(pa.field(dropout_column(deg_space, arm), list_type))
    return pa.schema(fields)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def _asarray_f32(x: Any) -> list[float]:
    """Coerce a list / ndarray / torch tensor to a plain fp32 python list.

    Returns a Python list rather than an ``np.ndarray`` because pyarrow
    builds list columns from Python iterables of iterables (sequences of
    per-row arrays), and for a single row the conversion to a list is
    effectively free since pyarrow copies the values into the internal
    buffer anyway.
    """
    arr = np.asarray(x, dtype=np.float32).ravel()
    return arr.tolist()


class GeneArraysParquetWriter:
    """Streaming writer for the per-condition gene-arrays sidecar parquet.

    The writer keeps an in-memory buffer of rows, then flushes a parquet
    row group every ``batch_size`` rows.  This bounds peak RAM at
    ``O(batch_size * n_genes)`` regardless of how many conditions are
    written.

    The writer normalises the incoming producer-specific predicted-arm
    label (``"pred"`` for the PertGen producer, ``"bootstrap"`` for the
    bootstrap ceiling producer) to a single canonical ``pred_mean`` /
    ``pred_var`` column so downstream readers never need to know which
    producer wrote the file.

    Example:
        >>> w = GeneArraysParquetWriter(path, condition_variables=["donor", "cytokine"])
        ... w.write_condition(
        ...     condition_keys={"donor": "D1", "cytokine": "G-CSF"},
        ...     gene_summary=gs, deg_detail=dd,
        ... )
    """

    def __init__(
        self,
        out_path: Path,
        condition_variables: Sequence[str],
        *,
        batch_size: int = 64,
        compression: str = "zstd",
        compression_level: int = 5,
    ) -> None:
        """Open a new parquet writer backed by a fixed schema.

        Args:
            out_path: Destination parquet file.  Parent directory must
                exist.
            condition_variables: Conditioning columns for ``ck__*`` keys.
            batch_size: Number of rows to buffer before writing a row
                group.  Larger batches compress slightly better but use
                more RAM.
            compression: Parquet compression codec (``"zstd"``,
                ``"snappy"``, ``"gzip"``, ``"none"``).
            compression_level: Codec-specific level (``zstd`` sweet spot
                for float data is 3-7).
        """
        self._out_path = Path(out_path)
        self._condition_variables = tuple(sorted(set(condition_variables)))
        if not self._condition_variables:
            raise ValueError("GeneArraysParquetWriter requires non-empty condition_variables.")
        self._batch_size = int(batch_size)
        self._schema = _arrow_schema(self._condition_variables, include_ctrl=True)
        self._column_order: list[str] = [f.name for f in self._schema]
        self._writer = pq.ParquetWriter(
            self._out_path,
            self._schema,
            compression=compression,
            compression_level=compression_level,
        )
        self._buf: list[dict[str, Any]] = []
        self._n_written: int = 0

    def __enter__(self) -> "GeneArraysParquetWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    # -- main API ----------------------------------------------------------

    def write_condition(
        self,
        *,
        condition_keys: Mapping[str, Any],
        gene_summary: Mapping[str, Mapping[str, Any]],
        deg_detail: Mapping[str, Mapping[str, Any]],
        dropout_summary: Optional[Mapping[str, Mapping[str, Any]]] = None,
        pred_label: str = "pred",
    ) -> None:
        """Queue one condition row, flushing if the batch is full.

        Args:
            condition_keys: ``{var: value}`` for this scored group (must
                use the writer's ``condition_variables`` naming).
            gene_summary: Nested dict as returned by
                :func:`src.eval.metrics.gene_summary.compute_gene_summary`.
                Keys are spaces (``"raw_int_rgs"`` / ``"log1p_rgs"`` /
                ``"log1p_slice_med_rgs"``),
                values are arm-to-array dicts using the producer's
                ``pred_label`` prefix (``"pred_mean"`` or
                ``"bootstrap_mean"``).
            deg_detail: Nested dict as returned by
                :func:`src.eval.metrics.deg.compute_deg_detail`: keys are
                DEG target spaces (``"raw_int_rgs"``), values are the per-space
                block with the
                z / q / lfc arrays plus ``fdr_threshold`` and
                ``log2_fc_space`` scalars.  Missing spaces are padded with
                ``NaN`` scalars and ``None`` arrays so the schema stays
                fixed.
            dropout_summary: Optional nested dict as returned by
                :func:`src.eval.metrics.gene_summary.compute_dropout_summary`:
                per-space per-gene exact-zero fractions for the control,
                truth and predicted arms.  When ``None``, the dropout
                columns are written as ``None`` so downstream readers can
                detect absence.
            pred_label: The producer's predicted-arm label.  Used to
                resolve ``{pred_label}_mean`` → canonical ``pred_mean``
                in the parquet columns.
        """
        keys = dict(condition_keys)
        row: dict[str, Any] = {}
        for var in self._condition_variables:
            if var in keys:
                row[ck_column(var)] = str(keys[var])
            else:
                row[ck_column(var)] = None
        for deg_space in DEG_SPACES:
            block = deg_detail.get(deg_space)
            if block is None:
                row[deg_scalar_column(deg_space, "fdr_threshold")] = float("nan")
                row[deg_scalar_column(deg_space, "log2_fc_space")] = ""
            else:
                row[deg_scalar_column(deg_space, "fdr_threshold")] = float(
                    block.get("fdr_threshold", float("nan"))
                )
                row[deg_scalar_column(deg_space, "log2_fc_space")] = str(
                    block.get("log2_fc_space", "")
                )

        # Gene-summary arrays: re-map producer-specific ``{pred_label}_*``
        # keys to the canonical ``pred_*`` column names.
        for space in GENE_SUMMARY_SPACES:
            block = gene_summary.get(space)
            if block is None:
                # Producer omitted this space entirely (shouldn't happen
                # for current producers, but keep the schema stable).
                for arm in GENE_SUMMARY_ARMS:
                    row[gs_column(space, arm)] = None
                continue
            row[gs_column(space, "true_mean")] = _asarray_f32(block["true_mean"])
            row[gs_column(space, "true_var")] = _asarray_f32(block["true_var"])
            row[gs_column(space, "pred_mean")] = _asarray_f32(block[f"{pred_label}_mean"])
            row[gs_column(space, "pred_var")] = _asarray_f32(block[f"{pred_label}_var"])
            ctrl_mean = block.get("ctrl_mean")
            ctrl_var = block.get("ctrl_var")
            row[gs_column(space, "ctrl_mean")] = (
                _asarray_f32(ctrl_mean) if ctrl_mean is not None else None
            )
            row[gs_column(space, "ctrl_var")] = (
                _asarray_f32(ctrl_var) if ctrl_var is not None else None
            )

        for deg_space in DEG_SPACES:
            block = deg_detail.get(deg_space)
            for arm in DEG_ARRAY_KEYS:
                if block is None or block.get(arm) is None:
                    row[deg_column(deg_space, arm)] = None
                else:
                    row[deg_column(deg_space, arm)] = _asarray_f32(block[arm])

        for deg_space in DEG_SPACES:
            block = None if dropout_summary is None else dropout_summary.get(deg_space)
            for arm in DROPOUT_ARRAY_KEYS:
                if block is None or block.get(arm) is None:
                    row[dropout_column(deg_space, arm)] = None
                else:
                    row[dropout_column(deg_space, arm)] = _asarray_f32(block[arm])

        self._buf.append(row)
        if len(self._buf) >= self._batch_size:
            self._flush()

    def close(self) -> None:
        """Flush any buffered rows and close the underlying parquet file."""
        if self._buf:
            self._flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None  # type: ignore[assignment]
            logger.info(
                "Wrote %d conditions to %s (%.1f MiB)",
                self._n_written, self._out_path,
                self._out_path.stat().st_size / (1024 * 1024),
            )

    # -- internal ----------------------------------------------------------

    def _flush(self) -> None:
        """Convert the buffer to a pyarrow Table and append as a row group."""
        # Build columns in schema order for a stable RecordBatch layout.
        columns: dict[str, list[Any]] = {name: [] for name in self._column_order}
        for row in self._buf:
            for name in self._column_order:
                columns[name].append(row.get(name))
        table = pa.table(columns, schema=self._schema)
        self._writer.write_table(table)
        self._n_written += len(self._buf)
        self._buf.clear()


# ---------------------------------------------------------------------------
# Reader (used by SummaryBundle)
# ---------------------------------------------------------------------------


# Process-wide cache keyed by absolute parquet path.  Plot runs typically
# read the same sidecar multiple times (once for scatters, once for DEG
# diagnostics, etc.); this avoids re-parsing the parquet each time.
_DATAFRAME_CACHE: dict[str, pd.DataFrame] = {}


def load_gene_arrays_dataframe(path: Path) -> pd.DataFrame:
    """Return the gene-arrays parquet as a pandas ``DataFrame`` (cached).

    Array columns come back as object dtype with each cell a ``np.ndarray``
    (pyarrow-engine default for ``list<float32>`` columns).  The pandas
    index is the integer row index; use :func:`parquet_row_condition_keys`
    on each row for conditioning keys.

    Args:
        path: Absolute or relative path to the parquet file.

    Returns:
        One-row-per-condition DataFrame; raises ``FileNotFoundError`` if
        the file is missing.
    """
    abs_path = str(Path(path).resolve())
    cached = _DATAFRAME_CACHE.get(abs_path)
    if cached is not None:
        return cached
    df = pd.read_parquet(path, engine="pyarrow")
    _DATAFRAME_CACHE[abs_path] = df
    return df


def resolve_sidecar_path(summary_json: Path, header: Mapping[str, Any]) -> Optional[Path]:
    """Find the gene-arrays parquet sidecar for a summary JSON.

    Resolution order:

    1. ``header["gene_arrays_parquet"]`` (relative to the JSON's parent
       directory).
    2. ``<summary_dir>/metrics_gene_arrays.parquet``.
    3. ``<summary_dir>/bootstrap_gene_arrays.parquet``.

    Returns ``None`` if no sidecar exists.
    """
    summary_dir = Path(summary_json).resolve().parent
    hinted = header.get("gene_arrays_parquet")
    if isinstance(hinted, str) and hinted:
        hinted_path = (summary_dir / hinted).resolve()
        if hinted_path.is_file():
            return hinted_path
        logger.warning(
            "metrics_summary header hints at gene_arrays_parquet=%s but the "
            "file is missing; falling back to conventional names.",
            hinted_path,
        )
    for name in (METRICS_GENE_ARRAYS_FILENAME, BOOTSTRAP_GENE_ARRAYS_FILENAME):
        candidate = summary_dir / name
        if candidate.is_file():
            return candidate.resolve()
    return None


__all__ = [
    "BOOTSTRAP_GENE_ARRAYS_FILENAME",
    "CK_PREFIX",
    "DEG_ARRAY_KEYS",
    "DEG_SPACES",
    "DROPOUT_ARRAY_KEYS",
    "GENE_SUMMARY_ARMS",
    "GENE_SUMMARY_SPACES",
    "GeneArraysParquetWriter",
    "METRICS_GENE_ARRAYS_FILENAME",
    "ck_column",
    "condition_key_from_mapping",
    "deg_column",
    "deg_scalar_column",
    "dropout_column",
    "gs_column",
    "load_gene_arrays_dataframe",
    "parquet_row_condition_keys",
    "resolve_sidecar_path",
]
