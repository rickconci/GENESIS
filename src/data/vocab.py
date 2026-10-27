"""Condition vocabulary: doses parsed from Tahoe's drug strings, and the one
canonical string every condition value is keyed by (encoders, split predicates,
HBSS, eval caches), so strict string equality is always safe.

A column has two views, per-row keys and integer codes into a label table. The
first is derived from the second, so the two cannot drift apart.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray


# Tahoe hides the dose inside a string, ``[('Drug', 0.05, 'uM')]``. 

_CONC_PATTERN = re.compile(
    r",\s*([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)\s*,\s*['\"]?(?:uM|um)['\"]?",
    re.IGNORECASE,
)


def parse_doses(strings: Any) -> NDArray[np.float64]:
    """The dose of each ``[('name', conc, 'uM')]`` string; raises if any has none."""
    strings = np.asarray(strings, dtype=object)
    extracted = pd.Series(strings).str.extract(_CONC_PATTERN, expand=False)
    doses = extracted.astype(np.float64).to_numpy()
    bad = strings[np.isnan(doses)]
    if bad.size:
        raise ValueError(f"{bad.size} drug strings have no parseable dose, e.g. {list(bad[:5])}")
    return doses


def concentrations_from_obs(obs: pd.DataFrame, key: str = "drugname_drugconc") -> NDArray[np.float64]:
    """Per-row dose parsed from ``obs[key]``; raises on missing or unparseable rows.

    The regex runs once per distinct string, not per row: Tahoe has ~1,138
    ``drugname_drugconc`` categories over ~100M cells.
    """
    ser = obs[key]
    if not isinstance(ser.dtype, pd.CategoricalDtype):
        ser = ser.astype("category")
    codes = ser.cat.codes.to_numpy()
    if (codes < 0).any():
        raise ValueError(f"{key}: {int((codes < 0).sum())} rows have no value, so no dose")
    return parse_doses(ser.cat.categories.to_numpy())[codes]


# Define canonical keys.

def to_categorical_str(val: Any) -> str:
    """Floats print at float32 precision (``"5.0"``, ``"0.05"``); anything else is
    ``str(val)``, so drug names stay byte-identical."""
    if isinstance(val, (float, np.floating)):
        return str(np.float32(val))
    return str(val)


_PLATE_NUM_RE = re.compile(r"(?i)^plate(\d+)$")


def normalize_plate(value: object) -> str:
    """Canonical plate identifier: strip ``plate`` prefix, keep the numeric tail."""
    s = str(value).strip()
    m = _PLATE_NUM_RE.fullmatch(s)
    return m.group(1) if m else s


def condition_key(var: str, value: Any) -> str:
    """The key a condition value is looked up by in a run's encoder.

    Plates match by number: the encoder stores ``plate3``, eval queues carry ``3``.
    """
    return normalize_plate(value) if var == "plate" else to_categorical_str(value)


# Define columns of keys.



def obs_column_to_canonical_codes(series: pd.Series) -> tuple[NDArray[np.int64], NDArray[np.object_]]:
    """Per-row codes and their labels: ``labels[codes]`` is the column's keys.

    A missing float is dose 0 (``"0.0"``); any other missing value is ``"nan"``.
    """
    if pd.api.types.is_float_dtype(series):
        # Unique at float32 precision, so 0.05 and its float32 twin share a code.
        uniq, codes = np.unique(series.fillna(0.0).to_numpy(dtype=np.float32), return_inverse=True)
        return codes.astype(np.int64), np.array([to_categorical_str(u) for u in uniq], dtype=object)
    cat = series if isinstance(series.dtype, pd.CategoricalDtype) else series.astype("category")
    # Missing rows have code -1, which indexes the trailing "nan". factorize merges
    # categories that share a key (0.05 and its float32 twin).
    keys = np.array([to_categorical_str(c) for c in cat.cat.categories] + ["nan"], dtype=object)
    key_codes, labels = pd.factorize(keys)
    return key_codes[cat.cat.codes.to_numpy()].astype(np.int64), np.asarray(labels, dtype=object)


def obs_values_to_categorical(series: pd.Series) -> NDArray[np.object_]:
    """Per-row canonical keys of an obs column."""
    codes, labels = obs_column_to_canonical_codes(series)
    return labels[codes]


def aggregate_condition_summary(obs: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    """Cells per conditioning tuple, as ``[*group_cols, "num_cells"]`` with canonical keys.

    Each row's per-column codes are raveled into one int64, so counting is a 1-D
    ``np.unique`` (~2 s on 100M rows). ``np.unique`` on the 4-field tuples
    themselves falls back to a Python-level sort and hung ``dm.setup()`` for minutes.
    """
    cols = list(group_cols)
    if not cols:
        raise ValueError("group_cols must be non-empty")
    missing = [c for c in cols if c not in obs.columns]
    if missing:
        raise KeyError(f"Condition columns {missing} not in obs")
    if len(obs) == 0:
        return pd.DataFrame(columns=[*cols, "num_cells"])
    codes, labels = zip(*(obs_column_to_canonical_codes(obs[c]) for c in cols))
    shape = tuple(len(lab) for lab in labels)
    keys, counts = np.unique(np.ravel_multi_index(codes, shape), return_counts=True)
    per_col = np.unravel_index(keys, shape)
    out = {c: lab[i] for c, lab, i in zip(cols, labels, per_col)}
    return pd.DataFrame({**out, "num_cells": counts.astype(np.int64)})
