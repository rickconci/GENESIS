"""Pseudobulk response of one plate: per-gene log2FC against that plate's controls, gated and depth-adjusted.

Figure 4's arithmetic, one plate at a time:

    log2FC_g = log2((m_g + 1e-9) / (c_g + 1e-9))      m, c = mean raw count over perturbed, control cells
    gate_g   = mean over the condition's plates of g's control detection >= 0.10
    x_g      = log2FC_g - d                             d = log2(mean tscp per perturbed / per control cell)

d is the plate's measured library-size shift over the whole transcriptome (the h5ad's tscp_count), one
number per (line, drug, dose, plate); every arm is given truth's d, so truth - arm is unchanged by it.
A gene outside the gate has no measurable response (its log2FC is the pseudocount's, +-20 or more), so it
is dropped as NaN rather than kept at a value.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PSEUDOCOUNT = 1e-9
DETECTION_FLOOR = 0.10
DepthKey = tuple[str, str, float, str]   # (cell_name as in the dataset, drug, dose rounded to 5 d.p., "plateN")


def log2fc(pert_mean: np.ndarray, ctrl_mean: np.ndarray) -> np.ndarray:
    """Per-gene log2 fold change of two mean-count vectors."""
    m, c = np.asarray(pert_mean, np.float64), np.asarray(ctrl_mean, np.float64)
    if m.shape != c.shape:
        raise ValueError(f"perturbed mean {m.shape} and control mean {c.shape} differ")
    if (m < 0).any() or (c < 0).any() or not (np.isfinite(m).all() and np.isfinite(c).all()):
        raise ValueError("means must be finite counts >= 0")
    return np.log2((m + PSEUDOCOUNT) / (c + PSEUDOCOUNT))


def detection_gate(ctrl_detection: Sequence[np.ndarray], floor: float = DETECTION_FLOOR) -> np.ndarray:
    """Genes whose control detection, averaged with equal weight over the condition's plates, is >= floor."""
    if len(ctrl_detection) == 0:
        raise ValueError("no plates to gate on")
    d = np.stack([np.asarray(x, np.float64) for x in ctrl_detection])
    if not np.isfinite(d).all() or (d < 0).any() or (d > 1).any():
        raise ValueError("control detection must be fractions in [0, 1]")
    return d.mean(axis=0) >= floor


def response(pert_mean: np.ndarray, ctrl_mean: np.ndarray, gate: np.ndarray, depth: float) -> np.ndarray:
    """Depth-adjusted log2FC on the gated genes, NaN elsewhere."""
    if not np.isfinite(depth):
        raise ValueError(f"depth {depth} is not finite")
    x = log2fc(pert_mean, ctrl_mean)
    if gate.shape != x.shape or gate.dtype != bool:
        raise ValueError(f"gate must be a boolean mask of shape {x.shape}")
    return np.where(gate, x - depth, np.nan)


def load_depth(path: Path) -> dict[DepthKey, float]:
    """The library-size table (one row per condition and plate) as {key: depth_log2fc}.

    Refuses a table whose depth column is not log2(mean_tscp / ctrl_mean_tscp), or that lists a key twice.
    """
    t = pd.read_csv(path, sep="\t")
    expect = np.log2(t.mean_tscp / t.ctrl_mean_tscp)
    if not np.allclose(t.depth_log2fc, expect, atol=1e-6, equal_nan=False):
        raise ValueError(f"{path}: depth_log2fc is not log2(mean_tscp / ctrl_mean_tscp)")
    keys = list(zip(t.cell_name.astype(str), t.drug.astype(str), t.dose.astype(float).round(5), t.plate.astype(str)))
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path}: {len(keys) - len(set(keys))} duplicate (cell, drug, dose, plate) rows")
    return dict(zip(keys, t.depth_log2fc.astype(float)))


def depth_of(table: dict[DepthKey, float], cell_name: str, drug: str, dose: float, plate: str) -> float:
    key = (str(cell_name), str(drug), round(float(dose), 5), str(plate))
    if key not in table:
        raise KeyError(f"no library-size row for {key}")
    return table[key]
