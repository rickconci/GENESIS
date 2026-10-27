"""Assign a covariate a model never emitted (PBMC cell type) to its generated cells.

A model conditioned on donor x cytokine emits one mixed cloud per condition. To score it
per cell type, each cell is labelled by a linear classifier trained on real cells, and the
cloud is split by label. Only labels that are one released name ("mapped") are scored;
merged labels (e.g. CD4_T_cell, which spans several released T types) are kept aside.

The classifier is plain arrays (StandardScaler(with_mean=False) -> one-vs-rest linear):

    label = classes[argmax((x / scale) @ coef.T + intercept)],  x = log1p(CP10K) on its genes

so it runs in any numpy, with no pickled sklearn objects tied to one version. The CP10K
library is the panel's own (only the genes the model emits): a full-transcriptome
library would leak genes the model never generated.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

# expm1 of log1p(counts) on a 2k panel is O(100) per cell; CP10K would be ~1e4.
_CP10K_LIBRARY = 5000.0


@dataclass(frozen=True)
class LinearCellClassifier:
    genes: list[str]
    scale: np.ndarray       # (G,)
    coef: np.ndarray        # (K, G)
    intercept: np.ndarray   # (K,)
    classes: list[str]
    mapped: list[str]       # per class: the one released name it stands for, or ""

    @classmethod
    def load(cls, path: Path) -> "LinearCellClassifier":
        with np.load(path, allow_pickle=False) as z:
            out = cls([str(g) for g in z["genes"]], z["scale"], z["coef"], z["intercept"],
                      [str(c) for c in z["classes"]], [str(m) for m in z["mapped"]])
        k, g = out.coef.shape
        if out.scale.shape != (g,) or out.intercept.shape != (k,) or len(out.classes) != k \
                or len(out.mapped) != k or len(out.genes) != g:
            raise ValueError(f"{path}: inconsistent classifier shapes")
        return out

    def save(self, path: Path) -> None:
        np.savez(path, genes=np.array(self.genes), scale=self.scale, coef=self.coef,
                 intercept=self.intercept, classes=np.array(self.classes), mapped=np.array(self.mapped))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Labels for log1p(CP10K) rows. Mirrors sklearn: float32 in-place scaling, then a
        float64 decision function, so labels match ``Pipeline.predict`` exactly."""
        xs = np.asarray(x, dtype=np.float32).copy()
        xs /= self.scale
        scores = xs @ self.coef.T + self.intercept
        return np.asarray(self.classes, dtype=object)[np.argmax(scores, axis=1)]


def from_sklearn(pipeline, genes: Sequence[str], mapping: Mapping[str, str]) -> LinearCellClassifier:
    """One-off conversion of a fitted StandardScaler + SGDClassifier pipeline."""
    scaler, linear = pipeline.steps[0][1], pipeline.steps[-1][1]
    if getattr(scaler, "with_mean", True) or len(pipeline.steps) != 2:
        raise ValueError("expected StandardScaler(with_mean=False) -> linear classifier")
    classes = [str(c) for c in linear.classes_]
    if unknown := sorted(set(mapping) - set(classes)):
        raise ValueError(f"mapping names labels the classifier lacks: {unknown}")
    return LinearCellClassifier([str(g) for g in genes], np.asarray(scaler.scale_), np.asarray(linear.coef_),
                                np.asarray(linear.intercept_), classes, [mapping.get(c, "") for c in classes])


def to_counts(x: np.ndarray) -> np.ndarray:
    """log1p(counts) -> non-negative integer counts."""
    return np.rint(np.expm1(np.clip(x, 0, None))).astype(np.int32)


def log1p_cp10k(counts: np.ndarray) -> np.ndarray:
    """Integer counts -> log1p(CP10K) over these genes. An empty row stays all-zero."""
    tot = counts.sum(axis=1, keepdims=True).astype(np.float32)
    scale = np.divide(1e4, tot, out=np.zeros_like(tot), where=tot > 0)
    return np.log1p(counts.astype(np.float32) * scale)


def assert_log1p_counts(pred: np.ndarray, what: str) -> None:
    """Refuse a CP10K-normalised or non-finite cloud: it would be classified as noise."""
    lib = float(np.median(np.expm1(np.clip(pred[:256], 0, None)).sum(axis=1)))
    if not np.isfinite(lib) or lib <= 0 or lib > _CP10K_LIBRARY:
        raise SystemExit(f"{what}: median expm1 library is {lib}, not log1p(counts) on this panel")
