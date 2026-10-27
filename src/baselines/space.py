"""The space a baseline is fit in, and the way back to counts.

  raw               the arm's mean raw counts, emitted through emit_per_cell_delta (CP10k of the HVGs)
  log1p_cp10k_full  the mean over cells of log1p(1e4 x / L_full), L_full = the cell's full-transcriptome
                    library: the space Ahlmann-Eltze et al. (Nat Methods 2025) and State fit their
                    baselines in. Emitted as the STATE arm is (emit_in_fit_space): matched control cells
                    moved by the fitted delta in this space, denormalised with each cell's own L_full.

The HVG cache matrices do not carry L_full; a sidecar namespace ``<namespace>__with_libfull`` does,
and its manifest points at the same matrices.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import numpy as np

RAW, LOG1P_CP10K_FULL = "raw", "log1p_cp10k_full"
FIT_SPACES = (RAW, LOG1P_CP10K_FULL)
CP10K = 1e4


class MissingLibrary(RuntimeError):
    """No usable L_full. A RuntimeError on purpose: fit.read_arms drops an arm on ValueError, and a
    missing sidecar must stop the fit, not quietly thin it."""


def library_namespace(namespace: Path) -> Path:
    side = Path(f"{namespace}__with_libfull")
    if not side.is_dir():
        raise MissingLibrary(f"{side} missing: a log1p_cp10k_full fit needs each cell's full-transcriptome library")
    return side


def checked_library(lib: Optional[np.ndarray], n_cells: int, where: str) -> np.ndarray:
    if lib is None or lib.shape != (n_cells,):
        raise MissingLibrary(f"{where}: L_full {None if lib is None else lib.shape} for {n_cells} cells")
    if not (lib > 0).all():
        raise MissingLibrary(f"{where}: {int((lib <= 0).sum())} cells have L_full <= 0")
    return lib


def to_fit_space(x: np.ndarray, lib: np.ndarray) -> np.ndarray:
    return np.log1p(x * (CP10K / lib)[:, None]).astype(np.float32)


def from_fit_space(y: np.ndarray, lib: np.ndarray) -> np.ndarray:
    """Counts, each cell at its own L_full (the STATE arm's denorm, clip included)."""
    return np.expm1(np.clip(y, 0.0, None)) * (lib / CP10K)[:, None]


class FitSpaceReader:
    """An eval reader (on a ``__with_libfull`` namespace) whose cells come back in log1p(1e4 x / L_full),
    so averaging them averages logs. Whole matrices only: L_full lines up row for row with the full
    matrix, not with a subsample."""

    def __init__(self, reader):
        self.reader = reader

    def perturbed(self, keys: Mapping, *, max_cells: Optional[int] = None, seed=None) -> np.ndarray:
        _whole(max_cells)
        x = self.reader.perturbed(keys)
        return to_fit_space(x, checked_library(self.reader.perturbed_library_full(keys), len(x), f"perturbed {dict(keys)}"))

    def control(self, keys: Mapping, *, max_cells: Optional[int] = None, seed=None) -> np.ndarray:
        return self.control_with_library(keys, max_cells=max_cells)[0]

    def control_with_library(self, keys: Mapping, *, max_cells: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        """(cells in the fit space, their L_full): what the emission moves and denormalises."""
        _whole(max_cells)
        x = self.reader.control(keys)
        lib = checked_library(self.reader.control_library_full(keys), len(x), f"control {dict(keys)}")
        return to_fit_space(x, lib), lib


def _whole(max_cells: Optional[int]) -> None:
    if max_cells is not None:
        raise NotImplementedError("a subsample would desync rows from L_full: fit log1p_cp10k_full on all cells")
