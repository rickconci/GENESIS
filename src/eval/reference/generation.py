"""Generate a model's cells for each replicate of a condition, at that replicate's real size.

A model scored against an experimental replicate must get the same number of cells as
the real replicate it stands in for: more cells would give its DEG calls more power than
the experiment had.  The counts come from a scored pair table (``n_truth_i`` /
``n_truth_j``), so every downsampled or re-seeded arm lands on exactly the counts of the
arm it is compared with.  Each (condition, replicate) draws from its own seed, so a file
is reproducible on its own and independent of generation order.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from src.eval.reference.replicate import ReplicateDesign, _replicate_label


def replicate_cell_counts(d: ReplicateDesign, pairs: pd.DataFrame) -> dict[tuple, int]:
    """(condition values..., replicate) -> n_truth, asserted consistent across pairs."""
    rep = d.replicate_col
    need = [*d.condition_cols, f"{rep}_i", f"{rep}_j", "n_truth_i", "n_truth_j"]
    if missing := [c for c in need if c not in pairs.columns]:
        raise KeyError(f"pair table is missing {missing}")
    out: dict[tuple, int] = {}
    for r in pairs[need].to_dict("records"):
        cond = tuple(r[c] for c in d.condition_cols)
        for side in ("i", "j"):
            key, n = (*cond, str(r[f"{rep}_{side}"])), int(r[f"n_truth_{side}"])
            if out.setdefault(key, n) != n:
                raise ValueError(f"n_truth mismatch for {key}: {out[key]} vs {n}")
    return out


def replicate_seed(d: ReplicateDesign, cond: Mapping[str, Any], rep: str) -> int:
    """A stable 32-bit seed per (condition, replicate); for Tahoe ``cell|drug|dose|plateN``."""
    parts = [f"{float(cond[c]):.8g}" if isinstance(cond[c], (float, np.floating)) else str(cond[c])
             for c in d.condition_cols]
    raw = "|".join([*parts, _replicate_label(d, rep)]).encode()
    return int.from_bytes(hashlib.blake2s(raw, digest_size=4).digest(), "little")


def replicate_k_seed(d: ReplicateDesign, cond: Mapping[str, Any], rep: str, draw: int) -> int:
    """Seed of draw ``draw`` of one replicate. Draw 0 is ``replicate_seed``, the seed of every
    arm generate.py wrote (the main GENESIS pair table scores the model eval's own dump instead).
    K draws are K separate plate-size clouds, scored one by one and averaged, as the split-half
    K-rep loop does; concatenating them would hand DEG K times the experiment's cells."""
    base = replicate_seed(d, cond, rep)
    if int(draw) == 0:
        return base
    raw = f"{base}|draw{int(draw)}".encode()
    return int.from_bytes(hashlib.blake2s(raw, digest_size=4).digest(), "little")


def generate_condition(bundle, values: Mapping[str, Any], *, n_cells: int, batch_size: int,
                       generator: torch.Generator) -> np.ndarray:
    """(n_cells, n_genes) for one condition (column -> value), generated in chunks."""
    idx = {c: bundle.encode_condition(c, v) for c, v in values.items()}
    chunks, done = [], 0
    while done < n_cells:
        b = int(min(batch_size, n_cells - done))
        enc = {f"{c}_idx": torch.full((b,), int(v), dtype=torch.long, device=bundle.device)
               for c, v in idx.items()}
        chunks.append(bundle.generate(generator=generator, **enc))
        done += b
    out = torch.cat(chunks, dim=0).cpu().numpy().astype(np.float32, copy=False)
    if out.shape != (n_cells, len(bundle.gene_names)):
        raise ValueError(f"got {out.shape}, expected ({n_cells}, {len(bundle.gene_names)})")
    return out


def atomic_save(path: Path, arr: np.ndarray) -> None:
    """Write then rename, so a killed run never leaves a truncated file that looks done."""
    tmp = path.with_name(path.name + ".tmp.npy")
    if tmp.exists():
        tmp.unlink()
    np.save(tmp, arr)
    tmp.replace(path)
