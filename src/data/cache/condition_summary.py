"""Condition summaries: one row per condition with its cell count.

``condition_summary_from_h5ads`` is the source of truth (every cell, plus the
registry's metadata joins); ``build_condition_summary_from_eval_cache`` only
sees conditions that have an eval NPZ. Filename parsing lives here because
nothing else reads those names.
"""
from __future__ import annotations

import os
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.config.dataset_registry import DatasetConditionSpec
from src.data.cache.eval_cache_build import PERTURBED_ARRAY_KEY

_KNOWN_CONDITION_KEYS = frozenset({
    "cell_name", "cell_line", "cell_type", "cytokine", "donor", "drug", "dose",
    "plate", "gene_target", "sample",
})
_KEY_ALIASES = {"cell": "cell_name"}
_LEGACY_PERT_RES = [
    re.compile(
        r"^perturbed_cache__cell_name_(?P<cell_name>.+?)"
        r"__dose_(?P<dose>.+?)__drug_(?P<drug>.+?)"
        r"__plate_(?P<plate>.+?)\.npz$",
    ),
    re.compile(
        r"^perturbed_cache__cell_(?P<cell_name>.+?)"
        r"__drug_(?P<drug>.+?)__dose_(?P<dose>.+?)"
        r"__plate_(?P<plate>.+?)\.npz$",
    ),
]
NPZ_READ_ERRORS = (EOFError, OSError, ValueError, KeyError, zipfile.BadZipFile)


def _split_segment(segment: str, known_keys: frozenset[str]) -> tuple[str, str]:
    matches = [k for k in known_keys if segment.startswith(f"{k}_")]
    if not matches:
        raise ValueError(f"cannot infer key from NPZ segment {segment!r}")
    key = max(matches, key=len)
    return key, segment[len(key) + 1:]


def parse_cache_npz_keys(
    name: str, prefix: str, *, known_keys: Optional[Iterable[str]] = None,
) -> dict[str, str] | None:
    """Parse ``{prefix}__k1_v1__k2_v2....npz`` into a key dict."""
    keys = _KNOWN_CONDITION_KEYS | frozenset(known_keys or ())
    stem = f"{prefix}__"
    if not name.startswith(stem) or not name.endswith(".npz"):
        return None
    body = name[len(stem):-4]
    if not body:
        return None
    out: dict[str, str] = {}
    try:
        for segment in body.split("__"):
            k, v = _split_segment(segment, keys)
            out[k] = v
    except ValueError:
        return None
    return out or None


def normalize_condition_keys(keys: Mapping[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in keys.items():
        nk = _KEY_ALIASES.get(k, k)
        if nk == "dose":
            try:
                v = f"{float(v):.6g}"
            except ValueError:
                pass
        elif nk == "plate" and not v.startswith("plate"):
            v = f"plate{v}"
        out[nk] = v
    return out


def parse_perturbed_npz_keys(name: str, *, known_keys: Optional[Iterable[str]] = None) -> dict[str, str] | None:
    raw = parse_cache_npz_keys(name, "perturbed_cache", known_keys=known_keys)
    if raw is None:
        for r in _LEGACY_PERT_RES:
            if (m := r.match(name)) is not None:
                raw = dict(m.groupdict())
                break
    return normalize_condition_keys(raw) if raw else None


def npz_n_rows(path: Path, array_key: str = PERTURBED_ARRAY_KEY) -> int:
    """CSR row count from NPZ header only (no data-array load)."""
    with np.load(path) as z:
        shape_key = f"{array_key}_shape"
        if shape_key in z.files:
            return int(z[shape_key][0])
        if array_key in z.files:
            return int(z[array_key].shape[0])
        raise KeyError(f"{path} missing {shape_key!r} and {array_key!r}")


def _probe_one(path: Path) -> tuple[dict[str, str] | None, int | None, str | None]:
    keys = parse_perturbed_npz_keys(path.name)
    if keys is None:
        return None, None, f"unparsed filename: {path.name}"
    try:
        return keys, npz_n_rows(path, PERTURBED_ARRAY_KEY), None
    except NPZ_READ_ERRORS as exc:
        return keys, None, f"{type(exc).__name__}: {exc}"


def build_condition_summary_from_eval_cache(
    namespace: Path | str, columns: Sequence[str], *, n_workers: int = 32, min_cells: int = 0,
) -> pd.DataFrame:
    """One row per perturbed NPZ: conditioning columns + ``num_cells``."""
    ns = Path(namespace)
    if not ns.is_dir():
        raise FileNotFoundError(f"eval cache namespace not found: {ns}")
    paths = [Path(e.path) for e in os.scandir(ns)
             if e.is_file() and e.name.startswith("perturbed_cache__") and e.name.endswith(".npz")]
    if not paths:
        raise ValueError(f"no perturbed_cache NPZs under {ns}")
    rows, corrupt = [], 0
    with ThreadPoolExecutor(max_workers=max(1, int(n_workers))) as pool:
        for fut in tqdm([pool.submit(_probe_one, p) for p in paths],
                        total=len(paths), desc="probe perturbed NPZs"):
            keys, n_cells, err = fut.result()
            if keys is None or n_cells is None:
                corrupt += 1
                continue
            if n_cells < int(min_cells):
                continue
            row = {c: keys[c] for c in columns if c in keys}
            if missing := [c for c in columns if c not in row]:
                raise KeyError(f"{paths[0].parent}: NPZ keys missing columns {missing} (parsed={sorted(keys)})")
            row["num_cells"] = int(n_cells)
            rows.append(row)
    if not rows:
        raise ValueError(f"no rows after probe (namespace={ns}, min_cells={min_cells}, corrupt={corrupt})")
    if corrupt:
        print(f"skipped unreadable/unparsed perturbed NPZs: {corrupt}")
    df = pd.DataFrame(rows)
    return df[list(columns) + ["num_cells"]].sort_values(list(columns)).reset_index(drop=True)


def build_condition_summary_for_spec(
    spec: DatasetConditionSpec, *, n_workers: int = 32, min_cells: int = 0,
) -> pd.DataFrame:
    if spec.eval_cache_namespace is None:
        raise ValueError("DatasetConditionSpec.eval_cache_namespace is required")
    return build_condition_summary_from_eval_cache(
        spec.eval_cache_namespace, spec.conditioning_vars, n_workers=n_workers, min_cells=min_cells,
    )


def condition_summary_from_h5ads(
    h5ad_paths: Sequence[Path], columns: Sequence[str], *, metadata_config: object,
) -> pd.DataFrame:
    """Count cells per ``columns`` tuple over every h5ad, then apply metadata joins.

    Reads obs only (no X). ``dose`` is derived from the concentration key when
    it is not an obs column, exactly as the training datamodule does.
    """
    import anndata as ad

    from src.data.vocab import aggregate_condition_summary
    from src.data.data_utils import enrich_obs_with_metadata, prepare_cache_obs_for_indexing

    if not h5ad_paths:
        raise ValueError("no h5ad files to summarise")
    frames = []
    for path in h5ad_paths:
        adata = ad.read_h5ad(path, backed="r")
        try:
            obs = prepare_cache_obs_for_indexing(adata.obs, condition_variables=list(columns))
            frames.append(obs[list(columns)].copy())
        finally:
            if getattr(adata, "file", None) is not None:
                adata.file.close()
    return enrich_obs_with_metadata(
        aggregate_condition_summary(pd.concat(frames, ignore_index=True), list(columns)),
        metadata_config,
    )
