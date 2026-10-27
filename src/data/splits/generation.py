"""Split generation core: the policy, the condition summary, train blocks, writers.

A split YAML is plain data that ``src/data/splits/predicates.py`` reads at train time. The
schemes that *draw* splits live in ``src/data/splits/schemes.py``; this module
holds what they share. Nothing here names a dataset: columns come from the
dataset registry, constants from ``dataset_split_policies.yaml``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from omegaconf import OmegaConf

from src.config.dataset_registry import DatasetRegistry
from src.data.vocab import obs_values_to_categorical, to_categorical_str
from src.data.splits.data_class import SplitConfig
from src.data.data_utils import enrich_obs_with_metadata
from src.data.splits.predicates import split_membership_masks

REPO_ROOT = Path(__file__).resolve().parents[3]  # src/data/splits/ -> PertGen
POLICIES_PATH = REPO_ROOT / "core_util_data" / "splits" / "dataset_split_policies.yaml"


# The policy: one dataset's split inputs, read from YAML

@dataclass(frozen=True)
class SplitPolicy:
    """One dataset entry from ``dataset_split_policies.yaml``."""

    dataset_alias: str
    split_dataset: str
    condition_summary: Path
    doses: Tuple[float, ...]
    hbss: Mapping[str, Any]
    families: Mapping[str, Mapping[str, Any]]


def load_split_policy(dataset_alias: str, *, policies_path: Optional[Path] = None) -> SplitPolicy:
    path = Path(policies_path or POLICIES_PATH)
    if not path.is_file():
        raise FileNotFoundError(f"split policy YAML not found: {path}")
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True) or {}
    if dataset_alias not in raw:
        raise KeyError(f"dataset {dataset_alias!r} missing from {path}; available: {sorted(raw)}")
    block = dict(raw[dataset_alias])
    hbss = dict(block.get("hbss") or {})
    if not hbss.get("balance_axes"):
        raise ValueError(f"policy {dataset_alias!r}: hbss.balance_axes is required")
    if not block.get("condition_summary"):
        raise ValueError(f"policy {dataset_alias!r}: condition_summary is required")
    return SplitPolicy(
        dataset_alias=dataset_alias,
        split_dataset=str(block.get("split_dataset") or dataset_alias),
        condition_summary=REPO_ROOT / str(block["condition_summary"]),
        doses=tuple(float(d) for d in block.get("doses") or ()),
        hbss=hbss,
        families={str(k): dict(v) for k, v in dict(block.get("families") or {}).items()},
    )


# Deterministic randomness. hash() is salted per process, so a
# split drawn with it cannot be redrawn; sha256 of the labels can.

def subseed(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


# The condition summary: one row per observed condition

def _is_control_row(row: pd.Series, control_pert: Mapping[str, Any]) -> bool:
    for col, expected in control_pert.items():
        actual = row[col]
        try:
            if float(actual) == float(expected):
                continue
        except (TypeError, ValueError):
            pass
        if str(actual) != str(expected):
            return False
    return True


def filter_summary(summary: pd.DataFrame, registry: DatasetRegistry, *, doses: Sequence[float]) -> pd.DataFrame:
    """Keep the canonical doses (controls always survive). Missing columns raise."""
    needed = set(registry.biological_context_columns) | set(registry.perturbation_columns) | set(registry.control_pert)
    missing = needed - set(summary.columns)
    if missing:
        raise ValueError(f"condition summary missing columns: {sorted(missing)}")
    out = summary.copy()
    if doses and "dose" in out.columns:
        dose_keys = {to_categorical_str(d) for d in doses}
        is_ctrl = out.apply(lambda row: _is_control_row(row, registry.control_pert), axis=1)
        keep = is_ctrl | pd.Series(obs_values_to_categorical(out["dose"]), index=out.index).isin(dose_keys)
        out = out.loc[keep]
    return out.reset_index(drop=True)


def load_condition_summary(policy: SplitPolicy, registry: DatasetRegistry) -> pd.DataFrame:
    """The policy summary + the registry's metadata joins (e.g. Organ), dose-filtered."""
    path = policy.condition_summary
    if not path.is_file():
        raise FileNotFoundError(f"condition summary not found: {path}")
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"condition summary empty: {path}")
    return filter_summary(enrich_obs_with_metadata(df, registry), registry, doses=policy.doses)


# The train block: all conditions, an HBSS fraction, or a cell cap

def train_block(*, hbss_pct: int = 100, hbss: Optional[Mapping[str, Any]] = None,
                seed: int = 0, cells_target: Optional[int] = None,
                cells_keep_controls: bool = False) -> Dict[str, Any]:
    if cells_target is not None:
        cells = {"target_n": int(cells_target), "seed": int(seed)}
        if cells_keep_controls:              # absent otherwise, so shipped splits stay byte-identical
            cells["keep_controls"] = True
        return {"conditions": {"sample": "all"}, "cells": cells}
    if not 0 < hbss_pct <= 100:
        raise ValueError(f"hbss_pct must be in (0, 100]; got {hbss_pct}")
    if hbss_pct == 100:
        return {"conditions": {"sample": "all"}}
    if not hbss:
        raise ValueError("an HBSS train block needs the policy's hbss spec")
    spec = {k: v for k, v in dict(hbss).items() if k != "seed"}
    return {"conditions": {"sample": "hbss", "fraction": round(hbss_pct / 100.0, 4),
                           "hbss": {**spec, "seed": int(seed)}}}


def estimate_train_cells(
    summary: pd.DataFrame,
    holdout_payload: Mapping[str, Any],
    *,
    registry: DatasetRegistry,
    hbss: Mapping[str, Any],
    hbss_pct: int,
    seed: int,
    method: str,
) -> int:
    """Train cells left after the holdout and a ``hbss_pct``% condition draw.

    ``method="hbss"`` runs the HBSS draw itself; ``"random_conditions"`` draws the
    same fraction of conditions uniformly. Control cells always count in full.
    """
    if "num_cells" not in summary.columns:
        raise ValueError("condition summary missing num_cells column")
    if method not in ("hbss", "random_conditions"):
        raise ValueError(f"cells_estimate must be 'hbss' or 'random_conditions'; got {method!r}")
    cfg = SplitConfig.from_dict({**holdout_payload, "train": {"sample": "all"}})
    is_train, _, _, is_ctrl = split_membership_masks(summary, cfg, dict(registry.control_pert))
    ctrl_cells = int(summary.loc[is_train & is_ctrl, "num_cells"].sum())
    pert = summary.loc[is_train & ~is_ctrl]
    if hbss_pct >= 100:
        return ctrl_cells + int(pert["num_cells"].sum())
    fraction = hbss_pct / 100.0
    if method == "random_conditions":
        n_keep = min(len(pert), max(1, int(round(fraction * len(pert)))))
        pick = np.random.default_rng(int(seed)).choice(len(pert), size=n_keep, replace=False)
        return ctrl_cells + int(pert.iloc[pick]["num_cells"].sum())
    from src.data.splits.data_class import HBSSSpec
    from src.data.sampling.hbss_sampling import run_hbss

    spec = HBSSSpec(**{**{k: v for k, v in dict(hbss).items() if k != "seed"}, "seed": int(seed)})
    axes = list(spec.balance_axes)
    chosen = {tuple(str(v) for v in row) for row in run_hbss(pert, spec, fraction=fraction).selected[axes].to_numpy()}
    keys = pd.MultiIndex.from_arrays([obs_values_to_categorical(pert[c]) for c in axes], names=axes)
    return ctrl_cells + int(pert.loc[keys.isin(chosen), "num_cells"].sum())


# The checks: a split that empties train or test is a bug

def validate_split_on_summary(
    summary: pd.DataFrame,
    payload: Mapping[str, Any],
    control_pert: Mapping[str, Any],
) -> Dict[str, int]:
    cfg = SplitConfig.from_dict(dict(payload))
    is_train, is_val, is_test, _ = split_membership_masks(summary, cfg, control_pert)
    counts = {"n_train": int(is_train.sum()), "n_val": int(is_val.sum()), "n_test": int(is_test.sum())}
    if counts["n_train"] == 0 or counts["n_test"] == 0:
        raise ValueError(f"split empties train or test on the condition summary: {counts}")
    return counts


# Writing the files

def write_split_yaml(path: Path, payload: Mapping[str, Any], *, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(header)
        yaml.safe_dump(dict(payload), handle, default_flow_style=False, sort_keys=False,
                       allow_unicode=True, width=120)


def write_manifest(path: Path, entries: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(entries), indent=2) + "\n", encoding="utf-8")
