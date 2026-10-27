"""Train conditions from pre-built eval_cache NPZ (one dataset, one namespace)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..cache.eval_cache_build import EVAL_NPZ_MODE
from ...eval.cache.eval_cache_io import _stringify_condition_value, resolve_eval_cache_mode
from ...eval.cache.eval_npz_reader import EvalNpzReader
from ..vocab import aggregate_condition_summary, to_categorical_str
from ..data_utils import filter_condition_summary_for_train

if TYPE_CHECKING:
    from ..datamodule import PertGenDataModule


def cell_set_log(enabled: bool, msg: str, *args: object) -> None:
    """Print cell-set debug line when ``cell_set_debug`` is enabled."""
    if not enabled:
        return
    text = msg % args if args else msg
    print(f"  [cell_set] {text}", flush=True)


@dataclass(frozen=True)
class CellSetCondition:
    """One train unit: perturbed NPZ or control pool (absolute expression match)."""

    pert_keys: Dict[str, str]       # full condition → model codes (+ pert NPZ if not control)
    ctrl_keys: Dict[str, str]       # match keys only → control NPZ
    is_control: bool = False
    budget: Optional[int] = None    # train-row count: draw only from that many fixed rows


def _ctrl_keys(
    pert_keys: Mapping[str, str],
    vars_: Sequence[str],
    control_condition: Mapping[str, Any],
) -> Dict[str, str]:
    """Control NPZ keys = shared bio/batch columns only (not the control pert value)."""
    ctrl_vars = set(control_condition.keys())
    return {k: pert_keys[k] for k in vars_ if k not in ctrl_vars}


def _is_control_keys(
    keys: Mapping[str, str],
    control_condition: Mapping[str, Any],
) -> bool:
    """True iff ``keys`` match every entry of ``control_condition`` (stringified)."""
    if not control_condition:
        return False
    for k, v in control_condition.items():
        if str(keys.get(k, "")) != to_categorical_str(v):
            return False
    return True


def load_cell_set_conditions(
    dm: "PertGenDataModule",
    *,
    debug: bool = False,
) -> Tuple[EvalNpzReader, Tuple[CellSetCondition, ...]]:
    """Train conditions (perturbed + control) from eval_cache + split-filtered summary."""
    if dm._condition_summary is None:
        raise RuntimeError("cell_set: call dm.setup() before building the train loader.")
    if not getattr(dm, "eval_manifest", None):
        raise RuntimeError("cell_set: eval_manifest missing; call dm.setup() first.")
    if not dm.config.split_config:
        raise ValueError("cell_set: split_config is required.")

    ns = Path(dm.eval_manifest.eval_cache_dir).resolve()
    if resolve_eval_cache_mode(ns) != EVAL_NPZ_MODE:
        raise ValueError(f"cell_set: NPZ eval_cache required at {ns}")

    vars_ = tuple(dm.config.model_conditioning_variables or ())
    if not vars_:
        raise ValueError("cell_set: model_conditioning_variables is empty.")
    ctrl = dict(dm.config.control_condition or {})
    min_cells = int(getattr(dm.config, "cell_set_min_cells", 1) or 1)
    split_path = Path(dm.config.split_config).expanduser()

    train_df = filter_condition_summary_for_train(
        dm._condition_summary,
        split_config=split_path,
        control_condition=ctrl or None,
        drop_control_rows=False,
        min_cells=min_cells,
    )
    missing = [c for c in vars_ if c not in train_df.columns]
    if missing:
        raise KeyError(f"condition_summary missing columns {missing!r}")

    # Control rows share one control_cache pool per match-key tuple — keep one unit each.
    seen_ctrl_pools: set[tuple[tuple[str, str], ...]] = set()
    conditions: List[CellSetCondition] = []
    n_ctrl_dropped = 0
    for row in train_df.itertuples(index=False):
        pert_keys = {v: _stringify_condition_value(getattr(row, v)) for v in vars_}
        ck = _ctrl_keys(pert_keys, vars_, ctrl)
        is_ctrl = _is_control_keys(pert_keys, ctrl)
        if is_ctrl:
            pool_id = tuple(sorted(ck.items()))
            if pool_id in seen_ctrl_pools:
                n_ctrl_dropped += 1
                continue
            seen_ctrl_pools.add(pool_id)
        conditions.append(CellSetCondition(pert_keys, ck, is_control=is_ctrl))

    if getattr(dm.config, "cell_set_budget_from_train_indices", False) is True:
        conditions = _apply_train_budget(dm, conditions, vars_, ctrl)
    if not conditions:
        raise ValueError(
            "cell_set: no train conditions (check split + control_condition + min_cells).",
        )

    n_ctrl = sum(1 for c in conditions if c.is_control)
    n_pert = len(conditions) - n_ctrl
    cell_set_log(debug, "eval_cache_dir=%s", ns)
    cell_set_log(
        debug,
        "train conditions=%d (pert=%d ctrl=%d dedup_dropped=%d) min_cells>=%d vars=%s",
        len(conditions), n_pert, n_ctrl, n_ctrl_dropped, min_cells, list(vars_),
    )
    cell_set_log(debug, "control_condition=%s", ctrl)
    for i, c in enumerate(conditions[:3]):
        cell_set_log(
            debug, "  cond[%d] is_control=%s keys=%s ctrl_pool=%s",
            i, c.is_control, c.pert_keys, c.ctrl_keys,
        )
    if len(conditions) > 3:
        cell_set_log(debug, "  ... (%d more conditions)", len(conditions) - 3)

    print(
        f"  CellSet: {len(conditions):,} train conditions "
        f"({n_pert:,} pert + {n_ctrl:,} ctrl) from {ns}",
        flush=True,
    )
    return EvalNpzReader(ns), tuple(conditions)


def _apply_train_budget(
    dm: "PertGenDataModule",
    conditions: List[CellSetCondition],
    vars_: Sequence[str],
    ctrl: Mapping[str, Any],
) -> List[CellSetCondition]:
    """Give each unit its count of the split's TRAIN rows; drop units with none.

    The eval cache holds every cell of a condition, but a cell-budgeted split trains on a
    subset. Counting ``dm.train_indices`` with the function that built the condition
    summary gives labels identical to the units' keys. Control pools sum their rows.
    """
    obs = dm._collection.obs
    counts = aggregate_condition_summary(obs.iloc[np.asarray(dm.train_indices)], list(vars_))
    pert: Dict[tuple, int] = {}
    pools: Dict[tuple, int] = {}
    for r in counts.itertuples(index=False):
        keys = {v: _stringify_condition_value(getattr(r, v)) for v in vars_}
        if _is_control_keys(keys, ctrl):
            pid = tuple(sorted(_ctrl_keys(keys, vars_, ctrl).items()))
            pools[pid] = pools.get(pid, 0) + int(r.num_cells)
        else:
            pert[tuple(keys[v] for v in vars_)] = int(r.num_cells)
    out, seen = [], set()
    for c in conditions:
        key = (("ctrl", tuple(sorted(c.ctrl_keys.items()))) if c.is_control
               else ("pert", tuple(c.pert_keys[v] for v in vars_)))
        seen.add(key)
        n = (pools if c.is_control else pert).get(key[1], 0)
        if n > 0:
            out.append(replace(c, budget=n))
    unknown = [("pert", k) for k, n in pert.items() if n and ("pert", k) not in seen]
    unknown += [("ctrl", k) for k, n in pools.items() if n and ("ctrl", k) not in seen]
    if unknown:
        raise ValueError(f"cell_set budget: {len(unknown)} train units have no cell_set unit, e.g. {unknown[:3]}")
    total, n_train = sum(c.budget for c in out), len(dm.train_indices)
    if total != n_train:
        raise ValueError(f"cell_set budget: units hold {total:,} train rows, the split has {n_train:,}")
    table = pd.DataFrame([{**c.pert_keys, "is_control": c.is_control, "budget": c.budget} for c in out])
    workdir = getattr(dm, "_workdir", None)
    if isinstance(workdir, Path):
        table.to_parquet(workdir / "cell_set_budget.parquet", index=False)
    print(f"  CellSet budget: {len(out):,} of {len(conditions):,} units keep train rows "
          f"({total:,} rows; median {int(table['budget'].median())}, "
          f"{(table['budget'] >= 256).mean():.0%} >= 256)", flush=True)
    return out


def _fixed_subset(keys: Mapping[str, str], n_file: int, budget: int, seed: int) -> np.ndarray:
    """The same ``budget`` rows of a unit's file every epoch: seeded by the unit's keys."""
    raw = "|".join(f"{k}={keys[k]}" for k in sorted(keys)) + f"|{int(seed)}"
    s = int.from_bytes(hashlib.blake2s(raw.encode(), digest_size=8).digest(), "little")
    return np.random.default_rng(s).permutation(n_file)[: int(budget)]


def sample_cells(
    reader: EvalNpzReader,
    keys: Mapping[str, str],
    max_cells: int,
    *,
    perturbed: bool,
    seed: int,
    debug: bool = False,
    budget: Optional[int] = None,
    subset_seed: int = 0,
) -> np.ndarray:
    """Up to ``max_cells`` rows for one condition; fewer when the pool is smaller.

    With a ``budget`` (a cell-budgeted split), rows come only from a fixed, seeded subset of
    exactly ``budget`` of the file's rows, the same every epoch: ``min(max_cells, budget)``
    of them per draw, without replacement.

    ``max_cells`` is a cap, never a target. Topping a short pool up by resampling
    with replacement would claim ``max_cells`` cells of evidence while term 1
    still only sees the distinct ones — so a 3-cell condition returns 3 rows and
    the grouped energy score divides by 3. The reader subsamples (without
    replacement) before densifying, so the cap also bounds the densify cost.
    """
    side = "pert" if perturbed else "ctrl"
    cap = int(max_cells)
    if budget is not None:
        full = (reader.perturbed_matrix if perturbed else reader.control_matrix)(dict(keys))
        n_file = int(full.shape[0])
        if not 0 < int(budget) <= n_file:
            raise ValueError(f"cell_set: budget {budget} for keys={dict(keys)!r} outside 1..{n_file} file rows")
        subset = _fixed_subset(keys, n_file, int(budget), subset_seed)
        take = np.sort(np.random.default_rng(int(seed)).choice(subset, size=min(cap, int(budget)), replace=False))
        rows = full[take]
        mat = np.asarray(rows.toarray() if hasattr(rows, "toarray") else rows, dtype=np.float32)
        cell_set_log(debug, "sample %s keys=%s n=%d cap=%d budget=%d seed=%d", side, dict(keys), len(take), cap,
                     int(budget), seed)
        return mat
    loader = reader.perturbed if perturbed else reader.control
    mat = np.asarray(loader(dict(keys), max_cells=cap, seed=int(seed)), dtype=np.float32)
    n = int(mat.shape[0])
    if n == 0:
        raise ValueError(f"cell_set: empty matrix for keys={dict(keys)!r}")
    cell_set_log(debug, "sample %s keys=%s n=%d cap=%d seed=%d", side, dict(keys), n, cap, seed)
    return mat


def encode_scalars(
    encoders: Mapping[str, Any],
    keys: Mapping[str, str],
    *,
    prefix: str = "",
) -> Dict[str, int]:
    """One int code per conditioning var (constant across the S cells in a set)."""
    return {
        f"{prefix}{var}_idx": int(enc.transform([str(keys.get(var, ""))])[0])
        for var, enc in encoders.items()
    }
