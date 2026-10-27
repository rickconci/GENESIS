"""Fit the four linear baselines for one split, from the eval cache, in one fit space.

    perturbation_mean  matched control + the perturbation's mean shift over held contexts
    context_mean       the context's mean expression over its train perturbations
    linear_ridge       the perturbation-mean table, ridge-fit in a gene PCA basis
    additive           matched control + mu + one main effect per axis

One code path for every dataset: the roles (src/baselines/roles.py) say which column is the
held axis, the kept axes, the stratum, the perturbation and the control unit, and each fit
is a formula over the resulting arm table. Each kind is saved as ``<kind>.npz``
(src/baselines/artifacts.py).

TRAIN = every non-control condition with >= min_cells cells MINUS the split's test and
validation queues: split YAMLs define only the held-out predicates. The perturbation shift
is averaged over contexts, so a test perturbation seen in OTHER contexts legitimately has
one; what is excluded are the held-out conditions themselves.

The control of a stratum is the cell-count-weighted mix of its control units (PBMC: the PBS
wells), read per unit, so a stratum whose control is held out still gets a delta control.
Those cells are never scored. Means are summed in float64 and stored in float32.

The fit space (src/baselines/space.py) is what each arm and control mean averages: raw counts, or
log1p(1e4 x / L_full) per cell as the reference papers fit (log1p_cp10k_full, on all cells). It is
saved in the artifact, so the scorer always emits a fit the matching way: both move real control
cells by the fitted delta (src/eval/arms/baseline.py).

HBSS splits (``train: {sample: hbss}``) are NOT applied here: the complement would
silently be 100% of train. Pass ``train_subset``, the run's own
hbss_selected_triplets.parquet, so the fit sees what the run saw; otherwise we raise.
"""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from src.baselines import additive, context_mean, linear_ridge, perturbation_mean
from src.baselines.artifacts import KINDS, Baseline
from src.baselines.roles import Arms, BaselineConditionRoles
from src.baselines.space import FIT_SPACES, RAW, FitSpaceReader
from src.data.splits.data_class import SplitConfig
from src.data.splits.predicates import apply_split_predicate
from src.eval.cache.eval_cache_io import load_work_queue_or_none
from src.eval.cache.eval_npz_reader import EvalNpzReader

log = logging.getLogger(__name__)


# 1. Define the train set: the complement of the held-out queues

def held_out(summary: Path, split_yaml: Path, roles: BaselineConditionRoles,
             min_cells: int) -> tuple[list[dict], dict[str, int]]:
    """Condition keys of the split's test + validation queues. Empty validation is fine
    (peek splits have none); an empty test queue is not."""
    keys, sizes = [], {}
    for split in ("test", "validation"):
        wq = load_work_queue_or_none(
            condition_summary_parquet=summary, split_yaml=split_yaml, eval_split=split,
            condition_variables=roles.condition_variables,
            control_condition=roles.control_condition, min_cells=min_cells,
        )
        sizes[split] = 0 if wq is None else len(wq)
        keys += [] if wq is None else [r.condition_keys for r in wq]
    if sizes["test"] == 0:
        raise SystemExit(f"TEST work queue is empty for {split_yaml}; nothing to hold out.")
    log.info("held out | test=%d validation=%d", sizes["test"], sizes["validation"])
    return keys, sizes


def require_subset_for_hbss(split_yaml: Path, train_subset: Optional[Path]) -> None:
    if train_subset is None and re.search(r"sample:\s*hbss", Path(split_yaml).read_text()):
        raise SystemExit(f"{split_yaml} subsamples train with HBSS, which the fit does not "
                         "apply; pass the run's hbss_selected_triplets.parquet as the train "
                         "subset or it would fit on 100% of train")


def subset_keys(path: Path, roles: BaselineConditionRoles, columns: Sequence[str]) -> set[tuple]:
    """Rows of a parquet as the same key tuples train rows match on. HBSS writes dose as
    float32 or as text ("0.05"); either way it must round like the train side or nothing matches."""
    frame = pd.read_parquet(path)
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise SystemExit(f"{path} lacks train-subset columns {missing}")
    return {_match_key(roles, r, columns) for r in frame[list(columns)].to_dict("records")}


def _match_key(roles: BaselineConditionRoles, keys: dict[str, Any], columns: Sequence[str]) -> tuple:
    """Compare conditions on a ROUNDED dose: summaries store dose as float32 (0.05 reads back as
    0.05000000074505806), so an exact match would miss held-out conditions and fit them as train."""
    return tuple(round(float(keys[c]), 6) if c == roles.dose_var else str(keys[c]) for c in columns)


def train_rows(summary: Path, roles: BaselineConditionRoles, held: list[dict], min_cells: int,
               train_subset: Optional[Path], limit: int) -> list[dict[str, str]]:
    """Sorted train conditions as {condition column: value as the cache names it}."""
    cond = roles.condition_variables
    frame = pd.read_parquet(summary)
    missing = [c for c in (*cond, "num_cells") if c not in frame.columns]
    if missing:
        raise ValueError(f"{summary} missing columns {missing}")
    frame = frame[(frame[roles.pert_var].astype(str) != roles.control_pert_value) & (frame["num_cells"] >= min_cells)]
    render = [(lambda v: str(float(v))) if c == roles.dose_var else str for c in cond]   # "5.0", as the cache keys doses
    rows = [dict(zip(cond, r)) for r in sorted({tuple(f(v) for f, v in zip(render, vals))
                                                for vals in zip(*(frame[c] for c in cond))})]
    excl = {_match_key(roles, k, cond) for k in held}
    rows = [r for r in rows if _match_key(roles, r, cond) not in excl]
    if train_subset is not None:
        cols = roles.held_vars + roles.pert_vars
        kept, n0 = subset_keys(train_subset, roles, cols), len(rows)
        rows = [r for r in rows if _match_key(roles, r, cols) in kept]
        log.info("train subset | %s: %d keys -> %d of %d conditions", Path(train_subset).name, len(kept), len(rows), n0)
    if limit > 0:
        rows = rows[:limit]
        log.warning("limit=%d | fit is a smoke test, not the panel artifact", limit)
    if not rows:
        raise SystemExit("TRAIN is empty: every non-control condition is held out. "
                         "Check the split YAML and condition summary agree on this dataset.")
    return rows


def check_disjoint(rows: list[dict[str, str]], split_yaml: Path, roles: BaselineConditionRoles) -> None:
    """Raise if any train row satisfies the split's own test or validation predicate.

    Train is a subtraction (summary minus the eval work queues), so a held-out condition the
    queue misses, or a key the subtraction fails to match, would be fitted with no error.
    Applying the YAML's predicates to
    the result is a second, independent path, so both have to agree.
    """
    cfg = SplitConfig.from_dict(yaml.safe_load(Path(split_yaml).read_text()) or {})
    frame = pd.DataFrame(rows)
    if roles.dose_var:
        frame[roles.dose_var] = frame[roles.dose_var].astype(float)
    for name, predicate in (("test", cfg.test), ("validation", cfg.validation)):
        if not predicate:   # an empty predicate selects every row; peek splits have no validation
            continue
        hit = frame[apply_split_predicate(frame, predicate)]
        if len(hit):
            raise SystemExit(f"{len(hit)} train conditions match the {name} predicate of {split_yaml}, "
                             f"e.g. {hit.iloc[0].to_dict()}: the fit would see held-out data")


# 2. Build the arm table: one control per stratum, one mean per train condition

def control_units(summary: Path, roles: BaselineConditionRoles) -> list:
    """Every control unit (PBMC: the PBS wells), in order of first appearance in the
    summary: that order fixes the float summation order of each mix. No unit column:
    one pool per stratum."""
    if roles.unit_var is None:
        return [None]
    frame = pd.read_parquet(summary)
    ctrl = frame.loc[frame[roles.pert_var].astype(str) == roles.control_pert_value, roles.unit_var]
    units = list(dict.fromkeys(ctrl.astype(str)))
    if not units:
        raise SystemExit(f"{summary} has no {roles.pert_var}={roles.control_pert_value} rows to mix")
    return units


def control_mean(reader: EvalNpzReader, roles: BaselineConditionRoles, stratum: dict[str, str],
                 units: list, max_cells: Optional[int]) -> tuple[np.ndarray, list[str]]:
    """Cell-count-weighted mean over the control units this stratum has on disk.

    A unit with no cells of this stratum has no file and adds nothing to a weighted
    mean, so the mix of the units that exist is the control (reported as partial).
    Unusable only when every unit is empty.
    """
    total, n, missing = None, 0, []
    for u in units:
        keys = {**stratum, **({roles.unit_var: u} if u is not None else {})}
        try:
            mat = reader.control(keys, max_cells=max_cells, seed=0)
        except FileNotFoundError:
            missing.append(str(u))
            continue
        if mat.ndim != 2 or mat.shape[0] == 0:
            missing.append(str(u))
            continue
        block = mat.astype(np.float64).sum(axis=0)
        total = block if total is None else total + block
        n += int(mat.shape[0])
    if total is None:
        raise ValueError(f"no control cells in units {units}")
    return (total / n).astype(np.float32), missing


def read_arms(reader: EvalNpzReader, roles: BaselineConditionRoles, rows: list[dict[str, str]], units: list,
              n_genes: int, max_cells: Optional[int], workers: int) -> tuple[Arms, dict]:
    """The arm table, in ``rows`` order. An arm whose stratum has no control, or whose
    cells cannot be read, is dropped and logged; the counts go into the fit summary."""
    match = roles.control_match_keys
    stratum_of = [tuple(r[c] for c in match) for r in rows]
    ctrl, fail, partial = {}, [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {s: pool.submit(control_mean, reader, roles, dict(zip(match, s)), units, max_cells)
                   for s in sorted(set(stratum_of))}
        for s, fut in futures.items():
            try:
                mu, missing = fut.result()
            except ValueError as exc:
                fail.append(f"{s}: {exc}")
                log.error("control | %s | %s; its arms are dropped", s, exc)
                continue
            ctrl[s] = _check_genes(mu, n_genes, f"control {s}")
            if missing:
                partial.append(f"{s}: missing {','.join(missing)}")
                log.warning("partial control mix | %s: missing units %s", s, ",".join(missing))
    log.info("controls | ok=%d empty=%d partial=%d", len(ctrl), len(fail), len(partial))

    def read(row: dict[str, str]) -> np.ndarray:
        mat = reader.perturbed(row, max_cells=max_cells, seed=0)
        return mat.astype(np.float64, copy=False).mean(axis=0).astype(np.float32)

    readable = [(r, s) for r, s in zip(rows, stratum_of) if s in ctrl]
    keys, means, ctrls, read_fail = [], [], [], 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for (row, s), fut in zip(readable, [pool.submit(read, r) for r, _ in readable]):
            try:
                mu = fut.result()
            except (FileNotFoundError, ValueError) as exc:
                read_fail += 1
                log.error("perturbed read failed | %s | %s", row, exc)
                continue
            keys.append(roles.arm_key(row))
            means.append(_check_genes(mu, n_genes, f"perturbed {row}"))
            ctrls.append(ctrl[s])
    log.info("perturbed | ok=%d fail=%d no_control=%d", len(keys), read_fail, len(rows) - len(readable))
    if not keys:
        raise RuntimeError("no train arm could be read")
    report = {"n_train_conditions": len(rows), "n_arms_fit": len(keys), "n_read_fail": read_fail,
              "n_skipped_no_control": len(rows) - len(readable), "n_control_strata": len(ctrl),
              "n_control_fail": len(fail), "control_fail": fail,
              "n_control_partial": len(partial), "control_partial": partial}
    return Arms.build(keys, np.stack(means), np.stack(ctrls)), report


def _check_genes(vec: np.ndarray, n_genes: int, where: str) -> np.ndarray:
    if vec.shape != (n_genes,):
        raise ValueError(f"{where} has {vec.shape} genes, the gene list has {n_genes}")
    return vec


# 3. Fit the baselines: each kind a formula over the arm table

def fit_kinds(arms: Arms, roles: BaselineConditionRoles, genes: Sequence[str], kinds: set[str], *,
              pca_dim: int = 100, ridge_lambda: float = 0.1, fit_space: str = RAW) -> dict[str, Baseline]:
    """The requested kinds as artifacts. The ridge regresses the perturbation-mean table,
    so it is fit whenever the ridge is asked for, saved or not."""
    if not kinds or not kinds <= set(KINDS):
        raise ValueError(f"kinds {sorted(kinds)} must be a non-empty subset of {KINDS}")
    tables: dict[str, dict] = {}
    shift = perturbation_mean.fit(arms) if kinds & {"perturbation_mean", "linear_ridge"} else None
    if "perturbation_mean" in kinds:
        tables["perturbation_mean"] = {"shift": shift}
    if "linear_ridge" in kinds:
        tables["linear_ridge"] = {"shift": linear_ridge.fit(arms, shift, pca_dim=pca_dim, ridge_lambda=ridge_lambda)}
    if "context_mean" in kinds:
        tables["context_mean"] = {"context": context_mean.fit(arms)}
    if "additive" in kinds:
        mu, effects = additive.fit(arms)
        tables["additive"] = {"mu": {(): mu}, **dict(zip(additive.term_names(list(roles.kept_vars)), effects))}
    params = {"pca_dim": pca_dim, "ridge_lambda": ridge_lambda, "n_arms": len(arms.keys), "fit_space": fit_space}
    for kind, t in tables.items():
        log.info("fit %s | %s", kind, {name: len(v) for name, v in t.items()})
    return {k: Baseline(k, tuple(genes), roles.axes, tables[k], params) for k in KINDS if k in tables}


def fit_split(*, reader: EvalNpzReader, roles: BaselineConditionRoles, summary: Path, split_yaml: Path,
              genes: list[str], out_dir: Path, kinds: set[str], fit_cells: int, workers: int,
              ridge_lambda: float, pca_dim: int, min_cells: int, train_subset: Optional[Path],
              limit: int, fit_space: str = RAW) -> dict:
    """Fit and save ``<kind>.npz`` per kind; returns the fit summary. A log1p_cp10k_full fit needs a
    reader on the ``__with_libfull`` namespace (space.library_namespace) and all cells."""
    if fit_space not in FIT_SPACES:
        raise ValueError(f"fit_space {fit_space!r} not in {FIT_SPACES}")
    if fit_space != RAW:
        if fit_cells > 0:
            raise SystemExit(f"{fit_space} pairs each cell with its L_full: fit on all cells (--fit-cells 0)")
        reader = FitSpaceReader(reader)
    require_subset_for_hbss(split_yaml, train_subset)
    held, sizes = held_out(summary, split_yaml, roles, min_cells)
    rows = train_rows(summary, roles, held, min_cells, train_subset, limit)
    check_disjoint(rows, split_yaml, roles)
    units = control_units(summary, roles)
    arms, report = read_arms(reader, roles, rows, units, len(genes), fit_cells if fit_cells > 0 else None, workers)
    fits = fit_kinds(arms, roles, genes, kinds, pca_dim=pca_dim, ridge_lambda=ridge_lambda, fit_space=fit_space)
    for kind, fit in fits.items():
        fit.save(out_dir / f"{kind}.npz")
        log.info("saved | %s (%.1f MB)", out_dir / f"{kind}.npz", (out_dir / f"{kind}.npz").stat().st_size / 1e6)
    return {"kinds": sorted(kinds), "fit_space": fit_space, "axes": roles.axes, "control_unit": roles.unit_var, "control_units": units,
            "n_test": sizes["test"], "n_validation": sizes["validation"], **report,
            "tables": {k: {name: len(t) for name, t in f.tables.items()} for k, f in fits.items()}}


def write_fit_summary(out_dir: Path, fields: dict, kinds: set[str]) -> Path:
    """fit_summary.json; a partial refit (e.g. only additive) gets its own name so it
    never clobbers the full fit's summary."""
    name = "fit_summary.json" if set(kinds) == set(KINDS) else f"fit_summary__{'__'.join(sorted(kinds))}.json"
    fields = {**fields, "generated_utc": datetime.now(timezone.utc).isoformat()}
    (out_dir / name).write_text(json.dumps(fields, indent=2))
    return out_dir / name
