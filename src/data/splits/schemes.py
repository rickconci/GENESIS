"""The three split schemes behind the published splits, generic over datasets.

  scaling       same holdout; train on fewer conditions (HBSS) or fewer cells
  overlap_bio   each held-out bio entity sees k of its train perturbations
  overlap_pert  each held-out perturbation trains in k of a fixed pool of bios

Overlap schemes are Latin squares: entity i on row r gets ``levels[(i - r) % n]``,
so every entity visits every level once and every row spans all levels. A level
is a prefix of one fixed queue per (seed, entity), so 15 ⊂ 30 ⊂ 80 and the seed
only reorders the queue.

Each scheme is pure: (summary, policy family) -> list[GeneratedSplit]. Column
names come from the dataset registry (``Axes``), constants from the policy YAML.
``scripts/data/make_splits.py`` writes the result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config.dataset_registry import DatasetRegistry
from src.data.splits.data_class import SplitConfig
from src.data.splits.generation import (
    REPO_ROOT,
    SplitPolicy,
    estimate_train_cells,
    subseed,
    train_block,
    validate_split_on_summary,
)
from src.data.splits.predicates import split_membership_masks


@dataclass(frozen=True)
class Axes:
    """Which columns mean bio, perturbation and dose for this dataset."""

    bio: str
    pert: str
    dose: Optional[str]
    control: str
    control_pert: Mapping[str, Any]

    @classmethod
    def from_registry(cls, registry: DatasetRegistry) -> "Axes":
        bios, perts = list(registry.biological_context_columns), list(registry.perturbation_columns)
        if len(bios) != 1 or not 1 <= len(perts) <= 2:
            raise ValueError(f"split schemes need 1 bio and 1-2 pert columns; got {bios}, {perts}")
        return cls(bio=bios[0], pert=perts[0], dose=perts[1] if len(perts) == 2 else None,
                   control=str(registry.control_pert[perts[0]]), control_pert=dict(registry.control_pert))


@dataclass(frozen=True)
class GeneratedSplit:
    rel_path: Path            # relative to the family's out_dir
    header: str
    payload: Dict[str, Any]
    manifest: Dict[str, Any]


def canon_dose(value: object, doses: Sequence[float]) -> float:
    """Snap a float32-ish dose onto the canonical list. Anything else is a bug."""
    x = float(value)  # type: ignore[arg-type]
    if x != x:
        raise ValueError(f"NaN dose {value!r}")
    for canon in doses:
        if abs(x - canon) < 1e-6:
            return canon
    raise ValueError(f"dose {value!r} is not one of {tuple(doses)}")


def held_entities(blocks: Any, column: str) -> List[str]:
    """Entities a val/test predicate names. A mapping is one block, not a list of keys."""
    blocks = blocks if isinstance(blocks, list) else [blocks or {}]
    return sorted({str(e) for blk in blocks for e in (blk.get(column) or [])})


def latin_level(i: int, row: int, levels: Sequence[int]) -> int:
    if not 0 <= i < len(levels) or not 0 <= row < len(levels):
        raise ValueError(f"entity {i} / row {row} outside a {len(levels)}-level square")
    return levels[(i - row) % len(levels)]


def seeded_queue(items: Sequence[str], *seed_parts: object) -> List[str]:
    """Sorted items, then one permutation fixed by ``seed_parts``."""
    ordered = sorted(set(items))
    if len(ordered) != len(list(items)):
        raise ValueError(f"duplicate entity in queue {seed_parts}")
    rng = np.random.default_rng(subseed(*seed_parts))
    return [ordered[i] for i in rng.permutation(len(ordered))]


# The scaling scheme: the holdout is copied verbatim; only `train` changes

def scaling_splits(summary: pd.DataFrame, policy: SplitPolicy, registry: DatasetRegistry,
                   family: Mapping[str, Any], base: Mapping[str, Any]) -> List[GeneratedSplit]:
    seed = int(family.get("seed", 0))
    # The dataset tag comes from the policy; a legacy base split may carry an old alias.
    holdout = {**{k: v for k, v in base.items() if k != "train"}, "dataset": policy.split_dataset}
    held = held_entities(base.get("test"), str(policy.hbss["candidate_axis"]))
    if not held:
        raise ValueError(f"base split names no held-out {policy.hbss['candidate_axis']!r}")
    targets = {int(k): int(v) for k, v in dict(family.get("cell_targets") or {}).items()}
    twins = twin_cell_targets(family, seed, REPO_ROOT)
    if set(twins) - set(map(int, family["cells_pcts"])):
        raise ValueError(f"cell_targets_from names {sorted(twins)}% but cells_pcts is {family['cells_pcts']}")
    for pct, n in twins.items():
        if targets.get(pct, n) != n:
            raise ValueError(f"cell_targets[{pct}] = {targets[pct]:,} but its conds twin drew {n:,} perturbed cells")
        targets[pct] = n
    keep_ctrl = bool(family.get("cells_keep_controls", False))   # cell legs budget perturbed rows only
    legs = [("conds", int(p), None) for p in family["conds_pcts"]]
    for pct in family["cells_pcts"]:
        n = targets.get(int(pct)) or estimate_train_cells(
            summary, holdout, registry=registry, hbss=policy.hbss, hbss_pct=int(pct),
            seed=seed, method=str(family.get("cells_estimate", "hbss")))
        legs.append(("cells", int(pct), n))
    out = []
    for mode, pct, n in legs:
        tag = f"hbss{pct:03d}" if n is None else str(family.get("cells_tag", "celln{n}_mhbss{pct:03d}")).format(n=n, pct=pct)
        name = str(family["name"]).format(tag=tag, seed=seed)
        rel = Path(name) if family.get("flat") else Path(f"entity_seed{seed}") / name
        payload = {**holdout, "train": train_block(hbss_pct=pct, hbss=policy.hbss, seed=seed, cells_target=n,
                                                   cells_keep_controls=keep_ctrl and n is not None)}
        manifest = {
            "scheme": "scaling", "dataset": policy.dataset_alias, "scaling_mode": mode, "entity_seed": seed,
            "hbss_pct": pct if n is None else None, "celln_target": n, "celln_match_hbss_pct": pct if n else None,
            "base_split": str(family["base_split"]), "holdout_entities": held, "yaml": str(rel)}
        if keep_ctrl and n is not None:
            manifest["cells_keep_controls"] = True
        out.append(GeneratedSplit(rel, _scaling_header(policy, family, mode, pct, n, seed, held, keep_ctrl),
                                  payload, manifest))
    return out


def twin_cell_targets(family: Mapping[str, Any], seed: int, root: Path) -> Dict[int, int]:
    """Perturbed cells each matched conds run drew, from ``cell_targets_from: {pct: run dir}``.

    Optional per family: cells-only or conds-only families leave it out. Never re-estimate a
    twin's count: legacy HBSS breaks ties per CPU, so only the finished run knows what it drew.
    """
    out: Dict[int, int] = {}
    for pct, rel in dict(family.get("cell_targets_from") or {}).items():
        path = root / str(rel) / "split_cache" / "hbss_meta.json"
        if not path.is_file():
            raise FileNotFoundError(f"cell_targets_from[{pct}]: no {path}; drop the key to use cell_targets alone")
        meta = json.loads(path.read_text(encoding="utf-8"))
        if (round(float(meta["fraction"]) * 100), int(meta["seed"])) != (int(pct), int(seed)):
            raise ValueError(f"cell_targets_from[{pct}] drew {meta['fraction']} at seed {meta['seed']}, "
                             f"not {pct}% at seed {seed}: wrong twin")
        out[int(pct)] = int(meta["n_train_rows_pert_after"])
    return out


def _scaling_header(policy: SplitPolicy, family: Mapping[str, Any], mode: str, pct: int,
                    n: Optional[int], seed: int, held: Sequence[str], keep_ctrl: bool = False) -> str:
    what = f"train=all conditions, cells capped at {n:,}" if n else f"train=hbss{pct:03d} (HBSS condition fraction)"
    if n and keep_ctrl:
        what = f"train=all conditions, every control kept, perturbed cells capped at {n:,}"
    return (f"# Generated by scripts/data/make_splits.py --dataset {policy.dataset_alias} (scheme=scaling)\n"
            f"# scaling_mode={mode} pct={pct} seed={seed}; {what}\n"
            f"# validation/test copied verbatim from {Path(str(family['base_split'])).name}: "
            f"{len(held)} held-out {policy.hbss['candidate_axis']}\n")


# The overlap_bio scheme: each held-out bio entity keeps a prefix of its
# train-perturbation queue; the rest of that pool goes to train_exclusions.

def predicate_pairs(blocks: Any, axes: Axes, doses: Sequence[float], held: Sequence[str]) -> set:
    """(perturbation, dose) pairs a val/test predicate names, same on every held-out entity."""
    pairs: set = set()
    for block in blocks if isinstance(blocks, list) else [blocks]:
        if set(map(str, block[axes.bio])) != set(held):
            raise ValueError(f"holdout block names {sorted(block[axes.bio])}, expected {list(held)}")
        ds = [canon_dose(d, doses) for d in block[axes.dose]] if axes.dose else [None]
        pairs |= {(str(p), d) for p in block[axes.pert] for d in ds}
    return pairs


def train_pools(summary: pd.DataFrame, axes: Axes, doses: Sequence[float], held: Sequence[str], *,
                val_pairs: set, test_pairs: set) -> Dict[str, Dict[str, Tuple]]:
    """Per held-out entity: perturbation -> the doses it currently trains at.

    All held-out entities must share one perturbation pool; the ladder prefixes it.
    """
    if val_pairs & test_pairs:
        raise ValueError(f"val and test name the same pair: {sorted(val_pairs & test_pairs)[:5]}")
    frame = summary.loc[summary[axes.bio].astype(str).isin(held)].copy()
    frame[axes.pert] = frame[axes.pert].astype(str)
    frame = frame.loc[frame[axes.pert] != axes.control]
    frame["_dose"] = frame[axes.dose].map(lambda d: canon_dose(d, doses)) if axes.dose else None
    pools: Dict[str, Dict[str, Tuple]] = {}
    for ent in held:
        by_pert: Dict[str, set] = {}
        sub = frame.loc[frame[axes.bio].astype(str) == ent]
        for pert, dose in zip(sub[axes.pert], sub["_dose"], strict=True):
            by_pert.setdefault(pert, set()).add(dose)
        held_out = val_pairs | test_pairs
        train = {p: tuple(sorted(d for d in ds if (p, d) not in held_out)) for p, ds in by_pert.items()}
        pools[ent] = {p: ds for p, ds in train.items() if ds}
        if not pools[ent]:
            raise ValueError(f"{ent}: no train perturbations in the condition summary")
    if any(set(pools[e]) != set(pools[held[0]]) for e in held):
        raise ValueError("held-out entities do not share one train-perturbation pool")
    return pools


def exclusion_blocks(ent: str, pool: Mapping[str, Tuple], peeked: Sequence[str], axes: Axes) -> List[dict]:
    """Withhold every current train dose of the perturbations outside the prefix."""
    if set(peeked) - set(pool):
        raise ValueError(f"{ent}: peeked perturbations not in the pool: {sorted(set(peeked) - set(pool))[:5]}")
    by_doses: Dict[Tuple, List[str]] = {}
    for pert, ds in pool.items():
        if pert not in set(peeked):
            by_doses.setdefault(ds, []).append(pert)
    blocks = []
    for ds in sorted(by_doses):
        block = {axes.bio: [ent], axes.pert: sorted(by_doses[ds])}
        if axes.dose:
            block[axes.dose] = list(ds)
        blocks.append(block)
    return blocks


def realized_overlap(pools: Mapping[str, Mapping[str, Tuple]], payload: Mapping, axes: Axes,
                     doses: Sequence[float]) -> Dict[str, int]:
    """Perturbations per held-out entity that still keep a train dose after exclusions."""
    excluded = set()
    for blk in payload.get("train_exclusions") or []:
        ds = [canon_dose(d, doses) for d in blk[axes.dose]] if axes.dose else [None]
        excluded |= {(blk[axes.bio][0], str(p), d) for p in blk[axes.pert] for d in ds}
    return {ent: sum(any((ent, p, d) not in excluded for d in ds) for p, ds in pool.items())
            for ent, pool in pools.items()}


def overlap_bio_splits(summary: pd.DataFrame, policy: SplitPolicy, registry: DatasetRegistry,
                       family: Mapping[str, Any], base: Mapping[str, Any], name: str) -> List[GeneratedSplit]:
    axes, doses, levels = Axes.from_registry(registry), policy.doses, tuple(family["levels"])
    held = held_entities(base["test"], axes.bio)
    if len(held) != len(levels):
        raise ValueError(f"{len(held)} held-out entities need {len(held)} levels; got {list(levels)}")
    pools = train_pools(summary, axes, doses, held, val_pairs=predicate_pairs(base["validation"], axes, doses, held),
                        test_pairs=predicate_pairs(base["test"], axes, doses, held))
    pool_size = len(pools[held[0]])
    if max(levels) > pool_size:
        raise ValueError(f"level {max(levels)} exceeds the {pool_size}-perturbation pool")
    budget = family.get("cell_budget") or {}
    out = []
    for seed in family["seeds"]:
        for row in range(len(held)):
            k = {e: latin_level(i, row, levels) for i, e in enumerate(held)}
            peeked = {e: seeded_queue(list(pools[e]), seed, e)[: k[e]] for e in held}
            payload = {"dataset": policy.split_dataset, "controls_to_train": True, "validation": base["validation"],
                       "test": base["test"], "train": train_block(cells_target=budget.get("target_n"),
                                                                  seed=int(budget.get("seed", 0)))}
            excl = [b for e in held for b in exclusion_blocks(e, pools[e], peeked[e], axes)]
            if excl:
                payload["train_exclusions"] = excl
            if realized_overlap(pools, payload, axes, doses) != k:
                raise RuntimeError(f"seed={seed} row={row}: realised overlap != Latin levels {k}")
            counts = validate_split_on_summary(summary, payload, axes.control_pert)
            rel = Path(f"entity_seed{seed}") / f"{name}_r{row}_es{seed}.yaml"
            header = (f"# Generated by scripts/data/make_splits.py --dataset {policy.dataset_alias} "
                      f"--family {name} (scheme=overlap_bio)\n# entity_seed={seed} row=r{row}; levels {list(levels)}; "
                      f"this row: {', '.join(f'{e}={k[e]}' for e in held)}\n")
            out.append(GeneratedSplit(rel, header, payload, {
                "scheme": "overlap_bio", "dataset": policy.dataset_alias, "entity_seed": seed, "row": row,
                "levels": list(levels), "peek_per_bio": k, "peeked_perts": peeked, "pool_size": pool_size,
                "cell_budget": budget or None, **{f"{c}_conditions": v for c, v in counts.items()},
                "yaml": str(rel)}))
    return out


# The overlap_pert scheme: one fixed test block (test bios x held-out perts);
# each held-out perturbation trains in a k-prefix of its queue over a bio pool.

def stratified_pool(strata: Mapping[str, str], n_pool: int, seed: int) -> List[str]:
    """Draw n_pool bios so each stratum gets its share (largest remainder).

    One stratum reduces to a uniform draw. A remainder tie at the cutoff would
    need an arbitrary choice, so it raises.
    """
    if not 0 < n_pool < len(strata):
        raise ValueError(f"n_pool={n_pool} must be in 1..{len(strata) - 1}")
    by: Dict[str, List[str]] = {}
    for bio, stratum in strata.items():
        if not isinstance(stratum, str) or not stratum:
            raise ValueError(f"{bio}: missing stratum {stratum!r}")
        by.setdefault(stratum, []).append(bio)
    n = len(strata)
    seats = {s: len(b) * n_pool // n for s, b in by.items()}
    rem = {s: len(b) * n_pool % n for s, b in by.items()}
    extra, ranked = n_pool - sum(seats.values()), sorted(rem, key=lambda s: (-rem[s], s))
    if 0 < extra < len(ranked) and rem[ranked[extra - 1]] == rem[ranked[extra]]:
        raise ValueError(f"stratum remainder tie at the cutoff: {ranked[extra - 1]} vs {ranked[extra]}")
    for s in ranked[:extra]:
        seats[s] += 1
    rng = np.random.default_rng(subseed("pool", seed))
    pool: List[str] = []
    for s in sorted(by):
        bios = sorted(by[s])
        pool += [bios[i] for i in rng.permutation(len(bios))[: seats[s]]]
    return sorted(pool)


def draw_groups(perts: Sequence[str], *, control: str, n_test: int, n_groups: int,
                label: str, seed: int) -> List[List[str]]:
    """Uniform draw of n_test non-control perturbations, split into n_groups."""
    if n_test % n_groups:
        raise ValueError(f"{n_test} perturbations do not split into {n_groups} groups")
    ordered = sorted(set(map(str, perts)) - {control})
    if n_test > len(ordered):
        raise ValueError(f"n_test={n_test} exceeds the {len(ordered)} non-control perturbations")
    rng = np.random.default_rng(subseed(label, seed))
    picked = [ordered[i] for i in rng.permutation(len(ordered))[:n_test]]
    size = n_test // n_groups
    return [sorted(picked[g * size:(g + 1) * size]) for g in range(n_groups)]


def check_realized(summary: pd.DataFrame, payload: Mapping, peeked: Mapping[str, Sequence[str]],
                   test_bios: Sequence[str], axes: Axes, doses: Sequence[float]) -> Dict[str, int]:
    """Raise unless train/val/test on the summary are exactly what the design says."""
    cols = [axes.bio, axes.pert] + ([axes.dose] if axes.dose else [])
    frame = summary.loc[:, cols].astype({axes.bio: str, axes.pert: str})
    is_train, is_val, is_test, _ = split_membership_masks(frame, SplitConfig.from_dict(dict(payload)), axes.control_pert)
    if int(is_train.sum() + is_val.sum() + is_test.sum()) != len(frame):
        raise RuntimeError("a summary row is in more than one split, or in none")
    trained = frame.loc[is_train].groupby(axes.pert)[axes.bio].agg(set)
    for pert in frame[axes.pert].unique():
        want = set(peeked[pert]) if pert in peeked else set(frame[axes.bio])
        if trained.get(pert, set()) != want:
            raise RuntimeError(f"{pert}: trains in {len(trained.get(pert, set()))} bios, expected {len(want)}")
    test = frame.loc[is_test]
    if set(zip(test[axes.bio], test[axes.pert])) != {(b, p) for b in test_bios for p in peeked}:
        raise RuntimeError("test pairs differ from test_bios x held-out perturbations")
    if axes.dose and test[axes.dose].map(lambda d: canon_dose(d, doses)).nunique() != len(doses):
        raise RuntimeError("test does not cover every canonical dose")
    return {"n_train": int(is_train.sum()), "n_val": int(is_val.sum()), "n_test": int(is_test.sum())}


def _pert_payload(policy: SplitPolicy, axes: Axes, family: Mapping[str, Any], test_bios: Sequence[str],
                  pool: Sequence[str], groups: Sequence[Sequence[str]], *, seed: int, row: int):
    """One run's YAML payload, plus the pool bios each held-out perturbation trains in."""
    def dose() -> dict:  # a fresh list per block: a shared one is written as a YAML anchor
        return {axes.dose: list(policy.doses)} if axes.dose else {}
    peeked, staging = {}, []
    for g, perts in enumerate(groups):
        k = latin_level(g, row, family["levels"])
        for pert in perts:
            queue = seeded_queue(pool, seed, pert)
            peeked[pert] = queue[:k]
            if queue[k:]:  # an empty validation predicate would match every row
                staging.append({axes.bio: sorted(queue[k:]), axes.pert: [pert], **dose()})
    if not staging:
        raise ValueError(f"row {row}: no validation staging; every perturbation is at k={len(pool)}")
    budget = family.get("cell_budget") or {}
    payload = {"dataset": policy.split_dataset, "controls_to_train": True, "validation": staging,
               "test": [{axes.bio: sorted(test_bios), axes.pert: sorted(p for grp in groups for p in grp), **dose()}],
               "train": train_block(cells_target=budget.get("target_n"), seed=int(budget.get("seed", 0)))}
    return payload, peeked


def overlap_pert_splits(summary: pd.DataFrame, policy: SplitPolicy, registry: DatasetRegistry,
                        family: Mapping[str, Any], name: str) -> List[GeneratedSplit]:
    axes, levels, sel = Axes.from_registry(registry), list(family["levels"]), int(family["selection_seed"])
    by_bio = summary.astype({axes.bio: str}).drop_duplicates(axes.bio).set_index(axes.bio)
    col = family.get("stratify_by")
    strata = {b: (str(v) if col else "all") for b, v in (by_bio[col] if col else by_bio[axes.pert]).items()}
    pool = stratified_pool(strata, int(family["n_pool_bios"]), sel)
    test_bios = sorted(set(strata) - set(pool))
    # The label is part of the seed: "<pert column>s" reproduces the shipped draws.
    groups = draw_groups(summary[axes.pert].unique(), control=axes.control, n_test=int(family["n_test_perts"]),
                         n_groups=len(levels), label=f"{axes.pert}s", seed=sel)
    out, tests = [], set()
    for seed in family["seeds"]:
        for row in range(len(levels)):
            payload, peeked = _pert_payload(policy, axes, family, test_bios, pool, groups, seed=seed, row=row)
            counts = check_realized(summary, payload, peeked, test_bios, axes, policy.doses)
            tests.add(repr(payload["test"]))
            rel = Path(f"entity_seed{seed}") / f"{name}_r{row}_es{seed}.yaml"
            header = (f"# Generated by scripts/data/make_splits.py --dataset {policy.dataset_alias} "
                      f"--family {name} (scheme=overlap_pert)\n# entity_seed={seed} row=r{row}; levels {levels}; "
                      f"group g gets levels[(g - row) % {len(levels)}]. Test block is identical in every run.\n")
            out.append(GeneratedSplit(rel, header, payload, {
                "scheme": "overlap_pert", "dataset": policy.dataset_alias, "entity_seed": seed, "row": row,
                "levels": levels, "selection_seed": sel, "test_bios": test_bios, "pool_bios": pool,
                "pert_groups": groups, "peek_per_pert": {p: len(c) for p, c in peeked.items()},
                "peeked_bios": peeked, "cell_budget": family.get("cell_budget"),
                **{f"{c}_conditions": v for c, v in counts.items()}, "yaml": str(rel)}))
    if len(tests) != 1:
        raise RuntimeError(f"test block differs across runs: {len(tests)} distinct")
    return out
