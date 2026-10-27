"""Figure 3b's numbers: per-cell-line, per-seed overlap curves for the Tahoe bio Latin runs.

Each trained model is one Latin row: the five holdout lines sit at five different drug-overlap levels in
the same run. The curve is therefore not the pooled test score. Test conditions are averaged inside a cell
line, then labelled with that line's k from the split manifest. Figure 3's extract writes the strict-drug
table (main/overlap_bio/cell_seed_overlap_strict_drug.csv).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.figures.figure3.score.scaling import BIO_LEVELS as LEVELS
from scripts.figures.figure3.score.scaling import CONTROL_DRUG

REPO = Path(__file__).resolve().parents[4]
CELL_LINES: tuple[str, ...] = ("C32", "HOP62", "HepG2/C3A", "Hs 766T", "PANC-1")
ENTITY_SEEDS: tuple[int, ...] = (0, 1, 2)
# k=0 trains only DMSO_TF. The axis counts that unperturbed condition as 1.
# 15/30/80/164 stay the number of peeked drugs (DMSO is still in train).
DISPLAY_K = {0: 1, 15: 15, 30: 30, 80: 80, 164: 164}
# Same headline trio as Figure 2: adaptive-k truth DEG @100 (raw counts), and Pearson r Δ
# and energy distance in log1p_slice_med_rgs, each cell scaled to a common panel library
# (Fig 3b follows the depth-scaled space of every other panel).
METRICS: tuple[tuple[str, str, str], ...] = (
    (
        "deg_overlap_k100",
        "deg_raw_int_rgs/deg_overlap_top_p_ranked_k100_fdr0p001_lfc0p1_adaptivek_truth_detfloor10",
        "DEG overlap @100",
    ),
    ("pearson_delta", "log1p_slice_med_rgs/pearson_r_diff", "Pearson r Δ"),
    ("energy_distance", "log1p_slice_med_rgs/energy_distance", "Energy distance"),
)
_RUN_ID = re.compile(r"tahoe_bio_celln10_r(\d)_es(\d)")

DEFAULT_MANIFEST = REPO / "core_util_data/splits/tahoe/bio/tahoe_overlap_bio/manifest.json"
DEFAULT_RUNS = REPO / "runs/tahoe/bio/scaling/overlap_celln10"


def run_id_from_path(path: str) -> tuple[int, int]:
    """Latin row and entity seed baked into the run directory name."""
    m = _RUN_ID.search(path)
    if not m:
        raise ValueError(f"cannot parse row/seed from {path}")
    return int(m.group(1)), int(m.group(2))


def metric_value(condition: Mapping, key: str) -> float | None:
    """Pull one metric. Absent means the score is undefined (truth called no DEGs).

    A present NaN is a bug, not that case: the writer drops undefined scores
    instead of storing NaN.
    """
    metrics = condition.get("metrics") or {}
    if not isinstance(metrics, dict):
        raise TypeError(f"metrics is {type(metrics).__name__}, expected dict")
    for group in metrics.values():
        if isinstance(group, dict) and key in group:
            raw = group[key]
            value = float(raw)
            if not np.isfinite(value):
                raise ValueError(f"{key} is not finite: {raw!r}")
            return value
    return None


def strict_test_drugs(
    test_pairs: set[tuple[str, float]],
    pools: Mapping[str, Mapping[str, tuple[float, ...]]],
) -> set[str]:
    """Test drugs with no dose in any held-out line's peek pool.

    The base split tests 268 drugs per line, but 69 keep their 5 µM dose in the
    pool, so at high k one dose trains while its sibling doses are scored. Only
    drugs that can never train on a held-out line give a strict by-drug test.
    """
    pooled = set().union(*(set(p) for p in pools.values()))
    strict = {d for d, _ in test_pairs} - pooled
    if not strict:
        raise ValueError("every test drug has a dose in the peek pool")
    return strict


def mean_by_cell(
    summary: Mapping,
    peek_per_cell: Mapping[str, int],
    metric_keys: Sequence[str],
    drugs: set[str] | None = None,
) -> list[dict]:
    """One row per (cell line, metric): mean over that line's test conditions.

    DMSO is not a peeked-drug outcome. Any other cell line is outside the
    five-line holdout and means the test block moved. `drugs`, if given,
    restricts the mean to those test drugs.
    """
    if str(summary.get("eval_split", "")).lower() != "test":
        raise ValueError(f"expected eval_split=test, got {summary.get('eval_split')!r}")
    buckets: dict[tuple[str, str], list[float]] = {}
    n_undefined: dict[tuple[str, str], int] = {}
    n_control = 0
    for cond in summary.get("results") or []:
        keys = cond.get("condition_keys") or {}
        drug = str(keys.get("drug", ""))
        cell = str(keys.get("cell_name", ""))
        if drug == CONTROL_DRUG:
            n_control += 1
            continue
        if cell not in peek_per_cell:
            raise KeyError(f"test cell {cell!r} is not in the Latin assignment")
        if drugs is not None and drug not in drugs:
            continue
        for mk in metric_keys:
            value = metric_value(cond, mk)
            if value is None:
                n_undefined[(cell, mk)] = n_undefined.get((cell, mk), 0) + 1
                continue
            buckets.setdefault((cell, mk), []).append(value)
    if n_control:
        print(f"  skipped {n_control} {CONTROL_DRUG} conditions")
    rows = []
    for cell, k in peek_per_cell.items():
        for mk in metric_keys:
            vals = buckets.get((cell, mk))
            if not vals:
                raise RuntimeError(f"no defined scores for cell={cell} metric={mk}")
            skipped = n_undefined.get((cell, mk), 0)
            if skipped:
                print(f"  {cell} {mk.split('/')[-1]}: {skipped} conditions undefined")
            rows.append({
                "cell_line": cell,
                "overlap": int(k),
                "metric_key": mk,
                "metric_mean": float(np.mean(vals)),
                "n_conditions": len(vals),
                "n_undefined": skipped,
            })
    return rows


def load_curves(manifest_path: Path, runs_root: Path, drugs: set[str] | None = None) -> pd.DataFrame:
    """Join every after-fit test summary onto its manifest peek map."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_id = {(int(e["row"]), int(e["entity_seed"])): e for e in manifest}
    if len(by_id) != len(manifest):
        raise ValueError("manifest has two entries for the same (row, entity_seed)")
    summaries = sorted(runs_root.glob("**/after_fit/test/metrics_summary.json"))
    if not summaries:
        raise FileNotFoundError(f"no after_fit test summaries under {runs_root}")
    metric_keys = [mk for _short, mk, _label in METRICS]
    rows: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for path in summaries:
        summary = json.loads(path.read_text(encoding="utf-8"))
        row, seed = run_id_from_path(str(summary.get("run_dir") or path))
        if (row, seed) in seen:
            raise RuntimeError(f"two summaries for row={row} seed={seed}")
        seen.add((row, seed))
        entry = by_id.get((row, seed))
        if entry is None:
            raise KeyError(f"no manifest entry for row={row} seed={seed}")
        peek = {str(c): int(k) for c, k in entry["peek_per_cell"].items()}
        if drugs is not None:
            # The split is the claim; the run's own peek list is the check.
            leaked = drugs & {str(d) for ds in entry["peeked_drugs"].values() for d in ds}
            if leaked:
                raise ValueError(f"r{row} es{seed} peeked strict drugs: {sorted(leaked)[:5]}")
        print(f"r{row} es{seed}  {path.parent}")
        for rec in mean_by_cell(summary, peek, metric_keys, drugs):
            rec["row"] = row
            rec["entity_seed"] = seed
            rows.append(rec)
    frame = pd.DataFrame(rows)
    _assert_full_factorial(frame)
    # Undefined DEG rows are truth calls with zero DEGs. That set is a property
    # of the condition, so a cell line must drop the same count at every k.
    drift = frame.groupby(["metric_key", "cell_line"])["n_conditions"].nunique()
    if (drift != 1).any():
        bad = drift[drift != 1]
        raise ValueError(
            "scored-condition count changes with k or seed; the mean would "
            f"mix different test sets:\n{bad}"
        )
    return frame


def _assert_full_factorial(frame: pd.DataFrame) -> None:
    """Every line × k × seed × metric once. A missing corner is a dropped run."""
    for _short, key, _label in METRICS:
        sub = frame.loc[frame["metric_key"] == key]
        counts = sub.groupby(["cell_line", "overlap", "entity_seed"]).size()
        if (counts != 1).any():
            raise ValueError(f"{key}: a (cell, k, seed) was scored more than once")
        expected = len(CELL_LINES) * len(LEVELS) * len(ENTITY_SEEDS)
        if len(counts) != expected:
            raise ValueError(f"{key}: {len(counts)} points, expected {expected}")
        got_k = set(sub["overlap"].astype(int))
        if got_k != set(LEVELS):
            raise ValueError(f"{key}: overlaps {sorted(got_k)} != {list(LEVELS)}")


def strict_drug_set() -> set[str]:
    """Test drugs never trained on a held-out line at any dose (199 of 268; Figure 3b)."""
    # Imported here, not at the top: src.data pulls in torch (~15 s), which only this registry walk needs.
    import yaml

    from src.config.dataset_registry import load_dataset_registry
    from src.data.splits.generation import load_condition_summary, load_split_policy
    from src.data.splits.schemes import Axes, predicate_pairs, train_pools

    policy, registry = load_split_policy("tahoe"), load_dataset_registry("tahoe")
    axes, held = Axes.from_registry(registry), list(CELL_LINES)
    base_split = REPO / policy.families["tahoe_overlap_bio"]["base_split"]
    base = yaml.safe_load(base_split.read_text(encoding="utf-8"))
    val = predicate_pairs(base["validation"], axes, policy.doses, held)
    test = predicate_pairs(base["test"], axes, policy.doses, held)
    pools = train_pools(load_condition_summary(policy, registry), axes, policy.doses, held,
                        val_pairs=val, test_pairs=test)
    drugs = strict_test_drugs(test, pools)
    print(f"strict by-drug test: {len(drugs)} of {len({d for d, _ in test})} test drugs")
    return drugs
