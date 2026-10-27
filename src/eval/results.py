"""Read scored arms back: per-condition metrics, the keep rule, the split-half reference.

Every figure that compares arms needs the same three things:
  1. per-condition values out of each arm's ``metrics_summary.json`` (``load_percond``);
  2. the keep rule: a condition counts for a metric only if EVERY plotted arm has that
     metric there (``keep_conditions``), so all bars average the same conditions;
  3. the split-half reference for the split (``resolve_ceiling``).
The arms themselves are declared in ``configs/eval/arms.yaml`` (``load_panels``).
"""
from __future__ import annotations

import glob
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from src.paths import relocated_split_path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARMS_YAML = REPO_ROOT / "configs" / "eval" / "arms.yaml"

# The metrics: (title, per-condition group, key, higher_is_better, log_y).
# Report space is `log1p_slice_med_rgs` (each cell scaled to a common panel library, so
# depth differences between plates and methods do not enter);
# DEG is defined on `raw_int_rgs` in every space.
# Every arm must sit on the same DEG floor (`..._detfloor10`): an arm on another floor
# shares no DEG key and the keep rule drops that metric to n=0.
DEG_METRIC = (
    "DEG@100\nadaptive-k (truth)", "deg",
    "deg_raw_int_rgs/deg_overlap_top_p_ranked_k100_fdr0p001_lfc0p1_adaptivek_truth_detfloor10",
    True, False,
)


def metrics_for_space(space: str) -> list[tuple]:
    """DEG plus Pearson r Δ and energy in `space`."""
    return [
        DEG_METRIC,
        ("Pearson r Δ", "pseudobulk_delta", f"{space}/pearson_r_diff", True, False),
        ("Energy dist.\n(log)", "distributional", f"{space}/energy_distance", False, True),
    ]


METRICS = metrics_for_space("log1p_slice_med_rgs")


# Per-condition values

def condkey(entry: Mapping[str, Any]) -> Any:
    """Condition identity. Its "k=v" tuple form is stored in published parquets: keep it."""
    ck = entry.get("condition_keys")
    if isinstance(ck, (list, tuple)):
        return tuple(str(x) for x in ck)
    if isinstance(ck, dict):
        return tuple(f"{k}={ck[k]}" for k in sorted(ck))
    return str(ck)


def _metric_row(blob: Any, metrics: Sequence[tuple]) -> dict[str, float]:
    row = {}
    for _t, grp, key, _hb, _lg in metrics:
        v = blob.get(grp, {}).get(key) if isinstance(blob, dict) else None
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            row[key] = float(v)
    return row


def load_percond(path: str | os.PathLike, metrics: Sequence[tuple] | None = None) -> dict:
    """path -> {condkey: {metric_key: value}}. NaN and absent metrics are left out."""
    metrics = metrics if metrics is not None else METRICS
    d = json.load(open(path))
    return {condkey(e): _metric_row(e.get("metrics", {}), metrics) for e in d["results"]}


def load_repeat_std(path: str | os.PathLike, metrics: Sequence[tuple] | None = None) -> dict:
    """path -> {condkey: {metric_key: std across repeats}}. Missing std stays absent."""
    metrics = metrics if metrics is not None else METRICS
    d = json.load(open(path))
    return {condkey(e): _metric_row(e.get("metrics_std", {}), metrics) for e in d["results"]}


def keep_conditions(percond: Mapping[str, Mapping], names: Iterable[str], common: Sequence,
                    key: str | Sequence[str]) -> list:
    """The keep rule: conditions in `common` where every named arm has `key` (or every key).

    Pass a panel's full metric list to score every metric on ONE condition set: a condition
    where any arm leaves any metric undefined then drops from all of them.
    """
    names, keys = list(names), ([key] if isinstance(key, str) else list(key))
    return [c for c in common if all(k in percond[n][c] for n in names for k in keys)]


def mean_sem(percond: Mapping[str, Mapping], names: Iterable[str], key: str, keep: Sequence) -> dict:
    """Per arm (mean, SEM across conditions) over `keep`; NaN when fewer than two."""
    out = {}
    for name in names:
        vals = np.array([percond[name][c][key] for c in keep if key in percond[name].get(c, {})], float)
        out[name] = ((vals.mean(), vals.std(ddof=1) / math.sqrt(vals.size))
                     if vals.size > 1 else (np.nan, np.nan))
    return out


# The split-half reference

_CEIL_GLOB = str(REPO_ROOT / "results/{ds}/split_half/noise_ceiling/geneset_{gs}__norm_raw_counts"
                 "/nr_20__splitcfg_*__mincells_4__test/noise_ceiling.json")

def resolve_ceiling(ds: str, gs: str, split_yaml: str) -> str:
    """Newest nr_20 rgs ceiling whose RECORDED split_config is `split_yaml`.

    Matched by the split recorded inside each ceiling, not a pinned splitcfg hash:
    the hashes drift when a split file is edited, and a pinned hash then silently
    pairs a panel with a stale ceiling.
    """
    want = os.path.realpath(split_yaml)
    hits = []
    for p in glob.glob(_CEIL_GLOB.format(ds=ds, gs=gs)):
        try:
            d = json.load(open(p))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[warn] unreadable ceiling {p}: {exc}")
            continue
        if os.path.realpath(relocated_split_path(str(d.get("split_config", "")))) != want:
            continue
        if not any(str(k).startswith("log1p_rgs/") for k in d.get("mean_metrics", {})):
            continue                      # pre-rgs ceiling: no comparable keys
        hits.append((os.path.getmtime(p), p))
    if not hits:
        raise SystemExit(f"no nr_20 rgs ceiling for {ds} built from {split_yaml}; "
                         "compute it with scripts/eval/reference.py split_half --n-repeats 20")
    return max(hits)[1]


# The arm registry

def _abs(p: str) -> str:
    return p if os.path.isabs(p) else str(REPO_ROOT / p)


def load_panels(path: str | os.PathLike = ARMS_YAML) -> dict[str, dict]:
    """configs/eval/arms.yaml -> {panel: setup}, paths absolute.

    A setup has ``models`` / ``references`` as (name, color, metrics_summary path)
    triples, ``ceiling_split`` as (dataset, geneset, split path), and ``out``.
    """
    reg = yaml.safe_load(open(path))
    arms, panels = reg["arms"], {}
    for name, raw in reg["panels"].items():
        setup = dict(raw)
        for role, dest in (("bars", "models"), ("references", "references")):
            triples = []
            for entry in setup.pop(role, []):
                entry = {"arm": entry} if isinstance(entry, str) else entry
                if entry["arm"] not in arms:
                    raise KeyError(f"panel {name}: unknown arm {entry['arm']!r}")
                arm = {**arms[entry["arm"]], **{k: v for k, v in entry.items() if k != "arm"}}
                triples.append((arm["name"], arm["color"], _abs(arm["path"])))
            if triples:
                setup[dest] = triples
        cs = setup["ceiling_split"]
        setup["ceiling_split"] = (cs["dataset"], cs["geneset"], _abs(cs["split"]))
        if "ceiling_json" in setup:
            setup["ceiling_json"] = _abs(setup["ceiling_json"])
        if "metrics" in setup:
            setup["metrics"] = [(m["title"], m["group"], m["key"], m["higher_is_better"], m["log_y"])
                                for m in setup["metrics"]]
        if "report_spaces" in setup:
            setup["report_spaces"] = tuple(setup["report_spaces"])
        setup["out"] = _abs(setup["out"])
        panels[name] = setup
    return panels


def resolve_arms(setup: Mapping, key: str, allow_missing: bool = False) -> list[tuple]:
    """Existing (name, color, path) arms of setup[key]. A missing BAR arm is fatal:
    a panel one arm short looks complete and publishable. References may be skipped."""
    arms, missing = [], []
    for name, color, path in setup.get(key, []):
        if os.path.exists(path):
            arms.append((name, color, path))
        else:
            missing.append((name, path))
            print(f"[warn] skipping {name}: missing {path}")
    if missing and key == "models" and not allow_missing:
        raise SystemExit("refusing to plot a panel missing bar model(s): "
                         + "; ".join(f"{n} -> {p}" for n, p in missing)
                         + "\nScore the arm first, or pass --allow-missing-arms.")
    return arms


def arm_provenance(panel: str, models: Sequence[tuple], ceiling_path: str, ceiling_n: Any) -> list[dict]:
    """arms.csv rows for one panel: its split-half reference, then each bar arm's
    metrics_summary.json. Paths are kept whole: this column exists to be opened."""
    cd = json.loads(Path(ceiling_path).read_text())
    rows = [{"panel": panel, "model": "(split-half reference)", "n_conditions": ceiling_n,
             "generated_utc": "", "checkpoint": "",
             "eval_cache_namespace": cd.get("eval_cache_namespace", ""),
             "split_config": cd.get("split_config", ""), "path": os.path.dirname(ceiling_path)}]
    for name, _color, p in models:
        d = json.loads(Path(p).read_text())
        rows.append({"panel": panel, "model": name, "n_conditions": d.get("n_conditions", ""),
                     "generated_utc": d.get("generated_utc", ""),
                     "checkpoint": str(d.get("checkpoint", "") or d.get("run_dir", "")),
                     "eval_cache_namespace": d.get("eval_cache_namespace", ""),
                     "split_config": "", "path": os.path.dirname(p)})
    return rows

