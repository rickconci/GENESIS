"""Figure 1's data: every file under core_deliverables/data/figure1/, rebuilt from the replication ladder's scores and
the settings runs' test evals in core_results/figure1/, then checked against that folder's MANIFEST.tsv.

    python -m scripts.figures.figure1.extract.extract            # rebuild in memory and compare; writes nothing
    python -m scripts.figures.figure1.extract.extract --write    # overwrite what differs, and its MANIFEST rows

One input is frozen rather than rebuilt: Figure 1d's reps.parquet is the ladder's own output (scripts/eval/reference.py
ladder), which draws its cells from the Tahoe-100M evaluation cache on a GPU. It is checked against the ladder's
manifest and cohort, then shipped as written.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from dataclasses import fields
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.figures import CORE
from src.config.config import PertGenConfig
from src.eval import results as mb

REPO = CORE.parent
RAW = REPO / "core_results" / "figure1"
OUT = CORE / "data" / "figure1"


# ----------------------------------------------------------------------------
# Figure 1d: two arms of one condition, split-half or across plates, at 50 to 1,600 cells per arm
# ----------------------------------------------------------------------------

def reps() -> bytes:
    """The ladder's scores as written, once they are its manifest's full grid on its cohort: every condition x
    strategy x cells-per-arm rung x draw exactly once, with every metric the ladder scored."""
    lad = RAW / "ladder"
    man, blob = json.loads((lad / "manifest.json").read_text()), (lad / "reps.parquet").read_bytes()
    cohort, t = pd.read_csv(lad / Path(man["cohort"]).name), pd.read_parquet(io.BytesIO(blob))
    ids = [*cohort.columns, "strategy", "n_cells", "rep"]
    # distinct rows, each inside the grid, as many as the grid has cells: then every cell is there once
    inside = (t[list(cohort.columns)].merge(cohort, how="left", indicator=True)["_merge"].eq("both").all()
              and set(t.strategy) <= set(man["strategies"]) and set(t.n_cells) <= set(man["ladder_cells_per_arm"])
              and set(t.rep) <= set(range(man["n_repeats"])))
    size = len(cohort) * len(man["strategies"]) * len(man["ladder_cells_per_arm"]) * man["n_repeats"]
    if (not inside or cohort.duplicated().any() or man["n_conditions_scored"] != len(cohort) or len(t) != size
            or t.duplicated(ids).any() or len(t.columns.difference([*ids, "sh_plate"])) != man["n_metrics"]):
        raise RuntimeError(f"reps.parquet is not the full ladder of {man['n_conditions_scored']} cohort conditions")
    return blob


# ----------------------------------------------------------------------------
# ED genesis settings: Figure 2's GENESIS retrained with one setting changed, each scored by its own after-fit eval
# ----------------------------------------------------------------------------

KEYS = [m[2] for m in mb.METRICS]          # Figure 2c's three: DEG overlap on counts, Pearson Δ and energy in slice_med
# staged run -> (timestamp, recipe) of the run whose eval it must be: <timestamp>_..._s64_<recipe>_5ep_state2k_...
RUNS = {"genesis": ("260916_114938", "h4096x8192_nrs256_vg05rs256"),
        "m16": ("261005_235052", "h4096x8192_nrs16_vg05rs16"), "m64": ("261006_044718", "h4096x8192_nrs64_vg05rs64"),
        "lambda0": ("260909_220316", "h4096x8192_nrs256_vg00rs256"),
        "lambda0.1": ("260909_220316", "h4096x8192_nrs256_vg01rs256"),
        "lambda0.3": ("260916_114940", "h4096x8192_nrs256_vg03rs256"),
        "lambda1": ("261004_162151", "h4096x8192_nrs256_vg10rs256"),
        "depth0": ("261022_123317", "hnone_nrs256_vg05rs256"), "depth1": ("261021_130248", "h8192_nrs256_vg05rs256")}
# setting -> {value: run}. Figure 2's GENESIS (m = 256, λ = 0.5, two hidden layers) is in all three rows.
SETTINGS = {"n_repeats": {16: "m16", 64: "m64", 256: "genesis"},
            "variogram_weight": {0.0: "lambda0", 0.1: "lambda0.1", 0.3: "lambda0.3", 0.5: "genesis", 1.0: "lambda1"},
            "depth": {0: "depth0", 1: "depth1", 2: "genesis"}}
HIDDEN = {0: [], 1: [8192], 2: [4096, 8192]}
# config keys a setting's runs may differ in: the setting itself, and bookkeeping. The Figure 2 run's resolved
# file stores lr_init (the warmup start) as 5e-7; its yaml and the depth runs' say 0.
BOOK = {"run_name_suffix", "wandb_run_name", "runs_split_slug", "model_eval_compute_hallmark"}
# filled in from the data at run time: some λ runs saved the filled values, the others the unfilled defaults
RUNTIME = {"n_categories", "num_output_genes", "num_tokens", "total_training_steps"}
MAY_DIFFER = {"n_repeats": BOOK | {"n_repeats", "variogram_repeat_subsample", "between_epoch_curriculum_schedule",
                                   "curriculum_target_product", "wandb_project", "workdir", "_wandb"},
              "variogram_weight": BOOK | RUNTIME | {"variogram_loss_weight", "use_eval_callback",
                                                    "log_train_reconstruction_metrics"},   # the last two only log
              "depth": BOOK | {"condition_mlp_hidden_dims", "lr_init"}}
# keys some runs store that PertGenConfig does not define, at their defaults. Both are resource knobs (how often
# training logs RAM; the GPU-resident loader's size cap), not numerics.
RETIRED = {"memory_monitor_every_n_steps": 1000, "gpu_resident_max_gib": 20.0}


@cache
def scores(name: str) -> tuple[tuple[str, str, float], ...]:
    """(condition, metric, value) of one run's eval: three generations per test condition, from that run, and
    reproducing the run's own means. Absent and NaN values are left out (undefined DEG overlap)."""
    path = RAW / "settings" / name / "metrics_summary.json"
    blob, (ts, recipe) = json.loads(path.read_text()), RUNS[name]
    got = (blob["model_eval_n_repeats"], blob["eval_split"])
    if got != (3, "test") or not re.fullmatch(rf"{ts}_.*_s64_{recipe}_5ep_state2k_tahoe_bio_100cond_seed0",
                                              Path(blob["run_dir"]).name):
        raise RuntimeError(f"{name}: {got} from {blob['run_dir']}, expected 3 generations on test from run {ts}")
    rows = tuple(("|".join(c), k, v[k]) for c, v in mb.load_percond(path, mb.METRICS).items() for k in KEYS if k in v)
    for k in KEYS:
        if not np.isclose(np.mean([v for _, m, v in rows if m == k]), blob["mean_metrics"][k], rtol=0, atol=1e-9):
            raise RuntimeError(f"{name} {k}: recomputed mean differs from the run's summary")
    return rows


def config(name: str) -> dict:
    return yaml.safe_load((RAW / "settings" / name / "config_resolved.yaml").read_text())


def anchored(v: object) -> object:
    """Some configs store data paths absolute, others relative: compare from the data anchor on."""
    if isinstance(v, dict):
        return {k: anchored(x) for k, x in v.items()}
    for a in ("eval_cache/", "core_util_data/", "csr_cache"):
        if isinstance(v, str) and a in v:
            return v[v.index(a):]
    return v


def only_differ_in(setting: str) -> None:
    """The setting's runs agree on every resolved-config key but MAY_DIFFER's. A key some run lacks passes when
    every run that has it holds the schema default."""
    default = RETIRED | {f.name: f.default_factory() if callable(f.default_factory) else f.default
                         for f in fields(PertGenConfig)}
    cfgs = [config(n) for n in SETTINGS[setting].values()]
    js = lambda v: json.dumps(anchored(v), sort_keys=True, default=str)  # noqa: E731
    diff = []
    for k in sorted(set().union(*cfgs)):
        vals = [c[k] for c in cfgs if k in c]
        if len(vals) < len(cfgs) and k in default and all(js(v) == js(default[k]) for v in vals):
            continue
        if k == "lr_init" and set(map(float, vals)) <= {0.0, 5e-7}:
            diff += [k] if len(set(map(float, vals))) > 1 else []
        elif len(vals) < len(cfgs) or len({js(v) for v in vals}) > 1:
            diff.append(k)
    if extra := sorted(set(diff) - MAY_DIFFER[setting]):
        raise RuntimeError(f"the {setting} runs differ in more than {setting}: {extra}")
    print(f"  {setting} config keys that differ: {diff}")


def per_condition(setting: str) -> pd.DataFrame:
    """(setting, condition, metric_key, value), runs in the setting's order."""
    return pd.DataFrame([(v, *r) for v, name in SETTINGS[setting].items() for r in scores(name)],
                        columns=[setting, "condition", "metric_key", "value"])


@cache
def n_repeats() -> pd.DataFrame:
    """Generated cells per observed cell, m = 16, 64, 256."""
    only_differ_in("n_repeats")
    return per_condition("n_repeats")


@cache
def variogram() -> pd.DataFrame:
    """Variogram weight λ = 0, 0.1, 0.3, 0.5, 1."""
    only_differ_in("variogram_weight")
    return per_condition("variogram_weight")


def depth() -> pd.DataFrame:
    """0, 1 or 2 hidden layers; per metric, the conditions all three runs define."""
    for d, name in SETTINGS["depth"].items():
        if list(config(name)["condition_mlp_hidden_dims"]) != HIDDEN[d]:
            raise RuntimeError(f"depth {d}: {name} has hidden dims {config(name)['condition_mlp_hidden_dims']}")
    only_differ_in("depth")
    per = per_condition("depth")
    per = per[per.groupby(["metric_key", "condition"]).depth.transform("nunique").eq(len(HIDDEN))]
    if per.groupby(["metric_key", "depth"]).size().groupby("metric_key").nunique().ne(1).any():
        raise RuntimeError("depths disagree on a metric's condition count")
    return per


def m_plotted() -> pd.DataFrame:
    """Mean and s.e.m. per m, each metric over the conditions all three runs define."""
    pc, rows = n_repeats(), []
    for k in KEYS:
        wide = pc[pc.metric_key.eq(k)].pivot(index="condition", columns="n_repeats", values="value").dropna()
        mean, sd = wide.mean(), wide.std()
        rows += [dict(metric_key=k, n_repeats=m, mean=mean[m], sem=sd[m] / np.sqrt(len(wide)), n_conditions=len(wide))
                 for m in wide]
    return pd.DataFrame(rows)


def lambda_plotted() -> pd.DataFrame:
    """Mean and s.e.m. per λ, each over the conditions that run defines (all five define the same ones)."""
    pc, rows = variogram(), []
    for k in KEYS:
        s = pc[pc.metric_key.eq(k)].groupby("variogram_weight").value.agg(["mean", "std", "count"])
        rows += [dict(metric_key=k, variogram_weight=w, mean=r["mean"], sem=r["std"] / np.sqrt(r["count"]),
                      n_conditions=int(r["count"])) for w, r in s.iterrows()]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Then every file, compared with MANIFEST.tsv
# ----------------------------------------------------------------------------

ED = "ED_genesis_settings"
TABLES = {
    "main/reps.parquet": reps,
    f"{ED}/ed_energy_settings/num_repeats_scaling_source_data_plotted.csv": m_plotted,
    f"{ED}/ed_energy_settings/per_condition.parquet": n_repeats,
    f"{ED}/ed_energy_settings/variogram_scaling_source_data_plotted.csv": lambda_plotted,
    f"{ED}/ed_settings/depth_per_condition.parquet": depth,
    f"{ED}/variogram_scaling/per_condition.parquet": variogram,
}


def serialize(rel: str, t: pd.DataFrame | bytes) -> bytes:
    if isinstance(t, bytes):
        return t
    if rel.endswith(".parquet"):
        t.to_parquet(buf := io.BytesIO(), index=False)
        return buf.getvalue()
    return t.to_csv(index=False, float_format="%.6g").encode()     # the plotted tables' writer rounded so


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="overwrite the files that differ, and their MANIFEST rows")
    write = ap.parse_args().write
    manifest = pd.read_csv(OUT / "MANIFEST.tsv", sep="\t", dtype=str, keep_default_na=False).set_index("path")
    if odd := set(manifest.index) ^ set(TABLES):
        raise RuntimeError(f"MANIFEST.tsv and this script list different files: {sorted(odd)}")
    differ = []
    for rel, build in TABLES.items():
        blob = serialize(rel, build())
        sha, want = hashlib.sha256(blob).hexdigest(), manifest.at[rel, "sha256"]
        state = "same bytes" if sha == want else "DIFFERS"
        # a parquet's bytes carry the writer's library versions; its values are what the figure reads
        if state == "DIFFERS" and rel.endswith(".parquet") and hashlib.sha256((OUT / rel).read_bytes()).hexdigest() == want \
                and pd.read_parquet(io.BytesIO(blob)).equals(pd.read_parquet(OUT / rel)):
            state = "same values"
        print(f"{state:12s} {rel}", flush=True)
        if state == "DIFFERS":
            differ.append(rel)
            if write:
                (OUT / rel).parent.mkdir(parents=True, exist_ok=True)
                (OUT / rel).write_bytes(blob)
                manifest.loc[rel, ["bytes", "sha256"]] = [str(len(blob)), sha]
    if write and differ:
        manifest.reset_index().to_csv(OUT / "MANIFEST.tsv", sep="\t", index=False)
        print(f"wrote {len(differ)} files and their MANIFEST.tsv rows")
    elif differ:
        raise SystemExit(f"{len(differ)} of {len(TABLES)} files differ from MANIFEST.tsv (--write to replace them)")
    else:
        print(f"all {len(TABLES)} files match MANIFEST.tsv")


if __name__ == "__main__":
    main()
