"""Figure 3's data: every file under core_deliverables/data/figure3/, rebuilt from the runs' test evals and score
tables in core_results/figure3/ and from Figure 2's staged data, then checked against that folder's MANIFEST.tsv.

    python -m scripts.figures.figure3.extract.extract            # rebuild in memory and compare; writes nothing
    python -m scripts.figures.figure3.extract.extract --write    # overwrite what differs, and its MANIFEST rows

Two inputs are frozen rather than rebuilt: the training-run tables (scaling_runs.csv, the scaling_arms CSVs) count
rows in each run's split cache and run.log, which live with the checkpoints in runs/ (the data release);
scripts/figures/figure3/score/main/score.sh rebuilds them from there.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from collections import defaultdict
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.figures import CORE
from scripts.figures.figure2.draw import data as f2
from scripts.figures.figure3.draw.data import BIO_LEVELS, METRICS, cis
from src.eval import results as mb
from src.figures import metrics
from src.figures.palette import ARMS
from src.stats.equivalence import mark_no_gap, repeat_tests

REPO = CORE.parent
RAW = REPO / "core_results" / "figure3"
F2 = CORE / "data" / "figure2" / "main"
OUT = CORE / "data" / "figure3"
CONTROL = "DMSO_TF"                          # core_util_data/datasets/tahoe.yaml


# ----------------------------------------------------------------------------
# Read per-condition scores off one eval's metrics_summary.json
# ----------------------------------------------------------------------------

def metric_value(cond: dict, key: str) -> float | None:
    """Absent means undefined (truth called no DEGs); the writer drops those, so a stored NaN is a bug."""
    for group in (cond.get("metrics") or {}).values():
        if isinstance(group, dict) and key in group:
            if not np.isfinite(v := float(group[key])):
                raise ValueError(f"{key} is not finite: {group[key]!r}")
            return v
    return None


def split_half_scores(path: Path, k: int, run: str | None = None) -> list[dict]:
    """(condition, metric, value) of every finite Figure 2c metric. The eval must be Figure 2c's protocol
    (split-half on test) at K=k model repeats, and, given ``run``, that run's checkpoint."""
    blob = json.loads(path.read_text())
    got = (blob.get("noise_ceiling_strategy"), blob.get("eval_split"), blob.get("model_eval_n_repeats"))
    if got != ("split_half", "test", k):
        raise RuntimeError(f"{path.name}: {got}, expected split-half on test at K={k}")
    if run and run not in str(blob.get("checkpoint", "")):
        raise RuntimeError(f"{path.name}: checkpoint {blob.get('checkpoint')!r} is not run {run}")
    rows = []
    for e in blob["results"]:
        ck = e["condition_keys"]
        flat = {key: v for g in (e.get("metrics") or {}).values() if isinstance(g, dict) for key, v in g.items()}
        rows += [dict(condition="|".join(f"{c}={ck[c]}" for c in sorted(ck)), metric=m, value=float(flat[key]))
                 for m, (key, _t, _hb) in METRICS.items() if flat.get(key) is not None and np.isfinite(flat[key])]
    return rows


# ----------------------------------------------------------------------------
# Figure 3a: the 3-run arms, and ED 12c, the uniform-draw cells arms, on Figure 2c's conditions
# ----------------------------------------------------------------------------

def split_half_runs() -> pd.DataFrame:
    """Every Figure 3a run but Figure 2's own GENESIS, scored as Figure 2c (K=1)."""
    stems = pd.read_csv(RAW / "scaling_runs.csv").stem
    return pd.DataFrame([dict(stem=s, **r) for s in stems[stems.ne("genesis")]
                         for r in split_half_scores(RAW / "split_half" / f"{s}.json", 1)])


# stem -> the run each eval must come from. 5% and 1% cells were scored only by their K=3 training eval, so the
# 25% cells arm, scored both ways, must show K=3 barely moves a mean before K=3 stands in for K=1.
UNIFORM = {"genesis_25conds": "260918_150610", "genesis_5conds": "260920_164413", "genesis_25cells": "260918_151050",
           "genesis_5cells": "260920_162654", "genesis_1cells": "261016_085931"}
K3 = ("genesis_5cells", "genesis_1cells")
SHIFT_MAX = 0.005


def uniform_cells() -> pd.DataFrame:
    """ED 12c: one run per point, Figure 2's CIs, the split-half reference and the two best other arms."""
    k_of = {"GENESIS": 1, **{s: 3 if s in K3 else 1 for s in UNIFORM}}
    got = pd.DataFrame([dict(stem=s, **r) for s, run in UNIFORM.items()
                        for r in split_half_scores(RAW / "uniform_cells" / f"{s}.json", k_of[s], run)])
    k3 = pd.DataFrame(split_half_scores(RAW / "uniform_cells" / "genesis_25cells_k3.json", 3, UNIFORM["genesis_25cells"]))
    arms = pd.read_csv(RAW / "scaling_arms_uniform_cells.csv").replace({"stem": {"genesis": "GENESIS"}})
    stems = {(r.axis, r.pct_kept): r.stem for r in arms.itertuples()} | {("cells", 100): "GENESIS"}
    by = arms.set_index("stem")
    pc = pd.read_parquet(F2 / "model_comparison" / "per_condition.parquet")
    pc = pc[pc.panel.eq("tahoe_bio")].replace({"model": metrics.ARM_NAME})
    ceil = pd.read_csv(F2 / "model_comparison" / "ceilings.csv")
    ceil = ceil[ceil.panel.eq("tahoe_bio") & ceil.report_space.eq(metrics.SPACE)].set_index("metric_key")
    published = pd.read_csv(F2 / "confidence_intervals.csv")
    rows = []
    for m, (key, _t, hb) in METRICS.items():
        full = pc[pc.metric_key.eq(key)].pivot(index="condition", columns="model", values="value")
        if full.isna().any().any():
            raise RuntimeError(f"{m}: Figure 2c's table has a hole")
        full = full.join(got[got.metric.eq(m)].pivot(index="condition", columns="stem", values="value"))
        block = plotted(full, list(UNIFORM), m)
        same_checkpoint(block, k3[k3.metric.eq(m)], m)
        others = {a: float(block[a].mean()) for a in ARMS if a != "GENESIS"}
        best = sorted(others, key=others.get, reverse=hb)[:2]
        cols = list(dict.fromkeys([*arms.stem, *best]))
        # Figure 2's CIs are on its full set, so they are checked there; the whiskers use the plotted set
        checked = cis(full, [c for c in cols if full[c].notna().all()], key, published)
        ci = checked if len(block) == len(full) else cis(block, cols, key, None)
        for axis, series in (("conds", "fewer conditions"), ("cells", "fewer cells")):
            rows += [dict(metric=m, series=series, pct_kept=p, n_train=by.n_train[s], n_conds=by.n_conds[s], n_runs=1,
                          model_eval_n_repeats=k_of[s], mean=ci[s][0], ci95_lo=ci[s][1], ci95_hi=ci[s][2],
                          n_conditions=len(block))
                     for (a, p), s in sorted(stems.items(), key=lambda kv: kv[0][1]) if a == axis]
        rows += [dict(metric=m, series="Split-half", mean=float(ceil.at[key, "split_half_reference_mean"]),
                      split_half_repeat_sd=float(ceil.at[key, "split_half_reference_repeat_std"]),
                      n_conditions=len(block))]
        rows += [dict(metric=m, series=a, mean=ci[a][0], ci95_lo=ci[a][1], ci95_hi=ci[a][2], n_conditions=len(block))
                 for a in best]
    return pd.DataFrame(rows)


def plotted(full: pd.DataFrame, stems: list[str], m: str) -> pd.DataFrame:
    """Figure 2c's conditions every arm scored: a few may go (5% cells never scored two DEG conditions), as long
    as GENESIS barely moves."""
    block = full[full[stems].notna().all(axis=1)]
    shift = float(block.GENESIS.mean() - full.GENESIS.mean())
    if len(full) - len(block) > 5 or abs(shift) >= 0.001:
        raise RuntimeError(f"{m}: {len(full) - len(block)} conditions unscored, GENESIS moves {shift:+.4f}")
    return block


def same_checkpoint(block: pd.DataFrame, k3: pd.DataFrame, m: str) -> None:
    both = k3.set_index("condition").value.reindex(block.index)
    if both.isna().any() or (shift := abs(both.mean() - block.genesis_25cells.mean())) >= SHIFT_MAX:
        raise RuntimeError(f"{m}: the 25% cells arm's K=3 eval misses conditions or moves its mean (>= {SHIFT_MAX})")
    print(f"  25% cells {m}: K=3 moves the mean by {shift:.4f}")


# ----------------------------------------------------------------------------
# ED 12a-b: Figure 2a's plate pairs, and 25 cells per combination against all of them
# ----------------------------------------------------------------------------

def per_pair() -> pd.DataFrame:
    """The repeat's Pearson Δ and GENESIS's three scores on Figure 2a's pairs, which ED 12b re-tests."""
    t = f2.pairs(F2 / "plate_pairs")
    return t[[*f2.KEYS, f"{f2.REPEAT}__pearson_r_diff", *(f"GENESIS__{m}" for m in METRICS)]]


def percond25() -> pd.DataFrame:
    """Every shard of the 25-cells-per-combination scoring. Overlapping shards rescored some directions with
    the same seeds, so identical rows collapse; a key that still repeats is two different results."""
    d = pd.concat(map(pd.read_parquet, sorted((RAW / "shallow_pc25" / "scores").glob("*__shard?of?.parquet"))),
                  ignore_index=True).drop_duplicates()
    key = [c for c in ("cohort", "cid", "cid_y", "cond_id_x", "cond_id_y", "arm", "redraw", "half", "split") if c in d]
    if d.duplicated(key).any():
        raise RuntimeError(f"duplicate {key} rows: two runs' shards mixed?")
    return d.reset_index(drop=True)


def train614() -> pd.DataFrame:
    """Both models (all cells, 25 per combination) on the 614 plate-6/14 training combinations, one row each."""
    t = RAW / "shallow_pc25" / "train614"
    n = len(pd.read_csv(t / "conds_614.tsv", sep="\t"))
    rep = pd.concat([pd.read_parquet(t / f"pairs__{m}.parquet").assign(model=m) for m in ("full", "pc25")],
                    ignore_index=True)
    if rep.groupby("model").size().ne(n).any() or rep.duplicated(["model", "cell_name", "drug", "dose"]).any():
        raise RuntimeError(f"train614: want {n} rows per model, no duplicates")
    return rep


# ----------------------------------------------------------------------------
# Figure 3b: a held-out cell line that saw 0-164 perturbations in training (5 Latin rows x 3 entity seeds)
# ----------------------------------------------------------------------------

def strict_drugs() -> set[str]:
    """Test drugs no held-out line trains on at any dose (199 of 268). The base split keeps 69 test drugs' 5 µM
    dose in the peek pool, so at high k a sibling dose would train while its test dose is scored."""
    # imported here: src.data pulls in torch, which only this registry walk needs
    from src.config.dataset_registry import load_dataset_registry
    from src.data.splits.generation import load_condition_summary, load_split_policy
    from src.data.splits.schemes import Axes, predicate_pairs, train_pools
    policy, registry = load_split_policy("tahoe"), load_dataset_registry("tahoe")
    axes, held = Axes.from_registry(registry), list(LINES)
    base = yaml.safe_load((REPO / policy.families["tahoe_overlap_bio"]["base_split"]).read_text())
    val, test = (predicate_pairs(base[s], axes, policy.doses, held) for s in ("validation", "test"))
    pools = train_pools(load_condition_summary(policy, registry), axes, policy.doses, held,
                        val_pairs=val, test_pairs=test)
    drugs = {d for d, _ in test} - set().union(*map(set, pools.values()))
    print(f"  strict by-drug test: {len(drugs)} of {len({d for d, _ in test})} test drugs")
    return drugs


LINES = ("C32", "HOP62", "HepG2/C3A", "Hs 766T", "PANC-1")
BIO_MANIFEST = REPO / "core_util_data/splits/tahoe/bio/tahoe_overlap_bio/manifest.json"


def overlap_bio() -> pd.DataFrame:
    """Per run and held-out line, the mean over its strict-drug test conditions, at that line's overlap k."""
    drugs, rows = strict_drugs(), []
    for e in json.loads(BIO_MANIFEST.read_text()):
        row, seed = int(e["row"]), int(e["entity_seed"])
        blob = json.loads((RAW / "overlap_bio" / f"r{row}_es{seed}.json").read_text())
        if f"tahoe_bio_celln10_r{row}_es{seed}" not in blob["run_dir"] or blob["eval_split"] != "test":
            raise RuntimeError(f"r{row}_es{seed}.json is not that run's test eval: {blob['run_dir']}")
        if leaked := drugs & {d for ds in e["peeked_drugs"].values() for d in ds}:
            raise RuntimeError(f"r{row} es{seed} peeked strict drugs: {sorted(leaked)[:5]}")
        vals, undefined = defaultdict(list), defaultdict(int)
        for c in blob["results"]:
            cell, drug = str(c["condition_keys"]["cell_name"]), str(c["condition_keys"]["drug"])
            if drug == CONTROL:                      # not a peeked-drug outcome
                continue
            if cell not in e["peek_per_cell"]:
                raise KeyError(f"test cell {cell!r} is not in the Latin assignment")
            if drug not in drugs:
                continue
            for key, *_ in METRICS.values():
                if (v := metric_value(c, key)) is None:
                    undefined[cell, key] += 1
                else:
                    vals[cell, key].append(v)
        rows += [dict(cell_line=cell, overlap=int(k), metric_key=key, metric_mean=float(np.mean(vals[cell, key])),
                      n_conditions=len(vals[cell, key]), n_undefined=undefined[cell, key], row=row, entity_seed=seed)
                 for cell, k in e["peek_per_cell"].items() for key, *_ in METRICS.values()]
    t = pd.DataFrame(rows).sort_values(["metric_key", "cell_line", "overlap", "entity_seed"]).reset_index(drop=True)
    # the same line x k x seed twice is a duplicated run; a missing one, a dropped run. Undefined DEG is a
    # property of the condition, so a line must score the same conditions at every k and seed.
    if (t.groupby(["metric_key", "cell_line", "overlap", "entity_seed"]).size().ne(1).any()
            or len(t) != len(METRICS) * len(LINES) * len(BIO_LEVELS) * 3 or set(t.overlap) != set(BIO_LEVELS)
            or t.groupby(["metric_key", "cell_line"]).n_conditions.nunique().ne(1).any() or t.metric_mean.isna().any()):
        raise RuntimeError("overlap table is not 3 metrics x 5 lines x 5 k x 3 seeds on fixed conditions")
    return t


# ----------------------------------------------------------------------------
# ED 13: a held-out compound seen in k pool lines (a), and PBMC on fewer cells or conditions (b)
# ----------------------------------------------------------------------------

PERT_METRICS = {"deg": metrics.DEG, "pd": metrics.PEARSON, "ed": metrics.ENERGY}     # ED 13a's short names


def pert_overlap() -> pd.DataFrame:
    """Per held-out compound, the mean over its test conditions in the run where it trained in k pool lines
    (1, 5, 15 or 20): 4 Latin rows x 3 entity seeds, 40 compounds each."""
    manifest = json.loads((RAW / "pert_overlap" / "manifest.json").read_text())
    if len(by_id := {(int(e["row"]), int(e["entity_seed"])): e for e in manifest}) != 12:
        raise RuntimeError(f"manifest has {len(by_id)} distinct splits, expected 12")
    rows = []
    for (row, seed), e in sorted(by_id.items()):
        blob = json.loads((RAW / "pert_overlap" / f"r{row}_es{seed}.json").read_text())
        if not re.search(rf"entity_seed{seed}/[^/]*_r{row}_es{seed}/", blob["run_dir"]) or \
                (blob["n_conditions"], blob["n_skipped"]) != (4023, 0):
            raise RuntimeError(f"r{row}_es{seed}.json is not that run's full test eval")
        peek, vals = {d: int(k) for d, k in e["peek_per_drug"].items()}, defaultdict(list)
        if odd := set(peek) ^ {str(c["condition_keys"]["drug"]) for c in blob["results"]}:
            raise RuntimeError(f"r{row} es{seed}: eval and manifest disagree on {sorted(odd)[:5]}")
        for c in blob["results"]:
            for short, key in PERT_METRICS.items():
                if (v := metric_value(c, key)) is not None:
                    vals[str(c["condition_keys"]["drug"]), short].append(v)
        rows += [dict(drug=d, overlap=k, entity_seed=seed, metric=s, metric_mean=float(np.mean(vals[d, s])),
                      n_conditions=len(vals[d, s])) for d, k in peek.items() for s in PERT_METRICS]
    t = pd.DataFrame(rows)
    if len(t) != 40 * 4 * 3 * len(PERT_METRICS) or t.metric_mean.isna().any():
        raise RuntimeError(f"{len(t)} rows or a compound with no score: expected 40 compounds x 4 k x 3 seeds")
    return t


# % kept per series, both ending on the full model. No 5% conditions arm: it kept no perturbed condition of a
# held-out donor.
SERIES = {"cells": (5, 25, 100), "conds": (25, 100)}


@cache
def pbmc() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(plotted means, tests). PBMC repeats no condition, so each reduced arm is tested against the full model,
    with Figure 2a's two no-gap tests and the condition as the cluster, on the conditions all four arms score."""
    arms = pd.read_csv(RAW / "pbmc_cells_conds" / "scaling_arms.csv")
    leg = {(r.axis, r.pct_kept): r.leg for r in arms.itertuples()}
    leg |= {(a, 100): leg["full", 100] for a in SERIES}
    percond = {r.leg: mb.load_percond(RAW / "pbmc_cells_conds" / f"{r.axis}{r.pct_kept}.json") for r in arms.itertuples()}
    n_train, rows, tests = arms.set_index("leg").n_train, [], []
    for key, _t, hb in METRICS.values():
        wide = pd.DataFrame({g: {"|".join(c): v[key] for c, v in p.items() if key in v}
                             for g, p in percond.items()}).dropna().sort_index()
        ref, cluster = wide[leg["full", 100]].to_numpy(), wide.index.to_numpy()
        # one score per condition, no plate directions: the averaged win rate is the plain one
        t = pd.DataFrame([dict(metric=key, arm=leg[a, p], **r) for a in SERIES for p in SERIES[a][:-1]
                          for r in repeat_tests(ref, *(3 * [wide[leg[a, p]].to_numpy()]), cluster, hb)])
        tests.append(t.assign(mark=mark_no_gap(t)))
        rows += [dict(metric_key=key, series={"cells": "fewer cells", "conds": "fewer conditions"}[a], pct_kept=p,
                      n_train=n_train[leg[a, p]], mean=float(y.mean()), sem=float(y.std(ddof=1) / np.sqrt(len(y))),
                      n_conditions=len(wide))
                 for a in SERIES for p in SERIES[a] for y in [wide[leg[a, p]].to_numpy()]]
    return pd.DataFrame(rows), pd.concat(tests, ignore_index=True)


# ----------------------------------------------------------------------------
# Then every file, compared with MANIFEST.tsv
# ----------------------------------------------------------------------------

ED12, ED13 = "ED_12cells_per_condition", "ED_scaling_pert_split"
TABLES = {
    "main/figure2/confidence_intervals.csv": lambda: (F2 / "confidence_intervals.csv").read_bytes(),
    "main/figure2/per_condition.parquet": lambda: (F2 / "model_comparison" / "per_condition.parquet").read_bytes(),
    "main/scaling_runs.csv": lambda: (RAW / "scaling_runs.csv").read_bytes(),
    "main/split_half/scaling_runs.parquet": split_half_runs,
    "main/overlap_bio/cell_seed_overlap_strict_drug.csv": overlap_bio,
    f"{ED12}/ed_cell_sampling/figure3a_uniform_cells_split_source_data_plotted.csv": uniform_cells,
    f"{ED12}/ed_cells_per_combination/figure3a_source_data_per_pair.csv": per_pair,
    f"{ED12}/ed_cells_per_combination/percond25_pc25.parquet": percond25,
    f"{ED12}/ed_cells_per_combination/train614_repeat.parquet": train614,
    f"{ED13}/drug_seed_overlap.csv": pert_overlap,
    f"{ED13}/parse_bio_cells_conds_source_data_plotted.csv": lambda: pbmc()[0],
    f"{ED13}/parse_bio_cells_conds_source_data_statistics.csv": lambda: pbmc()[1],
}
SIX_DIGITS = {f"{ED12}/ed_cell_sampling/figure3a_uniform_cells_split_source_data_plotted.csv",
              f"{ED12}/ed_cells_per_combination/figure3a_source_data_per_pair.csv",
              f"{ED13}/parse_bio_cells_conds_source_data_plotted.csv",
              f"{ED13}/parse_bio_cells_conds_source_data_statistics.csv"}


def serialize(rel: str, t: pd.DataFrame | bytes) -> bytes:
    if isinstance(t, bytes):
        return t
    if rel.endswith(".parquet"):
        t.to_parquet(buf := io.BytesIO(), index=False)
        return buf.getvalue()
    return t.to_csv(index=False, float_format="%.6g" if rel in SIX_DIGITS else None).encode()


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
