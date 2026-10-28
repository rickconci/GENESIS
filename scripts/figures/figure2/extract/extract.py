"""Figure 2's data: every file under core_deliverables/data/figure2/, rebuilt from the scores in core_results/figure2/
and the splits in core_util_data/, then checked against that folder's MANIFEST.tsv.

    python -m scripts.figures.figure2.extract.extract            # rebuild in memory and compare; writes nothing
    python -m scripts.figures.figure2.extract.extract --write    # overwrite what differs, and its MANIFEST rows
    python -m scripts.figures.figure2.extract.extract --out DIR  # a fresh copy of the folder in DIR, rebuilt; the folder untouched

Scorer outputs pass through as scored: the plate-pair tables and two PBMC well-draw files. The rest is computed here:
per-condition scores and split-half references by each panel's keep rule (configs/eval/arms.yaml names its arms), the
plate-pair drop lists, the 95% cluster-bootstrap CIs and the Extended Data tables. Figure 3's extract reads four of
these files (main/), so this one runs first.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.figures import CORE
from scripts.figures.figure2.draw import data as f2
from src.eval import results as mb             # extract only: importing src.eval restyles matplotlib
from src.figures import metrics
from src.figures.palette import ARMS
from src.stats.bootstrap import DATASET_UNITS, ci_rows, condition_keys
from src.stats.equivalence import clustered_ratio, point_test

REPO = CORE.parent
RAW = REPO / "core_results" / "figure2"
OUT = CORE / "data" / "figure2"
REG = yaml.safe_load(mb.ARMS_YAML.read_text())
SPACE = metrics.SPACE
METRICS = mb.metrics_for_space(SPACE)
SPACES = ("log1p_rgs", SPACE)                   # Figure 2c's ceilings.csv keeps both report spaces
ALL = tuple({m[2]: m for sp in SPACES for m in mb.metrics_for_space(sp)}.values())   # DEG is in both: once


# ----------------------------------------------------------------------------
# Per-condition scores: a registry panel's arms on its keep rule, against its split-half reference
# ----------------------------------------------------------------------------

# An ablation panel is scored on its main split's test set, so it shares that panel's reference (and no response)
REFERENCE = {"tahoe_bio_input_ablation": "tahoe_bio"}
# Every bar's n (DEG; Pearson Δ and energy): Pearson Δ and energy on every condition the reference
# scores, DEG where truth calls DEGs in the reference and in every arm's draw. An arm rescored on other conditions
# moves every bar of its panel, so a panel whose sets moved is refused rather than rebuilt.
N_RULE = {"tahoe_bio": (4064, 4211), "parse_bio": (2395, 4268), "tahoe_pert_fixed": (3396, 4023),
          "parse_pert_fixed": (1828, 3456)}
NO_RESPONSE = RAW / "no_response"


def arms(panel: str, role: str = "bars") -> list[tuple[str, str, Path]]:
    """(name, style key, staged metrics_summary.json) per arm, in registry order. Arm a.b is split_half/a/b.json."""
    out = []
    for e in REG["panels"][panel].get(role, []):
        e = {"arm": e} if isinstance(e, str) else e
        arm = {**REG["arms"][e["arm"]], **e}
        out.append((arm["name"], arm["color"], RAW / "split_half" / f"{e['arm'].replace('.', '/', 1)}.json"))
    return out


@cache
def scores(path: Path, keys: tuple = ALL) -> dict:
    """{condition: {metric: value}}; a metric undefined there is absent, never NaN (src/eval/results.py)."""
    return mb.load_percond(path, keys)


@cache
def reference(panel: str, keys: tuple = ALL) -> tuple[dict, dict]:
    """(per-condition mean, per-condition repeat s.d.) of the panel's split-half reference. The staged file pins it:
    resolve_ceiling's newest-file glob would follow any rescore. Refused unless it is the 20-redraw test reference
    of the split the panel names."""
    path = RAW / "split_half" / REFERENCE.get(panel, panel) / "reference.json"
    head, split = json.loads(path.read_text()), REG["panels"][panel]["ceiling_split"]["split"]
    if (Path(head["split_config"]).name, head["n_repeats"], head["eval_split"]) != (Path(split).name, 20, "test"):
        raise RuntimeError(f"{panel}: {path} is not the nr_20 test reference of {split}")
    return mb.load_percond(path, keys), mb.load_repeat_std(path, keys)


@cache
def cells(path: Path) -> tuple[dict, object]:
    """({condition: n_truth}, seed) of one scored file: which cells its truth half was drawn from, and how."""
    blob = json.loads(path.read_text())
    return {mb.condkey(e): e["n_truth"] for e in blob["results"]}, blob.get("seed")


@cache
def no_response(panel: str) -> dict:
    """{condition: {metric: value}}: the condition's own controls scored as the prediction (scoring.NoResponse), on
    the conditions some arm failed. Its Δ is exactly 0, so Pearson Δ is undefined there; the rule sets it to 0. Refused
    if it is not that: a defined Pearson Δ, or a DEG overlap other than 0."""
    path = NO_RESPONSE / f"{REFERENCE.get(panel, panel)}.json"
    pc, pearson = scores(path), [m[2] for m in ALL if m[2].endswith("/pearson_r_diff")]
    for c, v in pc.items():
        if any(k in v for k in pearson) or v.get(mb.DEG_METRIC[2], 0.0) != 0.0:
            raise RuntimeError(f"{path.name} {c}: not no response (Pearson Δ defined, or DEG overlap not 0)")
    return {c: {**v, **dict.fromkeys(pearson, 0.0)} for c, v in pc.items()}


@cache
def filled(panel: str) -> dict[str, dict]:
    """Every arm of the panel (bars, then references) on every condition its split-half reference scores:
    the arm's own score, or, where it failed, no response. A condition the truth cannot score (DEG overlap
    where truth calls no DEGs) still drops for every arm, by the keep rule. No response must stand on the cells the
    reference drew from and with the failed arm's seed, so its truth half is the one that arm would have scored."""
    ceil, _sd = reference(panel)
    ref_n, _seed = cells(RAW / "split_half" / REFERENCE.get(panel, panel) / "reference.json")
    out = {}
    for name, _k, path in arms(panel) + arms(panel, "references"):
        own = scores(path)
        if odd := set(own) - set(ceil):
            raise RuntimeError(f"{panel} {name}: {len(odd)} conditions its reference does not score")
        for c in own:                            # only a model's failure is filled: one it scored stays as scored
            if (key := next((m[2] for m in METRICS[1:] if m[2] in ceil[c] and m[2] not in own[c]), None)):
                raise RuntimeError(f"{panel} {name} {c}: scored, but {key} undefined where truth defines it")
        if gap := sorted(set(ceil) - set(own)):
            nr, nr_path = no_response(panel), NO_RESPONSE / f"{REFERENCE.get(panel, panel)}.json"
            nr_n, nr_seed = cells(nr_path)
            if (missing := [c for c in gap if c not in nr]) or any(nr_n[c] != ref_n[c] for c in gap):
                raise RuntimeError(f"{panel} {name}: no response lacks {len(missing)} of its {len(gap)} failures, "
                                   "or draws from other cells than the reference")
            if (seed := cells(path)[1]) != nr_seed:
                raise RuntimeError(f"{panel} {name}: scored with seed {seed}, no response with {nr_seed}")
            print(f"  {panel} {name}: {len(gap)} failed conditions scored as no response")
        out[name] = {**own, **{c: no_response(panel)[c] for c in gap}}
    return out


def keep(pc: dict, names: list[str], common: list, key: str, ceil: dict) -> list:
    """The plotter's rule: every condition where every named arm and the reference define ``key``.
    Every arm is ``filled``, so only the truth can leave ``key`` undefined."""
    return [c for c in mb.keep_conditions(pc, names, common, key) if key in ceil.get(c, {})]


def mean_of(values: list[float], what: str) -> float:
    if not values:                                 # the plotter drew no line here; a table row of NaN would hide it
        raise RuntimeError(f"{what}: no conditions to average")
    return float(np.mean(values))


@cache
def per_condition(panel: str) -> pd.DataFrame:
    """One row per (arm, condition, metric) a Figure 2c-style panel averages (figure2_extract.py's extract)."""
    pc = {n: filled(panel)[n] for n, _k, _p in arms(panel)}
    ceil, _sd = reference(panel)
    common = sorted(ceil)
    rows, n = [], []
    for _t, _g, key, _hb, _lg in METRICS:
        kept = keep(pc, list(pc), common, key, ceil)
        n.append(len(kept))
        rows += [(panel, name, "|".join(c), key, pc[name][c][key]) for name in pc for c in kept]
    if n[1:] != [len(ceil)] * 2 or tuple(n) != (N_RULE[panel][0], *2 * N_RULE[panel][1:]):
        raise RuntimeError(f"{panel}: n per metric {n}, the rule gives {N_RULE[panel]} ({len(ceil)} truth conditions)")
    return pd.DataFrame(rows, columns=["panel", "model", "condition", "metric_key", "value"])


def ceilings(panels: list[str], spaces: tuple[str, ...]) -> pd.DataFrame:
    """The reference line per panel, space and metric: its mean and mean repeat s.d. on the bars' own conditions,
    to 6 decimals (metric_bars.py --space each --export-tables). A space leaves out an arm never scored in it."""
    rows = []
    for panel in panels:
        pc = {n: filled(panel)[n] for n, _k, _p in arms(panel)}
        ceil, sd = reference(panel)
        common = sorted(ceil)
        for sp in spaces:
            bars = [n for n in pc if any(f"{sp}/pearson_r_diff" in pc[n].get(c, {}) for c in common)]
            for title, _g, key, _hb, _lg in mb.metrics_for_space(sp):
                kept, what = keep(pc, bars, common, key, ceil), f"{panel} {sp} {key}"
                rows.append(dict(panel=panel, report_space=sp, metric=title.replace("\n", " "), metric_key=key,
                                 split_half_reference_mean=round(mean_of([ceil[c][key] for c in kept], what), 6),
                                 split_half_reference_repeat_std=round(mean_of(
                                     [sd[c][key] for c in kept if key in sd.get(c, {})], what), 6)))
    return pd.DataFrame(rows)


@cache
def conditioning(panel: str) -> dict[str, pd.DataFrame]:
    """ED conditioning's three inputs: per-condition scores, the split-half reference, and each arm's fill and hatch.

    The conditions are Figure 2c's for this panel's reference split, not the ones this panel could score on its
    own. Figure 2c keeps a condition only when every arm there defines the metric, so DEG drops conditions where
    a competitor calls no overlap and these bars would otherwise still have one."""
    from src.eval.plots.style import COL, HATCH
    models, refs = arms(panel), arms(panel, "references")
    roles = [("bar", n, k) for n, k, _p in models] + [("reference_line", n, k) for n, k, _p in refs]
    pc = filled(panel)
    joint = [n for n, _k, _p in models + (refs if REG["panels"][panel].get("intersect_references") else [])]
    ceil, sd = reference(panel)
    common = sorted(ceil)
    fig2c = per_condition(REFERENCE.get(panel, panel))
    rows, ref = [], []
    for _t, _g, key, hb, logy in METRICS:
        kept = keep(pc, joint, common, key, ceil)
        on_2c = set(fig2c.loc[fig2c.metric_key.eq(key), "condition"])
        joined = ["|".join(c) for c in kept]
        if missing := on_2c - set(joined):
            raise RuntimeError(f"{panel} {key}: Figure 2c scores {len(missing)} conditions this panel does not")
        kept = [c for c, j in zip(kept, joined) if j in on_2c]
        rows += [("default", role, n, "|".join(c), key, pc[n][c][key]) for role, n, _k in roles
                 for c in kept if key in pc[n].get(c, {})]
        ref.append(("default", key, hb, logy, len(kept), mean_of([ceil[c][key] for c in kept], f"{panel} {key}"),
                    mean_of([sd[c][key] for c in kept if key in sd.get(c, {})], f"{panel} {key} s.d.")))
    return {"per_condition.parquet": pd.DataFrame(rows, columns=["space", "role", "arm", "condition", "metric_key",
                                                                 "value"]),
            "reference.csv": pd.DataFrame(ref, columns=["space", "metric_key", "higher_is_better", "log_axis",
                                                        "n_conditions", "split_half_reference_mean",
                                                        "split_half_reference_repeat_sd"]),
            "arms.csv": pd.DataFrame([(n, k, role, COL[k], HATCH.get(k, "")) for role, n, k in roles],
                                     columns=["arm", "style_key", "role", "fill", "hatch"])}


# ----------------------------------------------------------------------------
# Read the plate pairs: Figure 2a-b's (Tahoe bio) and ED pert generalization's, as the seeded scorer wrote them
# ----------------------------------------------------------------------------

PAIRS = RAW / "plate_pairs"
BIO_SPLIT = REPO / "core_util_data/splits/tahoe/bio/tahoe_legacy_bio_holdout.yaml"   # the bio pairs' split


def pair_table(split: str, stem: str) -> bytes:
    """One arm's table, byte for byte, once it is in Figure 2's report space and its split's pair rules hold."""
    path = PAIRS / split / f"pair_metrics__{stem}.parquet"
    if set(pd.read_parquet(path, columns=["report_space"]).report_space) != {SPACE}:
        raise ValueError(f"{path}: not scored in {SPACE}")
    (figure2a_pairs if split == "tahoe_bio" else pert_pairs)()       # raises unless the split's pair rules hold
    return path.read_bytes()


def genesis_pairs() -> pd.DataFrame:
    return pd.read_parquet(PAIRS / "tahoe_bio" / "pair_metrics__genesis.parquet")


@cache
def seen_in_training() -> pd.DataFrame:
    """(cell line, compound) of repeat pairs whose combination trained at another dose: not held out, so a and b
    drop them. TRAIN is the baselines' (all - test - validation), from the split and the summary."""
    from src.eval.reference.replicate import drop_seen_in_training, replicate_design, trained_combinations  # torch
    trained = trained_combinations("tahoe", BIO_SPLIT)
    _kept, seen = drop_seen_in_training(genesis_pairs(), replicate_design("tahoe"), trained)
    return seen[["cell_name", "drug"]].drop_duplicates().sort_values(["cell_name", "drug"])


@cache
def too_few_cells() -> pd.DataFrame:
    """Repeat pairs with < 50 truth cells on a plate: a stray population, not a repeat."""
    from src.eval.reference.replicate import drop_small_plates
    _kept, small = drop_small_plates(genesis_pairs())
    return small[f2.KEYS].sort_values(["cell_name", "drug", "dose"])


@cache
def figure2a_pairs() -> pd.DataFrame:
    """Figure 2a-b's pairs by data.py:pairs' rule, its two drop lists built here instead of read: shared by every
    arm, neither list, finite on every mean. 530 pairs from 500 combinations, or the bars moved."""
    frames = f2.read_arms(PAIRS / "tahoe_bio")
    seen = set(seen_in_training().itertuples(index=False, name=None))
    small = {f2.pair_key(*r) for r in too_few_cells().itertuples(index=False)}
    kept = [k for k in sorted(set.intersection(*(set(f.index) for f in frames.values())))
            if k[:2] not in seen and k not in small]
    t = pd.concat([f2.wide(frames, kept, m) for m, _t, _hb in f2.PAIR_METRICS], axis=1)
    finite = np.isfinite(t[[f"{a}__{m}" for m, *_ in f2.PAIR_METRICS for a in [f2.REPEAT, *frames]]]
                         .to_numpy(float)).all(axis=1)
    if not np.isfinite(t[finite].to_numpy(float)).all():
        raise ValueError("a kept pair has a finite cross_mean but a non-finite direction")
    t = t[finite].reset_index()
    if (len(t), len(t[["cell_name", "drug", "dose"]].drop_duplicates())) != (530, 500):
        raise RuntimeError(f"{len(t)} pairs, expected 530 pairs from 500 combinations")
    return t


@cache
def benchmark_pairs() -> pd.DataFrame:
    """ED pert's repeat benchmark: of the 423 test conditions measured on exactly two plates, those with >= 50 truth
    cells on both (the bio rule). The census is one pert-overlap run's after-fit test eval: every run's test block."""
    blob = json.loads((PAIRS / "tahoe_pert_fixed" / "census_r0_es0.json").read_text())
    if (blob.get("n_conditions"), blob.get("n_skipped")) != (4023, 0):
        raise RuntimeError("the census is not the full pert test eval (4023 conditions, none skipped)")
    t = pd.DataFrame([dict(cell_name=str(k["cell_name"]), drug=str(k["drug"]), dose=float(k["dose"]),
                           plate=str(k["plate"]).removeprefix("plate"), n_truth=int(r["n_truth"]))
                      for r in blob["results"] for k in [r["condition_keys"]]])
    if t.duplicated([*f2.KEYS[:3], "plate"]).any() or not t.plate.str.isdigit().all() or t.n_truth.min() < 4:
        raise RuntimeError("census: a duplicated (condition, plate), a plate not named plateN, or < 4 truth cells")
    pairs = []
    for (c, d, x), g in t.groupby(["cell_name", "drug", "dose"], sort=True):
        if len(g) > 2:
            raise RuntimeError(f"{c} x {d} x {x} sits on {len(g)} plates: one pair per condition is the design")
        if len(g) == 2:
            a, b = g.sort_values("plate", key=lambda s: s.astype(int)).itertuples()
            pairs.append((c, d, x, a.plate, b.plate, min(a.n_truth, b.n_truth)))
    gap = pd.DataFrame(pairs, columns=[*f2.KEYS, "n_min"])
    if len(gap) != 423:
        raise RuntimeError(f"{len(gap)} two-plate conditions, expected 423")
    return gap[gap.n_min >= f2.MIN_CELLS][f2.KEYS].reset_index(drop=True)


@cache
def pert_pairs() -> pd.DataFrame:
    """ED pert's pairs, by data.py:pert_pairs (every benchmark pair scored, extras only under 50 cells, GENESIS's
    repeat in every table to 1e-9), each metric blanked where the repeat or an arm leaves it undefined."""
    return f2.pert_pairs(PAIRS / "tahoe_pert_fixed", io.StringIO(benchmark_pairs().to_csv(index=False)))


# ----------------------------------------------------------------------------
# Then the 95% CIs every bar carries: paired cluster bootstrap of the plotted mean (src/stats/bootstrap.py)
# ----------------------------------------------------------------------------

def pair_cis(pairs: pd.DataFrame) -> list[dict]:
    """a: the repeat and every arm on each metric's pairs; Tahoe clusters line x drug (drawn) and x dose."""
    cols, rows = [f2.REPEAT, *ARMS], []
    for unit in DATASET_UNITS["tahoe"]:
        for m, _t, _hb in f2.PAIR_METRICS:
            sub = pairs[pairs[f"{f2.REPEAT}__{m}"].notna()]
            rows += ci_rows(sub[[f"{a}__{m}" for a in cols]].set_axis(cols, axis=1), sub, unit, panel="a", metric=m,
                            dataset="tahoe")
    return rows


def condition_cis(pc: pd.DataFrame, panels: list[tuple[str, str]]) -> list[dict]:
    """c: one row per plotted condition, one column per arm (the keep rule gives them one set, so a NaN means the
    plotted set is not the bootstrapped one). PBMC clusters donor x cytokine: a sample's cell types and wells."""
    pc, rows = pc.replace({"model": metrics.ARM_NAME}), []
    for panel, ds in panels:
        for key, _t, _hb in f2.COND_METRICS:
            s = pc[pc.panel.eq(panel) & pc.metric_key.eq(key)].pivot(index="condition", columns="model", values="value")
            if s.empty or s.isna().any().any():
                raise ValueError(f"{panel} {key}: arms cover different conditions")
            for unit in DATASET_UNITS[ds]:
                rows += ci_rows(s, condition_keys(s.index.to_series()), unit, panel=f"c_{panel}", metric=key,
                                dataset=ds)
    return rows


def plotted_means(ci: pd.DataFrame, pairs: pd.DataFrame, pc: pd.DataFrame) -> pd.DataFrame:
    """The draws refuse a CI off its bar by 1e-12: the bootstrap's mean must be the mean each bar plots."""
    bars = pc.replace({"model": metrics.ARM_NAME}).groupby(["panel", "model", "metric_key"]).value.mean()
    for r in ci.itertuples():
        bar = (pairs.loc[pairs[f"{f2.REPEAT}__{r.metric}"].notna(), f"{r.arm}__{r.metric}"].mean() if r.panel == "a"
               else bars[r.panel.removeprefix("c_"), r.arm, r.metric])
        if abs(r.mean - bar) > 1e-12:
            raise RuntimeError(f"{r.panel} {r.metric} {r.arm}: CI mean {r.mean} is not the plotted {bar}")
    return ci


def figure2c() -> pd.DataFrame:
    return pd.concat([per_condition(p) for p in ("tahoe_bio", "parse_bio")], ignore_index=True)


def figure2_cis() -> pd.DataFrame:
    """Figure 2: a's 530 pairs and the repeat, c's per-metric condition sets (src/stats/bootstrap.py)."""
    rows = pair_cis(figure2a_pairs()) + condition_cis(figure2c(), [("tahoe_bio", "tahoe"), ("parse_bio", "parse")])
    return plotted_means(pd.DataFrame(rows), figure2a_pairs(), figure2c())


PERT = ("tahoe_pert_fixed", "parse_pert_fixed")


def pert_cis() -> pd.DataFrame:
    """ED pert generalization: a's benchmark pairs (each metric its own set), c's conditions (ed_pert_stats.py)."""
    pc = pd.concat([per_condition(p) for p in PERT], ignore_index=True)
    rows = pair_cis(pert_pairs()) + condition_cis(pc, [("tahoe_pert_fixed", "tahoe"), ("parse_pert_fixed", "parse")])
    return plotted_means(pd.DataFrame(rows), pert_pairs(), pc)


# ----------------------------------------------------------------------------
# ED objective: GENESIS 25% cells against the same model retrained on a set-level MMD loss
# ----------------------------------------------------------------------------

MMD = RAW / "loss_mmd"
G, S = "GENESIS 25% cells", "Set-level MMD 25% cells"
MMD_STEM = {G: "genesis_25cells", S: "rbfset_25cells"}
MMD_METRICS = [("deg_adaptivek_truth", True), ("pearson_r_diff", True), ("energy_distance", False), ("mmd_rbf", False)]
MMD_SPLIT = (*METRICS, ("MMD (RBF)", "distributional", f"{SPACE}/mmd_rbf", False, True))


def mmd_pairs() -> pd.DataFrame:
    """a: Figure 2a's 530 pairs (figure2a_pairs); the repeat and
    each arm's cross scores (mean, both directions)."""
    idx = pd.MultiIndex.from_tuples([f2.pair_key(*k) for k in figure2a_pairs()[f2.KEYS].itertuples(index=False)])
    tab = {}
    for arm, stem in MMD_STEM.items():
        df = pd.read_parquet(MMD / "plate_pairs" / f"{stem}.parquet")
        if set(df.report_space) != {SPACE}:
            raise ValueError(f"{stem}: not scored in {SPACE}")
        tab[arm] = df.set_index(pd.MultiIndex.from_tuples(
            [f2.pair_key(*k) for k in df[f2.KEYS].itertuples(index=False)]))
    if len(idx) != 530 or any(len(idx.difference(df.index)) for df in tab.values()):
        raise RuntimeError(f"{len(idx)} Figure 2a pairs (want 530), or an MMD table misses some")
    g, out = tab[G].loc[idx], pd.DataFrame(index=idx)
    for m in ("deg_adaptivek_truth", "pearson_r_diff"):   # truth vs truth: each table must carry GENESIS's repeat
        if not np.allclose(tab[S].loc[idx, f"true_true__{m}"], g[f"true_true__{m}"], equal_nan=True):
            raise RuntimeError(f"the MMD table's experimental repeat ({m}) differs from GENESIS's")
    for m, _hb in MMD_METRICS:
        out[f"{f2.REPEAT}__{m}"] = g[f"true_true__{m}"]
        for arm, df in tab.items():
            out[f"{arm}__{m}"] = df.loc[idx, f"cross_mean__{m}"]
    if not np.isfinite(out.to_numpy(float)).all():
        raise RuntimeError("non-finite score on a Figure 2a pair")
    return out.rename_axis(f2.KEYS).reset_index()


@cache
def f2c_keep() -> dict[str, set[str]]:
    """Figure 2c's Tahoe condition ids, per metric. A condition one of its arms cannot score is out of
    every bar and out of the reference: nearest-neighbor leaves top-100 DEG undefined on 34 conditions,
    and those near-zero overlaps must not pull this panel's split-half line down. MMD is not a Figure 2c
    metric, so it uses the energy set."""
    pc = per_condition("tahoe_bio")
    deg, energy = mb.DEG_METRIC[2], f"{SPACE}/energy_distance"
    key = {"deg_adaptivek_truth": deg, "pearson_r_diff": f"{SPACE}/pearson_r_diff",
           "energy_distance": energy, "mmd_rbf": energy}
    return {m: set(pc.loc[pc.metric_key.eq(k), "condition"]) for m, k in key.items()}


def mmd_conditions() -> pd.DataFrame:
    """b: every condition the reference and both loss arms score, at Figure 2c's K=1 draw. ``condition`` is
    Figure 2c's id; ``mmd_tables`` cuts each metric to that panel's keep set. ``sd__`` is the reference's
    per-condition repeat s.d."""
    ceil, sd = reference("tahoe_bio", MMD_SPLIT)
    src = {"Split-half reference": ceil}
    for arm, stem in MMD_STEM.items():
        path = MMD / "split_half" / f"{stem}.json"
        if json.loads(path.read_text()).get("model_eval_n_repeats", 1) != 1:
            raise RuntimeError(f"{arm}: not the K=1 draw Figure 2c's arms use")
        src[arm] = scores(path, MMD_SPLIT)
    col = {"deg_adaptivek_truth": mb.DEG_METRIC[2], **{m: f"{SPACE}/{m}" for m, _hb in MMD_METRICS[1:]}}
    rows = []
    for c in sorted(set.intersection(*map(set, src.values()))):
        row = dict(item.split("=", 1) for item in c)
        row["condition"] = "|".join(c)
        for a, v in src.items():
            for m, k in col.items():
                row[f"{a}__{m}"] = v[c].get(k, np.nan)
        for m, k in col.items():
            row[f"sd__{m}"] = sd.get(c, {}).get(k, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def share_better(g: np.ndarray, m: np.ndarray, cluster: np.ndarray, higher_is_better: bool) -> dict:
    """Share of units where GENESIS beats the MMD arm (ties 1/2), its 95% cluster-bootstrap CI (seed 0) and the
    condition-clustered P against 1/2: a condition's plates or pairs are one sample."""
    better = np.where(g == m, 0.5, (g > m) if higher_is_better else (g < m))
    est = clustered_ratio(better.astype(float), np.ones_like(g), cluster, n_boot=10_000, seed=0, alpha=0.025)
    return dict(share=est.value, ci_lo=est.boot_lo, ci_hi=est.boot_hi, p=point_test(est, 0.5, log_scale=False)["p"],
                n_units=len(g), n_clusters=est.dof + 1)


@cache
def mmd_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(plotted means, GENESIS vs MMD unit by unit). Each metric on the units where the reference and both arms
    define it; panel b then keeps only Figure 2c's conditions for that metric."""
    plotted, h2h = [], []
    keep = f2c_keep()
    for lab, df, ref, ref_col in (("a", mmd_pairs(), "Experimental replicate reference", f2.REPEAT),
                                  ("b", mmd_conditions(), "Split-half reference", "Split-half reference")):
        for m, hb in MMD_METRICS:
            cols = [f"{ref_col}__{m}", f"{G}__{m}", f"{S}__{m}"]
            d = df[df[cols].notna().all(axis=1)]
            # Figure 2c's band: the mean of the reference's per-condition repeat s.d. on these same conditions
            repeat_std = np.nan
            if lab == "b":
                d = d[d.condition.isin(keep[m])]
                if len(d) != len(keep[m]):
                    raise RuntimeError(f"{m}: {len(d)} conditions on the loss arms, Figure 2c keeps {len(keep[m])}")
                s = d[f"sd__{m}"]
                if not np.isfinite(s.to_numpy(float)).all():
                    raise RuntimeError(f"{m}: a plotted condition has no split-half repeat s.d.")
                repeat_std = float(s.mean())
            plotted += [dict(panel=lab, metric=m, arm=a, mean=float(d[c].mean()), n_units=len(d),
                             split_half_repeat_std=repeat_std if a == ref and lab == "b" else np.nan)
                        for a, c in zip([ref, G, S], cols)]
            cluster = d[["cell_name", "drug", "dose"]].astype(str).agg("|".join, axis=1).to_numpy()
            h2h.append(dict(panel=lab, metric=m, vs=S, **share_better(d[cols[1]].to_numpy(float),
                                                                     d[cols[2]].to_numpy(float), cluster, hb)))
    return pd.DataFrame(plotted), pd.DataFrame(h2h)


# ----------------------------------------------------------------------------
# ED PBMC wells (equal-size control draws within vs between wells) and ED split coverage
# ----------------------------------------------------------------------------

WELLS = RAW / "pbmc_wells"


def well_draws_per_condition() -> pd.DataFrame:
    """Each condition's mean over its within-well and its between-well draws; the summary's counts must agree."""
    per = pd.read_csv(WELLS / "well_draws_rows.csv").pivot_table(
        index=["donor", "cytokine", "cell_type", "bc1_well"], columns="arm", values=["deg", "pearson_delta"], aggfunc="mean")
    summary = json.loads((WELLS / "well_draws_summary.json").read_text())
    for m in ("deg", "pearson_delta"):
        if (n := len(per[m].dropna())) != summary[m]["n_conditions"]:
            raise RuntimeError(f"well_draws {m}: rows give n={n}, the summary says {summary[m]['n_conditions']}")
    per.columns = [f"{m}__{a}" for m, a in per.columns]
    return per.reset_index()


def plate_layout() -> pd.DataFrame:
    """One row per first-round (bc1) well: its cytokine and cells over the 12 donors' plates, from the PBMC main run's
    condition census. ED PBMC wells a draws one cytokine per well on every plate, so anything else is refused."""
    d = pd.read_parquet(WELLS / "condition_summary.parquet")
    u = d.groupby(["donor", "cytokine", "bc1_well"], as_index=False).num_cells.sum()
    lay = u.groupby(["bc1_well", "cytokine"]).agg(n_donors=("donor", "nunique"), cells=("num_cells", "sum")).reset_index()
    if lay.bc1_well.nunique() != 96 or len(lay) != 96 or not lay.n_donors.eq(12).all():
        raise RuntimeError("plate layout is not one cytokine per well on all 12 plates")
    lay["row"], lay["col"] = lay.bc1_well.str[0], lay.bc1_well.str[1:].astype(int)
    return lay[["bc1_well", "row", "col", "cytokine", "n_donors", "cells"]].sort_values(["row", "col"])


SPLITS = {"tahoe_bio": ("tahoe", "tahoe/bio/tahoe_legacy_bio_holdout"),
          "tahoe_pert": ("tahoe", "tahoe/pert/tahoe_overlap_pert_fixed_r0_es0_allcells"),
          "parse_bio": ("parse", "parse/bio/parse_cytokines"),
          "parse_pert": ("parse", "parse/pert/parse_overlap_pert_fixed_r0_es0")}
LOW, HIGH = {0.05, 0.5}, {5.0}                  # Tahoe doses (uM) on each side of a concentration split


def split_combinations() -> pd.DataFrame:
    """Each benchmark split applied to its condition summary by the membership code training uses: one row per
    (split, context, perturbation, dose), plates pooled (they must agree), controls flagged (all in training). A
    context x perturbation may mix sets only as a concentration split: held out at 0.05/0.5 uM, trained at 5."""
    from src.config.dataset_registry import load_dataset_registry   # src.data pulls in torch: only here
    from src.data.splits.data_class import SplitConfig
    from src.data.splits.generation import load_condition_summary, load_split_policy
    from src.data.splits.predicates import split_membership_masks
    combos = []
    for key, (ds, path) in SPLITS.items():
        reg = load_dataset_registry(ds)
        s = load_condition_summary(load_split_policy(ds), reg)
        cfg = SplitConfig.from_dict(yaml.safe_load((REPO / f"core_util_data/splits/{path}.yaml").read_text()))
        tr, va, te, ctrl = split_membership_masks(s, cfg, dict(reg.control_pert))
        if ((tr.astype(int) + va + te) != 1).any() or (ctrl & ~tr).any():
            raise ValueError(f"{path}: a condition is unassigned or in two sets, or a control is outside training")
        ctx, pert = reg.biological_context_columns[0], reg.perturbation_columns[0]
        s = pd.DataFrame({"context": s[ctx].astype(str).values, "perturbation": s[pert].astype(str).values,
                          "dose": s["dose"].astype(float).round(4).values if "dose" in s else np.nan,
                          "set": np.select([va, te], ["validation", "test"], "train"), "control": ctrl})
        mixed = s.groupby(["context", "perturbation"]).set.transform("nunique").gt(1)
        for (c, p), g in s[mixed].groupby(["context", "perturbation"]):
            if g.set.nunique() != 2 or set(g.dose[g.set.eq("train")]) != HIGH or set(g.dose[g.set.ne("train")]) != LOW:
                raise ValueError(f"{path}: {c} x {p} mixes sets {sorted(set(g.set))} other than a concentration split")
        per_dose = s.drop_duplicates().assign(split=key)
        if per_dose.duplicated(["context", "perturbation", "dose"]).any():
            raise ValueError(f"{path}: one concentration of a pair is in different sets on different plates")
        combos.append(per_dose)
    return pd.concat(combos, ignore_index=True)[["split", "context", "perturbation", "dose", "set", "control"]]


# ----------------------------------------------------------------------------
# Then every file, compared with MANIFEST.tsv
# ----------------------------------------------------------------------------

STEMS = list(f2.PAIR_ARMS)
EDP, EDC = "ED_pert_generalization/ed_pert_generalization", "ED_conditioning"
TABLES = {
    "main/confidence_intervals.csv": figure2_cis,
    "main/model_comparison/ceilings.csv": lambda: ceilings(["tahoe_bio", "parse_bio"], SPACES),
    "main/model_comparison/per_condition.parquet": figure2c,
    **{f"main/plate_pairs/pair_metrics__{s}.parquet": (lambda s=s: pair_table("tahoe_bio", s)) for s in STEMS},
    "main/plate_pairs/seen_in_training.csv": seen_in_training,
    "main/plate_pairs/too_few_cells.csv": too_few_cells,
    **{f"{EDC}/{d}/{f}": (lambda p=p, f=f: conditioning(p)[f])
       for d, p in (("tahoe", "tahoe_bio_input_ablation"),)
       for f in ("arms.csv", "per_condition.parquet", "reference.csv")},
    "ED_objective/loss_ablation_mmd_source_data_genesis_vs_mmd.csv": lambda: mmd_tables()[1],
    "ED_objective/loss_ablation_mmd_source_data_plotted.csv": lambda: mmd_tables()[0],
    "ED_pbmc_wells/plate_layout.csv": plate_layout,
    "ED_pbmc_wells/well_draws_energy.csv": lambda: (WELLS / "well_draws_energy.csv").read_bytes(),
    "ED_pbmc_wells/well_draws_per_condition.csv": well_draws_per_condition,
    "ED_pbmc_wells/well_draws_summary.json": lambda: (WELLS / "well_draws_summary.json").read_bytes(),
    f"{EDP}/confidence_intervals.csv": pert_cis,
    **{f"{EDP}/{p}__ceilings.csv": (lambda p=p: ceilings([p], (SPACE,))) for p in PERT},
    **{f"{EDP}/{p}__per_condition.parquet": (lambda p=p: per_condition(p)) for p in PERT},
    **{f"ED_pert_generalization/ed_pert_vs_repeat/pair_metrics__{s}.parquet": (lambda s=s: pair_table("tahoe_pert_fixed", s))
       for s in STEMS},
    "ED_pert_generalization/tahoe_pert_overlap/cross_batch/benchmark_pairs.csv": benchmark_pairs,
    "ED_split_coverage/split_combinations.parquet": split_combinations,
}
FLOAT_FORMAT = {**{f"{EDC}/{d}/{f}": "%.10g" for d in ("tahoe", "pbmc") for f in ("arms.csv", "reference.csv")},
                "ED_objective/loss_ablation_mmd_source_data_genesis_vs_mmd.csv": "%.6g",
                "ED_objective/loss_ablation_mmd_source_data_plotted.csv": "%.6g",
                "ED_pbmc_wells/well_draws_per_condition.csv": "%.8g"}   # as each old writer rounded; the rest full
# metric_bars.py --export-tables writes with csv.DictWriter, whose lines end in CRLF (Figure 2's own ceilings.csv: LF)
CRLF = {f"{EDP}/{p}__ceilings.csv" for p in PERT}


def serialize(rel: str, t: pd.DataFrame | bytes) -> bytes:
    if isinstance(t, bytes):
        return t
    if rel.endswith(".parquet"):
        t.to_parquet(buf := io.BytesIO(), index=False)
        return buf.getvalue()
    return t.to_csv(index=False, float_format=FLOAT_FORMAT.get(rel),
                    lineterminator="\r\n" if rel in CRLF else "\n").encode()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="overwrite the files that differ, and their MANIFEST rows")
    ap.add_argument("--out", type=Path, help="copy the folder to this fresh path and write the rebuilt files there, "
                                             "leaving core_deliverables/data/figure2 untouched")
    args = ap.parse_args()
    dest, write = OUT, args.write or args.out is not None
    if args.out is not None:
        if args.out.exists():                   # never merge into an old copy: its stale files would look rebuilt
            raise SystemExit(f"--out {args.out} exists; give a fresh path")
        dest = Path(shutil.copytree(OUT, args.out))
    manifest = pd.read_csv(dest / "MANIFEST.tsv", sep="\t", dtype=str, keep_default_na=False).set_index("path")
    if odd := set(manifest.index) ^ set(TABLES):
        raise RuntimeError(f"MANIFEST.tsv and this script list different files: {sorted(odd)}")
    differ = []
    for rel, build in TABLES.items():
        blob = serialize(rel, build())
        sha, want = hashlib.sha256(blob).hexdigest(), manifest.at[rel, "sha256"]
        state = "same bytes" if sha == want else "DIFFERS"
        # a parquet's bytes carry the writer's library versions; its values are what the figure reads
        if state == "DIFFERS" and rel.endswith(".parquet") and hashlib.sha256((dest / rel).read_bytes()).hexdigest() == want \
                and pd.read_parquet(io.BytesIO(blob)).equals(pd.read_parquet(dest / rel)):
            state = "same values"
        print(f"{state:12s} {rel}", flush=True)
        if state == "DIFFERS":
            differ.append(rel)
            if write:
                (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                (dest / rel).write_bytes(blob)
                manifest.loc[rel, ["bytes", "sha256"]] = [str(len(blob)), sha]
    if write and differ:
        manifest.reset_index().to_csv(dest / "MANIFEST.tsv", sep="\t", index=False)
        print(f"wrote {len(differ)} files and their MANIFEST.tsv rows into {dest}")
    elif differ:
        raise SystemExit(f"{len(differ)} of {len(TABLES)} files differ from MANIFEST.tsv (--write to replace them)")
    else:
        print(f"all {len(TABLES)} files match MANIFEST.tsv")


if __name__ == "__main__":
    main()
