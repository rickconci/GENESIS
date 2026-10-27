#!/usr/bin/env python3
"""Draw a small seeded subset of test conditions, rerun a driver on it, check it against the published files.

A level-3 driver (scripts/figures/figureN/score/<panel>/score.sh) re-scores GENESIS checkpoints. A full run takes GPU-days; with
SUBSET=<dir> it scores only the conditions frozen in <dir>/<name>.json and `check` compares the rerun with the
published core_results row by row, so a regression shows up in minutes.

  draw     --split <yaml> --n 50 --out <dir>/<name>.json [--keys K... --summary <parquet>]
           [--pairs <tsv> --n-pairs 10]
  query    <run_dir | summary.parquet> <subset.json>   -> `index in [...]` for score.py --conditions-query
  summary  <subset.json> <out.parquet>        -> the subset's rows of the summary it was drawn from,
                                                 for --condition-summary
  pairs    <subset.json> <out.tsv>            -> the drawn replicate-pair rows, for replicate_pairs --conditions
  failed   <reference.json> <arm.json>... --summary <parquet> --out <parquet>
                                              -> the summary's rows of the conditions some arm failed,
                                                 for score.py no_response --condition-summary
  check    <produced tree> <published tree> [--only <rel>...] [--new-sample <rel>...] [--new-sample-file <file>]
                                              exit 1 on any difference

  python scripts/eval/subset.py check $OUT/core_results core_results --only <placed>... --new-sample <rel>...
      (what _env.sh's finish runs under SUBSET: only the files that one driver placed)
  python scripts/eval/subset.py check $OUT/core_results core_results --new-sample-file $OUT/.new_sample
      (the whole tree after a full run; finish prints it)

A condition is the dataset's eval group keys (Tahoe: cell_name, drug, dose, plate; Parse: donor, cytokine),
read from core_util_data/datasets/<dataset>.yaml, so one script serves both datasets. A panel scored on finer
conditions (Figure 2's PBMC: donor x cytokine x cell_type x bc1_well) names its keys and summary with --keys
and --summary.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.vocab import to_categorical_str  # noqa: E402

MIN_CELLS = 4  # the eval's own floor (reference.json: eval_min_cells); a condition below it is never scored
PAIR_KEYS = ("cell_name", "drug", "dose", "plate_i", "plate_j")  # one replicate pair of the plate-pair tables
MAX_LINES = 20  # failure lines printed per file; the per-metric max |diff| line always covers all of them


# 1. Draw the subset once from a split YAML, then read it back on every rerun

def canonical_keys(df: pd.DataFrame, keys: tuple[str, ...]) -> list[tuple[str, ...]]:
    """The work queue's string form of each row's condition (to_categorical_str, as encode_rows)."""
    keys_of = list(zip(*(map(to_categorical_str, df[k]) for k in keys)))
    if len(set(keys_of)) != len(keys_of):
        raise ValueError(f"a condition appears twice under keys {keys}")
    return keys_of


def draw_conditions(pool: pd.DataFrame, keys: tuple[str, ...], n: int, seed: int) -> list[dict]:
    """Uniform draw over sorted keys, so the subset does not depend on any file's row order."""
    ordered = sorted(canonical_keys(pool, keys))
    if n > len(ordered):
        raise ValueError(f"asked for {n} conditions but the pool has only {len(ordered)}")
    rng = np.random.default_rng(seed)
    return [dict(zip(keys, ordered[i])) for i in sorted(rng.choice(len(ordered), n, replace=False))]


def subset_rows(summary: pd.DataFrame, keys: tuple[str, ...], conds: list[dict], where: str) -> pd.DataFrame:
    """The summary's rows for the frozen conditions; a missing one would make every later check vacuous."""
    want = {tuple(c[k] for k in keys) for c in conds}
    rows = summary[[k in want for k in canonical_keys(summary, keys)]]
    if len(rows) != len(want):
        raise SystemExit(f"{where}: {len(rows)} rows match the {len(want)} frozen conditions")
    return rows


def dataset_of(split: Path) -> tuple[Any, Any]:
    """(registry, split policy) of the dataset a split YAML names, for its condition keys and summary."""
    import yaml
    from src.config.dataset_registry import load_dataset_registry
    from src.data.splits.generation import load_split_policy

    reg = load_dataset_registry(yaml.safe_load(split.read_text())["dataset"])
    return reg, load_split_policy(reg.name)


def repo_path(p: Path) -> str:
    """Repo-relative when inside the repo, so a subset JSON reads the same on every host."""
    p = Path(p).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def draw(split: Path, n: int, seed: int, pairs: Path | None, n_pairs: int,
         keys: Sequence[str] | None = None, summary: Path | None = None) -> dict:
    """keys / summary default to the dataset registry's condition keys and its split policy's summary."""
    from src.data.data_utils import filter_condition_summary_for_eval

    reg, policy = dataset_of(split)
    keys = tuple(keys or reg.derive_conditioning_vars())
    summary = Path(summary or policy.condition_summary)
    pool = filter_condition_summary_for_eval(pd.read_parquet(summary), split_config=split,
                                             eval_splits=("test",), control_condition=reg.control_pert,
                                             drop_control_rows=True, min_cells=MIN_CELLS)
    out = {"seed": seed, "split": repo_path(split), "dataset": reg.name, "keys": list(keys), "summary": repo_path(summary),
           "pool": f"{len(pool)} eligible test conditions (min_cells {MIN_CELLS}, controls dropped)",
           "conditions": draw_conditions(pool, keys, n, seed)}
    if pairs is not None:  # the replicate-pair TSV: one row per condition, its plates comma-joined
        tsv = pd.read_csv(pairs, sep="\t", dtype=str)
        picked = np.random.default_rng(seed).choice(len(tsv), n_pairs, replace=False)
        out |= {"pairs_source": str(pairs), "pairs": tsv.iloc[sorted(picked)].to_dict("records")}
    return out


# 2. The subset as each scoring command's own input

def load(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def query(source: Path, sub: dict) -> str:
    """Row labels of the summary a scorer reads: a run dir's own (score.py genesis reads it from the run's
    manifest), or a summary parquet (every other arm's --condition-summary). Two summaries of one split can
    order their rows differently, so a query is only ever valid for the file it was made from."""
    from src.data.data_utils import load_eval_manifest

    path = source if Path(source).suffix == ".parquet" else load_eval_manifest(source).condition_summary
    rows = subset_rows(pd.read_parquet(path), tuple(sub["keys"]), sub["conditions"], f"summary {path}")
    return f"index in {sorted(int(i) for i in rows.index)}"


def summary_of(sub: dict) -> pd.DataFrame:
    """The subset's rows of the summary it names; a subset that names none was drawn from the split policy's."""
    path = ROOT / sub["summary"] if "summary" in sub else dataset_of(ROOT / sub["split"])[1].condition_summary
    return subset_rows(pd.read_parquet(path), tuple(sub["keys"]), sub["conditions"], f"summary {path}")


def failed(reference: Path, arms: Sequence[Path]) -> dict:
    """The conditions the split-half reference scores and some arm does not, as a subset (keys, conditions):
    a condition the truth can score but an arm failed on is scored as that arm predicting no response."""
    keys = tuple(load(reference)["condition_variables"])
    def scored(path: Path) -> set[tuple[str, ...]]:
        return {tuple(to_categorical_str(r["condition_keys"][k]) for k in keys) for r in load(path)["results"]}
    truth, gone = scored(reference), set()
    for arm in arms:
        if miss := truth - scored(arm):
            print(f"failed: {arm}: {len(miss)} of {len(truth)} conditions")
            gone |= miss
    return {"keys": list(keys), "reference": repo_path(reference), "arms": [repo_path(a) for a in arms],
            "conditions": [dict(zip(keys, c)) for c in sorted(gone)]}


# 3. Compare. Every file becomes {row key: {leaf path: value}}; a difference is a failure line

def leaves(x: Any, path: str = "") -> dict[str, Any]:
    """Every leaf of nested dicts/lists keyed by its path, so a metric moved between groups still shows."""
    if isinstance(x, dict):
        return {k: v for key, val in x.items() for k, v in leaves(val, f"{path}/{key}" if path else str(key)).items()}
    if isinstance(x, list):
        return {k: v for i, val in enumerate(x) for k, v in leaves(val, f"{path}/{i}").items()}
    return {path: x}


def same(a: Any, b: Any) -> bool:
    return a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))


def gap(a: Any, b: Any) -> float:
    numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (a, b))
    return abs(a - b) if numeric and not (math.isnan(a) or math.isnan(b)) else math.inf


# Published metric keys no kernel emits, so a rerun lacking them is expected, not a failure.
# Evidence: src/eval/metrics/deg.py emits the top-p overlap only as `..._adaptivek_truth_...` (no plain
# `..._detfloor10` variant), and no source file names pearson_r_allpert.
# Add a key only with that kind of evidence: any other missing key is a failure.
RETIRED = frozenset({
    *(f"deg_raw_int_rgs/deg_overlap_top_p_ranked_k{k}_fdr0p001_lfc0p1_detfloor10" for k in (20, 50, 100, 500)),
    "log1p_rgs/pearson_r_allpert",
    "log1p_slice_med_rgs/pearson_r_allpert",
    # the permutation energy test (--energy-test-n-perms): no source file implements it
    # (`git grep energy_test -- src scripts configs` finds only this list); the energy distance itself is emitted
    *(f"{space}/energy_test_{m}" for space in ("log1p_rgs", "log1p_slice_med_rgs")
      for m in ("distance", "pvalue", "significant")),
})


# The one tolerance: Figure 2's two no-response files were scored on CPU, and no rerun reproduces their energy and
# MMD leaves bit for bit (they differ by <= 5.7e-6 at every thread count tried); every other leaf is exact. Only these
# files, only these leaves; everything else stays exact.
TOLERANCE = {rel: 6e-6 for rel in ("figure2/no_response/tahoe_bio.json", "figure2/no_response/parse_bio.json")}
TOLERANT = re.compile(r"energy_distance|mmd")


def retired(leaf: str) -> bool:
    """A flat column, or a JSON leaf `metrics/<group>/<key>` (a reference's repeat s.d.: `metrics_std/<group>/<key>`),
    whose whole key is in RETIRED (never a suffix match)."""
    return re.fullmatch(r"(?:metrics(?:_std)?/\w+/)?(.+)", leaf)[1] in RETIRED


def compare_rows(produced: dict, published: dict, where: str, tol: float = 0.0) -> tuple[list, list, int]:
    """(failures, notes, rows compared). A published leaf the rerun lacks fails; an extra rerun leaf is a note.
    tol > 0 lets TOLERANT leaves differ by up to tol (reported as a note); every other leaf must be equal."""
    fails, notes, worst, n, within = [], [], {}, 0, 0.0
    gone = {m for k, new in produced.items() if k in published for m in published[k].keys() - new.keys()}
    gone_retired = {m for m in gone if retired(m)}  # once per file, not per row
    notes += [f"retired: {m}" for m in sorted(gone_retired)]
    for key, new in produced.items():
        if key not in published:
            fails.append(f"row {key} is not in published")
            continue
        n += 1
        old = published[key]
        fails += [f"{key} {m}: published has it, rerun does not" for m in old.keys() - new.keys() - gone_retired]
        for m in sorted(old.keys() & new.keys()):
            if isinstance(old[m], str) and old[m].startswith("/"):  # names the machine it was made on, not a result
                notes.append(f"machine path, not compared: {m}")
                continue
            if not same(old[m], new[m]):
                d = gap(old[m], new[m])
                if tol and TOLERANT.search(m) and d <= tol:
                    within = max(within, d)
                    continue
                worst[m] = max(worst.get(m, 0.0), d)
                fails.append(f"{key} {m}: published={old[m]!r} rerun={new[m]!r}")
        if extra := sorted(new.keys() - old.keys()):
            notes.append(f"extra in rerun, not published: {extra}")
    if within:
        notes.append(f"energy/MMD within the {tol:g} tolerance (max |diff| {within:.3g})")
    if n == 0:
        fails.append(f"{where}: no row of the rerun is in the published file (zero rows compared)")
    if worst:
        fails.append("max |diff| per metric: " + ", ".join(f"{m}={d:.3g}" for m, d in sorted(worst.items())))
    return fails, sorted(set(notes)), n


def summary_table(path: Path) -> dict[tuple, dict[str, Any]]:
    """results[] of a metrics_summary / noise_ceiling JSON, keyed by condition_keys."""
    rows: dict = {}
    for r in load(path)["results"]:
        key = tuple(sorted((k, str(v)) for k, v in r["condition_keys"].items()))
        if key in rows:
            raise ValueError(f"{path}: condition {key} appears twice")
        rows[key] = leaves({k: v for k, v in r.items() if k != "condition_keys"})
    return rows


def pair_table(t: pd.DataFrame, path: Path) -> dict[tuple, dict[str, Any]]:
    keys = list(zip(*(map(to_categorical_str, t[k]) for k in PAIR_KEYS)))
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path}: a replicate pair appears twice")
    vals = t.drop(columns=list(PAIR_KEYS)).astype(object).where(t.notna(), float("nan")).to_dict("records")
    return dict(zip(keys, vals))


def narrow_floats(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A column in two float widths (float32 vs float64) is compared at the narrower one."""
    a, b = a.copy(), b.copy()
    for c in a.columns.intersection(b.columns):
        if a[c].dtype != b[c].dtype and a[c].dtype.kind == b[c].dtype.kind == "f":
            a[c], b[c] = a[c].astype(np.float32).astype(float), b[c].astype(np.float32).astype(float)
    return a, b


def compare_pairs(new: Path, old: Path, rel: str) -> tuple[list, list, int]:
    """A published column the rerun lacks fails (as for JSON); an extra rerun column is a note. A column in
    two dtype widths (float32 vs float64) compares at the narrower one."""
    a, b = narrow_floats(pd.read_parquet(new), pd.read_parquet(old))
    shared = [*PAIR_KEYS, *sorted((a.columns.intersection(b.columns)).difference(PAIR_KEYS))]
    fails, notes, n = compare_rows(pair_table(a[shared], new), pair_table(b[shared], old), rel)
    gone = b.columns.difference(a.columns)
    notes += [f"retired: {c}" for c in gone if retired(c)]
    fails += [f"column {c}: published has it, rerun does not" for c in gone if not retired(c)]
    if extra := sorted(a.columns.difference(b.columns)):
        notes.append(f"extra in rerun, not published: {extra}")
    return fails, notes, n


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")


def compare_tables(new: Path, old: Path, rel: str) -> tuple[list, list, int]:
    """Any other table: the non-float columns and dose (a float condition key, as in PAIR_KEYS) identify a row (a
    metric is never an identifier), the other float columns are its values, so a subset rerun is checked on the rows
    it scored."""
    a, b = narrow_floats(read_table(new), read_table(old))
    keys = [c for c in b.columns if b[c].dtype.kind != "f" or c == "dose"]
    def rows(t: pd.DataFrame, path: Path) -> dict:
        key = list(zip(*(t[k].astype(str) for k in keys)))
        if len(set(key)) != len(key):
            raise ValueError(f"{path}: a row appears twice (key columns {keys})")
        vals = t.drop(columns=keys, errors="ignore").astype(object).reset_index(drop=True)
        return dict(zip(key, vals.to_dict("index").values()))  # not "records": a table with no value column gives []
    missing = [f"column {c}: published has it, rerun does not" for c in keys if c not in a.columns]
    if missing:
        return missing, [], 0
    return compare_rows(rows(a, new), rows(b, old), rel)


def is_pair_table(t: pd.DataFrame) -> bool:
    """One row per replicate pair; a table that repeats pairs (the ladder's reps) is an ordinary table."""
    return set(PAIR_KEYS) <= set(t.columns) and not t.duplicated(list(PAIR_KEYS)).any()


def compare_file(rel: str, new: Path, old: Path) -> tuple[list, list, int]:
    if new.suffix == ".json":
        a, b = load(new), load(old)
        if isinstance(a, dict) and isinstance(a.get("results"), list) \
                and all("condition_keys" in r for r in a["results"][:1]):  # empty results: still a row table, zero rows
            return compare_rows(summary_table(new), summary_table(old), rel, TOLERANCE.get(rel, 0.0))
        return compare_rows({(): leaves(a)}, {(): leaves(b)}, rel)
    if new.suffix == ".parquet" and is_pair_table(pd.read_parquet(old)):
        return compare_pairs(new, old, rel)
    # a text table headed by # comments is a hand-curated input placed as-is (Figure 4's gene sets and stories): bytes
    hand_curated = new.suffix in (".csv", ".tsv") and old.read_bytes().startswith(b"#")
    if new.suffix in (".parquet", ".csv", ".tsv") and not hand_curated:
        return compare_tables(new, old, rel)
    return ([] if new.read_bytes() == old.read_bytes() else [f"{rel}: bytes differ"]), [], 1


def print_means(rel: str, new: Path, old: Path | None) -> None:
    """For a file the paper never had a number for: published vs rerun mean per metric, for eyeballing only.
    A table without one row per replicate pair (Figure 4's NES and plate tables) is averaged column by column as
    it is: no key to join on, and a mean needs none."""
    def means(path: Path) -> dict:
        if path.suffix == ".json":
            rows = pd.DataFrame(list(summary_table(path).values())).apply(pd.to_numeric, errors="coerce")
        else:  # a pair table's values without its keys; any other table as it is
            rows = read_table(path)
            if is_pair_table(rows):
                rows = pd.DataFrame(list(pair_table(rows, path).values())).apply(pd.to_numeric, errors="coerce")
        return rows.mean(numeric_only=True).to_dict()  # a string column has no mean and is left out

    if new.suffix not in (".json", ".parquet", ".csv", ".tsv"):
        print(f"[new-sample] {rel}: not a table, no means printed; never a failure")
        return
    new_m, old_m = means(new), (means(old) if old is not None and old.is_file() else {})
    print(f"[new-sample] {rel}: published mean (all rows) vs rerun mean (subset rows); never a failure")
    for m, v in sorted(new_m.items()):
        print(f"  {m}: published={old_m.get(m, float('nan')):.4g} rerun={v:.4g}")


def check(produced: Path, published: Path, new_sample: Sequence[str] = (), only: Sequence[str] | None = None) -> int:
    """0 only if every produced file equals its published twin on the rows the rerun scored.

    `only` names the files one driver placed: the rest of the tree is other drivers' and is not compared. A driver
    that placed nothing, or only new samples, passes with a note: it said so. A whole tree must compare something."""
    produced, published = Path(produced), Path(published)
    names = sorted(str(p.relative_to(produced)) for p in produced.rglob("*") if p.is_file())
    bad = len(unknown := sorted(set(new_sample) - set(names)))  # a typo here would silently exempt nothing
    for rel in unknown:
        print(f"FAIL --new-sample {rel}: not among the produced files")
    if only is not None:
        bad += len(gone := sorted(set(only) - set(names)))
        for rel in gone:
            print(f"FAIL --only {rel}: not among the produced files")
        if not only:
            print("note: this driver placed nothing, nothing to compare")
            return int(bad > 0)
        names = sorted(set(only) & set(names))
    elif not names:
        print(f"FAIL: {produced} holds no files, nothing was compared")
        return 1
    compared = 0
    for rel in names:
        new, old = produced / rel, published / rel
        if rel in new_sample:
            print_means(rel, new, old)
            continue
        if not old.is_file():
            fails, notes, n = [f"{rel}: not in {published}"], [], 0
        else:
            fails, notes, n = compare_file(rel, new, old)
        print(f"{'FAIL' if fails else 'ok  '} {rel} ({n} rows compared)")
        more = [f"... {len(fails) - MAX_LINES} more", fails[-1]] if len(fails) > MAX_LINES else []
        for line in (*fails[:MAX_LINES], *more, *(f"note: {m}" for m in notes)):
            print(f"  {line}")
        bad += bool(fails)
        compared += 1
    if not compared and only is None:  # every file was exempt: nothing was checked, so nothing can pass
        print("FAIL: every produced file is in --new-sample, nothing was compared")
        return 1
    if not compared:
        print("note: every placed file is a new sample (reported above), nothing to compare")
    return int(bad > 0)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw")
    d.add_argument("--split", type=Path, required=True, help="split YAML; its `dataset:` picks the condition keys")
    d.add_argument("--n", type=int, default=50)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--keys", nargs="+", help="condition keys (default: the dataset registry's)")
    d.add_argument("--summary", type=Path, help="condition summary to draw from (default: the split policy's)")
    d.add_argument("--pairs", type=Path, help="replicate-pair TSV (columns cell_name, drug, dose, plates)")
    d.add_argument("--n-pairs", type=int, default=10)
    q = sub.add_parser("query")
    q.add_argument("source", type=Path, help="a run dir (its manifest's summary) or a summary parquet")
    q.add_argument("subset", type=Path)
    s = sub.add_parser("summary")
    s.add_argument("subset", type=Path)
    s.add_argument("out", type=Path)
    pr = sub.add_parser("pairs")
    pr.add_argument("subset", type=Path)
    pr.add_argument("out", type=Path)
    f = sub.add_parser("failed")
    f.add_argument("reference", type=Path, help="the panel's split-half reference.json")
    f.add_argument("arms", type=Path, nargs="+", help="every arm's metrics_summary.json the panel draws")
    f.add_argument("--summary", type=Path, required=True, help="the condition summary no_response reads")
    f.add_argument("--out", type=Path, required=True, help="parquet: that summary's rows of the failed conditions")
    c = sub.add_parser("check")
    c.add_argument("produced", type=Path)
    c.add_argument("published", type=Path)
    c.add_argument("--new-sample", nargs="*", default=[], help="paths (relative to the trees) with no published twin")
    c.add_argument("--new-sample-file", type=Path, help="more such paths, one per line ($OUT/.new_sample)")
    c.add_argument("--only", nargs="*", help="compare only these paths: the files one driver placed")
    a = p.parse_args(argv)
    if a.cmd == "draw":
        if a.out.exists():
            raise SystemExit(f"{a.out} exists; a frozen subset is never redrawn")
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(draw(a.split.resolve(), a.n, a.seed, a.pairs, a.n_pairs, a.keys, a.summary), indent=2))
    elif a.cmd == "query":
        print(query(a.source, load(a.subset)))
    elif a.cmd == "summary":
        if a.out.exists():
            raise SystemExit(f"{a.out} exists; never overwritten")
        a.out.parent.mkdir(parents=True, exist_ok=True)
        summary_of(load(a.subset)).to_parquet(a.out, index=False)
    elif a.cmd == "pairs":  # the TSV's own rows and columns, as strings: a dose stays "5.0", plates stay "6,14"
        if a.out.exists():
            raise SystemExit(f"{a.out} exists; never overwritten")
        if not (rows := load(a.subset).get("pairs")):
            raise SystemExit(f"{a.subset} has no pairs: draw it with --pairs")
        a.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(a.out, sep="\t", index=False)
    elif a.cmd == "failed":
        if a.out.exists():
            raise SystemExit(f"{a.out} exists; never overwritten")
        a.out.parent.mkdir(parents=True, exist_ok=True)
        summary_of({**failed(a.reference, a.arms), "summary": a.summary}).to_parquet(a.out, index=False)
    else:
        listed = a.new_sample_file.read_text().split() if a.new_sample_file else []  # a missing file raises
        return check(a.produced, a.published, [*a.new_sample, *listed], a.only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
