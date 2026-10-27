#!/usr/bin/env python3
"""Figure 3a's training-set sizes, counted from the runs themselves (main/score.sh, ed_12cells_per_condition/score.sh):

  --runs               OUT/scaling_runs.csv                   one row per run, 3 per point (Figure 3a's variants)
  --cells-arm uniform  OUT/scaling_arms_uniform_cells.csv     one row per arm, the uniform-draw cells arm (ED 12c)

Training cells are what each run.log says it trained on (downsampling is not exact); training conditions are a
conds arm's HBSS pick, else counted from its cached train rows, because a uniform cell budget can empty a condition.

    python scripts/figures/figure3/score/training_runs.py --runs --out <dir>
    python scripts/figures/figure3/score/training_runs.py --cells-arm uniform --out <dir>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from scripts.figures.figure3.score.scaling import CELLS_ARMS, CONTROL_DRUG, REPLICATES  # noqa: E402

RUNS = REPO / "runs/tahoe"
N_TRAIN_RE = re.compile(r"n_train=([\d,]+)")
# Row-level obs codes (cell line, drug + dose) of the training cache, one sidecar per shard
HVG_CACHE = "{data:tahoe}/hvg_cache/geneset_c0d6420ffc0fa861__n_2000"   # src/paths.py: expand_path


# 1. Training-set sizes, read from each run

def n_train(run_dir: Path) -> int:
    """Training cells actually used, as the run logged them (downsampling is not exact)."""
    m = N_TRAIN_RE.search((run_dir / "run.log").read_text(encoding="utf-8", errors="replace"))
    if not m:
        raise RuntimeError(f"no n_train in {run_dir / 'run.log'}")
    return int(m.group(1).replace(",", ""))


def obs_conditions() -> tuple[np.ndarray, np.ndarray]:
    """Per global row: a condition code (cell line x drug x dose) and whether it is a control."""
    from src.paths import expand_path
    cache = Path(expand_path(HVG_CACHE))
    man = json.loads((cache / "manifest.json").read_text())
    cond, ctrl, off = [], [], 0
    for s in man["shards"]:                                  # global row = shard offset + local row
        if s["global_obs_offset"] != off:
            raise RuntimeError(f"shard {s['shard_id']}: offset {s['global_obs_offset']} != {off}")
        z = np.load(cache / s["cache_subdir"] / "obs_sidecar.npz", allow_pickle=True)
        if not np.array_equal(z["local_obs_index"], np.arange(s["n_obs"])):
            raise RuntimeError(f"shard {s['shard_id']}: rows are not stored in order")
        cell = pd.Series(z["cell_name__categories"][z["cell_name__codes"]]).astype(str)
        dose = pd.Series(z["drugname_drugconc__categories"][z["drugname_drugconc__codes"]]).astype(str)
        cond.append(cell.str.cat(dose, sep="|").to_numpy())
        ctrl.append(z["drug__categories"][z["drug__codes"]] == CONTROL_DRUG)
        off += s["n_obs"]
    if off != man["total_obs"]:
        raise RuntimeError(f"{off} rows read, manifest says {man['total_obs']}")
    return pd.factorize(np.concatenate(cond))[0], np.concatenate(ctrl)


def n_conds(run_dir: Path, obs: tuple[np.ndarray, np.ndarray]) -> tuple[int, int]:
    """(training conditions, condition pool). A conds arm logs its HBSS pick; any other arm is
    counted from its cached train rows, because a uniform cell budget can empty a condition."""
    meta = run_dir / "hbss_meta.json"
    if meta.is_file():
        m = json.loads(meta.read_text())
        return m["n_selected"], m["n_total_triplets"]
    code, ctrl = obs
    idx = np.load(run_dir / "split_cache" / "train_idx.npy")
    return len(np.unique(code[idx[~ctrl[idx]]])), -1        # pool: checked against the 100% arm


# 2. The two tables; every run must draw from the 100% arm's condition pool

def same_pool(t: pd.DataFrame) -> pd.DataFrame:
    if not t.pool[t.pool.ge(0)].eq(t.n_conds[t.stem.eq("genesis")].item()).all():
        raise RuntimeError(f"condition pool disagrees across runs:\n{t}")
    return t.drop(columns="pool")


def runs_table(obs: tuple[np.ndarray, np.ndarray]) -> pd.DataFrame:
    """scaling_runs.csv: one row per REPLICATES run, with its control cells."""
    rows = []
    for a, p, i, s, r in REPLICATES:
        conds, pool = n_conds(RUNS / r, obs)
        idx = np.load(RUNS / r / "split_cache" / "train_idx.npy")
        if len(idx) != (n := n_train(RUNS / r)):
            raise RuntimeError(f"{s}: split cache has {len(idx)} train rows, run.log says {n}")
        rows.append(dict(axis=a, pct_kept=p, run=i, stem=s, n_train=n, n_ctrl=int(obs[1][idx].sum()),
                         n_conds=conds, pool=pool))
    return same_pool(pd.DataFrame(rows))


def arms_table(scaling: list, obs: tuple[np.ndarray, np.ndarray]) -> pd.DataFrame:
    """scaling_arms*.csv: one row per arm (one run each)."""
    rows = []
    for a, p, s, r in scaling:
        conds, pool = n_conds(RUNS / r, obs)
        rows.append({"axis": a, "pct_kept": p, "stem": s, "n_train": n_train(RUNS / r), "n_conds": conds, "pool": pool})
    return same_pool(pd.DataFrame(rows))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    job = ap.add_mutually_exclusive_group(required=True)
    job.add_argument("--runs", action="store_true", help="scaling_runs.csv (Figure 3a's 3-run variants)")
    job.add_argument("--cells-arm", choices=CELLS_ARMS, help="which cells arm the scaling_arms table lists")
    ap.add_argument("--out", type=Path, required=True, help="directory the table is written to")
    a = ap.parse_args()
    obs = obs_conditions()
    name, table = (("scaling_runs.csv", runs_table(obs)) if a.runs
                   else (CELLS_ARMS[a.cells_arm][0], arms_table(CELLS_ARMS[a.cells_arm][1], obs)))
    print(table.to_string(index=False))
    a.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.out / name, index=False)
    print(f"wrote {a.out / name}")


if __name__ == "__main__":
    main()
