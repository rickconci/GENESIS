#!/usr/bin/env python3
"""The PBMC fewer-cells/fewer-conditions arms (ED 13b) -> OUT/scaling_arms.csv: per arm its run and the training cells
its run.log reports. The 5%-conditions arm is left out (no perturbed condition of a held-out donor); the 100% arm is the
donor-holdout model, not Figure 2c's.

    python scripts/figures/figure3/score/ed_scaling_pert_split/pbmc_runs.py --out <dir>      # ed_scaling_pert_split/score.sh
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[5]
RUNS = REPO / "runs/parse/bio/scaling"
# (axis, nominal % kept, run family). 100 % is shared by both axes; full_100 links to
# runs/parse/bio/main, which also holds newer runs, so the family is named exactly.
ARMS = [("full", 100, "full_100/parse_bio_well_conds100_5ep_donorholdout_seed0"),
        ("cells", 25, "n_cells_025/parse_bio_well_cells025_20ep_donorholdout_seed0"),
        ("cells", 5, "n_cells_005/parse_bio_well_cells005_100ep_donorholdout_seed0"),
        ("conds", 25, "n_conds_025/parse_bio_well_conds025_20ep_donorholdout_seed0")]
N_TRAIN_RE = re.compile(r"n_train=([\d,]+)")


def one(path_glob: str) -> Path:
    hits = sorted(RUNS.glob(path_glob))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly one {RUNS / path_glob}, found {len(hits)}")
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="directory scaling_arms.csv is written to")
    out = ap.parse_args().out
    rows = []
    for axis, pct, family in ARMS:
        # the run whose test eval the arm was published from (its after-fit summary marks the finished run)
        run = one(f"{family}/*/eval/after_fit/test/metrics_summary.json").parents[3]
        if not (m := N_TRAIN_RE.search((run / "run.log").read_text(errors="replace"))):
            raise RuntimeError(f"no n_train in {run / 'run.log'}")
        rows.append(dict(axis=axis, pct_kept=pct, leg=family.split("/")[0], n_train=int(m.group(1).replace(",", "")),
                         run=str(run.relative_to(REPO))))
    arms = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    arms.to_csv(out / "scaling_arms.csv", index=False)
    print(arms[["axis", "pct_kept", "leg", "n_train"]].to_string(index=False))
    print(f"wrote {out / 'scaling_arms.csv'}")


if __name__ == "__main__":
    main()
