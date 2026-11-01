#!/usr/bin/env python3
"""The PBMC fewer-cells/fewer-conditions arms (ED 13b) -> OUT/scaling_arms.csv: per arm its model and the training cells
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
MODELS = REPO / "models"                    # the GENESIS checkpoints, by release name (README.md)
# (axis, nominal % kept, leg as published, model). 100 % is shared by both axes.
ARMS = [("full", 100, "full_100", "reproduce/figure3/pbmc_scaling/full"),
        ("cells", 25, "n_cells_025", "reproduce/figure3/pbmc_scaling/cells25"),
        ("cells", 5, "n_cells_005", "reproduce/figure3/pbmc_scaling/cells5"),
        ("conds", 25, "n_conds_025", "reproduce/figure3/pbmc_scaling/conditions25")]
N_TRAIN_RE = re.compile(r"n_train=([\d,]+)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="directory scaling_arms.csv is written to")
    out = ap.parse_args().out
    rows = []
    for axis, pct, leg, model in ARMS:
        log = MODELS / model / "run.log"
        if not (m := N_TRAIN_RE.search(log.read_text(errors="replace"))):
            raise RuntimeError(f"no n_train in {log}")
        rows.append(dict(axis=axis, pct_kept=pct, leg=leg, n_train=int(m.group(1).replace(",", "")), run=model))
    arms = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    arms.to_csv(out / "scaling_arms.csv", index=False)
    print(arms[["axis", "pct_kept", "leg", "n_train"]].to_string(index=False))
    print(f"wrote {out / 'scaling_arms.csv'}")


if __name__ == "__main__":
    main()
