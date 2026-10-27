#!/usr/bin/env python3
"""Every value the PROGENy Figure 4 on the 3,072 panel draws, into the bundle's data/: GENESIS 5% vs State 5%, three
splits (raw, centred, depth). Reads the eval cache (and OmniPath once, if the PROGENy file is missing): never run
automatically.
    python scripts/figures/figure4/score/main/progeny_tables.py --bundle <bundle> [--check]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
from scripts.figures.figure4.score.main import truth as T  # noqa: E402  truth from the n_3072 cache
from scripts.figures.figure4.score import drug_axis  # noqa: E402
from scripts.figures.figure4.extract.progeny import (FLOOR, MIX, N_REAL, assert_pad_ungated, depth, fetch_progeny,  # noqa: E402
                                      gated_pairs, mae_table, progeny, safe, score, write)

LIBRARY = Path("context_analysis/data/upstream/library_size__tahoe.tsv")   # tscp per condition and its controls
ARMS = ["genesis", "state"]


# ----------------------------------------------------------------------------
# Responses: the plate mixture, held-out lines, truth from the n_3072 cache
# ----------------------------------------------------------------------------

def responses_heldout(truth: T.Truth, conds: dict, replicated: set, means: dict
                      ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """(keys, lfc, gate) for every held-out line in ``truth``: per-gene log2FC over the whole panel
    (0 outside the gate) with its gene gate. keys = (cell_name, drug, arm, slot): the mixture of the
    drug's plates (slot MIX) and, for a ``replicated`` pair, each plate alone (slot 0 / 1, plate a =
    the lower plate number), of truth and of every arm. The gate is the truth controls' detection,
    the same for every arm."""
    keys, lfc, gate = [], [], []

    def add(c: str, drug: str, arm: str, slot: int, x: np.ndarray, ok: np.ndarray) -> None:
        if not np.isfinite(x[ok]).all():
            raise SystemExit(f"{c}/{drug}/{arm}/slot {slot}: a gated log2FC is not finite")
        keys.append((c, drug, arm, slot))
        lfc.append(np.where(ok, x, 0.0))
        gate.append(ok)

    for c in sorted({k[0] for k in truth.pmean}):
        for drug, plates in conds.items():
            ps = [f"plate{p}" for p in plates]
            if any((c, drug, p) not in truth.pmean for p in ps):
                continue
            ctl = [truth.cmean[(c, p)] for p in ps]
            ok = np.mean([truth.D[(c, p)] for p in ps], axis=0) >= FLOOR
            eps = len(ps) * 1e-9
            den = sum(ctl) + eps

            def every(arm: str, ms: list[np.ndarray]) -> None:
                add(c, drug, arm, MIX, np.log2((sum(ms) + eps) / den), ok)
                if drug in replicated:
                    for slot, (m_, c_) in enumerate(zip(ms, ctl)):
                        add(c, drug, arm, slot, np.log2((m_ + 1e-9) / (c_ + 1e-9)), ok)
            every("truth", [truth.pmean[(c, drug, p)] for p in ps])
            for m in ARMS:
                every(m, [means[(c, drug, m, int(p))] for p in plates])
    return pd.DataFrame(keys, columns=["cell_name", "drug", "arm", "slot"]), np.array(lfc), np.array(gate)


# ----------------------------------------------------------------------------
# Checks and tables
# ----------------------------------------------------------------------------

def check_rows(keys: pd.DataFrame, held: set, conds: dict, replicated: set) -> None:
    """Raise unless truth has one mixture row per held-out (line, drug), two single-plate rows per
    replicated one and none for the rest, and every arm scores exactly truth's rows."""
    truth = keys[keys.arm == "truth"]
    if (n := int((truth.slot == MIX).sum())) != len(held) * len(conds):
        raise SystemExit(f"{n} truth mixture rows, expected {len(held) * len(conds)}")
    if set(truth.drug[truth.slot != MIX]) != replicated or (truth.slot != MIX).sum() != 2 * len(held) * len(replicated):
        raise SystemExit("single-plate rows are not exactly the replicated pairs, two plates each")
    rows = set(zip(truth.cell_name, truth.drug, truth.slot))
    for m in ARMS:
        sub = keys[keys.arm == m]
        if set(zip(sub.cell_name, sub.drug, sub.slot)) != rows:
            raise SystemExit(f"{m} does not score exactly truth's rows")
    print(f"  ✓ {len(rows)} truth rows, every arm on the same rows")


def coverage(truth: T.Truth, members: list[str], conds: dict) -> pd.Series:
    """Per held-out line: members whose controls detect them (D >= FLOOR), averaged over its gated
    (drug, plate) rows, on the n_3072 controls."""
    gi = {g: i for i, g in enumerate(truth.genes)}
    cols = [gi[m] for m in members]
    per = {}
    for (c, drug, plate) in truth.pmean:
        if drug in conds:
            per.setdefault(c, []).append(int((truth.D[(c, plate)][cols] >= FLOOR).sum()))
    return pd.Series({c: float(np.mean(v)) for c, v in per.items()})


def tables(keys, lfc, gate, sets: pd.DataFrame, truth: T.Truth, conds: dict, held: set, axis: set,
           centre: bool, shift: np.ndarray | None) -> dict[str, pd.DataFrame]:
    """Figure 4's files for one split, held-out lines only. Values for every drug scored; metrics over the
    ``axis`` drugs, the ones the main panels draw."""
    t = score(keys, lfc, gate, sets, truth.genes, centre, shift).assign(set="grid")
    mv = t[t.slot == MIX].assign(held_out=lambda x: x.cell_name.isin(held))
    mv = mv[["set", "programme", "cell_name", "drug", "arm", "v", "n_genes", "genes", "held_out"]]
    pv = t[t.slot != MIX][["set", "programme", "cell_name", "drug", "arm", "slot", "v"]]
    cov = pd.concat([coverage(truth, list(g.gene), conds).rename("mean_members").rename_axis("cell_name")
                     .reset_index().assign(programme=p) for p, g in sets.groupby("programme", sort=False)])
    cov = cov[["programme", "cell_name", "mean_members"]].assign(held_out=lambda x: x.cell_name.isin(held))
    met = mae_table(mv[mv.drug.isin(axis)], "cell_name", "drug", sorted(held), ARMS)
    return {"programmes.tsv": sets, "module_values.parquet": mv, "plate_values.parquet": pv,
            "coverage.tsv": cov, "metrics.tsv": met}


def main() -> None:
    """Truth from the n_3072 cache, GENESIS 5% and State 5% means, on the drug axis; the mixture is responses_heldout's,
    the three splits progeny.score's (centre / depth shift). Nothing is written unless truth reproduces the
    atlas on the shared genes and every arm scores truth's rows."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, required=True, help="reads BUNDLE/data/upstream, writes BUNDLE/data/<split>/")
    ap.add_argument("--check", action="store_true", help="verify, write nothing")
    a = ap.parse_args()
    bundle = a.bundle.resolve()
    up = bundle / "data/upstream"
    _, test = gated_pairs(up)
    held = {safe(c) for c in test}
    ax = drug_axis.table(bundle / "data/drug_axis/drug_axis.tsv")
    conds, replicated, axis = drug_axis.conditions(ax), set(ax.drug[ax.replicated]), set(ax.drug[ax.on_axis])
    truth = T.load_truth(test, conds)
    T.check_against_atlas(truth, T.ATLAS)
    if len(truth.genes) != N_REAL + 6:
        raise SystemExit(f"expected the 3,072 panel, got {len(truth.genes)} genes")
    z = np.load(up / "arm_means__progeny3072.npz")
    M = z["means"]      # each z[...] decompresses the whole array anew: read it once
    means = {(c, d, m, int(p)): M[i]
             for i, (c, d, m, p) in enumerate(zip(z["cell_name"], z["drug"], z["model"], z["pred_plate"]))}
    print("responses (held-out lines, truth + GENESIS + State, all genes):")
    keys, lfc, gate = responses_heldout(truth, conds, replicated, means)
    assert_pad_ungated(gate)
    check_rows(keys, held, conds, replicated)
    dep = depth(keys, conds, bundle / LIBRARY)
    if not (tsv := up / "progeny_omnipath_academic.tsv").exists():
        fetch_progeny(tsv)                                      # network, once
    sets = progeny(tsv, set(truth.genes[:N_REAL]))
    common = {"gated_pairs.tsv": pd.DataFrame([dict(drug=d, plate_a=conds[d][0], plate_b=conds[d][1])
                                               for d in sorted(replicated)]),
              "heldout_lines.tsv": pd.DataFrame(dict(cell_name=[safe(c) for c in test], display=test))}
    for split, centre, shift in (("tahoe_progeny_raw", False, None), ("tahoe_progeny_centred", True, None),
                                 ("tahoe_progeny_depth", False, dep)):
        print(f"{split}:")
        write(bundle / "data", split, tables(keys, lfc, gate, sets, truth, conds, held, axis, centre, shift) | common,
              a.check)


if __name__ == "__main__":
    main()
