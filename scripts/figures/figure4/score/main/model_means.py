#!/usr/bin/env python3
"""GENESIS 5% and State 5% on the 3,072 panel: the per-gene mean counts Figure 4 reads, into
data/upstream/arm_means__progeny3072.npz. Reads the predictions (CPU): never run automatically.

    python scripts/figures/figure4/score/main/model_means.py --bundle <bundle>
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
import src.eval.readout.deg_readout as R  # noqa: E402  the loaders every readout uses
from scripts.figures.figure4.score import drug_axis  # noqa: E402  the drugs and their 5 uM plates
from scripts.figures.figure4.extract.progeny import DOSE, gated_pairs, safe  # noqa: E402
from src.model.inference_output import to_raw_integer  # noqa: E402

MEANS = Path("data/upstream/arm_means__progeny3072.npz")    # in the bundle; the counterfactual reads it too
AXIS = Path("data/drug_axis/drug_axis.tsv")
THREADS = 8   # per worker: five lines x 8 = 40 of the node's 48 cores
PANEL = R.resolve_panel("progeny3072")
ARMS = ("genesis", "state")


def mean_count(x: np.ndarray) -> np.ndarray:
    """Per-gene mean over cells after the scorer's snap of log1p cells onto the count lattice."""
    return to_raw_integer(torch.as_tensor(x, dtype=torch.float32), R.MODEL_SPACE).to(torch.float64).mean(0).numpy()


def assert_genesis_axis(summary: Path, genes: list[str]) -> None:
    """GENESIS persists are positional .npy: their columns are the run summary's gene_names."""
    if json.loads(Path(summary).read_text())["gene_names"] != genes:
        raise SystemExit(f"GENESIS gene axis in {summary} differs from the panel's: positional means would mislabel genes")


def score_line(cell: str, conds: dict, genesis: Path) -> tuple[list[dict], list[np.ndarray]]:
    """One held-out line (display spelling, 'HepG2/C3A'): a row per (drug, arm, plate the arm is
    conditioned on). A process of its own, so the five lines run side by side."""
    torch.set_num_threads(THREADS)
    G, keys, means = len(PANEL.genes), [], []
    for drug, plates in conds.items():
        for p in plates:
            n = R.load_perturbed(PANEL.cache, cell_name=cell, drug=drug, dose=DOSE, plate=f"plate{p}", n_genes=G).shape[0]
            for arm in ARMS:
                root, fmt = (genesis / "predictions", PANEL.arms[arm][1]) if arm == "genesis" else PANEL.arms[arm]
                x = R.load_pred(R.pred_path(root, fmt, drug, cell, DOSE, p), PANEL.genes)
                if x.shape != (n, G):
                    raise SystemExit(f"{cell} / {drug} / plate{p} / {arm}: prediction {x.shape}, measured ({n}, {G})")
                keys.append(dict(cell_name=safe(cell), drug=drug, model=arm, pred_plate=int(p)))
                means.append(mean_count(x))
    print(f"  {cell}: {len(conds)} drugs", flush=True)
    return keys, means


def score(conds: dict, test: list[str], genesis: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """(keys, means) over the held-out lines ``test``, one worker each; ``genesis``: the GENESIS persist."""
    assert_genesis_axis(genesis / "metrics_summary.json", PANEL.genes)
    with ProcessPoolExecutor(len(test), mp_context=get_context("spawn")) as ex:
        parts = list(ex.map(score_line, test, [conds] * len(test), [genesis] * len(test)))
    return pd.DataFrame([k for ks, _ in parts for k in ks]), np.array([m for _, ms in parts for m in ms])


def main() -> None:
    """Both arms were trained on the same cells (the Fig 3a 5% cells split, r0) and the same 3,072
    genes, and generated as many cells as each test condition measured. For every held-out line x
    drug of the axis (data/drug_axis/drug_axis.tsv: 106 drugs, 1-3 plates each) x plate at 5 uM they are read
    through deg_readout's ``progeny3072`` panel (GENESIS: its K=1 test persist, .npy; State: one .npz per
    condition whose gene_names load_pred checks), snapped onto the count lattice the scorer's way, and kept as
    the mean count per gene: all Figure 4 uses.

    Refuses to write unless GENESIS's gene axis is the panel's (a .npy carries no gene names), every
    (line, drug, plate, arm) row exists, every mean is finite, and every prediction holds exactly as
    many cells as the measured condition in the n_3072 cache.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, required=True, help="reads BUNDLE/data, writes BUNDLE/" + str(MEANS))
    ap.add_argument("--genesis", type=Path, default=PANEL.arms["genesis"][0].parent,
                    help="the GENESIS K=1 test persist (predictions/ + metrics_summary.json); default: the progeny3072 panel's")
    args = ap.parse_args()
    bundle = args.bundle.resolve()
    out = bundle / MEANS
    if out.exists():
        raise SystemExit(f"{out.relative_to(bundle)} exists; delete it to rescore")
    _, test = gated_pairs(bundle / "data/upstream")
    conds = drug_axis.conditions(drug_axis.table(bundle / AXIS))
    keys, means = score(conds, test, args.genesis.resolve())
    want = len(test) * sum(len(p) for p in conds.values()) * len(ARMS)
    if len(keys) != want or keys.duplicated().any():
        raise SystemExit(f"{len(keys)} rows ({keys.duplicated().sum()} duplicated), expected {want}")
    if not np.isfinite(means).all():
        raise SystemExit("non-finite mean counts")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, means=means, **{c: np.asarray(keys[c].tolist()) for c in keys})   # no pickled objects
    print(f"wrote {out.relative_to(bundle)}: {means.shape}")


if __name__ == "__main__":
    main()
