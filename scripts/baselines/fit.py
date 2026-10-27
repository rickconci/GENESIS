#!/usr/bin/env python3
"""Fit the linear baselines (perturbation_mean, context_mean, linear_ridge, additive) for one split.

The roles come from the dataset registry, overridable like score.py (src/baselines/roles.py);
every kind is saved as <out-dir>/<kind>.npz:

  # Tahoe: held axis cell_name, plates as strata
  python scripts/baselines/fit.py --dataset tahoe --split-yaml <split.yaml> \\
      --eval-cache-namespace <ns> --gene-names core_util_data/tahoe/gene_subsets/STATE_HVG.txt --out-dir <dir>

  # PBMC per-well cache: held axis donor, cell type kept, wells mixed into each control
  python scripts/baselines/fit.py --dataset parse --split-yaml <split.yaml> \\
      --condition-variables donor cytokine cell_type bc1_well --control-match-keys donor cell_type \\
      --eval-cache-namespace <ns>__cell_type__well --condition-summary <summary> \\
      --gene-names core_util_data/parse/gene_subsets/pbmc_real_selected_genes.pkl --out-dir <dir>

Then score each kind with ``scripts/eval/score.py baseline --artifact <dir>/<kind>.npz`` on the same split.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.splits.generation import load_split_policy  # noqa: E402
from src.baselines.artifacts import KINDS  # noqa: E402
from src.baselines.fit import fit_split, write_fit_summary  # noqa: E402
from src.baselines.roles import baseline_condition_roles  # noqa: E402
from src.baselines.space import FIT_SPACES, RAW, library_namespace  # noqa: E402
from src.eval.cache.eval_npz_reader import EvalNpzReader  # noqa: E402
from src.eval.scoring import read_gene_names  # noqa: E402

log = logging.getLogger("fit_baselines")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="registry alias: condition columns, control, match keys")
    ap.add_argument("--split-yaml", type=Path, required=True)
    ap.add_argument("--eval-cache-namespace", type=Path, required=True)
    ap.add_argument("--gene-names", type=Path, required=True, help="ordered gene list (.txt or .pkl)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--condition-summary", type=Path, help="default: the dataset's split policy summary")
    ap.add_argument("--condition-variables", nargs="+", help="override the registry's condition columns")
    ap.add_argument("--control-match-keys", nargs="+", help="override the registry's control match keys")
    ap.add_argument("--train-subset", type=Path,
                    help="HBSS runs: the run's hbss_selected_triplets.parquet; train keeps only its "
                         "(held axis, perturbation) keys")
    ap.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS),
                    help="which artifacts to (re)write; e.g. 'additive' alone extends a dir")
    ap.add_argument("--fit-cells", type=int, default=512,
                    help="cells subsampled per perturbed condition and control unit for its mean (<= 0: all)")
    ap.add_argument("--fit-space", choices=FIT_SPACES, default=RAW,
                    help="what each mean averages (src/baselines/space.py); log1p_cp10k_full reads L_full from "
                         "<namespace>__with_libfull and needs --fit-cells 0")
    ap.add_argument("--ridge-lambda", type=float, default=0.1, help="0.1 = the Nature Methods reference")
    ap.add_argument("--pca-dim", type=int, default=100)
    ap.add_argument("--min-cells", type=int, default=4)
    ap.add_argument("--workers", type=int, default=12, help="NFS read threads")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: cap the train conditions (0 = all)")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args()
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(message)s")

    t0 = time.time()
    roles = baseline_condition_roles(a.dataset, condition_variables=a.condition_variables,
                                     control_match_keys=a.control_match_keys)
    summary = Path(a.condition_summary or load_split_policy(a.dataset).condition_summary)
    genes = read_gene_names(a.gene_names)
    log.info("dataset=%s roles=%s unit=%s control=%s=%r", a.dataset, roles.axes, roles.unit_var,
             roles.pert_var, roles.control_pert_value)
    ns = a.eval_cache_namespace if a.fit_space == RAW else library_namespace(a.eval_cache_namespace)
    fields = fit_split(reader=EvalNpzReader(ns), roles=roles, summary=summary,
                       split_yaml=a.split_yaml, genes=genes, out_dir=a.out_dir, kinds=set(a.kinds),
                       fit_cells=a.fit_cells, workers=a.workers, ridge_lambda=a.ridge_lambda, pca_dim=a.pca_dim,
                       min_cells=a.min_cells, train_subset=a.train_subset, limit=a.limit, fit_space=a.fit_space)
    fields = {"dataset": a.dataset, **fields, "split_yaml": str(a.split_yaml),
              "cache_namespace": str(ns), "condition_summary": str(summary),
              "gene_names": str(a.gene_names), "n_genes": len(genes), "normalization_id": "raw_counts",
              "train_subset": str(a.train_subset) if a.train_subset else None, "fit_cells": a.fit_cells,
              "ridge_lambda": a.ridge_lambda, "pca_dim": a.pca_dim, "min_cells": a.min_cells, "limit": a.limit,
              "elapsed_seconds": time.time() - t0}
    log.info("wrote %s in %.1f min", write_fit_summary(a.out_dir, fields, set(a.kinds)), (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
