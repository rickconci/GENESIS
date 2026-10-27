"""Argparse + entry points for the two designs (scripts/eval/reference.py split_half, replicate_pairs)."""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import torch

from src.config import DatasetConditionSpec
from src.config.config import PertGenConfig
from src.data.data_utils import EvalManifest, parse_geneset_dirname
from src.data.splits.generation import load_split_policy
from src.eval.cache.eval_cache_io import gene_set_hash
from src.eval.reference.pairs import (
    DEFAULT_REPORT_SPACE, REPORT_SPACES, EvalNpzReader, load_neighbors, load_reference,
    prediction_path, score_replicate_pair, set_dataset_median_umi,
)
from src.eval.reference.replicate import enumerate_pairs, replicate_design
from src.eval.orchestration.artifacts import compute_noise_ceiling
from src.eval.scoring import read_gene_names
from src.paths import expand_path

log = logging.getLogger("reference")


def namespace(template: str) -> Path:
    ns = Path(expand_path(template)).resolve()
    if not (ns.parent.name.startswith("normalization_") and ns.is_dir()):
        raise SystemExit(f"--eval-cache must be an existing <...>/normalization_<id>/geneset_<hash>__n_<N>: {ns}")
    return ns


def namespace_genes(ns: Path, gene_names: Path) -> list[str]:
    parsed = parse_geneset_dirname(ns.name)
    if parsed is None:
        raise SystemExit(f"eval cache basename must be geneset_<hash>__n_<N>[__suffix]: {ns.name!r}")
    genes = read_gene_names(gene_names)
    if (gene_set_hash(genes), len(genes)) != parsed:
        raise SystemExit(f"{gene_names} is not the gene set of {ns.name} (hash/length differ)")
    return genes


def device_for(name: str) -> torch.device:
    """--device as asked: a CUDA device without CUDA stops here, never falls back to hours on the CPU."""
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"--device {name}: CUDA is not available (pass --device cpu to score on the CPU)")
    return torch.device(name)


def cmd_split_half(a: argparse.Namespace) -> int:
    """split_half: each condition's cells in disjoint halves, scored against each other -> noise_ceiling.json."""
    spec = DatasetConditionSpec.from_alias(a.dataset)
    ns = namespace(a.eval_cache)
    genes = namespace_genes(ns, a.gene_names)
    cond_vars = list(a.condition_variables or spec.conditioning_vars)
    manifest = EvalManifest(
        condition_summary=Path(a.condition_summary or load_split_policy(a.dataset).condition_summary),
        eval_cache_dir=ns, group_cols=tuple(cond_vars), gene_set_hash=gene_set_hash(genes),
        n_genes=len(genes), split_config=str(a.split_config.resolve()),
        normalization_id=ns.parent.name.removeprefix("normalization_"),
    )
    config = PertGenConfig(
        dataset=spec.dataset_alias, split_config=str(a.split_config.resolve()),
        model_conditioning_variables=cond_vars,
        control_condition={str(k): str(v) for k, v in spec.control_condition.items()},
        control_match_keys=list(a.control_match_keys or spec.control_match_keys),
        noise_ceiling_n_repeats=a.n_repeats, eval_min_cells=a.min_cells,
        eval_metric_set=a.metric_set, eval_gene_sidecars=a.gene_sidecars,
        seed=a.seed, noise_ceiling_dir=str(a.out_root) if a.out_root else None, overwrite_noise_ceiling=a.overwrite,
    )
    written = compute_noise_ceiling(config, genes, manifest, eval_splits=tuple(a.eval_splits),
                                    drop_control_rows=not a.score_controls, max_conditions=a.max_conditions)
    for split, path in written.items():
        log.info("%s -> %s", split, path)
    return 0


def cmd_replicate_pairs(a: argparse.Namespace) -> int:
    """replicate_pairs: one pair-table row per (condition, replicate pair) for one model -> --out parquet."""
    if a.fmt == "pseudobulk" and a.report_space != "log1p_rgs":
        raise SystemExit("--fmt pseudobulk is a log1p_rgs pseudobulk; slice_med needs per-cell depth")
    if (a.pred_dir is None) != (a.fmt == "copy"):
        raise SystemExit("--pred-dir is required for npy/npz/pseudobulk and meaningless for --fmt copy")
    if a.out.is_file() and not a.overwrite:
        log.info("exists, skipping (use --overwrite): %s", a.out)
        return 0
    d = replicate_design(a.dataset)
    log.info("dataset=%s replicate=%s L_ref=%s", d.dataset, d.replicate_col, set_dataset_median_umi(d.dataset))
    neighbors = load_neighbors(a.neighbor_tsv, d) if a.fmt == "copy" else {}
    ref = load_reference(a.reference, d, a.report_space)
    device = device_for(a.device)
    pairs = enumerate_pairs(d, pd.read_csv(a.conditions, sep="\t"), a.replicates_col)
    log.info("%d replicate pairs from %s; model=%s fmt=%s ctrl_mode=full energy_cells=%s", len(pairs),
             a.conditions, a.model, a.fmt, "all" if a.energy_all_cells else "capped")
    reader = EvalNpzReader(namespace(a.eval_cache), cache_size=64)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    rows, skipped, t0 = [], 0, time.perf_counter()
    for cond, i, j in pairs:
        ffmt = "npy" if a.fmt == "pseudobulk" else a.fmt
        if a.fmt != "copy" and not all(prediction_path(a.pred_dir, d, cond, r, ffmt).is_file() for r in (i, j)):
            skipped += 1
            continue
        row = score_replicate_pair(d, reader, cond, i, j, model=a.model, fmt=a.fmt, pred_dir=a.pred_dir,
                                   report_space=a.report_space, ctrl_mode="full", neighbors=neighbors,
                                   ref=ref, device=device, copy_ctrl=a.copy_ctrl,
                                   energy_all_cells=a.energy_all_cells)
        if row is None:
            skipped += 1
            continue
        rows.append(row)
        if len(rows) % 25 == 0:
            pd.DataFrame(rows).to_parquet(a.out, index=False)
            log.info("%d pairs scored (%d skipped)", len(rows), skipped)
    if not rows:
        raise SystemExit(f"no pairs scored for {a.model}; check --pred-dir / --fmt")
    pd.DataFrame(rows).to_parquet(a.out, index=False)
    log.info("wrote %d pairs (skipped %d) in %.1fs -> %s", len(rows), skipped, time.perf_counter() - t0, a.out)
    return 0


def add_split_half_parser(sub: argparse._SubParsersAction, *, common: argparse.ArgumentParser) -> None:
    s = sub.add_parser("split_half", parents=[common], help="disjoint halves of one condition's cells")
    s.add_argument("--split-config", type=Path, required=True)
    s.add_argument("--gene-names", type=Path, required=True, help="the namespace's gene list (txt or pkl)")
    s.add_argument("--eval-splits", nargs="+", default=["test"], choices=("validation", "test", "train"))
    s.add_argument("--n-repeats", type=int, default=20, help="the published references are nr_20")
    s.add_argument("--metric-set", default="core", help="configs/eval/metrics.yaml set; extended = every key")
    s.add_argument("--gene-sidecars", action="store_true", help="also write the per-gene parquet")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--min-cells", type=int, default=4)
    s.add_argument("--condition-summary", type=Path, default=None, help="default: the dataset's split policy")
    s.add_argument("--condition-variables", nargs="+", default=None, help="default: the dataset registry")
    s.add_argument("--control-match-keys", nargs="+", default=None, help="default: the dataset registry")
    s.add_argument("--score-controls", action="store_true", help="keep control rows (e.g. test PBS wells)")
    s.add_argument("--max-conditions", type=int, default=None, help="seeded subset (gates, smoke tests)")
    s.add_argument("--out-root", type=Path, default=None, help="default: results/<dataset>/split_half/noise_ceiling")
    s.add_argument("--overwrite", action="store_true")


def add_replicate_pairs_parser(sub: argparse._SubParsersAction, *, common: argparse.ArgumentParser) -> None:
    r = sub.add_parser("replicate_pairs", parents=[common], help="a model vs the experimental repeat")
    r.add_argument("--conditions", type=Path, required=True,
                   help="TSV: the condition columns plus a comma-separated replicate list per condition")
    r.add_argument("--replicates-col", default="plates", help="column of --conditions listing the replicates")
    r.add_argument("--model", required=True, help="label written to the 'model' column")
    r.add_argument("--fmt", choices=("npy", "npz", "copy", "pseudobulk"), required=True,
                   help="copy: a neighbour context's measured cells; pseudobulk: one (1, G) log1p npy per file")
    r.add_argument("--pred-dir", type=Path, default=None, help="prediction root; required unless --fmt copy")
    r.add_argument("--neighbor-tsv", type=Path, default=None,
                   help="--fmt copy: <context column> -> neighbor (one per line), or score.py copy_neighbor's "
                        "neighbors.tsv (one per condition and replicate, neighbor_<context column>)")
    r.add_argument("--reference", type=Path, default=None,
                   help="parquet of precomputed true_true__* on the SAME gene set; pairs it lacks are computed")
    r.add_argument("--copy-ctrl", choices=("heldout", "own"), default="heldout",
                   help="--fmt copy: controls the copied cells are scored against; heldout (published) = the "
                        "test context's, own = the neighbour's (score_replicate_pair)")
    r.add_argument("--energy-all-cells", action="store_true",
                   help="energy distance on every cell of each replicate, not a 2,048-cell subsample; MMD and "
                        "every other metric are unchanged (rows get energy_cells=all)")
    r.add_argument("--report-space", choices=REPORT_SPACES, default=DEFAULT_REPORT_SPACE)
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--overwrite", action="store_true")

