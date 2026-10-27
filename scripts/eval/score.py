#!/usr/bin/env python3
"""Score one arm against the eval truth cache -> <out-dir>/metrics_summary.json (+ per-gene parquet on --gene-sidecars).

Arms:
  genesis   a GENESIS checkpoint: generate and score in one streaming pass (data from the run config)
  npz       external model, one prediction NPZ per row (STATE, CellFlow, PerturbDiff, ...)
  baseline  fitted perturbation_mean / context_mean / linear_ridge / additive artifact
            (scripts/baselines/fit.py writes one <kind>.npz per kind)
  copy_neighbor  the closest train context's cells for the test perturbation (--metric cosine)
  copy_lines     no scoring: ONE closest train context per held-out context over every perturbation
                 (the copy baseline Figure 4 draws), as a TSV
  no_response    the condition's own matched controls as the prediction: what a condition an arm
                 failed on is scored as (--conditions-query picks those conditions)

  python scripts/eval/score.py genesis --run-dir <run> --checkpoint last --eval-split test

  python scripts/eval/score.py npz --model state --predictions-root <dir> \\
      --dataset tahoe --split-yaml <split.yaml> --eval-cache-namespace <ns> --gene-names <genes> --out-dir <dir>
  python scripts/eval/score.py baseline --artifact additive.npz \\
      --dataset tahoe --split-yaml <split.yaml> --eval-cache-namespace <ns> --gene-names <genes> --out-dir <dir>

Condition columns, control and control-match keys come from the dataset registry
(core_util_data/datasets/<dataset>.yaml); the condition summary defaults to the
dataset's split policy. The level-3 drivers (scripts/figures/figureN/score/<panel>/score.sh) are the
record of every published arm's exact arguments; the split-half reference is scripts/eval/reference.py.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import DatasetConditionSpec, build_eval_profile  # noqa: E402
from src.data.splits.generation import load_split_policy  # noqa: E402
from src.eval.cache.eval_cache_io import gene_set_hash  # noqa: E402
from src.eval.plots.gene_arrays import METRICS_GENE_ARRAYS_FILENAME  # noqa: E402
from src.eval.metrics.reporting import load_metric_set
from src.eval.metrics.spaces import set_dataset_median_umi  # noqa: E402
from src.eval.orchestration.model_eval import pregenerated_eval_plan  # noqa: E402
from src.eval.scoring import (ArmSource, NoResponse, build_work, read_gene_names, score_conditions,  # noqa: E402
                              write_summary)
from src.model.inference_output import ModelOutputSpace  # noqa: E402

log = logging.getLogger("score")
ROOT = Path(__file__).resolve().parents[2]


def _common(args: argparse.Namespace):
    """Dataset semantics, gene list, work queue, device: everything an arm does not own."""
    if args.device.startswith("cuda") and not torch.cuda.is_available():  # a bad GPU index: stop, never score on CPU
        raise SystemExit(f"--device {args.device}: CUDA is not available (pass --device cpu to score on the CPU)")
    device = torch.device(args.device)
    spec = DatasetConditionSpec.from_alias(args.dataset)
    cond_vars = tuple(args.condition_variables or spec.conditioning_vars)
    control = {str(k): str(v) for k, v in spec.control_condition.items()}
    match_keys = tuple(args.control_match_keys or spec.control_match_keys)
    summary = args.condition_summary or load_split_policy(args.dataset).condition_summary
    genes = read_gene_names(args.gene_names)
    ns = args.eval_cache_namespace.resolve()
    if not ns.is_dir():
        raise SystemExit(f"--eval-cache-namespace not found: {ns}")
    log.info("composition L_ref=%.1f (dataset=%s)", set_dataset_median_umi(args.dataset), args.dataset)
    work = build_work(condition_summary=summary, split_yaml=args.split_yaml, eval_split=args.eval_split,
                      condition_variables=cond_vars, control_condition=control, control_match_keys=match_keys,
                      min_cells=args.min_cells, max_conditions=args.max_conditions or None,
                      conditions_query=args.conditions_query)
    return spec, cond_vars, genes, ns, work, device


def _plan(args: argparse.Namespace, ctrl_det_floors: Optional[tuple[float, ...]] = None):
    """The K=1 split-half plan for --metric-set, writing the per-gene parquet on --gene-sidecars."""
    return pregenerated_eval_plan(ctrl_det_floors=ctrl_det_floors or (0.10,),
                                  metrics=load_metric_set(args.metric_set), gene_sidecars=args.gene_sidecars)


def _shared_fields(args, spec, genes, cond_vars) -> dict:
    return {"dataset": spec.dataset_alias, "gene_set_id": gene_set_hash(genes), "n_genes": len(genes),
            "eval_split": args.eval_split, "model_eval_n_repeats": 1, "metric_set": args.metric_set,
            "gene_arrays_parquet": METRICS_GENE_ARRAYS_FILENAME if args.gene_sidecars else None,
            **({"conditions_query": args.conditions_query} if args.conditions_query else {})}


def cmd_npz(args: argparse.Namespace) -> dict:
    """An external model's prediction NPZs (src/eval/arms/npz.py), one per work-queue row, scored as log1p."""
    from src.eval.arms.npz import (NpzPredGetter, build_prediction_index, default_index_variables,
                                   npz_gene_names, qrow_lookup_key, resolve_gene_reindex)

    spec, cond_vars, genes, ns, work, device = _common(args)
    root = args.predictions_root.resolve()
    index_vars = tuple(args.index_variables or default_index_variables(cond_vars))
    index = build_prediction_index(root, index_vars, max_workers=args.index_workers)
    hits = sum(1 for q in work.rows if qrow_lookup_key(q, index_vars) in index)
    log.info("%s coverage | matched=%d/%d rows index_vars=%s", args.model, hits, len(work.rows), list(index_vars))
    getter = NpzPredGetter(index, index_vars, gene_names=genes,
                           gene_reindex=resolve_gene_reindex(npz_gene_names(index), genes),
                           match_rows=args.match_rows, min_cells=args.min_cells)
    out = ModelOutputSpace.LOG1P
    plan = _plan(args, tuple(f for f in args.ctrl_det_floors if f >= 0))
    fields = {"model_id": f"{args.model}__{root.name}", "model": args.model, "competitor": args.model,
              **_shared_fields(args, spec, genes, cond_vars), "run_dir": str(root), "checkpoint": str(root),
              "model_output_space": out.value, "noise_ceiling_strategy": "split_half",
              "predictions_npz_root": str(root), "predictions_n_indexed": len(index),
              "predictions_row_hits": hits, "predictions_row_misses": len(work.rows) - hits,
              "predictions_match_rows": bool(args.match_rows),
              "index_variables": list(index_vars), "condition_variables_work_queue": list(cond_vars)}
    return dict(getter=getter, out=out, plan=plan, fields=fields, genes=genes, ns=ns, work=work, device=device)


def cmd_baseline(args: argparse.Namespace) -> dict:
    """A fitted baseline artifact. Two context axes (PBMC donor x cell type) emit as the published
    PBMC bars did (whole control pool, adler32 seeds); one context axis as Tahoe's did."""
    from src.eval.arms.baseline import (EMISSION_PER_CELL, EMISSION_TILE_MEAN, BaselinePredGetter,
                                        ThreeFactorPredGetter, validate_gene_agreement)
    from src.baselines.artifacts import Baseline
    from src.baselines.roles import baseline_condition_roles
    from src.baselines.space import RAW
    from src.eval.cache.eval_npz_reader import EvalNpzReader

    artifact = args.artifact.resolve()
    spec, cond_vars, genes, ns, work, device = _common(args)
    fit = Baseline.load(artifact)
    validate_gene_agreement(fit.genes, genes, artifact)
    roles = baseline_condition_roles(args.dataset, condition_variables=args.condition_variables,
                                     control_match_keys=args.control_match_keys)
    reader, space = EvalNpzReader(ns), fit.params.get("fit_space", RAW)
    fields = {"model_id": f"baseline__{fit.kind}", "baseline_kind": fit.kind, **_shared_fields(args, spec, genes, cond_vars),
              "run_dir": str(artifact.parent), "checkpoint": str(artifact), "baseline_fit_space": space,
              "noise_ceiling_strategy": "split_half", "baseline_artifact": str(artifact), "baseline_axes": fit.axes,
              "condition_variables_work_queue": list(cond_vars)}
    # The getter says how to read its rows: raw counts for a raw fit, log1p(counts) for a fit-space one.
    if len(roles.context_vars) == 2:
        if args.emission != "per_cell":
            raise SystemExit("two-context-axis fits are emitted per cell only")
        getter = ThreeFactorPredGetter(fit=fit, roles=roles, npz_reader=reader, n_genes=len(genes))
        fields.update(baseline_family="three_factor", model_output_space=getter.output_space.value,
                      baseline_broadcast_strategy=EMISSION_PER_CELL if space == RAW else f"per_cell_ctrl_plus_delta_{space}")
        return dict(getter=getter, out=getter.output_space, plan=_plan(args), fields=fields, genes=genes, ns=ns,
                    work=work, device=device)
    emission = EMISSION_PER_CELL if args.emission == "per_cell" else EMISSION_TILE_MEAN
    getter = BaselinePredGetter(fit=fit, n_genes=len(genes), npz_reader=reader, roles=roles, emission=emission,
                                ctrl_pool_cells=args.emission_ctrl_pool, pool_cache_size=args.emission_pool_cache,
                                seed=args.seed)
    fields.update(baseline_broadcast_strategy=emission if space == RAW else f"per_cell_ctrl_plus_delta_{space}",
                  baseline_emission_ctrl_pool=int(args.emission_ctrl_pool) if space == RAW else 0,
                  baseline_ctrl_source="eval_plate_matched_from_cache", model_output_space=getter.output_space.value,
                  plot_aggregate_over=build_eval_profile(args.dataset, repo_root=ROOT).plot_aggregate_over)
    return dict(getter=getter, out=getter.output_space, plan=_plan(args), fields=fields, genes=genes, ns=ns, work=work, device=device,
                after=lambda f: f.update(baseline_plate_ctrl_means_cached=getter.n_ctrl_means_cached,
                                         baseline_plate_ctrl_means_missing=getter.n_ctrl_means_missing))


def cmd_copy_neighbor(args: argparse.Namespace) -> dict:
    """The closest train context's raw cells for the test perturbation (src/eval/arms/copy_neighbor.py)."""
    from src.baselines.condition_stats import load_condition_stats
    from src.baselines.nearest_neighbor import (assert_no_held_arms, build_deltas, load_deltas, tag_of, train_arms,
                                                train_controls)
    from src.baselines.roles import baseline_condition_roles
    from src.eval.arms.copy_neighbor import CopyNeighborPredGetter, CopySource, check_expected_neighbors
    from src.eval.cache.eval_npz_reader import EvalNpzReader

    spec, cond_vars, genes, ns, work, device = _common(args)
    roles = baseline_condition_roles(args.dataset, condition_variables=args.condition_variables,
                                     control_match_keys=args.control_match_keys)
    summary = args.condition_summary or load_split_policy(args.dataset).condition_summary
    train = train_arms(summary, args.split_yaml, roles, args.min_cells)
    reader = EvalNpzReader(ns, cache_size=64)
    deltas = args.deltas.resolve()
    if deltas.is_file():
        summaries = load_deltas(deltas, roles)
    elif args.build_deltas:
        if args.train_cache is None:
            raise SystemExit("--build-deltas reads the per-condition stats of --train-cache <hvg_cache/geneset_...>")
        stats = load_condition_stats(args.train_cache, roles.condition_variables, genes=genes)
        summaries = build_deltas(stats, train, train_controls(summary, args.split_yaml, roles), roles, deltas)
    else:
        raise SystemExit(f"{deltas} missing; --build-deltas builds it (and re-picks every neighbour)")
    assert_no_held_arms(summaries, {tag_of(r.condition_keys, roles) for r in work.rows})
    getter = CopyNeighborPredGetter(args.metric, summaries=summaries, roles=roles, source=CopySource(train, roles),
                                    npz_reader=reader, plate_matched=args.plate_matched,
                                    own_controls=args.own_controls)
    out_dir = args.out_dir.resolve()

    def after(fields: dict) -> None:
        choices = getter.choices_frame()
        choices.to_csv(out_dir / "neighbors.tsv", sep="\t", index=False)
        fields.update(copy_neighbor_misses=dict(getter.misses), copy_neighbor_choices="neighbors.tsv")
        if args.expect_neighbors:
            check_expected_neighbors(choices, args.expect_neighbors, roles)

    fields = {"model_id": f"copy_neighbor_{args.metric}", "model": f"copy_neighbor_{args.metric}",
              **_shared_fields(args, spec, genes, cond_vars), "run_dir": str(deltas.parent),
              "checkpoint": str(deltas), "model_output_space": ModelOutputSpace.RAW.value,
              "noise_ceiling_strategy": "split_half", "copy_neighbor_metric": args.metric,
              "copy_neighbor_deltas": str(deltas), "copy_neighbor_plate_matched": bool(args.plate_matched),
              "copy_neighbor_context": list(roles.context_vars),
              "copy_neighbor_controls": "neighbor" if args.own_controls else "heldout",
              "condition_variables_work_queue": list(cond_vars)}
    return dict(getter=getter, out=ModelOutputSpace.RAW, plan=_plan(args), fields=fields, genes=genes, ns=ns,
                work=work, device=device, after=after)


def cmd_no_response(args: argparse.Namespace) -> dict:
    """The condition's own controls as the prediction (src/eval/scoring.py NoResponse): what a condition the
    truth can score but an arm failed on counts as for that arm. Score it with that arm's seed and profile,
    so its truth half and control halves are the ones the arm would have drawn."""
    if args.gene_sidecars:
        raise SystemExit("no_response writes no per-gene parquet: its prediction is the control half itself")
    spec, cond_vars, genes, ns, work, device = _common(args)
    fields = {"model_id": "no_response", "model": "no_response", **_shared_fields(args, spec, genes, cond_vars),
              "run_dir": "", "checkpoint": "", "model_output_space": ModelOutputSpace.RAW.value,
              "noise_ceiling_strategy": "split_half", "condition_variables_work_queue": list(cond_vars)}
    return dict(source=NoResponse(), plan=_plan(args), fields=fields, genes=genes, ns=ns, work=work, device=device)


def cmd_copy_lines(args: argparse.Namespace) -> int:
    """One neighbour per held-out context (src/baselines/nearest_neighbor.py:line_neighbors) -> TSV."""
    import pandas as pd

    from src.baselines.nearest_neighbor import canon_pert, held_contexts, line_neighbors, load_deltas
    from src.baselines.roles import baseline_condition_roles

    roles = baseline_condition_roles(args.dataset, condition_variables=args.condition_variables,
                                     control_match_keys=args.control_match_keys)
    summary = args.condition_summary or load_split_policy(args.dataset).condition_summary
    held = held_contexts(summary, args.split_yaml, roles)
    plotted = None
    if args.plotted_drugs:
        if bool(roles.dose_var) != (args.plotted_dose is not None):
            raise SystemExit(f"--plotted-dose is {'required' if roles.dose_var else 'meaningless'} for {args.dataset}")
        drugs = sorted(set(pd.read_csv(args.plotted_drugs, sep="\t")[roles.pert_var].astype(str)))
        plotted = [canon_pert((d, args.plotted_dose) if roles.dose_var else (d,), roles) for d in drugs]
    table = line_neighbors(load_deltas(args.deltas.resolve(), roles), held, plotted)
    ctx = pd.DataFrame(table.pop("context").tolist(), columns=list(roles.context_vars))
    out = pd.concat([ctx, table], axis=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, sep="\t", index=False)
    log.info("%d held-out contexts -> %s\n%s", len(out), args.out, out.to_string(index=False))
    return 0


def add_genesis_parser(sub) -> None:
    """GENESIS: generate from a training run's checkpoint and score, streaming on one GPU
    (src/eval/orchestration/model_eval.py:run_model_eval). Reads <run-dir>/eval/eval_manifest.json."""
    p = sub.add_parser("genesis", help="generate + score a GENESIS checkpoint (run config supplies the data)")
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Absolute path to the training run directory.",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Checkpoint path or filename under <run-dir>/checkpoints/. "
            "Use 'last' (default) for the newest .ckpt (prefers last.ckpt)."
        ),
    )
    p.add_argument(
        "--out-dir", type=Path, default=None,
        help=(
            "Default: results/<dataset>/split_half/pertgen/<run_name>__<ckpt_stem>__<split> "
            "(<dataset> from run config_resolved.yaml)."
        ),
    )
    p.add_argument(
        "--results-subdir",
        default=None,
        help=(
            "Optional subgroup under the dataset results root "
            "(e.g. frac_90_golden -> results/tahoe/split_half/pertgen/frac_90_golden/...)."
        ),
    )
    p.add_argument("--eval-split", default="validation", choices=("test", "validation", "train"))
    p.add_argument(
        "--conditions-query",
        default=None,
        help=(
            "Optional pandas query on the condition summary, applied after the "
            "split filter (e.g. \"cell_name in ['C32','PANC-1']\").  Use to carve "
            "a composition-matched subset -- e.g. score train conditions on the "
            "same cell lines the test split holds out."
        ),
    )
    p.add_argument(
        "--max-conditions",
        type=int,
        default=None,
        help=(
            "Cap the number of scored conditions with a seeded uniform sample. "
            "Effectively required for --eval-split train, which can hold tens of "
            "thousands of conditions.  Recorded in metrics_summary.json."
        ),
    )
    p.add_argument(
        "--conditions-seed",
        type=int,
        default=0,
        help="Seed for --max-conditions sampling (same subset across checkpoints).",
    )
    p.add_argument(
        "--score-controls",
        action="store_true",
        help="Keep control rows in the work queue (e.g. score test PBS wells).",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Override config cache_dir (mmap CSR root). Use when the run config "
            "does not point at your prebuilt cache path."
        ),
    )
    p.add_argument(
        "--model-eval-n-repeats",
        type=int,
        default=1,
        help="Override config model_eval_n_repeats (K-rep count for all eval methods).",
    )
    p.add_argument("--metric-set", default=None,
                   help="configs/eval/metrics.yaml set (default: the run config's eval_metric_set, core); "
                        "extended reproduces the published files")
    p.add_argument("--gene-sidecars", action=argparse.BooleanOptionalAction, default=None,
                   help="write the per-gene parquet (default: the run config's eval_gene_sidecars)")
    p.add_argument("--min-cells", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16384)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compiled inference (default: on; pads the last batch). A checkpoint trained with "
             "compile_model=True keeps its own compile (e.g. max-autotune); published GENESIS numbers "
             "are compiled, and eager differs at the draw level.",
    )
    p.add_argument("--pad-last-batch", action="store_true")
    p.add_argument(
        "--pipeline-depth", type=int, default=3,
        help=(
            "Max number of conditions whose GPU prediction buffers are "
            "allocated simultaneously in the streaming pipeline (default 3)."
        ),
    )
    p.add_argument(
        "--metrics-workers", type=int, default=1,
        help=(
            "Threads in ConditionPrefetcher for 1-ahead truth/ctrl I/O. "
            "1 is plenty for mmap-backed eval caches; bump only if many "
            "small conditions stall on prefetch (default 1)."
        ),
    )
    p.add_argument(
        "--persist-predictions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also async-D2H each completed condition's rep-major prediction "
            "buffer and write it to <out-dir>/predictions/<cond>.npy.  Off "
            "by default (predictions stay on GPU, are scored, then freed)."
        ),
    )
    p.add_argument(
        "--ctrl-det-floors",
        type=float,
        nargs="+",
        default=[0.10],
        help=(
            "Independent-filter thresholds on pooled control detection "
            "fraction P(count>0); one `_detfloor{NN}` DEG-family variant "
            "per threshold (0.10 -> `_detfloor10`). Negative values are "
            "dropped; all negative -> the 0.10 default. Rationale: Bourgon "
            "et al. 2010; Squair et al. 2021."
        ),
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="INFO = milestones only, no GPU sync for logs; DEBUG = per-condition detail.",
    )


def cmd_genesis(args: argparse.Namespace) -> int:
    """The run's own config and eval manifest, with the CLI overrides applied, into run_model_eval."""
    from src.eval.orchestration.log_utils import EvalStartupProfiler
    from src.eval.orchestration.model_eval import default_model_eval_out_dir, run_model_eval

    log = logging.getLogger("score.genesis")

    run_dir = Path(args.run_dir).expanduser()
    if not run_dir.is_absolute():
        raise SystemExit(
            f"--run-dir must be an absolute path (got {run_dir}).\n"
            f"Example: --run-dir {ROOT / 'runs' / '<run_name>'}",
        )
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    from src.training.utils import resolve_run_checkpoint
    from src.config import load_config

    log.info("Using run dir: %s", run_dir)
    config = load_config(config_path=run_dir/ "config_resolved.yaml")
    config.model_eval_n_repeats = int(args.model_eval_n_repeats)
    if args.metric_set is not None:
        config.eval_metric_set = args.metric_set
    if args.gene_sidecars is not None:
        config.eval_gene_sidecars = args.gene_sidecars
    if args.cache_dir is not None:
        config.cache_dir = str(args.cache_dir.expanduser().resolve())
        log.info("Using cache_dir override: %s", config.cache_dir)

  
    profiler = EvalStartupProfiler(label="startup")
    with profiler.phase("ensure_eval_manifest"):
        from src.data.data_utils import load_eval_manifest
        manifest = load_eval_manifest(run_dir)

    with profiler.phase("resolve_checkpoint"):
        ckpt = resolve_run_checkpoint(run_dir, args.checkpoint)
    log.info("Using checkpoint: %s", ckpt)
    log.info(
        "Eval method: split_half (K=%d) | metric_set=%s | gene_sidecars=%s",
        config.model_eval_n_repeats, config.eval_metric_set, config.eval_gene_sidecars,
    )
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = default_model_eval_out_dir(
            run_dir, ckpt, args.eval_split,
            config=config,
            results_subdir=args.results_subdir,
        )
    log.info("Output dir: %s", out_dir)


    # Negative floors are dropped; if none remain, run_model_eval gets None and its plan uses the 0.10 default.
    _floors_kept = [float(t) for t in args.ctrl_det_floors if float(t) >= 0.0]
    ctrl_floors: Optional[tuple[float, ...]] = tuple(_floors_kept) if _floors_kept else None

    result = run_model_eval(
        run_dir=run_dir,
        eval_split=args.eval_split,
        manifest=manifest,
        config=config,
        checkpoint=ckpt,
        out_dir=out_dir,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        min_cells=args.min_cells,
        overwrite=args.overwrite,
        pipeline_depth=args.pipeline_depth,
        metrics_workers=args.metrics_workers,
        use_compile=args.compile,
        persist_predictions=bool(args.persist_predictions),
        ctrl_det_floors=ctrl_floors,
        startup_profiler=profiler,
        conditions_query=args.conditions_query,
        max_conditions=args.max_conditions,
        conditions_seed=args.conditions_seed,
        drop_control_rows=not args.score_controls,
    )
    log.info(
        "Done | split=%s conditions=%d -> %s",
        result.eval_split, result.n_conditions, result.metrics_path,
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="arm", required=True)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--dataset", required=True, help="registry alias: condition columns, control, match keys")
    shared.add_argument("--split-yaml", type=Path, required=True)
    shared.add_argument("--eval-cache-namespace", type=Path, required=True)
    shared.add_argument("--gene-names", type=Path, required=True, help="ordered gene list (.txt or .pkl)")
    shared.add_argument("--out-dir", type=Path, required=True)
    shared.add_argument("--condition-summary", type=Path, help="default: the dataset's split policy summary")
    shared.add_argument("--condition-variables", nargs="+", help="override the registry's condition columns")
    shared.add_argument("--control-match-keys", nargs="+", help="override the registry's control match keys")
    shared.add_argument("--eval-split", default="test", choices=("test", "validation", "train"))
    shared.add_argument("--min-cells", type=int, default=4)
    shared.add_argument("--device", default="cuda")
    shared.add_argument("--metrics-workers", type=int, default=4)
    shared.add_argument("--seed", type=int, default=42)
    shared.add_argument("--overwrite", action="store_true")
    shared.add_argument("--metric-set", default="core",
                        help="configs/eval/metrics.yaml set; extended reproduces the published files")
    shared.add_argument("--gene-sidecars", action="store_true", help="also write the per-gene parquet")
    shared.add_argument("--max-conditions", type=int, default=0,
                        help="smoke test / gate: score a seeded random subset of N conditions (0 = all)")
    shared.add_argument("--conditions-query", default=None,
                        help="pandas query on the condition summary after the split filter, e.g. the conditions "
                             "an arm failed (\"cell_name == 'HOP62' and drug == 'olaparib'\")")
    n = sub.add_parser("npz", parents=[shared], help="external model prediction NPZs")
    n.add_argument("--model", required=True, help="label for model_id")
    n.add_argument("--predictions-root", type=Path, required=True)
    n.add_argument("--index-variables", nargs="+", help="NPZ index fields (default: condition columns, plate->plate_norm)")
    n.add_argument("--index-workers", type=int, default=16)
    n.add_argument("--match-rows", action="store_true",
                   help="resample each cloud to the truth's cell count (classified per-cell-type clouds)")
    n.add_argument("--ctrl-det-floors", type=float, nargs="+", default=[0.10],
                   help="DEG detection floors on the pooled control mean (all negative -> the 0.10 default)")
    b = sub.add_parser("baseline", parents=[shared], help="fitted linear baseline artifact")
    b.add_argument("--artifact", type=Path, required=True)
    b.add_argument("--emission", choices=("per_cell", "tile_mean"), default="per_cell",
                   help="per_cell (every published arm) shifts real control cells by the baseline delta; "
                        "tile_mean repeats the one predicted vector n_cells times")
    b.add_argument("--emission-ctrl-pool", type=int, default=2048, help="control cells cached per stratum (0 = all)")
    b.add_argument("--emission-pool-cache", type=int, default=8, help="control pools kept in the LRU")
    c = sub.add_parser("copy_neighbor", parents=[shared], help="closest train context's cells")
    c.add_argument("--deltas", type=Path, required=True, help="per-arm PCA-100 response deltas (deltas.npz)")
    c.add_argument("--build-deltas", action="store_true", help="build --deltas when missing, from --train-cache")
    c.add_argument("--train-cache", type=Path,
                   help="HVG train cache whose condition_stats/ the deltas read (scripts/data/build_cache.py stats)")
    c.add_argument("--metric", choices=("cosine",), required=True,
                   help="distance between PCA-100 response deltas; cosine is the only one")
    c.add_argument("--plate-matched", action="store_true",
                   help="copy the neighbour's cells from the truth's own plate (else its first train plate)")
    c.add_argument("--own-controls", action="store_true",
                   help="score the copied cells against the neighbour's own controls, not the held-out "
                        "context's (the published default); see CopyNeighborPredGetter")
    c.add_argument("--expect-neighbors", type=Path,
                   help="TSV <context>, neighbor that a figure cites; raise if the copy disagrees")
    sub.add_parser("no_response", parents=[shared], help="the condition's own controls (a failed arm's score)")
    cl = sub.add_parser("copy_lines", help="one closest train context per held-out context (no scoring)")
    cl.add_argument("--dataset", required=True)
    cl.add_argument("--split-yaml", type=Path, required=True, help="its test block names the held-out contexts")
    cl.add_argument("--deltas", type=Path, required=True, help="the split's per-arm PCA-100 deltas (deltas.npz)")
    cl.add_argument("--condition-summary", type=Path, help="default: the dataset's split policy summary")
    cl.add_argument("--condition-variables", nargs="+", help="override the registry's condition columns")
    cl.add_argument("--control-match-keys", nargs="+", help="override the registry's control match keys")
    cl.add_argument("--plotted-drugs", type=Path, help="TSV whose perturbation column lists the drugs a figure draws")
    cl.add_argument("--plotted-dose", type=float, help="the dose those drugs are drawn at (datasets with a dose)")
    cl.add_argument("--out", type=Path, required=True)
    add_genesis_parser(sub)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    if args.arm == "copy_lines":
        logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s | %(message)s")
        return cmd_copy_lines(args)
    if args.arm == "genesis":
        logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                            format="%(asctime)s %(levelname)s %(name)s | %(message)s", datefmt="%H:%M:%S")
        return cmd_genesis(args)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s", datefmt="%H:%M:%S")

    out_dir = args.out_dir.resolve()
    metrics_path = out_dir / "metrics_summary.json"
    if metrics_path.is_file() and not args.overwrite:
        raise SystemExit(f"{metrics_path} exists; pass --overwrite to replace it")
    out_dir.mkdir(parents=True, exist_ok=True)
    arm = {"npz": cmd_npz, "baseline": cmd_baseline, "copy_neighbor": cmd_copy_neighbor,
           "no_response": cmd_no_response}[args.arm](args)
    work = arm["work"]
    source = arm["source"] if "source" in arm else ArmSource(arm["getter"], arm["out"])
    results, skips = score_conditions(source, cache_ns=arm["ns"], bundles=work.bundles,
                                      condition_variables=work.condition_variables, plan=arm["plan"],
                                      seed=args.seed, device=arm["device"], metrics_workers=args.metrics_workers,
                                      gene_arrays_path=out_dir / METRICS_GENE_ARRAYS_FILENAME)
    if "getter" in arm:
        log.info("prediction misses: %d", arm["getter"].miss_count)
    if "after" in arm:
        arm["after"](arm["fields"])
    fields = {**arm["fields"], "gene_summary_pred_label": "pred",
              "condition_variables": list(work.condition_variables), "eval_group_keys": list(work.eval_group_keys),
              "eval_cache_namespace": str(arm["ns"]), "normalization_id": "raw_counts",
              "results_subdir": metrics_path.parent.parent.name, "gene_names": list(arm["genes"]),
              "seed": int(args.seed)}
    write_summary(metrics_path, fields=fields, results=results, skips=skips)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
