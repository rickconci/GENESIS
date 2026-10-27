#!/usr/bin/env python3
"""PertGen training entry point.

Config: the dataclass defaults (src/config/config.py), then --config's YAML, then CLI overrides (key=value).
Every run names its YAML; each published run's is under configs/.
Usage:
    python -m src.main --config configs/tahoe/bio/main/mlp_h4096x8192_nrs256_vg05_5ep_state2k.yaml
    python -m src.main --config my.yaml workdir=runs/run2 lr=1e-4
    # Resume restores the checkpoint config, optimizer, and epoch; the run's own resolved YAML is the base.
    python -m src.main --config <run>/config_resolved.yaml --workdir <run> resume_ckpt=<run>/checkpoints/last.ckpt
    # weights_ckpt copies weights only; the launch YAML is the config, optimizer starts fresh.
    python -m src.main --config my.yaml --workdir runs/run2 weights_ckpt=runs/run1/checkpoints/last.ckpt
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import WandbLogger, CSVLogger

from src.training.callbacks import (
    DataFractionCheckpoint,
    ModelEvalCallback,
)
from src.config import DatasetPaths, PertGenConfig, load_config
from src.training.utils import (
    _build_run_name,
    hydrate_total_training_steps,
    load_weights,
    print_training_run_summary,
    resolve_start,
    tee_terminal_log,
)
from src.data import PertGenDataModule
from src.model.lightning_module import PertGenLightningModule

from pytorch_lightning.profilers import PyTorchProfiler


# Unlock Tensor Cores on NVIDIA GPUs for float32 matmuls
torch.set_float32_matmul_precision("high")

# Allow PertGenConfig to be unpickled from checkpoints (PyTorch 2.6+ defaults to weights_only=True)
torch.serialization.add_safe_globals([PertGenConfig, DatasetPaths])


def run(workdir: Path, config: PertGenConfig) -> None:
    """Run training with the given workdir and config. Used by both CLI and Hydra entry points."""
    workdir.mkdir(parents=True, exist_ok=True)

    config, ckpt_path_for_fit = resolve_start(config)

    resolved_path = workdir / "config_resolved.yaml"
    OmegaConf.save(OmegaConf.create(config.__dict__), str(resolved_path))
    print(f"  Resolved config saved to {resolved_path}")


    num_workers = int(getattr(config, "num_workers", 0))

    # Distributed loader safety is decided in PertGenDataModule.train_dataloader(),
    # which is the only place that knows whether the backend forks or spawns: the
    # cache/parquet backends spawn (CUDA-safe, workers kept) while h5ad forks
    # (workers dropped). Zeroing it here would also penalise the spawn-safe path.
    config.num_workers = num_workers

    # --- Data ---
    print("  Data: setting up PertGenDataModule (load data, split, encoders, HVG)...", flush=True)

    dm = PertGenDataModule(
        config=config,
        workdir=workdir
    )
    from src.eval.orchestration.artifacts import maybe_build_cache
    maybe_build_cache(config, rank0=True)

    dm.setup()

    print_training_run_summary(
        workdir=workdir,
        config=config,
        dm=dm,
        ckpt_path_for_fit=ckpt_path_for_fit,
    )

    print("  Model: initializing PertGenLightningModule (backbone=condition_mlp)...", flush=True)
    pl.seed_everything(config.seed)
    t_model = time.monotonic()
    model = PertGenLightningModule(config)
    load_weights(model, config)

    compile_on = bool(getattr(config, "compile_model", False))
    compile_msg = (
        f"torch.compile(mode={getattr(config, 'compile_mode', 'reduce-overhead')})"
        if compile_on
        else "eager"
    )
    print(
        f"  Model: initialized ({compile_msg}) [{time.monotonic() - t_model:.1f}s]",
        flush=True,
    )

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,} total, {n_trainable:,} trainable")
    print("=" * 72)

    loggers = [CSVLogger(save_dir=str(workdir), name="csv_logs")]
    
    if config.use_wandb:
        # Respect explicit name from yaml/CLI; otherwise auto-generate with timestamp
        if not getattr(config, "wandb_run_name", None):
            timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
            config.wandb_run_name = f"{_build_run_name(config)}_{timestamp}"

        print(f"  WandB run name: {config.wandb_run_name}")

        wandb_id_path = workdir / "wandb_run_id.txt"
        wandb_id = None
        if ckpt_path_for_fit and wandb_id_path.exists():
            wandb_id = wandb_id_path.read_text().strip()

        wandb_logger = WandbLogger(
            project=config.wandb_project,
            entity=config.wandb_entity or None,
            name=config.wandb_run_name,   # ✅ consistent, timestamped
            id=wandb_id,
            resume="allow",
            save_dir=str(workdir),
            config=config.__dict__,
        )
        (workdir / "wandb_run_id.txt").write_text(wandb_logger.experiment.id)

        loggers.append(wandb_logger)   

    ckpt_dir = workdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    data_frac = getattr(config, "checkpoint_data_fraction", 0.1)
    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            save_last=True,
            save_top_k=0,
            every_n_epochs=1,
        ),
        DataFractionCheckpoint(
            save_fraction=data_frac,
            total_train_samples=len(dm.train_indices),
            dirpath=str(ckpt_dir),
        ),
        LearningRateMonitor(logging_interval="step"),
        # Rich uses live terminal rendering; with pipes (`tee`, `> file`) stdout is
        # not a TTY so iteration rate is missing. TQDM prints plain lines in logs.
        (
            RichProgressBar()
            if sys.stdout.isatty()
            else TQDMProgressBar(refresh_rate=10)
        ),
    ]
    if str(getattr(config, "train_loader_mode", "per_cell")).strip().lower() == "cell_set":
        from src.data.ablations.cell_set_dataloader import CellSetEpochCallback

        callbacks.append(CellSetEpochCallback())
    
    if config.use_early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor=config.early_stopping_monitor,
                patience=config.early_stopping_patience + 1,
                mode=config.early_stopping_mode,
                check_on_train_epoch_end=True,
            ),
        )

    if config.use_eval_callback:
        callbacks.append(
            ModelEvalCallback(
                workdir=workdir,
                every_n_epochs=config.eval_callback_every_n_epochs,
                splits=config.model_eval_splits,
            ),
        )

    use_profiler = bool(getattr(config, "use_profiler", False))
    profiler_skip = int(getattr(config, "profiler_skip_steps", 10))
    profiler_warmup = int(getattr(config, "profiler_warmup_steps", 1))
    profiler_active = int(getattr(config, "profiler_steps", 200))
    profiler_wait = 1
    profiler_total_steps = profiler_skip + profiler_wait + profiler_warmup + profiler_active

    limit_train_batches = getattr(config, "limit_train_batches", None)
    if use_profiler and limit_train_batches is None:
        limit_train_batches = profiler_total_steps
    elif (
        use_profiler
        and limit_train_batches is not None
        and isinstance(limit_train_batches, int)
        and limit_train_batches < profiler_total_steps
    ):
        print(
            f"  WARNING: limit_train_batches={limit_train_batches} < "
            f"profiler schedule length {profiler_total_steps}; "
            f"active profiling may be truncated",
            flush=True,
        )

    if use_profiler:
        perf_logs_dir = workdir / "perf_logs"
        perf_logs_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"  Profiler: enabled — {profiler_active} active steps "
            f"(skip={profiler_skip}, wait={profiler_wait}, warmup={profiler_warmup}); "
            f"train capped at {limit_train_batches} batches; "
            f"traces -> {perf_logs_dir} "
            f"(TensorBoard + Chrome; open chrome://tracing and load *.json)",
            flush=True,
        )
        profiler = PyTorchProfiler(
            dirpath=str(perf_logs_dir),
            filename="profiler_trace",
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(perf_logs_dir)),
            schedule=torch.profiler.schedule(
                skip_first=profiler_skip,
                wait=profiler_wait,
                warmup=profiler_warmup,
                active=profiler_active,
                repeat=1,
            ),
            record_shapes=True,
            with_stack=True,
            export_to_chrome=True,
        )
    else:
        profiler = None

    accumulate = getattr(config, "accumulate_grad_batches", 1)
    accelerator = getattr(config, "accelerator", "auto")

    # Never rebuild the train DataLoader between epochs. persistent_workers=True
    # keeps spawn workers alive; ModelEvalCallback is the only path that
    # SIGKILLs them (to free RAM for inline eval). With use_eval_callback=false
    # that callback is not registered, so this must stay 0 or every epoch
    # respawns workers the way PerturbDiff's train_dataloader() does.
    reload_dl = 0

    steps_line = hydrate_total_training_steps(
        config, dm,
        devices=1, num_nodes=1,
        limit_train_batches=limit_train_batches,
        accumulate_grad_batches=accumulate,
    )
    print(f"  Train: {steps_line}", flush=True)

    trainer = pl.Trainer(
        default_root_dir=str(workdir),
        max_epochs=config.max_epochs,
        devices=1,
        accelerator=accelerator,
        reload_dataloaders_every_n_epochs=reload_dl,
        limit_train_batches=limit_train_batches if limit_train_batches is not None else 1.0,
        accumulate_grad_batches=accumulate,
        gradient_clip_val=config.grad_clip if config.grad_clip > 0 else None,
        log_every_n_steps=config.log_every_n_steps,
        logger=loggers,
        callbacks=callbacks,
        precision=config.precision if config.precision else "bf16-mixed",
        deterministic=False,
        enable_checkpointing=True,
        profiler=profiler,
    )

    print(
        f"  Train: starting trainer.fit "
        f"(max_epochs={config.max_epochs}, batch_size={dm.batch_size}, "
        f"n_train={len(dm.train_indices):,})...",
        flush=True,
    )
    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path_for_fit)

    if config.run_model_eval_after_fit:
        from src.eval.orchestration.artifacts import (
            compute_noise_ceiling,
            maybe_build_eval_cache,
            resolve_noise_ceiling_json_path,
        )
        from src.data.cache.eval_cache_build import DEFAULT_EVAL_SPLITS
        from src.eval.plots import plot_model_vs_ceiling_across_conditions
        from src.eval.orchestration.model_eval import ModelEvalResult, run_model_eval

        splits = list(config.model_eval_after_fit_splits or DEFAULT_EVAL_SPLITS)

        # 1) ensure per-condition NPZ eval cache under eval_cache/
        print("  [eval] (1/4) eval NPZ cache...", flush=True)
        cache_review = maybe_build_eval_cache(config, dm)
        build_note = "built missing files" if cache_review["did_build"] else "no build needed"
        print(
            f"  [eval] eval NPZ cache | {cache_review['n_existing']}/{cache_review['n_total']} "
            f"present ({cache_review['n_to_build']} missing) "
            f"splits={cache_review['eval_splits']} | {build_note}",
            flush=True,
        )

        # 2) model forward + metrics
        print(f"  [eval] (2/4) model eval | splits={splits}...", flush=True)
        eval_results: dict[str, ModelEvalResult] = {}
        for split in splits:
            eval_results[split] = run_model_eval(
                config=config,
                manifest=dm.eval_manifest,
                eval_split=split,
                run_dir=workdir,
                out_dir=workdir / "eval" / "after_fit" / split,
                device=config.model_eval_device,
                batch_size=config.model_eval_batch_size,
                seed=config.seed,
                min_cells=config.eval_min_cells,
                overwrite=config.overwrite_model_eval,
                pipeline_depth=config.model_eval_pipeline_depth,
                metrics_workers=config.model_eval_metrics_workers,
                use_compile=config.model_eval_compile,
                persist_predictions=config.model_eval_persist_predictions,
            )

        # 3) noise ceiling on the same eval cache
        nc_paths: dict[str, Path] = {}
        if config.build_noise_ceiling:
            print("  [eval] (3/4) noise ceiling...", flush=True)
            nc_paths = compute_noise_ceiling(config, dm._gene_names, dm.eval_manifest)
            if nc_paths:
                print(f"  [eval] noise ceiling splits: {list(nc_paths.keys())}", flush=True)
        else:
            print("  [eval] (3/4) noise ceiling skipped (build_noise_ceiling=false)", flush=True)

        # 4) cross-condition model vs ceiling plots (needs steps 2 + 3 outputs)
        if config.plot_model_vs_ceiling_after_fit:
            print("  [eval] (4/4) model vs ceiling plots...", flush=True)
            for split in splits:
                ceiling_path = nc_paths.get(split) or resolve_noise_ceiling_json_path(
                    config,
                    dm._gene_names,
                    split,
                    normalization_id=dm.eval_manifest.normalization_id,
                )
                if not ceiling_path.is_file():
                    print(
                        f"  [eval] model vs ceiling plots skipped: no ceiling at {ceiling_path}",
                        flush=True,
                    )
                    continue
                plot_dir = workdir / "eval" / "model_vs_ceiling" / split
                from src.data.data_utils import resolve_plot_aggregate_drop

                plot_drop = resolve_plot_aggregate_drop(
                    tuple(dm.model_conditioning_variables),
                    config.plot_aggregate_over,
                )
                saved = plot_model_vs_ceiling_across_conditions(
                    ceiling_path,
                    [eval_results[split].metrics_path],
                    out_dir=plot_dir,
                    drop_variables=plot_drop,
                )
                print(
                    f"  [eval] model vs ceiling plots | split={split} | "
                    f"{len(saved)} files -> {plot_dir}",
                    flush=True,
                )
        else:
            print(
                "  [eval] (4/4) model vs ceiling plots skipped "
                "(plot_model_vs_ceiling_after_fit=false)",
                flush=True,
            )

        print(f"  [eval] finished splits: {list(eval_results.keys())}", flush=True)


def main() -> None:
    """CLI entry point: --config (required), an optional --workdir; remaining args are OmegaConf overrides."""
    import argparse

    p = argparse.ArgumentParser(description="Train PertGen on Tahoe scRNA-seq data")
    p.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Run parent under repo root (default: runs/<dataset>/<split>/<model>/<run_name>)",
    )
    p.add_argument(
        "--workdir",
        type=str,
        default=None,
        help="Override full workdir (default: <PertGen>/<runs-dir>/<wandb_run_name>)",
    )
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="the run's YAML config, e.g. one under configs/ (CLI key=value overrides are applied on top)",
    )
    args, overrides = p.parse_known_args()
    
    config: PertGenConfig = load_config(
        config_path=Path(args.config) if args.config else None,
        overrides=overrides,
    )
    if not getattr(config, "wandb_run_name", None):
        if getattr(config, "split_config", None) and Path(config.split_config).is_file():
            import yaml
            with open(Path(config.split_config), "r") as f:
                sc = yaml.safe_load(f)
                train_sc = sc.get("train", {})
                cond_sc = train_sc.get("conditions", train_sc)
                if "fraction" in cond_sc and cond_sc.get("sample") != "all":
                    config._subsample_train_fraction = cond_sc.get("fraction")
                    config._subsample_train_type = cond_sc.get("sample")
                cells_sc = train_sc.get("cells") or {}
                cells_target_n = cells_sc.get("target_n")
                if cells_target_n is not None:
                    config._train_cells_target_n = int(cells_target_n)

        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        config.wandb_run_name = f"{timestamp}_{_build_run_name(config)}"

    from src.config.config import resolve_workdir

    workdir = (
        Path(args.workdir).expanduser().resolve()
        if args.workdir
        else resolve_workdir(
            config,
            run_name=config.wandb_run_name,
            fallback_runs_dir=args.runs_dir,
            config_path=args.config,
        )
    )
    log_path = workdir / "run.log"
    print(f"  workdir: {workdir}")
    print(f"  terminal log: {log_path}", flush=True)
    with tee_terminal_log(log_path):
        run(workdir, config)

if __name__ == "__main__":
    main()
