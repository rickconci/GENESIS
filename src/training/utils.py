from __future__ import annotations

import math
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import IO, TYPE_CHECKING, Iterator, TextIO, Union

import torch
from src.config.config import (
    PertGenConfig,
    blank_cache_paths_to_none,
    sanitize_config_mapping,
)
from src.config.loader import (
    _hydrate_dataset_paths,
    _hydrate_inline_block_from_registry,
    resolve_config_paths,
)

if TYPE_CHECKING:
    from src.data.datamodule import PertGenDataModule

def _optimizer_steps_per_epoch(
    n_units: int,
    batch_size: int,
    *,
    world_size: int,
    limit_train_batches: Union[int, float, bool, None],
    accumulate_grad_batches: int,
) -> int:
    """Optimizer steps in one epoch for a fixed ``batch_size``."""
    per_epoch = (n_units // max(1, world_size)) // max(1, batch_size)
    ltb = 1.0 if limit_train_batches is None else limit_train_batches
    if type(ltb) is bool:
        pass
    elif type(ltb) is int:
        per_epoch = min(per_epoch, ltb)
    else:
        per_epoch = max(0, int(per_epoch * float(ltb)))
    return math.ceil(per_epoch / max(1, accumulate_grad_batches))


def _estimate_total_training_steps(
    config: PertGenConfig,
    n_units: int,
    *,
    world_size: int,
    limit_train_batches: Union[int, float, bool, None],
    accumulate_grad_batches: int,
    default_batch_size: int,
) -> tuple[int, str]:
    """Return (total optimizer steps, detail suffix for logging)."""
    max_epochs = max(int(config.max_epochs), 1)
    batch_size = max(1, int(default_batch_size))
    per_epoch = _optimizer_steps_per_epoch(
        n_units,
        batch_size,
        world_size=world_size,
        limit_train_batches=limit_train_batches,
        accumulate_grad_batches=accumulate_grad_batches,
    )
    return per_epoch * max_epochs, f"per_epoch={per_epoch:,}"


def hydrate_total_training_steps(
    config: PertGenConfig,
    dm: PertGenDataModule,
    *,
    devices: object,
    num_nodes: int,
    limit_train_batches: Union[int, float, bool, None],
    accumulate_grad_batches: int,
) -> str:
    """Set ``config.total_training_steps`` for WSD LambdaLR without building a DataLoader.

    ``trainer.estimated_stepping_batches`` calls ``fit_loop.setup_data()`` during
    ``configure_optimizers`` because PyTorch's ``LambdaLR`` invokes ``lr_lambda(0)``
    at construction. Closed-form math here keeps the schedule exact and lets
    dataloader errors surface at ``fit``.
    """
    n_devices = (
        devices
        if isinstance(devices, int)
        else len(devices)
        if isinstance(devices, (list, tuple))
        else max(1, torch.cuda.device_count())
        if torch.cuda.is_available()
        else 1
    )
    world_size = int(n_devices) * int(num_nodes)

    mode = str(getattr(config, "train_loader_mode", "per_cell")).strip().lower()
    if mode == "cell_set":
        from src.data.sampling.eval_cache_cell_sets import load_cell_set_conditions

        n_units = len(load_cell_set_conditions(dm, debug=False)[1])
    else:
        n_units = len(dm.train_indices)

    total_steps, step_detail = _estimate_total_training_steps(
        config,
        n_units,
        world_size=world_size,
        limit_train_batches=limit_train_batches,
        accumulate_grad_batches=accumulate_grad_batches,
        default_batch_size=int(dm.batch_size),
    )
    config.total_training_steps = total_steps
    return (
        f"total_training_steps={config.total_training_steps:,} "
        f"(mode={mode}, {step_detail}, accumulate={accumulate_grad_batches}, "
        f"max_epochs={config.max_epochs}, world_size={world_size})"
    )


def _build_run_name(config: "PertGenConfig") -> str:
    """WandB name for a condition-MLP run: batch, lr, split stem, width, loss, seed."""

    parts = []

    bs = getattr(config, "batch_size", "?")
    parts.append(f"bs{bs}")

    lr_val = getattr(config, "lr", None)
    if lr_val is not None:
        parts.append(f"lr{float(lr_val):g}")

    try:
        wd = float(getattr(config, "weight_decay", 0.0))
    except (TypeError, ValueError):
        wd = 0.0
    parts.append(f"wd{wd:g}")

    # --- 2. Split tag (derived from split_config path basename) ---
    # Examples:
    #   core_util_data/splits/tahoe/bio/tahoe_legacy_bio_holdout.yaml           -> biohold
    # `tahoe_legacy_perturbation_holdout` is another recorded name of the same split. It
    # holds out 5 CELL LINES crossed with drug lists -- a bio-axis split -- so both
    # names map to `biohold`.
    _SPLIT_ABBREV = {
        "tahoe_legacy_bio_holdout":           "biohold",
        "tahoe_legacy_perturbation_holdout":  "biohold",
        "tahoe_organ_scoped_lbp":             "organlbp",
        "pbmc_default":                       "pbmc",
    }
    sc = getattr(config, "split_config", None) or ""
    stem = Path(str(sc)).stem if sc else ""
    parts.append(_SPLIT_ABBREV.get(stem, stem[:12] if stem else "nosplit"))

    hidden = "x".join(str(int(dim)) for dim in getattr(config, "condition_mlp_hidden_dims", [2048, 3072]))
    parts.append(f"condmlp-D{config.condition_mlp_tokdim}-h{hidden}")

    cmode_str = "cond"
    if "plate" in getattr(config, "model_conditioning_variables", []):
        cmode_str += "-plate"
    parts.append(cmode_str)
    parts.append("mlpout-relu")  # kept so names match the published run dirs

    # --- 6. Variogram loss (score term) ---
    vw = float(getattr(config, "variogram_loss_weight", 0.0))
    if vw > 0.0:
        vp = float(getattr(config, "variogram_p", 0.5))
        tag = f"vg{vw:g}p{vp:g}"
        npairs = int(getattr(config, "variogram_n_pairs", 0))
        if npairs <= 0:
            tag += "npall"
        elif npairs % 1000 == 0 and npairs >= 1000:
            tag += f"np{npairs // 1000}k"
        else:
            tag += f"np{npairs}"
        parts.append(tag)
    else:
        parts.append("vg0")

    # --- 8. Target space (always log1p) ---
    parts.append("log1p")

    # --- 9. Subsample (optional, sourced from split YAML) ---
    subsample_fraction = getattr(config, "_subsample_train_fraction", None)
    subsample_type = getattr(config, "_subsample_train_type", "sub")
    if subsample_fraction:
        try:
            xxx = int(round(float(subsample_fraction) * 100))
            parts.append(f"{subsample_type}f{xxx:03d}")
        except (TypeError, ValueError):
            parts.append(str(subsample_type))

    cells_target_n = getattr(config, "_train_cells_target_n", None)
    if cells_target_n is not None:
        parts.append(f"celln{int(cells_target_n)}")

    # --- 10. Training seed (disambiguates parallel sweep jobs with identical config) ---
    seed = getattr(config, "seed", None)
    if seed is not None:
        parts.append(f"s{int(seed)}")

    # --- 11. Custom Run Name Suffix ---
    run_name_suffix = getattr(config, "run_name_suffix", None)
    if run_name_suffix:
        parts.append(str(run_name_suffix))

    return "_".join(parts)


class _TeeStream:
    """Mirror writes to a terminal stream and a log file (line-buffered flush)."""

    def __init__(self, stream: TextIO, log_file: IO[str]) -> None:
        self._stream = stream
        self._log_file = log_file

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._log_file.write(data)
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._log_file.flush()

    def fileno(self) -> int:
        return self._stream.fileno()

    def isatty(self) -> bool:
        return self._stream.isatty()


@contextmanager
def tee_terminal_log(log_path: Path) -> Iterator[Path]:
    """Duplicate stdout/stderr to ``log_path`` while keeping terminal output."""
    log_path = log_path.expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "a", encoding="utf-8", buffering=1)
    log_f.write(f"\n{'=' * 72}\n")
    log_f.write(f"PertGen run log started {datetime.now().isoformat()}\n")
    log_f.write(f"{'=' * 72}\n")
    log_f.flush()

    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(old_out, log_f)
    sys.stderr = _TeeStream(old_err, log_f)
    try:
        yield log_path
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        log_f.flush()
        log_f.close()


def resolve_run_checkpoint(
    run_dir: Union[str, Path],
    checkpoint: Union[str, Path, None] = None,
) -> Path:
    """Resolve a checkpoint under a training run directory.

    * ``None``, ``\"last\"``, or ``\"last.ckpt\"``: prefer ``checkpoints/last.ckpt``,
      else the newest ``*.ckpt`` by mtime in ``<run_dir>/checkpoints/``.
    * Path to an existing file: returned resolved.
    * Bare filename: looked up under ``<run_dir>/checkpoints/``.
    """
    run = Path(run_dir).resolve()
    ckpt_dir = run / "checkpoints"
    spec = "" if checkpoint is None else str(checkpoint).strip()

    if not spec or spec.lower() in ("last", "last.ckpt"):
        if not ckpt_dir.is_dir():
            raise FileNotFoundError(f"Checkpoint directory missing: {ckpt_dir} (run {run})")
        last = ckpt_dir / "last.ckpt"
        if last.is_file():
            return last.resolve()
        ckpts = list(ckpt_dir.glob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No .ckpt files in {ckpt_dir}")
        return max(ckpts, key=lambda p: p.stat().st_mtime).resolve()

    path = Path(spec)
    if path.is_file():
        return path.resolve()

    search = [path] if path.is_absolute() else [
        Path.cwd() / path, ckpt_dir / path.name, ckpt_dir / path, run / path,
    ]
    for candidate in search:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(f"Checkpoint not found for {checkpoint!r} under run {run}")


def _config_from_checkpoint(ckpt_path: Path) -> PertGenConfig:
    """Load PertGen config from a Lightning checkpoint (hyper_parameters.config)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    config = hparams.get("config")
    if config is None:
        raise ValueError(f"Checkpoint {ckpt_path} has no 'config' in hyper_parameters.")
    if isinstance(config, dict):
        cfg = PertGenConfig(**sanitize_config_mapping(config))
    elif isinstance(config, PertGenConfig):
        # vars(), not fields(): stale embedded configs may lack fields added after
        # training (asdict() fails) and carry removed ones sanitize must still check.
        cfg = PertGenConfig(**sanitize_config_mapping(vars(config)))
    else:
        raise TypeError(
            f"Unsupported hyper_parameters['config'] type {type(config).__name__}; "
            "expected dict or PertGenConfig."
        )
    _hydrate_inline_block_from_registry(cfg)
    _hydrate_dataset_paths(cfg)
    resolve_config_paths(cfg)
    blank_cache_paths_to_none(cfg)
    return cfg


def resolve_start(config: PertGenConfig) -> tuple[PertGenConfig, str | None]:
    """Choose how a checkpoint enters training. Return ``(config, resume_path)``.

    ``resume_ckpt`` continues the saved run. The returned config is the one
    stored in the checkpoint, and ``resume_path`` is passed to
    ``trainer.fit(ckpt_path=...)``, which restores weights, optimizer, and epoch.

    ``weights_ckpt`` starts a new run from the launch config. The path stays
    on ``config``; :func:`load_weights` copies the ``state_dict`` once the
    module exists. ``resume_path`` is ``None``, so ``fit`` builds a fresh
    optimizer at epoch 0.

    The two cannot be combined: a resume already contains those weights.
    """
    resume, weights = config.resume_ckpt, config.weights_ckpt
    if resume and weights:
        raise SystemExit("resume_ckpt and weights_ckpt are mutually exclusive")
    if not resume:
        return config, None
    path = Path(resume).resolve()
    if not path.is_file():
        raise SystemExit(f"resume_ckpt not found: {path}")
    print(f"  Resuming from checkpoint: {path}")
    loaded = _config_from_checkpoint(path)
    # The state_dict is restored by fit(); copying it again would be a second load.
    loaded.weights_ckpt = None
    loaded.resume_ckpt = str(path)
    return loaded, str(path)


def load_weights(model: torch.nn.Module, config: PertGenConfig) -> None:
    """Copy ``state_dict`` from ``config.weights_ckpt``. Optimizer and epoch stay fresh."""
    weights_ckpt = config.weights_ckpt
    if not weights_ckpt:
        return
    weights_path = Path(weights_ckpt).resolve()
    if not weights_path.is_file():
        raise SystemExit(f"weights_ckpt not found: {weights_path}")
    blob = torch.load(weights_path, map_location="cpu", weights_only=False)
    sd = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    model.load_state_dict(sd, strict=True)
    print(f"  Weights: loaded from weights_ckpt={weights_path}", flush=True)
    if config.head_bias_init_from_train_mean:
        # Would overwrite the loaded head.bias on on_fit_start.
        config.head_bias_init_from_train_mean = False
        print("  Weights: forcing head_bias_init_from_train_mean=false (weights_ckpt already has a trained head)",
              flush=True)


def _kv(label: str, value: object, *, w: int = 22) -> str:
    """Aligned ``label: value`` line for the training banner."""
    return f"  {label + ':':<{w}} {value}"


def _architecture_kv_rows(config: PertGenConfig) -> list[tuple[str, object]]:
    """Banner rows for the condition MLP the published configs train."""
    rows: list[tuple[str, object]] = [
        ("backbone_type", str(getattr(config, "backbone_type", "condition_mlp"))),
        ("condition_mlp_tokdim", config.condition_mlp_tokdim),
        ("condition_mlp_hidden_dims", list(getattr(config, "condition_mlp_hidden_dims", []))),
    ]
    return rows


def _batch_lines(dm: "PertGenDataModule") -> list[str]:
    """``batch_size`` line. ``n_repeats`` is fixed and printed with the model rows."""
    return [_kv("batch_size", dm.batch_size)]


def print_training_run_summary(
    *,
    workdir: Path,
    config: PertGenConfig,
    dm: "PertGenDataModule",
    ckpt_path_for_fit: str | None,
) -> None:
    """Emit the rank-0 training banner (resolved paths, arch, split, hparams).

    Args:
        workdir: Experiment output directory.
        config: Fully merged training config (post-``setup`` data-dependent fields).
        dm: Data module after :meth:`~PertGenDataModule.setup`.
        ckpt_path_for_fit: Checkpoint path when resuming training, else ``None``.
    """
    split_path = Path(str(config.split_config)).resolve()
    eff_dir = Path(str(config.data_dir)).resolve()
    dd = str(eff_dir)
    use_hvg = config.hvg_genes_path is not None

    from src.data.splits.data_class import SplitConfig

    split_cfg = getattr(dm, "split_config", None)
    if not isinstance(split_cfg, SplitConfig):
        raise RuntimeError(
            "DataModule must expose split_config as SplitConfig after setup(); "
            f"got {type(split_cfg).__name__!r}."
        )

    def _predicate_or_len(block: object, *, label: str) -> int:
        if isinstance(block, list):
            return len(block)
        if isinstance(block, dict):
            return len(block)
        raise TypeError(
            f"split.{label} must be a dict or list of dicts, got {type(block).__name__!r}."
        )

    train_spec = split_cfg.train
    ts = train_spec.sample
    if ts == "hbss" and train_spec.fraction is not None:
        ts = f"{ts}  fraction={train_spec.fraction}"
    split_dataset = split_cfg.dataset
    controls_to_train = split_cfg.controls_to_train
    n_test_or = _predicate_or_len(split_cfg.test, label="test")
    n_val_or = _predicate_or_len(split_cfg.validation, label="validation")

    nt, nv, ns = len(dm.train_indices), len(dm.val_indices), len(dm.test_indices)

    lines: list[str] = [
        "=" * 72,
        "PertGen Training",
        "=" * 72,
        _kv("workdir", workdir),
        _kv("data_dir", dd),
        _kv("strategy", config.strategy),
        _kv("devices", config.devices),
        *(_kv(k, v) for k, v in _architecture_kv_rows(config)),
        _kv("num_output_genes", config.num_output_genes),
    ]
    if getattr(config, "n_categories", None):
        for k, v in config.n_categories.items():
            lines.append(_kv(f"n_{k}", v))
    lines += [
        _kv("n_repeats", config.n_repeats),
        *_batch_lines(dm),
        _kv("lr", config.lr),
        _kv("max_epochs", config.max_epochs),
        _kv("seed", config.seed),
    ]
    if ckpt_path_for_fit:
        lines.append(_kv("resume_ckpt", ckpt_path_for_fit))
    lines += [
        _kv("split_config", split_path),
        _kv("split.dataset", split_dataset),
        _kv(
            "split.predicates",
            f"test_OR={n_test_or}  val_OR={n_val_or}  controls_to_train={controls_to_train}",
        ),
        _kv("split.train", ts),
        _kv("split.indices", f"train={nt:,}  val={nv:,}  test={ns:,}  sum={nt + nv + ns:,}"),
        "=" * 72,
    ]
    print("\n".join(lines), flush=True)
