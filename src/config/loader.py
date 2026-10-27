"""Load and merge config from YAML + CLI overrides using OmegaConf."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from omegaconf import OmegaConf

from .config import (
    DatasetPaths,
    PertGenConfig,
    blank_cache_paths_to_none,
    epoch_lr_milestones,
    migrate_legacy_cache_keys,
    resolve_condition_input_variables,
    sanitize_config_mapping,
)
from .dataset_registry import (
    DatasetRegistry,
    load_dataset_registry,
    resolve_registry_key,
)


def resolve_config_paths(cfg: PertGenConfig) -> None:
    """Expand relative paths against the repo root (absolute left unchanged)."""
    from src.paths import PERTGEN_REPO_ROOT

    repo_root = PERTGEN_REPO_ROOT

    def _join(name: str, base: Path) -> None:
        val = getattr(cfg, name, None)
        if not isinstance(val, str):
            return
        p = Path(val).expanduser()
        setattr(cfg, name, val if p.is_absolute() else str((base / p).resolve()))

    for n in ("data_dir", "hvg_genes_path", "split_config"):
        _join(n, repo_root)

    joins = getattr(cfg, "metadata_joins", None) or []
    if joins:
        resolved: List[Any] = []
        for entry in joins:
            if not isinstance(entry, dict):
                resolved.append(entry)
                continue
            item = dict(entry)
            parquet = item.get("parquet")
            if isinstance(parquet, str) and parquet.strip():
                p = Path(parquet).expanduser()
                item["parquet"] = str(p if p.is_absolute() else (repo_root / p).resolve())
            resolved.append(item)
        cfg.metadata_joins = resolved


_REGISTRY_TO_DATASETPATHS_FIELDS = (
    "data_dir",
    "cache_dir",
    "eval_cache_root",
    "eval_cache_dir",
    "noise_ceiling_dir",
    "files_glob",
    "model_conditioning_variables",
    "control_condition",
    "control_match_keys",
    "metadata_joins",
    "plot_aggregate_over",
    "build_cache",
    "overwrite_cache",
    "build_eval_cache",
    "overwrite_eval_cache",
    "build_noise_ceiling",
    "noise_ceiling_n_repeats",
    "overwrite_noise_ceiling",
)


def _is_unset_dataset_paths_value(value: Any) -> bool:
    """True when an inline preset field should be filled from the registry."""
    return value is None or (isinstance(value, (list, str)) and len(value) == 0)


def _hydrate_inline_block_from_registry(cfg: PertGenConfig) -> None:
    """Populate the inline ``cfg.<dataset>`` block from the registry YAML when missing.

    Allows ``configs/<run>.yaml`` to set only ``dataset: tahoe`` (or a slim
    ``tahoe:`` block with overrides). ``core_util_data/datasets/<name>.yaml``
    is SSOT for dataset semantics and run defaults. Inline blocks always win —
    registry only fills *unset* fields.
    """
    raw = getattr(cfg, "dataset", None)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return
    alias = str(raw).strip()
    try:
        registry_key = resolve_registry_key(alias)
        registry: DatasetRegistry = load_dataset_registry(alias)
    except (FileNotFoundError, ValueError):
        # No registry entry for this alias → leave inline-block path alone.
        return

    block = getattr(cfg, alias, None)
    if block is None:
        block = DatasetPaths()
        setattr(cfg, alias, block)

    projected = registry.to_dataset_paths_defaults()
    for name in _REGISTRY_TO_DATASETPATHS_FIELDS:
        cur = getattr(block, name, None)
        if _is_unset_dataset_paths_value(cur) and name in projected:
            setattr(block, name, projected[name])

    if not getattr(block, "registry_key", None):
        block.registry_key = registry_key


def _hydrate_dataset_paths(cfg: PertGenConfig) -> None:
    """If ``dataset`` names a preset, copy unset paths from that block."""
    raw = getattr(cfg, "dataset", None)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return
    key = str(raw).strip()
    block = getattr(cfg, key, None)
    if block is None:
        raise ValueError(
            f"dataset={key!r} but no matching preset field "
            f"(add '{key}:' block or rely on registry hydration)."
        )
    if not isinstance(block, DatasetPaths):
        raise TypeError(
            f"Preset {key!r} must be DatasetPaths (got {type(block).__name__})."
        )
    _if_unset = (
        "data_dir",
        "hvg_genes_path",
        "cache_dir",
        "eval_cache_root",
        "eval_cache_dir",
        "noise_ceiling_dir",
        "split_config",
        "files_glob",
        "model_conditioning_variables",
        "control_condition",
        "metadata_joins",
        "control_match_keys",
        "plot_aggregate_over",
    )
    _from_preset = (
        "build_cache",
        "build_eval_cache",
        "build_noise_ceiling",
        "noise_ceiling_n_repeats",
        "overwrite_cache",
        "overwrite_eval_cache",
        "overwrite_noise_ceiling",
    )

    for name in _from_preset:
        pv = getattr(block, name, None)
        if pv is not None:
            setattr(cfg, name, pv)

    for name in _if_unset:
        pv = getattr(block, name, None)
        if pv is None:
            continue
        cur = getattr(cfg, name, None)
        # The dataclass default ``*.h5ad`` must not hide the registry's plate glob.
        if (
            name == "files_glob"
            and cur == "*.h5ad"
            and pv != "*.h5ad"
        ):
            setattr(cfg, name, pv)
        elif _is_unset_dataset_paths_value(cur):
            setattr(cfg, name, pv)


def load_config(
    config_path: Optional[Path] = None,
    overrides: Optional[List[str]] = None,
) -> PertGenConfig:
    """Dataclass defaults, then YAML, then OmegaConf CLI; hydrate presets, resolve paths, validate."""
    if config_path is None:
        from src.paths import PERTGEN_REPO_ROOT

        config_path = PERTGEN_REPO_ROOT / "configs" / "mlp_default.yaml"
    overrides = overrides or []

    defaults = OmegaConf.to_container(OmegaConf.structured(PertGenConfig), resolve=True)
    yaml_raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    cli_raw = OmegaConf.to_container(OmegaConf.from_cli(overrides), resolve=True)
    merged = OmegaConf.merge(
        OmegaConf.create(defaults),
        OmegaConf.create(yaml_raw),
        OmegaConf.create(cli_raw),
    )
    merged_dict = migrate_legacy_cache_keys(OmegaConf.to_container(merged, resolve=True))
    cfg = PertGenConfig(**sanitize_config_mapping(merged_dict))

    # Registry first (fills inline block from core_util_data/datasets/<name>.yaml
    # when missing), then existing inline-block hydration of top-level fields.
    _hydrate_inline_block_from_registry(cfg)
    _hydrate_dataset_paths(cfg)
    blank_cache_paths_to_none(cfg)

    ds = getattr(cfg, "dataset", None)
    if ds:
        if not getattr(cfg, "hvg_genes_path", None):
            raise ValueError(
                f"dataset={ds!r} but hvg_genes_path is unset; "
                f"set top-level hvg_genes_path in your model config YAML.",
            )
        # split_config=None is valid for whole-dataset eval-cache builds (no split filter).
        # Training / model-eval entrypoints validate split_config when they need it.

    if str(cfg.backbone_type) not in ("condition_mlp",):
        raise ValueError(
            f"backbone_type must be one of ('condition_mlp',), got {cfg.backbone_type!r}"
        )
    if cfg.devices not in (1, "1", "auto") or str(cfg.strategy) not in ("auto",) or int(cfg.num_nodes) != 1:
        raise ValueError(
            "devices/strategy/num_nodes must be one of (1, '1', 'auto') / ('auto',) / (1,), "
            f"got {cfg.devices!r} / {cfg.strategy!r} / {cfg.num_nodes!r}"
        )

    resolve_config_paths(cfg)
    if cfg.files_glob is None:
        cfg.files_glob = "*.h5ad"

    # Fail at load: input subset must be inside the eval-identity MCV, and the LR
    # schedule must not pin a per-epoch n_repeats / batch_size.
    resolve_condition_input_variables(cfg)
    epoch_lr_milestones(cfg)
    return cfg
