"""Dataset registry: ``core_util_data/datasets/<name>.yaml`` is SSOT for dataset semantics.

The registry decouples *what a dataset is* (which obs columns identify
biological context vs. perturbation, what counts as a control, where the
files live) from *how a particular run uses it* (training hyperparameters,
gene set, eval cache namespace).

Two consumers:

* :class:`DatasetRegistry` — raw, faithful representation of the YAML file.
  Read-only; fields mirror the YAML keys 1:1.
* :class:`DatasetConditionSpec` — derived view used by eval CLIs. Wraps a
  registry plus per-run overrides (e.g. an explicit eval cache namespace)
  and exposes helpers to build condition keys.

Aliases match training YAML ``dataset:`` keys: ``tahoe → tahoe``,
``parse → parse``. Legacy keys (``tahoe100M``, ``pbmc``) still resolve.
``split_config`` and ``hvg_genes_path`` live in model configs, not registry YAMLs.

This module imports nothing from :mod:`src.eval` or :mod:`src.data` — it
sits at the bottom of the dependency graph.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# Aliases and paths
# ---------------------------------------------------------------------------

# Stable mapping from training-YAML ``dataset:`` keys to registry filenames.
# Also includes identity entries so registry keys resolve to themselves.
DATASET_ALIAS_MAP: Dict[str, str] = {
    "tahoe": "tahoe",
    "parse": "parse",
    # Legacy registry filenames / training keys still resolve.
    "tahoe100M": "tahoe",
    "pbmc": "parse",
}

# Reverse: registry key → canonical training alias (used for ``results/<dataset>/`` paths).
_REGISTRY_TO_ALIAS: Dict[str, str] = {
    "tahoe": "tahoe",
    "parse": "parse",
    "tahoe100M": "tahoe",
    "pbmc": "parse",
}


def resolve_registry_key(alias_or_key: str) -> str:
    """Map a training alias (``tahoe``) or registry key (``tahoe100M``) to a registry key."""
    if not isinstance(alias_or_key, str) or not alias_or_key.strip():
        raise ValueError(f"dataset alias must be a non-empty string, got {alias_or_key!r}")
    key = alias_or_key.strip()
    return DATASET_ALIAS_MAP.get(key, key)


def resolve_dataset_alias(alias_or_key: str) -> str:
    """Inverse of :func:`resolve_registry_key`. Used for results paths (``results/tahoe/...``)."""
    rk = resolve_registry_key(alias_or_key)
    return _REGISTRY_TO_ALIAS.get(rk, rk)


def _registry_root() -> Path:
    """``<repo>/core_util_data/datasets``. Imported lazily to avoid early ``src.paths`` cost."""
    from src.paths import PERTGEN_REPO_ROOT

    return PERTGEN_REPO_ROOT / "core_util_data" / "datasets"


def registry_yaml_path(alias_or_key: str) -> Path:
    """Resolve ``<repo>/core_util_data/datasets/<registry_key>.yaml``. Existence not checked."""
    rk = resolve_registry_key(alias_or_key)
    return _registry_root() / f"{rk}.yaml"


# ---------------------------------------------------------------------------
# Raw registry YAML
# ---------------------------------------------------------------------------


def _as_str_tuple(value: Any) -> Tuple[str, ...]:
    """Normalise list-or-None YAML values to a tuple of strings (preserve order, drop blanks)."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(v) for v in value if v is not None and str(v) != "")


def _as_optional_str(value: Any) -> Optional[str]:
    """Trim and return None for empty/None inputs (loaders treat blank strings as unset)."""
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _as_optional_bool(value: Any) -> Optional[bool]:
    """Preserve explicit ``false``; treat blank strings as unset."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "1", "yes"):
            return True
        if s in ("false", "0", "no"):
            return False
        return None
    return bool(value)


def _as_optional_int(value: Any) -> Optional[int]:
    """Parse optional integer YAML scalars; blank → unset."""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return int(value)


@dataclass(frozen=True)
class DatasetRegistry:
    """Faithful representation of one ``core_util_data/datasets/<name>.yaml`` file.

    Field names mirror the YAML keys. Fields are populated from the YAML at
    load time and are never mutated downstream — :class:`DatasetConditionSpec`
    or :class:`PertGenConfig` carry per-run overrides separately.

    Attributes:
        name: Human-readable name (matches ``dataset:`` in split YAMLs).
        data_dir: Directory containing per-plate h5ads.
        plate_glob: Glob pattern for plate files inside ``data_dir``.
        biological_context_columns: Obs columns identifying bio context
            (e.g. ``("cell_name",)`` for Tahoe, ``("donor",)`` for Parse).
        perturbation_columns: Obs columns identifying perturbations
            (e.g. ``("drug", "dose")`` for Tahoe, ``("cytokine",)`` for Parse).
        plate_column: Optional batch column treated as plate; ``None`` when
            the dataset has no plate concept (Parse).
        conc_key: Optional concentration key for Tahoe-style drug+dose joins.
        experimental_replicate_column: Column whose values are independent technical
            replicates of the same condition (Tahoe: ``plate``). ``None`` when no
            condition is ever measured twice (Parse: one sample per donor × cytokine),
            so no experimental-replicate reference exists for the dataset.
        control_pert: Mapping of perturbation column → control value
            (e.g. ``{"drug": "DMSO_TF", "dose": 0.0}``).
        metadata_joins: Tuple of join descriptors enriching obs with extra
            columns (Organ, MOA, ...).
        control_match_keys: Optional override for μ_ctrl grouping. When None,
            consumers default to ``(plate_column,) + biological_context_columns``.
        model_conditioning_variables: Optional override for the eval-cache key.
            When None, derived as
            ``biological_context_columns + perturbation_columns + (plate_column,)``.
        hbss_subsample_dir: Optional directory of HBSS pre-computed parquets.
        hbss_filename_pattern: Filename template for HBSS parquets.
        cache_dir: Optional mmap CSR training-cache root.
        build_cache: Build training cache when absent.
        overwrite_cache: Force rebuild of mmap CSR shards.
        eval_cache_root: Root for eval NPZ namespaces.
        eval_cache_dir: Full eval NPZ namespace override.
        build_eval_cache: Build per-condition eval truth cache.
        overwrite_eval_cache: Force rebuild of eval NPZ cache.
        plot_aggregate_over: Cross-condition plot aggregation column.
        build_noise_ceiling: Run noise-ceiling pass after eval cache.
        noise_ceiling_n_repeats: Repeats per condition for noise ceiling.
        overwrite_noise_ceiling: Force rebuild of noise-ceiling JSON.
        noise_ceiling_dir: Optional noise-ceiling output root override.
        median_umi: Median full-transcriptome UMI per cell (L_ref of ``log1p_slice_med_rgs``).
    """

    name: str
    data_dir: str
    plate_glob: str
    biological_context_columns: Tuple[str, ...]
    perturbation_columns: Tuple[str, ...]
    plate_column: Optional[str]
    conc_key: Optional[str]
    control_pert: Mapping[str, Any]
    experimental_replicate_column: Optional[str] = None
    metadata_joins: Tuple[Mapping[str, Any], ...] = ()
    control_match_keys: Optional[Tuple[str, ...]] = None
    model_conditioning_variables: Optional[Tuple[str, ...]] = None
    hbss_subsample_dir: Optional[str] = None
    hbss_filename_pattern: Optional[str] = None
    cache_dir: Optional[str] = None
    build_cache: Optional[bool] = None
    overwrite_cache: Optional[bool] = None
    eval_cache_root: Optional[str] = None
    eval_cache_dir: Optional[str] = None
    build_eval_cache: Optional[bool] = None
    overwrite_eval_cache: Optional[bool] = None
    plot_aggregate_over: Optional[str] = None
    build_noise_ceiling: Optional[bool] = None
    noise_ceiling_n_repeats: Optional[int] = None
    overwrite_noise_ceiling: Optional[bool] = None
    noise_ceiling_dir: Optional[str] = None
    median_umi: Optional[float] = None
    source_path: Optional[Path] = None

    @classmethod
    def from_yaml(cls, path: Path) -> "DatasetRegistry":
        """Load and validate one registry YAML file.

        Raises:
            FileNotFoundError: ``path`` does not exist.
            ValueError: required keys are missing or have empty bio-context /
                perturbation column lists.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"dataset registry YAML not found: {path}")
        # Resolve OmegaConf interpolations early so downstream code sees plain
        # python types — never DictConfig nodes that hide their schema.
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: top-level must be a mapping, got {type(raw).__name__}")

        ctx = _as_str_tuple(raw.get("biological_context_columns"))
        pert = _as_str_tuple(raw.get("perturbation_columns"))
        if not ctx:
            raise ValueError(f"{path}: biological_context_columns is empty")
        if not pert:
            raise ValueError(f"{path}: perturbation_columns is empty")

        joins = raw.get("metadata_joins") or []
        if not isinstance(joins, list):
            raise ValueError(f"{path}: metadata_joins must be a list, got {type(joins).__name__}")
        joins_tuple = tuple(dict(j) if isinstance(j, dict) else j for j in joins)

        control_pert = raw.get("control_pert") or {}
        if not isinstance(control_pert, dict) or not control_pert:
            raise ValueError(f"{path}: control_pert is required and must be a non-empty mapping")

        name = str(raw.get("name") or path.stem)
        # PERTGEN_DATA_ROOT relocates every dataset here, once, so `{data:<ds>}` and direct
        # `.data_dir` readers agree. Not stat'ed: a wrong root must fail loudly at the first read.
        data_root = os.environ.get("PERTGEN_DATA_ROOT", "")
        data_dir = f"{data_root.rstrip('/')}/{name}" if data_root else str(raw.get("data_dir") or "")
        if "{root:" in data_dir:                   # the machine's location lives in configs/roots.yaml
            from src.paths import expand_path      # here, not at import: src.paths reads this registry
            data_dir = expand_path(data_dir)

        return cls(
            name=name,
            data_dir=data_dir,
            plate_glob=str(raw.get("plate_glob") or ""),
            biological_context_columns=ctx,
            perturbation_columns=pert,
            plate_column=_as_optional_str(raw.get("plate_column")),
            conc_key=_as_optional_str(raw.get("conc_key")),
            experimental_replicate_column=_as_optional_str(raw.get("experimental_replicate_column")),
            control_pert=dict(control_pert),
            metadata_joins=joins_tuple,
            control_match_keys=(
                _as_str_tuple(raw["control_match_keys"])
                if raw.get("control_match_keys") is not None
                else None
            ),
            model_conditioning_variables=(
                _as_str_tuple(raw["model_conditioning_variables"])
                if raw.get("model_conditioning_variables") is not None
                else None
            ),
            hbss_subsample_dir=_as_optional_str(raw.get("hbss_subsample_dir")),
            hbss_filename_pattern=_as_optional_str(raw.get("hbss_filename_pattern")),
            cache_dir=_as_optional_str(raw.get("cache_dir")),
            build_cache=_as_optional_bool(raw.get("build_cache")),
            overwrite_cache=_as_optional_bool(raw.get("overwrite_cache")),
            eval_cache_root=_as_optional_str(raw.get("eval_cache_root")),
            eval_cache_dir=_as_optional_str(raw.get("eval_cache_dir")),
            build_eval_cache=_as_optional_bool(raw.get("build_eval_cache")),
            overwrite_eval_cache=_as_optional_bool(raw.get("overwrite_eval_cache")),
            plot_aggregate_over=_as_optional_str(raw.get("plot_aggregate_over")),
            build_noise_ceiling=_as_optional_bool(raw.get("build_noise_ceiling")),
            noise_ceiling_n_repeats=_as_optional_int(raw.get("noise_ceiling_n_repeats")),
            overwrite_noise_ceiling=_as_optional_bool(raw.get("overwrite_noise_ceiling")),
            noise_ceiling_dir=_as_optional_str(raw.get("noise_ceiling_dir")),
            median_umi=float(raw["median_umi"]) if raw.get("median_umi") is not None else None,
            source_path=path.resolve(),
        )

    # -- derived properties --------------------------------------------------

    def derive_conditioning_vars(self) -> Tuple[str, ...]:
        """Default eval-cache key: bio + pert + plate (deduped, plate last when present)."""
        if self.model_conditioning_variables is not None:
            return self.model_conditioning_variables
        seen: List[str] = []
        for col in (*self.biological_context_columns, *self.perturbation_columns):
            if col not in seen:
                seen.append(col)
        if self.plate_column and self.plate_column not in seen:
            seen.append(self.plate_column)
        return tuple(seen)

    def derive_control_match_keys(self) -> Tuple[str, ...]:
        """Default μ_ctrl grouping: plate (if any) + bio context."""
        if self.control_match_keys is not None:
            return self.control_match_keys
        keys: List[str] = []
        if self.plate_column:
            keys.append(self.plate_column)
        for c in self.biological_context_columns:
            if c not in keys:
                keys.append(c)
        return tuple(keys)

    def to_dataset_paths_defaults(self) -> Dict[str, Any]:
        """Project registry fields onto :class:`DatasetPaths` attribute names.

        Used by :func:`src.config.loader.load_config` to hydrate unset inline
        preset blocks. Only explicitly set registry values are returned — ``None``
        means "no default" (e.g. ``cache_dir: null`` in YAML).
        """
        out: Dict[str, Any] = {
            "data_dir": self.data_dir or None,
            "files_glob": self.plate_glob or None,
            "cache_dir": self.cache_dir,
            "eval_cache_root": self.eval_cache_root,
            "eval_cache_dir": self.eval_cache_dir,
            "plot_aggregate_over": self.plot_aggregate_over,
            "noise_ceiling_dir": self.noise_ceiling_dir,
            "build_cache": self.build_cache,
            "overwrite_cache": self.overwrite_cache,
            "build_eval_cache": self.build_eval_cache,
            "overwrite_eval_cache": self.overwrite_eval_cache,
            "build_noise_ceiling": self.build_noise_ceiling,
            "noise_ceiling_n_repeats": self.noise_ceiling_n_repeats,
            "overwrite_noise_ceiling": self.overwrite_noise_ceiling,
            "model_conditioning_variables": list(self.derive_conditioning_vars()) or None,
            "control_condition": dict(self.control_pert) if self.control_pert else None,
            "control_match_keys": list(self.derive_control_match_keys()) or None,
            "metadata_joins": (
                [dict(j) if isinstance(j, dict) else j for j in self.metadata_joins]
                if self.metadata_joins
                else None
            ),
        }
        return {k: v for k, v in out.items() if v is not None}


def load_dataset_registry(
    alias_or_key: str,
    *,
    path: Optional[Path] = None,
) -> DatasetRegistry:
    """Load a registry by alias (``tahoe``), registry key (``tahoe100M``), or explicit path.

    Args:
        alias_or_key: Lookup key. Used when ``path`` is None.
        path: Explicit YAML path; takes precedence over ``alias_or_key`` and
            short-circuits alias resolution.

    Returns:
        :class:`DatasetRegistry` parsed from the resolved YAML file.
    """
    if path is not None:
        return DatasetRegistry.from_yaml(Path(path))
    return DatasetRegistry.from_yaml(registry_yaml_path(alias_or_key))


# ---------------------------------------------------------------------------
# Dataset condition spec (consumer interface)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetConditionSpec:
    """Dataset-agnostic semantic spec for kNN baselines and exploration.

    Built from a :class:`DatasetRegistry` plus optional per-run overrides
    (eval cache namespace, alternate gene set). Treat this as a value object:
    every method returns a derived value or a key, never mutates state.

    Attributes:
        registry_key: Canonical registry name (``tahoe100M`` / ``pbmc``).
        dataset_alias: Training-YAML alias (``tahoe`` / ``parse``); used in
            ``results/<dataset>/...`` paths so existing artifacts don't move.
        context_vars: Bio-context columns (length ≥ 1).
        perturbation_vars: Perturbation columns (length ≥ 1).
        conditioning_vars: Full eval-cache key (bio + pert + plate); the order
            here matches what :class:`src.eval.cache.eval_cache_io` expects.
        plate_column: Optional plate column name; ``None`` when absent.
        control_condition: Mapping of perturbation column → control value.
        control_match_keys: μ_ctrl grouping columns (subset of
            ``conditioning_vars``).
        eval_cache_namespace: Optional override for the per-run NPZ namespace
            (``.../eval_cache/normalization_<id>/geneset_<hash>__n_<N>``).
        data_dir: h5ad root for this dataset.
        metadata_joins: Pass-through to obs enrichment.
    """

    registry_key: str
    dataset_alias: str
    context_vars: Tuple[str, ...]
    perturbation_vars: Tuple[str, ...]
    conditioning_vars: Tuple[str, ...]
    plate_column: Optional[str]
    control_condition: Mapping[str, Any]
    control_match_keys: Tuple[str, ...]
    data_dir: Optional[str] = None
    eval_cache_namespace: Optional[Path] = None
    metadata_joins: Tuple[Mapping[str, Any], ...] = ()

    # -- constructors --------------------------------------------------------

    @classmethod
    def from_registry(
        cls,
        registry: DatasetRegistry,
        *,
        eval_cache_namespace: Optional[Path] = None,
        control_condition: Optional[Mapping[str, Any]] = None,
        control_match_keys: Optional[Sequence[str]] = None,
        conditioning_vars: Optional[Sequence[str]] = None,
    ) -> "DatasetConditionSpec":
        """Build a spec from a loaded registry. Optional kwargs let CLIs override fields."""
        return cls(
            registry_key=registry.name,
            dataset_alias=resolve_dataset_alias(registry.name),
            context_vars=registry.biological_context_columns,
            perturbation_vars=registry.perturbation_columns,
            conditioning_vars=tuple(conditioning_vars or registry.derive_conditioning_vars()),
            plate_column=registry.plate_column,
            control_condition=dict(control_condition or registry.control_pert),
            control_match_keys=tuple(control_match_keys or registry.derive_control_match_keys()),
            data_dir=registry.data_dir or None,
            eval_cache_namespace=Path(eval_cache_namespace) if eval_cache_namespace else None,
            metadata_joins=registry.metadata_joins,
        )

    @classmethod
    def from_alias(
        cls,
        alias_or_key: str,
        *,
        path: Optional[Path] = None,
        eval_cache_namespace: Optional[Path] = None,
        control_condition: Optional[Mapping[str, Any]] = None,
        control_match_keys: Optional[Sequence[str]] = None,
        conditioning_vars: Optional[Sequence[str]] = None,
    ) -> "DatasetConditionSpec":
        """Convenience: load registry then call :meth:`from_registry`."""
        reg = load_dataset_registry(alias_or_key, path=path)
        return cls.from_registry(
            reg,
            eval_cache_namespace=eval_cache_namespace,
            control_condition=control_condition,
            control_match_keys=control_match_keys,
            conditioning_vars=conditioning_vars,
        )

    @classmethod
    def from_config(cls, cfg: Any) -> "DatasetConditionSpec":
        """Build a spec from a hydrated :class:`PertGenConfig`.

        Reads the active ``dataset:`` alias and merges any per-run overrides
        already on the config (``model_conditioning_variables``, ``control_condition``,
        ``control_match_keys``, ``eval_cache_dir``).
        """
        ds = getattr(cfg, "dataset", None)
        if not ds:
            raise ValueError("PertGenConfig.dataset is unset; cannot build DatasetConditionSpec")
        ns = getattr(cfg, "eval_cache_dir", None)
        return cls.from_alias(
            str(ds),
            eval_cache_namespace=Path(ns) if ns else None,
            control_condition=getattr(cfg, "control_condition", None),
            control_match_keys=tuple(getattr(cfg, "control_match_keys", None) or ()) or None,
            conditioning_vars=(
                tuple(getattr(cfg, "model_conditioning_variables", None) or ()) or None
            ),
        )

    # -- key helpers ---------------------------------------------------------

    def _row_subkey(self, row: Mapping[str, Any], cols: Sequence[str]) -> Tuple[Any, ...]:
        """Extract ``(row[c1], row[c2], ...)`` with explicit KeyError on missing column."""
        try:
            return tuple(row[c] for c in cols)
        except KeyError as e:
            missing = e.args[0]
            raise KeyError(
                f"row missing required column {missing!r}; have {list(row.keys())}"
            ) from e

    def context_key(self, row: Mapping[str, Any]) -> Tuple[Any, ...]:
        """``(row[ctx_var_1], ...)`` for the bio-context axis."""
        return self._row_subkey(row, self.context_vars)

    def perturbation_key(self, row: Mapping[str, Any]) -> Tuple[Any, ...]:
        """``(row[pert_var_1], ...)`` for the perturbation axis."""
        return self._row_subkey(row, self.perturbation_vars)

    def conditioning_key(self, row: Mapping[str, Any]) -> Tuple[Any, ...]:
        """Full eval-cache key (bio + pert + plate)."""
        return self._row_subkey(row, self.conditioning_vars)

    def control_match_key(self, row: Mapping[str, Any]) -> Tuple[Any, ...]:
        """μ_ctrl grouping key (e.g. ``(plate, cell_name)`` for Tahoe)."""
        return self._row_subkey(row, self.control_match_keys)

    def is_control(self, row: Mapping[str, Any]) -> bool:
        """True iff every entry of ``control_condition`` matches ``row``.

        Numeric comparisons go through ``float()`` to absorb dose stringification
        (eval cache keys store doses as strings; train rows may have floats).
        """
        for col, expected in self.control_condition.items():
            if col not in row:
                return False
            actual = row[col]
            try:
                if float(actual) == float(expected):
                    continue
            except (TypeError, ValueError):
                pass
            if str(actual) != str(expected):
                return False
        return True


__all__ = [
    "DATASET_ALIAS_MAP",
    "DatasetConditionSpec",
    "DatasetRegistry",
    "load_dataset_registry",
    "registry_yaml_path",
    "resolve_dataset_alias",
    "resolve_registry_key",
]
