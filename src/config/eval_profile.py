"""Eval-CLI profile adapter — wraps the dataset registry for ``scripts/eval/``.

Two layers:

* :class:`EvalProfile` — frozen dataclass holding everything an eval CLI
  needs: the :class:`DatasetConditionSpec` (semantics from the registry)
  plus per-environment paths (truth cache root, condition summary, gene
  names file) that vary by deployment but not by dataset.
* :func:`build_eval_profile` — constructor that resolves an alias
  (``tahoe`` / ``parse``) to an :class:`EvalProfile`. Per-run overrides
  for ``condition_summary``, ``truth_cache_root``, ``gene_names``,
  ``split_yaml`` are accepted as keyword arguments.

The dataset *defaults* for ``condition_summary`` and ``truth_cache_root``
are stored here (not in the registry YAML) because they are
deployment-level: a different lab cluster will mount them at different
paths. Both defaults are Tahoe — Parse and future datasets ship
with ``None`` and require an explicit CLI flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from .dataset_registry import DatasetConditionSpec, load_dataset_registry, resolve_dataset_alias


# Default repo paths used by Tahoe eval CLIs. The truth cache lives under the registry's
# data_dir (core_util_data/datasets/tahoe.yaml), the one place that names a machine.
_TAHOE_DEFAULT_GENE_NAMES = "core_util_data/tahoe/gene_subsets/STATE_HVG.txt"
_TAHOE_DEFAULT_SUMMARY = "core_util_data/tahoe/metadata/quadruplet_summary.parquet"
_TAHOE_DEFAULT_SPLIT = "core_util_data/splits/tahoe/bio/tahoe_legacy_bio_holdout.yaml"


@dataclass(frozen=True)
class EvalProfile:
    """All-in-one bundle for a pregenerate / score CLI.

    Attributes:
        spec: The :class:`DatasetConditionSpec` (registry-derived).
        condition_summary: Path to the per-condition summary parquet.
        truth_cache_root: Eval-cache root for ``resolve_cache_namespace``.
        default_split_yaml: Default split YAML when ``--split-yaml`` is
            unset (CLI scripts may overwrite).
        default_gene_names: Default gene list for ``resolve_cache_namespace``.
        results_subdir: Default leaf directory under
            ``results/<dataset>/split_half/`` for this CLI.
        plot_aggregate_over: Optional column to aggregate cross-condition
            metrics over (Tahoe: ``None``; if a dataset wants
            plate-aggregation set this to ``"plate"``).
    """

    spec: DatasetConditionSpec
    condition_summary: Optional[Path] = None
    truth_cache_root: Optional[Path] = None
    default_split_yaml: Optional[Path] = None
    default_gene_names: Optional[Path] = None
    results_subdir: str = "pregenerated_baselines"
    plot_aggregate_over: Optional[str] = None

    # -- views into the spec -------------------------------------------------

    @property
    def name(self) -> str:
        """Dataset alias (``tahoe`` / ``parse``) — used in result paths."""
        return self.spec.dataset_alias

    @property
    def condition_variables(self) -> Tuple[str, ...]:
        return self.spec.conditioning_vars

    @property
    def control_condition(self) -> Mapping[str, Any]:
        return self.spec.control_condition

    @property
    def control_match_keys(self) -> Tuple[str, ...]:
        return self.spec.control_match_keys


def build_eval_profile(
    alias: str,
    *,
    repo_root: Optional[Path] = None,
    condition_summary: Optional[Path] = None,
    truth_cache_root: Optional[Path] = None,
    split_yaml: Optional[Path] = None,
    gene_names: Optional[Path] = None,
    results_subdir: str = "pregenerated_baselines",
    plot_aggregate_over: Optional[str] = None,
) -> EvalProfile:
    """Build an :class:`EvalProfile` from a dataset alias plus optional overrides.

    Args:
        alias: Registry alias (``tahoe`` / ``parse`` / ``tahoe100M`` / ``pbmc``).
        repo_root: Optional override for the PertGen repo root. When None,
            uses :data:`src.paths.PERTGEN_REPO_ROOT` (lazy import to avoid
            circular dependency).
        condition_summary: Override for the condition summary parquet.
        truth_cache_root: Override for the eval-cache root.
        split_yaml: Override for the default split YAML.
        gene_names: Override for the canonical gene-names file.
        results_subdir: Leaf directory under ``results/<dataset>/split_half/``.
        plot_aggregate_over: Optional column name for aggregating per-condition
            metrics in cross-condition plots.

    Returns:
        Fully populated :class:`EvalProfile` ready to plug into a CLI.
    """
    if repo_root is None:
        from src.paths import PERTGEN_REPO_ROOT  # imported lazily

        repo_root = PERTGEN_REPO_ROOT

    spec = DatasetConditionSpec.from_alias(alias)

    # Per-alias deployment defaults. Only Tahoe has them filled in;
    # other aliases require explicit CLI flags.
    canonical_alias = resolve_dataset_alias(alias)
    if canonical_alias == "tahoe":
        condition_summary = condition_summary or (repo_root / _TAHOE_DEFAULT_SUMMARY)
        truth_cache_root = truth_cache_root or Path(load_dataset_registry("tahoe").data_dir) / "eval_cache"
        split_yaml = split_yaml or (repo_root / _TAHOE_DEFAULT_SPLIT)
        gene_names = gene_names or (repo_root / _TAHOE_DEFAULT_GENE_NAMES)

    return EvalProfile(
        spec=spec,
        condition_summary=Path(condition_summary) if condition_summary else None,
        truth_cache_root=Path(truth_cache_root) if truth_cache_root else None,
        default_split_yaml=Path(split_yaml) if split_yaml else None,
        default_gene_names=Path(gene_names) if gene_names else None,
        results_subdir=results_subdir,
        plot_aggregate_over=plot_aggregate_over,
    )
