"""The split-half reference: how well truth predicts itself.

``SplitHalfSampler`` draws the two halves. ``run_split_half_reference`` scores them
through the shared loop (same row schema as a model's metrics_summary.json) into
``noise_ceiling.json``.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import torch

from src.data.cache.eval_cache_build import EVAL_NPZ_MODE
from ..cache.eval_cache_io import QuadrupletRow, group_by_condition, resolve_eval_cache_mode
from ..orchestration.log_utils import log_step

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitHalfSampler:
    """Disjoint split-half sampler; an odd leftover truth or control cell is dropped."""

    name: str = "split_half"
    pred_label: str = "split_half_b"

    @property
    def strategy_params(self) -> dict[str, float]:
        return {}

    def sample(
        self,
        truth_raw: torch.Tensor,
        control_raw: torch.Tensor,
        *,
        rep: int,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(truth_a, truth_b, ctrl_a, ctrl_b, b_idx)`` for repeat ``rep``; raises below 4 cells per pool."""
        torch.manual_seed(int(seed) + int(rep))
        n = int(truth_raw.shape[0])
        half = n // 2
        if half < 2:
            raise ValueError(
                f"SplitHalfSampler needs N >= 4 to produce 2 halves of >=2 cells; got N={n}.",
            )
        perm = torch.randperm(n, device=truth_raw.device)
        a_idx = perm[:half]
        b_idx = perm[half : 2 * half]

        m = int(control_raw.shape[0])
        half_m = m // 2
        if half_m < 2:
            raise ValueError(f"SplitHalfSampler needs M >= 4 control cells; got M={m}.")
        perm_m = torch.randperm(m, device=control_raw.device)
        c_a_idx = perm_m[:half_m]
        c_b_idx = perm_m[half_m : 2 * half_m]

        return (
            truth_raw[a_idx],
            truth_raw[b_idx],
            control_raw[c_a_idx],
            control_raw[c_b_idx],
            b_idx,
        )

    def min_cells(self) -> int:
        return 4

    def target_arm_size(self, n_truth: int) -> int:
        return n_truth // 2


class NoiseCeilingError(RuntimeError):
    """The reference cannot be trusted: no conditions, or eval-cache truth/controls missing.

    Unlike a model run, a reference with holes would silently shrink every model's denominator,
    so infrastructure skips are fatal here rather than logged.
    """


_INFRA_SKIPS = ("missing_truth_or_control", "missing_control_npz")


def run_split_half_reference(
    *,
    truth_dir: Union[str, Path],
    work_queue: Sequence[QuadrupletRow],
    n_repeats: int = 20,
    seed: int = 42,
    device: Union[str, torch.device] = "cuda",
    metrics_workers: int = 4,
    condition_variables: Sequence[str],
    gene_arrays_parquet: Path,
    eval_group_keys: Optional[Sequence[str]] = None,
    ctrl_det_floors: tuple[float, ...] = (0.10,),
    dataset: Optional[str] = None,
    metrics: Optional[frozenset[str]] = None,
    gene_sidecars: bool = False,
) -> tuple[list[dict[str, Any]], Counter]:
    """Score truth half A against truth half B for every condition: (rows, skip reasons)."""
    from ..metrics import spaces
    from ..orchestration.model_eval import ModelEvalPlan
    from ..scoring import SplitHalfReference, score_conditions

    if dataset is not None:
        spaces.set_dataset_median_umi(str(dataset))
    elif spaces._ACTIVE_L_REF is None:
        raise ValueError(
            "run_split_half_reference requires dataset=... (or a prior "
            "set_dataset_median_umi call) for log1p_slice_med_rgs."
        )
    ns = Path(truth_dir)
    if not ns.is_dir():
        raise FileNotFoundError(f"Eval truth namespace not found: {ns}")
    if resolve_eval_cache_mode(ns) != EVAL_NPZ_MODE:
        raise ValueError(
            f"Split-half reference requires NPZ eval truth at {ns} "
            f"(mode={resolve_eval_cache_mode(ns)!r}). "
            "Rebuild with scripts/data/build_cache.py eval.",
        )
    bundles = group_by_condition(work_queue, eval_group_keys=eval_group_keys)
    if not bundles:
        raise NoiseCeilingError("Noise-ceiling work queue produced 0 conditions.")
    plan = ModelEvalPlan(active=True, k=int(n_repeats), metrics=metrics, gene_sidecars=gene_sidecars,
                         ctrl_det_floors=tuple(ctrl_det_floors) or (0.10,))
    log_step(logger, "setup", "split_half | %d repeats | %d conditions | truth=%s", n_repeats, len(bundles), ns)
    results, skips = score_conditions(
        SplitHalfReference(n_repeats), cache_ns=ns, bundles=bundles, condition_variables=condition_variables,
        plan=plan, seed=seed, device=torch.device(device), metrics_workers=metrics_workers,
        gene_arrays_path=Path(gene_arrays_parquet), desc="Split-half reference",
    )
    if any(skips.get(k, 0) for k in _INFRA_SKIPS):
        raise NoiseCeilingError(
            f"Noise-ceiling hit missing eval-cache NPZs or controls (skip_reasons={dict(skips)}). "
            "Rebuild the eval cache."
        )
    if not results:
        raise NoiseCeilingError(f"Noise-ceiling scored 0 conditions (skip_reasons={dict(skips)}).")
    return results, skips


run_noise_ceiling = run_split_half_reference  # on-disk and config names stay noise_ceiling
