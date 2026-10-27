"""Shared optimizer + WSD schedule used by every Lightning module in PertGen.

``PertGenLightningModule`` delegates ``configure_optimizers`` to
:func:`build_wsd_optimizer_and_scheduler` so AdamW grouping, fused kernels,
and the warmup / stable / decay schedule stay in one place.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from ..config import PertGenConfig, epoch_lr_milestones


def compute_wsd_step_counts(
    total_steps: int,
    *,
    warmup_steps_min: int = 500,
    warmup_steps_max: Optional[int] = None,
    wsd_stable_frac: float = 0.85,
) -> Tuple[int, int, int]:
    """Return ``(warmup_steps, stable_steps, decay_window)`` for WSD LambdaLR."""
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    frac = float(wsd_stable_frac)
    if frac <= 0.0:
        raise ValueError(f"wsd_stable_frac must be > 0, got {frac}")
    min_warmup = int(warmup_steps_min)
    max_warmup: Optional[int] = warmup_steps_max
    if max_warmup is not None and int(max_warmup) < min_warmup:
        logging.getLogger(__name__).warning(
            "warmup_steps_max=%s < warmup_steps_min=%s; cap takes precedence.",
            max_warmup,
            min_warmup,
        )
    # At least 5% of the run, then clamped to [warmup_steps_min, warmup_steps_max].
    warmup_steps = max(min_warmup, int(0.05 * total_steps))
    if max_warmup is not None:
        warmup_steps = min(warmup_steps, int(max_warmup))
    stable_steps = int(frac * total_steps)
    decay_window = max(1, total_steps - (warmup_steps + stable_steps))
    return warmup_steps, stable_steps, decay_window


def warmup_fraction(
    current_step: int,
    warmup_steps: int,
    cfg: PertGenConfig,
) -> float:
    """Linear ramp from ``lr_init / lr`` to 1.

    The optimizer is created at ``lr`` and ``LambdaLR`` multiplies it, so
    ``lr_init`` is an absolute learning rate. ``0`` ramps from zero.
    """
    if warmup_steps <= 0:
        return 1.0
    peak = float(cfg.lr)
    start = float(cfg.lr_init)
    if not math.isfinite(peak) or peak <= 0.0:
        raise ValueError(f"lr must be positive and finite, got {peak}")
    if not math.isfinite(start) or start < 0.0 or start > peak:
        raise ValueError(f"lr_init must lie in [0, lr], got lr_init={start}, lr={peak}")
    progress = min(1.0, float(current_step) / float(warmup_steps))
    start_frac = start / peak
    return start_frac + (1.0 - start_frac) * progress


def epoch_lr_multiplier(milestones: Sequence[Tuple[int, float]], epoch: int) -> float:
    """``lr_mul`` of the last milestone at or before ``epoch`` (sorted, first at epoch 0)."""
    return next(mul for e, mul in reversed(milestones) if e <= epoch)


def wsd_lr_multiplier(
    current_step: int,
    total_steps: int,
    cfg: PertGenConfig,
    *,
    epoch_mul: float = 1.0,
) -> float:
    """WSD multiplier x the epoch's peak scale, for ``LambdaLR`` (no trainer side effects)."""
    warmup_steps, stable_steps, decay_window = compute_wsd_step_counts(
        total_steps,
        warmup_steps_min=cfg.warmup_steps_min,
        warmup_steps_max=cfg.warmup_steps_max,
        wsd_stable_frac=cfg.wsd_stable_frac,
    )
    if current_step < warmup_steps:
        return warmup_fraction(current_step, warmup_steps, cfg) * epoch_mul
    if current_step < warmup_steps + stable_steps:
        return epoch_mul
    progress = float(current_step - (warmup_steps + stable_steps)) / float(decay_window)
    return max(0.1, 1.0 - progress) * epoch_mul


def build_adamw_param_groups(
    model: torch.nn.Module,
    weight_decay: float,
    no_decay_name_substrings: Iterable[str] = ("embed",),
) -> List[Dict[str, Any]]:
    """Two AdamW groups: MLP matrices decay, biases and embedding tables do not.

    1D parameters (biases, norm scales) and any name containing ``embed`` get
    ``weight_decay=0``. The condition tables are 2D, so the shape rule misses
    them; a substring match still catches them under a ``torch.compile`` prefix.
    Decay on those pulls the per-gene output bias and rare condition rows
    toward zero on every step.
    """
    no_decay_substrings = tuple(no_decay_name_substrings)
    decay: List[torch.nn.Parameter] = []
    no_decay: List[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_excluded = param.ndim <= 1 or any(sub in name for sub in no_decay_substrings)
        (no_decay if is_excluded else decay).append(param)
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _build_optimizer(module: pl.LightningModule, cfg: PertGenConfig) -> torch.optim.Optimizer:
    """AdamW, fused on CUDA when ``grad_clip`` is 0.

    Fused AdamW unscales gradients internally, which conflicts with Lightning's
    AMP gradient clipping. Enable it only when no clipping is configured.
    """
    optim_groups = build_adamw_param_groups(module, weight_decay=cfg.weight_decay)
    use_fused = torch.cuda.is_available() and cfg.grad_clip <= 0
    try:
        return AdamW(optim_groups, lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
    except TypeError:
        return AdamW(optim_groups, lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8)


def build_wsd_optimizer_and_scheduler(
    module: pl.LightningModule,
    cfg: PertGenConfig,
) -> Dict[str, Any]:
    """Return the Lightning ``configure_optimizers`` dict (optimizer + WSD LambdaLR).

    WSD = Warmup–Stable–Decay (Hu et al.):
        warmup (linear lr_init/lr → 1)  |  stable (1.0, ~85% of steps)  |  decay (linear → 0.1)
    times the current epoch's ``lr_mul`` (``between_epoch_curriculum_schedule``).

    ``total_training_steps`` must be precomputed on ``cfg`` by ``src.main`` before
    ``trainer.fit`` — reading it from ``trainer.estimated_stepping_batches`` here
    would trigger ``fit_loop.setup_data()`` inside ``LambdaLR.__init__`` and silently
    swallow dataloader errors.
    """
    optimizer = _build_optimizer(module, cfg)

    total_steps = cfg.total_training_steps
    if total_steps <= 0:
        raise ValueError(
            "cfg.total_training_steps must be a positive int before fit; "
            "src.main.run() sets this prior to pl.Trainer(...). If you call "
            "the Lightning module directly, populate it yourself."
        )

    milestones = epoch_lr_milestones(cfg)
    warmup_steps = compute_wsd_step_counts(
        total_steps,
        warmup_steps_min=cfg.warmup_steps_min,
        warmup_steps_max=cfg.warmup_steps_max,
        wsd_stable_frac=cfg.wsd_stable_frac,
    )[0]

    def lr_lambda(current_step: int) -> float:
        mul = epoch_lr_multiplier(milestones, int(module.trainer.current_epoch))
        # A step inside the warmup ramp would make the LR fall and then climb again.
        if current_step < warmup_steps and mul != milestones[0][1]:
            raise ValueError(
                f"lr_mul changes to {mul:g} at step {current_step}, inside the "
                f"{warmup_steps}-step warmup; move that milestone to a later epoch."
            )
        return wsd_lr_multiplier(current_step, total_steps, cfg, epoch_mul=mul)

    scheduler = LambdaLR(optimizer, lr_lambda)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
    }
