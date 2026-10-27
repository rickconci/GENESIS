"""Training-loop pieces for the ablation objectives (Extended Data Fig. 9).

The published runs never touch this file. ``lightning_module`` reaches it in two
places: the RBF bandwidth at fit start (both RBF arms) and ``training_step``'s
``group_by_condition`` branch (the set-level arm). The squared-error arm needs
nothing here; it is a branch of :func:`~src.model.losses.pertgen_loss`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import torch

from ..config import PertGenConfig, resolve_condition_input_variables
from .pertgen import sample_condition_noise
from .losses_ablations import grouped_rbf_kernel_score, median_pairwise_distance

if TYPE_CHECKING:
    from .lightning_module import PertGenLightningModule


# ----------------------------------------------------------------------------
# The RBF bandwidth: fixed once per run, shared by the per-cell and set-level arms
# ----------------------------------------------------------------------------

def rbf_bandwidths(config: PertGenConfig) -> tuple[float, ...]:
    """The fixed RBF bandwidths: training-cell median distance x the configured scales."""
    if config.rbf_bandwidth_median is None:
        raise RuntimeError("train_loss=rbf_kernel_score has no bandwidth: rbf_bandwidth_median is unset "
                           "(it is estimated at fit start, or set it in the config)")
    return tuple(config.rbf_bandwidth_median * s for s in config.rbf_bandwidth_scales)


def estimate_rbf_bandwidth(module: "PertGenLightningModule") -> None:
    """Median distance between training cells (log1p), once, before the first step.

    Rows are drawn evenly from the first 8 training batches: the loader shuffles in
    contiguous blocks, so one batch would hold only a few conditions. The median is
    written beside the run and reused when the run resumes, so the kernel never moves.
    """
    config = module.config
    path = Path(module.trainer.default_root_dir) / "rbf_bandwidth.json"
    if path.is_file():
        config.rbf_bandwidth_median = float(json.loads(path.read_text())["median_pairwise_distance"])
        print(f"  rbf bandwidth: median {config.rbf_bandwidth_median:.4f} reused from {path}", flush=True)
        return
    n_batches, n_rows = 8, config.rbf_bandwidth_n_rows
    g = torch.Generator().manual_seed(config.seed)
    rows = []
    for i, batch in enumerate(module._scan_loader()):
        ge = module._gene_expression_for_loss(batch).detach().float().cpu()
        rows.append(ge[torch.randperm(ge.shape[0], generator=g)[: -(-n_rows // n_batches)]])
        if i + 1 >= n_batches:
            break
    x = torch.cat(rows)[:n_rows]
    med = median_pairwise_distance(x.to(module.device))
    config.rbf_bandwidth_median = med
    info = {"median_pairwise_distance": med, "n_rows": int(x.shape[0]), "n_batches": len(rows),
            "scales": list(config.rbf_bandwidth_scales), "bandwidths": list(rbf_bandwidths(config)),
            "space": "log1p"}
    path.write_text(json.dumps(info, indent=2) + "\n")
    print(f"  rbf bandwidth: median {med:.4f} from {x.shape[0]:,} training cells over {len(rows)} batches "
          f"-> sigmas {[round(b, 4) for b in info['bandwidths']]} ({path})", flush=True)


# ----------------------------------------------------------------------------
# The set-level step: ONE sample pool per condition (Extended Data Fig. 9a)
# ----------------------------------------------------------------------------

def set_level_rbf_step(module: "PertGenLightningModule", batch: Dict[str, Any]) -> torch.Tensor:
    """The set-level RBF kernel score ablation's training step.

    The per-cell path tiles each row ``n_repeats`` times, so a condition
    appearing ``m`` times in the batch has its law evaluated ``m * n_repeats``
    times. Here it is evaluated ``n_repeats`` times regardless of ``m``, which
    is why cells-per-condition stops costing forward passes.

    ``n_repeats`` is R' — samples per CONDITION.
    """
    config = module.config
    n_repeats = config.n_repeats
    if config.train_loss != "rbf_kernel_score" or config.train_loader_mode != "cell_set" or n_repeats < 2:
        raise ValueError("group_by_condition is the set-level rbf_kernel_score ablation: it needs "
                         "train_loss=rbf_kernel_score, train_loader_mode=cell_set and n_repeats >= 2 "
                         f"(R', samples per condition); got {config.train_loss!r}, "
                         f"{config.train_loader_mode!r}, {n_repeats}")
    if config.variogram_loss_weight != 0.0:
        raise ValueError("set-level rbf_kernel_score scores no auxiliary term; set variogram_loss_weight to 0")

    cond_vars = resolve_condition_input_variables(config)
    # (C, K): cell_set ships one code row per condition.
    cond_keys = torch.stack([batch[f"{v}_idx"].reshape(-1).long() for v in cond_vars], dim=1)

    gene_expression = module._gene_expression_for_loss(batch)
    device, dtype = gene_expression.device, gene_expression.dtype

    counts = batch.get("cell_counts")
    if counts is None:
        raise ValueError("group_by_condition needs a cell_set batch (missing cell_counts)")
    if cond_keys.shape[0] != counts.shape[0]:
        raise ValueError(f"cell_set batch: {cond_keys.shape[0]} condition code rows "
                         f"but {counts.shape[0]} cell_counts")
    targets, mask = pad_ragged_targets(gene_expression, counts)
    C = cond_keys.shape[0]
    # Each condition's term weighs its training-cell budget, so the set-level
    # objective is the per-cell one (where every cell is a term).
    budget = batch.get("cell_budget")
    if budget is None or tuple(budget.shape) != (C,):
        raise ValueError(f"set-level rbf_kernel_score needs batch['cell_budget'] of shape ({C},); "
                         f"got {None if budget is None else tuple(budget.shape)} "
                         "(enable cell_set_budget_from_train_indices)")

    conditions = {f"{var}_idx": cond_keys[:, j].repeat_interleave(n_repeats, dim=0)
                  for j, var in enumerate(cond_vars)}
    noise = sample_condition_noise(config, C * n_repeats, device=device, dtype=dtype)
    predictions = module.model(noise=noise, **conditions)["predictions"].reshape(C, n_repeats, -1)

    metrics = grouped_rbf_kernel_score(
        predictions, targets, mask, rbf_bandwidths(config), budget.to(predictions.device))
    metrics["grouped_n_conditions"] = torch.tensor(float(C), device=device)
    metrics["grouped_cells_per_condition"] = mask.sum(dim=1).to(dtype).mean()
    for key, val in metrics.items():
        module.log(f"train/{key}", val, prog_bar=(key == "loss"))
    return metrics["loss"]


def pad_ragged_targets(gene_expression: torch.Tensor, cell_counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(N_total, G)`` cells, concatenated by condition -> ``(C, m_max, G)`` targets + ``(C, m_max)`` mask.

    The cell_set loader ships the partition as ``cell_counts``, so this only pads.
    Short conditions are never topped up with copies: ``mask`` marks the real cells
    and :func:`~src.model.losses_ablations.grouped_rbf_kernel_score` divides by their count.
    """
    if int(cell_counts.sum()) != gene_expression.shape[0]:
        raise ValueError(f"cell_counts sum to {int(cell_counts.sum())} but gene_expression has "
                         f"{gene_expression.shape[0]} rows")
    if not bool((cell_counts > 0).all()):
        raise ValueError(f"every condition needs >= 1 cell; got min={int(cell_counts.min())}")

    counts = cell_counts.to(device=gene_expression.device, dtype=torch.int64)
    C, m_max = counts.shape[0], int(counts.max())
    # Row-major selection over (C, m_max) walks condition 0's cells first, which
    # is exactly the order the loader concatenated them in.
    mask = torch.arange(m_max, device=counts.device)[None, :] < counts[:, None]
    targets = gene_expression.new_zeros(C, m_max, gene_expression.shape[-1])
    targets[mask] = gene_expression
    return targets, mask
