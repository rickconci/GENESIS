"""Lightning module: trains the condition MLP on energy score + variogram, in log1p.

The Extended Data Fig. 9 ablations reuse this class; their extra pieces live in
``lightning_ablations``.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from ..config import PertGenConfig, resolve_condition_input_variables
from ..data.data_utils import _numpy_batch_to_tensor, densify_gene_expression
from ..training.optimizer import build_wsd_optimizer_and_scheduler
from .lightning_ablations import estimate_rbf_bandwidth, rbf_bandwidths, set_level_rbf_step
from .losses import pertgen_loss
from .pertgen import OptimizedCoreModel, is_deterministic, sample_condition_noise

logger = logging.getLogger("pertgen")


class PertGenLightningModule(pl.LightningModule):
    """Trains OptimizedCoreModel: each of B observed cells gets n_repeats generated cells.

    Every cell's condition is repeated n_repeats times with fresh noise, so a step
    runs B * n_repeats forward passes (docs/noise_and_n_repeats.md).
    """

    def __init__(self, config: PertGenConfig) -> None:
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        if not config.apply_log1p:
            raise ValueError("GENESIS trains in log1p: set apply_log1p=true")
        if is_deterministic(config):
            logger.warning("train_loss=mse: deterministic model, condition noise is zero")

        self.model = OptimizedCoreModel(config)
        if config.compile_model:
            # In place, so parameter keys stay model.X: compiled and eager modules share
            # checkpoints. Only the set-level ablation changes its row count (C * n_repeats)
            # per step; dynamic=True symbolises every int, layer widths included, which
            # pushes Triton index math to fp32, so only that run pays for it.
            self.model.compile(fullgraph=True, mode=config.compile_mode, dynamic=config.group_by_condition)

    def on_fit_start(self) -> None:
        """Warmstart the head bias, then fix the RBF bandwidth for the RBF ablations that leave it unset."""
        if self.config.head_bias_init_from_train_mean:
            self._init_head_bias_from_train_mean()
        if self.config.train_loss == "rbf_kernel_score" and self.config.rbf_bandwidth_median is None:
            estimate_rbf_bandwidth(self)

    def _init_head_bias_from_train_mean(self) -> None:
        """Warmstart the output head's bias so predictions at step 0 match the
        marginal per-gene mean of the training data.

        Rationale: at default init `head.bias ≈ 0`, so predictions ≈ 0 and the
        energy-score loss is essentially `E[||Y||]` — dominated by the target
        marginal magnitude, not by structure the model needs to learn. Setting
        `head.bias = μ_gene` (≥ 0, so ReLU passes it through) drops step-0 loss to `√Σᵢ var(yᵢ)`, the
        "always-predict-mean" baseline.

        On the set-level (cell_set) ablation this is a plain mean over the cells drawn,
        so weighted by condition rather than by cell -- the one way its init differs
        from the per-cell partner's.
        """
        head = self.model.condition_mlp.head
        t_scan = time.perf_counter()
        total = torch.zeros(head.bias.numel(), dtype=torch.float64, device=head.bias.device)
        n_rows = n_batches = 0
        for batch in self._scan_loader():
            ge = self._gene_expression_for_loss(
                {k: v.to(head.bias.device) if hasattr(v, "to") else v for k, v in batch.items()})
            total += ge.detach().to(torch.float64).sum(dim=0)
            n_rows += ge.shape[0]
            n_batches += 1
            if n_batches >= self.config.head_bias_init_n_batches:
                break
        if n_rows == 0:
            raise RuntimeError("head_bias_init: consumed 0 rows from train_dataloader")

        mu = (total / n_rows).to(head.bias.dtype)  # (num_output_genes,)
        bias = mu.clamp(min=0.0)  # mu >= 0 on log1p targets, so ReLU(bias) = mu exactly
        with torch.no_grad():
            head.bias.copy_(bias)
        print(f"  head_bias_init: warmstarted from {n_rows:,} rows over {n_batches} batches in "
              f"{time.perf_counter() - t_scan:.0f}s | μ_gene mean={float(mu.mean()):.4f} std={float(mu.std()):.4f}",
              flush=True)

    def _scan_loader(self) -> DataLoader:
        """A throwaway single-process loader over the training dataset, for start-of-fit scans.

        Partially iterating the datamodule's persistent-worker, spawn-context, cache-backed
        DataLoader triggers a SIGABRT on shutdown (workers' C++ background threads abort).
        Grabbing just the dataset object is safe -- spawn only happens on iter(), not on
        DataLoader construction -- so the proxy is discarded without ever being iterated.
        """
        proxy = self.trainer.datamodule.train_dataloader()
        # One process. scDataset exposes that order as in_process(); a loader without it
        # (cell_set) is already single-process.
        ds = proxy.dataset.in_process() if hasattr(proxy.dataset, "in_process") else proxy.dataset
        loader = DataLoader(ds, batch_size=None, num_workers=0, pin_memory=False, collate_fn=_numpy_batch_to_tensor)
        del proxy  # discarded before iter() → no workers ever spawned
        return loader

    def _gene_expression_for_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Raw counts -> dense log1p training target. The loss and the RBF bandwidth scan both read it."""
        return torch.log1p(densify_gene_expression(batch["gene_expression"]))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        cfg = self.config
        # The set-level ablation (Extended Data Fig. 9a) scores one sample pool per
        # condition. Every other run uses the per-cell step below.
        if cfg.group_by_condition:
            return set_level_rbf_step(self, batch)
        # A cell_set batch carries one code row per condition, which only the set-level step reads.
        if "cell_counts" in batch:
            raise ValueError("train_loader_mode='cell_set' needs group_by_condition=true")

        # Codes are tiled once per generated cell. The target stays (B, G): the energy score
        # broadcasts it and the variogram slices it, so tiling would only cost memory
        # (3.4 GiB of fp32 at the main config's 1792 cells x 256 repeats x 2000 genes).
        conditions = {f"{v}_idx": batch[f"{v}_idx"].repeat_interleave(cfg.n_repeats, dim=0)
                      for v in resolve_condition_input_variables(cfg)}
        gene_expression = self._gene_expression_for_loss(batch)
        noise = sample_condition_noise(cfg, gene_expression.size(0) * cfg.n_repeats,
                                       device=gene_expression.device, dtype=gene_expression.dtype)

        metrics = pertgen_loss(
            gene_expression,
            self.model(noise=noise, **conditions)["predictions"],
            cfg,
            rbf_bandwidths=rbf_bandwidths(cfg) if cfg.train_loss == "rbf_kernel_score" else None,
        )
        for key, val in metrics.items():
            self.log(f"train/{key}", val, prog_bar=(key == "loss"))
        return metrics["loss"]

    # ------------------------------------------------------------------
    # Optimiser (WSD schedule, fused AdamW)
    # ------------------------------------------------------------------
    def configure_optimizers(self) -> Dict[str, Any]:
        return build_wsd_optimizer_and_scheduler(self, self.config)
