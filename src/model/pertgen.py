"""PertGen generator: experimental condition + noise → one cell's expression profile.

::

    <key>_idx per modality, noise
            │
            ▼
    ConditionEmbedding ──► (B, (M+1)·d)            one d-vector per modality, noise last
            │
            ▼
    ConditionMLP       ──► (B, num_output_genes)   Linear→GELU blocks, then a Linear head
            │                                      (hidden_dims=[]: the head alone, depth 0)
            ▼
    ReLU                                           expression is non-negative

Fresh noise per draw makes the network a conditional sampler; ``train_loss='mse'``
feeds zero noise and regresses the condition mean with the same network.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, Optional, Sequence

from ..config import PertGenConfig

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Condition Embedding
# ---------------------------------------------------------------------------


class ConditionEmbedding(nn.Module):
    """One learned ``tokdim`` vector per condition modality, concatenated with the noise.

    Modalities go in sorted-key order, so the flat layout and the init RNG
    stream do not depend on dict order. Weights start at ``N(0, tokdim^-1/2)``
    per element, the scale of the noise.
    """

    def __init__(self, n_categories: Dict[str, int], tokdim: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleDict()
        for key in sorted(n_categories):
            if n_categories[key] > 0:
                embed = nn.Embedding(n_categories[key], tokdim)
                nn.init.normal_(embed.weight, mean=0.0, std=float(tokdim) ** -0.5)
                self.embeddings[key] = embed

    @property
    def n_condition_modalities(self) -> int:
        """Number of non-noise condition modalities."""
        return len(self.embeddings)

    def forward(self, condition_tensors: Dict[str, torch.Tensor], noise: torch.Tensor) -> torch.Tensor:
        """``(B,)`` int64 index per modality + ``(B, tokdim)`` noise → ``(B, (M+1)·tokdim)``."""
        vectors = [embed(condition_tensors[key]) for key, embed in self.embeddings.items()]
        return torch.cat([*vectors, noise], dim=1)


def is_deterministic(config: PertGenConfig) -> bool:
    """True for an MSE checkpoint: one output per condition, whatever the noise draw."""
    return config.train_loss == "mse"


def sample_condition_noise(
    config: PertGenConfig,
    n: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """``(n, tokdim)`` noise for training and generation alike; the one place it is drawn.

    The noise is one more condition vector, so it has the embeddings' width and scale:
    ``U(-√3, √3) · tokdim^-1/2`` matches their ``N(0, tokdim^-1/2)`` in the first two
    moments. MSE regresses the condition mean, so its noise is exactly zero.
    """
    dim = int(config.condition_mlp_tokdim)
    if is_deterministic(config):
        return torch.zeros(n, dim, device=device, dtype=dtype)
    u = torch.rand(n, dim, device=device, dtype=dtype, generator=generator)
    return (u - 0.5) * (2 * math.sqrt(3)) * float(dim) ** -0.5


# ---------------------------------------------------------------------------
# Condition MLP
# ---------------------------------------------------------------------------


class ConditionMLP(nn.Module):
    """``[Linear → GELU] × len(hidden_dims)``, then a plain ``Linear`` head to the genes.

    Blocks are named ``Sequential(linear, act)`` so the parameters live at
    ``blocks.<i>.linear.*``, the layout every trained checkpoint was saved with.

    Args:
        input_dim: Concatenated input width.
        hidden_dims: Hidden layer widths. Empty means no hidden layer: the
            model is the single linear head (the depth-0 ablation).
        output_dim: Final output width.
    """

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> None:
        super().__init__()
        dims = [input_dim, *[int(dim) for dim in hidden_dims]]
        if any(dim <= 0 for dim in dims) or output_dim <= 0:
            raise ValueError(
                f"ConditionMLP dimensions must be positive, got dims={dims!r} "
                f"output_dim={output_dim}"
            )
        self.blocks = nn.Sequential(*[
            nn.Sequential(OrderedDict(linear=nn.Linear(in_d, out_d), act=nn.GELU()))
            for in_d, out_d in zip(dims[:-1], dims[1:])
        ])
        self.head = nn.Linear(dims[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(x))


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------


class OptimizedCoreModel(nn.Module):
    """Condition embedding → :class:`ConditionMLP` → ReLU.

    State-dict layout (every trained checkpoint loads strictly into it)::

        condition_embedding.embeddings.<key>.weight
        condition_mlp.blocks.<i>.linear.{weight,bias}
        condition_mlp.head.{weight,bias}
    """

    def __init__(self, config: PertGenConfig) -> None:
        super().__init__()
        # load_config checks this too, but checkpoints are rebuilt without it.
        if config.backbone_type != "condition_mlp":
            raise ValueError(f"backbone_type must be 'condition_mlp', got {config.backbone_type!r}")
        self.num_output_genes = int(config.num_output_genes)
        if self.num_output_genes <= 0:
            raise ValueError("config.num_output_genes must be > 0 (DataModule.setup sets it)")
        d = int(config.condition_mlp_tokdim)

        # Construction order is the init RNG order: embeddings, MLP blocks, head.
        self.condition_embedding = ConditionEmbedding(config.n_categories, tokdim=d)
        n_inputs = self.condition_embedding.n_condition_modalities + 1  # + noise
        self.condition_mlp = ConditionMLP(n_inputs * d, config.condition_mlp_hidden_dims, self.num_output_genes)

    def forward(
        self,
        noise: torch.Tensor,
        **condition_tensors: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """``(B, d)`` noise + ``<key>_idx`` ``(B,)`` tensors → ``{"predictions": (B, num_output_genes)}``."""
        # Strip the "_idx" suffix the lightning module appends to keys.
        mapped_conditions = {
            (k[:-4] if k.endswith("_idx") else k): v
            for k, v in condition_tensors.items()
        }
        x = self.condition_embedding(condition_tensors=mapped_conditions, noise=noise)
        return {"predictions": F.relu(self.condition_mlp(x))}
