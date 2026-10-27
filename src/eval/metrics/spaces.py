"""Report spaces: snap every prediction to integer counts, then project.

Every published number lives in a round-geom-snapped (``*_rgs``) space derived
from one integer matrix, so truth, controls and any model are scored on the
same lattice. ``log1p_slice_med_rgs`` needs the dataset's median UMI (L_ref):
bind it once per process with :func:`set_dataset_median_umi`.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Union

import torch

if TYPE_CHECKING:
    from ...model.inference_output import ModelOutputSpace

from ...config.dataset_registry import load_dataset_registry


# Report spaces are ALWAYS round-geom-snapped (``*_rgs``): preds go through
# ``to_raw_integer`` once; log spaces are derived from that integer matrix.
#   log1p_rgs             — log1p of snapped counts (depth + composition)
#   log1p_slice_med_rgs   — log1p(x / lib · S) with
#                           S = n_genes / FULL_TX_GENES · L_ref
#                           (composition only; S ≈ expected HVG mass after
#                           full-tx normalize_total to the dataset median UMI)
#   raw_int_rgs           — snapped integer counts (delta + DEG)
#
# Dual report: ``log1p_rgs`` judges depth and composition together;
# ``log1p_slice_med_rgs`` isolates composition. Models trained in a
# library-normalized space are parked at median depth before scoring —
# depth errors then show up honestly on ``log1p_rgs``.
FULL_TX_GENES: int = 20_000


@functools.lru_cache(maxsize=None)
def dataset_median_umi(dataset: str) -> float:
    """``median_umi`` from the dataset registry YAML: the dataset's median full-tx UMI (L_ref)."""
    reg = load_dataset_registry(str(dataset).strip())
    if reg.median_umi is None:
        raise KeyError(f"dataset={dataset!r}: no median_umi in {reg.source_path}")
    return float(reg.median_umi)


# Module-level L_ref for the active scoring run. Set once via
# :func:`set_dataset_median_umi` before projecting to slice_med.
_ACTIVE_L_REF: float | None = None


def set_dataset_median_umi(dataset: str) -> float:
    """Bind the composition L_ref for this scoring process. Returns L_ref."""
    global _ACTIVE_L_REF
    _ACTIVE_L_REF = dataset_median_umi(dataset)
    return _ACTIVE_L_REF


def slice_med_scale(n_genes: int, *, l_ref: float | None = None) -> float:
    """``S = n_genes / FULL_TX_GENES · L_ref`` for ``log1p_slice_med_rgs``."""
    ref = float(l_ref) if l_ref is not None else _ACTIVE_L_REF
    if ref is None:
        raise RuntimeError(
            "slice_med_scale: call set_dataset_median_umi(dataset) before "
            "scoring, or pass l_ref= explicitly."
        )
    if n_genes <= 0:
        raise ValueError(f"n_genes must be positive; got {n_genes}")
    return float(n_genes) / float(FULL_TX_GENES) * ref


def slice_med_l_ref(dataset: str | None) -> float:
    """Median full-transcriptome UMI for ``dataset`` (its registry ``median_umi``).

    The training target and the eval denorm both read ``L_ref`` from here so
    a slice-med run cannot drift from ``log1p_slice_med_rgs``.
    """
    if dataset is None or not str(dataset).strip():
        raise ValueError("slice_med_log1p requires config.dataset so L_ref can be looked up")
    return dataset_median_umi(str(dataset))


def log1p_slice_med_torch(counts: torch.Tensor, *, dataset: str) -> torch.Tensor:
    """``log1p((x / sum(x)) · S)`` with the slice library floored at 1.

    ``S`` is :func:`slice_med_scale` for ``counts.shape[-1]`` genes and this
    dataset's :func:`slice_med_l_ref`. Same map eval applies when it projects
    snapped counts into ``log1p_slice_med_rgs``.
    """
    if counts.ndim != 2:
        raise ValueError(f"counts must be (N, G), got {tuple(counts.shape)}")
    x = counts if counts.dtype == torch.float32 else counts.to(torch.float32)
    scale = slice_med_scale(int(x.shape[-1]), l_ref=slice_med_l_ref(dataset))
    return _to_log1p_cpm_torch(x, scale=scale)


PSEUDOBULK_METRIC_SPACES: tuple[str, ...] = (
    "log1p_slice_med_rgs", "log1p_rgs",
)
PSEUDOBULK_DELTA_METRIC_SPACES: tuple[str, ...] = (
    "log1p_slice_med_rgs", "log1p_rgs", "raw_int_rgs",
)
DEG_SPACES: tuple[str, ...] = ("raw_int_rgs",)


def _to_log1p_cpm_torch(x: torch.Tensor, scale: float = 1e4) -> torch.Tensor:
    """raw → log1p(counts-per-``scale``), GPU. One alloc, then in-place ops.

    ``torch.log1p((x / lib) * 1e4)`` allocates three full-size temporaries
    (~3·numel·dtype bytes peak). On large control pools (~200k cells × 2k
    genes ≈ 1.76 GiB fp32) that's >5 GiB transient and trips OOMs on
    shared GPUs. Using ``x.div(lib)`` for the single alloc and then
    in-place ``mul_/log1p_`` keeps peak at ~1·numel.
    """
    lib = x.sum(dim=1, keepdim=True).clamp(min=1.0)
    out = x.div(lib)
    out.mul_(scale)
    out.log1p_()
    return out


@torch.no_grad()
def _project_raw_to_spaces(raw: torch.Tensor) -> dict[str, torch.Tensor]:
    """Integer counts (truth, controls, snapped preds) -> every ``*_rgs`` report space.

    ``log1p_slice_med_rgs`` needs the active L_ref (:func:`set_dataset_median_umi`).
    """
    raw_f = raw.to(torch.float32)
    return {
        "raw_int_rgs": raw,
        "log1p_rgs": torch.log1p(raw_f),
        "log1p_slice_med_rgs": _to_log1p_cpm_torch(raw_f, scale=slice_med_scale(int(raw_f.shape[-1]))),
    }


def _project_predictions_to_spaces(
    predicted_model_space: torch.Tensor,
    model_output_space: Union[str, "ModelOutputSpace"],
) -> dict[str, torch.Tensor]:
    """Snap preds onto integer counts (:func:`to_raw_integer`), then project like truth."""
    from ...model.inference_output import to_raw_integer

    return _project_raw_to_spaces(to_raw_integer(predicted_model_space, model_output_space))
