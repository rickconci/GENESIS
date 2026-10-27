"""Inference: the output spaces, the count-lattice snap, and the GENESIS bundle.

A. GENESIS predicts ``log1p`` counts; the baseline arms hand the scorer ``raw``
   counts. :func:`to_raw_integer` snaps either onto integer counts, the one input
   every ``*_rgs`` report space (``raw_int``, ``log1p``, ``log1p_slice_med``) is
   derived from in ``src.eval.metrics.eval_metrics``.
B. :class:`PertGenInferenceBundle` loads a trained run and generates cells for
   named conditions.
"""

from __future__ import annotations

import hashlib
import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import torch

from ..config import PertGenConfig
from ..data.vocab import condition_key
from ..training.utils import _config_from_checkpoint, resolve_run_checkpoint
from .lightning_module import PertGenLightningModule
from .pertgen import sample_condition_noise

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# A. Output spaces + the count-lattice snap
# ---------------------------------------------------------------------------

class ModelOutputSpace(str, Enum):
    """The space a prediction lives in: ``LOG1P`` for GENESIS, ``RAW`` for the baseline arms."""

    RAW = "raw"
    LOG1P = "log1p"


ArrayLike = Union[np.ndarray, torch.Tensor]


def to_raw_integer(x: ArrayLike, space: Union[str, ModelOutputSpace]) -> ArrayLike:
    """Snap a prediction to the nearest integer counts, measured in the space it lives in.

    Raw rounds (0/1 cut at 0.5). For log1p, ``argmin_n |log1p(r) - log1p(n)|`` over
    ``lo <= r < lo + 1`` cuts at the geometric midpoint ``sqrt((lo+1)(lo+2)) - 1``
    (0.414, 1.449, 2.464, ...): the projection of the model output onto real count
    matrices under the metric its loss is Euclidean in. Nearly all of the effect is
    the 0/1 detection call; the cut applies at every boundary on purpose, since a
    0/1-only rule scores better but is no projection. Integers are fixed points, so
    cached truth passes through unchanged. numpy in, numpy out; torch in, torch out.
    """
    space = ModelOutputSpace(space)
    xp = torch if isinstance(x, torch.Tensor) else np
    r = xp.clip(x, 0.0, None)
    if space is ModelOutputSpace.RAW:
        return xp.round(r)
    r = xp.expm1(r)
    lo = xp.floor(r)
    return xp.where(r >= xp.sqrt((lo + 1.0) * (lo + 2.0)) - 1.0, lo + 1.0, lo)


# ---------------------------------------------------------------------------
# B. PertGenInferenceBundle
# ---------------------------------------------------------------------------

class PertGenInferenceBundle:
    """A trained GENESIS run, ready to generate cells for named conditions.

    GENESIS embeds each condition variable (cell line, drug, dose, plate, ...)
    through an index table, and the run's ``encoder_idx_to_name.json`` lists each
    table's rows in order. Generating is therefore two steps: look each value up
    (:meth:`encode_condition`), then :meth:`generate` from the index tensors::

        bundle = PertGenInferenceBundle.from_run(run_dir)
        idx = {f"{v}_idx": torch.full((n,), bundle.encode_condition(v, val), device=bundle.device)
               for v, val in condition.items()}
        cells = bundle.generate(generator=g, **idx)      # (n, n_genes), log1p
    """

    output_space = ModelOutputSpace.LOG1P

    def __init__(self, *, run_dir: Path, checkpoint_path: Path, config: PertGenConfig,
                 gene_names: Sequence[str], encoder_idx_to_name: Mapping[str, Sequence[Any]],
                 model: torch.nn.Module, device: torch.device) -> None:
        from ..eval.cache.eval_cache_io import gene_set_hash  # lazy: src.eval imports this module

        self.run_dir, self.checkpoint_path, self.config = run_dir, checkpoint_path, config
        self.gene_names = list(gene_names)
        self.gene_set_hash = gene_set_hash(self.gene_names)
        self.model_id = hashlib.sha256(f"{run_dir.name}::{checkpoint_path.name}".encode()).hexdigest()[:16]
        self.encoder_idx_to_name = dict(encoder_idx_to_name)
        self.condition_variables = [v for v in config.model_conditioning_variables if v in encoder_idx_to_name]
        self._index = {v: {condition_key(v, name): i for i, name in enumerate(encoder_idx_to_name[v])}
                       for v in self.condition_variables}
        self.device = device
        self.compiled = config.compile_model  # a compiled graph wants every batch the same shape
        self._model = model

    @classmethod
    def from_run(cls, run_dir: Union[str, Path], *, checkpoint: Optional[Union[str, Path]] = None,
                 device: Union[str, torch.device] = "cuda", use_compile: bool = False) -> "PertGenInferenceBundle":
        """Load a run directory: its checkpoint, ``encoder_idx_to_name.json`` and ``output_gene_names.txt``.

        ``checkpoint`` is a path, or ``"last"``/``None`` for the newest ``*.ckpt``.
        ``use_compile`` compiles the model with the run's own ``compile_mode``, as in training.
        """
        device = torch.device(device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(f"GENESIS inference runs on CUDA; got device={device}, "
                               f"cuda available={torch.cuda.is_available()}")
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        run = Path(run_dir).resolve()
        ckpt = resolve_run_checkpoint(run, checkpoint)
        encoders = json.loads((run / "encoder_idx_to_name.json").read_text(encoding="utf-8"))
        gene_names = [g for g in map(str.strip, (run / "output_gene_names.txt").read_text(encoding="utf-8").splitlines()) if g]

        config = _config_from_checkpoint(ckpt)
        config.compile_model = bool(use_compile)
        lm = PertGenLightningModule(config)
        lm.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"], strict=True)
        model = lm.model.to(device).eval()
        _check_encoder_tables(model, encoders)

        logger.info("GENESIS bundle | run=%s ckpt=%s genes=%d conditions=%s compiled=%s",
                    run, ckpt.name, len(gene_names), list(encoders), config.compile_model)
        return cls(run_dir=run, checkpoint_path=ckpt, config=config, gene_names=gene_names,
                   encoder_idx_to_name=encoders, model=model, device=device)

    def encode_condition(self, var: str, value: Any) -> int:
        """Row of ``value`` in the ``var`` table; raises on a value the run never trained on."""
        if var not in self._index:
            raise KeyError(f"{var!r} is not a conditioning variable of this run: {self.condition_variables}")
        key = condition_key(var, value)
        if key not in self._index[var]:
            raise KeyError(f"{var} {value!r} (key {key!r}) is not in the run's encoder, "
                           f"e.g. {list(self._index[var])[:3]}")
        return self._index[var][key]

    @torch.inference_mode()
    def generate(self, noise: Optional[torch.Tensor] = None, *, generator: Optional[torch.Generator] = None,
                 **condition_idx: torch.Tensor) -> torch.Tensor:
        """``(n, n_genes)`` fp32 log1p cells, one per row of the ``<var>_idx`` tensors.

        Noise is drawn here from ``generator`` unless given; the streaming eval passes
        its own, seeded per condition. bf16 autocast, as in training.
        """
        if noise is None:
            n = len(next(iter(condition_idx.values())))
            noise = sample_condition_noise(self.config, n, device=self.device, generator=generator)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return self._model(noise=noise, **condition_idx)["predictions"].float()

    def release(self) -> None:
        """Drop the model so its GPU memory can be reclaimed."""
        self._model = None
        torch.cuda.empty_cache()


def _check_encoder_tables(model: torch.nn.Module, encoder_idx_to_name: Mapping[str, Sequence[Any]]) -> None:
    """Every embedding table must have exactly as many rows as its encoder list.

    Both come from one datamodule at training time, so a mismatch means the encoder
    JSON and the checkpoint come from different runs: every index would address the
    wrong embedding row, and the predictions would be plausible, wrong and silent.
    """
    bad = [f"{v}: encoder JSON has {len(encoder_idx_to_name.get(v, ()))} categories, "
           f"the checkpoint's table {t.num_embeddings} rows"
           for v, t in model.condition_embedding.embeddings.items()
           if len(encoder_idx_to_name.get(v, ())) != t.num_embeddings]
    if bad:
        raise ValueError("encoder_idx_to_name.json does not match the checkpoint:\n  " + "\n  ".join(bad))


__all__ = ["ModelOutputSpace", "PertGenInferenceBundle", "to_raw_integer"]
