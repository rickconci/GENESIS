"""Cell-set train DataLoader: C conditions × m_c cells × G genes from eval_cache.

Each step yields absolute cell sets (perturbed or control). Condition keys
(``model_conditioning_variables``, ``control_condition``) come from the dataset
registry (``core_util_data/datasets/*.yaml``) via config hydration.

Sets are **ragged**: ``use_cell_set`` caps a condition's cells but never tops a
short one up, so ``m_c`` is whatever that condition actually has. Rows are
concatenated and the partition travels alongside them as ``cell_counts``, which
is what lets the training step skip rediscovering groups from the row codes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Mapping

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from ..data_utils import _numpy_batch_to_tensor
from ..sampling.eval_cache_cell_sets import (
    CellSetCondition,
    cell_set_log,
    encode_scalars,
    load_cell_set_conditions,
    sample_cells,
)

if TYPE_CHECKING:
    from ..datamodule import PertGenDataModule




def _epoch_batches(
    n_conds: int,
    batch_size: int,
    *,
    seed: int,
    epoch: int,
    rank: int,
    world_size: int,
    drop_last: bool,
    debug: bool = False,
) -> List[np.ndarray]:
    """Shuffle train conditions each epoch; chunk into batches of size B."""
    rng = np.random.default_rng(int(seed) + int(epoch))
    order = rng.permutation(int(n_conds))
    if world_size > 1:
        order = order[rank::world_size]
    bs = int(batch_size)
    batches: List[np.ndarray] = []
    for i in range(0, len(order), bs):
        chunk = order[i : i + bs]
        if drop_last and len(chunk) < bs:
            continue
        batches.append(chunk)
    if not batches:
        raise ValueError(
            f"cell_set: no batches (n_conds={n_conds}, batch_size={bs}, "
            f"rank={rank}, world_size={world_size}).",
        )
    cell_set_log(
        debug,
        "epoch_batches n_conds=%d B=%d n_steps=%d epoch_seed=%d rank=%d/%d first_batch=%s",
        n_conds, bs, len(batches), int(seed) + int(epoch), rank, world_size,
        batches[0].tolist(),
    )
    return batches


class CellSetDataset(Dataset):
    """One dataset item = one training batch of B cell sets (pert or control)."""

    def __init__(
        self,
        reader: Any,
        conditions: tuple[CellSetCondition, ...],
        encoders: Mapping[str, Any],
        *,
        batch_size: int,
        max_cells: int,
        seed: int,
        drop_last: bool = True,
        batch_size_value: Any | None = None,
        debug: bool = False,
    ) -> None:
        self._reader = reader
        self._conditions = conditions
        self._debug = bool(debug)
        # Model codes from full condition keys (DMSO etc. already in pert_keys for controls).
        self._codes = [encode_scalars(encoders, c.pert_keys) for c in conditions]
        self._idx_keys = list(self._codes[0].keys())
        self._batch_size = int(batch_size)
        self._batch_size_value = batch_size_value
        self._max_cells = int(max_cells)
        self._seed = int(seed)
        self._drop_last = bool(drop_last)
        self._epoch = 0
        self._batches: List[np.ndarray] = []
        cell_set_log(
            self._debug,
            "dataset init n_conditions=%d C=%d max_cells=%d idx_keys=%s",
            len(conditions), self._batch_size, self._max_cells, self._idx_keys,
        )
        self._rebuild()

    def _cur_bs(self) -> int:
        if self._batch_size_value is not None:
            return int(self._batch_size_value.value)
        return self._batch_size

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch == self._epoch and self._batches:
            return
        self._epoch = epoch
        self._rebuild()

    def _rebuild(self) -> None:
        rank, world = 0, 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world = torch.distributed.get_world_size()
        self._batches = _epoch_batches(
            len(self._conditions),
            self._cur_bs(),
            seed=self._seed,
            epoch=self._epoch,
            rank=rank,
            world_size=world,
            drop_last=self._drop_last,
            debug=self._debug,
        )

    def __len__(self) -> int:
        return len(self._batches)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        cond_ids = self._batches[int(index)]
        C, cap = len(cond_ids), self._max_cells
        base_seed = self._seed + self._epoch * 1_000_003 + int(index) * 10_007
        cell_set_log(
            self._debug,
            "__getitem__ step=%d epoch=%d cond_ids=%s C=%d cap=%d base_seed=%d",
            index, self._epoch, cond_ids.tolist(), C, cap, base_seed,
        )
        # One code per CONDITION, not per row: the loader already knows the
        # partition, and `cell_counts` carries it to the loss intact.
        idx_out = {k: np.empty(C, dtype=np.int64) for k in self._idx_keys}
        counts = np.empty(C, dtype=np.int64)
        budgets = [self._conditions[int(cid)].budget for cid in cond_ids]
        if any(b is None for b in budgets) and any(b is not None for b in budgets):
            raise ValueError("cell_set: some units carry a budget and some do not")
        chunks: List[np.ndarray] = []

        for j, cid in enumerate(cond_ids):
            cid = int(cid)
            cond = self._conditions[cid]
            cell_set_log(
                self._debug,
                "  set j=%d cid=%d is_control=%s keys=%s codes=%s",
                j, cid, cond.is_control, cond.pert_keys, self._codes[cid],
            )
            # Controls live in control_cache (match keys); perts in perturbed_cache.
            ge = sample_cells(
                self._reader,
                cond.ctrl_keys if cond.is_control else cond.pert_keys,
                cap,
                perturbed=not cond.is_control,
                seed=base_seed + j,
                debug=self._debug,
                budget=cond.budget,
                subset_seed=self._seed,
            )
            chunks.append(ge)
            counts[j] = ge.shape[0]
            for k, code in self._codes[cid].items():
                idx_out[k][j] = code

        ge = np.concatenate(chunks, axis=0)
        out = {
            "gene_expression": ge,
            "lib_size": ge.sum(axis=-1, dtype=np.float32),
            "cell_counts": counts,
            **idx_out,
        }
        if budgets[0] is not None:                  # each condition's loss weight
            out["cell_budget"] = np.asarray(budgets, dtype=np.int64)
        cell_set_log(
            self._debug,
            "  batch out ge=%s lib_size[%0.1f..%0.1f] m[min=%d max=%d sum=%d]",
            ge.shape, float(out["lib_size"].min()), float(out["lib_size"].max()),
            int(counts.min()), int(counts.max()), int(counts.sum()),
        )
        for k in self._idx_keys[:4]:
            cell_set_log(self._debug, "  idx %s head=%s", k, out[k][: min(4, C)].tolist())
        return out


class CellSetEpochCallback(pl.Callback):
    """Reshuffle condition order at each training epoch."""

    def on_train_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        ds = getattr(trainer.datamodule, "_cell_set_dataset", None)
        if ds is not None:
            cell_set_log(getattr(ds, "_debug", False), "set_epoch -> %d", trainer.current_epoch)
            ds.set_epoch(trainer.current_epoch)


def _cell_set_collate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Unwrap batch_size=1 DataLoader; module-level so spawn can pickle it."""
    if len(rows) != 1:
        raise ValueError(f"cell_set collate expects one dataset item; got {len(rows)}")
    return _numpy_batch_to_tensor(rows[0])


def cell_set_train_dataloader(dm: "PertGenDataModule") -> DataLoader:
    """Build train DataLoader for ``train_loader_mode='cell_set'``."""
    import os

    os.environ["PERTGEN_NUM_WORKERS"] = str(int(dm.config.num_workers))
    debug = bool(getattr(dm.config, "cell_set_debug", False))
    reader, conditions = load_cell_set_conditions(dm, debug=debug)
    cap = int(getattr(dm.config, "use_cell_set", 256))
    dataset = CellSetDataset(
        reader,
        conditions,
        dm.encoders,
        batch_size=dm.batch_size,
        max_cells=cap,
        seed=int(dm.config.seed),
        drop_last=True,
        batch_size_value=getattr(dm, "_shared_batch_size", None),
        debug=debug,
    )
    dm._cell_set_dataset = dataset
    use_workers = dm.config.num_workers > 0
    ctx = ("spawn" if dm._uses_spawn_dataloader() else "fork") if use_workers else None
    print(
        f"  Data: cell_set DataLoader (eval_cache; C={dm.batch_size}, "
        f"max_cells={cap}; n_steps/epoch={len(dataset)}; "
        f"num_workers={dm.config.num_workers})",
        flush=True,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        num_workers=dm.config.num_workers,
        pin_memory=True,
        collate_fn=_cell_set_collate,
        persistent_workers=use_workers,
        multiprocessing_context=ctx,
        worker_init_fn=dm._worker_init_fn if use_workers else None,
    )
