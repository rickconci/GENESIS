"""PertGen training callbacks.

PertGen trains per cell but is scored per condition. A training batch is a bag
of single cells; every eval metric compares all predicted cells of one
condition with its true cells and the control pool, K times. No per-cell batch
can compute that, so there is no validation or test DataLoader. Instead
``ModelEvalCallback`` runs the genesis scoring pipeline (the same one as
``scripts/eval/score.py genesis``) in process between epochs.

Running eval in process is why this file kills processes: the persistent train
workers are still alive, and the eval truth has to fit beside them.

``DataFractionCheckpoint`` is unrelated: it saves every fraction of the train
set seen.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import signal
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional

import pytorch_lightning as pl
import torch

logger = logging.getLogger("pertgen")


# --- Inline eval: score whole conditions between epochs -----------------

def should_run_eval_callback(epoch: int, every_n_epochs: int) -> bool:
    """True when ``epoch`` (0-based, as in Lightning) should trigger model eval."""
    every = max(1, int(every_n_epochs))
    return (int(epoch) + 1) % every == 0


def flatten_eval_summary_for_wandb(
    summary: Mapping[str, Any],
    *,
    split: str,
    prefix: str = "eval_callback",
) -> dict[str, float]:
    """Flatten ``metrics_summary.json`` overall means for Lightning/wandb logging."""
    from src.eval.metrics.reporting import flatten_clustered_metrics

    out: dict[str, float] = {}
    mean_metrics = summary.get("mean_metrics")
    if isinstance(mean_metrics, Mapping):
        for key, value in flatten_clustered_metrics(mean_metrics).items():
            safe = str(key).replace("/", "_")
            out[f"{prefix}/{split}/{safe}"] = float(value)

    for meta_key in ("n_conditions", "n_skipped", "forward_seconds"):
        if meta_key in summary:
            out[f"{prefix}/{split}/{meta_key}"] = float(summary[meta_key])
    return out


class ModelEvalCallback(pl.Callback):
    """Run streaming model eval between training epochs on the one training GPU.

    Drives :func:`src.eval.orchestration.model_eval.run_model_eval` — the same
    pipeline as ``scripts/eval/score.py genesis`` —
    after selected epochs.  Overall per-split metrics from
    ``metrics_summary.json`` are logged to wandb under
    ``eval_callback/<split>/``.

    Persistent train workers are torn down first. They keep prefetching the CSR
    cache otherwise, and the eval truth does not fit beside that.

    Requires ``dm.eval_manifest`` and NPZ eval truth.  Register **after**
    ``ModelCheckpoint`` so the eval snapshot reflects the epoch that just
    finished.

    Args:
        workdir: Training run directory (contains ``checkpoints/``, ``eval/``).
        every_n_epochs: Run eval every N completed epochs (default 1).
        splits: Eval splits to score; ``None`` → validation + test.
    """

    def __init__(
        self,
        workdir: str | Path,
        *,
        every_n_epochs: int = 1,
        splits: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.workdir = Path(workdir).resolve()
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.splits = list(splits) if splits is not None else None
        # D-state workers that survived per-epoch SIGKILL accumulate here and
        # are reaped in ``teardown`` once their blocked HDF5 syscalls drain.
        self._zombie_pids: list[int] = []

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        epoch = int(trainer.current_epoch)
        if not should_run_eval_callback(epoch, self.every_n_epochs):
            return
        self._run_eval(trainer, pl_module, epoch=epoch)

    def _run_eval(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *,
        epoch: int,
    ) -> None:
        config = pl_module.config
        dm = trainer.datamodule
        if dm is None:
            raise RuntimeError(
                "ModelEvalCallback requires trainer.datamodule with eval_manifest."
            )
        manifest = getattr(dm, "eval_manifest", None)
        if manifest is None:
            raise RuntimeError(
                "ModelEvalCallback: eval_manifest missing; call dm.setup() first."
            )
        truth_dir = getattr(manifest, "eval_truth_dir", None)
        if truth_dir is None or not Path(truth_dir).is_dir():
            raise FileNotFoundError(
                "ModelEvalCallback: eval truth cache missing. "
                "Set build_eval_cache=true and rerun setup, or build via "
                "scripts/data/build_cache.py eval."
            )

        from src.data.cache.eval_cache_build import DEFAULT_EVAL_SPLITS
        from src.eval.orchestration.log_utils import headline_metrics
        from src.eval.orchestration.model_eval import run_model_eval

        splits = self.splits or list(config.model_eval_splits or DEFAULT_EVAL_SPLITS)
        # Snapshot weights from the epoch that just finished. Do not rely on
        # checkpoints/last.ckpt — with DataFractionCheckpoint-only saves it may be
        # missing and resolve_run_checkpoint would reuse a stale fraction ckpt.
        ckpt_dir = self.workdir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt = ckpt_dir / f"eval_epoch_{epoch:04d}.ckpt"
        trainer.save_checkpoint(str(ckpt))
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"ModelEvalCallback: failed to write eval checkpoint {ckpt}"
            )
        out_root = self.workdir / "eval" / "callback" / f"epoch_{epoch:04d}"
        device = config.model_eval_device
        train_device = pl_module.device

        logger.info(
            "[ModelEvalCallback] epoch=%d | splits=%s | ckpt=%s | out=%s",
            epoch,
            splits,
            ckpt.name,
            out_root,
        )
        print(
            f"  [eval_callback] epoch={epoch} starting model eval "
            f"| splits={splits} | ckpt={ckpt.name}",
            flush=True,
        )

        t0 = time.perf_counter()
        # Leave the train module on GPU. ``.to("cpu")`` invalidates CUDA graphs
        # (addresses move) and ``compiler.reset()`` throws them away — both
        # force a 30+ min recapture next epoch. Eval runs eager in leftover VRAM.

        self._zombie_pids.extend(_release_train_workers(trainer))

        wandb_metrics: dict[str, float] = {}
        try:
            for split in splits:
                result = run_model_eval(
                    config=config,
                    manifest=manifest,
                    eval_split=split,
                    run_dir=self.workdir,
                    out_dir=out_root / split,
                    checkpoint=ckpt,
                    device=device,
                    batch_size=config.model_eval_batch_size,
                    seed=config.seed,
                    min_cells=config.eval_min_cells,
                    overwrite=True,
                    pipeline_depth=1,
                    metrics_workers=0,
                    sync_cuda_streams=True,
                    use_compile=False,  # never compile next to the trainer's CUDA graphs
                    pad_last_batch=True,
                    persist_predictions=config.model_eval_persist_predictions,
                )
                if not result.metrics_path.is_file():
                    raise FileNotFoundError(
                        f"ModelEvalCallback: metrics missing at {result.metrics_path}"
                    )
                with result.metrics_path.open() as f:
                    summary = json.load(f)
                wandb_metrics.update(flatten_eval_summary_for_wandb(summary, split=split))
                mean_line = headline_metrics(summary.get("mean_metrics", {}))
                print(
                    f"  [eval_callback] epoch={epoch} {split} | "
                    f"conds={result.n_conditions} skipped={result.n_skipped} | "
                    f"{mean_line} -> {result.metrics_path}",
                    flush=True,
                )
        finally:
            # Release drops this before eval. Repeat so a crash mid-shutdown
            # still clears the fit loop.
            _drop_train_loader(trainer)
            if train_device.type == "cuda":
                torch.cuda.empty_cache()

        elapsed = time.perf_counter() - t0
        wandb_metrics["eval_callback/wall_seconds"] = float(elapsed)
        pl_module.log_dict(
            wandb_metrics,
            on_epoch=True,
            prog_bar=False,
            rank_zero_only=True,
        )
        logger.info(
            "[ModelEvalCallback] epoch=%d finished in %.1fs | logged %d scalars",
            epoch,
            elapsed,
            len(wandb_metrics),
        )
        print(
            f"  [eval_callback] epoch={epoch} done in {elapsed:.1f}s "
            f"| logged {len(wandb_metrics)} scalars to wandb",
            flush=True,
        )

    def teardown(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: str,
    ) -> None:
        # Final reap of D-state workers accumulated across eval boundaries.
        # Runs even on exception (Lightning calls teardown in the trainer's
        # finally block), which is the whole point: we don't want to leak
        # process-tree garbage if the run crashes mid-eval.
        if stage != "fit" or not self._zombie_pids:
            return
        n_before = len(set(self._zombie_pids))
        leaked = _reap_stuck_workers(self._zombie_pids)
        if leaked:
            logger.warning(
                "[ModelEvalCallback] %d/%d D-state worker(s) still alive at "
                "fit teardown (kernel I/O never drained): pids=%s",
                len(leaked),
                n_before,
                leaked,
            )
        else:
            logger.info(
                "[ModelEvalCallback] reaped %d D-state worker(s) at fit teardown",
                n_before,
            )
        self._zombie_pids = []


# --- Releasing the train workers ---------------------------------------------
# persistent_workers keeps the spawn workers alive across epochs, and Lightning
# drops the iterator before its own shutdown, so they keep prefetching the CSR
# cache through eval. A polite join also aborts inside HDF5, which torch treats
# as a fatal worker death. So: ask torch to shut down, SIGKILL whoever is left,
# and remember pids stuck in a disk read until fit ends. Listed in call order.

def _release_train_workers(trainer: pl.Trainer) -> list[int]:
    """Stop persistent train workers. Returns pids still stuck in a disk read.

    Only ``ModelEvalCallback._run_eval`` calls this. With ``use_eval_callback``
    false the workers stay up for the whole run.
    """
    root = os.getpid()
    before = _descendant_count(root)
    iters = _train_worker_iters(trainer)
    pids = [
        int(worker.pid)
        for it in iters
        for worker in it._workers
        if getattr(worker, "pid", None) is not None
    ]
    errors: list[str] = []
    for it in iters:
        if getattr(it, "_shutdown", False):
            continue
        try:
            it._shutdown_workers()
        except RuntimeError as exc:
            # torch joins after marking itself shut down, and its finally already
            # disarms the SIGCHLD handler. The HDF5 abort is the usual path.
            errors.append(f"{type(exc).__name__}: {exc}")
    fetcher = getattr(trainer.fit_loop, "_data_fetcher", None)
    if fetcher is not None:
        try:
            fetcher.teardown()
        except Exception as exc:
            errors.append(f"fetcher.teardown: {type(exc).__name__}: {exc}")

    stuck = _kill_workers(pids)
    _drop_train_loader(trainer)
    after = _descendant_count(root)
    if errors:
        logger.warning(
            "[eval_callback] train-worker shutdown raised; killed the pids anyway: %s",
            "; ".join(errors),
        )
    if stuck:
        logger.warning(
            "[eval_callback] workers still blocked in HDF5; reaping at fit end: pids=%s",
            stuck,
        )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print(
        f"  [eval_callback] released train workers "
        f"(descendant procs {before} -> {after}"
        + (f", still blocked={stuck}" if stuck else "")
        + ")",
        flush=True,
    )
    return stuck


def _kill_workers(pids: list[int], *, grace_s: float = 10.0) -> list[int]:
    """Wait out torch's shutdown, then SIGTERM, then SIGKILL.

    Pids still alive are blocked in uninterruptible I/O. The SIGKILL stays
    queued until that HDF5 read returns.
    """
    unique = sorted(set(pids))
    deadline = time.time() + grace_s
    while time.time() < deadline:
        if not _alive(unique):
            return []
        time.sleep(0.1)
    _signal(unique, signal.SIGTERM)
    time.sleep(0.5)
    _signal(unique, signal.SIGKILL)
    time.sleep(0.2)
    return _alive(unique)


def _reap_stuck_workers(pids: list[int], *, wait_s: float = 60.0) -> list[int]:
    """Re-send SIGKILL until the blocked reads return, or ``wait_s`` elapses."""
    if not pids:
        return []
    unique = sorted(set(pids))
    deadline = time.time() + wait_s
    while time.time() < deadline:
        alive = _alive(unique)
        if not alive:
            return []
        _signal(alive, signal.SIGKILL)
        time.sleep(1.0)
    return _alive(unique)


def _drop_train_loader(trainer: pl.Trainer) -> None:
    """Forget the fit-loop loader. The next epoch rebuilds it in ``setup_data()``."""
    trainer.fit_loop._data_fetcher = None
    trainer.fit_loop._combined_loader = None


def _train_worker_iters(trainer: pl.Trainer) -> list[Any]:
    """The train multiprocessing iterator, in both places Lightning stores it.

    ``CombinedLoader.reset()`` can clear one ref and leave the other live.
    """
    from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter

    fit_loop = trainer.fit_loop
    holders = [getattr(fit_loop, "_combined_loader", None)]
    fetcher = getattr(fit_loop, "_data_fetcher", None)
    other = getattr(fetcher, "_combined_loader", None) if fetcher else None
    if other is not holders[0]:
        holders.append(other)

    out: list[Any] = []
    seen: set[int] = set()
    for combined in holders:
        for it in _iters_of(combined):
            if not isinstance(it, _MultiProcessingDataLoaderIter) or id(it) in seen:
                continue
            seen.add(id(it))
            out.append(it)
    return out


def _iters_of(combined: Any) -> Iterator[Any]:
    """Yield every iterator a Lightning combined loader still points at."""
    if combined is None:
        return
    mode_it = getattr(combined, "_iterator", None)
    if mode_it is not None:
        yield from getattr(mode_it, "iterators", []) or []
    for loader in combined.flattened:
        yield getattr(loader, "_iterator", None)


def _alive(pids: list[int]) -> list[int]:
    """Pids that still exist. ``kill(pid, 0)`` probes without signalling."""
    alive: list[int] = []
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        alive.append(pid)
    return alive


def _signal(pids: list[int], sig: int) -> None:
    for pid in _alive(pids):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def _descendant_count(root_pid: int) -> int:
    """Processes under ``root_pid``. The release log prints the before/after delta."""
    seen: set[int] = set()
    frontier = [root_pid]
    while frontier:
        nxt: list[int] = []
        for pid in frontier:
            children = Path(f"/proc/{pid}/task/{pid}/children")
            if not children.is_file():
                continue
            for child_s in children.read_text().split():
                child = int(child_s)
                if child in seen:
                    continue
                seen.add(child)
                nxt.append(child)
        frontier = nxt
    return len(seen)


# --- Checkpoints by fraction of the train set seen --------------------------

class DataFractionCheckpoint(pl.Callback):
    """Save checkpoints at regular fractions of the training set consumed.

    Threshold is ``samples_seen / total_train_samples`` — i.e. how many full
    passes over the training set the model has seen.  This is the natural
    run-comparable primitive: across runs with different batch sizes,
    ``n_repeats`` schedules, or curricula, "checkpoint at 0.5" always means
    "model has seen half the training set".  Thresholding on Lightning's
    ``current_epoch`` would not be comparable when curricula change how much
    data each Lightning epoch consumes.

    Filename format::

        checkpoint-dataep{data_epoch_frac:.2f}-step{global_step:06d}.ckpt

    where ``dataep1.50`` reads as "1.5 data-epochs (passes over the train set)
    consumed".  The ``checkpoint-ep*`` prefix some runs carry has the same
    semantics.

    Honest diagnostic: with ``checkpoint_data_fraction=0.5`` and
    ``max_epochs=3`` you should expect 6 saves (0.5..3.0).  If you see fewer,
    the run consumed less than ``max_epochs`` data-epochs.

    Implements ``state_dict`` / ``load_state_dict`` so ``_samples_seen`` and
    ``_next_threshold`` survive ``trainer.fit(ckpt_path=...)`` resume.

    Args:
        save_fraction: Save every this fraction of the training set (e.g. 0.5).
        total_train_samples: Train-set size (typically ``len(dm.train_indices)``).
        dirpath: Directory to write checkpoint files.
    """

    def __init__(
        self,
        save_fraction: float,
        total_train_samples: int,
        dirpath: str | Path,
    ) -> None:
        super().__init__()
        if save_fraction <= 0:
            raise ValueError(f"save_fraction must be > 0, got {save_fraction}")
        if total_train_samples <= 0:
            raise ValueError(
                f"total_train_samples must be > 0, got {total_train_samples}"
            )
        self.save_fraction = float(save_fraction)
        self.total_train_samples = int(total_train_samples)
        self.dirpath = Path(dirpath)
        self._samples_seen: int = 0
        self._next_threshold: float = float(save_fraction)

    # ---- persist across resume ----------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "samples_seen": self._samples_seen,
            "next_threshold": self._next_threshold,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._samples_seen = state_dict.get("samples_seen", 0)
        self._next_threshold = state_dict.get("next_threshold", self.save_fraction)

    # ---- per-batch tracking -------------------------------------------
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> None:
        # ``gene_expression`` is set unconditionally by the fetch transform.
        B = batch["gene_expression"].size(0)
        self._samples_seen += B
        data_epoch_frac = self._samples_seen / self.total_train_samples

        pl_module.log("train/data_epoch_frac", data_epoch_frac)
        pl_module.log("train/samples_seen", float(self._samples_seen))

        if data_epoch_frac < self._next_threshold:
            return

        ckpt_name = (
            f"checkpoint-dataep{data_epoch_frac:.2f}"
            f"-step{trainer.global_step:06d}.ckpt"
        )
        ckpt_path = self.dirpath / ckpt_name
        trainer.save_checkpoint(str(ckpt_path))

        logger.info(
            "[DataFractionCheckpoint] Saved %s  "
            "(data_epoch_frac=%.3f, samples=%d)",
            ckpt_name,
            data_epoch_frac,
            self._samples_seen,
        )
        # Advance past every threshold crossed in this batch.  Important when
        # ``save_fraction`` is small relative to a single global batch's step.
        while self._next_threshold <= data_epoch_frac:
            self._next_threshold += self.save_fraction
