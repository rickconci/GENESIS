"""Forward-loop orchestration for generating GENESIS predictions during model eval.

:func:`run_streaming_eval` is a generator that yields :class:`CompletedCondition`
records as the forward loop fills each condition's rep-major GPU buffer.  Predictions
never leave the GPU; the caller scores each condition on a second CUDA stream and drops
the reference, freeing the buffer for the next condition (bounded ring).  This is what
:mod:`src.eval.orchestration.model_eval` uses to keep generation and scoring overlapping
on a single GPU.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from ...model.inference_output import PertGenInferenceBundle
from ...model.pertgen import sample_condition_noise
from ..cache.eval_cache_io import ConditionBundle, QuadrupletRow
from .log_utils import log_detail

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Work-queue flattening
# ---------------------------------------------------------------------------


def _flatten_work_queue(
    work_queue: Sequence[QuadrupletRow],
    *,
    device: torch.device,
    categories: list[str],
) -> tuple[
    dict[str, torch.Tensor], np.ndarray, int,
]:
    """Build pinned CPU tensors of categorical indices, one row per pseudocell slot.

    Returns:
        ``(tensors_dict, row_lookup, n_total)``
        where ``row_lookup[i]`` is the work-queue position of slot ``i``.
        Tensors are ``pin_memory=True`` on CUDA for overlapping H2D copies.
    """
    n_total = sum(int(r.num_cells) for r in work_queue)
    pin = device.type == "cuda"

    tensors = {}
    for cat in categories:
        tensors[f"{cat}_idx"] = torch.empty(n_total, dtype=torch.int64, pin_memory=pin)
    row_lookup = np.empty(n_total, dtype=np.int32)

    off = 0
    for g, qrow in enumerate(work_queue):
        nc = int(qrow.num_cells)
        if nc <= 0:
            continue
        sl = slice(off, off + nc)
        cond_idx = qrow.condition_idx or {}
        for cat in categories:
            idx = int(cond_idx.get(cat, -1))
            if idx < 0:
                raise ValueError(
                    f"QuadrupletRow has unencoded index for '{cat}' "
                    f"(condition_keys={qrow.condition_keys!r}). "
                    "Pass a PertGenInferenceBundle to load_work_queue so "
                    "condition_idx is populated."
                )
            tensors[f"{cat}_idx"][sl] = idx
        row_lookup[sl] = g
        off += nc
    return tensors, row_lookup, n_total


def _validate_prediction_shape(
    pred: torch.Tensor,
    *,
    n_genes: int,
    batch_len: int,
) -> None:
    """Validate the first forward output before copying it into CPU buffers.

    Args:
        pred: Model predictions returned by :meth:`PertGenInferenceBundle.generate`.
        n_genes: Expected number of output genes.
        batch_len: Number of real examples in the current batch. With padded
            static-shape inference, ``pred.shape[0]`` may be larger than this.

    Raises:
        ValueError: If the prediction tensor is not shaped ``(batch, n_genes)``
            or does not contain all real examples from the current batch.
    """
    if pred.ndim != 2:
        raise ValueError(
            "PertGen forward returned an unexpected prediction rank: "
            f"shape={tuple(pred.shape)} expected_rank=2"
        )
    if int(pred.shape[0]) < int(batch_len):
        raise ValueError(
            "PertGen forward returned fewer rows than requested: "
            f"shape={tuple(pred.shape)} batch_len={batch_len}"
        )
    if int(pred.shape[1]) != int(n_genes):
        raise ValueError(
            "PertGen forward returned the wrong number of genes: "
            f"shape={tuple(pred.shape)} expected_genes={n_genes}"
        )


# ---------------------------------------------------------------------------
# Streaming eval pipeline (gen + score overlap)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompletedCondition:
    """One condition whose GPU rep-major prediction buffer is ready for scoring.

    ``pred_t`` lives on ``device`` and has shape ``(K * N_c, n_genes)`` in
    *rep-major* layout: ``pred_t[r * N_c : (r + 1) * N_c]`` is rep ``r``'s
    full-N prediction slice across all qrows of the condition, concatenated
    in ``cb.rows`` order.  ``k_eff`` is the largest rep count fully written
    (always ``K_plan`` in practice; smaller only if a qrow's inflated budget
    fell short due to integer rounding).

    The caller MUST issue any consumer-side CUDA work on a stream that
    waits on ``gen_event`` (``stream.wait_event(gen_event)``) so kernels
    in stream_gen finish before the buffer is read.  After scoring, the
    caller drops the reference (``del pred_t``) to let the next condition
    re-use that GPU memory.
    """

    cond_idx: int
    bundle: ConditionBundle
    pred_t: torch.Tensor
    n_truth: int
    k_eff: int
    gen_event: Optional[torch.cuda.Event]


@dataclass
class StreamingPregenStats:
    """Bookkeeping for one streaming run, filled in place as the loop runs."""

    n_quadruplets: int = 0
    n_total_cells: int = 0
    n_batches: int = 0
    n_conditions_total: int = 0
    n_conditions_yielded: int = 0
    n_overflow_slots: int = 0
    elapsed_seconds: float = 0.0

    @property
    def slots_per_second(self) -> float:
        # Throughput is computed on real (non-overflow) cells the model emitted.
        return (self.n_total_cells / self.elapsed_seconds) if self.elapsed_seconds > 0 else 0.0


@dataclass(frozen=True)
class _ScatterPlan:
    """Pre-computed mapping from global emission slot -> per-condition buffer row.

    ``dst_cond`` / ``dst_offset`` are pinned (when CUDA) so per-batch slices
    can be H2D-copied non-blocking onto the gen stream.  ``cond_sizes`` and
    ``cond_first_batch_idx`` drive lazy buffer allocation and completion
    detection in the streaming loop.
    """

    dst_cond: torch.Tensor       # int32 (n_total,) -- target cond_idx or -1
    dst_offset: torch.Tensor     # int64 (n_total,) -- row inside cond buffer
    cond_sizes: list[int]        # K * N_c per cond (rep-major buffer height)
    cond_n_truth: list[int]      # N_c per cond (so consumer knows rep stride)
    cond_k_eff: list[int]        # k_eff per cond (floor(min M_q / orig_q))
    cond_first_slot: list[int]   # first inflated-queue slot for each cond
    cond_last_slot: list[int]    # last inflated-queue slot (exclusive) per cond
    cond_bundles: list[ConditionBundle]


def _build_scatter_plan(
    *,
    inflated_queue: Sequence[QuadrupletRow],
    condition_bundles: Sequence[ConditionBundle],
    orig_n_by_id: Mapping[int, int],
    plan_k: int,
    device: torch.device,
) -> _ScatterPlan:
    """Build per-slot scatter map + per-condition layout metadata.

    Pure: no GPU forwards, just numpy bookkeeping.

    For each inflated-queue qrow with ``M_q`` cells and original ``orig_q``:
    emission slot ``j`` in ``[0, M_q)`` has rep ``r = j // orig_q`` and
    within-rep offset ``wro = j % orig_q``.  When ``r < plan_k`` it lands at
    row ``r * N_c + qrow_offset_within_rep + wro`` in condition ``c``'s
    rep-major buffer; otherwise it overflows the buffer (``dst_cond = -1``)
    and is dropped at scatter time.  In practice the inflation in
    ``_inflate_work_queue`` makes every qrow's ``M_q`` exactly ``K * orig_q``,
    so overflow is 0; we keep the dropper for defensive robustness.
    """
    # qrow position -> (cond_idx, qrow_offset_within_rep)
    qrow_to_cond: dict[int, tuple[int, int]] = {}
    cond_bundles_list = list(condition_bundles)
    cond_sizes: list[int] = []
    cond_n_truth: list[int] = []
    cond_k_eff: list[int] = []
    for c_idx, cb in enumerate(cond_bundles_list):
        n_c = sum(int(orig_n_by_id[id(q)]) for q in cb.rows)
        cond_n_truth.append(n_c)
        cond_sizes.append(int(plan_k) * n_c)
        # k_eff is the smallest #full reps across qrows in this condition;
        # equals plan_k whenever the inflation budget hit K*orig_q per qrow
        # (always true given ``_inflate_work_queue``, but we report defensively).
        k_caps: list[int] = []
        for q in cb.rows:
            orig_q = int(orig_n_by_id[id(q)])
            if orig_q <= 0:
                k_caps.append(0)
                continue
            k_caps.append(int(q.num_cells) // orig_q)
        cond_k_eff.append(min(plan_k, min(k_caps) if k_caps else 0))
        wro_running = 0
        for q in cb.rows:
            qrow_to_cond[id(q)] = (c_idx, wro_running)
            wro_running += int(orig_n_by_id[id(q)])

    # Per inflated qrow, lay out emission slots into the global scatter map.
    n_total = sum(int(q.num_cells) for q in inflated_queue)
    dst_cond_np = np.full(n_total, -1, dtype=np.int32)
    dst_offset_np = np.zeros(n_total, dtype=np.int64)
    cond_first_slot = [n_total] * len(cond_bundles_list)
    cond_last_slot = [0] * len(cond_bundles_list)

    cursor = 0
    for q in inflated_queue:
        m_q = int(q.num_cells)
        if m_q <= 0:
            continue
        c_idx, qrow_off = qrow_to_cond[id(q)]
        n_c = cond_n_truth[c_idx]
        orig_q = int(orig_n_by_id[id(q)])
        # Compute (r, wro) for each emission slot via cheap arange math; mask
        # out reps that overflow the (K_plan * N_c) buffer.
        local = np.arange(m_q, dtype=np.int64)
        r = local // max(orig_q, 1)
        wro = local % max(orig_q, 1)
        valid = (r < int(plan_k)) & (orig_q > 0)
        dst_cond_np[cursor : cursor + m_q] = np.where(valid, c_idx, -1).astype(np.int32)
        dst_offset_np[cursor : cursor + m_q] = np.where(
            valid, r * n_c + qrow_off + wro, 0,
        )
        cond_first_slot[c_idx] = min(cond_first_slot[c_idx], cursor)
        cond_last_slot[c_idx] = max(cond_last_slot[c_idx], cursor + m_q)
        cursor += m_q

    pin = device.type == "cuda"
    dst_cond_t = torch.from_numpy(dst_cond_np)
    dst_offset_t = torch.from_numpy(dst_offset_np)
    if pin:
        dst_cond_t = dst_cond_t.pin_memory()
        dst_offset_t = dst_offset_t.pin_memory()
    return _ScatterPlan(
        dst_cond=dst_cond_t,
        dst_offset=dst_offset_t,
        cond_sizes=cond_sizes,
        cond_n_truth=cond_n_truth,
        cond_k_eff=cond_k_eff,
        cond_first_slot=cond_first_slot,
        cond_last_slot=cond_last_slot,
        cond_bundles=cond_bundles_list,
    )


def _scatter_pred_into_cond_buffers(
    *,
    pred: torch.Tensor,
    batch_dst_cond: torch.Tensor,
    batch_dst_offset: torch.Tensor,
    cond_buffers: Mapping[int, torch.Tensor],
) -> None:
    """Route a batch's predictions into the active per-condition GPU buffers.

    A batch typically spans 1-2 conditions (work queue is sorted by
    condition keys, qrows within a condition are contiguous), so the
    per-condition loop here is short and cheap.  We use boolean masking +
    ``index_copy_`` so the actual GPU memcpy is a single contiguous range
    per condition (since dst_offset is monotonically increasing within a
    qrow's contiguous slots).
    """
    # GPU-side: get unique cond_idx values in this batch (drop -1 sentinels).
    uniques = torch.unique(batch_dst_cond)
    for c_tensor in uniques.tolist():
        c_idx = int(c_tensor)
        if c_idx < 0:
            continue
        buf = cond_buffers.get(c_idx)
        if buf is None:
            raise RuntimeError(
                f"Streaming scatter: condition buffer missing for cond_idx={c_idx}; "
                f"the scheduler must allocate the buffer before its first batch.",
            )
        mask = batch_dst_cond == c_idx
        # index_copy_ requires int64 indices; dst_offset is already int64.
        buf.index_copy_(0, batch_dst_offset[mask], pred.index_select(0, mask.nonzero(as_tuple=False).squeeze(1)))


# Per-condition noise (model-eval path).
#
# Each (work-queue row, repeat) draws its noise from its own seed, so a condition's cells
# do not depend on which other conditions share the run, their order or the batch size:
# one condition scored alone gets the same cells as inside the full queue.

def condition_noise_seed(base_seed: int, condition_keys: Mapping[str, Any], rep: int) -> int:
    """Stable 63-bit seed for one (condition, repeat); independent of key order."""
    raw = repr((int(base_seed), sorted((str(k), str(v)) for k, v in condition_keys.items()), int(rep)))
    return int.from_bytes(hashlib.blake2b(raw.encode(), digest_size=8).digest(), "little") & ((1 << 63) - 1)


def _noise_segments(inflated_queue: Sequence[QuadrupletRow], orig_n_by_id: Mapping[int, int],
                    base_seed: int, seed_of: Optional[Callable[[QuadrupletRow, int], int]] = None,
                    ) -> tuple[np.ndarray, list[int]]:
    """Slot-stream layout as segment starts (+ the end) and one seed per segment.

    Mirrors ``_build_scatter_plan``: qrow ``q`` emits ``M_q`` consecutive slots, slot
    ``j`` being repeat ``j // orig_q``; empty qrows emit nothing. ``seed_of(q, rep)``
    lets a caller bring its own seed per segment (a cloud request carries one);
    the default is :func:`condition_noise_seed`.
    """
    starts: list[int] = []
    seeds: list[int] = []
    cursor = 0
    for q in inflated_queue:
        m_q, orig_q = int(q.num_cells), max(int(orig_n_by_id[id(q)]), 1)
        if m_q <= 0:
            continue
        for rep, off in enumerate(range(0, m_q, orig_q)):
            starts.append(cursor + off)
            seeds.append(condition_noise_seed(base_seed, q.condition_keys, rep) if seed_of is None
                         else int(seed_of(q, rep)))
        cursor += m_q
    return np.asarray(starts + [cursor], dtype=np.int64), seeds


def _segment_noise(bundle: PertGenInferenceBundle, starts: np.ndarray, seeds: list[int], lo: int, hi: int,
                   eff_len: int, gen: torch.Generator, cache: dict) -> torch.Tensor:
    """Noise for slots ``[lo, hi)`` (+ zero padding to ``eff_len``).

    Each segment is always drawn whole and then sliced: ``randn(n)[:k]`` is not
    ``randn(k)`` (torch picks its sampling kernel by size), so drawing just the part a
    batch needs would make a draw depend on where the batch edge falls. ``cache`` keeps
    the last segment, which the next batch usually continues.
    """
    parts = []
    s = int(np.searchsorted(starts, lo, side="right")) - 1
    while s < len(seeds) and starts[s] < hi:
        a, b = int(starts[s]), int(starts[s + 1])
        if cache.get("seg") != s:
            gen.manual_seed(seeds[s])
            cache.update(seg=s, noise=sample_condition_noise(bundle.config, b - a, device=bundle.device, generator=gen))
        parts.append(cache["noise"][max(lo - a, 0): min(hi, b) - a])
        s += 1
    noise = torch.cat(parts, dim=0)
    if noise.shape[0] != hi - lo:
        raise RuntimeError(f"noise for slots [{lo}, {hi}) has {noise.shape[0]} rows")
    if eff_len > hi - lo:   # padded tail slots: their outputs are discarded
        noise = torch.cat([noise, noise.new_zeros(eff_len - (hi - lo), noise.shape[1])], dim=0)
    return noise


def run_streaming_eval(
    *,
    bundle: PertGenInferenceBundle,
    inflated_queue: Sequence[QuadrupletRow],
    condition_bundles: Sequence[ConditionBundle],
    orig_n_by_id: Mapping[int, int],
    plan_k: int,
    batch_size: int,
    seed: int = 42,
    pad_last_batch: bool = False,
    pipeline_depth: int = 3,
    stream_gen: Optional[torch.cuda.Stream] = None,
    progress: bool = True,
    release_model_after: bool = True,
    stats_out: Optional[StreamingPregenStats] = None,
    progress_label: str = "Stream eval forward",
    seed_of: Optional[Callable[[QuadrupletRow, int], int]] = None,
) -> Iterator[CompletedCondition]:
    """Yield :class:`CompletedCondition` records as the forward pass fills them.

    Design overview:

    1. Pre-compute the ``(global_slot -> (cond_idx, dst_row))`` scatter map and
       per-condition completion windows ``[first_slot, last_slot)`` so we
       know exactly when each condition becomes complete from the batch loop.
    2. Run the global batched forward on ``stream_gen`` (default: a freshly
       created CUDA stream so generation kernels can overlap consumer-side
       scoring kernels on the default stream).
    3. After each batch, scatter the output rows into per-condition rep-major
       GPU buffers; lazily allocate a buffer the first time we write into it,
       capped to ``pipeline_depth`` simultaneously live buffers.
    4. When a condition's last_slot has been written, record a per-condition
       ``cuda.Event`` on stream_gen and yield it.  The consumer is expected
       to ``wait_event`` on its own stream before reading ``pred_t``, and to
       drop the reference once scoring completes so the buffer can be GC'd.

    Args:
        bundle: Loaded :class:`PertGenInferenceBundle`.
        inflated_queue: Output of ``_inflate_work_queue`` -- same order and
            qrow identity used to key ``orig_n_by_id``.
        condition_bundles: Output of
            ``group_by_condition(inflated_queue, eval_group_keys=...)``.
            Must list bundles in the *same order* their first qrow appears
            in ``inflated_queue`` (which ``group_by_condition`` guarantees).
        orig_n_by_id: ``id(qrow) -> original num_cells`` map from inflation.
        plan_k: ``model_eval_n_repeats`` (rep-major buffer is ``K * N_c``).
        batch_size: Forward-pass batch size.
        seed: Base seed for inference-time noise; each (condition, repeat) derives its
            own from it (:func:`condition_noise_seed`).
        pad_last_batch: Pad final batch to static shape (forced True if
            the bundle was compiled).
        pipeline_depth: Max number of conditions whose GPU buffers are
            allocated simultaneously.  The producer blocks (the generator
            stops yielding new conditions) once this cap is reached until
            the caller drops a buffer reference.
        stream_gen: Optional CUDA stream for generation.  When omitted on
            CUDA, a fresh stream is created so the producer kernels run
            concurrently with consumer scoring on the default stream.
        progress: Show a tqdm progress bar over forward batches.
        release_model_after: Free the bundle's model weights once all
            forwards complete (frees GPU memory for any remaining scoring).
        stats_out: Optional :class:`StreamingPregenStats` that the function
            fills in-place as the loop runs (so the caller can reach it
            after the generator is exhausted).
        progress_label: Label for the tqdm bar.

    Yields:
        :class:`CompletedCondition` instances in the same order conditions
        first appear in ``inflated_queue``.
    """
    device = bundle.device
    categories = bundle.condition_variables

    if bundle.compiled and not pad_last_batch:
        logger.warning(
            "Bundle is torch.compile'd but pad_last_batch=False; forcing True "
            "to keep a static shape for the final tail batch.",
        )
        pad_last_batch = True

    n_quadruplets = len(inflated_queue)
    from src.eval.orchestration.log_utils import log_step

    t_flat = time.perf_counter()
    log_step(
        logger, "stream:setup",
        "flattening inflated work queue | qrows=%d",
        n_quadruplets,
    )
    tensors_a, _row_lookup, n_total = _flatten_work_queue(
        inflated_queue, device=device, categories=categories,
    )
    log_step(
        logger, "stream:setup",
        "work queue flattened in %.1fs | slots=%d batch_size=%d",
        time.perf_counter() - t_flat, n_total, batch_size,
    )
    n_batches = (n_total + batch_size - 1) // batch_size if n_total > 0 else 0
    is_cuda = device.type == "cuda"

    t_scatter = time.perf_counter()
    scatter = _build_scatter_plan(
        inflated_queue=inflated_queue,
        condition_bundles=condition_bundles,
        orig_n_by_id=orig_n_by_id,
        plan_k=int(plan_k),
        device=device,
    )
    n_conds = len(scatter.cond_bundles)
    log_step(
        logger, "stream:setup",
        "scatter plan built in %.1fs | conditions=%d ~batches=%d K=%d pipeline_depth=%d",
        time.perf_counter() - t_scatter, n_conds, n_batches,
        int(plan_k), int(pipeline_depth),
    )

    log_detail(
        logger, "stream_pregen",
        "%d qrows, %d slots, %d conditions, ~%d batches | "
        "K=%d pipeline_depth=%d pad_last=%s",
        n_quadruplets, n_total, n_conds, n_batches,
        int(plan_k), int(pipeline_depth), pad_last_batch,
    )

    # Producer / consumer state.
    cond_buffers: dict[int, torch.Tensor] = {}
    next_cond_to_start = 0  # index into scatter.cond_bundles
    next_cond_to_yield = 0  # next condition we owe the consumer

    # Noise comes from per-(condition, repeat) generators; the global seed only pins
    # anything else that might draw from the global RNG.
    torch.manual_seed(int(seed))
    if is_cuda:
        torch.cuda.manual_seed_all(int(seed))
    seg_starts, seg_seeds = _noise_segments(inflated_queue, orig_n_by_id, int(seed), seed_of)
    noise_gen, noise_cache = torch.Generator(device=device), {}

    # Generation stream: defaults to a fresh CUDA stream so producer kernels
    # can overlap with consumer scoring on the default stream.
    owned_stream = False
    if is_cuda and stream_gen is None:
        stream_gen = torch.cuda.Stream(device=device)
        owned_stream = True

    # Reusable padded device input buffers when pad_last_batch=True.
    pad_buffers: dict[str, torch.Tensor] = {}
    if pad_last_batch and n_total > 0:
        for k in tensors_a:
            pad_buffers[k] = torch.zeros(batch_size, dtype=torch.int64, device=device)

    # Move the precomputed scatter map onto the device once (small: 12 bytes
    # per slot total) so the per-batch slices are device-resident.
    dst_cond_dev = scatter.dst_cond.to(device, non_blocking=True)
    dst_offset_dev = scatter.dst_offset.to(device, non_blocking=True)
    # The pad zeros and these copies are queued on the current stream, the batches on stream_gen:
    # without this wait batch 0 can read them half-written (its clouds then change run to run).
    if is_cuda and stream_gen is not None:
        stream_gen.wait_stream(torch.cuda.current_stream(device))
    n_genes = len(bundle.gene_names)
    checked_prediction_shape = False
    overflow_count = 0
    forward_batches_done = 0

    def _allocate_buffer_for(c_idx: int) -> torch.Tensor:
        """Create the rep-major GPU buffer for condition ``c_idx``."""
        size = scatter.cond_sizes[c_idx]
        buf = torch.empty((size, n_genes), dtype=torch.float32, device=device)
        cond_buffers[c_idx] = buf
        return buf

    def _maybe_yield_completed(batch_end: int) -> Iterator[CompletedCondition]:
        """Yield any conditions whose last slot was written at or before batch_end."""
        nonlocal next_cond_to_yield
        while next_cond_to_yield < n_conds and scatter.cond_last_slot[next_cond_to_yield] <= batch_end:
            c_idx = next_cond_to_yield
            buf = cond_buffers.pop(c_idx)
            event: Optional[torch.cuda.Event] = None
            if is_cuda:
                event = torch.cuda.Event()
                event.record(stream_gen if stream_gen is not None else torch.cuda.current_stream(device))
            yield CompletedCondition(
                cond_idx=c_idx,
                bundle=scatter.cond_bundles[c_idx],
                pred_t=buf,
                n_truth=scatter.cond_n_truth[c_idx],
                k_eff=scatter.cond_k_eff[c_idx],
                gen_event=event,
            )
            next_cond_to_yield += 1

    def _ensure_capacity_for_next_cond() -> bool:
        """Allocate buffers for any condition whose slots start in the next batch.

        Returns ``True`` when the producer can proceed; ``False`` when the
        pipeline_depth cap is hit (the caller must yield completed conditions
        before continuing).
        """
        nonlocal next_cond_to_start
        if next_cond_to_start >= n_conds:
            return True
        return len(cond_buffers) < int(pipeline_depth)

    t0 = time.perf_counter()
    starts = range(0, n_total, batch_size) if n_total > 0 else range(0, 0)
    log_step(
        logger, "stream:forward",
        "starting batched forward | ~%d batches over %d slots",
        n_batches, n_total,
    )
    bar = tqdm(
        starts,
        desc=progress_label,
        unit="batch",
        disable=(not progress) or n_total == 0,
    )

    try:
        for batch_start in bar:
            batch_end = min(batch_start + batch_size, n_total)
            batch_len = batch_end - batch_start
            eff_len = batch_size if pad_last_batch else batch_len

            # Allocate buffers for every condition whose slots begin in this
            # batch.  Before allocating beyond pipeline_depth, we opportunistically
            # yield any conditions completed by the previous batch so the
            # caller can drop their buffers; the cap is "soft" because the
            # scatter MUST have a buffer for every cond_idx that appears in
            # this batch -- we never skip allocation, only encourage early
            # back-pressure when it would help memory.
            while (
                next_cond_to_start < n_conds
                and scatter.cond_first_slot[next_cond_to_start] < batch_end
            ):
                if not _ensure_capacity_for_next_cond():
                    yield from _maybe_yield_completed(batch_start)
                _allocate_buffer_for(next_cond_to_start)
                next_cond_to_start += 1

            # Forward + scatter on stream_gen (overlaps with consumer scoring
            # on default stream).
            ctx_stream = torch.cuda.stream(stream_gen) if (is_cuda and stream_gen is not None) else _NullCtx()
            with ctx_stream:
                if pad_last_batch:
                    conditions_in: dict[str, torch.Tensor] = {}
                    for k, ta in tensors_a.items():
                        pad_buffers[k][:batch_len].copy_(ta[batch_start:batch_end], non_blocking=True)
                        if batch_len < batch_size:
                            pad_buffers[k][batch_len:].zero_()
                        conditions_in[k] = pad_buffers[k]
                else:
                    conditions_in = {
                        k: ta[batch_start:batch_end].to(device, non_blocking=True)
                        for k, ta in tensors_a.items()
                    }

                noise = _segment_noise(bundle, seg_starts, seg_seeds, batch_start, batch_end, eff_len,
                                       noise_gen, noise_cache)
                pred = bundle.generate(noise=noise, **conditions_in)
                if not checked_prediction_shape:
                    _validate_prediction_shape(
                        pred,
                        n_genes=n_genes,
                        batch_len=batch_len,
                    )
                    log_detail(
                        logger, "stream_pregen",
                        "first batch OK | pred_shape=%s genes=%d",
                        tuple(pred.shape), n_genes,
                    )
                    checked_prediction_shape = True

                batch_dst_cond = dst_cond_dev[batch_start:batch_end]
                batch_dst_offset = dst_offset_dev[batch_start:batch_end]
                # Defensive: count overflow slots once (real preds emitted that
                # the scatter map decided to drop -- 0 in the typical inflation).
                if overflow_count == 0:
                    overflow_count += int((batch_dst_cond < 0).sum().item())
                _scatter_pred_into_cond_buffers(
                    pred=pred[:batch_len],
                    batch_dst_cond=batch_dst_cond,
                    batch_dst_offset=batch_dst_offset,
                    cond_buffers=cond_buffers,
                )
                # Drop references so the GPU allocator can reuse forward
                # activations before the next launch.
                del pred, noise

            forward_batches_done += 1

            # Yield any conditions whose last slot was just written.
            yield from _maybe_yield_completed(batch_end)

    finally:
        bar.close()

    # Final drain: yield any conditions completed by the very last batch.
    yield from _maybe_yield_completed(n_total)

    elapsed = time.perf_counter() - t0
    if stats_out is not None:
        stats_out.n_quadruplets = n_quadruplets
        stats_out.n_total_cells = n_total
        stats_out.n_batches = n_batches
        stats_out.n_conditions_total = n_conds
        stats_out.n_conditions_yielded = next_cond_to_yield
        stats_out.n_overflow_slots = overflow_count
        stats_out.elapsed_seconds = elapsed

    if release_model_after:
        bundle.release()
        if is_cuda:
            alloc = torch.cuda.memory_allocated() / 1e6
            reserv = torch.cuda.memory_reserved() / 1e6
            log_detail(
                logger, "stream_pregen",
                "GPU after release | alloc=%.0f MiB reserved=%.0f MiB",
                alloc, reserv,
            )
    if owned_stream:
        # Clean up the dedicated gen stream once we're done with it.
        del stream_gen


class _NullCtx:
    """No-op context manager used when there is no CUDA stream to enter."""

    def __enter__(self) -> "_NullCtx":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None


__all__ = [
    "CompletedCondition",
    "StreamingPregenStats",
    "run_streaming_eval",
    "condition_noise_seed",
]
