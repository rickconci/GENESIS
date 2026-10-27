"""Score anything against the eval truth cache: one loop, one summary layout.

Every scored thing is a PredictionSource: GENESIS generating on the fly
(model_eval.GenesisSource), an arm reading predicted cells through a ``pred_getter``
(src/eval/arms/: baseline, npz dump, copy neighbour), the split-half reference
(truth half A vs truth half B), or no response (the condition's own controls: what a
condition an arm failed on is scored as). Everything else is shared and lives here:

  1. the work queue: the split's conditions from the condition summary;
  2. the loop: prefetch truth, stack controls, take the source's prediction, score
     the K-rep split-half protocol, optionally stream per-gene arrays to parquet;
  3. the summary: common fields + the caller's own provenance fields.

A pseudobulk-only arm (a deterministic MSE model: one log1p vector, no cells) has its own
kernel, ``score_pseudobulk_condition``: the same K=1 split-half draw, Pearson r and
Pearson Δ only.

The metric math is src/eval/orchestration/model_eval.py (_score_condition_kreps).
"""
from __future__ import annotations

import contextlib
import json
import logging
import pickle
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

import torch

from src.data.data_utils import resolve_eval_group_keys
from src.eval.cache.eval_cache_io import ConditionPrefetcher, ControlCache, group_by_condition, load_work_queue
from src.eval.cache.eval_npz_reader import EvalNpzReader
from src.eval.plots.gene_arrays import GeneArraysParquetWriter
from src.eval.metrics.eval_metrics import score_pair
from src.eval.metrics.reporting import METRIC_GROUP_ORDER, check_metric_keys, materialize_metric_tensor_batch
from src.eval.metrics.spaces import (
    DEG_SPACES,
    PSEUDOBULK_DELTA_METRIC_SPACES,
    PSEUDOBULK_METRIC_SPACES,
)
from src.eval.reference.pairs import (
    PSEUDOBULK_SHORT,
    metric_cols,
    score_true_pseudobulk,
)
from src.eval.reference.split_half import SplitHalfSampler
from src.eval.orchestration.log_utils import headline_metrics, reset_eval_metric_logs
from src.eval.orchestration.model_eval import (
    KRepResult,
    ModelEvalPlan,
    _compute_condition_extras,
    _condition_eligible_for_kreps,
    _load_condition_tensors,
    _score_condition_kreps,
    _stable_cond_seed,
    aggregate_rep_metrics,
    build_result_row,
    pregenerated_eval_plan,
    summary_overall_stats,
)
from src.eval.reference.designs import CloudRequest, dedupe
from src.eval.reference.truth_io import GroupPrefetcher, GroupRef
from src.model.inference_output import ModelOutputSpace

log = logging.getLogger(__name__)


def read_gene_names(path: Path) -> list[str]:
    """Ordered gene list from a newline text file or a pickled sequence."""
    path = Path(path)
    if path.suffix == ".pkl":
        with path.open("rb") as fh:
            names = [str(g) for g in pickle.load(fh)]
    else:
        names = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not names:
        raise SystemExit(f"Gene names file is empty: {path}")
    return names


# The work queue: which conditions get scored, and their controls

@dataclass(frozen=True)
class Work:
    rows: list                      # QuadrupletRow per (condition, plate) group
    bundles: list                   # rows grouped into scored conditions
    eval_group_keys: tuple
    condition_variables: tuple      # sorted keys of a scored condition (parquet columns)


def build_work(*, condition_summary: Path, split_yaml: Path, eval_split: str,
               condition_variables: Sequence[str], control_condition: Mapping[str, str],
               control_match_keys: Sequence[str], min_cells: int, max_conditions: Optional[int] = None,
               conditions_query: Optional[str] = None) -> Work:
    """``max_conditions``: a seeded random subset, for smoke tests and gates; None = all.
    ``conditions_query``: a pandas query on the split's summary rows (e.g. only the conditions an arm failed)."""
    rows = load_work_queue(
        condition_summary_parquet=condition_summary, split_yaml=split_yaml, eval_split=eval_split,
        condition_variables=tuple(condition_variables), control_condition=dict(control_condition) or None,
        bundle=None, min_cells=min_cells, control_match_keys=tuple(control_match_keys),
        max_conditions=max_conditions, conditions_query=conditions_query,
    )
    keys = resolve_eval_group_keys(tuple(condition_variables), dict(control_condition), tuple(control_match_keys))
    bundles = group_by_condition(rows, eval_group_keys=keys)
    if not bundles:
        raise SystemExit(f"no {eval_split} conditions in {split_yaml} with >= {min_cells} cells")
    log.info("Work queue | rows=%d conditions=%d eval_group_keys=%s", len(rows), len(bundles), keys)
    return Work(rows, bundles, tuple(keys), tuple(sorted(bundles[0].key.as_dict().keys())))


def _result_extra(payload: Any, cb: Any) -> dict[str, Any]:
    extra: dict[str, Any] = {"n_groups": len(cb.rows)}
    if getattr(payload, "per_plate_counts", None):
        extra["n_plates"] = len(payload.per_plate_counts)
        extra["per_plate_counts"] = payload.per_plate_counts
    return extra


# Prediction sources: whatever stands opposite the truth half, one per condition
#
# The loop below is the same for GENESIS, a fitted baseline, an external dump and the split-half
# reference; only where the predicted cells come from differs. A source yields one opaque handle per
# condition, in work order, and turns it into a Prediction once truth and controls are on the device.

@dataclass
class Prediction:
    pred_t: Optional[torch.Tensor]       # rep-major (K*N, G); None = the reference's other truth half
    k: int                               # reps to score
    output_space: Any                    # the space pred_t is in (after any denorm)
    extra: dict = field(default_factory=dict)       # source-specific row fields
    pred_ctrl_t: Optional[torch.Tensor] = None      # copy arm --own-controls only


class PredictionSource:
    """Base: no generation, nothing to wait on. Subclasses override ``resolve`` at least."""

    output_space: Any = ModelOutputSpace.RAW
    pred_getter: Optional[Callable] = None     # handed to the prefetcher (arms that read files)
    may_skip: bool = True                      # False: a condition it cannot score raises instead

    def handles(self, bundles: Sequence) -> Iterator[Any]:
        return iter([None] * len(bundles))

    def resolve(self, handle: Any, payload: Any, truth_t: torch.Tensor, ctrl_t: torch.Tensor,
                control_cache: ControlCache) -> Optional[Prediction]:
        raise NotImplementedError

    def scoring_stream(self):  # GENESIS scores on its own CUDA stream
        return contextlib.nullcontext()

    def on_missing_control(self, handle: Any, key_dict: Mapping[str, Any]) -> bool:
        return False  # False = re-raise the FileNotFoundError

    def close(self) -> None:
        pass


class ArmSource(PredictionSource):
    """Any ``pred_getter`` arm (src/eval/arms/: npz dump, baseline, copy neighbour): K=1."""

    def __init__(self, pred_getter: Callable, output_space: Any):
        self.pred_getter, self.output_space = pred_getter, output_space

    def resolve(self, handle, payload, truth_t, ctrl_t, control_cache):
        if payload.pred is None:
            return None
        key_dict = payload.bundle.key.as_dict()
        # Opt-in (copy arm, --own-controls): the prediction's Δ against its neighbour's controls
        pred_ctrl_t = (control_cache.stack(self.pred_getter.neighbor_control_keys(key_dict, payload.ctrl_keys))
                       if getattr(self.pred_getter, "own_controls", False) else None)
        pred_t = torch.from_numpy(payload.pred).to(truth_t.device, dtype=torch.float32, non_blocking=True)
        return Prediction(pred_t, 1, self.output_space, _result_extra(payload, payload.bundle), pred_ctrl_t)

    def on_missing_control(self, handle: Any, key_dict: Mapping[str, Any]) -> bool:
        # No control cells for this stratum (PBMC Donor5 Plasmablast has no PBS cells): no arm has a Δ
        # there, so it is skipped for every arm and counted in n_skipped. The reference raises instead.
        return True


class SplitHalfReference(PredictionSource):
    """Truth half A against truth half B: the score a perfect model would get (noise_ceiling.json)."""

    def __init__(self, n_repeats: int):
        self.n_repeats = int(n_repeats)

    def resolve(self, handle, payload, truth_t, ctrl_t, control_cache):
        return Prediction(None, self.n_repeats, ModelOutputSpace.RAW, {"n_control_groups": int(payload.bundle.n_groups)})


class NoResponse(PredictionSource):
    """A model that predicts nothing happens: the condition's own matched controls.

    A condition the truth can score but an arm could not is scored for that arm as this, not dropped.
    The prediction's half IS ``ctrl_b``, the control half every arm's Δ is taken against
    (``_score_no_response``), so its Δ is exactly zero: no gene is called DE (DEG overlap 0 wherever
    truth calls DEGs), Pearson Δ is undefined (a constant Δ; the figures set it to 0), and energy is
    truth half against control half. Truth half, control halves, seed and spaces are a K=1 arm's.
    It stands in for a failure, so it must score every condition it is handed: skipping one would
    drop that condition again, silently. It raises instead.
    """

    may_skip = False

    def resolve(self, handle, payload, truth_t, ctrl_t, control_cache):
        return Prediction(ctrl_t, 1, ModelOutputSpace.RAW, {"prediction": "matched_controls"})


class GenesisSource(PredictionSource):
    """GENESIS generating while the previous condition scores: two CUDA streams on one GPU.

    ``stream_gen`` runs the forward + scatter (run_streaming_eval) and records a ``gen_event``
    the moment a condition's rep-major buffer is full; ``stream_metrics`` waits on that event,
    so generation of condition i+1 overlaps scoring of condition i. On CPU both are the default
    stream. ``sync_cuda_streams`` (inline training eval) scores on the default stream instead:
    the overlap plus DataLoader teardown can corrupt the parent heap.
    """

    def __init__(self, *, bundle: Any, inflated_queue: Sequence, orig_n_by_id: Mapping[int, int], k: int,
                 batch_size: int, seed: int, pad_last_batch: bool, pipeline_depth: int, eval_split: str,
                 sync_cuda_streams: bool = False, persist_dumper: Any = None):
        from src.eval.orchestration.pregeneration import StreamingPregenStats

        self.bundle, self.output_space, self.dev = bundle, bundle.output_space, bundle.device
        self._gen_kw = dict(inflated_queue=inflated_queue, orig_n_by_id=orig_n_by_id, plan_k=int(k),
                            batch_size=int(batch_size), seed=int(seed), pipeline_depth=int(pipeline_depth),
                            pad_last_batch=bool(pad_last_batch or bundle.compiled),
                            progress_label=f"Stream eval forward [{eval_split}]")
        self.persist_dumper, self.stats = persist_dumper, StreamingPregenStats()
        is_cuda = self.dev.type == "cuda"
        self.stream_gen = torch.cuda.Stream(device=self.dev) if is_cuda else None
        self.stream_metrics = torch.cuda.Stream(device=self.dev) if is_cuda and not sync_cuda_streams else None

    def handles(self, bundles):
        from src.eval.orchestration.pregeneration import run_streaming_eval
        # the prefetcher walks the same list; resolve() checks the two stay in lockstep
        return run_streaming_eval(bundle=self.bundle, condition_bundles=bundles, stream_gen=self.stream_gen,
                                  release_model_after=True, stats_out=self.stats, progress=False, **self._gen_kw)

    def scoring_stream(self):
        return (torch.cuda.stream(self.stream_metrics) if self.stream_metrics is not None
                else contextlib.nullcontext())

    def resolve(self, completed, payload, truth_t, ctrl_t, control_cache):
        if payload.bundle is not completed.bundle:
            raise RuntimeError(f"stream eval out of lockstep: generated {completed.bundle.key.as_dict()!r}, "
                               f"prefetched {payload.bundle.key.as_dict()!r}")
        if completed.gen_event is not None and self.stream_metrics is not None:
            self.stream_metrics.wait_event(completed.gen_event)
            # pred_t is read on stream_metrics: without this the allocator could hand its storage
            # back out once `completed` is dropped, while scoring kernels still read it
            completed.pred_t.record_stream(self.stream_metrics)
        elif completed.gen_event is not None:
            completed.gen_event.synchronize()
        if self.persist_dumper is not None:
            self.persist_dumper.submit(cond_key=payload.bundle.key.as_dict(), pred_t=completed.pred_t, gen_event=None)
        return Prediction(completed.pred_t, int(completed.k_eff), self.output_space)

    def on_missing_control(self, completed, key_dict):
        # donor-holdout PBS-test donors have truth but no pooled control: keep the generated
        # cloud for well-vs-well energy, score nothing
        from src.eval.orchestration.model_eval import persist_generated_without_control
        return persist_generated_without_control(self.persist_dumper, pred_t=completed.pred_t, cond_key=key_dict)

    def close(self) -> None:
        # metrics kernels may still be queued on stream_metrics; finish before anything is torn down
        if self.dev.type == "cuda":
            if self.stream_metrics is not None:
                self.stream_metrics.synchronize()
            torch.cuda.synchronize(self.dev)


# Cloud sources: whoever fulfils a design's requests (src/eval/reference/designs.py)
#
# ``open(requests)`` sees every request of the run, deduped, in group order; ``fulfil(requests)``
# hands back one group's clouds as {request.key: (n_cells, G) cells in the model's output space}.

class StoredClouds:
    """Clouds already on disk: ``load(request)`` reads one (e.g. a replicate .npy)."""

    def __init__(self, load: Callable[[CloudRequest], Any]):
        self.load = load

    def open(self, requests: Sequence[CloudRequest]) -> None:
        pass

    def fulfil(self, requests: Sequence[CloudRequest]) -> dict[tuple, Any]:
        out = {}
        for req in requests:
            cells = self.load(req)
            if cells.shape[0] != req.n_cells:
                raise ValueError(f"stored cloud for {dict(req.row)} has {cells.shape[0]} cells, asked for {req.n_cells}")
            out[req.key] = cells
        return out

    def scoring_stream(self):
        return contextlib.nullcontext()

    def close(self) -> None:
        pass


class LiveClouds:
    """GENESIS generating every request in one batched stream, while the previous group scores.

    Each request is its own row of ``run_streaming_eval`` with its own noise seed, drawn whole
    as one segment. Requests in the work queue's order (one per qrow and rep, reps innermost)
    lay out the K-rep eval's slot stream (GenesisSource), so batch boundaries and cells match it.
    Two CUDA streams as GenesisSource: generation on ``stream_gen``, scoring on ``stream_metrics``
    after waiting on each cloud's ``gen_event``.
    """

    def __init__(self, bundle: Any, *, batch_size: int, pipeline_depth: int = 3, sync_cuda_streams: bool = False,
                 label: str = "Stream generate"):
        from src.eval.orchestration.pregeneration import StreamingPregenStats

        self.bundle, self.dev, self.label = bundle, bundle.device, label
        self.batch_size, self.pipeline_depth = int(batch_size), int(pipeline_depth)
        self.stats, self._stream, self._keys, self._pending = StreamingPregenStats(), None, [], []
        is_cuda = self.dev.type == "cuda"
        self.stream_gen = torch.cuda.Stream(device=self.dev) if is_cuda else None
        self.stream_metrics = torch.cuda.Stream(device=self.dev) if is_cuda and not sync_cuda_streams else None

    def open(self, requests: Sequence[CloudRequest]) -> None:
        from src.eval.cache.eval_cache_io import ConditionBundle, ConditionKey, QuadrupletRow
        from src.eval.orchestration.pregeneration import run_streaming_eval

        if self._stream is not None:
            raise RuntimeError("LiveClouds is one stream; open it once")
        rows = [QuadrupletRow(num_cells=int(r.n_cells), condition_keys={k: str(v) for k, v in r.row.items()},
                              control_id_keys={}, condition_idx={v: int(self.bundle.encode_condition(v, r.row[v]))
                                                                 for v in self.bundle.condition_variables})
                for r in requests]
        seed_by_row = {id(q): int(r.seed) for q, r in zip(rows, requests)}
        self._keys = [r.key for r in requests]
        self._stream = run_streaming_eval(
            bundle=self.bundle, inflated_queue=rows, orig_n_by_id={id(q): q.num_cells for q in rows},
            condition_bundles=[ConditionBundle(ConditionKey(r.key[0]), [q]) for q, r in zip(rows, requests)],
            plan_k=1, batch_size=self.batch_size, pad_last_batch=bool(self.bundle.compiled),
            pipeline_depth=self.pipeline_depth, stream_gen=self.stream_gen, progress=False, release_model_after=True,
            stats_out=self.stats, progress_label=self.label, seed_of=lambda q, rep: seed_by_row[id(q)])

    def fulfil(self, requests: Sequence[CloudRequest]) -> dict[tuple, Any]:
        """Advance the stream until this group's clouds are full; they come in ``open``'s order."""
        want, out = {r.key for r in requests}, {}
        for done in self._stream:
            key = self._keys[done.cond_idx]
            if key not in want:
                raise RuntimeError(f"live clouds out of order: generated {key}, the group wants {sorted(want)}")
            out[key] = done.pred_t
            self._pending.append((done.gen_event, done.pred_t))
            if len(out) == len(want):
                break
        return out

    @contextlib.contextmanager
    def scoring_stream(self):
        stream = self.stream_metrics
        with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
            for event, cells in self._pending:
                if event is None:
                    continue
                if stream is None:
                    event.synchronize()
                    continue
                stream.wait_event(event)
                cells.record_stream(stream)  # else the allocator may recycle it while scoring still reads it
            self._pending.clear()
            yield

    def close(self) -> None:
        if self.dev.type == "cuda":
            if self.stream_metrics is not None:
                self.stream_metrics.synchronize()
            torch.cuda.synchronize(self.dev)


# Score every group in one pass, for any list of designs

def score_groups(groups: Sequence[GroupRef], designs: Sequence[Any], source: Any, *, reader: Any, spec: Any,
                 workers: int = 2) -> dict[str, list[dict]]:
    """Every design's requests up front (deduped), then per group: read truth, take its clouds, score.

    Rows keep each design's own schema. A cloud belongs to one group: a live source hands
    each over once, so two groups asking for one cloud is an error, not a silent re-use.
    """
    names = [d.name for d in designs]
    if not designs or len(set(names)) != len(names):
        raise ValueError(f"score_groups needs designs with distinct names; got {names}")
    per_group = [dedupe([req for d in designs for req in d.requests(ref)]) for ref in groups]
    owner: dict[tuple, int] = {}
    for g, reqs in enumerate(per_group):
        for req in reqs:
            if owner.setdefault(req.key, g) != g:
                raise ValueError(f"groups {owner[req.key]} and {g} both ask for the cloud {req.key}")
    source.open([req for reqs in per_group for req in reqs])
    out: dict[str, list[dict]] = {n: [] for n in names}
    try:
        for group, requests in zip(GroupPrefetcher(reader, spec, max_workers=workers).iter_groups(list(groups)),
                                   per_group):
            clouds = source.fulfil(requests) if requests else {}
            if missing := [req.key for req in requests if req.key not in clouds]:
                raise KeyError(f"{type(source).__name__} fulfilled {len(requests) - len(missing)} of "
                               f"{len(requests)} clouds; first missing {missing[0]}")
            with source.scoring_stream():
                for d in designs:
                    out[d.name].extend(d.score(group, clouds))
            del clouds  # the next group's clouds reuse this memory
    finally:
        source.close()
    return out


# The loop: score every condition

def score_conditions(source: PredictionSource, *, cache_ns: Path, bundles: Sequence,
                     condition_variables: Sequence[str], plan: ModelEvalPlan, seed: int, device: torch.device,
                     metrics_workers: int, gene_arrays_path: Optional[Path], desc: str = "Metrics",
                     ) -> tuple[list[dict], Counter]:
    """Score every condition: (result rows, skip reasons). Skips are logged, never silent.

    Draws no random numbers of its own: the sampler's ``manual_seed(cond_seed + rep)`` and
    GENESIS's per-(condition, rep) noise seeds are the only ones in play.
    """
    from tqdm.auto import tqdm

    reader = EvalNpzReader(cache_ns)
    control_cache = ControlCache(cache_ns, indexed=reader, device=device, dtype=torch.float32)
    prefetcher = ConditionPrefetcher(cache_ns, indexed=reader, pred_getter=source.pred_getter,
                                     max_workers=int(metrics_workers))
    writer = (GeneArraysParquetWriter(gene_arrays_path, condition_variables=list(condition_variables))
              if plan.gene_sidecars else None)
    results: list[dict] = []
    skips: Counter = Counter()
    t0 = time.perf_counter()
    reset_eval_metric_logs()
    bar = tqdm(total=len(bundles), desc=desc, unit="cond")

    def skip(reason: str, key_dict: Mapping[str, Any], detail: str = "") -> None:
        if not source.may_skip:
            raise RuntimeError(f"{type(source).__name__} must score {dict(key_dict)}: {reason} {detail}")
        skips[reason] += 1
        log.warning("Skip %s | %s %s", key_dict, reason, detail)
        bar.update(1)

    try:
        # Advance both iterators OUTSIDE the scoring stream: GENESIS allocates its buffers on the gen stream.
        for i, (handle, payload) in enumerate(zip(source.handles(bundles), prefetcher.iter_conditions(bundles))):
            key_dict = payload.bundle.key.as_dict()
            if payload.truth is None or not payload.ctrl_keys:
                skip("missing_truth_or_control", key_dict)
                continue
            with source.scoring_stream():
                try:
                    tensors = _load_condition_tensors(payload, control_cache=control_cache, dev=device,
                                                      log_headroom=i == 0)
                except FileNotFoundError as exc:
                    if not source.on_missing_control(handle, key_dict):
                        raise
                    skip("missing_control_npz", key_dict, repr(exc))
                    continue
                truth_t, ctrl_t = tensors
                if not _condition_eligible_for_kreps(plan, truth_t, ctrl_t):
                    skip("insufficient_cells", key_dict, f"n_truth={truth_t.shape[0]} n_ctrl={ctrl_t.shape[0]}")
                    continue
                pred = source.resolve(handle, payload, truth_t, ctrl_t, control_cache)
                if pred is None or pred.k < 1:
                    skip("no_prediction", key_dict)
                    continue
                if isinstance(source, NoResponse):
                    rep = _score_no_response(plan=plan, truth_t=truth_t, ctrl_t=ctrl_t, seed=int(seed),
                                             key_dict=key_dict)
                else:
                    rep = _score_condition_kreps(plan=plan, truth_t=truth_t, ctrl_t=ctrl_t, pred_t_all=pred.pred_t,
                                                 k_eff=pred.k, output_space=pred.output_space, seed=int(seed),
                                                 key_dict=key_dict, first_condition_verbose=i == 0,
                                                 pred_ctrl_t=pred.pred_ctrl_t)
                n_truth = int(truth_t.shape[0])
                if writer is not None:
                    _write_sidecar(writer, pred, truth_t, ctrl_t, seed=int(seed), key_dict=key_dict)
                if i == 0:
                    log.info("first condition %s | %s", key_dict, headline_metrics(rep.means))
                results.append(build_result_row(
                    key_dict=key_dict, n_pred=n_truth if pred.pred_t is None else int(pred.pred_t.shape[0]),
                    n_truth=n_truth, n_ctrl=int(ctrl_t.shape[0]), metrics=rep.means, metrics_std=rep.stds,
                    k=rep.k_effective, extra=pred.extra))
                # drop the GPU buffers now (GENESIS's rep-major one too): the next condition reuses them
                del truth_t, ctrl_t, pred, tensors, handle
            bar.update(1)
    finally:
        bar.close()
        source.close()
        if writer is not None:
            writer.close()
    log.info("Metrics done | scored=%d skipped=%d %s | %.1fs", len(results), sum(skips.values()), dict(skips),
             time.perf_counter() - t0)
    return results, skips


def _score_no_response(*, plan: ModelEvalPlan, truth_t: torch.Tensor, ctrl_t: torch.Tensor, seed: int,
                       key_dict: Mapping[str, Any]) -> KRepResult:
    """``_score_condition_kreps`` at K=1 with ``ctrl_b`` itself where the prediction's half stands.

    Same sampler, seed, truth half, control halves, metric set and score_pair call as any K=1 arm;
    only ``pred_arm`` differs. ``ctrl_b`` is counts, a fixed point of the snap, so Δ = ctrl_b - ctrl_b
    is exactly 0 in every space. Its size is the control half's, not the truth half's: energy is a
    U-statistic, so the size moves only its noise, not its expectation.
    """
    t0 = time.perf_counter()
    truth_arm, _truth_b, ctrl_a, ctrl_b, _idx = SplitHalfSampler().sample(
        truth_t, ctrl_t, rep=0, seed=_stable_cond_seed(seed, key_dict))
    rep = score_pair(truth_arm, ctrl_b, ctrl_a, ctrl_b, a_space="raw", b_space="raw", metrics=plan.metrics,
                     ctrl_det_floors=plan.ctrl_det_floors or (0.10,))
    check_metric_keys(rep, plan.metrics)
    means, stds, n_reps = aggregate_rep_metrics(materialize_metric_tensor_batch([rep]))
    return KRepResult(means=means, stds=stds, n_reps_per_key=n_reps, k_effective=1,
                      elapsed_seconds=time.perf_counter() - t0)


def _write_sidecar(writer: GeneArraysParquetWriter, pred: Prediction, truth_t: torch.Tensor, ctrl_t: torch.Tensor,
                   *, seed: int, key_dict: Mapping[str, Any]) -> None:
    """Per-gene arrays against the FULL control pool (diagnostic, single baseline)."""
    if pred.pred_t is not None:  # rep 0 of a model: its whole first block against all of truth
        extras = _compute_condition_extras(pred_t=pred.pred_t[: truth_t.shape[0]], truth_t=truth_t, ctrl_t=ctrl_t,
                                           output_space=pred.output_space)
        label = "pred"
    else:  # the reference: rep 0's two truth halves (re-drawing reseeds, so nothing downstream moves)
        sampler = SplitHalfSampler()
        half_a, half_b, *_ = sampler.sample(truth_t, ctrl_t, rep=0, seed=_stable_cond_seed(seed, key_dict))
        extras = _compute_condition_extras(pred_t=half_b, truth_t=half_a, ctrl_t=ctrl_t,
                                           output_space=ModelOutputSpace.RAW, pred_label=sampler.pred_label)
        label = sampler.pred_label
    writer.write_condition(condition_keys=key_dict, gene_summary=extras.gene_summary, deg_detail=extras.deg_detail,
                           dropout_summary=extras.dropout_summary, pred_label=label)


def score_pseudobulk_condition(*, truth_t: torch.Tensor, ctrl_t: torch.Tensor, pred_t: torch.Tensor,
                               seed: int, key_dict: Mapping[str, Any],
                               plan: Optional[ModelEvalPlan] = None) -> Optional[dict[str, float]]:
    """``log1p_rgs`` Pearson r and Pearson Δ of a pseudobulk-only arm on one condition.

    ``pred_t`` is the arm's log1p vector tiled to each row's measured cell count, aligned
    with ``truth_t``. The draw is exactly :func:`score_pregenerated_condition`'s (sampler,
    seed, truth half with ctrl_a, prediction with ctrl_b), so the arm shares its truth and
    control halves with every Figure 2c arm; the predicted pseudobulk is the mean of the
    drawn rows, never snapped. None below the K-rep minima, as for cell arms.
    """
    plan = plan or pregenerated_eval_plan()
    if pred_t.shape != truth_t.shape:
        raise ValueError(f"prediction rows must align with truth: {tuple(pred_t.shape)} vs {tuple(truth_t.shape)}")
    if not _condition_eligible_for_kreps(plan, truth_t, ctrl_t):
        return None
    truth_arm, _, ctrl_a, ctrl_b, idx = SplitHalfSampler().sample(
        truth_t, ctrl_t, rep=0, seed=_stable_cond_seed(int(seed), key_dict))
    out = score_true_pseudobulk(truth_arm, pred_t[idx].mean(0), ctrl_a, ctrl_b)
    cols = metric_cols("log1p_rgs")
    return {cols[s]: out[s] for s in PSEUDOBULK_SHORT}


# The summary: shared fields, then the arm's own

def write_summary(path: Path, *, fields: Mapping[str, Any], results: list[dict], skips: Mapping[str, int]) -> dict:
    """metrics_summary.json / noise_ceiling.json: the caller's provenance, then what every reader expects."""
    summary: dict[str, Any] = {
        **fields,
        "metric_spaces": list(PSEUDOBULK_METRIC_SPACES),
        "pseudobulk_delta_spaces": list(PSEUDOBULK_DELTA_METRIC_SPACES),
        "deg_spaces": list(DEG_SPACES),
        "dropout_spaces": list(DEG_SPACES),
        "gene_summary_spaces": ["raw_int_rgs", "log1p_rgs", "log1p_slice_med_rgs"],
        "metric_groups": list(METRIC_GROUP_ORDER),
        "n_conditions": len(results),
        "n_skipped": int(sum(skips.values())),
        "skip_reasons": dict(skips),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    summary.update(summary_overall_stats(results))
    Path(path).write_text(json.dumps(summary, indent=2))
    log.info("wrote %s (%d conditions, skipped %s)", path, len(results), dict(skips))
    return summary
