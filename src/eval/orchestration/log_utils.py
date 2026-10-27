"""Distilled eval pipeline logging — phase markers at key steps, optional first-condition detail.

Entry points call :func:`configure_eval_logging` once; library code uses
:func:`log_step` for milestones and :func:`log_detail` for debug-only extras.

Handlers use :class:`logging.handlers.QueueHandler` so formatting and stdout I/O
run on a background thread and do not block the GPU scoring loop.
"""

from __future__ import annotations

import atexit
import logging
import queue
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from logging.handlers import QueueHandler, QueueListener
from typing import Any, Iterator, Optional, Union

import numpy as np
import torch

_EVAL_VERBOSE: bool = False
_EVAL_STATE: dict[str, Any] = {"metric_bundles_seen": set()}
_QUEUE_LISTENER: Optional[QueueListener] = None

_EVAL_LOGGER_NAMES: tuple[str, ...] = (
    "src.eval.orchestration.model_eval",
    "src.eval.orchestration.pregeneration",
    "src.eval.cache.eval_cache_io",
    "src.eval.metrics.eval_metrics",
    "src.eval.metrics.deg",
    "src.eval.reference.split_half",
    "src.eval.orchestration.log_utils",
    "pregenerate_pertgen",
)

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s | %(message)s"
_LOG_DATEFMT = "%H:%M:%S"


def eval_verbose() -> bool:
    return _EVAL_VERBOSE


def _stop_queue_listener() -> None:
    global _QUEUE_LISTENER
    if _QUEUE_LISTENER is not None:
        _QUEUE_LISTENER.stop()
        _QUEUE_LISTENER = None


def _install_root_handlers(*, level: int, non_blocking: bool) -> None:
    """Attach stream handlers to the root logger (optionally via a queue)."""
    global _QUEUE_LISTENER
    _stop_queue_listener()

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    if non_blocking:
        log_queue: queue.Queue[Any] = queue.Queue(-1)
        root.addHandler(QueueHandler(log_queue))
        _QUEUE_LISTENER = QueueListener(
            log_queue, stream_handler, respect_handler_level=True,
        )
        _QUEUE_LISTENER.start()
        atexit.register(_stop_queue_listener)
    else:
        root.addHandler(stream_handler)


def configure_eval_logging(
    *,
    debug: bool,
    level: str = "INFO",
    non_blocking: bool = True,
) -> None:
    """Configure logging for an eval run.

    ``debug=True`` enables :func:`log_detail` and per-repeat metric-bundle logs.
    ``non_blocking=True`` (default) routes records through a queue so stdout
    does not stall the scoring thread.
    """
    global _EVAL_VERBOSE
    _EVAL_VERBOSE = bool(debug)
    lvl_name = level.upper()
    lvl = getattr(logging, lvl_name, logging.DEBUG if debug else logging.INFO)
    _install_root_handlers(level=lvl, non_blocking=non_blocking)
    for name in _EVAL_LOGGER_NAMES:
        logging.getLogger(name).setLevel(lvl)


def reset_eval_metric_logs() -> None:
    """Clear per-run metric bundle log dedup (call at start of each metrics pass)."""
    _EVAL_STATE["metric_bundles_seen"] = set()


def format_scalar_floats(
    metrics: dict[str, float],
    *,
    prefer: tuple[str, ...] = (),
    max_keys: int = 10,
) -> str:
    """Format materialized scalar metrics (no GPU tensors)."""
    ordered = [k for k in prefer if k in metrics]
    ordered.extend(k for k in sorted(metrics) if k not in ordered)
    if not ordered:
        return f"{len(metrics)} keys"
    parts = []
    for k in ordered[:max_keys]:
        val = float(metrics[k])
        parts.append(f"{k}=nan" if np.isnan(val) else f"{k}={val:.4g}")
    if len(ordered) > max_keys:
        parts.append("...")
    return ", ".join(parts)


def _metric_bundle_summary(metrics: dict[str, Any]) -> str:
    """Describe a metric dict without reading GPU scalar values."""
    n = len(metrics)
    if not metrics:
        return "0 keys"
    on_gpu = sum(
        1 for v in metrics.values()
        if isinstance(v, torch.Tensor) and v.device.type != "cpu"
    )
    if on_gpu:
        return f"{n} keys ({on_gpu} on GPU, values deferred)"
    return f"{n} keys"


def log_metric_bundle(
    logger: logging.Logger,
    bundle_id: str,
    metrics: dict[str, torch.Tensor],
    *,
    prefer: tuple[str, ...] = (),  # noqa: ARG001 — kept for call-site stability
) -> None:
    """Log one metric family without forcing GPU→host scalar syncs.

    First occurrence per :func:`reset_eval_metric_logs` cycle is INFO (key count
    only). Later occurrences appear only when eval logging was configured with
    ``debug=True``.
    """
    line = _metric_bundle_summary(metrics)
    seen: set[str] = _EVAL_STATE.setdefault("metric_bundles_seen", set())
    first = bundle_id not in seen
    if first:
        seen.add(bundle_id)
        log_step(logger, bundle_id, "%s", line)
    elif _EVAL_VERBOSE:
        log_detail(logger, bundle_id, "%s", line)


def ensure_eval_phase_logging() -> None:
    """Make :func:`log_step` milestones visible (INFO) for any entry point."""
    if eval_verbose():
        return
    root = logging.getLogger()
    if not root.handlers:
        configure_eval_logging(debug=False, level="INFO")
    else:
        for name in _EVAL_LOGGER_NAMES:
            logging.getLogger(name).setLevel(logging.INFO)


def log_step(logger: logging.Logger, phase: str, msg: str, *args: Any) -> None:
    """Pipeline milestone — always emitted at INFO."""
    logger.info("[eval:%s] " + msg, phase, *args)


def log_detail(logger: logging.Logger, phase: str, msg: str, *args: Any) -> None:
    """Extra detail when :func:`configure_eval_logging` was called with ``debug=True``."""
    if _EVAL_VERBOSE:
        logger.info("[eval:%s] " + msg, phase, *args)


# ---------------------------------------------------------------------------
# Startup profiler
# ---------------------------------------------------------------------------


@dataclass
class EvalStartupProfiler:
    """Accumulate monotonic timings for named startup phases.

    Pure-data container; the only side effect is one ``log_step`` per
    :meth:`phase` exit and one summary table on :meth:`log_summary`. Phase
    names are arbitrary; nesting is not tracked — flat list semantics.

    Example::

        prof = EvalStartupProfiler()
        with prof.phase("ensure_index"):
            ...
        with prof.phase("bundle_load"):
            ...
        prof.log_summary(logger)
    """

    label: str = "startup"
    durations: list[tuple[str, float]] = field(default_factory=list)
    _start: float = field(default_factory=lambda: time.perf_counter(), init=False)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Time one phase; logs ``[eval:startup] <name> done in Xs`` on exit."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = float(time.perf_counter() - t0)
            self.durations.append((name, dt))
            logger = logging.getLogger("src.eval.orchestration.log_utils")
            log_step(logger, self.label, "%s done in %.2fs", name, dt)

    def total(self) -> float:
        """Total wall time since profiler construction (monotonic)."""
        return float(time.perf_counter() - self._start)

    def log_summary(self, logger: logging.Logger) -> None:
        """Emit one compact table of phase times at INFO."""
        if not self.durations:
            log_step(logger, self.label, "no phases recorded (total=%.2fs)", self.total())
            return
        total = sum(dt for _, dt in self.durations)
        wall = self.total()
        log_step(
            logger, self.label,
            "phase summary | wall=%.2fs tracked=%.2fs (%.0f%%)",
            wall, total, 100.0 * total / max(wall, 1e-9),
        )
        width = max(len(name) for name, _ in self.durations)
        for name, dt in self.durations:
            log_step(
                logger, self.label,
                "  %-*s %6.2fs  (%4.1f%%)",
                width, name, dt, 100.0 * dt / max(wall, 1e-9),
            )


def shapes_line(**arms: Union[torch.Tensor, np.ndarray]) -> str:
    """Compact ``name=(n,g)`` listing without GPU scalar syncs."""
    parts = []
    for name, t in arms.items():
        parts.append(f"{name}={list(t.shape)}")
    return " ".join(parts)


def headline_metrics(metrics: dict[str, Any]) -> str:
    """The core metrics (configs/eval/metrics.yaml), which every metric set scores, on one line."""
    from src.eval.metrics.reporting import load_metric_set

    parts = []
    for k in sorted(load_metric_set("core")):
        if k not in metrics:
            continue
        v = metrics[k]
        short = k.split("/", 1)[-1] if "/" in k else k
        try:
            parts.append(f"{short}={float(v):.3f}")
        except (TypeError, ValueError):
            parts.append(f"{short}={v!r}")
    if not parts:
        return f"{len(metrics)} metrics"
    return ", ".join(parts)
