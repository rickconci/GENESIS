"""Eval truth cache: paths, hashes, work queue, prefetched IO.

Truth is the per-condition NPZ namespace from :mod:`src.data.cache.eval_cache_build`
(h5ad). Readers are :class:`EvalNpzReader`.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Union,
)

import numpy as np
import pandas as pd
import torch

if TYPE_CHECKING:
    import pandas as pd

    from ...model.inference_output import PertGenInferenceBundle

logger = logging.getLogger(__name__)


# ===========================================================================
# Paths and hashes
# ===========================================================================

from ...data.data_utils import (
    control_npz_path,
    gene_set_hash,
    load_json,
    npz_path,
    perturbed_npz_path,
    read_eval_namespace_manifest,
    resolve_cache_namespace,
    safe_filename_part,
    save_json,
    save_npz,
)

# ===========================================================================
# Work-queue rows
# ===========================================================================


@dataclass(frozen=True, eq=False)
class QuadrupletRow:
    """One unit of work: one conditioning group + its cell count.

    Attributes:
        num_cells: Number of cells in this group.
        condition_keys: ``{var: stringified_value}`` for **all**
            ``model_conditioning_variables`` - uniquely identifies the
            perturbed truth NPZ on disk.
        control_id_keys: Subset of ``condition_keys`` that identifies the
            matching control NPZ. ``control_match_keys`` when set, else
            ``model_conditioning_variables`` minus ``control_condition``.
        condition_idx: ``{var: encoded_int_idx}`` from the bundle's encoders.
            ``-1`` when the bundle is missing or the value is empty.

    Notes:
        Equality and hashing are **identity-based** (``eq=False``). Two
        distinct rows can legitimately share identical
        ``condition_keys`` / ``num_cells`` (e.g. the K-repeat split-half
        protocol in :mod:`src.eval.orchestration.model_eval` replaces rows with new
        instances that carry an inflated ``num_cells``). Content-based
        equality would also be impossible without giving up the
        ergonomic ``dict`` typing of the three mapping fields, since the
        auto-generated ``__hash__`` would try to hash those dicts.
        Downstream code that needs to key on rows (e.g. the predictions
        cache in :mod:`src.eval.orchestration.pregeneration`) does so by instance; for
        content-based grouping use :class:`ConditionKey` /
        :func:`group_by_condition`.
    """

    num_cells: int
    condition_keys: dict[str, str]
    control_id_keys: dict[str, str]
    condition_idx: dict[str, int] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        """Convenience: ``row.<var>`` resolves to ``condition_keys[var]``."""
        keys = self.__dict__.get("condition_keys")
        if keys is not None and name in keys:
            return keys[name]
        raise AttributeError(name)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other


@dataclass(frozen=True)
class ConditionKey:
    """Hashable identifier for one perturbed condition (drops per-control vars).

    ``items`` is a sorted ``((var, value), ...)`` tuple so the key is both
    hashable and JSON-friendly. Attribute access (``key.<var>``) is forwarded
    to :meth:`as_dict` for ergonomics.
    """

    items: tuple[tuple[str, str], ...]

    def as_dict(self) -> dict[str, str]:
        return dict(self.items)

    def __getattr__(self, name: str) -> Any:
        d = self.__dict__.get("items")
        if d is not None:
            for k, v in d:
                if k == name:
                    return v
        raise AttributeError(name)


@dataclass
class ConditionBundle:
    """All :class:`QuadrupletRow`s sharing one :class:`ConditionKey`."""

    key: ConditionKey
    rows: list[QuadrupletRow] = field(default_factory=list)

    @property
    def n_groups(self) -> int:
        return len(self.rows)

    @property
    def n_cells(self) -> int:
        return sum(r.num_cells for r in self.rows)

    def per_control_count(self) -> dict[tuple, int]:
        """Cells per unique ``control_id_keys`` tuple in this bundle."""
        out: dict[tuple, int] = {}
        for r in self.rows:
            key = tuple(sorted(r.control_id_keys.items()))
            out[key] = out.get(key, 0) + r.num_cells
        return out


# ===========================================================================
# Encoding + predicate-driven work queue
# ===========================================================================


def _stringify_condition_value(v: Any) -> str:
    """Match datamodule's categorical-string representation (NaN -> ``""``)."""
    import pandas as pd

    from ...data.vocab import to_categorical_str

    if isinstance(v, float) and pd.isna(v):
        return ""
    return to_categorical_str(v)


def encode_rows(
    filt: "pd.DataFrame",
    *,
    condition_variables: Sequence[str],
    control_condition: Optional[Mapping[str, Any]] = None,
    bundle: Optional["PertGenInferenceBundle"] = None,
    control_match_keys: Optional[Sequence[str]] = None,
) -> list[QuadrupletRow]:
    """Encode rows of ``filt`` into :class:`QuadrupletRow` instances.

    Args:
        filt: Pre-filtered DataFrame (control rows dropped, ``num_cells``
            threshold applied). Must contain every column in
            ``condition_variables`` plus ``num_cells``.
        condition_variables: ``model_conditioning_variables`` from the config.
        control_condition: Variables fixed by the control population (e.g.
            ``{"drug": "DMSO_TF", "dose": 0.0}``). Used when
            ``control_match_keys`` is omitted: remaining MCV columns populate
            ``control_id_keys``. Empty / ``None`` -> every condition variable
            is part of ``control_id_keys`` (no control distinction).
        bundle: Optional inference bundle; when present, ``condition_idx`` is
            populated through ``bundle.encode_condition``.
        control_match_keys: Explicit control-NPZ keys (e.g. Parse well eval:
            ``[donor, cell_type]`` so the cytokine well is not copied onto
            PBS). Default: MCV minus ``control_condition`` keys.
    """
    import pandas as pd

    from ...data.data_utils import resolve_control_match_keys

    missing = [c for c in condition_variables if c not in filt.columns]
    if missing:
        raise KeyError(
            f"Missing condition_variables {missing} in DataFrame "
            f"(columns: {list(filt.columns)}).",
        )
    if "num_cells" not in filt.columns:
        raise KeyError("Missing required 'num_cells' column in DataFrame.")

    control_id_vars = resolve_control_match_keys(
        condition_variables, control_condition, control_match_keys,
    )
    unknown = [k for k in control_id_vars if k not in condition_variables]
    if unknown:
        raise KeyError(
            f"control_match_keys {unknown} not in condition_variables "
            f"{list(condition_variables)}.",
        )
    # Ablated runs only encode generator-input vars (encoder json subset).
    # Full ``condition_keys`` still identify the eval NPZ.
    encoders = getattr(bundle, "encoder_idx_to_name", None) if bundle is not None else None

    rows: list[QuadrupletRow] = []
    for r in filt.itertuples(index=False):
        condition_keys: dict[str, str] = {}
        condition_idx: dict[str, int] = {}
        for var in condition_variables:
            val = _stringify_condition_value(getattr(r, var))
            condition_keys[var] = val
            if bundle is not None and val != "":
                if encoders is not None and var not in encoders:
                    condition_idx[var] = -1
                else:
                    condition_idx[var] = int(bundle.encode_condition(var, val))
            else:
                condition_idx[var] = -1
        control_id_keys = {k: condition_keys[k] for k in control_id_vars}
        nc = int(pd.to_numeric(getattr(r, "num_cells", 0), errors="coerce") or 0)
        rows.append(
            QuadrupletRow(
                num_cells=nc,
                condition_keys=condition_keys,
                control_id_keys=control_id_keys,
                condition_idx=condition_idx,
            )
        )
    return rows


def load_work_queue(
    *,
    condition_summary_parquet: Union[str, Path],
    split_yaml: Union[str, Path],
    eval_split: str,
    condition_variables: Sequence[str],
    control_condition: Optional[Mapping[str, Any]] = None,
    metadata_joins: Optional[Sequence[Mapping[str, Any]]] = None,
    bundle: Optional["PertGenInferenceBundle"] = None,
    min_cells: int = 4,
    conditions_query: Optional[str] = None,
    max_conditions: Optional[int] = None,
    conditions_seed: int = 0,
    control_match_keys: Optional[Sequence[str]] = None,
    drop_control_rows: bool = True,
) -> list[QuadrupletRow]:
    """Build the work queue for one eval split via a split-YAML predicate.

    Pipeline:

    1. Read ``condition_summary_parquet`` (one row per conditioning group).
       PertGen-generated summaries are already metadata-enriched at setup time;
       pass ``metadata_joins`` only for external parquets missing joined columns.
    2. Optionally enrich with ``metadata_joins`` when joined predicate columns
       are not already present.
    3. Apply the ``<eval_split>:`` predicate from ``split_yaml`` via
       :func:`apply_split_predicate`.
    4. Drop rows matching ``control_condition`` (unless
       ``drop_control_rows=False``, needed to score test PBS wells).
    5. Drop rows with ``num_cells < min_cells``.
    6. Encode each surviving row into a :class:`QuadrupletRow`.

    Args:
        condition_summary_parquet: Path to the condition-summary parquet
            (one row per unique ``model_conditioning_variables`` combination,
            plus ``num_cells``).
        split_yaml: Path to the split-config YAML.
        eval_split: ``"test"`` / ``"validation"`` / ...
        condition_variables: ``model_conditioning_variables`` from the config.
        control_condition: ``{var: value}`` identifying control rows
            (typically from ``config.control_condition``). Pass an empty dict
            or ``None`` to skip control filtering.
        metadata_joins: Optional ``[{left_on, parquet, columns}]`` entries
            (matches ``PertGenConfig.metadata_joins``).
        bundle: Optional :class:`PertGenInferenceBundle` for categorical
            encoding. When omitted, ``condition_idx`` is left at ``-1``.
        min_cells: Minimum cells per condition.
        conditions_query: Optional ``pandas.DataFrame.query`` expression
            applied to the filtered condition summary (e.g.
            ``"cell_name in ['C32', 'PANC-1']"``).  Lets a caller carve a
            composition-matched subset out of a large split.
        max_conditions: Optional cap; a seeded uniform random sample is
            drawn when the split has more eligible conditions than this.
            Intended for ``eval_split="train"``, which can hold tens of
            thousands of conditions.
        conditions_seed: Seed for the ``max_conditions`` sample, so the same
            subset is scored across checkpoints and runs.

    Raises:
        ValueError: When the filtered queue is empty.
        KeyError: When the YAML has no ``<eval_split>:`` block or the parquet
            is missing a required column.
    """
    import pandas as pd

    from ...data.data_utils import filter_condition_summary_for_eval

    if not condition_variables:
        raise ValueError("load_work_queue requires non-empty condition_variables.")

    obs = pd.read_parquet(condition_summary_parquet)
    n_before = len(obs)

    filt = filter_condition_summary_for_eval(
        obs,
        split_config=Path(split_yaml),
        eval_splits=(eval_split,),
        control_condition=control_condition,
        drop_control_rows=drop_control_rows,
        min_cells=min_cells,
    )
    logger.info(
        "Eval row filter | split=%s rows=%d -> %d (drop_control=%s, min_cells>=%d)",
        eval_split, n_before, len(filt), drop_control_rows, min_cells,
    )

    if conditions_query:
        n_pre_query = len(filt)
        filt = filt.query(conditions_query)
        logger.info(
            "Eval row filter | conditions_query=%r %d -> %d",
            conditions_query, n_pre_query, len(filt),
        )
    if max_conditions is not None and len(filt) > int(max_conditions):
        filt = filt.sample(
            n=int(max_conditions), random_state=int(conditions_seed), replace=False,
        )
        logger.info(
            "Eval row filter | subsampled to max_conditions=%d (seed=%d)",
            int(max_conditions), int(conditions_seed),
        )

    sort_cols = [c for c in condition_variables if c in filt.columns]
    if sort_cols:
        filt = filt.sort_values(sort_cols, kind="mergesort")

    rows = encode_rows(
        filt,
        condition_variables=condition_variables,
        control_condition=control_condition,
        bundle=bundle,
        control_match_keys=control_match_keys,
    )

    if not rows:
        raise ValueError(
            f"Empty work queue after filters (eval_split={eval_split!r}, "
            f"min_cells={min_cells}, control_condition={control_condition!r}).",
        )

    total_cells = sum(r.num_cells for r in rows)
    logger.info(
        "Work queue ready | conditions=%d total_cells=%d encoded=%s",
        len(rows), total_cells, bundle is not None,
    )
    return rows


def load_work_queue_or_none(
    *,
    condition_summary_parquet: Union[str, Path],
    split_yaml: Union[str, Path],
    eval_split: str,
    condition_variables: Sequence[str],
    control_condition: Optional[Mapping[str, Any]] = None,
    metadata_joins: Optional[Sequence[Mapping[str, Any]]] = None,
    bundle: Optional["PertGenInferenceBundle"] = None,
    min_cells: int = 4,
    conditions_query: Optional[str] = None,
    max_conditions: Optional[int] = None,
    conditions_seed: int = 0,
    control_match_keys: Optional[Sequence[str]] = None,
    drop_control_rows: bool = True,
) -> Optional[list[QuadrupletRow]]:
    """Build the work queue, or return ``None`` when the split has 0 conditions.

    Checks the split config first (authoritative). Also catches the
    :func:`load_work_queue` empty-queue ``ValueError`` as a safety net -- which
    is also what turns an over-restrictive ``conditions_query`` into a clean
    "0 conditions" skip rather than a crash.
    """
    from ...data.data_utils import count_eval_split_conditions

    n_eligible = count_eval_split_conditions(
        condition_summary_parquet,
        split_config=Path(split_yaml),
        eval_split=eval_split,
        control_condition=control_condition,
        min_cells=min_cells,
        drop_control_rows=drop_control_rows,
    )
    if n_eligible == 0:
        logger.info(
            "Eval split %r has 0 eligible conditions (split_config); skipping.",
            eval_split,
        )
        return None
    try:
        return load_work_queue(
            condition_summary_parquet=condition_summary_parquet,
            split_yaml=split_yaml,
            eval_split=eval_split,
            condition_variables=condition_variables,
            control_condition=control_condition,
            metadata_joins=metadata_joins,
            bundle=bundle,
            min_cells=min_cells,
            conditions_query=conditions_query,
            max_conditions=max_conditions,
            conditions_seed=conditions_seed,
            control_match_keys=control_match_keys,
            drop_control_rows=drop_control_rows,
        )
    except ValueError as exc:
        if "Empty work queue" in str(exc):
            logger.info(
                "Eval split %r produced empty work queue after filters; skipping (%s).",
                eval_split,
                exc,
            )
            return None
        raise


def group_by_condition(
    rows: Sequence[QuadrupletRow],
    *,
    eval_group_keys: Optional[Sequence[str]] = None,
) -> list[ConditionBundle]:
    """Bucket work-queue rows into scored eval conditions.

    When ``eval_group_keys`` is set (from
    :func:`src.data.data_utils.resolve_eval_group_keys`, defaulting to all
    ``model_conditioning_variables``), each bundle key is that full tuple.
    When omitted, the grouping drops ``control_id_keys``.
    """
    by_key: dict[ConditionKey, ConditionBundle] = {}
    for r in rows:
        if eval_group_keys is not None:
            items = tuple(
                sorted(
                    (k, r.condition_keys[k])
                    for k in eval_group_keys
                    if k in r.condition_keys
                ),
            )
        else:
            ctrl_vars = set(r.control_id_keys.keys())
            items = tuple(
                sorted(
                    (k, v) for k, v in r.condition_keys.items() if k not in ctrl_vars
                ),
            )
        key = ConditionKey(items=items)
        bundle = by_key.setdefault(key, ConditionBundle(key=key, rows=[]))
        bundle.rows.append(r)
    return list(by_key.values())


# ===========================================================================
# Eval truth (per-condition NPZ)
# ===========================================================================


def _sorted_key_tuple(keys: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(k), str(v)) for k, v in keys.items()))


def resolve_eval_cache_mode(namespace: Path) -> str:
    """Return the namespace mode; missing manifest + NPZ files is ``npz_v1``."""
    ns = Path(namespace)
    manifest_path = ns / "manifest.json"
    if manifest_path.is_file():
        try:
            mode = str(read_eval_namespace_manifest(ns).get("mode", ""))
            if mode:
                return mode
        except (OSError, RuntimeError):
            pass
    if any(ns.glob("perturbed_cache__*.npz")):
        from src.data.cache.eval_cache_build import EVAL_NPZ_MODE
        return EVAL_NPZ_MODE
    return str(read_eval_namespace_manifest(ns).get("mode", ""))


def open_eval_truth_reader(namespace: Path) -> Any:
    """Open the NPZ reader. Anything else is a leftover train-cache index: raise."""
    from src.data.cache.eval_cache_build import EVAL_NPZ_MODE
    from .eval_npz_reader import EvalNpzReader

    ns = Path(namespace)
    mode = resolve_eval_cache_mode(ns)
    if mode != EVAL_NPZ_MODE:
        raise ValueError(f"eval cache {ns} has mode={mode!r}; only {EVAL_NPZ_MODE} is built")
    return EvalNpzReader(ns)


# ===========================================================================
# ControlCache
# ===========================================================================


class ControlCache:
    """Lazy GPU-side cache of control truth keyed by ``control_id_keys``."""

    def __init__(
        self,
        namespace: Path,
        *,
        indexed: Any = None,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self._ns = Path(namespace)
        self._device = torch.device(device)
        self._dtype = dtype
        self._gpu_cache: dict[tuple, torch.Tensor] = {}
        self._indexed = indexed if indexed is not None else open_eval_truth_reader(self._ns)

    @property
    def namespace(self) -> Path:
        return self._ns

    @property
    def device(self) -> torch.device:
        return self._device

    def get(self, keys: Mapping[str, Any]) -> torch.Tensor:
        """Return the ``(n_ctrl, G)`` control tensor for one ``keys`` group."""
        cache_key = _sorted_key_tuple(keys)
        cached = self._gpu_cache.get(cache_key)
        if cached is not None:
            return cached

        arr = self._indexed.control({str(k): str(v) for k, v in keys.items()})
        t = torch.from_numpy(np.asarray(arr)).to(
            self._device, dtype=self._dtype, non_blocking=True,
        )
        self._gpu_cache[cache_key] = t
        return t

    def stack(self, keys: Sequence[Mapping[str, Any]]) -> torch.Tensor:
        """Concatenate controls for a sequence of ``control_id_keys`` groups."""
        if not keys:
            raise ValueError("ControlCache.stack called with empty keys.")
        return torch.cat([self.get(kw) for kw in keys], dim=0)

    def __len__(self) -> int:
        return len(self._gpu_cache)

    def clear(self) -> None:
        """Drop all cached tensors."""
        self._gpu_cache.clear()


# ===========================================================================
# ConditionPrefetcher
# ===========================================================================


PredGetter = Callable[[QuadrupletRow], Optional[np.ndarray]]


@dataclass
class ConditionPayload:
    """CPU-side payload for one condition, ready to move to GPU and score.

    ``ctrl_library_full`` is the concatenated control-pool full-tx UMI vector
    (prospective depth prior). Required for cp10k denorm; never filled from
    perturbed truth.
    """

    bundle: ConditionBundle
    pred: Optional[np.ndarray]
    truth: Optional[np.ndarray]
    ctrl_keys: list[dict[str, str]]
    ctrl_library_full: Optional[np.ndarray] = None


class ConditionPrefetcher:
    """1-ahead prefetch of truth arrays + optional predictions per condition bundle."""

    def __init__(
        self,
        namespace: Path,
        *,
        indexed: Any = None,
        pred_getter: Optional[PredGetter] = None,
        max_workers: int = 4,
        require_ctrl_library_full: bool = False,
    ) -> None:
        self._ns = Path(namespace)
        self._pred_getter = pred_getter
        self._max_workers = int(max_workers)
        self._require_ctrl_library_full = bool(require_ctrl_library_full)
        self._indexed = indexed if indexed is not None else open_eval_truth_reader(self._ns)

    def _load_perturbed_truth(self, condition_keys: dict[str, str]) -> np.ndarray:
        return self._indexed.perturbed(condition_keys)

    def _load_ctrl_library_full(self, control_id_keys: dict[str, str]) -> np.ndarray:
        loader = getattr(self._indexed, "control_library_full", None)
        if loader is None:
            raise RuntimeError(
                "require_ctrl_library_full=True but truth reader has no "
                f"control_library_full (got {type(self._indexed)!r}). "
                "Point eval_cache_dir at a __with_libfull NPZ namespace."
            )
        lib = loader(control_id_keys)
        if lib is None:
            raise RuntimeError(
                f"control_library_full missing for {dict(control_id_keys)} under "
                f"{self._ns} — rebuild / use __with_libfull eval cache."
            )
        return np.asarray(lib, dtype=np.float64).ravel()

    def _prefetch(self, cond_bundle: ConditionBundle) -> ConditionPayload:
        pred_parts: list[np.ndarray] = []
        truth_parts: list[np.ndarray] = []
        ctrl_keys: list[dict[str, str]] = []
        ctrl_lib_parts: list[np.ndarray] = []

        for qrow in cond_bundle.rows:
            if self._pred_getter is not None:
                p = self._pred_getter(qrow)
                if p is not None and p.size > 0:
                    pred_parts.append(np.ascontiguousarray(p, dtype=np.float32))

            try:
                truth_parts.append(self._load_perturbed_truth(qrow.condition_keys))
            except KeyError as exc:
                logger.error("Missing perturbed index: %s", exc)
                continue
            ctrl_keys.append(qrow.control_id_keys)
            if self._require_ctrl_library_full:
                ctrl_lib_parts.append(
                    self._load_ctrl_library_full(qrow.control_id_keys),
                )

        ctrl_library_full = (
            np.concatenate(ctrl_lib_parts, axis=0) if ctrl_lib_parts else None
        )
        return ConditionPayload(
            bundle=cond_bundle,
            pred=np.concatenate(pred_parts, axis=0) if pred_parts else None,
            truth=np.concatenate(truth_parts, axis=0) if truth_parts else None,
            ctrl_keys=ctrl_keys,
            ctrl_library_full=ctrl_library_full,
        )

    def iter_conditions(
        self, bundles: Sequence[ConditionBundle],
    ) -> Iterator[ConditionPayload]:
        """Yield :class:`ConditionPayload` per bundle, 1-ahead prefetched.

        GPU work for condition ``i`` happens while the pool reads NPZs for
        condition ``i+1``. Memory stays bounded to two conditions of truth.
        """
        n = len(bundles)
        if n == 0:
            return
        if self._max_workers <= 0:
            for bundle in bundles:
                yield self._prefetch(bundle)
            return
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures: dict[int, Any] = {0: pool.submit(self._prefetch, bundles[0])}
            for i in range(n):
                if i + 1 < n:
                    futures[i + 1] = pool.submit(self._prefetch, bundles[i + 1])
                yield futures.pop(i).result()


__all__ = [
    # Paths and hashes
    "gene_set_hash",
    "safe_filename_part",
    "resolve_cache_namespace",
    "npz_path",
    "control_npz_path",
    "perturbed_npz_path",
    "load_json",
    "save_json",
    "save_npz",
    # Work-queue rows
    "QuadrupletRow",
    "ConditionKey",
    "ConditionBundle",
    "ConditionPayload",
    "ConditionPrefetcher",
    "ControlCache",
    "open_eval_truth_reader",
    "resolve_eval_cache_mode",
    # Work-queue construction
    "encode_rows",
    "group_by_condition",
    "load_work_queue",
    "load_work_queue_or_none",
]
