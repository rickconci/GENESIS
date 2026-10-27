import logging
import warnings

warnings.filterwarnings("ignore")

import hashlib
import json
import pickle
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .splits.data_class import PredicateBlock, SplitConfig

_EVAL_CACHE_SPLITS = frozenset({"validation", "test"})
# ``train`` is scorable too (train-fit vs generalization diagnostics), but it is
# not a YAML predicate block: it is the complement of validation/test minus
# ``train_exclusions``.  See :func:`train_membership_mask`.
_SCORABLE_SPLITS = _EVAL_CACHE_SPLITS | {"train"}

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


# Extra suffix is a namespace variant of the same gene set
# (``__with_libfull``, ``__cell_type``, ``__cell_type__well``).
_GENESET_DIR_RE = re.compile(
    r"^geneset_(?P<hash>[0-9a-f]+)__n_(?P<n>\d+)(?:__.+)?$"
)


def parse_geneset_dirname(name: str) -> Optional[Tuple[str, int]]:
    """``geneset_<hash>__n_<N>[__suffix]`` → ``(hash, n)``, else ``None``."""
    m = _GENESET_DIR_RE.match(str(name))
    return (m.group("hash"), int(m.group("n"))) if m else None


@dataclass(frozen=True)
class EvalManifest:
    """Resolved paths for eval scripts (``workdir/eval/eval_manifest.json``).

    ``eval_cache_dir`` is the per-condition NPZ truth namespace used for
    scoring (validation/test).
    """

    condition_summary: Path
    eval_cache_dir: Path
    group_cols: tuple[str, ...]
    gene_set_hash: str
    n_genes: int
    split_config: Optional[str]
    normalization_id: Optional[str] = None

    @property
    def eval_truth_dir(self) -> Path:
        """Per-condition NPZ eval truth namespace (alias for ``eval_cache_dir``)."""
        return self.eval_cache_dir

    def to_dict(self) -> dict[str, Any]:
        out = {
            "condition_summary": str(self.condition_summary),
            "eval_cache_dir": str(self.eval_cache_dir),
            "group_cols": list(self.group_cols),
            "gene_set_hash": self.gene_set_hash,
            "n_genes": self.n_genes,
            "split_config": self.split_config,
            "normalization_id": self.normalization_id,
        }
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalManifest":
        eval_cache_raw = data.get("eval_cache_dir")
        if not eval_cache_raw:
            raise KeyError(
                "eval_manifest missing eval_cache_dir — rebuild eval artifacts "
                "(run dm.setup() or scripts/data/build_cache.py eval).",
            )
        return cls(
            condition_summary=Path(data["condition_summary"]),
            eval_cache_dir=Path(eval_cache_raw),
            group_cols=tuple(data.get("group_cols") or ()),
            gene_set_hash=str(data.get("gene_set_hash", "")),
            n_genes=int(data.get("n_genes", 0)),
            split_config=data.get("split_config"),
            normalization_id=data.get("normalization_id"),
        )


def align_eval_cache_dir_to_gene_set(manifest: EvalManifest) -> EvalManifest:
    """Rewrite ``eval_cache_dir`` to sibling ``geneset_<hash>__n_<N>`` when stale.

    Setup can record the training ``gene_set_hash`` while a config override pins
    a different geneset folder. Training panel wins.
    """
    gsh, n = (manifest.gene_set_hash or "").strip(), int(manifest.n_genes or 0)
    if not gsh or n <= 0:
        return manifest
    cur = Path(manifest.eval_cache_dir)
    parsed = parse_geneset_dirname(cur.name)
    if parsed is not None and parsed[0] == gsh:
        return manifest
    expected = (cur.parent / f"geneset_{gsh}__n_{n}").resolve()
    if not expected.is_dir():
        raise FileNotFoundError(
            f"eval_cache_dir={cur} mismatches gene_set_hash={gsh}; missing {expected}",
        )
    logger.warning("eval_cache_dir %s → %s (gene_set_hash=%s)", cur, expected, gsh)
    return replace(manifest, eval_cache_dir=expected)


def write_eval_manifest(workdir: Union[str, Path], manifest: EvalManifest) -> Path:
    """Write ``<workdir>/eval/eval_manifest.json`` (same layout scripts expect)."""
    wd = Path(workdir)
    path = wd / "eval" / "eval_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2))
    return path


def _heal_moved_paths(workdir: Path, manifest: EvalManifest) -> EvalManifest:
    """Re-anchor paths a run recorded at a location that has since moved (in memory, logged).

    The condition summary lives in the run's own eval/ folder, so it follows the run;
    a split goes through the split move table. A path that still does not exist raises.
    """
    import dataclasses

    from src.paths import relocated_split_path

    changes = {}
    summary = Path(manifest.condition_summary)
    if not summary.is_file() and summary.name == "condition_summary.parquet" and summary.parent.name == "eval":
        here = workdir / "eval" / summary.name
        if not here.is_file():
            raise FileNotFoundError(f"condition summary missing at {summary} and at {here}")
        changes["condition_summary"] = here
    split = manifest.split_config
    if split and not Path(split).is_file():
        moved = relocated_split_path(str(split))
        if not Path(moved).is_file():
            raise FileNotFoundError(f"split_config {split} is missing and has no recorded move")
        changes["split_config"] = moved
    for k, v in changes.items():
        logger.warning("eval manifest %s: %s moved -> %s (healed in memory)", workdir.name, k, v)
    return dataclasses.replace(manifest, **changes) if changes else manifest


def load_eval_manifest(workdir: Union[str, Path]) -> EvalManifest:
    """Read ``eval/eval_manifest.json``; heal ``eval_cache_dir`` to training gene set."""
    path = Path(workdir) / "eval" / "eval_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"No eval manifest at {path}.")
    manifest = _heal_moved_paths(Path(workdir), EvalManifest.from_dict(json.loads(path.read_text())))
    aligned = align_eval_cache_dir_to_gene_set(manifest)
    if aligned.eval_cache_dir != manifest.eval_cache_dir:
        write_eval_manifest(workdir, aligned)
        logger.warning("Rewrote %s → %s", path, aligned.eval_cache_dir)
    return aligned


def log_split_setup_summary(
    *,
    split_s: float,
    split_cfg: Any,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    hbss_summary: Optional[Dict[str, Any]] = None,
    cells_summary: Optional[Dict[str, Any]] = None,
    hbss_balance_report: Optional[pd.DataFrame] = None,
) -> None:
    """Rank-0 style stdout for split + optional HBSS (used from ``PertGenDataModule.setup``)."""
    tc = split_cfg.train
    tsamp = tc.sample if tc.sample != "hbss" else f"hbss({tc.fraction})"
    if tc.cells.target_n is not None:
        tsamp = f"{tsamp}+celln{tc.cells.target_n}"
    elif tc.cells.per_condition is not None:
        tsamp = f"{tsamp}+{tc.cells.per_condition}/cond"
    print(
        f"  Data: split {split_s:.1f}s | dataset={split_cfg.dataset} "
        f"controls_to_train={split_cfg.controls_to_train} train.sample={tsamp} "
        f"→ train,val,test={len(train_idx):,},{len(val_idx):,},{len(test_idx):,}",
        flush=True,
    )
    if hbss_summary:
        h = hbss_summary
        print(
            f"  Data: HBSS triplets={h['n_triplets_kept']:,}/{h['n_triplets_total']:,} "
            f"pert_rows={h['n_train_rows_pert_after']:,}/{h['n_train_rows_pert_before']:,} "
            f"ctrl_kept={h['n_train_rows_ctrl_kept']:,} "
            f"IPF_converged={h['ipf_converged']} iters={h['ipf_iters']}",
            flush=True,
        )
    if cells_summary:
        c = cells_summary
        if "per_condition" in c:
            print(
                f"  Data: {c['per_condition']} cells/condition "
                f"groups={c['n_groups']:,} short={c['n_groups_short']:,} "
                f"pert_rows={c['n_after']:,}/{c['n_before']:,} "
                f"controls_kept={c.get('n_kept_fixed', 0):,} seed={c['seed']}",
                flush=True,
            )
        else:
            print(
                f"  Data: cell budget target={c['target_n']:,} "
                f"rows={c['n_after']:,}/{c['n_before']:,} "
                f"applied={c['applied']} seed={c['seed']}",
                flush=True,
            )
    if hbss_balance_report is not None:
        print("  HBSS balance:\n" + hbss_balance_report.to_string(index=False), flush=True)


def safe_filename_part(value: Any) -> str:
    """Return a compact filesystem-safe token (alphanumerics + ``._-``)."""
    cleaned = re.sub(r"[^\w.\-]+", "_", str(value).strip()).strip("_")
    if not cleaned:
        return "unknown"
    return cleaned[:180]


def gene_set_hash(gene_names: Sequence[str]) -> str:
    """Return a stable 16-hex hash for a gene set (order-insensitive)."""
    unique_sorted = sorted({str(g) for g in gene_names})
    payload = "\n".join(unique_sorted).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def resolve_cache_namespace(
    cache_root: Union[str, Path],
    *,
    normalization_id: str,
    gene_set_hash_value: str,
    n_genes: int,
    create: bool = False,
) -> Path:
    """``<cache_root>/normalization_<norm>/geneset_<hash>__n_<N>`` (optionally mkdir)."""
    ns = (
        Path(cache_root)
        / f"normalization_{normalization_id}"
        / f"geneset_{gene_set_hash_value}__n_{n_genes}"
    )
    if create:
        ns.mkdir(parents=True, exist_ok=True)
    return ns


def npz_path(namespace: Path, prefix: str, keys: Mapping[str, Any]) -> Path:
    """Canonical NPZ name: ``<prefix>__<k1>_<v1>__<k2>_<v2>...npz`` (sorted keys)."""
    parts = [prefix]
    for k in sorted(keys):
        parts.append(f"{k}_{safe_filename_part(keys[k])}")
    return Path(namespace) / ("__".join(parts) + ".npz")


def control_npz_path(namespace: Path, control_id_keys: Mapping[str, Any]) -> Path:
    """Path to the matched control NPZ for one ``control_id_keys`` group."""
    return npz_path(namespace, "control_cache", control_id_keys)


def perturbed_npz_path(namespace: Path, condition_keys: Mapping[str, Any]) -> Path:
    """Path to the perturbed truth NPZ for one ``condition_keys`` group."""
    return npz_path(namespace, "perturbed_cache", condition_keys)


# Eval index v3 (CSR-style mmap layout): keys parquet + flat int64 + indptr.
# Keys parquet is small (a few MB at most) and read eagerly.
# `flat` is the CSR-style concatenation of all per-group cell IDs (int64) and
# `indptr` gives [start, end) per group row, so a lookup is
# ``flat[indptr[row]:indptr[row+1]]`` — a zero-copy mmap view.
EVAL_INDEX_PERTURBED_KEYS_FILE = "perturbed_keys.parquet"
EVAL_INDEX_PERTURBED_FLAT_FILE = "perturbed_flat.npy"
EVAL_INDEX_PERTURBED_INDPTR_FILE = "perturbed_indptr.npy"
EVAL_INDEX_CONTROL_KEYS_FILE = "control_keys.parquet"
EVAL_INDEX_CONTROL_FLAT_FILE = "control_flat.npy"
EVAL_INDEX_CONTROL_INDPTR_FILE = "control_indptr.npy"

# Legacy v2 files (nested-list parquet); kept as constants so the writer can
# unlink them on rebuild. Nothing reads them.
EVAL_INDEX_LEGACY_PERTURBED_FILE = "condition_index.parquet"
EVAL_INDEX_LEGACY_CONTROL_FILE = "control_index.parquet"


def perturbed_keys_path(namespace: Path) -> Path:
    """Path to the perturbed-conditions keys parquet (one row per group)."""
    return Path(namespace) / EVAL_INDEX_PERTURBED_KEYS_FILE


def perturbed_flat_path(namespace: Path) -> Path:
    """Concatenated int64 obs indices for every perturbed condition."""
    return Path(namespace) / EVAL_INDEX_PERTURBED_FLAT_FILE


def perturbed_indptr_path(namespace: Path) -> Path:
    """CSR-style int64 offsets into ``perturbed_flat.npy`` (length n_rows+1)."""
    return Path(namespace) / EVAL_INDEX_PERTURBED_INDPTR_FILE


def control_keys_path(namespace: Path) -> Path:
    """Path to the control-pool keys parquet (one row per pool)."""
    return Path(namespace) / EVAL_INDEX_CONTROL_KEYS_FILE


def control_flat_path(namespace: Path) -> Path:
    """Concatenated int64 obs indices for every control pool."""
    return Path(namespace) / EVAL_INDEX_CONTROL_FLAT_FILE


def control_indptr_path(namespace: Path) -> Path:
    """CSR-style int64 offsets into ``control_flat.npy`` (length n_rows+1)."""
    return Path(namespace) / EVAL_INDEX_CONTROL_INDPTR_FILE


def perturbed_index_files(namespace: Path) -> tuple[Path, Path, Path]:
    """Triple of (keys.parquet, flat.npy, indptr.npy) for perturbed conditions."""
    return (
        perturbed_keys_path(namespace),
        perturbed_flat_path(namespace),
        perturbed_indptr_path(namespace),
    )


def control_index_files(namespace: Path) -> tuple[Path, Path, Path]:
    """Triple of (keys.parquet, flat.npy, indptr.npy) for control pools."""
    return (
        control_keys_path(namespace),
        control_flat_path(namespace),
        control_indptr_path(namespace),
    )


def read_eval_namespace_manifest(namespace: Path) -> dict[str, Any]:
    """Read eval cache namespace ``manifest.json`` (``mode=index_csr_v3``)."""
    path = Path(namespace) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Eval namespace manifest missing: {path}. "
            "Run scripts/data/build_cache.py eval after cache setup.",
        )
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid eval namespace manifest: {path}") from exc


def prepare_cache_obs_for_indexing(
    obs: pd.DataFrame,
    *,
    condition_variables: Sequence[str],
    control_condition: Optional[Mapping[str, Any]] = None,
    match_keys: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Validate obs columns and derive ``dose`` when needed for eval index build."""
    from .vocab import concentrations_from_obs as _concentrations_from_obs

    out = obs
    mcv = list(condition_variables)
    ctrl = dict(control_condition or {})
    mk = list(match_keys or [])
    needed = sorted(set(mcv) | set(ctrl.keys()) | set(mk))

    if "dose" in needed and "dose" not in out.columns and "drugname_drugconc" in out.columns:
        out = out.copy()
        out["dose"] = _concentrations_from_obs(out, key="drugname_drugconc")

    missing = [c for c in needed if c not in out.columns]
    if missing:
        raise KeyError(
            f"Cache obs missing columns required for eval index {missing}. "
            f"Available: {list(out.columns)}",
        )
    return out


def resolve_control_match_keys(
    condition_variables: Sequence[str],
    control_condition: Optional[Mapping[str, Any]],
    control_match_keys: Optional[Sequence[str]] = None,
) -> list[str]:
    """Columns identifying the matched control pool (build + eval must agree)."""
    ctrl = dict(control_condition or {})
    return list(control_match_keys or [v for v in condition_variables if v not in ctrl])


def resolve_eval_group_keys(
    condition_variables: Sequence[str],
    control_condition: Optional[Mapping[str, Any]] = None,
    control_match_keys: Optional[Sequence[str]] = None,
    eval_group_keys: Optional[Sequence[str]] = None,
) -> list[str]:
    """Variables that define one scored row in ``metrics_summary.results``.

    Default: all ``model_conditioning_variables`` (e.g. Tahoe
    ``cell_name, drug, dose, plate``; Parse ``donor, cytokine``).
    Metrics are computed per row; cross-plate means belong in plotting
    (see config ``plot_aggregate_over``).
    """
    del control_condition, control_match_keys  # reserved for explicit overrides only
    if eval_group_keys is not None:
        keys = list(eval_group_keys)
        missing = [k for k in keys if k not in condition_variables]
        if missing:
            raise ValueError(
                f"eval_group_keys {missing} not in model_conditioning_variables "
                f"{list(condition_variables)}.",
            )
        return keys
    return list(condition_variables)


def resolve_plot_aggregate_drop(
    condition_variables: Sequence[str],
    plot_aggregate_over: Optional[str] = None,
) -> tuple[str, ...]:
    """Axes to nanmean-over when building plot-level metric dicts.

    ``plot_aggregate_over`` from dataset config / ``metrics_summary.json``:

    * ``null``, omitted, or ``"none"`` → ``()`` (every scored row is its own
      plot point, including plate when present).
    * e.g. ``"plate"`` → average per-plate metrics before cross-condition plots.
    """
    if plot_aggregate_over is None:
        return ()
    key = str(plot_aggregate_over).strip()
    if not key or key.lower() == "none":
        return ()
    mcv = list(condition_variables)
    if key not in mcv:
        raise ValueError(
            f"plot_aggregate_over={key!r} not in model_conditioning_variables "
            f"{mcv}.",
        )
    return (key,)


def _ensure_obs_columns_categorical(
    obs: pd.DataFrame, columns: List[str],
) -> pd.DataFrame:
    """Cast non-categorical obs columns to categorical (in-place) so downstream
    predicate / encoder paths stay O(K) over categories rather than O(n_obs)
    on materialised strings. Idempotent."""
    for col in columns:
        if col not in obs.columns:
            continue
        if not isinstance(obs[col].dtype, pd.CategoricalDtype):
            obs[col] = obs[col].astype("category")
    return obs


def enrich_obs_with_metadata(obs: pd.DataFrame, config: Any) -> pd.DataFrame:
    """Left-merge ``config.metadata_joins`` (``left_on``, ``parquet``, ``columns``) onto obs.

    When the join key is a Categorical we never materialise len(obs) strings: we
    look up new columns on the join key's *categories* (typically 10²–10³ rows),
    factorize them, and gather via ``cat.codes``. This is the path that makes
    Tahoe metadata joins O(K) instead of O(100M).
    """
    import time as _time

    root = Path(__file__).resolve().parents[2]
    for j in getattr(config, "metadata_joins", None) or []:
        k, cols, p = j["left_on"], j["columns"], Path(j["parquet"])
        if k not in obs.columns:
            raise ValueError(f"metadata join left_on {k!r} not in obs")
        p = p if p.is_absolute() else root / p
        if not p.is_file():
            raise FileNotFoundError(f"metadata_join parquet {p} is not a readable file.")
        t = _time.monotonic()
        meta = pd.read_parquet(p)[[k, *cols]]
        if meta.groupby(k)[cols].nunique().max().max() > 1:
            raise ValueError(f"metadata join on {k!r} from {p} is not 1-to-1 for {cols!r}")
        meta = meta.drop_duplicates(k)
        new_cols = [c for c in cols if c not in obs.columns]
        if not new_cols:
            continue
        if isinstance(obs[k].dtype, pd.CategoricalDtype):
            unique_cats = pd.DataFrame({k: obs[k].cat.categories})
            mapped = unique_cats.merge(meta[[k, *new_cols]], on=k, how="left")
            codes = obs[k].cat.codes.to_numpy()
            for c in new_cols:
                mapped_codes, new_categories = pd.factorize(mapped[c])
                broadcast = np.where(codes == -1, -1, mapped_codes[codes])
                obs[c] = pd.Categorical.from_codes(broadcast, categories=new_categories)
        else:
            obs = obs.merge(meta[[k, *new_cols]], on=k, how="left")
        print(
            f"  Data: metadata join {k!r} <- {p.name} (+{len(new_cols)} cols) "
            f"[{_time.monotonic() - t:.1f}s]",
            flush=True,
        )
    return obs


def _coerce_split_config(
    split_config: Union[SplitConfig, Mapping[str, Any], Path, str],
) -> SplitConfig:
    if isinstance(split_config, SplitConfig):
        return split_config
    if isinstance(split_config, (str, Path)):
        import yaml

        raw = yaml.safe_load(Path(split_config).read_text()) or {}
        return SplitConfig.from_dict(raw)
    return SplitConfig.from_dict(split_config)


def drop_control_condition_rows(
    summary: pd.DataFrame,
    control_condition: Optional[Mapping[str, Any]],
) -> pd.DataFrame:
    """Remove rows matching ``control_condition`` (same rules as split predicates)."""
    from .vocab import to_categorical_str

    if not control_condition:
        return summary
    mask = pd.Series(True, index=summary.index)
    for k, v in control_condition.items():
        if k not in summary.columns:
            continue
        if pd.api.types.is_float_dtype(summary[k]):
            mask &= np.abs(summary[k].astype(float) - float(v)) < 1e-6
        else:
            mask &= summary[k].astype(str) == to_categorical_str(v)
    return summary.loc[~mask]


def train_membership_mask(
    summary: pd.DataFrame,
    cfg: SplitConfig,
) -> np.ndarray:
    """Row mask for train membership: not validation, not test, not excluded.

    Condition-summary counterpart of the ``is_train`` leg of
    :func:`src.data.splits.predicates.split_membership_masks`, including its
    ``train_exclusions`` step.  Control handling is left to the caller
    (``drop_control_rows``); with ``controls_to_train`` the control rows that
    a val/test predicate happens to match are train rows, but every eval
    caller drops controls anyway, so the distinction does not surface here.

    Note: this is *split membership*, not the trained-on set.  A split YAML
    whose ``train:`` block subsamples (``sample: hbss``, ``fraction: ...``)
    trains on a subset of these rows, and that subsample is not recoverable
    from the YAML alone -- it lives in the run's ``split_cache/``.  Only
    ``sample: all`` legs have train membership == trained-on.
    """
    from .splits.predicates import apply_split_predicate

    is_val = apply_split_predicate(summary, cfg.validation)
    is_test = apply_split_predicate(summary, cfg.test)
    is_train = ~(is_val | is_test)
    if getattr(cfg, "train_exclusions", None):
        is_train &= ~apply_split_predicate(summary, cfg.train_exclusions)
    return is_train


def filter_condition_summary_for_eval(
    summary: pd.DataFrame,
    *,
    split_config: Union[SplitConfig, Mapping[str, Any], Path, str],
    eval_splits: Sequence[str],
    control_condition: Optional[Mapping[str, Any]] = None,
    drop_control_rows: bool = False,
    min_cells: Optional[int] = None,
) -> pd.DataFrame:
    """Rows eligible for model-eval work queue (split YAML predicate on condition summary).

    Used by :func:`src.eval.cache.eval_cache_io.load_work_queue` to filter
    ``condition_summary`` for one eval split (drop controls, optional ``min_cells``).
    Eval truth matrices come from the per-condition NPZ cache under ``eval_cache/``.
    """
    from .splits.predicates import apply_split_predicate

    if len(summary) == 0:
        return summary.copy()

    cfg = _coerce_split_config(split_config)
    splits = tuple(str(s).strip().lower() for s in eval_splits)
    if not splits:
        raise ValueError("eval_splits must name at least one of validation, test, train.")
    unknown = [s for s in splits if s not in _SCORABLE_SPLITS]
    if unknown:
        raise KeyError(
            f"Unknown eval_splits {unknown!r}; allowed: {sorted(_SCORABLE_SPLITS)!r}.",
        )

    mask = np.zeros(len(summary), dtype=bool)
    for split in splits:
        if split == "train":
            # Not a predicate block -- the complement of val/test (see
            # train_membership_mask).  Scored for train-fit diagnostics.
            mask |= train_membership_mask(summary, cfg)
            continue
        predicate: PredicateBlock = getattr(cfg, split)
        mask |= apply_split_predicate(summary, predicate)

    filt = summary.loc[mask].copy()
    if drop_control_rows:
        filt = drop_control_condition_rows(filt, control_condition)
    if min_cells is not None:
        filt = filt[
            pd.to_numeric(filt["num_cells"], errors="coerce").fillna(0) >= int(min_cells)
        ].copy()
    return filt


def filter_condition_summary_for_train(
    summary: pd.DataFrame,
    *,
    split_config: Union[SplitConfig, Mapping[str, Any], Path, str],
    control_condition: Optional[Mapping[str, Any]] = None,
    drop_control_rows: bool = True,
    min_cells: Optional[int] = None,
) -> pd.DataFrame:
    """Rows eligible for cell-set training (train split = not val and not test).

    Mirrors :func:`split_membership_masks` at condition-summary granularity:
    perturbed train conditions are those outside validation/test predicates.
    """
    from .splits.predicates import apply_split_predicate

    if len(summary) == 0:
        return summary.copy()

    cfg = _coerce_split_config(split_config)
    is_val = apply_split_predicate(summary, cfg.validation)
    is_test = apply_split_predicate(summary, cfg.test)
    is_train = ~(is_val | is_test)
    filt = summary.loc[is_train].copy()
    if drop_control_rows:
        filt = drop_control_condition_rows(filt, control_condition)
    if min_cells is not None:
        filt = filt[
            pd.to_numeric(filt["num_cells"], errors="coerce").fillna(0) >= int(min_cells)
        ].copy()
    return filt


def count_eval_split_conditions(
    summary: Union[pd.DataFrame, Path, str],
    *,
    split_config: Union[SplitConfig, Mapping[str, Any], Path, str],
    eval_split: str,
    control_condition: Optional[Mapping[str, Any]] = None,
    min_cells: int = 4,
    drop_control_rows: bool = True,
) -> int:
    """Count conditions eligible for one eval split.

    Uses the same split-YAML predicate + ``min_cells`` filter as
    :func:`src.eval.cache.eval_cache_io.load_work_queue`. Default drops
    control rows; pass ``drop_control_rows=False`` to keep test PBS wells.
    """
    if isinstance(summary, (Path, str)):
        summary = pd.read_parquet(Path(summary))
    filt = filter_condition_summary_for_eval(
        summary,
        split_config=split_config,
        eval_splits=(eval_split,),
        control_condition=control_condition,
        drop_control_rows=drop_control_rows,
        min_cells=min_cells,
    )
    return int(len(filt))


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON object from file."""
    return json.loads(Path(path).read_text())


def save_json(path: Path, data: dict[str, Any]) -> None:
    """Atomically save a JSON object to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    """Save a compressed NPZ file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)




def _fast_fit_label_encoder(values: pd.Series) -> LabelEncoder:
    """Equivalent of ``LabelEncoder().fit(values.astype(str))`` without the
    100M-row Python-string materialisation.

    Calling ``obs[var].astype(str).values`` on the full obs column before
    fitting would, for Tahoe (~100M rows × 4 conditioning columns), allocate
    four 100M-entry object arrays of Python ``str``s and then run ``np.unique``
    over them — together ~50s per variable (~200s total).

    So we read uniques *first*:

        * ``pd.Categorical`` columns (the codes+categories sidecar layout
          produced by :class:`RawHvgShardCache`): use ``cat.categories``
          directly — typically 10–10³ entries — and never touch the 100M-row
          codes array.
        * any other dtype: ``pd.unique`` is C-level (hash table), ~1–2s
          for 100M rows; the ``str`` cast then runs over the small unique
          set, not the full column.

    Matches the "NaN → 'nan' string class" semantics of ``astype(str).fit``
    (sklearn sorts ``classes_``; we do too via a final ``np.unique``).

    Args:
        values: Pandas Series for one obs column.

    Returns:
        Fitted :class:`~sklearn.preprocessing.LabelEncoder`. Direct
        ``classes_`` assignment is a supported path: sklearn's
        :meth:`LabelEncoder.transform` only requires a sorted ``classes_``
        array (it calls ``np.searchsorted``).
    """
    if isinstance(values.dtype, pd.CategoricalDtype):
        cats = values.cat.categories.astype(str).to_numpy()
        if bool(values.isna().any()):
            cats = np.append(cats, "nan")
    else:
        uniques = pd.unique(values.to_numpy())
        cats = np.array([str(c) for c in uniques], dtype=object)
    classes = np.unique(cats)  # sorted + de-duplicated (sklearn invariant).
    le = LabelEncoder()
    le.classes_ = classes
    return le


def load_hvg_gene_names_from_file(path: Union[str, Path]) -> List[str]:
    """Gene names from ``.pkl`` (list) or text (one per line)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"HVG genes file not found: {path}")
    if path.suffix.lower() == ".pkl":
        with path.open("rb") as fin:
            raw = pickle.load(fin)
        genes = [str(g).strip() for g in raw if str(g).strip()]
    else:
        genes = [s.strip() for s in path.read_text().splitlines() if s.strip()]
    if not genes:
        raise ValueError(f"HVG genes file {path} is empty.")
    return list(dict.fromkeys(genes))


def load_hvg_indices_from_genes_file(
    path: Union[str, Path],
    var_names: List[str],
) -> tuple[np.ndarray, List[str]]:
    """Column indices and file gene list for genes in ``path`` (``.txt`` or ``.pkl``)."""
    genes = load_hvg_gene_names_from_file(path)
    idx = {str(g): i for i, g in enumerate(var_names)}
    present = [g for g in genes if g in idx]
    missing = [g for g in genes if g not in idx]
    if missing:
        logger.warning(
            "HVG genes file %s: %d/%d not in dataset (%d vars); examples: %s",
            path,
            len(missing),
            len(genes),
            len(var_names),
            sorted(missing)[:10],
        )
    if not present:
        raise ValueError(
            f"HVG genes file {path} lists no genes present in dataset "
            f"({len(var_names)} vars)."
        )
    return np.array([idx[g] for g in present], dtype=np.int64), genes


def scipy_csr_to_torch_sparse_csr(mat: sparse.csr_matrix) -> torch.Tensor:
    """Convert a SciPy CSR matrix to ``torch.sparse_csr_tensor`` (CPU)."""
    mat = mat.tocsr()
    data = np.asarray(mat.data, dtype=np.float32)
    indptr = np.asarray(mat.indptr, dtype=np.int64)
    indices = np.asarray(mat.indices, dtype=np.int64)
    return torch.sparse_csr_tensor(
        torch.from_numpy(indptr),
        torch.from_numpy(indices),
        torch.from_numpy(data),
        size=tuple(mat.shape),
        dtype=torch.float32,
    )


def densify_gene_expression(t: torch.Tensor) -> torch.Tensor:
    """Return dense (B, G) gene expression; no-op if already dense."""
    if t.layout == torch.sparse_csr:
        return t.to_dense()
    return t


def _numpy_batch_to_tensor(batch: Any) -> Dict[str, Any]:
    """Convert dataset batch dict to torch (CPU). Used as collate_fn.

    gene_expression may be NumPy (dense) or SciPy CSR (sparse); other keys are NumPy.
    ``cond_idx`` if present must be a dict mapping column names to int64 ndarray;
    it is converted to tensors while preserving dict structure.
    With batch_size=None the DataLoader may pass a list of one batch; we accept that.
    Also accepts MultiIndexable (has .items() yielding (name, array) pairs).
    """
    if isinstance(batch, list) and len(batch) == 1:
        batch = batch[0]
    if not isinstance(batch, dict) and not hasattr(batch, "items"):
        raise TypeError(f"Expected dict-like batch, got {type(batch)}")
    out: Dict[str, Any] = {}
    for key, v in batch.items():
        if key == "cond_idx":
            if not isinstance(v, dict):
                raise TypeError(f"cond_idx must be a dict, got {type(v).__name__}.")
            out[key] = {
                ck: torch.from_numpy(cv) if isinstance(cv, np.ndarray) else cv
                for ck, cv in v.items()
            }
            continue
        if key == "gene_expression" and sparse.issparse(v):
            out[key] = scipy_csr_to_torch_sparse_csr(v)
        elif isinstance(v, np.ndarray):
            out[key] = torch.from_numpy(v)
        elif sparse.issparse(v):
            raise TypeError(
                f"Unexpected sparse array for key {key!r}; "
                "only gene_expression may be sparse."
            )
        else:
            out[key] = v
    return out


def probe_normalization_id(
    n_cells: int,
    get_expression_row: Callable[[int], np.ndarray],
    *,
    n_probe: int = 10,
    seed: int | None = None,
) -> str:
    """Vote ``raw_counts`` or ``log1p`` from random single-cell expression vectors."""
    n = int(n_cells)
    if n <= 0:
        raise ValueError(f"probe_normalization_id requires n_cells > 0, got {n}.")
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(int(n_probe), n), replace=False)
    votes = 0
    for i in idx:
        x = np.asarray(get_expression_row(int(i)), dtype=np.float64).ravel()
        nz = x[x > 0]
        if x.max() > 50 and nz.size and np.allclose(nz, np.rint(nz), atol=1e-6):
            votes += 1
    return "raw_counts" if votes > len(idx) // 2 else "log1p"
