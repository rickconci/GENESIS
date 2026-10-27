"""Reporting: which metric keys exist, how they are grouped, and how they leave the GPU.

Nothing here computes a metric. :mod:`.eval_metrics` scores; this module names the
metric sets (``configs/eval/metrics.yaml``), groups flat keys for
``metrics_summary.json``, picks the headline keys the logs show first, and turns
0-d device tensors into floats in one host transfer.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from ...paths import PERTGEN_REPO_ROOT

# ---------------------------------------------------------------------------
# Which keys exist: headline keys for the logs, metric sets, JSON groups
# ---------------------------------------------------------------------------

# Logged first in each bundle's detail line.
_PSEUDOBULK_HEADLINE: tuple[str, ...] = (
    "pearson_r", "mse", "mae", "r2", "r2_diff",
    "pearson_r_diff", "spearman_r_diff",
)

_DEG_HEADLINE: tuple[str, ...] = (
    "jaccard_sig_fdr0p001_lfc0p1_detfloor10",
    "n_sig_degs_a_fdr0p001_lfc0p1_detfloor10",
    "n_sig_degs_b_fdr0p001_lfc0p1_detfloor10",
    "spearman_lfc_sig_fdr0p001_lfc0p1_detfloor10",
    "spearman_z_dir_truth_top_k100_fdr0p001_lfc0p1_detfloor10",
    "roc_auc_fdr0p001_lfc0p1_detfloor10",
    "pr_auc_dir_match_fdr0p001_lfc0p1_detfloor10",
)

# Grouping written into ``metrics_summary.json`` ``results[].metrics``.
METRIC_GROUP_ORDER: tuple[str, ...] = (
    "deg",
    "pseudobulk_arm",
    "pseudobulk_delta",
    "distributional",
)
_DELTA_METRIC_NAMES: frozenset[str] = frozenset({
    "r2_diff",
    "pearson_r_diff",
    "spearman_r_diff",
    "delta_l2_sq_norm_ratio",
})
_DISTRIBUTIONAL_METRIC_NAMES: frozenset[str] = frozenset({"mmd_rbf", "energy_distance"})


def load_metric_set(name: str) -> frozenset[str] | None:
    """Keys of metric set ``name`` in ``configs/eval/metrics.yaml``; ``None`` means every key."""
    import yaml

    path = PERTGEN_REPO_ROOT / "configs" / "eval" / "metrics.yaml"
    sets = yaml.safe_load(path.read_text())
    if name not in sets:
        raise KeyError(f"unknown metric set {name!r}; {path} defines {sorted(sets)}")
    keys = sets[name]
    if keys == "all":
        return None
    if not isinstance(keys, list) or not keys:
        raise ValueError(f"metric set {name!r} in {path} must be `all` or a non-empty key list")
    return frozenset(keys)


def check_metric_keys(computed: Mapping[str, Any], metrics: frozenset[str] | None) -> None:
    """Raise when a requested key was not computed: a typo in metrics.yaml must not vanish."""
    missing = sorted(set(metrics or ()) - set(computed))
    if missing:
        raise KeyError(f"metric set asks for keys no kernel emits: {missing}")


def metric_group_for_key(key: str) -> str:
    """Classify a flat ``<space>/<name>`` or ``deg_<space>/<name>`` key."""
    if key.startswith("deg_"):
        return "deg"
    metric_name = key.split("/", 1)[-1]
    if metric_name in _DISTRIBUTIONAL_METRIC_NAMES:
        return "distributional"
    if metric_name in _DELTA_METRIC_NAMES:
        return "pseudobulk_delta"
    return "pseudobulk_arm"


def cluster_flat_metrics(
    flat: Mapping[str, float],
) -> dict[str, dict[str, float]]:
    """Bucket flat metric dict into the four JSON groups for one condition."""
    grouped: dict[str, dict[str, float]] = {
        name: {} for name in METRIC_GROUP_ORDER
    }
    for key, value in flat.items():
        group = metric_group_for_key(str(key))
        grouped[group][str(key)] = float(value)
    return {name: grouped[name] for name in METRIC_GROUP_ORDER if grouped[name]}


def flatten_clustered_metrics(
    metrics: Mapping[str, Any],
) -> dict[str, float]:
    """Flatten grouped ``metrics``; an already-flat dict passes through."""
    if not metrics:
        return {}
    first = next(iter(metrics.values()))
    if isinstance(first, dict):
        out: dict[str, float] = {}
        for name in METRIC_GROUP_ORDER:
            block = metrics.get(name)
            if isinstance(block, Mapping):
                out.update({str(k): float(v) for k, v in block.items()})
        for name, block in metrics.items():
            if name in METRIC_GROUP_ORDER or not isinstance(block, Mapping):
                continue
            out.update({str(k): float(v) for k, v in block.items()})
        return out
    return {str(k): float(v) for k, v in metrics.items()}



# ---------------------------------------------------------------------------
# One host sync per bundle
# ---------------------------------------------------------------------------


def materialize_metric_tensor_batch(
    tensor_rows: list[dict[str, torch.Tensor]],
) -> list[dict[str, float]]:
    """0-d tensor dicts (the same keys in every row) -> float dicts, in ONE host transfer.

    Every primitive returns 0-d device tensors so K reps accumulate without a
    per-scalar ``.item()`` sync. Values are cast to float32 so integer counts
    and float64 statistics stack together.
    """
    if not tensor_rows:
        return []
    keys = list(tensor_rows[0].keys())
    n_keys = len(keys)
    flat = [row[k].to(torch.float32) for row in tensor_rows for k in keys]
    values = torch.stack(flat).detach().cpu().tolist()
    return [
        dict(zip(keys, values[i * n_keys:(i + 1) * n_keys]))
        for i in range(len(tensor_rows))
    ]


def materialize_metric_tensors(tensors: dict[str, torch.Tensor]) -> dict[str, float]:
    """One dict of 0-d tensors -> floats, in one host transfer (key order kept)."""
    return materialize_metric_tensor_batch([tensors])[0] if tensors else {}
