"""Multi-model summary CSV and win-rate sidecar artifacts."""

from __future__ import annotations

import csv
import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .boxplots import _comparison_boxplot_rows, _truncate_middle_label
from .conditions import safe_filename_part
from .metric_labels import is_lower_better, pretty_metric_name

logger = logging.getLogger(__name__)


def _build_metric_value_matrix(
    *,
    metric: str,
    model_order: Sequence[str],
    model_data: dict[str, dict[str, dict[str, float]]],
) -> tuple[list[str], list[str], np.ndarray]:
    """Stack per-model condition values into an ``(M, N)`` float matrix.

    Args:
        metric: Metric key (e.g.
            ``deg_raw_int/deg_overlap_top_p_ranked_k20_fdr0p001_lfc0p1_adaptivek_detfloor10``).
        model_order: Candidate model ids (rows of the returned matrix).
        model_data: ``model_id -> condition_key -> metric_dict``.

    Returns:
        Tuple ``(kept_models, conds, values)``:

        * ``kept_models`` -- subset of ``model_order`` that has at least one
          finite value for ``metric``.
        * ``conds`` -- sorted union of all condition keys that have a finite
          value for ``metric`` for at least one kept model.
        * ``values`` -- ``(len(kept_models), len(conds))`` float64 array;
          missing / non-finite entries are ``np.nan``.
    """
    kept_models: list[str] = []
    cond_set: set[str] = set()
    for mid in model_order:
        per_cond = model_data.get(mid, {})
        has_any = False
        for ck, md in per_cond.items():
            if metric not in md:
                continue
            v = float(md[metric])
            if not math.isfinite(v):
                continue
            has_any = True
            cond_set.add(ck)
        if has_any:
            kept_models.append(mid)

    conds = sorted(cond_set)
    if not kept_models or not conds:
        return kept_models, conds, np.empty((len(kept_models), len(conds)), dtype=np.float64)

    values = np.full((len(kept_models), len(conds)), np.nan, dtype=np.float64)
    for i, mid in enumerate(kept_models):
        per_cond = model_data.get(mid, {})
        for j, ck in enumerate(conds):
            md = per_cond.get(ck)
            if md is None or metric not in md:
                continue
            v = float(md[metric])
            if math.isfinite(v):
                values[i, j] = v
    return kept_models, conds, values


def _compute_pairwise_winrate(
    values: np.ndarray,
    *,
    lower_better: bool,
    tie_rtol: float = 1e-9,
    tie_atol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise win-rate matrix from a model x condition value matrix.

    For every ordered pair ``(i, j)``, only conditions where both ``values[i]``
    and ``values[j]`` are finite are eligible.  A tie (``np.isclose``) counts as
    ``0.5`` for both directions, so ``W[i, j] + W[j, i] == 1`` whenever any
    eligible conditions exist.  The diagonal is ``NaN``.

    Args:
        values: ``(M, N)`` per-model per-condition values; non-finite = missing.
        lower_better: If True, smaller is better for this metric.
        tie_rtol: Relative tolerance for ``np.isclose``.
        tie_atol: Absolute tolerance for ``np.isclose``.

    Returns:
        ``(W, n_compared)`` both shape ``(M, M)``.  ``W`` is in ``[0, 1]`` (NaN
        on the diagonal or empty pairs); ``n_compared[i, j]`` is the number of
        conditions with both finite values.
    """
    m = values.shape[0]
    W = np.full((m, m), np.nan, dtype=np.float64)
    n_cmp = np.zeros((m, m), dtype=np.int64)
    if m == 0 or values.shape[1] == 0:
        return W, n_cmp

    finite = np.isfinite(values)
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            mask = finite[i] & finite[j]
            n = int(mask.sum())
            n_cmp[i, j] = n
            if n == 0:
                continue
            vi = values[i, mask]
            vj = values[j, mask]
            ties = np.isclose(vi, vj, rtol=tie_rtol, atol=tie_atol)
            if lower_better:
                wins = (vi < vj) & (~ties)
            else:
                wins = (vi > vj) & (~ties)
            W[i, j] = (float(wins.sum()) + 0.5 * float(ties.sum())) / float(n)
    return W, n_cmp


def _compute_best_of_all_winrate(
    values: np.ndarray,
    *,
    lower_better: bool,
    tie_rtol: float = 1e-9,
    tie_atol: float = 1e-12,
) -> tuple[np.ndarray, int]:
    """Per-model fraction of conditions where this model is the (tied-)best.

    For each condition with at least two finite values, the strict best gets a
    weight of ``1 / T`` (``T`` = number of tied-best models).  Conditions where
    only a single model has data are ignored (no comparison possible).

    Args:
        values: ``(M, N)`` per-model per-condition values; non-finite = missing.
        lower_better: If True, smaller is better for this metric.
        tie_rtol: Relative tolerance for tied-best detection.
        tie_atol: Absolute tolerance for tied-best detection.

    Returns:
        ``(rates, n_eligible)``: ``rates`` length ``M`` in ``[0, 1]`` (NaN if
        ``n_eligible == 0``), and the count of conditions used.
    """
    m, n = values.shape
    if m == 0 or n == 0:
        return np.full(m, np.nan, dtype=np.float64), 0

    finite = np.isfinite(values)
    n_finite_per_cond = finite.sum(axis=0)
    eligible_mask = n_finite_per_cond >= 2
    n_eligible = int(eligible_mask.sum())
    if n_eligible == 0:
        return np.full(m, np.nan, dtype=np.float64), 0

    weights = np.zeros(m, dtype=np.float64)
    for j in np.where(eligible_mask)[0]:
        col = values[:, j]
        mask_j = finite[:, j]
        if lower_better:
            best_val = np.nanmin(col)
        else:
            best_val = np.nanmax(col)
        is_best = mask_j & np.isclose(col, best_val, rtol=tie_rtol, atol=tie_atol)
        n_best = int(is_best.sum())
        if n_best == 0:
            continue
        weights[is_best] += 1.0 / float(n_best)

    return weights / float(n_eligible), n_eligible


def _save_model_winrate_artifacts(
    *,
    metric: str,
    model_order: Sequence[str],
    model_id_labels: dict[str, str],
    model_data: dict[str, dict[str, dict[str, float]]],
    out_dir: Path,
    dpi: int,
) -> list[Path]:
    """Per-metric pairwise win-rate heatmap + best-of-all win-rate bar.

    Three artifacts written under ``out_dir``:

    * ``model_pairwise_winrate__<metric>.csv`` -- full ``(M, M)`` win-rate
      matrix (rows beat columns), plus per-pair sample sizes.
    * ``model_pairwise_winrate__<metric>.png`` -- annotated heatmap of the
      same matrix (models sorted by mean pairwise win-rate, best last).
    * ``model_winrate_summary__<metric>.png`` -- two-panel bar chart with the
      best-of-all win rate (fraction of conditions where this model is the
      single best) and the mean pairwise win rate (row mean of the heatmap).

    The metric's "lower is better" flag (via :func:`is_lower_better`) is applied
    consistently to both win-rate computations.  Ties are treated as half-wins
    in pairwise comparisons and shared evenly in best-of-all.

    Args:
        metric: Metric key (e.g. ``log1p_cpm/r2``).
        model_order: Candidate model ids to consider.
        model_id_labels: Display labels for model ids.
        model_data: Condition-level metric map by model id.
        out_dir: Summary output directory.
        dpi: Figure DPI.

    Returns:
        Saved artifact paths.  Empty when fewer than two models have any
        finite values for the metric.
    """
    kept_models, conds, values = _build_metric_value_matrix(
        metric=metric, model_order=model_order, model_data=model_data,
    )
    if len(kept_models) < 2 or len(conds) == 0:
        logger.info(
            "Metric %s: <2 models with finite values; skipping win-rate artifacts.",
            metric,
        )
        return []

    lower_better = is_lower_better(metric)
    W, n_cmp = _compute_pairwise_winrate(values, lower_better=lower_better)
    boa_rates, n_eligible = _compute_best_of_all_winrate(
        values, lower_better=lower_better,
    )

    safe_m = safe_filename_part(metric.replace("/", "-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    labels = [model_id_labels.get(mid, mid) for mid in kept_models]
    pairwise_means = np.nanmean(W, axis=1)
    order = list(np.argsort(pairwise_means))  # worst-first
    ordered_ids = [kept_models[i] for i in order]
    ordered_labels = [labels[i] for i in order]
    W_ord = W[np.ix_(order, order)]
    n_ord = n_cmp[np.ix_(order, order)]
    boa_ord = boa_rates[order]
    pairwise_ord = pairwise_means[order]

    csv_path = out_dir / f"model_pairwise_winrate__{safe_m}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["row_model_id", "row_model_label"]
            + [f"vs::{lid}" for lid in ordered_labels]
            + ["mean_pairwise_winrate", "best_of_all_winrate"]
            + [f"n_compared::{lid}" for lid in ordered_labels]
        )
        for i, mid in enumerate(ordered_ids):
            row = [mid, ordered_labels[i]]
            row.extend(
                "" if not math.isfinite(float(W_ord[i, j])) else f"{float(W_ord[i, j]):.6g}"
                for j in range(len(ordered_ids))
            )
            row.append(
                "" if not math.isfinite(float(pairwise_ord[i])) else f"{float(pairwise_ord[i]):.6g}"
            )
            row.append(
                "" if not math.isfinite(float(boa_ord[i])) else f"{float(boa_ord[i]):.6g}"
            )
            row.extend(int(n_ord[i, j]) for j in range(len(ordered_ids)))
            writer.writerow(row)
    saved.append(csv_path)
    logger.info("Saved pairwise win-rate CSV: %s", csv_path)

    n = len(ordered_ids)
    fig_size = max(6.0, 0.55 * n + 3.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    cmap = plt.get_cmap("RdBu_r")
    cmap.set_bad(color="#dddddd")
    im = ax.imshow(W_ord, vmin=0.0, vmax=1.0, cmap=cmap, aspect="equal")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(ordered_labels, rotation=90, fontsize=8)
    ax.set_yticklabels(ordered_labels, fontsize=8)
    ax.set_xlabel("Opponent model (column)", fontsize=11)
    ax.set_ylabel("Model (row)", fontsize=11)
    direction = "lower better" if lower_better else "higher better"
    ax.set_title(
        f"{pretty_metric_name(metric)} — pairwise win rate (row beats column)\n"
        f"({direction}; ties = 0.5; rows ranked worst → best)",
        fontsize=13, fontweight="bold",
    )
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            v = float(W_ord[i, j])
            if not math.isfinite(v):
                continue
            text_color = "white" if (v >= 0.7 or v <= 0.3) else "black"
            ax.text(
                j, i, f"{v:.2f}",
                ha="center", va="center",
                fontsize=7, color=text_color,
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Win rate", fontsize=10)
    fig.tight_layout()
    heatmap_path = out_dir / f"model_pairwise_winrate__{safe_m}.png"
    fig.savefig(heatmap_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    saved.append(heatmap_path)
    logger.info("Saved pairwise win-rate heatmap: %s", heatmap_path)

    fig, axes = plt.subplots(
        1, 2, figsize=(max(10.0, 0.55 * n + 5.0), 5.0), sharey=True,
    )
    y_pos = np.arange(n)
    axes[0].barh(y_pos, boa_ord, color="#0072B2", alpha=0.85)
    axes[0].axvline(1.0 / float(n), color="black", linestyle=":", linewidth=1.0,
                    label=f"chance (1/{n})")
    axes[0].set_xlabel("Best-of-all win rate", fontsize=11)
    axes[0].set_xlim(0.0, max(0.05, float(np.nanmax(boa_ord)) * 1.15) if n_eligible else 1.0)
    axes[0].set_title(
        f"Best-of-all win rate\n({n_eligible} eligible conditions)",
        fontsize=12, fontweight="bold",
    )
    axes[0].legend(fontsize=8, loc="lower right")
    for i, v in enumerate(boa_ord):
        if math.isfinite(float(v)):
            axes[0].text(float(v), i, f"  {float(v):.2f}", va="center", fontsize=8)

    axes[1].barh(y_pos, pairwise_ord, color="#D55E00", alpha=0.85)
    axes[1].axvline(0.5, color="black", linestyle=":", linewidth=1.0, label="parity (0.5)")
    axes[1].set_xlabel("Mean pairwise win rate", fontsize=11)
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_title("Mean pairwise win rate", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=8, loc="lower right")
    for i, v in enumerate(pairwise_ord):
        if math.isfinite(float(v)):
            axes[1].text(float(v), i, f"  {float(v):.2f}", va="center", fontsize=8)

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(ordered_labels, fontsize=8)
    axes[0].set_ylabel("Model (worst → best by mean pairwise win rate)", fontsize=10)
    fig.suptitle(
        f"{pretty_metric_name(metric)} — model win rates "
        f"({'lower better' if lower_better else 'higher better'})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    bar_path = out_dir / f"model_winrate_summary__{safe_m}.png"
    fig.savefig(bar_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    saved.append(bar_path)
    logger.info("Saved win-rate summary bar plot: %s", bar_path)

    return saved


def _save_model_metric_summary_artifacts(
    *,
    metric: str,
    nc_mean: dict[str, dict[str, float]],
    model_order: Sequence[str],
    model_id_labels: dict[str, str],
    model_data: dict[str, dict[str, dict[str, float]]],
    out_dir: Path,
) -> list[Path]:
    """Write per-metric ranking summary CSV (ceiling + models).

    The main cross-condition figure is already a boxplot; this sidecar only
    persists the numeric summary table.
    """
    if nc_mean:
        cond_keys = sorted(
            ck for ck, md in nc_mean.items()
            if metric in md and math.isfinite(float(md[metric]))
        )
    else:
        cond_keys = sorted({
            ck
            for md in model_data.values()
            for ck, block in md.items()
            if metric in block and math.isfinite(float(block[metric]))
        })
    if not cond_keys:
        return []

    box_rows = _comparison_boxplot_rows(
        metric=metric,
        nc_mean=nc_mean,
        model_data=model_data,
        model_order=model_order,
        model_id_labels=model_id_labels,
        cond_keys=cond_keys,
    )
    if not box_rows:
        return []

    csv_rows: list[dict[str, Any]] = []
    for i, row in enumerate(box_rows, start=1):
        arr = np.asarray(row["values"], dtype=np.float64)
        csv_rows.append({
            "rank_worst_to_best": i,
            "series_kind": str(row["kind"]),
            "series_id": str(row.get("model_id", "noise_ceiling")),
            "series_label": str(row["label"]),
            "n_conditions": int(arr.size),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        })

    safe_m = safe_filename_part(metric.replace("/", "-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"model_metric_summary__{safe_m}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank_worst_to_best",
                "series_kind",
                "series_id",
                "series_label",
                "n_conditions",
                "mean",
                "median",
                "std",
                "min",
                "max",
            ],
        )
        writer.writeheader()
        for r in csv_rows:
            writer.writerow(
                {
                    "rank_worst_to_best": r["rank_worst_to_best"],
                    "series_kind": r["series_kind"],
                    "series_id": r["series_id"],
                    "series_label": r["series_label"],
                    "n_conditions": r["n_conditions"],
                    "mean": f"{float(r['mean']):.8g}",
                    "median": f"{float(r['median']):.8g}",
                    "std": f"{float(r['std']):.8g}",
                    "min": f"{float(r['min']):.8g}",
                    "max": f"{float(r['max']):.8g}",
                }
            )
    logger.info("Saved metric summary CSV: %s", csv_path)
    return [csv_path]
