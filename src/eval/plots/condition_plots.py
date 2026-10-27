"""Per-condition gene-level scatter and histogram plots."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .bundles import GeneLevelArrays
from .conditions import safe_filename_part
from .constants import PLOT_TYPES, PRED_COLOR, TRUE_COLOR
from .metric_labels import norm_display_label
from .scatter import draw_scatter_panel

logger = logging.getLogger(__name__)


def plot_condition_scatter_mean_var(
    true_mean: np.ndarray,
    pred_mean: np.ndarray,
    true_var: np.ndarray,
    pred_var: np.ndarray,
    condition_label: str,
    normalization_label: str,
    out_dir: Path,
    *,
    figsize: tuple[float, float] = (14, 6),
    dpi: int = 200,
    alpha: float = 0.25,
    s: float = 4.0,
    filename_suffix: str = "",
) -> Path:
    """Two-panel scatter of true vs predicted gene mean + variance.

    Args:
        true_mean: Shape ``(n_genes,)``.
        pred_mean: Shape ``(n_genes,)``.
        true_var: Shape ``(n_genes,)``.
        pred_var: Shape ``(n_genes,)``.
        condition_label: Human-readable condition string (``"cell | drug | dose"``).
        normalization_label: Display label for the x/y axis (e.g. ``"log1p CPM"``).
        out_dir: Output directory (created if missing).
        figsize: Matplotlib figsize for the two-panel figure.
        dpi: Resolution.
        alpha: Scatter point transparency.
        s: Scatter point size.
        filename_suffix: Extra string appended before ``.png`` (e.g. space tag).

    Returns:
        Path to the saved combined PNG.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, (true_arr, pred_arr, panel) in zip(
        axes,
        [
            (true_mean, pred_mean, "Gene Mean"),
            (true_var, pred_var, "Gene Variance"),
        ],
    ):
        draw_scatter_panel(
            ax, true_arr, pred_arr,
            xlabel=f"True {panel} ({normalization_label})",
            ylabel=f"Predicted {panel} ({normalization_label})",
            title=f"{condition_label} — {panel}",
            alpha=alpha, s=s,
        )

    fig.suptitle(
        f"Pseudobulk Scatter — {condition_label}",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    safe_label = safe_filename_part(condition_label)
    tag = f"_{filename_suffix}" if filename_suffix else ""
    path = out_dir / f"scatter_mean_var{tag}__{safe_label}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Scatter saved: %s", path)
    return path


def plot_gene_variance_histogram(
    true_var: np.ndarray,
    pred_var: np.ndarray,
    condition_label: str,
    normalization_label: str,
    out_dir: Path,
    *,
    dpi: int = 200,
    bins: int = 80,
    filename_suffix: str = "",
) -> Path:
    """Overlaid histogram of per-gene variance (true vs pred)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(true_var, bins=bins, alpha=0.5, density=True, label="True perturbed", color=TRUE_COLOR)
    ax.hist(pred_var, bins=bins, alpha=0.5, density=True, label="Predicted", color=PRED_COLOR)
    ax.set_xlabel(f"Gene Variance ({normalization_label})", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(
        f"Gene Variance Distribution — {condition_label}",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    stats_text = (
        f"True: median={np.median(true_var):.4f}\n"
        f"Pred: median={np.median(pred_var):.4f}\n"
        f"n_genes={len(true_var)}"
    )
    ax.text(
        0.97, 0.95, stats_text, transform=ax.transAxes, va="top", ha="right",
        fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    safe_label = safe_filename_part(condition_label)
    tag = f"_{filename_suffix}" if filename_suffix else ""
    path = out_dir / f"hist_gene_variance{tag}__{safe_label}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Gene variance histogram -> %s", path)
    return path


def resolve_plot_set(plots_arg: Sequence[str]) -> set[str]:
    """Resolve a user-provided plots list (may contain ``"all"`` / ``"none"``)."""
    result: set[str] = set()
    for p in plots_arg:
        if p == "all":
            result.update(PLOT_TYPES)
        elif p == "none":
            pass
        elif p in PLOT_TYPES:
            result.add(p)
        else:
            logger.warning("Unknown plot type %r; ignoring (valid: %s)", p, PLOT_TYPES)
    return result


def plot_condition_all(
    arrays: GeneLevelArrays,
    out_dir: Path,
    plots: set[str],
    *,
    dpi: int = 200,
) -> list[Path]:
    """Generate requested per-condition plots from a :class:`GeneLevelArrays`."""
    if not plots:
        return []
    saved: list[Path] = []
    norm_label = norm_display_label(arrays.space)
    if "scatter_mean_var" in plots:
        saved.append(
            plot_condition_scatter_mean_var(
                true_mean=arrays.true_mean,
                pred_mean=arrays.pred_mean,
                true_var=arrays.true_var,
                pred_var=arrays.pred_var,
                condition_label=arrays.condition_label,
                normalization_label=norm_label,
                out_dir=out_dir,
                dpi=dpi,
                filename_suffix=arrays.space,
            )
        )
    if "gene_var_hist" in plots:
        saved.append(
            plot_gene_variance_histogram(
                true_var=arrays.true_var,
                pred_var=arrays.pred_var,
                condition_label=arrays.condition_label,
                normalization_label=norm_label,
                out_dir=out_dir,
                dpi=dpi,
                filename_suffix=arrays.space,
            )
        )
    return saved
