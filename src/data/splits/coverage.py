from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .data_class import SplitConfig

logger = logging.getLogger(__name__)

PredicateBlock = Union[dict[str, Any], list[dict[str, Any]], None]


def _str_col(obs: pd.DataFrame, col: str) -> pd.Series:
    if col not in obs.columns:
        raise ValueError(f"Column {col!r} not found in obs.")
    return obs[col].astype(str)


def plot_axes_from_registry(bio_cols: Sequence[str], pert_cols: Sequence[str]) -> tuple[str, str, tuple[str, ...]]:
    if not bio_cols:
        raise ValueError("biological_context_columns must be non-empty.")
    if not pert_cols:
        raise ValueError("perturbation_columns must be non-empty.")
    bio, pert = bio_cols[0], pert_cols[0]
    return bio, pert, tuple(c for c in pert_cols[1:] if c != pert)


def collect_literal_predicate_values(predicate: PredicateBlock, column: str) -> set[str]:
    if not predicate:
        return set()
    out: set[str] = set()
    for block in (predicate if isinstance(predicate, list) else [predicate]):
        if column not in block:
            continue
        val = block[column]
        if str(val).lower() in ("any", "false"):
            continue
        out.update(str(v) for v in val) if isinstance(val, list) else out.add(str(val))
    return out


def holdout_axis_values(split_cfg: SplitConfig, column: str) -> set[str]:
    return collect_literal_predicate_values(split_cfg.validation, column) | collect_literal_predicate_values(split_cfg.test, column)


def marginalize_combination_counts(
    obs: pd.DataFrame, bio_column: str, pert_column: str, *, marginalize_columns: Sequence[str] = (),
) -> pd.Series:
    if obs.empty:
        return pd.Series(dtype=np.int64)
    frame = pd.DataFrame({bio_column: _str_col(obs, bio_column), pert_column: _str_col(obs, pert_column)})
    counts = frame.groupby([bio_column, pert_column], observed=True).size()
    counts.index.set_names([bio_column, pert_column], inplace=True)
    return counts.astype(np.int64)


def build_presence_matrix(
    train_obs: pd.DataFrame, bio_values: Sequence[str], pert_rows: Sequence[str],
    bio_column: str, pert_column: str, *, marginalize_columns: Sequence[str] = (), use_counts: bool = False,
) -> np.ndarray:
    lookup = marginalize_combination_counts(train_obs, bio_column, pert_column, marginalize_columns=marginalize_columns).to_dict()
    mat = np.zeros((len(pert_rows), len(bio_values)), dtype=np.float64)
    for i, pert in enumerate(pert_rows):
        for j, bio in enumerate(bio_values):
            val = lookup.get((bio, pert), 0)
            mat[i, j] = float(val) if use_counts else float(val > 0)
    return mat


def _stable_partition(values: Sequence[str], member: set[str], *, member_first: bool) -> list[str]:
    """Keep ``values`` order; move ``member`` hits to the front or back."""
    hit = [v for v in values if v in member]
    rest = [v for v in values if v not in member]
    return [*hit, *rest] if member_first else [*rest, *hit]


def sorted_plot_pert_rows(obs: pd.DataFrame, pert_column: str, control_condition: Mapping[str, Any]) -> tuple[list[str], str]:
    pert_vals = sorted(_str_col(obs, pert_column).unique())
    ctrl = str(control_condition.get(pert_column, ""))
    if ctrl in pert_vals:
        pert_vals.remove(ctrl)
        return [ctrl, *pert_vals], ctrl
    return pert_vals, ctrl


def coverage_plot_axes(
    obs: pd.DataFrame,
    train_obs: pd.DataFrame,
    bio_column: str,
    pert_column: str,
    split_cfg: SplitConfig,
    control_condition: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    """Coverage-plot order: test bios rightmost, test perts topmost under control.

    Test perts split into two tiers so the top-right corner reads as pure holdout:
    perts with no train rows at all on the test bios come first, then perts that
    still have some. The second tier exists because the heatmap marginalizes over
    the remaining pert columns — on Tahoe, 69 drugs are held out at doses 0.05/0.5
    but not 5.0, so their dose-collapsed cell is legitimately filled.
    """
    test_bios = collect_literal_predicate_values(split_cfg.test, bio_column)
    bio_values = _stable_partition(
        sorted(_str_universe(obs, bio_column)), test_bios, member_first=False,
    )
    test_perts = collect_literal_predicate_values(split_cfg.test, pert_column)
    on_test_bios = train_obs[_str_col(train_obs, bio_column).isin(test_bios)]
    partly_trained = {
        p for _bio, p in marginalize_combination_counts(on_test_bios, bio_column, pert_column).index
    }
    ctrl = str(control_condition.get(pert_column, ""))

    def tier(pert: str) -> int:
        if pert == ctrl:
            return 0
        if pert not in test_perts:
            return 3
        return 2 if pert in partly_trained else 1

    pert_rows = sorted(_str_universe(obs, pert_column), key=lambda p: (tier(p), p))
    return bio_values, pert_rows


def check_axis_coverage(
    train_obs: pd.DataFrame,
    column: str,
    universe: Sequence[str],
    holdout_values: Optional[set[str]] = None,
) -> dict:
    """Check every value in ``universe`` appears marginally in ``train_obs[column]``.

    ``holdout_values`` is ignored (kept for call-site compatibility). Axis coverage
    always uses the full universe; holdout predicates do not shrink the expected set.
    """
    del holdout_values
    expected = [str(v) for v in universe]
    if column not in train_obs.columns:
        raise ValueError(f"Column {column!r} not in train_obs.")
    present = set(_str_col(train_obs, column).unique())
    missing = tuple(v for v in expected if v not in present)
    return {
        "column": column,
        "n_expected": len(expected),
        "n_present": sum(v in present for v in expected),
        "missing": missing,
        "ok": not missing,
    }


def check_test_predicate_leakage(
    train_obs: pd.DataFrame,
    split_cfg: SplitConfig,
    *,
    key_columns: Sequence[str] = (),
) -> dict:
    """Warn if any train rows match the test predicate (split leakage)."""
    from .predicates import apply_split_predicate

    if train_obs.empty or not split_cfg.test:
        return {"n_leaked_rows": 0, "ok": True, "preview": ()}

    test_mask_on_train = apply_split_predicate(train_obs, split_cfg.test)
    n_leaked = int(test_mask_on_train.sum())
    preview: tuple[str, ...] = ()
    if n_leaked and key_columns:
        cols = [c for c in key_columns if c in train_obs.columns]
        if cols:
            preview = tuple(
                train_obs.loc[test_mask_on_train, cols]
                .drop_duplicates()
                .head(5)
                .astype(str)
                .agg("|".join, axis=1)
                .tolist()
            )
    return {"n_leaked_rows": n_leaked, "ok": n_leaked == 0, "preview": preview}


def format_axis_report(
    results: Sequence[dict],
    *,
    header: str = "Axis coverage (train marginal, full universe):",
) -> str:
    lines = [header]
    for res in results:
        lines.append(
            f"  [{'PASS' if res['ok'] else 'FAIL'}] {res['column']}: "
            f"{res['n_present']}/{res['n_expected']} universe values seen in train"
        )
        if res["missing"]:
            preview = ", ".join(res["missing"][:12])
            suffix = f" (+{len(res['missing']) - 12} more)" if len(res["missing"]) > 12 else ""
            lines.append(f"         missing from train: {preview}{suffix}")
    return "\n".join(lines)


def estimate_f_min(
    obs_pert_train: pd.DataFrame, bio_column: str, pert_column: str,
    *, marginalize_columns: Sequence[str] = (), control_pert_value: Optional[str] = None,
) -> dict:
    if obs_pert_train.empty:
        return dict(n_unique_tuples=0, n_bio=0, n_pert=0, n_target_min=0, f_min=float("inf"))
    bio, pert = _str_col(obs_pert_train, bio_column), _str_col(obs_pert_train, pert_column)
    if control_pert_value is not None:
        keep = pert != str(control_pert_value)
        bio, pert = bio[keep], pert[keep]
    n_bio, n_pert = int(bio.nunique()), int(pert.nunique())
    n_target_min = max(n_bio, n_pert)
    n_unique = int(pd.DataFrame({bio_column: bio, pert_column: pert}).drop_duplicates().shape[0])
    f_min = n_target_min / n_unique if n_unique else float("inf")
    return dict(n_unique_tuples=n_unique, n_bio=n_bio, n_pert=n_pert, n_target_min=n_target_min, f_min=f_min)


def _str_universe(obs: pd.DataFrame, column: str) -> list[str]:
    s = obs[column]
    if isinstance(s.dtype, pd.CategoricalDtype):
        return [str(c) for c in s.cat.categories]
    return sorted({str(v) for v in s.unique()})


def axis_diagnostics(
    full_obs: pd.DataFrame,
    train_obs: pd.DataFrame,
    bio_col: str,
    pert_col: str,
    split_cfg: SplitConfig,
    control_pert: Mapping[str, Any],
) -> list[dict]:
    bio_u = sorted(_str_universe(full_obs, bio_col))
    pert_u = [
        p for p in sorted(_str_universe(full_obs, pert_col))
        if p != str(control_pert.get(pert_col, ""))
    ]
    return [
        check_axis_coverage(train_obs, bio_col, bio_u),
        check_axis_coverage(train_obs, pert_col, pert_u),
    ]


def _sparse_index_ticks(n_rows: int) -> np.ndarray:
    """Return a small set of y-axis index ticks for large pert libraries."""
    if n_rows <= 1:
        return np.array([0], dtype=np.int64)
    candidates = (0, n_rows // 4, n_rows // 2, (3 * n_rows) // 4, n_rows - 1)
    return np.array(sorted(set(candidates)), dtype=np.int64)


def plot_coverage_heatmap_index(
    matrix: np.ndarray,
    bio_cols: Sequence[str],
    *,
    title: str,
    output: Path,
    use_counts: bool = False,
    n_perts_total: Optional[int] = None,
    skip_control_row: bool = True,
    column_marginal: Optional[np.ndarray] = None,
    marginal_ylabel: Optional[str] = None,
) -> None:
    """Index-based heatmap for full perturbation libraries (y = pert index)."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n_perts, n_bios = matrix.shape
    n_label = n_perts_total if n_perts_total is not None else n_perts
    heat_h = max(4.0, min(0.004 * n_perts, 72.0))
    fig_w = max(6.0, min(0.35 * n_bios + 2.0, 14.0))
    fig = plt.figure(figsize=(fig_w, heat_h + 1.8))
    gs = GridSpec(2, 1, height_ratios=[1.0, heat_h], hspace=0.12)

    data_rows = matrix[1:] if skip_control_row and n_perts > 1 else matrix
    n_data = int(data_rows.shape[0])
    ax_marg = fig.add_subplot(gs[0, 0])
    # column_marginal overrides the pert-fraction bar. The overlap plots pass
    # train cell counts here so a 10% cell cap is visible; the heatmap stays
    # binary presence unless use_counts is also set.
    if column_marginal is not None:
        marg_vals = np.asarray(column_marginal, dtype=np.float64)
        if marg_vals.shape != (n_bios,):
            raise ValueError(
                f"column_marginal length {marg_vals.shape} != n_bios {n_bios}"
            )
        marg_ylabel = marginal_ylabel or "Train cells"
        marg_ylim_top = None
    elif use_counts:
        marg_vals = data_rows.sum(axis=0)
        marg_ylabel = "Train row count"
        marg_ylim_top = None
    else:
        marg_vals = data_rows.sum(axis=0) / n_data if n_data else np.zeros(n_bios)
        marg_ylabel = "Train pert fraction"
        marg_ylim_top = 1.05
    ax_marg.bar(np.arange(n_bios), marg_vals, color="steelblue", width=0.8)
    if column_marginal is not None:
        from matplotlib.ticker import FuncFormatter
        ax_marg.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{int(v):,}"))
    ax_marg.set_xticks(np.arange(n_bios))
    ax_marg.set_xticklabels(bio_cols, rotation=45, ha="right", fontsize=8)
    ax_marg.set_ylabel(marg_ylabel, fontsize=8)
    ax_marg.set_ylim(0, marg_ylim_top)
    ax_marg.tick_params(labelsize=7)

    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(
        matrix, aspect="auto", interpolation="nearest",
        cmap="YlOrRd" if use_counts else "Blues", vmin=0,
    )
    ax.set_xticks(np.arange(n_bios))
    ax.set_xticklabels(bio_cols, rotation=90, fontsize=8)
    ax.set_xlabel("Biological context (test holdouts at right)")
    y_ticks = _sparse_index_ticks(n_perts)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(int(t)) for t in y_ticks], fontsize=7)
    ax.set_ylabel(f"Perturbation index (0…{n_perts - 1})")
    fig.suptitle(
        f"{title}\n{n_label:,} perturbations (test holdouts at top); y-axis = index, not gene name",
        fontsize=9,
    )
    fig.colorbar(
        im, ax=ax, fraction=0.02, pad=0.02,
        label="Row count" if use_counts else "Present in train",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote split coverage index heatmap → %s (%dx%d)", output, n_perts, n_bios)


def plot_coverage_heatmap(
    matrix: np.ndarray,
    pert_rows: Sequence[str],
    bio_cols: Sequence[str],
    *,
    title: str,
    output: Path,
    use_counts: bool = False,
) -> None:
    """Gene-labeled heatmap for small/medium perturbation subsets."""
    import matplotlib.pyplot as plt

    n_perts, n_bios = matrix.shape
    fig, ax = plt.subplots(figsize=(max(8, min(0.35 * n_bios, 48)), max(6, min(0.22 * n_perts, 72))))
    im = ax.imshow(
        matrix, aspect="auto", interpolation="nearest",
        cmap="YlOrRd" if use_counts else "Blues", vmin=0,
    )
    ax.set_xticks(np.arange(n_bios))
    ax.set_yticks(np.arange(n_perts))
    ax.set_xticklabels(bio_cols, rotation=90, fontsize=7)
    ax.set_yticklabels(pert_rows, fontsize=7)
    ax.set_xlabel("Biological context (test holdouts at right)")
    ax.set_ylabel("Perturbation (gene name)")
    ax.set_title(f"{title}\n{len(pert_rows):,} perturbations shown (gene labels)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Row count" if use_counts else "Present in train")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote split coverage gene-label heatmap → %s", output)


def _f_min_sidecar_payload(
    f_min: dict,
    pre_axis: Sequence[dict],
    *,
    cfg_frac: Optional[float],
    pre_axis_ok: bool,
) -> dict:
    return {
        "n_unique_tuples": f_min["n_unique_tuples"],
        "n_bio": f_min["n_bio"],
        "n_pert": f_min["n_pert"],
        "n_target_min": f_min["n_target_min"],
        "f_min": None if f_min["f_min"] == float("inf") else f_min["f_min"],
        "configured_fraction": cfg_frac,
        "hbss_fraction_ok": (
            None if cfg_frac is None
            else cfg_frac >= f_min["f_min"] and pre_axis_ok and f_min["n_unique_tuples"] > 0
        ),
        "pre_hbss_axis_ok": pre_axis_ok,
        "pre_hbss_axis_coverage": [
            {k: list(r[k]) if k == "missing" else r[k] for k in ("column", "ok", "n_expected", "n_present", "missing")}
            for r in pre_axis
        ],
    }


def write_split_coverage_artifacts(
    workdir: Union[str, Path],
    *,
    full_obs: pd.DataFrame,
    obs_eval: pd.DataFrame,
    train_idx: np.ndarray,
    pert_train_idx_pre_hbss: np.ndarray,
    split_cfg: SplitConfig,
    bio_columns: Sequence[str],
    pert_columns: Sequence[str],
    control_pert: Mapping[str, Any],
    dataset_name: str,
    split_config_path: str,
    use_counts: bool = False,
    max_pert_rows: Optional[int] = None,
    png_path: Optional[Path] = None,
    json_path: Optional[Path] = None,
    raise_on_axis_failure: bool = True,
    column_marginal: Optional[np.ndarray] = None,
) -> tuple[Path, Path]:
    """Write split-coverage heatmap + f_min JSON (default: ``<workdir>/eval/``).

    The main ``split_coverage.png`` is an index-based heatmap over the full pert
    library. Test-predicate bios are rightmost and test perts sit just below the
    control row (top). When ``max_pert_rows`` is set, also writes
    ``split_coverage_gene_labels.png`` with gene names capped to that many rows.
    """
    bio_col, pert_col, marg_cols = plot_axes_from_registry(bio_columns, pert_columns)
    png_path = png_path or (Path(workdir) / "eval" / "split_coverage.png")
    json_path = json_path or png_path.with_name(f"{png_path.stem}_f_min.json")

    # Collapse to unique (bio, pert) combinations before the pre-HBSS diagnostics.
    # Both consumers are multiplicity-invariant: estimate_f_min uses only nunique()
    # and drop_duplicates(), and check_axis_coverage only set(...unique()). Neither
    # can see how many cells back a combination, so this is exact, not an
    # approximation. It matters because _str_col does .astype(str) on every row --
    # on Tahoe that is 92.4M rows collapsing to ~53k unique triplets (~1,700x),
    # which is the difference between tens of minutes and under a second.
    pert_train_pre = (
        obs_eval.loc[:, [bio_col, pert_col]]
        .iloc[pert_train_idx_pre_hbss]
        .drop_duplicates()
    )
    ctrl_val = control_pert.get(pert_col)
    f_min = estimate_f_min(
        pert_train_pre, bio_col, pert_col, marginalize_columns=marg_cols,
        control_pert_value=str(ctrl_val) if ctrl_val is not None else None,
    )
    cfg_frac = (
        float(split_cfg.train.fraction)
        if split_cfg.train.sample == "hbss" and split_cfg.train.fraction is not None
        else None
    )
    pre_axis = axis_diagnostics(full_obs, pert_train_pre, bio_col, pert_col, split_cfg, control_pert)
    pre_axis_ok = all(r["ok"] for r in pre_axis)
    print(format_f_min_report(f_min, configured_fraction=cfg_frac, pre_hbss_axis_ok=pre_axis_ok))
    print(format_axis_report(pre_axis, header="Axis coverage (perturbed train, pre-HBSS, full universe):"))
    if cfg_frac is not None and (cfg_frac < f_min["f_min"] or not pre_axis_ok):
        logger.warning(
            "HBSS axis coverage may fail: fraction=%.6f f_min=%.6f pre_hbss_axis_ok=%s",
            cfg_frac, f_min["f_min"], pre_axis_ok,
        )

    train_obs = obs_eval.iloc[train_idx].copy()
    leakage = check_test_predicate_leakage(
        train_obs,
        split_cfg,
        key_columns=(bio_col, pert_col),
    )
    if not leakage["ok"]:
        logger.warning(
            "Test-predicate leakage: %d train rows match test predicate%s",
            leakage["n_leaked_rows"],
            f" (e.g. {leakage['preview']})" if leakage["preview"] else "",
        )
    bio_values, pert_rows = coverage_plot_axes(
        full_obs, train_obs, bio_col, pert_col, split_cfg, control_pert,
    )
    n_perts_total = len(pert_rows)

    matrix = build_presence_matrix(
        train_obs, bio_values, pert_rows, bio_col, pert_col,
        marginalize_columns=marg_cols, use_counts=use_counts,
    )
    n_combos, n_possible = int((matrix > 0).sum()), n_perts_total * len(bio_values)
    logger.info(
        "Train combination coverage: %d/%d cells filled (%.2f%%) over %d perts × %d bios",
        n_combos, n_possible, 100.0 * n_combos / n_possible if n_possible else 0,
        n_perts_total, len(bio_values),
    )

    split_name = Path(split_config_path).name
    if column_marginal is not None:
        n_cells = int(np.asarray(column_marginal).sum())
        title = f"TRAIN coverage — {dataset_name} / {split_name} ({n_cells:,} train cells)"
    else:
        title = (
            f"TRAIN coverage — {dataset_name} / {split_name} "
            f"({len(train_obs):,} rows"
            + (f", HBSS f={split_cfg.train.fraction}" if split_cfg.train.sample == "hbss" else "")
            + ")"
        )
    plot_coverage_heatmap_index(
        matrix, bio_values, title=title, output=png_path,
        use_counts=use_counts, n_perts_total=n_perts_total,
        column_marginal=column_marginal, marginal_ylabel="Train cells",
    )

    if max_pert_rows is not None and n_perts_total > max_pert_rows + 1:
        capped_rows = pert_rows[:1] + pert_rows[1 : max_pert_rows + 1]
        logger.info(
            "Writing gene-label detail plot: control + %d/%d perts",
            max_pert_rows, n_perts_total - 1,
        )
        capped_matrix = build_presence_matrix(
            train_obs, bio_values, capped_rows, bio_col, pert_col,
            marginalize_columns=marg_cols, use_counts=use_counts,
        )
        gene_labels_path = png_path.with_name(f"{png_path.stem}_gene_labels{png_path.suffix}")
        plot_coverage_heatmap(
            capped_matrix, capped_rows, bio_values, title=title,
            output=gene_labels_path, use_counts=use_counts,
        )

    axis_results = axis_diagnostics(full_obs, train_obs, bio_col, pert_col, split_cfg, control_pert)
    print(format_axis_report(axis_results))
    axis_ok = all(r["ok"] for r in axis_results)
    if not axis_ok:
        missing_summary = "; ".join(
            f"{r['column']} missing {len(r['missing'])}" for r in axis_results if not r["ok"]
        )
        msg = f"Marginal axis coverage check FAILED — {missing_summary}"
        logger.error(msg)
        if raise_on_axis_failure:
            raise ValueError(msg)
    if not leakage["ok"]:
        msg = f"Holdout leakage check FAILED — {leakage['n_leaked_rows']} train rows match test predicate"
        logger.error(msg)
        if raise_on_axis_failure:
            raise ValueError(msg)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _f_min_sidecar_payload(f_min, pre_axis, cfg_frac=cfg_frac, pre_axis_ok=pre_axis_ok)
    payload["holdout_leakage_ok"] = leakage["ok"]
    payload["holdout_leakage"] = {"n_leaked_rows": leakage["n_leaked_rows"]}
    json_path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote f_min sidecar → %s", json_path)
    return png_path, json_path


def format_f_min_report(est: dict, *, configured_fraction: Optional[float] = None, pre_hbss_axis_ok: bool = True) -> str:
    lines = [
        "Axis-coverage minimum fraction (pre-HBSS perturbed train):",
        f"  N={est['n_unique_tuples']:,} unique (bio, pert) tuples; n_bio={est['n_bio']}, n_pert={est['n_pert']}; n_target_min=max(n_bio,n_pert)={est['n_target_min']}",
    ]
    lines.append("  f_min=inf (empty perturbed train pool)" if est["n_unique_tuples"] == 0 else f"  f_min={est['f_min']:.6f} ({100.0 * est['f_min']:.4f}%)")
    if not pre_hbss_axis_ok:
        lines.append(
            "  WARNING: perturbed train is missing universe bios/perts marginally — "
            "HBSS subsampling cannot restore full-axis coverage."
        )
    if configured_fraction is not None:
        ok = configured_fraction >= est["f_min"] and pre_hbss_axis_ok and est["n_unique_tuples"] > 0
        lines.append(f"  HBSS configured fraction={configured_fraction:.6f} vs f_min: [{'PASS' if ok else 'FAIL'}]")
        if configured_fraction < est["f_min"] and est["n_unique_tuples"] > 0:
            lines.append("  WARNING: configured fraction < f_min — axis coverage is mathematically impossible under random/tuple subsampling.")
    return "\n".join(lines)


# The coverage CLI's inputs

def train_cell_marginal(
    train_obs: pd.DataFrame,
    bio_values: Sequence[str],
    bio_col: str,
    *,
    target_n: Optional[int],
    seed: int,
) -> np.ndarray:
    """Train cells per biological context, in ``bio_values`` order.

    A condition summary has one row per condition and a ``num_cells`` weight.
    The bar is then one draw of ``target_n`` from those weights, which is the
    aggregate of the trainer's uniform cell subsample. A cell-level table
    (``num_cells`` absent, or summing to the row count) is counted as-is,
    because ``compute_train_split`` already applied the budget to those rows.
    """
    from src.data.sampling.cell_subsampling import allocate_cell_budget

    bios = train_obs[bio_col].astype(str)
    if "num_cells" in train_obs.columns:
        grouped = train_obs.groupby(bios, sort=False)["num_cells"].sum()
        weights = np.array([int(grouped.get(b, 0)) for b in bio_values], dtype=np.int64)
        if int(weights.sum()) == len(train_obs):
            return weights
        if target_n is None:
            return weights
        return allocate_cell_budget(weights, int(target_n), int(seed))
    counts = bios.value_counts()
    return np.array([int(counts.get(b, 0)) for b in bio_values], dtype=np.int64)


def load_obs_table(registry, *, obs_path=None, data_dir=None) -> pd.DataFrame:
    """obs of a whole dataset: a summary/obs file (--obs-path), or every h5ad."""
    from ..vocab import concentrations_from_obs as _concentrations_from_obs
    from ..data_utils import _ensure_obs_columns_categorical

    if obs_path is not None:
        path = obs_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"--obs-path not found: {path}")
        logger.info("Loading obs from %s", path)
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".pkl":
            return pd.read_pickle(path)
        raise ValueError(f"Unsupported --obs-path suffix {path.suffix!r}; use .parquet or .pkl")
    root_dir = Path(data_dir or registry.data_dir).expanduser().resolve()
    import anndata as ad
    pattern = registry.plate_glob or "*.h5ad"
    files = sorted(root_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {pattern!r} under {root_dir}")
    cond_cols = list(dict.fromkeys(c for c in [
        *(registry.biological_context_columns), *(registry.perturbation_columns),
        *([registry.plate_column] if registry.plate_column else []),
    ] if c))
    parts = []
    for p in files:
        logger.info("Reading obs from %s", p.name)
        sub = ad.read_h5ad(p, backed="r").obs.copy()
        if "drugname_drugconc" in sub.columns and "dose" not in sub.columns:
            sub["dose"] = _concentrations_from_obs(sub, key="drugname_drugconc")
        valid = [c for c in cond_cols if c in sub.columns]
        if "dose" in sub.columns and "dose" not in valid:
            valid = [*valid, "dose"]
        parts.append(_ensure_obs_columns_categorical(sub[valid].copy(), valid))
    obs = pd.concat(parts, axis=0, ignore_index=True)
    logger.info("Concatenated obs: n_obs=%s cols=%d", f"{len(obs):,}", len(obs.columns))
    return obs


# Overlap membership: which held-out perturbation trains on which bio, per run

TRAIN, VAL, TEST = 0, 1, 2
MEMBERSHIP_COLORS = ["#0072B2", "#E69F00", "#009E73"]   # same hues as the seed lines


def _manifest_field(entry: Mapping[str, Any], prefix: str) -> Any:
    """The one manifest key starting with `prefix_` (pool_cells, pool_donors, pool_bios, ...)."""
    hits = [k for k in entry if k.startswith(prefix + "_")]
    if len(hits) != 1:
        raise KeyError(f"manifest entry needs exactly one {prefix}_* key; found {hits}")
    return entry[hits[0]]


def overlap_membership(entry: Mapping[str, Any]) -> tuple[list[str], list[str], np.ndarray, list[int]]:
    """Rows (pool then test bios), perturbations sorted by k, and a (bio, pert) train/val/test grid."""
    pool, test = list(_manifest_field(entry, "pool")), list(_manifest_field(entry, "test"))
    peeked: Mapping[str, Sequence[str]] = _manifest_field(entry, "peeked")
    rows = pool + test
    perts = sorted(peeked, key=lambda p: (len(peeked[p]), p))
    ks = [len(peeked[p]) for p in perts]
    pool_set, test_set = set(pool), set(test)
    grid = np.empty((len(rows), len(perts)), dtype=np.int8)
    for j, pert in enumerate(perts):
        train = set(peeked[pert])
        if not train <= pool_set:
            raise ValueError(f"{pert} trains outside the pool: {sorted(train - pool_set)[:3]}")
        for i, bio in enumerate(rows):
            grid[i, j] = TEST if bio in test_set else TRAIN if bio in train else VAL
    return rows, perts, grid, ks


def plot_overlap_membership(rows: Sequence[str], perts: Sequence[str], grid: np.ndarray, ks: Sequence[int],
                            title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    n_pool = next(i for i, row in enumerate(grid) if (row == TEST).all())
    fig, ax = plt.subplots(figsize=(max(14.0, 0.32 * len(perts) + 3.2), max(8.0, 0.22 * len(rows) + 1.6)),
                           layout="constrained")
    ax.imshow(grid, cmap=ListedColormap(MEMBERSHIP_COLORS), vmin=0, vmax=2, interpolation="nearest", aspect="equal")
    ax.set_xticks(range(len(perts)), perts, rotation=90, fontsize=7, ha="center")
    ax.set_yticks(range(len(rows)), rows, fontsize=7)
    ax.axhline(n_pool - 0.5, color="white", lw=1.5)
    for j, k in enumerate(ks):
        if j and k != ks[j - 1]:
            ax.axvline(j - 0.5, color="white", lw=0.8)
    levels = "  |  ".join(str(k) for k in sorted(set(ks)))
    ax.set_xlabel(f"Held-out perturbation, grouped by training contexts   k = {levels}")
    ax.set_ylabel(f"Context   (top {n_pool}: pool,  bottom {len(rows) - n_pool}: always test)")
    ax.set_title(title, loc="left", fontsize=11)
    ax.legend(handles=[Patch(facecolor=MEMBERSHIP_COLORS[c], label=l)
                       for c, l in ((TRAIN, "train"), (VAL, "validation"), (TEST, "test"))],
              loc="upper right", bbox_to_anchor=(1.0, 1.12), ncol=3, frameon=False, fontsize=8)
    ax.tick_params(length=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
