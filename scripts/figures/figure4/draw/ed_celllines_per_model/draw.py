"""Extended Data: one depth-adjusted PROGENy heatmap per held-out Tahoe line, stacked TRUE, GENESIS, STATE,
ADDITIVE, NN.

GENESIS and State are the Figure 4 depth module values. Additive is the baseline compare_baselines scored and
Figure 4 does not draw; its cached scores are refused unless their per-pathway mean |error| matches that
script's published depth sidecar. NN is Fig. 2a's nearest neighbor (nn_own): the neighbor line's measured 5 µM
cells on the drug's own plates, against the neighbor's own plate controls. Both caches are written by the
extract step (hours from the cell sums); a missing one stops the build.

    python -m scripts.figures.figure4.draw.ed_celllines_per_model.draw
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.figures import CORE  # noqa: E402
from scripts.figures.figure4.draw import data as D  # noqa: E402
from scripts.figures.figure4.draw import panels as F  # noqa: E402
from src.figures import page  # noqa: E402

ORDER = ("truth", "genesis", "state", "additive", "nn")       # the page's blocks, top to bottom
LABEL = {"truth": "TRUE", "genesis": "GENESIS", "state": "STATE", "additive": "ADDITIVE", "nn": "NN"}


# ----------------------------------------------------------------------------
# 1. Read the two baselines from their caches; Additive is checked against the published sidecar
# ----------------------------------------------------------------------------

def mae_like_sidecar(pred: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Per (pathway, line, arm): how many drugs were scored, and the mean |error| over those drugs.
    A drug the arm did not score is absent, not a zero. A scored drug with no truth, or a NaN score,
    would change that mean without a trace, so both raise."""
    if pred.v.isna().any():
        raise RuntimeError(f"{int(pred.v.isna().sum())} scored pathway values are NaN")
    ref = truth.loc[truth.arm.eq("truth"), ["programme", "cell_name", "drug", "v"]].rename(columns={"v": "truth"})
    m = pred.merge(ref, on=["programme", "cell_name", "drug"], how="left")
    if m.truth.isna().any():
        raise RuntimeError(f"{int(m.truth.isna().sum())} scored rows have no measured pathway value")
    return (m.assign(mae=(m.v - m.truth).abs())
            .groupby(["programme", "cell_name", "arm"], sort=False)
            .agg(n=("mae", "size"), mae=("mae", "mean")).reset_index())


def assert_sidecar(pred: pd.DataFrame, truth: pd.DataFrame, published: pd.DataFrame, atol: float = 1e-6) -> None:
    """Refuse unless ``pred`` reproduces ``published`` (the depth MAE sidecar) row for row."""
    got = mae_like_sidecar(pred, truth).set_index(["programme", "cell_name", "arm"])
    pub = published.set_index(["programme", "cell_name", "arm"])
    if not got.index.symmetric_difference(pub.index).empty:
        raise RuntimeError(f"MAE rows differ from the sidecar: "
                           f"missing {list(pub.index.difference(got.index)[:3])}, "
                           f"extra {list(got.index.difference(pub.index)[:3])}")
    got = got.reindex(pub.index)
    if not np.array_equal(got.n.to_numpy(), pub.n.to_numpy()):
        raise RuntimeError("a pathway was scored on a different number of drugs than the sidecar")
    if not np.allclose(got.mae.to_numpy(), pub.mae.to_numpy(), rtol=0, atol=atol):
        i = int(np.abs(got.mae.to_numpy() - pub.mae.to_numpy()).argmax())
        raise RuntimeError(f"MAE differs from the sidecar at {pub.index[i]}: "
                           f"{got.mae.iloc[i]:.6g} vs {pub.mae.iloc[i]:.6g}")


def baselines(d: Path) -> pd.DataFrame:
    """Additive and NN, depth-adjusted mixture scores. The Additive cache also holds compare_baselines'
    copy_closest_15: both are checked against the sidecar, then only Additive is kept."""
    cb = d / "compare_baselines"
    for f in ("depth_module_values.parquet", "depth_module_values__nn_own.parquet"):
        if not (cb / f).exists():
            raise FileNotFoundError(f"{cb / f} missing: the extract step scores it from the cached cell sums")
    pred = pd.read_parquet(cb / "depth_module_values.parquet")
    truth = pd.read_parquet(d / "tahoe_progeny_depth/module_values.parquet")
    assert_sidecar(pred, truth, pd.read_csv(cb / "summary_metrics__depth.tsv", sep="\t"))
    print("  ✓ Additive's MAE matches summary_metrics__depth.tsv", flush=True)
    nn = pd.read_parquet(cb / "depth_module_values__nn_own.parquet")
    return pd.concat([pred[pred.arm == "additive"], nn], ignore_index=True)


# ----------------------------------------------------------------------------
# 2. Draw the stack: five blocks, Figure 4's heatmap, one line a figure
# ----------------------------------------------------------------------------

def dataset(d: Path) -> dict:
    """The depth split, axis drugs, with Additive and NN appended. An arm outside those two would silently
    replace a published row, so it raises."""
    base = baselines(d)
    S = D.dataset(d / "tahoe_progeny_depth", D.axis_table(d / "drug_axis/drug_axis.tsv"), ["genesis", "state"])
    if set(base.arm) != {"additive", "nn_own"}:
        raise RuntimeError(f"baselines must be additive and nn_own, got {sorted(set(base.arm))}")
    extra = base.loc[base.drug.isin(S["cond_order"]), ["programme", "cell_name", "drug", "arm", "v"]].assign(set="grid")
    mv = pd.concat([S["mv"], extra.assign(arm=extra.arm.replace({"nn_own": "nn"}))], ignore_index=True)
    return {**S, "mv": mv, "arms": ["genesis", "state", "additive", "nn"]}


def blocks_for(mv: pd.DataFrame, rows: list[tuple[str, str]], cols: list[str]) -> list:
    """[(block name, [(row label, values over cols)])] in ORDER. A drug an arm did not score is NaN
    (the heatmap's grey), never 0: 0 is a real score."""
    if missing := [a for a in ORDER if a not in set(mv.arm)]:
        raise RuntimeError(f"missing {missing}")
    key = ["programme", "drug", "arm"]
    if mv.duplicated(key).any():
        raise RuntimeError(f"duplicate programme/drug/arm:\n{mv.loc[mv.duplicated(key), key].head(3)}")
    v = mv.set_index(key).v
    return [(LABEL[arm], [(lab, [v[(r, c, arm)] if (r, c, arm) in v.index else np.nan for c in cols])
                          for lab, r in rows]) for arm in ORDER]


def main(data: Path = CORE / "data" / "figure4", out: Path = CORE / "figure4") -> None:
    F.style()
    S = dataset(data / "ED_celllines_per_model")
    t = S["mv"][S["mv"].set == "grid"]
    cols, col_groups = F.column_order(S, t)
    names, groups = F.pathway_rows(S["progs"])
    for c in S["held"]:
        rows = [(f"{p}{'*' if S['cov'].get((p, c), 0) < 3 else ''}", p) for p in names]   # * = < 3 members detected
        fig = F.heatmap(blocks_for(t[t.cell_name == c], rows, cols), cols,
                        title=f"{S['disp'][c]} (held out) · PROGENy",
                        left_in=0.36 + 0.035 * max(len(lab) for lab, _p in rows), row_in=0.08,
                        groups=groups, col_groups=col_groups)
        page.save(fig, out / "ed" / "ED_celllines_per_model" / c, dpi=600)
        plt.close(fig)


if __name__ == "__main__":
    main()
