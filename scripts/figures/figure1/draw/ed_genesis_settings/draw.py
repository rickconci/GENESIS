"""Extended Data: sensitivity to training settings. Three rows of the same size, Fig. 2c's report space.

    a  generated cells per observed cell, m in {16, 64, 256}
    b  variogram weight, lambda in {0, 0.1, 0.3, 0.5, 1}
    c  network depth, 0, 1, or 2 hidden layers; depth 2 is the Fig. 2 model itself

Each is the main Tahoe-100M GENESIS (m = 256, lambda = 0.5, depth 2; Fig. 2) retrained with one setting changed.
Each point is the mean over held-out conditions of the run's own post-fit test evaluation. The error bars are
Fig. 2's 95% cluster-bootstrap intervals (src/stats/bootstrap.py: cell line x drug), recomputed here and
checked against the published means. The three rows share one model, which must score the same in each, or
nothing is drawn.

    python -m scripts.figures.figure1.draw.ed_genesis_settings.draw
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.ticker as ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.figures import CORE  # noqa: E402
from src.figures import ed_page, metrics, page  # noqa: E402
from src.figures.bars import metric_title, whisker  # noqa: E402
from src.figures.palette import FILL, GRID  # noqa: E402
from src.stats.bootstrap import cluster_bootstrap_ci, condition_keys  # noqa: E402

H = 150.0
ROW = 50.0                                         # row()'s height, banner to x label
# Fig. 2c's names: energy is logged there, and here, so the title can say (log)
METRICS = [(metrics.DEG, "Top-100 DEG overlap", True), (metrics.PEARSON, "Pearson Δ", True),
           (metrics.ENERGY, "Energy distance (log)", False)]
MAIN = {"n_repeats": 256, "variogram_weight": 0.5, "depth": 2}   # filled: the Fig. 2 model (two hidden layers)
COL_X = [12.0 + 58.0 * i for i in range(3)]


def with_ci(published: pd.DataFrame, per: pd.DataFrame, setting: str) -> pd.DataFrame:
    """Fig. 2's interval on the plotted mean: one cluster draw per metric, shared by every setting."""
    rows = []
    for key, _title, _hb in METRICS:
        g = published[published.metric_key.eq(key)].sort_values(setting)
        wide = per[per.metric_key.eq(key)].pivot(index="condition", columns=setting, values="value")
        wide = wide.reindex(columns=list(g[setting])).dropna()
        if len(wide) != int(g.n_conditions.iloc[0]) or wide.isna().any().any():
            raise RuntimeError(f"{setting} / {key}: per-condition table is not the plotted set")
        cl = condition_keys(pd.Series(wide.index)).loc[:, ["cell_name", "drug"]].astype(str).agg("|".join, axis=1)
        mean, lo, hi = cluster_bootstrap_ci(wide.to_numpy(float), cl.to_numpy())
        if not np.allclose(mean, g["mean"], rtol=1e-5, atol=0):
            raise RuntimeError(f"{setting} / {key}: bootstrap mean is not the plotted mean")
        rows += [dict(metric_key=key, **{setting: v}, mean=m, lo=a, hi=z, n_conditions=len(wide))
                 for v, m, a, z in zip(g[setting], mean, lo, hi)]
    return pd.DataFrame(rows)


def load(data: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The three settings' means and CIs, after checking that the shared model agrees across all three."""
    m = pd.read_csv(data / "ed_energy_settings/num_repeats_scaling_source_data_plotted.csv")
    lam = pd.read_csv(data / "ed_energy_settings/variogram_scaling_source_data_plotted.csv")
    per = pd.read_parquet(data / "ed_energy_settings/per_condition.parquet")
    got = per.groupby(["metric_key", "n_repeats"])["value"].agg(["mean", "count"])
    want = m.set_index(["metric_key", "n_repeats"])
    if not (np.allclose(got.loc[want.index, "mean"], want["mean"], rtol=1e-5) and (got.loc[want.index, "count"] == want.n_conditions).all()):
        raise RuntimeError("num_repeats_scaling: per-condition means differ from its published table")
    shared = m[m.n_repeats.eq(256)].set_index("metric_key")["mean"]
    if not np.allclose(shared, lam[lam.variogram_weight.eq(0.5)].set_index("metric_key").loc[shared.index, "mean"], rtol=1e-6):
        raise RuntimeError("the shared model (m = 256, lambda = 0.5) differs between the two settings")
    m = with_ci(m, per, "n_repeats")
    lam = with_ci(lam, pd.read_parquet(data / "variogram_scaling/per_condition.parquet"), "variogram_weight")
    depth_per = pd.read_parquet(data / "ed_settings/depth_per_condition.parquet")
    depth = depth_per.groupby(["metric_key", "depth"]).value.agg(mean="mean", n_conditions="count").reset_index()
    depth = with_ci(depth, depth_per, "depth")
    got = depth[depth.depth.eq(2)].set_index("metric_key")["mean"]
    if not np.allclose(got, shared.loc[got.index], rtol=1e-6, atol=0):
        raise RuntimeError("depth 2 is not the shared model of a and b (m = 256, lambda = 0.5)")
    return m, lam, depth


def row(fig, t: pd.DataFrame, setting: str, y0: float, letter: str, title: str, xlabel: str, fmt) -> None:
    page.letter(fig, letter, 1, y0)
    ed_page.banner_line(fig, y0, "Tahoe-100M", title)   # Fig. 2c: dataset · what this row varies
    blue = FILL["GENESIS"]
    for (key, title, hb), x0 in zip(METRICS, COL_X):
        q = t[t.metric_key.eq(key)].sort_values(setting)
        if q.n_conditions.nunique() != 1:
            raise RuntimeError(f"{setting} / {key}: settings scored on different condition sets")
        xs, mean = np.arange(len(q)), q["mean"].to_numpy()
        lo, hi = q["lo"].to_numpy(), q["hi"].to_numpy()
        ax = page.axes_mm(fig, x0, y0 + 11.0, 42.0, 24.0)
        ax.plot(xs, mean, "-", color=blue, lw=0.8, zorder=3)
        whisker(ax, xs, mean, lo, hi, color=blue, zorder=4)
        main = q[setting].to_numpy() == MAIN[setting]
        ax.plot(xs[~main], mean[~main], "o", ms=2.6, mfc="white", mec=blue, mew=0.8, zorder=5)
        ax.plot(xs[main], mean[main], "o", ms=2.6, color=blue, zorder=6)
        if hb:
            pad = 0.15 * (hi.max() - lo.min())
            ax.set_ylim(lo.min() - pad, hi.max() + pad)
        else:                                          # Fig. 2c logs energy; a CI through 0 cannot
            if lo.min() <= 0:
                raise RuntimeError(f"{setting} / {key}: energy CI crosses 0, so a log axis cannot show it")
            ax.set_yscale("log")
            ax.set_ylim(lo.min() / 1.25, hi.max() * 1.25)
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _p: f"{v:g}"))
        ax.set_xlim(-0.4, len(q) - 0.6)
        ax.set_xticks(xs, [fmt(v) for v in q[setting]])
        ax.set_xlabel(xlabel)
        metric_title(ax, title, hb, pad=2)
        ax.grid(axis="y", color=GRID, lw=0.4)


def main(data: Path = CORE / "data" / "figure1", out: Path = CORE / "figure1") -> None:
    m, lam, depth = load(data / "ED_genesis_settings")
    ed_page.main_figure_style(page.style)
    fig = ed_page.figure(H)
    row(fig, m, "n_repeats", 0 * ROW, "a", "Generated cells per observed cell",
        "Generated cells per observed cell, m", lambda v: f"{int(v)}")
    row(fig, lam, "variogram_weight", 1 * ROW, "b", "Variogram weight", "Variogram weight, λ",
        lambda v: f"{v:g}")
    row(fig, depth, "depth", 2 * ROW, "c", "Hidden layers", "Hidden layers", lambda v: f"{int(v)}")
    ed_page.save(fig, out / "ed" / "ED_genesis_settings")


if __name__ == "__main__":
    main()
