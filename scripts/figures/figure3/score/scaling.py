"""Figure 3's data rules: GENESIS trained on less, scored on exactly Figure 2a's plate pairs and tested
against the experimental repeat with Figure 2a's tests; each point's mean over its 3 runs; panel b's
per-line overlap table; and the training runs behind every point. Every loader takes the bundle's data/
directory. Shared by Figure 3, its variants, the shallow-denoising supporting figures (their Figure 3a
benchmark) and Extended Data (the run lists).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.figures.figure3.score import repeat_pairs
from scripts.figures.figure3.score.repeat_pairs import KEYS
from src.stats.bootstrap import PRIMARY, UNITS, cluster_bootstrap_ci
from src.stats.equivalence import NO_GAP_TESTS, mark_no_gap, repeat_tests

PAIR_METRICS = [m for m, _t, _hb in repeat_pairs.PAIR_METRICS]          # Figure 2a's three, in its order
HIGHER_IS_BETTER = {m: hb for m, _t, hb in repeat_pairs.PAIR_METRICS}
REPEAT = "Experimental repeat"
BIO_LEVELS = (0, 15, 30, 80, 164)            # level 0 = control cells only, drawn at 1
CONTROL_DRUG = "DMSO_TF"                     # core_util_data/datasets/tahoe.yaml


# ----------------------------------------------------------------------------
# 1. The runs behind each point: the scaling arms and their 3 replicates (Figure 3's extract reads them)
# ----------------------------------------------------------------------------

# (axis, % of conditions or cells kept, stem, run dir under runs/tahoe). The 100% arm is Figure 2's GENESIS.
# Cells arm = the MATCHED runs: every training control kept, perturbed cells drawn to the conds arm's
# perturbed count. 5% = the 25-epoch run (~96k steps); 25% = 20 epochs.
SCALING = [
    ("conds", 100, "genesis", "bio/main/h4096x8192_nrs256_vg05rs256_5ep_state2k_tahoe_bio_100cond_seed0/260916_114938_cur-between_epoch-tp458752-bs1792xnrm256_nre256_lr0.0009_wd0.0001_biohold_condmlp-D256-h4096x8192_cond-plate_mlpout-relu_vg0.5p0.5np6k_log1p_s64_h4096x8192_nrs256_vg05rs256_5ep_state2k_tahoe_bio_100cond_seed0"),
    ("conds", 25, "genesis_25conds", "bio/scaling/n_conds_025/h4096x8192_nrs256_vg05rs256_20ep_state2k_tahoe_bio_25cond_seed0/260918_150610_cur-between_epoch-tp458752-bs1792xnrm256_nre256_lr0.0009_wd0.0001_tahoe_scalin_condmlp-D256-h4096x8192_cond-plate_mlpout-relu_vg0.5p0.5np6k_log1p_hbssf025_s64_h4096x8192_nrs256_vg05rs256_20ep_state2k_tahoe_bio_25cond_seed0"),
    ("conds", 5, "genesis_5conds", "bio/scaling/n_conds_005/h4096x8192_nrs256_vg05rs256_100ep_state2k_tahoe_bio_5cond_seed0/260920_164413_cur-between_epoch-tp458752-bs1792xnrm256_nre256_lr0.0009_wd0.0001_tahoe_scalin_condmlp-D256-h4096x8192_cond-plate_mlpout-relu_vg0.5p0.5np6k_log1p_hbssf005_s64_h4096x8192_nrs256_vg05rs256_100ep_state2k_tahoe_bio_5cond_seed0"),
    ("cells", 25, "genesis_25cells_allctrl", "bio/scaling/n_cells_025_keepctrl/h4096x8192_nrs256_vg05rs256_20ep_state2k_tahoe_bio_25cell_keepctrl_seed0/261009_003648_cur-between_epoch-tp458752-bs1792xnrm256_nre256_lr0.0009_wd0.0001_tahoe_scalin_condmlp-D256-h4096x8192_cond-plate_mlpout-relu_vg0.5p0.5np6k_log1p_celln22129534_s64_h4096x8192_nrs256_vg05rs256_20ep_state2k_tahoe_bio_25cell_keepctrl_seed0"),
    ("cells", 5, "genesis_5cells_allctrl", "bio/scaling/n_cells_005_keepctrl/h4096x8192_nrs256_vg05rs256_25ep_state2k_tahoe_bio_5cell_keepctrl_seed0/261006_113942_cur-between_epoch-tp458752-bs1792xnrm256_nre256_lr0.0009_wd0.0001_tahoe_scalin_condmlp-D256-h4096x8192_cond-plate_mlpout-relu_vg0.5p0.5np6k_log1p_celln4579207_s64_h4096x8192_nrs256_vg05rs256_25ep_state2k_tahoe_bio_5cell_keepctrl_seed0"),
]
# The uniform-draw cells arm, a figure variant. Cells drawn uniformly over all training
# rows, controls included, so it keeps fewer controls and more perturbed cells than the conds arm at the same total.
UNIFORM_CELLS = [
    ("cells", 25, "genesis_25cells", "bio/scaling/n_cells_025/h4096x8192_nrs256_vg05rs256_20ep_state2k_tahoe_bio_25cell_seed0/260918_151050_cur-between_epoch-tp458752-bs1792xnrm256_nre256_lr0.0009_wd0.0001_tahoe_scalin_condmlp-D256-h4096x8192_cond-plate_mlpout-relu_vg0.5p0.5np6k_log1p_celln24921726_s64_h4096x8192_nrs256_vg05rs256_20ep_state2k_tahoe_bio_25cell_seed0"),
    ("cells", 5, "genesis_5cells", "bio/scaling/n_cells_005/h4096x8192_nrs256_vg05rs256_100ep_state2k_tahoe_bio_5cell_seed0/260920_162654_cur-between_epoch-tp458752-bs1792xnrm256_nre256_lr0.0009_wd0.0001_tahoe_scalin_condmlp-D256-h4096x8192_cond-plate_mlpout-relu_vg0.5p0.5np6k_log1p_celln6761692_s64_h4096x8192_nrs256_vg05rs256_100ep_state2k_tahoe_bio_5cell_seed0"),
    # 1%: cells-only, no conds twin; 87 ep (~44.5k steps), not the 5% arm's 100 ep (~377k)
    ("cells", 1, "genesis_1cells", "bio/scaling/n_cells_001/h4096x8192_nrs256_vg05rs256_87ep_state2k_tahoe_bio_cells001_seed0/261016_085931_cur-between_epoch-tp458752-bs1792xnrm256_nre256_lr0.0009_wd0.0001_tahoe_scalin_condmlp-D256-h4096x8192_cond-plate_mlpout-relu_vg0.5p0.5np6k_log1p_celln918254_s64_h4096x8192_nrs256_vg05rs256_87ep_state2k_tahoe_bio_cells001_seed0"),
]
CELLS_ARMS = {"matched": ("scaling_arms.csv", SCALING), "uniform": ("scaling_arms_uniform_cells.csv", SCALING[:3] + UNIFORM_CELLS)}
# Every run behind Figure 3a's variants, 3 per point (25% = 11 ep, 5% = 25 ep).
# Replicate r draws its subsample with seed r and trains with seed 64 + r; the 100% runs share the full data, so
# only their training seed differs. (axis, % kept, replicate, stem, run dir)
REPLICATES = [
    ("conds", 100, "r0", "genesis", SCALING[0][3]),
    ("conds", 100, "r1", "genesis_100_mseed65", "bio/main/h4096x8192_nrs256_vg05rs256_5ep_state2k_tahoe_bio_100cond_seed0_mseed65/261012_174122_cur-between_epoch-tp458752-bs1792xnrm256_nre256_lr0.0009_wd0.0001_biohold_condmlp-D256-h4096x8192_cond-plate_mlpout-relu_vg0.5p0.5np6k_log1p_s65_h4096x8192_nrs256_vg05rs256_5ep_state2k_tahoe_bio_100cond_seed0_mseed65"),
    ("conds", 100, "r2", "genesis_100_mseed66", "bio/main/h4096x8192_nrs256_vg05rs256_5ep_state2k_tahoe_bio_100cond_seed0_mseed66/261012_212356_cur-between_epoch-tp458752-bs1792xnrm256_nre256_lr0.0009_wd0.0001_biohold_condmlp-D256-h4096x8192_cond-plate_mlpout-relu_vg0.5p0.5np6k_log1p_s66_h4096x8192_nrs256_vg05rs256_5ep_state2k_tahoe_bio_100cond_seed0_mseed66"),
    ("conds", 25, "r0", "genesis_25conds_11ep_r0", "bio/scaling/n_conds_025/h4096x8192_nrs256_vg05rs256_11ep_state2k_tahoe_bio_25cond_seed0/261014_085704_cur-between_epoch-tp458752-bs1792xnrm256_nre256_lr0.0009_wd0.0001_tahoe_scalin_condmlp-D256-h4096x8192_cond-plate_mlpout-relu_vg0.5p0.5np6k_log1p_hbssf025_s64_h4096x8192_nrs256_vg05rs256_11ep_state2k_tahoe_bio_25cond_seed0"),
    ("conds", 25, "r1", "genesis_25conds_11ep_r1", "bio/scaling/n_conds_025/h4096x8192_nrs256_vg05rs256_11ep_state2k_tahoe_bio_25cond_seed1/261014_234413_cur-between_epoch-tp458752-bs1792xnrm256_nre256_lr0.0009_wd0.0001_tahoe_scalin_condmlp-D256-h4096x8192_cond-plate_mlpout-relu_vg0.5p0.5np6k_log1p_hbssf025_s65_h4096x8192_nrs256_vg05rs256_11ep_state2k_tahoe_bio_25cond_seed1"),
    ("conds", 25, "r2", "genesis_25conds_11ep_r2", "bio/scaling/n_conds_025/h4096x8192_nrs256_vg05rs256_11ep_state2k_tahoe_bio_25cond_seed2/261015_134031_cur-between_epoch-tp458752-bs1792xnrm256_nre256_lr0.0009_wd0.0001_tahoe_scalin_condmlp-D256-h4096x8192_cond-plate_mlpout-relu_vg0.5p0.5np6k_log1p_hbssf025_s66_h4096x8192_nrs256_vg05rs256_11ep_state2k_tahoe_bio_25cond_seed2"),
    ("conds", 5, "r0", "genesis_5conds_25ep_r0", "bio/scaling/n_conds_005/h4096x8192_nrs256_vg05rs256_25ep_state2k_tahoe_bio_5cond_seed0/261013_143712_cur-between_epoch-tp458752-bs1792xnrm256_nre256_lr0.0009_wd0.0001_tahoe_scalin_condmlp-D256-h4096x8192_cond-plate_mlpout-relu_vg0.5p0.5np6k_log1p_hbssf005_s64_h4096x8192_nrs256_vg05rs256_25ep_state2k_tahoe_bio_5cond_seed0"),
    ("conds", 5, "r1", "genesis_5conds_25ep_r1", "bio/scaling/n_conds_005/h4096x8192_nrs256_vg05rs256_25ep_state2k_tahoe_bio_5cond_seed1/261013_202453_cur-between_epoch-tp458752-bs1792xnrm256_nre256_lr0.0009_wd0.0001_tahoe_scalin_condmlp-D256-h4096x8192_cond-plate_mlpout-relu_vg0.5p0.5np6k_log1p_hbssf005_s65_h4096x8192_nrs256_vg05rs256_25ep_state2k_tahoe_bio_5cond_seed1"),
    ("conds", 5, "r2", "genesis_5conds_25ep_r2", "bio/scaling/n_conds_005/h4096x8192_nrs256_vg05rs256_25ep_state2k_tahoe_bio_5cond_seed2/261015_064123_cur-between_epoch-tp458752-bs1792xnrm256_nre256_lr0.0009_wd0.0001_tahoe_scalin_condmlp-D256-h4096x8192_cond-plate_mlpout-relu_vg0.5p0.5np6k_log1p_hbssf005_s66_h4096x8192_nrs256_vg05rs256_25ep_state2k_tahoe_bio_5cond_seed2"),
    ("cells", 25, "r0", "genesis_25cells_allctrl_11ep_r0", "bio/scaling/n_cells_025_keepctrl/h4096x8192_nrs256_vg05rs256_11ep_state2k_tahoe_bio_25cell_keepctrl_seed0/261014_115029_cur-between_epoch-tp458752-bs1792xnrm256_nre256_lr0.0009_wd0.0001_tahoe_scalin_condmlp-D256-h4096x8192_cond-plate_mlpout-relu_vg0.5p0.5np6k_log1p_celln22129534_s64_h4096x8192_nrs256_vg05rs256_11ep_state2k_tahoe_bio_25cell_keepctrl_seed0"),
    ("cells", 25, "r1", "genesis_25cells_allctrl_11ep_r1", "bio/scaling/n_cells_025_keepctrl/h4096x8192_nrs256_vg05rs256_11ep_state2k_tahoe_bio_25cell_keepctrl_seed1/261016_181513_cur-between_epoch-tp458752-bs1792xnrm256_nre256_lr0.0009_wd0.0001_tahoe_scalin_condmlp-D256-h4096x8192_cond-plate_mlpout-relu_vg0.5p0.5np6k_log1p_celln21887263_s65_h4096x8192_nrs256_vg05rs256_11ep_state2k_tahoe_bio_25cell_keepctrl_seed1"),
    ("cells", 25, "r2", "genesis_25cells_allctrl_11ep_r2", "bio/scaling/n_cells_025_keepctrl/h4096x8192_nrs256_vg05rs256_11ep_state2k_tahoe_bio_25cell_keepctrl_seed2/261016_181556_cur-between_epoch-tp458752-bs1792xnrm256_nre256_lr0.0009_wd0.0001_tahoe_scalin_condmlp-D256-h4096x8192_cond-plate_mlpout-relu_vg0.5p0.5np6k_log1p_celln21750407_s66_h4096x8192_nrs256_vg05rs256_11ep_state2k_tahoe_bio_25cell_keepctrl_seed2"),
    ("cells", 5, "r0", "genesis_5cells_allctrl", SCALING[4][3]),
    # r1/r2: perturbed cells matched to conds5 r1/r2's perturbed count
    ("cells", 5, "r1", "genesis_5cells_allctrl_r1", "bio/scaling/n_cells_005_keepctrl/h4096x8192_nrs256_vg05rs256_25ep_state2k_tahoe_bio_5cell_keepctrl_seed1/261018_182521_cur-between_epoch-tp458752-bs1792xnrm256_nre256_lr0.0009_wd0.0001_tahoe_scalin_condmlp-D256-h4096x8192_cond-plate_mlpout-relu_vg0.5p0.5np6k_log1p_celln4665879_s65_h4096x8192_nrs256_vg05rs256_25ep_state2k_tahoe_bio_5cell_keepctrl_seed1"),
    ("cells", 5, "r2", "genesis_5cells_allctrl_r2", "bio/scaling/n_cells_005_keepctrl/h4096x8192_nrs256_vg05rs256_25ep_state2k_tahoe_bio_5cell_keepctrl_seed2/261019_205600_cur-between_epoch-tp458752-bs1792xnrm256_nre256_lr0.0009_wd0.0001_tahoe_scalin_condmlp-D256-h4096x8192_cond-plate_mlpout-relu_vg0.5p0.5np6k_log1p_celln4632324_s66_h4096x8192_nrs256_vg05rs256_25ep_state2k_tahoe_bio_5cell_keepctrl_seed2"),
]


# ----------------------------------------------------------------------------
# 2. Read the plate pairs: Figure 2a's, then each downsampled arm on exactly those pairs
# ----------------------------------------------------------------------------

def read_pairs(data: Path, stem: str) -> pd.DataFrame:
    df = pd.read_parquet(data / "plate_pairs" / f"pair_metrics__{stem}.parquet").set_index(KEYS)
    if df.index.has_duplicates:
        raise ValueError(f"{stem}: duplicate plate-pair keys")
    return df


def load_pairs(data: Path) -> pd.DataFrame:
    """Figure 2's pair set (repeat_pairs.load_pairs, the same rule on this bundle's copies of its
    files), indexed by the pair."""
    return repeat_pairs.load_pairs(data).set_index(KEYS)


def add_scaling_arms(pairs: pd.DataFrame, arms: pd.DataFrame, data: Path) -> pd.DataFrame:
    """The downsampled GENESIS arms on exactly Figure 2's pairs; none may be missing."""
    out = pairs.copy()
    for r in arms[arms.stem.ne("genesis")].itertuples():
        df = read_pairs(data, r.stem)
        if missing := len(pairs.index.difference(df.index)):
            raise RuntimeError(f"{r.stem}: missing {missing} of Figure 2's pairs")
        df = df.loc[pairs.index]
        # DEG's repeat is deterministic, so a mismatch means a different pair or gene set
        if not np.array_equal(df["true_true__deg_adaptivek_truth"], pairs[f"{REPEAT}__deg_adaptivek_truth"]):
            raise RuntimeError(f"{r.stem}: its repeat disagrees with Figure 2's")
        for m in PAIR_METRICS:
            out[f"{r.stem}__{m}"] = df[f"cross_mean__{m}"].to_numpy()
            for d in ("ij", "ji"):
                if not np.isfinite(v := df[f"cross_{d}__{m}"].to_numpy()).all():
                    raise RuntimeError(f"{r.stem}: non-finite cross_{d}__{m} on Figure 2's pairs")
                out[f"{r.stem}__{m}__{d}"] = v
    return out


# ----------------------------------------------------------------------------
# 3. Load the 3 runs per point and pool them (Figure 3a's variants)
# ----------------------------------------------------------------------------

# A run's seed sets its subsample as well as its initialisation, so a point's mean is over both; the CIs and
# tests then run once, on the means. Figure 2's seed-64 run is the GENESIS column; every other run is its stem.
def column(stem: str) -> str:
    return "GENESIS" if stem == "genesis" else stem


def load_runs(data: Path) -> pd.DataFrame:
    """data/scaling_runs.csv (training_runs.py --runs): one row per run, 3 per point."""
    runs = pd.read_csv(data / "scaling_runs.csv")
    if (n := runs.groupby(["axis", "pct_kept"]).size()).ne(3).any():
        raise ValueError(f"expected 3 runs per point:\n{n}")
    return runs


def pooled_arms(runs: pd.DataFrame) -> pd.DataFrame:
    """One arm per point, stem pool_<axis><pct>; cell and condition counts are the means over its runs."""
    if runs.n_ctrl.nunique() != 1:
        raise ValueError(f"runs train on different control sets: {sorted(runs.n_ctrl.unique())}")
    arms = (runs.assign(n_pert=runs.n_train - runs.n_ctrl).groupby(["axis", "pct_kept"], sort=False)
            [["n_train", "n_pert", "n_ctrl", "n_conds"]].mean().reset_index())
    arms["stem"] = "pool_" + arms.axis + arms.pct_kept.astype(str)
    return arms


def pool_runs(frame: pd.DataFrame, runs: pd.DataFrame, suffixes: list[str]) -> pd.DataFrame:
    """Per row (pair or condition), each point's mean over its runs, as pool_<axis><pct><suffix>."""
    new = {}
    for (a, p), g in runs.groupby(["axis", "pct_kept"]):
        for sfx in suffixes:
            v = frame[[f"{column(s)}{sfx}" for s in g.stem]].to_numpy(float)
            if not np.isfinite(v).all():
                raise ValueError(f"{a} {p}%{sfx}: non-finite run values; a mean would hide them")
            new[f"pool_{a}{p}{sfx}"] = v.mean(axis=1)
    return frame.assign(**new)


def mean_sem(y: np.ndarray) -> tuple[float, float]:
    if not np.isfinite(y).all() or len(y) < 2:
        raise RuntimeError(f"cannot summarise {len(y)} values with {int((~np.isfinite(y)).sum())} non-finite")
    return float(y.mean()), float(y.std(ddof=1) / np.sqrt(len(y)))


# ----------------------------------------------------------------------------
# 4. Test every point against the repeat, with Figure 2a's tests and CIs
# ----------------------------------------------------------------------------

# Figure 2a's tests, unchanged, on the same pairs: both have the null of NO gap from the
# repeat, two-sided, condition-clustered t:
#   R = mean(arm) / mean(repeat); H0 R = 1
#   win rate against the repeat, the two directions' votes averaged; H0 50% (replicate level)
# A point gets Figure 2a's star when NEITHER finds a gap: no difference detected, not equivalence.
# The 100% point IS Figure 2's GENESIS, so its rows must reproduce Figure 2's table. The
# non-inferiority margins (5%, 10%) are kept in the table, not drawn.
R_TEST = NO_GAP_TESTS[0]


def tests_vs_repeat(pairs: pd.DataFrame, arms: pd.DataFrame) -> pd.DataFrame:
    cluster = pairs.index.to_frame()[["cell_name", "drug", "dose"]].astype(str).agg("|".join, axis=1).to_numpy()
    rows = []
    for m, hb in HIGHER_IS_BETTER.items():
        ref = pairs[f"{REPEAT}__{m}"].to_numpy()
        for r in arms.itertuples():
            col = f"{column(r.stem)}__{m}"
            model = pairs[col].to_numpy()
            key = dict(metric=m, axis=r.axis, pct_kept=r.pct_kept, arm=r.stem)
            ab, ba = (pairs[f"{col}__{d}"].to_numpy() for d in ("ij", "ji"))
            rows += [dict(key, **t) for t in repeat_tests(ref, model, ab, ba, cluster, hb)]
    t = pd.DataFrame(rows)
    t["mark"] = mark_no_gap(t)                                         # one mark, on the R row
    return t


def pair_cis(pairs: pd.DataFrame, arms: list[str], m: str, published: pd.DataFrame
             ) -> dict[str, tuple[float, float, float]]:
    """(mean, lo, hi) per arm on the plate pairs: Figure 2's cluster bootstrap (line x drug, doses pooled),
    paired across the arms. ``published``: Figure 2's confidence_intervals.csv; every arm Figure 2a drew
    must match it, or the plate pairs are not Figure 2a's."""
    cl = pairs.index.to_frame()[UNITS[PRIMARY["tahoe"]]].astype(str).agg("|".join, axis=1).to_numpy()
    mean, lo, hi = cluster_bootstrap_ci(pairs[[f"{a}__{m}" for a in arms]].to_numpy(float), cl)
    out = {a: (float(x), float(y), float(z)) for a, x, y, z in zip(arms, mean, lo, hi)}
    pub = published[published.panel.eq("a") & published.metric.eq(m) & published.drawn].set_index("arm")
    if REPEAT not in pub.index or "GENESIS" not in pub.index:
        raise RuntimeError(f"{m}: Figure 2a's CI table has no repeat / GENESIS row to check against")
    for a in (a for a in arms if a in pub.index):
        if not np.allclose(out[a], pub.loc[a, ["mean", "lo", "hi"]].to_numpy(float), rtol=0, atol=1e-9):
            raise RuntimeError(f"{m} {a}: CI {out[a]} != Figure 2a {tuple(pub.loc[a, ['mean', 'lo', 'hi']])}")
    return out


# ----------------------------------------------------------------------------
# 5. Build panel b's table: 5 held-out lines x 5 overlap levels x 3 entity seeds
# ----------------------------------------------------------------------------

def load_bio(data: Path, metric_key: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Seed mean and s.d. per (held-out line, overlap level), from the 5 x 5 x 3 table."""
    t = pd.read_csv(data / "overlap_bio" / "cell_seed_overlap_strict_drug.csv")
    t = t[t.metric_key.eq(metric_key)]
    if len(t) != 5 * len(BIO_LEVELS) * 3 or t.metric_mean.isna().any():
        raise ValueError(f"{metric_key}: expected a full 5 x {len(BIO_LEVELS)} x 3 table, got {len(t)} rows")
    g = t.groupby(["cell_line", "overlap"])["metric_mean"]
    mean, sd = g.mean().unstack()[list(BIO_LEVELS)], g.std(ddof=1).unstack()[list(BIO_LEVELS)]
    return mean.to_numpy(), sd.to_numpy(), list(mean.index)
