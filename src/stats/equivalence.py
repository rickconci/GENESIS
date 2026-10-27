"""Is a model as close to the truth as an experimental replicate?  Clustered equivalence tests.

Each row is one *pair*: a reference score (two real replicates scored against each other,
e.g. plate A vs plate B) and a model score on the same pair (the model scored against the
same truth).  A difference test cannot show sameness -- failing to reject "no difference"
is not evidence, and more noise makes it easier -- so both tests here flip the null to
"the gap is at least a margin" and reject it at alpha (one 90% CI per alpha=0.05):

  ratio      R = mean(model) / mean(reference), pair-weighted.  Non-inferiority: the
             CI bound on the *worse* side clears 1 - m (higher is better) or 1 + m (lower
             is better).  A relative margin on the difference is the same test, since
             Δ / mean(reference) = R - 1, so one margin reads the same on every metric.
  win rate   P = share of pairs where the model is worse (ties count 1/2).  Equivalence:
             the CI sits inside a band around 1/2.  One vote per pair, so a few extreme
             pairs cannot move it the way they can move a mean.

A third test keeps the ordinary null of NO gap (``point_test``, two-sided, 95% CI): a low p
finds a gap, a high p only fails to find one, so it is read with its CI.  When a pair has
two directions -- m(A, B') and m(B, A'), a model standing in for each replicate -- the win
rate averages the two VOTES (``averaged_worse_share``), not the scores: each single vote is a
fair coin under a perfect generator, a vote on the averaged score is not.

Both R and P are ratios of sums, so both come from one estimator.  Pairs that share a
replicate (a condition run on three plates gives three pairs) are not independent, so the
CLUSTER (condition) is the unit everywhere:
  bootstrap  -- resample clusters (multinomial weights), percentile CI
  t          -- delta-method influence per cluster, cluster-robust SE, t on C - 1 df
R's t-test runs on log R, where a ratio's sampling distribution is closer to symmetric.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05


# First, one clustered estimator: sum(num) / sum(den)
@dataclass(frozen=True)
class ClusteredRatio:
    value: float
    boot_lo: float  # percentile CI at (alpha, 1 - alpha), i.e. the 90% CI for alpha=0.05
    boot_hi: float
    se: float       # cluster-robust SE of ``value`` (delta method)
    dof: int        # n_clusters - 1


def _check_pairs(*cols: np.ndarray, cluster: np.ndarray) -> None:
    n = len(cluster)
    for c in cols:
        if c.shape != (n,):
            raise ValueError(f"per-pair arrays must all be shape ({n},); got {c.shape}")
        if not np.isfinite(c).all():
            raise ValueError(f"{int((~np.isfinite(c)).sum())} non-finite per-pair values; filter before testing")


@functools.lru_cache(maxsize=4)
def _boot_weights(C: int, n_boot: int, seed: int) -> np.ndarray:
    """A cluster bootstrap is a multinomial reweighting of per-cluster sums.

    Cached: a figure runs every arm x metric x test on the same clusters and seed, so it would redraw
    one matrix ~100 times (~0.5 s each, most of a figure's draw). Float, as matmul promotes the counts
    anyway (exact: bit-identical CIs). Read-only, since every caller shares it.
    """
    w = np.random.default_rng(seed).multinomial(C, np.full(C, 1 / C), size=n_boot).astype(np.float64)
    w.flags.writeable = False
    return w


def clustered_ratio(num: np.ndarray, den: np.ndarray, cluster: np.ndarray,
                    n_boot: int = 10_000, seed: int = 0, alpha: float = ALPHA) -> ClusteredRatio:
    """sum(num) / sum(den) with a cluster bootstrap CI and a cluster-robust SE."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    _check_pairs(num, den, cluster=cluster)
    ids = np.unique(cluster, return_inverse=True)[1]
    C = int(ids.max()) + 1
    if C < 2:
        raise ValueError("need >= 2 clusters for a clustered CI")
    s_num, s_den = np.bincount(ids, num, C), np.bincount(ids, den, C)
    if s_den.sum() <= 0:
        raise ValueError(f"denominator must sum to > 0, got {s_den.sum()}")
    value = s_num.sum() / s_den.sum()
    w = _boot_weights(C, n_boot, seed)
    lo, hi = np.quantile((w @ s_num) / (w @ s_den), [alpha, 1 - alpha])
    psi = (s_num - value * s_den) / s_den.sum()  # each cluster's influence; sums to zero
    se = float(np.sqrt(C / (C - 1) * (psi ** 2).sum()))
    return ClusteredRatio(float(value), float(lo), float(hi), se, C - 1)


# Then the two quantities the tests are about
def ratio_to_reference(ref: np.ndarray, model: np.ndarray, cluster: np.ndarray,
                       **kw) -> ClusteredRatio:
    """R = mean(model) / mean(reference); both means must be positive."""
    ref, model = np.asarray(ref, float), np.asarray(model, float)
    _check_pairs(ref, model, cluster=cluster)
    if ref.mean() <= 0 or model.mean() <= 0:
        raise ValueError(f"a ratio test needs positive means; got reference {ref.mean()}, model {model.mean()}")
    return clustered_ratio(model, ref, cluster, **kw)


def _votes(ref: np.ndarray, model: np.ndarray, higher_is_better: bool) -> np.ndarray:
    """1 where the model is worse than the reference, 0 where better, 1/2 on a tie."""
    worse = (model < ref) if higher_is_better else (model > ref)
    return np.where(model == ref, 0.5, worse.astype(float))


def worse_share(ref: np.ndarray, model: np.ndarray, cluster: np.ndarray,
                higher_is_better: bool, **kw) -> ClusteredRatio:
    """P(model worse than reference), ties = 1/2."""
    ref, model = np.asarray(ref, float), np.asarray(model, float)
    _check_pairs(ref, model, cluster=cluster)
    w = _votes(ref, model, higher_is_better)
    return clustered_ratio(w, np.ones_like(w), cluster, **kw)


def averaged_worse_share(ref: np.ndarray, model_ab: np.ndarray, model_ba: np.ndarray,
                         cluster: np.ndarray, higher_is_better: bool, **kw) -> ClusteredRatio:
    """P(model worse), each pair's two directions averaged: [vote m(A, B') + vote m(B, A')] / 2.

    Under a perfect generator m(A, B') is exchangeable with the reference m(A, B), so each
    vote is a fair coin and the null of 1/2 is exact.  A vote on the averaged score pits a
    mean of two scores against one score, and its null need not be 1/2.
    """
    ref, a, b = (np.asarray(x, float) for x in (ref, model_ab, model_ba))
    _check_pairs(ref, a, b, cluster=cluster)
    w = 0.5 * (_votes(ref, a, higher_is_better) + _votes(ref, b, higher_is_better))
    return clustered_ratio(w, np.ones_like(w), cluster, **kw)


# Finally the test: does the CI sit inside [lo, hi]?
def bounds_test(est: ClusteredRatio, lo: float, hi: float, log_scale: bool,
                alpha: float = ALPHA) -> dict:
    """Two one-sided tests against [lo, hi]; use lo=0 or hi=inf for a one-sided test.

    Returns the verdict from each CI method and the t-based p-value (max of the two
    one-sided p-values).  ``log_scale`` runs the t-test on log(value) (for R).
    """
    if not lo < hi:
        raise ValueError(f"need lo < hi, got [{lo}, {hi}]")
    if log_scale:
        if lo < 0 or est.value <= 0:
            raise ValueError("log-scale test needs lo >= 0 and a positive estimate")
        x, se = np.log(est.value), est.se / est.value
        with np.errstate(divide="ignore"):
            b_lo, b_hi = np.log(lo), np.log(hi)
    else:
        x, se, b_lo, b_hi = est.value, est.se, lo, hi
    tq = stats.t.ppf(1 - alpha, est.dof)
    t_lo, t_hi = x - tq * se, x + tq * se
    if se == 0:  # degenerate: no spread at all, the verdict is the point estimate's
        p = 0.0 if b_lo < x < b_hi else 1.0
    else:
        p = max(stats.t.sf((x - b_lo) / se, est.dof), stats.t.cdf((x - b_hi) / se, est.dof))
    back = np.exp if log_scale else (lambda v: v)
    return dict(pass_boot=bool(lo < est.boot_lo and est.boot_hi < hi),
                pass_t=bool(b_lo < t_lo and t_hi < b_hi),
                p_t=float(p), t_lo=float(back(t_lo)), t_hi=float(back(t_hi)))


# Or the plain question: is there any gap at all?
def point_test(est: ClusteredRatio, null: float, log_scale: bool, alpha: float = ALPHA) -> dict:
    """Two-sided t-test of value == null, with its (1 - alpha) t-CI: 95% at the default.

    Unlike ``bounds_test`` (one-sided alpha, a 90% CI), alpha here is split over both sides.
    A low p finds a gap; a high p alone shows nothing, so report the CI with it.
    ``log_scale`` tests log(value) == log(null) (for R).
    """
    if log_scale and (null <= 0 or est.value <= 0):
        raise ValueError("log-scale test needs a positive null and estimate")
    x, se, x0 = ((np.log(est.value), est.se / est.value, np.log(null)) if log_scale
                 else (est.value, est.se, null))
    if se == 0:  # degenerate: no spread at all, the point estimate decides
        p = 1.0 if x == x0 else 0.0
    else:
        p = 2 * stats.t.sf(abs(x - x0) / se, est.dof)
    tq = stats.t.ppf(1 - alpha / 2, est.dof)
    back = np.exp if log_scale else (lambda v: v)
    return dict(p=float(p), ci_lo=float(back(x - tq * se)), ci_hi=float(back(x + tq * se)))


def compare_to_reference(ref: np.ndarray, model: np.ndarray, cluster: np.ndarray,
                         higher_is_better: bool, ni_margins: tuple[float, ...] = (0.05, 0.10),
                         win_band: tuple[float, float] = (0.40, 0.60),
                         n_boot: int = 10_000, seed: int = 0) -> list[dict]:
    """Every test for one (model, metric): ratio non-inferiority per margin + win-rate band."""
    kw = dict(n_boot=n_boot, seed=seed)
    r = ratio_to_reference(ref, model, cluster, **kw)
    p = worse_share(ref, model, cluster, higher_is_better, **kw)
    # the smallest symmetric margin the worse-side bound clears: descriptive, never a margin to pick
    worst = r.boot_lo if higher_is_better else r.boot_hi
    base = dict(n_pairs=len(cluster), n_clusters=r.dof + 1,
                mean_reference=float(np.mean(ref)), mean_model=float(np.mean(model)))
    rows = []
    for m in ni_margins:
        lo, hi = ((1 - m, np.inf) if higher_is_better else (0.0, 1 + m))
        rows.append(dict(base, test="ratio_noninferiority", margin=m, bound_lo=lo, bound_hi=hi,
                         estimate=r.value, boot_lo=r.boot_lo, boot_hi=r.boot_hi,
                         smallest_passing_margin=abs(1 - worst), **bounds_test(r, lo, hi, log_scale=True)))
    rows.append(dict(base, test="winrate_equivalence", margin=win_band[1] - 0.5,
                     bound_lo=win_band[0], bound_hi=win_band[1],
                     estimate=p.value, boot_lo=p.boot_lo, boot_hi=p.boot_hi,
                     smallest_passing_margin=max(0.5 - p.boot_lo, p.boot_hi - 0.5),
                     **bounds_test(p, *win_band, log_scale=False)))
    return rows


# And every test of one arm against the repeat, as Figures 2a and 3a report them
NO_GAP_TESTS = ("ratio_no_gap", "winrate_averaged_no_gap")  # H0 R = 1; H0 averaged win rate = 1/2
NO_GAP_MARK = "n.s."


def repeat_tests(ref: np.ndarray, model: np.ndarray, model_ab: np.ndarray, model_ba: np.ndarray,
                 cluster: np.ndarray, higher_is_better: bool, n_boot: int = 10_000, seed: int = 0) -> list[dict]:
    """One arm on one metric: the non-inferiority rows (90% CI), then the two no-gap tests (95%).

    ``model`` is the pair's averaged score, ``model_ab`` / ``model_ba`` its two directions
    m(A, B') and m(B, A'), which the averaged win rate votes on.
    """
    rows = [dict(ci=0.90, **t) for t in compare_to_reference(ref, model, cluster, higher_is_better,
                                                             n_boot=n_boot, seed=seed)]
    kw = dict(n_boot=n_boot, seed=seed, alpha=0.025)  # bootstrap percentiles 2.5 / 97.5: a 95% CI
    for test, est, null, log in (
            (NO_GAP_TESTS[0], ratio_to_reference(ref, model, cluster, **kw), 1.0, True),
            (NO_GAP_TESTS[1], averaged_worse_share(ref, model_ab, model_ba, cluster, higher_is_better, **kw), 0.5, False)):
        pt = point_test(est, null, log_scale=log)
        rows.append(dict(ci=0.95, n_pairs=len(cluster), n_clusters=est.dof + 1,
                         mean_reference=float(np.mean(ref)), mean_model=float(np.mean(model)),
                         test=test, null=null, estimate=est.value, boot_lo=est.boot_lo, boot_hi=est.boot_hi,
                         p_t=pt["p"], t_lo=pt["ci_lo"], t_hi=pt["ci_hi"],
                         gap_t=pt["p"] < 0.05, gap_boot=not est.boot_lo < null < est.boot_hi))
    return rows


def mark_no_gap(t: pd.DataFrame) -> np.ndarray:
    """``n.s.`` on each (metric, arm)'s R row when NEITHER no-gap test finds a gap, else "".

    No gap detected is not equivalence; the non-inferiority rows answer that. Warns when the
    bootstrap and t verdicts of a no-gap test disagree.
    """
    no_gap = t.test.isin(NO_GAP_TESTS)
    found = t[no_gap].groupby(["metric", "arm"]).p_t.min() < 0.05
    ns = ~found.reindex(pd.MultiIndex.from_frame(t[["metric", "arm"]])).to_numpy(dtype=bool)
    if len(split := t[no_gap & (t.gap_t != t.gap_boot)]):
        print(f"WARNING: bootstrap and t verdicts disagree on {len(split)} no-gap tests:\n"
              f"{split[['metric', 'arm', 'test', 'p_t', 'gap_boot', 'gap_t']]}")
    return np.where(t.test.eq(NO_GAP_TESTS[0]) & ns, NO_GAP_MARK, "")
