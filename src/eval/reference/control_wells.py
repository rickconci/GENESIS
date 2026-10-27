"""How much does a condition's response change when only its control well changes?

A dataset with several control wells per unit (PBMC: six PBS wells per donor x cell type)
gives a reference with the perturbed cells held fixed: score one condition's cells against
control well i, again against well j, and compare the two deltas and DEG calls. Two arms:

  real      the measured control wells
  shuffled  the unit's control cells pooled and re-split into pseudo-wells of the real wells'
            sizes: the same control sampling noise, and no well effect

real < shuffled is the well effect. One kernel (``score_raw_pair``, delta + DEG only).
"""
from __future__ import annotations

from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from src.eval.reference.pairs import score_raw_pair

DEG_KEY = "deg_raw_int_rgs/deg_overlap_top_p_ranked_k100_fdr0p001_lfc0p1_adaptivek_truth_detfloor10"
PEARSON_KEY = "log1p_rgs/pearson_r_diff"


def pseudo_wells(real: dict[str, np.ndarray], wells: Sequence[str], seed: int) -> dict[str, np.ndarray]:
    """The wells' pooled cells, shuffled into pseudo-wells with the real wells' sizes (in ``wells`` order)."""
    pool = np.concatenate([real[w] for w in wells], axis=0)
    perm = np.random.default_rng(seed).permutation(pool.shape[0])
    cuts = np.cumsum([real[w].shape[0] for w in wells])[:-1]
    return dict(zip(wells, (pool[idx] for idx in np.split(perm, cuts))))


def swap_pairs(pert: torch.Tensor, ctrl: dict[str, torch.Tensor],
               wells: Sequence[str]) -> list[tuple[str, str, float, float]]:
    """(well i, well j, DEG overlap, Pearson delta) for every unordered pair of control wells."""
    out = []
    for wi, wj in combinations(wells, 2):
        m = score_raw_pair(pert, pert, ctrl[wi], ctrl[wj], dist=False)
        out.append((wi, wj, m[DEG_KEY], m[PEARSON_KEY]))
    return out


def summarize(pairs: pd.DataFrame, keys: Sequence[str], n_boot: int, seed: int) -> dict:
    """Average the well pairs within a condition, then bootstrap conditions (95% percentile CI)."""
    per = pairs.pivot_table(index=list(keys), columns="arm", values=["deg", "pearson_delta"], aggfunc="mean")
    rng = np.random.default_rng(seed)
    out = {}
    for metric in ("deg", "pearson_delta"):
        sub = per[metric].dropna()
        real, shuf = sub["real"].to_numpy(), sub["shuffled"].to_numpy()
        idx = rng.integers(0, len(sub), size=(n_boot, len(sub)))

        def ci(v: np.ndarray) -> list[float]:
            return [float(x) for x in np.percentile(v[idx].mean(axis=1), [2.5, 97.5])]
        out[metric] = {"n_conditions": int(len(sub)), "n_dropped_undefined": int(len(per) - len(sub)),
                       "real_mean": float(real.mean()), "real_ci": ci(real),
                       "shuffled_mean": float(shuf.mean()), "shuffled_ci": ci(shuf),
                       "diff_mean": float((real - shuf).mean()), "diff_ci": ci(real - shuf)}
    return out


# Within- vs between-well (design: D. D'Ascenzo). The real-vs-shuffled test above compares wells of
# unequal sizes; here every control sample has the same s_a cells, so a gap cannot come from sample size.
#   within   two disjoint s_a samples of ONE well: control sampling noise alone
#   between  s_a cells of well i against s_a of well j: the same noise plus the well effect
# The cytokine cells stay fixed in both arms; only the control sample changes.

MIN_HALF = 50  # a stratum is kept only if s_a reaches this


def half_size(sizes: Mapping[str, int], min_half: int = MIN_HALF) -> tuple[list[str], int] | None:
    """(wells kept, s_a) for one stratum, or None.

    s_a is half the smallest well, so two disjoint samples fit in every kept well. If that is below
    ``min_half``, the single smallest well is dropped and the next smallest sets s_a.
    """
    order = sorted(sizes, key=lambda w: (sizes[w], w))
    for n_drop in (0, 1):
        s_a = sizes[order[n_drop]] // 2
        if s_a >= min_half:
            return [w for w in sizes if w not in order[:n_drop]], s_a
    return None


def well_draws(sizes: Mapping[str, int], wells: Sequence[str], s_a: int, n_draws: int,
               seed: int) -> list[tuple[str, str, str, int, np.ndarray, np.ndarray]]:
    """(arm, well_i, well_j, draw, idx_a, idx_b): row indices into the wells' control cells.

    within: idx_a, idx_b disjoint, both in well_i (= well_j); between: idx_a in well_i, idx_b in well_j.
    """
    rng, out = np.random.default_rng(seed), []
    for w in wells:
        for d in range(n_draws):
            if sizes[w] < 2 * s_a:
                raise ValueError(f"well {w} has {sizes[w]} cells, fewer than 2 x s_a = {2 * s_a}")
            p = rng.permutation(sizes[w])
            out.append(("within", w, w, d, p[:s_a], p[s_a:2 * s_a]))
    for wi, wj in combinations(wells, 2):
        for d in range(n_draws):
            out.append(("between", wi, wj, d, rng.choice(sizes[wi], s_a, replace=False),
                        rng.choice(sizes[wj], s_a, replace=False)))
    return out


def within_between_tests(rows: pd.DataFrame, keys: Sequence[str], cluster: Sequence[str],
                         n_boot: int, seed: int) -> dict:
    """Per condition, the mean over its within draws (reference) and its between draws, then Fig. 2a's
    no-gap tests: R = mean(between) / mean(within) against 1 and the share of conditions where between
    agrees less than within against 1/2, clustered by ``cluster`` (conditions sharing control draws).
    """
    from src.stats.equivalence import point_test, ratio_to_reference, worse_share

    out = {}
    for metric in [m for m in ("deg", "pearson_delta", "energy") if m in rows]:
        higher_is_better = metric != "energy"            # agreement for DEG / r_delta; distance for energy
        per = rows.pivot_table(index=list(keys), columns="arm", values=metric, aggfunc="mean").dropna()
        clu = per.reset_index()[list(cluster)].astype(str).agg("|".join, axis=1).to_numpy()
        ref, mod = per["within"].to_numpy(), per["between"].to_numpy()
        r = ratio_to_reference(ref, mod, clu, n_boot=n_boot, seed=seed, alpha=0.025)
        w = worse_share(ref, mod, clu, higher_is_better, n_boot=n_boot, seed=seed, alpha=0.025)
        out[metric] = {"n_conditions": int(len(per)), "n_clusters": int(len(set(clu))),
                       "within_mean": float(ref.mean()), "between_mean": float(mod.mean()),
                       "ratio": {"estimate": r.value, "boot": [r.boot_lo, r.boot_hi], **point_test(r, 1.0, log_scale=True)},
                       "between_worse_share": {"estimate": w.value, "boot": [w.boot_lo, w.boot_hi],
                                               **point_test(w, 0.5, log_scale=False)}}
    return out
