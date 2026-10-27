#!/usr/bin/env python3
"""Step 1 of the shallow-sampling analysis: which training conditions, and which of their cells.

A Figure 3a cells arm trained on a uniform draw of rows, so each training condition (cell line x drug x dose x
plate) was seen through a fraction of its cells; the rest never reached the model and are the truth. Writes the
DRAWN and LEFT-OUT global rows per condition to OUT/{conditions.parquet, rows.npz}, from which `same_plate_sample`
and `cross_pairs` pick the cohorts. The constants below are shared by every script here.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))   # every script here imports cohort first: this puts src/ on its path
from src.data.vocab import parse_doses, to_categorical_str  # noqa: E402
from src.data.data_utils import load_eval_manifest  # noqa: E402
from src.eval.reference.pairs import MIN_CELLS_PER_PLATE  # noqa: E402
from src.paths import expand_path  # noqa: E402

# The Figure 3a cells arms, each a uniform draw of training rows (controls included), last.ckpt.
# SHALLOW_ARM picks one for every script here; 005 is the default and writes where it always has.
ARMS = {
    "005": ("runs/tahoe/bio/scaling/n_cells_005/h4096x8192_nrs256_vg05rs256_100ep_state2k_tahoe_bio_5cell_seed0/"
            "260920_162654_cur-between_epoch-tp458752-bs1792xnrm256_nre256_lr0.0009_wd0.0001_tahoe_scalin_condmlp-"
            "D256-h4096x8192_cond-plate_mlpout-relu_vg0.5p0.5np6k_log1p_celln6761692_s64_h4096x8192_nrs256_vg05rs256_"
            "100ep_state2k_tahoe_bio_5cell_seed0", ""),
    "025": ("runs/tahoe/bio/scaling/n_cells_025/h4096x8192_nrs256_vg05rs256_20ep_state2k_tahoe_bio_25cell_seed0/"
            "260918_151050_cur-between_epoch-tp458752-bs1792xnrm256_nre256_lr0.0009_wd0.0001_tahoe_scalin_condmlp-"
            "D256-h4096x8192_cond-plate_mlpout-relu_vg0.5p0.5np6k_log1p_celln24921726_s64_h4096x8192_nrs256_vg05rs256_"
            "20ep_state2k_tahoe_bio_25cell_seed0", "n_cells_025"),
    # NOT a fraction: exactly 25 perturbed cells per condition (every control kept), so training depth is
    # fixed and independent of condition size (percond.py)
    "pc25": ("runs/tahoe/bio/scaling/n_cells_percond25/h4096x8192_nrs256_vg05rs256_25ep_state2k_tahoe_bio_percond25_seed0/"
             "261012_220500_cur-between_epoch-tp458752-bs1792xnrm256_nre256_lr0.0009_wd0.0001_tahoe_scalin_condmlp-"
             "D256-h4096x8192_cond-plate_mlpout-relu_vg0.5p0.5np6k_log1p_s64_h4096x8192_nrs256_vg05rs256_25ep_"
             "state2k_tahoe_bio_percond25_seed0", "n_cells_percond25"),
}
ARM = os.environ.get("SHALLOW_ARM", "005")
if ARM not in ARMS:
    raise SystemExit(f"SHALLOW_ARM={ARM!r}; expected one of {sorted(ARMS)}")
RUN = REPO / ARMS[ARM][0]
HVG_CACHE = "{data:tahoe}/hvg_cache/geneset_c0d6420ffc0fa861__n_2000"  # raw counts by global row, the run's genes
# SHALLOW_OUT redirects every script's outputs (a level-3 driver writes under its own $OUT, never results/)
OUT = Path(os.environ.get("SHALLOW_OUT") or REPO / "results/tahoe/scaling_results/shallow_denoising" / ARMS[ARM][1])
MIN_DRAWN = 4      # the eval's min_cells: fewer drawn cells cannot make a pseudobulk
SPLIT_PLATES = ("plate6", "plate14")   # the reproducible 5 uM plate pair
MIN_HALF = 50                          # >= 50 held-out cells per half; results are then read against
                                       # the split-half score, not cut at a depth
SEED = 20261009
KEYS = ["cell_name", "drug", "dose", "plate"]


# 1. One row per perturbed training cell: its condition and whether it was drawn

def obs_keys() -> pd.DataFrame:
    """(cell_name, drug, dose, plate) per global row, as categorical codes over the whole cache."""
    cache = Path(expand_path(HVG_CACHE))
    man = json.loads((cache / "manifest.json").read_text())
    cats: dict[str, dict[str, int]] = {k: {} for k in KEYS}
    cols: dict[str, list[np.ndarray]] = {k: [] for k in KEYS}
    for s in man["shards"]:
        z = np.load(cache / s["cache_subdir"] / "obs_sidecar.npz", allow_pickle=True)
        if not np.array_equal(z["local_obs_index"], np.arange(s["n_obs"])):
            raise RuntimeError(f"shard {s['shard_id']}: rows are not stored in order")
        # dose comes from drugname_drugconc exactly as the datamodule derives it
        dose_cats = [to_categorical_str(float(v)) for v in parse_doses(z["drugname_drugconc__categories"])]
        src = {"cell_name": ("cell_name", None), "drug": ("drug", None),
               "dose": ("drugname_drugconc", dose_cats), "plate": ("plate", None)}
        for k, (field, names) in src.items():
            names = names if names is not None else [str(c) for c in z[f"{field}__categories"]]
            lut = np.array([cats[k].setdefault(n, len(cats[k])) for n in names], dtype=np.int32)
            cols[k].append(lut[z[f"{field}__codes"]])
    if sum(len(c) for c in cols["plate"]) != man["total_obs"]:
        raise RuntimeError("sidecars do not cover the manifest's rows")
    return pd.DataFrame({k: pd.Categorical.from_codes(np.concatenate(cols[k]), list(cats[k])) for k in KEYS})


def split_rows(obs: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Conditions with their drawn/left counts, plus every perturbed training row grouped by condition."""
    sc = RUN / "split_cache"
    pre = np.load(sc / "pert_train_idx_pre_hbss.npy")   # every perturbed training row, before the draw
    drawn = np.isin(pre, np.load(sc / "train_idx.npy"))
    df = obs.iloc[pre].reset_index(drop=True)
    df["row"], df["drawn"] = pre, drawn
    df = df.sort_values([*KEYS, "drawn", "row"], kind="stable")    # per condition: left rows, then drawn rows
    g = df.groupby(KEYS, observed=True, sort=False)
    cond = g.drawn.agg(n_drawn="sum", n_tot="size").reset_index()
    cond["n_left"] = cond.n_tot - cond.n_drawn
    cond["offset"] = np.concatenate([[0], np.cumsum(cond.n_tot.to_numpy())[:-1]])
    return cond, df.row.to_numpy(np.int64), df.drawn.to_numpy()


def check_against_run(cond: pd.DataFrame) -> None:
    """Every condition's total must equal the run's own condition summary: same keys, same counts."""
    summ = pd.read_parquet(load_eval_manifest(RUN).condition_summary)
    m = cond.astype({k: str for k in KEYS}).merge(summ.astype({k: str for k in KEYS}), on=KEYS, how="left")
    if m.num_cells.isna().any():
        raise RuntimeError(f"{int(m.num_cells.isna().sum())} conditions missing from the run's condition summary, "
                           f"e.g. {m[m.num_cells.isna()][KEYS].head(3).to_dict('records')}")
    if bad := int((m.num_cells != m.n_tot).sum()):
        raise RuntimeError(f"{bad} conditions disagree with the run's cell counts")


# 2. The cohorts

def eligible(cond: pd.DataFrame) -> pd.Series:
    """Enough drawn cells to be an estimate, enough left-out cells to be a truth (the repeat benchmark's 50)."""
    return (cond.n_drawn >= MIN_DRAWN) & (cond.n_left >= MIN_CELLS_PER_PLATE)


def same_plate_sample(cond: pd.DataFrame) -> np.ndarray:
    """Every condition on SPLIT_PLATES whose held-out cells split into two halves of >= MIN_HALF."""
    keep = eligible(cond) & (cond.n_left >= 2 * MIN_HALF) & cond.plate.isin(SPLIT_PLATES)
    return np.sort(cond.cond_id[keep].to_numpy())


def cross_pairs(cond: pd.DataFrame) -> pd.DataFrame:
    """One plate pair per combination with >= 2 eligible plates, both directions (X -> Y and Y -> X), so a
    compound repeated on every plate (Adagrasib 0.05 uM, up to 14 plates) weighs as one combination, not 182
    pairs. The pair is drawn with a per-combination seed, so adding or removing combinations moves no other pick."""
    ok = cond[eligible(cond)]
    out = []
    for key, g in ok.groupby(["cell_name", "drug", "dose"], observed=True, sort=True):
        if len(g) < 2:
            continue
        rng = np.random.default_rng([SEED, *(ord(ch) for ch in "|".join(map(str, key)))])
        x, y = g.sort_values("plate").cond_id.to_numpy()[rng.choice(len(g), size=2, replace=False)]
        out += [(int(x), int(y)), (int(y), int(x))]
    return pd.DataFrame(out, columns=["cond_id_x", "cond_id_y"])


def main() -> None:
    obs = obs_keys()
    cond, rows, drawn = split_rows(obs)
    check_against_run(cond)
    cond["cond_id"] = np.arange(len(cond))
    OUT.mkdir(parents=True, exist_ok=True)
    cond = cond.astype({k: str for k in KEYS})
    cond.to_parquet(OUT / "conditions.parquet", index=False)
    np.savez(OUT / "rows.npz", rows=rows, drawn=drawn)
    ok, pairs = cond[eligible(cond)], cross_pairs(cond)
    print(f"{len(cond):,} training conditions; drawn {int(drawn.sum()):,} / left {int((~drawn).sum()):,} cells")
    print(f"same_plate: {len(ok):,} eligible; n_drawn median {ok.n_drawn.median():.0f} "
          f"(5-95% {ok.n_drawn.quantile(.05):.0f}-{ok.n_drawn.quantile(.95):.0f})")
    print(f"cross_plate: {len(pairs) // 2:,} combinations, one plate pair each, both directions -> {OUT}")


if __name__ == "__main__":
    main()
