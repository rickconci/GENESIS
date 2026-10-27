#!/usr/bin/env bash
# Figure 1: the replication ladder, split-half vs the experimental repeat at matched cells per arm (Fig 1d).
# In:   eval cache only (truth, no checkpoint); core_util_data/tahoe/eval_inputs/ladder_cohort.csv
#       The cohort is a FROZEN input, not redrawn: 100 of 175 eligible two-plate conditions, drawn with seed 17.
# Out:  $OUT/core_results/figure1/ladder/{reps.parquet, manifest.json, cohort_exactly_two_plates.csv}
# Ships as-is (not rerun): nothing.   Also feeds: nothing.
# Cost: full ~12 min GPU; SUBSET ~2.5 min. Run on an H100: the published file was made on one, and an A100
# differs from it by ~4e-5, so the exact SUBSET check fails there.
#   GPU=0 OUT=<new dir> [SUBSET=1] bash scripts/figures/figure1/score/main/score.sh
source "$(dirname "$0")/../../../../eval/_env.sh"
need_gpu

# 1. Run the ladder. SUBSET takes the first 5 cohort rows with --limit, never a cut CSV: each condition is
# seeded seed + its row index, so a cut CSV would re-draw the cells and differ from the published rows by construction.
# n-repeats 20 and seed 17 are the values recorded in the published manifest.json.
run work/ladder/reps.parquet \
    "$PY" -u scripts/eval/reference.py ladder --dataset tahoe \
    --eval-cache '{data:tahoe}/eval_cache/normalization_raw_counts/geneset_c0d6420ffc0fa861__n_2000' \
    --cohort core_util_data/tahoe/eval_inputs/ladder_cohort.csv --n-repeats 20 --seed 17 --device cuda:0 \
    ${SUBSET:+--limit 5} --out "$OUT/work/ladder"

# 2. Place the results under $OUT/core_results
place work/ladder/reps.parquet figure1/ladder/reps.parquet
# (full run only) manifest and cohort differ by construction in a subset run (5 conditions, the --limit cohort), so only a full run places them
if [ -z "${SUBSET:-}" ]; then
    run work/ladder/cohort_exactly_two_plates.csv \
        cp core_util_data/tahoe/eval_inputs/ladder_cohort.csv "$OUT/work/ladder/cohort_exactly_two_plates.csv"
    place work/ladder/manifest.json figure1/ladder/manifest.json
    place work/ladder/cohort_exactly_two_plates.csv figure1/ladder/cohort_exactly_two_plates.csv
fi
finish
