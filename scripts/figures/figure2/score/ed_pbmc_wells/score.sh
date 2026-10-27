#!/usr/bin/env bash
# Extended Data (Fig 2 PBMC wells): control wells against each other, within vs between wells, at the sizes of the test conditions (truth only).
# In:   eval cache only (no checkpoint); core_results/figure2/split_half/parse_bio/GENESIS.json (names the conditions to score)
#       runs/parse/bio/main/parse_bio_well_conds100_5ep_donorholdout_seed0/260921_192149_…/eval/condition_summary.parquet
# Out:  $OUT/core_results/figure2/pbmc_wells/{well_draws_rows.csv, well_draws_energy.csv, well_draws_summary.json, condition_summary.parquet}
# Ships as-is (not rerun): nothing.   Needs first: figure2/score/main/score.sh (the GENESIS file is read from core_results/, as published).
# Cost: CPU, 3-6 h per shard (4 shards, run one after another here; start the shards by hand to parallelise). SUBSET: first 10 conditions per stratum, ~15 min.
#   OUT=<new dir> [SUBSET=<dir>] bash scripts/figures/figure2/score/ed_pbmc_wells/score.sh
source "$(dirname "$0")/../../../../eval/_env.sh"

# 1. The cache, the conditions, and the six control wells (H7-H12) swapped against each other
NS=$("$PY" -m src.paths '{data:parse}/eval_cache/normalization_raw_counts/geneset_63a547306aa0e6cd__n_2000__cell_type__well')
WD=("$PY" -u scripts/eval/reference.py well_draws --dataset parse --eval-cache "$NS"
    --conditions-from core_results/figure2/split_half/parse_bio/GENESIS.json --unit-cols donor cell_type
    --swap-col bc1_well --swap-values H7 H8 H9 H10 H11 H12 --device cpu --out-prefix "$OUT/work/pbmc_wells/well_draws")

# 2. Draw: four strata shards, then merge (the merge refuses unless every stratum is present). SUBSET is one limited run, tagged _limit10.
if [ -z "${SUBSET:-}" ]; then
    for i in 0 1 2 3; do run work/pbmc_wells/well_draws_rows_shard${i}of4.csv "${WD[@]}" --shard $i --n-shards 4; done
    run work/pbmc_wells/well_draws_summary.json "${WD[@]}" --n-shards 4 --merge
    place work/pbmc_wells/well_draws_summary.json figure2/pbmc_wells/well_draws_summary.json
    TAG=
else
    run work/pbmc_wells/well_draws_rows_limit10.csv "${WD[@]}" --limit 10
    TAG=_limit10
fi
place work/pbmc_wells/well_draws_rows$TAG.csv figure2/pbmc_wells/well_draws_rows.csv
place work/pbmc_wells/well_draws_energy$TAG.csv figure2/pbmc_wells/well_draws_energy.csv

# 3. The condition summary the draws were scored against is the run's own file, copied as it is
S=$(need_run "$ROOT/runs/parse/bio/main/parse_bio_well_conds100_5ep_donorholdout_seed0/260921_192149_*/eval/condition_summary.parquet")
run work/pbmc_wells/condition_summary.parquet cp "$S" "$OUT/work/pbmc_wells/condition_summary.parquet"
place work/pbmc_wells/condition_summary.parquet figure2/pbmc_wells/condition_summary.parquet
finish
