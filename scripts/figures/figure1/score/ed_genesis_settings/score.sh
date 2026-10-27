#!/usr/bin/env bash
# Extended Data (Fig 1 settings): the main Tahoe GENESIS retrained with one training setting changed, each scored on the test split.
# In:   last.ckpt of nine runs under runs/tahoe/bio/ (table below); their config_resolved.yaml
# Out:  $OUT/core_results/figure1/settings/<run>/{metrics_summary.json, config_resolved.yaml}
# Ships as-is (not rerun): nothing.   Needs first: nothing.
# UNSEEDED (NEW_SAMPLE): genesis, lambda0, lambda0.1, lambda0.3, lambda1 (published without per-condition seeding).
# Cost: ~9 min per run on an H100 (full, 4,211 conditions); SUBSET ~1 min per run. Run on an H100.
#   GPU=0 OUT=<new dir> [SUBSET=<dir>] bash scripts/figures/figure1/score/ed_genesis_settings/score.sh
source "$(dirname "$0")/../../../../eval/_env.sh"
need_gpu

# 1. The nine runs: <name> <experiment dir under runs/tahoe/bio> <run stamp>. The stamp picks one of several launches of an experiment.
RUNS=(
    "genesis   main/h4096x8192_nrs256_vg05rs256_5ep_state2k_tahoe_bio_100cond_seed0            260916_114938"
    "lambda0   ablation/variogram/h4096x8192_nrs256_vg00rs256_5ep_state2k_tahoe_bio_100cond_seed0 260909_220316"
    "lambda0.1 ablation/variogram/h4096x8192_nrs256_vg01rs256_5ep_state2k_tahoe_bio_100cond_seed0 260909_220316"
    "lambda0.3 ablation/variogram/h4096x8192_nrs256_vg03rs256_5ep_state2k_tahoe_bio_100cond_seed0 260916_114940"
    "lambda1   ablation/variogram/h4096x8192_nrs256_vg10rs256_5ep_state2k_tahoe_bio_100cond_seed0 261004_162151"
    "m16       ablation/n_repeats/h4096x8192_nrs16_vg05rs16_5ep_state2k_tahoe_bio_100cond_seed0   261005_235052"
    "m64       ablation/n_repeats/h4096x8192_nrs64_vg05rs64_5ep_state2k_tahoe_bio_100cond_seed0   261006_044718"
    "depth0    ablation/depth/hnone_nrs256_vg05rs256_5ep_state2k_tahoe_bio_100cond_seed0          261022_123317"
    "depth1    ablation/depth/h8192_nrs256_vg05rs256_5ep_state2k_tahoe_bio_100cond_seed0          261021_130248"
)

# 2. SUBSET only: draw 50 test conditions once; each run's --conditions-query is cut from it below
if [ -n "${SUBSET:-}" ] && [ ! -e "$SUBSET/tahoe_bio.json" ]; then
    mkdir -p "$SUBSET"
    "$PY" scripts/eval/subset.py draw --split core_util_data/splits/tahoe/bio/tahoe_legacy_bio_holdout.yaml --n 50 --out "$SUBSET/tahoe_bio.json"
fi

# 3. Rescore every run's last.ckpt with its own after-fit settings (read from each run's config_resolved.yaml and
# the published metrics_summary.json: model_eval_n_repeats 3, seed 64, model_eval_compile false). The score.py defaults
# (seed 42, compile on) are NOT those, so both are passed. metric-set extended = the published 53 metrics.
for row in "${RUNS[@]}"; do
    read -r name exp stamp <<<"$row"
    run=$(need_run "runs/tahoe/bio/$exp/${stamp}_*")  # a missing run fails here, loudly
    run work/settings/$name/metrics_summary.json \
        "$PY" -u scripts/eval/score.py genesis --run-dir "$ROOT/$run" --checkpoint last --eval-split test \
        --model-eval-n-repeats 3 --seed 64 --no-compile --metric-set extended \
        ${SUBSET:+--conditions-query "$("$PY" scripts/eval/subset.py query "$run" "$SUBSET/tahoe_bio.json")"} \
        --out-dir "$OUT/work/settings/$name"
    # 4. config_resolved.yaml is the run's own file, copied as it is
    run work/settings/$name/config_resolved.yaml cp "$run/config_resolved.yaml" "$OUT/work/settings/$name/config_resolved.yaml"
    place work/settings/$name/metrics_summary.json figure1/settings/$name/metrics_summary.json
    place work/settings/$name/config_resolved.yaml figure1/settings/$name/config_resolved.yaml
done
NEW_SAMPLE="figure1/settings/genesis/metrics_summary.json figure1/settings/lambda0/metrics_summary.json figure1/settings/lambda0.1/metrics_summary.json figure1/settings/lambda0.3/metrics_summary.json figure1/settings/lambda1/metrics_summary.json"
finish
