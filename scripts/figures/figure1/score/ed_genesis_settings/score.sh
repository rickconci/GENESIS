#!/usr/bin/env bash
# Extended Data (Fig 1 settings): the main Tahoe GENESIS retrained with one training setting changed, each scored on the test split.
# In:   nine models (table below): models/tahoe and models/reproduce/figure1/settings/<name>; their config_resolved.yaml
# Out:  $OUT/core_results/figure1/settings/<run>/{metrics_summary.json, config_resolved.yaml}
# Ships as-is (not rerun): nothing.   Needs first: nothing.
# UNSEEDED (NEW_SAMPLE): genesis, lambda0, lambda0.1, lambda0.3, lambda1 (published without per-condition seeding).
# Cost: ~9 min per run on an H100 (full, 4,211 conditions); SUBSET ~1 min per run. Run on an H100.
#   GPU=0 OUT=<new dir> [SUBSET=<dir>] bash scripts/figures/figure1/score/ed_genesis_settings/score.sh
source "$(dirname "$0")/../../../../eval/_env.sh"
need_gpu

# 1. The nine models: <name> <model>. Figure 2's GENESIS (m = 256, λ = 0.5, two hidden layers) is the reference row;
# each other model changes one of those settings.
RUNS=(
    "genesis   tahoe"
    "lambda0   reproduce/figure1/settings/lambda0"
    "lambda0.1 reproduce/figure1/settings/lambda0.1"
    "lambda0.3 reproduce/figure1/settings/lambda0.3"
    "lambda1   reproduce/figure1/settings/lambda1"
    "m16       reproduce/figure1/settings/m16"
    "m64       reproduce/figure1/settings/m64"
    "depth0    reproduce/figure1/settings/depth0"
    "depth1    reproduce/figure1/settings/depth1"
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
    read -r name m <<<"$row"
    run=$(model "$m")  # a missing model fails here, loudly
    run work/settings/$name/metrics_summary.json \
        "$PY" -u scripts/eval/score.py genesis --run-dir "$run" --checkpoint last --eval-split test \
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
