#!/usr/bin/env bash
# Extended Data (Fig 2 conditioning): Tahoe GENESIS trained on all four condition-input sets, scored on the bio split test conditions.
# In:   last.ckpt of four runs under runs/tahoe/bio/ablation/cond_input/ (table below)
# Out:  $OUT/core_results/figure2/split_half/tahoe_bio_input_ablation/CIA_{all,bio,pert,plate}.json
# Ships as-is (not rerun): nothing (the panel is Tahoe only).   Needs first: nothing (the plot reads Fig 2 main's tahoe_bio/reference.json).
# Cost: ~7 min per run (full), ~20 min for pert; SUBSET ~1 min each. Run on an H100.
#   GPU=0 OUT=<new dir> [SUBSET=<dir>] bash scripts/figures/figure2/score/ed_conditioning/score.sh
source "$(dirname "$0")/../../../../eval/_env.sh"
need_gpu

# 1. The four runs: <arm> <experiment dir under runs/tahoe/bio/ablation/cond_input> <run stamp>. Earlier stamps of an experiment are failed launches.
RUNS=(
    "all   cia_tahoe_bio_hbss100_all4_r0_seed0       260831_120928"
    "bio   cia_tahoe_bio_hbss100_plate_bio_r0_seed0  261006_183708"
    "pert  cia_tahoe_bio_hbss100_plate_pert_r0_seed0 261007_221043"
    "plate cia_tahoe_bio_hbss100_plate_r0_seed0      261006_112405"
)

# 2. SUBSET only: draw 50 test conditions once
if [ -n "${SUBSET:-}" ] && [ ! -e "$SUBSET/tahoe_bio.json" ]; then
    mkdir -p "$SUBSET"
    "$PY" scripts/eval/subset.py draw --split core_util_data/splits/tahoe/bio/tahoe_legacy_bio_holdout.yaml --n 50 --out "$SUBSET/tahoe_bio.json"
fi

# 3. Score each arm as Fig 2c scores GENESIS: compiled (the numbers depend on it), one generated draw per observed cell, seed 42, per-condition seeding
for row in "${RUNS[@]}"; do
    read -r arm exp stamp <<<"$row"
    run=$(need_run "runs/tahoe/bio/ablation/cond_input/$exp/${stamp}_*")
    run work/cia/$arm/metrics_summary.json \
        "$PY" -u scripts/eval/score.py genesis --compile --run-dir "$ROOT/$run" --checkpoint last --eval-split test \
        --model-eval-n-repeats 1 --metric-set extended \
        ${SUBSET:+--conditions-query "$("$PY" scripts/eval/subset.py query "$run" "$SUBSET/tahoe_bio.json")"} \
        --out-dir "$OUT/work/cia/$arm"
    place work/cia/$arm/metrics_summary.json figure2/split_half/tahoe_bio_input_ablation/CIA_$arm.json
done
finish
