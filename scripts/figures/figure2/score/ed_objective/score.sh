#!/usr/bin/env bash
# Extended Data (Fig 2 objective): GENESIS trained with the energy+variogram loss vs a set-level kernel (MMD) loss, both on 25% of the cells,
# scored against the split-half reference (bio test set) and against the experimental repeat (two-plate test pairs).
# In:   last.ckpt of  runs/tahoe/bio/ablation/mmd_loss/h4096x8192_rbfset_c7x256_32ep_…/261005_213507_…   (set-level MMD)
#                     runs/tahoe/bio/scaling/n_cells_025/h4096x8192_nrs256_vg05rs256_20ep_…/260918_151050_…   (GENESIS, 25% cells)
#       core_util_data/tahoe/eval_inputs/multi_plate_test_conditions.tsv   (the 545 two-plate test combinations)
# Out:  $OUT/core_results/figure2/loss_mmd/{split_half/{rbfset,genesis}_25cells.json, plate_pairs/{rbfset,genesis}_25cells.parquet}
# Ships as-is (not rerun): nothing.   Needs first: nothing.
# UNSEEDED (NEW_SAMPLE): plate_pairs/genesis_25cells.parquet (its published predictions were drawn unseeded;
# the rerun regenerates them, seeded per file).
# Cost: split-half ~11 + ~54 min, plate pairs ~1 h each (generate + score); SUBSET ~5 min. Run on an H100.
#   GPU=0 OUT=<new dir> [SUBSET=<dir>] bash scripts/figures/figure2/score/ed_objective/score.sh
source "$(dirname "$0")/../../../../eval/_env.sh"
need_gpu

# 1. The two runs and the plate-pair inputs. A run's stamp picks one of several launches of an experiment (crashed ones have no last.ckpt).
M_RUN=$(need_run "$ROOT/runs/tahoe/bio/ablation/mmd_loss/h4096x8192_rbfset_c7x256_32ep_state2k_tahoe_bio_25cell_seed0/261005_213507_*")
G_RUN=$(need_run "$ROOT/runs/tahoe/bio/scaling/n_cells_025/h4096x8192_nrs256_vg05rs256_20ep_state2k_tahoe_bio_25cell_seed0/260918_151050_*")
NS=$("$PY" -m src.paths '{data:tahoe}/eval_cache/normalization_raw_counts/geneset_c0d6420ffc0fa861__n_2000')
PAIRS=core_util_data/tahoe/eval_inputs/multi_plate_test_conditions.tsv

# 2. SUBSET only: 50 test conditions (split-half) and the first 10 plate pairs (a cut TSV is safe here: every pair is seeded by its own key)
if [ -n "${SUBSET:-}" ]; then
    [ -f "$SUBSET/tahoe_bio.json" ] || "$PY" scripts/eval/subset.py draw --split core_util_data/splits/tahoe/bio/tahoe_legacy_bio_holdout.yaml --n 50 --out "$SUBSET/tahoe_bio.json"
    mkdir -p "$OUT/work"; head -n 11 "$PAIRS" > "$OUT/work/pairs_subset.tsv"; PAIRS="$OUT/work/pairs_subset.tsv"
fi

# 3. Score each run on the split-half test set: compiled, one draw per observed cell, seed 42, per-condition seeding
for arm in rbfset:"$M_RUN" genesis:"$G_RUN"; do
    run work/split_half/${arm%%:*}_25cells/metrics_summary.json \
        "$PY" -u scripts/eval/score.py genesis --compile --run-dir "${arm#*:}" --checkpoint last --eval-split test \
        --model-eval-n-repeats 1 --metric-set extended \
        ${SUBSET:+--conditions-query "$("$PY" scripts/eval/subset.py query "${arm#*:}" "$SUBSET/tahoe_bio.json")"} \
        --out-dir "$OUT/work/split_half/${arm%%:*}_25cells"
    place work/split_half/${arm%%:*}_25cells/metrics_summary.json figure2/loss_mmd/split_half/${arm%%:*}_25cells.json
done

# 4. Plate pairs: generate each run's cells at the two plates' observed counts, then score the pair against the real
# replicate. Energy on a 2,048-cell draw, as published (the main Fig 2 table uses every cell).
for arm in rbfset:"$M_RUN" genesis:"$G_RUN"; do
    name=${arm%%:*}_25cells
    # each step's own output is its done-file; generate.py also resumes file by file
    run work/plate_pairs/$name/preds \
        "$PY" -u scripts/eval/generate.py replicates --dataset tahoe --run-dir "${arm#*:}" --conditions "$PAIRS" --eval-cache "$NS" \
        --out-pred "$OUT/work/plate_pairs/$name/preds" --device cuda --batch-size 2048
    run work/plate_pairs/$name.parquet \
        "$PY" -u scripts/eval/reference.py replicate_pairs --dataset tahoe --eval-cache "$NS" --conditions "$PAIRS" \
        --model $name --fmt npy --pred-dir "$OUT/work/plate_pairs/$name/preds" --report-space log1p_slice_med_rgs \
        --device cuda:0 --out "$OUT/work/plate_pairs/$name.parquet"
    place work/plate_pairs/$name.parquet figure2/loss_mmd/plate_pairs/$name.parquet
done
NEW_SAMPLE="figure2/loss_mmd/plate_pairs/genesis_25cells.parquet"
finish
