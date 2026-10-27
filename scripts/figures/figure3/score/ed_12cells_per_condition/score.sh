#!/usr/bin/env bash
# ED 12: the uniform-draw cells arms on Figure 2c's conditions (c), and GENESIS trained on 25 cells per combination (a-b).
# In:   uniform arms: runs/tahoe/bio/scaling/{n_conds_025: 260918_150610, n_conds_005: 260920_164413, n_cells_001: 261016_085931,
#       n_cells_005: 260920_162654, n_cells_025: 260918_151050}/checkpoints/last.ckpt (+ run.log, split_cache: training_runs.py);
#       25 per combination: runs/tahoe/bio/scaling/n_cells_percond25/…/261012_220500_*/checkpoints/last.ckpt, and Figure 2's
#       GENESIS (runs/tahoe/bio/main/…/260916_114938_*) for train614; core_util_data/tahoe/eval_inputs/train614_conds.tsv
#       (FROZEN: the 3,941 plate-6/14 training combinations; a byte copy of the published list; no script here builds it);
#       core_deliverables/data/figure2/main/plate_pairs/ (Figure 2a's 530 pairs: the test_cross cohort).
# Out:  $OUT/core_results/figure3/{uniform_cells/*.json (6), scaling_arms_uniform_cells.csv, shallow_pc25/scores/*.parquet,
#       shallow_pc25/train614/{conds_614.tsv, pairs__full.parquet, pairs__pc25.parquet}}
# Ships as-is (not rerun): nothing.   Needs first: figure2/score/ed_objective/score.sh with the same OUT (uniform_cells/genesis_25cells.json
# IS its split_half/genesis_25cells.json), and Figure 2's extract (the test_cross pairs).
# Cost: full ~1 day GPU (percond.py 1.6-1.8 h per same_plate/test shard); SUBSET: 50 conditions, 10 train614 combinations,
# no percond scoring (see 6). Run on an H100.
# genesis_5cells.json and genesis_25cells_k3.json are UNSEEDED (after-fit evals): new samples, reported, not checked.
#   GPU=0 OUT=<new dir> [SUBSET=<dir>] bash scripts/figures/figure3/score/ed_12cells_per_condition/score.sh
source "$(dirname "$0")/../../../../eval/_env.sh"
need_gpu
NEW_SAMPLE="figure3/uniform_cells/genesis_5cells.json figure3/uniform_cells/genesis_25cells_k3.json"
NS='{data:tahoe}/eval_cache/normalization_raw_counts/geneset_c0d6420ffc0fa861__n_2000'

# 1. Resolve every checkpoint first, so a moved or missing run fails here and not hours into the queue.
#    run_of <glob>: the one run dir matching <glob> that holds a last.ckpt
run_of() { need_run "$1" checkpoints/last.ckpt; }
UNIFORM=(  # <published stem>=<K model repeats>=<its run under runs/tahoe/bio/scaling/>; K as each published file records
    'genesis_25conds=1=n_conds_025/*_20ep_*_seed0/260918_150610_*'
    'genesis_5conds=1=n_conds_005/*_100ep_*_seed0/260920_164413_*'
    'genesis_1cells=3=n_cells_001/*_seed0/261016_085931_*'
    'genesis_5cells=3=n_cells_005/*_100ep_*_seed0/260920_162654_*'
    'genesis_25cells_k3=3=n_cells_025/*_seed0/260918_151050_*'
)
for e in "${UNIFORM[@]}"; do run_of "runs/tahoe/bio/scaling/${e##*=}" >/dev/null; done
PC25=$(run_of 'runs/tahoe/bio/scaling/n_cells_percond25/*_seed0/261012_220500_*')
FULL=$(run_of 'runs/tahoe/bio/main/*_100cond_seed0/260916_114938_*')

# 2. (SUBSET) Freeze the sample once: 50 of Figure 2c's test conditions, the same file figure2/score/main/score.sh draws.
#    Drawn even under DRY_RUN: the queries below read it.
[ -z "${SUBSET:-}" ] || [ -e "$SUBSET/tahoe_bio.json" ] || "$PY" scripts/eval/subset.py draw \
    --split core_util_data/splits/tahoe/bio/tahoe_legacy_bio_holdout.yaml --n 50 --seed 0 --out "$SUBSET/tahoe_bio.json"

# 3. Score the uniform-draw arms on Figure 2c's split-half test set (last.ckpt, extended metrics). 5% and 1% cells are
#    published at K=3 only, and 25% cells is scored at K=3 too, to show K barely moves a mean.
for e in "${UNIFORM[@]}"; do
    stem=${e%%=*}; k=${e#*=}; k=${k%%=*}; RUN=$(run_of "runs/tahoe/bio/scaling/${e##*=}")
    Q=${SUBSET:+$("$PY" scripts/eval/subset.py query "$RUN" "$SUBSET/tahoe_bio.json")}
    run "work/uniform_cells/$stem/metrics_summary.json" \
        "$PY" -u scripts/eval/score.py genesis --run-dir "$RUN" --checkpoint last --eval-split test \
        --model-eval-n-repeats "$k" --metric-set extended --gene-sidecars ${Q:+--conditions-query "$Q"} \
        --device cuda:0 --out-dir "$OUT/work/uniform_cells/$stem"
    place "work/uniform_cells/$stem/metrics_summary.json" "figure3/uniform_cells/$stem.json"
done
# 25% cells at K=1 is ED_objective's genesis_25cells (same checkpoint, same settings): that driver's file, not a second GPU run
place core_results/figure2/loss_mmd/split_half/genesis_25cells.json figure3/uniform_cells/genesis_25cells.json

# 4. Count each uniform arm's training cells and conditions (CPU; deterministic, so SUBSET runs it in full too)
run work/training_runs/scaling_arms_uniform_cells.csv \
    "$PY" -u scripts/figures/figure3/score/training_runs.py --cells-arm uniform --out "$OUT/work/training_runs"
place work/training_runs/scaling_arms_uniform_cells.csv figure3/scaling_arms_uniform_cells.csv

# 5. train614: both models (all cells, 25 per combination) on the plate-6/14 training combinations, each plate a
#    replicate of the other. SUBSET takes the first 10 combinations: generate.py seeds every file by its own name, so a
#    cut list redraws nothing.
CONDS=core_util_data/tahoe/eval_inputs/train614_conds.tsv
mkdir -p "$OUT/work/train614"
[ -z "${SUBSET:-}" ] || { head -n 11 "$CONDS" > "$OUT/work/train614/conds_subset.tsv"; CONDS=$OUT/work/train614/conds_subset.tsv; }
for m in full pc25; do
    [ "$m" = full ] && RUN=$FULL || RUN=$PC25
    # each step's own output is its done-file; generate.py also skips the files it already wrote
    run "work/train614/pred/$m" \
        "$PY" -u scripts/eval/generate.py replicates --dataset tahoe --run-dir "$RUN" --checkpoint last \
        --conditions "$CONDS" --eval-cache "$NS" --out-pred "$OUT/work/train614/pred/$m" --device cuda:0
    run "work/train614/pairs__$m.parquet" \
        "$PY" -u scripts/eval/reference.py replicate_pairs --dataset tahoe --eval-cache "$NS" \
        --conditions "$CONDS" --model "genesis_$m" --fmt npy --pred-dir "$OUT/work/train614/pred/$m" \
        --energy-all-cells --report-space log1p_slice_med_rgs --device cuda:0 --out "$OUT/work/train614/pairs__$m.parquet"
    place "work/train614/pairs__$m.parquet" "figure3/shallow_pc25/train614/pairs__$m.parquet"
done
# (full run only) the combination list itself: a subset run scored a cut of it
if [ -z "${SUBSET:-}" ]; then
    run work/train614/conds_614.tsv cp "$CONDS" "$OUT/work/train614/conds_614.tsv"
    place work/train614/conds_614.tsv figure3/shallow_pc25/train614/conds_614.tsv
fi

# 6. (full run only) Score the 25-per-combination run against n true cells, cohort by cohort. Not under SUBSET: its items
#    are percond.py's own cohorts, not test conditions, and --limit writes differently named, shorter shards with no
#    published twin. Each published cohort is regenerated as one clean shard cover; the published set also holds
#    overlapping partial shards (cross 0of4/2of4, cross_deep 0of5-4of5), whose rows the extract collapses as duplicates.
export SHALLOW_ARM=pc25 SHALLOW_OUT="$OUT/work/shallow_pc25"
if [ -z "${SUBSET:-}" ]; then
    run work/shallow_pc25/rows.npz "$PY" -u scripts/figures/figure3/score/ed_12cells_per_condition/shallow/cohort.py
    run work/shallow_pc25/percond/items.parquet \
        "$PY" -u scripts/figures/figure3/score/ed_12cells_per_condition/shallow/percond.py --cohort-only --pairs-data core_deliverables/data/figure2/main
    for spec in same_plate:4 test:2 same_plate_pool:3 test_pool:2 cross_pooled:2 cross_deep:2 cross:2:e test_cross:1:e; do
        IFS=: read -r c n e <<<"$spec"   # e: energy on every cell (a whole other plate is the truth)
        dir=percond/scores${e:+_energy_all_cells}
        for ((k = 0; k < n; k++)); do
            run "work/shallow_pc25/$dir/${c}__shard${k}of$n.parquet" \
                "$PY" -u scripts/figures/figure3/score/ed_12cells_per_condition/shallow/percond.py --cohort "$c" --shard "$k" --n-shards "$n" \
                ${e:+--energy-all-cells} --device cuda:0
            place "work/shallow_pc25/$dir/${c}__shard${k}of$n.parquet" "figure3/shallow_pc25/scores/${c}__shard${k}of$n.parquet"
        done
    done
fi
finish
