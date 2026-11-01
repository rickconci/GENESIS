#!/usr/bin/env bash
# ED 13: a held-out compound that trained in k pool lines (a: 4 Latin rows x 3 entity seeds), and PBMC on fewer cells or conditions (b).
# In:   a: models/reproduce/figure3/pert_overlap/row<r>_seed<s> (12: Latin row r, entity seed s);
#          core_util_data/splits/tahoe/pert/tahoe_overlap_pert_fixed/manifest.json (tracked; the published manifest is a byte copy).
#       b: models/reproduce/figure3/pbmc_scaling/{full,cells25,cells5,conditions25} and their run.log (pbmc_runs.py, CPU).
# Out:  $OUT/core_results/figure3/{pert_overlap/{r{0-3}_es{0-2}.json, manifest.json}, pbmc_cells_conds/{full100,cells25,cells5,conds25}.json, pbmc_cells_conds/scaling_arms.csv}
# Ships as-is (not rerun): nothing.   Needs first: nothing.
# Cost: full ~5 h GPU (12 Tahoe pert + 4 PBMC evals); SUBSET: 50 conditions each. Run on an H100.
# Every JSON here is UNSEEDED (after-fit evals of the training runs): the rerun is a new sample, reported, not checked.
#   GPU=0 OUT=<new dir> [SUBSET=<dir>] bash scripts/figures/figure3/score/ed_scaling_pert_split/score.sh
source "$(dirname "$0")/../../../../eval/_env.sh"
need_gpu
NEW_SAMPLE="$(for r in 0 1 2 3; do for s in 0 1 2; do printf 'figure3/pert_overlap/r%d_es%d.json ' "$r" "$s"; done; done)"
NEW_SAMPLE+="figure3/pbmc_cells_conds/full100.json figure3/pbmc_cells_conds/cells25.json figure3/pbmc_cells_conds/cells5.json figure3/pbmc_cells_conds/conds25.json"

# 1. Resolve every model first, so a missing one fails here and not hours into the queue.
pert_run() { model "reproduce/figure3/pert_overlap/row$1_seed$2"; }
PBMC=(  # <published stem>=<its model under models/reproduce/figure3/pbmc_scaling/> (pbmc_runs.py: ARMS)
    'full100=full'
    'cells25=cells25'
    'cells5=cells5'
    'conds25=conditions25'
)
for r in 0 1 2 3; do for s in 0 1 2; do pert_run "$r" "$s" >/dev/null; done; done
for e in "${PBMC[@]}"; do model "reproduce/figure3/pbmc_scaling/${e#*=}" >/dev/null; done
FULL=$(model reproduce/figure3/pbmc_scaling/full)

# 2. (SUBSET) Freeze the samples once, drawn even under DRY_RUN (the queries below read them). The 12 pert splits
#    hold out one test set, so r0_es0's split draws for all; PBMC conditions are donor x cytokine x cell type x well.
[ -z "${SUBSET:-}" ] || [ -e "$SUBSET/tahoe_pert_overlap.json" ] || "$PY" scripts/eval/subset.py draw --n 50 --seed 0 \
    --split core_util_data/splits/tahoe/pert/tahoe_overlap_pert_fixed/entity_seed0/tahoe_overlap_pert_fixed_r0_es0.yaml \
    --out "$SUBSET/tahoe_pert_overlap.json"
[ -z "${SUBSET:-}" ] || [ -e "$SUBSET/parse_bio_well.json" ] || "$PY" scripts/eval/subset.py draw --n 50 --seed 0 \
    --split core_util_data/splits/parse/bio/parse_bio_well_split/parse_bio_well_hbss100_pbs_donorholdout_es0.yaml \
    --keys donor cytokine cell_type bc1_well --summary "$FULL/eval/condition_summary.parquet" --out "$SUBSET/parse_bio_well.json"

# 3. Score the 12 compound-overlap runs on their test set: an eval-only rerun of last.ckpt with the published
#    after-fit settings (K=1, extended metrics)
for r in 0 1 2 3; do for s in 0 1 2; do
    RUN=$(pert_run "$r" "$s")
    Q=${SUBSET:+$("$PY" scripts/eval/subset.py query "$RUN" "$SUBSET/tahoe_pert_overlap.json")}
    run "work/pert_overlap/r${r}_es$s/metrics_summary.json" \
        "$PY" -u scripts/eval/score.py genesis --run-dir "$RUN" --checkpoint last --eval-split test \
        --model-eval-n-repeats 1 --metric-set extended --gene-sidecars ${Q:+--conditions-query "$Q"} \
        --device cuda:0 --out-dir "$OUT/work/pert_overlap/r${r}_es$s"
    place "work/pert_overlap/r${r}_es$s/metrics_summary.json" "figure3/pert_overlap/r${r}_es$s.json"
done; done

# 4. The split manifest the extract reads each run's k from: the tracked file, copied
run work/pert_overlap/manifest.json \
    cp core_util_data/splits/tahoe/pert/tahoe_overlap_pert_fixed/manifest.json "$OUT/work/pert_overlap/manifest.json"
place work/pert_overlap/manifest.json figure3/pert_overlap/manifest.json

# 5. Score the 4 PBMC arms on the donor-holdout test set, same after-fit settings (K=1, extended metrics)
for e in "${PBMC[@]}"; do
    stem=${e%%=*}; RUN=$(model "reproduce/figure3/pbmc_scaling/${e#*=}")
    Q=${SUBSET:+$("$PY" scripts/eval/subset.py query "$RUN" "$SUBSET/parse_bio_well.json")}
    run "work/pbmc_cells_conds/$stem/metrics_summary.json" \
        "$PY" -u scripts/eval/score.py genesis --run-dir "$RUN" --checkpoint last --eval-split test \
        --model-eval-n-repeats 1 --metric-set extended --gene-sidecars ${Q:+--conditions-query "$Q"} \
        --device cuda:0 --out-dir "$OUT/work/pbmc_cells_conds/$stem"
    place "work/pbmc_cells_conds/$stem/metrics_summary.json" "figure3/pbmc_cells_conds/$stem.json"
done

# 6. Each PBMC arm's training cells, from its run.log (CPU; SUBSET runs it in full too)
run work/pbmc_cells_conds/scaling_arms.csv "$PY" -u scripts/figures/figure3/score/ed_scaling_pert_split/pbmc_runs.py --out "$OUT/work/pbmc_cells_conds"
place work/pbmc_cells_conds/scaling_arms.csv figure3/pbmc_cells_conds/scaling_arms.csv
finish
