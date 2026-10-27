#!/usr/bin/env bash
# ED 13: a held-out compound that trained in k pool lines (a: 4 Latin rows x 3 entity seeds), and PBMC on fewer cells or conditions (b).
# In:   a: runs/tahoe/pert/scaling/overlap_celln10_fixed/entity_seed<s>/*_r<r>_es<s>/<ts>_*/checkpoints/last.ckpt (12);
#          core_util_data/splits/tahoe/pert/tahoe_overlap_pert_fixed/manifest.json (tracked; the published manifest is a byte copy).
#       b: runs/parse/bio/main/parse_bio_well_conds100_5ep_donorholdout_seed0/<ts>_*/ and
#          runs/parse/bio/scaling/{n_cells_025,n_cells_005,n_conds_025}/*donorholdout_seed0/<ts>_*/: checkpoints/last.ckpt
#          and their run.log (pbmc_runs.py, CPU, reads the 100% arm through the link runs/parse/bio/scaling/full_100).
# Out:  $OUT/core_results/figure3/{pert_overlap/{r{0-3}_es{0-2}.json, manifest.json}, pbmc_cells_conds/{full100,cells25,cells5,conds25}.json, pbmc_cells_conds/scaling_arms.csv}
# Ships as-is (not rerun): nothing.   Needs first: nothing.
# Cost: full ~5 h GPU (12 Tahoe pert + 4 PBMC evals); SUBSET: 50 conditions each. Run on an H100.
# Every JSON here is UNSEEDED (after-fit evals of the training runs): the rerun is a new sample, reported, not checked.
#   GPU=0 OUT=<new dir> [SUBSET=<dir>] bash scripts/figures/figure3/score/ed_scaling_pert_split/score.sh
source "$(dirname "$0")/../../../../eval/_env.sh"
need_gpu
NEW_SAMPLE="$(for r in 0 1 2 3; do for s in 0 1 2; do printf 'figure3/pert_overlap/r%d_es%d.json ' "$r" "$s"; done; done)"
NEW_SAMPLE+="figure3/pbmc_cells_conds/full100.json figure3/pbmc_cells_conds/cells25.json figure3/pbmc_cells_conds/cells5.json figure3/pbmc_cells_conds/conds25.json"

# 1. Resolve every checkpoint first, so a moved or missing run fails here and not hours into the queue.
#    run_of <glob>: the one run dir matching <glob> that holds a last.ckpt
run_of() { need_run "$1" checkpoints/last.ckpt; }
pert_run() { run_of "runs/tahoe/pert/scaling/overlap_celln10_fixed/entity_seed$2/*_r$1_es$2/2*"; }
PBMC=(  # <published stem>=<its run> (pbmc_runs.py: ARMS)
    'full100=runs/parse/bio/main/parse_bio_well_conds100_5ep_donorholdout_seed0/260921_192149_*'
    'cells25=runs/parse/bio/scaling/n_cells_025/parse_bio_well_cells025_20ep_donorholdout_seed0/260925_215259_*'
    'cells5=runs/parse/bio/scaling/n_cells_005/parse_bio_well_cells005_100ep_donorholdout_seed0/260926_014742_*'
    'conds25=runs/parse/bio/scaling/n_conds_025/parse_bio_well_conds025_20ep_donorholdout_seed0/260925_215300_*'
)
for r in 0 1 2 3; do for s in 0 1 2; do pert_run "$r" "$s" >/dev/null; done; done
for e in "${PBMC[@]}"; do run_of "${e#*=}" >/dev/null; done
FULL=$(run_of "${PBMC[0]#*=}")
# pbmc_runs.py names the 100% arm runs/parse/bio/scaling/full_100/..., as the published scaling_arms.csv records it:
# a link to runs/parse/bio/main
need_path runs/parse/bio/scaling/full_100

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
    stem=${e%%=*}; RUN=$(run_of "${e#*=}")
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
