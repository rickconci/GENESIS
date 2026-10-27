#!/usr/bin/env bash
# Extended Data (Fig 2 perturbation generalization): GENESIS, the four linear baselines, copy_closest and the split-half
# reference on the Tahoe and PBMC held-out-perturbation test sets; Tahoe's plate pairs against the experimental repeat; no response.
# In:   GENESIS tahoe_pert  runs/tahoe/pert/main/h4096x8192_…/261003_180534_…/checkpoints/last.ckpt   (seed 42)
#       GENESIS parse_pert  runs/parse/pert/main/parse_pert_well_fixed_r0_es0_5ep_seed0/261006_005520_…/checkpoints/last.ckpt (seed 0)
#       core_util_data/tahoe/eval_inputs/{two_plate_conditions.tsv, two_plate_conditions__min50.tsv, two_plate_counts.parquet, copy_closest_pert_neighbors.tsv}
#       the after-fit eval of runs/tahoe/pert/scaling/overlap_celln10_fixed/entity_seed0/…r0_es0/260930_075411_…
# Out:  $OUT/core_results/figure2/{split_half/{tahoe,parse}_pert_fixed/{GENESIS,Additive,LinRidge,PertMean,CtxMean,copy_closest,reference}.json,
#       plate_pairs/tahoe_pert_fixed/{pair_metrics__{genesis,copy_closest,additive,context_mean,linear_ridge,perturbation_mean}.parquet,census_r0_es0.json},
#       no_response/{tahoe,parse}_pert_fixed.json}
# Ships as-is (not rerun): STATE, CellFlow, PerturbDiff (split_half/*_pert_fixed/*.json, plate_pairs/tahoe_pert_fixed/pair_metrics__*.parquet)
# Needs first: nothing.
# Cost: full ~6 GPU-h (fits ~50 min, references ~1.7 h, plate pairs ~1.2 h); SUBSET is dominated by the fits (always full). Run on an H100.
#   GPU=0 OUT=<new dir> [SUBSET=<dir>] bash scripts/figures/figure2/score/ed_pert_generalization/score.sh
source "$(dirname "$0")/../../../../eval/_env.sh"
need_gpu

# 1. The two panels (plain paths: score.py and fit.py do not expand {data:...})
T_RUN=$(need_run "$ROOT/runs/tahoe/pert/main/h4096x8192_nrs256_vg05rs256_5ep_state2k_tahoe_pert_fixed_allcells_r0_es0/261003_180534_*")
P_RUN=$(need_run "$ROOT/runs/parse/pert/main/parse_pert_well_fixed_r0_es0_5ep_seed0/261006_005520_*")
T_DATA=$("$PY" -m src.paths '{data:tahoe}'); P_DATA=$("$PY" -m src.paths '{data:parse}')
T_SPLIT=core_util_data/splits/tahoe/pert/tahoe_overlap_pert_fixed/entity_seed0/tahoe_overlap_pert_fixed_r0_es0_allcells.yaml
P_SPLIT=core_util_data/splits/parse/pert/parse_overlap_pert_fixed/entity_seed0/parse_overlap_pert_fixed_r0_es0.yaml
T_NS=$T_DATA/eval_cache/normalization_raw_counts/geneset_c0d6420ffc0fa861__n_2000
P_NS=$P_DATA/eval_cache/normalization_raw_counts/geneset_63a547306aa0e6cd__n_2000__cell_type__well
T_GENES=core_util_data/tahoe/gene_subsets/STATE_HVG.txt
P_GENES=core_util_data/parse/gene_subsets/pbmc_real_selected_genes.pkl
T_SUMMARY=core_util_data/tahoe/metadata/quadruplet_summary.parquet
P_SUMMARY=$(need_run "$ROOT/runs/parse/bio/main/parse_bio_well_conds100_5ep_donorholdout_seed0/260921_192149_*/eval/condition_summary.parquet")
T=(--dataset tahoe --split-yaml "$T_SPLIT" --condition-summary "$T_SUMMARY" --eval-cache-namespace "$T_NS" --gene-names "$T_GENES")
P=(--dataset parse --condition-variables donor cytokine cell_type bc1_well --control-match-keys donor cell_type
   --split-yaml "$P_SPLIT" --condition-summary "$P_SUMMARY" --eval-cache-namespace "$P_NS" --gene-names "$P_GENES")
EI=core_util_data/tahoe/eval_inputs
PAIRS=$EI/two_plate_conditions.tsv; PAIRS50=$EI/two_plate_conditions__min50.tsv

# 2. SUBSET only: 50 conditions per panel (a query names rows of ONE summary: GENESIS reads its run's, the other arms the panel's;
# reference.py has no query flag, so it gets a cut summary), and the first 10 plate pairs (every pair is seeded by its own key)
if [ -n "${SUBSET:-}" ]; then
    [ -f "$SUBSET/tahoe_pert_fixed.json" ] || "$PY" scripts/eval/subset.py draw --split "$T_SPLIT" --n 50 --out "$SUBSET/tahoe_pert_fixed.json"
    [ -f "$SUBSET/parse_pert_fixed.json" ] || "$PY" scripts/eval/subset.py draw --split "$P_SPLIT" --n 50 \
        --keys donor cytokine cell_type bc1_well --summary "$P_SUMMARY" --out "$SUBSET/parse_pert_fixed.json"
    TQ_GEN=$("$PY" scripts/eval/subset.py query "$T_RUN" "$SUBSET/tahoe_pert_fixed.json")
    PQ_GEN=$("$PY" scripts/eval/subset.py query "$P_RUN" "$SUBSET/parse_pert_fixed.json")
    TQ=$("$PY" scripts/eval/subset.py query "$T_SUMMARY" "$SUBSET/tahoe_pert_fixed.json")
    PQ=$("$PY" scripts/eval/subset.py query "$P_SUMMARY" "$SUBSET/parse_pert_fixed.json")
    mkdir -p "$OUT/work/subset"
    run work/subset/tahoe_pert_fixed.parquet "$PY" scripts/eval/subset.py summary "$SUBSET/tahoe_pert_fixed.json" "$OUT/work/subset/tahoe_pert_fixed.parquet"
    run work/subset/parse_pert_fixed.parquet "$PY" scripts/eval/subset.py summary "$SUBSET/parse_pert_fixed.json" "$OUT/work/subset/parse_pert_fixed.parquet"
    head -n 11 "$PAIRS" > "$OUT/work/pairs.tsv"; head -n 11 "$PAIRS50" > "$OUT/work/pairs50.tsv"; PAIRS=$OUT/work/pairs.tsv; PAIRS50=$OUT/work/pairs50.tsv
fi

# 3. Fit the four baselines on every training cell, in log1p(1e4 x / L_full). Never subset: the fit is deterministic on the whole split.
run work/baselines/tahoe/fit_summary.json "$PY" -u scripts/baselines/fit.py "${T[@]}" --fit-cells 0 --fit-space log1p_cp10k_full --out-dir "$OUT/work/baselines/tahoe"
run work/baselines/parse/fit_summary.json "$PY" -u scripts/baselines/fit.py "${P[@]}" --fit-cells 0 --fit-space log1p_cp10k_full --out-dir "$OUT/work/baselines/parse"

# 4. Score GENESIS compiled as published; noise is seeded per (condition, repeat). PBMC persists predictions: that is what makes the
# scorer skip the one condition with no control cells (Donor5 x Plasmablast) instead of stopping.
run work/split_half/tahoe/GENESIS/metrics_summary.json \
    "$PY" -u scripts/eval/score.py genesis --compile --run-dir "$T_RUN" --checkpoint last --eval-split test \
    --model-eval-n-repeats 1 --seed 42 --metric-set extended ${SUBSET:+--conditions-query "$TQ_GEN"} --out-dir "$OUT/work/split_half/tahoe/GENESIS"
run work/split_half/parse/GENESIS/metrics_summary.json \
    "$PY" -u scripts/eval/score.py genesis --compile --run-dir "$P_RUN" --checkpoint last --eval-split test \
    --model-eval-n-repeats 1 --seed 0 --persist-predictions --metric-set extended ${SUBSET:+--conditions-query "$PQ_GEN"} --out-dir "$OUT/work/split_half/parse/GENESIS"

# 5. Score the baselines: each condition's real matched control cells plus the fitted delta
for arm in Additive:additive LinRidge:linear_ridge PertMean:perturbation_mean CtxMean:context_mean; do
    run "work/split_half/tahoe/${arm%:*}/metrics_summary.json" \
        "$PY" -u scripts/eval/score.py baseline "${T[@]}" --artifact "$OUT/work/baselines/tahoe/${arm#*:}.npz" \
        --seed 42 --metric-set extended ${SUBSET:+--conditions-query "$TQ"} --out-dir "$OUT/work/split_half/tahoe/${arm%:*}"
    run "work/split_half/parse/${arm%:*}/metrics_summary.json" \
        "$PY" -u scripts/eval/score.py baseline "${P[@]}" --artifact "$OUT/work/baselines/parse/${arm#*:}.npz" \
        --seed 0 --metric-set extended ${SUBSET:+--conditions-query "$PQ"} --out-dir "$OUT/work/split_half/parse/${arm%:*}"
done

# 6. Score copy_closest: the closest training context's own cells. Deltas are rebuilt from the HVG cache (--build-deltas, deterministic).
run work/split_half/tahoe/copy_closest/metrics_summary.json \
    "$PY" -u scripts/eval/score.py copy_neighbor "${T[@]}" --metric cosine --build-deltas \
    --deltas "$OUT/work/split_half/tahoe/copy_closest/deltas.npz" --train-cache "$T_DATA/hvg_cache/geneset_c0d6420ffc0fa861__n_2000" \
    --plate-matched --own-controls --seed 0 --metric-set extended ${SUBSET:+--conditions-query "$TQ"} --out-dir "$OUT/work/split_half/tahoe/copy_closest"
run work/split_half/parse/copy_closest/metrics_summary.json \
    "$PY" -u scripts/eval/score.py copy_neighbor "${P[@]}" --metric cosine --build-deltas \
    --deltas "$OUT/work/split_half/parse/copy_closest/deltas.npz" --train-cache "$P_DATA/hvg_cache/geneset_63a547306aa0e6cd__n_2000" \
    --own-controls --seed 0 --metric-set extended ${SUBSET:+--conditions-query "$PQ"} --out-dir "$OUT/work/split_half/parse/copy_closest"

# 7. Score the split-half reference (truth only, 20 redraws). --metric-set extended: the published files hold every key.
# SUBSET's --condition-summary comes later on the line, and the later one wins.
T_REF=work/reference/tahoe/geneset_c0d6420ffc0fa861__n_2000__norm_raw_counts/nr_20__splitcfg_9484594cb189159a__mincells_4__test/noise_ceiling.json
P_REF=work/reference/parse/geneset_63a547306aa0e6cd__n_2000__norm_raw_counts/nr_20__splitcfg_f9489b7b854c457e__mincells_4__test/noise_ceiling.json
run "$T_REF" "$PY" -u scripts/eval/reference.py split_half --dataset tahoe --split-config "$T_SPLIT" --eval-cache "$T_NS" \
    --gene-names "$T_GENES" --condition-summary "$T_SUMMARY" --n-repeats 20 --seed 42 --min-cells 4 --metric-set extended \
    ${SUBSET:+--condition-summary "$OUT/work/subset/tahoe_pert_fixed.parquet"} --out-root "$OUT/work/reference/tahoe"
run "$P_REF" "$PY" -u scripts/eval/reference.py split_half --dataset parse --split-config "$P_SPLIT" --eval-cache "$P_NS" \
    --gene-names "$P_GENES" --condition-variables donor cytokine cell_type bc1_well --control-match-keys donor cell_type \
    --condition-summary "$P_SUMMARY" --n-repeats 20 --seed 42 --min-cells 4 --metric-set extended \
    ${SUBSET:+--condition-summary "$OUT/work/subset/parse_pert_fixed.parquet"} --out-root "$OUT/work/reference/parse"

# 8. Place the split-half files
for panel in tahoe parse; do
    for arm in GENESIS Additive LinRidge PertMean CtxMean copy_closest; do
        place "work/split_half/$panel/$arm/metrics_summary.json" "figure2/split_half/${panel}_pert_fixed/$arm.json"
    done
done
place "$T_REF" figure2/split_half/tahoe_pert_fixed/reference.json
place "$P_REF" figure2/split_half/parse_pert_fixed/reference.json

# 9. Plate pairs against the experimental repeat (Tahoe): GENESIS cells at each plate's observed count, scored on every two-plate
# test pair with every cell in the energy (--energy-all-cells), in the depth-scaled log space. The baselines read the pair list and the
# seeded truth-truth rows from GENESIS's table, so GENESIS goes first; copy_closest scores the pairs with >= 50 cells per plate
# (its neighbour table has no pick for three stray small plates). SUBSET generates from the cut TSV, a full run from the frozen counts.
SIZES=(--counts-from "$EI/two_plate_counts.parquet"); [ -z "${SUBSET:-}" ] || SIZES=(--conditions "$PAIRS" --eval-cache "$T_NS")
PP=work/plate_pairs
run $PP/preds \
    "$PY" -u scripts/eval/generate.py replicates --dataset tahoe --run-dir "$T_RUN" "${SIZES[@]}" --out-pred "$OUT/$PP/preds" --device cuda --batch-size 2048
run $PP/model_vs_batch_gap__genesis.parquet \
    "$PY" -u scripts/eval/reference.py replicate_pairs --dataset tahoe --eval-cache "$T_NS" --conditions "$PAIRS" --model genesis --fmt npy \
    --pred-dir "$OUT/$PP/preds" --energy-all-cells --report-space log1p_slice_med_rgs --device cuda:0 --out "$OUT/$PP/model_vs_batch_gap__genesis.parquet"
run $PP/model_vs_batch_gap__additive.parquet \
    "$PY" -u scripts/eval/reference.py replicate_pairs_baselines --results "$OUT/$PP" --baseline-root "$OUT/work/baselines/tahoe" \
    --check-pred-root "$OUT/$PP/preds" --energy-all-cells --report-space log1p_slice_med_rgs --device cuda:0
run $PP/model_vs_batch_gap__copy_closest.parquet \
    "$PY" -u scripts/eval/reference.py replicate_pairs --dataset tahoe --eval-cache "$T_NS" --conditions "$PAIRS50" --model copy_closest --fmt copy \
    --copy-ctrl own --neighbor-tsv "$EI/copy_closest_pert_neighbors.tsv" --energy-all-cells --report-space log1p_slice_med_rgs --device cuda:0 \
    --out "$OUT/$PP/model_vs_batch_gap__copy_closest.parquet"
for m in genesis copy_closest additive context_mean linear_ridge perturbation_mean; do
    place $PP/model_vs_batch_gap__$m.parquet figure2/plate_pairs/tahoe_pert_fixed/pair_metrics__$m.parquet
done
# the census is the after-fit eval of the overlap run r0_es0 (used as truth only), copied as it is
CENSUS=$(need_run "$ROOT/runs/tahoe/pert/scaling/overlap_celln10_fixed/entity_seed0/nogroup_state2k_vg05_nrs96_15ep_tahoe_pert_fixed_celln10_r0_es0/260930_075411_*/eval/after_fit/test/metrics_summary.json")
run $PP/census_r0_es0.json cp "$CENSUS" "$OUT/$PP/census_r0_es0.json"
place $PP/census_r0_es0.json figure2/plate_pairs/tahoe_pert_fixed/census_r0_es0.json

# 10. Score no response where an arm failed, LAST because it reads every arm's file: the conditions the reference scores minus those an arm
# scored (the shipped comparators too), with the failed arms' seed (0). A SUBSET run's arms hold 50 rows, so there the published files name the failures.
ARMS=$OUT/core_results/figure2/split_half; [ -z "${SUBSET:-}" ] || ARMS=core_results/figure2/split_half
SHIPPED=core_results/figure2/split_half
for panel in tahoe parse; do
    S=$T_SUMMARY; [ "$panel" = tahoe ] || S=$P_SUMMARY
    run work/no_response/${panel}_failed.parquet \
        "$PY" scripts/eval/subset.py failed "$ARMS/${panel}_pert_fixed/reference.json" "$ARMS/${panel}_pert_fixed/"{GENESIS,Additive,LinRidge,PertMean,CtxMean,copy_closest}.json \
        "$SHIPPED/${panel}_pert_fixed/"{STATE,CellFlow,PerturbDiff}.json --summary "$S" --out "$OUT/work/no_response/${panel}_failed.parquet"
done
run work/no_response/tahoe/metrics_summary.json "$PY" -u scripts/eval/score.py no_response "${T[@]}" \
    --condition-summary "$OUT/work/no_response/tahoe_failed.parquet" --seed 0 --metric-set extended --device cpu --out-dir "$OUT/work/no_response/tahoe"
run work/no_response/parse/metrics_summary.json "$PY" -u scripts/eval/score.py no_response "${P[@]}" \
    --condition-summary "$OUT/work/no_response/parse_failed.parquet" --seed 0 --metric-set extended --device cpu --out-dir "$OUT/work/no_response/parse"
place work/no_response/tahoe/metrics_summary.json figure2/no_response/tahoe_pert_fixed.json
place work/no_response/parse/metrics_summary.json figure2/no_response/parse_pert_fixed.json
finish
