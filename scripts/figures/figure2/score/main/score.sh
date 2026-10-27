#!/usr/bin/env bash
# Figure 2: (c) GENESIS, the four linear baselines, copy_closest and the split-half reference on the Tahoe-100M and PBMC
# biological-context test sets, plus each condition's own controls ("no response") where an arm failed; (a/b) the
# Tahoe arms against the experimental repeat, one row per pair of plates of a multi-plate test condition.
# In:   GENESIS tahoe_bio  runs/tahoe/bio/main/h4096x8192_…/260916_114938_…/checkpoints/last.ckpt   (seed 42)
#       GENESIS parse_bio  runs/parse/bio/main/parse_bio_cytokines_5ep_seed0/261006_020118_…/checkpoints/last.ckpt (seed 0)
#       The baselines are fitted and copy_closest's deltas rebuilt here, from the eval and HVG caches named below.
#       core_util_data/tahoe/eval_inputs/{multi_plate_test_conditions.tsv (the 545 multi-plate test conditions,
#       frozen), copy_closest_line__cosine_v2.tsv (one neighbour line per held-out line, score.py copy_lines)}
# Out:  $OUT/core_results/figure2/split_half/{tahoe_bio,parse_bio}/{GENESIS,Additive,LinRidge,PertMean,CtxMean,
#       copy_closest,reference}.json, $OUT/core_results/figure2/no_response/{tahoe_bio,parse_bio}.json and
#       $OUT/core_results/figure2/plate_pairs/tahoe_bio/pair_metrics__{genesis,copy_closest,additive,context_mean,
#       linear_ridge,perturbation_mean}.parquet
# Ships as-is (not rerun): core_results/figure2/split_half/{tahoe_bio,parse_bio}/{STATE,CellFlow,PerturbDiff}.json,
#       core_results/figure2/plate_pairs/tahoe_bio/pair_metrics__{state,cellflow,perturbdiff}.parquet
# Needs first: nothing.   Also feeds: ED_metrics, ED_split_coverage; Figure 3 and ED_12cells read these outputs.
# Cost: full ~6.5 GPU-h (references ~1.75 h, baseline scoring ~1.8 h, plate pairs ~1.5 h, GENESIS ~35 min, fits
#       ~35 min); SUBSET ~19 min on an H100, the fits (~7 min) always in full. GPU: GENESIS, copy_closest and the
#       references were published on an H100: check them on an H100 (an A100 drifts ~4e-5).
#       The baselines were published on an A100: check them on an A100; their fits are bit-identical on either.
#   GPU=0 OUT=<new dir> [SUBSET=<dir of frozen subsets>] bash scripts/figures/figure2/score/main/score.sh
source "$(dirname "$0")/../../../../eval/_env.sh"
need_gpu

# The two panels. score.py and fit.py take plain paths, so {data:tahoe} / {data:parse} (configs/roots.yaml) are spelled
# out. Tahoe's summary is the split policy's default, named so a query and the scorer read one file; PBMC's is the
# per-well summary of the parse_bio_well run (conditions donor x cytokine x cell_type x bc1_well).
T_RUN=$(need_run "$ROOT/runs/tahoe/bio/main/h4096x8192_nrs256_vg05rs256_5ep_state2k_tahoe_bio_100cond_seed0/260916_114938_cur-between_epoch-tp458752-bs1792xnrm256_nre256_lr0.0009_wd0.0001_biohold_condmlp-D256-h4096x8192_cond-plate_mlpout-relu_vg0.5p0.5np6k_log1p_s64_h4096x8192_nrs256_vg05rs256_5ep_state2k_tahoe_bio_100cond_seed0")
P_RUN=$(need_run "$ROOT/runs/parse/bio/main/parse_bio_cytokines_5ep_seed0/261006_020118_cur-between_epoch-tp229376-bs896xnrm256_nre256_lr0.0009_wd0.0001_parse_cytoki_condmlp-D256-h4096x8192_cond_mlpout-relu_vg0.5p0.5np6k_log1p_s64_parse_bio_cytokines_5ep_seed0")
T_DATA=$("$PY" -m src.paths '{data:tahoe}')     # configs/roots.yaml: this machine's data roots
P_DATA=$("$PY" -m src.paths '{data:parse}')
T_SPLIT=core_util_data/splits/tahoe/bio/tahoe_legacy_bio_holdout.yaml
P_SPLIT=core_util_data/splits/parse/bio/parse_bio_well_split/parse_bio_well_hbss100_pbs_donorholdout_es0.yaml
T_NS=$T_DATA/eval_cache/normalization_raw_counts/geneset_c0d6420ffc0fa861__n_2000
P_NS=$P_DATA/eval_cache/normalization_raw_counts/geneset_63a547306aa0e6cd__n_2000__cell_type__well
T_GENES=core_util_data/tahoe/gene_subsets/STATE_HVG.txt
P_GENES=core_util_data/parse/gene_subsets/pbmc_real_selected_genes.pkl
T_SUMMARY=core_util_data/tahoe/metadata/quadruplet_summary.parquet
P_SUMMARY=runs/parse/bio/main/parse_bio_well_conds100_5ep_donorholdout_seed0/260921_192149_cur-between_epoch-tp458752-bs1792xnrm256_nre256_lr0.0009_wd0.0001_parse_bio_we_condmlp-D256-h4096x8192_cond_mlpout-relu_vg0.5p0.5np6k_log1p_s64_parse_bio_well_conds100_5ep_donorholdout_seed0/eval/condition_summary.parquet
T=(--dataset tahoe --split-yaml "$T_SPLIT" --condition-summary "$T_SUMMARY" --eval-cache-namespace "$T_NS" --gene-names "$T_GENES")
PAIRS=core_util_data/tahoe/eval_inputs/multi_plate_test_conditions.tsv
NEIGHBORS=core_util_data/tahoe/eval_inputs/copy_closest_line__cosine_v2.tsv
P=(--dataset parse --condition-variables donor cytokine cell_type bc1_well --control-match-keys donor cell_type
   --split-yaml "$P_SPLIT" --condition-summary "$P_SUMMARY" --eval-cache-namespace "$P_NS" --gene-names "$P_GENES")

# 1. Freeze the SUBSET: 50 conditions per panel and 10 plate-pair conditions, drawn once (PBMC's on the panel's own keys
# and summary). A query names
# rows of ONE summary file: GENESIS reads its run's own, every other score.py arm the panel's (copy_neighbor needs the
# whole summary for its training arms, so a query, not a cut summary); reference.py has no query flag, so a cut summary.
[ -z "${SUBSET:-}" ] || [ -f "$SUBSET/tahoe_bio.json" ] || "$PY" scripts/eval/subset.py draw --split "$T_SPLIT" --n 50 --seed 0 --out "$SUBSET/tahoe_bio.json"
[ -z "${SUBSET:-}" ] || [ -f "$SUBSET/parse_bio.json" ] || "$PY" scripts/eval/subset.py draw --split "$P_SPLIT" --n 50 --seed 0 \
    --keys donor cytokine cell_type bc1_well --summary "$P_SUMMARY" --out "$SUBSET/parse_bio.json"
TQ_GEN=${SUBSET:+$("$PY" scripts/eval/subset.py query "$T_RUN" "$SUBSET/tahoe_bio.json")}
PQ_GEN=${SUBSET:+$("$PY" scripts/eval/subset.py query "$P_RUN" "$SUBSET/parse_bio.json")}
TQ=${SUBSET:+$("$PY" scripts/eval/subset.py query "$T_SUMMARY" "$SUBSET/tahoe_bio.json")}
PQ=${SUBSET:+$("$PY" scripts/eval/subset.py query "$P_SUMMARY" "$SUBSET/parse_bio.json")}
[ -z "${SUBSET:-}" ] || run work/subset/tahoe_bio.parquet "$PY" scripts/eval/subset.py summary "$SUBSET/tahoe_bio.json" "$OUT/work/subset/tahoe_bio.parquet"
[ -z "${SUBSET:-}" ] || run work/subset/parse_bio.parquet "$PY" scripts/eval/subset.py summary "$SUBSET/parse_bio.json" "$OUT/work/subset/parse_bio.parquet"
[ -z "${SUBSET:-}" ] || [ -f "$SUBSET/tahoe_bio_pairs.json" ] || "$PY" scripts/eval/subset.py draw --split "$T_SPLIT" --n 50 --seed 0 \
    --pairs "$PAIRS" --n-pairs 10 --out "$SUBSET/tahoe_bio_pairs.json"
[ -z "${SUBSET:-}" ] || run work/subset/tahoe_bio_pairs.tsv "$PY" scripts/eval/subset.py pairs "$SUBSET/tahoe_bio_pairs.json" "$OUT/work/subset/tahoe_bio_pairs.tsv"

# 2. Fit the four baselines on every training cell, in log1p(1e4 x / L_full) (~35 min Tahoe, ~1.3 min PBMC; a GPU PCA).
# Never subset: the fit is deterministic on the whole training split, whichever conditions are scored.
run work/baselines/tahoe_bio/fit_summary.json \
    "$PY" -u scripts/baselines/fit.py "${T[@]}" --fit-cells 0 --fit-space log1p_cp10k_full --out-dir "$OUT/work/baselines/tahoe_bio"
run work/baselines/parse_bio/fit_summary.json \
    "$PY" -u scripts/baselines/fit.py "${P[@]}" --fit-cells 0 --fit-space log1p_cp10k_full --out-dir "$OUT/work/baselines/parse_bio"

# 3. Score GENESIS on the split-half test set: generate and score in one pass, compiled as published (eager differs at
# the draw level); noise is seeded per (condition, repeat), so a subset scores as in the full run.
run work/split_half/tahoe_bio/GENESIS/metrics_summary.json \
    "$PY" -u scripts/eval/score.py genesis --compile --run-dir "$T_RUN" --checkpoint last --eval-split test \
    --model-eval-n-repeats 1 --seed 42 --metric-set extended ${SUBSET:+--conditions-query "$TQ_GEN"} \
    --out-dir "$OUT/work/split_half/tahoe_bio/GENESIS"
run work/split_half/parse_bio/GENESIS/metrics_summary.json \
    "$PY" -u scripts/eval/score.py genesis --compile --run-dir "$P_RUN" --checkpoint last --eval-split test \
    --model-eval-n-repeats 1 --seed 0 --metric-set extended ${SUBSET:+--conditions-query "$PQ_GEN"} \
    --out-dir "$OUT/work/split_half/parse_bio/GENESIS"

# 4. Score the baselines: each condition's real matched control cells plus the fitted delta, per cell
for arm in Additive:additive LinRidge:linear_ridge PertMean:perturbation_mean CtxMean:context_mean; do
    run "work/split_half/tahoe_bio/${arm%:*}/metrics_summary.json" \
        "$PY" -u scripts/eval/score.py baseline "${T[@]}" --artifact "$OUT/work/baselines/tahoe_bio/${arm#*:}.npz" \
        --seed 42 --metric-set extended ${SUBSET:+--conditions-query "$TQ"} --out-dir "$OUT/work/split_half/tahoe_bio/${arm%:*}"
    run "work/split_half/parse_bio/${arm%:*}/metrics_summary.json" \
        "$PY" -u scripts/eval/score.py baseline "${P[@]}" --artifact "$OUT/work/baselines/parse_bio/${arm#*:}.npz" \
        --seed 0 --metric-set extended ${SUBSET:+--conditions-query "$PQ"} --out-dir "$OUT/work/split_half/parse_bio/${arm%:*}"
done

# 5. Score copy_closest: the closest training context's own cells for the test perturbation. Its deltas.npz (22 MB on
# Tahoe) is rebuilt from the HVG cache's condition stats (--build-deltas, deterministic), never shipped.
run work/split_half/tahoe_bio/copy_closest/metrics_summary.json \
    "$PY" -u scripts/eval/score.py copy_neighbor "${T[@]}" --metric cosine --build-deltas \
    --deltas "$OUT/work/split_half/tahoe_bio/copy_closest/deltas.npz" --train-cache "$T_DATA/hvg_cache/geneset_c0d6420ffc0fa861__n_2000" \
    --plate-matched --own-controls --seed 0 --metric-set extended ${SUBSET:+--conditions-query "$TQ"} \
    --out-dir "$OUT/work/split_half/tahoe_bio/copy_closest"
run work/split_half/parse_bio/copy_closest/metrics_summary.json \
    "$PY" -u scripts/eval/score.py copy_neighbor "${P[@]}" --metric cosine --build-deltas \
    --deltas "$OUT/work/split_half/parse_bio/copy_closest/deltas.npz" --train-cache "$P_DATA/hvg_cache/geneset_63a547306aa0e6cd__n_2000" \
    --own-controls --seed 0 --metric-set extended ${SUBSET:+--conditions-query "$PQ"} \
    --out-dir "$OUT/work/split_half/parse_bio/copy_closest"

# 6. Score the split-half reference: each condition's cells halved and scored against each other, 20 redraws (truth
# only). --metric-set extended: the published files predate the field and hold every key. SUBSET's --condition-summary
# comes later on the line, and the later one wins.
T_REF=work/reference/tahoe_bio/geneset_c0d6420ffc0fa861__n_2000__norm_raw_counts/nr_20__splitcfg_e99f611ef11c7096__mincells_4__test/noise_ceiling.json
P_REF=work/reference/parse_bio/geneset_63a547306aa0e6cd__n_2000__norm_raw_counts/nr_20__splitcfg_7757bf1b97f87e4c__mincells_4__test/noise_ceiling.json
run "$T_REF" \
    "$PY" -u scripts/eval/reference.py split_half --dataset tahoe --split-config "$T_SPLIT" --eval-cache "$T_NS" \
    --gene-names "$T_GENES" --condition-summary "$T_SUMMARY" --n-repeats 20 --seed 42 --min-cells 4 --metric-set extended \
    ${SUBSET:+--condition-summary "$OUT/work/subset/tahoe_bio.parquet"} --out-root "$OUT/work/reference/tahoe_bio"
run "$P_REF" \
    "$PY" -u scripts/eval/reference.py split_half --dataset parse --split-config "$P_SPLIT" --eval-cache "$P_NS" \
    --gene-names "$P_GENES" --condition-variables donor cytokine cell_type bc1_well --control-match-keys donor cell_type \
    --condition-summary "$P_SUMMARY" --n-repeats 20 --seed 42 --min-cells 4 --metric-set extended \
    ${SUBSET:+--condition-summary "$OUT/work/subset/parse_bio.parquet"} --out-root "$OUT/work/reference/parse_bio"

# 7. Place the split-half files under $OUT/core_results
for panel in tahoe_bio parse_bio; do
    for arm in GENESIS Additive LinRidge PertMean CtxMean copy_closest; do
        place "work/split_half/$panel/$arm/metrics_summary.json" "figure2/split_half/$panel/$arm.json"
    done
done
place "$T_REF" figure2/split_half/tahoe_bio/reference.json
place "$P_REF" figure2/split_half/parse_bio/reference.json

# 8. Score no response where an arm failed, LAST because it reads every arm's file: the conditions the reference scores
# minus those an arm scored, over every arm Fig 2c draws (the shipped comparators too), with the failed arms' seed (0).
# A SUBSET run's arms hold 50 rows, so there the published files name the failures (1 Tahoe, 48 PBMC), all checked.
# CPU, as the published files were scored.
ARMS=$OUT/core_results/figure2/split_half; [ -z "${SUBSET:-}" ] || ARMS=core_results/figure2/split_half
SHIPPED=core_results/figure2/split_half
run work/no_response/tahoe_bio_failed.parquet \
    "$PY" scripts/eval/subset.py failed "$ARMS/tahoe_bio/reference.json" "$ARMS/tahoe_bio/"{GENESIS,Additive,LinRidge,PertMean,CtxMean,copy_closest}.json \
    "$SHIPPED/tahoe_bio/"{STATE,CellFlow,PerturbDiff}.json --summary "$T_SUMMARY" --out "$OUT/work/no_response/tahoe_bio_failed.parquet"
run work/no_response/parse_bio_failed.parquet \
    "$PY" scripts/eval/subset.py failed "$ARMS/parse_bio/reference.json" "$ARMS/parse_bio/"{GENESIS,Additive,LinRidge,PertMean,CtxMean,copy_closest}.json \
    "$SHIPPED/parse_bio/"{STATE,CellFlow,PerturbDiff}.json --summary "$P_SUMMARY" --out "$OUT/work/no_response/parse_bio_failed.parquet"
run work/no_response/tahoe_bio/metrics_summary.json \
    "$PY" -u scripts/eval/score.py no_response "${T[@]}" --condition-summary "$OUT/work/no_response/tahoe_bio_failed.parquet" \
    --seed 0 --metric-set extended --device cpu --out-dir "$OUT/work/no_response/tahoe_bio"
run work/no_response/parse_bio/metrics_summary.json \
    "$PY" -u scripts/eval/score.py no_response "${P[@]}" --condition-summary "$OUT/work/no_response/parse_bio_failed.parquet" \
    --seed 0 --metric-set extended --device cpu --out-dir "$OUT/work/no_response/parse_bio"
place work/no_response/tahoe_bio/metrics_summary.json figure2/no_response/tahoe_bio.json
place work/no_response/parse_bio/metrics_summary.json figure2/no_response/parse_bio.json

# 9. Generate GENESIS's cells for each (condition, plate) of the pairs at that plate's measured size: one seeded draw per
# file, so a subset generates as in the full run. Done once its prediction folder is; generation resumes file by file
# itself. SUBSET's --conditions comes later on the line, and the later one wins (as in 10).
PP=work/plate_pairs/tahoe_bio
run work/plate_pairs/genesis_pred \
    "$PY" -u scripts/eval/generate.py replicates --dataset tahoe --run-dir "$T_RUN" --conditions "$PAIRS" \
    ${SUBSET:+--conditions "$OUT/work/subset/tahoe_bio_pairs.tsv"} --eval-cache "$T_NS" --device cuda:0 \
    --out-pred "$OUT/work/plate_pairs/genesis_pred"

# 10. Score GENESIS and copy_closest against the experimental repeat (Fig 2a/b), one row per pair of plates, in
# log1p_slice_med_rgs with energy on every cell. copy_closest copies its neighbour line's cells and
# scores them against that line's own plate controls.
run $PP/model_vs_batch_gap__genesis.parquet \
    "$PY" -u scripts/eval/reference.py replicate_pairs --dataset tahoe --eval-cache "$T_NS" --conditions "$PAIRS" \
    ${SUBSET:+--conditions "$OUT/work/subset/tahoe_bio_pairs.tsv"} --model genesis --fmt npy \
    --pred-dir "$OUT/work/plate_pairs/genesis_pred" --energy-all-cells --report-space log1p_slice_med_rgs --device cuda:0 \
    --out "$OUT/$PP/model_vs_batch_gap__genesis.parquet"
run $PP/model_vs_batch_gap__copy_closest.parquet \
    "$PY" -u scripts/eval/reference.py replicate_pairs --dataset tahoe --eval-cache "$T_NS" --conditions "$PAIRS" \
    ${SUBSET:+--conditions "$OUT/work/subset/tahoe_bio_pairs.tsv"} --model copy_closest --fmt copy \
    --neighbor-tsv "$NEIGHBORS" --copy-ctrl own --energy-all-cells --report-space log1p_slice_med_rgs --device cuda:0 \
    --out "$OUT/$PP/model_vs_batch_gap__copy_closest.parquet"

# 11. Score the four baselines (step 2's fits) on GENESIS's pair list, copying its seeded true_true; the energy self-check
# reads the cells step 9 generated. Additive is written last.
run $PP/model_vs_batch_gap__additive.parquet \
    "$PY" -u scripts/eval/reference.py replicate_pairs_baselines --results "$OUT/$PP" --baseline-root "$OUT/work/baselines/tahoe_bio" \
    --check-pred-root "$OUT/work/plate_pairs/genesis_pred" --energy-all-cells --report-space log1p_slice_med_rgs --device cuda:0

# 12. Place the pair tables. GENESIS's published cells were drawn unseeded: a seeded rerun is a new, equivalent
# draw, so its table is reported (NEW_SAMPLE), not checked
for arm in genesis copy_closest additive context_mean linear_ridge perturbation_mean; do
    place "$PP/model_vs_batch_gap__$arm.parquet" "figure2/plate_pairs/tahoe_bio/pair_metrics__$arm.parquet"
done
NEW_SAMPLE=figure2/plate_pairs/tahoe_bio/pair_metrics__genesis.parquet
finish
