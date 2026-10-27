#!/usr/bin/env bash
# Figure 4 (main; also feeds ED_celllines_per_model): the PROGENy drug axis, GENESIS and State means, context, baselines.
# In:   GENESIS 5% cells, PROGENy 3,072 genes: runs/tahoe/bio/biology/progeny3072_cells005_keepctrl/
#       h4096x8192_nrs256_vg05rs256_25ep_progeny3072_tahoe_bio_5cell_kc_seed0/261016_181841_*/checkpoints/last.ckpt
#       Frozen inputs: core_util_data/tahoe/eval_inputs/{gene_sets.tsv, stories/{core_pair,curated_picks}.tsv} (hand-curated);
#       eval_inputs/drug_expectations.tsv (ctx_apriori.py's mechanism -> pathway table from the curated a priori tables;
#       not rerun: its line-feature checks need the DepMap 23Q4 download and the response atlas, both untracked);
#       eval_inputs/copy_closest_line__*.tsv (the neighbor tables); eval_inputs/fig4_checks/ (the published pathway tables the
#       producers check their own truth against); core_util_data/tahoe/metadata/drug_metadata.parquet (tracked annotation).
#       Reads the level-2 drug axis and depth tables (core_deliverables/data/figure4/main/): a selection rule applied to
#       level-3 scores.
#       Read where they ship, not regenerated (inputs, not scores): core_results/figure4/progeny/{library_size__tahoe.tsv
#       (10.06 MB, frozen), progeny_omnipath_academic.tsv (40.9 MB OmniPath download), eval_cache_n3072/ (941
#       files copied from the n_3072 eval cache, which scripts/data/build_cache.py eval builds)}.
# Out:  $OUT/core_results/figure4/{drug_axis/{genesis_test_k1_metrics_summary.json, drug_metadata.parquet},
#       plate_pairs/pair_metrics__genesis.parquet, progeny/arm_means__progeny3072.npz, context_analysis/{ctx_pass/*.npz,
#       drug_expectations.tsv, gene_sets.tsv}, compare_baselines/{summary_metrics__depth.tsv, depth_module_values.parquet},
#       stories/{core_pair,curated_picks}.tsv}
# Ships as-is (not rerun): State step 200k (comparator; model_means reads its predictions).   Also feeds: ED_celllines_per_model.
# Needs first: figure2/score/main/score.sh (its plate_pairs/tahoe_bio/pair_metrics__genesis.parquet).
# Cost: GPU ~41 min persist (4,211 conditions, ~95 GB); CPU ~3 h (ctx_means reads the ~95 GB, compare_baselines fits).
# GPU model: not recorded for the published persist; run SUBSET on an H100.
#   GPU=0 OUT=<new dir> [SUBSET=<dir>] bash scripts/figures/figure4/score/main/score.sh
source "$(dirname "$0")/../../../../eval/_env.sh"
need_gpu
RUN=$(need_run 'runs/tahoe/bio/biology/progeny3072_cells005_keepctrl/h4096x8192_nrs256_vg05rs256_25ep_progeny3072_tahoe_bio_5cell_kc_seed0/261016_181841_*' checkpoints/last.ckpt)
# The n_3072 caches no command line names: the eval cache (score.py through the run's config, the means through
# deg_readout's progeny3072 panel) and, for the full run's Additive, the HVG cache (compare_baselines.py)
need_path '{data:tahoe}/eval_cache/normalization_raw_counts/geneset_12aead7e0e3f60fa__n_3072'
[ -n "${SUBSET:-}" ] || need_path '{data:tahoe}/hvg_cache/geneset_12aead7e0e3f60fa__n_3072'
SPLIT=core_util_data/splits/tahoe/bio/tahoe_scaling_bio_keepctrl/entity_seed0/tahoe_scaling_bio_celln4579207_keepctrl_mhbss005_es0.yaml
BUNDLE="$OUT/work/fig4_bundle"
M=scripts/figures/figure4/score/main
L2=core_deliverables/data/figure4/main
CHK=core_util_data/tahoe/eval_inputs/fig4_checks

# 1. Draw the subset (SUBSET only): 50 seeded test conditions of the run's split
[ -z "${SUBSET:-}" ] || [ -e "$SUBSET/figure4_progeny3072.json" ] || run work/subset/figure4_progeny3072.json \
    "$PY" scripts/eval/subset.py draw --split "$SPLIT" --n 50 --seed 0 --out "$SUBSET/figure4_progeny3072.json"

# 2. Score GENESIS on the test set and persist its predictions: every downstream mean reads them. K=1, seed 42
# (score.py's default), extended metrics: the published metrics_summary.json's settings
run work/genesis_progeny3072/metrics_summary.json \
    "$PY" -u scripts/eval/score.py genesis --run-dir "$RUN" --checkpoint last --eval-split test \
    --model-eval-n-repeats 1 --seed 42 --metric-set extended --gene-sidecars --persist-predictions \
    ${SUBSET:+--conditions-query "$("$PY" scripts/eval/subset.py query "$RUN" "$SUBSET/figure4_progeny3072.json")"} \
    --out-dir "$OUT/work/genesis_progeny3072"
place work/genesis_progeny3072/metrics_summary.json figure4/drug_axis/genesis_test_k1_metrics_summary.json

# 3. Copy the tracked drug annotation and the hand-curated tables: inputs, placed so core_results stays whole
run work/inputs/drug_metadata.parquet install -Dm644 core_util_data/tahoe/metadata/drug_metadata.parquet "$OUT/work/inputs/drug_metadata.parquet"
run work/inputs/gene_sets.tsv install -Dm644 core_util_data/tahoe/eval_inputs/gene_sets.tsv "$OUT/work/inputs/gene_sets.tsv"
run work/inputs/core_pair.tsv install -Dm644 core_util_data/tahoe/eval_inputs/stories/core_pair.tsv "$OUT/work/inputs/core_pair.tsv"
run work/inputs/curated_picks.tsv install -Dm644 core_util_data/tahoe/eval_inputs/stories/curated_picks.tsv "$OUT/work/inputs/curated_picks.tsv"
run work/inputs/drug_expectations.tsv install -Dm644 core_util_data/tahoe/eval_inputs/drug_expectations.tsv "$OUT/work/inputs/drug_expectations.tsv"
place work/inputs/drug_metadata.parquet figure4/drug_axis/drug_metadata.parquet
place work/inputs/gene_sets.tsv figure4/context_analysis/gene_sets.tsv
place work/inputs/core_pair.tsv figure4/stories/core_pair.tsv
place work/inputs/curated_picks.tsv figure4/stories/curated_picks.tsv
place work/inputs/drug_expectations.tsv figure4/context_analysis/drug_expectations.tsv

# (full run only) steps 4-8 read every drug-axis condition's predictions (step 2's, --genesis) and cell sums: a subset
# persist cannot feed them
if [ -z "${SUBSET:-}" ]; then
    # 4. Stage the producers' --bundle layout, one tracked file each (stage <source> <bundle path>)
    stage() { run "work/fig4_bundle/$2" install -Dm644 "$1" "$BUNDLE/$2"; }
    stage "$L2/drug_axis/drug_axis.tsv" data/drug_axis/drug_axis.tsv
    stage "$L2/tahoe_progeny_depth/module_values.parquet" data/tahoe_progeny_depth/module_values.parquet
    for s in raw depth; do stage "$L2/tahoe_progeny_depth/programmes.tsv" "data/tahoe_progeny_$s/programmes.tsv"; done  # one table
    for f in tahoe_progeny_raw/module_values.parquet tahoe_progeny_centred/module_values.parquet \
             tahoe_progeny_raw/metrics.tsv tahoe_progeny_centred/metrics.tsv tahoe_progeny_depth/metrics.tsv; do
        stage "$CHK/data/$f" "data/$f"
    done
    stage core_results/figure4/progeny/progeny_omnipath_academic.tsv data/upstream/progeny_omnipath_academic.tsv
    stage core_results/figure4/progeny/library_size__tahoe.tsv context_analysis/data/upstream/library_size__tahoe.tsv
    stage core_util_data/tahoe/eval_inputs/gene_sets.tsv context_analysis/data/apriori/gene_sets.tsv
    # 5. Gate the drug axis on Figure 2's plate pairs (truth vs truth DEG >= 0.2 at 5 uM). Figure 4 reads only these
    # truth columns, identical to the published table's (274 gated pairs, max |diff| 0); its model columns differ
    run work/fig4_bundle/data/upstream/tahoe_bio/pair_metrics__genesis.parquet \
        install -Dm644 "$OUT/core_results/figure2/plate_pairs/tahoe_bio/pair_metrics__genesis.parquet" \
        "$BUNDLE/data/upstream/tahoe_bio/pair_metrics__genesis.parquet"
    place work/fig4_bundle/data/upstream/tahoe_bio/pair_metrics__genesis.parquet figure4/plate_pairs/pair_metrics__genesis.parquet
    # 6. Read GENESIS and State's mean counts on the drug axis (106 drugs, 5 uM, 5 held-out lines; CPU 5 x 8 threads)
    run work/fig4_bundle/data/upstream/arm_means__progeny3072.npz "$PY" -u "$M/model_means.py" --bundle "$BUNDLE" --genesis "$OUT/work/genesis_progeny3072"
    place work/fig4_bundle/data/upstream/arm_means__progeny3072.npz figure4/progeny/arm_means__progeny3072.npz
    # 7. Read every test condition's truth and GENESIS means for the context analysis (all 4,211; CPU 5 x 6, ~95 GiB read)
    run work/fig4_bundle/context_analysis/data/upstream/ctx_pass/PANC-1__genesis.npz "$PY" -u "$M/ctx_means.py" --bundle "$BUNDLE" --genesis "$OUT/work/genesis_progeny3072"
    for line in C32 HOP62 HepG2_C3A Hs_766T PANC-1; do for arm in genesis truth; do
        place "work/fig4_bundle/context_analysis/data/upstream/ctx_pass/${line}__${arm}.npz" "figure4/context_analysis/ctx_pass/${line}__${arm}.npz"
    done; done
    # 8. Fit the Additive on the cells GENESIS trained on and read the nearest neighbor (CPU, 90 GB, ~17 min), then
    # the Additive's per-pathway values for ED_celllines_per_model (its NN needs the pair values:
    # score/ed_program_summary/score.sh)
    run work/fig4_bundle/data/upstream/compare_baselines/summary_metrics__depth.tsv \
        "$PY" -u "$M/compare_baselines.py" --bundle "$BUNDLE"
    run work/fig4_bundle/data/upstream/compare_baselines/depth_module_values.parquet \
        "$PY" -u "$M/baseline_values.py" additive --bundle "$BUNDLE"
    for f in summary_metrics__depth.tsv depth_module_values.parquet; do
        place "work/fig4_bundle/data/upstream/compare_baselines/$f" "figure4/compare_baselines/$f"
    done
fi
finish
