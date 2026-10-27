#!/usr/bin/env bash
# ED_program_summary: Figure 4's pathway responses cross-batch, on Figure 2a's 530 plate pairs (axis correlations).
# In:   figure4/score/main/score.sh's staged bundle ($OUT/work/fig4_bundle: its tables and compare_baselines cell sums), plus the
#       level-2 held-out lines and the frozen check references below (core_util_data/tahoe/eval_inputs/fig4_checks);
#       figure4/score/main/score.sh's GENESIS persist ($OUT/work/genesis_progeny3072); State through the progeny3072 panel.
#       pair_means.npz / neighbor_means.npz (~60 GB of predictions read) are rebuilt by step 1, never shipped.
# Out:  $OUT/core_results/figure4/axis_correlation/{pair_values__depth.parquet, profile_per_group.csv,
#       {profile,compound,line}_summary.csv}, compare_baselines/depth_module_values__nn_own.parquet (ED_celllines_per_model's
#       NN: its check reads step 2's pair values)
# Ships as-is (not rerun): State step 200k (comparator).   Also feeds: ED_celllines_per_model (step 4).
# Needs first: figure4/score/main/score.sh (full run: its GENESIS persist, the bundle and its compare_baselines fit).
# Cost: CPU ~1 h (step 1 reads ~60 GB of predictions), ~1 min step 2. No GPU.
#   OUT=<same dir as figure4/score/main/score.sh> bash scripts/figures/figure4/score/ed_program_summary/score.sh
source "$(dirname "$0")/../../../../eval/_env.sh"
BUNDLE="$OUT/work/fig4_bundle"
TAB=work/fig4_bundle/tables/supporting/tahoe_progeny_depth/axis_correlation
L2=core_deliverables/data/figure4/main
CHK=core_util_data/tahoe/eval_inputs/fig4_checks

# (full run only) steps 2-4 read every pair's predictions and the full fit: a subset has nothing to feed them
if [ -z "${SUBSET:-}" ]; then
    # The n_3072 caches pair_values.py and baseline_values.py read by themselves (no command line names them)
    need_path '{data:tahoe}/eval_cache/normalization_raw_counts/geneset_12aead7e0e3f60fa__n_3072'
    need_path '{data:tahoe}/hvg_cache/geneset_12aead7e0e3f60fa__n_3072'
    # 1. Stage what only these producers read: the held-out lines and the published values they check against
    stage() { run "work/fig4_bundle/$2" install -Dm644 "$1" "$BUNDLE/$2"; }
    stage "$L2/tahoe_progeny_depth/heldout_lines.tsv" data/tahoe_progeny_depth/heldout_lines.tsv
    stage "$CHK/data/tahoe_progeny_depth/plate_values.parquet" data/tahoe_progeny_depth/plate_values.parquet
    stage "$CHK/context_panel_source_data.csv" tables/supporting/tahoe_progeny_depth/context/context_panel_source_data.csv
    stage "$CHK/context_panel_source_data__state.csv" \
        tables/supporting/tahoe_progeny_depth/state/context/context_panel_source_data.csv
    # 2. Score every arm's depth-adjusted pathway values on each plate of Figure 2a's pairs (rebuilds the means caches)
    run work/fig4_bundle/data/axis_correlation/pair_values__depth.parquet \
        "$PY" -u scripts/figures/figure4/score/ed_program_summary/pair_values.py --bundle "$BUNDLE" --genesis "$OUT/work/genesis_progeny3072"
    place work/fig4_bundle/data/axis_correlation/pair_values__depth.parquet figure4/axis_correlation/pair_values__depth.parquet
    # 3. Correlate each model with the experimental replicate along the profile, compound and line axes (default arms)
    run "$TAB/line_summary.csv" "$PY" -u scripts/figures/figure4/score/ed_program_summary/axis_correlation.py --bundle "$BUNDLE"
    for f in profile_per_group.csv profile_summary.csv compound_summary.csv line_summary.csv; do
        place "$TAB/$f" "figure4/axis_correlation/$f"
    done
    # 4. Score Fig. 2a's nearest neighbor (its cells, its own controls) per pathway, checked against step 2's NN own
    run work/fig4_bundle/data/upstream/compare_baselines/depth_module_values__nn_own.parquet \
        "$PY" -u scripts/figures/figure4/score/main/baseline_values.py nn --bundle "$BUNDLE" \
        --pairs core_util_data/tahoe/eval_inputs/benchmark_pairs.csv
    place work/fig4_bundle/data/upstream/compare_baselines/depth_module_values__nn_own.parquet \
        figure4/compare_baselines/depth_module_values__nn_own.parquet
fi
finish
