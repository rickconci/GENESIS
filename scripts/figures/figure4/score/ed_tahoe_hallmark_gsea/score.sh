#!/usr/bin/env bash
# ED_tahoe_hallmark_gsea: Hallmark NES per plate for truth, GENESIS, Additive and Copy closest on the tahoe-bio test set.
# In:   GENESIS 15% conditions, union 5,750 genes: runs/tahoe/bio/biology/union5750_conds015/
#       h4096x8192_nrs256_vg03rs256_33ep_union5750_tahoe_bio_15cond_seed0/260925_013450_*/checkpoints/last.ckpt
#       Frozen: core_util_data/tahoe/eval_inputs/neighbors.parquet (the copy-neighbour table; shipped, not rebuilt).
#       Read where they ship (> 10 MB): core_results/figure4/progeny/library_size__tahoe.tsv (frozen).
#       Untracked, read where they are: results/tahoe/union5750/response_atlas (1.7 GB; the Copy side's check) and
#       arm_means__union5750_hbss015.npz (16 MB; its Additive check, skipped for this log1p_cp10k_full fit).
# Out:  $OUT/core_results/figure4/hallmark_gsea/{nes.parquet, plates.tsv}
# Ships as-is (not rerun): nothing.   Also feeds: nothing.
# UNSEEDED: the published persist (pregenerate_pertgen, eager, no seeding) cannot be reproduced exactly, so a
# SUBSET run reports hallmark_gsea/* (NEW_SAMPLE) instead of checking them.
# Cost: GPU ~32 min persist (~190 GB); CPU 2.5 min fit, 25 min GSEA (12 workers, 90 GB cap).
# GPU model: not recorded for the published persist; run SUBSET on an H100.
#   GPU=0 OUT=<new dir> [SUBSET=<dir>] bash scripts/figures/figure4/score/ed_tahoe_hallmark_gsea/score.sh
source "$(dirname "$0")/../../../../eval/_env.sh"
need_gpu
RUN=$(need_run 'runs/tahoe/bio/biology/union5750_conds015/h4096x8192_nrs256_vg03rs256_33ep_union5750_tahoe_bio_15cond_seed0/260925_013450_*' checkpoints/last.ckpt)
SPLIT=core_util_data/splits/tahoe/bio/tahoe_scaling_bio_split/entity_seed0/tahoe_scaling_bio_hbss015_es0.yaml
CACHE=$("$PY" -m src.paths '{data:tahoe}/eval_cache/normalization_raw_counts/geneset_e639c3a3c608d54a__n_5750')
NEW_SAMPLE="figure4/hallmark_gsea/nes.parquet figure4/hallmark_gsea/plates.tsv"

# 1. Draw the subset (SUBSET only): 50 seeded test conditions of the run's split
[ -z "${SUBSET:-}" ] || [ -e "$SUBSET/figure4_union5750_hbss015.json" ] || run work/subset/figure4_union5750_hbss015.json \
    "$PY" scripts/eval/subset.py draw --split "$SPLIT" --n 50 --seed 0 --out "$SUBSET/figure4_union5750_hbss015.json"

# 2. Score GENESIS on the test set and persist its predictions: the GENESIS side of every NES. K=1, extended metrics
run work/genesis_union5750_hbss015/metrics_summary.json \
    "$PY" -u scripts/eval/score.py genesis --run-dir "$RUN" --checkpoint last --eval-split test \
    --model-eval-n-repeats 1 --seed 42 --metric-set extended --gene-sidecars --persist-predictions \
    ${SUBSET:+--conditions-query "$("$PY" scripts/eval/subset.py query "$RUN" "$SUBSET/figure4_union5750_hbss015.json")"} \
    --out-dir "$OUT/work/genesis_union5750_hbss015"

# 3. Fit the Additive on exactly the triplets the 15% run kept (deterministic, all cells, log1p_cp10k_full)
run work/baselines/union5750_hbss015/additive.npz \
    "$PY" -u scripts/baselines/fit.py --dataset tahoe --split-yaml "$SPLIT" --eval-cache-namespace "$CACHE" \
    --gene-names core_util_data/tahoe/gene_subsets/STATE_HVG_plus_hallmark_5750.txt \
    --train-subset "$RUN/hbss_selected_triplets.parquet" --kinds additive --fit-cells 0 --fit-space log1p_cp10k_full \
    --out-dir "$OUT/work/baselines/union5750_hbss015"

# 4. Run every stage on steps 2-3's GENESIS and Additive: per-plate pseudobulks, preranked GSEA on the 50 Hallmark
# sets, agreement (its cache needs the frozen neighbour table first)
run work/hallmark/neighbors.parquet install -Dm644 core_util_data/tahoe/eval_inputs/neighbors.parquet "$OUT/work/hallmark/neighbors.parquet"
run work/hallmark/nes.parquet "$PY" -u scripts/figures/figure4/score/ed_tahoe_hallmark_gsea/hallmark_gsea.py --stage all --workers 12 --out "$OUT/work/hallmark" \
    --genesis "$OUT/work/genesis_union5750_hbss015" --additive-root "$OUT/work/baselines/union5750_hbss015" \
    --library core_results/figure4/progeny/library_size__tahoe.tsv
place work/hallmark/nes.parquet figure4/hallmark_gsea/nes.parquet
place work/hallmark/plates.tsv figure4/hallmark_gsea/plates.tsv
finish
