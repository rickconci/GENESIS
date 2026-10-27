#!/usr/bin/env bash
# Figure 3: GENESIS on fewer training conditions or cells (3a, 3 runs per point) and a held-out line that trained on 0-164 of its perturbations (3b).
# In:   3a: the 14 runs other than Figure 2's GENESIS, runs/tahoe/<glob below>/checkpoints/last.ckpt (one per training timestamp);
#       their split_cache/train_idx.npy + run.log and the n_2000 HVG obs cache (training_runs.py, CPU).
#       3b: the 15 runs runs/tahoe/bio/scaling/overlap_celln10/entity_seed<s>/*_r<r>_es<s>/<ts>_*/checkpoints/last.ckpt.
# Out:  $OUT/core_results/figure3/{scaling_runs.csv, split_half/genesis_*.json (14), overlap_bio/r{0-4}_es{0-2}.json (15)}
#       Not regenerated: figure3/scaling_arms.csv and figure3/plate_pairs/pair_metrics__genesis_*.parquet (no extract reads them).
# Ships as-is (not rerun): nothing.   Needs first: figure2/score/main/score.sh (level 2 only: figure3/extract/extract.py reads Figure 2's
# per_condition and CIs from Figure 2's extract, which reads that driver's results).
# Cost: full ~8 h GPU (split-half 500-1,500 s x 14, overlap 256-593 s x 15) + scaling_runs.csv on CPU; SUBSET: 50 conditions each.
# Run on an H100: exactness depends on the GPU model, and the published files were made on H100s.
# overlap_bio is UNSEEDED (after-fit evals of the training runs): its rerun is a new sample, reported, not checked.
#   GPU=0 OUT=<new dir> [SUBSET=<dir>] bash scripts/figures/figure3/score/main/score.sh
source "$(dirname "$0")/../../../../eval/_env.sh"
need_gpu
NEW_SAMPLE="$(for r in 0 1 2 3 4; do for s in 0 1 2; do printf 'figure3/overlap_bio/r%d_es%d.json ' "$r" "$s"; done; done)"

# 1. Resolve every checkpoint first, so a moved or missing run fails here and not hours into the queue.
#    run_of <glob>: the one run dir matching <glob> that holds a last.ckpt (r2_es0 also has a crashed twin without one)
run_of() { need_run "$1" checkpoints/last.ckpt; }
SPLIT_HALF=(  # <published stem>=<its run under runs/tahoe/> (scripts/figures/figure3/score/scaling.py: REPLICATES)
    'genesis_100_mseed65=bio/main/*_mseed65/261012_174122_*'
    'genesis_100_mseed66=bio/main/*_mseed66/261012_212356_*'
    'genesis_25conds_11ep_r0=bio/scaling/n_conds_025/*_seed0/261014_085704_*'
    'genesis_25conds_11ep_r1=bio/scaling/n_conds_025/*_seed1/261014_234413_*'
    'genesis_25conds_11ep_r2=bio/scaling/n_conds_025/*_seed2/261015_134031_*'
    'genesis_5conds_25ep_r0=bio/scaling/n_conds_005/*_seed0/261013_143712_*'
    'genesis_5conds_25ep_r1=bio/scaling/n_conds_005/*_seed1/261013_202453_*'
    'genesis_5conds_25ep_r2=bio/scaling/n_conds_005/*_seed2/261015_064123_*'
    'genesis_25cells_allctrl_11ep_r0=bio/scaling/n_cells_025_keepctrl/*_seed0/261014_115029_*'
    'genesis_25cells_allctrl_11ep_r1=bio/scaling/n_cells_025_keepctrl/*_seed1/261016_181513_*'
    'genesis_25cells_allctrl_11ep_r2=bio/scaling/n_cells_025_keepctrl/*_seed2/261016_181556_*'
    'genesis_5cells_allctrl=bio/scaling/n_cells_005_keepctrl/*_seed0/261006_113942_*'
    'genesis_5cells_allctrl_r1=bio/scaling/n_cells_005_keepctrl/*_seed1/261018_182521_*'
    'genesis_5cells_allctrl_r2=bio/scaling/n_cells_005_keepctrl/*_seed2/261019_205600_*'
)
overlap_run() { run_of "runs/tahoe/bio/scaling/overlap_celln10/entity_seed$2/*_r$1_es$2/2*"; }
for e in "${SPLIT_HALF[@]}"; do run_of "runs/tahoe/${e#*=}" >/dev/null; done
for r in 0 1 2 3 4; do for s in 0 1 2; do overlap_run "$r" "$s" >/dev/null; done; done

# 2. (SUBSET) Freeze the sample once: 50 of Figure 2c's test conditions, the same file figure2/score/main/score.sh draws
#    (every Figure 3 run here holds out Figure 2c's test set). Drawn even under DRY_RUN: the queries below read it.
[ -z "${SUBSET:-}" ] || [ -e "$SUBSET/tahoe_bio.json" ] || "$PY" scripts/eval/subset.py draw \
    --split core_util_data/splits/tahoe/bio/tahoe_legacy_bio_holdout.yaml --n 50 --seed 0 --out "$SUBSET/tahoe_bio.json"

# 3. Count each 3a run's training cells, controls and conditions (CPU; deterministic, so SUBSET runs it in full too)
run work/training_runs/scaling_runs.csv \
    "$PY" -u scripts/figures/figure3/score/training_runs.py --runs --out "$OUT/work/training_runs"
place work/training_runs/scaling_runs.csv figure3/scaling_runs.csv

# 4. Score the 14 Figure 3a runs on Figure 2c's split-half test set, as Figure 2c is scored (K=1, last.ckpt)
for e in "${SPLIT_HALF[@]}"; do
    stem=${e%%=*}; RUN=$(run_of "runs/tahoe/${e#*=}")
    Q=${SUBSET:+$("$PY" scripts/eval/subset.py query "$RUN" "$SUBSET/tahoe_bio.json")}
    run "work/split_half/$stem/metrics_summary.json" \
        "$PY" -u scripts/eval/score.py genesis --run-dir "$RUN" --checkpoint last --eval-split test \
        --model-eval-n-repeats 1 --metric-set extended --gene-sidecars ${Q:+--conditions-query "$Q"} \
        --device cuda:0 --out-dir "$OUT/work/split_half/$stem"
    place "work/split_half/$stem/metrics_summary.json" "figure3/split_half/$stem.json"
done

# 5. Score the 15 Figure 3b runs (5 Latin rows x 3 entity seeds) on the same test set: an eval-only rerun of
#    last.ckpt with the published after-fit settings (K=1, extended metrics)
for r in 0 1 2 3 4; do for s in 0 1 2; do
    RUN=$(overlap_run "$r" "$s")
    Q=${SUBSET:+$("$PY" scripts/eval/subset.py query "$RUN" "$SUBSET/tahoe_bio.json")}
    run "work/overlap_bio/r${r}_es$s/metrics_summary.json" \
        "$PY" -u scripts/eval/score.py genesis --run-dir "$RUN" --checkpoint last --eval-split test \
        --model-eval-n-repeats 1 --metric-set extended --gene-sidecars ${Q:+--conditions-query "$Q"} \
        --device cuda:0 --out-dir "$OUT/work/overlap_bio/r${r}_es$s"
    place "work/overlap_bio/r${r}_es$s/metrics_summary.json" "figure3/overlap_bio/r${r}_es$s.json"
done; done
finish
