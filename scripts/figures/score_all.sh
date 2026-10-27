#!/usr/bin/env bash
# Level 3 for every figure: GENESIS checkpoints -> core_results/figureN, every driver in dependency order (Figure 3 and
# Figure 4 read Figure 2's results, so Figure 2 runs first). The environment (OUT, GPU, SUBSET, DRY_RUN, PY) passes through
# to each driver unchanged; see scripts/eval/_env.sh.
#   OUT=<scratch dir> GPU=<idx> [SUBSET=<dir>] [DRY_RUN=1] bash scripts/figures/score_all.sh
# Every driver runs even when one fails (a later one that needs its results fails too, and says so); the closing
# summary gives each driver's result: ok, MISSING <n> (DRY_RUN found n missing inputs, caches or runs) or
# FAIL <exit code>. The exit code is non-zero unless every driver is ok.
# Level 2 is scripts/figures/extract_all.sh; level 1 is python -m scripts.figures.draw_all.
set -uo pipefail
cd "$(dirname "$0")"
DRIVERS=(
    figure1/score/main/score.sh figure1/score/ed_genesis_settings/score.sh
    figure2/score/main/score.sh figure2/score/ed_conditioning/score.sh figure2/score/ed_objective/score.sh
    figure2/score/ed_pbmc_wells/score.sh figure2/score/ed_pert_generalization/score.sh
    figure3/score/main/score.sh figure3/score/ed_12cells_per_condition/score.sh figure3/score/ed_scaling_pert_split/score.sh
    figure4/score/main/score.sh figure4/score/ed_program_summary/score.sh figure4/score/ed_tahoe_hallmark_gsea/score.sh
)
for d in "${DRIVERS[@]}"; do [ -f "$d" ] || { echo "score_all: missing driver $d" >&2; exit 1; }; done
RESULT=(); FAILED=0
# a dry run's driver output passes through a file too, for finish's "dry run: <n> missing" line (and its exit 2)
[ -z "${DRY_RUN:-}" ] || { LOG="$(mktemp)"; trap 'rm -f "$LOG"' EXIT; }
for d in "${DRIVERS[@]}"; do
    echo "== $d"
    if [ -n "${DRY_RUN:-}" ]; then bash "$d" | tee "$LOG"; rc=${PIPESTATUS[0]}; else bash "$d"; rc=$?; fi
    n=; if [ "$rc" = 2 ] && [ -n "${DRY_RUN:-}" ]; then n=$(sed -n 's/^dry run: \([0-9]*\) missing (listed above)$/\1/p' "$LOG"); fi
    if [ "$rc" = 0 ]; then status=ok; elif [ -n "$n" ]; then status="MISSING $n"; else status="FAIL $rc"; fi
    RESULT+=("$(printf '%-11s %s' "$status" "$d")")
    [ "$rc" = 0 ] || FAILED=1
done
echo "== summary"
printf '%s\n' "${RESULT[@]}"
exit "$FAILED"
