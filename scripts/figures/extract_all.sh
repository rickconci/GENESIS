#!/usr/bin/env bash
# Level 2 for every figure: core_results/figureN -> core_deliverables/data/figureN, one extract per figure.
#   bash scripts/figures/extract_all.sh            # rebuild in memory and compare with the committed data; writes nothing
#   bash scripts/figures/extract_all.sh --write    # overwrite what differs, and its MANIFEST rows
# Level 1 is python -m scripts.figures.draw_all; level 3 is scripts/figures/score_all.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-python}"
EXTRACTS=(scripts.figures.figure1.extract.extract scripts.figures.figure2.extract.extract scripts.figures.figure3.extract.extract
          scripts.figures.figure4.extract.extract)
for m in "${EXTRACTS[@]}"; do
    echo "== $m"
    "$PY" -m "$m" "$@"
done
