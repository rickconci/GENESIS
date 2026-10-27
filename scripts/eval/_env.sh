# Sourced by every level-3 driver (scripts/figures/figureN/score/<panel>/score.sh): the shared shell environment.
#
#   OUT=<scratch dir> [GPU=<idx>] [SUBSET=<dir of frozen subsets>] [PY=<python>] bash scripts/figures/figureN/score/<panel>/score.sh
#   DRY_RUN=1 ... : print every command and flag any path argument ({data:...}/{root:...} expanded), run or need_path
#                   that does not exist; run, copy and check nothing; exit 2 if anything was missing
#
# Everything a driver writes lands under $OUT; $OUT/core_results/ mirrors core_results/ path for path, so staging a
# full run is one copy of it over the real tree (after backing up the files it replaces). core_results/ itself is only
# ever read here: a driver that wrote into it would turn "reproduce the published numbers" into "overwrite them".
set -euo pipefail

# 1. Find the repo root and guard OUT before anything slow
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -n "${OUT:-}" ] || { echo "_env.sh: set OUT to a scratch directory outside the repo's results" >&2; exit 1; }
OUT="$(realpath -m "$OUT")"
# place writes $OUT/core_results/..., so OUT must not be inside the protected trees NOR make that path resolve to them
# (OUT=$ROOT, OUT=. at the root, or a parent of the root would all overwrite the published results)
for protected in core_results core_deliverables; do
    real="$(realpath -m "$ROOT/$protected")"  # a protected tree may be a symlink to the data release's disk
    case "$OUT/" in "$ROOT/$protected/"*|"$real/"*) echo "_env.sh: OUT=$OUT is inside $protected/; refusing" >&2; exit 1;; esac
    [ "$(realpath -m "$OUT/$protected")" != "$real" ] \
        || { echo "_env.sh: OUT=$OUT would place files into the real $protected/; refusing" >&2; exit 1; }
done
cd "$ROOT"
mkdir -p "$OUT"
# One OUT holds one mode: a done marker is keyed on the output name alone, so a full run into a SUBSET run's OUT
# would skip every step and place the subset's files as the full result. A dry run checks the mode, never records it.
MODE=full; [ -z "${SUBSET:-}" ] || MODE=subset
if [ -f "$OUT/.mode" ]; then
    [ "$(cat "$OUT/.mode")" = "$MODE" ] \
        || { echo "_env.sh: OUT=$OUT holds a $(cat "$OUT/.mode") run; this is a $MODE run: use a new OUT" >&2; exit 1; }
elif [ -z "${DRY_RUN:-}" ]; then
    echo "$MODE" > "$OUT/.mode"
fi

# 2. Pick the interpreter: fail here, not an hour into a job
PY="${PY:-python}"
if ! import_err="$("$PY" -c "import torch, src" 2>&1)"; then
    echo "_env.sh: '$PY' cannot import torch and src (set PY to the project's python): $(tail -n 1 <<<"$import_err")" >&2
    exit 1
fi

# GPU is needed only by drivers that generate; they call need_gpu
[ -z "${GPU:-}" ] || export CUDA_VISIBLE_DEVICES="$GPU"  # unset GPU leaves a caller's CUDA_VISIBLE_DEVICES alone
need_gpu() { [ -n "${GPU:-}${DRY_RUN:-}" ] || { echo "this driver needs a GPU: set GPU=<index>" >&2; exit 1; }; }

# 3. The helpers every driver is written in
PLACED=()
declare -A EXPANDED=()  # DRY_RUN's {data:...}/{root:...} expansions, one python call per template

# missing <message>: report what a real run would stop at; DRY_RUN counts it, so finish can exit 2 instead of "ok".
# need_run reports from inside $(...), a subshell no variable leaves, so the count is the lines of a file.
[ -z "${DRY_RUN:-}" ] || { MISSED="$(mktemp)"; trap 'rm -f "$MISSED"' EXIT; }
missing() { echo "$1" >&2; [ -z "${DRY_RUN:-}" ] || echo >> "$MISSED"; }

# need_run <glob> [<inside>]: the one path matching <glob> (that holds <inside>, when given: a run folder with
# checkpoints/last.ckpt, never its crashed twin without one). None or several: a real run stops here, before any
# GPU time; DRY_RUN reports it and goes on with <glob> as a placeholder, so one dry run lists every missing run.
need_run() {
    local m=($1${2:+/$2}) n=0
    [ ! -e "${m[0]}" ] || n=${#m[@]}
    if [ "$n" = 1 ]; then echo "${m[0]%"${2:+/$2}"}"; return; fi
    if [ "$n" = 0 ]; then missing "missing: $1${2:+/$2} (download the GENESIS checkpoints)"
    else missing "need_run: $1${2:+/$2} matches $n paths, expected one"; fi
    [ -n "${DRY_RUN:-}" ] || exit 1
    echo "$1"
}

# run <done-file> <cmd...>: skip once <done-file> finished, so a rerun resumes instead of redoing hours of work.
# "Finished" is a marker written after the command exits 0, not the file itself: some scorers write partial output as
# they go, and a killed run must not look done. <done-file> is the step's own output, one per step: two steps sharing
# it would skip the second as soon as the first finished, so a reuse stops the driver.
declare -A DONE_FILES=()
run() {
    [ -z "${DONE_FILES[$1]+set}" ] \
        || { echo "run: done-file $1 is already an earlier step's; give each step its own output as done-file" >&2; exit 1; }
    DONE_FILES[$1]=1
    local done_file="$OUT/$1" mark="$OUT/.done/$1"; shift
    if [ -e "$mark" ]; then echo "skip (done): ${done_file#"$OUT"/}"; return; fi
    if [ -n "${DRY_RUN:-}" ]; then  # a wrong run dir or input path should cost seconds, not a GPU slot
        echo "+ $*"; local a
        for a in "$@"; do
            case "$a" in *'{'*'}'*)  # a template: check the path it names on this machine
                [ -n "${EXPANDED[$a]+set}" ] || EXPANDED[$a]=$("$PY" -m src.paths "$a"); a=${EXPANDED[$a]};; esac
            # a glob is need_run's placeholder for a run it already reported missing
            case "$a" in -*|*'*'*|"$OUT"/*) ;; */*) [ -e "$a" ] || missing "  missing: $a";; esac
        done
        return
    fi
    echo "+ $*"
    "$@" || { local rc=$?; echo "run: failed with exit $rc: $*" >&2; exit "$rc"; }  # stop even where set -e is off
    mkdir -p "$(dirname "$mark")"; : > "$mark"
}

# need_path <path or {template}>: a cache or input a producer reads by itself, which no command line names, so run
# cannot see it. A real run stops here, before any GPU time; DRY_RUN reports it and goes on.
need_path() {
    local p=$1
    case "$p" in *'{'*'}'*) p=$("$PY" -m src.paths "$p");; esac
    [ -n "$p" ] || { echo "need_path: $1 expands to nothing" >&2; exit 1; }
    if [ -e "$p" ]; then return; fi
    missing "missing: $p (read inside a producer; no command line names it)"
    [ -n "${DRY_RUN:-}" ] || exit 1
}

# place <produced> <rel>: copy $OUT/<produced> to $OUT/core_results/<rel>, the same <rel> as under core_results/
place() {
    if [ -n "${DRY_RUN:-}" ]; then PLACED+=("$2"); return; fi
    [ -f "$OUT/$1" ] || { echo "place: $OUT/$1 was not produced" >&2; exit 1; }
    mkdir -p "$(dirname "$OUT/core_results/$2")"
    cp "$OUT/$1" "$OUT/core_results/$2"; PLACED+=("$2")
}

# finish: list what was placed; with SUBSET, compare what THIS driver placed to the published files (exit 1 on any
# difference): every driver of score_all.sh shares one OUT, and another driver's files are not this one's to check.
# NEW_SAMPLE (space-separated <rel>) names placed files that are a new (unseeded) sample: their published and rerun
# means are printed, never checked. They accumulate in $OUT/.new_sample, one per line, for the whole-tree compare.
finish() {
    if [ "${#PLACED[@]}" -gt 0 ]; then printf 'placed %s\n' "${PLACED[@]}"; else echo "placed nothing"; fi
    if [ -n "${DRY_RUN:-}" ]; then
        echo "dry run: nothing was run, copied or checked"
        local n=$(( $(wc -l < "$MISSED") ))
        [ "$n" -eq 0 ] || { echo "dry run: $n missing (listed above)"; exit 2; }  # score_all.sh reads this line
        return
    fi
    local new_sample=(); read -ra new_sample <<<"${NEW_SAMPLE:-}"  # read, not $NEW_SAMPLE unquoted: no globbing
    touch "$OUT/.new_sample"
    [ "${#new_sample[@]}" -eq 0 ] || printf '%s\n' "${new_sample[@]}" >> "$OUT/.new_sample"
    sort -u -o "$OUT/.new_sample" "$OUT/.new_sample"  # a rerun lists each path once
    if [ -n "${SUBSET:-}" ]; then
        "$PY" scripts/eval/subset.py check "$OUT/core_results" core_results --only "${PLACED[@]}" --new-sample "${new_sample[@]}"
    else
        echo "full run: compare with  $PY scripts/eval/subset.py check $OUT/core_results core_results --new-sample-file $OUT/.new_sample"
    fi
}
