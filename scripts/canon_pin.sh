#!/usr/bin/env bash
# Canonicalization pin gate: explore a fixed set of lemmas with
# `explore_canonical_matches` (exhaustive breadth, fixed depth/node budgets)
# on a BASELINE build and on the working tree, and compare the AND/OR graphs.
# Which systems merge is exactly what canonicalization decides, so a changed
# graph is a changed canonicalization (or a changed solver, which the
# comparison tells apart).
#
#   scripts/canon_pin.sh fast             5-15 min warm: small changes
#   scripts/canon_pin.sh full             ~2 h (default list): big changes
#   scripts/canon_pin.sh calibrate fast|full
#                                         time every candidate lemma with the
#                                         WORKING TREE's explorer and rewrite
#                                         the tier's list (see below)
#
# How a run goes:
#   1. BASE (default `@{upstream}`, i.e. origin/cs-canon on the canon branch)
#      is resolved to a commit and its explorer built in a reusable detached
#      worktree under CACHE_DIR (its own target dir, so the working tree's
#      build is never touched); the binary is kept per commit.
#   2. The working tree's explorer is built (`cargo build --release
#      --example explore_canonical_matches`), uncommitted changes included.
#   3. The tier's list is expanded to (theory, lemma) jobs. A row without
#      lemmas means every lemma `--list-lemmas` reports on BOTH builds; a
#      theory whose lemma set differs between them is a LEMMA_SET_DIFF.
#   4. Phase A runs the BASE explorer for every job not yet in the cache and
#      stores its normalized JSON (the "stored graph", see canon_pin.py
#      normalize). The cache key covers the base commit, the theory (+
#      includes + flags), the lemma, the budget, maude, bliss and the rayon
#      thread count, so a stale hit is impossible and a warm cache makes a
#      re-run cost only the branch side.
#   5. Phase B runs the working tree's explorer and compares (canon_pin.py
#      compare). Differences keep both JSONs under DIFF_DIR for `jq`.
#
# Row statuses (OUT, one per job):
#   SAME, FP_ONLY           pass (FP_ONLY: the canonical form changed, the
#                           merges did not -- reported, not a failure)
#   MERGE_DIFF, CANON_FAIL_DIFF, SOLVER_DIFF, STOP_DIFF, METHOD_CHECK_DIFF,
#   OTHER_DIFF              the graphs differ (see canon_pin.py)
#   STATUS_DIFF             the explorer exited differently (e.g. a new panic)
#   BRANCH_TIMEOUT          the base finished within TIMEOUT, the branch not
#   LEMMA_SET_DIFF          the builds list different lemmas for a theory
#   NOT_COMPARED            the BASE produced no graph (timeout/setup error):
#                           counted and listed, not a failure -- calibrate
#                           keeps such lemmas off the lists
# Skipped up front and listed: missing files, `--diff` theories (the port has
# no diff-mode prover). A row without lemmas does not reach auto-sources'
# generated `AUTO_typing` lemma (`--list-lemmas` runs no Maude, so it cannot
# generate it); name it explicitly in a row to pin it.
#
# Determinism: only depth and node budgets bound a run (both deterministic);
# no --time-budget/--max-rss-gb is ever passed, a TIMEOUT only kills. Rayon's
# thread count is pinned (RAYON_NUM_THREADS, default 4) like rs_ref_check
# pins --processors. `BASE=HEAD scripts/canon_pin.sh fast` on a clean tree
# compares a build against itself and must be all SAME.
#
# The baseline explorer must have the pin interface (-D, --auto-sources,
# schema 4): a BASE older than the commit that added this script is refused.
#
# Env (all modes): BASE, CACHE_DIR (default ~/.cache/tamarin-rs/canon_pin),
#   JOBS, DEPTH, NODES, TIMEOUT (per explorer run, seconds), LIST, CORPUS,
#   FLAGS_MAP, OUT (row TSV), DIFF_DIR, RAYON_NUM_THREADS.
# Tier presets:     DEPTH NODES  TIMEOUT LIST
#   fast             4    1000    120   scripts/canon_pin_fast.txt
#   full             6   10000    900   scripts/canon_pin_full.txt
# calibrate: CANDIDATES (default parity_corpus_fast.txt / parity_corpus.txt),
#   TARGET_MIN (warm-run wall-time target: fast 7, full 120), CALIB (the
#   timing TSV; an existing one is reused, delete it to re-time).
set -u
export LC_ALL=C

usage() {
    echo "usage: $0 fast|full" >&2
    echo "       $0 calibrate fast|full" >&2
}
MODE="${1:-}"
case "$MODE" in
    fast|full) TIER=$MODE; shift ;;
    calibrate) TIER="${2:-}"; case "$TIER" in fast|full) shift 2 ;; *) usage; exit 2 ;; esac ;;
    *) usage; exit 2 ;;
esac
[ $# -eq 0 ] || { echo "canon_pin: unexpected argument '$1' (settings go in env vars)" >&2; usage; exit 2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -r "$HERE/gate_common.sh" ] || { echo "canon_pin: missing $HERE/gate_common.sh" >&2; exit 2; }
. "$HERE/gate_common.sh"
PY="$HERE/canon_pin.py"

ROOT="${ROOT:-$(cd "$HERE/.." && pwd)}"
CORPUS="${CORPUS:-$ROOT/tamarin-prover/examples}"
FLAGS_MAP="${FLAGS_MAP:-$ROOT/scripts/file_flags.tsv}"
CACHE_DIR="${CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/tamarin-rs/canon_pin}"
case "$TIER" in
    fast) DEPTH="${DEPTH:-4}"; NODES="${NODES:-1000}"; TIMEOUT="${TIMEOUT:-120}"
          LIST="${LIST:-$ROOT/scripts/canon_pin_fast.txt}"; TARGET_MIN="${TARGET_MIN:-7}"
          CANDIDATES="${CANDIDATES:-$ROOT/scripts/parity_corpus_fast.txt}" ;;
    full) DEPTH="${DEPTH:-6}"; NODES="${NODES:-10000}"; TIMEOUT="${TIMEOUT:-900}"
          LIST="${LIST:-$ROOT/scripts/canon_pin_full.txt}"; TARGET_MIN="${TARGET_MIN:-120}"
          CANDIDATES="${CANDIDATES:-$ROOT/scripts/parity_corpus.txt}" ;;
esac
cores=$(nproc 2>/dev/null || echo 4)
mem_gib=$(awk '/^MemTotal:/ {print int($2 / 1048576)}' /proc/meminfo 2>/dev/null || echo 8)
jobs_default=$(( cores / 2 < mem_gib / 3 ? cores / 2 : mem_gib / 3 ))
JOBS="${JOBS:-$(( jobs_default < 1 ? 1 : jobs_default ))}"
OUT="${OUT:-/tmp/canon_pin_$TIER.tsv}"
DIFF_DIR="${DIFF_DIR:-/tmp/canon_pin_${TIER}_diffs}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-4}"
export CORPUS FLAGS_MAP CACHE_DIR DEPTH NODES TIMEOUT PY DIFF_DIR

# The top-N search variables change what the solver does; a run that
# inherits one from the caller's shell pins something other than it claims.
for v in TAM_RS_TOP_METHODS TAM_RS_MERGE TAM_RS_TOPN_BATCH TAM_RS_SEARCH_ORDER \
         TAM_RS_TOPN_ALT_COST TAM_RS_TOPN_STATS; do
    if [ -n "${!v+x}" ]; then
        echo "canon_pin: $v is set in the environment; unset it" >&2; exit 2
    fi
done

# --- preflight -------------------------------------------------------------------
MAUDE=$(resolve_maude) || exit 2
maude_on_path "$MAUDE"
BLISS="${BLISS_PATH:-$(command -v bliss 2>/dev/null || true)}"
[ -n "$BLISS" ] && [ -x "$BLISS" ] || { echo "canon_pin: no bliss (set BLISS_PATH or put bliss on PATH)" >&2; exit 2; }
export BLISS_PATH="$BLISS"
MAUDE_VER=$("$MAUDE" --version 2>&1 | head -1)
BLISS_VER=$("$BLISS" -version 2>&1 | head -1)
command -v python3 >/dev/null || { echo "canon_pin: python3 not found" >&2; exit 2; }
# oom_prologue's address-space cap is set after the cargo builds (below),
# so it only ever applies to explorer runs.

mkdir -p "$CACHE_DIR"
exec 9>"$CACHE_DIR/.lock"
flock -n 9 || { echo "canon_pin: another canon_pin run holds $CACHE_DIR/.lock" >&2; exit 2; }

WORK=$(mktemp -d "${TMPDIR:-/tmp}/canon_pin.XXXXXX")
trap 'rm -rf "$WORK"' EXIT
export WORK

EXAMPLE=explore_canonical_matches

# --- builds ------------------------------------------------------------------------
build_branch() {
    echo "canon_pin: building the working tree's explorer" >&2
    ( cd "$ROOT" && cargo build --release --example "$EXAMPLE" 2>&1 | tail -3 ) >&2
    BRANCH_BIN="${CARGO_TARGET_DIR:-$ROOT/target}/release/examples/$EXAMPLE"
    [ -x "$BRANCH_BIN" ] || { echo "canon_pin: working-tree build failed ($BRANCH_BIN missing)" >&2; exit 2; }
}

# build_base: sets BASE_SHA and BASE_BIN, building BASE's explorer unless a
# binary for that commit is already kept.
build_base() {
    local ref="${BASE:-}" wt="$CACHE_DIR/base-src" pin
    # Not `${BASE:-@{upstream}}`: the expansion ends at the first `}`.
    [ -n "$ref" ] || ref='@{upstream}'
    BASE_SHA=$(git -C "$ROOT" rev-parse --verify --quiet "$ref^{commit}") || {
        echo "canon_pin: BASE '$ref' does not name a commit (set BASE=<ref>)" >&2; exit 2; }
    BASE_BIN="$CACHE_DIR/bin/$BASE_SHA/$EXAMPLE"
    echo "canon_pin: BASE $ref = $BASE_SHA" >&2
    if [ -x "$BASE_BIN" ]; then return 0; fi
    git -C "$ROOT" worktree prune
    if [ -e "$wt/.git" ]; then
        git -C "$wt" checkout -q --force --detach "$BASE_SHA" || exit 2
    else
        rm -rf "$wt"
        git -C "$ROOT" worktree add -q --detach "$wt" "$BASE_SHA" || exit 2
    fi
    # The build include_str!s files from the tamarin-prover submodule: check
    # out the commit BASE pins, as a worktree of the local submodule clone.
    pin=$(git -C "$ROOT" ls-tree "$BASE_SHA" tamarin-prover | awk '{print $3}')
    [ -n "$pin" ] || { echo "canon_pin: BASE $BASE_SHA pins no tamarin-prover submodule" >&2; exit 2; }
    git -C "$ROOT/tamarin-prover" cat-file -e "$pin^{commit}" 2>/dev/null || {
        echo "canon_pin: submodule commit $pin (pinned by BASE) is not in $ROOT/tamarin-prover;" \
             "fetch it (git -C tamarin-prover fetch)" >&2; exit 2; }
    git -C "$ROOT/tamarin-prover" worktree prune
    if [ -e "$wt/tamarin-prover/.git" ]; then
        git -C "$wt/tamarin-prover" checkout -q --force --detach "$pin" || exit 2
    else
        rm -rf "$wt/tamarin-prover"
        git -C "$ROOT/tamarin-prover" worktree add -q --detach "$wt/tamarin-prover" "$pin" || exit 2
    fi
    echo "canon_pin: building BASE's explorer in $wt" >&2
    ( cd "$wt" && CARGO_TARGET_DIR="$CACHE_DIR/base-target" \
        cargo build --release --example "$EXAMPLE" 2>&1 | tail -3 ) >&2
    [ -x "$CACHE_DIR/base-target/release/examples/$EXAMPLE" ] || {
        echo "canon_pin: BASE build failed" >&2; exit 2; }
    mkdir -p "$(dirname "$BASE_BIN")"
    cp "$CACHE_DIR/base-target/release/examples/$EXAMPLE" "$BASE_BIN.tmp" && mv "$BASE_BIN.tmp" "$BASE_BIN"
}

# interface_check <bin> <label>: the explorer takes the load flags and lists
# a theory. An explorer from before this gate takes neither.
interface_check() {
    "$1" --list-lemmas -D=canon_pin_probe --auto-sources "$CORPUS/Tutorial.spthy" >/dev/null 2>&1 || {
        echo "canon_pin: the $2 explorer ($1) predates the pin interface (-D/--auto-sources," \
             "schema 4); set BASE to a commit that contains scripts/canon_pin.sh" >&2
        exit 2; }
}

# --- jobs ----------------------------------------------------------------------------
# explorer_args <rel>: the explorer's load flags from file_flags.tsv (the
# prover's -D=FLAG and --auto-sources; --stop-on-trace does not apply to the
# explorer), `@cd` kept as a marker, `--diff` reported as DIFF.
explorer_args() {
    local fl tok out=""
    fl=$(flags_for "$1")
    for tok in $fl; do
        case "$tok" in
            --diff) echo DIFF; return ;;
            -D=*|--auto-sources|@cd) out="$out $tok" ;;
        esac
    done
    echo "${out# }"
}

# ikey <rel> <file>: theory sha + include shas + flags (rs_ref_check.sh's ikey).
ikey() {
    local h fl inc; h=$(sha256sum "$2" | cut -d' ' -f1); fl=$(flags_for "$1")
    inc=$(include_shas "$2")
    if [ -n "$inc" ]; then h="${h}__i$(printf '%s' "$inc" | sha256sum | cut -c1-12)"; fi
    if [ -n "$fl" ]; then h="${h}__f$(printf '%s' "$fl" | sha256sum | cut -c1-12)"; fi
    printf '%s' "$h"
}

# run_explorer <bin> <rel> <lemma> <out.json> <log> <extra args...>: one
# explorer run from the right directory with the right flags, no debug env.
# Prints "<status>\t<secs>": OK, EXIT=<n> or TIMEOUT. The explorer writes its
# JSON for exit 0, 4 (stopped by a panic) and 5 (size invariant).
run_explorer() {
    local bin="$1" rel="$2" lemma="$3" out="$4" log="$5" f="$CORPUS/$2" args rundir="" farg t0 rc
    shift 5
    args=$(explorer_args "$rel"); farg="$f"
    if [[ " $args " == *" @cd "* ]]; then args=${args//@cd/}; rundir=$(dirname "$f"); farg=$(basename "$f"); fi
    t0=$(date +%s.%N)
    ( [ -n "$rundir" ] && cd "$rundir"
      exec env -u PROGRESS -u DUMP_FORMULAS -u PROFILE -u PROFILE_CANON -u DUMP_AUTOMORPHISMS \
          -u DUMP_DUMMY_SWAPS -u DUMP_DIMACS_AT -u DUMP_CANON_PANIC -u TAM_ALLOW_NO_BLISS \
          timeout "$TIMEOUT" "$bin" "$farg" "$lemma" $args --max-depth "$DEPTH" \
          --max-nodes "$NODES" --out "$out" "$@" ) >"$log" 2>&1
    rc=$?
    local secs; secs=$(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN {printf "%.2f", b - a}')
    case "$rc" in
        0) printf 'OK\t%s\n' "$secs" ;;
        124) printf 'TIMEOUT\t%s\n' "$secs" ;;
        *) printf 'EXIT=%s\t%s\n' "$rc" "$secs" ;;
    esac
}

# lemmas_of <bin> <rel>: "OK <lemma>...", "DIFF" or "ERROR <message>".
lemmas_of() {
    local args f="$CORPUS/$2" rundir="" farg json
    args=$(explorer_args "$2"); farg="$f"
    [ "$args" = DIFF ] && { echo DIFF; return; }
    args=${args//--auto-sources/}
    if [[ " $args " == *" @cd "* ]]; then args=${args//@cd/}; rundir=$(dirname "$f"); farg=$(basename "$f"); fi
    json=$( ( [ -n "$rundir" ] && cd "$rundir"; timeout 300 "$1" --list-lemmas $args "$farg" 2>/dev/null ) )
    printf '%s' "$json" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception as e:
    print("ERROR --list-lemmas printed no JSON"); sys.exit()
if d.get("diff"):
    print("DIFF")
elif d.get("error"):
    print("ERROR " + " ".join(str(d["error"]).split())[:200])
else:
    print("OK " + " ".join(l["name"] for l in d.get("lemmas", [])))'
}
export -f run_explorer explorer_args flags_for lemmas_of ikey include_shas

# job_key <rel> <lemma>: the cache key of the BASE result.
job_key() {
    printf '%s|%s|%s|d%s|n%s|%s|%s|rayon%s|schema4' "$BASE_SHA" "$(ikey "$1" "$CORPUS/$1")" "$2" \
        "$DEPTH" "$NODES" "$MAUDE_VER" "$BLISS_VER" "$RAYON_NUM_THREADS" | sha256sum | cut -c1-32
}

# list_rows <file>: "rel<TAB>lemma..." per non-comment row (inline `#`
# comments dropped).
list_rows() {
    sed -e 's/#.*$//' -e 's/[[:space:]]\+$//' "$1" | grep . | awk '{rel = $1; $1 = ""; sub(/^ /, ""); print rel "\t" $0}'
}

# expand_jobs <list> <bin...>: the (rel, lemma) jobs into $WORK/jobs.tsv,
# left-out rows into $WORK/skipped.tsv, failure rows into $WORK/pre.tsv.
# Rows without lemmas are listed with every given binary; the first one's
# lemma list is used, and a different list from another is a LEMMA_SET_DIFF.
expand_jobs() {
    local list="$1"; shift
    : > "$WORK/jobs.tsv"; : > "$WORK/skipped.tsv"; : > "$WORK/pre.tsv"; : > "$WORK/expand.txt"
    local rel lemmas l
    while IFS=$'\t' read -r rel lemmas; do
        if [ ! -f "$CORPUS/$rel" ]; then printf '%s\tno such file\n' "$rel" >> "$WORK/skipped.tsv"; continue; fi
        if [ "$(explorer_args "$rel")" = DIFF ]; then printf '%s\tdiff theory (no diff-mode prover)\n' "$rel" >> "$WORK/skipped.tsv"; continue; fi
        if [ -z "$lemmas" ]; then printf '%s\n' "$rel" >> "$WORK/expand.txt"; continue; fi
        for l in $lemmas; do printf '%s\t%s\n' "$rel" "$l" >> "$WORK/jobs.tsv"; done
    done < <(list_rows "$list" | sort -u)
    [ -s "$WORK/expand.txt" ] || return 0
    echo "canon_pin: listing the lemmas of $(grep -c . "$WORK/expand.txt") theories" >&2
    local i=0 bin
    for bin in "$@"; do
        export LIST_BIN="$bin"
        xargs -d '\n' -P "$JOBS" -I{} bash -c 'printf "%s\t%s\n" "$1" "$(lemmas_of "$LIST_BIN" "$1")"' _ {} \
            < "$WORK/expand.txt" | sort > "$WORK/listed.$i"
        i=$((i + 1))
    done
    local first rest other
    while IFS=$'\t' read -r rel first; do
        case "$first" in
            DIFF) printf '%s\tdiff theory (no diff-mode prover)\n' "$rel" >> "$WORK/skipped.tsv"; continue ;;
            ERROR*) printf '%s\t-\tNOT_COMPARED\tbase --list-lemmas: %s\n' "$rel" "${first#ERROR }" >> "$WORK/pre.tsv"; continue ;;
        esac
        rest=${first#OK}; rest=${rest# }
        if [ "$i" -gt 1 ]; then
            other=$(awk -F'\t' -v r="$rel" '$1 == r {print $2; exit}' "$WORK/listed.1")
            if [ "$other" != "$first" ]; then
                printf '%s\t-\tLEMMA_SET_DIFF\tbase: %s | branch: %s\n' "$rel" "${first:0:200}" "${other:0:200}" >> "$WORK/pre.tsv"
                continue
            fi
        fi
        [ -n "$rest" ] || { printf '%s\tno lemmas\n' "$rel" >> "$WORK/skipped.tsv"; continue; }
        for l in $rest; do printf '%s\t%s\n' "$rel" "$l" >> "$WORK/jobs.tsv"; done
    done < "$WORK/listed.0"
}

print_skipped() {
    local s; s=$(grep -c . "$WORK/skipped.tsv")
    [ "$s" = 0 ] && return 0
    echo "canon_pin: $s theory row(s) left out:"
    awk -F'\t' '{c[$2]++} END {for (k in c) printf "    %4d  %s\n", c[k], k}' "$WORK/skipped.tsv"
}

# progress <total> <ok-status-regex>: a live counter on stderr, rows that
# need attention printed as they arrive; stdin passes through.
progress() {
    awk -F'\t' -v t="$1" -v ok="$2" '{
        print; fflush()
        n++
        if ($3 !~ ok) printf "\r  %-17s %s %s  %s\033[K\n", $3, $1, $2, $4 > "/dev/stderr"
        printf "\r  [%d/%d] %s %s\033[K", n, t, $1, $2 > "/dev/stderr"
        fflush("/dev/stderr")
      } END { print "" > "/dev/stderr" }'
}

# --- calibrate ---------------------------------------------------------------------------
if [ "$MODE" = calibrate ]; then
    build_branch
    interface_check "$BRANCH_BIN" "working tree"
    oom_prologue
    CALIB="${CALIB:-$CACHE_DIR/calibrate_${TIER}_d${DEPTH}_n${NODES}.tsv}"
    if [ -s "$CALIB" ]; then
        echo "canon_pin: reusing the timings in $CALIB (delete it to re-time)"
    else
        [ -f "$CANDIDATES" ] || { echo "canon_pin: CANDIDATES '$CANDIDATES' does not exist" >&2; exit 2; }
        expand_jobs "$CANDIDATES" "$BRANCH_BIN"
        print_skipped
        if [ -s "$WORK/pre.tsv" ]; then
            echo "canon_pin: $(grep -c . "$WORK/pre.tsv") theory row(s) left out, --list-lemmas failed:"
            awk -F'\t' '{print "    " $1 "\t" $4}' "$WORK/pre.tsv"
        fi
        TOTAL=$(grep -c . "$WORK/jobs.tsv")
        echo "canon_pin: timing $TOTAL lemma(s) at depth $DEPTH, $NODES nodes, TIMEOUT=${TIMEOUT}s, JOBS=$JOBS"
        export BRANCH_BIN
        calib_one() {
            local rel="${1%%$'\t'*}" lemma="${1#*$'\t'}" d st secs info
            d=$(mktemp -d "$WORK/c.XXXXXX")
            IFS=$'\t' read -r st secs < <(run_explorer "$BRANCH_BIN" "$rel" "$lemma" "$d/out.json" "$d/log")
            info=$'0\t-'
            if [ -s "$d/out.json" ]; then
                info=$(python3 -c 'import json, sys
d = json.load(open(sys.argv[1]))
print(str(d["sizes"]["graph"]) + "\t" + d["stop_reason"])' "$d/out.json")
                [ "$st" = OK ] || st="EXIT_JSON"
            fi
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$rel" "$lemma" "$st" "$secs" "$info" "$(flags_for "$rel")"
            rm -rf "$d"
        }
        export -f calib_one
        { echo "# rel	lemma	status	secs	graph	stop_reason	flags (depth $DEPTH, nodes $NODES, JOBS $JOBS, $(hostname), $(date -I))"
          xargs -d '\n' -P "$JOBS" -I{} bash -c 'calib_one "$1"' _ {} < "$WORK/jobs.tsv" \
              | awk -F'\t' -v t="$TOTAL" '{print; fflush(); n++; printf "\r  [%d/%d] %-8s %6ss %s %s\033[K", n, t, $3, $4, $1, $2 > "/dev/stderr"} END {print "" > "/dev/stderr"}'
        } > "$CALIB.tmp" && mv "$CALIB.tmp" "$CALIB"
    fi
    budget=$(( TARGET_MIN * 60 * JOBS ))
    { echo "# canon_pin $TIER tier: \`relpath [lemma...]\` rows; inline \`#\` comments are ignored."
      echo "# Generated by: scripts/canon_pin.sh calibrate $TIER ($(hostname), $(date -I)),"
      echo "# depth $DEPTH, $NODES nodes, TIMEOUT ${TIMEOUT}s; budget ${TARGET_MIN} min warm at JOBS=$JOBS."
      echo "# Candidates: ${CANDIDATES#$ROOT/}. Edit freely; re-run calibrate to regenerate."
      python3 "$PY" select "$TIER" "$CALIB" "$CORPUS" "$budget"
    } > "$LIST.tmp" && mv "$LIST.tmp" "$LIST"
    sed -n '1,/^[^#]/p' "$LIST" | grep '^#' | sed -n '5,40p'
    echo "canon_pin: wrote $(grep -vc '^#' "$LIST") row(s) to $LIST"
    echo "DONE_CANON_PIN_CALIBRATE tier=$TIER rows=$(grep -vc '^#' "$LIST")"
    exit 0
fi

# --- fast / full -----------------------------------------------------------------------------
[ -f "$LIST" ] || { echo "canon_pin: LIST '$LIST' does not exist (run: $0 calibrate $TIER)" >&2; exit 2; }
build_base
build_branch
interface_check "$BASE_BIN" "BASE"
interface_check "$BRANCH_BIN" "working tree"
if [ "$BASE_SHA" = "$(git -C "$ROOT" rev-parse HEAD)" ] && [ -z "$(git -C "$ROOT" status --porcelain --untracked-files=no -- crates Cargo.toml Cargo.lock)" ]; then
    echo "canon_pin: BASE is HEAD and the tree is clean: this run checks determinism (expect all SAME)"
fi
export BASE_BIN BRANCH_BIN BASE_SHA MAUDE_VER BLISS_VER
oom_prologue

expand_jobs "$LIST" "$BASE_BIN" "$BRANCH_BIN"
print_skipped
while IFS=$'\t' read -r rel lemma; do
    printf '%s\t%s\t%s\n' "$rel" "$lemma" "$(job_key "$rel" "$lemma")"
done < <(sort -u "$WORK/jobs.tsv") > "$WORK/keyed.tsv"
TOTAL=$(grep -c . "$WORK/keyed.tsv")
[ "$TOTAL" -gt 0 ] || { echo "canon_pin: no jobs to run" >&2; exit 2; }
RES="$CACHE_DIR/results"
mkdir -p "$RES" "$DIFF_DIR"
export RES
# DIFF_DIR is emptied for every run: only a directory this script made.
if [ -n "$(ls -A "$DIFF_DIR")" ] && [ ! -e "$DIFF_DIR/.canon_pin_diffs" ]; then
    echo "canon_pin: DIFF_DIR '$DIFF_DIR' is not empty and not a canon_pin diff directory" >&2; exit 2
fi
touch "$DIFF_DIR/.canon_pin_diffs"

# Phase A: the BASE explorer for every job not cached yet. The status file is
# written last: a job without one (a killed run) is simply run again.
base_job() {
    local rel lemma key d st secs
    IFS=$'\t' read -r rel lemma key <<< "$1"
    d=$(mktemp -d "$WORK/a.XXXXXX")
    IFS=$'\t' read -r st secs < <(run_explorer "$BASE_BIN" "$rel" "$lemma" "$d/out.json" "$d/log")
    if [ -s "$d/out.json" ]; then
        python3 "$PY" normalize "$d/out.json" "$RES/$key.json.gz" || st="NORMALIZE_FAILED"
    fi
    tail -n 200 "$d/log" | gzip > "$RES/$key.log.gz"
    printf '%s\t%s\n' "$st" "$secs" > "$RES/$key.status"
    printf '%s\t%s\t%s\t%s\n' "$rel" "$lemma" "$st" "${secs}s"
    rm -rf "$d"
}
export -f base_job
awk -F'\t' -v res="$RES" '{ if ((getline x < (res "/" $3 ".status")) <= 0) print; close(res "/" $3 ".status") }' \
    "$WORK/keyed.tsv" > "$WORK/missing.tsv"
MISSING=$(grep -c . "$WORK/missing.tsv")
echo "canon_pin: $TIER — $TOTAL job(s), depth $DEPTH, $NODES nodes, TIMEOUT=${TIMEOUT}s, JOBS=$JOBS"
echo "canon_pin: phase A (BASE ${BASE_SHA:0:12}): $((TOTAL - MISSING)) cached, $MISSING to run"
if [ "$MISSING" -gt 0 ]; then
    xargs -d '\n' -P "$JOBS" -I{} bash -c 'base_job "$1"' _ {} < "$WORK/missing.tsv" \
        | progress "$MISSING" '^OK$' > /dev/null
fi

# Phase B: the working tree's explorer, compared against the cache.
branch_job() {
    local rel lemma key d bst bsecs st secs row slug
    IFS=$'\t' read -r rel lemma key <<< "$1"
    bst="(no base run)"; bsecs="?"
    [ -f "$RES/$key.status" ] && IFS=$'\t' read -r bst bsecs < "$RES/$key.status"
    if [ ! -s "$RES/$key.json.gz" ]; then
        printf '%s\t%s\tNOT_COMPARED\tbase %s after %ss\n' "$rel" "$lemma" "$bst" "$bsecs"; return 0
    fi
    d=$(mktemp -d "$WORK/b.XXXXXX")
    IFS=$'\t' read -r st secs < <(run_explorer "$BRANCH_BIN" "$rel" "$lemma" "$d/out.json" "$d/log")
    if [ "$st" = TIMEOUT ]; then
        row="BRANCH_TIMEOUT"$'\t'"base took ${bsecs}s, branch killed after ${TIMEOUT}s"
    elif [ "$st" != "$bst" ]; then
        row="STATUS_DIFF"$'\t'"base $bst, branch $st: $(grep -m1 -iE 'panicked|failed|error' "$d/log" | cut -c1-160)"
    elif ! python3 "$PY" normalize "$d/out.json" "$d/branch.json.gz" 2>"$d/norm.err"; then
        row="STATUS_DIFF"$'\t'"branch JSON unreadable: $(head -c 160 "$d/norm.err")"
    else
        row=$(python3 "$PY" compare "$RES/$key.json.gz" "$d/branch.json.gz")
    fi
    case "${row%%$'\t'*}" in
        SAME|FP_ONLY) ;;
        *)  slug="${rel//\//__}__$lemma"
            mkdir -p "$DIFF_DIR/$slug"
            cp "$RES/$key.json.gz" "$DIFF_DIR/$slug/base.json.gz"
            [ -s "$d/branch.json.gz" ] && cp "$d/branch.json.gz" "$DIFF_DIR/$slug/branch.json.gz"
            cp "$RES/$key.log.gz" "$DIFF_DIR/$slug/base.log.gz"
            tail -n 200 "$d/log" | gzip > "$DIFF_DIR/$slug/branch.log.gz" ;;
    esac
    printf '%s\t%s\t%s\n' "$rel" "$lemma" "$row"
    rm -rf "$d"
}
export -f branch_job
find "$DIFF_DIR" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +
echo "canon_pin: phase B (working tree)"
: > "$OUT"
xargs -d '\n' -P "$JOBS" -I{} bash -c 'branch_job "$1"' _ {} < "$WORK/keyed.tsv" \
    | progress "$TOTAL" '^(SAME|FP_ONLY)$' > "$WORK/rows.tsv"
cat "$WORK/pre.tsv" "$WORK/rows.tsv" | sort > "$OUT"

# --- report -------------------------------------------------------------------------------------
echo "=== SUMMARY ==="
awk -F'\t' '{c[$3]++} END {for (k in c) printf "  %-17s %d\n", k, c[k]}' "$OUT" | sort
PASS='^(SAME|FP_ONLY|NOT_COMPARED)$'
echo "=== needs attention ==="
awk -F'\t' -v ok="$PASS" '$3 !~ ok {print "  " $3 "\t" $1 " " $2 "\t" $4}' "$OUT"
if grep -q $'\tNOT_COMPARED\t' "$OUT"; then
    echo "=== not compared (the BASE produced no graph) ==="
    awk -F'\t' '$3 == "NOT_COMPARED" {print "  " $1 " " $2 "\t" $4}' "$OUT"
fi
echo "  results: $OUT"
if [ -n "$(find "$DIFF_DIR" -mindepth 1 -maxdepth 1 -type d)" ]; then
    echo "  diffs:   $DIFF_DIR (base/branch JSON and logs per job)"
fi
fails=$(awk -F'\t' -v ok="$PASS" '$3 !~ ok' "$OUT" | grep -c .)
fp=$(grep -c $'\tFP_ONLY\t' "$OUT")
nc=$(grep -c $'\tNOT_COMPARED\t' "$OUT")
# A job with no row at all (a killed xargs child) compared nothing.
norow=$(awk -F'\t' 'FILENAME == ARGV[1] {seen[$1 "\t" $2] = 1; next} !(($1 "\t" $2) in seen)' \
            "$WORK/rows.tsv" "$WORK/keyed.tsv" | grep -c .)
bad=''
[ "$fails" = 0 ] || bad="FAIL=$fails"
[ "$norow" = 0 ] || bad="${bad:+$bad }NOROW=$norow"
tier_uc=$(printf '%s' "$TIER" | tr a-z A-Z)
echo "DONE_CANON_PIN_$tier_uc verdict=${bad:-OK} files=$(cut -f1 "$OUT" | sort -u | grep -c .) lemmas=$TOTAL" \
     "fp_only=$fp not_compared=$nc base=${BASE_SHA:0:12} depth=$DEPTH nodes=$NODES"
[ -z "$bad" ]
