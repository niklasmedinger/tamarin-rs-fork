#!/usr/bin/env bash
# Verification gates for the opt-in top-N proof search (TAM_RS_TOP_METHODS,
# crates/tamarin-theory/src/constraint/solver/topn_search.rs).
#
#   scripts/topn_gate.sh default   The default output is unchanged: with no top-N
#                                  variable set, BIN must reproduce the committed
#                                  reference (scripts/rs_ref_check.sh check).
#   scripts/topn_gate.sh top1      N = 1 is the greedy prover: the same binary, run
#                                  once plain and once with TAM_RS_TOP_METHODS=1,
#                                  compared lemma by lemma (see "top1 verdicts").
#   scripts/topn_gate.sh replay    The proofs the top-N search prints are proofs:
#                                  each theory is proved with top-N and written out
#                                  (--output), then the same binary checks the
#                                  written theory WITHOUT top-N and without --prove,
#                                  replaying every stored step with the solver.
#
# Every mode runs over the fast parity corpus by default and ends in a
# `DONE_TOPN_<MODE> verdict=... files=N` line; the exit status is the verdict.
#
# Files a mode cannot compare are left out up front and listed, never silently
# dropped:
#   - SAPIC theories when merging is on: `config_for_lemma` refuses them, the
#     canonicalizer's color table does not cover actions generated from processes;
#   - `--stop-on-trace` files in `top1`: the top-N search ignores the cut
#     strategy, so its N = 1 run is the default Dfs search, not the one asked for;
#   - `--diff` files in `replay`: the port has no diff-mode prover.
#
# top1 verdicts. Each file gets one row: SAME (identical stripped output) or
# the worst kind of difference among its lemmas —
#   DIFF_VERDICT  a lemma's verdict differs: always a failure;
#   DIFF_OTHER    output outside the lemma proofs differs: always a failure;
#   DIFF_PROOF    a proof without a trace differs: a failure with MERGE=0; with
#                 MERGE=1 expected, since a merged class applies its
#                 representative's first method to every system in it, where
#                 ranking would have used goal numbers the canonical form drops;
#   DIFF_TRACE    only the proof of a lemma with a trace (exists-trace verified,
#                 all-traces falsified) differs: expected, the search reports a
#                 different trace than greedy's depth-bucketed leftmost one.
#
# Env (all modes): BIN (default target/release/tamarin-rs), ALLOWLIST (default
#   scripts/parity_corpus_fast.txt), FLAGS_MAP, CORPUS, JOBS, TIMEOUT, DERIV,
#   OUT (result TSV).
# top1:   MERGE (1 = TAM_RS_MERGE on, the default; 0 = off), BATCH (default 8).
#         B = 1 is the exact configuration: batches of more entries expand
#         sibling cases concurrently and can settle a different trace first.
# replay: N (default 3), MERGE (default 1), BATCH (default 8), ORDER (iddfs|bfs,
#   default iddfs), DEADLINE_MS (per-lemma search deadline, default 60000; a
#   lemma cut by it is still replayed, as the incomplete proof it is), KEEP (a
#   directory: keep every proved/checked theory there).
set -u
export LC_ALL=C

usage() {
    echo "usage: $0 default|top1|replay" >&2
}
MODE="${1:-}"
case "$MODE" in default|top1|replay) shift ;; *) usage; exit 2 ;; esac
[ $# -eq 0 ] || { echo "topn_gate: unexpected argument '$1' (settings go in env vars)" >&2; usage; exit 2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -r "$HERE/gate_common.sh" ] || { echo "topn_gate: missing $HERE/gate_common.sh" >&2; exit 2; }
. "$HERE/gate_common.sh"

ROOT="${ROOT:-$(cd "$HERE/.." && pwd)}"
CORPUS="${CORPUS:-$ROOT/tamarin-prover/examples}"
BIN="${BIN:-$ROOT/target/release/tamarin-rs}"
ALLOWLIST="${ALLOWLIST:-$ROOT/scripts/parity_corpus_fast.txt}"
FLAGS_MAP="${FLAGS_MAP:-$ROOT/scripts/file_flags.tsv}"
export CORPUS FLAGS_MAP

[ -x "$BIN" ] || { echo "topn_gate: no executable prover at '$BIN' (set BIN=)" >&2; exit 2; }
[ -f "$ALLOWLIST" ] || { echo "topn_gate: ALLOWLIST '$ALLOWLIST' does not exist" >&2; exit 2; }
BIN="$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")"

# The variables that switch the top-N search on. A gate run that inherits one
# of them from the caller's shell would test something other than it claims.
TOPN_VARS=(TAM_RS_TOP_METHODS TAM_RS_MERGE TAM_RS_TOPN_BATCH TAM_RS_SEARCH_ORDER TAM_RS_TOPN_ALT_COST TAM_RS_TOPN_STATS)
for v in "${TOPN_VARS[@]}"; do
    if [ -n "${!v+x}" ]; then
        echo "topn_gate: $v is set in the environment; unset it (the gate sets what each run needs)" >&2
        exit 2
    fi
done

WORK=$(mktemp -d "${TMPDIR:-/tmp}/topn_gate.XXXXXX")
trap 'rm -rf "$WORK"' EXIT

# is_sapic <relpath>: the theory has a top-level process (what `Theory::is_sapic`
# records from the parser: exactly one `process:` item).
is_sapic() { grep -qE '^[[:space:]]*process[[:space:]]*:' "$CORPUS/$1"; }

# select_files <mode>: the allowlist minus what <mode> cannot compare, into
# $WORK/files.txt; the left-out files and the reason go to $WORK/skipped.tsv.
select_files() {
    local mode="$1" rel fl
    : > "$WORK/files.txt"; : > "$WORK/skipped.tsv"
    while IFS= read -r rel; do
        fl=$(flags_for "$rel")
        if [ "$mode" = top1 ] && [[ " $fl " == *" --stop-on-trace"* ]]; then
            printf '%s\tstop-on-trace (top-N ignores the cut strategy)\n' "$rel" >> "$WORK/skipped.tsv"; continue
        fi
        if [ "$mode" = replay ] && [[ " $fl " == *" --diff "* ]]; then
            printf '%s\tdiff mode (no diff-mode prover)\n' "$rel" >> "$WORK/skipped.tsv"; continue
        fi
        if [ "${MERGE:-1}" = 1 ] && [ -f "$CORPUS/$rel" ] && is_sapic "$rel"; then
            printf '%s\tSAPIC (merging refuses theories with processes)\n' "$rel" >> "$WORK/skipped.tsv"; continue
        fi
        printf '%s\n' "$rel" >> "$WORK/files.txt"
    done < <(grep -v '^[[:space:]]*#' "$ALLOWLIST" | grep . | sort -u)
    local n s; n=$(grep -c . "$WORK/files.txt"); s=$(grep -c . "$WORK/skipped.tsv")
    echo "topn_gate: $mode — $n file(s) to compare, $s left out:"
    awk -F'\t' '{c[$2]++} END {for (k in c) printf "    %4d  %s\n", c[k], k}' "$WORK/skipped.tsv"
    [ "$n" -gt 0 ] || { echo "topn_gate: nothing left to compare" >&2; exit 2; }
}

# wrapper <path> [env assignments...]: an executable that runs BIN with exactly
# these top-N variables set, and no search deadline unless one is among them
# (a deadline makes output depend on machine speed).
wrapper() {
    local path="$1"; shift
    {
        echo '#!/usr/bin/env bash'
        printf 'unset %s TAM_PROVE_DEADLINE_MS\n' "${TOPN_VARS[*]}"
        for a in "$@"; do printf 'export %q\n' "$a"; done
        printf 'exec %q "$@"\n' "$BIN"
    } > "$path"
    chmod +x "$path"
}

case "$MODE" in
default)
    # rs_ref_check runs BIN itself; the guard above already made sure no top-N
    # variable reaches it.
    BIN="$BIN" ALLOWLIST="$ALLOWLIST" "$HERE/rs_ref_check.sh" check
    rc=$?
    echo "DONE_TOPN_DEFAULT verdict=$([ "$rc" = 0 ] && echo OK || echo FAIL) files=$(grep -v '^[[:space:]]*#' "$ALLOWLIST" | grep -c .)"
    exit "$rc"
    ;;

top1)
    MERGE="${MERGE:-1}"; BATCH="${BATCH:-8}"; TIMEOUT="${TIMEOUT:-600}"; DERIV="${DERIV:-30}"
    cores=$(nproc 2>/dev/null || echo 4)
    JOBS="${JOBS:-$(( cores >= 24 ? 3 : cores >= 12 ? 2 : 1 ))}"
    OUT="${OUT:-/tmp/topn_gate_top1.tsv}"
    select_files top1
    MAUDE=$(resolve_maude) || exit 2
    maude_on_path "$MAUDE"
    oom_prologue
    wrapper "$WORK/greedy"
    top_env=("TAM_RS_TOP_METHODS=1" "TAM_RS_TOPN_BATCH=$BATCH")
    [ "$MERGE" = 1 ] && top_env+=("TAM_RS_MERGE=1")
    wrapper "$WORK/top1" "${top_env[@]}"
    export WORK TIMEOUT DERIV
    echo "topn_gate: top1 — ${top_env[*]} against the plain prover;" \
         "JOBS=$JOBS TIMEOUT=${TIMEOUT}s OUT=$OUT"

    # compare_lemmas <greedy> <top1>: both stripped outputs, split into the
    # text before the first lemma, one section per lemma, the text after the
    # last one and the summary block; prints "<status>\t<details>".
    compare_lemmas() {
        awk '
            FNR == 1 { f++; sect = "(before the lemmas)"; insum = 0 }
            /^==========/ { insum = 1 }
            insum {
                if (match($0, /^  [^ ]+ \((all-traces|exists-trace)\): /)) {
                    name = $1
                    rest = $0; sub(/^  [^ ]+ \(/, "", rest)
                    quant = rest; sub(/\).*/, "", quant)
                    verdict = rest; sub(/^[^:]*: /, "", verdict); sub(/ \([0-9]+ steps\)$/, "", verdict)
                    v[f, name] = verdict; q[name] = quant
                } else if ($0 !~ /^==========/) {
                    text[f, "(summary)"] = text[f, "(summary)"] $0 "\n"; sects["(summary)"] = 1
                }
                next
            }
            /^lemma / { sect = $2; sub(/[:\[].*$/, "", sect); lemma[sect] = 1 }
            /^end$/ { sect = "(after the lemmas)" }
            { text[f, sect] = text[f, sect] $0 "\n"; sects[sect] = 1 }
            END {
                rank["DIFF_TRACE"] = 1; rank["DIFF_PROOF"] = 2; rank["DIFF_OTHER"] = 3; rank["DIFF_VERDICT"] = 4
                worst = ""; details = ""
                for (s in sects) {
                    if (text[1, s] == text[2, s]) continue
                    if (s in lemma) {
                        if (v[1, s] != v[2, s]) k = "DIFF_VERDICT"
                        else if ((q[s] == "exists-trace" && v[1, s] == "verified") || v[1, s] ~ /found trace/) k = "DIFF_TRACE"
                        else k = "DIFF_PROOF"
                    } else k = "DIFF_OTHER"
                    details = details s ":" substr(k, 6) " "
                    if (worst == "" || rank[k] > rank[worst]) worst = k
                }
                for (key in v) {
                    split(key, part, SUBSEP)
                    if (part[1] == 1 && v[1, part[2]] != v[2, part[2]] && !(part[2] SUBSEP "x" in seen)) {
                        seen[part[2] SUBSEP "x"] = 1
                        details = details part[2] ":VERDICT[" v[1, part[2]] "|" v[2, part[2]] "] "
                        worst = "DIFF_VERDICT"
                    }
                }
                if (worst == "") { worst = "DIFF_OTHER"; details = "(whitespace or ordering) " }
                printf "%s\t%s\n", worst, details
            }' "$1" "$2"
    }
    # one <rel> -> "rel \t status \t detail"
    one() {
        local rel="$1" f="$CORPUS/$1" fl rundir="" farg d rg rt
        fl=$(flags_for "$rel"); farg="$f"
        if [[ " $fl " == *" @cd "* ]]; then fl=${fl//@cd/}; rundir=$(dirname "$f"); farg=$(basename "$f"); fi
        d=$(mktemp -d "$WORK/one.XXXXXX")
        ( [ -n "$rundir" ] && cd "$rundir"
          timeout "$TIMEOUT" "$WORK/greedy" $fl --derivcheck-timeout="$DERIV" --prove "$farg" ) >"$d/g.raw" 2>/dev/null; rg=$?
        ( [ -n "$rundir" ] && cd "$rundir"
          timeout "$TIMEOUT" "$WORK/top1" $fl --derivcheck-timeout="$DERIV" --prove "$farg" ) >"$d/t.raw" 2>"$d/t.err"; rt=$?
        if [ "$rg" = 124 ] || [ "$rt" = 124 ]; then
            printf '%s\tTIMEOUT\tgreedy=%s top1=%s\n' "$rel" "$rg" "$rt"; rm -rf "$d"; return 0
        fi
        if [ "$rg" != 0 ] || [ "$rt" != 0 ]; then
            printf '%s\tERROR\tgreedy=%s top1=%s %s\n' "$rel" "$rg" "$rt" "$(grep -m1 panicked "$d/t.err" | cut -c1-120)"
            rm -rf "$d"; return 0
        fi
        strip_env < "$d/g.raw" > "$d/g"; strip_env < "$d/t.raw" > "$d/t"
        if cmp -s "$d/g" "$d/t"; then printf '%s\tSAME\t-\n' "$rel"
        else printf '%s\t%s\n' "$rel" "$(compare_lemmas "$d/g" "$d/t")"; fi
        rm -rf "$d"
    }
    export -f one compare_lemmas flags_for strip_env

    mapfile -t FILES < "$WORK/files.txt"
    TOTAL=${#FILES[@]}
    : > "$OUT"
    printf '%s\n' "${FILES[@]}" \
        | xargs -P "$JOBS" -I{} bash -c 'one "$@"' _ {} \
        | tee -a "$OUT" \
        | awk -v t="$TOTAL" '{
            n++
            if ($2 != "SAME") printf "\r  %-13s %s  %s\033[K\n", $2, $1, $3 > "/dev/stderr"
            printf "\r  [%d/%d] %s\033[K", n, t, $1 > "/dev/stderr"
            fflush("/dev/stderr")
          } END { print "" > "/dev/stderr" }' >/dev/null
    sort -o "$OUT" "$OUT"
    echo "=== SUMMARY ==="
    awk -F'\t' '{c[$2]++} END {for (k in c) printf "  %-13s %d\n", k, c[k]}' "$OUT"
    # What must not differ: everything but a reported trace; with merging also
    # proofs, which a merged class may take from its representative.
    allowed='^(SAME|DIFF_TRACE)$'
    [ "$MERGE" = 1 ] && allowed='^(SAME|DIFF_TRACE|DIFF_PROOF)$'
    echo "=== needs attention ==="
    awk -F'\t' -v ok="$allowed" '$2 !~ ok {print "  "$2"\t"$1"\t"$3}' "$OUT"
    echo "  results: $OUT"
    fails=$(awk -F'\t' -v ok="$allowed" '$2 !~ ok' "$OUT" | grep -c .)
    norow=$(awk -F'\t' -v out="$OUT" 'FILENAME==out{seen[$1]=1;next} !($1 in seen)' \
                "$OUT" "$WORK/files.txt" | grep -c .)
    bad=''
    [ "$fails" = 0 ] || bad="FAIL=$fails"
    [ "$norow" = 0 ] || bad="${bad:+$bad }NOROW=$norow"
    echo "DONE_TOPN_TOP1 verdict=${bad:-OK} files=$TOTAL merge=$MERGE batch=$BATCH"
    [ -z "$bad" ]
    ;;

replay)
    N="${N:-3}"; MERGE="${MERGE:-1}"; BATCH="${BATCH:-8}"; ORDER="${ORDER:-iddfs}"
    DEADLINE_MS="${DEADLINE_MS:-60000}"; TIMEOUT="${TIMEOUT:-1800}"; DERIV="${DERIV:-30}"
    cores=$(nproc 2>/dev/null || echo 4)
    JOBS="${JOBS:-$(( cores >= 24 ? 3 : cores >= 12 ? 2 : 1 ))}"
    OUT="${OUT:-/tmp/topn_gate_replay.tsv}"
    KEEP="${KEEP:-}"
    [ -z "$KEEP" ] || mkdir -p "$KEEP"
    select_files replay
    MAUDE=$(resolve_maude) || exit 2
    maude_on_path "$MAUDE"
    oom_prologue
    top_env=("TAM_RS_TOP_METHODS=$N" "TAM_RS_TOPN_BATCH=$BATCH" "TAM_RS_SEARCH_ORDER=$ORDER"
             "TAM_PROVE_DEADLINE_MS=$DEADLINE_MS")
    [ "$MERGE" = 1 ] && top_env+=("TAM_RS_MERGE=1")
    wrapper "$WORK/topn" "${top_env[@]}"
    wrapper "$WORK/check"
    export WORK TIMEOUT DERIV KEEP
    echo "topn_gate: replay — prove with ${top_env[*]}, check with the plain prover;" \
         "JOBS=$JOBS TIMEOUT=${TIMEOUT}s OUT=$OUT"

    # verdicts <stdout>: "lemma<TAB>verdict<TAB>steps" per summary line.
    verdicts() {
        sed -nE 's/^  ([^ ]+) \((all-traces|exists-trace)\): (.*) \(([0-9]+) steps\)$/\1\t\2: \3\t\4/p' "$1"
    }
    # one <rel> → "rel \t status \t detail"
    one() {
        local rel="$1" f="$CORPUS/$1" fl rundir="" farg d rc
        fl=$(flags_for "$rel"); farg="$f"
        if [[ " $fl " == *" @cd "* ]]; then fl=${fl//@cd/}; rundir=$(dirname "$f"); farg=$(basename "$f"); fi
        d=$(mktemp -d "$WORK/one.XXXXXX")
        ( [ -n "$rundir" ] && cd "$rundir"
          timeout "$TIMEOUT" "$WORK/topn" $fl --derivcheck-timeout="$DERIV" --prove \
              --output="$d/proved.spthy" "$farg" ) >"$d/prove.out" 2>"$d/prove.err"; rc=$?
        if [ "$rc" = 124 ]; then printf '%s\tPROVE_TIMEOUT\t-\n' "$rel"; rm -rf "$d"; return 0; fi
        if [ "$rc" != 0 ] || [ ! -s "$d/proved.spthy" ]; then
            printf '%s\tPROVE_ERROR\trc=%s %s\n' "$rel" "$rc" "$(grep -m1 -i 'panicked\|error' "$d/prove.err" | cut -c1-160)"
            rm -rf "$d"; return 0
        fi
        # The check: no top-N variable, no --prove. Run from the same directory
        # with the same flags, so oracles and preprocessor flags resolve as above.
        ( [ -n "$rundir" ] && cd "$rundir"
          timeout "$TIMEOUT" "$WORK/check" $fl --derivcheck-timeout="$DERIV" \
              --output="$d/checked.spthy" "$d/proved.spthy" ) >"$d/check.out" 2>"$d/check.err"; rc=$?
        if [ "$rc" = 124 ]; then printf '%s\tCHECK_TIMEOUT\t-\n' "$rel"; rm -rf "$d"; return 0; fi
        if [ "$rc" != 0 ] || [ ! -s "$d/checked.spthy" ]; then
            printf '%s\tCHECK_ERROR\trc=%s %s\n' "$rel" "$rc" "$(grep -m1 -i 'panicked\|error' "$d/check.err" | cut -c1-160)"
            rm -rf "$d"; return 0
        fi
        if [ -n "$KEEP" ]; then
            local k="$KEEP/${rel//\//__}"
            cp "$d/proved.spthy" "$k.proved.spthy"; cp "$d/checked.spthy" "$k.checked.spthy"
        fi
        verdicts "$d/prove.out" > "$d/v.proved"; verdicts "$d/check.out" > "$d/v.checked"
        local n_lemmas unann bad="" incomplete
        n_lemmas=$(grep -c . "$d/v.proved")
        incomplete=$(grep -c 'analysis incomplete' "$d/v.proved")
        # A stored step the checker cannot replay is printed `/* unannotated */`.
        unann=$(grep -c 'unannotated' "$d/checked.spthy")
        # Per lemma: the verdict must match; the step count too, except where a
        # trace was found — the proof shows only the path to the trace, and the
        # checker adds every case it leaves out as `sorry`.
        bad=$(awk -F'\t' '
            FILENAME == ARGV[1] { v[$1] = $2; s[$1] = $3; next }
            !($1 in v)          { printf "%s:unproved ", $1; next }
            v[$1] != $2         { printf "%s:proved[%s]checked[%s] ", $1, v[$1], $2; next }
            v[$1] !~ /exists-trace: verified|found trace/ && s[$1] != $3 {
                                  printf "%s:steps proved %s checked %s ", $1, s[$1], $3 }
            ' "$d/v.proved" "$d/v.checked")
        if [ "$(grep -c . "$d/v.checked")" != "$n_lemmas" ]; then bad="${bad}lemma-count "; fi
        if [ "$n_lemmas" = 0 ]; then
            printf '%s\tNOLEMMA\t-\n' "$rel"
        elif [ "$unann" != 0 ]; then
            printf '%s\tUNANNOTATED\t%s step(s) %s\n' "$rel" "$unann" "$bad"
        elif [ -n "$bad" ]; then
            printf '%s\tREPLAY_DIFF\t%s\n' "$rel" "$bad"
        else
            printf '%s\tREPLAY_OK\tlemmas=%s incomplete=%s\n' "$rel" "$n_lemmas" "$incomplete"
        fi
        rm -rf "$d"
    }
    export -f one verdicts flags_for

    mapfile -t FILES < "$WORK/files.txt"
    TOTAL=${#FILES[@]}
    : > "$OUT"
    printf '%s\n' "${FILES[@]}" \
        | xargs -P "$JOBS" -I{} bash -c 'one "$@"' _ {} \
        | tee -a "$OUT" \
        | awk -v t="$TOTAL" '{
            n++
            if ($2 != "REPLAY_OK" && $2 != "NOLEMMA") printf "\r  %-14s %s  %s\033[K\n", $2, $1, $3 > "/dev/stderr"
            printf "\r  [%d/%d] %s\033[K", n, t, $1 > "/dev/stderr"
            fflush("/dev/stderr")
          } END { print "" > "/dev/stderr" }' >/dev/null
    sort -o "$OUT" "$OUT"
    echo "=== SUMMARY ==="
    awk -F'\t' '{c[$2]++} END {for (k in c) printf "  %-14s %d\n", k, c[k]}' "$OUT"
    awk -F'\t' '$2 == "REPLAY_OK" { split($3, a, /[= ]/); l += a[2]; i += a[4] }
                END { printf "  lemmas replayed: %d (of which the search left %d incomplete)\n", l, i }' "$OUT"
    echo "=== needs attention ==="
    awk -F'\t' '$2 != "REPLAY_OK" && $2 != "NOLEMMA" {print "  "$2"\t"$1"\t"$3}' "$OUT"
    echo "  results: $OUT"
    fails=$(awk -F'\t' '$2 != "REPLAY_OK" && $2 != "NOLEMMA"' "$OUT" | grep -c .)
    # A file with no row at all (a killed xargs child) compared nothing.
    norow=$(awk -F'\t' -v out="$OUT" 'FILENAME==out{seen[$1]=1;next} !($1 in seen)' \
                "$OUT" "$WORK/files.txt" | grep -c .)
    bad=''
    [ "$fails" = 0 ] || bad="FAIL=$fails"
    [ "$norow" = 0 ] || bad="${bad:+$bad }NOROW=$norow"
    echo "DONE_TOPN_REPLAY verdict=${bad:-OK} files=$TOTAL n=$N merge=$MERGE batch=$BATCH order=$ORDER deadline_ms=$DEADLINE_MS"
    [ -z "$bad" ]
    ;;
esac
