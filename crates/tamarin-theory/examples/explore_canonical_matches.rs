//! Empirical test: how much does canonicalization (`canon`/`canon_fingerprint`)
//! shrink a real proof search by merging $\alphaeqac$-equal constraint systems
//! reached via DIFFERENT paths?
//!
//! Tamarin's real prover is a GREEDY search: at each node,
//! `candidate_methods` ranks every applicable `ProofMethod` by heuristic
//! and only the first one whose `exec_proof_method` succeeds is taken
//! (`search::expand_inner`/`run_proof_search`). That walks exactly ONE
//! path through the proof tree, so it can never observe two DIFFERENT
//! paths reaching the "same" (canonically) system.
//!
//! This binary instead explores EXHAUSTIVELY (breadth-first, bounded by
//! `--max-depth`/`--max-nodes`): at every node it executes ALL of
//! `candidate_methods`' candidates, and every resulting case becomes a
//! child (or, with `--top-methods N`, only the N best-ranked applicable
//! candidates -- see below). Each child is canonicalized (`canonicalize_constraint_system`)
//! and fingerprinted as soon as it is generated; a child whose fingerprint
//! was already seen is MERGED into the existing node instead of becoming a
//! new one. The result is written as JSON (see "Output" below); profiling
//! and diagnostics go to the terminal.
//!
//! ## The AND/OR search graph
//!
//! - An **OR node** is one canonical constraint system: a choice among proof
//!   methods. It does not keep the `System` itself (that lives only in the
//!   BFS queue until the node is expanded).
//! - Each OR node owns one inline **AND node** per applicable proof method:
//!   the method, its kind, and its cases in order, each case pointing to a
//!   child OR node. A proof via that method must close ALL its cases.
//!   AND nodes are never shared; only OR nodes merge. A method with ZERO
//!   cases closed its branch -- but `sorry` also yields zero cases, so check
//!   `kind` before reading that as "proved". Methods whose
//!   `exec_proof_method` returns `None` get no AND node, only a count
//!   (`inapplicable`).
//!
//! ## Merging is sound: duplicates are never re-expanded
//!
//! A child canonically equal to an existing OR node becomes an edge to that
//! node and is not expanded again. This never loses a reachable system:
//!
//! 1. **Two canonically-equal systems have the same candidate children,
//!    up to renaming.** `exec_proof_method`/`candidate_methods` are
//!    functions of a system's LOGICAL content, not of its incidental
//!    representation (`NodeId` numbering, variable display names) -- this
//!    has to hold, or Tamarin's own proof search would be unsound. The one
//!    parameter that looks path-dependent, `depth` (passed to
//!    `candidate_methods(sys, ctx, depth)`), only selects which ROUND-ROBIN
//!    heuristic RANKING orders the SAME full goal list (`rank_goals_with`'s
//!    `rankings[depth % n]`, mirroring HS's `useHeuristic`) -- it reorders
//!    candidates, it never filters which ones exist, and this explorer takes
//!    every candidate anyway.
//! 2. **Each OR node is created at the MINIMUM depth of its canonical
//!    class.** Nodes are expanded in FIFO order and every child's depth is
//!    its parent's + 1, so nodes are created in non-decreasing depth order;
//!    the first occurrence of a class is its shallowest. Its expansion
//!    therefore has AT LEAST as much remaining `--max-depth` budget as any
//!    later occurrence would have had.
//!
//! Claim 1 is checked two ways:
//!
//! - **At every merge** (always on): the merged system's candidate proof
//!   methods (`candidate_methods` at its OWN depth) must equal the
//!   representative's, each renamed through its own system's canonical
//!   labelling (`canon::canonicalize_proof_method`) and compared as a
//!   MULTISET -- ranking uses `depth`, `looping` and `nr`, which the
//!   canonical form drops, so order may differ; a symmetry of the system
//!   also maps the multiset onto itself, whichever tied labelling won. A
//!   mismatch is printed and recorded in `method_check` (with both paths, to
//!   reproduce in the GUI) and exploration continues. This checks the
//!   methods offered, not the children they produce.
//! - **In aggregate**: `--no-merge` switches merging off (every child is a
//!   new node -- a plain tree search); its `graph`/`processed` must equal the
//!   merged run's `tree`. `--no-canon` is the same tree search without
//!   canonicalizing anything, so it costs what a search without merging
//!   costs (and checks nothing).
//!
//! ## Three sizes
//!
//! - `graph`: OR nodes -- distinct canonical systems.
//! - `processed`: occurrences actually canonicalized by this merging search
//!   (the root plus every generated child, duplicates included).
//! - `tree`: nodes a plain tree search would visit within the depth bound.
//!   Computed from the graph by layered root-path counting (`sizes`), NOT
//!   by multiplying subtree sizes by visit counts: a node reached from two
//!   different parents would be counted twice by the latter, and a node
//!   reached deeper than its first discovery gets only the remaining depth
//!   budget. Relies on claim 1 above.
//!
//! `graph <= processed <= tree` always holds (asserted), overall and per
//! depth for the last two.
//!
//! ## Output
//!
//! Written to `--out` (default `explore_<theory-stem>_<lemma>.json`) as one
//! compact JSON object, `schema_version` 4 (use `jq .` to read it), via a
//! temporary file and a rename (a killed run never leaves a truncated one):
//! `theory`, `lemma`, `argv`, `load` (`defines`, `auto_sources` as given,
//! `sapic`: the theory has a process), `params`, `stop_reason` (`exhausted`,
//! `max_nodes`, `time_budget`, `max_rss` or `panic`), `truncated` (anything but
//! `exhausted`), `lower_bound` (some node is unexpanded or failed to
//! canonicalize or expand, so `tree` undercounts), `timing` (`setup_secs`,
//! `explore_secs`, and the wall time per stage, always measured:
//! `canonicalize_secs` (every occurrence) of which `canonicalize_merged_secs`
//! (occurrences merged into an existing node), `fingerprint_secs`,
//! `candidate_methods_secs` (ranking, for every created node),
//! `exec_proof_method_secs`, `method_check_secs` (this tool's own merge
//! check)), `peak_rss_mib`, `canonicalize` (`failures` and up to 20
//! `examples`, each with its `kind` -- `panic` or `error` --, `path` and
//! `message`), `panic` (null, or where expanding a node panicked: `node`,
//! `path`, `activity`, `message`), `sleep` (`--sleep-sets`, else null:
//! `skipped_methods`, the candidates skipped because they were asleep, and
//! `asleep_nodes`), `invariant_violations` (see "Three
//! sizes"; expected only after a `panic` stop), `sizes` (the three totals plus
//! `graph_by_min_depth`, `graph_distinct_at_depth`, `processed_by_depth`,
//! `tree_by_depth`; counts above `u64::MAX` are strings), `status_counts`,
//! `method_check` (`checked` merges, `mismatches`, `failures` to canonicalize
//! a method, and up to 20 `examples`, each with the `representative`'s and
//! the `duplicate`'s depth and path as `[{method, case}, …]` steps plus the
//! methods only on either side), `canon` (see "What canonicalization did"), `methods` (interned method text, including
//! candidate methods that were never applied) and `nodes`: per OR node its `depth`
//! (min depth), `status` (`expanded`, `finished` + `result`, `depth_limit`,
//! `unexpanded`, `asleep`, `canon_panic`, `exec_panic`), `canon_err` (canonicalization returned an
//! error: expanded normally but never merged into), `fingerprint` (a hex
//! digest of the canonical form the node was created under, null when
//! canonicalization failed or with `--no-canon`), `first_parent`
//! (`[parent, and_index, case_index]` of the occurrence that created it),
//! `inapplicable`, `untried` (candidates skipped by `--top-methods`),
//! `asleep` (candidates skipped by `--sleep-sets`),
//! `canon_secs`/`methods_secs`/`exec_secs` (canonicalizing the occurrence
//! that created it, ranking its methods, executing them), and
//! `and` (`method` index, `kind`, `cases` as
//! `[name, child_id]` pairs).
//!
//! Usage: `cargo run --example explore_canonical_matches -- <theory.spthy> <lemma> [--max-depth N] [--max-nodes N] [--out PATH] [--no-merge] [--no-canon] [--sleep-sets] [--top-methods N] [--time-budget SECS] [--max-rss-gb GB] [--setup-timeout SECS] [--heartbeat SECS] [--trace] [-D=FLAG ...] [--auto-sources]`
//! or `... -- --list-lemmas <theory.spthy> [-D=FLAG ...] [--auto-sources]`.
//!
//! The theory is loaded as `tamarin-rs --prove` loads it
//! (`common::try_load_theory_for_proving`): `-D` defines and `#include`s,
//! `_restrict` lifting, SAPIC process and accountability translation, the
//! NDC-checked intruder rules, and auto-sources (`--auto-sources` or the
//! theory's `configuration:` block). SAPIC theories and accountability's
//! generated lemmas work as they do in the prover.
//!
//! - `-D=FLAG` (repeatable; the value attached, as for `tamarin-rs`) and
//!   `--auto-sources`: as for `tamarin-rs`.
//! - `--max-depth` (default 4): don't expand nodes at this depth.
//! - `--max-nodes` (default 2000): stop expanding once this many
//!   occurrences have been visited (canonicalized, unless `--no-canon`).
//!   Checked before each expansion, so it can be exceeded by one node's
//!   fan-out; queued nodes stay `unexpanded`.
//! - `--no-canon`: don't canonicalize: every occurrence is a node of its
//!   own, as with `--no-merge`, but without paying for canonical forms,
//!   fingerprints or the merge check. No `bliss` needed.
//! - `--sleep-sets`: skip method orders already explored. Each queued node
//!   carries a sleep set of goal numbers (`GoalStatus::nr`, which a goal
//!   keeps through the substitutions that instantiate it, so it names "the
//!   same goal" in the descendants; see `goal_nr` for the one exception,
//!   which only wakes a goal up). Expanding a node skips the candidate
//!   methods whose goal is asleep (recorded as `asleep`), then applies the
//!   first applicable ones as usual (up to `--top-methods`). The children of
//!   the k-th applied method get the node's sleep set plus the goals of the
//!   methods applied before it: a path that solves goal B and then A is
//!   then not explored once the path solving A and then B has been. A node
//!   with nothing left to apply but asleep methods -- every candidate is
//!   asleep, or the awake ones don't apply (typically `simplify`, ranked
//!   last) -- gets status `asleep`.
//!   Needs `--no-merge` or `--no-canon`: a sleep set belongs to a path, and
//!   a merged node is reached along several.
//! - `--top-methods N`: at each node, apply only the first N candidate
//!   methods whose `exec_proof_method` succeeds, in `candidate_methods`'
//!   (heuristic) order; inapplicable candidates don't count, the rest are
//!   recorded as `untried`. `--top-methods 1` follows the prover's own
//!   greedy choice at every node. The merge check still compares ALL
//!   candidates. With a heuristic of several round-robin rankings the order
//!   depends on the depth, and a merged node is expanded only once, at its
//!   first (minimum) depth -- so its top N follow that depth's ranking.
//! - `--time-budget SECS`: likewise, stop expanding once the process has
//!   run this long (setup included); the JSON is written as usual. Also
//!   checked only between expansions, so one expansion can overrun it.
//! - `--max-rss-gb GB`: likewise, stop expanding once the resident set
//!   reaches GB GiB (read from `/proc/self/status`), so a run approaching
//!   an external memory limit still writes its JSON instead of being
//!   killed. One expansion can overshoot it (a large `splitEqs` can take
//!   over 1 GB), so leave headroom below any hard limit.
//! - `--setup-timeout SECS`: give up if setup (parse, elaborate, maude,
//!   source precomputation) takes longer; setup can't be interrupted, so a
//!   watchdog thread exits the process with code 6 (no JSON).
//! - `--heartbeat SECS`: log a progress line (processed, graph, queue,
//!   rate, RSS, and the current node's step and how long it has run) every
//!   SECS, from a background thread, so a single long step (one
//!   `exec_proof_method` can take minutes) is visible as such.
//! - `--trace`: log every expansion and every applied method, so a run
//!   killed from outside still shows where it was.
//! - `--list-lemmas`: print one JSON object describing the theory instead
//!   of exploring: `lines`, `diff` (only parses with the `diff` flag, or has
//!   diff/equivalence lemmas; the port has no diff-mode prover),
//!   `processes` (SAPIC), `rules` (after translation), `lemmas` (`name`,
//!   `trace_quantifier`, `attributes`, `modulo`; accountability's generated
//!   lemmas included) and `error`.
//!
//! ## What canonicalization did
//!
//! Every occurrence is canonicalized through its term graph
//! (`canon_term_graph`). The JSON's `canon` object summarizes it:
//! `canonicalizations`, per-canonicalization `secs` (without the diagnostic
//! generator check) and `wall_secs` (with it), the term graphs' `vertices`
//! and `edges`, `aut` (`|Aut|` of each term graph, 1 where bliss was skipped)
//! and `nontrivial_aut`, `generators`, `generators_moving_free_vars`,
//! `generators_moving_vars` (free or local to an equation-store alternative),
//! `non_genuine_generators` (generators that are not a symmetry of the
//! system: the encoding lost information) with up to 20
//! `non_genuine_examples`, `generator_moves` (what the generators permute),
//! `bliss_skipped` (systems without free variables) and `phase_secs` (`term`,
//! `graph`, `bliss`, `content`, `check`).
//!
//! ## Batch runs
//!
//! `experiments/run_explore.py` (in the thesis repository) runs this over
//! many theories/lemmas, keeping each run's stderr as its log. Every log
//! line is prefixed with the seconds since start, and names the phase
//! (`setup`, `explore`, `write`) as it changes. A panic's message is
//! followed by a `panic context` line: the phase, the OR node being
//! expanded, its path, and the step at that node (which method, which case
//! being visited, which stage of visiting it). A panic while expanding a
//! node (outside canonicalization, whose panics are recorded per node)
//! stops the exploration there, marks the node `exec_panic`, and still
//! writes the JSON. Exit codes: 0 done (whatever `stop_reason`), 2 usage,
//! 3 setup failed (read/parse/elaborate/maude/lemma; no JSON), 4 stopped by
//! a panic (partial JSON), 5 a size invariant failed (JSON written), 6 setup
//! exceeded `--setup-timeout` (no JSON), 101 an uncaught panic.
//!
//! Env vars (all opt-in, unset = off), all terminal-only:
//! - `PROGRESS=1` -- stderr heartbeat every 20 canonicalizations (elapsed,
//!   rate, graph size, queue length), to tell "still working" from "hung".
//!   For a DH-heavy theory such as `wireguard.spthy` the bottleneck is
//!   `candidate_methods`' enormous branching factor.
//! - `DUMP_FORMULAS=1` -- dump `sys.formulas`/`sys.solved_formulas`/
//!   `sys.nodes`/`sys.last_atom`/`sys.goals` for every occurrence, to stderr.
//! - `PROFILE=1` -- prints the wall time per stage (always recorded in the
//!   JSON's `timing`): `canonicalize_constraint_system`,
//!   fingerprint + dedup lookup, `candidate_methods` (ranking, including
//!   its `is_finished` check, for every created node), `exec_proof_method`
//!   (solving -- the one most likely to route through `MaudeHandle`), and
//!   the merge check (candidate methods of merged occurrences, method
//!   signatures). The first two are what merging costs, the next two
//!   Tamarin's own search, the last this tool's verification.
//! - `DUMP_TERM_GRAPH_AT=N` -- writes the N-th canonicalized occurrence as
//!   a term (`<--out>.occ<N>.term.txt`), its term graph as bliss input
//!   (`.tg.dimacs`) and every vertex's color and key (`.tg.keys`).
//! - `TERM_GRAPH_CHECK=0` -- skip the term graph's check that every
//!   automorphism generator is a symmetry of the system (on by default; its
//!   time is reported apart, as `phase_secs.check`).
//! - `DUMP_CANON_PANIC=1` -- for the first occurrence whose canonicalization
//!   panics, prints its rule instances and the rest of the system (formulas,
//!   equation store, goals, ...) to stderr, to find which content the
//!   canonicalization missed.
//! - `DUMP_MERGES_INTO=ID[,ID...]` -- for each listed OR node id, writes
//!   the occurrence that created it and every occurrence merged into it
//!   (path, system, canonical labelling and canonical form) to
//!   `<--out>.node<ID>.rep.txt` / `<--out>.node<ID>.dup<N>.txt` (`N` =
//!   `processed`), to diff two systems a merge claims are equal. Node ids
//!   are this run's: rerun with the same arguments to hit the same ones.
//!
//! Requires `bliss` on `PATH` (or `$BLISS_PATH`), except with `--no-canon`
//! -- skips (via `bliss_available()`'s panic-unless-opted-out gate) if
//! `TAM_ALLOW_NO_BLISS=1` is set.

// Example/dev tool: prints results to stdout by design; allow the
// `disallowed_macros` convention freeze for this example binary.
#![allow(clippy::disallowed_macros)]

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use tamarin_term::alpha_eq_ac::CanonLabelling;
use tamarin_term::fingerprint::Fingerprint;
use tamarin_theory::bliss_proc::{bliss_available, to_dimacs};
use tamarin_theory::canon::{canonicalize_proof_method, CanonicalSystem};
use tamarin_theory::canon_fingerprint::{
    fingerprint_constraint_system, fingerprint_proof_method, CanonicalSystemFingerprint,
};
use tamarin_theory::canon_term_graph::{
    canonicalize_constraint_system_via_term_graph_with_labelling, system_to_term, term_to_graph,
    TermGraphOptions, TermGraphStats, SYSTEM_SECTIONS,
};
use tamarin_theory::constraint::solver::context::ProofContext;
use tamarin_theory::constraint::solver::proof_method::{
    exec_proof_method, ProofMethod, Result as FinishedResult,
};
use tamarin_theory::constraint::solver::search::candidate_methods;
use tamarin_theory::constraint::system::System;
use tamarin_theory::pretty_system::pretty_fact;
use tamarin_theory::pretty_theory::pretty_proof_method_inline;
use tamarin_theory::prove::ProverSession;
use tamarin_theory::rule::rule_name_string;
use tamarin_utils::{env_gate, FastMap};

mod common;

// =============================================================================
// Diagnostics for batch runs: timestamped log lines, breadcrumb, panic context
// =============================================================================
//
// A batch run keeps only this process's stderr (see "Batch runs" in the
// module docs). When a run dies -- a panic outside canonicalization, a hard
// timeout, the memory limit -- the log's last lines must say where: the
// phase, the OR node being expanded, its path, and the step at that node.

static PROCESS_START: OnceLock<Instant> = OnceLock::new();

/// Seconds since the process started (the first call).
fn elapsed_secs() -> f64 {
    PROCESS_START
        .get_or_init(Instant::now)
        .elapsed()
        .as_secs_f64()
}

/// `eprintln!` with the elapsed time as a prefix.
macro_rules! log {
    ($($arg:tt)*) => {
        eprintln!("[{:>9.2}s] {}", elapsed_secs(), format_args!($($arg)*))
    };
}

/// What the explorer is doing right now, printed by the panic hook and the
/// heartbeat thread (hence a process-wide mutex, not a thread-local).
struct Breadcrumb {
    phase: &'static str,
    /// The OR node being expanded and its depth.
    node: Option<(OrId, u32)>,
    /// That node's path from the root.
    path: String,
    /// The step at that node (which method runs, which child is visited).
    activity: String,
    /// When the current activity (or its latest sub-step) started.
    since: Option<Instant>,
}

static BREADCRUMB: Mutex<Breadcrumb> = Mutex::new(Breadcrumb {
    phase: "",
    node: None,
    path: String::new(),
    activity: String::new(),
    since: None,
});

/// Progress counters, published by the explorer for the heartbeat thread.
static PROCESSED: AtomicU64 = AtomicU64::new(0);
static GRAPH_NODES: AtomicU64 = AtomicU64::new(0);
static QUEUED: AtomicU64 = AtomicU64::new(0);

/// Runs `f` on the breadcrumb. A poisoned lock (a panic while holding it)
/// is still usable: the breadcrumb is plain data.
fn with_breadcrumb<T>(f: impl FnOnce(&mut Breadcrumb) -> T) -> T {
    let mut guard = BREADCRUMB.lock().unwrap_or_else(|e| e.into_inner());
    f(&mut guard)
}

fn set_phase(phase: &'static str) {
    with_breadcrumb(|b| {
        b.phase = phase;
        b.node = None;
        b.path.clear();
        b.activity.clear();
        b.since = Some(Instant::now());
    });
    log!("phase: {phase}");
}

fn set_breadcrumb_node(id: OrId, depth: u32, path: String) {
    with_breadcrumb(|b| {
        b.node = Some((id, depth));
        b.path = path;
        b.activity.clear();
        b.since = Some(Instant::now());
    });
}

fn set_activity(activity: String) {
    with_breadcrumb(|b| {
        b.activity = activity;
        b.since = Some(Instant::now());
    });
}

/// Replaces the sub-step after `" :: "` in the activity (e.g. which stage
/// of visiting a child runs), keeping what the expansion set before it.
fn set_activity_step(step: &str) {
    with_breadcrumb(|b| {
        if let Some(at) = b.activity.find(" :: ") {
            b.activity.truncate(at);
        }
        b.activity.push_str(" :: ");
        b.activity.push_str(step);
        b.since = Some(Instant::now());
    });
}

fn breadcrumb_activity() -> String {
    with_breadcrumb(|b| b.activity.clone())
}

fn breadcrumb_text() -> String {
    // `try_lock`: the panic hook may run while this thread holds the lock.
    let Ok(b) = BREADCRUMB.try_lock() else {
        return "<breadcrumb busy>".to_string();
    };
    let node = b
        .node
        .map_or_else(|| "-".to_string(), |(id, d)| format!("#{id} (depth {d})"));
    format!(
        "phase={} node={node} activity={} path={}",
        b.phase,
        if b.activity.is_empty() {
            "-"
        } else {
            &b.activity
        },
        if b.path.is_empty() { "-" } else { &b.path }
    )
}

/// `--heartbeat`: a background thread logging progress every `every`, also
/// while one step (a single `exec_proof_method`, say) runs for minutes --
/// what the node is doing and for how long.
fn spawn_heartbeat(every: Duration) {
    std::thread::spawn(move || loop {
        std::thread::sleep(every);
        let (phase, node, activity, secs) = with_breadcrumb(|b| {
            (
                b.phase,
                b.node,
                activity_for_log(&b.activity),
                b.since.map_or(0.0, |t| t.elapsed().as_secs_f64()),
            )
        });
        let processed = PROCESSED.load(Ordering::Relaxed);
        log!(
            "heartbeat: phase={phase} processed={processed} graph={} queue={} rate={:.2}/s \
             rss={}MiB | node={} for {secs:.0}s: {}",
            GRAPH_NODES.load(Ordering::Relaxed),
            QUEUED.load(Ordering::Relaxed),
            processed as f64 / elapsed_secs().max(0.001),
            memory_mib("VmRSS:").map_or_else(|| "?".to_string(), |m| m.to_string()),
            node.map_or_else(|| "-".to_string(), |(id, d)| format!("#{id} (depth {d})")),
            if activity.is_empty() { "-" } else { &activity },
        );
    });
}

/// Chains a hook after the default one (message, location, backtrace) that
/// adds the breadcrumb. Also fires for panics `catch_unwind` recovers from.
fn install_panic_hook() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        default_hook(info);
        log!("panic context: {}", breadcrumb_text());
    }));
}

/// `VmRSS`/`VmHWM` (current/peak resident set) in MiB, from
/// `/proc/self/status`; `None` off Linux.
fn memory_mib(field: &str) -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with(field))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib / 1024)
}

/// The activity for a heartbeat line: the description cut to 300 bytes,
/// but always with its ` :: <step>` suffix (which stage of visiting a child
/// runs) -- the part that matters when one step takes minutes.
fn activity_for_log(activity: &str) -> String {
    match activity.rfind(" :: ") {
        Some(at) => format!(
            "{} ...{}",
            truncate_for_log(&activity[..at], 300),
            &activity[at..]
        ),
        None => truncate_for_log(activity, 300).to_string(),
    }
}

/// `s` cut to at most `max` bytes (at a char boundary), for log lines.
fn truncate_for_log(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

// =============================================================================
// The AND/OR search graph
// =============================================================================

type OrId = u32;

#[derive(Debug, Clone, Copy)]
enum Status {
    /// Still queued when `--max-nodes` stopped the search -- every node's
    /// status until it is dequeued.
    Unexpanded,
    Expanded,
    /// `candidate_methods` offered only `Finished` with this result.
    Finished(&'static str),
    DepthLimit,
    /// `--sleep-sets`: nothing was left to apply but asleep methods (every
    /// candidate was asleep, or the awake ones didn't apply) -- the paths
    /// through it are explored elsewhere.
    Asleep,
    /// Canonicalization panicked: never merged into, never expanded.
    CanonPanic,
    /// Expanding the node panicked (outside canonicalization, e.g. in
    /// `exec_proof_method`); exploration stopped there. Its `and` holds
    /// what was built before the panic.
    ExecPanic,
}

impl Status {
    fn name(self) -> &'static str {
        match self {
            Status::Unexpanded => "unexpanded",
            Status::Expanded => "expanded",
            Status::Finished(_) => "finished",
            Status::DepthLimit => "depth_limit",
            Status::Asleep => "asleep",
            Status::CanonPanic => "canon_panic",
            Status::ExecPanic => "exec_panic",
        }
    }
}

/// One applied proof method of its (owning) OR node.
struct AndNode {
    /// Index into [`Graph::methods`].
    method: u32,
    kind: &'static str,
    /// In `exec_proof_method`'s order. Case names are unique only within one
    /// method's result, and two cases may reach the same OR node.
    cases: Vec<(String, OrId)>,
}

struct OrNode {
    /// Depth of the occurrence that created the node -- its minimum depth.
    depth: u32,
    status: Status,
    /// Canonicalization returned an error: the node is expanded normally
    /// but, having no fingerprint, can never be merged into.
    canon_err: bool,
    /// [`fingerprint_hex`] of the canonical form the node was created
    /// under; `None` when canonicalization failed.
    fingerprint: Option<String>,
    /// `(parent, and_index, case_index)` of the occurrence that created it;
    /// `None` for the root.
    first_parent: Option<(OrId, u32, u32)>,
    /// Candidate methods whose `exec_proof_method` returned `None`.
    inapplicable: u32,
    /// Candidate methods never tried because `--top-methods` were already
    /// applied.
    untried: u32,
    /// Candidate methods skipped because their goal was asleep
    /// (`--sleep-sets`).
    asleep: u32,
    /// Seconds spent canonicalizing (and fingerprinting) the occurrence
    /// that created the node, ranking its candidate methods, and executing
    /// the methods applied to it -- the latter two are what a search that
    /// merges into this node saves per merged occurrence (plus the same for
    /// the node's subtree).
    canon_secs: f64,
    methods_secs: f64,
    exec_secs: f64,
    and: Vec<AndNode>,
    /// The candidate methods of the system that created the node, for
    /// checking every later merge into it; `None` if canonicalization (of
    /// the system or of a method) failed.
    methods_sig: Option<MethodSignature>,
}

/// A system's candidate proof methods, each canonicalized through the
/// system's own labelling and fingerprinted, paired with its interned text,
/// sorted by fingerprint: a multiset. Order can't count -- ranking uses
/// `depth`, `looping` and `nr`, which the canonical form drops.
type MethodSignature = Vec<(Fingerprint, u32)>;

/// The entries only in `a` and only in `b` (as text ids), comparing the
/// two sorted multisets by fingerprint.
fn multiset_difference(a: &MethodSignature, b: &MethodSignature) -> (Vec<u32>, Vec<u32>) {
    let (mut i, mut j) = (0, 0);
    let (mut only_a, mut only_b) = (Vec::new(), Vec::new());
    loop {
        match (a.get(i), b.get(j)) {
            (Some(x), Some(y)) if x.0 == y.0 => {
                i += 1;
                j += 1;
            }
            (Some(x), Some(y)) if x.0 < y.0 => {
                only_a.push(x.1);
                i += 1;
            }
            (_, Some(y)) => {
                only_b.push(y.1);
                j += 1;
            }
            (Some(x), None) => {
                only_a.push(x.1);
                i += 1;
            }
            (None, None) => return (only_a, only_b),
        }
    }
}

#[cfg(test)]
mod multiset_difference_tests {
    use super::multiset_difference;

    /// Fingerprint `f` with text id `id` (ids differ between the two sides
    /// in practice -- same method, different raw variable names).
    fn e(f: u8, id: u32) -> ([u8; 32], u32) {
        ([f; 32], id)
    }

    #[test]
    fn equal_multisets_have_no_difference() {
        let a = vec![e(1, 0), e(2, 1), e(2, 2)];
        let b = vec![e(1, 7), e(2, 8), e(2, 9)];
        assert_eq!(multiset_difference(&a, &b), (vec![], vec![]));
    }

    #[test]
    fn multiplicity_counts() {
        let a = vec![e(1, 0), e(2, 1), e(2, 2)];
        let b = vec![e(1, 7), e(2, 8), e(3, 9)];
        assert_eq!(multiset_difference(&a, &b), (vec![2], vec![9]));
    }

    #[test]
    fn one_side_empty() {
        let a = vec![e(1, 0), e(4, 1)];
        assert_eq!(multiset_difference(&a, &vec![]), (vec![0, 1], vec![]));
        assert_eq!(multiset_difference(&vec![], &a), (vec![], vec![0, 1]));
    }
}

#[derive(Default)]
struct Graph {
    nodes: Vec<OrNode>,
    /// Interned method text: a goal persists down a path, so the same
    /// (possibly several-KB) text recurs across many nodes.
    methods: Vec<String>,
    method_ids: FastMap<String, u32>,
    /// Occurrences canonicalized, by the depth they were reached at.
    processed_by_depth: Vec<u64>,
}

impl Graph {
    fn node(&self, id: OrId) -> &OrNode {
        &self.nodes[id as usize]
    }

    fn node_mut(&mut self, id: OrId) -> &mut OrNode {
        &mut self.nodes[id as usize]
    }

    fn add_node(&mut self, depth: usize, first_parent: Option<(OrId, u32, u32)>) -> OrId {
        let id = OrId::try_from(self.nodes.len()).expect("more than u32::MAX OR nodes");
        self.nodes.push(OrNode {
            depth: depth as u32,
            status: Status::Unexpanded,
            canon_err: false,
            fingerprint: None,
            first_parent,
            inapplicable: 0,
            untried: 0,
            asleep: 0,
            canon_secs: 0.0,
            methods_secs: 0.0,
            exec_secs: 0.0,
            and: Vec::new(),
            methods_sig: None,
        });
        id
    }

    fn intern_method(&mut self, text: String) -> u32 {
        if let Some(&id) = self.method_ids.get(&text) {
            return id;
        }
        let id = self.methods.len() as u32;
        self.methods.push(text.clone());
        self.method_ids.insert(text, id);
        id
    }

    /// The `(method, case)` steps from the root to the occurrence reached via
    /// `edge` (`None` = the root), rebuilt from `first_parent` links: every
    /// OR node on it was created by exactly that step, so this is a real
    /// proof-tree path, walkable in the GUI.
    fn path_steps(&self, edge: Option<(OrId, u32, u32)>) -> Vec<(&str, &str)> {
        let mut steps = Vec::new();
        let mut cur = edge;
        while let Some((parent, and_idx, case_idx)) = cur {
            let and = &self.node(parent).and[and_idx as usize];
            steps.push((
                self.methods[and.method as usize].as_str(),
                and.cases[case_idx as usize].0.as_str(),
            ));
            cur = self.node(parent).first_parent;
        }
        steps.reverse();
        steps
    }

    /// [`Self::path_steps`] as `method [case] -> method [case] -> ...`.
    fn path(&self, edge: Option<(OrId, u32, u32)>) -> String {
        let steps = self.path_steps(edge);
        if steps.is_empty() {
            return "<root>".to_string();
        }
        steps
            .iter()
            .map(|(m, c)| format!("{m} [{c}]"))
            .collect::<Vec<_>>()
            .join(" -> ")
    }

    /// [`Self::path_steps`] as JSON: `[{"method": …, "case": …}, …]`.
    fn path_json(&self, edge: Option<(OrId, u32, u32)>) -> Value {
        Value::Array(
            self.path_steps(edge)
                .into_iter()
                .map(|(method, case)| json!({ "method": method, "case": case }))
                .collect(),
        )
    }

    /// Every OR node's children, one entry per case edge (with multiplicity).
    fn children(&self) -> Vec<Vec<OrId>> {
        self.nodes
            .iter()
            .map(|n| {
                n.and
                    .iter()
                    .flat_map(|a| a.cases.iter().map(|&(_, c)| c))
                    .collect()
            })
            .collect()
    }
}

fn method_kind(m: &ProofMethod) -> &'static str {
    match m {
        ProofMethod::Sorry(_) => "sorry",
        ProofMethod::Simplify => "simplify",
        ProofMethod::SolveGoal(_) => "solve",
        ProofMethod::Induction => "induction",
        ProofMethod::Finished(_) => "finished",
        ProofMethod::Invalidated => "invalidated",
    }
}

/// The `nr` of the goal `method` solves, or `None` if it solves none of
/// `sys`'s goals. Substitution rewrites a goal in place and keeps its
/// status, and of two goals it makes equal the older `nr` survives, so the
/// number names the same goal in the descendants of `sys`. The exception is
/// a `KU` goal on a message variable, product or union whose term changes:
/// substitution re-inserts it (decomposing it) under new numbers, so a
/// sleeping one wakes up -- safe, it only prunes less. New goals never reuse
/// a number.
fn goal_nr(sys: &System, method: &ProofMethod) -> Option<u64> {
    let ProofMethod::SolveGoal(goal) = method else {
        return None;
    };
    sys.goals
        .iter()
        .find(|(g, _)| g == goal)
        .map(|(_, status)| status.nr)
}

fn result_name(r: &FinishedResult) -> &'static str {
    match r {
        FinishedResult::Solved => "solved",
        FinishedResult::Contradictory(_) => "contradictory",
        FinishedResult::Unfinishable => "unfinishable",
    }
}

/// Unmerged-tree sizes, derived from the merged graph.
mod sizes {
    pub struct TreeSizes {
        /// Nodes a plain tree search would visit at each depth.
        pub by_depth: Vec<u128>,
        /// Distinct graph nodes with at least one root path of that length.
        pub distinct_at_depth: Vec<u64>,
    }

    /// Layered root-path counting: `paths_0(root) = 1`,
    /// `paths_{d+1}(c) = Σ_{edges p→c} paths_d(p)` (with edge multiplicity),
    /// `by_depth[d] = Σ_c paths_d(c)` for `d <= max_depth`. Layering gives a
    /// node reached deeper than its first discovery only the remaining depth
    /// budget, and terminates on cycles. `children[i]` lists node `i`'s
    /// children, one entry per edge.
    pub fn tree_sizes(children: &[Vec<u32>], root: u32, max_depth: usize) -> TreeSizes {
        let mut by_depth = Vec::new();
        let mut distinct_at_depth = Vec::new();
        let mut layer = vec![0u128; children.len()];
        layer[root as usize] = 1;
        for d in 0..=max_depth {
            let reached: Vec<u128> = layer.iter().copied().filter(|&p| p > 0).collect();
            if reached.is_empty() {
                break;
            }
            by_depth.push(
                reached
                    .iter()
                    .try_fold(0u128, |acc, &p| acc.checked_add(p))
                    .expect("tree size overflows u128"),
            );
            distinct_at_depth.push(reached.len() as u64);
            if d == max_depth {
                break;
            }
            let mut next = vec![0u128; children.len()];
            for (node, &paths) in layer.iter().enumerate() {
                if paths == 0 {
                    continue;
                }
                for &c in &children[node] {
                    let slot = &mut next[c as usize];
                    *slot = slot.checked_add(paths).expect("tree size overflows u128");
                }
            }
            layer = next;
        }
        TreeSizes {
            by_depth,
            distinct_at_depth,
        }
    }

    #[cfg(test)]
    mod tests {
        use super::tree_sizes;

        fn total(children: &[Vec<u32>], max_depth: usize) -> (Vec<u128>, u128) {
            let by_depth = tree_sizes(children, 0, max_depth).by_depth;
            let total = by_depth.iter().sum();
            (by_depth, total)
        }

        #[test]
        fn single_node() {
            assert_eq!(total(&[vec![]], 5), (vec![1], 1));
        }

        /// root→A, root→C, A→B, C→B: B is merged but occurs twice in the
        /// tree. Multiplying subtree sizes by visit counts would give 7.
        #[test]
        fn diamond_counts_the_shared_node_once_per_path() {
            let children = vec![vec![1, 2], vec![3], vec![3], vec![]];
            assert_eq!(total(&children, 5), (vec![1, 2, 2], 5));
            assert_eq!(tree_sizes(&children, 0, 5).distinct_at_depth, vec![1, 2, 1]);
        }

        /// root⇒A twice, A⇒B twice: parallel edges multiply.
        #[test]
        fn parallel_edges_multiply() {
            let children = vec![vec![1, 1], vec![2, 2], vec![]];
            assert_eq!(total(&children, 5), (vec![1, 2, 4], 7));
        }

        #[test]
        fn self_loop_is_bounded_by_max_depth() {
            assert_eq!(total(&[vec![0]], 3), (vec![1, 1, 1, 1], 4));
        }

        /// Node 1 is first found at depth 1 (and expanded, child 3), and
        /// reached again at depth 3 via 2→4→1. At depth 3 == max_depth it
        /// must contribute no children.
        #[test]
        fn a_deeper_occurrence_gets_only_the_remaining_depth_budget() {
            let children = vec![vec![1, 2], vec![3], vec![4], vec![], vec![1]];
            assert_eq!(total(&children, 3), (vec![1, 2, 2, 1], 6));
        }
    }
}

// =============================================================================
// Profiling (`PROFILE`) -- terminal only
// =============================================================================

/// Runs `f`, adding its wall time to `acc` when `on`.
/// Runs `f`, adds its wall time to `acc`, and returns its result and that
/// time.
fn timed<T>(acc: &mut Duration, f: impl FnOnce() -> T) -> (T, Duration) {
    let start = Instant::now();
    let out = f();
    let took = start.elapsed();
    *acc += took;
    (out, took)
}

/// `secs` rounded to microseconds, for the per-node JSON timings.
fn micros(secs: f64) -> f64 {
    (secs * 1e6).round() / 1e6
}

/// `part` as a percentage of `total` (0 when `total` is zero).
fn pct(part: Duration, total: Duration) -> f64 {
    if total.is_zero() {
        0.0
    } else {
        100.0 * part.as_secs_f64() / total.as_secs_f64()
    }
}

/// Wall time per stage, always measured (written to the JSON's `timing`;
/// `PROFILE=1` also prints it). What a search that merges canonically equal
/// systems would pay is `canonicalize` + `fingerprint` for every occurrence,
/// on top of the search's own `candidate_methods` + `exec_proof_method`;
/// `method_check` is this tool's own verification of merges.
#[derive(Default)]
struct StageTimers {
    /// Canonicalizing every occurrence (including failed ones).
    canonicalize: Duration,
    /// The part of `canonicalize` spent on occurrences merged into an
    /// existing node.
    canonicalize_merged: Duration,
    fingerprint: Duration,
    /// Ranking the candidate methods of every node that was created.
    candidate_methods: Duration,
    exec_proof_method: Duration,
    /// Merge verification only: candidate methods of merged occurrences,
    /// canonicalizing every occurrence's methods, comparing them.
    method_check: Duration,
}

// =============================================================================
// What canonicalization did (the JSON's `canon` object)
// =============================================================================

/// Count, sum, mean, median, 95th percentile and maximum of `values`.
fn distribution(values: &[f64]) -> Value {
    if values.is_empty() {
        return json!({"n": 0});
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let sum: f64 = sorted.iter().sum();
    let at = |q: f64| sorted[((sorted.len() - 1) as f64 * q).round() as usize];
    json!({
        "n": sorted.len(),
        "sum": sum,
        "mean": sum / sorted.len() as f64,
        "p50": at(0.5),
        "p95": at(0.95),
        "max": sorted[sorted.len() - 1],
    })
}

/// What canonicalization did over a run.
#[derive(Default)]
struct CanonSummary {
    /// Wall time per canonicalization, the generator check included.
    wall_secs: Vec<f64>,
    /// The time a search would pay: the wall time minus the generator check
    /// (a diagnostic only).
    secs: Vec<f64>,
    vertices: Vec<f64>,
    edges: Vec<f64>,
    /// `|Aut|` of the term graph (1 where bliss did not run).
    aut: Vec<f64>,
    generators: u64,
    /// See `TermGraphStats`.
    generators_moving_free_vars: u64,
    generators_moving_vars: u64,
    non_genuine_generators: u64,
    /// What the generators permute (`TermGraphStats::generator_moves`).
    generator_moves: BTreeMap<String, u64>,
    /// Occurrences with a non-genuine generator, with what it moves and
    /// which fields of the form it changes.
    non_genuine_examples: Vec<Value>,
    bliss_skipped: u64,
    term_secs: f64,
    graph_secs: f64,
    bliss_secs: f64,
    content_secs: f64,
    check_secs: f64,
}

impl CanonSummary {
    fn record(&mut self, wall: Duration, s: &TermGraphStats) {
        self.wall_secs.push(wall.as_secs_f64());
        let production = s.time_term + s.time_graph + s.time_bliss + s.time_content;
        self.secs.push(production.as_secs_f64());
        self.vertices.push(s.vertices as f64);
        self.edges.push(s.edges as f64);
        self.aut.push(
            s.aut_size
                .as_deref()
                .map_or(1.0, |a| a.parse().unwrap_or(f64::NAN)),
        );
        self.generators += s.generators as u64;
        self.generators_moving_free_vars += s.generators_moving_free_vars as u64;
        self.generators_moving_vars += s.generators_moving_vars as u64;
        self.non_genuine_generators += s.non_genuine_generators as u64;
        for (moved, n) in &s.generator_moves {
            *self.generator_moves.entry(moved.clone()).or_insert(0) += *n as u64;
        }
        self.bliss_skipped += u64::from(!s.bliss_ran);
        self.term_secs += s.time_term.as_secs_f64();
        self.graph_secs += s.time_graph.as_secs_f64();
        self.bliss_secs += s.time_bliss.as_secs_f64();
        self.content_secs += s.time_content.as_secs_f64();
        self.check_secs += s.time_check.as_secs_f64();
    }

    /// Logs and keeps (up to [`FAILURE_EXAMPLES_LIMIT`]) an occurrence whose
    /// term graph has a generator that is not a symmetry of the system.
    fn record_non_genuine(&mut self, stats: &TermGraphStats, processed: u64, path: String) {
        if stats.non_genuine_generators == 0 {
            return;
        }
        log!(
            "term graph: {} non-genuine generator(s) at processed={processed} {path}: moves {:?}, \
             fields {:?}",
            stats.non_genuine_generators,
            stats.non_genuine_moves,
            stats.non_genuine_fields
        );
        if self.non_genuine_examples.len() < FAILURE_EXAMPLES_LIMIT {
            self.non_genuine_examples.push(json!({
                "processed": processed,
                "path": path,
                "moves": stats.non_genuine_moves,
                "fields": stats.non_genuine_fields,
            }));
        }
    }

    fn json(&self) -> Value {
        json!({
            "canonicalizations": self.secs.len(),
            "secs": distribution(&self.secs),
            "wall_secs": distribution(&self.wall_secs),
            "vertices": distribution(&self.vertices),
            "edges": distribution(&self.edges),
            "aut": distribution(&self.aut),
            "nontrivial_aut": self.aut.iter().filter(|&&a| a > 1.0).count(),
            "generators": self.generators,
            "generators_moving_free_vars": self.generators_moving_free_vars,
            "generators_moving_vars": self.generators_moving_vars,
            "non_genuine_generators": self.non_genuine_generators,
            "generator_moves": self.generator_moves,
            "non_genuine_examples": self.non_genuine_examples,
            "bliss_skipped": self.bliss_skipped,
            "phase_secs": {
                "term": self.term_secs,
                "graph": self.graph_secs,
                "bliss": self.bliss_secs,
                "content": self.content_secs,
                "check": self.check_secs,
            },
        })
    }

    /// One line for the terminal.
    fn line(&self) -> String {
        let total: f64 = self.secs.iter().sum();
        let max_aut = self.aut.iter().copied().fold(1.0, f64::max);
        let mut line = format!(
            "{} canonicalizations in {total:.2}s (mean {:.2}ms), vertices mean {:.0}, \
             |Aut| > 1 in {}, max |Aut| {max_aut:.3e}, generators {} (moving free variables {}, \
             not genuine {}), bliss skipped {}",
            self.secs.len(),
            1000.0 * total / self.secs.len().max(1) as f64,
            self.vertices.iter().sum::<f64>() / self.vertices.len().max(1) as f64,
            self.aut.iter().filter(|&&a| a > 1.0).count(),
            self.generators,
            self.generators_moving_free_vars,
            self.non_genuine_generators,
            self.bliss_skipped
        );
        let mut moves: Vec<(&String, &u64)> = self.generator_moves.iter().collect();
        moves.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        if !moves.is_empty() {
            let top: Vec<String> = moves
                .iter()
                .take(3)
                .map(|(m, n)| format!("{m} ×{n}"))
                .collect();
            line += &format!("; generators move {}", top.join(", "));
        }
        line
    }
}

/// `DUMP_TERM_GRAPH_AT`: `sys` as a term, its graph as bliss input, and the
/// key of every vertex.
fn dump_term_graph(sys: &System, stem: &str) {
    let term = system_to_term(sys);
    let graph = term_to_graph(&term);
    let keys: String = graph
        .keys
        .iter()
        .enumerate()
        .map(|(i, k)| format!("{} {} {:?}\n", i + 1, graph.colors[i], k))
        .collect();
    let written = std::fs::write(format!("{stem}.term.txt"), format!("{term}\n"))
        .and_then(|()| match to_dimacs(&graph.colors, &graph.edges) {
            Ok(dimacs) => std::fs::write(format!("{stem}.tg.dimacs"), dimacs),
            Err(e) => Err(std::io::Error::other(e.to_string())),
        })
        .and_then(|()| std::fs::write(format!("{stem}.tg.keys"), keys));
    log!("DUMP_TERM_GRAPH_AT: wrote {stem}.term.txt/.tg.dimacs/.tg.keys: {written:?}");
}

// =============================================================================
// Exploration
// =============================================================================

struct Flags {
    progress: bool,
    /// `--trace`: one log line per expansion and per applied method.
    trace: bool,
    dump_formulas: bool,
    profile: bool,
    /// `DUMP_CANON_PANIC=1`: print the first system whose canonicalization
    /// panics.
    dump_canon_panic: bool,
    /// `DUMP_MERGES_INTO=ID[,ID...]`, with the `--out` path the dumps are
    /// named after.
    dump_merges_into: Option<(BTreeSet<OrId>, String)>,
    /// `DUMP_TERM_GRAPH_AT=N`, with the `--out` path the dump is named after.
    dump_term_graph_at: Option<(u64, String)>,
    /// Check every automorphism generator (`TERM_GRAPH_CHECK`, on unless
    /// `0`).
    term_graph_check: bool,
}

/// The goals asleep at a node, by `GoalStatus::nr` (see `--sleep-sets`).
type SleepSet = BTreeSet<u64>;

/// A node awaiting expansion: the `System` only it still carries, its
/// candidate methods if already computed (for the method check), and its
/// sleep set (empty without `--sleep-sets`).
struct Queued {
    id: OrId,
    sys: System,
    methods: Option<Vec<ProofMethod>>,
    sleep: SleepSet,
}

struct Explorer<'a> {
    ctx: &'a ProofContext,
    flags: Flags,
    /// What canonicalization did.
    canon_summary: CanonSummary,
    max_depth: usize,
    no_merge: bool,
    /// `--no-canon`: never canonicalize (implies `no_merge`).
    no_canon: bool,
    /// `--sleep-sets`: skip method orders already explored.
    sleep_sets: bool,
    /// `--top-methods`: apply only this many candidate methods per node.
    top_methods: Option<usize>,
    graph: Graph,
    by_fingerprint: FastMap<CanonicalSystemFingerprint, OrId>,
    queue: VecDeque<Queued>,
    /// Candidate methods skipped because their goal was asleep.
    sleep_skipped: u64,
    processed: u64,
    canonicalize_failures: usize,
    method_check: MethodCheck,
    timers: StageTimers,
    started: Instant,
    /// `--time-budget`: stop expanding once the PROCESS has run this long.
    time_budget: Option<Duration>,
    /// `--max-rss-gb`, in MiB: stop expanding once the resident set is this
    /// large.
    max_rss_mib: Option<u64>,
    /// The first [`FAILURE_EXAMPLES_LIMIT`] canonicalization failures.
    canon_failure_examples: Vec<Value>,
    /// Set when expanding a node panicked (see [`StopReason::Panic`]).
    panic: Option<Value>,
}

/// Why [`Explorer::run`] stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StopReason {
    /// The queue ran empty: everything within `--max-depth` was explored.
    Exhausted,
    /// `--max-nodes` occurrences were canonicalized.
    MaxNodes,
    /// `--time-budget` ran out.
    TimeBudget,
    /// The resident set reached `--max-rss-gb`.
    MaxRss,
    /// Expanding a node panicked outside canonicalization. Exploration
    /// stops there: the panic may have left the shared solver state (e.g.
    /// the maude connection) inconsistent.
    Panic,
}

impl StopReason {
    fn name(self) -> &'static str {
        match self {
            StopReason::Exhausted => "exhausted",
            StopReason::MaxNodes => "max_nodes",
            StopReason::TimeBudget => "time_budget",
            StopReason::MaxRss => "max_rss",
            StopReason::Panic => "panic",
        }
    }
}

const FAILURE_EXAMPLES_LIMIT: usize = 20;

/// Results of checking, at every merge, that the merged system offers the
/// same candidate proof methods as the representative (see the module docs).
#[derive(Default)]
struct MethodCheck {
    /// Merges whose both sides had a method signature.
    checked: u64,
    mismatches: u64,
    /// Occurrences where canonicalizing a proof method panicked.
    failures: u64,
    /// The first [`METHOD_MISMATCH_EXAMPLES_LIMIT`] mismatches, for the JSON.
    examples: Vec<Value>,
}

const METHOD_MISMATCH_EXAMPLES_LIMIT: usize = 20;

fn panic_message(payload: &(dyn std::any::Any + Send)) -> &str {
    payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .unwrap_or("<non-string panic payload>")
}

impl Explorer<'_> {
    /// Canonicalizes one occurrence of `sys`, reached at `depth` via `edge`
    /// (`None` for the root), and returns its OR node: an existing one if
    /// `sys` is canonically equal to an already-seen system, otherwise a new
    /// one -- queued for expansion unless canonicalization panicked.
    fn visit(
        &mut self,
        sys: System,
        depth: usize,
        edge: Option<(OrId, u32, u32)>,
        sleep: SleepSet,
    ) -> OrId {
        let ctx = self.ctx;
        self.processed += 1;
        if self.graph.processed_by_depth.len() <= depth {
            self.graph.processed_by_depth.resize(depth + 1, 0);
        }
        self.graph.processed_by_depth[depth] += 1;

        self.publish_progress();
        if self.flags.progress && self.processed.is_multiple_of(20) {
            let elapsed = self.started.elapsed().as_secs_f64();
            eprintln!(
                "[progress] {} canonicalized in {elapsed:.1}s ({:.2}/s), depth={depth}, \
                 graph={}, queue={}",
                self.processed,
                self.processed as f64 / elapsed.max(0.001),
                self.graph.nodes.len(),
                self.queue.len()
            );
        }

        if self.flags.dump_formulas {
            dump_formulas(&sys, &self.graph.path(edge));
        }

        // A canonicalization panic (e.g. a `theta` miss) must not end the
        // whole exploration: this tool's job is to FIND such gaps across
        // every path, not stop at the first.
        if let Some((n, out)) = &self.flags.dump_term_graph_at
            && self.processed == *n
        {
            dump_term_graph(&sys, &format!("{out}.occ{n}"));
        }
        if self.no_canon {
            // `expand` ranks the methods, so a node at the depth limit never
            // pays for it.
            let id = self.graph.add_node(depth, edge);
            self.queue.push_back(Queued {
                id,
                sys,
                methods: None,
                sleep,
            });
            return id;
        }
        set_activity_step("canonicalize");
        let options = TermGraphOptions {
            check_generators: self.flags.term_graph_check,
            certificate: false,
        };
        let (canon, canon_took) = timed(&mut self.timers.canonicalize, || {
            catch_unwind(AssertUnwindSafe(|| {
                canonicalize_constraint_system_via_term_graph_with_labelling(&sys, options)
            }))
        });
        let mut fingerprint_took = Duration::ZERO;
        let mut methods_took = Duration::ZERO;

        let (fingerprint, methods, methods_sig) = match canon {
            Err(payload) => {
                let message = panic_message(&*payload).to_string();
                if self.flags.dump_canon_panic && self.canonicalize_failures == 0 {
                    dump_system(&sys);
                }
                log!(
                    "canonicalize_constraint_system PANICKED at {}: {message}",
                    self.graph.path(edge)
                );
                let id = self.graph.add_node(depth, edge);
                let node = self.graph.node_mut(id);
                node.status = Status::CanonPanic;
                node.canon_secs = canon_took.as_secs_f64();
                self.record_canon_failure("panic", Some(id), depth, edge, message);
                return id;
            }
            Ok(Err(e)) => {
                log!(
                    "canonicalize_constraint_system failed at {}: {e:?}",
                    self.graph.path(edge)
                );
                self.record_canon_failure("error", None, depth, edge, format!("{e:?}"));
                (None, None, None)
            }
            Ok(Ok((canon, labelling, stats))) => {
                self.canon_summary.record(canon_took, &stats);
                let path = self.graph.path(edge);
                self.canon_summary
                    .record_non_genuine(&stats, self.processed, path);
                let (fp, took) = timed(&mut self.timers.fingerprint, || {
                    fingerprint_constraint_system(&canon)
                });
                fingerprint_took = took;
                if self.flags.dump_merges_into.is_some() {
                    self.dump_merge_occurrence(&sys, &canon, &labelling, &fp, depth, edge);
                }
                set_activity_step("candidate_methods");
                let start = Instant::now();
                let methods = candidate_methods(&sys, ctx, depth)
                    .unwrap_or_else(|e| panic!("candidate_methods: {e}"));
                methods_took = start.elapsed();
                let start = Instant::now();
                let sig = self.method_signature(&sys, &labelling, &methods, edge);
                self.timers.method_check += start.elapsed();
                (Some(fp), Some(methods), sig)
            }
        };

        if let (Some(fp), false) = (fingerprint, self.no_merge)
            && let Some(&existing) = self.by_fingerprint.get(&fp)
        {
            // A merged occurrence's candidate methods only feed the merge
            // check.
            self.timers.canonicalize_merged += canon_took;
            self.timers.method_check += methods_took;
            if let Some(sig) = &methods_sig {
                let start = Instant::now();
                self.check_methods(existing, sig, depth, edge);
                self.timers.method_check += start.elapsed();
            }
            return existing;
        }
        self.timers.candidate_methods += methods_took;
        let id = self.graph.add_node(depth, edge);
        let node = self.graph.node_mut(id);
        node.canon_err = fingerprint.is_none();
        node.fingerprint = fingerprint.as_ref().map(fingerprint_hex);
        node.methods_sig = methods_sig;
        node.canon_secs = (canon_took + fingerprint_took).as_secs_f64();
        node.methods_secs = methods_took.as_secs_f64();
        if let (Some(fp), false) = (fingerprint, self.no_merge) {
            self.by_fingerprint.insert(fp, id);
        }
        self.queue.push_back(Queued {
            id,
            sys,
            methods,
            sleep,
        });
        id
    }

    /// `DUMP_MERGES_INTO`: if this occurrence creates, or merges into, one
    /// of the requested OR nodes, writes it -- path, system, labelling and
    /// canonical form -- to `<out>.node<ID>.rep.txt` (the occurrence that
    /// created the node) or `<out>.node<ID>.dup<processed>.txt` (one merged
    /// into it), for diffing.
    fn dump_merge_occurrence(
        &self,
        sys: &System,
        canon: &CanonicalSystem,
        labelling: &CanonLabelling,
        fp: &CanonicalSystemFingerprint,
        depth: usize,
        edge: Option<(OrId, u32, u32)>,
    ) {
        let Some((ids, out)) = &self.flags.dump_merges_into else {
            return;
        };
        let existing = if self.no_merge {
            None
        } else {
            self.by_fingerprint.get(fp).copied()
        };
        // A new node gets the next id (`Graph::add_node`).
        let id = existing.unwrap_or(self.graph.nodes.len() as OrId);
        if !ids.contains(&id) {
            return;
        }
        let file = match existing {
            Some(_) => format!("{out}.node{id}.dup{}.txt", self.processed),
            None => format!("{out}.node{id}.rep.txt"),
        };
        let text = format!(
            "node {id}, occurrence {} at depth {depth}, {}\npath: {}\nfingerprint: {}\n\n\
             === system ===\n{}\n=== labelling ===\n{labelling:#?}\n\n=== canonical form ===\n{}",
            self.processed,
            if existing.is_some() {
                "merged into it"
            } else {
                "created it"
            },
            self.graph.path(edge),
            fingerprint_hex(fp),
            system_text(sys),
            canonical_text(canon),
        );
        match std::fs::write(&file, text) {
            Ok(()) => log!("DUMP_MERGES_INTO: wrote {file}"),
            Err(e) => log!("DUMP_MERGES_INTO: cannot write {file}: {e}"),
        }
    }

    /// Counts a canonicalization failure (`kind` "panic" or "error") and
    /// keeps the first [`FAILURE_EXAMPLES_LIMIT`] for the JSON. `node` is
    /// the OR node it created, if any (a failed canonicalization that
    /// returned an error creates its node later, unmergeable).
    fn record_canon_failure(
        &mut self,
        kind: &str,
        node: Option<OrId>,
        depth: usize,
        edge: Option<(OrId, u32, u32)>,
        message: String,
    ) {
        self.canonicalize_failures += 1;
        if self.canon_failure_examples.len() < FAILURE_EXAMPLES_LIMIT {
            self.canon_failure_examples.push(json!({
                "kind": kind,
                "node": node,
                "depth": depth,
                "path": self.graph.path_json(edge),
                "message": message,
            }));
        }
    }

    /// Publishes the progress counters the heartbeat thread reports.
    fn publish_progress(&self) {
        PROCESSED.store(self.processed, Ordering::Relaxed);
        GRAPH_NODES.store(self.graph.nodes.len() as u64, Ordering::Relaxed);
        QUEUED.store(self.queue.len() as u64, Ordering::Relaxed);
    }

    /// `methods` (offered for `sys`) as a [`MethodSignature`] under `sys`'s
    /// canonical `labelling`, or `None` if canonicalizing one of them
    /// panicked -- itself a canonicalization gap, reported here.
    fn method_signature(
        &mut self,
        sys: &System,
        labelling: &CanonLabelling,
        methods: &[ProofMethod],
        edge: Option<(OrId, u32, u32)>,
    ) -> Option<MethodSignature> {
        let fingerprints = catch_unwind(AssertUnwindSafe(|| {
            methods
                .iter()
                .map(|m| fingerprint_proof_method(&canonicalize_proof_method(m, sys, labelling)))
                .collect::<Vec<_>>()
        }));
        match fingerprints {
            Ok(fps) => {
                let mut sig: MethodSignature = fps
                    .into_iter()
                    .zip(methods)
                    .map(|(fp, m)| (fp, self.graph.intern_method(pretty_proof_method_inline(m))))
                    .collect();
                sig.sort_unstable();
                Some(sig)
            }
            Err(payload) => {
                self.method_check.failures += 1;
                eprintln!(
                    "canonicalize_proof_method PANICKED at {}: {}",
                    self.graph.path(edge),
                    panic_message(&*payload)
                );
                None
            }
        }
    }

    /// Checks that the occurrence reached at `depth` via `edge`, merged into
    /// `rep`, offers the same candidate methods (`sig`) as `rep` did.
    /// A mismatch is recorded, with both paths, and exploration continues.
    fn check_methods(
        &mut self,
        rep: OrId,
        sig: &MethodSignature,
        depth: usize,
        edge: Option<(OrId, u32, u32)>,
    ) {
        let rep_node = self.graph.node(rep);
        let Some(rep_sig) = &rep_node.methods_sig else {
            return;
        };
        self.method_check.checked += 1;
        if rep_sig.iter().map(|e| e.0).eq(sig.iter().map(|e| e.0)) {
            return;
        }
        self.method_check.mismatches += 1;
        let (only_rep, only_dup) = multiset_difference(rep_sig, sig);
        let texts = |ids: &[u32]| -> Vec<&str> {
            ids.iter()
                .map(|&i| self.graph.methods[i as usize].as_str())
                .collect()
        };
        eprintln!(
            "METHOD MISMATCH merging into #{rep} (depth {}) via {}\n  \
             duplicate (depth {depth}) via {}\n  only in representative: {:?}\n  \
             only in duplicate: {:?}",
            rep_node.depth,
            self.graph.path(rep_node.first_parent),
            self.graph.path(edge),
            texts(&only_rep),
            texts(&only_dup)
        );
        if self.method_check.examples.len() < METHOD_MISMATCH_EXAMPLES_LIMIT {
            let example = json!({
                "representative": {
                    "id": rep,
                    "depth": rep_node.depth,
                    "path": self.graph.path_json(rep_node.first_parent),
                },
                "duplicate": { "depth": depth, "path": self.graph.path_json(edge) },
                "only_in_representative": texts(&only_rep),
                "only_in_duplicate": texts(&only_dup),
            });
            self.method_check.examples.push(example);
        }
    }

    /// Expands queued nodes breadth-first until the queue is empty,
    /// `max_nodes` occurrences have been canonicalized, the time budget is
    /// spent, or an expansion panics. Nodes left in the queue stay
    /// `Unexpanded`.
    fn run(&mut self, max_nodes: u64) -> StopReason {
        while let Some(queued) = self.queue.pop_front() {
            let stop = if self.processed >= max_nodes {
                Some(StopReason::MaxNodes)
            } else if self
                .time_budget
                .is_some_and(|b| elapsed_secs() >= b.as_secs_f64())
            {
                Some(StopReason::TimeBudget)
            } else if self
                .max_rss_mib
                .is_some_and(|max| memory_mib("VmRSS:").is_some_and(|rss| rss >= max))
            {
                Some(StopReason::MaxRss)
            } else {
                None
            };
            if let Some(stop) = stop {
                log!(
                    "stopping ({}): processed={} graph={} queue={} rss={}MiB",
                    stop.name(),
                    self.processed,
                    self.graph.nodes.len(),
                    self.queue.len() + 1,
                    memory_mib("VmRSS:").map_or_else(|| "?".to_string(), |m| m.to_string())
                );
                self.queue.push_front(queued);
                return stop;
            }
            let Queued {
                id,
                sys,
                methods,
                sleep,
            } = queued;
            let expanded =
                catch_unwind(AssertUnwindSafe(|| self.expand(id, &sys, methods, &sleep)));
            if let Err(payload) = expanded {
                self.record_expand_panic(id, panic_message(&*payload).to_string());
                return StopReason::Panic;
            }
        }
        StopReason::Exhausted
    }

    /// Marks `id` as [`Status::ExecPanic`] after its expansion panicked,
    /// dropping the placeholder case of a child that was being visited
    /// (always the last case of the last AND node, so every other node's
    /// `first_parent` stays valid), and records the panic for the JSON.
    fn record_expand_panic(&mut self, id: OrId, message: String) {
        let activity = breadcrumb_activity();
        let node = self.graph.node_mut(id);
        node.status = Status::ExecPanic;
        if let Some(last) = node.and.last_mut() {
            last.cases.retain(|&(_, child)| child != OrId::MAX);
        }
        let depth = node.depth;
        let first_parent = node.first_parent;
        log!(
            "expansion of #{id} PANICKED, stopping exploration: {message} (while: {})",
            truncate_for_log(&activity, 500)
        );
        self.panic = Some(json!({
            "node": id,
            "depth": depth,
            "path": self.graph.path_json(first_parent),
            "activity": activity,
            "message": message,
        }));
    }

    /// `methods` are `sys`'s candidate methods if `visit` already computed
    /// them (every node except `canon_err` ones and with `--no-canon`);
    /// `sleep` is the node's sleep set.
    fn expand(
        &mut self,
        id: OrId,
        sys: &System,
        methods: Option<Vec<ProofMethod>>,
        sleep: &SleepSet,
    ) {
        let ctx = self.ctx;
        let depth = self.graph.node(id).depth as usize;
        if depth >= self.max_depth {
            self.graph.node_mut(id).status = Status::DepthLimit;
            return;
        }
        let path = self.graph.path(self.graph.node(id).first_parent);
        set_breadcrumb_node(id, depth as u32, path);
        let methods = methods.unwrap_or_else(|| {
            set_activity("candidate_methods".to_string());
            let (methods, took) = timed(&mut self.timers.candidate_methods, || {
                candidate_methods(sys, ctx, depth)
                    .unwrap_or_else(|e| panic!("candidate_methods: {e}"))
            });
            self.graph.node_mut(id).methods_secs += took.as_secs_f64();
            methods
        });
        // `candidate_methods` offers exactly `[Finished(r)]` for a terminal
        // (solved/contradictory/unfinishable) system.
        if let [ProofMethod::Finished(result)] = methods.as_slice() {
            self.graph.node_mut(id).status = Status::Finished(result_name(result));
            if self.flags.trace {
                log!("finished #{id} depth={depth}: {}", result_name(result));
            }
            return;
        }
        let methods = if self.sleep_sets {
            let candidates = methods.len();
            let awake: Vec<ProofMethod> = methods
                .into_iter()
                .filter(|m| !goal_nr(sys, m).is_some_and(|nr| sleep.contains(&nr)))
                .collect();
            let skipped = candidates - awake.len();
            self.graph.node_mut(id).asleep = skipped as u32;
            self.sleep_skipped += skipped as u64;
            if awake.is_empty() && skipped > 0 {
                self.graph.node_mut(id).status = Status::Asleep;
                if self.flags.trace {
                    log!("asleep #{id} depth={depth}: all {skipped} candidate(s) asleep");
                }
                return;
            }
            awake
        } else {
            methods
        };
        if self.flags.trace {
            log!(
                "expand #{id} depth={depth} methods={} processed={} graph={} queue={}",
                methods.len(),
                self.processed,
                self.graph.nodes.len(),
                self.queue.len()
            );
        }
        self.graph.node_mut(id).status = Status::Expanded;
        let method_count = methods.len();
        let mut applied = 0usize;
        // The goals of the methods applied here so far: asleep in the
        // children of every later one.
        let mut explored = SleepSet::new();
        for (k, method) in methods.into_iter().enumerate() {
            // `candidate_methods` is in heuristic order, so this keeps the
            // best-ranked applicable methods -- `--top-methods 1` follows
            // the prover's own greedy choice.
            if self.top_methods.is_some_and(|n| applied >= n) {
                self.graph.node_mut(id).untried = (method_count - k) as u32;
                break;
            }
            let text = pretty_proof_method_inline(&method);
            let step = format!("method {}/{method_count} {text}", k + 1);
            if self.flags.trace {
                log!("  exec {}", truncate_for_log(&step, 300));
            }
            set_activity(format!("exec_proof_method {step}"));
            let (exec, took) = timed(&mut self.timers.exec_proof_method, || {
                exec_proof_method(ctx, &method, sys)
                    .unwrap_or_else(|e| panic!("exec_proof_method: {e}"))
            });
            self.graph.node_mut(id).exec_secs += took.as_secs_f64();
            let Some(cases) = exec else {
                self.graph.node_mut(id).inapplicable += 1;
                continue;
            };
            applied += 1;
            let method_id = self.graph.intern_method(text);
            let and_idx = self.graph.node(id).and.len() as u32;
            self.graph.node_mut(id).and.push(AndNode {
                method: method_id,
                kind: method_kind(&method),
                cases: Vec::with_capacity(cases.len()),
            });
            if self.flags.trace {
                log!("    -> {} case(s)", cases.len());
            }
            let child_sleep: SleepSet = sleep.union(&explored).copied().collect();
            for (case_name, child_sys) in cases {
                set_activity(format!("visit case [{case_name}] of {step}"));
                // Pushed with a placeholder target first, so the child's own
                // path (diagnostics, `first_parent`) is already resolvable
                // while it is being canonicalized.
                let slots = &mut self.graph.node_mut(id).and[and_idx as usize].cases;
                let case_idx = slots.len() as u32;
                slots.push((case_name, OrId::MAX));
                let child = self.visit(
                    child_sys,
                    depth + 1,
                    Some((id, and_idx, case_idx)),
                    child_sleep.clone(),
                );
                self.graph.node_mut(id).and[and_idx as usize].cases[case_idx as usize].1 = child;
            }
            if self.sleep_sets
                && let Some(nr) = goal_nr(sys, &method)
            {
                explored.insert(nr);
            }
        }
        // The awake candidates didn't apply (typically `simplify`, ranked
        // last, on a system that is already simplified): nothing is left to
        // explore here but asleep methods.
        if applied == 0 && self.graph.node(id).asleep > 0 {
            self.graph.node_mut(id).status = Status::Asleep;
            if self.flags.trace {
                log!("asleep #{id} depth={depth}: no awake candidate applies");
            }
        }
    }
}

/// The rule instances and non-graph part of `sys`, for `DUMP_CANON_PANIC`.
fn dump_system(sys: &System) {
    eprintln!("{}--- end of system ---", system_text(sys));
}

/// `sys`'s rule instances, edges, `<` atoms and non-graph part, as text.
fn system_text(sys: &System) -> String {
    let mut out = String::from("--- rule instances ---\n");
    for (nid, ru) in sys.nodes_in_map_order() {
        let facts = |fs: &[tamarin_theory::fact::LNFact]| {
            fs.iter().map(pretty_fact).collect::<Vec<_>>().join(", ")
        };
        out += &format!(
            "  {nid} : {}[{}] --[{}]-> [{}]\n",
            rule_name_string(ru),
            facts(&ru.premises),
            facts(&ru.actions),
            facts(&ru.conclusions)
        );
    }
    out += "--- edges ---\n";
    for e in sys.edges.iter() {
        out += &format!(
            "  {}:{} >-> {}:{}\n",
            e.src.0, e.src.1 .0, e.tgt.0, e.tgt.1 .0
        );
    }
    out += "--- less_atoms ---\n";
    for a in sys.less_atoms.iter() {
        out += &format!("  {} < {} ({:?})\n", a.smaller, a.larger, a.reason);
    }
    out += &format!("--- last_atom ---\n{:?}\n", sys.last_atom);
    out += &format!(
        "--- non-graph part ---\n{}\n",
        tamarin_theory::pretty_system::pretty_non_graph_system(sys)
    );
    out
}

/// A [`CanonicalSystem`] as text: its label, then each section.
fn canonical_text(canon: &CanonicalSystem) -> String {
    let (source_kind, side, contradictory) = canon.label();
    let mut out = format!(
        "--- label ---\nsource_kind={source_kind:?} side={side:?} contradictory={contradictory}\n"
    );
    for (name, section) in SYSTEM_SECTIONS.iter().zip(canon.sections()) {
        out.push_str(&format!("--- {name} ---\n{section}\n"));
    }
    out
}

fn dump_formulas(sys: &System, path: &str) {
    eprintln!("--- formulas/solved_formulas at {path} ---");
    for f in sys.formulas.iter() {
        eprintln!(
            "  formula: {}",
            tamarin_theory::pretty_formula::pretty_guarded(f)
        );
    }
    for f in sys.solved_formulas.iter() {
        eprintln!(
            "  solved_formula: {}",
            tamarin_theory::pretty_formula::pretty_guarded(f)
        );
    }
    eprintln!(
        "  nodes: {:?}",
        sys.nodes
            .iter()
            .map(|(n, _)| n.to_string())
            .collect::<Vec<_>>()
    );
    eprintln!("  last_atom: {:?}", sys.last_atom);
    eprintln!("  goals ({} total):", sys.goals.len());
    for (goal, status) in sys.goals.iter() {
        eprintln!(
            "    {goal:?} solved={} looping={}",
            status.solved, status.looping
        );
    }
}

// =============================================================================
// JSON output
// =============================================================================

/// A count as a JSON number, or as a string when it exceeds `u64`.
fn count_value(n: u128) -> Value {
    u64::try_from(n).map_or_else(|_| Value::String(n.to_string()), Value::from)
}

struct Summary {
    graph: u64,
    processed: u64,
    tree: u128,
    lower_bound: bool,
    /// The JSON `sizes` object.
    sizes: Value,
    status_counts: BTreeMap<&'static str, u64>,
    /// Violated size invariants (see the module docs' "Three sizes"),
    /// recorded rather than asserted so the JSON is still written. Expected
    /// after [`StopReason::Panic`] (the occurrence being visited was counted
    /// but never added); otherwise a bug in this tool.
    violations: Vec<String>,
}

fn summarize(graph: &Graph, max_depth: usize) -> Summary {
    let tree_sizes = sizes::tree_sizes(&graph.children(), 0, max_depth);
    let tree: u128 = tree_sizes.by_depth.iter().sum();
    let processed: u64 = graph.processed_by_depth.iter().sum();
    let graph_size = graph.nodes.len() as u64;

    let mut graph_by_min_depth = vec![0u64; graph.processed_by_depth.len()];
    let mut status_counts: BTreeMap<&'static str, u64> = [
        "expanded",
        "finished",
        "depth_limit",
        "unexpanded",
        "asleep",
        "canon_panic",
        "exec_panic",
        "canon_err",
    ]
    .into_iter()
    .map(|s| (s, 0))
    .collect();
    for n in &graph.nodes {
        graph_by_min_depth[n.depth as usize] += 1;
        *status_counts
            .get_mut(n.status.name())
            .expect("every status is listed") += 1;
        if n.canon_err {
            *status_counts.get_mut("canon_err").expect("listed") += 1;
        }
    }
    let lower_bound = status_counts["unexpanded"] > 0
        || status_counts["canon_panic"] > 0
        || status_counts["exec_panic"] > 0;

    let mut violations = Vec::new();
    if !(graph_size <= processed && u128::from(processed) <= tree) {
        violations.push(format!(
            "graph={graph_size} processed={processed} tree={tree}"
        ));
    }
    for (d, &p) in graph.processed_by_depth.iter().enumerate() {
        let t = tree_sizes.by_depth.get(d).copied().unwrap_or(0);
        if u128::from(p) > t {
            violations.push(format!("depth {d}: processed={p} tree={t}"));
        }
    }

    let sizes = json!({
        "graph": graph_size,
        "processed": processed,
        "tree": count_value(tree),
        "graph_by_min_depth": graph_by_min_depth,
        "graph_distinct_at_depth": tree_sizes.distinct_at_depth,
        "processed_by_depth": graph.processed_by_depth,
        "tree_by_depth": tree_sizes.by_depth.iter().map(|&n| count_value(n)).collect::<Vec<_>>(),
    });
    Summary {
        graph: graph_size,
        processed,
        tree,
        lower_bound,
        sizes,
        status_counts,
        violations,
    }
}

/// One hex digest (128 bits) of all of a canonical system's per-field
/// fingerprints, for the JSON: what a snapshot compares to see whether the
/// canonical FORM changed while the merges stayed the same.
fn fingerprint_hex(fp: &CanonicalSystemFingerprint) -> String {
    let mut h = tamarin_utils::fingerprint::FingerprintHasher::new();
    h.tag(&format!("{fp:?}"));
    h.finish()[..16]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn node_json(id: usize, n: &OrNode) -> Value {
    let mut v = json!({
        "id": id,
        "depth": n.depth,
        "status": n.status.name(),
        "canon_err": n.canon_err,
        "fingerprint": n.fingerprint,
        "first_parent": n.first_parent.map(|(p, a, c)| json!([p, a, c])),
        "inapplicable": n.inapplicable,
        "untried": n.untried,
        "asleep": n.asleep,
        "canon_secs": micros(n.canon_secs),
        "methods_secs": micros(n.methods_secs),
        "exec_secs": micros(n.exec_secs),
        "and": n.and.iter().map(|a| json!({
            "method": a.method,
            "kind": a.kind,
            "cases": a.cases.iter().map(|(name, child)| json!([name, child])).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
    });
    if let Status::Finished(result) = n.status {
        v["result"] = json!(result);
    }
    v
}

// =============================================================================
// CLI
// =============================================================================

struct Args {
    theory_path: String,
    lemma: String,
    max_depth: usize,
    max_nodes: u64,
    out: Option<String>,
    no_merge: bool,
    /// `--no-canon`: never canonicalize; implies `no_merge`.
    no_canon: bool,
    /// `--sleep-sets`: skip method orders already explored.
    sleep_sets: bool,
    /// `--top-methods N`: apply only the N best-ranked applicable candidate
    /// methods per node (default: all).
    top_methods: Option<usize>,
    /// `--time-budget SECS`: stop expanding once the process has run this
    /// long (setup included), then write the JSON as usual.
    time_budget: Option<f64>,
    /// `--max-rss-gb GB`: stop expanding once the resident set reaches this
    /// many GiB, then write the JSON as usual.
    max_rss_gb: Option<f64>,
    /// `--setup-timeout SECS`: give up (exit 6, no JSON) if setup takes
    /// longer than this.
    setup_timeout: Option<f64>,
    /// `--heartbeat SECS`: log a progress line every SECS (background thread).
    heartbeat: Option<f64>,
    /// `--trace`: log every expansion and every applied method.
    trace: bool,
    /// `-D=<flag>` / `--auto-sources`: how the theory is loaded.
    load: common::LoadOpts,
}

enum Mode {
    /// `--list-lemmas <theory>`.
    ListLemmas(String, common::LoadOpts),
    Explore(Args),
}

const USAGE: &str = "usage: explore_canonical_matches <theory.spthy> <lemma> \
     [--max-depth N] [--max-nodes N] [--out PATH] [--no-merge] [--no-canon] [--sleep-sets] \
     [--top-methods N] [--time-budget SECS] [--max-rss-gb GB] [--setup-timeout SECS] \
     [--heartbeat SECS] [--trace] [-D=FLAG ...] [--auto-sources]\n       \
     explore_canonical_matches --list-lemmas <theory.spthy> [-D=FLAG ...] [--auto-sources]";

fn usage_error(message: &str) -> ! {
    eprintln!("{message}\n{USAGE}");
    std::process::exit(EXIT_USAGE);
}

/// Takes the load flags (`-D=FLAG`, `--defines=FLAG`, `--auto-sources`) out
/// of `raw`, leaving the rest. As in `tamarin-rs`, a define's value must be
/// attached: a bare `-D` is refused rather than eating the next argument.
fn take_load_opts(raw: &[String]) -> (Vec<String>, common::LoadOpts) {
    let mut rest = Vec::new();
    let mut load = common::LoadOpts::default();
    for a in raw {
        if a == "--auto-sources" {
            load.auto_sources = true;
        } else if let Some(flag) = a
            .strip_prefix("-D=")
            .or_else(|| a.strip_prefix("--defines="))
        {
            load.defines.push(flag.to_string());
        } else if a == "-D" || a == "--defines" {
            usage_error("-D wants its flag attached: -D=FLAG");
        } else {
            rest.push(a.clone());
        }
    }
    (rest, load)
}

fn parse_args(raw: &[String]) -> Mode {
    let (raw, load) = take_load_opts(raw);
    let raw = &raw[..];
    if raw.get(1).map(String::as_str) == Some("--list-lemmas") {
        match raw.get(2) {
            Some(path) if raw.len() == 3 => return Mode::ListLemmas(path.clone(), load),
            _ => usage_error("--list-lemmas wants exactly one theory path"),
        }
    }
    if raw.len() < 3 {
        usage_error("missing <theory.spthy> <lemma>");
    }
    let mut args = Args {
        theory_path: raw[1].clone(),
        lemma: raw[2].clone(),
        max_depth: 4,
        max_nodes: 2000,
        out: None,
        no_merge: false,
        no_canon: false,
        sleep_sets: false,
        top_methods: None,
        time_budget: None,
        max_rss_gb: None,
        setup_timeout: None,
        heartbeat: None,
        trace: false,
        load,
    };
    let mut i = 3;
    while i < raw.len() {
        let flag = raw[i].as_str();
        let value = || {
            raw.get(i + 1)
                .unwrap_or_else(|| usage_error(&format!("{flag} wants a value")))
        };
        let number = |v: &String| -> f64 {
            v.parse()
                .unwrap_or_else(|_| usage_error(&format!("{flag} wants a number, got {v:?}")))
        };
        match flag {
            "--max-depth" => args.max_depth = number(value()) as usize,
            "--max-nodes" => args.max_nodes = number(value()) as u64,
            "--out" => args.out = Some(value().clone()),
            "--time-budget" => args.time_budget = Some(number(value())),
            "--max-rss-gb" => args.max_rss_gb = Some(number(value())),
            "--setup-timeout" => args.setup_timeout = Some(number(value())),
            "--heartbeat" => args.heartbeat = Some(number(value())),
            "--no-merge" => args.no_merge = true,
            "--no-canon" => {
                args.no_canon = true;
                args.no_merge = true;
            }
            "--sleep-sets" => args.sleep_sets = true,
            "--top-methods" => {
                let n = number(value());
                if n < 1.0 || n.fract() != 0.0 {
                    usage_error(&format!("--top-methods wants a positive integer, got {n}"));
                }
                args.top_methods = Some(n as usize);
            }
            "--trace" => args.trace = true,
            other => usage_error(&format!("unrecognized argument: {other}")),
        }
        i += if matches!(
            flag,
            "--no-merge" | "--no-canon" | "--sleep-sets" | "--trace"
        ) {
            1
        } else {
            2
        };
    }
    if args.sleep_sets && !args.no_merge {
        usage_error(
            "--sleep-sets needs --no-merge or --no-canon: a sleep set belongs to a path, \
             and a merged node is reached along several",
        );
    }
    Mode::Explore(args)
}

fn default_out_path(theory_path: &str, lemma: &str) -> String {
    let stem = std::path::Path::new(theory_path)
        .file_stem()
        .map_or_else(|| "theory".into(), |s| s.to_string_lossy());
    format!("explore_{stem}_{lemma}.json")
}

// Exit codes (see the module docs' "Batch runs"). A Rust panic that escapes
// everything below exits with 101.
const EXIT_USAGE: i32 = 2;
const EXIT_SETUP_ERROR: i32 = 3;
const EXIT_PANIC_STOP: i32 = 4;
const EXIT_INVARIANT_VIOLATION: i32 = 5;
const EXIT_SETUP_TIMEOUT: i32 = 6;

/// `--setup-timeout`: setup (elaboration, intruder variants, source
/// precomputation) is one uninterruptible call, so a watchdog thread ends
/// the process if it is still running after `limit`. `done` is set once
/// setup returns.
fn spawn_setup_watchdog(limit: Duration, done: Arc<AtomicBool>) {
    std::thread::spawn(move || {
        let start = Instant::now();
        while !done.load(Ordering::Relaxed) {
            if start.elapsed() >= limit {
                log!(
                    "setup exceeded --setup-timeout {:.0}s, giving up (no JSON)",
                    limit.as_secs_f64()
                );
                std::process::exit(EXIT_SETUP_TIMEOUT);
            }
            std::thread::sleep(Duration::from_millis(200));
        }
    });
}

/// `--list-lemmas`: prints one JSON object describing the theory and its
/// lemmas, for a batch runner to plan jobs from. `error` is non-null (and
/// the exit code [`EXIT_SETUP_ERROR`]) if it can't be read, parsed or
/// elaborated; the fields gathered before that are still printed.
fn list_lemmas(theory_path: &str, load: &common::LoadOpts) -> i32 {
    let mut out = json!({ "theory": theory_path, "error": null });
    let code = match describe_theory(theory_path, load, &mut out) {
        Ok(()) => 0,
        Err(e) => {
            out["error"] = json!(e);
            EXIT_SETUP_ERROR
        }
    };
    println!("{out}");
    code
}

/// Fills `out` for [`list_lemmas`]: `lines`; `diff` (the theory is meant
/// for `--diff` mode: it only parses with the `diff` flag, or it has diff/
/// equivalence lemmas -- the port has no diff-mode prover); `processes`
/// (SAPIC); `rules`; and `lemmas`, exactly the names
/// `ProverSession::context_and_system_for_lemma` can look up -- after SAPIC
/// and accountability translation, so accountability's generated lemmas are
/// listed too. `rules` counts the translated rules.
fn describe_theory(path: &str, load: &common::LoadOpts, out: &mut Value) -> Result<(), String> {
    use tamarin_parser::ast::TheoryItem;
    use tamarin_theory::theory::TraceQuantifier;

    let source = std::fs::read_to_string(path).map_err(|e| format!("read: {e}"))?;
    out["lines"] = json!(source.lines().count());
    let defines: Vec<&str> = load.defines.iter().map(String::as_str).collect();
    let with_diff: Vec<&str> = defines.iter().copied().chain(["diff"]).collect();
    let plain = tamarin_parser::parse_theory(&source, &defines);
    let with_diff_flag = tamarin_parser::parse_theory(&source, &with_diff);
    let diff_items = with_diff_flag.as_ref().map_or(0, |t| {
        t.items
            .iter()
            .filter(|i| {
                matches!(
                    i,
                    TheoryItem::DiffLemma(_)
                        | TheoryItem::EquivLemma(..)
                        | TheoryItem::DiffEquivLemma(_)
                )
            })
            .count()
    });
    out["diff"] = json!(diff_items > 0 || (plain.is_err() && with_diff_flag.is_ok()));
    let parsed = plain.map_err(|e| format!("parse: {e}"))?;
    out["processes"] = json!(parsed.items.iter().any(|i| matches!(
        i,
        TheoryItem::TopLevelProcess(_) | TheoryItem::ProcessDef(_)
    )));
    let translated = catch_unwind(AssertUnwindSafe(|| {
        common::try_load_theory_translated(path, load)
    }))
    .map_err(|p| format!("load panicked: {}", panic_message(&*p)))??;
    let elaborated = &translated.theory;
    out["rules"] = json!(elaborated.rules().count());
    out["lemmas"] = elaborated
        .lemmas()
        .map(|l| {
            json!({
                "name": l.name,
                "trace_quantifier": match l.trace_quantifier {
                    TraceQuantifier::AllTraces => "all-traces",
                    TraceQuantifier::ExistsTrace => "exists-trace",
                },
                "attributes": l.attributes.iter().map(|a| format!("{a:?}")).collect::<Vec<_>>(),
            })
        })
        .collect();
    Ok(())
}

/// Loads the theory as `tamarin-rs --prove` does (`common::
/// try_load_theory_for_proving`: SAPIC and accountability translation, the
/// NDC-checked intruder cache, auto-sources, the prover session) and builds
/// the lemma's proof context and initial system -- the per-lemma setup the
/// prover's proof search starts from. Also returns whether the theory is a
/// SAPIC one. Errors instead of panicking, so a batch log says which step
/// failed.
fn setup(args: &Args) -> Result<(ProverSession, ProofContext, System, bool), String> {
    let loaded = common::try_load_theory_for_proving(&args.theory_path, &args.load)?;
    log!(
        "loaded: {} rule(s) after translation, sapic={}, auto_sources={}; maude started",
        loaded.translated_rules,
        loaded.is_sapic,
        loaded.auto_sources
    );
    let (ctx, initial_sys) = loaded
        .session
        .context_and_system_for_lemma(&args.lemma)
        .map_err(|e| format!("context_and_system_for_lemma({}): {e}", args.lemma))?;
    Ok((loaded.session, ctx, initial_sys, loaded.is_sapic))
}

/// Writes `doc` to `out` via a temporary file and a rename, so a run killed
/// while writing never leaves a truncated JSON behind.
fn write_json_atomically(out: &str, doc: &Value) -> std::io::Result<()> {
    let tmp = format!("{out}.tmp");
    let file = std::fs::File::create(&tmp)?;
    let mut writer = std::io::BufWriter::new(file);
    serde_json::to_writer(&mut writer, doc)?;
    std::io::Write::flush(&mut writer)?;
    std::fs::rename(&tmp, out)
}

fn main() {
    PROCESS_START.get_or_init(Instant::now);
    install_panic_hook();
    let argv: Vec<String> = std::env::args().collect();
    let args = match parse_args(&argv) {
        Mode::ListLemmas(path, load) => std::process::exit(list_lemmas(&path, &load)),
        Mode::Explore(args) => args,
    };

    log!("argv: {argv:?}");
    log!(
        "pid={} cwd={} maude={} BLISS_PATH={:?}",
        std::process::id(),
        std::env::current_dir().map_or_else(|_| "?".to_string(), |d| d.display().to_string()),
        common::maude_binary(),
        std::env::var("BLISS_PATH").ok()
    );
    if !args.no_canon && !bliss_available() {
        eprintln!("bliss not available and TAM_ALLOW_NO_BLISS=1 set -- nothing to do, exiting");
        return;
    }

    // `--heartbeat 0` (or less) means off, not a busy loop.
    if let Some(every) = args.heartbeat.filter(|&secs| secs > 0.0) {
        spawn_heartbeat(Duration::from_secs_f64(every));
    }
    set_phase("setup");
    let setup_done = Arc::new(AtomicBool::new(false));
    if let Some(limit) = args.setup_timeout.filter(|&secs| secs > 0.0) {
        spawn_setup_watchdog(Duration::from_secs_f64(limit), Arc::clone(&setup_done));
    }
    // The session lives as long as the exploration, as the prover keeps its
    // session while proving.
    let (_session, ctx, initial_sys, sapic) = match catch_unwind(AssertUnwindSafe(|| setup(&args)))
    {
        Ok(Ok(setup)) => setup,
        Ok(Err(e)) => {
            log!("setup failed: {e}");
            std::process::exit(EXIT_SETUP_ERROR);
        }
        Err(payload) => {
            log!("setup panicked: {}", panic_message(&*payload));
            std::process::exit(EXIT_SETUP_ERROR);
        }
    };
    setup_done.store(true, Ordering::Relaxed);
    let setup_secs = elapsed_secs();
    log!(
        "setup done in {setup_secs:.2}s, peak rss {}MiB",
        memory_mib("VmHWM:").map_or_else(|| "?".to_string(), |m| m.to_string())
    );

    set_phase("explore");
    let mut explorer = Explorer {
        ctx: &ctx,
        flags: Flags {
            progress: env_gate!("PROGRESS"),
            trace: args.trace,
            dump_formulas: env_gate!("DUMP_FORMULAS"),
            profile: env_gate!("PROFILE"),
            dump_canon_panic: env_gate!("DUMP_CANON_PANIC"),
            dump_merges_into: std::env::var("DUMP_MERGES_INTO").ok().map(|v| {
                let ids = v
                    .split(',')
                    .map(|s| {
                        s.trim()
                            .parse()
                            .expect("DUMP_MERGES_INTO: comma-separated node ids")
                    })
                    .collect();
                let out = args
                    .out
                    .clone()
                    .unwrap_or_else(|| default_out_path(&args.theory_path, &args.lemma));
                (ids, out)
            }),
            dump_term_graph_at: std::env::var("DUMP_TERM_GRAPH_AT")
                .ok()
                .and_then(|v| v.parse().ok())
                .map(|n| {
                    let out = args
                        .out
                        .clone()
                        .unwrap_or_else(|| default_out_path(&args.theory_path, &args.lemma));
                    (n, out)
                }),
            term_graph_check: std::env::var("TERM_GRAPH_CHECK").as_deref() != Ok("0"),
        },
        canon_summary: CanonSummary::default(),
        max_depth: args.max_depth,
        no_merge: args.no_merge,
        no_canon: args.no_canon,
        sleep_sets: args.sleep_sets,
        top_methods: args.top_methods,
        graph: Graph::default(),
        by_fingerprint: FastMap::default(),
        queue: VecDeque::new(),
        sleep_skipped: 0,
        processed: 0,
        canonicalize_failures: 0,
        method_check: MethodCheck::default(),
        timers: StageTimers::default(),
        started: Instant::now(),
        time_budget: args.time_budget.map(Duration::from_secs_f64),
        max_rss_mib: args.max_rss_gb.map(|gb| (gb * 1024.0) as u64),
        canon_failure_examples: Vec::new(),
        panic: None,
    };
    // The root's own visit runs outside `run`'s per-expansion panic capture;
    // with no node to attach a partial result to, a panic there only logs.
    match catch_unwind(AssertUnwindSafe(|| {
        explorer.visit(initial_sys, 0, None, SleepSet::new())
    })) {
        Ok(root) => assert_eq!(root, 0, "the root is the first OR node"),
        Err(payload) => {
            log!("visiting the root PANICKED: {}", panic_message(&*payload));
            std::process::exit(EXIT_PANIC_STOP);
        }
    }
    let stop = explorer.run(args.max_nodes);
    let explore_secs = elapsed_secs() - setup_secs;

    set_phase("write");
    let summary = summarize(&explorer.graph, args.max_depth);
    for v in &summary.violations {
        log!("size invariant violated: {v}");
    }
    let nodes: Vec<Value> = explorer
        .graph
        .nodes
        .iter()
        .enumerate()
        .map(|(id, n)| node_json(id, n))
        .collect();
    let check = &explorer.method_check;
    let peak_rss_mib = memory_mib("VmHWM:");
    let doc = json!({
        "schema_version": 4,
        "theory": args.theory_path,
        "lemma": args.lemma,
        "argv": argv,
        "load": {
            "defines": args.load.defines,
            "auto_sources": args.load.auto_sources,
            "sapic": sapic,
        },
        "params": {
            "max_depth": args.max_depth,
            "max_nodes": args.max_nodes,
            "no_merge": args.no_merge,
            "no_canon": args.no_canon,
            "sleep_sets": args.sleep_sets,
            "top_methods": args.top_methods,
            "time_budget": args.time_budget,
            "max_rss_gb": args.max_rss_gb,
            "setup_timeout": args.setup_timeout,
        },
        "stop_reason": stop.name(),
        "truncated": stop != StopReason::Exhausted,
        "lower_bound": summary.lower_bound,
        "timing": {
            "setup_secs": setup_secs,
            "explore_secs": explore_secs,
            "canonicalize_secs": explorer.timers.canonicalize.as_secs_f64(),
            "canonicalize_merged_secs": explorer.timers.canonicalize_merged.as_secs_f64(),
            "fingerprint_secs": explorer.timers.fingerprint.as_secs_f64(),
            "candidate_methods_secs": explorer.timers.candidate_methods.as_secs_f64(),
            "exec_proof_method_secs": explorer.timers.exec_proof_method.as_secs_f64(),
            "method_check_secs": explorer.timers.method_check.as_secs_f64(),
        },
        "peak_rss_mib": peak_rss_mib,
        "sizes": summary.sizes,
        "status_counts": summary.status_counts,
        "canonicalize": {
            "failures": explorer.canonicalize_failures,
            "examples": explorer.canon_failure_examples,
        },
        "method_check": {
            "checked": check.checked,
            "mismatches": check.mismatches,
            "failures": check.failures,
            "examples": check.examples,
        },
        "canon": explorer.canon_summary.json(),
        "panic": explorer.panic,
        "sleep": args.sleep_sets.then(|| json!({
            "skipped_methods": explorer.sleep_skipped,
            "asleep_nodes": summary.status_counts["asleep"],
        })),
        "invariant_violations": summary.violations,
        "methods": explorer.graph.methods,
        "nodes": nodes,
    });

    let out = args
        .out
        .clone()
        .unwrap_or_else(|| default_out_path(&args.theory_path, &args.lemma));
    if let Err(e) = write_json_atomically(&out, &doc) {
        log!("writing {out} failed: {e}");
        std::process::exit(EXIT_SETUP_ERROR);
    }

    println!(
        "=== {} :: {} === graph={} processed={} tree={} canonicalize_failures={} \
         method_mismatches={}/{} method_canon_failures={} \
         stop_reason={} lower_bound={} (max_depth={}, max_nodes={}{}) -> {out}",
        args.theory_path,
        args.lemma,
        summary.graph,
        summary.processed,
        summary.tree,
        explorer.canonicalize_failures,
        explorer.method_check.mismatches,
        explorer.method_check.checked,
        explorer.method_check.failures,
        stop.name(),
        summary.lower_bound,
        args.max_depth,
        args.max_nodes,
        if args.no_canon {
            ", no_canon"
        } else if args.no_merge {
            ", no_merge"
        } else {
            ""
        }
    );
    if args.sleep_sets {
        println!(
            "sleep sets: {} candidate method(s) skipped, {} node(s) asleep",
            explorer.sleep_skipped, summary.status_counts["asleep"]
        );
    }
    if !args.no_canon {
        println!("  canon: {}", explorer.canon_summary.line());
    }

    if explorer.flags.profile {
        let t = &explorer.timers;
        let total = t.canonicalize
            + t.fingerprint
            + t.candidate_methods
            + t.exec_proof_method
            + t.method_check;
        println!(
            "\n--- PROFILE: wall time by pipeline stage, summed over {} occurrence(s) ---",
            summary.processed
        );
        for (label, d) in [
            (
                "canonicalize_constraint_system (term graph):",
                t.canonicalize,
            ),
            ("fingerprint + dedup lookup:", t.fingerprint),
            (
                "candidate_methods (ranking, incl. is_finished):",
                t.candidate_methods,
            ),
            (
                "exec_proof_method (solving -- maude/AC here):",
                t.exec_proof_method,
            ),
            (
                "merge check (methods of merged + signatures):",
                t.method_check,
            ),
        ] {
            println!(
                "  {label:<47}{:>8.2}s ({:>5.1}%)",
                d.as_secs_f64(),
                pct(d, total)
            );
        }
        println!(
            "  sum of measured stages: {:.2}s (wall clock may exceed this by parse/elaborate/setup)",
            total.as_secs_f64()
        );
    }

    let code = if stop == StopReason::Panic {
        EXIT_PANIC_STOP
    } else if !summary.violations.is_empty() {
        EXIT_INVARIANT_VIOLATION
    } else {
        0
    };
    log!("done: stop_reason={} exit={code}", stop.name());
    std::process::exit(code);
}
