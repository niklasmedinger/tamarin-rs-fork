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
//! child. Each child is canonicalized (`canonicalize_constraint_system`)
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
//!   merged run's `tree`.
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
//! compact JSON object, `schema_version` 3 (use `jq .` to read it), via a
//! temporary file and a rename (a killed run never leaves a truncated one):
//! `theory`, `lemma`, `argv`, `params`, `stop_reason` (`exhausted`,
//! `max_nodes`, `time_budget`, `max_rss` or `panic`), `truncated` (anything but
//! `exhausted`), `lower_bound` (some node is unexpanded or failed to
//! canonicalize or expand, so `tree` undercounts), `timing` (`setup_secs`,
//! `explore_secs`), `peak_rss_mib`, `canonicalize` (`failures` and up to 20
//! `examples`, each with its `kind` -- `panic` or `error` --, `path` and
//! `message`), `panic` (null, or where expanding a node panicked: `node`,
//! `path`, `activity`, `message`), `invariant_violations` (see "Three
//! sizes"; expected only after a `panic` stop), `sizes` (the three totals plus
//! `graph_by_min_depth`, `graph_distinct_at_depth`, `processed_by_depth`,
//! `tree_by_depth`; counts above `u64::MAX` are strings), `status_counts`,
//! `method_check` (`checked` merges, `mismatches`, `failures` to canonicalize
//! a method, and up to 20 `examples`, each with the `representative`'s and
//! the `duplicate`'s depth and path as `[{method, case}, …]` steps plus the
//! methods only on either side), `methods` (interned method text, including
//! candidate methods that were never applied) and `nodes`: per OR node its `depth`
//! (min depth), `status` (`expanded`, `finished` + `result`, `depth_limit`,
//! `unexpanded`, `canon_panic`, `exec_panic`), `canon_err` (canonicalization returned an
//! error: expanded normally but never merged into), `first_parent`
//! (`[parent, and_index, case_index]` of the occurrence that created it),
//! `inapplicable`, and `and` (`method` index, `kind`, `cases` as
//! `[name, child_id]` pairs).
//!
//! Usage: `cargo run --example explore_canonical_matches -- <theory.spthy> <lemma> [--max-depth N] [--max-nodes N] [--out PATH] [--no-merge] [--time-budget SECS] [--max-rss-gb GB] [--heartbeat SECS] [--trace]`
//! or `... -- --list-lemmas <theory.spthy>`.
//!
//! - `--max-depth` (default 4): don't expand nodes at this depth.
//! - `--max-nodes` (default 2000): stop expanding once this many
//!   occurrences have been canonicalized. Checked before each expansion, so
//!   it can be exceeded by one node's fan-out; queued nodes stay
//!   `unexpanded`.
//! - `--time-budget SECS`: likewise, stop expanding once the process has
//!   run this long (setup included); the JSON is written as usual. Also
//!   checked only between expansions, so one expansion can overrun it.
//! - `--max-rss-gb GB`: likewise, stop expanding once the resident set
//!   reaches GB GiB (read from `/proc/self/status`), so a run approaching
//!   an external memory limit still writes its JSON instead of being
//!   killed. One expansion can overshoot it (a large `splitEqs` can take
//!   over 1 GB), so leave headroom below any hard limit.
//! - `--heartbeat SECS`: log a progress line (processed, graph, queue,
//!   rate, RSS, and the current node's step and how long it has run) every
//!   SECS, from a background thread, so a single long step (one
//!   `exec_proof_method` can take minutes) is visible as such.
//! - `--trace`: log every expansion and every applied method, so a run
//!   killed from outside still shows where it was.
//! - `--list-lemmas`: print one JSON object describing the theory instead
//!   of exploring: `lines`, `diff` (only parses with the `diff` flag, or has
//!   diff/equivalence lemmas; the port has no diff-mode prover),
//!   `processes` (SAPIC), `rules`, `lemmas` (`name`, `trace_quantifier`,
//!   `attributes`, `modulo`) and `error`.
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
//! a panic (partial JSON), 5 a size invariant failed (JSON written), 101 an
//! uncaught panic.
//!
//! Env vars (all opt-in, unset = off), all terminal-only:
//! - `PROGRESS=1` -- stderr heartbeat every 20 canonicalizations (elapsed,
//!   rate, graph size, queue length), to tell "still working" from "hung".
//!   For a DH-heavy theory such as `wireguard.spthy` the bottleneck is
//!   `candidate_methods`' enormous branching factor.
//! - `DUMP_FORMULAS=1` -- dump `sys.formulas`/`sys.solved_formulas`/
//!   `sys.nodes`/`sys.last_atom`/`sys.goals` for every occurrence, to stderr.
//! - `PROFILE=1` -- wall time per stage: `canonicalize_constraint_system`,
//!   fingerprint + dedup lookup, `candidate_methods` (ranking, including
//!   its `is_finished` check; computed for every occurrence, merged ones
//!   too, for the method check), and `exec_proof_method` (solving -- the one most likely to
//!   route through `MaudeHandle`). The first two are this tool's layer, the
//!   last two Tamarin's own search.
//! - `PROFILE_CANON=1` -- breaks the canonicalization bucket down into its
//!   internal stages: Stage A graph extraction, Stage C's external `bliss`
//!   subprocess (dimacs, spawn and parse), Stage F's automorphism GROUP
//!   CLOSURE and group MINIMIZATION, Stage D's per-survivor re-seeding, and
//!   Stage G (formulas/solved_formulas/lemmas/eq_store.subst/eq_store.conj/
//!   subterm_store/goals, each timed and counted via
//!   `canon::canonicalize_system_content_seeded_profiled`). **Stage G runs
//!   ONCE PER SURVIVOR**, like `canonicalize_constraint_system` itself -- a
//!   node with many tied survivors pays it that many times, usually the real
//!   explanation for an outsized Stage G total. Also tracks graph size,
//!   `|generators|`/`|Aut(G)|`/`|survivors|`, which `VertexKind`s the
//!   automorphism generators move (and whether the swapped contents are
//!   equal up to renaming), and how far candidate encodings would shrink the
//!   group (`Refinements`): a refinement only splits color classes, so its
//!   group is exactly the subset of the current closed group preserving it,
//!   compared against the content automorphisms (the survivors, the floor
//!   for any encoding). Re-runs the ENTIRE pipeline a second time to get the
//!   sub-timings, so expect roughly double the wall time.
//! - `DUMP_AUTOMORPHISMS=N` (requires `PROFILE_CANON=1`) -- prints up to N
//!   occurrences per category (`genuine`, `fixed_by_skeleton`,
//!   `fixed_by_local_content`, `fixed_by_literal_incidence`,
//!   `fixed_by_positional_incidence`, `residual`: the cheapest candidate
//!   that would leave exactly the content automorphisms) with their path, the
//!   group sizes under each candidate, every generator's moved content
//!   vertices, and the `<` constraints of moved dummy timepoints.
//! - `DUMP_DUMMY_SWAPS=1` (requires `PROFILE_CANON=1`) -- prints the path to
//!   the first 3 occurrences whose automorphism generators move ONLY
//!   `Dummy`/`LessRelation` vertices, plus their `sys.nodes`/
//!   `sys.less_atoms`, so one can be reproduced in the GUI. Raw NodeIds are
//!   internal to this run; compare the printed `sys.nodes`/`less_atoms`
//!   against what the GUI shows at the same point in the path.
//!
//! Requires `bliss` on `PATH` (or `$BLISS_PATH`) -- skips (via
//! `bliss_available()`'s panic-unless-opted-out gate) if
//! `TAM_ALLOW_NO_BLISS=1` is set.

// Example/dev tool: prints results to stdout by design; allow the
// `disallowed_macros` convention freeze for this example binary.
#![allow(clippy::disallowed_macros)]

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use tamarin_theory::bliss_proc::{
    bliss_available, canonical_edges, canonical_vertex_order, generate_group, graph_part_to_dimacs,
    run_bliss, Permutation,
};
use tamarin_term::alpha_eq_ac::{canonicalize_alpha_eq_ac, CanonLabelling};
use tamarin_term::fingerprint::{fingerprint_term, Fingerprint};
use tamarin_term::lterm::{HasFrees, LNTerm, LSort, LVar};
use tamarin_term::term::Term;
use tamarin_term::vterm::Lit;
use tamarin_theory::canon::{
    canonicalize_constraint_system_with_labelling, canonicalize_graph_part_seeded,
    canonicalize_graph_part, canonicalize_proof_method, canonicalize_system_content_seeded_profiled,
    fact_to_term,
    minimal_graph_part_labelings, rule_to_term, ContentStageTimes,
};
use tamarin_theory::canon_color::{Color, ColorTable};
use tamarin_theory::pretty_system::pretty_fact;
use tamarin_theory::rule::rule_name_string;
use tamarin_theory::canon_fingerprint::{
    fingerprint_constraint_system, fingerprint_proof_method, CanonicalSystemFingerprint,
};
use tamarin_theory::canon_graph::{extract_graph_part, GraphPart, VertexKind};
use tamarin_theory::constraint::solver::context::{CutStrategy, ProofContext};
use tamarin_theory::constraint::solver::proof_method::{
    exec_proof_method, ProofMethod, Result as FinishedResult,
};
use tamarin_theory::constraint::solver::search::candidate_methods;
use tamarin_theory::constraint::system::System;
use tamarin_theory::pretty_theory::pretty_proof_method_inline;
use tamarin_theory::prove::{build_lemma_proof_context, CliHeuristic};
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
    PROCESS_START.get_or_init(Instant::now).elapsed().as_secs_f64()
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
        if b.activity.is_empty() { "-" } else { &b.activity },
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
                truncate_for_log(&b.activity, 300).to_string(),
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
    /// `(parent, and_index, case_index)` of the occurrence that created it;
    /// `None` for the root.
    first_parent: Option<(OrId, u32, u32)>,
    /// Candidate methods whose `exec_proof_method` returned `None`.
    inapplicable: u32,
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
            first_parent,
            inapplicable: 0,
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
            .map(|n| n.and.iter().flat_map(|a| a.cases.iter().map(|&(_, c)| c)).collect())
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
        ProofMethod::RawSolve(_) => "raw_solve",
    }
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
        TreeSizes { by_depth, distinct_at_depth }
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
// Profiling (`PROFILE`, `PROFILE_CANON`) -- terminal only
// =============================================================================

/// Runs `f`, adding its wall time to `acc` when `on`.
fn timed<T>(on: bool, acc: &mut Duration, f: impl FnOnce() -> T) -> T {
    let start = Instant::now();
    let out = f();
    if on {
        *acc += start.elapsed();
    }
    out
}

/// `part` as a percentage of `total` (0 when `total` is zero).
fn pct(part: Duration, total: Duration) -> f64 {
    if total.is_zero() {
        0.0
    } else {
        100.0 * part.as_secs_f64() / total.as_secs_f64()
    }
}

/// `PROFILE_CANON=1`'s per-node breakdown of `canonicalize_constraint_system`'s
/// OWN internal stages -- see [`profile_canonicalize_stages`]. `content`
/// sums [`ContentStageTimes`]' durations over every survivor (Stage G runs
/// once PER SURVIVOR in production, not once per node -- the real reason a
/// large `survivors` count multiplies Stage G's total cost).
#[derive(Default)]
struct CanonStageTimes {
    extract: Duration,
    bliss: Duration,
    group_closure: Duration,
    group_minimization: Duration,
    /// `canonical_vertex_order`+`canonical_edges`+`canonicalize_graph_part_seeded`,
    /// summed across every survivor -- the same AC-canonicalization
    /// machinery as `group_minimization`, but seeded and re-run per
    /// SURVIVOR rather than per raw GROUP element.
    stage_d_per_survivor: Duration,
    content: ContentStageTimes,
    vertices: usize,
    edges: usize,
    generators: usize,
    group_size: usize,
    survivors: usize,
    /// What KIND of vertex the node's generators actually move -- see
    /// [`classify_generator_moves`].
    swaps: SwapKindCounts,
    /// `Some("<nid_a> <-> <nid_b>")` when this node's generators move
    /// ONLY `Dummy`/`LessRelation` vertices (no `RuleInstance`/`Action`/
    /// other structural kind at all) -- naming one concrete swapped pair
    /// of `Dummy` `NodeId`s, so `DUMP_DUMMY_SWAPS=1` can report exactly
    /// which timepoints these are for a node worth inspecting further.
    dummy_less_only_example: Option<String>,
    group_sizes: GroupSizes,
    /// Local content + positional literal incidence keeps exactly the
    /// content automorphisms (same set, not just the same size).
    positional_exact: bool,
    /// Time spent on the refinement analysis -- not canonicalization, so
    /// excluded from the stage sum and subtracted from the wall cross-check.
    analysis: Duration,
    /// [`automorphism_report`], when requested and the group is nontrivial.
    report: Option<String>,
}

/// Per-`VertexKind`-label counts of vertex SLOTS moved by a node's
/// automorphism generators (a transposition contributes 2, a 3-cycle 3,
/// etc. -- see [`classify_generator_moves`]), split by whether the moved
/// vertex's content is equal to its image's UP TO RENAMING + AC (same local
/// canonical form, [`Refinements::local`]). A move between locally different
/// contents is one a content-aware coloring would forbid; content-free
/// structural vertices (`Dummy`, `LessRelation`, ...) always count as same.
#[derive(Debug, Default, Clone)]
struct SwapKindCounts {
    same_local: BTreeMap<&'static str, u64>,
    different_local: BTreeMap<&'static str, u64>,
}

impl SwapKindCounts {
    fn merge(&mut self, other: &SwapKindCounts) {
        for (k, v) in &other.same_local {
            *self.same_local.entry(k).or_insert(0) += v;
        }
        for (k, v) in &other.different_local {
            *self.different_local.entry(k).or_insert(0) += v;
        }
    }

    /// Every label with at least one moved slot.
    fn labels(&self) -> BTreeSet<&'static str> {
        self.same_local
            .keys()
            .chain(self.different_local.keys())
            .copied()
            .collect()
    }
}

/// Candidate refinements of the vertex coloring, per vertex of one graph
/// part. A refinement keeps the graph and only splits color classes, so the
/// automorphism group bliss would report under it is exactly the subset of
/// the current (closed) group that preserves it -- evaluated here without
/// touching `canon_color.rs`. Every key is invariant under the renaming + AC
/// equivalence canonicalization works modulo; a key that weren't would make
/// equivalent systems canonicalize differently.
struct Refinements {
    /// Current color + the content's skeleton: its local canonical form with
    /// every variable replaced by one placeholder per sort (names are kept:
    /// they are never renamed).
    skeleton: Vec<(Color, Option<Fingerprint>)>,
    /// Current color + the content's local canonical form (canonicalized on
    /// its own, modulo renaming + AC) -- the finest per-vertex invariant.
    local: Vec<(Color, Option<Fingerprint>)>,
    /// Per distinct raw variable in vertex content: its sort and the sorted
    /// indices of the vertices containing it -- what variable vertices with
    /// incidence edges would add to the graph. Sorted. Names are left out:
    /// a name is content, already part of the skeleton.
    occurrences: Vec<(LSort, Vec<usize>)>,
    /// Like `occurrences`, but each occurrence also carries its argument path
    /// inside the vertex's term (unordered below AC/commutative symbols) --
    /// literal vertices whose incidence edges are labelled by position.
    positions: Vec<(LSort, Vec<(usize, String)>)>,
}

/// Records every variable occurrence in `t` (content of `vertex`) with its
/// argument path: `symbol/index;` per step, `symbol/*;` below an AC or
/// commutative symbol, whose argument order is not invariant.
fn collect_positions(
    t: &LNTerm,
    vertex: usize,
    path: &mut String,
    out: &mut BTreeMap<LVar, Vec<(usize, String)>>,
) {
    use std::fmt::Write as _;
    match t {
        Term::Lit(Lit::Var(v)) => out.entry(*v).or_default().push((vertex, path.clone())),
        Term::Lit(Lit::Con(_)) => {}
        Term::App(sym, args) => {
            let unordered = sym.is_ac() || sym.is_c();
            for (i, a) in args.iter().enumerate() {
                let len = path.len();
                if unordered {
                    write!(path, "{sym:?}/*;").ok();
                } else {
                    write!(path, "{sym:?}/{i};").ok();
                }
                collect_positions(a, vertex, path, out);
                path.truncate(len);
            }
        }
    }
}

/// The term a content vertex contributes to the graph-part term (see
/// `canon::graph_part_to_term`); `None` for content-free vertices.
fn content_term(v: &VertexKind) -> Option<LNTerm> {
    match v {
        VertexKind::RuleInstance(_, ru) => Some(rule_to_term(ru)),
        VertexKind::Action(_, fact) => Some(fact_to_term(fact)),
        _ => None,
    }
}

impl Refinements {
    fn new(part: &GraphPart) -> Self {
        let n = part.vertices.len();
        let mut skeleton = Vec::with_capacity(n);
        let mut local = Vec::with_capacity(n);
        let mut occurrences: BTreeMap<LVar, Vec<usize>> = BTreeMap::new();
        let mut positions: BTreeMap<LVar, Vec<(usize, String)>> = BTreeMap::new();
        for (i, v) in part.vertices.iter().enumerate() {
            let color = part.vertex_color(i);
            let Some(raw) = content_term(v) else {
                skeleton.push((color, None));
                local.push((color, None));
                continue;
            };
            let canonical = canonicalize_alpha_eq_ac(&raw);
            let skel = canonical.clone().map_free(&mut |v| LVar::new("_", v.sort, 0));
            skeleton.push((color, Some(fingerprint_term(&skel))));
            local.push((color, Some(fingerprint_term(&canonical))));
            let mut raw_vars = BTreeSet::new();
            raw.for_each_free(&mut |v| {
                raw_vars.insert(*v);
            });
            for v in raw_vars {
                occurrences.entry(v).or_default().push(i);
            }
            collect_positions(&raw, i, &mut String::new(), &mut positions);
        }
        let mut occurrences: Vec<(LSort, Vec<usize>)> =
            occurrences.into_iter().map(|(v, vs)| (v.sort, vs)).collect();
        occurrences.sort();
        let mut positions: Vec<(LSort, Vec<(usize, String)>)> = positions
            .into_iter()
            .map(|(v, mut ps)| {
                ps.sort();
                (v.sort, ps)
            })
            .collect();
        positions.sort();
        Refinements {
            skeleton,
            local,
            occurrences,
            positions,
        }
    }

    /// Like [`Self::preserves_occurrences`], with positions: whether some
    /// literal permutation maps every (vertex, path) occurrence onto
    /// (`g`(vertex), path).
    fn preserves_positions(&self, g: &Permutation) -> bool {
        let mut mapped: Vec<(LSort, Vec<(usize, String)>)> = self
            .positions
            .iter()
            .map(|(sort, ps)| {
                let mut image: Vec<(usize, String)> =
                    ps.iter().map(|(v, p)| (g.image_of(*v), p.clone())).collect();
                image.sort();
                (*sort, image)
            })
            .collect();
        mapped.sort();
        mapped == self.positions
    }

    /// Whether some literal permutation maps every vertex's literal set onto
    /// its image's under `g`, i.e. whether `g` survives literal vertices.
    fn preserves_occurrences(&self, g: &Permutation) -> bool {
        let mut mapped: Vec<(LSort, Vec<usize>)> = self
            .occurrences
            .iter()
            .map(|(sort, vs)| {
                let mut image: Vec<usize> = vs.iter().map(|&v| g.image_of(v)).collect();
                image.sort_unstable();
                (*sort, image)
            })
            .collect();
        mapped.sort();
        mapped == self.occurrences
    }
}

/// Whether `g` maps every vertex to one with the same key.
fn preserves<K: PartialEq>(keys: &[K], g: &Permutation) -> bool {
    (0..keys.len()).all(|v| keys[g.image_of(v)] == keys[v])
}

/// Automorphism group sizes of one graph part under the current coloring
/// and each candidate refinement ([`Refinements`]), plus the survivor count
/// -- the size of the CONTENT automorphism group, i.e. what a complete
/// encoding would reduce `|Aut(G)|` to.
#[derive(Debug, Default, Clone, Copy)]
struct GroupSizes {
    current: usize,
    skeleton: usize,
    local: usize,
    local_occurrences: usize,
    local_positions: usize,
    survivors: usize,
}

impl GroupSizes {
    /// The cheapest candidate that makes the group equal the content
    /// automorphisms.
    fn category(&self) -> &'static str {
        if self.current == self.survivors {
            "genuine"
        } else if self.skeleton == self.survivors {
            "fixed_by_skeleton"
        } else if self.local == self.survivors {
            "fixed_by_local_content"
        } else if self.local_occurrences == self.survivors {
            "fixed_by_literal_incidence"
        } else if self.local_positions == self.survivors {
            "fixed_by_positional_incidence"
        } else {
            "residual"
        }
    }
}

fn truncate(s: String, max: usize) -> String {
    if s.chars().count() <= max {
        return s;
    }
    let cut: String = s.chars().take(max).collect();
    format!("{cut}…")
}

fn vertex_label(v: &VertexKind) -> Option<String> {
    Some(match v {
        VertexKind::RuleInstance(nid, ru) => format!("rule {} @ {nid}", rule_name_string(ru)),
        VertexKind::Action(nid, fact) => {
            format!("action {} @ {nid}", truncate(pretty_fact(fact), 110))
        }
        VertexKind::Dummy(nid) => format!("dummy {nid}"),
        _ => return None,
    })
}

/// A human-readable account of `part`'s automorphism generators: for each,
/// which refinements it survives and which content vertices it moves, plus
/// the `<` constraints of every moved `Dummy` timepoint (where they come
/// from is usually the reason the timepoints look alike).
fn automorphism_report(
    sys: &System,
    part: &GraphPart,
    generators: &[Permutation],
    refinements: &Refinements,
    sizes: GroupSizes,
) -> String {
    use std::fmt::Write as _;
    let mut out = format!(
        "|Aut(G)|={} skeleton={} local={} local+incidence={} local+positional={} survivors={}",
        sizes.current,
        sizes.skeleton,
        sizes.local,
        sizes.local_occurrences,
        sizes.local_positions,
        sizes.survivors
    );
    let mut moved_dummies = BTreeSet::new();
    for (k, g) in generators.iter().enumerate() {
        let yes_no = |b: bool| if b { "kept" } else { "removed" };
        let local = preserves(&refinements.local, g);
        write!(
            out,
            "\n  generator {} -- skeleton: {}, local: {}, +incidence: {}, +positional: {}",
            k + 1,
            yes_no(preserves(&refinements.skeleton, g)),
            yes_no(local),
            yes_no(local && refinements.preserves_occurrences(g)),
            yes_no(local && refinements.preserves_positions(g)),
        )
        .ok();
        for v in 0..part.vertices.len() {
            let w = g.image_of(v);
            // Each transposition once; longer cycles as individual arrows.
            if w == v || (g.image_of(w) == v && w < v) {
                continue;
            }
            let (Some(from), Some(to)) = (vertex_label(&part.vertices[v]), vertex_label(&part.vertices[w]))
            else {
                continue;
            };
            let arrow = if g.image_of(w) == v { "<->" } else { "->" };
            write!(out, "\n      {from}  {arrow}  {to}").ok();
            for x in [v, w] {
                if let VertexKind::Dummy(nid) = &part.vertices[x] {
                    moved_dummies.insert(*nid);
                }
            }
        }
    }
    if !moved_dummies.is_empty() {
        write!(out, "\n  < constraints of moved dummy timepoints:").ok();
        for la in sys.less_atoms_in_set_order() {
            if moved_dummies.contains(&la.smaller) || moved_dummies.contains(&la.larger) {
                write!(out, "\n      {} < {}  ({:?})", la.smaller, la.larger, la.reason).ok();
            }
        }
    }
    out
}

/// The coarse label [`SwapKindCounts`] buckets by -- content-free
/// structural markers vs. real rule-instance/action content.
fn vertex_kind_label(v: &VertexKind) -> &'static str {
    match v {
        VertexKind::RuleInstance(_, _) => "RuleInstance",
        VertexKind::Action(_, _) => "Action",
        VertexKind::Dummy(_) => "Dummy",
        VertexKind::EdgeRelation(_, _) => "EdgeRelation",
        VertexKind::LessRelation => "LessRelation",
        VertexKind::AtTimepointRelation => "AtTimepointRelation",
        VertexKind::LastAtomRelation => "LastAtomRelation",
    }
}

/// Every `(vertex, image)` pair some generator actually moves -- fixed
/// points skipped. Walks bliss's own minimal generating set, not the full
/// closed group: a compound element's moved-vertex set is already a subset
/// of the union of the generators' own, so this is representative without
/// walking the whole group.
fn moved_slots<'a>(
    part: &'a GraphPart,
    generators: &'a [Permutation],
) -> impl Iterator<Item = (&'a VertexKind, &'a VertexKind)> + 'a {
    generators.iter().flat_map(move |g| {
        (0..part.vertices.len()).filter_map(move |v| {
            let target = g.image_of(v);
            (target != v).then(|| (&part.vertices[v], &part.vertices[target]))
        })
    })
}

/// Classifies every vertex slot `part`'s automorphism generators move, by
/// `VertexKind` and by whether the moved vertex's content equals its
/// image's up to renaming + AC (`local` keys, [`Refinements::local`]).
fn classify_generator_moves(
    part: &GraphPart,
    generators: &[Permutation],
    local: &[(Color, Option<Fingerprint>)],
) -> SwapKindCounts {
    let mut counts = SwapKindCounts::default();
    for g in generators {
        for v in 0..part.vertices.len() {
            let w = g.image_of(v);
            if w == v {
                continue;
            }
            let bucket = if local[v] == local[w] {
                &mut counts.same_local
            } else {
                &mut counts.different_local
            };
            *bucket.entry(vertex_kind_label(&part.vertices[v])).or_insert(0) += 1;
        }
    }
    counts
}

/// See `CanonStageTimes::dummy_less_only_example`.
fn find_dummy_less_only_example(
    part: &GraphPart,
    generators: &[Permutation],
    swaps: &SwapKindCounts,
) -> Option<String> {
    let labels = swaps.labels();
    if labels.is_empty() || !labels.iter().all(|k| *k == "Dummy" || *k == "LessRelation") {
        return None;
    }
    moved_slots(part, generators).find_map(|slot| match slot {
        (VertexKind::Dummy(nid_a), VertexKind::Dummy(nid_b)) => Some(format!("{nid_a} <-> {nid_b}")),
        _ => None,
    })
}

/// Re-derives `canonicalize_constraint_system`'s FULL pipeline from
/// `canon.rs`'s `pub` building blocks: `extract_graph_part` ->
/// `graph_part_to_dimacs`+`run_bliss` -> `generate_group` ->
/// `minimal_graph_part_labelings` -> Stage G's
/// `canonicalize_system_content_seeded_profiled`, run ONCE PER SURVIVOR
/// exactly like `canonicalize_constraint_system` itself does -- purely to
/// measure where the time inside it goes.
///
/// Returns `None` for the two failure cases `canonicalize_constraint_system`
/// itself would return `Err` for (both extremely unlikely given the real
/// call just succeeded moments before this runs on the SAME `sys`) --
/// the caller should just skip recording stats for that node rather than
/// treat it as a hard error, since this is diagnostic-only.
fn profile_canonicalize_stages(
    sys: &System,
    colors: &ColorTable,
    want_report: bool,
) -> Option<CanonStageTimes> {
    let t0 = Instant::now();
    let part = extract_graph_part(sys, colors);
    let extract = t0.elapsed();

    if part.vertices.is_empty() {
        // Mirrors `canonicalize_constraint_system`'s own empty-graph-part
        // fast path -- no bliss call, no automorphism group at all, and
        // Stage G still runs (once, on the trivial empty labelling).
        let (graph_term, labelling) = canonicalize_graph_part_seeded(&[], &BTreeSet::new());
        let (_, content) = canonicalize_system_content_seeded_profiled(sys, &labelling, graph_term);
        return Some(CanonStageTimes {
            extract,
            content,
            survivors: 1,
            ..Default::default()
        });
    }

    let t1 = Instant::now();
    let dimacs = graph_part_to_dimacs(&part).ok()?;
    let result = run_bliss(&dimacs).ok()?;
    let bliss = t1.elapsed();

    let t2 = Instant::now();
    let group = generate_group(&result.generators, part.vertices.len());
    let group_size = group.len();
    let group_closure = t2.elapsed();

    // `minimal_graph_part_labelings` closes the group again internally;
    // closure is negligible next to minimization, so `group_minimization`
    // simply absorbs that second closure.
    let t3 = Instant::now();
    let survivors = minimal_graph_part_labelings(&part, &result);
    let group_minimization = t3.elapsed();

    // Stage D (re-derive the SEEDED labelling for each survivor) + Stage G,
    // once per survivor. Stage D re-runs the same AC-canonicalization
    // machinery as minimization, so it gets its own timer.
    let mut stage_d_per_survivor = Duration::ZERO;
    let mut content = ContentStageTimes::default();
    for (i, (labeling, _)) in survivors.iter().enumerate() {
        let t4 = Instant::now();
        let ordered = canonical_vertex_order(&part, labeling);
        let edges = canonical_edges(&part, labeling);
        let (graph_term, labelling) = canonicalize_graph_part_seeded(&ordered, &edges);
        stage_d_per_survivor += t4.elapsed();
        let (_, c) = canonicalize_system_content_seeded_profiled(sys, &labelling, graph_term);
        // The counts describe `sys` itself, identical for every survivor:
        // take them once, and add only the durations after that.
        if i == 0 {
            content = c;
        } else {
            content.add_durations(&c);
        }
    }

    // Not part of canonicalization: how much finer colorings/encodings
    // would shrink the group (see `Refinements`).
    let t5 = Instant::now();
    let refinements = Refinements::new(&part);
    let count = |keep: &dyn Fn(&Permutation) -> bool| group.iter().filter(|g| keep(g)).count();
    let group_sizes = GroupSizes {
        current: group_size,
        skeleton: count(&|g| preserves(&refinements.skeleton, g)),
        local: count(&|g| preserves(&refinements.local, g)),
        local_occurrences: count(&|g| {
            preserves(&refinements.local, g) && refinements.preserves_occurrences(g)
        }),
        local_positions: count(&|g| {
            preserves(&refinements.local, g) && refinements.preserves_positions(g)
        }),
        survivors: survivors.len(),
    };
    assert!(
        group_sizes.local_positions >= group_sizes.survivors,
        "every content automorphism preserves local content and positional literal \
         incidence: {group_sizes:?}"
    );
    // The content automorphisms themselves: the elements whose relabelled
    // graph-part term equals the identity's. `positional_exact` = they are
    // exactly the elements preserving local content + positional incidence
    // (containment plus equal size), not merely as many.
    let graph_term = |lab: &Permutation| {
        canonicalize_graph_part(&canonical_vertex_order(&part, lab), &canonical_edges(&part, lab))
    };
    let base = graph_term(&result.canonical_labeling);
    let content_automorphisms: Vec<&Permutation> = group
        .iter()
        .filter(|g| graph_term(&result.canonical_labeling.compose(g)) == base)
        .collect();
    assert_eq!(content_automorphisms.len(), survivors.len());
    let positional_exact = group_sizes.local_positions == content_automorphisms.len()
        && content_automorphisms
            .iter()
            .all(|g| preserves(&refinements.local, g) && refinements.preserves_positions(g));
    let swaps = classify_generator_moves(&part, &result.generators, &refinements.local);
    let dummy_less_only_example = find_dummy_less_only_example(&part, &result.generators, &swaps);
    let report = (want_report && group_size > 1)
        .then(|| automorphism_report(sys, &part, &result.generators, &refinements, group_sizes));
    let analysis = t5.elapsed();

    Some(CanonStageTimes {
        extract,
        bliss,
        group_closure,
        group_minimization,
        stage_d_per_survivor,
        content,
        vertices: part.vertices.len(),
        edges: part.edges.len(),
        generators: result.generators.len(),
        group_size,
        survivors: survivors.len(),
        swaps,
        dummy_less_only_example,
        group_sizes,
        positional_exact,
        analysis,
        report,
    })
}

/// `PROFILE_CANON=1` totals, accumulated over every profiled occurrence.
#[derive(Default)]
struct CanonProfile {
    nodes: usize,
    /// Coarse wall time of each whole `profile_canonicalize_stages` call,
    /// measured from OUTSIDE it -- a cross-check against the sum of its own
    /// internal per-stage timers, to tell a genuine "first call vs second
    /// call" asymmetry (e.g. subprocess-spawn cost growing with the parent's
    /// memory footprint) apart from an accounting bug in this tool.
    wall: Duration,
    extract: Duration,
    bliss: Duration,
    group_closure: Duration,
    group_minimization: Duration,
    stage_d_per_survivor: Duration,
    /// Stage G, summed across nodes (durations AND counts).
    content: ContentStageTimes,
    max_eq_store_conj_alternatives: usize,
    total_vertices: usize,
    total_edges: usize,
    max_generators: usize,
    max_group_size: usize,
    max_survivors: usize,
    total_group_size: u64,
    // How much of the automorphism group `minimal_graph_part_labelings`
    // keeps as survivors: survivors ~= group_size means the group reflects
    // REAL alpha-equivalences (minimization work is inherent); survivors <<
    // group_size means bliss's coloring can't tell apart vertices whose
    // content differs, so a finer coloring could shrink the group itself.
    // Nontrivial groups (|Aut(G)| > 1) only: a trivial group's ratio is
    // vacuously 1/1 and would just dilute the average.
    nontrivial_nodes: usize,
    nontrivial_survivors: u64,
    nontrivial_group_size: u64,
    nontrivial_all_survive: usize,
    nontrivial_ratio_sum: f64,
    /// group_size -> (node count, total survivors), so the relationship
    /// between group size and survivor count can be read off directly
    /// rather than guessed from one aggregate ratio.
    group_size_histogram: BTreeMap<usize, (u64, u64)>,
    swap_kinds: SwapKindCounts,
    /// Group sizes under each candidate refinement, summed over
    /// nontrivial-group nodes (each sum = group elements minimization would
    /// canonicalize), and how many of those nodes each makes trivial.
    refined_sums: GroupSizes,
    refined_trivial: GroupSizes,
    /// Nontrivial-group nodes by [`GroupSizes::category`].
    categories: BTreeMap<&'static str, usize>,
    /// Nontrivial-group nodes where local content + positional incidence
    /// keeps exactly the content automorphisms.
    positional_exact: usize,
    /// Refinement-analysis time (see `CanonStageTimes::analysis`).
    analysis: Duration,
}

impl CanonProfile {
    fn record(&mut self, s: &CanonStageTimes) {
        self.nodes += 1;
        self.extract += s.extract;
        self.bliss += s.bliss;
        self.group_closure += s.group_closure;
        self.group_minimization += s.group_minimization;
        self.stage_d_per_survivor += s.stage_d_per_survivor;
        self.content += &s.content;
        self.max_eq_store_conj_alternatives = self
            .max_eq_store_conj_alternatives
            .max(s.content.num_eq_store_conj_alternatives);
        self.total_vertices += s.vertices;
        self.total_edges += s.edges;
        self.max_generators = self.max_generators.max(s.generators);
        self.max_group_size = self.max_group_size.max(s.group_size);
        self.max_survivors = self.max_survivors.max(s.survivors);
        self.total_group_size += s.group_size as u64;
        if s.group_size > 1 {
            self.nontrivial_nodes += 1;
            self.nontrivial_survivors += s.survivors as u64;
            self.nontrivial_group_size += s.group_size as u64;
            self.nontrivial_ratio_sum += s.survivors as f64 / s.group_size as f64;
            if s.survivors == s.group_size {
                self.nontrivial_all_survive += 1;
            }
            let entry = self.group_size_histogram.entry(s.group_size).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += s.survivors as u64;
            self.swap_kinds.merge(&s.swaps);
            let g = s.group_sizes;
            let (sums, trivial) = (&mut self.refined_sums, &mut self.refined_trivial);
            for (sum, triv, size) in [
                (&mut sums.current, &mut trivial.current, g.current),
                (&mut sums.skeleton, &mut trivial.skeleton, g.skeleton),
                (&mut sums.local, &mut trivial.local, g.local),
                (&mut sums.local_occurrences, &mut trivial.local_occurrences, g.local_occurrences),
                (&mut sums.local_positions, &mut trivial.local_positions, g.local_positions),
                (&mut sums.survivors, &mut trivial.survivors, g.survivors),
            ] {
                *sum += size;
                *triv += usize::from(size == 1);
            }
            *self.categories.entry(g.category()).or_insert(0) += 1;
            self.positional_exact += usize::from(s.positional_exact);
        }
        self.analysis += s.analysis;
    }

    /// `t_canonicalize` is `PROFILE=1`'s real canonicalization time, when
    /// that flag is also on.
    fn print(&self, t_canonicalize: Option<Duration>) {
        let c = &self.content;
        let total = self.extract
            + self.bliss
            + self.group_closure
            + self.group_minimization
            + self.stage_d_per_survivor
            + c.total_duration();
        let row = |label: &str, d: Duration, note: String| {
            println!(
                "  {label:<45}{:>8.2}s ({:>5.1}%){note}",
                d.as_secs_f64(),
                pct(d, total)
            );
        };
        println!(
            "\n--- PROFILE_CANON: canonicalize_constraint_system's OWN internal \
             stages, summed over {} occurrence(s) (re-derived a second time, discarded \
             -- see the module docs) ---",
            self.nodes
        );
        row("Stage A  extract_graph_part:", self.extract, String::new());
        row("Stage C  dimacs + external bliss subprocess:", self.bliss, String::new());
        row(
            "Stage F  automorphism GROUP CLOSURE:",
            self.group_closure,
            "  (generate_group)".into(),
        );
        row(
            "Stage F  automorphism group MINIMIZATION:",
            self.group_minimization,
            "  (re-canonicalize the graph part once per group element)".into(),
        );
        row(
            "Stage D  re-seed per survivor:",
            self.stage_d_per_survivor,
            "  (canonicalize_graph_part_seeded, once per SURVIVOR -- same \
             AC-canonicalization cost class as group MINIMIZATION above, just re-run seeded)"
                .into(),
        );
        row(
            "Stage G  formulas:",
            c.formulas,
            format!("  ({} formula(s) total)", c.num_formulas),
        );
        row(
            "Stage G  solved_formulas:",
            c.solved_formulas,
            format!("  ({} total)", c.num_solved_formulas),
        );
        row("Stage G  lemmas:", c.lemmas, format!("  ({} total)", c.num_lemmas));
        row(
            "Stage G  eq_store.subst:",
            c.eq_store_subst,
            format!("  ({} entries total)", c.num_eq_store_subst),
        );
        row(
            "Stage G  eq_store.conj:",
            c.eq_store_conj,
            format!(
                "  ({} EqDisj / {} alternatives total, max {} alternatives on one node)",
                c.num_eq_store_conj,
                c.num_eq_store_conj_alternatives,
                self.max_eq_store_conj_alternatives
            ),
        );
        row(
            "Stage G  subterm_store:",
            c.subterm_store,
            format!(
                "  ({} subterm/solved-subterm/neg-subterm entries total)",
                c.num_subterms + c.num_solved_subterms + c.num_neg_subterms
            ),
        );
        row(
            "Stage G  goals:",
            c.goals,
            format!("  ({} goal(s) total, before dropping Split)", c.num_goals),
        );
        println!(
            "  sum of measured stages: {:.2}s (Stage G is run ONCE PER SURVIVOR, already \
             summed accordingly -- see CanonStageTimes::content's own doc comment)",
            total.as_secs_f64()
        );
        println!(
            "  coarse wall time of the whole profile_canonicalize_stages call: {:.2}s \
             (cross-check against the {:.2}s sum above -- a big gap between THESE two means \
             an accounting bug in this tool, not a real pipeline stage; excludes {:.2}s of \
             refinement analysis)",
            self.wall.saturating_sub(self.analysis).as_secs_f64(),
            total.as_secs_f64(),
            self.analysis.as_secs_f64()
        );
        if let Some(t_canonicalize) = t_canonicalize {
            println!(
                "  (vs. real canonicalize_constraint_system time t_canonicalize={:.2}s from \
                 PROFILE above -- a gap between THIS and the coarse wall time just above is \
                 the interesting one: same pipeline, same node, run twice back to back, \
                 should cost about the same)",
                t_canonicalize.as_secs_f64()
            );
        }
        if self.nodes == 0 {
            return;
        }
        println!(
            "  graph size: {:.1} vertices/node, {:.1} edges/node (avg)",
            self.total_vertices as f64 / self.nodes as f64,
            self.total_edges as f64 / self.nodes as f64
        );
        println!(
            "  automorphism group: max |generators|={}, max |Aut(G)|={}, avg |Aut(G)|={:.1}, \
             max |survivors|={}",
            self.max_generators,
            self.max_group_size,
            self.total_group_size as f64 / self.nodes as f64,
            self.max_survivors
        );
        if self.nontrivial_nodes == 0 {
            return;
        }
        let n = self.nontrivial_nodes;
        println!(
            "  survivors/|Aut(G)| (nodes with a NONTRIVIAL group only, n={n}): \
             avg ratio={:.3} (unweighted per-node average), \
             pooled ratio={:.3} (total survivors / total |Aut(G)|), \
             {}/{n} nodes ({:.1}%) have EVERY \
             automorphism survive minimization (survivors == |Aut(G)|)",
            self.nontrivial_ratio_sum / n as f64,
            self.nontrivial_survivors as f64 / self.nontrivial_group_size as f64,
            self.nontrivial_all_survive,
            100.0 * self.nontrivial_all_survive as f64 / n as f64
        );
        println!(
            "  a ratio near 1.0 means bliss's coloring already finds close to the REAL \
             alpha-equivalence classes (minimization work is inherent, not wasted); a \
             ratio well below 1.0 means many reported automorphisms are SPURIOUS -- \
             indistinguishable by color/structure alone but not actually content-equal \
             -- and a finer `ColorTable` could shrink |Aut(G)| itself (and so both \
             Stage F and Stage D's cost) without changing the canonical RESULT"
        );
        println!("  |Aut(G)| -> (node count, avg survivors):");
        for (group_size, (count, total_survivors)) in &self.group_size_histogram {
            println!(
                "    {group_size:>3} -> {count:>5} node(s), avg survivors={:.2} \
                 (ratio={:.3})",
                *total_survivors as f64 / *count as f64,
                *total_survivors as f64 / (*count as f64 * *group_size as f64)
            );
        }
        println!(
            "\n  WHAT the automorphism generators actually move, by VertexKind \
             (summed over every generator on every nontrivial-group node -- a \
             transposition contributes 2 moved slots, a k-cycle k; \"same\" means the \
             moved vertex's content equals its image's up to renaming + AC, \"different\" \
             that it doesn't, i.e. a content-aware color would forbid the move):"
        );
        let kinds = &self.swap_kinds;
        for label in kinds.labels() {
            let same = kinds.same_local.get(label).copied().unwrap_or(0);
            let different = kinds.different_local.get(label).copied().unwrap_or(0);
            println!(
                "    {label:<20} moved={:>7}  same={same:>7}  different={different:>7}",
                same + different
            );
        }
        let (sums, trivial) = (&self.refined_sums, &self.refined_trivial);
        println!(
            "\n  CANDIDATE REFINEMENTS: summed |Aut(G)| over the {n} nontrivial-group nodes \
             (= group elements MINIMIZATION canonicalizes), relative size, and nodes left \
             with a trivial group:"
        );
        for (label, sum, triv) in [
            ("current coloring", sums.current, trivial.current),
            ("+ content skeleton (symbols + sorts)", sums.skeleton, trivial.skeleton),
            ("+ local content (canonical form)", sums.local, trivial.local),
            ("+ local content + literal incidence", sums.local_occurrences, trivial.local_occurrences),
            ("+ local content + positional incidence", sums.local_positions, trivial.local_positions),
            ("content automorphisms (floor)", sums.survivors, trivial.survivors),
        ] {
            println!(
                "    {label:<40} {sum:>8}  ({:>5.3})  trivial on {triv:>5}/{n}",
                sum as f64 / sums.current as f64
            );
        }
        println!(
            "  local content + positional incidence keeps EXACTLY the content automorphisms \
             (same set) on {}/{n} nodes",
            self.positional_exact
        );
        println!(
            "  nodes by what would make |Aut(G)| equal the content automorphisms: {}",
            self.categories
                .iter()
                .map(|(c, k)| format!("{c}={k}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
}

/// `PROFILE=1` stage timers.
#[derive(Default)]
struct StageTimers {
    canonicalize: Duration,
    fingerprint: Duration,
    candidate_methods: Duration,
    exec_proof_method: Duration,
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
    profile_canon: bool,
    dump_dummy_swaps: bool,
    /// `DUMP_AUTOMORPHISMS=N`: examples to print per [`GroupSizes::category`].
    dump_automorphisms: usize,
}

struct Explorer<'a> {
    ctx: &'a ProofContext,
    flags: Flags,
    max_depth: usize,
    no_merge: bool,
    graph: Graph,
    by_fingerprint: FastMap<CanonicalSystemFingerprint, OrId>,
    /// Nodes awaiting expansion, with the `System` only they still carry and
    /// its candidate methods if already computed (for the method check).
    queue: VecDeque<(OrId, System, Option<Vec<ProofMethod>>)>,
    processed: u64,
    canonicalize_failures: usize,
    method_check: MethodCheck,
    timers: StageTimers,
    canon_profile: CanonProfile,
    dummy_swap_examples_shown: usize,
    /// `DUMP_AUTOMORPHISMS` examples printed so far, per category.
    automorphism_examples_shown: BTreeMap<&'static str, usize>,
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

const DUMMY_SWAP_EXAMPLES_LIMIT: usize = 3;

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
    fn visit(&mut self, sys: System, depth: usize, edge: Option<(OrId, u32, u32)>) -> OrId {
        let ctx = self.ctx;
        self.processed += 1;
        if self.graph.processed_by_depth.len() <= depth {
            self.graph.processed_by_depth.resize(depth + 1, 0);
        }
        self.graph.processed_by_depth[depth] += 1;

        self.publish_progress();
        if self.flags.progress && self.processed % 20 == 0 {
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
        set_activity_step("canonicalize");
        let canon = timed(self.flags.profile, &mut self.timers.canonicalize, || {
            catch_unwind(AssertUnwindSafe(|| {
                canonicalize_constraint_system_with_labelling(&sys, &ctx.color_table)
            }))
        });

        if self.flags.profile_canon {
            self.profile_canon(&sys, edge);
        }

        let (fingerprint, methods, methods_sig) = match canon {
            Err(payload) => {
                let message = panic_message(&*payload).to_string();
                log!(
                    "canonicalize_constraint_system PANICKED at {}: {message}",
                    self.graph.path(edge)
                );
                let id = self.graph.add_node(depth, edge);
                self.graph.node_mut(id).status = Status::CanonPanic;
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
            Ok(Ok((canon, labelling))) => {
                let fp = timed(self.flags.profile, &mut self.timers.fingerprint, || {
                    fingerprint_constraint_system(&canon)
                });
                set_activity_step("candidate_methods");
                let methods = timed(self.flags.profile, &mut self.timers.candidate_methods, || {
                    candidate_methods(&sys, ctx, depth)
                });
                let sig = self.method_signature(&sys, &labelling, &methods, edge);
                (Some(fp), Some(methods), sig)
            }
        };

        if let (Some(fp), false) = (fingerprint, self.no_merge) {
            if let Some(&existing) = self.by_fingerprint.get(&fp) {
                if let Some(sig) = &methods_sig {
                    self.check_methods(existing, sig, depth, edge);
                }
                return existing;
            }
        }
        let id = self.graph.add_node(depth, edge);
        let node = self.graph.node_mut(id);
        node.canon_err = fingerprint.is_none();
        node.methods_sig = methods_sig;
        if let (Some(fp), false) = (fingerprint, self.no_merge) {
            self.by_fingerprint.insert(fp, id);
        }
        self.queue.push_back((id, sys, methods));
        id
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
            ids.iter().map(|&i| self.graph.methods[i as usize].as_str()).collect()
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
        while let Some((id, sys, methods)) = self.queue.pop_front() {
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
                self.queue.push_front((id, sys, methods));
                return stop;
            }
            let expanded = catch_unwind(AssertUnwindSafe(|| self.expand(id, &sys, methods)));
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
    /// them (every node except `canon_err` ones).
    fn expand(&mut self, id: OrId, sys: &System, methods: Option<Vec<ProofMethod>>) {
        let ctx = self.ctx;
        let profile = self.flags.profile;
        let depth = self.graph.node(id).depth as usize;
        if depth >= self.max_depth {
            self.graph.node_mut(id).status = Status::DepthLimit;
            return;
        }
        let path = self.graph.path(self.graph.node(id).first_parent);
        set_breadcrumb_node(id, depth as u32, path);
        let methods = methods.unwrap_or_else(|| {
            set_activity("candidate_methods".to_string());
            timed(profile, &mut self.timers.candidate_methods, || {
                candidate_methods(sys, ctx, depth)
            })
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
        for (k, method) in methods.into_iter().enumerate() {
            let text = pretty_proof_method_inline(&method);
            let step = format!("method {}/{method_count} {text}", k + 1);
            if self.flags.trace {
                log!("  exec {}", truncate_for_log(&step, 300));
            }
            set_activity(format!("exec_proof_method {step}"));
            let exec = timed(profile, &mut self.timers.exec_proof_method, || {
                exec_proof_method(ctx, &method, sys)
            });
            let Some(cases) = exec else {
                self.graph.node_mut(id).inapplicable += 1;
                continue;
            };
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
            for (case_name, child_sys) in cases {
                set_activity(format!("visit case [{case_name}] of {step}"));
                // Pushed with a placeholder target first, so the child's own
                // path (diagnostics, `first_parent`) is already resolvable
                // while it is being canonicalized.
                let slots = &mut self.graph.node_mut(id).and[and_idx as usize].cases;
                let case_idx = slots.len() as u32;
                slots.push((case_name, OrId::MAX));
                let child = self.visit(child_sys, depth + 1, Some((id, and_idx, case_idx)));
                self.graph.node_mut(id).and[and_idx as usize].cases[case_idx as usize].1 = child;
            }
        }
    }

    /// `PROFILE_CANON=1` (and `DUMP_DUMMY_SWAPS=1`) for one occurrence.
    fn profile_canon(&mut self, sys: &System, edge: Option<(OrId, u32, u32)>) {
        let start = Instant::now();
        let want_report = self.flags.dump_automorphisms > 0;
        let profiled = catch_unwind(AssertUnwindSafe(|| {
            profile_canonicalize_stages(sys, &self.ctx.color_table, want_report)
        }))
        .ok()
        .flatten();
        self.canon_profile.wall += start.elapsed();
        let Some(stages) = profiled else {
            return;
        };
        self.canon_profile.record(&stages);
        if let Some(report) = &stages.report {
            let category = stages.group_sizes.category();
            let shown = self.automorphism_examples_shown.entry(category).or_insert(0);
            if *shown < self.flags.dump_automorphisms {
                *shown += 1;
                eprintln!(
                    "\n[DUMP_AUTOMORPHISMS {category} #{shown}] reached via:\n  {}\n  {report}",
                    self.graph.path(edge)
                );
            }
        }
        let Some(example) = &stages.dummy_less_only_example else {
            return;
        };
        if !self.flags.dump_dummy_swaps || self.dummy_swap_examples_shown >= DUMMY_SWAP_EXAMPLES_LIMIT
        {
            return;
        }
        self.dummy_swap_examples_shown += 1;
        eprintln!(
            "\n[DUMP_DUMMY_SWAPS #{}] node with a Dummy/LessRelation-only automorphism \
             (|Aut(G)|={}, swaps {example}), reached via:\n  {}",
            self.dummy_swap_examples_shown,
            stages.group_size,
            self.graph.path(edge)
        );
        eprintln!(
            "  sys.nodes (rule-instance timepoints): {:?}",
            sys.nodes.iter().map(|(n, _)| n.to_string()).collect::<Vec<_>>()
        );
        eprintln!("  sys.less_atoms (i < j constraints):");
        for la in sys.less_atoms_in_set_order() {
            eprintln!("    {} < {}  ({:?})", la.smaller, la.larger, la.reason);
        }
    }
}

fn dump_formulas(sys: &System, path: &str) {
    eprintln!("--- formulas/solved_formulas at {path} ---");
    for f in sys.formulas.iter() {
        eprintln!("  formula: {}", tamarin_theory::pretty_formula::pretty_guarded(f));
    }
    for f in sys.solved_formulas.iter() {
        eprintln!("  solved_formula: {}", tamarin_theory::pretty_formula::pretty_guarded(f));
    }
    eprintln!(
        "  nodes: {:?}",
        sys.nodes.iter().map(|(n, _)| n.to_string()).collect::<Vec<_>>()
    );
    eprintln!("  last_atom: {:?}", sys.last_atom);
    eprintln!("  goals ({} total):", sys.goals.len());
    for (goal, status) in sys.goals.iter() {
        eprintln!("    {goal:?} solved={} looping={}", status.solved, status.looping);
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
        "canon_panic",
        "exec_panic",
        "canon_err",
    ]
    .into_iter()
    .map(|s| (s, 0))
    .collect();
    for n in &graph.nodes {
        graph_by_min_depth[n.depth as usize] += 1;
        *status_counts.get_mut(n.status.name()).expect("every status is listed") += 1;
        if n.canon_err {
            *status_counts.get_mut("canon_err").expect("listed") += 1;
        }
    }
    let lower_bound = status_counts["unexpanded"] > 0
        || status_counts["canon_panic"] > 0
        || status_counts["exec_panic"] > 0;

    let mut violations = Vec::new();
    if !(graph_size <= processed && u128::from(processed) <= tree) {
        violations.push(format!("graph={graph_size} processed={processed} tree={tree}"));
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

fn node_json(id: usize, n: &OrNode) -> Value {
    let mut v = json!({
        "id": id,
        "depth": n.depth,
        "status": n.status.name(),
        "canon_err": n.canon_err,
        "first_parent": n.first_parent.map(|(p, a, c)| json!([p, a, c])),
        "inapplicable": n.inapplicable,
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
    /// `--time-budget SECS`: stop expanding once the process has run this
    /// long (setup included), then write the JSON as usual.
    time_budget: Option<f64>,
    /// `--max-rss-gb GB`: stop expanding once the resident set reaches this
    /// many GiB, then write the JSON as usual.
    max_rss_gb: Option<f64>,
    /// `--heartbeat SECS`: log a progress line every SECS (background thread).
    heartbeat: Option<f64>,
    /// `--trace`: log every expansion and every applied method.
    trace: bool,
}

enum Mode {
    /// `--list-lemmas <theory>`.
    ListLemmas(String),
    Explore(Args),
}

const USAGE: &str = "usage: explore_canonical_matches <theory.spthy> <lemma> \
     [--max-depth N] [--max-nodes N] [--out PATH] [--no-merge] \
     [--time-budget SECS] [--max-rss-gb GB] [--heartbeat SECS] [--trace]\n       \
     explore_canonical_matches --list-lemmas <theory.spthy>";

fn usage_error(message: &str) -> ! {
    eprintln!("{message}\n{USAGE}");
    std::process::exit(EXIT_USAGE);
}

fn parse_args(raw: &[String]) -> Mode {
    if raw.get(1).map(String::as_str) == Some("--list-lemmas") {
        match raw.get(2) {
            Some(path) if raw.len() == 3 => return Mode::ListLemmas(path.clone()),
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
        time_budget: None,
        max_rss_gb: None,
        heartbeat: None,
        trace: false,
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
            "--heartbeat" => args.heartbeat = Some(number(value())),
            "--no-merge" => args.no_merge = true,
            "--trace" => args.trace = true,
            other => usage_error(&format!("unrecognized argument: {other}")),
        }
        i += if matches!(flag, "--no-merge" | "--trace") { 1 } else { 2 };
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

/// `--list-lemmas`: prints one JSON object describing the theory and its
/// lemmas, for a batch runner to plan jobs from. `error` is non-null (and
/// the exit code [`EXIT_SETUP_ERROR`]) if it can't be read, parsed or
/// elaborated; the fields gathered before that are still printed.
fn list_lemmas(theory_path: &str) -> i32 {
    let mut out = json!({ "theory": theory_path, "error": null });
    let code = match describe_theory(theory_path, &mut out) {
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
/// `build_lemma_proof_context` can look up.
fn describe_theory(path: &str, out: &mut Value) -> Result<(), String> {
    use tamarin_parser::ast::TheoryItem;
    use tamarin_theory::theory::TraceQuantifier;

    let source = std::fs::read_to_string(path).map_err(|e| format!("read: {e}"))?;
    out["lines"] = json!(source.lines().count());
    let plain = tamarin_parser::parse_theory(&source, &[]);
    let with_diff_flag = tamarin_parser::parse_theory(&source, &["diff"]);
    let diff_items = with_diff_flag.as_ref().map_or(0, |t| {
        t.items
            .iter()
            .filter(|i| {
                matches!(
                    i,
                    TheoryItem::DiffLemma(_) | TheoryItem::EquivLemma(..) | TheoryItem::DiffEquivLemma(_)
                )
            })
            .count()
    });
    out["diff"] = json!(diff_items > 0 || (plain.is_err() && with_diff_flag.is_ok()));
    let parsed = plain.map_err(|e| format!("parse: {e}"))?;
    out["processes"] = json!(parsed
        .items
        .iter()
        .any(|i| matches!(i, TheoryItem::TopLevelProcess(_) | TheoryItem::ProcessDef(_))));
    let elaborated = catch_unwind(AssertUnwindSafe(|| tamarin_theory::elaborate::elaborate(&parsed)))
        .map_err(|p| format!("elaborate panicked: {}", panic_message(&*p)))?
        .map_err(|e| format!("elaborate: {}", e.message))?;
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
                "modulo": l.modulo,
            })
        })
        .collect();
    Ok(())
}

/// Parses and elaborates the theory, starts maude on its signature, and
/// builds the lemma's proof context -- the same per-lemma setup
/// `prove_lemma` uses. Errors instead of panicking, so a batch log says
/// which step failed.
fn setup(
    args: &Args,
) -> Result<(ProofContext, System, tamarin_theory::elaborate::UserFunsForTheoryGuard), String> {
    let (parsed, elaborated, maude) = common::try_load_theory_with_maude(&args.theory_path)?;
    log!("parsed and elaborated: {} rule(s); maude started", elaborated.rules().count());
    let (ctx, initial_sys, _skeleton_tree, user_funs_guard) = build_lemma_proof_context(
        &parsed,
        &args.lemma,
        maude,
        None,
        "",
        &CliHeuristic::default(),
        CutStrategy::Dfs,
        None,
    )
    .map_err(|e| format!("build_lemma_proof_context({}): {e:?}", args.lemma))?;
    Ok((ctx, initial_sys, user_funs_guard))
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
        Mode::ListLemmas(path) => std::process::exit(list_lemmas(&path)),
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
    if !bliss_available() {
        eprintln!("bliss not available and TAM_ALLOW_NO_BLISS=1 set -- nothing to do, exiting");
        return;
    }

    // `--heartbeat 0` (or less) means off, not a busy loop.
    if let Some(every) = args.heartbeat.filter(|&secs| secs > 0.0) {
        spawn_heartbeat(Duration::from_secs_f64(every));
    }
    set_phase("setup");
    // `_user_funs_guard` must stay alive for the WHOLE exploration:
    // canonicalization needs the installed signature.
    let (ctx, initial_sys, _user_funs_guard) = match catch_unwind(AssertUnwindSafe(|| setup(&args))) {
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
            profile_canon: env_gate!("PROFILE_CANON"),
            dump_dummy_swaps: env_gate!("DUMP_DUMMY_SWAPS"),
            dump_automorphisms: std::env::var("DUMP_AUTOMORPHISMS")
                .map_or(0, |v| v.parse().unwrap_or(3)),
        },
        max_depth: args.max_depth,
        no_merge: args.no_merge,
        graph: Graph::default(),
        by_fingerprint: FastMap::default(),
        queue: VecDeque::new(),
        processed: 0,
        canonicalize_failures: 0,
        method_check: MethodCheck::default(),
        timers: StageTimers::default(),
        canon_profile: CanonProfile::default(),
        dummy_swap_examples_shown: 0,
        automorphism_examples_shown: BTreeMap::new(),
        started: Instant::now(),
        time_budget: args.time_budget.map(Duration::from_secs_f64),
        max_rss_mib: args.max_rss_gb.map(|gb| (gb * 1024.0) as u64),
        canon_failure_examples: Vec::new(),
        panic: None,
    };
    // The root's own visit runs outside `run`'s per-expansion panic capture;
    // with no node to attach a partial result to, a panic there only logs.
    match catch_unwind(AssertUnwindSafe(|| explorer.visit(initial_sys, 0, None))) {
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
        "schema_version": 3,
        "theory": args.theory_path,
        "lemma": args.lemma,
        "argv": argv,
        "params": {
            "max_depth": args.max_depth,
            "max_nodes": args.max_nodes,
            "no_merge": args.no_merge,
            "time_budget": args.time_budget,
            "max_rss_gb": args.max_rss_gb,
        },
        "stop_reason": stop.name(),
        "truncated": stop != StopReason::Exhausted,
        "lower_bound": summary.lower_bound,
        "timing": {
            "setup_secs": setup_secs,
            "explore_secs": explore_secs,
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
        "panic": explorer.panic,
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
        if args.no_merge { ", no_merge" } else { "" }
    );

    if explorer.flags.profile {
        let t = &explorer.timers;
        let total = t.canonicalize + t.fingerprint + t.candidate_methods + t.exec_proof_method;
        println!(
            "\n--- PROFILE: wall time by pipeline stage, summed over {} occurrence(s) ---",
            summary.processed
        );
        for (label, d) in [
            ("canonicalize_constraint_system (bliss+graph):", t.canonicalize),
            ("fingerprint + dedup lookup:", t.fingerprint),
            ("candidate_methods (ranking, incl. is_finished):", t.candidate_methods),
            ("exec_proof_method (solving -- maude/AC here):", t.exec_proof_method),
        ] {
            println!("  {label:<47}{:>8.2}s ({:>5.1}%)", d.as_secs_f64(), pct(d, total));
        }
        println!(
            "  sum of measured stages: {:.2}s (wall clock may exceed this by parse/elaborate/setup)",
            total.as_secs_f64()
        );
    }

    if explorer.flags.profile_canon {
        explorer
            .canon_profile
            .print(explorer.flags.profile.then_some(explorer.timers.canonicalize));
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
