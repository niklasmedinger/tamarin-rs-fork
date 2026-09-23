//! Empirical test: does canonicalization/fingerprinting (Stage G,
//! `canon`/`canon_fingerprint`) actually detect $\alphaeqac$-equal
//! constraint systems reached via DIFFERENT paths through a real proof
//! search?
//!
//! Tamarin's real prover is a GREEDY search: at each node,
//! `candidate_methods` ranks every applicable `ProofMethod` by heuristic
//! and only the first one whose `exec_proof_method` succeeds is taken
//! (`search::expand_inner`/`run_proof_search`). That walks exactly ONE
//! path through the proof tree, so it can never observe two DIFFERENT
//! paths reaching the "same" (canonically) system -- there is only ever
//! one path to begin with.
//!
//! This binary instead explores MANUALLY and EXHAUSTIVELY: at every node,
//! it takes ALL of `candidate_methods`' candidates (not just the
//! heuristic's pick), executes each one, and recurses into every
//! resulting child system -- turning the single greedy path into the full
//! proof TREE the eventual tree-search-over-proof-methods approach would
//! explore. At every node reached, the system is canonicalized
//! (`canon::canonicalize_constraint_system`) and fingerprinted
//! (`canon_fingerprint::fingerprint_constraint_system`), and compared
//! against every OTHER node's fingerprint seen so far
//! (`compare_fingerprints`). A system is identified by the PATH taken to
//! reach it -- the sequence of (proof method, case name) pairs from the
//! root -- so a reported match names both paths that reach the same
//! canonical system.
//!
//! This is explicitly a first, cheap proof-of-concept: a full proof tree
//! (every applicable method, not just the ranked-best one) is exponential
//! in general, so `--max-depth`/`--max-nodes` bound the exploration.
//!
//! ## Duplicates are not re-expanded (tree-search-WITH-merging)
//!
//! A node whose canonical system exactly matches an already-KEPT node
//! (see `nodes_kept`/`nodes_deduplicated` below) is not expanded further
//! -- its children are never computed or enqueued. This is sound, not
//! just a heuristic pruning shortcut: it never loses a reachable system.
//! The argument has two parts.
//!
//! 1. **Two canonically-equal systems have the same candidate children,
//!    up to renaming.** `exec_proof_method`/`candidate_methods` are
//!    fundamentally functions of a system's LOGICAL content, not of its
//!    incidental representation (`NodeId` numbering, variable display
//!    names) -- this has to hold, or Tamarin's own proof search would be
//!    unsound (two alpha-equivalent constraint systems represent the same
//!    underlying trace set, so the solver MUST treat them identically).
//!    The one parameter that looks path-dependent, `depth` (passed to
//!    `candidate_methods(sys, ctx, depth)`), only selects which
//!    ROUND-ROBIN heuristic RANKING orders the SAME full goal list
//!    (`rank_goals_with`'s `rankings[depth % n]`, mirroring HS's
//!    `useHeuristic`) -- it reorders candidates, it never filters which
//!    ones exist. Since this explorer takes EVERY candidate rather than
//!    just the first-ranked one, that reordering has no effect on which
//!    children get discovered.
//! 2. **The kept representative is always found at the MINIMUM depth of
//!    its canonical class.** The queue is FIFO (`VecDeque::push_back`/
//!    `pop_front`) and every child's depth is its parent's depth + 1, so
//!    BFS visits nodes in non-decreasing depth order. The first time a
//!    canonical class is seen is therefore its shallowest occurrence --
//!    every later rediscovery of the "same" system is found at depth >=
//!    the kept representative's depth. So the kept representative's own
//!    (already-scheduled) expansion has AT LEAST as much remaining
//!    `--max-depth` budget to explore that class's descendants as a
//!    later duplicate would have had. Skipping the duplicate's expansion
//!    therefore cannot lose any depth-bounded descendant that expanding
//!    it separately would have found.
//!
//! `nodes_kept + nodes_deduplicated == nodes_discovered` still holds (see
//! below) -- a deduplicated node is still COUNTED and still compared
//! against every prior node for match reporting, it just isn't expanded.
//! Turning the tree into an actual graph (reifying the merge -- e.g.
//! recording which nodes a kept representative stands in for, or
//! reporting the resulting node-count savings as its own metric) is still
//! future work; this binary prunes on the merge but doesn't reify it.
//!
//! Usage: `cargo run --example explore_canonical_matches -- <theory.spthy> <lemma> [--max-depth N] [--max-nodes N] [--close-threshold N]`
//!
//! - `--max-depth` (default 4): stop expanding a path once it reaches
//!   this many proof-method applications.
//! - `--max-nodes` (default 2000): stop the whole exploration once this
//!   many nodes have been visited (BFS order, so this is a breadth-first
//!   prefix of the bounded-depth tree).
//! - `--close-threshold` (default 1): also report "close" matches, where
//!   at most this many of the twelve tracked `CanonicalSystem` fields
//!   differ (0 = exact matches only).
//!
//! Env vars (all opt-in, unset = off):
//! - `QUIET=1` -- suppress the per-match `EXACT MATCH`/`CLOSE MATCH`
//!   printing (and the `format_path` cost of building it). The summary
//!   line and `DEPTH_STATS` output (if requested) still print; counting
//!   is unaffected. Useful for an AC-heavy theory (e.g. one with
//!   `builtins: diffie-hellman`): `PathStep::method` holds the FULL
//!   pretty-printed goal term, which can run to several KB per step for
//!   deeply nested DH/hash terms, and printing two whole paths per match
//!   -- times potentially thousands of matches -- makes a live terminal
//!   crawl and wastes disk when redirected to a file. **Does NOT fix the
//!   underlying slowness**, though: measured directly (`PROGRESS=1`)
//!   against `wireguard.spthy`, `QUIET=1` alone still only reaches
//!   ~15-20 nodes/s, because `candidate_methods`' branching factor for a
//!   DH-heavy theory is itself enormous (the BFS queue can pass 20,000
//!   pending nodes while still under 1,300 processed, stuck around depth
//!   3-4) -- see `PROGRESS=1` below to watch this directly.
//! - `DEPTH_STATS=1` -- print the per-depth node/dedup/finished-leaf
//!   breakdown at the end (see the printing code's own comments for the
//!   exact shape).
//! - `PROGRESS=1` -- print a stderr heartbeat every 20 nodes (elapsed
//!   time, nodes/sec, queue length). A long-running exploration otherwise
//!   prints nothing until it finishes or hits `--max-nodes`, so there's
//!   no way from the outside to tell "still working" from "hung".
//! - `DUMP_FORMULAS=1` -- dump `sys.formulas`/`sys.solved_formulas`/
//!   `sys.nodes`/`sys.last_atom` at every node visited, to stderr.
//! - `PROFILE=1` -- accumulate and print, at the end, wall time spent in
//!   each pipeline stage: `canonicalize_constraint_system` (bliss +
//!   graph extraction), `fingerprint_constraint_system` + the
//!   `compare_fingerprints` loop, `is_finished`/`candidate_methods`
//!   (ranking), and `exec_proof_method` (actually solving each
//!   candidate -- the one most likely to route through `MaudeHandle`).
//!   Use alongside `QUIET=1` for clean numbers (otherwise printing time
//!   leaks into the fingerprint/compare stage).
//! - `PROFILE_CANON=1` -- breaks the `canonicalize_constraint_system`
//!   bucket above down FURTHER into its own internal stages: Stage A
//!   graph extraction, Stage C's external `bliss` subprocess call (dimacs
//!   + spawn + parse), Stage F's automorphism GROUP CLOSURE
//!   (`generate_group`) and group MINIMIZATION (re-canonicalizing the
//!   graph part once per group element to find the minimum-term
//!   survivor(s)), and Stage G (formulas/solved_formulas/lemmas/
//!   eq_store.subst/eq_store.conj/subterm_store, each timed and counted
//!   separately via `canon::canonicalize_system_content_seeded_profiled`
//!   -- exposed as `pub` from `canon.rs` specifically for this). **Stage
//!   G runs ONCE PER SURVIVOR**, exactly like
//!   `canonicalize_constraint_system` itself (`for (labeling, ...) in
//!   &survivors { ... }`) -- a node with a large tied-survivor count
//!   pays Stage G's full cost that many times over, which is usually the
//!   real explanation for an outsized Stage G total, not any single
//!   formula/equation being expensive to canonicalize on its own. Also
//!   tracks vertex/edge counts, `|generators|`/`|Aut(G)|`/`|survivors|`,
//!   and formula/lemma/`eq_store`/`subterm_store` sizes per node, to
//!   correlate slow stages with what's actually driving them. Re-does
//!   the ENTIRE pipeline a SECOND time to get these sub-timings (the
//!   real `canonicalize_constraint_system` call still runs once,
//!   normally, for the actual result used for matching) -- expect
//!   roughly double the wall time under this flag; it is a diagnostic
//!   multiplier, not a regression. Also classifies, per nontrivial-group
//!   node, WHICH `VertexKind` the generators actually move (and whether
//!   the moved vertex's content is a byte-identical duplicate of what
//!   it's swapped with) -- printed as a summary table at the end.
//! - `DUMP_DUMMY_SWAPS=1` (requires `PROFILE_CANON=1` too) -- prints the
//!   PATH to the first 3 nodes whose automorphism generators move ONLY
//!   `Dummy`/`LessRelation` vertices (a `Dummy(nid)` with no
//!   `RuleInstance`/`Action` of its own, connected only via an `i < j`
//!   ordering constraint, freely permuted against another such node),
//!   plus that node's `sys.nodes`/`sys.less_atoms`, to stderr -- so one
//!   can be reproduced and inspected directly in the GUI.
//!
//! Requires `bliss` on `PATH` (or `$BLISS_PATH`) -- like every other
//! canonicalization test/example, skips (via `bliss_available()`'s own
//! panic-unless-opted-out gate) if `TAM_ALLOW_NO_BLISS=1` is set.
//!
//! ## `nodes_discovered` vs. `nodes_kept`/`nodes_deduplicated`
//!
//! Every successfully-canonicalized node is kept in `visited` regardless
//! of whether it exactly matches an already-visited node (a later node
//! still needs to compare against it). So `nodes_discovered` counts every
//! node reached, NOT the number of distinct canonical systems found --
//! `nodes_kept` (first-seen representative of its canonical class) is the
//! distinct count; `nodes_deduplicated` (exactly matches an
//! already-kept representative) is everything else.
//! `nodes_kept + nodes_deduplicated == nodes_discovered` always holds
//! (asserted at the end of `main`). This is a DIFFERENT count from
//! `exact_matches`: that counts matching PAIRS printed above (so a class
//! of 3 mutually-matching nodes reports 3 pairs but only contributes 2 to
//! `nodes_deduplicated`, since the first of the 3 is the kept
//! representative, not a duplicate of anything).

// Example/dev tool: prints results to stdout by design; allow the
// `disallowed_macros` convention freeze for this example binary.
#![allow(clippy::disallowed_macros)]

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use tamarin_theory::bliss_proc::{
    bliss_available, canonical_edges, canonical_vertex_order, generate_group, graph_part_to_dimacs,
    run_bliss,
};
use tamarin_theory::canon::{canonicalize_constraint_system, canonicalize_graph_part};
use tamarin_theory::canon_fingerprint::{
    compare_fingerprints, fingerprint_constraint_system, CanonicalSystemFingerprint,
};
use tamarin_theory::canon_graph::{extract_graph_part, VertexKind};
use tamarin_theory::constraint::solver::context::CutStrategy;
use tamarin_theory::constraint::solver::proof_method::{exec_proof_method, is_finished};
use tamarin_theory::constraint::solver::search::candidate_methods;
use tamarin_theory::constraint::system::System;
use tamarin_theory::pretty_theory::pretty_proof_method_inline;
use tamarin_theory::prove::{build_lemma_proof_context, CliHeuristic};

mod common;

/// One step of a path: which method was applied, and which of its
/// (possibly several) resulting cases was taken.
#[derive(Debug, Clone)]
struct PathStep {
    method: String,
    case: String,
}

fn format_path(path: &[PathStep]) -> String {
    if path.is_empty() {
        return "<root>".to_string();
    }
    path.iter()
        .map(|s| format!("{} [{}]", s.method, s.case))
        .collect::<Vec<_>>()
        .join(" -> ")
}

/// A node whose system has already been canonicalized and fingerprinted,
/// kept around ONLY as (path, fingerprint) -- not the original
/// `CanonicalSystem`/`System` -- since fingerprints are what
/// `compare_fingerprints` needs and are far cheaper to hold onto across a
/// whole exploration than the full systems. Keeping the originals too
/// (for interactive debugging of a reported match) is exactly the
/// follow-up scoped out of this first pass -- see the module docs.
struct Visited {
    path: Vec<PathStep>,
    fingerprint: CanonicalSystemFingerprint,
}

/// `PROFILE_CANON=1`'s per-node breakdown of `canonicalize_constraint_system`'s
/// OWN internal stages -- see [`profile_canonicalize_stages`]. `content`
/// is the SUM of [`tamarin_theory::canon::ContentStageTimes`] over every
/// survivor (see that struct's own doc comment on why Stage G runs once
/// PER SURVIVOR in production, not once per node -- the real reason a
/// large `survivors` count multiplies Stage G's total cost).
#[derive(Default)]
struct CanonStageTimes {
    extract: std::time::Duration,
    bliss: std::time::Duration,
    group_closure: std::time::Duration,
    group_minimization: std::time::Duration,
    /// `canonical_vertex_order`+`canonical_edges`+`canonicalize_graph_part_seeded`,
    /// summed across every survivor -- see the doc comment where this is
    /// computed for why it's split out from `group_minimization` (same
    /// AC-canonicalization machinery, but seeded and re-run per SURVIVOR
    /// rather than per raw GROUP element).
    stage_d_per_survivor: std::time::Duration,
    content: tamarin_theory::canon::ContentStageTimes,
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
    /// of `Dummy` `NodeId`s, so a caller (`DUMP_DUMMY_SWAPS=1`) can
    /// report exactly which timepoints these are for a node worth
    /// inspecting further.
    dummy_less_only_example: Option<String>,
}

/// Per-`VertexKind`-label counts of vertex SLOTS moved by a node's
/// automorphism generators (a transposition contributes 2, a 3-cycle 3,
/// etc. -- see [`classify_generator_moves`]), split further into
/// `identical_debug`/`different_debug`: whether the moved vertex's
/// `{:?}` rendering is byte-identical to the vertex its generator maps
/// it to. Answers "does this theory's real data only ever swap
/// content-free structural vertices (Dummy/EdgeRelation/...), or does it
/// swap real RuleInstance/Action content too -- and when it does, is
/// that content a literal duplicate or genuinely different (varying only
/// in ways canonicalization happens to erase, e.g. variable identity)?"
/// -- the question the crude "canonicalization erases the difference"
/// hand-wave from before was never actually checked against.
#[derive(Debug, Default, Clone)]
struct SwapKindCounts {
    moved_total: BTreeMap<&'static str, u64>,
    identical_debug: BTreeMap<&'static str, u64>,
    different_debug: BTreeMap<&'static str, u64>,
}

impl SwapKindCounts {
    fn merge(&mut self, other: &SwapKindCounts) {
        for (k, v) in &other.moved_total {
            *self.moved_total.entry(k).or_insert(0) += v;
        }
        for (k, v) in &other.identical_debug {
            *self.identical_debug.entry(k).or_insert(0) += v;
        }
        for (k, v) in &other.different_debug {
            *self.different_debug.entry(k).or_insert(0) += v;
        }
    }
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

/// Classifies every vertex slot `part`'s automorphism GENERATORS (bliss's
/// own minimal reported set, not the full closed group -- a compound
/// element's moved-vertex set is already a subset of the union of the
/// generators' own, so this is representative without needing to walk
/// the whole group) actually move, by `VertexKind` and by whether the
/// moved vertex's content is a byte-identical `{:?}` match to whatever
/// its generator maps it to.
fn classify_generator_moves(
    part: &tamarin_theory::canon_graph::GraphPart,
    generators: &[tamarin_theory::bliss_proc::Permutation],
) -> SwapKindCounts {
    let mut counts = SwapKindCounts::default();
    for g in generators {
        for v in 0..part.vertices.len() {
            let target = g.image_of(v);
            if target == v {
                continue;
            }
            let label = vertex_kind_label(&part.vertices[v]);
            *counts.moved_total.entry(label).or_insert(0) += 1;
            if format!("{:?}", part.vertices[v]) == format!("{:?}", part.vertices[target]) {
                *counts.identical_debug.entry(label).or_insert(0) += 1;
            } else {
                *counts.different_debug.entry(label).or_insert(0) += 1;
            }
        }
    }
    counts
}

/// `Some("<nid_a> <-> <nid_b>")` naming one concrete swapped pair of
/// `Dummy` `NodeId`s when `swaps` shows the node's generators move ONLY
/// `Dummy`/`LessRelation` vertices -- see `CanonStageTimes::dummy_less_only_example`'s
/// own doc comment for why this is worth naming.
fn find_dummy_less_only_example(
    part: &tamarin_theory::canon_graph::GraphPart,
    generators: &[tamarin_theory::bliss_proc::Permutation],
    swaps: &SwapKindCounts,
) -> Option<String> {
    let moved_kinds: BTreeSet<&str> = swaps.moved_total.keys().copied().collect();
    if moved_kinds.is_empty() || !moved_kinds.iter().all(|k| *k == "Dummy" || *k == "LessRelation") {
        return None;
    }
    for g in generators {
        for v in 0..part.vertices.len() {
            let target = g.image_of(v);
            if target == v {
                continue;
            }
            if let VertexKind::Dummy(nid_a) = &part.vertices[v] {
                if let VertexKind::Dummy(nid_b) = &part.vertices[target] {
                    return Some(format!("{nid_a} <-> {nid_b}"));
                }
            }
        }
    }
    None
}

/// Re-derives `canonicalize_constraint_system`'s FULL pipeline --
/// `canon.rs`'s `pub`(-now) building blocks: `extract_graph_part` ->
/// `graph_part_to_dimacs`+`run_bliss` -> `generate_group` ->
/// `minimal_graph_part_labelings`'s own per-element minimization loop
/// (inlined here so group-closure and minimization can be timed
/// SEPARATELY rather than as the one `minimal_graph_part_labelings`
/// bucket) -> Stage G's `canonicalize_system_content_seeded_profiled`,
/// run ONCE PER SURVIVOR exactly like `canonicalize_constraint_system`
/// itself does (`for (labeling, graph_term) in &survivors { ... }`) --
/// purely to measure where the time inside it goes.
///
/// Returns `None` for the two failure cases `canonicalize_constraint_system`
/// itself would return `Err` for (both extremely unlikely given the real
/// call just succeeded moments before this runs on the SAME `sys`) --
/// the caller should just skip recording stats for that node rather than
/// treat it as a hard error, since this is diagnostic-only.
fn profile_canonicalize_stages(
    sys: &System,
    colors: &tamarin_theory::canon_color::ColorTable,
) -> Option<CanonStageTimes> {
    use tamarin_theory::canon::{
        canonicalize_graph_part_seeded, canonicalize_system_content_seeded_profiled,
    };

    let t0 = std::time::Instant::now();
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

    let t1 = std::time::Instant::now();
    let dimacs = graph_part_to_dimacs(&part).ok()?;
    let result = run_bliss(&dimacs).ok()?;
    let bliss = t1.elapsed();

    let t2 = std::time::Instant::now();
    let group = generate_group(&result.generators, part.vertices.len());
    let group_closure = t2.elapsed();

    let t3 = std::time::Instant::now();
    let mut survivors: Vec<(_, tamarin_term::lterm::LNTerm)> = Vec::new();
    for g in &group {
        let candidate_labeling = result.canonical_labeling.compose(g);
        let ordered = canonical_vertex_order(&part, &candidate_labeling);
        let edges = canonical_edges(&part, &candidate_labeling);
        let term = canonicalize_graph_part(&ordered, &edges);
        match survivors.first() {
            None => survivors.push((candidate_labeling, term)),
            Some((_, best)) => match term.cmp(best) {
                std::cmp::Ordering::Less => {
                    survivors.clear();
                    survivors.push((candidate_labeling, term));
                }
                std::cmp::Ordering::Equal => survivors.push((candidate_labeling, term)),
                std::cmp::Ordering::Greater => {}
            },
        }
    }
    let group_minimization = t3.elapsed();

    // Stage D (re-derive the SEEDED labelling for each survivor -- this
    // re-runs the SAME AC-canonicalization machinery as `canonicalize_graph_part`
    // above, just seeded, once per survivor) + Stage G, once per survivor
    // -- see this function's own doc comment. Stage D's own re-derivation
    // was the ORIGINAL source of a ~9s/1000-node gap between this
    // function's stage-sum and its own coarse wall time on wireguard.spthy
    // -- it does real, expensive AC-canonicalization work and was
    // initially left unwrapped by a timer entirely (a bug in this
    // instrumentation, not a mystery in the pipeline -- see the
    // `t_canon_wall_total` cross-check this bug was caught by).
    let mut t_stage_d_per_survivor = std::time::Duration::ZERO;
    let mut content = tamarin_theory::canon::ContentStageTimes::default();
    for (labeling, _) in &survivors {
        let t4 = std::time::Instant::now();
        let ordered = canonical_vertex_order(&part, labeling);
        let edges = canonical_edges(&part, labeling);
        let (graph_term, labelling) = canonicalize_graph_part_seeded(&ordered, &edges);
        t_stage_d_per_survivor += t4.elapsed();
        let (_, c) = canonicalize_system_content_seeded_profiled(sys, &labelling, graph_term);
        content.formulas += c.formulas;
        content.solved_formulas += c.solved_formulas;
        content.lemmas += c.lemmas;
        content.eq_store_subst += c.eq_store_subst;
        content.eq_store_conj += c.eq_store_conj;
        content.subterm_store += c.subterm_store;
        content.goals += c.goals;
        // Counts are per-`sys`, not per-survivor -- identical every
        // iteration, so just take the last (any) one rather than sum.
        content.num_formulas = c.num_formulas;
        content.num_solved_formulas = c.num_solved_formulas;
        content.num_lemmas = c.num_lemmas;
        content.num_eq_store_subst = c.num_eq_store_subst;
        content.num_eq_store_conj = c.num_eq_store_conj;
        content.num_eq_store_conj_alternatives = c.num_eq_store_conj_alternatives;
        content.num_subterms = c.num_subterms;
        content.num_solved_subterms = c.num_solved_subterms;
        content.num_neg_subterms = c.num_neg_subterms;
        content.num_goals = c.num_goals;
    }

    let swaps = classify_generator_moves(&part, &result.generators);
    let dummy_less_only_example = find_dummy_less_only_example(&part, &result.generators, &swaps);

    Some(CanonStageTimes {
        extract,
        bliss,
        group_closure,
        group_minimization,
        stage_d_per_survivor: t_stage_d_per_survivor,
        content,
        vertices: part.vertices.len(),
        edges: part.edges.len(),
        generators: result.generators.len(),
        group_size: group.len(),
        survivors: survivors.len(),
        swaps,
        dummy_less_only_example,
    })
}

struct Args {
    theory_path: String,
    lemma: String,
    max_depth: usize,
    max_nodes: usize,
    close_threshold: usize,
}

fn parse_args() -> Args {
    let raw: Vec<String> = std::env::args().collect();
    if raw.len() < 3 {
        eprintln!(
            "usage: explore_canonical_matches <theory.spthy> <lemma> \
             [--max-depth N] [--max-nodes N] [--close-threshold N]"
        );
        std::process::exit(2);
    }
    let mut max_depth = 4usize;
    let mut max_nodes = 2000usize;
    let mut close_threshold = 1usize;
    let mut i = 3;
    while i < raw.len() {
        match raw[i].as_str() {
            "--max-depth" => {
                max_depth = raw[i + 1].parse().expect("--max-depth wants an integer");
                i += 2;
            }
            "--max-nodes" => {
                max_nodes = raw[i + 1].parse().expect("--max-nodes wants an integer");
                i += 2;
            }
            "--close-threshold" => {
                close_threshold = raw[i + 1]
                    .parse()
                    .expect("--close-threshold wants an integer");
                i += 2;
            }
            other => {
                eprintln!("unrecognized argument: {other}");
                std::process::exit(2);
            }
        }
    }
    Args {
        theory_path: raw[1].clone(),
        lemma: raw[2].clone(),
        max_depth,
        max_nodes,
        close_threshold,
    }
}

fn main() {
    let args = parse_args();

    if !bliss_available() {
        eprintln!("bliss not available and TAM_ALLOW_NO_BLISS=1 set -- nothing to do, exiting");
        return;
    }

    // `elaborated` itself isn't needed below: `build_lemma_proof_context`'s
    // `ctx.color_table` is built from this same theory (`ProofContext::new_impl`),
    // so it's the `&ColorTable` `canonicalize_constraint_system` needs --
    // no separate `&Theory` argument any more (see `canon_color.rs`'s own
    // "why this table takes..." doc section).
    let (parsed, _elaborated, maude) = common::load_theory_with_maude(&args.theory_path);

    // Same setup `prove_lemma` itself uses -- see `build_lemma_proof_context`'s
    // own doc comment for exactly why this is a shared helper rather than
    // hand-rolled here. `_user_funs_guard` must stay alive for the WHOLE
    // exploration below (canonicalization needs the installed signature).
    let (ctx, initial_sys, _skeleton_tree, _user_funs_guard) = build_lemma_proof_context(
        &parsed,
        &args.lemma,
        maude,
        None,
        "",
        &CliHeuristic::default(),
        CutStrategy::Dfs,
        None,
    )
    .unwrap_or_else(|e| panic!("build_lemma_proof_context({}): {e:?}", args.lemma));

    let mut visited: Vec<Visited> = Vec::new();
    let mut queue: VecDeque<(Vec<PathStep>, System, usize)> = VecDeque::new();
    queue.push_back((Vec::new(), initial_sys, 0));

    let mut total_nodes = 0usize;
    let mut canonicalize_failures = 0usize;
    let mut exact_matches = 0usize;
    let mut close_matches = 0usize;
    // Every successfully-canonicalized node is pushed to `visited`
    // regardless of whether it exactly matches something already there
    // (needed so a LATER node can still be compared against it -- see
    // `Visited`'s own doc comment). That means `visited.len()` -- and so
    // `total_nodes` -- counts every node DISCOVERED, not the number of
    // distinct canonical systems found: a class of 3 mutually-matching
    // nodes contributes 3 to `total_nodes` but only 1 NEW canonical
    // system. `nodes_kept` (first-seen representative of its class) and
    // `nodes_deduplicated` (exactly matched an already-kept
    // representative) separate the two -- `nodes_kept + nodes_deduplicated`
    // is asserted to equal every successfully-canonicalized node below,
    // so a caller who only wants the DISTINCT count uses `nodes_kept`
    // rather than `total_nodes`.
    let mut nodes_kept = 0usize;
    let mut nodes_deduplicated = 0usize;

    // `DEPTH_STATS=1`: per-depth node counts, and per-depth counts of
    // FINISHED (solved/contradictory/unfinishable) leaves, broken down by
    // the path's FIRST proof method (`candidate_methods` offers several
    // top-level strategies -- e.g. `Simplify` vs `Induction` -- and this
    // BFS explores every one of them, not just the single one the real
    // greedy heuristic picks; a lemma whose heuristic-preferred branch
    // finishes shallow can still have a much deeper sibling branch under
    // a DIFFERENT top-level choice, which a GUI user following the
    // heuristic's suggestions would never manually walk into). Added to
    // diagnose exactly that shape of "why is there a node this deep when
    // I can't find one in the GUI" question before trusting depth-N
    // output at face value.
    let depth_stats = std::env::var("DEPTH_STATS").is_ok();
    // `QUIET=1`: suppress the per-match `EXACT MATCH`/`CLOSE MATCH`
    // println!s. Counting/classification (`exact_matches`/`close_matches`/
    // `nodes_kept`/`nodes_deduplicated`) still happens regardless -- this
    // only skips the terminal write AND the `format_path` calls that
    // build it. Needed for AC-heavy theories (e.g. `diffie-hellman`):
    // `PathStep::method` is `pretty_proof_method_inline`'s FULL pretty-print
    // of the goal term, which for a Diffie-Hellman/hash-heavy theory like
    // wireguard.spthy can run to several KB per step (deeply nested
    // `h(...)`/`aead(...)`/exponent terms) -- printing two whole paths per
    // match, times potentially thousands of matches, is what actually
    // makes the terminal crawl (confirmed: a 120s run against
    // wireguard.spthy produced ~1MB/17.5k lines of match output alone
    // before even finishing).
    let quiet = std::env::var("QUIET").is_ok();
    // `PROGRESS=1`: a stderr heartbeat every 20 nodes (elapsed time,
    // nodes/sec, queue length) -- a long exploration otherwise prints
    // NOTHING until it either finishes or hits `--max-nodes`, so there is
    // no way to tell "still working" from "hung" from the outside.
    let progress = std::env::var("PROGRESS").is_ok();
    let progress_start = std::time::Instant::now();
    // `PROFILE=1`: accumulate wall time spent in each pipeline stage, to
    // answer "where's the bottleneck" with real numbers instead of
    // guessing. Four stages, in the order they run per node:
    //   1. `t_canonicalize`  -- `canonicalize_constraint_system` (Stage
    //      A-G: graph extraction, the `bliss` subprocess, formula/eq-store/
    //      subterm-store canonization). Independent of the Tamarin solver
    //      itself.
    //   2. `t_fingerprint`   -- `fingerprint_constraint_system` plus the
    //      `compare_fingerprints` loop against every prior `visited` entry
    //      (so this grows with how many DISTINCT+duplicate nodes have
    //      accumulated so far -- an O(n) cost per node, O(n^2) overall).
    //   3. `t_candidate_methods` -- ranking/selecting which goals are
    //      candidates at this node (`rank_goals_with`'s heuristic scoring).
    //   4. `t_exec_proof_method` -- actually SOLVING each candidate goal
    //      (unification, narrowing, source-case lookups -- the one most
    //      likely to route through `MaudeHandle` for an AC-heavy theory).
    // Stages 3+4 together are "Tamarin's own proof-search logic"; stages
    // 1+2 are "this tool's canonicalization/dedup layer".
    let profile = std::env::var("PROFILE").is_ok();
    let mut t_canonicalize = std::time::Duration::ZERO;
    let mut t_fingerprint = std::time::Duration::ZERO;
    let mut t_candidate_methods = std::time::Duration::ZERO;
    let mut t_exec_proof_method = std::time::Duration::ZERO;
    // `PROFILE_CANON=1`: the `t_canonicalize` bucket above broken down
    // further -- see [`profile_canonicalize_stages`]'s own doc comment.
    let profile_canon = std::env::var("PROFILE_CANON").is_ok();
    // `DUMP_DUMMY_SWAPS=1` (requires `PROFILE_CANON=1` too): print the
    // PATH to the first few nodes whose automorphism generators move
    // ONLY `Dummy`/`LessRelation` vertices -- i.e. a `Dummy(nid)` with no
    // `RuleInstance`/`Action` vertex of its own, connected only via an
    // `i < j` ordering constraint, getting freely permuted against
    // another such node. Reproduce the printed path in the GUI to
    // inspect one of these directly (the raw NodeIds themselves are an
    // internal detail of THIS run, not meaningful across a fresh replay
    // -- the printed `sys.nodes`/`less_atoms` dump is what to compare
    // against what the GUI shows at the same point in the path).
    let dump_dummy_swaps = std::env::var("DUMP_DUMMY_SWAPS").is_ok();
    let mut dummy_swap_examples_shown = 0usize;
    const DUMMY_SWAP_EXAMPLES_LIMIT: usize = 3;
    let mut t_canon_extract = std::time::Duration::ZERO;
    let mut t_canon_bliss = std::time::Duration::ZERO;
    let mut t_canon_group_closure = std::time::Duration::ZERO;
    let mut t_canon_group_minimization = std::time::Duration::ZERO;
    let mut t_canon_stage_d_per_survivor = std::time::Duration::ZERO;
    let mut canon_profiled_nodes = 0usize;
    let mut canon_total_vertices = 0usize;
    let mut canon_total_edges = 0usize;
    let mut canon_max_generators = 0usize;
    let mut canon_max_group_size = 0usize;
    let mut canon_max_survivors = 0usize;
    let mut canon_total_group_size = 0u64;
    // How much of the automorphism group `minimal_graph_part_labelings`
    // actually keeps as a survivor -- i.e. is `generate_group` mostly
    // reporting REAL alpha-equivalences (survivors ~= group_size, so the
    // minimization work is inherent to genuine symmetry) or mostly
    // SPURIOUS ones bliss's coloring couldn't tell apart but the full
    // vertex CONTENT can (survivors << group_size, so a finer coloring
    // upfront could shrink the group bliss reports in the first place,
    // cutting both Stage F and Stage D's cost). Only counted for nodes
    // with a NONTRIVIAL group (group_size > 1) -- a trivial group's
    // ratio is vacuously 1/1 and would just dilute the average.
    let mut canon_total_survivors_nontrivial = 0u64;
    let mut canon_total_group_size_nontrivial = 0u64;
    let mut canon_nodes_nontrivial_group = 0usize;
    let mut canon_nodes_all_survive = 0usize;
    let mut canon_sum_survivor_ratio = 0.0f64;
    // Keyed by group_size -- (node count, total survivors) -- so the
    // relationship between group size and survivor count can be read
    // directly off real data instead of guessed from one aggregate ratio
    // (e.g. "small groups mostly survive fully, large groups mostly
    // don't" would show up here even if the OVERALL average ratio looks
    // moderate).
    let mut canon_group_size_histogram: BTreeMap<usize, (u64, u64)> = BTreeMap::new();
    // What KIND of vertex the found automorphisms actually move, summed
    // across every profiled node -- see `SwapKindCounts`'s own doc
    // comment for exactly what this answers.
    let mut canon_swap_kinds = SwapKindCounts::default();
    // Stage G (`canonicalize_system_content_seeded_profiled`), summed
    // ACROSS SURVIVORS already (see `CanonStageTimes::content`'s own doc
    // comment) -- so these durations are the REAL total Stage G cost per
    // node, not a single representative call.
    let mut t_content_formulas = std::time::Duration::ZERO;
    let mut t_content_solved_formulas = std::time::Duration::ZERO;
    let mut t_content_lemmas = std::time::Duration::ZERO;
    let mut t_content_eq_store_subst = std::time::Duration::ZERO;
    let mut t_content_eq_store_conj = std::time::Duration::ZERO;
    let mut t_content_subterm_store = std::time::Duration::ZERO;
    let mut t_content_goals = std::time::Duration::ZERO;
    let mut canon_total_formulas = 0u64;
    let mut canon_total_solved_formulas = 0u64;
    let mut canon_total_lemmas = 0u64;
    let mut canon_total_eq_store_subst = 0u64;
    let mut canon_total_eq_store_conj = 0u64;
    let mut canon_total_eq_store_conj_alternatives = 0u64;
    let mut canon_total_subterms = 0u64;
    let mut canon_total_goals = 0u64;
    let mut canon_max_eq_store_conj_alternatives = 0usize;
    // Coarse wall time of the WHOLE `profile_canonicalize_stages` call,
    // measured from OUTSIDE it -- a cross-check against the sum of its
    // own internal per-stage timers, to catch a genuine "first call vs
    // second call" asymmetry (e.g. subprocess-spawn cost scaling with
    // the parent process's growing memory footprint) rather than an
    // accounting bug in this tool's own stage-summing.
    let mut t_canon_wall_total = std::time::Duration::ZERO;
    let mut nodes_by_depth: BTreeMap<usize, usize> = BTreeMap::new();
    // Per-depth halves of `nodes_kept`/`nodes_deduplicated` (see their own
    // doc comment above for what each means) -- e.g. a depth whose nodes
    // are almost entirely `nodes_deduplicated_by_depth` entries is a
    // depth where the tree-search-with-merging approach would save the
    // most: many DIFFERENT paths reaching the same handful of distinct
    // systems.
    let mut nodes_kept_by_depth: BTreeMap<usize, usize> = BTreeMap::new();
    let mut nodes_deduplicated_by_depth: BTreeMap<usize, usize> = BTreeMap::new();
    let mut finished_by_branch: BTreeMap<String, BTreeMap<usize, usize>> = BTreeMap::new();

    while let Some((path, sys, depth)) = queue.pop_front() {
        if total_nodes >= args.max_nodes {
            eprintln!(
                "--max-nodes={} reached with {} nodes still queued -- stopping early",
                args.max_nodes,
                queue.len()
            );
            break;
        }
        total_nodes += 1;

        if progress && total_nodes % 20 == 0 {
            let elapsed = progress_start.elapsed().as_secs_f64();
            eprintln!(
                "[progress] {total_nodes} node(s) in {elapsed:.1}s ({:.2} nodes/s), \
                 depth={depth}, queue={}, kept={nodes_kept}, deduplicated={nodes_deduplicated}",
                total_nodes as f64 / elapsed.max(0.001),
                queue.len()
            );
        }

        if depth_stats {
            *nodes_by_depth.entry(depth).or_insert(0) += 1;
        }

        if std::env::var("DUMP_FORMULAS").is_ok() {
            eprintln!("--- formulas/solved_formulas at {} ---", format_path(&path));
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

        // Canonicalization is otherwise-untested against real, varied
        // proof-search states -- a panic (e.g. a `theta`-miss) anywhere
        // inside it must not kill the whole exploration before every
        // other path has been tried. Caught here, not fixed: this tool's
        // job is to FIND such gaps, not paper over them.
        let stage_start = std::time::Instant::now();
        let canon_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            canonicalize_constraint_system(&sys, &ctx.color_table)
        }));
        if profile {
            t_canonicalize += stage_start.elapsed();
        }
        // Re-derives (and discards) the same canonicalization a second
        // time purely to get its internal stage breakdown -- see
        // `profile_canonicalize_stages`'s own doc comment for why this
        // duplication is necessary (Stage G's content canonization isn't
        // `pub`) and why it's acceptable (diagnostic-only, discarded).
        if profile_canon {
            let stage_start2 = std::time::Instant::now();
            let profiled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                profile_canonicalize_stages(&sys, &ctx.color_table)
            }))
            .ok()
            .flatten();
            t_canon_wall_total += stage_start2.elapsed();
            if let Some(stages) = profiled {
                t_canon_extract += stages.extract;
                t_canon_bliss += stages.bliss;
                t_canon_group_closure += stages.group_closure;
                t_canon_group_minimization += stages.group_minimization;
                t_canon_stage_d_per_survivor += stages.stage_d_per_survivor;
                canon_profiled_nodes += 1;
                canon_total_vertices += stages.vertices;
                canon_total_edges += stages.edges;
                canon_max_generators = canon_max_generators.max(stages.generators);
                canon_max_group_size = canon_max_group_size.max(stages.group_size);
                canon_max_survivors = canon_max_survivors.max(stages.survivors);
                canon_total_group_size += stages.group_size as u64;
                if stages.group_size > 1 {
                    canon_nodes_nontrivial_group += 1;
                    canon_total_survivors_nontrivial += stages.survivors as u64;
                    canon_total_group_size_nontrivial += stages.group_size as u64;
                    canon_sum_survivor_ratio += stages.survivors as f64 / stages.group_size as f64;
                    if stages.survivors == stages.group_size {
                        canon_nodes_all_survive += 1;
                    }
                    let entry = canon_group_size_histogram.entry(stages.group_size).or_insert((0, 0));
                    entry.0 += 1;
                    entry.1 += stages.survivors as u64;
                    canon_swap_kinds.merge(&stages.swaps);
                    if dump_dummy_swaps && dummy_swap_examples_shown < DUMMY_SWAP_EXAMPLES_LIMIT {
                        if let Some(example) = &stages.dummy_less_only_example {
                            dummy_swap_examples_shown += 1;
                            eprintln!(
                                "\n[DUMP_DUMMY_SWAPS #{dummy_swap_examples_shown}] node with a \
                                 Dummy/LessRelation-only automorphism (|Aut(G)|={}, \
                                 swaps {example}), reached via:\n  {}",
                                stages.group_size,
                                format_path(&path)
                            );
                            eprintln!(
                                "  sys.nodes (rule-instance timepoints): {:?}",
                                sys.nodes.iter().map(|(n, _)| n.to_string()).collect::<Vec<_>>()
                            );
                            eprintln!("  sys.less_atoms (i < j constraints):");
                            for la in sys.less_atoms_in_set_order() {
                                eprintln!(
                                    "    {} < {}  ({:?})",
                                    la.smaller, la.larger, la.reason
                                );
                            }
                        }
                    }
                }
                t_content_formulas += stages.content.formulas;
                t_content_solved_formulas += stages.content.solved_formulas;
                t_content_lemmas += stages.content.lemmas;
                t_content_eq_store_subst += stages.content.eq_store_subst;
                t_content_eq_store_conj += stages.content.eq_store_conj;
                t_content_subterm_store += stages.content.subterm_store;
                t_content_goals += stages.content.goals;
                canon_total_formulas += stages.content.num_formulas as u64;
                canon_total_solved_formulas += stages.content.num_solved_formulas as u64;
                canon_total_lemmas += stages.content.num_lemmas as u64;
                canon_total_eq_store_subst += stages.content.num_eq_store_subst as u64;
                canon_total_eq_store_conj += stages.content.num_eq_store_conj as u64;
                canon_total_eq_store_conj_alternatives +=
                    stages.content.num_eq_store_conj_alternatives as u64;
                canon_total_subterms += (stages.content.num_subterms
                    + stages.content.num_solved_subterms
                    + stages.content.num_neg_subterms) as u64;
                canon_total_goals += stages.content.num_goals as u64;
                canon_max_eq_store_conj_alternatives = canon_max_eq_store_conj_alternatives
                    .max(stages.content.num_eq_store_conj_alternatives);
            }
        }

        // Set below only when this node exactly matches an
        // already-kept canonical class -- see the module docs'
        // "why skipping a duplicate's expansion is sound" section for
        // the argument that this never loses a reachable system.
        let mut is_duplicate_node = false;

        match canon_result {
            Err(panic_payload) => {
                canonicalize_failures += 1;
                let msg = panic_payload
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| panic_payload.downcast_ref::<&str>().copied())
                    .unwrap_or("<non-string panic payload>");
                eprintln!(
                    "canonicalize_constraint_system PANICKED at {}: {msg}",
                    format_path(&path)
                );
                continue;
            }
            Ok(canonicalize_result) => match canonicalize_result {
            Ok(canon) => {
                let stage_start = std::time::Instant::now();
                let fp = fingerprint_constraint_system(&canon);
                // Tracks whether THIS node exactly matches any node
                // already in `visited` -- i.e. whether it's a duplicate
                // of an already-kept canonical class, rather than the
                // first representative of a new one. Still compares
                // against every prior entry (not just kept
                // representatives) so every matching PAIR is reported,
                // matching the existing exact/close-match printing below.
                let mut is_duplicate = false;
                for prior in &visited {
                    let report = compare_fingerprints(&prior.fingerprint, &fp);
                    let mismatched = report.mismatched_fields();
                    if mismatched.is_empty() {
                        exact_matches += 1;
                        is_duplicate = true;
                        if !quiet {
                            println!(
                                "EXACT MATCH (all {}/{} fields):\n  A: {}\n  B: {}\n",
                                report.matched_count(),
                                report.total_count(),
                                format_path(&prior.path),
                                format_path(&path)
                            );
                        }
                    } else if mismatched.len() <= args.close_threshold {
                        close_matches += 1;
                        if !quiet {
                            println!(
                                "CLOSE MATCH ({}/{} fields, mismatched: {mismatched:?}):\n  A: {}\n  B: {}\n",
                                report.matched_count(),
                                report.total_count(),
                                format_path(&prior.path),
                                format_path(&path)
                            );
                        }
                    }
                }
                if profile {
                    t_fingerprint += stage_start.elapsed();
                }
                if is_duplicate {
                    nodes_deduplicated += 1;
                    is_duplicate_node = true;
                    if depth_stats {
                        *nodes_deduplicated_by_depth.entry(depth).or_insert(0) += 1;
                    }
                } else {
                    nodes_kept += 1;
                    if depth_stats {
                        *nodes_kept_by_depth.entry(depth).or_insert(0) += 1;
                    }
                }
                visited.push(Visited {
                    path: path.clone(),
                    fingerprint: fp,
                });
            }
            Err(e) => {
                canonicalize_failures += 1;
                eprintln!(
                    "canonicalize_constraint_system failed at {}: {e:?}",
                    format_path(&path)
                );
            }
            },
        }

        // Don't re-expand a node whose canonical system was ALREADY
        // discovered -- turns the exhaustive TREE walk into the
        // tree-search-WITH-MERGING the module docs describe as future
        // work: an exact duplicate's whole subtree is, up to renaming,
        // identical to the subtree the FIRST (kept) occurrence of its
        // canonical class already explores. See the module docs' own
        // section for why this loses no reachable system.
        if is_duplicate_node {
            continue;
        }

        if depth >= args.max_depth {
            continue;
        }
        // A terminal system (solved/contradictory/unfinishable) has no
        // further candidates worth expanding -- `candidate_methods` would
        // just report `[Finished(_)]`, whose `exec_proof_method` produces
        // no cases anyway, so skipping it here is purely an efficiency
        // shortcut, not a correctness one. Timed as part of
        // `t_candidate_methods` (`PROFILE=1`): `is_finished` is itself a
        // full contradiction sweep, the same cost class as ranking.
        let stage_start = std::time::Instant::now();
        let finished = is_finished(&ctx, &sys);
        if profile {
            t_candidate_methods += stage_start.elapsed();
        }
        if finished.is_some() {
            if depth_stats {
                let branch = path
                    .first()
                    .map(|s| s.method.clone())
                    .unwrap_or_else(|| "<root>".to_string());
                *finished_by_branch.entry(branch).or_default().entry(depth).or_insert(0) += 1;
            }
            continue;
        }

        let stage_start = std::time::Instant::now();
        let methods = candidate_methods(&sys, &ctx, depth);
        if profile {
            t_candidate_methods += stage_start.elapsed();
        }
        for method in methods {
            let stage_start = std::time::Instant::now();
            let exec_result = exec_proof_method(&ctx, &method, &sys);
            if profile {
                t_exec_proof_method += stage_start.elapsed();
            }
            let Some(cases) = exec_result else {
                continue;
            };
            let method_str = pretty_proof_method_inline(&method);
            for (case_name, child_sys) in cases {
                let mut child_path = path.clone();
                child_path.push(PathStep {
                    method: method_str.clone(),
                    case: case_name,
                });
                queue.push_back((child_path, child_sys, depth + 1));
            }
        }
    }

    // `nodes_kept`/`nodes_deduplicated` only cover successfully-
    // canonicalized nodes (a `canonicalize_failures` node was never
    // classified into either bucket) -- see `nodes_kept`'s own doc
    // comment above for exactly what each count means. A mismatch here
    // is a real bookkeeping bug in the loop above, not a possible outcome
    // of any input theory, so this asserts (crashes loudly) rather than
    // silently reporting a wrong number.
    assert_eq!(
        nodes_kept + nodes_deduplicated,
        total_nodes - canonicalize_failures,
        "nodes_kept + nodes_deduplicated should equal every successfully-canonicalized node"
    );

    println!(
        "=== {} :: {} === nodes_discovered={} canonicalize_failures={} nodes_kept={} \
         nodes_deduplicated={} exact_matches={} close_matches={} \
         (max_depth={}, max_nodes={}, close_threshold={})",
        args.theory_path,
        args.lemma,
        total_nodes,
        canonicalize_failures,
        nodes_kept,
        nodes_deduplicated,
        exact_matches,
        close_matches,
        args.max_depth,
        args.max_nodes,
        args.close_threshold
    );

    if depth_stats {
        println!(
            "\n--- DEPTH_STATS: nodes by depth (discovered = kept + deduplicated, modulo \
             any canonicalize_failures at that depth) ---"
        );
        for (d, discovered) in &nodes_by_depth {
            let kept = nodes_kept_by_depth.get(d).copied().unwrap_or(0);
            let deduplicated = nodes_deduplicated_by_depth.get(d).copied().unwrap_or(0);
            let dedup_pct = if *discovered == 0 {
                0.0
            } else {
                100.0 * deduplicated as f64 / *discovered as f64
            };
            println!(
                "  depth {d}: discovered={discovered} kept={kept} deduplicated={deduplicated} \
                 ({dedup_pct:.1}% deduplicated)"
            );
        }
        println!(
            "\n--- DEPTH_STATS: FINISHED (solved/contradictory/unfinishable) leaves by \
             depth, grouped by the path's first proof method ---"
        );
        for (branch, by_depth) in &finished_by_branch {
            let total: usize = by_depth.values().sum();
            print!("  [{branch}] first step -- {total} finished leaf/leaves at depth(s): ");
            let parts: Vec<String> = by_depth.iter().map(|(d, n)| format!("{d}x{n}")).collect();
            println!("{}", parts.join(", "));
        }
    }

    if profile {
        let total = t_canonicalize + t_fingerprint + t_candidate_methods + t_exec_proof_method;
        let pct = |d: std::time::Duration| {
            if total.as_secs_f64() == 0.0 {
                0.0
            } else {
                100.0 * d.as_secs_f64() / total.as_secs_f64()
            }
        };
        println!(
            "\n--- PROFILE: wall time by pipeline stage, summed over {total_nodes} node(s) ---"
        );
        println!(
            "  canonicalize_constraint_system (bliss+graph):  {:>8.2}s ({:>5.1}%)",
            t_canonicalize.as_secs_f64(),
            pct(t_canonicalize)
        );
        println!(
            "  fingerprint + compare against visited:         {:>8.2}s ({:>5.1}%)",
            t_fingerprint.as_secs_f64(),
            pct(t_fingerprint)
        );
        println!(
            "  is_finished + candidate_methods (ranking):     {:>8.2}s ({:>5.1}%)",
            t_candidate_methods.as_secs_f64(),
            pct(t_candidate_methods)
        );
        println!(
            "  exec_proof_method (solving -- maude/AC here):  {:>8.2}s ({:>5.1}%)",
            t_exec_proof_method.as_secs_f64(),
            pct(t_exec_proof_method)
        );
        println!(
            "  sum of measured stages: {:.2}s (wall clock may exceed this by parse/elaborate/setup)",
            total.as_secs_f64()
        );
    }

    if profile_canon {
        let content_total = t_content_formulas
            + t_content_solved_formulas
            + t_content_lemmas
            + t_content_eq_store_subst
            + t_content_eq_store_conj
            + t_content_subterm_store
            + t_content_goals;
        let grand_total = t_canon_extract
            + t_canon_bliss
            + t_canon_group_closure
            + t_canon_group_minimization
            + t_canon_stage_d_per_survivor
            + content_total;
        let pct = |d: std::time::Duration| {
            if grand_total.as_secs_f64() == 0.0 {
                0.0
            } else {
                100.0 * d.as_secs_f64() / grand_total.as_secs_f64()
            }
        };
        println!(
            "\n--- PROFILE_CANON: canonicalize_constraint_system's OWN internal \
             stages, summed over {canon_profiled_nodes} node(s) (re-derived a second \
             time, discarded -- see the module docs) ---"
        );
        println!(
            "  Stage A  extract_graph_part:                {:>8.2}s ({:>5.1}%)",
            t_canon_extract.as_secs_f64(),
            pct(t_canon_extract)
        );
        println!(
            "  Stage C  dimacs + external bliss subprocess: {:>8.2}s ({:>5.1}%)",
            t_canon_bliss.as_secs_f64(),
            pct(t_canon_bliss)
        );
        println!(
            "  Stage F  automorphism GROUP CLOSURE:         {:>8.2}s ({:>5.1}%)  (generate_group)",
            t_canon_group_closure.as_secs_f64(),
            pct(t_canon_group_closure)
        );
        println!(
            "  Stage F  automorphism group MINIMIZATION:    {:>8.2}s ({:>5.1}%)  (re-canonicalize \
             the graph part once per group element)",
            t_canon_group_minimization.as_secs_f64(),
            pct(t_canon_group_minimization)
        );
        println!(
            "  Stage D  re-seed per survivor:               {:>8.2}s ({:>5.1}%)  \
             (canonicalize_graph_part_seeded, once per SURVIVOR -- same AC-canonicalization \
             cost class as group MINIMIZATION above, just re-run seeded)",
            t_canon_stage_d_per_survivor.as_secs_f64(),
            pct(t_canon_stage_d_per_survivor)
        );
        println!(
            "  Stage G  formulas:                           {:>8.2}s ({:>5.1}%)  ({} formula(s) total)",
            t_content_formulas.as_secs_f64(),
            pct(t_content_formulas),
            canon_total_formulas
        );
        println!(
            "  Stage G  solved_formulas:                    {:>8.2}s ({:>5.1}%)  ({} total)",
            t_content_solved_formulas.as_secs_f64(),
            pct(t_content_solved_formulas),
            canon_total_solved_formulas
        );
        println!(
            "  Stage G  lemmas:                             {:>8.2}s ({:>5.1}%)  ({} total)",
            t_content_lemmas.as_secs_f64(),
            pct(t_content_lemmas),
            canon_total_lemmas
        );
        println!(
            "  Stage G  eq_store.subst:                     {:>8.2}s ({:>5.1}%)  ({} entries total)",
            t_content_eq_store_subst.as_secs_f64(),
            pct(t_content_eq_store_subst),
            canon_total_eq_store_subst
        );
        println!(
            "  Stage G  eq_store.conj:                      {:>8.2}s ({:>5.1}%)  ({} EqDisj / \
             {} alternatives total, max {} alternatives on one node)",
            t_content_eq_store_conj.as_secs_f64(),
            pct(t_content_eq_store_conj),
            canon_total_eq_store_conj,
            canon_total_eq_store_conj_alternatives,
            canon_max_eq_store_conj_alternatives
        );
        println!(
            "  Stage G  subterm_store:                      {:>8.2}s ({:>5.1}%)  ({} \
             subterm/solved-subterm/neg-subterm entries total)",
            t_content_subterm_store.as_secs_f64(),
            pct(t_content_subterm_store),
            canon_total_subterms
        );
        println!(
            "  Stage G  goals:                              {:>8.2}s ({:>5.1}%)  ({} \
             goal(s) total, before dropping Action/Split)",
            t_content_goals.as_secs_f64(),
            pct(t_content_goals),
            canon_total_goals
        );
        println!(
            "  sum of measured stages: {:.2}s (Stage G is run ONCE PER SURVIVOR, already \
             summed accordingly -- see CanonStageTimes::content's own doc comment)",
            grand_total.as_secs_f64()
        );
        println!(
            "  coarse wall time of the whole profile_canonicalize_stages call: {:.2}s \
             (cross-check against the {:.2}s sum above -- a big gap between THESE two means \
             an accounting bug in this tool, not a real pipeline stage)",
            t_canon_wall_total.as_secs_f64(),
            grand_total.as_secs_f64()
        );
        if profile {
            println!(
                "  (vs. real canonicalize_constraint_system time t_canonicalize={:.2}s from \
                 PROFILE above -- a gap between THIS and the coarse wall time just above is \
                 the interesting one: same pipeline, same node, run twice back to back, \
                 should cost about the same)",
                t_canonicalize.as_secs_f64()
            );
        }
        if canon_profiled_nodes > 0 {
            println!(
                "  graph size: {:.1} vertices/node, {:.1} edges/node (avg)",
                canon_total_vertices as f64 / canon_profiled_nodes as f64,
                canon_total_edges as f64 / canon_profiled_nodes as f64
            );
            println!(
                "  automorphism group: max |generators|={canon_max_generators}, \
                 max |Aut(G)|={canon_max_group_size}, avg |Aut(G)|={:.1}, \
                 max |survivors|={canon_max_survivors}",
                canon_total_group_size as f64 / canon_profiled_nodes as f64
            );
            if canon_nodes_nontrivial_group > 0 {
                println!(
                    "  survivors/|Aut(G)| (nodes with a NONTRIVIAL group only, n={canon_nodes_nontrivial_group}): \
                     avg ratio={:.3} (unweighted per-node average), \
                     pooled ratio={:.3} (total survivors / total |Aut(G)|), \
                     {canon_nodes_all_survive}/{canon_nodes_nontrivial_group} nodes ({:.1}%) have EVERY \
                     automorphism survive minimization (survivors == |Aut(G)|)",
                    canon_sum_survivor_ratio / canon_nodes_nontrivial_group as f64,
                    canon_total_survivors_nontrivial as f64 / canon_total_group_size_nontrivial as f64,
                    100.0 * canon_nodes_all_survive as f64 / canon_nodes_nontrivial_group as f64
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
                for (group_size, (count, total_survivors)) in &canon_group_size_histogram {
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
                     transposition contributes 2 moved slots, a k-cycle k; \"identical\" \
                     means the moved vertex's {{:?}} rendering exactly matches the vertex \
                     its generator maps it to -- a literal duplicate; \"different\" means \
                     it doesn't, i.e. the automorphism moves genuinely different content):"
                );
                let mut labels: BTreeSet<&'static str> =
                    canon_swap_kinds.moved_total.keys().copied().collect();
                labels.extend(canon_swap_kinds.identical_debug.keys().copied());
                labels.extend(canon_swap_kinds.different_debug.keys().copied());
                for label in labels {
                    let total = canon_swap_kinds.moved_total.get(label).copied().unwrap_or(0);
                    let identical = canon_swap_kinds.identical_debug.get(label).copied().unwrap_or(0);
                    let different = canon_swap_kinds.different_debug.get(label).copied().unwrap_or(0);
                    println!(
                        "    {label:<18} moved={total:>7}  identical={identical:>7}  \
                         different={different:>7}"
                    );
                }
            }
        }
    }
}
