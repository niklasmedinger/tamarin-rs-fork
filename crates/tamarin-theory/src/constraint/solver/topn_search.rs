//! (No HS analog.) Top-N proof search: an opt-in alternative to the greedy
//! driver in [`super::search`]. It tries up to N of the heuristic's ranked
//! proof methods at each constraint system instead of only the first one that
//! applies, and optionally merges canonically equal systems reached along
//! different paths.
//!
//! The greedy driver commits to `candidate_methods`' first applicable method
//! at every node. Tamarin's problem is undecidable and its heuristics are
//! incomplete, so that line can fail to close — it runs into the depth limit
//! or never terminates — where another ranked method would have closed the
//! branch. This module searches the AND/OR graph of [`super::search_graph`]
//! over the top N methods: OR nodes are systems (or classes of canonically
//! equal ones), AND nodes applied methods with their cases, and statuses a
//! least fixpoint over the graph.
//!
//! ## Order
//!
//! A frontier [`Entry`] means "apply the next method of this class". Under
//! [`SearchOrder::IdDfs`] the frontier is a max-heap on
//! `(depth, N - idx, Reverse(class))`: the deepest entry first, then the
//! heuristic's rank, then the oldest class. Greedy proofs are deep and
//! narrow, and this order follows the heuristic's first choice down to the
//! leaves before it looks at any alternative, so a greedy proof that closes
//! is found at greedy cost. When the first choice runs into the depth limit,
//! the alternatives come up bottom-up, in the heuristic's order at each
//! depth. The limit deepens like the greedy driver's (4, 8, 16, …): entries
//! at the limit are parked and retried in the next iteration. The graph is
//! kept across iterations, so no method runs twice. [`SearchOrder::Bfs`] is
//! the same engine with the shallowest entry first.
//!
//! ## Concurrency
//!
//! Each step takes a batch of up to B entries and runs in three phases:
//! select (sequential: pop entries, move each class's `System` into a task),
//! expand (in parallel: rank, execute, check and canonicalize the cases —
//! the expensive part), integrate (sequential, in selection order: merge,
//! push entries, propagate statuses). Workers never touch the graph, and each
//! gets its own maude handle whose counter is seeded from its system
//! ([`WorkerEnv`]). A batch always runs to completion, also when one of its
//! tasks finds a trace; the search stops after it. So the outcome depends on
//! B but not on the thread count or on timing, the deadline apart.
//!
//! ## Output
//!
//! The settled graph is turned back into an ordinary [`ProofNode`] tree
//! ([`Materializer`]): starting from the root system, only the settling
//! method of each class runs again. With merging, a system reached along a
//! different path than its class's representative gets the method whose
//! canonical form equals the representative's. The result prints and replays
//! like any proof.
//!
//! ## Configuration
//!
//! Opt-in; with `TAM_RS_TOP_METHODS` unset the greedy driver runs unchanged.
//! Read once per process ([`top_n_from_env`]):
//! - `TAM_RS_TOP_METHODS=N`: search the first N applicable methods per system;
//! - `TAM_RS_SEARCH_ORDER=iddfs|bfs`: the frontier order, default `iddfs`;
//! - `TAM_RS_MERGE` (presence): merge canonically equal systems (needs `bliss`);
//! - `TAM_RS_CANON_INVARIANT=off|on|check`: how merging canonicalizes
//!   ([`super::canon_table`]), as in the MCGS search;
//! - `TAM_RS_TOPN_BATCH=B`: entries expanded per step, default the rayon
//!   thread count; pin it for reproducible runs;
//! - `TAM_RS_TOPN_STATS` (presence): one summary line per search on stderr,
//!   and one per iteration.
//! - `TAM_RS_SEARCH_STATS=SECS`: `[search-stats]` lines on stderr, the merge
//!   counters both graph searches share in one format: one every SECS
//!   seconds while the search runs (`0`: none), so that a run killed at a
//!   timeout leaves its last counts, and a last one with `final=1`
//!   ([`search_graph::StatsReporter`]).

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use rayon::prelude::*;

use crate::canon_term_graph::system_to_term;
use crate::constraint::solver::canon_table::{canon_mode_from_env, CanonicalTable};
use crate::constraint::solver::context::ProofContext;
use crate::constraint::solver::reduction::avoid_fresh_state;
use crate::constraint::solver::search::{self, candidate_methods, ProofNode};
use crate::constraint::solver::search_graph::{
    self, apply_method, classify, kept_system, Applied, Child, ClassId, End, Interned, KeptSystems,
    Materializer, MergeCounts, StatsReporter, Status, Stored, WorkStats, WorkerEnv, DEPTH_CAP,
    ROOT,
};
use crate::constraint::system::System;
use crate::prove::ProveError;
use tamarin_utils::FastMap;

// =============================================================================
// Configuration
// =============================================================================

/// Which frontier entry a search step takes first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchOrder {
    /// Deepest first, then the heuristic's rank; the depth limit deepens.
    IdDfs,
    /// Shallowest first, then the heuristic's rank.
    Bfs,
}

/// The top-N search's settings, stamped onto each lemma's `ProofContext`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TopNConfig {
    /// At most this many methods per system: the first that apply, in the
    /// heuristic's order.
    pub n: u32,
    pub order: SearchOrder,
    /// Merge canonically equal systems into one class.
    pub merge: bool,
    /// Frontier entries expanded concurrently per step.
    pub batch: usize,
}

fn parse_positive(var: &str, value: &str) -> usize {
    value
        .parse::<usize>()
        .ok()
        .filter(|&n| n >= 1)
        .unwrap_or_else(|| panic!("{var}={value:?}: expected a positive integer"))
}

/// The configuration from the environment, read once per process; `None`
/// keeps the greedy search. The value-bearing variables use a hand-rolled
/// `OnceLock` and the presence-only one `env_gate!` (see
/// `tamarin_utils::env_gate`).
pub fn top_n_from_env() -> Option<TopNConfig> {
    static CONFIG: OnceLock<Option<TopNConfig>> = OnceLock::new();
    *CONFIG.get_or_init(|| {
        let n = match std::env::var("TAM_RS_TOP_METHODS") {
            Err(std::env::VarError::NotPresent) => return None,
            Ok(v) => parse_positive("TAM_RS_TOP_METHODS", &v),
            Err(e) => panic!("TAM_RS_TOP_METHODS: {e}"),
        };
        let n = u32::try_from(n).unwrap_or_else(|_| panic!("TAM_RS_TOP_METHODS={n}: too large"));
        let order = match std::env::var("TAM_RS_SEARCH_ORDER").as_deref() {
            Err(std::env::VarError::NotPresent) | Ok("iddfs") => SearchOrder::IdDfs,
            Ok("bfs") => SearchOrder::Bfs,
            other => panic!("TAM_RS_SEARCH_ORDER={other:?}: expected `iddfs` or `bfs`"),
        };
        let batch = match std::env::var("TAM_RS_TOPN_BATCH") {
            Err(std::env::VarError::NotPresent) => rayon::current_num_threads(),
            Ok(v) => parse_positive("TAM_RS_TOPN_BATCH", &v),
            Err(e) => panic!("TAM_RS_TOPN_BATCH: {e}"),
        };
        Some(TopNConfig {
            n,
            order,
            merge: tamarin_utils::env_gate!("TAM_RS_MERGE"),
            batch,
        })
    })
}

/// [`top_n_from_env`] for one lemma of a theory, checked against what
/// merging needs up front rather than at the first canonicalization.
/// Theories with processes need nothing extra: the prover translates them
/// before building the proof context.
pub fn config_for_lemma() -> Option<TopNConfig> {
    let config = top_n_from_env()?;
    if config.merge {
        // Panics itself, with install hints, unless TAM_ALLOW_NO_BLISS=1.
        assert!(
            crate::bliss_proc::bliss_available(),
            "TAM_RS_MERGE needs a working `bliss` executable"
        );
    }
    Some(config)
}

// =============================================================================
// The AND/OR graph
// =============================================================================

/// The first depth limit of [`SearchOrder::IdDfs`], as in the greedy driver.
const FIRST_LIMIT: u32 = 4;

/// What the top-N search keeps per class beyond the shared graph.
#[derive(Debug, Default)]
struct TopNExt {
    /// `(depth, rank)` of the class's one live frontier entry: `None` while
    /// it is being expanded or has no entry. Any other entry is stale.
    pending: Option<(u32, u32)>,
}

impl search_graph::DumpExt for TopNExt {
    fn and_fields(&self, a: usize, row: &mut serde_json::Map<String, serde_json::Value>) {
        // Methods are applied in the heuristic's order, the first that apply,
        // so the method's index is its rank among the applicable ones.
        row.insert("rank".into(), serde_json::json!(a));
    }
}

type SearchGraph = search_graph::SearchGraph<TopNExt>;

impl SearchGraph {
    fn entry_is_live(&self, e: &Entry) -> bool {
        self.class(e.class.0).ext.pending == Some((e.depth, e.rank))
    }
}

// =============================================================================
// The frontier
// =============================================================================

/// "Apply the next method of `class`", ordered for a max-heap. The derived
/// order compares the fields top to bottom.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Entry {
    /// `depth` for ([`SearchOrder::IdDfs`]) or `u32::MAX - depth`
    /// for ([`SearchOrder::Bfs`]).
    primary: u32,
    /// `N - idx`: the heuristic's first choice ranks highest.
    rank: u32,
    /// The oldest class first among equals. Cases of a class are sorted
    /// lexicographically and given ascending ClassIds.
    class: Reverse<ClassId>,
    /// Follows from `primary`, so it never decides the order.
    depth: u32,
}

impl Entry {
    fn new(order: SearchOrder, depth: u32, rank: u32, class: ClassId) -> Self {
        let primary = match order {
            SearchOrder::IdDfs => depth,
            SearchOrder::Bfs => u32::MAX - depth,
        };
        Entry {
            primary,
            rank,
            class: Reverse(class),
            depth,
        }
    }
}

// =============================================================================
// Workers
// =============================================================================

struct Task {
    class: ClassId,
    depth: u32,
    /// The children lie at or below `--bound`: none is examined.
    cut_children: bool,
    stored: Stored,
}

enum Outcome {
    Applied(Applied),
    /// No proof method can be applied anymore.
    Terminal,
    /// The deadline passed; the result is discarded.
    Aborted,
    /// Ranking or applying a method failed; the search fails with it.
    Failed(ProveError),
}

struct Expansion {
    class: ClassId,
    stored: Stored,
    outcome: Outcome,
    work: WorkStats,
}

/// Applies the class's next applicable method and prepares its cases.
/// With merging, `table` is the search's canonical table, which the cases
/// are prepared for.
fn expand_task(ctx: &ProofContext, task: Task, table: Option<&CanonicalTable>) -> Expansion {
    let start = Instant::now();
    let Task {
        class,
        depth,
        cut_children,
        mut stored,
    } = task;
    let mut work = WorkStats::default();
    let stop = search::deadline_reached;
    if stored.ranked.is_none() {
        match candidate_methods(&stored.sys, ctx, depth as usize) {
            Ok(ranked) => stored.ranked = Some(ranked),
            Err(e) => {
                return Expansion {
                    class,
                    stored,
                    outcome: Outcome::Failed(e),
                    work,
                };
            }
        }
    }
    let outcome = loop {
        if stop() {
            break Outcome::Aborted;
        }
        let ranked = stored.ranked.as_ref().expect("ranked above");
        let Some(method) = ranked.get(stored.cursor).cloned() else {
            break Outcome::Terminal;
        };
        let applied = match apply_method(
            ctx,
            &stored,
            &method,
            cut_children,
            table,
            table.is_some(),
            &mut work,
        ) {
            Ok(applied) => applied,
            Err(e) => break Outcome::Failed(e),
        };
        let Some(applied) = applied else {
            if stop() {
                break Outcome::Aborted;
            }
            stored.cursor += 1;
            continue;
        };
        stored.cursor += 1;
        break Outcome::Applied(applied);
    };
    work.busy = start.elapsed();
    Expansion {
        class,
        stored,
        outcome,
        work,
    }
}

// =============================================================================
// The engine
// =============================================================================

/// The counters at the start of an iteration, for its report.
struct IterationStart {
    at: Instant,
    classes: usize,
    applied: u64,
    execs: u64,
    canons: u64,
}

#[derive(Debug, Default)]
struct Stats {
    iterations: u32,
    batches: u64,
    tasks: u64,
    aborted: u64,
    applied: u64,
    leaves: u64,
    merges: u64,
    work: WorkStats,
    /// Wall time of the parallel phases.
    batch_wall: Duration,
}

struct Engine<'a> {
    ctx: &'a ProofContext,
    config: TopNConfig,
    graph: SearchGraph,
    /// Indexed by class; see [`Stored`].
    store: Vec<Option<Stored>>,
    /// The systems of pending classes of the canonical table that the store
    /// dropped: the table needs them to canonicalize a representative.
    retained: FastMap<ClassId, Box<System>>,
    frontier: BinaryHeap<Entry>,
    /// Live entries at or beyond the depth limit, for the next iteration.
    parked: Vec<Entry>,
    /// `--bound`: no class at this depth or deeper is expanded.
    bound: u32,
    deadline: Instant,
    /// With merging: the classes' canonical keys.
    table: CanonicalTable,
    stats: Stats,
    reporter: StatsReporter,
}

impl<'a> Engine<'a> {
    fn new(ctx: &'a ProofContext, config: TopNConfig, bound: u32, deadline: Instant) -> Self {
        Engine {
            ctx,
            config,
            graph: SearchGraph::new(),
            store: Vec::new(),
            retained: FastMap::default(),
            frontier: BinaryHeap::new(),
            parked: Vec::new(),
            bound,
            deadline,
            table: CanonicalTable::new(canon_mode_from_env()),
            stats: Stats::default(),
            reporter: StatsReporter::new("topn"),
        }
    }

    fn merge_counts(&self) -> MergeCounts {
        let (stored, stored_unkeyed) =
            search_graph::stored_counts(&self.graph, &self.store, self.config.merge);
        MergeCounts {
            applied: self.stats.applied,
            leaves: self.stats.leaves,
            merges: self.stats.merges,
            // Every merge happens when a case is created; whether a later
            // step goes through it is not tracked here.
            merges_used: self.stats.merges,
            stored,
            stored_unkeyed,
            retained: self.retained.len() as u64,
            ..MergeCounts::default()
        }
        .with_table(&self.table)
    }

    /// A `[search-stats]` progress line, if one is due.
    fn report_progress(&mut self) {
        let counts = self.merge_counts();
        self.reporter.tick(
            &self.ctx.lemma_name,
            &self.graph,
            &self.table,
            &counts,
            &self.stats.work,
        );
    }

    #[cfg(test)]
    fn add_class(&mut self, status: Status, depth: u32, stored: Option<Stored>) -> ClassId {
        let id = self.graph.add_class(status, depth);
        self.adopt(id, stored);
        id
    }

    /// Gives the new class `c` its slot in the store.
    fn adopt(&mut self, c: ClassId, stored: Option<Stored>) {
        assert_eq!(self.store.len(), c as usize, "classes are adopted in order");
        self.store.push(stored);
    }

    fn push_entry(&mut self, c: ClassId, rank: u32) {
        let depth = self.graph.class(c).min_depth;
        self.graph.class_mut(c).ext.pending = Some((depth, rank));
        self.frontier
            .push(Entry::new(self.config.order, depth, rank, c));
    }

    /// `N - idx` of `c`'s next method.
    fn next_rank(&self, c: ClassId) -> u32 {
        self.config.n - self.graph.class(c).ands.len() as u32
    }

    fn add_root(&mut self, sys: System) {
        let mut work = WorkStats::default();
        let child = if self.bound == 0 {
            Child::Cut
        } else {
            classify(
                self.ctx,
                sys,
                self.config.merge.then_some(&self.table),
                &mut work,
            )
        };
        self.stats.work.add(&work);
        let root = self.intern(child, 0);
        assert_eq!(root, ROOT, "the root is the first class");
    }

    /// The class of a prepared case reached at `depth`: a new one, or with
    /// merging an existing one reached again.
    fn intern(&mut self, child: Child, depth: u32) -> ClassId {
        let (c, interned) = self
            .graph
            .intern(child, depth, &mut self.table, &mut self.stats.work);
        match interned {
            Interned::Cut => self.adopt(c, None),
            Interned::Leaf => {
                self.stats.leaves += 1;
                self.adopt(c, None);
            }
            Interned::New(stored) => {
                self.adopt(c, Some(stored));
                self.push_entry(c, self.config.n);
            }
            Interned::Merged => {
                self.stats.merges += 1;
                self.lower_depth(c, depth);
                self.requeue_if_dormant(c);
            }
        }
        c
    }

    /// Records that `c` was reached at `depth`, and its descendants
    /// correspondingly shallower. A live entry moves to the new depth; the
    /// old one goes stale.
    fn lower_depth(&mut self, c: ClassId, depth: u32) {
        let mut work = vec![(c, depth)];
        while let Some((c, depth)) = work.pop() {
            if depth >= self.graph.class(c).min_depth {
                continue;
            }
            self.graph.class_mut(c).min_depth = depth;
            if let Some((_, rank)) = self.graph.class(c).ext.pending {
                self.push_entry(c, rank);
            }
            for and in &self.graph.class(c).ands {
                work.extend(and.cases.iter().map(|&(_, child)| (child, depth + 1)));
            }
        }
    }

    /// Gives `c` an entry again if it lost it as irrelevant: a new edge may
    /// have made it relevant.
    fn requeue_if_dormant(&mut self, c: ClassId) {
        let class = self.graph.class(c);
        if class.status == Status::Open
            && !class.closed
            && class.ext.pending.is_none()
            && self.store[c as usize].is_some()
        {
            self.push_entry(c, self.next_rank(c));
        }
    }

    /// Phase 1: pops up to B live, relevant entries below `limit`.
    fn select_batch(&mut self, limit: u32) -> Vec<Task> {
        let mut tasks = Vec::new();
        while tasks.len() < self.config.batch {
            let Some(e) = self.frontier.pop() else {
                break;
            };
            if !self.graph.entry_is_live(&e) {
                continue;
            }
            let c = e.class.0;
            if self.graph.class(c).status != Status::Open {
                self.graph.class_mut(c).ext.pending = None;
                if let Some(stored) = self.store[c as usize].take() {
                    self.set_aside(c, stored);
                }
                continue;
            }
            if !self.graph.is_relevant(c) {
                // The system stays: a later edge can make the class relevant
                // again (`requeue_if_dormant`).
                self.graph.class_mut(c).ext.pending = None;
                continue;
            }
            if e.depth >= limit {
                self.parked.push(e);
                continue;
            }
            self.graph.class_mut(c).ext.pending = None;
            let stored = self.store[c as usize]
                .take()
                .expect("a class with a live entry keeps its system");
            tasks.push(Task {
                class: c,
                depth: e.depth,
                cut_children: e.depth + 1 >= self.bound,
                stored,
            });
        }
        tasks
    }

    /// Phase 2: expands the tasks concurrently; the results keep the tasks'
    /// order.
    fn expand_batch(&mut self, env: &WorkerEnv, tasks: Vec<Task>) -> Vec<Expansion> {
        let start = Instant::now();
        let count = tasks.len();
        let results = {
            let ctx = self.ctx;
            // Workers read the table while this thread waits for them.
            let table = self.config.merge.then_some(&self.table);
            let expand = |task: Task| -> Expansion {
                let avoid_next = avoid_fresh_state(&task.stored.sys);
                env.run(ctx, avoid_next, |worker_ctx| {
                    expand_task(worker_ctx, task, table)
                })
            };
            if count == 1 {
                tasks.into_iter().map(&expand).collect::<Vec<_>>()
            } else {
                tasks.into_par_iter().map(&expand).collect::<Vec<_>>()
            }
        };
        self.stats.batches += 1;
        self.stats.tasks += count as u64;
        self.stats.batch_wall += start.elapsed();
        results
    }

    /// `stored`, the system of `c`, which the search is done with: set aside
    /// if `c` is a pending class of the canonical table, which needs it to
    /// canonicalize the class's representative when a system with its
    /// invariant comes; dropped otherwise.
    fn set_aside(&mut self, c: ClassId, stored: Stored) {
        if self.table.is_pending(c) {
            self.retained.insert(c, stored.sys);
        }
    }

    /// Drops the set-aside systems of the classes the canonical table keyed
    /// since: it needs them no more.
    fn forget_keyed(&mut self) {
        for c in self.table.take_keyed() {
            self.retained.remove(&c);
        }
    }

    /// Phase 3: adds one expansion to the graph.
    fn integrate(&mut self, expansion: Expansion) -> Result<(), ProveError> {
        let Expansion {
            class: c,
            stored,
            outcome,
            work,
        } = expansion;
        self.stats.work.add(&work);
        match outcome {
            Outcome::Failed(e) => return Err(e),
            Outcome::Aborted => {
                // The search stops after this batch; keep the system anyway
                // so the graph stays consistent.
                self.stats.aborted += 1;
                self.store[c as usize] = Some(stored);
            }
            Outcome::Terminal => {
                self.graph.close(c);
                self.set_aside(c, stored);
            }
            Outcome::Applied(Applied {
                method,
                method_key,
                cases,
            }) => {
                self.stats.applied += 1;
                let a = self.graph.add_and(c, method, method_key);
                let depth = self.graph.class(c).min_depth + 1;
                for (name, child) in cases {
                    let child = self.intern(child, depth);
                    self.graph.add_case(c, a, name, child);
                }
                let applied = self.graph.class(c).ands.len() as u32;
                let candidates = stored.ranked.as_ref().map_or(0, Vec::len);
                if applied < self.config.n && stored.cursor < candidates {
                    self.store[c as usize] = Some(stored);
                    self.push_entry(c, self.next_rank(c));
                } else {
                    self.graph.class_mut(c).closed = true;
                    self.set_aside(c, stored);
                }
                self.graph.propagate(vec![(c, a)]);
                // `closed` may also settle `c` without any method changing.
                let parents = self.graph.recheck(c);
                self.graph.propagate(parents);
            }
        }
        Ok(())
    }

    fn end(&self) -> Option<End> {
        if self.graph.class(ROOT).status != Status::Open {
            Some(End::Settled)
        } else if Instant::now() >= self.deadline {
            Some(End::Deadline)
        } else {
            None
        }
    }

    fn iteration_start(&self) -> IterationStart {
        IterationStart {
            at: Instant::now(),
            classes: self.graph.classes.len(),
            applied: self.stats.applied,
            execs: self.stats.work.execs,
            canons: self.stats.work.canons,
        }
    }

    /// `TAM_RS_TOPN_STATS`: one line per iteration, with what it added and the
    /// entries it parked at the limit.
    fn report_iteration(&self, limit: u32, start: &IterationStart) {
        if !tamarin_utils::env_gate!("TAM_RS_TOPN_STATS") {
            return;
        }
        eprintln!(
            "[topn-iter] lemma={} iteration={} limit={} classes=+{} applied=+{} execs=+{} \
             canons=+{} parked={} root={:?} secs={:.3}",
            self.ctx.lemma_name,
            self.stats.iterations,
            limit,
            self.graph.classes.len() - start.classes,
            self.stats.applied - start.applied,
            self.stats.work.execs - start.execs,
            self.stats.work.canons - start.canons,
            self.parked.len(),
            self.graph.class(ROOT).status,
            start.at.elapsed().as_secs_f64(),
        );
    }

    /// Fails with the first error in batch order, so which error a search
    /// reports does not depend on which worker met one first.
    fn search(&mut self, env: &WorkerEnv) -> Result<End, ProveError> {
        let mut limit = match self.config.order {
            SearchOrder::IdDfs => FIRST_LIMIT.min(self.bound),
            SearchOrder::Bfs => self.bound,
        };
        loop {
            self.stats.iterations += 1;
            let start = self.iteration_start();
            loop {
                if let Some(end) = self.end() {
                    self.report_iteration(limit, &start);
                    return Ok(end);
                }
                let tasks = self.select_batch(limit);
                if tasks.is_empty() {
                    break;
                }
                let mut results = self.expand_batch(env, tasks);
                // What no worker could key, before any case is placed. A
                // representative may be a class this batch expanded, whose
                // system is in its expansion until integrated.
                let mut expanded: FastMap<ClassId, &System> = FastMap::default();
                let mut items = Vec::new();
                for expansion in &mut results {
                    expanded.insert(expansion.class, &expansion.stored.sys);
                    if let Outcome::Applied(applied) = &mut expansion.outcome {
                        items.extend(Child::prepared_mut(&mut applied.cases));
                    }
                }
                self.table.precompute(
                    items,
                    |r| {
                        kept_system(&self.store, &self.retained, r)
                            .or_else(|| expanded.get(&r).copied())
                            .map(system_to_term)
                    },
                    &mut self.stats.work,
                );
                for result in results {
                    self.integrate(result)?;
                }
                self.forget_keyed();
                self.report_progress();
            }
            self.report_iteration(limit, &start);
            if let Some(end) = self.end() {
                return Ok(end);
            }
            if self.parked.is_empty() {
                return Ok(End::Exhausted);
            }
            // Parked entries lie below `bound`, so `limit < bound` here.
            limit = limit.saturating_mul(2).min(self.bound);
            let parked = std::mem::take(&mut self.parked);
            self.frontier.extend(parked);
        }
    }
}

// =============================================================================
// Entry point
// =============================================================================

/// Runs the top-N search on `initial` and returns the proof tree, in the
/// shape the greedy driver returns it. Called by `search::run_proof_search`
/// when the lemma's context carries a [`TopNConfig`].
pub fn run(
    ctx: &ProofContext,
    initial: System,
    proof_bound: usize,
    config: TopNConfig,
    deadline: Instant,
) -> Result<ProofNode, ProveError> {
    let start = Instant::now();
    let bound = u32::try_from(proof_bound)
        .unwrap_or(u32::MAX)
        .min(DEPTH_CAP);
    let mut engine = Engine::new(ctx, config, bound, deadline);
    engine.add_root(initial.clone());
    let end = engine.search(&WorkerEnv::capture(Some(deadline)))?;
    let searched = start.elapsed();
    engine.reporter.finish(
        &ctx.lemma_name,
        &engine.graph,
        &engine.table,
        &engine.merge_counts(),
        &engine.stats.work,
        end,
    );

    // Materializing runs methods again; the deadline the search may have hit
    // must not cut that short.
    let no_deadline = search::capture_search_tls(None);
    let materialize_start = Instant::now();
    let kept = KeptSystems::new(
        std::mem::take(&mut engine.store),
        std::mem::take(&mut engine.retained),
    );
    let materializer = Materializer {
        graph: &engine.graph,
        table: &engine.table,
        kept: &kept,
        merge: config.merge,
        bound,
        end,
        env: WorkerEnv::capture(None),
        label: "top-N",
    };
    let root = {
        let _tls = search::SearchTlsGuard::install(no_deadline);
        materializer.root(ctx, initial)?
    };
    let materialized = materialize_start.elapsed();
    (engine.store, engine.retained) = kept.into_inner();

    if tamarin_utils::env_gate!("TAM_RS_TOPN_STATS") {
        print_stats(ctx, &config, &engine, end, &root, searched, materialized);
    }
    Ok(root)
}

fn print_stats(
    ctx: &ProofContext,
    config: &TopNConfig,
    engine: &Engine<'_>,
    end: End,
    root: &ProofNode,
    searched: Duration,
    materialized: Duration,
) {
    let s = &engine.stats;
    let threads = rayon::current_num_threads().min(config.batch).max(1);
    let capacity = s.batch_wall.as_secs_f64() * threads as f64;
    let efficiency = if capacity > 0.0 {
        s.work.busy.as_secs_f64() / capacity
    } else {
        0.0
    };
    // Classes reached along two or more edges: what merging saved a second
    // expansion of.
    let shared = engine
        .graph
        .classes
        .iter()
        .filter(|c| c.parents.len() > 1)
        .count();
    eprintln!(
        "[topn] lemma={} n={} order={:?} merge={} batch={} end={:?} root={:?} \
         classes={} leaves={} merges={} shared={} execs={} applied={} canons={} canon_s={:.3} \
         iterations={} batches={} tasks={} aborted={} efficiency={:.2} \
         search_s={:.3} materialize_s={:.3}",
        ctx.lemma_name,
        config.n,
        config.order,
        config.merge,
        config.batch,
        end,
        root.status,
        engine.graph.classes.len(),
        s.leaves,
        s.merges,
        shared,
        s.work.execs,
        s.applied,
        s.work.canons,
        s.work.canon_time.as_secs_f64(),
        s.iterations,
        s.batches,
        s.tasks,
        s.aborted,
        efficiency,
        searched.as_secs_f64(),
        materialized.as_secs_f64(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::solver::proof_method::ProofMethod;
    use crate::constraint::solver::search::NodeStatus;

    // --- The frontier (no maude) -----------------------------------------

    #[test]
    fn iddfs_takes_the_deepest_entry_then_the_best_ranked_then_the_oldest() {
        let order = SearchOrder::IdDfs;
        let mut heap = BinaryHeap::from([
            Entry::new(order, 2, 3, 5),
            Entry::new(order, 3, 1, 9),
            Entry::new(order, 3, 3, 7),
            Entry::new(order, 3, 3, 4),
        ]);
        let popped: Vec<(u32, u32, ClassId)> =
            std::iter::from_fn(|| heap.pop().map(|e| (e.primary, e.rank, e.class.0))).collect();
        assert_eq!(popped, vec![(3, 3, 4), (3, 3, 7), (3, 1, 9), (2, 3, 5)]);
    }

    #[test]
    fn bfs_takes_the_shallowest_entry_first() {
        let order = SearchOrder::Bfs;
        let mut heap = BinaryHeap::from([
            Entry::new(order, 3, 3, 1),
            Entry::new(order, 1, 1, 2),
            Entry::new(order, 1, 3, 3),
        ]);
        let popped: Vec<ClassId> = std::iter::from_fn(|| heap.pop().map(|e| e.class.0)).collect();
        assert_eq!(popped, vec![3, 2, 1]);
    }

    #[test]
    fn only_the_latest_entry_of_a_class_is_live() {
        let mut g = SearchGraph::new();
        let c = g.add_class(Status::Open, 0);
        let old = Entry::new(SearchOrder::IdDfs, 5, 3, c);
        let new = Entry::new(SearchOrder::IdDfs, 2, 3, c);
        g.class_mut(c).ext.pending = Some((2, 3));
        assert!(!g.entry_is_live(&old));
        assert!(g.entry_is_live(&new));
        g.class_mut(c).ext.pending = None;
        assert!(!g.entry_is_live(&new));
    }

    // --- The frontier's bookkeeping (maude only for the context) -----------

    /// A class parked at the depth limit that a merge reaches shallower runs
    /// within the iteration; the parked entry goes stale.
    #[test]
    fn a_shallower_path_brings_a_parked_class_back_within_the_iteration() {
        let Some(ctx) = ctx() else {
            return;
        };
        let deadline = Instant::now() + Duration::from_secs(3600);
        let mut engine = Engine::new(&ctx, config(2, false, 1), DEPTH_CAP, deadline);
        // The first class is the root, so it is always relevant.
        let c = engine.add_class(Status::Open, 5, Some(Stored::new(System::empty(), None)));
        engine.push_entry(c, 2);
        assert!(
            engine.select_batch(4).is_empty(),
            "depth 5 waits for a higher limit"
        );
        assert_eq!(engine.parked.len(), 1);
        engine.lower_depth(c, 2);
        let tasks = engine.select_batch(4);
        assert_eq!(
            tasks.iter().map(|t| (t.class, t.depth)).collect::<Vec<_>>(),
            vec![(c, 2)]
        );
        let parked = std::mem::take(&mut engine.parked);
        engine.frontier.extend(parked);
        assert!(
            engine.select_batch(8).is_empty(),
            "the parked entry is stale"
        );
    }

    // --- Against the greedy driver (maude) -------------------------------

    use crate::constraint::solver::search::run_proof_search;
    use crate::constraint::solver::search_graph::test_support::{
        ctx, empty_disjunction, two_branch_disjunction, two_true_formulas, unproducible_action,
    };
    use crate::pretty_theory::pretty_proof_body;

    fn config(n: u32, merge: bool, batch: usize) -> TopNConfig {
        TopNConfig {
            n,
            order: SearchOrder::IdDfs,
            merge,
            batch,
        }
    }

    /// With one method per system the search is the greedy one, whatever the
    /// batch size and with or without merging. With three, a verdict the
    /// greedy search reached stays; one it left open may be decided.
    fn assert_agrees_with_greedy(make: impl Fn() -> System, bound: usize, merge_too: bool) {
        let Some(mut ctx) = ctx() else {
            return;
        };
        let greedy = run_proof_search(&ctx, make(), bound).expect("proof search");
        let expected = pretty_proof_body(&greedy);
        let merges: &[bool] = if merge_too && crate::bliss_proc::bliss_available() {
            &[false, true]
        } else {
            &[false]
        };
        for &merge in merges {
            for batch in [1, 4] {
                ctx.top_n = Some(config(1, merge, batch));
                let top1 = run_proof_search(&ctx, make(), bound).expect("proof search");
                assert_eq!(
                    pretty_proof_body(&top1),
                    expected,
                    "n=1 merge={merge} batch={batch}"
                );
                assert_eq!(top1.status, greedy.status);
                ctx.top_n = Some(config(3, merge, batch));
                let top3 = run_proof_search(&ctx, make(), bound).expect("proof search");
                if matches!(
                    greedy.status,
                    NodeStatus::Solved | NodeStatus::Contradictory
                ) {
                    assert_eq!(
                        top3.status, greedy.status,
                        "n=3 merge={merge} batch={batch}"
                    );
                }
            }
        }
    }

    #[test]
    fn an_empty_disjunction_closes_like_the_greedy_search() {
        assert_agrees_with_greedy(empty_disjunction, 5, true);
    }

    #[test]
    fn a_two_branch_disjunction_finds_the_trace_like_the_greedy_search() {
        assert_agrees_with_greedy(two_branch_disjunction, 10, true);
    }

    #[test]
    fn simplify_then_solved_like_the_greedy_search() {
        assert_agrees_with_greedy(two_true_formulas, 5, true);
    }

    #[test]
    fn the_proof_bound_cuts_like_the_greedy_search() {
        assert_agrees_with_greedy(unproducible_action, 1, false);
    }

    /// With `--bound 1` the greedy search's `simplify` ends in `sorry`; the
    /// second-ranked method closes the system within the bound.
    #[test]
    fn a_lower_ranked_method_closes_what_the_first_leaves_open() {
        let Some(mut ctx) = ctx() else {
            return;
        };
        let greedy = run_proof_search(&ctx, unproducible_action(), 1).expect("proof search");
        assert_eq!(greedy.status, NodeStatus::Sorry);
        for batch in [1, 4] {
            ctx.top_n = Some(config(3, false, batch));
            let top3 = run_proof_search(&ctx, unproducible_action(), 1).expect("proof search");
            assert_eq!(top3.status, NodeStatus::Contradictory, "batch={batch}");
            assert!(
                matches!(top3.method, ProofMethod::SolveGoal(_)),
                "the proof uses the second-ranked method, got {:?}",
                top3.method
            );
            assert!(top3.children.is_empty(), "the goal has no producing rule");
        }
    }
}
