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
//! The limit applies to a path's cost, not its depth: each step costs one,
//! plus `idx * alt_cost` for the heuristic's `idx`-th method
//! ([`TopNConfig::alt_cost`], default 0, where the cost is the depth). A
//! positive `alt_cost` lets the heuristic's first choices run deeper than
//! alternatives within one iteration.
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
//! - `TAM_RS_TOPN_BATCH=B`: entries expanded per step, default the rayon
//!   thread count; pin it for reproducible runs;
//! - `TAM_RS_TOPN_ALT_COST=W`: what an alternative method adds to a path's
//!   cost against the depth limit, times its rank; default 0;
//! - `TAM_RS_TOPN_STATS` (presence): one summary line per search on stderr,
//!   and one per iteration.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use rayon::prelude::*;

use crate::constraint::solver::context::ProofContext;
use crate::constraint::solver::reduction::avoid_fresh_state;
use crate::constraint::solver::search::{self, candidate_methods, ProofNode};
use crate::constraint::solver::search_graph::{
    self, apply_method, classify, Applied, Child, ClassId, End, Interned, Materializer, Status,
    Stored, WorkStats, WorkerEnv, DEPTH_CAP, ROOT,
};
use crate::constraint::system::System;

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
    /// What a method of heuristic rank `idx` adds to a path's cost beyond its
    /// one level: `idx * alt_cost`. [`SearchOrder::IdDfs`]'s depth limit
    /// applies to this cost, so with `alt_cost > 0` the heuristic's first
    /// choices reach deeper than alternatives within one iteration; `0` makes
    /// the cost the depth.
    pub alt_cost: u32,
}

fn parse_positive(var: &str, value: &str) -> usize {
    value
        .parse::<usize>()
        .ok()
        .filter(|&n| n >= 1)
        .unwrap_or_else(|| panic!("{var}={value:?}: expected a positive integer"))
}

fn parse_u32(var: &str, value: &str) -> u32 {
    value
        .parse::<u32>()
        .unwrap_or_else(|_| panic!("{var}={value:?}: expected a non-negative integer"))
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
        let alt_cost = match std::env::var("TAM_RS_TOPN_ALT_COST") {
            Err(std::env::VarError::NotPresent) => 0,
            Ok(v) => parse_u32("TAM_RS_TOPN_ALT_COST", &v),
            Err(e) => panic!("TAM_RS_TOPN_ALT_COST: {e}"),
        };
        Some(TopNConfig {
            n,
            order,
            merge: tamarin_utils::env_gate!("TAM_RS_MERGE"),
            batch,
            alt_cost,
        })
    })
}

/// [`top_n_from_env`] for one lemma of a theory, checked against what
/// merging needs up front rather than at the first canonicalization.
/// Theories with processes need nothing extra: the prover translates them
/// before building the proof context, whose color table therefore covers
/// the generated rules' actions.
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
    /// The cheapest cost the class was reached at ([`TopNConfig::alt_cost`]);
    /// tracked apart from `min_depth`, which may come from another path.
    min_cost: u32,
    /// The class's one live frontier entry: `None` while it is being expanded
    /// or has no entry. Any other entry of the class is stale.
    pending: Option<Pending>,
}

type SearchGraph = search_graph::SearchGraph<TopNExt>;

impl SearchGraph {
    /// The class's [`Pending`] if `e`, popped from the frontier, is its live
    /// entry; `None` for a stale one.
    fn live(&self, e: &Entry) -> Option<Pending> {
        let pending = self
            .class(e.class.0)
            .ext
            .pending
            .filter(|p| p.entry == *e)?;
        assert!(
            !pending.parked,
            "top-N search: a live entry is in the frontier and parked at once (class #{})",
            e.class.0
        );
        Some(pending)
    }
}

// =============================================================================
// The frontier
// =============================================================================

/// "Apply the next method of `class`". Its fields are exactly the frontier
/// order, compared top to bottom by the derived `Ord` of a max-heap. The
/// cost, which decides only whether the entry runs in this iteration, is in
/// the class's [`Pending`].
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
        }
    }
}

/// A class's live entry, with what the order leaves out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Pending {
    entry: Entry,
    depth: u32,
    /// The class's cheapest cost plus `idx * alt_cost`: the entry runs in the
    /// first iteration whose limit exceeds it.
    cost: u32,
    /// Whether the entry waits in [`Engine::parked`] rather than in the
    /// frontier.
    parked: bool,
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
}

struct Expansion {
    class: ClassId,
    stored: Stored,
    outcome: Outcome,
    work: WorkStats,
}

/// Applies the class's next applicable method and prepares its cases.
fn expand_task(ctx: &ProofContext, task: Task, merge: bool) -> Expansion {
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
        stored.ranked = Some(candidate_methods(&stored.sys, ctx, depth as usize));
    }
    let outcome = loop {
        if stop() {
            break Outcome::Aborted;
        }
        let ranked = stored.ranked.as_ref().expect("ranked above");
        let Some(method) = ranked.get(stored.cursor).cloned() else {
            break Outcome::Terminal;
        };
        let Some(applied) = apply_method(ctx, &stored, &method, cut_children, merge, &mut work)
        else {
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
    frontier: BinaryHeap<Entry>,
    /// Classes whose live entry waits for a higher limit
    /// ([`Pending::parked`]). A class may appear twice; it gets its entry
    /// back once.
    parked: Vec<ClassId>,
    /// `--bound`: no class at this depth or deeper is expanded.
    bound: u32,
    deadline: Instant,
    stats: Stats,
}

impl<'a> Engine<'a> {
    fn new(ctx: &'a ProofContext, config: TopNConfig, bound: u32, deadline: Instant) -> Self {
        Engine {
            ctx,
            config,
            graph: SearchGraph::new(),
            store: Vec::new(),
            frontier: BinaryHeap::new(),
            parked: Vec::new(),
            bound,
            deadline,
            stats: Stats::default(),
        }
    }

    #[cfg(test)]
    fn add_class(
        &mut self,
        status: Status,
        depth: u32,
        cost: u32,
        stored: Option<Stored>,
    ) -> ClassId {
        let id = self.graph.add_class(status, depth);
        self.adopt(id, cost, stored);
        id
    }

    /// Gives the new class `c` its cost and its slot in the store.
    fn adopt(&mut self, c: ClassId, cost: u32, stored: Option<Stored>) {
        assert_eq!(self.store.len(), c as usize, "classes are adopted in order");
        self.graph.class_mut(c).ext.min_cost = cost;
        self.store.push(stored);
    }

    /// `idx * alt_cost`: what applying the heuristic's `idx`-th method adds.
    fn alt_cost(&self, idx: u32) -> u32 {
        idx.saturating_mul(self.config.alt_cost)
    }

    /// Makes "apply `c`'s method of rank `rank`" the class's live entry, at
    /// its current depth and cost. The frontier gets the entry unless it
    /// already holds this one: then only the cost changed, which is not part
    /// of the order.
    fn push_entry(&mut self, c: ClassId, rank: u32) {
        let class = self.graph.class(c);
        let depth = class.min_depth;
        let cost = class
            .ext
            .min_cost
            .saturating_add(self.alt_cost(self.config.n - rank));
        let entry = Entry::new(self.config.order, depth, rank, c);
        let queued = class
            .ext
            .pending
            .is_some_and(|p| p.entry == entry && !p.parked);
        self.graph.class_mut(c).ext.pending = Some(Pending {
            entry,
            depth,
            cost,
            parked: false,
        });
        if !queued {
            self.frontier.push(entry);
        }
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
            classify(self.ctx, sys, self.config.merge, &mut work)
        };
        self.stats.work.canons += work.canons;
        self.stats.work.canon_time += work.canon_time;
        let root = self.intern(child, 0, 0);
        assert_eq!(root, ROOT, "the root is the first class");
    }

    /// The class of a prepared case reached at `depth` and `cost`: a new one,
    /// or with merging an existing one reached again.
    fn intern(&mut self, child: Child, depth: u32, cost: u32) -> ClassId {
        let (c, interned) = self.graph.intern(child, depth);
        match interned {
            Interned::Cut => self.adopt(c, cost, None),
            Interned::Leaf => {
                self.stats.leaves += 1;
                self.adopt(c, cost, None);
            }
            Interned::New(stored) => {
                self.adopt(c, cost, Some(stored));
                self.push_entry(c, self.config.n);
            }
            Interned::Merged => {
                self.stats.merges += 1;
                self.lower(c, depth, cost);
                self.requeue_if_dormant(c);
            }
        }
        c
    }

    /// Records that `c` was reached at `depth` and `cost`, and its descendants
    /// correspondingly: each minimum is lowered on its own. A live entry
    /// moves to the new values: a shallower one replaces it in the frontier,
    /// leaving the old one stale; a cheaper one keeps its place, or comes
    /// back from the parked ones, whose limit it may now be under.
    fn lower(&mut self, c: ClassId, depth: u32, cost: u32) {
        let mut work = vec![(c, depth, cost)];
        while let Some((c, depth, cost)) = work.pop() {
            let class = self.graph.class(c);
            if depth >= class.min_depth && cost >= class.ext.min_cost {
                continue;
            }
            let (depth, cost) = (depth.min(class.min_depth), cost.min(class.ext.min_cost));
            let class = self.graph.class_mut(c);
            class.min_depth = depth;
            class.ext.min_cost = cost;
            if let Some(pending) = class.ext.pending {
                self.push_entry(c, pending.entry.rank);
            }
            for (a, and) in self.graph.class(c).ands.iter().enumerate() {
                let step = cost
                    .saturating_add(self.alt_cost(a as u32))
                    .saturating_add(1);
                work.extend(and.cases.iter().map(|&(_, child)| (child, depth + 1, step)));
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

    /// Moves the parked entries back into the frontier for the next
    /// iteration. A class listed twice gets its entry back once, and one whose
    /// entry was replaced meanwhile ([`Self::push_entry`]) gets none.
    fn unpark(&mut self) {
        for c in std::mem::take(&mut self.parked) {
            let Some(pending) = self
                .graph
                .class_mut(c)
                .ext
                .pending
                .as_mut()
                .filter(|p| p.parked)
            else {
                continue;
            };
            pending.parked = false;
            let entry = pending.entry;
            self.frontier.push(entry);
        }
    }

    /// Phase 1: pops up to B live, relevant entries below `limit`.
    fn select_batch(&mut self, limit: u32) -> Vec<Task> {
        let mut tasks = Vec::new();
        while tasks.len() < self.config.batch {
            let Some(e) = self.frontier.pop() else {
                break;
            };
            let Some(pending) = self.graph.live(&e) else {
                continue;
            };
            let c = e.class.0;
            if self.graph.class(c).status != Status::Open {
                self.graph.class_mut(c).ext.pending = None;
                self.store[c as usize] = None;
                continue;
            }
            if !self.graph.is_relevant(c) {
                // The system stays: a later edge can make the class relevant
                // again (`requeue_if_dormant`).
                self.graph.class_mut(c).ext.pending = None;
                continue;
            }
            // Nothing is parked at the largest limit: a saturated cost must
            // not keep an entry parked forever.
            if pending.cost >= limit && limit < u32::MAX {
                self.graph.class_mut(c).ext.pending = Some(Pending {
                    parked: true,
                    ..pending
                });
                self.parked.push(c);
                continue;
            }
            self.graph.class_mut(c).ext.pending = None;
            let stored = self.store[c as usize]
                .take()
                .expect("a class with a live entry keeps its system");
            tasks.push(Task {
                class: c,
                depth: pending.depth,
                cut_children: pending.depth + 1 >= self.bound,
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
            let merge = self.config.merge;
            let expand = |task: Task| -> Expansion {
                let avoid_next = avoid_fresh_state(&task.stored.sys);
                env.run(ctx, avoid_next, |worker_ctx| {
                    expand_task(worker_ctx, task, merge)
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

    /// Phase 3: adds one expansion to the graph.
    fn integrate(&mut self, expansion: Expansion) {
        let Expansion {
            class: c,
            stored,
            outcome,
            work,
        } = expansion;
        self.stats.work.add(&work);
        match outcome {
            Outcome::Aborted => {
                // The search stops after this batch; keep the system anyway
                // so the graph stays consistent.
                self.stats.aborted += 1;
                self.store[c as usize] = Some(stored);
            }
            Outcome::Terminal => self.graph.close(c),
            Outcome::Applied(Applied {
                method,
                method_fp,
                cases,
            }) => {
                self.stats.applied += 1;
                let a = self.graph.add_and(c, method, method_fp);
                let class = self.graph.class(c);
                let depth = class.min_depth + 1;
                let cost = class
                    .ext
                    .min_cost
                    .saturating_add(self.alt_cost(u32::from(a)))
                    .saturating_add(1);
                for (name, child) in cases {
                    let child = self.intern(child, depth, cost);
                    self.graph.add_case(c, a, name, child);
                }
                let applied = self.graph.class(c).ands.len() as u32;
                let candidates = stored.ranked.as_ref().map_or(0, Vec::len);
                if applied < self.config.n && stored.cursor < candidates {
                    self.store[c as usize] = Some(stored);
                    self.push_entry(c, self.next_rank(c));
                } else {
                    self.graph.class_mut(c).closed = true;
                }
                self.graph.propagate(vec![(c, a)]);
                // `closed` may also settle `c` without any method changing.
                let parents = self.graph.recheck(c);
                self.graph.propagate(parents);
            }
        }
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

    fn search(&mut self, env: &WorkerEnv) -> End {
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
                    return end;
                }
                let tasks = self.select_batch(limit);
                if tasks.is_empty() {
                    break;
                }
                let results = self.expand_batch(env, tasks);
                for result in results {
                    self.integrate(result);
                }
            }
            self.report_iteration(limit, &start);
            if let Some(end) = self.end() {
                return end;
            }
            if self.parked.is_empty() {
                return End::Exhausted;
            }
            // The limit applies to the cost and `bound` to the depth: with
            // `alt_cost`, an entry at a depth below `bound` can cost more than
            // `bound`, so the limit keeps doubling past it.
            limit = limit.saturating_mul(2);
            self.unpark();
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
) -> ProofNode {
    let start = Instant::now();
    let bound = u32::try_from(proof_bound)
        .unwrap_or(u32::MAX)
        .min(DEPTH_CAP);
    let mut engine = Engine::new(ctx, config, bound, deadline);
    engine.add_root(initial.clone());
    let end = engine.search(&WorkerEnv::capture(Some(deadline)));
    let searched = start.elapsed();

    // Materializing runs methods again; the deadline the search may have hit
    // must not cut that short.
    let previous_deadline = search::replace_deadline(None);
    let materialize_start = Instant::now();
    let materializer = Materializer {
        graph: &engine.graph,
        merge: config.merge,
        bound,
        end,
        env: WorkerEnv::capture(None),
        label: "top-N",
    };
    let root = materializer.root(ctx, initial);
    let materialized = materialize_start.elapsed();
    search::replace_deadline(previous_deadline);

    if tamarin_utils::env_gate!("TAM_RS_TOPN_STATS") {
        print_stats(ctx, &config, &engine, end, &root, searched, materialized);
    }
    root
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
        "[topn] lemma={} n={} order={:?} merge={} batch={} alt_cost={} end={:?} root={:?} \
         classes={} leaves={} merges={} shared={} execs={} applied={} canons={} canon_s={:.3} \
         iterations={} batches={} tasks={} aborted={} efficiency={:.2} \
         search_s={:.3} materialize_s={:.3}",
        ctx.lemma_name,
        config.n,
        config.order,
        config.merge,
        config.batch,
        config.alt_cost,
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
        g.class_mut(c).ext.pending = Some(Pending {
            entry: new,
            depth: 2,
            cost: 2,
            parked: false,
        });
        assert!(g.live(&old).is_none());
        assert!(g.live(&new).is_some());
        g.class_mut(c).ext.pending = None;
        assert!(g.live(&new).is_none());
    }

    // --- The frontier's bookkeeping (maude only for the context) -----------

    /// An engine over hand-made classes: the first one is the root, so it is
    /// always relevant.
    fn engine(ctx: &ProofContext, alt_cost: u32) -> Engine<'_> {
        let config = TopNConfig {
            alt_cost,
            ..config(2, false, 1)
        };
        Engine::new(
            ctx,
            config,
            DEPTH_CAP,
            Instant::now() + Duration::from_secs(3600),
        )
    }

    fn with_system() -> Option<Stored> {
        Some(Stored::new(System::empty(), None))
    }

    #[test]
    fn a_cheaper_path_changes_the_cost_but_not_the_frontier() {
        let Some(ctx) = ctx() else {
            return;
        };
        let mut engine = engine(&ctx, 4);
        let c = engine.add_class(Status::Open, 0, 5, None);
        engine.push_entry(c, 2);
        engine.lower(c, 0, 1);
        assert_eq!(engine.frontier.len(), 1, "the key did not change");
        assert_eq!(engine.graph.class(c).ext.pending.map(|p| p.cost), Some(1));
    }

    #[test]
    fn a_cheaper_path_brings_a_parked_entry_back_within_the_iteration() {
        let Some(ctx) = ctx() else {
            return;
        };
        let mut engine = engine(&ctx, 4);
        let c = engine.add_class(Status::Open, 0, 5, with_system());
        engine.push_entry(c, 2);
        assert!(
            engine.select_batch(4).is_empty(),
            "cost 5 waits for a higher limit"
        );
        assert_eq!(engine.parked, vec![c]);
        engine.lower(c, 0, 2);
        let tasks = engine.select_batch(4);
        assert_eq!(tasks.iter().map(|t| t.class).collect::<Vec<_>>(), vec![c]);
        engine.unpark();
        assert!(
            engine.frontier.is_empty(),
            "the expanded entry is not parked any more"
        );
    }

    #[test]
    fn a_class_parked_twice_gets_its_entry_back_once() {
        let Some(ctx) = ctx() else {
            return;
        };
        let mut engine = engine(&ctx, 4);
        let c = engine.add_class(Status::Open, 0, 6, with_system());
        engine.push_entry(c, 2);
        assert!(engine.select_batch(4).is_empty());
        engine.lower(c, 0, 5);
        assert!(engine.select_batch(4).is_empty(), "still above the limit");
        assert_eq!(engine.parked, vec![c, c]);
        engine.unpark();
        assert_eq!(engine.frontier.len(), 1);
        assert_eq!(engine.select_batch(8).len(), 1);
        assert!(engine.frontier.is_empty());
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
            alt_cost: 0,
        }
    }

    /// With one method per system the search is the greedy one, whatever the
    /// batch size and with or without merging. With three, a verdict the
    /// greedy search reached stays; one it left open may be decided.
    fn assert_agrees_with_greedy(make: impl Fn() -> System, bound: usize, merge_too: bool) {
        let Some(mut ctx) = ctx() else {
            return;
        };
        let greedy = run_proof_search(&ctx, make(), bound);
        let expected = pretty_proof_body(&greedy);
        let merges: &[bool] = if merge_too && crate::bliss_proc::bliss_available() {
            &[false, true]
        } else {
            &[false]
        };
        for &merge in merges {
            for batch in [1, 4] {
                for alt_cost in [0, 8] {
                    // One method per system leaves no alternative to weight.
                    ctx.top_n = Some(TopNConfig {
                        alt_cost,
                        ..config(1, merge, batch)
                    });
                    let top1 = run_proof_search(&ctx, make(), bound);
                    assert_eq!(
                        pretty_proof_body(&top1),
                        expected,
                        "n=1 merge={merge} batch={batch} alt_cost={alt_cost}"
                    );
                    assert_eq!(top1.status, greedy.status);
                }
                ctx.top_n = Some(config(3, merge, batch));
                let top3 = run_proof_search(&ctx, make(), bound);
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
    /// second-ranked method closes the system within the bound, also when its
    /// cost exceeds the bound: the bound cuts by depth, not by cost.
    #[test]
    fn a_lower_ranked_method_closes_what_the_first_leaves_open() {
        let Some(mut ctx) = ctx() else {
            return;
        };
        let greedy = run_proof_search(&ctx, unproducible_action(), 1);
        assert_eq!(greedy.status, NodeStatus::Sorry);
        for (batch, alt_cost) in [(1, 0), (4, 0), (1, 8), (4, 8)] {
            ctx.top_n = Some(TopNConfig {
                alt_cost,
                ..config(3, false, batch)
            });
            let top3 = run_proof_search(&ctx, unproducible_action(), 1);
            assert_eq!(
                top3.status,
                NodeStatus::Contradictory,
                "batch={batch} alt_cost={alt_cost}"
            );
            assert!(
                matches!(top3.method, ProofMethod::SolveGoal(_)),
                "the proof uses the second-ranked method, got {:?}",
                top3.method
            );
            assert!(top3.children.is_empty(), "the goal has no producing rule");
        }
    }
}
