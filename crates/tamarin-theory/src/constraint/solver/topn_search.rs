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
//! branch. This module searches the AND/OR graph over the top N methods:
//!
//! - an OR node ([`Class`]) is one constraint system or, with merging on, one
//!   class of canonically equal systems ([`crate::canon`]); it is the choice
//!   among the system's methods;
//! - an AND node ([`And`]) is one applied method with its cases; a proof via
//!   that method must close all of them.
//!
//! Statuses are a least fixpoint over this graph ([`SearchGraph::propagate`]):
//! an AND is Contradictory once all its cases are, a class once one of its
//! ANDs is. A Solved system anywhere makes every system above it Solved,
//! since a solution of a refined system solves the system it refines. A class
//! settles only from settled children, so a cycle (a system reached again
//! below itself) never justifies itself.
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
use std::collections::{BTreeMap, BinaryHeap};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use rayon::prelude::*;
use tamarin_term::alpha_eq_ac::CanonLabelling;
use tamarin_term::fingerprint::Fingerprint;
use tamarin_utils::FastMap;

use crate::canon::{canonicalize_constraint_system_with_labelling, canonicalize_proof_method};
use crate::canon_fingerprint::{
    fingerprint_constraint_system, fingerprint_proof_method, CanonicalSystemFingerprint,
};
use crate::constraint::solver::context::ProofContext;
use crate::constraint::solver::proof_method::{
    exec_proof_method, is_finished, ProofMethod, Result as MethodResult,
};
use crate::constraint::solver::reduction::avoid_fresh_state;
use crate::constraint::solver::search::{self, candidate_methods, NodeStatus, ProofNode};
use crate::constraint::system::System;
use crate::elaborate::CollectedUserFuns;

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

pub type ClassId = u32;
const ROOT: ClassId = 0;

/// The first depth limit of [`SearchOrder::IdDfs`], as in the greedy driver.
const FIRST_LIMIT: u32 = 4;
/// Depths stay far below this; it only keeps `depth + 1` from overflowing.
const DEPTH_CAP: u32 = u32::MAX / 4;

/// A final verdict about a class: a property of its constraint systems, so
/// it holds for every system of the class at any depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Settled {
    Contradictory,
    Solved,
    Unfinishable,
}

impl Settled {
    fn of(result: &MethodResult) -> Settled {
        match result {
            MethodResult::Solved => Settled::Solved,
            MethodResult::Contradictory(_) => Settled::Contradictory,
            MethodResult::Unfinishable => Settled::Unfinishable,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Open,
    Settled(Settled),
}

/// One applied method of its class.
struct And {
    method: ProofMethod,
    /// The method canonicalized through the representative's labelling
    /// (merging on): identifies it among the candidates of any other system
    /// of the class when the proof is materialized.
    method_fp: Option<Fingerprint>,
    /// Sorted by case name, like the greedy driver's children.
    cases: Vec<(String, ClassId)>,
    status: Status,
}

/// One OR node: a constraint system, or a class of canonically equal ones.
struct Class {
    status: Status,
    /// The shallowest depth the class was reached at: its entries' depth.
    min_depth: u32,
    /// The cheapest cost the class was reached at ([`TopNConfig::alt_cost`]);
    /// tracked apart from `min_depth`, which may come from another path.
    min_cost: u32,
    /// Applied methods, in the heuristic's order.
    ands: Vec<And>,
    /// No further method will be applied: N were, the candidates ran out,
    /// or the class is a leaf.
    closed: bool,
    /// The class's one live frontier entry: `None` while it is being expanded
    /// or has no entry. Any other entry of the class is stale.
    pending: Option<Pending>,
    parents: Vec<(ClassId, u16)>,
    /// The edge that created the class; `None` for the root.
    first_parent: Option<(ClassId, u16)>,
    /// The method whose status settled the class, when it settled; `None`
    /// for a leaf and while open. Later methods can settle the same way
    /// through the class itself (a case leading back to it), so the proof
    /// follows this one: its cases all settled before the class did.
    settled_by: Option<u16>,
    /// When the class settled, on [`SearchGraph::clock`].
    settled_at: u64,
}

/// The classes, their methods and the transposition table. Holds no
/// `System`: those carry `Cell` caches, so they may move between threads but
/// not be shared, and the graph is read by the materializer's workers.
struct SearchGraph {
    classes: Vec<Class>,
    by_key: FastMap<CanonicalSystemFingerprint, ClassId>,
    /// Counts settlements, to order them ([`Class::settled_at`]).
    clock: u64,
}

impl SearchGraph {
    fn new() -> Self {
        SearchGraph {
            classes: Vec::new(),
            by_key: FastMap::default(),
            clock: 0,
        }
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    fn class(&self, c: ClassId) -> &Class {
        &self.classes[c as usize]
    }

    fn class_mut(&mut self, c: ClassId) -> &mut Class {
        &mut self.classes[c as usize]
    }

    fn add_class(&mut self, status: Status, depth: u32, cost: u32) -> ClassId {
        let id = ClassId::try_from(self.classes.len()).expect("more than u32::MAX classes");
        let settled_at = if status == Status::Open {
            0
        } else {
            self.tick()
        };
        self.classes.push(Class {
            status,
            min_depth: depth,
            min_cost: cost,
            ands: Vec::new(),
            closed: status != Status::Open,
            pending: None,
            parents: Vec::new(),
            first_parent: None,
            settled_by: None,
            settled_at,
        });
        id
    }

    fn add_and(&mut self, c: ClassId, method: ProofMethod, method_fp: Option<Fingerprint>) -> u16 {
        let ands = &mut self.class_mut(c).ands;
        let a = u16::try_from(ands.len()).expect("more than u16::MAX methods per class");
        ands.push(And {
            method,
            method_fp,
            cases: Vec::new(),
            status: Status::Open,
        });
        a
    }

    fn add_case(&mut self, c: ClassId, a: u16, name: String, child: ClassId) {
        self.class_mut(c).ands[a as usize].cases.push((name, child));
        let edge = (c, a);
        let class = self.class_mut(child);
        class.parents.push(edge);
        if class.first_parent.is_none() && child != ROOT {
            class.first_parent = Some(edge);
        }
    }

    /// Whether expanding `c` can still matter: it is the root, or some parent
    /// is open through an open method. Statuses only ever settle, so this
    /// only changes back when a new edge reaches `c`.
    fn is_relevant(&self, c: ClassId) -> bool {
        c == ROOT
            || self.class(c).parents.iter().any(|&(p, a)| {
                let parent = self.class(p);
                parent.status == Status::Open && parent.ands[a as usize].status == Status::Open
            })
    }

    /// The class's [`Pending`] if `e`, popped from the frontier, is its live
    /// entry; `None` for a stale one.
    fn live(&self, e: &Entry) -> Option<Pending> {
        let pending = self.class(e.class.0).pending.filter(|p| p.entry == *e)?;
        assert!(
            !pending.parked,
            "top-N search: a live entry is in the frontier and parked at once (class #{})",
            e.class.0
        );
        Some(pending)
    }

    fn and_status(&self, c: ClassId, a: u16) -> Status {
        let and = &self.class(c).ands[a as usize];
        // A `sorry` method (an oracle that ranked nothing, with quit-on-empty)
        // closes nothing, whatever its cases, as in the greedy driver.
        if matches!(and.method, ProofMethod::Sorry(_)) {
            return Status::Open;
        }
        let mut all_contradictory = true;
        let mut all_settled = true;
        for &(_, child) in &and.cases {
            match self.class(child).status {
                Status::Settled(Settled::Solved) => return Status::Settled(Settled::Solved),
                Status::Settled(Settled::Contradictory) => {}
                Status::Settled(Settled::Unfinishable) => all_contradictory = false,
                Status::Open => {
                    all_contradictory = false;
                    all_settled = false;
                }
            }
        }
        if all_contradictory {
            Status::Settled(Settled::Contradictory)
        } else if all_settled {
            Status::Settled(Settled::Unfinishable)
        } else {
            Status::Open
        }
    }

    /// The status `c`'s methods give it. Two methods are two complete case
    /// splits of the same systems, so one finding a trace while another
    /// proves there is none means a false-positive merge or a solver bug.
    fn class_status(&self, c: ClassId) -> Status {
        let class = self.class(c);
        let mut solved = false;
        let mut contradictory = false;
        let mut all_unfinishable = !class.ands.is_empty();
        for and in &class.ands {
            match and.status {
                Status::Settled(Settled::Solved) => solved = true,
                Status::Settled(Settled::Contradictory) => contradictory = true,
                Status::Settled(Settled::Unfinishable) => {}
                Status::Open => all_unfinishable = false,
            }
        }
        if solved && contradictory {
            panic!(
                "top-N search: two methods of class #{c} disagree, one finds a trace and \
                 another proves there is none: a false-positive merge or a solver bug\n{}",
                self.describe_conflict(c)
            );
        }
        if solved {
            Status::Settled(Settled::Solved)
        } else if contradictory {
            Status::Settled(Settled::Contradictory)
        } else if all_unfinishable && class.closed {
            Status::Settled(Settled::Unfinishable)
        } else {
            Status::Open
        }
    }

    /// The `method [case]` steps from the root to `c` along first-parent
    /// edges: a proof path that reaches the class's representative system.
    /// A parent is always created before its children, so this ends.
    fn path_to(&self, c: ClassId) -> String {
        let mut steps = Vec::new();
        let mut cur = c;
        while let Some((p, a)) = self.class(cur).first_parent {
            let and = &self.class(p).ands[a as usize];
            let case = and
                .cases
                .iter()
                .find(|&&(_, child)| child == cur)
                .map_or("?", |(name, _)| name.as_str());
            steps.push(format!(
                "{} [{case}]",
                crate::pretty_theory::pretty_proof_method_inline(&and.method)
            ));
            cur = p;
        }
        steps.reverse();
        if steps.is_empty() {
            "<root>".into()
        } else {
            steps.join(" -> ")
        }
    }

    /// For a class whose methods disagree: its path and, per method that
    /// settled, its cases with how many edges reach each case's class (a
    /// merged class, reached along several, is where a false positive hides).
    /// A case merged into a class created elsewhere also gets that class's
    /// path: replaying both paths gives the two systems that were merged.
    fn describe_conflict(&self, c: ClassId) -> String {
        let class = self.class(c);
        let mut out = format!("  path to #{c}: {}\n", self.path_to(c));
        for (i, and) in class.ands.iter().enumerate() {
            if and.status == Status::Open {
                continue;
            }
            out += &format!(
                "  method {i} ({:?}): {}\n",
                and.status,
                crate::pretty_theory::pretty_proof_method_inline(&and.method)
            );
            for (name, child) in &and.cases {
                let cl = self.class(*child);
                out += &format!(
                    "    case {name} -> #{child} {:?}, reached along {} edge(s), first via {}\n",
                    cl.status,
                    cl.parents.len(),
                    cl.first_parent
                        .map_or("-".into(), |(p, a)| format!("#{p} method {a}"))
                );
                if cl.first_parent != Some((c, i as u16)) {
                    out += &format!("      path to #{child}: {}\n", self.path_to(*child));
                }
                if cl.status == Status::Settled(Settled::Solved) {
                    out += &format!("      trace from #{child}: {}\n", self.trace_from(*child));
                }
            }
        }
        out
    }

    /// The `method [case]` steps from the Solved class `c` down to a solved
    /// leaf, the way the proof would take them. A step into a class created
    /// along another edge is marked `(merged)`: there the path continues
    /// with the representative's system, not the one this path produced.
    fn trace_from(&self, mut c: ClassId) -> String {
        let mut steps = Vec::new();
        let mut seen = vec![c];
        loop {
            let class = self.class(c);
            let next = class.ands.iter().enumerate().find_map(|(a, and)| {
                if and.status != Status::Settled(Settled::Solved) {
                    return None;
                }
                and.cases
                    .iter()
                    .find(|&&(_, d)| {
                        self.class(d).status == Status::Settled(Settled::Solved) && !seen.contains(&d)
                    })
                    .map(|(name, d)| (a, and, name, *d))
            });
            let Some((a, and, name, d)) = next else {
                break;
            };
            let merged = self.class(d).first_parent != Some((c, a as u16));
            steps.push(format!(
                "{} [{name}] -> #{d}{}",
                crate::pretty_theory::pretty_proof_method_inline(&and.method),
                if merged { " (merged)" } else { "" }
            ));
            seen.push(d);
            c = d;
        }
        if steps.is_empty() {
            "<a solved leaf>".into()
        } else {
            steps.join(" -> ")
        }
    }

    /// Re-derives `c`'s status after one of its methods changed or it was
    /// closed; returns the parent edges to re-check if it settled.
    fn recheck(&mut self, c: ClassId) -> Vec<(ClassId, u16)> {
        let status = self.class_status(c);
        if self.class(c).status != Status::Open || status == Status::Open {
            return Vec::new();
        }
        let settled_by = self
            .class(c)
            .ands
            .iter()
            .position(|and| and.status == status)
            .expect("a class settles through one of its methods");
        let settled_at = self.tick();
        let class = self.class_mut(c);
        class.status = status;
        class.settled_by = Some(settled_by as u16);
        class.settled_at = settled_at;
        class.parents.clone()
    }

    /// Propagates upward from the methods in `work`, whose cases changed.
    fn propagate(&mut self, mut work: Vec<(ClassId, u16)>) {
        while let Some((c, a)) = work.pop() {
            let status = self.and_status(c, a);
            let and = &mut self.class_mut(c).ands[a as usize];
            if and.status == status {
                continue;
            }
            and.status = status;
            work.extend(self.recheck(c));
        }
    }

    /// Marks `c` as getting no further method, which can settle it as
    /// Unfinishable.
    fn close(&mut self, c: ClassId) {
        self.class_mut(c).closed = true;
        let parents = self.recheck(c);
        self.propagate(parents);
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

/// A class's system and how far its expansion got. It sits in the engine's
/// store while the class may still get a method, and in a task while one of
/// its methods runs.
struct Stored {
    sys: Box<System>,
    /// `sys`'s canonical labelling (merging on), to canonicalize the methods
    /// applied to it.
    labelling: Option<CanonLabelling>,
    /// `candidate_methods`, computed at the first expansion.
    ranked: Option<Vec<ProofMethod>>,
    /// The next candidate to try.
    cursor: usize,
}

impl Stored {
    fn new(sys: System, labelling: Option<CanonLabelling>) -> Self {
        Stored {
            sys: Box::new(sys),
            labelling,
            ranked: None,
            cursor: 0,
        }
    }
}

struct Task {
    class: ClassId,
    depth: u32,
    /// The children lie at or below `--bound`: none is examined.
    cut_children: bool,
    stored: Stored,
}

/// One case of an applied method, as a worker prepared it.
enum Child {
    /// At or below `--bound`.
    Cut,
    /// Already finished; never canonicalized, so never merged.
    Finished(Settled),
    /// Merging on: its canonical key.
    Keyed(CanonicalSystemFingerprint, Stored),
    /// Merging off.
    Unkeyed(Stored),
}

enum Outcome {
    Applied {
        method: ProofMethod,
        method_fp: Option<Fingerprint>,
        cases: Vec<(String, Child)>,
    },
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

#[derive(Debug, Default, Clone, Copy)]
struct WorkStats {
    execs: u64,
    canons: u64,
    canon_time: Duration,
    busy: Duration,
}

/// What a worker thread needs before it runs solver code for this search:
/// the thread-locals the greedy driver's fan-out also replicates (the
/// deadline, the user-function signature) and a maude handle of its own.
/// `exec_proof_method` resets its handle's fresh-variable counter at every
/// step, so a handle shared by two threads would race; seeding each task's
/// counter from its system also keeps the results independent of the
/// thread that runs them.
///
/// The handle always talks to the lemma's own maude process, never to one
/// from `maude_pool`: a pooled process can return an AC-equal result with
/// its arguments in another order, and whether a task gets a pooled process
/// depends on what the other lemmas' searches hold at that moment, so the
/// printed proof would depend on timing. Maude calls then take turns on one
/// process, which cost nothing measurable where tried (the solver's own work
/// and canonicalization dominate).
struct WorkerEnv {
    deadline: Option<Instant>,
    user_funs: CollectedUserFuns,
}

impl WorkerEnv {
    fn capture(deadline: Option<Instant>) -> Self {
        WorkerEnv {
            deadline,
            user_funs: crate::elaborate::snapshot_user_funs(),
        }
    }

    /// Runs `f` with this search's thread-locals installed and a maude handle
    /// seeded at `avoid_next`. Restores the calling thread's deadline after:
    /// rayon runs part of a parallel map on the calling thread when that is a
    /// pool thread itself.
    fn run<R>(&self, ctx: &ProofContext, avoid_next: u64, f: impl FnOnce(&ProofContext) -> R) -> R {
        let previous_deadline = search::replace_deadline(self.deadline);
        let _user_funs = crate::elaborate::set_user_funs_from_collected(&self.user_funs);
        let maude = ctx.maude.with_fresh_counter_next(avoid_next);
        let result = f(&ctx.with_swapped_maude(maude));
        search::replace_deadline(previous_deadline);
        result
    }
}

fn canonical_key(
    ctx: &ProofContext,
    sys: &System,
    work: &mut WorkStats,
) -> (CanonicalSystemFingerprint, CanonLabelling) {
    let start = Instant::now();
    let (canon, labelling) = canonicalize_constraint_system_with_labelling(sys, &ctx.color_table)
        .unwrap_or_else(|e| panic!("TAM_RS_MERGE: canonicalizing a constraint system failed: {e}"));
    let key = fingerprint_constraint_system(&canon);
    work.canons += 1;
    work.canon_time += start.elapsed();
    (key, labelling)
}

/// Prepares one case: finished, or else keyed for merging.
fn classify(ctx: &ProofContext, sys: System, merge: bool, work: &mut WorkStats) -> Child {
    if let Some(result) = is_finished(ctx, &sys) {
        return Child::Finished(Settled::of(&result));
    }
    if merge {
        let (key, labelling) = canonical_key(ctx, &sys, work);
        Child::Keyed(key, Stored::new(sys, Some(labelling)))
    } else {
        Child::Unkeyed(Stored::new(sys, None))
    }
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
        work.execs += 1;
        let Some(mut cases) = exec_proof_method(ctx, &method, &stored.sys) else {
            // `exec_proof_method` also answers `None`, without trying the
            // method, once the deadline has passed.
            if stop() {
                break Outcome::Aborted;
            }
            stored.cursor += 1;
            continue;
        };
        stored.cursor += 1;
        cases.sort_by(|a, b| a.0.cmp(&b.0));
        let method_fp = stored.labelling.as_ref().map(|labelling| {
            fingerprint_proof_method(&canonicalize_proof_method(&method, &stored.sys, labelling))
        });
        let cases = cases
            .into_iter()
            .map(|(name, sys)| {
                let child = if cut_children {
                    Child::Cut
                } else {
                    classify(ctx, sys, merge, &mut work)
                };
                (name, child)
            })
            .collect();
        break Outcome::Applied {
            method,
            method_fp,
            cases,
        };
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

/// Why the search stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum End {
    /// The root settled.
    Settled,
    /// Nothing left to expand within the top N methods.
    Exhausted,
    Deadline,
}

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

    fn add_class(
        &mut self,
        status: Status,
        depth: u32,
        cost: u32,
        stored: Option<Stored>,
    ) -> ClassId {
        let id = self.graph.add_class(status, depth, cost);
        self.store.push(stored);
        id
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
            .min_cost
            .saturating_add(self.alt_cost(self.config.n - rank));
        let entry = Entry::new(self.config.order, depth, rank, c);
        let queued = class.pending.is_some_and(|p| p.entry == entry && !p.parked);
        self.graph.class_mut(c).pending = Some(Pending {
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
        match child {
            Child::Cut => {
                let c = self.add_class(Status::Open, depth, cost, None);
                self.graph.class_mut(c).closed = true;
                c
            }
            Child::Finished(settled) => {
                self.stats.leaves += 1;
                self.add_class(Status::Settled(settled), depth, cost, None)
            }
            Child::Unkeyed(stored) => {
                let c = self.add_class(Status::Open, depth, cost, Some(stored));
                self.push_entry(c, self.config.n);
                c
            }
            Child::Keyed(key, stored) => {
                if let Some(&existing) = self.graph.by_key.get(&key) {
                    self.stats.merges += 1;
                    self.lower(existing, depth, cost);
                    self.requeue_if_dormant(existing);
                    return existing;
                }
                let c = self.add_class(Status::Open, depth, cost, Some(stored));
                self.graph.by_key.insert(key, c);
                self.push_entry(c, self.config.n);
                c
            }
        }
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
            if depth >= class.min_depth && cost >= class.min_cost {
                continue;
            }
            let (depth, cost) = (depth.min(class.min_depth), cost.min(class.min_cost));
            let class = self.graph.class_mut(c);
            class.min_depth = depth;
            class.min_cost = cost;
            if let Some(pending) = class.pending {
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
            && class.pending.is_none()
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
                self.graph.class_mut(c).pending = None;
                self.store[c as usize] = None;
                continue;
            }
            if !self.graph.is_relevant(c) {
                // The system stays: a later edge can make the class relevant
                // again (`requeue_if_dormant`).
                self.graph.class_mut(c).pending = None;
                continue;
            }
            // Nothing is parked at the largest limit: a saturated cost must
            // not keep an entry parked forever.
            if pending.cost >= limit && limit < u32::MAX {
                self.graph.class_mut(c).pending = Some(Pending {
                    parked: true,
                    ..pending
                });
                self.parked.push(c);
                continue;
            }
            self.graph.class_mut(c).pending = None;
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
        self.stats.work.execs += work.execs;
        self.stats.work.canons += work.canons;
        self.stats.work.canon_time += work.canon_time;
        self.stats.work.busy += work.busy;
        match outcome {
            Outcome::Aborted => {
                // The search stops after this batch; keep the system anyway
                // so the graph stays consistent.
                self.stats.aborted += 1;
                self.store[c as usize] = Some(stored);
            }
            Outcome::Terminal => self.graph.close(c),
            Outcome::Applied {
                method,
                method_fp,
                cases,
            } => {
                self.stats.applied += 1;
                let a = self.graph.add_and(c, method, method_fp);
                let class = self.graph.class(c);
                let depth = class.min_depth + 1;
                let cost = class
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
// Materializing the proof
// =============================================================================

/// A case of a method run again, resolved against the graph.
enum Resolved {
    /// At or below `--bound`.
    Cut(System),
    Finished(System, MethodResult),
    /// A system of this class, with its canonical labelling when merging.
    Class(System, Option<CanonLabelling>, ClassId),
}

impl Resolved {
    fn sys(&self) -> &System {
        match self {
            Resolved::Cut(sys) | Resolved::Finished(sys, _) | Resolved::Class(sys, _, _) => sys,
        }
    }
}

/// Rebuilds a [`ProofNode`] tree from the settled graph by running the
/// settling method of each class again, starting from the root system.
struct Materializer<'g> {
    graph: &'g SearchGraph,
    merge: bool,
    bound: u32,
    end: End,
    env: WorkerEnv,
}

fn retain(mut node: ProofNode) -> ProofNode {
    if search::drop_sys_after_expand(&node, search::sys_retention()) {
        node.sys = System::default();
    }
    node
}

fn leaf(method: ProofMethod, sys: System, status: NodeStatus) -> ProofNode {
    retain(ProofNode {
        method,
        sys,
        children: BTreeMap::new(),
        status,
        annotated: true,
    })
}

fn sorry(sys: System, message: String) -> ProofNode {
    leaf(ProofMethod::Sorry(Some(message)), sys, NodeStatus::Sorry)
}

impl Materializer<'_> {
    fn root(&self, ctx: &ProofContext, sys: System) -> ProofNode {
        let resolved = self.resolve(ctx, sys, 0, Some(ROOT));
        self.build(ctx, resolved, 0, None, &[])
    }

    /// Maps `items` through `f`, concurrently when there are several, each in
    /// its own worker environment.
    fn map_workers<T: Send, R: Send>(
        &self,
        ctx: &ProofContext,
        items: Vec<T>,
        sys_of: impl Fn(&T) -> &System + Sync,
        f: impl Fn(&ProofContext, T) -> R + Sync,
    ) -> Vec<R> {
        let run = |item: T| -> R {
            let avoid_next = avoid_fresh_state(sys_of(&item));
            self.env
                .run(ctx, avoid_next, |worker_ctx| f(worker_ctx, item))
        };
        if items.len() <= 1 {
            items.into_iter().map(run).collect()
        } else {
            items.into_par_iter().map(run).collect()
        }
    }

    /// The class of `sys`, reached at `depth`. Merging off, a case's class is
    /// the graph's (`known`); merging on, it is looked up by canonical key.
    fn resolve(
        &self,
        ctx: &ProofContext,
        sys: System,
        depth: u32,
        known: Option<ClassId>,
    ) -> Resolved {
        if depth >= self.bound {
            return Resolved::Cut(sys);
        }
        if let Some(result) = is_finished(ctx, &sys) {
            return Resolved::Finished(sys, result);
        }
        if !self.merge {
            let class = known.expect("merging off: the class of a case comes from the graph");
            return Resolved::Class(sys, None, class);
        }
        let (key, labelling) = canonical_key(ctx, &sys, &mut WorkStats::default());
        let class = *self.graph.by_key.get(&key).unwrap_or_else(|| {
            panic!("materializing the proof: a case reaches a system the search never saw")
        });
        Resolved::Class(sys, Some(labelling), class)
    }

    /// Whether `resolved` counts as settled `settled` below a class that
    /// settled at `before`: a finished system always, a class only if it
    /// settled earlier. The proof follows only such cases, so it cannot loop.
    fn settled_before(&self, resolved: &Resolved, settled: Settled, before: u64) -> bool {
        match resolved {
            Resolved::Cut(_) => false,
            Resolved::Finished(_, result) => Settled::of(result) == settled,
            Resolved::Class(_, _, d) => {
                let class = self.graph.class(*d);
                class.status == Status::Settled(settled) && class.settled_at < before
            }
        }
    }

    /// Whether class `d` reaches a finished Solved system through Solved
    /// classes without passing through `avoid` (the path to it): a trace the
    /// proof can show below `d` without looping.
    fn reaches_trace(&self, d: ClassId, avoid: &[ClassId]) -> bool {
        let mut seen = vec![false; self.graph.classes.len()];
        let mut stack = vec![d];
        while let Some(x) = stack.pop() {
            if seen[x as usize] || avoid.contains(&x) {
                continue;
            }
            seen[x as usize] = true;
            let class = self.graph.class(x);
            if class.status != Status::Settled(Settled::Solved) {
                continue;
            }
            if class.ands.is_empty() {
                return true;
            }
            for and in &class.ands {
                if and.status == Status::Settled(Settled::Solved) {
                    stack.extend(and.cases.iter().map(|&(_, y)| y));
                }
            }
        }
        false
    }

    /// Whether a case continues a trace below the path `avoid`.
    fn continues_trace(&self, resolved: &Resolved, avoid: &[ClassId]) -> bool {
        match resolved {
            Resolved::Cut(_) => false,
            Resolved::Finished(_, result) => Settled::of(result) == Settled::Solved,
            Resolved::Class(_, _, d) => self.reaches_trace(*d, avoid),
        }
    }

    fn build(
        &self,
        ctx: &ProofContext,
        resolved: Resolved,
        depth: u32,
        edge: Option<(ClassId, u16)>,
        path: &[ClassId],
    ) -> ProofNode {
        match resolved {
            Resolved::Cut(sys) => sorry(sys, format!("bound {} hit", self.bound)),
            Resolved::Finished(sys, result) => {
                let status = search::node_status_of(&result);
                leaf(ProofMethod::Finished(result), sys, status)
            }
            Resolved::Class(sys, labelling, c) => {
                self.expand(ctx, sys, labelling, c, depth, edge, path)
            }
        }
    }

    /// The method the proof shows for `c`, reached along `avoid` (the path,
    /// `c` included). A trace takes the first method, in the heuristic's
    /// order, with a case that continues it without looping; other settled
    /// classes take the method that settled them, whose cases all settled
    /// earlier; an open class takes its first method, if any.
    fn choose(&self, c: ClassId, avoid: &[ClassId]) -> Option<usize> {
        let class = self.graph.class(c);
        match class.status {
            Status::Open => (!class.ands.is_empty()).then_some(0),
            Status::Settled(Settled::Solved) => Some(
                class
                    .ands
                    .iter()
                    .position(|and| {
                        and.status == Status::Settled(Settled::Solved)
                            && and.cases.iter().any(|&(_, d)| self.reaches_trace(d, avoid))
                    })
                    .expect("a Solved class reaches a trace that avoids the path to it"),
            ),
            Status::Settled(_) => Some(
                class
                    .settled_by
                    .expect("a settled class that is not a leaf records its method")
                    as usize,
            ),
        }
    }

    fn open_message(&self, c: ClassId) -> String {
        let class = self.graph.class(c);
        if class.closed && class.ands.is_empty() {
            "no method".into()
        } else if self.end == End::Deadline {
            "deadline reached".into()
        } else {
            "top-N: undecided".into()
        }
    }

    /// Among `sys`'s candidates, the method whose canonical form is `fp`:
    /// canonically equal systems offer the same methods up to renaming.
    fn matching_method(
        &self,
        ctx: &ProofContext,
        sys: &System,
        labelling: &CanonLabelling,
        fp: Fingerprint,
        depth: u32,
        c: ClassId,
    ) -> ProofMethod {
        candidate_methods(sys, ctx, depth as usize)
            .into_iter()
            .find(|m| fingerprint_proof_method(&canonicalize_proof_method(m, sys, labelling)) == fp)
            .unwrap_or_else(|| {
                panic!(
                    "materializing the proof: no candidate method of a system of class #{c} \
                     matches the method applied to the class: canonically equal systems \
                     offer different methods"
                )
            })
    }

    #[allow(clippy::too_many_arguments)]
    fn expand(
        &self,
        ctx: &ProofContext,
        sys: System,
        labelling: Option<CanonLabelling>,
        c: ClassId,
        depth: u32,
        edge: Option<(ClassId, u16)>,
        path: &[ClassId],
    ) -> ProofNode {
        let class = self.graph.class(c);
        if path.contains(&c) {
            return sorry(sys, "top-N: cycle".into());
        }
        if class.status == Status::Open && edge.is_some() && edge != class.first_parent {
            // An undecided class is shown once, below the edge that created
            // it; unfolding it everywhere can take exponential space.
            return sorry(sys, "top-N: undecided, shown elsewhere".into());
        }
        let mut path = path.to_vec();
        path.push(c);
        let Some(a) = self.choose(c, &path) else {
            return sorry(sys, self.open_message(c));
        };
        let and = &class.ands[a];
        let method = match (&labelling, and.method_fp) {
            (Some(labelling), Some(fp)) => self.matching_method(ctx, &sys, labelling, fp, depth, c),
            _ => and.method.clone(),
        };
        let mut cases = exec_proof_method(ctx, &method, &sys).unwrap_or_else(|| {
            panic!("materializing the proof: a method applied during the search no longer applies (class #{c})")
        });
        cases.sort_by(|x, y| x.0.cmp(&y.0));
        let known: Vec<Option<ClassId>> = if self.merge {
            vec![None; cases.len()]
        } else {
            let names: Vec<&str> = cases.iter().map(|(name, _)| name.as_str()).collect();
            let graph_names: Vec<&str> = and.cases.iter().map(|(name, _)| name.as_str()).collect();
            assert_eq!(
                names, graph_names,
                "materializing the proof: running a method again gave other cases (class #{c})"
            );
            and.cases.iter().map(|&(_, child)| Some(child)).collect()
        };
        let items: Vec<((String, System), Option<ClassId>)> =
            cases.into_iter().zip(known).collect();
        let resolved: Vec<(String, Resolved)> = self.map_workers(
            ctx,
            items,
            |((_, sys), _)| sys,
            |worker_ctx, ((name, sys), known)| {
                (name, self.resolve(worker_ctx, sys, depth + 1, known))
            },
        );
        let selected: Vec<(String, Resolved)> = match class.status {
            // A trace needs one path: the first case in name order that
            // continues it without looping, like the greedy driver's
            // `extract_solved_path` takes the leftmost solved leaf.
            Status::Settled(Settled::Solved) => {
                let first = resolved
                    .into_iter()
                    .find(|(_, r)| self.continues_trace(r, &path))
                    .expect("the method chosen for a trace has a case that continues it");
                vec![first]
            }
            Status::Settled(Settled::Contradictory) => {
                for (name, r) in &resolved {
                    assert!(
                        self.settled_before(r, Settled::Contradictory, class.settled_at),
                        "materializing the proof: case `{name}` of the method that closed \
                         class #{c} is not contradictory here"
                    );
                }
                resolved
            }
            _ => resolved,
        };
        let a = a as u16;
        let children: BTreeMap<String, ProofNode> = self
            .map_workers(
                ctx,
                selected,
                |(_, r)| r.sys(),
                |worker_ctx, (name, r)| {
                    (
                        name,
                        self.build(worker_ctx, r, depth + 1, Some((c, a)), &path),
                    )
                },
            )
            .into_iter()
            .collect();
        let status = if children.is_empty() {
            // Zero cases close the branch, except under a `sorry` method; as
            // in the greedy driver.
            if matches!(method, ProofMethod::Sorry(_)) {
                NodeStatus::Sorry
            } else {
                NodeStatus::Contradictory
            }
        } else {
            search::rollup_from_children(&children)
        };
        retain(ProofNode {
            method,
            sys,
            children,
            status,
            annotated: true,
        })
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

    // --- The graph (no maude) --------------------------------------------

    fn open(g: &mut SearchGraph) -> ClassId {
        g.add_class(Status::Open, 0, 0)
    }

    fn leaf_class(g: &mut SearchGraph, settled: Settled) -> ClassId {
        g.add_class(Status::Settled(settled), 0, 0)
    }

    /// Adds a method to `c` with `children` as its cases and propagates.
    fn and(g: &mut SearchGraph, c: ClassId, children: &[ClassId]) -> u16 {
        let a = g.add_and(c, ProofMethod::Simplify, None);
        for (i, &child) in children.iter().enumerate() {
            g.add_case(c, a, format!("case_{i}"), child);
        }
        g.propagate(vec![(c, a)]);
        a
    }

    #[test]
    fn a_method_closes_a_class_only_when_all_its_cases_are_contradictory() {
        let mut g = SearchGraph::new();
        let root = open(&mut g);
        let pending = open(&mut g);
        let contra = leaf_class(&mut g, Settled::Contradictory);
        and(&mut g, root, &[pending, contra]);
        assert_eq!(g.class(root).status, Status::Open);
        and(&mut g, root, &[contra, contra]);
        assert_eq!(
            g.class(root).status,
            Status::Settled(Settled::Contradictory)
        );
    }

    #[test]
    fn a_solved_system_anywhere_makes_the_root_solved() {
        let mut g = SearchGraph::new();
        let root = open(&mut g);
        let a = open(&mut g);
        let b = open(&mut g);
        let contra = leaf_class(&mut g, Settled::Contradictory);
        let solved = leaf_class(&mut g, Settled::Solved);
        and(&mut g, root, &[a]);
        and(&mut g, a, &[b, contra]);
        assert_eq!(g.class(root).status, Status::Open);
        and(&mut g, b, &[solved]);
        assert_eq!(g.class(a).status, Status::Settled(Settled::Solved));
        assert_eq!(g.class(root).status, Status::Settled(Settled::Solved));
    }

    #[test]
    #[should_panic(expected = "disagree")]
    fn methods_that_disagree_panic() {
        let mut g = SearchGraph::new();
        let root = open(&mut g);
        let contra = leaf_class(&mut g, Settled::Contradictory);
        let solved = leaf_class(&mut g, Settled::Solved);
        and(&mut g, root, &[contra]);
        and(&mut g, root, &[solved]);
    }

    #[test]
    fn a_cycle_does_not_justify_itself() {
        let mut g = SearchGraph::new();
        let a = open(&mut g);
        let b = open(&mut g);
        and(&mut g, a, &[b]);
        and(&mut g, b, &[a]);
        g.close(a);
        g.close(b);
        assert_eq!(g.class(a).status, Status::Open);
        assert_eq!(g.class(b).status, Status::Open);
        // A real proof of `b` settles both.
        let contra = leaf_class(&mut g, Settled::Contradictory);
        and(&mut g, b, &[contra]);
        assert_eq!(g.class(b).status, Status::Settled(Settled::Contradictory));
        assert_eq!(g.class(a).status, Status::Settled(Settled::Contradictory));
    }

    /// `b` settles through its second method; `a` then settles through `b`,
    /// which makes `b`'s first method (back to `a`) settle too. The proof of
    /// `b` must follow the second: the first loops through `a`.
    #[test]
    fn a_class_records_the_method_that_settled_it_first() {
        let mut g = SearchGraph::new();
        let a = open(&mut g);
        let b = open(&mut g);
        let contra = leaf_class(&mut g, Settled::Contradictory);
        and(&mut g, a, &[b]);
        and(&mut g, b, &[a]);
        and(&mut g, b, &[contra]);
        assert_eq!(g.class(b).settled_by, Some(1));
        assert_eq!(g.class(a).settled_by, Some(0));
        assert_eq!(
            g.class(b).ands[0].status,
            Status::Settled(Settled::Contradictory)
        );
        assert!(g.class(contra).settled_at < g.class(b).settled_at);
        assert!(g.class(b).settled_at < g.class(a).settled_at);
    }

    /// `b`'s first case leads back to `a`, which is solved only through `b`:
    /// below `b`, only its second case continues the trace without looping.
    #[test]
    fn a_trace_is_continued_only_through_cases_that_avoid_the_path() {
        let mut g = SearchGraph::new();
        let a = open(&mut g);
        let b = open(&mut g);
        let solved = leaf_class(&mut g, Settled::Solved);
        and(&mut g, a, &[b]);
        and(&mut g, b, &[a, solved]);
        assert_eq!(g.class(a).status, Status::Settled(Settled::Solved));
        let m = Materializer {
            graph: &g,
            merge: false,
            bound: DEPTH_CAP,
            end: End::Settled,
            env: WorkerEnv::capture(None),
        };
        assert!(m.reaches_trace(a, &[]));
        assert!(m.reaches_trace(b, &[a]));
        assert!(!m.reaches_trace(a, &[b]), "a's only trace runs through b");
        assert_eq!(m.choose(b, &[a, b]), Some(0));
    }

    #[test]
    fn unfinishable_needs_a_closed_class() {
        let mut g = SearchGraph::new();
        let root = open(&mut g);
        let unfinishable = leaf_class(&mut g, Settled::Unfinishable);
        and(&mut g, root, &[unfinishable]);
        assert_eq!(
            g.class(root).status,
            Status::Open,
            "another method may still close it"
        );
        g.close(root);
        assert_eq!(g.class(root).status, Status::Settled(Settled::Unfinishable));
    }

    #[test]
    fn a_sorry_method_closes_nothing() {
        let mut g = SearchGraph::new();
        let root = open(&mut g);
        let a = g.add_and(root, ProofMethod::Sorry(Some("oracle".into())), None);
        g.propagate(vec![(root, a)]);
        g.close(root);
        assert_eq!(g.class(root).status, Status::Open);
    }

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
        let c = open(&mut g);
        let old = Entry::new(SearchOrder::IdDfs, 5, 3, c);
        let new = Entry::new(SearchOrder::IdDfs, 2, 3, c);
        g.class_mut(c).pending = Some(Pending {
            entry: new,
            depth: 2,
            cost: 2,
            parked: false,
        });
        assert!(g.live(&old).is_none());
        assert!(g.live(&new).is_some());
        g.class_mut(c).pending = None;
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
        assert_eq!(engine.graph.class(c).pending.map(|p| p.cost), Some(1));
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
    use crate::pretty_theory::pretty_proof_body;
    use crate::test_maude::maude_path;
    use tamarin_term::maude_sig::pair_maude_sig;

    /// `None` only when no maude resolves (`TAM_ALLOW_NO_MAUDE`); a maude
    /// that resolves but does not start panics, as in `search`'s tests.
    fn ctx() -> Option<ProofContext> {
        let path = maude_path()?;
        let h = tamarin_term::maude_proc::MaudeHandle::start(&path, pair_maude_sig())
            .unwrap_or_else(|e| panic!("maude at {path} failed to start: {e:?}"));
        Some(ProofContext::new(h, Vec::new()))
    }

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

    fn past_initial(sys: &mut System) {
        sys.add_less(crate::constraint::constraints::LessAtom::new(
            tamarin_term::lterm::LVar::new("a", tamarin_term::lterm::LSort::Node, 0),
            tamarin_term::lterm::LVar::new("b", tamarin_term::lterm::LSort::Node, 0),
            crate::constraint::constraints::Reason::Fresh,
        ));
    }

    #[test]
    fn an_empty_disjunction_closes_like_the_greedy_search() {
        assert_agrees_with_greedy(
            || {
                let mut sys = System::empty();
                past_initial(&mut sys);
                sys.formulas_mut()
                    .push(std::sync::Arc::new(crate::guarded::gfalse()));
                sys.add_goal(crate::constraint::constraints::Goal::Disj(
                    crate::constraint::constraints::Disj::new(Vec::new()),
                ));
                sys
            },
            5,
            true,
        );
    }

    #[test]
    fn a_two_branch_disjunction_finds_the_trace_like_the_greedy_search() {
        assert_agrees_with_greedy(
            || {
                let mut sys = System::empty();
                past_initial(&mut sys);
                sys.add_goal(crate::constraint::constraints::Goal::Disj(
                    crate::constraint::constraints::Disj::new(vec![
                        crate::guarded::gtrue(),
                        crate::guarded::gfalse(),
                    ]),
                ));
                sys
            },
            10,
            true,
        );
    }

    #[test]
    fn simplify_then_solved_like_the_greedy_search() {
        assert_agrees_with_greedy(
            || {
                let mut sys = System::empty();
                past_initial(&mut sys);
                sys.formulas_mut()
                    .push(std::sync::Arc::new(crate::guarded::gtrue()));
                sys.formulas_mut()
                    .push(std::sync::Arc::new(crate::guarded::gtrue()));
                sys
            },
            5,
            true,
        );
    }

    /// An `Out` action goal in a context without rules: `simplify` ranks
    /// first but leaves it open, solving the goal closes the system.
    fn unproducible_action() -> System {
        use tamarin_term::vterm::Lit;
        let mut sys = System::empty();
        let v = tamarin_term::lterm::LVar::new("x", tamarin_term::lterm::LSort::Msg, 0);
        let v2 = tamarin_term::lterm::LVar::new("y", tamarin_term::lterm::LSort::Msg, 0);
        let tx: tamarin_term::lterm::LNTerm = tamarin_term::term::Term::Lit(Lit::Var(v));
        let ty: tamarin_term::lterm::LNTerm = tamarin_term::term::Term::Lit(Lit::Var(v2));
        let i = tamarin_term::lterm::LVar::new("i", tamarin_term::lterm::LSort::Node, 0);
        sys.add_goal(crate::constraint::constraints::Goal::Action(
            i,
            crate::fact::out_fact(tx),
        ));
        sys.subterm_store_mut().add(ty.clone(), ty);
        sys
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
