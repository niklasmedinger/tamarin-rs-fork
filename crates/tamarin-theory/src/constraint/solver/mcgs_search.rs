//! (No HS analog.) Monte Carlo graph search (MCGS): an opt-in alternative to
//! the greedy driver in [`super::search`] and to [`super::topn_search`]. A
//! port of TamRL's search (`tamarinml`, `src/rl/mcts.py`) without its
//! network: selection is TamRL's prior-only score (`--deactivate_model`),
//! and the prior combines the heuristic's rank with a BM25 lexical score
//! ([`super::bm25`], TamRL's `idf_prior.py`).
//!
//! The search runs on the AND/OR graph of [`super::search_graph`]: OR nodes
//! are systems (with `TAM_RS_MERGE`, classes of canonically equal ones, which
//! makes the tree a graph), AND nodes applied methods with their cases.
//!
//! ## One step
//!
//! 1. Traverse from the root. At an OR node take the method with the best
//!    score `pb_c(N, N_a) · P(a) / Σ P`, where
//!    `pb_c = (ln((N + base + 1) / base) + init) · √N / (N_a + 1)` with
//!    AlphaZero's constants. `N` is the node's visits and `N_a` the method's.
//!    The constants only scale every score of a node alike, so the pick is
//!    the method with the largest `P(a) / (N_a + 1)`: each method gets
//!    visits roughly in proportion to its prior. At an AND node take the
//!    least visited open case (the cases' priors are uniform). Every pick
//!    counts a visit.
//! 2. A method not yet applied (a *virtual* one) is applied when picked. If
//!    it does not apply it is dropped, its prior leaves the sum, and the
//!    next pick is tried.
//! 3. The traversal ends at a system not yet expanded, which is expanded:
//!    rank its `candidate_methods`, compute their priors, and apply the
//!    `TAM_RS_MCGS_TOP_N` with the best priors, in parallel. The others stay
//!    virtual. Integrating stops once the system settles, like TamRL's
//!    expansion stops at the first closing method.
//!
//! Statuses need no backpropagation: [`search_graph::SearchGraph::propagate`]
//! settles every ancestor at once. Visit counts are the only statistics,
//! and they change only along the traversed path.
//!
//! ## Priors
//!
//! `P_i = exp(log_softmax(−w_h · i + bias_i) / T)` for the heuristic's
//! `i`-th candidate. `bias` is the BM25 score of the candidate against the
//! query, z-normalized over the system's candidates and scaled by
//! `TAM_RS_MCGS_IDF_WEIGHT`. The query is the lemma's text and/or the
//! methods applied along the path that first reached the system. When a
//! later traversal reaches an expanded system along a path with fewer
//! methods, its priors are recomputed from that path (TamRL's
//! `maybe_repath`); its descendants keep theirs.
//!
//! The candidates are `candidate_methods`' list as it is, before knowing
//! which apply: TamRL's API filters out the methods that do not apply
//! first, which costs applying every method of every system. A method that
//! turns out not to apply leaves the prior sum when it is tried.
//!
//! ## Dead ends
//!
//! A traversal that neither expands nor applies anything ends at a system
//! with nothing left to pick, which happens when every way down is settled,
//! closed or leads back onto the path (merging creates cycles). Such a step
//! triggers a sweep that marks every open system that can no longer reach
//! anything to expand as dead, so later traversals avoid it. Deadness is
//! permanent: nothing new appears below a system that reaches no frontier.
//! When the root itself is dead, the search is exhausted.
//!
//! ## Output
//!
//! As in the top-N search, the settled graph is turned back into a
//! [`ProofNode`] tree by [`Materializer`]. The search is deterministic: the
//! eager methods of an expansion run in parallel, but their results are
//! integrated in prior order.
//!
//! ## Configuration
//!
//! Opt-in; with `TAM_RS_MCGS_BUDGET` unset the greedy driver runs unchanged.
//! Read once per process ([`mcgs_from_env`]):
//! - `TAM_RS_MCGS_BUDGET=S`: at most S steps per lemma, `0` for no limit
//!   but the deadline; selects this search;
//! - `TAM_RS_MCGS_TOP_N=N`: methods applied when a system is expanded,
//!   default 50, `0` for all;
//! - `TAM_RS_MCGS_TEMPERATURE=T`: default 1;
//! - `TAM_RS_MCGS_HEURISTIC_WEIGHT=W`: the rank bias `w_h`, default 0.3;
//! - `TAM_RS_MCGS_IDF_WEIGHT=W`: the BM25 bias's spread, default 1; `0`
//!   skips BM25;
//! - `TAM_RS_MCGS_IDF_QUERY=formula+path|formula|path`: the BM25 query,
//!   default `formula+path`;
//! - `TAM_RS_MERGE` (presence): merge canonically equal systems (needs
//!   `bliss`), shared with the top-N search;
//! - `TAM_RS_MCGS_STATS` (presence): one summary line per search on stderr.
//! - `TAM_RS_SEARCH_STATS=SECS`: `[search-stats]` lines on stderr, the merge
//!   counters both graph searches share in one format: one every SECS
//!   seconds while the search runs (`0`: none), so that a run killed at a
//!   timeout leaves its last counts, and a last one with `final=1`
//!   ([`search_graph::StatsReporter`]).

use std::sync::OnceLock;
use std::time::{Duration, Instant};

use rayon::prelude::*;

use crate::constraint::solver::bm25;
use crate::constraint::solver::context::ProofContext;
use crate::constraint::solver::proof_method::ProofMethod;
use crate::constraint::solver::reduction::avoid_fresh_state;
use crate::constraint::solver::search::{self, candidate_methods, ProofNode};
use crate::constraint::solver::search_graph::{
    self, apply_method, classify, Applied, Child, ClassId, End, Interned, Materializer,
    MergeCounts, StatsReporter, Status, Stored, WorkStats, WorkerEnv, DEPTH_CAP, ROOT,
};
use crate::constraint::system::System;

// =============================================================================
// Configuration
// =============================================================================

/// What the BM25 query is made of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdfQuery {
    /// The methods applied along the path to the system.
    Path,
    /// The lemma's text.
    Formula,
    FormulaPath,
}

impl IdfQuery {
    fn path(self) -> bool {
        matches!(self, IdfQuery::Path | IdfQuery::FormulaPath)
    }

    fn formula(self) -> bool {
        matches!(self, IdfQuery::Formula | IdfQuery::FormulaPath)
    }
}

/// The MCGS search's settings, stamped onto each lemma's `ProofContext`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct McgsConfig {
    /// At most this many steps; `0` for no limit but the deadline.
    pub budget: u64,
    /// Methods applied when a system is expanded, best prior first; `0` for
    /// all.
    pub top_n: u32,
    pub temperature: f64,
    /// The rank bias per position in the heuristic's order.
    pub heuristic_weight: f64,
    /// The BM25 bias's standard deviation over a system's candidates; `0`
    /// skips BM25.
    pub idf_weight: f64,
    pub idf_query: IdfQuery,
    /// Merge canonically equal systems into one class.
    pub merge: bool,
}

impl McgsConfig {
    /// The defaults, with `budget` steps.
    pub fn with_budget(budget: u64) -> Self {
        McgsConfig {
            budget,
            top_n: 50,
            temperature: 1.0,
            heuristic_weight: 0.3,
            idf_weight: 1.0,
            idf_query: IdfQuery::FormulaPath,
            merge: false,
        }
    }
}

fn env_value<T: std::str::FromStr>(var: &str, default: T, what: &str) -> T {
    match std::env::var(var) {
        Err(std::env::VarError::NotPresent) => default,
        Ok(v) => v
            .parse()
            .unwrap_or_else(|_| panic!("{var}={v:?}: expected {what}")),
        Err(e) => panic!("{var}: {e}"),
    }
}

/// The configuration from the environment, read once per process; `None`
/// keeps the greedy search.
pub fn mcgs_from_env() -> Option<McgsConfig> {
    static CONFIG: OnceLock<Option<McgsConfig>> = OnceLock::new();
    *CONFIG.get_or_init(|| {
        std::env::var_os("TAM_RS_MCGS_BUDGET")?;
        let defaults = McgsConfig::with_budget(0);
        let budget = env_value("TAM_RS_MCGS_BUDGET", 0u64, "a non-negative integer");
        let top_n = env_value(
            "TAM_RS_MCGS_TOP_N",
            defaults.top_n,
            "a non-negative integer",
        );
        let temperature = env_value("TAM_RS_MCGS_TEMPERATURE", defaults.temperature, "a number");
        assert!(
            temperature.is_finite() && temperature > 0.0,
            "TAM_RS_MCGS_TEMPERATURE={temperature}: expected a positive number"
        );
        let heuristic_weight = env_value(
            "TAM_RS_MCGS_HEURISTIC_WEIGHT",
            defaults.heuristic_weight,
            "a number",
        );
        let idf_weight = env_value("TAM_RS_MCGS_IDF_WEIGHT", defaults.idf_weight, "a number");
        assert!(
            heuristic_weight.is_finite() && idf_weight.is_finite(),
            "TAM_RS_MCGS_HEURISTIC_WEIGHT and TAM_RS_MCGS_IDF_WEIGHT must be finite"
        );
        let idf_query = match std::env::var("TAM_RS_MCGS_IDF_QUERY").as_deref() {
            Err(std::env::VarError::NotPresent) | Ok("formula+path") => IdfQuery::FormulaPath,
            Ok("formula") => IdfQuery::Formula,
            Ok("path") => IdfQuery::Path,
            other => panic!(
                "TAM_RS_MCGS_IDF_QUERY={other:?}: expected `formula+path`, `formula` or `path`"
            ),
        };
        Some(McgsConfig {
            budget,
            top_n,
            temperature,
            heuristic_weight,
            idf_weight,
            idf_query,
            merge: tamarin_utils::env_gate!("TAM_RS_MERGE"),
        })
    })
}

/// [`mcgs_from_env`] for one lemma of a theory, checked against the top-N
/// search and against what merging needs up front.
pub fn config_for_lemma() -> Option<McgsConfig> {
    let config = mcgs_from_env()?;
    assert!(
        super::topn_search::top_n_from_env().is_none(),
        "TAM_RS_MCGS_BUDGET and TAM_RS_TOP_METHODS select two different searches; set one"
    );
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
// Per-class statistics and selection
// =============================================================================

/// AlphaZero's `pb_c_base` and `pb_c_init`, as TamRL's paper runs use them.
/// Under prior-only selection they scale all scores of a node alike.
const PB_C_BASE: f64 = 3200.0;
const PB_C_INIT: f64 = 0.001;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActionState {
    /// Not applied yet.
    Virtual,
    /// Applied: the class's method of this index.
    Applied(u16),
    /// Tried; it does not apply.
    NotApplicable,
}

/// One candidate method of a class, by its rank in the heuristic's order.
#[derive(Debug, Clone, Copy)]
struct Action {
    /// Unnormalized; the score divides by the sum over the live actions.
    prior: f64,
    state: ActionState,
}

/// What the MCGS search keeps per class.
#[derive(Debug, Default)]
struct McgsExt {
    /// Its candidates were ranked and its first methods applied.
    expanded: bool,
    /// `candidate_methods`, in the heuristic's order.
    ranked: Vec<ProofMethod>,
    /// Per candidate, its BM25 document; empty with BM25 off.
    docs: Vec<Vec<String>>,
    /// Per candidate.
    actions: Vec<Action>,
    /// Per applied method (the class's `ands`): its candidate's rank.
    and_rank: Vec<usize>,
    /// Per applied method: its visits.
    and_visits: Vec<u32>,
    /// Per applied method, per case: the case's visits.
    case_visits: Vec<Vec<u32>>,
    /// Methods on the path the priors were computed from.
    path_len: u32,
    /// Open, but nothing to expand is reachable from it any more.
    dead: bool,
}

impl McgsExt {
    /// TamRL's OR-node visit count: its methods' visits, plus one for its
    /// expansion.
    fn visits(&self) -> u32 {
        self.and_visits.iter().sum::<u32>() + u32::from(self.expanded)
    }

    fn prior_sum(&self) -> f64 {
        self.actions
            .iter()
            .filter(|a| a.state != ActionState::NotApplicable)
            .map(|a| a.prior)
            .sum()
    }

    fn has_virtual(&self) -> bool {
        self.actions.iter().any(|a| a.state == ActionState::Virtual)
    }
}

type SearchGraph = search_graph::SearchGraph<McgsExt>;

/// TamRL's prior-only score (`ucb_score` with `deactivate_model`).
fn score(parent_visits: u32, visits: u32, prior: f64, prior_sum: f64) -> f64 {
    let n = f64::from(parent_visits);
    let pb_c = (((n + PB_C_BASE + 1.0) / PB_C_BASE).ln() + PB_C_INIT) * n.sqrt()
        / (f64::from(visits) + 1.0);
    pb_c * prior / prior_sum
}

/// A class no traversal should enter: settled, dead, closed without an open
/// method that has cases, or already on the path.
fn enterable(g: &SearchGraph, d: ClassId, on_path: &[ClassId]) -> bool {
    let class = g.class(d);
    let stuck = class.closed
        && !class
            .ands
            .iter()
            .any(|and| and.status == Status::Open && !and.cases.is_empty());
    class.status == Status::Open && !class.ext.dead && !stuck && !on_path.contains(&d)
}

fn method_selectable(g: &SearchGraph, c: ClassId, a: usize, on_path: &[ClassId]) -> bool {
    let and = &g.class(c).ands[a];
    and.status == Status::Open && and.cases.iter().any(|&(_, d)| enterable(g, d, on_path))
}

/// A pick at an OR node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pick {
    /// An applied method.
    Method(u16),
    /// A virtual method, by rank.
    Virtual(usize),
}

/// The picks at class `c`, best first. Ties keep TamRL's order: applied
/// methods in the order applied, then virtual ones by prior.
fn ranked_picks(g: &SearchGraph, c: ClassId, on_path: &[ClassId]) -> Vec<Pick> {
    let ext = &g.class(c).ext;
    let visits = ext.visits();
    let sum = ext.prior_sum();
    let mut scored: Vec<(f64, Pick)> = Vec::new();
    for a in 0..g.class(c).ands.len() {
        if method_selectable(g, c, a, on_path) {
            let prior = ext.actions[ext.and_rank[a]].prior;
            scored.push((
                score(visits, ext.and_visits[a], prior, sum),
                Pick::Method(a as u16),
            ));
        }
    }
    for i in by_prior(&ext.actions, |a| a.state == ActionState::Virtual) {
        scored.push((
            score(visits, 0, ext.actions[i].prior, sum),
            Pick::Virtual(i),
        ));
    }
    // Stable: equal scores keep the order above.
    scored.sort_by(|x, y| y.0.total_cmp(&x.0));
    scored.into_iter().map(|(_, pick)| pick).collect()
}

/// The ranks of the actions that satisfy `keep`, best prior first; equal
/// priors in the heuristic's order.
fn by_prior(actions: &[Action], keep: impl Fn(&Action) -> bool) -> Vec<usize> {
    let mut ranks: Vec<usize> = (0..actions.len()).filter(|&i| keep(&actions[i])).collect();
    ranks.sort_by(|&x, &y| actions[y].prior.total_cmp(&actions[x].prior));
    ranks
}

/// The case to descend into below method `a` of `c`: the best score under
/// the uniform prior, which is the least visited enterable case, the first
/// in name order among equals.
fn pick_case(g: &SearchGraph, c: ClassId, a: u16, on_path: &[ClassId]) -> Option<usize> {
    let class = g.class(c);
    let and = &class.ands[a as usize];
    let visits = &class.ext.case_visits[a as usize];
    let total: u32 = visits.iter().sum();
    let prior = 1.0 / and.cases.len() as f64;
    let mut best: Option<(f64, usize)> = None;
    for (k, &(_, d)) in and.cases.iter().enumerate() {
        if !enterable(g, d, on_path) {
            continue;
        }
        let s = score(total, visits[k], prior, 1.0);
        if best.map_or(true, |(b, _)| s > b) {
            best = Some((s, k));
        }
    }
    best.map(|(_, k)| k)
}

/// The priors of `n` ranked candidates whose BM25 documents are `docs`,
/// against `query`.
fn priors(config: &McgsConfig, docs: &[Vec<String>], n: usize, query: &[String]) -> Vec<f64> {
    let mut logits: Vec<f64> = (0..n)
        .map(|i| -config.heuristic_weight * i as f64)
        .collect();
    if config.idf_weight != 0.0 {
        let bias = bm25::z_bias(&bm25::bm25_plus_scores(docs, query), config.idf_weight);
        for (l, b) in logits.iter_mut().zip(bias) {
            *l += b;
        }
    }
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let log_sum = max + logits.iter().map(|l| (l - max).exp()).sum::<f64>().ln();
    logits
        .into_iter()
        .map(|l| ((l - log_sum) / config.temperature).exp())
        .collect()
}

// =============================================================================
// The engine
// =============================================================================

#[derive(Debug, Default)]
struct Stats {
    steps: u64,
    expansions: u64,
    materializations: u64,
    not_applicable: u64,
    applied: u64,
    leaves: u64,
    merges: u64,
    repaths: u64,
    /// Steps that neither expanded nor applied anything.
    idle_steps: u64,
    work: WorkStats,
}

/// How applying a virtual method went.
enum Materialized {
    Applied(u16),
    NotApplicable,
    /// The deadline passed; the method stays virtual.
    Aborted,
}

struct Engine<'a> {
    ctx: &'a ProofContext,
    config: McgsConfig,
    graph: SearchGraph,
    /// Indexed by class: its system while it may still get a method.
    store: Vec<Option<Stored>>,
    /// `--bound`: no class at this depth or deeper is expanded.
    bound: u32,
    deadline: Instant,
    /// The lemma's BM25 query tokens.
    formula_query: Vec<String>,
    stats: Stats,
    reporter: StatsReporter,
}

impl<'a> Engine<'a> {
    fn new(ctx: &'a ProofContext, config: McgsConfig, bound: u32, deadline: Instant) -> Self {
        let formula_query = if config.idf_weight != 0.0 && config.idf_query.formula() {
            bm25::lemma_query(&ctx.lemma_text)
        } else {
            Vec::new()
        };
        Engine {
            ctx,
            config,
            graph: SearchGraph::new(),
            store: Vec::new(),
            bound,
            deadline,
            formula_query,
            stats: Stats::default(),
            reporter: StatsReporter::new("mcgs"),
        }
    }

    fn merge_counts(&self) -> MergeCounts {
        MergeCounts {
            applied: self.stats.applied,
            leaves: self.stats.leaves,
            merges: self.stats.merges,
        }
    }

    /// A `[search-stats]` progress line, if one is due.
    fn report_progress(&mut self) {
        let counts = self.merge_counts();
        self.reporter
            .tick(&self.ctx.lemma_name, &self.graph, &counts, &self.stats.work);
    }

    fn add_root(&mut self, sys: System) {
        let mut work = WorkStats::default();
        let child = if self.bound == 0 {
            Child::Cut
        } else {
            classify(self.ctx, sys, self.config.merge, &mut work)
        };
        self.stats.work.add(&work);
        let root = self.intern(child, 0);
        assert_eq!(root, ROOT, "the root is the first class");
    }

    /// The class of a prepared case reached at `depth`.
    fn intern(&mut self, child: Child, depth: u32) -> ClassId {
        let (c, interned) = self.graph.intern(child, depth);
        let stored = match interned {
            Interned::Cut => {
                self.graph.class_mut(c).ext.dead = true;
                None
            }
            Interned::Leaf => {
                self.stats.leaves += 1;
                None
            }
            Interned::New(stored) => Some(stored),
            Interned::Merged => {
                self.stats.merges += 1;
                let class = self.graph.class_mut(c);
                class.min_depth = class.min_depth.min(depth);
                return c;
            }
        };
        assert_eq!(self.store.len(), c as usize, "classes are stored in order");
        self.store.push(stored);
        c
    }

    /// The BM25 query for a system reached along `path`.
    fn query(&self, path: &[(ClassId, u16)]) -> Vec<String> {
        let mut query = Vec::new();
        if self.config.idf_weight == 0.0 {
            return query;
        }
        if self.config.idf_query.path() {
            for &(p, a) in path {
                let ext = &self.graph.class(p).ext;
                query.extend(ext.docs[ext.and_rank[a as usize]].iter().cloned());
            }
        }
        if self.config.idf_query.formula() {
            query.extend(self.formula_query.iter().cloned());
        }
        query
    }

    fn cut_children(&self, c: ClassId) -> bool {
        self.graph.class(c).min_depth + 1 >= self.bound
    }

    /// Adds an applied method of `c`, the candidate of rank `rank`, with its
    /// cases.
    fn integrate(&mut self, c: ClassId, rank: usize, applied: Applied) -> u16 {
        let Applied {
            method,
            method_fp,
            cases,
        } = applied;
        self.stats.applied += 1;
        let a = self.graph.add_and(c, method, method_fp);
        let ext = &mut self.graph.class_mut(c).ext;
        assert_eq!(
            ext.and_rank.len(),
            a as usize,
            "methods are recorded in order"
        );
        ext.actions[rank].state = ActionState::Applied(a);
        ext.and_rank.push(rank);
        ext.and_visits.push(0);
        ext.case_visits.push(vec![0; cases.len()]);
        let depth = self.graph.class(c).min_depth + 1;
        for (name, child) in cases {
            let d = self.intern(child, depth);
            self.graph.add_case(c, a, name, d);
        }
        self.graph.propagate(vec![(c, a)]);
        a
    }

    /// Records that `c`'s virtual method `rank` was tried and does not apply.
    fn drop_action(&mut self, c: ClassId, rank: usize) {
        self.stats.not_applicable += 1;
        self.graph.class_mut(c).ext.actions[rank].state = ActionState::NotApplicable;
    }

    /// After `c`'s methods changed: a settled class needs its system no
    /// more, and one without virtual methods gets no further method.
    fn tidy(&mut self, c: ClassId) {
        let class = self.graph.class(c);
        if class.status != Status::Open {
            self.store[c as usize] = None;
        } else if !class.ext.has_virtual() && !class.closed {
            self.store[c as usize] = None;
            self.graph.close(c);
        }
    }

    /// Ranks `c`'s candidates, computes their priors from `path`, and
    /// applies the best `top_n` concurrently.
    fn expand(&mut self, env: &WorkerEnv, c: ClassId, path: &[(ClassId, u16)]) {
        self.stats.expansions += 1;
        let depth = self.graph.class(c).min_depth;
        let cut = self.cut_children(c);
        let merge = self.config.merge;
        let use_bm25 = self.config.idf_weight != 0.0;
        let stored = self.store[c as usize]
            .as_ref()
            .expect("an unexpanded open class keeps its system");
        let (ranked, docs) = env.run(self.ctx, avoid_fresh_state(&stored.sys), |worker_ctx| {
            let ranked = candidate_methods(&stored.sys, worker_ctx, depth as usize);
            let docs: Vec<Vec<String>> = if use_bm25 {
                ranked
                    .iter()
                    .map(|m| bm25::tokens(&bm25::method_text(m, &stored.sys)))
                    .collect()
            } else {
                Vec::new()
            };
            (ranked, docs)
        });
        let priors = priors(&self.config, &docs, ranked.len(), &self.query(path));
        let actions: Vec<Action> = priors
            .into_iter()
            .map(|prior| Action {
                prior,
                state: ActionState::Virtual,
            })
            .collect();
        let mut eager = by_prior(&actions, |_| true);
        if self.config.top_n > 0 {
            eager.truncate(self.config.top_n as usize);
        }
        let tasks: Vec<(usize, ProofMethod)> =
            eager.iter().map(|&i| (i, ranked[i].clone())).collect();
        let ext = &mut self.graph.class_mut(c).ext;
        ext.expanded = true;
        ext.ranked = ranked;
        ext.docs = docs;
        ext.actions = actions;
        ext.path_len = path.len() as u32;

        let results: Vec<(usize, Option<Applied>, WorkStats)> = {
            let ctx = self.ctx;
            let stored = self.store[c as usize].as_ref().expect("kept above");
            if tasks.len() <= 1 {
                tasks
                    .into_iter()
                    .map(|(i, method)| {
                        let (applied, work) = run_method(ctx, env, stored, &method, cut, merge);
                        (i, applied, work)
                    })
                    .collect()
            } else {
                // A `System` may move between threads but not be shared:
                // each task gets its own copy.
                let forked: Vec<(usize, ProofMethod, Stored)> = tasks
                    .into_iter()
                    .map(|(i, method)| (i, method, stored.fork()))
                    .collect();
                forked
                    .into_par_iter()
                    .map(|(i, method, stored)| {
                        let (applied, work) = run_method(ctx, env, &stored, &method, cut, merge);
                        (i, applied, work)
                    })
                    .collect()
            }
        };
        for (rank, applied, work) in results {
            self.stats.work.add(&work);
            if self.graph.class(c).status != Status::Open {
                // Settled by an earlier method: the rest stay virtual and
                // are never picked.
                continue;
            }
            match applied {
                Some(applied) => {
                    self.integrate(c, rank, applied);
                }
                None if Instant::now() >= self.deadline => {}
                None => self.drop_action(c, rank),
            }
        }
        self.tidy(c);
    }

    /// Applies `c`'s virtual method `rank`.
    fn materialize(&mut self, env: &WorkerEnv, c: ClassId, rank: usize) -> Materialized {
        self.stats.materializations += 1;
        let cut = self.cut_children(c);
        let method = self.graph.class(c).ext.ranked[rank].clone();
        let stored = self.store[c as usize]
            .as_ref()
            .expect("a class with a virtual method keeps its system");
        let (applied, work) = run_method(self.ctx, env, stored, &method, cut, self.config.merge);
        self.stats.work.add(&work);
        let outcome = match applied {
            Some(applied) => Materialized::Applied(self.integrate(c, rank, applied)),
            None if Instant::now() >= self.deadline => return Materialized::Aborted,
            None => {
                self.drop_action(c, rank);
                Materialized::NotApplicable
            }
        };
        self.tidy(c);
        outcome
    }

    /// TamRL's `maybe_repath`: recomputes `c`'s priors when `path` reaches
    /// it with fewer methods than the path they were computed from.
    fn maybe_repath(&mut self, c: ClassId, path: &[(ClassId, u16)]) {
        if self.config.idf_weight == 0.0 || !self.config.idf_query.path() {
            return;
        }
        if path.len() as u32 >= self.graph.class(c).ext.path_len {
            return;
        }
        self.stats.repaths += 1;
        let query = self.query(path);
        let ext = &self.graph.class(c).ext;
        let fresh = priors(&self.config, &ext.docs, ext.actions.len(), &query);
        let ext = &mut self.graph.class_mut(c).ext;
        for (action, prior) in ext.actions.iter_mut().zip(fresh) {
            action.prior = prior;
        }
        ext.path_len = path.len() as u32;
    }

    /// The method to descend through at `c`, its visit counted; applies
    /// virtual methods as they are picked. `None` when nothing is left to
    /// pick or `c` settled meanwhile.
    fn pick_method(
        &mut self,
        env: &WorkerEnv,
        c: ClassId,
        on_path: &[ClassId],
        progress: &mut bool,
    ) -> Option<u16> {
        loop {
            let a = match *ranked_picks(&self.graph, c, on_path).first()? {
                Pick::Method(a) => a,
                Pick::Virtual(rank) => {
                    *progress = true;
                    match self.materialize(env, c, rank) {
                        Materialized::Aborted => return None,
                        Materialized::NotApplicable => continue,
                        Materialized::Applied(a) => {
                            if self.graph.class(c).status != Status::Open {
                                return None;
                            }
                            if !method_selectable(&self.graph, c, a as usize, on_path) {
                                continue;
                            }
                            a
                        }
                    }
                }
            };
            self.graph.class_mut(c).ext.and_visits[a as usize] += 1;
            return Some(a);
        }
    }

    /// One traversal from the root; whether it expanded or applied anything.
    fn step(&mut self, env: &WorkerEnv) -> bool {
        let mut path: Vec<(ClassId, u16)> = Vec::new();
        let mut on_path = vec![ROOT];
        let mut c = ROOT;
        let mut progress = false;
        while self.graph.class(c).status == Status::Open {
            if !self.graph.class(c).ext.expanded {
                if self.store[c as usize].is_some() {
                    self.expand(env, c, &path);
                    progress = true;
                }
                break;
            }
            self.maybe_repath(c, &path);
            let Some(a) = self.pick_method(env, c, &on_path, &mut progress) else {
                break;
            };
            let k = if self.graph.class(c).ands[a as usize].cases.len() == 1 {
                0
            } else {
                let k = pick_case(&self.graph, c, a, &on_path)
                    .expect("a selectable method has an enterable case");
                self.graph.class_mut(c).ext.case_visits[a as usize][k] += 1;
                k
            };
            path.push((c, a));
            c = self.graph.class(c).ands[a as usize].cases[k].1;
            on_path.push(c);
        }
        progress
    }

    /// Marks every open class that reaches nothing to expand as dead, and
    /// drops the systems of settled classes; whether the root is alive.
    fn sweep(&mut self) -> bool {
        let n = self.graph.classes.len();
        let mut alive = vec![false; n];
        let mut work: Vec<ClassId> = Vec::new();
        for c in 0..n as ClassId {
            let class = self.graph.class(c);
            let frontier = class.status == Status::Open
                && self.store[c as usize].is_some()
                && (!class.ext.expanded || class.ext.has_virtual());
            if frontier {
                alive[c as usize] = true;
                work.push(c);
            }
        }
        while let Some(x) = work.pop() {
            for &(p, a) in &self.graph.class(x).parents {
                let parent = self.graph.class(p);
                if !alive[p as usize]
                    && parent.status == Status::Open
                    && parent.ands[a as usize].status == Status::Open
                {
                    alive[p as usize] = true;
                    work.push(p);
                }
            }
        }
        for c in 0..n as ClassId {
            if self.graph.class(c).status != Status::Open {
                self.store[c as usize] = None;
            } else if !alive[c as usize] {
                self.graph.class_mut(c).ext.dead = true;
            }
        }
        alive[ROOT as usize]
    }

    fn search(&mut self, env: &WorkerEnv) -> End {
        loop {
            if self.graph.class(ROOT).status != Status::Open {
                return End::Settled;
            }
            if Instant::now() >= self.deadline {
                return End::Deadline;
            }
            if self.config.budget > 0 && self.stats.steps >= self.config.budget {
                return End::Budget;
            }
            self.stats.steps += 1;
            self.report_progress();
            if !self.step(env) {
                self.stats.idle_steps += 1;
                if self.graph.class(ROOT).status == Status::Open
                    && Instant::now() < self.deadline
                    && !self.sweep()
                {
                    return End::Exhausted;
                }
            }
        }
    }
}

/// Applies `method` to `stored`'s system in a worker environment.
fn run_method(
    ctx: &ProofContext,
    env: &WorkerEnv,
    stored: &Stored,
    method: &ProofMethod,
    cut: bool,
    merge: bool,
) -> (Option<Applied>, WorkStats) {
    let start = Instant::now();
    let mut work = WorkStats::default();
    let applied = env.run(ctx, avoid_fresh_state(&stored.sys), |worker_ctx| {
        apply_method(worker_ctx, stored, method, cut, merge, &mut work)
    });
    work.busy = start.elapsed();
    (applied, work)
}

// =============================================================================
// Entry point
// =============================================================================

/// Runs the MCGS search on `initial` and returns the proof tree, in the
/// shape the greedy driver returns it. Called by `search::run_proof_search`
/// when the lemma's context carries a [`McgsConfig`].
pub fn run(
    ctx: &ProofContext,
    initial: System,
    proof_bound: usize,
    config: McgsConfig,
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
    engine.reporter.finish(
        &ctx.lemma_name,
        &engine.graph,
        &engine.merge_counts(),
        &engine.stats.work,
        end,
    );

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
        label: "MCGS",
    };
    let root = materializer.root(ctx, initial);
    let materialized = materialize_start.elapsed();
    search::replace_deadline(previous_deadline);

    if tamarin_utils::env_gate!("TAM_RS_MCGS_STATS") {
        print_stats(ctx, &config, &engine, end, &root, searched, materialized);
    }
    root
}

fn print_stats(
    ctx: &ProofContext,
    config: &McgsConfig,
    engine: &Engine<'_>,
    end: End,
    root: &ProofNode,
    searched: Duration,
    materialized: Duration,
) {
    let s = &engine.stats;
    // Classes reached along two or more edges: what merging saved a second
    // expansion of.
    let shared = engine
        .graph
        .classes
        .iter()
        .filter(|c| c.parents.len() > 1)
        .count();
    let dead = engine.graph.classes.iter().filter(|c| c.ext.dead).count();
    eprintln!(
        "[mcgs] lemma={} budget={} top_n={} temperature={} heuristic_weight={} idf_weight={} \
         idf_query={:?} merge={} end={:?} root={:?} steps={} idle_steps={} expansions={} \
         classes={} leaves={} merges={} shared={} dead={} applied={} materializations={} \
         not_applicable={} repaths={} execs={} canons={} canon_s={:.3} busy_s={:.3} \
         search_s={:.3} materialize_s={:.3}",
        ctx.lemma_name,
        config.budget,
        config.top_n,
        config.temperature,
        config.heuristic_weight,
        config.idf_weight,
        config.idf_query,
        config.merge,
        end,
        root.status,
        s.steps,
        s.idle_steps,
        s.expansions,
        engine.graph.classes.len(),
        s.leaves,
        s.merges,
        shared,
        dead,
        s.applied,
        s.materializations,
        s.not_applicable,
        s.repaths,
        s.work.execs,
        s.work.canons,
        s.work.canon_time.as_secs_f64(),
        s.work.busy.as_secs_f64(),
        searched.as_secs_f64(),
        materialized.as_secs_f64(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::solver::search::{run_proof_search, NodeStatus};
    use crate::constraint::solver::search_graph::test_support::{
        ctx, empty_disjunction, two_branch_disjunction, two_true_formulas, unproducible_action,
    };
    use crate::constraint::solver::search_graph::Settled;
    use crate::pretty_theory::pretty_proof_body;

    // --- Selection (no maude) ---------------------------------------------

    /// An expanded class whose candidates have `priors`, all virtual.
    fn expanded(g: &mut SearchGraph, priors: &[f64]) -> ClassId {
        let c = g.add_class(Status::Open, 0);
        let ext = &mut g.class_mut(c).ext;
        ext.expanded = true;
        ext.actions = priors
            .iter()
            .map(|&prior| Action {
                prior,
                state: ActionState::Virtual,
            })
            .collect();
        c
    }

    fn open(g: &mut SearchGraph) -> ClassId {
        g.add_class(Status::Open, 0)
    }

    /// Applies `c`'s candidate `rank` with `children` as its cases.
    fn apply(g: &mut SearchGraph, c: ClassId, rank: usize, children: &[ClassId]) -> u16 {
        let a = g.add_and(c, ProofMethod::Simplify, None);
        let ext = &mut g.class_mut(c).ext;
        ext.actions[rank].state = ActionState::Applied(a);
        ext.and_rank.push(rank);
        ext.and_visits.push(0);
        ext.case_visits.push(vec![0; children.len()]);
        for (i, &d) in children.iter().enumerate() {
            g.add_case(c, a, format!("case_{i}"), d);
        }
        g.propagate(vec![(c, a)]);
        a
    }

    #[test]
    fn an_or_node_spreads_its_visits_in_proportion_to_the_priors() {
        let mut g = SearchGraph::new();
        let c = expanded(&mut g, &[0.6, 0.4]);
        let (x, y) = (open(&mut g), open(&mut g));
        apply(&mut g, c, 0, &[x]);
        apply(&mut g, c, 1, &[y]);
        assert_eq!(ranked_picks(&g, c, &[c])[0], Pick::Method(0));
        g.class_mut(c).ext.and_visits = vec![1, 0];
        assert_eq!(
            ranked_picks(&g, c, &[c])[0],
            Pick::Method(1),
            "0.6/2 < 0.4/1"
        );
        g.class_mut(c).ext.and_visits = vec![1, 1];
        assert_eq!(
            ranked_picks(&g, c, &[c])[0],
            Pick::Method(0),
            "0.6/2 > 0.4/2"
        );
    }

    #[test]
    fn virtual_methods_follow_applied_ones_among_equals_and_leave_when_they_do_not_apply() {
        let mut g = SearchGraph::new();
        let c = expanded(&mut g, &[0.5, 0.5, 0.2]);
        let x = open(&mut g);
        apply(&mut g, c, 0, &[x]);
        assert_eq!(
            ranked_picks(&g, c, &[c]),
            vec![Pick::Method(0), Pick::Virtual(1), Pick::Virtual(2)]
        );
        assert!((g.class(c).ext.prior_sum() - 1.2).abs() < 1e-12);
        g.class_mut(c).ext.actions[1].state = ActionState::NotApplicable;
        assert_eq!(
            ranked_picks(&g, c, &[c]),
            vec![Pick::Method(0), Pick::Virtual(2)]
        );
        assert!((g.class(c).ext.prior_sum() - 0.7).abs() < 1e-12);
        // A better prior wins over an applied method.
        let d = expanded(&mut g, &[0.2, 0.8]);
        let z = open(&mut g);
        apply(&mut g, d, 0, &[z]);
        assert_eq!(ranked_picks(&g, d, &[d])[0], Pick::Virtual(1));
    }

    #[test]
    fn an_and_node_takes_the_least_visited_enterable_case() {
        let mut g = SearchGraph::new();
        let c = expanded(&mut g, &[1.0]);
        let (x, y, z) = (open(&mut g), open(&mut g), open(&mut g));
        let a = apply(&mut g, c, 0, &[x, y, z]);
        assert_eq!(
            pick_case(&g, c, a, &[c]),
            Some(0),
            "all unvisited: name order"
        );
        g.class_mut(c).ext.case_visits[0] = vec![1, 0, 0];
        assert_eq!(pick_case(&g, c, a, &[c]), Some(1));
        assert_eq!(pick_case(&g, c, a, &[c, y]), Some(2), "y is on the path");
        g.class_mut(z).ext.dead = true;
        assert_eq!(pick_case(&g, c, a, &[c, y]), Some(0), "z is dead");
    }

    #[test]
    fn settled_dead_closed_and_on_path_classes_are_never_entered() {
        let mut g = SearchGraph::new();
        let c = expanded(&mut g, &[1.0, 1.0, 1.0, 1.0]);
        let unfinishable = g.add_class(Status::Settled(Settled::Unfinishable), 1);
        let dead = open(&mut g);
        g.class_mut(dead).ext.dead = true;
        let stuck = open(&mut g);
        g.close(stuck);
        apply(&mut g, c, 0, &[unfinishable]);
        apply(&mut g, c, 1, &[dead]);
        apply(&mut g, c, 2, &[stuck]);
        apply(&mut g, c, 3, &[c]);
        assert_eq!(g.class(c).status, Status::Open);
        assert!(ranked_picks(&g, c, &[c]).is_empty());
    }

    #[test]
    fn priors_follow_the_rank_and_lean_towards_lexical_matches() {
        let config = McgsConfig {
            idf_weight: 0.0,
            ..McgsConfig::with_budget(0)
        };
        let p = priors(&config, &[], 3, &[]);
        assert!(p[0] > p[1] && p[1] > p[2]);
        assert!(
            (p.iter().sum::<f64>() - 1.0).abs() < 1e-12,
            "a softmax at T = 1"
        );
        assert!(((p[0] / p[1]).ln() - 0.3).abs() < 1e-12);
        let hot = McgsConfig {
            temperature: 2.0,
            ..config
        };
        let q = priors(&hot, &[], 3, &[]);
        assert!(
            ((q[0] / q[1]).ln() - 0.15).abs() < 1e-12,
            "the bias divided by T"
        );

        let docs = vec![vec!["a".to_string()], vec!["b".into()], vec!["c".into()]];
        let lexical = McgsConfig {
            heuristic_weight: 0.0,
            idf_weight: 1.0,
            ..config
        };
        let p = priors(&lexical, &docs, 3, &["c".to_string()]);
        assert!(p[2] > p[0] && (p[0] - p[1]).abs() < 1e-12);
        let p = priors(&lexical, &docs, 3, &["z".to_string()]);
        assert!((p[0] - p[2]).abs() < 1e-12, "no overlap: no preference");
    }

    /// The priors of `x` were computed from a path of three methods; a path
    /// of one recomputes them from its query, a longer one leaves them.
    #[test]
    fn a_shorter_path_recomputes_the_priors() {
        let Some(ctx) = ctx() else {
            return;
        };
        let config = McgsConfig {
            heuristic_weight: 0.0,
            idf_query: IdfQuery::Path,
            ..McgsConfig::with_budget(0)
        };
        let mut engine = Engine::new(&ctx, config, DEPTH_CAP, Instant::now());
        let root = expanded(&mut engine.graph, &[1.0]);
        let x = expanded(&mut engine.graph, &[0.5, 0.5]);
        engine.graph.class_mut(root).ext.docs = vec![vec!["b".into()]];
        apply(&mut engine.graph, root, 0, &[x]);
        let ext = &mut engine.graph.class_mut(x).ext;
        ext.docs = vec![vec!["a".into()], vec!["b".into()]];
        ext.path_len = 3;

        engine.maybe_repath(x, &[(root, 0), (root, 0), (root, 0), (root, 0)]);
        assert_eq!(
            engine.graph.class(x).ext.actions[1].prior,
            0.5,
            "a longer path"
        );
        engine.maybe_repath(x, &[(root, 0)]);
        let ext = &engine.graph.class(x).ext;
        assert_eq!(ext.path_len, 1);
        assert!(
            ext.actions[1].prior > ext.actions[0].prior,
            "the path's `b` favors the second method"
        );
    }

    // --- Against the greedy driver (maude) -------------------------------

    fn mcgs(top_n: u32, merge: bool) -> McgsConfig {
        McgsConfig {
            top_n,
            merge,
            ..McgsConfig::with_budget(500)
        }
    }

    /// A verdict the greedy search reaches, MCGS reaches too, for one or all
    /// methods per expansion, with and without merging, and the same proof
    /// on a second run.
    fn assert_reaches_the_greedy_verdict(make: impl Fn() -> System, bound: usize) {
        let Some(mut ctx) = ctx() else {
            return;
        };
        ctx.lemma_text = std::sync::Arc::from("lemma l: \"All x #i. Out(x) @ #i ==> F\"");
        let greedy = run_proof_search(&ctx, make(), bound);
        let merges: &[bool] = if crate::bliss_proc::bliss_available() {
            &[false, true]
        } else {
            &[false]
        };
        for &merge in merges {
            for top_n in [1, 0] {
                ctx.mcgs = Some(mcgs(top_n, merge));
                let first = run_proof_search(&ctx, make(), bound);
                if matches!(
                    greedy.status,
                    NodeStatus::Solved | NodeStatus::Contradictory
                ) {
                    assert_eq!(first.status, greedy.status, "top_n={top_n} merge={merge}");
                }
                let second = run_proof_search(&ctx, make(), bound);
                assert_eq!(
                    pretty_proof_body(&first),
                    pretty_proof_body(&second),
                    "deterministic: top_n={top_n} merge={merge}"
                );
            }
        }
    }

    #[test]
    fn an_empty_disjunction_closes_like_the_greedy_search() {
        assert_reaches_the_greedy_verdict(empty_disjunction, 5);
    }

    #[test]
    fn a_two_branch_disjunction_finds_the_trace_like_the_greedy_search() {
        assert_reaches_the_greedy_verdict(two_branch_disjunction, 10);
    }

    #[test]
    fn simplify_then_solved_like_the_greedy_search() {
        assert_reaches_the_greedy_verdict(two_true_formulas, 5);
    }

    /// With `--bound 1` the greedy search's `simplify` ends in `sorry`; MCGS
    /// gives up on it and closes the system with the second-ranked method,
    /// whether that one runs at the expansion or later.
    #[test]
    fn a_lower_ranked_method_closes_what_the_first_leaves_open() {
        let Some(mut ctx) = ctx() else {
            return;
        };
        let greedy = run_proof_search(&ctx, unproducible_action(), 1);
        assert_eq!(greedy.status, NodeStatus::Sorry);
        for top_n in [1, 0] {
            ctx.mcgs = Some(mcgs(top_n, false));
            let proof = run_proof_search(&ctx, unproducible_action(), 1);
            assert_eq!(proof.status, NodeStatus::Contradictory, "top_n={top_n}");
            assert!(
                matches!(proof.method, ProofMethod::SolveGoal(_)),
                "the proof uses the second-ranked method, got {:?}",
                proof.method
            );
            assert!(proof.children.is_empty(), "the goal has no producing rule");
        }
    }

    #[test]
    fn the_budget_stops_the_search() {
        let Some(mut ctx) = ctx() else {
            return;
        };
        ctx.mcgs = Some(McgsConfig {
            top_n: 1,
            ..McgsConfig::with_budget(1)
        });
        let proof = run_proof_search(&ctx, unproducible_action(), 10);
        assert!(
            matches!(proof.method, ProofMethod::Simplify),
            "one step applies the best-ranked method only, got {:?}",
            proof.method
        );
        assert_ne!(proof.status, NodeStatus::Contradictory);
    }
}
