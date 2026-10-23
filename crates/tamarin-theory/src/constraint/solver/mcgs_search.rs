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
//! ## One round
//!
//! A round runs up to `TAM_RS_MCGS_BATCH` traversals (B, default 1), one
//! after another, and then expands the systems they end at together.
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
//! 3. The traversal ends at a system not yet expanded and claims it. A later
//!    traversal of the round never enters a claimed system, and the visits
//!    the earlier ones counted already lead it elsewhere.
//! 4. With merging, the claimed systems are keyed (see below).
//! 5. The claimed systems are expanded in parallel: each ranks its
//!    `candidate_methods`, computes their priors, and applies the
//!    `TAM_RS_MCGS_TOP_N` with the best priors, in parallel too. The others
//!    stay virtual. The expansions are integrated in the order their
//!    traversals ran; integrating stops once a system settles, like TamRL's
//!    expansion stops at the first closing method.
//!
//! With B = 1 a round is one traversal and one expansion. With a larger B a
//! traversal sees the visits and claims of the round's earlier traversals,
//! but not their expansions.
//!
//! Statuses need no backpropagation: [`search_graph::SearchGraph::propagate`]
//! settles every ancestor at once. Visit counts are the only statistics,
//! and they change only along the traversed path.
//!
//! ## Merging
//!
//! With `TAM_RS_MERGE`, a case is canonicalized when a traversal first
//! enters it, not when its method is applied: an expansion applies up to
//! `TAM_RS_MCGS_TOP_N` methods at once, and most of their cases are never
//! entered. A case whose canonical form is an existing class's merges into
//! that class ([`search_graph::SearchGraph::merge_into`]): its edge leads
//! there, and the traversal goes on from there. Until a case is entered, its
//! class is one system, so a case equal to a settled class or to a system on
//! the path is found to be one only when entered.
//!
//! `TAM_RS_MCGS_KEYING=eager` restores the keying before just-in-time
//! keying, for ablations: every case is canonicalized when its method is
//! applied, in the worker that applies it, and a case whose canonical form
//! is an existing class's merges into it right away, so its system is never
//! stored. That costs a canonicalization for every case, entered or not, and
//! saves the memory of the duplicates no traversal entered yet.
//!
//! `TAM_RS_MCGS_SHADOW_KEYS` measures merging without doing it: with merging
//! off, a claimed system is canonicalized at the point just-in-time keying
//! would key it, and a key seen before marks it a duplicate (`dup_of` in
//! the graph dumps, `shadow_dups`); nothing merges, so the search is the
//! unmerged one.
//!
//! A round keys its claimed systems in passes. A pass canonicalizes the
//! claimed systems without a key in parallel, then decides for each, in
//! traversal order, whether it is a new class or merges. Two claimed systems
//! with one key are thus one class, expanded once: the second merges into
//! the first, which is claimed, so its traversal ends. A traversal whose
//! system merged into a class it may enter goes on from there and may claim
//! another system, which the next pass keys.
//!
//! What merging saved shows in the `[search-stats]` lines as `expanded`,
//! the classes the search expanded, against `unfolded`, their occurrences
//! once the graph is unfolded into a tree ([`search_graph::unfold`]): what a
//! search without merging would have to expand to reach the same systems
//! along every path the graph has to them. It is not what the unmerged
//! search expands, which makes other choices. Under eager keying the tree
//! also has the cases merged when their method was applied that no
//! traversal entered.
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
//! [`ProofNode`] tree by [`Materializer`]. The search is deterministic. The
//! claimed systems of a round are keyed and expanded in parallel, and an
//! expansion's eager methods run in parallel, but each task depends only on
//! its own system (its maude handle's counter is seeded from it,
//! [`WorkerEnv`]), and every result is integrated in a fixed order:
//! traversal order, then prior order. So the outcome depends on B but not on
//! the thread count or on timing, the deadline apart.
//!
//! ## Configuration
//!
//! Opt-in; with `TAM_RS_MCGS_BUDGET` unset the greedy driver runs unchanged.
//! Read once per process ([`mcgs_from_env`]):
//! - `TAM_RS_MCGS_BUDGET=S`: at most S traversals per lemma, `0` for no
//!   limit but the deadline; selects this search;
//! - `TAM_RS_MCGS_BATCH=B`: traversals per round, default 1, the sequential
//!   search;
//! - `TAM_RS_MCGS_TOP_N=N`: methods applied when a system is expanded,
//!   default 50, `0` for all;
//! - `TAM_RS_MCGS_TEMPERATURE=T`: default 1;
//! - `TAM_RS_MCGS_HEURISTIC_WEIGHT=W`: the rank bias `w_h`, default 0.3;
//! - `TAM_RS_MCGS_IDF_WEIGHT=W`: the BM25 bias's spread, default 1; `0`
//!   skips BM25;
//! - `TAM_RS_MCGS_IDF_QUERY=formula+path|formula|path`: the BM25 query,
//!   default `formula+path`;
//! - `TAM_RS_MCGS_MAX_METHODS=K`: at most K methods per system, the first
//!   K that apply in prior order, default `0` for no cap. With K = 1 the
//!   search walks a single proof tree, the one the prior's first choices
//!   make, and only chooses which case to descend into;
//! - `TAM_RS_MCGS_CASE_ORDER=visits|first`: the case to descend into at an
//!   AND node: the least visited (TamRL's uniform case prior, the default)
//!   or the first open one in name order, which with K = 1 is a depth-first
//!   walk of that tree;
//! - `TAM_RS_MERGE` (presence): merge canonically equal systems (needs
//!   `bliss`), shared with the top-N search;
//! - `TAM_RS_MCGS_KEYING=jit|eager`: with merging, canonicalize a case when
//!   a traversal first enters it (the default) or when its method is applied
//!   (see "Merging");
//! - `TAM_RS_MCGS_SHADOW_KEYS` (presence): without merging, canonicalize
//!   claimed systems and count the duplicates, without merging them;
//! - `TAM_RS_MCGS_STATS` (presence): one summary line per search on stderr.
//! - `TAM_RS_SEARCH_STATS=SECS`: `[search-stats]` lines on stderr, the merge
//!   counters both graph searches share in one format: one every SECS
//!   seconds while the search runs (`0`: none), so that a run killed at a
//!   timeout leaves its last counts, and a last one with `final=1`
//!   ([`search_graph::StatsReporter`]).

use std::sync::OnceLock;
use std::time::{Duration, Instant};

use rayon::prelude::*;

use tamarin_term::alpha_eq_ac::CanonLabelling;

use crate::canon_fingerprint::CanonicalSystemFingerprint;
use crate::constraint::solver::bm25;
use crate::constraint::solver::context::ProofContext;
use crate::constraint::solver::proof_method::ProofMethod;
use crate::constraint::solver::reduction::avoid_fresh_state;
use crate::constraint::solver::search::{self, candidate_methods, ProofNode};
use crate::constraint::solver::search_graph::{
    self, apply_method, canonical_key, classify, Applied, Child, ClassId, End, Interned,
    Materializer, MergeCounts, StatsReporter, Status, Stored, WorkStats, WorkerEnv, DEPTH_CAP,
    ROOT,
};
use crate::constraint::system::System;
use crate::prove::ProveError;
use tamarin_utils::FastMap;

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

/// Which open case of an applied method a traversal descends into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaseOrder {
    /// The least visited: TamRL's uniform prior over the cases.
    Visits,
    /// The first in name order.
    First,
}

/// When a case is canonicalized, with merging on (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Keying {
    /// When a traversal first enters it.
    Jit,
    /// When its method is applied.
    Eager,
}

/// The MCGS search's settings, stamped onto each lemma's `ProofContext`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct McgsConfig {
    /// At most this many traversals; `0` for no limit but the deadline.
    pub budget: u64,
    /// Traversals per round, whose systems are keyed and expanded together.
    pub batch: usize,
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
    /// At most this many methods per class, the first that apply in prior
    /// order; `0` for no cap.
    pub max_methods: u32,
    pub case_order: CaseOrder,
    /// Merge canonically equal systems into one class.
    pub merge: bool,
    /// With merging: when a case is canonicalized.
    pub keying: Keying,
    /// Without merging: canonicalize claimed systems and count duplicates.
    pub shadow_keys: bool,
}

impl McgsConfig {
    /// Whether cases are canonicalized when their method is applied.
    fn key_cases(&self) -> bool {
        self.merge && self.keying == Keying::Eager
    }

    /// The defaults, with `budget` steps.
    pub fn with_budget(budget: u64) -> Self {
        McgsConfig {
            budget,
            batch: 1,
            top_n: 50,
            temperature: 1.0,
            heuristic_weight: 0.3,
            idf_weight: 1.0,
            idf_query: IdfQuery::FormulaPath,
            max_methods: 0,
            case_order: CaseOrder::Visits,
            merge: false,
            keying: Keying::Jit,
            shadow_keys: false,
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
        let batch = env_value("TAM_RS_MCGS_BATCH", defaults.batch, "a positive integer");
        assert!(
            batch >= 1,
            "TAM_RS_MCGS_BATCH=0: expected a positive integer"
        );
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
        let max_methods = env_value(
            "TAM_RS_MCGS_MAX_METHODS",
            defaults.max_methods,
            "a non-negative integer",
        );
        let case_order = match std::env::var("TAM_RS_MCGS_CASE_ORDER").as_deref() {
            Err(std::env::VarError::NotPresent) | Ok("visits") => CaseOrder::Visits,
            Ok("first") => CaseOrder::First,
            other => panic!("TAM_RS_MCGS_CASE_ORDER={other:?}: expected `visits` or `first`"),
        };
        let keying = match std::env::var("TAM_RS_MCGS_KEYING").as_deref() {
            Err(std::env::VarError::NotPresent) | Ok("jit") => Keying::Jit,
            Ok("eager") => Keying::Eager,
            other => panic!("TAM_RS_MCGS_KEYING={other:?}: expected `jit` or `eager`"),
        };
        Some(McgsConfig {
            budget,
            batch,
            top_n,
            temperature,
            heuristic_weight,
            idf_weight,
            idf_query,
            max_methods,
            case_order,
            merge: tamarin_utils::env_gate!("TAM_RS_MERGE"),
            keying,
            shadow_keys: tamarin_utils::env_gate!("TAM_RS_MCGS_SHADOW_KEYS"),
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
    assert!(
        !(config.merge && config.shadow_keys),
        "TAM_RS_MCGS_SHADOW_KEYS measures the search without merging; unset TAM_RS_MERGE"
    );
    if config.merge || config.shadow_keys {
        // Panics itself, with install hints, unless TAM_ALLOW_NO_BLISS=1.
        assert!(
            crate::bliss_proc::bliss_available(),
            "TAM_RS_MERGE and TAM_RS_MCGS_SHADOW_KEYS need a working `bliss` executable"
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
    /// [`McgsConfig::max_methods`]: no method is applied beyond this many.
    max_methods: u32,
    /// The class this case merged into when a traversal entered it: it has
    /// no edge since.
    merged_into: Option<ClassId>,
    /// Claimed by a traversal of the current round, which expands it.
    in_flight: bool,
    /// Per applied method, per case: the case merged into an open class
    /// when the method was applied (eager keying), and no traversal has
    /// gone through it yet.
    pending_merge: Vec<Vec<bool>>,
    /// CPU seconds spent applying the class's methods: what expanding it
    /// cost, eager keying's canonicalizations included.
    expand_s: f64,
    /// With shadow keys: the earlier claimed class with the same key.
    dup_of: Option<ClassId>,
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

    /// Whether a virtual method may still be applied.
    fn has_virtual(&self) -> bool {
        !self.capped() && self.actions.iter().any(|a| a.state == ActionState::Virtual)
    }

    /// The class has as many methods as [`McgsConfig::max_methods`] allows.
    fn capped(&self) -> bool {
        self.max_methods > 0 && self.and_rank.len() >= self.max_methods as usize
    }
}

impl search_graph::DumpExt for McgsExt {
    fn class_fields(&self, row: &mut serde_json::Map<String, serde_json::Value>) {
        use serde_json::json;
        row.insert("expanded".into(), json!(self.expanded));
        row.insert("dead".into(), json!(self.dead));
        row.insert("merged_into".into(), json!(self.merged_into));
        row.insert("expand_s".into(), json!(self.expand_s));
        row.insert("dup_of".into(), json!(self.dup_of));
        row.insert("visits".into(), json!(self.visits()));
        row.insert("path_len".into(), json!(self.path_len));
        // Per candidate, in the heuristic's order: its prior, and `V`
        // (virtual), `N` (does not apply) or the index of its applied method.
        row.insert(
            "priors".into(),
            json!(self.actions.iter().map(|a| a.prior).collect::<Vec<_>>()),
        );
        row.insert(
            "states".into(),
            json!(self
                .actions
                .iter()
                .map(|a| match a.state {
                    ActionState::Virtual => "V".to_string(),
                    ActionState::NotApplicable => "N".to_string(),
                    ActionState::Applied(i) => i.to_string(),
                })
                .collect::<Vec<_>>()),
        );
    }

    fn and_fields(&self, a: usize, row: &mut serde_json::Map<String, serde_json::Value>) {
        use serde_json::json;
        let rank = self.and_rank[a];
        row.insert("rank".into(), json!(rank));
        row.insert("prior".into(), json!(self.actions[rank].prior));
        row.insert("visits".into(), json!(self.and_visits[a]));
        row.insert("case_visits".into(), json!(self.case_visits[a]));
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
/// method that has cases, already on the path, or claimed by another
/// traversal of the round.
fn enterable(g: &SearchGraph, d: ClassId, on_path: &[ClassId]) -> bool {
    let class = g.class(d);
    let stuck = class.closed
        && !class
            .ands
            .iter()
            .any(|and| and.status == Status::Open && !and.cases.is_empty());
    class.status == Status::Open
        && !class.ext.dead
        && !class.ext.in_flight
        && !stuck
        && !on_path.contains(&d)
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
    let virtuals = if ext.capped() {
        Vec::new()
    } else {
        by_prior(&ext.actions, |a| a.state == ActionState::Virtual)
    };
    for i in virtuals {
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

/// The case to descend into below method `a` of `c`: with
/// [`CaseOrder::Visits`] the best score under the uniform prior, which is the
/// least visited enterable case, the first in name order among equals; with
/// [`CaseOrder::First`] the first enterable case.
fn pick_case(
    g: &SearchGraph,
    c: ClassId,
    a: u16,
    on_path: &[ClassId],
    order: CaseOrder,
) -> Option<usize> {
    let class = g.class(c);
    let and = &class.ands[a as usize];
    if order == CaseOrder::First {
        return and
            .cases
            .iter()
            .position(|&(_, d)| enterable(g, d, on_path));
    }
    let visits = &class.ext.case_visits[a as usize];
    let total: u32 = visits.iter().sum();
    let prior = 1.0 / and.cases.len() as f64;
    let mut best: Option<(f64, usize)> = None;
    for (k, &(_, d)) in and.cases.iter().enumerate() {
        if !enterable(g, d, on_path) {
            continue;
        }
        let s = score(total, visits[k], prior, 1.0);
        if best.is_none_or(|(b, _)| s > b) {
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

/// The priors the search gives `ranked`, the candidates of `sys` in the
/// heuristic's order, when it expands `sys` reached along `path`: each
/// method applied on the way down with the system it was applied to. Lets
/// the proof profiler (`examples/profile_reference_proof.rs`) compare a
/// proof's choices with what this search would prefer at the same system.
pub fn candidate_priors(
    config: &McgsConfig,
    lemma_text: &str,
    sys: &System,
    ranked: &[ProofMethod],
    path: &[(&ProofMethod, &System)],
) -> Vec<f64> {
    let (docs, query) = bm25_inputs(config, lemma_text, sys, ranked, path);
    priors(config, &docs, ranked.len(), &query)
}

/// The BM25 bias of [`candidate_priors`] at weight 1: the z-normalized BM25
/// scores of `ranked` against the query, which the prior scales by
/// `idf_weight` and adds to the rank bias. Lets a caller recompute the prior
/// for other weights and temperatures.
pub fn candidate_bm25_z(
    config: &McgsConfig,
    lemma_text: &str,
    sys: &System,
    ranked: &[ProofMethod],
    path: &[(&ProofMethod, &System)],
) -> Vec<f64> {
    let config = McgsConfig {
        idf_weight: 1.0,
        ..*config
    };
    let (docs, query) = bm25_inputs(&config, lemma_text, sys, ranked, path);
    bm25::z_bias(&bm25::bm25_plus_scores(&docs, &query), 1.0)
}

/// The candidates' BM25 documents and the query, as the search builds them.
fn bm25_inputs(
    config: &McgsConfig,
    lemma_text: &str,
    sys: &System,
    ranked: &[ProofMethod],
    path: &[(&ProofMethod, &System)],
) -> (Vec<Vec<String>>, Vec<String>) {
    let mut docs = Vec::new();
    let mut query = Vec::new();
    if config.idf_weight != 0.0 {
        docs = ranked
            .iter()
            .map(|m| bm25::tokens(&bm25::method_text(m, sys)))
            .collect();
        if config.idf_query.path() {
            for (m, s) in path {
                query.extend(bm25::tokens(&bm25::method_text(m, s)));
            }
        }
        if config.idf_query.formula() {
            query.extend(bm25::lemma_query(lemma_text));
        }
    }
    (docs, query)
}

// =============================================================================
// The engine
// =============================================================================

#[derive(Debug, Default)]
struct Stats {
    /// Traversals.
    steps: u64,
    expansions: u64,
    materializations: u64,
    not_applicable: u64,
    applied: u64,
    leaves: u64,
    merges: u64,
    /// See [`MergeCounts`].
    merges_used: u64,
    merges_settled: u64,
    merges_on_path: u64,
    shadow_dups: u64,
    key_wall: Duration,
    repaths: u64,
    /// Traversals that neither claimed a system nor applied anything.
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

/// One traversal of a round.
struct Traversal {
    /// The edges taken: a class and its method.
    path: Vec<(ClassId, u16)>,
    /// The classes entered, the root first.
    on_path: Vec<ClassId>,
    end: TraversalEnd,
    /// It claimed a system or applied a method.
    progress: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TraversalEnd {
    /// The unexpanded class it claimed.
    Leaf(ClassId),
    /// Nothing left to pick along its path.
    Done,
}

impl Traversal {
    fn from_root() -> Self {
        Traversal {
            path: Vec::new(),
            on_path: vec![ROOT],
            end: TraversalEnd::Done,
            progress: false,
        }
    }
}

/// What expanding a claimed class takes from the graph.
struct LeafTask {
    c: ClassId,
    stored: Stored,
    depth: u32,
    cut: bool,
    /// The BM25 query of the path that reached it.
    query: Vec<String>,
    path_len: u32,
}

/// A claimed class's expansion, as a worker computed it.
struct Expansion {
    c: ClassId,
    stored: Stored,
    path_len: u32,
    ranked: Vec<ProofMethod>,
    docs: Vec<Vec<String>>,
    actions: Vec<Action>,
    /// Per eager method, best prior first: its rank and how applying it went.
    results: Vec<(usize, Result<Option<Applied>, ProveError>, WorkStats)>,
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
    /// With shadow keys: each key seen, and the first class that had it.
    shadow: FastMap<CanonicalSystemFingerprint, ClassId>,
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
            shadow: FastMap::default(),
            stats: Stats::default(),
            reporter: StatsReporter::new("mcgs"),
        }
    }

    fn merge_counts(&self) -> MergeCounts {
        let (stored, stored_unkeyed) =
            search_graph::stored_counts(&self.graph, &self.store, self.config.merge);
        MergeCounts {
            applied: self.stats.applied,
            leaves: self.stats.leaves,
            merges: self.stats.merges,
            // No sleep sets.
            skipped_asleep: 0,
            asleep: 0,
            merges_used: self.stats.merges_used,
            merges_settled: self.stats.merges_settled,
            merges_on_path: self.stats.merges_on_path,
            shadow_dups: self.stats.shadow_dups,
            stored,
            stored_unkeyed,
            key_wall: self.stats.key_wall,
            unfolding: Some(search_graph::unfold(&self.graph, |class| class.ext.expanded)),
        }
    }

    /// A `[search-stats]` progress line, if one is due.
    fn report_progress(&mut self) {
        if !self.reporter.due() {
            return;
        }
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
        self.intern_case(child, depth).0
    }

    /// [`Self::intern`], and whether the case merged into an existing class,
    /// which only an eagerly keyed case does.
    fn intern_case(&mut self, child: Child, depth: u32) -> (ClassId, bool) {
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
            Interned::Merged if self.config.key_cases() => {
                self.stats.merges += 1;
                let class = self.graph.class_mut(c);
                class.min_depth = class.min_depth.min(depth);
                return (c, true);
            }
            Interned::Merged => unreachable!(
                "MCGS keys a case when a traversal enters it; only the root, the first \
                 class, is keyed when created"
            ),
        };
        assert_eq!(self.store.len(), c as usize, "classes are stored in order");
        self.store.push(stored);
        (c, false)
    }

    /// Gives `c`, a claimed class without a key, the key `key` its system
    /// `stored` canonicalizes to. If no class has it, it is `c`'s, and `c`
    /// keeps the labelling for its methods. Otherwise `c` merges into the
    /// class that has it, which is returned.
    fn settle_key(
        &mut self,
        c: ClassId,
        mut stored: Stored,
        key: CanonicalSystemFingerprint,
        labelling: CanonLabelling,
    ) -> Option<ClassId> {
        let Some(existing) = self.graph.key(c, key) else {
            stored.labelling = Some(labelling);
            self.store[c as usize] = Some(stored);
            return None;
        };
        let ext = &mut self.graph.class_mut(c).ext;
        ext.in_flight = false;
        ext.merged_into = Some(existing);
        self.graph.merge_into(c, existing);
        self.stats.merges += 1;
        self.stats.merges_used += 1;
        if self.graph.class(existing).status != Status::Open {
            self.stats.merges_settled += 1;
        }
        Some(existing)
    }

    /// Keys the systems the round's traversals claimed, with merging on (see
    /// the module docs), in passes: each canonicalizes the claimed systems
    /// without a key in parallel, then settles their keys in traversal
    /// order. A traversal whose system merged goes on from the class it
    /// merged into, if it may enter it, and may claim another system, which
    /// the next pass keys. Each pass merges or keys every system it takes,
    /// and none creates a class, so the passes end.
    fn key_leaves(&mut self, env: &WorkerEnv, ts: &mut [Traversal]) -> Result<(), ProveError> {
        if !self.config.merge {
            if self.config.shadow_keys {
                self.shadow_key_leaves(ts);
            }
            return Ok(());
        }
        loop {
            let unkeyed: Vec<(usize, ClassId)> = ts
                .iter()
                .enumerate()
                .filter_map(|(i, t)| match t.end {
                    TraversalEnd::Leaf(c) if !self.graph.class(c).keyed => Some((i, c)),
                    _ => None,
                })
                .collect();
            if unkeyed.is_empty() {
                return Ok(());
            }
            // A `System` may move between threads but not be shared: each
            // task takes its claimed system out of the store.
            let owned: Vec<Stored> = unkeyed
                .iter()
                .map(|&(_, c)| {
                    self.store[c as usize]
                        .take()
                        .expect("a claimed class keeps its system")
                })
                .collect();
            let pass = Instant::now();
            let keyed = map_in_order(owned, |stored| {
                let start = Instant::now();
                let mut work = WorkStats::default();
                let (key, labelling) = canonical_key(&stored.sys, &mut work);
                work.busy = start.elapsed();
                (stored, key, labelling, work)
            });
            self.stats.key_wall += pass.elapsed();
            let mut merged = Vec::new();
            for (&(i, c), (stored, key, labelling, work)) in unkeyed.iter().zip(keyed) {
                self.stats.work.add(&work);
                if let Some(existing) = self.settle_key(c, stored, key, labelling) {
                    merged.push((i, existing));
                }
            }
            for (i, existing) in merged {
                let t = &mut ts[i];
                t.on_path.pop();
                t.end = TraversalEnd::Done;
                if t.on_path.contains(&existing) {
                    self.stats.merges_on_path += 1;
                }
                if enterable(&self.graph, existing, &t.on_path) {
                    t.on_path.push(existing);
                    self.select_from(env, t)?;
                }
            }
        }
    }

    /// Shadow keys: canonicalizes the round's claimed systems, in parallel,
    /// and marks each whose key an earlier claimed system had as its
    /// duplicate. Nothing merges, and every system goes back to its class.
    fn shadow_key_leaves(&mut self, ts: &[Traversal]) {
        let claimed: Vec<ClassId> = ts
            .iter()
            .filter_map(|t| match t.end {
                TraversalEnd::Leaf(c) => Some(c),
                TraversalEnd::Done => None,
            })
            .collect();
        let owned: Vec<Stored> = claimed
            .iter()
            .map(|&c| {
                self.store[c as usize]
                    .take()
                    .expect("a claimed class keeps its system")
            })
            .collect();
        let pass = Instant::now();
        let keyed = map_in_order(owned, |stored| {
            let mut work = WorkStats::default();
            let (key, _) = canonical_key(&stored.sys, &mut work);
            (stored, key, work)
        });
        self.stats.key_wall += pass.elapsed();
        for (c, (stored, key, work)) in claimed.into_iter().zip(keyed) {
            self.stats.work.add(&work);
            self.store[c as usize] = Some(stored);
            match self.shadow.get(&key) {
                Some(&first) => {
                    self.graph.class_mut(c).ext.dup_of = Some(first);
                    self.stats.shadow_dups += 1;
                }
                None => {
                    self.shadow.insert(key, c);
                }
            }
        }
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
        let mut pending = Vec::with_capacity(cases.len());
        for (name, child) in cases {
            let (d, merged) = self.intern_case(child, depth);
            // A merge into a settled class takes effect now: the case is
            // closed. One into an open class is used once a traversal goes
            // through it.
            let settled = merged && self.graph.class(d).status != Status::Open;
            if settled {
                self.stats.merges_used += 1;
                self.stats.merges_settled += 1;
            }
            pending.push(merged && !settled);
            self.graph.add_case(c, a, name, d);
        }
        self.graph.class_mut(c).ext.pending_merge.push(pending);
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

    /// Expands the classes the round's traversals claimed, concurrently (see
    /// [`expand_leaf`]), and integrates them in traversal order. Fails with
    /// the first error in that order, whichever worker meets one first.
    fn expand_leaves(&mut self, env: &WorkerEnv, ts: &[Traversal]) -> Result<(), ProveError> {
        let mut tasks = Vec::new();
        for t in ts {
            let TraversalEnd::Leaf(c) = t.end else {
                continue;
            };
            assert!(
                !self.config.merge || self.graph.class(c).keyed,
                "class #{c} is expanded unkeyed: its methods need its labelling"
            );
            tasks.push(LeafTask {
                c,
                stored: self.store[c as usize]
                    .take()
                    .expect("a claimed class keeps its system"),
                depth: self.graph.class(c).min_depth,
                cut: self.cut_children(c),
                query: self.query(&t.path),
                path_len: t.path.len() as u32,
            });
        }
        let (ctx, config) = (self.ctx, self.config);
        let expansions = map_in_order(tasks, |task| expand_leaf(ctx, env, &config, task));
        for expansion in expansions {
            self.integrate_expansion(expansion?)?;
        }
        Ok(())
    }

    /// Records `expansion`'s ranking and adds its applied methods in prior
    /// order, until the class settles.
    fn integrate_expansion(&mut self, expansion: Expansion) -> Result<(), ProveError> {
        let Expansion {
            c,
            stored,
            path_len,
            ranked,
            docs,
            actions,
            results,
        } = expansion;
        self.stats.expansions += 1;
        let ext = &mut self.graph.class_mut(c).ext;
        ext.expanded = true;
        ext.in_flight = false;
        ext.ranked = ranked;
        ext.docs = docs;
        ext.actions = actions;
        ext.path_len = path_len;
        ext.max_methods = self.config.max_methods;
        self.store[c as usize] = Some(stored);
        for (rank, applied, work) in results {
            self.stats.work.add(&work);
            self.graph.class_mut(c).ext.expand_s += work.busy.as_secs_f64();
            let applied = applied?;
            if self.graph.class(c).status != Status::Open {
                // Settled by an earlier method: the rest stay virtual and
                // are never picked.
                continue;
            }
            match applied {
                // Past the cap the method stays virtual, and is never picked.
                Some(_) if self.graph.class(c).ext.capped() => {}
                Some(applied) => {
                    self.integrate(c, rank, applied);
                }
                None if Instant::now() >= self.deadline => {}
                None => self.drop_action(c, rank),
            }
        }
        self.tidy(c);
        Ok(())
    }

    /// Applies `c`'s virtual method `rank`.
    fn materialize(
        &mut self,
        env: &WorkerEnv,
        c: ClassId,
        rank: usize,
    ) -> Result<Materialized, ProveError> {
        self.stats.materializations += 1;
        let cut = self.cut_children(c);
        let method = self.graph.class(c).ext.ranked[rank].clone();
        let stored = self.store[c as usize]
            .as_ref()
            .expect("a class with a virtual method keeps its system");
        let (applied, work) =
            run_method(self.ctx, env, stored, &method, cut, self.config.key_cases());
        self.stats.work.add(&work);
        self.graph.class_mut(c).ext.expand_s += work.busy.as_secs_f64();
        let outcome = match applied? {
            Some(applied) => Materialized::Applied(self.integrate(c, rank, applied)),
            None if Instant::now() >= self.deadline => return Ok(Materialized::Aborted),
            None => {
                self.drop_action(c, rank);
                Materialized::NotApplicable
            }
        };
        self.tidy(c);
        Ok(outcome)
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
    ) -> Result<Option<u16>, ProveError> {
        loop {
            let Some(&pick) = ranked_picks(&self.graph, c, on_path).first() else {
                return Ok(None);
            };
            let a = match pick {
                Pick::Method(a) => a,
                Pick::Virtual(rank) => {
                    *progress = true;
                    match self.materialize(env, c, rank)? {
                        Materialized::Aborted => return Ok(None),
                        Materialized::NotApplicable => continue,
                        Materialized::Applied(a) => {
                            if self.graph.class(c).status != Status::Open {
                                return Ok(None);
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
            return Ok(Some(a));
        }
    }

    /// Continues `t` from the last class it entered down to an unexpanded
    /// class with a system, which it claims, or to where nothing is left to
    /// pick.
    fn select_from(&mut self, env: &WorkerEnv, t: &mut Traversal) -> Result<(), ProveError> {
        let mut c = *t.on_path.last().expect("a traversal starts at a class");
        t.end = TraversalEnd::Done;
        while self.graph.class(c).status == Status::Open {
            let ext = &self.graph.class(c).ext;
            if !ext.expanded {
                if self.store[c as usize].is_some() && !ext.in_flight {
                    self.graph.class_mut(c).ext.in_flight = true;
                    t.end = TraversalEnd::Leaf(c);
                    t.progress = true;
                }
                return Ok(());
            }
            self.maybe_repath(c, &t.path);
            let Some(a) = self.pick_method(env, c, &t.on_path, &mut t.progress)? else {
                return Ok(());
            };
            let k = if self.graph.class(c).ands[a as usize].cases.len() == 1 {
                0
            } else {
                let k = pick_case(&self.graph, c, a, &t.on_path, self.config.case_order)
                    .expect("a selectable method has an enterable case");
                self.graph.class_mut(c).ext.case_visits[a as usize][k] += 1;
                k
            };
            let pending = &mut self.graph.class_mut(c).ext.pending_merge[a as usize][k];
            if *pending {
                *pending = false;
                self.stats.merges_used += 1;
            }
            t.path.push((c, a));
            c = self.graph.class(c).ands[a as usize].cases[k].1;
            t.on_path.push(c);
        }
        Ok(())
    }

    /// One round (see the module docs): up to `batch` traversals within the
    /// budget, then their systems keyed and expanded together. Whether any
    /// traversal claimed a system or applied a method.
    fn round(&mut self, env: &WorkerEnv) -> Result<bool, ProveError> {
        let mut ts = Vec::new();
        while ts.len() < self.config.batch
            && (self.config.budget == 0 || self.stats.steps < self.config.budget)
            && self.graph.class(ROOT).status == Status::Open
        {
            self.stats.steps += 1;
            let mut t = Traversal::from_root();
            self.select_from(env, &mut t)?;
            ts.push(t);
        }
        self.key_leaves(env, &mut ts)?;
        self.expand_leaves(env, &ts)?;
        let idle = ts.iter().filter(|t| !t.progress).count();
        self.stats.idle_steps += idle as u64;
        Ok(idle < ts.len())
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

    fn search(&mut self, env: &WorkerEnv) -> Result<End, ProveError> {
        loop {
            if self.graph.class(ROOT).status != Status::Open {
                return Ok(End::Settled);
            }
            if Instant::now() >= self.deadline {
                return Ok(End::Deadline);
            }
            if self.config.budget > 0 && self.stats.steps >= self.config.budget {
                return Ok(End::Budget);
            }
            self.report_progress();
            // An idle round triggers a sweep (see the module docs).
            if !self.round(env)?
                && self.graph.class(ROOT).status == Status::Open
                && Instant::now() < self.deadline
                && !self.sweep()
            {
                return Ok(End::Exhausted);
            }
        }
    }
}

/// Expands a claimed class off the search thread: ranks its candidates,
/// computes their priors from the query of its path, and applies the best
/// `top_n` concurrently, each to its own copy of the system.
fn expand_leaf(
    ctx: &ProofContext,
    env: &WorkerEnv,
    config: &McgsConfig,
    task: LeafTask,
) -> Result<Expansion, ProveError> {
    let LeafTask {
        c,
        stored,
        depth,
        cut,
        query,
        path_len,
    } = task;
    let use_bm25 = config.idf_weight != 0.0;
    let (ranked, docs) = env.run(ctx, avoid_fresh_state(&stored.sys), |worker_ctx| {
        let ranked = candidate_methods(&stored.sys, worker_ctx, depth as usize)?;
        let docs: Vec<Vec<String>> = if use_bm25 {
            ranked
                .iter()
                .map(|m| bm25::tokens(&bm25::method_text(m, &stored.sys)))
                .collect()
        } else {
            Vec::new()
        };
        Ok::<_, ProveError>((ranked, docs))
    })?;
    let priors = priors(config, &docs, ranked.len(), &query);
    let actions: Vec<Action> = priors
        .into_iter()
        .map(|prior| Action {
            prior,
            state: ActionState::Virtual,
        })
        .collect();
    let mut eager = by_prior(&actions, |_| true);
    if config.top_n > 0 {
        eager.truncate(config.top_n as usize);
    }
    let tasks: Vec<(usize, ProofMethod)> = eager.iter().map(|&i| (i, ranked[i].clone())).collect();
    let results = if tasks.len() <= 1 {
        tasks
            .into_iter()
            .map(|(i, method)| {
                let (applied, work) =
                    run_method(ctx, env, &stored, &method, cut, config.key_cases());
                (i, applied, work)
            })
            .collect()
    } else {
        // A `System` may move between threads but not be shared: each task
        // gets its own copy.
        let forked: Vec<(usize, ProofMethod, Stored)> = tasks
            .into_iter()
            .map(|(i, method)| (i, method, stored.fork()))
            .collect();
        forked
            .into_par_iter()
            .map(|(i, method, stored)| {
                let (applied, work) =
                    run_method(ctx, env, &stored, &method, cut, config.key_cases());
                (i, applied, work)
            })
            .collect()
    };
    Ok(Expansion {
        c,
        stored,
        path_len,
        ranked,
        docs,
        actions,
        results,
    })
}

/// Maps `items` through `f`, concurrently when there are several; the
/// results keep the items' order.
fn map_in_order<T: Send, R: Send>(items: Vec<T>, f: impl Fn(T) -> R + Sync + Send) -> Vec<R> {
    if items.len() <= 1 {
        items.into_iter().map(f).collect()
    } else {
        items.into_par_iter().map(f).collect()
    }
}

/// Applies `method` to `stored`'s system in a worker environment. With
/// `key_cases` (eager keying) its cases are canonicalized here; otherwise
/// they stay unkeyed, and [`Engine::key_leaves`] keys a case when a
/// traversal claims it.
fn run_method(
    ctx: &ProofContext,
    env: &WorkerEnv,
    stored: &Stored,
    method: &ProofMethod,
    cut: bool,
    key_cases: bool,
) -> (Result<Option<Applied>, ProveError>, WorkStats) {
    let start = Instant::now();
    let mut work = WorkStats::default();
    let applied = env.run(ctx, avoid_fresh_state(&stored.sys), |worker_ctx| {
        apply_method(worker_ctx, stored, method, cut, key_cases, &mut work)
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
        &engine.merge_counts(),
        &engine.stats.work,
        end,
    );

    // Materializing runs methods again; the deadline the search may have hit
    // must not cut that short.
    let no_deadline = search::capture_search_tls(None);
    let materialize_start = Instant::now();
    let materializer = Materializer {
        graph: &engine.graph,
        merge: config.merge,
        bound,
        end,
        env: WorkerEnv::capture(None),
        label: "MCGS",
    };
    let root = {
        let _tls = search::SearchTlsGuard::install(no_deadline);
        materializer.root(ctx, initial)?
    };
    let materialized = materialize_start.elapsed();

    if tamarin_utils::env_gate!("TAM_RS_MCGS_STATS") {
        print_stats(ctx, &config, &engine, end, &root, searched, materialized);
    }
    Ok(root)
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
    // Merging on: the classes with a key, and the cases no traversal entered
    // yet, which hold their systems unkeyed.
    let keyed = engine.graph.by_key.len();
    let unkeyed = engine
        .graph
        .classes
        .iter()
        .zip(&engine.store)
        .filter(|(c, stored)| config.merge && !c.keyed && stored.is_some())
        .count();
    let (_, peak_rss) = search_graph::rss_mb();
    eprintln!(
        "[mcgs] lemma={} budget={} batch={} keying={:?} shadow_keys={} top_n={} temperature={} \
         heuristic_weight={} idf_weight={} \
         idf_query={:?} max_methods={} case_order={:?} merge={} end={:?} root={:?} steps={} \
         idle_steps={} expansions={} \
         classes={} keyed={} unkeyed={} leaves={} merges={} merges_used={} merges_settled={} \
         merges_on_path={} shadow_dups={} shared={} dead={} applied={} \
         materializations={} not_applicable={} repaths={} execs={} canons={} canon_s={:.3} \
         key_wall_s={:.3} busy_s={:.3} search_s={:.3} materialize_s={:.3} peak_rss_mb={}",
        ctx.lemma_name,
        config.budget,
        config.batch,
        config.keying,
        config.shadow_keys,
        config.top_n,
        config.temperature,
        config.heuristic_weight,
        config.idf_weight,
        config.idf_query,
        config.max_methods,
        config.case_order,
        config.merge,
        end,
        root.status,
        s.steps,
        s.idle_steps,
        s.expansions,
        engine.graph.classes.len(),
        keyed,
        unkeyed,
        s.leaves,
        s.merges,
        s.merges_used,
        s.merges_settled,
        s.merges_on_path,
        s.shadow_dups,
        shared,
        dead,
        s.applied,
        s.materializations,
        s.not_applicable,
        s.repaths,
        s.work.execs,
        s.work.canons,
        s.work.canon_time.as_secs_f64(),
        s.key_wall.as_secs_f64(),
        s.work.busy.as_secs_f64(),
        searched.as_secs_f64(),
        materialized.as_secs_f64(),
        peak_rss.map_or("-".to_string(), |v| v.to_string()),
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
        ext.pending_merge.push(vec![false; children.len()]);
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
            pick_case(&g, c, a, &[c], CaseOrder::Visits),
            Some(0),
            "all unvisited: name order"
        );
        g.class_mut(c).ext.case_visits[0] = vec![1, 0, 0];
        assert_eq!(pick_case(&g, c, a, &[c], CaseOrder::Visits), Some(1));
        assert_eq!(
            pick_case(&g, c, a, &[c, y], CaseOrder::Visits),
            Some(2),
            "y is on the path"
        );
        g.class_mut(z).ext.dead = true;
        assert_eq!(
            pick_case(&g, c, a, &[c, y], CaseOrder::Visits),
            Some(0),
            "z is dead"
        );
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
    /// methods per expansion, without merging, with merging under either
    /// keying, with shadow keys, one or four traversals per round, and the
    /// same proof on a second run.
    fn assert_reaches_the_greedy_verdict(make: impl Fn() -> System, bound: usize) {
        let Some(mut ctx) = ctx() else {
            return;
        };
        ctx.lemma_text = std::sync::Arc::from("lemma l: \"All x #i. Out(x) @ #i ==> F\"");
        let greedy = run_proof_search(&ctx, make(), bound).expect("proof search");
        let bliss = crate::bliss_proc::bliss_available();
        // (merge, keying, shadow keys)
        let mut modes = vec![(false, Keying::Jit, false)];
        if bliss {
            modes.push((true, Keying::Jit, false));
            modes.push((true, Keying::Eager, false));
            modes.push((false, Keying::Jit, true));
        }
        for &(merge, keying, shadow_keys) in &modes {
            for top_n in [1, 0] {
                for batch in [1, 4] {
                    let setting = format!(
                        "top_n={top_n} merge={merge} keying={keying:?} shadow={shadow_keys} \
                         batch={batch}"
                    );
                    ctx.mcgs = Some(McgsConfig {
                        batch,
                        keying,
                        shadow_keys,
                        ..mcgs(top_n, merge)
                    });
                    let first = run_proof_search(&ctx, make(), bound).expect("proof search");
                    if matches!(
                        greedy.status,
                        NodeStatus::Solved | NodeStatus::Contradictory
                    ) {
                        assert_eq!(first.status, greedy.status, "{setting}");
                    }
                    let second = run_proof_search(&ctx, make(), bound).expect("proof search");
                    assert_eq!(
                        pretty_proof_body(&first),
                        pretty_proof_body(&second),
                        "deterministic: {setting}"
                    );
                }
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
        let greedy = run_proof_search(&ctx, unproducible_action(), 1).expect("proof search");
        assert_eq!(greedy.status, NodeStatus::Sorry);
        for top_n in [1, 0] {
            ctx.mcgs = Some(mcgs(top_n, false));
            let proof = run_proof_search(&ctx, unproducible_action(), 1).expect("proof search");
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
        let proof = run_proof_search(&ctx, unproducible_action(), 10).expect("proof search");
        assert!(
            matches!(proof.method, ProofMethod::Simplify),
            "one step applies the best-ranked method only, got {:?}",
            proof.method
        );
        assert_ne!(proof.status, NodeStatus::Contradictory);
    }

    // --- Rounds (maude) ----------------------------------------------------

    /// An engine whose root, `two_branch_disjunction`, is expanded with one
    /// applied method whose cases are `cases`, created unkeyed at depth 1.
    fn engine_with_cases(
        ctx: &ProofContext,
        config: McgsConfig,
        cases: Vec<System>,
    ) -> (Engine<'_>, u16, Vec<ClassId>) {
        let deadline = Instant::now() + Duration::from_secs(60);
        let mut engine = Engine::new(ctx, config, DEPTH_CAP, deadline);
        engine.add_root(two_branch_disjunction());
        let ext = &mut engine.graph.class_mut(ROOT).ext;
        ext.expanded = true;
        ext.actions = vec![Action {
            prior: 1.0,
            state: ActionState::Virtual,
        }];
        let classes: Vec<ClassId> = cases
            .into_iter()
            .map(|sys| engine.intern(Child::Unkeyed(Stored::new(sys, None)), 1))
            .collect();
        let a = apply(&mut engine.graph, ROOT, 0, &classes);
        (engine, a, classes)
    }

    /// The case a traversal claimed is not enterable for the next traversal
    /// of the round.
    #[test]
    fn a_claimed_class_is_not_claimed_again_in_its_round() {
        let Some(ctx) = ctx() else {
            return;
        };
        let config = McgsConfig {
            idf_weight: 0.0,
            batch: 2,
            ..McgsConfig::with_budget(0)
        };
        let (mut engine, _, cases) = engine_with_cases(&ctx, config, vec![unproducible_action()]);
        let env = WorkerEnv::capture(None);
        let mut first = Traversal::from_root();
        engine.select_from(&env, &mut first).expect("select");
        let mut second = Traversal::from_root();
        engine.select_from(&env, &mut second).expect("select");
        assert_eq!(first.end, TraversalEnd::Leaf(cases[0]));
        assert_eq!(second.end, TraversalEnd::Done, "the only case is claimed");
        assert!(!second.progress);
    }

    // --- Merging (maude, bliss) ------------------------------------------

    /// Cases are created unkeyed. Two claimed cases with one key are keyed
    /// in one pass: the first keeps the key, the second merges into it, and
    /// its traversal ends, since the first is claimed.
    #[test]
    fn two_claimed_cases_with_one_key_are_one_class() {
        let Some(ctx) = ctx() else {
            return;
        };
        if !crate::bliss_proc::bliss_available() {
            return;
        }
        let config = McgsConfig {
            merge: true,
            idf_weight: 0.0,
            batch: 2,
            ..McgsConfig::with_budget(0)
        };
        let (mut engine, a, cases) = engine_with_cases(
            &ctx,
            config,
            vec![unproducible_action(), unproducible_action()],
        );
        assert_eq!(
            engine.stats.work.canons, 1,
            "only the root is keyed when created"
        );
        let env = WorkerEnv::capture(None);
        let mut ts: Vec<Traversal> = (0..2)
            .map(|_| {
                let mut t = Traversal::from_root();
                engine.select_from(&env, &mut t).expect("select");
                t
            })
            .collect();
        assert_eq!(
            ts.iter().map(|t| t.end).collect::<Vec<_>>(),
            vec![TraversalEnd::Leaf(cases[0]), TraversalEnd::Leaf(cases[1])]
        );

        engine.key_leaves(&env, &mut ts).expect("key");
        assert_eq!(ts[0].end, TraversalEnd::Leaf(cases[0]));
        assert_eq!(ts[1].end, TraversalEnd::Done);
        assert!(engine.graph.class(cases[0]).keyed);
        assert!(engine.store[cases[0] as usize]
            .as_ref()
            .is_some_and(|stored| stored.labelling.is_some()));
        assert!(engine.store[cases[1] as usize].is_none());
        let targets: Vec<ClassId> = engine.graph.class(ROOT).ands[a as usize]
            .cases
            .iter()
            .map(|&(_, d)| d)
            .collect();
        assert_eq!(targets, vec![cases[0], cases[0]]);
        assert_eq!(engine.stats.merges, 1);
        assert_eq!(engine.stats.work.canons, 3);
    }

    /// A traversal whose case merges into an expanded class goes on from it
    /// and claims that class's case, which the next pass keys.
    #[test]
    fn a_merged_traversal_goes_on_and_the_next_pass_keys_its_new_claim() {
        let Some(ctx) = ctx() else {
            return;
        };
        if !crate::bliss_proc::bliss_available() {
            return;
        }
        let config = McgsConfig {
            merge: true,
            idf_weight: 0.0,
            ..McgsConfig::with_budget(0)
        };
        let (mut engine, a, cases) = engine_with_cases(&ctx, config, vec![unproducible_action()]);
        // An expanded class with the case's key, and one unkeyed case below.
        let existing = engine.intern(Child::Unkeyed(Stored::new(unproducible_action(), None)), 3);
        let (key, _) = canonical_key(&unproducible_action(), &mut WorkStats::default());
        assert_eq!(engine.graph.key(existing, key), None);
        let ext = &mut engine.graph.class_mut(existing).ext;
        ext.expanded = true;
        ext.actions = vec![Action {
            prior: 1.0,
            state: ActionState::Virtual,
        }];
        let below = engine.intern(Child::Unkeyed(Stored::new(two_true_formulas(), None)), 4);
        apply(&mut engine.graph, existing, 0, &[below]);

        let env = WorkerEnv::capture(None);
        let mut ts = vec![Traversal::from_root()];
        engine.select_from(&env, &mut ts[0]).expect("select");
        assert_eq!(ts[0].end, TraversalEnd::Leaf(cases[0]));
        engine.key_leaves(&env, &mut ts).expect("key");
        assert_eq!(ts[0].end, TraversalEnd::Leaf(below));
        assert_eq!(ts[0].path, vec![(ROOT, a), (existing, 0)]);
        assert_eq!(engine.graph.class(cases[0]).ext.merged_into, Some(existing));
        assert_eq!(
            engine.graph.class(ROOT).ands[a as usize].cases[0].1,
            existing
        );
        assert!(engine.graph.class(below).keyed);
        assert_eq!(engine.stats.merges, 1);
        assert_eq!(
            engine.stats.work.canons, 3,
            "the root, the case, the claim below"
        );
    }

    /// Eager keying canonicalizes a case when its method is applied: a
    /// second case with the first's key merges into its class at once, its
    /// system is never stored, and no class is left unkeyed. The merge is
    /// used once a traversal goes through it.
    #[test]
    fn eager_keying_merges_a_case_when_its_method_is_applied() {
        let Some(ctx) = ctx() else {
            return;
        };
        if !crate::bliss_proc::bliss_available() {
            return;
        }
        let config = McgsConfig {
            merge: true,
            keying: Keying::Eager,
            idf_weight: 0.0,
            ..McgsConfig::with_budget(0)
        };
        let deadline = Instant::now() + Duration::from_secs(60);
        let mut engine = Engine::new(&ctx, config, DEPTH_CAP, deadline);
        engine.add_root(two_branch_disjunction());
        let ext = &mut engine.graph.class_mut(ROOT).ext;
        ext.expanded = true;
        ext.actions = vec![Action {
            prior: 1.0,
            state: ActionState::Virtual,
        }];
        let mut work = WorkStats::default();
        let cases: Vec<(String, Child)> = (0..2)
            .map(|i| {
                let child = classify(&ctx, unproducible_action(), true, &mut work);
                (format!("case_{i}"), child)
            })
            .collect();
        let a = engine.integrate(
            ROOT,
            0,
            Applied {
                method: ProofMethod::Simplify,
                method_fp: None,
                cases,
            },
        );
        let targets: Vec<ClassId> = engine.graph.class(ROOT).ands[a as usize]
            .cases
            .iter()
            .map(|&(_, d)| d)
            .collect();
        assert_eq!(
            targets[0], targets[1],
            "the second case merged into the first"
        );
        assert_eq!(engine.stats.merges, 1);
        assert_eq!(
            engine.stats.merges_used, 0,
            "no traversal went through it yet"
        );
        assert_eq!(engine.graph.class(targets[0]).ext.merged_into, None);
        assert_eq!(
            engine.store.len(),
            engine.graph.classes.len(),
            "the merged case made no class"
        );
        let counts = engine.merge_counts();
        assert_eq!(counts.stored_unkeyed, 0);
        assert_eq!(counts.stored, 2, "the root and the one case class");

        let env = WorkerEnv::capture(None);
        for _ in 0..2 {
            let mut t = Traversal::from_root();
            engine.select_from(&env, &mut t).expect("select");
            if let TraversalEnd::Leaf(c) = t.end {
                engine.graph.class_mut(c).ext.in_flight = false;
            }
        }
        assert_eq!(
            engine.stats.merges_used, 1,
            "the second traversal goes through the merged case"
        );
    }

    /// Shadow keys find a duplicate among claimed systems without merging
    /// it: both classes keep their systems and stay apart.
    #[test]
    fn shadow_keys_mark_a_duplicate_without_merging() {
        let Some(ctx) = ctx() else {
            return;
        };
        if !crate::bliss_proc::bliss_available() {
            return;
        }
        let config = McgsConfig {
            shadow_keys: true,
            idf_weight: 0.0,
            batch: 2,
            ..McgsConfig::with_budget(0)
        };
        let (mut engine, a, cases) = engine_with_cases(
            &ctx,
            config,
            vec![unproducible_action(), unproducible_action()],
        );
        let env = WorkerEnv::capture(None);
        let mut ts: Vec<Traversal> = (0..2)
            .map(|_| {
                let mut t = Traversal::from_root();
                engine.select_from(&env, &mut t).expect("select");
                t
            })
            .collect();
        engine.key_leaves(&env, &mut ts).expect("key");
        assert_eq!(
            ts.iter().map(|t| t.end).collect::<Vec<_>>(),
            vec![TraversalEnd::Leaf(cases[0]), TraversalEnd::Leaf(cases[1])],
            "nothing merged"
        );
        assert_eq!(engine.graph.class(cases[1]).ext.dup_of, Some(cases[0]));
        assert_eq!(engine.graph.class(cases[0]).ext.dup_of, None);
        assert_eq!(engine.stats.shadow_dups, 1);
        assert_eq!(engine.stats.merges, 0);
        assert!(engine.store[cases[0] as usize].is_some());
        assert!(engine.store[cases[1] as usize].is_some());
        assert_eq!(
            engine.graph.class(ROOT).ands[a as usize]
                .cases
                .iter()
                .map(|&(_, d)| d)
                .collect::<Vec<_>>(),
            cases
        );
    }

    /// A case no traversal entered has no key; the proof shows it open.
    #[test]
    fn a_case_no_traversal_entered_shows_as_undecided() {
        let Some(mut ctx) = ctx() else {
            return;
        };
        if !crate::bliss_proc::bliss_available() {
            return;
        }
        ctx.mcgs = Some(McgsConfig {
            top_n: 1,
            merge: true,
            ..McgsConfig::with_budget(1)
        });
        let proof = run_proof_search(&ctx, unproducible_action(), 10).expect("proof search");
        assert!(
            matches!(proof.method, ProofMethod::Simplify),
            "got {:?}",
            proof.method
        );
        let case = proof.children.values().next().expect("simplify has a case");
        assert!(
            matches!(&case.method, ProofMethod::Sorry(Some(m)) if m == "MCGS: undecided"),
            "got {:?}",
            case.method
        );
    }
}
