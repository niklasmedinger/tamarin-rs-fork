//! (No HS analog.) The AND/OR search graph shared by the RS-only proof
//! searches ([`super::topn_search`], [`super::mcgs_search`]): the graph and
//! its status fixpoint, the worker-side preparation of a method's cases, and
//! the materializer that turns a settled graph back into a [`ProofNode`].
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
//! What a search keeps per class beyond that (its frontier entry, its visit
//! counts and priors) lives in the class's extension `X` ([`Class::ext`]).

use std::collections::BTreeMap;
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
// The AND/OR graph
// =============================================================================

pub type ClassId = u32;
pub(crate) const ROOT: ClassId = 0;

/// Depths stay far below this; it only keeps `depth + 1` from overflowing.
pub(crate) const DEPTH_CAP: u32 = u32::MAX / 4;

/// A final verdict about a class: a property of its constraint systems, so
/// it holds for every system of the class at any depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Settled {
    Contradictory,
    Solved,
    Unfinishable,
}

impl Settled {
    pub(crate) fn of(result: &MethodResult) -> Settled {
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
pub(crate) struct And {
    pub(crate) method: ProofMethod,
    /// The method canonicalized through the representative's labelling
    /// (merging on): identifies it among the candidates of any other system
    /// of the class when the proof is materialized.
    pub(crate) method_fp: Option<Fingerprint>,
    /// Sorted by case name, like the greedy driver's children.
    pub(crate) cases: Vec<(String, ClassId)>,
    pub(crate) status: Status,
}

/// One OR node: a constraint system, or a class of canonically equal ones.
pub(crate) struct Class<X> {
    pub(crate) status: Status,
    /// The shallowest depth the class was reached at.
    pub(crate) min_depth: u32,
    /// Applied methods, in the order they were applied.
    pub(crate) ands: Vec<And>,
    /// No further method will be applied: the search is done with it, the
    /// candidates ran out, or the class is a leaf.
    pub(crate) closed: bool,
    pub(crate) parents: Vec<(ClassId, u16)>,
    /// The edge that created the class; `None` for the root.
    pub(crate) first_parent: Option<(ClassId, u16)>,
    /// The method whose status settled the class, when it settled; `None`
    /// for a leaf and while open. Later methods can settle the same way
    /// through the class itself (a case leading back to it), so the proof
    /// follows this one: its cases all settled before the class did.
    pub(crate) settled_by: Option<u16>,
    /// When the class settled, on [`SearchGraph::clock`].
    pub(crate) settled_at: u64,
    /// What the search keeps per class.
    pub(crate) ext: X,
}

/// The classes, their methods and the transposition table. Holds no
/// `System`: those carry `Cell` caches, so they may move between threads but
/// not be shared, and the graph is read by the materializer's workers.
pub(crate) struct SearchGraph<X> {
    pub(crate) classes: Vec<Class<X>>,
    pub(crate) by_key: FastMap<CanonicalSystemFingerprint, ClassId>,
    /// Counts settlements, to order them ([`Class::settled_at`]).
    clock: u64,
}

/// How [`SearchGraph::intern`] placed a prepared case.
pub(crate) enum Interned {
    /// A new class at or below `--bound`, closed at once.
    Cut,
    /// A new, already settled class.
    Leaf,
    /// A new open class, with its system.
    New(Stored),
    /// An existing class reached again (merging on).
    Merged,
}

impl<X: Default> SearchGraph<X> {
    pub(crate) fn new() -> Self {
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

    pub(crate) fn class(&self, c: ClassId) -> &Class<X> {
        &self.classes[c as usize]
    }

    pub(crate) fn class_mut(&mut self, c: ClassId) -> &mut Class<X> {
        &mut self.classes[c as usize]
    }

    pub(crate) fn add_class(&mut self, status: Status, depth: u32) -> ClassId {
        let id = ClassId::try_from(self.classes.len()).expect("more than u32::MAX classes");
        let settled_at = if status == Status::Open {
            0
        } else {
            self.tick()
        };
        self.classes.push(Class {
            status,
            min_depth: depth,
            ands: Vec::new(),
            closed: status != Status::Open,
            parents: Vec::new(),
            first_parent: None,
            settled_by: None,
            settled_at,
            ext: X::default(),
        });
        id
    }

    pub(crate) fn add_and(
        &mut self,
        c: ClassId,
        method: ProofMethod,
        method_fp: Option<Fingerprint>,
    ) -> u16 {
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

    pub(crate) fn add_case(&mut self, c: ClassId, a: u16, name: String, child: ClassId) {
        self.class_mut(c).ands[a as usize].cases.push((name, child));
        let edge = (c, a);
        let class = self.class_mut(child);
        class.parents.push(edge);
        if class.first_parent.is_none() && child != ROOT {
            class.first_parent = Some(edge);
        }
    }

    /// The class of a prepared case reached at `depth`: a new one, or with
    /// merging an existing one reached again. The caller keeps the system
    /// of a new open class and does its own bookkeeping for the others.
    pub(crate) fn intern(&mut self, child: Child, depth: u32) -> (ClassId, Interned) {
        match child {
            Child::Cut => {
                let c = self.add_class(Status::Open, depth);
                self.class_mut(c).closed = true;
                (c, Interned::Cut)
            }
            Child::Finished(settled) => (
                self.add_class(Status::Settled(settled), depth),
                Interned::Leaf,
            ),
            Child::Unkeyed(stored) => (self.add_class(Status::Open, depth), Interned::New(stored)),
            Child::Keyed(key, stored) => {
                if let Some(&existing) = self.by_key.get(&key) {
                    return (existing, Interned::Merged);
                }
                let c = self.add_class(Status::Open, depth);
                self.by_key.insert(key, c);
                (c, Interned::New(stored))
            }
        }
    }

    /// Whether expanding `c` can still matter: it is the root, or some parent
    /// is open through an open method. Statuses only ever settle, so this
    /// only changes back when a new edge reaches `c`.
    pub(crate) fn is_relevant(&self, c: ClassId) -> bool {
        c == ROOT
            || self.class(c).parents.iter().any(|&(p, a)| {
                let parent = self.class(p);
                parent.status == Status::Open && parent.ands[a as usize].status == Status::Open
            })
    }

    pub(crate) fn and_status(&self, c: ClassId, a: u16) -> Status {
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
    pub(crate) fn class_status(&self, c: ClassId) -> Status {
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
                "AND/OR search: two methods of class #{c} disagree, one finds a trace and \
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
                        self.class(d).status == Status::Settled(Settled::Solved)
                            && !seen.contains(&d)
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
    pub(crate) fn recheck(&mut self, c: ClassId) -> Vec<(ClassId, u16)> {
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
    pub(crate) fn propagate(&mut self, mut work: Vec<(ClassId, u16)>) {
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
    pub(crate) fn close(&mut self, c: ClassId) {
        self.class_mut(c).closed = true;
        let parents = self.recheck(c);
        self.propagate(parents);
    }
}

// =============================================================================
// Workers
// =============================================================================

/// A class's system and how far its expansion got. It sits in a search's
/// store while the class may still get a method, and in a task while one of
/// its methods runs.
pub(crate) struct Stored {
    pub(crate) sys: Box<System>,
    /// `sys`'s canonical labelling (merging on), to canonicalize the methods
    /// applied to it.
    pub(crate) labelling: Option<CanonLabelling>,
    /// `candidate_methods`, computed at the first expansion (top-N search).
    pub(crate) ranked: Option<Vec<ProofMethod>>,
    /// The next candidate to try (top-N search).
    pub(crate) cursor: usize,
}

impl Stored {
    pub(crate) fn new(sys: System, labelling: Option<CanonLabelling>) -> Self {
        Stored {
            sys: Box::new(sys),
            labelling,
            ranked: None,
            cursor: 0,
        }
    }

    /// An independent copy of the system and its labelling, for a task
    /// that runs on another thread: a `System` may move between threads but
    /// not be shared.
    pub(crate) fn fork(&self) -> Self {
        Stored::new((*self.sys).clone(), self.labelling.clone())
    }
}

/// One case of an applied method, as a worker prepared it.
pub(crate) enum Child {
    /// At or below `--bound`.
    Cut,
    /// Already finished; never canonicalized, so never merged.
    Finished(Settled),
    /// Merging on: its canonical key.
    Keyed(CanonicalSystemFingerprint, Stored),
    /// Merging off.
    Unkeyed(Stored),
}

/// A method that applied, with its prepared cases.
pub(crate) struct Applied {
    pub(crate) method: ProofMethod,
    pub(crate) method_fp: Option<Fingerprint>,
    /// Sorted by case name.
    pub(crate) cases: Vec<(String, Child)>,
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct WorkStats {
    pub(crate) execs: u64,
    pub(crate) canons: u64,
    pub(crate) canon_time: Duration,
    pub(crate) busy: Duration,
}

impl WorkStats {
    pub(crate) fn add(&mut self, other: &WorkStats) {
        self.execs += other.execs;
        self.canons += other.canons;
        self.canon_time += other.canon_time;
        self.busy += other.busy;
    }
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
pub(crate) struct WorkerEnv {
    deadline: Option<Instant>,
    user_funs: CollectedUserFuns,
}

impl WorkerEnv {
    pub(crate) fn capture(deadline: Option<Instant>) -> Self {
        WorkerEnv {
            deadline,
            user_funs: crate::elaborate::snapshot_user_funs(),
        }
    }

    /// Runs `f` with this search's thread-locals installed and a maude handle
    /// seeded at `avoid_next`. Restores the calling thread's deadline after:
    /// rayon runs part of a parallel map on the calling thread when that is a
    /// pool thread itself.
    pub(crate) fn run<R>(
        &self,
        ctx: &ProofContext,
        avoid_next: u64,
        f: impl FnOnce(&ProofContext) -> R,
    ) -> R {
        let previous_deadline = search::replace_deadline(self.deadline);
        let _user_funs = crate::elaborate::set_user_funs_from_collected(&self.user_funs);
        let maude = ctx.maude.with_fresh_counter_next(avoid_next);
        let result = f(&ctx.with_swapped_maude(maude));
        search::replace_deadline(previous_deadline);
        result
    }
}

pub(crate) fn canonical_key(
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
pub(crate) fn classify(
    ctx: &ProofContext,
    sys: System,
    merge: bool,
    work: &mut WorkStats,
) -> Child {
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

/// Applies `method` to the stored system and prepares its cases; `None`
/// when the method does not apply, which `exec_proof_method` also answers,
/// without trying the method, once the deadline has passed.
pub(crate) fn apply_method(
    ctx: &ProofContext,
    stored: &Stored,
    method: &ProofMethod,
    cut_children: bool,
    merge: bool,
    work: &mut WorkStats,
) -> Option<Applied> {
    work.execs += 1;
    let mut cases = exec_proof_method(ctx, method, &stored.sys)?;
    cases.sort_by(|a, b| a.0.cmp(&b.0));
    let method_fp = stored.labelling.as_ref().map(|labelling| {
        fingerprint_proof_method(&canonicalize_proof_method(method, &stored.sys, labelling))
    });
    let cases = cases
        .into_iter()
        .map(|(name, sys)| {
            let child = if cut_children {
                Child::Cut
            } else {
                classify(ctx, sys, merge, work)
            };
            (name, child)
        })
        .collect();
    Some(Applied {
        method: method.clone(),
        method_fp,
        cases,
    })
}

// =============================================================================
// Materializing the proof
// =============================================================================

/// Why the search stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum End {
    /// The root settled.
    Settled,
    /// Nothing left to expand.
    Exhausted,
    /// The step budget is spent.
    Budget,
    Deadline,
}

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
pub(crate) struct Materializer<'g, X> {
    pub(crate) graph: &'g SearchGraph<X>,
    pub(crate) merge: bool,
    pub(crate) bound: u32,
    pub(crate) end: End,
    pub(crate) env: WorkerEnv,
    /// The search's name in the proof's `sorry` messages.
    pub(crate) label: &'static str,
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

impl<X: Default + Sync> Materializer<'_, X> {
    pub(crate) fn root(&self, ctx: &ProofContext, sys: System) -> ProofNode {
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
    pub(crate) fn reaches_trace(&self, d: ClassId, avoid: &[ClassId]) -> bool {
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
    /// `c` included). A trace takes the first applied method with a case
    /// that continues it without looping; other settled classes take the
    /// method that settled them, whose cases all settled earlier; an open
    /// class takes its first method, if any.
    pub(crate) fn choose(&self, c: ClassId, avoid: &[ClassId]) -> Option<usize> {
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
            format!("{}: undecided", self.label)
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
            return sorry(sys, format!("{}: cycle", self.label));
        }
        if class.status == Status::Open && edge.is_some() && edge != class.first_parent {
            // An undecided class is shown once, below the edge that created
            // it; unfolding it everywhere can take exponential space.
            return sorry(sys, format!("{}: undecided, shown elsewhere", self.label));
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

/// Contexts and small systems for the searches' tests against the greedy
/// driver.
#[cfg(test)]
pub(crate) mod test_support {
    use crate::constraint::constraints::{Disj, Goal, LessAtom, Reason};
    use crate::constraint::solver::context::ProofContext;
    use crate::constraint::system::System;
    use crate::test_maude::maude_path;
    use std::sync::Arc;
    use tamarin_term::lterm::{LNTerm, LSort, LVar};
    use tamarin_term::maude_sig::pair_maude_sig;

    /// `None` only when no maude resolves (`TAM_ALLOW_NO_MAUDE`); a maude
    /// that resolves but does not start panics, as in `search`'s tests.
    pub(crate) fn ctx() -> Option<ProofContext> {
        let path = maude_path()?;
        let h = tamarin_term::maude_proc::MaudeHandle::start(&path, pair_maude_sig())
            .unwrap_or_else(|e| panic!("maude at {path} failed to start: {e:?}"));
        Some(ProofContext::new(h, Vec::new()))
    }

    fn past_initial(sys: &mut System) {
        sys.add_less(LessAtom::new(
            LVar::new("a", LSort::Node, 0),
            LVar::new("b", LSort::Node, 0),
            Reason::Fresh,
        ));
    }

    /// A false formula and an empty disjunction goal.
    pub(crate) fn empty_disjunction() -> System {
        let mut sys = System::empty();
        past_initial(&mut sys);
        sys.formulas_mut().push(Arc::new(crate::guarded::gfalse()));
        sys.add_goal(Goal::Disj(Disj::new(Vec::new())));
        sys
    }

    /// A disjunction goal whose first branch is a trace.
    pub(crate) fn two_branch_disjunction() -> System {
        let mut sys = System::empty();
        past_initial(&mut sys);
        sys.add_goal(Goal::Disj(Disj::new(vec![
            crate::guarded::gtrue(),
            crate::guarded::gfalse(),
        ])));
        sys
    }

    /// Two true formulas: `simplify`, then solved.
    pub(crate) fn two_true_formulas() -> System {
        let mut sys = System::empty();
        past_initial(&mut sys);
        sys.formulas_mut().push(Arc::new(crate::guarded::gtrue()));
        sys.formulas_mut().push(Arc::new(crate::guarded::gtrue()));
        sys
    }

    /// An `Out` action goal in a context without rules: `simplify` ranks
    /// first but leaves it open, solving the goal closes the system.
    pub(crate) fn unproducible_action() -> System {
        use tamarin_term::vterm::Lit;
        let mut sys = System::empty();
        let tx: LNTerm = tamarin_term::term::Term::Lit(Lit::Var(LVar::new("x", LSort::Msg, 0)));
        let ty: LNTerm = tamarin_term::term::Term::Lit(Lit::Var(LVar::new("y", LSort::Msg, 0)));
        let i = LVar::new("i", LSort::Node, 0);
        sys.add_goal(Goal::Action(i, crate::fact::out_fact(tx)));
        sys.subterm_store_mut().add(ty.clone(), ty);
        sys
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Graph = SearchGraph<()>;

    fn open(g: &mut Graph) -> ClassId {
        g.add_class(Status::Open, 0)
    }

    fn leaf_class(g: &mut Graph, settled: Settled) -> ClassId {
        g.add_class(Status::Settled(settled), 0)
    }

    /// Adds a method to `c` with `children` as its cases and propagates.
    fn and(g: &mut Graph, c: ClassId, children: &[ClassId]) -> u16 {
        let a = g.add_and(c, ProofMethod::Simplify, None);
        for (i, &child) in children.iter().enumerate() {
            g.add_case(c, a, format!("case_{i}"), child);
        }
        g.propagate(vec![(c, a)]);
        a
    }

    #[test]
    fn a_method_closes_a_class_only_when_all_its_cases_are_contradictory() {
        let mut g = Graph::new();
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
        let mut g = Graph::new();
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
        let mut g = Graph::new();
        let root = open(&mut g);
        let contra = leaf_class(&mut g, Settled::Contradictory);
        let solved = leaf_class(&mut g, Settled::Solved);
        and(&mut g, root, &[contra]);
        and(&mut g, root, &[solved]);
    }

    #[test]
    fn a_cycle_does_not_justify_itself() {
        let mut g = Graph::new();
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
        let mut g = Graph::new();
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
        let mut g = Graph::new();
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
            label: "test",
        };
        assert!(m.reaches_trace(a, &[]));
        assert!(m.reaches_trace(b, &[a]));
        assert!(!m.reaches_trace(a, &[b]), "a's only trace runs through b");
        assert_eq!(m.choose(b, &[a, b]), Some(0));
    }

    #[test]
    fn unfinishable_needs_a_closed_class() {
        let mut g = Graph::new();
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
        let mut g = Graph::new();
        let root = open(&mut g);
        let a = g.add_and(root, ProofMethod::Sorry(Some("oracle".into())), None);
        g.propagate(vec![(root, a)]);
        g.close(root);
        assert_eq!(g.class(root).status, Status::Open);
    }
}
