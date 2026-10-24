// Currently GPL 3.0 until granted permission by the upstream authors
// of the tamarin-prover sources this file cites; list them with:
//   scripts/gen_license_headers.py --authors <this file>

//! (No HS analog.) Canonicalizing a whole constraint [`System`] through ONE
//! colored digraph: the system is written as one term ([`system_to_term`]),
//! the term becomes a colored digraph by the standard reduction from
//! AC-equality with commutative binders to graph isomorphism (Basin, *Equality
//! of terms containing associative-commutative functions and commutative
//! binding operators is isomorphism complete*, Cornell TR 89-1020, Lemma 2;
//! [`term_to_graph`]), and bliss's canonical labeling of that graph names the
//! system's variables.
//!
//! Every free variable is a vertex with an edge to each of its occurrences,
//! so bliss sees the whole system, and its canonical labeling is already
//! canonical: an automorphism of a faithful encoding is a symmetry of the
//! system, so any labeling bliss picks gives the same result, and nothing
//! has to be minimized over the automorphism group.
//!
//! **The canonical form is the term.** bliss only decides the variable
//! names (θ); the form is the system term renamed by them and
//! normalized modulo its AC and C operators (step 3,
//! [`canonicalize_constraint_system_via_term_graph_with_labelling`]). Two
//! claims carry the correctness argument:
//!
//! - (A) [`system_to_term`] is faithful: two systems are alpha-AC equal iff
//!   their terms are. Soundness rests on (A) alone: equal forms are
//!   alpha-AC equal terms, so the systems are equal, whatever the graph or
//!   bliss did. A field `system_to_term` leaves out therefore merges systems
//!   that differ in it, a false positive; the exhaustive destructurings in
//!   `system_to_term` (and [`System::canon_fields`]) make a new field a
//!   compile error until it is decided.
//! - (B) [`term_to_graph`] is faithful (the reduction). Completeness rests on
//!   (B): a lossy graph has automorphisms that are no symmetries of the term,
//!   and different labelings then give different forms, a missed merge. The
//!   generator check ([`TermGraphOptions::check_generators`]) detects it.
//!
//! # Step 1: the system as one term
//!
//! [`STerm`] is a term over a signature of [`Op`]s, each free (ordered
//! arguments), AC (a multiset of arguments, nested applications flattened)
//! or C (two unordered arguments), plus commutative binders
//! ([`STerm::Exists`]). `LNTerm` cannot hold it: it has no binders, and its
//! AC symbols belong to the theory.
//!
//! ```text
//! ∃_Root x̄. ∃_Witnesses w̄. System[source_kind, side, contradictory](
//!     Nodes{…}, Edges{…}, LessAtoms{…}, Last(#i) | ⟨NoLast⟩,
//!     Formulas{…}, SolvedFormulas{…}, Lemmas{…},
//!     EqSubst{…}, EqConj{…}, Subterms{…}, SolvedSubterms{…}, NegSubterms{…}, Goals{…} )
//! ```
//!
//! Each `Name{…}` is an AC section operator with one argument per entry of
//! its field. The root binder binds every free variable of the system, the
//! witness binder every range variable of the equation store's
//! alternatives, each renamed apart as a witness of its own alternative
//! ([`Scope`] says why binding them at the top is the same as binding them
//! in their alternative).
//!
//! **Collections.** A field that is a `Vec` in `System` is a multiset: an AC
//! operator keeps duplicates as separate arguments, because the solver
//! keeps alpha-equivalent entries apart and their count is proof-search
//! state (CONVENTIONS rule 2). That covers `nodes`, `edges`, `less_atoms`,
//! `formulas`, `solved_formulas`, `lemmas`, `goals`, a disjunction goal's
//! alternatives, `eq_store.conj`, each `EqDisj`'s alternatives,
//! `subterm_store.subterms`/`solved_subterms`, and a quantifier's guards.
//! Sets are deduplicated modulo AC first: `subterm_store.neg_subterms` (a
//! set in `System`), and `Conj`/`Disj` inside a formula (which
//! `ac_normalize_guarded` deduplicates). Deduplicating modulo AC before
//! renaming equals deduplicating after it, since the renaming is a bijection.
//! Sequences stay ordered: a rule's premises/actions/conclusions (edges and
//! goals refer to their indices), arguments of free symbols, and a
//! quantifier's binders (de Bruijn indices refer to their positions).
//!
//! **Field audit** (CONVENTIONS rule 3; PLAN.md's field audit):
//!
//! | Field | Term |
//! |---|---|
//! | `nodes` | `RuleInst[rule](#i, Prems(…), Acts(…), Concs(…))`; `rule` is the rule's identity ([`RuleId`]); `new_vars` excluded (derived from the facts) |
//! | facts | `Fact[tag](t1 … tn)`; annotations excluded: solver hints without semantics |
//! | `LNTerm` | `Fun(f)` free / AC (flattened) / C by the symbol; a name is a constant; a variable an occurrence |
//! | `edges` | `Edge[c, p](#src, #tgt)` |
//! | `less_atoms` | `Less(#a, #b)`; `reason` excluded |
//! | `last_atom` | `Last(#i)` or the constant `NoLast` |
//! | `formulas`, `solved_formulas`, `lemmas` | the `ac_normalize_guarded` form, translated as below; three separate sections |
//! | `eq_store.subst` | `EqBind(#x, t)` |
//! | `eq_store.conj` | `EqDisj{ EqAlt{ EqBind(#x, t) … } … }`: every variable in an alternative's range is that alternative's own, a witness ([`VarRef::Witness`], [`witness`]); `split_id` and `next_split` excluded |
//! | `subterm_store` | `Sub(s, t)` in three sections; `contradictory` in `System`'s label; `old_neg_subterms`, `propagated` excluded |
//! | `goals` | `Goal[solved](payload)`; `Split` goals (they only name an `EqDisj`), `looping` and `nr` excluded |
//! | `source_kind`, `side` | `System`'s label |
//! | `next_goal_nr`, caches | excluded (PLAN.md's field audit) |
//!
//! Formulas keep their de Bruijn indices as constants, so a quantifier
//! block's binder order stays significant: two formulas differing only in it
//! get different forms, a missed merge.
//! `Conj`/`Disj` and a quantifier's guards are AC and `=` is C. A formula's
//! terms and facts are written like the system's (`Fun(f)`, `Fact[tag]`), with
//! a free variable as the same variable and a bound one as its de Bruijn
//! index.
//!
//! **Closed subterms** (an `LNTerm`, fact, formula, atom or formula term
//! without a free variable) become one constant keyed by the fingerprint of
//! their AC normal form (for the formula parts: of their translation,
//! normalized): two of them are equal modulo AC iff their normal forms are,
//! and no renaming touches them. This keeps lemmas and ground terms down to
//! one vertex each.
//!
//! # Step 2: the term as a colored digraph
//!
//! [`term_to_graph`] follows the reduction case by case. Every vertex gets a
//! key `(label, pos)`, where `pos` is its argument position under a free
//! parent and `None` under an AC or C parent or a binder:
//!
//! - a constant is one vertex;
//! - an application is one vertex with an edge to each argument (an AC
//!   operator's nested applications of itself are flattened into it);
//! - a variable occurrence is a NEW vertex `Occ(sort)`, with an edge from the
//!   vertex of the variable, `BoundVar(scope, sort)`;
//! - a binder is a vertex with an edge to each of its variables' vertices and
//!   one to its body.
//!
//! Departures from the paper, each keeping the encoding faithful:
//! - Edge labels (argument positions) are folded into the child's key: bliss
//!   colors vertices only, and every vertex but a variable vertex has exactly
//!   one parent, so "my i-th child" and "my child at position i" coincide.
//! - Every occurrence gets its own vertex. Pointing the parents at one vertex
//!   per variable would make the AC terms `f(x, x, y)` and `f(x, y, y)` the
//!   same edge set, since bliss drops duplicate edges.
//! - Sorts are part of the variable and occurrence labels, so a renaming that
//!   an isomorphism induces is sort-respecting.
//!
//! A vertex's color is the RANK of its key among the keys of the graph: an
//! isomorphism between two systems' graphs maps keys to equal keys, so both
//! have the same key set and the same ranks.
//!
//! # Step 3: the names and the form
//!
//! bliss's canonical labeling orders the graph's vertices. The free variables
//! and the witnesses (the two binders' variable vertices) are named per sort
//! in that order (`mv0`, `fv0`, `tv0`, …): that is θ. The form is the system
//! term without its two binders, renamed by θ ([`rename`]) and normalized
//! ([`normalize`]): AC operators flattened and their arguments sorted, C
//! arguments sorted, duplicates kept. A witness stays a `VarRef::Witness` in
//! the form.
//!
//! Any other canonical labeling is this one composed with an automorphism of
//! the graph. Under a faithful encoding every automorphism is a symmetry of
//! the system, so the names may differ but the form does not; the generator
//! check confirms it for each generator that moves a variable vertex.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::time::{Duration, Instant};

use tamarin_term::alpha_eq_ac::{rename_var, CanonLabelling};
use tamarin_term::fingerprint::fingerprint_term;
use tamarin_term::function_symbols::FunSym;
use tamarin_term::lterm::{BVar, HasFrees, LNTerm, LSort, LVar, Name};
use tamarin_term::term::Term;
use tamarin_term::vterm::Lit;
use tamarin_utils::fingerprint::{Fingerprint, FingerprintHasher};
use tamarin_utils::FastMap;

use crate::atom::{Atom, ProtoAtom};
use crate::bliss_proc::{run_bliss_with_stats, to_dimacs, BlissError, Color, Permutation};
use crate::canon::{fact_to_term, rule_id, CanonicalSystem, RuleId};
use crate::constraint::constraints::{Edge, Goal, LessAtom};
use crate::constraint::system::{CanonFields, GoalStatus, Side, SourceKind, System};
use crate::fact::{Fact, FactTag, LNFact};
use crate::formula::{BLNTerm, Quantifier};
use crate::guarded::Guarded;
use crate::rule::{Rule, RuleACInst};
use crate::tools::equation_store::EqDisj;
use crate::tools::subterm_store::SubtermConstraint;
use crate::tools::{EquationStore, SubtermStore};

// =============================================================================
// The term language
// =============================================================================

/// A term of the system's signature (see the module docs). `Ord` is only a
/// total order to sort the arguments of AC and C operators by
/// ([`normalize`]), and `Hash` feeds [`fingerprint_sterm`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum STerm {
    App(Op, Vec<STerm>),
    Var(VarRef),
    Const(ConstKey),
    /// A commutative binder `∃ x1 … xn. body`.
    Exists(Scope, Vec<LVar>, Box<STerm>),
}

/// A variable occurrence: a free variable of the system, or a witness, a
/// range variable of an equation-store alternative renamed apart from every
/// other variable ([`witness`]). Both are bound at the top of the system
/// term, by different binders ([`Scope`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum VarRef {
    Free(LVar),
    Witness(LVar),
}

/// Which binder an [`STerm::Exists`] is. Both sit at the top of the system
/// term, the witness binder directly under the root binder.
///
/// A range variable of an equation-store alternative exists only in that
/// alternative. Renamed apart ([`witness`]), it occurs only in its
/// alternative's range terms, so binding it at the top is the same as
/// binding it in the alternative: a renaming of the whole term maps the
/// witnesses of one alternative onto those of another, and free variables
/// onto free variables. A binder of their own gives the witnesses' variable
/// vertices a label of their own, so bliss's first partition already
/// separates them from the free variables; being a witness is part of the
/// system, so the finer partition costs no merge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Scope {
    /// Binds every free variable of the system.
    Root,
    /// Binds every witness of the equation store.
    Witnesses,
}

/// The first character of a witness's name ([`witness`]). No variable of a
/// theory or of the solver contains a NUL, so no witness can coincide with
/// one ([`Translator::free_var`] checks it).
const WITNESS_MARK: char = '\0';

/// The witness for the `k`-th range variable `v` (in raw order) of
/// alternative `a` of the equation store's disjunction `d`: `v`'s sort and a
/// name determined by `(d, a, k)` alone. Distinct for every alternative and
/// every variable in it, and the same whenever the disjunction is translated,
/// so a disjunction translated on its own ([`canonical_eq_disj`]) gets the
/// witnesses it has in its system's term.
fn witness(d: usize, a: usize, k: usize, v: LVar) -> LVar {
    LVar::new(format!("{WITNESS_MARK}w{d}.{a}"), v.sort, k as u64)
}

/// How an operator treats its arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpKind {
    Free,
    Ac,
    C,
}

/// The fields of the system that are collections, in `System`'s argument
/// order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Section {
    Nodes,
    Edges,
    LessAtoms,
    Formulas,
    SolvedFormulas,
    Lemmas,
    EqSubst,
    EqConj,
    Subterms,
    SolvedSubterms,
    NegSubterms,
    Goals,
}

/// Which group of a rule's facts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FactGroup {
    Prems,
    Acts,
    Concs,
}

/// The operators. Structural ones are their own variants, so they can never
/// collide with a theory symbol (which only ever appears wrapped in `Fun` or
/// `Fact`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Op {
    System {
        source_kind: Option<SourceKind>,
        side: Option<Side>,
        contradictory: bool,
    },
    Section(Section),
    RuleInst(RuleId),
    FactGroup(FactGroup),
    Fact(FactTag),
    Fun(FunSym),
    Edge(usize, usize),
    Less,
    Last,
    EqBind,
    EqDisj,
    EqAlt,
    Sub,
    Goal {
        solved: bool,
    },
    GoalAction,
    GoalChain(usize, usize),
    GoalPremise(usize),
    GoalDisj,
    GoalSubterm,
    // Formulas (`Guarded`) ...
    GConj,
    GDisj,
    /// A quantifier block: existential or not, and the sorts of its binders
    /// in order.
    GQuant {
        ex: bool,
        sorts: Vec<LSort>,
    },
    GGuards,
    GEq,
    GLess,
    GSubterm,
    GAction,
    GLast,
    /// The atom-level sugar placeholder (`ProtoAtom::Syntactic`), which has
    /// no terms and which the solver skips.
    GSyntactic,
}

impl Op {
    pub fn kind(&self) -> OpKind {
        match self {
            Op::Section(_) | Op::EqDisj | Op::EqAlt | Op::GoalDisj => OpKind::Ac,
            Op::GConj | Op::GDisj | Op::GGuards => OpKind::Ac,
            Op::GEq => OpKind::C,
            Op::Fun(sym) if sym.is_ac() => OpKind::Ac,
            Op::Fun(sym) if sym.is_c() => OpKind::C,
            _ => OpKind::Free,
        }
    }
}

/// What a closed subterm was, so fingerprints of different kinds of object
/// never compare equal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ClosedKind {
    Term,
    Fact,
    Formula,
    Atom,
    FormulaTerm,
}

/// A leaf that no renaming touches.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConstKey {
    /// A name (`'a'`, `~'n'`, `%'n'`), in a system term or a formula: names
    /// are constants.
    Name(Name),
    /// A de Bruijn index: a bound occurrence in a formula.
    DeBruijn(u64),
    NoLast,
    /// A subterm without free variables, by the fingerprint of its AC normal
    /// form (see the module docs).
    Closed(ClosedKind, Fingerprint),
}

// =============================================================================
// Step 1: System -> STerm
// =============================================================================

/// The names of the arguments of the system term's `System[…]` body, in
/// order: the sections a fingerprint and a comparison of two forms report.
pub const SYSTEM_SECTIONS: [&str; 13] = [
    "nodes",
    "edges",
    "less_atoms",
    "last",
    "formulas",
    "solved_formulas",
    "lemmas",
    "eq_subst",
    "eq_conj",
    "subterms",
    "solved_subterms",
    "neg_subterms",
    "goals",
];

/// The whole of `sys` as one closed term (see the module docs).
///
/// Every field of the system is either translated or excluded on purpose,
/// and the destructurings below are exhaustive (no `..`), so a new field is
/// a compile error until it is decided: an omitted field would merge systems
/// that differ in it.
pub fn system_to_term(sys: &System) -> STerm {
    let CanonFields {
        nodes,
        edges,
        less_atoms,
        last_atom,
        formulas,
        solved_formulas,
        lemmas,
        eq_store,
        subterm_store,
        goals,
        source_kind,
        side,
    } = sys.canon_fields();
    let EquationStore {
        subst,
        conj,
        // An allocation counter for split ids.
        next_split: _,
    } = eq_store;
    let SubtermStore {
        subterms,
        solved_subterms,
        contradictory,
        neg_subterms,
        // A change-detection snapshot for one simplification pass.
        old_neg_subterms: _,
    } = subterm_store;

    let mut free: BTreeSet<LVar> = BTreeSet::new();
    let mut witnesses: BTreeSet<LVar> = BTreeSet::new();
    let mut tr = Translator {
        free: &mut free,
        witnesses: &mut witnesses,
    };

    let nodes = nodes
        .iter()
        .map(|(nid, ru)| tr.rule_instance(*nid, ru))
        .collect();
    let edges = edges
        .iter()
        .map(|Edge { src, tgt }| {
            STerm::App(
                Op::Edge(src.1 .0, tgt.1 .0),
                vec![tr.free_var(src.0), tr.free_var(tgt.0)],
            )
        })
        .collect();
    let less_atoms = less_atoms
        .iter()
        .map(
            |LessAtom {
                 smaller,
                 larger,
                 reason: _,
             }| {
                // `reason` is provenance only.
                STerm::App(Op::Less, vec![tr.free_var(*smaller), tr.free_var(*larger)])
            },
        )
        .collect();
    let last = match last_atom {
        Some(nid) => STerm::App(Op::Last, vec![tr.free_var(nid)]),
        None => STerm::Const(ConstKey::NoLast),
    };
    let formulas = formulas.iter().map(|f| tr.formula(f)).collect();
    let solved_formulas = solved_formulas.iter().map(|f| tr.formula(f)).collect();
    let lemmas = lemmas.iter().map(|f| tr.formula(f)).collect();

    let eq_subst = subst
        .iter()
        .map(|(x, t)| STerm::App(Op::EqBind, vec![tr.free_var(*x), tr.term(t, None)]))
        .collect();
    let eq_conj = conj
        .iter()
        .enumerate()
        .map(|(d, disj)| tr.eq_disj(d, disj))
        .collect();

    let pair = |tr: &mut Translator, s: &LNTerm, t: &LNTerm| {
        STerm::App(Op::Sub, vec![tr.term(s, None), tr.term(t, None)])
    };
    // `propagated` is bookkeeping of the subterm simplification.
    let constraint = |tr: &mut Translator,
                      SubtermConstraint {
                          small,
                          big,
                          propagated: _,
                      }: &SubtermConstraint| pair(tr, small, big);
    let subterms = subterms.iter().map(|c| constraint(&mut tr, c)).collect();
    let solved_subterms = solved_subterms
        .iter()
        .map(|c| constraint(&mut tr, c))
        .collect();
    let neg_subterms: BTreeSet<(LNTerm, LNTerm)> = neg_subterms
        .iter()
        .map(|(s, t)| (ac_normal(s), ac_normal(t)))
        .collect();
    let neg_subterms = neg_subterms
        .iter()
        .map(|(s, t)| pair(&mut tr, s, t))
        .collect();

    let goals = goals
        .iter()
        .filter_map(
            |(
                goal,
                GoalStatus {
                    solved,
                    // Heuristic and path-dependent bookkeeping.
                    looping: _,
                    nr: _,
                },
            )| {
                tr.goal(goal)
                    .map(|payload| STerm::App(Op::Goal { solved: *solved }, vec![payload]))
            },
        )
        .collect();

    let section = |s: Section, items: Vec<STerm>| STerm::App(Op::Section(s), items);
    // In the order of `SYSTEM_SECTIONS`.
    let body = STerm::App(
        Op::System {
            source_kind,
            side,
            contradictory: *contradictory,
        },
        vec![
            section(Section::Nodes, nodes),
            section(Section::Edges, edges),
            section(Section::LessAtoms, less_atoms),
            last,
            section(Section::Formulas, formulas),
            section(Section::SolvedFormulas, solved_formulas),
            section(Section::Lemmas, lemmas),
            section(Section::EqSubst, eq_subst),
            section(Section::EqConj, eq_conj),
            section(Section::Subterms, subterms),
            section(Section::SolvedSubterms, solved_subterms),
            section(Section::NegSubterms, neg_subterms),
            section(Section::Goals, goals),
        ],
    );
    let body = STerm::Exists(
        Scope::Witnesses,
        witnesses.into_iter().collect(),
        Box::new(body),
    );
    STerm::Exists(Scope::Root, free.into_iter().collect(), Box::new(body))
}

/// The AC normal form of `t` (the smart constructors re-sort AC arguments
/// on rebuild).
fn ac_normal(t: &LNTerm) -> LNTerm {
    t.clone().map_free(&mut |v| v)
}

/// Translates the parts of one system, collecting its free variables.
struct Translator<'a> {
    free: &'a mut BTreeSet<LVar>,
    witnesses: &'a mut BTreeSet<LVar>,
}

impl Translator<'_> {
    fn free_var(&mut self, v: LVar) -> STerm {
        assert!(
            !v.name.starts_with(WITNESS_MARK),
            "a free variable of the system is named like a witness: {v:?}"
        );
        self.free.insert(v);
        STerm::Var(VarRef::Free(v))
    }

    fn rule_instance(&mut self, nid: LVar, ru: &RuleACInst) -> STerm {
        let Rule {
            // In the label, as `rule_id(ru)`.
            info: _,
            premises,
            actions,
            conclusions,
            // The fresh variables' instantiations, which occur in the facts.
            new_vars: _,
        } = ru;
        let group = |tr: &mut Self, g: FactGroup, facts: &[LNFact]| {
            STerm::App(Op::FactGroup(g), facts.iter().map(|f| tr.fact(f)).collect())
        };
        let node = self.free_var(nid);
        let prems = group(self, FactGroup::Prems, premises);
        let acts = group(self, FactGroup::Acts, actions);
        let concs = group(self, FactGroup::Concs, conclusions);
        STerm::App(Op::RuleInst(rule_id(ru)), vec![node, prems, acts, concs])
    }

    fn fact(&mut self, fact: &LNFact) -> STerm {
        let (tag, terms) = fact.canon_parts();
        if terms.iter().all(is_ground) {
            let fp = fingerprint_term(&ac_normal(&fact_to_term(fact)));
            return STerm::Const(ConstKey::Closed(ClosedKind::Fact, fp));
        }
        let args = terms.iter().map(|t| self.term(t, None)).collect();
        STerm::App(Op::Fact(*tag), args)
    }

    /// `t`, with every variable free, or, in a range term of an
    /// equation-store alternative, its witness from `witnesses`.
    fn term(&mut self, t: &LNTerm, witnesses: Option<&BTreeMap<LVar, LVar>>) -> STerm {
        if is_ground(t) {
            return STerm::Const(ConstKey::Closed(
                ClosedKind::Term,
                fingerprint_term(&ac_normal(t)),
            ));
        }
        self.term_rec(t, witnesses)
    }

    fn term_rec(&mut self, t: &LNTerm, witnesses: Option<&BTreeMap<LVar, LVar>>) -> STerm {
        match t {
            Term::Lit(Lit::Var(v)) => match witnesses {
                None => self.free_var(*v),
                Some(names) => {
                    let w = *names
                        .get(v)
                        .unwrap_or_else(|| panic!("range variable {v:?} has no witness"));
                    self.witnesses.insert(w);
                    STerm::Var(VarRef::Witness(w))
                }
            },
            Term::Lit(Lit::Con(n)) => STerm::Const(ConstKey::Name(*n)),
            Term::App(sym, args) => STerm::App(
                Op::Fun(*sym),
                args.iter().map(|a| self.term(a, witnesses)).collect(),
            ),
        }
    }

    /// One formula, translated in the form `ac_normalize_guarded` gives it
    /// (deduplicated conjunctions and disjunctions, binder names erased).
    fn formula(&mut self, g: &Guarded) -> STerm {
        self.guarded(&crate::canon::ac_normalize_guarded(g))
    }

    fn guarded(&mut self, g: &Guarded) -> STerm {
        if !has_free(g) {
            return closed(ClosedKind::Formula, self.guarded_parts(g));
        }
        self.guarded_parts(g)
    }

    fn guarded_parts(&mut self, g: &Guarded) -> STerm {
        match g {
            Guarded::Atom(a) => self.atom(a),
            Guarded::Conj(items) => {
                STerm::App(Op::GConj, items.iter().map(|g| self.guarded(g)).collect())
            }
            Guarded::Disj(items) => {
                STerm::App(Op::GDisj, items.iter().map(|g| self.guarded(g)).collect())
            }
            Guarded::GGuarded {
                qua,
                vars,
                guards,
                body,
            } => {
                let op = Op::GQuant {
                    ex: matches!(qua, Quantifier::Ex),
                    // Binder names are display hints; bound occurrences
                    // are de Bruijn indices.
                    sorts: vars.iter().map(|(_, sort)| *sort).collect(),
                };
                let guards = STerm::App(Op::GGuards, guards.iter().map(|a| self.atom(a)).collect());
                STerm::App(op, vec![guards, self.guarded(body)])
            }
        }
    }

    fn atom(&mut self, a: &Atom<BLNTerm>) -> STerm {
        if !has_free(a) {
            return closed(ClosedKind::Atom, self.atom_parts(a));
        }
        self.atom_parts(a)
    }

    fn atom_parts(&mut self, a: &Atom<BLNTerm>) -> STerm {
        let two = |tr: &mut Self, op: Op, x: &BLNTerm, y: &BLNTerm| {
            let args = vec![tr.formula_term(x), tr.formula_term(y)];
            STerm::App(op, args)
        };
        match a {
            ProtoAtom::EqE(x, y) => two(self, Op::GEq, x, y),
            ProtoAtom::Less(x, y) => two(self, Op::GLess, x, y),
            ProtoAtom::Subterm(x, y) => two(self, Op::GSubterm, x, y),
            ProtoAtom::Action(t, fact) => {
                let args = vec![self.formula_fact(fact), self.formula_term(t)];
                STerm::App(Op::GAction, args)
            }
            ProtoAtom::Last(t) => STerm::App(Op::GLast, vec![self.formula_term(t)]),
            ProtoAtom::Syntactic(_) => STerm::App(Op::GSyntactic, Vec::new()),
        }
    }

    /// A fact of a formula, written like a fact of the system; annotations
    /// are left out (see the module docs).
    fn formula_fact(&mut self, fact: &Fact<BLNTerm>) -> STerm {
        let (tag, terms) = fact.canon_parts();
        STerm::App(
            Op::Fact(*tag),
            terms.iter().map(|t| self.formula_term(t)).collect(),
        )
    }

    /// A term of a formula, written like a term of the system, with a bound
    /// variable as its de Bruijn index.
    fn formula_term(&mut self, t: &BLNTerm) -> STerm {
        if matches!(t, Term::App(..)) && !has_free(t) {
            return closed(ClosedKind::FormulaTerm, self.formula_term_parts(t));
        }
        self.formula_term_parts(t)
    }

    fn formula_term_parts(&mut self, t: &BLNTerm) -> STerm {
        match t {
            Term::Lit(Lit::Var(BVar::Free(v))) => self.free_var(*v),
            Term::Lit(Lit::Var(BVar::Bound(i))) => STerm::Const(ConstKey::DeBruijn(*i)),
            Term::Lit(Lit::Con(n)) => STerm::Const(ConstKey::Name(*n)),
            Term::App(sym, args) => STerm::App(
                Op::Fun(*sym),
                args.iter().map(|a| self.formula_term(a)).collect(),
            ),
        }
    }

    /// The equation store's disjunction number `d`. The range variables of
    /// each alternative are its own: they become the alternative's witnesses
    /// ([`witness`]).
    fn eq_disj(&mut self, d: usize, disj: &EqDisj) -> STerm {
        let EqDisj {
            // Only names the disjunction for its `Split` goal, which is left
            // out too.
            split_id: _,
            substs,
        } = disj;
        let alternatives = substs
            .iter()
            .enumerate()
            .map(|(a, alt)| {
                let mut range_vars: BTreeSet<LVar> = BTreeSet::new();
                for (_, t) in alt.iter() {
                    t.for_each_free(&mut |v| {
                        range_vars.insert(*v);
                    });
                }
                let names: BTreeMap<LVar, LVar> = range_vars
                    .into_iter()
                    .enumerate()
                    .map(|(k, v)| (v, witness(d, a, k, v)))
                    .collect();
                let binds = alt
                    .iter()
                    .map(|(x, t)| {
                        STerm::App(
                            Op::EqBind,
                            vec![self.free_var(*x), self.term(t, Some(&names))],
                        )
                    })
                    .collect();
                STerm::App(Op::EqAlt, binds)
            })
            .collect();
        STerm::App(Op::EqDisj, alternatives)
    }

    /// One goal's payload; `None` for a `Split` goal (see the module docs).
    fn goal(&mut self, goal: &Goal) -> Option<STerm> {
        Some(match goal {
            Goal::Split(_) => return None,
            Goal::Action(nid, fact) => {
                let args = vec![self.free_var(*nid), self.fact(fact)];
                STerm::App(Op::GoalAction, args)
            }
            Goal::Chain((src, conc), (tgt, prem)) => {
                let args = vec![self.free_var(*src), self.free_var(*tgt)];
                STerm::App(Op::GoalChain(conc.0, prem.0), args)
            }
            Goal::Premise((nid, prem), fact) => {
                let args = vec![self.free_var(*nid), self.fact(fact)];
                STerm::App(Op::GoalPremise(prem.0), args)
            }
            Goal::Disj(disj) => STerm::App(
                Op::GoalDisj,
                disj.0.iter().map(|g| self.formula(g)).collect(),
            ),
            Goal::Subterm((small, big)) => {
                let args = vec![self.term(small, None), self.term(big, None)];
                STerm::App(Op::GoalSubterm, args)
            }
        })
    }
}

fn is_ground(t: &LNTerm) -> bool {
    match t {
        Term::Lit(Lit::Var(_)) => false,
        Term::Lit(Lit::Con(_)) => true,
        Term::App(_, args) => args.iter().all(is_ground),
    }
}

/// Whether `x` (a formula, atom or formula term) has a free variable.
fn has_free<T: HasFrees>(x: &T) -> bool {
    let mut any = false;
    x.for_each_free(&mut |_| any = true);
    any
}

/// The constant for a subterm without free variables: the fingerprint of its
/// translation, normalized (see the module docs).
fn closed(kind: ClosedKind, t: STerm) -> STerm {
    STerm::Const(ConstKey::Closed(kind, fingerprint_sterm(&normalize(t))))
}

// =============================================================================
// Step 2: STerm -> colored digraph
// =============================================================================

/// What a vertex stands for (see the module docs).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Label {
    Op(Op),
    Const(ConstKey),
    Binder(Scope),
    BoundVar(Scope, LSort),
    Occ(LSort),
}

/// A vertex's key: its label and its argument position under a free parent.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VKey {
    pub label: Label,
    pub pos: Option<usize>,
}

/// The colored digraph of a system term.
#[derive(Debug, Clone, Default)]
pub struct TermGraph {
    pub keys: Vec<VKey>,
    /// `colors[v]`: the rank of `keys[v]` among the distinct keys.
    pub colors: Vec<Color>,
    pub edges: Vec<(usize, usize)>,
    /// The root binder's variable vertices: the system's free variables.
    pub free_vars: Vec<(usize, LVar)>,
    /// The witness binder's variable vertices: the equation store's
    /// witnesses.
    pub witness_vars: Vec<(usize, LVar)>,
    /// Each vertex's parent in the term (a variable vertex's is its binder,
    /// an occurrence's the application it is an argument of); `None` for the
    /// root.
    pub parents: Vec<Option<usize>>,
}

/// The colored digraph of `t` (see the module docs). Panics on a variable
/// occurrence no enclosing binder binds: a translation bug, never a case to
/// pass over.
pub fn term_to_graph(t: &STerm) -> TermGraph {
    let mut b = GraphBuilder {
        graph: TermGraph::default(),
        scopes: Vec::new(),
    };
    b.build(t, None);
    let mut graph = b.graph;
    let ranks: BTreeMap<&VKey, Color> = graph
        .keys
        .iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .zip(0..)
        .collect();
    graph.colors = graph.keys.iter().map(|k| ranks[k]).collect();
    graph
}

struct GraphBuilder {
    graph: TermGraph,
    /// The enclosing binders, innermost last: their scope and variable
    /// vertices.
    scopes: Vec<(Scope, BTreeMap<LVar, usize>)>,
}

impl GraphBuilder {
    fn vertex(&mut self, label: Label, pos: Option<usize>) -> usize {
        self.graph.keys.push(VKey { label, pos });
        self.graph.parents.push(None);
        self.graph.keys.len() - 1
    }

    fn edge(&mut self, src: usize, tgt: usize) {
        self.graph.edges.push((src, tgt));
    }

    /// The edge from `parent` to its term child `child`.
    fn child_edge(&mut self, parent: usize, child: usize) {
        self.graph.parents[child] = Some(parent);
        self.edge(parent, child);
    }

    fn build(&mut self, t: &STerm, pos: Option<usize>) -> usize {
        match t {
            STerm::Const(c) => self.vertex(Label::Const(c.clone()), pos),
            STerm::Var(r) => {
                let (v, scope) = match r {
                    VarRef::Free(v) => (v, Scope::Root),
                    VarRef::Witness(v) => (v, Scope::Witnesses),
                };
                let w = self
                    .scopes
                    .iter()
                    .rev()
                    .find(|(s, _)| *s == scope)
                    .and_then(|(_, vars)| vars.get(v))
                    .copied();
                let w = w.unwrap_or_else(|| {
                    panic!("term_to_graph: no binder binds {r:?} -- system_to_term missed {v}")
                });
                let occ = self.vertex(Label::Occ(v.sort), pos);
                self.edge(w, occ);
                occ
            }
            STerm::App(op, args) => {
                let me = self.vertex(Label::Op(op.clone()), pos);
                match op.kind() {
                    OpKind::Free => {
                        for (i, a) in args.iter().enumerate() {
                            let child = self.build(a, Some(i));
                            self.child_edge(me, child);
                        }
                    }
                    OpKind::C => {
                        for a in args {
                            let child = self.build(a, None);
                            self.child_edge(me, child);
                        }
                    }
                    OpKind::Ac => {
                        let mut frontier = Vec::new();
                        flatten(op, args, &mut frontier);
                        for a in frontier {
                            let child = self.build(a, None);
                            self.child_edge(me, child);
                        }
                    }
                }
                me
            }
            STerm::Exists(scope, vars, body) => {
                let me = self.vertex(Label::Binder(*scope), pos);
                let mut bound = BTreeMap::new();
                for v in vars {
                    let w = self.vertex(Label::BoundVar(*scope, v.sort), None);
                    self.child_edge(me, w);
                    bound.insert(*v, w);
                    match scope {
                        Scope::Root => self.graph.free_vars.push((w, *v)),
                        Scope::Witnesses => self.graph.witness_vars.push((w, *v)),
                    }
                }
                self.scopes.push((*scope, bound));
                let child = self.build(body, None);
                self.scopes.pop();
                self.child_edge(me, child);
                me
            }
        }
    }
}

/// The arguments of an AC application of `op`, with nested applications of
/// `op` replaced by their own arguments (the reduction's frontier).
fn flatten<'t>(op: &Op, args: &'t [STerm], out: &mut Vec<&'t STerm>) {
    for a in args {
        match a {
            STerm::App(inner, inner_args) if inner == op => flatten(op, inner_args, out),
            _ => out.push(a),
        }
    }
}

// =============================================================================
// Step 3: the canonical form of a term
// =============================================================================
//
// The variables of the system term renamed by the canonical renaming, then
// the whole normalized modulo the AC and C operators: the canonical form.

/// θ from a labeling that puts vertex `w` at canonical position
/// `position(w)`: the free variables and the witnesses, named per sort
/// (`mv0`, `fv0`, `tv0`, …) in the order of their vertices' positions.
fn labelling_from_positions(
    graph: &TermGraph,
    position: impl Fn(usize) -> usize,
) -> CanonLabelling {
    let mut vars: Vec<(usize, LVar)> = graph
        .free_vars
        .iter()
        .chain(&graph.witness_vars)
        .map(|&(w, v)| (position(w), v))
        .collect();
    vars.sort();
    let mut labelling = CanonLabelling::empty();
    for (_, v) in &vars {
        labelling.bind_next(*v);
    }
    labelling
}

/// `t` with every variable, free or witness, renamed by `theta`. A variable
/// `theta` does not cover panics ([`rename_var`]): the labelling names every
/// variable of the system term, so a miss is a translation bug.
pub fn rename(t: &STerm, theta: &BTreeMap<LVar, LVar>) -> STerm {
    match t {
        STerm::Var(VarRef::Free(v)) => STerm::Var(VarRef::Free(rename_var(theta, *v))),
        STerm::Var(VarRef::Witness(v)) => STerm::Var(VarRef::Witness(rename_var(theta, *v))),
        STerm::Const(_) => t.clone(),
        STerm::App(op, args) => {
            STerm::App(op.clone(), args.iter().map(|a| rename(a, theta)).collect())
        }
        STerm::Exists(scope, vars, body) => STerm::Exists(
            *scope,
            vars.iter().map(|v| rename_var(theta, *v)).collect(),
            Box::new(rename(body, theta)),
        ),
    }
}

/// The normal form of `t` modulo the AC and C operators ([`Op::kind`]):
/// nested applications of an AC operator are flattened into one, as
/// [`term_to_graph`] flattens them, and the arguments of AC and C operators
/// and a binder's variables are sorted. Duplicates stay: an AC operator is a
/// multiset.
pub fn normalize(t: STerm) -> STerm {
    match t {
        STerm::App(op, args) => {
            let args: Vec<STerm> = args.into_iter().map(normalize).collect();
            match op.kind() {
                OpKind::Free => STerm::App(op, args),
                OpKind::C => {
                    let mut args = args;
                    args.sort();
                    STerm::App(op, args)
                }
                OpKind::Ac => {
                    let mut flat = Vec::with_capacity(args.len());
                    for a in args {
                        match a {
                            // Already normalized, hence flat and sorted.
                            STerm::App(inner, inner_args) if inner == op => flat.extend(inner_args),
                            a => flat.push(a),
                        }
                    }
                    flat.sort();
                    STerm::App(op, flat)
                }
            }
        }
        STerm::Exists(scope, mut vars, body) => {
            vars.sort();
            STerm::Exists(scope, vars, Box::new(normalize(*body)))
        }
        t @ (STerm::Var(_) | STerm::Const(_)) => t,
    }
}

/// A SHA-256 fingerprint of `t`, through its derived `Hash`: the variables
/// hash by name, sort and index, the operators and constants by content, and
/// every variable-length part with its length, so different terms feed
/// different byte sequences.
pub fn fingerprint_sterm(t: &STerm) -> Fingerprint {
    let mut h = Sha256Hasher(FingerprintHasher::new());
    std::hash::Hash::hash(t, &mut h);
    h.0.finish()
}

// =============================================================================
// A cheap invariant of the form
// =============================================================================

/// The name every variable takes in [`abstract_vars`]. It starts with a NUL,
/// like a witness ([`WITNESS_MARK`]), so it names no variable of a theory or
/// of the solver; it only ever meets other placeholders, in a hash.
const PLACEHOLDER: &str = "\0sort";

/// How often each variable, free or witness, occurs in a term; a binder's
/// variables are not occurrences.
type Occurrences = FastMap<VarRef, u64>;

fn count_occurrences(t: &STerm, counts: &mut Occurrences) {
    match t {
        STerm::Var(v) => *counts.entry(*v).or_insert(0) += 1,
        STerm::Const(_) => {}
        STerm::App(_, args) => args.iter().for_each(|a| count_occurrences(a, counts)),
        STerm::Exists(_, _, body) => count_occurrences(body, counts),
    }
}

/// The placeholder of `v`: its sort, and in place of an index, how often `v`
/// occurs.
fn placeholder(v: VarRef, counts: &Occurrences) -> LVar {
    let (VarRef::Free(x) | VarRef::Witness(x)) = v;
    LVar {
        name: PLACEHOLDER,
        sort: x.sort,
        idx: counts.get(&v).copied().unwrap_or(0),
    }
}

/// `t` with every variable, free or witness, and every binder's variables
/// replaced by its placeholder: one per sort and number of occurrences in
/// `t`. A free variable stays a `VarRef::Free` and a witness a
/// `VarRef::Witness`, as renaming keeps them. Names, de Bruijn indices and
/// closed subterms are constants and stay. Counts first, in a pass of its
/// own: a placeholder needs its variable's count before anything sorts it.
pub fn abstract_vars(t: &STerm) -> STerm {
    let mut counts = Occurrences::default();
    count_occurrences(t, &mut counts);
    abstract_counted(t, &counts)
}

fn abstract_counted(t: &STerm, counts: &Occurrences) -> STerm {
    match t {
        STerm::Var(v @ VarRef::Free(_)) => STerm::Var(VarRef::Free(placeholder(*v, counts))),
        STerm::Var(v @ VarRef::Witness(_)) => STerm::Var(VarRef::Witness(placeholder(*v, counts))),
        STerm::Const(_) => t.clone(),
        STerm::App(op, args) => STerm::App(
            op.clone(),
            args.iter().map(|a| abstract_counted(a, counts)).collect(),
        ),
        STerm::Exists(scope, vars, body) => {
            // The root binder binds the free variables, the other the
            // witnesses.
            let var_ref = |v: LVar| match scope {
                Scope::Root => VarRef::Free(v),
                Scope::Witnesses => VarRef::Witness(v),
            };
            STerm::Exists(
                *scope,
                vars.iter()
                    .map(|v| placeholder(var_ref(*v), counts))
                    .collect(),
                Box::new(abstract_counted(body, counts)),
            )
        }
    }
}

/// A cheap invariant of a system term ([`system_to_term`]): its fingerprint
/// once every variable is the placeholder of its sort and its number of
/// occurrences ([`abstract_vars`]) and the result is normalized modulo AC
/// and C ([`normalize`]). No bliss, no graph.
///
/// Two systems with the same canonical form have the same invariant, so a
/// different invariant proves the forms differ, and only an equal one leaves
/// the question to bliss. The argument rests on the form alone, not on the
/// faithfulness claims (A) and (B) of the module docs: a form is
/// `normalize(rename(body, θ))` for a θ that keeps sorts and keeps free
/// variables free and witnesses witnesses, so
///
/// - renaming by θ does not change the abstraction: a variable and its image
///   have the same sort and occur equally often, so they get the same
///   placeholder;
/// - flattening and reordering AC and C arguments keeps every occurrence
///   (`normalize` keeps duplicates), so they leave the counts alone; the
///   abstraction is then a substitution, under which equality modulo AC and
///   C is preserved, and `normalize` is a normal form for that equality, so
///   normalizing before abstracting does not change the result;
/// - the binders bind exactly the variables of the body, which θ maps one to
///   one, so their abstracted lists, sorted, agree too.
///
/// Normalizing after abstracting is what makes it an invariant: `x = 'a'` and
/// `'a' = x` (`=` is C), or an AC section whose entries are ordered by their
/// variables' names, abstract to terms that are equal only modulo AC and C.
/// Witnesses are abstracted for the same reason: their names come from raw
/// positions in the equation store ([`witness`]).
///
/// Coarser than the form: systems that differ only in how variables that
/// occur equally often are linked share it: `<x, y>` next to `<'a', <y, x>>`,
/// against `<x, y>` next to `<'a', <x, y>>`.
pub fn invariant(term: &STerm) -> Fingerprint {
    fingerprint_sterm(&normalize(abstract_vars(term)))
}

/// [`FingerprintHasher`] as a [`std::hash::Hasher`], for a derived `Hash`.
/// Each write is length-prefixed ([`FingerprintHasher::bytes`]).
struct Sha256Hasher(FingerprintHasher);

impl std::hash::Hasher for Sha256Hasher {
    fn write(&mut self, bytes: &[u8]) {
        self.0.bytes(bytes);
    }

    /// Never read: the digest is [`fingerprint_sterm`]'s result.
    fn finish(&self) -> u64 {
        unreachable!("Sha256Hasher is read through FingerprintHasher::finish")
    }
}

// =============================================================================
// The driver
// =============================================================================

/// What the driver does beyond computing the canonical form.
#[derive(Debug, Clone, Copy, Default)]
pub struct TermGraphOptions {
    /// Check that every automorphism generator bliss reports is a symmetry of
    /// the system ([`TermGraphStats::non_genuine_generators`]). Costs one
    /// content pass per generator that moves a free variable.
    pub check_generators: bool,
    /// Compute [`TermGraphStats::certificate`]; runs bliss even when the
    /// system has no free variables.
    pub certificate: bool,
}

/// Measurements of one canonicalization.
#[derive(Debug, Clone, Default)]
pub struct TermGraphStats {
    pub vertices: usize,
    pub edges: usize,
    pub free_vars: usize,
    /// Whether bliss ran: it is skipped when the system has no free variable
    /// and no certificate was asked for.
    pub bliss_ran: bool,
    /// `|Aut|` of the term graph, as bliss prints it.
    pub aut_size: Option<String>,
    pub bliss_search_nodes: u64,
    pub generators: usize,
    /// Generators that move at least one free-variable vertex.
    pub generators_moving_free_vars: usize,
    /// Generators that move at least one variable vertex, a free variable's
    /// or a witness's: the ones that can change θ, and so the ones the
    /// generator check checks. The others only permute the inside of the
    /// term (identical AC arguments).
    pub generators_moving_vars: usize,
    /// Generators under which the canonical form changes: each one is a
    /// symmetry of the graph that is not one of the system, i.e. the
    /// encoding lost information. Counted only with
    /// [`TermGraphOptions::check_generators`].
    pub non_genuine_generators: usize,
    /// What the generators permute: per generator, each distinct
    /// `parent > subtree` pair of labels of a moved subtree whose parent stays
    /// fixed (see [`generator_moves`]).
    pub generator_moves: BTreeMap<String, usize>,
    /// [`Self::generator_moves`] of the non-genuine generators only.
    pub non_genuine_moves: BTreeMap<String, usize>,
    /// The sections of the form ([`SYSTEM_SECTIONS`]) a non-genuine
    /// generator changes.
    pub non_genuine_fields: BTreeMap<String, usize>,
    /// A hash of bliss's canonical graph: the key table, the colors in
    /// canonical order and the canonical edges. Equal for isomorphic graphs;
    /// whether it is also only equal for equivalent systems is what comparing
    /// it with the form measures.
    pub certificate: Option<Fingerprint>,
    pub time_term: Duration,
    pub time_graph: Duration,
    pub time_bliss: Duration,
    pub time_content: Duration,
    pub time_check: Duration,
}

/// Canonicalizes `sys` through its term graph (see the module docs),
/// returning the canonical form, the labelling θ it was built with, and the
/// measurements.
///
/// The steps: the term and its graph; bliss's canonical labeling of the
/// graph; θ names the free variables and the witnesses, per sort, in the
/// order of their vertices' canonical positions; the form is the term
/// renamed by θ and normalized.
pub fn canonicalize_constraint_system_via_term_graph_with_labelling(
    sys: &System,
    options: TermGraphOptions,
) -> Result<(CanonicalSystem, CanonLabelling, TermGraphStats), BlissError> {
    let start = Instant::now();
    let term = system_to_term(sys);
    let time_term = start.elapsed();
    let (canon, labelling, mut stats) = canonicalize_term(&term, options)?;
    stats.time_term = time_term;
    Ok((canon, labelling, stats))
}

/// [`canonicalize_constraint_system_via_term_graph_with_labelling`] from the
/// system's term ([`system_to_term`]): everything after the term needs only
/// the term, so a caller that keeps a system's term can canonicalize it after
/// the system is gone. `time_term` stays zero.
pub fn canonicalize_term(
    term: &STerm,
    options: TermGraphOptions,
) -> Result<(CanonicalSystem, CanonLabelling, TermGraphStats), BlissError> {
    let mut stats = TermGraphStats::default();

    let start = Instant::now();
    let graph = term_to_graph(term);
    stats.vertices = graph.keys.len();
    stats.edges = graph.edges.len();
    stats.free_vars = graph.free_vars.len();
    stats.time_graph = start.elapsed();

    let has_vars = !graph.free_vars.is_empty() || !graph.witness_vars.is_empty();
    let mut labeling: Option<Permutation> = None;
    let mut generators: Vec<Permutation> = Vec::new();
    if has_vars || options.certificate {
        let start = Instant::now();
        let (result, bliss_stats) = run_bliss_with_stats(&to_dimacs(&graph.colors, &graph.edges)?)?;
        stats.time_bliss = start.elapsed();
        stats.bliss_ran = true;
        stats.aut_size = Some(bliss_stats.aut_size);
        stats.bliss_search_nodes = bliss_stats.search_nodes;
        stats.generators = result.generators.len();
        if options.certificate {
            stats.certificate = Some(certificate(&graph, &result.canonical_labeling));
        }
        labeling = Some(result.canonical_labeling);
        generators = result.generators;
    }
    let position = |w: usize| {
        labeling
            .as_ref()
            .expect("bliss runs whenever there is a variable")
            .image_of(w)
    };

    let labelling = labelling_from_positions(&graph, position);

    let start = Instant::now();
    let canon = content_under(term, &labelling);
    stats.time_content = start.elapsed();

    // Any other canonical labeling is the canonical one composed with an
    // automorphism: under a symmetry of the system it names the variables
    // differently but gives the same form.
    let start = Instant::now();
    let variable_vertices: Vec<usize> = graph
        .free_vars
        .iter()
        .chain(&graph.witness_vars)
        .map(|&(w, _)| w)
        .collect();
    for g in &generators {
        let moves = generator_moves(&graph, g);
        for moved in &moves {
            *stats.generator_moves.entry(moved.clone()).or_insert(0) += 1;
        }
        if graph.free_vars.iter().any(|&(w, _)| g.image_of(w) != w) {
            stats.generators_moving_free_vars += 1;
        }
        if variable_vertices.iter().all(|&w| g.image_of(w) == w) {
            continue;
        }
        stats.generators_moving_vars += 1;
        if options.check_generators {
            let moved = labelling_from_positions(&graph, |w| position(g.image_of(w)));
            let other = content_under(term, &moved);
            if other != canon {
                stats.non_genuine_generators += 1;
                for moved in &moves {
                    *stats.non_genuine_moves.entry(moved.clone()).or_insert(0) += 1;
                }
                let report = crate::canon_fingerprint::compare_fingerprints(
                    &crate::canon_fingerprint::fingerprint_constraint_system(&canon),
                    &crate::canon_fingerprint::fingerprint_constraint_system(&other),
                );
                for field in report.mismatched_fields() {
                    *stats
                        .non_genuine_fields
                        .entry(field.to_string())
                        .or_insert(0) += 1;
                }
            }
        }
    }
    stats.time_check = start.elapsed();

    Ok((canon, labelling, stats))
}

/// The distinct `parent > subtree` label pairs of the subtrees `g` moves
/// whose parent it fixes: what a symmetry of the term graph swaps, e.g.
/// `EqDisj > EqAlt` (two alpha-equivalent alternatives) or
/// `Binder(Witnesses) > BoundVar(Witnesses, Msg)` (two witnesses).
pub fn generator_moves(graph: &TermGraph, g: &Permutation) -> BTreeSet<String> {
    (0..graph.keys.len())
        .filter(|&v| g.image_of(v) != v)
        .filter_map(|v| {
            let parent = graph.parents[v]?;
            (g.image_of(parent) == parent).then(|| {
                format!(
                    "{} > {}",
                    label_name(&graph.keys[parent]),
                    label_name(&graph.keys[v])
                )
            })
        })
        .collect()
}

/// A short name for a vertex's label, for [`generator_moves`].
fn label_name(key: &VKey) -> String {
    match &key.label {
        Label::Op(op) => match op {
            Op::Fun(sym) => format!("Fun({})", fun_sym_name(sym)),
            Op::Fact(tag) => format!("Fact({})", crate::fact::fact_tag_name(tag)),
            Op::RuleInst(_) => "RuleInst".into(),
            other => {
                let shown = format!("{other}");
                shown.split('[').next().unwrap_or(&shown).to_string()
            }
        },
        Label::Const(c) => match c {
            ConstKey::Closed(kind, _) => format!("Closed({kind:?})"),
            _ => "Const".into(),
        },
        Label::Binder(scope) => format!("Binder({scope:?})"),
        Label::BoundVar(scope, sort) => format!("BoundVar({scope:?},{sort:?})"),
        Label::Occ(sort) => format!("Occ({sort:?})"),
    }
}

/// The canonical form of the system term `term` under `labelling`: its
/// `System[…]` body renamed and normalized. A variable `labelling` does not
/// name panics.
fn content_under(term: &STerm, labelling: &CanonLabelling) -> CanonicalSystem {
    let STerm::Exists(Scope::Root, _, witnesses) = term else {
        panic!("system_to_term built a term without the root binder: {term}")
    };
    let STerm::Exists(Scope::Witnesses, _, body) = &**witnesses else {
        panic!("system_to_term built a term without the witness binder: {term}")
    };
    CanonicalSystem::new(normalize(rename(body, labelling.theta())))
}

/// `goal` translated like a goal of the system term, without its solved flag,
/// not renamed yet: [`under`] its system's labelling, it is the goal as the
/// system's canonical form writes it. `None` for a `Split` goal, which the
/// form leaves out.
pub(crate) fn goal_term(goal: &Goal) -> Option<STerm> {
    let (mut free, mut witnesses) = (BTreeSet::new(), BTreeSet::new());
    let mut tr = Translator {
        free: &mut free,
        witnesses: &mut witnesses,
    };
    tr.goal(goal)
}

/// `disj`, the equation store's disjunction number `conj_index`, translated
/// with the witnesses it has in the system term ([`witness`]), not renamed
/// yet: [`under`] its system's labelling, it is the disjunction as the
/// system's canonical form writes it.
pub(crate) fn eq_disj_term(disj: &EqDisj, conj_index: usize) -> STerm {
    let (mut free, mut witnesses) = (BTreeSet::new(), BTreeSet::new());
    let mut tr = Translator {
        free: &mut free,
        witnesses: &mut witnesses,
    };
    tr.eq_disj(conj_index, disj)
}

/// A part of a system's term (a goal, a disjunction) as the system's
/// canonical form writes it: renamed by the system's `labelling`, normalized.
pub(crate) fn under(part: &STerm, labelling: &CanonLabelling) -> STerm {
    normalize(rename(part, labelling.theta()))
}

/// A hash of `graph` relabeled by `labeling` (see
/// [`TermGraphStats::certificate`]). Colors are ranks among this graph's
/// keys, so the key table itself is hashed too: two graphs with different
/// keys can share every rank.
fn certificate(graph: &TermGraph, labeling: &Permutation) -> Fingerprint {
    let mut h = FingerprintHasher::new();
    let table: BTreeSet<&VKey> = graph.keys.iter().collect();
    h.u64(table.len() as u64);
    for key in table {
        h.bytes(format!("{key:?}").as_bytes());
    }
    let mut colors = vec![0; graph.colors.len()];
    for (v, c) in graph.colors.iter().enumerate() {
        colors[labeling.image_of(v)] = *c;
    }
    h.u64(colors.len() as u64);
    for c in colors {
        h.u64(u64::from(c));
    }
    let edges: BTreeSet<(usize, usize)> = graph
        .edges
        .iter()
        .map(|&(s, t)| (labeling.image_of(s), labeling.image_of(t)))
        .collect();
    h.u64(edges.len() as u64);
    for (s, t) in edges {
        h.u64(s as u64).u64(t as u64);
    }
    h.finish()
}

// =============================================================================
// Display (for dumps and tests)
// =============================================================================

impl fmt::Display for STerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            STerm::Var(VarRef::Free(v)) => write!(f, "{v}"),
            // A raw witness's name starts with a NUL.
            STerm::Var(VarRef::Witness(v)) => {
                write!(f, "{}", v.to_string().replace(WITNESS_MARK, "?"))
            }
            STerm::Const(c) => write!(f, "⟨{c}⟩"),
            STerm::Exists(scope, vars, body) => {
                write!(f, "∃_{scope:?}")?;
                for v in vars {
                    write!(f, " {}", v.to_string().replace(WITNESS_MARK, "?"))?;
                }
                write!(f, ". {body}")
            }
            STerm::App(op, args) => {
                let (open, close) = match op.kind() {
                    OpKind::Ac => ('{', '}'),
                    OpKind::Free | OpKind::C => ('(', ')'),
                };
                write!(f, "{op}{open}")?;
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{a}")?;
                }
                write!(f, "{close}")
            }
        }
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::System {
                source_kind,
                side,
                contradictory,
            } => write!(f, "System[{source_kind:?},{side:?},{contradictory}]"),
            Op::Section(s) => write!(f, "{s:?}"),
            Op::RuleInst(id) => match id {
                RuleId::Fresh => write!(f, "RuleInst[Fresh]"),
                RuleId::Proto(name) => write!(f, "RuleInst[{name}]"),
                RuleId::Intr(info) => write!(f, "RuleInst[{info:?}]"),
            },
            Op::FactGroup(g) => write!(f, "{g:?}"),
            Op::Fact(tag) => write!(f, "{}", crate::fact::fact_tag_name(tag)),
            Op::Fun(sym) => write!(f, "{}", fun_sym_name(sym)),
            Op::Edge(c, p) => write!(f, "Edge[{c},{p}]"),
            Op::GoalChain(c, p) => write!(f, "ChainG[{c},{p}]"),
            Op::GoalPremise(p) => write!(f, "PremiseG[{p}]"),
            Op::Goal { solved } => write!(f, "Goal[{}]", if *solved { "solved" } else { "open" }),
            Op::GQuant { ex, sorts } => {
                let sorts: Vec<String> = sorts.iter().map(|s| format!("{s:?}")).collect();
                write!(f, "{}[{}]", if *ex { "Ex" } else { "All" }, sorts.join(","))
            }
            other => write!(f, "{other:?}"),
        }
    }
}

impl fmt::Display for ConstKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConstKey::Name(n) => write!(f, "{}", LNTerm::Lit(Lit::Con(*n))),
            ConstKey::DeBruijn(i) => write!(f, "db{i}"),
            ConstKey::NoLast => write!(f, "NoLast"),
            ConstKey::Closed(kind, fp) => {
                write!(f, "{kind:?}:")?;
                for b in &fp[..4] {
                    write!(f, "{b:02x}")?;
                }
                Ok(())
            }
        }
    }
}

/// A function symbol's name, for display.
fn fun_sym_name(sym: &FunSym) -> String {
    match sym {
        FunSym::NoEq(s) => String::from_utf8_lossy(s.name).into_owned(),
        other => format!("{other:?}"),
    }
}

#[cfg(test)]
#[path = "canon_term_graph_tests.rs"]
mod tests;
