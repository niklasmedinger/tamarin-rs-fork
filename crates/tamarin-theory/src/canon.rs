//! The canonical form of a constraint system and its parts that are not
//! terms of the system term (`canon_term_graph`):
//!
//! - facts lifted to `LNTerm`s ([`fact_to_term`]), for the fingerprint of a
//!   fact without variables, which the system term keeps as one constant;
//! - a rule instance's identity ([`rule_id`]): two instances of differently
//!   named rules with the same facts are different proof-search states,
//!   since a proof names the rule it uses and rule names can enter the
//!   ranking (oracles, tactics). SAPIC produces such pairs
//!   (`sapic/fast/basic/channels1`'s `incx_1_121` and `incx_3_121`). A
//!   rule's `new_vars` are left out: they are derived from its facts;
//! - guarded formulas in AC normal form ([`ac_normalize_guarded`]) and their
//!   fingerprints ([`fingerprint_guarded`]);
//! - [`CanonicalSystem`], the entry points
//!   [`canonicalize_constraint_system`] and
//!   [`canonicalize_constraint_system_with_labelling`], and proof methods
//!   renamed into a system's canonical form ([`canonicalize_proof_method`]).

use std::sync::Arc;

use tamarin_term::function_symbols::{Constructability, FunSym, NoEqSym, Privacy};
use tamarin_term::lterm::LNTerm;
use tamarin_term::term::f_app_list;

use crate::fact::{FactTag, LNFact};
use crate::rule::{IntrRuleACInfo, ProtoRuleName, RuleACInst, RuleInfo};

/// A NUL-prefixed marker name for `tag`, distinct for every distinct
/// `FactTag` (the variant name is baked into the string, so e.g. a
/// user-defined `Proto` fact literally named `"Fresh"` can never collide
/// with `FactTag::Fresh`'s marker). The NUL prefix additionally guarantees
/// this can never collide with a real function symbol drawn from a Tamarin
/// model's signature, since no valid Tamarin identifier contains one.
fn fact_tag_marker_name(tag: &FactTag) -> Vec<u8> {
    let suffix = match tag {
        FactTag::Proto(mult, name, arity) => format!("Proto:{mult:?}:{name}:{arity}"),
        FactTag::Fresh => "Fresh".to_string(),
        FactTag::Out => "Out".to_string(),
        FactTag::In => "In".to_string(),
        FactTag::Ku => "Ku".to_string(),
        FactTag::Kd => "Kd".to_string(),
        FactTag::Ded => "Ded".to_string(),
        FactTag::Term => "Term".to_string(),
    };
    let mut name = vec![0u8];
    name.extend_from_slice(suffix.as_bytes());
    name
}

/// The 0-ary marker term for a fact's tag (see the module docs).
fn fact_tag_marker(tag: &FactTag) -> LNTerm {
    let sym = NoEqSym::new(
        fact_tag_marker_name(tag),
        0,
        Privacy::Public,
        Constructability::Constructor,
    );
    LNTerm::App(FunSym::NoEq(sym), Arc::from([]))
}

/// Lifts a fact `F(t_1, ..., t_n)` to the term `List(marker(F), t_1, ...,
/// t_n)`: the marker is a function symbol, so two facts with different tags
/// never lift to equal terms. Annotations are left out (solver hints).
pub fn fact_to_term(fact: &LNFact) -> LNTerm {
    let mut args = Vec::with_capacity(fact.terms.len() + 1);
    args.push(fact_tag_marker(&fact.tag));
    args.extend(fact.terms.iter().cloned());
    f_app_list(args)
}

/// A rule's identity: what tells two rule instances with the same facts
/// apart. Intruder rules carry their full info, so a protocol rule named
/// like an intruder rule never collides with it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RuleId {
    Fresh,
    Proto(&'static str),
    Intr(IntrRuleACInfo),
}

/// The identity of `ru`'s rule.
pub fn rule_id(ru: &RuleACInst) -> RuleId {
    match &ru.info {
        RuleInfo::Proto(p) => match p.name {
            ProtoRuleName::Fresh => RuleId::Fresh,
            ProtoRuleName::Stand(s) => RuleId::Proto(s),
        },
        RuleInfo::Intr(info) => RuleId::Intr(info.clone()),
    }
}

// =============================================================================
// Guarded formulas
// =============================================================================
//
// A formula enters the system term in AC normal form
// ([`ac_normalize_guarded`]): conjunctions and disjunctions flattened,
// deduplicated (`gconj`/`gdisj`) and sorted, the sides of `=` ordered, a
// quantifier's guards sorted, AC operands inside terms sorted, `em`'s
// arguments sorted, binder display names and fact annotations erased. Bound
// variables are de Bruijn indices, so only their position counts, and the
// order inside a binder block stays significant. A formula without free
// variables becomes one constant of the system term, keyed by
// [`fingerprint_guarded`]; the free variables of the others are renamed with
// the rest of the system term, which sorts the AC parts again afterwards.

use crate::guarded::{
    self, cmp_atom, cmp_guarded, cmp_term, ga, gall, gex, BVar, GAtom,
    GBinding, GFact, GTerm, Guarded, Quant,
};
use tamarin_parser::ast as p;
use tamarin_term::lterm::LSort;
use tamarin_utils::fingerprint::{Fingerprint, FingerprintHasher};

/// `p::SortHint` to `LSort`. `Untagged` (a bare, sigil-less name) resolves
/// to `Msg`, matching the parser's own default for that position; every
/// other hint (including the `:msg|:pub|...` suffix spelling) maps to its
/// named sort. Small, self-contained duplicate of the same conversion this
/// crate already carries at several call sites (e.g.
/// `constraint::solver::sources::varspec_sort_to_lsort`,
/// `elaborate::lnterm_to_term`) — kept local rather than imported across a
/// module boundary that has no other reason to depend on this one.
fn sort_hint_to_lsort(s: p::SortHint) -> LSort {
    use p::{SortHint as S, SuffixSort as SS};
    match s {
        S::Msg | S::Untagged | S::Suffix(SS::Msg) => LSort::Msg,
        S::Pub | S::Suffix(SS::Pub) => LSort::Pub,
        S::Fresh | S::Suffix(SS::Fresh) => LSort::Fresh,
        S::Node | S::Suffix(SS::Node) => LSort::Node,
        S::Nat | S::Suffix(SS::Nat) => LSort::Nat,
    }
}

/// A parser-AST variable spec as the `LVar` it denotes.
pub(crate) fn varspec_to_lvar(v: &p::VarSpec) -> tamarin_term::lterm::LVar {
    tamarin_term::lterm::LVar::new(v.name.as_str(), sort_hint_to_lsort(v.sort), v.idx)
}

/// Canonically orders an atom's operands where doing so is sound: `=` is
/// symmetric (`s = t` and `t = s` are the same atom), so its two sides are
/// reordered smaller-first by [`cmp_term`]. Every other `GAtom` variant is
/// directional (`Less`/`Subterm`, or simply not binary) and is returned
/// unchanged.
///
/// A fact's annotations are dropped: they are hints to the solver's goal
/// ranking with no meaning of their own, and [`fact_to_term`] leaves them out
/// of rule and goal facts too.
fn normalize_atom(a: &GAtom) -> GAtom {
    let t = ac_normalize_term;
    let f = |fact: &GFact| GFact {
        persistent: fact.persistent,
        name: fact.name.clone(),
        args: fact.args.iter().map(ac_normalize_term).collect(),
        annotations: Vec::new(),
    };
    match a {
        GAtom::Eq(x, y) => {
            let (x, y) = (t(x), t(y));
            // Compare the AC-NORMALIZED sides: swapping has to be decided
            // after flattening/sorting any AC operator each side carries,
            // not before — e.g. `xor(b,a) = c` and `c = xor(a,b)` must
            // settle on the same orientation, which only holds once both
            // `xor`s are already in their sorted form.
            if cmp_term(&x, &y) == std::cmp::Ordering::Greater {
                GAtom::Eq(y, x)
            } else {
                GAtom::Eq(x, y)
            }
        }
        GAtom::Less(x, y) => GAtom::Less(t(x), t(y)),
        GAtom::LessMset(x, y) => GAtom::LessMset(t(x), t(y)),
        GAtom::Subterm(x, y) => GAtom::Subterm(t(x), t(y)),
        GAtom::Action(fact, time) => GAtom::Action(f(fact), t(time)),
        GAtom::Last(x) => GAtom::Last(t(x)),
        GAtom::Pred(fact) => GAtom::Pred(f(fact)),
    }
}

/// Brings a term reachable from a guarded formula's atoms into `CAN_AC`
/// normal form: recursively normalizes every subterm bottom-up, then at
/// each AC operator (`xor`/`union`/`mult`/`tplus`/a user `[AC]` symbol,
/// i.e. [`crate::guarded::is_ac_binop`]) flattens the (now-normalized)
/// chain and sorts the flattened operands by [`cmp_term`] before rebuilding
/// — the same flatten-then-sort `CAN_AC` already applies to `LNTerm` via
/// `f_app_ac` (`tamarin_term::term`), just rebuilt here as a right-nested
/// `GTerm::BinOp` chain since `GTerm` (unlike `Term`) has no n-ary AC
/// application node to flatten INTO.
///
/// The bilinear-pairing C symbol `em` — commutative but not associative,
/// parsed as `GTerm::App("em", [_, _])` (see
/// [`crate::guarded::funsym_key`]'s special case) — gets the same
/// treatment minus the flattening: its two (normalized) arguments are
/// sorted, never merged with a nested `em`.
fn ac_normalize_term(t: &GTerm) -> GTerm {
    match t {
        GTerm::BinOp(op, a, b) if guarded::is_ac_binop(op) => {
            let combined = GTerm::BinOp(*op, ga(ac_normalize_term(a)), ga(ac_normalize_term(b)));
            let mut leaves = Vec::new();
            guarded::flatten_ac_binop(op, &combined, &mut leaves);
            let mut sorted: Vec<GTerm> = leaves.into_iter().cloned().collect();
            sorted.sort_by(cmp_term);
            rebuild_ac_chain(*op, sorted)
        }
        GTerm::BinOp(op, a, b) => {
            GTerm::BinOp(*op, ga(ac_normalize_term(a)), ga(ac_normalize_term(b)))
        }
        GTerm::App(n, args) if &**n == "em" && args.len() == 2 => {
            let mut sorted: Vec<GTerm> = args.iter().map(ac_normalize_term).collect();
            sorted.sort_by(cmp_term);
            GTerm::App(n.clone(), sorted.into())
        }
        GTerm::App(n, args) => GTerm::App(n.clone(), args.iter().map(ac_normalize_term).collect()),
        GTerm::AlgApp(n, a, b) => GTerm::AlgApp(
            n.clone(),
            ga(ac_normalize_term(a)),
            ga(ac_normalize_term(b)),
        ),
        GTerm::Pair(items) => GTerm::Pair(items.iter().map(ac_normalize_term).collect()),
        GTerm::Diff(a, b) => GTerm::Diff(ga(ac_normalize_term(a)), ga(ac_normalize_term(b))),
        GTerm::PatMatch(x) => GTerm::PatMatch(ga(ac_normalize_term(x))),
        GTerm::Var(_)
        | GTerm::PubLit(_)
        | GTerm::FreshLit(_)
        | GTerm::NatLit(_)
        | GTerm::Number(_)
        | GTerm::NumberOne
        | GTerm::NatOne
        | GTerm::DhNeutral => t.clone(),
    }
}

/// Rebuilds a flattened, sorted (>= 2 elements — an actual `BinOp` node
/// always flattens to at least its own two leaves) AC operand list as a
/// right-nested `BinOp` chain, preserving the sorted left-to-right order:
/// `[s0, s1, s2]` becomes `BinOp(op, s0, BinOp(op, s1, s2))`.
fn rebuild_ac_chain(op: p::BinOp, sorted: Vec<GTerm>) -> GTerm {
    let mut rev = sorted.into_iter().rev();
    let mut acc = rev
        .next()
        .expect("an AC BinOp always flattens to at least two leaves");
    for x in rev {
        acc = GTerm::BinOp(op, ga(x), ga(acc));
    }
    acc
}

/// Brings `g` into `CAN_AC` normal form: recursively normalizes every
/// subformula bottom-up, then at each level — symmetric `=` operands
/// reordered, a quantifier's guard atoms sorted, and `Conj`/`Disj` children
/// flattened, deduplicated (via [`crate::guarded::gconj`] /
/// [`crate::guarded::gdisj`]), and sorted — all by the total orders
/// [`cmp_term`]/[`cmp_atom`]/[`cmp_guarded`] already established in
/// `guarded.rs`. Every quantifier's binder list also has its (purely
/// cosmetic) display names erased, since only each binder's De-Bruijn
/// POSITION is semantically meaningful — see the module docs.
pub fn ac_normalize_guarded(g: &Guarded) -> Guarded {
    match g {
        Guarded::Atom(a) => Guarded::Atom(normalize_atom(a)),
        Guarded::Disj(items) => {
            let mut normalized: Vec<Guarded> = items.iter().map(ac_normalize_guarded).collect();
            normalized.sort_by(cmp_guarded);
            guarded::gdisj(normalized)
        }
        Guarded::Conj(items) => {
            let mut normalized: Vec<Guarded> = items.iter().map(ac_normalize_guarded).collect();
            normalized.sort_by(cmp_guarded);
            guarded::gconj(normalized)
        }
        Guarded::GGuarded {
            qua,
            vars,
            guards,
            body,
        } => {
            let anonymous_vars: Vec<GBinding> = vars
                .iter()
                .map(|b| GBinding {
                    name: String::new(),
                    sort: b.sort,
                })
                .collect();
            let mut sorted_guards: Vec<GAtom> = guards.iter().map(normalize_atom).collect();
            sorted_guards.sort_by(cmp_atom);
            let normalized_body = ac_normalize_guarded(body);
            match qua {
                Quant::All => gall(anonymous_vars, sorted_guards, normalized_body),
                Quant::Ex => gex(anonymous_vars, sorted_guards, normalized_body),
            }
        }
    }
}

/// Fingerprints a guarded formula, normally one in AC normal form
/// ([`ac_normalize_guarded`]). Shares its digest type with
/// [`tamarin_term::fingerprint::fingerprint_term`].
///
/// A Merkle hash, exactly like `fingerprint_term`: a compound node's
/// fingerprint is built from the fingerprints of its already-fingerprinted
/// children (the "subhash" composition), not their raw content, so a
/// `Guarded`'s `Conj`/`Disj`/`GGuarded` fingerprint is a function of its
/// subformulas' fingerprints, and (once a constraint-system-level
/// canonizer exists) a system's fingerprint can be built the same way from
/// its rules'/formulas' fingerprints in turn. Only a genuine leaf (a
/// literal, a name, a bare De-Bruijn index) ever hashes its own fields
/// directly — there's nothing further to recurse into.
pub fn fingerprint_guarded(g: &Guarded) -> Fingerprint {
    let mut h = FingerprintHasher::new();
    match g {
        Guarded::Atom(a) => {
            h.tag("Atom");
            h.digest(&fingerprint_atom(a));
        }
        Guarded::Disj(items) => {
            h.tag("Disj");
            h.u64(items.len() as u64);
            for it in items.iter() {
                h.digest(&fingerprint_guarded(it));
            }
        }
        Guarded::Conj(items) => {
            h.tag("Conj");
            h.u64(items.len() as u64);
            for it in items.iter() {
                h.digest(&fingerprint_guarded(it));
            }
        }
        Guarded::GGuarded {
            qua,
            vars,
            guards,
            body,
        } => {
            h.tag("GGuarded");
            hash_quant(&mut h, *qua);
            h.u64(vars.len() as u64);
            for v in vars.iter() {
                hash_binding(&mut h, v);
            }
            h.u64(guards.len() as u64);
            for a in guards.iter() {
                h.digest(&fingerprint_atom(a));
            }
            h.digest(&fingerprint_guarded(body));
        }
    }
    h.finish()
}

fn hash_quant(h: &mut FingerprintHasher, q: Quant) {
    h.u8(match q {
        Quant::All => 0,
        Quant::Ex => 1,
    });
}

fn hash_binding(h: &mut FingerprintHasher, b: &GBinding) {
    // `name` is purely cosmetic (erased to `""` by `ac_normalize_guarded`)
    // but still hashed for totality — a value that skipped normalization
    // would then also produce a different fingerprint, rather than
    // silently colliding with a properly-normalized one.
    h.bytes(b.name.as_bytes());
    hash_sort_hint(h, b.sort);
}

pub(crate) fn fingerprint_atom(a: &GAtom) -> Fingerprint {
    let mut h = FingerprintHasher::new();
    match a {
        GAtom::Eq(x, y) => {
            h.tag("Eq");
            h.digest(&fingerprint_gterm(x));
            h.digest(&fingerprint_gterm(y));
        }
        GAtom::Less(x, y) => {
            h.tag("Less");
            h.digest(&fingerprint_gterm(x));
            h.digest(&fingerprint_gterm(y));
        }
        GAtom::LessMset(x, y) => {
            h.tag("LessMset");
            h.digest(&fingerprint_gterm(x));
            h.digest(&fingerprint_gterm(y));
        }
        GAtom::Subterm(x, y) => {
            h.tag("Subterm");
            h.digest(&fingerprint_gterm(x));
            h.digest(&fingerprint_gterm(y));
        }
        GAtom::Action(fact, time) => {
            h.tag("Action");
            h.digest(&fingerprint_fact(fact));
            h.digest(&fingerprint_gterm(time));
        }
        GAtom::Last(x) => {
            h.tag("Last");
            h.digest(&fingerprint_gterm(x));
        }
        GAtom::Pred(fact) => {
            h.tag("Pred");
            h.digest(&fingerprint_fact(fact));
        }
    }
    h.finish()
}

fn fingerprint_fact(f: &GFact) -> Fingerprint {
    let mut h = FingerprintHasher::new();
    h.tag("GFact");
    h.u8(u8::from(f.persistent));
    h.bytes(f.name.as_bytes());
    h.u64(f.args.len() as u64);
    for a in f.args.iter() {
        h.digest(&fingerprint_gterm(a));
    }
    h.u64(f.annotations.len() as u64);
    for ann in f.annotations.iter() {
        h.u8(fact_annotation_byte(ann));
    }
    h.finish()
}

fn fact_annotation_byte(a: &p::FactAnnotation) -> u8 {
    match a {
        p::FactAnnotation::SolveFirst => 0,
        p::FactAnnotation::SolveLast => 1,
        p::FactAnnotation::NoSources => 2,
    }
}

pub(crate) fn fingerprint_gterm(t: &GTerm) -> Fingerprint {
    let mut h = FingerprintHasher::new();
    match t {
        GTerm::Var(v) => {
            h.tag("Var");
            hash_bvar(&mut h, v);
        }
        GTerm::PubLit(s) => {
            h.tag("PubLit");
            h.bytes(s.as_bytes());
        }
        GTerm::FreshLit(s) => {
            h.tag("FreshLit");
            h.bytes(s.as_bytes());
        }
        GTerm::NatLit(s) => {
            h.tag("NatLit");
            h.bytes(s.as_bytes());
        }
        GTerm::Number(n) => {
            h.tag("Number");
            h.u64(*n);
        }
        GTerm::NumberOne => {
            h.tag("NumberOne");
        }
        GTerm::NatOne => {
            h.tag("NatOne");
        }
        GTerm::DhNeutral => {
            h.tag("DhNeutral");
        }
        GTerm::App(n, args) => {
            h.tag("App");
            h.bytes(n.as_bytes());
            h.u64(args.len() as u64);
            for a in args.iter() {
                h.digest(&fingerprint_gterm(a));
            }
        }
        GTerm::AlgApp(n, a, b) => {
            h.tag("AlgApp");
            h.bytes(n.as_bytes());
            h.digest(&fingerprint_gterm(a));
            h.digest(&fingerprint_gterm(b));
        }
        GTerm::Pair(items) => {
            h.tag("Pair");
            h.u64(items.len() as u64);
            for it in items.iter() {
                h.digest(&fingerprint_gterm(it));
            }
        }
        GTerm::Diff(a, b) => {
            h.tag("Diff");
            h.digest(&fingerprint_gterm(a));
            h.digest(&fingerprint_gterm(b));
        }
        GTerm::BinOp(op, a, b) => {
            h.tag("BinOp");
            hash_binop(&mut h, op);
            h.digest(&fingerprint_gterm(a));
            h.digest(&fingerprint_gterm(b));
        }
        GTerm::PatMatch(x) => {
            h.tag("PatMatch");
            h.digest(&fingerprint_gterm(x));
        }
    }
    h.finish()
}

fn hash_binop(h: &mut FingerprintHasher, op: &p::BinOp) {
    match op {
        p::BinOp::Exp => {
            h.tag("Exp");
        }
        p::BinOp::Mult => {
            h.tag("Mult");
        }
        p::BinOp::Union => {
            h.tag("Union");
        }
        p::BinOp::Xor => {
            h.tag("Xor");
        }
        p::BinOp::NatPlus => {
            h.tag("NatPlus");
        }
        p::BinOp::AcFct(name) => {
            h.tag("AcFct");
            h.bytes(name.as_bytes());
        }
    }
}

fn hash_bvar(h: &mut FingerprintHasher, v: &BVar) {
    match v {
        BVar::Bound(n) => {
            h.tag("Bound");
            h.u64(*n as u64);
        }
        BVar::Free(vs) => {
            h.tag("Free");
            hash_varspec(h, vs);
        }
    }
}

fn hash_varspec(h: &mut FingerprintHasher, v: &p::VarSpec) {
    h.bytes(v.name.as_bytes());
    h.u64(v.idx);
    hash_sort_hint(h, v.sort);
    match &v.typ {
        Some(t) => {
            h.tag("Some");
            h.bytes(t.as_bytes());
        }
        None => {
            h.tag("None");
        }
    }
}

fn hash_sort_hint(h: &mut FingerprintHasher, s: p::SortHint) {
    use p::{SortHint as S, SuffixSort as SS};
    let b = match s {
        S::Msg => 0,
        S::Pub => 1,
        S::Fresh => 2,
        S::Node => 3,
        S::Nat => 4,
        S::Suffix(SS::Msg) => 5,
        S::Suffix(SS::Pub) => 6,
        S::Suffix(SS::Fresh) => 7,
        S::Suffix(SS::Node) => 8,
        S::Suffix(SS::Nat) => 9,
        // Should never occur in an ALREADY-CANONICAL formula (`subst_via_
        // theta_guarded`'s successful-lookup path always produces a
        // resolved `SortHint` via `lvar_to_varspec`), but still given a
        // distinct tag rather than panicking or aliasing a resolved sort.
        S::Untagged => 10,
    };
    h.u8(b);
}

// =============================================================================
// The canonical form of a constraint system
// =============================================================================

use crate::bliss_proc::BlissError;
use crate::canon_term_graph::{
    canonical_eq_disj, canonical_goal, Op, STerm, SYSTEM_SECTIONS,
};
use crate::constraint::constraints::Goal;
use crate::constraint::solver::proof_method::{ProofMethod, Result as FinishedResult};
use crate::constraint::solver::Contradiction;
use crate::constraint::system::{Side, SourceKind, System};
use tamarin_term::alpha_eq_ac::CanonLabelling;

/// The canonical form of a constraint `System`: its system term
/// (`canon_term_graph::system_to_term`) without the root binder, every
/// variable renamed by the canonical labelling and the whole normalized modulo
/// its AC and C operators (`canon_term_graph::normalize`). Two systems have
/// equal forms iff they are equal up to a sort-respecting renaming of their
/// variables and AC.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalSystem {
    /// `System[source_kind, side, contradictory](sections…)`.
    term: STerm,
}

impl CanonicalSystem {
    /// Panics unless `term` is a `System[…]` application with one argument
    /// per [`SYSTEM_SECTIONS`].
    pub(crate) fn new(term: STerm) -> Self {
        match &term {
            STerm::App(Op::System { .. }, sections) if sections.len() == SYSTEM_SECTIONS.len() => {}
            other => panic!(
                "a canonical form is a System[…] term with {} sections, got {other}",
                SYSTEM_SECTIONS.len()
            ),
        }
        CanonicalSystem { term }
    }

    pub fn term(&self) -> &STerm {
        &self.term
    }

    /// The arguments of the `System[…]` body, named by [`SYSTEM_SECTIONS`].
    pub fn sections(&self) -> &[STerm] {
        match &self.term {
            STerm::App(_, sections) => sections,
            _ => unreachable!("checked by CanonicalSystem::new"),
        }
    }

    /// The section named `name` in [`SYSTEM_SECTIONS`]; panics on an unknown
    /// name.
    pub fn section(&self, name: &str) -> &STerm {
        let i = SYSTEM_SECTIONS
            .iter()
            .position(|s| *s == name)
            .unwrap_or_else(|| panic!("no section {name:?} in {SYSTEM_SECTIONS:?}"));
        &self.sections()[i]
    }

    /// The `System[…]` label: `(source_kind, side, contradictory)`, the last
    /// being the subterm store's flag.
    pub fn label(&self) -> (Option<SourceKind>, Option<Side>, bool) {
        match &self.term {
            STerm::App(
                Op::System {
                    source_kind,
                    side,
                    contradictory,
                },
                _,
            ) => (*source_kind, *side, *contradictory),
            _ => unreachable!("checked by CanonicalSystem::new"),
        }
    }
}

/// Canonicalizes a whole constraint `System`: two systems get the same form
/// iff they are equal up to a sort-respecting renaming of their variables and
/// AC. bliss's canonical labeling of the system's term graph names the
/// variables, and the form is the system term renamed by it
/// ([`crate::canon_term_graph`]).
///
/// Fallible only through the bliss subprocess.
pub fn canonicalize_constraint_system(sys: &System) -> Result<CanonicalSystem, BlissError> {
    canonicalize_constraint_system_with_labelling(sys).map(|(canon, _)| canon)
}

/// [`canonicalize_constraint_system`], also returning the labelling θ that
/// maps `sys` onto its canonical form, for canonicalizing anything else derived
/// from `sys` consistently with it (its proof methods,
/// [`canonicalize_proof_method`]). When `sys` has a symmetry, another
/// renaming gives the same form, so anything canonicalized through it should
/// be compared as a multiset, which a symmetry maps onto itself.
pub fn canonicalize_constraint_system_with_labelling(
    sys: &System,
) -> Result<(CanonicalSystem, CanonLabelling), BlissError> {
    crate::canon_term_graph::canonicalize_constraint_system_via_term_graph_with_labelling(
        sys,
        crate::canon_term_graph::TermGraphOptions::default(),
    )
    .map(|(canon, labelling, _)| (canon, labelling))
}

/// A [`ProofMethod`] renamed into its system's canonical form (see
/// [`canonicalize_proof_method`]): two canonically equal systems must offer
/// the same MULTISET of these. Only a multiset -- `candidate_methods` ranks
/// goals using `depth`, `GoalStatus::looping` and `GoalStatus::nr`, none of
/// which are part of the canonical form, so the order may differ.
#[derive(Debug, Clone, PartialEq)]
pub enum CanonicalProofMethod {
    Simplify,
    Induction,
    Sorry(Option<String>),
    /// Result and contradiction KIND only: the payload (`NodeId`s, a term)
    /// belongs to whichever contradiction `is_finished` happens to find
    /// first, which depends on raw iteration order.
    Finished {
        result: &'static str,
        contradiction: Option<&'static str>,
    },
    /// The goal as the system's canonical form writes it, without its solved
    /// flag.
    Solve(STerm),
    /// `solve` of a case split, identified by the disjunction of the
    /// equation store it names, as the system's canonical form writes it:
    /// the form drops split ids.
    SolveSplit(STerm),
}

/// Renames `method` (offered for `sys`) into `sys`'s canonical form through
/// `labelling`, `sys`'s canonical labelling from
/// [`canonicalize_constraint_system_with_labelling`]. Panics on a variable
/// `labelling` doesn't name, on a split goal naming no `EqDisj` in
/// `sys.eq_store.conj`, and on the display-only `Invalidated`/`RawSolve`,
/// which `candidate_methods` never produces.
pub fn canonicalize_proof_method(
    method: &ProofMethod,
    sys: &System,
    labelling: &CanonLabelling,
) -> CanonicalProofMethod {
    match method {
        ProofMethod::Simplify => CanonicalProofMethod::Simplify,
        ProofMethod::Induction => CanonicalProofMethod::Induction,
        ProofMethod::Sorry(reason) => CanonicalProofMethod::Sorry(reason.clone()),
        ProofMethod::Finished(result) => {
            let (result, contradiction) = match result {
                FinishedResult::Solved => ("solved", None),
                FinishedResult::Contradictory(c) => {
                    ("contradictory", c.as_ref().map(contradiction_kind))
                }
                FinishedResult::Unfinishable => ("unfinishable", None),
            };
            CanonicalProofMethod::Finished {
                result,
                contradiction,
            }
        }
        ProofMethod::SolveGoal(Goal::Split(id)) => {
            let index = sys
                .eq_store
                .conj
                .iter()
                .position(|d| d.split_id == *id)
                .unwrap_or_else(|| panic!("split goal {id:?} names no EqDisj in eq_store.conj"));
            CanonicalProofMethod::SolveSplit(canonical_eq_disj(
                &sys.eq_store.conj[index],
                index,
                labelling,
            ))
        }
        ProofMethod::SolveGoal(goal) => CanonicalProofMethod::Solve(
            canonical_goal(goal, labelling)
                .expect("only Goal::Split has no goal term, handled above"),
        ),
        ProofMethod::Invalidated | ProofMethod::RawSolve(_) => {
            panic!("{method:?} is display-only and never a candidate proof method")
        }
    }
}

fn contradiction_kind(c: &Contradiction) -> &'static str {
    match c {
        Contradiction::Cyclic => "cyclic",
        Contradiction::SubtermCyclic => "subterm_cyclic",
        Contradiction::NonNormalTerms => "non_normal_terms",
        Contradiction::ForbiddenExp => "forbidden_exp",
        Contradiction::ForbiddenBP => "forbidden_bp",
        Contradiction::ForbiddenKD => "forbidden_kd",
        Contradiction::ImpossibleChain => "impossible_chain",
        Contradiction::ForbiddenChain => "forbidden_chain",
        Contradiction::ForbiddenACConstrChain => "forbidden_ac_constr_chain",
        Contradiction::NonInjectiveFactInstance(..) => "non_injective_fact_instance",
        Contradiction::IncompatibleEqs => "incompatible_eqs",
        Contradiction::FormulasFalse => "formulas_false",
        Contradiction::SuperfluousLearn(..) => "superfluous_learn",
        Contradiction::NodeAfterLast(..) => "node_after_last",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tamarin_term::lterm::{LSort, LVar};
    use tamarin_term::vterm::var_term;

    use crate::bliss_proc::bliss_available;
    use crate::canon_term_graph::canonicalize_constraint_system_via_term_graph_with_labelling;
    use crate::constraint::system::GoalStatus;
    use crate::fact::{in_fact, proto_fact, Multiplicity};
    use crate::guarded::formula_to_guarded;
    use crate::rule::{PremIdx, Rule};
    use crate::tools::equation_store::{EqDisj, LNSubstVFresh, SplitId};
    use tamarin_parser::parser::parse_formula_str;

    fn v(name: &str, sort: LSort) -> LNTerm {
        var_term(LVar::new(name, sort, 0))
    }

    fn node(idx: u64) -> crate::constraint::constraints::NodeId {
        LVar::new("i", LSort::Node, idx)
    }

    /// Parses a surface formula string straight to its guarded form.
    fn g(s: &str) -> Guarded {
        let f = parse_formula_str(s).unwrap_or_else(|e| panic!("parse {s:?}: {e}"));
        formula_to_guarded(&f).unwrap_or_else(|e| panic!("formula_to_guarded {s:?}: {e}"))
    }

    /// A ground 0-ary NoEq function symbol term: never renamed, ordered by
    /// name.
    fn zero_ary_fun_term(name: &'static str) -> LNTerm {
        let sym = NoEqSym::new(
            name.as_bytes().to_vec(),
            0,
            Privacy::Public,
            Constructability::Constructor,
        );
        LNTerm::App(FunSym::NoEq(sym), Arc::from([]))
    }

    fn rule_ac_inst(
        name: &'static str,
        premises: Vec<LNFact>,
        actions: Vec<LNFact>,
        conclusions: Vec<LNFact>,
    ) -> RuleACInst {
        use crate::rule::{ProtoRuleACInstInfo, RuleAttributes};
        Rule::new(
            RuleInfo::Proto(ProtoRuleACInstInfo {
                name: ProtoRuleName::Stand(name),
                attributes: RuleAttributes::empty(),
                loop_breakers: Vec::new(),
            }),
            premises,
            conclusions,
            actions,
        )
    }

    /// A labelling naming `vars` in order (`mv0`, `tv0`, … per sort).
    fn labelling_covering(vars: &[LVar]) -> CanonLabelling {
        let mut labelling = CanonLabelling::empty();
        for x in vars {
            labelling.bind_next(*x);
        }
        labelling
    }

    // -- Formulas in AC normal form (`ac_normalize_guarded`) ---------------

    #[test]
    fn swapped_eq_operands_normalize_equally() {
        assert_eq!(ac_normalize_guarded(&g("x = y")), ac_normalize_guarded(&g("y = x")));
    }

    #[test]
    fn reordered_conjunction_operands_normalize_equally() {
        assert_eq!(
            ac_normalize_guarded(&g("P(x) @ #i & Q(y) @ #j")),
            ac_normalize_guarded(&g("Q(y) @ #j & P(x) @ #i"))
        );
    }

    /// `(A & B) & C` and `C & (B & A)`: flattened, then sorted.
    #[test]
    fn nested_conjunction_flattens_and_sorts() {
        assert_eq!(
            ac_normalize_guarded(&g("(P(x) @ #i & Q(y) @ #j) & R(z) @ #k")),
            ac_normalize_guarded(&g("R(z) @ #k & (Q(y) @ #j & P(x) @ #i)"))
        );
    }

    /// Binder display names are cosmetic: only a bound variable's de Bruijn
    /// position counts, also for several binders in one block and for
    /// nested quantifiers.
    #[test]
    fn bound_variable_display_names_do_not_count() {
        for (a, b) in [
            ("Ex x #i. G(x)@i", "Ex a #b. G(a)@b"),
            ("All x y #i. P(x,y) @ #i ==> F", "All a b #i. P(a,b) @ #i ==> F"),
            (
                "All k #i. Setup(k) @ #i ==> Ex j #t. Foo(j) @ #t",
                "All p #q. Setup(p) @ #q ==> Ex w #z. Foo(w) @ #z",
            ),
        ] {
            let (fa, fb) = (ac_normalize_guarded(&g(a)), ac_normalize_guarded(&g(b)));
            assert_eq!(fa, fb, "{a} vs {b}");
            assert_eq!(fingerprint_guarded(&fa), fingerprint_guarded(&fb));
        }
    }

    #[test]
    fn different_fact_names_stay_different() {
        assert_ne!(
            ac_normalize_guarded(&g("P(x) @ #i")),
            ac_normalize_guarded(&g("Q(x) @ #i"))
        );
    }

    /// Names are constants: they stay as written, `=` over two names is
    /// still ordered, and formulas differing only in a name (here a message
    /// tag, also under a quantifier) stay different.
    #[test]
    fn names_are_kept_fixed() {
        let got = ac_normalize_guarded(&g("~'m' = ~'n'"));
        assert_eq!(got, ac_normalize_guarded(&g("~'n' = ~'m'")));
        match &got {
            Guarded::Atom(GAtom::Eq(GTerm::FreshLit(a), GTerm::FreshLit(b))) => {
                let mut names = [a.as_str(), b.as_str()];
                names.sort();
                assert_eq!(names, ["m", "n"]);
            }
            other => panic!("expected Eq(FreshLit, FreshLit), got {other:?}"),
        }
        let (one, two) = (
            ac_normalize_guarded(&g("P(<'1', x>) @ #i")),
            ac_normalize_guarded(&g("P(<'2', x>) @ #i")),
        );
        assert_ne!(one, two);
        assert_ne!(fingerprint_guarded(&one), fingerprint_guarded(&two));
        assert_ne!(
            ac_normalize_guarded(&g("not (Ex #j. Reveal('c') @ #j)")),
            ac_normalize_guarded(&g("not (Ex #j. Reveal('d') @ #j)"))
        );
    }

    /// AC operands inside an atom's terms are sorted (`xor` here), across
    /// re-association too; `=`'s sides are ordered on their own.
    #[test]
    fn xor_operands_are_sorted_across_reassociation() {
        assert_eq!(
            ac_normalize_guarded(&g("a XOR b = c")),
            ac_normalize_guarded(&g("b XOR a = c"))
        );
        assert_eq!(
            ac_normalize_guarded(&g("a XOR b XOR c = z")),
            ac_normalize_guarded(&g("c XOR (a XOR b) = z"))
        );
    }

    /// The bilinear pairing `em` is commutative (C), not associative.
    #[test]
    fn em_arguments_are_sorted() {
        assert_eq!(
            ac_normalize_guarded(&g("P(em(a,b)) @ #i")),
            ac_normalize_guarded(&g("P(em(b,a)) @ #i"))
        );
    }

    #[test]
    fn ac_normalize_guarded_is_idempotent() {
        let once = ac_normalize_guarded(&g("z = y & Ex x #i. G(x)@i"));
        let twice = ac_normalize_guarded(&once);
        assert_eq!(once, twice);
        assert_eq!(fingerprint_guarded(&once), fingerprint_guarded(&twice));
    }

    // -- Whole systems -----------------------------------------------------

    /// An action goal's `solved` flag decides whether `candidate_methods`
    /// offers it, so two systems differing only in it are different
    /// proof-search states.
    #[test]
    fn systems_differing_only_in_an_action_goals_solved_status_are_not_alpha_eq() {
        if !bliss_available() {
            return;
        }
        let with_action_goal = |solved| {
            let mut sys = System::empty();
            sys.content_mut().goals = Arc::new(vec![(
                Goal::Action(node(0), crate::fact::ku_fact(zero_ary_fun_term("m"))),
                GoalStatus {
                    solved,
                    ..Default::default()
                },
            )]);
            canonicalize_constraint_system(&sys)
                .unwrap_or_else(|e| panic!("canonicalize (solved={solved}): {e:?}"))
        };
        let (open, solved) = (with_action_goal(false), with_action_goal(true));
        assert_eq!(open.section("nodes"), solved.section("nodes"));
        assert_ne!(open, solved);
    }

    /// `!KU(<'1', x>)` and `!KU(<'2', x>)` unify with different rule
    /// conclusions: names are constants, while renaming `x` still merges.
    #[test]
    fn systems_differing_only_in_a_message_tag_are_not_alpha_eq() {
        if !bliss_available() {
            return;
        }
        use crate::canon_fingerprint::fingerprint_constraint_system;
        use tamarin_term::builtin::pair;
        use tamarin_term::lterm::pub_term;

        let with_ku_goal = |tag: &str, var: &str| {
            let mut sys = System::empty();
            sys.content_mut().goals = Arc::new(vec![(
                Goal::Action(
                    node(0),
                    crate::fact::ku_fact(pair(pub_term(tag), v(var, LSort::Msg))),
                ),
                GoalStatus::default(),
            )]);
            canonicalize_constraint_system(&sys)
                .unwrap_or_else(|e| panic!("canonicalize ('{tag}', {var}): {e:?}"))
        };
        let (one_x, two_x, one_y) = (
            with_ku_goal("1", "x"),
            with_ku_goal("2", "x"),
            with_ku_goal("1", "y"),
        );
        assert_ne!(one_x, two_x);
        assert_ne!(
            fingerprint_constraint_system(&one_x),
            fingerprint_constraint_system(&two_x)
        );
        assert_eq!(one_x, one_y);
    }

    /// Two rules with the same facts but different names (SAPIC's
    /// `channels1`): which one a node instantiates counts.
    #[test]
    fn systems_differing_only_in_a_rule_name_are_not_alpha_eq() {
        if !bliss_available() {
            return;
        }
        let with_rule = |name: &'static str, var: &str| {
            let mut sys = System::empty();
            let fact = |f: &str| proto_fact(Multiplicity::Linear, f, vec![v(var, LSort::Msg)]);
            sys.add_node(
                node(0),
                rule_ac_inst(name, vec![in_fact(v(var, LSort::Msg))], vec![], vec![fact("M")]),
            );
            canonicalize_constraint_system(&sys)
                .unwrap_or_else(|e| panic!("canonicalize ({name}, {var}): {e:?}"))
        };
        assert_ne!(with_rule("A1", "x"), with_rule("A2", "x"));
        assert_eq!(with_rule("A1", "x"), with_rule("A1", "y"));
    }

    /// A pending goal is part of the system: the KU sub-goals the solver
    /// inserts directly into `goals` appear nowhere else.
    #[test]
    fn systems_differing_only_in_a_pending_goal_are_not_alpha_eq() {
        let mut with_goal = System::empty();
        with_goal.content_mut().goals = Arc::new(vec![(
            Goal::Subterm((zero_ary_fun_term("small"), zero_ary_fun_term("big"))),
            GoalStatus::default(),
        )]);
        assert_ne!(
            canonicalize_constraint_system(&with_goal).expect("canonicalize"),
            canonicalize_constraint_system(&System::empty()).expect("canonicalize")
        );
    }

    /// A system without variables (the root of every proof, before any rule
    /// instance) is its own canonical form and needs no bliss, which
    /// refuses an empty graph. Not gated on `bliss_available()`.
    #[test]
    fn a_system_without_variables_canonicalizes_without_bliss() {
        use crate::guarded::{gfalse, gtrue};
        let with = |f: Guarded| {
            let mut sys = System::empty();
            sys.formulas_mut().push(Arc::new(f));
            let (canon, _, stats) = canonicalize_constraint_system_via_term_graph_with_labelling(
                &sys,
                crate::canon_term_graph::TermGraphOptions::default(),
            )
            .expect("canonicalize");
            assert!(!stats.bliss_ran);
            canon
        };
        let canon = with(gtrue());
        assert!(matches!(canon.section("formulas"), STerm::App(_, fs) if fs.len() == 1));
        assert_ne!(canon, with(gfalse()));
    }

    // -- canonicalize_proof_method -----------------------------------------

    /// The same goal under two raw namings, each renamed by its own system's
    /// renaming, gives the same canonical method.
    #[test]
    fn alpha_renamed_solve_goals_canonicalize_equally() {
        let solve = |nid: LVar, var: &str| {
            let x = LVar::new(var, LSort::Msg, 0);
            let method = ProofMethod::SolveGoal(Goal::Premise(
                (nid, PremIdx(0)),
                in_fact(v(var, LSort::Msg)),
            ));
            canonicalize_proof_method(&method, &System::empty(), &labelling_covering(&[nid, x]))
        };
        assert_eq!(solve(node(0), "x"), solve(node(5), "y"));
        assert_ne!(
            solve(node(0), "x"),
            canonicalize_proof_method(&ProofMethod::Simplify, &System::empty(), &CanonLabelling::empty())
        );
    }

    /// A split goal is identified by its disjunction as the form writes it,
    /// not by its raw split id (which the form drops) nor by the raw names
    /// of the alternative's local variables.
    #[test]
    fn split_methods_canonicalize_by_their_disjunctions_content() {
        if !bliss_available() {
            return;
        }
        let x = LVar::new("x", LSort::Msg, 7);
        let solve_split = |id: i64, witness_idx: u64| {
            let mut sys = System::empty();
            sys.content_mut().goals = Arc::new(vec![(
                Goal::Action(node(0), crate::fact::ku_fact(var_term(x))),
                GoalStatus::default(),
            )]);
            sys.eq_store_mut().conj.push(EqDisj {
                split_id: SplitId(id),
                substs: vec![LNSubstVFresh::from_list(vec![(
                    x,
                    tamarin_term::builtin::pair(
                        var_term(LVar::new("w", LSort::Msg, witness_idx)),
                        var_term(LVar::new("w", LSort::Msg, witness_idx + 1)),
                    ),
                )])],
            });
            let (_, labelling) = canonicalize_constraint_system_with_labelling(&sys)
                .expect("canonicalize");
            canonicalize_proof_method(
                &ProofMethod::SolveGoal(Goal::Split(SplitId(id))),
                &sys,
                &labelling,
            )
        };
        let a = solve_split(0, 100);
        assert!(matches!(a, CanonicalProofMethod::SolveSplit(_)));
        assert_eq!(a, solve_split(3, 500));
    }

    #[test]
    #[should_panic(expected = "names no EqDisj")]
    fn a_split_method_naming_no_disjunction_panics() {
        let _ = canonicalize_proof_method(
            &ProofMethod::SolveGoal(Goal::Split(SplitId(9))),
            &System::empty(),
            &CanonLabelling::empty(),
        );
    }

    /// Which contradiction instance `is_finished` reports first depends on
    /// raw iteration order, so only its kind may count.
    #[test]
    fn finished_methods_ignore_the_contradiction_payload() {
        let finished = |c: Contradiction| {
            canonicalize_proof_method(
                &ProofMethod::Finished(FinishedResult::Contradictory(Some(c))),
                &System::empty(),
                &CanonLabelling::empty(),
            )
        };
        assert_eq!(
            finished(Contradiction::NodeAfterLast(node(0), node(1))),
            finished(Contradiction::NodeAfterLast(node(7), node(3)))
        );
        assert_ne!(
            finished(Contradiction::NodeAfterLast(node(0), node(1))),
            finished(Contradiction::Cyclic)
        );
    }
}
