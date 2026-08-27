//! Lifting `CAN_alphaeqac` (canonization of terms for alpha-equivalence
//! modulo AC, `tamarin_term::alpha_eq_ac`) to facts and rules, per
//! `work.tex`'s "Canonization of Facts and Rule Instances": facts and rule
//! instances carry no equational theory of their own, so canonizing them
//! reduces to canonizing a single term that faithfully encodes their
//! structure.
//!
//! Unlike `work.tex`'s formula, which uses one fixed-arity symbol `f` for
//! every fact (`f(t_1, ..., t_n)`) and rule (`f(i, p_1, ..., p_k, a_1, ...,
//! a_l, c_1, ..., c_m)`), a fact's/rule's arity varies per instance here, so
//! we build the lifted term with [`FunSym::List`] — a free n-ary combinator
//! with no equational theory of its own — instead of committing to one
//! symbol per shape.
//!
//! - A fact `F(t_1, ..., t_n)` becomes `List(marker(F), t_1, ..., t_n)`,
//!   where `marker(F)` is a distinguishing 0-ary constructor symbol built
//!   from `F`'s tag (see [`fact_tag_marker`]). It is a function symbol, not
//!   a literal, so `CAN_alphaeqac` — which only ever renames literals —
//!   leaves it untouched: two facts with different tags can therefore never
//!   canonize equal, no matter their arguments.
//! - A rule is lifted the same way, one level up: its premises, actions,
//!   and conclusions each become their own `List` of lifted fact terms, and
//!   the three groups are combined into one outer `List` (`rule_to_term`).
//!   Grouping — rather than one flat concatenation of the `k + l + m`
//!   facts, as `work.tex`'s formula reads literally — keeps a rule's
//!   premise/action/conclusion boundaries structurally explicit: nothing
//!   else here plays the role `work.tex`'s `i` (an external timepoint, not
//!   part of the rule itself) does, so without grouping, e.g. a
//!   1-premise/1-action/1-conclusion rule could canonize equal to a
//!   0-premise/2-action/1-conclusion rule whenever their facts happen to
//!   line up.
//!
//! `new_vars` is intentionally NOT part of the lifted term. `Rule<I>::
//! new_vars` is a deterministic function of the premises, actions, and
//! conclusions alone — computed once as `(conclusion vars ∪ action vars) −
//! premise vars` (Haskell `newVariables`) and always kept in lockstep with
//! the rest of the rule by the same machinery that already walks
//! premises/actions/conclusions (`HasFrees for Rule<I>` and
//! `apply_subst_rule` in `rule.rs` fold/substitute over all four fields
//! identically); it never diverges from what the other three fields imply.
//! Two rules with alpha-eq-ac premises/actions/conclusions therefore always
//! have alpha-eq-ac `new_vars` too, so including `new_vars` in the lifted
//! term could only ever be redundant, never distinguishing. This mirrors an
//! existing precedent in this codebase: `write_rule_to_key_excl_new_vars`
//! (`constraint/solver/sources.rs`) already excludes `new_vars` when
//! building the canonical key used to detect redundant/duplicate
//! constraint systems, with the comment "Crucial: rule.new_vars EXCLUDED
//! per `compareRulesUpToNewVars`."
//!
//! `info: I` is intentionally NOT part of the lifted term either, mirroring
//! `Rule<I>`'s own `HasFrees` impl (`rule.rs`): the generic bound is
//! `Clone`, not a to-term conversion, so this does not by itself distinguish
//! rules that differ only in name/info — callers who need that (e.g. two
//! differently-named rules should never be treated as $\alphaeqac$) must
//! compare `info` separately.

use std::sync::Arc;

use tamarin_term::alpha_eq_ac::canonicalize_alpha_eq_ac;
use tamarin_term::function_symbols::{Constructability, FunSym, NoEqSym, Privacy};
use tamarin_term::lterm::LNTerm;
use tamarin_term::subst::Subst;
use tamarin_term::term::f_app_list;

use crate::fact::{FactTag, LNFact};
use crate::rule::Rule;

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
/// t_n)` (see the module docs).
pub fn fact_to_term(fact: &LNFact) -> LNTerm {
    let mut args = Vec::with_capacity(fact.terms.len() + 1);
    args.push(fact_tag_marker(&fact.tag));
    args.extend(fact.terms.iter().cloned());
    f_app_list(args)
}

/// Canonizes a fact w.r.t. $\alphaeqac$: two facts are $\alphaeqac$ iff
/// their [`fact_to_term`] lifts canonize syntactically equal.
pub fn canonicalize_fact(fact: &LNFact) -> LNTerm {
    canonicalize_alpha_eq_ac(&fact_to_term(fact), Subst::empty())
}

/// Lifts a group of facts (a rule's premises, actions, or conclusions) to
/// `List(fact_to_term(f_1), ..., fact_to_term(f_n))`.
fn facts_to_term(facts: &[LNFact]) -> LNTerm {
    f_app_list(facts.iter().map(fact_to_term).collect())
}

/// Lifts a rule to a term nesting its premises, actions, and conclusions
/// (see the module docs for why `info` and `new_vars` are excluded, and why
/// the groups are nested rather than flattened).
pub fn rule_to_term<I>(rule: &Rule<I>) -> LNTerm {
    f_app_list(vec![
        facts_to_term(&rule.premises),
        facts_to_term(&rule.actions),
        facts_to_term(&rule.conclusions),
    ])
}

/// Canonizes a rule w.r.t. $\alphaeqac$ (see [`rule_to_term`]).
pub fn canonicalize_rule<I>(rule: &Rule<I>) -> LNTerm {
    canonicalize_alpha_eq_ac(&rule_to_term(rule), Subst::empty())
}

// =============================================================================
// Guarded formulas
// =============================================================================
//
// Per `work.tex`'s "Canonization of Formula Constraints": a guarded
// formula's free variables and name constants are always bound/introduced
// elsewhere in the constraint system (by an action formula, a rule
// constraint, or a rule/fact argument — see the paragraph building up to
// `ex:canon_guarded`), so by the time formulas are canonized a canonical
// labelling `theta` for them already exists (computed while canonizing the
// system's rule instances/action formulas, which this module doesn't yet do
// — TODO.md's graph-canonization item). `theta` is assumed EXHAUSTIVE: every
// free variable and name constant `g` mentions must have an entry, or
// canonization panics (see [`lookup_theta`]) — a miss means `theta` was
// built incorrectly, not a case to paper over with a silent fallback. Given
// that labelling, canonizing a `Guarded` is two steps:
//
// 1. Substitute every free variable AND name constant via `theta`
//    ([`subst_via_theta_guarded`]) — matching `tamarin_term::alpha_eq_ac`'s
//    own literal model, where a name constant (`PubLit`/`FreshLit`/`NatLit`
//    here, `Lit::Con(Name{..})` there) is renamed exactly like a variable.
//    Bound variables need no substitution: a `Guarded`'s bound occurrences
//    are already De-Bruijn indices (position-determined), which is already
//    a canonical representation — the only non-canonical thing about them is
//    each binder's cosmetic display NAME (`GBinding::name`), which carries
//    no semantic content and is erased (see [`ac_normalize_guarded`]).
// 2. Bring $\land$/$\lor$ (`Guarded::Conj`/`Disj`) and $=$ (`GAtom::Eq`) into
//    AC normal form ([`ac_normalize_guarded`]), reusing this crate's
//    existing `guarded.rs` machinery: `gconj`/`gdisj` already flatten nested
//    conjunctions/disjunctions and drop duplicates (mirroring `CAN_AC`'s
//    flattening of terms); this file adds the sorting half — by the
//    already-HS-faithful [`crate::guarded::cmp_guarded`] /
//    [`crate::guarded::cmp_atom`] total orders, which are exactly the "order
//    on formulas" / "atoms according to string representation" `work.tex`
//    only gestures at ("we compare atoms according to string representation
//    and fix some order on the function symbols" — `\todo{Better
//    explanation?}"). `work.tex`'s own worked example additionally assumes
//    an illustrative `'∃' < '='` operator order that is NOT what
//    `cmp_guarded` (Atom < Disj < Conj < GGuarded) gives; this module uses
//    the codebase's real, already-tested order rather than inventing a
//    second one just to match the paper's illustration — see
//    [`tests::canonicalize_guarded_matches_work_tex_example`] for the
//    concrete difference this makes.

use crate::guarded::{
    self, cmp_atom, cmp_guarded, cmp_term, ga, gall, gex, map_guarded_atoms, BVar, GAtom, GBinding,
    GFact, GTerm, Guarded, Quant,
};
use tamarin_parser::ast as p;
use tamarin_term::lterm::{LSort, Name, NameId, NameTag};
use tamarin_term::vterm::Lit;
use tamarin_utils::fingerprint::{Fingerprint, FingerprintHasher};

/// Same choice `tamarin_term::alpha_eq_ac` makes internally: a literal is
/// either a name constant or a variable.
type LNLit = Lit<Name, tamarin_term::lterm::LVar>;

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

/// `LSort` to `p::SortHint` (the inverse of [`sort_hint_to_lsort`], modulo
/// `Untagged`/`Suffix`, which `LSort` has no counterpart for).
fn lsort_to_sort_hint(s: LSort) -> p::SortHint {
    match s {
        LSort::Msg => p::SortHint::Msg,
        LSort::Pub => p::SortHint::Pub,
        LSort::Fresh => p::SortHint::Fresh,
        LSort::Node => p::SortHint::Node,
        LSort::Nat => p::SortHint::Nat,
    }
}

/// A parser-AST variable spec as the `LVar` it denotes.
fn varspec_to_lvar(v: &p::VarSpec) -> tamarin_term::lterm::LVar {
    tamarin_term::lterm::LVar::new(v.name.as_str(), sort_hint_to_lsort(v.sort), v.idx)
}

/// An `LVar` as the parser-AST variable spec that denotes it (no SAPIC type
/// annotation — canonical variables never carry one).
fn lvar_to_varspec(v: &tamarin_term::lterm::LVar) -> p::VarSpec {
    p::VarSpec {
        name: v.name.to_string(),
        idx: v.idx,
        sort: lsort_to_sort_hint(v.sort),
        typ: None,
    }
}

/// The name constant a `PubLit`/`FreshLit`/`NatLit`'s source string denotes,
/// for looking it up in `theta`.
fn name_lit(tag: NameTag, s: &str) -> Name {
    Name {
        tag,
        id: NameId::new(s),
    }
}

/// Looks up a literal in `theta`, panicking if it has no entry. A guarded
/// formula's free variables and name constants are ALWAYS assumed to be
/// already covered by an exhaustively-computed canonical labelling (per
/// `work.tex`'s argument that every free variable in a constraint system's
/// formula is bound by some action formula or rule constraint elsewhere in
/// the same system): a miss here means `theta` was built incorrectly
/// (missing an entry) or this formula wasn't actually closed the way
/// `work.tex` assumes — either way a caller bug, not a case to paper over
/// with a silent fallback.
fn lookup_theta<'t>(theta: &'t std::collections::BTreeMap<LNLit, LNLit>, key: &LNLit) -> &'t LNLit {
    theta.get(key).unwrap_or_else(|| {
        panic!(
            "canonicalize_guarded: {key:?} has no entry in theta — every free \
             variable/name constant of a guarded formula must already be covered \
             by the canonical labelling (work.tex's guardedness argument), so a \
             miss here is a caller bug, not a case to fall back on silently"
        )
    })
}

/// Substitutes every free variable and name constant of `g` via `theta`,
/// leaving Bound variables (already canonical by De-Bruijn position)
/// untouched. Panics if some free variable or name constant has no entry in
/// `theta` — see [`lookup_theta`].
///
/// `theta` is the same `LNLit -> LNLit` map [`tamarin_term::alpha_eq_ac`]
/// itself produces: a sort-respecting, bijective, CATEGORY-respecting
/// (`Lit::Var` never maps to `Lit::Con` or vice versa) canonical labelling.
/// A `Guarded`'s `GTerm::Var(BVar::Free(_))` leaves look themselves up as
/// `Lit::Var`; its `PubLit`/`FreshLit`/`NatLit(String)` leaves — the
/// constants `alpha_eq_ac` calls `Lit::Con(Name{..})` — look themselves up
/// as `Lit::Con` under the matching `NameTag`.
pub fn subst_via_theta_guarded(
    g: &Guarded,
    theta: &std::collections::BTreeMap<LNLit, LNLit>,
) -> Guarded {
    map_guarded_atoms(g, &mut |_depth, a| subst_atom_via_theta(a, theta))
}

fn subst_atom_via_theta(a: &GAtom, theta: &std::collections::BTreeMap<LNLit, LNLit>) -> GAtom {
    let t = |x: &GTerm| subst_term_via_theta(x, theta);
    let f = |x: &GFact| subst_fact_via_theta(x, theta);
    match a {
        GAtom::Eq(x, y) => GAtom::Eq(t(x), t(y)),
        GAtom::Less(x, y) => GAtom::Less(t(x), t(y)),
        GAtom::LessMset(x, y) => GAtom::LessMset(t(x), t(y)),
        GAtom::Subterm(x, y) => GAtom::Subterm(t(x), t(y)),
        GAtom::Action(fact, time) => GAtom::Action(f(fact), t(time)),
        GAtom::Last(x) => GAtom::Last(t(x)),
        GAtom::Pred(fact) => GAtom::Pred(f(fact)),
    }
}

fn subst_fact_via_theta(fact: &GFact, theta: &std::collections::BTreeMap<LNLit, LNLit>) -> GFact {
    GFact {
        persistent: fact.persistent,
        name: fact.name.clone(),
        args: fact
            .args
            .iter()
            .map(|a| subst_term_via_theta(a, theta))
            .collect(),
        annotations: fact.annotations.clone(),
    }
}

fn subst_term_via_theta(t: &GTerm, theta: &std::collections::BTreeMap<LNLit, LNLit>) -> GTerm {
    let rec = |x: &GTerm| subst_term_via_theta(x, theta);
    match t {
        GTerm::Var(BVar::Free(v)) => {
            let key = Lit::Var(varspec_to_lvar(v));
            match lookup_theta(theta, &key) {
                Lit::Var(canon) => GTerm::Var(BVar::Free(lvar_to_varspec(canon))),
                Lit::Con(_) => panic!(
                    "canonicalize_guarded: variable {v:?} mapped to a name constant \
                     in theta — a sort-respecting substitution never does this"
                ),
            }
        }
        GTerm::Var(BVar::Bound(_)) => t.clone(),
        GTerm::PubLit(s) => match lookup_theta(theta, &Lit::Con(name_lit(NameTag::Pub, s))) {
            Lit::Con(canon) => GTerm::PubLit(canon.id.as_str().to_string()),
            Lit::Var(_) => panic!(
                "canonicalize_guarded: pub name {s:?} mapped to a variable in theta \
                 — a sort-respecting substitution never does this"
            ),
        },
        GTerm::FreshLit(s) => match lookup_theta(theta, &Lit::Con(name_lit(NameTag::Fresh, s))) {
            Lit::Con(canon) => GTerm::FreshLit(canon.id.as_str().to_string()),
            Lit::Var(_) => panic!(
                "canonicalize_guarded: fresh name {s:?} mapped to a variable in theta \
                 — a sort-respecting substitution never does this"
            ),
        },
        GTerm::NatLit(s) => match lookup_theta(theta, &Lit::Con(name_lit(NameTag::Nat, s))) {
            Lit::Con(canon) => GTerm::NatLit(canon.id.as_str().to_string()),
            Lit::Var(_) => panic!(
                "canonicalize_guarded: nat name {s:?} mapped to a variable in theta \
                 — a sort-respecting substitution never does this"
            ),
        },
        // Built-in 0-ary constant TERMS (`one`/`tone`/`DH_neutral`), not name
        // literals: `alpha_eq_ac`'s own literal model has no entry for these
        // either (they're NoEq function applications of arity 0, per
        // `term.rs`'s `one_sym`/`nat_one_sym`/`dh_neutral_sym`), so there is
        // nothing to look up in `theta`.
        GTerm::Number(_) | GTerm::NumberOne | GTerm::NatOne | GTerm::DhNeutral => t.clone(),
        GTerm::App(n, args) => GTerm::App(n.clone(), args.iter().map(rec).collect()),
        GTerm::AlgApp(n, x, y) => GTerm::AlgApp(n.clone(), ga(rec(x)), ga(rec(y))),
        GTerm::Pair(items) => GTerm::Pair(items.iter().map(rec).collect()),
        GTerm::Diff(x, y) => GTerm::Diff(ga(rec(x)), ga(rec(y))),
        GTerm::BinOp(op, x, y) => GTerm::BinOp(*op, ga(rec(x)), ga(rec(y))),
        GTerm::PatMatch(x) => GTerm::PatMatch(ga(rec(x))),
    }
}

/// Canonically orders an atom's operands where doing so is sound: `=` is
/// symmetric (`s = t` and `t = s` are the same atom), so its two sides are
/// reordered smaller-first by [`cmp_term`]. Every other `GAtom` variant is
/// directional (`Less`/`Subterm`, or simply not binary) and is returned
/// unchanged.
fn normalize_atom(a: &GAtom) -> GAtom {
    let t = ac_normalize_term;
    let f = |fact: &GFact| GFact {
        persistent: fact.persistent,
        name: fact.name.clone(),
        args: fact.args.iter().map(ac_normalize_term).collect(),
        annotations: fact.annotations.clone(),
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

/// Canonizes a guarded formula w.r.t. $\alphaeqac$: substitutes its free
/// variables via the already-canonical `theta`, then brings the result into
/// `CAN_AC` normal form (see the module docs, [`subst_via_theta_guarded`],
/// and [`ac_normalize_guarded`]).
pub fn canonicalize_guarded(
    g: &Guarded,
    theta: &std::collections::BTreeMap<LNLit, LNLit>,
) -> Guarded {
    ac_normalize_guarded(&subst_via_theta_guarded(g, theta))
}

/// Fingerprints an ALREADY-CANONICAL guarded formula — the output of
/// [`canonicalize_guarded`]. Shares its digest type with
/// [`tamarin_term::fingerprint::fingerprint_term`] (which also covers
/// [`canonicalize_fact`]/[`canonicalize_rule`]'s outputs, both plain
/// canonical `LNTerm`s): one uniform fingerprint type across every
/// canonized object kind this crate produces.
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

fn fingerprint_atom(a: &GAtom) -> Fingerprint {
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

fn fingerprint_gterm(t: &GTerm) -> Fingerprint {
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

#[cfg(test)]
mod tests {
    use super::*;
    use tamarin_term::alpha_eq_ac::fingerprint_term;
    use tamarin_term::lterm::{LSort, LVar};
    use tamarin_term::vterm::var_term;

    use crate::fact::{fresh_fact, in_fact, out_fact, proto_fact, Multiplicity};

    fn v(name: &str, sort: LSort) -> LNTerm {
        var_term(LVar::new(name, sort, 0))
    }

    fn rule(premises: Vec<LNFact>, actions: Vec<LNFact>, conclusions: Vec<LNFact>) -> Rule<()> {
        Rule::new((), premises, conclusions, actions)
    }

    // -- Facts ----------------------------------------------------------

    /// Two `Proto` facts of the same tag, differing only by a renamed
    /// variable, canonize equal — `canonicalize_fact` correctly delegates to
    /// `CAN_alphaeqac` on the lifted term.
    #[test]
    fn same_tag_facts_renamed_vars_are_alpha_eq() {
        let f1 = proto_fact(Multiplicity::Linear, "Login", vec![v("x", LSort::Msg)]);
        let f2 = proto_fact(Multiplicity::Linear, "Login", vec![v("y", LSort::Msg)]);
        assert_eq!(canonicalize_fact(&f1), canonicalize_fact(&f2));
        assert_eq!(
            fingerprint_term(&canonicalize_fact(&f1)),
            fingerprint_term(&canonicalize_fact(&f2))
        );
    }

    /// Two structurally-identical-looking facts with DIFFERENT tags must
    /// never canonize equal — this is what the marker (a function symbol,
    /// never renamed) is for.
    #[test]
    fn different_tag_facts_are_not_alpha_eq() {
        let f1 = proto_fact(Multiplicity::Linear, "Login", vec![v("x", LSort::Msg)]);
        let f2 = proto_fact(Multiplicity::Linear, "Logout", vec![v("x", LSort::Msg)]);
        assert_ne!(canonicalize_fact(&f1), canonicalize_fact(&f2));
    }

    /// Persistent vs. linear facts of the same name/arity are different
    /// tags and so must not canonize equal either.
    #[test]
    fn same_name_different_multiplicity_facts_are_not_alpha_eq() {
        let f1 = proto_fact(Multiplicity::Linear, "St", vec![v("x", LSort::Msg)]);
        let f2 = proto_fact(Multiplicity::Persistent, "St", vec![v("x", LSort::Msg)]);
        assert_ne!(canonicalize_fact(&f1), canonicalize_fact(&f2));
    }

    /// A built-in tag (`Fresh`) must never collide with a user-defined
    /// `Proto` fact whose display name happens to read the same way
    /// (`show_fact_tag` would print both as `"Fr"`/`"!Fr"`-shaped text) —
    /// the marker's variant-prefixed encoding keeps them apart.
    #[test]
    fn builtin_tag_does_not_collide_with_same_named_proto_fact() {
        let builtin = fresh_fact(v("n", LSort::Fresh));
        let user_defined = proto_fact(Multiplicity::Linear, "Fr", vec![v("n", LSort::Fresh)]);
        assert_ne!(
            canonicalize_fact(&builtin),
            canonicalize_fact(&user_defined)
        );
    }

    // -- Rules ------------------------------------------------------------

    /// Two rules with alpha-eq premises/actions/conclusions (renamed
    /// variables throughout) canonize equal.
    #[test]
    fn alpha_eq_rules_canonize_equal() {
        let r1 = rule(
            vec![in_fact(v("x", LSort::Msg))],
            vec![proto_fact(
                Multiplicity::Linear,
                "Recv",
                vec![v("x", LSort::Msg)],
            )],
            vec![out_fact(v("x", LSort::Msg))],
        );
        let r2 = rule(
            vec![in_fact(v("y", LSort::Msg))],
            vec![proto_fact(
                Multiplicity::Linear,
                "Recv",
                vec![v("y", LSort::Msg)],
            )],
            vec![out_fact(v("y", LSort::Msg))],
        );
        assert_eq!(canonicalize_rule(&r1), canonicalize_rule(&r2));
        assert_eq!(
            fingerprint_term(&canonicalize_rule(&r1)),
            fingerprint_term(&canonicalize_rule(&r2))
        );
    }

    /// The whole rule is lifted as ONE term, so a variable shared between a
    /// premise and a conclusion must canonize to the SAME literal in both
    /// places — the propagation regression from `alpha_eq_ac.rs`'s
    /// `wrong_ac_canon_example_propagates_shared_literal`, now at the rule
    /// level: `r1`'s `x` is shared between its `In` premise and first `Out`
    /// conclusion, `r2`'s `y` plays the same shared role at the same
    /// conclusion INDEX (`List` is not AC — see
    /// `same_facts_different_group_assignment_are_not_alpha_eq` — so the
    /// index has to line up, only the local names differ), so `{x -> y}`
    /// witnesses $\alphaeqac$ even though `r1`/`r2` each also mention a
    /// second, unshared variable (`z`/`w`) at the second conclusion index.
    #[test]
    fn shared_variable_across_premise_and_conclusion_propagates() {
        let r1 = rule(
            vec![in_fact(v("x", LSort::Msg))],
            vec![],
            vec![out_fact(v("x", LSort::Msg)), out_fact(v("z", LSort::Msg))],
        );
        let r2 = rule(
            vec![in_fact(v("y", LSort::Msg))],
            vec![],
            vec![out_fact(v("y", LSort::Msg)), out_fact(v("w", LSort::Msg))],
        );
        assert_eq!(canonicalize_rule(&r1), canonicalize_rule(&r2));
        assert_eq!(
            fingerprint_term(&canonicalize_rule(&r1)),
            fingerprint_term(&canonicalize_rule(&r2))
        );
    }

    /// A variable shared between a premise and a conclusion is NOT
    /// $\alphaeqac$ to the same rule with that sharing broken (two distinct
    /// variables instead) — the negative counterpart of the previous test,
    /// exactly `alpha_eq_ac.rs::ac_canon`'s point at the rule level.
    #[test]
    fn breaking_shared_variable_is_not_alpha_eq() {
        let shared = rule(
            vec![in_fact(v("x", LSort::Msg))],
            vec![],
            vec![out_fact(v("x", LSort::Msg))],
        );
        let not_shared = rule(
            vec![in_fact(v("x", LSort::Msg))],
            vec![],
            vec![out_fact(v("y", LSort::Msg))],
        );
        assert_ne!(canonicalize_rule(&shared), canonicalize_rule(&not_shared));
    }

    /// Moving a fact from one group to another (same total facts, same
    /// arguments) must not canonize equal — this is what nesting the three
    /// groups in `rule_to_term`, rather than flattening them, is for.
    #[test]
    fn same_facts_different_group_assignment_are_not_alpha_eq() {
        let f = proto_fact(Multiplicity::Linear, "P", vec![v("x", LSort::Msg)]);
        let as_premise = rule(vec![f.clone()], vec![], vec![]);
        let as_conclusion = rule(vec![], vec![], vec![f]);
        assert_ne!(
            canonicalize_rule(&as_premise),
            canonicalize_rule(&as_conclusion)
        );
    }

    /// `new_vars` does NOT participate in the lifted term: two otherwise-
    /// identical rules that differ only in `new_vars` canonize equal.
    /// `new_vars` is a deterministic function of premises/actions/conclusions
    /// (see the module docs), so once those agree, `new_vars` carries no
    /// distinguishing information; dropping it from `rule_to_term` mirrors
    /// `write_rule_to_key_excl_new_vars` in `constraint/solver/sources.rs`.
    #[test]
    fn new_vars_do_not_affect_canonization() {
        let base = rule(vec![in_fact(v("x", LSort::Msg))], vec![], vec![]);
        let with_new_var = base.clone().with_new_vars(vec![v("n", LSort::Fresh)]);
        assert_eq!(canonicalize_rule(&base), canonicalize_rule(&with_new_var));
        assert_eq!(
            fingerprint_term(&canonicalize_rule(&base)),
            fingerprint_term(&canonicalize_rule(&with_new_var))
        );
    }

    /// Idempotence carries over from `CAN_alphaeqac` (`can1`): canonizing an
    /// already-canonized rule term is a fixed point.
    #[test]
    fn canonicalize_rule_is_idempotent() {
        let r = rule(
            vec![in_fact(v("x", LSort::Msg))],
            vec![proto_fact(
                Multiplicity::Linear,
                "Recv",
                vec![v("x", LSort::Msg)],
            )],
            vec![out_fact(v("x", LSort::Msg))],
        );
        let once = canonicalize_rule(&r);
        let twice = canonicalize_alpha_eq_ac(&once, Subst::empty());
        assert_eq!(once, twice);
        assert_eq!(fingerprint_term(&once), fingerprint_term(&twice));
    }

    /// End-to-end sanity check against the real `RuleACInst` type the
    /// constraint system actually stores (`SystemContent::nodes`,
    /// `constraint/system.rs`), not just the `Rule<()>` placeholder used
    /// elsewhere in this module: two instances of "the same" rule — same
    /// `ProtoRuleACInstInfo.name`, alpha-renamed variables, and DIFFERENT
    /// `new_vars` — canonize equal.
    #[test]
    fn rule_ac_inst_renamed_vars_and_new_vars_are_alpha_eq() {
        use crate::rule::{
            ProtoRuleACInstInfo, ProtoRuleName, RuleACInst, RuleAttributes, RuleInfo,
        };

        let info = || {
            RuleInfo::Proto(ProtoRuleACInstInfo {
                name: ProtoRuleName::Stand("Login"),
                attributes: RuleAttributes::empty(),
                loop_breakers: Vec::new(),
            })
        };

        let r1: RuleACInst = Rule::new(
            info(),
            vec![in_fact(v("x", LSort::Msg))],
            vec![out_fact(v("x", LSort::Msg))],
            vec![proto_fact(
                Multiplicity::Linear,
                "Recv",
                vec![v("x", LSort::Msg)],
            )],
        );
        let r2: RuleACInst = Rule::new(
            info(),
            vec![in_fact(v("y", LSort::Msg))],
            vec![out_fact(v("y", LSort::Msg))],
            vec![proto_fact(
                Multiplicity::Linear,
                "Recv",
                vec![v("y", LSort::Msg)],
            )],
        )
        .with_new_vars(vec![v("n", LSort::Fresh)]);

        assert_eq!(canonicalize_rule(&r1), canonicalize_rule(&r2));
        assert_eq!(
            fingerprint_term(&canonicalize_rule(&r1)),
            fingerprint_term(&canonicalize_rule(&r2))
        );
    }

    // -- Guarded formulas ---------------------------------------------------

    use crate::guarded::formula_to_guarded;
    use tamarin_parser::parser::parse_formula_str;

    /// Parses a surface formula string straight to its guarded form —
    /// mirrors `guarded_tests.rs`'s own `g()` helper, panicking (rather
    /// than returning a `Result`) since every call site here is on the
    /// happy path.
    fn g(s: &str) -> Guarded {
        let f = parse_formula_str(s).unwrap_or_else(|e| panic!("parse {s:?}: {e}"));
        formula_to_guarded(&f).unwrap_or_else(|e| panic!("formula_to_guarded {s:?}: {e}"))
    }

    /// A canonical msg variable literal — same naming scheme as the
    /// term-level tests in `tamarin_term::alpha_eq_ac`'s own test suite
    /// (`mv0`, `mv1`, ...): one fixed name `"mv"`, index in `LVar::idx`.
    fn mv(idx: u64) -> LNLit {
        Lit::Var(LVar::new("mv", LSort::Msg, idx))
    }

    /// A bare free `VarSpec` (`Untagged` sort, idx 0) — matches what the
    /// parser assigns to a plain identifier like `y` in a formula string.
    fn vs(name: &str) -> p::VarSpec {
        p::VarSpec {
            name: name.to_string(),
            idx: 0,
            sort: p::SortHint::Untagged,
            typ: None,
        }
    }

    /// A free node (timepoint) `VarSpec` — matches what the parser assigns
    /// to a bare `#i` occurrence outside any binder.
    fn vs_node(name: &str) -> p::VarSpec {
        p::VarSpec {
            name: name.to_string(),
            idx: 0,
            sort: p::SortHint::Node,
            typ: None,
        }
    }

    /// The `LNLit` key/value `theta` uses for `v`.
    fn var_lit(v: &p::VarSpec) -> LNLit {
        Lit::Var(varspec_to_lvar(v))
    }

    /// An exhaustive IDENTITY `theta` over `vars`: each variable maps to
    /// itself. Point 2 requires `theta` to cover every free variable/name a
    /// formula mentions (a miss panics), so tests whose free variables don't
    /// need to actually CHANGE still have to give `theta` an entry for each
    /// of them.
    fn identity_theta(vars: &[p::VarSpec]) -> std::collections::BTreeMap<LNLit, LNLit> {
        vars.iter().map(|v| (var_lit(v), var_lit(v))).collect()
    }

    // -- 1) work.tex's `ex:canon_guarded`: $z = y \land \exists\ x\ i.\
    //    g(x)@i$, free `y`,`z`, canonical labelling $\theta = \{y \mapsto
    //    mv_1, z \mapsto mv_2\}$. The independent check: build the "expected"
    //    side by parsing the SAME formula with the canonical names spelled
    //    directly (`mv.1 = mv.2 & ...`) and canonizing via an IDENTITY
    //    `theta` (no value changes, but point 2 still requires an entry for
    //    every free variable) — two different code paths (substitute-then-
    //    normalize vs. already-canonical-then-normalize) that must converge.
    //
    //    The result's operand order differs from work.tex's own — the paper
    //    illustrates with an assumed `'∃' < '='` operator order, but
    //    `cmp_guarded`'s real (Atom < GGuarded) order sorts the equality
    //    BEFORE the existential, the opposite way around. See the module
    //    docs for why this module uses the codebase's real order rather
    //    than the paper's illustrative one.
    #[test]
    fn canonicalize_guarded_matches_work_tex_example() {
        let original = g("z = y & Ex x #i. G(x)@i");
        let theta = std::collections::BTreeMap::from([
            (var_lit(&vs("y")), mv(1)),
            (var_lit(&vs("z")), mv(2)),
        ]);
        let got = canonicalize_guarded(&original, &theta);

        // `mv.1`/`mv.2` — dot-suffix index syntax — spells the SAME `LVar`
        // as `mv(1)`/`mv(2)`: one name `"mv"`, index in the `idx` field,
        // matching `alpha_eq_ac.rs`'s actual canonical-naming scheme; a bare
        // `mv1`/`mv2` would instead parse as two unrelated names both at
        // `idx` 0.
        let identity = std::collections::BTreeMap::from([(mv(1), mv(1)), (mv(2), mv(2))]);
        let expected = canonicalize_guarded(&g("mv.1 = mv.2 & Ex x #i. G(x)@i"), &identity);
        assert_eq!(got, expected);
        assert_eq!(fingerprint_guarded(&got), fingerprint_guarded(&expected));

        // Pin the shape explicitly too: the equality atom sorts BEFORE the
        // existential (per `cmp_guarded`'s Atom < GGuarded order), and its
        // operands come out `mv1 = mv2` (smaller-first), matching the
        // string work.tex itself settles on for the equality's own operand
        // order (`mv_1 = mv_2`), just not the top-level conjunct order.
        match &got {
            Guarded::Conj(items) => {
                assert_eq!(items.len(), 2);
                assert!(matches!(&items[0], Guarded::Atom(GAtom::Eq(_, _))));
                assert!(matches!(
                    &items[1],
                    Guarded::GGuarded { qua: Quant::Ex, .. }
                ));
            }
            other => panic!("expected a top-level Conj, got {other:?}"),
        }
    }

    // -- 2) `=` is symmetric: `x = y` and `y = x` canonize equal. ----------
    #[test]
    fn swapped_eq_operands_are_alpha_eq() {
        let theta = identity_theta(&[vs("x"), vs("y")]);
        let f1 = canonicalize_guarded(&g("x = y"), &theta);
        let f2 = canonicalize_guarded(&g("y = x"), &theta);
        assert_eq!(f1, f2);
        assert_eq!(fingerprint_guarded(&f1), fingerprint_guarded(&f2));
    }

    // -- 3) `&` is commutative: swapping a top-level conjunction's operands
    //    canonizes equal. ----------------------------------------------------
    #[test]
    fn reordered_conjunction_operands_are_alpha_eq() {
        let theta = identity_theta(&[vs("x"), vs("y"), vs_node("i"), vs_node("j")]);
        let f1 = canonicalize_guarded(&g("P(x) @ #i & Q(y) @ #j"), &theta);
        let f2 = canonicalize_guarded(&g("Q(y) @ #j & P(x) @ #i"), &theta);
        assert_eq!(f1, f2);
        assert_eq!(fingerprint_guarded(&f1), fingerprint_guarded(&f2));
    }

    // -- 4) Nested conjunctions flatten (via `gconj`, reused unchanged from
    //    `guarded.rs`) before sorting, so `(A & B) & C` and `C & (B & A)`
    //    — same three conjuncts, different nesting AND order — canonize
    //    equal, exactly `CAN_AC`'s flatten-then-sort for AC terms. ----------
    #[test]
    fn nested_conjunction_flattens_and_sorts() {
        let theta = identity_theta(&[
            vs("x"),
            vs("y"),
            vs("z"),
            vs_node("i"),
            vs_node("j"),
            vs_node("k"),
        ]);
        let f1 = canonicalize_guarded(&g("(P(x) @ #i & Q(y) @ #j) & R(z) @ #k"), &theta);
        let f2 = canonicalize_guarded(&g("R(z) @ #k & (Q(y) @ #j & P(x) @ #i)"), &theta);
        assert_eq!(f1, f2);
        assert_eq!(fingerprint_guarded(&f1), fingerprint_guarded(&f2));
    }

    // -- 5) A quantifier's bound-variable display names are purely cosmetic
    //    — only their De-Bruijn POSITION matters — so `Ex x #i. G(x)@i` and
    //    `Ex a #b. G(a)@b` (same structure, differently-named binders)
    //    canonize equal. Without erasing `GBinding::name` in
    //    `ac_normalize_guarded`, this would fail: `Guarded`'s derived
    //    `PartialEq` compares binder names structurally. Both formulas are
    //    fully closed (x/i, a/b are bound), so the empty `theta` is already
    //    exhaustive. -----------------------------------------------------
    #[test]
    fn bound_variable_display_names_do_not_affect_canonization() {
        let empty = std::collections::BTreeMap::new();
        let f1 = canonicalize_guarded(&g("Ex x #i. G(x)@i"), &empty);
        let f2 = canonicalize_guarded(&g("Ex a #b. G(a)@b"), &empty);
        assert_eq!(f1, f2);
        assert_eq!(fingerprint_guarded(&f1), fingerprint_guarded(&f2));
    }

    // -- 5b) Same point, but with TWO co-bound variables in one binder list
    //    (rather than one variable per quantifier): `All x y #i. P(x,y) @
    //    #i` and `All a b #i. P(a,b) @ #i` must canonize equal too — both
    //    binder names change, not just one, and the two variables keep
    //    their relative POSITIONS (`x`/`a` first, `y`/`b` second). --------
    #[test]
    fn multi_variable_binder_display_names_do_not_affect_canonization() {
        let empty = std::collections::BTreeMap::new();
        let f1 = canonicalize_guarded(&g("All x y #i. P(x,y) @ #i ==> F"), &empty);
        let f2 = canonicalize_guarded(&g("All a b #i. P(a,b) @ #i ==> F"), &empty);
        assert_eq!(f1, f2);
        assert_eq!(fingerprint_guarded(&f1), fingerprint_guarded(&f2));
    }

    // -- 5c) Same point again, with NESTED quantifiers (a binder inside
    //    another binder's body), each level renamed independently. --------
    #[test]
    fn nested_quantifier_display_names_do_not_affect_canonization() {
        let empty = std::collections::BTreeMap::new();
        let f1 = canonicalize_guarded(
            &g("All k #i. Setup(k) @ #i ==> Ex j #t. Foo(j) @ #t"),
            &empty,
        );
        let f2 = canonicalize_guarded(
            &g("All p #q. Setup(p) @ #q ==> Ex w #z. Foo(w) @ #z"),
            &empty,
        );
        assert_eq!(f1, f2);
        assert_eq!(fingerprint_guarded(&f1), fingerprint_guarded(&f2));
    }

    // -- 6) Two formulas built from differently-named free variables, each
    //    with a `theta` renaming them to the SAME canonical labelling,
    //    canonize equal — the general form of work.tex's example. ----------
    #[test]
    fn free_variables_are_substituted_via_theta() {
        let theta1 = std::collections::BTreeMap::from([
            (var_lit(&vs("y1")), mv(1)),
            (var_lit(&vs("z1")), mv(2)),
        ]);
        let theta2 = std::collections::BTreeMap::from([
            (var_lit(&vs("y2")), mv(1)),
            (var_lit(&vs("z2")), mv(2)),
        ]);
        let f1 = canonicalize_guarded(&g("z1 = y1 & Ex x #i. G(x)@i"), &theta1);
        let f2 = canonicalize_guarded(&g("z2 = y2 & Ex x #i. G(x)@i"), &theta2);
        assert_eq!(f1, f2);
        assert_eq!(fingerprint_guarded(&f1), fingerprint_guarded(&f2));
    }

    // -- 7) Different fact names are never alpha-eq, substitution or not. --
    #[test]
    fn different_fact_names_are_not_alpha_eq() {
        let theta = identity_theta(&[vs("x"), vs_node("i")]);
        let f1 = canonicalize_guarded(&g("P(x) @ #i"), &theta);
        let f2 = canonicalize_guarded(&g("Q(x) @ #i"), &theta);
        assert_ne!(f1, f2);
    }

    // -- 8) Point 1: name CONSTANTS (`PubLit`/`FreshLit`/`NatLit`) are
    //    substituted via `theta` exactly like variables — `~'n'`/`~'m'` (two
    //    fresh names) rename to `~'fn0'`/`~'fn1'`, cross-checked the same
    //    way as test 1 (identity `theta` over the already-canonical names).
    #[test]
    fn fresh_name_literals_are_substituted_via_theta() {
        let n = Name::new(NameTag::Fresh, "n");
        let m = Name::new(NameTag::Fresh, "m");
        let fn0 = Name::new(NameTag::Fresh, "fn0");
        let fn1 = Name::new(NameTag::Fresh, "fn1");
        let theta = std::collections::BTreeMap::from([
            (Lit::Con(n), Lit::Con(fn0)),
            (Lit::Con(m), Lit::Con(fn1)),
        ]);
        // AC normalization needed here too
        let got = canonicalize_guarded(&g("~'m' = ~'n'"), &theta);

        let identity = std::collections::BTreeMap::from([
            (Lit::Con(fn0), Lit::Con(fn0)),
            (Lit::Con(fn1), Lit::Con(fn1)),
        ]);
        let expected = canonicalize_guarded(&g("~'fn0' = ~'fn1'"), &identity);
        assert_eq!(got, expected);
        assert_eq!(fingerprint_guarded(&got), fingerprint_guarded(&expected));

        match &got {
            Guarded::Atom(GAtom::Eq(GTerm::FreshLit(a), GTerm::FreshLit(b))) => {
                assert_eq!(a, "fn0");
                assert_eq!(b, "fn1");
            }
            other => panic!("expected Eq(FreshLit, FreshLit), got {other:?}"),
        }
    }

    // -- 9) Point 1, continued: a name constant nested inside a FACT
    //    argument (not just a bare equality operand) is substituted too —
    //    exercises `subst_fact_via_theta`, not just `subst_term_via_theta`
    //    at the top level of an atom. --------------------------------------
    #[test]
    fn name_literal_inside_fact_argument_is_substituted() {
        let a = Name::new(NameTag::Pub, "a");
        let pn0 = Name::new(NameTag::Pub, "pn0");
        let i = vs_node("i");
        let theta = std::collections::BTreeMap::from([
            (Lit::Con(a), Lit::Con(pn0)),
            (var_lit(&i), var_lit(&i)),
        ]);
        let got = canonicalize_guarded(&g("P('a') @ #i"), &theta);
        match &got {
            Guarded::Atom(GAtom::Action(f, _)) => {
                assert_eq!(f.args.len(), 1);
                assert!(matches!(&f.args[0], GTerm::PubLit(s) if s == "pn0"));
            }
            other => panic!("expected an Action atom, got {other:?}"),
        }
    }

    // -- A term inside an atom is ALSO brought into `CAN_AC` normal form,
    //    not just the formula-level `&`/`|`/`=`: an AC operator's operands
    //    (`xor` here) are sorted, so `a XOR b = c` and `b XOR a = c`
    //    canonize equal even though `=`'s two SIDES (`xor(..)` vs `c`)
    //    never need swapping (`xor(..)` already sorts after a bare `c` per
    //    `cmp_term`'s Lit-before-FApp class ordering, so this test isolates
    //    the fix from `normalize_atom`'s Eq-side-swap). ---------------------
    #[test]
    fn xor_equality_operands_are_ac_normalized() {
        let theta = identity_theta(&[vs("a"), vs("b"), vs("c")]);
        let f1 = canonicalize_guarded(&g("a XOR b = c"), &theta);
        let f2 = canonicalize_guarded(&g("b XOR a = c"), &theta);
        assert_eq!(f1, f2);
        match &f1 {
            // `cmp_term`'s Lit-class-before-FApp-class ordering
            // (`term_class`) puts the bare variable `c` before the `xor`
            // application, so `=`'s sides settle here, not swapped: this is
            // `normalize_atom`'s (correct) Eq-side ordering, not the AC
            // fix itself — see `xor(a,b)`'s own operand order below for
            // that.
            Guarded::Atom(GAtom::Eq(
                GTerm::Var(BVar::Free(c)),
                GTerm::BinOp(p::BinOp::Xor, x, y),
            )) => {
                assert_eq!(c.name, "c");
                // Sorted by `cmp_term`: both `a`/`b` are `Msg` vars with the
                // same `idx` (0), so the tie-break is the name string.
                assert!(matches!(&**x, GTerm::Var(BVar::Free(v)) if v.name == "a"));
                assert!(matches!(&**y, GTerm::Var(BVar::Free(v)) if v.name == "b"));
            }
            other => panic!("expected Eq(Var(c), BinOp(Xor, _, _)), got {other:?}"),
        }
    }

    // -- The same point, but exercising FLATTENING across different
    //    (re)association, not just a 2-operand swap: `a XOR b XOR c` (left-
    //    associated by the parser: `(a xor b) xor c`) and `c XOR (a XOR b)`
    //    — three leaves, differently nested AND ordered — must flatten to
    //    the same sorted operand list, exactly `CAN_AC`'s flatten-then-sort
    //    for `LNTerm`'s `f_app_ac`. --------------------------------------
    #[test]
    fn xor_chain_flattens_across_reassociation() {
        let theta = identity_theta(&[vs("a"), vs("b"), vs("c"), vs("z")]);
        let f1 = canonicalize_guarded(&g("a XOR b XOR c = z"), &theta);
        let f2 = canonicalize_guarded(&g("c XOR (a XOR b) = z"), &theta);
        assert_eq!(f1, f2);
        match &f1 {
            // Same Lit-before-FApp ordering as the previous test puts the
            // bare `z` first.
            Guarded::Atom(GAtom::Eq(
                GTerm::Var(BVar::Free(z)),
                GTerm::BinOp(p::BinOp::Xor, x, y),
            )) => {
                assert_eq!(z.name, "z");
                // Right-nested, sorted left-to-right: `xor(a, xor(b, c))`.
                assert!(matches!(&**x, GTerm::Var(BVar::Free(v)) if v.name == "a"));
                match &**y {
                    GTerm::BinOp(p::BinOp::Xor, y1, y2) => {
                        assert!(matches!(&**y1, GTerm::Var(BVar::Free(v)) if v.name == "b"));
                        assert!(matches!(&**y2, GTerm::Var(BVar::Free(v)) if v.name == "c"));
                    }
                    other => panic!("expected a nested Xor, got {other:?}"),
                }
            }
            other => panic!("expected Eq(Var(z), BinOp(Xor, _, _)), got {other:?}"),
        }
    }

    // -- The bilinear-pairing C symbol `em` (commutative, not associative,
    //    parsed as a plain `GTerm::App("em", [_, _])`) gets the same
    //    argument-sorting treatment, minus the flattening. ------------------
    #[test]
    fn em_arguments_are_sorted() {
        let theta = identity_theta(&[vs("a"), vs("b"), vs_node("i")]);
        let f1 = canonicalize_guarded(&g("P(em(a,b)) @ #i"), &theta);
        let f2 = canonicalize_guarded(&g("P(em(b,a)) @ #i"), &theta);
        assert_eq!(f1, f2);
        match &f1 {
            Guarded::Atom(GAtom::Action(fact, _)) => match &fact.args[..] {
                [GTerm::App(n, args)] if &**n == "em" => {
                    assert!(
                        matches!(&args[0], GTerm::Var(v) if matches!(v, BVar::Free(v) if v.name == "a"))
                    );
                    assert!(
                        matches!(&args[1], GTerm::Var(v) if matches!(v, BVar::Free(v) if v.name == "b"))
                    );
                }
                other => panic!("expected [App(\"em\", [_, _])], got {other:?}"),
            },
            other => panic!("expected an Action atom, got {other:?}"),
        }
    }

    // -- 10) Point 2: a free variable with NO entry in `theta` is a caller
    //    bug, not a case to fall back on silently — canonization panics. --
    #[test]
    #[should_panic(expected = "has no entry in theta")]
    fn missing_free_variable_in_theta_panics() {
        let empty = std::collections::BTreeMap::new();
        let _ = canonicalize_guarded(&g("x = y"), &empty);
    }

    // -- 11) Point 2, continued: same for a missing NAME constant. ---------
    #[test]
    #[should_panic(expected = "has no entry in theta")]
    fn missing_name_literal_in_theta_panics() {
        let empty = std::collections::BTreeMap::new();
        let _ = canonicalize_guarded(&g("~'n' = ~'n'"), &empty);
    }

    // -- 12) Idempotence: canonizing an already-canonical formula (with an
    //    identity `theta` — no further substitution to actually do, but
    //    point 2 still requires `mv(1)`/`mv(2)` to be covered) is a fixed
    //    point. -----------------------------------------------------------
    #[test]
    fn canonicalize_guarded_is_idempotent() {
        let theta = std::collections::BTreeMap::from([
            (var_lit(&vs("y")), mv(1)),
            (var_lit(&vs("z")), mv(2)),
        ]);
        let once = canonicalize_guarded(&g("z = y & Ex x #i. G(x)@i"), &theta);
        let identity = std::collections::BTreeMap::from([(mv(1), mv(1)), (mv(2), mv(2))]);
        let twice = canonicalize_guarded(&once, &identity);
        assert_eq!(once, twice);
        assert_eq!(fingerprint_guarded(&once), fingerprint_guarded(&twice));
    }
}
