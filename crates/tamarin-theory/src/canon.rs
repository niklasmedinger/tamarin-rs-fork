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

use std::collections::BTreeSet;
use std::sync::Arc;

use tamarin_term::alpha_eq_ac::{
    canonicalize_alpha_eq_ac, canonicalize_alpha_eq_ac_seeded, CanonLabelling,
};
use tamarin_term::function_symbols::{Constructability, FunSym, NoEqSym, Privacy};
use tamarin_term::lterm::LNTerm;
use tamarin_term::term::f_app_list;

use crate::canon_graph::VertexKind;
use crate::fact::{FactTag, LNFact};
use crate::rule::{ConcIdx, PremIdx, Rule};

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
    canonicalize_alpha_eq_ac(&fact_to_term(fact))
}

/// Canonizes a fact w.r.t. $\alphaeqac$, accumulating into `labelling`
/// (see [`tamarin_term::alpha_eq_ac::canonicalize_alpha_eq_ac_seeded`]) --
/// the entry point a constraint-system-wide vertex loop uses so a
/// variable/name shared between several vertices' facts gets the SAME
/// canonical literal everywhere, rather than each fact being canonized in
/// isolation like [`canonicalize_fact`] does. [`canonicalize_fact`]
/// itself is unaffected -- both are backed by the same underlying
/// `Canonizer`, just with an empty labelling that's immediately discarded.
pub fn canonicalize_fact_seeded(fact: &LNFact, labelling: &mut CanonLabelling) -> LNTerm {
    canonicalize_alpha_eq_ac_seeded(&fact_to_term(fact), labelling)
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
    canonicalize_alpha_eq_ac(&rule_to_term(rule))
}

/// Canonizes a rule w.r.t. $\alphaeqac$, accumulating into `labelling` --
/// the [`canonicalize_fact_seeded`] counterpart for a `RuleInstance`
/// graph vertex (`canon_graph::VertexKind::RuleInstance`).
pub fn canonicalize_rule_seeded<I>(rule: &Rule<I>, labelling: &mut CanonLabelling) -> LNTerm {
    canonicalize_alpha_eq_ac_seeded(&rule_to_term(rule), labelling)
}

// =============================================================================
// Action-formula vertices (`canon_graph::VertexKind::Action`)
// =============================================================================
//
// `VertexKind::Action`'s payload is `LNFact` (changed 2026-09-23, was
// `GFact`): the `GFact -> LNFact` signature-aware bridge a formula-derived
// action needs now happens once, at extraction time, in
// `canon_graph::action_vertex_fact` -- called from
// `canon_graph::extract_graph_part` itself, so a `VertexKind::Action`
// reaching this module is already fully resolved and canonizes through
// the exact same path a rule instance's own facts do
// ([`canonicalize_fact_seeded`]), with no separate bridging step or
// dedicated combinator needed here at all.

// =============================================================================
// Graph parts (Stage D/G assembly: a whole `GraphPart`, in canonical vertex
// order, INCLUDING its structural vertices and its edges)
// =============================================================================
//
// An earlier version of this section (`vertex_sequence_to_term`) lifted only
// the `RuleInstance`/`Action` vertices, skipping structural ones
// (`Dummy`/`EdgeRelation`/`LessRelation`/`AtTimepointRelation`) and ignoring
// edges entirely, on the theory that `bliss_proc::CanonicalGraph` covered
// them separately. That left a real gap: two graphs with a genuinely
// different vertex COUNT (e.g. one has an extra `Dummy` the other lacks) or
// a genuinely different EDGE set could still canonize to the identical term,
// since neither was represented at all. Fixed by encoding EVERY vertex
// (structural ones via a marker, exactly like `fact_tag_marker` marks a
// fact's tag) and the edge set (as pairs of canonical-position markers) into
// the same term.
//
// REVISED 2026-09-23, per direct user instruction: every intermediate term
// this section builds is now wrapped in a TYPE-TAGGED NoEq function
// (`graph_tag`/`graph_marker`) instead of a raw, untagged `f_app_list` --
// `VerticesTerm(...)`/`EdgesTerm(...)`/`EdgeTerm(src, tgt)`/`GraphPartTerm(vertices, edges)`
// at the `graph_part_to_term` level, `RuleInstanceVertexTerm(nid, content)`/
// `ActionVertexTerm(nid, content)`/`DummyVertexTerm(nid)` at the per-vertex
// level. Purely a clarity/collision-avoidance change, not a new gap fix:
// every one of these terms was already injective before this revision (the
// content/position embedded inside each `List` already made it
// well-defined), but an untagged `List(x, y)` gave a reader/debugger
// inspecting the raw term no way to tell WHICH kind of pairing they were
// looking at without also inspecting its position in the tree.

/// A NUL,NUL-prefixed marker name for `suffix`, distinct from both a real
/// function symbol (no valid Tamarin identifier contains a NUL byte) and
/// from [`fact_tag_marker_name`]'s own NUL-prefixed markers (which use a
/// SINGLE leading NUL followed immediately by ASCII text — a second NUL
/// byte here guarantees the two marker families can never collide).
fn graph_marker_name(suffix: &str) -> Vec<u8> {
    let mut name = vec![0u8, 0u8];
    name.extend_from_slice(suffix.as_bytes());
    name
}

/// An `args.len()`-ary NoEq function named `tag` (via [`graph_marker_name`],
/// so it's collision-proof against both real theory-declared function
/// symbols and [`fact_tag_marker`]'s own marker family — see that
/// function's own doc comment). Used to TYPE-tag `graph_part_to_term`'s own
/// intermediate terms (REVISED 2026-09-23, per direct user instruction:
/// was built from raw, untagged `f_app_list` calls — e.g. a graph part's
/// outer `(vertices, edges)` pairing was structurally indistinguishable
/// from any other 2-element `List` elsewhere in the term, and a
/// `RuleInstance`/`Action`/`Dummy` vertex's own `(NodeId, content)` pairing
/// all shared that same untagged shape too), so two structurally-similar
/// but conceptually-different sub-terms can never collide, and a debugger
/// inspecting the raw term sees which kind of thing it's looking at
/// directly in the function symbol's own name rather than having to infer
/// it from shape/position alone.
fn graph_tag(tag: &str, args: Vec<LNTerm>) -> LNTerm {
    let sym = NoEqSym::new(
        graph_marker_name(tag),
        args.len(),
        Privacy::Public,
        Constructability::Constructor,
    );
    LNTerm::App(FunSym::NoEq(sym), args.into())
}

/// The 0-ary marker term for `suffix` (see [`graph_marker_name`]) — a
/// function symbol, never a literal, so it is NEVER renamed by
/// `CAN_alphaeqac` (unlike e.g. a public name, which — being a literal —
/// the canonizer would try to assign a fresh canonical name to, making it
/// useless as a fixed marker of vertex/edge IDENTITY). A 0-ary special
/// case of [`graph_tag`].
fn graph_marker(suffix: &str) -> LNTerm {
    graph_tag(suffix, Vec::new())
}

/// Lifts one graph vertex to a term: a `RuleInstance`/`Action`/`Dummy`
/// vertex lifts as a type-tagged `RuleInstanceVertexTerm(node_id_term(nid), content)` /
/// `ActionVertexTerm(node_id_term(nid), content)` / `DummyVertexTerm(node_id_term(nid))`
/// -- ITS OWN NodeId FIRST, as a bare variable literal, directly before the
/// vertex's own content ([`rule_to_term`]; [`fact_to_term`], `VertexKind::Action`'s
/// payload already being a plain `LNFact`; or nothing at all for `Dummy`,
/// whose tag alone already identifies it). A structural,
/// NodeId-free vertex (`EdgeRelation`/`LessRelation`/`AtTimepointRelation`/
/// `LastAtomRelation`) lifts to a marker identifying its KIND (and, for
/// `EdgeRelation`, its port indices — the same information its
/// `ColorTable` color already encodes, here reified as a term instead of
/// a side-channel integer).
///
/// **Embedding the NodeId is not cosmetic — it closes a real theta-coverage
/// gap.** Before this, a vertex's OWN `NodeId` was discarded entirely
/// (`RuleInstance(_, ru) => rule_to_term(ru)`, `_` dropping the id): a
/// `NodeId` never appeared as a literal ANYWHERE in the graph-part term,
/// so `theta` (built purely from literals the Canonizer discovers while
/// walking that term) never contained an entry for any `NodeId`, ever --
/// regardless of how well-connected that node's vertex was. That was
/// invisible as long as every formula only ever referenced a `NodeId`
/// while BOUND (bound variables need no theta lookup at all — see
/// `subst_via_theta_guarded`'s own doc comment), which held for every
/// fixture this pipeline had been tested against. It breaks the moment a
/// formula references a `NodeId` FREE: `guarded::to_induction_hypothesis`
/// (the `ginduct`/`Induction` proof method) does exactly that, injecting
/// a free top-level `Last(v)` disjunct per Node-sorted quantified
/// variable — `canonicalize_guarded`'s `lookup_theta` then panics on the
/// very first such formula it meets, since `theta` never had anywhere to
/// put that entry. Embedding each vertex's `NodeId` as a literal, in
/// canonical vertex order and directly before that vertex's own content,
/// makes the Canonizer discover and canonically rename it exactly like
/// any other literal (to `canonical_var(LSort::Node, _)`'s reserved
/// `"tv"` family, previously dead code for want of a caller) — so
/// `theta` becomes genuinely exhaustive over `NodeId`s, not just over
/// vertex content, closing the gap at its source rather than papering
/// over one formula at a time.
fn vertex_to_term(v: &VertexKind) -> LNTerm {
    match v {
        VertexKind::RuleInstance(nid, ru) => {
            graph_tag("RuleInstanceVertexTerm", vec![node_id_term(*nid), rule_to_term(ru)])
        }
        VertexKind::Action(nid, fact) => {
            graph_tag("ActionVertexTerm", vec![node_id_term(*nid), fact_to_term(fact)])
        }
        VertexKind::Dummy(nid) => graph_tag("DummyVertexTerm", vec![node_id_term(*nid)]),
        VertexKind::EdgeRelation(conc, prem) => {
            graph_marker(&format!("EdgeRelation:{}:{}", conc.0, prem.0))
        }
        VertexKind::LessRelation => graph_marker("LessRelation"),
        VertexKind::AtTimepointRelation => graph_marker("AtTimepointRelation"),
        VertexKind::LastAtomRelation => graph_marker("LastAtomRelation"),
    }
}

/// Lifts a `NodeId` (an `LVar` of `LSort::Node`) to a bare variable
/// literal term, so it participates in Canonizer discovery exactly like
/// any other literal — see [`vertex_to_term`]'s own doc comment for why
/// this exists.
fn node_id_term(nid: crate::constraint::constraints::NodeId) -> LNTerm {
    tamarin_term::vterm::var_term(nid)
}

/// A marker term naming canonical vertex position `i` — used to encode an
/// edge's endpoints. A function symbol (like [`graph_marker`]), not a
/// name/variable literal: a canonical POSITION is already a fixed,
/// meaningful number, not something `CAN_alphaeqac` should be free to
/// rename.
fn index_marker(i: usize) -> LNTerm {
    graph_marker(&format!("idx:{i}"))
}

/// Lifts a canonical-order vertex sequence
/// (`bliss_proc::canonical_vertex_order`'s output) PLUS its
/// canonical-position edge set (`bliss_proc::canonical_edges`'s output)
/// to ONE term — the constraint-system-wide generalization of
/// [`rule_to_term`]/[`fact_to_term`]: every vertex becomes its own
/// sub-term ([`vertex_to_term`]), nested inside one type-tagged
/// `VerticesTerm(...)` in vertex order, paired with a second type-tagged
/// `EdgesTerm(...)` of edges (each edge itself a type-tagged
/// `EdgeTerm(src, tgt)` of [`index_marker`]s), the two combined under an
/// outer `GraphPartTerm(vertices, edges)` (all via [`graph_tag`] --
/// REVISED 2026-09-23 from raw, untagged `f_app_list`/`List` applications,
/// see this section's own module doc comment for why). None of these
/// wrapper functions are AC, so every ordering here stays significant: the
/// vertex order is what makes two isomorphic-but-differently-numbered
/// systems line up position-for-position once each is in its OWN bliss
/// canonical order; `edges` is expected pre-sorted (a `BTreeSet`, as
/// `canonical_edges` returns) so the edge list doesn't depend on
/// `GraphPart::edges`'s own creation order.
pub fn graph_part_to_term(ordered: &[&VertexKind], edges: &BTreeSet<(usize, usize)>) -> LNTerm {
    let vertices_term = graph_tag(
        "VerticesTerm",
        ordered.iter().map(|v| vertex_to_term(v)).collect(),
    );
    let edges_term = graph_tag(
        "EdgesTerm",
        edges
            .iter()
            .map(|&(src, tgt)| graph_tag("EdgeTerm", vec![index_marker(src), index_marker(tgt)]))
            .collect(),
    );
    graph_tag("GraphPartTerm", vec![vertices_term, edges_term])
}

/// Canonizes a graph part w.r.t. $\alphaeqac$ (see
/// [`graph_part_to_term`]) — the graph-part-wide counterpart to
/// [`canonicalize_rule`]/[`canonicalize_fact`], calling
/// [`canonicalize_alpha_eq_ac`] exactly once over the WHOLE assembled
/// term rather than threading a [`CanonLabelling`] across many separate
/// per-vertex calls: since every vertex's content is nested inside one
/// term before canonization ever runs, one `Canonizer` pass already
/// treats every vertex's literals under one shared worklist, which is
/// all the accumulating (`_seeded`) functions above exist to simulate
/// across SEPARATE calls.
///
/// This is a SUFFICIENT canonical encoding of everything
/// `canon_graph::extract_graph_part` extracts (rule instances, action
/// facts, `System::edges`, and `less_atoms` — the last two both entering
/// the graph via `EdgeRelation`/`LessRelation` vertices, so they're
/// already covered by `ordered`/`edges` with no separate handling
/// needed). It does NOT cover what `extract_graph_part` leaves out of the
/// graph entirely: `lemmas`, `eq_store`, `subterm_store`, and any formula
/// content beyond ground action atoms — those are handled separately, by
/// `canonicalize_constraint_system` (Stage G, below), which extends this
/// function's own accumulated labelling through them.
pub fn canonicalize_graph_part(
    ordered: &[&VertexKind],
    edges: &BTreeSet<(usize, usize)>,
) -> LNTerm {
    canonicalize_alpha_eq_ac(&graph_part_to_term(ordered, edges))
}

// =============================================================================
// Minimum over automorphisms (Stage F)
// =============================================================================
//
// work.tex's own suggested tie-break for a non-unique graph canonizer --
// "pick the graph with the lexicographically smallest adjacency matrix" --
// cannot actually work: by the definition of automorphism, EVERY element
// of Aut(G) preserves the adjacency matrix exactly, so that criterion is
// vacuous for choosing among automorphism-related labelings (it only ever
// distinguishes graphs that aren't isomorphic to begin with). The
// resolution needs the CONTENT the graph's colors are coarser than --
// i.e. `canonicalize_graph_part`'s own term-level result.

/// Resolves "minimum over automorphisms" for a graph part: applies every
/// element of the CLOSED automorphism group (`generate_group` on
/// `result.generators` -- NOT just the raw generators bliss reports; see
/// its own doc comment for why that would miss candidates) to
/// `result.canonical_labeling`, canonizes each resulting candidate via
/// [`canonicalize_graph_part`], and returns every `(labeling, term)` pair
/// achieving the minimum term.
///
/// **This resolves the ambiguity only at the graph-part level.** When the
/// minimum is achieved by more than one labeling — a real possibility
/// whenever the graph has a genuinely symmetric substructure whose
/// content also ties under canonization — a caller needing a single,
/// well-defined SYSTEM-wide canonical form must NOT pick one of the
/// returned survivors arbitrarily: a formula elsewhere in the system can
/// reference one of the tied substructures' variables asymmetrically,
/// and different (both graph-part-minimal) tie-breaks can then make two
/// genuinely $\alphaeqac$ systems land on different final canonical
/// forms. Breaking such a tie needs the REST of the system (formulas/
/// lemmas/eq_store/subterm_store) as a secondary key, by extending each
/// survivor's own accumulated labelling and re-minimizing at the
/// full-system level — implemented in `canonicalize_constraint_system`
/// (below), which is the caller that actually needs more than one
/// survivor; THIS function deliberately still stops at returning every
/// tied candidate rather than picking one itself, since it has no way to
/// know whether the caller needs that system-level tie-break.
pub fn minimal_graph_part_labelings(
    part: &crate::canon_graph::GraphPart,
    result: &crate::bliss_proc::BlissResult,
) -> Vec<(crate::bliss_proc::Permutation, LNTerm)> {
    use crate::bliss_proc::{canonical_edges, canonical_vertex_order, generate_group};

    let group = generate_group(&result.generators, part.vertices.len());
    let mut best: Vec<(crate::bliss_proc::Permutation, LNTerm)> = Vec::new();
    for g in group {
        let candidate_labeling = result.canonical_labeling.compose(&g);
        let ordered = canonical_vertex_order(part, &candidate_labeling);
        let edges = canonical_edges(part, &candidate_labeling);
        let term = canonicalize_graph_part(&ordered, &edges);
        match best.first() {
            None => best.push((candidate_labeling, term)),
            Some((_, best_term)) => match term.cmp(best_term) {
                std::cmp::Ordering::Less => {
                    best.clear();
                    best.push((candidate_labeling, term));
                }
                std::cmp::Ordering::Equal => {
                    best.push((candidate_labeling, term));
                }
                std::cmp::Ordering::Greater => {}
            },
        }
    }
    best
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
// system's rule instances/action formulas via graph-part canonization,
// below — Stage G's `canonicalize_constraint_system` is the actual call
// site that threads that `theta` in here). `theta` is assumed EXHAUSTIVE: every
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

// =============================================================================
// Whole-system assembly (Stage G) -- field-by-field canonicalization of a
// whole constraint `System`
// =============================================================================
//
// `canonicalize_constraint_system` composes every earlier stage:
//   A/B (extract_graph_part) -> C (run_bliss) -> F (minimal_graph_part_labelings)
//   -> G (this section: extend each graph-part survivor's own labelling
//   through formulas/solved_formulas/lemmas/eq_store/subterm_store, then
//   take the minimum `CanonicalSystem` over survivors).
//
// Every field, including `eq_store.conj`'s per-alternative range-term
// canonicalization, is real, working code, exercised end to end by
// `tests/canonicalize_constraint_system_tutorial.rs` against two real
// captured Tutorial systems -- though that fixture pair happens to have an
// empty `eq_store.conj`, so `canonicalize_eq_disj_alternative`'s own logic
// (below) is not yet validated against real non-empty data, only reasoned
// through and unit-tested directly (`tests::eq_disj_alternative_*`).

use crate::bliss_proc::BlissError;
use crate::constraint::constraints::Goal;
use crate::constraint::solver::proof_method::{ProofMethod, Result as FinishedResult};
use crate::constraint::solver::Contradiction;
use crate::constraint::system::{GoalStatus, Side, SourceKind, System};
use crate::tools::equation_store::{EqDisj, EquationStore, LNSubst, LNSubstVFresh};
use crate::tools::subterm_store::{SubtermConstraint, SubtermStore};
use tamarin_term::alpha_eq_ac::apply_literal_renaming;
use tamarin_term::lterm::LVar;
use tamarin_term::vterm::var_term;

/// The complete canonical form of a constraint `System` (Stage G).
///
/// `PartialEq` only, no `Eq`/`Ord` derive: `Guarded` itself derives only
/// `PartialEq` (no `Eq`, no `Ord` -- only the free [`cmp_guarded`]), so
/// neither derives here for `CanonicalSystem` either. A full total order
/// needs the dedicated [`cmp_canonical_system`] function instead.
#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalSystem {
    /// Rule instances, action facts, `System::edges`, `less_atoms` (via
    /// `LessRelation`), and `last_atom` (via `LastAtomRelation`) --
    /// everything [`graph_part_to_term`] covers.
    pub graph_part: LNTerm,
    /// `formulas`: canonicalized via the graph part's accumulated
    /// labelling, sorted via `cmp_guarded` -- **NOT deduplicated**
    /// (REVISED 2026-09-23, per direct user instruction: `stores_contains`
    /// only prevents RAW-syntactic duplicates at insertion, not
    /// alpha-equivalent ones, so two solver-tracked-as-distinct formulas
    /// could canonicalize identically; see [`sort_guarded`]'s own doc
    /// comment for the full reasoning, which mirrors `eq_store.conj`'s
    /// pre-existing multiset-preservation policy). KEPT SEPARATE
    /// from `solved_formulas` -- membership in one store vs. the other is
    /// live proof-search state (e.g. `ProofMethod::Induction`'s
    /// applicability, `isInitialSystem`), not pure memoization, so
    /// merging them would conflate two non-interchangeable systems.
    pub formulas: Vec<Guarded>,
    /// `solved_formulas`, canonicalized the same way as `formulas`
    /// (sorted, not deduplicated), as its own independent list.
    pub solved_formulas: Vec<Guarded>,
    /// `lemmas`, canonicalized the same way.
    pub lemmas: Vec<Guarded>,
    pub eq_store: CanonicalEqStore,
    pub subterm_store: CanonicalSubtermStore,
    /// `sys.goals`, minus `Goal::Split` (dropped -- it only NAMES an
    /// `EqDisj` already present in `eq_store.conj`; nothing else ever
    /// reads its raw id, the exact same reasoning that already dropped
    /// `EqDisj::split_id` itself). See [`canonicalize_goals`] for each
    /// variant's canonicalization and why `GoalStatus::solved` is kept
    /// (content-relevant: gates whether `candidate_methods` offers the goal
    /// again) while `::looping`/`::nr` are not (pure heuristic-ranking /
    /// path-dependent bookkeeping). Sorted, NOT deduplicated -- see
    /// [`canonicalize_goals`]'s own doc comment.
    pub goals: Vec<CanonicalGoal>,
    pub source_kind: Option<SourceKind>,
    pub side: Option<Side>,
}

/// Canonicalized form of one non-`Split` [`Goal`], paired
/// with whether it's currently solved -- see [`CanonicalSystem::goals`]'s
/// own doc comment.
///
/// `PartialEq` only, no `Eq`/`Ord`: `Disj` embeds `Vec<Guarded>`, and
/// `Guarded` itself has no `Ord`/`Eq` -- same reason [`CanonicalSystem`]
/// itself only derives `PartialEq`. [`cmp_canonical_goal`] is the total
/// order.
#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalGoal {
    pub kind: CanonicalGoalKind,
    pub solved: bool,
}

/// The canonicalized payload of one [`CanonicalGoal`] -- see
/// [`canonicalize_goals`] for exactly how each variant is derived from
/// the matching [`Goal`] variant.
#[derive(Debug, Clone, PartialEq)]
pub enum CanonicalGoalKind {
    /// `Goal::Action(LVar, LNFact)`: the `NodeId` and the fact canonicalized
    /// via `theta`. The same `(NodeId, fact)` is also a `VertexKind::Action`
    /// vertex in `graph_part` (which is what puts its literals into `theta`);
    /// the vertex records that the action exists, this entry records that
    /// it is a GOAL and whether it is solved -- neither of which the vertex
    /// carries, and both of which decide what `candidate_methods` offers.
    Action(LNLit, LNTerm),
    /// `Goal::Chain(NodeConc, NodePrem)`: both endpoints' `NodeId`s
    /// canonicalized via the graph part's `theta` -- no new literal, the
    /// port indices need no renaming.
    Chain(LNLit, ConcIdx, LNLit, PremIdx),
    /// `Goal::Premise(NodePrem, LNFact)`: the `NodeId` canonicalized the
    /// same way as `Chain`'s endpoints, plus the fact itself
    /// canonicalized via `theta` (its own literals should already be
    /// covered -- `NodePrem` names an EXISTING rule instance's own
    /// premise slot, whose fact the graph part already discovered).
    Premise(LNLit, PremIdx, LNTerm),
    /// `Goal::Disj(Disj<Guarded>)`: each alternative canonicalized via
    /// `canonicalize_guarded` against `theta`, then sorted -- NOT
    /// deduplicated, same treatment as `formulas`/`solved_formulas`/`lemmas`.
    Disj(Vec<Guarded>),
    /// `Goal::Subterm((LNTerm, LNTerm))`: both terms canonicalized via
    /// `apply_literal_renaming` -- same treatment as `subterm_store`'s
    /// pairs.
    Subterm(LNTerm, LNTerm),
}

/// All four fields are `Ord` on their own, so this derives it directly
/// (unlike [`CanonicalSystem`], which needs a custom comparison only
/// because of its `Vec<Guarded>` fields).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CanonicalEqStore {
    /// `eq_store.subst`, canonical-key sorted.
    pub subst: Vec<(LNLit, LNTerm)>,
    /// `eq_store.conj` -- `split_id` DROPPED: its only consumer is
    /// `Goal::Split`, which `CanonicalSystem::goals` drops too (it only
    /// NAMES an `EqDisj` already present here; nothing else ever reads a
    /// `split_id` value -- see `CanonicalSystem::goals`'s own doc
    /// comment). Each alternative is stored
    /// fully canonicalized, not as a raw `LNSubstVFresh`: domain keys
    /// renamed via the shared `theta`, each range term canonicalized via
    /// a per-alternative-forked `CanonLabelling` pass (see
    /// [`canonicalize_eq_disj_alternative`]). The middle `Vec` (one
    /// `EqDisj`'s alternatives) and the
    /// outer `Vec` (the list of `EqDisj`s) are both sorted by their own
    /// canonicalized content directly, as MULTISETS: duplicates from
    /// alpha-equivalent alternatives/disjunctions are preserved, never
    /// deduplicated. Tamarin's own solver keeps them distinct on purpose
    /// (see `tools/equation_store.rs`'s own comment on its
    /// `applyBound`/`S.fromList` handling for the proven real regression:
    /// alpha-canonically deduping a disjunction's alternatives there
    /// collapses a real 6-substitution case to 5, dropping a split case
    /// and changing the proof tree), so collapsing them here would
    /// likewise conflate two non-interchangeable systems.
    pub conj: Vec<Vec<Vec<(LNLit, LNTerm)>>>,
}

/// All four fields are `Ord` on their own, so this derives it directly --
/// see [`CanonicalEqStore`]'s own doc comment.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CanonicalSubtermStore {
    pub subterms: Vec<(LNTerm, LNTerm)>,
    pub solved_subterms: Vec<(LNTerm, LNTerm)>,
    pub contradictory: bool,
    pub neg_subterms: Vec<(LNTerm, LNTerm)>,
}

/// Lexicographic comparison over [`CanonicalSystem`]'s fields, in the same
/// order the driver ([`canonicalize_constraint_system`]) fills them in.
/// Needed because `Guarded` has no `Ord` impl, so `CanonicalSystem` can't
/// `#[derive(Ord)]` -- see its own doc comment. This is what the
/// system-level tie-break minimizes over when `minimal_graph_part_labelings`
/// returns more than one graph-part-level survivor.
pub fn cmp_canonical_system(a: &CanonicalSystem, b: &CanonicalSystem) -> std::cmp::Ordering {
    a.graph_part
        .cmp(&b.graph_part)
        .then_with(|| cmp_guarded_slice(&a.formulas, &b.formulas))
        .then_with(|| cmp_guarded_slice(&a.solved_formulas, &b.solved_formulas))
        .then_with(|| cmp_guarded_slice(&a.lemmas, &b.lemmas))
        .then_with(|| a.eq_store.cmp(&b.eq_store))
        .then_with(|| a.subterm_store.cmp(&b.subterm_store))
        .then_with(|| cmp_canonical_goal_slice(&a.goals, &b.goals))
        .then_with(|| a.source_kind.cmp(&b.source_kind))
        .then_with(|| a.side.cmp(&b.side))
}

/// Lexicographic comparison of two already-canonical [`CanonicalGoal`]
/// slices via [`cmp_canonical_goal`] (no `Ord for CanonicalGoal` -- see
/// its own doc comment), tie-broken by length -- same shape as
/// [`cmp_guarded_slice`], just for goals.
fn cmp_canonical_goal_slice(a: &[CanonicalGoal], b: &[CanonicalGoal]) -> std::cmp::Ordering {
    for (x, y) in a.iter().zip(b.iter()) {
        let c = cmp_canonical_goal(x, y);
        if c != std::cmp::Ordering::Equal {
            return c;
        }
    }
    a.len().cmp(&b.len())
}

/// Total order over [`CanonicalGoal`] -- constructor rank
/// (`Action < Chain < Premise < Disj < Subterm`, this enum's declaration order,
/// mirroring [`crate::constraint::constraints::cmp_goal`]'s own style)
/// then payload, `solved` last (so two goals with the same canonicalized
/// content but different solved-status sort adjacently, not scattered).
fn cmp_canonical_goal(a: &CanonicalGoal, b: &CanonicalGoal) -> std::cmp::Ordering {
    cmp_canonical_goal_kind(&a.kind, &b.kind).then_with(|| a.solved.cmp(&b.solved))
}

fn cmp_canonical_goal_kind(a: &CanonicalGoalKind, b: &CanonicalGoalKind) -> std::cmp::Ordering {
    fn rank(k: &CanonicalGoalKind) -> u8 {
        match k {
            CanonicalGoalKind::Action(..) => 0,
            CanonicalGoalKind::Chain(..) => 1,
            CanonicalGoalKind::Premise(..) => 2,
            CanonicalGoalKind::Disj(_) => 3,
            CanonicalGoalKind::Subterm(..) => 4,
        }
    }
    let (ra, rb) = (rank(a), rank(b));
    if ra != rb {
        return ra.cmp(&rb);
    }
    // Rank equality above guarantees the same variant, so each `let …
    // else` binding of `b` is infallible.
    match a {
        CanonicalGoalKind::Action(n1, f1) => {
            let CanonicalGoalKind::Action(n2, f2) = b else {
                unreachable!("rank matched Action")
            };
            n1.cmp(n2).then_with(|| f1.cmp(f2))
        }
        CanonicalGoalKind::Chain(cl1, ci1, pl1, pi1) => {
            let CanonicalGoalKind::Chain(cl2, ci2, pl2, pi2) = b else {
                unreachable!("rank matched Chain")
            };
            cl1.cmp(cl2)
                .then_with(|| ci1.cmp(ci2))
                .then_with(|| pl1.cmp(pl2))
                .then_with(|| pi1.cmp(pi2))
        }
        CanonicalGoalKind::Premise(pl1, pi1, f1) => {
            let CanonicalGoalKind::Premise(pl2, pi2, f2) = b else {
                unreachable!("rank matched Premise")
            };
            pl1.cmp(pl2).then_with(|| pi1.cmp(pi2)).then_with(|| f1.cmp(f2))
        }
        CanonicalGoalKind::Disj(d1) => {
            let CanonicalGoalKind::Disj(d2) = b else {
                unreachable!("rank matched Disj")
            };
            cmp_guarded_slice(d1, d2)
        }
        CanonicalGoalKind::Subterm(a1, b1) => {
            let CanonicalGoalKind::Subterm(a2, b2) = b else {
                unreachable!("rank matched Subterm")
            };
            a1.cmp(a2).then_with(|| b1.cmp(b2))
        }
    }
}

/// Lexicographic comparison of two already-canonical `Guarded` slices via
/// `cmp_guarded` (there is no `Ord for Guarded` to fall back on -- see
/// `cmp_canonical_system`'s own doc comment), tie-broken by length when one
/// is a prefix of the other.
fn cmp_guarded_slice(a: &[Guarded], b: &[Guarded]) -> std::cmp::Ordering {
    for (x, y) in a.iter().zip(b.iter()) {
        let c = cmp_guarded(x, y);
        if c != std::cmp::Ordering::Equal {
            return c;
        }
    }
    a.len().cmp(&b.len())
}

/// Sorts `v` via `cmp_guarded` (no `Ord for Guarded`, see above) into a
/// content-driven, cross-system-stable order -- **deliberately does NOT
/// dedup**, unlike the similarly-named `sort_dedup_guarded` local closure
/// in `constraint/solver/rename_precise.rs` (that one mirrors HS's `S.Set
/// LNGuarded` rebuild, which sorts AND collision-dedups after every
/// rename -- a real solver-internal behavior, not this function's
/// concern). REVISED 2026-09-23, per direct user instruction: every
/// `CanonicalSystem` field built from a `System` `Vec` (not an
/// already-deduplicated `Set`-typed field, e.g. `neg_subterms`) must
/// preserve its exact multiplicity, not just its distinct content --
/// exactly the same principle `eq_store.conj`'s multiset preservation was
/// already built on (see `CanonicalEqStore::conj`'s own doc comment: a
/// solver-level dedup can silently drop a real split case, confirmed on a
/// real regression). `formulas`/`solved_formulas`/`lemmas` (`Vec<Arc<Guarded>>`
/// in `System`) and a `Goal::Disj`'s own alternatives (`Vec<Guarded>`) are
/// exactly this shape -- two syntactically-identical entries are not
/// assumed to be redundant, since nothing here has proven the solver never
/// produces a real duplicate.
fn sort_guarded(mut v: Vec<Guarded>) -> Vec<Guarded> {
    v.sort_by(cmp_guarded);
    v
}

/// The `_seeded` sibling of [`canonicalize_graph_part`] the driver needs:
/// also returns the [`CanonLabelling`] the canonization produced, so a
/// caller can CONTINUE it through the rest of the system instead of
/// discarding it. Needed because a graph-part tie must be broken using the
/// rest of the system, not by picking one of several tied survivors
/// arbitrarily and discarding its labelling -- see
/// [`minimal_graph_part_labelings`]'s own doc comment for the concrete
/// counterexample.
pub fn canonicalize_graph_part_seeded(
    ordered: &[&VertexKind],
    edges: &BTreeSet<(usize, usize)>,
) -> (LNTerm, CanonLabelling) {
    let mut labelling = CanonLabelling::empty();
    let term = canonicalize_alpha_eq_ac_seeded(&graph_part_to_term(ordered, edges), &mut labelling);
    (term, labelling)
}

/// Canonizes a whole constraint `System` (Stage G, composing A-F): builds
/// the graph part, asks bliss for its automorphism group, resolves
/// "minimum over automorphisms" at the graph-part level (Stage F), then --
/// for EVERY graph-part-level survivor, not just bliss's own pick --
/// extends that survivor's own accumulated labelling through the rest of
/// the system and takes the minimum `CanonicalSystem` over all of them
/// (the system-level tie-break Stage F's own doc comment requires).
///
/// Fallible only through the two bliss subprocess steps
/// (`graph_part_to_dimacs`/`run_bliss`); everything after that is pure.
pub fn canonicalize_constraint_system(
    sys: &System,
    colors: &crate::canon_color::ColorTable,
) -> Result<CanonicalSystem, BlissError> {
    canonicalize_constraint_system_with_labelling(sys, colors).map(|(canon, _)| canon)
}

/// [`canonicalize_constraint_system`], also returning the winning
/// survivor's [`CanonLabelling`] -- the renaming that maps `sys` onto its
/// canonical form, for canonicalizing anything else derived from `sys`
/// consistently with it (e.g. its proof methods, [`canonicalize_proof_method`]).
/// With several tied survivors this is one of them; anything canonicalized
/// through it should be compared as a multiset, which a symmetry of the
/// system maps onto itself.
pub fn canonicalize_constraint_system_with_labelling(
    sys: &System,
    colors: &crate::canon_color::ColorTable,
) -> Result<(CanonicalSystem, CanonLabelling), BlissError> {
    // Stage A+B: vertex/edge structure plus the caller-supplied vertex
    // coloring, in one call -- see `canon_graph`'s own module docs for
    // why `GraphPart` owns its `ColorTable` rather than threading it
    // separately. No `&Theory` needed here at all: `colors` is unique per
    // theory and the caller already has one on hand (a `ProofContext`
    // carries its own `color_table`; see `canon_color.rs`'s own "why this
    // table takes..." doc section).
    let part = crate::canon_graph::extract_graph_part(sys, colors);

    // An empty graph part (no rule-instance/action/dummy vertices at all
    // -- e.g. the root of a proof, before Simplify/goal-solving
    // introduces any rule instances, or induction's `empty_trace` base
    // case) is TRIVIALLY canonical: there is exactly one empty vertex
    // sequence and one empty edge set, so there is nothing for an
    // automorphism search to resolve. Bliss itself refuses a 0-vertex
    // graph (`graph_part_to_dimacs` returns `BlissError::EmptyGraph`
    // rather than handing bliss a DIMACS header it would choke on), so
    // skip Stage C/F entirely here rather than hard-failing a case that
    // has a perfectly well-defined canonical form and plenty of other
    // canonicalizable content (formulas, eq_store, ...) still to process.
    if part.vertices.is_empty() {
        let (graph_term, labelling) = canonicalize_graph_part_seeded(&[], &BTreeSet::new());
        let canon = canonicalize_system_content_seeded(sys, &labelling, graph_term);
        return Ok((canon, labelling));
    }

    // Stage C: bliss's own canonical labeling plus a GENERATING set for
    // the graph's automorphism group (not necessarily the full group --
    // see `generate_group`'s own doc comment for why iterating just the
    // raw generators can miss candidates).
    let dimacs = crate::bliss_proc::graph_part_to_dimacs(&part)?;
    let result = crate::bliss_proc::run_bliss(&dimacs)?;

    // Stage F: every labeling in the CLOSED automorphism group achieving
    // the minimum graph-part TERM (not just bliss's single pick) --
    // `minimal_graph_part_labelings` deliberately returns every tied
    // survivor rather than choosing one, because a graph-part-level tie
    // is not always resolvable without looking at the rest of the system
    // (see its own doc comment for the concrete counterexample; the loop
    // below is what actually resolves it, per survivor).
    let survivors = minimal_graph_part_labelings(&part, &result);
    debug_assert!(
        !survivors.is_empty(),
        "bliss always returns at least the identity labeling as a candidate"
    );

    // Stage G: for EACH survivor, re-derive its own accumulated
    // `CanonLabelling` (not just its term -- `minimal_graph_part_labelings`
    // only returns the term, so this recomputes the same canonization a
    // second time via the `_seeded` sibling to also recover the
    // labelling; cheap relative to the bliss subprocess call this reuses
    // no new invocation of) and extend it through formulas ->
    // solved_formulas -> lemmas -> eq_store -> subterm_store -- a fixed,
    // deterministic order.
    let mut candidates: Vec<(CanonicalSystem, CanonLabelling)> = Vec::with_capacity(survivors.len());
    for (labeling, graph_term) in &survivors {
        let ordered = crate::bliss_proc::canonical_vertex_order(&part, labeling);
        let edges = crate::bliss_proc::canonical_edges(&part, labeling);
        let (graph_term_again, labelling) = canonicalize_graph_part_seeded(&ordered, &edges);
        debug_assert_eq!(
            &graph_term_again, graph_term,
            "re-running canonicalize_graph_part_seeded for a winning labeling must \
             reproduce the same term minimal_graph_part_labelings already found"
        );
        let canon = canonicalize_system_content_seeded(sys, &labelling, graph_term_again);
        candidates.push((canon, labelling));
    }

    // Stage F's system-level tie-break: the minimum CanonicalSystem over
    // every graph-part-level survivor, via the dedicated comparison
    // function `Guarded`'s missing `Ord` impl forces (see
    // `cmp_canonical_system`'s own doc comment).
    Ok(candidates
        .into_iter()
        .min_by(|(a, _), (b, _)| cmp_canonical_system(a, b))
        .expect("`candidates` has one entry per (non-empty) `survivors` entry"))
}

/// Extends `labelling` (already seeded from a graph-part survivor) through
/// every remaining field of `sys` this module treats as part of the
/// canonical form, building the rest of a [`CanonicalSystem`].
///
/// Read-only (`&CanonLabelling`, not `&mut`): nothing past the graph part
/// ever discovers a new literal EXCEPT `eq_store.conj`'s per-alternative
/// range terms, and even those are canonized against their own
/// independent FORK of `labelling` (see
/// [`canonicalize_eq_disj_alternative`]'s own doc comment for why),
/// never `labelling` itself -- so `labelling` truly never changes once
/// this function is called.
///
/// `pub(crate)` (not private) so [`canonicalize_system_content_seeded_profiled`]'s
/// own doc comment -- and any other crate-internal profiling -- can
/// reference it directly; still not part of the crate's public API.
pub(crate) fn canonicalize_system_content_seeded(
    sys: &System,
    labelling: &CanonLabelling,
    graph_part: LNTerm,
) -> CanonicalSystem {
    // formulas / solved_formulas / lemmas: read-only against the `theta`
    // accumulated by the graph part -- `canonicalize_guarded` assumes
    // `theta` is EXHAUSTIVE (work.tex's guardedness argument: every free
    // variable/name constant a formula mentions is bound elsewhere in the
    // system) and panics on a miss rather than silently miscanonizing
    // (see `lookup_theta`). Kept as three SEPARATE sets, not unioned --
    // `solved_formulas` membership is live proof-search state, not pure
    // memoization: `ProofMethod::Induction`'s applicability and
    // `isInitialSystem` both key off it.
    let formulas = sort_guarded(
        sys.formulas
            .iter()
            .map(|f| canonicalize_guarded(f, labelling.theta()))
            .collect(),
    );
    let solved_formulas = sort_guarded(
        sys.solved_formulas
            .iter()
            .map(|f| canonicalize_guarded(f, labelling.theta()))
            .collect(),
    );
    let lemmas = sort_guarded(
        sys.lemmas
            .iter()
            .map(|f| canonicalize_guarded(f, labelling.theta()))
            .collect(),
    );

    let eq_store = canonicalize_eq_store(&sys.eq_store, labelling);
    let subterm_store = canonicalize_subterm_store(&sys.subterm_store, labelling.theta());
    let goals = canonicalize_goals(&sys.goals, labelling.theta());

    CanonicalSystem {
        graph_part,
        formulas,
        solved_formulas,
        lemmas,
        eq_store,
        subterm_store,
        goals,
        source_kind: sys.source_kind,
        side: sys.side,
    }
}

/// Per-field wall-time and size breakdown of Stage G
/// ([`canonicalize_system_content_seeded_profiled`]) -- everything past
/// the graph part: formulas/solved_formulas/lemmas, `eq_store.subst` vs
/// `eq_store.conj` (split out separately since `conj` alone runs
/// [`canonicalize_eq_disj_alternative`]'s per-alternative
/// `CanonLabelling` fork-and-discard machinery, the one part of Stage G
/// with a plausibly different cost profile from a flat per-item map),
/// and `subterm_store`.
///
/// Exists because `canonicalize_system_content_seeded` itself isn't
/// public (this crate's canonicalization internals are not meant to be
/// a stable external API), so a caller outside `canon.rs` -- e.g.
/// `explore_canonical_matches.rs`'s `PROFILE_CANON=1` -- has no other way
/// to see where Stage G's time goes; `canonicalize_constraint_system`'s
/// own timing (measured externally) only accounts for it by
/// SUBTRACTION, as one opaque lump.
#[derive(Debug, Default, Clone, Copy)]
pub struct ContentStageTimes {
    pub formulas: std::time::Duration,
    pub solved_formulas: std::time::Duration,
    pub lemmas: std::time::Duration,
    pub eq_store_subst: std::time::Duration,
    pub eq_store_conj: std::time::Duration,
    pub subterm_store: std::time::Duration,
    pub goals: std::time::Duration,
    pub num_formulas: usize,
    pub num_solved_formulas: usize,
    pub num_lemmas: usize,
    pub num_eq_store_subst: usize,
    /// Number of `EqDisj` entries in `eq_store.conj` -- NOT the number of
    /// alternatives (see `num_eq_store_conj_alternatives`): one `EqDisj`
    /// can hold many alternatives (`sigma_i1 ∨ … ∨ sigma_ik_i`), each of
    /// which pays its own `canonicalize_eq_disj_alternative` fork cost.
    pub num_eq_store_conj: usize,
    /// Total alternatives summed across every `EqDisj` in `eq_store.conj`
    /// -- the real multiplier on `canonicalize_eq_disj_alternative`
    /// calls (and so on `eq_store_conj`'s own time), not
    /// `num_eq_store_conj` alone.
    pub num_eq_store_conj_alternatives: usize,
    pub num_subterms: usize,
    pub num_solved_subterms: usize,
    pub num_neg_subterms: usize,
    /// `sys.goals.len()` BEFORE dropping `Action`/`Split` -- the raw
    /// count, not `CanonicalSystem::goals.len()` (which excludes both).
    pub num_goals: usize,
}

impl ContentStageTimes {
    /// Adds `other`'s durations only. The `num_*` counts describe the
    /// `System` itself, so they are identical across one node's survivors
    /// and must not be summed there. Destructured without `..` so a new
    /// field forces a decision here.
    pub fn add_durations(&mut self, other: &Self) {
        let ContentStageTimes {
            formulas,
            solved_formulas,
            lemmas,
            eq_store_subst,
            eq_store_conj,
            subterm_store,
            goals,
            num_formulas: _,
            num_solved_formulas: _,
            num_lemmas: _,
            num_eq_store_subst: _,
            num_eq_store_conj: _,
            num_eq_store_conj_alternatives: _,
            num_subterms: _,
            num_solved_subterms: _,
            num_neg_subterms: _,
            num_goals: _,
        } = *other;
        self.formulas += formulas;
        self.solved_formulas += solved_formulas;
        self.lemmas += lemmas;
        self.eq_store_subst += eq_store_subst;
        self.eq_store_conj += eq_store_conj;
        self.subterm_store += subterm_store;
        self.goals += goals;
    }

    /// The summed duration of every stage.
    pub fn total_duration(&self) -> std::time::Duration {
        self.formulas
            + self.solved_formulas
            + self.lemmas
            + self.eq_store_subst
            + self.eq_store_conj
            + self.subterm_store
            + self.goals
    }
}

/// Sums durations AND counts -- for aggregating across different nodes.
impl std::ops::AddAssign<&ContentStageTimes> for ContentStageTimes {
    fn add_assign(&mut self, other: &ContentStageTimes) {
        self.add_durations(other);
        self.num_formulas += other.num_formulas;
        self.num_solved_formulas += other.num_solved_formulas;
        self.num_lemmas += other.num_lemmas;
        self.num_eq_store_subst += other.num_eq_store_subst;
        self.num_eq_store_conj += other.num_eq_store_conj;
        self.num_eq_store_conj_alternatives += other.num_eq_store_conj_alternatives;
        self.num_subterms += other.num_subterms;
        self.num_solved_subterms += other.num_solved_subterms;
        self.num_neg_subterms += other.num_neg_subterms;
        self.num_goals += other.num_goals;
    }
}

/// The [`canonicalize_system_content_seeded`] this function's own doc
/// comment describes, reimplemented here (same private helpers this
/// module already has, just called directly instead of through the one
/// monolithic function) with a [`std::time::Instant`] wrapped around each
/// stage and size counters recorded alongside -- see
/// [`ContentStageTimes`] for exactly what each field means.
///
/// `canonicalize_constraint_system`'s own hot path keeps calling the
/// plain, non-instrumented `canonicalize_system_content_seeded` --
/// nothing about its behavior or performance changes. This is a
/// deliberately separate, `pub` sibling for profiling callers only.
pub fn canonicalize_system_content_seeded_profiled(
    sys: &System,
    labelling: &CanonLabelling,
    graph_part: LNTerm,
) -> (CanonicalSystem, ContentStageTimes) {
    let mut stats = ContentStageTimes {
        num_formulas: sys.formulas.len(),
        num_solved_formulas: sys.solved_formulas.len(),
        num_lemmas: sys.lemmas.len(),
        num_eq_store_subst: sys.eq_store.subst.len(),
        num_eq_store_conj: sys.eq_store.conj.len(),
        num_eq_store_conj_alternatives: sys.eq_store.conj.iter().map(|d| d.substs.len()).sum(),
        num_subterms: sys.subterm_store.subterms.len(),
        num_solved_subterms: sys.subterm_store.solved_subterms.len(),
        num_neg_subterms: sys.subterm_store.neg_subterms.len(),
        num_goals: sys.goals.len(),
        ..Default::default()
    };

    let t = std::time::Instant::now();
    let formulas = sort_guarded(
        sys.formulas
            .iter()
            .map(|f| canonicalize_guarded(f, labelling.theta()))
            .collect(),
    );
    stats.formulas = t.elapsed();

    let t = std::time::Instant::now();
    let solved_formulas = sort_guarded(
        sys.solved_formulas
            .iter()
            .map(|f| canonicalize_guarded(f, labelling.theta()))
            .collect(),
    );
    stats.solved_formulas = t.elapsed();

    let t = std::time::Instant::now();
    let lemmas = sort_guarded(
        sys.lemmas
            .iter()
            .map(|f| canonicalize_guarded(f, labelling.theta()))
            .collect(),
    );
    stats.lemmas = t.elapsed();

    // `eq_store.subst` and `eq_store.conj`, timed SEPARATELY -- mirrors
    // `canonicalize_eq_store`'s own body exactly, just split so the
    // per-alternative fork cost in `conj` doesn't hide inside a single
    // `eq_store` bucket.
    let t = std::time::Instant::now();
    let mut subst: Vec<(LNLit, LNTerm)> = sys
        .eq_store
        .subst
        .iter()
        .map(|(v, term)| {
            let canon_key = *lookup_theta(labelling.theta(), &Lit::Var(*v));
            let canon_term = apply_literal_renaming(term, labelling.theta());
            (canon_key, canon_term)
        })
        .collect();
    subst.sort();
    stats.eq_store_subst = t.elapsed();

    let t = std::time::Instant::now();
    let mut conj: Vec<Vec<Vec<(LNLit, LNTerm)>>> = sys
        .eq_store
        .conj
        .iter()
        .map(|disj| canonicalize_eq_disj(disj, labelling))
        .collect();
    conj.sort();
    stats.eq_store_conj = t.elapsed();

    let eq_store = CanonicalEqStore { subst, conj };

    let t = std::time::Instant::now();
    let subterm_store = canonicalize_subterm_store(&sys.subterm_store, labelling.theta());
    stats.subterm_store = t.elapsed();

    let t = std::time::Instant::now();
    let goals = canonicalize_goals(&sys.goals, labelling.theta());
    stats.goals = t.elapsed();

    let canon = CanonicalSystem {
        graph_part,
        formulas,
        solved_formulas,
        lemmas,
        eq_store,
        subterm_store,
        goals,
        source_kind: sys.source_kind,
        side: sys.side,
    };
    (canon, stats)
}

/// Canonicalizes `store` -- `eq_store.subst` and `eq_store.conj`.
fn canonicalize_eq_store(store: &EquationStore, labelling: &CanonLabelling) -> CanonicalEqStore {
    // eq_store.subst: domain vars are real, graph-reachable variables --
    // same guardedness-style assumption as formulas, so a missing theta
    // entry PANICS rather than silently passing the raw var through, on
    // BOTH halves: `lookup_theta` for the domain key, and
    // `apply_literal_renaming` for the range term (which enforces the
    // identical assumption the identical way -- see its own doc comment).
    let mut subst: Vec<(LNLit, LNTerm)> = store
        .subst
        .iter()
        .map(|(v, t)| {
            let canon_key = *lookup_theta(labelling.theta(), &Lit::Var(*v));
            let canon_term = apply_literal_renaming(t, labelling.theta());
            (canon_key, canon_term)
        })
        .collect();
    subst.sort();

    // eq_store.conj: `split_id` DROPPED (see this file's struct docs
    // above). Each `EqDisj`'s alternatives are a MULTISET (see
    // `CanonicalEqStore::conj`'s own doc comment for why duplicates must
    // survive), sorted once canonical; likewise the outer list of
    // `EqDisj`s.
    let mut conj: Vec<Vec<Vec<(LNLit, LNTerm)>>> = store
        .conj
        .iter()
        .map(|disj| canonicalize_eq_disj(disj, labelling))
        .collect();
    conj.sort();

    CanonicalEqStore { subst, conj }
}

/// Canonicalizes one `EqDisj`'s alternatives, preserving multiplicity (no
/// alpha-dedup -- see `CanonicalEqStore::conj`'s own doc comment) and
/// sorting by the now-canonical content.
fn canonicalize_eq_disj(disj: &EqDisj, labelling: &CanonLabelling) -> Vec<Vec<(LNLit, LNTerm)>> {
    let mut alternatives: Vec<Vec<(LNLit, LNTerm)>> = disj
        .substs
        .iter()
        .map(|alt| canonicalize_eq_disj_alternative(alt, labelling))
        .collect();
    alternatives.sort();
    alternatives
}

/// Canonicalizes ONE alternative substitution (one `sigma_ij` in
/// EquationStore.hs's `sigma_i1 ∨ … ∨ sigma_ik_i` notation) of an
/// `EqDisj`.
///
/// **Domain keys** are real, graph-reachable variables (the `x_i` in
/// EquationStore.hs's own semantics) -- canonicalized via the shared
/// `theta` (panicking on a miss, same guardedness-style assumption as
/// everywhere else -- see [`lookup_theta`]) and sorted by that CANONICAL
/// identity, not raw `LVar` Ord, so the range terms below get visited
/// (and their local witnesses numbered) in a content-driven,
/// cross-system-stable order rather than one that depends on incidental
/// proof-search allocation order.
///
/// **Range terms** are entirely, uniformly LOCAL/fresh: per `SubstVFresh`'s
/// own contract, every variable in a range term is existentially bound to
/// THIS ONE alternative, regardless of whether its raw identity happens
/// to look like a real system variable's -- a routine case, not a corner
/// case: Maude unifies actual terms containing actual system variables,
/// and `freshen_witness_range` leaves a variable that legitimately
/// belongs to the original input equations untouched. So:
///
/// 1. **Freshen the WHOLE alternative's range in one call**
///    ([`LNSubstVFresh::fresh_to_free_avoiding`] -- shares one rename
///    cache across every entry, so a variable repeated across two range
///    terms gets the SAME fresh identity both times, and leaves domain
///    keys untouched), with an allocator that avoids every raw variable
///    index already claimed by `labelling.theta()`
///    ([`raw_var_idx_avoid_floor`]).
/// 2. **Canonize each freshened range term against a FRESH FORK of
///    `labelling`** (`labelling.clone()`, mutated across this one
///    alternative's own entries so a witness shared between two of its
///    own range terms still gets tied together -- see
///    `eq_disj_alternative_shares_a_witness_across_two_of_its_own_range_terms`),
///    then DISCARD the fork. Never mutate the real, continuing
///    `labelling` itself.
///
/// Step 2's fork is NOT optional, unlike an earlier revision of this
/// function assumed. `EqDisj`'s alternatives (`sigma_i1 ∨ … ∨ sigma_ik_i`)
/// are each other's INDEPENDENT existential scopes -- both within one
/// `EqDisj` and across different ones -- so two alternatives with the
/// EXACT SAME shape, differing only in raw witness identity (mirroring
/// Maude's arbitrary per-call numbering), must canonize to the IDENTICAL
/// result regardless of which one happens to be processed first. Canonizing
/// against a SHARED, advancing `labelling` breaks that: whichever
/// alternative runs first claims the lower canonical indices, and an
/// otherwise-identical sibling processed after it is forced to discover
/// its own local witnesses starting from wherever the first one left the
/// counters, landing on DIFFERENT canonical names for the same shape --
/// caught directly by
/// `eq_disj_preserves_multiplicity_of_alpha_equivalent_alternatives`
/// (which failed under that earlier revision). Forking wholesale and
/// discarding afterward sidesteps this entirely: every alternative always
/// starts from the exact same baseline `labelling`, so processing order
/// can never leak into the result. This is safe to discard because
/// nothing downstream of `eq_store.conj` ever discovers a new literal
/// (`subterm_store` is the only field processed after it, and it's
/// read-only against `theta`, same as every other field before
/// `eq_store.conj` -- see `canonicalize_system_content_seeded`'s own doc
/// comment), so there is no legitimate consumer for an advanced counter
/// to be preserved for in the first place.
fn canonicalize_eq_disj_alternative(
    alt: &LNSubstVFresh,
    labelling: &CanonLabelling,
) -> Vec<(LNLit, LNTerm)> {
    let mut entries: Vec<(LNLit, LVar)> = alt
        .iter()
        .map(|(v, _)| (*lookup_theta(labelling.theta(), &Lit::Var(*v)), *v))
        .collect();
    // `entries` is created from the alternative's domain keys; i.e., a BTreeMap's keys.
    // Thus, each key is unique. This is important for the injectivity assertion below.
    entries.sort_by_key(|(canon, _)| *canon);

    // INVARIANT: `theta` is injective -- a core `Canonizer` guarantee, not
    // an incidental implementation detail. Every newly-discovered literal
    // gets a BRAND NEW canonical index (`Canonizer::canonize_literals`
    // zips a batch's distinct original literals 1:1 against indices drawn
    // from a monotonic, never-reset, never-reused counter; a literal
    // already in `theta` is excluded from "uncanonicalized" before that
    // even runs, so it's never reprocessed), so two DISTINCT domain keys
    // of one alternative can never canonicalize to the SAME `LNLit`. If
    // they somehow did (e.g. a `theta` hand-built via `CanonLabelling::
    // from_theta` with a bug, bypassing the Canonizer's own accumulation
    // entirely), the stable sort above would fall back to `alt`'s raw,
    // incidental `BTreeMap` order to break the tie -- silently making
    // which of their two (possibly different) range terms is associated
    // with which output position depend on raw `LVar` identity rather
    // than content, exactly the class of bug this whole design exists to
    // rule out. Fail loudly instead of ever risking that silently.
    for pair in entries.windows(2) {
        assert_ne!(
            pair[0].0, pair[1].0,
            "canonicalize_eq_disj_alternative: two distinct domain keys \
             canonicalized to the SAME literal ({:?}) -- theta is no \
             longer injective, which should be impossible; the ordering \
             between their (potentially different) range terms would be \
             undefined",
            pair[0].0
        );
    }

    if entries.is_empty() {
        // No domain keys, so no range terms either -- nothing to freshen
        // or canonize.
        return Vec::new();
    }

    let mut next_idx = raw_var_idx_avoid_floor(labelling.theta());
    let freshened: LNSubst = alt.fresh_to_free_avoiding(|n| {
        let start = next_idx;
        next_idx += n;
        start
    });

    let mut scratch = labelling.clone();
    entries
        .into_iter()
        .map(|(canon_key, v)| {
            // `fresh_to_free_avoiding`'s output drops a trivial `v -> v`
            // mapping (`Subst::from_list`'s own trivial-drop) -- which
            // can only happen if `v` had no occurrences anywhere in the
            // alternative's range terms at all (freshening never
            // reassigns a variable to its own original identity, since
            // the allocator only ever hands out indices strictly above
            // everything currently in scope). Either way, `v` itself
            // (unchanged) is the correct range term to canonize.
            let range_term = freshened
                .image_of(&v)
                .cloned()
                .unwrap_or_else(|| var_term(v));
            let canon_term = canonicalize_alpha_eq_ac_seeded(&range_term, &mut scratch);
            (canon_key, canon_term)
        })
        .collect()
}

/// The smallest raw variable index guaranteed not to collide with any
/// `Lit::Var` key already in `theta` -- the floor
/// [`canonicalize_eq_disj_alternative`]'s range-freshening allocator must
/// start from so a freshly-minted witness identity can never accidentally
/// match an existing (and therefore already-resolved) literal. Only
/// `theta`'s KEYS matter here (what a lookup is keyed by, and the raw
/// literals originally seen -- proof-search-allocated indices, typically
/// large and monotonically increasing over a run), not its VALUES (the
/// canonical output literals, a completely separate, small,
/// sequentially-counted namespace that a lookup is never keyed by) or
/// `Con` keys (name constants -- freshening only ever touches `Var`
/// occurrences, never `Con` ones, so a name constant can never collide
/// with a freshened identity regardless).
fn raw_var_idx_avoid_floor(theta: &std::collections::BTreeMap<LNLit, LNLit>) -> u64 {
    theta
        .keys()
        .filter_map(|lit| match lit {
            Lit::Var(v) => Some(v.idx),
            Lit::Con(_) => None,
        })
        .max()
        .map_or(0, |m| m + 1)
}

/// Canonicalizes `store` (the `subterm_store`): drops
/// `propagated`/`old_neg_subterms` (Rust-only bookkeeping, not
/// semantically meaningful content), renames every term pair via `theta`
/// (reusing `apply_literal_renaming`, same as `eq_store.subst`'s range), and
/// re-sorts everything -- `neg_subterms` is already a sorted, deduplicated
/// set pre-canonicalization, but renaming can change relative order and
/// even collapse two distinct pairs into one, so the existing sort/dedup
/// can't be trusted to survive it unchanged.
fn canonicalize_subterm_store(
    store: &SubtermStore,
    theta: &std::collections::BTreeMap<LNLit, LNLit>,
) -> CanonicalSubtermStore {
    let canon_pair = |c: &SubtermConstraint| {
        (
            apply_literal_renaming(&c.small, theta),
            apply_literal_renaming(&c.big, theta),
        )
    };

    let mut subterms: Vec<(LNTerm, LNTerm)> = store.subterms.iter().map(canon_pair).collect();
    subterms.sort();
    let mut solved_subterms: Vec<(LNTerm, LNTerm)> =
        store.solved_subterms.iter().map(canon_pair).collect();
    solved_subterms.sort();
    let mut neg_subterms: Vec<(LNTerm, LNTerm)> = store
        .neg_subterms
        .iter()
        .map(|(a, b)| {
            (
                apply_literal_renaming(a, theta),
                apply_literal_renaming(b, theta),
            )
        })
        .collect();
    neg_subterms.sort();
    neg_subterms.dedup();

    CanonicalSubtermStore {
        subterms,
        solved_subterms,
        contradictory: store.contradictory,
        neg_subterms,
    }
}

/// Canonicalizes `sys.goals` into [`CanonicalGoal`]s -- read-only against
/// `theta`, same guardedness-style assumption as every other Stage G
/// field (a missing entry panics via [`lookup_theta`]/
/// [`apply_literal_renaming`], not a silent fallback).
///
/// `Goal::Action` is kept even though `canon_graph::extract_graph_part`
/// also turns it into a `VertexKind::Action` vertex: the vertex (deduped
/// against formula-derived actions) only says the action exists. Whether
/// it is also a goal, and whether that goal is solved, is recorded only
/// here -- two systems differing in either offer different proof methods.
/// `Goal::Split` is skipped: it only NAMES an `EqDisj` already in
/// `eq_store.conj` (see [`CanonicalSystem::goals`]'s own doc comment).
///
/// Every kept variant keeps [`GoalStatus::solved`] (content-relevant
/// -- see [`CanonicalGoal`]'s own doc comment) but drop `::looping`/`::nr`
/// (pure heuristic-ranking / path-dependent bookkeeping, matching
/// `next_goal_nr`'s own exclusion from the canonical form). Sorted, but
/// **NOT deduped** (REVISED 2026-09-23, per direct user instruction --
/// same reasoning as [`sort_guarded`]'s own doc comment): although
/// `add_goal_with_loop_flag`'s own insertion-time dedup (`canonical_goal_for_dedup`,
/// `constraint/system.rs`) makes `sys.goals` map-like under RAW `Goal`
/// equality, that's raw identity, not alpha-equivalence -- two goals that
/// are raw-distinct (e.g. a `Disj` over different free-variable
/// identities) but canonicalize to the SAME `CanonicalGoal` are exactly
/// the kind of real, solver-tracked-as-distinct entries a post-rename
/// dedup here would silently collapse.
fn canonicalize_goals(
    goals: &[(Goal, GoalStatus)],
    theta: &std::collections::BTreeMap<LNLit, LNLit>,
) -> Vec<CanonicalGoal> {
    let mut out: Vec<CanonicalGoal> = goals
        .iter()
        .filter_map(|(goal, status)| {
            canonicalize_goal_kind(goal, theta).map(|kind| CanonicalGoal {
                kind,
                solved: status.solved,
            })
        })
        .collect();
    out.sort_by(cmp_canonical_goal);
    out
}

/// One goal's canonical payload via `theta` (panicking on an uncovered
/// literal, like every other Stage G field) -- `None` for `Goal::Split`,
/// whose raw id the canonical form drops (see [`canonicalize_goals`]).
pub fn canonicalize_goal_kind(
    goal: &Goal,
    theta: &std::collections::BTreeMap<LNLit, LNLit>,
) -> Option<CanonicalGoalKind> {
    Some(match goal {
        Goal::Split(_) => return None,
        Goal::Action(nid, fact) => CanonicalGoalKind::Action(
            *lookup_theta(theta, &Lit::Var(*nid)),
            apply_literal_renaming(&fact_to_term(fact), theta),
        ),
        Goal::Chain((conc_nid, conc_idx), (prem_nid, prem_idx)) => CanonicalGoalKind::Chain(
            *lookup_theta(theta, &Lit::Var(*conc_nid)),
            *conc_idx,
            *lookup_theta(theta, &Lit::Var(*prem_nid)),
            *prem_idx,
        ),
        Goal::Premise((prem_nid, prem_idx), fact) => CanonicalGoalKind::Premise(
            *lookup_theta(theta, &Lit::Var(*prem_nid)),
            *prem_idx,
            apply_literal_renaming(&fact_to_term(fact), theta),
        ),
        Goal::Disj(disj) => CanonicalGoalKind::Disj(sort_guarded(
            disj.0.iter().map(|g| canonicalize_guarded(g, theta)).collect(),
        )),
        Goal::Subterm((small, big)) => CanonicalGoalKind::Subterm(
            apply_literal_renaming(small, theta),
            apply_literal_renaming(big, theta),
        ),
    })
}

/// A [`ProofMethod`] renamed through a system's canonical labelling (see
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
    Solve(CanonicalGoalKind),
    /// `solve` of a case split, identified by the canonicalized alternatives
    /// of the `EqDisj` it names -- the canonical form drops split ids.
    SolveSplit(Vec<Vec<(LNLit, LNTerm)>>),
}

/// Renames `method` (offered for `sys`) through `labelling`, `sys`'s
/// canonical labelling from [`canonicalize_constraint_system_with_labelling`].
/// Panics on a literal `labelling` doesn't cover, on a split goal naming no
/// `EqDisj` in `sys.eq_store.conj`, and on the display-only
/// `Invalidated`/`RawSolve`, which `candidate_methods` never produces.
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
            let disj = sys
                .eq_store
                .conj
                .iter()
                .find(|d| d.split_id == *id)
                .unwrap_or_else(|| panic!("split goal {id:?} names no EqDisj in eq_store.conj"));
            CanonicalProofMethod::SolveSplit(canonicalize_eq_disj(disj, labelling))
        }
        ProofMethod::SolveGoal(goal) => CanonicalProofMethod::Solve(
            canonicalize_goal_kind(goal, labelling.theta())
                .expect("only Goal::Split has no goal kind, handled above"),
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
    use tamarin_term::alpha_eq_ac::fingerprint_term;
    use tamarin_term::function_symbols::pair_sym;
    use tamarin_term::lterm::{LSort, LVar};
    use tamarin_term::term::{f_app_no_eq, Term};
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
        let twice = canonicalize_alpha_eq_ac(&once);
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

    // -- Accumulating (seeded) canonization -------------------------------
    //
    // `canonicalize_fact_seeded`/`canonicalize_rule_seeded` are what a
    // constraint-system-wide vertex loop (canonizing every
    // `RuleInstance`/`Action` graph vertex in canonical order, per
    // `canon_graph::VertexKind`) would actually call, one per vertex,
    // threading ONE `CanonLabelling` through all of them -- `VertexKind::Action`
    // holds a plain `LNFact` (see its own doc comment for why: bridging a
    // formula-derived action's `GFact` now happens once, at graph
    // EXTRACTION time, in `canon_graph::action_vertex_fact` -- see that
    // module's own tests for the bridge itself), so an action vertex's
    // fact canonizes through this EXACT SAME call as a rule instance's
    // own facts, no separate combinator needed. This test pins the
    // property that machinery depends on: a variable shared between
    // DIFFERENT `canonicalize_fact_seeded`/`canonicalize_rule_seeded`
    // calls canonizes to the SAME literal in both, regardless of what the
    // shared variable happened to be named originally -- the graph-part
    // level counterpart (`graph_part_shares_a_variable_between_a_rule_instance_and_an_action_vertex`,
    // below) confirms this holds for a REAL `VertexKind::Action`/`RuleInstance`
    // pair specifically, not just two arbitrary facts.

    #[test]
    fn canonicalize_fact_seeded_and_rule_seeded_agree_regardless_of_the_shared_variables_name() {
        // Pair A and pair B are the SAME shape, differing only in what the
        // fact/rule's shared variable happens to be called.
        let fact_a = proto_fact(Multiplicity::Linear, "P", vec![v("shared", LSort::Msg)]);
        let rule_a = rule(vec![in_fact(v("shared", LSort::Msg))], vec![], vec![]);
        let fact_b = proto_fact(Multiplicity::Linear, "P", vec![v("s", LSort::Msg)]);
        let rule_b = rule(vec![in_fact(v("s", LSort::Msg))], vec![], vec![]);

        let mut labelling_a = CanonLabelling::empty();
        let fact_term_a = canonicalize_fact_seeded(&fact_a, &mut labelling_a);
        let rule_term_a = canonicalize_rule_seeded(&rule_a, &mut labelling_a);

        let mut labelling_b = CanonLabelling::empty();
        let fact_term_b = canonicalize_fact_seeded(&fact_b, &mut labelling_b);
        let rule_term_b = canonicalize_rule_seeded(&rule_b, &mut labelling_b);

        assert_eq!(fact_term_a, fact_term_b);
        assert_eq!(rule_term_a, rule_term_b);
    }

    // -- Graph parts (`graph_part_to_term`/`canonicalize_graph_part`) --

    fn rule_ac_inst(
        name: &'static str,
        premises: Vec<LNFact>,
        actions: Vec<LNFact>,
        conclusions: Vec<LNFact>,
    ) -> crate::rule::RuleACInst {
        use crate::rule::{ProtoRuleACInstInfo, ProtoRuleName, RuleAttributes, RuleInfo};
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

    fn node(idx: u64) -> crate::constraint::constraints::NodeId {
        LVar::new("i", LSort::Node, idx)
    }

    #[test]
    fn graph_part_of_rule_instances_matches_a_hand_built_term() {
        let r1 = rule_ac_inst("A", vec![in_fact(v("x", LSort::Msg))], vec![], vec![]);
        let r2 = rule_ac_inst("B", vec![], vec![], vec![out_fact(v("y", LSort::Msg))]);
        let vertices = [
            VertexKind::RuleInstance(node(0), r1.clone()),
            VertexKind::RuleInstance(node(1), r2.clone()),
        ];
        let ordered: Vec<&VertexKind> = vertices.iter().collect();
        let edges = BTreeSet::from([(0usize, 1usize)]);

        let expected_vertices = graph_tag(
            "VerticesTerm",
            vec![
                graph_tag("RuleInstanceVertexTerm", vec![node_id_term(node(0)), rule_to_term(&r1)]),
                graph_tag("RuleInstanceVertexTerm", vec![node_id_term(node(1)), rule_to_term(&r2)]),
            ],
        );
        let expected_edges = graph_tag(
            "EdgesTerm",
            vec![graph_tag("EdgeTerm", vec![index_marker(0), index_marker(1)])],
        );
        let expected = graph_tag("GraphPartTerm", vec![expected_vertices, expected_edges]);
        assert_eq!(graph_part_to_term(&ordered, &edges), expected);
    }

    /// Direct test of the fix's mechanism: a vertex's own `NodeId` must
    /// now be covered by `theta` after graph-part canonization -- before
    /// `vertex_to_term` embedded it as a literal, `theta` never had an
    /// entry for ANY `NodeId`, regardless of how well-connected its
    /// vertex was (see `vertex_to_term`'s own doc comment for the full
    /// story).
    #[test]
    fn canonicalize_graph_part_seeded_covers_each_vertexs_own_node_id() {
        let r1 = rule_ac_inst("A", vec![], vec![], vec![]);
        let vertices = [VertexKind::RuleInstance(node(0), r1)];
        let ordered: Vec<&VertexKind> = vertices.iter().collect();
        let edges = BTreeSet::new();

        let (_, labelling) = canonicalize_graph_part_seeded(&ordered, &edges);

        let key = Lit::Var(node(0));
        assert!(
            labelling.theta().contains_key(&key),
            "expected theta to cover the vertex's own NodeId, got {:?}",
            labelling.theta()
        );
    }

    /// Regression test for the exact bug this fix closes:
    /// `guarded::to_induction_hypothesis` (the `ginduct`/`Induction` proof
    /// method) can produce a formula referencing a `NodeId` FREE -- e.g.
    /// `last(#i)`, outside any quantifier -- which used to panic in
    /// `canonicalize_guarded` because `theta` never had an entry for any
    /// `NodeId` at all. Confirmed directly against a real captured proof
    /// state too (`examples/explore_canonical_matches.rs` against
    /// `Tutorial.spthy`'s `Client_session_key_secrecy`, `induction
    /// [non_empty_trace]`): panicked before this fix, canonicalizes
    /// cleanly after it.
    #[test]
    fn a_formula_referencing_a_vertexs_node_id_via_last_canonicalizes_without_panicking() {
        let r1 = rule_ac_inst("A", vec![], vec![], vec![]);
        let vertices = [VertexKind::RuleInstance(node(0), r1)];
        let ordered: Vec<&VertexKind> = vertices.iter().collect();
        let edges = BTreeSet::new();

        let (_, labelling) = canonicalize_graph_part_seeded(&ordered, &edges);

        let last_i = Guarded::Atom(GAtom::Last(GTerm::Var(BVar::Free(lvar_to_varspec(&node(
            0,
        ))))));
        let canonicalized = canonicalize_guarded(&last_i, labelling.theta());
        // The canonical form must reference the CANONICAL name, not the
        // raw NodeId unchanged -- confirming it was actually renamed via
        // theta, not silently passed through.
        match canonicalized {
            Guarded::Atom(GAtom::Last(GTerm::Var(BVar::Free(vs)))) => {
                assert_ne!(varspec_to_lvar(&vs), node(0));
            }
            other => panic!("expected a Last atom, got {other:?}"),
        }
    }

    /// THE regression test for the false positive this redesign fixes
    /// (`vertex_sequence_to_term`'s earlier version skipped structural
    /// vertices entirely): two graph parts whose `RuleInstance` vertices
    /// are identical -- same content, same relative order -- but whose
    /// TOTAL vertex count differs (one has an extra structural `Dummy`
    /// the other lacks entirely) must NOT canonize equal. They are not
    /// even the same size, let alone $\alphaeqac$.
    #[test]
    fn graph_part_to_term_distinguishes_different_vertex_counts() {
        let r1 = rule_ac_inst("A", vec![in_fact(v("x", LSort::Msg))], vec![], vec![]);
        let r2 = rule_ac_inst("B", vec![], vec![], vec![out_fact(v("y", LSort::Msg))]);

        let without_dummy = [
            VertexKind::RuleInstance(node(0), r1.clone()),
            VertexKind::RuleInstance(node(1), r2.clone()),
        ];
        let with_dummy = [
            VertexKind::RuleInstance(node(0), r1),
            VertexKind::Dummy(node(2)),
            VertexKind::RuleInstance(node(1), r2),
        ];

        let a: Vec<&VertexKind> = without_dummy.iter().collect();
        let b: Vec<&VertexKind> = with_dummy.iter().collect();
        let no_edges = BTreeSet::new();
        assert_ne!(
            canonicalize_graph_part(&a, &no_edges),
            canonicalize_graph_part(&b, &no_edges),
            "a graph part with an extra structural vertex must not canonize equal to \
             one without it, even if their RuleInstance content is otherwise identical"
        );
    }

    /// Different structural KINDS at the same position must also be
    /// told apart (not just "structural vs not").
    #[test]
    fn graph_part_to_term_distinguishes_structural_vertex_kinds() {
        let dummy = [VertexKind::Dummy(node(0))];
        let less = [VertexKind::LessRelation];
        let a: Vec<&VertexKind> = dummy.iter().collect();
        let b: Vec<&VertexKind> = less.iter().collect();
        let no_edges = BTreeSet::new();
        assert_ne!(
            graph_part_to_term(&a, &no_edges),
            graph_part_to_term(&b, &no_edges)
        );
    }

    /// Two graph parts with the IDENTICAL vertex sequence but DIFFERENT
    /// edges must not canonize equal -- edges are now part of the
    /// encoding, not left to a separate, unrelated shape check.
    #[test]
    fn graph_part_to_term_distinguishes_different_edge_sets() {
        let r1 = rule_ac_inst("A", vec![], vec![], vec![]);
        let r2 = rule_ac_inst("B", vec![], vec![], vec![]);
        let vertices = [
            VertexKind::RuleInstance(node(0), r1),
            VertexKind::RuleInstance(node(1), r2),
        ];
        let ordered: Vec<&VertexKind> = vertices.iter().collect();

        let edge_0_to_1 = BTreeSet::from([(0usize, 1usize)]);
        let edge_1_to_0 = BTreeSet::from([(1usize, 0usize)]);
        assert_ne!(
            graph_part_to_term(&ordered, &edge_0_to_1),
            graph_part_to_term(&ordered, &edge_1_to_0)
        );
    }

    /// `List` is not AC, so vertex ORDER is significant: two sequences
    /// with the same vertices in a DIFFERENT order must not canonize
    /// equal (unless that reordering happens to itself be an
    /// automorphism of the underlying content, which this example is
    /// deliberately built to avoid).
    #[test]
    fn graph_part_order_is_significant() {
        let r1 = rule_ac_inst("A", vec![in_fact(v("x", LSort::Msg))], vec![], vec![]);
        let r2 = rule_ac_inst("B", vec![], vec![], vec![out_fact(v("y", LSort::Msg))]);

        let forward = [
            VertexKind::RuleInstance(node(0), r1.clone()),
            VertexKind::RuleInstance(node(1), r2.clone()),
        ];
        let backward = [
            VertexKind::RuleInstance(node(1), r2),
            VertexKind::RuleInstance(node(0), r1),
        ];

        let a: Vec<&VertexKind> = forward.iter().collect();
        let b: Vec<&VertexKind> = backward.iter().collect();
        let no_edges = BTreeSet::new();
        assert_ne!(
            canonicalize_graph_part(&a, &no_edges),
            canonicalize_graph_part(&b, &no_edges)
        );
    }

    /// The end-to-end property this whole design exists for, expressed
    /// directly at the graph-part level (the bliss-level counterpart
    /// lives in `bliss_tutorial_alphaeqac.rs`): a `RuleInstance` vertex
    /// and an `Action` vertex sharing a variable canonize that variable
    /// to the SAME literal, automatically -- ONE `Canonizer` pass over
    /// the whole nested term treats a shared literal VALUE as one entry
    /// no matter which sub-term it occurs in, so no explicit accumulator
    /// threading is needed here (contrast the `_seeded` functions above,
    /// which exist for the case where each vertex must be canonized in
    /// its OWN separate call).
    #[test]
    fn graph_part_shares_a_variable_between_a_rule_instance_and_an_action_vertex() {
        let rule_a = rule_ac_inst("R", vec![in_fact(v("shared", LSort::Msg))], vec![], vec![]);
        let action_a = proto_fact(Multiplicity::Linear, "Foo", vec![v("shared", LSort::Msg)]);
        let seq_a = [
            VertexKind::RuleInstance(node(0), rule_a),
            VertexKind::Action(node(0), action_a),
        ];

        let rule_b = rule_ac_inst("R", vec![in_fact(v("s", LSort::Msg))], vec![], vec![]);
        let action_b = proto_fact(Multiplicity::Linear, "Foo", vec![v("s", LSort::Msg)]);
        let seq_b = [
            VertexKind::RuleInstance(node(0), rule_b),
            VertexKind::Action(node(0), action_b),
        ];

        let a: Vec<&VertexKind> = seq_a.iter().collect();
        let b: Vec<&VertexKind> = seq_b.iter().collect();
        let no_edges = BTreeSet::new();
        assert_eq!(
            canonicalize_graph_part(&a, &no_edges),
            canonicalize_graph_part(&b, &no_edges),
            "the variable shared between the rule instance and the action vertex must \
             canonize identically regardless of its original name"
        );
    }

    // -- Minimum over automorphisms (`minimal_graph_part_labelings`) --

    use crate::bliss_proc::bliss_available;
    use crate::canon_color::ColorTable;

    fn theory(src: &str) -> crate::theory::Theory {
        let parsed =
            tamarin_parser::parser::parse_theory(src, &[]).unwrap_or_else(|e| panic!("parse: {e}"));
        crate::elaborate::elaborate(&parsed).unwrap_or_else(|e| panic!("elaborate: {e:?}"))
    }

    /// A `ColorTable` built from no protocol rules and an empty
    /// `IntrRuleCache` -- correct (if not exhaustive) for any test whose
    /// `System` has no graph-part vertices to color at all (see the
    /// empty-graph-part tests below), and needs no maude process.
    fn empty_color_table() -> ColorTable {
        ColorTable::build(
            &[],
            &crate::constraint::solver::context::IntrRuleCache::from(Vec::new()),
        )
    }

    /// The `ColorTable` for `thy`'s own declared protocol rules, paired
    /// with an EMPTY `IntrRuleCache` -- correct (if not exhaustive) for
    /// any test below that colors only `RuleInfo::Proto` vertices (every
    /// one of them; none of these hand-built `GraphPart`s uses a
    /// `RuleInfo::Intr` instance), and avoids needing a real maude
    /// process. See `canon_color.rs`'s own test module for the
    /// maude-backed helper real intruder-rule coverage needs.
    fn color_table(thy: &crate::theory::Theory) -> ColorTable {
        let protocol_rules: Vec<crate::theory::OpenProtoRule> = thy.rules().cloned().collect();
        ColorTable::build(
            &protocol_rules,
            &crate::constraint::solver::context::IntrRuleCache::from(Vec::new()),
        )
    }

    /// A ground 0-ary NoEq function symbol term, e.g. for `name = "aaa"`
    /// a term that is never renamed by `CAN_alphaeqac` (unlike a name/var
    /// literal) and orders by NAME -- exactly what's needed to build
    /// content whose relative order is fixed and predictable across
    /// canonization, for testing which automorphism candidate wins.
    fn zero_ary_fun_term(name: &'static str) -> LNTerm {
        let sym = NoEqSym::new(
            name.as_bytes().to_vec(),
            0,
            Privacy::Public,
            Constructability::Constructor,
        );
        LNTerm::App(FunSym::NoEq(sym), Arc::from([]))
    }

    /// A ground-symbol (but not ground-ARGUMENT) 2-ary NoEq function
    /// term, e.g. `binary_fun_term("f", a, b)` = `f(a, b)` -- same
    /// no-signature-needed construction as [`zero_ary_fun_term`], just
    /// with real argument terms and a nonzero arity. `f`'s own NoEq-ness
    /// means it has no equational theory: no AC, no commutativity, so
    /// `f(a, b)` and `f(b, a)` are genuinely different terms whenever
    /// `a != b`.
    fn binary_fun_term(name: &'static str, a: LNTerm, b: LNTerm) -> LNTerm {
        let sym = NoEqSym::new(
            name.as_bytes().to_vec(),
            2,
            Privacy::Public,
            Constructability::Constructor,
        );
        LNTerm::App(FunSym::NoEq(sym), Arc::from([a, b]))
    }

    fn rule_with_conclusion(name: &'static str, concl: LNTerm) -> crate::rule::RuleACInst {
        rule_ac_inst(name, vec![], vec![], vec![out_fact(concl)])
    }

    /// `minimal_graph_part_labelings` on bliss's own $G_1$ shape (4
    /// vertices, vertex 1 distinctly colored, `Aut(G) = {id, (3 4)}` --
    /// see `bliss_proc::tests::bliss_g1_example_has_the_documented_automorphism_group`),
    /// but with REAL rule-instance content at vertices 3/4 that's
    /// asymmetric enough to fully resolve the tie: exactly ONE labeling
    /// must survive, and it must be the one placing the lexicographically
    /// smaller content ahead of the larger one.
    #[test]
    fn minimal_graph_part_labelings_resolves_a_tie_broken_by_content() {
        if !bliss_available() {
            return;
        }
        use crate::bliss_proc::{graph_part_to_dimacs, run_bliss};
        use crate::canon_graph::{GraphEdge, GraphPart};

        let colors = color_table(&theory(
            "theory T begin\n\
             rule Hub:\n  [] --> []\n\
             rule Leaf:\n  [] --> []\n\
             end",
        ));

        let a_fun = zero_ary_fun_term("aaa");
        let z_fun = zero_ary_fun_term("zzz");

        // Vertex 0 = hub (uniquely colored, like G_1's vertex 1); vertices
        // 1/2 = the two symmetric leaves (like G_1's vertices 3/4), one
        // holding the lexicographically SMALLER content, one the LARGER.
        let vertices = vec![
            VertexKind::RuleInstance(node(0), rule_ac_inst("Hub", vec![], vec![], vec![])),
            VertexKind::RuleInstance(node(1), rule_with_conclusion("Leaf", z_fun)),
            VertexKind::RuleInstance(node(2), rule_with_conclusion("Leaf", a_fun)),
        ];
        let edges = vec![GraphEdge { src: 0, tgt: 1 }, GraphEdge { src: 0, tgt: 2 }];
        let part = GraphPart {
            vertices,
            edges,
            colors,
        };

        let dimacs = graph_part_to_dimacs(&part).unwrap_or_else(|e| panic!("dimacs: {e}"));
        let result = run_bliss(&dimacs).unwrap_or_else(|e| panic!("run_bliss: {e}"));
        assert_eq!(
            result.generators.len(),
            1,
            "one hub with two symmetric leaves has the same {{id, swap}} automorphism \
             group as bliss's own G_1 example"
        );

        let survivors = minimal_graph_part_labelings(&part, &result);
        assert_eq!(
            survivors.len(),
            1,
            "the leaves' asymmetric content (aaa vs zzz) must fully resolve the tie \
             the graph SHAPE alone leaves open"
        );

        // The winning term must be strictly smaller than the one and only
        // OTHER candidate (the non-surviving labeling) -- confirms the
        // filter actually discriminated on content, not just accepted
        // whichever candidate came first.
        let group = crate::bliss_proc::generate_group(&result.generators, 3);
        let all_terms: Vec<LNTerm> = group
            .iter()
            .map(|g| {
                let labeling = result.canonical_labeling.compose(g);
                let ordered = crate::bliss_proc::canonical_vertex_order(&part, &labeling);
                let edges = crate::bliss_proc::canonical_edges(&part, &labeling);
                canonicalize_graph_part(&ordered, &edges)
            })
            .collect();
        assert_eq!(
            all_terms.len(),
            2,
            "Aut(G) = {{id, swap}} has exactly 2 elements"
        );
        assert_eq!(&survivors[0].1, all_terms.iter().min().unwrap());
        assert!(
            all_terms.iter().any(|t| *t != survivors[0].1),
            "the two candidates must actually differ, or this test isn't exercising \
             the content-based tie-break at all"
        );
    }

    /// THE regression test for why `minimal_graph_part_labelings` MUST
    /// use `generate_group`'s full closure rather than iterating over
    /// `{id} ∪ result.generators` directly: a graph with automorphism
    /// group `Aut(G) = Z2 × Z2` (two independent, disjoint-support leaf
    /// swaps -- bliss reports 2 generators for it, not 4 elements), built
    /// so that the TRUE minimum is achieved ONLY by applying BOTH swaps
    /// together. "Naive" iteration over just `{id, g1, g2}` never tries
    /// `g1∘g2`, so it converges on a candidate that is NOT the true
    /// minimum.
    ///
    /// Construction: two independent hub+2-leaves components (`HubA`
    /// with `LeafA` children, `HubB` with `LeafB` children, no edges
    /// between the components). Each component's two leaves hold the
    /// SAME pair of contents (`zzz` and `aaa`) as the OTHER component's
    /// leaves, so minimizing EACH component independently requires its
    /// OWN swap decision -- fixing one component's swap does nothing for
    /// the other's, so only the labeling applying BOTH swaps reaches the
    /// lexicographically smallest overall sequence.
    #[test]
    fn naive_generator_only_iteration_misses_the_true_minimum() {
        if !bliss_available() {
            return;
        }
        use crate::bliss_proc::{
            canonical_edges, canonical_vertex_order, generate_group, graph_part_to_dimacs,
            run_bliss,
        };
        use crate::canon_graph::{GraphEdge, GraphPart};

        let colors = color_table(&theory(
            "theory T begin\n\
             rule HubA:\n  [] --> []\n\
             rule HubB:\n  [] --> []\n\
             rule LeafA:\n  [] --> []\n\
             rule LeafB:\n  [] --> []\n\
             end",
        ));

        // NOTE: which physical vertex holds "aaa" vs "zzz" here was chosen
        // empirically (by inspecting bliss's actual reported canonical
        // labeling for this exact graph) so that bliss's OWN default
        // labeling (`g = id`) is sub-optimal for BOTH leaf pairs at once
        // -- otherwise `id` or a single generator could coincidentally
        // already reach the true minimum, and the test would not
        // actually exercise the gap `generate_group` closes. See this
        // test's own doc comment.
        let vertices = vec![
            VertexKind::RuleInstance(node(0), rule_ac_inst("HubA", vec![], vec![], vec![])), // 0
            VertexKind::RuleInstance(
                node(1),
                rule_with_conclusion("LeafA", zero_ary_fun_term("aaa")),
            ), // 1
            VertexKind::RuleInstance(
                node(2),
                rule_with_conclusion("LeafA", zero_ary_fun_term("zzz")),
            ), // 2
            VertexKind::RuleInstance(node(3), rule_ac_inst("HubB", vec![], vec![], vec![])), // 3
            VertexKind::RuleInstance(
                node(4),
                rule_with_conclusion("LeafB", zero_ary_fun_term("aaa")),
            ), // 4
            VertexKind::RuleInstance(
                node(5),
                rule_with_conclusion("LeafB", zero_ary_fun_term("zzz")),
            ), // 5
        ];
        let edges = vec![
            GraphEdge { src: 0, tgt: 1 },
            GraphEdge { src: 0, tgt: 2 },
            GraphEdge { src: 3, tgt: 4 },
            GraphEdge { src: 3, tgt: 5 },
        ];
        let part = GraphPart {
            vertices,
            edges,
            colors,
        };

        let dimacs = graph_part_to_dimacs(&part).unwrap_or_else(|e| panic!("dimacs: {e}"));
        let result = run_bliss(&dimacs).unwrap_or_else(|e| panic!("run_bliss: {e}"));
        assert_eq!(
            result.generators.len(),
            2,
            "two independent leaf-pair swaps -- bliss should report exactly 2 generators"
        );

        let full_group = generate_group(&result.generators, part.vertices.len());
        assert_eq!(
            full_group.len(),
            4,
            "Aut(G) = Z2 x Z2 (two independent swaps) has 4 elements: id, g1, g2, g1*g2 \
             -- strictly more than bliss's own 2 reported generators"
        );

        // "Naive" iteration: only {id} ∪ the raw generators bliss
        // reported -- i.e. exactly what a caller would try if it skipped
        // `generate_group`'s closure step.
        let naive_candidates: Vec<crate::bliss_proc::Permutation> = std::iter::once(
            crate::bliss_proc::Permutation::identity(part.vertices.len()),
        )
        .chain(result.generators.iter().cloned())
        .collect();
        assert_eq!(
            naive_candidates.len(),
            3,
            "naive set: id + 2 raw generators"
        );
        let naive_min = naive_candidates
            .iter()
            .map(|g| {
                let labeling = result.canonical_labeling.compose(g);
                let ordered = canonical_vertex_order(&part, &labeling);
                let edges = canonical_edges(&part, &labeling);
                canonicalize_graph_part(&ordered, &edges)
            })
            .min()
            .unwrap();

        // Correct: minimize over the FULL closed group (what
        // `minimal_graph_part_labelings` actually does).
        let survivors = minimal_graph_part_labelings(&part, &result);
        let true_min = &survivors[0].1;

        assert!(
            *true_min < naive_min,
            "the true minimum (found only by trying g1∘g2, via full group closure) must \
             be strictly smaller than whatever naive {{id}} ∪ raw-generators iteration \
             finds -- otherwise this test isn't actually exercising the gap group \
             closure fixes"
        );
    }

    /// A THIRD `minimal_graph_part_labelings` scenario, distinct from the
    /// two above: here the automorphism itself is genuine (bliss is
    /// right that swapping the two `Fr` vertices preserves the graph's
    /// COLOR/STRUCTURE), but the swap is NOT actually content-preserving
    /// once real vertex CONTENT (not just color) is considered --
    /// because a THIRD vertex, `K(f(x, y))`, references BOTH `Fr`
    /// vertices' own variables ASYMMETRICALLY through `f`, a function
    /// symbol with NO equational theory (no commutativity to make
    /// `f(_, _)`'s two argument orders equal).
    ///
    /// Concretely: `Fr(x)`/`Fr(y)` are colored identically (same action
    /// tag, same arity) and have no edges distinguishing them, so bliss
    /// reports exactly one generator swapping them -- `Aut(G) = {id,
    /// swap}`, size 2. Which one of `{x, y}` gets canonized to the
    /// canonical variable `mv(0)` (vs. `mv(1)`) depends entirely on
    /// vertex ORDER (the Canonizer discovers literals in graph-part
    /// vertex-sequence order -- see `vertex_to_term`'s own doc comment):
    /// under the identity ordering `x` is discovered first (`x -> mv(0)`,
    /// `y -> mv(1)`), giving `K(f(mv(0), mv(1)))`; under the swapped
    /// ordering `y` is discovered first (`y -> mv(0)`, `x -> mv(1)`),
    /// giving `K(f(mv(1), mv(0)))` -- `x` and `y` trade canonical
    /// identities, but `f`'s own two ARGUMENT POSITIONS never move, so
    /// the two candidate terms are genuinely different (not just
    /// differently-labelled). `minimal_graph_part_labelings` must -- and,
    /// per this test, does -- pick the lexicographically smaller of the
    /// two: `K(f(mv(0), mv(1)))` (`mv(0) < mv(1)` as canonical literals),
    /// discarding the swapped ordering's `K(f(mv(1), mv(0)))` even though
    /// it comes from an equally-real graph automorphism.
    #[test]
    fn minimal_graph_part_labelings_discards_a_real_automorphism_whose_content_actually_differs() {
        if !bliss_available() {
            return;
        }
        use crate::bliss_proc::{graph_part_to_dimacs, run_bliss};
        use crate::canon_graph::GraphPart;

        // `VertexKind::Action` holds a plain `LNFact` now, so `f(x, y)`
        // is built directly via `NoEqSym::new` -- same as `zero_ary_fun_term`,
        // just 2-ary -- with NO signature/elaboration needed at all: a
        // plain free/NoEq symbol, no AC, no commutativity, nothing that
        // could make `f(a, b) = f(b, a)`.
        let fxy = binary_fun_term("f", v("x", LSort::Msg), v("y", LSort::Msg));

        // `Fr`/`K` are both fixed `BUILTIN_ACTION_NAMES` entries, colored
        // the same regardless of theory -- no protocol rules/actions of
        // our own to declare, so the empty table already covers them.
        let colors = empty_color_table();

        let vertices = vec![
            VertexKind::Action(
                node(0),
                proto_fact(Multiplicity::Linear, "Fr", vec![v("x", LSort::Msg)]),
            ),
            VertexKind::Action(
                node(1),
                proto_fact(Multiplicity::Linear, "Fr", vec![v("y", LSort::Msg)]),
            ),
            VertexKind::Action(node(2), proto_fact(Multiplicity::Linear, "K", vec![fxy])),
        ];
        let part = GraphPart {
            vertices,
            edges: Vec::new(),
            colors,
        };

        let dimacs = graph_part_to_dimacs(&part).unwrap_or_else(|e| panic!("dimacs: {e}"));
        let result = run_bliss(&dimacs).unwrap_or_else(|e| panic!("run_bliss: {e}"));
        assert_eq!(
            result.generators.len(),
            1,
            "the two identically-colored, edge-free Fr vertices should give bliss exactly \
             one generator (the swap); the K vertex, differently colored, must stay fixed"
        );
        let group = crate::bliss_proc::generate_group(&result.generators, part.vertices.len());
        assert_eq!(
            group.len(),
            2,
            "Aut(G) = {{id, swap}} has exactly 2 elements"
        );

        let survivors = minimal_graph_part_labelings(&part, &result);
        assert_eq!(
            survivors.len(),
            1,
            "f has no equational theory, so f(mv(0), mv(1)) and f(mv(1), mv(0)) are \
             genuinely different terms -- the tie must resolve to a SINGLE survivor, not \
             both (contrast a scenario where the swapped content is ALSO identical, which \
             would leave both as ties)"
        );

        // Pin down exactly which candidate wins, not just that a unique
        // one does: identity order (`Fr(x)` first) must be the survivor,
        // and it must be the STRICTLY smaller of the two candidates.
        let identity_order: Vec<&VertexKind> = part.vertices.iter().collect();
        let swapped_order: Vec<&VertexKind> =
            vec![&part.vertices[1], &part.vertices[0], &part.vertices[2]];
        let no_edges = BTreeSet::new();
        let identity_term = canonicalize_graph_part(&identity_order, &no_edges);
        let swapped_term = canonicalize_graph_part(&swapped_order, &no_edges);

        assert_ne!(
            identity_term, swapped_term,
            "the two candidates must actually differ, or this test isn't exercising the \
             content-based tie-break at all"
        );
        assert!(
            identity_term < swapped_term,
            "canonizing Fr(x) before Fr(y) (giving K(f(mv(0), mv(1)))) must be the \
             LEXICOGRAPHICALLY SMALLER candidate -- the one minimization is expected to keep"
        );
        assert_eq!(
            survivors[0].1, identity_term,
            "minimal_graph_part_labelings must pick the K(f(mv(0), mv(1))) variant -- the \
             one from canonizing Fr(x) before Fr(y) -- discarding the swapped K(f(mv(1), \
             mv(0))) variant even though bliss reported the swap as a real automorphism"
        );
    }

    // -- Stage G: eq_store.conj --------------------------------------------

    /// A `CanonLabelling` whose theta already covers `domain_vars`, each
    /// mapped to its own distinct canonical var of the same sort -- enough
    /// to satisfy `lookup_theta`'s panic-on-miss for a domain-key lookup,
    /// without running a real graph-part/formula canonization pass first.
    /// Uses the SAME per-sort naming as the real scheme (`canonical_var` in
    /// `tamarin-term/src/alpha_eq_ac.rs`: `mv`/`fv`/`pv`/`tv`/`nv` for
    /// msg/fresh/pub/node/nat) rather than one sort-blind `cv` family, so a
    /// reader of these tests sees the same convention `CanonLabelling`
    /// itself produces -- NOT the real `CanonLabelling` scheme otherwise
    /// (this hands out indices purely by `domain_vars`' own order, one
    /// counter per sort, rather than by any real Canonizer discovery
    /// order). Canonical keys are handed out in `domain_vars`' own order
    /// within each sort (index 0, 1, ...), so a test can predict which
    /// output entry belongs to which input domain var.
    fn labelling_covering(domain_vars: &[LVar]) -> CanonLabelling {
        let mut theta: std::collections::BTreeMap<LNLit, LNLit> = std::collections::BTreeMap::new();
        let mut next_idx: std::collections::BTreeMap<LSort, u64> = std::collections::BTreeMap::new();
        for dv in domain_vars {
            let idx = next_idx.entry(dv.sort).or_insert(0);
            let name = match dv.sort {
                LSort::Msg => "mv",
                LSort::Fresh => "fv",
                LSort::Pub => "pv",
                LSort::Node => "tv",
                LSort::Nat => "nv",
            };
            theta.insert(Lit::Var(*dv), Lit::Var(LVar::new(name, dv.sort, *idx)));
            *idx += 1;
        }
        CanonLabelling::from_theta(theta)
    }

    #[test]
    fn eq_disj_alternative_with_no_domain_keys_is_empty() {
        let alt: LNSubstVFresh = LNSubstVFresh::from_list(Vec::<(LVar, LNTerm)>::new());
        let labelling = CanonLabelling::empty();
        assert_eq!(
            canonicalize_eq_disj_alternative(&alt, &labelling),
            Vec::new()
        );
    }

    #[test]
    fn eq_disj_alternative_domain_via_theta_range_discovered_fresh() {
        let x = LVar::new("x", LSort::Msg, 7);
        let w = LVar::new("w", LSort::Msg, 900); // an undiscovered witness
        let alt: LNSubstVFresh = LNSubstVFresh::from_list(vec![(x, var_term(w))]);
        let labelling = labelling_covering(&[x]);

        let out = canonicalize_eq_disj_alternative(&alt, &labelling);

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, Lit::Var(LVar::new("mv", LSort::Msg, 0)));
        // The witness got SOME fresh canonical literal of its own -- not
        // `w` itself (which was never a canonical form to begin with),
        // and not x's own canonical key.
        assert_ne!(out[0].1, var_term(w));
        assert_ne!(out[0].1, LNTerm::Lit(out[0].0));
    }

    /// The core correctness property: two alternatives with the SAME
    /// logical shape but DIFFERENT raw witness identities (mirroring
    /// Maude's arbitrary per-call witness numbering) canonize to the
    /// IDENTICAL result.
    #[test]
    fn eq_disj_alternative_is_witness_numbering_invariant() {
        let x = LVar::new("x", LSort::Msg, 7);
        let alt_a: LNSubstVFresh = LNSubstVFresh::from_list(vec![(
            x,
            f_app_no_eq(
                pair_sym(),
                vec![
                    var_term(LVar::new("w", LSort::Msg, 100)),
                    var_term(LVar::new("w", LSort::Msg, 101)),
                ],
            ),
        )]);
        let alt_b: LNSubstVFresh = LNSubstVFresh::from_list(vec![(
            x,
            f_app_no_eq(
                pair_sym(),
                vec![
                    var_term(LVar::new("z", LSort::Msg, 500)),
                    var_term(LVar::new("z", LSort::Msg, 501)),
                ],
            ),
        )]);

        let labelling_a = labelling_covering(&[x]);
        let labelling_b = labelling_covering(&[x]);

        let out_a = canonicalize_eq_disj_alternative(&alt_a, &labelling_a);
        let out_b = canonicalize_eq_disj_alternative(&alt_b, &labelling_b);

        assert_eq!(out_a, out_b);
    }

    /// A witness appearing in TWO of an alternative's own range terms (not
    /// just twice within one) must canonize to the SAME canonical literal
    /// both times -- confirming the per-entry `canonicalize_alpha_eq_ac_seeded`
    /// calls correctly share identity via the one mutable `labelling`
    /// threaded across them, the same way two different graph vertices
    /// sharing a variable already do (Stage D).
    #[test]
    fn eq_disj_alternative_shares_a_witness_across_two_of_its_own_range_terms() {
        let p = LVar::new("p", LSort::Msg, 7);
        let q = LVar::new("q", LSort::Msg, 8);
        let w = LVar::new("w", LSort::Msg, 900);
        let alt: LNSubstVFresh = LNSubstVFresh::from_list(vec![
            (p, f_app_no_eq(pair_sym(), vec![var_term(w), var_term(w)])),
            (q, var_term(w)),
        ]);
        let labelling = labelling_covering(&[p, q]);

        let out = canonicalize_eq_disj_alternative(&alt, &labelling);

        assert_eq!(out.len(), 2);
        let p_key = Lit::Var(LVar::new("mv", LSort::Msg, 0));
        let q_key = Lit::Var(LVar::new("mv", LSort::Msg, 1));
        let p_range = &out.iter().find(|(k, _)| *k == p_key).expect("p entry").1;
        let q_range = &out.iter().find(|(k, _)| *k == q_key).expect("q entry").1;
        match p_range {
            Term::App(_, args) => {
                assert_eq!(args.len(), 2);
                // Both occurrences of `w` WITHIN p's own pair(..) term
                // share one canonical literal.
                assert_eq!(args[0], args[1]);
                // AND q's lone range term (just `w`) matches that SAME
                // literal -- the shared identity crosses entries, not
                // just occurrences within one.
                assert_eq!(&args[0], q_range);
            }
            other => panic!("expected a pair(..) term, got {other:?}"),
        }
    }

    /// Regression test for the exact collision risk `canonicalize_eq_disj_alternative`'s
    /// own doc comment describes: a range variable whose raw `LVar`
    /// happens to coincide with an UNRELATED real variable already
    /// covered by `theta`. `SubstVFresh`'s "range vars are fresh" is only
    /// an interpretive convention (not a data-level guarantee), so this
    /// raw-identity collision must not cause the witness to be silently
    /// treated as the real variable it happens to share an `LVar` with.
    #[test]
    fn eq_disj_alternative_range_witness_does_not_alias_an_unrelated_real_variable() {
        let x = LVar::new("x", LSort::Msg, 7);
        let y = LVar::new("y", LSort::Msg, 42);
        // `w` deliberately reuses `y`'s exact raw identity.
        let w = y;
        let alt: LNSubstVFresh = LNSubstVFresh::from_list(vec![(x, var_term(w))]);
        let labelling = labelling_covering(&[x, y]);
        let y_canonical = *labelling
            .theta()
            .get(&Lit::Var(y))
            .expect("labelling_covering covers y");

        let out = canonicalize_eq_disj_alternative(&alt, &labelling);

        assert_eq!(out.len(), 1);
        assert_ne!(
            out[0].1,
            LNTerm::Lit(y_canonical),
            "the witness (raw-identical to `y`) must get its OWN canonical \
             identity, not `y`'s -- aliasing them would silently fuse an \
             existentially-local witness with an unrelated real variable"
        );
    }

    /// Reviewed corner case, found IMPOSSIBLE under normal operation: two
    /// distinct domain keys of one alternative canonicalizing to the SAME
    /// literal, with DIFFERENT range terms -- which would make the
    /// domain-key sort's tie-break (and thus which range term ends up
    /// associated with that canonical position) depend on incidental raw
    /// `LVar` order rather than content. `theta` is injective by
    /// construction (every `Canonizer`-discovered literal gets a
    /// brand-new canonical index -- see `canonicalize_eq_disj_alternative`'s
    /// own doc comment), so this can't happen via the real accumulation
    /// path; every OTHER test in this file builds its labelling that way.
    /// This test instead hand-constructs a `theta` that violates
    /// injectivity directly (via `CanonLabelling::from_theta`, bypassing
    /// the `Canonizer` entirely -- something production code never does),
    /// to confirm the function fails LOUDLY rather than silently picking
    /// an arbitrary, non-deterministic order if that invariant is ever
    /// broken some other way in the future.
    #[test]
    #[should_panic(expected = "theta is no longer injective")]
    fn eq_disj_alternative_panics_if_theta_is_not_injective() {
        let v1 = LVar::new("v1", LSort::Msg, 1);
        let v2 = LVar::new("v2", LSort::Msg, 2);
        let shared_canonical = LVar::new("mv", LSort::Msg, 0);
        let mut theta: std::collections::BTreeMap<LNLit, LNLit> = std::collections::BTreeMap::new();
        // Deliberately broken: two DIFFERENT raw domain vars mapped to the
        // SAME canonical literal -- impossible via real accumulation.
        theta.insert(Lit::Var(v1), Lit::Var(shared_canonical));
        theta.insert(Lit::Var(v2), Lit::Var(shared_canonical));
        let labelling = CanonLabelling::from_theta(theta);

        let alt: LNSubstVFresh = LNSubstVFresh::from_list(vec![
            (v1, var_term(LVar::new("w", LSort::Msg, 100))),
            (v2, var_term(LVar::new("w", LSort::Msg, 200))),
        ]);

        canonicalize_eq_disj_alternative(&alt, &labelling);
    }

    #[test]
    fn eq_disj_preserves_multiplicity_of_alpha_equivalent_alternatives() {
        let x = LVar::new("x", LSort::Msg, 7);
        let alt_a: LNSubstVFresh =
            LNSubstVFresh::from_list(vec![(x, var_term(LVar::new("w", LSort::Msg, 100)))]);
        let alt_b: LNSubstVFresh =
            LNSubstVFresh::from_list(vec![(x, var_term(LVar::new("w", LSort::Msg, 200)))]);
        let disj = EqDisj {
            split_id: crate::tools::equation_store::SplitId(0),
            substs: vec![alt_a, alt_b],
        };
        let labelling = labelling_covering(&[x]);

        let out = canonicalize_eq_disj(&disj, &labelling);

        assert_eq!(
            out.len(),
            2,
            "two alpha-equivalent alternatives must NOT be deduplicated -- \
             Tamarin's own solver keeps them distinct on purpose (see \
             tools/equation_store.rs's own comment on its \
             applyBound/S.fromList handling)"
        );
        assert_eq!(
            out[0], out[1],
            "and their canonical forms must actually be equal"
        );
    }

    // -- Stage G: goals (`canonicalize_goals`) -----------------------------

    #[test]
    fn canonicalize_goals_drops_split_goals() {
        let goals = vec![(
            Goal::Split(crate::tools::equation_store::SplitId(0)),
            GoalStatus::default(),
        )];
        let out = canonicalize_goals(&goals, &std::collections::BTreeMap::new());
        assert!(
            out.is_empty(),
            "Split only names an EqDisj already in eq_store.conj and must not \
             appear in the canonical goal list: {out:?}"
        );
    }

    #[test]
    fn canonicalize_goals_keeps_action_goals_with_their_solved_status() {
        let nid = node(0);
        let x = LVar::new("x", LSort::Msg, 0);
        let labelling = labelling_covering(&[nid, x]);
        let fact = proto_fact(Multiplicity::Linear, "P", vec![v("x", LSort::Msg)]);
        let goal = |solved| {
            vec![(
                Goal::Action(nid, fact.clone()),
                GoalStatus {
                    solved,
                    ..Default::default()
                },
            )]
        };

        let open = canonicalize_goals(&goal(false), labelling.theta());
        let solved = canonicalize_goals(&goal(true), labelling.theta());

        let expected_kind = CanonicalGoalKind::Action(
            Lit::Var(LVar::new("tv", LSort::Node, 0)),
            apply_literal_renaming(&fact_to_term(&fact), labelling.theta()),
        );
        assert_eq!(open.len(), 1);
        assert_eq!(open[0].kind, expected_kind);
        assert!(!open[0].solved);
        assert_eq!(solved[0].kind, expected_kind);
        assert!(solved[0].solved);
        assert_ne!(open, solved, "solved-ness of an action goal must survive canonicalization");
    }

    /// System-level regression: an action goal's `solved` flag decides
    /// whether `candidate_methods` offers it, so two systems differing only
    /// in it are different proof-search states. The action's
    /// `VertexKind::Action` vertex is identical in both, so only the goal
    /// list can tell them apart.
    #[test]
    fn systems_differing_only_in_an_action_goals_solved_status_are_not_alpha_eq() {
        if !bliss_available() {
            return;
        }
        let colors = empty_color_table();
        let with_action_goal = |solved| {
            let mut sys = System::empty();
            sys.content_mut().goals = std::sync::Arc::new(vec![(
                Goal::Action(node(0), crate::fact::ku_fact(zero_ary_fun_term("m"))),
                GoalStatus {
                    solved,
                    ..Default::default()
                },
            )]);
            canonicalize_constraint_system(&sys, &colors)
                .unwrap_or_else(|e| panic!("canonicalize (solved={solved}): {e:?}"))
        };

        let open = with_action_goal(false);
        let solved = with_action_goal(true);

        assert_eq!(
            open.graph_part, solved.graph_part,
            "the action vertex itself is identical in both systems"
        );
        assert_ne!(
            open, solved,
            "an open and a solved action goal must not canonicalize identically"
        );
    }

    #[test]
    fn canonicalize_goals_canonicalizes_chain_via_theta() {
        let conc_nid = node(0);
        let prem_nid = node(1);
        let labelling = labelling_covering(&[conc_nid, prem_nid]);
        let goals = vec![(
            Goal::Chain((conc_nid, ConcIdx(0)), (prem_nid, PremIdx(1))),
            GoalStatus {
                solved: true,
                ..Default::default()
            },
        )];

        let out = canonicalize_goals(&goals, labelling.theta());

        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].kind,
            CanonicalGoalKind::Chain(
                Lit::Var(LVar::new("tv", LSort::Node, 0)),
                ConcIdx(0),
                Lit::Var(LVar::new("tv", LSort::Node, 1)),
                PremIdx(1),
            )
        );
        assert!(out[0].solved, "GoalStatus::solved must survive canonicalization");
    }

    #[test]
    fn canonicalize_goals_canonicalizes_premise_fact_via_theta() {
        let prem_nid = node(0);
        let x = LVar::new("x", LSort::Msg, 0);
        let labelling = labelling_covering(&[prem_nid, x]);
        let fact = in_fact(v("x", LSort::Msg));
        let goals = vec![(
            Goal::Premise((prem_nid, PremIdx(2)), fact.clone()),
            GoalStatus::default(),
        )];

        let out = canonicalize_goals(&goals, labelling.theta());

        assert_eq!(out.len(), 1);
        let expected_fact_term = apply_literal_renaming(&fact_to_term(&fact), labelling.theta());
        assert_eq!(
            out[0].kind,
            CanonicalGoalKind::Premise(
                Lit::Var(LVar::new("tv", LSort::Node, 0)),
                PremIdx(2),
                expected_fact_term,
            )
        );
    }

    #[test]
    fn canonicalize_goals_canonicalizes_disj_and_preserves_alternative_multiplicity() {
        let x = LVar::new("x", LSort::Msg, 0);
        let labelling = labelling_covering(&[x]);
        // Two IDENTICAL alternatives -- must NOT collapse to one (REVISED
        // 2026-09-23: `sort_guarded` no longer dedups, same reasoning as
        // `eq_store.conj`'s multiset preservation -- a solver-level
        // duplicate is not assumed redundant).
        let goals = vec![(
            Goal::Disj(crate::constraint::constraints::Disj(vec![g("Foo(x)"), g("Foo(x)")])),
            GoalStatus::default(),
        )];

        let out = canonicalize_goals(&goals, labelling.theta());

        assert_eq!(out.len(), 1);
        match &out[0].kind {
            CanonicalGoalKind::Disj(alts) => {
                assert_eq!(
                    alts.len(),
                    2,
                    "two identical alternatives must NOT be deduplicated -- multiplicity \
                     must survive canonicalization"
                );
                assert_eq!(alts[0], canonicalize_guarded(&g("Foo(x)"), labelling.theta()));
                assert_eq!(alts[1], canonicalize_guarded(&g("Foo(x)"), labelling.theta()));
            }
            other => panic!("expected Disj, got {other:?}"),
        }
    }

    #[test]
    fn canonicalize_goals_canonicalizes_subterm_via_theta() {
        let x = LVar::new("x", LSort::Msg, 0);
        // `v(name, sort)` always builds idx 0 -- `y` must match that, or
        // `labelling_covering` seeds a DIFFERENT literal than the one
        // `v("y", ..)` below actually produces.
        let y = LVar::new("y", LSort::Msg, 0);
        let labelling = labelling_covering(&[x, y]);
        let goals = vec![(
            Goal::Subterm((v("x", LSort::Msg), v("y", LSort::Msg))),
            GoalStatus::default(),
        )];

        let out = canonicalize_goals(&goals, labelling.theta());

        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].kind,
            CanonicalGoalKind::Subterm(
                var_term(LVar::new("mv", LSort::Msg, 0)),
                var_term(LVar::new("mv", LSort::Msg, 1)),
            )
        );
    }

    /// The exhaustiveness contract every other Stage G field already has
    /// (`lookup_theta`/`apply_literal_renaming` panic on a miss, not
    /// silently pass through -- see `apply_literal_renaming`'s own doc
    /// comment in `tamarin-term`) applies to goals too: a `Chain` goal
    /// referencing a `NodeId` the graph part never discovered must panic
    /// loudly, not silently drop the reference.
    #[test]
    #[should_panic(expected = "no entry in theta")]
    fn canonicalize_goals_panics_on_a_node_id_missing_from_theta() {
        let uncovered_nid = node(99);
        let goals = vec![(
            Goal::Chain((uncovered_nid, ConcIdx(0)), (uncovered_nid, PremIdx(0))),
            GoalStatus::default(),
        )];
        let empty_theta = std::collections::BTreeMap::new();
        let _ = canonicalize_goals(&goals, &empty_theta);
    }

    /// THE regression test this whole feature exists for, at the
    /// `canonicalize_constraint_system` level (not just `canonicalize_goals`
    /// in isolation): a system with a genuine pending goal must NOT
    /// canonicalize identically to an otherwise-identical system without
    /// it. Confirmed via real wireguard.spthy data before this fix: two
    /// sibling `!KU(mac1)`/`!KU(mac2)` sub-goals (synthesized by
    /// `insert_goal_with_loop_flag`'s pair-decomposition, `reduction.rs`)
    /// were completely invisible to canonicalization -- their `NodeId`s
    /// fell back to bare content-free `Dummy` vertices, and their own
    /// term content (which SPECIFIC message was pending derivation)
    /// appeared NOWHERE in the canonical form at all. Uses an otherwise
    /// EMPTY graph part (ground `zero_ary_fun_term`s for the `Subterm`
    /// pair, so no literal needs `theta` coverage) specifically so this
    /// test needs no bliss subprocess -- matches
    /// `empty_graph_part_systems_with_different_formulas_are_not_alpha_eq`'s
    /// own precedent for the same reason.
    #[test]
    fn systems_differing_only_in_a_pending_goal_are_not_alpha_eq() {
        let colors = empty_color_table();

        let mut sys_a = System::empty();
        sys_a.content_mut().goals = std::sync::Arc::new(vec![(
            Goal::Subterm((zero_ary_fun_term("small"), zero_ary_fun_term("big"))),
            GoalStatus::default(),
        )]);
        let sys_b = System::empty();

        let canon_a = canonicalize_constraint_system(&sys_a, &colors)
            .unwrap_or_else(|e| panic!("canonicalize a: {e:?}"));
        let canon_b = canonicalize_constraint_system(&sys_b, &colors)
            .unwrap_or_else(|e| panic!("canonicalize b: {e:?}"));

        assert_ne!(
            canon_a, canon_b,
            "a system with a genuine pending Subterm goal must not canonicalize \
             identically to one without it"
        );
    }

    // -- canonicalize_proof_method --

    /// The same goal under two different raw namings, each renamed through
    /// its own system's labelling, must give the same canonical method.
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
            canonicalize_proof_method(
                &ProofMethod::Simplify,
                &System::empty(),
                &CanonLabelling::empty()
            )
        );
    }

    /// A split goal is identified by its `EqDisj`'s canonicalized content,
    /// not by its raw split id (which the canonical form drops).
    #[test]
    fn split_methods_canonicalize_by_their_disjunctions_content() {
        use crate::tools::equation_store::SplitId;
        let x = LVar::new("x", LSort::Msg, 7);
        let solve_split = |id: i64, witness_idx: u64| {
            let mut sys = System::empty();
            sys.eq_store_mut().conj.push(EqDisj {
                split_id: SplitId(id),
                substs: vec![LNSubstVFresh::from_list(vec![(
                    x,
                    var_term(LVar::new("w", LSort::Msg, witness_idx)),
                )])],
            });
            canonicalize_proof_method(
                &ProofMethod::SolveGoal(Goal::Split(SplitId(id))),
                &sys,
                &labelling_covering(&[x]),
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
            &ProofMethod::SolveGoal(Goal::Split(crate::tools::equation_store::SplitId(9))),
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

    // -- canonicalize_constraint_system: empty graph parts --

    /// Regression test for the `EmptyGraph` bug: a system with NO nodes at
    /// all (the root of every proof, before Simplify/goal-solving
    /// introduces any rule instances; induction's `empty_trace` base
    /// case) has a graph part with zero vertices, which bliss itself
    /// refuses to canonicalize. `canonicalize_constraint_system` must
    /// special-case this rather than propagating `BlissError::EmptyGraph`
    /// -- an empty graph part is trivially canonical (exactly one empty
    /// vertex sequence, one empty edge set), and there's plenty of OTHER
    /// canonicalizable content (formulas here) that shouldn't be blocked
    /// by it. Deliberately does NOT gate on `bliss_available()`: the
    /// whole point of the fix is that this path never touches bliss.
    #[test]
    fn empty_graph_part_canonicalizes_via_the_trivial_form_not_bliss() {
        use crate::guarded::gtrue;
        let mut sys = System::empty();
        sys.formulas_mut().push(std::sync::Arc::new(gtrue()));
        // Colors are never queried here -- an empty-graph-part `System`
        // has no vertices at all, so an empty table (no maude needed) is
        // as good as a real theory's.
        let colors = empty_color_table();

        let canon = canonicalize_constraint_system(&sys, &colors)
            .unwrap_or_else(|e| panic!("canonicalize_constraint_system: {e:?}"));

        assert_eq!(
            canon.graph_part,
            canonicalize_graph_part(&[], &BTreeSet::new()),
            "an empty graph part's canonical term must be the same fixed \
             trivial term every time"
        );
        assert_eq!(canon.formulas, vec![gtrue()]);
    }

    /// Two DIFFERENT empty-graph-part systems (different formulas) must
    /// still canonicalize to DIFFERENT `CanonicalSystem`s -- the trivial
    /// graph-part shortcut must not collapse everything with no nodes
    /// into one indistinguishable bucket.
    #[test]
    fn empty_graph_part_systems_with_different_formulas_are_not_alpha_eq() {
        use crate::guarded::{gfalse, gtrue};
        let colors = empty_color_table();

        let mut sys_a = System::empty();
        sys_a.formulas_mut().push(std::sync::Arc::new(gtrue()));
        let canon_a = canonicalize_constraint_system(&sys_a, &colors).expect("canonicalize a");

        let mut sys_b = System::empty();
        sys_b.formulas_mut().push(std::sync::Arc::new(gfalse()));
        let canon_b = canonicalize_constraint_system(&sys_b, &colors).expect("canonicalize b");

        assert_ne!(canon_a, canon_b);
    }
}
