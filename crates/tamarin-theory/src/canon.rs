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
//! - guarded formulas in AC normal form ([`ac_normalize_guarded`]);
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
// quantifier's guards sorted, terms rebuilt by the smart constructors (AC
// operands flattened and sorted, `em`'s arguments sorted), binder display
// names and fact annotations erased. Bound variables are de Bruijn indices, so
// only their position counts, and the order inside a binder block stays
// significant. A formula without free variables becomes one constant of the
// system term; the free variables of the others are renamed with the rest of
// the system term, which sorts the AC parts again afterwards.

use crate::atom::{Atom, ProtoAtom};
use crate::fact::Fact;
use crate::formula::{BLNTerm, Quantifier};
use crate::guarded::{self, Guarded};

/// `t` rebuilt by the smart constructors, which flatten and sort the
/// arguments of AC symbols and sort those of C symbols (`em`).
fn ac_normalize_term(t: &BLNTerm) -> BLNTerm {
    tamarin_term::term::map_lits(t, &mut |l| l.clone())
}

/// An atom with its terms in AC normal form and the sides of `=` ordered:
/// `=` is symmetric, every other atom is directional. A fact's annotations
/// are dropped: they are hints to the solver's goal ranking with no meaning
/// of their own, and [`fact_to_term`] leaves them out of rule and goal facts
/// too.
fn normalize_atom(a: &Atom<BLNTerm>) -> Atom<BLNTerm> {
    let t = ac_normalize_term;
    match a {
        ProtoAtom::EqE(x, y) => {
            // Ordered after normalizing: `xor(b, a) = c` and `c = xor(a, b)`
            // settle on the same orientation only once both `xor`s are sorted.
            let (x, y) = (t(x), t(y));
            if x <= y {
                ProtoAtom::EqE(x, y)
            } else {
                ProtoAtom::EqE(y, x)
            }
        }
        ProtoAtom::Subterm(x, y) => ProtoAtom::Subterm(t(x), t(y)),
        ProtoAtom::Less(x, y) => ProtoAtom::Less(t(x), t(y)),
        ProtoAtom::Last(x) => ProtoAtom::Last(t(x)),
        ProtoAtom::Action(time, fact) => {
            let (tag, terms) = fact.canon_parts();
            ProtoAtom::Action(t(time), Fact::new(*tag, terms.iter().map(t).collect()))
        }
        ProtoAtom::Syntactic(sugar) => ProtoAtom::Syntactic(*sugar),
    }
}

/// The items of a conjunction or disjunction sorted; any other formula
/// unchanged.
fn sort_items(g: Guarded) -> Guarded {
    let sorted = |items: &[Guarded]| {
        let mut items = items.to_vec();
        items.sort();
        items.into()
    };
    match g {
        Guarded::Conj(items) => Guarded::Conj(sorted(&items)),
        Guarded::Disj(items) => Guarded::Disj(sorted(&items)),
        g => g,
    }
}

/// Brings `g` into AC normal form, bottom-up: atoms by [`normalize_atom`],
/// a quantifier's guards sorted, and `Conj`/`Disj` children flattened,
/// deduplicated ([`guarded::gconj`]/[`guarded::gdisj`]) and sorted, all by
/// the derived (Haskell) order on formulas. Every quantifier's binder list
/// also has its display names erased: only a binder's de Bruijn position
/// means anything (see the section docs).
pub fn ac_normalize_guarded(g: &Guarded) -> Guarded {
    match g {
        Guarded::Atom(a) => Guarded::Atom(normalize_atom(a)),
        Guarded::Disj(items) => sort_items(guarded::gdisj(
            items.iter().map(ac_normalize_guarded).collect(),
        )),
        Guarded::Conj(items) => sort_items(guarded::gconj(
            items.iter().map(ac_normalize_guarded).collect(),
        )),
        Guarded::GGuarded {
            qua,
            vars,
            guards,
            body,
        } => {
            let anonymous_vars: Vec<(String, tamarin_term::lterm::LSort)> = vars
                .iter()
                .map(|(_, sort)| (String::new(), *sort))
                .collect();
            let mut sorted_guards: Vec<Atom<BLNTerm>> = guards.iter().map(normalize_atom).collect();
            sorted_guards.sort();
            let normalized_body = ac_normalize_guarded(body);
            match qua {
                Quantifier::All => guarded::gall(anonymous_vars, sorted_guards, normalized_body),
                Quantifier::Ex => guarded::gex(anonymous_vars, sorted_guards, normalized_body),
            }
        }
    }
}

// =============================================================================
// The canonical form of a constraint system
// =============================================================================

use crate::bliss_proc::BlissError;
use crate::canon_term_graph::{eq_disj_term, goal_term, under, Op, STerm, SYSTEM_SECTIONS};
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
/// `sys.eq_store().conj`, and on the display-only `Invalidated`, which
/// `candidate_methods` never produces.
pub fn canonicalize_proof_method(
    method: &ProofMethod,
    sys: &System,
    labelling: &CanonLabelling,
) -> CanonicalProofMethod {
    pre_canonical_proof_method(method, sys).under(labelling)
}

/// A [`ProofMethod`] translated like its system's term but not renamed yet:
/// everything [`canonicalize_proof_method`] needs of the system. Renaming it
/// ([`Self::under`]) needs only the system's labelling, which can be computed
/// later from the system's term, after the system itself is gone.
#[derive(Debug, Clone, PartialEq)]
pub struct PreCanonicalProofMethod(CanonicalProofMethod);

impl PreCanonicalProofMethod {
    /// The method in the canonical form of its system, whose canonical
    /// labelling is `labelling`.
    pub fn under(&self, labelling: &CanonLabelling) -> CanonicalProofMethod {
        match &self.0 {
            CanonicalProofMethod::Solve(goal) => CanonicalProofMethod::Solve(under(goal, labelling)),
            CanonicalProofMethod::SolveSplit(disj) => {
                CanonicalProofMethod::SolveSplit(under(disj, labelling))
            }
            other => other.clone(),
        }
    }
}

/// `method`, offered for `sys`, translated for [`PreCanonicalProofMethod::under`].
/// Panics as [`canonicalize_proof_method`] does, except on variables.
pub fn pre_canonical_proof_method(method: &ProofMethod, sys: &System) -> PreCanonicalProofMethod {
    PreCanonicalProofMethod(match method {
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
                .eq_store()
                .conj
                .iter()
                .position(|d| d.split_id == *id)
                .unwrap_or_else(|| panic!("split goal {id:?} names no EqDisj in eq_store.conj"));
            CanonicalProofMethod::SolveSplit(eq_disj_term(&sys.eq_store().conj[index], index))
        }
        ProofMethod::SolveGoal(goal) => CanonicalProofMethod::Solve(
            goal_term(goal).expect("only Goal::Split has no goal term, handled above"),
        ),
        ProofMethod::Invalidated => {
            panic!("{method:?} is display-only and never a candidate proof method")
        }
    })
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
    use crate::rule::{PremIdx, Rule};
    use crate::tools::equation_store::{EqDisj, LNSubstVFresh, SplitId};
    use tamarin_parser::parser::parse_formula_str;
    use tamarin_term::maude_sig::{bp_maude_sig, pair_maude_sig, xor_maude_sig, MaudeSig};
    use tamarin_term::term::Term;
    use tamarin_term::vterm::Lit;

    fn v(name: &str, sort: LSort) -> LNTerm {
        var_term(LVar::new(name, sort, 0))
    }

    fn node(idx: u64) -> crate::constraint::constraints::NodeId {
        LVar::new("i", LSort::Node, idx)
    }

    /// The signature the tests' formulas are parsed and elaborated under:
    /// pairing, `xor` and the bilinear pairing `em`.
    fn test_sig() -> MaudeSig {
        pair_maude_sig()
            .merge(xor_maude_sig())
            .merge(bp_maude_sig())
    }

    /// Parses a surface formula string straight to its guarded form.
    fn g(s: &str) -> Guarded {
        let sig = test_sig();
        let f = parse_formula_str(s, &sig).unwrap_or_else(|e| panic!("parse {s:?}: {e}"));
        crate::elaborate::formula_to_guarded_parsed(&f, &sig)
            .unwrap_or_else(|e| panic!("formula_to_guarded {s:?}: {e:?}"))
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
        assert_eq!(
            ac_normalize_guarded(&g("x = y")),
            ac_normalize_guarded(&g("y = x"))
        );
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
            (
                "All x y #i. P(x,y) @ #i ==> F",
                "All a b #i. P(a,b) @ #i ==> F",
            ),
            (
                "All k #i. Setup(k) @ #i ==> Ex j #t. Foo(j) @ #t",
                "All p #q. Setup(p) @ #q ==> Ex w #z. Foo(w) @ #z",
            ),
        ] {
            let (fa, fb) = (ac_normalize_guarded(&g(a)), ac_normalize_guarded(&g(b)));
            assert_eq!(fa, fb, "{a} vs {b}");
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
            Guarded::Atom(ProtoAtom::EqE(Term::Lit(Lit::Con(a)), Term::Lit(Lit::Con(b)))) => {
                let mut names = [a.id.as_str(), b.id.as_str()];
                names.sort();
                assert_eq!(names, ["m", "n"]);
            }
            other => panic!("expected `=` over two names, got {other:?}"),
        }
        let (one, two) = (
            ac_normalize_guarded(&g("P(<'1', x>) @ #i")),
            ac_normalize_guarded(&g("P(<'2', x>) @ #i")),
        );
        assert_ne!(one, two);
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
                rule_ac_inst(
                    name,
                    vec![in_fact(v(var, LSort::Msg))],
                    vec![],
                    vec![fact("M")],
                ),
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
            canonicalize_proof_method(
                &ProofMethod::Simplify,
                &System::empty(),
                &CanonLabelling::empty()
            )
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
            let (_, labelling) =
                canonicalize_constraint_system_with_labelling(&sys).expect("canonicalize");
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
