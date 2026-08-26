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

#[cfg(test)]
mod tests {
    use super::*;
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
        println!("base canon: {:?}", canonicalize_rule(&base));
        assert_eq!(canonicalize_rule(&base), canonicalize_rule(&with_new_var));
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
    }
}
