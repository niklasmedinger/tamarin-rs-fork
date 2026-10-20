use super::*;

use std::sync::Arc;

use tamarin_parser::ast as p;
use tamarin_parser::parser::parse_formula_str;
use tamarin_term::builtin::{pair, pk, union, xor};
use tamarin_term::lterm::pub_term;
use tamarin_term::vterm::var_term;

use crate::bliss_proc::bliss_available;
use crate::constraint::constraints::{Edge, NodeId};
use crate::constraint::system::GoalStatus;
use crate::fact::{
    fresh_fact, in_fact, ku_fact, out_fact, proto_fact, proto_fact_ann, FactAnnotation,
    Multiplicity,
};
use crate::guarded::formula_to_guarded;
use crate::rule::{
    ConcIdx, PremIdx, ProtoRuleACInstInfo, ProtoRuleName, Rule, RuleAttributes, RuleInfo,
};
use crate::tools::equation_store::{EqDisj, LNSubst, LNSubstVFresh, SplitId};

fn msg(name: &str, idx: u64) -> LVar {
    LVar::new(name, LSort::Msg, idx)
}

fn fresh(name: &str, idx: u64) -> LVar {
    LVar::new(name, LSort::Fresh, idx)
}

fn node(idx: u64) -> NodeId {
    LVar::new("i", LSort::Node, idx)
}

fn v(x: LVar) -> LNTerm {
    var_term(x)
}

fn fact(name: &str, terms: Vec<LNTerm>) -> LNFact {
    proto_fact(Multiplicity::Linear, name, terms)
}

fn rule(
    name: &'static str,
    prems: Vec<LNFact>,
    acts: Vec<LNFact>,
    concs: Vec<LNFact>,
) -> RuleACInst {
    Rule::new(
        RuleInfo::Proto(ProtoRuleACInstInfo {
            name: ProtoRuleName::Stand(name),
            attributes: RuleAttributes::empty(),
            loop_breakers: Vec::new(),
        }),
        prems,
        concs,
        acts,
    )
}

fn g(s: &str) -> Guarded {
    let f = parse_formula_str(s).unwrap_or_else(|e| panic!("parse {s:?}: {e}"));
    formula_to_guarded(&f).unwrap_or_else(|e| panic!("formula_to_guarded {s:?}: {e}"))
}

fn alt(entries: Vec<(LVar, LNTerm)>) -> LNSubstVFresh {
    LNSubstVFresh::from_list(entries)
}

fn set_conj(sys: &mut System, conj: Vec<Vec<LNSubstVFresh>>) {
    let store = sys.eq_store_mut();
    store.conj = conj
        .into_iter()
        .enumerate()
        .map(|(i, substs)| EqDisj {
            split_id: SplitId(i as i64),
            substs,
        })
        .collect();
}

fn goals(sys: &mut System, goals: Vec<(Goal, bool)>) {
    sys.content_mut().goals = Arc::new(
        goals
            .into_iter()
            .map(|(goal, solved)| {
                (
                    goal,
                    GoalStatus {
                        solved,
                        ..Default::default()
                    },
                )
            })
            .collect(),
    );
}

const ALL: TermGraphOptions = TermGraphOptions {
    check_generators: true,
    certificate: true,
};

/// The term-graph form, with the generator check and the certificate.
fn canon(sys: &System) -> (CanonicalSystem, CanonLabelling, TermGraphStats) {
    let (form, labelling, stats) =
        canonicalize_constraint_system_via_term_graph_with_labelling(sys, ALL)
            .expect("canonicalize");
    assert_eq!(
        stats.non_genuine_generators, 0,
        "every automorphism of the term graph must be a symmetry of the system"
    );
    (form, labelling, stats)
}

fn form(sys: &System) -> CanonicalSystem {
    canon(sys).0
}

// -- The worked example (PLAN.md) ---------------------------------------------

/// Two rule instances joined by an edge, a formula, and an equation-store
/// disjunction `{x ↦ pk(u1)} ∨ {x ↦ u2 ⊕ v2}`, over the given variables.
fn worked_example(x: LVar, k: LVar, i: NodeId, j: NodeId, locals: [LVar; 3]) -> System {
    let st = || fact("St", vec![v(x), v(k)]);
    let mut sys = System::empty();
    sys.add_node(i, rule("R1", vec![], vec![], vec![st()]));
    sys.add_node(
        j,
        rule("R2", vec![st()], vec![fact("Fin", vec![v(x)])], vec![]),
    );
    sys.content_mut().edges.push(Edge {
        src: (i, ConcIdx(0)),
        tgt: (j, PremIdx(0)),
    });
    let formula = format!("All #t. K({}) @ #t ==> F", pretty_var(x));
    sys.formulas_mut().push(Arc::new(g(&formula)));
    let [u1, u2, v2] = locals;
    set_conj(
        &mut sys,
        vec![vec![
            alt(vec![(x, pk(v(u1)))]),
            alt(vec![(x, xor(v(u2), v(v2)))]),
        ]],
    );
    sys
}

/// `x` as the formula parser reads it back to the same `LVar`.
fn pretty_var(x: LVar) -> String {
    format!("{}.{}", x.name, x.idx)
}

fn worked() -> System {
    worked_example(
        msg("x", 0),
        fresh("k", 0),
        node(0),
        node(1),
        [msg("u", 0), msg("u", 1), msg("v", 1)],
    )
}

#[test]
fn the_worked_example_has_one_automorphism_and_it_fixes_every_free_variable() {
    if !bliss_available() {
        return;
    }
    let sys = worked();
    let term = system_to_term(&sys);
    let shown = term.to_string();
    for part in [
        "∃_Root",
        "Nodes{RuleInst[R1](",
        "Edges{Edge[0,0](",
        "LessAtoms{}, ⟨NoLast⟩",
        "∃_Witnesses ?w0.0 ?w0.1 ?w0.1.1. System",
        "EqConj{EqDisj{EqAlt{",
    ] {
        assert!(shown.contains(part), "{part:?} not in {shown}");
    }

    let (_, labelling, stats) = canon(&sys);
    assert_eq!(stats.free_vars, 4, "x, ~k, #i, #j");
    assert_eq!(
        stats.aut_size.as_deref(),
        Some("2"),
        "only u2 <-> v2 under the xor of the second alternative"
    );
    assert_eq!(stats.generators_moving_free_vars, 0);
    assert_eq!(
        stats.generators_moving_vars, 1,
        "the swap moves two witnesses; the check confirms it"
    );
    assert_eq!(
        stats.generator_moves.keys().collect::<Vec<_>>(),
        [
            "Binder(Witnesses) > BoundVar(Witnesses,Msg)",
            "Fun(Ac(Xor)) > Occ(Msg)"
        ]
    );
    let theta = labelling.theta();
    assert_eq!(theta[&msg("x", 0)], LVar::new("mv", LSort::Msg, 0));
    assert_eq!(theta[&fresh("k", 0)], LVar::new("fv", LSort::Fresh, 0));
    assert_ne!(theta[&node(0)], theta[&node(1)]);
}

#[test]
fn a_renamed_copy_has_the_same_form_and_certificate() {
    if !bliss_available() {
        return;
    }
    let a = worked();
    let b = worked_example(
        msg("y", 7),
        fresh("q", 3),
        node(9),
        node(2),
        [msg("w", 5), msg("w", 4), msg("w", 3)],
    );
    let (form_a, theta_a, stats_a) = canon(&a);
    let (form_b, theta_b, stats_b) = canon(&b);
    assert_eq!(form_a, form_b);
    assert_eq!(stats_a.certificate, stats_b.certificate);
    assert_eq!(theta_a.theta()[&node(0)], theta_b.theta()[&node(9)]);
    assert_eq!(theta_a.theta()[&msg("x", 0)], theta_b.theta()[&msg("y", 7)]);
}

#[test]
fn moving_the_edge_changes_the_form() {
    if !bliss_available() {
        return;
    }
    let a = worked();
    let mut b = worked();
    b.content_mut().edges[0] = Edge {
        src: (node(1), ConcIdx(0)),
        tgt: (node(0), PremIdx(0)),
    };
    assert_ne!(form(&a), form(&b));
}

// -- Faithfulness of the encoding ---------------------------------------------

fn ku_goal_system(terms: Vec<LNTerm>) -> System {
    let mut sys = System::empty();
    goals(
        &mut sys,
        terms
            .into_iter()
            .enumerate()
            .map(|(n, t)| (Goal::Action(node(n as u64), ku_fact(t)), false))
            .collect(),
    );
    sys
}

/// `x ∪ x ∪ y` next to `<'a', x>`: which variable occurs twice under the AC
/// symbol is pinned by the second goal. One occurrence vertex per
/// occurrence: with one vertex per variable, `x ∪ x ∪ y` and `x ∪ y ∪ y`
/// would have the same edges.
fn doubled_union(doubled: LNTerm, single: LNTerm, anchored: LNTerm) -> System {
    ku_goal_system(vec![
        union(union(doubled.clone(), doubled), single),
        pair(pub_term("a"), anchored),
    ])
}

#[test]
fn ac_multiplicity_of_a_variable_counts() {
    if !bliss_available() {
        return;
    }
    let (x, y) = (v(msg("x", 0)), v(msg("y", 0)));
    let xxy = doubled_union(x.clone(), y.clone(), x.clone());
    let xyy = doubled_union(y.clone(), x.clone(), x.clone());
    let (z, w) = (v(msg("z", 3)), v(msg("w", 1)));
    let zzw = doubled_union(z.clone(), w, z);
    assert_ne!(form(&xxy), form(&xyy));
    assert_eq!(form(&xxy), form(&zzw));
}

#[test]
fn a_duplicated_formula_is_not_a_single_one() {
    if !bliss_available() {
        return;
    }
    let with = |n: usize| {
        let mut sys = System::empty();
        for _ in 0..n {
            sys.formulas_mut()
                .push(Arc::new(g("All #t. P(x.0) @ #t ==> F")));
        }
        sys
    };
    assert_ne!(form(&with(1)), form(&with(2)));
}

#[test]
fn names_are_constants() {
    if !bliss_available() {
        return;
    }
    let ku = |tag: &str, var: &str| ku_goal_system(vec![pair(pub_term(tag), v(msg(var, 0)))]);
    assert_ne!(form(&ku("1", "x")), form(&ku("2", "x")));
    assert_eq!(form(&ku("1", "x")), form(&ku("1", "y")));
}

#[test]
fn which_node_is_last_counts() {
    if !bliss_available() {
        return;
    }
    let with_last = |last: u64| {
        let mut sys = System::empty();
        sys.add_node(node(0), rule("A", vec![], vec![], vec![]));
        sys.add_node(node(1), rule("A", vec![], vec![], vec![fact("M", vec![])]));
        sys.content_mut().last_atom = Some(node(last));
        sys
    };
    assert_ne!(form(&with_last(0)), form(&with_last(1)));
}

/// Two rules with the same facts but different names give different forms;
/// renaming the variable still merges.
#[test]
fn which_rule_a_node_instantiates_counts() {
    if !bliss_available() {
        return;
    }
    let with_rule = |name: &'static str, var: LVar| {
        let mut sys = System::empty();
        sys.add_node(
            node(0),
            rule(
                name,
                vec![fact("In", vec![v(var)])],
                vec![],
                vec![fact("M", vec![v(var)])],
            ),
        );
        form(&sys)
    };
    assert_ne!(with_rule("A1", msg("x", 0)), with_rule("A2", msg("x", 0)));
    assert_eq!(with_rule("A1", msg("x", 0)), with_rule("A1", msg("y", 3)));
}

/// Annotations are solver hints: rule facts and formula facts that differ
/// only in them are equal.
#[test]
fn fact_annotations_do_not_count() {
    if !bliss_available() {
        return;
    }
    let annotated = |ann: bool| {
        let anns = if ann {
            [FactAnnotation::SolveFirst].into_iter().collect()
        } else {
            Default::default()
        };
        let mut sys = System::empty();
        let st = proto_fact_ann(Multiplicity::Linear, "St", anns, vec![v(msg("x", 0))]);
        sys.add_node(node(0), rule("R", vec![st], vec![], vec![]));
        let formula_fact = GFact {
            persistent: false,
            name: "P".into(),
            args: Arc::from([GTerm::Var(BVar::Free(p::VarSpec {
                name: "x".into(),
                idx: 0,
                sort: p::SortHint::Msg,
                typ: None,
            }))]),
            annotations: if ann {
                vec![p::FactAnnotation::SolveFirst]
            } else {
                Vec::new()
            },
        };
        sys.formulas_mut().push(Arc::new(Guarded::Disj(Arc::from([
            Guarded::Atom(GAtom::Pred(formula_fact)),
            crate::guarded::gfalse(),
        ]))));
        sys
    };
    assert_eq!(form(&annotated(false)), form(&annotated(true)));
}

// -- The equation store -------------------------------------------------------

/// Each alternative is its own scope: renaming its local variables and
/// reordering the alternatives changes nothing.
#[test]
fn alpha_equivalent_alternatives_in_either_order_are_equal() {
    if !bliss_available() {
        return;
    }
    let (x, y) = (msg("x", 0), msg("y", 0));
    let system = |alts: Vec<LNSubstVFresh>| {
        let mut sys = ku_goal_system(vec![pair(v(x), v(y))]);
        set_conj(&mut sys, vec![alts]);
        sys
    };
    let (u, w) = (msg("u", 1), msg("w", 2));
    let a = system(vec![
        alt(vec![(x, pk(v(u))), (y, v(u))]),
        alt(vec![(x, v(w)), (y, pk(v(w)))]),
    ]);
    let b = system(vec![
        alt(vec![(x, v(u)), (y, pk(v(u)))]),
        alt(vec![(x, pk(v(w))), (y, v(w))]),
    ]);
    let swapped_keys = system(vec![
        alt(vec![(x, pk(v(u))), (y, v(u))]),
        alt(vec![(x, pk(v(w))), (y, v(w))]),
    ]);
    assert_eq!(form(&a), form(&b));
    assert_ne!(form(&a), form(&swapped_keys));
}

/// A range variable is local even when its raw identity is a key's: it
/// cannot be captured by the free variable of the same name.
#[test]
fn a_range_variable_named_like_a_key_is_still_local() {
    if !bliss_available() {
        return;
    }
    let x = msg("x", 0);
    let system = |range: LVar| {
        let mut sys = ku_goal_system(vec![v(x)]);
        set_conj(&mut sys, vec![vec![alt(vec![(x, pk(v(range)))])]]);
        sys
    };
    assert_eq!(form(&system(x)), form(&system(msg("u", 7))));
}

/// The store-only variables `a`, `b` link two disjunctions (the shape of
/// idbased/BP_IBS_1): renaming them and reordering everything changes
/// nothing, and cutting the link changes the form.
#[test]
fn store_only_variables_link_disjunctions() {
    if !bliss_available() {
        return;
    }
    let (x, x1) = (msg("x", 0), msg("x", 1));
    let linking = |a: LVar, b: LVar, w: u64| {
        let w0 = v(msg("w", w));
        let (w1, w2) = (v(msg("w", w + 1)), v(msg("w", w + 2)));
        vec![
            alt(vec![
                (x, w0.clone()),
                (x1, w0.clone()),
                (a, w0.clone()),
                (b, w0),
            ]),
            alt(vec![
                (x, w1.clone()),
                (x1, w2.clone()),
                (a, w1.clone()),
                (b, w2.clone()),
            ]),
            alt(vec![(x, w1.clone()), (x1, w2.clone()), (a, w2), (b, w1)]),
        ]
    };
    let variant = |a: LVar, b: LVar, w: u64| {
        let p = pub_term("P");
        vec![
            alt(vec![(a, v(msg("w", w))), (b, p.clone())]),
            alt(vec![(a, p), (b, v(msg("w", w + 1)))]),
        ]
    };
    let system = |conj: Vec<Vec<LNSubstVFresh>>| {
        let mut sys = ku_goal_system(vec![pair(v(x), v(x1))]);
        set_conj(&mut sys, conj);
        sys
    };
    let (a, b) = (msg("x", 4), msg("x", 5));
    let linked = system(vec![linking(a, b, 20), variant(a, b, 30)]);
    let (a2, b2) = (msg("y", 41), msg("y", 40));
    let mut reversed = linking(a2, b2, 70);
    reversed.reverse();
    let renamed = system(vec![variant(a2, b2, 50), reversed]);
    let (c, d) = (msg("x", 6), msg("x", 7));
    let unlinked = system(vec![linking(a, b, 20), variant(c, d, 30)]);
    assert_eq!(form(&linked), form(&renamed));
    assert_ne!(form(&linked), form(&unlinked));
}

/// The store-only variables `a`, `b` look alike (each occurs once, with a
/// variable range), but swapping them gives a different store that is
/// equivalent only via the renaming `a ↔ b`.
#[test]
fn symmetric_store_only_variables_get_the_same_form() {
    if !bliss_available() {
        return;
    }
    let (x, a, b) = (msg("x", 0), msg("x", 4), msg("x", 5));
    let system = |first: LVar, second: LVar| {
        let w = v(msg("w", 20));
        let mut sys = ku_goal_system(vec![v(x)]);
        set_conj(
            &mut sys,
            vec![vec![
                alt(vec![(x, w.clone()), (first, w.clone())]),
                alt(vec![(x, pub_term("P")), (second, w)]),
            ]],
        );
        sys
    };
    assert_eq!(form(&system(a, b)), form(&system(b, a)));
}

/// `subst`'s domain variables no longer occur anywhere else once
/// `substSystem` has applied it; they are free variables all the same.
#[test]
fn a_substituted_variable_outside_the_graph_is_named() {
    if !bliss_available() {
        return;
    }
    let x = msg("x", 0);
    let system = |gone: LVar| {
        let mut sys = ku_goal_system(vec![v(x)]);
        sys.eq_store_mut().subst = LNSubst::from_list(vec![(gone, v(x))]);
        sys
    };
    let (canon, labelling, _) = canon(&system(msg("x", 9)));
    assert!(labelling.theta().contains_key(&msg("x", 9)));
    assert_eq!(canon, form(&system(msg("y", 3))));
    assert_ne!(canon, form(&ku_goal_system(vec![v(x)])));
}

/// The induction hypothesis puts a free `Last(#v)` into a formula; the term
/// graph names `#v` like any other free variable.
#[test]
fn a_free_node_variable_only_in_a_formula_is_named() {
    if !bliss_available() {
        return;
    }
    let vs = p::VarSpec {
        name: "v".into(),
        idx: 3,
        sort: p::SortHint::Node,
        typ: None,
    };
    let mut sys = System::empty();
    sys.formulas_mut()
        .push(Arc::new(Guarded::Atom(GAtom::Last(GTerm::Var(
            BVar::Free(vs.clone()),
        )))));
    let (_, labelling, _) = canon(&sys);
    assert_eq!(
        labelling.theta()[&varspec_to_lvar(&vs)],
        LVar::new("tv", LSort::Node, 0)
    );
}

// -- Ties that only part of the system breaks --------------------------------

/// `!KU(x ⊕ y)` alone cannot tell `x` from `y`; a subterm goal on `x` can.
/// Swapping the raw indices must not change the form.
#[test]
fn a_renaming_tie_is_broken_by_the_rest_of_the_system() {
    if !bliss_available() {
        return;
    }
    let with_raw_indices = |x_idx: u64, y_idx: u64| {
        let (x, y) = (v(msg("x", x_idx)), v(msg("y", y_idx)));
        let mut sys = System::empty();
        goals(
            &mut sys,
            vec![
                (Goal::Action(node(0), ku_fact(xor(x.clone(), y))), false),
                (Goal::Subterm((x, pub_term("c"))), false),
            ],
        );
        sys
    };
    assert_eq!(form(&with_raw_indices(1, 2)), form(&with_raw_indices(2, 1)));
}

fn ku_pair_goals(goals_spec: &[(u64, &str, &str, bool)]) -> System {
    let mut sys = System::empty();
    goals(
        &mut sys,
        goals_spec
            .iter()
            .map(|&(n, tag, var, solved)| {
                let t = pair(pub_term(tag), v(msg(var, 0)));
                (Goal::Action(node(n), ku_fact(t)), solved)
            })
            .collect(),
    );
    sys
}

#[test]
fn a_tie_break_is_independent_of_node_numbering_and_variable_names() {
    if !bliss_available() {
        return;
    }
    let original = ku_pair_goals(&[
        (0, "a", "x", true),
        (1, "a", "y", false),
        (2, "b", "x", false),
        (3, "b", "y", false),
    ]);
    let renumbered_and_renamed = ku_pair_goals(&[
        (0, "a", "w", false),
        (1, "b", "u", false),
        (2, "b", "w", false),
        (3, "a", "u", true),
    ]);
    let solved_moved = ku_pair_goals(&[
        (0, "a", "x", false),
        (1, "a", "y", false),
        (2, "b", "x", true),
        (3, "b", "y", false),
    ]);
    assert_eq!(form(&original), form(&renumbered_and_renamed));
    assert_ne!(form(&original), form(&solved_moved));
}

/// Two interchangeable sessions: bliss reports their swap, it moves free
/// variables, and it is a symmetry of the system.
#[test]
fn a_symmetry_of_the_system_is_a_genuine_generator() {
    if !bliss_available() {
        return;
    }
    let sys = ku_pair_goals(&[(0, "a", "x", false), (1, "a", "y", false)]);
    let (_, _, stats) = canon(&sys);
    assert_eq!(stats.aut_size.as_deref(), Some("2"));
    assert_eq!(stats.generators_moving_free_vars, 1);
}

/// The shape of `xor0`'s split: `~a ⊕ ~b` and `~a.1 ⊕ ~b.1`, and the
/// disjunction `{a, a.1 ↦ u; b, b.1 ↦ v} ∨ {a, b.1 ↦ v; b, a.1 ↦ u}`.
/// Swapping `~a` and `~b` swaps the two alternatives: a symmetry, which the
/// check must confirm, local variable names included.
#[test]
fn a_symmetry_swapping_equation_store_alternatives_is_genuine() {
    if !bliss_available() {
        return;
    }
    let (a, b, a1, b1) = (fresh("a", 0), fresh("b", 0), fresh("a", 1), fresh("b", 1));
    let (u, w) = (fresh("x", 6), fresh("x", 7));
    let mut sys = System::empty();
    let mut gs = vec![
        (Goal::Action(node(0), ku_fact(xor(v(a), v(b)))), false),
        (Goal::Action(node(1), ku_fact(xor(v(a1), v(b1)))), false),
    ];
    gs.extend((2..8).map(|n| (Goal::Action(node(n), ku_fact(pub_term("c"))), false)));
    goals(&mut sys, gs);
    set_conj(
        &mut sys,
        vec![vec![
            alt(vec![(a, v(u)), (b, v(w)), (a1, v(u)), (b1, v(w))]),
            alt(vec![(a, v(w)), (b, v(u)), (a1, v(u)), (b1, v(w))]),
        ]],
    );
    let (_, _, stats) = canon(&sys);
    assert!(stats.generators_moving_free_vars >= 1);
}

#[test]
fn a_system_without_free_variables_skips_bliss() {
    let mut sys = System::empty();
    sys.formulas_mut().push(Arc::new(crate::guarded::gtrue()));
    let (form_true, labelling, stats) =
        canonicalize_constraint_system_via_term_graph_with_labelling(
            &sys,
            TermGraphOptions::default(),
        )
        .expect("canonicalize");
    assert!(!stats.bliss_ran);
    assert!(labelling.theta().is_empty());

    let mut other = System::empty();
    other
        .formulas_mut()
        .push(Arc::new(crate::guarded::gfalse()));
    let (form_false, _, _) = canonicalize_constraint_system_via_term_graph_with_labelling(
        &other,
        TermGraphOptions::default(),
    )
    .expect("canonicalize");
    assert_ne!(form_true, form_false);
}

// -- Which systems are equal -------------------------------------------------

/// Over a set of systems, exactly the renamed copies get the same form.
#[test]
fn exactly_the_renamed_copies_are_equal() {
    if !bliss_available() {
        return;
    }
    let (x, y, z) = (v(msg("x", 0)), v(msg("y", 0)), v(msg("z", 2)));
    let mut systems = vec![
        worked(),
        worked_example(
            msg("y", 7),
            fresh("q", 3),
            node(9),
            node(2),
            [msg("w", 5), msg("w", 4), msg("w", 3)],
        ),
        doubled_union(x.clone(), y.clone(), x.clone()),
        doubled_union(y.clone(), x.clone(), x.clone()),
        doubled_union(z.clone(), x.clone(), z.clone()),
        ku_pair_goals(&[
            (0, "a", "x", true),
            (1, "a", "y", false),
            (2, "b", "x", false),
        ]),
        ku_pair_goals(&[
            (0, "a", "w", true),
            (1, "b", "w", false),
            (2, "a", "u", false),
        ]),
        ku_pair_goals(&[
            (0, "a", "x", false),
            (1, "a", "y", true),
            (2, "b", "x", false),
        ]),
    ];
    let mut moved = worked();
    moved.content_mut().edges[0] = Edge {
        src: (node(1), ConcIdx(0)),
        tgt: (node(0), PremIdx(0)),
    };
    systems.push(moved);

    let forms: Vec<CanonicalSystem> = systems.iter().map(form).collect();
    let renamed_copies = [(0, 1), (2, 4), (5, 6)];
    for i in 0..systems.len() {
        for j in i + 1..systems.len() {
            assert_eq!(
                forms[i] == forms[j],
                renamed_copies.contains(&(i, j)),
                "systems {i} and {j}"
            );
        }
    }
}

// --- Step 3: the canonical form of a term --------------------------------

fn free(x: LVar) -> STerm {
    STerm::Var(VarRef::Free(x))
}

fn ac_union(args: Vec<STerm>) -> STerm {
    STerm::App(
        Op::Fun(tamarin_term::function_symbols::FunSym::Ac(
            tamarin_term::function_symbols::AcSym::Union,
        )),
        args,
    )
}

#[test]
fn normalize_flattens_and_sorts_an_ac_operator() {
    let (a, b, c) = (free(msg("a", 1)), free(msg("b", 2)), free(msg("c", 3)));
    let nested = ac_union(vec![a.clone(), ac_union(vec![b.clone(), c.clone()])]);
    let flat_reordered = ac_union(vec![c, b, a]);
    assert_eq!(normalize(nested), normalize(flat_reordered));
}

#[test]
fn normalize_keeps_duplicates_under_an_ac_operator() {
    let (a, b) = (free(msg("a", 1)), free(msg("b", 2)));
    let doubled = normalize(ac_union(vec![a.clone(), a.clone(), b.clone()]));
    let single = normalize(ac_union(vec![a, b]));
    assert_ne!(doubled, single);
    assert_ne!(fingerprint_sterm(&doubled), fingerprint_sterm(&single));
}

#[test]
fn normalize_sorts_c_arguments_but_not_free_ones() {
    let (x, y) = (free(msg("x", 1)), free(msg("y", 2)));
    let eq = |l: &STerm, r: &STerm| normalize(STerm::App(Op::GEq, vec![l.clone(), r.clone()]));
    assert_eq!(eq(&x, &y), eq(&y, &x));
    let edge = |l: &STerm, r: &STerm| normalize(STerm::App(Op::Edge(0, 0), vec![l.clone(), r.clone()]));
    assert_ne!(edge(&x, &y), edge(&y, &x));
}

#[test]
fn renaming_then_normalizing_identifies_renamed_copies() {
    let (x, y, u, v) = (msg("x", 1), msg("y", 2), msg("u", 7), msg("v", 9));
    let t1 = ac_union(vec![free(x), STerm::App(Op::GEq, vec![free(x), free(y)])]);
    let t2 = ac_union(vec![STerm::App(Op::GEq, vec![free(v), free(u)]), free(u)]);
    let theta1: BTreeMap<LVar, LVar> = [(x, msg("mv", 0)), (y, msg("mv", 1))].into();
    let theta2: BTreeMap<LVar, LVar> = [(u, msg("mv", 0)), (v, msg("mv", 1))].into();
    let f1 = normalize(rename(&t1, &theta1));
    let f2 = normalize(rename(&t2, &theta2));
    assert_eq!(f1, f2);
    assert_eq!(fingerprint_sterm(&f1), fingerprint_sterm(&f2));
}

#[test]
#[should_panic(expected = "not covered by the canonical renaming")]
fn renaming_a_free_variable_the_labelling_misses_panics() {
    rename(&free(msg("x", 1)), &BTreeMap::new());
}

// -- Facts and rule instances ---------------------------------------------

/// A system with one rule instance.
fn one_node(ru: RuleACInst) -> System {
    let mut sys = System::empty();
    sys.add_node(node(0), ru);
    sys
}

/// A fact's tag counts (its name, its multiplicity, built-in or not); the
/// names of its variables do not.
#[test]
fn a_facts_tag_counts_and_its_variable_names_do_not() {
    if !bliss_available() {
        return;
    }
    let with = |f: LNFact| form(&one_node(rule("R", vec![f], vec![], vec![])));
    let (x, y) = (msg("x", 0), msg("y", 3));
    let st = |m: Multiplicity| proto_fact(m, "St", vec![v(x)]);
    assert_eq!(with(fact("Login", vec![v(x)])), with(fact("Login", vec![v(y)])));
    assert_ne!(with(fact("Login", vec![v(x)])), with(fact("Logout", vec![v(x)])));
    assert_ne!(with(st(Multiplicity::Linear)), with(st(Multiplicity::Persistent)));
    let n = fresh("n", 0);
    assert_ne!(
        with(fresh_fact(v(n))),
        with(proto_fact(Multiplicity::Linear, "Fr", vec![v(n)])),
        "a built-in tag never collides with a protocol fact of the same name"
    );
}

/// A rule instance counts up to renaming: which variables its facts share,
/// and which group a fact is in, count; `new_vars`, derived from the facts,
/// does not.
#[test]
fn a_rule_instance_counts_up_to_renaming() {
    if !bliss_available() {
        return;
    }
    let (x, y, u) = (msg("x", 0), msg("y", 0), msg("u", 4));
    let forward = |a: LVar, b: LVar| one_node(rule("R", vec![in_fact(v(a))], vec![], vec![out_fact(v(b))]));
    assert_eq!(form(&forward(x, x)), form(&forward(u, u)));
    assert_ne!(form(&forward(x, x)), form(&forward(x, y)));
    let f = fact("P", vec![v(x)]);
    assert_ne!(
        form(&one_node(rule("R", vec![f.clone()], vec![], vec![]))),
        form(&one_node(rule("R", vec![], vec![], vec![f])))
    );
    let base = rule("R", vec![in_fact(v(x))], vec![], vec![]);
    assert_eq!(
        form(&one_node(base.clone())),
        form(&one_node(base.with_new_vars(vec![v(fresh("n", 0))])))
    );
}

// -- Formulas ----------------------------------------------------------------

fn formula_form(f: &str) -> CanonicalSystem {
    let mut sys = System::empty();
    sys.formulas_mut().push(Arc::new(g(f)));
    form(&sys)
}

/// `work.tex`'s example `z = y ∧ ∃ x i. G(x)@i`: the names of the free
/// variables and the order of the conjuncts do not count, which variable
/// is which does.
#[test]
fn a_formulas_free_variables_are_renamed_with_the_system() {
    if !bliss_available() {
        return;
    }
    let original = formula_form("z = y & Ex x #i. G(x)@i");
    assert_eq!(original, formula_form("b.4 = a.2 & Ex x #i. G(x)@i"));
    assert_eq!(original, formula_form("(Ex x #i. G(x)@i) & y = z"));
    assert_ne!(original, formula_form("z = z & Ex x #i. G(x)@i"));
}

/// Bound variables are de Bruijn indices and are never permuted, so the
/// order of the binders in one block counts: `∀ x y` and `∀ y x` over the
/// same body give different forms. A missed merge, never a wrong one.
#[test]
fn the_binder_order_inside_a_block_counts() {
    if !bliss_available() {
        return;
    }
    assert_ne!(
        formula_form("All x y #i. P(x, y) @ #i ==> F"),
        formula_form("All y x #i. P(x, y) @ #i ==> F")
    );
    assert_eq!(
        formula_form("All x y #i. P(x, y) @ #i ==> F"),
        formula_form("All a b #j. P(a, b) @ #j ==> F")
    );
}

// -- The equation store, continued -------------------------------------------

/// Maude numbers a unifier's witnesses arbitrarily: their numbering does not
/// count.
#[test]
fn an_alternatives_witness_numbering_does_not_count() {
    if !bliss_available() {
        return;
    }
    let x = msg("x", 0);
    let system = |w: u64| {
        let mut sys = ku_goal_system(vec![v(x)]);
        let (w1, w2) = (msg("w", w), msg("w", w + 1));
        set_conj(&mut sys, vec![vec![alt(vec![(x, pair(v(w1), v(w2)))])]]);
        sys
    };
    assert_eq!(form(&system(100)), form(&system(500)));
}

/// One witness in two range terms of an alternative is not two witnesses.
#[test]
fn a_witness_shared_across_range_terms_counts() {
    if !bliss_available() {
        return;
    }
    let (p, q) = (msg("p", 0), msg("q", 0));
    let system = |second: LVar| {
        let w = msg("w", 9);
        let mut sys = ku_goal_system(vec![pair(v(p), v(q))]);
        set_conj(
            &mut sys,
            vec![vec![alt(vec![(p, pair(v(w), v(w))), (q, v(second))])]],
        );
        sys
    };
    assert_ne!(form(&system(msg("w", 9))), form(&system(msg("w", 10))));
}

/// `w1 ⊕ w2` alone cannot tell its witnesses apart; `pair(w, w)` for one of
/// them can. The four alternatives are equal up to renaming, so their forms
/// are equal whichever witness the pair uses and whatever the keys' raw
/// indices: the tie is settled by content, not by raw variable order.
#[test]
fn a_tie_between_witnesses_is_settled_by_the_other_range_term() {
    if !bliss_available() {
        return;
    }
    let (w1, w2) = (msg("w", 900), msg("w", 901));
    let mut forms = Vec::new();
    for (p_idx, q_idx) in [(7, 8), (8, 7)] {
        let (p, q) = (msg("p", p_idx), msg("q", q_idx));
        for w in [w1, w2] {
            let mut sys = ku_goal_system(vec![pair(v(p), v(q))]);
            set_conj(
                &mut sys,
                vec![vec![alt(vec![(p, xor(v(w1), v(w2))), (q, pair(v(w), v(w)))])]],
            );
            forms.push(form(&sys));
        }
    }
    assert!(forms.windows(2).all(|pair| pair[0] == pair[1]));
}

/// A witness raw-identical to an unrelated free variable is still local.
#[test]
fn a_witness_named_like_another_free_variable_is_still_local() {
    if !bliss_available() {
        return;
    }
    let (x, y) = (msg("x", 0), msg("y", 42));
    let system = |w: LVar| {
        let mut sys = ku_goal_system(vec![pair(v(x), v(y))]);
        set_conj(&mut sys, vec![vec![alt(vec![(x, pk(v(w)))])]]);
        sys
    };
    assert_eq!(form(&system(y)), form(&system(msg("u", 5))));
}

/// Alpha-equivalent alternatives of one disjunction are kept apart: the
/// solver keeps them, and their number is proof-search state.
#[test]
fn alpha_equivalent_alternatives_are_not_deduplicated() {
    if !bliss_available() {
        return;
    }
    let x = msg("x", 0);
    let system = |n: u64| {
        let mut sys = ku_goal_system(vec![v(x)]);
        let alts = (0..n).map(|k| alt(vec![(x, pk(v(msg("w", 100 + k))))])).collect();
        set_conj(&mut sys, vec![alts]);
        sys
    };
    assert_ne!(form(&system(1)), form(&system(2)));
}

// -- Goals -------------------------------------------------------------------

/// A `Split` goal only names a disjunction the form has anyway.
#[test]
fn a_split_goal_is_left_out() {
    if !bliss_available() {
        return;
    }
    let x = msg("x", 0);
    let base = || {
        let mut sys = ku_goal_system(vec![v(x)]);
        set_conj(
            &mut sys,
            vec![vec![alt(vec![(x, pk(v(msg("w", 1))))]), alt(vec![(x, v(msg("w", 2)))])]],
        );
        sys
    };
    let mut with_split = base();
    let mut gs: Vec<(Goal, GoalStatus)> = with_split.goals.iter().cloned().collect();
    gs.push((Goal::Split(SplitId(0)), GoalStatus::default()));
    with_split.content_mut().goals = Arc::new(gs);
    assert_eq!(form(&base()), form(&with_split));
}

/// Chain, premise, subterm and disjunction goals are renamed with the rest of
/// the system; the direction of a subterm goal counts.
#[test]
fn goals_are_renamed_with_the_system() {
    if !bliss_available() {
        return;
    }
    let system = |a: NodeId, b: NodeId, x: LVar, y: LVar, swap: bool| {
        let (s, t) = if swap { (v(y), v(x)) } else { (v(x), v(y)) };
        let foo = format!("Foo({})", pretty_var(x));
        let mut sys = System::empty();
        goals(
            &mut sys,
            vec![
                (Goal::Chain((a, ConcIdx(0)), (b, PremIdx(1))), true),
                (Goal::Premise((b, PremIdx(2)), in_fact(v(x))), false),
                (Goal::Subterm((s, t)), false),
                (
                    Goal::Disj(crate::constraint::constraints::Disj(vec![g(&foo), g(&foo)])),
                    false,
                ),
            ],
        );
        sys
    };
    let (x, y) = (msg("x", 0), msg("y", 0));
    let original = form(&system(node(0), node(1), x, y, false));
    assert_eq!(
        original,
        form(&system(node(5), node(3), msg("u", 2), msg("w", 4), false))
    );
    assert_ne!(original, form(&system(node(0), node(1), x, y, true)));
}

/// A disjunction goal keeps duplicate alternatives.
#[test]
fn a_disjunction_goals_duplicate_alternatives_are_kept() {
    if !bliss_available() {
        return;
    }
    let with = |n: usize| {
        let mut sys = System::empty();
        let alternatives = (0..n).map(|_| g("Foo(x.0)")).collect();
        goals(
            &mut sys,
            vec![(Goal::Disj(crate::constraint::constraints::Disj(alternatives)), false)],
        );
        form(&sys)
    };
    assert_ne!(with(1), with(2));
}

/// The same raw range variable in two alternatives is two witnesses: each
/// alternative's are its own.
#[test]
fn witnesses_of_different_alternatives_are_independent() {
    if !bliss_available() {
        return;
    }
    let x = msg("x", 0);
    let system = |first: LVar, second: LVar| {
        let mut sys = ku_goal_system(vec![v(x)]);
        set_conj(
            &mut sys,
            vec![vec![alt(vec![(x, pk(v(first)))]), alt(vec![(x, v(second))])]],
        );
        sys
    };
    let w = msg("w", 1);
    assert_eq!(form(&system(w, w)), form(&system(msg("u", 3), msg("v", 8))));
}

/// A witness's name starts with a NUL, which no variable of a theory or of
/// the solver has; a free variable that did would be confused with one.
#[test]
#[should_panic(expected = "named like a witness")]
fn a_free_variable_named_like_a_witness_panics() {
    let sys = ku_goal_system(vec![v(LVar::new("\0w0.0", LSort::Msg, 0))]);
    system_to_term(&sys);
}
