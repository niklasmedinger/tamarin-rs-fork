// Currently GPL 3.0 until granted permission by the upstream authors
// of the tamarin-prover sources this file cites; list them with:
//   scripts/gen_license_headers.py --authors <this file>

//! Pins the thesis's running example (`examples/canon_worked_example.rs`,
//! whose pipeline this test reuses): what the chapter says about Γ, Δ and Γ'
//! has to stay true of the code, at the level of the term the chapter defines
//! and of the term the code builds.
//!
//! Skips (via `bliss_available()`'s own panic-unless-opted-out gate) if
//! `bliss` is not on `PATH`/`$BLISS_PATH` and `TAM_ALLOW_NO_BLISS=1` is
//! set; otherwise a missing bliss fails loudly rather than reporting a
//! vacuous green.

use std::collections::BTreeMap;

use tamarin_term::builtin::xor;
use tamarin_term::lterm::{LSort, LVar};
use tamarin_term::vterm::var_term;
use tamarin_theory::bliss_proc::bliss_available;
use tamarin_theory::canon_fingerprint::{compare_fingerprints, fingerprint_constraint_system};
use tamarin_theory::canon_term_graph::system_to_term;

#[path = "../examples/canon_worked_example.rs"]
#[allow(dead_code)]
mod worked;

use worked::{
    action, delta_example, gamma_example, gamma_prime_example, induced_renaming, latex, msg, node,
    render, run, study, system, thesis_term, Notation, Run, Study,
};

/// The two levels of a study.
fn levels(s: &Study) -> [&Run; 2] {
    [&s.thesis, &s.implementation]
}

#[test]
fn gamma_and_delta_are_equal_at_both_levels_through_the_expected_renaming() {
    if !bliss_available() {
        return;
    }
    let gamma = study(&gamma_example()).expect("Γ");
    let delta = study(&delta_example()).expect("Δ");
    let expected: BTreeMap<LVar, LVar> = BTreeMap::from([
        (msg("x"), msg("v")),
        (msg("y"), msg("u")),
        (node("i"), node("l")),
        (node("j"), node("k")),
    ]);
    for (a, b) in levels(&gamma).into_iter().zip(levels(&delta)) {
        assert_ne!(
            a.dimacs, b.dimacs,
            "equal inputs would make the example show nothing"
        );
        assert_eq!(a.canonical_dimacs, b.canonical_dimacs);
        assert_eq!(a.form, b.form);
        assert_eq!(a.key, b.key);
        assert_eq!(induced_renaming(&a.sigma, &b.sigma), expected);
        let (mut keys_a, mut keys_b) = (a.graph.keys.clone(), b.graph.keys.clone());
        keys_a.sort();
        keys_b.sort();
        assert_eq!(keys_a, keys_b, "isomorphic graphs have equal keys");
        for r in [a, b] {
            assert_eq!(r.stats.aut_size, "1", "B(x)@j tells x and y apart");
        }
    }
}

#[test]
fn the_thesis_term_is_the_one_the_chapter_defines() {
    if !bliss_available() {
        return;
    }
    let gamma = study(&gamma_example()).expect("Γ");
    let term = latex(&gamma.thesis.term, Notation::Term);
    assert_eq!(
        term,
        r"\exists_{\mathit{msg}}\ x, y.\ \exists_{\mathit{tmp}}\ i, j.\ \mathsf{Sys}(\mathsf{Fml}(\mathsf{Act}(\mathsf{A}(x \oplus y), i)), \mathsf{Fml}(\mathsf{Act}(\mathsf{B}(x), j)))",
        "one typed binder per sort, one Fml per constraint"
    );
    // Two binders, four variable vertices, thirteen vertices for `Sys(…)`
    // with its five occurrences; eighteen term edges and five occurrence
    // edges.
    assert_eq!(gamma.thesis.graph.keys.len(), 19);
    assert_eq!(gamma.thesis.graph.edges.len(), 23);
}

#[test]
fn the_pipeline_reproduces_the_library_on_the_real_system_term() {
    if !bliss_available() {
        return;
    }
    for example in [gamma_example(), delta_example(), gamma_prime_example()] {
        let s = study(&example).expect("study");
        let (form, labelling, stats) = &s.library;
        assert_eq!(form.term(), &s.implementation.form);
        assert_eq!(labelling.theta(), s.implementation.sigma.theta());
        assert_eq!(stats.non_genuine_generators, 0);
    }
}

#[test]
fn the_reported_labelling_is_the_one_behind_the_renaming() {
    if !bliss_available() {
        return;
    }
    for example in [gamma_example(), delta_example()] {
        let s = study(&example).expect("study");
        for r in levels(&s) {
            let rho = &r.bliss.canonical_labeling;
            for sort in [LSort::Msg, LSort::Node] {
                let mut vars: Vec<(usize, LVar)> = r
                    .graph
                    .free_vars
                    .iter()
                    .filter(|(_, x)| x.sort == sort)
                    .map(|&(w, x)| (rho.image_of(w), x))
                    .collect();
                vars.sort();
                for (index, (_, x)) in vars.iter().enumerate() {
                    assert_eq!(r.sigma.theta()[x].idx, index as u64);
                    assert_eq!(r.sigma.theta()[x].sort, sort);
                }
            }
        }
    }
}

#[test]
fn the_automorphism_of_gamma_prime_is_a_symmetry_and_gives_the_same_form() {
    if !bliss_available() {
        return;
    }
    let prime = study(&gamma_prime_example()).expect("Γ'");
    for r in levels(&prime) {
        assert_eq!(r.stats.aut_size, "2");
        assert_eq!(r.bliss.generators.len(), 1);
        let g = &r.bliss.generators[0];
        let vertex = |x: LVar| {
            r.graph
                .free_vars
                .iter()
                .find(|(_, y)| *y == x)
                .expect("a variable vertex")
                .0
        };
        let (x, y) = (vertex(msg("x")), vertex(msg("y")));
        assert_eq!(g.image_of(x), y);
        assert_eq!(g.image_of(y), x);

        let (other, form) = &r.generator_forms[0];
        assert_eq!(form, &r.form, "both labellings give one form");
        assert_ne!(
            other.theta(),
            r.sigma.theta(),
            "the labellings differ in the names they give x and y"
        );
    }
}

#[test]
fn canonicalizing_the_canonical_system_changes_nothing() {
    if !bliss_available() {
        return;
    }
    for example in [gamma_example(), delta_example()] {
        let s = study(&example).expect("study");
        let again = study(&example.renamed(&s.implementation.sigma)).expect("renamed");
        assert_eq!(again.implementation.form, s.implementation.form);
        for (x, c) in again.implementation.sigma.theta() {
            assert_eq!(x, c, "the canonical names are already canonical");
        }
    }
}

#[test]
fn a_system_that_is_not_a_renaming_gets_another_form_and_key() {
    if !bliss_available() {
        return;
    }
    let (x, y, z) = (msg("x"), msg("y"), msg("z"));
    let (i, j) = (node("i"), node("j"));
    let a = action(i, "A", xor(var_term(x), var_term(y)));
    let controls = [
        ("B on a third variable", action(j, "B", var_term(z))),
        ("a shared timepoint", action(i, "B", var_term(x))),
    ];
    let gamma = study(&gamma_example()).expect("Γ");
    for (what, b) in controls {
        let term = system_to_term(&system(&[a.clone(), b]));
        let thesis = run(&thesis_term(&term)).expect("thesis level");
        let implementation = run(&term).expect("implementation level");
        for (r, g) in [thesis, implementation].iter().zip(levels(&gamma)) {
            assert_ne!(r.form, g.form, "{what}");
            assert_ne!(r.key, g.key, "{what}");
            assert_ne!(r.canonical_dimacs, g.canonical_dimacs, "{what}");
        }
    }
}

#[test]
fn the_keys_of_gamma_and_delta_match_section_by_section() {
    if !bliss_available() {
        return;
    }
    let gamma = study(&gamma_example()).expect("Γ");
    let delta = study(&delta_example()).expect("Δ");
    let report = compare_fingerprints(
        &fingerprint_constraint_system(&gamma.library.0),
        &fingerprint_constraint_system(&delta.library.0),
    );
    assert!(report.is_full_match(), "{:?}", report.mismatched_fields());
}

#[test]
fn the_report_and_the_fragments_cover_every_step() {
    if !bliss_available() {
        return;
    }
    let gamma = study(&gamma_example()).expect("Γ");
    let delta = study(&delta_example()).expect("Δ");
    let prime = study(&gamma_prime_example()).expect("Γ'");
    let (report, tex) = render(&gamma, &delta, &prime);
    for part in [
        "bliss input (DIMACS)",
        "Canonical graph",
        "Canonical renaming σ",
        "Canonical form",
        "Canonical constraint system",
        "key tuple",
    ] {
        assert!(report.contains(part), "{part:?} not in the report");
    }
    for part in [
        "subsec:canon:term",
        "subsec:canon:graph",
        "subsec:canon:form",
        "Automorphisms Need No Tie-Breaking",
        "From Forms to Keys",
        "subsec:canon:impl",
        r"\begin{tikzpicture}",
    ] {
        assert!(tex.contains(part), "{part:?} not in the fragments");
    }
}
