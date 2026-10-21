// Currently GPL 3.0 until granted permission by the upstream authors
// of the tamarin-prover sources this file cites; list them with:
//   scripts/gen_license_headers.py --authors <this file>

//! End-to-end whole-system canonicalization check, against the same two
//! real captured proof states `bliss_tutorial_alphaeqac.rs` validates at
//! the term-graph level: `examples/tutorial_khu_client_system.json` and
//! `examples/tutorial_client_khu_system.json` -- two constraint systems
//! reached by solving the same two goals in a DIFFERENT order, believed
//! $\alphaeqac$.
//!
//! Drives `canon::canonicalize_constraint_system` and asserts the two
//! systems' full `CanonicalSystem`s are identical. Both fixtures happen to
//! have an empty `eqStore` and `subtermStore`, so this does NOT exercise
//! `eq_store.conj`'s per-alternative range-term canonicalization (unit-tested
//! in `canon.rs` and `canon_term_graph_tests.rs`).
//!
//! Also fingerprints both `CanonicalSystem`s (`canon_fingerprint`) and
//! checks the field-by-field match report, not just `CanonicalSystem`'s
//! own `PartialEq` -- real end-to-end confirmation that fingerprinting a
//! genuinely $\alphaeqac$ pair of real captured systems reports a full,
//! all-fields match, with a much more targeted failure message
//! (which SPECIFIC field(s) diverged) than a bare struct-equality
//! assertion would give if this pair ever stopped matching.
//!
//! Skips (via `bliss_available()`'s own panic-unless-opted-out gate) if
//! `bliss` is not on `PATH`/`$BLISS_PATH` and `TAM_ALLOW_NO_BLISS=1` is
//! set; otherwise a missing bliss fails loudly rather than reporting a
//! vacuous green.

use std::path::PathBuf;

use tamarin_theory::bliss_proc::bliss_available;
use tamarin_theory::canon::canonicalize_constraint_system;
use tamarin_theory::canon_fingerprint::{compare_fingerprints, fingerprint_constraint_system};
use tamarin_theory::system_import::system_from_json;

mod common;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join(name)
}

fn load_system(
    fixture_name: &str,
    sig: &tamarin_term::maude_sig::MaudeSig,
) -> tamarin_theory::constraint::system::System {
    let text = std::fs::read_to_string(fixture(fixture_name))
        .unwrap_or_else(|e| panic!("read {fixture_name}: {e}"));
    let json: serde_json::Value =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {fixture_name} as JSON: {e}"));
    system_from_json(&json, sig).unwrap_or_else(|e| panic!("system_from_json({fixture_name}): {e}"))
}

#[test]
fn tutorial_khu_client_and_client_khu_systems_canonicalize_identically_as_whole_systems() {
    if !bliss_available() {
        return;
    }

    // Reconstructing and canonicalizing the systems needs Tutorial.spthy's
    // signature (`h`/`aenc`/`adec`/`pk`) installed.
    let sig = common::theory_signature(&common::tutorial_theory_path());

    let sys_a = load_system("tutorial_khu_client_system.json", &sig);
    let sys_b = load_system("tutorial_client_khu_system.json", &sig);

    let canon_a = canonicalize_constraint_system(&sys_a)
        .unwrap_or_else(|e| panic!("canonicalize_constraint_system(a): {e:?}"));
    let canon_b = canonicalize_constraint_system(&sys_b)
        .unwrap_or_else(|e| panic!("canonicalize_constraint_system(b): {e:?}"));

    assert_eq!(
        canon_a, canon_b,
        "the two systems' full CanonicalSystems (graph part + formulas + \
         solved_formulas + lemmas + eq_store + subterm_store + goals + \
         source_kind/side) should be identical"
    );

    let fp_a = fingerprint_constraint_system(&canon_a);
    let fp_b = fingerprint_constraint_system(&canon_b);
    let report = compare_fingerprints(&fp_a, &fp_b);
    assert!(
        report.is_full_match(),
        "the two systems' CanonicalSystem fingerprints should match on \
         every field ({}/{}) -- fields that did NOT match: {:?}",
        report.matched_count(),
        report.total_count(),
        report.mismatched_fields()
    );
}

/// The same pair with the term graph's diagnostics: equal certificates of
/// the canonical graphs, and every automorphism generator a symmetry of the
/// system.
#[test]
fn tutorial_systems_have_equal_certificates_and_only_genuine_generators() {
    use tamarin_theory::canon_term_graph::{
        canonicalize_constraint_system_via_term_graph_with_labelling, TermGraphOptions,
    };
    if !bliss_available() {
        return;
    }
    let sig = common::theory_signature(&common::tutorial_theory_path());
    let options = TermGraphOptions {
        check_generators: true,
        certificate: true,
    };
    let canon = |name: &str| {
        canonicalize_constraint_system_via_term_graph_with_labelling(
            &load_system(name, &sig),
            options,
        )
        .unwrap_or_else(|e| panic!("term-graph canonicalization of {name}: {e:?}"))
    };
    let (form_a, _, stats_a) = canon("tutorial_khu_client_system.json");
    let (form_b, _, stats_b) = canon("tutorial_client_khu_system.json");
    assert_eq!(form_a, form_b);
    assert_eq!(stats_a.certificate, stats_b.certificate);
    assert_eq!(
        stats_a.non_genuine_generators + stats_b.non_genuine_generators,
        0
    );
}
