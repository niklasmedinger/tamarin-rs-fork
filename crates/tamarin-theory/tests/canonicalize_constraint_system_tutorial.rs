// Currently GPL 3.0 until granted permission by the upstream authors
// of the tamarin-prover sources this file cites; list them with:
//   scripts/gen_license_headers.py --authors <this file>

//! End-to-end whole-system canonicalization check (Stage G), against the
//! same two real captured proof states `bliss_tutorial_alphaeqac.rs`
//! already validates at the graph-part level:
//! `examples/tutorial_khu_client_system.json` and
//! `examples/tutorial_client_khu_system.json` -- two constraint systems
//! reached by solving the same two goals in a DIFFERENT order, believed
//! $\alphaeqac$.
//!
//! This test drives `canon::canonicalize_constraint_system` -- the Stage G
//! driver that extends graph-part canonization through
//! formulas/solved_formulas/lemmas/eq_store/subterm_store -- and asserts
//! the two systems' full `CanonicalSystem`s are identical, not just their
//! graph parts. Both fixtures happen to have an empty `eqStore` and
//! `subtermStore`, so this does NOT exercise `eq_store.conj`'s
//! per-alternative range-term canonicalization
//! (`canon::canonicalize_eq_disj_alternative`, fully implemented but only
//! unit-tested directly so far, not against real captured data -- see
//! `canon.rs`'s own test module) or a multi-survivor Stage F tie-break
//! (both systems' graph parts have a trivial automorphism group, per
//! `bliss_tutorial_alphaeqac.rs`).
//!
//! Also fingerprints both `CanonicalSystem`s (`canon_fingerprint`) and
//! checks the field-by-field match report, not just `CanonicalSystem`'s
//! own `PartialEq` -- real end-to-end confirmation that fingerprinting a
//! genuinely $\alphaeqac$ pair of real captured systems reports a full,
//! all-twelve-fields match, with a much more targeted failure message
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
use tamarin_theory::constraint::solver::context::ProofContext;
use tamarin_theory::elaborate::set_user_funs_for_theory;
use tamarin_theory::system_import::system_from_json;

mod common;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join(name)
}

fn load_system(fixture_name: &str) -> tamarin_theory::constraint::system::System {
    let text = std::fs::read_to_string(fixture(fixture_name))
        .unwrap_or_else(|e| panic!("read {fixture_name}: {e}"));
    let json: serde_json::Value =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {fixture_name} as JSON: {e}"));
    system_from_json(&json).unwrap_or_else(|e| panic!("system_from_json({fixture_name}): {e}"))
}

#[test]
fn tutorial_khu_client_and_client_khu_systems_canonicalize_identically_as_whole_systems() {
    if !bliss_available() {
        return;
    }

    // Tutorial.spthy declares its own functions (`h`/`aenc`/`adec`/`pk`),
    // so its real `IntrRuleCache` contains theory-specific Constr/Destr
    // rules alongside the fixed special ones -- building the `ColorTable`
    // `canonicalize_constraint_system` now takes needs a real maude
    // process, not just an elaborated `Theory` (see `canon_color.rs`'s
    // own "why this table takes..." doc section). A `ProofContext` is
    // the natural way to get one: it's the SAME table a real proof
    // search would use, built once at `ProofContext::new` time from its
    // own `rules`/`intruder_rules` (`ctx.color_table`).
    let Some((parsed, elaborated, maude)) =
        common::load_theory_with_maude(&common::tutorial_theory_path())
    else {
        return;
    };
    // `canonicalize_constraint_system` -> `extract_graph_part` ->
    // `collect_action_atoms` -> `gfact_to_fact`/`fact_to_lnfact` needs the
    // theory's signature installed in the thread-local `USER_FUNS`
    // context for the whole call -- same requirement
    // `bliss_tutorial_alphaeqac.rs` documents.
    let _guard = set_user_funs_for_theory(&parsed);
    let protocol_rules: Vec<tamarin_theory::theory::OpenProtoRule> =
        elaborated.rules().cloned().collect();
    let ctx = ProofContext::new(maude, protocol_rules);
    let colors = &ctx.color_table;

    let sys_a = load_system("tutorial_khu_client_system.json");
    let sys_b = load_system("tutorial_client_khu_system.json");

    let canon_a = canonicalize_constraint_system(&sys_a, colors)
        .unwrap_or_else(|e| panic!("canonicalize_constraint_system(a): {e:?}"));
    let canon_b = canonicalize_constraint_system(&sys_b, colors)
        .unwrap_or_else(|e| panic!("canonicalize_constraint_system(b): {e:?}"));

    assert_eq!(
        canon_a, canon_b,
        "the two systems' full CanonicalSystems (graph part + formulas + \
         solved_formulas + lemmas + eq_store + subterm_store + \
         source_kind/side) should be identical -- not just their graph \
         parts, which `bliss_tutorial_alphaeqac.rs` already confirms \
         separately"
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
