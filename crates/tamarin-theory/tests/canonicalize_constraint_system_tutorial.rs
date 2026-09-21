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
//! Skips (via `bliss_available()`'s own panic-unless-opted-out gate) if
//! `bliss` is not on `PATH`/`$BLISS_PATH` and `TAM_ALLOW_NO_BLISS=1` is
//! set; otherwise a missing bliss fails loudly rather than reporting a
//! vacuous green.

use std::path::PathBuf;

use tamarin_theory::bliss_proc::bliss_available;
use tamarin_theory::canon::canonicalize_constraint_system;
use tamarin_theory::elaborate::{elaborate, set_user_funs_for_theory};
use tamarin_theory::system_import::system_from_json;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join(name)
}

fn tutorial_theory_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tamarin-prover/examples/Tutorial.spthy")
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

    let src = std::fs::read_to_string(tutorial_theory_path())
        .unwrap_or_else(|e| panic!("read Tutorial.spthy: {e}"));
    let parsed = tamarin_parser::parse_theory(&src, &[]).expect("parse Tutorial.spthy");
    // `canonicalize_constraint_system` -> `extract_graph_part` ->
    // `collect_action_atoms` -> `gfact_to_fact`/`fact_to_lnfact` needs the
    // theory's signature installed in the thread-local `USER_FUNS`
    // context for the whole call -- same requirement
    // `bliss_tutorial_alphaeqac.rs` documents.
    let _guard = set_user_funs_for_theory(&parsed);
    let elaborated = elaborate(&parsed).expect("elaborate Tutorial.spthy");

    let sys_a = load_system("tutorial_khu_client_system.json");
    let sys_b = load_system("tutorial_client_khu_system.json");

    let canon_a = canonicalize_constraint_system(&sys_a, &elaborated)
        .unwrap_or_else(|e| panic!("canonicalize_constraint_system(a): {e:?}"));
    let canon_b = canonicalize_constraint_system(&sys_b, &elaborated)
        .unwrap_or_else(|e| panic!("canonicalize_constraint_system(b): {e:?}"));

    assert_eq!(
        canon_a, canon_b,
        "the two systems' full CanonicalSystems (graph part + formulas + \
         solved_formulas + lemmas + eq_store + subterm_store + \
         source_kind/side) should be identical -- not just their graph \
         parts, which `bliss_tutorial_alphaeqac.rs` already confirms \
         separately"
    );
}
