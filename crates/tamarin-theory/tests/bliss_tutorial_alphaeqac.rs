// Currently GPL 3.0 until granted permission by the upstream authors
// of the tamarin-prover sources this file cites; list them with:
//   scripts/gen_license_headers.py --authors <this file>

//! End-to-end $\alphaeqac$ check, straight from real captured proof
//! states: `examples/tutorial_khu_client_system.json` and
//! `examples/tutorial_client_khu_system.json` are two constraint
//! systems captured from the HS interactive session against
//! `tamarin-prover/examples/Tutorial.spthy`, reached by solving the
//! same two goals ("khu"/coerce-then-`Client_1`, vs `Client_1`-then-
//! "khu") in a DIFFERENT order. They are believed $\alphaeqac$ — same
//! logical content, different node-id numbering/solve order — and since
//! (per manual inspection) the resulting graph has no non-trivial
//! automorphism, `bliss_proc::canonicalize` should map both to the
//! exact same `CanonicalGraph`, with no "minimum over automorphisms"
//! search needed (out of scope for this stage anyway — see
//! `bliss_proc`'s own module docs).
//!
//! Skips (via `bliss_available()`'s own panic-unless-opted-out gate) if
//! `bliss` is not on `PATH`/`$BLISS_PATH` and `TAM_ALLOW_NO_BLISS=1` is
//! set; otherwise a missing bliss fails loudly rather than reporting a
//! vacuous green.

use std::path::PathBuf;

use tamarin_theory::bliss_proc::{bliss_available, canonicalize};
use tamarin_theory::canon_graph::extract_graph_part;
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
fn tutorial_khu_client_and_client_khu_systems_canonicalize_identically() {
    if !bliss_available() {
        return;
    }

    let src = std::fs::read_to_string(tutorial_theory_path())
        .unwrap_or_else(|e| panic!("read Tutorial.spthy: {e}"));
    let parsed = tamarin_parser::parse_theory(&src, &[]).expect("parse Tutorial.spthy");
    let _guard = set_user_funs_for_theory(&parsed);
    let elaborated = elaborate(&parsed).expect("elaborate Tutorial.spthy");

    let sys_a = load_system("tutorial_khu_client_system.json");
    let sys_b = load_system("tutorial_client_khu_system.json");

    let part_a = extract_graph_part(&sys_a, &elaborated);
    let part_b = extract_graph_part(&sys_b, &elaborated);

    // Sanity check before asking bliss anything: if the two graph parts
    // don't even have the same SHAPE (vertex/edge counts), they cannot
    // possibly be isomorphic, and the failure is much easier to read
    // here than as a `CanonicalGraph` mismatch below.
    assert_eq!(
        part_a.vertices.len(),
        part_b.vertices.len(),
        "vertex count mismatch -- the two systems don't even have the same graph SHAPE"
    );
    assert_eq!(
        part_a.edges.len(),
        part_b.edges.len(),
        "edge count mismatch -- the two systems don't even have the same graph SHAPE"
    );

    let canon_a = canonicalize(&part_a).unwrap_or_else(|e| panic!("canonicalize a: {e}"));
    let canon_b = canonicalize(&part_b).unwrap_or_else(|e| panic!("canonicalize b: {e}"));

    assert_eq!(
        canon_a, canon_b,
        "the two systems' graph parts should be alphaeqac -- same canonical color \
         multiset and edge structure once each is relabeled by its own bliss \
         canonical labeling"
    );
}
