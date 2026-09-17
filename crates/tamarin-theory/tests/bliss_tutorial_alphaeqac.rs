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
//! logical content, different node-id numbering/solve order.
//!
//! Deliberately drives `bliss_proc`'s lower-level pipeline
//! (`graph_part_to_dimacs` + `run_bliss` + `apply_labeling`) instead of
//! the `canonicalize` convenience wrapper, so this test can inspect
//! `BlissResult::generators` directly: comparing each system's OWN
//! bliss-canonical labeling is only a valid $\alphaeqac$ check when the
//! graph has NO non-trivial automorphism (see `CanonicalGraph`'s own doc
//! comment — "minimum over automorphisms" is explicitly out of scope for
//! this stage). Rather than taking that on faith from a comment ("per
//! manual inspection"), this test asserts `generators.is_empty()` for
//! BOTH systems, so a future change that introduces symmetry into the
//! extracted graph (e.g. a new same-colored, interchangeable vertex
//! pair) fails LOUDLY here instead of silently reporting a false
//! `CanonicalGraph` mismatch (or worse, a false match) for the wrong
//! reason.
//!
//! Skips (via `bliss_available()`'s own panic-unless-opted-out gate) if
//! `bliss` is not on `PATH`/`$BLISS_PATH` and `TAM_ALLOW_NO_BLISS=1` is
//! set; otherwise a missing bliss fails loudly rather than reporting a
//! vacuous green.

use std::path::PathBuf;

use tamarin_theory::bliss_proc::{
    apply_labeling, bliss_available, canonical_edges, canonical_vertex_order, graph_part_to_dimacs,
    run_bliss,
};
use tamarin_theory::canon::canonicalize_graph_part;
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

    let dimacs_a =
        graph_part_to_dimacs(&part_a).unwrap_or_else(|e| panic!("graph_part_to_dimacs a: {e}"));
    let dimacs_b =
        graph_part_to_dimacs(&part_b).unwrap_or_else(|e| panic!("graph_part_to_dimacs b: {e}"));
    let result_a = run_bliss(&dimacs_a).unwrap_or_else(|e| panic!("run_bliss a: {e}"));
    let result_b = run_bliss(&dimacs_b).unwrap_or_else(|e| panic!("run_bliss b: {e}"));

    // The load-bearing assumption a plain `canonicalize` comparison would
    // otherwise take on faith: with a non-trivial automorphism group,
    // each system's own bliss-picked canonical labeling need not agree
    // with the other's even for truly alphaeqac graphs (bliss's internal
    // orbit tie-break has no reason to coincide across two differently-
    // numbered inputs) -- see this file's own module docs.
    assert!(
        result_a.generators.is_empty(),
        "system a's graph part has a non-trivial automorphism group ({} generator(s))",
        result_a.generators.len()
    );
    assert!(
        result_b.generators.is_empty(),
        "system b's graph part has a non-trivial automorphism group ({} generator(s))",
        result_b.generators.len()
    );

    let canon_a = apply_labeling(&part_a, &result_a.canonical_labeling);
    let canon_b = apply_labeling(&part_b, &result_b.canonical_labeling);

    assert_eq!(
        canon_a, canon_b,
        "the two systems' graph parts should be alphaeqac -- same canonical color \
         multiset and edge structure once each is relabeled by its own bliss \
         canonical labeling"
    );

    // The actual production canonical form (see `CanonicalGraph`'s own
    // doc comment for why the shape check above is only a diagnostic,
    // not sufficient on its own): put each system's vertices in its OWN
    // bliss canonical order, keeping full RuleACInst/GFact content this
    // time (not just colors -- see `canonical_vertex_order`), plus the
    // matching canonical-position edge set (`canonical_edges`), and
    // confirm the two systems' full graph parts -- rule instances,
    // action-formula facts, edges, and less-atoms (which enter the graph
    // via `LessRelation` vertices/edges, not a separate mechanism) --
    // canonize to the IDENTICAL term.
    let ordered_a = canonical_vertex_order(&part_a, &result_a.canonical_labeling);
    let ordered_b = canonical_vertex_order(&part_b, &result_b.canonical_labeling);
    let edges_a = canonical_edges(&part_a, &result_a.canonical_labeling);
    let edges_b = canonical_edges(&part_b, &result_b.canonical_labeling);
    assert_eq!(
        canonicalize_graph_part(&ordered_a, &edges_a),
        canonicalize_graph_part(&ordered_b, &edges_b),
        "the two systems' full graph parts should canonize to the identical term \
         once each is walked in its own bliss canonical order"
    );
}
