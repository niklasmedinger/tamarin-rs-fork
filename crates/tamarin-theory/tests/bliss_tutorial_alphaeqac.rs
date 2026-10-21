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
//! Deliberately drives the term graph's lower-level pipeline
//! (`system_to_term` + `term_to_graph` + `to_dimacs` + `run_bliss`) instead
//! of `canonicalize_constraint_system`, checking the graph layer on its own:
//! the two systems' term graphs must be isomorphic, i.e. each relabeled by
//! its own bliss canonical labeling gives the same vertex keys and edges.
//! `canonicalize_constraint_system_tutorial.rs` checks the canonical forms.
//!
//! Skips (via `bliss_available()`'s own panic-unless-opted-out gate) if
//! `bliss` is not on `PATH`/`$BLISS_PATH` and `TAM_ALLOW_NO_BLISS=1` is
//! set; otherwise a missing bliss fails loudly rather than reporting a
//! vacuous green.

use std::collections::BTreeSet;
use std::path::PathBuf;

use tamarin_theory::bliss_proc::{bliss_available, run_bliss, to_dimacs};
use tamarin_theory::canon_term_graph::{system_to_term, term_to_graph, TermGraph, VKey};
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

/// `graph` relabeled by its own bliss canonical labeling: the vertex keys in
/// canonical order and the canonical edges.
fn canonical_graph(graph: &TermGraph) -> (Vec<VKey>, BTreeSet<(usize, usize)>) {
    let dimacs = to_dimacs(&graph.colors, &graph.edges).unwrap_or_else(|e| panic!("dimacs: {e}"));
    let labeling = run_bliss(&dimacs)
        .unwrap_or_else(|e| panic!("run_bliss: {e}"))
        .canonical_labeling;
    let mut keys = vec![None; graph.keys.len()];
    for (v, key) in graph.keys.iter().enumerate() {
        keys[labeling.image_of(v)] = Some(key.clone());
    }
    let keys = keys
        .into_iter()
        .map(|k| k.expect("the labeling is a bijection"))
        .collect();
    let edges = graph
        .edges
        .iter()
        .map(|&(s, t)| (labeling.image_of(s), labeling.image_of(t)))
        .collect();
    (keys, edges)
}

#[test]
fn tutorial_khu_client_and_client_khu_systems_have_isomorphic_term_graphs() {
    if !bliss_available() {
        return;
    }
    // Reconstructing the systems' terms needs Tutorial.spthy's signature
    // (`h`/`aenc`/`adec`/`pk`) installed.
    let sig = common::theory_signature(&common::tutorial_theory_path());

    let graph_a = term_to_graph(&system_to_term(&load_system(
        "tutorial_khu_client_system.json",
        &sig,
    )));
    let graph_b = term_to_graph(&system_to_term(&load_system(
        "tutorial_client_khu_system.json",
        &sig,
    )));

    // Sanity check before asking bliss anything: graphs of different sizes
    // cannot be isomorphic, and the failure is easier to read here.
    assert_eq!(
        graph_a.keys.len(),
        graph_b.keys.len(),
        "vertex count mismatch"
    );
    assert_eq!(
        graph_a.edges.len(),
        graph_b.edges.len(),
        "edge count mismatch"
    );
    assert_eq!(
        graph_a.free_vars.len(),
        graph_b.free_vars.len(),
        "free variable count mismatch"
    );

    assert_eq!(
        canonical_graph(&graph_a),
        canonical_graph(&graph_b),
        "the two systems' term graphs should be isomorphic -- the same keys and \
         edges once each is relabeled by its own bliss canonical labeling"
    );
}
