// Currently GPL 3.0 until granted permission by the upstream authors
// of the tamarin-prover sources this file cites; list them with:
//   scripts/gen_license_headers.py --authors <this file>

//! Shared setup for this crate's canonicalization integration tests: the
//! Tutorial theory the captured system fixtures come from, and its signature.

use std::path::{Path, PathBuf};

use tamarin_term::maude_sig::MaudeSig;

/// Read, parse and elaborate `theory_path` for its signature, which
/// `system_from_json` parses a captured system under. No maude is needed.
pub fn theory_signature(theory_path: &Path) -> MaudeSig {
    let source = std::fs::read_to_string(theory_path).expect("read theory");
    let parsed = tamarin_parser::parse_theory(&source, &[]).expect("parse theory");
    tamarin_theory::elaborate::elaborate(&parsed)
        .expect("elaborate")
        .signature
}

pub fn tutorial_theory_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tamarin-prover/examples/Tutorial.spthy")
}
