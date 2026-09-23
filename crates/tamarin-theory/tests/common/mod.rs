// Currently GPL 3.0 until granted permission by the upstream authors
// of the tamarin-prover sources this file cites; list them with:
//   scripts/gen_license_headers.py --authors <this file>

//! Shared setup for this crate's integration tests: read → parse →
//! elaborate a theory file and boot a Maude handle on its full
//! signature, plus the `$MAUDE_PATH` resolution probe every maude-gated
//! integration test in this file needs.
//!
//! `crates/tamarin-theory/src/test_maude.rs` is the crate-internal
//! sibling of [`maude_path`] below — an integration test cannot see a
//! `#[cfg(test)]` module of the library it links, so this is a mirrored
//! copy (per that file's own discipline-scan test, which enumerates
//! every sanctioned copy across the workspace). Keep the two in sync.

use std::path::{Path, PathBuf};

use tamarin_term::maude_proc::MaudeHandle;

/// Absolute maude locations probed before `PATH` is walked.
const MAUDE_CANDIDATES: [&str; 2] = ["/usr/local/bin/maude", "/usr/bin/maude"];

/// Last resort, after `PATH`: the linuxbrew prefix this project's maude
/// lives under on the development box, which is deliberately not on
/// `PATH`.
const MAUDE_LINUXBREW: &str = "/home/linuxbrew/.linuxbrew/bin/maude";

/// The maude every maude-gated test in this crate's `tests/` directory
/// runs against: `$MAUDE_PATH`, else the first existing
/// [`MAUDE_CANDIDATES`] entry, else a `PATH` walk, else
/// [`MAUDE_LINUXBREW`].
///
/// Resolving NOTHING is a misconfiguration, not a reason to skip: every
/// maude-gated test opens with `let Some(path) = common::maude_path()
/// else { return };`, so a `None` here reports the same green run with
/// and without maude installed. Panic instead — unless
/// `TAM_ALLOW_NO_MAUDE=1` explicitly asks for the old silent skip. A
/// `MAUDE_PATH` naming a file that does not exist is the same
/// misconfiguration and panics too.
pub fn maude_path() -> Option<String> {
    if let Ok(p) = std::env::var("MAUDE_PATH") {
        assert!(
            std::path::Path::new(&p).exists(),
            "MAUDE_PATH={p} does not exist; unset it to fall back to \
             {MAUDE_CANDIDATES:?} / PATH / {MAUDE_LINUXBREW}, or point it at a \
             real maude — skipping every maude-gated test here would report \
             green vacuously"
        );
        return Some(p);
    }
    if let Some(c) = MAUDE_CANDIDATES
        .iter()
        .find(|c| std::path::Path::new(c).exists())
    {
        return Some((*c).to_string());
    }
    if let Some(p) = std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|d| d.join("maude"))
            .find(|p| p.is_file())
    }) {
        return Some(p.to_string_lossy().into_owned());
    }
    if std::path::Path::new(MAUDE_LINUXBREW).exists() {
        return Some(MAUDE_LINUXBREW.to_string());
    }
    if std::env::var("TAM_ALLOW_NO_MAUDE").as_deref() == Ok("1") {
        return None;
    }
    panic!(
        "no maude found: MAUDE_PATH unset, none of {MAUDE_CANDIDATES:?} exist, \
         nothing named `maude` on PATH, and no {MAUDE_LINUXBREW}. Every \
         maude-gated test in this crate's `tests/` directory would skip and \
         the run would be green having proved nothing. Install maude, set \
         MAUDE_PATH, or set TAM_ALLOW_NO_MAUDE=1 to accept the silent skip."
    );
}

/// Read, parse, and elaborate `theory_path`, then start Maude on the
/// elaborated signature — mirrors `examples/common/mod.rs`'s helper of
/// the same name/shape. `None` when [`maude_path`] resolves nothing
/// (`TAM_ALLOW_NO_MAUDE=1`); callers skip the same way every other
/// maude-gated test in this crate does.
#[allow(dead_code)]
pub fn load_theory_with_maude(
    theory_path: &Path,
) -> Option<(
    tamarin_parser::ast::Theory,
    tamarin_theory::theory::Theory,
    MaudeHandle,
)> {
    let path = maude_path()?;
    let source = std::fs::read_to_string(theory_path).expect("read theory");
    let parsed = tamarin_parser::parse_theory(&source, &[]).expect("parse theory");
    let elaborated = tamarin_theory::elaborate::elaborate(&parsed).expect("elaborate");
    let maude = MaudeHandle::start(&path, elaborated.signature.maude_sig.clone())
        .unwrap_or_else(|e| panic!("maude at {path} failed to start: {e:?}"));
    Some((parsed, elaborated, maude))
}

#[allow(dead_code)]
pub fn tutorial_theory_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tamarin-prover/examples/Tutorial.spthy")
}
