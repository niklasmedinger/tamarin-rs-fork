//! Shared setup for the dev example binaries: read → parse → elaborate a
//! theory file and boot a Maude handle on its full signature, plus
//! corpus-root resolution and `.spthy` collection for the corpus walkers.
//!
//! Lives in `examples/common/` (a subdirectory, so cargo does not treat it
//! as an example target); each example pulls it in with `mod common;`.
//! Individual examples use only a subset of these helpers, so each is
//! marked `#[allow(dead_code)]`.

use std::path::{Path, PathBuf};

use tamarin_term::maude_proc::MaudeHandle;

/// The examples corpus root: `$CORPUS_ROOT` if set, else the
/// `tamarin-prover/examples/` directory in the submodule, relative to this
/// crate's manifest.
#[allow(dead_code)]
pub fn corpus_root() -> PathBuf {
    std::env::var("CORPUS_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tamarin-prover/examples")
        })
}

/// Collect every `.spthy` file under `root`, sorted by path.
#[allow(dead_code)]
pub fn collect_spthy(root: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("spthy"))
        .map(|e| e.path().to_path_buf())
        .collect();
    files.sort();
    files
}

/// Read, parse, and elaborate `theory_path`, then start Maude on the
/// elaborated signature (`$MAUDE_PATH` overrides the binary, else `maude`
/// on `PATH`).  The elaborated signature carries the full `MaudeSig`
/// (aenc/pk/user-declared symbols); booting Maude on the default sig would
/// leave those symbols unparseable and corrupt any downstream unification.
#[allow(dead_code)]
pub fn load_theory_with_maude(
    theory_path: &str,
) -> (
    tamarin_parser::ast::Theory,
    tamarin_theory::theory::Theory,
    MaudeHandle,
) {
    try_load_theory_with_maude(theory_path).unwrap_or_else(|e| panic!("{e}"))
}

/// [`load_theory_with_maude`], returning which step failed instead of
/// panicking -- for batch runs, whose logs must say why a theory was skipped.
#[allow(dead_code)]
pub fn try_load_theory_with_maude(
    theory_path: &str,
) -> Result<
    (
        tamarin_parser::ast::Theory,
        tamarin_theory::theory::Theory,
        MaudeHandle,
    ),
    String,
> {
    let source = std::fs::read_to_string(theory_path).map_err(|e| format!("read theory: {e}"))?;
    let parsed =
        tamarin_parser::parse_theory(&source, &[]).map_err(|e| format!("parse theory: {e}"))?;
    let elaborated = tamarin_theory::elaborate::elaborate(&parsed)
        .map_err(|e| format!("elaborate: {}", e.message))?;
    let maude_path = maude_binary();
    let maude = MaudeHandle::start(&maude_path, elaborated.signature.maude_sig.clone())
        .map_err(|e| format!("start maude ({maude_path}): {e:?}"))?;
    Ok((parsed, elaborated, maude))
}

/// What `tamarin-rs` takes from its command line when loading a theory for
/// proving, as far as the loaded theory depends on it.
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct LoadOpts {
    /// `-D=<flag>` preprocessor defines (`#ifdef`).
    pub defines: Vec<String>,
    /// `--auto-sources` (OR-ed with the theory's own `configuration:` block).
    pub auto_sources: bool,
}

/// A theory after [`try_load_theory_translated`]: SAPIC processes and
/// accountability lemmas translated, exactly as `tamarin-rs --prove` sees
/// it before closing.
#[allow(dead_code)]
pub struct Translated {
    /// The TRANSLATED parser theory. `build_lemma_proof_context` (and the
    /// CLI's `ProverSession`) re-elaborate from this, so it must be the
    /// post-translation one.
    pub parsed: tamarin_parser::ast::Theory,
    pub elaborated: tamarin_theory::theory::Theory,
    /// `--auto-sources` or the in-file `configuration:` block's.
    pub auto_sources: bool,
    /// The user function-symbol bundle of the loaded theory (see
    /// `elaborate::set_user_funs_for_theory`). Guards restore the previous
    /// bundle on drop, so any guard installed later must be dropped first.
    pub user_funs_guard: tamarin_theory::elaborate::UserFunsForTheoryGuard,
}

/// A theory after [`try_load_theory_for_proving`]: translated and closed as
/// far as proving depends on it.
#[allow(dead_code)]
pub struct Loaded {
    pub translated: Translated,
    pub maude: MaudeHandle,
    /// The NDC-checked intruder-rule cache every `ProofContext` of the
    /// theory is built with.
    pub ndc_cache: tamarin_theory::constraint::solver::context::IntrRuleCache,
}

/// Loads `theory_path` the way `tamarin-rs --prove` does up to the point it
/// needs Maude: the stages of `run.rs::run_batch` and
/// `TheoryPipeline::translate_theory` (mirrored for the web in
/// `tamarin-server/src/theory_io.rs::load_from_source`). Wellformedness
/// reports are left out: proving doesn't read them.
#[allow(dead_code)]
pub fn try_load_theory_translated(theory_path: &str, opts: &LoadOpts) -> Result<Translated, String> {
    let source = std::fs::read_to_string(theory_path).map_err(|e| format!("read theory: {e}"))?;
    // `-D` flags and `#include`s resolved against the theory's directory
    // (run.rs `run_batch`, `parse_theory_with_base`).
    let flags: Vec<&str> = opts.defines.iter().map(String::as_str).collect();
    let base_dir = Path::new(theory_path).parent().map(Path::to_path_buf);
    let mut parsed = tamarin_parser::parse_theory_with_base(&source, &flags, base_dir)
        .map_err(|e| format!("parse theory: {e}"))?;
    // `_restrict(φ)` lifting, right after parsing (run.rs `run_batch`).
    tamarin_theory::rule_restriction::lift_rule_restrictions(&mut parsed)
        .map_err(|e| format!("_restrict expansion: {}", e.message))?;
    // The in-file `configuration:` block's `--auto-sources` (run.rs
    // `run_batch`, `configAutoSources`).
    let config = parsed
        .configuration
        .as_deref()
        .map(tamarin_theory::prove::parse_config_block)
        .unwrap_or_default();
    if let Some(msg) = &config.flag_error {
        return Err(format!("configuration block: {msg}"));
    }
    let mut elaborated =
        tamarin_theory::elaborate::elaborate(&parsed).map_err(|e| format!("elaborate: {}", e.message))?;
    elaborated.in_file = theory_path.to_string();
    // SAPIC and accountability translation (run.rs
    // `TheoryPipeline::translate_theory`), under the theory's user
    // function-symbol bundle.
    let user_funs_guard = tamarin_theory::elaborate::set_user_funs_for_theory(&parsed);
    let user_set_heuristic = !elaborated.heuristic.is_empty();
    tamarin_sapic::apply::apply_sapic(&mut parsed, &mut elaborated, user_set_heuristic)
        .map_err(|e| format!("SAPIC translation: {}", e.message))?;
    tamarin_accountability::translate(&mut parsed, &mut elaborated)
        .map_err(|e| format!("accountability translation: {e}"))?;
    Ok(Translated {
        parsed,
        elaborated,
        auto_sources: opts.auto_sources || config.auto_sources,
        user_funs_guard,
    })
}

/// [`try_load_theory_translated`], then the parts of closing the theory
/// proving depends on: start Maude on the translated signature, run the
/// NDC pass for the intruder-rule cache (run.rs
/// `TheoryPipeline::check_translated_theory`) and, if asked for, add the
/// auto-sources lemma (run.rs `TheoryPipeline::close_translated_theory`).
/// Partial evaluation and the derivation checks are left out.
#[allow(dead_code)]
pub fn try_load_theory_for_proving(theory_path: &str, opts: &LoadOpts) -> Result<Loaded, String> {
    let mut translated = try_load_theory_translated(theory_path, opts)?;
    let maude_path = maude_binary();
    let maude = MaudeHandle::start(&maude_path, translated.elaborated.signature.maude_sig.clone())
        .map_err(|e| format!("start maude ({maude_path}): {e:?}"))?;
    // run.rs silences the saturation trace until the close proper.
    tamarin_theory::constraint::solver::sources::set_show_saturation_steps(false);
    let checked = tamarin_theory::close_rule::check_close_intr_rule(
        &maude,
        None,
        translated.elaborated.options.deduction_chain_check,
    );
    let ndc_cache: tamarin_theory::constraint::solver::context::IntrRuleCache = checked.cache.into();
    if translated.auto_sources {
        tamarin_theory::auto_sources::apply_auto_sources(
            &mut translated.parsed,
            &mut translated.elaborated,
            maude.clone(),
            None,
            Some(&ndc_cache),
        );
    }
    Ok(Loaded {
        translated,
        maude,
        ndc_cache,
    })
}

/// The maude binary the examples start: `$MAUDE_PATH`, else `maude` on `PATH`.
#[allow(dead_code)]
pub fn maude_binary() -> String {
    std::env::var("MAUDE_PATH").unwrap_or_else(|_| "maude".to_string())
}
