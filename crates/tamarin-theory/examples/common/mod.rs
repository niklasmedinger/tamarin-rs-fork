//! Shared setup for the dev example binaries: read → parse → elaborate a
//! theory file and boot a Maude handle on its full signature, corpus-root
//! resolution and `.spthy` collection for the corpus walkers, and the Tamarin
//! oracle runner with its wellformedness banner scanner.
//!
//! Lives in `examples/common/` (a subdirectory, so cargo does not treat it
//! as an example target); each example pulls it in with `mod common;`.
//! Individual examples use only a subset of these helpers, so each is
//! marked `#[allow(dead_code)]`.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use tamarin_term::maude_proc::MaudeHandle;

#[allow(dead_code)]
pub fn corpus_root() -> PathBuf {
    tamarin_test_support::corpus_root()
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
/// elaborated signature.  The elaborated signature carries the full
/// `MaudeSig` (aenc/pk/user-declared symbols); booting Maude on the default
/// sig would leave those symbols unparseable and corrupt any downstream
/// unification.
///
/// The binary comes from `tamarin_test_support::maude_path`, and a bare
/// `maude` for the OS to look up on `$PATH` is the last resort.  An example
/// never skips: whatever it resolved goes to `MaudeHandle::start`, and a
/// machine with no maude stops with that error.
#[allow(dead_code)]
pub fn load_theory_with_maude(theory_path: &str) -> (tamarin_theory::theory::Theory, MaudeHandle) {
    let source = std::fs::read_to_string(theory_path).expect("read theory");
    let parsed = tamarin_parser::parse_theory(&source, &[]).expect("parse theory");
    let elaborated = tamarin_theory::elaborate::elaborate(&parsed).expect("elaborate");
    let maude_path = tamarin_test_support::maude_path().unwrap_or_else(|| "maude".to_string());
    let maude = MaudeHandle::start(&maude_path, elaborated.signature.clone()).expect("start maude");
    (elaborated, maude)
}

/// Run `bin` (a `tamarin-prover` binary) on `path` with `flags` and return
/// the set of wellformedness topics it emits, or None if the process fails
/// to launch. stdout and stderr are concatenated before scanning.
#[allow(dead_code)]
pub fn run_tamarin(bin: &str, path: &Path, flags: &[String]) -> Option<BTreeSet<String>> {
    let mut cmd = Command::new(bin);
    for f in flags {
        cmd.arg(f);
    }
    cmd.arg(path);
    let out = cmd.output().ok()?;
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    Some(extract_topics(&combined))
}

/// A wellformedness topic header is a non-blank line underlined by a line of
/// nothing but `=` characters; banner sections that share that shape (the
/// `theory …` / `analyzed:` / version headers) are filtered out by name.
#[allow(dead_code)]
pub fn extract_topics(s: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut prev: Option<&str> = None;
    for line in s.lines() {
        if !line.is_empty()
            && line.chars().all(|c| c == '=')
            && let Some(p) = prev
        {
            let p = p.trim();
            if !p.is_empty() {
                // Filter banner lines that aren't actual topics.
                if !p.starts_with("analyzed:")
                    && !p.starts_with("summary of summaries")
                    && !p.contains("Tamarin version")
                    && !p.contains("Maude version")
                    && !p.starts_with("theory ")
                    && !p.starts_with("Generated from:")
                    && !p.starts_with("Compiled at")
                {
                    out.insert(p.to_string());
                }
            }
        }
        prev = Some(line);
    }
    out
}

/// The maude binary the examples start: [`tamarin_test_support::maude_path`],
/// else `maude` on `PATH`.
#[allow(dead_code)]
pub fn maude_binary() -> String {
    tamarin_test_support::maude_path().unwrap_or_else(|| "maude".to_string())
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
    pub theory: tamarin_theory::theory::Theory,
    /// The signature every Maude process of the theory is started on: the
    /// elaborated one, taken before translation and before closing joins the
    /// NDC verdicts into it (run.rs `run_batch`'s `maude_sig`).
    pub maude_sig: tamarin_term::maude_sig::MaudeSig,
    /// `--auto-sources` or the in-file `configuration:` block's.
    pub auto_sources: bool,
}

/// A theory after [`try_load_theory_for_proving`]: closed as far as proving
/// depends on it, with the prover session every lemma's proof starts from.
#[allow(dead_code)]
pub struct Loaded {
    pub session: tamarin_theory::prove::ProverSession,
    /// The translated rules, before closing removes those without variants.
    pub translated_rules: usize,
    pub auto_sources: bool,
    pub is_sapic: bool,
}

/// Loads `theory_path` the way `tamarin-rs --prove` does up to the point it
/// needs Maude: the stages of `run.rs::run_batch` and
/// `TheoryPipeline::translate_theory` (mirrored for the web in
/// `tamarin-server/src/theory_io.rs::load_from_source`). Wellformedness
/// reports are left out: proving doesn't read them.
#[allow(dead_code)]
pub fn try_load_theory_translated(
    theory_path: &str,
    opts: &LoadOpts,
) -> Result<Translated, String> {
    let source = std::fs::read_to_string(theory_path).map_err(|e| format!("read theory: {e}"))?;
    // `-D` flags and `#include`s resolved against the theory's directory
    // (run.rs `run_batch`, `parse_theory_with_base`).
    let flags: Vec<&str> = opts.defines.iter().map(String::as_str).collect();
    let base_dir = Path::new(theory_path).parent().map(Path::to_path_buf);
    let parsed = tamarin_parser::parse_theory_with_base(&source, &flags, base_dir)
        .map_err(|e| format!("parse theory: {e}"))?;
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
    let mut theory = tamarin_theory::elaborate::elaborate_with_in_file(&parsed, theory_path)
        .map_err(|e| format!("elaborate: {}", e.message))?;
    let maude_sig = theory.signature.clone();
    // SAPIC and accountability translation (run.rs
    // `TheoryPipeline::translate_theory`).
    let user_set_heuristic = !theory.heuristic.is_empty();
    tamarin_sapic::apply::apply_sapic(&mut theory, user_set_heuristic)
        .map_err(|e| format!("SAPIC translation: {}", e.message))?;
    tamarin_accountability::translate(&mut theory)
        .map_err(|e| format!("accountability translation: {e}"))?;
    Ok(Translated {
        theory,
        maude_sig,
        auto_sources: opts.auto_sources || config.auto_sources,
    })
}

/// [`try_load_theory_translated`], then the parts of closing the theory
/// proving depends on, in run.rs's order (`check_translated_theory`, then
/// `close_translated_theory`): start Maude, prepare the rules (variants,
/// loop breakers), run the NDC pass for the intruder-rule cache and join its
/// verdicts into the signature, add the auto-sources lemma if asked for, and
/// build the prover session. Partial evaluation and the derivation checks
/// are left out: the first is opt-in, the second only reports.
#[allow(dead_code)]
pub fn try_load_theory_for_proving(theory_path: &str, opts: &LoadOpts) -> Result<Loaded, String> {
    let Translated {
        mut theory,
        maude_sig,
        auto_sources,
    } = try_load_theory_translated(theory_path, opts)?;
    let translated_rules = theory.rules().count();
    let maude_path = maude_binary();
    let maude = MaudeHandle::start(&maude_path, maude_sig)
        .map_err(|e| format!("start maude ({maude_path}): {e:?}"))?;
    let parameters = tamarin_theory::constraint::solver::sources::IntegerParameters::default();
    tamarin_theory::tools::rule_variants::prepare_theory_rules(&mut theory, &maude, None, true)
        .map_err(|e| format!("rule variants: {e}"))?;
    let checked = tamarin_theory::close_rule::check_close_intr_rule(
        &maude,
        Some(theory.name.as_str()),
        theory.options.deduction_chain_check,
        &theory.intruder_rules,
        parameters,
    )
    .map_err(|e| format!("NDC check: {e}"))?;
    let ndc_cache: tamarin_theory::constraint::solver::context::IntrRuleCache =
        checked.cache.into();
    for f in &checked.ndc_funs {
        let sig = std::mem::take(&mut theory.signature);
        theory.signature = sig.join_ndc_in_sig(*f, tamarin_term::function_symbols::NdcState::IsNdc);
    }
    if auto_sources {
        tamarin_theory::auto_sources::apply_auto_sources(
            &mut theory,
            maude.clone(),
            None,
            Some(&ndc_cache),
            parameters,
        )
        .map_err(|e| format!("auto-sources: {e}"))?;
    }
    let is_sapic = theory.is_sapic();
    let session = tamarin_theory::prove::ProverSession::build(
        std::sync::Arc::new(theory),
        maude,
        tamarin_theory::prove::ProverSessionOptions {
            ndc_cache: Some(ndc_cache),
            parameters,
            loop_breakers_prepared: true,
            ..Default::default()
        },
    )
    .map_err(|e| format!("prover session: {e}"))?;
    Ok(Loaded {
        session,
        translated_rules,
        auto_sources,
        is_sapic,
    })
}
