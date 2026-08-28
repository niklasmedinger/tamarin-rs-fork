// Currently GPL 3.0 until granted permission by the upstream authors
// of the tamarin-prover sources this file cites; list them with:
//   scripts/gen_license_headers.py --authors <this file>

//! Loads a constraint-system JSON dump — the body of the HS web UI's
//! `GET /thy/trace/<idx>/system/*TheoryPath` route (see
//! `tamarin_theory::system_import`'s module docs for the exact schema and
//! how the round trip works) — and prints its graph part
//! (`tamarin_theory::canon_graph`) as Graphviz DOT to stdout.
//!
//! Usage:
//!   cargo run -p tamarin-theory --example system_json_to_graphviz -- \
//!       [--theory <model.spthy>] [<system.json> | -]
//!
//! With no JSON path (or `-`), reads from stdin — paste a copied dump and
//! press Ctrl-D. Pipe stdout into `dot -Tsvg -o out.svg` (or `-Tpng`, ...)
//! to actually render it.
//!
//! ## Function symbols
//!
//! Every term/fact in the dump is plain Tamarin surface syntax, and
//! parsing it back (`system_import::system_from_json`) needs to resolve
//! each function symbol's real arity/privacy/AC-ness — which isn't
//! recoverable from the dump text alone, only from the source model's
//! SIGNATURE (`functions:`/`builtins:`/`equations:` declarations). `Fr`,
//! `In`, `Out`, `!KU`/`!KD`, and pairing (`<a, b>`) are baked into the
//! term algebra independent of any theory declaration (confirmed by
//! `system_import`'s own tests, which use a totally empty
//! `theory T begin end` and still parse those fine) — so a dump of a
//! simple/example-only system will parse with NO `--theory` flag at all.
//! Anything using a custom function or a `builtins:` block (symmetric/
//! asymmetric encryption, signatures, Diffie-Hellman, XOR, ...) needs the
//! REAL model's signature installed first, or parsing fails with an
//! elaboration error. Since the dump was taken from a live HS session
//! already running on some `<model.spthy>`, that same file is always
//! available to point `--theory` at — reusing it is simpler and lower-risk
//! than teaching the HS endpoint to also serialize its own signature
//! (duplicate information that could drift from the model file, for a gap
//! this flag already closes with zero HS-side changes).

// Example/dev tool: prints DOT to stdout by design (pipe into `dot
// -Tsvg`) — allow the `disallowed_macros` convention freeze (stdout is
// normally reserved as the byte-parity surface) for this example binary,
// matching `dump_proof.rs`'s own precedent.
#![allow(clippy::disallowed_macros)]

use std::io::Read;

use tamarin_parser::parser::parse_theory;
use tamarin_theory::canon_graph::{extract_graph_part, to_graphviz};
use tamarin_theory::elaborate::set_user_funs_for_theory;
use tamarin_theory::system_import::system_from_json;

fn main() {
    let (theory_path, json_path) = parse_args();

    let theory_src = match &theory_path {
        Some(p) => read_file_or_die(p, "theory file"),
        None => "theory T begin\nend".to_string(),
    };
    let parsed_theory = parse_theory(&theory_src, &[]).unwrap_or_else(|e| {
        die(&format!("failed to parse theory: {e}"));
    });
    // Kept alive for the whole run: `system_from_json` reads this
    // scoped-global signature context via `elaborate::term_to_lnterm`/
    // `fact_to_lnfact` internally, per `system_import`'s documented
    // precondition.
    let _guard = set_user_funs_for_theory(&parsed_theory);
    if theory_path.is_none() {
        eprintln!(
            "note: no --theory given -- using a minimal built-ins-only signature. \
             Pass --theory <model.spthy> if the dump references custom functions \
             or a builtins: block (see this example's module docs)."
        );
    }

    let json_text = match json_path.as_deref() {
        None | Some("-") => {
            let mut s = String::new();
            std::io::stdin()
                .read_to_string(&mut s)
                .unwrap_or_else(|e| die(&format!("failed to read JSON from stdin: {e}")));
            s
        }
        Some(p) => read_file_or_die(p, "JSON file"),
    };
    let json: serde_json::Value = serde_json::from_str(&json_text)
        .unwrap_or_else(|e| die(&format!("input is not valid JSON: {e}")));

    let sys = system_from_json(&json).unwrap_or_else(|e| {
        eprintln!("failed to reconstruct System from JSON: {e}");
        eprintln!(
            "hint: if the error above mentions parsing or elaborating a term/fact, \
             the dump likely references a function symbol not in the installed \
             signature -- pass --theory <model.spthy> for the model this system \
             was extracted from."
        );
        std::process::exit(1);
    });

    let part = extract_graph_part(&sys);
    print!("{}", to_graphviz(&part));
}

fn parse_args() -> (Option<String>, Option<String>) {
    let mut theory_path: Option<String> = None;
    let mut json_path: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--theory" => {
                theory_path = Some(
                    args.next()
                        .unwrap_or_else(|| die("--theory requires a path argument")),
                );
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => {
                if json_path.is_some() {
                    eprintln!("unexpected extra argument: {other}");
                    print_usage();
                    std::process::exit(2);
                }
                json_path = Some(other.to_string());
            }
        }
    }
    (theory_path, json_path)
}

fn read_file_or_die(path: &str, what: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| die(&format!("failed to read {what} {path:?}: {e}")))
}

fn die(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(1);
}

fn print_usage() {
    eprintln!(
        "usage: system_json_to_graphviz [--theory <model.spthy>] [<system.json> | -]\n\
         \n\
         Reads a constraint-system JSON dump (the HS web UI's \
         GET /thy/trace/<idx>/system/*TheoryPath response body) and prints \
         its graph part as Graphviz DOT to stdout.\n\
         \n\
         With no JSON path (or '-'), reads from stdin -- paste the dump \
         and press Ctrl-D.\n\
         \n\
         Pass --theory <model.spthy> (the same file the HS session was \
         loaded on) whenever the system references custom functions or a \
         builtins: block beyond Fr/In/Out/!KU and pairing."
    );
}
