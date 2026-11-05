// Currently GPL 3.0; see README.md for licensing details.
// Derived from the upstream tamarin-prover sources referenced below.

//! A user `functions: h/2` together with `builtins: xor` must not change the
//! verdicts.
//!
//! The elaborator used to fold the arguments of a rule term `h(k, ~b)` into
//! `h(<k, ~b>)` because `h` is on a fixed list of unary builtin names,
//! although the theory declares `h` binary (HS `naryOpApp` folds by the
//! declared arity, Theory/Text/Parser/Term.hs).  Maude, told
//! `h : Msg Msg -> Msg`, answered "no parse for command" on the variant query
//! of rule `R` (`tamXCFUh(tamXCFUpair(..))`), and the run stopped with
//! `Maude error: parse error`.  A caller that reads the empty reply as "no
//! variants" instead loses the XOR variant in which `x XOR k` cancels to
//! `zero` and proves `never_zero`.
//!
//! tamarin-prover 1.11.0 on [`THEORY_H`]: `never_zero` falsified - found
//! trace, `some_zero` verified.  Renaming `h` to `g` ([`THEORY_G`]) gave
//! these verdicts before the fix too.

mod common;

use common::{maude_available, normalize_stdout, strip_maude_banner};

/// The temp subdirectory this suite writes its theories to.
const TMP_DIR: &str = "tamarin_prover_user_h_arity";

/// The attacker learns `~k` from `Gen` and sends it to `R`, so `x XOR k`
/// becomes `zero`.  `h` only occurs in an action of `R`.
const THEORY_H: &str = r#"theory FalseProofH2Xor
begin
builtins: xor
functions: h/2
rule Gen: [Fr(~k)] --> [Out(~k), Key(~k)]
rule R: [Key(k), Fr(~b), In(x)] --[A(h(k,~b)), Got(x XOR k)]-> []
lemma never_zero: all-traces "All #i. Got(zero)@i ==> F"
lemma some_zero: exists-trace "Ex #i. Got(zero)@i"
end
"#;

/// [`THEORY_H`] with `h` renamed to `g`, a name the elaborator never folded.
const THEORY_G: &str = r#"theory FalseProofG2Xor
begin
builtins: xor
functions: g/2
rule Gen: [Fr(~k)] --> [Out(~k), Key(~k)]
rule R: [Key(k), Fr(~b), In(x)] --[A(g(k,~b)), Got(x XOR k)]-> []
lemma never_zero: all-traces "All #i. Got(zero)@i ==> F"
lemma some_zero: exists-trace "Ex #i. Got(zero)@i"
end
"#;

/// Prove every lemma of `theory`, returning `(exit code, normalized stdout,
/// stderr minus the maude banner)`.
fn prove(stem: &str, theory: &str) -> (i32, String, String) {
    let (code, stdout, stderr) = common::run_raw(TMP_DIR, stem, theory, &["--prove"]);
    (code, normalize_stdout(&stdout), strip_maude_banner(&stderr))
}

fn assert_hs_verdicts(stdout: &str, stderr: &str) {
    for line in [
        "never_zero (all-traces): falsified - found trace",
        "some_zero (exists-trace): verified",
    ] {
        assert!(
            stdout.contains(line),
            "expected `{line}` (tamarin-prover 1.11.0); got stdout:\n{stdout}\nstderr:\n{stderr}"
        );
    }
}

#[test]
fn user_binary_h_with_xor_does_not_prove_a_false_lemma() {
    if !maude_available() {
        eprintln!("skipping: maude not on path");
        return;
    }
    let (code, stdout, stderr) = prove("h2_xor", THEORY_H);
    assert_eq!(code, 0, "stderr: {stderr}");
    assert_hs_verdicts(&stdout, &stderr);
}

/// Control: the same theory with `g/2` gives the 1.11.0 verdicts.
#[test]
fn user_binary_g_with_xor_gives_the_hs_verdicts() {
    if !maude_available() {
        eprintln!("skipping: maude not on path");
        return;
    }
    let (code, stdout, stderr) = prove("g2_xor", THEORY_G);
    assert_eq!(code, 0, "stderr: {stderr}");
    assert_hs_verdicts(&stdout, &stderr);
}
