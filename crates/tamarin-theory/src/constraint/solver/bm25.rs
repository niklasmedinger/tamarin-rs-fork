//! (No HS analog.) A BM25 lexical prior over the candidate proof methods of
//! one constraint system, for [`super::mcgs_search`]. A port of TamRL's
//! `src/rl/idf_prior.py` (`tamarinml`, branch `idf_heuristic`).
//!
//! The idea: the methods relevant to a lemma tend to mention the same
//! identifiers as the lemma's formula and as the methods already applied on
//! the way to the system. BM25 turns that overlap into a score. In IR terms:
//! - the corpus is the system's candidate methods and nothing else, so
//!   document frequencies come from the siblings only;
//! - a document is one candidate's rendered text, as a bag of tokens;
//! - the query is the lemma's text and/or the methods along the search path.
//!
//! As in TamRL, the scorer is `rank_bm25`'s `BM25Plus` with `delta = 0`
//! rather than `BM25Okapi`: Okapi floors the IDF of every term in more than
//! half the documents to a shared epsilon, which with a handful of siblings
//! hits a large share of all terms; `ln((N + 1) / df)` stays positive.
//! The scores are z-normalized per system ([`z_bias`]), which makes the bias
//! scale-free.
//!
//! Tokens drop the `.INDEX` and `:SORT` annotations of variables, so `a.0`
//! and `a.1` share the token `a`, and keep underscores, so fact names such
//! as `St_1_UE` stay one (discriminative) token.

use std::sync::OnceLock;

use tamarin_utils::FastMap;

use crate::constraint::constraints::Goal;
use crate::constraint::solver::proof_method::ProofMethod;
use crate::constraint::system::System;

/// `rank_bm25`'s defaults.
const K1: f64 = 1.5;
const B: f64 = 0.75;

/// A variable's `.INDEX` and optional `:SORT`, as Tamarin prints them.
fn var_annotation() -> &'static fancy_regex::Regex {
    static RE: OnceLock<fancy_regex::Regex> = OnceLock::new();
    RE.get_or_init(|| {
        fancy_regex::Regex::new(r"\.\d+(?:\s*:\s*(?:fresh|msg|node|pub|nat))?\b")
            .expect("the variable-annotation pattern is valid")
    })
}

/// Lowercased `[A-Za-z0-9_]+` words of `text`, with variable annotations
/// removed first.
pub(crate) fn tokens(text: &str) -> Vec<String> {
    let stripped = var_annotation().replace_all(text, " ").to_lowercase();
    stripped
        .split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
        .filter(|word| !word.is_empty())
        .map(str::to_string)
        .collect()
}

/// The text BM25 sees for one method: its kind, then its arguments, as the
/// Tamarin-ML API's `toJSONProofmethod`/`toJSONGoal` give them to TamRL.
pub(crate) fn method_text(method: &ProofMethod, sys: &System) -> String {
    let goal = match method {
        ProofMethod::Simplify => return "simplify".into(),
        ProofMethod::Induction => return "induction".into(),
        ProofMethod::SolveGoal(goal) => goal,
        other => return crate::pretty_theory::pretty_proof_method_inline(other),
    };
    let (kind, args) = match goal {
        Goal::Action(..) => ("action", vec![]),
        Goal::Chain(..) => ("chain", vec![]),
        Goal::Premise(..) => ("premise", vec![]),
        Goal::Disj(_) => ("disj", vec![]),
        Goal::Subterm(_) => ("subterm", vec![]),
        // `splitEqs(n)` names the split by number only; the disjuncts carry
        // the identifiers.
        Goal::Split(id) => (
            "split",
            crate::pretty_system::pretty_split_disjuncts(sys, *id),
        ),
    };
    let args = if args.is_empty() {
        crate::pretty_theory::solve_goal_to_doc(goal).render()
    } else {
        args.join(" | ")
    };
    format!("{kind}: {args}")
}

/// The query tokens of a lemma: its text with the attribute block of its
/// header (`lemma name [attrs]:`) removed, which only configures the prover.
pub(crate) fn lemma_query(plaintext: &str) -> Vec<String> {
    static HEADER: OnceLock<fancy_regex::Regex> = OnceLock::new();
    let header = HEADER.get_or_init(|| {
        fancy_regex::Regex::new(r"(?m)^\s*lemma\s+[A-Za-z0-9_]+\s*(\[[^\]]*\])?\s*:")
            .expect("the lemma-header pattern is valid")
    });
    let Ok(Some(m)) = header.find(plaintext) else {
        return tokens(plaintext);
    };
    let attributes =
        fancy_regex::Regex::new(r"\[[^\]]*\]").expect("the attribute pattern is valid");
    let head = attributes.replace_all(m.as_str(), " ");
    tokens(&format!("{head}{}", &plaintext[m.end()..]))
}

/// `rank_bm25.BM25Plus(docs, delta=0).get_scores(query)`: one score per
/// document. Query tokens count with their multiplicity. An empty query or
/// a corpus of empty documents scores all zero.
pub(crate) fn bm25_plus_scores(docs: &[Vec<String>], query: &[String]) -> Vec<f64> {
    let n = docs.len();
    if query.is_empty() || docs.iter().all(Vec::is_empty) {
        return vec![0.0; n];
    }
    let avgdl = docs.iter().map(Vec::len).sum::<usize>() as f64 / n as f64;
    let freqs: Vec<FastMap<&str, usize>> = docs
        .iter()
        .map(|doc| {
            let mut f = FastMap::default();
            for t in doc {
                *f.entry(t.as_str()).or_insert(0) += 1;
            }
            f
        })
        .collect();
    let mut df: FastMap<&str, usize> = FastMap::default();
    for f in &freqs {
        for &t in f.keys() {
            *df.entry(t).or_insert(0) += 1;
        }
    }
    let mut scores = vec![0.0; n];
    for q in query {
        let Some(&d) = df.get(q.as_str()) else {
            continue;
        };
        let idf = ((n + 1) as f64).ln() - (d as f64).ln();
        for (i, f) in freqs.iter().enumerate() {
            let tf = f.get(q.as_str()).copied().unwrap_or(0) as f64;
            let norm = K1 * (1.0 - B + B * docs[i].len() as f64 / avgdl);
            scores[i] += idf * (tf * (K1 + 1.0)) / (norm + tf);
        }
    }
    scores
}

/// `weight` times the z-scores of `scores` (population standard deviation),
/// so the bias's standard deviation is exactly `weight`. All zero when the
/// scores carry no preference: a constant vector, up to float residue that
/// z-scoring would otherwise amplify to full strength.
pub(crate) fn z_bias(scores: &[f64], weight: f64) -> Vec<f64> {
    let n = scores.len();
    if n == 0 {
        return Vec::new();
    }
    let mean = scores.iter().sum::<f64>() / n as f64;
    let std = (scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
    let max_abs = scores.iter().fold(0.0_f64, |m, s| m.max(s.abs()));
    if std <= 1e-6 * max_abs.max(1.0) {
        return vec![0.0; n];
    }
    scores.iter().map(|s| weight * (s - mean) / std).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(ws: &[&str]) -> Vec<String> {
        ws.iter().map(|w| w.to_string()).collect()
    }

    // The expected tokens and scores below were computed with TamRL's
    // regexes and `rank_bm25` 0.2.2 (`BM25Plus(docs, delta=0.0)`).

    #[test]
    fn tokens_drop_variable_annotations_and_keep_underscores() {
        assert_eq!(tokens("!KU( ~n.1 ) @ #vk.2"), words(&["ku", "n", "vk"]));
        assert_eq!(
            tokens("St_1_UE( tid.0:fresh, x.12a, y.3 : msg )"),
            words(&["st_1_ue", "tid", "x", "12a", "y"])
        );
        assert_eq!(
            tokens("action: Out( <'1', x> ) @ #i"),
            words(&["action", "out", "1", "x", "i"])
        );
        assert_eq!(tokens("a.1:msgfoo z.0:node"), words(&["a", "msgfoo", "z"]));
    }

    #[test]
    fn the_lemma_query_skips_the_attributes() {
        let text = "lemma Client_auth [reuse, heuristic=S]:\n  \
                    \"All x #i. Commit(x) @ #i ==> Ex #j. Running(x) @ #j\"";
        assert_eq!(
            lemma_query(text),
            words(&[
                "lemma",
                "client_auth",
                "all",
                "x",
                "i",
                "commit",
                "x",
                "i",
                "ex",
                "j",
                "running",
                "x",
                "j"
            ])
        );
    }

    fn assert_close(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-12, "got {got:?}, want {want:?}");
        }
    }

    #[test]
    fn bm25_plus_matches_rank_bm25() {
        let docs = vec![
            words(&["action", "st_1_ue", "tid", "supi"]),
            words(&["premise", "k", "x", "x"]),
            words(&["simplify"]),
            words(&["chain", "st_1_ue", "k"]),
        ];
        let query = words(&["st_1_ue", "k", "k", "supi", "missing"]);
        assert_close(
            &bm25_plus_scores(&docs, &query),
            &[
                2.1962857776593525,
                1.5935490989115741,
                0.0,
                2.7488721956224653,
            ],
        );
        let docs = vec![words(&["a", "b"]), words(&["a"]), vec![]];
        assert_close(
            &bm25_plus_scores(&docs, &words(&["a", "b"])),
            &[1.434097614951611, std::f64::consts::LN_2, 0.0],
        );
    }

    #[test]
    fn no_query_or_no_text_scores_zero() {
        let docs = vec![words(&["a"]), words(&["b"])];
        assert_eq!(bm25_plus_scores(&docs, &[]), vec![0.0, 0.0]);
        assert_eq!(
            bm25_plus_scores(&[vec![], vec![]], &words(&["a"])),
            vec![0.0, 0.0]
        );
    }

    #[test]
    fn the_z_bias_has_the_weight_as_its_spread_and_ignores_constant_scores() {
        let bias = z_bias(&[1.0, 2.0, 3.0, 6.0], 0.5);
        let mean = bias.iter().sum::<f64>() / 4.0;
        let std = (bias.iter().map(|b| (b - mean).powi(2)).sum::<f64>() / 4.0).sqrt();
        assert!(mean.abs() < 1e-12);
        assert!((std - 0.5).abs() < 1e-12);
        assert!(bias[3] > bias[2] && bias[2] > bias[1] && bias[1] > bias[0]);
        assert_eq!(z_bias(&[3.0, 3.0, 3.0], 1.0), vec![0.0; 3]);
        assert_eq!(z_bias(&[1e9, 1e9 + 1e-3], 1.0), vec![0.0; 2]);
    }
}
