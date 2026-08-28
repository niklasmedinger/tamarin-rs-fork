// Currently GPL 3.0 until granted permission by the upstream authors
// of the tamarin-prover sources this file cites; list them with:
//   scripts/gen_license_headers.py --authors <this file>

//! (No HS analog.) Reconstructs a [`System`] from the JSON dump produced
//! by the HS web UI's `GET /thy/trace/<idx>/system/*TheoryPath` route
//! (`tamarin-prover/src/Web/Handler.hs`'s `getTheorySystemR`/
//! `systemToJSON`) — for cross-implementation testing: extract a real
//! constraint system from a real HS proof search, reconstruct it here,
//! and check [`crate::canon_graph::extract_graph_part`] (and, once it
//! exists, full canonization) against it — including checking pairs the
//! user already knows should be $\alphaeqac$.
//!
//! Every term/fact/formula leaf in the dump is ordinary pretty-printed
//! Tamarin surface syntax (not a bespoke structured encoding), so
//! reconstruction reuses the SAME parser (`tamarin_parser`) + elaboration
//! ([`crate::elaborate`]) pipeline used to load the source model, instead
//! of a second, separately-validated signature-aware term decoder.
//!
//! **Precondition**: the caller must have already installed the source
//! model's signature (e.g. via [`crate::elaborate::set_user_funs_for_theory`]
//! from parsing/elaborating the SAME `.spthy` file the HS dump was taken
//! from) before calling [`system_from_json`] — elaboration
//! (`term_to_lnterm`/`fact_to_lnfact`) is signature-aware and reads that
//! already-installed context; this module does not set it up itself,
//! since the caller already did so once to load/drive the same model.

use std::fmt;
use std::sync::Arc;

use serde_json::Value;

use tamarin_parser::ast as p;
use tamarin_parser::parser::{parse_formula_str, parse_term_str};
use tamarin_term::lterm::{LNTerm, LVar};
use tamarin_term::term::Term;
use tamarin_term::vterm::Lit;

use crate::constraint::constraints::{Edge, LessAtom, NodeId, Reason};
use crate::constraint::system::System;
use crate::elaborate;
use crate::fact::LNFact;
use crate::guarded::{formula_to_guarded, GAtom, Guarded};
use crate::guarded_types::gfact_to_fact;
use crate::rule::{
    ConcIdx, PremIdx, ProtoRuleACInstInfo, ProtoRuleName, Rule, RuleACInst, RuleAttributes,
    RuleInfo,
};
use crate::tools::equation_store::{EqDisj, EquationStore, LNSubst, LNSubstVFresh, SplitId};
use crate::tools::subterm_store::{SortedPairSet, SubtermConstraint, SubtermStore};

/// Any failure reconstructing a `System` from a dump: malformed JSON
/// shape, a term/fact/formula string that doesn't parse, or one that
/// parses but doesn't elaborate against the currently-installed
/// signature (see the module precondition).
#[derive(Debug)]
pub struct ImportError(pub String);

impl fmt::Display for ImportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "system_import: {}", self.0)
    }
}

impl std::error::Error for ImportError {}

type Res<T> = Result<T, ImportError>;

/// Reconstructs a [`System`] from `json` (parsed from the HS
/// `getTheorySystemR` response body). See the module docs for the
/// signature-installation precondition.
pub fn system_from_json(json: &Value) -> Res<System> {
    let mut sys = System::default();

    if let Some(s) = opt_str(json, "lastAtom")? {
        sys.content_mut().last_atom = Some(parse_node_id(s)?);
    }

    for n in req_array(json, "nodes")? {
        let id = parse_node_id(req_str(n, "id")?)?;
        let rule_name = req_str(n, "ruleName")?;
        let premises = parse_fact_array(n, "premises")?;
        let actions = parse_fact_array(n, "actions")?;
        let conclusions = parse_fact_array(n, "conclusions")?;
        // KNOWN GAP, found via `canon_color`'s real-data testing: every
        // reconstructed node is built as `RuleInfo::Proto`, even when the
        // ORIGINAL rule was actually a built-in intruder rule
        // (`RuleInfo::Intr`, e.g. `ISend`/`IRecv`) -- because the JSON
        // schema's `"ruleName"` field is `getRuleName ru`
        // (`Web/Handler.hs`), a FLATTENED string that renders both cases
        // identically (a user protocol rule literally named `"Send"` and
        // the built-in `ISend` rule both serialize as `"ruleName": "Send"`
        // -- see `canon_color.rs`'s own doc comments for why this
        // ambiguity matters for vertex coloring). A real captured system
        // hit exactly this: NSLPK3's `Send`-named node is structurally
        // the built-in ISend rule (premises `!KU(s)`, conclusion `In(s)`,
        // action `K(s)`), but reconstructs here as a Proto rule, so
        // `ColorTable::rule_color` colors it as if the theory itself
        // declared a rule named `"Send"` rather than as the built-in.
        // Fixing this needs `systemToJSON` to also serialize which case
        // it is (and which `IntrRuleACInfo` variant, for the `Intr` case)
        // -- an HS-side schema change, not attempted here.
        let ru: RuleACInst = Rule::new(
            RuleInfo::Proto(ProtoRuleACInstInfo {
                name: ProtoRuleName::Stand(tamarin_term::intern::intern_str(rule_name)),
                attributes: RuleAttributes::default(),
                loop_breakers: Vec::new(),
            }),
            premises,
            // `Rule::new`'s real parameter order is
            // `(info, premises, conclusions, actions)` -- NOT
            // premises/actions/conclusions in JSON field order. Got this
            // backwards on the first pass; caught only by rendering a
            // REAL captured system to DOT and visually spotting actions
            // and conclusions swapped in the rendered record (a
            // synthetic test with one of the two lists empty per node
            // couldn't have caught it — see the module tests).
            conclusions,
            actions,
        );
        sys.add_node(id, ru);
    }

    for e in req_array(json, "edges")? {
        let src = parse_node_id(req_str(e, "src")?)?;
        let src_idx = ConcIdx(req_u64(e, "srcIdx")? as usize);
        let tgt = parse_node_id(req_str(e, "tgt")?)?;
        let tgt_idx = PremIdx(req_u64(e, "tgtIdx")? as usize);
        sys.content_mut().edges.push(Edge {
            src: (src, src_idx),
            tgt: (tgt, tgt_idx),
        });
    }

    for l in req_array(json, "less")? {
        let smaller = parse_node_id(req_str(l, "smaller")?)?;
        let larger = parse_node_id(req_str(l, "larger")?)?;
        let reason = parse_reason(req_str(l, "reason")?)?;
        sys.content_mut()
            .less_atoms
            .push(LessAtom::new(smaller, larger, reason));
    }

    for f in req_array(json, "formulas")? {
        sys.formulas_mut().push(Arc::new(parse_guarded(as_str(f)?)?));
    }
    for f in req_array(json, "solvedFormulas")? {
        sys.content_mut()
            .solved_formulas
            .push(Arc::new(parse_guarded(as_str(f)?)?));
    }
    for f in req_array(json, "lemmas")? {
        sys.content_mut()
            .lemmas
            .push(Arc::new(parse_guarded(as_str(f)?)?));
    }

    *sys.eq_store_mut() = parse_eq_store(req_obj(json, "eqStore")?)?;
    *sys.subterm_store_mut() = parse_subterm_store(req_obj(json, "subtermStore")?)?;

    Ok(sys)
}

// ---------------------------------------------------------------------
// Leaf parsing: every term/fact/formula string round-trips through the
// SAME parser + elaborator the source model itself was loaded with.
// ---------------------------------------------------------------------

/// A bare term string (e.g. `"#i.1"`, `"senc(x, y)"`) to its elaborated
/// `LNTerm`, using the AC function-symbol names of the
/// currently-installed signature (see the module precondition).
fn parse_term(s: &str) -> Res<LNTerm> {
    let ac_names = elaborate::current_user_ac_names();
    let t = parse_term_str(s, &ac_names)
        .map_err(|e| ImportError(format!("parse_term_str({s:?}): {e}")))?;
    elaborate::term_to_lnterm(&t)
        .ok_or_else(|| ImportError(format!("term_to_lnterm({s:?}): elaboration failed")))
}

/// A bare term string that must denote a `NodeId` (a Node-sorted `LVar`)
/// — every `id`/`src`/`tgt`/`smaller`/`larger`/`lastAtom` field.
fn parse_node_id(s: &str) -> Res<NodeId> {
    match parse_term(s)? {
        Term::Lit(Lit::Var(v)) => Ok(v),
        other => Err(ImportError(format!(
            "{s:?} did not elaborate to a variable (NodeId): {other:?}"
        ))),
    }
}

/// A bare fact string (e.g. `"Fr(~n)"`, `"!KU(x)"`) to an elaborated
/// `LNFact`. Reuses the formula parser/guarded-conversion: a fact with no
/// trailing `@ timepoint` parses as a bare `GAtom::Pred` atom (confirmed
/// directly against the parser — see the canonization plan's testing
/// section), so this needs no dedicated fact-string parser.
fn parse_fact(s: &str) -> Res<LNFact> {
    let f = parse_formula_str(s).map_err(|e| ImportError(format!("parse_formula_str({s:?}): {e}")))?;
    let g = formula_to_guarded(&f)
        .map_err(|e| ImportError(format!("formula_to_guarded({s:?}): {e:?}")))?;
    let gfact = match g {
        Guarded::Atom(GAtom::Pred(fact)) => fact,
        other => {
            return Err(ImportError(format!(
                "{s:?} did not parse as a bare fact (Pred atom): {other:?}"
            )))
        }
    };
    let pfact: p::Fact = gfact_to_fact(&gfact);
    elaborate::fact_to_lnfact(&pfact)
        .map_err(|e| ImportError(format!("fact_to_lnfact({s:?}): {e:?}")))
}

fn parse_fact_array(obj: &Value, key: &str) -> Res<Vec<LNFact>> {
    req_array(obj, key)?.iter().map(|v| parse_fact(as_str(v)?)).collect()
}

/// A formula string (`formulas`/`solvedFormulas`/`lemmas` entries) to a
/// `Guarded` — may be quantified, unlike [`parse_fact`]'s bare-atom case.
/// Same recipe `canon.rs`'s own tests already use to round-trip formula
/// strings.
fn parse_guarded(s: &str) -> Res<Guarded> {
    let f = parse_formula_str(s).map_err(|e| ImportError(format!("parse_formula_str({s:?}): {e}")))?;
    formula_to_guarded(&f).map_err(|e| ImportError(format!("formula_to_guarded({s:?}): {e:?}")))
}

fn parse_reason(s: &str) -> Res<Reason> {
    match s {
        "Formula" => Ok(Reason::Formula),
        "InjectiveFacts" => Ok(Reason::InjectiveFacts),
        "Fresh" => Ok(Reason::Fresh),
        "Adversary" => Ok(Reason::Adversary),
        "NormalForm" => Ok(Reason::NormalForm),
        other => Err(ImportError(format!("unknown Reason tag {other:?}"))),
    }
}

// ---------------------------------------------------------------------
// eqStore / subtermStore
// ---------------------------------------------------------------------

/// A `[varString, termString]` JSON pair to an `(LVar, LNTerm)` binding.
fn parse_subst_pair(v: &Value) -> Res<(LVar, LNTerm)> {
    let arr = as_array(v)?;
    if arr.len() != 2 {
        return Err(ImportError(format!(
            "subst pair must have exactly 2 elements, got {}",
            arr.len()
        )));
    }
    match parse_term(as_str(&arr[0])?)? {
        Term::Lit(Lit::Var(v)) => Ok((v, parse_term(as_str(&arr[1])?)?)),
        other => Err(ImportError(format!(
            "subst pair's first element did not elaborate to a variable: {other:?}"
        ))),
    }
}

fn parse_term_pair(v: &Value) -> Res<(LNTerm, LNTerm)> {
    let arr = as_array(v)?;
    if arr.len() != 2 {
        return Err(ImportError(format!(
            "term pair must have exactly 2 elements, got {}",
            arr.len()
        )));
    }
    Ok((parse_term(as_str(&arr[0])?)?, parse_term(as_str(&arr[1])?)?))
}

fn parse_eq_store(obj: &Value) -> Res<EquationStore> {
    let subst_pairs: Vec<(LVar, LNTerm)> = req_array(obj, "subst")?
        .iter()
        .map(parse_subst_pair)
        .collect::<Res<_>>()?;
    let mut conj: Vec<EqDisj> = Vec::new();
    for c in req_array(obj, "conj")? {
        let split_id = SplitId(req_i64(c, "splitId")?);
        let mut substs: Vec<LNSubstVFresh> = Vec::new();
        for d in req_array(c, "disjuncts")? {
            let pairs: Vec<(LVar, LNTerm)> =
                as_array(d)?.iter().map(parse_subst_pair).collect::<Res<_>>()?;
            substs.push(LNSubstVFresh::from_list(pairs));
        }
        conj.push(EqDisj { split_id, substs });
    }
    Ok(EquationStore {
        subst: LNSubst::from_list(subst_pairs),
        conj,
        // Not carried by the dump — a monotonic allocation counter, not
        // semantic content (see the canonization plan's field audit).
        // Zero is a safe placeholder: it only needs to be >= the
        // highest split id actually used, and a freshly-imported system
        // is never split further before being canonized/compared.
        next_split: SplitId(0),
    })
}

fn parse_subterm_store(obj: &Value) -> Res<SubtermStore> {
    let to_constraints = |key: &str| -> Res<Vec<SubtermConstraint>> {
        req_array(obj, key)?
            .iter()
            .map(|v| {
                let (small, big) = parse_term_pair(v)?;
                Ok(SubtermConstraint {
                    small,
                    big,
                    propagated: false,
                })
            })
            .collect()
    };
    let to_pair_set = |key: &str| -> Res<SortedPairSet> {
        let pairs: Vec<(LNTerm, LNTerm)> =
            req_array(obj, key)?.iter().map(parse_term_pair).collect::<Res<_>>()?;
        Ok(SortedPairSet::rebuild_from(pairs))
    };
    Ok(SubtermStore {
        subterms: to_constraints("subterms")?,
        solved_subterms: to_constraints("solvedSubterms")?,
        contradictory: obj
            .get("contradictory")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        neg_subterms: to_pair_set("negSubterms")?,
        old_neg_subterms: to_pair_set("oldNegSubterms")?,
    })
}

// ---------------------------------------------------------------------
// Tiny JSON accessors — deliberately not a general-purpose helper crate
// dependency: every access here is a required (or explicitly optional)
// field of the fixed schema `systemToJSON` emits, so a missing/wrong-type
// field is always a caller bug or a schema drift, not a case to paper
// over silently.
// ---------------------------------------------------------------------

fn req_obj<'a>(v: &'a Value, key: &str) -> Res<&'a Value> {
    v.get(key)
        .ok_or_else(|| ImportError(format!("missing field {key:?}")))
}

fn req_array<'a>(v: &'a Value, key: &str) -> Res<&'a Vec<Value>> {
    req_obj(v, key)?
        .as_array()
        .ok_or_else(|| ImportError(format!("field {key:?} is not an array")))
}

fn req_str<'a>(v: &'a Value, key: &str) -> Res<&'a str> {
    req_obj(v, key)?
        .as_str()
        .ok_or_else(|| ImportError(format!("field {key:?} is not a string")))
}

fn opt_str<'a>(v: &'a Value, key: &str) -> Res<Option<&'a str>> {
    match v.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(other) => Ok(Some(other.as_str().ok_or_else(|| {
            ImportError(format!("field {key:?} is not a string or null"))
        })?)),
    }
}

fn req_u64(v: &Value, key: &str) -> Res<u64> {
    req_obj(v, key)?
        .as_u64()
        .ok_or_else(|| ImportError(format!("field {key:?} is not a non-negative integer")))
}

fn req_i64(v: &Value, key: &str) -> Res<i64> {
    req_obj(v, key)?
        .as_i64()
        .ok_or_else(|| ImportError(format!("field {key:?} is not an integer")))
}

fn as_str(v: &Value) -> Res<&str> {
    v.as_str()
        .ok_or_else(|| ImportError(format!("expected a string, got {v:?}")))
}

fn as_array(v: &Value) -> Res<&Vec<Value>> {
    v.as_array()
        .ok_or_else(|| ImportError(format!("expected an array, got {v:?}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tamarin_parser::parser::parse_theory;

    /// Installs a minimal, no-custom-functions signature (built-ins only
    /// — `Fr`/`In`/`Out`/pairing all resolve without any `functions:`/
    /// `builtins:` declaration), matching `elaborate_tests.rs`'s own
    /// `set_user_funs_for_theory` usage pattern. The returned guard must
    /// be kept alive for the duration of any `system_from_json` call.
    fn install_test_signature() -> elaborate::UserFunsForTheoryGuard {
        let thy = parse_theory("theory T begin\nend", &[]).expect("parse minimal theory");
        elaborate::set_user_funs_for_theory(&thy)
    }

    fn sample_json() -> Value {
        json!({
            "lastAtom": "#i.2",
            "nodes": [
                {
                    "id": "#i.1",
                    "ruleName": "Create",
                    "premises": ["Fr(~n)"],
                    "actions": [],
                    "conclusions": ["Out(<m, ~n>)"]
                },
                {
                    "id": "#i.2",
                    "ruleName": "Receive",
                    "premises": ["In(x)"],
                    "actions": ["Recv(x)"],
                    "conclusions": []
                }
            ],
            "edges": [
                { "src": "#i.1", "srcIdx": 0, "tgt": "#i.2", "tgtIdx": 0 }
            ],
            "less": [
                { "smaller": "#i.1", "larger": "#i.2", "reason": "Formula" }
            ],
            "formulas": ["Recv(x) @ #i.2"],
            "solvedFormulas": [],
            "lemmas": [],
            "eqStore": {
                "subst": [["z", "m"]],
                "conj": []
            },
            "subtermStore": {
                "subterms": [],
                "solvedSubterms": [],
                "negSubterms": [],
                "oldNegSubterms": [],
                "contradictory": false
            }
        })
    }

    #[test]
    fn round_trips_a_small_system_end_to_end() {
        let _guard = install_test_signature();
        let sys = system_from_json(&sample_json()).expect("system_from_json");

        assert_eq!(sys.nodes.len(), 2);
        // Pins the premises/actions/conclusions -> `Rule::new` argument
        // mapping directly (that mapping order does NOT match the JSON's
        // field order — see the `Rule::new` call site's own comment for
        // the bug this caught): a synthetic test where one of
        // actions/conclusions is empty per node can't catch a swap, so
        // check the SECOND node's fact lists by content instead of just
        // counting vertices.
        let (_, second_rule) = &sys.nodes[1];
        assert_eq!(
            second_rule.actions.iter().map(crate::pretty_system::pretty_fact).collect::<Vec<_>>(),
            vec!["Recv(x)".to_string()]
        );
        assert!(second_rule.conclusions.is_empty());
        assert_eq!(sys.edges.len(), 1);
        assert_eq!(sys.less_atoms.len(), 1);
        assert_eq!(sys.formulas.len(), 1);
        assert!(sys.solved_formulas.is_empty());
        assert!(sys.lemmas.is_empty());
        assert_eq!(sys.eq_store.subst.to_list().len(), 1);
        assert!(sys.eq_store.conj.is_empty());
        assert!(sys.subterm_store.subterms.is_empty());
        assert!(!sys.subterm_store.contradictory);
        assert_eq!(sys.last_atom, Some(parse_node_id("#i.2").expect("parse #i.2")));

        // The actual payload this whole pipeline exists for: the
        // imported System's graph part should look exactly like a
        // hand-built one would -- 2 rule-instance vertices, 1 action
        // vertex (Recv @ #i.2), and a reified relation vertex for each
        // of the real system Edge, the Less atom, and the action's own
        // AtTimepoint link (each relation contributes 2 graph edges:
        // src -> relation -> tgt -- see `canon_graph`'s module docs).
        let part = crate::canon_graph::extract_graph_part(&sys);
        use crate::canon_graph::VertexKind;
        let count = |pred: &dyn Fn(&VertexKind) -> bool| part.vertices.iter().filter(|v| pred(v)).count();
        assert_eq!(count(&|v| matches!(v, VertexKind::RuleInstance(_, _))), 2);
        assert_eq!(count(&|v| matches!(v, VertexKind::Action(_, _))), 1);
        assert_eq!(count(&|v| matches!(v, VertexKind::EdgeRelation)), 1);
        assert_eq!(count(&|v| matches!(v, VertexKind::LessRelation)), 1);
        assert_eq!(count(&|v| matches!(v, VertexKind::AtTimepointRelation)), 1);
        assert_eq!(part.edges.len(), 6);

        // And it renders to a well-formed DOT document without panicking.
        let dot = crate::canon_graph::to_graphviz(&part);
        assert!(dot.starts_with("digraph G {\n"));
    }

    #[test]
    fn missing_required_field_is_a_clean_error_not_a_panic() {
        let _guard = install_test_signature();
        let mut bad = sample_json();
        bad.as_object_mut().unwrap().remove("nodes");
        let err = system_from_json(&bad).expect_err("missing `nodes` must error");
        assert!(err.0.contains("nodes"), "error should name the field: {err}");
    }

    #[test]
    fn unparseable_term_is_a_clean_error_not_a_panic() {
        let _guard = install_test_signature();
        let mut bad = sample_json();
        bad["nodes"][0]["premises"] = json!(["("]);
        let err = system_from_json(&bad).expect_err("garbage term text must error");
        assert!(err.0.contains('('), "error should mention the bad text: {err}");
    }
}

