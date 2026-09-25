// Currently GPL 3.0 until granted permission by the upstream authors
// of the tamarin-prover sources this file cites; list them with:
//   scripts/gen_license_headers.py --authors <this file>

//! Per-FIELD fingerprinting of a [`CanonicalSystem`] (Stage G's output) --
//! deliberately NOT one combined hash.
//!
//! [`canon::fingerprint_guarded`]'s own doc comment once sketched the
//! obvious next step as "a system's fingerprint can be built the same way
//! from its rules'/formulas' fingerprints in turn" -- i.e. one Merkle hash
//! over the whole `CanonicalSystem`, giving a binary "same system or not"
//! outcome. That is NOT what this module does, on purpose: two systems
//! that are almost-but-not-quite $\alphaeqac$ (e.g. differing only in
//! `eq_store.conj` because of a remaining equational-solving-order
//! artifact, with an otherwise byte-identical graph part) are far more
//! useful to distinguish from two systems that are simply unrelated, and a
//! single combined hash can never tell those two cases apart -- both just
//! come out "different."
//!
//! Instead, [`CanonicalSystemFingerprint`] mirrors [`CanonicalSystem`]'s
//! own field structure one-to-one: each field gets its OWN
//! [`Fingerprint`], and `eq_store`/`subterm_store` -- themselves multi-field
//! structs -- get their own per-field fingerprint structs in turn, rather
//! than being collapsed to one hash each (which would reintroduce the same
//! binary-outcome problem one level down). [`compare_fingerprints`] then
//! reports which SPECIFIC fields matched and which didn't
//! ([`FieldMatch::mismatched_fields`]), which is the actual point: a
//! caller can tell a "close match" (most fields agree) from a "no match"
//! (nothing agrees) from an exact match, using only the compact
//! fingerprints -- no need to hold onto or re-walk the original
//! `CanonicalSystem`s just to compare them field by field.
//!
//! Every leaf-level hash reuses the existing Merkle-hash primitives
//! (`tamarin_term::fingerprint::fingerprint_term`,
//! [`canon::fingerprint_guarded`]) unchanged; this module only adds the
//! SEQUENCE-level composition (`u64(len)` + `digest` each element, exactly
//! the pattern `fingerprint_guarded`'s own `Conj`/`Disj` handling already
//! uses) needed to fold a `Vec<_>` field into one fingerprint, plus the
//! top-level field-by-field assembly.

use crate::canon::{
    self, CanonicalEqStore, CanonicalGoal, CanonicalGoalKind, CanonicalProofMethod,
    CanonicalSubtermStore, CanonicalSystem,
};
use crate::constraint::system::{SourceKind, Side};
use crate::guarded::Guarded;

use tamarin_term::fingerprint::{fingerprint_term, Fingerprint};
use tamarin_term::lterm::{LNTerm, Name};
use tamarin_term::vterm::Lit;
use tamarin_utils::fingerprint::FingerprintHasher;

/// Same literal model `tamarin_term::alpha_eq_ac`/`canon` use internally:
/// a term leaf is either a name constant or a variable.
type LNLit = Lit<Name, tamarin_term::lterm::LVar>;

/// Per-field fingerprints of a [`CanonicalSystem`] -- see the module docs
/// for why these are kept separate rather than combined into one hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CanonicalSystemFingerprint {
    pub graph_part: Fingerprint,
    pub formulas: Fingerprint,
    pub solved_formulas: Fingerprint,
    pub lemmas: Fingerprint,
    pub eq_store: CanonicalEqStoreFingerprint,
    pub subterm_store: CanonicalSubtermStoreFingerprint,
    /// `CanonicalSystem::goals` -- one `Vec<_>` field, same treatment as
    /// `formulas`/`solved_formulas`/`lemmas` (not broken out per-variant:
    /// `Goal::Action`/`Split` are already excluded before this point, so
    /// there's no equivalent of `eq_store.subst` vs `.conj` needing
    /// separate tracking here).
    pub goals: Fingerprint,
    /// Small, already-`Copy`/`Eq` enums -- kept as native values rather
    /// than hashed. Hashing a 2-variant enum through SHA-256 would lose
    /// information (a caller could no longer see WHAT differs, just THAT
    /// it does) for no compactness gain.
    pub source_kind: Option<SourceKind>,
    pub side: Option<Side>,
}

/// Per-field fingerprints of a [`CanonicalEqStore`]. A separate struct
/// (rather than one `Fingerprint` for the whole `eq_store` field) for the
/// same reason [`CanonicalSystemFingerprint`] itself has one field per
/// `CanonicalSystem` field: collapsing `subst`/`conj` into one hash would
/// hide which of the two actually diverged between two systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CanonicalEqStoreFingerprint {
    pub subst: Fingerprint,
    pub conj: Fingerprint,
}

/// Per-field fingerprints of a [`CanonicalSubtermStore`] -- see
/// [`CanonicalEqStoreFingerprint`]'s own doc comment for why this isn't
/// one combined hash either.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CanonicalSubtermStoreFingerprint {
    pub subterms: Fingerprint,
    pub solved_subterms: Fingerprint,
    /// Kept as a native `bool`, not hashed -- see
    /// [`CanonicalSystemFingerprint::source_kind`]'s own doc comment.
    pub contradictory: bool,
    pub neg_subterms: Fingerprint,
}

/// Fingerprints every field of `sys` independently (see the module docs).
pub fn fingerprint_constraint_system(sys: &CanonicalSystem) -> CanonicalSystemFingerprint {
    CanonicalSystemFingerprint {
        graph_part: fingerprint_term(&sys.graph_part),
        formulas: fingerprint_guarded_seq(&sys.formulas),
        solved_formulas: fingerprint_guarded_seq(&sys.solved_formulas),
        lemmas: fingerprint_guarded_seq(&sys.lemmas),
        eq_store: fingerprint_eq_store(&sys.eq_store),
        subterm_store: fingerprint_subterm_store(&sys.subterm_store),
        goals: fingerprint_goals(&sys.goals),
        source_kind: sys.source_kind,
        side: sys.side,
    }
}

fn fingerprint_eq_store(store: &CanonicalEqStore) -> CanonicalEqStoreFingerprint {
    CanonicalEqStoreFingerprint {
        subst: fingerprint_lnlit_term_pairs(&store.subst),
        conj: fingerprint_eq_conj(&store.conj),
    }
}

fn fingerprint_subterm_store(store: &CanonicalSubtermStore) -> CanonicalSubtermStoreFingerprint {
    CanonicalSubtermStoreFingerprint {
        subterms: fingerprint_term_pairs(&store.subterms),
        solved_subterms: fingerprint_term_pairs(&store.solved_subterms),
        contradictory: store.contradictory,
        neg_subterms: fingerprint_term_pairs(&store.neg_subterms),
    }
}

/// Folds a sequence of already-canonical `Guarded` formulas into one
/// fingerprint -- the exact same "length-prefix, then subhash each
/// element in order" composition [`canon::fingerprint_guarded`] already
/// uses internally for `Conj`/`Disj`, just applied one level up. Order is
/// significant here on purpose: `formulas`/`solved_formulas`/`lemmas` are
/// already sorted via `cmp_guarded` by the time a `CanonicalSystem`
/// exists, so hashing them in sequence order is hashing CONTENT, not
/// incidental construction order.
fn fingerprint_guarded_seq(items: &[Guarded]) -> Fingerprint {
    let mut h = FingerprintHasher::new();
    h.u64(items.len() as u64);
    for it in items {
        h.digest(&canon::fingerprint_guarded(it));
    }
    h.finish()
}

/// Folds `CanonicalSystem::goals` into one fingerprint -- same
/// "length-prefix, then subhash each element in order" composition as
/// [`fingerprint_guarded_seq`], just over [`CanonicalGoal`] instead of
/// `Guarded`. Order is content-driven here too: `goals` is sorted via
/// `cmp_canonical_goal` by the time a `CanonicalSystem` exists.
fn fingerprint_goals(goals: &[CanonicalGoal]) -> Fingerprint {
    let mut h = FingerprintHasher::new();
    h.u64(goals.len() as u64);
    for g in goals {
        h.digest(&fingerprint_canonical_goal(g));
    }
    h.finish()
}

/// Fingerprints one [`CanonicalGoal`]: a variant tag (so `Chain`/`Premise`/
/// `Disj`/`Subterm` never collide even if their payloads happened to hash
/// the same by coincidence -- same discipline
/// [`canon::fingerprint_guarded`] uses for `Guarded`'s own variants) plus
/// the payload, then `solved` (content-relevant -- see
/// [`CanonicalGoal`]'s own doc comment).
fn fingerprint_canonical_goal(g: &CanonicalGoal) -> Fingerprint {
    let mut h = FingerprintHasher::new();
    hash_goal_kind(&mut h, &g.kind);
    h.u8(u8::from(g.solved));
    h.finish()
}

/// Fingerprints one [`CanonicalProofMethod`]: a variant tag plus its
/// payload. Compare methods of two systems as a sorted multiset of these.
pub fn fingerprint_proof_method(m: &CanonicalProofMethod) -> Fingerprint {
    let mut h = FingerprintHasher::new();
    match m {
        CanonicalProofMethod::Simplify => {
            h.tag("Simplify");
        }
        CanonicalProofMethod::Induction => {
            h.tag("Induction");
        }
        CanonicalProofMethod::Sorry(reason) => {
            h.tag("Sorry");
            match reason {
                Some(r) => h.u8(1).tag(r),
                None => h.u8(0),
            };
        }
        CanonicalProofMethod::Finished {
            result,
            contradiction,
        } => {
            h.tag("Finished").tag(result);
            match contradiction {
                Some(c) => h.u8(1).tag(c),
                None => h.u8(0),
            };
        }
        CanonicalProofMethod::Solve(kind) => {
            h.tag("Solve");
            hash_goal_kind(&mut h, kind);
        }
        CanonicalProofMethod::SolveSplit(alts) => {
            h.tag("SolveSplit");
            h.digest(&fingerprint_eq_disj(alts));
        }
    }
    h.finish()
}

/// The variant tag and payload of a [`CanonicalGoalKind`] -- shared by goal
/// and proof-method fingerprints. Tags keep `Chain`/`Premise`/`Disj`/...
/// from colliding even if their payloads happened to hash the same (same
/// discipline [`canon::fingerprint_guarded`] uses for `Guarded`'s variants).
fn hash_goal_kind(h: &mut FingerprintHasher, kind: &CanonicalGoalKind) {
    match kind {
        CanonicalGoalKind::Action(nid_lit, fact_term) => {
            h.tag("Action");
            h.digest(&fingerprint_lnlit(nid_lit));
            h.digest(&fingerprint_term(fact_term));
        }
        CanonicalGoalKind::Chain(conc_lit, conc_idx, prem_lit, prem_idx) => {
            h.tag("Chain");
            h.digest(&fingerprint_lnlit(conc_lit));
            h.u64(conc_idx.0 as u64);
            h.digest(&fingerprint_lnlit(prem_lit));
            h.u64(prem_idx.0 as u64);
        }
        CanonicalGoalKind::Premise(prem_lit, prem_idx, fact_term) => {
            h.tag("Premise");
            h.digest(&fingerprint_lnlit(prem_lit));
            h.u64(prem_idx.0 as u64);
            h.digest(&fingerprint_term(fact_term));
        }
        CanonicalGoalKind::Disj(alts) => {
            h.tag("Disj");
            h.digest(&fingerprint_guarded_seq(alts));
        }
        CanonicalGoalKind::Subterm(small, big) => {
            h.tag("Subterm");
            h.digest(&fingerprint_term(small));
            h.digest(&fingerprint_term(big));
        }
    }
}

/// Fingerprints a bare `LNLit` by wrapping it as a one-node `Term::Lit`
/// and reusing [`fingerprint_term`] unchanged -- `fingerprint_term`'s own
/// `Lit` handling (`hash_lit`) isn't exposed as a separate function, and
/// wrapping is free (`Lit` is `Copy`, no allocation), so there's nothing
/// to gain from duplicating that logic here.
fn fingerprint_lnlit(l: &LNLit) -> Fingerprint {
    fingerprint_term(&LNTerm::Lit(*l))
}

/// Folds a sequence of `(LNLit, LNTerm)` pairs (`eq_store.subst`, and one
/// alternative's own canonicalized range terms inside `eq_store.conj`)
/// into one fingerprint, preserving pair order (already canonical-order
/// for `subst`; see [`fingerprint_eq_disj`] for why order matters, or
/// doesn't, inside `conj`).
fn fingerprint_lnlit_term_pairs(pairs: &[(LNLit, LNTerm)]) -> Fingerprint {
    let mut h = FingerprintHasher::new();
    h.u64(pairs.len() as u64);
    for (k, v) in pairs {
        h.digest(&fingerprint_lnlit(k));
        h.digest(&fingerprint_term(v));
    }
    h.finish()
}

/// Folds a sequence of `(LNTerm, LNTerm)` pairs (`subterm_store`'s three
/// term-pair fields) into one fingerprint, preserving pair order (all
/// three are already sorted by the time a `CanonicalSystem` exists).
fn fingerprint_term_pairs(pairs: &[(LNTerm, LNTerm)]) -> Fingerprint {
    let mut h = FingerprintHasher::new();
    h.u64(pairs.len() as u64);
    for (a, b) in pairs {
        h.digest(&fingerprint_term(a));
        h.digest(&fingerprint_term(b));
    }
    h.finish()
}

/// Folds `eq_store.conj: Vec<Vec<Vec<(LNLit, LNTerm)>>>` (the outer list
/// of `EqDisj`s) into one fingerprint via [`fingerprint_eq_disj`] per
/// disjunction, in order -- already canonical-content order (see
/// `CanonicalEqStore::conj`'s own doc comment in `canon.rs`).
fn fingerprint_eq_conj(conj: &[Vec<Vec<(LNLit, LNTerm)>>]) -> Fingerprint {
    let mut h = FingerprintHasher::new();
    h.u64(conj.len() as u64);
    for disj in conj {
        h.digest(&fingerprint_eq_disj(disj));
    }
    h.finish()
}

/// Folds one `EqDisj`'s alternatives (`Vec<Vec<(LNLit, LNTerm)>>`) into
/// one fingerprint via [`fingerprint_lnlit_term_pairs`] per alternative,
/// in order. Order here is content-driven, not incidental: a
/// `CanonicalSystem`'s `conj` stores alternatives sorted by their own
/// canonicalized content as a MULTISET (duplicates preserved, never
/// deduplicated -- see `CanonicalEqStore::conj`'s own doc comment), so two
/// $\alphaeqac$-equivalent systems' corresponding disjunctions always
/// produce alternatives in the same relative order here too.
fn fingerprint_eq_disj(alts: &[Vec<(LNLit, LNTerm)>]) -> Fingerprint {
    let mut h = FingerprintHasher::new();
    h.u64(alts.len() as u64);
    for alt in alts {
        h.digest(&fingerprint_lnlit_term_pairs(alt));
    }
    h.finish()
}

/// How many of [`CanonicalSystemFingerprint`]'s (and its two nested
/// structs') fields [`FieldMatch`] tracks -- kept as one constant so
/// [`FieldMatch::total_count`] can't silently drift out of sync with the
/// struct's own field list.
const FIELD_COUNT: usize = 13;

/// A field-by-field comparison of two [`CanonicalSystemFingerprint`]s --
/// the actual "close match" report: which SPECIFIC fields agreed and
/// which didn't, not just whether the two systems match overall. See the
/// module docs for why this exists instead of a single combined
/// match/no-match hash comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldMatch {
    pub graph_part: bool,
    pub formulas: bool,
    pub solved_formulas: bool,
    pub lemmas: bool,
    pub eq_store_subst: bool,
    pub eq_store_conj: bool,
    pub subterm_store_subterms: bool,
    pub subterm_store_solved_subterms: bool,
    pub subterm_store_contradictory: bool,
    pub subterm_store_neg_subterms: bool,
    pub goals: bool,
    pub source_kind: bool,
    pub side: bool,
}

impl FieldMatch {
    /// All twelve fields, in [`CanonicalSystem`]'s own field order (its
    /// two nested stores' fields inlined in place) -- the single source of
    /// truth [`Self::matched_count`]/[`Self::mismatched_fields`] both walk,
    /// so the two can never disagree about which fields exist.
    fn fields(&self) -> [(&'static str, bool); FIELD_COUNT] {
        [
            ("graph_part", self.graph_part),
            ("formulas", self.formulas),
            ("solved_formulas", self.solved_formulas),
            ("lemmas", self.lemmas),
            ("eq_store.subst", self.eq_store_subst),
            ("eq_store.conj", self.eq_store_conj),
            ("subterm_store.subterms", self.subterm_store_subterms),
            (
                "subterm_store.solved_subterms",
                self.subterm_store_solved_subterms,
            ),
            (
                "subterm_store.contradictory",
                self.subterm_store_contradictory,
            ),
            ("subterm_store.neg_subterms", self.subterm_store_neg_subterms),
            ("goals", self.goals),
            ("source_kind", self.source_kind),
            ("side", self.side),
        ]
    }

    /// How many of the twelve tracked fields matched.
    pub fn matched_count(&self) -> usize {
        self.fields().into_iter().filter(|(_, matched)| *matched).count()
    }

    /// The total number of fields tracked (always [`FIELD_COUNT`]) --
    /// paired with [`Self::matched_count`] for a "N of M" close-match
    /// score.
    pub fn total_count(&self) -> usize {
        FIELD_COUNT
    }

    /// Every field that did NOT match, in `CanonicalSystem`'s own field
    /// order -- the actual diagnostic payload: which specific part of two
    /// otherwise-similar systems diverged, e.g. `["eq_store.conj"]` for
    /// two systems that agree everywhere except a remaining
    /// equational-solving-order artifact.
    pub fn mismatched_fields(&self) -> Vec<&'static str> {
        self.fields()
            .into_iter()
            .filter(|(_, matched)| !matched)
            .map(|(name, _)| name)
            .collect()
    }

    /// True iff every tracked field matched -- recovers the old binary
    /// "same system or not" outcome for a caller that just wants that,
    /// without giving up the per-field detail for callers that want more.
    pub fn is_full_match(&self) -> bool {
        self.matched_count() == self.total_count()
    }
}

/// Compares two [`CanonicalSystemFingerprint`]s field by field. Cheap and
/// needs neither original `CanonicalSystem` -- the whole point of
/// fingerprinting first: a caller can hold onto just the (small, `Copy`)
/// fingerprints for many systems and only reach for an original system's
/// full content (e.g. from a debug store keyed by its fingerprint) once
/// this report says two are worth a closer look.
pub fn compare_fingerprints(
    a: &CanonicalSystemFingerprint,
    b: &CanonicalSystemFingerprint,
) -> FieldMatch {
    FieldMatch {
        graph_part: a.graph_part == b.graph_part,
        formulas: a.formulas == b.formulas,
        solved_formulas: a.solved_formulas == b.solved_formulas,
        lemmas: a.lemmas == b.lemmas,
        eq_store_subst: a.eq_store.subst == b.eq_store.subst,
        eq_store_conj: a.eq_store.conj == b.eq_store.conj,
        subterm_store_subterms: a.subterm_store.subterms == b.subterm_store.subterms,
        subterm_store_solved_subterms: a.subterm_store.solved_subterms
            == b.subterm_store.solved_subterms,
        subterm_store_contradictory: a.subterm_store.contradictory
            == b.subterm_store.contradictory,
        subterm_store_neg_subterms: a.subterm_store.neg_subterms == b.subterm_store.neg_subterms,
        goals: a.goals == b.goals,
        source_kind: a.source_kind == b.source_kind,
        side: a.side == b.side,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::guarded::{gfalse, gtrue};
    use tamarin_term::lterm::{LSort, LVar};
    use tamarin_term::vterm::var_term;

    fn term(name: &str, idx: u64) -> LNTerm {
        var_term(LVar::new(name, LSort::Msg, idx))
    }

    fn empty_eq_store() -> CanonicalEqStore {
        CanonicalEqStore {
            subst: Vec::new(),
            conj: Vec::new(),
        }
    }

    fn empty_subterm_store() -> CanonicalSubtermStore {
        CanonicalSubtermStore {
            subterms: Vec::new(),
            solved_subterms: Vec::new(),
            contradictory: false,
            neg_subterms: Vec::new(),
        }
    }

    /// A minimal, otherwise-empty `CanonicalSystem` -- every test below
    /// starts from this and changes exactly ONE field, so a `FieldMatch`
    /// against the baseline isolates that one field precisely.
    fn baseline_system() -> CanonicalSystem {
        CanonicalSystem {
            graph_part: term("g", 0),
            formulas: Vec::new(),
            solved_formulas: Vec::new(),
            lemmas: Vec::new(),
            eq_store: empty_eq_store(),
            subterm_store: empty_subterm_store(),
            goals: Vec::new(),
            source_kind: None,
            side: None,
        }
    }

    #[test]
    fn fingerprinting_is_deterministic() {
        let sys = baseline_system();
        assert_eq!(
            fingerprint_constraint_system(&sys),
            fingerprint_constraint_system(&sys)
        );
    }

    #[test]
    fn identical_systems_fully_match() {
        let a = fingerprint_constraint_system(&baseline_system());
        let b = fingerprint_constraint_system(&baseline_system());
        let report = compare_fingerprints(&a, &b);
        assert!(report.is_full_match());
        assert_eq!(report.matched_count(), report.total_count());
        assert!(report.mismatched_fields().is_empty());
    }

    #[test]
    fn systems_differing_only_in_graph_part_mismatch_only_that_field() {
        let a = fingerprint_constraint_system(&baseline_system());
        let mut sys_b = baseline_system();
        sys_b.graph_part = term("different", 0);
        let b = fingerprint_constraint_system(&sys_b);

        let report = compare_fingerprints(&a, &b);
        assert!(!report.is_full_match());
        assert_eq!(report.matched_count(), report.total_count() - 1);
        assert_eq!(report.mismatched_fields(), vec!["graph_part"]);
    }

    #[test]
    fn systems_differing_only_in_formulas_mismatch_only_that_field() {
        let a = fingerprint_constraint_system(&baseline_system());
        let mut sys_b = baseline_system();
        sys_b.formulas = vec![gtrue()];
        let b = fingerprint_constraint_system(&sys_b);

        let report = compare_fingerprints(&a, &b);
        assert_eq!(report.mismatched_fields(), vec!["formulas"]);
    }

    #[test]
    fn systems_differing_only_in_side_mismatch_only_that_field() {
        let a = fingerprint_constraint_system(&baseline_system());
        let mut sys_b = baseline_system();
        sys_b.side = Some(Side::LHS);
        let b = fingerprint_constraint_system(&sys_b);

        let report = compare_fingerprints(&a, &b);
        assert_eq!(report.mismatched_fields(), vec!["side"]);
        // `side`/`source_kind` are kept as native values, not hashed --
        // confirm the fingerprint itself actually carries the real value,
        // not just a bool of whether it was set.
        assert_eq!(a.side, None);
        assert_eq!(b.side, Some(Side::LHS));
    }

    #[test]
    fn systems_differing_only_in_subterm_store_contradictory_mismatch_only_that_field() {
        let a = fingerprint_constraint_system(&baseline_system());
        let mut sys_b = baseline_system();
        sys_b.subterm_store.contradictory = true;
        let b = fingerprint_constraint_system(&sys_b);

        let report = compare_fingerprints(&a, &b);
        assert_eq!(
            report.mismatched_fields(),
            vec!["subterm_store.contradictory"]
        );
    }

    #[test]
    fn systems_differing_only_in_eq_store_conj_mismatch_only_that_field() {
        let a = fingerprint_constraint_system(&baseline_system());
        let mut sys_b = baseline_system();
        sys_b.eq_store.conj = vec![vec![vec![(
            Lit::Var(LVar::new("x", LSort::Msg, 0)),
            term("y", 0),
        )]]];
        let b = fingerprint_constraint_system(&sys_b);

        let report = compare_fingerprints(&a, &b);
        assert_eq!(report.mismatched_fields(), vec!["eq_store.conj"]);
    }

    #[test]
    fn systems_differing_only_in_goals_mismatch_only_that_field() {
        let a = fingerprint_constraint_system(&baseline_system());
        let mut sys_b = baseline_system();
        sys_b.goals = vec![CanonicalGoal {
            kind: CanonicalGoalKind::Subterm(term("a", 0), term("b", 0)),
            solved: false,
        }];
        let b = fingerprint_constraint_system(&sys_b);

        let report = compare_fingerprints(&a, &b);
        assert_eq!(report.mismatched_fields(), vec!["goals"]);
    }

    /// Two goals with the IDENTICAL canonicalized content but different
    /// `solved` status must fingerprint DIFFERENTLY -- `solved` is
    /// content-relevant (see `CanonicalGoal`'s own doc comment: it gates
    /// whether `candidate_methods` offers the goal again), so collapsing
    /// it would make two systems with genuinely different remaining
    /// proof-search obligations look identical.
    #[test]
    fn goals_differing_only_in_solved_status_fingerprint_differently() {
        let mut sys_a = baseline_system();
        sys_a.goals = vec![CanonicalGoal {
            kind: CanonicalGoalKind::Subterm(term("a", 0), term("b", 0)),
            solved: false,
        }];
        let mut sys_b = baseline_system();
        sys_b.goals = vec![CanonicalGoal {
            kind: CanonicalGoalKind::Subterm(term("a", 0), term("b", 0)),
            solved: true,
        }];

        let a = fingerprint_constraint_system(&sys_a);
        let b = fingerprint_constraint_system(&sys_b);
        assert_ne!(a.goals, b.goals);
    }

    /// A genuine "close match": several fields differ at once, and the
    /// report must name EXACTLY those, in field order, leaving every
    /// untouched field reported as matching.
    #[test]
    fn multiple_field_differences_are_all_reported() {
        let a = fingerprint_constraint_system(&baseline_system());
        let mut sys_b = baseline_system();
        sys_b.formulas = vec![gtrue()];
        sys_b.lemmas = vec![gfalse()];
        sys_b.source_kind = Some(SourceKind::RawSources);
        let b = fingerprint_constraint_system(&sys_b);

        let report = compare_fingerprints(&a, &b);
        assert!(!report.is_full_match());
        assert_eq!(report.matched_count(), report.total_count() - 3);
        assert_eq!(
            report.mismatched_fields(),
            vec!["formulas", "lemmas", "source_kind"]
        );
    }

    /// A degenerate but legal "close match" of zero: every field differs.
    #[test]
    fn completely_different_systems_mismatch_every_field() {
        let a = fingerprint_constraint_system(&baseline_system());
        let sys_b = CanonicalSystem {
            graph_part: term("different", 0),
            formulas: vec![gtrue()],
            solved_formulas: vec![gtrue()],
            lemmas: vec![gtrue()],
            eq_store: CanonicalEqStore {
                subst: vec![(Lit::Var(LVar::new("x", LSort::Msg, 0)), term("y", 0))],
                conj: vec![vec![vec![(
                    Lit::Var(LVar::new("x", LSort::Msg, 0)),
                    term("y", 0),
                )]]],
            },
            subterm_store: CanonicalSubtermStore {
                subterms: vec![(term("a", 0), term("b", 0))],
                solved_subterms: vec![(term("a", 0), term("b", 0))],
                contradictory: true,
                neg_subterms: vec![(term("a", 0), term("b", 0))],
            },
            goals: vec![CanonicalGoal {
                kind: CanonicalGoalKind::Subterm(term("a", 0), term("b", 0)),
                solved: false,
            }],
            source_kind: Some(SourceKind::RawSources),
            side: Some(Side::LHS),
        };
        let b = fingerprint_constraint_system(&sys_b);

        let report = compare_fingerprints(&a, &b);
        assert_eq!(report.matched_count(), 0);
        assert_eq!(report.mismatched_fields().len(), report.total_count());
    }

    /// `eq_store.conj`'s alternatives are a MULTISET, not deduplicated
    /// (see `CanonicalEqStore::conj`'s own doc comment in `canon.rs`) --
    /// confirm fingerprinting doesn't accidentally collapse two
    /// occurrences of the same alternative into one, which would make two
    /// systems with a different alpha-equivalent-alternative COUNT
    /// fingerprint identically.
    #[test]
    fn eq_store_conj_fingerprint_preserves_multiplicity() {
        let alt = vec![(
            Lit::Var(LVar::new("x", LSort::Msg, 0)),
            term("y", 0),
        )];
        let mut sys_one = baseline_system();
        sys_one.eq_store.conj = vec![vec![alt.clone()]];
        let mut sys_two = baseline_system();
        sys_two.eq_store.conj = vec![vec![alt.clone(), alt]];

        let fp_one = fingerprint_constraint_system(&sys_one);
        let fp_two = fingerprint_constraint_system(&sys_two);
        assert_ne!(fp_one.eq_store.conj, fp_two.eq_store.conj);
    }
}
