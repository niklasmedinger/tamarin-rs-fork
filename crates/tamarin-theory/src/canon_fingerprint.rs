// Currently GPL 3.0 until granted permission by the upstream authors
// of the tamarin-prover sources this file cites; list them with:
//   scripts/gen_license_headers.py --authors <this file>

//! Per-SECTION fingerprinting of a [`CanonicalSystem`] -- deliberately NOT
//! one combined hash.
//!
//! [`CanonicalSystemFingerprint`] keeps one [`Fingerprint`] per section of the
//! canonical form (`canon_term_graph::SYSTEM_SECTIONS`) and the form's label
//! as plain values. Two systems that differ in one section only (e.g. the
//! equation store's disjunctions) are far more useful to tell apart from two
//! unrelated systems, and a single combined hash cannot: both just come out
//! "different". [`compare_fingerprints`] reports which sections matched
//! ([`FieldMatch::mismatched_fields`]), using only the compact fingerprints.
//!
//! The table of a merging search keys on the whole tuple, so it merges
//! exactly the systems with equal forms (up to a SHA-256 collision).

use crate::canon::{CanonicalProofMethod, CanonicalSystem};
use crate::canon_term_graph::{fingerprint_sterm, SYSTEM_SECTIONS};
use crate::constraint::system::{Side, SourceKind};

use tamarin_utils::fingerprint::{Fingerprint, FingerprintHasher};

/// How many sections a canonical form has.
const SECTION_COUNT: usize = SYSTEM_SECTIONS.len();

/// The fingerprints of a [`CanonicalSystem`]: one per section, and its label
/// as plain values (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CanonicalSystemFingerprint {
    /// Small `Copy` enums, kept as values rather than hashed, so a caller
    /// can see WHAT differs, not just THAT it does.
    pub source_kind: Option<SourceKind>,
    pub side: Option<Side>,
    /// The subterm store's `contradictory` flag.
    pub contradictory: bool,
    /// `sections[i]` fingerprints the section `SYSTEM_SECTIONS[i]`.
    pub sections: [Fingerprint; SECTION_COUNT],
}

/// Fingerprints every section of `sys` independently (see the module docs).
pub fn fingerprint_constraint_system(sys: &CanonicalSystem) -> CanonicalSystemFingerprint {
    let (source_kind, side, contradictory) = sys.label();
    let sections: Vec<Fingerprint> = sys.sections().iter().map(fingerprint_sterm).collect();
    CanonicalSystemFingerprint {
        source_kind,
        side,
        contradictory,
        sections: sections
            .try_into()
            .expect("a canonical form has one section per SYSTEM_SECTIONS entry"),
    }
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
        CanonicalProofMethod::Solve(goal) => {
            h.tag("Solve").digest(&fingerprint_sterm(goal));
        }
        CanonicalProofMethod::SolveSplit(disj) => {
            h.tag("SolveSplit").digest(&fingerprint_sterm(disj));
        }
    }
    h.finish()
}

/// How many fields a [`FieldMatch`] compares: the three label values and the
/// sections.
const FIELD_COUNT: usize = 3 + SECTION_COUNT;

/// A field-by-field comparison of two [`CanonicalSystemFingerprint`]s: which
/// label values and sections agreed, not just whether the two systems match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldMatch {
    fields: [(&'static str, bool); FIELD_COUNT],
}

impl FieldMatch {
    /// How many of the fields matched.
    pub fn matched_count(&self) -> usize {
        self.fields.iter().filter(|(_, matched)| *matched).count()
    }

    /// The number of fields compared (always [`FIELD_COUNT`]), for an
    /// "N of M" score next to [`Self::matched_count`].
    pub fn total_count(&self) -> usize {
        FIELD_COUNT
    }

    /// Every field that did NOT match: `source_kind`, `side`,
    /// `contradictory`, then the sections in `SYSTEM_SECTIONS` order.
    pub fn mismatched_fields(&self) -> Vec<&'static str> {
        self.fields
            .iter()
            .filter(|(_, matched)| !matched)
            .map(|(name, _)| *name)
            .collect()
    }

    /// True iff every field matched.
    pub fn is_full_match(&self) -> bool {
        self.matched_count() == self.total_count()
    }
}

/// Compares two [`CanonicalSystemFingerprint`]s field by field.
pub fn compare_fingerprints(
    a: &CanonicalSystemFingerprint,
    b: &CanonicalSystemFingerprint,
) -> FieldMatch {
    let mut fields = [("", false); FIELD_COUNT];
    fields[0] = ("source_kind", a.source_kind == b.source_kind);
    fields[1] = ("side", a.side == b.side);
    fields[2] = ("contradictory", a.contradictory == b.contradictory);
    for (i, name) in SYSTEM_SECTIONS.iter().enumerate() {
        fields[3 + i] = (name, a.sections[i] == b.sections[i]);
    }
    FieldMatch { fields }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canon_term_graph::{Op, Section, STerm, VarRef};
    use tamarin_term::lterm::{LSort, LVar};

    const SECTIONS: [Section; 12] = [
        Section::Nodes,
        Section::Edges,
        Section::LessAtoms,
        Section::Formulas,
        Section::SolvedFormulas,
        Section::Lemmas,
        Section::EqSubst,
        Section::EqConj,
        Section::Subterms,
        Section::SolvedSubterms,
        Section::NegSubterms,
        Section::Goals,
    ];

    /// A canonical form with every section empty, `last` absent, and the
    /// given label.
    fn form_with(
        contradictory: bool,
        change: impl Fn(&str, STerm) -> STerm,
    ) -> CanonicalSystem {
        let mut sections: Vec<STerm> = SECTIONS
            .iter()
            .map(|s| STerm::App(Op::Section(*s), Vec::new()))
            .collect();
        sections.insert(3, STerm::Const(crate::canon_term_graph::ConstKey::NoLast));
        let sections = SYSTEM_SECTIONS
            .iter()
            .zip(sections)
            .map(|(name, section)| change(name, section))
            .collect();
        CanonicalSystem::new(STerm::App(
            Op::System {
                source_kind: None,
                side: None,
                contradictory,
            },
            sections,
        ))
    }

    fn baseline() -> CanonicalSystem {
        form_with(false, |_, section| section)
    }

    /// `name`'s section with one variable more.
    fn changed(name: &'static str) -> CanonicalSystem {
        form_with(false, move |n, section| match section {
            STerm::App(op, mut args) if n == name => {
                args.push(STerm::Var(VarRef::Free(LVar::new("x", LSort::Msg, 0))));
                STerm::App(op, args)
            }
            STerm::Const(_) if n == name => STerm::App(
                Op::Last,
                vec![STerm::Var(VarRef::Free(LVar::new("t", LSort::Node, 0)))],
            ),
            section => section,
        })
    }

    #[test]
    fn equal_forms_have_equal_fingerprints() {
        let (a, b) = (baseline(), baseline());
        assert_eq!(
            fingerprint_constraint_system(&a),
            fingerprint_constraint_system(&b)
        );
        let report = compare_fingerprints(
            &fingerprint_constraint_system(&a),
            &fingerprint_constraint_system(&b),
        );
        assert!(report.is_full_match());
        assert_eq!(report.matched_count(), FIELD_COUNT);
        assert!(report.mismatched_fields().is_empty());
    }

    /// Changing one section changes exactly that section's fingerprint.
    #[test]
    fn each_section_is_reported_on_its_own() {
        let base = fingerprint_constraint_system(&baseline());
        for name in SYSTEM_SECTIONS {
            let other = fingerprint_constraint_system(&changed(name));
            let report = compare_fingerprints(&base, &other);
            assert_eq!(report.mismatched_fields(), vec![name], "changing {name}");
            assert_eq!(report.matched_count(), FIELD_COUNT - 1);
        }
    }

    #[test]
    fn the_contradictory_flag_is_its_own_field() {
        let report = compare_fingerprints(
            &fingerprint_constraint_system(&baseline()),
            &fingerprint_constraint_system(&form_with(true, |_, section| section)),
        );
        assert_eq!(report.mismatched_fields(), vec!["contradictory"]);
    }

    #[test]
    fn proof_methods_fingerprint_by_variant_and_payload() {
        let goal = |name: &str| STerm::Var(VarRef::Free(LVar::new(name, LSort::Msg, 0)));
        let fp = fingerprint_proof_method;
        assert_eq!(
            fp(&CanonicalProofMethod::Solve(goal("x"))),
            fp(&CanonicalProofMethod::Solve(goal("x")))
        );
        assert_ne!(
            fp(&CanonicalProofMethod::Solve(goal("x"))),
            fp(&CanonicalProofMethod::Solve(goal("y")))
        );
        assert_ne!(
            fp(&CanonicalProofMethod::Solve(goal("x"))),
            fp(&CanonicalProofMethod::SolveSplit(goal("x")))
        );
        assert_ne!(
            fp(&CanonicalProofMethod::Simplify),
            fp(&CanonicalProofMethod::Induction)
        );
    }
}
