// Currently GPL 3.0 until granted permission by the upstream authors
// of the tamarin-prover sources this file cites; list them with:
//   scripts/gen_license_headers.py --authors <this file>

//! (No HS analog.) Builds the vertex-coloring `c: V -> N` `TODO.md`'s
//! "Canonizing the graph" section specifies for a [`crate::canon_graph::GraphPart`]
//! — Stage B of "Canonizing the constraint system", following Stage A
//! ([`crate::canon_graph::extract_graph_part`]).
//!
//! Implements exactly the THREE reserved-color-block scheme `TODO.md`
//! describes first, not the further skeleton-strengthening refinement
//! (erase literals to sorts, `CAN_AC` the result) it goes on to sketch —
//! that is a separate, later enhancement, not attempted here:
//!
//! 1. This crate's own structural relation-vertex kinds
//!    (`VertexKind::Dummy`/`EdgeRelation`/`LessRelation`/
//!    `AtTimepointRelation`) — fixed, identical across every theory.
//! 2. Tamarin's built-in rule names and built-in action names — also
//!    fixed, identical across every theory (see
//!    [`BUILTIN_RULE_NAMES`]/[`BUILTIN_ACTION_NAMES`] for exactly which
//!    names and how they were derived).
//! 3. One specific theory's own protocol rule names and protocol action
//!    names — collected from the theory, each sorted independently
//!    (`BTreeSet`, so the source file's declaration order never leaks
//!    in), and colored in that order. The only block that varies by
//!    theory — the reason [`ColorTable::build`] needs a `&Theory` at
//!    all, rather than blocks 1–2 being the whole story.
//!
//! Within each of blocks 2 and 3, rule names are colored before action
//! names (two separate sorted sets, not one merged alphabetical list) —
//! so a rule and an action that happen to share a literal name (e.g. a
//! contrived theory with both a rule and an action fact named `"Foo"`)
//! still get different colors, for free, since rule-name and
//! action-name colors always land in disjoint sub-ranges.
//!
//! **Soundness, not completeness, is the bar** (`TODO.md`'s own words:
//! "the soundness condition on a color function is only that it be
//! constant on alphaeqac-classes... finer is better but never
//! necessary"): [`ColorTable::rule_color`]/[`action_color`] PANIC on a
//! name outside every block (documented per-function) rather than
//! silently guessing — this table does not (yet) cover
//! theory-dependent intruder-deduction rule names (construction/
//! destruction rules synthesized from `functions:`/`builtins:`
//! declarations, e.g. a `senc`/`sdec` pair's rules), which is a real,
//! open gap flagged here rather than papered over.
//!
//! NOT [`crate::constraint::system::graph::color`] — see
//! `canon_graph.rs`'s module docs for why that "color" (a cosmetic HSV
//! fill palette for DOT rendering) is a different concept from this one
//! (graph-theoretic vertex coloring for the canonizer) despite the
//! shared word.

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use crate::canon_graph::VertexKind;
use crate::fact::fact_tag_name;
use crate::rule::{IntrRuleACInfo, ProtoRuleName, RuleACInst, RuleInfo};
use crate::theory::Theory;

/// A vertex color: an arbitrary but STABLE non-negative integer. Two
/// vertices sharing a color are indistinguishable to the graph
/// canonizer's initial partition; see the module docs for the
/// soundness bar this has to clear (constant on $\alphaeqac$-classes).
pub type Color = u32;

// =============================================================================
// Block 1 — structural relation-vertex kinds (fixed, universal)
// =============================================================================

const STRUCTURAL_DUMMY: Color = 0;
const STRUCTURAL_EDGE_RELATION: Color = 1;
const STRUCTURAL_LESS_RELATION: Color = 2;
const STRUCTURAL_AT_TIMEPOINT_RELATION: Color = 3;
/// One past the last structural color — where block 2 starts.
const STRUCTURAL_BLOCK_SIZE: Color = 4;

// =============================================================================
// Block 2 — Tamarin built-ins (fixed, universal)
// =============================================================================

/// Every name [`rule_name_string`]/`intr_rule_name_string` can produce
/// for a rule instance that ISN'T declared in a `.spthy` file's own
/// `rule NAME: ...` blocks, restricted to the names that don't vary by
/// theory (i.e. excluding `IntrRuleACInfo::ConstrRule`/`DestrRule` —
/// construction/destruction rules synthesized from a theory's own
/// `functions:`/`builtins:` declarations, and therefore genuinely
/// theory-specific despite also being "not user-written" — see the
/// module docs' completeness caveat).
///
/// Derived directly from `rule::intr_rule_name_string`'s match arms
/// (`Coerce`/`IRecv`→`"Recv"`/`ISend`→`"Send"`/`PubConstr`/
/// `NatConstr`/`FreshConstr`/`IEquality`→`"Equality"`) plus
/// `ProtoRuleName::Fresh`'s own rendering (`"FreshRule"`, per
/// `rule_name_string`'s `RuleInfo::Proto` arm) — NOT
/// `rule::RESERVED_RULE_NAMES`, which is a checked-against-user-input
/// reserved-WORD list for a different purpose (rejecting a user's own
/// rule from being named e.g. `"pub"`/`"fresh"`) and uses different
/// spellings (lowercase `"irecv"`/`"isend"`, bare `"Fresh"`) that never
/// actually appear as a real `RuleACInst`'s rendered name.
///
/// Sorted alphabetically — this array's OWN index order is its color
/// assignment order (`STRUCTURAL_BLOCK_SIZE + position`, see
/// [`builtin_rule_color`]), so declaration order here matters.
pub const BUILTIN_RULE_NAMES: [&str; 8] = [
    "Coerce",
    "Equality",
    "FreshConstr",
    "FreshRule",
    "NatConstr",
    "PubConstr",
    "Recv",
    "Send",
];

/// Every built-in action-fact NAME: the fixed (non-`Proto`) `FactTag`
/// variants' display names (`fact_tag_name`: `Fr`/`Out`/`In`/`KU`/`KD`/
/// `Ded`/`Term`), plus `"K"` — a `FactTag::Proto(Linear, "K", 1)` fact
/// by construction (see `fact.rs`'s `k_log_fact`/HS `kLogFact`), but a
/// genuinely built-in one: the intruder-knowledge logging action every
/// `ISend`/`IRecv`/construction/destruction rule emits, not something a
/// user's protocol theory declares. Deliberately excludes `"Smaller"`
/// (`predicate::smaller_fact`): that is a pattern used to look up a
/// user-declared `Smaller` predicate DEFINITION, not itself a fact tag
/// that occurs as an actual rule action.
///
/// **Not proven exhaustive against every construct HS ships** — flagged
/// explicitly (see the module docs) rather than silently assumed
/// complete; [`ColorTable::action_color`] panics on a miss instead of
/// guessing, so an omission here surfaces loudly rather than silently
/// mis-coloring.
///
/// Sorted alphabetically — this array's OWN index order is its color
/// assignment order (`STRUCTURAL_BLOCK_SIZE + BUILTIN_RULE_NAMES.len() +
/// position`, see [`ColorTable::action_color`]), so declaration order
/// here is load-bearing.
pub const BUILTIN_ACTION_NAMES: [&str; 8] = ["Ded", "Fr", "In", "K", "KD", "KU", "Out", "Term"];

// =============================================================================
// The table
// =============================================================================

/// The full vertex-coloring table for one theory. See the module docs
/// for the three-block scheme.
///
/// Only block 3 (this theory's own protocol rule/action names) is
/// actually STORED — blocks 1 and 2 are fixed for every theory, so
/// their colors are computed directly from [`BUILTIN_RULE_NAMES`]/
/// [`BUILTIN_ACTION_NAMES`]'s positions rather than carried in an
/// instance.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ColorTable {
    /// Keyed by the rule's INTERNED name (`ProtoRuleName::Stand`'s
    /// payload is already `&'static str` — see `rule.rs`'s own doc
    /// comment on that field), not a rendered/allocated `String`.
    theory_rule_colors: BTreeMap<&'static str, Color>,
    theory_action_colors: BTreeMap<String, Color>,
}

impl ColorTable {
    /// Builds the color table for `theory`. Deterministic given the
    /// theory's set of protocol rule/action NAMES alone: the same
    /// theory always produces the same table, and two theories with the
    /// same rule/action names produce the same table too, regardless of
    /// what order those rules happen to be declared in the source file
    /// (collected into a `BTreeSet` before any color is assigned).
    pub fn build(theory: &Theory) -> Self {
        let builtin_actions: BTreeSet<&str> = BUILTIN_ACTION_NAMES.into_iter().collect();

        // Rule names: every `theory.rules()` entry is a `ProtoRuleE`
        // (`Rule<ProtoRuleEInfo>`) — i.e. genuinely a user-declared
        // protocol rule (the open-theory level has no `RuleInfo`/`Intr`
        // case to collide with at all: that wrapper only appears once
        // rules are AC-instantiated for proof search). So there is
        // nothing to filter here, unlike actions below.
        let mut theory_rule_names: BTreeSet<&'static str> = BTreeSet::new();
        let mut theory_action_names: BTreeSet<String> = BTreeSet::new();
        for r in theory.rules() {
            if let ProtoRuleName::Stand(s) = r.rule.info.name {
                theory_rule_names.insert(s);
            }
            for fa in &r.rule.actions {
                theory_action_names.insert(fact_tag_name(&fa.tag));
            }
        }
        // Actions have no structural tag distinguishing "built-in" from
        // "user-declared" (`GFact`/`LNFact` carry only a bare name
        // string — see `BUILTIN_ACTION_NAMES`'s own doc comment), so a
        // theory action fact literally named e.g. `"K"` is
        // INDISTINGUISHABLE from the built-in `K` logging action and
        // deliberately reuses its color rather than getting a second,
        // conflicting one for the same string.
        theory_action_names.retain(|n| !builtin_actions.contains(n.as_str()));

        // Block 3 starts AFTER both builtin sub-blocks (2a rules, 2b
        // actions) — matching `action_color`'s own on-the-fly builtin
        // computation, which places 2b right after 2a. The two must
        // agree on this base or two different sub-blocks silently claim
        // the same colors (caught by this module's own
        // `builtin_and_theory_blocks_never_overlap` test — a real bug
        // the first version of this function had, before block 3 was
        // fixed to start here rather than right after block 2a).
        let mut next: Color =
            STRUCTURAL_BLOCK_SIZE + BUILTIN_RULE_NAMES.len() as Color + BUILTIN_ACTION_NAMES.len() as Color;
        let mut theory_rule_colors: BTreeMap<&'static str, Color> = BTreeMap::new();
        for name in &theory_rule_names {
            theory_rule_colors.insert(name, next);
            next += 1;
        }

        let mut theory_action_colors: BTreeMap<String, Color> = BTreeMap::new();
        for name in &theory_action_names {
            theory_action_colors.insert(name.clone(), next);
            next += 1;
        }

        ColorTable {
            theory_rule_colors,
            theory_action_colors,
        }
    }

    /// The color for a rule INSTANCE (block 2a or 3a). Dispatches on
    /// `ru.info`'s actual STRUCTURE (`RuleInfo::Proto` vs `Intr`, and
    /// for `Intr`, which fixed variant) rather than the rendered display
    /// string (`rule::rule_name_string`) — this matters: a theory can
    /// perfectly well declare its own protocol rule literally named
    /// `"Send"`, which renders IDENTICALLY to the built-in `ISend`
    /// intruder rule's name but is a completely different rule with a
    /// completely different `RuleInfo::Proto(..)` origin. Dispatching on
    /// the string would have silently and wrongly colored that user
    /// rule as if it were the built-in one (caught by testing against a
    /// real captured system whose `Send` rule was the protocol's own,
    /// not the intruder's — see this module's test of the same name).
    ///
    /// Panics if `ru` is an intruder construction/destruction rule (the
    /// one `RuleInfo::Intr` case this table doesn't cover — see the
    /// module docs' completeness caveat) or a `Stand` name this table
    /// wasn't built from (a caller/table mismatch bug, not expected in
    /// normal use).
    pub fn rule_color(&self, ru: &RuleACInst) -> Color {
        match &ru.info {
            RuleInfo::Proto(p) => match p.name {
                ProtoRuleName::Fresh => builtin_rule_color("FreshRule"),
                ProtoRuleName::Stand(s) => self.theory_rule_colors.get(s).copied().unwrap_or_else(|| {
                    panic!(
                        "ColorTable::rule_color: protocol rule {s:?} is not a name \
                         this table was built from (table/system theory mismatch?)"
                    )
                }),
            },
            RuleInfo::Intr(info) => match builtin_intr_rule_name(info) {
                Some(name) => builtin_rule_color(name),
                None => panic!(
                    "ColorTable::rule_color: {info:?} is an intruder construction/ \
                     destruction rule -- theory-dependent (synthesized from this \
                     theory's own functions:/builtins: declarations), and not yet \
                     covered by this table (see the module docs' completeness \
                     caveat)"
                ),
            },
        }
    }

    /// The color for an action-fact name (block 2b or 3b). Panics if
    /// `name` is neither a built-in nor a name this table was built
    /// from — see the module docs' completeness caveat.
    pub fn action_color(&self, name: &str) -> Color {
        if let Some(pos) = BUILTIN_ACTION_NAMES.iter().position(|n| *n == name) {
            return STRUCTURAL_BLOCK_SIZE + BUILTIN_RULE_NAMES.len() as Color + pos as Color;
        }
        self.theory_action_colors.get(name).copied().unwrap_or_else(|| {
            panic!(
                "ColorTable::action_color: {name:?} is neither a built-in action \
                 name nor a protocol action name this table was built from (see \
                 this module's completeness caveat)"
            )
        })
    }

    /// The color for any [`VertexKind`] — the single entry point a
    /// caller building the full coloring `c: V -> N` for a `GraphPart`
    /// actually wants (`part.vertices.iter().map(|v| table.vertex_color(v))`).
    pub fn vertex_color(&self, v: &VertexKind) -> Color {
        match v {
            VertexKind::Dummy(_) => STRUCTURAL_DUMMY,
            VertexKind::EdgeRelation => STRUCTURAL_EDGE_RELATION,
            VertexKind::LessRelation => STRUCTURAL_LESS_RELATION,
            VertexKind::AtTimepointRelation => STRUCTURAL_AT_TIMEPOINT_RELATION,
            VertexKind::RuleInstance(_, ru) => self.rule_color(ru),
            VertexKind::Action(_, fact) => self.action_color(&fact.name),
        }
    }
}

/// The color of a name known to be in [`BUILTIN_RULE_NAMES`] — panics
/// (an internal-consistency bug, not a user-facing error) if it somehow
/// isn't, since every caller of this function has already established
/// membership.
fn builtin_rule_color(name: &str) -> Color {
    let pos = BUILTIN_RULE_NAMES
        .iter()
        .position(|n| *n == name)
        .unwrap_or_else(|| panic!("builtin_rule_color: {name:?} not in BUILTIN_RULE_NAMES"));
    STRUCTURAL_BLOCK_SIZE + pos as Color
}

/// The [`BUILTIN_RULE_NAMES`] entry for a fixed (theory-independent)
/// `IntrRuleACInfo` variant, or `None` for `ConstrRule`/`DestrRule`
/// (theory-dependent — see the module docs' completeness caveat).
fn builtin_intr_rule_name(info: &IntrRuleACInfo) -> Option<&'static str> {
    match info {
        IntrRuleACInfo::Coerce => Some("Coerce"),
        IntrRuleACInfo::IRecv => Some("Recv"),
        IntrRuleACInfo::ISend => Some("Send"),
        IntrRuleACInfo::PubConstr => Some("PubConstr"),
        IntrRuleACInfo::NatConstr => Some("NatConstr"),
        IntrRuleACInfo::FreshConstr => Some("FreshConstr"),
        IntrRuleACInfo::IEquality => Some("Equality"),
        IntrRuleACInfo::ConstrRule { .. } | IntrRuleACInfo::DestrRule { .. } => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::{ProtoRuleACInstInfo, Rule, RuleAttributes};
    use tamarin_parser::parser::parse_theory;

    fn theory(src: &str) -> Theory {
        let parsed = parse_theory(src, &[]).unwrap_or_else(|e| panic!("parse: {e}"));
        crate::elaborate::elaborate(&parsed).unwrap_or_else(|e| panic!("elaborate: {e:?}"))
    }

    const EMPTY: &str = "theory T begin\nend";

    const TWO_RULES: &str = "theory T begin\n\
        rule Zebra:\n  [] --[ Beta() ]-> []\n\
        rule Apple:\n  [] --[ Alpha() ]-> []\n\
        end";

    /// A standalone `RuleACInst` for a user-declared protocol rule named
    /// `name` — for exercising `ColorTable::rule_color` directly,
    /// without needing a full parsed/elaborated theory.
    fn proto_rule_instance(name: &'static str) -> RuleACInst {
        Rule::new(
            RuleInfo::Proto(ProtoRuleACInstInfo {
                name: ProtoRuleName::Stand(name),
                attributes: RuleAttributes::default(),
                loop_breakers: Vec::new(),
            }),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
    }

    fn fresh_rule_instance() -> RuleACInst {
        Rule::new(
            RuleInfo::Proto(ProtoRuleACInstInfo {
                name: ProtoRuleName::Fresh,
                attributes: RuleAttributes::default(),
                loop_breakers: Vec::new(),
            }),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
    }

    fn intr_rule_instance(info: IntrRuleACInfo) -> RuleACInst {
        Rule::new(RuleInfo::Intr(info), Vec::new(), Vec::new(), Vec::new())
    }

    fn nid() -> crate::constraint::constraints::NodeId {
        tamarin_term::lterm::LVar::new("i", tamarin_term::lterm::LSort::Node, 0)
    }

    #[test]
    fn structural_colors_are_fixed_and_distinct() {
        let table = ColorTable::build(&theory(EMPTY));
        let colors = [
            table.vertex_color(&VertexKind::Dummy(nid())),
            table.vertex_color(&VertexKind::EdgeRelation),
            table.vertex_color(&VertexKind::LessRelation),
            table.vertex_color(&VertexKind::AtTimepointRelation),
        ];
        let mut sorted = colors.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 4, "structural colors must be pairwise distinct");
        // Fixed means fixed: same theory-independent values every time.
        assert_eq!(colors, [0, 1, 2, 3]);
    }

    #[test]
    fn empty_theory_still_colors_every_builtin_name() {
        let table = ColorTable::build(&theory(EMPTY));
        table.rule_color(&fresh_rule_instance());
        for info in [
            IntrRuleACInfo::Coerce,
            IntrRuleACInfo::IRecv,
            IntrRuleACInfo::ISend,
            IntrRuleACInfo::PubConstr,
            IntrRuleACInfo::NatConstr,
            IntrRuleACInfo::FreshConstr,
            IntrRuleACInfo::IEquality,
        ] {
            table.rule_color(&intr_rule_instance(info)); // must not panic
        }
        for name in BUILTIN_ACTION_NAMES {
            table.action_color(name); // must not panic
        }
    }

    #[test]
    fn builtin_and_theory_blocks_never_overlap() {
        let table = ColorTable::build(&theory(TWO_RULES));
        let mut all: Vec<Color> = vec![
            table.vertex_color(&VertexKind::Dummy(nid())),
            table.vertex_color(&VertexKind::EdgeRelation),
            table.vertex_color(&VertexKind::LessRelation),
            table.vertex_color(&VertexKind::AtTimepointRelation),
            table.rule_color(&fresh_rule_instance()),
            table.rule_color(&intr_rule_instance(IntrRuleACInfo::ISend)),
            table.rule_color(&intr_rule_instance(IntrRuleACInfo::IRecv)),
            table.rule_color(&proto_rule_instance("Apple")),
            table.rule_color(&proto_rule_instance("Zebra")),
        ];
        all.extend(BUILTIN_ACTION_NAMES.map(|n| table.action_color(n)));
        all.push(table.action_color("Alpha"));
        all.push(table.action_color("Beta"));
        let mut sorted = all.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), all.len(), "every color must be unique");
    }

    /// The concrete ask this whole module exists for: the theory's own
    /// rule/action names are sorted INDEPENDENTLY of source-file
    /// declaration order (`Zebra` declared before `Apple`, `Beta`
    /// before `Alpha`, but colors must come out alphabetical).
    #[test]
    fn theory_specific_names_are_colored_in_sorted_not_declaration_order() {
        let table = ColorTable::build(&theory(TWO_RULES));
        let apple = table.rule_color(&proto_rule_instance("Apple"));
        let zebra = table.rule_color(&proto_rule_instance("Zebra"));
        assert!(apple < zebra, "Apple ({apple}) should sort before Zebra ({zebra})");

        let alpha = table.action_color("Alpha");
        let beta = table.action_color("Beta");
        assert!(alpha < beta, "Alpha ({alpha}) should sort before Beta ({beta})");
    }

    /// The headline invariant: building the table twice from
    /// independently-parsed copies of the SAME theory text gives the
    /// SAME table.
    #[test]
    fn same_theory_always_produces_the_same_table() {
        let a = ColorTable::build(&theory(TWO_RULES));
        let b = ColorTable::build(&theory(TWO_RULES));
        assert_eq!(a, b);
    }

    /// A theory whose declaration order is the REVERSE of another's,
    /// but with the same rule/action NAMES, still produces the same
    /// table — pinning that declaration order truly never leaks in.
    #[test]
    fn declaration_order_does_not_affect_the_table() {
        const REVERSED: &str = "theory T begin\n\
            rule Apple:\n  [] --[ Alpha() ]-> []\n\
            rule Zebra:\n  [] --[ Beta() ]-> []\n\
            end";
        let forward = ColorTable::build(&theory(TWO_RULES));
        let reversed = ColorTable::build(&theory(REVERSED));
        assert_eq!(forward, reversed);
    }

    /// The bug this module's real-data test caught (see the session
    /// notes / `ColorTable::rule_color`'s own doc comment): a theory's
    /// own PROTOCOL rule literally named `"Send"` must NOT be colored
    /// like the built-in `ISend` intruder rule, even though
    /// `rule::rule_name_string` would render both identically. They are
    /// structurally distinct (`RuleInfo::Proto` vs `Intr`), so they must
    /// get DIFFERENT colors.
    #[test]
    fn theory_protocol_rule_named_like_a_builtin_does_not_collide() {
        const COLLIDING: &str = "theory T begin\n\
            rule Send:\n  [] --> []\n\
            end";
        let table = ColorTable::build(&theory(COLLIDING));
        let builtin_send = table.rule_color(&intr_rule_instance(IntrRuleACInfo::ISend));
        let theory_send = table.rule_color(&proto_rule_instance("Send"));
        assert_ne!(
            builtin_send, theory_send,
            "a user protocol rule named Send must not collide with the built-in ISend rule"
        );
    }

    /// The mirror-image, UNAVOIDABLE limitation on the action side (see
    /// `BUILTIN_ACTION_NAMES`'s own doc comment): `GFact`/`LNFact` carry
    /// only a bare name string, with no structural tag distinguishing
    /// "built-in" from "user-declared" the way `RuleInfo` does for
    /// rules — so a theory action fact literally named `"Fr"` DOES
    /// reuse the built-in `Fr` action's color, by construction.
    #[test]
    fn theory_action_named_like_a_builtin_does_collide() {
        const COLLIDING: &str = "theory T begin\n\
            rule R:\n  [] --[ Fr(~n) ]-> []\n\
            end";
        let table = ColorTable::build(&theory(COLLIDING));
        assert_eq!(
            table.action_color("Fr"),
            ColorTable::build(&theory(EMPTY)).action_color("Fr")
        );
    }
}


