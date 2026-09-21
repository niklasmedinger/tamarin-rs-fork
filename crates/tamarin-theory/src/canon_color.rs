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
//! 1. This crate's own structural relation-vertex kinds. Two parts:
//!    - `VertexKind::Dummy`/`LessRelation`/`AtTimepointRelation`/
//!      `LastAtomRelation` — four FIXED colors (`STRUCTURAL_FIXED_COUNT`;
//!      bumped from three when `LastAtomRelation` was added), identical
//!      across every theory.
//!    - `VertexKind::EdgeRelation(ConcIdx, PremIdx)` — one color per
//!      distinct `(ConcIdx, PremIdx)` PAIR that can occur for this
//!      theory, packed right after the four fixed colors above (still
//!      block 1: these are structural port positions, not theory
//!      content). The pair's bounds are discovered by
//!      [`ColorTable::build`] scanning every fixed built-in rule's own
//!      premise/conclusion counts *and* this theory's own declared
//!      rules' — so, unlike the four fixed colors, this sub-block's
//!      SIZE (though not its structure) is theory-dependent: a theory
//!      whose own rule has more premises than any built-in rule widens
//!      it. See [`ColorTable::edge_relation_color`] for why this
//!      information matters (a payload-free `EdgeRelation` marker would
//!      make an edge landing in premise slot 0 graph-indistinguishable
//!      from one landing in slot 1).
//! 2. Tamarin's built-in rule names and built-in action names — also
//!    fixed, identical across every theory (see
//!    [`BUILTIN_RULE_NAMES`]/[`BUILTIN_ACTION_NAMES`] for exactly which
//!    names and how they were derived).
//! 3. One specific theory's own protocol rule names and protocol action
//!    names — collected from the theory, each sorted independently
//!    (`BTreeSet`, so the source file's declaration order never leaks
//!    in), and colored in that order. The only block whose CONTENT (as
//!    opposed to just block 1's size) varies by theory — the reason
//!    [`ColorTable::build`] needs a `&Theory` at all, rather than blocks
//!    1–2 being the whole story.
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
use crate::intruder_rules::{nat_intruder_rules, special_intruder_rules};
use crate::rule::{ConcIdx, IntrRuleACInfo, PremIdx, ProtoRuleName, RuleACInst, RuleInfo};
use crate::theory::Theory;

/// A vertex color: an arbitrary but STABLE non-negative integer. Two
/// vertices sharing a color are indistinguishable to the graph
/// canonizer's initial partition; see the module docs for the
/// soundness bar this has to clear (constant on $\alphaeqac$-classes).
pub type Color = u32;

// =============================================================================
// Block 1 — structural relation-vertex kinds
// =============================================================================

/// The four colors that don't depend on premise/conclusion port
/// positions — fixed, identical across every theory.
const STRUCTURAL_DUMMY: Color = 0;
const STRUCTURAL_LESS_RELATION: Color = 1;
const STRUCTURAL_AT_TIMEPOINT_RELATION: Color = 2;
const STRUCTURAL_LAST_ATOM_RELATION: Color = 3;
/// One past the last FIXED structural color — where the
/// `EdgeRelation(ConcIdx, PremIdx)` sub-block starts (still block 1; see
/// the module docs). That sub-block's own size is
/// `max_conc_count * max_prem_count`, computed by [`ColorTable::build`]
/// and stored per-table — see [`ColorTable::block1_size`].
const STRUCTURAL_FIXED_COUNT: Color = 4;

/// `STRUCTURAL_FIXED_COUNT` plus the `EdgeRelation` sub-block's size —
/// i.e. where block 2 (Tamarin built-ins) starts. A free function (not a
/// method) so [`ColorTable::build`] can call it before `Self` exists yet.
fn block1_size(max_conc_count: usize, max_prem_count: usize) -> Color {
    STRUCTURAL_FIXED_COUNT + (max_conc_count * max_prem_count) as Color
}

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
    /// One past the largest `ConcIdx`/`PremIdx` seen across every fixed
    /// built-in rule and this theory's own declared rules — the bounds
    /// [`ColorTable::edge_relation_color`]'s packing needs. See
    /// [`ColorTable::build`] for how these are discovered.
    max_conc_count: usize,
    max_prem_count: usize,
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
        // The `EdgeRelation` sub-block's bounds: the largest premise/
        // conclusion COUNT (not index — one past the max index) among
        // every fixed built-in rule and this theory's own declared
        // rules. Seeded from the fixed built-ins (`special_intruder_rules`
        // called with `diff: true` so `IEquality`'s 2 premises are
        // counted, matching `BUILTIN_RULE_NAMES` unconditionally
        // including "Equality" regardless of this theory's own diff
        // setting) so the sub-block never shrinks below what the
        // built-ins alone need, then widened by this theory's own rules.
        let mut max_conc_count: usize = 0;
        let mut max_prem_count: usize = 0;
        for r in special_intruder_rules(true).iter().chain(nat_intruder_rules().iter()) {
            max_conc_count = max_conc_count.max(r.conclusions.len());
            max_prem_count = max_prem_count.max(r.premises.len());
        }
        for r in theory.rules() {
            if let ProtoRuleName::Stand(s) = r.rule.info.name {
                theory_rule_names.insert(s);
            }
            for fa in &r.rule.actions {
                theory_action_names.insert(fact_tag_name(&fa.tag));
            }
            max_conc_count = max_conc_count.max(r.rule.conclusions.len());
            max_prem_count = max_prem_count.max(r.rule.premises.len());
        }
        // Actions have no structural tag distinguishing "built-in" from
        // "user-declared" (`GFact`/`LNFact` carry only a bare name
        // string — see `BUILTIN_ACTION_NAMES`'s own doc comment), so a
        // theory action fact literally named e.g. `"K"` is
        // INDISTINGUISHABLE from the built-in `K` logging action and
        // deliberately reuses its color rather than getting a second,
        // conflicting one for the same string.
        theory_action_names.retain(|n| !builtin_actions.contains(n.as_str()));

        // Block 3 starts AFTER block 1 (structural, now including the
        // EdgeRelation sub-block) and both builtin sub-blocks (2a rules,
        // 2b actions) — matching `action_color`'s own on-the-fly builtin
        // computation, which places 2b right after 2a. All three must
        // agree on this base or two different sub-blocks silently claim
        // the same colors (caught by this module's own
        // `builtin_and_theory_blocks_never_overlap` test — a real bug
        // the first version of this function had, before block 3 was
        // fixed to start here rather than right after block 2a).
        let mut next: Color = block1_size(max_conc_count, max_prem_count)
            + BUILTIN_RULE_NAMES.len() as Color
            + BUILTIN_ACTION_NAMES.len() as Color;
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
            max_conc_count,
            max_prem_count,
        }
    }

    /// `STRUCTURAL_FIXED_COUNT` plus this table's own `EdgeRelation`
    /// sub-block size — where block 2 (Tamarin built-ins) starts for
    /// this specific table. See the module docs' block-1 description.
    fn block1_size(&self) -> Color {
        block1_size(self.max_conc_count, self.max_prem_count)
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
                ProtoRuleName::Fresh => self.builtin_rule_color("FreshRule"),
                ProtoRuleName::Stand(s) => self.theory_rule_colors.get(s).copied().unwrap_or_else(|| {
                    panic!(
                        "ColorTable::rule_color: protocol rule {s:?} is not a name \
                         this table was built from (table/system theory mismatch?)"
                    )
                }),
            },
            RuleInfo::Intr(info) => match builtin_intr_rule_name(info) {
                Some(name) => self.builtin_rule_color(name),
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
            return self.block1_size() + BUILTIN_RULE_NAMES.len() as Color + pos as Color;
        }
        self.theory_action_colors.get(name).copied().unwrap_or_else(|| {
            panic!(
                "ColorTable::action_color: {name:?} is neither a built-in action \
                 name nor a protocol action name this table was built from (see \
                 this module's completeness caveat)"
            )
        })
    }

    /// The color for one `EdgeRelation(conc, prem)` vertex — part of
    /// block 1 (see the module docs). Every distinct `(ConcIdx, PremIdx)`
    /// combination this table was built to cover gets its own color,
    /// packed as `conc * max_prem_count + prem` right after the four
    /// fixed structural colors. This is exactly the information a bare,
    /// payload-free `EdgeRelation` marker discarded: without it, an edge
    /// landing in premise slot 0 of a rule instance was
    /// graph-indistinguishable from one landing in slot 1, so the graph
    /// canonizer could not tell apart two systems that actually differ
    /// in which premise consumes which conclusion.
    ///
    /// Panics if `conc`/`prem` is out of the range this table was built
    /// from (a table/system theory mismatch, not expected in normal use
    /// — mirrors [`ColorTable::rule_color`]'s own panic-on-mismatch
    /// style).
    pub fn edge_relation_color(&self, conc: ConcIdx, prem: PremIdx) -> Color {
        assert!(
            conc.0 < self.max_conc_count && prem.0 < self.max_prem_count,
            "ColorTable::edge_relation_color: {conc:?}/{prem:?} is out of range \
             (max_conc_count={}, max_prem_count={}) -- table/system theory mismatch?",
            self.max_conc_count,
            self.max_prem_count
        );
        STRUCTURAL_FIXED_COUNT + (conc.0 * self.max_prem_count + prem.0) as Color
    }

    /// The color for any [`VertexKind`] — the single entry point a
    /// caller building the full coloring `c: V -> N` for a `GraphPart`
    /// actually wants (`part.vertices.iter().map(|v| table.vertex_color(v))`).
    pub fn vertex_color(&self, v: &VertexKind) -> Color {
        match v {
            VertexKind::Dummy(_) => STRUCTURAL_DUMMY,
            VertexKind::EdgeRelation(conc, prem) => self.edge_relation_color(*conc, *prem),
            VertexKind::LessRelation => STRUCTURAL_LESS_RELATION,
            VertexKind::AtTimepointRelation => STRUCTURAL_AT_TIMEPOINT_RELATION,
            VertexKind::LastAtomRelation => STRUCTURAL_LAST_ATOM_RELATION,
            VertexKind::RuleInstance(_, ru) => self.rule_color(ru),
            VertexKind::Action(_, fact) => self.action_color(&fact.name),
        }
    }

    /// The color of a name known to be in [`BUILTIN_RULE_NAMES`] —
    /// panics (an internal-consistency bug, not a user-facing error) if
    /// it somehow isn't, since every caller has already established
    /// membership. A method (not a free function) because the base
    /// (block 2's start) depends on this table's own `EdgeRelation`
    /// sub-block size.
    fn builtin_rule_color(&self, name: &str) -> Color {
        let pos = BUILTIN_RULE_NAMES
            .iter()
            .position(|n| *n == name)
            .unwrap_or_else(|| panic!("builtin_rule_color: {name:?} not in BUILTIN_RULE_NAMES"));
        self.block1_size() + pos as Color
    }
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
            table.vertex_color(&VertexKind::LessRelation),
            table.vertex_color(&VertexKind::AtTimepointRelation),
        ];
        let mut sorted = colors.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 3, "structural colors must be pairwise distinct");
        // Fixed means fixed: same theory-independent values every time.
        assert_eq!(colors, [0, 1, 2]);
    }

    /// The concrete ask of this session's follow-up: an `EdgeRelation`
    /// vertex's color must depend on ITS OWN `(ConcIdx, PremIdx)` pair,
    /// not just be a single fixed marker color -- otherwise an edge
    /// landing in premise slot 0 is graph-indistinguishable from one
    /// landing in slot 1.
    #[test]
    fn edge_relation_colors_differ_by_port_index_and_are_deterministic() {
        // No fixed built-in rule has 2 conclusions (max is 1 -- see
        // `special_intruder_rules`), so `ConcIdx(1)` needs a theory rule
        // that actually has a 2nd conclusion to be in range at all.
        const TWO_CONCLUSIONS: &str = "theory T begin\n\
            rule R:\n  [] --[ Beta() ]-> [ A(), B() ]\n\
            end";
        let table = ColorTable::build(&theory(TWO_CONCLUSIONS));
        let c00 = table.edge_relation_color(ConcIdx(0), PremIdx(0));
        let c01 = table.edge_relation_color(ConcIdx(0), PremIdx(1));
        let c10 = table.edge_relation_color(ConcIdx(1), PremIdx(0));
        assert_ne!(c00, c01, "different PremIdx must get different colors");
        assert_ne!(c00, c10, "different ConcIdx must get different colors");
        assert_ne!(c01, c10);
        // Deterministic: calling again with the same indices reproduces
        // the same color.
        assert_eq!(c00, table.edge_relation_color(ConcIdx(0), PremIdx(0)));
        // Placed right after the 3 fixed structural colors (block 1).
        assert!(
            [c00, c01, c10].iter().all(|c| *c >= STRUCTURAL_FIXED_COUNT),
            "EdgeRelation colors must land after the 3 fixed structural colors"
        );
    }

    /// `ColorTable::build` seeds the `EdgeRelation` range from the fixed
    /// built-in rules' own premise counts (`IEquality` has 2), so even an
    /// EMPTY theory must already support `PremIdx(0)` and `PremIdx(1)`
    /// without panicking.
    #[test]
    fn empty_theory_already_supports_the_builtins_own_premise_range() {
        let table = ColorTable::build(&theory(EMPTY));
        table.edge_relation_color(ConcIdx(0), PremIdx(0));
        table.edge_relation_color(ConcIdx(0), PremIdx(1)); // IEquality's 2nd premise
    }

    /// A theory whose own rule declares MORE premises than any built-in
    /// rule widens the `EdgeRelation` range accordingly -- discovery is
    /// genuinely theory-dependent, not just baked-in from the builtins.
    #[test]
    fn theory_rule_with_more_premises_than_any_builtin_widens_the_edge_relation_range() {
        const THREE_PREMISES: &str = "theory T begin\n\
            rule R:\n  [ A(), B(), C() ] --> []\n\
            end";
        let table = ColorTable::build(&theory(THREE_PREMISES));
        // PremIdx(2) would be out of range for a table built from EMPTY
        // (see the panic test below) but must be in range here.
        table.edge_relation_color(ConcIdx(0), PremIdx(2));
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn edge_relation_color_panics_on_an_out_of_range_index() {
        let table = ColorTable::build(&theory(EMPTY));
        // No built-in or EMPTY-theory rule has 3 conclusions.
        table.edge_relation_color(ConcIdx(3), PremIdx(0));
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
            table.vertex_color(&VertexKind::LessRelation),
            table.vertex_color(&VertexKind::AtTimepointRelation),
            table.edge_relation_color(ConcIdx(0), PremIdx(0)),
            table.edge_relation_color(ConcIdx(0), PremIdx(1)),
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



