// Currently GPL 3.0 until granted permission by the upstream authors
// of the tamarin-prover sources this file cites; list them with:
//   scripts/gen_license_headers.py --authors <this file>

//! (No HS analog.) Builds the vertex-coloring `c: V -> N` `TODO.md`'s
//! "Canonizing the graph" section specifies for a [`crate::canon_graph::GraphPart`]
//! — Stage B of "Canonizing the constraint system", following Stage A
//! ([`crate::canon_graph::extract_graph_part`]).
//!
//! Two layers. The BASE color ([`ColorTable::vertex_color`]) comes from
//! a FOUR reserved-color-block scheme (originally `TODO.md`'s three-block
//! sketch; a fourth was added 2026-09-22 once intruder
//! construction/destruction rules needed covering too — see block 2's own
//! write-up). The final per-graph coloring ([`ColorTable::shape_colors`])
//! refines it by content shape — see "Shape refinement" below. The base
//! blocks:
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
//!      [`ColorTable::build`] scanning every intruder rule in the
//!      `IntrRuleCache` it's given *and* this theory's own declared
//!      rules' — so, unlike the four fixed colors, this sub-block's
//!      SIZE (though not its structure) is theory-dependent: a theory
//!      whose own rule (or synthesized intruder rule) has more premises
//!      than any other widens it. See [`ColorTable::edge_relation_color`]
//!      for why this information matters (a payload-free `EdgeRelation`
//!      marker would make an edge landing in premise slot 0
//!      graph-indistinguishable from one landing in slot 1).
//! 2. The `ProtoRuleName::Fresh` protocol rule (`FreshRule`) — ONE more
//!    fixed color, right after block 1. Not an intruder rule (so it's
//!    never in an `IntrRuleCache`) and not a user-declared protocol rule
//!    either (so it's never in `theory.rules()`/`protocol_rules`): it's
//!    a Tamarin-wide constant, present in every theory, hence its own
//!    dedicated fixed slot rather than living in block 3 or 4.
//! 3. EVERY intruder rule in the `IntrRuleCache` [`ColorTable::build`] is
//!    given — the fixed ones (`Coerce`/`IRecv`/`ISend`/`PubConstr`/
//!    `NatConstr`/`FreshConstr`/`IEquality`) AND the theory-dependent
//!    construction/destruction rules synthesized from this theory's own
//!    `functions:`/`builtins:` declarations (`IntrRuleACInfo::ConstrRule`/
//!    `DestrRule`) — colored UNIFORMLY via one sorted set
//!    (`IntrRuleACInfo` derives `Ord` directly, so no separate key type is
//!    needed). Deliberately NOT split into "fixed, hardcoded" vs.
//!    "theory-dependent, uncovered" the way an earlier revision of this
//!    table was: the `IntrRuleCache` a caller supplies IS the exhaustive,
//!    already-computed set of every intruder rule that theory's own
//!    proof search can ever instantiate (see [`ColorTable::build`]'s own
//!    doc comment for where that cache comes from), so there is nothing
//!    left to hardcode a parallel list for.
//! 4. One specific theory's own protocol rule names and protocol action
//!    names — collected from the `&[OpenProtoRule]` the caller supplies,
//!    each sorted independently (`BTreeSet`, so the source file's
//!    declaration order never leaks in), and colored in that order.
//!
//! Within each of blocks 3 and 4, rule names are colored before action
//! names (two separate sorted sets, not one merged alphabetical list) —
//! so a rule and an action that happen to share a literal name (e.g. a
//! contrived theory with both a rule and an action fact named `"Foo"`)
//! still get different colors, for free, since rule-name and
//! action-name colors always land in disjoint sub-ranges.
//!
//! **Shape refinement** (`TODO.md`'s skeleton-strengthening sketch: erase
//! variables to sorts, `CAN_AC` the result). A base color only knows a
//! vertex's rule name or fact tag, so e.g. sibling `!KU(t_i)` goals from a
//! tuple decomposition are all interchangeable to bliss, and every such
//! swap is a group element Stage F has to minimize over. The refined key
//! of a vertex is `(base color, shape)`, where the shape of a
//! `RuleInstance`/`Action` is its content term (`canon::rule_to_term`/
//! `canon::fact_to_term`) with every variable replaced by a placeholder
//! for its sort, rebuilt through the AC/C smart constructors
//! ([`erase_variables`]). Names are kept: they are never renamed (see
//! `tamarin_term::alpha_eq_ac`'s module doc), so `!KU(<'1', x>)` and
//! `!KU(<'2', x>)` get different shapes. This is constant on
//! $\alphaeqac$-classes: a sort-respecting variable renaming never changes
//! the erased term, and erasure maps AC-equal terms to AC-equal terms,
//! whose normal forms coincide. No canonization is needed.
//!
//! The keys are then numbered by RANK among the distinct keys of the SAME
//! graph part, never by the order vertices are encountered: vertex order
//! follows `NodeId` numbering, which differs between equivalent systems,
//! and bliss compares color VALUES, not just the partition they induce.
//! An isomorphism between equivalent systems maps every vertex to one with
//! the same key, so both have the same key set and hence the same ranks.
//!
//! **Soundness, not completeness, is the bar** (`TODO.md`'s own words:
//! "the soundness condition on a color function is only that it be
//! constant on alphaeqac-classes... finer is better but never
//! necessary"): [`ColorTable::rule_color`]/[`action_color`] still PANIC
//! on a rule/action name outside every block (documented per-function)
//! rather than silently guessing — this now only happens on a genuine
//! caller/table mismatch (a `System` containing a rule instance this
//! table's `IntrRuleCache`/`protocol_rules` never covered), not on a
//! structurally-expected-but-unimplemented case the way the old
//! `ConstrRule`/`DestrRule` gap was.
//!
//! **Why this table takes `&[OpenProtoRule]` + `&IntrRuleCache`, not
//! `&Theory`**: a `ColorTable` is unique per theory, and the CALLER —
//! whoever elaborated the theory / built the `ProofContext` a real proof
//! search runs under, or a test constructing both directly — already has
//! both pieces on hand (a `ProofContext` carries its own `rules`/
//! `intruder_rules` already; see [`crate::constraint::solver::context::ProofContext`]'s
//! own `color_table` field, built once alongside them). Neither
//! `canon_graph::extract_graph_part` nor `canon::canonicalize_constraint_system`
//! need a `&Theory` at all once the caller supplies the table directly —
//! see their own doc comments.
//!
//! NOT [`crate::constraint::system::graph::color`] — see
//! `canon_graph.rs`'s module docs for why that "color" (a cosmetic HSV
//! fill palette for DOT rendering) is a different concept from this one
//! (graph-theoretic vertex coloring for the canonizer) despite the
//! shared word.

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use tamarin_term::lterm::{HasFrees, LNTerm, LVar};

use crate::canon::{fact_to_term, rule_to_term};
use crate::canon_graph::VertexKind;
use crate::constraint::solver::context::IntrRuleCache;
use crate::fact::fact_tag_name;
use crate::rule::{ConcIdx, IntrRuleACInfo, PremIdx, ProtoRuleName, RuleACInst, RuleInfo};
use crate::theory::OpenProtoRule;

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
// Block 2 — the fixed `FreshRule` slot
// =============================================================================

/// `block1_size(...)` plus one — the single dedicated color for
/// `ProtoRuleName::Fresh` (`FreshRule`; see the module docs' block 2).
/// `FreshRule` is neither an intruder rule (so it never appears in a
/// caller-supplied [`IntrRuleCache`]) nor a user-declared protocol rule
/// (`fresh` is a reserved name no theory can redeclare — `rule::
/// RESERVED_RULE_NAMES`), so nothing else in this table would ever cover
/// it without this dedicated slot. A free function (not a method) so
/// [`ColorTable::build`] can call it before `Self` exists yet, mirroring
/// [`block1_size`].
fn block2_size(max_conc_count: usize, max_prem_count: usize) -> Color {
    block1_size(max_conc_count, max_prem_count) + 1
}

// =============================================================================
// Block 3a — every intruder rule in the caller-supplied `IntrRuleCache`
// =============================================================================

// =============================================================================
// Block 3b — fixed built-in action names
// =============================================================================

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
/// assignment order (right after block 3a's dynamic `intr_rule_colors`,
/// see [`ColorTable::action_color`]), so declaration order here is
/// load-bearing.
pub const BUILTIN_ACTION_NAMES: [&str; 8] = ["Ded", "Fr", "In", "K", "KD", "KU", "Out", "Term"];

// =============================================================================
// The table
// =============================================================================

/// The full vertex-coloring table for one theory. See the module docs
/// for the four-block scheme.
///
/// Blocks 3a (`intr_rule_colors`) and 4 (`theory_rule_colors`/
/// `theory_action_colors`) are STORED — blocks 1, 2 and 3b are fixed for
/// every theory, so their colors are computed directly from
/// [`BUILTIN_ACTION_NAMES`]'s positions (plus `max_conc_count`/
/// `max_prem_count` for block 1's `EdgeRelation` sub-block) rather than
/// carried in an instance.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ColorTable {
    /// Block 3a: every intruder rule the `IntrRuleCache` this table was
    /// built from contains — fixed (`Coerce`/`IRecv`/`ISend`/…) AND
    /// theory-specific (`ConstrRule`/`DestrRule`) alike, colored
    /// uniformly. `IntrRuleACInfo` derives `Ord`, so it is usable as a
    /// `BTreeMap` key directly — no separate name/key type is needed the
    /// way rule/action NAMES need one for blocks 3b/4.
    intr_rule_colors: BTreeMap<IntrRuleACInfo, Color>,
    /// Keyed by the rule's INTERNED name (`ProtoRuleName::Stand`'s
    /// payload is already `&'static str` — see `rule.rs`'s own doc
    /// comment on that field), not a rendered/allocated `String`.
    theory_rule_colors: BTreeMap<&'static str, Color>,
    theory_action_colors: BTreeMap<String, Color>,
    /// One past the largest `ConcIdx`/`PremIdx` seen across every rule in
    /// the `IntrRuleCache` this table was built from and this theory's
    /// own declared rules — the bounds [`ColorTable::edge_relation_color`]'s
    /// packing needs. See [`ColorTable::build`] for how these are
    /// discovered.
    max_conc_count: usize,
    max_prem_count: usize,
}

impl ColorTable {
    /// Builds the color table this specific theory's `System`s need:
    /// every rule instance the AC/canonicalization machinery ever colors
    /// is either one of `protocol_rules`' own declared rules, the fixed
    /// `ProtoRuleName::Fresh`, or an intruder rule `intruder_rules`
    /// contains — so those two arguments alone are enough (no `&Theory`
    /// needed; see the module docs' "why this table takes..." section).
    ///
    /// Deterministic given `protocol_rules`' and `intruder_rules`' sets
    /// of rule/action NAMES (resp. `IntrRuleACInfo` values) alone: the
    /// same inputs always produce the same table, regardless of what
    /// order the rules happen to arrive in (collected into `BTreeSet`s/
    /// a `BTreeMap` before any color is assigned).
    pub fn build(protocol_rules: &[OpenProtoRule], intruder_rules: &IntrRuleCache) -> Self {
        let builtin_actions: BTreeSet<&str> = BUILTIN_ACTION_NAMES.into_iter().collect();

        let mut theory_rule_names: BTreeSet<&'static str> = BTreeSet::new();
        let mut theory_action_names: BTreeSet<String> = BTreeSet::new();
        let mut max_conc_count: usize = 0;
        let mut max_prem_count: usize = 0;
        // Block 3a's contents AND the `EdgeRelation` sub-block's lower
        // bound both come from the same scan: every rule this specific
        // theory's `IntrRuleCache` actually contains (fixed AND
        // Constr/Destr alike — deliberately not filtered, see the module
        // docs' block-3 write-up), deduped via the map's own keys.
        let mut intr_rule_colors: BTreeMap<IntrRuleACInfo, Color> = BTreeMap::new();
        for r in intruder_rules.iter() {
            intr_rule_colors.entry(r.info.clone()).or_insert(0);
            max_conc_count = max_conc_count.max(r.conclusions.len());
            max_prem_count = max_prem_count.max(r.premises.len());
        }
        // Rule names: every `protocol_rules` entry is a `ProtoRuleE`
        // (`Rule<ProtoRuleEInfo>`) — i.e. genuinely a user-declared
        // protocol rule (the open-theory level has no `RuleInfo`/`Intr`
        // case to collide with at all: that wrapper only appears once
        // rules are AC-instantiated for proof search). So there is
        // nothing to filter here, unlike actions below.
        for r in protocol_rules {
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

        // Block 3a starts right after block 2 (the single `FreshRule`
        // slot). Assigned in a second pass (the first pass above only
        // discovered the SET of `IntrRuleACInfo` values, via the map's
        // keys) so every entry gets a color from the same monotonic
        // counter block 3b/4 continue from — matching `action_color`'s
        // own on-the-fly block-3b computation, which places it right
        // after block 3a. All must agree on this base or two different
        // sub-blocks silently claim the same colors (caught by this
        // module's own `builtin_and_theory_blocks_never_overlap` test —
        // a real bug an earlier version of this function had).
        let mut next: Color = block2_size(max_conc_count, max_prem_count);
        for color in intr_rule_colors.values_mut() {
            *color = next;
            next += 1;
        }
        next += BUILTIN_ACTION_NAMES.len() as Color;

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
            intr_rule_colors,
            theory_rule_colors,
            theory_action_colors,
            max_conc_count,
            max_prem_count,
        }
    }

    /// `STRUCTURAL_FIXED_COUNT` plus this table's own `EdgeRelation`
    /// sub-block size — where block 2 (the `FreshRule` slot) starts for
    /// this specific table. See the module docs' block-1 description.
    fn block1_size(&self) -> Color {
        block1_size(self.max_conc_count, self.max_prem_count)
    }

    /// One past [`Self::block1_size`] — where block 3a
    /// (`intr_rule_colors`) starts for this specific table.
    fn block2_size(&self) -> Color {
        self.block1_size() + 1
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
    /// Panics if `ru` is an intruder rule this table's `IntrRuleCache`
    /// didn't contain, or a `Stand` name this table's `protocol_rules`
    /// wasn't built from (both are caller/table mismatch bugs, not
    /// expected in normal use — the whole point of block 3a covering
    /// every `IntrRuleCache` entry uniformly is that a `ConstrRule`/
    /// `DestrRule` is no longer a special case here, unlike the earlier
    /// revision of this table).
    pub fn rule_color(&self, ru: &RuleACInst) -> Color {
        match &ru.info {
            RuleInfo::Proto(p) => match p.name {
                ProtoRuleName::Fresh => self.block1_size(),
                ProtoRuleName::Stand(s) => self.theory_rule_colors.get(s).copied().unwrap_or_else(|| {
                    panic!(
                        "ColorTable::rule_color: protocol rule {s:?} is not a name \
                         this table was built from (table/system theory mismatch?)"
                    )
                }),
            },
            RuleInfo::Intr(info) => self.intr_rule_colors.get(info).copied().unwrap_or_else(|| {
                panic!(
                    "ColorTable::rule_color: {info:?} is not covered by this table \
                     -- this table's IntrRuleCache did not contain it (table/system \
                     theory mismatch?)"
                )
            }),
        }
    }

    /// The color for an action-fact name (block 3b or 4). Panics if
    /// `name` is neither a built-in nor a name this table was built
    /// from — see the module docs' completeness caveat.
    pub fn action_color(&self, name: &str) -> Color {
        if let Some(pos) = BUILTIN_ACTION_NAMES.iter().position(|n| *n == name) {
            return self.block2_size() + self.intr_rule_colors.len() as Color + pos as Color;
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
            VertexKind::Action(_, fact) => self.action_color(&fact_tag_name(&fact.tag)),
        }
    }

    /// The base color of every vertex, in order -- [`Self::vertex_color`]
    /// without the shape refinement. Still a valid coloring for bliss (it
    /// is constant on $\alphaeqac$-classes), just a coarser one.
    pub fn base_colors(&self, vertices: &[VertexKind]) -> Vec<Color> {
        vertices.iter().map(|v| self.vertex_color(v)).collect()
    }

    /// The coloring handed to bliss: each vertex's `(base color, shape)`
    /// key, numbered by its rank among the distinct keys of `vertices`
    /// (see the module docs' "Shape refinement" for why rank, not
    /// encounter order). Ranking sorts by base color first, so a lower
    /// base color always gets a lower final color.
    pub fn shape_colors(&self, vertices: &[VertexKind]) -> Vec<Color> {
        let keys: Vec<(Color, Option<LNTerm>)> = vertices
            .iter()
            .map(|v| (self.vertex_color(v), shape_term(v)))
            .collect();
        let ranks: BTreeMap<&(Color, Option<LNTerm>), Color> = keys
            .iter()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .zip(0..)
            .collect();
        keys.iter().map(|k| ranks[k]).collect()
    }
}

/// The shape of a content vertex's term (see the module docs' "Shape
/// refinement"); `None` for content-free structural vertices.
fn shape_term(v: &VertexKind) -> Option<LNTerm> {
    match v {
        VertexKind::RuleInstance(_, ru) => Some(erase_variables(rule_to_term(ru))),
        VertexKind::Action(_, fact) => Some(erase_variables(fact_to_term(fact))),
        _ => None,
    }
}

/// `t` with every variable replaced by one placeholder per sort, names
/// kept. The ARBITRARY (non-monotone) [`HasFrees::map_free`] rebuilds
/// through the smart constructors, so AC arguments are re-flattened and
/// re-sorted, and C arguments re-sorted, around the placeholders -- i.e.
/// the `CAN_AC` normal form of the erased term.
fn erase_variables(t: LNTerm) -> LNTerm {
    t.map_free(&mut |v| LVar::new("_", v.sort, 0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tamarin_term::term::Term;
    use tamarin_term::vterm::Lit;
    use crate::constraint::solver::context::ProofContext;
    use crate::rule::{ProtoRuleACInstInfo, Rule, RuleAttributes};
    use crate::theory::Theory;
    use tamarin_parser::parser::parse_theory;
    use tamarin_term::maude_proc::MaudeHandle;

    fn theory(src: &str) -> Theory {
        let parsed = parse_theory(src, &[]).unwrap_or_else(|e| panic!("parse: {e}"));
        crate::elaborate::elaborate(&parsed).unwrap_or_else(|e| panic!("elaborate: {e:?}"))
    }

    /// Elaborates `src`, boots a maude process on ITS OWN signature
    /// (booting on a different theory's signature would make
    /// [`ProofContext::assemble_intruder_rules`]'s subterm-constructor-rule
    /// generation see the wrong function symbols), and builds a
    /// [`ColorTable`] exactly the way `ProofContext::new_impl` does — from
    /// the elaborated theory's own protocol rules and a freshly-assembled
    /// [`IntrRuleCache`]. This is the caller-supplies-everything design
    /// this module was rewritten for (see the module docs): the table no
    /// longer has a `&Theory`-only construction path to fall back on, so
    /// every test in this module needs a real (if trivial) maude process.
    ///
    /// `None` when no maude is resolvable (`TAM_ALLOW_NO_MAUDE=1` and
    /// nothing found) — callers do `let Some(table) = table_for(...) else
    /// { return };`, the same documented skip every other maude-backed
    /// test module in this crate uses (see
    /// [`crate::test_maude::maude_path`]'s own doc comment).
    fn table_for(src: &str) -> Option<ColorTable> {
        let path = crate::test_maude::maude_path()?;
        let elaborated = theory(src);
        let maude = MaudeHandle::start(&path, elaborated.signature.maude_sig.clone())
            .unwrap_or_else(|e| panic!("maude at {path} failed to start: {e:?}"));
        let protocol_rules: Vec<OpenProtoRule> = elaborated.rules().cloned().collect();
        let intruder_rules = IntrRuleCache::from(ProofContext::assemble_intruder_rules(
            &elaborated.signature.maude_sig,
            &maude,
        ));
        Some(ColorTable::build(&protocol_rules, &intruder_rules))
    }

    const EMPTY: &str = "theory T begin\nend";

    const TWO_RULES: &str = "theory T begin\n\
        rule Zebra:\n  [ A(), B() ] --[ Beta() ]-> []\n\
        rule Apple:\n  [ A(), B() ] --[ Alpha() ]-> []\n\
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
        let Some(table) = table_for(EMPTY) else { return };
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
        // No fixed intruder rule in a trace-mode `IntrRuleCache` has more
        // than 1 premise or 1 conclusion (see `special_intruder_rules`'s
        // own doc comment: `IEquality`, the one 2-premise special rule,
        // is diff-mode-only and never in a real trace-mode cache), so
        // `ConcIdx(1)`/`PremIdx(1)` both need a theory rule that actually
        // has a 2nd premise/conclusion to be in range at all.
        const TWO_CONCLUSIONS: &str = "theory T begin\n\
            rule R:\n  [ A(), B() ] --[ Beta() ]-> [ A(), B() ]\n\
            end";
        let Some(table) = table_for(TWO_CONCLUSIONS) else { return };
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

    /// An EMPTY theory's real `IntrRuleCache` still covers `PremIdx(0)`/
    /// `ConcIdx(0)` -- the `EdgeRelation` sub-block's bounds now come
    /// entirely from what the caller's `IntrRuleCache`/`protocol_rules`
    /// actually contain (see the module docs' "why this table takes..."
    /// section), replacing the OLD hardcoded `special_intruder_rules(true)`
    /// seeding this table no longer does. Deliberately does NOT assert an
    /// exact upper bound: the base message algebra's own pairing
    /// constructor (`subterm_constructor_rules`, always present -- pairing
    /// is built into the term algebra independent of any theory's own
    /// `functions:`/`builtins:` declarations) already gives even an EMPTY
    /// theory more than 1 premise, so a test pinning that exact number
    /// would just be re-deriving `subterm_constructor_rules`' internals
    /// rather than testing this table.
    #[test]
    fn empty_theory_edge_relation_range_covers_only_what_the_real_cache_needs() {
        let Some(table) = table_for(EMPTY) else { return };
        table.edge_relation_color(ConcIdx(0), PremIdx(0));
    }

    /// A theory whose own rule declares MORE premises than any built-in
    /// rule widens the `EdgeRelation` range accordingly -- discovery is
    /// genuinely theory-dependent, not just baked-in from the builtins.
    #[test]
    fn theory_rule_with_more_premises_than_any_builtin_widens_the_edge_relation_range() {
        const THREE_PREMISES: &str = "theory T begin\n\
            rule R:\n  [ A(), B(), C() ] --> []\n\
            end";
        let Some(table) = table_for(THREE_PREMISES) else { return };
        // PremIdx(2) would be out of range for a table built from EMPTY
        // (see the panic test above) but must be in range here.
        table.edge_relation_color(ConcIdx(0), PremIdx(2));
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn edge_relation_color_panics_on_an_out_of_range_index() {
        let Some(table) = table_for(EMPTY) else {
            panic!("out of range"); // keep should_panic green under TAM_ALLOW_NO_MAUDE
        };
        // No built-in or EMPTY-theory rule has 3 conclusions.
        table.edge_relation_color(ConcIdx(3), PremIdx(0));
    }

    /// The set block 3a actually covers for an EMPTY theory's real
    /// cache: the 5 trace-mode `special_intruder_rules` (`IEquality` and
    /// `NatConstr` are excluded — see the module docs' block-3 write-up
    /// and `special_intruder_rules`'s own doc comment on `IEquality`
    /// being diff-mode-only; `NatConstr` never enters a real cache at all
    /// unless the nat plugin's own assembly path adds it, which
    /// `ProofContext::assemble_intruder_rules`'s trace-mode path never
    /// does) plus the one fixed `FreshRule` slot.
    #[test]
    fn empty_theory_still_colors_every_name_its_real_cache_actually_has() {
        let Some(table) = table_for(EMPTY) else { return };
        table.rule_color(&fresh_rule_instance());
        for info in [
            IntrRuleACInfo::Coerce,
            IntrRuleACInfo::IRecv,
            IntrRuleACInfo::ISend,
            IntrRuleACInfo::PubConstr,
            IntrRuleACInfo::FreshConstr,
        ] {
            table.rule_color(&intr_rule_instance(info)); // must not panic
        }
        for name in BUILTIN_ACTION_NAMES {
            table.action_color(name); // must not panic
        }
    }

    #[test]
    fn builtin_and_theory_blocks_never_overlap() {
        let Some(table) = table_for(TWO_RULES) else { return };
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
        let Some(table) = table_for(TWO_RULES) else { return };
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
        let Some(a) = table_for(TWO_RULES) else { return };
        let Some(b) = table_for(TWO_RULES) else { return };
        assert_eq!(a, b);
    }

    /// A theory whose declaration order is the REVERSE of another's,
    /// but with the same rule/action NAMES, still produces the same
    /// table — pinning that declaration order truly never leaks in.
    #[test]
    fn declaration_order_does_not_affect_the_table() {
        const REVERSED: &str = "theory T begin\n\
            rule Apple:\n  [ A(), B() ] --[ Alpha() ]-> []\n\
            rule Zebra:\n  [ A(), B() ] --[ Beta() ]-> []\n\
            end";
        let Some(forward) = table_for(TWO_RULES) else { return };
        let Some(reversed) = table_for(REVERSED) else { return };
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
        let Some(table) = table_for(COLLIDING) else { return };
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
        let Some(table) = table_for(COLLIDING) else { return };
        let Some(empty_table) = table_for(EMPTY) else { return };
        assert_eq!(table.action_color("Fr"), empty_table.action_color("Fr"));
    }

    // -- Shape refinement (`shape_colors`) ----------------------------------
    //
    // Only built-in `KU` actions and structural vertices below, which a
    // default (empty) table already colors -- no maude needed.

    fn var(name: &str, sort: LSort) -> LNTerm {
        var_idx(name, sort, 0)
    }

    fn var_idx(name: &str, sort: LSort, idx: u64) -> LNTerm {
        tamarin_term::vterm::var_term(LVar::new(name, sort, idx))
    }

    fn fun2(name: &'static str, a: LNTerm, b: LNTerm) -> LNTerm {
        use tamarin_term::function_symbols::{Constructability, FunSym, NoEqSym, Privacy};
        let sym = NoEqSym::new(name.as_bytes().to_vec(), 2, Privacy::Public, Constructability::Constructor);
        LNTerm::App(FunSym::NoEq(sym), std::sync::Arc::from([a, b]))
    }

    fn ku(t: LNTerm) -> VertexKind {
        VertexKind::Action(nid(), crate::fact::ku_fact(t))
    }

    use tamarin_term::lterm::LSort;

    #[test]
    fn shape_colors_separate_different_sorts_and_symbols_but_not_renamings() {
        let table = ColorTable::default();
        let colors = table.shape_colors(&[
            ku(var("x", LSort::Msg)),
            ku(var("y", LSort::Msg)),
            ku(var("p", LSort::Pub)),
            ku(fun2("sign", var("x", LSort::Msg), var("k", LSort::Fresh))),
            ku(fun2("mac", var("x", LSort::Msg), var("k", LSort::Fresh))),
        ]);
        assert_eq!(colors[0], colors[1], "renamed msg variables have the same shape");
        assert_ne!(colors[0], colors[2], "a msg and a pub variable differ in shape");
        assert_ne!(colors[3], colors[4], "sign(..) and mac(..) differ in shape");
    }

    /// Names are never renamed, so they are part of the shape: `!KU('1')`
    /// and `!KU('2')` differ, like `!KU(~'n')` and `!KU(~'m')`.
    #[test]
    fn shape_colors_keep_names() {
        use tamarin_term::lterm::{fresh_term, pub_term};
        let table = ColorTable::default();
        let colors = table.shape_colors(&[
            ku(pub_term("1")),
            ku(pub_term("2")),
            ku(pub_term("1")),
            ku(fresh_term("n")),
            ku(fresh_term("m")),
        ]);
        assert_ne!(colors[0], colors[1], "'1' and '2' differ in shape");
        assert_eq!(colors[0], colors[2], "the same name has the same shape");
        assert_ne!(colors[3], colors[4], "~'n' and ~'m' differ in shape");
    }

    /// Erasure must re-sort AC arguments: `xor(x.0:msg, $p.1)` and its
    /// renaming `xor(x.1:msg, $p.0)` are sorted differently before erasure
    /// (`LVar` orders by index first), and would get different shapes
    /// without it.
    #[test]
    fn shape_colors_are_invariant_under_renaming_across_ac_argument_order() {
        use tamarin_term::builtin::xor;
        let table = ColorTable::default();
        let t1 = xor(var_idx("x", LSort::Msg, 0), var_idx("p", LSort::Pub, 1));
        let t2 = xor(var_idx("x", LSort::Msg, 1), var_idx("p", LSort::Pub, 0));
        assert_ne!(
            matches!(&t1, Term::App(_, args) if matches!(args[0], Term::Lit(Lit::Var(v)) if v.sort == LSort::Msg)),
            matches!(&t2, Term::App(_, args) if matches!(args[0], Term::Lit(Lit::Var(v)) if v.sort == LSort::Msg)),
            "precondition: the two terms order their msg/pub arguments differently"
        );
        let colors = table.shape_colors(&[ku(t1), ku(t2)]);
        assert_eq!(colors[0], colors[1]);
    }

    /// Colors are ranks among the part's distinct keys, so listing the
    /// same vertices in a different order permutes the colors along with
    /// them, and a lower base color always gets a lower final color.
    #[test]
    fn shape_colors_do_not_depend_on_vertex_order() {
        let table = ColorTable::default();
        let vertices = vec![
            ku(var("p", LSort::Pub)),
            VertexKind::Dummy(nid()),
            ku(var("x", LSort::Msg)),
            VertexKind::LessRelation,
        ];
        let reversed: Vec<VertexKind> = vertices.iter().rev().cloned().collect();
        let forward = table.shape_colors(&vertices);
        let mut backward = table.shape_colors(&reversed);
        backward.reverse();
        assert_eq!(forward, backward);

        let base = table.base_colors(&vertices);
        for i in 0..vertices.len() {
            for j in 0..vertices.len() {
                if base[i] < base[j] {
                    assert!(forward[i] < forward[j], "rank must preserve base-color order");
                }
            }
        }
    }
}



