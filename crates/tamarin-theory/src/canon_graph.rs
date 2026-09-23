// Currently GPL 3.0 until granted permission by the upstream authors
// of the tamarin-prover sources this file cites; list them with:
//   scripts/gen_license_headers.py --authors <this file>

//! (No HS analog.) Extraction of work.tex's graph part `gp(Γ) = (V, E,
//! c)` from a constraint [`System`] — Stage A of "Canonizing the
//! constraint system", per `work.tex`'s "Canonization of Graphs of Rule
//! Instances" and TODO.md's "Canonizing the graph" bullets. This module
//! builds the VERTEX and EDGE structure only: the coloring `c`
//! (TODO.md's skeleton scheme) and the external graph-canonizer
//! integration (Bliss) are later, separate stages.
//!
//! NOT [`crate::constraint::system::graph::repr`]: that module
//! (`GraphRepr`) is a DOT/JSON *visualization* intermediate
//! representation, with unrelated concepts like `Missing`/
//! `UnsolvedAction`/clustering, for a human-facing renderer — not
//! work.tex's mathematical graph part. Nor
//! [`crate::constraint::system::graph::color`]: that "color" is an
//! unrelated cosmetic HSV fill palette for DOT node rendering, the same
//! word for a different concept from the graph-theoretic vertex
//! coloring a later stage will need. Both are useful *style* references
//! (see [`extract_graph_part`]'s dummy-vertex handling, which mirrors
//! `graph::repr::NodeType::Missing`), but neither is reused directly.
//!
//! Three design decisions worth flagging, all made explicitly rather
//! than following work.tex's own draft literally:
//!
//! - Action-formula constraints (`f @ i`) are never merged, even when
//!   several share a timepoint `i` — see [`VertexKind::Action`]'s doc
//!   comment for why.
//! - A `NodeId` referenced only inside a quantifier's scope (a
//!   `Guarded::GGuarded` binder's `BVar::Bound` occurrences) is never
//!   turned into a vertex — only ground, already-committed action atoms
//!   are; see [`collect_action_atoms`].
//! - Every binary RELATION (a real `System::edges` conclusion→premise
//!   connection, an `i < j` less-than atom, and an action's link to its
//!   own timepoint) is reified as its own dedicated intermediate vertex
//!   — `src -> RelationVertex -> tgt` — rather than as a directly typed
//!   edge between `src` and `tgt`. Per `tamarin-prover/TODO.md`'s
//!   "Canonizing the graph" section ("dummy vertices which encode the
//!   less than relation and edge constraints get a fixed integer as
//!   color"), a relation's KIND lives entirely in its vertex's color (a
//!   later stage), so [`GraphEdge`] itself carries no kind at all — the
//!   graph canonizer only ever needs to reason about vertex colors,
//!   never edge colors. See [`VertexKind::EdgeRelation`]/
//!   [`VertexKind::LessRelation`]/[`VertexKind::AtTimepointRelation`].
//! - `last_atom` is reified the same way, but as a UNARY marker rather
//!   than a binary relation (there is no "other side"): a fresh
//!   [`VertexKind::LastAtomRelation`] vertex with a single edge to
//!   `last_atom`'s own vertex. Added 2026-09-18 after a real gap: giving
//!   `last_atom`'s `NodeId` a vertex without marking it as `last_atom`
//!   left two systems differing only in WHICH node is `last_atom`
//!   canonizing identically — a bare [`VertexKind::Dummy`] vertex looks
//!   the same whether or not it happens to be `last_atom`.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::canon_color::{Color, ColorTable};
use crate::constraint::constraints::NodeId;
use crate::constraint::system::System;
use crate::fact::LNFact;
use crate::guarded::{BVar, GAtom, GFact, GTerm, Guarded};
use crate::pretty_system::pretty_fact;
use crate::rule::{rule_name_string, ConcIdx, PremIdx, RuleACInst};

use tamarin_parser::ast::{SortHint, SuffixSort, VarSpec};
use tamarin_term::lterm::{LSort, LVar};
use tamarin_utils::color::{hsv_to_hex, Hsv};

/// What a vertex represents. Work.tex's graph part has two vertex kinds
/// (`i : ri` rule instances, `f @ i` action-formula constraints); the
/// rest are this implementation's own additions — see the module docs'
/// third design-decision bullet for [`VertexKind::EdgeRelation`]/
/// [`VertexKind::LessRelation`]/[`VertexKind::AtTimepointRelation`].
#[derive(Debug, Clone, PartialEq)]
pub enum VertexKind {
    /// `i : ri` — a rule-instance vertex.
    RuleInstance(NodeId, RuleACInst),
    /// Action-formula constraint `f @ i`.
    ///
    /// Each action atom gets its OWN vertex — even when several share a
    /// timepoint `i` — rather than merging same-timepoint actions into
    /// one vertex the way work.tex's own draft sketch proposed.
    /// Merging would still require permuting the merged
    /// actions against each other at canonization time (work.tex's own
    /// unresolved "PROBLEM" comment about tie-breaking `f(a), f(b) @ j`
    /// is exactly this). Keeping them as separate vertices instead lets
    /// the graph canonizer's own automorphism search resolve that
    /// permutation as an ordinary part of finding `Aut(G)`, with no
    /// separate AC-wrapping tie-break mechanism needed at all.
    ///
    /// Payload is `LNFact`, not the `GFact` a formula atom's own action
    /// is originally found in (`collect_action_atoms` still walks
    /// `Guarded`/`GFact` -- that structure is unavoidable while formulas
    /// are the source), and not the raw `LNFact` a `Goal::Action` already
    /// carries either -- both get bridged/normalized to this ONE shape
    /// HERE, at extraction time (`action_vertex_fact` for the
    /// `GFact` case), rather than each caller downstream (`vertex_to_term`,
    /// the `ColorTable`, the DOT renderer) handling two different payload
    /// types. Changed 2026-09-23 (was `GFact`) specifically so a
    /// `Goal::Action(nid, fact)` -- which already IS an `LNFact`, no
    /// elaboration needed -- can become a vertex through the exact same
    /// path a formula-derived action does, with no parallel vertex kind.
    Action(NodeId, LNFact),
    /// A `NodeId` with neither a rule instance nor an action atom of its
    /// own, referenced only via a relation endpoint or `last_atom`.
    /// Mirrors `graph::repr::NodeType::Missing`'s reason for existing:
    /// every `NodeId` the system references anywhere must resolve to
    /// SOME vertex, or a later stage's canonical-labelling lookup (which
    /// is exhaustive-or-panic, per `canon.rs`'s `canonicalize_guarded`)
    /// panics on a missing entry.
    Dummy(NodeId),
    /// Reifies one `System::edges` conclusion→premise connection:
    /// `src -> EdgeRelation(conc, prem) -> tgt` replaces a directly typed
    /// `src -> tgt` edge. See the module docs' third bullet. Carries the
    /// original edge's port indices — which conclusion slot on `src` and
    /// which premise slot on `tgt` this connection actually occupies.
    /// Without them, an edge landing in premise slot 0 of a rule
    /// instance would be graph-indistinguishable from one landing in
    /// slot 1 (both routed through a bare, payload-free `EdgeRelation`
    /// vertex) — losing real structural information the coloring stage
    /// ([`crate::canon_color`]) needs to tell apart which specific
    /// premise/conclusion an edge occupies.
    EdgeRelation(ConcIdx, PremIdx),
    /// Reifies one `i < j` less-than atom: `smaller -> LessRelation ->
    /// larger`. See the module docs' third bullet.
    LessRelation,
    /// Reifies the link between an [`VertexKind::Action`] vertex and the
    /// [`VertexKind::RuleInstance`]/[`VertexKind::Dummy`] vertex sharing
    /// its timepoint: `action -> AtTimepointRelation -> timepoint`. Not
    /// part of work.tex's own edge set — needed so the graph canonizer
    /// can never place an action at a timepoint other than the one it
    /// actually constrains — but given the same vertex-not-edge
    /// treatment as `EdgeRelation`/`LessRelation` for uniformity (see
    /// the module docs' third bullet).
    AtTimepointRelation,
    /// Reifies `System::last_atom`: `LastAtomRelation -> target`, where
    /// `target` is whichever `RuleInstance`/`Dummy` vertex owns the
    /// `NodeId` `last_atom` names. A UNARY marker, unlike
    /// `EdgeRelation`/`LessRelation`/`AtTimepointRelation` (there is no
    /// "other side" to a `last_atom` reference) — see the module docs'
    /// fourth bullet for the gap this closes: without this marker, a
    /// `last_atom` target that has no rule instance of its own gets a
    /// bare [`VertexKind::Dummy`] vertex indistinguishable from any
    /// other Dummy, so two systems differing only in WHICH node is
    /// `last_atom` would canonize identically.
    LastAtomRelation,
}

/// A directed structural edge, referencing vertices by their index into
/// [`GraphPart::vertices`]. Edges carry no kind of their own — every
/// relation's kind is encoded by the dedicated vertex it's routed
/// through instead (see the module docs' third design-decision bullet).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphEdge {
    pub src: usize,
    pub tgt: usize,
}

/// The extracted graph part `(V, E, c)` — vertices, edges, AND the
/// coloring `c` (`TODO.md`'s skeleton scheme, [`crate::canon_color`]),
/// bundled together as ONE value rather than passed around as two
/// separately-threaded arguments. A `ColorTable` is meaningful only
/// relative to the specific `GraphPart` it colors (it's built from the
/// same theory the part's `System` came from), so keeping them apart
/// invited a caller to mismatch a part with the wrong table — this
/// couples them at construction time instead: [`extract_graph_part`] is
/// the only place a `GraphPart` is built, and it always builds its own
/// `colors` alongside `vertices`/`edges`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GraphPart {
    pub vertices: Vec<VertexKind>,
    pub edges: Vec<GraphEdge>,
    pub colors: ColorTable,
}

impl GraphPart {
    /// The color of `self.vertices[idx]`, per `self.colors` — the
    /// entry point most callers actually want over reaching into
    /// `self.colors.vertex_color(&self.vertices[idx])` directly.
    pub fn vertex_color(&self, idx: usize) -> Color {
        self.colors.vertex_color(&self.vertices[idx])
    }
}

/// Extracts the graph part from `sys` (Stage A — see the module docs),
/// pairing it with the caller-supplied `colors` table ([`crate::canon_color`],
/// Stage B) in the same call — `colors` must be the table for the SAME
/// (elaborated) theory `sys` was produced from, or vertex coloring later
/// panics on a rule/action name this table wasn't built to cover. This
/// function needs no `&Theory` at all: `colors` is unique per theory and
/// the caller — whoever elaborated the theory / built the `ProofContext`
/// a real proof search runs under, or a test constructing both directly
/// — already has one on hand (see [`ColorTable::build`]'s own doc
/// comment for why it takes `&[OpenProtoRule]` + `&IntrRuleCache` rather
/// than `&Theory`).
pub fn extract_graph_part(sys: &System, colors: &ColorTable) -> GraphPart {
    let mut vertices: Vec<VertexKind> = Vec::new();
    let mut edges: Vec<GraphEdge> = Vec::new();
    // NodeId -> index of its RuleInstance/Dummy vertex. Every NodeId the
    // system references gets exactly one entry here. Action vertices are
    // NOT indexed by this map — several can legitimately share a NodeId.
    let mut node_vertex: BTreeMap<NodeId, usize> = BTreeMap::new();

    // 1. Rule-instance vertices, in HS `M.toList` order — a deterministic
    //    INPUT order; the graph canonizer (a later stage), not this
    //    order, decides the final canonical vertex order.
    for (nid, ru) in sys.nodes_in_map_order() {
        let idx = vertices.len();
        vertices.push(VertexKind::RuleInstance(*nid, ru.clone()));
        node_vertex.insert(*nid, idx);
    }

    // 2. Edges — a dummy vertex for any endpoint not already a rule
    //    instance (mirrors `graph::repr::NodeType::Missing`), reified as
    //    `src -> EdgeRelation -> tgt`.
    for e in sys.edges_in_set_order() {
        let s = get_vertex_or_create_dummy_vertex(e.src.0, &mut vertices, &mut node_vertex);
        let t = get_vertex_or_create_dummy_vertex(e.tgt.0, &mut vertices, &mut node_vertex);
        push_relation(
            &mut vertices,
            &mut edges,
            VertexKind::EdgeRelation(e.src.1, e.tgt.1),
            s,
            t,
        );
    }

    // 3. Less-than atoms, reified as `smaller -> LessRelation -> larger`.
    for la in sys.less_atoms_in_set_order() {
        let s = get_vertex_or_create_dummy_vertex(la.smaller, &mut vertices, &mut node_vertex);
        let t = get_vertex_or_create_dummy_vertex(la.larger, &mut vertices, &mut node_vertex);
        push_relation(&mut vertices, &mut edges, VertexKind::LessRelation, s, t);
    }

    // 4. `last_atom` — reified as `LastAtomRelation -> target` (a unary
    //    marker, not a full `push_relation`, since there is no "other
    //    side"), so WHICH vertex is `last_atom` is discoverable via edge
    //    topology, not just "some vertex exists for this NodeId" (which
    //    leaves it indistinguishable from a bare edge endpoint with no
    //    rule instance of its own — see `VertexKind::LastAtomRelation`'s
    //    own doc comment).
    if let Some(la) = sys.last_atom {
        let target = get_vertex_or_create_dummy_vertex(la, &mut vertices, &mut node_vertex);
        let marker_idx = vertices.len();
        vertices.push(VertexKind::LastAtomRelation);
        edges.push(GraphEdge {
            src: marker_idx,
            tgt: target,
        });
    }

    // 5. Action-formula vertices: one per (unquantified) action
    //    atom found at the top level of `formulas` ∪ `solved_formulas`
    //    (the canonization plan's field audit treats these as a single
    //    unioned set: `solved_formulas` is pure memoisation of
    //    already-processed formulas, not independent content, so an atom
    //    present in both must still yield only ONE vertex — the dedup
    //    below). `lemmas` are excluded: they are universally-quantified
    //    background assumptions (see `SystemContent::lemmas`'s own doc
    //    comment), not per-instance ground facts, and work.tex's
    //    graph-part model has no lemma-derived vertex kind to begin
    //    with. Each action's link to its timepoint is reified as
    //    `action -> AtTimepointRelation -> timepoint`.
    //
    //    ALSO one per `Goal::Action(nid, fact)` in `sys.goals` -- regardless
    //    of `GoalStatus::solved` (a solved goal is kept "for replay", per
    //    its own doc comment, not folded back into `formulas`, so its
    //    content would otherwise never appear ANYWHERE the graph part
    //    looks). This closes a real gap: `!KU(_)` sub-goals synthesized by
    //    `insert_goal_with_loop_flag`'s pair/inv/prod decomposition
    //    (`reduction.rs`) carry genuinely new term content -- e.g. two
    //    sibling `!KU(mac1)`/`!KU(mac2)` goals -- that never reaches
    //    `sys.formulas` at all, so before this it was invisible to
    //    canonicalization entirely: the goal's own `NodeId` fell back to a
    //    bare content-free `Dummy` vertex, silently discarding which TERM
    //    was actually pending derivation there. Deduped against the
    //    formula-derived actions below (a goal-action that also happens
    //    to already be a formula atom must still yield only ONE vertex).
    let mut raw_actions: Vec<(NodeId, GFact)> = Vec::new();
    for f in sys.formulas.iter().chain(sys.solved_formulas.iter()) {
        collect_action_atoms(f, &mut raw_actions);
    }
    // Bridge each `GFact` to the `LNFact` `VertexKind::Action` now holds
    // (`action_vertex_fact`) BEFORE sorting/deduping -- two syntactically
    // different `GFact`s can still elaborate to the identical `LNFact`
    // (e.g. differing only in a spelling the signature normalizes), so
    // deduping on the POST-elaboration form is the semantically correct
    // one, not just a convenient side effect of `LNFact` already
    // deriving `Ord` (dropping the bespoke `cmp_fact` comparator).
    let mut actions: Vec<(NodeId, LNFact)> = raw_actions
        .into_iter()
        .map(|(nid, gfact)| (nid, action_vertex_fact(&gfact)))
        .collect();
    for (goal, _status) in sys.goals.iter() {
        if let crate::constraint::constraints::Goal::Action(nid, fact) = goal {
            actions.push((*nid, fact.clone()));
        }
    }
    actions.sort();
    actions.dedup();

    for (nid, fact) in actions {
        let node_idx = get_vertex_or_create_dummy_vertex(nid, &mut vertices, &mut node_vertex);
        let action_idx = vertices.len();
        vertices.push(VertexKind::Action(nid, fact));
        push_relation(
            &mut vertices,
            &mut edges,
            VertexKind::AtTimepointRelation,
            action_idx,
            node_idx,
        );
    }

    GraphPart {
        vertices,
        edges,
        colors: colors.clone(),
    }
}

/// Returns the index of `nid`'s `RuleInstance`/`Dummy` vertex, creating a
/// fresh [`VertexKind::Dummy`] on first reference.
fn get_vertex_or_create_dummy_vertex(
    nid: NodeId,
    vertices: &mut Vec<VertexKind>,
    node_vertex: &mut BTreeMap<NodeId, usize>,
) -> usize {
    *node_vertex.entry(nid).or_insert_with(|| {
        let idx = vertices.len();
        vertices.push(VertexKind::Dummy(nid));
        idx
    })
}

/// Reifies a binary relation `src ~ tgt` as its own fresh vertex of kind
/// `relation_kind`, with edges `src -> relation -> tgt` (preserving
/// orientation) — see the module docs' third design-decision bullet.
///
/// Every call creates a NEW vertex, even for a repeated `(src, tgt)`
/// pair or relation kind — matching [`VertexKind::Action`]'s own
/// never-merge philosophy: two distinct relation instances must never be
/// silently identified with each other just because their endpoints (or
/// kind) happen to coincide.
fn push_relation(
    vertices: &mut Vec<VertexKind>,
    edges: &mut Vec<GraphEdge>,
    relation_kind: VertexKind,
    src: usize,
    tgt: usize,
) {
    let relation_idx = vertices.len();
    vertices.push(relation_kind);
    edges.push(GraphEdge {
        src,
        tgt: relation_idx,
    });
    edges.push(GraphEdge {
        src: relation_idx,
        tgt,
    });
}

/// Collects `(timepoint, fact)` for every unquantified action
/// atom reachable from `g` by recursing ONLY through `Guarded::Conj` —
/// i.e. through the formula store's own implicit top-level conjunction.
///
/// An action atom nested inside a `Guarded::Disj` alternative or a
/// `Guarded::GGuarded` quantifier's guards/body is deliberately NOT
/// collected: such an atom is not yet a committed, ground constraint — a
/// `Disj` alternative may not hold, and a `GGuarded` binder's variable
/// has no `NodeId` of its own until the quantifier is instantiated (its
/// occurrences are De-Bruijn `BVar::Bound` indices, not free variables —
/// let alone `NodeId`s — at all). Turning either into a vertex would be
/// turning something that isn't yet part of the trace into part of the
/// graph.
fn collect_action_atoms(g: &Guarded, out: &mut Vec<(NodeId, GFact)>) {
    match g {
        Guarded::Atom(GAtom::Action(fact, GTerm::Var(BVar::Free(vs)))) if is_node_sort(vs.sort) => {
            out.push((varspec_to_node_id(vs), fact.clone()));
        }
        Guarded::Conj(items) => {
            for item in items.iter() {
                collect_action_atoms(item, out);
            }
        }
        _ => {}
    }
}

/// Converts a formula-derived action atom's `GFact` to the `LNFact`
/// [`VertexKind::Action`] actually holds, so a formula-sourced action and
/// a `Goal::Action`'s already-`LNFact` action become the exact same
/// vertex shape (see `VertexKind::Action`'s own doc comment for why this
/// unification exists).
///
/// Reuses the SAME bridge `system_import::parse_fact` already uses in
/// production to reconstruct `LNFact`s from a captured `System`'s formula
/// atoms: [`crate::guarded::gfact_to_fact`] (purely structural, no
/// signature needed) then [`crate::elaborate::fact_to_lnfact`] (resolves
/// the fact's function-symbol names against the CURRENTLY INSTALLED
/// signature — see `elaborate::set_user_funs_for_theory`'s own doc
/// comment). The caller ([`extract_graph_part`]) must have that signature
/// installed for the SAME theory `sys` came from — the same precondition
/// `extract_graph_part`'s own doc comment already states for its
/// caller-supplied `ColorTable`.
///
/// **Panics** if `gfact` still carries a `Bound` variable, or if the
/// installed signature can't elaborate one of its terms. Both are
/// treated as caller-contract violations, not recoverable runtime
/// conditions: [`collect_action_atoms`] only ever extracts a `GFact` from
/// a GROUND, already-committed conjunct (never from inside a
/// `Guarded::Disj` alternative or a `GGuarded` binder's body — see that
/// function's own doc comment), so every `GFact` this is actually called
/// on is assumed already closed: a leftover `Bound` var surfacing here
/// means that extraction discipline was violated somewhere upstream,
/// which should fail loudly rather than silently mis-canonize. This is
/// why [`crate::guarded::gfact_to_fact`] is used deliberately instead of
/// the fallible `crate::guarded::try_gfact_to_fact`: a violation surfaces
/// immediately, at the point of conversion, instead of limping forward as
/// a `None`/`Result::Err` a caller could mishandle.
fn action_vertex_fact(gfact: &GFact) -> LNFact {
    let pfact = crate::guarded::gfact_to_fact(gfact);
    crate::elaborate::fact_to_lnfact(&pfact).unwrap_or_else(|e| {
        panic!(
            "action_vertex_fact: {e} -- {gfact:?} did not elaborate under the currently \
             installed signature (wrong/missing set_user_funs_for_theory guard for this \
             system's theory?)"
        )
    })
}

/// Whether a parser-AST sort hint denotes `LSort::Node` — covers both the
/// bare-sigil (`SortHint::Node`, e.g. `#i`) and suffix
/// (`SortHint::Suffix(SuffixSort::Node)`, e.g. `i:node`) spellings the
/// parser can produce for a timepoint variable. Small local duplicate of
/// the relevant arm of `canon.rs`'s private `sort_hint_to_lsort` — kept
/// local rather than imported across a module boundary that has no other
/// reason to depend on `canon.rs`, matching that function's own doc
/// comment's rationale for staying local.
fn is_node_sort(s: SortHint) -> bool {
    matches!(s, SortHint::Node | SortHint::Suffix(SuffixSort::Node))
}

/// A parser-AST variable spec of Node sort as the `NodeId` (`LVar`) it
/// denotes.
fn varspec_to_node_id(v: &VarSpec) -> NodeId {
    LVar::new(v.name.as_str(), LSort::Node, v.idx)
}

// =============================================================================
// Graphviz rendering
// =============================================================================
//
// Renders a [`GraphPart`] as a Graphviz DOT document, for visually
// inspecting/comparing extracted graph parts (this crate's own
// `constraint::system::dot`/`dot_showdot` render a whole `System` for the
// interactive UI and batch `--output-dot`; this is a separate, much
// smaller renderer for the `GraphPart` this module extracts, since a
// `GraphPart` has already thrown away the port-level premise/conclusion
// detail — `NodeConc`/`NodePrem` — those renderers key edges off of).
//
// Shapes are chosen to visually echo the ones HS/`dot_showdot.rs` use for
// the same conceptual role where one exists (not byte-identical — this
// format has no premise/conclusion PORTS to route edges through, and no
// compact/full or clustering options), and to give each of the three
// relation-vertex kinds its own small, visually-lightweight shape (they
// are structural plumbing, not content, so they should read as smaller/
// quieter than a `RuleInstance`/`Action` vertex, not compete with them):
//   - [`VertexKind::RuleInstance`]: a `record` shape with the same
//     three-row shape `dot_showdot.rs`'s non-"boring" rule nodes use
//     (premises / `#i : RuleName[actions]` / conclusions) — see
//     `dot_showdot.rs`'s `mk_node`/`rule_label_doc`.
//   - [`VertexKind::Action`]: a plain `ellipse`, echoing
//     `dot_showdot.rs`'s `NodeType::UnsolvedAction` (`mk_simple_node`).
//   - [`VertexKind::Dummy`]: a `diamond`, dashed — HS's closest analog is
//     `NodeType::Missing`'s `trapezium`/`invtrapezium`, but `Dummy` here
//     doesn't distinguish a missing conclusion from a missing premise
//     (see [`VertexKind::Dummy`]'s own doc comment), so one shape covers
//     both, plus whatever else can create a dummy (`last_atom`, a bare
//     relation endpoint).
//   - [`VertexKind::EdgeRelation`]: a tiny `point` — the closest DOT has
//     to "just a pass-through connector, no content of its own".
//   - [`VertexKind::LessRelation`]: a small `triangle` — evokes an
//     ordering/comparison ("<").
//   - [`VertexKind::AtTimepointRelation`]: a small `hexagon` — visually
//     distinct from every other shape used here, matching that it is
//     this module's own addition with no HS/work.tex counterpart.
//   - [`VertexKind::LastAtomRelation`]: a small `star` — at most one per
//     graph part (`last_atom` is an `Option<NodeId>`), so it doesn't need
//     to visually blend in with a family of repeated relation shapes the
//     way the others do.
//
// The FILL color, in contrast, is not hardcoded per kind — it comes
// straight from `part.colors` via [`dot_fill_color`]: two vertices the
// canonizer considers indistinguishable (same `ColorTable` color) always
// render with the exact same fill, and two vertices it distinguishes
// always render with visibly different fills. This makes the coloring
// stage's own output directly inspectable (e.g. "why do these two
// `RuleInstance` vertices look the same color? oh, they're both the
// built-in `ISend` rule") instead of every rule instance always being
// the same static light blue regardless of which rule it actually is.

/// Renders `part` as a self-contained `digraph G { ... }` DOT document.
pub fn to_graphviz(part: &GraphPart) -> String {
    let mut out = String::new();
    out.push_str("digraph G {\n");
    out.push_str("  rankdir=TB;\n");
    out.push_str("  node [fontname=\"Helvetica\", fontsize=10];\n");
    out.push_str("  edge [fontname=\"Helvetica\", fontsize=10];\n\n");

    for (idx, v) in part.vertices.iter().enumerate() {
        let fill = dot_fill_color(part.colors.vertex_color(v));
        write_vertex(&mut out, idx, v, &fill);
    }
    out.push('\n');
    for e in &part.edges {
        write_edge(&mut out, e);
    }

    out.push_str("}\n");
    out
}

/// Maps a [`Color`] (an opaque `ColorTable` integer — only equality
/// between two colors is meaningful, not order or magnitude) to a
/// visually distinct DOT hex fill color, for [`to_graphviz`]'s use.
///
/// Uses golden-angle hue stepping (`color * φ⁻¹ mod 1`, scaled to
/// `[0, 360)`) — a standard technique for generating a sequence of
/// well-SEPARATED hues without needing to know the total color count up
/// front (unlike `tamarin_utils::color::gen_color_groups`, which needs a
/// group-size layout ahead of time and is already spoken for as the
/// literal HS-faithful `nodeColorMap` port in
/// `constraint::system::graph::color` — a different, unrelated cosmetic
/// palette, per this module's own top-level doc comment). Consecutive
/// integers land far apart on the hue wheel, so two DIFFERENT small
/// colors are very unlikely to look alike, even though nothing here
/// GUARANTEES distinctness for arbitrarily many colors (a real
/// `ColorTable` only ever has on the order of a few dozen colors, so
/// this is not a practical concern). Saturation/value are fixed at a
/// pastel level so dark vertex-label text stays legible on every fill.
fn dot_fill_color(c: Color) -> String {
    const GOLDEN_ANGLE_TURNS: f64 = 0.618_033_988_749_895; // 1/phi
    let hue = ((c as f64) * GOLDEN_ANGLE_TURNS).fract() * 360.0;
    hsv_to_hex(Hsv::new(hue, 0.45, 0.92))
}

fn write_vertex(out: &mut String, idx: usize, v: &VertexKind, fill: &str) {
    match v {
        VertexKind::RuleInstance(nid, ru) => {
            let prem_cells: Vec<String> = ru
                .premises
                .iter()
                .map(|fa| escape_record_field(&pretty_fact(fa)))
                .collect();
            let conc_cells: Vec<String> = ru
                .conclusions
                .iter()
                .map(|fa| escape_record_field(&pretty_fact(fa)))
                .collect();
            let mut mid = format!("V{idx}  {nid} : {}", rule_name_string(ru));
            if !ru.actions.is_empty() {
                let acts: Vec<String> = ru.actions.iter().map(pretty_fact).collect();
                write!(mid, "[{}]", acts.join(", ")).ok();
            }
            let mid = escape_record_field(&mid);

            let mut label = String::from("{");
            if !prem_cells.is_empty() {
                write!(label, "{{{}}}|", prem_cells.join("|")).ok();
            }
            label.push_str(&mid);
            if !conc_cells.is_empty() {
                write!(label, "|{{{}}}", conc_cells.join("|")).ok();
            }
            label.push('}');
            writeln!(
                out,
                "  n{idx} [shape=record, style=filled, fillcolor=\"{fill}\", label=\"{label}\"];"
            )
            .ok();
        }
        VertexKind::Action(nid, fact) => {
            let fact_str = pretty_fact(fact);
            let label = escape_dot_label(&format!("V{idx}  {fact_str} @ {nid}"));
            writeln!(
                out,
                "  n{idx} [shape=ellipse, style=filled, fillcolor=\"{fill}\", label=\"{label}\"];"
            )
            .ok();
        }
        VertexKind::Dummy(nid) => {
            let label = escape_dot_label(&format!("V{idx}  {nid}"));
            writeln!(
                out,
                "  n{idx} [shape=diamond, style=\"filled,dashed\", fillcolor=\"{fill}\", label=\"{label}\"];"
            )
            .ok();
        }
        VertexKind::EdgeRelation(conc, prem) => {
            let xlabel = escape_dot_label(&format!("C{}\u{2192}P{}", conc.0, prem.0));
            writeln!(
                out,
                "  n{idx} [shape=point, width=0.40, style=filled, fillcolor=\"{fill}\", label=\"\", xlabel=\"{xlabel}\"];"
            )
            .ok();
        }
        VertexKind::LessRelation => {
            let label = escape_dot_label("<");
            writeln!(
                out,
                "  n{idx} [shape=triangle, width=0.25, height=0.2, style=filled, fillcolor=\"{fill}\", label=\"{label}\"];"
            )
            .ok();
        }
        VertexKind::AtTimepointRelation => {
            let label = escape_dot_label(&format!("V{idx}"));
            writeln!(
                out,
                "  n{idx} [shape=hexagon, width=0.25, height=0.2, style=filled, fillcolor=\"{fill}\", label=\"{label}\"];"
            )
            .ok();
        }
        VertexKind::LastAtomRelation => {
            let label = escape_dot_label("last");
            writeln!(
                out,
                "  n{idx} [shape=star, width=0.3, height=0.3, style=filled, fillcolor=\"{fill}\", label=\"{label}\"];"
            )
            .ok();
        }
    }
}

fn write_edge(out: &mut String, e: &GraphEdge) {
    // No per-kind styling any more: a relation's kind lives entirely in
    // the vertex it's routed through (see the module docs' third
    // design-decision bullet), so every edge renders uniformly.
    writeln!(out, "  n{} -> n{};", e.src, e.tgt).ok();
}

/// Escapes a plain (non-record) DOT quoted-label value: backslash and
/// double-quote are the only characters that can break out of the
/// surrounding `label="..."` attribute syntax.
fn escape_dot_label(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Escapes a DOT *record*-shape label field: on top of
/// [`escape_dot_label`]'s pair, a record label additionally treats `{`,
/// `}`, `<`, `>`, and `|` as structural (row/column/port syntax), so
/// content containing them — e.g. a pretty-printed pair term `<a, b>` — has
/// to have them escaped too, or it silently corrupts the record's shape
/// instead of erroring.
fn escape_record_field(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if matches!(c, '\\' | '"' | '{' | '}' | '<' | '>' | '|') {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::constraints::{Edge, LessAtom, Reason};
    use crate::guarded::formula_to_guarded;
    use crate::rule::{
        ConcIdx, PremIdx, ProtoRuleACInstInfo, ProtoRuleName, Rule, RuleAttributes, RuleInfo,
    };
    use crate::theory::Theory;
    use std::sync::Arc;
    use tamarin_parser::parser::{parse_formula_str, parse_theory};

    /// Parses+elaborates `src` — mirrors `canon_color.rs`'s own test
    /// helper of the same name/shape.
    fn theory(src: &str) -> Theory {
        let parsed = parse_theory(src, &[]).unwrap_or_else(|e| panic!("parse: {e}"));
        crate::elaborate::elaborate(&parsed).unwrap_or_else(|e| panic!("elaborate: {e:?}"))
    }

    /// The `&ColorTable` [`extract_graph_part`] needs, built from `src`'s
    /// own protocol rules and an EMPTY `IntrRuleCache` — none of this
    /// module's tests color a `RuleInfo::Intr` vertex (only
    /// `to_graphviz_renders_a_well_formed_digraph_document`/
    /// `same_timepoint_actions_render_as_two_ellipses_with_attimepoint_relations`
    /// actually query colors at all, and only for THIS theory's own
    /// declared rules/actions), so an empty cache is both correct here
    /// and avoids needing a real maude process for tests that are
    /// otherwise purely structural. See `canon_color.rs`'s own test
    /// module for the maude-backed helper real intruder-rule coverage
    /// needs.
    fn color_table(src: &str) -> ColorTable {
        let elaborated = theory(src);
        let protocol_rules: Vec<crate::theory::OpenProtoRule> =
            elaborated.rules().cloned().collect();
        ColorTable::build(
            &protocol_rules,
            &crate::constraint::solver::context::IntrRuleCache::from(Vec::new()),
        )
    }

    const EMPTY: &str = "theory T begin\nend";

    fn nid(name: &str, idx: u64) -> NodeId {
        LVar::new(name, LSort::Node, idx)
    }

    fn proto_rule(name: &str) -> RuleACInst {
        Rule::new(
            RuleInfo::Proto(ProtoRuleACInstInfo {
                name: ProtoRuleName::Stand(tamarin_term::intern::intern_str(name)),
                attributes: RuleAttributes::default(),
                loop_breakers: Vec::new(),
            }),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
    }

    /// Parses a surface formula string straight to its guarded form —
    /// mirrors `canon.rs`'s own test helper of the same name/shape.
    fn g(s: &str) -> Guarded {
        let f = parse_formula_str(s).unwrap_or_else(|e| panic!("parse {s:?}: {e}"));
        formula_to_guarded(&f).unwrap_or_else(|e| panic!("formula_to_guarded {s:?}: {e}"))
    }

    /// Parses `s` as a bare fact (reusing [`g`]'s formula-parser path) and
    /// unwraps the `GAtom::Pred` atom it must produce -- the same shape
    /// [`collect_action_atoms`] hands to [`action_vertex_fact`].
    fn gfact(s: &str) -> GFact {
        match g(s) {
            Guarded::Atom(GAtom::Pred(f)) => f,
            other => panic!("{s:?} did not parse as a bare fact (Pred atom): {other:?}"),
        }
    }

    /// Installs a minimal, no-custom-functions signature --
    /// `action_vertex_fact`'s precondition (mirrors `system_import.rs`'s
    /// own `install_test_signature`).
    fn install_empty_signature() -> crate::elaborate::UserFunsForTheoryGuard {
        let thy = parse_theory("theory T begin\nend", &[]).expect("parse minimal theory");
        crate::elaborate::set_user_funs_for_theory(&thy)
    }

    #[test]
    fn action_vertex_fact_converts_a_ground_gfact_to_the_matching_lnfact() {
        let _guard = install_empty_signature();
        let converted = action_vertex_fact(&gfact("P(x)"));
        let expected = crate::fact::proto_fact(
            crate::fact::Multiplicity::Linear,
            "P",
            vec![tamarin_term::vterm::var_term(LVar::new(
                "x",
                LSort::Msg,
                0,
            ))],
        );
        assert_eq!(converted, expected);
    }

    /// The explicit invariant this design relies on: an action vertex's
    /// `GFact` is assumed already closed (no leftover `Bound` var) by
    /// construction ([`collect_action_atoms`] never extracts one from
    /// inside a `Disj`/`GGuarded` scope). A violation must panic loudly,
    /// not be swallowed by a fallible conversion.
    #[test]
    #[should_panic(expected = "left-over bound variable")]
    fn action_vertex_fact_panics_on_a_leftover_bound_variable() {
        let _guard = install_empty_signature();
        let bound_gfact = GFact {
            persistent: false,
            name: "P".to_string(),
            args: std::sync::Arc::from([GTerm::Var(BVar::Bound(0))]),
            annotations: Vec::new(),
        };
        let _ = action_vertex_fact(&bound_gfact);
    }

    fn action_fact_names(part: &GraphPart) -> Vec<(NodeId, String)> {
        let mut out: Vec<(NodeId, String)> = part
            .vertices
            .iter()
            .filter_map(|v| match v {
                VertexKind::Action(nid, fact) => {
                    Some((*nid, crate::fact::fact_tag_name(&fact.tag)))
                }
                _ => None,
            })
            .collect();
        out.sort();
        out
    }

    /// Asserts that `part` reifies the relation `src ~ tgt` via SOME
    /// vertex of kind `relation_kind` — i.e. `GraphEdge{src, rel}` and
    /// `GraphEdge{rel, tgt}` both exist for some `rel` whose
    /// `part.vertices[rel] == relation_kind`. This is the standard shape
    /// every relation (`EdgeRelation`/`LessRelation`/
    /// `AtTimepointRelation`) now takes — see the module docs' third
    /// design-decision bullet.
    fn assert_relation(part: &GraphPart, src: usize, relation_kind: &VertexKind, tgt: usize) {
        let found = part.vertices.iter().enumerate().any(|(rel, vk)| {
            vk == relation_kind
                && part.edges.contains(&GraphEdge { src, tgt: rel })
                && part.edges.contains(&GraphEdge { src: rel, tgt })
        });
        assert!(
            found,
            "expected a {relation_kind:?} vertex reifying {src} ~ {tgt}, not found in {part:?}"
        );
    }

    #[test]
    fn rule_instances_and_edge_get_extracted() {
        let mut sys = System::default();
        sys.add_node(nid("i", 1), proto_rule("A"));
        sys.add_node(nid("i", 2), proto_rule("B"));
        sys.content_mut().edges.push(Edge {
            src: (nid("i", 1), ConcIdx(0)),
            tgt: (nid("i", 2), PremIdx(0)),
        });

        let part = extract_graph_part(&sys, &color_table(EMPTY));

        // 2 rule instances + 1 reified EdgeRelation vertex.
        assert_eq!(part.vertices.len(), 3);
        assert!(matches!(&part.vertices[0], VertexKind::RuleInstance(n, _) if *n == nid("i", 1)));
        assert!(matches!(&part.vertices[1], VertexKind::RuleInstance(n, _) if *n == nid("i", 2)));
        assert_relation(&part, 0, &VertexKind::EdgeRelation(ConcIdx(0), PremIdx(0)), 1);
        assert_eq!(part.edges.len(), 2);
    }

    #[test]
    fn edge_endpoint_without_rule_instance_gets_dummy_vertex() {
        let mut sys = System::default();
        sys.add_node(nid("i", 1), proto_rule("A"));
        // `i.2` is referenced by the edge but has no rule instance.
        sys.content_mut().edges.push(Edge {
            src: (nid("i", 1), ConcIdx(0)),
            tgt: (nid("i", 2), PremIdx(0)),
        });

        let part = extract_graph_part(&sys, &color_table(EMPTY));

        // 1 rule instance + 1 dummy + 1 reified EdgeRelation vertex.
        assert_eq!(part.vertices.len(), 3);
        assert!(matches!(&part.vertices[1], VertexKind::Dummy(n) if *n == nid("i", 2)));
        assert_relation(&part, 0, &VertexKind::EdgeRelation(ConcIdx(0), PremIdx(0)), 1);
        assert_eq!(part.edges.len(), 2);
    }

    #[test]
    fn less_atom_endpoints_get_dummy_vertices_and_a_less_relation() {
        let mut sys = System::default();
        sys.content_mut()
            .less_atoms
            .push(LessAtom::new(nid("i", 1), nid("i", 2), Reason::Formula));

        let part = extract_graph_part(&sys, &color_table(EMPTY));

        // 2 dummies + 1 reified LessRelation vertex.
        assert_eq!(part.vertices.len(), 3);
        assert!(matches!(&part.vertices[0], VertexKind::Dummy(n) if *n == nid("i", 1)));
        assert!(matches!(&part.vertices[1], VertexKind::Dummy(n) if *n == nid("i", 2)));
        assert_relation(&part, 0, &VertexKind::LessRelation, 1);
        assert_eq!(part.edges.len(), 2);
    }

    #[test]
    fn last_atom_gets_a_vertex_even_with_no_edges() {
        let mut sys = System::default();
        sys.content_mut().last_atom = Some(nid("i", 7));

        let part = extract_graph_part(&sys, &color_table(EMPTY));

        // A Dummy for the target NodeId, PLUS a LastAtomRelation marker
        // reifying that it specifically is `last_atom` (see
        // `VertexKind::LastAtomRelation`'s own doc comment for why a
        // bare Dummy alone isn't enough).
        assert_eq!(
            part.vertices,
            vec![VertexKind::Dummy(nid("i", 7)), VertexKind::LastAtomRelation]
        );
        assert_relation_unary(&part, 1, 0);
    }

    /// Asserts a UNARY relation `GraphEdge { src, tgt }` exists directly
    /// (no intermediate vertex — unlike [`assert_relation`], which is for
    /// the binary `src -> relation -> tgt` shape).
    fn assert_relation_unary(part: &GraphPart, src: usize, tgt: usize) {
        assert!(
            part.edges.contains(&GraphEdge { src, tgt }),
            "expected an edge {src} -> {tgt}, not found in {part:?}"
        );
    }

    #[test]
    fn extract_graph_part_distinguishes_which_node_is_last_atom() {
        // Two otherwise-identical systems (same two rule instances, same
        // insertion order) that differ ONLY in which node is `last_atom`.
        // Before the fix, `last_atom`'s target got a bare vertex
        // indistinguishable from any other Dummy/RuleInstance, so these
        // two graph parts canonicalized identically — the bug this test
        // guards against.
        let mut sys_a = System::default();
        sys_a.add_node(nid("i", 1), proto_rule("A"));
        sys_a.add_node(nid("i", 2), proto_rule("A"));
        sys_a.content_mut().last_atom = Some(nid("i", 1));

        let mut sys_b = System::default();
        sys_b.add_node(nid("i", 1), proto_rule("A"));
        sys_b.add_node(nid("i", 2), proto_rule("A"));
        sys_b.content_mut().last_atom = Some(nid("i", 2));

        let part_a = extract_graph_part(&sys_a, &color_table(EMPTY));
        let part_b = extract_graph_part(&sys_b, &color_table(EMPTY));

        // Same vertex set (both nodes have rule instances, plus the
        // LastAtomRelation marker), but the marker's edge target differs.
        assert_eq!(part_a.vertices, part_b.vertices);
        assert_ne!(part_a.edges, part_b.edges);
        assert_relation_unary(&part_a, 2, 0);
        assert_relation_unary(&part_b, 2, 1);
    }

    /// Pins the key design decision: two action facts sharing a
    /// timepoint become TWO separate vertices (never merged into one),
    /// each reifying its OWN `AtTimepointRelation` back to the shared
    /// timepoint's vertex.
    #[test]
    fn same_timepoint_actions_get_separate_vertices_not_merged() {
        let mut sys = System::default();
        sys.content_mut()
            .formulas
            .push(Arc::new(g("P(x) @ #i & Q(y) @ #i")));

        let part = extract_graph_part(&sys, &color_table(EMPTY));

        assert_eq!(
            action_fact_names(&part),
            vec![
                (nid("i", 0), "P".to_string()),
                (nid("i", 0), "Q".to_string())
            ]
        );
        let action_indices: Vec<usize> = part
            .vertices
            .iter()
            .enumerate()
            .filter_map(|(idx, v)| matches!(v, VertexKind::Action(_, _)).then_some(idx))
            .collect();
        assert_eq!(action_indices.len(), 2);
        let timepoint_idx = part
            .vertices
            .iter()
            .position(|v| matches!(v, VertexKind::Dummy(n) if *n == nid("i", 0)))
            .expect("timepoint dummy vertex");
        for a in action_indices {
            assert_relation(&part, a, &VertexKind::AtTimepointRelation, timepoint_idx);
        }
    }

    /// The same ground atom present in both `formulas` and
    /// `solved_formulas` (the union-as-memoisation model) must still
    /// yield only ONE vertex, not two.
    #[test]
    fn action_atom_in_both_formulas_and_solved_formulas_dedups_to_one_vertex() {
        let mut sys = System::default();
        sys.content_mut().formulas.push(Arc::new(g("P(x) @ #i")));
        sys.content_mut()
            .solved_formulas
            .push(Arc::new(g("P(x) @ #i")));

        let part = extract_graph_part(&sys, &color_table(EMPTY));

        assert_eq!(
            action_fact_names(&part),
            vec![(nid("i", 0), "P".to_string())]
        );
    }

    /// An action atom that only occurs under a quantifier (its timepoint
    /// is a De-Bruijn `BVar::Bound` occurrence, not yet instantiated to
    /// any concrete `NodeId`) must NOT become a vertex — only the
    /// sibling ground conjunct does.
    #[test]
    fn quantified_action_atom_is_not_turned_into_a_vertex() {
        let mut sys = System::default();
        sys.content_mut()
            .formulas
            .push(Arc::new(g("P(z) @ #j & Ex x #i. Q(x) @ #i")));

        let part = extract_graph_part(&sys, &color_table(EMPTY));

        assert_eq!(
            action_fact_names(&part),
            vec![(nid("j", 0), "P".to_string())]
        );
    }

    #[test]
    fn escape_record_field_escapes_structural_and_quote_characters() {
        // A pretty-printed pair term looks like `<x, y>` — `<`/`>` are
        // record port syntax and must not reach the label unescaped.
        assert_eq!(escape_record_field("<x, y>"), "\\<x, y\\>");
        assert_eq!(escape_record_field("a|b"), "a\\|b");
        assert_eq!(escape_record_field("{a}"), "\\{a\\}");
        assert_eq!(escape_record_field("say \"hi\""), "say \\\"hi\\\"");
        assert_eq!(escape_record_field("back\\slash"), "back\\\\slash");
    }

    #[test]
    fn dot_fill_color_is_deterministic_and_a_well_formed_hex_string() {
        let c1 = dot_fill_color(1);
        let c2 = dot_fill_color(2);
        // Well-formed `#rrggbb`: `rgb_to_hex`'s own format.
        for c in [&c1, &c2] {
            assert_eq!(c.len(), 7);
            assert!(c.starts_with('#'));
            assert!(c[1..].chars().all(|ch| ch.is_ascii_hexdigit()));
        }
        // Deterministic: same table color -> same fill, every time.
        assert_eq!(c1, dot_fill_color(1));
        // Different table colors -> (in practice, for small inputs)
        // different fills -- the whole point of driving fill color from
        // the table instead of a per-`VertexKind` constant.
        assert_ne!(c1, c2);
    }

    #[test]
    fn escape_dot_label_escapes_only_quote_and_backslash() {
        assert_eq!(escape_dot_label("plain text"), "plain text");
        assert_eq!(escape_dot_label("say \"hi\""), "say \\\"hi\\\"");
        assert_eq!(escape_dot_label("back\\slash"), "back\\\\slash");
        // Record-structural characters are NOT special outside a record
        // label, so they pass through untouched here.
        assert_eq!(escape_dot_label("<a | b>"), "<a | b>");
    }

    #[test]
    fn to_graphviz_renders_a_well_formed_digraph_document() {
        // A has a conclusion, B a premise -- matching the manual `Edge`
        // below (A's ConcIdx(0) -> B's PremIdx(0)). The `EdgeRelation`
        // sub-block's bounds now come entirely from what `color_table`'s
        // rules/cache actually declare (no more artificial widening from
        // hardcoded fixed-rule premise counts), so a theory template
        // with FEWER ports than an edge references would make
        // `edge_relation_color` panic as out-of-range.
        const RULES_A_AND_B: &str = "theory T begin\n\
            rule A:\n  [] --> [ M() ]\n\
            rule B:\n  [ M() ] --> []\n\
            end";
        let mut sys = System::default();
        sys.add_node(nid("i", 1), proto_rule("A"));
        sys.add_node(nid("i", 2), proto_rule("B"));
        sys.content_mut().edges.push(Edge {
            src: (nid("i", 1), ConcIdx(0)),
            tgt: (nid("i", 2), PremIdx(0)),
        });
        let part = extract_graph_part(&sys, &color_table(RULES_A_AND_B));

        let dot = to_graphviz(&part);

        assert!(dot.starts_with("digraph G {\n"));
        assert!(dot.trim_end().ends_with('}'));
        assert!(dot.contains("n0 [shape=record"));
        assert!(dot.contains("n1 [shape=record"));
        assert!(dot.contains(": A"));
        assert!(dot.contains(": B"));
        // The EdgeRelation vertex (n2) sits between them: n0 -> n2 -> n1.
        assert!(dot.contains("n2 [shape=point"));
        assert!(dot.contains("n0 -> n2;"));
        assert!(dot.contains("n2 -> n1;"));
    }

    /// Pins the visual counterpart of
    /// `same_timepoint_actions_get_separate_vertices_not_merged`: two
    /// same-timepoint actions render as two SEPARATE ellipse nodes (never
    /// one merged node), each with its OWN `AtTimepointRelation` hexagon
    /// vertex back to their shared timepoint.
    #[test]
    fn same_timepoint_actions_render_as_two_ellipses_with_attimepoint_relations() {
        const RULE_WITH_P_AND_Q_ACTIONS: &str = "theory T begin\n\
            rule R:\n  [] --[ P(), Q() ]-> []\n\
            end";
        let mut sys = System::default();
        sys.content_mut()
            .formulas
            .push(Arc::new(g("P(x) @ #i & Q(y) @ #i")));
        let part = extract_graph_part(&sys, &color_table(RULE_WITH_P_AND_Q_ACTIONS));

        let dot = to_graphviz(&part);

        assert_eq!(dot.matches("shape=ellipse").count(), 2);
        assert_eq!(dot.matches("shape=hexagon").count(), 2);
        assert!(dot.contains(" P("));
        assert!(dot.contains(" Q("));
    }

    /// A `Dummy` vertex (no rule instance) renders as its own distinct
    /// shape, not silently as a record/ellipse.
    #[test]
    fn dummy_vertex_renders_as_a_dashed_diamond() {
        let mut sys = System::default();
        sys.content_mut().last_atom = Some(nid("i", 7));
        let part = extract_graph_part(&sys, &color_table(EMPTY));

        let dot = to_graphviz(&part);

        assert!(dot.contains("shape=diamond, style=\"filled,dashed\""));
        assert!(dot.contains("#i.7"));
    }

    /// A less-than atom renders its own `LessRelation` triangle vertex,
    /// distinct from `EdgeRelation`'s point and `AtTimepointRelation`'s
    /// hexagon.
    #[test]
    fn less_relation_renders_as_a_triangle() {
        let mut sys = System::default();
        sys.content_mut()
            .less_atoms
            .push(LessAtom::new(nid("i", 1), nid("i", 2), Reason::Formula));
        let part = extract_graph_part(&sys, &color_table(EMPTY));

        let dot = to_graphviz(&part);

        assert_eq!(dot.matches("shape=triangle").count(), 1);
        assert_eq!(dot.matches("shape=point").count(), 0);
        assert_eq!(dot.matches("shape=hexagon").count(), 0);
    }
}
