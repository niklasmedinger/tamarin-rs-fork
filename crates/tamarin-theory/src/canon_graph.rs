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

use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::constraint::constraints::NodeId;
use crate::constraint::system::System;
use crate::guarded::{cmp_fact, BVar, GAtom, GFact, GTerm, Guarded};
use crate::pretty_formula::pretty_guarded;
use crate::pretty_system::pretty_fact;
use crate::rule::{rule_name_string, RuleACInst};

use tamarin_parser::ast::{SortHint, SuffixSort, VarSpec};
use tamarin_term::lterm::{LSort, LVar};

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
    Action(NodeId, GFact),
    /// A `NodeId` with neither a rule instance nor an action atom of its
    /// own, referenced only via a relation endpoint or `last_atom`.
    /// Mirrors `graph::repr::NodeType::Missing`'s reason for existing:
    /// every `NodeId` the system references anywhere must resolve to
    /// SOME vertex, or a later stage's canonical-labelling lookup (which
    /// is exhaustive-or-panic, per `canon.rs`'s `canonicalize_guarded`)
    /// panics on a missing entry.
    Dummy(NodeId),
    /// Reifies one `System::edges` conclusion→premise connection:
    /// `src -> EdgeRelation -> tgt` replaces a directly typed
    /// `src -> tgt` edge. See the module docs' third bullet. Carries no
    /// payload — a bare marker; port indices (`ConcIdx`/`PremIdx`) are
    /// not tracked here, matching this module's existing node-level
    /// (not port-level) granularity for edges.
    EdgeRelation,
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

/// The extracted graph part `(V, E)` — the coloring `c` is a later,
/// separate stage (TODO.md's skeleton scheme).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GraphPart {
    pub vertices: Vec<VertexKind>,
    pub edges: Vec<GraphEdge>,
}

/// Extracts the graph part from `sys` (Stage A — see the module docs).
pub fn extract_graph_part(sys: &System) -> GraphPart {
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
        push_relation(&mut vertices, &mut edges, VertexKind::EdgeRelation, s, t);
    }

    // 3. Less-than atoms, reified as `smaller -> LessRelation -> larger`.
    for la in sys.less_atoms_in_set_order() {
        let s = get_vertex_or_create_dummy_vertex(la.smaller, &mut vertices, &mut node_vertex);
        let t = get_vertex_or_create_dummy_vertex(la.larger, &mut vertices, &mut node_vertex);
        push_relation(&mut vertices, &mut edges, VertexKind::LessRelation, s, t);
    }

    // 4. `last_atom` — a bare reference needs a vertex even when nothing
    //    else in the system points at it.
    if let Some(la) = sys.last_atom {
        get_vertex_or_create_dummy_vertex(la, &mut vertices, &mut node_vertex);
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
    let mut actions: Vec<(NodeId, GFact)> = Vec::new();
    for f in sys.formulas.iter().chain(sys.solved_formulas.iter()) {
        collect_action_atoms(f, &mut actions);
    }
    actions.sort_by(|(n1, f1), (n2, f2)| n1.cmp(n2).then_with(|| cmp_fact(f1, f2)));
    actions
        .dedup_by(|(n1, f1), (n2, f2)| n1 == n2 && cmp_fact(f1, f2) == std::cmp::Ordering::Equal);

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

    GraphPart { vertices, edges }
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
//   - [`VertexKind::EdgeRelation`]: a tiny filled `point` — the closest
//     DOT has to "just a pass-through connector, no content of its own".
//   - [`VertexKind::LessRelation`]: a small `triangle` — evokes an
//     ordering/comparison ("<").
//   - [`VertexKind::AtTimepointRelation`]: a small `hexagon` — visually
//     distinct from every other shape used here, matching that it is
//     this module's own addition with no HS/work.tex counterpart.

/// Renders `part` as a self-contained `digraph G { ... }` DOT document.
pub fn to_graphviz(part: &GraphPart) -> String {
    let mut out = String::new();
    out.push_str("digraph G {\n");
    out.push_str("  rankdir=TB;\n");
    out.push_str("  node [fontname=\"Helvetica\", fontsize=10];\n");
    out.push_str("  edge [fontname=\"Helvetica\", fontsize=10];\n\n");

    for (idx, v) in part.vertices.iter().enumerate() {
        write_vertex(&mut out, idx, v);
    }
    out.push('\n');
    for e in &part.edges {
        write_edge(&mut out, e);
    }

    out.push_str("}\n");
    out
}

fn write_vertex(out: &mut String, idx: usize, v: &VertexKind) {
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
                "  n{idx} [shape=record, style=filled, fillcolor=\"#d6e8ff\", label=\"{label}\"];"
            )
            .ok();
        }
        VertexKind::Action(nid, fact) => {
            let fact_str = pretty_guarded(&Guarded::Atom(GAtom::Pred(fact.clone())));
            let label = escape_dot_label(&format!("V{idx}  {fact_str} @ {nid}"));
            writeln!(
                out,
                "  n{idx} [shape=ellipse, style=filled, fillcolor=\"#ffe4a3\", label=\"{label}\"];"
            )
            .ok();
        }
        VertexKind::Dummy(nid) => {
            let label = escape_dot_label(&format!("V{idx}  {nid}"));
            writeln!(
                out,
                "  n{idx} [shape=diamond, style=dashed, label=\"{label}\"];"
            )
            .ok();
        }
        VertexKind::EdgeRelation => {
            writeln!(
                out,
                "  n{idx} [shape=point, width=0.40, style=filled, fillcolor=black, label=\"\"];"
            )
            .ok();
        }
        VertexKind::LessRelation => {
            let label = escape_dot_label("<");
            writeln!(
                out,
                "  n{idx} [shape=triangle, width=0.25, height=0.2, style=filled, fillcolor=\"#c9a0ff\", label=\"{label}\"];"
            )
            .ok();
        }
        VertexKind::AtTimepointRelation => {
            let label = escape_dot_label(&format!("V{idx}"));
            writeln!(
                out,
                "  n{idx} [shape=hexagon, width=0.25, height=0.2, style=filled, fillcolor=\"#b0e0b0\", label=\"{label}\"];"
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
    use std::sync::Arc;
    use tamarin_parser::parser::parse_formula_str;

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

    fn action_fact_names(part: &GraphPart) -> Vec<(NodeId, String)> {
        let mut out: Vec<(NodeId, String)> = part
            .vertices
            .iter()
            .filter_map(|v| match v {
                VertexKind::Action(nid, fact) => Some((*nid, fact.name.clone())),
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

        let part = extract_graph_part(&sys);

        // 2 rule instances + 1 reified EdgeRelation vertex.
        assert_eq!(part.vertices.len(), 3);
        assert!(matches!(&part.vertices[0], VertexKind::RuleInstance(n, _) if *n == nid("i", 1)));
        assert!(matches!(&part.vertices[1], VertexKind::RuleInstance(n, _) if *n == nid("i", 2)));
        assert_relation(&part, 0, &VertexKind::EdgeRelation, 1);
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

        let part = extract_graph_part(&sys);

        // 1 rule instance + 1 dummy + 1 reified EdgeRelation vertex.
        assert_eq!(part.vertices.len(), 3);
        assert!(matches!(&part.vertices[1], VertexKind::Dummy(n) if *n == nid("i", 2)));
        assert_relation(&part, 0, &VertexKind::EdgeRelation, 1);
        assert_eq!(part.edges.len(), 2);
    }

    #[test]
    fn less_atom_endpoints_get_dummy_vertices_and_a_less_relation() {
        let mut sys = System::default();
        sys.content_mut()
            .less_atoms
            .push(LessAtom::new(nid("i", 1), nid("i", 2), Reason::Formula));

        let part = extract_graph_part(&sys);

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

        let part = extract_graph_part(&sys);

        assert_eq!(part.vertices, vec![VertexKind::Dummy(nid("i", 7))]);
        assert!(part.edges.is_empty());
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

        let part = extract_graph_part(&sys);

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

        let part = extract_graph_part(&sys);

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

        let part = extract_graph_part(&sys);

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
        let mut sys = System::default();
        sys.add_node(nid("i", 1), proto_rule("A"));
        sys.add_node(nid("i", 2), proto_rule("B"));
        sys.content_mut().edges.push(Edge {
            src: (nid("i", 1), ConcIdx(0)),
            tgt: (nid("i", 2), PremIdx(0)),
        });
        let part = extract_graph_part(&sys);

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
        let mut sys = System::default();
        sys.content_mut()
            .formulas
            .push(Arc::new(g("P(x) @ #i & Q(y) @ #i")));
        let part = extract_graph_part(&sys);

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
        let part = extract_graph_part(&sys);

        let dot = to_graphviz(&part);

        assert!(dot.contains("shape=diamond, style=dashed"));
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
        let part = extract_graph_part(&sys);

        let dot = to_graphviz(&part);

        assert_eq!(dot.matches("shape=triangle").count(), 1);
        assert_eq!(dot.matches("shape=point").count(), 0);
        assert_eq!(dot.matches("shape=hexagon").count(), 0);
    }
}
