// Currently GPL 3.0 until granted permission by the upstream authors
// of the tamarin-prover sources this file cites; list them with:
//   scripts/gen_license_headers.py --authors <this file>

//! (No HS analog.) A subprocess driver for `bliss` — graph canonical
//! labeling / automorphism-group computation — the external tool
//! `TODO.md`'s "Canonizing the graph" section names as the intended
//! Stage C/F graph canonizer.
//!
//! Mirrors `tamarin_term::maude_proc`'s pattern of spawning and driving
//! an external tool over stdin/stdout rather than binding via FFI, but
//! MUCH simpler: bliss is a one-shot batch tool (write the whole input
//! graph, close stdin, read all of stdout until the process exits), not
//! a persistent line-oriented REPL, so there is no protocol/prompt
//! parsing and no need for a process pool — every call spawns a fresh
//! `bliss` process and waits for it to finish. `TAM_ALLOW_NO_BLISS=1`
//! mirrors `TAM_ALLOW_NO_MAUDE`'s escape hatch (see
//! [`bliss_available`]).
//!
//! ## Input format
//!
//! bliss's own DIMACS-like graph format, verified directly against
//! bliss 0.77's own source (`src/digraph.cc`'s `Digraph::read_dimacs`)
//! — see [`graph_part_to_dimacs`]:
//!
//! ```text
//! p edge <N> <M>
//! n <vertex 1-indexed> <color>        (one line per vertex)
//! e <src 1-indexed> <tgt 1-indexed>   (exactly M lines)
//! ```
//!
//! ## Output format
//!
//! With `-directed -can -v=0` (verified both against bliss 0.77's
//! source, `src/bliss.cc`/`src/utils.cc`'s `print_permutation`, AND
//! empirically against hand-built graphs — see this module's tests):
//! zero or more automorphism-GENERATOR lines, each followed by exactly
//! one canonical-labeling line:
//!
//! ```text
//! Generator: (1,2,3)(4,5)
//! Canonical labeling: (1,3)
//! ```
//!
//! Both are permutations of the 1-indexed vertex numbers in CYCLE
//! notation: `(a,b,c)` means `a -> b -> c -> a`. A vertex that maps to
//! itself is a FIXED POINT and is omitted from every cycle entirely
//! (not printed as a length-1 cycle `(a)`); an entirely-identity
//! permutation prints as the literal empty `()`. Reconstructing the
//! FULL permutation therefore requires knowing the vertex count `N` up
//! front and defaulting every unmentioned vertex to itself — see
//! [`Permutation::from_cycle_notation`].
//!
//! ## Caveats investigated before writing any of this
//!
//! - **Directed, not undirected.** bliss defaults to undirected graphs;
//!   `GraphPart`'s edges are meaningfully directed (e.g. a
//!   `LessRelation`'s `smaller -> relation -> larger` orientation is
//!   semantic content, not incidental), so every call here passes
//!   `-directed` and uses `Digraph::read_dimacs`'s format, not
//!   `Graph::read_dimacs`'s.
//! - **Colors are an equivalence signal, not a magnitude.** bliss's
//!   partition refinement only cares which vertices share a color, not
//!   the numeric VALUE — so directly using `ColorTable`'s `u32` values
//!   (see `graph_part_to_dimacs`) is exactly right; nothing needs
//!   normalizing/remapping into a smaller range first.
//! - **`-v=0` is required, not optional**: bliss's default verbosity is
//!   1 (`static unsigned int verbose_level = 1;`, `src/bliss.cc`), which
//!   interleaves human-readable statistics into stdout — without
//!   `-v=0` the output would not even be reliably line-parseable as
//!   just `Generator:`/`Canonical labeling:` lines.
//! - **Duplicate edges / self-loops cannot arise from a `GraphPart`.**
//!   Every edge in a `GraphPart` has at least one endpoint that is a
//!   freshly-allocated relation vertex (`canon_graph::push_relation`
//!   never reuses or merges a relation vertex across calls), so two
//!   identical `(src, tgt)` pairs, or `src == tgt`, cannot occur even
//!   for a degenerate input (e.g. a `less_atoms` entry relating a node
//!   to itself still routes through its own fresh `LessRelation`
//!   vertex, never a direct self-loop). No dedup/self-loop handling is
//!   implemented here because the input this module is actually fed
//!   never needs it.
//! - **Empty graphs are rejected by bliss** (`Digraph::read_dimacs`
//!   errors on `nof_vertices <= 0`) — [`graph_part_to_dimacs`]/
//!   [`canonicalize`] return a clear [`BlissError::EmptyGraph`] instead
//!   of letting bliss fail with a cryptic parse error.
//! - **Vertex numbering does not affect correctness, only tie-breaking
//!   among automorphisms** — the concern this task's own author flagged
//!   up front. Confirmed directly: two hand-built graphs, isomorphic via
//!   a NON-identity vertex renumbering, produce canonical labelings
//!   that — once each is applied to its own graph — yield byte-for-byte
//!   the same relabeled colors/edges (see this module's
//!   `differently_numbered_isomorphic_graphs_agree_after_canonicalizing`
//!   test, which mirrors the exact manual experiment run against the
//!   real `bliss` binary while designing this module). Vertex order
//!   only matters for WHICH labeling bliss picks among several
//!   automorphism-equivalent choices when the automorphism group is
//!   non-trivial — exactly the "minimum over automorphisms" question
//!   this stage deliberately leaves open (see [`CanonicalGraph`]'s own
//!   doc comment).

use std::collections::BTreeSet;
use std::fmt;
use std::fmt::Write as _;
use std::io::Write as _;
use std::process::{Command, Stdio};

use crate::canon_color::Color;
use crate::canon_graph::{GraphPart, VertexKind};

const ALLOW_NO_BLISS_ENV: &str = "TAM_ALLOW_NO_BLISS";

/// Errors that can arise from the bliss bridge.
#[derive(Debug)]
pub enum BlissError {
    /// stdin/stdout I/O failure talking to the bliss subprocess.
    Io(std::io::Error),
    /// The `bliss` binary could not be launched (e.g. not found on PATH).
    Spawn(String),
    /// bliss's output could not be parsed as expected.
    Parse(String),
    /// bliss exited with a non-zero status.
    NonZeroExit { status: String, stderr: String },
    /// The graph has no vertices — bliss's own `read_dimacs` rejects
    /// this (`nof_vertices <= 0`), so this is caught here first with a
    /// clearer message.
    EmptyGraph,
}

impl fmt::Display for BlissError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BlissError::Io(e) => write!(f, "io error: {e}"),
            BlissError::Spawn(s) => write!(f, "spawn error: {s}"),
            BlissError::Parse(s) => write!(f, "parse error: {s}"),
            BlissError::NonZeroExit { status, stderr } => {
                write!(f, "bliss exited with {status}: {stderr}")
            }
            BlissError::EmptyGraph => write!(f, "cannot canonicalize an empty graph (0 vertices)"),
        }
    }
}

impl std::error::Error for BlissError {}

/// The `bliss` binary to invoke: `$BLISS_PATH` if set, else bare
/// `bliss` (resolved via `$PATH`) — mirrors
/// `tamarin-prover/tests/common/mod.rs`'s `maude_path`/`MAUDE_PATH`
/// convention.
fn bliss_binary() -> String {
    std::env::var("BLISS_PATH").unwrap_or_else(|_| "bliss".to_string())
}

/// True when the `bliss` executable is runnable (`bliss -version` exits
/// successfully). PANICS (not a silent `false`/skip) when it is not,
/// unless `TAM_ALLOW_NO_BLISS=1` is set — mirrors
/// `tamarin-prover/tests/common/mod.rs`'s `maude_available()` test-gating
/// precedent: a machine where bliss is simply absent should fail loudly
/// when a bliss-backed test runs, not let every such test report a
/// vacuous green by silently skipping. Every bliss-backed test in this
/// crate calls this first.
pub fn bliss_available() -> bool {
    let bin = bliss_binary();
    let ok = Command::new(&bin)
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if ok {
        return true;
    }
    assert!(
        std::env::var(ALLOW_NO_BLISS_ENV).as_deref() == Ok("1"),
        "no working `bliss` executable found (tried `{bin} -version`) -- install \
         bliss (https://users.aalto.fi/~tjunttil/bliss/download.html) and put it \
         on PATH, or point BLISS_PATH at it. Set {ALLOW_NO_BLISS_ENV}=1 to skip \
         bliss-backed tests deliberately."
    );
    false
}

// =============================================================================
// Permutation
// =============================================================================

/// A permutation of `0..n` vertex indices (0-indexed internally, even
/// though bliss's own textual format is 1-indexed — the `+1`/`-1`
/// conversion is entirely contained in [`Self::from_cycle_notation`]/
/// [`graph_part_to_dimacs`]). `perm[i]` is where vertex `i` maps TO.
///
/// Used for both bliss's canonical labeling (`perm[old_vertex] =
/// new_vertex`) and each automorphism generator (`perm[v] = v'`, a
/// relabeling under which the graph maps to itself) — the same
/// underlying mathematical object serves both roles; only the caller's
/// interpretation differs.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Permutation(Vec<usize>);

impl Permutation {
    pub fn identity(n: usize) -> Self {
        Permutation((0..n).collect())
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Where vertex `v` maps to.
    pub fn image_of(&self, v: usize) -> usize {
        self.0[v]
    }

    pub fn as_slice(&self) -> &[usize] {
        &self.0
    }

    /// `self ∘ other`: apply `other` first, then `self` — so
    /// `self.compose(other).image_of(v) == self.image_of(other.image_of(v))`.
    ///
    /// Used two ways: [`generate_group`]'s BFS closure composes
    /// candidate group elements with generators to discover new ones,
    /// and a caller searching "minimum over automorphisms" composes
    /// bliss's own `canonical_labeling` with each element of the closed
    /// `Aut(G)` (`labeling.compose(&g)`) to enumerate every OTHER
    /// canonical labeling bliss could equally validly have chosen (see
    /// the module docs' vertex-numbering caveat and `CanonicalGraph`'s
    /// own doc comment on "minimum over automorphisms").
    pub fn compose(&self, other: &Permutation) -> Permutation {
        Permutation(other.0.iter().map(|&v| self.image_of(v)).collect())
    }

    /// Parses bliss's 1-indexed cycle notation (e.g. `"(1,2,3)(4,5)"`,
    /// or `"()"` for the identity), given the total vertex count `n`
    /// (needed because fixed points are omitted from the text — see the
    /// module docs).
    fn from_cycle_notation(s: &str, n: usize) -> Result<Self, BlissError> {
        let mut perm: Vec<usize> = (0..n).collect();
        let s = s.trim();
        if s.is_empty() || s == "()" {
            return Ok(Permutation(perm));
        }
        for cycle in s.split(')') {
            let cycle = cycle.trim();
            if cycle.is_empty() {
                continue;
            }
            let cycle = cycle.strip_prefix('(').ok_or_else(|| {
                BlissError::Parse(format!("expected '(' to start a cycle in {s:?}"))
            })?;
            let elems: Vec<usize> = cycle
                .split(',')
                .map(|tok| {
                    tok.trim().parse::<usize>().map_err(|e| {
                        BlissError::Parse(format!("bad vertex number {tok:?} in {s:?}: {e}"))
                    })
                })
                .collect::<Result<_, _>>()?;
            if elems.is_empty() {
                continue;
            }
            for i in 0..elems.len() {
                let from = elems[i].checked_sub(1).ok_or_else(|| {
                    BlissError::Parse(format!("vertex 0 is invalid (bliss is 1-indexed): {s:?}"))
                })?;
                let to = elems[(i + 1) % elems.len()].checked_sub(1).ok_or_else(|| {
                    BlissError::Parse(format!("vertex 0 is invalid (bliss is 1-indexed): {s:?}"))
                })?;
                if from >= n || to >= n {
                    return Err(BlissError::Parse(format!(
                        "vertex out of range [1,{n}] in cycle notation: {s:?}"
                    )));
                }
                perm[from] = to;
            }
        }
        Ok(Permutation(perm))
    }
}

/// Closes `generators` (typically `BlissResult::generators` — a
/// GENERATING SET for `Aut(G)`, not the whole group) under composition,
/// via a standard BFS from the identity, returning every element of the
/// group they generate.
///
/// **This is necessary, not optional**: bliss reports generators, and a
/// generating set does not enumerate the whole group by itself. Concrete
/// counterexample: if `Aut(G) = ⟨g₁, g₂⟩` with `g₁` and `g₂` independent
/// transpositions (disjoint supports), the group has FOUR elements — the
/// identity, `g₁`, `g₂`, AND `g₁∘g₂` — but bliss only ever reports the
/// two generators. Iterating over just `{id, g₁, g₂}` (skipping the BFS
/// closure this function performs) can miss the actual minimum when
/// finding it requires applying both independent symmetries at once —
/// see `canon::tests::naive_generator_only_iteration_misses_the_true_minimum`
/// for a worked, empirically-verified example built exactly this way.
///
/// Standard group-closure algorithm: starting from `{id}`, repeatedly
/// compose every element discovered so far with every generator, adding
/// any newly-seen result, until nothing new appears. This reaches every
/// element of the generated (possibly non-abelian) group regardless of
/// the fixed composition order, since every group element is some
/// (finite, as the group itself is finite) product of generators.
pub fn generate_group(generators: &[Permutation], n: usize) -> Vec<Permutation> {
    let identity = Permutation::identity(n);
    let mut seen: BTreeSet<Permutation> = BTreeSet::from([identity.clone()]);
    let mut frontier: Vec<Permutation> = vec![identity];
    while let Some(p) = frontier.pop() {
        for g in generators {
            let q = p.compose(g);
            if seen.insert(q.clone()) {
                frontier.push(q);
            }
        }
    }
    seen.into_iter().collect()
}

// =============================================================================
// DIMACS writer
// =============================================================================

/// Renders `part`'s vertices/edges as bliss's DIMACS-like input format
/// (see the module docs), coloring each vertex via `part.colors` (the
/// `ColorTable` a `GraphPart` now always carries — see
/// `canon_graph::GraphPart`'s own doc comment).
///
/// Bliss numbers vertices from 1 — `GraphPart`'s own indices (0-based)
/// are shifted by `+1` here; nothing else about `GraphPart`'s vertex
/// order is significant to bliss, whose whole job is finding a
/// canonical order independent of the INPUT order (see the module
/// docs' caveat list).
pub fn graph_part_to_dimacs(part: &GraphPart) -> Result<String, BlissError> {
    if part.vertices.is_empty() {
        return Err(BlissError::EmptyGraph);
    }
    let mut out = String::new();
    writeln!(out, "p edge {} {}", part.vertices.len(), part.edges.len()).ok();
    for (idx, v) in part.vertices.iter().enumerate() {
        writeln!(out, "n {} {}", idx + 1, part.colors.vertex_color(v)).ok();
    }
    for e in &part.edges {
        writeln!(out, "e {} {}", e.src + 1, e.tgt + 1).ok();
    }
    Ok(out)
}

// =============================================================================
// Running bliss
// =============================================================================

/// The result of running bliss with `-directed -can` on a graph:
/// bliss's own canonical vertex numbering plus a generating set for the
/// graph's automorphism group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlissResult {
    pub canonical_labeling: Permutation,
    pub generators: Vec<Permutation>,
}

/// Runs `bliss -directed -can -v=0` once on `dimacs_input`, parsing its
/// stdout. Spawns a FRESH process every call — no pool, no persistent
/// process (unlike `maude_proc::MaudeHandle`): bliss is a one-shot batch
/// tool, not an interactive REPL, so there is no protocol state to keep
/// alive between calls, and the caller doesn't need one either.
///
/// The vertex count is recovered by scanning `dimacs_input` for its own
/// `p edge N M` line, rather than taking it as a separate parameter —
/// one less thing for a caller to keep in sync with the input it just
/// built.
pub fn run_bliss(dimacs_input: &str) -> Result<BlissResult, BlissError> {
    let n = parse_vertex_count(dimacs_input)?;
    let bin = bliss_binary();

    let mut child = Command::new(&bin)
        .args(["-directed", "-can", "-v=0"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| BlissError::Spawn(format!("{bin}: {e}")))?;

    {
        // Scoped so the `ChildStdin` is dropped (closing the pipe) before
        // `wait_with_output` blocks reading stdout -- bliss reads all of
        // stdin before writing anything, so writing everything up front
        // (no threaded stdin/stdout interleaving) cannot deadlock for the
        // graph sizes this module is actually fed (well under a pipe
        // buffer's worth of bytes).
        let stdin = child.stdin.as_mut().expect("piped stdin");
        stdin
            .write_all(dimacs_input.as_bytes())
            .map_err(BlissError::Io)?;
    }

    let output = child.wait_with_output().map_err(BlissError::Io)?;
    if !output.status.success() {
        return Err(BlissError::NonZeroExit {
            status: output.status.to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }

    parse_bliss_stdout(&String::from_utf8_lossy(&output.stdout), n)
}

fn parse_vertex_count(dimacs_input: &str) -> Result<usize, BlissError> {
    for line in dimacs_input.lines() {
        if let Some(rest) = line.strip_prefix("p edge ") {
            let n: usize = rest
                .split_whitespace()
                .next()
                .ok_or_else(|| BlissError::Parse(format!("malformed 'p edge' line: {line:?}")))?
                .parse()
                .map_err(|e| BlissError::Parse(format!("bad vertex count in {line:?}: {e}")))?;
            return Ok(n);
        }
    }
    Err(BlissError::Parse(
        "no 'p edge N M' line found in DIMACS input".to_string(),
    ))
}

fn parse_bliss_stdout(stdout: &str, n: usize) -> Result<BlissResult, BlissError> {
    let mut generators = Vec::new();
    let mut canonical_labeling = None;
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("Generator: ") {
            generators.push(Permutation::from_cycle_notation(rest, n)?);
        } else if let Some(rest) = line.strip_prefix("Canonical labeling: ") {
            canonical_labeling = Some(Permutation::from_cycle_notation(rest, n)?);
        } else if !line.trim().is_empty() {
            return Err(BlissError::Parse(format!(
                "unexpected bliss output line: {line:?}"
            )));
        }
    }
    let canonical_labeling = canonical_labeling.ok_or_else(|| {
        BlissError::Parse(format!(
            "no 'Canonical labeling:' line in bliss output -- was -can passed? \
             full stdout: {stdout:?}"
        ))
    })?;
    Ok(BlissResult {
        canonical_labeling,
        generators,
    })
}

// =============================================================================
// Applying the canonical labeling
// =============================================================================

/// A graph, relabeled by a canonical labeling: vertex colors in
/// canonical-index order, plus the edge set (also in canonical
/// indices, deduplicated/ordered via `BTreeSet` so two structurally
/// identical relabelings compare `Eq` regardless of source iteration
/// order).
///
/// Two graphs that are isomorphic (as colored digraphs) produce
/// IDENTICAL `CanonicalGraph` values when each is relabeled by its OWN
/// canonical labeling from bliss — confirmed directly against the real
/// `bliss` binary (see the module docs' caveat list and
/// `differently_numbered_isomorphic_graphs_agree_after_canonicalizing`).
///
/// **This is a cheap, coloring-level DIAGNOSTIC, not a sufficient
/// $\alphaeqac$ check, and not part of the production canonical form.**
/// Equality here is NECESSARY but not SUFFICIENT: it only says the two
/// graphs have the same colored-digraph shape, which does not imply
/// their vertices canonize to the same content (a `ColorTable` color is
/// deliberately coarser than full term content — see `canon_color`'s own
/// soundness note). Two vertices sharing a color could still hold
/// genuinely different, non-$\alphaeqac$ term content. The actual
/// production canonical form is `canon::canonicalize_graph_part`, which
/// canonizes full vertex content plus edges as ONE term — this struct
/// remains useful only as an isolated check of the coloring+bliss layer
/// on its own, independent of the (separate, more involved) content
/// canonization machinery, which is why it's still exercised directly in
/// this module's own tests and in `bliss_tutorial_alphaeqac.rs`.
///
/// **This does NOT implement "minimum over automorphisms"** (`TODO.md`'s
/// question; the canonization plan's Stage F) **and never will** — that's
/// deliberately out of scope for `CanonicalGraph`/`canonicalize`, which
/// exist only as a coloring+bliss-layer diagnostic (see above): when the
/// graph has a non-trivial automorphism group, bliss's own internal
/// tie-break among orbit-equivalent labelings is used AS IS here, not
/// searched over for a lexicographically-smallest result, so this is
/// correct as a canonical form only when the graph has NO non-trivial
/// automorphisms. The REAL "minimum over automorphisms" resolution lives
/// in `canon::minimal_graph_part_labelings` (Stage F, implemented), which
/// searches `generate_group(&result.generators, ..)`'s full closure —
/// `BlissResult::generators` is exactly what it searches over.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalGraph {
    pub vertex_colors: Vec<Color>,
    pub edges: BTreeSet<(usize, usize)>,
}

/// Relabels `part`'s vertices/edges according to `labeling` (typically
/// `BlissResult::canonical_labeling`), coloring each ORIGINAL vertex via
/// `part.colors` before relabeling.
pub fn apply_labeling(part: &GraphPart, labeling: &Permutation) -> CanonicalGraph {
    let n = part.vertices.len();
    let mut vertex_colors = vec![0; n];
    for (old_idx, v) in part.vertices.iter().enumerate() {
        vertex_colors[labeling.image_of(old_idx)] = part.colors.vertex_color(v);
    }
    CanonicalGraph {
        vertex_colors,
        edges: canonical_edges(part, labeling),
    }
}

/// `part`'s edges, remapped to canonical positions per `labeling` —
/// factored out of [`apply_labeling`] so a caller that wants the FULL
/// vertex content (via `canonical_vertex_order`, unlike
/// [`CanonicalGraph`], which only keeps colors) can still get the
/// matching canonical edge set without needing colors at all. This is
/// exactly what `canon::canonicalize_graph_part` needs alongside
/// `canonical_vertex_order`'s output.
pub fn canonical_edges(part: &GraphPart, labeling: &Permutation) -> BTreeSet<(usize, usize)> {
    part.edges
        .iter()
        .map(|e| (labeling.image_of(e.src), labeling.image_of(e.tgt)))
        .collect()
}

/// Runs bliss on `part`'s graph part (colored via `part.colors`) and
/// applies the resulting canonical labeling, in one call — the
/// composition [`graph_part_to_dimacs`] + [`run_bliss`] +
/// [`apply_labeling`] most callers actually want.
pub fn canonicalize(part: &GraphPart) -> Result<CanonicalGraph, BlissError> {
    let dimacs = graph_part_to_dimacs(part)?;
    let result = run_bliss(&dimacs)?;
    Ok(apply_labeling(part, &result.canonical_labeling))
}

/// Reorders `part`'s vertices into canonical position order, per
/// `labeling` (typically `BlissResult::canonical_labeling`) — the
/// complementary operation to [`apply_labeling`]/[`CanonicalGraph`], for
/// a caller that wants to canonize the vertices' own CONTENT rather than
/// just check the graph's shape.
///
/// Unlike [`apply_labeling`], this keeps each vertex's FULL `VertexKind`
/// payload (the `RuleACInst`/`GFact`/`NodeId` it carries) instead of
/// collapsing it to a bare `Color` — `CanonicalGraph` deliberately throws
/// that content away to stay a pure, naming-insensitive shape digest
/// (see its own doc comment): two $\alphaeqac$ systems have genuinely
/// different variable names in their `RuleACInst`s, so keeping raw
/// content in something meant for a `==` shape check would make that
/// check fail for the wrong reason. This function is for the opposite
/// need — `tamarin_theory::canon::canonicalize_vertex_sequence`, which
/// canonizes that content, is exactly the place a naming difference
/// SHOULD be resolved (by renaming to a shared canonical literal), not
/// hidden by throwing the content away first.
pub fn canonical_vertex_order<'a>(part: &'a GraphPart, labeling: &Permutation) -> Vec<&'a VertexKind> {
    let mut ordered: Vec<Option<&'a VertexKind>> = vec![None; part.vertices.len()];
    for (old_idx, v) in part.vertices.iter().enumerate() {
        ordered[labeling.image_of(old_idx)] = Some(v);
    }
    ordered
        .into_iter()
        .map(|slot| slot.expect("labeling is a bijection over part.vertices -- every slot filled"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permutation_parses_empty_cycle_notation_as_identity() {
        let p = Permutation::from_cycle_notation("()", 4).unwrap();
        assert_eq!(p.as_slice(), &[0, 1, 2, 3]);
    }

    #[test]
    fn permutation_parses_a_single_cycle() {
        // (1,2,3) on 3 vertices: 0-indexed 0->1->2->0.
        let p = Permutation::from_cycle_notation("(1,2,3)", 3).unwrap();
        assert_eq!(p.as_slice(), &[1, 2, 0]);
    }

    #[test]
    fn permutation_parses_multiple_disjoint_cycles_and_keeps_fixed_points() {
        // (1,3)(4,5) on 5 vertices: 0<->2 swap, 3<->4 swap, 1 fixed.
        let p = Permutation::from_cycle_notation("(1,3)(4,5)", 5).unwrap();
        assert_eq!(p.as_slice(), &[2, 1, 0, 4, 3]);
    }

    #[test]
    fn permutation_rejects_out_of_range_vertices() {
        let err = Permutation::from_cycle_notation("(1,9)", 3).unwrap_err();
        assert!(matches!(err, BlissError::Parse(_)));
    }

    #[test]
    fn dimacs_writer_rejects_an_empty_graph_part() {
        let part = GraphPart::default();
        let err = graph_part_to_dimacs(&part).unwrap_err();
        assert!(matches!(err, BlissError::EmptyGraph));
    }

    /// Unlike `apply_labeling`, `canonical_vertex_order` must keep each
    /// vertex's FULL content (here, distinct `NodeId`s), just reordered
    /// -- not collapse it to a bare color the way `CanonicalGraph` does.
    #[test]
    fn canonical_vertex_order_reorders_full_vertex_content() {
        use tamarin_term::lterm::{LSort, LVar};

        let part = GraphPart {
            vertices: vec![
                VertexKind::Dummy(LVar::new("i", LSort::Node, 0)),
                VertexKind::Dummy(LVar::new("i", LSort::Node, 1)),
                VertexKind::Dummy(LVar::new("i", LSort::Node, 2)),
            ],
            ..Default::default()
        };

        // 3-cycle: 0->2, 1->0, 2->1 (0-indexed).
        let labeling = Permutation::from_cycle_notation("(1,3,2)", 3).unwrap();
        assert_eq!(labeling.image_of(0), 2);
        assert_eq!(labeling.image_of(1), 0);
        assert_eq!(labeling.image_of(2), 1);

        let ordered = canonical_vertex_order(&part, &labeling);

        // Canonical position `labeling.image_of(old_idx)` must hold
        // exactly the vertex that was originally at `old_idx`.
        assert_eq!(*ordered[2], part.vertices[0]);
        assert_eq!(*ordered[0], part.vertices[1]);
        assert_eq!(*ordered[1], part.vertices[2]);
    }

    // ---------------------------------------------------------------
    // Live bliss tests -- gated on `bliss_available()`.
    // ---------------------------------------------------------------

    #[test]
    fn bliss_reports_version() {
        // `bliss_available()` already asserted this succeeds (or skips
        // cleanly under `TAM_ALLOW_NO_BLISS=1`); nothing further to
        // check here beyond "didn't panic".
        bliss_available();
    }

    #[test]
    fn triangle_with_uniform_color_has_a_nontrivial_automorphism() {
        if !bliss_available() {
            return;
        }
        let dimacs = "p edge 3 3\nn 1 0\nn 2 0\nn 3 0\ne 1 2\ne 2 3\ne 3 1\n";
        let result = run_bliss(dimacs).expect("run_bliss");
        assert!(
            !result.generators.is_empty(),
            "a uniformly-colored directed triangle has rotational automorphisms"
        );
    }

    /// The manual experiment this module's design was validated against
    /// (see the module docs): two hand-built, differently-numbered but
    /// isomorphic colored digraphs must canonicalize to the SAME
    /// `CanonicalGraph`.
    #[test]
    fn differently_numbered_isomorphic_graphs_agree_after_canonicalizing() {
        if !bliss_available() {
            return;
        }
        // Graph A: 1(color5) -> 2(color7) -> 3(color9).
        let a = "p edge 3 2\nn 1 5\nn 2 7\nn 3 9\ne 1 2\ne 2 3\n";
        // Graph B: same graph, relabeled (v1=color7 mid, v2=color9 sink,
        // v3=color5 src; edges src->mid->sink = 3->1, 1->2).
        let b = "p edge 3 2\nn 1 7\nn 2 9\nn 3 5\ne 3 1\ne 1 2\n";

        let ra = run_bliss(a).expect("run_bliss a");
        let rb = run_bliss(b).expect("run_bliss b");
        assert!(ra.generators.is_empty());
        assert!(rb.generators.is_empty());

        let colors_a = [5u32, 7, 9];
        let colors_b = [7u32, 9, 5];
        let edges_a = [(0usize, 1usize), (1, 2)];
        let edges_b = [(2usize, 0usize), (0, 1)];

        let relabel = |colors: &[u32], edges: &[(usize, usize)], labeling: &Permutation| {
            let mut vc = vec![0u32; colors.len()];
            for (i, c) in colors.iter().enumerate() {
                vc[labeling.image_of(i)] = *c;
            }
            let es: BTreeSet<(usize, usize)> = edges
                .iter()
                .map(|(s, t)| (labeling.image_of(*s), labeling.image_of(*t)))
                .collect();
            (vc, es)
        };

        let canon_a = relabel(&colors_a, &edges_a, &ra.canonical_labeling);
        let canon_b = relabel(&colors_b, &edges_b, &rb.canonical_labeling);
        assert_eq!(canon_a, canon_b);
    }

    // -----------------------------------------------------------------
    // generate_group -- pure permutation algebra, no bliss needed.
    // -----------------------------------------------------------------

    /// A single order-2 generator (a transposition) closes to exactly
    /// `{id, g}` -- the group is already closed by construction, so this
    /// mainly pins that `generate_group` doesn't do anything strange for
    /// the simplest possible non-trivial case.
    #[test]
    fn generate_group_closes_a_single_transposition() {
        let swap_2_3 = Permutation::from_cycle_notation("(3,4)", 4).unwrap();
        let mut group = generate_group(std::slice::from_ref(&swap_2_3), 4);
        group.sort();
        let mut expected = vec![Permutation::identity(4), swap_2_3];
        expected.sort();
        assert_eq!(group, expected);
    }

    /// THE case `generate_group` exists for: two INDEPENDENT
    /// (disjoint-support) transpositions generate a 4-element group --
    /// `{id, g1, g2, g1∘g2}` -- not just the 2 generators plus identity.
    /// A caller iterating over only `{id, g1, g2}` would never see
    /// `g1∘g2` at all.
    #[test]
    fn generate_group_of_two_independent_transpositions_has_four_elements() {
        let g1 = Permutation::from_cycle_notation("(1,2)", 6).unwrap(); // swaps 0,1
        let g2 = Permutation::from_cycle_notation("(3,4)", 6).unwrap(); // swaps 2,3 (disjoint from g1)
        let group = generate_group(&[g1.clone(), g2.clone()], 6);
        assert_eq!(
            group.len(),
            4,
            "Z2 x Z2 (two independent transpositions) has exactly 4 elements"
        );

        let id = Permutation::identity(6);
        let g1_g2 = id.compose(&g1).compose(&g2); // apply g1 then g2 (order doesn't matter, disjoint)
        let mut expected = vec![id, g1, g2, g1_g2];
        expected.sort();
        let mut got = group;
        got.sort();
        assert_eq!(got, expected);
    }

    // -----------------------------------------------------------------
    // Real bliss run against bliss's OWN documented automorphism example
    // (https://users.aalto.fi/~tjunttil/bliss/definitions.html, also
    // work.tex's $G_1$ example, \Cref{ex:graph_iso}): 4 vertices, vertex
    // 1 colored distinctly from 2/3/4, edges {1,2},{1,3},{1,4},{2,3},{2,4}
    // (encoded here as a symmetric pair of directed edges per undirected
    // edge, since `bliss_proc` always runs bliss with `-directed`).
    // Documented automorphism group: `{id, (3 4)}`, generated by the
    // single transposition swapping vertices 3 and 4.
    // -----------------------------------------------------------------

    #[test]
    fn bliss_g1_example_has_the_documented_automorphism_group() {
        if !bliss_available() {
            return;
        }
        let dimacs = "p edge 4 10\n\
                      n 1 1\n\
                      n 2 0\n\
                      n 3 0\n\
                      n 4 0\n\
                      e 1 2\ne 2 1\n\
                      e 1 3\ne 3 1\n\
                      e 1 4\ne 4 1\n\
                      e 2 3\ne 3 2\n\
                      e 2 4\ne 4 2\n";
        let result = run_bliss(dimacs).expect("run_bliss");

        assert_eq!(
            result.generators.len(),
            1,
            "bliss's own documented generating set for this graph is the single \
             transposition (3 4)"
        );

        let group = generate_group(&result.generators, 4);
        let mut got = group;
        got.sort();
        let mut expected = vec![
            Permutation::identity(4),
            Permutation::from_cycle_notation("(3,4)", 4).unwrap(),
        ];
        expected.sort();
        assert_eq!(
            got, expected,
            "Aut(G_1) is documented as exactly {{id, (3 4)}} -- two elements"
        );
    }
}
