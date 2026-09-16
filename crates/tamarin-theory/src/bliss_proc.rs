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
use crate::canon_graph::GraphPart;

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
#[derive(Debug, Clone, PartialEq, Eq)]
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
/// **This does NOT (yet) implement "minimum over automorphisms"**
/// (`TODO.md`'s still-open question; the canonization plan's Stage F):
/// when the graph has a non-trivial automorphism group, bliss's own
/// internal tie-break among orbit-equivalent labelings is used AS IS,
/// not searched over for a lexicographically-smallest result. Correct
/// as a canonical form only when the graph has NO non-trivial
/// automorphisms — deliberately out of scope for this stage; see
/// `BlissResult::generators`, which already carries what a future Stage
/// F would need to search over.
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
    let edges = part
        .edges
        .iter()
        .map(|e| (labeling.image_of(e.src), labeling.image_of(e.tgt)))
        .collect();
    CanonicalGraph {
        vertex_colors,
        edges,
    }
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
}
