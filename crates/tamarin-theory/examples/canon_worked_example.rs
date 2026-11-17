// Currently GPL 3.0 until granted permission by the upstream authors
// of the tamarin-prover sources this file cites; list them with:
//   scripts/gen_license_headers.py --authors <this file>

// Example/dev tool: prints its report to stdout by design; allow the
// `disallowed_macros` convention freeze for this example binary.
#![allow(clippy::disallowed_macros)]

//! The thesis's running example through the canonizer, step by step:
//!
//! ```text
//! Γ = {A(x ⊕ y)@i, B(x)@j}     Δ = {B(v)@k, A(u ⊕ v)@l}     Γ' = {A(x ⊕ y)@i}
//! ```
//!
//! Each atom is one entry of `System::formulas`. The example runs every
//! system at two levels, through the same [`run`]:
//!
//! - **Thesis level**: `T(Γ) = ∃_msg x̄. ∃_tmp ī. Sys(c1, …, cn)`, the term the
//!   chapter defines, with one typed binder per sort of `fv(Γ)`.
//!   [`thesis_term`] projects it out of the real system term, so it is not
//!   written by hand: the `Formulas` section plays `Sys` (both are AC),
//!   `GAction` plays `Act`, and the root binders are the system term's own.
//!   Only the chapter's `Fml` around each constraint is added ([`fml_op`]).
//! - **Implementation level**: the unchanged [`system_to_term`], with its
//!   thirteen sections and its typed binders. [`study`] also runs the
//!   library's [`canonicalize_term`] on it, and
//!   `tests/canon_worked_example.rs` checks that [`run`] agrees with it, which
//!   shows that the thesis level follows the code's procedure.
//!
//! The two levels have different graphs, so their labellings differ, and
//! either can name the variables differently.
//!
//! ```text
//! cargo run --example canon_worked_example -- [--out DIR]
//! ```
//!
//! prints the report; `--out` also writes it to `DIR` with the bliss inputs
//! (`*.dimacs`) and the LaTeX fragments (`fragments.tex`).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Arc;

use tamarin_term::alpha_eq_ac::{rename_var, CanonLabelling};
use tamarin_term::builtin::xor;
use tamarin_term::function_symbols::{AcSym, Constructability, FunSym, NoEqSym, Privacy};
use tamarin_term::lterm::{LNTerm, LSort, LVar};
use tamarin_term::vterm::var_term;
use tamarin_theory::atom::ProtoAtom;
use tamarin_theory::bliss_proc::{
    run_bliss_with_stats, to_dimacs, BlissError, BlissResult, BlissStats,
};
use tamarin_theory::canon::CanonicalSystem;
use tamarin_theory::canon_fingerprint::fingerprint_constraint_system;
use tamarin_theory::canon_term_graph::{
    canonicalize_term, fingerprint_sterm, normalize, rename, system_to_term, term_to_graph, Binder,
    ConstKey, Label, Op, STerm, Scope, Section, TermGraph, TermGraphOptions, TermGraphStats,
    VarRef, SYSTEM_SECTIONS,
};
use tamarin_theory::constraint::system::System;
use tamarin_theory::fact::{proto_fact, Multiplicity};
use tamarin_theory::guarded::{lift_free_atom, Guarded};
use tamarin_theory::pretty_formula::pretty_guarded;
use tamarin_utils::fingerprint::Fingerprint;

// =============================================================================
// The systems
// =============================================================================

pub fn msg(name: &str) -> LVar {
    LVar::new(name, LSort::Msg, 0)
}

pub fn node(name: &str) -> LVar {
    LVar::new(name, LSort::Node, 0)
}

fn v(x: LVar) -> LNTerm {
    var_term(x)
}

/// The formula constraint `name(t)@time`.
pub fn action(time: LVar, name: &str, t: LNTerm) -> Guarded {
    let fact = proto_fact(Multiplicity::Linear, name, vec![t]);
    Guarded::Atom(lift_free_atom(&ProtoAtom::Action(v(time), fact)))
}

/// `{A(x ⊕ y)@i, B(x)@j}`.
fn gamma(vars: &[LVar]) -> Vec<Guarded> {
    let &[x, y, i, j] = vars else {
        panic!("Γ has four variables")
    };
    vec![action(i, "A", xor(v(x), v(y))), action(j, "B", v(x))]
}

/// `{B(v)@k, A(u ⊕ v)@l}`.
fn delta(vars: &[LVar]) -> Vec<Guarded> {
    let &[u, w, k, l] = vars else {
        panic!("Δ has four variables")
    };
    vec![action(k, "B", v(w)), action(l, "A", xor(v(u), v(w)))]
}

/// `{A(x ⊕ y)@i}`.
fn gamma_prime(vars: &[LVar]) -> Vec<Guarded> {
    let &[x, y, i] = vars else {
        panic!("Γ' has three variables")
    };
    vec![action(i, "A", xor(v(x), v(y)))]
}

/// A system of formula constraints, in its written order.
pub fn system(formulas: &[Guarded]) -> System {
    let mut sys = System::empty();
    for f in formulas {
        sys.formulas_mut().push(Arc::new(f.clone()));
    }
    sys
}

/// One of the example's systems, over variables that can be replaced.
#[derive(Clone)]
pub struct Example {
    pub name: &'static str,
    /// The name of the system in LaTeX.
    pub tex: &'static str,
    /// The stem of the files written for it.
    pub stem: &'static str,
    pub vars: Vec<LVar>,
    build: fn(&[LVar]) -> Vec<Guarded>,
}

impl Example {
    pub fn formulas(&self) -> Vec<Guarded> {
        (self.build)(&self.vars)
    }

    pub fn system(&self) -> System {
        system(&self.formulas())
    }

    /// The same system over `sigma`'s images. A variable `sigma` does not
    /// name panics ([`rename_var`]).
    pub fn renamed(&self, sigma: &CanonLabelling) -> Example {
        let vars = self
            .vars
            .iter()
            .map(|x| rename_var(sigma.theta(), *x))
            .collect();
        Example {
            vars,
            ..self.clone()
        }
    }
}

pub fn gamma_example() -> Example {
    Example {
        name: "Γ",
        tex: r"\Gamma",
        stem: "gamma",
        vars: vec![msg("x"), msg("y"), node("i"), node("j")],
        build: gamma,
    }
}

pub fn delta_example() -> Example {
    Example {
        name: "Δ",
        tex: r"\Delta",
        stem: "delta",
        vars: vec![msg("u"), msg("v"), node("k"), node("l")],
        build: delta,
    }
}

pub fn gamma_prime_example() -> Example {
    Example {
        name: "Γ'",
        tex: r"\Gamma'",
        stem: "gamma_prime",
        vars: vec![msg("x"), msg("y"), node("i")],
        build: gamma_prime,
    }
}

// =============================================================================
// The pipeline
// =============================================================================

/// `term` and everything the canonizer derives from it.
pub struct Run {
    pub term: STerm,
    pub graph: TermGraph,
    /// The bliss input.
    pub dimacs: String,
    pub bliss: BlissResult,
    pub stats: BlissStats,
    /// The canonical graph: `graph` relabeled by bliss's labeling, as DIMACS.
    pub canonical_dimacs: String,
    /// The canonical renaming.
    pub sigma: CanonLabelling,
    /// The term under its binders, renamed by `sigma` and normalized.
    pub form: STerm,
    /// The whole term, binders included, renamed by `sigma` and normalized:
    /// `C_AC(T(Γ)σ)`. The binders bind exactly `form`'s variables, per sort
    /// and scope, so `form` determines it.
    pub bound_form: STerm,
    /// The hash of `form`.
    pub key: Fingerprint,
    /// For each generator `g`: the renaming from the labeling `ρ ∘ g`, and the
    /// form under it.
    pub generator_forms: Vec<(CanonLabelling, STerm)>,
}

/// The term under all its binders.
fn body(term: &STerm) -> &STerm {
    match term {
        STerm::Exists(_, _, inner) => body(inner),
        t => t,
    }
}

/// The renaming from a labeling that puts vertex `w` at `position(w)`: the
/// variables named per sort, in the order of their vertices' positions.
fn name_variables(graph: &TermGraph, position: impl Fn(usize) -> usize) -> CanonLabelling {
    let mut vars: Vec<(usize, LVar)> = graph
        .free_vars
        .iter()
        .chain(&graph.witness_vars)
        .map(|&(w, x)| (position(w), x))
        .collect();
    vars.sort();
    let mut sigma = CanonLabelling::empty();
    for (_, x) in vars {
        sigma.bind_next(x);
    }
    sigma
}

/// Runs the canonizer's steps on `term`, a term of [`system_to_term`]'s shape
/// or its [`thesis_term`].
pub fn run(term: &STerm) -> Result<Run, BlissError> {
    let graph = term_to_graph(term);
    let dimacs = to_dimacs(&graph.colors, &graph.edges)?;
    let (bliss, stats) = run_bliss_with_stats(&dimacs)?;
    let rho = &bliss.canonical_labeling;

    let mut colors = vec![0; graph.colors.len()];
    for (w, color) in graph.colors.iter().enumerate() {
        colors[rho.image_of(w)] = *color;
    }
    let edges: BTreeSet<(usize, usize)> = graph
        .edges
        .iter()
        .map(|&(s, t)| (rho.image_of(s), rho.image_of(t)))
        .collect();
    let canonical_dimacs = to_dimacs(&colors, &edges.into_iter().collect::<Vec<_>>())?;

    let form_under = |sigma: &CanonLabelling| normalize(rename(body(term), sigma.theta()));
    let sigma = name_variables(&graph, |w| rho.image_of(w));
    let form = form_under(&sigma);
    let bound_form = normalize(rename(term, sigma.theta()));
    let key = fingerprint_sterm(&form);
    let generator_forms = bliss
        .generators
        .iter()
        .map(|g| {
            let other = name_variables(&graph, |w| rho.image_of(g.image_of(w)));
            let form = form_under(&other);
            (other, form)
        })
        .collect();

    Ok(Run {
        term: term.clone(),
        graph,
        dimacs,
        canonical_dimacs,
        sigma,
        form,
        bound_form,
        key,
        generator_forms,
        bliss,
        stats,
    })
}

/// The thesis's `T(Γ)` inside the system term `term`: the root binders, one
/// per sort, over the `Formulas` section. Panics unless the rest of the term
/// is empty, witness binders included, so nothing is dropped that `T` would
/// have to encode.
pub fn thesis_term(term: &STerm) -> STerm {
    let mut binders: Vec<(Binder, Vec<LVar>)> = Vec::new();
    let mut system = term;
    while let STerm::Exists(binder, vars, inner) = system {
        assert_eq!(
            binder.scope,
            Scope::Root,
            "not a system of formulas only: {term}"
        );
        binders.push((*binder, vars.clone()));
        system = inner;
    }
    let STerm::App(Op::System { .. }, sections) = system else {
        panic!("not a system term: {term}")
    };
    let mut formulas = None;
    for section in sections {
        match section {
            STerm::App(Op::Section(Section::Formulas), _) => formulas = Some(section.clone()),
            STerm::App(Op::Section(_), args) if args.is_empty() => {}
            STerm::Const(ConstKey::NoLast) => {}
            other => panic!("not a system of formulas only: {other}"),
        }
    }
    let Some(STerm::App(sys, constraints)) = formulas else {
        panic!("a system term has a Formulas section: {term}")
    };
    let constraints = constraints
        .into_iter()
        .map(|c| STerm::App(fml_op(), vec![c]))
        .collect();
    binders
        .into_iter()
        .rev()
        .fold(STerm::App(sys, constraints), |body, (binder, vars)| {
            STerm::Exists(binder, vars, Box::new(body))
        })
}

/// The chapter's `Fml`, which marks a constraint of `Sys` as a formula. The
/// code has no such operator because its `Formulas` section marks its
/// formulas instead, so a free unary symbol stands in for it at the thesis
/// level.
pub fn fml_op() -> Op {
    Op::Fun(FunSym::NoEq(NoEqSym::new(
        "Fml",
        1,
        Privacy::Public,
        Constructability::Constructor,
    )))
}

/// One system at both levels.
pub struct Study {
    pub example: Example,
    pub thesis: Run,
    pub implementation: Run,
    /// The library's own result on the implementation level's term.
    pub library: (CanonicalSystem, CanonLabelling, TermGraphStats),
}

pub fn study(example: &Example) -> Result<Study, BlissError> {
    let term = system_to_term(&example.system());
    let options = TermGraphOptions {
        check_generators: true,
        certificate: true,
    };
    Ok(Study {
        example: example.clone(),
        thesis: run(&thesis_term(&term))?,
        implementation: run(&term)?,
        library: canonicalize_term(&term, options)?,
    })
}

/// `b⁻¹ ∘ a`, which maps the variables of `a`'s system to those of `b`'s when
/// the forms are equal.
pub fn induced_renaming(a: &CanonLabelling, b: &CanonLabelling) -> BTreeMap<LVar, LVar> {
    let inverse: BTreeMap<LVar, LVar> = b.theta().iter().map(|(x, c)| (*c, *x)).collect();
    a.theta()
        .iter()
        .map(|(x, c)| (*x, rename_var(&inverse, *c)))
        .collect()
}

// =============================================================================
// Reading a graph
// =============================================================================

/// The variable each occurrence vertex belongs to.
fn occurrences(graph: &TermGraph) -> BTreeMap<usize, LVar> {
    let vars = variable_vertices(graph);
    graph
        .edges
        .iter()
        .filter_map(|(s, t)| vars.get(s).map(|x| (*t, *x)))
        .collect()
}

/// The variable of each variable vertex.
fn variable_vertices(graph: &TermGraph) -> BTreeMap<usize, LVar> {
    graph
        .free_vars
        .iter()
        .chain(&graph.witness_vars)
        .copied()
        .collect()
}

/// The variable vertices in canonical order: position, vertex, variable.
fn by_position(run: &Run) -> Vec<(usize, usize, LVar)> {
    let rho = &run.bliss.canonical_labeling;
    let mut vars: Vec<(usize, usize, LVar)> = variable_vertices(&run.graph)
        .into_iter()
        .map(|(w, x)| (rho.image_of(w), w, x))
        .collect();
    vars.sort();
    vars
}

/// Whether edge `(s, t)` is a term edge, as opposed to the edge from a
/// variable's vertex to one of its occurrences.
fn is_term_edge(graph: &TermGraph, s: usize, t: usize) -> bool {
    graph.parents[t] == Some(s)
}

fn vertex_label(graph: &TermGraph, w: usize) -> String {
    let occurrences = occurrences(graph);
    let variables = variable_vertices(graph);
    match &graph.keys[w].label {
        Label::Op(op) => op.to_string(),
        Label::Const(c) => format!("⟨{c}⟩"),
        Label::Binder(Binder { scope, sort }) => format!("Binder({scope:?},{sort:?})"),
        Label::BoundVar => format!("BoundVar w_{}", variables[&w]),
        Label::Occ => format!("Occ of {}", occurrences[&w]),
    }
}

// =============================================================================
// The text report
// =============================================================================

fn hex(fp: &Fingerprint) -> String {
    fp.iter().map(|b| format!("{b:02x}")).collect()
}

/// 1-based, like DIMACS and bliss's own output.
fn permutation_line(run: &Run, image: impl Fn(usize) -> usize) -> String {
    (0..run.graph.keys.len())
        .map(|w| format!("{}→{}", w + 1, image(w) + 1))
        .collect::<Vec<_>>()
        .join(" ")
}

/// The steps of `run`, for a system called `name`.
fn report_run(name: &str, level: &str, run: &Run) -> String {
    let graph = &run.graph;
    let rho = &run.bliss.canonical_labeling;
    let mut out = String::new();
    let _ = writeln!(out, "--- {name}, {level} level ---\n");
    let _ = writeln!(out, "Term\n  {}\n", run.term);

    let _ = writeln!(
        out,
        "Graph: {} vertices, {} edges (v and pos are 1-based, as in DIMACS and the chapter)",
        graph.keys.len(),
        graph.edges.len()
    );
    let _ = writeln!(
        out,
        "  {:>3} {:>5} {:>6} {:>4}  label",
        "v", "color", "parent", "pos"
    );
    for (w, key) in graph.keys.iter().enumerate() {
        let parent = graph.parents[w].map_or("-".to_string(), |p| (p + 1).to_string());
        let pos = key.pos.map_or("⊥".to_string(), |p| (p + 1).to_string());
        let _ = writeln!(
            out,
            "  {:>3} {:>5} {:>6} {:>4}  {}",
            w + 1,
            graph.colors[w],
            parent,
            pos,
            vertex_label(graph, w)
        );
    }
    let term_edges: Vec<String> = graph
        .edges
        .iter()
        .filter(|(s, t)| is_term_edge(graph, *s, *t))
        .map(|(s, t)| format!("{}→{}", s + 1, t + 1))
        .collect();
    let occurrence_edges: Vec<String> = graph
        .edges
        .iter()
        .filter(|(s, t)| !is_term_edge(graph, *s, *t))
        .map(|(s, t)| format!("{}→{}", s + 1, t + 1))
        .collect();
    let _ = writeln!(out, "  term edges:       {}", term_edges.join(" "));
    let _ = writeln!(out, "  occurrence edges: {}\n", occurrence_edges.join(" "));

    let _ = writeln!(out, "bliss input (DIMACS)\n{}", run.dimacs);

    let _ = writeln!(
        out,
        "bliss: |Aut| = {}, {} search nodes, {} generators",
        run.stats.aut_size,
        run.stats.search_nodes,
        run.bliss.generators.len()
    );
    let _ = writeln!(out, "  ρ: {}", permutation_line(run, |w| rho.image_of(w)));
    for g in &run.bliss.generators {
        let _ = writeln!(
            out,
            "  generator: {}",
            permutation_line(run, |w| g.image_of(w))
        );
    }
    let _ = writeln!(
        out,
        "\nCanonical graph 𝒞(G) (DIMACS)\n{}",
        run.canonical_dimacs
    );

    let _ = writeln!(out, "Canonical renaming σ");
    let _ = writeln!(
        out,
        "  {:<8} {:>6} {:>5}  σ(variable)",
        "variable", "vertex", "ρ"
    );
    for (position, w, x) in by_position(run) {
        let _ = writeln!(
            out,
            "  {:<8} {:>6} {:>5}  {}",
            x.to_string(),
            w + 1,
            position + 1,
            rename_var(run.sigma.theta(), x)
        );
    }
    let _ = writeln!(
        out,
        "\nCanonical form, with its binders\n  {}",
        run.bound_form
    );
    let _ = writeln!(
        out,
        "Canonical form, body (what the key hashes)\n  {}\n",
        run.form
    );
    let _ = writeln!(out, "key (SHA-256 of the form)\n  {}\n", hex(&run.key));
    for (i, (other, form)) in run.generator_forms.iter().enumerate() {
        let theta: Vec<String> = other
            .theta()
            .iter()
            .map(|(x, c)| format!("{x}↦{c}"))
            .collect();
        let _ = writeln!(out, "Under generator {}\n  σ: {}", i + 1, theta.join(", "));
        let _ = writeln!(out, "  form: {form}\n  same form: {}\n", *form == run.form);
    }
    out
}

/// The canonical constraint system, in Tamarin's own notation.
fn canonical_constraints(example: &Example, sigma: &CanonLabelling) -> Vec<String> {
    let mut lines: Vec<String> = example
        .renamed(sigma)
        .formulas()
        .iter()
        .map(pretty_guarded)
        .collect();
    lines.sort();
    lines
}

fn report_study(s: &Study) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "======== {} ========\n", s.example.name);
    for f in s.example.formulas() {
        let _ = writeln!(out, "  {}", pretty_guarded(&f));
    }
    let _ = writeln!(out);
    out += &report_run(s.example.name, "thesis", &s.thesis);
    let _ = writeln!(out, "Canonical constraint system (thesis level)");
    for line in canonical_constraints(&s.example, &s.thesis.sigma) {
        let _ = writeln!(out, "  {line}");
    }
    let _ = writeln!(out);
    out += &report_run(s.example.name, "implementation", &s.implementation);
    let (form, _, stats) = &s.library;
    let fingerprint = fingerprint_constraint_system(form);
    let _ = writeln!(
        out,
        "Library result: {} non-genuine generators of {}",
        stats.non_genuine_generators, stats.generators
    );
    let _ = writeln!(
        out,
        "key tuple: label {:?}, one SHA-256 per section",
        form.label()
    );
    for (name, fp) in SYSTEM_SECTIONS.iter().zip(&fingerprint.sections) {
        let _ = writeln!(out, "  {name:<16} {}", hex(fp));
    }
    let _ = writeln!(out);
    out
}

fn report_comparison(a: &Study, b: &Study) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "======== {} against {} ========\n",
        a.example.name, b.example.name
    );
    for (level, x, y) in [
        ("thesis", &a.thesis, &b.thesis),
        ("implementation", &a.implementation, &b.implementation),
    ] {
        let renaming: Vec<String> = induced_renaming(&x.sigma, &y.sigma)
            .iter()
            .map(|(p, q)| format!("{p}↦{q}"))
            .collect();
        let _ = writeln!(out, "{level} level");
        let _ = writeln!(out, "  bliss inputs equal:     {}", x.dimacs == y.dimacs);
        let _ = writeln!(
            out,
            "  canonical graphs equal: {}",
            x.canonical_dimacs == y.canonical_dimacs
        );
        let _ = writeln!(out, "  forms equal:            {}", x.form == y.form);
        let _ = writeln!(out, "  keys equal:             {}", x.key == y.key);
        let _ = writeln!(out, "  σ_b⁻¹ ∘ σ_a:            {}\n", renaming.join(", "));
    }
    out
}

// =============================================================================
// LaTeX
// =============================================================================

/// How [`latex`] writes a term.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Notation {
    /// The thesis's `Sys(Act(f, i), …)`.
    Term,
    /// As a set of constraints, `{f@i, …}`.
    Constraint,
    /// The code's own operators, each section on a line of an `aligned`.
    Implementation,
}

/// A canonical variable in the thesis's naming (`m_1`, `#t_1`, counted from
/// one per sort), any other variable by its name.
fn latex_var(x: LVar) -> String {
    match x.name {
        "mv" => format!("m_{{{}}}", x.idx + 1),
        "tv" => format!(r"\#t_{{{}}}", x.idx + 1),
        name if x.idx == 0 => name.to_string(),
        name => format!("{name}_{{{}}}", x.idx),
    }
}

fn latex_sort(sort: LSort) -> &'static str {
    match sort {
        LSort::Msg => r"\mathit{msg}",
        LSort::Node => r"\mathit{tmp}",
        LSort::Fresh => r"\mathit{fresh}",
        LSort::Pub => r"\mathit{pub}",
        LSort::Nat => r"\mathit{nat}",
    }
}

/// A typed binder's symbol: `∃` subscripted with its sort, and marked `wit`
/// when it binds witnesses.
fn latex_binder(Binder { scope, sort }: Binder) -> String {
    match scope {
        Scope::Root => format!(r"\exists_{{{}}}", latex_sort(sort)),
        Scope::Witnesses => format!(r"\exists^{{\mathsf{{wit}}}}_{{{}}}", latex_sort(sort)),
    }
}

fn latex_list(args: &[STerm], notation: Notation, separator: &str) -> String {
    args.iter()
        .map(|a| latex(a, notation))
        .collect::<Vec<_>>()
        .join(separator)
}

/// `t` as math. Closed subterms and names show as the constant they are; the
/// running example has none at the thesis level.
pub fn latex(t: &STerm, notation: Notation) -> String {
    match t {
        STerm::Var(VarRef::Free(x) | VarRef::Witness(x)) => latex_var(*x),
        STerm::Const(c) => format!(r"\langle\mathtt{{{c}}}\rangle"),
        STerm::Exists(binder, vars, inner) => {
            let vars: Vec<String> = vars.iter().map(|x| latex_var(*x)).collect();
            format!(
                r"{}\ {}.\ {}",
                latex_binder(*binder),
                vars.join(", "),
                latex(inner, notation)
            )
        }
        STerm::App(Op::Fun(FunSym::Ac(AcSym::Xor)), args) => {
            latex_list(args, notation, r" \oplus ")
        }
        // As a set of constraints, every member is a formula: no `Fml`.
        STerm::App(op, args) if notation == Notation::Constraint && *op == fml_op() => {
            latex(&args[0], notation)
        }
        STerm::App(Op::GAction, args) if notation == Notation::Constraint => {
            format!(
                "{}@{}",
                latex(&args[0], notation),
                latex(&args[1], notation)
            )
        }
        STerm::App(Op::GAction, args) if notation == Notation::Term => {
            format!(r"\mathsf{{Act}}({})", latex_list(args, notation, ", "))
        }
        STerm::App(Op::Section(Section::Formulas), args) if notation == Notation::Constraint => {
            format!(r"\{{{}\}}", latex_list(args, notation, r",\ "))
        }
        STerm::App(Op::Section(Section::Formulas), args) if notation == Notation::Term => {
            format!(r"\mathsf{{Sys}}({})", latex_list(args, notation, ", "))
        }
        STerm::App(Op::Section(s), args) => {
            format!(
                r"\mathsf{{{s:?}}}\{{{}\}}",
                latex_list(args, notation, ", ")
            )
        }
        STerm::App(Op::System { .. }, args) => {
            let sections = latex_list(args, notation, ",\\\\\n  &\\quad ");
            format!("\\mathsf{{System}}(\\\\\n  &\\quad {sections})")
        }
        STerm::App(op, args) => {
            format!(r"\mathsf{{{op}}}({})", latex_list(args, notation, ", "))
        }
    }
}

/// A vertex's label as math. Vertices carry no variable names or sorts (a
/// variable's sort is its binder's), except that the chapter calls a variable
/// vertex `w_x`; the occurrence's variable is only in the text of the vertex
/// table.
fn latex_label(graph: &TermGraph, w: usize) -> String {
    match &graph.keys[w].label {
        Label::Op(Op::Fun(FunSym::Ac(AcSym::Xor))) => r"\oplus".to_string(),
        Label::Op(Op::Section(Section::Formulas)) => r"\mathsf{Sys}".to_string(),
        Label::Op(Op::GAction) => r"\mathsf{Act}".to_string(),
        Label::Op(op) => format!(r"\mathsf{{{op}}}"),
        Label::Const(c) => format!(r"\langle\mathtt{{{c}}}\rangle"),
        Label::Binder(binder) => latex_binder(*binder),
        Label::BoundVar => format!("w_{{{}}}", latex_var(variable_vertices(graph)[&w])),
        Label::Occ => r"\mathit{occ}".to_string(),
    }
}

/// A vertex's key as math, `(label, position)`, as the chapter writes it:
/// the position is 1-based, `⊥` under a parent that is not free. Unlike
/// [`latex_label`], a variable vertex's key carries no name: all of them are
/// `(var, ⊥)`.
fn latex_key(graph: &TermGraph, w: usize) -> String {
    let key = &graph.keys[w];
    let label = match key.label {
        Label::BoundVar => r"\mathit{var}".to_string(),
        _ => latex_label(graph, w),
    };
    format!("({label}, {})", latex_pos(key.pos))
}

fn latex_pos(pos: Option<usize>) -> String {
    pos.map_or(r"\bot".to_string(), |p| (p + 1).to_string())
}

/// The x position and depth of each vertex in a drawing of the term tree:
/// leaves side by side in vertex order, a parent centred over its children.
fn tree_layout(graph: &TermGraph) -> Vec<(f64, usize)> {
    fn place(
        w: usize,
        depth: usize,
        children: &[Vec<usize>],
        next: &mut f64,
        out: &mut [(f64, usize)],
    ) -> f64 {
        let xs: Vec<f64> = children[w]
            .iter()
            .map(|c| place(*c, depth + 1, children, next, out))
            .collect();
        let x = match (xs.first(), xs.last()) {
            (Some(first), Some(last)) => (first + last) / 2.0,
            _ => {
                *next += 1.0;
                *next - 1.0
            }
        };
        out[w] = (x, depth);
        x
    }
    let n = graph.keys.len();
    let mut children = vec![Vec::new(); n];
    for (w, parent) in graph.parents.iter().enumerate() {
        if let Some(p) = parent {
            children[*p].push(w);
        }
    }
    let mut out = vec![(0.0, 0); n];
    place(0, 0, &children, &mut 0.0, &mut out);
    out
}

/// `run`'s graph as a TikZ picture. Each vertex shows its label and, small,
/// its position: together they are its key. Colors (ranks of keys) are left
/// out, since they depend on which keys the graph has.
fn tikz(run: &Run) -> String {
    let graph = &run.graph;
    let mut out = String::new();
    out += r"\begin{tikzpicture}[
    x=1.25cm, y=-1.3cm, >=stealth,
    vertex/.style={draw, circle, inner sep=1.5pt, minimum size=7mm, font=\small},
    posno/.style={font=\tiny, text=gray, inner sep=1pt},
    term/.style={->},
    occurrence/.style={->, dashed, gray}]
";
    for (w, (x, depth)) in tree_layout(graph).iter().enumerate() {
        let _ = writeln!(
            out,
            "  \\node[vertex, label={{[posno]above right:${}$}}] (v{}) at ({x:.2}, {depth}) {{${}$}};",
            latex_pos(graph.keys[w].pos),
            w + 1,
            latex_label(graph, w)
        );
    }
    for (s, t) in &graph.edges {
        let style = if is_term_edge(graph, *s, *t) {
            "term"
        } else {
            "occurrence"
        };
        let _ = writeln!(out, "  \\draw[{style}] (v{}) -- (v{});", s + 1, t + 1);
    }
    out += "\\end{tikzpicture}\n";
    out
}

/// The vertices with their keys; the variable a vertex is or occurs as is
/// only a reading aid.
fn latex_vertex_table(run: &Run) -> String {
    let graph = &run.graph;
    let occurrences = occurrences(graph);
    let variables = variable_vertices(graph);
    let mut out = String::from(
        "\\begin{tabular}{rrll}\n  \\toprule\n  \
         $v$ & parent & key & variable \\\\\n  \\midrule\n",
    );
    for w in 0..graph.keys.len() {
        let parent = graph.parents[w].map_or("--".to_string(), |p| (p + 1).to_string());
        let variable = match (variables.get(&w), occurrences.get(&w)) {
            (Some(x), _) => format!("${}$", latex_var(*x)),
            (_, Some(x)) => format!("occurrence of ${}$", latex_var(*x)),
            _ => String::new(),
        };
        let _ = writeln!(
            out,
            "  {} & {parent} & ${}$ & {variable} \\\\",
            w + 1,
            latex_key(graph, w)
        );
    }
    out += "  \\bottomrule\n\\end{tabular}\n";
    out
}

/// The colors bliss gets: each distinct key of the graph and its rank among
/// them.
fn latex_color_table(run: &Run) -> String {
    let graph = &run.graph;
    let mut ranks: BTreeMap<usize, (String, usize)> = BTreeMap::new();
    for w in 0..graph.keys.len() {
        let color = graph.colors[w] as usize;
        ranks
            .entry(color)
            .or_insert_with(|| (latex_key(graph, w), 0))
            .1 += 1;
    }
    let mut out = String::from(
        "\\begin{tabular}{rlr}\n  \\toprule\n  \
         color & key & vertices \\\\\n  \\midrule\n",
    );
    for (color, (key, count)) in ranks {
        let _ = writeln!(out, "  {color} & ${key}$ & {count} \\\\");
    }
    out += "  \\bottomrule\n\\end{tabular}\n";
    out
}

/// `ρ` as a two-row table, split so that it fits a line.
fn latex_rho_table(run: &Run, tex: &str) -> String {
    const PER_ROW: usize = 12;
    let rho = &run.bliss.canonical_labeling;
    let n = run.graph.keys.len();
    let mut out = format!("\\begin{{tabular}}{{r*{{{PER_ROW}}}{{r}}}}\n  \\toprule\n");
    for (i, start) in (0..n).step_by(PER_ROW).enumerate() {
        if i > 0 {
            out += "  \\midrule\n";
        }
        let row: Vec<usize> = (start..n.min(start + PER_ROW)).collect();
        let vs: Vec<String> = row.iter().map(|w| (w + 1).to_string()).collect();
        let images: Vec<String> = row
            .iter()
            .map(|w| (rho.image_of(*w) + 1).to_string())
            .collect();
        let _ = writeln!(out, "  $v$ & {} \\\\", vs.join(" & "));
        let _ = writeln!(out, "  $\\rho_{{{tex}}}(v)$ & {} \\\\", images.join(" & "));
    }
    out += "  \\bottomrule\n\\end{tabular}\n";
    out
}

fn latex_sigma_table(run: &Run, tex: &str) -> String {
    let mut out = format!(
        "\\begin{{tabular}}{{lrrl}}\n  \\toprule\n  \
         variable & vertex & $\\rho_{{{tex}}}$ & $\\sigma_{{{tex}}}$ \\\\\n  \\midrule\n"
    );
    for (position, w, x) in by_position(run) {
        let _ = writeln!(
            out,
            "  ${}$ & {} & {} & ${}$ \\\\",
            latex_var(x),
            w + 1,
            position + 1,
            latex_var(rename_var(run.sigma.theta(), x))
        );
    }
    out += "  \\bottomrule\n\\end{tabular}\n";
    out
}

fn latex_dimacs(dimacs: &str) -> String {
    format!(
        "\\begin{{lstlisting}}[basicstyle=\\ttfamily\\footnotesize]\n{dimacs}\\end{{lstlisting}}\n"
    )
}

/// A commented block of `fragments.tex`.
fn block(out: &mut String, title: &str, content: &str) {
    let _ = writeln!(out, "% --- {title} ---\n{content}");
}

fn tex_display(math: &str) -> String {
    format!("\\[\n  {math}\n\\]\n")
}

/// The LaTeX for each step, at the thesis level unless a title says otherwise.
fn fragments(studies: &[&Study], prime: &Study) -> String {
    let mut out = String::from(
        "% The running example, generated by\n\
         %   cargo run --example canon_worked_example -- --out DIR\n\
         % Needs amsmath, booktabs, listings and tikz. Only the blocks are meant\n\
         % to be used: the file is not a document.\n\n",
    );
    for s in studies {
        let (name, tex, run) = (s.example.name, s.example.tex, &s.thesis);
        let term = latex(&run.term, Notation::Term);
        block(
            &mut out,
            &format!("subsec:canon:term: T({name})"),
            &tex_display(&format!("T({tex}) = {term}")),
        );
        block(
            &mut out,
            &format!("subsec:canon:graph: G_T({name})"),
            &tikz(run),
        );
        block(
            &mut out,
            &format!("subsec:canon:graph: vertices of G_T({name})"),
            &latex_vertex_table(run),
        );
        block(
            &mut out,
            &format!("bliss input for G_T({name})"),
            &latex_dimacs(&run.dimacs),
        );
        block(
            &mut out,
            &format!("subsec:canon:form: canonical labelling of G_T({name})"),
            &latex_rho_table(run, tex),
        );
        block(
            &mut out,
            &format!("subsec:canon:form: canonical renaming of {name}"),
            &latex_sigma_table(run, tex),
        );
        block(
            &mut out,
            &format!(
                "subsec:canon:form: canonical form of {name}, C_AC(T({name})σ), with its binders"
            ),
            &tex_display(&format!(
                "\\mathcal{{C}}_{{\\alpha\\text{{-}}AC}}({tex}) = {}",
                latex(&run.bound_form, Notation::Term)
            )),
        );
        block(
            &mut out,
            &format!(
                "subsec:canon:form: canonical form of {name}, body only (what the key hashes; \
                 the binders bind exactly its variables, per sort)"
            ),
            &tex_display(&format!(
                "\\mathcal{{C}}_{{\\alpha\\text{{-}}AC}}({tex}) = {}",
                latex(&run.form, Notation::Term)
            )),
        );
        block(
            &mut out,
            &format!("subsec:canon:form: canonical constraint system of {name}"),
            &tex_display(&latex(&run.form, Notation::Constraint)),
        );
        block(
            &mut out,
            &format!("From Forms to Keys: key({name})"),
            &tex_display(&format!(
                "\\mathit{{key}}({tex}) = \\mathtt{{{}\\ldots}}",
                &hex(&run.key)[..16]
            )),
        );
    }

    if let [a, b] = studies {
        let renaming: Vec<String> = induced_renaming(&a.thesis.sigma, &b.thesis.sigma)
            .iter()
            .map(|(p, q)| format!(r"{} \mapsto {}", latex_var(*p), latex_var(*q)))
            .collect();
        block(
            &mut out,
            "subsec:canon:form: the renaming the forms induce",
            &tex_display(&format!(
                "\\sigma_{{{}}}^{{-1}} \\circ \\sigma_{{{}}} = \\{{{}\\}}",
                b.example.tex,
                a.example.tex,
                renaming.join(r",\ ")
            )),
        );
        block(
            &mut out,
            "bliss output: the canonical graph, equal for both systems",
            &latex_dimacs(&a.thesis.canonical_dimacs),
        );
    }

    let run = &prime.thesis;
    let tex = prime.example.tex;
    block(
        &mut out,
        "Automorphisms Need No Tie-Breaking: the term",
        &tex_display(&format!("T({tex}) = {}", latex(&run.term, Notation::Term))),
    );
    block(
        &mut out,
        "Automorphisms Need No Tie-Breaking: the graph",
        &tikz(run),
    );
    let mut labellings = format!(
        "\\[\n  \\begin{{aligned}}\n    \\sigma_{{{tex}}} : {{}}& {} \\\\\n",
        latex_renaming(&run.sigma)
    );
    for (other, form) in &run.generator_forms {
        let _ = writeln!(
            labellings,
            "    \\text{{under the automorphism}} : {{}}& {} \\quad\\text{{gives}}\\quad {} \\\\",
            latex_renaming(other),
            latex(form, Notation::Constraint)
        );
    }
    let _ = writeln!(
        labellings,
        "    \\text{{canonical form}} : {{}}& {}\n  \\end{{aligned}}\n\\]",
        latex(&run.form, Notation::Constraint)
    );
    block(
        &mut out,
        "Automorphisms Need No Tie-Breaking: both labellings give one form",
        &labellings,
    );

    if let [a, ..] = studies {
        block(
            &mut out,
            &format!(
                "subsec:canon:impl: Calling bliss: colors as ranks of the keys of G_T({})",
                a.example.name
            ),
            &latex_color_table(&a.thesis),
        );
        block(
            &mut out,
            &format!(
                "subsec:canon:impl: the system term of {} as the code builds it \
                 (System[source_kind, side, subterm_store.contradictory])",
                a.example.name
            ),
            &format!(
                "\\[\n  \\begin{{aligned}}\n  &{}\n  \\end{{aligned}}\n\\]\n",
                latex(&a.implementation.term, Notation::Implementation)
            ),
        );
    }
    out
}

fn latex_renaming(sigma: &CanonLabelling) -> String {
    let pairs: Vec<String> = sigma
        .theta()
        .iter()
        .map(|(x, c)| format!(r"{} \mapsto {}", latex_var(*x), latex_var(*c)))
        .collect();
    format!(r"\{{{}\}}", pairs.join(r",\ "))
}

// =============================================================================
// Output
// =============================================================================

/// The whole text report and the LaTeX fragments.
pub fn render(gamma: &Study, delta: &Study, prime: &Study) -> (String, String) {
    let report = format!(
        "{}{}{}{}",
        report_study(gamma),
        report_study(delta),
        report_comparison(gamma, delta),
        report_study(prime)
    );
    (report, fragments(&[gamma, delta], prime))
}

fn write_outputs(
    dir: &PathBuf,
    report: &str,
    tex: &str,
    gamma: &Study,
    delta: &Study,
    prime: &Study,
) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join("report.txt"), report)?;
    std::fs::write(dir.join("fragments.tex"), tex)?;
    std::fs::write(dir.join("canonical.dimacs"), &gamma.thesis.canonical_dimacs)?;
    for s in [gamma, delta, prime] {
        let stem = s.example.stem;
        std::fs::write(dir.join(format!("{stem}.dimacs")), &s.thesis.dimacs)?;
    }
    for s in [gamma, delta] {
        let stem = s.example.stem;
        std::fs::write(
            dir.join(format!("{stem}.impl.dimacs")),
            &s.implementation.dimacs,
        )?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out_dir = match args.iter().map(String::as_str).collect::<Vec<_>>()[..] {
        [] => None,
        ["--out", dir] => Some(PathBuf::from(dir)),
        _ => {
            eprintln!("usage: canon_worked_example [--out DIR]");
            std::process::exit(2);
        }
    };

    let gamma = study(&gamma_example())?;
    let delta = study(&delta_example())?;
    let prime = study(&gamma_prime_example())?;
    let (report, tex) = render(&gamma, &delta, &prime);
    print!("{report}");
    if let Some(dir) = out_dir {
        write_outputs(&dir, &report, &tex, &gamma, &delta, &prime)?;
        eprintln!("wrote {}", dir.display());
    }
    Ok(())
}
