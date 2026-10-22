//! Profiles one proof of a lemma, so that a search that failed (or
//! succeeded) on it can be lined up against it: the search diagnosis in
//! `canon/experiments/search_diagnosis`.
//!
//! Usage:
//!
//! ```text
//! profile_reference_proof <theory.spthy> <lemma> --out <profile.jsonl>
//!     [--rank-with <other.spthy>] [--time-expansion]
//! ```
//!
//! The proof is the one `tamarin-rs --prove` would print with the same
//! environment: greedy without `TAM_RS_*`, the top-N or MCGS search's
//! materialized proof with their variables (they select the search in the
//! lemma's context). Each proof node is one JSON line:
//! - `path` (case names from the root), `depth`, `method`, `status`;
//! - `key` and `method_fp`: the system's canonical key and the method's
//!   canonical fingerprint, written as a graph dump writes them
//!   (`TAM_RS_SEARCH_DUMP`, `search_graph::system_key_hex`), so a dump and a
//!   profile join on them;
//! - `rank`: where the node's method sits among the system's candidates in
//!   this theory's context: `raw` in `candidate_methods`' list, `applicable`
//!   among the ones that apply (greedy picks the first that applies, so 0
//!   along a greedy proof) with the ranks of those ranked above it that do
//!   not (`inapplicable_before`), and MCGS's prior for it (`McgsConfig` defaults,
//!   which the evaluation pins) with its rank among the priors and its
//!   share of their sum; `bm25_z`, every candidate's BM25 bias at weight 1,
//!   recomputes the prior offline for other weights;
//! - `rank_with`: the same in the context of `--rank-with`'s theory, which
//!   must have the same rules (only its heuristics or tactics differ; the
//!   roots' canonical keys are checked): a tactic proof seen through the
//!   default heuristic's eyes;
//! - `expand_s` with `--time-expansion`: what MCGS's expansion of the system
//!   costs, sequentially (rank, apply the 50 best-prior candidates,
//!   canonicalize every unfinished case).
//!
//! The last line summarizes the proof: size, depth distribution, and the
//! discrepancies (nodes whose method is not the first applicable one in the
//! ranking context) per root-to-leaf path and by depth.

// Example/dev tool: writes its report to stdout/stderr by design.
#![allow(clippy::disallowed_macros)]

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod common;

use std::collections::BTreeMap;
use std::io::Write;
use std::time::Instant;

use serde_json::{json, Value};
use tamarin_theory::canon::{
    canonicalize_constraint_system_with_labelling, canonicalize_proof_method,
};
use tamarin_theory::canon_fingerprint::{fingerprint_constraint_system, fingerprint_proof_method};
use tamarin_theory::constraint::solver::context::ProofContext;
use tamarin_theory::constraint::solver::mcgs_search::{
    candidate_bm25_z, candidate_priors, McgsConfig,
};
use tamarin_theory::constraint::solver::proof_method::{
    exec_proof_method, is_finished, ProofMethod,
};
use tamarin_theory::constraint::solver::search::{candidate_methods, run_proof_search, ProofNode};
use tamarin_theory::constraint::solver::search_graph::{method_key_hex, system_key_hex};
use tamarin_theory::constraint::system::System;
use tamarin_theory::pretty_theory::pretty_proof_method_inline;

struct Args {
    theory: String,
    lemma: String,
    out: String,
    rank_with: Option<String>,
    time_expansion: bool,
}

fn parse_args() -> Result<Args, String> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut positional = Vec::new();
    let (mut out, mut rank_with, mut time_expansion) = (None, None, false);
    let mut it = raw.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--out" => out = Some(it.next().ok_or("--out needs a path")?),
            "--rank-with" => rank_with = Some(it.next().ok_or("--rank-with needs a theory")?),
            "--time-expansion" => time_expansion = true,
            _ if a.starts_with("--") => return Err(format!("unknown option {a}")),
            _ => positional.push(a),
        }
    }
    let [theory, lemma]: [String; 2] = positional
        .try_into()
        .map_err(|_| "expected <theory.spthy> <lemma>".to_string())?;
    Ok(Args {
        theory,
        lemma,
        out: out.ok_or("--out is required")?,
        rank_with,
        time_expansion,
    })
}

fn context(theory: &str, lemma: &str) -> Result<(ProofContext, System), String> {
    let opts = common::LoadOpts {
        keep_systems: true,
        ..Default::default()
    };
    let loaded = common::try_load_theory_for_proving(theory, &opts)?;
    loaded
        .session
        .context_and_system_for_lemma(lemma)
        .map_err(|e| format!("context_and_system_for_lemma({lemma}) in {theory}: {e}"))
}

/// The system's canonical key and the method's canonical fingerprint, as a
/// graph dump names them.
fn keys(sys: &System, method: &ProofMethod) -> Result<(String, String), String> {
    let (canon, labelling) = canonicalize_constraint_system_with_labelling(sys)
        .map_err(|e| format!("canonicalize: {e}"))?;
    let key = system_key_hex(&fingerprint_constraint_system(&canon));
    let method_fp = fingerprint_proof_method(&canonicalize_proof_method(method, sys, &labelling));
    Ok((key, method_key_hex(&method_fp)))
}

/// Where `method` sits among `sys`'s candidates in `ctx`.
fn rank(
    ctx: &ProofContext,
    sys: &System,
    depth: usize,
    method: &ProofMethod,
    path: &[(&ProofMethod, &System)],
    mcgs: &McgsConfig,
) -> Result<Value, String> {
    let ranked =
        candidate_methods(sys, ctx, depth).map_err(|e| format!("candidate_methods: {e}"))?;
    let Some(raw) = ranked.iter().position(|m| m == method) else {
        return Ok(json!({"candidates": ranked.len(), "raw": null}));
    };
    let mut applicable = 0;
    let mut inapplicable = Vec::new();
    for (i, m) in ranked[..raw].iter().enumerate() {
        if exec_proof_method(ctx, m, sys)
            .map_err(|e| format!("exec_proof_method: {e}"))?
            .is_some()
        {
            applicable += 1;
        } else {
            inapplicable.push(i);
        }
    }
    let priors = candidate_priors(mcgs, &ctx.lemma_text, sys, &ranked, path);
    let sum: f64 = priors.iter().sum();
    let prior = priors[raw];
    Ok(json!({
        "candidates": ranked.len(),
        "raw": raw,
        "applicable": applicable,
        "inapplicable_before": inapplicable,
        "prior": prior,
        "prior_rank": priors.iter().filter(|&&q| q > prior).count(),
        "prior_share": prior / sum,
        "bm25_z": candidate_bm25_z(mcgs, &ctx.lemma_text, sys, &ranked, path),
    }))
}

/// Seconds MCGS's expansion of `sys` takes, run sequentially.
fn expansion_secs(
    ctx: &ProofContext,
    sys: &System,
    depth: usize,
    path: &[(&ProofMethod, &System)],
    mcgs: &McgsConfig,
) -> Result<f64, String> {
    let start = Instant::now();
    let ranked =
        candidate_methods(sys, ctx, depth).map_err(|e| format!("candidate_methods: {e}"))?;
    let priors = candidate_priors(mcgs, &ctx.lemma_text, sys, &ranked, path);
    let mut order: Vec<usize> = (0..ranked.len()).collect();
    order.sort_by(|&x, &y| priors[y].total_cmp(&priors[x]));
    if mcgs.top_n > 0 {
        order.truncate(mcgs.top_n as usize);
    }
    for i in order {
        let Some(cases) = exec_proof_method(ctx, &ranked[i], sys)
            .map_err(|e| format!("exec_proof_method: {e}"))?
        else {
            continue;
        };
        for (_, case) in cases {
            if is_finished(ctx, &case).is_none() {
                canonicalize_constraint_system_with_labelling(&case)
                    .map_err(|e| format!("canonicalize: {e}"))?;
            }
        }
    }
    Ok(start.elapsed().as_secs_f64())
}

struct Profiler<'a> {
    ctx: &'a ProofContext,
    rank_ctx: Option<&'a ProofContext>,
    time_expansion: bool,
    mcgs: McgsConfig,
    rows: Vec<Value>,
    /// Per root-to-leaf path: its discrepancies.
    path_discrepancies: Vec<u32>,
    /// Discrepancies by depth.
    discrepancy_depths: BTreeMap<usize, u32>,
    depth_hist: BTreeMap<usize, u32>,
}

impl Profiler<'_> {
    fn walk<'n>(
        &mut self,
        node: &'n ProofNode,
        names: &mut Vec<String>,
        path: &mut Vec<(&'n ProofMethod, &'n System)>,
        discrepancies: u32,
    ) -> Result<(), String> {
        let depth = path.len();
        *self.depth_hist.entry(depth).or_default() += 1;
        let inner = !matches!(
            node.method,
            ProofMethod::Finished(_) | ProofMethod::Sorry(_)
        );
        let mut row = json!({
            "kind": "node",
            "path": names,
            "depth": depth,
            "method": pretty_proof_method_inline(&node.method),
            "status": format!("{:?}", node.status),
            "children": node.children.len(),
        });
        let mut here = discrepancies;
        if inner {
            let (key, method_fp) = keys(&node.sys, &node.method)?;
            row["key"] = json!(key);
            row["method_fp"] = json!(method_fp);
            let own = rank(self.ctx, &node.sys, depth, &node.method, path, &self.mcgs)?;
            let by = match self.rank_ctx {
                Some(rctx) => {
                    let r = rank(rctx, &node.sys, depth, &node.method, path, &self.mcgs)?;
                    row["rank_with"] = r.clone();
                    r
                }
                None => own.clone(),
            };
            row["rank"] = own;
            if by["applicable"].as_u64() != Some(0) {
                here += 1;
                *self.discrepancy_depths.entry(depth).or_default() += 1;
            }
            if self.time_expansion {
                let ctx = self.rank_ctx.unwrap_or(self.ctx);
                row["expand_s"] = json!(expansion_secs(ctx, &node.sys, depth, path, &self.mcgs)?);
            }
        }
        self.rows.push(row);
        if node.children.is_empty() {
            self.path_discrepancies.push(here);
            return Ok(());
        }
        path.push((&node.method, &node.sys));
        for (name, child) in &node.children {
            names.push(name.clone());
            self.walk(child, names, path, here)?;
            names.pop();
        }
        path.pop();
        Ok(())
    }
}

fn run(args: Args) -> Result<(), String> {
    let (ctx, initial) = context(&args.theory, &args.lemma)?;
    let rank_ctx = match &args.rank_with {
        Some(other) => {
            let (rctx, rinitial) = context(other, &args.lemma)?;
            let probe = ProofMethod::Simplify;
            let (k1, _) = keys(&initial, &probe)?;
            let (k2, _) = keys(&rinitial, &probe)?;
            if k1 != k2 {
                return Err(format!(
                    "--rank-with: the initial systems of {} and {other} differ canonically; \
                     the two theories must have the same rules",
                    args.theory
                ));
            }
            Some(rctx)
        }
        None => None,
    };
    let start = Instant::now();
    let root = run_proof_search(&ctx, initial, usize::MAX).map_err(|e| format!("prove: {e}"))?;
    let search_s = start.elapsed().as_secs_f64();
    eprintln!(
        "proof found in {search_s:.1}s, status {:?}; profiling",
        root.status
    );

    let mut profiler = Profiler {
        ctx: &ctx,
        rank_ctx: rank_ctx.as_ref(),
        time_expansion: args.time_expansion,
        mcgs: McgsConfig::with_budget(0),
        rows: Vec::new(),
        path_discrepancies: Vec::new(),
        discrepancy_depths: BTreeMap::new(),
        depth_hist: BTreeMap::new(),
    };
    profiler.walk(&root, &mut Vec::new(), &mut Vec::new(), 0)?;

    let paths = &profiler.path_discrepancies;
    let summary = json!({
        "kind": "summary",
        "theory": args.theory,
        "lemma": args.lemma,
        "rank_with": args.rank_with,
        "status": format!("{:?}", root.status),
        "search_s": search_s,
        "size": profiler.rows.len(),
        "max_depth": profiler.depth_hist.keys().max(),
        "depth_hist": profiler.depth_hist,
        "leaves": paths.len(),
        "discrepancies_max_per_path": paths.iter().max(),
        "discrepancies_mean_per_path": paths.iter().map(|&d| f64::from(d)).sum::<f64>() / paths.len().max(1) as f64,
        "discrepancy_depths": profiler.discrepancy_depths,
        "expand_s_total": args.time_expansion.then(|| profiler
            .rows
            .iter()
            .filter_map(|r| r["expand_s"].as_f64())
            .sum::<f64>()),
    });
    let file = std::fs::File::create(&args.out).map_err(|e| format!("{}: {e}", args.out))?;
    let mut out = std::io::BufWriter::new(file);
    for row in profiler.rows.iter().chain(std::iter::once(&summary)) {
        serde_json::to_writer(&mut out, row).map_err(|e| e.to_string())?;
        writeln!(out).map_err(|e| e.to_string())?;
    }
    out.flush().map_err(|e| e.to_string())?;
    eprintln!("{}", summary);
    Ok(())
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("profile_reference_proof: {e}");
            std::process::exit(2);
        }
    };
    // Proofs of the evaluation's big models recurse deeply.
    let worker = std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(move || run(args))
        .expect("spawn the profiler thread");
    match worker.join().expect("profiler thread panicked") {
        Ok(()) => {}
        Err(e) => {
            eprintln!("profile_reference_proof: {e}");
            std::process::exit(1);
        }
    }
}
