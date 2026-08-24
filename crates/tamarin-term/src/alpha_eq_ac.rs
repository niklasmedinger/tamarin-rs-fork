// Currently GPL 3.0 until granted permission by the upstream authors
// of the tamarin-prover sources this file cites; list them with:
//   scripts/gen_license_headers.py --authors <this file>

//! Canonization of terms for alpha equivalence modulo AC ($\alphaeqac$), as
//! developed in `work.tex` §"Canonization of Constraint Systems" /
//! "Canonization of Terms".
//!
//! `CAN_alphaeqac` computes a canonical representative for a term's
//! $\alphaeqac$-equivalence class by 1) canonically renaming its literals
//! (variables and names) by sort, delaying the renaming of literals that
//! only occur under AC/C symbols until no permutation-free choice remains,
//! and 2) bringing the result into `CAN_AC` normal form. See Algorithm 1
//! (`CAN_alphaeqac`) and Theorem `thm:can_alphaeqac` in `work.tex`.
//!
//! This module is under active development (see TODO.md); the tests below
//! are written test-first and pin down the required equivalences before the
//! algorithm is implemented.

use crate::{
    lterm::{LNTerm, LSort, LVar, Name},
    subst::Subst,
    vterm::Lit,
};
use std::collections::{BTreeMap, BTreeSet, HashMap};

type LNLit = Lit<Name, LVar>;

/// Extended term positions, per work.tex's "Canonizer for $\alphaeqac$":
/// unlike ordinary positions, an AC application is crossed by *symbol*
/// (grouping its flattened arguments, optionally filtered by the head
/// symbol of a matched child) instead of by index, since AC arguments have
/// no canonical order to index into.
pub mod position {
    use crate::function_symbols::FunSym;
    use crate::lterm::LNTerm;
    use crate::term::Term;
    use std::collections::BTreeSet;

    use super::LNLit;

    /// One step of a position. `work.tex` writes a step as the crossed
    /// node's own symbol followed by either an index or a further symbol;
    /// here the crossed symbol is implicit (it is `t`'s head at the point
    /// the step is applied), so a step only needs to carry the selector.
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub enum PosStep {
        /// `p = f \cdot i \cdot p'`: the `i`-th argument of a fixed-order
        /// (non-AC) application.
        Arg(usize),
        /// Crossing an AC application: `None` is `p = f \cdot \epsilon`
        /// (select the whole flattened argument multiset — always the
        /// last step of a position); `Some(sym)` is `p = f \cdot g \cdot p'`
        /// (select the flattened arguments headed by `sym`, then continue).
        AcGroup(Option<FunSym>),
    }

    /// A position: a sequence of [`PosStep`]s. The empty position `ε`
    /// denotes the term itself.
    pub type Position = Vec<PosStep>;

    /// $\mathit{Pos}(t)$: every valid position of `t`, including `ε`.
    pub fn positions(t: &LNTerm) -> Vec<Position> {
        let mut out = vec![Vec::new()];
        let mut prefix = Vec::new();
        collect_positions(t, &mut prefix, &mut out);
        out
    }

    fn collect_positions(t: &LNTerm, prefix: &mut Position, out: &mut Vec<Position>) {
        match t {
            Term::Lit(_) => {}
            Term::App(FunSym::Ac(_), args) => {
                // `p = f . \epsilon`: the whole flattened argument multiset.
                prefix.push(PosStep::AcGroup(None));
                out.push(prefix.clone());
                prefix.pop();

                // `p = f . g . p'` for each distinct head symbol `g` among
                // the (already AC-flattened) children.
                let mut seen = BTreeSet::new();
                for a in args.iter() {
                    if let Term::App(sym, _) = a {
                        if seen.insert(*sym) {
                            prefix.push(PosStep::AcGroup(Some(*sym)));
                            out.push(prefix.clone());
                            collect_positions(a, prefix, out);
                            prefix.pop();
                        }
                    }
                }
            }
            Term::App(_, args) => {
                for (i, a) in args.iter().enumerate() {
                    prefix.push(PosStep::Arg(i));
                    out.push(prefix.clone());
                    collect_positions(a, prefix, out);
                    prefix.pop();
                }
            }
        }
    }

    /// $t|_p$: the (possibly multi-valued, at an AC-crossing step) set of
    /// subterms reachable at position `p`. Empty iff `p \notin \mathit{Pos}(t)`.
    pub fn subterms_at<'t>(t: &'t LNTerm, p: &[PosStep]) -> BTreeSet<&'t LNTerm> {
        let mut out = BTreeSet::new();
        collect_subterms_at(t, p, &mut out);
        out
    }

    fn collect_subterms_at<'t>(t: &'t LNTerm, p: &[PosStep], out: &mut BTreeSet<&'t LNTerm>) {
        let Some((step, rest)) = p.split_first() else {
            out.insert(t);
            return;
        };
        match (step, t) {
            (PosStep::Arg(i), Term::App(f, args)) if !f.is_ac() => {
                if let Some(a) = args.get(*i) {
                    collect_subterms_at(a, rest, out);
                }
            }
            (PosStep::AcGroup(filter), Term::App(f, args)) if f.is_ac() => {
                for a in args.iter() {
                    let matches = match (filter, a) {
                        (None, _) => true,
                        (Some(sym), Term::App(s, _)) => s == sym,
                        (Some(_), Term::Lit(_)) => false,
                    };
                    if !matches {
                        continue;
                    }
                    if rest.is_empty() {
                        out.insert(a);
                    } else {
                        collect_subterms_at(a, rest, out);
                    }
                }
            }
            _ => {}
        }
    }

    /// $\mathit{LitPos}(t, p)$: the set of literals among $t|_p$.
    pub fn lit_pos(t: &LNTerm, p: &[PosStep]) -> BTreeSet<LNLit> {
        subterms_at(t, p)
            .into_iter()
            .filter_map(|s| match s {
                Term::Lit(l) => Some(l.clone()),
                Term::App(..) => None,
            })
            .collect()
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::builtin::xor;
        use crate::function_symbols::{Constructability, NoEqSym, Privacy};
        use crate::lterm::LSort;
        use crate::lterm::LVar;
        use crate::term::f_app_no_eq;
        use crate::vterm::var_term;

        fn v(name: &str) -> LNTerm {
            var_term(LVar::new(name, LSort::Msg, 0))
        }

        fn no_eq_sym(name: &'static str, arity: usize) -> NoEqSym {
            NoEqSym::new(name, arity, Privacy::Public, Constructability::Constructor)
        }

        /// \Cref{ex:positions}: $t = f(\oplus(a, b, g(c, d)), d)$, all `msg`.
        /// Returns `(t, FunSym::NoEq(g))` so tests can name the `g`-selector.
        fn example_term() -> LNTerm {
            let g = no_eq_sym("g", 2);
            let d = v("d");
            let inner_g = f_app_no_eq(g, vec![v("c"), d.clone()]);
            let group = xor(xor(v("a"), v("b")), inner_g); // flattens to {a, b, g(c,d)}
            let f = no_eq_sym("f", 2);
            f_app_no_eq(f, vec![group, d])
        }

        /// t = f(\oplus(a, b, g(c, d), g(e, f)), d), all `msg`.
        fn example_term2() -> LNTerm {
            let g = no_eq_sym("g", 2);
            let d = v("d");
            let inner_g1 = f_app_no_eq(g, vec![v("c"), d.clone()]);
            let inner_g2 = f_app_no_eq(g, vec![v("e"), v("f")]);
            let group = xor(xor(v("a"), v("b")), xor(inner_g1, inner_g2)); // flattens to {a, b, g(c,d), g(e,f)}
            let f = no_eq_sym("f", 2);
            f_app_no_eq(f, vec![group, d])
        }

        /// t = f(\oplus(a, b, g(c, d), h(e, f)), d), all `msg`.
        fn example_term3() -> LNTerm {
            let g = no_eq_sym("g", 2);
            let d = v("d");
            let inner_g1 = f_app_no_eq(g, vec![v("c"), d.clone()]);
            let h = no_eq_sym("h", 2);
            let inner_h = f_app_no_eq(h, vec![v("e"), v("f")]);
            let group = xor(xor(v("a"), v("b")), xor(inner_g1, inner_h)); // flattens to {a, b, g(c,d), h(e,f)}
            let f = no_eq_sym("f", 2);
            f_app_no_eq(f, vec![group, d])
        }

        /// List function symbol should work as expected when used at the top-level
        fn example_term_list() -> LNTerm {
            let g = no_eq_sym("g", 2);
            let d = v("d");
            let inner_g = f_app_no_eq(g, vec![v("c"), d.clone()]);
            let group = xor(xor(v("a"), v("b")), inner_g); // flattens to {a, b, g(c,d)}
            Term::App(FunSym::List, vec![group, d].into())
        }

        #[test]
        fn positions_match_example() {
            let t = example_term();
            let g = FunSym::NoEq(no_eq_sym("g", 2));
            let got: BTreeSet<_> = positions(&t).into_iter().collect();
            let want: BTreeSet<Position> = [
                vec![],
                vec![PosStep::Arg(0)],
                vec![PosStep::Arg(0), PosStep::AcGroup(None)],
                vec![PosStep::Arg(0), PosStep::AcGroup(Some(g))],
                vec![PosStep::Arg(0), PosStep::AcGroup(Some(g)), PosStep::Arg(0)],
                vec![PosStep::Arg(0), PosStep::AcGroup(Some(g)), PosStep::Arg(1)],
                vec![PosStep::Arg(1)],
            ]
            .into_iter()
            .collect();
            assert_eq!(got, want);
        }

        #[test]
        fn positions_of_example_term_match_example_term_with_list_head() {
            let t = example_term();
            let t_list = example_term_list();
            let got: BTreeSet<_> = positions(&t).into_iter().collect();
            let want: BTreeSet<Position> = positions(&t_list).into_iter().collect();
            assert_eq!(got, want);
        }

        #[test]
        fn lit_pos_of_ac_group_is_a_and_b() {
            let t = example_term();
            let p = [PosStep::Arg(0), PosStep::AcGroup(None)];
            assert_eq!(
                lit_pos(&t, &p),
                BTreeSet::from([lit(&v("a")), lit(&v("b"))])
            );
        }

        #[test]
        fn lit_pos_of_c_and_nested_d() {
            let t = example_term();
            let g = FunSym::NoEq(no_eq_sym("g", 2));
            let c_pos = [PosStep::Arg(0), PosStep::AcGroup(Some(g)), PosStep::Arg(0)];
            let d_pos = [PosStep::Arg(0), PosStep::AcGroup(Some(g)), PosStep::Arg(1)];
            assert_eq!(lit_pos(&t, &c_pos), BTreeSet::from([lit(&v("c"))]));
            assert_eq!(lit_pos(&t, &d_pos), BTreeSet::from([lit(&v("d"))]));
        }

        #[test]
        fn lit_pos_of_outer_d_is_the_same_literal() {
            let t = example_term();
            let outer_d = [PosStep::Arg(1)];
            assert_eq!(lit_pos(&t, &outer_d), BTreeSet::from([lit(&v("d"))]));
        }

        #[test]
        fn subterms_at_ac_selector_is_the_single_matching_child() {
            let t = example_term();
            let g = no_eq_sym("g", 2);
            let inner_g = f_app_no_eq(g, vec![v("c"), v("d")]);
            let g = FunSym::NoEq(g);
            let p = [PosStep::Arg(0), PosStep::AcGroup(Some(g))];
            assert_eq!(subterms_at(&t, &p), BTreeSet::from([&inner_g]));
        }

        #[test]
        fn subterms_at_ac_selector_has_both_matching_childs() {
            let t = example_term2();
            let g = no_eq_sym("g", 2);
            let g1 = f_app_no_eq(g, vec![v("c"), v("d")]);
            let g2 = f_app_no_eq(g, vec![v("e"), v("f")]);
            let g = FunSym::NoEq(g);
            let p = [PosStep::Arg(0), PosStep::AcGroup(Some(g))];
            assert_eq!(subterms_at(&t, &p), BTreeSet::from([&g1, &g2]));
        }

        #[test]
        fn ac_selector_chooses_only_subterms_of_matching_child() {
            let t = example_term3();
            let g = no_eq_sym("g", 2);
            let g_app = f_app_no_eq(g, vec![v("c"), v("d")]);
            let g = FunSym::NoEq(g);
            let p = [PosStep::Arg(0), PosStep::AcGroup(Some(g))];
            assert_eq!(subterms_at(&t, &p), BTreeSet::from([&g_app]));
        }

        /// \Cref{thm:alphaeqac_pos}: $s \alphaeqac t \implies \mathit{Pos}(s)
        /// = \mathit{Pos}(t)$. Renaming every variable of `example_term` to a
        /// fresh (consistently-shared) name must not change its position set.
        #[test]
        fn positions_are_invariant_under_variable_renaming() {
            let t1 = example_term();
            let g = no_eq_sym("g", 2);
            let s = v("s"); // renamed from `d`, shared like `d` was
            let inner_g = f_app_no_eq(g, vec![v("r"), s.clone()]);
            let group = xor(xor(v("p"), v("q")), inner_g);
            let f = no_eq_sym("f", 2);
            let t2 = f_app_no_eq(f, vec![group, s]);

            assert_eq!(
                positions(&t1).into_iter().collect::<BTreeSet<_>>(),
                positions(&t2).into_iter().collect::<BTreeSet<_>>()
            );
        }

        /// Same theorem, exercised by commuting `xor`'s (AC) arguments and
        /// nesting instead of renaming: `f_app_ac` flattens and sorts both
        /// constructions to the same shape, so the position set is unchanged.
        #[test]
        fn positions_are_invariant_under_ac_commutation() {
            let t1 = example_term();
            let g = no_eq_sym("g", 2);
            let d = v("d");
            let inner_g = f_app_no_eq(g, vec![v("c"), d.clone()]);
            let group = xor(inner_g, xor(v("b"), v("a"))); // swapped order/nesting
            let f = no_eq_sym("f", 2);
            let t2 = f_app_no_eq(f, vec![group, d]);

            assert_eq!(
                positions(&t1).into_iter().collect::<BTreeSet<_>>(),
                positions(&t2).into_iter().collect::<BTreeSet<_>>()
            );
        }

        /// Both at once, on `example_term2` (two `g`-headed AC children):
        /// fresh variable names throughout, plus a different AC nesting and
        /// argument order for both the outer `xor` and the two `g`-children.
        #[test]
        fn positions_are_invariant_under_renaming_and_ac_commutation() {
            let t1 = example_term2();
            let g = no_eq_sym("g", 2);
            let s = v("s"); // renamed from `d`, shared like `d` was
            let g1 = f_app_no_eq(g, vec![v("p"), v("q")]);
            let g2 = f_app_no_eq(g, vec![v("r"), s.clone()]);
            let group = xor(xor(g2, g1), xor(v("t"), v("u")));
            let f = no_eq_sym("f", 2);
            let t2 = f_app_no_eq(f, vec![group, s]);

            assert_eq!(
                positions(&t1).into_iter().collect::<BTreeSet<_>>(),
                positions(&t2).into_iter().collect::<BTreeSet<_>>()
            );
        }

        #[test]
        fn invalid_position_is_empty() {
            let t = example_term();
            // `Arg` does not apply directly to an AC node.
            let p = [PosStep::Arg(0), PosStep::Arg(0)];
            assert!(subterms_at(&t, &p).is_empty());
        }

        /// Extracts the literal out of a term built by [`v`] (a bare variable).
        fn lit(t: &LNTerm) -> LNLit {
            match t {
                Term::Lit(l) => l.clone(),
                Term::App(..) => panic!("not a literal"),
            }
        }
    }
}

use position::{PosStep, Position};

/// Struct modeling a bucket with `count` uncanonized literals of `sort`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BucketKey {
    /// The number of uncanonized literals at a position.
    count: usize,
    /// The sort of the uncanonized literals counted in this bucket.
    sort: LSort,
}

impl PartialOrd for BucketKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BucketKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.count
            .cmp(&other.count)
            .then_with(|| self.sort.cmp(&other.sort))
    }
}

/// The dynamic scheduling structure driving Algorithm 1 (`CAN_alphaeqac`):
/// a bucket queue keyed by how many literals at a position are still
/// uncanonized, i.e. work.tex's `[(LitPos(p), p) | p in Pos(t)]` worklist
/// sorted by "1) number of literals left to be canonized ... 2)
/// lexicographic order on positions". Positions and literal sets are
/// static once computed from `t`; only which literals are still
/// uncanonized changes as the algorithm progresses, so canonizing one
/// literal only needs to relocate the (typically few) positions it
/// actually occurs in, via `occurs_in`, instead of re-sorting every entry.
struct Canonizer {
    /// The term to canonize
    term: LNTerm,
    /// The current canonical labelling of literals, i.e., a sort-respeting bijection from the literals of `term` to a canonical set of fresh literals.
    subst: Subst<LNLit, LNLit>,

    /// The next fresh literal of the msg sort to be used in the canonical labelling.
    fresh_msg: u64,
    /// The next fresh literal of the pub sort to be used in the canonical labelling.
    fresh_pub: u64,
    /// The next fresh literal of the fresh sort to be used in the canonical labelling.
    fresh_fresh: u64,
    /// The next fresh literal of the nat sort to be used in the canonical labelling.
    fresh_nat: u64,
    /// The next fresh literal of the node sort to be used in the canonical labelling.
    fresh_node: u64,
    /// `(count, sort) -> positions currently having exactly that many
    /// uncanonized literals of that sort`, in ascending key order. A
    /// `BTreeMap` rather than an array of buckets so the smallest nonempty
    /// count is a single lookup regardless of how wide any single AC
    /// application's arity is, and so a position's count can drop by more
    /// than one bucket in a single step (when several of its literals are
    /// canonized together) without breaking a monotonic-cursor assumption.
    buckets: BTreeMap<BucketKey, BTreeSet<Position>>,
    /// The literals still uncanonized at each position currently tracked
    /// in `buckets`, i.e. work.tex's `LitPos(t,p) \ dom(\theta)`.
    remaining: HashMap<Position, BTreeSet<LNLit>>,
    /// Inverted index: the positions a literal occurs in, so canonizing it
    /// only touches those entries. Static once built.
    occurs_in: HashMap<LNLit, Vec<Position>>,
}

impl Canonizer {
    /// Builds the initial worklist for `t`: every position of `t` paired
    /// with its literal set, bucketed by set size and indexed by literal.
    /// Positions with no literals (e.g. an AC-group position selecting only
    /// compound subterms) are omitted, since they are never popped.
    fn new(t: &LNTerm) -> Self {
        Self::new_inner(t, Subst::empty())
    }

    fn new_with_subst(t: &LNTerm, subst: Subst<LNLit, LNLit>) -> Self {
        Self::new_inner(t, subst)
    }

    fn new_inner(t: &LNTerm, subst: Subst<LNLit, LNLit>) -> Self {
        let mut buckets: BTreeMap<BucketKey, BTreeSet<Position>> = BTreeMap::new();
        let mut remaining: HashMap<Position, BTreeSet<LNLit>> = HashMap::new();
        let mut occurs_in: HashMap<LNLit, Vec<Position>> = HashMap::new();

        for p in position::positions(t) {
            let lits = position::lit_pos(t, &p);
            if lits.is_empty() {
                continue;
            }
            for l in &lits {
                occurs_in.entry(l.clone()).or_default().push(p.clone());
            }
            let uncanonicalized_lits: BTreeSet<_> = lits
                .into_iter()
                .filter(|l| !subst.contains_var(l))
                .collect();
            let mut sort_counts: BTreeMap<LSort, usize> = BTreeMap::new();
            for l in &uncanonicalized_lits {
                *sort_counts.entry(l.sort()).or_default() += 1;
            }
            for (sort, count) in sort_counts {
                buckets
                    .entry(BucketKey { count, sort })
                    .or_default()
                    .insert(p.clone());
            }
            remaining.insert(p, uncanonicalized_lits);
        }

        Canonizer {
            term: t.clone(),
            fresh_fresh: 0,
            fresh_msg: 0,
            fresh_pub: 0,
            fresh_nat: 0,
            fresh_node: 0,
            subst,
            buckets,
            remaining,
            occurs_in,
        }
    }

    // Computes the canonical form of `self.term` under the given canonicalizing substitution, returning the canonicalized term and the final substitution.
    fn canonize(&mut self) -> (LNTerm, Subst<LNLit, LNLit>) {
        unimplemented!()
    }
}

/// Canonize `t` with respect to $\alphaeqac$: two terms are $\alphaeqac$ iff
/// their canonical forms are syntactically equal (`thm:can_alphaeqac_can2`).
pub fn canonicalize_alpha_eq_ac(_t: &LNTerm, _subst: Subst<LNLit, LNLit>) -> LNTerm {
    unimplemented!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtin::{emap, xor};
    use crate::function_symbols::{Constructability, NoEqSym, Privacy};
    use crate::lterm::{fresh_term, LSort, LVar};
    use crate::term::{f_app_no_eq, Term};
    use crate::vterm::var_term;

    /// A variable literal of the given sort, as an `LNTerm`.
    fn v(name: &str, sort: LSort) -> LNTerm {
        var_term(LVar::new(name, sort, 0))
    }

    /// A fresh, public, arity-`n` NoEq symbol for use as an uninterpreted
    /// function symbol in tests (no relation to any builtin).
    fn no_eq_sym(name: &'static str, arity: usize) -> NoEqSym {
        NoEqSym::new(name, arity, Privacy::Public, Constructability::Constructor)
    }

    // -- 1) NoEq-only terms: renaming variables of the same sort is alpha
    //    equivalent, no AC involved. --------------------------------------
    #[test]
    #[ignore = "canonicalize_alpha_eq_ac not yet implemented"]
    fn noeq_vars_renamed_are_alpha_eq() {
        let f = no_eq_sym("f", 2);
        let t1 = f_app_no_eq(
            f,
            vec![
                v("x", LSort::Msg),
                v("y", LSort::Pub),
                v("z", LSort::Fresh),
                v("w", LSort::Nat),
                v("u", LSort::Node),
            ],
        );
        let t2 = f_app_no_eq(
            f,
            vec![
                v("a", LSort::Msg),
                v("b", LSort::Pub),
                v("c", LSort::Fresh),
                v("d", LSort::Nat),
                v("u", LSort::Node),
            ],
        );
        assert_eq!(
            canonicalize_alpha_eq_ac(&t1, Subst::empty()),
            canonicalize_alpha_eq_ac(&t2, Subst::empty())
        );
    }

    // -- 2) Same as 1) but the outer symbol is AC (`xor`, work.tex's
    //    `\oplus`). Cf. \Cref{ex:ex1canon}: `xor(a,b)` and `xor(x,y)` (all
    //    `msg`) are $\alphaeqac$ via `{a -> y, b -> x}`. ---------------------
    #[test]
    #[ignore = "canonicalize_alpha_eq_ac not yet implemented"]
    fn ac_vars_renamed_are_alpha_eq() {
        let t1 = xor(v("a", LSort::Msg), v("b", LSort::Msg));
        let t2 = xor(v("x", LSort::Msg), v("y", LSort::Msg));
        assert_eq!(
            canonicalize_alpha_eq_ac(&t1, Subst::empty()),
            canonicalize_alpha_eq_ac(&t2, Subst::empty())
        );
    }

    // -- 3) Same as 1) but the outer symbol is C (`emap`), i.e. commutative
    //    but not associative. ----------------------------------------------
    #[test]
    #[ignore = "canonicalize_alpha_eq_ac not yet implemented"]
    fn c_vars_renamed_are_alpha_eq() {
        let t1 = emap(v("x", LSort::Msg), v("y", LSort::Msg));
        let t2 = emap(v("a", LSort::Msg), v("b", LSort::Msg));
        assert_eq!(
            canonicalize_alpha_eq_ac(&t1, Subst::empty()),
            canonicalize_alpha_eq_ac(&t2, Subst::empty())
        );
    }

    // -- 4) NoEq: same shape as 1) but the sort at each position differs
    //    between the two terms, so no sort-respecting renaming can make them
    //    equal — positions matter for a non-AC symbol. ----------------------
    #[test]
    #[ignore = "canonicalize_alpha_eq_ac not yet implemented"]
    fn noeq_position_sort_mismatch_not_alpha_eq() {
        let f = no_eq_sym("f", 2);
        let t1 = f_app_no_eq(f, vec![v("x", LSort::Msg), v("y", LSort::Fresh)]);
        let t2 = f_app_no_eq(f, vec![v("a", LSort::Fresh), v("b", LSort::Msg)]);
        assert_ne!(
            canonicalize_alpha_eq_ac(&t1, Subst::empty()),
            canonicalize_alpha_eq_ac(&t2, Subst::empty())
        );
    }

    // -- 5) AC: swapping argument ORDER never breaks $\alphaeqac$ (that is
    //    the whole point of AC canonization), so the mismatch has to come
    //    from a different MULTISET of sorts: `{msg, msg}` vs `{msg, fresh}`.
    #[test]
    #[ignore = "canonicalize_alpha_eq_ac not yet implemented"]
    fn ac_sort_multiset_mismatch_not_alpha_eq() {
        let t1 = xor(v("x", LSort::Msg), v("y", LSort::Msg));
        let t2 = xor(v("a", LSort::Msg), v("b", LSort::Fresh));
        assert_ne!(
            canonicalize_alpha_eq_ac(&t1, Subst::empty()),
            canonicalize_alpha_eq_ac(&t2, Subst::empty())
        );
    }

    // -- 6) Same reasoning as 5) for a C symbol. -----------------------------
    #[test]
    #[ignore = "canonicalize_alpha_eq_ac not yet implemented"]
    fn c_sort_multiset_mismatch_not_alpha_eq() {
        let t1 = emap(v("x", LSort::Msg), v("y", LSort::Msg));
        let t2 = emap(v("a", LSort::Msg), v("b", LSort::Fresh));
        assert_ne!(
            canonicalize_alpha_eq_ac(&t1, Subst::empty()),
            canonicalize_alpha_eq_ac(&t2, Subst::empty())
        );
    }

    // -- 7) Idempotence (\Cref{can1} / \Cref{thm:can_alphaeqac_can1}), on the
    //    worked AC example \Cref{ex:ac_canon}: `f(xor(c,d), xor(a,c), a)`.
    #[test]
    #[ignore = "canonicalize_alpha_eq_ac not yet implemented"]
    fn canon_is_idempotent() {
        let f = no_eq_sym("f", 3);
        let a = v("a", LSort::Msg);
        let c = v("c", LSort::Msg);
        let d = v("d", LSort::Msg);
        let t = f_app_no_eq(f, vec![xor(c.clone(), d), xor(a.clone(), c), a]);
        let once = canonicalize_alpha_eq_ac(&t, Subst::empty());
        let twice = canonicalize_alpha_eq_ac(&once, Subst::empty());
        assert_eq!(once, twice);
    }

    // -- 8) \Cref{ex:wrong_ac_canon}: `f(xor(a,b), a)` and `f(xor(x,y), y)`
    //    (all `msg`) are $\alphaeqac$ via `{a -> y, b -> x}`, but a naive
    //    canonizer that renames literals under AC symbols independently of
    //    their occurrences elsewhere gets this wrong (see the chapter
    //    discussion right after the example). This regression-tests that
    //    our implementation propagates the renaming of `a` correctly.
    #[test]
    #[ignore = "canonicalize_alpha_eq_ac not yet implemented"]
    fn wrong_ac_canon_example_propagates_shared_literal() {
        let f = no_eq_sym("f", 2);
        let a = v("a", LSort::Msg);
        let b = v("b", LSort::Msg);
        let x = v("x", LSort::Msg);
        let y = v("y", LSort::Msg);
        let t1 = f_app_no_eq(f, vec![xor(a.clone(), b), a]);
        let t2 = f_app_no_eq(f, vec![xor(x, y.clone()), y]);
        assert_eq!(
            canonicalize_alpha_eq_ac(&t1, Subst::empty()),
            canonicalize_alpha_eq_ac(&t2, Subst::empty())
        );
    }

    // -- 9) AC associativity/flattening interacting with renaming: two
    //    different nestings of the same three (differently named) literals
    //    under `xor` flatten to the same multiset and must canonize equal.
    #[test]
    #[ignore = "canonicalize_alpha_eq_ac not yet implemented"]
    fn ac_associativity_flattening_alpha_eq() {
        let t1 = xor(
            xor(v("a", LSort::Msg), v("b", LSort::Msg)),
            v("c", LSort::Msg),
        );
        let t2 = xor(
            v("x", LSort::Msg),
            xor(v("y", LSort::Msg), v("z", LSort::Msg)),
        );
        assert_eq!(
            canonicalize_alpha_eq_ac(&t1, Subst::empty()),
            canonicalize_alpha_eq_ac(&t2, Subst::empty())
        );
    }

    // -- 10) Literals are names ($\mathcal{N}$) as well as variables
    //    ($\mathcal{V}$); renaming a fresh NAME constant must be handled the
    //    same way as renaming a variable.
    #[test]
    #[ignore = "canonicalize_alpha_eq_ac not yet implemented"]
    fn fresh_names_renamed_are_alpha_eq() {
        let f = no_eq_sym("f", 1);
        let t1 = f_app_no_eq(f, vec![fresh_term("n")]);
        let t2 = f_app_no_eq(f, vec![fresh_term("m")]);
        assert_eq!(
            canonicalize_alpha_eq_ac(&t1, Subst::empty()),
            canonicalize_alpha_eq_ac(&t2, Subst::empty())
        );
    }

    // -- 11) A name and a variable of the SAME sort are not interchangeable:
    //    a sort-respecting renaming never maps a name to a variable or vice
    //    versa (the chapter's naming scheme reserves disjoint canonical
    //    families, e.g. `fn_i` for fresh names vs `fv_i` for fresh
    //    variables), so the two terms below must canonize differently.
    #[test]
    #[ignore = "canonicalize_alpha_eq_ac not yet implemented"]
    fn name_and_variable_of_same_sort_not_interchangeable() {
        let f = no_eq_sym("f", 1);
        let t1 = f_app_no_eq(f, vec![fresh_term("n")]);
        let t2 = f_app_no_eq(f, vec![v("x", LSort::Fresh)]);
        assert_ne!(
            canonicalize_alpha_eq_ac(&t1, Subst::empty()),
            canonicalize_alpha_eq_ac(&t2, Subst::empty())
        );
    }

    // -- 12) AC: the same MULTISET of sorts at swapped argument positions —
    //    i.e. commutativity itself, not just a per-position rename — must
    //    still canonize equal: `xor(a:msg, b:pub) ~ xor(f:pub, g:msg)`.
    #[test]
    #[ignore = "canonicalize_alpha_eq_ac not yet implemented"]
    fn ac_swap_position_with_matching_sorts_alpha_eq() {
        let t1 = xor(v("a", LSort::Msg), v("b", LSort::Pub));
        let t2 = xor(v("f", LSort::Pub), v("g", LSort::Msg));
        assert_eq!(
            canonicalize_alpha_eq_ac(&t1, Subst::empty()),
            canonicalize_alpha_eq_ac(&t2, Subst::empty())
        );
    }

    // -- 13) Same commutativity property for a C symbol (`emap`). -----------
    #[test]
    #[ignore = "canonicalize_alpha_eq_ac not yet implemented"]
    fn c_swap_position_with_matching_sorts_alpha_eq() {
        let t1 = emap(v("a", LSort::Msg), v("b", LSort::Pub));
        let t2 = emap(v("f", LSort::Pub), v("g", LSort::Msg));
        assert_eq!(
            canonicalize_alpha_eq_ac(&t1, Subst::empty()),
            canonicalize_alpha_eq_ac(&t2, Subst::empty())
        );
    }

    // -- 14) An AC term covering all five `LSort`s at once, permuted in both
    //    argument order and literal naming between `t1` and `t2`.
    #[test]
    #[ignore = "canonicalize_alpha_eq_ac not yet implemented"]
    fn ac_all_sorts_alpha_eq() {
        use crate::function_symbols::AcSym;
        use crate::term::f_app_ac;

        let t1 = f_app_ac(
            AcSym::Xor,
            vec![
                v("a", LSort::Msg),
                v("b", LSort::Pub),
                v("c", LSort::Fresh),
                v("d", LSort::Nat),
                v("e", LSort::Node),
            ],
        );
        let t2 = f_app_ac(
            AcSym::Xor,
            vec![
                v("v", LSort::Node),
                v("w", LSort::Nat),
                v("x", LSort::Fresh),
                v("y", LSort::Pub),
                v("z", LSort::Msg),
            ],
        );
        assert_eq!(
            canonicalize_alpha_eq_ac(&t1, Subst::empty()),
            canonicalize_alpha_eq_ac(&t2, Subst::empty())
        );
    }

    /// Extracts the literal out of a term built by [`v`] (a bare variable).
    fn lit_of(t: &LNTerm) -> LNLit {
        match t {
            Term::Lit(l) => l.clone(),
            Term::App(..) => panic!("not a literal"),
        }
    }

    // -- 15) `Worklist::new`: a single AC group of two literals goes into
    //    one bucket, keyed by its size, and both literals are indexed back
    //    to that one position. -----------------------------------------------
    #[test]
    fn worklist_new_buckets_by_literal_count() {
        let t = xor(v("a", LSort::Msg), v("b", LSort::Msg));
        let wl = Canonizer::new(&t);

        let key = BucketKey {
            count: 2,
            sort: LSort::Msg,
        };
        assert_eq!(wl.buckets.keys().copied().collect::<Vec<_>>(), vec![key]);
        let group_pos = wl.buckets[&key].iter().next().unwrap().clone();
        assert_eq!(wl.remaining[&group_pos].len(), 2);
        assert_eq!(
            wl.occurs_in[&lit_of(&v("a", LSort::Msg))],
            vec![group_pos.clone()]
        );
        assert_eq!(wl.occurs_in[&lit_of(&v("b", LSort::Msg))], vec![group_pos]);
    }

    #[test]
    fn worklist_new_buckets_by_literal_count_per_sort() {
        let t = xor(
            xor(v("a", LSort::Msg), v("b", LSort::Pub)),
            v("c", LSort::Fresh),
        );
        let wl = Canonizer::new(&t);

        assert_eq!(wl.buckets.len(), 3);
        for sort in [LSort::Msg, LSort::Pub, LSort::Fresh] {
            assert!(wl.buckets.contains_key(&BucketKey { count: 1, sort }));
        }

        let group_pos = vec![PosStep::AcGroup(None)];
        for positions in wl.buckets.values() {
            assert_eq!(positions, &BTreeSet::from([group_pos.clone()]));
        }
        assert_eq!(wl.remaining[&group_pos].len(), 3);
    }

    // -- 16) A literal shared across two positions (\Cref{ex:ac_canon}'s
    //    `f(xor(c,d), xor(a,c), a)`) is indexed under both. --------------------
    #[test]
    fn worklist_new_indexes_shared_literal_across_positions() {
        let f = no_eq_sym("f", 3);
        let a = v("a", LSort::Msg);
        let c = v("c", LSort::Msg);
        let d = v("d", LSort::Msg);
        let t = f_app_no_eq(
            f,
            vec![xor(c.clone(), d), xor(a.clone(), c.clone()), a.clone()],
        );

        let wl = Canonizer::new(&t);

        // `c` occurs under the first `xor`'s AC group (with `d`) and the
        // second `xor`'s AC group (with `a`).
        let c_positions = vec![
            vec![PosStep::Arg(0), PosStep::AcGroup(None)],
            vec![PosStep::Arg(1), PosStep::AcGroup(None)],
        ];
        // `a` occurs under the second `xor`'s AC group and bare as `f`'s
        // third argument.
        let a_positions = vec![
            vec![PosStep::Arg(1), PosStep::AcGroup(None)],
            vec![PosStep::Arg(2)],
        ];

        assert_eq!(wl.occurs_in[&lit_of(&c)], c_positions);
        assert_eq!(wl.occurs_in[&lit_of(&a)], a_positions);
    }
}
