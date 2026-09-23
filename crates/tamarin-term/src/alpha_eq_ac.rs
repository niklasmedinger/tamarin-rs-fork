//! Canonization of terms for alpha equivalence modulo AC ($\alphaeqac$), as
//! developed in `work.tex` §"Canonization of Constraint Systems" /
//! "Canonization of Terms".
//!
//! `CAN_alphaeqac` computes a canonical representative for a term's
//! $\alphaeqac$-equivalence class by 1) canonically renaming its literals
//! (variables and names) by sort, delaying the renaming of literals that
//! only occur under AC/C symbols until no permutation-free choice remains,
//! and 2) bringing the result into AC-normal form. See Algorithm 1
//! (`CAN_alphaeqac`) and Theorem `thm:can_alphaeqac` in `work.tex`.
//!
//! This module implements the base, TERM-level case; lifting to facts,
//! rule instances, guarded formulas, whole graph parts, and finally whole
//! constraint systems (`tamarin_theory::canon`'s `canonicalize_fact`/
//! `canonicalize_rule`/`canonicalize_guarded`/`canonicalize_graph_part`/
//! `canonicalize_constraint_system`) all build on the accumulating
//! `CanonLabelling`/`canonicalize_alpha_eq_ac_seeded` entry point below.
use crate::{
    lterm::{LNTerm, LSort, LVar, Name, NameTag},
    term::{f_app, Term},
    vterm::Lit,
};

use itertools::Itertools;
#[allow(clippy::disallowed_types)]
use std::collections::{BTreeMap, BTreeSet};
use tamarin_utils::fresh::FastFreshState;

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
    #[repr(u32)]
    // Explicit repr and descriminants to ensure that positions of non-AC symbols
    // are _less_ than positions of AC symbols
    pub enum PosStep {
        /// `p = f \cdot i \cdot p'`: the `i`-th argument of a fixed-order
        /// (non-AC) application.
        Arg(usize) = 0,
        /// Crossing an AC application: `None` is `p = f \cdot \epsilon`
        /// (select the whole flattened argument multiset — always the
        /// last step of a position); `Some(sym)` is `p = f \cdot g \cdot p'`
        /// (select the flattened arguments headed by `sym`, then continue).
        ///
        /// We do not have an own variant for C symbols because their positions
        /// behave just as AC symbols' positions do. The only difference is that
        /// C symbols are not flattened.
        AcGroup(Option<FunSym>) = 1,
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
            Term::App(FunSym::Ac(_), args) | Term::App(FunSym::C(_), args) => {
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
            (PosStep::AcGroup(filter), Term::App(f, args)) if f.is_ac() || f.is_c() => {
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

        /// Put another list on top. Everything should still work as expected.
        fn example_term_double_list_head() -> LNTerm {
            Term::App(FunSym::List, vec![example_term()].into())
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
        fn positions_double_list_head_example_term() {
            let t = example_term_double_list_head();
            let g = FunSym::NoEq(no_eq_sym("g", 2));
            let got: BTreeSet<_> = positions(&t).into_iter().collect();
            let want: BTreeSet<Position> = [
                vec![],
                vec![PosStep::Arg(0)],
                vec![PosStep::Arg(0), PosStep::Arg(0)],
                vec![PosStep::Arg(0), PosStep::Arg(0), PosStep::AcGroup(None)],
                vec![PosStep::Arg(0), PosStep::Arg(0), PosStep::AcGroup(Some(g))],
                vec![
                    PosStep::Arg(0),
                    PosStep::Arg(0),
                    PosStep::AcGroup(Some(g)),
                    PosStep::Arg(0),
                ],
                vec![
                    PosStep::Arg(0),
                    PosStep::Arg(0),
                    PosStep::AcGroup(Some(g)),
                    PosStep::Arg(1),
                ],
                vec![PosStep::Arg(0), PosStep::Arg(1)],
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

use position::Position;

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

/// One [`FastFreshState`] counter per [`LSort`]. [`Canonizer`] holds TWO
/// of these -- `fresh_names` and `fresh_vars` -- kept fully independent
/// even for the same sort: a canonical name (`fn_i`) and a canonical
/// variable (`fv_i`) never collide as literal strings, but sharing one
/// counter between them (the original design) meant a batch's name count
/// could shift where its variable indices start, and vice versa --
/// canonizing the same sort's variables could land on different indices
/// purely because of how many names happened to be canonized alongside
/// them in the same [`Canonizer::canonize_literals`] call. Splitting the
/// supplies makes each kind's indices depend only on that kind's own
/// canonization history.
#[derive(Debug, Clone, Copy, Default)]
struct PerSortFresh {
    msg: FastFreshState,
    pub_: FastFreshState,
    fresh: FastFreshState,
    nat: FastFreshState,
    node: FastFreshState,
}

impl PerSortFresh {
    /// All five sorts seeded from the same starting state -- mirrors
    /// [`CanonLabelling::from_theta`]'s choice to seed every SORT from
    /// one shared, conservative bound rather than computing a tighter
    /// per-sort one. Unaffected by the name/var split above, which only
    /// separates NAME supplies from VARIABLE supplies, not one sort's
    /// supply from another's.
    fn uniform(state: FastFreshState) -> Self {
        PerSortFresh {
            msg: state,
            pub_: state,
            fresh: state,
            nat: state,
            node: state,
        }
    }

    fn get_mut(&mut self, sort: LSort) -> &mut FastFreshState {
        match sort {
            LSort::Msg => &mut self.msg,
            LSort::Pub => &mut self.pub_,
            LSort::Fresh => &mut self.fresh,
            LSort::Nat => &mut self.nat,
            LSort::Node => &mut self.node,
        }
    }
}

/// An accumulating canonical labelling, threaded across a SEQUENCE of
/// [`canonicalize_alpha_eq_ac_seeded`] calls (one per graph vertex, in
/// canonical vertex order -- see `tamarin_theory::canon`'s vertex-level
/// canonization, which lifts this to facts/rule instances/action-formula
/// facts). Bundles the winning `theta` with the [`PerSortFresh`] cursors
/// live at the point it was produced.
///
/// **Why a bundle, not just `theta`**: a canonical NAME literal's index
/// is baked into its display string with no separate numeric field (see
/// `canonical_name`'s own doc comment) — `crate::lterm::avoid` only ever
/// sees `Lit::Var`, so reconstructing "where to resume allocating fresh
/// indices" purely from `theta`'s rendered image can silently miss an
/// already-used canonical name and later re-derive it for a different
/// original identifier, breaking injectivity of the accumulated
/// labelling (the exact bug `avoid_names`/`avoid_vars` were written to
/// fix). Carrying the cursors forward explicitly — mutate-in-place
/// across calls, never reconstructed — sidesteps that class of bug
/// entirely rather than just patching this one instance of it.
///
/// `Clone` because a caller minimizing over several candidates (e.g. a
/// future automorphism search) needs each candidate to fork its own
/// independent accumulator rather than share mutable state.
#[derive(Debug, Clone, Default)]
pub struct CanonLabelling {
    theta: BTreeMap<LNLit, LNLit>,
    fresh_names: PerSortFresh,
    fresh_vars: PerSortFresh,
}

impl CanonLabelling {
    /// The starting point for a fresh accumulation: no literal fixed yet,
    /// every fresh-index cursor at 0.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Builds a labelling from a THETA OBTAINED SOME OTHER WAY (not
    /// accumulated incrementally via this type) — e.g. one already
    /// computed by a caller that doesn't (yet) thread a `CanonLabelling`
    /// through its own pipeline. Re-derives safe fresh-index cursors from
    /// `theta`'s image via [`avoid_names`]/[`avoid_vars`], the one
    /// remaining use for those two functions now that the primary path
    /// (`empty` + repeated [`canonicalize_alpha_eq_ac_seeded`] calls)
    /// tracks its own cursors explicitly and never needs to reconstruct
    /// them.
    pub fn from_theta(theta: BTreeMap<LNLit, LNLit>) -> Self {
        let fresh_names = PerSortFresh::uniform(avoid_names(theta.values()));
        let fresh_vars = PerSortFresh::uniform(avoid_vars(theta.values()));
        CanonLabelling {
            theta,
            fresh_names,
            fresh_vars,
        }
    }

    /// The literal-to-literal renaming accumulated so far.
    pub fn theta(&self) -> &BTreeMap<LNLit, LNLit> {
        &self.theta
    }

    /// Consumes the labelling, keeping only the accumulated renaming --
    /// what a caller that's done accumulating (e.g. about to canonize a
    /// formula via `canon::canonicalize_guarded`, which takes `theta` by
    /// reference and needs nothing else from this type) actually wants.
    pub fn into_theta(self) -> BTreeMap<LNLit, LNLit> {
        self.theta
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
    /// The current candidate canonical labellings of literals, i.e., sort-respeting renamings of the literals of `term` to a canonical set of fresh literals.
    subst: Vec<BTreeMap<LNLit, LNLit>>,

    /// The next fresh index to use for a canonical NAME (`fn_i`/`pn_i`/...)
    /// of each sort -- independent of `fresh_vars` even for the same sort;
    /// see [`PerSortFresh`]'s own doc comment for why.
    fresh_names: PerSortFresh,
    /// The next fresh index to use for a canonical VARIABLE (`mv_i`/`fv_i`/...)
    /// of each sort -- independent of `fresh_names` even for the same sort.
    fresh_vars: PerSortFresh,
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
    remaining: BTreeMap<Position, BTreeSet<LNLit>>,
    /// Inverted index: the positions a literal occurs in, so canonizing it
    /// only touches those entries. Static once built.
    occurs_in: BTreeMap<LNLit, Vec<Position>>,
    /// The number of permutations of the terms literals that were considered during canonization. This is the number of substitutions that were generated and stored in `self.subst`.
    /// This field only exists to support testing and is not used in the canonization algorithm itself.
    considered_permutations: usize,
}

impl Canonizer {
    /// Builds the initial worklist for `t`: every position of `t` paired
    /// with its literal set, bucketed by set size and indexed by literal.
    /// Positions with no literals (e.g. an AC-group position selecting only
    /// compound subterms) are omitted, since they are never popped.
    /// Starts from an empty [`CanonLabelling`] -- no accumulation.
    fn new(t: &LNTerm) -> Self {
        Self::new_with_labelling(t, CanonLabelling::empty())
    }

    /// Like [`Self::new`], but seeded from `labelling`: every literal
    /// already in `labelling.theta` is treated as already-canonized (its
    /// position's uncanonized-literal count excludes it), and the fresh-
    /// index cursors pick up exactly where `labelling` left them -- see
    /// [`CanonLabelling`]'s own doc comment for why this is now a plain
    /// field copy rather than a `crate::lterm::avoid`/`avoid_names`/
    /// `avoid_vars` reconstruction.
    fn new_with_labelling(t: &LNTerm, labelling: CanonLabelling) -> Self {
        let CanonLabelling {
            theta,
            fresh_names,
            fresh_vars,
        } = labelling;

        let mut buckets: BTreeMap<BucketKey, BTreeSet<Position>> = BTreeMap::new();
        let mut remaining: BTreeMap<Position, BTreeSet<LNLit>> = BTreeMap::new();
        let mut occurs_in: BTreeMap<LNLit, Vec<Position>> = BTreeMap::new();

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
                .filter(|l| !theta.contains_key(l))
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
            fresh_names,
            fresh_vars,
            subst: vec![theta],
            buckets,
            remaining,
            occurs_in,
            considered_permutations: 0,
        }
    }

    /// Returns the amount of permutations of the terms literals that were considered during canonization. This is the number of substitutions that were generated and stored in `self.subst`.
    fn considered_permutations(&self) -> usize {
        self.considered_permutations
    }

    /// Get the next literal to be canonized, i.e. the literal of the smallest sort that occurs in the most positions with the fewest remaining uncanonized literals.
    /// Assumes that the returned literals are canonized immediately afterwards and updates internal data structures accordingly.
    fn next_literals(&mut self) -> Option<Vec<LNLit>> {
        let (k, mut candidate_positions) = self.buckets.pop_first()?;
        assert!(
            !candidate_positions.is_empty(),
            "bucket queue invariant violated: empty bucket for key {:?}",
            k
        );

        // SAFETY: Safe to unwrap since !candidate_positions.is_empty() is asserted above.
        let next_pos = candidate_positions.pop_first().unwrap();
        // Put the rest of the bucket back before updating, so the update pass
        // sees every still-bucketed position and can relocate the ones it
        // affects. `next_pos` stays detached: it is fully canonized for
        // `k.sort` and must not re-enter a bucket of that sort.
        if !candidate_positions.is_empty() {
            self.buckets.insert(k, candidate_positions);
        }

        // Get the literals of the next position. Can still contain literals that are not of the sort of the bucket key.
        let next_pos_all_lits = self.remaining.get(&next_pos);
        assert!(
            next_pos_all_lits.is_some(),
            "remaining invariant violated: position {:?} not found",
            next_pos
        );

        // SAFETY: Safe to unwrap since it is asserted to be Some above.
        let next_pos_all_lits = next_pos_all_lits.unwrap();
        // Filter by sort to get the next literal to canonize.
        let next_lits = next_pos_all_lits
            .iter()
            .filter(|l| l.sort() == k.sort)
            .cloned()
            .collect::<Vec<_>>();
        assert!(k.count == next_lits.len(), "bucket queue invariant violated: position {:?} has {} uncanonized literals of sort {:?} but bucket key has count {}", next_pos, next_lits.len(), k.sort, k.count);

        for lit in next_lits.iter() {
            assert!(
                self.occurs_in.contains_key(lit),
                "occurs_in invariant violated: literal {:?} not found",
                lit
            );
            assert!(
                self.occurs_in[lit].contains(&next_pos),
                "occurs_in invariant violated: position {:?} contains lit {:?}
                but it is not in occurs_in[lit] = {:?}",
                next_pos,
                lit,
                self.occurs_in[lit]
            );
            assert!(k.sort == lit.sort(), "bucket queue invariant violated: literal {:?} has sort {:?} but bucket key has sort {:?}", lit, lit.sort(), k.sort);
        }

        self.update_buckets_and_remaining(&next_lits, k.sort, &next_pos);

        Some(next_lits)
    }

    /// Updates the buckets and remaining uncanonized literals after canonizing the given literals.
    /// First, updates the remaining uncanonized literals for each position that contains any of the canonized literals, deleting the position from the map if it has no remaining uncanonized literals.
    /// Then, it updates the buckets of positions by removing the position from its old bucket and inserting it into its new bucket, if it still has remaining uncanonized literals.
    ///
    /// All of `lits` have sort `sort`, so only buckets of that sort change.
    /// `popped` was already detached from its bucket by the caller and is
    /// therefore only updated in `remaining`.
    fn update_buckets_and_remaining(&mut self, lits: &[LNLit], sort: LSort, popped: &Position) {
        // `occurs_in` is what keeps this cheap: a position can only change
        // bucket if it mentions one of the canonized literals, so there is
        // no need to scan the buckets of `sort` for stale entries.
        let affected: BTreeSet<Position> = lits
            .iter()
            .flat_map(|l| self.occurs_in.get(l).into_iter().flatten().cloned())
            .collect();

        for p in affected {
            let remaining = self.remaining.get_mut(&p);
            assert!(
                remaining.is_some(),
                "remaining invariant violated: position {:?} not found",
                p
            );
            // SAFETY: Safe to unwrap since it is asserted to be Some above.
            let remaining = remaining.unwrap();

            // The position's bucket key is a function of `remaining`, so the
            // count before removal identifies the bucket it currently sits in.
            let old_count = remaining.iter().filter(|l| l.sort() == sort).count();
            for lit in lits {
                remaining.remove(lit);
            }
            let new_count = remaining.iter().filter(|l| l.sort() == sort).count();
            let now_empty = remaining.is_empty();
            assert!(
                new_count < old_count,
                "occurs_in invariant violated: position {:?} was reported to contain one of {:?} but none was removed",
                p,
                lits
            );

            if now_empty {
                self.remaining.remove(&p);
            }

            if p != *popped {
                let old_key = BucketKey {
                    count: old_count,
                    sort,
                };
                let old_bucket = self.buckets.get_mut(&old_key);
                assert!(
                    old_bucket.is_some(),
                    "bucket queue invariant violated: no bucket {:?} for position {:?}",
                    old_key,
                    p
                );
                // SAFETY: Safe to unwrap since it is asserted to be Some above.
                let old_bucket = old_bucket.unwrap();
                let was_present = old_bucket.remove(&p);
                assert!(
                    was_present,
                    "bucket queue invariant violated: position {:?} not found in its bucket {:?}",
                    p, old_key
                );
                if old_bucket.is_empty() {
                    self.buckets.remove(&old_key);
                }
            }

            if new_count > 0 {
                self.buckets
                    .entry(BucketKey {
                        count: new_count,
                        sort,
                    })
                    .or_default()
                    .insert(p);
            }
        }
    }

    /// Canonically renames a batch of literals of one sort sharing a
    /// position (as returned by [`Self::next_literals`]): extends every
    /// current candidate renaming in `self.subst` by every permutation of
    /// the batch, per Algorithm 1 (`CAN_alphaeqac`)'s `perms` / crossproduct
    /// step.
    ///
    /// A sort-respecting renaming never turns a name into a variable or vice
    /// versa (`work.tex`'s naming scheme reserves disjoint canonical
    /// families, e.g. `fn_i` vs `fv_i`), so the batch is first split into its
    /// names and variables and permuted independently; the two permutation
    /// sets are then combined via cross product, exactly like `perms(lits)`
    /// would if it only ever permuted within each category.
    fn canonize_literals(&mut self, lits: &[LNLit]) {
        let sort = lits[0].sort();
        for lit in lits.iter() {
            assert!(
                lit.sort() == sort,
                "canonize_literal invariant violated: all lits must have the same sort, but {:?} has sort {:?} while the first has sort {:?}",
                lit,
                lit.sort(),
                sort
            );
        }

        let mut names: Vec<Name> = Vec::new();
        let mut vars: Vec<LVar> = Vec::new();
        for l in lits {
            match l {
                Lit::Con(n) => names.push(*n),
                Lit::Var(v) => vars.push(*v),
            }
        }

        // Names and variables of the same sort draw from INDEPENDENT
        // fresh-index supplies (`fresh_names`/`fresh_vars`): even though
        // the canonical families never collide as literal strings (`fn_i`
        // vs `fv_i`), sharing one counter meant a batch's name count
        // could shift where its variable indices start (and vice versa)
        // -- see `PerSortFresh`'s own doc comment.
        let name_idx = self.allocate_fresh_name_indices(sort, names.len() as u64);
        let canonical_names: Vec<Name> = (0..names.len() as u64)
            .map(|i| canonical_name(sort, name_idx + i))
            .collect();
        let var_idx = self.allocate_fresh_var_indices(sort, vars.len() as u64);
        let canonical_vars: Vec<LVar> = (0..vars.len() as u64)
            .map(|i| canonical_var(sort, var_idx + i))
            .collect();

        let current_substs = std::mem::take(&mut self.subst);
        let mut new_substs =
            Vec::with_capacity(current_substs.len() * names.len().max(1) * vars.len().max(1));
        for name_perm in names.iter().permutations(names.len()) {
            for var_perm in vars.iter().permutations(vars.len()) {
                for existing in &current_substs {
                    let mut ext = existing.clone();
                    for (&orig, canon) in name_perm.iter().zip(canonical_names.iter()) {
                        ext.insert(Lit::Con(*orig), Lit::Con(*canon));
                    }
                    for (&orig, canon) in var_perm.iter().zip(canonical_vars.iter()) {
                        ext.insert(Lit::Var(*orig), Lit::Var(*canon));
                    }
                    new_substs.push(ext);
                }
            }
        }
        self.subst = new_substs;
    }

    /// Allocate `n` fresh NAME indices for `sort`, returning the first
    /// allocated index. Independent of [`Self::allocate_fresh_var_indices`]
    /// even for the same `sort` -- see [`PerSortFresh`]'s own doc comment.
    fn allocate_fresh_name_indices(&mut self, sort: LSort, n: u64) -> u64 {
        self.fresh_names.get_mut(sort).fresh_idents(n)
    }

    /// Allocate `n` fresh VARIABLE indices for `sort`, returning the first
    /// allocated index. Independent of [`Self::allocate_fresh_name_indices`]
    /// even for the same `sort`.
    fn allocate_fresh_var_indices(&mut self, sort: LSort, n: u64) -> u64 {
        self.fresh_vars.get_mut(sort).fresh_idents(n)
    }

    /// Computes the canonical form of `self.term`: drives [`Self::next_literals`]
    /// / [`Self::canonize_literal`] to exhaustion to build every candidate
    /// renaming, then applies each renaming and keeps the lexicographically
    /// smallest result, which is automatically in `CAN_AC` normal form since
    /// [`apply_literal_renaming`] rebuilds through the term's smart
    /// constructors (Algorithm 1, `CAN_alphaeqac`, `work.tex`).
    ///
    /// Returns the winning candidate packaged as a [`CanonLabelling`], not
    /// a bare `theta`: `self.fresh_names`/`self.fresh_vars` at this point
    /// are well-defined regardless of WHICH candidate wins, since
    /// [`Self::allocate_fresh_name_indices`]/[`Self::allocate_fresh_var_indices`]
    /// mutate the SAME shared counters before branching into the
    /// permutation cross-product -- every live candidate consumed
    /// identical index ranges, just assigned to different literals.
    fn canonize(&mut self) -> (LNTerm, CanonLabelling) {
        while let Some(lits) = self.next_literals() {
            self.canonize_literals(&lits);
        }

        let term = self.term.clone();
        let candidates = std::mem::take(&mut self.subst);
        self.considered_permutations = candidates.len();
        let (canon_term, theta) = candidates
            .into_iter()
            .map(|subst| {
                let canon_term = apply_literal_renaming(&term, &subst);
                (canon_term, subst)
            })
            .min_by(|(a, _), (b, _)| a.cmp(b))
            .expect("Canonizer::subst always holds at least one candidate renaming");
        let labelling = CanonLabelling {
            theta,
            fresh_names: self.fresh_names,
            fresh_vars: self.fresh_vars,
        };
        (canon_term, labelling)
    }
}

/// The NAME-side half of [`CanonLabelling::from_theta`]'s reconstruction: a
/// `FastFreshState` that won't generate any index already used by a
/// canonical NAME literal (`Lit::Con`) in `lits`. Paired with
/// [`avoid_vars`] for the variable-side half -- kept as two separate
/// functions (rather than one scanning both kinds at once) now that
/// `Canonizer` keeps independent supplies for names and variables (see
/// [`PerSortFresh`]'s own doc comment): each half only ever needs to see
/// its own kind.
///
/// **Why not just `crate::lterm::avoid`**: that function (and the
/// `HasFrees`/`bounds_var_idx` machinery it's built on) only ever visits
/// `Lit::Var` -- a `Lit::Con(Name)` is a structural no-op for `HasFrees`
/// (see `lterm.rs`'s own `HasFrees for Lit<C, V>` impl: `l @ Lit::Con(_)
/// => l`, no callback). So reconstructing "where to resume allocating
/// fresh indices" from an arbitrary theta's rendered image alone (this
/// function's whole reason for existing) can silently miss a canonical
/// NAME literal already in that image (e.g. `fn7`) -- a later call could
/// then re-derive the same literal name `fn7` for a DIFFERENT original
/// identifier, breaking injectivity. [`CanonLabelling`]'s PRIMARY
/// accumulation path (`empty` + repeated seeded calls) sidesteps this
/// class of bug entirely by never reconstructing in the first place; this
/// function remains only for [`CanonLabelling::from_theta`]'s fallback
/// case, where reconstruction is unavoidable because no cursor history
/// exists to carry forward.
///
/// Computes the max index the same way `avoid`/`bounds_var_idx` do
/// ([`avoid_indices`]: reserve `[0, max+1)` so the next fresh index
/// starts at `max+1`), but reading the numeric suffix [`canonical_name`]
/// bakes into a name literal's string instead of relying on `HasFrees`.
fn avoid_names<'a>(lits: impl IntoIterator<Item = &'a LNLit>) -> FastFreshState {
    avoid_indices(lits.into_iter().filter_map(|l| match l {
        Lit::Con(name) => canonical_name_idx(name),
        Lit::Var(_) => None,
    }))
}

/// The VARIABLE-side half of [`CanonLabelling::from_theta`]'s
/// reconstruction -- the counterpart to [`avoid_names`]. Unlike the name
/// side, an `LVar`'s index is already visible to
/// `crate::lterm::avoid`/`HasFrees`; this exists mainly so both halves
/// share the exact same "reserve `max+1`" logic ([`avoid_indices`]) and
/// so `from_theta` reads symmetrically for the two independent supplies
/// it seeds.
fn avoid_vars<'a>(lits: impl IntoIterator<Item = &'a LNLit>) -> FastFreshState {
    avoid_indices(lits.into_iter().filter_map(|l| match l {
        Lit::Var(v) => Some(v.idx),
        Lit::Con(_) => None,
    }))
}

/// Shared "reserve `[0, max+1)`" logic behind [`avoid_names`]/[`avoid_vars`]
/// -- mirrors `crate::lterm::avoid`'s own reservation logic, just over a
/// plain `u64` index iterator instead of a `HasFrees` term.
fn avoid_indices(indices: impl IntoIterator<Item = u64>) -> FastFreshState {
    let max = indices.into_iter().max();
    let mut s = FastFreshState::nothing_used();
    if let Some(max) = max {
        s.fresh_idents(max + 1);
    }
    s
}

/// The inverse of [`canonical_name`]'s encoding: the `idx` baked into a
/// canonical name's string (`"fn7"` -> `Some(7)`), or `None` if `name`
/// isn't of that `<letters><digits>` shape (i.e. not a name this module
/// itself produced).
fn canonical_name_idx(name: &Name) -> Option<u64> {
    let s = name.id.as_str();
    let digits_start = s.find(|c: char| c.is_ascii_digit())?;
    s[digits_start..].parse().ok()
}

/// The canonical variable of `sort` at index `idx`, per `work.tex`'s naming
/// scheme (`mv_i`/`fv_i`/`pv_i`/`tv_i`/`nv_i` for msg/fresh/pub/node/nat).
fn canonical_var(sort: LSort, idx: u64) -> LVar {
    let name = match sort {
        LSort::Msg => "mv",
        LSort::Fresh => "fv",
        LSort::Pub => "pv",
        LSort::Node => "tv",
        LSort::Nat => "nv",
    };
    LVar::new(name, sort, idx)
}

/// The canonical name of `sort` at index `idx`, per `work.tex`'s naming
/// scheme (`fn_i`/`pn_i`/`tn_i`/`nn_i` for fresh/pub/node/nat names). Since
/// `Name` has no separate index field like `LVar` does, the index is baked
/// into the name string. Msg-sorted names only arise from the `Abbrev` tag,
/// which never occurs in a parsed or solved term; the `mn_i` case is
/// included only for totality.
fn canonical_name(sort: LSort, idx: u64) -> Name {
    let (tag, prefix) = match sort {
        LSort::Fresh => (NameTag::Fresh, "fn"),
        LSort::Pub => (NameTag::Pub, "pn"),
        LSort::Node => (NameTag::Node, "tn"),
        LSort::Nat => (NameTag::Nat, "nn"),
        LSort::Msg => (NameTag::Abbrev, "mn"),
    };
    Name::new(tag, format!("{prefix}{idx}"))
}

/// Applies a literal-to-literal renaming to `t`, rebuilding through the
/// term's smart constructors ([`f_app`]) so the result stays in `CAN_AC`
/// normal form.
///
/// **Panics if `t` mentions a literal outside `ren`'s domain.** Every
/// caller runs against an already-exhaustive renaming: [`Canonizer::canonize`]
/// (above) only reaches this once `next_literals` has run to exhaustion, and
/// `tamarin_theory::canon`'s Stage G rewrites run strictly AFTER the graph
/// part has discovered and canonized every literal the system can
/// legitimately contain (`work.tex`'s guardedness argument), everything
/// past that point being read-only against an already-exhaustive `theta`.
/// So an uncovered literal here means an earlier stage failed to discover
/// real system content -- and passing it through silently would leak that
/// literal's ORIGINAL, un-canonicalized identity into the "canonical"
/// result, which is precisely how two NON-alpha-equivalent systems end up
/// canonicalizing identically. Panicking turns that silent soundness gap
/// into a loud, immediately-diagnosable crash.
///
/// `pub`, not `pub(crate)`: `tamarin_theory::canon`'s whole-system assembly
/// (Stage G) reuses this directly for `eq_store.subst`'s range rewrite
/// rather than reimplementing it, and `tamarin_theory` is a separate crate
/// from this one, so `pub(crate)` would not have been visible there.
pub fn apply_literal_renaming(t: &LNTerm, ren: &BTreeMap<LNLit, LNLit>) -> LNTerm {
    match t {
        Term::Lit(l) => Term::Lit(ren.get(l).copied().unwrap_or_else(|| {
            panic!(
                "apply_literal_renaming: literal {l:?} is not covered by the renaming -- \
                 an uncovered literal means an earlier canonicalization stage missed real \
                 system content (see this function's own doc comment)"
            )
        })),
        Term::App(sym, args) => {
            let mapped: Vec<LNTerm> = args
                .iter()
                .map(|a| apply_literal_renaming(a, ren))
                .collect();
            f_app(*sym, mapped)
        }
    }
}

/// Canonizes `t` w.r.t. $\alphaeqac$ using and updating `labelling` in
/// place: a literal already in `labelling.theta()` keeps its existing
/// image; a new literal gets this call's own local permutation tie-break
/// (the `Canonizer` machinery, unchanged); the winning candidate's
/// newly-fixed entries and advanced fresh-index cursors are folded back
/// into `labelling`. This is what lets several terms — e.g. every graph
/// vertex's own fact, in canonical vertex order — share ONE consistent
/// labelling: a variable/name common to two different vertices' facts
/// gets assigned the SAME canonical literal in both, rather than each
/// term being canonized in isolation (see `tamarin_theory::canon`'s
/// `canonicalize_fact_seeded`/`canonicalize_rule_seeded`/
/// `canonicalize_action_fact_seeded`, which lift this to facts/rules/
/// action-formula facts the same way `canonicalize_fact`/
/// `canonicalize_rule` already lift the non-accumulating
/// [`canonicalize_alpha_eq_ac`]).
pub fn canonicalize_alpha_eq_ac_seeded(t: &LNTerm, labelling: &mut CanonLabelling) -> LNTerm {
    let mut c = Canonizer::new_with_labelling(t, std::mem::take(labelling));
    let (canon_term, updated) = c.canonize();
    *labelling = updated;
    canon_term
}

/// Canonize `t` with respect to $\alphaeqac$: two terms are $\alphaeqac$ iff
/// their canonical forms are syntactically equal (`thm:can_alphaeqac_can2`).
///
/// Behavior-preserving entry point for a caller that doesn't need to
/// accumulate a labelling across several terms — equivalent to
/// [`canonicalize_alpha_eq_ac_seeded`] with a fresh, empty
/// [`CanonLabelling`] that's then discarded.
pub fn canonicalize_alpha_eq_ac(t: &LNTerm) -> LNTerm {
    canonicalize_alpha_eq_ac_seeded(t, &mut CanonLabelling::empty())
}

// `Fingerprint`/`fingerprint_term` now live in `crate::fingerprint` (SHA-256,
// a Merkle hash over the term's structure — no seed, so it's reproducible
// across processes/runs, unlike `std`'s default hasher family). Re-exported
// here under their original path so existing callers/tests are unaffected.
pub use crate::fingerprint::{fingerprint_term, Fingerprint};

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crate::alpha_eq_ac::position::PosStep;
    use crate::alpha_eq_ac::*;
    use crate::builtin::{emap, xor};
    use crate::function_symbols::{Constructability, NoEqSym, Privacy};
    use crate::lterm::{fresh_term, LNTerm, LSort, LVar};
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

    /// Canonizes `t1` and `t2` independently and asserts that each
    /// considered exactly its expected number of candidate permutations
    /// (`expected_perms1` for `t1`, `expected_perms2` for `t2` —
    /// `considered_permutations()` is a pure function of a term's worklist
    /// shape (position/sort structure), not of the literals' names, so two
    /// terms only ever agree on this count when their worklist shapes
    /// agree, which is not implied by them being alpha-eq-AC or not: e.g.
    /// two non-alpha-eq-AC terms can still land in the same-size AC group
    /// and so need the same count, per `noeq_position_sort_mismatch_not_alpha_eq`).
    /// Returns the two canonical forms so callers still make their own
    /// `assert_eq!`/`assert_ne!` call on them, matching whichever
    /// equivalence the test is pinning down.
    fn canonize_and_assert_perms(
        t1: &LNTerm,
        t2: &LNTerm,
        expected_perms1: usize,
        expected_perms2: usize,
    ) -> (LNTerm, LNTerm) {
        let mut c1 = Canonizer::new(t1);
        let mut c2 = Canonizer::new(t2);
        let (ct1, _) = c1.canonize();
        let (ct2, _) = c2.canonize();
        assert_eq!(
            c1.considered_permutations(),
            expected_perms1,
            "t1 considered {} permutations, expected {}",
            c1.considered_permutations(),
            expected_perms1
        );
        assert_eq!(
            c2.considered_permutations(),
            expected_perms2,
            "t2 considered {} permutations, expected {}",
            c2.considered_permutations(),
            expected_perms2
        );
        (ct1, ct2)
    }

    // -- 0) `CanonLabelling::from_theta` seeding: the fresh-index counters
    //    must start past every literal already in the seed substitution's
    //    image, whether that literal is a canonical VARIABLE or a
    //    canonical NAME -- see `avoid_names`/`avoid_vars`'s own doc
    //    comments for the bug this covers (`crate::lterm::avoid` alone
    //    only sees `Lit::Var`) -- and the two kinds' supplies must stay
    //    fully INDEPENDENT even for the same sort (see `PerSortFresh`'s
    //    own doc comment). The term being canonized is irrelevant to this
    //    seeding behavior, so every case below uses the same unrelated
    //    dummy term. ---------------------------------------------------

    /// Names only: a substitution whose image contains just a canonical
    /// NAME literal (`fn7`) must push the next fresh Fresh-sorted NAME
    /// index to 8. Before the seeding fix, `Lit::Con` was invisible to
    /// `crate::lterm::avoid`, so this would have wrongly stayed at 0.
    #[test]
    fn canonizer_seeding_advances_past_a_canonical_name_in_the_substitution_image() {
        let t = v("dummy", LSort::Msg);
        let mut subst = BTreeMap::new();
        subst.insert(
            Lit::Con(Name::new(NameTag::Fresh, "origFresh")),
            Lit::Con(canonical_name(LSort::Fresh, 7)),
        );
        let mut c = Canonizer::new_with_labelling(&t, CanonLabelling::from_theta(subst));
        assert_eq!(
            c.allocate_fresh_name_indices(LSort::Fresh, 1),
            8,
            "seeding from a substitution containing the canonical name fn7 must \
             make the next fresh Fresh-sorted NAME index 8, not collide with fn7"
        );
    }

    /// Vars only: the case `crate::lterm::avoid` already handled
    /// correctly even before the seeding fix -- pinned here as a
    /// regression guard now that seeding goes through `avoid_vars`
    /// instead.
    #[test]
    fn canonizer_seeding_advances_past_a_canonical_var_in_the_substitution_image() {
        let t = v("dummy", LSort::Msg);
        let mut subst = BTreeMap::new();
        subst.insert(
            Lit::Var(LVar::new("origPub", LSort::Pub, 0)),
            Lit::Var(canonical_var(LSort::Pub, 3)),
        );
        let mut c = Canonizer::new_with_labelling(&t, CanonLabelling::from_theta(subst));
        assert_eq!(
            c.allocate_fresh_var_indices(LSort::Pub, 1),
            4,
            "seeding from a substitution containing the canonical var pv3 must \
             make the next fresh Pub-sorted VARIABLE index 4"
        );
    }

    /// Both, of the SAME sort: the NAME supply and the VARIABLE supply
    /// must seed and advance completely INDEPENDENTLY of each other --
    /// neither one's index is affected by the other kind's index, no
    /// matter which one happens to be larger. Checked both ways round so
    /// a fix that accidentally shares state between the two kinds (e.g.
    /// still taking a combined max) can't pass by accident.
    #[test]
    fn canonizer_seeding_keeps_the_name_and_var_supplies_independent_for_the_same_sort() {
        let t = v("dummy", LSort::Msg);

        let mut name_larger = BTreeMap::new();
        name_larger.insert(
            Lit::Var(LVar::new("origFresh1", LSort::Fresh, 0)),
            Lit::Var(canonical_var(LSort::Fresh, 2)),
        );
        name_larger.insert(
            Lit::Con(Name::new(NameTag::Fresh, "origFresh2")),
            Lit::Con(canonical_name(LSort::Fresh, 9)),
        );
        let mut c1 = Canonizer::new_with_labelling(&t, CanonLabelling::from_theta(name_larger));
        assert_eq!(
            c1.allocate_fresh_name_indices(LSort::Fresh, 1),
            10,
            "the name fn9 must push the NAME supply to 10, regardless of fv2"
        );
        assert_eq!(
            c1.allocate_fresh_var_indices(LSort::Fresh, 1),
            3,
            "the VARIABLE supply must start at 3 (past fv2 only), unaffected by fn9"
        );

        let mut var_larger = BTreeMap::new();
        var_larger.insert(
            Lit::Var(LVar::new("origFresh1", LSort::Fresh, 0)),
            Lit::Var(canonical_var(LSort::Fresh, 9)),
        );
        var_larger.insert(
            Lit::Con(Name::new(NameTag::Fresh, "origFresh2")),
            Lit::Con(canonical_name(LSort::Fresh, 2)),
        );
        let mut c2 = Canonizer::new_with_labelling(&t, CanonLabelling::from_theta(var_larger));
        assert_eq!(
            c2.allocate_fresh_var_indices(LSort::Fresh, 1),
            10,
            "the var fv9 must push the VARIABLE supply to 10, regardless of fn2"
        );
        assert_eq!(
            c2.allocate_fresh_name_indices(LSort::Fresh, 1),
            3,
            "the NAME supply must start at 3 (past fn2 only), unaffected by fv9"
        );
    }

    /// The headline property `CanonLabelling`/`canonicalize_alpha_eq_ac_seeded`
    /// exist for: canonizing TWO SEPARATE terms in sequence, sharing ONE
    /// labelling, assigns a literal common to BOTH terms the SAME
    /// canonical literal in both outputs -- unlike two independent
    /// `canonicalize_alpha_eq_ac` calls, which would each pick their own
    /// local tie-break with no relation to each other.
    #[test]
    fn canonicalize_alpha_eq_ac_seeded_shares_a_common_variable_across_two_terms() {
        let shared = v("shared", LSort::Fresh);
        let t1 = f_app_no_eq(no_eq_sym("f", 1), vec![shared.clone()]);
        let t2 = f_app_no_eq(no_eq_sym("g", 1), vec![shared]);

        let mut labelling = CanonLabelling::empty();
        let c1 = canonicalize_alpha_eq_ac_seeded(&t1, &mut labelling);
        let c2 = canonicalize_alpha_eq_ac_seeded(&t2, &mut labelling);

        // The shared variable must have been assigned exactly one entry
        // in the accumulated theta (not re-canonized independently the
        // second time).
        assert_eq!(
            labelling.theta().len(),
            1,
            "the shared variable must be fixed exactly once across both calls"
        );

        // Extracting the argument each canonical term actually carries
        // confirms it's the SAME canonical literal in both, not merely
        // that theta happens to have one entry.
        let arg_of = |t: &LNTerm| match t {
            Term::App(_, args) => args[0].clone(),
            _ => panic!("expected a unary application"),
        };
        assert_eq!(
            arg_of(&c1),
            arg_of(&c2),
            "the variable shared between t1 and t2 must canonize to the identical literal in both"
        );
    }

    /// The concrete, directly observable consequence of the split:
    /// canonizing ONE name and ONE var of the SAME sort in the SAME batch
    /// (`Canonizer::canonize_literals`) must assign each its own index-0
    /// canonical literal. Under the old shared counter, the var would
    /// have been pushed to index 1 purely because the name happened to
    /// be allocated first within `canonize_literals`.
    #[test]
    fn canonize_literals_gives_a_same_sort_name_and_var_independent_index_0() {
        let t = v("dummy", LSort::Msg); // arbitrary -- canonize_literals never reads self.term
        let mut c = Canonizer::new(&t);
        let orig_name = Name::new(NameTag::Fresh, "origFresh");
        let orig_var = LVar::new("origFreshVar", LSort::Fresh, 0);
        c.canonize_literals(&[Lit::Con(orig_name), Lit::Var(orig_var)]);

        assert_eq!(
            c.subst.len(),
            1,
            "one name and one var: no permutation ambiguity within either kind"
        );
        let subst = &c.subst[0];
        assert_eq!(
            subst.get(&Lit::Con(orig_name)),
            Some(&Lit::Con(canonical_name(LSort::Fresh, 0))),
            "the name must canonize to fn0"
        );
        assert_eq!(
            subst.get(&Lit::Var(orig_var)),
            Some(&Lit::Var(canonical_var(LSort::Fresh, 0))),
            "the var must ALSO canonize to index 0 -- its own supply, untouched by the name"
        );
    }

    // -- 1) NoEq-only terms: renaming variables of the same sort is alpha
    //    equivalent, no AC involved. --------------------------------------
    #[test]
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
        // Perms: 1 both
        let (ct1, ct2) = canonize_and_assert_perms(&t1, &t2, 1, 1);
        assert_eq!(ct1, ct2);
        assert_eq!(fingerprint_term(&ct1), fingerprint_term(&ct2));
    }

    // -- 2) Same as 1) but the outer symbol is AC (`xor`, work.tex's
    //    `\oplus`). Cf. \Cref{ex:ex1canon}: `xor(a,b)` and `xor(x,y)` (all
    //    `msg`) are $\alphaeqac$ via `{a -> y, b -> x}`. ---------------------
    #[test]
    fn ac_vars_renamed_are_alpha_eq() {
        let t1 = xor(v("a", LSort::Msg), v("b", LSort::Msg));
        let t2 = xor(v("x", LSort::Msg), v("y", LSort::Msg));
        // Perms: 2 both
        let (ct1, ct2) = canonize_and_assert_perms(&t1, &t2, 2, 2);
        assert_eq!(ct1, ct2);
        assert_eq!(fingerprint_term(&ct1), fingerprint_term(&ct2));
    }

    // -- 3) Same as 1) but the outer symbol is C (`emap`), i.e. commutative
    //    but not associative. ----------------------------------------------
    #[test]
    fn c_vars_renamed_are_alpha_eq() {
        let t1 = emap(v("x", LSort::Msg), v("y", LSort::Msg));
        let t2 = emap(v("a", LSort::Msg), v("b", LSort::Msg));
        // Perms: 2 both
        let (ct1, ct2) = canonize_and_assert_perms(&t1, &t2, 2, 2);
        assert_eq!(ct1, ct2);
        assert_eq!(fingerprint_term(&ct1), fingerprint_term(&ct2));
    }

    // -- 4) NoEq: same shape as 1) but the sort at each position differs
    //    between the two terms, so no sort-respecting renaming can make them
    //    equal — positions matter for a non-AC symbol. ----------------------
    #[test]
    fn noeq_position_sort_mismatch_not_alpha_eq() {
        let f = no_eq_sym("f", 2);
        let t1 = f_app_no_eq(f, vec![v("x", LSort::Msg), v("y", LSort::Fresh)]);
        let t2 = f_app_no_eq(f, vec![v("a", LSort::Fresh), v("b", LSort::Msg)]);
        // Perms: 1 both
        let (ct1, ct2) = canonize_and_assert_perms(&t1, &t2, 1, 1);
        assert_ne!(ct1, ct2);
    }

    // -- 5) AC: swapping argument ORDER never breaks $\alphaeqac$ (that is
    //    the whole point of AC canonization), so the mismatch has to come
    //    from a different MULTISET of sorts: `{msg, msg}` vs `{msg, fresh}`.
    #[test]
    fn ac_sort_multiset_mismatch_not_alpha_eq() {
        let t1 = xor(v("x", LSort::Msg), v("y", LSort::Msg));
        let t2 = xor(v("a", LSort::Msg), v("b", LSort::Fresh));
        // Perms: 2 for t1, 1 for t2 — t1 is the same shape as
        // `ac_vars_renamed_are_alpha_eq`'s term (2 same-sort vars in one AC
        // group, hence 2!), while t2's differing sorts split into two
        // singleton buckets (no permutation choice each).
        let (ct1, ct2) = canonize_and_assert_perms(&t1, &t2, 2, 1);
        assert_ne!(ct1, ct2);
    }

    // -- 6) Same reasoning as 5) for a C symbol. -----------------------------
    #[test]
    fn c_sort_multiset_mismatch_not_alpha_eq() {
        let t1 = emap(v("x", LSort::Msg), v("y", LSort::Msg));
        let t2 = emap(v("a", LSort::Msg), v("b", LSort::Fresh));
        // Perms: 2 for t1, 1 for t2 — same reasoning as 5).
        let (ct1, ct2) = canonize_and_assert_perms(&t1, &t2, 2, 1);
        assert_ne!(ct1, ct2);
    }

    // -- 7) Idempotence (\Cref{can1} / \Cref{thm:can_alphaeqac_can1}), on the
    //    worked AC example \Cref{ex:ac_canon}: `f(xor(c,d), xor(a,c), a)`.
    #[test]
    fn canon_is_idempotent() {
        let f = no_eq_sym("f", 3);
        let a = v("a", LSort::Msg);
        let c = v("c", LSort::Msg);
        let d = v("d", LSort::Msg);
        let t = f_app_no_eq(f, vec![xor(c.clone(), d), xor(a.clone(), c), a]);
        let once = canonicalize_alpha_eq_ac(&t);
        // Perms: 1 both — "both" here means both applications of `canonize`
        // (on `t` and on its canonical form `once`), since idempotence is
        // about one term's canonization being a fixed point, not a
        // comparison between two different terms.
        let (ct, ct_once) = canonize_and_assert_perms(&t, &once, 1, 1);
        assert_eq!(ct, ct_once);
        assert_eq!(fingerprint_term(&ct), fingerprint_term(&ct_once));
    }

    // -- 8) \Cref{ex:wrong_ac_canon}: `f(xor(a,b), a)` and `f(xor(x,y), y)`
    //    (all `msg`) are $\alphaeqac$ via `{a -> y, b -> x}`, but a naive
    //    canonizer that renames literals under AC symbols independently of
    //    their occurrences elsewhere gets this wrong (see the chapter
    //    discussion right after the example). This regression-tests that
    //    our implementation propagates the renaming of `a` correctly.
    #[test]
    fn wrong_ac_canon_example_propagates_shared_literal() {
        let f = no_eq_sym("f", 2);
        let a = v("a", LSort::Msg);
        let b = v("b", LSort::Msg);
        let x = v("x", LSort::Msg);
        let y = v("y", LSort::Msg);
        let t1 = f_app_no_eq(f, vec![xor(a.clone(), b), a]);
        let t2 = f_app_no_eq(f, vec![xor(x, y.clone()), y]);
        // Perms: 1 both
        let (ct1, ct2) = canonize_and_assert_perms(&t1, &t2, 1, 1);
        assert_eq!(ct1, ct2);
        assert_eq!(fingerprint_term(&ct1), fingerprint_term(&ct2));
    }

    // -- 9) AC associativity/flattening interacting with renaming: two
    //    different nestings of the same three (differently named) literals
    //    under `xor` flatten to the same multiset and must canonize equal.
    #[test]
    fn ac_associativity_flattening_alpha_eq() {
        let t1 = xor(
            xor(v("a", LSort::Msg), v("b", LSort::Msg)),
            v("c", LSort::Msg),
        );
        let t2 = xor(
            v("x", LSort::Msg),
            xor(v("y", LSort::Msg), v("z", LSort::Msg)),
        );
        // Perms: 6 both
        let (ct1, ct2) = canonize_and_assert_perms(&t1, &t2, 6, 6);
        assert_eq!(ct1, ct2);
        assert_eq!(fingerprint_term(&ct1), fingerprint_term(&ct2));
    }

    // -- 10) Literals are names ($\mathcal{N}$) as well as variables
    //    ($\mathcal{V}$); renaming a fresh NAME constant must be handled the
    //    same way as renaming a variable.
    #[test]
    fn fresh_names_renamed_are_alpha_eq() {
        let f = no_eq_sym("f", 1);
        let t1 = f_app_no_eq(f, vec![fresh_term("n")]);
        let t2 = f_app_no_eq(f, vec![fresh_term("m")]);
        // Perms: 1 both
        let (ct1, ct2) = canonize_and_assert_perms(&t1, &t2, 1, 1);
        assert_eq!(ct1, ct2);
        assert_eq!(fingerprint_term(&ct1), fingerprint_term(&ct2));
    }

    // -- 11) A name and a variable of the SAME sort are not interchangeable:
    //    a sort-respecting renaming never maps a name to a variable or vice
    //    versa (the chapter's naming scheme reserves disjoint canonical
    //    families, e.g. `fn_i` for fresh names vs `fv_i` for fresh
    //    variables), so the two terms below must canonize differently.
    #[test]
    fn name_and_variable_of_same_sort_not_interchangeable() {
        let f = no_eq_sym("f", 1);
        let t1 = f_app_no_eq(f, vec![fresh_term("n")]);
        let t2 = f_app_no_eq(f, vec![v("x", LSort::Fresh)]);
        // Perms: 1 both
        let (ct1, ct2) = canonize_and_assert_perms(&t1, &t2, 1, 1);
        assert_ne!(ct1, ct2);
    }

    // -- 12) AC: the same MULTISET of sorts at swapped argument positions —
    //    i.e. commutativity itself, not just a per-position rename — must
    //    still canonize equal: `xor(a:msg, b:pub) ~ xor(f:pub, g:msg)`.
    #[test]
    fn ac_swap_position_with_matching_sorts_alpha_eq() {
        let t1 = xor(v("a", LSort::Msg), v("b", LSort::Pub));
        let t2 = xor(v("f", LSort::Pub), v("g", LSort::Msg));
        // Perms: 1 both
        let (ct1, ct2) = canonize_and_assert_perms(&t1, &t2, 1, 1);
        assert_eq!(ct1, ct2);
        assert_eq!(fingerprint_term(&ct1), fingerprint_term(&ct2));
    }

    // -- 13) Same commutativity property for a C symbol (`emap`). -----------
    #[test]
    fn c_swap_position_with_matching_sorts_alpha_eq() {
        let t1 = emap(v("a", LSort::Msg), v("b", LSort::Pub));
        let t2 = emap(v("f", LSort::Pub), v("g", LSort::Msg));
        // Perms: 1 both
        let (ct1, ct2) = canonize_and_assert_perms(&t1, &t2, 1, 1);
        assert_eq!(ct1, ct2);
        assert_eq!(fingerprint_term(&ct1), fingerprint_term(&ct2));
    }

    // -- 14) An AC term covering all five `LSort`s at once, permuted in both
    //    argument order and literal naming between `t1` and `t2`.
    #[test]
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
        // Perms: 1 both
        let (ct1, ct2) = canonize_and_assert_perms(&t1, &t2, 1, 1);
        assert_eq!(ct1, ct2);
        assert_eq!(fingerprint_term(&ct1), fingerprint_term(&ct2));
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

    // -- 17) Driving `next_literals` to exhaustion on \Cref{table:canon_ex1}'s
    //    `t = f(xor(a, b, g(c,d)), d)` must reproduce that table's execution:
    //    `c` first (singleton at the lexicographically smallest position),
    //    then `d`, then the pair `{a, b}` at the AC group. -------------------
    #[test]
    fn next_literals_follows_work_tex_example() {
        let g = no_eq_sym("g", 2);
        let f = no_eq_sym("f", 2);
        let (a, b) = (v("a", LSort::Msg), v("b", LSort::Msg));
        let (c, d) = (v("c", LSort::Msg), v("d", LSort::Msg));
        let inner_g = f_app_no_eq(g, vec![c.clone(), d.clone()]);
        let group = xor(xor(a.clone(), b.clone()), inner_g);
        let t = f_app_no_eq(f, vec![group, d.clone()]);

        let mut canon = Canonizer::new(&t);
        let mut got = Vec::new();
        while let Some(lits) = canon.next_literals() {
            got.push(lits);
        }

        assert_eq!(
            got,
            vec![
                vec![lit_of(&c)],
                vec![lit_of(&d)],
                vec![lit_of(&a), lit_of(&b)],
            ]
        );
        assert!(canon.buckets.is_empty());
        assert!(canon.remaining.is_empty());
    }

    // -- 18) The relocation path: on \Cref{ex:wrong_ac_canon}'s
    //    `f(xor(a, b), a)`, canonizing `a` at the singleton position `f1`
    //    drops the AC group from 2 uncanonized msg literals to 1, so that
    //    position must MOVE from bucket `(2, Msg)` to `(1, Msg)` rather
    //    than be dropped. ----------------------------------------------------
    #[test]
    fn next_literals_relocates_position_to_smaller_bucket() {
        let f = no_eq_sym("f", 2);
        let (a, b) = (v("a", LSort::Msg), v("b", LSort::Msg));
        let t = f_app_no_eq(f, vec![xor(a.clone(), b.clone()), a.clone()]);

        let mut canon = Canonizer::new(&t);
        assert_eq!(canon.next_literals(), Some(vec![lit_of(&a)]));

        // `a` was shared, so the AC group relocated instead of vanishing.
        let group_pos = vec![PosStep::Arg(0), PosStep::AcGroup(None)];
        assert_eq!(
            canon.buckets,
            BTreeMap::from([(
                BucketKey {
                    count: 1,
                    sort: LSort::Msg
                },
                BTreeSet::from([group_pos])
            )])
        );

        assert_eq!(canon.next_literals(), Some(vec![lit_of(&b)]));
        assert_eq!(canon.next_literals(), None);
        assert!(canon.remaining.is_empty());
    }

    // -- 19) One position, two sorts: it sits in a `Fresh` and a `Msg`
    //    bucket at once, so it survives the first pop (which only clears
    //    its `Fresh` literal) and is popped again for its `Msg` ones. -------
    #[test]
    fn next_literals_drains_one_position_per_sort() {
        let (a, b) = (v("a", LSort::Msg), v("b", LSort::Msg));
        let c = v("c", LSort::Fresh);
        let t = xor(xor(a.clone(), b.clone()), c.clone());

        let mut canon = Canonizer::new(&t);
        let mut got = Vec::new();
        while let Some(lits) = canon.next_literals() {
            got.push(lits);
        }

        // `(1, Fresh)` sorts before `(2, Msg)`: count first, then sort.
        assert_eq!(got, vec![vec![lit_of(&c)], vec![lit_of(&a), lit_of(&b)]]);
        assert!(canon.buckets.is_empty());
        assert!(canon.remaining.is_empty());
    }

    /// A canonical msg variable `mv_idx`, per `work.tex`'s naming scheme
    /// (indices here are 0-based, matching [`Canonizer`]'s fresh-index
    /// counters, rather than the chapter's 1-based `mv_1, mv_2, ...`).
    fn mv(idx: u64) -> LNTerm {
        var_term(LVar::new("mv", LSort::Msg, idx))
    }

    // -- 20) `Canonizer::canonize`, pinned to \Cref{table:canon_ex1}'s exact
    //    execution of $t = f(\oplus(a, b, g(c, d)), d)$: `c` and `d` are
    //    singletons canonized first (in that order, per `next_literals`'
    //    already-tested schedule), leaving `{a, b}` as the only AC group
    //    still needing a permutation choice. Both permutations collapse to
    //    the same `CAN_AC`-sorted result here since `a`/`b` occur nowhere
    //    else in `t`, so the canonical form is pinned exactly (0-based
    //    counterpart of the table's `f(\oplus(mv_3, mv_4, g(mv_1, mv_2)),
    //    mv_2)`).
    #[test]
    fn canonize_follows_work_tex_table_example() {
        let g = no_eq_sym("g", 2);
        let f = no_eq_sym("f", 2);
        let (a, b) = (v("a", LSort::Msg), v("b", LSort::Msg));
        let (c, d) = (v("c", LSort::Msg), v("d", LSort::Msg));
        let inner_g = f_app_no_eq(g, vec![c, d.clone()]);
        let group = xor(xor(a, b), inner_g);
        let t = f_app_no_eq(f, vec![group, d]);

        let mut canon = Canonizer::new(&t);
        let (canon_term, subst) = canon.canonize();

        let want = f_app_no_eq(
            f,
            vec![
                xor(xor(mv(2), mv(3)), f_app_no_eq(g, vec![mv(0), mv(1)])),
                mv(1),
            ],
        );
        assert_eq!(canon_term, want);
        assert_eq!(fingerprint_term(&canon_term), fingerprint_term(&want));
        assert_eq!(subst.theta().len(), 4);
        // Perms: 2
        assert_eq!(canon.considered_permutations(), 2);
    }

    // -- 21) `Canonizer::canonize`, pinned to \Cref{ex:ac_canon}: $t =
    //    f(\oplus(c, d), \oplus(a, c), a)$. `a` is canonized first (it is a
    //    singleton at position `f2`), which shrinks `\oplus(a, c)` to the
    //    singleton `c`, which in turn shrinks `\oplus(c, d)` to the
    //    singleton `d` — no permutation is ever considered, matching the
    //    chapter's point that tracking remaining-literal counts per position
    //    avoids the combinatorial fallback from \Cref{ex:naive_ac_canon}.
    //    0-based counterpart of the chapter's `f(\oplus(mv_2, mv_3),
    //    \oplus(mv_1, mv_2), mv_1)`.
    #[test]
    fn canonize_follows_work_tex_ac_canon_example() {
        let f = no_eq_sym("f", 3);
        let a = v("a", LSort::Msg);
        let c = v("c", LSort::Msg);
        let d = v("d", LSort::Msg);
        let t = f_app_no_eq(f, vec![xor(c.clone(), d), xor(a.clone(), c), a]);

        let mut canon = Canonizer::new(&t);
        let (canon_term, subst) = canon.canonize();

        let want = f_app_no_eq(f, vec![xor(mv(1), mv(2)), xor(mv(0), mv(1)), mv(0)]);
        assert_eq!(canon_term, want);
        assert_eq!(fingerprint_term(&canon_term), fingerprint_term(&want));
        assert_eq!(subst.theta().len(), 3);
        // Perms: 1
        assert_eq!(canon.considered_permutations(), 1);
    }
}
