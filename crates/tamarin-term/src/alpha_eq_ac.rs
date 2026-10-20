//! Canonical variable names for canonicalization modulo alpha equivalence
//! and AC ($\alphaeqac$).
//!
//! A constraint system is canonicalized in `tamarin_theory::canon_term_graph`:
//! it is written as one term, bliss's canonical labeling of that term's graph
//! orders its variables, and [`CanonLabelling::bind_next`] names them in that
//! order. The canonical form is the term renamed by the resulting renaming θ
//! ([`rename_var`]) and normalized.
//!
//! **Names are never renamed.** A name (`'a'`, `~'n'`, `%'n'`) is a
//! constant the theory refers to by identity -- rules, restrictions and
//! lemmas mention specific names, and the solver never creates a name that
//! persists in a system -- so two terms that differ only in a name are not
//! interchangeable (`<'1', x>` and `<'2', x>` unify with different rule
//! conclusions). Like a function symbol, a name is its own canonical form;
//! the canonical labelling is a renaming of variables only. (This deviates
//! from `work.tex`, which renames names too.)
use crate::lterm::{LSort, LVar};

#[allow(clippy::disallowed_types)]
use std::collections::BTreeMap;
use tamarin_utils::fresh::FastFreshState;

/// One [`FastFreshState`] counter per [`LSort`]: the next index of a
/// canonical variable (`mv_i`/`fv_i`/...) of each sort.
#[derive(Debug, Clone, Copy, Default)]
struct PerSortFresh {
    msg: FastFreshState,
    pub_: FastFreshState,
    fresh: FastFreshState,
    nat: FastFreshState,
    node: FastFreshState,
}

impl PerSortFresh {
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

/// A canonical labelling θ: a renaming of VARIABLES only (names are never
/// renamed, see the module doc), built by naming variables one by one in an
/// order the caller chose canonically ([`Self::bind_next`]).
#[derive(Debug, Clone, Default)]
pub struct CanonLabelling {
    theta: BTreeMap<LVar, LVar>,
    fresh_vars: PerSortFresh,
}

impl CanonLabelling {
    /// No variable named yet; every sort's next index is 0.
    pub fn empty() -> Self {
        Self::default()
    }

    /// The renaming built so far.
    pub fn theta(&self) -> &BTreeMap<LVar, LVar> {
        &self.theta
    }

    /// Binds `v` to the next unused canonical variable of its sort and
    /// returns it. The caller names the variables in an order it chose
    /// canonically (in `tamarin_theory::canon_term_graph`, the order of their
    /// vertices in bliss's canonical labeling). Panics if `v` is already
    /// bound: rebinding would break θ's injectivity.
    pub fn bind_next(&mut self, v: LVar) -> LVar {
        assert!(
            !self.theta.contains_key(&v),
            "CanonLabelling::bind_next: {v:?} is already bound to {:?}",
            self.theta[&v]
        );
        let idx = self.fresh_vars.get_mut(v.sort).fresh_idents(1);
        let canonical = canonical_var(v.sort, idx);
        self.theta.insert(v, canonical);
        canonical
    }
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

/// The canonical image of `v` under `theta`.
///
/// **Panics if `v` is outside `theta`'s domain.** `tamarin_theory`
/// renames a system only through a θ that its term graph built over every
/// variable of the system term. So an uncovered variable means an earlier
/// stage failed to discover real system content -- and passing it through
/// silently would leak its ORIGINAL, un-canonicalized identity into the
/// "canonical" result, which is precisely how two NON-alpha-equivalent
/// systems end up canonicalizing identically. Panicking turns that silent
/// soundness gap into a loud, immediately-diagnosable crash.
pub fn rename_var(theta: &BTreeMap<LVar, LVar>, v: LVar) -> LVar {
    *theta.get(&v).unwrap_or_else(|| {
        panic!(
            "rename_var: variable {v:?} is not covered by the canonical renaming -- \
             an uncovered variable means an earlier canonicalization stage missed \
             real system content (see this function's own doc comment)"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each sort counts on its own, in the order the variables are bound.
    #[test]
    fn bind_next_names_each_sort_in_order() {
        let mut labelling = CanonLabelling::empty();
        let x = LVar::new("x", LSort::Msg, 7);
        let k = LVar::new("k", LSort::Fresh, 3);
        let y = LVar::new("y", LSort::Msg, 1);
        assert_eq!(labelling.bind_next(x), LVar::new("mv", LSort::Msg, 0));
        assert_eq!(labelling.bind_next(k), LVar::new("fv", LSort::Fresh, 0));
        assert_eq!(labelling.bind_next(y), LVar::new("mv", LSort::Msg, 1));
        assert_eq!(rename_var(labelling.theta(), x), LVar::new("mv", LSort::Msg, 0));
    }

    #[test]
    #[should_panic(expected = "already bound")]
    fn binding_a_variable_twice_panics() {
        let mut labelling = CanonLabelling::empty();
        let x = LVar::new("x", LSort::Msg, 0);
        labelling.bind_next(x);
        labelling.bind_next(x);
    }

    #[test]
    #[should_panic(expected = "not covered by the canonical renaming")]
    fn renaming_an_uncovered_variable_panics() {
        rename_var(&BTreeMap::new(), LVar::new("x", LSort::Msg, 0));
    }
}
