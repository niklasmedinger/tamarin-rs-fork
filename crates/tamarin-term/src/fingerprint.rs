//! Content fingerprinting for canonical terms (see
//! `tamarin_utils::fingerprint` for the underlying primitive and rationale,
//! and `tamarin-theory::canon`'s module docs for the fingerprinting
//! investigation this implements).
//!
//! [`fingerprint_term`] is a Merkle hash: an application's fingerprint is
//! built from the fingerprints of its ALREADY-fingerprinted arguments (the
//! "subhash" composition), not their raw content, and only a literal leaf
//! ever hashes its own fields directly (there is nothing further to
//! recurse into). This is also exactly the composition the eventual
//! constraint-system fingerprint needs: `canonicalize_fact`/
//! `canonicalize_rule` (`tamarin-theory::canon`) both lift to a plain
//! `LNTerm` via `List`, so a rule's fingerprint is ALREADY, by construction,
//! built from its facts' (and their arguments') fingerprints — no separate
//! `fingerprint_fact`/`fingerprint_rule` implementation is needed.

use crate::function_symbols::{
    AcFctSym, AcSym, CSym, Constructability, FunSym, NdcState, NoEqSym, Privacy,
};
use crate::lterm::{LNTerm, LSort, Name, NameTag};
use crate::term::Term;
use crate::vterm::Lit;

pub use tamarin_utils::fingerprint::Fingerprint;
use tamarin_utils::fingerprint::FingerprintHasher;

/// Fingerprints an ALREADY-CANONICAL term — the output of
/// [`crate::alpha_eq_ac::canonicalize_alpha_eq_ac`] (or, via
/// `tamarin-theory::canon`, of `canonicalize_fact`/`canonicalize_rule`,
/// which both also produce a canonical `LNTerm`). Two terms are
/// $\alphaeqac$ iff their canonical forms' fingerprints match.
pub fn fingerprint_term(t: &LNTerm) -> Fingerprint {
    let mut h = FingerprintHasher::new();
    match t {
        Term::Lit(l) => {
            h.tag("Lit");
            hash_lit(&mut h, l);
        }
        Term::App(sym, args) => {
            h.tag("App");
            hash_funsym(&mut h, sym);
            h.u64(args.len() as u64);
            for a in args.iter() {
                // Subhash: the child's OWN fingerprint, not its raw content.
                h.digest(&fingerprint_term(a));
            }
        }
    }
    h.finish()
}

fn hash_lit(h: &mut FingerprintHasher, l: &Lit<Name, crate::lterm::LVar>) {
    match l {
        Lit::Con(name) => {
            h.tag("Con");
            hash_name(h, name);
        }
        Lit::Var(v) => {
            h.tag("Var");
            h.bytes(v.name.as_bytes());
            hash_lsort(h, v.sort);
            h.u64(v.idx);
        }
    }
}

fn hash_name(h: &mut FingerprintHasher, n: &Name) {
    h.u8(name_tag_byte(n.tag));
    h.bytes(n.id.as_str().as_bytes());
}

/// Explicit, stable byte tags — deliberately NOT derived from the enum's
/// own discriminant (which is an implementation detail, not a promise), so
/// the fingerprint format doesn't silently shift if a variant is ever
/// reordered.
fn name_tag_byte(t: NameTag) -> u8 {
    match t {
        NameTag::Fresh => 0,
        NameTag::Pub => 1,
        NameTag::Node => 2,
        NameTag::Nat => 3,
        NameTag::Abbrev => 4,
    }
}

fn hash_lsort(h: &mut FingerprintHasher, s: LSort) {
    let b = match s {
        LSort::Msg => 0,
        LSort::Pub => 1,
        LSort::Fresh => 2,
        LSort::Node => 3,
        LSort::Nat => 4,
    };
    h.u8(b);
}

fn hash_privacy(h: &mut FingerprintHasher, p: Privacy) {
    h.u8(match p {
        Privacy::Private => 0,
        Privacy::Public => 1,
    });
}

fn hash_constructability(h: &mut FingerprintHasher, c: Constructability) {
    h.u8(match c {
        Constructability::Constructor => 0,
        Constructability::Destructor => 1,
    });
}

fn hash_ndc(h: &mut FingerprintHasher, n: NdcState) {
    h.u8(match n {
        NdcState::IsNdc => 0,
        NdcState::NotNdc => 1,
        NdcState::IsNdcDiff => 2,
        NdcState::IsNdcBoth => 3,
    });
}

/// A symbol's full identity — every field `NoEqSym`/`AcFctSym`'s own
/// `Eq`/`Ord` compare, so the fingerprint is consistent with equality by
/// construction: two symbols the type system considers equal always hash
/// the same, and no field that could make two symbols distinct is left out.
fn hash_no_eq_sym(h: &mut FingerprintHasher, s: &NoEqSym) {
    h.bytes(s.name);
    h.u64(s.arity as u64);
    hash_privacy(h, s.privacy);
    hash_constructability(h, s.constructability);
    hash_ndc(h, s.ndc);
}

fn hash_ac_fct_sym(h: &mut FingerprintHasher, s: &AcFctSym) {
    h.bytes(s.name);
    hash_privacy(h, s.privacy);
    hash_constructability(h, s.constructability);
    hash_ndc(h, s.ndc);
}

fn hash_ac_sym(h: &mut FingerprintHasher, sym: &AcSym) {
    match sym {
        AcSym::Union => {
            h.tag("Union");
        }
        AcSym::Mult => {
            h.tag("Mult");
        }
        AcSym::Xor => {
            h.tag("Xor");
        }
        AcSym::NatPlus => {
            h.tag("NatPlus");
        }
        AcSym::AcFct(s) => {
            h.tag("AcFct");
            hash_ac_fct_sym(h, s);
        }
    }
}

fn hash_funsym(h: &mut FingerprintHasher, sym: &FunSym) {
    match sym {
        FunSym::NoEq(s) => {
            h.tag("NoEq");
            hash_no_eq_sym(h, s);
        }
        FunSym::Ac(ac) => {
            h.tag("Ac");
            hash_ac_sym(h, ac);
        }
        FunSym::C(CSym::EMap) => {
            h.tag("C-EMap");
        }
        FunSym::List => {
            h.tag("List");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtin::xor;
    use crate::function_symbols::{AcFctSym, Constructability, NdcState, NoEqSym, Privacy};
    use crate::lterm::{fresh_term, LSort, LVar};
    use crate::term::f_app_no_eq;
    use crate::vterm::var_term;

    fn v(name: &str, sort: LSort) -> LNTerm {
        var_term(LVar::new(name, sort, 0))
    }

    fn no_eq_sym(name: &'static str, arity: usize) -> NoEqSym {
        NoEqSym::new(name, arity, Privacy::Public, Constructability::Constructor)
    }

    #[test]
    fn same_term_gives_same_fingerprint() {
        let t = v("x", LSort::Msg);
        assert_eq!(fingerprint_term(&t), fingerprint_term(&t));
    }

    #[test]
    fn different_variables_give_different_fingerprints() {
        assert_ne!(
            fingerprint_term(&v("x", LSort::Msg)),
            fingerprint_term(&v("y", LSort::Msg))
        );
    }

    /// A variable and a name of the same sort must never collide — mirrors
    /// `alpha_eq_ac.rs`'s `name_and_variable_of_same_sort_not_interchangeable`.
    #[test]
    fn variable_and_name_of_same_sort_give_different_fingerprints() {
        let f = no_eq_sym("f", 1);
        let t1 = f_app_no_eq(f, vec![fresh_term("n")]);
        let t2 = f_app_no_eq(f, vec![v("x", LSort::Fresh)]);
        assert_ne!(fingerprint_term(&t1), fingerprint_term(&t2));
    }

    /// The Merkle/subhash property itself: changing a DEEPLY NESTED
    /// argument changes the outer fingerprint. If the recursion silently
    /// dropped a child's contribution, this would (wrongly) hold equal.
    #[test]
    fn changing_a_nested_argument_changes_the_fingerprint() {
        let g = no_eq_sym("g", 2);
        let f = no_eq_sym("f", 1);
        let t1 = f_app_no_eq(f, vec![f_app_no_eq(g, vec![v("a", LSort::Msg), v("b", LSort::Msg)])]);
        let t2 = f_app_no_eq(f, vec![f_app_no_eq(g, vec![v("a", LSort::Msg), v("c", LSort::Msg)])]);
        assert_ne!(fingerprint_term(&t1), fingerprint_term(&t2));
    }

    /// AC commutativity: `xor(a,b)` and `xor(b,a)` are the SAME term
    /// (`f_app_ac` sorts at construction), so of course their fingerprints
    /// match — but this pins that the fingerprint is computed on the
    /// (already-sorted) `Term` value, not on some order-sensitive
    /// traversal that could disagree with `Term`'s own `PartialEq`.
    #[test]
    fn ac_terms_equal_modulo_argument_order_fingerprint_equal() {
        let t1 = xor(v("a", LSort::Msg), v("b", LSort::Msg));
        let t2 = xor(v("b", LSort::Msg), v("a", LSort::Msg));
        assert_eq!(t1, t2, "sanity: f_app_ac already normalizes these equal");
        assert_eq!(fingerprint_term(&t1), fingerprint_term(&t2));
    }

    /// Different function symbols applied to the same arguments must not
    /// collide (`hash_funsym` actually distinguishes them).
    #[test]
    fn different_function_symbols_give_different_fingerprints() {
        let f = no_eq_sym("f", 1);
        let g = no_eq_sym("g", 1);
        let x = v("x", LSort::Msg);
        assert_ne!(
            fingerprint_term(&f_app_no_eq(f, vec![x.clone()])),
            fingerprint_term(&f_app_no_eq(g, vec![x]))
        );
    }

    /// Two `NoEqSym`s that differ ONLY in a field beyond name/arity
    /// (constructability here) must still fingerprint differently —
    /// `hash_no_eq_sym` has to cover every field the type's own `Eq` does.
    #[test]
    fn no_eq_syms_differing_only_in_constructability_give_different_fingerprints() {
        let ctor = NoEqSym::new("f", 1, Privacy::Public, Constructability::Constructor);
        let dtor = NoEqSym::new("f", 1, Privacy::Public, Constructability::Destructor);
        let x = v("x", LSort::Msg);
        assert_ne!(
            fingerprint_term(&f_app_no_eq(ctor, vec![x.clone()])),
            fingerprint_term(&f_app_no_eq(dtor, vec![x]))
        );
    }

    /// Same point for `AcFctSym`'s ndc field, exercising `hash_ac_fct_sym`
    /// (a separately-written twin of `hash_no_eq_sym`) directly.
    #[test]
    fn ac_fct_syms_differing_only_in_ndc_give_different_fingerprints() {
        use crate::term::f_app_acfct;
        let a = AcFctSym::new(
            "add",
            Privacy::Public,
            Constructability::Constructor,
            NdcState::NotNdc,
        );
        let b = AcFctSym::new(
            "add",
            Privacy::Public,
            Constructability::Constructor,
            NdcState::IsNdc,
        );
        let (x, y) = (v("x", LSort::Msg), v("y", LSort::Msg));
        assert_ne!(
            fingerprint_term(&f_app_acfct(a, vec![x.clone(), y.clone()])),
            fingerprint_term(&f_app_acfct(b, vec![x, y]))
        );
    }
}
