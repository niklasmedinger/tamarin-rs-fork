//! (No HS analog.) The canonical table the graph searches merge through
//! ([`super::search_graph`], [`super::topn_search`], [`super::mcgs_search`]):
//! it places each constraint system in a class of canonically equal ones,
//! keyed by the system's canonical form ([`crate::canon`]).
//!
//! Canonicalizing a system costs a bliss subprocess. Unless
//! `TAM_RS_CANON_INVARIANT=off`, the table first computes the system's cheap
//! invariant ([`invariant`]): equal canonical forms have equal invariants, so
//! a system whose invariant no class has is a class of its own, and needs no
//! canonicalization at all. Only a system whose invariant a class already has
//! is canonicalized, and with it that class's representative if it was not
//! yet. The table therefore places every system exactly where canonicalizing
//! every system places it: an invariant only ever answers "different" when
//! the forms differ.
//!
//! **Pending classes.** A class placed without a canonicalization is pending:
//! the table keeps nothing of its representative but, once computed, its key.
//! The search keeps the representative's system, in its store while the class
//! may get methods and set aside once it drops it there; a `System` shares
//! most of its parts with its relatives, its term none. When a system with the
//! same invariant comes, the representative's term is built again from that
//! system ([`RepTerm`]) and canonicalized; the class then has a key, and the
//! search may drop the system it set aside ([`CanonicalTable::take_keyed`]).
//!
//! **Where the work happens.** Workers prepare their systems
//! ([`CanonicalTable::prepare`]) while the search thread waits for them, so
//! they may read the table: a system whose invariant has a class is
//! canonicalized right there. [`CanonicalTable::precompute`] then
//! canonicalizes, in parallel, what no worker could: the pending
//! representatives their systems collide with, whose terms only the search
//! thread can build (a `System` may move between threads but not be
//! shared), and systems whose invariant only an earlier system of the same
//! batch has. Last, [`CanonicalTable::intern`] places the systems, one by
//! one, in the search's fixed order; the decisions never depend on which
//! thread did what.
//!
//! **Methods.** A class placed with a canonicalization gives its system its
//! labelling, and the methods applied to it are fingerprinted then. A pending
//! class has none, so its methods keep their pre-canonical form
//! ([`crate::canon::PreCanonicalProofMethod`]), and the proof materializer
//! renames them through [`CanonicalTable::late_labelling`].
//!
//! **Modes** (`TAM_RS_CANON_INVARIANT`, read once):
//! - unset or `on`: the invariant first;
//! - `off`: every system is canonicalized, in its worker;
//! - `check`: places systems as `off` does, and computes every invariant too.
//!   Two systems with one key and different invariants panic; the collision
//!   counters are those `on` would see.

use std::sync::OnceLock;
use std::time::Instant;

use rayon::prelude::*;

use tamarin_term::alpha_eq_ac::CanonLabelling;
use tamarin_utils::fingerprint::Fingerprint;
use tamarin_utils::{FastMap, FastSet};

use crate::canon_fingerprint::{fingerprint_constraint_system, CanonicalSystemFingerprint};
use crate::canon_term_graph::{
    canonicalize_term, invariant, system_to_term, STerm, TermGraphOptions,
};
use crate::constraint::solver::search_graph::{canonical_key, ClassId, WorkStats};
use crate::constraint::system::System;

/// How the table decides (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CanonMode {
    Off,
    On,
    Check,
}

impl CanonMode {
    /// The mode as `TAM_RS_CANON_INVARIANT` spells it.
    pub(crate) fn name(self) -> &'static str {
        match self {
            CanonMode::Off => "off",
            CanonMode::On => "on",
            CanonMode::Check => "check",
        }
    }
}

/// `TAM_RS_CANON_INVARIANT`, read once per process.
pub(crate) fn canon_mode_from_env() -> CanonMode {
    static MODE: OnceLock<CanonMode> = OnceLock::new();
    *MODE.get_or_init(
        || match std::env::var("TAM_RS_CANON_INVARIANT").as_deref() {
            Err(std::env::VarError::NotPresent) | Ok("on") => CanonMode::On,
            Ok("off") => CanonMode::Off,
            Ok("check") => CanonMode::Check,
            other => panic!("TAM_RS_CANON_INVARIANT={other:?}: expected `off`, `on` or `check`"),
        },
    )
}

/// The term of a pending class's representative, built from the system the
/// search kept for it; `None` if the search has no system for the class,
/// which panics where the term is needed.
pub(crate) trait RepTerm: Fn(ClassId) -> Option<STerm> {}
impl<F: Fn(ClassId) -> Option<STerm>> RepTerm for F {}

/// A system's canonical key, and the labelling that maps it onto its form.
#[derive(Debug, Clone)]
pub(crate) struct Canon {
    pub(crate) key: CanonicalSystemFingerprint,
    pub(crate) labelling: CanonLabelling,
}

/// What a worker prepared of one system for [`CanonicalTable::intern`].
pub(crate) struct Prepared {
    /// The system's term, until its key is computed or it is placed (`on`).
    term: Option<STerm>,
    /// `on` and `check`.
    invariant: Option<Fingerprint>,
    /// Its key, once computed.
    canon: Option<Canon>,
}

impl Prepared {
    /// A system with a known key, for tests of the graph searches.
    #[cfg(test)]
    pub(crate) fn with_key(key: CanonicalSystemFingerprint) -> Self {
        Prepared {
            term: None,
            invariant: None,
            canon: Some(Canon {
                key,
                labelling: CanonLabelling::empty(),
            }),
        }
    }
}

/// The classes with one invariant (`on`, `check`).
#[derive(Debug, Clone, Copy)]
enum Bucket {
    /// One class, not canonicalized yet.
    Pending(ClassId),
    /// Canonicalized classes only, each in [`CanonicalTable::by_key`].
    Keyed,
}

/// Where [`CanonicalTable::intern`] placed a system.
pub(crate) struct Placed {
    pub(crate) class: ClassId,
    /// Into a class that existed.
    pub(crate) merged: bool,
    /// The labelling of a new class placed with a canonicalization; `None`
    /// for a pending class and after a merge.
    pub(crate) labelling: Option<CanonLabelling>,
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct TableStats {
    /// Systems placed whose invariant a class already had: under `on`, each
    /// one is canonicalized.
    pub(crate) inv_hits: u64,
    /// Of `inv_hits`, those that became a class of their own: the same
    /// invariant, another form.
    pub(crate) inv_false: u64,
}

/// See the module docs. `Sync`: workers read it while the search waits.
pub(crate) struct CanonicalTable {
    mode: CanonMode,
    by_key: FastMap<CanonicalSystemFingerprint, ClassId>,
    by_invariant: FastMap<Fingerprint, Bucket>,
    /// The pending classes, with their representative's key once computed:
    /// by [`Self::precompute`], or by the proof materializer's workers.
    pending: FastMap<ClassId, OnceLock<Canon>>,
    /// The labellings of the classes canonicalized after they were placed:
    /// their methods kept pre-canonical forms.
    late: FastMap<ClassId, CanonLabelling>,
    /// Pending classes keyed since [`Self::take_keyed`] last drained it.
    newly_keyed: Vec<ClassId>,
    /// Per class, its invariant (`on`, `check`).
    invariants: FastMap<ClassId, Fingerprint>,
    stats: TableStats,
}

/// `term`'s canonical key ([`canonicalize_term`]); panics where bliss fails.
pub(crate) fn canon_of_term(term: &STerm, work: &mut WorkStats) -> Canon {
    let start = Instant::now();
    let (form, labelling, _) = canonicalize_term(term, TermGraphOptions::default())
        .unwrap_or_else(|e| panic!("TAM_RS_MERGE: canonicalizing a constraint system failed: {e}"));
    let key = fingerprint_constraint_system(&form);
    work.record_canon(start.elapsed());
    Canon { key, labelling }
}

/// The key of `r`'s representative, its term built by `rep_term`, which
/// counts with the canonicalization's time.
fn canon_of_rep(r: ClassId, rep_term: &impl RepTerm, work: &mut WorkStats) -> Canon {
    let start = Instant::now();
    let term = rep_term(r).unwrap_or_else(|| {
        panic!("class #{r} is pending, but the search kept no system for its representative")
    });
    work.canon_time += start.elapsed();
    canon_of_term(&term, work)
}

/// `sys`'s term and its invariant, timed together as the invariant.
fn term_and_invariant(sys: &System, work: &mut WorkStats) -> (STerm, Fingerprint) {
    let start = Instant::now();
    let term = system_to_term(sys);
    let fp = invariant(&term);
    work.record_invariant(start.elapsed());
    (term, fp)
}

impl CanonicalTable {
    pub(crate) fn new(mode: CanonMode) -> Self {
        CanonicalTable {
            mode,
            by_key: FastMap::default(),
            by_invariant: FastMap::default(),
            pending: FastMap::default(),
            late: FastMap::default(),
            newly_keyed: Vec::new(),
            invariants: FastMap::default(),
            stats: TableStats::default(),
        }
    }

    /// Prepares `sys` for [`Self::intern`]. Called by workers, while nothing
    /// changes the table: under `on`, a system whose invariant a class has is
    /// canonicalized here.
    pub(crate) fn prepare(&self, sys: &System, work: &mut WorkStats) -> Prepared {
        match self.mode {
            CanonMode::Off => {
                let (key, labelling) = canonical_key(sys, work);
                Prepared {
                    term: None,
                    invariant: None,
                    canon: Some(Canon { key, labelling }),
                }
            }
            CanonMode::Check => {
                let (term, invariant) = term_and_invariant(sys, work);
                Prepared {
                    canon: Some(canon_of_term(&term, work)),
                    term: None,
                    invariant: Some(invariant),
                }
            }
            CanonMode::On => {
                let (term, invariant) = term_and_invariant(sys, work);
                if !self.by_invariant.contains_key(&invariant) {
                    return Prepared {
                        term: Some(term),
                        invariant: Some(invariant),
                        canon: None,
                    };
                }
                Prepared {
                    canon: Some(canon_of_term(&term, work)),
                    term: None,
                    invariant: Some(invariant),
                }
            }
        }
    }

    /// Under `on`, canonicalizes, in parallel, what [`Self::intern`] will need
    /// for `items` and the workers could not compute: the pending
    /// representatives they collide with, whose terms `rep_term` builds on
    /// this thread, and each item whose invariant no class has but an earlier
    /// item has too. `items` are in the order `intern` will place them. Every
    /// caller of `intern` calls it first, for the same items.
    pub(crate) fn precompute<'a>(
        &mut self,
        items: impl IntoIterator<Item = &'a mut Prepared>,
        rep_term: impl RepTerm,
        work: &mut WorkStats,
    ) {
        if self.mode != CanonMode::On {
            return;
        }
        let mut items: Vec<&mut Prepared> = items.into_iter().collect();
        let mut first: FastMap<Fingerprint, usize> = FastMap::default();
        let mut needed = vec![false; items.len()];
        let mut reps: Vec<ClassId> = Vec::new();
        let mut seen_reps: FastSet<ClassId> = FastSet::default();
        for (i, item) in items.iter().enumerate() {
            let invariant = item
                .invariant
                .expect("on: every prepared system has an invariant");
            match self.by_invariant.get(&invariant) {
                Some(Bucket::Pending(r)) => {
                    if self.pending[r].get().is_none() && seen_reps.insert(*r) {
                        reps.push(*r);
                    }
                    needed[i] = item.canon.is_none();
                }
                Some(Bucket::Keyed) => needed[i] = item.canon.is_none(),
                None if item.canon.is_some() => {}
                None => match first.entry(invariant) {
                    std::collections::hash_map::Entry::Vacant(slot) => {
                        slot.insert(i);
                    }
                    std::collections::hash_map::Entry::Occupied(earlier) => {
                        needed[*earlier.get()] = true;
                        needed[i] = true;
                    }
                },
            }
        }
        // The representatives' terms on this thread, from the search's
        // systems; their canonicalizations in parallel with the items'.
        let mut rep_works = Vec::new();
        let rep_terms: Vec<(ClassId, STerm)> = reps
            .iter()
            .map(|&r| {
                let start = Instant::now();
                let term = rep_term(r).unwrap_or_else(|| {
                    panic!("class #{r} is pending, but the search kept no system for its representative")
                });
                let mut w = WorkStats::default();
                w.canon_time += start.elapsed();
                rep_works.push(w);
                (r, term)
            })
            .collect();
        let targets: Vec<&mut Prepared> = items
            .iter_mut()
            .zip(needed)
            .filter(|(_, needed)| *needed)
            .map(|(item, _)| &mut **item)
            .collect();
        let (item_works, rep_canons): (Vec<WorkStats>, Vec<(ClassId, Canon, WorkStats)>) =
            rayon::join(
                || {
                    targets
                        .into_par_iter()
                        .map(|item| {
                            let mut w = WorkStats::default();
                            let term = item
                                .term
                                .take()
                                .expect("on: an unkeyed system keeps its term");
                            item.canon = Some(canon_of_term(&term, &mut w));
                            w
                        })
                        .collect()
                },
                || {
                    rep_terms
                        .into_par_iter()
                        .map(|(r, term)| {
                            let mut w = WorkStats::default();
                            let canon = canon_of_term(&term, &mut w);
                            (r, canon, w)
                        })
                        .collect()
                },
            );
        for (r, canon, w) in rep_canons {
            work.add(&w);
            if self.pending[&r].set(canon).is_err() {
                unreachable!("class #{r}: only unkeyed representatives are canonicalized here");
            }
        }
        for w in item_works.iter().chain(&rep_works) {
            work.add(w);
        }
    }

    /// Places `item` (see the module docs): into the class it is canonically
    /// equal to, or into a new one, which `new_class` makes. A pending class
    /// `item` collides with gets the key [`Self::precompute`] computed for
    /// its representative; a key of `item`'s own that no one computed is
    /// computed here.
    pub(crate) fn intern(
        &mut self,
        item: Prepared,
        new_class: impl FnOnce() -> ClassId,
        work: &mut WorkStats,
    ) -> Placed {
        let Prepared {
            term,
            invariant,
            canon,
        } = item;
        if self.mode != CanonMode::On {
            let canon = canon.expect("off and check: every prepared system has its key");
            if let Some(invariant) = invariant {
                let hit = self.by_invariant.contains_key(&invariant);
                let placed = self.place_keyed(canon, Some(invariant), new_class);
                self.count(hit, &placed);
                return placed;
            }
            return self.place_keyed(canon, None, new_class);
        }
        let invariant = invariant.expect("on: every prepared system has an invariant");
        let Some(bucket) = self.by_invariant.get(&invariant).copied() else {
            if let Some(canon) = canon {
                // Keyed in `precompute`: an earlier system of the batch has
                // the same invariant.
                return self.place_keyed(canon, Some(invariant), new_class);
            }
            let class = new_class();
            self.by_invariant.insert(invariant, Bucket::Pending(class));
            self.invariants.insert(class, invariant);
            self.pending.insert(class, OnceLock::new());
            return Placed {
                class,
                merged: false,
                labelling: None,
            };
        };
        if let Bucket::Pending(r) = bucket {
            self.key_pending(r, invariant);
        }
        let canon = canon.unwrap_or_else(|| {
            canon_of_term(
                term.as_ref().expect("on: an unkeyed system keeps its term"),
                work,
            )
        });
        let placed = self.place_keyed(canon, Some(invariant), new_class);
        self.count(true, &placed);
        placed
    }

    fn count(&mut self, hit: bool, placed: &Placed) {
        if hit {
            self.stats.inv_hits += 1;
            if !placed.merged {
                self.stats.inv_false += 1;
            }
        }
    }

    /// Places a canonicalized system by its key. A merge into a class with
    /// another invariant panics: two systems with one canonical form must
    /// have one invariant.
    fn place_keyed(
        &mut self,
        canon: Canon,
        invariant: Option<Fingerprint>,
        new_class: impl FnOnce() -> ClassId,
    ) -> Placed {
        if let Some(&class) = self.by_key.get(&canon.key) {
            if let Some(invariant) = invariant {
                let known = self.invariants[&class];
                assert_eq!(
                    known, invariant,
                    "class #{class}: two systems with one canonical form have different invariants"
                );
            }
            return Placed {
                class,
                merged: true,
                labelling: None,
            };
        }
        let class = new_class();
        self.by_key.insert(canon.key, class);
        if let Some(invariant) = invariant {
            self.by_invariant.insert(invariant, Bucket::Keyed);
            self.invariants.insert(class, invariant);
        }
        Placed {
            class,
            merged: false,
            labelling: Some(canon.labelling),
        }
    }

    /// Gives `r`, the pending class of `invariant`'s bucket, the key
    /// [`Self::precompute`] computed for it: a system with the same invariant
    /// is about to be placed.
    fn key_pending(&mut self, r: ClassId, invariant: Fingerprint) {
        let canon = self
            .pending
            .remove(&r)
            .and_then(OnceLock::into_inner)
            .unwrap_or_else(|| {
                panic!(
                    "class #{r}: a system with its invariant is placed, but its representative \
                     was not canonicalized first (precompute)"
                )
            });
        // No other class can have the key: one with the same form would have
        // the same invariant, and have come through this bucket.
        let other = self.by_key.insert(canon.key, r);
        assert!(
            other.is_none(),
            "class #{r}: its canonical form is class #{other:?}'s, under another invariant"
        );
        self.late.insert(r, canon.labelling);
        self.by_invariant.insert(invariant, Bucket::Keyed);
        self.newly_keyed.push(r);
    }

    /// The key of `r`, a pending class, computed once.
    fn rep_canon(&self, r: ClassId, rep_term: &impl RepTerm, work: &mut WorkStats) -> &Canon {
        self.pending[&r].get_or_init(|| canon_of_rep(r, rep_term, work))
    }

    /// The class of a system canonicalized as `canon`, if the search placed
    /// one like it; `invariant` is the system's, computed only if needed (a
    /// pending class, whose representative's term `rep_term` builds). For
    /// the proof materializer, which revisits the search's systems.
    pub(crate) fn class_of(
        &self,
        canon: &Canon,
        invariant: impl FnOnce() -> Fingerprint,
        rep_term: impl RepTerm,
        work: &mut WorkStats,
    ) -> Option<ClassId> {
        if let Some(&class) = self.by_key.get(&canon.key) {
            return Some(class);
        }
        if self.mode != CanonMode::On {
            return None;
        }
        match *self.by_invariant.get(&invariant())? {
            Bucket::Pending(r) => {
                (self.rep_canon(r, &rep_term, work).key == canon.key).then_some(r)
            }
            Bucket::Keyed => None,
        }
    }

    /// The labelling of class `c`'s representative, if `c` was not keyed
    /// when placed: its methods kept pre-canonical forms. Computed once for
    /// a pending class, from the term `rep_term` builds.
    pub(crate) fn late_labelling(
        &self,
        c: ClassId,
        rep_term: impl RepTerm,
        work: &mut WorkStats,
    ) -> Option<&CanonLabelling> {
        if let Some(labelling) = self.late.get(&c) {
            return Some(labelling);
        }
        self.pending.get(&c)?;
        Some(&self.rep_canon(c, &rep_term, work).labelling)
    }

    /// Whether `c` is pending: the search keeps its representative's system
    /// for the table, after it is done with it itself.
    pub(crate) fn is_pending(&self, c: ClassId) -> bool {
        self.pending.contains_key(&c)
    }

    /// The pending classes keyed since the last call: the search no longer
    /// needs to keep their systems for the table.
    pub(crate) fn take_keyed(&mut self) -> Vec<ClassId> {
        std::mem::take(&mut self.newly_keyed)
    }

    /// The key and class of every class with a key.
    pub(crate) fn keys(&self) -> impl Iterator<Item = (&CanonicalSystemFingerprint, ClassId)> {
        self.by_key.iter().map(|(key, &class)| (key, class))
    }

    pub(crate) fn invariant_of(&self, c: ClassId) -> Option<&Fingerprint> {
        self.invariants.get(&c)
    }

    pub(crate) fn mode(&self) -> CanonMode {
        self.mode
    }

    pub(crate) fn keyed(&self) -> usize {
        self.by_key.len()
    }

    pub(crate) fn pending(&self) -> usize {
        self.pending.len()
    }

    pub(crate) fn stats(&self) -> TableStats {
        self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use tamarin_term::lterm::{LSort, LVar};
    use tamarin_term::vterm::var_term;

    use crate::bliss_proc::bliss_available;
    use crate::constraint::constraints::Goal;
    use crate::constraint::system::GoalStatus;
    use crate::fact::ku_fact;
    use tamarin_term::builtin::pair;
    use tamarin_term::lterm::{pub_term, LNTerm};

    /// A system with the goals `KU(t_n) @ #i.n`.
    fn ku(terms: Vec<LNTerm>) -> System {
        let mut sys = System::empty();
        sys.content_mut().goals = std::sync::Arc::new(
            terms
                .into_iter()
                .enumerate()
                .map(|(n, t)| {
                    let node = LVar::new("i", LSort::Node, n as u64);
                    (Goal::Action(node, ku_fact(t)), GoalStatus::default())
                })
                .collect(),
        );
        sys
    }

    fn msg(name: &str, idx: u64) -> LNTerm {
        var_term(LVar::new(name, LSort::Msg, idx))
    }

    /// `<x, y>` next to `<'a', <second, third>>`: every variable occurs
    /// twice, so `crossed(x, y, y, x)` and `crossed(x, y, x, y)` are
    /// different systems with the same invariant.
    fn crossed(x: &LNTerm, y: &LNTerm, second: &LNTerm, third: &LNTerm) -> System {
        ku(vec![
            pair(x.clone(), y.clone()),
            pair(pub_term("a"), pair(second.clone(), third.clone())),
        ])
    }

    /// `crossed(x, y, y, x)`, a renamed copy of it, `crossed(x, y, x, y)` (a
    /// false collision with the first) and a system with another invariant.
    fn four() -> [System; 4] {
        let (x, y) = (msg("x", 0), msg("y", 0));
        let (u, w) = (msg("u", 7), msg("w", 2));
        [
            crossed(&x, &y, &y, &x),
            crossed(&u, &w, &w, &u),
            crossed(&x, &y, &x, &y),
            ku(vec![msg("z", 3)]),
        ]
    }

    /// Places `systems` one by one, each new class numbered in order, as a
    /// search would: the search keeps each class's system, which builds a
    /// pending representative's term.
    fn place_all(table: &mut CanonicalTable, systems: &[System]) -> Vec<Placed> {
        let mut work = WorkStats::default();
        // Per class, the system that made it.
        let mut made_by: Vec<usize> = Vec::new();
        systems
            .iter()
            .enumerate()
            .map(|(i, sys)| {
                let mut item = table.prepare(sys, &mut work);
                let rep_term = |r: ClassId| Some(system_to_term(&systems[made_by[r as usize]]));
                table.precompute([&mut item], rep_term, &mut work);
                table.intern(
                    item,
                    || {
                        made_by.push(i);
                        made_by.len() as ClassId - 1
                    },
                    &mut work,
                )
            })
            .collect()
    }

    /// The term of class `c`'s representative, for a search whose classes
    /// `systems` made in order.
    fn rep_terms<'a>(systems: &'a [&'a System]) -> impl Fn(ClassId) -> Option<STerm> + 'a {
        move |c| systems.get(c as usize).map(|sys| system_to_term(sys))
    }

    fn classes(placed: &[Placed]) -> Vec<(ClassId, bool)> {
        placed.iter().map(|p| (p.class, p.merged)).collect()
    }

    /// Every mode places systems the same way: a renamed copy merges, and a
    /// false collision (same invariant, another form) stays apart.
    #[test]
    fn every_mode_places_systems_alike() {
        if !bliss_available() {
            return;
        }
        let systems = four();
        let mut results = Vec::new();
        for mode in [CanonMode::Off, CanonMode::On, CanonMode::Check] {
            let mut table = CanonicalTable::new(mode);
            results.push(classes(&place_all(&mut table, &systems)));
        }
        assert_eq!(
            results[0],
            vec![(0, false), (0, true), (1, false), (2, false)]
        );
        assert_eq!(results[1], results[0], "on");
        assert_eq!(results[2], results[0], "check");
    }

    /// Under `on`, a system with an invariant of its own is never
    /// canonicalized; a collision canonicalizes it, in its worker, and the
    /// representative, in `precompute`, from the system the search kept.
    #[test]
    fn on_canonicalizes_only_on_a_collision() {
        if !bliss_available() {
            return;
        }
        let [first, copy, other, lone] = four();
        let kept = [&first, &lone];
        let mut table = CanonicalTable::new(CanonMode::On);
        let mut work = WorkStats::default();
        let first = table.prepare(&first, &mut work);
        let placed = table.intern(first, || 0, &mut work);
        assert!(!placed.merged && placed.labelling.is_none());
        assert_eq!((work.canons, table.pending(), table.keyed()), (0, 1, 0));
        assert!(table.is_pending(0));

        let lone = table.prepare(&lone, &mut work);
        table.intern(lone, || 1, &mut work);
        assert_eq!(work.canons, 0, "another invariant: no bliss");

        // The renamed copy's worker keys the copy; `precompute` keys the
        // representative.
        let mut copy = table.prepare(&copy, &mut work);
        assert_eq!(work.canons, 1);
        table.precompute([&mut copy], rep_terms(&kept), &mut work);
        assert_eq!(work.canons, 2);
        let placed = table.intern(copy, || unreachable!("a renamed copy merges"), &mut work);
        assert_eq!((placed.class, placed.merged), (0, true));
        assert_eq!((table.pending(), table.keyed()), (1, 1));
        assert_eq!(
            table.take_keyed(),
            vec![0],
            "the search may drop class 0's system"
        );
        assert!(table.take_keyed().is_empty());
        assert!(table.late_labelling(0, rep_terms(&[]), &mut work).is_some());

        // A false collision: canonicalized, a class of its own.
        let mut other = table.prepare(&other, &mut work);
        table.precompute([&mut other], rep_terms(&kept), &mut work);
        let placed = table.intern(other, || 2, &mut work);
        assert!(!placed.merged && placed.labelling.is_some());
        let stats = table.stats();
        assert_eq!((stats.inv_hits, stats.inv_false), (2, 1));
        assert_eq!(work.canons, 3);
    }

    /// Two systems of one batch with an invariant new to the table: the
    /// workers cannot see the collision, `precompute` keys both, and they
    /// merge as when placed one after the other.
    #[test]
    fn precompute_keys_a_collision_inside_a_batch() {
        if !bliss_available() {
            return;
        }
        let [first, copy, _, lone] = four();
        let mut table = CanonicalTable::new(CanonMode::On);
        let mut work = WorkStats::default();
        let mut batch: Vec<Prepared> = [first, copy, lone]
            .iter()
            .map(|sys| table.prepare(sys, &mut work))
            .collect();
        assert_eq!(work.canons, 0);
        table.precompute(batch.iter_mut(), rep_terms(&[]), &mut work);
        assert_eq!(work.canons, 2, "the colliding pair, not the lone system");
        let canons = work.canons;
        let mut next = 0;
        let placed: Vec<(ClassId, bool)> = batch
            .into_iter()
            .map(|item| {
                let p = table.intern(
                    item,
                    || {
                        next += 1;
                        next - 1
                    },
                    &mut work,
                );
                (p.class, p.merged)
            })
            .collect();
        assert_eq!(placed, vec![(0, false), (0, true), (1, false)]);
        assert_eq!(work.canons, canons, "nothing left for intern to key");
    }

    /// The materializer finds a pending class by its key through the
    /// invariant, from the representative's kept system, and a system the
    /// search never placed nowhere.
    #[test]
    fn class_of_finds_a_pending_class() {
        if !bliss_available() {
            return;
        }
        let [first, copy, other, lone] = four();
        let mut table = CanonicalTable::new(CanonMode::On);
        let mut work = WorkStats::default();
        let placed = table.prepare(&first, &mut work);
        table.intern(placed, || 0, &mut work);
        let kept = [&first];
        let find = |sys: &System, work: &mut WorkStats| {
            let term = system_to_term(sys);
            let canon = canon_of_term(&term, work);
            table.class_of(&canon, || invariant(&term), rep_terms(&kept), work)
        };
        assert_eq!(find(&copy, &mut work), Some(0));
        assert_eq!(find(&other, &mut work), None);
        assert_eq!(find(&lone, &mut work), None);
    }
}
