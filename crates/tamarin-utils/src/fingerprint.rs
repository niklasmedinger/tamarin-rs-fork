//! A deterministic, cross-process content fingerprint.
//!
//! (No HS analog — used by the Rust port's canonization work, per
//! `work.tex`, to let a proof search keep a compact stand-in for a
//! canonical term/fact/rule/formula instead of the full object.)
//!
//! Rust's default `HashMap`/`HashSet` hasher (`RandomState`) seeds
//! per-process, so the same logical value hashes differently across
//! separate runs — useless for anything compared across processes (a disk
//! cache, parallel workers, golden tests). SHA-256 has no such seed: the
//! same bytes always produce the same digest, on any machine, in any
//! process. [`FingerprintHasher`] wraps it with a few self-delimiting
//! primitives so composing calls can never produce an accidental
//! collision by shifting a field boundary.

use sha2::{Digest, Sha256};

/// A 256-bit content fingerprint.
pub type Fingerprint = [u8; 32];

/// Incremental builder for a [`Fingerprint`]. Every method either writes a
/// FIXED number of bytes (`u8`/`u64`/`digest`) or LENGTH-PREFIXES a
/// variable-length one (`bytes`/`tag`), so no sequence of calls can ever be
/// confused with a different sequence that happens to concatenate to the
/// same bytes (e.g. `tag("ab"); tag("c")` vs `tag("a"); tag("bc")` — without
/// the length prefix these would hash identically).
///
/// This is the primitive; it carries no notion of "what a term/fact/rule/
/// formula is" — that type-specific field layout lives in each crate that
/// has one (`tamarin_term::fingerprint`, `tamarin_theory::canon`).
pub struct FingerprintHasher(Sha256);

impl FingerprintHasher {
    pub fn new() -> Self {
        FingerprintHasher(Sha256::new())
    }

    /// A short ASCII tag identifying WHAT is being hashed next (a variant
    /// name, a node kind, ...) — domain separation, so a differently-SHAPED
    /// value can never collide with this one merely by writing the same
    /// bytes in a different arrangement. Length-prefixed like [`Self::bytes`].
    pub fn tag(&mut self, s: &str) -> &mut Self {
        self.bytes(s.as_bytes())
    }

    /// A length-prefixed byte string.
    pub fn bytes(&mut self, b: &[u8]) -> &mut Self {
        self.0.update((b.len() as u64).to_le_bytes());
        self.0.update(b);
        self
    }

    pub fn u8(&mut self, n: u8) -> &mut Self {
        self.0.update([n]);
        self
    }

    pub fn u64(&mut self, n: u64) -> &mut Self {
        self.0.update(n.to_le_bytes());
        self
    }

    /// A previously-computed sub-fingerprint — the "subhash" half of a
    /// Merkle-style composition: a compound value's fingerprint is built
    /// from the fingerprints of its children, not their raw content, so
    /// hashing a large structure never redoes work below a node whose
    /// fingerprint is already known. Already fixed-size, so (unlike
    /// [`Self::bytes`]) it needs no length prefix — but every OTHER field
    /// still needs one, precisely so a `digest` can never be confused with
    /// a same-length `bytes`/`tag` call carrying different content.
    pub fn digest(&mut self, d: &Fingerprint) -> &mut Self {
        self.0.update(d);
        self
    }

    pub fn finish(self) -> Fingerprint {
        self.0.finalize().into()
    }
}

impl Default for FingerprintHasher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest_of(f: impl FnOnce(&mut FingerprintHasher)) -> Fingerprint {
        let mut h = FingerprintHasher::new();
        f(&mut h);
        h.finish()
    }

    #[test]
    fn same_calls_give_the_same_digest() {
        let a = digest_of(|h| {
            h.tag("App").bytes(b"f").u64(2);
        });
        let b = digest_of(|h| {
            h.tag("App").bytes(b"f").u64(2);
        });
        assert_eq!(a, b);
    }

    #[test]
    fn different_calls_give_different_digests() {
        let a = digest_of(|h| {
            h.tag("App").bytes(b"f").u64(2);
        });
        let b = digest_of(|h| {
            h.tag("App").bytes(b"g").u64(2);
        });
        assert_ne!(a, b);
    }

    /// The whole point of length-prefixing: two calls that would
    /// concatenate to the SAME raw bytes without it must still produce
    /// different digests.
    #[test]
    fn length_prefixing_prevents_field_boundary_collisions() {
        let a = digest_of(|h| {
            h.tag("ab").tag("c");
        });
        let b = digest_of(|h| {
            h.tag("a").tag("bc");
        });
        assert_ne!(a, b, "\"ab\"+\"c\" must not collide with \"a\"+\"bc\"");
    }

    /// Same point for `bytes` vs `digest`: a `digest` call must not be
    /// confused with a `bytes` call of the same length carrying different
    /// content — `digest` skips the length prefix since a `Fingerprint` is
    /// always exactly 32 bytes, so nothing else may skip it and still land
    /// on the same wire form.
    #[test]
    fn digest_is_not_confusable_with_equal_length_bytes() {
        let sub: Fingerprint = digest_of(|h| {
            h.tag("leaf");
        });
        let a = digest_of(|h| {
            h.digest(&sub);
        });
        let b = digest_of(|h| {
            h.bytes(&sub);
        });
        assert_ne!(a, b);
    }

    #[test]
    fn u8_and_u64_are_not_confusable() {
        let a = digest_of(|h| {
            h.u8(5);
        });
        let b = digest_of(|h| {
            h.u64(5);
        });
        assert_ne!(a, b);
    }
}
