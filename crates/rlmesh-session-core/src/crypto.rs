//! The safety layer's cryptography: byte encodings, the
//! `Sealed` and `Signed` envelopes, replay windows, the participant's IK
//! handshake and the machine-pairing exchange. Sans-IO like the rest of the
//! crate: randomness comes from the caller ([`Rng`]), and nothing here knows
//! about Zenoh, protobuf or a clock. Payloads are opaque bytes.
//!
//! - [`mod@lp`][]: `lp`.
//! - [`sealed`]: the `Sealed` envelope, with a sender's per-kind `seq`
//!   counters and a receiver's replay windows ([`replay`]).
//! - [`signed`]: the host's `Signed` broadcasts.
//! - [`handshake`][]: `Noise_IK_25519_ChaChaPoly_BLAKE2s`.
//! - [`pairing`]: Noise XX, then commit-reveal and the SAS.
//!
//! Interop vectors are in `tests/vectors/crypto.json`.

use zeroize::Zeroize;

pub mod handshake;
pub mod lp;
mod noise;
pub mod pairing;
pub mod replay;
pub mod sealed;
pub mod signed;

pub use handshake::{HandshakeError, HostKeys, IkInitiator, IkResponder, ParticipantKeys};
pub use lp::{LpTooLong, lp};
pub use pairing::{PairLink, PairingError, Sas};
pub use replay::ReplayWindow;
pub use sealed::{
    Kind, KindSet, OpenError, Opened, Opening, SealError, SealReceiver, SealSender, Sealing,
    Security,
};
pub use signed::{SignError, SignedError, SignedKind, SignedReceiver, SignedSender};

/// A source of cryptographic randomness, implemented by the caller: the OS
/// on a host, the hardware RNG on a board, a fixed stream in tests.
pub trait Rng {
    /// Fill `dest` entirely with random bytes.
    ///
    /// # Errors
    /// [`RngError`] if the source failed; nothing drawn is used then.
    fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), RngError>;
}

/// The caller's [`Rng`] failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("the random number generator failed")]
pub struct RngError;

/// A 32-byte symmetric key (a `Sealed` key, a raw-split half), wiped on
/// drop. Not `Clone`: a key goes into exactly one [`SealSender`] (or
/// [`SealReceiver`]) by value, so one process can't run two `seq` counters
/// from 1 under the same key and reuse a nonce. Bytes leave only through
/// [`as_bytes`](Self::as_bytes), for the host's key delivery.
pub struct Key([u8; 32]);

impl Key {
    /// The key with these bytes: a delivered key or a stored one.
    ///
    /// The type can't stop bytes from being duplicated: the host holds every
    /// key it delivers, so keeping one sender per key and never restarting a
    /// key's counters (by building a second [`SealSender`] from the same
    /// bytes) is the caller's, and in practice the host's, responsibility.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// The key's bytes, to deliver it.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Whether two keys are equal, in constant time.
    pub fn ct_eq(&self, other: &Self) -> bool {
        subtle::ConstantTimeEq::ct_eq(&self.0[..], &other.0[..]).into()
    }
}

impl Drop for Key {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

impl core::fmt::Debug for Key {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("Key(..)")
    }
}

/// `secret`'s public key, by the same Montgomery ladder every key exchange
/// runs. `PublicKey::from` multiplies on the Edwards curve instead, which
/// without precomputed tables holds a lookup table on the stack (about 2 KB
/// more, on a board).
pub(crate) fn x25519_public(secret: &x25519_dalek::StaticSecret) -> [u8; 32] {
    let base = x25519_dalek::PublicKey::from(x25519_dalek::X25519_BASEPOINT_BYTES);
    secret.diffie_hellman(&base).to_bytes()
}

/// A static X25519 key pair: a host's or a participant's Noise identity.
#[derive(Clone)]
pub struct StaticKeypair {
    secret: x25519_dalek::StaticSecret,
    public: [u8; 32],
}

impl StaticKeypair {
    /// The key pair of this private key, which is wiped in place.
    pub fn from_private(private: &mut [u8; 32]) -> Self {
        // x25519-dalek takes the bytes by value: that argument is the one
        // copy left to the compiler.
        let secret = x25519_dalek::StaticSecret::from(*private);
        private.zeroize();
        let public = x25519_public(&secret);
        Self { secret, public }
    }

    /// A new key pair from `rng`.
    ///
    /// # Errors
    /// [`RngError`] if `rng` failed.
    pub fn generate(rng: &mut dyn Rng) -> Result<Self, RngError> {
        let mut private = zeroize::Zeroizing::new([0u8; 32]);
        rng.fill_bytes(&mut private[..])?;
        Ok(Self::from_private(&mut private))
    }

    /// The public key.
    pub fn public(&self) -> &[u8; 32] {
        &self.public
    }

    /// Copies the private key into `out`, for storage. It is copied from
    /// a reference, so no temporary copy is left behind; wiping `out` is
    /// the caller's job (e.g. a `Zeroizing` buffer).
    pub fn write_private(&self, out: &mut [u8; 32]) {
        out.copy_from_slice(self.secret.as_bytes());
    }

    fn private(&self) -> &[u8; 32] {
        self.secret.as_bytes()
    }
}

impl core::fmt::Debug for StaticKeypair {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("StaticKeypair")
            .field("public", &self.public)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ladder and the Edwards multiply agree: the same public key.
    #[test]
    fn x25519_public_is_the_edwards_result() {
        for byte in [0_u8, 1, 0x42, 0x7f, 0x80, 0xff] {
            let mut private = [byte; 32];
            private[0] = byte.wrapping_add(3);
            let secret = x25519_dalek::StaticSecret::from(private);
            let want = x25519_dalek::PublicKey::from(&secret).to_bytes();
            assert_eq!(x25519_public(&secret), want);
            assert_eq!(*StaticKeypair::from_private(&mut private).public(), want);
        }
    }

    const SECRET: [u8; 32] = [0xa7; 32];

    #[test]
    fn a_keypair_writes_back_the_private_key_it_was_made_from() {
        let mut private = SECRET;
        let pair = StaticKeypair::from_private(&mut private);
        let mut stored = [0; 32];
        pair.write_private(&mut stored);
        let again = StaticKeypair::from_private(&mut stored);
        assert_eq!(again.public(), pair.public());
    }

    #[test]
    fn a_key_and_a_keypair_never_print_their_secret() {
        let mut private = SECRET;
        assert_redacted(&Key::from_bytes(SECRET), &SECRET);
        assert_redacted(&StaticKeypair::from_private(&mut private), &SECRET);
    }

    #[test]
    fn sealing_and_opening_state_never_print_the_key() {
        let kinds = KindSet::of(&[Kind::Sample]);
        let mut sender = SealSender::new(1, Key::from_bytes(SECRET), kinds, Security::Mac);
        let receiver = SealReceiver::new(1, Key::from_bytes(SECRET), kinds, Security::Mac);
        let msg = sender.seal(Kind::Sample, "k", b"body").expect("seal");
        assert_redacted(&sender, &SECRET);
        assert_redacted(&receiver, &SECRET);
        assert_redacted(&sender.begin(Kind::Sample).expect("begin"), &SECRET);
        assert_redacted(&receiver.begin(&msg).expect("begin"), &SECRET);
    }

    #[test]
    fn a_signer_never_prints_its_signing_key() {
        assert_redacted(&SignedSender::new(&SECRET, 7), &SECRET);
    }
}

/// Asserts `value` prints something and none of it is `secret`, in the
/// forms a careless `Debug` would use: hex, a byte list, the raw bytes.
#[cfg(test)]
pub(crate) fn assert_redacted(value: &impl core::fmt::Debug, secret: &[u8]) {
    let hex: String = secret.iter().map(|b| format!("{b:02x}")).collect();
    let list: Vec<String> = secret.iter().map(ToString::to_string).collect();
    let list = list.join(",");
    for printed in [format!("{value:?}"), format!("{value:#?}")] {
        assert!(!printed.is_empty(), "it prints something");
        let compact: String = printed.split_whitespace().collect();
        assert!(!compact.to_lowercase().contains(&hex), "hex in {printed}");
        assert!(!compact.contains(&list), "the bytes in {printed}");
        assert!(
            !printed
                .as_bytes()
                .windows(secret.len())
                .any(|w| w == secret),
            "raw bytes in {printed}"
        );
    }
}

#[cfg(test)]
pub(crate) mod test_rng {
    use super::{Rng, RngError};

    /// A deterministic stream: byte i of draw n is a function of the seed,
    /// n and i. Not random; for tests only.
    pub(crate) struct TestRng {
        seed: u8,
        draws: u8,
    }

    impl TestRng {
        pub(crate) fn new(seed: u8) -> Self {
            Self { seed, draws: 0 }
        }
    }

    impl Rng for TestRng {
        fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), RngError> {
            self.draws = self.draws.wrapping_add(1);
            for (i, b) in dest.iter_mut().enumerate() {
                *b = self.seed.wrapping_mul(31)
                    ^ self.draws.wrapping_mul(17)
                    ^ (i as u8).wrapping_mul(7);
            }
            Ok(())
        }
    }

    /// An RNG that always fails.
    pub(crate) struct Broken;

    impl Rng for Broken {
        fn fill_bytes(&mut self, _: &mut [u8]) -> Result<(), RngError> {
            Err(RngError)
        }
    }
}
