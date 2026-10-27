//! The safety layer's cryptography (design §3, §8): byte encodings, the
//! `Sealed` and `Signed` envelopes, replay windows, the participant's IK
//! handshake and the machine-pairing exchange. Sans-IO like the rest of the
//! crate: randomness comes from the caller ([`Rng`]), and nothing here knows
//! about Zenoh, protobuf or a clock. Payloads are opaque bytes.
//!
//! - [`mod@lp`]: `lp` and `kdf` (§8.1).
//! - [`sealed`]: the `Sealed` envelope (§8.2), with a sender's per-kind `seq`
//!   counters and a receiver's replay windows ([`replay`]).
//! - [`signed`]: the host's `Signed` broadcasts (§8.7).
//! - [`handshake`]: `Noise_IK_25519_ChaChaPoly_BLAKE2s` (§8.3).
//! - [`pairing`]: Noise XX, then commit-reveal and the SAS (§3).
//!
//! Interop vectors (§8.8) are in `tests/vectors/crypto.json`.

use zeroize::Zeroize;

pub mod handshake;
pub mod lp;
mod noise;
pub mod pairing;
pub mod replay;
pub mod sealed;
pub mod signed;

pub use handshake::{HandshakeError, IkInitiator, IkKeys, IkResponder};
pub use lp::{LpTooLong, kdf, lp};
pub use pairing::{PairLink, PairingError, Sas};
pub use replay::ReplayWindow;
pub use sealed::{Kind, KindSet, OpenError, Opened, SealError, SealReceiver, SealSender};
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

/// A 32-byte symmetric key (a `Sealed` key, a raw-split half), wiped on drop.
#[derive(Clone, PartialEq, Eq)]
pub struct Key([u8; 32]);

impl Key {
    /// The key with these bytes.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// The key's bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
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

/// A static X25519 key pair: a host's or a participant's Noise identity.
#[derive(Clone)]
pub struct StaticKeypair {
    secret: x25519_dalek::StaticSecret,
    public: [u8; 32],
}

impl StaticKeypair {
    /// The key pair of this private key.
    pub fn from_private(private: [u8; 32]) -> Self {
        let secret = x25519_dalek::StaticSecret::from(private);
        let public = x25519_dalek::PublicKey::from(&secret).to_bytes();
        Self { secret, public }
    }

    /// A new key pair from `rng`.
    ///
    /// # Errors
    /// [`RngError`] if `rng` failed.
    pub fn generate(rng: &mut dyn Rng) -> Result<Self, RngError> {
        let mut private = [0u8; 32];
        rng.fill_bytes(&mut private)?;
        let pair = Self::from_private(private);
        private.zeroize();
        Ok(pair)
    }

    /// The public key.
    pub fn public(&self) -> &[u8; 32] {
        &self.public
    }

    /// The private key's bytes, for storage.
    pub fn private_bytes(&self) -> [u8; 32] {
        self.secret.to_bytes()
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
