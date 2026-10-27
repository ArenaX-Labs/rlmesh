//! snow, with the caller's randomness and key types that wipe themselves.
//!
//! snow resolves one `'static` RNG per handshake and draws exactly one
//! 32-byte ephemeral key from it. The ephemeral is drawn from the caller's
//! [`Rng`] into an [`Ephemeral`] first and served once, by a resolver that
//! resolves at most one RNG. The DH and cipher are this module's: the
//! X25519 secrets are `x25519_dalek::StaticSecret`s and the cipher keys
//! `ChaCha20Poly1305`s, which zeroize on drop (snow 0.10's own don't), and
//! the DH refuses a non-contributory (all-zero) result, so a low-order
//! static or ephemeral key fails the handshake. The hash is snow's BLAKE2s.

use core::cell::Cell;

use chacha20poly1305::aead::{AeadInPlace, KeyInit};
use chacha20poly1305::{ChaCha20Poly1305, Tag};
use snow::params::{CipherChoice, DHChoice, HashChoice, NoiseParams};
use snow::resolvers::{CryptoResolver, DefaultResolver};
use snow::types::{Cipher, Dh, Hash, Random};
use x25519_dalek::{PublicKey, StaticSecret};
use zeroize::{Zeroize, Zeroizing};

use super::handshake::HandshakeError;
use super::{Key, Rng};
use crate::prelude::*;

/// The longest Noise message.
pub(crate) const MAX_MESSAGE: usize = 65_535;
const TAG_LEN: usize = 16;

/// One handshake's ephemeral private key, drawn from the caller's RNG.
/// Kept (wiped on drop) where a side must be able to rebuild its state.
pub(crate) struct Ephemeral(Zeroizing<[u8; 32]>);

impl Ephemeral {
    pub(crate) fn draw(rng: &mut dyn Rng) -> Result<Self, HandshakeError> {
        let mut out = Self(Zeroizing::new([0; 32]));
        rng.fill_bytes(&mut out.0[..])
            .map_err(|_| HandshakeError::Rng)?;
        Ok(out)
    }
}

/// An RNG that serves one 32-byte draw, then fails.
struct Once {
    bytes: [u8; 32],
    live: bool,
}

impl Random for Once {
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), snow::Error> {
        if !self.live || dest.len() != self.bytes.len() {
            return Err(snow::Error::Rng);
        }
        dest.copy_from_slice(&self.bytes);
        self.bytes.zeroize();
        self.live = false;
        Ok(())
    }
}

impl Drop for Once {
    fn drop(&mut self) {
        self.bytes.zeroize();
    }
}

/// The resolver: this module's DH and cipher, snow's hash, and an RNG that
/// resolves once.
struct Resolver {
    rng: Cell<Option<Box<Once>>>,
}

impl CryptoResolver for Resolver {
    fn resolve_rng(&self) -> Option<Box<dyn Random>> {
        // Taken on the first resolve: a second RNG never repeats the draw.
        self.rng.take().map(|once| once as Box<dyn Random>)
    }
    fn resolve_dh(&self, choice: &DHChoice) -> Option<Box<dyn Dh>> {
        matches!(choice, DHChoice::Curve25519).then(|| Box::new(X25519::default()) as Box<dyn Dh>)
    }
    fn resolve_hash(&self, choice: &HashChoice) -> Option<Box<dyn Hash>> {
        DefaultResolver.resolve_hash(choice)
    }
    fn resolve_cipher(&self, choice: &CipherChoice) -> Option<Box<dyn Cipher>> {
        matches!(choice, CipherChoice::ChaChaPoly)
            .then(|| Box::new(ChaChaPoly::default()) as Box<dyn Cipher>)
    }
}

/// X25519 over a `StaticSecret` (wiped on drop), refusing all-zero results.
#[derive(Default)]
struct X25519 {
    secret: Option<StaticSecret>,
    public: [u8; 32],
}

impl X25519 {
    /// Take the private key, wiping `private` in place. (x25519-dalek takes
    /// the bytes by value: that argument is the one copy left to the
    /// compiler.)
    fn install(&mut self, private: &mut [u8; 32]) {
        let secret = StaticSecret::from(*private);
        private.zeroize();
        self.public = PublicKey::from(&secret).to_bytes();
        self.secret = Some(secret);
    }
}

impl Dh for X25519 {
    fn name(&self) -> &'static str {
        "25519"
    }
    fn pub_len(&self) -> usize {
        32
    }
    fn priv_len(&self) -> usize {
        32
    }
    fn set(&mut self, privkey: &[u8]) {
        let mut private = Zeroizing::new([0u8; 32]);
        let n = privkey.len().min(32);
        private[..n].copy_from_slice(&privkey[..n]);
        self.install(&mut private);
    }
    fn generate(&mut self, rng: &mut dyn Random) -> Result<(), snow::Error> {
        let mut private = Zeroizing::new([0u8; 32]);
        rng.try_fill_bytes(&mut private[..])?;
        self.install(&mut private);
        Ok(())
    }
    fn pubkey(&self) -> &[u8] {
        &self.public
    }
    fn privkey(&self) -> &[u8] {
        self.secret.as_ref().map_or(&[], |s| s.as_bytes())
    }
    fn dh(&self, pubkey: &[u8], out: &mut [u8]) -> Result<(), snow::Error> {
        let secret = self.secret.as_ref().ok_or(snow::Error::Dh)?;
        let remote: [u8; 32] = pubkey
            .get(..32)
            .and_then(|k| k.try_into().ok())
            .ok_or(snow::Error::Dh)?;
        let shared = secret.diffie_hellman(&PublicKey::from(remote));
        if !shared.was_contributory() {
            return Err(snow::Error::Dh);
        }
        out.get_mut(..32)
            .ok_or(snow::Error::Dh)?
            .copy_from_slice(shared.as_bytes());
        Ok(())
    }
}

/// ChaCha20-Poly1305 holding its key in the AEAD (wiped on drop, and on
/// every `set` as the old one drops).
struct ChaChaPoly(ChaCha20Poly1305);

impl Default for ChaChaPoly {
    fn default() -> Self {
        Self(ChaCha20Poly1305::new(&[0u8; 32].into()))
    }
}

fn noise_nonce(n: u64) -> chacha20poly1305::Nonce {
    let mut nonce = [0u8; 12];
    nonce[4..].copy_from_slice(&n.to_le_bytes());
    nonce.into()
}

impl Cipher for ChaChaPoly {
    fn name(&self) -> &'static str {
        "ChaChaPoly"
    }
    fn set(&mut self, key: &[u8; 32]) {
        self.0 = ChaCha20Poly1305::new(key.into());
    }
    fn encrypt(&self, nonce: u64, authtext: &[u8], plaintext: &[u8], out: &mut [u8]) -> usize {
        let len = plaintext.len();
        out[..len].copy_from_slice(plaintext);
        let tag = self
            .0
            .encrypt_in_place_detached(&noise_nonce(nonce), authtext, &mut out[..len])
            .unwrap_or_else(|_| unreachable!("a Noise message is far below the AEAD's limit"));
        out[len..len + TAG_LEN].copy_from_slice(&tag);
        len + TAG_LEN
    }
    fn decrypt(
        &self,
        nonce: u64,
        authtext: &[u8],
        ciphertext: &[u8],
        out: &mut [u8],
    ) -> Result<usize, snow::Error> {
        let len = ciphertext
            .len()
            .checked_sub(TAG_LEN)
            .ok_or(snow::Error::Decrypt)?;
        let body = out.get_mut(..len).ok_or(snow::Error::Decrypt)?;
        body.copy_from_slice(&ciphertext[..len]);
        let tag = Tag::clone_from_slice(&ciphertext[len..]);
        self.0
            .decrypt_in_place_detached(&noise_nonce(nonce), authtext, body, &tag)
            .map_err(|_| snow::Error::Decrypt)?;
        Ok(len)
    }
}

/// A builder for `pattern` whose one ephemeral key is `ephemeral`.
pub(crate) fn builder(
    pattern: &str,
    ephemeral: &Ephemeral,
) -> Result<snow::Builder<'static>, HandshakeError> {
    let params: NoiseParams = pattern.parse().map_err(|_| HandshakeError::Internal)?;
    // Filled in place on the heap, so no stack copy is left behind.
    let mut once = Box::new(Once {
        bytes: [0; 32],
        live: true,
    });
    once.bytes.copy_from_slice(&ephemeral.0[..]);
    let resolver = Box::new(Resolver {
        rng: Cell::new(Some(once)),
    });
    Ok(snow::Builder::with_resolver(params, resolver))
}

/// Write the next message with `payload`, `overhead` bytes longer than it.
pub(crate) fn write(
    hs: &mut snow::HandshakeState,
    payload: &[u8],
    overhead: usize,
) -> Result<Vec<u8>, HandshakeError> {
    if payload.len() + overhead > MAX_MESSAGE {
        return Err(HandshakeError::PayloadTooLong);
    }
    // snow wants room for a payload tag even before there is a key.
    let mut out = vec![0u8; payload.len() + overhead + TAG_LEN];
    let n = hs.write_message(payload, &mut out).map_err(map)?;
    out.truncate(n);
    Ok(out)
}

/// Read the next message and return its payload.
pub(crate) fn read(hs: &mut snow::HandshakeState, msg: &[u8]) -> Result<Vec<u8>, HandshakeError> {
    if msg.len() > MAX_MESSAGE {
        return Err(HandshakeError::Malformed);
    }
    let mut payload = vec![0u8; msg.len()];
    let n = hs.read_message(msg, &mut payload).map_err(map)?;
    payload.truncate(n);
    Ok(payload)
}

/// Read a message whose payload must be empty.
pub(crate) fn read_empty(hs: &mut snow::HandshakeState, msg: &[u8]) -> Result<(), HandshakeError> {
    if read(hs, msg)?.is_empty() {
        Ok(())
    } else {
        Err(HandshakeError::UnexpectedPayload)
    }
}

/// The handshake hash `h`.
pub(crate) fn hash(hs: &snow::HandshakeState) -> Result<[u8; 32], HandshakeError> {
    hs.get_handshake_hash()
        .try_into()
        .map_err(|_| HandshakeError::Internal)
}

/// The remote static key, once the handshake has it.
pub(crate) fn remote_static(hs: &snow::HandshakeState) -> Result<[u8; 32], HandshakeError> {
    hs.get_remote_static()
        .and_then(|key| key.try_into().ok())
        .ok_or(HandshakeError::Internal)
}

/// The raw split as keys: (initiator → responder, responder → initiator).
pub(crate) fn split(hs: &mut snow::HandshakeState) -> (Key, Key) {
    let (mut first, mut second) = hs.dangerously_get_raw_split();
    let keys = (Key::from_bytes(first), Key::from_bytes(second));
    first.zeroize();
    second.zeroize();
    keys
}

fn map(error: snow::Error) -> HandshakeError {
    match error {
        snow::Error::Decrypt => HandshakeError::Rejected,
        snow::Error::Dh => HandshakeError::LowOrder,
        snow::Error::Input => HandshakeError::Malformed,
        snow::Error::Rng => HandshakeError::Rng,
        _ => HandshakeError::Internal,
    }
}

/// A resolver whose static key claims a low-order public key (u = 0), as an
/// attacker would send: every DH with it is zero, which the attacker can
/// "compute" without a secret.
#[cfg(test)]
pub(crate) mod low_order {
    use super::*;

    /// The private-key marker that selects the low-order static.
    pub(crate) const MARKER: [u8; 32] = [0xee; 32];

    #[derive(Default)]
    struct Attacker {
        real: X25519,
        low: bool,
    }

    impl Dh for Attacker {
        fn name(&self) -> &'static str {
            "25519"
        }
        fn pub_len(&self) -> usize {
            32
        }
        fn priv_len(&self) -> usize {
            32
        }
        fn set(&mut self, privkey: &[u8]) {
            self.low = privkey == MARKER;
            self.real.set(privkey);
        }
        fn generate(&mut self, rng: &mut dyn Random) -> Result<(), snow::Error> {
            self.real.generate(rng)
        }
        fn pubkey(&self) -> &[u8] {
            if self.low {
                &[0; 32]
            } else {
                self.real.pubkey()
            }
        }
        fn privkey(&self) -> &[u8] {
            self.real.privkey()
        }
        fn dh(&self, pubkey: &[u8], out: &mut [u8]) -> Result<(), snow::Error> {
            if self.low {
                out[..32].fill(0);
                return Ok(());
            }
            self.real.dh(pubkey, out)
        }
    }

    struct AttackerResolver(Resolver);

    impl CryptoResolver for AttackerResolver {
        fn resolve_rng(&self) -> Option<Box<dyn Random>> {
            self.0.resolve_rng()
        }
        fn resolve_dh(&self, _: &DHChoice) -> Option<Box<dyn Dh>> {
            Some(Box::new(Attacker::default()))
        }
        fn resolve_hash(&self, choice: &HashChoice) -> Option<Box<dyn Hash>> {
            self.0.resolve_hash(choice)
        }
        fn resolve_cipher(&self, choice: &CipherChoice) -> Option<Box<dyn Cipher>> {
            self.0.resolve_cipher(choice)
        }
    }

    /// A builder whose static key is low-order when set to [`MARKER`].
    pub(crate) fn builder(pattern: &str, ephemeral: [u8; 32]) -> snow::Builder<'static> {
        let once = Box::new(Once {
            bytes: ephemeral,
            live: true,
        });
        let resolver = AttackerResolver(Resolver {
            rng: Cell::new(Some(once)),
        });
        snow::Builder::with_resolver(pattern.parse().unwrap(), Box::new(resolver))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rng_resolves_once() {
        let once = Box::new(Once {
            bytes: [7; 32],
            live: true,
        });
        let resolver = Resolver {
            rng: Cell::new(Some(once)),
        };
        let mut first = resolver.resolve_rng().unwrap();
        assert!(
            resolver.resolve_rng().is_none(),
            "a second resolve gets nothing"
        );
        let mut out = [0u8; 32];
        first.try_fill_bytes(&mut out).unwrap();
        assert_eq!(out, [7; 32]);
        assert!(
            first.try_fill_bytes(&mut out).is_err(),
            "and the one RNG draws once"
        );
    }

    #[test]
    fn x25519_refuses_low_order_points() {
        let mut dh = X25519::default();
        dh.set(&[1; 32]);
        let mut out = [0u8; 32];
        let mut one = [0u8; 32];
        one[0] = 1;
        // u = 0 and u = 1 are low-order: every scalar gives zero.
        for point in [[0u8; 32], one] {
            assert_eq!(dh.dh(&point, &mut out), Err(snow::Error::Dh));
        }
        let other = PublicKey::from(&StaticSecret::from([2; 32])).to_bytes();
        assert!(dh.dh(&other, &mut out).is_ok());
    }

    #[test]
    fn the_cipher_round_trips_and_rejects_a_bad_tag() {
        let mut c = ChaChaPoly::default();
        c.set(&[3; 32]);
        let mut sealed = [0u8; 21];
        assert_eq!(c.encrypt(5, b"ad", b"hello", &mut sealed), 21);
        let mut out = [0u8; 5];
        assert_eq!(c.decrypt(5, b"ad", &sealed, &mut out), Ok(5));
        assert_eq!(&out, b"hello");
        sealed[20] ^= 1;
        assert_eq!(
            c.decrypt(5, b"ad", &sealed, &mut out),
            Err(snow::Error::Decrypt)
        );
        assert_eq!(
            c.decrypt(5, b"ad", &sealed[..10], &mut out),
            Err(snow::Error::Decrypt)
        );
    }
}
