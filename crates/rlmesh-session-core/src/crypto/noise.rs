//! snow, with the caller's randomness. snow resolves one `'static` RNG per
//! handshake and draws exactly one 32-byte ephemeral key from it, so the
//! bytes are drawn from the caller's [`Rng`] up front and served once.

use snow::params::{CipherChoice, DHChoice, HashChoice, NoiseParams};
use snow::resolvers::{CryptoResolver, DefaultResolver};
use snow::types::{Cipher, Dh, Hash, Random};
use zeroize::Zeroize;

use super::Rng;
use super::handshake::HandshakeError;
use crate::prelude::*;

/// The longest Noise message.
pub(crate) const MAX_MESSAGE: usize = 65_535;

/// One ephemeral key's worth of the caller's randomness, served once.
struct Drawn(Option<[u8; 32]>);

impl Random for Drawn {
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), snow::Error> {
        match self.0.take() {
            Some(mut bytes) if dest.len() == bytes.len() => {
                dest.copy_from_slice(&bytes);
                bytes.zeroize();
                Ok(())
            }
            Some(mut bytes) => {
                bytes.zeroize();
                Err(snow::Error::Rng)
            }
            None => Err(snow::Error::Rng),
        }
    }
}

impl Drop for Drawn {
    fn drop(&mut self) {
        if let Some(bytes) = self.0.as_mut() {
            bytes.zeroize();
        }
    }
}

/// snow's default primitives, with [`Drawn`] as the RNG.
struct Resolver {
    ephemeral: [u8; 32],
}

impl CryptoResolver for Resolver {
    fn resolve_rng(&self) -> Option<Box<dyn Random>> {
        Some(Box::new(Drawn(Some(self.ephemeral))))
    }
    fn resolve_dh(&self, choice: &DHChoice) -> Option<Box<dyn Dh>> {
        DefaultResolver.resolve_dh(choice)
    }
    fn resolve_hash(&self, choice: &HashChoice) -> Option<Box<dyn Hash>> {
        DefaultResolver.resolve_hash(choice)
    }
    fn resolve_cipher(&self, choice: &CipherChoice) -> Option<Box<dyn Cipher>> {
        DefaultResolver.resolve_cipher(choice)
    }
}

impl Drop for Resolver {
    fn drop(&mut self) {
        self.ephemeral.zeroize();
    }
}

/// A builder for `pattern` whose one ephemeral key comes from `rng`.
pub(crate) fn builder(
    pattern: &str,
    rng: &mut dyn Rng,
) -> Result<snow::Builder<'static>, HandshakeError> {
    let params: NoiseParams = pattern.parse().map_err(|_| HandshakeError::Internal)?;
    let mut ephemeral = [0u8; 32];
    rng.fill_bytes(&mut ephemeral)
        .map_err(|_| HandshakeError::Rng)?;
    let resolver = Resolver { ephemeral };
    ephemeral.zeroize();
    Ok(snow::Builder::with_resolver(params, Box::new(resolver)))
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
    let mut out = vec![0u8; payload.len() + overhead + 16];
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

fn map(error: snow::Error) -> HandshakeError {
    match error {
        snow::Error::Decrypt | snow::Error::Dh => HandshakeError::Rejected,
        snow::Error::Input => HandshakeError::Malformed,
        snow::Error::Rng => HandshakeError::Rng,
        _ => HandshakeError::Internal,
    }
}
