//! A participant's credential and the IK handshake it runs: what it
//! attaches with.

use super::local_keypair;
use crate::crypto::StaticKeypair;
use crate::crypto::handshake::{self, HandshakeError, IkInitiator};
use crate::crypto::{Rng, SealReceiver, SealSender, SignedReceiver};
use crate::prelude::*;
use crate::wire::codec;
use crate::wire::{HandshakeAccept, HandshakeRequest, ParticipantId};

/// What a participant attaches with: its device's static key and the
/// host's static public key.
#[derive(Clone)]
pub struct Credential {
    device: StaticKeypair,
    host_static: [u8; 32],
}

impl core::fmt::Debug for Credential {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Credential")
            .field("device", &self.device)
            .field("host_static", &self.host_static)
            .finish()
    }
}

/// Two credentials are one when they hold the same device key for the
/// same host.
impl PartialEq for Credential {
    fn eq(&self, other: &Self) -> bool {
        self.device.public() == other.device.public() && self.host_static == other.host_static
    }
}

impl Eq for Credential {}

impl Credential {
    /// A device's key pair and the host's static public key.
    pub fn new(device: StaticKeypair, host_static: [u8; 32]) -> Self {
        Self {
            device,
            host_static,
        }
    }

    /// The `local` device of an embedded host, from its address: the local
    /// secret and the host's static public key.
    pub fn local(secret: &[u8; 32], host_static: [u8; 32]) -> Self {
        Self::new(local_keypair(secret), host_static)
    }

    /// The same device's key against another host's static key: a robot's
    /// IK with a satellite's run host, whose key its `Join` names.
    pub fn with_host(&self, host_static: [u8; 32]) -> Self {
        Self::new(self.device.clone(), host_static)
    }

    /// The device's static public key.
    pub fn device_public(&self) -> &[u8; 32] {
        self.device.public()
    }

    /// The host's static public key.
    pub fn host_static(&self) -> &[u8; 32] {
        &self.host_static
    }

    /// Starts a handshake as `participant` with the host instance
    /// `host_instance` (read from the public `host/state`; a wrong value
    /// only makes the handshake fail): the state and msg 1.
    ///
    /// # Errors
    /// [`HandshakeError`].
    pub fn start(
        &self,
        rng: &mut dyn Rng,
        tenant: &str,
        session: &str,
        host_instance: u64,
        participant: &ParticipantId,
    ) -> Result<(IkInitiator, HandshakeRequest), HandshakeError> {
        let prologue = handshake::prologue(tenant, session, host_instance, participant.as_str())
            .map_err(|_| HandshakeError::PayloadTooLong)?;
        let (initiator, noise) =
            IkInitiator::start(rng, &self.device, &self.host_static, &prologue)?;
        Ok((
            initiator,
            HandshakeRequest {
                participant: participant.clone(),
                noise,
            },
        ))
    }
}

/// A participant's side of a completed handshake.
#[derive(Debug)]
pub struct Attached {
    /// What the host issued.
    pub accept: HandshakeAccept,
    /// `k_up`: ctl requests and heartbeats.
    pub up: SealSender,
    /// `k_down`: ctl replies and heartbeat echoes.
    pub down: SealReceiver,
    /// The host's broadcasts, from the authenticated instance and key.
    pub host: SignedReceiver,
}

/// Why a handshake did not leave the participant attached.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AuthError {
    /// msg 2 failed to authenticate.
    #[error(transparent)]
    Handshake(#[from] HandshakeError),
    /// msg 2's payload is not a `HandshakeAccept`, or names a bad key.
    #[error("malformed handshake accept: {0}")]
    Malformed(String),
}

/// Reads msg 2: the participant's keys and the host's signing key.
///
/// # Errors
/// [`AuthError`].
pub fn finish(initiator: IkInitiator, msg2: &[u8]) -> Result<Attached, AuthError> {
    let (payload, keys) = initiator.finish(msg2)?;
    let accept = codec::decode_handshake_accept(&payload)
        .map_err(|err| AuthError::Malformed(err.to_string()))?;
    let host = SignedReceiver::new(&accept.host_sign_key, accept.host_instance)
        .map_err(|err| AuthError::Malformed(err.to_string()))?;
    let (up, down) = keys.into_participant(accept.up_key_id, accept.down_key_id);
    Ok(Attached {
        accept,
        up,
        down,
        host,
    })
}
