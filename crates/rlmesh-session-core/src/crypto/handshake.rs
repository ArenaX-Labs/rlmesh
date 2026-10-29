//! The participant's handshake (§8.3): `Noise_IK_25519_ChaChaPoly_BLAKE2s`,
//! participant as initiator, host as responder. msg 1 carries no payload;
//! msg 2 carries the `HandshakeAccept` bytes. `k_up` and `k_down` are the
//! raw split: the first half is initiator → responder (`k_up`).

use super::lp::{LpTooLong, push_lp};
use super::noise::{self, Ephemeral};
use super::sealed::{KindSet, SealReceiver, SealSender, Security};
use super::{Key, Rng, StaticKeypair};
use crate::prelude::*;

/// The Noise protocol name.
pub const PATTERN: &str = "Noise_IK_25519_ChaChaPoly_BLAKE2s";
/// The label the prologue starts with.
pub const PROLOGUE_LABEL: &[u8] = b"rlmesh/v2 handshake";
/// Length of msg 1 (`e, es, s, ss` with an empty payload).
pub const MSG1_LEN: usize = 32 + 48 + 16;
/// msg 2's length over its payload (`e, ee, se` and the payload's tag).
pub const MSG2_OVERHEAD: usize = 32 + 16;

/// The prologue: `lp("rlmesh/v2 handshake") ‖ lp(tenant) ‖ lp(session) ‖
/// u64 host_instance ‖ lp(participant)`.
///
/// # Errors
/// [`LpTooLong`] if a field is over 65,535 bytes.
pub fn prologue(
    tenant: &str,
    session: &str,
    host_instance: u64,
    participant: &str,
) -> Result<Vec<u8>, LpTooLong> {
    let mut out = Vec::new();
    push_lp(&mut out, PROLOGUE_LABEL)?;
    push_lp(&mut out, tenant.as_bytes())?;
    push_lp(&mut out, session.as_bytes())?;
    out.extend_from_slice(&host_instance.to_le_bytes());
    push_lp(&mut out, participant.as_bytes())?;
    Ok(out)
}

/// Why a handshake (IK or pairing XX) failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum HandshakeError {
    /// The caller's RNG failed.
    #[error("the random number generator failed")]
    Rng,
    /// A message failed to authenticate: another prologue, an unexpected
    /// static key, or changed bytes.
    #[error("the handshake message failed to authenticate")]
    Rejected,
    /// A key exchange gave the all-zero result: the other side sent a
    /// low-order (non-contributory) static or ephemeral key.
    #[error("a low-order key in the handshake")]
    LowOrder,
    /// A message of the wrong length.
    #[error("malformed handshake message")]
    Malformed,
    /// A message carried a payload where the pattern's step has none.
    #[error("unexpected handshake payload")]
    UnexpectedPayload,
    /// A payload too long for one Noise message.
    #[error("the handshake payload is too long")]
    PayloadTooLong,
    /// snow failed in a way the inputs can't cause.
    #[error("internal handshake error")]
    Internal,
}

/// The kinds the participant sends under `k_up`: ctl requests,
/// heartbeats, participant reports and history queries.
pub const UP: KindSet = KindSet::of_const(&[1, 3, 5, 15]);
/// The kinds the host sends under `k_down`: ctl replies, heartbeat
/// echoes, key deliveries and history pages.
pub const DOWN: KindSet = KindSet::of_const(&[2, 4, 6, 10]);

/// The participant's result of an IK handshake ([`IkInitiator::finish`]):
/// `k_up`, `k_down` and `h`. The keys leave only as the participant's
/// sender (`k_up`) and receiver (`k_down`); the host's side has its own
/// type ([`HostKeys`]), so neither side can make a second sender for the
/// other's key.
///
/// ```compile_fail,E0599
/// # fn f(keys: rlmesh_session_core::crypto::handshake::ParticipantKeys) {
/// let _ = keys.into_host(1, 2); // a participant never sends under `k_down`
/// # }
/// ```
#[derive(Debug)]
pub struct ParticipantKeys {
    up: Key,
    down: Key,
    handshake_hash: [u8; 32],
}

impl ParticipantKeys {
    /// The handshake hash `h`.
    pub fn handshake_hash(&self) -> &[u8; 32] {
        &self.handshake_hash
    }

    /// A sender under `k_up` and a receiver under `k_down`, with the ids
    /// from `HandshakeAccept`.
    pub fn into_participant(self, up_key_id: u32, down_key_id: u32) -> (SealSender, SealReceiver) {
        (
            SealSender::new(up_key_id, self.up, UP, Security::Seal),
            SealReceiver::new(down_key_id, self.down, DOWN, Security::Seal),
        )
    }
}

/// The host's result of an IK handshake ([`IkResponder::accept`]): the
/// keys leave only as the host's sender (`k_down`) and receiver (`k_up`).
///
/// ```compile_fail,E0599
/// # fn f(keys: rlmesh_session_core::crypto::handshake::HostKeys) {
/// let _ = keys.into_participant(1, 2); // the host never sends under `k_up`
/// # }
/// ```
#[derive(Debug)]
pub struct HostKeys {
    up: Key,
    down: Key,
    handshake_hash: [u8; 32],
}

impl HostKeys {
    /// The handshake hash `h`.
    pub fn handshake_hash(&self) -> &[u8; 32] {
        &self.handshake_hash
    }

    /// A sender under `k_down` and a receiver under `k_up`, with the ids
    /// the host put in `HandshakeAccept`.
    pub fn into_host(self, up_key_id: u32, down_key_id: u32) -> (SealSender, SealReceiver) {
        (
            SealSender::new(down_key_id, self.down, DOWN, Security::Seal),
            SealReceiver::new(up_key_id, self.up, UP, Security::Seal),
        )
    }
}

/// (`h`, `k_up`, `k_down`) after the last message.
fn split(hs: &mut snow::HandshakeState) -> Result<([u8; 32], Key, Key), HandshakeError> {
    let handshake_hash = noise::hash(hs)?;
    let (up, down) = noise::split(hs);
    Ok((handshake_hash, up, down))
}

/// The participant's side, after sending msg 1.
pub struct IkInitiator {
    hs: snow::HandshakeState,
}

impl core::fmt::Debug for IkInitiator {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("IkInitiator").finish_non_exhaustive()
    }
}

impl IkInitiator {
    /// Start a handshake with the host whose static key is `host_static`,
    /// returning the state and msg 1. `rng` gives the ephemeral key.
    ///
    /// # Errors
    /// [`HandshakeError`].
    pub fn start(
        rng: &mut dyn Rng,
        local: &StaticKeypair,
        host_static: &[u8; 32],
        prologue: &[u8],
    ) -> Result<(Self, Vec<u8>), HandshakeError> {
        let ephemeral = Ephemeral::draw(rng)?;
        let mut hs = noise::builder(PATTERN, &ephemeral)?
            .local_private_key(local.private())
            .and_then(|b| b.remote_public_key(host_static))
            .and_then(|b| b.prologue(prologue))
            .and_then(|b| b.build_initiator())
            .map_err(|_| HandshakeError::Internal)?;
        let msg1 = noise::write(&mut hs, &[], MSG1_LEN)?;
        Ok((Self { hs }, msg1))
    }

    /// Read msg 2: the `HandshakeAccept` bytes and the keys.
    ///
    /// Consumes the state: a forged or corrupted msg 2 ends this attempt,
    /// and the participant starts a new handshake (§8.3 doesn't ask for
    /// more; the host's reply is a single query reply).
    ///
    /// # Errors
    /// [`HandshakeError`].
    pub fn finish(mut self, msg2: &[u8]) -> Result<(Vec<u8>, ParticipantKeys), HandshakeError> {
        let accept = noise::read(&mut self.hs, msg2)?;
        let (handshake_hash, up, down) = split(&mut self.hs)?;
        Ok((
            accept,
            ParticipantKeys {
                up,
                down,
                handshake_hash,
            },
        ))
    }
}

/// The host's side, after reading msg 1: the participant's static key is
/// known and must be checked before [`accept`](Self::accept).
pub struct IkResponder {
    hs: snow::HandshakeState,
    initiator_static: [u8; 32],
}

impl core::fmt::Debug for IkResponder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("IkResponder")
            .field("initiator_static", &self.initiator_static)
            .finish_non_exhaustive()
    }
}

impl IkResponder {
    /// Read msg 1 under `prologue` (built from the request's participant
    /// name and the host's own tenant, session and instance). `rng` gives
    /// msg 2's ephemeral key. A low-order static or ephemeral key is
    /// refused ([`HandshakeError::LowOrder`]).
    ///
    /// # Errors
    /// [`HandshakeError`]; `UnexpectedPayload` if msg 1 has a payload.
    pub fn read_request(
        rng: &mut dyn Rng,
        local: &StaticKeypair,
        prologue: &[u8],
        msg1: &[u8],
    ) -> Result<Self, HandshakeError> {
        let ephemeral = Ephemeral::draw(rng)?;
        let mut hs = noise::builder(PATTERN, &ephemeral)?
            .local_private_key(local.private())
            .and_then(|b| b.prologue(prologue))
            .and_then(|b| b.build_responder())
            .map_err(|_| HandshakeError::Internal)?;
        noise::read_empty(&mut hs, msg1)?;
        let initiator_static = noise::remote_static(&hs)?;
        Ok(Self {
            hs,
            initiator_static,
        })
    }

    /// The participant's static X25519 key, authenticated by msg 1.
    pub fn initiator_static(&self) -> &[u8; 32] {
        &self.initiator_static
    }

    /// Accept: msg 2 carrying the `HandshakeAccept` bytes, and the keys.
    ///
    /// # Errors
    /// [`HandshakeError`].
    pub fn accept(mut self, accept: &[u8]) -> Result<(Vec<u8>, HostKeys), HandshakeError> {
        let msg2 = noise::write(&mut self.hs, accept, MSG2_OVERHEAD)?;
        let (handshake_hash, up, down) = split(&mut self.hs)?;
        Ok((
            msg2,
            HostKeys {
                up,
                down,
                handshake_hash,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::sealed::Kind;
    use crate::crypto::test_rng::{Broken, TestRng};

    fn keys() -> (StaticKeypair, StaticKeypair) {
        (
            StaticKeypair::from_private(&mut [1; 32]),
            StaticKeypair::from_private(&mut [2; 32]),
        )
    }

    fn prologue_for(participant: &str) -> Vec<u8> {
        prologue("acme", "demo", 9, participant).unwrap()
    }

    #[test]
    fn prologue_is_the_exact_bytes() {
        let p = prologue("t", "ss", 0x0102, "p").unwrap();
        let mut want = vec![19, 0];
        want.extend_from_slice(b"rlmesh/v2 handshake");
        want.extend_from_slice(&[1, 0, b't', 2, 0, b's', b's']);
        want.extend_from_slice(&[2, 1, 0, 0, 0, 0, 0, 0]);
        want.extend_from_slice(&[1, 0, b'p']);
        assert_eq!(p, want);
        assert_eq!(prologue(&"x".repeat(65_536), "s", 0, "p"), Err(LpTooLong));
    }

    #[test]
    fn completes_and_both_sides_talk() {
        let (participant, host) = keys();
        let p = prologue_for("arm");
        let (init, msg1) =
            IkInitiator::start(&mut TestRng::new(1), &participant, host.public(), &p).unwrap();
        assert_eq!(msg1.len(), MSG1_LEN);
        let resp = IkResponder::read_request(&mut TestRng::new(2), &host, &p, &msg1).unwrap();
        assert_eq!(resp.initiator_static(), participant.public());
        let (msg2, host_keys) = resp.accept(b"accept").unwrap();
        assert_eq!(msg2.len(), MSG2_OVERHEAD + 6);
        let (accept, keys) = init.finish(&msg2).unwrap();
        assert_eq!(accept, b"accept");
        assert_eq!(keys.handshake_hash(), host_keys.handshake_hash());
        assert!(keys.up.ct_eq(&host_keys.up) && keys.down.ct_eq(&host_keys.down));
        assert!(!keys.up.ct_eq(&keys.down));

        let (mut p_tx, mut p_rx) = keys.into_participant(10, 11);
        let (mut h_tx, mut h_rx) = host_keys.into_host(10, 11);
        let ctl = "rlmesh/v2/acme/s/demo/ctl";
        let mut up = p_tx.seal(Kind::CtlRequest, ctl, b"attach").unwrap();
        assert_eq!(h_rx.open(ctl, &mut up).unwrap().body, b"attach");
        let mut down = h_tx.seal(Kind::CtlReply, ctl, b"ok").unwrap();
        assert_eq!(p_rx.open(ctl, &mut down).unwrap().body, b"ok");
        // Each direction carries only its own kinds.
        assert!(p_tx.seal(Kind::CtlReply, ctl, b"").is_err());
        assert!(h_tx.seal(Kind::CtlRequest, ctl, b"").is_err());
        assert!(p_tx.seal(Kind::Heartbeat, ctl, b"").is_ok());
        assert!(h_tx.seal(Kind::HeartbeatEcho, ctl, b"").is_ok());
    }

    #[test]
    fn the_wrong_prologue_fails() {
        let (participant, host) = keys();
        let (_, msg1) = IkInitiator::start(
            &mut TestRng::new(1),
            &participant,
            host.public(),
            &prologue_for("arm"),
        )
        .unwrap();
        for other in [
            prologue_for("arm2"),
            prologue("acme", "demo", 10, "arm").unwrap(),
        ] {
            assert_eq!(
                IkResponder::read_request(&mut TestRng::new(2), &host, &other, &msg1).unwrap_err(),
                HandshakeError::Rejected
            );
        }
    }

    #[test]
    fn the_wrong_responder_static_fails() {
        let (participant, host) = keys();
        let impostor = StaticKeypair::from_private(&mut [3; 32]);
        let p = prologue_for("arm");
        // The participant expects `host`; `impostor` can't read msg 1.
        let (_, msg1) =
            IkInitiator::start(&mut TestRng::new(1), &participant, host.public(), &p).unwrap();
        assert_eq!(
            IkResponder::read_request(&mut TestRng::new(2), &impostor, &p, &msg1).unwrap_err(),
            HandshakeError::Rejected
        );
        // And a reply from anyone but the expected host fails at the participant.
        let (init, _) =
            IkInitiator::start(&mut TestRng::new(1), &participant, host.public(), &p).unwrap();
        let (_, msg1_imp) =
            IkInitiator::start(&mut TestRng::new(1), &participant, impostor.public(), &p).unwrap();
        let resp =
            IkResponder::read_request(&mut TestRng::new(2), &impostor, &p, &msg1_imp).unwrap();
        let (msg2, _) = resp.accept(b"").unwrap();
        assert_eq!(init.finish(&msg2).unwrap_err(), HandshakeError::Rejected);
    }

    #[test]
    fn a_low_order_participant_static_is_refused() {
        use crate::crypto::noise::low_order;
        let (_, host) = keys();
        let p = prologue_for("arm");
        let mut attacker = low_order::builder(PATTERN, [4; 32])
            .local_private_key(&low_order::MARKER)
            .and_then(|b| b.remote_public_key(host.public()))
            .and_then(|b| b.prologue(&p))
            .and_then(|b| b.build_initiator())
            .unwrap();
        let msg1 = noise::write(&mut attacker, &[], MSG1_LEN).unwrap();
        assert_eq!(
            &msg1[..32],
            &crate::crypto::StaticKeypair::from_private(&mut [4; 32]).public()[..]
        );
        assert_eq!(
            IkResponder::read_request(&mut TestRng::new(2), &host, &p, &msg1).unwrap_err(),
            HandshakeError::LowOrder
        );
    }

    #[test]
    fn a_low_order_ephemeral_is_refused() {
        let (participant, host) = keys();
        let p = prologue_for("arm");
        let (_, mut msg1) =
            IkInitiator::start(&mut TestRng::new(1), &participant, host.public(), &p).unwrap();
        msg1[..32].fill(0);
        assert_eq!(
            IkResponder::read_request(&mut TestRng::new(2), &host, &p, &msg1).unwrap_err(),
            HandshakeError::LowOrder
        );
    }

    #[test]
    fn tampered_messages_fail() {
        let (participant, host) = keys();
        let p = prologue_for("arm");
        let (init, msg1) =
            IkInitiator::start(&mut TestRng::new(1), &participant, host.public(), &p).unwrap();
        for at in [0, 40, msg1.len() - 1] {
            let mut bad = msg1.clone();
            bad[at] ^= 1;
            assert!(IkResponder::read_request(&mut TestRng::new(2), &host, &p, &bad).is_err());
        }
        assert_eq!(
            IkResponder::read_request(&mut TestRng::new(2), &host, &p, &msg1[..50]).unwrap_err(),
            HandshakeError::Malformed
        );
        let resp = IkResponder::read_request(&mut TestRng::new(2), &host, &p, &msg1).unwrap();
        let (mut msg2, _) = resp.accept(b"accept").unwrap();
        let last = msg2.len() - 1;
        msg2[last] ^= 1;
        assert_eq!(init.finish(&msg2).unwrap_err(), HandshakeError::Rejected);
    }

    #[test]
    fn rejects_a_payload_in_msg1() {
        let (participant, host) = keys();
        let p = prologue_for("arm");
        let ephemeral = Ephemeral::draw(&mut TestRng::new(1)).unwrap();
        let mut hs = noise::builder(PATTERN, &ephemeral)
            .unwrap()
            .local_private_key(participant.private())
            .and_then(|b| b.remote_public_key(host.public()))
            .and_then(|b| b.prologue(&p))
            .and_then(|b| b.build_initiator())
            .unwrap();
        let msg1 = noise::write(&mut hs, b"x", MSG1_LEN).unwrap();
        assert_eq!(
            IkResponder::read_request(&mut TestRng::new(2), &host, &p, &msg1).unwrap_err(),
            HandshakeError::UnexpectedPayload
        );
    }

    #[test]
    fn a_failing_rng_fails_the_handshake() {
        let (participant, host) = keys();
        let p = prologue_for("arm");
        assert_eq!(
            IkInitiator::start(&mut Broken, &participant, host.public(), &p).unwrap_err(),
            HandshakeError::Rng
        );
    }

    #[test]
    fn rejects_an_oversized_accept() {
        let (participant, host) = keys();
        let p = prologue_for("arm");
        let (_, msg1) =
            IkInitiator::start(&mut TestRng::new(1), &participant, host.public(), &p).unwrap();
        let resp = IkResponder::read_request(&mut TestRng::new(2), &host, &p, &msg1).unwrap();
        assert_eq!(
            resp.accept(&vec![0; 65_535]).unwrap_err(),
            HandshakeError::PayloadTooLong
        );
    }
}
