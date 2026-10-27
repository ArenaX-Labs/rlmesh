//! Machine pairing (§3): a Noise XX handshake between the laptop
//! (initiator) and rlmeshd (responder), then commit-reveal inside it, as in
//! Bluetooth numeric comparison:
//!
//! 1. rlmeshd draws `Nr` and sends `C = BLAKE2s(lp("rlmesh/v2 pair commit") ‖ h ‖ Nr)`.
//! 2. The laptop draws `Ni` and sends it.
//! 3. rlmeshd reveals `Nr`; the laptop checks it against `C`.
//! 4. Both show `SAS = u64_le(BLAKE2s(lp("rlmesh/v2 pair sas") ‖ h ‖ Ni ‖ Nr)[..8]) mod 10^6`.
//!
//! `h` is the handshake hash after XX. After XX every message is a `Sealed`
//! (kinds 16–19) under the XX raw split: key 1 laptop → rlmeshd, key 2
//! rlmeshd → laptop.

use subtle::ConstantTimeEq;
use zeroize::Zeroize;

use super::handshake::HandshakeError;
use super::lp::hash_labelled;
use super::sealed::{KindSet, SealReceiver, SealSender};
use super::{Key, Rng, RngError, StaticKeypair, noise};
use crate::prelude::*;

/// The Noise protocol name.
pub const PATTERN: &str = "Noise_XX_25519_ChaChaPoly_BLAKE2s";
/// The commitment's label.
pub const COMMIT_LABEL: &[u8] = b"rlmesh/v2 pair commit";
/// The SAS's label.
pub const SAS_LABEL: &[u8] = b"rlmesh/v2 pair sas";
/// The key id of laptop → rlmeshd messages.
pub const LAPTOP_KEY_ID: u32 = 1;
/// The key id of rlmeshd → laptop messages.
pub const DAEMON_KEY_ID: u32 = 2;
/// Length of XX msg 1 (`e`).
pub const XX1_LEN: usize = 32;
/// Length of XX msg 2 (`e, ee, s, es` and an empty payload's tag).
pub const XX2_LEN: usize = 32 + 48 + 16;
/// Length of XX msg 3 (`s, se` and an empty payload's tag).
pub const XX3_LEN: usize = 48 + 16;

/// Why pairing failed. Every failure aborts the exchange.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PairingError {
    /// The XX handshake failed.
    #[error(transparent)]
    Handshake(#[from] HandshakeError),
    /// The revealed `Nr` doesn't match the commitment `C`.
    #[error("the revealed nonce doesn't match the commitment")]
    RevealMismatch,
    /// A second `PairNonce` with a different `Ni` for the same commitment.
    #[error("a different nonce for the same commitment")]
    NonceChanged,
    /// The caller's RNG failed.
    #[error(transparent)]
    Rng(#[from] RngError),
}

/// A six-digit short authentication string, `000000`–`999999`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Sas(u32);

impl Sas {
    /// The code as a number below 10^6.
    pub fn value(self) -> u32 {
        self.0
    }
}

impl core::fmt::Display for Sas {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:06}", self.0)
    }
}

/// `C = BLAKE2s(lp("rlmesh/v2 pair commit") ‖ h ‖ Nr)`.
pub fn commitment(h: &[u8; 32], nr: &[u8; 32]) -> [u8; 32] {
    hash_labelled(COMMIT_LABEL, &[h, nr]).unwrap_or_else(|_| unreachable!("a short label"))
}

/// `SAS = u64_le(first 8 bytes of BLAKE2s(lp("rlmesh/v2 pair sas") ‖ h ‖ Ni ‖ Nr)) mod 10^6`.
pub fn sas(h: &[u8; 32], ni: &[u8; 32], nr: &[u8; 32]) -> Sas {
    let digest =
        hash_labelled(SAS_LABEL, &[h, ni, nr]).unwrap_or_else(|_| unreachable!("a short label"));
    let mut first = [0u8; 8];
    first.copy_from_slice(&digest[..8]);
    // The remainder is below 10^6, so it fits a u32.
    Sas((u64::from_le_bytes(first) % 1_000_000) as u32)
}

/// One side of a completed XX handshake: `h`, the other side's static key
/// and the sealed channel in both directions (pairing kinds only).
pub struct PairLink {
    handshake_hash: [u8; 32],
    remote_static: [u8; 32],
    /// This side's sender.
    pub sender: SealSender,
    /// This side's receiver.
    pub receiver: SealReceiver,
}

impl core::fmt::Debug for PairLink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PairLink")
            .field("handshake_hash", &self.handshake_hash)
            .field("remote_static", &self.remote_static)
            .finish_non_exhaustive()
    }
}

impl PairLink {
    fn new(hs: &mut snow::HandshakeState, laptop: bool) -> Result<Self, HandshakeError> {
        let handshake_hash = noise::hash(hs)?;
        let remote_static = noise::remote_static(hs)?;
        let (mut up, mut down) = hs.dangerously_get_raw_split();
        let (up_key, down_key) = (Key::from_bytes(up), Key::from_bytes(down));
        up.zeroize();
        down.zeroize();
        let up_tx = (LAPTOP_KEY_ID, &up_key);
        let down_tx = (DAEMON_KEY_ID, &down_key);
        let (send, recv) = if laptop {
            (up_tx, down_tx)
        } else {
            (down_tx, up_tx)
        };
        Ok(Self {
            handshake_hash,
            remote_static,
            sender: SealSender::new(send.0, send.1, KindSet::PAIRING),
            receiver: SealReceiver::new(recv.0, recv.1, KindSet::PAIRING),
        })
    }

    /// The handshake hash `h`.
    pub fn handshake_hash(&self) -> &[u8; 32] {
        &self.handshake_hash
    }

    /// The other side's static X25519 key (the host's, at the laptop; the
    /// laptop's, at rlmeshd).
    pub fn remote_static(&self) -> &[u8; 32] {
        &self.remote_static
    }
}

/// The laptop, after sending XX msg 1.
pub struct PairInitiator {
    hs: snow::HandshakeState,
}

impl core::fmt::Debug for PairInitiator {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PairInitiator").finish_non_exhaustive()
    }
}

impl PairInitiator {
    /// Start: the state and XX msg 1 (`xx1`).
    ///
    /// # Errors
    /// [`HandshakeError`].
    pub fn start(
        rng: &mut dyn Rng,
        local: &StaticKeypair,
    ) -> Result<(Self, Vec<u8>), HandshakeError> {
        let mut hs = noise::builder(PATTERN, rng)?
            .local_private_key(local.private())
            .and_then(|b| b.build_initiator())
            .map_err(|_| HandshakeError::Internal)?;
        let xx1 = noise::write(&mut hs, &[], XX1_LEN)?;
        Ok((Self { hs }, xx1))
    }

    /// Read XX msg 2 (`xx2`): the link and XX msg 3 (`xx3`) to send.
    ///
    /// # Errors
    /// [`HandshakeError`].
    pub fn finish(mut self, xx2: &[u8]) -> Result<(PairLink, Vec<u8>), HandshakeError> {
        if !noise::read(&mut self.hs, xx2)?.is_empty() {
            return Err(HandshakeError::UnexpectedPayload);
        }
        let xx3 = noise::write(&mut self.hs, &[], XX3_LEN)?;
        Ok((PairLink::new(&mut self.hs, true)?, xx3))
    }
}

/// rlmeshd, after replying with XX msg 2.
pub struct PairResponder {
    hs: snow::HandshakeState,
}

impl core::fmt::Debug for PairResponder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PairResponder").finish_non_exhaustive()
    }
}

impl PairResponder {
    /// Read XX msg 1 (`xx1`): the state and XX msg 2 (`xx2`).
    ///
    /// # Errors
    /// [`HandshakeError`].
    pub fn start(
        rng: &mut dyn Rng,
        local: &StaticKeypair,
        xx1: &[u8],
    ) -> Result<(Self, Vec<u8>), HandshakeError> {
        let mut hs = noise::builder(PATTERN, rng)?
            .local_private_key(local.private())
            .and_then(|b| b.build_responder())
            .map_err(|_| HandshakeError::Internal)?;
        if !noise::read(&mut hs, xx1)?.is_empty() {
            return Err(HandshakeError::UnexpectedPayload);
        }
        let xx2 = noise::write(&mut hs, &[], XX2_LEN)?;
        Ok((Self { hs }, xx2))
    }

    /// Read XX msg 3 (`xx3`): the link.
    ///
    /// # Errors
    /// [`HandshakeError`].
    pub fn finish(mut self, xx3: &[u8]) -> Result<PairLink, HandshakeError> {
        if !noise::read(&mut self.hs, xx3)?.is_empty() {
            return Err(HandshakeError::UnexpectedPayload);
        }
        PairLink::new(&mut self.hs, false)
    }
}

/// rlmeshd's commit-reveal: it commits to `Nr` before it sees `Ni`.
pub struct DaemonCommit {
    h: [u8; 32],
    nr: [u8; 32],
    c: [u8; 32],
    ni: Option<[u8; 32]>,
}

impl core::fmt::Debug for DaemonCommit {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DaemonCommit")
            .field("c", &self.c)
            .finish_non_exhaustive()
    }
}

impl Drop for DaemonCommit {
    fn drop(&mut self) {
        self.nr.zeroize();
    }
}

impl DaemonCommit {
    /// Draw `Nr` for the link's `h`.
    ///
    /// # Errors
    /// [`RngError`].
    pub fn new(rng: &mut dyn Rng, h: &[u8; 32]) -> Result<Self, RngError> {
        let mut nr = [0u8; 32];
        rng.fill_bytes(&mut nr)?;
        let c = commitment(h, &nr);
        Ok(Self {
            h: *h,
            nr,
            c,
            ni: None,
        })
    }

    /// `C`, for `PairCommit`.
    pub fn commitment(&self) -> &[u8; 32] {
        &self.c
    }

    /// Take the laptop's `Ni` (`PairNonce`): `Nr` to reveal (`PairReveal`)
    /// and the SAS. The same `Ni` again gives the same answer (a retry).
    ///
    /// # Errors
    /// [`PairingError::NonceChanged`] for a different `Ni`: the exchange aborts.
    pub fn reveal(&mut self, ni: &[u8; 32]) -> Result<([u8; 32], Sas), PairingError> {
        match self.ni {
            Some(seen) if !bool::from(seen.ct_eq(ni)) => return Err(PairingError::NonceChanged),
            Some(_) => {}
            None => self.ni = Some(*ni),
        }
        Ok((self.nr, sas(&self.h, ni, &self.nr)))
    }
}

/// The laptop's commit-reveal: it draws `Ni` only after it has `C`.
#[derive(Debug)]
pub struct LaptopCommit {
    h: [u8; 32],
    c: [u8; 32],
    ni: [u8; 32],
}

impl LaptopCommit {
    /// Take rlmeshd's `C` and draw `Ni`.
    ///
    /// # Errors
    /// [`RngError`].
    pub fn new(rng: &mut dyn Rng, h: &[u8; 32], c: &[u8; 32]) -> Result<Self, RngError> {
        let mut ni = [0u8; 32];
        rng.fill_bytes(&mut ni)?;
        Ok(Self { h: *h, c: *c, ni })
    }

    /// `Ni`, for `PairNonce`.
    pub fn nonce(&self) -> &[u8; 32] {
        &self.ni
    }

    /// Check the revealed `Nr` against `C` (in constant time): the SAS to show.
    ///
    /// # Errors
    /// [`PairingError::RevealMismatch`].
    pub fn check_reveal(&self, nr: &[u8; 32]) -> Result<Sas, PairingError> {
        if !bool::from(commitment(&self.h, nr).ct_eq(&self.c)) {
            return Err(PairingError::RevealMismatch);
        }
        Ok(sas(&self.h, &self.ni, nr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::sealed::{Kind, OpenError};
    use crate::crypto::test_rng::TestRng;

    const PAIR: &str = "rlmeshd/pair";

    fn handshake() -> (PairLink, PairLink) {
        let laptop = StaticKeypair::from_private([5; 32]);
        let daemon = StaticKeypair::from_private([6; 32]);
        let (init, xx1) = PairInitiator::start(&mut TestRng::new(1), &laptop).unwrap();
        assert_eq!(xx1.len(), XX1_LEN);
        let (resp, xx2) = PairResponder::start(&mut TestRng::new(2), &daemon, &xx1).unwrap();
        assert_eq!(xx2.len(), XX2_LEN);
        let (l, xx3) = init.finish(&xx2).unwrap();
        assert_eq!(xx3.len(), XX3_LEN);
        let d = resp.finish(&xx3).unwrap();
        assert_eq!(l.remote_static(), daemon.public());
        assert_eq!(d.remote_static(), laptop.public());
        (l, d)
    }

    #[test]
    fn pairs_and_both_sides_show_the_same_code() {
        let (mut l, mut d) = handshake();
        assert_eq!(l.handshake_hash(), d.handshake_hash());
        let h = *d.handshake_hash();

        let mut daemon = DaemonCommit::new(&mut TestRng::new(3), &h).unwrap();
        let mut commit = d
            .sender
            .seal(Kind::PairCommit, true, PAIR, daemon.commitment())
            .unwrap();
        let c: [u8; 32] = l
            .receiver
            .open(PAIR, &mut commit)
            .unwrap()
            .body
            .try_into()
            .unwrap();

        let laptop = LaptopCommit::new(&mut TestRng::new(4), &h, &c).unwrap();
        let mut nonce = l
            .sender
            .seal(Kind::PairNonce, true, PAIR, laptop.nonce())
            .unwrap();
        let ni: [u8; 32] = d
            .receiver
            .open(PAIR, &mut nonce)
            .unwrap()
            .body
            .try_into()
            .unwrap();

        let (nr, daemon_sas) = daemon.reveal(&ni).unwrap();
        assert_eq!(
            daemon.reveal(&ni).unwrap().1,
            daemon_sas,
            "a retry gets the same answer"
        );
        let laptop_sas = laptop.check_reveal(&nr).unwrap();
        assert_eq!(laptop_sas, daemon_sas);
        assert!(laptop_sas.value() < 1_000_000);
        assert_eq!(laptop_sas.to_string().len(), 6);
    }

    #[test]
    fn a_reveal_that_does_not_match_the_commitment_fails() {
        let h = [9u8; 32];
        let daemon = DaemonCommit::new(&mut TestRng::new(3), &h).unwrap();
        let laptop = LaptopCommit::new(&mut TestRng::new(4), &h, daemon.commitment()).unwrap();
        let mut nr = daemon.nr;
        nr[31] ^= 1;
        assert_eq!(laptop.check_reveal(&nr), Err(PairingError::RevealMismatch));
        // The right Nr under another transcript fails too.
        let other =
            LaptopCommit::new(&mut TestRng::new(4), &[8u8; 32], daemon.commitment()).unwrap();
        assert_eq!(
            other.check_reveal(&daemon.nr),
            Err(PairingError::RevealMismatch)
        );
        assert!(laptop.check_reveal(&daemon.nr).is_ok());
    }

    #[test]
    fn a_changed_nonce_aborts() {
        let mut daemon = DaemonCommit::new(&mut TestRng::new(3), &[9u8; 32]).unwrap();
        daemon.reveal(&[1; 32]).unwrap();
        assert_eq!(daemon.reveal(&[2; 32]), Err(PairingError::NonceChanged));
    }

    #[test]
    fn the_sas_is_the_documented_reduction() {
        assert_eq!(Sas(42).to_string(), "000042");
        let (h, ni, nr) = ([1u8; 32], [2u8; 32], [3u8; 32]);
        let digest = hash_labelled(SAS_LABEL, &[&h, &ni, &nr]).unwrap();
        let first = u64::from_le_bytes(digest[..8].try_into().unwrap());
        assert_eq!(u64::from(sas(&h, &ni, &nr).value()), first % 1_000_000);
    }

    #[test]
    fn the_link_carries_pairing_kinds_only_one_way_per_key() {
        let (mut l, mut d) = handshake();
        assert!(l.sender.seal(Kind::CtlRequest, true, PAIR, b"x").is_err());
        assert!(
            l.sender.seal(Kind::PairWait, false, PAIR, b"x").is_err(),
            "always encrypted"
        );
        let mut wait = l.sender.seal(Kind::PairWait, true, PAIR, b"").unwrap();
        // The laptop can't open its own message: it's under key 1, not 2.
        assert_eq!(
            l.receiver.open(PAIR, &mut wait.clone()),
            Err(OpenError::WrongKey(LAPTOP_KEY_ID))
        );
        assert!(d.receiver.open(PAIR, &mut wait).is_ok());
    }

    #[test]
    fn tampered_xx_messages_fail() {
        let laptop = StaticKeypair::from_private([5; 32]);
        let daemon = StaticKeypair::from_private([6; 32]);
        let (init, xx1) = PairInitiator::start(&mut TestRng::new(1), &laptop).unwrap();
        let (resp, mut xx2) = PairResponder::start(&mut TestRng::new(2), &daemon, &xx1).unwrap();
        xx2[50] ^= 1;
        assert_eq!(init.finish(&xx2).unwrap_err(), HandshakeError::Rejected);
        let (init, _) = PairInitiator::start(&mut TestRng::new(1), &laptop).unwrap();
        xx2[50] ^= 1;
        let (_, mut xx3) = init.finish(&xx2).unwrap();
        xx3[10] ^= 1;
        assert_eq!(resp.finish(&xx3).unwrap_err(), HandshakeError::Rejected);
    }
}
