//! Machine pairing (§3): a Noise XX handshake between the laptop
//! (initiator) and rlmeshd (responder), then commit-reveal inside it, as in
//! Bluetooth numeric comparison:
//!
//! 1. rlmeshd draws `Nr` and sends `C = BLAKE2s(lp("rlmesh/v2 pair commit") ‖ h ‖ Nr)`.
//! 2. The laptop draws `Ni` and sends it.
//! 3. rlmeshd reveals `Nr`; the laptop checks it against `C`.
//! 4. Both show `SAS = u64_le(BLAKE2s(lp("rlmesh/v2 pair sas") ‖ h ‖ Ni ‖ Nr)[..8]) mod 10^6`.
//!
//! XX is `Noise_XX_25519_ChaChaPoly_BLAKE2s` with prologue
//! `lp("rlmesh/v2 pair") ‖ exchange` and empty payloads, so `h` (the
//! handshake hash after XX) binds the exchange id. After XX every message
//! is a `Sealed` (kinds 16–19) under the XX raw split: key 1 laptop →
//! rlmeshd, key 2 rlmeshd → laptop.
//!
//! A message that fails its Noise check leaves a side's pending state as it
//! was (§3): each side keeps its inputs and replays its part of the
//! handshake from them, so a forged `xx2` or `xx3` can be followed by the
//! real one.

use subtle::ConstantTimeEq;
use zeroize::Zeroize;

use super::handshake::HandshakeError;
use super::lp::{hash_labelled, push_lp};
use super::noise::{self, Ephemeral};
use super::sealed::{KindSet, SealReceiver, SealSender, Security};
use super::{Rng, RngError, StaticKeypair};
use crate::prelude::*;

/// The Noise protocol name.
pub const PATTERN: &str = "Noise_XX_25519_ChaChaPoly_BLAKE2s";
/// The label the prologue starts with.
pub const PROLOGUE_LABEL: &[u8] = b"rlmesh/v2 pair";
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

/// The XX prologue: `lp("rlmesh/v2 pair") ‖ exchange`.
pub fn prologue(exchange: &[u8; 16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(2 + PROLOGUE_LABEL.len() + exchange.len());
    push_lp(&mut out, PROLOGUE_LABEL).unwrap_or_else(|_| unreachable!("a short label"));
    out.extend_from_slice(exchange);
    out
}

/// Why pairing failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PairingError {
    /// The XX handshake failed.
    #[error(transparent)]
    Handshake(#[from] HandshakeError),
    /// The revealed `Nr` doesn't match the commitment `C`.
    #[error("the revealed nonce doesn't match the commitment")]
    RevealMismatch,
    /// A second `PairNonce` with a different `Ni` for the same commitment:
    /// the exchange aborts.
    #[error("a different nonce for the same commitment")]
    NonceChanged,
    /// The exchange was aborted earlier (after [`NonceChanged`](Self::NonceChanged)).
    #[error("the exchange was aborted")]
    Aborted,
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
/// and the sealed channel in both directions (pairing kinds only, always
/// encrypted).
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
        let (up, down) = noise::split(hs);
        let (sender, receiver) = if laptop {
            (
                SealSender::new(LAPTOP_KEY_ID, up, KindSet::PAIRING, Security::Seal),
                SealReceiver::new(DAEMON_KEY_ID, down, KindSet::PAIRING, Security::Seal),
            )
        } else {
            (
                SealSender::new(DAEMON_KEY_ID, down, KindSet::PAIRING, Security::Seal),
                SealReceiver::new(LAPTOP_KEY_ID, up, KindSet::PAIRING, Security::Seal),
            )
        };
        Ok(Self {
            handshake_hash,
            remote_static,
            sender,
            receiver,
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

/// What a side keeps to replay its part of XX.
struct Inputs {
    local: StaticKeypair,
    prologue: Vec<u8>,
    ephemeral: Ephemeral,
}

impl Inputs {
    fn new(
        rng: &mut dyn Rng,
        local: &StaticKeypair,
        exchange: &[u8; 16],
    ) -> Result<Box<Self>, HandshakeError> {
        Ok(Box::new(Self {
            local: local.clone(),
            prologue: prologue(exchange),
            ephemeral: Ephemeral::draw(rng)?,
        }))
    }

    fn build(&self, initiator: bool) -> Result<snow::HandshakeState, HandshakeError> {
        let builder = noise::builder(PATTERN, &self.ephemeral)?
            .local_private_key(self.local.private())
            .and_then(|b| b.prologue(&self.prologue));
        let built = if initiator {
            builder.and_then(|b| b.build_initiator())
        } else {
            builder.and_then(|b| b.build_responder())
        };
        built.map_err(|_| HandshakeError::Internal)
    }
}

/// The laptop, after sending XX msg 1.
pub struct PairInitiator {
    /// Boxed: the secrets stay put while the state moves.
    inputs: Box<Inputs>,
}

impl core::fmt::Debug for PairInitiator {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PairInitiator").finish_non_exhaustive()
    }
}

impl PairInitiator {
    /// Start the exchange `exchange` (16 random bytes the laptop drew): the
    /// state and XX msg 1 (`xx1`).
    ///
    /// # Errors
    /// [`HandshakeError`].
    pub fn start(
        rng: &mut dyn Rng,
        local: &StaticKeypair,
        exchange: &[u8; 16],
    ) -> Result<(Self, Vec<u8>), HandshakeError> {
        let inputs = Inputs::new(rng, local, exchange)?;
        let xx1 = noise::write(&mut inputs.build(true)?, &[], XX1_LEN)?;
        Ok((Self { inputs }, xx1))
    }

    /// Read XX msg 2 (`xx2`): the link and XX msg 3 (`xx3`) to send.
    ///
    /// # Errors
    /// The unchanged state and the [`HandshakeError`]: a forged `xx2` can be
    /// followed by the real one.
    pub fn finish(self, xx2: &[u8]) -> Result<(PairLink, Vec<u8>), (Self, HandshakeError)> {
        match self.try_finish(xx2) {
            Ok(done) => Ok(done),
            Err(e) => Err((self, e)),
        }
    }

    fn try_finish(&self, xx2: &[u8]) -> Result<(PairLink, Vec<u8>), HandshakeError> {
        let mut hs = self.inputs.build(true)?;
        noise::write(&mut hs, &[], XX1_LEN)?;
        noise::read_empty(&mut hs, xx2)?;
        let xx3 = noise::write(&mut hs, &[], XX3_LEN)?;
        Ok((PairLink::new(&mut hs, true)?, xx3))
    }
}

/// rlmeshd, after replying with XX msg 2.
pub struct PairResponder {
    /// Boxed: the secrets stay put while the state moves.
    inputs: Box<Inputs>,
    xx1: Vec<u8>,
}

impl core::fmt::Debug for PairResponder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PairResponder").finish_non_exhaustive()
    }
}

impl PairResponder {
    /// Read XX msg 1 (`xx1`) of the exchange `exchange`: the state and XX
    /// msg 2 (`xx2`).
    ///
    /// # Errors
    /// [`HandshakeError`].
    pub fn start(
        rng: &mut dyn Rng,
        local: &StaticKeypair,
        exchange: &[u8; 16],
        xx1: &[u8],
    ) -> Result<(Self, Vec<u8>), HandshakeError> {
        let me = Self {
            inputs: Inputs::new(rng, local, exchange)?,
            xx1: xx1.to_vec(),
        };
        let mut hs = me.inputs.build(false)?;
        noise::read_empty(&mut hs, xx1)?;
        let xx2 = noise::write(&mut hs, &[], XX2_LEN)?;
        Ok((me, xx2))
    }

    /// Read XX msg 3 (`xx3`): the link.
    ///
    /// # Errors
    /// The unchanged state and the [`HandshakeError`]: a packet that fails
    /// its Noise check leaves the pending exchange untouched (§3).
    pub fn finish(self, xx3: &[u8]) -> Result<PairLink, (Self, HandshakeError)> {
        match self.try_finish(xx3) {
            Ok(link) => Ok(link),
            Err(e) => Err((self, e)),
        }
    }

    fn try_finish(&self, xx3: &[u8]) -> Result<PairLink, HandshakeError> {
        let mut hs = self.inputs.build(false)?;
        noise::read_empty(&mut hs, &self.xx1)?;
        noise::write(&mut hs, &[], XX2_LEN)?;
        noise::read_empty(&mut hs, xx3)?;
        PairLink::new(&mut hs, false)
    }
}

/// rlmeshd's commit-reveal: it commits to `Nr` before it sees `Ni`. A
/// changed `Ni` aborts it for good.
pub struct DaemonCommit {
    h: [u8; 32],
    nr: [u8; 32],
    c: [u8; 32],
    ni: Option<[u8; 32]>,
    aborted: bool,
}

impl core::fmt::Debug for DaemonCommit {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DaemonCommit")
            .field("c", &self.c)
            .field("aborted", &self.aborted)
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
            aborted: false,
        })
    }

    /// `C`, for `PairCommit`.
    ///
    /// # Errors
    /// [`PairingError::Aborted`] once aborted.
    pub fn commitment(&self) -> Result<&[u8; 32], PairingError> {
        if self.aborted {
            return Err(PairingError::Aborted);
        }
        Ok(&self.c)
    }

    /// Take the laptop's `Ni` (`PairNonce`): `Nr` to reveal (`PairReveal`)
    /// and the SAS. The same `Ni` again gives the same answer (a retry).
    ///
    /// # Errors
    /// [`PairingError::NonceChanged`] for a different `Ni`, which aborts the
    /// exchange; [`PairingError::Aborted`] on every call after that.
    pub fn reveal(&mut self, ni: &[u8; 32]) -> Result<([u8; 32], Sas), PairingError> {
        if self.aborted {
            return Err(PairingError::Aborted);
        }
        match self.ni {
            Some(seen) if !bool::from(seen.ct_eq(ni)) => {
                self.aborted = true;
                self.nr.zeroize();
                return Err(PairingError::NonceChanged);
            }
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
    use crate::crypto::noise::low_order;
    use crate::crypto::sealed::{Kind, OpenError};
    use crate::crypto::test_rng::TestRng;

    const PAIR: &str = "rlmeshd/pair";
    const EXCHANGE: [u8; 16] = [0x5a; 16];

    fn statics() -> (StaticKeypair, StaticKeypair) {
        (
            StaticKeypair::from_private([5; 32]),
            StaticKeypair::from_private([6; 32]),
        )
    }

    fn handshake() -> (PairLink, PairLink) {
        let (laptop, daemon) = statics();
        let (init, xx1) = PairInitiator::start(&mut TestRng::new(1), &laptop, &EXCHANGE).unwrap();
        assert_eq!(xx1.len(), XX1_LEN);
        let (resp, xx2) =
            PairResponder::start(&mut TestRng::new(2), &daemon, &EXCHANGE, &xx1).unwrap();
        assert_eq!(xx2.len(), XX2_LEN);
        let (l, xx3) = init.finish(&xx2).unwrap();
        assert_eq!(xx3.len(), XX3_LEN);
        let d = resp.finish(&xx3).unwrap();
        assert_eq!(l.remote_static(), daemon.public());
        assert_eq!(d.remote_static(), laptop.public());
        (l, d)
    }

    #[test]
    fn the_prologue_binds_the_exchange() {
        let mut want = vec![14, 0];
        want.extend_from_slice(b"rlmesh/v2 pair");
        want.extend_from_slice(&EXCHANGE);
        assert_eq!(prologue(&EXCHANGE), want);

        let (laptop, daemon) = statics();
        let (init, xx1) = PairInitiator::start(&mut TestRng::new(1), &laptop, &EXCHANGE).unwrap();
        // XX msg 1 is unauthenticated, so another exchange id fails at msg 2.
        let (_, xx2) = PairResponder::start(&mut TestRng::new(2), &daemon, &[0; 16], &xx1).unwrap();
        assert_eq!(init.finish(&xx2).unwrap_err().1, HandshakeError::Rejected);
    }

    #[test]
    fn pairs_and_both_sides_show_the_same_code() {
        let (mut l, mut d) = handshake();
        assert_eq!(l.handshake_hash(), d.handshake_hash());
        let h = *d.handshake_hash();

        let mut daemon = DaemonCommit::new(&mut TestRng::new(3), &h).unwrap();
        let mut commit = d
            .sender
            .seal(Kind::PairCommit, PAIR, daemon.commitment().unwrap())
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
            .seal(Kind::PairNonce, PAIR, laptop.nonce())
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
        let c = *daemon.commitment().unwrap();
        let laptop = LaptopCommit::new(&mut TestRng::new(4), &h, &c).unwrap();
        let mut nr = daemon.nr;
        nr[31] ^= 1;
        assert_eq!(laptop.check_reveal(&nr), Err(PairingError::RevealMismatch));
        // The right Nr under another transcript fails too.
        let other = LaptopCommit::new(&mut TestRng::new(4), &[8u8; 32], &c).unwrap();
        assert_eq!(
            other.check_reveal(&daemon.nr),
            Err(PairingError::RevealMismatch)
        );
        assert!(laptop.check_reveal(&daemon.nr).is_ok());
    }

    #[test]
    fn a_changed_nonce_aborts_for_good() {
        let mut daemon = DaemonCommit::new(&mut TestRng::new(3), &[9u8; 32]).unwrap();
        daemon.reveal(&[1; 32]).unwrap();
        assert_eq!(daemon.reveal(&[2; 32]), Err(PairingError::NonceChanged));
        assert_eq!(
            daemon.reveal(&[1; 32]),
            Err(PairingError::Aborted),
            "even the first Ni"
        );
        assert_eq!(daemon.commitment(), Err(PairingError::Aborted));
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
        assert!(l.sender.seal(Kind::CtlRequest, PAIR, b"x").is_err());
        let mut wait = l.sender.seal(Kind::PairWait, PAIR, b"").unwrap();
        // The laptop can't open its own message: it's under key 1, not 2.
        assert_eq!(
            l.receiver.open(PAIR, &mut wait.clone()),
            Err(OpenError::WrongKey(LAPTOP_KEY_ID))
        );
        assert!(d.receiver.open(PAIR, &mut wait).is_ok());
    }

    #[test]
    fn a_forged_xx3_leaves_the_daemon_ready_for_the_real_one() {
        let (laptop, daemon) = statics();
        let (init, xx1) = PairInitiator::start(&mut TestRng::new(1), &laptop, &EXCHANGE).unwrap();
        let (resp, xx2) =
            PairResponder::start(&mut TestRng::new(2), &daemon, &EXCHANGE, &xx1).unwrap();
        let (l, xx3) = init.finish(&xx2).unwrap();
        let mut garbage = xx3.clone();
        garbage[10] ^= 1;
        let (resp, err) = resp.finish(&garbage).unwrap_err();
        assert_eq!(err, HandshakeError::Rejected);
        let (resp, err) = resp.finish(&[0; 3]).unwrap_err();
        assert_eq!(err, HandshakeError::Malformed);
        let d = resp.finish(&xx3).unwrap();
        assert_eq!(d.handshake_hash(), l.handshake_hash());
    }

    #[test]
    fn a_forged_xx2_leaves_the_laptop_ready_for_the_real_one() {
        let (laptop, daemon) = statics();
        let (init, xx1) = PairInitiator::start(&mut TestRng::new(1), &laptop, &EXCHANGE).unwrap();
        let (resp, xx2) =
            PairResponder::start(&mut TestRng::new(2), &daemon, &EXCHANGE, &xx1).unwrap();
        let mut garbage = xx2.clone();
        garbage[50] ^= 1;
        let (init, err) = init.finish(&garbage).unwrap_err();
        assert_eq!(err, HandshakeError::Rejected);
        let (l, xx3) = init.finish(&xx2).unwrap();
        assert_eq!(
            resp.finish(&xx3).unwrap().handshake_hash(),
            l.handshake_hash()
        );
    }

    #[test]
    fn a_low_order_daemon_static_is_refused_by_the_laptop() {
        let (laptop, _) = statics();
        let (init, xx1) = PairInitiator::start(&mut TestRng::new(1), &laptop, &EXCHANGE).unwrap();
        let p = prologue(&EXCHANGE);
        let mut attacker = low_order::builder(PATTERN, [7; 32])
            .local_private_key(&low_order::MARKER)
            .and_then(|b| b.prologue(&p))
            .and_then(|b| b.build_responder())
            .unwrap();
        noise::read_empty(&mut attacker, &xx1).unwrap();
        let xx2 = noise::write(&mut attacker, &[], XX2_LEN).unwrap();
        assert_eq!(init.finish(&xx2).unwrap_err().1, HandshakeError::LowOrder);
    }

    #[test]
    fn a_low_order_laptop_static_is_refused_by_the_daemon() {
        let (_, daemon) = statics();
        let p = prologue(&EXCHANGE);
        let mut attacker = low_order::builder(PATTERN, [7; 32])
            .local_private_key(&low_order::MARKER)
            .and_then(|b| b.prologue(&p))
            .and_then(|b| b.build_initiator())
            .unwrap();
        let xx1 = noise::write(&mut attacker, &[], XX1_LEN).unwrap();
        let (resp, xx2) =
            PairResponder::start(&mut TestRng::new(2), &daemon, &EXCHANGE, &xx1).unwrap();
        noise::read_empty(&mut attacker, &xx2).unwrap();
        let xx3 = noise::write(&mut attacker, &[], XX3_LEN).unwrap();
        assert_eq!(resp.finish(&xx3).unwrap_err().1, HandshakeError::LowOrder);
    }
}
