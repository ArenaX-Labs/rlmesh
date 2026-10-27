//! The host's `Signed` broadcasts (§8.7): authority state and virtual ticks,
//! in clear, signed with the host's Ed25519 key.
//!
//! | Offset | Size | Field |
//! |---|---|---|
//! | 0 | 1 | `version` = 1 |
//! | 1 | 1 | `kind`: 20 authority state, 21 virtual tick |
//! | 2 | 8 | `host_instance` (u64 LE) |
//! | 10 | 8 | `seq` (u64 LE): one counter per host instance, across both kinds |
//! | 18 | n | `body` |
//! | 18 + n | 64 | Ed25519 signature over `lp("rlmesh/v2 signed") ‖ header ‖ lp(key expression) ‖ body` |

use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};

use super::lp::{LpTooLong, push_lp};
use crate::prelude::*;

/// The envelope version this build writes and reads.
pub const VERSION: u8 = 1;
/// Header length.
pub const HEADER_LEN: usize = 18;
/// Signature length.
pub const SIGNATURE_LEN: usize = 64;
/// The label the signature's input starts with.
pub const LABEL: &[u8] = b"rlmesh/v2 signed";

/// What a `Signed` carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SignedKind {
    /// `AuthorityState` on `host/state`.
    AuthorityState = 20,
    /// `VirtualTick` on `host/tick`.
    VirtualTick = 21,
}

impl SignedKind {
    /// The kind with this byte, if any.
    pub fn from_u8(byte: u8) -> Option<Self> {
        match byte {
            20 => Some(Self::AuthorityState),
            21 => Some(Self::VirtualTick),
            _ => None,
        }
    }

    /// The wire byte.
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    fn index(self) -> usize {
        usize::from(self.as_u8() - 20)
    }
}

/// A parsed `Signed` header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SignedHeader {
    /// The kind.
    pub kind: SignedKind,
    /// The host instance that signed it.
    pub host_instance: u64,
    /// The host's publication counter.
    pub seq: u64,
}

impl SignedHeader {
    /// The header's bytes.
    pub fn to_bytes(&self) -> [u8; HEADER_LEN] {
        let mut out = [0u8; HEADER_LEN];
        out[0] = VERSION;
        out[1] = self.kind.as_u8();
        out[2..10].copy_from_slice(&self.host_instance.to_le_bytes());
        out[10..18].copy_from_slice(&self.seq.to_le_bytes());
        out
    }
}

/// The bytes the signature covers.
pub fn signing_input(
    header: &[u8; HEADER_LEN],
    keyexpr: &str,
    body: &[u8],
) -> Result<Vec<u8>, LpTooLong> {
    let mut out = Vec::with_capacity(2 + LABEL.len() + HEADER_LEN + 2 + keyexpr.len() + body.len());
    push_lp(&mut out, LABEL)?;
    out.extend_from_slice(header);
    push_lp(&mut out, keyexpr.as_bytes())?;
    out.extend_from_slice(body);
    Ok(out)
}

/// Why a broadcast couldn't be signed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SignError {
    /// The key expression is over 65,535 bytes.
    #[error("the key expression is over 65535 bytes")]
    KeyExprTooLong,
    /// Every `seq` of this host instance is used.
    #[error("the host's publication counter is exhausted")]
    SeqExhausted,
}

/// Why a broadcast was dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SignedError {
    /// Shorter than a header and a signature.
    #[error("truncated")]
    Truncated,
    /// A version other than 1.
    #[error("unknown version {0}")]
    Version(u8),
    /// The kind byte is not 20 or 21.
    #[error("unknown kind {0}")]
    UnknownKind(u8),
    /// Signed by another host instance than the handshake authenticated.
    #[error("from host instance {0}, not the authenticated one")]
    WrongInstance(u64),
    /// Not above the last `seq` accepted for this kind.
    #[error("seq {0} is not above the last accepted")]
    Stale(u64),
    /// The key expression is over 65,535 bytes.
    #[error("the key expression is over 65535 bytes")]
    KeyExprTooLong,
    /// The signature doesn't verify.
    #[error("bad signature")]
    BadSignature,
    /// The host signing key is not a valid Ed25519 public key.
    #[error("invalid host signing key")]
    BadKey,
}

/// The host's side: its signing key and its one publication counter.
pub struct SignedSender {
    key: SigningKey,
    host_instance: u64,
    /// The next `seq`; 0 once `u64::MAX` is used.
    next: u64,
}

impl core::fmt::Debug for SignedSender {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SignedSender")
            .field("host_instance", &self.host_instance)
            .finish_non_exhaustive()
    }
}

impl SignedSender {
    /// A sender for `host_instance`, signing with the Ed25519 secret `secret`.
    pub fn new(secret: &[u8; 32], host_instance: u64) -> Self {
        Self {
            key: SigningKey::from_bytes(secret),
            host_instance,
            next: 1,
        }
    }

    /// The public key receivers verify with (`HandshakeAccept.host_sign_key`).
    pub fn verifying_key(&self) -> [u8; 32] {
        self.key.verifying_key().to_bytes()
    }

    /// Sign `body` as the next publication, of `kind`, on `keyexpr`.
    ///
    /// # Errors
    /// [`SignError`]; the counter is unchanged then.
    pub fn sign(
        &mut self,
        kind: SignedKind,
        keyexpr: &str,
        body: &[u8],
    ) -> Result<Vec<u8>, SignError> {
        if self.next == 0 {
            return Err(SignError::SeqExhausted);
        }
        let header = SignedHeader {
            kind,
            host_instance: self.host_instance,
            seq: self.next,
        }
        .to_bytes();
        let input = signing_input(&header, keyexpr, body).map_err(|_| SignError::KeyExprTooLong)?;
        let signature = self.key.sign(&input);
        self.next = self.next.wrapping_add(1);
        let mut out = Vec::with_capacity(HEADER_LEN + body.len() + SIGNATURE_LEN);
        out.extend_from_slice(&header);
        out.extend_from_slice(body);
        out.extend_from_slice(&signature.to_bytes());
        Ok(out)
    }

    #[cfg(test)]
    pub(crate) fn set_next(&mut self, seq: u64) {
        self.next = seq;
    }
}

/// A verified broadcast.
#[derive(Debug, PartialEq, Eq)]
pub struct Verified<'m> {
    /// The header.
    pub header: SignedHeader,
    /// The body, in clear.
    pub body: &'m [u8],
}

/// A receiver's side: the host's key and instance from the handshake
/// (`HandshakeAccept`), and the last `seq` accepted per kind. The first
/// `seq` of a kind has no floor.
#[derive(Debug, Clone)]
pub struct SignedReceiver {
    key: VerifyingKey,
    host_instance: u64,
    last: [Option<u64>; 2],
}

impl SignedReceiver {
    /// A receiver of `host_instance`'s broadcasts, signed by `host_sign_key`.
    ///
    /// # Errors
    /// [`SignedError::BadKey`] if the key is not a valid Ed25519 point.
    pub fn new(host_sign_key: &[u8; 32], host_instance: u64) -> Result<Self, SignedError> {
        let key = VerifyingKey::from_bytes(host_sign_key).map_err(|_| SignedError::BadKey)?;
        Ok(Self {
            key,
            host_instance,
            last: [None; 2],
        })
    }

    /// The last `seq` accepted for `kind`.
    pub fn last(&self, kind: SignedKind) -> Option<u64> {
        self.last[kind.index()]
    }

    /// Verify `msg`, received on `keyexpr`, and record its `seq`. Uses
    /// `verify_strict` (no small-order keys, canonical signatures).
    ///
    /// # Errors
    /// [`SignedError`]; nothing is recorded then.
    pub fn verify<'m>(
        &mut self,
        keyexpr: &str,
        msg: &'m [u8],
    ) -> Result<Verified<'m>, SignedError> {
        if msg.len() < HEADER_LEN + SIGNATURE_LEN {
            return Err(SignedError::Truncated);
        }
        if msg[0] != VERSION {
            return Err(SignedError::Version(msg[0]));
        }
        let kind = SignedKind::from_u8(msg[1]).ok_or(SignedError::UnknownKind(msg[1]))?;
        let mut word = [0u8; 8];
        word.copy_from_slice(&msg[2..10]);
        let host_instance = u64::from_le_bytes(word);
        word.copy_from_slice(&msg[10..18]);
        let seq = u64::from_le_bytes(word);
        if host_instance != self.host_instance {
            return Err(SignedError::WrongInstance(host_instance));
        }
        if self.last[kind.index()].is_some_and(|last| seq <= last) {
            return Err(SignedError::Stale(seq));
        }
        let (signed, signature) = msg.split_at(msg.len() - SIGNATURE_LEN);
        let mut header = [0u8; HEADER_LEN];
        header.copy_from_slice(&signed[..HEADER_LEN]);
        let body = &signed[HEADER_LEN..];
        let input =
            signing_input(&header, keyexpr, body).map_err(|_| SignedError::KeyExprTooLong)?;
        let mut sig = [0u8; SIGNATURE_LEN];
        sig.copy_from_slice(signature);
        self.key
            .verify_strict(&input, &Signature::from_bytes(&sig))
            .map_err(|_| SignedError::BadSignature)?;
        self.last[kind.index()] = Some(seq);
        Ok(Verified {
            header: SignedHeader {
                kind,
                host_instance,
                seq,
            },
            body,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STATE: &str = "rlmesh/v2/acme/s/demo/host/state";
    const TICK: &str = "rlmesh/v2/acme/s/demo/host/tick";

    fn host() -> SignedSender {
        SignedSender::new(&[3; 32], 77)
    }

    #[test]
    fn verifies_and_counts_across_both_kinds() {
        let mut tx = host();
        let mut rx = SignedReceiver::new(&tx.verifying_key(), 77).unwrap();
        let a = tx
            .sign(SignedKind::AuthorityState, STATE, b"state")
            .unwrap();
        let b = tx.sign(SignedKind::VirtualTick, TICK, b"tick").unwrap();
        let v = rx.verify(STATE, &a).unwrap();
        assert_eq!(v.body, b"state");
        assert_eq!(v.header.seq, 1);
        assert_eq!(rx.verify(TICK, &b).unwrap().header.seq, 2);
        assert_eq!(rx.last(SignedKind::AuthorityState), Some(1));
    }

    #[test]
    fn rejects_another_host_instance() {
        let mut tx = host();
        let mut rx = SignedReceiver::new(&tx.verifying_key(), 76).unwrap();
        let msg = tx.sign(SignedKind::AuthorityState, STATE, b"s").unwrap();
        assert_eq!(rx.verify(STATE, &msg), Err(SignedError::WrongInstance(77)));
    }

    #[test]
    fn rejects_a_seq_not_above_the_last_for_its_kind() {
        let mut tx = host();
        let mut rx = SignedReceiver::new(&tx.verifying_key(), 77).unwrap();
        let s1 = tx.sign(SignedKind::AuthorityState, STATE, b"1").unwrap();
        let t2 = tx.sign(SignedKind::VirtualTick, TICK, b"2").unwrap();
        let s3 = tx.sign(SignedKind::AuthorityState, STATE, b"3").unwrap();
        // The first of a kind has no floor: s3 first, then s1 is older.
        assert!(rx.verify(STATE, &s3).is_ok());
        assert_eq!(rx.verify(STATE, &s1), Err(SignedError::Stale(1)));
        assert_eq!(
            rx.verify(STATE, &s3),
            Err(SignedError::Stale(3)),
            "a repeat"
        );
        // Floors are per kind: tick 2 is below state 3 but still new.
        assert!(rx.verify(TICK, &t2).is_ok());
    }

    #[test]
    fn rejects_a_bad_signature_or_changed_bytes() {
        let mut tx = host();
        let mut rx = SignedReceiver::new(&tx.verifying_key(), 77).unwrap();
        let msg = tx.sign(SignedKind::AuthorityState, STATE, b"body").unwrap();
        // seq, body and signature (version, kind and instance fail earlier).
        for at in 10..msg.len() {
            let mut bad = msg.clone();
            bad[at] ^= 1;
            assert_eq!(
                rx.verify(STATE, &bad),
                Err(SignedError::BadSignature),
                "byte {at}"
            );
        }
        assert_eq!(
            rx.verify(TICK, &msg),
            Err(SignedError::BadSignature),
            "another key expression"
        );
        let mut kind = msg.clone();
        kind[1] = SignedKind::VirtualTick.as_u8();
        assert_eq!(rx.verify(STATE, &kind), Err(SignedError::BadSignature));
        let mut other =
            SignedReceiver::new(&SignedSender::new(&[4; 32], 77).verifying_key(), 77).unwrap();
        assert_eq!(
            other.verify(STATE, &msg),
            Err(SignedError::BadSignature),
            "another key"
        );
        // None of that moved the floor.
        assert!(rx.verify(STATE, &msg).is_ok());
    }

    #[test]
    fn rejects_bad_headers() {
        let mut tx = host();
        let mut rx = SignedReceiver::new(&tx.verifying_key(), 77).unwrap();
        let msg = tx.sign(SignedKind::AuthorityState, STATE, b"").unwrap();
        let mut bad = msg.clone();
        bad[0] = 0;
        assert_eq!(rx.verify(STATE, &bad), Err(SignedError::Version(0)));
        for kind in [0, 19, 22] {
            let mut bad = msg.clone();
            bad[1] = kind;
            assert_eq!(rx.verify(STATE, &bad), Err(SignedError::UnknownKind(kind)));
        }
        assert_eq!(
            rx.verify(STATE, &msg[..msg.len() - 1]),
            Err(SignedError::Truncated)
        );
    }

    #[test]
    fn seq_exhaustion_stops_signing() {
        let mut tx = host();
        tx.set_next(u64::MAX);
        assert!(tx.sign(SignedKind::VirtualTick, TICK, b"").is_ok());
        assert_eq!(
            tx.sign(SignedKind::VirtualTick, TICK, b""),
            Err(SignError::SeqExhausted)
        );
    }

    #[test]
    fn rejects_a_key_expression_over_lp_max() {
        let mut tx = host();
        let long = "k".repeat(super::super::lp::LP_MAX + 1);
        assert_eq!(
            tx.sign(SignedKind::VirtualTick, &long, b""),
            Err(SignError::KeyExprTooLong)
        );
        let mut rx = SignedReceiver::new(&tx.verifying_key(), 77).unwrap();
        let msg = tx.sign(SignedKind::VirtualTick, TICK, b"").unwrap();
        assert_eq!(rx.verify(&long, &msg), Err(SignedError::KeyExprTooLong));
        assert_eq!(
            rx.verify(TICK, &msg).unwrap().header.seq,
            1,
            "the failed sign used no seq"
        );
    }
}
