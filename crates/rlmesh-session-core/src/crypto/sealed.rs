//! The `Sealed` envelope (§8.2): a 15-byte header, the body (encrypted or
//! not) and a Poly1305 tag, under ChaCha20-Poly1305.
//!
//! | Offset | Size | Field |
//! |---|---|---|
//! | 0 | 1 | `version` = 1 |
//! | 1 | 1 | `flags`: bit 0 = encrypted, every other bit 0 |
//! | 2 | 1 | `kind` ([`Kind`]) |
//! | 3 | 4 | `key_id` (u32 LE) |
//! | 7 | 8 | `seq` (u64 LE) |
//! | 15 | n | `body` |
//! | 15 + n | 16 | tag |
//!
//! Nonce = `kind` ‖ 3 zero bytes ‖ `seq` (u64 LE). Encrypted: AAD =
//! `header ‖ lp(key expression)`, plaintext = body. Integrity only: AAD =
//! `header ‖ lp(key expression) ‖ body`, plaintext empty.

use chacha20poly1305::aead::{AeadInPlace, KeyInit};
use chacha20poly1305::{ChaCha20Poly1305, Nonce, Tag};

use super::Key;
use super::lp::{LpTooLong, lp_prefix};
use super::replay::{ReplayWindow, Replayed};
use crate::prelude::*;

/// The envelope version this build writes and reads.
pub const VERSION: u8 = 1;
/// Header length.
pub const HEADER_LEN: usize = 15;
/// Tag length.
pub const TAG_LEN: usize = 16;
/// Bytes a `Sealed` adds to its body.
pub const OVERHEAD: usize = HEADER_LEN + TAG_LEN;
/// `flags` bit 0: the body is encrypted.
pub const FLAG_ENCRYPTED: u8 = 1;

/// What a `Sealed` carries (§8.2). Unknown kinds are rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Kind {
    /// A ctl request.
    CtlRequest = 1,
    /// A ctl reply.
    CtlReply = 2,
    /// A heartbeat.
    Heartbeat = 3,
    /// A heartbeat echo.
    HeartbeatEcho = 4,
    /// A participant report.
    ParticipantReport = 5,
    /// A key delivery.
    KeyDelivery = 6,
    /// An action chunk.
    ActionChunk = 7,
    /// A sample.
    Sample = 8,
    /// An `_applied` entry.
    Applied = 9,
    /// A history page.
    HistoryPage = 10,
    /// A lockstep request.
    LockstepRequest = 11,
    /// A lockstep reply.
    LockstepReply = 12,
    /// An e-stop signal.
    EstopSignal = 13,
    /// A host session stream entry.
    HostSessionStream = 14,
    /// A history query.
    HistoryQuery = 15,
    /// Pairing: `PairCommit` (§3).
    PairCommit = 16,
    /// Pairing: `PairNonce` (§3).
    PairNonce = 17,
    /// Pairing: `PairReveal` (§3).
    PairReveal = 18,
    /// Pairing: `PairWait` and its `PairResult` reply (§3).
    PairWait = 19,
}

/// Whether a kind's body is encrypted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protection {
    /// Always encrypted.
    Encrypted,
    /// Always integrity only.
    IntegrityOnly,
    /// Set by the part's, stream's or lease's `security`: the sender picks,
    /// the receiver checks it against the key's ([`Opened::encrypted`]).
    Configured,
}

impl Kind {
    /// The kind with this byte, if any.
    pub fn from_u8(byte: u8) -> Option<Self> {
        use Kind::*;
        Some(match byte {
            1 => CtlRequest,
            2 => CtlReply,
            3 => Heartbeat,
            4 => HeartbeatEcho,
            5 => ParticipantReport,
            6 => KeyDelivery,
            7 => ActionChunk,
            8 => Sample,
            9 => Applied,
            10 => HistoryPage,
            11 => LockstepRequest,
            12 => LockstepReply,
            13 => EstopSignal,
            14 => HostSessionStream,
            15 => HistoryQuery,
            16 => PairCommit,
            17 => PairNonce,
            18 => PairReveal,
            19 => PairWait,
            _ => return None,
        })
    }

    /// The wire byte.
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Whether this kind is always encrypted, never, or configured: kinds
    /// under `k_up`/`k_down` (1–6, 10, 15) and pairing (16–19) always are,
    /// e-stop signals (13) never, and 7, 8, 9, 11, 12 and 14 follow their
    /// `security`.
    pub fn protection(self) -> Protection {
        use Kind::*;
        match self {
            CtlRequest | CtlReply | Heartbeat | HeartbeatEcho | ParticipantReport | KeyDelivery
            | HistoryPage | HistoryQuery | PairCommit | PairNonce | PairReveal | PairWait => {
                Protection::Encrypted
            }
            EstopSignal => Protection::IntegrityOnly,
            ActionChunk | Sample | Applied | LockstepRequest | LockstepReply
            | HostSessionStream => Protection::Configured,
        }
    }

    fn index(self) -> usize {
        usize::from(self.as_u8()) - 1
    }
}

const KINDS: usize = 19;

/// The kinds a key may carry (its purpose and scope, §8.5). A message of
/// any other kind under that key is rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KindSet(u32);

impl KindSet {
    /// Every kind.
    pub const ALL: Self = Self(((1 << KINDS) - 1) << 1);
    /// The pairing kinds, 16–19.
    pub const PAIRING: Self = Self(0b1111 << 16);

    /// Exactly these kinds.
    pub fn of(kinds: &[Kind]) -> Self {
        Self(kinds.iter().fold(0, |set, k| set | 1 << k.as_u8()))
    }

    /// Whether `kind` is in the set.
    pub fn contains(self, kind: Kind) -> bool {
        self.0 & 1 << kind.as_u8() != 0
    }
}

/// A parsed `Sealed` header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    /// Whether the body is encrypted (flags bit 0).
    pub encrypted: bool,
    /// The kind.
    pub kind: Kind,
    /// The key the message is sealed under.
    pub key_id: u32,
    /// The sender's counter for (key, kind).
    pub seq: u64,
}

impl Header {
    /// The header's bytes.
    pub fn to_bytes(&self) -> [u8; HEADER_LEN] {
        let mut out = [0u8; HEADER_LEN];
        out[0] = VERSION;
        out[1] = if self.encrypted { FLAG_ENCRYPTED } else { 0 };
        out[2] = self.kind.as_u8();
        out[3..7].copy_from_slice(&self.key_id.to_le_bytes());
        out[7..15].copy_from_slice(&self.seq.to_le_bytes());
        out
    }

    /// Parse the header at the front of `msg`, so a receiver can find the
    /// key named by `key_id`. Nothing is authenticated yet.
    ///
    /// # Errors
    /// [`OpenError`] for a short message, a version other than 1, an unknown
    /// flag bit or kind, or a protection the kind forbids.
    pub fn parse(msg: &[u8]) -> Result<Self, OpenError> {
        if msg.len() < OVERHEAD {
            return Err(OpenError::Truncated);
        }
        if msg[0] != VERSION {
            return Err(OpenError::Version(msg[0]));
        }
        if msg[1] & !FLAG_ENCRYPTED != 0 {
            return Err(OpenError::Flags(msg[1]));
        }
        let kind = Kind::from_u8(msg[2]).ok_or(OpenError::UnknownKind(msg[2]))?;
        let encrypted = msg[1] & FLAG_ENCRYPTED != 0;
        if !protection_allows(kind, encrypted) {
            return Err(OpenError::Protection(kind));
        }
        let key_id = u32::from_le_bytes([msg[3], msg[4], msg[5], msg[6]]);
        let mut seq = [0u8; 8];
        seq.copy_from_slice(&msg[7..15]);
        Ok(Self {
            encrypted,
            kind,
            key_id,
            seq: u64::from_le_bytes(seq),
        })
    }
}

fn protection_allows(kind: Kind, encrypted: bool) -> bool {
    match kind.protection() {
        Protection::Encrypted => encrypted,
        Protection::IntegrityOnly => !encrypted,
        Protection::Configured => true,
    }
}

/// The nonce of (kind, seq): `kind` ‖ 3 zero bytes ‖ `seq` (u64 LE).
pub fn nonce(kind: Kind, seq: u64) -> [u8; 12] {
    let mut out = [0u8; 12];
    out[0] = kind.as_u8();
    out[4..].copy_from_slice(&seq.to_le_bytes());
    out
}

/// AADs up to this size are built on the stack; larger ones (a long key
/// expression, or a long integrity-only body) take one allocation.
const INLINE_AAD: usize = 256;

/// Build the AAD (`header ‖ lp(keyexpr) [‖ body]`) and hand it to `f`.
fn with_aad<R>(
    header: &[u8; HEADER_LEN],
    keyexpr: &str,
    body: Option<&[u8]>,
    f: impl FnOnce(&[u8]) -> R,
) -> Result<R, LpTooLong> {
    let prefix = lp_prefix(keyexpr.as_bytes())?;
    let body = body.unwrap_or(&[]);
    let len = HEADER_LEN + 2 + keyexpr.len() + body.len();
    let parts: [&[u8]; 4] = [header, &prefix, keyexpr.as_bytes(), body];
    if len <= INLINE_AAD {
        let mut buf = [0u8; INLINE_AAD];
        let mut at = 0;
        for part in parts {
            buf[at..at + part.len()].copy_from_slice(part);
            at += part.len();
        }
        Ok(f(&buf[..len]))
    } else {
        Ok(f(&parts.concat()))
    }
}

/// Why a message couldn't be sealed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SealError {
    /// The key expression is over 65,535 bytes.
    #[error("the key expression is over 65535 bytes")]
    KeyExprTooLong,
    /// The kind is not one this key carries.
    #[error("kind {0:?} is not carried by this key")]
    WrongKind(Kind),
    /// The kind forbids this protection (an always-encrypted kind sent in
    /// clear, or an encrypted e-stop signal).
    #[error("kind {0:?} forbids this protection")]
    Protection(Kind),
    /// Every `seq` of this (key, kind) is used: the key must be replaced.
    #[error("the seq counter of kind {0:?} is exhausted")]
    SeqExhausted(Kind),
    /// The body is too long for the AEAD.
    #[error("the body is too long to seal")]
    TooLong,
}

/// Why a message was dropped. A receiver never processes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum OpenError {
    /// Shorter than a header and a tag.
    #[error("truncated")]
    Truncated,
    /// A version other than 1.
    #[error("unknown version {0}")]
    Version(u8),
    /// A flag bit other than bit 0 is set.
    #[error("unknown flags {0:#04x}")]
    Flags(u8),
    /// The kind byte names no kind.
    #[error("unknown kind {0}")]
    UnknownKind(u8),
    /// The kind forbids this protection.
    #[error("kind {0:?} forbids this protection")]
    Protection(Kind),
    /// The kind is not one this key carries.
    #[error("kind {0:?} is not carried by this key")]
    WrongKind(Kind),
    /// Sealed under another key.
    #[error("sealed under key {0}, not this one")]
    WrongKey(u32),
    /// The key expression is over 65,535 bytes.
    #[error("the key expression is over 65535 bytes")]
    KeyExprTooLong,
    /// Seen before, or older than the window.
    #[error(transparent)]
    Replayed(#[from] Replayed),
    /// The tag doesn't verify: the wrong key or key expression, or a
    /// changed header, body or tag.
    #[error("authentication failed")]
    Auth,
}

/// The sending side of one key: a dense `seq` counter per kind, from 1.
/// One sender per key (§8.2); callers sharing it across tasks hold one lock
/// per (key, kind) across sealing and enqueuing.
pub struct SealSender {
    aead: ChaCha20Poly1305,
    key_id: u32,
    kinds: KindSet,
    /// The next `seq` per kind; 0 once `u64::MAX` is used.
    next: [u64; KINDS],
}

impl core::fmt::Debug for SealSender {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SealSender")
            .field("key_id", &self.key_id)
            .finish_non_exhaustive()
    }
}

impl SealSender {
    /// A sender under `key` (id `key_id`) for `kinds`.
    pub fn new(key_id: u32, key: &Key, kinds: KindSet) -> Self {
        Self {
            aead: ChaCha20Poly1305::new(key.as_bytes().into()),
            key_id,
            kinds,
            next: [1; KINDS],
        }
    }

    /// The key's id.
    pub fn key_id(&self) -> u32 {
        self.key_id
    }

    /// Seal `body` as the next message of `kind` on `keyexpr`.
    ///
    /// # Errors
    /// [`SealError`]; the counter is unchanged then.
    pub fn seal(
        &mut self,
        kind: Kind,
        encrypted: bool,
        keyexpr: &str,
        body: &[u8],
    ) -> Result<Vec<u8>, SealError> {
        let mut out = Vec::with_capacity(OVERHEAD + body.len());
        self.seal_to(kind, encrypted, keyexpr, body, &mut out)?;
        Ok(out)
    }

    /// [`seal`](Self::seal), appending to `out` (which a caller can reuse).
    /// `out` is unchanged on error.
    ///
    /// # Errors
    /// [`SealError`]; the counter is unchanged then.
    pub fn seal_to(
        &mut self,
        kind: Kind,
        encrypted: bool,
        keyexpr: &str,
        body: &[u8],
        out: &mut Vec<u8>,
    ) -> Result<(), SealError> {
        if !self.kinds.contains(kind) {
            return Err(SealError::WrongKind(kind));
        }
        if !protection_allows(kind, encrypted) {
            return Err(SealError::Protection(kind));
        }
        let seq = self.next[kind.index()];
        if seq == 0 {
            return Err(SealError::SeqExhausted(kind));
        }
        let header = Header {
            encrypted,
            kind,
            key_id: self.key_id,
            seq,
        }
        .to_bytes();
        let nonce = Nonce::from(nonce(kind, seq));
        let start = out.len();
        out.extend_from_slice(&header);
        out.extend_from_slice(body);
        let sealed = if encrypted {
            let buffer = &mut out[start + HEADER_LEN..];
            with_aad(&header, keyexpr, None, |aad| {
                self.aead.encrypt_in_place_detached(&nonce, aad, buffer)
            })
        } else {
            with_aad(&header, keyexpr, Some(body), |aad| {
                self.aead.encrypt_in_place_detached(&nonce, aad, &mut [])
            })
        };
        match sealed {
            Ok(Ok(tag)) => {
                out.extend_from_slice(&tag);
                self.next[kind.index()] = seq.wrapping_add(1);
                Ok(())
            }
            Ok(Err(_)) => {
                out.truncate(start);
                Err(SealError::TooLong)
            }
            Err(LpTooLong) => {
                out.truncate(start);
                Err(SealError::KeyExprTooLong)
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn set_next(&mut self, kind: Kind, seq: u64) {
        self.next[kind.index()] = seq;
    }
}

/// An opened message: its header and its body (decrypted in place).
#[derive(Debug, PartialEq, Eq)]
pub struct Opened<'m> {
    /// The header. `encrypted` tells the receiver the protection actually
    /// used, to check against the key's `security`.
    pub header: Header,
    /// The body.
    pub body: &'m [u8],
}

impl Opened<'_> {
    /// Whether the body was encrypted.
    pub fn encrypted(&self) -> bool {
        self.header.encrypted
    }
}

/// The receiving side of one key: a replay window per kind.
pub struct SealReceiver {
    aead: ChaCha20Poly1305,
    key_id: u32,
    kinds: KindSet,
    windows: [ReplayWindow; KINDS],
}

impl core::fmt::Debug for SealReceiver {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SealReceiver")
            .field("key_id", &self.key_id)
            .finish_non_exhaustive()
    }
}

impl SealReceiver {
    /// A receiver of `key` (id `key_id`) for `kinds`.
    pub fn new(key_id: u32, key: &Key, kinds: KindSet) -> Self {
        Self {
            aead: ChaCha20Poly1305::new(key.as_bytes().into()),
            key_id,
            kinds,
            windows: [ReplayWindow::new(); KINDS],
        }
    }

    /// The key's id.
    pub fn key_id(&self) -> u32 {
        self.key_id
    }

    /// Open `msg`, received on `keyexpr`, in place: check the header, the
    /// replay window and the tag, decrypt, then record the `seq`. The body
    /// borrows `msg`. Allocates only for an AAD over 256 bytes.
    ///
    /// # Errors
    /// [`OpenError`]; the window is unchanged then. `msg` may hold
    /// garbage after a failed decryption.
    pub fn open<'m>(&mut self, keyexpr: &str, msg: &'m mut [u8]) -> Result<Opened<'m>, OpenError> {
        let header = Header::parse(msg)?;
        if header.key_id != self.key_id {
            return Err(OpenError::WrongKey(header.key_id));
        }
        if !self.kinds.contains(header.kind) {
            return Err(OpenError::WrongKind(header.kind));
        }
        let window = &mut self.windows[header.kind.index()];
        window.check(header.seq)?;
        let nonce = Nonce::from(nonce(header.kind, header.seq));
        let (head, rest) = msg.split_at_mut(HEADER_LEN);
        let (body, tag) = rest.split_at_mut(rest.len() - TAG_LEN);
        let tag = Tag::clone_from_slice(tag);
        let mut header_bytes = [0u8; HEADER_LEN];
        header_bytes.copy_from_slice(head);
        let verified = if header.encrypted {
            with_aad(&header_bytes, keyexpr, None, |aad| {
                self.aead.decrypt_in_place_detached(&nonce, aad, body, &tag)
            })
        } else {
            with_aad(&header_bytes, keyexpr, Some(body), |aad| {
                self.aead
                    .decrypt_in_place_detached(&nonce, aad, &mut [], &tag)
            })
        };
        match verified {
            Ok(Ok(())) => {}
            Ok(Err(_)) => return Err(OpenError::Auth),
            Err(LpTooLong) => return Err(OpenError::KeyExprTooLong),
        }
        window.insert(header.seq)?;
        Ok(Opened { header, body })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEYEXPR: &str = "rlmesh/v2/acme/s/demo/p/arm/ctl";

    fn pair(kinds: KindSet) -> (SealSender, SealReceiver) {
        let key = Key::from_bytes([7; 32]);
        (
            SealSender::new(42, &key, kinds),
            SealReceiver::new(42, &key, kinds),
        )
    }

    #[test]
    fn round_trips_in_both_modes() {
        let (mut tx, mut rx) = pair(KindSet::ALL);
        let mut msg = tx.seal(Kind::CtlRequest, true, KEYEXPR, b"attach").unwrap();
        assert_eq!(msg.len(), OVERHEAD + 6);
        assert_ne!(
            &msg[HEADER_LEN..HEADER_LEN + 6],
            b"attach",
            "encrypted on the wire"
        );
        let opened = rx.open(KEYEXPR, &mut msg).unwrap();
        assert_eq!(opened.body, b"attach");
        assert_eq!(
            opened.header,
            Header {
                encrypted: true,
                kind: Kind::CtlRequest,
                key_id: 42,
                seq: 1
            }
        );

        let mut msg = tx.seal(Kind::EstopSignal, false, KEYEXPR, b"stop").unwrap();
        assert_eq!(&msg[HEADER_LEN..HEADER_LEN + 4], b"stop", "in clear");
        let opened = rx.open(KEYEXPR, &mut msg).unwrap();
        assert_eq!(opened.body, b"stop");
        assert!(!opened.encrypted());
        assert_eq!(opened.header.seq, 1, "one counter per kind");

        // An empty body, and a body that doesn't fit the inline AAD.
        for body in [&[][..], &[5u8; 1000][..]] {
            for encrypted in [false, true] {
                let mut msg = tx.seal(Kind::Sample, encrypted, KEYEXPR, body).unwrap();
                assert_eq!(rx.open(KEYEXPR, &mut msg).unwrap().body, body);
            }
        }
    }

    #[test]
    fn counters_are_dense_per_kind() {
        let (mut tx, _) = pair(KindSet::ALL);
        let seqs: Vec<u64> = [Kind::Sample, Kind::Sample, Kind::Applied, Kind::Sample]
            .into_iter()
            .map(|k| {
                Header::parse(&tx.seal(k, false, KEYEXPR, b"").unwrap())
                    .unwrap()
                    .seq
            })
            .collect();
        assert_eq!(seqs, [1, 2, 1, 3]);
    }

    #[test]
    fn rejects_bad_headers() {
        let (mut tx, mut rx) = pair(KindSet::ALL);
        let good = tx.seal(Kind::Sample, true, KEYEXPR, b"x").unwrap();

        let mut msg = good.clone();
        msg[0] = 2;
        assert_eq!(rx.open(KEYEXPR, &mut msg), Err(OpenError::Version(2)));
        for flags in [0b10, 0b11, 0x80] {
            let mut msg = good.clone();
            msg[1] = flags;
            assert_eq!(rx.open(KEYEXPR, &mut msg), Err(OpenError::Flags(flags)));
        }
        for kind in [0, 20, 21, 255] {
            let mut msg = good.clone();
            msg[2] = kind;
            assert_eq!(
                rx.open(KEYEXPR, &mut msg),
                Err(OpenError::UnknownKind(kind))
            );
        }
        assert_eq!(
            rx.open(KEYEXPR, &mut good.clone()[..OVERHEAD - 1]),
            Err(OpenError::Truncated)
        );
        // The good one still opens: nothing above moved the window.
        assert!(rx.open(KEYEXPR, &mut good.clone()).is_ok());
    }

    #[test]
    fn rejects_a_protection_the_kind_forbids() {
        let (mut tx, mut rx) = pair(KindSet::ALL);
        assert_eq!(
            tx.seal(Kind::CtlRequest, false, KEYEXPR, b"x"),
            Err(SealError::Protection(Kind::CtlRequest))
        );
        assert_eq!(
            tx.seal(Kind::EstopSignal, true, KEYEXPR, b"x"),
            Err(SealError::Protection(Kind::EstopSignal))
        );
        // A downgraded flag on the wire is refused before the tag.
        let mut msg = tx.seal(Kind::KeyDelivery, true, KEYEXPR, b"x").unwrap();
        msg[1] = 0;
        assert_eq!(
            rx.open(KEYEXPR, &mut msg),
            Err(OpenError::Protection(Kind::KeyDelivery))
        );
    }

    #[test]
    fn rejects_a_kind_outside_the_keys_scope() {
        let key = Key::from_bytes([7; 32]);
        let mut tx = SealSender::new(1, &key, KindSet::PAIRING);
        assert_eq!(
            tx.seal(Kind::CtlRequest, true, KEYEXPR, b"x"),
            Err(SealError::WrongKind(Kind::CtlRequest))
        );
        let mut any = SealSender::new(1, &key, KindSet::ALL);
        let mut rx = SealReceiver::new(1, &key, KindSet::of(&[Kind::Sample, Kind::Applied]));
        let mut msg = any.seal(Kind::ActionChunk, false, KEYEXPR, b"x").unwrap();
        assert_eq!(
            rx.open(KEYEXPR, &mut msg),
            Err(OpenError::WrongKind(Kind::ActionChunk))
        );
        let mut msg = any.seal(Kind::Applied, false, KEYEXPR, b"x").unwrap();
        assert!(rx.open(KEYEXPR, &mut msg).is_ok());
    }

    #[test]
    fn rejects_the_wrong_key() {
        let mut tx = SealSender::new(42, &Key::from_bytes([7; 32]), KindSet::ALL);
        let mut rx = SealReceiver::new(42, &Key::from_bytes([8; 32]), KindSet::ALL);
        for encrypted in [false, true] {
            let mut msg = tx.seal(Kind::Sample, encrypted, KEYEXPR, b"x").unwrap();
            assert_eq!(rx.open(KEYEXPR, &mut msg), Err(OpenError::Auth));
        }
        let mut other = SealReceiver::new(43, &Key::from_bytes([7; 32]), KindSet::ALL);
        let mut msg = tx.seal(Kind::Sample, true, KEYEXPR, b"x").unwrap();
        assert_eq!(other.open(KEYEXPR, &mut msg), Err(OpenError::WrongKey(42)));
    }

    #[test]
    fn rejects_the_wrong_key_expression() {
        let (mut tx, mut rx) = pair(KindSet::ALL);
        for encrypted in [false, true] {
            let msg = tx.seal(Kind::Sample, encrypted, KEYEXPR, b"x").unwrap();
            for other in [
                "rlmesh/v2/acme/s/demo/p/arm/ctX",
                "",
                "rlmesh/v2/acme/s/demo/p/arm/ctl/",
            ] {
                assert_eq!(
                    rx.open(other, &mut msg.clone()),
                    Err(OpenError::Auth),
                    "{other}"
                );
            }
            // Still unseen: the failures didn't consume the seq.
            assert!(rx.open(KEYEXPR, &mut msg.clone()).is_ok());
        }
    }

    #[test]
    fn rejects_any_tampered_byte() {
        let (mut tx, mut rx) = pair(KindSet::ALL);
        for (kind, encrypted) in [
            (Kind::Sample, false),
            (Kind::Sample, true),
            (Kind::CtlReply, true),
        ] {
            let msg = tx.seal(kind, encrypted, KEYEXPR, b"body!").unwrap();
            // The seq, the body and the tag reach the tag check (0x10 keeps
            // seq 1 a fresh seq rather than 0).
            for at in 7..msg.len() {
                let mut bad = msg.clone();
                bad[at] ^= 0x10;
                assert_eq!(
                    rx.open(KEYEXPR, &mut bad),
                    Err(OpenError::Auth),
                    "byte {at}"
                );
            }
            // So does another valid kind or flag: both are in the nonce or AAD.
            let mut bad = msg.clone();
            bad[2] = Kind::Applied.as_u8();
            assert_eq!(rx.open(KEYEXPR, &mut bad), Err(OpenError::Auth));
            if kind == Kind::Sample {
                let mut bad = msg.clone();
                bad[1] ^= FLAG_ENCRYPTED;
                assert_eq!(rx.open(KEYEXPR, &mut bad), Err(OpenError::Auth));
            }
            // A changed key_id names another key.
            let mut bad = msg.clone();
            bad[3] ^= 1;
            assert_eq!(rx.open(KEYEXPR, &mut bad), Err(OpenError::WrongKey(43)));
            assert!(rx.open(KEYEXPR, &mut msg.clone()).is_ok());
        }
    }

    #[test]
    fn rejects_replays() {
        let (mut tx, mut rx) = pair(KindSet::ALL);
        let msgs: Vec<Vec<u8>> = (0..70)
            .map(|_| tx.seal(Kind::Sample, false, KEYEXPR, b"s").unwrap())
            .collect();
        // Out of order inside the window: fine once each.
        assert!(rx.open(KEYEXPR, &mut msgs[3].clone()).is_ok());
        assert!(rx.open(KEYEXPR, &mut msgs[1].clone()).is_ok());
        assert_eq!(
            rx.open(KEYEXPR, &mut msgs[1].clone()),
            Err(Replayed(2).into())
        );
        assert!(rx.open(KEYEXPR, &mut msgs[69].clone()).is_ok());
        // seq 6 is 64 below 70: the oldest still held; seq 5 has fallen out.
        assert!(rx.open(KEYEXPR, &mut msgs[5].clone()).is_ok());
        assert_eq!(
            rx.open(KEYEXPR, &mut msgs[4].clone()),
            Err(Replayed(5).into())
        );
        // Windows are per kind: another kind's seq 1 is new.
        let mut applied = tx.seal(Kind::Applied, false, KEYEXPR, b"a").unwrap();
        assert!(rx.open(KEYEXPR, &mut applied).is_ok());
    }

    #[test]
    fn a_forgery_does_not_move_the_window() {
        let (mut tx, mut rx) = pair(KindSet::ALL);
        let mut far = tx.seal(Kind::Sample, false, KEYEXPR, b"x").unwrap();
        far[7..15].copy_from_slice(&1000u64.to_le_bytes());
        assert_eq!(rx.open(KEYEXPR, &mut far), Err(OpenError::Auth));
        let mut next = tx.seal(Kind::Sample, false, KEYEXPR, b"y").unwrap();
        assert!(rx.open(KEYEXPR, &mut next).is_ok());
    }

    #[test]
    fn seq_exhaustion_stops_the_flow() {
        let (mut tx, mut rx) = pair(KindSet::ALL);
        tx.set_next(Kind::Sample, u64::MAX);
        let mut last = tx.seal(Kind::Sample, false, KEYEXPR, b"x").unwrap();
        assert_eq!(Header::parse(&last).unwrap().seq, u64::MAX);
        assert!(rx.open(KEYEXPR, &mut last).is_ok());
        assert_eq!(
            tx.seal(Kind::Sample, false, KEYEXPR, b"x"),
            Err(SealError::SeqExhausted(Kind::Sample))
        );
        let mut out = vec![9];
        assert!(
            tx.seal_to(Kind::Sample, false, KEYEXPR, b"x", &mut out)
                .is_err()
        );
        assert_eq!(out, [9], "unchanged on error");
        // Other kinds keep going.
        assert!(tx.seal(Kind::Applied, false, KEYEXPR, b"x").is_ok());
    }

    #[test]
    fn rejects_a_key_expression_over_lp_max() {
        let (mut tx, mut rx) = pair(KindSet::ALL);
        let long = "k".repeat(super::super::lp::LP_MAX + 1);
        assert_eq!(
            tx.seal(Kind::Sample, true, &long, b"x"),
            Err(SealError::KeyExprTooLong)
        );
        let mut msg = tx.seal(Kind::Sample, true, KEYEXPR, b"x").unwrap();
        assert_eq!(rx.open(&long, &mut msg), Err(OpenError::KeyExprTooLong));
        // The failed seal used no seq.
        assert_eq!(Header::parse(&msg).unwrap().seq, 1);
    }
}
