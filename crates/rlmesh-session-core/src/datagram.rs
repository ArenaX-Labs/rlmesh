//! The datagram path: a robot's realtime parts between its board and the
//! host over UDP instead of the session's TCP links, where a lost frame
//! stalls everything behind it until it is sent again.
//!
//! Only what is latest-wins at every layer above rides it: action chunks
//! down (`c/{robot}/{path}`) and a robot's samples and `_applied` entries up
//! (`d/{robot}/{path}`). Each datagram carries exactly the sealed envelope the
//! session would carry on that key, behind a small [`Header`] that names the
//! key by a route the host listed in the robot's device view. The seal is
//! what makes an unauthenticated socket safe: a datagram that does not open
//! under its route's key is dropped and counted ([`Arrivals::unopened`]),
//! and the seal's own sequence and replay window refuse a copy. The header's
//! sequence counts losses and reordering ([`Arrivals`]) and nothing else.
//!
//! The host sends a robot nothing until it opened a [`Kind::Probe`] from it
//! under the robot's `k_up`, and goes on sending to the address it last
//! opened a datagram from ([`Learned`]): a spoofed source redirects nothing,
//! and a robot whose address changes is followed.
//!
//! [`Kind::Probe`]: Kind::Probe

#[allow(unused_imports)]
use crate::prelude::*;

use bytes::Bytes;

use crate::crypto::sealed::OVERHEAD;
use crate::wire::codec;
use crate::wire::{
    ActionChunk, Applied, ChannelId, EdgeMode, Encoding, Epoch, Fingerprint, InputRange, Layout,
    LeaseToken, ParticipantId, ReasonCode, Sample, StreamSpec,
};

/// The header's version, which [`crate::wire::DatagramOffer`] names.
pub const VERSION: u8 = 1;
/// The header's first byte: its magic in the high nibble, its version in
/// the low.
const MAGIC: u8 = 0xD0;
/// The header's length.
pub const HEADER_LEN: usize = 8;
/// The most one datagram carries, header included: under a 1500-byte MTU's
/// 1472 bytes of UDP payload with room for a tunnel, so it is never
/// fragmented. A board reassembles no IP fragments, so a larger one would be
/// lost without a word.
pub const MAX_DATAGRAM: usize = 1400;
/// The most a receiver reads of one datagram: anything past
/// [`MAX_DATAGRAM`] up to this is read only to be refused.
pub const MAX_RECEIVED: usize = 1472;

/// What a datagram carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// An action chunk, host to board.
    Chunk = 1,
    /// A sample or `_applied` entry, board to host.
    Sample = 2,
    /// A probe, board to host: a `Sealed` [`DatagramProbe`] under the
    /// robot's `k_up`.
    ///
    /// [`DatagramProbe`]: crate::crypto::Kind::DatagramProbe
    Probe = 3,
}

impl Kind {
    fn of(byte: u8) -> Option<Self> {
        match byte {
            1 => Some(Self::Chunk),
            2 => Some(Self::Sample),
            3 => Some(Self::Probe),
            _ => None,
        }
    }
}

/// The header in front of every datagram, little-endian:
///
/// | Bytes | Field |
/// |---|---|
/// | 0 | `0xD0` + version ([`VERSION`]) |
/// | 1 | [`Kind`] |
/// | 2–3 | route |
/// | 4–7 | statistics sequence, per sender and direction |
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    /// What it carries.
    pub kind: Kind,
    /// The route: the key its payload belongs on.
    pub route: u16,
    /// Counts losses and reordering, never acceptance: the sealed payload's
    /// own sequence is what a receiver trusts.
    pub seq: u32,
}

/// Why a datagram was not read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unreadable {
    /// Shorter than the header, or not this magic, version or a known kind.
    Header,
    /// Longer than [`MAX_DATAGRAM`].
    TooLarge,
}

impl Header {
    /// The header's bytes.
    pub fn to_bytes(self) -> [u8; HEADER_LEN] {
        let mut out = [0; HEADER_LEN];
        out[0] = MAGIC | VERSION;
        out[1] = self.kind as u8;
        out[2..4].copy_from_slice(&self.route.to_le_bytes());
        out[4..8].copy_from_slice(&self.seq.to_le_bytes());
        out
    }

    /// The header of `datagram`, and its payload.
    ///
    /// # Errors
    /// [`Unreadable`]: refuse it unread.
    pub fn parse(datagram: &[u8]) -> Result<(Self, &[u8]), Unreadable> {
        if datagram.len() > MAX_DATAGRAM {
            return Err(Unreadable::TooLarge);
        }
        let (head, payload) = datagram
            .split_first_chunk::<HEADER_LEN>()
            .ok_or(Unreadable::Header)?;
        if head[0] != MAGIC | VERSION {
            return Err(Unreadable::Header);
        }
        let kind = Kind::of(head[1]).ok_or(Unreadable::Header)?;
        Ok((
            Self {
                kind,
                route: u16::from_le_bytes([head[2], head[3]]),
                seq: u32::from_le_bytes([head[4], head[5], head[6], head[7]]),
            },
            payload,
        ))
    }

    /// The datagram of `payload` behind this header, or `None` when it would
    /// be over [`MAX_DATAGRAM`] or the heap has no room now.
    pub fn frame(self, payload: &[u8]) -> Option<Vec<u8>> {
        let len = HEADER_LEN.checked_add(payload.len())?;
        if len > MAX_DATAGRAM {
            return None;
        }
        let mut out = Vec::new();
        out.try_reserve_exact(len).ok()?;
        out.extend_from_slice(&self.to_bytes());
        out.extend_from_slice(payload);
        Some(out)
    }
}

/// Whether a sealed envelope of `sealed` bytes fits one datagram.
pub fn fits(sealed: usize) -> bool {
    sealed.saturating_add(HEADER_LEN) <= MAX_DATAGRAM
}

/// The longest sealed sample `stream` can publish: every field of its
/// envelope at its widest, and its largest payload.
pub fn sample_bound(stream: &StreamSpec) -> usize {
    let payload = match (&stream.layout, stream.encoding) {
        (Layout::Packed { stride, .. }, Encoding::Raw) => *stride,
        (Layout::Packed { stride, .. }, Encoding::Jpeg { .. }) => {
            stride.saturating_add(crate::wire::part::MAX_ENCODED_OVERHEAD)
        }
        (Layout::Text, _) => crate::wire::part::MAX_TEXT_BYTES,
    };
    let widest = Sample {
        channel: stream.channel,
        epoch: Epoch(u64::MAX),
        seq: u64::MAX,
        capture_ns: i64::MIN,
        publish_ns: i64::MIN,
        tick: u64::MAX,
        // Only its length is encoded into the bound.
        payload: Bytes::new(),
        applied: stream.applied.then_some(Applied {
            mode: EdgeMode::Estop,
            present: true,
            lease_token: LeaseToken(u64::MAX),
            chunk_seq: u64::MAX,
            action_index: u32::MAX,
            reason: ReasonCode::Unspecified,
        }),
    };
    let envelope = codec::encode_sample(&widest).to_vec().len();
    envelope
        .saturating_add(payload)
        .saturating_add(varint_len(payload))
        .saturating_add(1)
        .saturating_add(OVERHEAD)
}

/// The longest sealed chunk of `count` actions of `stride` bytes that
/// `source` can send under a binding with `inputs` input streams: every
/// field of its envelope at its widest.
pub fn chunk_bound(
    channel: ChannelId,
    source: &ParticipantId,
    inputs: usize,
    count: u32,
    stride: usize,
) -> usize {
    let actions = (count as usize).saturating_mul(stride);
    let widest = ActionChunk {
        channel,
        epoch: Epoch(u64::MAX),
        lease_token: LeaseToken(u64::MAX),
        seq: u64::MAX,
        source: source.clone(),
        binding: Fingerprint::from(u64::MAX),
        based_on: (0..inputs.max(1))
            .map(|_| InputRange {
                channel: ChannelId(u32::MAX),
                seq_first: u64::MAX,
                seq_last: u64::MAX,
            })
            .collect(),
        t0_ns: i64::MIN,
        dt_ns: u64::MAX,
        replace_from_ns: i64::MIN,
        valid_until_ns: i64::MIN,
        count,
        // Only its length is encoded into the bound.
        actions: Bytes::new(),
        publish_ns: i64::MIN,
        tick: u64::MAX,
    };
    let envelope = codec::encode_chunk(&widest).map_or(usize::MAX, |e| e.to_vec().len());
    envelope
        .saturating_add(actions)
        .saturating_add(varint_len(actions))
        .saturating_add(1)
        .saturating_add(OVERHEAD)
}

/// The bytes of `n` as a protobuf varint.
fn varint_len(n: usize) -> usize {
    let bits = usize::BITS - n.leading_zeros();
    (bits.max(1) as usize).div_ceil(7)
}

/// What one end received on a robot's datagram path, by the other end's
/// statistics sequence: only datagrams that opened move it, so a forged
/// header changes nothing but [`Self::unopened`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Arrivals {
    highest: Option<u32>,
    /// Datagrams that opened.
    pub received: u64,
    /// Sequence numbers skipped and not (yet) arrived.
    pub lost: u64,
    /// Datagrams that arrived after a later one.
    pub reordered: u64,
    /// Datagrams that were unreadable or failed to open.
    pub unopened: u64,
}

impl Arrivals {
    /// A datagram that opened, with statistics sequence `seq`.
    pub fn opened(&mut self, seq: u32) {
        self.received += 1;
        let Some(highest) = self.highest else {
            self.highest = Some(seq);
            return;
        };
        // Signed distance, so the sequence may wrap.
        let ahead = seq.wrapping_sub(highest).cast_signed();
        if ahead > 0 {
            self.lost += u64::from(ahead.unsigned_abs() - 1);
            self.highest = Some(seq);
        } else if ahead < 0 {
            // It was counted lost when the later one arrived.
            self.reordered += 1;
            self.lost = self.lost.saturating_sub(1);
        }
    }

    /// A datagram that was unreadable or did not open.
    pub fn refused(&mut self) {
        self.unopened += 1;
    }

    /// Starts over: the other end's sequence starts over with it.
    pub fn restart(&mut self) {
        self.highest = None;
    }
}

/// Where the host sends a robot's datagrams: the source of the last one it
/// opened from that robot, and nowhere before the first. Only a datagram
/// that opened under the robot's keys moves it, so a spoofed source never
/// redirects the stream (nor turns the host on a third party), and a robot
/// whose address changes (NAT rebinding, a DHCP renewal) is followed with
/// its next one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Learned<A> {
    at: Option<A>,
    /// Times it moved after the first.
    pub moves: u32,
}

impl<A: PartialEq + Copy> Learned<A> {
    /// A datagram from `from` opened: send there from now on. Whether it
    /// moved.
    pub fn opened(&mut self, from: A) -> bool {
        match self.at {
            Some(at) if at == from => false,
            Some(_) => {
                self.at = Some(from);
                self.moves = self.moves.saturating_add(1);
                true
            }
            None => {
                self.at = Some(from);
                true
            }
        }
    }

    /// Where to send, if anywhere yet.
    pub fn get(&self) -> Option<A> {
        self.at
    }
}

/// One route's datagram waiting to go: only the newest. A newer one replaces
/// it, so a path that cannot keep up sends the latest instead of a backlog.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Latest<T> {
    waiting: Option<T>,
    /// Datagrams replaced before they went.
    pub replaced: u64,
}

impl<T> Latest<T> {
    /// `datagram` waits in place of any that did.
    pub fn put(&mut self, datagram: T) {
        if self.waiting.replace(datagram).is_some() {
            self.replaced += 1;
        }
    }

    /// The one waiting, now the caller's.
    pub fn take(&mut self) -> Option<T> {
        self.waiting.take()
    }

    /// Whether one is waiting.
    pub fn is_waiting(&self) -> bool {
        self.waiting.is_some()
    }
}

/// Route ids, handed out without reuse while held: a datagram for a route
/// that went (a part moved back to the session, a robot detached) names no
/// other route.
#[derive(Clone, Debug, Default)]
pub struct RouteIds {
    next: u16,
    held: alloc::collections::BTreeSet<u16>,
}

impl RouteIds {
    /// A free id (never 0), or `None` when every one is held.
    pub fn take(&mut self) -> Option<u16> {
        for _ in 0..=u16::MAX {
            self.next = self.next.wrapping_add(1).max(1);
            if self.held.insert(self.next) {
                return Some(self.next);
            }
        }
        None
    }

    /// `id` is free again.
    pub fn give_back(&mut self, id: u16) {
        self.held.remove(&id);
    }

    /// Ids held now.
    pub fn held(&self) -> usize {
        self.held.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::{ChannelPriority, DType, FieldSpec};

    #[test]
    fn a_header_round_trips_and_anything_else_is_refused_unread() {
        let header = Header {
            kind: Kind::Sample,
            route: 0xBEEF,
            seq: 0x0102_0304,
        };
        let datagram = header.frame(b"sealed").expect("fits");
        assert_eq!(
            datagram[..HEADER_LEN],
            [0xD1, 2, 0xEF, 0xBE, 4, 3, 2, 1],
            "the documented layout"
        );
        assert_eq!(Header::parse(&datagram), Ok((header, &b"sealed"[..])));
        assert_eq!(Header::parse(&datagram[..7]), Err(Unreadable::Header));
        for (at, byte) in [(0, 0xD2), (0, 0xC1), (1, 0), (1, 4)] {
            let mut bad = datagram.clone();
            bad[at] = byte;
            assert_eq!(Header::parse(&bad), Err(Unreadable::Header), "{at}={byte}");
        }
        // The largest that fits, and one byte more.
        let full = vec![0; MAX_DATAGRAM - HEADER_LEN];
        assert!(header.frame(&full).is_some());
        assert!(header.frame(&[0; MAX_DATAGRAM - HEADER_LEN + 1]).is_none());
        let mut over = header.frame(&full).expect("fits");
        over.push(0);
        assert_eq!(Header::parse(&over), Err(Unreadable::TooLarge));
        assert!(fits(MAX_DATAGRAM - HEADER_LEN) && !fits(MAX_DATAGRAM - HEADER_LEN + 1));
    }

    #[test]
    fn arrivals_count_losses_and_reordering_and_only_what_opened() {
        let mut a = Arrivals::default();
        for seq in [1, 2, 5, 3, 6, 6] {
            a.opened(seq);
        }
        a.refused();
        assert_eq!((a.received, a.lost, a.reordered, a.unopened), (6, 1, 1, 1));
        // Across the wrap.
        let mut a = Arrivals::default();
        for seq in [u32::MAX - 1, u32::MAX, 1] {
            a.opened(seq);
        }
        assert_eq!((a.received, a.lost, a.reordered), (3, 1, 0));
        // A sender that started over is not a loss of billions.
        a.restart();
        a.opened(7);
        assert_eq!(a.lost, 1);
    }

    #[test]
    fn an_address_is_learned_only_from_what_opened_and_followed_when_it_moves() {
        let mut learned = Learned::<u32>::default();
        assert_eq!(learned.get(), None, "nowhere before the first");
        assert!(learned.opened(1));
        assert!(!learned.opened(1));
        assert_eq!(learned.get(), Some(1));
        assert!(learned.opened(2), "a robot whose address changed");
        assert_eq!((learned.get(), learned.moves), (Some(2), 1));
    }

    #[test]
    fn the_newest_datagram_replaces_the_one_waiting() {
        let mut slot = Latest::default();
        slot.put(1);
        slot.put(2);
        slot.put(3);
        assert_eq!((slot.take(), slot.replaced), (Some(3), 2));
        assert!(!slot.is_waiting());
        assert_eq!(slot.take(), None);
    }

    #[test]
    fn route_ids_are_never_zero_nor_handed_out_twice_while_held() {
        let mut ids = RouteIds::default();
        let a = ids.take().expect("id");
        let b = ids.take().expect("id");
        assert!(a != 0 && b != 0 && a != b);
        ids.give_back(a);
        assert_eq!(ids.held(), 1);
        let mut all = RouteIds::default();
        for _ in 0..u16::MAX {
            all.take().expect("an id");
        }
        assert_eq!(all.take(), None, "every id held");
    }

    #[test]
    fn the_bounds_hold_every_real_sample_and_chunk_and_the_demo_fits() {
        let owner = ParticipantId::new("eyeball-left").expect("id");
        let field = FieldSpec::new("pos", DType::F32, &[2]);
        let layout = Layout::of(core::slice::from_ref(&field)).expect("layout");
        let stream = StreamSpec {
            channel: ChannelId(7),
            owner: owner.clone(),
            path: "eyes".into(),
            fields: vec![field],
            layout,
            encoding: Encoding::Raw,
            priority: ChannelPriority::Realtime,
            event: false,
            max_age: None,
            applied: true,
        };
        let real = Sample {
            channel: ChannelId(7),
            epoch: Epoch(3),
            seq: 40,
            capture_ns: 1_000_000,
            publish_ns: 1_000_100,
            tick: 0,
            payload: Bytes::from_static(&[0; 8]),
            applied: Some(Applied::absent(EdgeMode::Driving)),
        };
        let bound = sample_bound(&stream);
        assert!(codec::encode_sample(&real).to_vec().len() + OVERHEAD <= bound);
        assert!(fits(bound), "a two-joint state fits: {bound} bytes");

        let source = ParticipantId::new("policy").expect("id");
        let chunk = ActionChunk {
            channel: ChannelId(7),
            epoch: Epoch(u64::MAX),
            lease_token: LeaseToken(u64::MAX),
            seq: u64::MAX,
            source: source.clone(),
            binding: Fingerprint::from(9),
            based_on: vec![InputRange::one(ChannelId(1), u64::MAX)],
            t0_ns: i64::MIN,
            dt_ns: u64::MAX,
            replace_from_ns: i64::MIN,
            valid_until_ns: i64::MIN,
            count: 4,
            actions: Bytes::from(vec![0; 4 * 8]),
            publish_ns: i64::MIN,
            tick: u64::MAX,
        };
        let real = codec::encode_chunk(&chunk).expect("encodes").to_vec().len() + OVERHEAD;
        let bound = chunk_bound(ChannelId(7), &source, 1, 4, 8);
        assert!(real <= bound, "{real} > {bound}");
        assert!(
            fits(bound),
            "the demo's 4 actions x 2 joints fit: {bound} bytes"
        );
        // 200 actions of 2 x f32 do not.
        assert!(!fits(chunk_bound(ChannelId(7), &source, 1, 200, 8)));
    }
}
