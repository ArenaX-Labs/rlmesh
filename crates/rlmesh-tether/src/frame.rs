//! Frames: `type: u8 | seq: u8 | payload | crc: u16 LE`, COBS-encoded and
//! ended by `0x00` (spec §3). The CRC covers `type | seq | payload`.

use crate::{cobs, crc};

/// The largest decoded frame: header, payload and CRC.
pub const MAX_DECODED: usize = 256;
/// The largest encoded frame, without its delimiter.
pub const MAX_ENCODED: usize = MAX_DECODED + MAX_DECODED / 254 + 1;
/// The smallest decoded frame: header, a one-byte payload and CRC.
pub const MIN_DECODED: usize = 5;
/// The largest payload.
pub const MAX_PAYLOAD: usize = MAX_DECODED - 4;

/// One decoded, CRC-checked frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Frame<'a> {
    /// The message type. Below 0x80 board → bridge, from 0x80 bridge →
    /// board.
    pub ty: u8,
    /// The sender's frame counter, for loss accounting only.
    pub seq: u8,
    /// The message body.
    pub payload: &'a [u8],
}

/// Why a decoded frame was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameError {
    /// Shorter than [`MIN_DECODED`] or longer than [`MAX_DECODED`].
    Size,
    /// The CRC doesn't match.
    Crc,
}

impl<'a> Frame<'a> {
    /// Checks and splits a COBS-decoded frame.
    pub fn parse(decoded: &'a [u8]) -> Result<Self, FrameError> {
        if !(MIN_DECODED..=MAX_DECODED).contains(&decoded.len()) {
            return Err(FrameError::Size);
        }
        let (body, tail) = decoded.split_at(decoded.len() - 2);
        if crc::crc16(body) != u16::from_le_bytes([tail[0], tail[1]]) {
            return Err(FrameError::Crc);
        }
        Ok(Self {
            ty: body[0],
            seq: body[1],
            payload: &body[2..],
        })
    }
}

/// Writes one frame to `out`: COBS-encoded, with the trailing `0x00`.
/// `payload` is read in pieces (a message header, then a value), so nothing
/// is copied together first.
///
/// # Panics
///
/// When the payload is empty or longer than [`MAX_PAYLOAD`]: callers build
/// payloads from fixed message layouts.
pub fn write(ty: u8, seq: u8, payload: &[&[u8]], mut out: impl FnMut(u8)) {
    let len: usize = payload.iter().map(|p| p.len()).sum();
    assert!(
        (1..=MAX_PAYLOAD).contains(&len),
        "payload of {len} bytes (1..={MAX_PAYLOAD})"
    );
    let head = [ty, seq];
    let crc = payload
        .iter()
        .fold(crc::update(crc::INIT, &head), |c, p| crc::update(c, p))
        .to_le_bytes();
    let mut parts: [&[u8]; 6] = [&[]; 6];
    parts[0] = &head;
    let mut n = 1;
    for p in payload {
        assert!(n < parts.len() - 1, "at most 4 payload pieces");
        parts[n] = p;
        n += 1;
    }
    parts[n] = &crc;
    cobs::encode(&parts[..=n], &mut out);
    out(0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cobs::{DecodeError, Decoder};

    fn wire(ty: u8, seq: u8, payload: &[&[u8]]) -> ([u8; 300], usize) {
        let mut buf = [0_u8; 300];
        let mut n = 0;
        write(ty, seq, payload, |b| {
            buf[n] = b;
            n += 1;
        });
        (buf, n)
    }

    #[test]
    fn a_frame_round_trips_through_the_wire() {
        let (buf, n) = wire(0x83, 7, &[&[1, 0, 0, 60, 0], &[0, 0, 0x80, 0x3F]]);
        assert_eq!(buf[n - 1], 0, "ends with the delimiter");
        assert!(!buf[..n - 1].contains(&0));
        let mut d = Decoder::<MAX_DECODED>::new();
        let mut frames = 0;
        for &b in &buf[..n] {
            if let Some(decoded) = d.push(b) {
                let f = Frame::parse(decoded.expect("decodes")).expect("valid");
                assert_eq!((f.ty, f.seq), (0x83, 7));
                assert_eq!(f.payload, [1, 0, 0, 60, 0, 0, 0, 0x80, 0x3F]);
                frames += 1;
            }
        }
        assert_eq!(frames, 1);
    }

    #[test]
    fn the_largest_frame_fits_the_bounds() {
        let payload = [0x55_u8; MAX_PAYLOAD];
        let (buf, n) = wire(0x02, 0, &[&payload]);
        assert!(n - 1 <= MAX_ENCODED, "{} encoded bytes", n - 1);
        let mut d = Decoder::<MAX_DECODED>::new();
        let got = buf[..n]
            .iter()
            .find_map(|&b| d.push(b).map(|r| r.map(|f| f.len())));
        assert_eq!(got, Some(Ok(MAX_DECODED)));
        // One byte more doesn't fit a MAX_DECODED buffer.
        let mut small = Decoder::<{ MAX_DECODED - 1 }>::new();
        let got = buf[..n]
            .iter()
            .find_map(|&b| small.push(b).map(|r| r.map(|f| f.len())));
        assert_eq!(got, Some(Err(DecodeError::Overflow)));
    }

    #[test]
    fn corruption_and_size_are_refused() {
        let mut body = [0x85_u8, 3, 9, 0, 0];
        let crc = crc::crc16(&body[..3]).to_le_bytes();
        body[3..].copy_from_slice(&crc);
        assert!(Frame::parse(&body).is_ok());
        body[2] ^= 1;
        assert_eq!(Frame::parse(&body), Err(FrameError::Crc));
        assert_eq!(Frame::parse(&body[..4]), Err(FrameError::Size));
        assert_eq!(Frame::parse(&[0; MAX_DECODED + 1]), Err(FrameError::Size));
    }

    #[test]
    #[should_panic(expected = "payload of 0 bytes")]
    fn an_empty_payload_is_a_caller_bug() {
        let _ = wire(0x85, 0, &[]);
    }
}
