//! Consistent Overhead Byte Stuffing: a frame's bytes with every `0x00`
//! removed, so `0x00` can delimit frames on the wire.
//!
//! [`encode`] reads its source as a chain of slices and looks ahead in
//! place, so a frame is encoded straight from its header, payload and CRC
//! without first copying them together: a board needs no transmit buffer.
//! [`Decoder`] takes one byte at a time into a fixed buffer.

/// Calls `out` with the COBS encoding of `src` (the slices read as one
/// sequence), without the trailing `0x00` delimiter. At most
/// `len + len / 254 + 1` bytes.
pub fn encode(src: &[&[u8]], mut out: impl FnMut(u8)) {
    let total: usize = src.iter().map(|s| s.len()).sum();
    let at = |mut i: usize| {
        for s in src {
            if i < s.len() {
                return s[i];
            }
            i -= s.len();
        }
        0
    };
    let mut i = 0;
    loop {
        let mut run = 0;
        while run < 254 && i + run < total && at(i + run) != 0 {
            run += 1;
        }
        // run < 255 by the loop bound, so this is exact.
        out((run + 1) as u8);
        for k in 0..run {
            out(at(i + k));
        }
        i += run;
        if i == total {
            break;
        }
        if run < 254 {
            // A zero ends this block: the code byte stands for it.
            i += 1;
        }
    }
}

/// Why [`Decoder::push`] dropped a frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeError {
    /// More than the buffer holds arrived before the delimiter; everything
    /// up to and including it was discarded.
    Overflow,
    /// The delimiter came inside a block: the frame is cut short.
    Truncated,
}

/// A byte-at-a-time COBS decoder into a buffer of `N` bytes. Every `0x00`
/// ends the current frame, valid or not, and the next byte starts a new
/// one, so a bad frame never costs the frame after it.
#[derive(Debug)]
pub struct Decoder<const N: usize> {
    buf: [u8; N],
    len: usize,
    /// Data bytes left in the current block.
    left: u8,
    /// The current block was shorter than 254 bytes, so a zero follows it
    /// unless the frame ends first.
    zero: bool,
    /// Bytes of this frame seen, encoded.
    seen: usize,
    overflow: bool,
}

impl<const N: usize> Default for Decoder<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> Decoder<N> {
    /// An empty decoder.
    pub const fn new() -> Self {
        Self {
            buf: [0; N],
            len: 0,
            left: 0,
            zero: false,
            seen: 0,
            overflow: false,
        }
    }

    /// Feeds one wire byte. On a delimiter, returns the decoded frame or why
    /// it was dropped; `None` otherwise, and for an empty frame (two
    /// delimiters in a row).
    pub fn push(&mut self, byte: u8) -> Option<Result<&[u8], DecodeError>> {
        if byte == 0 {
            let (len, left, seen, overflow) = (self.len, self.left, self.seen, self.overflow);
            self.reset();
            return if overflow {
                Some(Err(DecodeError::Overflow))
            } else if seen == 0 {
                None
            } else if left != 0 {
                Some(Err(DecodeError::Truncated))
            } else {
                Some(Ok(&self.buf[..len]))
            };
        }
        if self.overflow {
            return None;
        }
        self.seen += 1;
        if self.left == 0 {
            // A code byte: emit the zero the previous block stood for.
            if self.zero && !self.put(0) {
                return None;
            }
            self.left = byte - 1;
            self.zero = byte != 0xFF;
        } else {
            self.left -= 1;
            if !self.put(byte) {
                return None;
            }
        }
        None
    }

    /// Drops any partial frame.
    pub fn reset(&mut self) {
        self.len = 0;
        self.left = 0;
        self.zero = false;
        self.seen = 0;
        self.overflow = false;
    }

    fn put(&mut self, byte: u8) -> bool {
        if self.len == N {
            self.overflow = true;
            return false;
        }
        self.buf[self.len] = byte;
        self.len += 1;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc(src: &[u8]) -> ([u8; 600], usize) {
        let mut out = [0_u8; 600];
        let mut n = 0;
        encode(&[src], |b| {
            out[n] = b;
            n += 1;
        });
        (out, n)
    }

    fn dec(wire: &[u8]) -> Result<([u8; 512], usize), DecodeError> {
        let mut d = Decoder::<512>::new();
        let mut got = None;
        for &b in wire.iter().chain(&[0]) {
            if let Some(r) = d.push(b) {
                got = Some(r.map(|f| {
                    let mut buf = [0_u8; 512];
                    buf[..f.len()].copy_from_slice(f);
                    (buf, f.len())
                }));
            }
        }
        got.expect("a frame")
    }

    #[test]
    fn the_reference_examples_encode_exactly() {
        let cases: &[(&[u8], &[u8])] = &[
            (&[0x00], &[0x01, 0x01]),
            (&[0x00, 0x00], &[0x01, 0x01, 0x01]),
            (&[0x00, 0x11, 0x00], &[0x01, 0x02, 0x11, 0x01]),
            (&[0x11, 0x22, 0x00, 0x33], &[0x03, 0x11, 0x22, 0x02, 0x33]),
            (&[0x11, 0x22, 0x33, 0x44], &[0x05, 0x11, 0x22, 0x33, 0x44]),
            (&[0x11, 0x00, 0x00, 0x00], &[0x02, 0x11, 0x01, 0x01, 0x01]),
            (&[], &[0x01]),
        ];
        for (src, want) in cases {
            let (out, n) = enc(src);
            assert_eq!(&out[..n], *want, "{src:02x?}");
        }
    }

    #[test]
    fn long_runs_split_at_254() {
        // 254 non-zero bytes: one full block, no trailing code.
        let run: [u8; 254] = core::array::from_fn(|i| (i % 255 + 1) as u8);
        let (out, n) = enc(&run);
        assert_eq!(n, 255);
        assert_eq!(out[0], 0xFF);
        // 255: a full block, then a block of one.
        let mut longer = [1_u8; 255];
        longer[254] = 7;
        let (out, n) = enc(&longer);
        assert_eq!(n, 257);
        assert_eq!((out[0], out[255], out[256]), (0xFF, 0x02, 7));
    }

    #[test]
    fn chained_slices_encode_like_one() {
        let whole = [9_u8, 0, 3, 0, 0, 5, 6];
        let (a, n) = enc(&whole);
        let mut b = [0_u8; 32];
        let mut m = 0;
        encode(&[&whole[..2], &[], &whole[2..5], &whole[5..]], |x| {
            b[m] = x;
            m += 1;
        });
        assert_eq!(&a[..n], &b[..m]);
    }

    #[test]
    fn every_length_and_zero_pattern_round_trips() {
        let mut seed = 0x1234_5678_u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for len in 0..=300 {
            for density in [0, 1, 4, 64] {
                let mut src = [0_u8; 300];
                for b in &mut src[..len] {
                    let r = next();
                    *b = if density > 0 && r % density == 0 {
                        0
                    } else {
                        (r >> 8) as u8 | 1
                    };
                }
                let (wire, n) = enc(&src[..len]);
                assert!(n <= len + len / 254 + 1);
                assert!(!wire[..n].contains(&0), "no zero in the encoding");
                let (got, m) = dec(&wire[..n]).expect("decodes");
                assert_eq!(&got[..m], &src[..len], "len {len}");
            }
        }
    }

    #[test]
    fn a_bad_frame_never_costs_the_next_one() {
        let mut d = Decoder::<16>::new();
        // A truncated block (code says 4 data bytes, 2 arrive), then a good frame.
        let wire = [0x05, 0xAA, 0xBB, 0x00, 0x03, 0x11, 0x22, 0x00];
        let mut results = [None, None];
        let mut k = 0;
        for b in wire {
            if let Some(r) = d.push(b) {
                results[k] = Some(r.map(|f| f.len()));
                k += 1;
            }
        }
        assert_eq!(results, [Some(Err(DecodeError::Truncated)), Some(Ok(2))]);
    }

    #[test]
    fn overflow_discards_through_the_delimiter() {
        let mut d = Decoder::<4>::new();
        let mut results = [None, None];
        let mut k = 0;
        let wire = [0x08, 1, 2, 3, 4, 5, 6, 7, 0x00, 0x02, 9, 0x00];
        for b in wire {
            if let Some(r) = d.push(b) {
                results[k] = Some(r.map(|f| f.len()));
                k += 1;
            }
        }
        assert_eq!(results, [Some(Err(DecodeError::Overflow)), Some(Ok(1))]);
        // Delimiters alone are not frames.
        assert!(d.push(0).is_none());
    }
}
