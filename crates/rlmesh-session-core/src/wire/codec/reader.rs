//! Protobuf fields read in place: nothing copied or allocated, every length
//! checked against what is left, and every error said without allocating.

use crate::wire::error::{Unread, WireError};

/// One field's value, borrowed from the message.
#[derive(Debug, Clone, Copy)]
pub(super) enum Value<'a> {
    Varint(u64),
    Fixed64(u64),
    /// No field this crate reads is one: skipped.
    Fixed32,
    Len(&'a [u8]),
}

/// The fields of one message, in order.
#[derive(Debug, Clone)]
pub(super) struct Fields<'a> {
    buf: &'a [u8],
    message: &'static str,
}

impl<'a> Fields<'a> {
    pub(super) fn new(buf: &'a [u8], message: &'static str) -> Self {
        Self { buf, message }
    }

    fn bad(&self, what: &'static str) -> WireError {
        WireError::unread(self.message, Unread::Malformed(what))
    }

    /// The next field and its value; `None` at the end.
    pub(super) fn next_field(&mut self) -> Result<Option<(u32, Value<'a>)>, WireError> {
        if self.buf.is_empty() {
            return Ok(None);
        }
        let key = varint(&mut self.buf).ok_or_else(|| self.bad("field key"))?;
        let tag = u32::try_from(key >> 3)
            .ok()
            .filter(|t| *t > 0)
            .ok_or_else(|| self.bad("field number"))?;
        let value = match key & 7 {
            0 => Value::Varint(varint(&mut self.buf).ok_or_else(|| self.bad("varint"))?),
            1 => Value::Fixed64(u64::from_le_bytes(self.take::<8>()?)),
            5 => {
                self.take::<4>()?;
                Value::Fixed32
            }
            2 => {
                let len = varint(&mut self.buf).ok_or_else(|| self.bad("length"))?;
                let len = usize::try_from(len)
                    .ok()
                    .filter(|l| *l <= self.buf.len())
                    .ok_or_else(|| self.bad("length past the end"))?;
                let (field, rest) = self.buf.split_at(len);
                self.buf = rest;
                Value::Len(field)
            }
            // Groups (3, 4) are protobuf 2's, never this schema's.
            _ => return Err(self.bad("wire type")),
        };
        Ok(Some((tag, value)))
    }

    fn take<const N: usize>(&mut self) -> Result<[u8; N], WireError> {
        let Some((head, rest)) = self.buf.split_first_chunk::<N>() else {
            return Err(self.bad("fixed field past the end"));
        };
        self.buf = rest;
        Ok(*head)
    }
}

/// A base-128 varint: at most ten bytes, the tenth holding only bit 63.
fn varint(buf: &mut &[u8]) -> Option<u64> {
    let mut value = 0u64;
    for i in 0..10 {
        let (&byte, rest) = buf.split_first()?;
        *buf = rest;
        if i == 9 && byte > 1 {
            return None;
        }
        value |= u64::from(byte & 0x7f) << (7 * i);
        if byte & 0x80 == 0 {
            return Some(value);
        }
    }
    None
}

impl<'a> Value<'a> {
    fn bad(message: &'static str) -> WireError {
        WireError::unread(message, Unread::Malformed("wire type"))
    }

    /// A `uint64`.
    pub(super) fn u64(self, message: &'static str) -> Result<u64, WireError> {
        match self {
            Self::Varint(v) => Ok(v),
            _ => Err(Self::bad(message)),
        }
    }

    /// An `int64`.
    pub(super) fn i64(self, message: &'static str) -> Result<i64, WireError> {
        self.u64(message).map(u64::cast_signed)
    }

    /// A `uint32`: truncated, as protobuf decoders do.
    #[allow(clippy::cast_possible_truncation)]
    pub(super) fn u32(self, message: &'static str) -> Result<u32, WireError> {
        self.u64(message).map(|v| v as u32)
    }

    /// An `int32` or an enum's raw value.
    #[allow(clippy::cast_possible_truncation)]
    pub(super) fn i32(self, message: &'static str) -> Result<i32, WireError> {
        self.u64(message).map(|v| v as i32)
    }

    pub(super) fn bool(self, message: &'static str) -> Result<bool, WireError> {
        self.u64(message).map(|v| v != 0)
    }

    /// A `double`.
    pub(super) fn f64(self, message: &'static str) -> Result<f64, WireError> {
        match self {
            Self::Fixed64(v) => Ok(f64::from_bits(v)),
            _ => Err(Self::bad(message)),
        }
    }

    /// A `bytes` field, or an embedded message.
    pub(super) fn bytes(self, message: &'static str) -> Result<&'a [u8], WireError> {
        match self {
            Self::Len(b) => Ok(b),
            _ => Err(Self::bad(message)),
        }
    }

    /// A `string`.
    pub(super) fn str(self, message: &'static str) -> Result<&'a str, WireError> {
        core::str::from_utf8(self.bytes(message)?)
            .map_err(|_| WireError::unread(message, Unread::Malformed("UTF-8")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::prelude::*;

    fn fields(buf: &[u8]) -> Result<Vec<(u32, u64)>, WireError> {
        let mut fields = Fields::new(buf, "test");
        let mut out = Vec::new();
        while let Some((tag, value)) = fields.next_field()? {
            let v = match value {
                Value::Varint(v) | Value::Fixed64(v) => v,
                Value::Fixed32 => 4,
                Value::Len(b) => b.len() as u64,
            };
            out.push((tag, v));
        }
        Ok(out)
    }

    #[test]
    fn it_reads_each_wire_type_and_refuses_what_runs_past_the_end() {
        // 1: varint 300; 2: 3 bytes; 3: fixed64; 4: fixed32.
        let buf = [
            0x08, 0xac, 0x02, 0x12, 3, b'a', b'b', b'c', 0x19, 1, 0, 0, 0, 0, 0, 0, 0, 0x25, 2, 0,
            0, 0,
        ];
        assert_eq!(fields(&buf).unwrap(), [(1, 300), (2, 3), (3, 1), (4, 4)]);
        for cut in 1..buf.len() {
            let truncated = fields(&buf[..cut]);
            let at_boundary = [3, 8, 17].contains(&cut);
            assert_eq!(truncated.is_ok(), at_boundary, "cut at {cut}");
        }
        // A length past the end, an eleven-byte varint, a group, field 0.
        assert!(fields(&[0x12, 0xff, 0xff, 0xff, 0xff, 0x0f]).is_err());
        assert!(
            fields(&[
                0x08, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 1
            ])
            .is_err()
        );
        assert!(fields(&[0x0b]).is_err());
        assert!(fields(&[0x00, 0x01]).is_err());
    }
}
