//! What a message a board decodes may hold, checked in place before the
//! decoder allocates for it.
//!
//! A protobuf decoder allocates per element of a repeated field: an empty
//! element costs two bytes on the wire and its whole size in memory (a
//! `PartSpec` takes 272 bytes on a 32-bit board), and a vector doubles as
//! it grows. A [`Shape`] caps each repeated field's count and each repeated
//! string's length, so the most a message can make its decoder allocate is
//! known before it runs. A message over a cap is refused unread.

use super::reader::{Fields, Value};
use crate::prelude::*;
use crate::wire::error::{Unread, WireError};

/// The caps of one message type.
#[derive(Debug)]
pub(super) struct Shape {
    pub(super) message: &'static str,
    pub(super) fields: &'static [(u32, Field)],
}

/// The cap on one field. A field without one is a singular scalar,
/// string or message holding none of these (bounded by the message's own
/// length), or one the decoder skips.
#[derive(Debug, Clone, Copy)]
pub(super) enum Field {
    /// A message field: at most `max` of it (1 for a singular one), each
    /// held to `shape`.
    Message { max: u16, shape: &'static Shape },
    /// A repeated string or bytes field: at most `max`, each at most `len`
    /// bytes.
    Text { max: u16, len: u16 },
    /// A repeated scalar field, packed or not: at most `max` values, each
    /// `width` bytes when packed (0: varints).
    Scalars { max: u16, width: u8 },
}

/// The most rules one shape has.
const MAX_RULES: usize = 8;

/// Checks `buf` against `shape`.
///
/// # Errors
/// [`WireError::Unread`]: malformed, or over a cap.
pub(super) fn check(buf: &[u8], shape: &Shape) -> Result<(), WireError> {
    debug_assert!(shape.fields.len() <= MAX_RULES);
    let over = || WireError::unread(shape.message, Unread::TooLarge(shape.message));
    let mut counts = [0u16; MAX_RULES];
    let mut fields = Fields::new(buf, shape.message);
    while let Some((tag, value)) = fields.next_field()? {
        let Some(at) = shape.fields.iter().position(|(t, _)| *t == tag) else {
            continue;
        };
        let (count, max) = match (shape.fields[at].1, value) {
            (Field::Message { max, shape }, Value::Len(sub)) => {
                check(sub, shape)?;
                (1, max)
            }
            (Field::Text { max, len }, Value::Len(text)) => {
                if text.len() > usize::from(len) {
                    return Err(over());
                }
                (1, max)
            }
            (Field::Scalars { max, width }, Value::Len(packed)) => {
                let n = match width {
                    0 => packed.iter().filter(|b| *b & 0x80 == 0).count(),
                    w => packed.len() / usize::from(w),
                };
                (u16::try_from(n).map_err(|_| over())?, max)
            }
            (Field::Scalars { max, .. }, _) => (1, max),
            _ => {
                return Err(WireError::unread(
                    shape.message,
                    Unread::Malformed("wire type"),
                ));
            }
        };
        counts[at] = counts[at].saturating_add(count);
        if counts[at] > max {
            return Err(over());
        }
    }
    Ok(())
}

/// Ids, paths and kinds, as a repeated string.
const NAME: u16 = 128;
/// A host address, `host:port`.
const ADDRESS: u16 = 256;

const EMPTY: Shape = Shape {
    message: "message",
    fields: &[],
};

/// `Key`: the parts a key covers.
pub(super) const KEY: Shape = Shape {
    message: "key",
    fields: &[(5, Field::Scalars { max: 256, width: 0 })],
};

/// `ActionChunk`: the inputs it was computed from.
pub(super) const CHUNK: Shape = Shape {
    message: "action chunk",
    fields: &[(
        6,
        Field::Message {
            max: 256,
            shape: &EMPTY,
        },
    )],
};

const JOIN: Shape = Shape {
    message: "join",
    fields: &[(
        4,
        Field::Text {
            max: 8,
            len: ADDRESS,
        },
    )],
};

const DEVICE_RECORD: Shape = Shape {
    message: "device record",
    fields: &[
        (4, Field::Text { max: 64, len: NAME }),
        (5, Field::Text { max: 64, len: NAME }),
    ],
};

const HOST_ORDER: Shape = Shape {
    message: "host order",
    fields: &[
        (
            4,
            Field::Message {
                max: 64,
                shape: &DEVICE_RECORD,
            },
        ),
        (
            5,
            Field::Message {
                max: 64,
                shape: &EMPTY,
            },
        ),
    ],
};

/// `Assignment`: a lobby's order.
pub(super) const ASSIGNMENT: Shape = Shape {
    message: "assignment",
    fields: &[
        (
            1,
            Field::Message {
                max: 1,
                shape: &JOIN,
            },
        ),
        (
            2,
            Field::Message {
                max: 1,
                shape: &HOST_ORDER,
            },
        ),
    ],
};

/// `RunState`: a run's members and addresses.
pub(super) const RUN_STATE: Shape = Shape {
    message: "run state",
    fields: &[
        (
            4,
            Field::Message {
                max: 64,
                shape: &EMPTY,
            },
        ),
        (
            7,
            Field::Text {
                max: 8,
                len: ADDRESS,
            },
        ),
    ],
};

const FIELD_SPEC: Shape = Shape {
    message: "field spec",
    fields: &[(3, Field::Scalars { max: 8, width: 0 })],
};

const FIELD_LIMIT: Shape = Shape {
    message: "field limit",
    fields: &[
        (2, Field::Scalars { max: 256, width: 8 }),
        (3, Field::Scalars { max: 256, width: 8 }),
    ],
};

const COMMAND: Shape = Shape {
    message: "command",
    fields: &[
        (
            2,
            Field::Message {
                max: 64,
                shape: &FIELD_SPEC,
            },
        ),
        (
            3,
            Field::Message {
                max: 64,
                shape: &FIELD_LIMIT,
            },
        ),
    ],
};

/// `PartSpec`.
pub(super) const PART: Shape = Shape {
    message: "part spec",
    fields: &[
        (
            5,
            Field::Message {
                max: 64,
                shape: &FIELD_SPEC,
            },
        ),
        (9, Field::Text { max: 64, len: NAME }),
        (
            13,
            Field::Message {
                max: 1,
                shape: &COMMAND,
            },
        ),
        (
            14,
            Field::Message {
                max: 32,
                shape: &EMPTY,
            },
        ),
        (18, Field::Text { max: 16, len: NAME }),
    ],
};

/// Decodes each `tag` field of `buf` with `f`, at most `max` of them and
/// each at least `min` bytes (what the smallest valid one encodes to, so
/// the count is bounded by the message's own length), into a vector
/// allocated once, fallibly, for exactly that many: what a repeated message
/// field becomes without the decoder's growing vector of its protobuf type.
///
/// # Errors
/// [`WireError::Unread`] over `max`, under `min` or out of memory; `f`'s.
pub(super) fn each<T>(
    buf: &[u8],
    message: &'static str,
    (tag, max, min): (u32, usize, usize),
    mut f: impl FnMut(&[u8]) -> Result<T, WireError>,
) -> Result<Vec<T>, WireError> {
    let mut n = 0;
    let mut fields = Fields::new(buf, message);
    while let Some((t, value)) = fields.next_field()? {
        if t == tag {
            if value.bytes(message)?.len() < min {
                return Err(WireError::unread(message, Unread::Malformed("too short")));
            }
            n += 1;
        }
    }
    if n > max {
        return Err(WireError::unread(message, Unread::TooLarge(message)));
    }
    let mut out = Vec::new();
    out.try_reserve_exact(n)
        .map_err(|_| WireError::unread(message, Unread::NoMemory))?;
    let mut fields = Fields::new(buf, message);
    while let Some((t, value)) = fields.next_field()? {
        if t == tag {
            out.push(f(value.bytes(message)?)?);
        }
    }
    Ok(out)
}

/// The `uint32` values of repeated field `tag` of `buf`, packed or not, at
/// most `max`, allocated as [`each`] does.
///
/// # Errors
/// [`WireError::Unread`].
pub(super) fn each_u32(
    buf: &[u8],
    message: &'static str,
    (tag, max): (u32, usize),
) -> Result<Vec<u32>, WireError> {
    let values = |value: Value<'_>, out: &mut dyn FnMut(u32)| -> Result<(), WireError> {
        match value {
            Value::Len(packed) => {
                let mut items = packed;
                while !items.is_empty() {
                    out(read_u32(&mut items, message)?);
                }
                Ok(())
            }
            other => {
                out(other.u32(message)?);
                Ok(())
            }
        }
    };
    let mut n = 0;
    let mut fields = Fields::new(buf, message);
    while let Some((t, value)) = fields.next_field()? {
        if t == tag {
            values(value, &mut |_| n += 1)?;
        }
    }
    if n > max {
        return Err(WireError::unread(message, Unread::TooLarge(message)));
    }
    let mut out = Vec::new();
    out.try_reserve_exact(n)
        .map_err(|_| WireError::unread(message, Unread::NoMemory))?;
    let mut fields = Fields::new(buf, message);
    while let Some((t, value)) = fields.next_field()? {
        if t == tag {
            values(value, &mut |v| out.push(v))?;
        }
    }
    Ok(out)
}

/// One varint of a packed field, truncated to `u32`.
#[allow(clippy::cast_possible_truncation)]
fn read_u32(items: &mut &[u8], message: &'static str) -> Result<u32, WireError> {
    let mut value = 0u64;
    for i in 0..10 {
        let Some((&byte, rest)) = items.split_first() else {
            break;
        };
        *items = rest;
        if i == 9 && byte > 1 {
            break;
        }
        value |= u64::from(byte & 0x7f) << (7 * i);
        if byte & 0x80 == 0 {
            return Ok(value as u32);
        }
    }
    Err(WireError::unread(
        message,
        Unread::Malformed("packed varint"),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message as _;
    use prost_types::field_descriptor_proto::{Label, Type};
    use prost_types::{DescriptorProto, FileDescriptorSet};

    fn messages() -> Vec<DescriptorProto> {
        let set =
            FileDescriptorSet::decode(super::super::FILE_DESCRIPTOR_SET).expect("descriptors");
        let mut all = Vec::new();
        let mut todo: Vec<DescriptorProto> = set
            .file
            .into_iter()
            .filter(|f| f.package() == "rlmesh.session.v2")
            .flat_map(|f| f.message_type)
            .collect();
        // Nested types too: a map field's entries.
        while let Some(mut message) = todo.pop() {
            todo.append(&mut message.nested_type);
            all.push(message);
        }
        all
    }

    fn message<'a>(all: &'a [DescriptorProto], name: &str) -> &'a DescriptorProto {
        let name = name.rsplit('.').next().expect("a name");
        all.iter()
            .find(|m| m.name() == name)
            .unwrap_or_else(|| panic!("no message {name}"))
    }

    /// Whether `name` holds a repeated field anywhere below it.
    fn grows(all: &[DescriptorProto], name: &str, depth: u8) -> bool {
        assert!(depth < 16, "{name} nests too deep");
        message(all, name).field.iter().any(|f| {
            f.label() == Label::Repeated
                || (f.r#type() == Type::Message && grows(all, f.type_name(), depth + 1))
        })
    }

    /// Every repeated field below `name` is capped by `shape` (`skipped`:
    /// top-level fields the caller reads with another decoder).
    fn covers(all: &[DescriptorProto], name: &str, shape: &Shape, skipped: &[i32]) {
        for field in &message(all, name).field {
            if skipped.contains(&field.number()) {
                continue;
            }
            let rule = shape
                .fields
                .iter()
                .find(|(t, _)| i32::try_from(*t).ok() == Some(field.number()))
                .map(|(_, r)| *r);
            let what = format!("{name}.{}", field.name());
            match (field.label(), field.r#type(), rule) {
                (_, Type::Message, Some(Field::Message { shape, .. })) => {
                    covers(all, field.type_name(), shape, &[]);
                }
                (Label::Repeated, Type::String | Type::Bytes, Some(Field::Text { .. })) => {}
                (Label::Repeated, Type::Message | Type::String | Type::Bytes, _) => {
                    panic!("{what} is repeated and uncapped");
                }
                (Label::Repeated, _, Some(Field::Scalars { width, .. })) => {
                    let fixed = match field.r#type() {
                        Type::Double | Type::Fixed64 | Type::Sfixed64 => 8,
                        Type::Float | Type::Fixed32 | Type::Sfixed32 => 4,
                        _ => 0,
                    };
                    assert_eq!(width, fixed, "{what}: packed width");
                }
                (Label::Repeated, _, _) => panic!("{what} is repeated and uncapped"),
                (_, Type::Message, None) => {
                    assert!(
                        !grows(all, field.type_name(), 0),
                        "{what} holds repeated fields and is uncapped"
                    );
                }
                (_, _, None) => {}
                (_, _, Some(rule)) => panic!("{what}: {rule:?} on a singular field"),
            }
        }
    }

    const CAPPED: Shape = Shape {
        message: "capped",
        fields: &[
            (1, Field::Text { max: 2, len: 4 }),
            (2, Field::Scalars { max: 3, width: 0 }),
            (3, Field::Scalars { max: 2, width: 8 }),
        ],
    };

    /// Field `tag`, length-delimited, holding `bytes`.
    fn delimited(tag: u8, bytes: &[u8]) -> Vec<u8> {
        let mut out = vec![tag << 3 | 2, u8::try_from(bytes.len()).expect("short")];
        out.extend_from_slice(bytes);
        out
    }

    #[test]
    fn caps_are_inclusive_for_texts_and_packed_or_unpacked_scalars() {
        let ok = |buf: Vec<u8>| check(&buf, &CAPPED).is_ok();
        assert!(ok([delimited(1, b"abcd"), delimited(1, b"wxyz")].concat()));
        assert!(!ok(delimited(1, b"abcde")), "a text over its length");
        assert!(!ok([b"a", b"b", b"c"].map(|t| delimited(1, t)).concat()));
        assert!(
            ok(delimited(2, &[1, 0xac, 0x02, 3])),
            "three packed varints"
        );
        assert!(!ok(delimited(2, &[1, 2, 3, 4])));
        assert!(ok(vec![0x10, 1, 0x10, 2, 0x10, 3]), "three unpacked");
        assert!(!ok(vec![0x10, 1, 0x10, 2, 0x10, 3, 0x10, 4]));
        assert!(ok(delimited(3, &[0; 16])), "two packed doubles");
        assert!(!ok(delimited(3, &[0; 24])));
    }

    #[test]
    fn repeated_fields_take_up_to_their_cap_and_packed_varints_decode_in_full() {
        let two = [
            delimited(1, b"ab"),
            delimited(1, b"cd"),
            delimited(4, b"other"),
        ]
        .concat();
        let lens = |max| each(&two, "t", (1, max, 0), |b| Ok(b.len()));
        assert_eq!(lens(2).expect("at the cap"), [2, 2]);
        assert!(lens(1).is_err());
        // 1, 300, 70000 and u32::MAX, packed beside an unrelated field.
        let packed = [
            delimited(1, b"x"),
            delimited(
                2,
                &[
                    1, 0xac, 0x02, 0xf0, 0xa2, 0x04, 0xff, 0xff, 0xff, 0xff, 0x0f,
                ],
            ),
        ]
        .concat();
        assert_eq!(
            each_u32(&packed, "t", (2, 4)).expect("at the cap"),
            [1, 300, 70_000, u32::MAX]
        );
        assert!(each_u32(&packed, "t", (2, 3)).is_err());
        // A ten-byte varint (a sign-extending encoder's) truncates to u32;
        // one past 64 bits is malformed.
        let ten = |last: u8| delimited(2, &[[0xff; 9].as_slice(), &[last]].concat());
        assert_eq!(
            each_u32(&ten(0x01), "t", (2, 1)).expect("ten bytes"),
            [u32::MAX]
        );
        assert!(each_u32(&ten(0x02), "t", (2, 1)).is_err());
    }

    #[test]
    fn every_repeated_field_a_board_decodes_is_capped() {
        let all = messages();
        covers(&all, "Key", &KEY, &[]);
        covers(&all, "ActionChunk", &CHUNK, &[]);
        covers(&all, "PartSpec", &PART, &[]);
        covers(&all, "Assignment", &ASSIGNMENT, &[]);
        covers(&all, "RunState", &RUN_STATE, &[]);
        // Decoded element by element, each capped: the view, a reply's
        // leases and keys, a delivery's keys.
        for unbounded in [
            "HeartbeatEcho",
            "EstopSignal",
            "HandshakeReply",
            "Lease",
            "Reason",
        ] {
            assert!(!grows(&all, unbounded, 0), "{unbounded} needs a shape");
        }
    }

    #[test]
    fn a_count_or_a_length_over_its_cap_is_refused_unread() {
        let refused =
            |buf: &[u8], shape: &Shape| check(buf, shape).err().and_then(|e| e.why_unread());
        // Two thousand empty inputs: four kilobytes on the wire, 48 KB once
        // decoded on a 32-bit board.
        let hostile: Vec<u8> = [0x32, 0x00].repeat(2000);
        assert_eq!(
            refused(&hostile, &CHUNK),
            Some(Unread::TooLarge("action chunk"))
        );
        assert!(check(&hostile[..512], &CHUNK).is_ok());
        // A name longer than any id, repeated.
        let mut part = Vec::new();
        for _ in 0..2 {
            part.extend([0x4a, 0x81, 0x01]);
            part.extend([b'x'; 129]);
        }
        assert_eq!(refused(&part, &PART), Some(Unread::TooLarge("part spec")));
        // A key covering 257 parts, packed.
        let mut key = vec![0x2a, 0x81, 0x02];
        key.extend([1; 257]);
        assert_eq!(refused(&key, &KEY), Some(Unread::TooLarge("key")));
        assert!(
            check(&key[..key.len() - 1], &KEY).is_err(),
            "a length past the end"
        );
    }
}
