//! The messages (spec §4): typed views over a frame's payload, and their
//! encoding. Lengths are exact; reserved bits must be zero.
//!
//! `STATE` and `COMMAND` carry a part's dense value, whose length only the
//! receiver knows (from the map), so [`decode`] takes it as a lookup.

use crate::frame::{self, Frame};

/// `HELLO`'s magic.
pub const MAGIC: [u8; 4] = *b"RLMT";
/// The protocol version this crate speaks.
pub const VERSION: u8 = 1;
/// The largest `MAP` slice.
pub const MAX_MAP_SLICE: usize = 248;
/// The longest board name.
pub const MAX_NAME: usize = 32;
/// The longest `NOTE` text.
pub const MAX_NOTE: usize = 48;
/// `part` meaning every part (`SAFE`).
pub const EVERY_PART: u8 = 0xFF;

/// Message types.
pub mod ty {
    /// Board → bridge: identification.
    pub const HELLO: u8 = 0x01;
    /// Board → bridge: a slice of the map.
    pub const MAP: u8 = 0x02;
    /// Board → bridge: one part's sample.
    pub const STATE: u8 = 0x03;
    /// Board → bridge: liveness and counters.
    pub const STATUS: u8 = 0x04;
    /// Board → bridge: a log line.
    pub const NOTE: u8 = 0x05;
    /// Bridge → board: read the map.
    pub const GET_MAP: u8 = 0x81;
    /// Bridge → board: start a generation.
    pub const ATTACH: u8 = 0x82;
    /// Bridge → board: a drive permit with its target.
    pub const COMMAND: u8 = 0x83;
    /// Bridge → board: run the safe behaviour.
    pub const SAFE: u8 = 0x84;
    /// Bridge → board: keep the link alive.
    pub const PING: u8 = 0x85;
    /// Bridge → board: end the generation.
    pub const DETACH: u8 = 0x86;
}

/// `HELLO` flags.
pub mod hello_flags {
    /// The e-stop pin latch is set.
    pub const ESTOP: u8 = 1 << 0;
    /// Every defined bit.
    pub const ALL: u8 = ESTOP;
}

/// `STATUS` flags.
pub mod status_flags {
    /// Attached.
    pub const ATTACHED: u8 = 1 << 0;
    /// E-stop latched by the board's pin.
    pub const ESTOP_LATCHED: u8 = 1 << 1;
    /// E-stop input active.
    pub const ESTOP_INPUT: u8 = 1 << 2;
    /// Every defined bit.
    pub const ALL: u8 = ATTACHED | ESTOP_LATCHED | ESTOP_INPUT;
}

/// `ATTACH` flags.
pub mod attach_flags {
    /// Clear the pin e-stop latch (with the input released).
    pub const CLEAR_ESTOP: u8 = 1 << 0;
    /// Every defined bit.
    pub const ALL: u8 = CLEAR_ESTOP;
}

/// `COMMAND` flags.
pub mod command_flags {
    /// A repeat of the last target (edge Hold).
    pub const HOLD: u8 = 1 << 0;
    /// Every defined bit.
    pub const ALL: u8 = HOLD;
}

/// The board's identification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hello<'a> {
    /// The library's version.
    pub lib_version: u16,
    /// 0 unknown, 1 AVR, 2 ARM Cortex-M, 3 RISC-V, 4 Xtensa.
    pub arch: u8,
    /// From the generated header.
    pub board_id: [u8; 8],
    /// Differs between boots.
    pub boot: u16,
    /// The last attach generation since boot, 0 if none.
    pub gen_: u8,
    /// [`hello_flags`].
    pub flags: u8,
    /// The map's length in bytes.
    pub map_len: u16,
    /// CRC-32/ISO-HDLC of the map.
    pub map_crc: u32,
    /// The link watchdog.
    pub watchdog_ms: u16,
    /// The largest decoded frame the board accepts.
    pub rx_max: u16,
    /// The default robot name, UTF-8.
    pub name: &'a [u8],
}

/// The board's liveness report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Status {
    /// The current boot.
    pub boot: u16,
    /// The current attach generation.
    pub gen_: u8,
    /// [`status_flags`].
    pub flags: u8,
    /// The last reset cause: 0 power-on, 1 external, 2 brown-out, 3
    /// watchdog, 4 other.
    pub reset: u8,
    /// Frames lost, from `seq` gaps.
    pub lost: u16,
    /// Frames dropped as invalid.
    pub bad: u16,
    /// `micros()` just before transmit.
    pub t_us: u32,
}

/// One message, borrowing its variable parts from the frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Msg<'a> {
    /// `HELLO`.
    Hello(Hello<'a>),
    /// `MAP`: `data` is the map from `offset`.
    Map {
        /// The boot the request named.
        boot: u16,
        /// Where `data` starts.
        offset: u16,
        /// At most [`MAX_MAP_SLICE`] bytes; empty past the end.
        data: &'a [u8],
    },
    /// `STATE`.
    State {
        /// Part index.
        part: u8,
        /// `micros()` at capture.
        t_us: u32,
        /// The dense sample.
        value: &'a [u8],
    },
    /// `STATUS`.
    Status(Status),
    /// `NOTE`.
    Note {
        /// 0 info, 1 warn, 2 error.
        level: u8,
        /// Library or sketch defined.
        code: u8,
        /// At most [`MAX_NOTE`] bytes.
        text: &'a [u8],
    },
    /// `GET_MAP`.
    GetMap {
        /// The boot of the `HELLO` being answered.
        boot: u16,
        /// Where to read from.
        offset: u16,
        /// 1..=[`MAX_MAP_SLICE`].
        max: u8,
    },
    /// `ATTACH`.
    Attach {
        /// The board's id, echoed.
        board_id: [u8; 8],
        /// The boot being attached.
        boot: u16,
        /// The new generation.
        gen_: u8,
        /// [`attach_flags`].
        flags: u8,
    },
    /// `COMMAND`.
    Command {
        /// The generation it belongs to.
        gen_: u8,
        /// Part index.
        part: u8,
        /// [`command_flags`].
        flags: u8,
        /// The permit's life from receipt.
        ttl_ms: u16,
        /// The dense command.
        value: &'a [u8],
    },
    /// `SAFE`.
    Safe {
        /// The generation it belongs to.
        gen_: u8,
        /// Part index, or [`EVERY_PART`].
        part: u8,
    },
    /// `PING`.
    Ping {
        /// The generation it belongs to.
        gen_: u8,
    },
    /// `DETACH`.
    Detach {
        /// The generation it belongs to.
        gen_: u8,
    },
}

/// Why a frame's payload isn't a valid message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MsgError {
    /// An unknown `type`.
    UnknownType,
    /// A `HELLO` without the magic.
    Magic,
    /// A `HELLO` of another protocol version (checked before its length).
    Version(u8),
    /// Not the message's exact length.
    Length,
    /// A reserved bit is set.
    Reserved,
    /// A field out of range (`max` 0, a name or note too long, a slice too
    /// long).
    Range,
    /// `STATE` or `COMMAND` for a part the receiver doesn't know.
    UnknownPart,
    /// A message type that travels the other way (§3.2): a bridge hearing a
    /// bridge message (an echoing adapter), or a board hearing a board one.
    Direction,
    /// `HELLO.name` isn't UTF-8.
    Utf8,
}

/// Which end is receiving: it decides which message types are valid, and
/// what `dense_len` looks up (sample lengths on the bridge, command lengths
/// on the board).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Receiver {
    /// The computer: accepts board → bridge types (below 0x80).
    Bridge,
    /// The board: accepts bridge → board types (0x80 and up).
    Board,
}

/// `HELLO.rx_max` bounds: room for `ATTACH` (16 bytes decoded), at most a
/// whole frame.
pub const RX_MAX: core::ops::RangeInclusive<u16> = 16..=256;
/// `HELLO.watchdog_ms` bounds (§6.2).
pub const WATCHDOG_MS: core::ops::RangeInclusive<u16> = 20..=5000;

fn u16_at(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[i], b[i + 1]])
}

fn u32_at(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
}

fn exact(p: &[u8], n: usize) -> Result<(), MsgError> {
    if p.len() == n {
        Ok(())
    } else {
        Err(MsgError::Length)
    }
}

/// `head` fixed bytes followed by exactly `len` value bytes, without
/// overflow however large `len` is.
fn head_and(p: &[u8], head: usize, len: usize) -> Result<(), MsgError> {
    if p.len().checked_sub(head) == Some(len) {
        Ok(())
    } else {
        Err(MsgError::Length)
    }
}

fn no_reserved(flags: u8, all: u8) -> Result<(), MsgError> {
    if flags & !all == 0 {
        Ok(())
    } else {
        Err(MsgError::Reserved)
    }
}

/// Decodes `frame`'s payload as received by `receiver`. `dense_len(part)`
/// gives a part's dense sample length on the bridge (`STATE`) and its dense
/// command length on the board (`COMMAND`).
pub fn decode<'a>(
    frame: &Frame<'a>,
    receiver: Receiver,
    dense_len: impl Fn(u8) -> Option<usize>,
) -> Result<Msg<'a>, MsgError> {
    let to_board = frame.ty & 0x80 != 0;
    if to_board != (receiver == Receiver::Board) {
        return Err(MsgError::Direction);
    }
    let p = frame.payload;
    match frame.ty {
        ty::HELLO => {
            if p.len() < 4 || p[..4] != MAGIC {
                return Err(MsgError::Magic);
            }
            if p.len() < 5 {
                return Err(MsgError::Length);
            }
            if p[4] != VERSION {
                return Err(MsgError::Version(p[4]));
            }
            if p.len() < 31 {
                return Err(MsgError::Length);
            }
            let name_len = usize::from(p[30]);
            if name_len > MAX_NAME {
                return Err(MsgError::Range);
            }
            exact(p, 31 + name_len)?;
            let flags = p[19];
            no_reserved(flags, hello_flags::ALL)?;
            let (watchdog_ms, rx_max) = (u16_at(p, 26), u16_at(p, 28));
            if !WATCHDOG_MS.contains(&watchdog_ms) || !RX_MAX.contains(&rx_max) {
                return Err(MsgError::Range);
            }
            let name = &p[31..];
            if core::str::from_utf8(name).is_err() {
                return Err(MsgError::Utf8);
            }
            let mut board_id = [0; 8];
            board_id.copy_from_slice(&p[8..16]);
            Ok(Msg::Hello(Hello {
                lib_version: u16_at(p, 5),
                arch: p[7],
                board_id,
                boot: u16_at(p, 16),
                gen_: p[18],
                flags,
                map_len: u16_at(p, 20),
                map_crc: u32_at(p, 22),
                watchdog_ms,
                rx_max,
                name,
            }))
        }
        ty::MAP => {
            if p.len() < 4 || p.len() > 4 + MAX_MAP_SLICE {
                return Err(MsgError::Length);
            }
            Ok(Msg::Map {
                boot: u16_at(p, 0),
                offset: u16_at(p, 2),
                data: &p[4..],
            })
        }
        ty::STATE => {
            let part = *p.first().ok_or(MsgError::Length)?;
            let len = dense_len(part).ok_or(MsgError::UnknownPart)?;
            head_and(p, 5, len)?;
            Ok(Msg::State {
                part,
                t_us: u32_at(p, 1),
                value: &p[5..],
            })
        }
        ty::STATUS => {
            exact(p, 13)?;
            no_reserved(p[3], status_flags::ALL)?;
            Ok(Msg::Status(Status {
                boot: u16_at(p, 0),
                gen_: p[2],
                flags: p[3],
                reset: p[4],
                lost: u16_at(p, 5),
                bad: u16_at(p, 7),
                t_us: u32_at(p, 9),
            }))
        }
        ty::NOTE => {
            if p.len() < 2 || p.len() > 2 + MAX_NOTE {
                return Err(MsgError::Length);
            }
            Ok(Msg::Note {
                level: p[0],
                code: p[1],
                text: &p[2..],
            })
        }
        ty::GET_MAP => {
            exact(p, 5)?;
            let max = p[4];
            if max == 0 || usize::from(max) > MAX_MAP_SLICE {
                return Err(MsgError::Range);
            }
            Ok(Msg::GetMap {
                boot: u16_at(p, 0),
                offset: u16_at(p, 2),
                max,
            })
        }
        ty::ATTACH => {
            exact(p, 12)?;
            no_reserved(p[11], attach_flags::ALL)?;
            let mut board_id = [0; 8];
            board_id.copy_from_slice(&p[..8]);
            Ok(Msg::Attach {
                board_id,
                boot: u16_at(p, 8),
                gen_: p[10],
                flags: p[11],
            })
        }
        ty::COMMAND => {
            if p.len() < 5 {
                return Err(MsgError::Length);
            }
            let part = p[1];
            let len = dense_len(part).ok_or(MsgError::UnknownPart)?;
            head_and(p, 5, len)?;
            no_reserved(p[2], command_flags::ALL)?;
            Ok(Msg::Command {
                gen_: p[0],
                part,
                flags: p[2],
                ttl_ms: u16_at(p, 3),
                value: &p[5..],
            })
        }
        ty::SAFE => {
            exact(p, 2)?;
            Ok(Msg::Safe {
                gen_: p[0],
                part: p[1],
            })
        }
        ty::PING => {
            exact(p, 1)?;
            Ok(Msg::Ping { gen_: p[0] })
        }
        ty::DETACH => {
            exact(p, 1)?;
            Ok(Msg::Detach { gen_: p[0] })
        }
        _ => Err(MsgError::UnknownType),
    }
}

/// Writes `msg` as one frame with `seq` to `out` (COBS-encoded, with its
/// delimiter). Nothing is written when a field is out of range or the
/// payload doesn't fit a frame; that's the error.
pub fn write(msg: &Msg<'_>, seq: u8, out: impl FnMut(u8)) -> Result<(), MsgError> {
    let mut head = [0_u8; 31];
    let (t, n, tail): (u8, usize, &[u8]) = match *msg {
        Msg::Hello(ref h) => {
            if h.name.len() > MAX_NAME
                || !WATCHDOG_MS.contains(&h.watchdog_ms)
                || !RX_MAX.contains(&h.rx_max)
            {
                return Err(MsgError::Range);
            }
            if core::str::from_utf8(h.name).is_err() {
                return Err(MsgError::Utf8);
            }
            no_reserved(h.flags, hello_flags::ALL)?;
            head[..4].copy_from_slice(&MAGIC);
            head[4] = VERSION;
            head[5..7].copy_from_slice(&h.lib_version.to_le_bytes());
            head[7] = h.arch;
            head[8..16].copy_from_slice(&h.board_id);
            head[16..18].copy_from_slice(&h.boot.to_le_bytes());
            head[18] = h.gen_;
            head[19] = h.flags;
            head[20..22].copy_from_slice(&h.map_len.to_le_bytes());
            head[22..26].copy_from_slice(&h.map_crc.to_le_bytes());
            head[26..28].copy_from_slice(&h.watchdog_ms.to_le_bytes());
            head[28..30].copy_from_slice(&h.rx_max.to_le_bytes());
            // A name is at most MAX_NAME (32) bytes, checked above.
            head[30] = h.name.len() as u8;
            (ty::HELLO, 31, h.name)
        }
        Msg::Map { boot, offset, data } => {
            if data.len() > MAX_MAP_SLICE {
                return Err(MsgError::Length);
            }
            head[..2].copy_from_slice(&boot.to_le_bytes());
            head[2..4].copy_from_slice(&offset.to_le_bytes());
            (ty::MAP, 4, data)
        }
        Msg::State { part, t_us, value } => {
            head[0] = part;
            head[1..5].copy_from_slice(&t_us.to_le_bytes());
            (ty::STATE, 5, value)
        }
        Msg::Status(ref s) => {
            no_reserved(s.flags, status_flags::ALL)?;
            head[..2].copy_from_slice(&s.boot.to_le_bytes());
            head[2] = s.gen_;
            head[3] = s.flags;
            head[4] = s.reset;
            head[5..7].copy_from_slice(&s.lost.to_le_bytes());
            head[7..9].copy_from_slice(&s.bad.to_le_bytes());
            head[9..13].copy_from_slice(&s.t_us.to_le_bytes());
            (ty::STATUS, 13, &[])
        }
        Msg::Note { level, code, text } => {
            if text.len() > MAX_NOTE {
                return Err(MsgError::Length);
            }
            head[0] = level;
            head[1] = code;
            (ty::NOTE, 2, text)
        }
        Msg::GetMap { boot, offset, max } => {
            if max == 0 || usize::from(max) > MAX_MAP_SLICE {
                return Err(MsgError::Range);
            }
            head[..2].copy_from_slice(&boot.to_le_bytes());
            head[2..4].copy_from_slice(&offset.to_le_bytes());
            head[4] = max;
            (ty::GET_MAP, 5, &[])
        }
        Msg::Attach {
            board_id,
            boot,
            gen_,
            flags,
        } => {
            no_reserved(flags, attach_flags::ALL)?;
            head[..8].copy_from_slice(&board_id);
            head[8..10].copy_from_slice(&boot.to_le_bytes());
            head[10] = gen_;
            head[11] = flags;
            (ty::ATTACH, 12, &[])
        }
        Msg::Command {
            gen_,
            part,
            flags,
            ttl_ms,
            value,
        } => {
            no_reserved(flags, command_flags::ALL)?;
            head[0] = gen_;
            head[1] = part;
            head[2] = flags;
            head[3..5].copy_from_slice(&ttl_ms.to_le_bytes());
            (ty::COMMAND, 5, value)
        }
        Msg::Safe { gen_, part } => {
            head[0] = gen_;
            head[1] = part;
            (ty::SAFE, 2, &[])
        }
        Msg::Ping { gen_ } => {
            head[0] = gen_;
            (ty::PING, 1, &[])
        }
        Msg::Detach { gen_ } => {
            head[0] = gen_;
            (ty::DETACH, 1, &[])
        }
    };
    if n + tail.len() > frame::MAX_PAYLOAD {
        return Err(MsgError::Length);
    }
    frame::write(t, seq, &[&head[..n], tail], out);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cobs::Decoder;
    use crate::frame::MAX_DECODED;

    fn round_trip(msg: Msg<'_>, dense: Option<usize>) {
        let mut wire = [0_u8; 300];
        let mut n = 0;
        write(&msg, 9, |b| {
            wire[n] = b;
            n += 1;
        })
        .expect("writes");
        let mut d = Decoder::<MAX_DECODED>::new();
        let mut seen = false;
        for &b in &wire[..n] {
            if let Some(decoded) = d.push(b) {
                let f = Frame::parse(decoded.expect("decodes")).expect("frame");
                assert_eq!(f.seq, 9);
                let to = if f.ty & 0x80 == 0 {
                    Receiver::Bridge
                } else {
                    Receiver::Board
                };
                assert_eq!(decode(&f, to, |_| dense), Ok(msg));
                seen = true;
            }
        }
        assert!(seen);
    }

    #[test]
    fn every_message_round_trips() {
        let hello = Hello {
            lib_version: 0x0102,
            arch: 1,
            board_id: [1, 2, 3, 4, 5, 6, 7, 8],
            boot: 0xBEEF,
            gen_: 7,
            flags: hello_flags::ESTOP,
            map_len: 600,
            map_crc: 0xDEAD_BEEF,
            watchdog_ms: 250,
            rx_max: 64,
            name: b"eye-1",
        };
        round_trip(Msg::Hello(hello), None);
        round_trip(Msg::Hello(Hello { name: b"", ..hello }), None);
        round_trip(
            Msg::Map {
                boot: 3,
                offset: 248,
                data: &[b'{'; MAX_MAP_SLICE],
            },
            None,
        );
        round_trip(
            Msg::Map {
                boot: 3,
                offset: 600,
                data: &[],
            },
            None,
        );
        round_trip(
            Msg::State {
                part: 1,
                t_us: u32::MAX,
                value: &[0, 0, 0x80, 0x3F],
            },
            Some(4),
        );
        round_trip(
            Msg::Status(Status {
                boot: 1,
                gen_: 2,
                flags: status_flags::ALL,
                reset: 3,
                lost: 4,
                bad: 5,
                t_us: 6,
            }),
            None,
        );
        round_trip(
            Msg::Note {
                level: 2,
                code: 9,
                text: b"loop() too slow",
            },
            None,
        );
        round_trip(
            Msg::GetMap {
                boot: 3,
                offset: 0,
                max: 248,
            },
            None,
        );
        round_trip(
            Msg::Attach {
                board_id: [9; 8],
                boot: 3,
                gen_: 255,
                flags: attach_flags::CLEAR_ESTOP,
            },
            None,
        );
        round_trip(
            Msg::Command {
                gen_: 1,
                part: 0,
                flags: command_flags::HOLD,
                ttl_ms: 60,
                value: &[0; 8],
            },
            Some(8),
        );
        round_trip(
            Msg::Safe {
                gen_: 1,
                part: EVERY_PART,
            },
            None,
        );
        round_trip(Msg::Ping { gen_: 1 }, None);
        round_trip(Msg::Detach { gen_: 1 }, None);
    }

    #[test]
    fn the_spec_lengths_hold() {
        // §4: HELLO 31 + name, STATUS 13, ATTACH 12, GET_MAP 5, SAFE 2,
        // PING/DETACH 1; COMMAND and STATE frames are dense + 9.
        let len = |msg: Msg<'_>| {
            let mut n = 0;
            let mut dec = Decoder::<MAX_DECODED>::new();
            let mut payload = 0;
            write(&msg, 0, |b| {
                n += 1;
                if let Some(Ok(f)) = dec.push(b) {
                    payload = f.len() - 4;
                }
            })
            .expect("writes");
            payload
        };
        let hello = Hello {
            lib_version: 1,
            arch: 0,
            board_id: [0; 8],
            boot: 0,
            gen_: 0,
            flags: 0,
            map_len: 0,
            map_crc: 0,
            watchdog_ms: 250,
            rx_max: 64,
            name: b"abc",
        };
        assert_eq!(len(Msg::Hello(hello)), 34);
        assert_eq!(
            len(Msg::Status(Status {
                boot: 0,
                gen_: 0,
                flags: 0,
                reset: 0,
                lost: 0,
                bad: 0,
                t_us: 0
            })),
            13
        );
        assert_eq!(
            len(Msg::Attach {
                board_id: [0; 8],
                boot: 0,
                gen_: 1,
                flags: 0
            }),
            12
        );
        assert_eq!(
            len(Msg::GetMap {
                boot: 0,
                offset: 0,
                max: 1
            }),
            5
        );
        assert_eq!(len(Msg::Safe { gen_: 1, part: 0 }), 2);
        assert_eq!(len(Msg::Ping { gen_: 1 }), 1);
        assert_eq!(
            len(Msg::Command {
                gen_: 1,
                part: 0,
                flags: 0,
                ttl_ms: 1,
                value: &[0; 8]
            }) + 4,
            8 + 9
        );
        assert_eq!(
            len(Msg::State {
                part: 0,
                t_us: 0,
                value: &[0; 8]
            }) + 4,
            8 + 9
        );
    }

    #[test]
    fn hello_fields_sit_at_their_spec_offsets() {
        let hello = Hello {
            lib_version: 0x0201,
            arch: 3,
            board_id: [0xB0, 0xB1, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7],
            boot: 0x1716,
            gen_: 0x18,
            flags: hello_flags::ESTOP,
            map_len: 0x2120,
            map_crc: 0x2524_2322,
            watchdog_ms: 0x1327,
            rx_max: 0x0100,
            name: b"n",
        };
        let mut dec = Decoder::<MAX_DECODED>::new();
        let mut payload = [0_u8; 32];
        write(&Msg::Hello(hello), 0, |b| {
            if let Some(Ok(f)) = dec.push(b) {
                payload.copy_from_slice(&f[2..f.len() - 2]);
            }
        })
        .expect("writes");
        let want: [u8; 32] = [
            b'R', b'L', b'M', b'T', 1, 0x01, 0x02, 3, 0xB0, 0xB1, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
            0xB7, 0x16, 0x17, 0x18, 1, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x27, 0x13, 0x00, 0x01,
            1, b'n',
        ];
        assert_eq!(payload, want);
    }

    /// Decodes as the end a frame of this type is meant for.
    fn decode_any<'a>(
        f: &Frame<'a>,
        dense: impl Fn(u8) -> Option<usize>,
    ) -> Result<Msg<'a>, MsgError> {
        let to = if f.ty & 0x80 == 0 {
            Receiver::Bridge
        } else {
            Receiver::Board
        };
        decode(f, to, dense)
    }

    #[test]
    fn a_frame_for_the_other_end_is_refused() {
        // An echoing adapter: the bridge hears its own COMMAND.
        let cmd = [1_u8, 0, 0, 60, 0, 0, 0, 0, 0];
        let f = frame(ty::COMMAND, &cmd);
        assert_eq!(
            decode(&f, Receiver::Bridge, |_| Some(4)),
            Err(MsgError::Direction)
        );
        assert_eq!(
            decode(&frame(ty::STATUS, &[0; 13]), Receiver::Board, |_| None),
            Err(MsgError::Direction)
        );
    }

    #[test]
    fn hello_ranges_and_name_are_checked() {
        let mut hello = [0_u8; 32];
        hello[..5].copy_from_slice(b"RLMT\x01");
        hello[26..28].copy_from_slice(&250_u16.to_le_bytes());
        hello[28..30].copy_from_slice(&64_u16.to_le_bytes());
        hello[30] = 1;
        hello[31] = b'a';
        let none = |_| None;
        assert!(decode_any(&frame(ty::HELLO, &hello), none).is_ok());
        hello[28..30].copy_from_slice(&8_u16.to_le_bytes()); // rx_max under 16
        assert_eq!(
            decode_any(&frame(ty::HELLO, &hello), none),
            Err(MsgError::Range)
        );
        hello[28..30].copy_from_slice(&64_u16.to_le_bytes());
        hello[26..28].copy_from_slice(&0_u16.to_le_bytes()); // watchdog 0
        assert_eq!(
            decode_any(&frame(ty::HELLO, &hello), none),
            Err(MsgError::Range)
        );
        hello[26..28].copy_from_slice(&250_u16.to_le_bytes());
        hello[31] = 0xFF; // not UTF-8
        assert_eq!(
            decode_any(&frame(ty::HELLO, &hello), none),
            Err(MsgError::Utf8)
        );
        // The magic alone is a short HELLO, not a bad magic.
        assert_eq!(
            decode_any(&frame(ty::HELLO, b"RLMT"), none),
            Err(MsgError::Length)
        );
    }

    #[test]
    fn a_huge_dense_length_is_a_length_error_not_a_panic() {
        let huge = |_| Some(usize::MAX - 2);
        assert_eq!(
            decode_any(&frame(ty::STATE, &[0; 7]), huge),
            Err(MsgError::Length)
        );
        assert_eq!(
            decode_any(&frame(ty::COMMAND, &[0; 7]), huge),
            Err(MsgError::Length)
        );
    }

    #[test]
    fn write_refuses_what_decode_would() {
        let mut wrote = false;
        let sink = |_| wrote = true;
        let big = [0_u8; 248];
        assert_eq!(
            write(
                &Msg::State {
                    part: 0,
                    t_us: 0,
                    value: &big
                },
                0,
                sink
            ),
            Err(MsgError::Length),
            "5 + 248 is over a frame"
        );
        let mut wrote2 = false;
        assert_eq!(
            write(
                &Msg::GetMap {
                    boot: 0,
                    offset: 0,
                    max: 0
                },
                0,
                |_| wrote2 = true
            ),
            Err(MsgError::Range)
        );
        let mut wrote3 = false;
        let hello = Hello {
            lib_version: 1,
            arch: 0,
            board_id: [0; 8],
            boot: 0,
            gen_: 0,
            flags: 0,
            map_len: 0,
            map_crc: 0,
            watchdog_ms: 250,
            rx_max: 64,
            name: &[0xFF],
        };
        assert_eq!(
            write(&Msg::Hello(hello), 0, |_| wrote3 = true),
            Err(MsgError::Utf8)
        );
        assert!(
            !wrote && !wrote2 && !wrote3,
            "nothing is written on an error"
        );
    }

    fn frame(ty: u8, payload: &[u8]) -> Frame<'_> {
        Frame {
            ty,
            seq: 0,
            payload,
        }
    }

    #[test]
    fn malformed_payloads_are_refused() {
        let none = |_| None;
        let eight = |_| Some(8);
        assert_eq!(
            decode_any(&frame(0x07, &[0]), none),
            Err(MsgError::UnknownType)
        );
        assert_eq!(
            decode_any(&frame(ty::HELLO, b"RLMX\x01"), none),
            Err(MsgError::Magic)
        );
        // A future version is reported as such, whatever its length.
        assert_eq!(
            decode_any(&frame(ty::HELLO, b"RLMT\x02xyz"), none),
            Err(MsgError::Version(2))
        );
        assert_eq!(
            decode_any(&frame(ty::HELLO, b"RLMT\x01"), none),
            Err(MsgError::Length)
        );
        let mut hello = [0_u8; 31];
        hello[..5].copy_from_slice(b"RLMT\x01");
        hello[19] = 0b10; // reserved flag
        assert_eq!(
            decode_any(&frame(ty::HELLO, &hello), none),
            Err(MsgError::Reserved)
        );
        hello[19] = 0;
        hello[30] = 33; // name too long
        assert_eq!(
            decode_any(&frame(ty::HELLO, &hello), none),
            Err(MsgError::Range)
        );
        hello[30] = 1; // claims a name, has none
        assert_eq!(
            decode_any(&frame(ty::HELLO, &hello), none),
            Err(MsgError::Length)
        );
        assert_eq!(
            decode_any(&frame(ty::STATUS, &[0; 12]), none),
            Err(MsgError::Length)
        );
        let mut status = [0_u8; 13];
        status[3] = 0b1000;
        assert_eq!(
            decode_any(&frame(ty::STATUS, &status), none),
            Err(MsgError::Reserved)
        );
        assert_eq!(
            decode_any(&frame(ty::GET_MAP, &[0, 0, 0, 0, 0]), none),
            Err(MsgError::Range)
        );
        assert_eq!(
            decode_any(&frame(ty::GET_MAP, &[0, 0, 0, 0, 249]), none),
            Err(MsgError::Range)
        );
        assert_eq!(
            decode_any(&frame(ty::MAP, &[0; 253]), none),
            Err(MsgError::Length)
        );
        assert_eq!(
            decode_any(&frame(ty::NOTE, &[0; 51]), none),
            Err(MsgError::Length)
        );
        // COMMAND: the exact dense length, a known part, no reserved flags.
        assert_eq!(
            decode_any(&frame(ty::COMMAND, &[1, 0, 0, 60, 0]), none),
            Err(MsgError::UnknownPart)
        );
        assert_eq!(
            decode_any(&frame(ty::COMMAND, &[1, 0, 0, 60, 0, 1, 2]), eight),
            Err(MsgError::Length)
        );
        let mut cmd = [0_u8; 13];
        cmd[2] = 0b10;
        assert_eq!(
            decode_any(&frame(ty::COMMAND, &cmd), eight),
            Err(MsgError::Reserved)
        );
        assert_eq!(
            decode_any(&frame(ty::STATE, &[0; 12]), eight),
            Err(MsgError::Length)
        );
        assert_eq!(
            decode_any(&frame(ty::ATTACH, &[0; 11]), none),
            Err(MsgError::Length)
        );
        let mut attach = [0_u8; 12];
        attach[11] = 0b10;
        assert_eq!(
            decode_any(&frame(ty::ATTACH, &attach), none),
            Err(MsgError::Reserved)
        );
        assert_eq!(
            decode_any(&frame(ty::SAFE, &[1]), none),
            Err(MsgError::Length)
        );
        assert_eq!(
            decode_any(&frame(ty::PING, &[]), none),
            Err(MsgError::Length)
        );
    }
}
