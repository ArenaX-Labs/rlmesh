//! The conformance vectors of spec §10 (`tests/vectors.json`, schema in
//! `tests/README.md`): this crate must reproduce them byte for byte, and
//! every implementation of the tether (the Arduino library's host runner
//! too) runs the same file.
//!
//! Regenerate with `RLMESH_REGENERATE_TETHER_VECTORS=1 cargo test -p
//! rlmesh-tether --test vectors`, then `mise run fmt:config` (only when the
//! contract changes: the board library checks against this file).

use std::path::PathBuf;

use rlmesh_tether::cobs::{self, DecodeError, Decoder};
use rlmesh_tether::frame::{Frame, FrameError};
use rlmesh_tether::msg::{
    self, EVERY_PART, Hello, MAX_MAP_SLICE, MAX_NAME, MAX_NOTE, Msg, MsgError, Receiver, Status,
    attach_flags, command_flags, hello_flags, status_flags, ty,
};
use rlmesh_tether::provision::{
    self, Addresses, DeviceInfo, MAX_ADDRESS, MAX_ID, Prov, Provision, device_flags, status,
};
use rlmesh_tether::{crc, pack};
use serde_json::{Map, Value, json};

const REGENERATE: &str = "RLMESH_REGENERATE_TETHER_VECTORS";

fn path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/vectors.json")
}

fn load() -> Value {
    let text = std::fs::read_to_string(path()).expect("read tests/vectors.json");
    serde_json::from_str(&text).expect("valid JSON")
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(v: &Value) -> Vec<u8> {
    let s = v.as_str().expect("a hex string");
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex"))
        .collect()
}

fn int<T: TryFrom<u64>>(v: &Value) -> T {
    let n = v.as_u64().expect("an integer");
    T::try_from(n).ok().expect("in range")
}

fn text(v: &Value) -> &str {
    v.as_str().expect("a string")
}

/// The bytes `b'a'..` repeated to `n`.
fn letters(n: usize) -> Vec<u8> {
    (0..n).map(|i| b'a' + (i % 26) as u8).collect()
}

/// Non-zero bytes `1..=255` repeated to `n`.
fn nonzero(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i % 255 + 1) as u8).collect()
}

fn encode(src: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    cobs::encode(&[src], |b| out.push(b));
    out
}

/// Writes `msg`, or panics with its error.
fn wire(msg: &Msg<'_>, seq: u8) -> Vec<u8> {
    let mut out = Vec::new();
    msg::write(msg, seq, |b| out.push(b)).expect("writes");
    out
}

fn receiver_for(ty: u8) -> Receiver {
    if ty & 0x80 == 0 {
        Receiver::Bridge
    } else {
        Receiver::Board
    }
}

fn receiver_name(r: Receiver) -> &'static str {
    match r {
        Receiver::Bridge => "bridge",
        Receiver::Board => "board",
    }
}

fn receiver_from(v: &Value) -> Receiver {
    match text(v) {
        "bridge" => Receiver::Bridge,
        "board" => Receiver::Board,
        other => panic!("unknown receiver {other}"),
    }
}

fn msg_error_name(e: MsgError) -> &'static str {
    match e {
        MsgError::UnknownType => "unknown_type",
        MsgError::Magic => "magic",
        MsgError::Version(_) => "version",
        MsgError::Length => "length",
        MsgError::Reserved => "reserved",
        MsgError::Range => "range",
        MsgError::UnknownPart => "unknown_part",
        MsgError::Direction => "direction",
        MsgError::Utf8 => "utf8",
    }
}

/// The error as a vector records it: `{"error": …}`, plus `"version"` for
/// a future `HELLO`.
fn msg_error_json(e: MsgError) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("error".into(), json!(msg_error_name(e)));
    if let MsgError::Version(v) = e {
        m.insert("version".into(), json!(v));
    }
    m
}

fn msg_json(m: &Msg<'_>) -> Value {
    match *m {
        Msg::Hello(ref h) => json!({
            "type": "HELLO",
            "lib_version": h.lib_version,
            "arch": h.arch,
            "board_id": hex(&h.board_id),
            "boot": h.boot,
            "gen": h.gen_,
            "flags": h.flags,
            "map_len": h.map_len,
            "map_crc": h.map_crc,
            "watchdog_ms": h.watchdog_ms,
            "rx_max": h.rx_max,
            "name": hex(h.name),
        }),
        Msg::Map { boot, offset, data } => json!({
            "type": "MAP", "boot": boot, "offset": offset, "data": hex(data),
        }),
        Msg::State { part, t_us, value } => json!({
            "type": "STATE", "part": part, "t_us": t_us, "value": hex(value),
        }),
        Msg::Status(ref s) => json!({
            "type": "STATUS",
            "boot": s.boot,
            "gen": s.gen_,
            "flags": s.flags,
            "reset": s.reset,
            "lost": s.lost,
            "bad": s.bad,
            "t_us": s.t_us,
        }),
        Msg::Note { level, code, text } => json!({
            "type": "NOTE", "level": level, "code": code, "text": hex(text),
        }),
        Msg::GetMap { boot, offset, max } => json!({
            "type": "GET_MAP", "boot": boot, "offset": offset, "max": max,
        }),
        Msg::Attach {
            board_id,
            boot,
            gen_,
            flags,
        } => json!({
            "type": "ATTACH", "board_id": hex(&board_id), "boot": boot, "gen": gen_, "flags": flags,
        }),
        Msg::Command {
            gen_,
            part,
            flags,
            ttl_ms,
            value,
        } => json!({
            "type": "COMMAND",
            "gen": gen_,
            "part": part,
            "flags": flags,
            "ttl_ms": ttl_ms,
            "value": hex(value),
        }),
        Msg::Safe { gen_, part } => json!({ "type": "SAFE", "gen": gen_, "part": part }),
        Msg::Ping { gen_ } => json!({ "type": "PING", "gen": gen_ }),
        Msg::Detach { gen_ } => json!({ "type": "DETACH", "gen": gen_ }),
    }
}

/// Builds the message a vector describes and hands it to `f` (the variable
/// parts borrow from a local buffer).
fn with_msg<R>(v: &Value, f: impl FnOnce(Msg<'_>) -> R) -> R {
    let bytes = |k: &str| unhex(&v[k]);
    let id = |k: &str| -> [u8; 8] { unhex(&v[k]).try_into().expect("8 bytes") };
    match text(&v["type"]) {
        "HELLO" => {
            let name = bytes("name");
            f(Msg::Hello(Hello {
                lib_version: int(&v["lib_version"]),
                arch: int(&v["arch"]),
                board_id: id("board_id"),
                boot: int(&v["boot"]),
                gen_: int(&v["gen"]),
                flags: int(&v["flags"]),
                map_len: int(&v["map_len"]),
                map_crc: int(&v["map_crc"]),
                watchdog_ms: int(&v["watchdog_ms"]),
                rx_max: int(&v["rx_max"]),
                name: &name,
            }))
        }
        "MAP" => {
            let data = bytes("data");
            f(Msg::Map {
                boot: int(&v["boot"]),
                offset: int(&v["offset"]),
                data: &data,
            })
        }
        "STATE" => {
            let value = bytes("value");
            f(Msg::State {
                part: int(&v["part"]),
                t_us: int(&v["t_us"]),
                value: &value,
            })
        }
        "STATUS" => f(Msg::Status(Status {
            boot: int(&v["boot"]),
            gen_: int(&v["gen"]),
            flags: int(&v["flags"]),
            reset: int(&v["reset"]),
            lost: int(&v["lost"]),
            bad: int(&v["bad"]),
            t_us: int(&v["t_us"]),
        })),
        "NOTE" => {
            let text = bytes("text");
            f(Msg::Note {
                level: int(&v["level"]),
                code: int(&v["code"]),
                text: &text,
            })
        }
        "GET_MAP" => f(Msg::GetMap {
            boot: int(&v["boot"]),
            offset: int(&v["offset"]),
            max: int(&v["max"]),
        }),
        "ATTACH" => f(Msg::Attach {
            board_id: id("board_id"),
            boot: int(&v["boot"]),
            gen_: int(&v["gen"]),
            flags: int(&v["flags"]),
        }),
        "COMMAND" => {
            let value = bytes("value");
            f(Msg::Command {
                gen_: int(&v["gen"]),
                part: int(&v["part"]),
                flags: int(&v["flags"]),
                ttl_ms: int(&v["ttl_ms"]),
                value: &value,
            })
        }
        "SAFE" => f(Msg::Safe {
            gen_: int(&v["gen"]),
            part: int(&v["part"]),
        }),
        "PING" => f(Msg::Ping {
            gen_: int(&v["gen"]),
        }),
        "DETACH" => f(Msg::Detach {
            gen_: int(&v["gen"]),
        }),
        other => panic!("unknown message type {other}"),
    }
}

/// `{"<part>": <dense length>}` as a lookup; parts not listed are unknown.
fn dense_lookup(v: &Value) -> impl Fn(u8) -> Option<usize> + '_ {
    move |part| {
        v.get("dense_len")
            .and_then(|m| m.get(part.to_string()))
            .map(int::<usize>)
    }
}

/// Dense lengths by part index.
type Parts = Vec<(u8, usize)>;
/// A name, the message, its `seq` and the parts it needs.
type MessageCase<'a> = (&'a str, Msg<'a>, u8, Parts);
/// A name, the receiver, a type and payload, the parts, and the error.
type Rejection<'a> = (&'a str, Receiver, u8, Vec<u8>, Parts, MsgError);
/// A name, the fields, offsets, lengths, stride and the dense value.
type PackCase<'a> = (&'a str, &'a str, Vec<usize>, Vec<usize>, usize, Vec<u8>);

fn dense_json(parts: &[(u8, usize)]) -> Value {
    Value::Object(
        parts
            .iter()
            .map(|(p, l)| (p.to_string(), json!(l)))
            .collect(),
    )
}

// ── Generation ───────────────────────────────────────────────────────────

fn crc_cases() -> Vec<Value> {
    let cases: &[(&str, Vec<u8>)] = &[
        ("empty", vec![]),
        ("catalogue check \"123456789\"", b"123456789".to_vec()),
        ("one zero byte", vec![0]),
        ("256 bytes 0x00..0xFF", (0..=255).collect()),
    ];
    cases
        .iter()
        .map(|(name, data)| {
            json!({
                "name": name,
                "data": hex(data),
                "crc16": crc::crc16(data),
                "crc32": crc::crc32(data),
            })
        })
        .collect()
}

fn cobs_encode_cases() -> Vec<Value> {
    let mut zero_after_block = nonzero(254);
    zero_after_block.push(0);
    let mut block_after_zero = vec![0];
    block_after_zero.extend(nonzero(254));
    let mut two_blocks = nonzero(254);
    two_blocks.extend(nonzero(254));
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("empty", vec![]),
        ("one zero", vec![0]),
        ("all zeros", vec![0; 5]),
        ("leading zero", vec![0, 0x11, 0x22]),
        ("trailing zeros", vec![0x11, 0, 0, 0]),
        ("zero in the middle", vec![0x11, 0x22, 0, 0x33]),
        ("no zeros", vec![0x11, 0x22, 0x33, 0x44]),
        ("run of 253", nonzero(253)),
        ("run of 254 ends exactly at a block", nonzero(254)),
        ("run of 255", nonzero(255)),
        ("a zero after a 254 block", zero_after_block),
        ("a 254 block after a zero", block_after_zero),
        ("two 254 blocks", two_blocks),
    ];
    cases
        .into_iter()
        .map(|(name, decoded)| {
            let encoded = encode(&decoded);
            assert!(encoded.len() <= decoded.len() + decoded.len() / 254 + 1);
            assert!(!encoded.contains(&0), "{name}");
            json!({ "name": name, "decoded": hex(&decoded), "encoded": hex(&encoded) })
        })
        .collect()
}

/// One decoder result: the decoded bytes, or why they were dropped.
type Decoded = Result<Vec<u8>, DecodeError>;

fn decode_stream_n<const N: usize>(wire: &[u8]) -> Vec<Decoded> {
    let mut d = Decoder::<N>::new();
    wire.iter()
        .filter_map(|&b| d.push(b).map(|r| r.map(<[u8]>::to_vec)))
        .collect()
}

/// Feeds `wire` to a decoder of `rx_max` bytes, one of the sizes the
/// vectors use.
fn decode_stream(wire: &[u8], rx_max: usize) -> Vec<Decoded> {
    match rx_max {
        16 => decode_stream_n::<16>(wire),
        64 => decode_stream_n::<64>(wire),
        254 => decode_stream_n::<254>(wire),
        256 => decode_stream_n::<256>(wire),
        512 => decode_stream_n::<512>(wire),
        _ => panic!("no decoder for rx_max {rx_max}"),
    }
}

fn decode_error_name(e: DecodeError) -> &'static str {
    match e {
        DecodeError::Overflow => "overflow",
        DecodeError::Truncated => "truncated",
    }
}

fn cobs_result_json(r: &Decoded) -> Value {
    match r {
        Ok(bytes) => json!({ "ok": hex(bytes) }),
        Err(e) => json!({ "error": decode_error_name(*e) }),
    }
}

fn frame_result_json(r: &Decoded) -> Value {
    match r {
        Ok(bytes) => match Frame::parse(bytes) {
            Ok(f) => json!({ "frame": { "type": f.ty, "seq": f.seq, "payload": hex(f.payload) } }),
            Err(FrameError::Size) => json!({ "error": "size" }),
            Err(FrameError::Crc) => json!({ "error": "crc" }),
        },
        Err(e) => json!({ "error": decode_error_name(*e) }),
    }
}

/// Joins wire pieces into one stream.
fn cat(parts: &[&[u8]]) -> Vec<u8> {
    parts.concat()
}

fn cobs_decode_cases() -> Vec<Value> {
    let block = encode(&nonzero(254));
    let two = encode(&[nonzero(254), nonzero(254)].concat());
    let mut sixteen_then_zero = nonzero(16);
    sixteen_then_zero.push(0);
    let cases: Vec<(&str, usize, Vec<u8>)> = vec![
        (
            "a 254 block, pinned form (no code byte after it)",
            256,
            cat(&[&block, &[0]]),
        ),
        (
            "a 254 block, trailing 0x01 variant",
            256,
            cat(&[&block, &[0x01, 0]]),
        ),
        (
            "a 254 block, pinned form, at exactly rx_max 254",
            254,
            cat(&[&block, &[0]]),
        ),
        (
            "a 254 block, trailing 0x01 variant, at exactly rx_max 254",
            254,
            cat(&[&block, &[0x01, 0]]),
        ),
        (
            "two 254 blocks, trailing 0x01 variant",
            512,
            cat(&[&two, &[0x01, 0]]),
        ),
        (
            "empty frames between delimiters are skipped",
            256,
            vec![0, 0, 0x03, 0x11, 0x22, 0, 0],
        ),
        (
            "a lone code byte decodes to an empty frame",
            256,
            vec![0x01, 0],
        ),
        ("all zeros", 256, cat(&[&encode(&[0; 5]), &[0]])),
        (
            "a truncated block, then a good frame",
            256,
            vec![0x05, 0xAA, 0xBB, 0, 0x03, 0x11, 0x22, 0],
        ),
        (
            "rx_max decoded bytes fit",
            16,
            cat(&[&encode(&nonzero(16)), &[0]]),
        ),
        (
            "one byte over rx_max overflows, then a good frame",
            16,
            cat(&[&encode(&nonzero(17)), &[0, 0x02, 0x09, 0]]),
        ),
        (
            "an implied zero over rx_max overflows (decoded bytes count)",
            16,
            cat(&[&encode(&sixteen_then_zero), &[0, 0x02, 0x09, 0]]),
        ),
        (
            "257 decoded bytes overflow a 256 buffer, then a good frame",
            256,
            cat(&[&encode(&nonzero(257)), &[0, 0x02, 0x09, 0]]),
        ),
        (
            "overflow and truncation in one frame: overflow wins",
            16,
            cat(&[&[0x20], &nonzero(20), &[0, 0x02, 0x09, 0]]),
        ),
        (
            "an unterminated frame yields nothing yet",
            256,
            vec![0x03, 0x11, 0x22],
        ),
    ];
    cases
        .into_iter()
        .map(|(name, rx_max, wire)| {
            let results: Vec<Value> = decode_stream(&wire, rx_max)
                .iter()
                .map(cobs_result_json)
                .collect();
            json!({ "name": name, "rx_max": rx_max, "wire": hex(&wire), "results": results })
        })
        .collect()
}

fn frame_cases() -> Vec<Value> {
    let ping = wire(&Msg::Ping { gen_: 3 }, 1);
    // A good PING's decoded bytes, with its CRC or a payload byte flipped.
    let decoded = |w: &[u8]| decode_stream(w, 256).remove(0).expect("decodes");
    let mut bad_crc = decoded(&ping);
    let last = bad_crc.len() - 1;
    bad_crc[last] ^= 0x01;
    let mut bad_payload = decoded(&ping);
    bad_payload[2] ^= 0x80;
    let attach = wire(
        &Msg::Attach {
            board_id: [1, 2, 3, 4, 5, 6, 7, 8],
            boot: 0x0102,
            gen_: 1,
            flags: 0,
        },
        0,
    );
    let command17 = wire(
        &Msg::Command {
            gen_: 1,
            part: 0,
            flags: 0,
            ttl_ms: 60,
            value: &[0x11; 8],
        },
        2,
    );
    let largest = wire(
        &Msg::Map {
            boot: 1,
            offset: 0,
            data: &letters(MAX_MAP_SLICE),
        },
        4,
    );
    let mut short = vec![ty::PING, 0];
    short.extend_from_slice(&crc::crc16(&short).to_le_bytes());
    let cases: Vec<(&str, usize, Vec<u8>)> = vec![
        ("a good frame", 256, ping.clone()),
        (
            "a bad CRC, then a good frame",
            256,
            cat(&[&encode(&bad_crc), &[0], &ping]),
        ),
        (
            "a corrupted payload byte, then a good frame",
            256,
            cat(&[&encode(&bad_payload), &[0], &ping]),
        ),
        (
            "a truncated block, then a good frame",
            256,
            cat(&[&[0x05, 0xAA, 0xBB, 0], &ping]),
        ),
        (
            "a 17-byte frame overflows rx_max 16, then a good frame",
            16,
            cat(&[&command17, &ping]),
        ),
        ("ATTACH (16 bytes decoded) fits rx_max 16", 16, attach),
        (
            "a frame under 5 decoded bytes, then a good frame",
            256,
            cat(&[&encode(&short), &[0], &ping]),
        ),
        (
            "a lone code byte is an empty frame, then a good frame",
            256,
            cat(&[&[0x01, 0], &ping]),
        ),
        ("the largest frame (256 decoded bytes)", 256, largest),
        (
            "257 decoded bytes overflow, then a good frame",
            256,
            cat(&[&encode(&nonzero(257)), &[0], &ping]),
        ),
        (
            "line noise and empty frames, then a good frame",
            256,
            cat(&[&[0, 0, 0x37, 0x42, 0, 0], &ping]),
        ),
        (
            "a sketch's text, then a provisioning frame with its leading delimiter (§14.2)",
            256,
            cat(&[b"booted\r\n", &prov_wire(&Prov::GetDevice, 0)]),
        ),
    ];
    cases
        .into_iter()
        .map(|(name, rx_max, wire)| {
            let results: Vec<Value> = decode_stream(&wire, rx_max)
                .iter()
                .map(frame_result_json)
                .collect();
            json!({ "name": name, "rx_max": rx_max, "wire": hex(&wire), "results": results })
        })
        .collect()
}

fn hello<'a>(name: &'a [u8]) -> Hello<'a> {
    Hello {
        lib_version: 0x0100,
        arch: 1,
        board_id: [0xB0, 0xB1, 0x00, 0xB3, 0xB4, 0x00, 0x00, 0xB7],
        boot: 0x0201,
        gen_: 0,
        flags: 0,
        map_len: 600,
        map_crc: 0xDEAD_BEEF,
        watchdog_ms: 250,
        rx_max: 64,
        name,
    }
}

fn message_cases() -> Vec<Value> {
    let name32 = letters(MAX_NAME);
    let map_full = letters(MAX_MAP_SLICE);
    let note_full = letters(MAX_NOTE);
    let state_full: Vec<u8> = (0..247).map(|i| i as u8).collect();
    let utf8_name = "œil-ü".as_bytes();
    let cases: Vec<MessageCase<'_>> = vec![
        ("HELLO", Msg::Hello(hello(b"eye-1")), 0, vec![]),
        ("HELLO, empty name", Msg::Hello(hello(b"")), 1, vec![]),
        ("HELLO, 32-byte name", Msg::Hello(hello(&name32)), 2, vec![]),
        (
            "HELLO, multi-byte UTF-8 name",
            Msg::Hello(hello(utf8_name)),
            3,
            vec![],
        ),
        (
            "HELLO, lowest rx_max and watchdog_ms, e-stop latched",
            Msg::Hello(Hello {
                rx_max: 16,
                watchdog_ms: 20,
                flags: hello_flags::ESTOP,
                gen_: 7,
                ..hello(b"n")
            }),
            4,
            vec![],
        ),
        (
            "HELLO, highest rx_max and watchdog_ms",
            Msg::Hello(Hello {
                rx_max: 256,
                watchdog_ms: 5000,
                arch: 4,
                ..hello(b"n")
            }),
            5,
            vec![],
        ),
        (
            "MAP, the largest slice (a 256-byte frame)",
            Msg::Map {
                boot: 0x0201,
                offset: 0,
                data: &map_full,
            },
            6,
            vec![],
        ),
        (
            "MAP, empty slice past the end",
            Msg::Map {
                boot: 0x0201,
                offset: 600,
                data: &[],
            },
            7,
            vec![],
        ),
        (
            "STATE, two f32",
            Msg::State {
                part: 0,
                t_us: 0x0102_0304,
                value: &[0, 0, 0xC0, 0x3F, 0, 0, 0x80, 0xBE],
            },
            8,
            vec![(0, 8)],
        ),
        (
            "STATE, the largest sample (a 256-byte frame)",
            Msg::State {
                part: 2,
                t_us: u32::MAX,
                value: &state_full,
            },
            9,
            vec![(2, 247)],
        ),
        (
            "STATUS, every flag",
            Msg::Status(Status {
                boot: 0x0201,
                gen_: 7,
                flags: status_flags::ALL,
                reset: 3,
                lost: 0,
                bad: 0xFFFF,
                t_us: 1_000_000,
            }),
            10,
            vec![],
        ),
        (
            "NOTE, empty text",
            Msg::Note {
                level: 0,
                code: 0,
                text: &[],
            },
            11,
            vec![],
        ),
        (
            "NOTE, 48-byte text",
            Msg::Note {
                level: 2,
                code: 9,
                text: &note_full,
            },
            12,
            vec![],
        ),
        (
            "NOTE, text that isn't UTF-8 (accepted: only HELLO.name is checked)",
            Msg::Note {
                level: 1,
                code: 3,
                text: &[0xFF, 0xFE, 0x41, 0xC0, 0x80],
            },
            13,
            vec![],
        ),
        (
            "GET_MAP, max 1",
            Msg::GetMap {
                boot: 0x0201,
                offset: 0,
                max: 1,
            },
            0,
            vec![],
        ),
        (
            "GET_MAP, max 248",
            Msg::GetMap {
                boot: 0x0201,
                offset: 248,
                max: 248,
            },
            1,
            vec![],
        ),
        (
            "ATTACH, clear e-stop",
            Msg::Attach {
                board_id: [0xB0, 0xB1, 0x00, 0xB3, 0xB4, 0x00, 0x00, 0xB7],
                boot: 0x0201,
                gen_: 255,
                flags: attach_flags::CLEAR_ESTOP,
            },
            2,
            vec![],
        ),
        (
            "COMMAND, hold",
            Msg::Command {
                gen_: 7,
                part: 0,
                flags: command_flags::HOLD,
                ttl_ms: 60,
                value: &[0, 0, 0x80, 0x3F, 0, 0, 0, 0],
            },
            3,
            vec![(0, 8)],
        ),
        (
            "COMMAND, a 16-byte frame (the smallest rx_max)",
            Msg::Command {
                gen_: 7,
                part: 1,
                flags: 0,
                ttl_ms: 0xFFFF,
                value: &[1, 2, 3, 0, 5, 6, 7],
            },
            4,
            vec![(1, 7)],
        ),
        (
            "SAFE, every part",
            Msg::Safe {
                gen_: 7,
                part: EVERY_PART,
            },
            254,
            vec![],
        ),
        (
            "SAFE, one part",
            Msg::Safe { gen_: 7, part: 0 },
            255,
            vec![],
        ),
        ("PING", Msg::Ping { gen_: 7 }, 0, vec![]),
        ("DETACH", Msg::Detach { gen_: 7 }, 1, vec![]),
    ];
    cases
        .into_iter()
        .map(|(name, m, seq, parts)| {
            let w = wire(&m, seq);
            // It decodes back to itself at the end it's meant for.
            let mut got = decode_stream(&w, 256);
            let bytes = got.remove(0).expect("decodes");
            let f = Frame::parse(&bytes).expect("frame");
            assert_eq!(f.seq, seq);
            let lens = |p: u8| parts.iter().find(|(q, _)| *q == p).map(|(_, l)| *l);
            assert_eq!(msg::decode(&f, receiver_for(f.ty), lens), Ok(m), "{name}");
            let mut v = json!({ "name": name, "seq": seq, "msg": msg_json(&m), "wire": hex(&w) });
            if !parts.is_empty() {
                v["dense_len"] = dense_json(&parts);
            }
            v
        })
        .collect()
}

/// A valid `HELLO` payload with `name`: edit its bytes to break it.
fn hello_payload(name: &[u8]) -> Vec<u8> {
    let mut p = b"RLMT\x01".to_vec();
    p.extend_from_slice(&[0x00, 0x01, 1]); // lib_version, arch
    p.extend_from_slice(&[0xB0, 0xB1, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7]);
    p.extend_from_slice(&[0x01, 0x02, 0, 0]); // boot, gen, flags
    p.extend_from_slice(&600_u16.to_le_bytes());
    p.extend_from_slice(&0xDEAD_BEEF_u32.to_le_bytes());
    p.extend_from_slice(&250_u16.to_le_bytes());
    p.extend_from_slice(&64_u16.to_le_bytes());
    p.push(u8::try_from(name.len()).expect("short"));
    p.extend_from_slice(name);
    p
}

fn with(mut p: Vec<u8>, at: usize, bytes: &[u8]) -> Vec<u8> {
    p[at..at + bytes.len()].copy_from_slice(bytes);
    p
}

fn rejection_cases() -> Vec<Value> {
    use MsgError as E;
    use Receiver::{Board, Bridge};
    let h = hello_payload(b"eye");
    let le = |n: u16| n.to_le_bytes();
    let mut long_name = h.clone();
    long_name.push(b'x');
    let mut cases: Vec<Rejection<'_>> = vec![
        // Direction (§3.2).
        (
            "COMMAND heard by the bridge (an echoing adapter)",
            Bridge,
            ty::COMMAND,
            vec![1, 0, 0, 60, 0, 1, 2, 3, 4],
            vec![(0, 4)],
            E::Direction,
        ),
        (
            "STATUS heard by the board",
            Board,
            ty::STATUS,
            vec![0; 13],
            vec![],
            E::Direction,
        ),
        (
            "HELLO heard by the board",
            Board,
            ty::HELLO,
            h.clone(),
            vec![],
            E::Direction,
        ),
        (
            "PING heard by the bridge",
            Bridge,
            ty::PING,
            vec![1],
            vec![],
            E::Direction,
        ),
        // Unknown types.
        (
            "type 0x06 at the bridge",
            Bridge,
            0x06,
            vec![0],
            vec![],
            E::UnknownType,
        ),
        (
            "type 0x00 at the bridge",
            Bridge,
            0x00,
            vec![0],
            vec![],
            E::UnknownType,
        ),
        (
            "type 0x7F at the bridge",
            Bridge,
            0x7F,
            vec![0],
            vec![],
            E::UnknownType,
        ),
        (
            "type 0x80 at the board",
            Board,
            0x80,
            vec![0],
            vec![],
            E::UnknownType,
        ),
        (
            "type 0x87 at the board",
            Board,
            0x87,
            vec![0],
            vec![],
            E::UnknownType,
        ),
        (
            "type 0xFF at the board",
            Board,
            0xFF,
            vec![0],
            vec![],
            E::UnknownType,
        ),
        // HELLO: magic, then version, then length.
        (
            "HELLO, bad magic",
            Bridge,
            ty::HELLO,
            with(h.clone(), 0, b"RLMX"),
            vec![],
            E::Magic,
        ),
        (
            "HELLO, shorter than the magic",
            Bridge,
            ty::HELLO,
            b"RLM".to_vec(),
            vec![],
            E::Magic,
        ),
        (
            "HELLO, the magic alone",
            Bridge,
            ty::HELLO,
            b"RLMT".to_vec(),
            vec![],
            E::Length,
        ),
        (
            "HELLO, magic and version alone",
            Bridge,
            ty::HELLO,
            b"RLMT\x01".to_vec(),
            vec![],
            E::Length,
        ),
        (
            "HELLO, a future version, short",
            Bridge,
            ty::HELLO,
            b"RLMT\x02xyz".to_vec(),
            vec![],
            E::Version(2),
        ),
        (
            "HELLO, a future version, full length",
            Bridge,
            ty::HELLO,
            with(h.clone(), 4, &[2]),
            vec![],
            E::Version(2),
        ),
        (
            "HELLO, version 0",
            Bridge,
            ty::HELLO,
            with(h.clone(), 4, &[0]),
            vec![],
            E::Version(0),
        ),
        (
            "HELLO, name_len past the payload",
            Bridge,
            ty::HELLO,
            with(h.clone(), 30, &[4]),
            vec![],
            E::Length,
        ),
        (
            "HELLO, a byte after the name",
            Bridge,
            ty::HELLO,
            long_name,
            vec![],
            E::Length,
        ),
        (
            "HELLO, 30 bytes (no name_len)",
            Bridge,
            ty::HELLO,
            h[..30].to_vec(),
            vec![],
            E::Length,
        ),
        // HELLO: ranges, reserved bits, UTF-8.
        (
            "HELLO, name_len 33",
            Bridge,
            ty::HELLO,
            with(hello_payload(&letters(33)), 30, &[33]),
            vec![],
            E::Range,
        ),
        (
            "HELLO, reserved flag bit 1",
            Bridge,
            ty::HELLO,
            with(h.clone(), 19, &[0x02]),
            vec![],
            E::Reserved,
        ),
        (
            "HELLO, reserved flag bit 7",
            Bridge,
            ty::HELLO,
            with(h.clone(), 19, &[0x81]),
            vec![],
            E::Reserved,
        ),
        (
            "HELLO, a reserved bit is reported before a range",
            Bridge,
            ty::HELLO,
            with(with(h.clone(), 19, &[0x02]), 28, &le(8)),
            vec![],
            E::Reserved,
        ),
        (
            "HELLO, rx_max 15",
            Bridge,
            ty::HELLO,
            with(h.clone(), 28, &le(15)),
            vec![],
            E::Range,
        ),
        (
            "HELLO, rx_max 257",
            Bridge,
            ty::HELLO,
            with(h.clone(), 28, &le(257)),
            vec![],
            E::Range,
        ),
        (
            "HELLO, watchdog_ms 19",
            Bridge,
            ty::HELLO,
            with(h.clone(), 26, &le(19)),
            vec![],
            E::Range,
        ),
        (
            "HELLO, watchdog_ms 5001",
            Bridge,
            ty::HELLO,
            with(h.clone(), 26, &le(5001)),
            vec![],
            E::Range,
        ),
        (
            "HELLO, a range is reported before UTF-8",
            Bridge,
            ty::HELLO,
            with(with(h.clone(), 26, &le(0)), 31, &[0xFF]),
            vec![],
            E::Range,
        ),
    ];
    let bad_names: &[(&str, &[u8])] = &[
        ("HELLO, name 0xFF", &[b'a', 0xFF]),
        ("HELLO, name with an overlong NUL (C0 80)", &[0xC0, 0x80]),
        (
            "HELLO, name with a surrogate (ED A0 80)",
            &[0xED, 0xA0, 0x80],
        ),
        (
            "HELLO, name ending mid-character (E2 82)",
            &[b'a', 0xE2, 0x82],
        ),
        (
            "HELLO, name above U+10FFFF (F4 90 80 80)",
            &[0xF4, 0x90, 0x80, 0x80],
        ),
        ("HELLO, name with a lone continuation byte", &[0x80]),
        (
            "HELLO, name with an overlong 3-byte form (E0 80 AF)",
            &[0xE0, 0x80, 0xAF],
        ),
    ];
    for (name, bytes) in bad_names {
        cases.push((
            name,
            Bridge,
            ty::HELLO,
            hello_payload(bytes),
            vec![],
            E::Utf8,
        ));
    }
    cases.extend([
        // Exact lengths.
        (
            "MAP, 3 bytes",
            Bridge,
            ty::MAP,
            vec![0; 3],
            vec![],
            E::Length,
        ),
        (
            "MAP, a 249-byte slice",
            Bridge,
            ty::MAP,
            vec![0; 4 + 249],
            vec![],
            E::Length,
        ),
        (
            "STATE, empty",
            Bridge,
            ty::STATE,
            vec![],
            vec![(0, 4)],
            E::Length,
        ),
        (
            "STATE, one byte short",
            Bridge,
            ty::STATE,
            vec![0; 8],
            vec![(0, 4)],
            E::Length,
        ),
        (
            "STATE, one byte long",
            Bridge,
            ty::STATE,
            vec![0; 10],
            vec![(0, 4)],
            E::Length,
        ),
        (
            "STATE, an unknown part",
            Bridge,
            ty::STATE,
            vec![3, 0, 0, 0, 0, 1, 2, 3, 4],
            vec![(0, 4)],
            E::UnknownPart,
        ),
        (
            "STATE, 1 byte with an unknown part",
            Bridge,
            ty::STATE,
            vec![9],
            vec![(0, 4)],
            E::UnknownPart,
        ),
        (
            "STATE, 4 bytes with an unknown part",
            Bridge,
            ty::STATE,
            vec![9, 0, 0, 0],
            vec![(0, 4)],
            E::UnknownPart,
        ),
        (
            "STATUS, 12 bytes",
            Bridge,
            ty::STATUS,
            vec![0; 12],
            vec![],
            E::Length,
        ),
        (
            "STATUS, 14 bytes",
            Bridge,
            ty::STATUS,
            vec![0; 14],
            vec![],
            E::Length,
        ),
        (
            "STATUS, reserved flag bit 3",
            Bridge,
            ty::STATUS,
            with(vec![0; 13], 3, &[0x08]),
            vec![],
            E::Reserved,
        ),
        ("NOTE, 1 byte", Bridge, ty::NOTE, vec![0], vec![], E::Length),
        (
            "NOTE, 49 bytes of text",
            Bridge,
            ty::NOTE,
            vec![0x41; 2 + 49],
            vec![],
            E::Length,
        ),
        (
            "GET_MAP, 4 bytes",
            Board,
            ty::GET_MAP,
            vec![0, 0, 0, 0],
            vec![],
            E::Length,
        ),
        (
            "GET_MAP, 6 bytes",
            Board,
            ty::GET_MAP,
            vec![0, 0, 0, 0, 1, 0],
            vec![],
            E::Length,
        ),
        (
            "GET_MAP, max 0",
            Board,
            ty::GET_MAP,
            vec![0, 0, 0, 0, 0],
            vec![],
            E::Range,
        ),
        (
            "GET_MAP, max 249",
            Board,
            ty::GET_MAP,
            vec![0, 0, 0, 0, 249],
            vec![],
            E::Range,
        ),
        (
            "ATTACH, 11 bytes",
            Board,
            ty::ATTACH,
            vec![0; 11],
            vec![],
            E::Length,
        ),
        (
            "ATTACH, 13 bytes",
            Board,
            ty::ATTACH,
            vec![0; 13],
            vec![],
            E::Length,
        ),
        (
            "ATTACH, reserved flag bit 1",
            Board,
            ty::ATTACH,
            with(vec![0; 12], 11, &[0x02]),
            vec![],
            E::Reserved,
        ),
        (
            "COMMAND, 4 bytes",
            Board,
            ty::COMMAND,
            vec![1, 0, 0, 60],
            vec![(0, 4)],
            E::Length,
        ),
        (
            "COMMAND, an unknown part",
            Board,
            ty::COMMAND,
            vec![1, 5, 0, 60, 0, 1, 2, 3, 4],
            vec![(0, 4)],
            E::UnknownPart,
        ),
        (
            "COMMAND, one byte short",
            Board,
            ty::COMMAND,
            vec![1, 0, 0, 60, 0, 1, 2, 3],
            vec![(0, 4)],
            E::Length,
        ),
        (
            "COMMAND, one byte long",
            Board,
            ty::COMMAND,
            vec![1, 0, 0, 60, 0, 1, 2, 3, 4, 5],
            vec![(0, 4)],
            E::Length,
        ),
        (
            "COMMAND, reserved flag bit 1",
            Board,
            ty::COMMAND,
            vec![1, 0, 0x02, 60, 0, 1, 2, 3, 4],
            vec![(0, 4)],
            E::Reserved,
        ),
        (
            "COMMAND, an unknown part is reported before a reserved bit",
            Board,
            ty::COMMAND,
            vec![1, 5, 0x02, 60, 0, 1, 2, 3, 4],
            vec![(0, 4)],
            E::UnknownPart,
        ),
        (
            "COMMAND, a length is reported before a reserved bit",
            Board,
            ty::COMMAND,
            vec![1, 0, 0x02, 60, 0, 1, 2, 3],
            vec![(0, 4)],
            E::Length,
        ),
        ("SAFE, 1 byte", Board, ty::SAFE, vec![1], vec![], E::Length),
        (
            "SAFE, 3 bytes",
            Board,
            ty::SAFE,
            vec![1, 0, 0],
            vec![],
            E::Length,
        ),
        ("PING, empty", Board, ty::PING, vec![], vec![], E::Length),
        (
            "PING, 2 bytes",
            Board,
            ty::PING,
            vec![1, 0],
            vec![],
            E::Length,
        ),
        (
            "DETACH, empty",
            Board,
            ty::DETACH,
            vec![],
            vec![],
            E::Length,
        ),
        (
            "DETACH, 2 bytes",
            Board,
            ty::DETACH,
            vec![1, 0],
            vec![],
            E::Length,
        ),
    ]);
    cases
        .into_iter()
        .map(|(name, receiver, t, payload, parts, want)| {
            let f = Frame {
                ty: t,
                seq: 0,
                payload: &payload,
            };
            let lens = |p: u8| parts.iter().find(|(q, _)| *q == p).map(|(_, l)| *l);
            assert_eq!(msg::decode(&f, receiver, lens), Err(want), "{name}");
            let mut v = json!({
                "name": name,
                "receiver": receiver_name(receiver),
                "type": t,
                "payload": hex(&payload),
            });
            if !parts.is_empty() {
                v["dense_len"] = dense_json(&parts);
            }
            v.as_object_mut()
                .expect("object")
                .extend(msg_error_json(want));
            v
        })
        .collect()
}

fn write_rejection_cases() -> Vec<Value> {
    use MsgError as E;
    let name33 = letters(33);
    let big = [0_u8; 248];
    let cases: Vec<(&str, Msg<'_>, MsgError)> = vec![
        (
            "HELLO, a 33-byte name",
            Msg::Hello(hello(&name33)),
            E::Range,
        ),
        (
            "HELLO, a non-UTF-8 name",
            Msg::Hello(hello(&[0xFF])),
            E::Utf8,
        ),
        (
            "HELLO, rx_max 15",
            Msg::Hello(Hello {
                rx_max: 15,
                ..hello(b"n")
            }),
            E::Range,
        ),
        (
            "HELLO, rx_max 257",
            Msg::Hello(Hello {
                rx_max: 257,
                ..hello(b"n")
            }),
            E::Range,
        ),
        (
            "HELLO, watchdog_ms 19",
            Msg::Hello(Hello {
                watchdog_ms: 19,
                ..hello(b"n")
            }),
            E::Range,
        ),
        (
            "HELLO, watchdog_ms 5001",
            Msg::Hello(Hello {
                watchdog_ms: 5001,
                ..hello(b"n")
            }),
            E::Range,
        ),
        (
            "HELLO, a reserved flag",
            Msg::Hello(Hello {
                flags: 0x02,
                ..hello(b"n")
            }),
            E::Reserved,
        ),
        (
            "HELLO, a reserved bit is reported before a range",
            Msg::Hello(Hello {
                flags: 0x02,
                rx_max: 8,
                ..hello(b"n")
            }),
            E::Reserved,
        ),
        (
            "HELLO, a reserved bit is reported before UTF-8",
            Msg::Hello(Hello {
                flags: 0x02,
                ..hello(&[0xFF])
            }),
            E::Reserved,
        ),
        (
            "HELLO, a range is reported before UTF-8",
            Msg::Hello(Hello {
                rx_max: 8,
                ..hello(&[0xFF])
            }),
            E::Range,
        ),
        (
            "MAP, a 249-byte slice",
            Msg::Map {
                boot: 0,
                offset: 0,
                data: &[0; 249],
            },
            E::Length,
        ),
        (
            "STATE, over a frame",
            Msg::State {
                part: 0,
                t_us: 0,
                value: &big,
            },
            E::Length,
        ),
        (
            "STATUS, a reserved flag",
            Msg::Status(Status {
                boot: 0,
                gen_: 0,
                flags: 0x08,
                reset: 0,
                lost: 0,
                bad: 0,
                t_us: 0,
            }),
            E::Reserved,
        ),
        (
            "NOTE, 49 bytes of text",
            Msg::Note {
                level: 0,
                code: 0,
                text: &[0x41; 49],
            },
            E::Length,
        ),
        (
            "GET_MAP, max 0",
            Msg::GetMap {
                boot: 0,
                offset: 0,
                max: 0,
            },
            E::Range,
        ),
        (
            "GET_MAP, max 249",
            Msg::GetMap {
                boot: 0,
                offset: 0,
                max: 249,
            },
            E::Range,
        ),
        (
            "ATTACH, a reserved flag",
            Msg::Attach {
                board_id: [0; 8],
                boot: 0,
                gen_: 1,
                flags: 0x02,
            },
            E::Reserved,
        ),
        (
            "COMMAND, a reserved flag",
            Msg::Command {
                gen_: 1,
                part: 0,
                flags: 0x02,
                ttl_ms: 60,
                value: &[0; 4],
            },
            E::Reserved,
        ),
        (
            "COMMAND, over a frame",
            Msg::Command {
                gen_: 1,
                part: 0,
                flags: 0,
                ttl_ms: 60,
                value: &big,
            },
            E::Length,
        ),
    ];
    cases
        .into_iter()
        .map(|(name, m, want)| {
            let mut wrote = false;
            assert_eq!(msg::write(&m, 0, |_| wrote = true), Err(want), "{name}");
            assert!(!wrote, "{name}: nothing is written on an error");
            let mut v = json!({ "name": name, "msg": msg_json(&m) });
            v.as_object_mut()
                .expect("object")
                .extend(msg_error_json(want));
            v
        })
        .collect()
}

fn f32s(xs: &[f32]) -> Vec<u8> {
    xs.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn pack_cases() -> Vec<Value> {
    let cases: Vec<PackCase<'_>> = vec![
        (
            "two f32 fields: aligned 16, dense 8",
            "pan: f32, tilt: f32",
            vec![0, 8],
            vec![4, 4],
            16,
            f32s(&[1.5, -0.25]),
        ),
        (
            "mixed f32, u8 and bool",
            "pos: f32, id: u8, on: bool",
            vec![0, 8, 16],
            vec![4, 1, 1],
            24,
            [f32s(&[0.5]), vec![7, 1]].concat(),
        ),
        (
            "a 2-D shape, then a u8 pair",
            "pos: f32[2][3], id: u8[2]",
            vec![0, 24],
            vec![24, 2],
            32,
            [f32s(&[1.0, 2.0, 3.0, -1.0, -2.0, -3.0]), vec![0xAA, 0x55]].concat(),
        ),
        (
            "f64 fields around an f32",
            "x: f64, k: f32, y: f64",
            vec![0, 8, 16],
            vec![8, 4, 8],
            24,
            [
                1.25_f64.to_le_bytes().to_vec(),
                f32s(&[3.0]),
                (-2.5_f64).to_le_bytes().to_vec(),
            ]
            .concat(),
        ),
        (
            "f32[3], bool, u8[2]",
            "pos: f32[3], on: bool, id: u8[2]",
            vec![0, 16, 24],
            vec![12, 1, 2],
            32,
            [f32s(&[0.1, 0.2, 0.3]), vec![0, 14, 15]].concat(),
        ),
    ];
    cases
        .into_iter()
        .map(|(name, fields, offsets, lens, stride, dense)| {
            assert_eq!(pack::dense_len(&lens), Some(dense.len()), "{name}");
            let mut aligned = vec![0xEE; stride];
            pack::to_aligned(&dense, &offsets, &lens, &mut aligned).expect("aligned");
            let mut back = vec![0; dense.len()];
            pack::to_dense(&aligned, &offsets, &lens, &mut back).expect("dense");
            assert_eq!(back, dense, "{name}");
            json!({
                "name": name,
                "fields": fields,
                "offsets": offsets,
                "lens": lens,
                "aligned": hex(&aligned),
                "dense": hex(&dense),
            })
        })
        .collect()
}

fn generate() -> Value {
    json!({
        "schema": 1,
        "crc": crc_cases(),
        "cobs_encode": cobs_encode_cases(),
        "cobs_decode": cobs_decode_cases(),
        "frames": frame_cases(),
        "messages": message_cases(),
        "rejections": rejection_cases(),
        "write_rejections": write_rejection_cases(),
        "pack": pack_cases(),
        "provision": provision_cases(),
        "provision_rejections": provision_rejection_cases(),
        "provision_write_rejections": provision_write_rejection_cases(),
    })
}

// ── Provisioning (spec §14) ──────────────────────────────────────────────

/// Writes `msg`, or panics with its error: a leading delimiter, then the
/// frame.
fn prov_wire(msg: &Prov<'_>, seq: u8) -> Vec<u8> {
    let mut out = Vec::new();
    provision::write(msg, seq, |b| out.push(b)).expect("writes");
    out
}

fn prov_json(m: &Prov<'_>) -> Value {
    match *m {
        Prov::GetDevice => json!({ "type": "GET_DEVICE" }),
        Prov::DeviceInfo(ref d) => json!({
            "type": "DEVICE_INFO",
            "lib_version": d.lib_version,
            "arch": d.arch,
            "flags": d.flags,
            "device_key": hex(&d.device_key),
            "host_id": hex(d.host_id),
            "name": hex(d.name),
        }),
        Prov::Provisioned { status, record_crc } => json!({
            "type": "PROVISIONED",
            "status": status,
            "record_crc": record_crc,
        }),
        Prov::Provision(ref p) => json!({
            "type": "PROVISION",
            "flags": p.flags,
            "host_key": hex(&p.host_key),
            "host_id": hex(p.host_id),
            "addresses": p.addresses.as_slice().iter().map(|a| hex(a)).collect::<Vec<_>>(),
        }),
    }
}

fn key32(v: &Value) -> [u8; 32] {
    unhex(v).try_into().expect("32 bytes")
}

/// Builds the message `v` describes (bytes borrowed from locals) and
/// passes it to `f`.
fn with_prov<R>(v: &Value, f: impl FnOnce(Prov<'_>) -> R) -> R {
    match text(&v["type"]) {
        "GET_DEVICE" => f(Prov::GetDevice),
        "DEVICE_INFO" => {
            let (host_id, name) = (unhex(&v["host_id"]), unhex(&v["name"]));
            f(Prov::DeviceInfo(DeviceInfo {
                lib_version: int(&v["lib_version"]),
                arch: int(&v["arch"]),
                flags: int(&v["flags"]),
                device_key: key32(&v["device_key"]),
                host_id: &host_id,
                name: &name,
            }))
        }
        "PROVISIONED" => f(Prov::Provisioned {
            status: int(&v["status"]),
            record_crc: int(&v["record_crc"]),
        }),
        "PROVISION" => {
            let host_id = unhex(&v["host_id"]);
            let list: Vec<Vec<u8>> = v["addresses"]
                .as_array()
                .expect("addresses")
                .iter()
                .map(unhex)
                .collect();
            let refs: Vec<&[u8]> = list.iter().map(Vec::as_slice).collect();
            f(Prov::Provision(Provision {
                flags: int(&v["flags"]),
                host_key: key32(&v["host_key"]),
                host_id: &host_id,
                addresses: Addresses::new(&refs).expect("at most 4"),
            }))
        }
        other => panic!("unknown provisioning type {other}"),
    }
}

fn device_info<'a>(host_id: &'a [u8], name: &'a [u8]) -> DeviceInfo<'a> {
    DeviceInfo {
        lib_version: 0x0200,
        arch: 4,
        flags: if host_id.is_empty() {
            0
        } else {
            device_flags::PROVISIONED
        },
        device_key: core::array::from_fn(|i| 0xA0 ^ i as u8),
        host_id,
        name,
    }
}

fn provision_of<'a>(host_id: &'a [u8], addresses: &[&'a [u8]]) -> Provision<'a> {
    Provision {
        flags: 0,
        host_key: core::array::from_fn(|i| 0x50 + i as u8),
        host_id,
        addresses: Addresses::new(addresses).expect("at most 4"),
    }
}

/// An address of `len` bytes: `h…h:7447`.
fn address(len: usize) -> Vec<u8> {
    let mut a = vec![b'h'; len - 5];
    a.extend_from_slice(b":7447");
    a
}

/// A PROVISION payload of exactly `len` bytes (36 + 32-byte host id + 4
/// addresses), with addresses as long as it takes.
fn provision_addresses(len: usize) -> Vec<Vec<u8>> {
    // 36 + 32 + 4 × (1 + a) = len.
    let total = len - 68 - 4;
    let base = total / 4;
    (0..4)
        .map(|i| address(base + usize::from(i < total % 4)))
        .collect()
}

fn provision_cases() -> Vec<Value> {
    let id32 = letters(MAX_ID);
    let full = provision_addresses(252);
    let full_refs: Vec<&[u8]> = full.iter().map(Vec::as_slice).collect();
    let longest = address(MAX_ADDRESS);
    let cases: Vec<(&str, Prov<'_>, u8)> = vec![
        ("GET_DEVICE", Prov::GetDevice, 0),
        (
            "DEVICE_INFO, not provisioned",
            Prov::DeviceInfo(device_info(b"", b"eye-3")),
            0,
        ),
        (
            "DEVICE_INFO, provisioned, 32-byte host id and name",
            Prov::DeviceInfo(device_info(&id32, &id32)),
            1,
        ),
        (
            "DEVICE_INFO, a multi-byte UTF-8 name",
            Prov::DeviceInfo(device_info(b"rlm-3f9a2c", "œil-ü".as_bytes())),
            2,
        ),
        (
            "PROVISION, one address",
            Prov::Provision(provision_of(b"rlm-3f9a2c", &[b"192.168.1.20:7447"])),
            0,
        ),
        (
            "PROVISION, IPv6 and a name, highest port",
            Prov::Provision(provision_of(
                b"rlm-3f9a2c",
                &[b"[fe80::1]:65532", b"robot.local:1"],
            )),
            1,
        ),
        (
            "PROVISION, a 63-byte address",
            Prov::Provision(provision_of(b"r", &[&longest])),
            2,
        ),
        (
            "PROVISION, the largest payload (252 bytes)",
            Prov::Provision(provision_of(&id32, &full_refs)),
            3,
        ),
        (
            "PROVISIONED, stored",
            Prov::Provisioned {
                status: status::STORED,
                record_crc: 0xDEAD_BEEF,
            },
            4,
        ),
        (
            "PROVISIONED, an unknown status",
            Prov::Provisioned {
                status: 9,
                record_crc: 0,
            },
            5,
        ),
    ];
    let mut buf = [0_u8; 252];
    let largest = provision::payload(&cases[7].1, &mut buf).expect("fits");
    assert_eq!(largest, 252, "the largest case is the largest payload");
    cases
        .into_iter()
        .map(|(name, m, seq)| {
            let wire = prov_wire(&m, seq);
            json!({ "name": name, "seq": seq, "msg": prov_json(&m), "wire": hex(&wire) })
        })
        .collect()
}

/// A valid PROVISION payload for `host_id` and `addresses`, by hand.
fn provision_payload(flags: u8, host_id: &[u8], addresses: &[&[u8]]) -> Vec<u8> {
    let mut p = vec![provision::VERSION, flags];
    p.extend((0..32).map(|i| 0x50 + i as u8));
    p.push(host_id.len() as u8);
    p.extend_from_slice(host_id);
    p.push(addresses.len() as u8);
    for a in addresses {
        p.push(a.len() as u8);
        p.extend_from_slice(a);
    }
    p
}

fn device_info_payload(flags: u8, host_id: &[u8], name: &[u8]) -> Vec<u8> {
    let mut p = provision::MAGIC.to_vec();
    p.extend_from_slice(&[provision::VERSION, 0x00, 0x02, 4, flags]);
    p.extend((0..32).map(|i| 0xA0 ^ i as u8));
    p.push(host_id.len() as u8);
    p.extend_from_slice(host_id);
    p.push(name.len() as u8);
    p.extend_from_slice(name);
    p
}

fn provision_rejection_cases() -> Vec<Value> {
    use MsgError as E;
    use Receiver::{Board, Bridge};
    use provision::ty as pt;
    let a: &[u8] = b"192.168.1.20:7447";
    let ok = provision_payload(0, b"rlm", &[a]);
    let mut short = ok.clone();
    short.pop();
    let mut long = ok.clone();
    long.push(0);
    let cases: Vec<(&str, Receiver, u8, Vec<u8>, MsgError)> = vec![
        // Direction and type.
        (
            "PROVISION heard by the daemon",
            Bridge,
            pt::PROVISION,
            ok.clone(),
            E::Direction,
        ),
        (
            "DEVICE_INFO heard by the board",
            Board,
            pt::DEVICE_INFO,
            device_info_payload(0, b"", b"e"),
            E::Direction,
        ),
        (
            "an unknown daemon type",
            Board,
            0x92,
            vec![0],
            E::UnknownType,
        ),
        // GET_DEVICE.
        (
            "GET_DEVICE, the tether's magic",
            Board,
            pt::GET_DEVICE,
            b"RLMT\x01".to_vec(),
            E::Magic,
        ),
        (
            "GET_DEVICE, too short for the magic",
            Board,
            pt::GET_DEVICE,
            b"RLM".to_vec(),
            E::Magic,
        ),
        (
            "GET_DEVICE, version 2",
            Board,
            pt::GET_DEVICE,
            b"RLMP\x02".to_vec(),
            E::Version(2),
        ),
        (
            "GET_DEVICE, a byte too long",
            Board,
            pt::GET_DEVICE,
            b"RLMP\x01\x00".to_vec(),
            E::Length,
        ),
        // DEVICE_INFO.
        (
            "DEVICE_INFO, version 2, checked before its length",
            Bridge,
            pt::DEVICE_INFO,
            b"RLMP\x02".to_vec(),
            E::Version(2),
        ),
        (
            "DEVICE_INFO, an empty name",
            Bridge,
            pt::DEVICE_INFO,
            device_info_payload(0, b"", b""),
            E::Range,
        ),
        (
            "DEVICE_INFO, a 33-byte name",
            Bridge,
            pt::DEVICE_INFO,
            device_info_payload(0, b"", &letters(33)),
            E::Range,
        ),
        (
            "DEVICE_INFO, a 33-byte host id",
            Bridge,
            pt::DEVICE_INFO,
            device_info_payload(1, &letters(33), b"e"),
            E::Range,
        ),
        (
            "DEVICE_INFO, a reserved flag",
            Bridge,
            pt::DEVICE_INFO,
            device_info_payload(2, b"", b"e"),
            E::Reserved,
        ),
        (
            "DEVICE_INFO, a non-UTF-8 name",
            Bridge,
            pt::DEVICE_INFO,
            device_info_payload(0, b"", &[0xFF]),
            E::Utf8,
        ),
        (
            "DEVICE_INFO, cut inside the key",
            Bridge,
            pt::DEVICE_INFO,
            device_info_payload(0, b"", b"e")[..30].to_vec(),
            E::Length,
        ),
        // PROVISIONED.
        (
            "PROVISIONED, a byte short",
            Bridge,
            pt::PROVISIONED,
            vec![0; 4],
            E::Length,
        ),
        // PROVISION.
        (
            "PROVISION, version 2",
            Board,
            pt::PROVISION,
            {
                let mut p = ok.clone();
                p[0] = 2;
                p
            },
            E::Version(2),
        ),
        ("PROVISION, empty", Board, pt::PROVISION, vec![], E::Length),
        (
            "PROVISION, cut inside the last address",
            Board,
            pt::PROVISION,
            short,
            E::Length,
        ),
        (
            "PROVISION, a byte past the last address",
            Board,
            pt::PROVISION,
            long,
            E::Length,
        ),
        (
            "PROVISION, a reserved flag",
            Board,
            pt::PROVISION,
            provision_payload(1, b"rlm", &[a]),
            E::Reserved,
        ),
        (
            "PROVISION, an empty host id",
            Board,
            pt::PROVISION,
            provision_payload(0, b"", &[a]),
            E::Range,
        ),
        (
            "PROVISION, a 33-byte host id",
            Board,
            pt::PROVISION,
            provision_payload(0, &letters(33), &[a]),
            E::Range,
        ),
        (
            "PROVISION, no address",
            Board,
            pt::PROVISION,
            provision_payload(0, b"rlm", &[]),
            E::Range,
        ),
        (
            "PROVISION, five addresses",
            Board,
            pt::PROVISION,
            provision_payload(0, b"rlm", &[a, a, a, a, a]),
            E::Range,
        ),
        (
            "PROVISION, an empty address",
            Board,
            pt::PROVISION,
            provision_payload(0, b"rlm", &[b""]),
            E::Range,
        ),
        (
            "PROVISION, a 64-byte address",
            Board,
            pt::PROVISION,
            provision_payload(0, b"rlm", &[&address(64)]),
            E::Range,
        ),
        (
            "PROVISION, port 0",
            Board,
            pt::PROVISION,
            provision_payload(0, b"rlm", &[b"h:0"]),
            E::Range,
        ),
        (
            "PROVISION, port 65533",
            Board,
            pt::PROVISION,
            provision_payload(0, b"rlm", &[b"h:65533"]),
            E::Range,
        ),
        (
            "PROVISION, no port",
            Board,
            pt::PROVISION,
            provision_payload(0, b"rlm", &[b"192.168.1.20"]),
            E::Range,
        ),
        (
            "PROVISION, a non-UTF-8 host id",
            Board,
            pt::PROVISION,
            provision_payload(0, &[0xFF], &[a]),
            E::Utf8,
        ),
    ];
    cases
        .into_iter()
        .map(|(name, receiver, ty, payload, error)| {
            let f = Frame {
                ty,
                seq: 0,
                payload: &payload,
            };
            assert_eq!(provision::decode(&f, receiver), Err(error), "{name}");
            let mut v = json!({
                "name": name,
                "receiver": receiver_name(receiver),
                "type": ty,
                "payload": hex(&payload),
            });
            v.as_object_mut()
                .expect("object")
                .extend(msg_error_json(error));
            v
        })
        .collect()
}

fn provision_write_rejection_cases() -> Vec<Value> {
    use MsgError as E;
    let id33 = letters(33);
    let a: &[u8] = b"192.168.1.20:7447";
    // A 253-byte payload: one past the frame.
    let over = provision_addresses(256);
    let over_refs: Vec<&[u8]> = over.iter().map(Vec::as_slice).collect();
    let id32 = letters(MAX_ID);
    let cases: Vec<(&str, Prov<'_>, MsgError)> = vec![
        (
            "DEVICE_INFO, an empty name",
            Prov::DeviceInfo(device_info(b"", b"")),
            E::Range,
        ),
        (
            "DEVICE_INFO, a 33-byte host id",
            Prov::DeviceInfo(device_info(&id33, b"e")),
            E::Range,
        ),
        (
            "DEVICE_INFO, a reserved flag",
            Prov::DeviceInfo(DeviceInfo {
                flags: 2,
                ..device_info(b"", b"e")
            }),
            E::Reserved,
        ),
        (
            "PROVISION, a reserved flag",
            Prov::Provision(Provision {
                flags: 1,
                ..provision_of(b"rlm", &[a])
            }),
            E::Reserved,
        ),
        (
            "PROVISION, no address",
            Prov::Provision(provision_of(b"rlm", &[])),
            E::Range,
        ),
        (
            "PROVISION, port 65533",
            Prov::Provision(provision_of(b"rlm", &[b"h:65533"])),
            E::Range,
        ),
        (
            "PROVISION, an empty host id",
            Prov::Provision(provision_of(b"", &[a])),
            E::Range,
        ),
        (
            "PROVISION, a 253-byte payload",
            Prov::Provision(provision_of(&id32, &over_refs)),
            E::Length,
        ),
    ];
    assert_eq!(
        provision::payload(&cases[7].1, &mut [0; 252]),
        Err(E::Length),
        "the oversized case is over by one"
    );
    cases
        .into_iter()
        .map(|(name, m, error)| {
            let mut wrote = false;
            assert_eq!(
                provision::write(&m, 0, |_| wrote = true),
                Err(error),
                "{name}"
            );
            assert!(!wrote, "{name} writes nothing");
            json!({ "name": name, "msg": prov_json(&m), "error": msg_error_name(error) })
        })
        .collect()
}

#[test]
fn the_file_matches_this_build() {
    let fresh = generate();
    if std::env::var_os(REGENERATE).is_some() {
        let mut text = serde_json::to_string_pretty(&fresh).expect("json");
        text.push('\n');
        std::fs::write(path(), text).expect("write");
    }
    assert_eq!(fresh, load(), "regenerate only when the contract changes");
}

// ── Checking the file ────────────────────────────────────────────────────
//
// These read only the file, as another implementation would, so a vector
// edited by hand is checked too.

fn cases<'a>(all: &'a Value, section: &str) -> &'a [Value] {
    all[section].as_array().expect("a section")
}

#[test]
fn crc_vectors() {
    for v in cases(&load(), "crc") {
        let data = unhex(&v["data"]);
        assert_eq!(crc::crc16(&data), int::<u16>(&v["crc16"]), "{}", v["name"]);
        assert_eq!(crc::crc32(&data), int::<u32>(&v["crc32"]), "{}", v["name"]);
    }
}

#[test]
fn cobs_vectors() {
    let all = load();
    for v in cases(&all, "cobs_encode") {
        let decoded = unhex(&v["decoded"]);
        let encoded = unhex(&v["encoded"]);
        assert_eq!(encode(&decoded), encoded, "{}", v["name"]);
        // Read as pieces, the same bytes encode alike.
        let (a, rest) = decoded.split_at(decoded.len() / 3);
        let (b, c) = rest.split_at(rest.len() / 2);
        let mut pieces = Vec::new();
        cobs::encode(&[a, &[], b, c], |x| pieces.push(x));
        assert_eq!(pieces, encoded, "{} in pieces", v["name"]);
        let got = decode_stream(&cat(&[&encoded, &[0]]), 512);
        assert_eq!(got, [Ok(decoded)], "{} decodes back", v["name"]);
    }
    for v in cases(&all, "cobs_decode") {
        let got: Vec<Value> = decode_stream(&unhex(&v["wire"]), int(&v["rx_max"]))
            .iter()
            .map(cobs_result_json)
            .collect();
        assert_eq!(
            &got,
            v["results"].as_array().expect("results"),
            "{}",
            v["name"]
        );
    }
}

#[test]
fn frame_vectors() {
    for v in cases(&load(), "frames") {
        let got: Vec<Value> = decode_stream(&unhex(&v["wire"]), int(&v["rx_max"]))
            .iter()
            .map(frame_result_json)
            .collect();
        assert_eq!(
            &got,
            v["results"].as_array().expect("results"),
            "{}",
            v["name"]
        );
    }
}

#[test]
fn message_vectors() {
    for v in cases(&load(), "messages") {
        let name = &v["name"];
        let want = unhex(&v["wire"]);
        let seq = int(&v["seq"]);
        // Encode from the fields.
        with_msg(&v["msg"], |m| assert_eq!(wire(&m, seq), want, "{name}"));
        // Decode from the wire.
        let bytes = decode_stream(&want, 256).remove(0).expect("decodes");
        let f = Frame::parse(&bytes).expect("frame");
        assert_eq!(f.seq, seq, "{name}");
        let m = msg::decode(&f, receiver_for(f.ty), dense_lookup(v)).expect("valid");
        assert_eq!(msg_json(&m), v["msg"], "{name}");
    }
}

#[test]
fn rejection_vectors() {
    for v in cases(&load(), "rejections") {
        let payload = unhex(&v["payload"]);
        let f = Frame {
            ty: int(&v["type"]),
            seq: 0,
            payload: &payload,
        };
        let err =
            msg::decode(&f, receiver_from(&v["receiver"]), dense_lookup(v)).expect_err("refused");
        let mut want = msg_error_json(err);
        want.insert("name".into(), v["name"].clone());
        let mut got = v.as_object().expect("object").clone();
        got.retain(|k, _| matches!(k.as_str(), "name" | "error" | "version"));
        assert_eq!(got, want);
    }
    for v in cases(&load(), "write_rejections") {
        let mut wrote = false;
        let err =
            with_msg(&v["msg"], |m| msg::write(&m, 0, |_| wrote = true)).expect_err("refused");
        assert!(!wrote, "{}", v["name"]);
        assert_eq!(text(&v["error"]), msg_error_name(err), "{}", v["name"]);
    }
}

#[test]
fn provision_vectors() {
    let all = load();
    for v in cases(&all, "provision") {
        let name = &v["name"];
        let want = unhex(&v["wire"]);
        let seq = int(&v["seq"]);
        with_prov(&v["msg"], |m| {
            let mut out = Vec::new();
            provision::write(&m, seq, |b| out.push(b)).expect("writes");
            assert_eq!(out, want, "{name}");
        });
        // The leading delimiter yields nothing; then one frame.
        assert_eq!(want[0], 0, "{name} starts with a delimiter");
        let bytes = decode_stream(&want, 256).remove(0).expect("decodes");
        let f = Frame::parse(&bytes).expect("frame");
        assert_eq!(f.seq, seq, "{name}");
        let m = provision::decode(&f, receiver_for(f.ty)).expect("valid");
        assert_eq!(prov_json(&m), v["msg"], "{name}");
    }
    for v in cases(&all, "provision_rejections") {
        let payload = unhex(&v["payload"]);
        let f = Frame {
            ty: int(&v["type"]),
            seq: 0,
            payload: &payload,
        };
        let err = provision::decode(&f, receiver_from(&v["receiver"])).expect_err("refused");
        let mut want = msg_error_json(err);
        want.insert("name".into(), v["name"].clone());
        let mut got = v.as_object().expect("object").clone();
        got.retain(|k, _| matches!(k.as_str(), "name" | "error" | "version"));
        assert_eq!(got, want);
    }
    for v in cases(&all, "provision_write_rejections") {
        let mut wrote = false;
        let err = with_prov(&v["msg"], |m| provision::write(&m, 0, |_| wrote = true))
            .expect_err("refused");
        assert!(!wrote, "{}", v["name"]);
        assert_eq!(text(&v["error"]), msg_error_name(err), "{}", v["name"]);
    }
}

#[test]
fn pack_vectors() {
    for v in cases(&load(), "pack") {
        let offsets: Vec<usize> = v["offsets"]
            .as_array()
            .expect("offsets")
            .iter()
            .map(int)
            .collect();
        let lens: Vec<usize> = v["lens"]
            .as_array()
            .expect("lens")
            .iter()
            .map(int)
            .collect();
        let aligned = unhex(&v["aligned"]);
        let dense = unhex(&v["dense"]);
        let mut got = vec![0; dense.len()];
        pack::to_dense(&aligned, &offsets, &lens, &mut got).expect("dense");
        assert_eq!(got, dense, "{}", v["name"]);
        let mut got = vec![0xEE; aligned.len()];
        pack::to_aligned(&dense, &offsets, &lens, &mut got).expect("aligned");
        assert_eq!(got, aligned, "{}", v["name"]);
    }
}

/// Values written out by hand, not taken from this crate, so a bug shared
/// by the crate and its generated vectors still shows.
#[test]
fn literal_values_not_from_the_crate() {
    // The zlib CRC-32 of the pangram, and the CRC catalogue's CRC-16 check.
    assert_eq!(
        crc::crc32(b"The quick brown fox jumps over the lazy dog"),
        0x414F_A339
    );
    assert_eq!(crc::crc16(b"123456789"), 0x29B1);
    // The COBS reference example: a zero inside the data.
    assert_eq!(
        encode(&[0x11, 0x22, 0x00, 0x33]),
        [0x03, 0x11, 0x22, 0x02, 0x33]
    );
    // One whole HELLO frame, field by field (spec §4.1), CRC and COBS
    // computed by a separate implementation.
    let decoded: &[u8] = &[
        0x01, 0x00, // type HELLO, seq 0
        b'R', b'L', b'M', b'T', 0x01, // magic, version
        0x00, 0x01, 0x01, // lib_version 0x0100, arch AVR
        0xB0, 0xB1, 0x00, 0xB3, 0xB4, 0x00, 0x00, 0xB7, // board_id
        0x01, 0x02, 0x00, 0x00, // boot 0x0201, gen 0, flags 0
        0x58, 0x02, 0xEF, 0xBE, 0xAD, 0xDE, // map_len 600, map_crc
        0xFA, 0x00, 0x40, 0x00, // watchdog_ms 250, rx_max 64
        0x05, b'e', b'y', b'e', b'-', b'1', // name
        0x06, 0xC9, // CRC-16 0xC906, little-endian
    ];
    let wire: &[u8] = &[
        0x02, 0x01, 0x06, b'R', b'L', b'M', b'T', 0x01, 0x05, 0x01, 0x01, 0xB0, 0xB1, 0x03, 0xB3,
        0xB4, 0x01, 0x04, 0xB7, 0x01, 0x02, 0x01, 0x08, 0x58, 0x02, 0xEF, 0xBE, 0xAD, 0xDE, 0xFA,
        0x02, 0x40, 0x09, 0x05, b'e', b'y', b'e', b'-', b'1', 0x06, 0xC9, 0x00,
    ];
    assert_eq!(crc::crc16(&decoded[..decoded.len() - 2]), 0xC906);
    assert_eq!(cat(&[&encode(decoded), &[0]]), wire);
    assert_eq!(
        crate::wire(&Msg::Hello(hello(b"eye-1")), 0),
        wire,
        "the crate writes it"
    );
    let all = load();
    let hello = cases(&all, "messages")
        .iter()
        .find(|v| v["name"] == "HELLO")
        .expect("the HELLO vector");
    assert_eq!(unhex(&hello["wire"]), wire);
    // A GET_DEVICE (spec §14.3) with its leading delimiter, CRC-16 0x5C31
    // and COBS computed by a separate implementation.
    let get_device: &[u8] = &[
        0x00, 0x02, 0x90, 0x08, b'R', b'L', b'M', b'P', 0x01, 0x31, 0x5C, 0x00,
    ];
    assert_eq!(prov_wire(&Prov::GetDevice, 0), get_device);
    let vector = cases(&all, "provision")
        .iter()
        .find(|v| v["name"] == "GET_DEVICE")
        .expect("the GET_DEVICE vector");
    assert_eq!(unhex(&vector["wire"]), get_device);
}
