//! The gateway framing: how v2 messages ride the browser's WebSockets.
//!
//! Binary frames are one tag byte and the message, so the gateway forwards
//! the bytes a session peer sent without re-encoding them. Text frames are
//! the gateway's own JSON (hello, subscribe, errors). `rlmesh-gateway`
//! defines the same tags and checks them against these in its tests.
//!
//! | tag | socket | direction | body |
//! |---|---|---|---|
//! | [`STATE`] | control | gateway → browser | an `AuthoritySnapshot` |
//! | [`ECHO`] | control | gateway → browser | a `HeartbeatEcho` |
//! | [`REPLY`] | control | gateway → browser | `u32` request id (LE), a `CtlReply` |
//! | [`REQUEST`] | control | browser → gateway | `u32` request id (LE), a `Request` |
//! | [`HEARTBEAT`] | control | browser → gateway | a `Heartbeat` |
//! | [`CHUNK`] | control | browser → gateway | an `ActionChunk` |
//! | [`SAMPLE`] | state, bulk | gateway → browser | `u32` subscription id (LE), flags (reserved, 0), a `Sample` |

/// An authority snapshot (host → browser).
pub const STATE: u8 = 0x01;
/// A heartbeat echo (host → browser).
pub const ECHO: u8 = 0x02;
/// A `ctl` reply (host → browser), after the request id.
pub const REPLY: u8 = 0x03;
/// A `ctl` request (browser → host), after the request id.
pub const REQUEST: u8 = 0x10;
/// A heartbeat (browser → host).
pub const HEARTBEAT: u8 = 0x11;
/// An action chunk (browser → edge).
pub const CHUNK: u8 = 0x12;
/// One sample of one subscription (publisher → browser).
pub const SAMPLE: u8 = 0x20;

/// The `SAMPLE` header: tag, subscription id, flags.
pub const SAMPLE_HEADER: usize = 6;

/// `tag` then `body`.
pub fn tagged(tag: u8, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(body.len() + 1);
    out.push(tag);
    out.extend_from_slice(body);
    out
}

/// `tag`, a little-endian `id`, then `body`.
pub fn with_id(tag: u8, id: u32, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(body.len() + 5);
    out.push(tag);
    out.extend_from_slice(&id.to_le_bytes());
    out.extend_from_slice(body);
    out
}

/// The `u32` after the tag.
pub fn read_id(frame: &[u8]) -> Option<u32> {
    let bytes: [u8; 4] = frame.get(1..5)?.try_into().ok()?;
    Some(u32::from_le_bytes(bytes))
}
