//! The browser framing: one tag byte, then the v2 message the session peer
//! sent (or will receive), unchanged. The `rlmesh-wasm` crate defines the
//! same tags (its `frame` module documents the table); this crate's tests
//! check that they agree.

/// An authority snapshot (host → browser).
pub const STATE: u8 = 0x01;
/// A heartbeat echo (host → browser).
pub const ECHO: u8 = 0x02;
/// A `ctl` reply (host → browser), after the request id (`u32` LE).
pub const REPLY: u8 = 0x03;
/// A `ctl` request (browser → host), after the request id (`u32` LE).
pub const REQUEST: u8 = 0x10;
/// A heartbeat (browser → host).
pub const HEARTBEAT: u8 = 0x11;
/// An action chunk (browser → edge).
pub const CHUNK: u8 = 0x12;
/// One sample of one subscription: the subscription id (`u32` LE), a flags
/// byte (reserved, 0), then the `Sample`.
pub const SAMPLE: u8 = 0x20;

/// `tag` then `body` (one copy: a WebSocket message is one buffer).
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

/// A `SAMPLE` frame for subscription `id`.
pub fn sample(id: u32, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(body.len() + 6);
    out.push(SAMPLE);
    out.extend_from_slice(&id.to_le_bytes());
    out.push(0);
    out.extend_from_slice(body);
    out
}

/// The `u32` after the tag.
pub fn read_id(frame: &[u8]) -> Option<u32> {
    let bytes: [u8; 4] = frame.get(1..5)?.try_into().ok()?;
    Some(u32::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_match_the_browser_binding() {
        use rlmesh_wasm::frame as browser;
        assert_eq!(
            [STATE, ECHO, REPLY, REQUEST, HEARTBEAT, CHUNK, SAMPLE],
            [
                browser::STATE,
                browser::ECHO,
                browser::REPLY,
                browser::REQUEST,
                browser::HEARTBEAT,
                browser::CHUNK,
                browser::SAMPLE
            ]
        );
        assert_eq!(sample(7, &[9]), [SAMPLE, 7, 0, 0, 0, 0, 9]);
        assert_eq!(sample(7, &[]).len(), browser::SAMPLE_HEADER);
        assert_eq!(read_id(&with_id(REPLY, 513, &[])), Some(513));
        assert_eq!(read_id(&[REPLY, 1]), None);
    }
}
