//! The board's half of USB provisioning (tether spec §14), sans-IO: bytes
//! from the serial port go in, and out come replies to write back, or a
//! record for the platform to store before the board confirms it. The
//! messages are `rlmesh-tether`'s, so the board and the daemon share one
//! codec and its vectors.

use alloc::string::String;
use alloc::vec::Vec;

use rlmesh_session_core::crypto::StaticKeypair;
use rlmesh_tether::cobs::Decoder;
use rlmesh_tether::frame::{Frame, MAX_DECODED};
use rlmesh_tether::msg::Receiver;
use rlmesh_tether::provision::{self, DeviceInfo, Prov, device_flags, status, ty};
use zeroize::Zeroizing;

/// A stored provisioning record: the `PROVISION` payload as received, and
/// what it says.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record {
    /// The payload, as the board stores it (one NVS blob).
    pub payload: Vec<u8>,
    /// The home host's static X25519 key.
    pub host_key: [u8; 32],
    /// Its host id.
    pub host_id: String,
    /// Its base addresses, `host:port`, in order.
    pub addresses: Vec<String>,
}

impl Record {
    /// Reads a stored payload, or `None` when it isn't a valid `PROVISION`.
    pub fn parse(payload: &[u8]) -> Option<Self> {
        let frame = Frame {
            ty: ty::PROVISION,
            seq: 0,
            payload,
        };
        let Ok(Prov::Provision(p)) = provision::decode(&frame, Receiver::Board) else {
            return None;
        };
        Some(Self {
            payload: payload.to_vec(),
            host_key: p.host_key,
            host_id: String::from(core::str::from_utf8(p.host_id).ok()?),
            addresses: p
                .addresses
                .as_slice()
                .iter()
                .map(|a| core::str::from_utf8(a).ok().map(String::from))
                .collect::<Option<_>>()?,
        })
    }
}

/// What a byte from the serial port asks of the platform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Step {
    /// Write these bytes to the serial port.
    Reply(Vec<u8>),
    /// Store this record (one atomic write), then call
    /// [`Responder::stored`] with the outcome.
    Store(Record),
}

/// The responder: one per serial port, for the life of the sketch.
pub struct Responder {
    decoder: Decoder<MAX_DECODED>,
    seq: u8,
    name: String,
    lib_version: u16,
    arch: u8,
    public: [u8; 32],
    host_id: Option<String>,
    /// The CRC of the record being stored.
    storing: Option<u32>,
}

impl core::fmt::Debug for Responder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Responder")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

/// The public key of the device key `private`.
pub fn public_key(private: &[u8; 32]) -> [u8; 32] {
    let mut copy = Zeroizing::new(*private);
    *StaticKeypair::from_private(&mut copy).public()
}

impl Responder {
    /// A responder for robot `name` with device key `private`, holding
    /// `record` if provisioned. `arch` as in `HELLO`.
    pub fn new(name: &str, private: &[u8; 32], arch: u8, record: Option<&Record>) -> Self {
        let v = |s: &str| s.parse::<u16>().unwrap_or(0);
        Self {
            decoder: Decoder::new(),
            seq: 0,
            name: String::from(name),
            // Major and minor, a byte each.
            lib_version: (v(env!("CARGO_PKG_VERSION_MAJOR")) << 8)
                | v(env!("CARGO_PKG_VERSION_MINOR")),
            arch,
            public: public_key(private),
            host_id: record.map(|r| r.host_id.clone()),
            storing: None,
        }
    }

    /// One byte from the serial port: what to do about it, if anything.
    /// Everything that isn't a valid `GET_DEVICE` or `PROVISION` is
    /// ignored (§14.4), the sketch's own text included.
    pub fn push(&mut self, byte: u8) -> Option<Step> {
        let decoded = match self.decoder.push(byte) {
            Some(Ok(decoded)) => decoded.to_vec(),
            _ => return None,
        };
        let frame = Frame::parse(&decoded).ok()?;
        match provision::decode(&frame, Receiver::Board) {
            Ok(Prov::GetDevice) => {
                let host_id = self.host_id.clone().unwrap_or_default();
                let info = Prov::DeviceInfo(DeviceInfo {
                    lib_version: self.lib_version,
                    arch: self.arch,
                    flags: if self.host_id.is_some() {
                        device_flags::PROVISIONED
                    } else {
                        0
                    },
                    device_key: self.public,
                    host_id: host_id.as_bytes(),
                    name: self.name.as_bytes(),
                });
                Some(Step::Reply(write(&mut self.seq, &info)))
            }
            Ok(Prov::Provision(_)) => {
                let payload = frame.payload;
                let Some(record) = Record::parse(payload) else {
                    return Some(Step::Reply(self.reply(status::INVALID, 0)));
                };
                self.storing = Some(provision::record_crc(payload));
                Some(Step::Store(record))
            }
            // A PROVISION out of range: refused, nothing written.
            Err(_) if frame.ty == ty::PROVISION => {
                Some(Step::Reply(self.reply(status::INVALID, 0)))
            }
            _ => None,
        }
    }

    /// The platform stored the record from [`Step::Store`] (`ok`), or
    /// couldn't: the `PROVISIONED` to write.
    pub fn stored(&mut self, record: &Record, ok: bool) -> Vec<u8> {
        let crc = self.storing.take().unwrap_or(0);
        if ok {
            self.host_id = Some(record.host_id.clone());
            self.reply(status::STORED, crc)
        } else {
            self.reply(status::STORAGE_FAILED, crc)
        }
    }

    fn reply(&mut self, status: u8, record_crc: u32) -> Vec<u8> {
        write(&mut self.seq, &Prov::Provisioned { status, record_crc })
    }
}

/// `msg` as one frame with the next `seq`.
fn write(seq: &mut u8, msg: &Prov<'_>) -> Vec<u8> {
    let mut out = Vec::new();
    // Every field was checked by decoding or is this board's own.
    let _ = provision::write(msg, *seq, |b| out.push(b));
    *seq = seq.wrapping_add(1);
    out
}

#[cfg(test)]
mod tests {
    use rlmesh_tether::provision::{Addresses, Provision};

    use super::*;

    fn feed(r: &mut Responder, wire: &[u8]) -> Vec<Step> {
        wire.iter().filter_map(|&b| r.push(b)).collect()
    }

    fn decode(wire: &[u8]) -> (u8, Vec<u8>) {
        let mut d = Decoder::<MAX_DECODED>::new();
        let decoded = wire
            .iter()
            .find_map(|&b| d.push(b).and_then(Result::ok).map(<[u8]>::to_vec))
            .expect("a frame");
        let f = Frame::parse(&decoded).expect("valid");
        (f.ty, f.payload.to_vec())
    }

    fn send(msg: &Prov<'_>) -> Vec<u8> {
        let mut out = Vec::new();
        provision::write(msg, 0, |b| out.push(b)).unwrap();
        out
    }

    #[test]
    fn a_board_identifies_then_stores_and_confirms_its_record() {
        let private = [7; 32];
        let mut board = Responder::new("eye-3", &private, 4, None);
        // The sketch's text before the frame changes nothing.
        let mut wire = b"booted\r\n".to_vec();
        wire.extend(send(&Prov::GetDevice));
        let steps = feed(&mut board, &wire);
        let [Step::Reply(reply)] = &steps[..] else {
            panic!("{steps:?}");
        };
        let (t, payload) = decode(reply);
        let f = Frame {
            ty: t,
            seq: 0,
            payload: &payload,
        };
        let Ok(Prov::DeviceInfo(info)) = provision::decode(&f, Receiver::Bridge) else {
            panic!("DEVICE_INFO");
        };
        assert_eq!(info.name, b"eye-3");
        assert_eq!(info.flags, 0);
        assert_eq!(info.device_key, public_key(&private));

        let p = Prov::Provision(Provision {
            flags: 0,
            host_key: [9; 32],
            host_id: b"rlm-3f9a2c",
            addresses: Addresses::new(&[b"192.168.1.20:7447"]).unwrap(),
        });
        let wire = send(&p);
        let steps = feed(&mut board, &wire);
        let [Step::Store(record)] = &steps[..] else {
            panic!("{steps:?}");
        };
        assert_eq!(record.host_id, "rlm-3f9a2c");
        assert_eq!(record.addresses, ["192.168.1.20:7447"]);
        assert_eq!(Record::parse(&record.payload).as_ref(), Some(record));
        let (t, payload) = decode(&board.stored(record, true));
        assert_eq!(t, ty::PROVISIONED);
        assert_eq!(payload[0], status::STORED);
        let mut buf = [0; 252];
        let n = provision::payload(&p, &mut buf).unwrap();
        assert_eq!(payload[1..], provision::record_crc(&buf[..n]).to_le_bytes());

        // Provisioned now: DEVICE_INFO says for whom.
        let steps = feed(&mut board, &send(&Prov::GetDevice));
        let [Step::Reply(reply)] = &steps[..] else {
            panic!("{steps:?}");
        };
        let (t, payload) = decode(reply);
        let f = Frame {
            ty: t,
            seq: 0,
            payload: &payload,
        };
        let Ok(Prov::DeviceInfo(info)) = provision::decode(&f, Receiver::Bridge) else {
            panic!("DEVICE_INFO");
        };
        assert_eq!(
            (info.flags, info.host_id),
            (device_flags::PROVISIONED, &b"rlm-3f9a2c"[..])
        );
    }

    #[test]
    fn a_failed_store_and_an_invalid_record_are_reported() {
        let mut board = Responder::new("eye-3", &[7; 32], 4, None);
        let record = Record::parse(&{
            let mut buf = [0; 252];
            let p = Prov::Provision(Provision {
                flags: 0,
                host_key: [9; 32],
                host_id: b"h",
                addresses: Addresses::new(&[b"h:1"]).unwrap(),
            });
            let n = provision::payload(&p, &mut buf).unwrap();
            buf[..n].to_vec()
        })
        .unwrap();
        let (_, payload) = decode(&board.stored(&record, false));
        assert_eq!(payload[0], status::STORAGE_FAILED);
        // A PROVISION with port 0, framed by hand: refused, nothing stored.
        let mut p = alloc::vec![provision::VERSION, 0];
        p.extend([9; 32]);
        p.extend([1, b'h', 1, 3, b'h', b':', b'0']);
        let mut wire = alloc::vec![0];
        rlmesh_tether::frame::write(ty::PROVISION, 0, &[&p], |b| wire.push(b));
        let steps = feed(&mut board, &wire);
        let [Step::Reply(reply)] = &steps[..] else {
            panic!("{steps:?}");
        };
        let (_, payload) = decode(reply);
        assert_eq!(payload, [status::INVALID, 0, 0, 0, 0]);
    }

    #[test]
    fn tether_frames_and_noise_are_ignored() {
        let mut board = Responder::new("eye-3", &[7; 32], 4, None);
        let mut ping = alloc::vec![0];
        rlmesh_tether::frame::write(rlmesh_tether::msg::ty::PING, 0, &[&[1]], |b| ping.push(b));
        assert!(feed(&mut board, &ping).is_empty());
        assert!(feed(&mut board, b"\x00garbage\x00\x01\x00").is_empty());
    }
}
