//! USB provisioning of a networked board (spec §14): the tether's framing
//! carries a device key one way and the home host's details the other.
//!
//! Errors and receivers are the tether's ([`MsgError`], [`Receiver`]): the
//! daemon receives as the bridge does. Every frame goes out with a leading
//! `0x00` ([`write()`]), since the port also carries the sketch's text.

use crate::frame::{self, Frame, MAX_PAYLOAD};
use crate::msg::{MsgError, Receiver};

/// `GET_DEVICE`'s and `DEVICE_INFO`'s magic.
pub const MAGIC: [u8; 4] = *b"RLMP";
/// The provisioning version this crate speaks.
pub const VERSION: u8 = 1;
/// The longest host id or robot name.
pub const MAX_ID: usize = 32;
/// The most addresses in a `PROVISION`.
pub const MAX_ADDRESSES: usize = 4;
/// The longest address.
pub const MAX_ADDRESS: usize = 63;
/// The highest base port: it has three class ports above it.
pub const MAX_PORT: u16 = u16::MAX - 3;

/// Message types.
pub mod ty {
    /// Board → daemon: the device key and what it holds.
    pub const DEVICE_INFO: u8 = 0x10;
    /// Board → daemon: the outcome of a `PROVISION`.
    pub const PROVISIONED: u8 = 0x11;
    /// Daemon → board: identify.
    pub const GET_DEVICE: u8 = 0x90;
    /// Daemon → board: store the home host's details.
    pub const PROVISION: u8 = 0x91;
}

/// `DEVICE_INFO` flags.
pub mod device_flags {
    /// The board holds a provisioning record.
    pub const PROVISIONED: u8 = 1 << 0;
    /// Every defined bit.
    pub const ALL: u8 = PROVISIONED;
}

/// `PROVISIONED.status`.
pub mod status {
    /// Stored.
    pub const STORED: u8 = 0;
    /// A field out of range: nothing written.
    pub const INVALID: u8 = 1;
    /// Storage failed: nothing written, or the old record kept.
    pub const STORAGE_FAILED: u8 = 2;
}

/// The board's identification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceInfo<'a> {
    /// The library's version.
    pub lib_version: u16,
    /// As in `HELLO`: 0 unknown, 1 AVR, 2 ARM Cortex-M, 3 RISC-V, 4 Xtensa.
    pub arch: u8,
    /// [`device_flags`].
    pub flags: u8,
    /// The X25519 public key.
    pub device_key: [u8; 32],
    /// The stored host id, empty when not provisioned.
    pub host_id: &'a [u8],
    /// The compiled robot name, UTF-8.
    pub name: &'a [u8],
}

/// 1 to [`MAX_ADDRESSES`] base addresses, `host:port`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Addresses<'a> {
    list: [&'a [u8]; MAX_ADDRESSES],
    count: usize,
}

impl<'a> Addresses<'a> {
    /// The first [`MAX_ADDRESSES`] of `list`, or `None` past that.
    pub fn new(list: &[&'a [u8]]) -> Option<Self> {
        if list.len() > MAX_ADDRESSES {
            return None;
        }
        let mut out = [&[][..]; MAX_ADDRESSES];
        out[..list.len()].copy_from_slice(list);
        Some(Self {
            list: out,
            count: list.len(),
        })
    }

    /// The addresses, in order.
    pub fn as_slice(&self) -> &[&'a [u8]] {
        &self.list[..self.count]
    }
}

/// The home host's details.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Provision<'a> {
    /// All reserved.
    pub flags: u8,
    /// The host's static X25519 key.
    pub host_key: [u8; 32],
    /// The host id, UTF-8.
    pub host_id: &'a [u8],
    /// Its base addresses.
    pub addresses: Addresses<'a>,
}

/// One provisioning message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prov<'a> {
    /// `DEVICE_INFO`.
    DeviceInfo(DeviceInfo<'a>),
    /// `PROVISIONED`.
    Provisioned {
        /// [`status`]; unknown values are accepted.
        status: u8,
        /// CRC-32/ISO-HDLC of the `PROVISION` payload as received.
        record_crc: u32,
    },
    /// `GET_DEVICE`.
    GetDevice,
    /// `PROVISION`.
    Provision(Provision<'a>),
}

/// Whether `a` is a base address: 1 to [`MAX_ADDRESS`] bytes of printable
/// ASCII `host:port`, an IPv6 host in brackets, `port` 1 to [`MAX_PORT`].
pub fn valid_address(a: &[u8]) -> bool {
    if a.is_empty() || a.len() > MAX_ADDRESS || !a.iter().all(|b| (0x21..=0x7E).contains(b)) {
        return false;
    }
    let Ok(text) = core::str::from_utf8(a) else {
        return false;
    };
    let Some((host, port)) = text.rsplit_once(':') else {
        return false;
    };
    let host_ok = match host.strip_prefix('[') {
        Some(inner) => inner.strip_suffix(']').is_some_and(|ip| !ip.is_empty()),
        None => !host.is_empty() && !host.contains(':'),
    };
    let port_ok = !port.starts_with('0')
        && port.bytes().all(|b| b.is_ascii_digit())
        && port
            .parse::<u16>()
            .is_ok_and(|p| (1..=MAX_PORT).contains(&p));
    host_ok && port_ok
}

/// The CRC a board answers a `PROVISION` with: CRC-32/ISO-HDLC of its
/// payload.
pub fn record_crc(payload: &[u8]) -> u32 {
    crate::crc::crc32(payload)
}

fn exact(p: &[u8], n: usize) -> Result<(), MsgError> {
    if p.len() == n {
        Ok(())
    } else {
        Err(MsgError::Length)
    }
}

fn utf8(b: &[u8]) -> Result<(), MsgError> {
    core::str::from_utf8(b)
        .map(|_| ())
        .map_err(|_| MsgError::Utf8)
}

fn key(p: &[u8]) -> [u8; 32] {
    let mut k = [0; 32];
    k.copy_from_slice(&p[..32]);
    k
}

/// Decodes `frame`'s payload as received by `receiver` (the daemon is
/// [`Receiver::Bridge`]).
pub fn decode<'a>(frame: &Frame<'a>, receiver: Receiver) -> Result<Prov<'a>, MsgError> {
    let to_board = frame.ty & 0x80 != 0;
    if to_board != (receiver == Receiver::Board) {
        return Err(MsgError::Direction);
    }
    let p = frame.payload;
    match frame.ty {
        ty::GET_DEVICE => {
            if p.len() < 4 || p[..4] != MAGIC {
                return Err(MsgError::Magic);
            }
            if p.len() < 5 {
                return Err(MsgError::Length);
            }
            if p[4] != VERSION {
                return Err(MsgError::Version(p[4]));
            }
            exact(p, 5)?;
            Ok(Prov::GetDevice)
        }
        ty::DEVICE_INFO => {
            if p.len() < 4 || p[..4] != MAGIC {
                return Err(MsgError::Magic);
            }
            if p.len() < 5 {
                return Err(MsgError::Length);
            }
            if p[4] != VERSION {
                return Err(MsgError::Version(p[4]));
            }
            // magic, version, lib_version, arch, flags, key, host_id_len.
            if p.len() < 42 {
                return Err(MsgError::Length);
            }
            let host_len = usize::from(p[41]);
            if p.len() < 43 + host_len {
                return Err(MsgError::Length);
            }
            let name_len = usize::from(p[42 + host_len]);
            exact(p, 43 + host_len + name_len)?;
            let flags = p[8];
            if flags & !device_flags::ALL != 0 {
                return Err(MsgError::Reserved);
            }
            if host_len > MAX_ID || name_len == 0 || name_len > MAX_ID {
                return Err(MsgError::Range);
            }
            let host_id = &p[42..42 + host_len];
            let name = &p[43 + host_len..];
            utf8(host_id)?;
            utf8(name)?;
            Ok(Prov::DeviceInfo(DeviceInfo {
                lib_version: u16::from_le_bytes([p[5], p[6]]),
                arch: p[7],
                flags,
                device_key: key(&p[9..]),
                host_id,
                name,
            }))
        }
        ty::PROVISIONED => {
            exact(p, 5)?;
            Ok(Prov::Provisioned {
                status: p[0],
                record_crc: u32::from_le_bytes([p[1], p[2], p[3], p[4]]),
            })
        }
        ty::PROVISION => {
            if p.is_empty() {
                return Err(MsgError::Length);
            }
            if p[0] != VERSION {
                return Err(MsgError::Version(p[0]));
            }
            // version, flags, key, host_id_len.
            if p.len() < 35 {
                return Err(MsgError::Length);
            }
            let host_len = usize::from(p[34]);
            if p.len() < 36 + host_len {
                return Err(MsgError::Length);
            }
            let count = usize::from(p[35 + host_len]);
            // Walk the addresses: their lengths frame the payload.
            let mut at = 36 + host_len;
            let mut list = [&[][..]; MAX_ADDRESSES];
            for i in 0..count {
                let len = usize::from(*p.get(at).ok_or(MsgError::Length)?);
                let address = p.get(at + 1..at + 1 + len).ok_or(MsgError::Length)?;
                if let Some(slot) = list.get_mut(i) {
                    *slot = address;
                }
                at += 1 + len;
            }
            exact(p, at)?;
            if p[1] != 0 {
                return Err(MsgError::Reserved);
            }
            if host_len == 0 || host_len > MAX_ID || count == 0 || count > MAX_ADDRESSES {
                return Err(MsgError::Range);
            }
            if !list[..count].iter().all(|a| valid_address(a)) {
                return Err(MsgError::Range);
            }
            let host_id = &p[35..35 + host_len];
            utf8(host_id)?;
            Ok(Prov::Provision(Provision {
                flags: p[1],
                host_key: key(&p[2..]),
                host_id,
                addresses: Addresses { list, count },
            }))
        }
        _ => Err(MsgError::UnknownType),
    }
}

/// Writes `msg` as one frame with `seq` to `out`: a leading `0x00`, then
/// the COBS-encoded frame with its delimiter. Nothing is written when a
/// field is out of range or the payload doesn't fit a frame.
pub fn write(msg: &Prov<'_>, seq: u8, mut out: impl FnMut(u8)) -> Result<(), MsgError> {
    let mut buf = [0_u8; MAX_PAYLOAD];
    let n = payload(msg, &mut buf)?;
    let t = match msg {
        Prov::DeviceInfo(_) => ty::DEVICE_INFO,
        Prov::Provisioned { .. } => ty::PROVISIONED,
        Prov::GetDevice => ty::GET_DEVICE,
        Prov::Provision(_) => ty::PROVISION,
    };
    out(0);
    frame::write(t, seq, &[&buf[..n]], out);
    Ok(())
}

/// `msg`'s payload into `buf`: its length, after the checks decoding makes.
pub fn payload(msg: &Prov<'_>, buf: &mut [u8; MAX_PAYLOAD]) -> Result<usize, MsgError> {
    let mut n = 0;
    let mut put = |bytes: &[u8]| -> Result<(), MsgError> {
        let end = n + bytes.len();
        buf.get_mut(n..end)
            .ok_or(MsgError::Length)?
            .copy_from_slice(bytes);
        n = end;
        Ok(())
    };
    match msg {
        Prov::GetDevice => {
            put(&MAGIC)?;
            put(&[VERSION])?;
        }
        Prov::DeviceInfo(d) => {
            if d.flags & !device_flags::ALL != 0 {
                return Err(MsgError::Reserved);
            }
            if d.host_id.len() > MAX_ID || d.name.is_empty() || d.name.len() > MAX_ID {
                return Err(MsgError::Range);
            }
            utf8(d.host_id)?;
            utf8(d.name)?;
            put(&MAGIC)?;
            put(&[VERSION])?;
            put(&d.lib_version.to_le_bytes())?;
            put(&[d.arch, d.flags])?;
            put(&d.device_key)?;
            // At most MAX_ID (32) bytes each, checked above.
            put(&[d.host_id.len() as u8])?;
            put(d.host_id)?;
            put(&[d.name.len() as u8])?;
            put(d.name)?;
        }
        Prov::Provisioned { status, record_crc } => {
            put(&[*status])?;
            put(&record_crc.to_le_bytes())?;
        }
        Prov::Provision(p) => {
            if p.flags != 0 {
                return Err(MsgError::Reserved);
            }
            let list = p.addresses.as_slice();
            if p.host_id.is_empty()
                || p.host_id.len() > MAX_ID
                || list.is_empty()
                || !list.iter().all(|a| valid_address(a))
            {
                return Err(MsgError::Range);
            }
            utf8(p.host_id)?;
            put(&[VERSION, p.flags])?;
            put(&p.host_key)?;
            // At most MAX_ID (32) bytes, MAX_ADDRESSES (4) addresses of at
            // most MAX_ADDRESS (63) bytes: checked above.
            put(&[p.host_id.len() as u8])?;
            put(p.host_id)?;
            put(&[list.len() as u8])?;
            for a in list {
                put(&[a.len() as u8])?;
                put(a)?;
            }
        }
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addresses_are_base_ports() {
        for ok in [
            "192.168.1.20:7447",
            "rlmesh.local:1",
            "[fe80::1]:65532",
            "h:7447",
        ] {
            assert!(valid_address(ok.as_bytes()), "{ok}");
        }
        for bad in [
            "",
            "192.168.1.20",
            ":7447",
            "h:0",
            "h:65533",
            "h:07447",
            "h: 7447",
            "fe80::1:7447",
            "[]:7447",
            "h:+7447",
        ] {
            assert!(!valid_address(bad.as_bytes()), "{bad}");
        }
        assert!(!valid_address(&[b'a'; 64]));
    }

    #[test]
    fn a_frame_starts_with_a_delimiter() {
        let mut out = [0_u8; 16];
        let mut n = 0;
        write(&Prov::GetDevice, 0, |b| {
            out[n] = b;
            n += 1;
        })
        .unwrap();
        assert_eq!(out[0], 0);
        assert_eq!(out[n - 1], 0);
    }
}
