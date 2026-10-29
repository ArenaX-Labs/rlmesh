//! `rlmesh devices provision`: a networked board's USB provisioning
//! (tether spec §14, design §3 item 3). The board answers `GET_DEVICE` with
//! its device key and compiled robot name; it stores the home host's key,
//! id and base addresses from `PROVISION`, and confirms with the record's
//! CRC. Only then is its key enrolled in `devices.toml`, as a board
//! (`kind = "board"`, default cap `model`) whose robot name is pinned to
//! it. The cable is the presence check; nothing on the link is
//! encrypted.
//!
//! It runs on the robot's machine, with rlmeshd running: rlmeshd gives the
//! host's details over its local channel ([`crate::pair::host_details`]).
// TODO: provisioning from a machine paired with the robot (design §3 item
// 3) needs an enroll verb on rlmeshd's pairing or ctl endpoint; add it when
// boards are flashed away from the robot.

use std::io::{self, Write};
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use rlmesh_session::auth::address::base32;
use rlmesh_tether::cobs::Decoder;
use rlmesh_tether::frame::{Frame, MAX_DECODED, MAX_PAYLOAD};
use rlmesh_tether::msg::Receiver;
use rlmesh_tether::provision::{
    self, Addresses, MAX_ADDRESSES, Prov, Provision, record_crc, status, valid_address,
};

use crate::devices::{DEVICES, Entry, Pin, edit};

/// How long one try waits for the board's answer (§14.4).
const TRY: Duration = Duration::from_millis(200);
/// Tries before giving up (§14.4).
const TRIES: u32 = 10;
/// A board's default cap (design §12 decision 6).
const BOARD_CAP: &str = "model";

/// A byte link to a board: its serial port, or a test's fake.
pub trait Port {
    /// Writes `bytes`.
    fn send(&mut self, bytes: &[u8]) -> io::Result<()>;
    /// Reads what arrives within `wait` into `buf`: 0 when nothing came.
    fn recv(&mut self, buf: &mut [u8], wait: Duration) -> io::Result<usize>;
}

/// What the home host gives a board.
pub struct Home {
    /// Its static X25519 key.
    pub host_key: [u8; 32],
    /// Its host id.
    pub host_id: String,
    /// Its base addresses, `host:port`.
    pub addresses: Vec<String>,
}

/// The board, once it stored the record.
#[derive(Debug, PartialEq, Eq)]
struct Board {
    device_key: [u8; 32],
    /// The host id it held before, if any.
    previous: Option<String>,
}

/// Sends `frame` every [`TRY`], up to [`TRIES`] times, until `answer` takes
/// a message the board sent: its result, or `None` when none did.
fn ask<T>(
    port: &mut impl Port,
    frame: &[u8],
    mut answer: impl FnMut(Prov<'_>) -> Option<T>,
) -> Result<Option<T>> {
    let mut decoder = Decoder::<MAX_DECODED>::new();
    let mut buf = [0_u8; 256];
    for _ in 0..TRIES {
        port.send(frame).context("writing to the board")?;
        let until = Instant::now() + TRY;
        loop {
            let wait = until.saturating_duration_since(Instant::now());
            let n = port
                .recv(&mut buf, wait)
                .context("reading from the board")?;
            if n == 0 {
                break;
            }
            for &byte in &buf[..n] {
                // A frame that fails a check (the sketch's text before a
                // delimiter, noise) is dropped, as §3.1 does.
                let Some(Ok(decoded)) = decoder.push(byte) else {
                    continue;
                };
                let Ok(frame) = Frame::parse(decoded) else {
                    continue;
                };
                if let Ok(msg) = provision::decode(&frame, Receiver::Bridge)
                    && let Some(out) = answer(msg)
                {
                    return Ok(Some(out));
                }
            }
        }
    }
    Ok(None)
}

fn frame(msg: &Prov<'_>) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    provision::write(msg, 0, |b| out.push(b)).map_err(|err| anyhow::anyhow!("{err:?}"))?;
    Ok(out)
}

/// The tether §14.4 exchange with the board behind `port`, which must be
/// robot `name`.
fn run(port: &mut impl Port, name: &str, home: &Home) -> Result<Board> {
    let info = ask(port, &frame(&Prov::GetDevice)?, |msg| match msg {
        Prov::DeviceInfo(info) => Some((
            info.device_key,
            String::from_utf8_lossy(info.name).into_owned(),
            String::from_utf8_lossy(info.host_id).into_owned(),
        )),
        _ => None,
    })?;
    let Some((device_key, board_name, previous)) = info else {
        bail!(
            "the board never answered: is it running an RLMesh sketch, at 115200 baud, and is \
             the port right?"
        );
    };
    if board_name != name {
        bail!("the board is robot {board_name:?}, not {name:?}: its name is compiled in");
    }
    // As many addresses as fit the payload.
    let usable: Vec<&[u8]> = home
        .addresses
        .iter()
        .map(String::as_bytes)
        .filter(|a| valid_address(a))
        .take(MAX_ADDRESSES)
        .collect();
    let mut payload = [0_u8; MAX_PAYLOAD];
    let mut fitted = None;
    for count in (1..=usable.len()).rev() {
        let record = Provision {
            flags: 0,
            host_key: home.host_key,
            host_id: home.host_id.as_bytes(),
            addresses: Addresses::new(&usable[..count]).context("at most 4 addresses")?,
        };
        if let Ok(n) = provision::payload(&Prov::Provision(record), &mut payload) {
            fitted = Some((record, n));
            break;
        }
    }
    let Some((record, n)) = fitted else {
        bail!(
            "rlmeshd has no address a board can store (host:port, at most 63 bytes): run it with \
             --listen <LAN IP>:<port>"
        );
    };
    let crc = record_crc(&payload[..n]);
    let mut mismatch = false;
    let stored = ask(port, &frame(&Prov::Provision(record))?, |msg| match msg {
        Prov::Provisioned {
            status: code,
            record_crc,
        } => match (code, record_crc == crc) {
            (status::STORED, true) => Some(Ok(())),
            (status::INVALID, _) => Some(Err("the board refused the record as invalid".into())),
            (_, false) => {
                mismatch = true;
                None
            }
            (code, true) => Some(Err(format!(
                "the board could not store the record (status {code})"
            ))),
        },
        _ => None,
    })?;
    match stored {
        Some(Ok(())) => Ok(Board {
            device_key,
            previous: (!previous.is_empty()).then_some(previous),
        }),
        Some(Err(why)) => bail!("{why}; nothing was enrolled"),
        None if mismatch => bail!(
            "the board confirmed another record than the one sent (CRC mismatch); nothing was \
             enrolled"
        ),
        None => bail!("the board never confirmed the record; nothing was enrolled"),
    }
}

/// Enrolls `board`'s key in the table in `dir` as robot `name`'s board,
/// and pins `name` to it: its device id, and whether it is new. A key
/// already there keeps its device (id, cap, parts and verbs), unless `max`
/// changes the cap.
fn enroll(dir: &Path, key: &[u8; 32], name: &str, max: Option<&str>) -> Result<(String, bool)> {
    let public_key = base32(key);
    edit(dir, |table| {
        let known = table
            .devices
            .iter()
            .find(|(_, entry)| entry.public_key == public_key)
            .map(|(id, _)| id.clone());
        let (id, new) = match known {
            Some(id) => (id, false),
            None => (table.free_id(name, "board"), true),
        };
        let entry = table.devices.entry(id.clone()).or_insert_with(|| Entry {
            public_key,
            kind: None,
            name: None,
            max_priority: BOARD_CAP.into(),
            parts: None,
            verbs: None,
        });
        entry.kind = Some("board".into());
        entry.name = Some(name.into());
        if let Some(max) = max {
            entry.max_priority = max.into();
        }
        // A networked board: no tethered board_id behind it.
        table.robots.insert(
            name.into(),
            Pin {
                device: id.clone(),
                board_id: None,
            },
        );
        Ok((id, new))
    })
}

/// `rlmesh devices provision --port TTY --name ROBOT [--max CLASS]`.
pub fn provision(
    dir: &Path,
    port: &Path,
    name: &str,
    max: Option<&str>,
    stdout: &mut impl Write,
) -> Result<()> {
    #[cfg(unix)]
    {
        let home = crate::pair::host_details(dir)?;
        let mut tty = crate::sys::Tty::open(port).with_context(|| port.display().to_string())?;
        provision_on(&mut tty, dir, name, max, &home, stdout)
    }
    // TODO: no serial port off Unix; add one when boards are provisioned
    // from Windows.
    #[cfg(not(unix))]
    {
        let _ = (dir, port, name, max, stdout);
        bail!("provisioning a board needs a Unix serial port (Linux or macOS)")
    }
}

fn provision_on(
    port: &mut impl Port,
    dir: &Path,
    name: &str,
    max: Option<&str>,
    home: &Home,
    stdout: &mut impl Write,
) -> Result<()> {
    if let Some(max) = max {
        crate::devices::class(max)?;
    }
    let board = run(port, name, home)?;
    if let Some(previous) = &board.previous
        && *previous != home.host_id
    {
        writeln!(
            stdout,
            "the board was provisioned for {previous}; revoke it there (rlmesh devices revoke)"
        )?;
    }
    let (id, new) = enroll(dir, &board.device_key, name, max)?;
    let verb = if new { "enrolled" } else { "kept" };
    writeln!(
        stdout,
        "board {name} provisioned for {}: {verb} as device {id} in {DEVICES}, robot {name} pinned to it",
        home.host_id
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::*;
    use crate::devices::Table;

    /// The tether §14 vectors (`rlmesh-tether/tests/vectors.json`).
    fn vector(name: &str) -> Vec<u8> {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../rlmesh-tether/tests/vectors.json"
        );
        let vectors: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).expect("vectors")).expect("json");
        let wire = vectors["provision"]
            .as_array()
            .expect("provision")
            .iter()
            .find(|v| v["name"] == name)
            .expect(name)["wire"]
            .as_str()
            .expect("wire");
        (0..wire.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&wire[i..i + 2], 16).expect("hex"))
            .collect()
    }

    /// A board on a fake port: `DEVICE_INFO` from the vectors (robot
    /// `eye-3`), after the sketch's text, and `PROVISIONED` with the CRC
    /// `crc` makes of the payload it got.
    struct FakeBoard {
        decoder: Decoder<MAX_DECODED>,
        out: VecDeque<u8>,
        crc: fn(u32) -> u32,
        stored: Vec<Vec<u8>>,
    }

    impl FakeBoard {
        fn new(crc: fn(u32) -> u32) -> Self {
            Self {
                decoder: Decoder::new(),
                out: VecDeque::new(),
                crc,
                stored: Vec::new(),
            }
        }
    }

    impl Port for FakeBoard {
        fn send(&mut self, bytes: &[u8]) -> io::Result<()> {
            for &byte in bytes {
                let Some(Ok(decoded)) = self.decoder.push(byte) else {
                    continue;
                };
                let frame = Frame::parse(decoded).expect("a valid frame");
                match provision::decode(&frame, Receiver::Board).expect("a message") {
                    Prov::GetDevice => {
                        self.out.extend(b"booting eye-3\r\n");
                        self.out.extend(vector("DEVICE_INFO, not provisioned"));
                    }
                    Prov::Provision(_) => {
                        self.stored.push(frame.payload.to_vec());
                        let crc = (self.crc)(record_crc(frame.payload));
                        let reply = Prov::Provisioned {
                            status: status::STORED,
                            record_crc: crc,
                        };
                        provision::write(&reply, 0, |b| self.out.push_back(b)).expect("write");
                    }
                    other => panic!("{other:?}"),
                }
            }
            Ok(())
        }

        fn recv(&mut self, buf: &mut [u8], _: Duration) -> io::Result<usize> {
            let n = buf.len().min(self.out.len());
            for (slot, byte) in buf.iter_mut().zip(self.out.drain(..n)) {
                *slot = byte;
            }
            Ok(n)
        }
    }

    fn home() -> Home {
        Home {
            host_key: [0x50; 32],
            host_id: "rlm-3f9a2c".into(),
            addresses: vec!["192.168.1.20:7447".into(), "not an address".into()],
        }
    }

    #[test]
    fn a_board_is_enrolled_only_on_its_confirmation_and_keeps_its_device() {
        let dir = tempfile::tempdir().expect("tempdir");
        // The vector's device key.
        let board_key: [u8; 32] = std::array::from_fn(|i| 0xa0 + i as u8);
        let mut out = Vec::new();
        let enrolled = |dir: &Path| Table::read(dir).expect("table").devices.len();

        // Another robot's board, and a board confirming another record:
        // nothing is enrolled.
        let err = provision_on(
            &mut FakeBoard::new(|c| c),
            dir.path(),
            "eye-4",
            None,
            &home(),
            &mut out,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("\"eye-3\", not \"eye-4\""),
            "{err}"
        );
        let err = provision_on(
            &mut FakeBoard::new(|c| c ^ 1),
            dir.path(),
            "eye-3",
            None,
            &home(),
            &mut out,
        )
        .unwrap_err();
        assert!(err.to_string().contains("CRC mismatch"), "{err}");
        assert_eq!(enrolled(dir.path()), 0);

        // A matching confirmation enrolls a board at model, pinned.
        let mut board = FakeBoard::new(|c| c);
        provision_on(&mut board, dir.path(), "eye-3", None, &home(), &mut out)
            .expect("provisioned");
        let record = &board.stored[0];
        assert!(record.windows(17).any(|w| w == b"192.168.1.20:7447"));
        let table = Table::read(dir.path()).expect("table");
        let entry = &table.devices["eye-3"];
        assert_eq!(entry.public_key, base32(&board_key));
        assert_eq!(
            (
                entry.kind.as_deref(),
                entry.name.as_deref(),
                entry.max_priority.as_str()
            ),
            (Some("board"), Some("eye-3"), "model")
        );
        assert_eq!(
            table.robots["eye-3"],
            Pin {
                device: "eye-3".into(),
                board_id: None
            }
        );

        // Provisioned again, with a cap: the same device.
        provision_on(
            &mut FakeBoard::new(|c| c),
            dir.path(),
            "eye-3",
            Some("operator"),
            &home(),
            &mut out,
        )
        .expect("again");
        let table = Table::read(dir.path()).expect("table");
        assert_eq!(table.devices.len(), 1);
        assert_eq!(table.devices["eye-3"].max_priority, "operator");
    }
}
