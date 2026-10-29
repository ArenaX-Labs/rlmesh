//! A board's provisioning responder (tether spec §14) on a serial device,
//! with a file for its NVS: rehearses `rlmesh devices provision` without a
//! board, over a pseudo-terminal pair (`socat`).
//!
//! ```sh
//! cargo run -p rlmesh-device --example fake_board -- <tty> <nvs-file> [name]
//! ```
//!
//! The file holds the device key (32 bytes), then the record. Linux and
//! macOS: the port is set up with `stty`.

#![allow(clippy::print_stdout, clippy::print_stderr, clippy::disallowed_methods)]

use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::process::Command;

use rlmesh_device::provision::{Record, Responder, Step, public_key};

fn main() {
    let mut args = std::env::args().skip(1);
    let (Some(tty), Some(nvs)) = (args.next(), args.next()) else {
        eprintln!("usage: fake_board <tty> <nvs-file> [name]");
        std::process::exit(2);
    };
    let name = args.next().unwrap_or_else(|| "eyes".into());
    let flag = if cfg!(target_os = "macos") {
        "-f"
    } else {
        "-F"
    };
    let ok = Command::new("stty")
        .args([flag, &tty, "115200", "raw", "-echo"])
        .status()
        .is_ok_and(|s| s.success());
    assert!(ok, "stty couldn't set up {tty}");

    // The device key: made on first run, as a board's first boot does.
    let stored = std::fs::read(&nvs).unwrap_or_default();
    let mut key = [0_u8; 32];
    if stored.len() >= 32 {
        key.copy_from_slice(&stored[..32]);
    } else {
        let mut urandom = std::fs::File::open("/dev/urandom").expect("/dev/urandom");
        urandom.read_exact(&mut key).expect("random");
        std::fs::write(&nvs, key).expect("write the key");
    }
    let record = stored.get(32..).and_then(Record::parse);
    let hex: String = public_key(&key)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    println!(
        "board {name}, device key {hex}, provisioned: {}",
        record.is_some()
    );

    let mut port = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&tty)
        .expect("open the port");
    let mut reader = port.try_clone().expect("clone");
    let mut responder = Responder::new(&name, &key, 4, record.as_ref());
    // The sketch's text, as a board prints at boot.
    port.write_all(b"booted\r\n").expect("write");
    let mut buf = [0_u8; 256];
    loop {
        let n = reader.read(&mut buf).expect("read");
        for &byte in &buf[..n] {
            match responder.push(byte) {
                None => {}
                Some(Step::Reply(bytes)) => port.write_all(&bytes).expect("write"),
                Some(Step::Store(record)) => {
                    let mut file = key.to_vec();
                    file.extend_from_slice(&record.payload);
                    let ok = std::fs::write(&nvs, file).is_ok();
                    println!(
                        "stored: host {} at {:?} ({ok})",
                        record.host_id, record.addresses
                    );
                    port.write_all(&responder.stored(&record, ok))
                        .expect("write");
                }
            }
        }
    }
}
