//! Checks a board running `integrations/arduino/examples/Eyes` over its
//! serial port, for the board's half of the spec (§5–6), until the bridge
//! lands: `mise run test:tether:board [port] [--clear-estop]`.
//!
//! It reads the `HELLO`s and the map, attaches, drives the eyes, and checks
//! that the board trips back to `HELLO` when the link goes quiet, refuses a
//! stale `gen`, and detaches. Press the board's reset button between runs
//! to see `boot` change and the reset cause.
//!
//! Linux and macOS: the port is set up with `stty`.

// A host tool, not the sans-IO crate: it reads the clock, spawns a reader
// and prints its report.
#![allow(clippy::disallowed_methods, clippy::print_stdout, clippy::print_stderr)]

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::process::{Command, ExitCode};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

use rlmesh_tether::cobs::Decoder;
use rlmesh_tether::crc::crc32;
use rlmesh_tether::frame::{self, Frame, MAX_ENCODED};
use rlmesh_tether::msg::{
    self, Hello, MAX_MAP_SLICE, Msg, Receiver as End, Status, attach_flags, hello_flags,
    status_flags,
};

/// The Eyes example's one part: `pos` for pan and tilt, two `f32`s, as both
/// command and sample.
const EYES: u8 = 0;
const EYES_LEN: usize = 8;
const BAUD: &str = "115200";

/// One frame from the board, as received.
struct Rx {
    at: Instant,
    ty: u8,
    seq: u8,
    payload: Vec<u8>,
}

impl Rx {
    fn msg(&self) -> Result<Msg<'_>, msg::MsgError> {
        let frame = Frame {
            ty: self.ty,
            seq: self.seq,
            payload: &self.payload,
        };
        msg::decode(&frame, End::Bridge, |part| {
            (part == EYES).then_some(EYES_LEN)
        })
    }
}

/// A `HELLO`'s fields, owned.
#[derive(Clone, Debug)]
struct Board {
    arch: u8,
    board_id: [u8; 8],
    boot: u16,
    gen_: u8,
    flags: u8,
    map_len: u16,
    map_crc: u32,
    watchdog_ms: u16,
    rx_max: u16,
    name: String,
    lib_version: u16,
}

impl From<&Hello<'_>> for Board {
    fn from(h: &Hello<'_>) -> Self {
        Self {
            arch: h.arch,
            board_id: h.board_id,
            boot: h.boot,
            gen_: h.gen_,
            flags: h.flags,
            map_len: h.map_len,
            map_crc: h.map_crc,
            watchdog_ms: h.watchdog_ms,
            rx_max: h.rx_max,
            name: String::from_utf8_lossy(h.name).into_owned(),
            lib_version: h.lib_version,
        }
    }
}

struct Link {
    port: File,
    rx: Receiver<Rx>,
    seq: u8,
    /// When the last frame went to the board.
    sent: Instant,
    /// The board's frames that failed to decode, and its `NOTE`s, so far.
    bad: u32,
}

impl Link {
    fn open(path: &str) -> Result<Self, String> {
        let flag = if cfg!(target_os = "macos") {
            "-f"
        } else {
            "-F"
        };
        // Raw bytes; -hupcl keeps DTR up on close, so an Uno isn't reset by
        // the next run's open.
        let ok = Command::new("stty")
            .args([flag, path, BAUD, "raw", "-echo", "-hupcl", "clocal"])
            .status()
            .map_err(|e| format!("stty: {e}"))?
            .success();
        if !ok {
            return Err(format!("stty couldn't set up {path}"));
        }
        let port = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|e| format!("{path}: {e}"))?;
        let mut reader = port.try_clone().map_err(|e| e.to_string())?;
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let mut cobs = Decoder::<{ frame::MAX_DECODED }>::new();
            let mut buf = [0_u8; 256];
            while let Ok(n) = reader.read(&mut buf) {
                if n == 0 {
                    thread::sleep(Duration::from_millis(1));
                    continue;
                }
                let at = Instant::now();
                for &byte in &buf[..n] {
                    let Some(Ok(decoded)) = cobs.push(byte) else {
                        continue;
                    };
                    let Ok(f) = Frame::parse(decoded) else {
                        continue;
                    };
                    let rx = Rx {
                        at,
                        ty: f.ty,
                        seq: f.seq,
                        payload: f.payload.to_vec(),
                    };
                    if tx.send(rx).is_err() {
                        return;
                    }
                }
            }
        });
        Ok(Self {
            port,
            rx,
            seq: 0,
            sent: Instant::now(),
            bad: 0,
        })
    }

    fn send(&mut self, m: &Msg<'_>) {
        let mut out = Vec::with_capacity(MAX_ENCODED + 1);
        msg::write(m, self.seq, |b| out.push(b)).expect("a valid message");
        self.seq = self.seq.wrapping_add(1);
        self.port.write_all(&out).expect("write to the board");
        self.port.flush().expect("flush to the board");
        self.sent = Instant::now();
    }

    /// The next frame, by `deadline`.
    fn next(&mut self, deadline: Instant) -> Option<Rx> {
        let left = deadline.saturating_duration_since(Instant::now());
        match self.rx.recv_timeout(left) {
            Ok(rx) => Some(rx),
            Err(RecvTimeoutError::Timeout) => None,
            Err(RecvTimeoutError::Disconnected) => panic!("the port closed"),
        }
    }

    /// Frames until one decodes to what `want` picks, by `deadline`.
    fn wait<T>(
        &mut self,
        deadline: Instant,
        mut want: impl FnMut(&Msg<'_>, &Rx) -> Option<T>,
    ) -> Option<T> {
        while let Some(rx) = self.next(deadline) {
            match rx.msg() {
                Ok(Msg::Note { level, code, text }) => {
                    self.bad += 1;
                    println!(
                        "    note (level {level}, code {code}): {}",
                        String::from_utf8_lossy(text)
                    );
                }
                Ok(m) => {
                    if let Some(t) = want(&m, &rx) {
                        return Some(t);
                    }
                }
                Err(e) => {
                    self.bad += 1;
                    println!("    undecodable frame 0x{:02x}: {e:?}", rx.ty);
                }
            }
        }
        None
    }

    fn hello(&mut self, within: Duration) -> Option<(Board, Instant)> {
        self.wait(Instant::now() + within, |m, rx| match m {
            Msg::Hello(h) => Some((Board::from(h), rx.at)),
            _ => None,
        })
    }

    /// `ATTACH`, retried as the bridge does, until a `STATUS` confirms it.
    fn attach(&mut self, board: &Board, gen_: u8, flags: u8, tries: u32) -> Option<Status> {
        let every = Duration::from_millis(u64::from((board.watchdog_ms / 3).min(100)));
        for _ in 0..tries {
            self.send(&Msg::Attach {
                board_id: board.board_id,
                boot: board.boot,
                gen_,
                flags,
            });
            let got = self.wait(Instant::now() + every, |m, _| match m {
                Msg::Status(s)
                    if s.flags & status_flags::ATTACHED != 0
                        && s.gen_ == gen_
                        && s.boot == board.boot =>
                {
                    Some(*s)
                }
                _ => None,
            });
            if got.is_some() {
                return got;
            }
        }
        None
    }
}

#[derive(Default)]
struct Report {
    failed: u32,
}

impl Report {
    fn check(&mut self, ok: bool, what: &str) -> bool {
        println!("  {} {what}", if ok { "ok  " } else { "FAIL" });
        if !ok {
            self.failed += 1;
        }
        ok
    }
}

fn arch(a: u8) -> &'static str {
    match a {
        1 => "AVR",
        2 => "ARM Cortex-M",
        3 => "RISC-V",
        4 => "Xtensa",
        _ => "unknown",
    }
}

fn reset(r: u8) -> &'static str {
    match r {
        0 => "power-on",
        1 => "external (reset button or upload)",
        2 => "brown-out: power servos separately",
        3 => "watchdog: the sketch blocked",
        _ => "other",
    }
}

/// The next `gen` after `last`, skipping 0 and `avoid`.
fn next_gen(last: u8, avoid: u8) -> u8 {
    let mut g = last;
    loop {
        g = g.wrapping_add(1);
        if g != 0 && g != avoid {
            return g;
        }
    }
}

fn read_map(link: &mut Link, board: &Board) -> Option<Vec<u8>> {
    let mut map = Vec::new();
    while map.len() < usize::from(board.map_len) {
        let offset = u16::try_from(map.len()).ok()?;
        let mut slice = None;
        // Retried every 200 ms, up to 5 times (§5.1 step 3).
        for _ in 0..6 {
            link.send(&Msg::GetMap {
                boot: board.boot,
                offset,
                max: u8::try_from(MAX_MAP_SLICE).ok()?,
            });
            slice = link.wait(
                Instant::now() + Duration::from_millis(200),
                |m, _| match m {
                    Msg::Map {
                        boot,
                        offset: o,
                        data,
                    } if *boot == board.boot && *o == offset => Some(data.to_vec()),
                    _ => None,
                },
            );
            if slice.is_some() {
                break;
            }
        }
        let slice = slice?;
        if slice.is_empty() {
            break;
        }
        map.extend_from_slice(&slice);
    }
    Some(map)
}

fn eyes(pan: f32, tilt: f32) -> [u8; EYES_LEN] {
    let mut v = [0; EYES_LEN];
    v[..4].copy_from_slice(&pan.to_le_bytes());
    v[4..].copy_from_slice(&tilt.to_le_bytes());
    v
}

fn run(path: &str, clear_estop: bool) -> Result<Report, String> {
    let mut r = Report::default();
    let mut link = Link::open(path)?;
    println!("{path} at {BAUD} baud");

    println!("discovery");
    // An Uno resets when the port opens: its bootloader takes a second.
    let Some((board, first)) = link.hello(Duration::from_secs(3)) else {
        r.check(
            false,
            "a HELLO within 3 s (is Eyes flashed, and the port right?)",
        );
        return Ok(r);
    };
    println!(
        "    \"{}\", {} (lib 0x{:04x}), id {}, boot {}, last gen {}",
        board.name,
        arch(board.arch),
        board.lib_version,
        board
            .board_id
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>(),
        board.boot,
        board.gen_,
    );
    println!(
        "    watchdog {} ms, rx_max {}, map {} bytes",
        board.watchdog_ms, board.rx_max, board.map_len
    );
    let estop = board.flags & hello_flags::ESTOP != 0;
    if estop {
        println!("    the stored e-stop latch is set");
    }
    if let Some((_, second)) = link.hello(Duration::from_secs(1)) {
        let gap = second.duration_since(first).as_millis();
        r.check(
            (400..=650).contains(&gap),
            &format!("HELLO every 500 ms while unattached ({gap} ms)"),
        );
    } else {
        r.check(false, "a second HELLO within 1 s");
    }

    println!("map");
    match read_map(&mut link, &board) {
        Some(map) => {
            r.check(
                map.len() == usize::from(board.map_len),
                &format!("{} of {} bytes", map.len(), board.map_len),
            );
            r.check(crc32(&map) == board.map_crc, "CRC-32 matches the HELLO");
            println!("    {}", String::from_utf8_lossy(&map));
        }
        None => {
            r.check(false, "every GET_MAP answered within 5 retries");
        }
    }

    println!("attach");
    let mut gen_ = next_gen(board.gen_, board.gen_);
    let flags = if clear_estop {
        attach_flags::CLEAR_ESTOP
    } else {
        0
    };
    let Some(status) = link.attach(&board, gen_, flags, 10) else {
        r.check(
            false,
            &format!("STATUS confirms gen {gen_} within 10 tries"),
        );
        return Ok(r);
    };
    r.check(true, &format!("attached as gen {gen_}"));
    println!("    last reset: {}", reset(status.reset));
    let latched = status.flags & status_flags::ESTOP_LATCHED != 0;
    if latched {
        let input = status.flags & status_flags::ESTOP_INPUT != 0;
        println!(
            "    e-stop latched{}: commands are dropped; rerun with --clear-estop",
            if input { ", button pressed" } else { "" }
        );
    }
    if clear_estop {
        r.check(!latched, "the e-stop latch cleared");
    }

    println!("drive (2 s at 50 Hz)");
    // ttl_ms per §6.1 for a 20 ms control period.
    let ttl_ms = 60.min(board.watchdog_ms);
    let start = Instant::now();
    let mut tick = start;
    let mut last_status = Instant::now();
    let mut status_gap = Duration::ZERO;
    let (mut states, mut matched, mut lost, mut bad) = (0_u32, 0_u32, 0_u16, 0_u16);
    let mut sent: Vec<[u8; EYES_LEN]> = Vec::new();
    while start.elapsed() < Duration::from_secs(2) {
        let t = start.elapsed().as_secs_f32();
        let value = eyes(0.5 * (t * 3.0).sin(), 0.3 * (t * 2.0).cos());
        link.send(&Msg::Command {
            gen_,
            part: EYES,
            flags: 0,
            ttl_ms,
            value: &value,
        });
        sent.push(value);
        tick += Duration::from_millis(20);
        while let Some(rx) = link.next(tick) {
            match rx.msg() {
                Ok(Msg::State { part, value, .. }) if part == EYES => {
                    states += 1;
                    // Eyes reports its last target as its state.
                    if sent.iter().rev().take(5).any(|s| s[..] == *value) {
                        matched += 1;
                    }
                }
                Ok(Msg::Status(s)) => {
                    status_gap = status_gap.max(rx.at.duration_since(last_status));
                    last_status = rx.at;
                    (lost, bad) = (s.lost, s.bad);
                }
                Ok(Msg::Hello(_)) => {
                    r.check(false, "no HELLO while attached (the board tripped)");
                }
                _ => {}
            }
        }
    }
    r.check(
        (80..=120).contains(&states),
        &format!("STATE at the sketch's 50 Hz ({states} in 2 s)"),
    );
    if !latched {
        r.check(
            matched + 5 >= states,
            &format!("STATE echoes the commands sent ({matched} of {states})"),
        );
    }
    r.check(
        status_gap <= Duration::from_millis(120),
        &format!(
            "STATUS at least every 100 ms (longest gap {} ms)",
            status_gap.as_millis()
        ),
    );
    r.check(
        bad == 0,
        &format!("the board dropped no frame as invalid ({bad})"),
    );
    println!("    the board counts {lost} frames lost from seq gaps");

    println!("link watchdog");
    // PINGs alone keep the link (not the permit) alive.
    let until = Instant::now() + Duration::from_millis(u64::from(board.watchdog_ms) * 2);
    let mut tripped = false;
    while Instant::now() < until {
        link.send(&Msg::Ping { gen_ });
        let next = Instant::now() + Duration::from_millis(u64::from(board.watchdog_ms / 3));
        tripped |= link
            .wait(next, |m, _| matches!(m, Msg::Hello(_)).then_some(()))
            .is_some();
    }
    r.check(!tripped, "PINGs keep the board attached");
    let quiet = link.sent;
    match link.hello(Duration::from_millis(u64::from(board.watchdog_ms) + 500)) {
        Some((again, at)) => {
            let after = at.duration_since(quiet).as_millis();
            let wd = u128::from(board.watchdog_ms);
            r.check(
                after >= wd.saturating_sub(10) && after <= wd + 60,
                &format!("a quiet link trips to HELLO after {after} ms (watchdog {wd} ms)"),
            );
            r.check(
                again.boot == board.boot && again.gen_ == gen_,
                "the HELLO keeps the boot and reports the gen it had",
            );
        }
        None => {
            r.check(false, "a quiet link trips to HELLO");
        }
    }

    println!("re-attach");
    let stale = link.attach(&board, gen_, 0, 3);
    r.check(
        stale.is_none(),
        &format!("the board refuses its last gen ({gen_})"),
    );
    gen_ = next_gen(gen_, gen_);
    r.check(
        link.attach(&board, gen_, 0, 10).is_some(),
        &format!("a new gen ({gen_}) attaches"),
    );
    link.send(&Msg::Detach { gen_ });
    let sent_at = link.sent;
    match link.hello(Duration::from_millis(500)) {
        Some((_, at)) => {
            let after = at.duration_since(sent_at).as_millis();
            r.check(
                after <= 100,
                &format!("DETACH goes back to HELLO at once ({after} ms)"),
            );
        }
        None => {
            r.check(false, "DETACH goes back to HELLO");
        }
    }
    if link.bad > 0 {
        println!(
            "    {} notes or undecodable frames from the board",
            link.bad
        );
    }
    Ok(r)
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let mut path = None;
    let mut clear_estop = false;
    for arg in args.by_ref() {
        match arg.as_str() {
            "--clear-estop" => clear_estop = true,
            p if !p.starts_with('-') && path.is_none() => path = Some(p.to_owned()),
            other => {
                eprintln!("usage: board_check [port] [--clear-estop] (unknown: {other})");
                return ExitCode::from(2);
            }
        }
    }
    let path = path.unwrap_or_else(|| "/dev/ttyACM0".to_owned());
    match run(&path, clear_estop) {
        Ok(r) if r.failed == 0 => {
            println!("every check passed");
            ExitCode::SUCCESS
        }
        Ok(r) => {
            println!("{} checks failed", r.failed);
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(2)
        }
    }
}
