//! Control timing under camera fan-out, through a session host, for each
//! way the host can listen.
//!
//! ```text
//! fanout_bench run --listeners one|four [--readers 2] [--camera-hz 30]
//!     [--width 640] [--height 480] [--ft-hz 1000] [--warmup-s 3] [--duration-s 20]
//! ```
//!
//! The host runs here; a robot, the readers and a prober are OS processes
//! of their own, each joining with one client session per traffic class.
//! The robot publishes a raw camera (Bulk) and a 72-byte force/torque
//! stream (Realtime); every reader subscribes to both, so the host relays
//! each frame once per reader. The prober holds a lease on the robot's
//! command part and queries the host's state (a Control round trip) every
//! 10 ms.
//!
//! `--listeners four` is a host listening on four consecutive ports, one
//! Zenoh session each, every class of a joiner dialing its own; `one` is a
//! host with one session listening on the base port, which every class
//! dials. `--camera-hz 0` runs without the camera.
//!
//! Prints one JSON object: the prober's Control round trips, heartbeat
//! inter-arrival gaps at the host (every remote participant; the period is
//! 50 ms), the host's heartbeat round trips, lease lapses and departures,
//! and each reader's frame rate and latencies (ms: p50/p99/max).

#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::collections::BTreeMap;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use rlmesh_session::address::AddressKey;
use rlmesh_session::media::synthetic_frame;
use rlmesh_session::transport::{Endpoints, Layout};
use rlmesh_session::wire::{
    Fingerprint, ParticipantId, PriorityClass, ReasonCode, Role, SessionId, SessionKeys, TenantId,
};
use rlmesh_session::{
    AuthorityEvent, Clock, HostConfig, Participant, ParticipantConfig, SessionHost,
    SubscribeOptions, SystemMonotonicClock, Transport, TransportConfig, TransportTuning,
};
use serde_json::{Value, json};

type BoxError = Box<dyn std::error::Error + Send + Sync>;

const ROBOT: &str = "robot";
const WAIT: Duration = Duration::from_secs(30);

#[derive(Clone, Debug)]
struct Args {
    role: String,
    listeners: String,
    readers: usize,
    camera_hz: u32,
    width: u32,
    height: u32,
    ft_hz: u32,
    warmup: Duration,
    duration: Duration,
    base: Option<SocketAddr>,
    dir: Option<PathBuf>,
    id: String,
}

fn parse_args() -> Result<Args, BoxError> {
    let mut raw = std::env::args().skip(1);
    let role = raw
        .next()
        .ok_or("usage: fanout_bench run|robot|reader|prober [options]")?;
    let mut opts = BTreeMap::new();
    while let Some(key) = raw.next() {
        let key = key
            .strip_prefix("--")
            .ok_or_else(|| format!("unexpected {key}"))?
            .to_string();
        opts.insert(
            key.clone(),
            raw.next().ok_or_else(|| format!("--{key} needs a value"))?,
        );
    }
    let num = |k: &str, d: f64| -> Result<f64, BoxError> {
        Ok(opts.get(k).map_or(Ok(d), |v| v.parse::<f64>())?)
    };
    let listeners = opts
        .get("listeners")
        .cloned()
        .unwrap_or_else(|| "one".into());
    if !matches!(listeners.as_str(), "one" | "four") {
        return Err("--listeners one|four".into());
    }
    Ok(Args {
        role,
        listeners,
        readers: num("readers", 2.0)? as usize,
        camera_hz: num("camera-hz", 30.0)? as u32,
        width: num("width", 640.0)? as u32,
        height: num("height", 480.0)? as u32,
        ft_hz: num("ft-hz", 1000.0)? as u32,
        warmup: Duration::from_secs_f64(num("warmup-s", 3.0)?),
        duration: Duration::from_secs_f64(num("duration-s", 20.0)?),
        base: opts.get("base").map(|b| b.parse()).transpose()?,
        dir: opts.get("dir").map(PathBuf::from),
        id: opts.get("id").cloned().unwrap_or_default(),
    })
}

fn tcp(ip: Ipv4Addr, port: u16) -> String {
    format!("tcp/{ip}:{port}")
}

/// A base port whose next `n - 1` ports were free a moment ago.
fn free_base(n: u16) -> Result<SocketAddr, BoxError> {
    let ip = Ipv4Addr::LOCALHOST;
    for _ in 0..64 {
        let first = std::net::TcpListener::bind((ip, 0))?;
        let base = first.local_addr()?.port();
        if base > u16::MAX - n {
            continue;
        }
        let rest: Result<Vec<_>, _> = (1..n)
            .map(|i| std::net::TcpListener::bind((ip, base + i)))
            .collect();
        if rest.is_ok() {
            return Ok(SocketAddr::from((ip, base)));
        }
    }
    Err("no free ports".into())
}

/// The host's listeners: one session on `base`, or one per class on
/// `base..base + 3`.
fn host_transport(listeners: &str, base: SocketAddr) -> TransportConfig {
    let ip = Ipv4Addr::LOCALHOST;
    match listeners {
        "four" => TransportConfig {
            layout: Layout::PerClass(std::array::from_fn(|i| Endpoints {
                listen: vec![tcp(ip, base.port() + i as u16)],
                connect: Vec::new(),
            })),
            tuning: TransportTuning::default(),
        },
        _ => TransportConfig::shared(&[&tcp(ip, base.port())], &[]),
    }
}

/// A joiner's client sessions, one per class, dialing the class's port or
/// the base.
fn joiner_transport(listeners: &str, base: SocketAddr) -> TransportConfig {
    let ip = Ipv4Addr::LOCALHOST;
    let offset = |i: usize| if listeners == "four" { i as u16 } else { 0 };
    TransportConfig {
        layout: Layout::PerClass(std::array::from_fn(|i| Endpoints {
            listen: Vec::new(),
            connect: vec![tcp(ip, base.port() + offset(i))],
        })),
        tuning: TransportTuning {
            client: true,
            connect_timeout: Some(WAIT),
            ..TransportTuning::default()
        },
    }
}

fn robot_map(args: &Args) -> String {
    let camera = if args.camera_hz == 0 {
        String::new()
    } else {
        format!(
            r#", "camera": {{"kind": "camera", "hz": {hz}, "width": {w}, "height": {h},
                 "max_age_ms": 100, "send": "latest(hz={hz})"}}"#,
            hz = args.camera_hz,
            w = args.width,
            h = args.height,
        )
    };
    format!(
        r#"{{
        "cmd": {{"kind": "x/cmd", "hz": 100, "fields": {{"v": ["float32", [6]]}},
                "command": "v", "priority": "control", "hold_max_ms": 100, "safe": "stop"}},
        "ft": {{"kind": "x/ft", "hz": {ft}, "fields": {{"wrench": ["float32", [18]]}},
               "priority": "realtime"}}{camera}
    }}"#,
        ft = args.ft_hz.max(1),
    )
}

/// p50/p99/max in ms (nearest rank), and the count.
fn percentiles(mut ns: Vec<i64>) -> Value {
    if ns.is_empty() {
        return Value::Null;
    }
    ns.sort_unstable();
    let at = |q: f64| ns[((ns.len() - 1) as f64 * q).round() as usize] as f64 / 1e6;
    json!({"n": ns.len(), "p50": at(0.5), "p99": at(0.99), "max": at(1.0)})
}

fn sleep_until(clock: &SystemMonotonicClock, deadline_ns: i64) {
    let now = clock.now_ns();
    if deadline_ns > now {
        std::thread::sleep(Duration::from_nanos((deadline_ns - now) as u64));
    }
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<(), BoxError> {
    let args = parse_args()?;
    let out = match args.role.as_str() {
        "run" => run(&args).await?,
        "robot" => robot(&args).await?,
        "reader" => reader(&args).await?,
        "prober" => prober(&args).await?,
        other => return Err(format!("unknown role {other}").into()),
    };
    println!("{out}");
    Ok(())
}

/// The session id and key the host leaves in `dir`.
async fn session_of(dir: &Path) -> Result<(SessionId, AddressKey), BoxError> {
    loop {
        if let Ok(text) = std::fs::read_to_string(dir.join("session"))
            && let Some((id, key)) = text.trim().split_once(' ')
            && let (Ok(id), Ok(key)) = (id.parse(), key.parse())
        {
            return Ok((id, key));
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// The measurement window the host leaves in `dir` once it is over.
async fn window_of(dir: &Path) -> (i64, i64) {
    loop {
        if let Ok(text) = std::fs::read_to_string(dir.join("window"))
            && let Some((a, b)) = text.trim().split_once(' ')
            && let (Ok(a), Ok(b)) = (a.parse(), b.parse())
        {
            return (a, b);
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

async fn join(args: &Args, id: &str, roles: Vec<Role>) -> Result<Participant, BoxError> {
    let dir = args.dir.as_ref().ok_or("--dir")?;
    let base = args.base.ok_or("--base")?;
    let (session, key) = session_of(dir).await?;
    let transport = Transport::open(&joiner_transport(&args.listeners, base)).await?;
    let mut config = ParticipantConfig::new(
        TenantId::new("bench")?,
        session,
        ParticipantId::new(id)?,
        roles,
        PriorityClass::Model,
    );
    config.clock = SystemMonotonicClock.shared();
    config.ctl_timeout = Duration::from_secs(5);
    config.credential = Some(key.credential());
    Ok(Participant::attach(transport, config).await?)
}

fn spawn(
    args: &Args,
    role: &str,
    id: &str,
    base: SocketAddr,
    dir: &Path,
) -> std::io::Result<std::process::Child> {
    Command::new(std::env::current_exe()?)
        .args([
            role,
            "--listeners",
            &args.listeners,
            "--camera-hz",
            &args.camera_hz.to_string(),
            "--width",
            &args.width.to_string(),
            "--height",
            &args.height.to_string(),
            "--ft-hz",
            &args.ft_hz.to_string(),
            "--base",
            &base.to_string(),
            "--dir",
            &dir.display().to_string(),
            "--id",
            id,
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
}

async fn run(args: &Args) -> Result<Value, BoxError> {
    let clock = SystemMonotonicClock;
    let base = free_base(if args.listeners == "four" { 4 } else { 1 })?;
    let transport = Transport::open(&host_transport(&args.listeners, base)).await?;
    let tenant = TenantId::new("bench")?;
    let session = SessionId::new();
    let mut config = HostConfig::new(tenant.clone(), session);
    config.clock = clock.shared();
    config.health_log = None;
    let key = AddressKey::of(&config.identity);
    let host = SessionHost::start(transport.clone(), config).await?;
    let mut events = host.events();
    let counts = Arc::new(Mutex::new(BTreeMap::<String, u64>::new()));
    let window = Arc::new(Mutex::new((i64::MAX, i64::MAX)));
    let in_window = {
        let window = window.clone();
        move |t: i64| {
            let (a, b) = *window.lock().expect("window");
            (a..b).contains(&t)
        }
    };
    let events_task = tokio::spawn({
        let (counts, in_window) = (counts.clone(), in_window.clone());
        async move {
            while let Ok(event) = events.recv().await {
                let name = match &event {
                    AuthorityEvent::LeaseEnded { reason, .. }
                        if reason.code == ReasonCode::HeartbeatLapse =>
                    {
                        "lease_lapsed".to_string()
                    }
                    AuthorityEvent::LeaseEnded { reason, .. } => {
                        format!("lease_ended_{:?}", reason.code)
                    }
                    AuthorityEvent::Detached { participant, .. } => {
                        format!("detached_{participant}")
                    }
                    AuthorityEvent::HolderLost { code, .. } => format!("holder_lost_{code:?}"),
                    _ => continue,
                };
                let late = !in_window(SystemMonotonicClock.now_ns());
                let name = if late {
                    format!("outside_window_{name}")
                } else {
                    name
                };
                *counts.lock().expect("counts").entry(name).or_default() += 1;
            }
        }
    });
    // Heartbeat arrivals at the host, by participant.
    let keys = SessionKeys::new(&tenant, session);
    let arrivals = Arc::new(Mutex::new(BTreeMap::<String, Vec<i64>>::new()));
    let _heartbeats = transport
        .control()
        .declare_subscriber(keys.heartbeats())
        .callback({
            let (arrivals, keys) = (arrivals.clone(), keys.clone());
            move |sample| {
                let now = SystemMonotonicClock.now_ns();
                if let Some(id) = keys.participant_of(sample.key_expr().as_str()) {
                    arrivals
                        .lock()
                        .expect("arrivals")
                        .entry(id.to_string())
                        .or_default()
                        .push(now);
                }
            }
        })
        .await?;
    let rtts = Arc::new(Mutex::new(Vec::<i64>::new()));
    let health_task = tokio::spawn({
        let (rtts, in_window) = (rtts.clone(), in_window.clone());
        let mut health = host.watch_health();
        async move {
            while health.changed().await.is_ok() {
                if !in_window(SystemMonotonicClock.now_ns()) {
                    continue;
                }
                let rows = health.borrow_and_update().clone();
                let mut rtts = rtts.lock().expect("rtts");
                rtts.extend(rows.iter().filter_map(|r| r.rtt_ns.map(|n| n as i64)));
            }
        }
    });

    let dir = tempfile::Builder::new().prefix("fanout-bench-").tempdir()?;
    rlmesh_session::address::write_private(
        &dir.path().join("session"),
        format!("{session} {key}").as_bytes(),
    )?;
    let mut children = vec![(
        "robot".to_string(),
        spawn(args, "robot", ROBOT, base, dir.path())?,
    )];
    for i in 0..args.readers {
        let id = format!("reader{i}");
        children.push((id.clone(), spawn(args, "reader", &id, base, dir.path())?));
    }
    children.push((
        "prober".into(),
        spawn(args, "prober", "prober", base, dir.path())?,
    ));

    // Everyone present, the prober's lease granted.
    let expected = args.readers + 2;
    let mut watch = host.watch();
    tokio::time::timeout(WAIT * 2, async {
        loop {
            {
                let s = watch.borrow_and_update();
                let present = s.roster.iter().filter(|p| p.present).count();
                if present >= expected && !s.leases.is_empty() {
                    return;
                }
            }
            if watch.changed().await.is_err() {
                return;
            }
        }
    })
    .await
    .map_err(|_| "the participants did not all join")?;
    tokio::time::sleep(args.warmup).await;
    let start = clock.now_ns();
    *window.lock().expect("window") = (start, i64::MAX);
    tokio::time::sleep(args.duration).await;
    let end = clock.now_ns();
    *window.lock().expect("window") = (start, end);
    std::fs::write(dir.path().join("window"), format!("{start} {end}"))?;

    let mut results = BTreeMap::new();
    for (id, child) in children {
        let output = tokio::task::spawn_blocking(move || child.wait_with_output()).await??;
        let text = String::from_utf8_lossy(&output.stdout);
        let value: Value = text
            .lines()
            .last()
            .and_then(|line| serde_json::from_str(line).ok())
            .unwrap_or(Value::Null);
        results.insert(id, value);
    }
    let gaps: Vec<i64> = arrivals
        .lock()
        .expect("arrivals")
        .values()
        .flat_map(|times| {
            times
                .windows(2)
                .filter(|w| (start..end).contains(&w[1]))
                .map(|w| w[1] - w[0])
                .collect::<Vec<_>>()
        })
        .collect();
    events_task.abort();
    health_task.abort();
    host.shutdown().await;
    transport.close().await?;
    let counts = counts.lock().expect("counts").clone();
    let rtts = rtts.lock().expect("rtts").clone();
    Ok(json!({
        "listeners": args.listeners,
        "readers": args.readers,
        "camera": {"hz": args.camera_hz, "width": args.width, "height": args.height},
        "loadavg": std::fs::read_to_string("/proc/loadavg").unwrap_or_default().trim(),
        "ctl_rtt_ms": results.get("prober").map(|p| p["ctl_rtt_ms"].clone()),
        "ctl_failed": results.get("prober").map(|p| p["failed"].clone()),
        "hb_gap_ms": percentiles(gaps),
        "hb_rtt_ms": percentiles(rtts),
        "events": counts,
        "children": results,
    }))
}

async fn robot(args: &Args) -> Result<Value, BoxError> {
    let robot = join(args, ROBOT, vec![Role::Embodiment]).await?;
    robot.declare_parts(&robot_map(args)).await?;
    let ft = Arc::new(robot.publisher("ft").await?);
    let camera = match args.camera_hz {
        0 => None,
        _ => Some(Arc::new(robot.publisher("camera").await?)),
    };
    let dir = args.dir.clone().ok_or("--dir")?;
    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let clock = SystemMonotonicClock;
    let ft_thread = std::thread::spawn({
        let (ft, done) = (ft.clone(), done.clone());
        let period = 1_000_000_000 / i64::from(args.ft_hz.max(1));
        move || {
            let mut next = clock.now_ns();
            while !done.load(std::sync::atomic::Ordering::Relaxed) {
                sleep_until(&clock, next);
                let _ = ft.publish(clock.now_ns(), Bytes::from(vec![0_u8; 72]));
                next += period;
                next = next.max(clock.now_ns() - period);
            }
        }
    });
    let camera_thread = camera.map(|camera| {
        let done = done.clone();
        let period = 1_000_000_000 / i64::from(args.camera_hz.max(1));
        let (w, h) = (args.width as usize, args.height as usize);
        std::thread::spawn(move || {
            let frames: Vec<Bytes> = (0..8)
                .map(|t| {
                    let mut frame = Vec::new();
                    synthetic_frame(w, h, t, &mut frame);
                    Bytes::from(frame)
                })
                .collect();
            let mut next = clock.now_ns();
            let mut i = 0;
            while !done.load(std::sync::atomic::Ordering::Relaxed) {
                sleep_until(&clock, next);
                if let Err(err) = camera.publish(clock.now_ns(), frames[i % frames.len()].clone()) {
                    eprintln!("camera: {err}");
                }
                i += 1;
                next += period;
                next = next.max(clock.now_ns() - period);
            }
        })
    });
    window_of(&dir).await;
    done.store(true, std::sync::atomic::Ordering::Relaxed);
    let _ = ft_thread.join();
    if let Some(t) = camera_thread {
        let _ = t.join();
    }
    robot.close().await;
    Ok(json!({}))
}

async fn reader(args: &Args) -> Result<Value, BoxError> {
    let reader = join(args, &args.id, vec![Role::Observer]).await?;
    let robot = ParticipantId::new(ROBOT)?;
    let has_camera = args.camera_hz > 0;
    reader
        .wait_for("the robot's parts", WAIT, |s| {
            s.part_at(&robot, "ft").is_some()
                && (!has_camera || s.part_at(&robot, "camera").is_some())
        })
        .await?;
    let record = |into: &Arc<Mutex<Vec<(i64, i64)>>>| -> SubscribeOptions {
        let into = into.clone();
        SubscribeOptions {
            on_sample: Some(Arc::new(move |r| {
                into.lock()
                    .expect("samples")
                    .push((r.receive_ns, r.receive_ns - r.sample.capture_ns));
            })),
            ..SubscribeOptions::default()
        }
    };
    let (frames, ft) = (Arc::default(), Arc::default());
    let _ft = reader.subscribe(&robot, "ft", record(&ft)).await?;
    let _camera = match has_camera {
        true => Some(reader.subscribe(&robot, "camera", record(&frames)).await?),
        false => None,
    };
    let dir = args.dir.clone().ok_or("--dir")?;
    let (start, end) = window_of(&dir).await;
    let in_window = |samples: &Arc<Mutex<Vec<(i64, i64)>>>| -> Vec<i64> {
        samples
            .lock()
            .expect("samples")
            .iter()
            .filter(|(t, _)| (start..end).contains(t))
            .map(|(_, l)| *l)
            .collect()
    };
    let frames = in_window(&frames);
    let seconds = (end - start) as f64 / 1e9;
    let result = json!({
        "fps": frames.len() as f64 / seconds,
        "frame_ms": percentiles(frames),
        "ft_ms": percentiles(in_window(&ft)),
    });
    reader.close().await;
    Ok(result)
}

async fn prober(args: &Args) -> Result<Value, BoxError> {
    let prober = join(args, "prober", vec![Role::Controller]).await?;
    let robot = ParticipantId::new(ROBOT)?;
    let view = prober
        .wait_for("the robot's parts", WAIT, |s| {
            s.part_at(&robot, "cmd").is_some()
        })
        .await?;
    let cmd = view.part_at(&robot, "cmd").ok_or("cmd")?.channel;
    prober
        .declare_binding(Fingerprint::from(1), "{}", vec![cmd], 0.0)
        .await?;
    prober.arm().await?;
    let lease = prober.grant_lease(cmd, None).await?;
    let dir = args.dir.clone().ok_or("--dir")?;
    let clock = SystemMonotonicClock;
    let mut samples = Vec::new();
    let mut failed = 0_u64;
    let window = loop {
        if let Ok(text) = std::fs::read_to_string(dir.join("window"))
            && let Some((a, b)) = text.trim().split_once(' ')
            && let (Ok(a), Ok(b)) = (a.parse::<i64>(), b.parse::<i64>())
        {
            break (a, b);
        }
        let asked = clock.now_ns();
        match prober.fetch_authority().await {
            Ok(_) => samples.push((asked, clock.now_ns() - asked)),
            Err(_) => failed += 1,
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    };
    let rtts: Vec<i64> = samples
        .iter()
        .filter(|(t, _)| (window.0..window.1).contains(t))
        .map(|(_, r)| *r)
        .collect();
    let _ = prober.release_lease(&lease).await;
    prober.close().await;
    Ok(json!({"ctl_rtt_ms": percentiles(rtts), "failed": failed}))
}
