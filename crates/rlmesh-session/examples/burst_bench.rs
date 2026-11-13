//! Realtime samples lost in short bursts, through a session host on
//! loopback, for each way the host can listen.
//!
//! ```text
//! burst_bench [--listeners one|four] [--joiner client|peer] [--n 25]
//!     [--spacing-us 1000] [--rounds 40] [--window 10] [--burn 0]
//!     [--camera-hz 0] [--width 640] [--height 480]
//! ```
//!
//! A host, a robot and a reader run in this process, each participant with
//! one Zenoh session per traffic class over TCP, as in the session tests.
//! Each round the robot publishes `--n` samples of a 72-byte Realtime stream
//! `--spacing-us` apart (0: back to back) to a reader subscribed with a
//! `--window`; the host also counts what reaches it, so a loss is placed on
//! the robot's link (missing at the host) or on the host's relay (at the host,
//! missing at the reader). `--burn` spins that many threads meanwhile;
//! `--camera-hz` adds a raw camera (Bulk) the reader also subscribes to.
//!
//! `--listeners one` is how a host listens: one session on one port; `four`
//! is a session per class on four consecutive ports. `--joiner peer` joins
//! as Zenoh peers that listen on a free port each, as the session tests did
//! before the host listened on one port: the host's gossip then links the
//! robot to the reader directly, and samples skip the host's relay.
//!
//! Prints one JSON object: samples published, arrived at the host, and
//! received and out of order at the reader, summed over the rounds, with
//! the rounds that lost anything. `RUST_LOG` (as
//! `zenoh_transport::common::pipeline=trace`) logs to stderr: Zenoh logs
//! the first sample of each drop episode, not those that follow it while
//! the queue stays congested.

#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::collections::BTreeMap;
use std::net::Ipv4Addr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use bytes::Bytes;
use rlmesh_session::media::synthetic_frame;
use rlmesh_session::transport::{Endpoints, Layout};
use rlmesh_session::wire::{ParticipantId, PriorityClass, Role, SessionId, SessionKeys, TenantId};
use rlmesh_session::{
    Clock, HostConfig, Participant, ParticipantConfig, SessionHost, SubscribeOptions,
    SystemMonotonicClock, TrafficClass, Transport, TransportConfig, TransportTuning,
};
use serde_json::json;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

const WAIT: Duration = Duration::from_secs(10);

struct Args {
    listeners: String,
    joiner: String,
    n: u64,
    spacing: Duration,
    rounds: u32,
    window: usize,
    burn: usize,
    camera_hz: u32,
    width: u32,
    height: u32,
}

fn parse_args() -> Result<Args, BoxError> {
    let mut opts = BTreeMap::new();
    let mut raw = std::env::args().skip(1);
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
    let num = |k: &str, d: u64| -> Result<u64, BoxError> {
        Ok(opts.get(k).map_or(Ok(d), |v| v.parse::<u64>())?)
    };
    let text = |k: &str, d: &str| opts.get(k).cloned().unwrap_or_else(|| d.into());
    let args = Args {
        listeners: text("listeners", "one"),
        joiner: text("joiner", "client"),
        n: num("n", 25)?,
        spacing: Duration::from_micros(num("spacing-us", 1000)?),
        rounds: num("rounds", 40)? as u32,
        window: num("window", 10)? as usize,
        burn: num("burn", 0)? as usize,
        camera_hz: num("camera-hz", 0)? as u32,
        width: num("width", 640)? as u32,
        height: num("height", 480)? as u32,
    };
    if !matches!(args.listeners.as_str(), "one" | "four") {
        return Err("--listeners one|four".into());
    }
    if !matches!(args.joiner.as_str(), "client" | "peer") {
        return Err("--joiner client|peer".into());
    }
    Ok(args)
}

fn tcp(port: u16) -> String {
    format!("tcp/{}:{port}", Ipv4Addr::LOCALHOST)
}

/// A base port whose next `n - 1` ports were free a moment ago.
fn free_base(n: u16) -> Result<u16, BoxError> {
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
            return Ok(base);
        }
    }
    Err("no free ports".into())
}

fn host_transport(args: &Args, base: u16) -> TransportConfig {
    match args.listeners.as_str() {
        "four" => TransportConfig {
            layout: Layout::PerClass(std::array::from_fn(|i| Endpoints {
                listen: vec![tcp(base + i as u16)],
                connect: Vec::new(),
            })),
            tuning: TransportTuning::default(),
        },
        _ => TransportConfig::shared(&[&tcp(base)], &[]),
    }
}

fn joiner_transport(args: &Args, base: u16) -> TransportConfig {
    let four = args.listeners == "four";
    TransportConfig {
        layout: Layout::PerClass(std::array::from_fn(|i| Endpoints {
            listen: match args.joiner.as_str() {
                "peer" => vec![tcp(0)],
                _ => Vec::new(),
            },
            connect: vec![tcp(base + if four { i as u16 } else { 0 })],
        })),
        tuning: TransportTuning {
            client: args.joiner == "client",
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
                 "max_age_ms": 100}}"#,
            hz = args.camera_hz,
            w = args.width,
            h = args.height,
        )
    };
    format!(
        r#"{{"ft": {{"kind": "x/ft", "hz": 1000, "fields": {{"wrench": ["float32", [18]]}},
               "priority": "realtime"}}{camera}}}"#
    )
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<(), BoxError> {
    let args = parse_args()?;
    // `RUST_LOG=zenoh_transport::common::pipeline=trace`: where Zenoh drops
    // a sample for want of a free transmit batch.
    if std::env::var_os("RUST_LOG").is_some() {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .with_writer(std::io::stderr)
            .try_init();
    }
    let clock = SystemMonotonicClock;
    let base = free_base(if args.listeners == "four" { 4 } else { 1 })?;
    let transport = Transport::open(&host_transport(&args, base)).await?;
    let tenant = TenantId::new("bench")?;
    let session = SessionId::new();
    let mut config = HostConfig::new(tenant.clone(), session);
    config.clock = clock.shared();
    config.health_log = None;
    let host = SessionHost::start(transport.clone(), config).await?;

    let join = |id: &'static str, roles: Vec<Role>| {
        let transport = joiner_transport(&args, base);
        let mut config = ParticipantConfig::new(
            tenant.clone(),
            session,
            ParticipantId::new(id).expect("id"),
            roles,
            PriorityClass::Model,
        );
        config.clock = clock.shared();
        config.credential = Some(host.credential());
        async move { Participant::attach(Transport::open(&transport).await?, config).await }
    };
    let robot = join("robot", vec![Role::Embodiment]).await?;
    robot.declare_parts(&robot_map(&args)).await?;
    let reader = join("reader", vec![Role::Observer]).await?;
    let robot_id = ParticipantId::new("robot")?;
    reader
        .wait_for("the robot's parts", WAIT, |s| {
            s.part_at(&robot_id, "ft").is_some()
        })
        .await?;
    let ft = robot.publisher("ft").await?;

    // What reaches the host, counted on its own Realtime session.
    let at_host = Arc::new(AtomicU64::new(0));
    let keys = SessionKeys::new(&tenant, session);
    let _at_host = transport
        .session(TrafficClass::Realtime)
        .declare_subscriber(keys.data(&robot_id, "ft"))
        .callback({
            let at_host = at_host.clone();
            move |_| {
                at_host.fetch_add(1, Ordering::Relaxed);
            }
        })
        .await?;
    let sub = reader
        .subscribe(
            &robot_id,
            "ft",
            SubscribeOptions {
                window: args.window,
                ..SubscribeOptions::default()
            },
        )
        .await?;
    let frames = Arc::new(AtomicU64::new(0));
    let _camera_sub = match args.camera_hz {
        0 => None,
        _ => {
            let frames = frames.clone();
            Some(
                reader
                    .subscribe(
                        &robot_id,
                        "camera",
                        SubscribeOptions {
                            on_sample: Some(Arc::new(move |_| {
                                frames.fetch_add(1, Ordering::Relaxed);
                            })),
                            ..SubscribeOptions::default()
                        },
                    )
                    .await?,
            )
        }
    };
    // Probe until a sample travels end to end.
    let started = Instant::now();
    while sub.stats().received == 0 {
        if started.elapsed() > WAIT {
            return Err("no probe reached the reader".into());
        }
        ft.publish(clock.now_ns(), Bytes::from(vec![0_u8; 72]))?;
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    tokio::time::sleep(Duration::from_millis(200)).await;

    let done = Arc::new(AtomicBool::new(false));
    let burners: Vec<_> = (0..args.burn)
        .map(|_| {
            let done = done.clone();
            std::thread::spawn(move || {
                let mut x = 0_u64;
                while !done.load(Ordering::Relaxed) {
                    x = std::hint::black_box(x.wrapping_mul(6_364_136_223_846_793_005) + 1);
                }
            })
        })
        .collect();
    let camera = match args.camera_hz {
        0 => None,
        _ => Some(robot.publisher("camera").await?),
    };
    let camera_thread = match camera {
        Some(camera) => {
            let done = done.clone();
            let period = Duration::from_secs(1) / args.camera_hz;
            let (w, h) = (args.width as usize, args.height as usize);
            Some(std::thread::spawn(move || {
                let frames: Vec<Bytes> = (0..8)
                    .map(|t| {
                        let mut frame = Vec::new();
                        synthetic_frame(w, h, t, &mut frame);
                        Bytes::from(frame)
                    })
                    .collect();
                let mut i = 0;
                while !done.load(Ordering::Relaxed) {
                    let _ = camera.publish(clock.now_ns(), frames[i % frames.len()].clone());
                    i += 1;
                    std::thread::sleep(period);
                }
            }))
        }
        None => None,
    };
    if args.camera_hz > 0 {
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    let (mut published, mut received, mut out_of_order, mut hosted) = (0, 0, 0, 0);
    let mut lossy_rounds = Vec::new();
    let frames_before = frames.load(Ordering::Relaxed);
    let measure_start = Instant::now();
    for round in 0..args.rounds {
        let before = sub.stats();
        let host_before = at_host.load(Ordering::Relaxed);
        let first = ft.last_seq() + 1;
        let (n, spacing) = (args.n, args.spacing);
        let ft_ref = &ft;
        let sent = std::thread::scope(|s| {
            s.spawn(|| {
                let mut sent = 0;
                for i in 0..n {
                    if ft_ref
                        .publish(clock.now_ns(), Bytes::from(vec![i as u8; 72]))
                        .ok()
                        .flatten()
                        .is_some()
                    {
                        sent += 1;
                    }
                    if !spacing.is_zero() {
                        std::thread::sleep(spacing);
                    }
                }
                sent
            })
            .join()
            .expect("publisher")
        });
        // A lost last sample never advances `last_seq`: wait it out.
        let last = first + args.n - 1;
        let waited = Instant::now();
        while sub.stats().last_seq < last && waited.elapsed() < Duration::from_millis(300) {
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        let after = sub.stats();
        let r = after.received - before.received;
        let o = after.out_of_order - before.out_of_order;
        let h = at_host.load(Ordering::Relaxed) - host_before;
        published += sent;
        received += r;
        out_of_order += o;
        hosted += h;
        if r < sent || o > 0 {
            lossy_rounds
                .push(json!({"round": round, "at_host": h, "received": r, "out_of_order": o}));
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let seconds = measure_start.elapsed().as_secs_f64();
    let fps = (frames.load(Ordering::Relaxed) - frames_before) as f64 / seconds;
    done.store(true, Ordering::Relaxed);
    for b in burners {
        let _ = b.join();
    }
    if let Some(t) = camera_thread {
        let _ = t.join();
    }
    println!(
        "{}",
        json!({
            "listeners": args.listeners,
            "joiner": args.joiner,
            "n": args.n,
            "spacing_us": args.spacing.as_micros() as u64,
            "rounds": args.rounds,
            "burn": args.burn,
            "camera_hz": args.camera_hz,
            "camera_fps": fps,
            "loadavg": std::fs::read_to_string("/proc/loadavg").unwrap_or_default().trim(),
            "published": published,
            "at_host": hosted,
            "received": received,
            "out_of_order": out_of_order,
            "lossy_rounds": lossy_rounds,
        })
    );
    drop(sub);
    reader.close().await;
    robot.close().await;
    host.shutdown().await;
    transport.close().await?;
    Ok(())
}
