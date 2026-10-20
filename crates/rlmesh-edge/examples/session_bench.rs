//! The v1/v2 latency benchmark (Haply/UR10 teleop shape, `bench2-spec.md`)
//! on the production crates: an embodiment process (the session host,
//! force/torque at 1 kHz, an optional camera, and the edge with its action
//! queue and fallback) and a controller process that re-plans action chunks
//! from the force/torque stream.
//!
//! ```text
//! session_bench both --camera off|raw|jpeg --horizon 1 --duration-s 30
//! ```
//!
//! `both` runs the embodiment here and spawns the controller as a second OS
//! process; `embodiment` and `controller` run one side each (`--base` names
//! the per-class TCP base port, `--session-file` passes the session id).
//! Every timestamp is the machine's `CLOCK_MONOTONIC`, shared by both
//! processes. The result is one JSON object on stdout:
//!
//! - `control_per_s`: Driving ticks per second at the robot;
//! - `loop_ms`: apply tick minus the capture of the ft sample the applied
//!   action was based on (p50/p99/p99.9/max);
//! - `chunk_age_ms`: chunk receipt at the edge minus that capture;
//! - `miss_pct`: ticks not Driving;
//! - `split`: loop p99 and miss % for the frame tick, the 5 ticks after it,
//!   and the others (T3);
//! - `camera`: frames decoded per second at the controller and capture to
//!   decoded latency;
//! - `safety`: SAFE entries, hold and safe ticks, lease lapses at the host;
//! - `cpu_skipped_ticks`: robot ticks more than 1 ms late;
//! - `snapshot_age_ms`, `transit_ms`: `chunk_age_ms` split at the chunk's
//!   `t0` as the edge converted it (with `--controller bound`, the policy's
//!   snapshot): the ft sample's age at the snapshot, then inference,
//!   packing and the link (per dispatched chunk).
//!
//! Options: `--camera off|raw|jpeg` (640x480 at `--camera-hz 20`; jpeg is
//! q75 with an 8 Mbit/s cap), `--horizon N` actions at 1 ms re-planned every
//! `--replan-ms` (default 1 for N=1, else 10), `--links per-class|shared`,
//! `--warmup-s 3`, `--duration-s 30`.
//!
//! `--controller raw` (the default) re-plans from the transport callback on
//! every `replan`-th ft sample. `--controller bound` is a bound policy
//! instead, the shape `rlmesh.v2` runs from Python: the binder's delivery
//! contracts (ft `latest@R`, the camera an input too), the
//! [`Controller`](rlmesh_session::Controller) run loop at the replan rate,
//! and `t0` at the snapshot. It is the Rust baseline of
//! `examples/python/v2/session_bench.py`.

#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use rlmesh_edge::{Edge, EdgeConfig, EdgeMode};
use rlmesh_session::media::{decode_image, synthetic_frame};
use rlmesh_session::transport::free_local_base;
use rlmesh_session::{
    AuthorityEvent, ChunkDraft, Clock, Controller, ControllerConfig, HostConfig, LeaseEnd,
    Observation, Offer, Participant, ParticipantConfig, Predictor, SessionHost, SubscribeOptions,
    SystemMonotonicClock, Transport, TransportConfig,
};
use rlmesh_spec::apply::{Tensor, Value as ActionValue};
use rlmesh_spec::{Policy, RunOptions};
use rlmesh_wire::{
    Fingerprint, InputRange, Mode, ParticipantId, PriorityClass, Role, Sample, SessionId, TenantId,
};
use serde_json::{Value, json};

const ROBOT: &str = "ur10";
const PERIOD_NS: i64 = 1_000_000;
const RING: usize = 1 << 16;
const FRAMES: u32 = 16;
const WAIT: Duration = Duration::from_secs(20);

type BoxError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Clone, Debug)]
struct Args {
    role: String,
    camera: String,
    camera_hz: u32,
    horizon: usize,
    replan: u64,
    links: String,
    controller: String,
    warmup_ns: i64,
    duration_ns: i64,
    base: Option<std::net::SocketAddr>,
    session_file: Option<PathBuf>,
    out: Option<PathBuf>,
}

fn parse_args() -> Result<Args, BoxError> {
    let mut raw = std::env::args().skip(1);
    let role = raw
        .next()
        .ok_or("usage: session_bench both|embodiment|controller [options]")?;
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
    let horizon = num("horizon", 1.0)?.max(1.0) as usize;
    Ok(Args {
        role,
        camera: opts.get("camera").cloned().unwrap_or_else(|| "off".into()),
        camera_hz: num("camera-hz", 20.0)? as u32,
        horizon,
        replan: num("replan-ms", if horizon == 1 { 1.0 } else { 10.0 })?.max(1.0) as u64,
        links: opts
            .get("links")
            .cloned()
            .unwrap_or_else(|| "per-class".into()),
        controller: opts
            .get("controller")
            .cloned()
            .unwrap_or_else(|| "raw".into()),
        warmup_ns: (num("warmup-s", 3.0)? * 1e9) as i64,
        duration_ns: (num("duration-s", 30.0)? * 1e9) as i64,
        base: opts.get("base").map(|b| b.parse()).transpose()?,
        session_file: opts.get("session-file").map(PathBuf::from),
        out: opts.get("out").map(PathBuf::from),
    })
}

fn to_args(
    args: &Args,
    role: &str,
    base: std::net::SocketAddr,
    session_file: &std::path::Path,
    out: &std::path::Path,
) -> Vec<String> {
    [
        role.to_string(),
        "--camera".into(),
        args.camera.clone(),
        "--camera-hz".into(),
        args.camera_hz.to_string(),
        "--horizon".into(),
        args.horizon.to_string(),
        "--replan-ms".into(),
        args.replan.to_string(),
        "--links".into(),
        args.links.clone(),
        "--controller".into(),
        args.controller.clone(),
        "--warmup-s".into(),
        (args.warmup_ns as f64 / 1e9).to_string(),
        "--duration-s".into(),
        (args.duration_ns as f64 / 1e9).to_string(),
        "--base".into(),
        base.to_string(),
        "--session-file".into(),
        session_file.display().to_string(),
        "--out".into(),
        out.display().to_string(),
    ]
    .into()
}

fn transport_config(args: &Args, listen: bool) -> Result<TransportConfig, BoxError> {
    let base = args.base.ok_or("--base")?;
    let any: std::net::SocketAddr = "127.0.0.1:0".parse()?;
    Ok(match args.links.as_str() {
        "shared" => {
            let ep = format!("tcp/{base}");
            if listen {
                TransportConfig::shared(&[&ep], &[])
            } else {
                TransportConfig::shared(&["tcp/127.0.0.1:0"], &[&ep])
            }
        }
        _ if listen => TransportConfig::per_class_tcp(Some(base), None)?,
        _ => TransportConfig::per_class_tcp(Some(any), Some(base))?,
    })
}

/// The robot map: wrench commands in (velocity-like: no hold), force/torque
/// at 1 kHz and an optional 640x480 camera out.
fn robot_map(args: &Args) -> String {
    let camera = match args.camera.as_str() {
        "off" => String::new(),
        encoding => {
            let (encoding, send) = if encoding == "jpeg" {
                (
                    r#", "encoding": "jpeg", "quality": 75"#,
                    format!("latest(hz={}, max_mbps=8)", args.camera_hz),
                )
            } else {
                ("", format!("latest(hz={})", args.camera_hz))
            };
            format!(
                r#", "camera": {{"kind": "camera", "hz": {}, "width": 640, "height": 480,
                   "max_age_ms": 100, "send": "{send}"{encoding}}}"#,
                args.camera_hz
            )
        }
    };
    format!(
        r#"{{
        "cmd": {{"kind": "x/wrench_cmd", "hz": 1000, "fields": {{"wrench": ["float32", [6]]}},
                "command": "wrench", "priority": "control", "hold_max_ms": 100, "safe": "stop"}},
        "ft": {{"kind": "x/ft", "hz": 1000, "fields": {{"wrench": ["float32", [18]]}},
               "priority": "realtime"}}{camera}
    }}"#
    )
}

fn participant_config(
    session: SessionId,
    id: &str,
    roles: Vec<Role>,
    priority: PriorityClass,
) -> Result<ParticipantConfig, BoxError> {
    let mut config = ParticipantConfig::new(
        TenantId::new("bench")?,
        session,
        ParticipantId::new(id)?,
        roles,
        priority,
    );
    config.clock = SystemMonotonicClock.shared();
    config.ctl_timeout = Duration::from_secs(5);
    Ok(config)
}

/// p50/p99/p99.9/max in ms (nearest rank).
fn percentiles(mut ns: Vec<i64>) -> Value {
    if ns.is_empty() {
        return Value::Null;
    }
    ns.sort_unstable();
    let at = |q: f64| {
        let rank = ((q * ns.len() as f64).ceil() as usize).clamp(1, ns.len());
        ns[rank - 1] as f64 / 1e6
    };
    json!({"n": ns.len(), "p50": at(0.5), "p99": at(0.99), "p999": at(0.999), "max": at(1.0)})
}

/// Sleeps until `deadline_ns`: a coarse sleep, then a short spin.
fn sleep_until(clock: &SystemMonotonicClock, deadline_ns: i64) {
    loop {
        let left = deadline_ns - clock.now_ns();
        if left <= 0 {
            return;
        }
        if left > 200_000 {
            std::thread::sleep(Duration::from_nanos((left - 150_000) as u64));
        } else {
            std::hint::spin_loop();
        }
    }
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<(), BoxError> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .with_writer(std::io::stderr)
        .init();
    let args = parse_args()?;
    match args.role.as_str() {
        "both" => both(args).await,
        "embodiment" => {
            let result = embodiment(&args).await?;
            write_out(&args, &result)
        }
        "controller" => {
            let result = controller(&args).await?;
            write_out(&args, &result)
        }
        other => Err(format!("unknown role {other}").into()),
    }
}

fn write_out(args: &Args, value: &Value) -> Result<(), BoxError> {
    let text = serde_json::to_string_pretty(value)?;
    match &args.out {
        Some(path) => std::fs::write(path, text)?,
        None => println!("{text}"),
    }
    Ok(())
}

async fn both(args: Args) -> Result<(), BoxError> {
    let dir = std::env::temp_dir().join(format!("rlmesh-bench-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let base = free_local_base("127.0.0.1".parse()?)?;
    let session_file = dir.join("session");
    let ctl_out = dir.join("ctl.json");
    let mut child = Command::new(std::env::current_exe()?)
        .args(to_args(&args, "controller", base, &session_file, &ctl_out))
        .stdout(Stdio::null())
        .spawn()?;
    let mut emb_args = args.clone();
    emb_args.base = Some(base);
    emb_args.session_file = Some(session_file);
    let emb = embodiment(&emb_args).await;
    let status = child.wait()?;
    let emb = emb?;
    let ctl: Value = if status.success() {
        serde_json::from_str(&std::fs::read_to_string(&ctl_out)?)?
    } else {
        json!({"error": format!("controller exited with {status}")})
    };
    let _ = std::fs::remove_dir_all(&dir);
    let mut result = emb;
    result["camera"] = ctl["camera"].clone();
    result["controller"] = ctl;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

struct TickRecord {
    capture_ns: i64,
    lateness_ns: i64,
    mode: EdgeMode,
    based_on: Option<u64>,
    /// The dispatched chunk: (token, seq, receipt, t0), edge clock.
    chunk: Option<(u64, u64, i64, i64)>,
}

async fn embodiment(args: &Args) -> Result<Value, BoxError> {
    let transport = Transport::open(&transport_config(args, true)?).await?;
    let session = SessionId::new();
    let mut host_config = HostConfig::new(TenantId::new("bench")?, session);
    host_config.clock = SystemMonotonicClock.shared();
    host_config.health_log = None;
    let host = SessionHost::start(transport.clone(), host_config).await?;
    let host_events = Arc::new(Mutex::new(BTreeMap::<&'static str, u64>::new()));
    let events_task = tokio::spawn({
        let mut events = host.events();
        let host_events = host_events.clone();
        async move {
            while let Ok(event) = events.recv().await {
                let key = match &event {
                    AuthorityEvent::LeaseEnded {
                        reason: LeaseEnd::Lapsed,
                        ..
                    } => "lease_lapsed",
                    AuthorityEvent::ModeChanged { to: Mode::Safe, .. } => "mode_safe",
                    _ => continue,
                };
                if let Ok(mut counts) = host_events.lock() {
                    *counts.entry(key).or_default() += 1;
                }
            }
        }
    });
    let robot = Participant::attach(
        transport.clone(),
        participant_config(
            session,
            ROBOT,
            vec![Role::Embodiment],
            PriorityClass::Scripted,
        )?,
    )
    .await?;
    let declared = robot.declare_parts(&robot_map(args)).await?;
    let robot_id = robot.id().clone();
    let cmd = declared.part_at(&robot_id, "cmd").ok_or("cmd")?.channel;
    let ft_channel = declared.part_at(&robot_id, "ft").ok_or("ft")?.channel;
    if let Some(path) = &args.session_file {
        std::fs::write(path, session.to_string())?;
    }

    let clock = SystemMonotonicClock;
    let ft_capture: Arc<Vec<AtomicI64>> = Arc::new((0..RING).map(|_| AtomicI64::new(0)).collect());
    // (based-on capture, receipt - capture) per accepted chunk.
    let chunk_ages: Arc<Mutex<Vec<(i64, i64)>>> = Arc::new(Mutex::new(Vec::with_capacity(40_000)));
    let edge = Arc::new(
        Edge::attach(
            &robot,
            "cmd",
            EdgeConfig {
                log_capacity: 0,
                ..EdgeConfig::default()
            },
            Some({
                let (ft_capture, chunk_ages) = (ft_capture.clone(), chunk_ages.clone());
                Arc::new(move |chunk, receive_ns| {
                    let Some(based) = chunk.based_on.iter().find(|r| r.channel == ft_channel)
                    else {
                        return;
                    };
                    let capture =
                        ft_capture[based.seq_last as usize % RING].load(Ordering::Relaxed);
                    if let Ok(mut ages) = chunk_ages.lock() {
                        ages.push((capture, receive_ns - capture));
                    }
                })
            }),
        )
        .await?,
    );
    robot
        .wait_for(
            "the controller's binding and a clock mapping",
            Duration::from_secs(60),
            // A bound controller leases only once its inputs are warm: the
            // samples start at its binding.
            |s| {
                (s.lease(cmd).is_some() || s.bindings.iter().any(|b| b.channels.contains(&cmd)))
                    && s.clock(robot.id()).is_some()
            },
        )
        .await?;
    tokio::time::sleep(Duration::from_millis(500)).await;
    let ft = Arc::new(robot.publisher("ft").await?);
    let camera = match args.camera.as_str() {
        "off" => None,
        _ => Some(Arc::new(robot.image_publisher("camera").await?)),
    };
    tokio::time::sleep(Duration::from_millis(300)).await;

    let stop = Arc::new(AtomicBool::new(false));
    let start_ns = clock.now_ns();
    let total_end = start_ns + args.warmup_ns + args.duration_ns + 200_000_000;
    let ft_thread = std::thread::spawn({
        let (ft, ft_capture, edge) = (ft.clone(), ft_capture.clone(), edge.clone());
        move || {
            let mut records: Vec<TickRecord> = Vec::with_capacity(40_000);
            let mut next = start_ns;
            let mut tick: u64 = 0;
            let mut value = [0_f32; 18];
            let mut skipped = 0_i64;
            while next < total_end {
                sleep_until(&clock, next);
                let capture = clock.now_ns();
                tick += 1;
                value[0] = tick as f32;
                let bytes: Vec<u8> = value.iter().flat_map(|v| v.to_le_bytes()).collect();
                let seq = ft.last_seq() + 1;
                ft_capture[seq as usize % RING].store(capture, Ordering::Relaxed);
                if let Err(err) = ft.publish(capture, Bytes::from(bytes)) {
                    eprintln!("ft publish: {err}");
                }
                let applied = edge.tick();
                let driving = applied
                    .action
                    .as_ref()
                    .filter(|_| applied.mode == EdgeMode::Driving);
                records.push(TickRecord {
                    capture_ns: capture,
                    lateness_ns: capture - next,
                    mode: applied.mode,
                    based_on: driving.and_then(|a| {
                        a.chunk
                            .based_on
                            .iter()
                            .find(|r| r.channel == ft_channel)
                            .map(|r| r.seq_last)
                    }),
                    chunk: driving.map(|a| {
                        let t0 = a.slot_ns - i64::from(a.index) * a.dt_ns;
                        (a.chunk.lease_token.0, a.chunk.seq, a.chunk.receive_ns, t0)
                    }),
                });
                next += PERIOD_NS;
                let now = clock.now_ns();
                if now > next + PERIOD_NS {
                    let n = (now - next) / PERIOD_NS;
                    skipped += n;
                    next += n * PERIOD_NS;
                }
            }
            (records, skipped)
        }
    });
    let cam_thread = camera.clone().map(|camera| {
        let stop = stop.clone();
        let period_ns = 1_000_000_000 / i64::from(args.camera_hz.max(1));
        std::thread::spawn(move || {
            let frames: Vec<Vec<u8>> = (0..FRAMES)
                .map(|t| {
                    let mut frame = Vec::new();
                    synthetic_frame(640, 480, t, &mut frame);
                    frame
                })
                .collect();
            let mut next = start_ns;
            let mut i = 0_usize;
            while next < total_end && !stop.load(Ordering::Relaxed) {
                sleep_until(&clock, next);
                match camera.offer(clock.now_ns(), &frames[i % frames.len()]) {
                    Ok(Offer::Queued { .. } | Offer::OverBudget | Offer::Coalesced) => {}
                    Err(err) => eprintln!("camera: {err}"),
                }
                i += 1;
                next += period_ns;
            }
        })
    });
    let (records, cpu_skipped) = ft_thread.join().map_err(|_| "ft thread panicked")?;
    stop.store(true, Ordering::Relaxed);
    if let Some(t) = cam_thread {
        t.join().map_err(|_| "camera thread panicked")?;
    }
    // Let the controller see the stream end.
    tokio::time::sleep(Duration::from_secs(2)).await;

    let anchor = records.first().map_or(start_ns, |r| r.capture_ns);
    let (win_start, win_end) = (
        anchor + args.warmup_ns,
        anchor + args.warmup_ns + args.duration_ns,
    );
    let in_window = |c: i64| (win_start..win_end).contains(&c);
    let window: Vec<&TickRecord> = records.iter().filter(|r| in_window(r.capture_ns)).collect();
    let capture_of = |seq: u64| ft_capture[seq as usize % RING].load(Ordering::Relaxed);
    let age = |r: &TickRecord| r.based_on.map(|seq| r.capture_ns - capture_of(seq));
    let duration_s = args.duration_ns as f64 / 1e9;
    let mut mode_ticks = BTreeMap::<String, u64>::new();
    for r in &window {
        *mode_ticks.entry(r.mode.to_string()).or_default() += 1;
    }
    let driving = mode_ticks.get("driving").copied().unwrap_or(0);
    let miss_pct = |ticks: &[&&TickRecord]| {
        100.0 * ticks.iter().filter(|r| r.mode != EdgeMode::Driving).count() as f64
            / ticks.len().max(1) as f64
    };
    let split = (args.camera != "off" && args.camera_hz > 0 && 1000 % args.camera_hz == 0).then(|| {
        let period = i64::from(1000 / args.camera_hz);
        let class = |c: i64| match ((c - anchor + PERIOD_NS / 2).div_euclid(PERIOD_NS)).rem_euclid(period) {
            0 => "frame_tick",
            r if r <= 5 => "after_5_ticks",
            _ => "other_ticks",
        };
        ["frame_tick", "after_5_ticks", "other_ticks"]
            .iter()
            .map(|name| {
                let ticks: Vec<&&TickRecord> = window.iter().filter(|r| class(r.capture_ns) == *name).collect();
                let ages: Vec<i64> = ticks.iter().filter_map(|r| age(r)).collect();
                (name.to_string(), json!({"ticks": ticks.len(), "loop_ms": percentiles(ages), "miss_pct": miss_pct(&ticks)}))
            })
            .collect::<BTreeMap<_, _>>()
    });
    let all: Vec<&&TickRecord> = window.iter().collect();
    let chunk_ages: Vec<i64> = chunk_ages
        .lock()
        .map(|a| {
            a.iter()
                .filter(|(c, _)| in_window(*c))
                .map(|(_, a)| *a)
                .collect()
        })
        .unwrap_or_default();
    // Each dispatched chunk once, split at its t0.
    let mut seen = std::collections::BTreeSet::new();
    let (mut snapshot_age, mut transit) = (Vec::new(), Vec::new());
    for r in &window {
        if let (Some((token, seq, receipt, t0)), Some(based)) = (r.chunk, r.based_on)
            && seen.insert((token, seq))
        {
            snapshot_age.push(t0 - capture_of(based));
            transit.push(receipt - t0);
        }
    }
    let stats = edge.stats();
    let result = json!({
        "config": {
            "camera": args.camera, "camera_hz": args.camera_hz, "horizon": args.horizon,
            "replan_ms": args.replan, "links": args.links, "duration_s": duration_s,
            "controller": args.controller,
        },
        "loadavg": std::fs::read_to_string("/proc/loadavg").unwrap_or_default().trim(),
        "ticks": window.len(),
        "control_per_s": driving as f64 / duration_s,
        "loop_ms": percentiles(window.iter().filter_map(|r| age(r)).collect()),
        "chunk_age_ms": percentiles(chunk_ages),
        "snapshot_age_ms": percentiles(snapshot_age),
        "transit_ms": percentiles(transit),
        "miss_pct": miss_pct(&all),
        "split": split,
        "mode_ticks": mode_ticks,
        "cpu_skipped_ticks": cpu_skipped,
        "late_ticks_over_1ms": window.iter().filter(|r| r.lateness_ns > 1_000_000).count(),
        "safety": {
            "safe_entries": stats.mode(EdgeMode::Safe).entries,
            "hold_ticks": stats.mode(EdgeMode::Hold).ticks,
            "safe_ticks": stats.mode(EdgeMode::Safe).ticks,
            "host": host_events.lock().map(|e| e.clone()).unwrap_or_default(),
        },
        "edge_rejected": stats.rejected.iter().map(|(r, n)| (format!("{r:?}"), *n)).collect::<BTreeMap<_, _>>(),
        "media": camera.as_ref().map(|c| { let m = c.stats(); json!({"sent": m.sent, "skipped_over_budget": m.skipped_over_budget, "superseded": m.superseded, "encode_ms_mean": m.encode_ns_total as f64 / m.encoded.max(1) as f64 / 1e6}) }),
    });
    drop(camera);
    drop(edge);
    robot.close().await;
    events_task.abort();
    host.shutdown().await;
    transport.close().await?;
    Ok(result)
}

/// The bound policy of `--controller bound`: the ft wrench in (and the
/// camera, so its frames cross the link), `horizon` 1 ms actions out.
fn bound_policy(args: &Args) -> String {
    let camera = if args.camera == "off" {
        String::new()
    } else {
        r#", "camera": {"kind": "camera", "at": "camera"}"#.to_string()
    };
    format!(
        r#"{{"hz": {hz},
            "inputs": {{"ft": {{"kind": "x/ft", "at": "ft", "fields": {{"wrench": ["float32", [18]]}}}}{camera}}},
            "actions": {{"cmd": {{"kind": "x/wrench_cmd", "at": "cmd", "command": "wrench",
                                 "fields": {{"wrench": ["float32", [6]]}}, "hz": 1000, "horizon": {h}}}}}}}"#,
        hz = 1000.0 / args.replan as f64,
        h = args.horizon,
    )
}

/// The "model" of `--controller bound`: hold the first six floats of the
/// wrench for the whole horizon.
struct HoldWrench {
    horizon: u32,
}

impl Predictor for HoldWrench {
    type Error = String;

    fn predict(
        &mut self,
        observation: Observation,
        _: i64,
    ) -> Result<Vec<(String, ActionValue)>, String> {
        let Some(ActionValue::Tensor(ft)) = observation.value("ft") else {
            return Err("no ft".into());
        };
        let held = &ft.data[..24];
        Ok(vec![(
            "cmd".into(),
            ActionValue::Tensor(Tensor {
                dtype: ft.dtype,
                shape: vec![self.horizon, 6],
                data: held.repeat(self.horizon as usize),
            }),
        )])
    }
}

async fn bound_controller(args: &Args, policy: Participant) -> Result<Value, BoxError> {
    let options = RunOptions::named("policy").robot(ROBOT)?;
    let controller = Arc::new(
        Controller::bind(
            Arc::new(policy),
            &Policy::parse(&bound_policy(args))?,
            &options,
            ControllerConfig {
                bind_timeout: WAIT,
                ..ControllerConfig::default()
            },
        )
        .await?,
    );
    controller.participant().arm().await?;
    // The embodiment measures warmup + duration after its first sample.
    let run_for = Duration::from_nanos((args.warmup_ns + args.duration_ns).unsigned_abs())
        + Duration::from_secs(2);
    let horizon = u32::try_from(args.horizon)?;
    let runner = controller.clone();
    tokio::task::spawn_blocking(move || {
        runner
            .run(&mut HoldWrench { horizon }, Some(run_for))
            .map_err(|err| format!("{err:?}"))
    })
    .await??;
    let stats = controller.stats();
    let controller = Arc::try_unwrap(controller).map_err(|_| "controller still shared")?;
    let participant = controller.close().await;
    if let Ok(participant) = Arc::try_unwrap(participant) {
        participant.close().await;
    }
    Ok(json!({
        "run": {
            "starts": stats.starts,
            "completions": stats.completions,
            "chunks_sent": stats.chunks_sent,
            "skipped": rlmesh_session::SkipReason::ALL
                .iter()
                .map(|r| (r.name(), stats.skipped(*r)))
                .collect::<BTreeMap<_, _>>(),
            "latency_p99_ms": stats.latency_p99.map(|d| d.as_secs_f64() * 1e3),
        },
    }))
}

async fn controller(args: &Args) -> Result<Value, BoxError> {
    let transport = Transport::open(&transport_config(args, false)?).await?;
    let session_file = args.session_file.as_ref().ok_or("--session-file")?;
    let session: SessionId = loop {
        if let Ok(text) = std::fs::read_to_string(session_file)
            && let Ok(id) = text.trim().parse()
        {
            break id;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    let policy = Participant::attach(
        transport.clone(),
        participant_config(
            session,
            "policy",
            vec![Role::Controller],
            PriorityClass::Model,
        )?,
    )
    .await?;
    if args.controller == "bound" {
        let result = bound_controller(args, policy).await;
        transport.close().await?;
        return result;
    }
    let robot = ParticipantId::new(ROBOT)?;
    let view = policy
        .wait_for("parts", WAIT, |s| s.part_at(&robot, "ft").is_some())
        .await?;
    let cmd = view.part_at(&robot, "cmd").ok_or("cmd")?.channel;
    let ft = view.part_at(&robot, "ft").ok_or("ft")?.channel;
    policy
        .declare_binding(Fingerprint::from(1), "{}", vec![cmd], 0.0)
        .await?;
    policy.arm().await?;
    let lease = policy.grant_lease(cmd, None).await?;
    let cmd = Arc::new(policy.chunk_publisher(&lease).await?);
    let clock = SystemMonotonicClock;
    let last_rx = Arc::new(AtomicI64::new(0));
    let send_errors = Arc::new(AtomicU64::new(0));
    // EWMA of the ft one-way delay: where the next chunk will land.
    let expected_ns = Arc::new(AtomicI64::new(0));
    let (horizon, replan) = (args.horizon, args.replan);
    let _ft = policy
        .subscribe(
            &robot,
            "ft",
            SubscribeOptions {
                on_sample: Some({
                    let (cmd, last_rx, errors, expected) = (
                        cmd.clone(),
                        last_rx.clone(),
                        send_errors.clone(),
                        expected_ns.clone(),
                    );
                    Arc::new(move |r| {
                        let capture = r.sample.capture_ns;
                        last_rx.store(r.receive_ns, Ordering::Relaxed);
                        let delay = r.receive_ns - capture;
                        let old = expected.load(Ordering::Relaxed);
                        expected.store(
                            if old == 0 {
                                delay
                            } else {
                                old + (delay - old) / 16
                            },
                            Ordering::Relaxed,
                        );
                        if r.sample.seq % replan != 0 {
                            return;
                        }
                        // The "model": hold the first six floats of the wrench.
                        let action = r.sample.payload.slice(..24.min(r.sample.payload.len()));
                        let actions = if horizon == 1 {
                            action
                        } else {
                            Bytes::from(action.repeat(horizon))
                        };
                        let draft = ChunkDraft {
                            based_on: vec![InputRange::one(ft, r.sample.seq)],
                            t0_ns: capture + PERIOD_NS,
                            dt_ns: PERIOD_NS as u64,
                            replace_from_ns: r.receive_ns + expected.load(Ordering::Relaxed),
                            valid_until_ns: capture + 500_000_000,
                            actions,
                            tick: 0,
                        };
                        if cmd.send(draft).is_err() {
                            errors.fetch_add(1, Ordering::Relaxed);
                        }
                    })
                }),
                ..SubscribeOptions::default()
            },
        )
        .await?;
    // Camera frames are decoded off the transport thread.
    let decoded = Arc::new(Mutex::new(Vec::<(i64, i64)>::new()));
    let (frame_tx, frame_rx) = std::sync::mpsc::channel::<Sample>();
    let decoding = Arc::new(AtomicBool::new(true));
    let decoder = {
        let (decoded, decoding) = (decoded.clone(), decoding.clone());
        let spec = policy
            .part(&robot, "camera")
            .ok()
            .and_then(|part| part.stream().ok());
        std::thread::spawn(move || {
            let Some(spec) = spec else {
                return;
            };
            while decoding.load(Ordering::Relaxed) {
                let Ok(sample) = frame_rx.recv_timeout(Duration::from_millis(100)) else {
                    continue;
                };
                if decode_image(&spec, &sample).is_ok()
                    && let Ok(mut d) = decoded.lock()
                {
                    d.push((sample.capture_ns, clock.now_ns() - sample.capture_ns));
                }
            }
        })
    };
    let _cam = match args.camera.as_str() {
        "off" => None,
        _ => Some(
            policy
                .subscribe(
                    &robot,
                    "camera",
                    SubscribeOptions {
                        on_sample: Some({
                            let frame_tx = Mutex::new(frame_tx);
                            Arc::new(move |r| {
                                if let Ok(tx) = frame_tx.lock() {
                                    let _ = tx.send(r.sample.clone());
                                }
                            })
                        }),
                        ..SubscribeOptions::default()
                    },
                )
                .await?,
        ),
    };
    // Run until the stream has started and then been quiet for 1.5 s.
    loop {
        tokio::time::sleep(Duration::from_millis(250)).await;
        let quiet = last_rx.load(Ordering::Relaxed);
        if quiet != 0 && clock.now_ns() - quiet > 1_500_000_000 {
            break;
        }
    }
    decoding.store(false, Ordering::Relaxed);
    let _ = decoder.join();
    // The camera window: frames captured in the embodiment's measurement
    // window, which starts warmup after the first ft capture.
    let frames = decoded.lock().map(|d| d.clone()).unwrap_or_default();
    let first = frames.first().map_or(0, |(c, _)| *c);
    let (start, end) = (
        first + args.warmup_ns,
        first + args.warmup_ns + args.duration_ns,
    );
    let in_window: Vec<i64> = frames
        .iter()
        .filter(|(c, _)| (start..end).contains(c))
        .map(|(_, l)| *l)
        .collect();
    let duration_s = args.duration_ns as f64 / 1e9;
    let result = json!({
        "camera": (args.camera != "off").then(|| json!({
            "fps": in_window.len() as f64 / duration_s,
            "latency_ms": percentiles(in_window),
        })),
        "chunk_send_errors": send_errors.load(Ordering::Relaxed),
    });
    let _ = policy.release_lease(&lease).await;
    policy.close().await;
    transport.close().await?;
    Ok(result)
}
