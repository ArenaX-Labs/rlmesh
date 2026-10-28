//! The browser example's robot (`examples/web/`): a session host, an
//! animated 7-DOF arm and a model policy that holds it, in one process, on
//! the production crates.
//!
//! ```text
//! cargo run -p rlmesh-session --example web_robot -- --listen-base 127.0.0.1:17600 --session-file session.id
//! ```
//!
//! - **Host**: a `SessionHost` on per-class TCP at `--listen-base` (class i on
//!   port + i), lease grace `--lease-grace-ms` (250 ms).
//! - **Robot** `arm7` (`rlmesh_session::edge::Robot`): `arm` (7 joints, `pos` and `vel`
//!   at 1 kHz, position commands, `safe: stop`, hold 100 ms), `gripper`
//!   (100 Hz, position, `safe: hold`) and `scene/cam` (640x480 at 20 Hz, JPEG
//!   q85; `--cam raw` sends the raw RGB). The control loop runs at 1 kHz; each
//!   tick is the edge tick of each command group (`Robot::due`).
//! - **Policy** (`--policy on`): a model-class controller that holds the
//!   arm and sways it, re-requesting the lease whenever it loses it. An
//!   operator (the browser) preempts it; when the operator releases, the
//!   policy takes the arm back.
//!
//! A held lease whose heartbeats stop lapses after the grace: the session
//! goes SAFE, the arm stops, and `arm` (any controller may) starts a fresh
//! epoch. The robot prints its edge counters every 5 s.

#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::f64::consts::PI;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use clap::Parser;
use rlmesh_session::address::AddressKey;
use rlmesh_session::controller::sleep_until;
use rlmesh_session::edge::{EdgeConfig, EdgeMode, Robot};
use rlmesh_session::spec::apply::{Tensor, Value};
use rlmesh_session::spec::{Policy, RunOptions};
use rlmesh_session::wire::{DType, Mode, ParticipantId, PriorityClass, Role, SessionId, TenantId};
use rlmesh_session::{
    Controller, ControllerConfig, HostConfig, Observation, Participant, ParticipantConfig,
    Predictor, SessionHost, SystemMonotonicClock, Transport, TransportConfig,
};

const ROBOT: &str = "arm7";
const JOINTS: usize = 7;
const PERIOD_NS: i64 = 1_000_000;
const MAX_VEL: f64 = 1.5;
const LIMIT: f64 = 2.6;

#[derive(Debug, Parser)]
struct Args {
    /// Per-class TCP base the host listens on.
    #[arg(long, default_value = "127.0.0.1:17600")]
    listen_base: SocketAddr,
    /// The tenant namespace.
    #[arg(long, default_value = "local")]
    tenant: String,
    /// The session id (fresh by default).
    #[arg(long)]
    session: Option<String>,
    /// Write the session id and its key (`{session}#{key}`) here once the
    /// robot is up.
    #[arg(long)]
    session_file: Option<PathBuf>,
    /// The camera's encoding: jpeg or raw.
    #[arg(long, default_value = "jpeg")]
    cam: String,
    /// Run the model policy that holds the arm (on or off).
    #[arg(long, default_value = "on")]
    policy: String,
    /// How long a lease holder may stay silent before its lease lapses.
    #[arg(long, default_value_t = 250)]
    lease_grace_ms: u64,
    /// Stop after this long (0: run until interrupted).
    #[arg(long, default_value_t = 0)]
    duration_s: u64,
}

fn robot_map(cam: &str) -> String {
    let (encoding, budget) = match cam {
        "raw" => (r#""encoding": "raw""#, 400),
        _ => (r#""encoding": "jpeg", "quality": 85"#, 40),
    };
    format!(
        r#"{{
        "arm": {{"kind": "joints", "count": {JOINTS}, "fields": ["pos", "vel"], "hz": 1000,
                "command": "position", "safe": "stop", "hold_max_ms": 100,
                "max_action_age_ms": 1000, "limits": {{"pos": [-{LIMIT}, {LIMIT}]}}}},
        "gripper": {{"kind": "gripper", "hz": 100, "command": "position", "safe": "hold",
                    "hold_max_ms": 200, "max_action_age_ms": 1000}},
        "scene/cam": {{"kind": "camera", "width": 640, "height": 480, "hz": 20, {encoding},
                      "send": "latest(max_mbps={budget})"}}
    }}"#
    )
}

/// Replans at 50 Hz, ten 10 ms actions per chunk: a slow sway of every joint.
const POLICY: &str = r#"{
    "hz": 50,
    "actions": {"arm": {"kind": "joints", "at": "arm", "command": "position", "hz": 100, "horizon": 10}}
}"#;

struct Sway;

impl Predictor for Sway {
    type Error = String;

    fn predict(
        &mut self,
        _: Observation,
        snapshot_ns: i64,
    ) -> Result<Vec<(String, Value)>, String> {
        let mut data = Vec::with_capacity(10 * JOINTS * 4);
        for k in 0..10 {
            let t = (snapshot_ns as f64) / 1e9 + k as f64 * 0.01;
            for j in 0..JOINTS {
                let q = 0.35 * (2.0 * PI * 0.15 * t + j as f64 * 0.7).sin();
                data.extend_from_slice(&(q as f32).to_le_bytes());
            }
        }
        Ok(vec![(
            "arm".into(),
            Value::Tensor(Tensor {
                dtype: DType::F32,
                shape: vec![10, JOINTS as u32],
                data,
            }),
        )])
    }
}

fn f32s(values: &[f64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| (*v as f32).to_le_bytes())
        .collect()
}

fn read_f32s(bytes: &[u8], out: &mut [f64]) {
    let (words, _) = bytes.as_chunks::<4>();
    for (o, b) in out.iter_mut().zip(words) {
        *o = f64::from(f32::from_le_bytes(*b));
    }
}

/// What the camera thread draws.
#[derive(Clone, Default)]
struct View {
    q: [f64; JOINTS],
    gripper: f64,
    edge: String,
    mode: String,
    holder: String,
}

async fn join(
    base: SocketAddr,
    tenant: &TenantId,
    (session, key): (SessionId, &AddressKey),
    id: &str,
    role: Role,
    priority: PriorityClass,
) -> Result<Participant, Box<dyn std::error::Error>> {
    // Listen too: peers dial the others they learn of directly.
    let any = SocketAddr::new(base.ip(), 0);
    let transport =
        Transport::open(&TransportConfig::per_class_tcp(Some(any), Some(base))?).await?;
    let mut config = ParticipantConfig::new(
        tenant.clone(),
        session,
        ParticipantId::new(id)?,
        vec![role],
        priority,
    );
    config.clock = SystemMonotonicClock.shared();
    config.heartbeat_period = Duration::from_millis(20);
    config.credential = Some(key.credential());
    Ok(Participant::attach(transport, config).await?)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,zenoh=warn".into()),
        )
        .with_writer(std::io::stderr)
        .init();
    let args = Args::parse();
    let tenant = TenantId::new(&args.tenant)?;
    let session = match &args.session {
        Some(id) => id.parse()?,
        None => SessionId::new(),
    };
    let transport = Transport::open(&TransportConfig::per_class_tcp(
        Some(args.listen_base),
        None,
    )?)
    .await?;
    let mut host = HostConfig::new(tenant.clone(), session);
    host.clock = SystemMonotonicClock.shared();
    host.lease_grace = Duration::from_millis(args.lease_grace_ms.max(1));
    host.health_log = None;
    let key = AddressKey::of(&host.identity);
    let host = SessionHost::start(transport, host).await?;

    let base = args.listen_base;
    let embodiment = join(
        base,
        &tenant,
        (session, &key),
        ROBOT,
        Role::Embodiment,
        PriorityClass::Scripted,
    )
    .await?;
    let robot =
        Arc::new(Robot::declare(embodiment, &robot_map(&args.cam), EdgeConfig::default()).await?);
    robot.participant().arm().await?;

    let policy = if args.policy == "on" {
        let p = join(
            base,
            &tenant,
            (session, &key),
            "policy",
            Role::Controller,
            PriorityClass::Model,
        )
        .await?;
        let controller = Controller::bind(
            Arc::new(p),
            &Policy::parse(POLICY)?,
            &RunOptions::named("policy").robot(ROBOT)?,
            ControllerConfig::default(),
        )
        .await?;
        let controller = Arc::new(controller);
        let runner = controller.clone();
        std::thread::spawn(move || {
            if let Err(err) = runner.run(&mut Sway, None) {
                eprintln!("policy stopped: {err}");
            }
        });
        Some(controller)
    } else {
        None
    };

    if let Some(path) = &args.session_file {
        // The session and its key, as the gateway's --session-file takes them.
        std::fs::write(path, format!("{session}#{key}"))?;
    }
    println!("SESSION {session}");
    println!("READY robot {ROBOT} on {base}");

    let stop = Arc::new(AtomicBool::new(false));
    let view = Arc::new(Mutex::new(View::default()));
    let control = std::thread::spawn({
        let (robot, stop, view) = (robot.clone(), stop.clone(), view.clone());
        move || control_loop(&robot, &stop, &view)
    });
    let camera = std::thread::spawn({
        let (robot, stop, view) = (robot.clone(), stop.clone(), view.clone());
        move || camera_loop(&robot, &stop, &view)
    });
    let report = tokio::spawn({
        let robot = robot.clone();
        async move {
            let mut tick = tokio::time::interval(Duration::from_secs(5));
            tick.tick().await;
            loop {
                tick.tick().await;
                if let Ok(stats) = robot.edge_stats("arm") {
                    let mode = robot.participant().authority().map(|v| v.mode);
                    eprintln!(
                        "[robot] mode {} arm accepted {} rejected {:?} safe reports {}",
                        mode.map_or("-".into(), |m| m.to_string()),
                        stats.accepted,
                        stats.rejected,
                        stats.safe_reports
                    );
                }
            }
        }
    });

    let duration = (args.duration_s > 0).then(|| Duration::from_secs(args.duration_s));
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        () = async { match duration { Some(d) => tokio::time::sleep(d).await, None => std::future::pending().await } } => {}
        () = terminate() => {}
    }
    eprintln!("[robot] shutting down");
    stop.store(true, Ordering::Relaxed);
    report.abort();
    let _ = control.join();
    let _ = camera.join();
    if let Some(policy) = policy {
        policy.stop();
    }
    host.shutdown().await;
    Ok(())
}

#[cfg(unix)]
async fn terminate() {
    use tokio::signal::unix::{SignalKind, signal};
    match signal(SignalKind::terminate()) {
        Ok(mut term) => {
            term.recv().await;
        }
        Err(_) => std::future::pending().await,
    }
}

#[cfg(not(unix))]
async fn terminate() {
    std::future::pending().await
}

/// 1 kHz: each tick is the edge tick of both groups, then the arm's sample;
/// the gripper publishes every 10th tick.
fn control_loop(robot: &Robot, stop: &AtomicBool, view: &Mutex<View>) {
    let clock = robot.participant().clock().clone();
    let now = || clock.now_ns();
    let dt = PERIOD_NS as f64 / 1e9;
    let (mut q, mut dq, mut target) = ([0.0; JOINTS], [0.0; JOINTS], [0.0; JOINTS]);
    let (mut grip, mut grip_target) = (0.5_f64, 0.5_f64);
    let mut edge = EdgeMode::Idle;
    let mut next = now();
    let mut tick: u64 = 0;
    while !stop.load(Ordering::Relaxed) {
        let t = now();
        match robot.due("arm") {
            Ok(Some(due)) => {
                edge = due.mode;
                if let Some(action) = &due.action {
                    read_f32s(&action.value, &mut target);
                }
                if due.safe.is_some() {
                    target = q;
                }
                if due.handover.is_some() {
                    let _ = robot.handover_done("arm");
                }
            }
            Ok(None) => edge = EdgeMode::Idle,
            Err(err) => eprintln!("[robot] arm: {err}"),
        }
        let stopped = matches!(edge, EdgeMode::Safe | EdgeMode::Estop);
        for j in 0..JOINTS {
            dq[j] = if stopped {
                0.0
            } else {
                ((target[j] - q[j]) * 25.0).clamp(-MAX_VEL, MAX_VEL)
            };
            q[j] = (q[j] + dq[j] * dt).clamp(-LIMIT, LIMIT);
        }
        if let Err(err) = robot.publish("arm", t, &[&f32s(&q), &f32s(&dq)]) {
            eprintln!("[robot] publish arm: {err}");
        }
        if tick.is_multiple_of(10) {
            if let Ok(Some(due)) = robot.due("gripper") {
                if let Some(action) = &due.action {
                    let mut v = [0.0];
                    read_f32s(&action.value, &mut v);
                    grip_target = v[0];
                }
                if due.handover.is_some() {
                    let _ = robot.handover_done("gripper");
                }
            }
            grip += ((grip_target - grip) * 0.2).clamp(-0.02, 0.02);
            let _ = robot.publish("gripper", t, &[&f32s(&[grip])]);
        }
        if tick.is_multiple_of(20)
            && let Ok(mut v) = view.lock()
        {
            v.q = q;
            v.gripper = grip;
            v.edge = edge.to_string();
            if let Some(authority) = robot.participant().authority() {
                v.mode = authority.mode.to_string();
                v.holder = robot
                    .part("arm")
                    .and_then(|p| authority.lease(p.channel))
                    .map_or("-".into(), |l| l.holder.to_string());
                if authority.mode == Mode::Safe {
                    v.edge = "safe".into();
                }
            }
        }
        tick += 1;
        next += PERIOD_NS;
        if now() - next > 50 * PERIOD_NS {
            // Far behind (a stall): restart the grid instead of bursting.
            next = now();
        }
        sleep_until(now, next);
    }
}

/// 20 Hz: render the arm and offer the frame (the image publisher encodes it
/// on its own thread).
fn camera_loop(robot: &Robot, stop: &AtomicBool, view: &Mutex<View>) {
    let clock = robot.participant().clock().clone();
    let now = || clock.now_ns();
    let mut renderer = render::Renderer::new();
    let mut frame = 0_u64;
    let mut next = now();
    while !stop.load(Ordering::Relaxed) {
        let t = now();
        let v = view.lock().map(|v| v.clone()).unwrap_or_default();
        let rgb = renderer.render(&render::Scene {
            q: &v.q,
            gripper: v.gripper,
            frame,
            mode: &v.mode,
            edge: &v.edge,
            holder: &v.holder,
            t_ms: t as f64 / 1e6,
        });
        if let Err(err) = robot.publish("scene/cam", t, &[rgb]) {
            eprintln!("[robot] camera: {err}");
        }
        frame += 1;
        next += 50 * PERIOD_NS;
        if now() > next {
            next = now();
        }
        sleep_until(now, next);
    }
}

mod render;
