//! The edge over a real session: a host and participants linked over
//! per-class TCP on localhost, and controllers and observers in second OS
//! processes.

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bytes::Bytes;
use rlmesh_edge::{Edge, EdgeConfig, EdgeMode, EdgeTick, Rejection};
use rlmesh_session::transport::open_local_listeners;
use rlmesh_session::{
    ChunkDraft, HostConfig, Participant, ParticipantConfig, SessionHost, SubscribeOptions,
    SystemMonotonicClock, Transport, TransportConfig,
};
use rlmesh_wire::{
    ChannelId, Fingerprint, InputRange, Lease, Mode, ParticipantId, PriorityClass, Role, SessionId,
    TenantId,
};
use serde_json::json;

const WAIT: Duration = Duration::from_secs(5);
const GRACE: Duration = Duration::from_millis(300);
const CHILD: &str = "RLMESH_EDGE_TEST_CHILD";
/// How long a new chunk route gets to settle before the first send.
const WARMUP: Duration = Duration::from_millis(100);

/// `cmd`: velocity-style joint commands (no hold), 2 s open-loop budget;
/// `ft`: force/torque; `arm` + `arm/tcp`: a command group.
const ROBOT: &str = r#"{
    "cmd": {"kind": "x/joints", "hz": 1000, "fields": {"pos": ["float32", [6]]},
            "command": "position", "priority": "control", "hold_max_ms": 100,
            "max_action_age_ms": 2000},
    "ft": {"kind": "x/ft", "hz": 1000, "fields": {"wrench": ["float32", [18]]},
           "priority": "realtime"},
    "arm": {"kind": "joints", "hz": 1000, "count": 6, "fields": ["pos"],
            "command": "position", "group": "arm", "hold_max_ms": 100},
    "arm/tcp": {"kind": "x/tcp", "hz": 1000, "fields": {"dpos": ["float32", [6]]},
                "command": "delta", "group": "arm", "hold_max_ms": 100}
}"#;

fn any_local_port() -> std::net::SocketAddr {
    "127.0.0.1:0".parse().expect("addr")
}

fn robot_id() -> ParticipantId {
    ParticipantId::new("robot").expect("id")
}

struct World {
    base: std::net::SocketAddr,
    tenant: TenantId,
    session: SessionId,
    host: SessionHost,
}

impl World {
    async fn start() -> Self {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "warn".into()),
            )
            .with_test_writer()
            .try_init();
        let (transport, base) = open_local_listeners("127.0.0.1".parse().expect("ip"))
            .await
            .expect("transport");
        let tenant = TenantId::new("edge").expect("tenant");
        let session = SessionId::new();
        let mut config = HostConfig::new(tenant.clone(), session);
        // One clock domain for every process of the test: the machine's.
        config.clock = SystemMonotonicClock.shared();
        config.lease_grace = GRACE;
        config.health_log = None;
        let host = SessionHost::start(transport, config).await.expect("host");
        Self {
            base,
            tenant,
            session,
            host,
        }
    }

    async fn join(&self, id: &str, roles: Vec<Role>, priority: PriorityClass) -> Participant {
        join(self.base, &self.tenant, self.session, id, roles, priority).await
    }

    async fn robot(&self) -> Participant {
        let robot = self
            .join("robot", vec![Role::Embodiment], PriorityClass::Scripted)
            .await;
        robot.declare_parts(ROBOT).await.expect("declare");
        robot
    }

    async fn controller(&self, id: &str, priority: PriorityClass) -> Participant {
        let p = self.join(id, vec![Role::Controller], priority).await;
        bind(&p, Fingerprint::from(priority as u64 + 1)).await;
        p
    }

    /// Spawns this test binary as a child running `scenario`.
    fn child(&self, scenario: &str, stdout: Stdio) -> Child {
        Command::new(std::env::current_exe().expect("test binary"))
            .args(["--exact", "child_process", "--nocapture"])
            .env(
                CHILD,
                format!("{},{},{},{scenario}", self.base, self.session, self.tenant),
            )
            .stdout(stdout)
            .spawn()
            .expect("spawn the child process")
    }
}

async fn join(
    base: std::net::SocketAddr,
    tenant: &TenantId,
    session: SessionId,
    id: &str,
    roles: Vec<Role>,
    priority: PriorityClass,
) -> Participant {
    let transport = Transport::open(
        &TransportConfig::per_class_tcp(Some(any_local_port()), Some(base)).expect("config"),
    )
    .await
    .expect("transport");
    let mut config = ParticipantConfig::new(
        tenant.clone(),
        session,
        ParticipantId::new(id).expect("id"),
        roles,
        priority,
    );
    config.clock = SystemMonotonicClock.shared();
    config.heartbeat_period = Duration::from_millis(20);
    Participant::attach(transport, config)
        .await
        .expect("attach")
}

/// Binds `p` to every commandable part of the robot.
async fn bind(p: &Participant, fingerprint: Fingerprint) {
    let view = p
        .wait_for("the robot's parts", WAIT, |s| {
            s.part_at(&robot_id(), "arm/tcp").is_some()
        })
        .await
        .expect("parts");
    let channels = view
        .parts
        .iter()
        .filter(|part| part.command.is_some())
        .map(|part| part.channel)
        .collect();
    p.declare_binding(fingerprint, "{}", channels, 0.0)
        .await
        .expect("bind");
}

fn channel(p: &Participant, path: &str) -> ChannelId {
    p.part(&robot_id(), path).expect(path).channel
}

async fn wait_until(what: &str, mut done: impl FnMut() -> bool) {
    let started = Instant::now();
    while !done() {
        assert!(started.elapsed() < WAIT, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Waits until the participant has a clock mapping (the edge refuses chunks
/// before it, and the chunk's source, have one).
async fn mapped(p: &Participant) {
    p.wait_for("a clock mapping", WAIT, |s| s.clock(p.id()).is_some())
        .await
        .expect("mapping");
}

/// A 1 kHz control loop on its own thread, as an embodiment runs one. The
/// robot's side of a handover (switching its control mode) is instant here.
fn tick_at_1khz(
    edge: Arc<Edge>,
    stop: Arc<AtomicBool>,
    ticks: Option<Arc<Mutex<Vec<EdgeTick>>>>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        while !stop.load(Ordering::Relaxed) {
            let tick = edge.tick();
            if tick.handover.is_some() {
                edge.handover_done();
            }
            if let Some(ticks) = &ticks
                && let Ok(mut ticks) = ticks.lock()
            {
                ticks.push(tick);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    })
}

/// `n` actions at 1 ms from now (session time), every byte `tag`.
fn horizon(p: &Participant, n: usize, tag: u8, valid_for: Duration) -> ChunkDraft {
    let now = p.session_now_ns();
    ChunkDraft {
        based_on: vec![InputRange::one(ChannelId(99), 1)],
        t0_ns: now + 1_000_000,
        dt_ns: 1_000_000,
        replace_from_ns: now,
        valid_until_ns: now + i64::try_from(valid_for.as_nanos()).expect("ns"),
        actions: Bytes::from(vec![tag; 24 * n]),
        tick: 0,
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn the_edge_fences_a_preempted_controller_and_drops_expired_chunks() {
    let world = World::start().await;
    let robot = world.robot().await;
    let policy = world.controller("policy", PriorityClass::Model).await;
    let operator = world.controller("operator", PriorityClass::Operator).await;
    let cmd = channel(&policy, "cmd");
    let edge = Edge::attach(&robot, "cmd", EdgeConfig::default(), None)
        .await
        .expect("edge");
    for p in [&robot, &policy, &operator] {
        mapped(p).await;
    }

    let armed = operator.arm().await.expect("arm");
    policy
        .wait_for("armed", WAIT, |s| s.epoch == armed.epoch)
        .await
        .expect("view");
    let model = policy.grant_lease(cmd, None).await.expect("lease");
    robot
        .wait_for("robot sees the model lease", WAIT, |s| {
            s.lease(cmd).is_some_and(|l| l.token == model.token)
        })
        .await
        .expect("view");
    let model_tx = policy.chunk_publisher(&model).await.expect("publisher");
    tokio::time::sleep(Duration::from_millis(200)).await;
    model_tx
        .send(horizon(&policy, 50, 1, Duration::from_millis(100)))
        .expect("send");
    wait_until("model chunk accepted", || edge.stats().accepted == 1).await;
    tokio::time::sleep(Duration::from_millis(2)).await;
    assert_eq!(
        edge.tick().action.expect("due").chunk.source,
        *policy.id(),
        "the model drives"
    );

    // Already void on arrival.
    model_tx
        .send(horizon(&policy, 50, 1, Duration::ZERO))
        .expect("send");
    wait_until("expired chunk rejected", || {
        edge.stats().rejected(Rejection::Expired) == 1
    })
    .await;

    // The operator jumps in; the model's in-flight chunks are fenced.
    let human = operator.grant_lease(cmd, None).await.expect("preempt");
    robot
        .wait_for("robot sees the preemption", WAIT, |s| {
            s.lease(cmd).is_some_and(|l| l.token == human.token)
        })
        .await
        .expect("view");
    let tick = edge.tick();
    assert_eq!(
        tick.mode,
        EdgeMode::Hold,
        "the fenced queue is not acted on"
    );
    assert!(tick.action.is_none() && tick.safe_behavior.is_some());
    model_tx
        .send(horizon(&policy, 50, 1, Duration::from_millis(100)))
        .expect("send");
    wait_until("model chunk fenced", || {
        edge.stats().rejected(Rejection::StaleToken) == 1
    })
    .await;
    let human_tx = operator.chunk_publisher(&human).await.expect("publisher");
    tokio::time::sleep(Duration::from_millis(50)).await;
    human_tx
        .send(horizon(&operator, 50, 2, Duration::from_millis(100)))
        .expect("send");
    wait_until("operator chunk accepted", || edge.stats().accepted == 2).await;
    tokio::time::sleep(Duration::from_millis(2)).await;
    let tick = edge.tick();
    assert_eq!(tick.mode, EdgeMode::Driving);
    assert_eq!(tick.action.expect("due").chunk.source, *operator.id());
    let stats = edge.stats();
    assert_eq!(stats.rejected_total(), 2, "{stats:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn chunks_stream_through_a_preemption_mid_chunk_and_the_old_queue_is_fenced() {
    let world = World::start().await;
    let robot = world.robot().await;
    let policy = world.controller("policy", PriorityClass::Model).await;
    let operator = world.controller("operator", PriorityClass::Operator).await;
    let cmd = channel(&policy, "cmd");
    let edge = Arc::new(
        Edge::attach(&robot, "cmd", EdgeConfig::default(), None)
            .await
            .expect("edge"),
    );
    for p in [&robot, &policy, &operator] {
        mapped(p).await;
    }
    let armed = operator.arm().await.expect("arm");
    policy
        .wait_for("armed", WAIT, |s| s.epoch == armed.epoch)
        .await
        .expect("view");
    let model = policy.grant_lease(cmd, None).await.expect("lease");
    robot
        .wait_for("robot sees the model lease", WAIT, |s| {
            s.lease(cmd).is_some()
        })
        .await
        .expect("view");
    let model_tx = policy.chunk_publisher(&model).await.expect("publisher");
    tokio::time::sleep(Duration::from_millis(200)).await;

    let stop = Arc::new(AtomicBool::new(false));
    let ticker = tick_at_1khz(edge.clone(), stop.clone(), None);
    // The model re-plans a 50 ms horizon every 10 ms.
    let streaming = Arc::new(AtomicBool::new(true));
    let model_loop = tokio::spawn({
        let streaming = streaming.clone();
        async move {
            while streaming.load(Ordering::Relaxed) {
                let _ = model_tx.send(horizon(&policy, 50, 1, Duration::from_millis(150)));
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            policy
        }
    });
    tokio::time::sleep(Duration::from_millis(300)).await;
    // The operator jumps in mid-chunk; the model keeps sending.
    let human = operator.grant_lease(cmd, None).await.expect("preempt");
    let human_tx = operator.chunk_publisher(&human).await.expect("publisher");
    robot
        .wait_for("robot sees the preemption", WAIT, |s| {
            s.lease(cmd).is_some_and(|l| l.token == human.token)
        })
        .await
        .expect("view");
    for _ in 0..30 {
        let _ = human_tx.send(horizon(&operator, 50, 2, Duration::from_millis(150)));
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    streaming.store(false, Ordering::Relaxed);
    let _policy = model_loop.await.expect("model loop");
    stop.store(true, Ordering::Relaxed);
    ticker.join().expect("ticker");

    let log = edge.drain_log();
    let driving: Vec<(String, u8)> = log
        .iter()
        .filter(|a| a.mode == EdgeMode::Driving)
        .filter_map(|a| a.action.as_ref())
        .map(|q| (q.chunk.source.to_string(), q.value[0]))
        .collect();
    let first_human = driving
        .iter()
        .position(|(source, _)| source == "operator")
        .expect("the operator drove");
    assert!(
        first_human > 100,
        "the model drove first ({first_human} ticks)"
    );
    assert!(
        driving[first_human..]
            .iter()
            .all(|(source, tag)| source == "operator" && *tag == 2),
        "no model action after the operator's first"
    );
    let stats = edge.stats();
    assert!(
        stats.queue.fenced > 0,
        "the model's queued actions were dropped: {stats:?}"
    );
    assert!(stats.rejected(Rejection::StaleToken) > 0, "{stats:?}");
    assert_eq!(stats.mode(EdgeMode::Safe).entries, 0, "{stats:?}");
    assert_eq!(world.host.snapshot().mode, Mode::Active);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_controller_that_stops_sending_holds_then_the_edge_takes_the_session_safe() {
    let world = World::start().await;
    let robot = world.robot().await;
    let policy = world.controller("policy", PriorityClass::Model).await;
    let cmd = channel(&policy, "cmd");
    let edge = Arc::new(
        Edge::attach(&robot, "cmd", EdgeConfig::default(), None)
            .await
            .expect("edge"),
    );
    mapped(&robot).await;
    mapped(&policy).await;
    policy.arm().await.expect("arm");
    let lease = policy.grant_lease(cmd, None).await.expect("lease");
    robot
        .wait_for("robot sees the lease", WAIT, |s| s.lease(cmd).is_some())
        .await
        .expect("view");
    let tx = policy.chunk_publisher(&lease).await.expect("publisher");
    tokio::time::sleep(Duration::from_millis(200)).await;
    let stop = Arc::new(AtomicBool::new(false));
    let ticker = tick_at_1khz(edge.clone(), stop.clone(), None);
    // One 20 ms chunk, then silence (heartbeats continue: the host alone
    // would never lapse the lease).
    tx.send(horizon(&policy, 20, 1, Duration::from_millis(150)))
        .expect("send");
    let snapshot = robot
        .wait_for("the edge's EnterSafe reaches the host", WAIT, |s| {
            s.mode == Mode::Safe
        })
        .await
        .expect("SAFE");
    assert!(snapshot.lease(cmd).is_none(), "SAFE revokes the lease");
    stop.store(true, Ordering::Relaxed);
    ticker.join().expect("ticker");
    let stats = edge.stats();
    assert_eq!(stats.safe_reports, 1, "{stats:?}");
    assert_eq!(stats.mode(EdgeMode::Safe).entries, 1);
    let hold = stats.mode(EdgeMode::Hold);
    // hold_max is 100 ms.
    assert!(
        (90_000_000..=130_000_000).contains(&hold.time_ns),
        "held {} ms",
        hold.time_ns / 1_000_000
    );
    assert_eq!(world.host.snapshot().mode, Mode::Safe);
}

#[tokio::test(flavor = "multi_thread")]
async fn with_the_host_gone_the_edge_falls_back_on_its_own_clock() {
    let world = World::start().await;
    let robot = world.robot().await;
    let policy = world.controller("policy", PriorityClass::Model).await;
    let cmd = channel(&policy, "cmd");
    let edge = Arc::new(
        Edge::attach(&robot, "cmd", EdgeConfig::default(), None)
            .await
            .expect("edge"),
    );
    mapped(&robot).await;
    mapped(&policy).await;
    policy.arm().await.expect("arm");
    let lease = policy.grant_lease(cmd, None).await.expect("lease");
    robot
        .wait_for("robot sees the lease", WAIT, |s| s.lease(cmd).is_some())
        .await
        .expect("view");
    let tx = policy.chunk_publisher(&lease).await.expect("publisher");
    tokio::time::sleep(Duration::from_millis(200)).await;
    // A long horizon (the part's open-loop budget is 2 s): the queue would
    // drive for a second, the most one chunk carries.
    tx.send(horizon(&policy, 1024, 1, Duration::from_secs(2)))
        .expect("send");
    wait_until("chunk accepted", || edge.stats().accepted == 1).await;
    let stop = Arc::new(AtomicBool::new(false));
    let ticker = tick_at_1khz(edge.clone(), stop.clone(), None);
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(edge.tick().mode, EdgeMode::Driving);

    let World { host, .. } = world;
    let gone = Instant::now();
    host.shutdown().await;
    wait_until("the edge falls back without the host", || {
        edge.stats().mode(EdgeMode::Safe).entries == 1
    })
    .await;
    let after = gone.elapsed();
    stop.store(true, Ordering::Relaxed);
    ticker.join().expect("ticker");
    // Within the lease grace (300 ms) plus the host's renewal cadence.
    assert!(
        after < GRACE + Duration::from_millis(150),
        "after {after:?}"
    );
    let stats = edge.stats();
    assert_eq!(stats.mode(EdgeMode::Hold).ticks, 0, "no hold: {stats:?}");
    assert_eq!(stats.safe_reports, 1);
    let tick = edge.tick();
    assert_eq!(tick.mode, EdgeMode::Safe);
    assert_eq!(tick.safe_behavior.as_deref(), Some("stop"));
}

/// Streams `tag` chunks under `lease` for `duration`, after letting a new
/// route settle for `warmup`.
async fn stream(p: &Participant, lease: &Lease, tag: u8, duration: Duration, warmup: Duration) {
    let tx = p.chunk_publisher(lease).await.expect("publisher");
    tokio::time::sleep(warmup).await;
    let stride = tx.stride();
    let started = Instant::now();
    while started.elapsed() < duration {
        let mut draft = horizon(p, 50, tag, Duration::from_millis(150));
        draft.actions = Bytes::from(vec![tag; stride * 50]);
        tx.send(draft).expect("send");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// The child half of the cross-process tests: runs only when a parent test
/// launches this binary with [`CHILD`] set to `base,session,tenant,scenario`.
#[tokio::test(flavor = "multi_thread")]
async fn child_process() {
    let Ok(spec) = std::env::var(CHILD) else {
        return;
    };
    let mut parts = spec.split(',');
    let base: std::net::SocketAddr = parts.next().expect("base").parse().expect("base");
    let session: SessionId = parts.next().expect("session").parse().expect("session");
    let tenant = TenantId::new(parts.next().expect("tenant")).expect("tenant");
    let scenario = parts.next().expect("scenario").to_string();
    if scenario == "observe-applied" {
        let observer = join(
            base,
            &tenant,
            session,
            "observer",
            vec![Role::Observer],
            PriorityClass::Scripted,
        )
        .await;
        observer
            .wait_for("the robot's parts", WAIT, |s| {
                s.part_at(&robot_id(), "cmd").is_some()
            })
            .await
            .expect("parts");
        let seen = Arc::new(Mutex::new(Vec::new()));
        let _applied = observer
            .subscribe_applied(
                &robot_id(),
                "cmd",
                SubscribeOptions {
                    on_sample: Some({
                        let seen = seen.clone();
                        Arc::new(move |r| {
                            if let (Ok(mut seen), Some(applied)) = (seen.lock(), r.sample.applied) {
                                seen.push((applied, r.sample.payload.first().copied()));
                            }
                        })
                    }),
                    ..SubscribeOptions::default()
                },
            )
            .await
            .expect("subscribe");
        // Ready: the parent starts driving once it sees this line.
        println!("CHILD_READY");
        tokio::time::sleep(Duration::from_millis(1500)).await;
        let seen = seen.lock().expect("seen").clone();
        let present: Vec<_> = seen.iter().filter(|(a, _)| a.present).collect();
        let report = json!({
            "samples": seen.len(),
            "present": present.len(),
            "driving": seen.iter().filter(|(a, _)| a.mode == rlmesh_wire::EdgeMode::Driving).count(),
            "tokens": present.iter().map(|(a, _)| a.lease_token.0).collect::<std::collections::BTreeSet<_>>(),
            "tags": present.iter().filter_map(|(_, tag)| *tag).collect::<std::collections::BTreeSet<_>>(),
            "seq_ordered": present.windows(2).all(|w| w[0].0.chunk_seq <= w[1].0.chunk_seq),
            "absent_empty": seen.iter().filter(|(a, _)| !a.present).all(|(_, tag)| tag.is_none()),
        });
        println!("CHILD_REPORT {report}");
        observer.close().await;
        std::process::exit(0);
    }
    let policy = join(
        base,
        &tenant,
        session,
        "remote-policy",
        vec![Role::Controller],
        PriorityClass::Model,
    )
    .await;
    bind(&policy, Fingerprint::from(70)).await;
    let armed = policy
        .wait_for("armed", WAIT, |s| s.mode == Mode::Armed)
        .await
        .expect("armed");
    let cmd = armed.part_at(&robot_id(), "cmd").expect("cmd").channel;
    match scenario.as_str() {
        "crash" => {
            let lease = policy.grant_lease(cmd, None).await.expect("lease");
            stream(&policy, &lease, 7, Duration::from_millis(600), WARMUP).await;
            // A crash: no detach, no release.
            std::process::exit(0);
        }
        "rebind" => {
            let lease = policy.grant_lease(cmd, None).await.expect("lease");
            stream(&policy, &lease, 7, Duration::from_millis(400), WARMUP).await;
            // A new binding: the lease comes back under a fresh token.
            let view = policy
                .declare_binding(Fingerprint::from(71), r#"{"rev": 2}"#, vec![cmd], 0.0)
                .await
                .expect("rebind");
            let rebound = view.lease(cmd).expect("rebound").clone();
            assert!(rebound.token > lease.token);
            stream(
                &policy,
                &rebound,
                8,
                Duration::from_millis(400),
                Duration::ZERO,
            )
            .await;
            policy.release_lease(&rebound).await.expect("release");
        }
        "handover" => {
            let (arm, tcp) = (
                armed.part_at(&robot_id(), "arm").expect("arm").channel,
                armed.part_at(&robot_id(), "arm/tcp").expect("tcp").channel,
            );
            let joint = policy.grant_lease(arm, None).await.expect("arm");
            stream(&policy, &joint, 1, Duration::from_millis(400), WARMUP).await;
            // Switch to the Cartesian member in one revision.
            let leases = policy
                .grant_leases(vec![tcp], vec![arm], None)
                .await
                .expect("switch");
            stream(
                &policy,
                &leases[0],
                2,
                Duration::from_millis(400),
                Duration::ZERO,
            )
            .await;
            policy.release_lease(&leases[0]).await.expect("release");
        }
        other => panic!("unknown scenario {other}"),
    }
    policy.close().await;
    std::process::exit(0);
}

/// The robot side of the cross-process tests: the edge on `path`, mapped,
/// armed, ticking at 1 kHz with its ticks recorded.
struct Driven {
    robot: Participant,
    edge: Arc<Edge>,
    stop: Arc<AtomicBool>,
    ticker: Option<std::thread::JoinHandle<()>>,
    ticks: Arc<Mutex<Vec<EdgeTick>>>,
}

impl Driven {
    async fn start(world: &World, path: &str) -> Self {
        let robot = world.robot().await;
        let edge = Arc::new(
            Edge::attach(&robot, path, EdgeConfig::default(), None)
                .await
                .expect("edge"),
        );
        mapped(&robot).await;
        robot.arm().await.expect("arm");
        let stop = Arc::new(AtomicBool::new(false));
        let ticks = Arc::new(Mutex::new(Vec::new()));
        let ticker = tick_at_1khz(edge.clone(), stop.clone(), Some(ticks.clone()));
        Self {
            robot,
            edge,
            stop,
            ticker: Some(ticker),
            ticks,
        }
    }

    fn stop(&mut self) -> Vec<EdgeTick> {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(ticker) = self.ticker.take() {
            ticker.join().expect("ticker");
        }
        std::mem::take(&mut *self.ticks.lock().expect("ticks"))
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_controller_in_another_process_drives_the_edge_and_its_crash_falls_back() {
    let world = World::start().await;
    let mut driven = Driven::start(&world, "cmd").await;
    let mut child = world.child("crash", Stdio::null());
    let remote = ParticipantId::new("remote-policy").expect("id");
    wait_until("the remote controller's chunks drive the edge", || {
        driven.edge.stats().mode(EdgeMode::Driving).ticks > 100
    })
    .await;
    // Its crash: the queue runs dry (or its lease lapses) and the session
    // goes SAFE.
    let snapshot = driven
        .robot
        .wait_for("SAFE after the controller process died", WAIT, |s| {
            s.mode == Mode::Safe
        })
        .await
        .expect("safe");
    let cmd = channel(&driven.robot, "cmd");
    assert!(snapshot.lease(cmd).is_none());
    wait_until("the edge latches safe", || {
        driven.edge.stats().mode(EdgeMode::Safe).entries == 1
    })
    .await;
    driven.stop();
    assert!(child.wait().expect("child").success());

    let log = driven.edge.drain_log();
    assert!(
        log.iter()
            .filter_map(|a| a.action.as_ref())
            .all(|q| q.chunk.source == remote && q.value[0] == 7),
        "every applied action is attributed to the remote controller"
    );
    let stats = driven.edge.stats();
    assert_eq!(stats.mode(EdgeMode::Safe).entries, 1, "{stats:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_controller_in_another_process_rebinds_and_its_old_token_is_fenced() {
    let world = World::start().await;
    let mut driven = Driven::start(&world, "cmd").await;
    let mut child = world.child("rebind", Stdio::null());
    assert!(child.wait().expect("child").success());
    let ticks = driven.stop();
    let remote = ParticipantId::new("remote-policy").expect("id");
    assert_eq!(
        world
            .host
            .snapshot()
            .binding(&remote)
            .map(|b| b.fingerprint),
        Some(Fingerprint::from(71)),
        "the host records the new binding"
    );
    let dispatched: Vec<(u64, u8)> = ticks
        .iter()
        .filter(|t| t.mode == EdgeMode::Driving)
        .filter_map(|t| t.action.as_ref())
        .map(|a| (a.chunk.lease_token.0, a.value[0]))
        .collect();
    let first_new = dispatched
        .iter()
        .position(|(_, tag)| *tag == 8)
        .expect("the rebound token drove");
    assert!(first_new > 100, "the first binding drove first");
    let (old, new) = (dispatched[0].0, dispatched[first_new].0);
    assert!(new > old, "a fresh token: {old} -> {new}");
    assert!(
        dispatched[..first_new].iter().all(|d| *d == (old, 7)),
        "the first binding's chunks under its own token"
    );
    assert!(
        dispatched[first_new..].iter().all(|d| *d == (new, 8)),
        "nothing of the old token after the new one drove"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_group_hands_over_between_members_driven_from_another_process() {
    let world = World::start().await;
    let mut driven = Driven::start(&world, "arm").await;
    assert_eq!(driven.edge.channels().len(), 2, "the whole group");
    let (arm, tcp) = (
        channel(&driven.robot, "arm"),
        channel(&driven.robot, "arm/tcp"),
    );
    let mut child = world.child("handover", Stdio::null());
    assert!(child.wait().expect("child").success());
    let ticks = driven.stop();
    let handovers: Vec<_> = ticks.iter().filter_map(|t| t.handover.clone()).collect();
    assert_eq!(handovers.len(), 1, "{handovers:?}");
    assert_eq!((handovers[0].from, handovers[0].to), (arm, tcp));
    assert_eq!(handovers[0].group, "arm");
    let at = ticks
        .iter()
        .position(|t| t.handover.is_some())
        .expect("handover tick");
    let driving = |t: &EdgeTick| t.mode == EdgeMode::Driving && t.action.is_some();
    assert!(
        ticks[..at]
            .iter()
            .filter(|t| driving(t))
            .all(|t| t.channel == Some(arm)),
        "only the joint member drove before"
    );
    let after: Vec<_> = ticks[at..].iter().filter(|t| driving(t)).collect();
    assert!(after.len() > 50, "the Cartesian member drove after");
    assert!(
        after.iter().all(|t| t.channel == Some(tcp)),
        "never the joint member again"
    );
    // Every tick publishes one applied sample per member; only the driving
    // member's is present.
    for t in &ticks {
        assert_eq!(t.applied.len(), 2);
        let present: Vec<_> = t.applied.iter().filter(|a| a.applied.present).collect();
        assert!(present.len() <= 1);
        if let Some(a) = present.first() {
            assert_eq!(Some(a.part), t.channel);
        }
    }
    assert_eq!(driven.edge.stats().handovers, 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_applied_stream_reaches_an_observer_in_another_process() {
    let world = World::start().await;
    let mut driven = Driven::start(&world, "cmd").await;
    let policy = world.controller("policy", PriorityClass::Model).await;
    mapped(&policy).await;
    let cmd = channel(&policy, "cmd");
    let mut child = world.child("observe-applied", Stdio::piped());
    let mut lines = BufReader::new(child.stdout.take().expect("stdout")).lines();
    let ready = tokio::task::spawn_blocking(move || {
        for line in lines.by_ref() {
            if line.as_ref().is_ok_and(|l| l.contains("CHILD_READY")) {
                return lines;
            }
        }
        panic!("the child never got ready");
    })
    .await
    .expect("ready");
    policy
        .wait_for("armed", WAIT, |s| s.mode == Mode::Armed)
        .await
        .expect("armed");
    let lease = policy.grant_lease(cmd, None).await.expect("lease");
    stream(&policy, &lease, 5, Duration::from_millis(600), WARMUP).await;
    let report = tokio::task::spawn_blocking(move || {
        for line in ready {
            let line = line.expect("line");
            if let Some(json) = line.split_once("CHILD_REPORT ").map(|(_, j)| j.to_string()) {
                return serde_json::from_str::<serde_json::Value>(&json).expect("json");
            }
        }
        panic!("no report");
    })
    .await
    .expect("report");
    assert!(child.wait().expect("child").success());
    driven.stop();
    // One sample per 1 kHz tick reached the other process: absent ones
    // (Idle, then Hold) carry no value, present ones the dispatched action
    // under the lease's token, in chunk order.
    assert!(report["samples"].as_u64().expect("n") > 500, "{report}");
    assert!(report["present"].as_u64().expect("n") > 300, "{report}");
    assert!(report["driving"].as_u64().expect("n") > 300, "{report}");
    assert_eq!(report["tokens"], json!([lease.token.0]), "{report}");
    assert_eq!(report["tags"], json!([5]), "{report}");
    assert_eq!(report["seq_ordered"], json!(true), "{report}");
    assert_eq!(report["absent_empty"], json!(true), "{report}");
}
