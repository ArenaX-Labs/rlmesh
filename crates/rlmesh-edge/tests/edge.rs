//! The edge over a real session: a host and participants linked over
//! per-class TCP on localhost, and a controller in a second OS process.

use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use bytes::Bytes;
use rlmesh_edge::{Edge, EdgeConfig, EdgeMode, Rejection};
use rlmesh_session::transport::free_local_base;
use rlmesh_session::{
    ChunkDraft, HostConfig, Participant, ParticipantConfig, SessionHost, SystemMonotonicClock,
    Transport, TransportConfig,
};
use rlmesh_wire::{
    Actuation, ChannelId, ChannelPriority, ChannelQos, ChannelSpec, Direction, MergePolicy, Mode,
    ObservationRef, ParticipantId, PriorityClass, Role, SessionId, TenantId, TensorLayout,
};

const WAIT: Duration = Duration::from_secs(5);
const GRACE: Duration = Duration::from_millis(300);
const CMD: ChannelId = ChannelId(1);
const FT: ChannelId = ChannelId(2);
const CHILD: &str = "RLMESH_EDGE_TEST_CHILD";

fn any_local_port() -> std::net::SocketAddr {
    "127.0.0.1:0".parse().expect("addr")
}

fn cmd_spec(hold_last: bool) -> ChannelSpec {
    let mut cmd = ChannelSpec::new(
        CMD,
        "joint_cmd",
        Direction::Actuate,
        "action/joint_pos",
        vec![TensorLayout::f32(&[6])],
        1000.0,
        ChannelQos::realtime(ChannelPriority::Control),
    );
    cmd.actuation = Some(Actuation {
        merge: MergePolicy::Replace,
        hold_max: Duration::from_millis(100),
        safe_behavior: "stop".into(),
        hold_last,
    });
    cmd
}

fn channels() -> Vec<ChannelSpec> {
    vec![
        cmd_spec(false),
        ChannelSpec::new(
            FT,
            "ft",
            Direction::Sense,
            "proprio/eef_wrench",
            vec![TensorLayout::f32(&[18])],
            1000.0,
            ChannelQos::realtime(ChannelPriority::Realtime),
        ),
    ]
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
        let base = free_local_base("127.0.0.1".parse().expect("ip")).expect("ports");
        let transport =
            Transport::open(&TransportConfig::per_class_tcp(Some(base), None).expect("config"))
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
        robot.declare_channels(channels()).await.expect("declare");
        robot
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

async fn wait_until(what: &str, mut done: impl FnMut() -> bool) {
    let started = Instant::now();
    while !done() {
        assert!(started.elapsed() < WAIT, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Waits until the participant has a clock mapping (the edge refuses chunks
/// before it has one).
async fn mapped(p: &Participant) {
    p.wait_for("a clock mapping", WAIT, |s| s.clock(p.id()).is_some())
        .await
        .expect("mapping");
}

/// A 1 kHz control loop on its own thread, as an embodiment runs one.
fn tick_at_1khz(edge: Arc<Edge>, stop: Arc<AtomicBool>) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        while !stop.load(Ordering::Relaxed) {
            let _ = edge.tick();
            std::thread::sleep(Duration::from_millis(1));
        }
    })
}

/// `n` actions at 1 ms from now (session time), tagged `tag`.
fn horizon(p: &Participant, n: usize, tag: u8, valid_for: Duration) -> ChunkDraft {
    let now = p.session_now_ns();
    ChunkDraft {
        based_on: vec![ObservationRef {
            channel: FT,
            seq: 1,
        }],
        t0_ns: now + 1_000_000,
        dt_ns: 1_000_000,
        replace_from_ns: now,
        valid_until_ns: now + i64::try_from(valid_for.as_nanos()).expect("ns"),
        actions: vec![vec![Bytes::from(vec![tag; 24])]; n],
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn the_edge_fences_a_preempted_controller_and_drops_expired_chunks() {
    let world = World::start().await;
    let robot = world.robot().await;
    let policy = world
        .join("policy", vec![Role::Controller], PriorityClass::Model)
        .await;
    let operator = world
        .join("operator", vec![Role::Controller], PriorityClass::Operator)
        .await;
    let edge = Edge::attach(&robot, CMD, EdgeConfig::default(), None)
        .await
        .expect("edge");
    mapped(&robot).await;

    let armed = operator.arm().await.expect("arm");
    policy
        .wait_for("armed", WAIT, |s| s.epoch == armed.epoch)
        .await
        .expect("view");
    let model = policy.grant_lease(CMD, None).await.expect("lease");
    robot
        .wait_for("robot sees the model lease", WAIT, |s| {
            s.lease(CMD).is_some_and(|l| l.token == model.token)
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
    let human = operator.grant_lease(CMD, None).await.expect("preempt");
    robot
        .wait_for("robot sees the preemption", WAIT, |s| {
            s.lease(CMD).is_some_and(|l| l.token == human.token)
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
    let policy = world
        .join("policy", vec![Role::Controller], PriorityClass::Model)
        .await;
    let operator = world
        .join("operator", vec![Role::Controller], PriorityClass::Operator)
        .await;
    let edge = Arc::new(
        Edge::attach(&robot, CMD, EdgeConfig::default(), None)
            .await
            .expect("edge"),
    );
    mapped(&robot).await;
    let armed = operator.arm().await.expect("arm");
    policy
        .wait_for("armed", WAIT, |s| s.epoch == armed.epoch)
        .await
        .expect("view");
    let model = policy.grant_lease(CMD, None).await.expect("lease");
    robot
        .wait_for("robot sees the model lease", WAIT, |s| {
            s.lease(CMD).is_some()
        })
        .await
        .expect("view");
    let model_tx = policy.chunk_publisher(&model).await.expect("publisher");
    tokio::time::sleep(Duration::from_millis(200)).await;

    let stop = Arc::new(AtomicBool::new(false));
    let ticker = tick_at_1khz(edge.clone(), stop.clone());
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
    let human = operator.grant_lease(CMD, None).await.expect("preempt");
    let human_tx = operator.chunk_publisher(&human).await.expect("publisher");
    robot
        .wait_for("robot sees the preemption", WAIT, |s| {
            s.lease(CMD).is_some_and(|l| l.token == human.token)
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
        .map(|q| (q.chunk.source.to_string(), q.leaves[0][0]))
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
    let policy = world
        .join("policy", vec![Role::Controller], PriorityClass::Model)
        .await;
    let edge = Arc::new(
        Edge::attach(&robot, CMD, EdgeConfig::default(), None)
            .await
            .expect("edge"),
    );
    mapped(&robot).await;
    policy.arm().await.expect("arm");
    let lease = policy.grant_lease(CMD, None).await.expect("lease");
    robot
        .wait_for("robot sees the lease", WAIT, |s| s.lease(CMD).is_some())
        .await
        .expect("view");
    let tx = policy.chunk_publisher(&lease).await.expect("publisher");
    tokio::time::sleep(Duration::from_millis(200)).await;
    let stop = Arc::new(AtomicBool::new(false));
    let ticker = tick_at_1khz(edge.clone(), stop.clone());
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
    assert!(snapshot.lease(CMD).is_none(), "SAFE revokes the lease");
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
    let policy = world
        .join("policy", vec![Role::Controller], PriorityClass::Model)
        .await;
    let edge = Arc::new(
        Edge::attach(
            &robot,
            CMD,
            EdgeConfig {
                max_action_age: Duration::from_secs(2),
                ..EdgeConfig::default()
            },
            None,
        )
        .await
        .expect("edge"),
    );
    mapped(&robot).await;
    policy.arm().await.expect("arm");
    let lease = policy.grant_lease(CMD, None).await.expect("lease");
    robot
        .wait_for("robot sees the lease", WAIT, |s| s.lease(CMD).is_some())
        .await
        .expect("view");
    let tx = policy.chunk_publisher(&lease).await.expect("publisher");
    tokio::time::sleep(Duration::from_millis(200)).await;
    // A long horizon: the queue would drive for a second and a half.
    tx.send(horizon(&policy, 1500, 1, Duration::from_secs(2)))
        .expect("send");
    wait_until("chunk accepted", || edge.stats().accepted == 1).await;
    let stop = Arc::new(AtomicBool::new(false));
    let ticker = tick_at_1khz(edge.clone(), stop.clone());
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

/// The controller half of the two-process test: runs only when the parent
/// test launches this binary with [`CHILD`] set. It attaches, takes the
/// lease, streams chunks, then dies without detaching.
#[tokio::test(flavor = "multi_thread")]
async fn child_controller_process() {
    let Ok(spec) = std::env::var(CHILD) else {
        return;
    };
    let mut parts = spec.split(',');
    let base: std::net::SocketAddr = parts.next().expect("base").parse().expect("base");
    let session: SessionId = parts.next().expect("session").parse().expect("session");
    let tenant = TenantId::new(parts.next().expect("tenant")).expect("tenant");
    let policy = join(
        base,
        &tenant,
        session,
        "remote-policy",
        vec![Role::Controller],
        PriorityClass::Model,
    )
    .await;
    let armed = policy
        .wait_for("armed", WAIT, |s| s.mode == Mode::Armed)
        .await
        .expect("armed");
    assert!(armed.channel(CMD).is_some());
    let lease = policy.grant_lease(CMD, None).await.expect("lease");
    let tx = policy.chunk_publisher(&lease).await.expect("publisher");
    tokio::time::sleep(Duration::from_millis(200)).await;
    let started = Instant::now();
    while started.elapsed() < Duration::from_millis(600) {
        tx.send(horizon(&policy, 50, 7, Duration::from_millis(150)))
            .expect("send");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    // A crash: no detach, no release.
    std::process::exit(0);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_controller_in_another_process_drives_the_edge_and_its_crash_falls_back() {
    let world = World::start().await;
    let robot = world.robot().await;
    let edge = Arc::new(
        Edge::attach(&robot, CMD, EdgeConfig::default(), None)
            .await
            .expect("edge"),
    );
    mapped(&robot).await;
    robot.arm().await.expect("arm");

    let mut child = Command::new(std::env::current_exe().expect("test binary"))
        .args(["--exact", "child_controller_process", "--nocapture"])
        .env(
            CHILD,
            format!("{},{},{}", world.base, world.session, world.tenant),
        )
        .stdout(Stdio::null())
        .spawn()
        .expect("spawn the controller process");
    let stop = Arc::new(AtomicBool::new(false));
    let ticker = tick_at_1khz(edge.clone(), stop.clone());

    let remote = ParticipantId::new("remote-policy").expect("id");
    wait_until("the remote controller's chunks drive the edge", || {
        edge.stats().mode(EdgeMode::Driving).ticks > 100
    })
    .await;
    // Its crash: the queue runs dry (or its lease lapses) and the session
    // goes SAFE.
    let snapshot = robot
        .wait_for("SAFE after the controller process died", WAIT, |s| {
            s.mode == Mode::Safe
        })
        .await
        .expect("safe");
    assert!(snapshot.lease(CMD).is_none());
    wait_until("the edge latches safe", || {
        edge.stats().mode(EdgeMode::Safe).entries == 1
    })
    .await;
    stop.store(true, Ordering::Relaxed);
    ticker.join().expect("ticker");
    assert!(child.wait().expect("child").success());

    let log = edge.drain_log();
    assert!(
        log.iter()
            .filter_map(|a| a.action.as_ref())
            .all(|q| q.chunk.source == remote && q.leaves[0][0] == 7),
        "every applied action is attributed to the remote controller"
    );
    let stats = edge.stats();
    assert_eq!(stats.mode(EdgeMode::Safe).entries, 1, "{stats:?}");
}
