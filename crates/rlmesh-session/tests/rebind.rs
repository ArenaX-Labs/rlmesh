//! A bound policy whose input's publisher goes away and comes back: a
//! policy that publishes restarts, a robot whose state is an input
//! restarts, a camera robot restarts. The controller binds again on its own
//! and keeps driving the robot it commands; a part that comes back as
//! something the policy cannot read stops the run. An optional input that
//! may be blanked binds blank at once while its robot is gone, and reads it
//! once it is back.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rlmesh_session::edge::{EdgeConfig, EdgeMode, Published, Robot};
use rlmesh_session::spec::apply::{Tensor, Value};
use rlmesh_session::spec::{Policy, RunOptions};
use rlmesh_session::transport::open_local_listener;
use rlmesh_session::wire::{
    DType, ParticipantId, PriorityClass, ReasonCode, Role, SessionId, TenantId,
};
use rlmesh_session::{
    Controller, ControllerConfig, HostConfig, Observation, Participant, ParticipantConfig,
    Predictor, RunConfig, RunError, RunEvent, SessionHost, SkipReason, SystemMonotonicClock,
    Transport, TransportConfig,
};

const WAIT: Duration = Duration::from_secs(20);

const ARM: &str = r#"{
    "arm": {"kind": "joints", "count": 6, "fields": ["pos"], "hz": 500,
            "command": "position", "safe": "stop", "hold_max_ms": 100, "max_action_age_ms": 1000}
}"#;

/// A tracker's target: two joints at 50 Hz, nothing commandable.
const TRACKER: &str = r#"{"target": {"kind": "joints", "count": 2, "fields": ["pos"], "hz": 50}}"#;

/// Reads the arm and `at`, drives the arm to the first value of `at`.
fn follow(at: &str, kind: &str) -> String {
    format!(
        r#"{{
    "hz": 50,
    "inputs": {{"q": {{"kind": "joints", "at": "arm", "fields": ["pos"]}},
               "target": {{"kind": "{kind}", "at": "{at}"}}}},
    "actions": {{"arm": {{"kind": "joints", "at": "arm", "command": "position",
                        "hz": 100, "chunk": 5}}}}
}}"#
    )
}

struct World {
    base: std::net::SocketAddr,
    tenant: TenantId,
    session: SessionId,
    host: SessionHost,
}

impl World {
    /// A host that re-arms robots on its own, as `rlmesh host
    /// --rearm-robots` does.
    async fn start() -> Self {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "warn".into()),
            )
            .with_test_writer()
            .try_init();
        let (transport, base) = open_local_listener("127.0.0.1".parse().expect("ip"))
            .await
            .expect("transport");
        let tenant = TenantId::new("rebind").expect("tenant");
        let session = SessionId::new();
        let mut config = HostConfig::new(tenant.clone(), session);
        config.clock = SystemMonotonicClock.shared();
        config.lease_grace = Duration::from_millis(300);
        config.presence_grace = Duration::from_millis(600);
        config.rearm_robots = true;
        config.health_log = None;
        let host = SessionHost::start(transport, config).await.expect("host");
        Self {
            base,
            tenant,
            session,
            host,
        }
    }

    async fn join(&self, id: &str, roles: Vec<Role>) -> Participant {
        let transport = Transport::open(&TransportConfig::per_class_tcp(self.base))
            .await
            .expect("transport");
        let priority = if roles.contains(&Role::Controller) {
            PriorityClass::Model
        } else {
            PriorityClass::Scripted
        };
        let mut config = ParticipantConfig::new(
            self.tenant.clone(),
            self.session,
            ParticipantId::new(id).expect("id"),
            roles,
            priority,
        );
        config.clock = SystemMonotonicClock.shared();
        config.heartbeat_period = Duration::from_millis(20);
        config.credential = Some(self.host.credential());
        Participant::attach(transport, config)
            .await
            .expect("attach")
    }

    async fn robot(&self, id: &str, map: &str) -> Arc<Robot> {
        let p = self.join(id, vec![Role::Embodiment]).await;
        Arc::new(
            Robot::declare(p, map, EdgeConfig::default())
                .await
                .expect("declare"),
        )
    }

    /// A policy that publishes `map` (its parts, none commandable), as
    /// `Policy(publishes=...)` does.
    async fn publisher(&self, id: &str, map: &str) -> Publisher {
        let p = self
            .join(id, vec![Role::Controller, Role::Embodiment])
            .await;
        let published = Published::declare(&p, map).await.expect("declare");
        Publisher {
            participant: Arc::new(p),
            published: Arc::new(published),
        }
    }

    async fn bind(&self, id: &str, policy: &str, timeout: Duration) -> Controller {
        let p = self.join(id, vec![Role::Controller]).await;
        let options = RunOptions::named(id).robot("ur10e").expect("robot");
        Controller::bind(
            Arc::new(p),
            &Policy::parse(policy).expect("policy"),
            &options,
            ControllerConfig {
                bind_timeout: timeout,
                ..ControllerConfig::default()
            },
        )
        .await
        .expect("bind")
    }

    fn present(&self, id: &str) -> bool {
        let id = ParticipantId::new(id).expect("id");
        self.host
            .snapshot()
            .participant(&id)
            .is_some_and(|p| p.present)
    }
}

struct Publisher {
    participant: Arc<Participant>,
    published: Arc<Published>,
}

fn f32s(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// What the arm dispatched: (time, mode, first joint).
type Log = Arc<Mutex<Vec<(Instant, EdgeMode, Option<f32>)>>>;

/// A loop that stops when its flag is set.
struct Loop {
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Loop {
    fn spawn(mut body: impl FnMut() + Send + 'static, period: Duration) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let thread = std::thread::spawn({
            let stop = stop.clone();
            move || {
                while !stop.load(Ordering::Relaxed) {
                    body();
                    std::thread::sleep(period);
                }
            }
        });
        Self {
            stop,
            thread: Some(thread),
        }
    }

    fn end(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            thread.join().expect("loop");
        }
    }
}

/// The arm's 500 Hz loop: publish, then apply what is due.
fn spin_arm(robot: Arc<Robot>, log: Log) -> Loop {
    let clock = robot.participant().clock().clone();
    let mut q = [0.0_f32; 6];
    Loop::spawn(
        move || {
            let _ = robot.publish("arm", clock.now_ns(), &[&f32s(&q)]);
            let due = robot.due("arm").ok().flatten();
            let value = due
                .as_ref()
                .and_then(|d| d.action.as_ref())
                .map(|a| f32::from_le_bytes(a.value[..4].try_into().expect("4 bytes")));
            if let Some(v) = value {
                q[0] = v;
            }
            let mode = due.map_or(EdgeMode::Idle, |d| d.mode);
            log.lock().expect("log").push((Instant::now(), mode, value));
        },
        Duration::from_millis(2),
    )
}

/// Publishes `fields` at about 50 Hz through `publish`; the newest seq
/// goes to `last`.
fn spin_publish(
    publish: impl Fn(&[&[u8]]) -> Option<u64> + Send + 'static,
    fields: Vec<Vec<u8>>,
    last: Arc<AtomicU64>,
) -> Loop {
    Loop::spawn(
        move || {
            let refs: Vec<&[u8]> = fields.iter().map(Vec::as_slice).collect();
            if let Some(seq) = publish(&refs) {
                last.store(seq, Ordering::Relaxed);
            }
        },
        Duration::from_millis(20),
    )
}

fn publish_robot(
    robot: Arc<Robot>,
    path: &'static str,
) -> impl Fn(&[&[u8]]) -> Option<u64> + Send + 'static {
    let clock = robot.participant().clock().clone();
    move |fields| robot.publish(path, clock.now_ns(), fields).ok().flatten()
}

fn publish_policy(
    publisher: &Publisher,
    path: &'static str,
) -> impl Fn(&[&[u8]]) -> Option<u64> + Send + 'static {
    let clock = publisher.participant.clock().clone();
    let published = publisher.published.clone();
    move |fields| {
        published
            .publish(path, clock.now_ns(), fields)
            .ok()
            .flatten()
    }
}

/// Kills a process: no detach, its heartbeats and its transport just stop.
async fn kill(participant: &Participant) {
    participant.set_heartbeats_enabled(false);
    participant.transport().close().await.expect("close");
}

/// Drives the arm to the first value of input `target` (an image's first
/// byte counts as a quarter below 50, three quarters above).
struct Follow;

impl Predictor for Follow {
    type Error = String;

    fn predict(
        &mut self,
        observation: Observation,
        _: i64,
    ) -> Result<Vec<(String, Value)>, String> {
        let target = match observation.value("target") {
            Some(Value::Tensor(t)) if t.dtype == DType::U8 => {
                if t.to_f64()[0] < 50.0 {
                    0.25
                } else {
                    0.75
                }
            }
            Some(Value::Tensor(t)) => t.to_f64()[0],
            other => return Err(format!("target: {other:?}")),
        };
        Ok(vec![(
            "arm".into(),
            Value::Tensor(Tensor::from_f64(DType::F32, vec![5, 6], &[target; 30])),
        )])
    }
}

/// The last five ticks since `since` dispatched `target`.
fn driven_since(log: &Log, target: f32, since: Instant) -> bool {
    let log = log.lock().expect("log");
    let recent: Vec<_> = log.iter().filter(|(at, ..)| *at >= since).collect();
    recent.len() >= 5
        && recent
            .iter()
            .rev()
            .take(5)
            .all(|(_, mode, value)| *mode == EdgeMode::Driving && *value == Some(target))
}

async fn wait_until(what: &str, mut done: impl FnMut() -> bool) {
    let started = Instant::now();
    while !done() {
        assert!(started.elapsed() < WAIT, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// The run's events so far, kept across drains.
struct Events {
    controller: Arc<Controller>,
    seen: Vec<RunEvent>,
}

impl Events {
    fn new(controller: &Arc<Controller>) -> Self {
        Self {
            controller: controller.clone(),
            seen: Vec::new(),
        }
    }

    fn any(&mut self, f: impl Fn(&RunEvent) -> bool) -> bool {
        self.seen.extend(self.controller.drain_events());
        self.seen.iter().any(f)
    }

    fn lost(&mut self, part: &str) -> bool {
        self.any(|e| matches!(e, RunEvent::InputLost { part: p, .. } if p == part))
    }

    fn rebound(&mut self, part: &str) -> bool {
        self.any(
            |e| matches!(e, RunEvent::Rebound { parts, .. } if parts.iter().any(|p| p == part)),
        )
    }
}

fn run(controller: &Arc<Controller>) -> std::thread::JoinHandle<Result<(), RunError<String>>> {
    let controller = controller.clone();
    std::thread::spawn(move || {
        controller.run(
            &mut Follow,
            RunConfig::default().duration(Duration::from_secs(120)),
        )
    })
}

#[tokio::test(flavor = "multi_thread")]
async fn a_policy_reading_a_publishing_policy_reads_it_again_after_it_restarts() {
    let world = World::start().await;
    let arm = world.robot("ur10e", ARM).await;
    arm.participant().arm().await.expect("arm");
    let log: Log = Arc::default();
    let arm_loop = spin_arm(arm.clone(), log.clone());
    let tracker = world.publisher("tracker", TRACKER).await;
    let first_seq = Arc::new(AtomicU64::new(0));
    let tracker_loop = spin_publish(
        publish_policy(&tracker, "target"),
        vec![f32s(&[0.25, 0.0])],
        first_seq.clone(),
    );
    let driver = Arc::new(
        world
            .bind("driver", &follow("tracker:target", "joints"), WAIT)
            .await,
    );
    let mut events = Events::new(&driver);
    let running = run(&driver);
    let started = Instant::now();
    wait_until("the arm driven to the tracker's target", || {
        driven_since(&log, 0.25, started)
    })
    .await;
    // Enough samples that the restarted tracker's seqs start well below.
    wait_until("the tracker past seq 40", || {
        first_seq.load(Ordering::Relaxed) >= 40
    })
    .await;

    tracker_loop.end();
    kill(&tracker.participant).await;
    drop(tracker);
    let high_water = first_seq.load(Ordering::Relaxed);
    wait_until("the host loses the tracker", || !world.present("tracker")).await;
    wait_until("the run reports the input lost", || {
        events.lost("tracker:target")
    })
    .await;
    wait_until("starts skipped for the gone publisher", || {
        driver.stats().skipped(SkipReason::PublisherGone) > 0
    })
    .await;
    assert!(driver.holds(), "the arm's lease is kept meanwhile");

    let tracker = world.publisher("tracker", TRACKER).await;
    let second_seq = Arc::new(AtomicU64::new(0));
    let tracker_loop = spin_publish(
        publish_policy(&tracker, "target"),
        vec![f32s(&[0.75, 0.0])],
        second_seq.clone(),
    );
    let back = Instant::now();
    wait_until("the arm driven to the restarted tracker's target", || {
        driven_since(&log, 0.75, back)
    })
    .await;
    let read_at = second_seq.load(Ordering::Relaxed);
    assert!(
        read_at <= high_water,
        "read only once its seqs passed the old ones ({read_at} > {high_water})"
    );
    wait_until("the run reports the input rebound", || {
        events.rebound("tracker:target")
    })
    .await;

    driver.stop();
    running.join().expect("run").expect("a clean stop");
    tracker_loop.end();
    arm_loop.end();
}

/// The base: wheels it drives itself, and its odometry, which the driver
/// reads.
const BASE: &str = r#"{
    "wheels": {"kind": "joints", "count": 2, "fields": ["pos"], "hz": 100,
               "command": "position", "safe": "stop", "hold_max_ms": 100},
    "odom": {"kind": "joints", "count": 2, "fields": ["pos"], "hz": 50}
}"#;

#[tokio::test(flavor = "multi_thread")]
async fn a_robot_whose_state_is_read_leaves_and_comes_back_and_the_arm_keeps_its_lease() {
    let world = World::start().await;
    let arm = world.robot("ur10e", ARM).await;
    arm.participant().arm().await.expect("arm");
    let log: Log = Arc::default();
    let arm_loop = spin_arm(arm.clone(), log.clone());
    let base = world.robot("base", BASE).await;
    let seq = Arc::new(AtomicU64::new(0));
    let base_loop = spin_publish(
        publish_robot(base.clone(), "odom"),
        vec![f32s(&[0.25, 0.0])],
        seq.clone(),
    );
    let driver = Arc::new(
        world
            .bind("driver", &follow("base:odom", "joints"), WAIT)
            .await,
    );
    let mut events = Events::new(&driver);
    let running = run(&driver);
    let started = Instant::now();
    wait_until("the arm driven to the base's odometry", || {
        driven_since(&log, 0.25, started)
    })
    .await;
    let arm_channel = driver
        .binding()
        .action("arm")
        .and_then(|a| a.part.clone())
        .expect("arm")
        .channel;
    let before = driver.binding().fingerprint;

    // The base shuts down cleanly: its parts go with it.
    base_loop.end();
    rlmesh_session::sdk::close_robot(base)
        .await
        .expect("detach");
    wait_until("the run reports the input lost", || {
        events.lost("base:odom")
    })
    .await;

    let base = world.robot("base", BASE).await;
    let base_loop = spin_publish(
        publish_robot(base.clone(), "odom"),
        vec![f32s(&[0.75, 0.0])],
        seq.clone(),
    );
    let back = Instant::now();
    wait_until("the arm driven to the new odometry", || {
        driven_since(&log, 0.75, back)
    })
    .await;
    wait_until("the run reports the input rebound", || {
        events.rebound("base:odom")
    })
    .await;
    assert_ne!(
        driver.binding().fingerprint,
        before,
        "bound to the new part"
    );
    // The arm's lease was re-issued with the binding, never given up.
    let snapshot = world.host.snapshot();
    let ends: Vec<ReasonCode> = snapshot
        .ended
        .iter()
        .filter(|e| e.lease.channel == arm_channel)
        .map(|e| e.reason.code)
        .collect();
    assert!(
        ends.iter().all(|code| *code == ReasonCode::Rebound),
        "the arm's lease ended otherwise: {ends:?}"
    );
    assert!(ends.contains(&ReasonCode::Rebound), "{ends:?}");
    assert!(driver.holds());

    driver.stop();
    running.join().expect("run").expect("a clean stop");
    base_loop.end();
    arm_loop.end();
}

const EYE: &str = r#"{"rgb": {"kind": "camera", "width": 32, "height": 24, "hz": 30}}"#;

#[tokio::test(flavor = "multi_thread")]
async fn a_policy_reading_a_camera_sees_it_again_after_the_camera_restarts() {
    let world = World::start().await;
    let arm = world.robot("ur10e", ARM).await;
    arm.participant().arm().await.expect("arm");
    let log: Log = Arc::default();
    let arm_loop = spin_arm(arm.clone(), log.clone());
    let eye = world.robot("eye", EYE).await;
    let eye_loop = spin_publish(
        publish_robot(eye.clone(), "rgb"),
        vec![vec![20_u8; 32 * 24 * 3]],
        Arc::default(),
    );
    let driver = Arc::new(
        world
            .bind("driver", &follow("eye:rgb", "camera"), WAIT)
            .await,
    );
    // An image's seq comes from its encoder: read it where it lands.
    let seq = |driver: &Controller| {
        driver
            .observe()
            .ok()
            .and_then(|(o, _)| Some(o.meta.input("target")?.samples.last()?.seq))
            .unwrap_or(0)
    };
    let mut events = Events::new(&driver);
    let running = run(&driver);
    let started = Instant::now();
    wait_until("the arm driven by the camera", || {
        driven_since(&log, 0.25, started)
    })
    .await;
    let mut high_water = 0;
    wait_until("the camera past seq 30", || {
        high_water = seq(&driver);
        high_water >= 30
    })
    .await;

    eye_loop.end();
    let eye = Arc::try_unwrap(eye).ok().expect("the only handle");
    kill(eye.participant()).await;
    drop(eye);
    wait_until("the host loses the camera", || !world.present("eye")).await;
    wait_until("the run reports the input lost", || events.lost("eye:rgb")).await;

    let eye = world.robot("eye", EYE).await;
    let eye_loop = spin_publish(
        publish_robot(eye.clone(), "rgb"),
        vec![vec![200_u8; 32 * 24 * 3]],
        Arc::default(),
    );
    let back = Instant::now();
    wait_until("the arm driven by the restarted camera", || {
        driven_since(&log, 0.75, back)
    })
    .await;
    let read_at = seq(&driver);
    assert!(
        read_at <= high_water,
        "read only once its seqs passed the old ones ({read_at} > {high_water})"
    );
    wait_until("the run reports the input rebound", || {
        events.rebound("eye:rgb")
    })
    .await;

    driver.stop();
    running.join().expect("run").expect("a clean stop");
    eye_loop.end();
    arm_loop.end();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_part_that_comes_back_as_something_else_stops_the_run_and_says_what_changed() {
    let world = World::start().await;
    let arm = world.robot("ur10e", ARM).await;
    arm.participant().arm().await.expect("arm");
    let log: Log = Arc::default();
    let arm_loop = spin_arm(arm.clone(), log.clone());
    let tracker = world.publisher("tracker", TRACKER).await;
    let tracker_loop = spin_publish(
        publish_policy(&tracker, "target"),
        vec![f32s(&[0.25, 0.0])],
        Arc::default(),
    );
    let driver = Arc::new(
        world
            .bind("driver", &follow("tracker:target", "joints"), WAIT)
            .await,
    );
    let running = run(&driver);
    let started = Instant::now();
    wait_until("the arm driven", || driven_since(&log, 0.25, started)).await;

    tracker_loop.end();
    kill(&tracker.participant).await;
    drop(tracker);
    wait_until("the host loses the tracker", || !world.present("tracker")).await;
    let _tracker = world
        .publisher(
            "tracker",
            r#"{"target": {"kind": "x/target", "hz": 50, "fields": {"xy": ["float32", [2]]}}}"#,
        )
        .await;
    let result = tokio::task::spawn_blocking(move || running.join().expect("run"))
        .await
        .expect("join");
    let Err(RunError::Rebind(why)) = result else {
        panic!("the run went on: {result:?}");
    };
    assert!(
        why.contains("tracker:target came back changed (kind joints, now x/target"),
        "{why}"
    );
    assert!(!driver.holds(), "the run let go of the arm");
    arm_loop.end();
}

/// Reads the arm and, when it is there, a camera.
const OPTIONAL_CAMERA: &str = r#"{
    "hz": 50,
    "inputs": {"q": {"kind": "joints", "at": "arm", "fields": ["pos"]},
               "target": {"kind": "camera", "at": "eye:rgb", "size": [24, 32],
                          "optional": true, "allow": ["blank"]}},
    "actions": {"arm": {"kind": "joints", "at": "arm", "command": "position",
                        "hz": 100, "chunk": 5}}
}"#;

#[tokio::test(flavor = "multi_thread")]
async fn an_optional_input_whose_robot_is_gone_binds_blank_at_once() {
    let world = World::start().await;
    let _arm = world.robot("ur10e", ARM).await;
    // Lost: its parts stay in the session for its return.
    let eye = world.robot("eye", EYE).await;
    kill(eye.participant()).await;
    drop(eye);
    wait_until("the host loses the camera", || !world.present("eye")).await;
    let bound = tokio::time::timeout(
        Duration::from_secs(60),
        world.bind("lost", OPTIONAL_CAMERA, Duration::from_secs(3600)),
    )
    .await
    .expect("no wait for the gone camera");
    let binding = bound.binding();
    let row = binding.input("target").expect("target");
    assert!(
        row.part.is_none(),
        "bound to the gone camera: {}",
        binding.table()
    );

    // Left cleanly: its parts went with it.
    let eye = world.robot("eye", EYE).await;
    rlmesh_session::sdk::close_robot(eye).await.expect("detach");
    let bound = tokio::time::timeout(
        Duration::from_secs(60),
        world.bind("left", OPTIONAL_CAMERA, Duration::from_secs(3600)),
    )
    .await
    .expect("no wait for the camera that left");
    let binding = bound.binding();
    let row = binding.input("target").expect("target");
    assert!(row.part.is_none(), "{}", binding.table());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_blank_input_reads_its_robot_once_it_joins() {
    let world = World::start().await;
    let arm = world.robot("ur10e", ARM).await;
    arm.participant().arm().await.expect("arm");
    let log: Log = Arc::default();
    let arm_loop = spin_arm(arm.clone(), log.clone());
    let driver = Arc::new(world.bind("driver", OPTIONAL_CAMERA, WAIT).await);
    assert!(
        driver
            .binding()
            .input("target")
            .expect("target")
            .part
            .is_none()
    );
    let mut events = Events::new(&driver);
    let running = run(&driver);
    // Blank: zeros, so the arm goes to a quarter.
    let started = Instant::now();
    wait_until("the arm driven with the camera blank", || {
        driven_since(&log, 0.25, started)
    })
    .await;

    let eye = world.robot("eye", EYE).await;
    let eye_loop = spin_publish(
        publish_robot(eye.clone(), "rgb"),
        vec![vec![200_u8; 32 * 24 * 3]],
        Arc::default(),
    );
    let joined = Instant::now();
    wait_until("the arm driven by the camera", || {
        driven_since(&log, 0.75, joined)
    })
    .await;
    wait_until("the run reports the input bound", || {
        events.rebound("eye:rgb")
    })
    .await;
    assert!(
        driver
            .binding()
            .input("target")
            .expect("target")
            .part
            .is_some()
    );

    driver.stop();
    running.join().expect("run").expect("a clean stop");
    eye_loop.end();
    arm_loop.end();
}

/// Reads the arm and holds it at 0.5.
const HOLD: &str = r#"{
    "hz": 50,
    "inputs": {"q": {"kind": "joints", "at": "arm", "fields": ["pos"]}},
    "actions": {"arm": {"kind": "joints", "at": "arm", "command": "position",
                        "hz": 100, "chunk": 5}}
}"#;

/// Holds the arm at 0.5.
struct Hold;

impl Predictor for Hold {
    type Error = String;

    fn predict(&mut self, _: Observation, _: i64) -> Result<Vec<(String, Value)>, String> {
        Ok(vec![(
            "arm".into(),
            Value::Tensor(Tensor::from_f64(DType::F32, vec![5, 6], &[0.5; 30])),
        )])
    }
}

/// A policy whose only robot closes cleanly: the run keeps going, its starts
/// skipped as `publisher_gone` with the robot named, for as long as the
/// robot stays away; when a robot of that name comes back, the policy binds
/// again and drives it. Bound to the robot by name (`named`), or to the
/// session's view.
async fn only_robot_closes_and_returns(named: bool) {
    let world = World::start().await;
    let arm = world.robot("ur10e", ARM).await;
    arm.participant().arm().await.expect("arm");
    let log: Log = Arc::default();
    let arm_loop = spin_arm(arm.clone(), log.clone());
    let p = world.join("holder", vec![Role::Controller]).await;
    let options = if named {
        RunOptions::named("holder").robot("ur10e").expect("robot")
    } else {
        RunOptions::named("holder")
    };
    let holder = Arc::new(
        Controller::bind(
            Arc::new(p),
            &Policy::parse(HOLD).expect("policy"),
            &options,
            ControllerConfig::default(),
        )
        .await
        .expect("bind"),
    );
    let mut events = Events::new(&holder);
    let running = {
        let holder = holder.clone();
        std::thread::spawn(move || {
            holder.run(
                &mut Hold,
                RunConfig::default().duration(Duration::from_secs(120)),
            )
        })
    };
    let started = Instant::now();
    wait_until("the arm held at 0.5", || driven_since(&log, 0.5, started)).await;

    // Its only robot shuts down cleanly: its parts go with it.
    arm_loop.end();
    rlmesh_session::sdk::close_robot(arm).await.expect("detach");
    wait_until("the run reports the input lost", || {
        events.lost("ur10e:arm")
    })
    .await;
    let lost = events
        .seen
        .iter()
        .find_map(|e| match e {
            RunEvent::InputLost { message, .. } => Some(message.clone()),
            _ => None,
        })
        .expect("lost");
    assert_eq!(
        lost,
        "input \"q\" lost its publisher: ur10e left the session; starts that need it are \
         skipped (publisher_gone) until it is back"
    );
    // It waits as long as the robot stays away, and never exits.
    let gone = holder.stats().skipped(SkipReason::PublisherGone);
    wait_until("starts skipped while the robot is away", || {
        holder.stats().skipped(SkipReason::PublisherGone) >= gone + 25
    })
    .await;
    assert!(!running.is_finished(), "the run stopped");

    // A robot of that name comes back.
    let arm = world.robot("ur10e", ARM).await;
    let arm_loop = spin_arm(arm.clone(), log.clone());
    let back = Instant::now();
    wait_until("the run binds again", || events.rebound("ur10e:arm")).await;
    wait_until("the arm held again", || driven_since(&log, 0.5, back)).await;
    assert!(holder.holds());

    holder.stop();
    running.join().expect("run").expect("a clean stop");
    arm_loop.end();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_policy_bound_to_its_only_robot_by_name_binds_again_when_it_closes_and_returns() {
    only_robot_closes_and_returns(true).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_policy_bound_to_the_view_of_its_only_robot_binds_again_when_it_closes_and_returns() {
    only_robot_closes_and_returns(false).await;
}
