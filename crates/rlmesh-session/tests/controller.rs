//! A declared robot ([`Robot`]) and bound policies ([`Controller`]) over a
//! real session: the run loop drives the edge, an operator preempts the
//! model and hands back, a failing model lets go of the robot.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rlmesh_session::edge::{EdgeConfig, EdgeMode, Robot};
use rlmesh_session::spec::apply::{Tensor, Value};
use rlmesh_session::spec::{Policy, RunOptions};
use rlmesh_session::transport::open_local_listeners;
use rlmesh_session::wire::{
    DType, LeaseToken, Mode, PriorityClass, ReasonCode, Role, SessionId, TenantId,
};
use rlmesh_session::{
    Controller, ControllerConfig, HostConfig, Observation, Participant, ParticipantConfig,
    Predictor, RunError, SessionHost, SkipReason, SystemMonotonicClock, Transport, TransportConfig,
};

const WAIT: Duration = Duration::from_secs(10);

/// The UR10e's arm and wrist F/T, at 500 Hz.
const UR10E: &str = r#"{
    "arm": {"kind": "joints", "count": 6, "fields": ["pos"], "hz": 500,
            "command": "position", "safe": "stop", "hold_max_ms": 100, "max_action_age_ms": 1000},
    "wrist/ft": {"kind": "wrench", "frame": "tool0", "hz": 500}
}"#;

/// Reads the joints and the wrench, commands 5 actions at 100 Hz, replans
/// at 50 Hz.
const POLICY: &str = r#"{
    "hz": 50,
    "inputs": {"q": {"kind": "joints", "at": "arm", "fields": ["pos"]},
               "ft": {"kind": "wrench", "at": "wrist/ft", "frame": "tool0"}},
    "actions": {"arm": {"kind": "joints", "at": "arm", "command": "position",
                        "hz": 100, "chunk": 5}}
}"#;

struct World {
    base: std::net::SocketAddr,
    tenant: TenantId,
    session: SessionId,
    host: SessionHost,
}

impl World {
    async fn start() -> Self {
        Self::start_with(|_| {}).await
    }

    /// A host with `configure`d rules.
    async fn start_with(configure: impl FnOnce(&mut HostConfig)) -> Self {
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
        let tenant = TenantId::new("controller").expect("tenant");
        let session = SessionId::new();
        let mut config = HostConfig::new(tenant.clone(), session);
        config.clock = SystemMonotonicClock.shared();
        config.lease_grace = Duration::from_millis(300);
        config.health_log = None;
        configure(&mut config);
        let host = SessionHost::start(transport, config).await.expect("host");
        Self {
            base,
            tenant,
            session,
            host,
        }
    }

    async fn join(&self, id: &str, role: Role, priority: PriorityClass) -> Participant {
        let transport = Transport::open(
            &TransportConfig::per_class_tcp(
                Some("127.0.0.1:0".parse().expect("addr")),
                Some(self.base),
            )
            .expect("config"),
        )
        .await
        .expect("transport");
        let mut config = ParticipantConfig::new(
            self.tenant.clone(),
            self.session,
            rlmesh_session::wire::ParticipantId::new(id).expect("id"),
            vec![role],
            priority,
        );
        config.clock = SystemMonotonicClock.shared();
        config.heartbeat_period = Duration::from_millis(20);
        config.credential = Some(self.host.credential());
        Participant::attach(transport, config)
            .await
            .expect("attach")
    }

    async fn robot(&self) -> Arc<Robot> {
        let p = self
            .join("ur10e", Role::Embodiment, PriorityClass::Scripted)
            .await;
        let robot = Robot::declare(p, UR10E, EdgeConfig::default())
            .await
            .expect("declare");
        robot.participant().arm().await.expect("arm");
        Arc::new(robot)
    }

    async fn controller(&self, id: &str, priority: PriorityClass) -> Arc<Controller> {
        let p = self.join(id, Role::Controller, priority).await;
        let options = RunOptions::named(id).robot("ur10e").expect("robot");
        let controller = Controller::bind(
            Arc::new(p),
            &Policy::parse(POLICY).expect("policy"),
            &options,
            ControllerConfig::default(),
        )
        .await
        .expect("bind");
        Arc::new(controller)
    }
}

fn f32s(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// What the robot dispatched: (time, mode, first joint, token).
type Log = Arc<Mutex<Vec<(Instant, EdgeMode, Option<f32>, Option<LeaseToken>)>>>;

/// The robot's 500 Hz loop: publish, then apply what is due.
fn spin(robot: Arc<Robot>, stop: Arc<AtomicBool>, log: Log) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let clock = robot.participant().clock().clone();
        let mut q = [0.5_f32; 6];
        while !stop.load(Ordering::Relaxed) {
            let now = clock.now_ns();
            robot
                .publish("arm", now, &[&f32s(&q)])
                .expect("publish arm");
            robot
                .publish("wrist/ft", now, &[&f32s(&[0.0; 3]), &f32s(&[0.0; 3])])
                .expect("publish ft");
            let due = robot.due("arm").expect("due");
            let entry = match &due {
                Some(due) => {
                    let value = due
                        .action
                        .as_ref()
                        .map(|a| f32::from_le_bytes(a.value[..4].try_into().expect("4 bytes")));
                    if let Some(v) = value {
                        q[0] = v;
                    }
                    (
                        due.mode,
                        value,
                        due.action.as_ref().map(|a| a.chunk.lease_token),
                    )
                }
                None => (EdgeMode::Idle, None, None),
            };
            log.lock()
                .expect("log")
                .push((Instant::now(), entry.0, entry.1, entry.2));
            std::thread::sleep(Duration::from_millis(2));
        }
    })
}

/// Commands every joint to `target` for the whole chunk.
struct Constant {
    target: f32,
    seen: usize,
}

impl Predictor for Constant {
    type Error = String;

    fn predict(
        &mut self,
        observation: Observation,
        _: i64,
    ) -> Result<Vec<(String, Value)>, String> {
        let Some(Value::Tensor(q)) = observation.value("q") else {
            return Err("no q".into());
        };
        assert_eq!(q.shape, [6]);
        self.seen += 1;
        Ok(vec![(
            "arm".into(),
            Value::Tensor(Tensor::from_f64(
                DType::F32,
                vec![5, 6],
                &[f64::from(self.target); 30],
            )),
        )])
    }
}

fn run_in_thread(
    controller: &Arc<Controller>,
    target: f32,
) -> std::thread::JoinHandle<(Result<(), RunError<String>>, usize)> {
    let controller = controller.clone();
    std::thread::spawn(move || {
        let mut model = Constant { target, seen: 0 };
        let result = controller.run(&mut model, Some(Duration::from_secs(30)));
        (result, model.seen)
    })
}

async fn wait_until(what: &str, mut done: impl FnMut() -> bool) {
    let started = Instant::now();
    while !done() {
        assert!(started.elapsed() < WAIT, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// The last five ticks dispatched `target`.
fn driven_to(log: &Log, target: f32) -> bool {
    let log = log.lock().expect("log");
    log.len() >= 5
        && log
            .iter()
            .rev()
            .take(5)
            .all(|(_, mode, value, _)| *mode == EdgeMode::Driving && *value == Some(target))
}

#[tokio::test(flavor = "multi_thread")]
async fn a_bound_policy_runs_warm_before_authority_and_the_robot_applies_its_chunks() {
    let world = World::start().await;
    let robot = world.robot().await;
    let model = world.controller("model", PriorityClass::Model).await;
    assert!(model.binding().table().contains("binding model on ur10e"));
    assert!(!model.holds(), "no lease before the inputs are warm");

    let (stop, log): (_, Log) = (Arc::new(AtomicBool::new(false)), Arc::default());
    let robot_loop = spin(robot.clone(), stop.clone(), log.clone());
    let run = run_in_thread(&model, 1.5);
    wait_until("the robot driven by the policy", || driven_to(&log, 1.5)).await;
    assert!(model.holds());
    let stats = model.stats();
    assert!(stats.completions > 0 && stats.chunks_sent > 0, "{stats:?}");
    assert_eq!(stats.send_errors, 0);

    model.stop();
    let (result, seen) = run.join().expect("run thread");
    result.expect("a clean stop");
    assert!(seen > 0);
    // A stopped run gives its leases back: released, not lapsed.
    wait_until("the lease released", || !model.holds()).await;
    stop.store(true, Ordering::Relaxed);
    robot_loop.join().expect("robot loop");
    let safe = robot.edge_stats("arm").expect("stats").mode(EdgeMode::Safe);
    assert_eq!(safe.entries, 0, "a release never latches safe");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_operator_preempts_the_model_and_hands_back_under_a_fresh_token() {
    let world = World::start().await;
    let robot = world.robot().await;
    let model = world.controller("model", PriorityClass::Model).await;
    let operator = world.controller("operator", PriorityClass::Operator).await;
    let (stop, log): (_, Log) = (Arc::new(AtomicBool::new(false)), Arc::default());
    let robot_loop = spin(robot.clone(), stop.clone(), log.clone());
    let run = run_in_thread(&model, 1.0);
    wait_until("the model driving", || driven_to(&log, 1.0)).await;
    let model_token = model.leases()[0].token;

    // The operator takes over and streams its own chunks.
    let leases = operator.acquire().await.expect("operator class preempts");
    assert!(leases[0].token > model_token);
    wait_until("the model's lease gone", || !model.holds()).await;
    let operator_driving = Arc::new(AtomicBool::new(true));
    let streaming = {
        let (operator, driving) = (operator.clone(), operator_driving.clone());
        std::thread::spawn(move || {
            while driving.load(Ordering::Relaxed) {
                if let Ok((observation, at)) = operator.observe() {
                    let value = Value::Tensor(Tensor::from_f64(DType::F32, vec![5, 6], &[2.0; 30]));
                    operator
                        .send(&[("arm".into(), value)], &observation.meta.based_on, at)
                        .expect("send");
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        })
    };
    wait_until("the operator driving", || driven_to(&log, 2.0)).await;
    assert!(
        model.stats().skipped(SkipReason::NoLease) > 0,
        "a standby skips its starts"
    );

    // The operator hands back: the model re-requests, gets a fresh token
    // and resumes.
    operator_driving.store(false, Ordering::Relaxed);
    streaming.join().expect("operator thread");
    operator.release().await.expect("release");
    wait_until("the model driving again", || driven_to(&log, 1.0)).await;
    let tokens: Vec<LeaseToken> = log
        .lock()
        .expect("log")
        .iter()
        .filter_map(|(_, _, value, token)| (*value == Some(1.0)).then_some(*token).flatten())
        .collect();
    let last = *tokens.last().expect("model tokens");
    assert!(last > leases[0].token, "a fresh token after the hand-back");
    assert!(tokens.contains(&model_token));

    model.stop();
    run.join().expect("run thread").0.expect("clean stop");
    stop.store(true, Ordering::Relaxed);
    robot_loop.join().expect("robot loop");
}

/// Fails on its third start.
struct Failing(usize);

impl Predictor for Failing {
    type Error = String;

    fn predict(&mut self, _: Observation, _: i64) -> Result<Vec<(String, Value)>, String> {
        self.0 += 1;
        if self.0 == 3 {
            return Err("the model crashed".into());
        }
        Ok(vec![(
            "arm".into(),
            Value::Tensor(Tensor::from_f64(DType::F32, vec![5, 6], &[0.25; 30])),
        )])
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_failing_model_stops_sending_and_lets_go_of_the_robot() {
    let world = World::start().await;
    let robot = world.robot().await;
    let model = world.controller("model", PriorityClass::Model).await;
    let (stop, log): (_, Log) = (Arc::new(AtomicBool::new(false)), Arc::default());
    let robot_loop = spin(robot.clone(), stop.clone(), log.clone());
    let result = {
        let model = model.clone();
        tokio::task::spawn_blocking(move || model.run(&mut Failing(0), Some(WAIT)))
            .await
            .expect("run")
    };
    assert!(matches!(result, Err(RunError::Predict(ref e)) if e == "the model crashed"));
    assert!(!model.holds());
    assert!(model.leases().is_empty());
    // A wrong shape is refused before anything is sent.
    let (observation, at) = model.observe().expect("warm inputs");
    let wrong = Value::Tensor(Tensor::from_f64(DType::F32, vec![6], &[0.0; 6]));
    let err = model
        .send(&[("arm".into(), wrong)], &observation.meta.based_on, at)
        .expect_err("[6] is not [5, 6]");
    assert!(err.to_string().contains("expected [5, 6]"), "{err}");
    stop.store(true, Ordering::Relaxed);
    robot_loop.join().expect("robot loop");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_local_estop_stops_the_robot_at_once_and_reaches_the_host_before_its_clear() {
    let world = World::start().await;
    let robot = world.robot().await;
    let model = world.controller("model", PriorityClass::Model).await;
    let (stop, log): (_, Log) = (Arc::new(AtomicBool::new(false)), Arc::default());
    let robot_loop = spin(robot.clone(), stop.clone(), log.clone());
    let run = run_in_thread(&model, 1.5);
    wait_until("the robot driven by the policy", || driven_to(&log, 1.5)).await;

    // Latched and cleared back to back: the edge stops on its next tick,
    // and the host still sees the Estop before the Clear.
    robot.estop("button");
    let latched_at = Instant::now();
    robot.clear_estop();
    wait_until("the edge in ESTOP", || {
        log.lock()
            .expect("log")
            .iter()
            .any(|(at, mode, ..)| *at >= latched_at && *mode == EdgeMode::Estop)
    })
    .await;
    let p = robot.participant();
    let view = p
        .wait_for("ESTOP, cleared", WAIT, |s| {
            s.mode == Mode::Estop && s.mode_reason.code == ReasonCode::EstopCleared
        })
        .await
        .expect("cleared");
    assert!(!view.estop_latched);
    assert!(
        view.ended
            .iter()
            .any(|e| { e.reason.code == ReasonCode::VoidedByEstop && e.reason.by == "ur10e" })
    );
    // Leaving ESTOP still needs disarm and arm.
    p.disarm().await.expect("disarm");
    p.arm().await.expect("arm");
    let resumed = Instant::now();
    wait_until("driven again", || {
        driven_to(&log, 1.5)
            && log
                .lock()
                .expect("log")
                .last()
                .is_some_and(|e| e.0 > resumed)
    })
    .await;
    model.stop();
    run.join().expect("run thread").0.expect("a clean stop");
    stop.store(true, Ordering::Relaxed);
    robot_loop.join().expect("robot loop");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_run_stops_with_session_lost_when_its_robot_is_lost() {
    let world = World::start().await;
    let robot = world.robot().await;
    let model = world.controller("model", PriorityClass::Model).await;
    let (stop, log): (_, Log) = (Arc::new(AtomicBool::new(false)), Arc::default());
    let robot_loop = spin(robot.clone(), stop.clone(), log.clone());
    let run = run_in_thread(&model, 1.5);
    wait_until("the robot driven by the policy", || driven_to(&log, 1.5)).await;
    // The robot process dies: no detach, its heartbeats just stop.
    stop.store(true, Ordering::Relaxed);
    robot_loop.join().expect("robot loop");
    let robot = Arc::try_unwrap(robot).ok().expect("the only handle");
    let transport = robot.participant().transport().clone();
    drop(robot);
    transport.close().await.expect("close");
    let (result, _) = tokio::task::spawn_blocking(move || run.join().expect("run thread"))
        .await
        .expect("join");
    assert!(
        matches!(result, Err(RunError::Lost(ref r)) if r.code == ReasonCode::RobotLost),
        "{result:?}"
    );
    assert!(!model.holds());
}

/// A camera offered faster than its `send` ceiling, read by a bound policy
/// (an `every` contract): the image publisher's rate gate and the plan's
/// run on one clock, so frames keep flowing at the ceiling once the gate's
/// burst is spent.
#[tokio::test(flavor = "multi_thread")]
async fn camera_frames_keep_flowing_under_a_send_ceiling_and_a_contract() {
    let world = World::start().await;
    let p = world
        .join("ur10e", Role::Embodiment, PriorityClass::Scripted)
        .await;
    let robot = Robot::declare(
        p,
        r#"{"cam": {"kind": "camera", "width": 32, "height": 24, "hz": 20,
                    "send": "latest(hz=20)"}}"#,
        EdgeConfig::default(),
    )
    .await
    .expect("declare");
    let watcher = world
        .join("watcher", Role::Controller, PriorityClass::Model)
        .await;
    let controller = Controller::bind(
        Arc::new(watcher),
        &Policy::parse(r#"{"hz": 50, "inputs": {"cam": {"kind": "camera", "at": "cam"}}}"#)
            .expect("policy"),
        &RunOptions::named("watcher").robot("ur10e").expect("robot"),
        ControllerConfig::default(),
    )
    .await
    .expect("bind");
    let frame = vec![7_u8; 32 * 24 * 3];
    let clock = robot.participant().clock().clone();
    let mut newest = 0;
    let started = Instant::now();
    while newest < 10 {
        assert!(started.elapsed() < WAIT, "frames stopped at seq {newest}");
        robot
            .publish("cam", clock.now_ns(), &[&frame])
            .expect("offer");
        tokio::time::sleep(Duration::from_millis(30)).await;
        if let Ok((observation, _)) = controller.observe() {
            newest = observation.meta.based_on[0].seq_last;
        }
    }
}

/// An eye: two joints at 500 Hz, held then safe on its own.
const EYE: &str = r#"{
    "eyes": {"kind": "joints", "names": ["pan", "tilt"], "fields": ["pos"], "hz": 500,
             "command": "position", "safe": "hold", "hold_max_ms": 100}
}"#;

/// One policy for two eyes: no robot named, so it binds the session's
/// view, and each action targets its eye by a qualified selector.
const EYES: &str = r#"{
    "hz": 50,
    "actions": {
        "one": {"kind": "joints", "at": "eye-1:eyes", "command": "position", "hz": 100,
                "chunk": 5},
        "two": {"kind": "joints", "at": "eye-2:eyes", "command": "position", "hz": 100,
                "chunk": 5}
    }
}"#;

/// Both eyes to `target`, for the whole chunk.
struct Gaze {
    target: f32,
}

impl Predictor for Gaze {
    type Error = String;

    fn predict(&mut self, _: Observation, _: i64) -> Result<Vec<(String, Value)>, String> {
        let gaze = || {
            Value::Tensor(Tensor::from_f64(
                DType::F32,
                vec![5, 2],
                &[f64::from(self.target); 10],
            ))
        };
        Ok(vec![("one".into(), gaze()), ("two".into(), gaze())])
    }
}

/// An eye's 500 Hz loop: publish its gaze, then apply what is due.
fn spin_eye(robot: Arc<Robot>, stop: Arc<AtomicBool>, log: Log) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let clock = robot.participant().clock().clone();
        let mut gaze = [0.0_f32; 2];
        while !stop.load(Ordering::Relaxed) {
            let _ = robot.publish("eyes", clock.now_ns(), &[&f32s(&gaze)]);
            let due = robot.due("eyes").ok().flatten();
            let entry = match &due {
                Some(due) => {
                    let value = due
                        .action
                        .as_ref()
                        .map(|a| f32::from_le_bytes(a.value[..4].try_into().expect("4 bytes")));
                    if let Some(v) = value {
                        gaze[0] = v;
                    }
                    (
                        due.mode,
                        value,
                        due.action.as_ref().map(|a| a.chunk.lease_token),
                    )
                }
                None => (EdgeMode::Idle, None, None),
            };
            log.lock()
                .expect("log")
                .push((Instant::now(), entry.0, entry.1, entry.2));
            std::thread::sleep(Duration::from_millis(2));
        }
    })
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
            .all(|(_, mode, value, _)| *mode == EdgeMode::Driving && *value == Some(target))
}

#[tokio::test(flavor = "multi_thread")]
async fn one_policy_drives_two_eyes_and_keeps_driving_one_when_the_other_drops_off() {
    let world = World::start().await;
    let eye = |name: &'static str| {
        let world = &world;
        async move {
            let p = world
                .join(name, Role::Embodiment, PriorityClass::Scripted)
                .await;
            Arc::new(
                Robot::declare(p, EYE, EdgeConfig::default())
                    .await
                    .expect("declare"),
            )
        }
    };
    let (one, two) = (eye("eye-1").await, eye("eye-2").await);
    one.participant().arm().await.expect("arm");

    // No robot named: the policy binds the view of both eyes.
    let p = world
        .join("tracker", Role::Controller, PriorityClass::Model)
        .await;
    let tracker = Arc::new(
        Controller::bind(
            Arc::new(p),
            &Policy::parse(EYES).expect("policy"),
            &RunOptions::named("tracker"),
            ControllerConfig::default(),
        )
        .await
        .expect("one binding over both eyes"),
    );
    let stop = Arc::new(AtomicBool::new(false));
    let (log_one, log_two): (Log, Log) = (Arc::default(), Arc::default());
    let loop_one = spin_eye(one.clone(), stop.clone(), log_one.clone());
    let stop_two = Arc::new(AtomicBool::new(false));
    let loop_two = spin_eye(two.clone(), stop_two.clone(), log_two.clone());
    let run = {
        let tracker = tracker.clone();
        std::thread::spawn(move || {
            tracker.run(&mut Gaze { target: 0.25 }, Some(Duration::from_secs(60)))
        })
    };
    let started = Instant::now();
    wait_until("both eyes driven by one policy", || {
        driven_since(&log_one, 0.25, started) && driven_since(&log_two, 0.25, started)
    })
    .await;

    // Eye 2 drops off the network: no more heartbeats, no more ticks.
    two.participant().set_heartbeats_enabled(false);
    stop_two.store(true, Ordering::Relaxed);
    loop_two.join().expect("eye 2 loop");
    let eye_two = rlmesh_session::wire::ParticipantId::new("eye-2").expect("id");
    wait_until("the host loses eye 2", || {
        world
            .host
            .snapshot()
            .participant(&eye_two)
            .is_some_and(|p| !p.present)
    })
    .await;
    let lost = Instant::now();
    // Eye 1 keeps being driven. On a loaded machine a late chunk may hold it
    // for a moment (its queue runs dry), so wait for driving rather than
    // sample a fixed window; but losing eye 2 must never take it safe.
    let modes = |since: Instant| {
        let log = log_one.lock().expect("log");
        let mut counts: std::collections::BTreeMap<String, usize> = Default::default();
        for (_, mode, ..) in log.iter().filter(|(at, ..)| *at >= since) {
            *counts.entry(format!("{mode:?}")).or_default() += 1;
        }
        counts
    };
    let settled = lost + Duration::from_millis(300);
    while Instant::now() < settled || !driven_since(&log_one, 0.25, settled) {
        assert!(
            lost.elapsed() < WAIT,
            "eye 1 keeps moving after eye 2 is lost: {:?}",
            modes(lost)
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let since_lost = modes(lost);
    assert!(
        !since_lost.contains_key("Safe") && !since_lost.contains_key("Estop"),
        "losing eye 2 took eye 1 safe: {since_lost:?}"
    );
    let snapshot = world.host.snapshot();
    assert_ne!(
        snapshot.mode,
        Mode::Safe,
        "one lost eye never stops the session"
    );
    assert!(!run.is_finished(), "the policy keeps running");

    tracker.stop();
    run.join().expect("run thread").expect("a clean stop");
    stop.store(true, Ordering::Relaxed);
    loop_one.join().expect("eye 1 loop");
}

/// A robot the host dropped (its heartbeats stopped past the presence
/// grace) joins again on its own through the SDK, and after an arm the same
/// policy drives it again; the other eye never stops.
#[tokio::test(flavor = "multi_thread")]
async fn an_sdk_robot_the_host_dropped_joins_again_on_its_own() {
    use rlmesh_session::address::Address;
    use rlmesh_session::sdk;
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .with_test_writer()
        .try_init();
    let host = sdk::EmbeddedHost::start(
        &Address::session(":0").expect("address"),
        Duration::from_millis(300),
    )
    .await
    .expect("host");
    let address = host.address().clone();
    let join = |name: &'static str| {
        let address = address.clone();
        async move {
            let slot: sdk::RobotSlot = Arc::new(Mutex::new(Some(Arc::new(
                sdk::connect_robot(&address, name, EYE, WAIT)
                    .await
                    .expect("declare"),
            ))));
            let keeper = sdk::keep_joined(slot.clone(), address, name.into(), EYE.into());
            (slot, keeper)
        }
    };
    let ((one, keep_one), (two, keep_two)) = (join("eye-1").await, join("eye-2").await);
    let current = |slot: &sdk::RobotSlot| slot.lock().expect("slot").clone().expect("robot");
    current(&one).participant().arm().await.expect("arm");
    let tracker = Arc::new(
        sdk::bind_policy(
            &address,
            "tracker",
            EYES,
            &RunOptions::named("tracker"),
            PriorityClass::Model,
            WAIT,
        )
        .await
        .expect("one binding over both eyes"),
    );
    // Each eye's loop reads its slot every tick, as a binding does.
    let spin = |slot: sdk::RobotSlot, stop: Arc<AtomicBool>, log: Log| {
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let robot = slot.lock().expect("slot").clone().expect("robot");
                let now = robot.participant().clock().now_ns();
                let _ = robot.publish("eyes", now, &[&f32s(&[0.0, 0.0])]);
                let due = robot.due("eyes").ok().flatten();
                let value = due
                    .as_ref()
                    .and_then(|d| d.action.as_ref())
                    .map(|a| f32::from_le_bytes(a.value[..4].try_into().expect("4 bytes")));
                let mode = due.map_or(EdgeMode::Idle, |d| d.mode);
                log.lock()
                    .expect("log")
                    .push((Instant::now(), mode, value, None));
                std::thread::sleep(Duration::from_millis(2));
            }
        })
    };
    let stop = Arc::new(AtomicBool::new(false));
    let (log_one, log_two): (Log, Log) = (Arc::default(), Arc::default());
    let loops = [
        spin(one.clone(), stop.clone(), log_one.clone()),
        spin(two.clone(), stop.clone(), log_two.clone()),
    ];
    let run = {
        let tracker = tracker.clone();
        std::thread::spawn(move || {
            tracker.run(&mut Gaze { target: 0.25 }, Some(Duration::from_secs(60)))
        })
    };
    let started = Instant::now();
    wait_until("both eyes driven", || {
        driven_since(&log_one, 0.25, started) && driven_since(&log_two, 0.25, started)
    })
    .await;

    // Eye 2's heartbeats stop: the host drops it, and it joins again.
    let first = current(&two);
    first.participant().set_heartbeats_enabled(false);
    wait_until("eye 2 joined again", || {
        !Arc::ptr_eq(&current(&two), &first)
    })
    .await;
    drop(first);
    let back = Instant::now();
    // It waits to be re-armed by name (robots don't re-arm on their own by
    // default); a routine arm is not one.
    let driver = current(&one);
    driver.participant().arm().await.expect("a no-op");
    let eye_two = current(&two).participant().id().clone();
    wait_until("the policy sees eye 2 held", || {
        tracker.held_refusals().iter().any(|(r, _)| *r == eye_two)
    })
    .await;
    let (_, why) = tracker.held_refusals().remove(0);
    assert!(why.contains("robot.rearm(\"eye-2\")"), "{why}");
    wait_until("the run reports the hold", || {
        tracker.drain_events().iter().any(
            |e| matches!(e, rlmesh_session::RunEvent::RobotHeld { message } if message.contains("eye-2")),
        )
    })
    .await;
    driver
        .participant()
        .rearm(&eye_two)
        .await
        .expect("re-arm eye 2");
    wait_until("the policy sees eye 2 re-armed", || {
        tracker.held_refusals().is_empty()
    })
    .await;
    wait_until("eye 2 driven again", || driven_since(&log_two, 0.25, back)).await;
    wait_until("eye 1 still driven", || driven_since(&log_one, 0.25, back)).await;

    tracker.stop();
    run.join().expect("run thread").expect("a clean stop");
    stop.store(true, Ordering::Relaxed);
    for spin in loops {
        spin.join().expect("eye loop");
    }
    keep_one.abort();
    keep_two.abort();
    host.close().await.expect("close");
}

/// Commands the arm, and reads nothing: nothing to warm or install.
const COMMAND_ONLY: &str = r#"{
    "hz": 50,
    "actions": {"arm": {"kind": "joints", "at": "arm", "command": "position",
                        "hz": 100, "chunk": 5}}
}"#;

fn command_values() -> Vec<(String, Value)> {
    let tensor = Tensor::from_f64(DType::F32, vec![5, 6], &[0.0; 30]);
    vec![("arm".into(), Value::Tensor(tensor))]
}

/// A controller bound and acquired at once, as an operator's first command
/// is, before the host has measured its clock.
#[tokio::test(flavor = "multi_thread")]
async fn the_first_chunk_after_an_explicit_acquire_is_accepted() {
    let world = World::start().await;
    let robot = world.robot().await;
    for round in 0..5 {
        let id = format!("op{round}");
        let p = world
            .join(&id, Role::Controller, PriorityClass::Operator)
            .await;
        let options = RunOptions::named(&id).robot("ur10e").expect("robot");
        let controller = Controller::bind(
            Arc::new(p),
            &Policy::parse(COMMAND_ONLY).expect("policy"),
            &options,
            ControllerConfig::default(),
        )
        .await
        .expect("bind");
        controller.acquire().await.expect("acquire");
        let before = robot.edge_stats("arm").expect("stats").accepted;
        let now = controller.participant().session_now_ns();
        controller.send(&command_values(), &[], now).expect("send");
        wait_until("the first chunk reaches the edge", || {
            let stats = robot.edge_stats("arm").expect("stats");
            stats.accepted > before || stats.rejected_total() > 0
        })
        .await;
        let stats = robot.edge_stats("arm").expect("stats");
        assert_eq!(stats.rejected_total(), 0, "round {round}: {stats:?}");
        controller.release().await.expect("release");
    }
}

/// A controller whose clock the host never maps: no heartbeat after attach.
#[tokio::test(flavor = "multi_thread")]
async fn an_acquire_whose_clock_is_never_mapped_says_whose() {
    let world = World::start().await;
    let robot = world.robot().await;
    let transport = Transport::open(
        &TransportConfig::per_class_tcp(
            Some("127.0.0.1:0".parse().expect("addr")),
            Some(world.base),
        )
        .expect("config"),
    )
    .await
    .expect("transport");
    let mut config = ParticipantConfig::new(
        world.tenant.clone(),
        world.session,
        rlmesh_session::wire::ParticipantId::new("mute").expect("id"),
        vec![Role::Controller],
        PriorityClass::Operator,
    );
    config.clock = SystemMonotonicClock.shared();
    config.heartbeat_period = Duration::from_secs(3600);
    config.credential = Some(world.host.credential());
    let p = Participant::attach(transport, config)
        .await
        .expect("attach");
    let controller = Controller::bind(
        Arc::new(p),
        &Policy::parse(COMMAND_ONLY).expect("policy"),
        &RunOptions::named("mute").robot("ur10e").expect("robot"),
        ControllerConfig {
            bind_timeout: Duration::from_millis(500),
            ..ControllerConfig::default()
        },
    )
    .await
    .expect("bind");
    let err = controller.acquire().await.expect_err("never mapped");
    assert!(err.to_string().contains("the clock of mute:"), "{err}");
    assert_eq!(robot.edge_stats("arm").expect("stats").rejected_total(), 0);
}
