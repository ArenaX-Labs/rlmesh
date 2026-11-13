//! A declared robot ([`Robot`]) and bound policies ([`Controller`]) over a
//! real session: the run loop drives the edge, an operator preempts the
//! model, which stays stopped until resumed, a failing model lets go of the
//! robot.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rlmesh_session::edge::{EdgeConfig, EdgeMode, Robot};
use rlmesh_session::spec::apply::{Tensor, Value};
use rlmesh_session::spec::{Policy, RunOptions};
use rlmesh_session::transport::open_local_listener;
use rlmesh_session::wire::version::LEASE_HANDBACK;
use rlmesh_session::wire::{
    CtlStatus, DType, LeaseToken, Mode, PriorityClass, ReasonCode, Role, SessionId, TenantId,
};
use rlmesh_session::{
    AuthorityEvent, Controller, ControllerConfig, HandBack, HostConfig, Observation, Parked,
    Participant, ParticipantConfig, Predictor, RunConfig, RunError, RunEvent, SessionError,
    SessionHost, SkipReason, SystemMonotonicClock, Transport, TransportConfig,
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
        let (transport, base) = open_local_listener("127.0.0.1".parse().expect("ip"))
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
        self.join_with(id, role, priority, |_| {}).await
    }

    async fn join_with(
        &self,
        id: &str,
        role: Role,
        priority: PriorityClass,
        configure: impl FnOnce(&mut ParticipantConfig),
    ) -> Participant {
        let transport = Transport::open(&TransportConfig::per_class_tcp(self.base))
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
        configure(&mut config);
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
        self.controller_with(id, priority, |_| {}).await
    }

    async fn controller_with(
        &self,
        id: &str,
        priority: PriorityClass,
        configure: impl FnOnce(&mut ParticipantConfig),
    ) -> Arc<Controller> {
        let p = self
            .join_with(id, Role::Controller, priority, configure)
            .await;
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
        let result = controller.run(
            &mut model,
            RunConfig::default().duration(Duration::from_secs(30)),
        );
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

/// The edge's mode on its last tick.
fn edge_mode(log: &Log) -> Option<EdgeMode> {
    log.lock().expect("log").last().map(|(_, mode, ..)| *mode)
}

/// The parked events drained so far.
fn parked_events(
    controller: &Controller,
    seen: &mut Vec<RunEvent>,
) -> Vec<(String, String, String)> {
    seen.extend(controller.drain_events());
    seen.iter()
        .filter_map(|e| match e {
            RunEvent::Parked { part, by, message } => {
                Some((part.clone(), by.clone(), message.clone()))
            }
            _ => None,
        })
        .collect()
}

/// Waits until `controller` has skipped `n` more starts as `parked`, the
/// robot's arm has gone safe, and nothing was sent meanwhile: it stayed
/// stopped although nobody holds the arm.
async fn stays_parked(world: &World, controller: &Controller, log: &Log, n: u64) {
    let (skipped, sent) = {
        let stats = controller.stats();
        (stats.skipped(SkipReason::Parked), stats.chunks_sent)
    };
    wait_until("parked starts with the arm safe", || {
        controller.stats().skipped(SkipReason::Parked) >= skipped + n
            && edge_mode(log) == Some(EdgeMode::Safe)
    })
    .await;
    assert!(!controller.holds_any());
    assert_eq!(controller.stats().chunks_sent, sent, "sent while parked");
    assert!(
        world.host.snapshot().leases.is_empty(),
        "nobody holds the arm"
    );
}

/// An operator takes the arm from a running model and lets go: the model
/// stays stopped, the arm goes safe, and only `resume` hands it back (under
/// a fresh token). The same when the operator's process dies.
#[tokio::test(flavor = "multi_thread")]
async fn an_operator_preempts_the_model_which_stays_stopped_until_resumed() {
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
    let operator_id = operator.participant().id().clone();
    let taken = Parked {
        part: "ur10e:arm".into(),
        by: operator_id.clone(),
        priority: PriorityClass::Operator,
    };
    assert_eq!(model.parked(), vec![taken.clone()]);
    let mut events = Vec::new();
    wait_until("the model says it is parked", || {
        !parked_events(&model, &mut events).is_empty()
    })
    .await;
    assert_eq!(
        parked_events(&model, &mut events),
        vec![(
            "ur10e:arm".to_string(),
            "operator".to_string(),
            "ur10e:arm was taken by operator at Operator; this policy stays stopped until \
             policy.resume()"
                .to_string()
        )]
    );
    assert!(model.stats().skipped(SkipReason::Parked) > 0);

    // The operator lets go: the model does not take the arm back.
    operator_driving.store(false, Ordering::Relaxed);
    streaming.join().expect("operator thread");
    operator.release().await.expect("release");
    stays_parked(&world, &model, &log, 25).await;
    assert_eq!(model.parked(), vec![taken.clone()]);

    // Resumed, it asks again and drives under a fresh token.
    assert_eq!(model.resume(), vec!["ur10e:arm".to_string()]);
    assert!(model.parked().is_empty());
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

    // The operator takes it again, and its process dies: its lease lapses,
    // and the model still stays stopped.
    operator.acquire().await.expect("operator preempts");
    wait_until("the model preempted", || !model.holds()).await;
    operator.participant().set_heartbeats_enabled(false);
    drop(operator);
    wait_until("the operator's lease lapsed", || {
        world.host.snapshot().leases.is_empty()
    })
    .await;
    stays_parked(&world, &model, &log, 25).await;
    assert_eq!(model.parked(), vec![taken]);
    assert_eq!(parked_events(&model, &mut events).len(), 2, "{events:#?}");
    assert_eq!(model.resume(), vec!["ur10e:arm".to_string()]);
    wait_until("the model driving after a resume", || driven_to(&log, 1.0)).await;

    model.stop();
    run.join().expect("run thread").0.expect("clean stop");
    stop.store(true, Ordering::Relaxed);
    robot_loop.join().expect("robot loop");
}

/// A loss that is not a higher class taking the arm leaves the model asking
/// for it again: a holder of its own class, a disarm and arm, its own
/// lease's lapse.
#[tokio::test(flavor = "multi_thread")]
async fn every_other_loss_leaves_the_model_asking_again() {
    let world = World::start().await;
    let robot = world.robot().await;
    let (stop, log): (_, Log) = (Arc::new(AtomicBool::new(false)), Arc::default());
    let robot_loop = spin(robot.clone(), stop.clone(), log.clone());
    // Another model holds the arm first: the run waits for it, unparked.
    let other = world.controller("other", PriorityClass::Model).await;
    other.acquire().await.expect("first come");
    let model = world.controller("model", PriorityClass::Model).await;
    let run = run_in_thread(&model, 1.0);
    wait_until("the model waiting for the arm", || {
        model.stats().skipped(SkipReason::NoLease) >= 10
    })
    .await;
    assert!(model.parked().is_empty());
    other.release().await.expect("release");
    wait_until("the model driving once the arm is free", || {
        driven_to(&log, 1.0)
    })
    .await;

    // Disarmed and armed: every lease ends, and it asks again.
    robot.participant().disarm().await.expect("disarm");
    wait_until("the lease voided", || !model.holds_any()).await;
    robot.participant().arm().await.expect("arm");
    let armed = Instant::now();
    wait_until("the model driving after the arm", || {
        model.holds() && driven_since(&log, 1.0, armed)
    })
    .await;

    // Its own lease lapses (no heartbeats past its ttl), then it is back.
    model.participant().set_heartbeats_enabled(false);
    wait_until("the model's lease lapsed", || {
        world.host.snapshot().leases.is_empty()
    })
    .await;
    model.participant().set_heartbeats_enabled(true);
    let back = Instant::now();
    wait_until("the model driving after its lapse", || {
        model.holds() && driven_since(&log, 1.0, back)
    })
    .await;
    assert!(model.parked().is_empty());
    assert_eq!(model.stats().skipped(SkipReason::Parked), 0);

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
        tokio::task::spawn_blocking(move || {
            model.run(&mut Failing(0), RunConfig::default().duration(WAIT))
        })
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
    // An eye whose chunks run dry past its hold (a starved test machine)
    // goes safe on its own: re-armed at once, so only what eye 2's loss
    // does to eye 1 decides the test, never the machine's load.
    let world = World::start_with(|config| config.rearm_robots = true).await;
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
            tracker.run(
                &mut Gaze { target: 0.25 },
                RunConfig::default().duration(Duration::from_secs(60)),
            )
        })
    };
    let started = Instant::now();
    wait_until("both eyes driven by one policy", || {
        driven_since(&log_one, 0.25, started) && driven_since(&log_two, 0.25, started)
    })
    .await;

    let mut events = world.host.events();
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
    // Eye 1 keeps being driven, after the loss as before it.
    while !driven_since(&log_one, 0.25, lost) {
        assert!(
            lost.elapsed() < WAIT,
            "eye 1 keeps moving after eye 2 is lost"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    // What the loss did, from the host's own account: it ended eye 2's
    // lease and held eye 2, and it never stopped the session, nor ended or
    // held anything of eye 1's.
    let eye_one = rlmesh_session::wire::ParticipantId::new("eye-1").expect("id");
    let mut told = Vec::new();
    loop {
        let event = match events.try_recv() {
            Ok(event) => event,
            Err(tokio::sync::broadcast::error::TryRecvError::Empty) => break,
            Err(err) => panic!("the host's events: {err}"),
        };
        told.push(event.to_string());
        match event {
            AuthorityEvent::ModeChanged { to, reason, .. } => assert!(
                !matches!(to, Mode::Safe | Mode::Estop),
                "one lost eye stopped the session: {reason}"
            ),
            AuthorityEvent::LeaseEnded { part, reason, .. } => assert!(
                !(part.starts_with("eye-1:") && reason.by == "eye-2"),
                "losing eye 2 ended eye 1's lease: {reason}"
            ),
            AuthorityEvent::RobotHeld { robot, reason } => assert!(
                !(robot == eye_one && reason.by == "eye-2"),
                "losing eye 2 held eye 1: {reason}"
            ),
            _ => {}
        }
    }
    assert!(
        told.iter().any(|e| e.contains("eye-2")),
        "the host told of eye 2's loss: {told:?}"
    );
    assert_ne!(
        world.host.snapshot().mode,
        Mode::Safe,
        "one lost eye never stops the session"
    );
    assert!(!run.is_finished(), "the policy keeps running");

    tracker.stop();
    run.join().expect("run thread").expect("a clean stop");
    stop.store(true, Ordering::Relaxed);
    loop_one.join().expect("eye 1 loop");
}

/// Eye 1 alone, for an operator.
const EYE_ONE: &str = r#"{
    "hz": 50,
    "actions": {
        "one": {"kind": "joints", "at": "eye-1:eyes", "command": "position", "hz": 100,
                "chunk": 5}
    }
}"#;

/// An operator takes one of the two eyes a policy drives and lets go: that
/// eye stays stopped (safe) until the policy is resumed, and the other is
/// driven all along.
#[tokio::test(flavor = "multi_thread")]
async fn an_operator_taking_one_eye_parks_only_that_eye() {
    let world = World::start_with(|config| config.rearm_robots = true).await;
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
    let bind = |name: &'static str, policy: &'static str, priority| {
        let world = &world;
        async move {
            let p = world.join(name, Role::Controller, priority).await;
            Arc::new(
                Controller::bind(
                    Arc::new(p),
                    &Policy::parse(policy).expect("policy"),
                    &RunOptions::named(name),
                    ControllerConfig::default(),
                )
                .await
                .expect("bind"),
            )
        }
    };
    let tracker = bind("tracker", EYES, PriorityClass::Model).await;
    let operator = bind("operator", EYE_ONE, PriorityClass::Operator).await;
    let stop = Arc::new(AtomicBool::new(false));
    let (log_one, log_two): (Log, Log) = (Arc::default(), Arc::default());
    let loops = [
        spin_eye(one.clone(), stop.clone(), log_one.clone()),
        spin_eye(two.clone(), stop.clone(), log_two.clone()),
    ];
    let run = {
        let tracker = tracker.clone();
        std::thread::spawn(move || {
            tracker.run(
                &mut Gaze { target: 0.25 },
                RunConfig::default().duration(Duration::from_secs(60)),
            )
        })
    };
    let started = Instant::now();
    wait_until("both eyes driven", || {
        driven_since(&log_one, 0.25, started) && driven_since(&log_two, 0.25, started)
    })
    .await;

    operator.acquire().await.expect("the operator takes eye 1");
    operator.release().await.expect("and lets go");
    let released = Instant::now();
    wait_until("eye 1 safe, eye 2 driven", || {
        edge_mode(&log_one) == Some(EdgeMode::Safe) && driven_since(&log_two, 0.25, released)
    })
    .await;
    let parked: Vec<String> = tracker.parked().into_iter().map(|p| p.part).collect();
    assert_eq!(parked, ["eye-1:eyes"]);
    // Starts go on, for eye 2: none is skipped as parked.
    let sent = tracker.stats().chunks_sent;
    wait_until("eye 2 still commanded", || {
        tracker.stats().chunks_sent >= sent + 10
    })
    .await;
    assert_eq!(tracker.stats().skipped(SkipReason::Parked), 0);
    assert_eq!(edge_mode(&log_one), Some(EdgeMode::Safe));
    assert!(
        world
            .host
            .snapshot()
            .leases
            .iter()
            .all(|l| l.holder.as_str() == "tracker"),
        "only eye 2 is leased, to the tracker"
    );
    assert_eq!(tracker.leases().len(), 1);

    assert_eq!(tracker.resume(), vec!["eye-1:eyes".to_string()]);
    let resumed = Instant::now();
    wait_until("both eyes driven again", || {
        driven_since(&log_one, 0.25, resumed) && driven_since(&log_two, 0.25, resumed)
    })
    .await;

    tracker.stop();
    run.join().expect("run thread").expect("a clean stop");
    stop.store(true, Ordering::Relaxed);
    for spin in loops {
        spin.join().expect("eye loop");
    }
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
            tracker.run(
                &mut Gaze { target: 0.25 },
                RunConfig::default().duration(Duration::from_secs(60)),
            )
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
/// Its acquire gives up after the wait timeout set after binding, not the
/// (long) bind timeout.
#[tokio::test(flavor = "multi_thread")]
async fn an_acquire_whose_clock_is_never_mapped_says_whose() {
    let world = World::start().await;
    let robot = world.robot().await;
    let transport = Transport::open(&TransportConfig::per_class_tcp(world.base))
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
    let mut controller = Controller::bind(
        Arc::new(p),
        &Policy::parse(COMMAND_ONLY).expect("policy"),
        &RunOptions::named("mute").robot("ur10e").expect("robot"),
        ControllerConfig {
            bind_timeout: Duration::from_secs(3600),
            ..ControllerConfig::default()
        },
    )
    .await
    .expect("bind");
    controller.set_wait_timeout(Duration::from_millis(500));
    let err = tokio::time::timeout(Duration::from_secs(60), controller.acquire())
        .await
        .expect("waits the wait timeout, not the bind timeout")
        .expect_err("never mapped");
    assert!(err.to_string().contains("the clock of mute:"), "{err}");
    assert_eq!(robot.edge_stats("arm").expect("stats").rejected_total(), 0);
}

/// `Duration::MAX` is a wait with no limit, for the bind and its later
/// waits alike, not an overflow.
#[tokio::test(flavor = "multi_thread")]
async fn a_bind_and_acquire_without_a_limit_complete() {
    let world = World::start().await;
    let _robot = world.robot().await;
    let p = world
        .join("unhurried", Role::Controller, PriorityClass::Operator)
        .await;
    let mut controller = Controller::bind(
        Arc::new(p),
        &Policy::parse(COMMAND_ONLY).expect("policy"),
        &RunOptions::named("unhurried")
            .robot("ur10e")
            .expect("robot"),
        ControllerConfig {
            bind_timeout: Duration::MAX,
            ..ControllerConfig::default()
        },
    )
    .await
    .expect("bind");
    controller.set_wait_timeout(Duration::MAX);
    tokio::time::timeout(Duration::from_secs(60), controller.acquire())
        .await
        .expect("acquired once the clocks are mapped")
        .expect("acquire");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_refusal_for_a_present_robot_comes_at_once() {
    let world = World::start().await;
    let _robot = world.robot().await;
    let p = world
        .join("wrong", Role::Controller, PriorityClass::Model)
        .await;
    let policy = Policy::parse(r#"{"hz": 50, "inputs": {"c": {"kind": "camera", "at": "cam"}}}"#)
        .expect("policy");
    let options = RunOptions::named("wrong").robot("ur10e").expect("robot");
    let bind = Controller::bind(
        Arc::new(p),
        &policy,
        &options,
        ControllerConfig {
            bind_timeout: Duration::from_secs(3600),
            ..ControllerConfig::default()
        },
    );
    let Err(err) = tokio::time::timeout(Duration::from_secs(60), bind)
        .await
        .expect("no wait out of the bind timeout")
    else {
        panic!("bound");
    };
    assert!(
        err.to_string().contains("ur10e has no camera part"),
        "{err}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn binding_to_a_robot_that_never_appears_names_it() {
    let world = World::start().await;
    let _robot = world.robot().await;
    let p = world
        .join("waiter", Role::Controller, PriorityClass::Model)
        .await;
    let Err(err) = Controller::bind(
        Arc::new(p),
        &Policy::parse(COMMAND_ONLY).expect("policy"),
        &RunOptions::named("waiter").robot("ghost").expect("robot"),
        ControllerConfig {
            bind_timeout: Duration::from_millis(100),
            ..ControllerConfig::default()
        },
    )
    .await
    else {
        panic!("bound");
    };
    assert!(
        err.to_string()
            .contains("waiting for robot ghost to declare its parts"),
        "{err}"
    );
}
/// With no robot in the session, a policy whose entries name their robots
/// says which from its first wait, as one reading them beside it does.
#[tokio::test(flavor = "multi_thread")]
async fn a_bind_with_no_robot_yet_names_the_robots_its_entries_name() {
    let world = World::start().await;
    let policy = r#"{"hz": 50,
        "inputs": {"q": {"kind": "joints", "at": "b:arm", "fields": ["pos"]}},
        "actions": {"arm": {"kind": "joints", "at": "a:arm", "command": "position"}}}"#;
    let p = world
        .join("tracker", Role::Controller, PriorityClass::Model)
        .await;
    let Err(err) = Controller::bind(
        Arc::new(p),
        &Policy::parse(policy).expect("policy"),
        &RunOptions::named("tracker"),
        ControllerConfig {
            bind_timeout: Duration::from_millis(100),
            ..ControllerConfig::default()
        },
    )
    .await
    else {
        panic!("bound");
    };
    let text = err.to_string();
    assert!(
        text.contains("waiting for robots b, a to declare their parts; input \"q\""),
        "{text}"
    );
}

/// Reads the arm, and a part of a robot that may not have started.
const READS_TRACKER: &str = r#"{
    "hz": 50,
    "inputs": {"q": {"kind": "joints", "at": "arm", "fields": ["pos"]},
               "target": {"kind": "joints", "at": "tracker:target", "fields": ["pos"]}}
}"#;

const TRACKER: &str = r#"{"target": {"kind": "joints", "count": 2, "fields": ["pos"], "hz": 50}}"#;

#[tokio::test(flavor = "multi_thread")]
async fn binding_waits_for_a_named_robot_that_starts_later() {
    let world = World::start().await;
    let _robot = world.robot().await;
    let p = world
        .join("reader", Role::Controller, PriorityClass::Model)
        .await;
    let options = RunOptions::named("reader").robot("ur10e").expect("robot");
    let policy = Policy::parse(READS_TRACKER).expect("policy");
    let binding = tokio::spawn(async move {
        Controller::bind(Arc::new(p), &policy, &options, ControllerConfig::default()).await
    });
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(!binding.is_finished(), "it waits for the tracker");
    let tracker = world
        .join("tracker", Role::Embodiment, PriorityClass::Scripted)
        .await;
    tracker.declare_parts(TRACKER).await.expect("declare");
    let controller = binding.await.expect("join").expect("bound");
    assert!(controller.binding().input("target").is_some());
}

/// A robot attached but not declared yet is waited for like one not
/// attached; once it has declared, a part it lacks is refused at once.
#[tokio::test(flavor = "multi_thread")]
async fn a_robot_attached_with_no_parts_is_waited_for_and_one_lacking_the_part_is_refused() {
    let world = World::start().await;
    let _robot = world.robot().await;
    let tracker = world
        .join("tracker", Role::Embodiment, PriorityClass::Scripted)
        .await;
    let config = ControllerConfig {
        bind_timeout: Duration::from_secs(3600),
        ..ControllerConfig::default()
    };
    let p = world
        .join("reader", Role::Controller, PriorityClass::Model)
        .await;
    let binding = tokio::spawn({
        let config = config.clone();
        async move {
            Controller::bind(
                Arc::new(p),
                &Policy::parse(READS_TRACKER).expect("policy"),
                &RunOptions::named("reader").robot("ur10e").expect("robot"),
                config,
            )
            .await
        }
    });
    tracker.declare_parts(TRACKER).await.expect("declare");
    let controller = tokio::time::timeout(Duration::from_secs(60), binding)
        .await
        .expect("bound once the tracker declared")
        .expect("join")
        .expect("bound");
    assert!(controller.binding().input("target").is_some());

    let p = world
        .join("other", Role::Controller, PriorityClass::Model)
        .await;
    let policy = Policy::parse(
        r#"{"hz": 50, "inputs": {"aim": {"kind": "joints", "at": "tracker:aim", "fields": ["pos"]}}}"#,
    )
    .expect("policy");
    let options = RunOptions::named("other").robot("ur10e").expect("robot");
    let bind = Controller::bind(Arc::new(p), &policy, &options, config);
    let Err(err) = tokio::time::timeout(Duration::from_secs(60), bind)
        .await
        .expect("refused at once")
    else {
        panic!("bound");
    };
    assert!(
        err.to_string()
            .contains("no part matches; tracker's joints parts: target"),
        "{err}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_named_input_that_never_appears_says_which_robot_and_entry() {
    let world = World::start().await;
    let _robot = world.robot().await;
    let p = world
        .join("reader", Role::Controller, PriorityClass::Model)
        .await;
    let Err(err) = Controller::bind(
        Arc::new(p),
        &Policy::parse(READS_TRACKER).expect("policy"),
        &RunOptions::named("reader").robot("ur10e").expect("robot"),
        ControllerConfig {
            bind_timeout: Duration::from_millis(100),
            ..ControllerConfig::default()
        },
    )
    .await
    else {
        panic!("bound");
    };
    let text = err.to_string();
    assert!(
        text.contains("waiting for robot tracker to declare its parts"),
        "{text}"
    );
    assert!(text.contains("at \"tracker:target\""), "{text}");
}

/// Echoes the arm's joints as one action: nothing else to read.
struct Echo;

impl Predictor for Echo {
    type Error = String;

    fn predict(&mut self, _: Observation, _: i64) -> Result<Vec<(String, Value)>, Self::Error> {
        Ok(command_values())
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_run_on_a_disarmed_session_says_it_waits_to_be_armed() {
    let world = World::start().await;
    let p = world
        .join("ur10e", Role::Embodiment, PriorityClass::Scripted)
        .await;
    let robot = Arc::new(
        Robot::declare(p, UR10E, EdgeConfig::default())
            .await
            .expect("declare"),
    );
    let p = world
        .join("waiter", Role::Controller, PriorityClass::Model)
        .await;
    let controller = Arc::new(
        Controller::bind(
            Arc::new(p),
            &Policy::parse(COMMAND_ONLY).expect("policy"),
            &RunOptions::named("waiter").robot("ur10e").expect("robot"),
            ControllerConfig::default(),
        )
        .await
        .expect("bind"),
    );
    let waiting = controller.waiting().expect("disarmed");
    assert!(waiting.starts_with("the session is DISARMED"), "{waiting}");
    let run = {
        let controller = controller.clone();
        std::thread::spawn(move || controller.run(&mut Echo, RunConfig::default()))
    };
    let waits = |controller: &Controller| {
        controller
            .drain_events()
            .into_iter()
            .filter(|e| matches!(e, rlmesh_session::RunEvent::WaitingToArm { .. }))
            .count()
    };
    let mut seen = 0;
    wait_until("the run says it waits", || {
        seen += waits(&controller);
        seen > 0
    })
    .await;
    robot.participant().arm().await.expect("arm");
    wait_until("the run commands once armed", || controller.holds_any()).await;
    assert_eq!(controller.waiting(), None);
    controller.stop();
    run.join()
        .expect("run")
        .map_err(|e| e.to_string())
        .expect("ran");
    assert_eq!(seen + waits(&controller), 1, "warned once while disarmed");
}

/// A policy that reads a part of its own would wait for itself: refused at
/// once, never waited for.
#[tokio::test(flavor = "multi_thread")]
async fn a_policy_reading_its_own_part_is_refused_at_once() {
    let world = World::start().await;
    let _robot = world.robot().await;
    let p = world
        .join("tracker", Role::Controller, PriorityClass::Model)
        .await;
    let policy = Policy::parse(
        r#"{"hz": 50, "inputs": {"aim": {"kind": "joints", "at": "tracker:aim", "fields": ["pos"]}},
            "actions": {"arm": {"kind": "joints", "at": "arm", "command": "position"}}}"#,
    )
    .expect("policy");
    let started = Instant::now();
    let Err(err) = Controller::bind(
        Arc::new(p),
        &policy,
        &RunOptions::named("tracker").robot("ur10e").expect("robot"),
        ControllerConfig::default(),
    )
    .await
    else {
        panic!("bound");
    };
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "waited for itself"
    );
    let text = err.to_string();
    assert!(
        text.contains(r#"input "aim" (at "tracker:aim") names tracker, this policy itself"#),
        "{text}"
    );
}

/// A lease grace no stall of a loaded machine outlasts, for the hand-back
/// tests, where leases change hands only by preemption and release. A lapse
/// would end the operator's lease (the host missing its heartbeats, or the
/// edge its renewals), its keeper would take the part again from nobody, and
/// no hand-back follows a lease that took its part from nobody.
const NO_LAPSE: Duration = Duration::from_secs(2);

/// An eye whose hold outlasts a hand-back on a loaded machine.
const EYE_HOLDING: &str = r#"{
    "eyes": {"kind": "joints", "names": ["pan", "tilt"], "fields": ["pos"], "hz": 500,
             "command": "position", "safe": "hold", "hold_max_ms": 500}
}"#;

/// Eye 2 alone, for a second operator.
const EYE_TWO: &str = r#"{
    "hz": 50,
    "actions": {
        "two": {"kind": "joints", "at": "eye-2:eyes", "command": "position", "hz": 100,
                "chunk": 5}
    }
}"#;

/// Streams `target` to action `entry` from `controller` until the flag
/// drops.
fn stream(
    controller: &Arc<Controller>,
    entry: &'static str,
    target: f64,
) -> (Arc<AtomicBool>, std::thread::JoinHandle<()>) {
    let (controller, on) = (controller.clone(), Arc::new(AtomicBool::new(true)));
    let flag = on.clone();
    let handle = std::thread::spawn(move || {
        while flag.load(Ordering::Relaxed) {
            if let Ok((observation, at)) = controller.observe() {
                let value = Value::Tensor(Tensor::from_f64(DType::F32, vec![5, 2], &[target; 10]));
                controller
                    .send(&[(entry.into(), value)], &observation.meta.based_on, at)
                    .expect("send");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    });
    (on, handle)
}

/// Two operators take one eye each from the policy driving both. The first
/// hands its eye back: the policy resumes that eye alone (a `HandedBack`
/// event), and between the release and its first chunk the eye only holds
/// the operator's last target, as on any change of holder, never going
/// safe. The second lets go plainly: its eye stays parked.
#[tokio::test(flavor = "multi_thread")]
async fn an_operator_hands_back_the_eye_it_took_and_only_that_eye_resumes() {
    let world = World::start_with(|config| {
        config.rearm_robots = true;
        config.lease_grace = NO_LAPSE;
    })
    .await;
    let eye = |name: &'static str| {
        let world = &world;
        async move {
            let p = world
                .join(name, Role::Embodiment, PriorityClass::Scripted)
                .await;
            Arc::new(
                Robot::declare(p, EYE_HOLDING, EdgeConfig::default())
                    .await
                    .expect("declare"),
            )
        }
    };
    let (one, two) = (eye("eye-1").await, eye("eye-2").await);
    one.participant().arm().await.expect("arm");
    let bind = |name: &'static str, policy: &'static str, priority| {
        let world = &world;
        async move {
            let p = world.join(name, Role::Controller, priority).await;
            Arc::new(
                Controller::bind(
                    Arc::new(p),
                    &Policy::parse(policy).expect("policy"),
                    &RunOptions::named(name),
                    ControllerConfig::default(),
                )
                .await
                .expect("bind"),
            )
        }
    };
    let tracker = bind("tracker", EYES, PriorityClass::Model).await;
    let first = bind("operator-1", EYE_ONE, PriorityClass::Operator).await;
    let second = bind("operator-2", EYE_TWO, PriorityClass::Operator).await;
    let stop = Arc::new(AtomicBool::new(false));
    let (log_one, log_two): (Log, Log) = (Arc::default(), Arc::default());
    let loops = [
        spin_eye(one.clone(), stop.clone(), log_one.clone()),
        spin_eye(two.clone(), stop.clone(), log_two.clone()),
    ];
    let run = {
        let tracker = tracker.clone();
        std::thread::spawn(move || {
            tracker.run(
                &mut Gaze { target: 0.25 },
                RunConfig::default().duration(Duration::from_secs(60)),
            )
        })
    };
    let started = Instant::now();
    wait_until("both eyes driven", || {
        driven_since(&log_one, 0.25, started) && driven_since(&log_two, 0.25, started)
    })
    .await;

    first.acquire().await.expect("operator 1 takes eye 1");
    second.acquire().await.expect("operator 2 takes eye 2");
    let (streaming, streamer) = stream(&first, "one", 0.75);
    let taken = Instant::now();
    wait_until("operator 1 driving eye 1", || {
        driven_since(&log_one, 0.75, taken)
    })
    .await;
    wait_until("both eyes parked", || tracker.parked().len() == 2).await;

    streaming.store(false, Ordering::Relaxed);
    streamer.join().expect("operator 1 thread");
    let released = Instant::now();
    let backs = first
        .release_to(PriorityClass::Model)
        .await
        .expect("handed back");
    assert_eq!(
        backs,
        [HandBack {
            to: tracker.participant().id().clone(),
            parts: vec!["eye-1:eyes".into()],
            taken: true,
        }]
    );
    wait_until("the tracker driving eye 1 again", || {
        driven_since(&log_one, 0.25, released)
    })
    .await;
    // From the release to the tracker's first action, eye 1 only held
    // operator 1's last target (its queue's tail until the edge saw the
    // release): no safe behaviour, no gap in dispatch.
    let since: Vec<_> = log_one
        .lock()
        .expect("log")
        .iter()
        .filter(|(at, ..)| *at >= released)
        .cloned()
        .collect();
    let first_action = since
        .iter()
        .position(|(_, mode, value, _)| *mode == EdgeMode::Driving && *value == Some(0.25))
        .expect("the tracker's first action");
    let gap = since[first_action].0 - released;
    assert!(
        since[..first_action]
            .iter()
            .all(
                |(_, mode, value, _)| matches!(mode, EdgeMode::Driving | EdgeMode::Hold)
                    && *value == Some(0.75)
            ),
        "eye 1 between the release and the tracker's first action ({gap:?}): {:?}",
        since[..first_action]
            .iter()
            .map(|(_, mode, value, _)| (*mode, *value))
            .collect::<Vec<_>>()
    );
    assert!(gap < Duration::from_millis(500), "within hold_max: {gap:?}");
    let tracker_lease = tracker
        .leases()
        .into_iter()
        .find(|l| world.host.snapshot().part_name(l.channel) == "eye-1:eyes")
        .expect("the tracker holds eye 1");
    assert_eq!(since[first_action].3, Some(tracker_lease.token));

    let parked: Vec<(String, String)> = tracker
        .parked()
        .into_iter()
        .map(|p| (p.part, p.by.to_string()))
        .collect();
    assert_eq!(parked, [("eye-2:eyes".into(), "operator-2".into())]);
    let handed: Vec<(String, String)> = tracker
        .drain_events()
        .into_iter()
        .filter_map(|e| match e {
            RunEvent::HandedBack { part, by, .. } => Some((part, by)),
            _ => None,
        })
        .collect();
    assert_eq!(handed, [("eye-1:eyes".into(), "operator-1".into())]);

    // A plain release leaves eye 2 parked: it goes safe, unleased.
    second.release().await.expect("plain release");
    wait_until("eye 2 safe", || edge_mode(&log_two) == Some(EdgeMode::Safe)).await;
    let sent = tracker.stats().chunks_sent;
    wait_until("eye 1 still commanded", || {
        tracker.stats().chunks_sent >= sent + 10
    })
    .await;
    assert_eq!(tracker.parked().len(), 1, "eye 2 stays parked");
    assert_eq!(edge_mode(&log_two), Some(EdgeMode::Safe));
    assert_eq!(tracker.leases().len(), 1, "only eye 1 is the tracker's");

    tracker.stop();
    run.join().expect("run thread").expect("a clean stop");
    stop.store(true, Ordering::Relaxed);
    for spin in loops {
        spin.join().expect("eye loop");
    }
}

/// A hand-back that cannot take is refused before anything is released,
/// and the operator keeps the arm: a model process too old to resume on
/// one, a part taken from nobody, nothing held, and a host too old to read
/// one.
#[tokio::test(flavor = "multi_thread")]
async fn a_hand_back_that_cannot_take_is_refused_and_the_operator_keeps_the_arm() {
    let refusal = |result: Result<Vec<HandBack>, SessionError>| match result {
        Err(SessionError::Rejected(err)) => (err.status, err.message),
        other => panic!("not refused: {other:?}"),
    };
    let world = World::start_with(|config| config.lease_grace = NO_LAPSE).await;
    let robot = world.robot().await;
    let (stop, log): (_, Log) = (Arc::new(AtomicBool::new(false)), Arc::default());
    let robot_loop = spin(robot.clone(), stop.clone(), log.clone());
    // A model built before hand-backs: it would stay parked.
    let model = world
        .controller_with("model", PriorityClass::Model, |c| c.supports.clear())
        .await;
    let operator = world.controller("operator", PriorityClass::Operator).await;
    let run = run_in_thread(&model, 1.0);
    wait_until("the model driving", || driven_to(&log, 1.0)).await;
    operator.acquire().await.expect("operator preempts");
    wait_until("the model parked", || !model.parked().is_empty()).await;
    assert_eq!(
        refusal(operator.release_to(PriorityClass::Model).await),
        (
            CtlStatus::Incompatible,
            "cannot hand ur10e:arm back to model: it does not support lease.handback and would \
             stay parked; release plainly and call resume() in its process"
                .into()
        )
    );
    assert!(operator.holds(), "nothing released");
    assert_eq!(model.parked().len(), 1);

    // Taken again while nobody held it: nothing is parked on it by this one.
    operator.release().await.expect("release");
    operator.acquire().await.expect("free part");
    let (status, message) = refusal(operator.release_to(PriorityClass::Model).await);
    assert_eq!(status, CtlStatus::InvalidRequest);
    assert!(
        message.ends_with("took it from nobody, so none is parked on it"),
        "{message}"
    );
    assert!(operator.holds());
    operator.release().await.expect("release");
    assert_eq!(
        refusal(operator.release_to(PriorityClass::Model).await).1,
        "operator holds no lease to hand back"
    );
    model.stop();
    run.join().expect("run thread").0.expect("clean stop");

    // A host that would read a hand-back as a plain release is not sent one.
    let old = World::start_with(|config| {
        config.supports.retain(|f| f != LEASE_HANDBACK);
        config.lease_grace = NO_LAPSE;
    })
    .await;
    let robot = old.robot().await;
    let (old_stop, old_log): (_, Log) = (Arc::new(AtomicBool::new(false)), Arc::default());
    let old_loop = spin(robot.clone(), old_stop.clone(), old_log.clone());
    let model = old.controller("model", PriorityClass::Model).await;
    let operator = old.controller("operator", PriorityClass::Operator).await;
    let run = run_in_thread(&model, 1.0);
    wait_until("the model driving", || driven_to(&old_log, 1.0)).await;
    operator.acquire().await.expect("operator preempts");
    let (status, message) = refusal(operator.release_to(PriorityClass::Model).await);
    assert_eq!(status, CtlStatus::Incompatible);
    assert!(
        message.starts_with("the session host does not support lease.handback"),
        "{message}"
    );
    assert!(operator.holds());
    model.stop();
    run.join().expect("run thread").0.expect("clean stop");
    for (stop, robot_loop) in [(stop, robot_loop), (old_stop, old_loop)] {
        stop.store(true, Ordering::Relaxed);
        robot_loop.join().expect("robot loop");
    }
}
