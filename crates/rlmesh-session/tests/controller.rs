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

/// The §14.1 UR10e's arm and wrist F/T, at 500 Hz.
const UR10E: &str = r#"{
    "arm": {"kind": "joints", "count": 6, "fields": ["pos"], "hz": 500,
            "command": "position", "hold_max_ms": 100, "max_action_age_ms": 1000},
    "wrist/ft": {"kind": "wrench", "frame": "tool0", "hz": 500}
}"#;

/// Reads the joints and the wrench, commands 5 actions at 100 Hz, replans
/// at 50 Hz.
const POLICY: &str = r#"{
    "hz": 50,
    "inputs": {"q": {"kind": "joints", "at": "arm", "fields": ["pos"]},
               "ft": {"kind": "wrench", "at": "wrist/ft", "frame": "tool0"}},
    "actions": {"arm": {"kind": "joints", "at": "arm", "command": "position",
                        "hz": 100, "horizon": 5}}
}"#;

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
        let tenant = TenantId::new("controller").expect("tenant");
        let session = SessionId::new();
        let mut config = HostConfig::new(tenant.clone(), session);
        config.clock = SystemMonotonicClock.shared();
        config.lease_grace = Duration::from_millis(300);
        config.health_log = None;
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

/// Commands every joint to `target` for the whole horizon.
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
