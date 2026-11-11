//! Shadow runs ([`RunMode::Shadow`]) over a real session: a policy that
//! cannot command still predicts and sends nothing, starts commanding the
//! moment it holds a lease, and keeps predicting while an operator has the
//! robot.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rlmesh_session::edge::{EdgeConfig, EdgeMode, Robot};
use rlmesh_session::spec::apply::{Tensor, Value};
use rlmesh_session::spec::{Policy, RunOptions};
use rlmesh_session::transport::open_local_listeners;
use rlmesh_session::wire::{DType, PriorityClass, Role, SessionId, TenantId};
use rlmesh_session::{
    AuthorityEvent, Controller, ControllerConfig, HostConfig, Observation, Participant,
    ParticipantConfig, Predictor, RunConfig, RunError, RunEvent, RunMode, SessionHost, SkipReason,
    Step, SystemMonotonicClock, Transport, TransportConfig,
};

const WAIT: Duration = Duration::from_secs(10);

const UR10E: &str = r#"{
    "arm": {"kind": "joints", "count": 6, "fields": ["pos"], "hz": 500,
            "command": "position", "safe": "stop", "hold_max_ms": 100, "max_action_age_ms": 1000}
}"#;

/// Reads the joints, commands 5 actions at 100 Hz, replans at 50 Hz.
const POLICY: &str = r#"{
    "hz": 50,
    "inputs": {"q": {"kind": "joints", "at": "arm", "fields": ["pos"]}},
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
        let tenant = TenantId::new("shadow").expect("tenant");
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

    /// The UR10e, declared and not armed.
    async fn robot(&self, id: &str) -> Arc<Robot> {
        let p = self
            .join(id, Role::Embodiment, PriorityClass::Scripted)
            .await;
        Arc::new(
            Robot::declare(p, UR10E, EdgeConfig::default())
                .await
                .expect("declare"),
        )
    }

    async fn controller(&self, id: &str, priority: PriorityClass) -> Arc<Controller> {
        let p = self.join(id, Role::Controller, priority).await;
        let options = RunOptions::named(id).robot("ur10e").expect("robot");
        Arc::new(
            Controller::bind(
                Arc::new(p),
                &Policy::parse(POLICY).expect("policy"),
                &options,
                ControllerConfig::default(),
            )
            .await
            .expect("bind"),
        )
    }
}

fn f32s(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The first joint of each action the robot dispatched.
type Log = Arc<Mutex<Vec<f32>>>;

/// The robot's 500 Hz loop: publish, then apply what is due.
fn spin(robot: Arc<Robot>, stop: Arc<AtomicBool>, log: Log) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let clock = robot.participant().clock().clone();
        while !stop.load(Ordering::Relaxed) {
            robot
                .publish("arm", clock.now_ns(), &[&f32s(&[0.5; 6])])
                .expect("publish arm");
            let due = robot.due("arm").expect("due");
            if let Some(action) = due
                .filter(|d| d.mode == EdgeMode::Driving)
                .and_then(|d| d.action)
            {
                let value = f32::from_le_bytes(action.value[..4].try_into().expect("4 bytes"));
                log.lock().expect("log").push(value);
            }
            std::thread::sleep(Duration::from_millis(2));
        }
    })
}

/// Chunks the robot's edge took in at all: accepted or refused.
fn chunks_received(robot: &Robot) -> u64 {
    let stats = robot.edge_stats("arm").expect("edge stats");
    stats.accepted + stats.rejected.iter().map(|(_, n)| n).sum::<u64>()
}

/// Commands call `n` as every joint's target, so the robot tells which
/// start it applies; can be held inside `predict` (`gate`).
#[derive(Clone, Default)]
struct Counter {
    calls: Arc<AtomicUsize>,
    gate: Arc<AtomicBool>,
    /// The call held at the gate.
    gated: Arc<Mutex<Option<usize>>>,
    steps: Arc<Mutex<Vec<Step>>>,
}

impl Predictor for Counter {
    type Error = String;

    fn predict(
        &mut self,
        observation: Observation,
        _: i64,
    ) -> Result<Vec<(String, Value)>, String> {
        if observation.value("q").is_none() {
            return Err("no q".into());
        }
        let n = self.calls.fetch_add(1, Ordering::Relaxed);
        if self.gate.load(Ordering::Relaxed) {
            *self.gated.lock().expect("gated") = Some(n);
            while self.gate.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        let target = f64::from(u32::try_from(n).expect("calls"));
        Ok(vec![(
            "arm".into(),
            Value::Tensor(Tensor::from_f64(DType::F32, vec![5, 6], &[target; 30])),
        )])
    }

    fn step(&mut self, step: &Step) -> Result<(), String> {
        self.steps.lock().expect("steps").push(step.clone());
        Ok(())
    }
}

fn run_in_thread(
    controller: &Arc<Controller>,
    model: &Counter,
    mode: RunMode,
) -> std::thread::JoinHandle<Result<(), RunError<String>>> {
    let (controller, mut model) = (controller.clone(), model.clone());
    std::thread::spawn(move || {
        controller.run(
            &mut model,
            RunConfig::default()
                .mode(mode)
                .duration(Duration::from_secs(60)),
        )
    })
}

async fn wait_until(what: &str, mut done: impl FnMut() -> bool) {
    let started = Instant::now();
    while !done() {
        assert!(started.elapsed() < WAIT, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// Every event drained so far, kept so a test can look twice.
#[derive(Default)]
struct Events(Vec<RunEvent>);

impl Events {
    fn drain(&mut self, controller: &Controller) -> &[RunEvent] {
        self.0.extend(controller.drain_events());
        &self.0
    }

    fn shadowing(&mut self, controller: &Controller) -> Vec<String> {
        self.drain(controller)
            .iter()
            .filter_map(|e| match e {
                RunEvent::Shadowing { message } => Some(message.clone()),
                _ => None,
            })
            .collect()
    }

    fn commanding(&mut self, controller: &Controller) -> usize {
        self.drain(controller)
            .iter()
            .filter(|e| matches!(e, RunEvent::Commanding { .. }))
            .count()
    }
}

/// Runs `model` in shadow for `n` more predictions than it has made, and
/// checks nothing reached the robot meanwhile.
async fn predicts_and_sends_nothing(
    controller: &Arc<Controller>,
    model: &Counter,
    robot: &Robot,
    n: usize,
) {
    wait_until("predictions in shadow", || {
        controller.stats().shadow_predictions >= n as u64
            && model.steps.lock().expect("steps").len() >= n
    })
    .await;
    let stats = controller.stats();
    assert_eq!(chunks_received(robot), 0, "the robot's edge got a chunk");
    assert_eq!(stats.chunks_sent, 0, "{stats:?}");
    assert_eq!(stats.send_errors, 0, "{stats:?}");
    assert!(stats.shadow_predictions >= n as u64, "{stats:?}");
    assert_eq!(stats.skipped(SkipReason::NoLease), 0, "{stats:?}");
    let steps = model.steps.lock().expect("steps");
    assert!(steps.len() >= n);
    for step in steps.iter() {
        assert!(step.shadow);
        let [action] = step.actions.as_slice() else {
            panic!("one action entry: {step:?}");
        };
        assert_eq!(action.seq, None);
        assert_eq!(action.part.as_deref(), Some("ur10e:arm"));
        // The actions as the edge would take them: the part's `pos` field.
        let Some(Value::Dict(fields)) = action.actions() else {
            panic!("converted actions: {action:?}");
        };
        let (name, pos) = &fields[0];
        assert_eq!((name.as_str(), pos.shape.as_slice()), ("pos", &[5, 6][..]));
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_shadow_run_on_a_disarmed_session_predicts_and_sends_nothing() {
    let world = World::start().await;
    let robot = world.robot("ur10e").await;
    let (stop, log): (_, Log) = (Arc::new(AtomicBool::new(false)), Arc::default());
    let robot_loop = spin(robot.clone(), stop.clone(), log.clone());
    let model = world.controller("model", PriorityClass::Model).await;
    let counter = Counter::default();
    let run = run_in_thread(&model, &counter, RunMode::Shadow);

    predicts_and_sends_nothing(&model, &counter, &robot, 50).await;
    let mut events = Events::default();
    let said = events.shadowing(&model);
    assert_eq!(said.len(), 1, "{said:?}");
    assert!(
        said[0].starts_with("shadowing: the session is DISARMED: this policy commands nothing"),
        "{said:?}"
    );
    assert!(!model.holds_any());
    assert!(log.lock().expect("log").is_empty());

    model.stop();
    run.join().expect("run").expect("a clean stop");
    stop.store(true, Ordering::Relaxed);
    robot_loop.join().expect("robot loop");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_shadow_run_sends_nothing_while_another_holder_has_its_lease() {
    let world = World::start().await;
    let robot = world.robot("ur10e").await;
    robot.participant().arm().await.expect("arm");
    let (stop, log): (_, Log) = (Arc::new(AtomicBool::new(false)), Arc::default());
    let robot_loop = spin(robot.clone(), stop.clone(), log.clone());
    // The operator holds the arm and sends nothing: every chunk the edge
    // takes in would be the model's.
    let operator = world.controller("operator", PriorityClass::Operator).await;
    operator.acquire().await.expect("operator lease");
    let model = world.controller("model", PriorityClass::Model).await;
    let counter = Counter::default();
    let run = run_in_thread(&model, &counter, RunMode::Shadow);

    predicts_and_sends_nothing(&model, &counter, &robot, 50).await;
    let said = Events::default().shadowing(&model);
    assert_eq!(said.len(), 1, "{said:?}");
    assert!(
        said[0].contains("no lease (no_lease): ur10e:arm is leased to operator"),
        "{said:?}"
    );
    assert!(operator.holds());

    model.stop();
    run.join().expect("run").expect("a clean stop");
    stop.store(true, Ordering::Relaxed);
    robot_loop.join().expect("robot loop");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_shadow_run_sends_nothing_while_its_robot_is_held() {
    let world = World::start().await;
    // Armed before the UR10e declares: it is held until re-armed by name.
    let other = world.robot("other").await;
    other.participant().arm().await.expect("arm");
    let robot = world.robot("ur10e").await;
    let (stop, log): (_, Log) = (Arc::new(AtomicBool::new(false)), Arc::default());
    let robot_loop = spin(robot.clone(), stop.clone(), log.clone());
    let model = world.controller("model", PriorityClass::Model).await;
    assert!(!model.held_refusals().is_empty(), "the UR10e is held");
    let counter = Counter::default();
    let run = run_in_thread(&model, &counter, RunMode::Shadow);

    predicts_and_sends_nothing(&model, &counter, &robot, 50).await;
    let said = Events::default().shadowing(&model);
    assert_eq!(said.len(), 1, "{said:?}");
    assert!(said[0].contains(r#"robot.rearm("ur10e")"#), "{said:?}");

    model.stop();
    run.join().expect("run").expect("a clean stop");
    stop.store(true, Ordering::Relaxed);
    robot_loop.join().expect("robot loop");
}

/// The start in flight when the lease arrives was read without it: it stays
/// a shadow start, and the robot first applies a later one.
#[tokio::test(flavor = "multi_thread")]
async fn a_shadow_run_commands_from_the_first_start_read_with_its_lease() {
    let world = World::start().await;
    let robot = world.robot("ur10e").await;
    let (stop, log): (_, Log) = (Arc::new(AtomicBool::new(false)), Arc::default());
    let robot_loop = spin(robot.clone(), stop.clone(), log.clone());
    let model = world.controller("model", PriorityClass::Model).await;
    let counter = Counter::default();
    let run = run_in_thread(&model, &counter, RunMode::Shadow);
    wait_until("a few shadow predictions", || {
        counter.calls.load(Ordering::Relaxed) >= 5
    })
    .await;

    // Hold a prediction, arm and take the lease meanwhile, then let it go.
    counter.gate.store(true, Ordering::Relaxed);
    wait_until("a prediction held", || {
        counter.gated.lock().expect("gated").is_some()
    })
    .await;
    let gated = counter.gated.lock().expect("gated").expect("gated");
    robot.participant().arm().await.expect("arm");
    wait_until("the lease", || model.holds()).await;
    counter.gate.store(false, Ordering::Relaxed);
    wait_until("the robot driven", || log.lock().expect("log").len() >= 20).await;

    let steps = model_steps(&counter);
    let gated_step = &steps[gated];
    assert!(
        gated_step.shadow,
        "the start read before the lease was sent"
    );
    assert_eq!(gated_step.actions[0].seq, None);
    let first = steps
        .iter()
        .position(|s| !s.shadow)
        .expect("a commanded start");
    assert!(first > gated);
    assert!(steps[first..].iter().all(|s| !s.shadow));
    assert!(steps[first].actions[0].seq.is_some());
    // The robot applies nothing predicted in shadow.
    let applied = log.lock().expect("log").clone();
    let first_value = f32::from(u16::try_from(first).expect("start"));
    assert!(
        applied.iter().all(|v| *v >= first_value),
        "applied {applied:?}, first commanded start {first}"
    );
    let stats = model.stats();
    assert_eq!(stats.shadow_predictions, first as u64, "{stats:?}");
    assert!(stats.chunks_sent > 0);
    assert_eq!(stats.skipped(SkipReason::NoLease), 0, "{stats:?}");
    let mut events = Events::default();
    assert_eq!(events.shadowing(&model).len(), 1);
    assert_eq!(events.commanding(&model), 1);
    let said = events.drain(&model);
    let commanding = said
        .iter()
        .position(|e| matches!(e, RunEvent::Commanding { .. }))
        .expect("commanding");
    assert!(
        said[..commanding]
            .iter()
            .any(|e| matches!(e, RunEvent::Shadowing { .. }))
    );
    assert!(
        said[commanding].to_string().contains("ur10e:arm"),
        "{said:?}"
    );
    assert!(!run.is_finished(), "the same run commands: no restart");

    model.stop();
    run.join().expect("run").expect("a clean stop");
    stop.store(true, Ordering::Relaxed);
    robot_loop.join().expect("robot loop");
}

fn model_steps(counter: &Counter) -> Vec<Step> {
    counter.steps.lock().expect("steps").clone()
}

/// What an operator taking the robot and letting go of it does to a model,
/// in each mode: released, it hands back at once; its process gone, its
/// lease lapses and the model takes the robot back.
async fn operator_hand_back(mode: RunMode) {
    let world = World::start().await;
    let robot = world.robot("ur10e").await;
    robot.participant().arm().await.expect("arm");
    let (stop, log): (_, Log) = (Arc::new(AtomicBool::new(false)), Arc::default());
    let robot_loop = spin(robot.clone(), stop.clone(), log.clone());
    let model = world.controller("model", PriorityClass::Model).await;
    let counter = Counter::default();
    let run = run_in_thread(&model, &counter, mode);
    wait_until("the model driving", || {
        model.holds() && model.stats().chunks_sent > 5
    })
    .await;

    // An operator takes the arm and lets go.
    let operator = world.controller("operator", PriorityClass::Operator).await;
    operator.acquire().await.expect("operator preempts");
    wait_until("the model preempted", || !model.holds()).await;
    // A start already past its lease check finishes as it began.
    tokio::time::sleep(Duration::from_millis(50)).await;
    let before = (model.stats(), counter.calls.load(Ordering::Relaxed));
    tokio::time::sleep(Duration::from_millis(300)).await;
    let during = (model.stats(), counter.calls.load(Ordering::Relaxed));
    match mode {
        // Preempted, the model's starts are skipped: predict does not run.
        RunMode::Command => {
            assert!(during.0.skipped(SkipReason::NoLease) > before.0.skipped(SkipReason::NoLease));
            assert_eq!(during.1, before.1, "predict ran without a lease");
        }
        RunMode::Shadow => {
            assert!(during.0.shadow_predictions > before.0.shadow_predictions);
            assert_eq!(during.0.skipped(SkipReason::NoLease), 0);
        }
    }
    assert_eq!(during.0.chunks_sent, before.0.chunks_sent);
    let released = Instant::now();
    operator.release().await.expect("release");
    wait_until("the model back after a release", || model.holds()).await;
    let back_after_release = released.elapsed();
    let sent = model.stats().chunks_sent;
    wait_until("the model sending again", || {
        model.stats().chunks_sent > sent
    })
    .await;

    // The operator takes the arm again and its process dies.
    operator.acquire().await.expect("operator preempts");
    wait_until("the model preempted", || !model.holds()).await;
    let mut host_events = world.host.events();
    // Its heartbeats stop and nothing of it runs any more: nothing released.
    operator.participant().set_heartbeats_enabled(false);
    drop(operator);
    let died = Instant::now();
    wait_until("the model back after the operator died", || model.holds()).await;
    let back_after_death = died.elapsed();
    let sent = model.stats().chunks_sent;
    wait_until("the model sending again", || {
        model.stats().chunks_sent > sent
    })
    .await;
    // The host ended the dead operator's lease by its lapse and held no
    // robot: the model took the arm back on its own.
    let mut told = Vec::new();
    while let Ok(event) = host_events.try_recv() {
        if let AuthorityEvent::RobotHeld { robot, .. } = &event {
            panic!("the operator's loss held {robot}");
        }
        told.push(event.to_string());
    }
    assert!(
        told.iter()
            .any(|e| e.contains("of operator") && e.contains("ended (HEARTBEAT_LAPSE")),
        "{told:#?}"
    );
    assert!(
        told.iter().any(|e| e.contains("granted to model")),
        "{told:#?}"
    );
    // Released, the arm is back at the next authority view; dead, only once
    // the operator's lease lapsed (300 ms after its last heartbeat).
    assert!(
        back_after_release < back_after_death,
        "{back_after_release:?}"
    );
    assert!(
        back_after_death >= Duration::from_millis(250),
        "{back_after_death:?}"
    );
    let mut events = Events::default();
    if mode == RunMode::Shadow {
        let said = events.shadowing(&model);
        assert!(
            said.iter().any(|m| m.contains("leased to operator")),
            "{said:?}"
        );
        // Commanding, then back after each preemption.
        assert!(events.commanding(&model) >= 3, "{:#?}", events.0);
    } else {
        assert!(events.shadowing(&model).is_empty());
        assert_eq!(events.commanding(&model), 0);
    }

    model.stop();
    run.join().expect("run").expect("a clean stop");
    stop.store(true, Ordering::Relaxed);
    robot_loop.join().expect("robot loop");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_commanding_model_takes_the_robot_back_when_an_operator_releases_or_dies() {
    operator_hand_back(RunMode::Command).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_shadow_model_predicts_while_preempted_and_takes_the_robot_back() {
    operator_hand_back(RunMode::Shadow).await;
}
