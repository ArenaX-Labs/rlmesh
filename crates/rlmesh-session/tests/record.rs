#![cfg(feature = "record")]
//! A recording of a live session: a robot, a model replanning, an operator
//! preempting it and handing back, all read back from the MCAP file.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use rlmesh_session::edge::{EdgeConfig, Robot};
use rlmesh_session::record::{
    Episode, Recorded, Recorder, Rolling, collect, decode_samples, keep, read, summarize,
};
use rlmesh_session::spec::apply::{Tensor, Value};
use rlmesh_session::spec::{Policy, RunOptions};
use rlmesh_session::transport::open_local_listeners;
use rlmesh_session::wire::{DType, PriorityClass, Role, SessionId, TenantId};
use rlmesh_session::{
    Controller, ControllerConfig, HostConfig, Observation, Participant, ParticipantConfig,
    Predictor, SessionHost, SessionServer, SystemMonotonicClock, Transport, TransportConfig,
};

const WAIT: Duration = Duration::from_secs(10);

const ROBOT: &str = r#"{
    "arm": {"kind": "joints", "count": 6, "fields": ["pos"], "hz": 500,
            "command": "position", "hold_max_ms": 100, "max_action_age_ms": 1000},
    "wrist/ft": {"kind": "wrench", "frame": "tool0", "hz": 500}
}"#;

const POLICY: &str = r#"{
    "hz": 50,
    "inputs": {"q": {"kind": "joints", "at": "arm", "fields": ["pos"]}},
    "actions": {"arm": {"kind": "joints", "at": "arm", "command": "position",
                        "hz": 100, "horizon": 5}}
}"#;

struct World {
    base: std::net::SocketAddr,
    tenant: TenantId,
    session: SessionId,
    /// `None` when a recording server hosts the session.
    _host: Option<SessionHost>,
}

impl World {
    async fn start() -> Self {
        let (transport, base) = open_local_listeners("127.0.0.1".parse().expect("ip"))
            .await
            .expect("transport");
        let tenant = TenantId::new("record").expect("tenant");
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
            _host: Some(host),
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
        Participant::attach(transport, config)
            .await
            .expect("attach")
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

struct Constant(f64);

impl Predictor for Constant {
    type Error = String;

    fn predict(&mut self, _: Observation, _: i64) -> Result<Vec<(String, Value)>, String> {
        Ok(vec![(
            "arm".into(),
            Value::Tensor(Tensor::from_f64(DType::F32, vec![5, 6], &[self.0; 30])),
        )])
    }
}

async fn wait_until_async<F: Future<Output = bool>>(what: &str, mut done: impl FnMut() -> F) {
    let started = Instant::now();
    while !done().await {
        assert!(started.elapsed() < WAIT, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

async fn wait_until(what: &str, mut done: impl FnMut() -> bool) {
    let started = Instant::now();
    while !done() {
        assert!(started.elapsed() < WAIT, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_session_records_to_mcap_and_reads_back_with_its_lease_timeline() {
    let world = World::start().await;
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("session.mcap");
    let observer = world
        .join("recorder", Role::Observer, PriorityClass::Scripted)
        .await;
    let recorder = Recorder::start(Arc::new(observer), &path)
        .await
        .expect("recorder");

    let robot = Robot::declare(
        world
            .join("ur10e", Role::Embodiment, PriorityClass::Scripted)
            .await,
        ROBOT,
        EdgeConfig::default(),
    )
    .await
    .expect("declare");
    robot.participant().arm().await.expect("arm");
    let robot = Arc::new(robot);
    let stop = Arc::new(AtomicBool::new(false));
    let spin = {
        let (robot, stop) = (robot.clone(), stop.clone());
        std::thread::spawn(move || {
            let clock = robot.participant().clock().clone();
            while !stop.load(Ordering::Relaxed) {
                let now = clock.now_ns();
                robot.publish("arm", now, &[&f32s(&[0.5; 6])]).expect("arm");
                robot
                    .publish("wrist/ft", now, &[&f32s(&[0.0; 3]), &f32s(&[0.0; 3])])
                    .expect("ft");
                robot.due("arm").expect("due");
                std::thread::sleep(Duration::from_millis(2));
            }
        })
    };

    let model = world.controller("model", PriorityClass::Model).await;
    let operator = world.controller("operator", PriorityClass::Operator).await;
    let run = {
        let model = model.clone();
        std::thread::spawn(move || model.run(&mut Constant(1.0), Some(Duration::from_secs(30))))
    };
    wait_until("the model holding the arm", || model.holds()).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    operator.acquire().await.expect("operator preempts");
    wait_until("the model preempted", || !model.holds()).await;
    for _ in 0..10 {
        if let Ok((observation, at)) = operator.observe() {
            let value = Value::Tensor(Tensor::from_f64(DType::F32, vec![5, 6], &[2.0; 30]));
            operator
                .send(&[("arm".into(), value)], &observation.meta.based_on, at)
                .expect("send");
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    operator.release().await.expect("release");
    wait_until("the model back", || model.holds()).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    model.stop();
    run.join().expect("run thread").expect("clean stop");
    stop.store(true, Ordering::Relaxed);
    spin.join().expect("robot loop");

    let stats = recorder.finish().await.expect("finish");
    assert!(stats.messages > 100, "{stats:?}");
    let summary = summarize(&path).expect("summary");
    let topic = |name: &str| {
        summary
            .topics
            .iter()
            .find(|t| t.topic == name)
            .unwrap_or_else(|| panic!("no topic {name} in {:#?}", summary.topics))
    };
    for name in [
        "/ur10e/arm",
        "/ur10e/wrist/ft",
        "/ur10e/arm/_applied",
        "/ur10e/arm/_chunks",
        "/_session/authority",
        "/_session/bindings",
    ] {
        let t = topic(name);
        assert!(t.messages > 0, "{t:?}");
        assert_eq!(t.undecodable, 0, "{t:?}");
        assert!(t.log_time_monotonic && t.publish_time_monotonic, "{t:?}");
    }
    assert_eq!(summary.messages, stats.messages);
    assert_eq!(
        topic("/ur10e/arm/_chunks").schema,
        "rlmesh.session.v2.ActionChunk"
    );
    assert_eq!(
        topic("/_session/bindings").messages,
        2,
        "model and operator"
    );
    assert!(summary.metadata.iter().any(|m| m == "rlmesh.robot.ur10e"));

    // model, then the operator (preemption), then the model under a fresh token.
    let arm: Vec<_> = summary
        .leases
        .iter()
        .filter(|l| l.part == "ur10e/arm" && l.holder.is_some())
        .collect();
    let holders: Vec<&str> = arm.iter().filter_map(|l| l.holder.as_deref()).collect();
    let first_operator = holders
        .iter()
        .position(|h| *h == "operator")
        .expect("an operator lease");
    assert_eq!(holders[0], "model");
    assert_eq!(holders.last().copied(), Some("model"));
    assert!(arm.last().expect("last").token > arm[first_operator].token);

    // Read back for replay: everything but the binding records, in log-time order.
    let messages = read(&path).expect("read");
    assert_eq!(
        messages.len() as u64,
        summary.messages - topic("/_session/bindings").messages
    );
    assert!(messages.windows(2).all(|w| w[0].log_ns <= w[1].log_ns));
    assert!(matches!(messages[0].message, Recorded::Authority(_)));
    assert!(
        messages
            .iter()
            .any(|m| matches!(m.message, Recorded::Chunk(_)))
    );
}

/// The segment files of one session's rolling recording, oldest first.
fn segments(root: &std::path::Path, session: SessionId) -> Vec<std::path::PathBuf> {
    files_in(&root.join("rolling").join(session.to_string()))
}

fn files_in(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .map(|d| d.map(|e| e.expect("entry").path()).collect())
        .unwrap_or_default();
    files.sort();
    files
}

/// A robot publishing `ft` every 10 ms for `ms`.
async fn publish_ft(robot: &Participant, ms: u64) {
    let ft = robot.publisher("ft").await.expect("publisher");
    for _ in 0..ms / 10 {
        ft.publish(robot.clock().now_ns(), bytes::Bytes::from(vec![0_u8; 32]))
            .expect("publish");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

const FT: &str = r#"{"ft": {"kind": "wrench", "frame": "tool0", "hz": 100}}"#;

#[tokio::test(flavor = "multi_thread")]
async fn a_recording_server_rolls_self_contained_segments_and_still_ends_sessions() {
    let (transport, base) = open_local_listeners("127.0.0.1".parse().expect("ip"))
        .await
        .expect("transport");
    let tenant = TenantId::new("rolling").expect("tenant");
    let mut config = HostConfig::new(tenant.clone(), SessionId::new());
    config.clock = SystemMonotonicClock.shared();
    config.health_log = None;
    let dir = tempfile::tempdir().expect("tempdir");
    let mut rolling = Rolling::new(dir.path());
    rolling.segment = Duration::from_millis(150);
    rolling.min_free_bytes = 0;
    let server = SessionServer::start_recording(transport, config, rolling)
        .await
        .expect("server");

    // A session the server starts for its first participant.
    let session = SessionId::new();
    let world = World {
        base,
        tenant,
        session,
        _host: None,
    };
    let robot = world
        .join("robot", Role::Embodiment, PriorityClass::Scripted)
        .await;
    robot.declare_parts(FT).await.expect("declare");
    publish_ft(&robot, 700).await;
    robot.detach().await.expect("detach");
    wait_until_async("the session to stop", || async {
        !server.sessions().await.contains(&session)
    })
    .await;

    let files = segments(dir.path(), session);
    assert!(files.len() >= 3, "{files:?}");
    for file in &files {
        let messages = read(file).expect("a complete segment");
        assert!(
            matches!(messages[0].message, Recorded::Authority(_)),
            "{file:?} opens with the authority snapshot, not {:?}",
            messages[0].message
        );
    }
    server.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn rolling_capture_keeps_its_budget_incidents_and_episodes_and_pauses_below_the_free_space_floor()
 {
    let world = World::start().await;
    let dir = tempfile::tempdir().expect("tempdir");
    let mut rolling = Rolling::new(dir.path());
    rolling.segment = Duration::from_millis(100);
    rolling.max_bytes = 1;
    rolling.min_free_bytes = 0;
    let observer = world
        .join("recorder", Role::Observer, PriorityClass::Scripted)
        .await;
    let recorder = Recorder::start_rolling(Arc::new(observer), rolling.clone(), world.session)
        .await
        .expect("recorder");
    let robot = world
        .join("robot", Role::Embodiment, PriorityClass::Scripted)
        .await;
    robot.declare_parts(FT).await.expect("declare");
    let started = std::time::SystemTime::now();
    publish_ft(&robot, 500).await;
    let kept = keep(&rolling, world.session, "keep").expect("kept on request");
    let episode = Episode {
        task: "hold".into(),
        start_ns: 0,
        end_ns: 1,
        success: Some(true),
    };
    let collected =
        collect(&rolling, world.session, "test set", &episode, started).expect("collected");
    // Collecting it again keeps the same links, never copying over them.
    collect(&rolling, world.session, "test set", &episode, started).expect("again");
    robot.estop("test").await.expect("estop");
    publish_ft(&robot, 500).await;
    let stats = recorder.finish().await.expect("finish");
    assert!(stats.segments >= 3, "{stats:?}");
    // The purge spares the open segment and the one just closed (another
    // session's could still be open).
    let rolling_kept = segments(dir.path(), world.session);
    assert!(
        rolling_kept.len() <= 2 && rolling_kept.last() == Some(&stats.path),
        "{rolling_kept:?}"
    );

    // Both incidents outlive the purge, the e-stop's with the segments
    // written after it, heartbeats included.
    let incidents = files_in(&dir.path().join("incidents").join(world.session.to_string()));
    assert_eq!(incidents.len(), 2, "{incidents:?}");
    assert!(incidents.contains(&kept));
    let estop = incidents
        .iter()
        .find(|d| d.to_string_lossy().ends_with("-estop"))
        .expect("the e-stop kept itself");
    let files = files_in(estop);
    assert!(files.len() >= 3, "{files:?}");
    let messages: Vec<_> = files.iter().flat_map(|f| read(f).expect("read")).collect();
    assert!(
        messages
            .iter()
            .any(|m| matches!(m.message, Recorded::Heartbeat(_)))
    );

    // The episode keeps its segments, decoded into the declared fields.
    assert!(collected.starts_with(dir.path().join("collected").join("test_set")));
    let json = std::fs::read_to_string(collected.join("episode.json")).expect("episode.json");
    assert!(json.contains(r#""task": "hold""#), "{json}");
    let samples = decode_samples(&read(&collected).expect("read the episode"));
    let ft = samples
        .iter()
        .find(|s| s.topic == "robot/ft")
        .expect("ft samples");
    let fields: Vec<_> = ft
        .fields
        .iter()
        .map(|f| (f.name.as_str(), f.bytes.len()))
        .collect();
    assert_eq!(fields, [("force", 12), ("torque", 12)]);

    // A disk below the floor gets nothing written.
    let full = tempfile::tempdir().expect("tempdir");
    let mut rolling = Rolling::new(full.path());
    rolling.min_free_bytes = u64::MAX;
    let observer = world
        .join("recorder-2", Role::Observer, PriorityClass::Scripted)
        .await;
    let recorder = Recorder::start_rolling(Arc::new(observer), rolling, world.session)
        .await
        .expect("recorder");
    publish_ft(&robot, 200).await;
    let stats = recorder.finish().await.expect("finish");
    assert_eq!((stats.segments, stats.messages), (0, 0), "{stats:?}");
    assert!(stats.paused_drops > 0, "{stats:?}");
    assert!(segments(full.path(), world.session).is_empty());
}
