//! The session's live link health, read from a participant
//! ([`Participant::health`]): any participant, an observer that neither
//! publishes nor commands too, reads every link the host measures, each
//! edge's refusals by reason and each participant's latest report.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use rlmesh_session::edge::{EdgeConfig, Rejection, Robot};
use rlmesh_session::spec::apply::{Tensor, Value};
use rlmesh_session::spec::{Policy, RunOptions};
use rlmesh_session::transport::open_local_listeners;
use rlmesh_session::wire::{DType, ParticipantId, PriorityClass, Role, SessionId, TenantId};
use rlmesh_session::{
    Controller, ControllerConfig, HostConfig, Observation, Participant, ParticipantConfig,
    Predictor, RunConfig, SessionError, SessionHost, SystemMonotonicClock, Transport,
    TransportConfig, sdk,
};

const WAIT: Duration = Duration::from_secs(10);

/// An eye: two joints at 500 Hz.
const EYE: &str = r#"{
    "eyes": {"kind": "joints", "names": ["pan", "tilt"], "fields": ["pos"], "hz": 500,
             "command": "position", "safe": "hold", "hold_max_ms": 100}
}"#;

/// Commands both eyes.
const TRACKER: &str = r#"{
    "hz": 50,
    "actions": {
        "one": {"kind": "joints", "at": "eye-1:eyes", "command": "position", "hz": 100,
                "chunk": 5},
        "two": {"kind": "joints", "at": "eye-2:eyes", "command": "position", "hz": 100,
                "chunk": 5}
    }
}"#;

/// Reads eye 1 and commands nothing.
const VIEWER: &str = r#"{
    "hz": 25,
    "inputs": {"gaze": {"kind": "joints", "at": "eye-1:eyes", "fields": ["pos"]}}
}"#;

/// Commands eye 2 alone.
const PROBE: &str = r#"{
    "hz": 25,
    "actions": {"two": {"kind": "joints", "at": "eye-2:eyes", "command": "position",
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
        let (transport, base) = open_local_listeners("127.0.0.1".parse().expect("ip"))
            .await
            .expect("transport");
        let tenant = TenantId::new("health").expect("tenant");
        let session = SessionId::new();
        let mut config = HostConfig::new(tenant.clone(), session);
        config.clock = SystemMonotonicClock.shared();
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
            ParticipantId::new(id).expect("id"),
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

    async fn eye(&self, name: &str) -> Arc<Robot> {
        let p = self
            .join(name, Role::Embodiment, PriorityClass::Scripted)
            .await;
        Arc::new(
            Robot::declare(p, EYE, EdgeConfig::default())
                .await
                .expect("declare"),
        )
    }

    async fn policy(&self, name: &str, policy: &str, priority: PriorityClass) -> Arc<Controller> {
        let p = self.join(name, Role::Controller, priority).await;
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
}

fn f32s(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// An eye's 500 Hz loop: publish its gaze, then take what is due.
fn spin(robot: Arc<Robot>, stop: Arc<AtomicBool>) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let clock = robot.participant().clock().clone();
        while !stop.load(Ordering::Relaxed) {
            let _ = robot.publish("eyes", clock.now_ns(), &[&f32s(&[0.0, 0.0])]);
            let _ = robot.due("eyes");
            std::thread::sleep(Duration::from_millis(2));
        }
    })
}

/// Both eyes to 0.25.
struct Gaze;

impl Predictor for Gaze {
    type Error = String;

    fn predict(&mut self, _: Observation, _: i64) -> Result<Vec<(String, Value)>, String> {
        let gaze = || Value::Tensor(Tensor::from_f64(DType::F32, vec![5, 2], &[0.25; 10]));
        Ok(vec![("one".into(), gaze()), ("two".into(), gaze())])
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
async fn an_observer_reads_every_link_and_each_robots_refusals_by_reason() {
    let world = World::start().await;
    let (one, two) = (world.eye("eye-1").await, world.eye("eye-2").await);
    one.participant().arm().await.expect("arm");
    let stop = Arc::new(AtomicBool::new(false));
    let loops = [
        spin(one.clone(), stop.clone()),
        spin(two.clone(), stop.clone()),
    ];
    let tracker = world.policy("tracker", TRACKER, PriorityClass::Model).await;
    let viewer = world.policy("viewer", VIEWER, PriorityClass::Model).await;
    let run = {
        let tracker = tracker.clone();
        std::thread::spawn(move || {
            tracker.run(
                &mut Gaze,
                RunConfig::default().duration(Duration::from_secs(60)),
            )
        })
    };
    wait_until("the tracker sending", || tracker.stats().chunks_sent > 10).await;

    // A prober takes eye 2 and sends it a value that is not a number: the
    // edge refuses it for that.
    let prober = world.policy("prober", PROBE, PriorityClass::Operator).await;
    prober.acquire().await.expect("the prober takes eye 2");
    let nan = Value::Tensor(Tensor::from_f64(DType::F32, vec![5, 2], &[f64::NAN; 10]));
    let refused = |robot: &Robot| {
        robot
            .edge_stats("eyes")
            .expect("edge stats")
            .rejected(Rejection::InvalidValue)
    };
    while refused(&two) == 0 {
        prober
            .send(
                &[("two".into(), nan.clone())],
                &[],
                prober.participant().session_now_ns(),
            )
            .expect("send");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // The viewer neither publishes nor commands. Its first read subscribes
    // and waits for a publication; it reads the robots' refusals once their
    // heartbeats carried them to the host.
    let viewer_id = viewer.participant().id().clone();
    let mut health = viewer.health(WAIT).await.expect("health");
    let two_id = ParticipantId::new("eye-2").expect("id");
    let started = Instant::now();
    while health.link(&two_id).is_none_or(|l| {
        !l.rejected
            .iter()
            .any(|(r, n)| *r == Rejection::InvalidValue && *n > 0)
    }) {
        assert!(started.elapsed() < WAIT, "{health:#?}");
        health = viewer
            .participant()
            .next_health(WAIT)
            .await
            .expect("the next health");
    }
    let links = sdk::health_links(viewer.participant(), &health);
    let link = |name: &str| {
        links
            .iter()
            .find(|l| l.participant == name)
            .unwrap_or_else(|| panic!("no link of {name} in {links:#?}"))
    };
    for name in ["eye-1", "eye-2", "tracker", "viewer", "prober"] {
        let l = link(name);
        assert!(l.heartbeats > 0, "{l:#?}");
        assert!(l.rtt_ms.is_some_and(|rtt| rtt > 0.0), "{l:#?}");
        assert!(l.offset_ms.is_some(), "{l:#?}");
        assert_eq!(l.health, "ok", "{l:#?}");
    }
    // Per robot: eye 2 refused for the value, eye 1 refused nothing.
    let invalid = link("eye-2")
        .rejected_by
        .iter()
        .find(|(reason, _)| reason == "invalid_value")
        .map(|(_, n)| *n);
    assert!(invalid.is_some_and(|n| n >= 1), "{:#?}", link("eye-2"));
    assert!(link("eye-1").rejected_by.is_empty(), "{:#?}", link("eye-1"));
    assert!(
        link("eye-1").last_chunk_accepted > 0,
        "{:#?}",
        link("eye-1")
    );
    // Reports: what each eye publishes, what the tracker's run did, what
    // the viewer received.
    let published = &link("eye-1").published;
    assert!(
        published
            .iter()
            .any(|(part, n, _)| part == "eye-1:eyes" && *n > 0),
        "{published:?}"
    );
    let run_counters = link("tracker").run.clone().expect("the tracker's run");
    assert!(run_counters.starts > 0 && run_counters.chunks_sent > 0);
    assert!(
        link("viewer")
            .last_seq_seen
            .iter()
            .any(|(part, seq)| part == "eye-1:eyes" && *seq > 0),
        "{:#?}",
        link("viewer")
    );
    assert!(health.link(&viewer_id).is_some());

    // Later reads return the newest at once; the next one comes about a
    // second later.
    let now = viewer.health(WAIT).await.expect("health");
    let next = viewer
        .participant()
        .next_health(WAIT)
        .await
        .expect("the next");
    assert!(next.host_ns > now.host_ns);
    match viewer
        .participant()
        .next_health(Duration::from_millis(1))
        .await
    {
        Err(SessionError::Timeout(_, what)) => assert!(what.contains("health"), "{what}"),
        other => panic!("{other:?}"),
    }

    tracker.stop();
    run.join().expect("run thread").expect("a clean stop");
    stop.store(true, Ordering::Relaxed);
    for spin in loops {
        spin.join().expect("eye loop");
    }
}
