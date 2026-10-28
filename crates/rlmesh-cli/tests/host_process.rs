//! The session host in its own OS process (`rlmesh host`), with
//! participants in this one, linked over per-class TCP on localhost.

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use bytes::Bytes;
use rlmesh_session::SessionAddress;
use rlmesh_session::address::{DEFAULT_SESSION, TENANT, session_id};
use rlmesh_session::media::synthetic_frame;
use rlmesh_session::record::{Recorded, read};
use rlmesh_session::spec::apply::Value;
use rlmesh_session::spec::{Policy, RunOptions};
use rlmesh_session::transport::free_local_base;
use rlmesh_session::wire::version::SPEC_VERSION;
use rlmesh_session::wire::{
    ContractMode, CtlStatus, Fingerprint, Mode, ParticipantId, PriorityClass, Role, SessionId,
    TenantId,
};
use rlmesh_session::{
    Inputs, ObserveError, Participant, ParticipantConfig, SessionError, SubscribeOptions,
    Transport, TransportConfig,
};

/// Participants listen too (the OS picks the ports): in peer mode, data
/// flows peer to peer once gossip tells peers where to connect.
fn any_local_port() -> std::net::SocketAddr {
    "127.0.0.1:0".parse().expect("addr")
}

const WAIT: Duration = Duration::from_secs(10);

/// The host process; killed on drop whatever the test did.
struct HostProcess {
    child: Child,
    session: SessionId,
    base: std::net::SocketAddr,
    /// Where it records its sessions.
    recordings: tempfile::TempDir,
    /// The address it printed, with its key.
    address: SessionAddress,
}

impl HostProcess {
    /// Spawns the host on free local ports, again on others when one was
    /// taken in between (the host then exits before printing its address).
    fn spawn() -> Self {
        for _ in 0..16 {
            if let Some(host) = Self::try_spawn() {
                return host;
            }
        }
        panic!("the host process never found free ports");
    }

    fn try_spawn() -> Option<Self> {
        let base = free_local_base("127.0.0.1".parse().expect("ip")).expect("ports");
        let recordings = tempfile::tempdir().expect("tempdir");
        let mut child = Command::new(env!("CARGO_BIN_EXE_rlmesh"))
            .args(["host", "--listen", &base.to_string(), "--record-dir"])
            .arg(recordings.path())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn the host");
        let stdout = child.stdout.take().expect("stdout");
        let mut line = String::new();
        BufReader::new(stdout)
            .read_line(&mut line)
            .expect("address line");
        if let Ok(address) = line.trim().parse::<SessionAddress>()
            && address.base == base
            && address.key.is_some()
        {
            return Some(Self {
                child,
                session: session_id(DEFAULT_SESSION),
                base,
                recordings,
                address,
            });
        }
        let _ = child.kill();
        let _ = child.wait();
        None
    }

    fn signal(&self, signal: i32) {
        let pid = i32::try_from(self.child.id()).expect("pid");
        let status = Command::new("kill")
            .args([format!("-{signal}"), pid.to_string()])
            .status()
            .expect("kill");
        assert!(status.success());
    }

    fn wait_exit(&mut self) -> std::process::ExitStatus {
        let started = Instant::now();
        loop {
            if let Some(status) = self.child.try_wait().expect("wait") {
                return status;
            }
            assert!(started.elapsed() < WAIT, "the host did not exit");
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

impl Drop for HostProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

async fn join(
    host: &HostProcess,
    id: &str,
    roles: Vec<Role>,
    priority: PriorityClass,
) -> Participant {
    let transport = Transport::open(
        &TransportConfig::per_class_tcp(Some(any_local_port()), Some(host.base)).expect("config"),
    )
    .await
    .expect("transport");
    let mut config = ParticipantConfig::new(
        TenantId::new(TENANT).expect("tenant"),
        host.session,
        ParticipantId::new(id).expect("id"),
        roles,
        priority,
    );
    config.heartbeat_period = Duration::from_millis(20);
    config.credential = Some(host.address.credential().expect("the printed key"));
    Participant::attach(transport, config)
        .await
        .expect("attach")
}

const ROBOT: &str = r#"{
    "cmd": {"kind": "joints", "hz": 1000, "count": 6, "fields": ["pos"],
            "command": "position", "priority": "control"},
    "ft": {"kind": "wrench", "frame": "tool0", "hz": 1000}
}"#;

#[tokio::test(flavor = "multi_thread")]
async fn a_host_process_arbitrates_leases_and_its_loss_is_seen() {
    let mut host = HostProcess::spawn();
    let robot = join(
        &host,
        "robot",
        vec![Role::Embodiment],
        PriorityClass::Scripted,
    )
    .await;
    let view = robot.declare_parts(ROBOT).await.expect("declare");
    let robot_id = robot.id().clone();
    let cmd = view.part_at(&robot_id, "cmd").expect("cmd").channel;
    let policy = join(
        &host,
        "policy",
        vec![Role::Controller],
        PriorityClass::Model,
    )
    .await;
    let operator = join(
        &host,
        "operator",
        vec![Role::Controller],
        PriorityClass::Operator,
    )
    .await;

    // Bindings, verbs, leases and preemption are decided in the other
    // process.
    for (p, fp) in [(&policy, 1), (&operator, 2)] {
        p.wait_for("the robot's parts", WAIT, |s| s.part(cmd).is_some())
            .await
            .expect("parts");
        p.declare_binding(Fingerprint::from(fp), "{}", vec![cmd], 0.0)
            .await
            .expect("bind");
    }
    let armed = operator.arm().await.expect("arm");
    policy
        .wait_for("armed", WAIT, |s| s.epoch == armed.epoch)
        .await
        .expect("view");
    let model = policy.grant_lease(cmd, None).await.expect("model lease");
    let human = operator.grant_lease(cmd, None).await.expect("preempt");
    assert!(human.token > model.token);
    assert_eq!(human.binding, Fingerprint::from(2));
    let view = robot
        .wait_for("the operator's lease", WAIT, |s| {
            s.mode == Mode::Active && s.lease(cmd).is_some_and(|l| l.token == human.token)
        })
        .await
        .expect("view");
    assert!(view.check_fence(cmd, model.token, view.epoch).is_err());

    // Heartbeats cross the process boundary both ways: clock mappings.
    for p in [&robot, &policy, &operator] {
        let snapshot = p
            .wait_for("a clock mapping", WAIT, |s| s.clock(p.id()).is_some())
            .await
            .expect("mapping");
        let mapping = snapshot.clock(p.id()).expect("mapping");
        assert!(mapping.error_ns < 50_000_000, "{mapping:?}");
    }

    // Samples flow peer to peer and the host's history store, in the other
    // process, serves a late subscriber.
    let ft = robot.publisher("ft").await.expect("publisher");
    tokio::time::sleep(Duration::from_millis(300)).await;
    for i in 0..3_u8 {
        ft.publish(robot.clock().now_ns(), Bytes::from(vec![i; 32]))
            .expect("publish");
    }
    tokio::time::sleep(Duration::from_millis(200)).await;
    let sub = policy
        .subscribe(
            &robot_id,
            "ft",
            SubscribeOptions {
                window: 1,
                warm_from_history: true,
                ..SubscribeOptions::default()
            },
        )
        .await
        .expect("subscribe");
    assert_eq!(sub.stats().from_history, 1, "a window of one");
    assert_eq!(sub.stats().last_seq, 3);

    // A crashed host: participants see its liveliness go, and verbs fail
    // instead of hanging.
    host.signal(9);
    let mut present = robot.host_present();
    tokio::time::timeout(WAIT, present.wait_for(|p| !*p))
        .await
        .expect("host loss seen in time")
        .expect("watch");
    let err = operator.arm().await.expect_err("no host");
    assert!(
        matches!(err, SessionError::NoReply(_) | SessionError::Zenoh(_)),
        "{err}"
    );
    assert!(!host.wait_exit().success());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_terminated_host_shuts_down_cleanly_with_its_session_recorded() {
    let mut host = HostProcess::spawn();
    let robot = join(
        &host,
        "robot",
        vec![Role::Embodiment],
        PriorityClass::Scripted,
    )
    .await;
    robot.declare_parts(ROBOT).await.expect("declare");
    let ft = robot.publisher("ft").await.expect("publisher");
    tokio::time::sleep(Duration::from_millis(300)).await;
    for i in 0..5_u8 {
        ft.publish(robot.clock().now_ns(), Bytes::from(vec![i; 32]))
            .expect("publish");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    host.signal(15);
    let status = host.wait_exit();
    assert!(status.success(), "clean exit: {status:?}");
    let mut present = robot.host_present();
    tokio::time::timeout(WAIT, present.wait_for(|p| !*p))
        .await
        .expect("withdrawn presence seen in time")
        .expect("watch");

    // The recorder finished its segment before the host stopped.
    let dir = host
        .recordings
        .path()
        .join("rolling")
        .join(host.session.to_string());
    let segments: Vec<_> = std::fs::read_dir(&dir)
        .expect("the session's recordings")
        .map(|e| e.expect("entry").path())
        .collect();
    assert_eq!(segments.len(), 1, "{segments:?}");
    let messages = read(&segments[0]).expect("a complete segment");
    assert!(matches!(messages[0].message, Recorded::Authority(_)));
    let samples = messages
        .iter()
        .filter(|m| matches!(m.message, Recorded::Sample(_)))
        .count();
    assert_eq!(samples, 5);
}

/// The conformance fixtures of `rlmesh_session::spec` (§14, §15).
fn fixture(kind: &str, name: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../rlmesh-session-core/tests/fixtures")
        .join(kind)
        .join(format!("{name}.json"));
    std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("{}: {err}", path.display()))
}

/// A packed value of `len` bytes.
fn value(len: usize, fill: u8) -> Bytes {
    Bytes::from(vec![fill; len])
}

#[tokio::test(flavor = "multi_thread")]
async fn a_declared_ur10e_and_a_bound_policy_meet_through_a_host_process() {
    let host = HostProcess::spawn();
    let robot = join(
        &host,
        "ur10e",
        vec![Role::Embodiment],
        PriorityClass::Scripted,
    )
    .await;
    let map = fixture("robots", "ur10e");
    let view = robot
        .declare_parts(&map)
        .await
        .expect("declare the §14.1 map");
    let robot_id = robot.id().clone();
    // The other process normalised the authored map with the kind table, and
    // fingerprints the declaration as this one would.
    let normalised =
        rlmesh_session::spec::normalise(&robot_id, SPEC_VERSION, &map).expect("normalise");
    assert_eq!(
        view.contract.get(&robot_id),
        Some(&rlmesh_session::spec::normalise::fingerprint(&normalised))
    );
    let wrist = view.part_at(&robot_id, "wrist/ft").expect("wrist/ft");
    assert_eq!(
        wrist
            .fields
            .iter()
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>(),
        ["force", "torque"]
    );

    // A policy binds against the host's parts and declares the binding.
    let policy = join(&host, "pi0", vec![Role::Controller], PriorityClass::Model).await;
    policy
        .wait_for("the robot's parts", WAIT, |s| {
            s.part_at(&robot_id, "task").is_some()
        })
        .await
        .expect("parts");
    let binding = policy
        .bind(
            &Policy::parse(&fixture("policies", "pi0_single")).expect("policy"),
            &RunOptions::named("pi0_single"),
        )
        .expect("binds as §15.1 says");
    let bound = policy.declare_bound(&binding).await.expect("declare");
    let record = bound.binding(policy.id()).expect("recorded");
    assert_eq!(record.fingerprint, binding.fingerprint);
    assert_eq!(record.table_json, binding.table_text());
    assert_eq!(record.contracts, binding.contracts);
    assert_eq!(record.grids.len(), 2, "arm and gripper");
    let modes: Vec<ContractMode> = binding.contracts.iter().map(|c| c.mode).collect();
    assert_eq!(
        modes,
        [
            ContractMode::Every,
            ContractMode::Latest,
            ContractMode::Latest,
            ContractMode::Latest,
            ContractMode::Every
        ]
    );
    let armed = policy.arm().await.expect("arm");
    robot
        .wait_for("armed", WAIT, |s| s.epoch == armed.epoch)
        .await
        .expect("view");

    // Install before authority: the robot has no publishers yet.
    let channels = binding.channels();
    let err = policy
        .grant_leases(channels.clone(), Vec::new(), None)
        .await
        .expect_err("no plan installed");
    assert_eq!(err.refusal().map(|e| e.status), Some(CtlStatus::NotReady));
    let scene = robot.image_publisher("scene/cam").await.expect("scene/cam");
    let wrist_cam = robot.image_publisher("wrist/cam").await.expect("wrist/cam");
    let arm = robot.publisher("arm").await.expect("arm");
    let gripper = robot.publisher("gripper").await.expect("gripper");
    let task = robot.publisher("task").await.expect("task");
    policy
        .wait_for("every bound stream's plan installed", WAIT, |s| {
            binding
                .contracts
                .iter()
                .all(|c| s.demand(c.channel).is_some_and(|d| d.acked >= d.revision))
        })
        .await
        .expect("acknowledged");
    let leases = policy
        .grant_leases(channels, Vec::new(), None)
        .await
        .expect("ready: granted");
    assert!(leases.iter().all(|l| l.binding == binding.fingerprint));

    // The controller's cache snapshots the published inputs.
    let inputs = Inputs::open(&policy, &binding).await.expect("inputs");
    tokio::time::sleep(Duration::from_millis(300)).await;
    // Publish like a live robot, until the controller observes: a single
    // sample can age past its staleness bound on a loaded host before the
    // cache snapshots it.
    let mut scene_frame = Vec::new();
    synthetic_frame(1280, 720, 0, &mut scene_frame);
    let mut wrist_frame = Vec::new();
    synthetic_frame(640, 480, 0, &mut wrist_frame);
    let arm_sample = value(arm.spec().layout.stride().expect("packed"), 1);
    let publish_inputs = || {
        scene
            .offer(robot.clock().now_ns(), &scene_frame)
            .expect("scene frame");
        wrist_cam
            .offer(robot.clock().now_ns(), &wrist_frame)
            .expect("wrist frame");
        arm.publish(robot.clock().now_ns(), arm_sample.clone())
            .expect("arm")
            .expect("sent");
        gripper
            .publish(robot.clock().now_ns(), value(8, 0))
            .expect("gripper");
        task.publish(
            robot.clock().now_ns(),
            Bytes::from_static(b"put the cup on the plate"),
        )
        .expect("task");
    };
    publish_inputs();
    let started = Instant::now();
    let observation = loop {
        match inputs.observe(policy.session_now_ns()) {
            Ok(observation) => break observation,
            Err(ObserveError::Skipped(skip)) => {
                assert!(started.elapsed() < WAIT, "still skipped: {skip:?}");
                tokio::time::sleep(Duration::from_millis(10)).await;
                publish_inputs();
            }
            Err(err) => panic!("{err}"),
        }
    };
    assert_eq!(observation.meta.based_on.len(), 5);
    assert!(observation.meta.inputs.iter().all(|i| !i.stale));
    // What predict receives: decoded, padded and resized images, the
    // joint positions alone, the text.
    let tensor = |entry: &str| match observation.value(entry) {
        Some(Value::Tensor(t)) => t.clone(),
        other => panic!("{entry}: {other:?}"),
    };
    for camera in ["image", "wrist_image"] {
        assert_eq!(tensor(camera).shape, [224, 224, 3]);
    }
    let state = tensor("state");
    assert_eq!(state.shape, [6]);
    assert_eq!(state.data, arm_sample[..24]);
    assert_eq!(
        observation.value("prompt"),
        Some(&Value::Text("put the cup on the plate".into()))
    );
}
