//! The session host in its own OS process (`rlmesh-session-host`), with
//! participants in this one, linked over per-class TCP on localhost.

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use bytes::Bytes;
use rlmesh_session::transport::free_local_base;
use rlmesh_session::{
    Participant, ParticipantConfig, SessionError, SubscribeOptions, Transport, TransportConfig,
};
use rlmesh_wire::{
    ChannelId, ChannelPriority, ChannelQos, ChannelSpec, Direction, Mode, ParticipantId,
    PriorityClass, Role, SessionId, TenantId, TensorLayout,
};

/// Participants listen too (the OS picks the ports): in peer mode, data
/// flows peer to peer once gossip tells peers where to connect.
fn any_local_port() -> std::net::SocketAddr {
    "127.0.0.1:0".parse().expect("addr")
}

const WAIT: Duration = Duration::from_secs(10);
const CMD: ChannelId = ChannelId(1);
const FT: ChannelId = ChannelId(2);

/// The host process; killed on drop whatever the test did.
struct HostProcess {
    child: Child,
    session: SessionId,
    base: std::net::SocketAddr,
}

impl HostProcess {
    fn spawn() -> Self {
        let base = free_local_base("127.0.0.1".parse().expect("ip")).expect("ports");
        let mut child = Command::new(env!("CARGO_BIN_EXE_rlmesh-session-host"))
            .args([
                "--tenant",
                "two-process",
                "--listen-base",
                &base.to_string(),
                "--lease-grace-ms",
                "300",
            ])
            .env("RUST_LOG", "warn")
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn the host");
        let stdout = child.stdout.take().expect("stdout");
        let mut line = String::new();
        BufReader::new(stdout)
            .read_line(&mut line)
            .expect("session id line");
        let session = line.trim().parse().expect("a session id");
        Self {
            child,
            session,
            base,
        }
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
        TenantId::new("two-process").expect("tenant"),
        host.session,
        ParticipantId::new(id).expect("id"),
        roles,
        priority,
    );
    config.heartbeat_period = Duration::from_millis(20);
    Participant::attach(transport, config)
        .await
        .expect("attach")
}

fn channels() -> Vec<ChannelSpec> {
    vec![
        ChannelSpec::new(
            CMD,
            "cmd",
            Direction::Actuate,
            "action/joint_pos",
            vec![TensorLayout::f32(&[6])],
            1000.0,
            ChannelQos::realtime(ChannelPriority::Control),
        ),
        ChannelSpec::new(
            FT,
            "ft",
            Direction::Sense,
            "proprio/eef_wrench",
            vec![TensorLayout::f32(&[6])],
            1000.0,
            ChannelQos::realtime(ChannelPriority::Realtime),
        ),
    ]
}

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
    robot.declare_channels(channels()).await.expect("declare");
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

    // Verbs, leases and preemption are decided in the other process.
    let armed = operator.arm().await.expect("arm");
    policy
        .wait_for("armed", WAIT, |s| s.epoch == armed.epoch)
        .await
        .expect("view");
    let model = policy.grant_lease(CMD, None).await.expect("model lease");
    let human = operator.grant_lease(CMD, None).await.expect("preempt");
    assert!(human.token > model.token);
    let view = robot
        .wait_for("the operator's lease", WAIT, |s| {
            s.mode == Mode::Active && s.lease(CMD).is_some_and(|l| l.token == human.token)
        })
        .await
        .expect("view");
    assert!(view.check_fence(CMD, model.token, view.epoch).is_err());

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
    let ft = robot.publisher(FT).await.expect("publisher");
    tokio::time::sleep(Duration::from_millis(300)).await;
    for i in 0..3_u8 {
        ft.publish(robot.clock().now_ns(), vec![Bytes::from(vec![i; 24])])
            .expect("publish");
    }
    tokio::time::sleep(Duration::from_millis(200)).await;
    let sub = policy
        .subscribe(
            FT,
            SubscribeOptions {
                warm_from_history: true,
                ..SubscribeOptions::default()
            },
        )
        .await
        .expect("subscribe");
    assert_eq!(sub.stats().from_history, 1, "LATEST keeps one sample");
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
async fn a_terminated_host_shuts_down_cleanly() {
    let mut host = HostProcess::spawn();
    let robot = join(
        &host,
        "robot",
        vec![Role::Embodiment],
        PriorityClass::Scripted,
    )
    .await;
    host.signal(15);
    let status = host.wait_exit();
    assert!(status.success(), "clean exit: {status:?}");
    let mut present = robot.host_present();
    tokio::time::timeout(WAIT, present.wait_for(|p| !*p))
        .await
        .expect("withdrawn presence seen in time")
        .expect("watch");
}
