//! A session host and several participants in one process, each with its own
//! transport (one Zenoh session per traffic class), linked over TCP on
//! localhost.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use bytes::Bytes;
use rlmesh_session::media::{decode_image, synthetic_frame};
use rlmesh_session::participant::{query_authority, wait_for_host};
use rlmesh_session::transport::free_local_base;
use rlmesh_session::{
    HostConfig, ImagePublisher, MonotonicClock, Participant, ParticipantConfig, SessionError,
    SessionHost, SubscribeOptions, TrafficClass, Transport, TransportConfig,
};
use rlmesh_wire::codec;
use rlmesh_wire::{
    ChannelId, ChannelPriority, ChannelQos, ChannelSpec, Congestion, CtlStatus, Delivery,
    Direction, Encoding, Epoch, FenceError, History, Mode, ParticipantId, PriorityClass, Role,
    Sample, SessionId, TenantId, TensorLayout,
};

/// Participants listen too (the OS picks the ports): in peer mode, data
/// flows peer to peer once gossip tells peers where to connect.
fn any_local_port() -> std::net::SocketAddr {
    "127.0.0.1:0".parse().expect("addr")
}

const WAIT: Duration = Duration::from_secs(5);
const GRACE: Duration = Duration::from_millis(300);
const JOINT_CMD: ChannelId = ChannelId(1);
const FT: ChannelId = ChannelId(2);
const CAMERA: ChannelId = ChannelId(3);
const FT_MAX_AGE: Duration = Duration::from_millis(40);

/// The robot's channels: commands in, force/torque and a camera out.
fn robot_channels() -> Vec<ChannelSpec> {
    let mut ft_qos = ChannelQos::realtime(ChannelPriority::Realtime);
    ft_qos.max_age = Some(FT_MAX_AGE);
    ft_qos.history = History::Window(4);
    vec![
        ChannelSpec::new(
            JOINT_CMD,
            "joint_cmd",
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
            vec![TensorLayout::f32(&[18])],
            1000.0,
            ft_qos,
        ),
        ChannelSpec::new(
            CAMERA,
            "wrist_cam",
            Direction::Sense,
            "image/wrist",
            vec![TensorLayout::u8_image(8, 8, 3)],
            30.0,
            ChannelQos {
                delivery: Delivery::BestEffort,
                history: History::Latest,
                max_age: None,
                deadline: None,
                priority: ChannelPriority::Bulk,
                congestion: Congestion::Drop,
                max_bitrate_bps: None,
            },
        ),
    ]
}

fn ft_leaves(value: f32) -> Vec<Bytes> {
    vec![Bytes::from(
        std::iter::repeat_n(value.to_le_bytes(), 18)
            .flatten()
            .collect::<Vec<u8>>(),
    )]
}

struct World {
    anchor: Instant,
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
        let anchor = Instant::now();
        let base = free_local_base("127.0.0.1".parse().expect("ip")).expect("ports");
        let transport =
            Transport::open(&TransportConfig::per_class_tcp(Some(base), None).expect("config"))
                .await
                .expect("host transport");
        let tenant = TenantId::new("test").expect("tenant");
        let session = SessionId::new();
        let mut config = HostConfig::new(tenant.clone(), session);
        config.clock = MonotonicClock::with_anchor(anchor, 0).shared();
        config.lease_grace = GRACE;
        config.health_log = Some(Duration::from_millis(100));
        let host = SessionHost::start(transport, config).await.expect("host");
        Self {
            anchor,
            base,
            tenant,
            session,
            host,
        }
    }

    async fn transport(&self) -> Transport {
        Transport::open(
            &TransportConfig::per_class_tcp(Some(any_local_port()), Some(self.base))
                .expect("config"),
        )
        .await
        .expect("participant transport")
    }

    fn config(
        &self,
        id: &str,
        roles: Vec<Role>,
        priority: PriorityClass,
        shift_ns: i64,
    ) -> ParticipantConfig {
        let mut config = ParticipantConfig::new(
            self.tenant.clone(),
            self.session,
            ParticipantId::new(id).expect("id"),
            roles,
            priority,
        );
        config.clock = MonotonicClock::with_anchor(self.anchor, shift_ns).shared();
        config.heartbeat_period = Duration::from_millis(20);
        config
    }

    /// A participant whose clock reads the host's plus `shift_ns`.
    async fn join(
        &self,
        id: &str,
        roles: Vec<Role>,
        priority: PriorityClass,
        shift_ns: i64,
    ) -> Participant {
        Participant::attach(
            self.transport().await,
            self.config(id, roles, priority, shift_ns),
        )
        .await
        .expect("attach")
    }

    /// The embodiment, with its channels declared.
    async fn robot(&self) -> Participant {
        let robot = self
            .join("robot", vec![Role::Embodiment], PriorityClass::Scripted, 0)
            .await;
        robot
            .declare_channels(robot_channels())
            .await
            .expect("declare");
        robot
    }

    async fn policy(&self) -> Participant {
        self.join("policy", vec![Role::Controller], PriorityClass::Model, 0)
            .await
    }

    async fn operator(&self) -> Participant {
        self.join(
            "operator",
            vec![Role::Controller],
            PriorityClass::Operator,
            0,
        )
        .await
    }
}

/// Arms through `by` and waits until `others` see the new epoch: a lease
/// request carries its sender's epoch, so one sent from a view that has not
/// caught up with the arm is refused (`StaleEpoch`), by design.
async fn arm_seen_by(by: &Participant, others: &[&Participant]) {
    let armed = by.arm().await.expect("arm");
    for p in others {
        p.wait_for("the armed epoch", WAIT, |s| s.epoch >= armed.epoch)
            .await
            .expect("view");
    }
}

fn status(err: SessionError) -> CtlStatus {
    match err {
        SessionError::Rejected(err) => err.status,
        other => panic!("expected a refusal, got {other}"),
    }
}

async fn wait_until(what: &str, mut done: impl FnMut() -> bool) {
    let started = Instant::now();
    while !done() {
        assert!(started.elapsed() < WAIT, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn attach_arm_and_grant_a_lease() {
    let world = World::start().await;
    let robot = world.robot().await;
    let policy = world.policy().await;

    let roster = world.host.snapshot().roster;
    assert_eq!(roster.len(), 2);
    assert_eq!(policy.authority().expect("view").mode, Mode::Disarmed);

    let armed = policy.arm().await.expect("arm");
    assert_eq!((armed.mode, armed.epoch), (Mode::Armed, Epoch(2)));
    let lease = policy.grant_lease(JOINT_CMD, None).await.expect("grant");
    assert_eq!(lease.holder, *policy.id());

    // The embodiment learns of it through the published update.
    let view = robot
        .wait_for("lease visible to the robot", WAIT, |s| {
            s.lease(JOINT_CMD).is_some()
        })
        .await
        .expect("update");
    assert_eq!(view.mode, Mode::Active);
    view.check_fence(JOINT_CMD, lease.token, view.epoch)
        .expect("fence");

    // An embodiment cannot take a lease.
    assert_eq!(
        status(
            robot
                .grant_lease(JOINT_CMD, None)
                .await
                .expect_err("refused")
        ),
        CtlStatus::Forbidden
    );
    policy.release_lease(&lease).await.expect("release");
    assert_eq!(world.host.snapshot().mode, Mode::Armed);
    policy.detach().await.expect("detach");
    robot.close().await;
    world.host.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn operator_preempts_the_model_and_fences_its_token() {
    let world = World::start().await;
    let robot = world.robot().await;
    let policy = world.policy().await;
    let operator = world.operator().await;

    arm_seen_by(&operator, &[&policy]).await;
    let model = policy.grant_lease(JOINT_CMD, None).await.expect("model");
    let human = operator
        .grant_lease(JOINT_CMD, None)
        .await
        .expect("preempt");
    assert!(human.token > model.token, "preemption bumps the token");

    let view = robot
        .wait_for("operator holds the lease", WAIT, |s| {
            s.lease(JOINT_CMD)
                .is_some_and(|l| l.holder == *operator.id())
        })
        .await
        .expect("update");
    assert_eq!(
        view.check_fence(JOINT_CMD, model.token, view.epoch),
        Err(FenceError::StaleToken {
            channel: JOINT_CMD,
            current: human.token,
            got: model.token
        })
    );
    view.check_fence(JOINT_CMD, human.token, view.epoch)
        .expect("new token");

    // The model's stale token is refused, and it cannot take the lease back.
    assert_eq!(
        status(policy.release_lease(&model).await.expect_err("refused")),
        CtlStatus::StaleToken
    );
    assert_eq!(
        status(policy.grant_lease(JOINT_CMD, None).await.expect_err("busy")),
        CtlStatus::LeaseBusy
    );
    policy
        .wait_for("model sees the handoff", WAIT, |s| {
            s.lease(JOINT_CMD).is_some_and(|l| l.token == human.token)
        })
        .await
        .expect("model view");

    // Handing back: the operator releases and the model is granted anew.
    operator.release_lease(&human).await.expect("release");
    let again = policy.grant_lease(JOINT_CMD, None).await.expect("again");
    assert!(again.token > human.token);
}

#[tokio::test(flavor = "multi_thread")]
async fn heartbeat_loss_lapses_the_lease_and_goes_safe() {
    let world = World::start().await;
    let robot = world.robot().await;
    let policy = world.policy().await;
    policy.arm().await.expect("arm");
    let lease = policy.grant_lease(JOINT_CMD, None).await.expect("grant");

    // Heartbeats keep it alive well past the grace.
    tokio::time::sleep(GRACE * 3).await;
    assert_eq!(world.host.snapshot().mode, Mode::Active);

    let stopped = Instant::now();
    policy.set_heartbeats_enabled(false);
    let view = robot
        .wait_for("SAFE after lapse", WAIT, |s| s.mode == Mode::Safe)
        .await
        .expect("safe");
    let elapsed = stopped.elapsed();
    assert!(view.leases.is_empty());
    assert!(elapsed >= GRACE - Duration::from_millis(30), "{elapsed:?}");
    assert!(elapsed < GRACE * 3, "lapse took {elapsed:?}");

    // Still present: only its heartbeats stopped. Re-arming is a fresh epoch.
    assert!(view.participant(policy.id()).is_some());
    policy.set_heartbeats_enabled(true);
    let armed = policy.arm().await.expect("re-arm");
    assert_eq!(armed.epoch, view.epoch.next());
    assert!(
        armed
            .check_fence(JOINT_CMD, lease.token, view.epoch)
            .is_err()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_vanished_holder_takes_the_session_safe() {
    let world = World::start().await;
    let robot = world.robot().await;
    let policy = world.policy().await;
    policy.arm().await.expect("arm");
    policy.grant_lease(JOINT_CMD, None).await.expect("grant");

    let gone = Instant::now();
    let id = policy.id().clone();
    let transport = policy.transport().clone();
    drop(policy);
    transport.close().await.expect("close");
    let view = robot
        .wait_for("SAFE after liveliness loss", WAIT, |s| s.mode == Mode::Safe)
        .await
        .expect("safe");
    assert!(view.participant(&id).is_none());
    tracing::info!("liveliness loss to SAFE in {:?}", gone.elapsed());
}

#[tokio::test(flavor = "multi_thread")]
async fn estop_latches_until_a_local_clear_then_disarm_and_arm() {
    let world = World::start().await;
    let robot = world.robot().await;
    let policy = world.policy().await;
    let operator = world.operator().await;
    arm_seen_by(&operator, &[&policy]).await;
    policy.grant_lease(JOINT_CMD, None).await.expect("grant");

    let stopped = policy.estop("test button").await.expect("estop");
    assert_eq!(stopped.mode, Mode::Estop);
    assert!(stopped.estop_latched && stopped.leases.is_empty());
    robot
        .wait_for("robot sees ESTOP", WAIT, |s| s.mode == Mode::Estop)
        .await
        .expect("update");

    for refused in [operator.arm().await, operator.disarm().await] {
        assert_eq!(
            status(refused.expect_err("latched")),
            CtlStatus::EstopLatched
        );
    }
    assert_eq!(
        status(
            policy
                .grant_lease(JOINT_CMD, None)
                .await
                .expect_err("estop")
        ),
        CtlStatus::EstopLatched
    );
    assert_eq!(
        status(operator.clear().await.expect_err("not local")),
        CtlStatus::Forbidden
    );

    let cleared = robot.clear().await.expect("local clear");
    assert_eq!(cleared.mode, Mode::Estop);
    assert!(!cleared.estop_latched);
    operator.disarm().await.expect("disarm");
    let armed = operator.arm().await.expect("arm");
    assert_eq!(armed.mode, Mode::Armed);
    assert_eq!(armed.epoch, stopped.epoch.next());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_late_observer_gets_the_snapshot_by_query() {
    let world = World::start().await;
    let _robot = world.robot().await;
    let policy = world.policy().await;
    policy.arm().await.expect("arm");
    let lease = policy.grant_lease(JOINT_CMD, None).await.expect("grant");

    // A fresh transport that never attaches.
    let transport = world.transport().await;
    let started = Instant::now();
    wait_for_host(&transport, world.host.keys(), WAIT)
        .await
        .expect("host present");
    let snapshot = query_authority(&transport, world.host.keys(), WAIT)
        .await
        .expect("query");
    tracing::info!("late join to snapshot in {:?}", started.elapsed());
    let host = world.host.snapshot();
    assert_eq!(
        (snapshot.epoch, snapshot.revision, snapshot.mode),
        (host.epoch, host.revision, host.mode)
    );
    assert_eq!(
        snapshot.lease(JOINT_CMD).map(|l| l.token),
        Some(lease.token)
    );

    // Attaching as an observer works the same way and is never a controller.
    let viewer = world
        .join("twin", vec![Role::Observer], PriorityClass::Scripted, 0)
        .await;
    assert_eq!(
        viewer
            .authority()
            .expect("view")
            .lease(JOINT_CMD)
            .map(|l| l.token),
        Some(lease.token)
    );
    assert_eq!(
        status(
            viewer
                .grant_lease(JOINT_CMD, None)
                .await
                .expect_err("observer")
        ),
        CtlStatus::Forbidden
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn heartbeats_estimate_each_participants_clock_offset_within_bound() {
    let world = World::start().await;
    // Clock domains 5 s ahead of and 2.5 s behind the host.
    let ahead = world
        .join(
            "ahead",
            vec![Role::Embodiment],
            PriorityClass::Scripted,
            5_000_000_000,
        )
        .await;
    let behind = world
        .join(
            "behind",
            vec![Role::Controller],
            PriorityClass::Model,
            -2_500_000_000,
        )
        .await;

    for (participant, true_offset) in [(&ahead, -5_000_000_000_i64), (&behind, 2_500_000_000)] {
        let snapshot = participant
            .wait_for("clock mapping", WAIT, |s| {
                s.clock(participant.id()).is_some()
            })
            .await
            .expect("mapping");
        let mapping = snapshot.clock(participant.id()).expect("mapping").clone();
        let error = (mapping.offset_ns - true_offset).unsigned_abs();
        tracing::info!(
            "{}: offset {} ns ± {} ns (true {true_offset}, off by {error} ns)",
            participant.id(),
            mapping.offset_ns,
            mapping.error_ns
        );
        assert!(
            error <= mapping.error_ns,
            "{}: |{} - {true_offset}| > {}",
            participant.id(),
            mapping.offset_ns,
            mapping.error_ns
        );
        assert!(
            mapping.error_ns < 50_000_000,
            "localhost bound should be well under 50 ms: {mapping:?}"
        );
        let (session_ns, _) = mapping.to_session_ns(0);
        assert!((session_ns - true_offset).unsigned_abs() <= mapping.error_ns);
        // The participant converts its own clock through the same mapping.
        let local = participant.clock().now_ns();
        assert!(
            (participant.to_session_ns(local) - (local + true_offset)).unsigned_abs()
                <= mapping.error_ns
        );
    }

    let mut health = world.host.watch_health();
    let rows = tokio::time::timeout(
        WAIT,
        health.wait_for(|rows| {
            rows.len() == 2 && rows.iter().all(|r| r.rtt_ns.is_some() && r.heartbeats > 5)
        }),
    )
    .await
    .expect("health in time")
    .expect("health")
    .clone();
    for row in rows {
        tracing::info!("{row}");
        assert!(row.heartbeat_gap_max_ns > 0);
        assert!(row.offset.is_some());
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn channels_are_declared_through_the_host_and_late_joiners_see_them() {
    let world = World::start().await;
    let robot = world.robot().await;
    // Re-declaring the same table is a no-op.
    let revision = world.host.snapshot().revision;
    robot
        .declare_channels(robot_channels())
        .await
        .expect("redeclare");
    assert_eq!(world.host.snapshot().revision, revision);

    // A late joiner learns the table with the rest of the snapshot.
    let twin = world
        .join("twin", vec![Role::Observer], PriorityClass::Scripted, 0)
        .await;
    let view = twin.authority().expect("view");
    assert_eq!(view.channels.len(), 3);
    let ft = view.channel(FT).expect("ft");
    assert_eq!(ft.owner.as_ref(), Some(robot.id()));
    assert_eq!(ft.layout, vec![TensorLayout::f32(&[18])]);
    assert_eq!(ft.qos.max_age, Some(FT_MAX_AGE));

    // Another embodiment cannot take the id; an observer cannot own
    // actuation; leases need a declared actuation channel.
    let other = world
        .join("other", vec![Role::Embodiment], PriorityClass::Scripted, 0)
        .await;
    let mut clash = robot_channels();
    clash.truncate(1);
    clash[0].name = "other_cmd".into();
    assert_eq!(
        status(
            other
                .declare_channels(clash.clone())
                .await
                .expect_err("clash")
        ),
        CtlStatus::ChannelConflict
    );
    assert_eq!(
        status(twin.declare_channels(clash).await.expect_err("observer")),
        CtlStatus::Forbidden
    );
    let policy = world.policy().await;
    policy.arm().await.expect("arm");
    for channel in [FT, ChannelId(99)] {
        assert_eq!(
            status(
                policy
                    .grant_lease(channel, None)
                    .await
                    .expect_err("not actuation")
            ),
            CtlStatus::UnknownChannel
        );
    }

    // Detaching the owner removes its channels.
    robot.detach().await.expect("detach");
    twin.wait_for("channels removed", WAIT, |s| s.channels.is_empty())
        .await
        .expect("removed");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_participant_needing_a_capability_the_session_lacks_is_refused() {
    let world = World::start().await;
    let mut config = world.config("future", vec![Role::Controller], PriorityClass::Model, 0);
    config.capabilities = vec!["merge.blend".into()];
    let err = Participant::attach(world.transport().await, config)
        .await
        .err()
        .expect("refused");
    assert_eq!(status(err), CtlStatus::Incompatible);
    assert!(world.host.snapshot().roster.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn samples_carry_seq_and_a_late_subscriber_warms_from_the_host_history() {
    let world = World::start().await;
    let robot = world.robot().await;
    let ft = robot.publisher(FT).await.expect("publisher");
    let early = world
        .join("early", vec![Role::Observer], PriorityClass::Scripted, 0)
        .await;
    let early_ft = early
        .subscribe(FT, SubscribeOptions::default())
        .await
        .expect("subscribe");
    // Let the subscriptions (the host's history store too) propagate.
    tokio::time::sleep(Duration::from_millis(200)).await;
    for i in 1..=6_u8 {
        let seq = ft
            .publish(robot.clock().now_ns(), ft_leaves(f32::from(i)))
            .expect("publish");
        assert_eq!(seq, u64::from(i));
    }
    wait_until("early subscriber has all six", || {
        early_ft.stats().last_seq == 6
    })
    .await;
    let stats = early_ft.stats();
    assert_eq!((stats.received, stats.lost), (6, 0));
    let latest = early_ft.latest().expect("latest");
    assert_eq!(latest.received.sample.epoch, world.host.snapshot().epoch);
    assert_eq!(latest.received.sample.leaves, ft_leaves(6.0));
    assert_eq!(
        early_ft
            .window()
            .iter()
            .map(|r| r.sample.seq)
            .collect::<Vec<_>>(),
        vec![3, 4, 5, 6],
        "WINDOW(4)"
    );
    // A value that does not fit the declaration is refused at the publisher.
    assert!(matches!(
        ft.publish(0, vec![Bytes::from(vec![0_u8; 8])]),
        Err(SessionError::InvalidValue { .. })
    ));

    // A late joiner is served the host's window.
    let late = world
        .join("late", vec![Role::Observer], PriorityClass::Scripted, 0)
        .await;
    let late_ft = late
        .subscribe(
            FT,
            SubscribeOptions {
                warm_from_history: true,
                ..SubscribeOptions::default()
            },
        )
        .await
        .expect("subscribe");
    let stats = late_ft.stats();
    assert_eq!(stats.last_seq, 6);
    assert_eq!((stats.received, stats.from_history), (4, 4), "the window");
    assert_eq!(stats.lost, 0, "history starts the seq count");

    // Heartbeats report the newest seq seen per channel.
    let mut health = world.host.watch_health();
    tokio::time::timeout(
        WAIT,
        health.wait_for(|rows| {
            rows.iter()
                .any(|r| r.participant == *late.id() && r.report.last_seq_seen == vec![(FT, 6)])
        }),
    )
    .await
    .expect("reported in time")
    .expect("health");
}

#[tokio::test(flavor = "multi_thread")]
async fn seq_gaps_count_as_losses_and_old_samples_go_stale() {
    let world = World::start().await;
    let robot = world.robot().await;
    let viewer = world
        .join("viewer", vec![Role::Observer], PriorityClass::Scripted, 0)
        .await;
    let delays = Arc::new(AtomicU64::new(0));
    let ft = viewer
        .subscribe(
            FT,
            SubscribeOptions {
                on_sample: Some({
                    let delays = delays.clone();
                    Arc::new(move |r| {
                        assert!(r.delay_ns() >= 0, "same clock domain: {}", r.delay_ns());
                        delays.fetch_add(1, Ordering::Relaxed);
                    })
                }),
                ..SubscribeOptions::default()
            },
        )
        .await
        .expect("subscribe");
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Raw frames on the channel's key, skipping seqs 3 and 4.
    let raw = world.transport().await;
    let key = robot.keys().samples(robot.id(), FT);
    let epoch = world.host.snapshot().epoch;
    tokio::time::sleep(Duration::from_millis(200)).await;
    for seq in [1_u64, 2, 5, 6] {
        let sample = Sample {
            channel: FT,
            seq,
            epoch,
            capture_ns: robot.clock().now_ns(),
            publish_ns: robot.clock().now_ns(),
            tick: 0,
            leaves: ft_leaves(0.5),
        };
        let frame = codec::encode_sample(&sample).to_vec();
        raw.session(TrafficClass::Realtime)
            .put(&key, frame)
            .await
            .expect("put");
    }
    wait_until("four samples", || ft.stats().last_seq == 6).await;
    let stats = ft.stats();
    assert_eq!((stats.received, stats.lost), (4, 2));
    assert_eq!(delays.load(Ordering::Relaxed), 4);

    let fresh = ft.latest().expect("latest");
    assert!(!fresh.stale, "age {} ns", fresh.age_ns);
    tokio::time::sleep(FT_MAX_AGE * 2).await;
    let old = ft.latest().expect("latest");
    assert!(old.stale && old.age_ns > FT_MAX_AGE.as_nanos() as i64);
    assert_eq!(ft.stats().stale_reads, 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn each_class_rides_its_own_session_and_jpeg_frames_decode() {
    let world = World::start().await;
    let robot = world
        .join("robot", vec![Role::Embodiment], PriorityClass::Scripted, 0)
        .await;
    let mut channels = robot_channels();
    let camera = channels
        .iter_mut()
        .find(|c| c.id == CAMERA)
        .expect("camera");
    camera.layout = vec![TensorLayout::u8_image(48, 64, 3)];
    camera.encoding = Encoding::Jpeg { quality: 80 };
    camera.qos.max_bitrate_bps = Some(4_000_000);
    robot.declare_channels(channels).await.expect("declare");
    let policy = world.policy().await;

    // Four distinct Zenoh sessions, each linked to the host's session of the
    // same class (and only to it).
    let transport = robot.transport();
    assert!(!transport.is_shared());
    let mut zids: Vec<_> = TrafficClass::ALL
        .iter()
        .map(|c| transport.session(*c).zid())
        .collect();
    zids.dedup();
    assert_eq!(zids.len(), 4);
    for class in TrafficClass::ALL {
        let peers: Vec<_> = transport.session(class).info().peers_zid().await.collect();
        assert!(!peers.is_empty(), "{} session has a peer", class.name());
    }

    let frames = Arc::new(std::sync::Mutex::new(Vec::new()));
    let cam_sub = policy
        .subscribe(
            CAMERA,
            SubscribeOptions {
                on_sample: Some({
                    let frames = frames.clone();
                    Arc::new(move |r| {
                        if let Ok(mut f) = frames.lock() {
                            f.push(r.sample.clone());
                        }
                    })
                }),
                ..SubscribeOptions::default()
            },
        )
        .await
        .expect("subscribe");
    let images = ImagePublisher::new(robot.publisher(CAMERA).await.expect("publisher"))
        .expect("image publisher");
    tokio::time::sleep(Duration::from_millis(300)).await;
    let mut frame = Vec::new();
    for t in 0..10 {
        synthetic_frame(64, 48, t, &mut frame);
        images.offer(robot.clock().now_ns(), &frame).expect("offer");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    wait_until("frames arrive", || {
        frames.lock().map(|f| f.len()).unwrap_or(0) >= 5
    })
    .await;
    let sample = frames.lock().expect("frames")[0].clone();
    let pixels = decode_image(cam_sub.spec(), &sample).expect("decode");
    assert_eq!(pixels.len(), 64 * 48 * 3);
    assert!(sample.leaves[0].len() < pixels.len() / 2, "compressed");
    let stats = images.stats();
    assert!(stats.sent >= 5 && stats.encoded == 10, "{stats:?}");
}
