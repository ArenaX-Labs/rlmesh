//! A session host and several participants in one process, each with its own
//! transport (one Zenoh session per traffic class), linked over TCP on
//! localhost.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use bytes::Bytes;
use rlmesh_session::media::{decode_image, synthetic_frame};
use rlmesh_session::participant::{query_authority, wait_for_host};
use rlmesh_session::transport::open_local_listeners;
use rlmesh_session::{
    ChunkDraft, HostConfig, MonotonicClock, Participant, ParticipantConfig, SessionError,
    SessionHost, SubscribeOptions, TrafficClass, Transport, TransportConfig, VirtualConfig,
};
use rlmesh_spec::{Policy, RunOptions};
use rlmesh_wire::codec;
use rlmesh_wire::{
    ChannelId, ClockMode, ContractMode, CtlStatus, Encoding, Epoch, FenceError, Fingerprint,
    HistoryQuery, Mode, ParticipantId, Plan, PriorityClass, Role, Sample, SessionId, TenantId,
    VirtualTick, Watermark,
};

/// Participants listen too (the OS picks the ports): in peer mode, data
/// flows peer to peer once gossip tells peers where to connect.
fn any_local_port() -> std::net::SocketAddr {
    "127.0.0.1:0".parse().expect("addr")
}

const WAIT: Duration = Duration::from_secs(5);
const GRACE: Duration = Duration::from_millis(300);
const FT_MAX_AGE: Duration = Duration::from_millis(40);

/// The robot: joint commands in (a group with a Cartesian alternative),
/// force/torque and a camera out.
fn robot_map(camera: &str) -> String {
    format!(
        r#"{{
        "arm": {{"kind": "joints", "hz": 1000, "count": 6, "fields": ["pos"],
                "command": "position", "priority": "control", "group": "arm"}},
        "arm/tcp": {{"kind": "x/tcp", "hz": 1000, "fields": {{"dpos": ["float32", [3]]}},
                    "command": "delta", "group": "arm"}},
        "ft": {{"kind": "x/ft", "hz": 1000, "fields": {{"wrench": ["float32", [18]]}},
               "priority": "realtime", "max_age_ms": 40}},
        "wrist/cam": {{"kind": "camera", "hz": 30, {camera}}}
    }}"#
    )
}

fn raw_camera() -> String {
    robot_map(r#""width": 8, "height": 8"#)
}

fn ft_payload(value: f32) -> Bytes {
    Bytes::from(
        std::iter::repeat_n(value.to_le_bytes(), 18)
            .flatten()
            .collect::<Vec<u8>>(),
    )
}

fn robot_id() -> ParticipantId {
    ParticipantId::new("robot").expect("id")
}

/// A fingerprint unique to a participant (the binder computes real ones).
fn fingerprint_of(p: &Participant, revision: u64) -> Fingerprint {
    let id = p
        .id()
        .as_str()
        .bytes()
        .fold(0_u64, |h, b| h.wrapping_mul(31).wrapping_add(u64::from(b)));
    Fingerprint::from(id.wrapping_add(revision))
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
        Self::start_with(|_| {}).await
    }

    async fn start_with(configure: impl FnOnce(&mut HostConfig)) -> Self {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "warn".into()),
            )
            .with_test_writer()
            .try_init();
        let anchor = Instant::now();
        let (transport, base) = open_local_listeners("127.0.0.1".parse().expect("ip"))
            .await
            .expect("host transport");
        let tenant = TenantId::new("test").expect("tenant");
        let session = SessionId::new();
        let mut config = HostConfig::new(tenant.clone(), session);
        config.clock = MonotonicClock::with_anchor(anchor, 0).shared();
        config.lease_grace = GRACE;
        config.health_log = Some(Duration::from_millis(100));
        configure(&mut config);
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

    /// The embodiment, with its parts declared.
    async fn robot(&self) -> Participant {
        let robot = self
            .join("robot", vec![Role::Embodiment], PriorityClass::Scripted, 0)
            .await;
        robot.declare_parts(&raw_camera()).await.expect("declare");
        robot
    }

    /// A controller bound to every commandable part of the robot.
    async fn controller(&self, id: &str, priority: PriorityClass) -> Participant {
        let p = self.join(id, vec![Role::Controller], priority, 0).await;
        let view = p
            .wait_for("the robot's parts", WAIT, |s| {
                s.part_at(&robot_id(), "arm").is_some()
            })
            .await
            .expect("parts");
        let channels = view
            .parts
            .iter()
            .filter(|part| part.command.is_some())
            .map(|part| part.channel)
            .collect();
        p.declare_binding(fingerprint_of(&p, 0), "{}", channels, 0.0)
            .await
            .expect("bind");
        p
    }

    async fn policy(&self) -> Participant {
        self.controller("policy", PriorityClass::Model).await
    }

    async fn operator(&self) -> Participant {
        self.controller("operator", PriorityClass::Operator).await
    }
}

fn channel(p: &Participant, path: &str) -> ChannelId {
    p.part(&robot_id(), path).expect(path).channel
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
    let arm = channel(&policy, "arm");

    let roster = world.host.snapshot().roster;
    assert_eq!(roster.len(), 2);
    assert_eq!(policy.authority().expect("view").mode, Mode::Disarmed);

    let armed = policy.arm().await.expect("arm");
    assert_eq!((armed.mode, armed.epoch), (Mode::Armed, Epoch(2)));
    let lease = policy.grant_lease(arm, None).await.expect("grant");
    assert_eq!(lease.holder, *policy.id());
    assert_eq!(lease.binding, fingerprint_of(&policy, 0));

    // The embodiment learns of it through the published update.
    let view = robot
        .wait_for("lease visible to the robot", WAIT, |s| {
            s.lease(arm).is_some()
        })
        .await
        .expect("update");
    assert_eq!(view.mode, Mode::Active);
    view.check_fence(arm, lease.token, view.epoch)
        .expect("fence");

    // An embodiment cannot take a lease.
    assert_eq!(
        status(robot.grant_lease(arm, None).await.expect_err("refused")),
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
    let arm = channel(&policy, "arm");

    arm_seen_by(&operator, &[&policy]).await;
    let model = policy.grant_lease(arm, None).await.expect("model");
    let human = operator.grant_lease(arm, None).await.expect("preempt");
    assert!(human.token > model.token, "preemption bumps the token");

    let view = robot
        .wait_for("operator holds the lease", WAIT, |s| {
            s.lease(arm).is_some_and(|l| l.holder == *operator.id())
        })
        .await
        .expect("update");
    assert_eq!(
        view.check_fence(arm, model.token, view.epoch),
        Err(FenceError::StaleToken {
            channel: arm,
            current: human.token,
            got: model.token
        })
    );
    view.check_fence(arm, human.token, view.epoch)
        .expect("new token");

    // The model's stale token is refused, and it cannot take the lease back.
    assert_eq!(
        status(policy.release_lease(&model).await.expect_err("refused")),
        CtlStatus::StaleToken
    );
    assert_eq!(
        status(policy.grant_lease(arm, None).await.expect_err("busy")),
        CtlStatus::LeaseBusy
    );
    policy
        .wait_for("model sees the handoff", WAIT, |s| {
            s.lease(arm).is_some_and(|l| l.token == human.token)
        })
        .await
        .expect("model view");

    // Handing back: the operator releases and the model is granted anew.
    operator.release_lease(&human).await.expect("release");
    let again = policy.grant_lease(arm, None).await.expect("again");
    assert!(again.token > human.token);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_rebinding_reissues_the_lease_under_a_fresh_token() {
    let world = World::start().await;
    let robot = world.robot().await;
    let policy = world.policy().await;
    let arm = channel(&policy, "arm");
    policy.arm().await.expect("arm");
    let lease = policy.grant_lease(arm, None).await.expect("grant");

    // A new binding table (another fingerprint) over the same part.
    let view = policy
        .declare_binding(fingerprint_of(&policy, 1), r#"{"rev": 2}"#, vec![arm], 0.0)
        .await
        .expect("rebind");
    let rebound = view.lease(arm).expect("kept").clone();
    assert!(rebound.token > lease.token, "a fresh token");
    assert_eq!(rebound.binding, fingerprint_of(&policy, 1));
    assert_eq!(view.mode, Mode::Active, "mode unchanged");
    let robot_view = robot
        .wait_for("the robot sees the new token", WAIT, |s| {
            s.lease(arm).is_some_and(|l| l.token == rebound.token)
        })
        .await
        .expect("view");
    assert!(matches!(
        robot_view.check_fence(arm, lease.token, robot_view.epoch),
        Err(FenceError::StaleToken { .. })
    ));
    assert_eq!(
        robot_view
            .binding(policy.id())
            .map(|b| b.table_json.as_str()),
        Some(r#"{"rev": 2}"#)
    );
    // A grant under the old fingerprint is no longer bound; a renewal under
    // the new one keeps the fresh token.
    let renewed = policy.grant_lease(arm, None).await.expect("renew");
    assert_eq!(renewed.token, rebound.token);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_command_group_holds_one_member_and_switches_atomically() {
    let world = World::start().await;
    let _robot = world.robot().await;
    let policy = world.policy().await;
    let operator = world.operator().await;
    let (arm, tcp) = (channel(&policy, "arm"), channel(&policy, "arm/tcp"));
    arm_seen_by(&operator, &[&policy]).await;
    let joint = policy.grant_lease(arm, None).await.expect("arm");
    assert_eq!(
        status(policy.grant_lease(tcp, None).await.expect_err("busy")),
        CtlStatus::GroupBusy
    );
    assert_eq!(
        status(
            policy
                .grant_leases(vec![arm, tcp], Vec::new(), None)
                .await
                .expect_err("both")
        ),
        CtlStatus::InvalidRequest
    );
    let switched = policy
        .grant_leases(vec![tcp], vec![arm], None)
        .await
        .expect("switch");
    assert!(switched[0].token > joint.token);
    let view = policy.authority().expect("view");
    assert!(view.lease(arm).is_none() && view.lease(tcp).is_some());
    // The operator takes the joint member: the model's Cartesian lease goes
    // in the same revision.
    let revision = view.revision;
    operator.grant_lease(arm, None).await.expect("preempt");
    let view = operator.authority().expect("view");
    assert_eq!(view.revision, revision + 1);
    assert!(view.lease(tcp).is_none());
}

#[tokio::test(flavor = "multi_thread")]
async fn heartbeat_loss_lapses_the_lease_and_goes_safe() {
    let world = World::start().await;
    let robot = world.robot().await;
    let policy = world.policy().await;
    let arm = channel(&policy, "arm");
    policy.arm().await.expect("arm");
    let lease = policy.grant_lease(arm, None).await.expect("grant");

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
    assert!(armed.check_fence(arm, lease.token, view.epoch).is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_vanished_holder_takes_the_session_safe() {
    let world = World::start().await;
    let robot = world.robot().await;
    let policy = world.policy().await;
    let arm = channel(&policy, "arm");
    policy.arm().await.expect("arm");
    policy.grant_lease(arm, None).await.expect("grant");

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
    let arm = channel(&policy, "arm");
    arm_seen_by(&operator, &[&policy]).await;
    policy.grant_lease(arm, None).await.expect("grant");

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
        status(policy.grant_lease(arm, None).await.expect_err("estop")),
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
    let arm = channel(&policy, "arm");
    policy.arm().await.expect("arm");
    let lease = policy.grant_lease(arm, None).await.expect("grant");

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
    assert_eq!(snapshot.lease(arm).map(|l| l.token), Some(lease.token));

    // Attaching as an observer works the same way and is never a controller.
    let viewer = world
        .join("twin", vec![Role::Observer], PriorityClass::Scripted, 0)
        .await;
    assert_eq!(
        viewer
            .authority()
            .expect("view")
            .lease(arm)
            .map(|l| l.token),
        Some(lease.token)
    );
    assert_eq!(
        status(viewer.grant_lease(arm, None).await.expect_err("observer")),
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
async fn parts_are_declared_through_the_host_and_late_joiners_see_them() {
    let world = World::start().await;
    let robot = world.robot().await;
    // Re-declaring the same map is a no-op.
    let revision = world.host.snapshot().revision;
    robot.declare_parts(&raw_camera()).await.expect("redeclare");
    assert_eq!(world.host.snapshot().revision, revision);

    // A late joiner learns the parts with the rest of the snapshot.
    let twin = world
        .join("twin", vec![Role::Observer], PriorityClass::Scripted, 0)
        .await;
    let view = twin.authority().expect("view");
    assert_eq!(view.parts.len(), 4);
    let ft = view.part_at(&robot_id(), "ft").expect("ft");
    assert_eq!(ft.owner, robot_id());
    assert_eq!(ft.max_age, Some(FT_MAX_AGE));
    assert_eq!(ft.stream().expect("stream").layout.stride(), Some(72));
    let arm = view.part_at(&robot_id(), "arm").expect("arm");
    assert!(
        arm.applied_channel.is_some(),
        "commandable: an applied stream"
    );

    // Only embodiments declare parts; a malformed map is refused with the
    // path named; a merge policy that is Later is UNSUPPORTED.
    assert_eq!(
        status(
            twin.declare_parts(&raw_camera())
                .await
                .expect_err("observer")
        ),
        CtlStatus::Forbidden
    );
    let err = robot
        .declare_parts(r#"{"ft": {"kind": "x/ft", "hz": 1000, "fields": {"w": ["float32", [6]]}, "colour": 1}}"#)
        .await
        .expect_err("unknown key");
    assert!(err.to_string().contains("ft"), "{err}");
    assert_eq!(
        status(
            robot
                .declare_parts(
                    r#"{"arm": {"kind": "joints", "names": ["a"], "hz": 500,
                        "command": "position", "merge": "blend"}}"#
                )
                .await
                .expect_err("a merge policy that is Later")
        ),
        CtlStatus::Unsupported
    );

    // Leases need a commandable part inside the requester's binding.
    let policy = world.policy().await;
    policy.arm().await.expect("arm");
    assert_eq!(
        status(
            policy
                .grant_lease(ft.channel, None)
                .await
                .expect_err("sensor")
        ),
        CtlStatus::InvalidRequest
    );
    let unbound = world
        .join("unbound", vec![Role::Controller], PriorityClass::Model, 0)
        .await;
    unbound
        .wait_for("armed", WAIT, |s| s.mode == Mode::Armed)
        .await
        .expect("view");
    assert_eq!(
        status(
            unbound
                .grant_lease(arm.channel, None)
                .await
                .expect_err("no binding")
        ),
        CtlStatus::NotBound
    );

    // Detaching the owner removes its parts.
    robot.detach().await.expect("detach");
    twin.wait_for("parts removed", WAIT, |s| s.parts.is_empty())
        .await
        .expect("removed");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_participant_requiring_a_feature_the_session_lacks_is_refused() {
    let world = World::start().await;
    let mut config = world.config("future", vec![Role::Controller], PriorityClass::Model, 0);
    config.requires = vec!["merge.blend".into()];
    let err = Participant::attach(world.transport().await, config)
        .await
        .err()
        .expect("refused");
    assert_eq!(status(err), CtlStatus::Incompatible);
    assert!(world.host.snapshot().roster.is_empty());
}

/// A policy windowing the force/torque stream and commanding the arm.
const WINDOWED: &str = r#"{
    "hz": 50,
    "inputs": {"ft": {"kind": "x/ft", "at": "ft", "fields": {"wrench": ["float32", [18]]},
                      "history": 10}},
    "actions": {"arm": {"kind": "joints", "at": "arm", "hz": 100, "horizon": 5}}
}"#;

#[tokio::test(flavor = "multi_thread")]
async fn a_bound_policy_waits_for_its_streams_plans_and_receives_every_window_sample() {
    let world = World::start().await;
    let robot = world.robot().await;
    let policy = world
        .join("policy", vec![Role::Controller], PriorityClass::Model, 0)
        .await;
    policy
        .wait_for("the robot's parts", WAIT, |s| {
            s.part_at(&robot_id(), "ft").is_some()
        })
        .await
        .expect("parts");
    let binding = policy
        .bind(
            &Policy::parse(WINDOWED).expect("policy"),
            &RunOptions::named("windowed"),
        )
        .expect("bound");
    let ft_channel = channel(&policy, "ft");
    assert_eq!(binding.contracts[0].mode, ContractMode::History);
    policy.declare_bound(&binding).await.expect("declare");
    arm_seen_by(&policy, &[&robot]).await;
    let arm = channel(&policy, "arm");

    // Install before authority: no publisher has installed ft's plan yet.
    let err = policy.grant_lease(arm, None).await.expect_err("not ready");
    assert_eq!(status(err), CtlStatus::NotReady);
    let ft = robot.publisher("ft").await.expect("publisher");
    let view = policy
        .wait_for("ft's plan installed", WAIT, |s| {
            s.demand(ft_channel).is_some_and(|d| d.acked == d.revision)
        })
        .await
        .expect("acknowledged");
    let demand = view.demand(ft_channel).expect("demand");
    assert_eq!(
        demand.plan,
        Plan::Every,
        "a window: each sample as published"
    );
    assert_eq!(ft.plan(), (Some(demand.plan), demand.revision));
    let lease = policy.grant_lease(arm, None).await.expect("ready: granted");
    assert_eq!(lease.binding, binding.fingerprint);
    assert_eq!(
        view.binding(policy.id()).and_then(|b| b.grid(arm)),
        Some(Duration::from_millis(10)),
        "the action grid rides the binding record"
    );

    // Every sample of the window travels at once, one per message.
    let sub = policy
        .subscribe(
            &robot_id(),
            "ft",
            SubscribeOptions {
                window: 10,
                ..SubscribeOptions::default()
            },
        )
        .await
        .expect("subscribe");
    tokio::time::sleep(Duration::from_millis(200)).await;
    for i in 1..=25_u8 {
        ft.publish(robot.clock().now_ns(), ft_payload(f32::from(i)))
            .expect("publish")
            .expect("a window coalesces nothing");
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    wait_until("all 25 samples", || sub.stats().last_seq == 25).await;
    let stats = sub.stats();
    assert_eq!((stats.received, stats.lost), (25, 0));
    assert_eq!(ft.coalesced(), 0);
    assert_eq!(
        sub.window()
            .iter()
            .map(|r| r.sample.seq)
            .collect::<Vec<_>>(),
        (16..=25).collect::<Vec<_>>()
    );

    // The controller leaving withdraws its demand: every sample again.
    policy.detach().await.expect("detach");
    robot
        .wait_for("the demand withdrawn", WAIT, |s| {
            s.demand(ft_channel).is_none()
        })
        .await
        .expect("withdrawn");
    wait_until("the default plan", || ft.plan() == (None, 0)).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn samples_carry_seq_and_a_late_subscriber_warms_from_the_paged_history() {
    let world = World::start().await;
    let robot = world.robot().await;
    let ft = robot.publisher("ft").await.expect("publisher");
    let early = world
        .join("early", vec![Role::Observer], PriorityClass::Scripted, 0)
        .await;
    let early_ft = early
        .subscribe(
            &robot_id(),
            "ft",
            SubscribeOptions {
                window: 4,
                ..SubscribeOptions::default()
            },
        )
        .await
        .expect("subscribe");
    // Let the subscriptions (the host's history store too) propagate.
    tokio::time::sleep(Duration::from_millis(200)).await;
    for i in 1..=6_u8 {
        let seq = ft
            .publish(robot.clock().now_ns(), ft_payload(f32::from(i)))
            .expect("publish");
        assert_eq!(seq, Some(u64::from(i)));
    }
    wait_until("early subscriber has all six", || {
        early_ft.stats().last_seq == 6
    })
    .await;
    let stats = early_ft.stats();
    assert_eq!((stats.received, stats.lost), (6, 0));
    let latest = early_ft.latest().expect("latest");
    assert_eq!(latest.received.sample.epoch, world.host.snapshot().epoch);
    assert_eq!(latest.received.sample.payload, ft_payload(6.0));
    assert_eq!(
        early_ft
            .window()
            .iter()
            .map(|r| r.sample.seq)
            .collect::<Vec<_>>(),
        vec![3, 4, 5, 6],
        "a window of four"
    );
    // A value that does not fit the stride is refused at the publisher.
    assert!(matches!(
        ft.publish(0, Bytes::from(vec![0_u8; 8])),
        Err(SessionError::InvalidValue { .. })
    ));

    // The store pages by epoch, watermark and limit.
    let stream = early
        .part(&robot_id(), "ft")
        .expect("ft")
        .stream()
        .expect("stream");
    let epoch = world.host.snapshot().epoch;
    let pages = early
        .query_history(
            &stream,
            HistoryQuery {
                epoch,
                after_seq: 1,
                until_seq: Some(5),
                limit: Some(3),
            },
        )
        .await
        .expect("history");
    let seqs: Vec<u64> = pages
        .iter()
        .flat_map(|p| p.samples.iter().map(|s| s.seq))
        .collect();
    assert_eq!(seqs, vec![3, 4, 5], "the newest three up to the watermark");
    assert!(!pages[0].truncated && pages[0].gaps.is_empty());
    let other_epoch = early
        .query_history(
            &stream,
            HistoryQuery {
                epoch: epoch.next(),
                ..HistoryQuery::default()
            },
        )
        .await
        .expect("history");
    assert!(other_epoch.iter().all(|p| p.samples.is_empty()));

    // A late joiner is served the host's history, then live samples.
    let late = world
        .join("late", vec![Role::Observer], PriorityClass::Scripted, 0)
        .await;
    let late_ft = late
        .subscribe(
            &robot_id(),
            "ft",
            SubscribeOptions {
                window: 4,
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
    tokio::time::sleep(Duration::from_millis(100)).await;
    ft.publish(robot.clock().now_ns(), ft_payload(7.0))
        .expect("publish");
    wait_until("the live sample after history", || {
        late_ft.stats().last_seq == 7
    })
    .await;

    // Heartbeats report the newest seq seen per stream.
    let mut health = world.host.watch_health();
    tokio::time::timeout(
        WAIT,
        health.wait_for(|rows| {
            rows.iter().any(|r| {
                r.participant == *late.id() && r.report.last_seq_seen == vec![(stream.channel, 7)]
            })
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
            &robot_id(),
            "ft",
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

    // Raw samples on the stream's key, skipping seqs 3 and 4.
    let raw = world.transport().await;
    let key = robot.keys().data(robot.id(), "ft");
    let channel = ft.spec().channel;
    let epoch = world.host.snapshot().epoch;
    tokio::time::sleep(Duration::from_millis(200)).await;
    for seq in [1_u64, 2, 5, 6] {
        let sample = Sample {
            channel,
            epoch,
            seq,
            capture_ns: robot.clock().now_ns(),
            publish_ns: robot.clock().now_ns(),
            tick: 0,
            payload: ft_payload(0.5),
            applied: None,
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
    robot
        .declare_parts(&robot_map(
            r#""width": 64, "height": 48, "encoding": "jpeg", "quality": 80,
               "send": "latest(hz=30, max_mbps=4)""#,
        ))
        .await
        .expect("declare");
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
            &robot_id(),
            "wrist/cam",
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
    assert_eq!(cam_sub.spec().encoding, Encoding::Jpeg { quality: 80 });
    let images = robot
        .image_publisher("wrist/cam")
        .await
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
    assert!(sample.payload.len() < pixels.len() / 2, "compressed");
    // Offered at 50 Hz against a 30 Hz ceiling: the ceiling coalesces some
    // before they are encoded.
    let stats = images.stats();
    assert!(stats.sent >= 5, "{stats:?}");
    assert!(
        stats.coalesced > 0 && stats.encoded + stats.coalesced == 10,
        "{stats:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_virtual_session_steps_through_the_tick_barrier() {
    const TICK_NS: u64 = 1_000_000;
    let world = World::start_with(|config| {
        config.virtual_clock = Some(VirtualConfig {
            tick_ns: TICK_NS,
            latency_ticks: 1,
            barrier_timeout_ns: 5_000_000_000,
        });
    })
    .await;
    let robot = world.robot().await;
    let policy = world
        .join("policy", vec![Role::Controller], PriorityClass::Model, 0)
        .await;
    let arm = channel(&policy, "arm");
    // A 500 Hz policy on 1 ms ticks: due every second tick.
    policy
        .declare_binding(fingerprint_of(&policy, 0), "{}", vec![arm], 500.0)
        .await
        .expect("bind");
    arm_seen_by(&policy, &[&robot]).await;
    let view = policy.authority().expect("view");
    assert_eq!(
        (view.clock_mode, view.tick_ns),
        (ClockMode::Virtual, TICK_NS)
    );
    let lease = policy.grant_lease(arm, None).await.expect("grant");
    let chunks = policy.chunk_publisher(&lease).await.expect("publisher");
    let ft = robot.publisher("ft").await.expect("publisher");
    let mut robot_ticks = robot.virtual_ticks().await.expect("ticks");
    let mut policy_ticks = policy.virtual_ticks().await.expect("ticks");
    tokio::time::sleep(Duration::from_millis(200)).await;
    let ft_channel = ft.spec().channel;

    let controller = tokio::spawn(async move {
        let mut decided = Vec::new();
        while decided.len() < 3 {
            let tick = policy_ticks
                .wait_for(WAIT, |t| matches!(t, VirtualTick::Observe(_)))
                .await
                .expect("observe");
            let VirtualTick::Observe(state) = tick else {
                continue;
            };
            if !state.due.contains(policy.id()) {
                continue;
            }
            let k = state.tick;
            let t0 = i64::try_from((k + 1) * TICK_NS).expect("ns");
            let seq = chunks
                .send(ChunkDraft {
                    t0_ns: t0,
                    replace_from_ns: t0,
                    valid_until_ns: t0 + 50_000_000,
                    actions: Bytes::from(vec![0_u8; 24 * 4]),
                    tick: k,
                    ..ChunkDraft::default()
                })
                .expect("send");
            let reference = rlmesh_wire::ChunkRef {
                channel: arm,
                lease_token: lease.token,
                seq,
            };
            policy.decided(k, vec![reference]).await.expect("decided");
            decided.push((k, reference));
        }
        (policy, decided)
    });

    // The embodiment: publish, ready, wait for the step, repeat.
    let mut steps = Vec::new();
    for k in 0..6_u64 {
        let seq = ft
            .publish_sample(
                i64::try_from(k * TICK_NS).expect("ns"),
                k,
                ft_payload(k as f32),
                None,
            )
            .expect("publish")
            .expect("virtual mode sends every sample");
        robot
            .ready(
                k,
                vec![Watermark {
                    channel: ft_channel,
                    seq,
                }],
            )
            .await
            .expect("ready");
        let step = robot_ticks
            .wait_for(WAIT, |t| matches!(t, VirtualTick::Step(s) if s.tick == k))
            .await
            .expect("step");
        let VirtualTick::Step(step) = step else {
            unreachable!("matched a step");
        };
        steps.push((step.tick, step.chunks));
    }
    let (_policy, decided) = controller.await.expect("controller");
    assert_eq!(
        decided.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
        vec![0, 2, 4],
        "due on even ticks"
    );
    // With one tick of latency, the decision of tick k is delivered at k + 1.
    for (k, reference) in &decided {
        let (_, chunks) = &steps[usize::try_from(k + 1).expect("tick")];
        assert_eq!(chunks, &vec![*reference], "tick {}", k + 1);
    }
    assert!(steps[0].1.is_empty() && steps[2].1.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn publishers_and_controllers_report_their_counters_to_observers() {
    let world = World::start().await;
    let robot = world.robot().await;
    let ft = robot.publisher("ft").await.expect("ft publisher");
    let policy = world.policy().await;
    let observer = world
        .join("cockpit", vec![Role::Observer], PriorityClass::Scripted, 0)
        .await;
    let reports = Arc::new(std::sync::Mutex::new(Vec::new()));
    let _sub = observer
        .transport()
        .session(TrafficClass::Data)
        .declare_subscriber(observer.keys().reports())
        .callback({
            let reports = reports.clone();
            move |sample| {
                let report = codec::decode_report(&sample.payload().to_bytes()).expect("decode");
                reports.lock().expect("reports").push(report);
            }
        })
        .await
        .expect("subscribe");
    for i in 0..5 {
        ft.publish(robot.clock().now_ns(), ft_payload(i as f32))
            .expect("publish");
    }
    policy.set_run_source(Some(Arc::new(|_| rlmesh_wire::RunCounters {
        starts: 7,
        busy_skips: 2,
        ..Default::default()
    })));
    let ft_channel = channel(&robot, "ft");
    wait_until("both reports", || {
        let reports = reports.lock().expect("reports");
        let from_robot = reports.iter().any(|r| {
            r.participant == robot_id()
                && r.run.is_none()
                && r.streams
                    .iter()
                    .any(|s| s.channel == ft_channel && s.produced() == 5)
        });
        let from_policy = reports.iter().any(|r| {
            r.participant.as_str() == "policy"
                && r.run
                    .is_some_and(|run| (run.starts, run.busy_skips) == (7, 2))
        });
        from_robot && from_policy
    })
    .await;
    // A pure observer has nothing to report.
    assert!(
        !reports
            .lock()
            .expect("reports")
            .iter()
            .any(|r| r.participant.as_str() == "cockpit")
    );
}
