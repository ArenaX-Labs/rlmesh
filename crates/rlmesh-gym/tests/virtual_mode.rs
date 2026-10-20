//! The virtual clock's run loop (§11.3) on a session host in virtual mode:
//! a seeded latency model gives the same schedule twice, and a stalled
//! barrier ends the episode `interrupted: TIMEOUT`.

#![allow(clippy::unwrap_used)]

use std::time::Duration;

use bytes::Bytes;
use rlmesh_gym::virtual_mode::{Faults, Interrupted, LatencyModel, VirtualController, robot_tick};
use rlmesh_session::{
    ChunkDraft, HostConfig, Participant, ParticipantConfig, SessionHost, Transport,
    TransportConfig, VirtualConfig,
};
use rlmesh_wire::{
    ChunkRef, Fingerprint, Mode, ParticipantId, PriorityClass, Role, SessionId, TenantId, Watermark,
};

const TICK_NS: u64 = 1_000_000;
const WAIT: Duration = Duration::from_secs(10);
const ROBOT: &str = r#"{
    "arm": {"kind": "joints", "count": 6, "fields": ["pos"], "hz": 1000,
            "command": "position", "max_action_age_ms": 1000},
    "wrist/ft": {"kind": "wrench", "frame": "tool0", "hz": 1000}
}"#;

struct World {
    base: std::net::SocketAddr,
    tenant: TenantId,
    session: SessionId,
    _host: SessionHost,
}

impl World {
    async fn start(barrier_timeout: Duration) -> Self {
        let (transport, base) =
            rlmesh_session::transport::open_local_listeners("127.0.0.1".parse().unwrap())
                .await
                .unwrap();
        let tenant = TenantId::new("local").unwrap();
        let session = SessionId::new();
        let mut config = HostConfig::new(tenant.clone(), session);
        config.health_log = None;
        config.virtual_clock = Some(VirtualConfig {
            tick_ns: TICK_NS,
            // d is the controller's (the seeded model below).
            latency_ticks: 0,
            barrier_timeout_ns: u64::try_from(barrier_timeout.as_nanos()).unwrap(),
        });
        let host = SessionHost::start(transport, config).await.unwrap();
        Self {
            base,
            tenant,
            session,
            _host: host,
        }
    }

    async fn join(&self, id: &str, role: Role) -> Participant {
        let transport = Transport::open(
            &TransportConfig::per_class_tcp(Some("127.0.0.1:0".parse().unwrap()), Some(self.base))
                .unwrap(),
        )
        .await
        .unwrap();
        Participant::attach(
            transport,
            ParticipantConfig::new(
                self.tenant.clone(),
                self.session,
                ParticipantId::new(id).unwrap(),
                vec![role],
                PriorityClass::Model,
            ),
        )
        .await
        .unwrap()
    }
}

/// (tick, d, t0, the step its chunk is listed on).
type Schedule = Vec<(u64, u64, i64, u64)>;

async fn run(seed: u64, ticks: u64) -> Schedule {
    let world = World::start(Duration::from_secs(5)).await;
    let robot = world.join("robot", Role::Embodiment).await;
    robot.declare_parts(ROBOT).await.unwrap();
    let policy = world.join("policy", Role::Controller).await;
    let robot_id = ParticipantId::new("robot").unwrap();
    let arm = policy
        .wait_for("parts", WAIT, |s| !s.parts.is_empty())
        .await
        .map(|_| policy.part(&robot_id, "arm").unwrap().channel)
        .unwrap();
    let armed = robot.arm().await.unwrap();
    policy
        .wait_for("armed", WAIT, |s| s.epoch >= armed.epoch)
        .await
        .unwrap();
    // 500 Hz on 1 ms ticks: due every second tick.
    policy
        .declare_binding(Fingerprint::from(seed), "{}", vec![arm], 500.0)
        .await
        .unwrap();
    let lease = policy.grant_lease(arm, None).await.unwrap();
    let chunks = policy.chunk_publisher(&lease).await.unwrap();
    let ft = robot.publisher("wrist/ft").await.unwrap();
    let ft_channel = ft.spec().channel;
    let mut robot_ticks = robot.virtual_ticks().await.unwrap();
    let mut policy_ticks = policy.virtual_ticks().await.unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;

    let faults = Faults {
        seed,
        drop_ppm: 250_000,
        ..Faults::default()
    };
    let controller = VirtualController {
        latency: LatencyModel::Uniform {
            seed,
            min: 1,
            max: 3,
        },
        faults,
        tick_ns: TICK_NS,
    };
    let decisions = tokio::spawn(async move {
        let mut decided = Vec::new();
        while decided.len() < (ticks / 2) as usize {
            let mut sent = None;
            let (k, due) = controller
                .decide(&policy, &mut policy_ticks, WAIT, |decision| {
                    // The faulted inputs are the seeded verdicts on the
                    // watermarked samples.
                    let expected: Vec<_> = decision
                        .state
                        .watermarks
                        .iter()
                        .filter_map(|w| faults.fault(w.channel, w.seq).map(|f| (w.channel, f)))
                        .collect();
                    assert_eq!(decision.faulted, expected);
                    let t0 = decision.t0_ns;
                    let seq = chunks.send(ChunkDraft {
                        t0_ns: t0,
                        replace_from_ns: t0,
                        valid_until_ns: t0 + 50_000_000,
                        actions: Bytes::from(vec![0_u8; 24]),
                        tick: decision.state.tick,
                        ..ChunkDraft::default()
                    })?;
                    sent = Some((decision.latency_ticks, t0, seq));
                    Ok(vec![ChunkRef {
                        channel: arm,
                        lease_token: lease.token,
                        seq,
                    }])
                })
                .await
                .unwrap();
            if due && let Some((d, t0, seq)) = sent {
                decided.push((k, d, t0, seq));
            }
        }
        (policy, decided)
    });

    let mut listed: Vec<(u64, u64)> = Vec::new();
    for k in 0..ticks {
        let payload: Vec<u8> = (0..6).flat_map(|_| (k as f32).to_le_bytes()).collect();
        let seq = ft
            .publish_sample(
                i64::try_from(k * TICK_NS).unwrap(),
                k,
                Bytes::from([payload.clone(), vec![0; 8]].concat()),
                None,
            )
            .unwrap()
            .unwrap();
        let step = robot_tick(
            &robot,
            &mut robot_ticks,
            k,
            vec![Watermark {
                channel: ft_channel,
                seq,
            }],
            WAIT,
        )
        .await
        .unwrap();
        listed.extend(step.chunks.iter().map(|c| (c.seq, step.tick)));
    }
    let (_policy, decided) = decisions.await.unwrap();
    decided
        .into_iter()
        .map(|(k, d, t0, seq)| {
            let at = listed
                .iter()
                .find(|(s, _)| *s == seq)
                .map(|(_, tick)| *tick)
                .unwrap();
            (k, d, t0, at)
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_seeded_latency_model_gives_the_same_schedule_twice() {
    let first = run(42, 12).await;
    let second = run(42, 12).await;
    assert_eq!(first, second, "deterministic");
    assert_eq!(
        first.iter().map(|(k, ..)| *k).collect::<Vec<_>>(),
        [0, 2, 4, 6, 8, 10],
        "due every second tick"
    );
    for (k, d, t0, listed_at) in &first {
        assert!((1..=3).contains(d));
        assert_eq!(
            *t0,
            i64::try_from((k + d) * TICK_NS).unwrap(),
            "starts at k + d"
        );
        assert_eq!(
            listed_at, k,
            "the barrier never runs Step{{k}} before Decided{{k}}"
        );
    }
    let other = run(43, 12).await;
    assert_ne!(
        first.iter().map(|e| e.1).collect::<Vec<_>>(),
        other.iter().map(|e| e.1).collect::<Vec<_>>(),
        "another seed, another schedule"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stalled_barrier_ends_the_episode_interrupted_timeout() {
    let world = World::start(Duration::from_millis(300)).await;
    let robot = world.join("robot", Role::Embodiment).await;
    robot.declare_parts(ROBOT).await.unwrap();
    let policy = world.join("policy", Role::Controller).await;
    let robot_id = ParticipantId::new("robot").unwrap();
    let arm = policy
        .wait_for("parts", WAIT, |s| !s.parts.is_empty())
        .await
        .map(|_| policy.part(&robot_id, "arm").unwrap().channel)
        .unwrap();
    let armed = robot.arm().await.unwrap();
    policy
        .wait_for("armed", WAIT, |s| s.epoch >= armed.epoch)
        .await
        .unwrap();
    policy
        .declare_binding(Fingerprint::from(1), "{}", vec![arm], 500.0)
        .await
        .unwrap();
    // ACTIVE: a timeout then takes the session to SAFE.
    let _lease = policy.grant_lease(arm, None).await.unwrap();
    let mut robot_ticks = robot.virtual_ticks().await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
    // The policy is due at tick 0 and never decides.
    let err = robot_tick(
        &robot,
        &mut robot_ticks,
        0,
        Vec::new(),
        Duration::from_secs(2),
    )
    .await
    .expect_err("no step");
    assert!(matches!(err, Interrupted::Timeout { tick: 0, .. }), "{err}");
    assert!(err.to_string().contains("interrupted: TIMEOUT"));
    let safe = robot
        .wait_for("SAFE", WAIT, |s| s.mode == Mode::Safe)
        .await
        .unwrap();
    assert_eq!(safe.mode, Mode::Safe, "the host ended the run");
}
