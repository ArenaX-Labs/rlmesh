//! The device's robot against a real session host over Zenoh on loopback,
//! with this test as its transport (one session per traffic class): what
//! zenoh-pico does on a board.

// The transport's side: it reads the clock, as a board's does.
#![allow(clippy::disallowed_methods)]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bytes::Bytes;
use rlmesh_device::robot::{Action, Config, Fault, Phase, Robot};
use rlmesh_session::address::AddressKey;
use rlmesh_session::auth::{Credential, Device, HostIdentity};
use rlmesh_session::authority::DeviceAccess;
use rlmesh_session::crypto::StaticKeypair;
use rlmesh_session::crypto::{Rng, RngError};
use rlmesh_session::transport::{TrafficClass, Transport, TransportConfig, open_local_listeners};
use rlmesh_session::wire::{
    ChannelId, EdgeMode, Fingerprint, InputRange, Mode, ParticipantId, PriorityClass, Role,
    SessionId, TenantId,
};
use rlmesh_session::{
    ChunkDraft, HostConfig, Participant, ParticipantConfig, SessionHost, SubscribeOptions,
};
use tokio::sync::mpsc;
use zenoh::query::ConsolidationMode;

const MAP: &str = r#"{"eyes": {"kind": "joints", "names": ["pan", "tilt"], "fields": ["pos"],
  "hz": 50, "command": "position", "safe": "hold",
  "limits": {"pos": [[-0.6, 0.6], [-0.4, 0.4]]}}}"#;
const WAIT: Duration = Duration::from_secs(10);

/// SplitMix64, seeded from the clock: the handshake's ephemeral keys.
struct Mix(u64);

impl Rng for Mix {
    fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), RngError> {
        for chunk in dest.chunks_mut(8) {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            chunk.copy_from_slice(&z.to_le_bytes()[..chunk.len()]);
        }
        Ok(())
    }
}

fn seeded() -> Mix {
    let t = std::time::SystemTime::UNIX_EPOCH
        .elapsed()
        .expect("clock")
        .as_nanos();
    Mix(t as u64)
}

fn pos(pan: f32, tilt: f32) -> [u8; 8] {
    let mut v = [0; 8];
    v[..4].copy_from_slice(&pan.to_le_bytes());
    v[4..].copy_from_slice(&tilt.to_le_bytes());
    v
}

enum Event {
    Sample(u8, String, Vec<u8>),
    Reply(u32, Vec<u8>),
    Done(u32),
}

struct World {
    tenant: TenantId,
    session: SessionId,
    base: std::net::SocketAddr,
    identity: Arc<HostIdentity>,
    host: SessionHost,
}

impl World {
    async fn start() -> Self {
        Self::start_with(Vec::new()).await
    }

    /// A host that knows `devices` besides `local`.
    async fn start_with(devices: Vec<Device>) -> Self {
        let (transport, base) = open_local_listeners("127.0.0.1".parse().expect("ip"))
            .await
            .expect("host transport");
        let tenant = TenantId::new("test").expect("tenant");
        let session = SessionId::new();
        let mut config = HostConfig::new(tenant.clone(), session);
        config.devices = devices;
        let identity = config.identity.clone();
        let host = SessionHost::start(transport, config).await.expect("host");
        Self {
            tenant,
            session,
            base,
            identity,
            host,
        }
    }

    async fn transport(&self) -> Transport {
        Transport::open(
            &TransportConfig::per_class_tcp(
                Some("127.0.0.1:0".parse().expect("addr")),
                Some(self.base),
            )
            .expect("config"),
        )
        .await
        .expect("transport")
    }

    fn robot(&self, key: AddressKey) -> Robot {
        self.robot_as(key.credential())
    }

    fn robot_as(&self, credential: Credential) -> Robot {
        Robot::new(Config {
            tenant: self.tenant.clone(),
            session: self.session,
            name: ParticipantId::new("eyes").expect("id"),
            credential,
            robot_json: MAP.to_owned(),
            heartbeat_ms: 50,
        })
        .expect("a valid map")
    }

    async fn controller(&self) -> Participant {
        let mut config = ParticipantConfig::new(
            self.tenant.clone(),
            self.session,
            ParticipantId::new("policy").expect("id"),
            vec![Role::Controller],
            PriorityClass::Model,
        );
        config.heartbeat_period = Duration::from_millis(20);
        config.credential = Some(self.host.credential());
        Participant::attach(self.transport().await, config)
            .await
            .expect("controller")
    }
}

/// What the board's loop saw: the actions its edge dispatched.
#[derive(Default)]
struct Seen {
    actions: Vec<(EdgeMode, Vec<u8>)>,
    /// The last tick's mode.
    mode: Option<EdgeMode>,
    /// Who the e-stop latch names, while latched.
    latched_by: Option<String>,
    published: u32,
}

/// Runs `robot` over a fresh transport to the world's host until `done`
/// holds or `limit` passes, carrying out its actions as a board would. Once
/// it runs, it ticks the part each loop and publishes a sample every 20 ms.
async fn drive(
    world: &World,
    mut robot: Robot,
    limit: Duration,
    seen: Arc<Mutex<Seen>>,
    mut done: impl FnMut(&Robot) -> bool + Send,
) -> Robot {
    let transport = world.transport().await;
    let (tx, mut rx) = mpsc::unbounded_channel();
    let mut subscribers = Vec::new();
    let mut tokens = Vec::new();
    let mut rng = seeded();
    let start = Instant::now();
    let now_us = || u64::try_from(start.elapsed().as_micros()).expect("µs");
    let mut queries = BTreeMap::new();
    let mut next_sample = 0;
    while start.elapsed() < limit && !done(&robot) {
        if robot.phase() == Phase::Running {
            let now = now_us();
            if let Some(tick) = robot.tick(0, now) {
                let mut seen = seen.lock().expect("seen");
                seen.mode = Some(tick.tick.mode);
                if let Some(action) = &tick.tick.action {
                    seen.actions.push((tick.tick.mode, action.value.to_vec()));
                }
            }
            seen.lock().expect("seen").latched_by = robot.estop_reason().map(|r| r.by.clone());
            if now >= next_sample {
                next_sample = now + 20_000;
                if robot.publish(0, now, &pos(0.1, 0.2), now).is_ok() {
                    seen.lock().expect("seen").published += 1;
                }
            }
        }
        for action in robot.poll(now_us()) {
            let class = |c: u8| TrafficClass::ALL[usize::from(c)];
            match action {
                Action::Subscribe { class: c, sub, key } => {
                    let tx = tx.clone();
                    let subscriber = transport
                        .session(class(c))
                        .declare_subscriber(key)
                        .callback(move |sample| {
                            let bytes = sample.payload().to_bytes().into_owned();
                            let key = sample.key_expr().as_str().to_owned();
                            let _ = tx.send(Event::Sample(sub, key, bytes));
                        })
                        .await
                        .expect("subscribe");
                    subscribers.push(subscriber);
                }
                Action::Token { class: c, key } => {
                    let token = transport.session(class(c)).liveliness().declare_token(key);
                    tokens.push(token.await.expect("token"));
                }
                Action::Query {
                    class: c,
                    id,
                    key,
                    payload,
                } => {
                    let (tx, zenoh) = (tx.clone(), transport.session(class(c)).clone());
                    queries.insert(
                        id,
                        tokio::spawn(async move {
                            let mut get = zenoh
                                .get(key)
                                .consolidation(ConsolidationMode::None)
                                .timeout(Duration::from_millis(u64::from(
                                    rlmesh_device::robot::QUERY_MS,
                                )));
                            if let Some(payload) = payload {
                                get = get.payload(payload);
                            }
                            if let Ok(replies) = get.await {
                                while let Ok(reply) = replies.recv_async().await {
                                    if let Ok(sample) = reply.result() {
                                        let bytes = sample.payload().to_bytes().into_owned();
                                        let _ = tx.send(Event::Reply(id, bytes));
                                    }
                                }
                            }
                            let _ = tx.send(Event::Done(id));
                        }),
                    );
                }
                Action::Publish {
                    class: c,
                    key,
                    payload,
                } => {
                    transport
                        .session(class(c))
                        .put(key, payload)
                        .await
                        .expect("publish");
                }
            }
        }
        let event = tokio::time::timeout(Duration::from_millis(1), rx.recv()).await;
        let mut next = match event {
            Ok(Some(event)) => Some(event),
            _ => None,
        };
        while let Some(event) = next {
            match event {
                Event::Sample(sub, key, mut bytes) => {
                    robot.sample(sub, &key, &mut bytes, now_us());
                }
                Event::Reply(id, bytes) => robot.reply(id, &bytes, now_us(), &mut rng),
                Event::Done(id) => {
                    queries.remove(&id);
                    robot.query_done(id, now_us());
                }
            }
            next = rx.try_recv().ok();
        }
    }
    for (_, query) in queries {
        query.abort();
    }
    robot
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_board_joins_declares_its_parts_and_heartbeats() {
    let world = World::start().await;
    let robot = world.robot(AddressKey::of(&world.identity));
    // Joined, then 5.5 s more: past the time it would count the host lost
    // without echoes.
    let mut running = None;
    let robot = drive(
        &world,
        robot,
        Duration::from_secs(15),
        Arc::default(),
        |r| {
            if r.phase() == Phase::Running {
                running.get_or_insert_with(Instant::now);
            }
            running.is_some_and(|at| at.elapsed() > Duration::from_millis(5500))
        },
    )
    .await;
    assert!(running.is_some(), "never joined: {:?}", robot.fault());
    assert_eq!(robot.phase(), Phase::Running, "fault: {:?}", robot.fault());
    assert!(robot.generation() >= 1);
    // Heartbeats are echoed, sealed both ways.
    assert!(robot.echoes() >= 20, "{} echoes", robot.echoes());
    // It asked for its device view at attach, and the host publishes it.
    assert!(robot.view().is_some(), "the board holds its device view");
    assert!(
        robot.views() > 0,
        "the host publishes the board's signed view"
    );

    let snapshot = world.host.snapshot();
    assert!(
        snapshot.roster.iter().any(|p| p.id.as_str() == "eyes"),
        "the host lists the board"
    );
    assert!(
        snapshot.parts.iter().any(|p| p.path.as_str() == "eyes"),
        "the host has its parts"
    );
    // Its clock is mapped from the heartbeats' timing: an edge needs it.
    let id = ParticipantId::new("eyes").expect("id");
    assert!(snapshot.clock(&id).is_some(), "the board's clock is mapped");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_controller_drives_the_board_and_reads_its_samples() {
    let world = World::start().await;
    let seen = Arc::new(Mutex::new(Seen::default()));
    let board = {
        let robot = world.robot(AddressKey::of(&world.identity));
        let world = &world;
        let seen = seen.clone();
        async move { drive(world, robot, Duration::from_secs(20), seen, |_| false).await }
    };
    let controller = async {
        let id = ParticipantId::new("eyes").expect("id");
        let policy = world.controller().await;
        let snapshot = policy
            .wait_for("the board, declared and mapped", WAIT, |s| {
                s.parts.iter().any(|p| p.owner == id) && s.clock(&id).is_some()
            })
            .await
            .expect("board");
        let eyes: ChannelId = snapshot
            .parts
            .iter()
            .find(|p| p.owner == id)
            .expect("eyes")
            .channel;
        let samples = policy
            .subscribe(&id, "eyes", SubscribeOptions::default())
            .await
            .expect("subscribe");
        policy
            .declare_binding(Fingerprint::from(1), "{}", vec![eyes], 0.0)
            .await
            .expect("binding");
        policy.arm().await.expect("arm");
        let lease = policy.grant_lease(eyes, None).await.expect("lease");
        let tx = policy.chunk_publisher(&lease).await.expect("publisher");
        // The board installs its k_cmd from the grant's delivery.
        tokio::time::sleep(Duration::from_millis(300)).await;
        let target = pos(0.25, -0.1);
        let deadline = Instant::now() + WAIT;
        while Instant::now() < deadline {
            let now = policy.session_now_ns();
            tx.send(ChunkDraft {
                based_on: vec![InputRange::one(ChannelId(99), 1)],
                t0_ns: now + 1_000_000,
                dt_ns: 20_000_000,
                replace_from_ns: now,
                valid_until_ns: now + 200_000_000,
                actions: Bytes::from([target; 5].concat()),
                tick: 0,
            })
            .expect("send");
            tokio::time::sleep(Duration::from_millis(50)).await;
            let driven = seen
                .lock()
                .expect("seen")
                .actions
                .iter()
                .any(|(mode, value)| *mode == EdgeMode::Driving && value[..] == target);
            if driven && samples.latest().is_some() {
                break;
            }
        }
        let received = samples.latest().expect("a sample from the board");
        assert_eq!(received.received.sample.payload[..], pos(0.1, 0.2));
        let host = world.host.snapshot();
        assert_ne!(host.mode, Mode::Safe, "{:?}", host.mode_reason);

        // The controller's e-stop signal: the board's edge latches it
        // directly (ESTOP_SIGNAL, by the controller) and stops driving.
        policy.estop("test").await.expect("e-stop");
        let deadline = Instant::now() + WAIT;
        loop {
            let (mode, by) = {
                let seen = seen.lock().expect("seen");
                (seen.mode, seen.latched_by.clone())
            };
            if mode == Some(EdgeMode::Estop) && by.as_deref() == Some("policy") {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "the board latched: {mode:?} by {by:?}"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        policy
            .wait_for("the host in ESTOP", WAIT, |s| s.mode == Mode::Estop)
            .await
            .expect("ESTOP");
        (target, received)
    };
    // The board loop runs until the controller is done: race them.
    let (target, _) = tokio::select! {
        result = controller => result,
        robot = board => panic!("the board's loop ended first: {:?}", robot.fault()),
    };
    let seen = seen.lock().expect("seen");
    assert!(
        seen.actions
            .iter()
            .any(|(mode, value)| *mode == EdgeMode::Driving && value[..] == target),
        "the board dispatched the controller's action: {} ticks with an action",
        seen.actions.len()
    );
    assert!(seen.published > 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_board_with_another_hosts_key_is_refused() {
    let world = World::start().await;
    let other = HostIdentity::generate(&mut seeded()).expect("identity");
    let robot = world.robot(AddressKey::of(&other));
    let robot = drive(
        &world,
        robot,
        Duration::from_secs(10),
        Arc::default(),
        |r| r.fault().is_some(),
    )
    .await;
    assert_eq!(robot.phase(), Phase::Retry);
    assert!(
        matches!(robot.fault(), Some(Fault::Refused(..) | Fault::Handshake)),
        "{:?}",
        robot.fault()
    );
    assert!(
        !world
            .host
            .snapshot()
            .roster
            .iter()
            .any(|p| p.id.as_str() == "eyes")
    );
}

/// A board provisioned for the host (tether spec §14) joins with its own
/// device key, under the access the daemon enrolled it with; an unenrolled
/// key may do nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_provisioned_board_joins_with_its_device_key() {
    let private = [0x42_u8; 32];
    let board = Device {
        id: "board-eyes".into(),
        public: rlmesh_device::provision::public_key(&private),
        access: DeviceAccess::new(Some(PriorityClass::Model)),
    };
    let world = World::start_with(vec![board]).await;
    let host_static = *world.identity.static_public();
    let credential = |key: [u8; 32]| {
        let mut key = key;
        Credential::new(StaticKeypair::from_private(&mut key), host_static)
    };
    let robot = world.robot_as(credential(private));
    let robot = drive(
        &world,
        robot,
        Duration::from_secs(10),
        Arc::default(),
        |r| r.phase() == Phase::Running,
    )
    .await;
    assert_eq!(robot.phase(), Phase::Running, "fault: {:?}", robot.fault());
    let snapshot = world.host.snapshot();
    let eyes = snapshot
        .roster
        .iter()
        .find(|p| p.id.as_str() == "eyes")
        .expect("the host lists the board");
    assert_eq!(eyes.device, "board-eyes");

    let world = World::start().await;
    let robot = world.robot_as(credential(private));
    let robot = drive(
        &world,
        robot,
        Duration::from_secs(10),
        Arc::default(),
        |r| r.fault().is_some(),
    )
    .await;
    assert_ne!(robot.phase(), Phase::Running, "an unenrolled key joined");
    assert!(
        matches!(robot.fault(), Some(Fault::Refused(..) | Fault::Handshake)),
        "{:?}",
        robot.fault()
    );
}
