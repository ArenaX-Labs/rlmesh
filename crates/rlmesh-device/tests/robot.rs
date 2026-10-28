//! The device's robot against a real session host over Zenoh on loopback,
//! with this test as its transport: what zenoh-pico does on a board.

// The transport's side: it reads the clock, as a board's does.
#![allow(clippy::disallowed_methods)]

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use rlmesh_device::robot::{Action, Config, Fault, Phase, Robot, Sub};
use rlmesh_session::auth::Credential;
use rlmesh_session::crypto::{Rng, RngError};
use rlmesh_session::transport::{Transport, TransportConfig, open_local_listeners};
use rlmesh_session::wire::ids::{ParticipantId, SessionId, TenantId};
use rlmesh_session::{HostConfig, SessionHost};
use tokio::sync::mpsc;
use zenoh::query::ConsolidationMode;

const MAP: &str = r#"{"eyes": {"kind": "joints", "names": ["pan", "tilt"], "fields": ["pos"],
  "hz": 50, "command": "position", "safe": "hold",
  "limits": {"pos": [[-0.6, 0.6], [-0.4, 0.4]]}}}"#;

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

enum Event {
    Sample(Sub, Vec<u8>),
    Reply(u32, Vec<u8>),
    Done(u32),
}

struct World {
    tenant: TenantId,
    session: SessionId,
    base: std::net::SocketAddr,
    host: SessionHost,
}

impl World {
    async fn start() -> Self {
        let (transport, base) = open_local_listeners("127.0.0.1".parse().expect("ip"))
            .await
            .expect("host transport");
        let tenant = TenantId::new("test").expect("tenant");
        let session = SessionId::new();
        let host = SessionHost::start(transport, HostConfig::new(tenant.clone(), session))
            .await
            .expect("host");
        Self {
            tenant,
            session,
            base,
            host,
        }
    }

    fn config(&self, credential: Credential) -> Config {
        Config {
            tenant: self.tenant.clone(),
            session: self.session,
            name: ParticipantId::new("eyes").expect("id"),
            credential,
            robot_json: MAP.to_owned(),
            heartbeat_ms: 50,
        }
    }
}

/// Runs `robot` over a fresh transport to the world's host until `done`
/// holds or `limit` passes, carrying out its actions as a board would.
async fn drive(
    world: &World,
    robot: &mut Robot,
    limit: Duration,
    mut done: impl FnMut(&Robot) -> bool,
) {
    let transport = Transport::open(
        &TransportConfig::per_class_tcp(
            Some("127.0.0.1:0".parse().expect("addr")),
            Some(world.base),
        )
        .expect("config"),
    )
    .await
    .expect("transport");
    let zenoh = transport.control().clone();
    let (tx, mut rx) = mpsc::unbounded_channel();
    let mut subscribers = Vec::new();
    let mut tokens = Vec::new();
    let mut rng = seeded();
    let start = Instant::now();
    let now_us = || u64::try_from(start.elapsed().as_micros()).expect("µs");
    let mut queries = BTreeMap::new();
    while start.elapsed() < limit && !done(robot) {
        for action in robot.poll(now_us()) {
            match action {
                Action::Subscribe { sub, key } => {
                    let tx = tx.clone();
                    let subscriber = zenoh
                        .declare_subscriber(key)
                        .callback(move |sample| {
                            let bytes = sample.payload().to_bytes().into_owned();
                            let _ = tx.send(Event::Sample(sub, bytes));
                        })
                        .await
                        .expect("subscribe");
                    subscribers.push(subscriber);
                }
                Action::Token { key } => {
                    tokens.push(zenoh.liveliness().declare_token(key).await.expect("token"));
                }
                Action::Query { id, key, payload } => {
                    let (tx, zenoh) = (tx.clone(), zenoh.clone());
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
                Action::Publish { key, payload } => {
                    zenoh.put(key, payload).await.expect("publish");
                }
            }
        }
        let event = tokio::time::timeout(Duration::from_millis(5), rx.recv()).await;
        let mut next = match event {
            Ok(Some(event)) => Some(event),
            _ => None,
        };
        while let Some(event) = next {
            match event {
                Event::Sample(sub, mut bytes) => robot.sample(sub, &mut bytes, now_us()),
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
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_board_joins_declares_its_parts_and_heartbeats() {
    let world = World::start().await;
    let mut robot = Robot::new(world.config(world.host.credential()));
    // Joined, then 3.5 s more: past the time it would count the host lost
    // without echoes.
    let mut running = None;
    drive(&world, &mut robot, Duration::from_secs(15), |r| {
        if r.phase() == Phase::Running {
            running.get_or_insert_with(Instant::now);
        }
        running.is_some_and(|at| at.elapsed() > Duration::from_millis(3500))
    })
    .await;
    assert!(running.is_some(), "never joined: {:?}", robot.fault());
    assert_eq!(robot.phase(), Phase::Running, "fault: {:?}", robot.fault());
    assert!(robot.generation() >= 1);
    // Heartbeats are echoed, sealed both ways.
    assert!(robot.echoes() >= 20, "{} echoes", robot.echoes());
    assert!(robot.snapshot().is_some());

    let snapshot = world.host.snapshot();
    assert!(
        snapshot.roster.iter().any(|p| p.id.as_str() == "eyes"),
        "the host lists the board"
    );
    assert!(
        snapshot
            .parts
            .iter()
            .any(|p| p.path.to_string().contains("eyes")),
        "the host has its parts"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_board_with_another_hosts_key_is_refused() {
    let world = World::start().await;
    let host_static = *world.host.credential().host_static();
    let mut robot = Robot::new(world.config(Credential::local(&[7; 32], host_static)));
    drive(&world, &mut robot, Duration::from_secs(10), |r| {
        r.fault().is_some()
    })
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
