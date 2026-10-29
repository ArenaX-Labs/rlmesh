//! A board waiting in its home's lobby (scene design §3), with this program
//! as its transport: rehearses the lobby against a real `rlmesh daemon`
//! without a board. It prints each phase as it reaches it, until killed.
//!
//! ```sh
//! cargo run -p rlmesh-device --example lobby_board -- <home host:port> <local.key> \
//!     [<satellite host:port>=<its local.key> ...]
//! ```
//!
//! It holds the daemon's `local.key`, so it attaches as device `local` and
//! presents the loopback link gate; a board on the LAN holds its own key.
//! A run on a satellite is followed there: its key file only opens that
//! daemon's loopback gate (the run's own host key comes with the join).

#![allow(clippy::print_stdout, clippy::disallowed_methods)]

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use rlmesh_device::robot::{Action, Config, Phase, Robot};
use rlmesh_session::address::{AddressKey, TENANT};
use rlmesh_session::crypto::{Rng, RngError};
use rlmesh_session::lobby::home_lobby;
use rlmesh_session::transport::TrafficClass;
use rlmesh_session::wire::{ParticipantId, SessionId, TenantId};
use tokio::sync::mpsc;
use zenoh::query::ConsolidationMode;

const MAP: &str = r#"{"eyes": {"kind": "joints", "names": ["pan", "tilt"], "fields": ["pos"],
  "hz": 50, "command": "position", "safe": "hold"}}"#;

struct Urandom(std::fs::File);

impl Rng for Urandom {
    fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), RngError> {
        std::io::Read::read_exact(&mut self.0, dest).map_err(|_| RngError)
    }
}

enum Event {
    Sample(u8, String, Vec<u8>),
    Reply(u32, Vec<u8>),
    Done(u32),
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() {
    let mut args = std::env::args().skip(1);
    let (Some(home), Some(key)) = (args.next(), args.next()) else {
        println!("usage: lobby_board <home host:port> <local.key>");
        std::process::exit(2);
    };
    let key: AddressKey = std::fs::read_to_string(&key)
        .expect("read local.key")
        .trim()
        .parse()
        .expect("a key");
    // The loopback gates' keys, by base address.
    let mut gates: BTreeMap<String, AddressKey> = BTreeMap::new();
    gates.insert(home.clone(), key.clone());
    for extra in args {
        let (base, file) = extra.split_once('=').expect("<host:port>=<local.key>");
        let gate = std::fs::read_to_string(file).expect("read a satellite's local.key");
        gates.insert(base.into(), gate.trim().parse().expect("a key"));
    }
    let mut transport = link(&home, &gates).await;
    let mut linked = home.clone();
    let mut robot = Robot::new(Config {
        tenant: TenantId::new(TENANT).expect("tenant"),
        session: SessionId::new(),
        name: ParticipantId::new("eyes").expect("id"),
        credential: key.credential(),
        robot_json: MAP.to_owned(),
        heartbeat_ms: 50,
        lobby: true,
        homes: vec![rlmesh_device::address::host_port(&home).expect("host:port")],
        home_id: String::new(),
    })
    .expect("a valid map");

    let (tx, mut rx) = mpsc::unbounded_channel();
    let mut subscribers = Vec::new();
    let mut tokens = Vec::new();
    let mut rng = Urandom(std::fs::File::open("/dev/urandom").expect("/dev/urandom"));
    let start = Instant::now();
    let now_us = || u64::try_from(start.elapsed().as_micros()).expect("µs");
    let mut queries = BTreeMap::new();
    let mut last = None;
    loop {
        let seen = (
            robot.phase(),
            robot.in_run(),
            robot.fault().cloned(),
            robot.left().cloned(),
        );
        if last.as_ref() != Some(&seen) {
            let (phase, in_run, fault, left) = &seen;
            let place = if *in_run { "run" } else { "lobby" };
            let fault = fault
                .as_ref()
                .map(|f| format!(" (last: {f:?})"))
                .unwrap_or_default();
            let left = left
                .as_ref()
                .map(|l| format!(" (left a run: {l:?})"))
                .unwrap_or_default();
            println!(
                "{:7.3} s  {place}: {phase:?}{fault}{left}",
                start.elapsed().as_secs_f64()
            );
            last = Some(seen);
        }
        if robot.phase() == Phase::Running
            && let Some(eyes) = robot.part("eyes")
        {
            let _ = robot.tick(eyes, now_us());
        }
        for action in robot.poll(now_us()) {
            let class = |c: u8| TrafficClass::ALL[usize::from(c)];
            match action {
                Action::Reset => {
                    subscribers.clear();
                    tokens.clear();
                    // A run elsewhere, or another home address: a link there.
                    let (host, port) = robot.host();
                    let target = format!("{host}:{port}");
                    if target != linked {
                        println!("          linking {target}");
                        transport = link(&target, &gates).await;
                        linked = target;
                    }
                }
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
                    timeout_ms,
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
                                .timeout(Duration::from_millis(u64::from(timeout_ms)));
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
                    let _ = transport.session(class(c)).put(key, payload).await;
                }
            }
        }
        let event = tokio::time::timeout(Duration::from_millis(2), rx.recv()).await;
        let mut next = match event {
            Ok(Some(event)) => Some(event),
            _ => None,
        };
        while let Some(event) = next {
            match event {
                Event::Sample(sub, key, mut bytes) => robot.sample(sub, &key, &mut bytes, now_us()),
                Event::Reply(id, bytes) => robot.reply(id, &bytes, now_us(), &mut rng),
                Event::Done(id) => {
                    queries.remove(&id);
                    robot.query_done(id, now_us());
                }
            }
            next = rx.try_recv().ok();
        }
    }
}

/// A transport to the session host at `base`, through its loopback gate when
/// `gates` holds its key.
async fn link(base: &str, gates: &BTreeMap<String, AddressKey>) -> rlmesh_session::Transport {
    let address = rlmesh_session::address::SessionAddress {
        key: gates.get(base).cloned(),
        ..home_lobby(Some(base)).expect("a base address")
    };
    address
        .link(
            tokio::time::Instant::now() + Duration::from_secs(10),
            Duration::from_secs(10),
        )
        .await
        .expect("a link")
}
