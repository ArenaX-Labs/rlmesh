//! A session host for `mise run test:esp32:host`: the default session of
//! tenant `local` on free loopback ports, printing its address with its key
//! (as `rlmesh host` does), until killed.
//!
//! Its links are not gated. A loopback embedded host gates them (Zenoh
//! `usrpwd`, keyed by the session), which zenoh-pico can't present; a board
//! on the LAN joins `rlmesh host --insecure` instead, whose LAN links are
//! plain. Session traffic is authenticated and sealed by the handshake
//! either way. For tests only.

// Test infrastructure owns the wall clock, as the board platform does.
#![allow(clippy::disallowed_methods, clippy::print_stdout)]

use std::io::Write;
use std::time::{Duration, Instant};

use bytes::Bytes;
use rlmesh_session::transport::{Transport, TransportConfig};
use rlmesh_session::wire::{
    ChannelId, Fingerprint, InputRange, ParticipantId, PriorityClass, Role,
};
use rlmesh_session::{ChunkDraft, Participant, ParticipantConfig, SubscribeOptions};

use rlmesh_session::address::{AddressKey, DEFAULT_SESSION, TENANT, session_id};
use rlmesh_session::transport::open_local_listeners;
use rlmesh_session::wire::TenantId;
use rlmesh_session::{HostConfig, SessionHost};

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() {
    let (transport, base) = open_local_listeners("127.0.0.1".parse().expect("ip"))
        .await
        .expect("listeners");
    let config = HostConfig::new(
        TenantId::new(TENANT).expect("tenant"),
        session_id(DEFAULT_SESSION),
    );
    let key = AddressKey::of(&config.identity);
    let host = SessionHost::start(transport, config).await.expect("host");
    println!("{base}#{key}");
    std::io::stdout().flush().expect("stdout");
    let transport = Transport::open(
        &TransportConfig::per_class_tcp(Some("127.0.0.1:0".parse().expect("ip")), Some(base))
            .expect("config"),
    )
    .await
    .expect("controller transport");
    let mut config = ParticipantConfig::new(
        TenantId::new(TENANT).expect("tenant"),
        session_id(DEFAULT_SESSION),
        ParticipantId::new("policy").expect("id"),
        vec![Role::Controller],
        PriorityClass::Model,
    );
    config.heartbeat_period = Duration::from_millis(20);
    config.credential = Some(host.credential());
    let policy = Participant::attach(transport, config)
        .await
        .expect("controller");
    let owner = ParticipantId::new("eyes").expect("id");
    let snapshot = policy
        .wait_for("board declared and mapped", Duration::from_secs(15), |s| {
            s.parts.iter().any(|p| p.owner == owner) && s.clock(&owner).is_some()
        })
        .await
        .expect("board");
    let channel = snapshot
        .parts
        .iter()
        .find(|p| p.owner == owner)
        .expect("part")
        .channel;
    let samples = policy
        .subscribe(&owner, "eyes", SubscribeOptions::default())
        .await
        .expect("samples");
    policy
        .declare_binding(Fingerprint::from(1), "{}", vec![channel], 0.0)
        .await
        .expect("binding");
    policy.arm().await.expect("arm");
    let lease = policy.grant_lease(channel, None).await.expect("lease");
    let publisher = policy.chunk_publisher(&lease).await.expect("publisher");
    tokio::time::sleep(Duration::from_millis(300)).await;
    // Advance only after the board echoes the applied action as its state.
    // Two distinct targets catch a transport that replays only its first chunk.
    let targets = [[0.25_f32, -0.1_f32], [-0.2_f32, 0.15_f32]].map(|pos| {
        pos.into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>()
    });
    let mut stage = 0;
    let deadline = Instant::now() + Duration::from_secs(12);
    loop {
        assert!(
            Instant::now() < deadline || stage == 2,
            "board did not apply both command targets"
        );
        let target = &targets[stage.min(1)];
        let now = policy.session_now_ns();
        publisher
            .send(ChunkDraft {
                based_on: vec![InputRange::one(ChannelId(99), 1)],
                t0_ns: now + 1_000_000,
                dt_ns: 20_000_000,
                replace_from_ns: now,
                valid_until_ns: now + 200_000_000,
                actions: Bytes::from(target.repeat(5)),
                tick: 0,
            })
            .expect("send command chunk");
        tokio::time::sleep(Duration::from_millis(50)).await;
        if stage < 2
            && samples
                .latest()
                .is_some_and(|s| s.received.sample.payload[..] == targets[stage])
        {
            stage += 1;
            if stage == 2 {
                println!(
                    "commands confirmed: both applied targets returned through board transport"
                );
                std::io::stdout().flush().expect("stdout");
            }
        }
    }
}
