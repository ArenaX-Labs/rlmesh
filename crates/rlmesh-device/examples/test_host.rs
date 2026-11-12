//! A session host for `mise run test:esp32:host`: the default session of
//! tenant `local` on free loopback ports, printing its address with its key
//! (as `rlmesh host` does), until killed. With `udp`, it relays the board's
//! realtime parts over UDP on the realtime class's port number, as
//! `rlmesh host --board-data udp` does. With `soak` after that, it drives
//! every commanded part of the board for as long as it runs, granting their
//! leases again whenever the board attaches anew, for the board's soak.
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
use rlmesh_session::host::DatagramRelay;
use rlmesh_session::transport::open_local_listeners;
use rlmesh_session::wire::TenantId;
use rlmesh_session::{HostConfig, SessionHost};

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() {
    let (transport, base) = open_local_listeners("127.0.0.1".parse().expect("ip"))
        .await
        .expect("listeners");
    let mut config = HostConfig::new(
        TenantId::new(TENANT).expect("tenant"),
        session_id(DEFAULT_SESSION),
    );
    let soak = std::env::args().nth(2).as_deref() == Some("soak");
    if std::env::args().nth(1).as_deref() == Some("udp") {
        let at = std::net::SocketAddr::new(base.ip(), base.port() + 1);
        config.datagram = Some(DatagramRelay::bind(at).await.expect("udp"));
    }
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
    if soak {
        drive_forever(&policy, &owner).await;
    }
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

/// Drives every commanded part of the board, 30 chunks a second of two
/// actions each, with leases granted again whenever the last ones ended (the
/// board attached anew), never giving up.
async fn drive_forever(policy: &Participant, owner: &ParticipantId) -> ! {
    loop {
        tokio::time::sleep(Duration::from_millis(200)).await;
        let Some(snapshot) = policy.authority() else {
            continue;
        };
        let parts: Vec<_> = snapshot
            .parts
            .iter()
            .filter(|p| &p.owner == owner && p.command.is_some())
            .cloned()
            .collect();
        if parts.is_empty() || snapshot.clock(owner).is_none() {
            continue;
        }
        let channels: Vec<ChannelId> = parts.iter().map(|p| p.channel).collect();
        let _ = policy
            .declare_binding(Fingerprint::from(1), "{}", channels.clone(), 0.0)
            .await;
        let _ = policy.arm().await;
        let _ = policy.rearm(owner).await;
        let Ok(leases) = policy.grant_leases(channels, Vec::new(), None).await else {
            continue;
        };
        let mut publishers = Vec::new();
        for lease in &leases {
            if let Ok(publisher) = policy.chunk_publisher(lease).await {
                publishers.push((lease.clone(), publisher));
            }
        }
        'driving: loop {
            for (lease, publisher) in &publishers {
                let held = policy.authority().is_some_and(|s| {
                    s.lease(lease.channel)
                        .is_some_and(|l| l.token == lease.token)
                });
                // Each joint's centre, as many as the part has.
                let width = publisher.stride() / 4;
                let action: Vec<u8> = std::iter::repeat_n(0.0_f32, width)
                    .flat_map(f32::to_le_bytes)
                    .collect();
                let now = policy.session_now_ns();
                let sent = publisher.send(ChunkDraft {
                    based_on: vec![InputRange::one(ChannelId(99), 1)],
                    t0_ns: now + 1_000_000,
                    dt_ns: 20_000_000,
                    replace_from_ns: now,
                    valid_until_ns: now + 200_000_000,
                    actions: Bytes::from(action.repeat(2)),
                    tick: 0,
                });
                if !held || sent.is_err() {
                    break 'driving;
                }
            }
            tokio::time::sleep(Duration::from_millis(33)).await;
        }
    }
}
