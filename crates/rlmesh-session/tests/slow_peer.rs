//! A peer that stops reading must not delay anyone else: a board that goes
//! radio-deaf (or a client that keeps talking but never reads) with the
//! session's keys subscribed, while healthy participants keep working.
//!
//! The deaf peer is a real Zenoh client configured as a board's zenoh-pico is
//! (no QoS, a 10 s lease), behind a relay that opens its links with a small
//! receive window and segment size, so the host's send buffer fills in
//! milliseconds as it would over Wi-Fi.

#![cfg(target_os = "linux")]

use std::io::{ErrorKind, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use rlmesh_session::address::open_listeners;
use rlmesh_session::transport::{Endpoints, free_local_base, zenoh_config};
use rlmesh_session::wire::{ParticipantId, PriorityClass, Role, SessionId, SessionKeys, TenantId};
use rlmesh_session::{
    HostConfig, Participant, ParticipantConfig, SessionHost, SystemMonotonicClock, Transport,
    TransportConfig, TransportTuning,
};
use socket2::{Domain, Protocol, Socket, Type};
use zenoh::Wait;
use zenoh::qos::CongestionControl;

/// A healthy participant's query must come back within this, every time. A
/// host held by the deaf peer misses it by seconds.
const BOUND: Duration = Duration::from_millis(1500);
/// How long the deaf peer is given to hold the host up.
const WINDOW: Duration = Duration::from_secs(4);

/// A Zenoh transport frame on a stream link: a length, and one `KeepAlive`.
const KEEP_ALIVE: [u8; 3] = [1, 0, 0x04];

/// A relay in front of the host's four class ports. Once deaf, it stops
/// reading from the host but keeps its lease alive with keep-alives: the
/// host's only way to learn the link is useless is that nothing drains.
struct Relay {
    base: SocketAddr,
    deaf: Arc<AtomicBool>,
    /// The host reset a link while it was deaf.
    evicted: Arc<AtomicBool>,
}

impl Relay {
    fn start(host: SocketAddr) -> Self {
        let base = free_local_base("127.0.0.1".parse().expect("ip")).expect("ports");
        let deaf = Arc::new(AtomicBool::new(false));
        let evicted = Arc::new(AtomicBool::new(false));
        for class in 0..4 {
            let listener =
                TcpListener::bind(SocketAddr::new(base.ip(), base.port() + class)).expect("bind");
            let to = SocketAddr::new(host.ip(), host.port() + class);
            let (deaf, evicted) = (deaf.clone(), evicted.clone());
            std::thread::spawn(move || {
                for client in listener.incoming().map_while(Result::ok) {
                    relay(client, narrow_link(to), deaf.clone(), evicted.clone());
                }
            });
        }
        Self {
            base,
            deaf,
            evicted,
        }
    }
}

/// A connection whose receive window closes after a few kilobytes.
fn narrow_link(to: SocketAddr) -> TcpStream {
    let socket = Socket::new(Domain::IPV4, Type::STREAM, Some(Protocol::TCP)).expect("socket");
    socket.set_recv_buffer_size(4096).expect("rcvbuf");
    socket.set_tcp_mss(1000).expect("mss");
    socket.connect(&to.into()).expect("connect");
    socket.into()
}

fn relay(client: TcpStream, host: TcpStream, deaf: Arc<AtomicBool>, evicted: Arc<AtomicBool>) {
    let (mut from_client, mut to_host) = (
        client.try_clone().expect("clone"),
        host.try_clone().expect("clone"),
    );
    let deaf_up = deaf.clone();
    // Client to host, whole frames at a time, so keep-alives start on a
    // frame boundary.
    std::thread::spawn(move || {
        from_client
            .set_read_timeout(Some(Duration::from_millis(20)))
            .expect("timeout");
        loop {
            if deaf_up.load(Ordering::Relaxed) {
                if to_host.write_all(&KEEP_ALIVE).is_err() {
                    evicted.store(true, Ordering::Relaxed);
                    return;
                }
                std::thread::sleep(Duration::from_millis(100));
                continue;
            }
            let mut length = [0_u8; 2];
            match from_client.read(&mut length[..1]) {
                Ok(0) => return,
                Ok(_) => {}
                Err(err) if matches!(err.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {
                    continue;
                }
                Err(_) => return,
            }
            let mut frame = Vec::new();
            let ok = read_patiently(&mut from_client, &mut length[1..]).is_ok() && {
                frame.resize(usize::from(u16::from_le_bytes(length)), 0);
                read_patiently(&mut from_client, &mut frame).is_ok()
            };
            if !ok || to_host.write_all(&length).is_err() || to_host.write_all(&frame).is_err() {
                return;
            }
        }
    });
    // Host to client, until deaf.
    let (mut from_host, mut to_client) = (host, client);
    std::thread::spawn(move || {
        from_host
            .set_read_timeout(Some(Duration::from_millis(20)))
            .expect("timeout");
        let mut buffer = vec![0_u8; 16 * 1024];
        while !deaf.load(Ordering::Relaxed) {
            match from_host.read(&mut buffer) {
                Ok(0) => return,
                Ok(n) => {
                    if to_client.write_all(&buffer[..n]).is_err() {
                        return;
                    }
                }
                Err(err) if matches!(err.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {}
                Err(_) => return,
            }
        }
        // Keep the socket open and unread for as long as the test runs.
        std::thread::sleep(Duration::from_secs(600));
    });
}

/// `read_exact` over a socket with a read timeout.
fn read_patiently(from: &mut TcpStream, mut into: &mut [u8]) -> std::io::Result<()> {
    while !into.is_empty() {
        match from.read(into) {
            Ok(0) => return Err(ErrorKind::UnexpectedEof.into()),
            Ok(n) => into = &mut into[n..],
            Err(err) if matches!(err.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {}
            Err(err) => return Err(err),
        }
    }
    Ok(())
}

fn client_tuning(lease: Duration) -> TransportTuning {
    TransportTuning {
        client: true,
        lease,
        ..TransportTuning::default()
    }
}

struct World {
    base: SocketAddr,
    tenant: TenantId,
    session: SessionId,
    host: SessionHost,
}

impl World {
    async fn start() -> Self {
        let (transport, base) = open_listeners("127.0.0.1:0".parse().expect("addr"), None)
            .await
            .expect("listeners");
        let tenant = TenantId::new("slow-peer").expect("tenant");
        let session = SessionId::new();
        let mut config = HostConfig::new(tenant.clone(), session);
        config.clock = SystemMonotonicClock.shared();
        config.lease_grace = Duration::from_millis(500);
        config.presence_grace = Duration::from_secs(2);
        config.health_log = None;
        let host = SessionHost::start(transport, config).await.expect("host");
        Self {
            base,
            tenant,
            session,
            host,
        }
    }

    async fn join(&self, id: &str, role: Role) -> Participant {
        let transport = Transport::open(
            &TransportConfig::per_class_tcp(None, Some(self.base))
                .expect("config")
                .with_tuning(client_tuning(Duration::from_secs(1))),
        )
        .await
        .expect("transport");
        let mut config = ParticipantConfig::new(
            self.tenant.clone(),
            self.session,
            ParticipantId::new(id).expect("id"),
            vec![role],
            PriorityClass::Scripted,
        );
        config.clock = SystemMonotonicClock.shared();
        config.heartbeat_period = Duration::from_millis(50);
        config.ctl_timeout = BOUND;
        config.credential = Some(self.host.credential());
        Participant::attach(transport, config)
            .await
            .expect("attach")
    }

    /// Opens a board's four links through `relay`, subscribed to every key
    /// of the session.
    async fn deaf_board(&self, relay: &Relay) -> Vec<zenoh::Session> {
        let keys = SessionKeys::new(&self.tenant, self.session);
        let mut sessions = Vec::new();
        for class in 0..4 {
            let endpoints = Endpoints {
                listen: Vec::new(),
                connect: vec![format!(
                    "tcp/{}:{}",
                    relay.base.ip(),
                    relay.base.port() + class
                )],
            };
            let mut config =
                zenoh_config(&endpoints, &client_tuning(Duration::from_secs(10))).expect("config");
            config
                .insert_json5("transport/unicast/qos/enabled", "false")
                .expect("qos");
            let session = zenoh::open(config).await.expect("board");
            session
                .declare_subscriber(format!("{}/**", keys.prefix()))
                .callback(|_| {})
                .background()
                .await
                .expect("subscribe");
            sessions.push(session);
        }
        sessions
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_peer_that_stops_reading_delays_no_one_else() {
    let world = World::start().await;
    let robot = world.join("robot", Role::Embodiment).await;
    robot
        .declare_parts(
            r#"{"arm": {"kind": "joints", "hz": 100, "count": 6, "fields": ["pos"],
                        "command": "position", "safe": "stop"}}"#,
        )
        .await
        .expect("declare");
    let observer = world.join("observer", Role::Observer).await;

    let relay = Relay::start(world.base);
    let board = world.deaf_board(&relay).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    relay.deaf.store(true, Ordering::Relaxed);

    // Fill the deaf link's queue: what the host routes to it stays there.
    let filling = Arc::new(AtomicBool::new(true));
    let filler = {
        let control = robot.transport().control().clone();
        let key = format!(
            "{}/filler",
            SessionKeys::new(&world.tenant, world.session).prefix()
        );
        let filling = filling.clone();
        std::thread::spawn(move || {
            while filling.load(Ordering::Relaxed) {
                let _ = control
                    .put(&key, vec![0_u8; 4096])
                    .congestion_control(CongestionControl::Drop)
                    .wait();
                std::thread::sleep(Duration::from_millis(1));
            }
        })
    };

    let started = Instant::now();
    let mut worst = Duration::ZERO;
    let mut joined = false;
    while started.elapsed() < WINDOW {
        let asked = Instant::now();
        let view = observer.fetch_authority().await.unwrap_or_else(|err| {
            panic!(
                "no answer {:?} after the peer went deaf: {err}",
                started.elapsed()
            )
        });
        worst = worst.max(asked.elapsed());
        assert!(
            view.participant(robot.id()).is_some_and(|p| p.present),
            "the host dropped a healthy robot {:?} in",
            started.elapsed()
        );
        // A participant joining tells every link, the deaf one too.
        if !joined && started.elapsed() > Duration::from_secs(1) {
            let late = world.join("late", Role::Observer).await;
            late.detach().await.expect("detach");
            joined = true;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    filling.store(false, Ordering::Relaxed);
    filler.join().expect("filler");
    // Closing them would wait on links nothing reads.
    std::mem::forget(board);
    assert!(worst < BOUND, "a query took {worst:?}");
    assert!(
        relay.evicted.load(Ordering::Relaxed),
        "the host still holds the deaf link"
    );
}
