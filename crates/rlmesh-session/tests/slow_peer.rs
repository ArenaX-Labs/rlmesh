//! A peer that stops reading must not delay anyone else, nor keep the host
//! from shutting down: a board that goes radio-deaf, a frozen or
//! debugger-stopped client, or one whose task stopped reading while its TCP
//! stack still acknowledges, with the session's keys subscribed, while
//! healthy participants keep working.
//!
//! The deaf peer is a real Zenoh client configured as a board's zenoh-pico is
//! (no QoS, a 10 s lease), behind a relay that opens its links with a small
//! receive window and segment size, so the host's send buffer fills in
//! milliseconds as it would over Wi-Fi. Once deaf, the relay never reads from
//! the host again but keeps every link open, acknowledging (its kernel does)
//! and sending keep-alives: nothing but the host itself ends such a link.

#![cfg(target_os = "linux")]

use std::io::{BufRead, BufReader, ErrorKind, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::os::fd::AsRawFd;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use rlmesh_session::address::open_listeners;
use rlmesh_session::transport::{Endpoints, free_local_base, zenoh_config};
use rlmesh_session::wire::{ParticipantId, PriorityClass, Role, SessionId, SessionKeys, TenantId};
use rlmesh_session::{
    HostConfig, Participant, ParticipantConfig, SessionHost, SessionServer, SystemMonotonicClock,
    Transport, TransportConfig, TransportTuning,
};
use socket2::{Domain, Protocol, Socket, Type};
use zenoh::Wait;
use zenoh::qos::CongestionControl;

/// A healthy participant's query must come back within this, every time,
/// and the host must hear its heartbeats at least this often, while a peer
/// stops reading: less than the host takes to tell that peer is dead, so
/// nothing healthy ever waits on it.
const BOUND: Duration = Duration::from_secs(2);
/// The host must cut a link it can no longer drain within this of the link
/// filling up: its 2.5 s stall rule, a scan, and slack for a busy machine.
const EVICTED: Duration = Duration::from_secs(4);
/// The host must be gone within this of being told to stop: its transport
/// closes within 3 s whatever its peers do.
const SHUTDOWN: Duration = Duration::from_secs(4);
/// How long the long run watches healthy participants with the deaf peer
/// there; the default runs watch until the host cut the peer.
const WATCH: Duration = Duration::from_secs(30);
/// How long the test waits for something that must happen before it calls
/// it a hang.
const HANG: Duration = Duration::from_secs(60);
/// Unread bytes on a relay link from which the host's side is full.
const FULL: i32 = 2048;

/// A Zenoh transport frame on a stream link: a length, and one `KeepAlive`.
const KEEP_ALIVE: [u8; 3] = [1, 0, 0x04];

/// One relayed link of the board's.
struct RelayLink {
    /// The relay's end of the link to the host.
    host: TcpStream,
    /// When the host's side of it filled up, once deaf.
    full: OnceLock<Instant>,
    /// When the host reset it.
    evicted: OnceLock<Instant>,
}

/// A relay in front of the host's four class ports. Once deaf, it stops
/// reading from the host but keeps its lease alive with keep-alives: the
/// host's only way to learn the link is useless is that nothing drains. It
/// never closes a link itself, not even when dropped.
struct Relay {
    base: SocketAddr,
    deaf: Arc<OnceLock<Instant>>,
    links: Arc<Mutex<Vec<Arc<RelayLink>>>>,
}

impl Relay {
    fn start(host: SocketAddr) -> Self {
        let base = free_local_base("127.0.0.1".parse().expect("ip")).expect("ports");
        let deaf: Arc<OnceLock<Instant>> = Arc::default();
        let links: Arc<Mutex<Vec<Arc<RelayLink>>>> = Arc::default();
        for class in 0..4 {
            let listener =
                TcpListener::bind(SocketAddr::new(base.ip(), base.port() + class)).expect("bind");
            let to = SocketAddr::new(host.ip(), host.port() + class);
            let (deaf, links) = (deaf.clone(), links.clone());
            std::thread::spawn(move || {
                for client in listener.incoming().map_while(Result::ok) {
                    // A deaf board dials nowhere: a frozen process does not.
                    if deaf.get().is_some() {
                        continue;
                    }
                    let Some(host) = narrow_link(to) else {
                        continue;
                    };
                    let link = Arc::new(RelayLink {
                        host: host.try_clone().expect("clone"),
                        full: OnceLock::new(),
                        evicted: OnceLock::new(),
                    });
                    links.lock().expect("links").push(link.clone());
                    relay(client, host, deaf.clone(), link);
                }
            });
        }
        // Notes when each deaf link's host side fills up.
        {
            let (deaf, links) = (deaf.clone(), links.clone());
            std::thread::spawn(move || {
                loop {
                    if deaf.get().is_some() {
                        for link in links.lock().expect("links").iter() {
                            if unread(&link.host) >= FULL && link.full.set(Instant::now()).is_ok() {
                                tracing::warn!(link = ?link.host.local_addr(), "the host filled a link to the board");
                            }
                        }
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
            });
        }
        Self { base, deaf, links }
    }

    fn go_deaf(&self) {
        let _ = self.deaf.set(Instant::now());
        tracing::warn!("the board stops reading");
    }

    /// Waits until the host can queue nothing more on some link to the
    /// deaf board: its send queue holds bytes and stopped changing.
    fn wait_stuck(&self) {
        let started = Instant::now();
        let mut last: Vec<(u64, u32)> = Vec::new();
        loop {
            let queues: Vec<u64> = self
                .links()
                .iter()
                .map(|link| host_queue(&link.host).unwrap_or(0))
                .collect();
            last.resize(queues.len(), (0, 0));
            for (queue, (was, same)) in queues.iter().zip(last.iter_mut()) {
                *same = if *queue > 0 && queue == was {
                    *same + 1
                } else {
                    0
                };
                *was = *queue;
            }
            if last.iter().any(|(_, same)| *same >= 10) {
                return;
            }
            assert!(
                started.elapsed() < HANG,
                "no link to the deaf board got stuck: {queues:?}"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    fn links(&self) -> Vec<Arc<RelayLink>> {
        self.links.lock().expect("links").clone()
    }

    /// For each link that filled up: how long after filling the host reset
    /// it, if it did.
    fn evictions(&self) -> Vec<Option<Duration>> {
        self.links()
            .iter()
            .filter_map(|link| {
                let full = *link.full.get()?;
                Some(
                    link.evicted
                        .get()
                        .map(|at| at.saturating_duration_since(full)),
                )
            })
            .collect()
    }
}

/// Clears the `TCP_USER_TIMEOUT` the host's listeners on `base` give the
/// links they accept, as a platform without it would have them: what
/// remains is the host's own rule. (This kernel resets a link whose zero
/// window's probes go unanswered for that long, and one that never reads
/// is such a link.)
#[allow(unsafe_code)]
fn without_kernel_timeout(base: SocketAddr) {
    let ports = base.port()..=base.port() + 3;
    let mut cleared = 0;
    for entry in std::fs::read_dir("/proc/self/fd").expect("fds").flatten() {
        let Some(fd) = entry.file_name().to_str().and_then(|n| n.parse().ok()) else {
            continue;
        };
        // SAFETY: borrowed only for this iteration; the descriptor stays
        // open (nothing else in the test closes the host's listeners).
        let fd = unsafe { std::os::fd::BorrowedFd::borrow_raw(fd) };
        let socket = socket2::SockRef::from(&fd);
        let listening = socket.is_listener().unwrap_or(false);
        let on_port = socket
            .local_addr()
            .ok()
            .and_then(|a| a.as_socket())
            .is_some_and(|a| ports.contains(&a.port()));
        if listening && on_port && socket.set_tcp_user_timeout(None).is_ok() {
            cleared += 1;
        }
    }
    assert_eq!(cleared, 4, "the host's four listeners");
}

/// Bytes the host's end of `link` holds unacknowledged or unsent, as the
/// kernel lists them (in whichever process the host runs).
fn host_queue(link: &TcpStream) -> Option<u64> {
    let (relay, host) = (link.local_addr().ok()?, link.peer_addr().ok()?);
    let hex = |addr: SocketAddr| match addr {
        SocketAddr::V4(v4) => format!(
            "{:08X}:{:04X}",
            u32::from_le_bytes(v4.ip().octets()),
            v4.port()
        ),
        SocketAddr::V6(_) => String::new(),
    };
    let (local, remote) = (hex(host), hex(relay));
    let table = std::fs::read_to_string("/proc/net/tcp").ok()?;
    table.lines().skip(1).find_map(|line| {
        let fields: Vec<&str> = line.split_whitespace().collect();
        (fields.get(1) == Some(&local.as_str()) && fields.get(2) == Some(&remote.as_str()))
            .then(|| u64::from_str_radix(fields.get(4)?.split(':').next()?, 16).ok())
            .flatten()
    })
}

/// Bytes the relay has not read from `link`.
#[allow(unsafe_code)]
fn unread(link: &TcpStream) -> i32 {
    let mut n: libc::c_int = 0;
    // SAFETY: `n` is valid for the int FIONREAD writes.
    let ok = unsafe { libc::ioctl(link.as_raw_fd(), libc::FIONREAD, &raw mut n) } == 0;
    if ok { n } else { 0 }
}

/// A connection whose receive window closes after a few kilobytes.
fn narrow_link(to: SocketAddr) -> Option<TcpStream> {
    let socket = Socket::new(Domain::IPV4, Type::STREAM, Some(Protocol::TCP)).expect("socket");
    socket.set_recv_buffer_size(4096).expect("rcvbuf");
    socket.set_tcp_mss(1000).expect("mss");
    socket.connect(&to.into()).ok()?;
    Some(socket.into())
}

fn relay(client: TcpStream, host: TcpStream, deaf: Arc<OnceLock<Instant>>, link: Arc<RelayLink>) {
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
            if deaf_up.get().is_some() {
                if to_host.write_all(&KEEP_ALIVE).is_err() {
                    let _ = link.evicted.set(Instant::now());
                    tracing::warn!(link = ?to_host.local_addr(), "the host cut a link to the board");
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
    // Host to client, until deaf; the link's `host` end stays open.
    let (mut from_host, mut to_client) = (host, client);
    std::thread::spawn(move || {
        from_host
            .set_read_timeout(Some(Duration::from_millis(20)))
            .expect("timeout");
        let mut buffer = vec![0_u8; 16 * 1024];
        while deaf.get().is_none() {
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

/// Opens a board's four links through `relay`, subscribed to `key`; `heard`
/// is set once a sample reaches it.
async fn deaf_board(relay: &Relay, key: &str, heard: &Arc<AtomicBool>) -> Vec<zenoh::Session> {
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
        let heard = heard.clone();
        session
            .declare_subscriber(key)
            .callback(move |_| heard.store(true, Ordering::Relaxed))
            .background()
            .await
            .expect("subscribe");
        sessions.push(session);
    }
    sessions
}

/// Traffic the host routes to the board: 4 KiB samples it may drop, and
/// small ones it must not (as a participant's control publications are).
struct Filler {
    running: Arc<AtomicBool>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

impl Filler {
    fn start(session: &zenoh::Session, prefix: &str) -> Self {
        let running = Arc::new(AtomicBool::new(true));
        let mut threads = Vec::new();
        for (key, size, congestion, period) in [
            ("filler", 4096, CongestionControl::Drop, 1),
            ("command", 64, CongestionControl::Block, 50),
        ] {
            let (session, running) = (session.clone(), running.clone());
            let key = format!("{prefix}/{key}");
            threads.push(std::thread::spawn(move || {
                while running.load(Ordering::Relaxed) {
                    let _ = session
                        .put(&key, vec![0_u8; size])
                        .congestion_control(congestion)
                        .wait();
                    std::thread::sleep(Duration::from_millis(period));
                }
            }));
        }
        Self { running, threads }
    }

    fn stop(self) {
        self.running.store(false, Ordering::Relaxed);
        for thread in self.threads {
            thread.join().expect("filler");
        }
    }
}

/// Logs as `RUST_LOG` says (by default warnings, and what the test
/// measured): the host says when it cuts a link.
fn log() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn,slow_peer=info".into()),
        )
        .with_writer(std::io::stderr)
        .try_init();
}

/// A plain Zenoh client of the host's control port, with a board's 10 s
/// lease: it publishes what the board subscribed to. Until the host cuts the
/// board, its BLOCK samples wait on the board's full queue on the host's
/// thread reading this link, which is why it is not a participant: the host
/// would let a participant's 1 s link lease lapse meanwhile.
async fn feeder(base: SocketAddr) -> zenoh::Session {
    zenoh::open(
        zenoh_config(
            &Endpoints {
                listen: Vec::new(),
                connect: vec![format!("tcp/{base}")],
            },
            &client_tuning(Duration::from_secs(10)),
        )
        .expect("config"),
    )
    .await
    .expect("feeder")
}

/// Waits until the board heard what the host routes to it.
async fn wait_heard(heard: &AtomicBool) {
    let started = Instant::now();
    while !heard.load(Ordering::Relaxed) {
        assert!(started.elapsed() < HANG, "the board never heard the filler");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

struct World {
    base: SocketAddr,
    tenant: TenantId,
    session: SessionId,
    host: SessionHost,
    transport: Transport,
}

impl World {
    /// A host on free local ports; with `kernel`, its links keep the
    /// kernel's timeout.
    async fn start(kernel: bool) -> Self {
        log();
        let (transport, base) = open_listeners("127.0.0.1:0".parse().expect("addr"), None)
            .await
            .expect("listeners");
        if !kernel {
            without_kernel_timeout(base);
        }
        let tenant = TenantId::new("slow-peer").expect("tenant");
        let session = SessionId::new();
        let mut config = HostConfig::new(tenant.clone(), session);
        config.clock = SystemMonotonicClock.shared();
        config.lease_grace = Duration::from_millis(500);
        // Past [`BOUND`], which the test checks itself.
        config.presence_grace = BOUND + Duration::from_secs(1);
        config.health_log = None;
        let host = SessionHost::start(transport.clone(), config)
            .await
            .expect("host");
        Self {
            base,
            tenant,
            session,
            host,
            transport,
        }
    }

    fn prefix(&self) -> String {
        SessionKeys::new(&self.tenant, self.session)
            .prefix()
            .to_string()
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
        // Long enough to measure a stalled host rather than give up on it.
        config.ctl_timeout = HANG;
        config.credential = Some(self.host.credential());
        Participant::attach(transport, config)
            .await
            .expect("attach")
    }

    /// Stops the host and closes its links, from a thread of its own so a
    /// close that never returns fails the test instead of hanging it: how
    /// long the host and its transport took.
    fn shut_down(self) -> (Duration, Duration) {
        let (done, finished) = std::sync::mpsc::channel();
        let Self {
            host, transport, ..
        } = self;
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("runtime");
            runtime.block_on(async move {
                let started = Instant::now();
                host.shutdown().await;
                let host = started.elapsed();
                let _ = done.send(host);
                transport.close().await.expect("close");
                let _ = done.send(started.elapsed() - host);
            });
        });
        let host = finished.recv_timeout(HANG).unwrap_or_else(|_| {
            panic!("the host task did not stop in {HANG:?} with a deaf peer linked")
        });
        let transport = finished.recv_timeout(HANG).unwrap_or_else(|_| {
            panic!(
                "the host's transport did not close in {HANG:?} with a deaf peer linked \
                 (the host task stopped in {host:?})"
            )
        });
        (host, transport)
    }
}

/// What a healthy observer saw while the deaf peer was there.
#[derive(Debug, Default)]
struct Watched {
    queries: Vec<Duration>,
    failed: usize,
    /// The longest a participant took to join and leave.
    churn: Duration,
    /// The longest the host went without a healthy participant's heartbeat.
    unheard: Duration,
}

impl Watched {
    fn quantile(&self, q: f64) -> Duration {
        let mut sorted = self.queries.clone();
        sorted.sort();
        sorted
            .get(((sorted.len() as f64 - 1.0) * q).round() as usize)
            .copied()
            .unwrap_or_default()
    }
}

/// Queries the host from `observer` every 50 ms, and has a participant
/// join and leave every two seconds (a join tells every link, the deaf one
/// too), checking that the host keeps hearing `healthy`: for `watch` at
/// least, and until every link to the deaf board that filled up was cut and
/// a participant joined and left meanwhile.
async fn watch_healthy(
    world: &World,
    relay: &Relay,
    observer: &Participant,
    healthy: &[&ParticipantId],
    watch: Duration,
) -> Watched {
    let mut watched = Watched::default();
    let started = Instant::now();
    let mut joined = Instant::now();
    let mut late = 0;
    let cut = || {
        let evictions = relay.evictions();
        !evictions.is_empty() && evictions.iter().all(Option::is_some)
    };
    while started.elapsed() < watch || late == 0 || !cut() {
        assert!(
            started.elapsed() < HANG,
            "the host never cut the deaf board: {:?}",
            relay.evictions()
        );
        let asked = Instant::now();
        match tokio::time::timeout(HANG, observer.fetch_authority()).await {
            Ok(Ok(view)) => {
                watched.queries.push(asked.elapsed());
                for id in healthy {
                    assert!(
                        view.participant(id).is_some_and(|p| p.present),
                        "the host dropped healthy {id} {:?} in",
                        started.elapsed()
                    );
                }
            }
            Ok(Err(err)) => {
                tracing::warn!("query failed {:?} in: {err}", started.elapsed());
                watched.failed += 1;
            }
            Err(_) => panic!("a query hung {:?} in", started.elapsed()),
        }
        for health in world.host.health() {
            if healthy.contains(&&health.participant)
                && let Some(age) = health.last_heartbeat_age_ns
            {
                watched.unheard = watched.unheard.max(Duration::from_nanos(age));
            }
        }
        if joined.elapsed() >= Duration::from_secs(2) {
            late += 1;
            let churned = Instant::now();
            let participant =
                tokio::time::timeout(HANG, world.join(&format!("late-{late}"), Role::Observer))
                    .await
                    .unwrap_or_else(|_| panic!("a join hung {:?} in", started.elapsed()));
            tokio::time::timeout(HANG, participant.detach())
                .await
                .unwrap_or_else(|_| panic!("a leave hung {:?} in", started.elapsed()))
                .expect("detach");
            watched.churn = watched.churn.max(churned.elapsed());
            joined = Instant::now();
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    watched
}

/// A robot, an observer, and a board that stops reading once the host
/// routes it what the robot publishes: what the observer saw for `watch`,
/// and how the host shut down after, with the board still linked.
async fn deaf_board_among_healthy(kernel: bool, watch: Duration) {
    let world = World::start(kernel).await;
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
    let heard = Arc::new(AtomicBool::new(false));
    let board = deaf_board(&relay, &format!("{}/**", world.prefix()), &heard).await;
    let feeder = feeder(world.base).await;
    let filler = Filler::start(&feeder, &world.prefix());
    // Deaf once the host routes the filler to the board, not before.
    wait_heard(&heard).await;
    relay.go_deaf();

    let watched = watch_healthy(
        &world,
        &relay,
        &observer,
        &[robot.id(), observer.id()],
        watch,
    )
    .await;
    let evictions = relay.evictions();
    filler.stop();
    // Closing them would wait on links nothing reads.
    std::mem::forget(board);
    drop((feeder, robot, observer));
    let (host, transport) = world.shut_down();
    tracing::info!(
        "kernel timeout {kernel}, watched {watch:?}: queries={} failed={} p50={:?} p99={:?} \
         max={:?} unheard={:?} join+leave max={:?} evictions={evictions:?} shutdown: \
         host={host:?} transport={transport:?}",
        watched.queries.len(),
        watched.failed,
        watched.quantile(0.5),
        watched.quantile(0.99),
        watched.quantile(1.0),
        watched.unheard,
        watched.churn,
    );
    assert_eq!(watched.failed, 0, "queries failed");
    assert!(
        watched.quantile(1.0) < BOUND,
        "a query took {:?}",
        watched.quantile(1.0)
    );
    assert!(
        watched.unheard < BOUND,
        "the host went {:?} without a heartbeat",
        watched.unheard
    );
    assert!(
        watched.churn < BOUND,
        "a participant took {:?} to join and leave",
        watched.churn
    );
    assert!(!evictions.is_empty(), "no link to the board filled up");
    for eviction in evictions {
        assert!(
            eviction.is_some_and(|e| e < EVICTED),
            "a full link was evicted after {eviction:?}"
        );
    }
    assert!(
        host + transport < SHUTDOWN,
        "shutdown took {host:?} + {transport:?}"
    );
}

/// The host's own rule, alone: what a host without the kernel's timeout
/// (macOS, Windows, an older Linux) relies on, until it cut the peer.
#[tokio::test(flavor = "multi_thread")]
async fn a_peer_that_stops_reading_delays_no_one_else() {
    deaf_board_among_healthy(false, Duration::ZERO).await;
}

/// The same for [`WATCH`]: past the deaf board's own Zenoh lease, with
/// participants joining and leaving all along.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "slow: 30 s of watching; mise run test:rust:slow"]
async fn a_peer_that_stops_reading_delays_no_one_else_for_long() {
    deaf_board_among_healthy(false, WATCH).await;
}

/// With the kernel's timeout, as a Linux host runs.
#[tokio::test(flavor = "multi_thread")]
async fn the_kernel_resets_a_peer_that_stops_reading() {
    deaf_board_among_healthy(true, Duration::ZERO).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn the_host_shuts_down_past_a_peer_that_stops_reading() {
    let world = World::start(false).await;
    let robot = world.join("robot", Role::Embodiment).await;
    let relay = Relay::start(world.base);
    let heard = Arc::new(AtomicBool::new(false));
    let board = deaf_board(&relay, &format!("{}/**", world.prefix()), &heard).await;
    let feeder = feeder(world.base).await;
    let filler = Filler::start(&feeder, &world.prefix());
    wait_heard(&heard).await;
    relay.go_deaf();
    // Stopped as soon as a link to the board is stuck, before the host
    // could have evicted it.
    tokio::task::block_in_place(|| relay.wait_stuck());
    filler.stop();
    std::mem::forget(board);
    drop((feeder, robot));
    let (host, transport) = world.shut_down();
    tracing::info!(
        "shutdown with a stuck link: host={host:?} transport={transport:?} evictions={:?}",
        relay.evictions()
    );
    assert!(
        host + transport < SHUTDOWN,
        "shutdown took {host:?} + {transport:?}"
    );
}

/// Set for the child process
/// [`a_host_process_exits_on_sigterm_past_a_peer_that_stops_reading`]
/// starts.
const CHILD: &str = "RLMESH_SLOW_PEER_HOST";

/// In the child only: a host as `rlmesh host` runs one, until SIGTERM,
/// without the kernel's timeout on its links.
#[test]
fn host_process() {
    if std::env::var_os(CHILD).is_none() {
        return;
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("runtime");
    log();
    runtime.block_on(async {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("signal");
        let (transport, base) = open_listeners("127.0.0.1:0".parse().expect("addr"), None)
            .await
            .expect("listeners");
        let config = HostConfig::new(
            TenantId::new("slow-peer").expect("tenant"),
            SessionId::new(),
        );
        let server = SessionServer::start(transport.clone(), config)
            .await
            .expect("server");
        without_kernel_timeout(base);
        // On a line of its own: the harness has begun one with the test's
        // name.
        let mut stdout = std::io::stdout();
        writeln!(stdout, "\nSLOW_PEER_BASE {base}").expect("base");
        stdout.flush().expect("flush");
        term.recv().await;
        server.shutdown().await;
        transport.close().await.expect("close");
    });
}

#[tokio::test(flavor = "multi_thread")]
async fn a_host_process_exits_on_sigterm_past_a_peer_that_stops_reading() {
    log();
    let mut child = Command::new(std::env::current_exe().expect("exe"))
        .args(["--exact", "host_process", "--nocapture", "--test-threads=1"])
        .env(CHILD, "1")
        .stdout(Stdio::piped())
        .spawn()
        .expect("child");
    let pid = child.id();
    // Killed whatever the outcome, by its PID.
    struct Reap(u32);
    impl Drop for Reap {
        #[allow(unsafe_code)]
        fn drop(&mut self) {
            // SAFETY: a plain signal to the child this test started.
            unsafe { libc::kill(self.0 as libc::pid_t, libc::SIGKILL) };
        }
    }
    let _reap = Reap(pid);
    let stdout = child.stdout.take().expect("stdout");
    let (found, base) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut lines = BufReader::new(stdout).lines().map_while(Result::ok);
        let line = lines
            .by_ref()
            .find_map(|line| line.strip_prefix("SLOW_PEER_BASE ").map(str::to_string));
        let _ = found.send(line);
        // Read to the end: the harness fails a child that can't write.
        lines.for_each(drop);
    });
    let base: SocketAddr = base
        .recv_timeout(HANG)
        .expect("the host process never said where it listens")
        .expect("the host's base")
        .parse()
        .expect("base");

    let relay = Relay::start(base);
    let heard = Arc::new(AtomicBool::new(false));
    let board = deaf_board(&relay, "**", &heard).await;
    let feeder = feeder(base).await;
    let filler = Filler::start(&feeder, "slow-peer/feed");
    wait_heard(&heard).await;
    relay.go_deaf();
    tokio::task::block_in_place(|| relay.wait_stuck());
    // Stopped while the host still reads it: a BLOCK put to a host that is
    // gone waits out the feeder's link lease.
    filler.stop();

    let started = Instant::now();
    #[allow(unsafe_code)]
    // SAFETY: a plain signal to the child this test started.
    unsafe {
        libc::kill(pid as libc::pid_t, libc::SIGTERM)
    };
    let status = loop {
        if let Some(status) = child.try_wait().expect("wait") {
            break status;
        }
        assert!(
            started.elapsed() < HANG,
            "the host process did not exit in {HANG:?} of SIGTERM"
        );
        std::thread::sleep(Duration::from_millis(10));
    };
    let exited = started.elapsed();
    std::mem::forget((board, feeder));
    tracing::info!("host process exited {exited:?} after SIGTERM: {status}");
    assert!(status.success(), "the host process failed: {status}");
    assert!(
        exited < SHUTDOWN,
        "the host process took {exited:?} to exit"
    );
}
