//! The device's robot against a real session host over Zenoh on loopback,
//! with this test as its transport (one session per traffic class): what
//! zenoh-pico does on a board.

// The transport's side: it reads the clock, as a board's does.
#![allow(clippy::disallowed_methods)]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bytes::Bytes;
#[cfg(not(feature = "fleet"))]
use rlmesh_device::provision::Record;
use rlmesh_device::robot::{
    Action, Config, Fault, Phase, Robot, SUB_COMMANDS, SUB_KEYS, SUB_VIEW, SUB_VIEW_SEALED,
};
use rlmesh_session::address::AddressKey;
use rlmesh_session::auth::{Credential, Device, HostIdentity};
use rlmesh_session::authority::DeviceAccess;
use rlmesh_session::crypto::StaticKeypair;
use rlmesh_session::crypto::{Rng, RngError, SignedChecker};
use rlmesh_session::edge::Rejection;
use rlmesh_session::host::DatagramRelay;
use rlmesh_session::transport::{TrafficClass, Transport, TransportConfig, open_local_listeners};
use rlmesh_session::wire::{
    ChannelId, CtlStatus, EdgeMode, Fingerprint, InputRange, Mode, ParticipantId, Plan,
    PriorityClass, Role, SessionId, TenantId,
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
/// zenoh-pico reassembles a message in 4096 bytes (`Z_FRAG_MAX_SIZE`, as
/// boards are built) and drops a larger one without a word.
const PICO_MESSAGE: usize = 4096;

/// The system allocator, counting what the calling thread allocates while
/// [`counted`] runs: a board's heap, as one receive spends it.
struct Counting;

/// What one counted call allocated.
#[derive(Clone, Copy, Debug, Default)]
struct Allocs {
    /// Allocations and reallocations made.
    count: u32,
    /// Bytes they asked for.
    bytes: usize,
    /// The largest single request.
    largest: usize,
    /// The most held at once beyond what was held before.
    peak: usize,
    live: isize,
}

thread_local! {
    static COUNTING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static ALLOCS: std::cell::Cell<Allocs> = const {
        std::cell::Cell::new(Allocs { count: 0, bytes: 0, largest: 0, peak: 0, live: 0 })
    };
}

impl Counting {
    fn note(grew: usize, shrank: usize) {
        if !COUNTING.with(std::cell::Cell::get) {
            return;
        }
        ALLOCS.with(|a| {
            let mut allocs = a.get();
            if grew > 0 {
                allocs.count += 1;
                allocs.bytes += grew;
                allocs.largest = allocs.largest.max(grew);
            }
            allocs.live += grew.cast_signed() - shrank.cast_signed();
            allocs.peak = allocs.peak.max(allocs.live.max(0).cast_unsigned());
            a.set(allocs);
        });
    }
}

// SAFETY: the system allocator's, with a thread-local tally beside it.
#[allow(unsafe_code)]
unsafe impl std::alloc::GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        Self::note(layout.size(), 0);
        unsafe { std::alloc::System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        Self::note(0, layout.size());
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, new: usize) -> *mut u8 {
        Self::note(new, layout.size());
        unsafe { std::alloc::System.realloc(ptr, layout, new) }
    }
}

#[global_allocator]
static COUNTING_ALLOC: Counting = Counting;

/// Runs `f`, with what it allocated on this thread.
fn counted<R>(f: impl FnOnce() -> R) -> (R, Allocs) {
    ALLOCS.with(|a| a.set(Allocs::default()));
    COUNTING.with(|c| c.set(true));
    let out = f();
    COUNTING.with(|c| c.set(false));
    (out, ALLOCS.with(std::cell::Cell::get))
}

/// One message a board took, and what taking it cost.
#[derive(Clone, Copy, Debug)]
struct Taken {
    at: Instant,
    what: Took,
    took: Duration,
    allocs: Allocs,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Took {
    SealedView,
    SignedView,
    Reply,
}

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
    Datagram(Vec<u8>),
}

struct World {
    tenant: TenantId,
    session: SessionId,
    base: std::net::SocketAddr,
    identity: Arc<HostIdentity>,
    host: SessionHost,
    /// Its datagram path, if it relays over UDP.
    relay: Option<Arc<DatagramRelay>>,
}

/// Client sessions to the host at `base`, one per class.
async fn links(base: std::net::SocketAddr) -> Transport {
    Transport::open(
        &TransportConfig::per_class_tcp(Some("127.0.0.1:0".parse().expect("addr")), Some(base))
            .expect("config"),
    )
    .await
    .expect("transport")
}

impl World {
    async fn start() -> Self {
        Self::start_with(Vec::new()).await
    }

    /// A host that knows `devices` besides `local`.
    async fn start_with(devices: Vec<Device>) -> Self {
        Self::start_configured(|config| config.devices = devices).await
    }

    /// A host that signs device views and seals none, as one from before
    /// sealed views.
    async fn start_unsealed() -> Self {
        Self::start_configured(|config| {
            config
                .supports
                .retain(|f| f != rlmesh_session::wire::version::VIEW_DEVICE_SEALED);
        })
        .await
    }

    async fn start_configured(configure: impl FnOnce(&mut HostConfig)) -> Self {
        let (transport, base) = open_local_listeners("127.0.0.1".parse().expect("ip"))
            .await
            .expect("host transport");
        let tenant = TenantId::new("test").expect("tenant");
        let session = SessionId::new();
        let mut config = HostConfig::new(tenant.clone(), session);
        configure(&mut config);
        let identity = config.identity.clone();
        let host = SessionHost::start(transport, config).await.expect("host");
        Self {
            tenant,
            session,
            base,
            identity,
            host,
            relay: None,
        }
    }

    async fn transport(&self) -> Transport {
        links(self.base).await
    }

    /// A host that relays boards' realtime parts over UDP on loopback.
    async fn start_datagram() -> Self {
        let relay = DatagramRelay::bind("127.0.0.1:0".parse().expect("addr"))
            .await
            .expect("udp");
        let mut world = Self::start_configured({
            let relay = relay.clone();
            move |config| config.datagram = Some(relay)
        })
        .await;
        world.relay = Some(relay);
        world
    }

    /// Stops the host and starts a new one at the same address with the same
    /// key: a new process, a new instance, an empty roster.
    async fn restart(self) -> Self {
        self.restart_with(None).await
    }

    /// [`Self::restart`], relaying over UDP on `datagram` when given.
    async fn restart_with(self, datagram: Option<Arc<DatagramRelay>>) -> Self {
        let Self {
            tenant,
            session,
            base,
            identity,
            host,
            ..
        } = self;
        host.shutdown().await;
        let started = Instant::now();
        // The ports free up as the old transport finishes closing.
        let transport = loop {
            let config = TransportConfig::per_class_tcp(Some(base), None).expect("config");
            match Transport::open(&config).await {
                Ok(transport) => break transport,
                Err(err) => assert!(started.elapsed() < WAIT, "no ports back: {err}"),
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        };
        let mut config = HostConfig::new(tenant.clone(), session);
        config.identity = identity.clone();
        config.datagram = datagram.clone();
        let host = SessionHost::start(transport, config).await.expect("host");
        Self {
            tenant,
            session,
            base,
            identity,
            host,
            relay: datagram,
        }
    }

    fn robot(&self, key: AddressKey) -> Box<Robot> {
        self.robot_as(key.credential())
    }

    fn robot_as(&self, credential: Credential) -> Box<Robot> {
        Robot::new(Config {
            tenant: self.tenant.clone(),
            session: self.session,
            name: ParticipantId::new("eyes").expect("id"),
            credential,
            robot_json: MAP.to_owned(),
            heartbeat_ms: 50,
            lobby: false,
            homes: vec![(self.base.ip().to_string(), self.base.port())],
            home_id: String::new(),
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
    /// Samples of the part that actually went out.
    sent: u32,
    /// Why the board last refused a chunk, and its invalid-value count.
    refused: (Option<Rejection>, u64),
    /// How the part's chunks arrived, and the robot's accepted count.
    timing: Option<rlmesh_session::wire::ChunkTiming>,
    accepted: u64,
    /// The robot's end of the datagram path.
    datagram: Option<rlmesh_session::wire::DatagramStats>,
}

/// A board's transport: one session per traffic class, as zenoh-pico opens
/// them, kept across runs so a test can break the links under it.
struct Board {
    /// `None` while the links are dead: nothing goes out or comes back
    /// until the robot asks to reopen them (zenoh-pico's sessions don't come
    /// back on their own; Zenoh's own client would, and hide the fault).
    transport: Option<Transport>,
    base: std::net::SocketAddr,
    tx: mpsc::UnboundedSender<Event>,
    rx: mpsc::UnboundedReceiver<Event>,
    subscribers: Vec<zenoh::pubsub::Subscriber<()>>,
    tokens: Vec<zenoh::liveliness::LivelinessToken>,
    queries: BTreeMap<u32, tokio::task::JoinHandle<()>>,
    rng: Mix,
    start: Instant,
    next_sample: u64,
    /// Links reopened at the robot's request.
    reconnects: u32,
    /// No new link can open before this: the network is down.
    offline_until: Option<Instant>,
    /// The transport never reports a query's end, as when zenoh-pico's
    /// executor is blocked in `recv`.
    no_query_end: bool,
    /// Queries that never reach the host, by key.
    lose: Box<dyn FnMut(&str) -> bool + Send>,
    /// Replies too large for zenoh-pico ([`PICO_MESSAGE`]) are lost, as
    /// on a board.
    pico: bool,
    /// Replies lost for their size.
    oversized: Arc<std::sync::atomic::AtomicU32>,
    /// Check signed views apart from the robot, as a board's transport
    /// does on another task; `checker` is the current link's.
    check_views: bool,
    checker: Option<SignedChecker>,
    /// Checkers taken, and views the checker refused.
    checkers: u32,
    views_refused: u32,
    /// Spoils the next signed view that arrives.
    forge_next_view: bool,
    /// Signature checks made, and those made after the robot took a sealed
    /// view (a board's transport skips them then).
    checks: u32,
    checks_after_sealed: u32,
    /// Sealed views never reach the robot, as from a host that doesn't seal.
    /// Signed views that arrived after the first sealed one.
    signed_after_sealed: u32,
    /// The first sealed view that arrived, kept for a replay.
    first_sealed: Option<(String, Vec<u8>)>,
    /// The last sealed view that arrived.
    last_sealed: Option<(String, Vec<u8>)>,
    /// Views and replies taken, with their cost.
    taken: Arc<Mutex<Vec<Taken>>>,
    /// A fading link: within each period, nothing arrives or leaves for
    /// the first part.
    fades: Arc<Mutex<Option<(Duration, Duration)>>>,
    /// Each chunk that arrived: when, the key it was sealed under, and
    /// whether the robot refused it as unauthenticated.
    chunks: Arc<Mutex<Vec<(Instant, u32, bool)>>>,
    /// Key deliveries never reach the robot, so grants for it wait.
    drop_keys: Arc<std::sync::atomic::AtomicBool>,
    /// The datagram socket the robot opened, connected to the host's UDP
    /// port, and its reader.
    udp: Option<(Arc<tokio::net::UdpSocket>, tokio::task::JoinHandle<()>)>,
    /// Datagrams lost on their way: those from the host, and those to it
    /// (a probe's lane, or any other).
    lose_down: Arc<std::sync::atomic::AtomicBool>,
    lose_up: Arc<std::sync::atomic::AtomicBool>,
    lose_probes: Arc<std::sync::atomic::AtomicBool>,
    /// A send the socket refused: the robot is told on its next loop.
    udp_failed: bool,
    /// Every datagram that arrived from the host, kept for a replay.
    datagrams: Arc<Mutex<Vec<Vec<u8>>>>,
    /// Datagrams sent, by lane.
    sent_up: Arc<Mutex<BTreeMap<String, u32>>>,
    /// Commands subscriptions declared on the session's links.
    command_subscriptions: u32,
    /// The last probe the robot sent.
    last_probe: Option<Vec<u8>>,
}

impl Board {
    async fn open(world: &World) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        Self {
            transport: Some(world.transport().await),
            base: world.base,
            tx,
            rx,
            subscribers: Vec::new(),
            tokens: Vec::new(),
            queries: BTreeMap::new(),
            rng: seeded(),
            start: Instant::now(),
            next_sample: 0,
            reconnects: 0,
            offline_until: None,
            no_query_end: false,
            lose: Box::new(|_| false),
            pico: false,
            oversized: Arc::default(),
            check_views: false,
            checker: None,
            checkers: 0,
            views_refused: 0,
            forge_next_view: false,
            checks: 0,
            checks_after_sealed: 0,
            signed_after_sealed: 0,
            first_sealed: None,
            last_sealed: None,
            taken: Arc::default(),
            fades: Arc::default(),
            chunks: Arc::default(),
            drop_keys: Arc::default(),
            udp: None,
            lose_down: Arc::default(),
            lose_up: Arc::default(),
            lose_probes: Arc::default(),
            udp_failed: false,
            datagrams: Arc::default(),
            sent_up: Arc::default(),
            command_subscriptions: 0,
            last_probe: None,
        }
    }

    /// Whether the link is in a fade now.
    fn faded(&self) -> bool {
        self.fades
            .lock()
            .expect("fades")
            .is_some_and(|(period, fade)| {
                let into = self.start.elapsed().as_micros() % period.as_micros().max(1);
                into < fade.as_micros()
            })
    }

    /// Hands `robot` a message as `take` does, timed and counted.
    fn take(&self, what: Took, take: impl FnOnce()) {
        let started = Instant::now();
        let ((), allocs) = counted(take);
        let took = started.elapsed();
        self.taken.lock().expect("taken").push(Taken {
            at: started,
            what,
            took,
            allocs,
        });
    }

    /// The links die with no word to the board: a host restarted, or Wi-Fi
    /// gone. What is in flight is lost.
    fn sever(&mut self) {
        self.close_udp();
        self.transport = None;
        self.subscribers.clear();
        self.tokens.clear();
        for (_, query) in std::mem::take(&mut self.queries) {
            query.abort();
        }
        while self.rx.try_recv().is_ok() {}
    }

    /// Runs `robot` until `done` holds or `limit` passes, carrying out its
    /// actions as a board would. Once it runs, it ticks the part each loop
    /// and publishes a sample every 20 ms.
    async fn run(
        &mut self,
        robot: &mut Robot,
        limit: Duration,
        seen: &Arc<Mutex<Seen>>,
        mut done: impl FnMut(&Robot) -> bool + Send,
    ) {
        let began = Instant::now();
        let start = self.start;
        let now_us = || u64::try_from(start.elapsed().as_micros()).expect("µs");
        while began.elapsed() < limit && !done(robot) {
            if robot.phase() == Phase::Running {
                let now = now_us();
                if let Some(tick) = robot.tick(0, now) {
                    let mut seen = seen.lock().expect("seen");
                    seen.mode = Some(tick.tick.mode);
                    if let Some(action) = &tick.tick.action {
                        seen.actions.push((tick.tick.mode, action.value.to_vec()));
                    }
                }
                {
                    let mut seen = seen.lock().expect("seen");
                    seen.latched_by = robot.estop_reason().map(|r| r.by.clone());
                    seen.refused = (
                        robot.last_refusal(),
                        robot.refusals(Rejection::InvalidValue),
                    );
                    seen.timing = robot.chunk_timing(0);
                    seen.accepted = robot.chunks_accepted();
                    seen.datagram = robot.datagram_stats();
                }
                if now >= self.next_sample {
                    self.next_sample = now + 20_000;
                    if robot.publish(0, now, &pos(0.1, 0.2), now).is_ok() {
                        seen.lock().expect("seen").published += 1;
                    }
                }
            }
            if std::mem::take(&mut self.udp_failed) {
                robot.datagram_failed(now_us());
            }
            for action in robot.poll(now_us()) {
                self.carry_out(action, seen).await;
            }
            let event = tokio::time::timeout(Duration::from_millis(1), self.rx.recv()).await;
            let mut next = match event {
                Ok(Some(event)) => Some(event),
                _ => None,
            };
            while let Some(event) = next {
                let faded = self.faded();
                match event {
                    Event::Sample(..) | Event::Reply(..) if faded => {}
                    Event::Sample(SUB_KEYS, ..)
                        if self.drop_keys.load(std::sync::atomic::Ordering::Relaxed) => {}
                    Event::Sample(sub, key, mut bytes) => {
                        if sub == SUB_VIEW && self.first_sealed.is_some() {
                            self.signed_after_sealed += 1;
                        }
                        if sub == SUB_VIEW_SEALED {
                            if self.first_sealed.is_none() {
                                self.first_sealed = Some((key.clone(), bytes.clone()));
                            }
                            self.last_sealed = Some((key.clone(), bytes.clone()));
                            self.take(Took::SealedView, || {
                                robot.sample(sub, &key, &mut bytes, now_us());
                            });
                        } else if self.check_views && sub == SUB_VIEW {
                            self.check_view(robot, &key, bytes, now_us());
                        } else if sub == SUB_VIEW {
                            self.take(Took::SignedView, || {
                                robot.sample(sub, &key, &mut bytes, now_us());
                            });
                        } else if sub >= SUB_COMMANDS {
                            let key_id = rlmesh_session::crypto::sealed::Header::parse(&bytes)
                                .map_or(0, |h| h.key_id);
                            let before = robot.refusals(Rejection::Unauthenticated);
                            robot.sample(sub, &key, &mut bytes, now_us());
                            let refused = robot.refusals(Rejection::Unauthenticated) > before;
                            self.chunks.lock().expect("chunks").push((
                                Instant::now(),
                                key_id,
                                refused,
                            ));
                        } else {
                            robot.sample(sub, &key, &mut bytes, now_us());
                        }
                    }
                    Event::Reply(id, bytes) => {
                        let rng = &mut self.rng;
                        let started = Instant::now();
                        let ((), allocs) = counted(|| robot.reply(id, &bytes, now_us(), rng));
                        self.taken.lock().expect("taken").push(Taken {
                            at: started,
                            what: Took::Reply,
                            took: started.elapsed(),
                            allocs,
                        });
                    }
                    Event::Done(id) => {
                        self.queries.remove(&id);
                        if !self.no_query_end {
                            robot.query_done(id, now_us());
                        }
                    }
                    Event::Datagram(bytes) => {
                        if !faded && !self.lose_down.load(std::sync::atomic::Ordering::Relaxed) {
                            self.datagrams
                                .lock()
                                .expect("datagrams")
                                .push(bytes.clone());
                            robot.datagram(&bytes, now_us());
                        }
                    }
                }
                next = self.rx.try_recv().ok();
            }
        }
    }

    /// A signed view as a board's transport takes it: verified apart from
    /// the robot, which is handed only what passed.
    fn check_view(&mut self, robot: &mut Robot, key: &str, mut bytes: Vec<u8>, now_us: u64) {
        if let Some(checker) = robot.take_checker() {
            self.checker = Some(checker);
            self.checkers += 1;
        }
        let Some(checker) = self.checker.as_mut() else {
            return;
        };
        if !robot.wants_signed_views() {
            // The transport drops it unchecked.
            return;
        }
        self.checks += 1;
        if robot.view_counts().0 > 0 {
            self.checks_after_sealed += 1;
        }
        if std::mem::take(&mut self.forge_next_view) {
            bytes[20] ^= 1;
        }
        match checker.check(key, bytes) {
            Ok(checked) => robot.sample_checked(SUB_VIEW, &checked, now_us),
            Err(_) => self.views_refused += 1,
        }
    }

    async fn carry_out(&mut self, action: Action, seen: &Arc<Mutex<Seen>>) {
        let class = |c: u8| TrafficClass::ALL[usize::from(c)];
        match action {
            Action::Reset => {
                self.subscribers.clear();
                self.tokens.clear();
            }
            Action::Reconnect => {
                self.sever();
                if self.offline_until.is_none_or(|at| Instant::now() >= at) {
                    self.transport = Some(links(self.base).await);
                    self.reconnects += 1;
                }
            }
            Action::Subscribe { class: c, sub, key } => {
                let Some(transport) = &self.transport else {
                    return;
                };
                if sub >= SUB_COMMANDS && sub != SUB_VIEW_SEALED {
                    self.command_subscriptions += 1;
                }
                let tx = self.tx.clone();
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
                self.subscribers.push(subscriber);
            }
            Action::Token { class: c, key } => {
                let Some(transport) = &self.transport else {
                    return;
                };
                let token = transport.session(class(c)).liveliness().declare_token(key);
                self.tokens.push(token.await.expect("token"));
            }
            Action::Query {
                class: c,
                id,
                timeout_ms,
                key,
                payload,
            } => {
                let tx = self.tx.clone();
                let (pico, oversized) = (self.pico, self.oversized.clone());
                let zenoh = match &self.transport {
                    Some(transport) if !(self.lose)(&key) && !self.faded() => {
                        transport.session(class(c)).clone()
                    }
                    _ => {
                        let _ = tx.send(Event::Done(id));
                        return;
                    }
                };
                self.queries.insert(
                    id,
                    tokio::spawn(async move {
                        // The key alone, of the reply's envelope.
                        let fits = |bytes: &[u8]| !pico || bytes.len() + key.len() <= PICO_MESSAGE;
                        let mut get = zenoh
                            .get(key.clone())
                            .consolidation(ConsolidationMode::None)
                            .timeout(Duration::from_millis(u64::from(timeout_ms)));
                        if let Some(payload) = payload {
                            get = get.payload(payload);
                        }
                        if let Ok(replies) = get.await {
                            while let Ok(reply) = replies.recv_async().await {
                                if let Ok(sample) = reply.result() {
                                    let bytes = sample.payload().to_bytes().into_owned();
                                    if fits(&bytes) {
                                        let _ = tx.send(Event::Reply(id, bytes));
                                    } else {
                                        oversized
                                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    }
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
                let Some(transport) = &self.transport else {
                    return;
                };
                if self.faded() {
                    return;
                }
                if key.ends_with("/eyes") {
                    seen.lock().expect("seen").sent += 1;
                }
                transport
                    .session(class(c))
                    .put(key, payload)
                    .await
                    .expect("publish");
            }
            Action::DatagramOpen { port } => {
                self.close_udp();
                let socket = tokio::net::UdpSocket::bind("127.0.0.1:0")
                    .await
                    .expect("udp");
                socket
                    .connect((self.base.ip(), port))
                    .await
                    .expect("connect");
                let socket = Arc::new(socket);
                let reader = tokio::spawn({
                    let (socket, tx) = (socket.clone(), self.tx.clone());
                    async move {
                        let mut buf = vec![0; 1473];
                        while let Ok(n) = socket.recv(&mut buf).await {
                            let _ = tx.send(Event::Datagram(buf[..n].to_vec()));
                        }
                    }
                });
                self.udp = Some((socket, reader));
            }
            Action::Datagram { lane, payload } => {
                let Some((socket, _)) = &self.udp else {
                    self.udp_failed = true;
                    return;
                };
                *self
                    .sent_up
                    .lock()
                    .expect("sent")
                    .entry(lane.clone())
                    .or_default() += 1;
                let probe = lane == rlmesh_device::robot::PROBE_LANE;
                let lose = if probe {
                    self.last_probe = Some(payload.clone());
                    self.lose_probes.load(std::sync::atomic::Ordering::Relaxed)
                } else {
                    self.lose_up.load(std::sync::atomic::Ordering::Relaxed)
                };
                if self.faded() || lose {
                    return;
                }
                if socket.send(&payload).await.is_err() {
                    self.udp_failed = true;
                }
            }
            Action::DatagramClose => self.close_udp(),
        }
    }

    fn close_udp(&mut self) {
        if let Some((_, reader)) = self.udp.take() {
            reader.abort();
        }
    }
}

impl Drop for Board {
    fn drop(&mut self) {
        for query in self.queries.values() {
            query.abort();
        }
        self.close_udp();
    }
}

/// Runs `robot` over a fresh transport to the world's host until `done`
/// holds or `limit` passes (see [`Board::run`]).
async fn drive(
    world: &World,
    mut robot: Box<Robot>,
    limit: Duration,
    seen: Arc<Mutex<Seen>>,
    done: impl FnMut(&Robot) -> bool + Send,
) -> Box<Robot> {
    Board::open(world)
        .await
        .run(&mut robot, limit, &seen, done)
        .await;
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_board_takes_views_checked_on_another_task_and_never_a_forged_one() {
    // Signed views only, as from a host that doesn't seal them.
    let world = World::start_unsealed().await;
    let mut robot = world.robot(AddressKey::of(&world.identity));
    let mut board = Board::open(&world).await;
    board.check_views = true;
    let seen = Arc::default();
    board
        .run(&mut robot, Duration::from_secs(10), &seen, |r| {
            r.phase() == Phase::Running && r.views() >= 2
        })
        .await;
    assert_eq!(robot.phase(), Phase::Running, "fault: {:?}", robot.fault());
    assert!(robot.views() >= 2, "{} views", robot.views());
    assert_eq!(board.checkers, 1, "one checker per link");
    assert!(robot.take_checker().is_none(), "already taken");
    let before = robot.view().expect("a view").host_ns;

    board.forge_next_view = true;
    let views = robot.views();
    board
        .run(&mut robot, Duration::from_secs(10), &seen, |r| {
            r.views() > views
        })
        .await;
    assert_eq!(board.views_refused, 1, "the forged view was refused");
    assert!(robot.view().expect("a view").host_ns > before);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_running_board_is_next_due_by_its_heartbeat_and_its_control_tick() {
    let world = World::start().await;
    let mut robot = world.robot(AddressKey::of(&world.identity));
    let mut board = Board::open(&world).await;
    board
        .run(&mut robot, Duration::from_secs(10), &Arc::default(), |r| {
            r.phase() == Phase::Running
        })
        .await;
    assert_eq!(robot.phase(), Phase::Running, "fault: {:?}", robot.fault());
    // The board's clock, as its run read it.
    let now_us = || u64::try_from(board.start.elapsed().as_micros()).expect("µs");
    // Drained: nothing waits, so the next due moment is the heartbeat's, at
    // most a period away.
    let now = now_us();
    let _ = robot.poll(now);
    let due = robot.next_due().expect("a heartbeat is scheduled");
    assert!(
        due <= now + 50_000,
        "due {} us after now",
        due.saturating_sub(now)
    );
    // Once ticked, the part's next tick (50 Hz) bounds it too.
    let part = robot.part("eyes").expect("part");
    let ticked = now_us();
    let _ = robot.tick(part, ticked);
    let _ = robot.poll(ticked);
    let due = robot.next_due().expect("due");
    assert!(
        due <= ticked + 20_000,
        "due {} us after the tick",
        due.saturating_sub(ticked)
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_board_takes_sealed_views_and_checks_no_signature_after_the_first() {
    let world = World::start().await;
    let mut robot = world.robot(AddressKey::of(&world.identity));
    let mut board = Board::open(&world).await;
    board.check_views = true;
    let seen = Arc::default();
    board
        .run(&mut robot, Duration::from_secs(15), &seen, |r| {
            r.phase() == Phase::Running && r.view_counts().0 >= 4
        })
        .await;
    assert_eq!(robot.phase(), Phase::Running, "fault: {:?}", robot.fault());
    let (sealed, _, refused) = robot.view_counts();
    assert!(sealed >= 4, "{sealed} sealed views");
    assert_eq!(refused, 0);
    assert!(!robot.wants_signed_views());
    assert_eq!(
        board.checks_after_sealed, 0,
        "no signature check after the first sealed view ({} before)",
        board.checks
    );
    assert_eq!(
        board.signed_after_sealed, 0,
        "no signed view sent once it takes sealed ones"
    );

    // The first sealed view again: a replay, refused and counted.
    let (key, mut first) = board.first_sealed.clone().expect("a sealed view");
    let held = robot.view().expect("a view").revision;
    robot.sample(SUB_VIEW_SEALED, &key, &mut first, 0);
    assert_eq!(robot.view_counts().2, 1, "the replay is refused");
    assert_eq!(robot.view().expect("a view").revision, held);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_board_falls_back_to_signed_views_from_a_host_that_doesnt_seal() {
    let world = World::start_unsealed().await;
    let mut robot = world.robot(AddressKey::of(&world.identity));
    let mut board = Board::open(&world).await;
    board.check_views = true;
    let seen = Arc::default();
    board
        .run(&mut robot, Duration::from_secs(15), &seen, |r| {
            r.phase() == Phase::Running && r.view_counts().1 >= 2
        })
        .await;
    assert_eq!(robot.phase(), Phase::Running, "fault: {:?}", robot.fault());
    let (sealed, signed, _) = robot.view_counts();
    assert_eq!(sealed, 0);
    assert!(signed >= 2, "{signed} signed views");
    assert!(robot.wants_signed_views());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_board_latched_before_it_joins_still_joins_and_reports_the_latch() {
    let world = World::start().await;
    let mut robot = world.robot(AddressKey::of(&world.identity));
    // Its e-stop pin held at boot: latched before it has a link, and read
    // as pressed on every loop after. One latch, its first reason.
    robot.estop("pin held at boot", 0);
    for t in 1..1000 {
        robot.estop("pin still held", t);
    }
    assert_eq!(
        robot.estop_reason().map(|r| r.detail.as_str()),
        Some("pin held at boot")
    );
    let robot = drive(
        &world,
        robot,
        Duration::from_secs(10),
        Arc::default(),
        |r| r.phase() == Phase::Running,
    )
    .await;
    assert_eq!(robot.phase(), Phase::Running, "fault: {:?}", robot.fault());
    assert!(
        robot.estop_reason().is_some(),
        "the latch outlives the join"
    );
    let snapshot = world.host.snapshot();
    assert!(
        snapshot.estop_latched,
        "the host learns the latch from the attach"
    );
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
    // Every accepted chunk was timed, and a later refusal (the e-stop's) only
    // received.
    let timing = seen.timing.expect("the part's chunk timing");
    assert_eq!(timing.accepted, seen.accepted);
    assert!(timing.received >= timing.accepted && timing.accepted > 0);
    assert_eq!(
        u64::from(timing.late_ms.iter().sum::<u32>()),
        timing.accepted
    );
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

/// A board provisioned for the host joins with its own
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

/// A board provisioned for its home joins the home's default session
/// directly, with its own device key and no lobby (only a `fleet` build
/// waits there), so one firmware serves every eye.
#[cfg(not(feature = "fleet"))]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_provisioned_board_joins_its_homes_default_session_without_the_lobby() {
    let private = [0x43_u8; 32];
    let board = Device {
        id: "board-eye-1".into(),
        public: rlmesh_device::provision::public_key(&private),
        access: DeviceAccess::new(Some(PriorityClass::Model)),
    };
    let (transport, base) = open_local_listeners("127.0.0.1".parse().expect("ip"))
        .await
        .expect("host transport");
    let tenant = TenantId::new(rlmesh_device::address::TENANT).expect("tenant");
    let session = SessionId::named(rlmesh_device::address::DEFAULT_SESSION);
    let mut config = HostConfig::new(tenant.clone(), session);
    config.devices = vec![board];
    let identity = config.identity.clone();
    let host = SessionHost::start(transport, config).await.expect("host");
    let world = World {
        tenant,
        session,
        base,
        identity,
        host,
        relay: None,
    };
    let record = Record {
        name: "eye-1".into(),
        payload: Vec::new(),
        host_key: *world.identity.static_public(),
        host_id: world.identity.host_id(),
        addresses: vec![base.to_string()],
    };
    let config =
        Config::provisioned(&private, &record, MAP.to_owned(), 50).expect("a provisioned config");
    let robot = Robot::new(config).expect("a valid map");
    let robot = drive(
        &world,
        robot,
        Duration::from_secs(10),
        Arc::default(),
        |r| r.phase() == Phase::Running,
    )
    .await;
    assert_eq!(robot.phase(), Phase::Running, "fault: {:?}", robot.fault());
    let eye = world
        .host
        .snapshot()
        .roster
        .into_iter()
        .find(|p| p.id.as_str() == "eye-1")
        .expect("the board is in the default session");
    assert_eq!(eye.device, "board-eye-1");
}

/// One policy for eyes `eye-1` and `eye-2`: no robot named, so it binds
/// the session's view, each action naming its eye.
const EYES_POLICY: &str = r#"{
    "hz": 25,
    "actions": {
        "one": {"kind": "joints", "at": "eye-1:eyes", "command": "position", "hz": 50,
                "chunk": 5},
        "two": {"kind": "joints", "at": "eye-2:eyes", "command": "position", "hz": 50,
                "chunk": 5}
    }
}"#;

/// Both eyes to 0.25 rad pan and tilt, for the whole chunk.
struct Gaze;

impl rlmesh_session::Predictor for Gaze {
    type Error = String;

    fn predict(
        &mut self,
        _: rlmesh_session::Observation,
        _: i64,
    ) -> Result<Vec<(String, rlmesh_session::spec::apply::Value)>, String> {
        use rlmesh_session::spec::apply::{Tensor, Value};
        use rlmesh_session::wire::DType;
        let gaze = || Value::Tensor(Tensor::from_f64(DType::F32, vec![5, 2], &[0.25; 10]));
        Ok(vec![("one".into(), gaze()), ("two".into(), gaze())])
    }
}

/// Two boards and one policy driving both: board 2 goes off the network
/// (its loop stops) and the host loses it, board 1 keeps moving, and once
/// board 2 is back it rejoins on its own and, with robots re-arming on
/// their own, is driven again.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn one_policy_drives_two_boards_and_a_board_back_from_the_network_is_driven_again() {
    let (transport, base) = open_local_listeners("127.0.0.1".parse().expect("ip"))
        .await
        .expect("host transport");
    let tenant = TenantId::new("test").expect("tenant");
    let session = SessionId::new();
    let mut config = HostConfig::new(tenant.clone(), session);
    config.rearm_robots = true;
    let identity = config.identity.clone();
    let host = SessionHost::start(transport, config).await.expect("host");
    let world = Arc::new(World {
        tenant,
        session,
        base,
        identity,
        host,
        relay: None,
    });
    let board = |name: &str| {
        Robot::new(Config {
            tenant: world.tenant.clone(),
            session: world.session,
            name: ParticipantId::new(name).expect("id"),
            credential: AddressKey::of(&world.identity).credential(),
            robot_json: MAP.to_owned(),
            heartbeat_ms: 50,
            lobby: false,
            homes: vec![(world.base.ip().to_string(), world.base.port())],
            home_id: String::new(),
        })
        .expect("a valid map")
    };
    let (seen_one, seen_two): (Arc<Mutex<Seen>>, Arc<Mutex<Seen>>) = Default::default();
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let one = tokio::spawn({
        let (world, seen, stop) = (world.clone(), seen_one.clone(), stop.clone());
        let robot = board("eye-1");
        async move {
            drive(&world, robot, Duration::from_secs(60), seen, move |_| {
                stop.load(std::sync::atomic::Ordering::Relaxed)
            })
            .await
        }
    });
    // Board 2 runs until both boards are driven, then drops off.
    let driven = |seen: &Arc<Mutex<Seen>>| {
        let seen = seen.lock().expect("seen");
        seen.actions
            .iter()
            .rev()
            .take(3)
            .filter(|(mode, value)| *mode == EdgeMode::Driving && value[..] == pos(0.25, 0.25))
            .count()
            == 3
    };
    let two_session = tokio::spawn({
        let (world, seen, seen_one) = (world.clone(), seen_two.clone(), seen_one.clone());
        let robot = board("eye-2");
        async move {
            drive(&world, robot, WAIT, seen.clone(), move |_| {
                driven(&seen) && driven(&seen_one)
            })
            .await
        }
    });
    let arm = world.controller().await;
    arm.arm().await.expect("arm");
    let policy = {
        let mut config = ParticipantConfig::new(
            world.tenant.clone(),
            world.session,
            ParticipantId::new("tracker").expect("id"),
            vec![Role::Controller],
            PriorityClass::Model,
        );
        config.heartbeat_period = Duration::from_millis(20);
        config.credential = Some(world.host.credential());
        let p = Participant::attach(world.transport().await, config)
            .await
            .expect("tracker");
        rlmesh_session::Controller::bind(
            Arc::new(p),
            &rlmesh_session::spec::Policy::parse(EYES_POLICY).expect("policy"),
            &rlmesh_session::spec::RunOptions::named("tracker"),
            rlmesh_session::ControllerConfig::default(),
        )
        .await
        .expect("one binding over both boards")
    };
    let policy = Arc::new(policy);
    let run = std::thread::spawn({
        let policy = policy.clone();
        move || {
            policy.run(
                &mut Gaze,
                rlmesh_session::RunConfig::default().duration(Duration::from_secs(60)),
            )
        }
    });
    let two = two_session.await.expect("board 2");
    assert_eq!(two.phase(), Phase::Running, "fault: {:?}", two.fault());

    // Board 2 is off the network: the host loses it, board 1 goes on.
    let eye_two = ParticipantId::new("eye-2").expect("id");
    let started = Instant::now();
    while world
        .host
        .snapshot()
        .participant(&eye_two)
        .is_some_and(|p| p.present)
    {
        assert!(started.elapsed() < WAIT, "the host never lost board 2");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    // Board 1 takes more actions with board 2 gone, however slowly a loaded
    // machine delivers them.
    let before = seen_one.lock().expect("seen").actions.len();
    let started = Instant::now();
    while seen_one.lock().expect("seen").actions.len() <= before + 10 {
        assert!(
            started.elapsed() < WAIT,
            "board 1 keeps moving while board 2 is gone"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(
        driven(&seen_one),
        "board 1 keeps moving while board 2 is gone"
    );
    assert_ne!(world.host.snapshot().mode, Mode::Safe);

    // Back: it finds the host gone quiet, joins again and is driven again.
    seen_two.lock().expect("seen").actions.clear();
    let two = drive(
        &world,
        two,
        Duration::from_secs(40),
        seen_two.clone(),
        |_| driven(&seen_two),
    )
    .await;
    let snapshot = world.host.snapshot();
    assert!(
        driven(&seen_two),
        "board 2 is driven again: phase {:?}, fault {:?}, last {:?}; host: {:?} roster {:?}, \
         leases {:?}, policy holds {:?}",
        two.phase(),
        two.fault(),
        seen_two.lock().expect("seen").actions.last(),
        snapshot.mode,
        snapshot
            .roster
            .iter()
            .map(|p| (p.id.to_string(), p.present, p.generation))
            .collect::<Vec<_>>(),
        snapshot
            .leases
            .iter()
            .map(|l| (l.channel, l.token))
            .collect::<Vec<_>>(),
        policy.leases().len()
    );
    assert!(driven(&seen_one), "board 1 never stopped");

    policy.stop();
    run.join().expect("run thread").expect("a clean stop");
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    one.await.expect("board 1");
}

/// A policy that reads the board's part as an input (latest, below its 50 Hz)
/// and commands it.
const READING_POLICY: &str = r#"{
    "hz": 30,
    "inputs": {"eyes": {"kind": "joints", "at": "eye-1:eyes"}},
    "actions": {
        "one": {"kind": "joints", "at": "eye-1:eyes", "command": "position", "hz": 50,
                "chunk": 5}
    }
}"#;

/// A part bound as a policy's input: the board installs the host's delivery
/// plan and acknowledges it, so the lease is granted, and then publishes at
/// the plan's rate rather than its own.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_policy_reading_a_boards_part_gets_its_lease_and_the_plan_applies() {
    let world = Arc::new(World::start().await);
    let seen = Arc::new(Mutex::new(Seen::default()));
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let board = tokio::spawn({
        let (world, seen, stop) = (world.clone(), seen.clone(), stop.clone());
        let robot = Robot::new(Config {
            tenant: world.tenant.clone(),
            session: world.session,
            name: ParticipantId::new("eye-1").expect("id"),
            credential: AddressKey::of(&world.identity).credential(),
            robot_json: MAP.to_owned(),
            heartbeat_ms: 50,
            lobby: false,
            homes: vec![(world.base.ip().to_string(), world.base.port())],
            home_id: String::new(),
        })
        .expect("a valid map");
        async move {
            drive(&world, robot, Duration::from_secs(40), seen, move |_| {
                stop.load(std::sync::atomic::Ordering::Relaxed)
            })
            .await
        }
    });
    let id = ParticipantId::new("eye-1").expect("id");
    let arm = world.controller().await;
    let snapshot = arm
        .wait_for("the board, declared and mapped", WAIT, |s| {
            s.parts.iter().any(|p| p.owner == id) && s.clock(&id).is_some()
        })
        .await
        .expect("board");
    let channel = snapshot
        .parts
        .iter()
        .find(|p| p.owner == id)
        .expect("eyes")
        .channel;
    arm.arm().await.expect("arm");
    let policy = {
        let mut config = ParticipantConfig::new(
            world.tenant.clone(),
            world.session,
            ParticipantId::new("tracker").expect("id"),
            vec![Role::Controller],
            PriorityClass::Model,
        );
        config.heartbeat_period = Duration::from_millis(20);
        config.credential = Some(world.host.credential());
        let p = Participant::attach(world.transport().await, config)
            .await
            .expect("tracker");
        rlmesh_session::Controller::bind(
            Arc::new(p),
            &rlmesh_session::spec::Policy::parse(READING_POLICY).expect("policy"),
            &rlmesh_session::spec::RunOptions::named("tracker"),
            rlmesh_session::ControllerConfig::default(),
        )
        .await
        .expect("a binding with an input")
    };
    let started = Instant::now();
    let refused = loop {
        match policy.acquire().await {
            Ok(_) => break None,
            Err(err) if started.elapsed() > WAIT => break Some(err),
            Err(_) => tokio::time::sleep(Duration::from_millis(50)).await,
        }
    };
    assert!(refused.is_none(), "never granted: {refused:?}");
    let demand = world
        .host
        .snapshot()
        .demand(channel)
        .cloned()
        .expect("the input's demand");
    assert_eq!(demand.acked, demand.revision, "{demand:?}");
    assert_eq!(demand.plan, Plan::Latest { hz: 30.0 });

    // It offers 50 Hz: at most 30 go out, whatever else the machine is doing.
    let before = seen.lock().expect("seen").sent;
    let started = Instant::now();
    tokio::time::sleep(Duration::from_secs(2)).await;
    let sent = seen.lock().expect("seen").sent - before;
    let bound = started.elapsed().as_secs_f64() * 30.0 + 3.0;
    assert!(
        f64::from(sent) <= bound && sent > 0,
        "{sent} samples sent, at most {bound}"
    );
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    board.await.expect("board");
}

/// A host restarted under a board that was not: the links are dead and only
/// the board can reopen them, then it joins the new instance.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_board_reopens_its_links_and_rejoins_a_restarted_host() {
    let world = World::start().await;
    let seen = Arc::default();
    let mut robot = world.robot(AddressKey::of(&world.identity));
    let mut board = Board::open(&world).await;
    board
        .run(&mut robot, WAIT, &seen, |r| r.phase() == Phase::Running)
        .await;
    assert_eq!(robot.phase(), Phase::Running, "fault: {:?}", robot.fault());
    // Past the time silence would count: links that answer are left alone.
    let joined = Instant::now();
    board
        .run(&mut robot, Duration::from_secs(6), &seen, move |_| {
            joined.elapsed() > Duration::from_millis(5500)
        })
        .await;
    assert_eq!(board.reconnects, 0, "fault: {:?}", robot.fault());
    assert_eq!(robot.attaches(), 1);
    // Bytes that open under no key: a refusal this attach counts.
    let at = u64::try_from(board.start.elapsed().as_micros()).expect("µs");
    robot.sample(SUB_COMMANDS, "not/a/chunk", &mut [1, 2, 3], at);
    assert_eq!(robot.refusals(Rejection::Unauthenticated), 1);

    let world = world.restart().await;
    board.sever();
    // The silent case: the dead link reports nothing, not even a query's end.
    board.no_query_end = true;
    let mut left = false;
    board
        .run(&mut robot, Duration::from_secs(30), &seen, |r| {
            left |= r.phase() != Phase::Running;
            left && r.phase() == Phase::Running
        })
        .await;
    assert_eq!(robot.phase(), Phase::Running, "fault: {:?}", robot.fault());
    assert!(board.reconnects >= 1, "it reopened its links");
    // This attach's refusals start over; the lifetime's are kept.
    assert_eq!(robot.attaches(), 2);
    assert_eq!(robot.refusals(Rejection::Unauthenticated), 0);
    assert_eq!(robot.chunks_refused(), 0);
    assert_eq!(robot.refusals_total(Rejection::Unauthenticated), 1);
    assert_eq!(robot.chunks_refused_total(), 1);
    let roster = world.host.snapshot().roster;
    assert!(roster.iter().any(|p| p.id.as_str() == "eyes" && p.present));
}

/// Wi-Fi gone long enough for the host to lose the board, then back, with
/// the host up all along: it reopens its links, retrying until they open.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_board_rejoins_after_the_network_returns_with_the_host_still_up() {
    let world = World::start().await;
    let seen = Arc::default();
    let mut robot = world.robot(AddressKey::of(&world.identity));
    let mut board = Board::open(&world).await;
    board
        .run(&mut robot, WAIT, &seen, |r| r.phase() == Phase::Running)
        .await;
    assert_eq!(robot.phase(), Phase::Running, "fault: {:?}", robot.fault());
    let first = robot.generation();

    board.sever();
    board.offline_until = Some(Instant::now() + Duration::from_secs(9));
    let mut left = false;
    board
        .run(&mut robot, Duration::from_secs(40), &seen, |r| {
            left |= r.phase() != Phase::Running;
            left && r.phase() == Phase::Running
        })
        .await;
    assert_eq!(robot.phase(), Phase::Running, "fault: {:?}", robot.fault());
    assert!(robot.generation() > first, "a new attach");
}

/// zenoh-pico's executor stuck in `recv`: no query ever reports its end, and
/// the first host query and the first `ctl` request are lost. The robot's own
/// deadlines carry it through both.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_board_joins_though_lost_queries_never_report_their_end() {
    let world = World::start().await;
    let seen = Arc::default();
    let mut robot = world.robot(AddressKey::of(&world.identity));
    let mut board = Board::open(&world).await;
    board.no_query_end = true;
    let (mut host_lost, mut ctl_lost) = (false, false);
    board.lose = Box::new(move |key| {
        if key.ends_with("/host/state") && !host_lost {
            host_lost = true;
            return true;
        }
        if key.ends_with("/ctl") && !ctl_lost {
            ctl_lost = true;
            return true;
        }
        false
    });
    board
        .run(&mut robot, Duration::from_secs(20), &seen, |r| {
            r.phase() == Phase::Running
        })
        .await;
    assert_eq!(robot.phase(), Phase::Running, "fault: {:?}", robot.fault());
}

/// A session whose state outgrew what a board's transport takes in (here a
/// policy's binding table alone fills it; in a session driven for a while,
/// its participants, parts, bindings and ended leases): the board still reads
/// the host instance and joins, and a full participant still reads the whole
/// state.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_board_joins_a_session_whose_state_outgrew_its_transport() {
    let world = World::start().await;
    let policy = world.controller().await;
    let table = format!(r#"{{"note": "{}"}}"#, "x".repeat(PICO_MESSAGE));
    policy
        .declare_binding(Fingerprint::from(7), &table, Vec::new(), 0.0)
        .await
        .expect("binding");
    let state = policy.fetch_authority().await.expect("the state");
    assert!(
        state.bindings.iter().any(|b| b.table_json == table),
        "a full reader gets the whole state"
    );
    let state = rlmesh_session::participant::query_authority(
        &world.transport().await,
        world.host.keys(),
        WAIT,
    )
    .await
    .expect("the state, unverified");
    assert!(state.bindings.iter().any(|b| b.table_json == table));

    let mut robot = world.robot(AddressKey::of(&world.identity));
    let mut board = Board::open(&world).await;
    board.pico = true;
    board
        .run(&mut robot, WAIT, &Arc::default(), |r| {
            r.phase() == Phase::Running
        })
        .await;
    assert_eq!(robot.phase(), Phase::Running, "fault: {:?}", robot.fault());
    assert!(
        board.oversized.load(std::sync::atomic::Ordering::Relaxed) > 0,
        "the state itself never reached the board"
    );
}

/// Refused in a session whose state is larger than a board reassembles, a
/// board still hears why: the refusal carries the reason, not the state.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_board_refused_in_a_session_whose_state_outgrew_its_transport_hears_why() {
    // Enrolled for everything but embodying a robot: its attach is refused.
    let private = [0x44_u8; 32];
    let mut access = DeviceAccess::new(Some(PriorityClass::Model));
    access.roles = vec![Role::Controller];
    let world = World::start_with(vec![Device {
        id: "board-eyes".into(),
        public: rlmesh_device::provision::public_key(&private),
        access,
    }])
    .await;
    let policy = world.controller().await;
    let table = format!(r#"{{"note": "{}"}}"#, "x".repeat(PICO_MESSAGE));
    policy
        .declare_binding(Fingerprint::from(7), &table, Vec::new(), 0.0)
        .await
        .expect("binding");

    let mut key = private;
    let credential = Credential::new(
        StaticKeypair::from_private(&mut key),
        *world.identity.static_public(),
    );
    let mut robot = world.robot_as(credential);
    let mut board = Board::open(&world).await;
    board.pico = true;
    board
        .run(&mut robot, WAIT, &Arc::default(), |r| r.fault().is_some())
        .await;
    match robot.fault() {
        Some(Fault::Refused(status, why)) => {
            assert_eq!(*status, CtlStatus::NotAuthorized, "{why}");
            assert!(why.contains("may not attach as embodiment"), "{why}");
        }
        other => panic!("refused without a reason: {other:?} in {:?}", robot.phase()),
    }
}

/// A chunk the edge refuses says why, and what can't be opened at all is
/// counted the same way.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_board_says_why_it_refused_a_chunk_and_counts_each_reason() {
    let world = World::start().await;
    let seen = Arc::new(Mutex::new(Seen::default()));
    let mut robot = world.robot(AddressKey::of(&world.identity));
    let mut board = Board::open(&world).await;
    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let driven = async {
        board
            .run(&mut robot, Duration::from_secs(20), &seen, {
                let done = done.clone();
                move |_| done.load(std::sync::atomic::Ordering::Relaxed)
            })
            .await;
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
        let eyes = snapshot
            .parts
            .iter()
            .find(|p| p.owner == id)
            .expect("eyes")
            .channel;
        policy
            .declare_binding(Fingerprint::from(1), "{}", vec![eyes], 0.0)
            .await
            .expect("binding");
        policy.arm().await.expect("arm");
        let lease = policy.grant_lease(eyes, None).await.expect("lease");
        let tx = policy.chunk_publisher(&lease).await.expect("publisher");
        let started = Instant::now();
        // A value that isn't a number: well formed, sealed, and refused for
        // it. Chunks that reach the board before its clock mapping or the
        // lease are refused for that instead: sent until one is refused
        // for its value, and the board says so.
        let bad = pos(f32::NAN, 0.0);
        let refused_for_value = || {
            let seen = seen.lock().expect("seen");
            seen.refused.0 == Some(Rejection::InvalidValue) && seen.refused.1 > 0
        };
        while !refused_for_value() {
            assert!(
                started.elapsed() < WAIT,
                "the board never refused a chunk for its value: {:?}",
                seen.lock().expect("seen").refused
            );
            let now = policy.session_now_ns();
            tx.send(ChunkDraft {
                based_on: vec![InputRange::one(ChannelId(99), 1)],
                t0_ns: now + 1_000_000,
                dt_ns: 20_000_000,
                replace_from_ns: now,
                valid_until_ns: now + 200_000_000,
                actions: Bytes::from([bad; 5].concat()),
                tick: 0,
            })
            .expect("send");
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        done.store(true, std::sync::atomic::Ordering::Relaxed);
        // Kept past the join: dropping the last handle of a Zenoh session
        // closes it in a nested `block_on` on this thread, which takes the
        // wakeup the board loop's timer sends meanwhile, and the loop
        // would never run again.
        (policy, tx)
    };
    let (_, _kept) = tokio::time::timeout(Duration::from_secs(60), async {
        tokio::join!(driven, controller)
    })
    .await
    .expect("the board loop and the controller finish");
    assert!(robot.refusals(Rejection::InvalidValue) >= 1);
    // Each refusal is counted once, under its reason.
    let by_reason = || {
        Rejection::ALL
            .iter()
            .map(|r| robot.refusals(*r))
            .sum::<u64>()
    };
    let refused = robot.chunks_refused();
    assert_eq!(by_reason(), refused);

    // Bytes that open under no key are the same kind of news.
    let unauthenticated = robot.refusals(Rejection::Unauthenticated);
    let at = u64::try_from(board.start.elapsed().as_micros()).expect("µs");
    robot.sample(SUB_COMMANDS, "not/a/chunk", &mut [1, 2, 3], at);
    assert_eq!(robot.last_refusal(), Some(Rejection::Unauthenticated));
    assert_eq!(
        robot.refusals(Rejection::Unauthenticated),
        unauthenticated + 1
    );
    assert_eq!(robot.chunks_refused(), refused + 1);
}

/// The demo's camera robot.
const CAMERA_PARTS: &str = r#"{"rgb": {"kind": "camera", "width": 32, "height": 24, "hz": 30}}"#;

/// The demo's tracker: three custom streams.
const TRACKER_PARTS: &str = r#"{
    "target": {"kind": "track/target", "hz": 30, "fields": {"xy": ["float32", [2]]}},
    "face": {"kind": "track/face", "hz": 30, "fields": {"box": ["float32", [4]]}},
    "hand": {"kind": "track/hand", "hz": 30, "fields": {"xyz": ["float32", [3]]}}
}"#;

/// The demo's viewer: six inputs, no actions.
const VIEWER_POLICY: &str = r#"{
    "hz": 10,
    "inputs": {
        "rgb": {"kind": "camera", "at": "cam:rgb"},
        "one": {"kind": "joints", "at": "eye-1:eyes"},
        "two": {"kind": "joints", "at": "eye-2:eyes"},
        "target": {"kind": "track/target", "at": "tracker:target",
                   "fields": {"xy": ["float32", [2]]}},
        "face": {"kind": "track/face", "at": "tracker:face", "fields": {"box": ["float32", [4]]}},
        "hand": {"kind": "track/hand", "at": "tracker:hand", "fields": {"xyz": ["float32", [3]]}}
    }
}"#;

impl World {
    /// A host whose robots re-arm on their own, as the demo's does.
    async fn start_rearming() -> Self {
        let (transport, base) = open_local_listeners("127.0.0.1".parse().expect("ip"))
            .await
            .expect("host transport");
        let tenant = TenantId::new("test").expect("tenant");
        let session = SessionId::new();
        let mut config = HostConfig::new(tenant.clone(), session);
        config.rearm_robots = true;
        let identity = config.identity.clone();
        let host = SessionHost::start(transport, config).await.expect("host");
        Self {
            tenant,
            session,
            base,
            identity,
            host,
            relay: None,
        }
    }

    async fn join(&self, id: &str, roles: Vec<Role>, class: PriorityClass) -> Participant {
        let mut config = ParticipantConfig::new(
            self.tenant.clone(),
            self.session,
            ParticipantId::new(id).expect("id"),
            roles,
            class,
        );
        config.heartbeat_period = Duration::from_millis(20);
        config.credential = Some(self.host.credential());
        Participant::attach(self.transport().await, config)
            .await
            .expect("participant")
    }

    fn board(&self, name: &str) -> Box<Robot> {
        Robot::new(Config {
            tenant: self.tenant.clone(),
            session: self.session,
            name: ParticipantId::new(name).expect("id"),
            credential: AddressKey::of(&self.identity).credential(),
            robot_json: MAP.to_owned(),
            heartbeat_ms: 50,
            lobby: false,
            homes: vec![(self.base.ip().to_string(), self.base.port())],
            home_id: String::new(),
        })
        .expect("a valid map")
    }

    /// Waits until the host's state satisfies `done`.
    async fn until(
        &self,
        what: &str,
        done: impl Fn(&rlmesh_session::wire::AuthoritySnapshot) -> bool,
    ) {
        let started = Instant::now();
        while !done(&self.host.snapshot()) {
            assert!(started.elapsed() < WAIT, "never: {what}");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}

/// A board running on its own task until `stop`, with its log of what it
/// took and its fades.
struct Running {
    task: tokio::task::JoinHandle<(Box<Robot>, Board)>,
    taken: Arc<Mutex<Vec<Taken>>>,
    fades: Arc<Mutex<Option<(Duration, Duration)>>>,
    chunks: Arc<Mutex<Vec<(Instant, u32, bool)>>>,
    drop_keys: Arc<std::sync::atomic::AtomicBool>,
    seen: Arc<Mutex<Seen>>,
}

async fn run_board(
    world: &Arc<World>,
    mut robot: Box<Robot>,
    stop: &Arc<std::sync::atomic::AtomicBool>,
) -> Running {
    let mut board = Board::open(world).await;
    let (taken, fades, chunks) = (
        board.taken.clone(),
        board.fades.clone(),
        board.chunks.clone(),
    );
    let drop_keys = board.drop_keys.clone();
    let seen: Arc<Mutex<Seen>> = Arc::default();
    let task = tokio::spawn({
        let (seen, stop) = (seen.clone(), stop.clone());
        async move {
            board
                .run(&mut robot, Duration::from_secs(120), &seen, move |_| {
                    stop.load(std::sync::atomic::Ordering::Relaxed)
                })
                .await;
            (robot, board)
        }
    });
    Running {
        task,
        taken,
        fades,
        chunks,
        drop_keys,
        seen,
    }
}

/// Views taken per second, replies per second, and what one view cost at
/// most, over `taken`.
fn rates(taken: &[Taken]) -> (f64, f64, Allocs, Duration) {
    let (Some(first), Some(last)) = (taken.first(), taken.last()) else {
        return (0.0, 0.0, Allocs::default(), Duration::ZERO);
    };
    let span = last.at.duration_since(first.at).as_secs_f64().max(1e-3);
    let views: Vec<&Taken> = taken
        .iter()
        .filter(|t| t.what == Took::SealedView)
        .collect();
    let replies = taken.iter().filter(|t| t.what == Took::Reply).count();
    let worst = views.iter().fold(Allocs::default(), |w, t| Allocs {
        count: w.count.max(t.allocs.count),
        bytes: w.bytes.max(t.allocs.bytes),
        largest: w.largest.max(t.allocs.largest),
        peak: w.peak.max(t.allocs.peak),
        live: 0,
    });
    let slowest = views.iter().map(|t| t.took).max().unwrap_or_default();
    (
        views.len() as f64 / span,
        replies as f64 / span,
        worst,
        slowest,
    )
}

/// The demo's session: two boards with a commanded part each, a camera
/// robot, a tracker publishing three parts, a desk, a driver commanding both
/// boards and a viewer reading six parts. What a board's view carries and
/// costs it, and how it grows as each joins.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_boards_view_in_the_demo_session() {
    use rlmesh_session_core::wire::{DeviceView, codec};
    let world = Arc::new(World::start_rearming().await);
    let eye = ParticipantId::new("eye-1").expect("id");
    let view_of = |world: &World| DeviceView::of(&world.host.snapshot(), &eye);
    let mut growth = Vec::new();
    let mut grew = |label: &str, world: &World| {
        let view = view_of(world);
        growth.push((
            label.to_owned(),
            codec::encode_view(&view).len(),
            view.parts.len(),
            view.members.len(),
            view.clocks.len(),
            view.leases.len(),
        ));
    };
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let one = run_board(&world, world.board("eye-1"), &stop).await;
    world
        .until("eye-1 declared and mapped", |s| s.clock(&eye).is_some())
        .await;
    grew("eye-1", &world);
    let two = run_board(&world, world.board("eye-2"), &stop).await;
    let eye_two = ParticipantId::new("eye-2").expect("id");
    world
        .until("eye-2 declared and mapped", |s| s.clock(&eye_two).is_some())
        .await;
    grew("eye-2", &world);
    let cam = world
        .join("cam", vec![Role::Embodiment], PriorityClass::Scripted)
        .await;
    cam.declare_parts(CAMERA_PARTS).await.expect("camera");
    grew("cam", &world);
    let tracker = world
        .join("tracker", vec![Role::Embodiment], PriorityClass::Scripted)
        .await;
    tracker.declare_parts(TRACKER_PARTS).await.expect("tracker");
    grew("tracker", &world);
    let desk = world
        .join("desk", vec![Role::Controller], PriorityClass::Operator)
        .await;
    desk.arm().await.expect("arm");
    grew("desk", &world);
    let driver = rlmesh_session::Controller::bind(
        Arc::new(
            world
                .join("driver", vec![Role::Controller], PriorityClass::Model)
                .await,
        ),
        &rlmesh_session::spec::Policy::parse(EYES_POLICY).expect("policy"),
        &rlmesh_session::spec::RunOptions::named("driver"),
        rlmesh_session::ControllerConfig::default(),
    )
    .await
    .expect("the driver binds both boards");
    let driver = Arc::new(driver);
    let run = std::thread::spawn({
        let driver = driver.clone();
        move || {
            driver.run(
                &mut Gaze,
                rlmesh_session::RunConfig::default().duration(Duration::from_secs(120)),
            )
        }
    });
    let driven = |seen: &Arc<Mutex<Seen>>| {
        seen.lock()
            .expect("seen")
            .actions
            .iter()
            .rev()
            .take(3)
            .filter(|(mode, value)| *mode == EdgeMode::Driving && value[..] == pos(0.25, 0.25))
            .count()
            == 3
    };
    let started = Instant::now();
    while !(driven(&one.seen) && driven(&two.seen)) {
        assert!(
            started.elapsed() < WAIT,
            "the driver never drove both boards"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    grew("driver", &world);
    let viewer = rlmesh_session::Controller::bind(
        Arc::new(
            world
                .join("viewer", vec![Role::Controller], PriorityClass::Model)
                .await,
        ),
        &rlmesh_session::spec::Policy::parse(VIEWER_POLICY).expect("policy"),
        &rlmesh_session::spec::RunOptions::named("viewer"),
        rlmesh_session::ControllerConfig::default(),
    )
    .await
    .expect("the viewer binds six inputs");
    world
        .until("the viewer's demands on both boards", |s| {
            s.demands.len() >= 2
        })
        .await;
    grew("viewer", &world);

    // Driven, steady: what each view costs eye-1, and how often one comes.
    one.taken.lock().expect("taken").clear();
    let started = Instant::now();
    while one
        .taken
        .lock()
        .expect("taken")
        .iter()
        .filter(|t| t.what == Took::SealedView)
        .count()
        < 100
    {
        assert!(started.elapsed() < WAIT, "eye-1 takes views");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let steady = one.taken.lock().expect("taken").clone();

    // eye-2 on a fading link: 400 ms of nothing every 1.5 s.
    two.taken.lock().expect("taken").clear();
    *two.fades.lock().expect("fades") =
        Some((Duration::from_millis(1500), Duration::from_millis(400)));
    let token = |s: &rlmesh_session::wire::AuthoritySnapshot| {
        s.parts
            .iter()
            .find(|p| p.owner == eye_two)
            .and_then(|p| s.lease(p.channel))
            .map(|l| l.token)
    };
    one.taken.lock().expect("taken").clear();
    // Until its lease lapsed and was granted again three times.
    let mut tokens = std::collections::BTreeSet::new();
    let started = Instant::now();
    while tokens.len() < 4 {
        assert!(
            started.elapsed() < Duration::from_secs(30),
            "eye-2's lease never lapses"
        );
        tokens.extend(token(&world.host.snapshot()));
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    *two.fades.lock().expect("fades") = None;
    let fading = two.taken.lock().expect("taken").clone();
    let beside = one.taken.lock().expect("taken").clone();

    let snapshot = world.host.snapshot();
    let view = DeviceView::of(&snapshot, &eye);
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    driver.stop();
    let _ = run.join();
    drop(viewer);
    let (one_robot, one_board) = one.task.await.expect("eye-1");
    let _ = two.task.await.expect("eye-2");
    let sealed = one_board.last_sealed.clone().expect("a sealed view").1;
    let views = |taken: &[Taken]| -> Vec<Taken> {
        taken
            .iter()
            .filter(|t| t.what == Took::SealedView)
            .copied()
            .collect()
    };
    let (steady_rate, steady_replies, worst, slowest) = rates(&steady);
    let (fading_rate, fading_replies, _, _) = rates(&fading);
    let (beside_rate, _, _, _) = rates(&beside);
    let steady = views(&steady);
    let free = steady.iter().filter(|t| t.allocs.count == 0).count();
    eprintln!(
        "eye-1's view: {} B sealed ({} B with its part); host-side it has {} part spec(s), \
         {} members, {} clocks, {} lease(s), {} grid(s), {} plan(s)",
        sealed.len(),
        codec::encode_view(&view).len(),
        view.parts.len(),
        view.members.len(),
        view.clocks.len(),
        view.leases.len(),
        view.grids.len(),
        view.demands.len(),
    );
    eprintln!(
        "driven: {steady_rate:.1} views/s, {steady_replies:.1} replies/s; {free} of {} views \
         allocated nothing, the most any did: {worst:?}; slowest taken in {slowest:?}",
        steady.len()
    );
    eprintln!(
        "eye-2 on a fading link: {fading_rate:.1} views/s, {fading_replies:.1} replies/s, {} \
         leases in {:.1} s; eye-1 meanwhile {beside_rate:.1} views/s",
        tokens.len(),
        fading
            .last()
            .zip(fading.first())
            .map_or(0.0, |(l, f)| l.at.duration_since(f.at).as_secs_f64())
    );
    eprintln!("as each joined (who, bytes with its part, parts, members, clocks, leases):");
    for row in &growth {
        eprintln!("  {row:?}");
    }
    // Its own part only, and none of it on the wire after its declaration.
    assert!(view.parts.iter().all(|p| p.owner == eye));
    assert!(
        sealed.len() + 150 < codec::encode_view(&view).len(),
        "the sealed view carries no part"
    );
    assert_eq!(
        view.members.len(),
        7,
        "every present member, for its e-stop keys"
    );
    // A view like the last allocates nothing; one that grows the view's
    // room or changes the keys held, a few small blocks.
    assert!(
        free * 10 >= steady.len() * 9,
        "{free} of {} views allocated nothing",
        steady.len()
    );
    // Refinements at most every quarter of the lease grace, plus changes.
    assert!(
        steady_rate < 20.0 && beside_rate < 20.0,
        "{steady_rate} and {beside_rate} views/s"
    );
    assert_eq!(one_robot.view_counts().2, 0, "eye-1 dropped no view");
}

/// A board that resets while driven (its loop stops dead, and a new process
/// joins under its name before the host has counted the old one lost)
/// refuses as unauthenticated only the chunks already in flight under its
/// old generation's lease: the driver stops sending under a lease the
/// moment its view shows it ended, even while its lease keeper is busy
/// with another robot's grant (here one whose key deliveries are lost, so
/// each grant waits out its time).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_board_back_from_a_reset_refuses_only_the_chunks_already_in_flight() {
    let world = Arc::new(World::start_rearming().await);
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let one_stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let one = run_board(&world, world.board("eye-1"), &one_stop).await;
    let two = run_board(&world, world.board("eye-2"), &stop).await;
    let (eye, eye_two) = (
        ParticipantId::new("eye-1").expect("id"),
        ParticipantId::new("eye-2").expect("id"),
    );
    world
        .until("both boards mapped", |s| {
            s.clock(&eye).is_some() && s.clock(&eye_two).is_some()
        })
        .await;
    let desk = world
        .join("desk", vec![Role::Controller], PriorityClass::Operator)
        .await;
    desk.arm().await.expect("arm");
    let driver = Arc::new(
        rlmesh_session::Controller::bind(
            Arc::new(
                world
                    .join("driver", vec![Role::Controller], PriorityClass::Model)
                    .await,
            ),
            &rlmesh_session::spec::Policy::parse(EYES_POLICY).expect("policy"),
            &rlmesh_session::spec::RunOptions::named("driver"),
            // Leases that outlast a reset: the driver must stop on the
            // view, not on its lease running out.
            rlmesh_session::ControllerConfig {
                lease_ttl: Some(Duration::from_secs(2)),
                ..rlmesh_session::ControllerConfig::default()
            },
        )
        .await
        .expect("the driver binds both boards"),
    );
    let run = std::thread::spawn({
        let driver = driver.clone();
        move || {
            driver.run(
                &mut Gaze,
                rlmesh_session::RunConfig::default().duration(Duration::from_secs(120)),
            )
        }
    });
    let driven = |seen: &Arc<Mutex<Seen>>| {
        seen.lock()
            .expect("seen")
            .actions
            .iter()
            .rev()
            .take(3)
            .filter(|(mode, value)| *mode == EdgeMode::Driving && value[..] == pos(0.25, 0.25))
            .count()
            == 3
    };
    let started = Instant::now();
    while !(driven(&one.seen) && driven(&two.seen)) {
        assert!(
            started.elapsed() < WAIT,
            "the driver never drove both boards"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let lease = |s: &rlmesh_session::wire::AuthoritySnapshot, robot: &ParticipantId| {
        s.parts
            .iter()
            .find(|p| &p.owner == robot)
            .and_then(|p| s.lease(p.channel))
            .map(|l| l.key_id)
    };

    // Every lease again, eye-2's grant waiting on keys it never gets: the
    // driver's lease keeper stays busy with it.
    two.drop_keys
        .store(true, std::sync::atomic::Ordering::Relaxed);
    desk.disarm().await.expect("disarm");
    desk.arm().await.expect("arm");
    world
        .until("eye-1 leased again, eye-2 not", |s| {
            lease(s, &eye).is_some() && lease(s, &eye_two).is_none()
        })
        .await;
    let before = lease(&world.host.snapshot(), &eye).expect("eye-1's lease");
    let started = Instant::now();
    while !driven(&one.seen) {
        assert!(started.elapsed() < WAIT, "eye-1 is driven again");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // The reset: nothing more in or out of the old process, and a new one
    // joins at once.
    *one.fades.lock().expect("fades") =
        Some((Duration::from_secs(3600), Duration::from_secs(3600)));
    let back = run_board(&world, world.board("eye-1"), &stop).await;
    world
        .until("eye-1's new process declared", |s| {
            s.participant(&eye)
                .is_some_and(|p| p.present && p.generation > 1)
                && lease(s, &eye) != Some(before)
        })
        .await;
    // Its commands subscribed, then two seconds: the old lease's whole ttl.
    let started = Instant::now();
    while back.chunks.lock().expect("chunks").is_empty() && started.elapsed() < WAIT {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(2) {
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let chunks = back.chunks.lock().expect("chunks").clone();
    let refused: Vec<u32> = chunks.iter().filter(|c| c.2).map(|c| c.1).collect();
    eprintln!(
        "after its reset eye-1 took {} chunks and refused {} as unauthenticated, under keys {:?}; \
         its lease before was under key {before}",
        chunks.len(),
        refused.len(),
        refused.iter().collect::<std::collections::BTreeSet<_>>()
    );
    assert!(
        refused.len() <= 3,
        "{} chunks under the old generation's lease refused",
        refused.len()
    );

    two.drop_keys
        .store(false, std::sync::atomic::Ordering::Relaxed);
    one_stop.store(true, std::sync::atomic::Ordering::Relaxed);
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    driver.stop();
    let _ = run.join();
    let _ = one.task.await;
    let _ = back.task.await;
    let _ = two.task.await;
}

/// A controller that leases the board's part and sends it a chunk of four
/// actions every 33 ms until `done` holds or [`WAIT`] passes: whether it
/// held, and the newest sample of the part the controller received.
async fn drive_board(
    world: &World,
    seen: &Arc<Mutex<Seen>>,
    mut done: impl FnMut(&Seen, bool) -> bool,
) -> (bool, Option<Vec<u8>>) {
    let id = ParticipantId::new("eyes").expect("id");
    let policy = world.controller().await;
    let snapshot = policy
        .wait_for("the board, declared and mapped", WAIT, |s| {
            s.parts.iter().any(|p| p.owner == id) && s.clock(&id).is_some()
        })
        .await
        .expect("board");
    let eyes = snapshot
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
    let target = pos(0.25, -0.1);
    let deadline = Instant::now() + WAIT;
    let mut held = false;
    while Instant::now() < deadline {
        let now = policy.session_now_ns();
        let _ = tx.send(ChunkDraft {
            based_on: vec![InputRange::one(ChannelId(99), 1)],
            t0_ns: now,
            dt_ns: 20_000_000,
            replace_from_ns: now,
            valid_until_ns: now + 200_000_000,
            actions: Bytes::from([target; 4].concat()),
            tick: 0,
        });
        tokio::time::sleep(Duration::from_millis(33)).await;
        if done(&seen.lock().expect("seen"), samples.latest().is_some()) {
            held = true;
            break;
        }
    }
    let latest = samples.latest().map(|s| s.received.sample.payload.to_vec());
    // Done driving: a lease that just stops sending looks like a lost path
    // to a board on the datagram path.
    let _ = policy.release_lease(&lease).await;
    (held, latest)
}

/// The board's health row at the host.
fn board_health(world: &World) -> Option<rlmesh_session::ParticipantHealth> {
    world
        .host
        .health()
        .into_iter()
        .find(|row| row.participant.as_str() == "eyes")
}

/// Waits for the host's health of the board to satisfy `ok`.
async fn wait_health(
    world: &World,
    what: &str,
    ok: impl Fn(&rlmesh_session::ParticipantHealth) -> bool,
) -> rlmesh_session::ParticipantHealth {
    let mut watch = world.host.watch_health();
    let deadline = Instant::now() + WAIT;
    loop {
        if let Some(row) = board_health(world).filter(|r| ok(r)) {
            return row;
        }
        let left = deadline.saturating_duration_since(Instant::now());
        assert!(
            !left.is_zero(),
            "the host's health never showed {what}: {:?}",
            board_health(world)
        );
        let _ = tokio::time::timeout(left, watch.changed()).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_board_on_the_datagram_path_takes_chunks_and_its_samples_reach_a_policy_and_the_recorder()
{
    use rlmesh_session::wire::DataPath;
    let world = World::start_datagram().await;
    let seen = Arc::new(Mutex::new(Seen::default()));
    let mut robot = world.robot(AddressKey::of(&world.identity));
    let mut board = Board::open(&world).await;
    // A recorder, as the host's own would: an observer of every stream.
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("session.mcap");
    let observer = {
        let mut config = ParticipantConfig::new(
            world.tenant.clone(),
            world.session,
            ParticipantId::new("recorder").expect("id"),
            vec![Role::Observer],
            PriorityClass::Scripted,
        );
        config.credential = Some(world.host.credential());
        Participant::attach(world.transport().await, config)
            .await
            .expect("observer")
    };
    let recorder = rlmesh_session::record::Recorder::start(Arc::new(observer), &path)
        .await
        .expect("recorder");
    let (held, latest) = tokio::select! {
        driven = drive_board(&world, &seen, |seen, sample| {
            sample && seen.datagram.is_some_and(|d| d.received >= 20)
                && seen.actions.iter().any(|(m, _)| *m == EdgeMode::Driving)
        }) => driven,
        () = board.run(&mut robot, Duration::from_secs(30), &seen, |_| false) => {
            panic!("the board's loop ended first: {:?}", robot.fault())
        }
    };
    assert!(
        held,
        "never driven over the datagram path: {:?}",
        seen.lock().expect("seen").datagram
    );
    let stats = robot.datagram_stats().expect("offered");
    assert!(stats.active && stats.unopened == 0, "{stats:?}");
    assert_eq!(robot.datagram_fallbacks(), 0);
    // Never both paths: no command subscription, no sample on the session.
    assert_eq!(board.command_subscriptions, 0);
    assert_eq!(
        seen.lock().expect("seen").sent,
        0,
        "samples published over TCP"
    );
    let lanes = board.sent_up.lock().expect("sent").clone();
    assert!(
        lanes.keys().any(|k| k.ends_with("/eyes")) && lanes.keys().any(|k| k.ends_with("_applied")),
        "{lanes:?}"
    );
    // The policy received the board's samples, relayed by the host.
    assert_eq!(latest.as_deref(), Some(&pos(0.1, 0.2)[..]));
    let row = tokio::select! {
        row = wait_health(&world, "the datagram path", |r| {
            r.data_path == DataPath::Datagram
                && r.datagram_host.is_some_and(|h| h.received > 0 && h.sent > 0)
        }) => row,
        () = board.run(&mut robot, Duration::from_secs(30), &seen, |_| false) => panic!("loop ended"),
    };
    let host = row.datagram_host.expect("host counters");
    assert!(
        host.active && host.sent > 0 && host.unopened == 0,
        "{host:?}"
    );
    let stats = recorder.finish().await.expect("finish");
    assert!(stats.messages > 0, "{stats:?}");
    let summary = rlmesh_session::record::summarize(&path).expect("summary");
    let samples = summary
        .topics
        .iter()
        .find(|t| t.topic == "/eyes/eyes")
        .unwrap_or_else(|| panic!("no samples recorded: {:#?}", summary.topics));
    assert!(
        samples.messages > 0 && samples.undecodable == 0,
        "{samples:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn flipping_the_hosts_board_data_and_reattaching_switches_paths() {
    let world = World::start_datagram().await;
    let seen = Arc::new(Mutex::new(Seen::default()));
    let mut robot = world.robot(AddressKey::of(&world.identity));
    let mut board = Board::open(&world).await;
    let on_udp =
        |r: &Robot| r.phase() == Phase::Running && r.datagram_stats().is_some_and(|d| d.active);
    board.run(&mut robot, WAIT, &seen, on_udp).await;
    assert!(on_udp(&robot), "{:?}", robot.fault());
    let attaches = robot.attaches();

    // The host comes back without the datagram path: the board re-attaches
    // on TCP, and is driven there.
    let world = world.restart().await;
    board
        .run(&mut robot, Duration::from_secs(20), &seen, |r| {
            r.attaches() > attaches && r.phase() == Phase::Running
        })
        .await;
    assert_eq!(robot.phase(), Phase::Running, "{:?}", robot.fault());
    assert_eq!(robot.datagram_stats(), None, "nothing offered");
    let accepted = robot.chunks_accepted();
    let subscriptions = board.command_subscriptions;
    let (held, _) = tokio::select! {
        driven = drive_board(&world, &seen, |seen, _| seen.accepted > accepted + 10) => driven,
        () = board.run(&mut robot, Duration::from_secs(30), &seen, |_| false) => panic!("loop ended"),
    };
    assert!(held, "not driven over TCP after the switch");
    assert_eq!(board.command_subscriptions, subscriptions + 1);

    // And with it again: back on UDP.
    let attaches = robot.attaches();
    let relay = DatagramRelay::bind("127.0.0.1:0".parse().expect("addr"))
        .await
        .expect("udp");
    let world = world.restart_with(Some(relay)).await;
    board
        .run(&mut robot, Duration::from_secs(20), &seen, |r| {
            r.attaches() > attaches && on_udp(r)
        })
        .await;
    assert!(on_udp(&robot), "{:?}", robot.fault());
    let accepted = robot.chunks_accepted();
    let subscriptions = board.command_subscriptions;
    let (held, _) = tokio::select! {
        driven = drive_board(&world, &seen, |seen, _| {
            seen.accepted > accepted + 10 && seen.datagram.is_some_and(|d| d.received > 10)
        }) => driven,
        () = board.run(&mut robot, Duration::from_secs(30), &seen, |_| false) => panic!("loop ended"),
    };
    assert!(held, "not driven over UDP after the switch back");
    assert_eq!(
        board.command_subscriptions, subscriptions,
        "no TCP subscription"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_board_or_host_without_the_path_keeps_every_part_on_tcp() {
    use rlmesh_session::wire::DataPath;
    // A board that does not take it (as one from before it), with a host
    // that offers it.
    let world = World::start_datagram().await;
    let seen = Arc::new(Mutex::new(Seen::default()));
    let mut robot = world.robot(AddressKey::of(&world.identity));
    robot.use_datagram(false);
    let mut board = Board::open(&world).await;
    let (held, _) = tokio::select! {
        driven = drive_board(&world, &seen, |seen, sample| sample && seen.accepted > 5) => driven,
        () = board.run(&mut robot, Duration::from_secs(30), &seen, |_| false) => panic!("loop ended"),
    };
    assert!(held);
    assert_eq!(robot.datagram_stats(), None);
    assert!(board.udp.is_none() && board.sent_up.lock().expect("sent").is_empty());
    tokio::select! {
        row = wait_health(&world, "the session's links", |r| {
            r.data_path == DataPath::Session && r.datagram_host.is_none()
        }) => row,
        () = board.run(&mut robot, Duration::from_secs(30), &seen, |_| false) => panic!("loop ended"),
    };
    drop(board);

    // A host that offers nothing, with a board that would take it.
    let world = World::start().await;
    let seen = Arc::new(Mutex::new(Seen::default()));
    let mut robot = world.robot(AddressKey::of(&world.identity));
    let mut board = Board::open(&world).await;
    let (held, _) = tokio::select! {
        driven = drive_board(&world, &seen, |seen, sample| sample && seen.accepted > 5) => driven,
        () = board.run(&mut robot, Duration::from_secs(30), &seen, |_| false) => panic!("loop ended"),
    };
    assert!(held);
    assert_eq!(robot.datagram_stats(), None);
    assert!(board.udp.is_none());
    assert_eq!(board.command_subscriptions, 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_hostile_datagram_changes_nothing_but_a_counter() {
    let world = World::start_datagram().await;
    let seen = Arc::new(Mutex::new(Seen::default()));
    let mut robot = world.robot(AddressKey::of(&world.identity));
    let mut board = Board::open(&world).await;
    let (held, _) = tokio::select! {
        driven = drive_board(&world, &seen, |seen, sample| {
            sample && seen.datagram.is_some_and(|d| d.received >= 10)
        }) => driven,
        () = board.run(&mut robot, Duration::from_secs(30), &seen, |_| false) => panic!("loop ended"),
    };
    assert!(held);
    let routes = robot
        .view()
        .and_then(|v| v.datagram.clone())
        .expect("routes in the view");
    let up = routes
        .routes
        .iter()
        .find(|r| r.up)
        .expect("an up route")
        .route;
    let port = board
        .udp
        .as_ref()
        .expect("socket")
        .0
        .peer_addr()
        .expect("peer")
        .port();
    let host_before = tokio::select! {
        row = wait_health(&world, "the host's counters", |r| r.datagram_host.is_some()) => row,
        () = board.run(&mut robot, Duration::from_secs(30), &seen, |_| false) => panic!("loop ended"),
    }
    .datagram_host
    .expect("host counters");

    // At the host, from a stranger: garbage, a sample that opens under no
    // key, a captured probe again, and one too large. The host answers none.
    let rogue = tokio::net::UdpSocket::bind("127.0.0.1:0")
        .await
        .expect("udp");
    rogue.connect(("127.0.0.1", port)).await.expect("connect");
    let header = |kind: u8, route: u16| {
        let mut h = vec![0xD1, kind];
        h.extend_from_slice(&route.to_le_bytes());
        h.extend_from_slice(&7_u32.to_le_bytes());
        h
    };
    let probe = board.last_probe.clone().expect("a probe the board sent");
    let mut forged = header(2, up);
    forged.extend([0x5a; 120]);
    for datagram in [b"not a datagram".to_vec(), forged, probe, vec![0xD1; 1500]] {
        rogue.send(&datagram).await.expect("send");
    }
    let mut buf = [0; 1500];
    assert!(
        tokio::time::timeout(Duration::from_millis(300), rogue.recv(&mut buf))
            .await
            .is_err(),
        "the host answered a stranger"
    );

    // At the board, as if from the host's address: garbage, a chunk
    // datagram again, one too large.
    let replay = board
        .datagrams
        .lock()
        .expect("datagrams")
        .last()
        .cloned()
        .expect("a chunk datagram");
    let (accepted, before) = (
        robot.chunks_accepted(),
        robot.datagram_stats().expect("stats"),
    );
    for datagram in [b"garbage".to_vec(), replay, vec![0xD1; 1472]] {
        robot.datagram(&datagram, 0);
    }
    let after = robot.datagram_stats().expect("stats");
    assert_eq!(after.unopened, before.unopened + 3, "{after:?}");
    assert_eq!(after.received, before.received);
    assert_eq!(robot.chunks_accepted(), accepted);
    assert_eq!(robot.phase(), Phase::Running);

    // The host counted what did not open (the garbage and the oversize one
    // name no route), and kept sending to the board: its chunks still arrive.
    let received = after.received;
    let (held, _) = tokio::select! {
        driven = drive_board(&world, &seen, |seen, _| {
            seen.datagram.is_some_and(|d| d.received >= received + 10)
        }) => driven,
        () = board.run(&mut robot, Duration::from_secs(30), &seen, |_| false) => panic!("loop ended"),
    };
    assert!(held, "the board's chunks stopped after the forgeries");
    let row = tokio::select! {
        row = wait_health(&world, "the forged sample and the replayed probe", |r| {
            r.datagram_host
                .is_some_and(|h| h.unopened >= host_before.unopened + 2)
        }) => row,
        () = board.run(&mut robot, Duration::from_secs(30), &seen, |_| false) => panic!("loop ended"),
    };
    let host = row.datagram_host.expect("host counters");
    assert!(host.active && host.fallbacks == 0, "{host:?}");
    let relay = world.relay.as_ref().expect("relay");
    assert!(relay.unrouted() >= 2, "{} unrouted", relay.unrouted());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_board_whose_chunk_datagrams_stop_falls_back_to_tcp_and_tells_the_host() {
    use rlmesh_session::wire::DataPath;
    use std::sync::atomic::Ordering;
    let world = World::start_datagram().await;
    let seen = Arc::new(Mutex::new(Seen::default()));
    let mut robot = world.robot(AddressKey::of(&world.identity));
    let mut board = Board::open(&world).await;
    let lose_down = board.lose_down.clone();
    let accepted_on_tcp = Arc::new(std::sync::atomic::AtomicU64::new(0));
    // Driven over UDP; then every datagram from the host is lost while the
    // control link stays healthy: the board goes back to TCP, and is driven
    // there.
    let (held, _) = tokio::select! {
        driven = drive_board(&world, &seen, |seen, _| {
            let Some(d) = seen.datagram else { return false };
            if d.received >= 10 {
                lose_down.store(true, Ordering::Relaxed);
            }
            if d.active || d.fallbacks == 0 {
                accepted_on_tcp.store(seen.accepted, Ordering::Relaxed);
                return false;
            }
            seen.accepted > accepted_on_tcp.load(Ordering::Relaxed) + 10
        }) => driven,
        () = board.run(&mut robot, Duration::from_secs(30), &seen, |_| false) => panic!("loop ended"),
    };
    assert!(
        held,
        "not driven over TCP after the fallback: {:?}",
        robot.datagram_stats()
    );
    assert_eq!(robot.datagram_fallbacks(), 1);
    let stats = robot.datagram_stats().expect("stats");
    assert!(!stats.active && stats.fallbacks == 1, "{stats:?}");
    assert_eq!(board.command_subscriptions, 1, "subscribed over TCP once");
    assert!(board.udp.is_none(), "the socket closed");
    // The host heard it in the heartbeats, and stopped relaying.
    let row = tokio::select! {
        row = wait_health(&world, "the board's fallback", |r| {
            r.data_path == DataPath::Session && r.datagram_board.is_some_and(|b| b.fallbacks == 1)
        }) => row,
        () = board.run(&mut robot, Duration::from_secs(30), &seen, |_| false) => panic!("loop ended"),
    };
    assert!(!row.datagram_host.expect("host").active || row.data_path == DataPath::Session);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_host_that_hears_no_samples_over_the_path_moves_the_board_back_to_tcp() {
    use rlmesh_session::wire::DataPath;
    use std::sync::atomic::Ordering;
    let world = World::start_datagram().await;
    let seen = Arc::new(Mutex::new(Seen::default()));
    let mut robot = world.robot(AddressKey::of(&world.identity));
    let mut board = Board::open(&world).await;
    // Its samples never reach the host over UDP (its probes do).
    board.lose_up.store(true, Ordering::Relaxed);
    let on_udp = |r: &Robot| r.datagram_stats().is_some_and(|d| d.active);
    board.run(&mut robot, WAIT, &seen, on_udp).await;
    assert!(on_udp(&robot), "{:?}", robot.fault());
    board
        .run(&mut robot, WAIT, &seen, |r| {
            r.datagram_stats().is_some_and(|d| !d.active)
        })
        .await;
    let stats = robot.datagram_stats().expect("stats");
    assert!(!stats.active, "{stats:?}");
    assert_eq!(robot.datagram_fallbacks(), 0, "the host's, not the board's");
    let row = tokio::select! {
        row = wait_health(&world, "the host's fallback", |r| {
            r.data_path == DataPath::Session && r.datagram_host.is_some_and(|h| h.fallbacks == 1)
        }) => row,
        () = board.run(&mut robot, Duration::from_secs(30), &seen, |_| false) => panic!("loop ended"),
    };
    assert!(!row.datagram_host.expect("host").active);
    // Its samples reach a policy over TCP, and its chunks too.
    let (held, latest) = tokio::select! {
        driven = drive_board(&world, &seen, |seen, sample| sample && seen.accepted > 5) => driven,
        () = board.run(&mut robot, Duration::from_secs(30), &seen, |_| false) => panic!("loop ended"),
    };
    let refused: Vec<_> = Rejection::ALL
        .iter()
        .map(|r| (*r, robot.refusals_total(*r)))
        .filter(|(_, n)| *n > 0)
        .collect();
    assert!(
        held,
        "accepted {}, refused {refused:?}, sample {latest:?}, {:?}, phase {:?} attaches {}",
        robot.chunks_accepted(),
        robot.datagram_stats(),
        robot.phase(),
        robot.attaches()
    );
    assert_eq!(latest.as_deref(), Some(&pos(0.1, 0.2)[..]));
    assert_eq!(board.command_subscriptions, 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_board_on_the_datagram_path_rides_out_an_outage_of_every_link() {
    use rlmesh_session::wire::DataPath;
    let world = World::start_datagram().await;
    let seen = Arc::new(Mutex::new(Seen::default()));
    let mut robot = world.robot(AddressKey::of(&world.identity));
    let mut board = Board::open(&world).await;
    let fades = board.fades.clone();
    let start = board.start;
    let mut outage: Option<Duration> = None;
    // Driven over UDP; then nothing at all arrives or leaves for 400 ms (the
    // robot goes safe, as it would over TCP), and for a second and a half
    // after: an outage of the whole link is no reason to leave the path.
    let (held, _) = tokio::select! {
        driven = drive_board(&world, &seen, |seen, _| {
            let Some(d) = seen.datagram else { return false };
            let into = start.elapsed();
            if outage.is_none() && d.received >= 10 {
                // One fade, from now, 400 ms long, in a period past the test.
                *fades.lock().expect("fades") =
                    Some((into + Duration::from_secs(600), into + Duration::from_millis(400)));
                outage = Some(into);
            }
            !d.active || outage.is_some_and(|at| into > at + Duration::from_millis(1900))
        }) => driven,
        () = board.run(&mut robot, Duration::from_secs(30), &seen, |_| false) => panic!("loop ended"),
    };
    let stats = robot.datagram_stats().expect("stats");
    assert!(held && stats.active, "{stats:?}");
    assert_eq!(robot.datagram_fallbacks(), 0);
    assert_eq!(board.command_subscriptions, 0);
    tokio::select! {
        row = wait_health(&world, "the path still in use", |r| {
            r.data_path == DataPath::Datagram && r.datagram_host.is_some_and(|h| h.fallbacks == 0)
        }) => row,
        () = board.run(&mut robot, Duration::from_secs(30), &seen, |_| false) => panic!("loop ended"),
    };
}
