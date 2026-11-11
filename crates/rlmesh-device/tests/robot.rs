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
    Action, Config, Fault, Phase, Robot, SUB_COMMANDS, SUB_VIEW, SUB_VIEW_SEALED,
};
use rlmesh_session::address::AddressKey;
use rlmesh_session::auth::{Credential, Device, HostIdentity};
use rlmesh_session::authority::DeviceAccess;
use rlmesh_session::crypto::StaticKeypair;
use rlmesh_session::crypto::{Rng, RngError, SignedChecker};
use rlmesh_session::edge::Rejection;
use rlmesh_session::transport::{TrafficClass, Transport, TransportConfig, open_local_listeners};
use rlmesh_session::wire::{
    ChannelId, EdgeMode, Fingerprint, InputRange, Mode, ParticipantId, Plan, PriorityClass, Role,
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
/// zenoh-pico reassembles a message in 4096 bytes (`Z_FRAG_MAX_SIZE`, as
/// boards are built) and drops a larger one without a word.
const PICO_MESSAGE: usize = 4096;

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
        links(self.base).await
    }

    /// Stops the host and starts a new one at the same address with the same
    /// key: a new process, a new instance, an empty roster.
    async fn restart(self) -> Self {
        let Self {
            tenant,
            session,
            base,
            identity,
            host,
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
        let host = SessionHost::start(transport, config).await.expect("host");
        Self {
            tenant,
            session,
            base,
            identity,
            host,
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
    drop_sealed: bool,
    /// The first sealed view that arrived, kept for a replay.
    first_sealed: Option<(String, Vec<u8>)>,
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
            drop_sealed: false,
            first_sealed: None,
        }
    }

    /// The links die with no word to the board: a host restarted, or Wi-Fi
    /// gone. What is in flight is lost.
    fn sever(&mut self) {
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
                }
                if now >= self.next_sample {
                    self.next_sample = now + 20_000;
                    if robot.publish(0, now, &pos(0.1, 0.2), now).is_ok() {
                        seen.lock().expect("seen").published += 1;
                    }
                }
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
                match event {
                    Event::Sample(sub, key, mut bytes) => {
                        if sub == SUB_VIEW_SEALED {
                            if self.drop_sealed {
                                // A host that doesn't seal.
                            } else {
                                if self.first_sealed.is_none() {
                                    self.first_sealed = Some((key.clone(), bytes.clone()));
                                }
                                robot.sample(sub, &key, &mut bytes, now_us());
                            }
                        } else if self.check_views && sub == SUB_VIEW {
                            self.check_view(robot, &key, bytes, now_us());
                        } else {
                            robot.sample(sub, &key, &mut bytes, now_us());
                        }
                    }
                    Event::Reply(id, bytes) => robot.reply(id, &bytes, now_us(), &mut self.rng),
                    Event::Done(id) => {
                        self.queries.remove(&id);
                        if !self.no_query_end {
                            robot.query_done(id, now_us());
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
                    Some(transport) if !(self.lose)(&key) => transport.session(class(c)).clone(),
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
                if key.ends_with("/eyes") {
                    seen.lock().expect("seen").sent += 1;
                }
                transport
                    .session(class(c))
                    .put(key, payload)
                    .await
                    .expect("publish");
            }
        }
    }
}

impl Drop for Board {
    fn drop(&mut self) {
        for query in self.queries.values() {
            query.abort();
        }
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
    let world = World::start().await;
    let mut robot = world.robot(AddressKey::of(&world.identity));
    let mut board = Board::open(&world).await;
    board.check_views = true;
    // Signed views only, as from a host that doesn't seal them.
    board.drop_sealed = true;
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

    // The first sealed view again: a replay, refused and counted.
    let (key, mut first) = board.first_sealed.clone().expect("a sealed view");
    let held = robot.view().expect("a view").revision;
    robot.sample(SUB_VIEW_SEALED, &key, &mut first, 0);
    assert_eq!(robot.view_counts().2, 1, "the replay is refused");
    assert_eq!(robot.view().expect("a view").revision, held);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_board_falls_back_to_signed_views_from_a_host_that_doesnt_seal() {
    let world = World::start().await;
    let mut robot = world.robot(AddressKey::of(&world.identity));
    let mut board = Board::open(&world).await;
    board.check_views = true;
    board.drop_sealed = true;
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
        move || policy.run(&mut Gaze, Some(Duration::from_secs(60)))
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
        tokio::time::sleep(Duration::from_millis(300)).await;
        let started = Instant::now();
        // A value that isn't a number: well formed, sealed, and refused.
        let bad = pos(f32::NAN, 0.0);
        while seen.lock().expect("seen").refused.0.is_none() {
            assert!(started.elapsed() < WAIT, "the board refused nothing");
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
    };
    tokio::join!(driven, controller);
    assert_eq!(robot.last_refusal(), Some(Rejection::InvalidValue));
    let invalid = robot.refusals(Rejection::InvalidValue);
    assert!(invalid >= 1);
    assert_eq!(robot.chunks_refused(), invalid);

    // Bytes that open under no key are the same kind of news.
    let at = u64::try_from(board.start.elapsed().as_micros()).expect("µs");
    robot.sample(SUB_COMMANDS, "not/a/chunk", &mut [1, 2, 3], at);
    assert_eq!(robot.last_refusal(), Some(Rejection::Unauthenticated));
    assert_eq!(robot.refusals(Rejection::Unauthenticated), 1);
    assert_eq!(robot.chunks_refused(), invalid + 1);
}
