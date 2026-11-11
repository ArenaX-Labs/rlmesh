//! A board as a robot in a session, sans-IO: the board's transport
//! (zenoh-pico) carries out the [`Action`]s it asks for and hands back what
//! arrives. The protocol is session-core's: the handshake and sealing are
//! its [`Link`], the command edge its [`EdgeCore`]; the host normalises the
//! map. This only puts them in the participant's order.
//!
//! The order: subscribe to the host's state, ticks, this
//! robot's heartbeat echo and key deliveries; read the host instance from
//! the state; handshake on `hs`; `Attach` and `DeclareParts` sealed on
//! `ctl`; then build an edge per command group, subscribe to each part's
//! commands, and heartbeat. Any failure starts over from the handshake after
//! [`RETRY_MS`]. While running it installs the delivery plan of each stream
//! it publishes and acknowledges each revision, as a session's publisher
//! does: a lease that depends on the stream waits for that.
//!
//! Traffic rides the host's per-class sessions (one port each, from the
//! address's): control on the first, a part's commands and samples on its
//! priority's ([`Action`]'s `class`).

use alloc::boxed::Box;
use alloc::collections::{BTreeMap, VecDeque};
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use bytes::Bytes;
use rlmesh_session_core::auth::keyring::KeyError;
use rlmesh_session_core::auth::{self, Credential, Link, LinkError, ReplyOutcome};
use rlmesh_session_core::crypto::handshake::IkInitiator;
use rlmesh_session_core::crypto::{Checked, Rng, SignedChecker, signed};
use rlmesh_session_core::edge::{EdgeConfig, EdgeCore, EdgeTick, Rejection};
use rlmesh_session_core::lobby::{self, LOBBY_SESSION, Leave, Lobby, WAIT_RETRY};
use rlmesh_session_core::wire::codec;
use rlmesh_session_core::wire::version::{
    PROTOCOL_MINOR, SPEC_VERSION, VIEW_DEVICE, VIEW_DEVICE_SEALED,
};
use rlmesh_session_core::wire::{
    AuthorityView, ChannelId, ChannelPriority, ClockMode, Command, CtlError, CtlStatus, DeviceView,
    Epoch, Health, Heartbeat, Join, Mode, PartSpec, ParticipantId, Plan, PriorityClass, Reason,
    ReasonCode, Role, Sample, SessionId, SessionKeys, StreamSpec, TenantId, TimingSample,
};

/// How long a query waits for its reply before the transport gives up.
pub const QUERY_MS: u32 = 2000;
/// The pause before starting over after a failure.
pub const RETRY_MS: u32 = 1000;
/// With no heartbeat echo and no valid signed state for this long, the host
/// counts as lost.
pub const HOST_LOST_MS: u32 = 5000;
/// How long past its timeout a query may go unreported before the robot
/// counts it over: a transport whose read task is stuck never reports it.
pub const QUERY_MARGIN_MS: u32 = 500;
/// With no reply from the host for this long, a failure also has the
/// transport drop and reopen its links: nothing else revives a link whose
/// peer is gone.
pub const LINK_LOST_MS: u32 = HOST_LOST_MS;
/// The longest pause between rounds of home addresses.
const MAX_BACKOFF_MS: u32 = 30_000;
/// `ctl` requests are resent (same id) this many times before they fail.
const CTL_TRIES: u8 = 3;
/// How often a robot missing keys asks for them again (`GetKeys`).
const GET_KEYS_MS: u64 = 250;
/// `ctl` requests waiting to be sent, at most: a host that stops answering
/// must not grow a board's memory. Past it the oldest that is not a stop
/// report goes; a stop report replaces the one waiting.
const CTL_QUEUE_MAX: usize = 16;

/// The control class: the host's first port.
pub const CLASS_CONTROL: u8 = 0;

/// Subscription ids, for [`Robot::sample`]: `host/view/{id}`, this robot's
/// signed device view.
pub const SUB_VIEW: u8 = 0;
/// `host/tick`.
pub const SUB_TICK: u8 = 1;
/// `p/{id}/hb/echo`.
pub const SUB_ECHO: u8 = 2;
/// `p/{id}/keys`.
pub const SUB_KEYS: u8 = 3;
/// `estop/*`: every member's e-stop signal. Its samples need their key.
pub const SUB_ESTOP: u8 = 4;
/// The commands of part `i` (in the map's order) are `SUB_COMMANDS + i`.
pub const SUB_COMMANDS: u8 = 5;
/// `p/{id}/view`: this robot's device view sealed under its `k_down`, from
/// a host that seals it ([`VIEW_DEVICE_SEALED`]). Past every part's commands.
pub const SUB_VIEW_SEALED: u8 = 255;

/// What a robot needs to join.
#[derive(Debug, Clone)]
pub struct Config {
    /// The session's tenant (`local` for an embedded host).
    pub tenant: TenantId,
    /// The session.
    pub session: SessionId,
    /// This robot's participant id and display name.
    pub name: ParticipantId,
    /// Its device key and the host's static key: from the address's key
    /// (device `local`), or provisioned.
    pub credential: Credential,
    /// The robot's map, as authored JSON.
    pub robot_json: String,
    /// The heartbeat period.
    pub heartbeat_ms: u32,
    /// Wait in the home's lobby for runs: `tenant` and
    /// `session` are then the lobby's, and each run's come with its join.
    pub lobby: bool,
    /// The home's base addresses (`host`, first port), tried in order: the
    /// address's one, or a provisioned board's stored ones.
    pub homes: Vec<(String, u16)>,
    /// The home's host id (`rlm-…`), for discovery: empty when not known.
    pub home_id: String,
}

impl Config {
    /// A board provisioned for its home: its device key,
    /// and the robot name, home's static key, host id and addresses from the stored
    /// `record`. It joins the home's default session directly, so every
    /// board runs the same firmware and only its provisioning differs; a
    /// `fleet` build waits in the home's lobby instead (unreleased).
    ///
    /// # Errors
    /// [`ConfigError::NoHome`] when the record has no usable address, or
    /// [`ConfigError::Name`] when its robot name is invalid.
    pub fn provisioned(
        device_key: &[u8; 32],
        record: &crate::provision::Record,
        robot_json: String,
        heartbeat_ms: u32,
    ) -> Result<Self, ConfigError> {
        let homes: Vec<(String, u16)> = record
            .addresses
            .iter()
            .filter_map(|a| crate::address::host_port(a))
            .collect();
        if homes.is_empty() {
            return Err(ConfigError::NoHome);
        }
        let mut private = zeroize::Zeroizing::new(*device_key);
        Ok(Self {
            tenant: TenantId::new(crate::address::TENANT)
                .map_err(|_| ConfigError::Map(String::from("tenant")))?,
            session: SessionId::named(crate::address::DEFAULT_SESSION),
            name: ParticipantId::new(record.name.clone()).map_err(|_| ConfigError::Name)?,
            credential: Credential::new(
                rlmesh_session_core::crypto::StaticKeypair::from_private(&mut private),
                record.host_key,
            ),
            robot_json,
            heartbeat_ms,
            lobby: cfg!(feature = "fleet"),
            homes,
            home_id: record.host_id.clone(),
        })
    }
}

/// Why a [`Config`] can't make a robot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    /// The map isn't a JSON object of parts with valid paths (the host
    /// checks the rest at `DeclareParts`).
    Map(String),
    /// A zero heartbeat period.
    Heartbeat,
    /// No home address.
    NoHome,
    /// An invalid provisioned robot participant id.
    Name,
}

/// What the transport is asked to do, on traffic class `class` (0 control,
/// 1 realtime, 2 data, 3 bulk: the address's port plus `class`). Queries
/// carry an id; their reply (or none) comes back through [`Robot::reply`]
/// and [`Robot::query_done`]. The robot doesn't depend on hearing either: a
/// query is over for it at its timeout plus [`QUERY_MARGIN_MS`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// Subscribe to `key`, passing its samples to [`Robot::sample`] as
    /// `sub`.
    Subscribe {
        /// The class.
        class: u8,
        /// What [`Robot::sample`] calls it.
        sub: u8,
        /// The key expression.
        key: String,
    },
    /// Declare a liveliness token on `key`, held while the robot runs.
    Token {
        /// The class.
        class: u8,
        /// The key expression.
        key: String,
    },
    /// Query `key` (with `payload`, if any), for at most `timeout_ms`.
    Query {
        /// The class.
        class: u8,
        /// Names the replies.
        id: u32,
        /// How long to wait for replies: [`QUERY_MS`], or a lobby wait's
        /// [`WAIT_RETRY`].
        timeout_ms: u32,
        /// The key expression.
        key: String,
        /// The query's payload.
        payload: Option<Vec<u8>>,
    },
    /// Drop every subscription and token: the robot moves to another
    /// session (the lobby, or a run), on the same host.
    Reset,
    /// Close every link to the host, however healthy it looks: replies
    /// stopped for [`LINK_LOST_MS`]. The next action on a class opens it
    /// again.
    Reconnect,
    /// Publish `payload` on `key`.
    Publish {
        /// The class.
        class: u8,
        /// The key expression.
        key: String,
        /// The bytes.
        payload: Vec<u8>,
    },
}

/// Where the robot is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Reading the host instance from its state.
    Host,
    /// Waiting for the handshake's reply.
    Handshake,
    /// Waiting for `Attach`'s reply.
    Attach,
    /// Waiting for `DeclareParts`' reply.
    Declare,
    /// Attached, declared and heartbeating.
    Running,
    /// Waiting to start over.
    Retry,
    /// In the lobby, attached and heartbeating, waiting for a run.
    Waiting,
    /// Another process of this device took over its name: it stays safe
    /// until reset.
    Stopped,
}

/// Why the robot last started over.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fault {
    /// No reply from the host (none that was its signed state).
    NoReply,
    /// The handshake failed: the address's key isn't this host's.
    Handshake,
    /// The host refused a request.
    Refused(CtlStatus, String),
    /// No heartbeat echo and no signed state for [`HOST_LOST_MS`].
    HostLost,
    /// A message failed to seal or decode (a bug, or the RNG failed), or
    /// the host's parts don't match the map.
    Internal,
}

/// Why [`Robot::publish`] sent nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublishError {
    /// Not running yet, or the stream's key hasn't arrived.
    NotReady,
    /// No such part.
    UnknownPart,
    /// Not the part's dense layout.
    Invalid,
}

/// A query the robot waits on: over at `deadline`, whatever the transport
/// reports.
#[derive(Debug, Clone, Copy)]
struct Asked {
    id: u32,
    timeout_ms: u32,
    deadline: u64,
}

impl Asked {
    fn new(id: u32, timeout_ms: u32, now_us: u64) -> Self {
        let wait = u64::from(timeout_ms) + u64::from(QUERY_MARGIN_MS);
        Self {
            id,
            timeout_ms,
            deadline: now_us + wait * 1000,
        }
    }
}

/// The `ctl` request in flight.
struct Pending {
    query: Asked,
    request: u64,
    command: Command,
    tries: u8,
}

/// At most a rate of samples, with a burst of two so that jitter around the
/// rate is not decimation. Whole microseconds: a classic ESP32 has no
/// double-precision unit, and this runs on every sample.
#[derive(Debug)]
struct Gate {
    period_us: u64,
    credit_us: u64,
    last_us: Option<u64>,
}

impl Gate {
    const BURST: u64 = 2;

    fn new(hz: f64) -> Self {
        // A cast saturates; the host admits only rates above zero.
        let period_us = ((1e6 / hz) as u64).max(1);
        Self {
            period_us,
            credit_us: period_us.saturating_mul(Self::BURST),
            last_us: None,
        }
    }

    fn admit(&mut self, now_us: u64) -> bool {
        if let Some(last) = self.last_us {
            let earned = self.credit_us.saturating_add(now_us.saturating_sub(last));
            self.credit_us = earned.min(self.period_us.saturating_mul(Self::BURST));
        }
        self.last_us = Some(now_us);
        let admitted = self.credit_us >= self.period_us;
        if admitted {
            self.credit_us -= self.period_us;
        }
        admitted
    }
}

/// What a stream nobody demands (no delivery plan, no `send` ceiling) is
/// published at, in hertz. The spec's default, every sample, suits a robot on
/// a wired host; a board on Wi-Fi carrying a 50 Hz state and its applied
/// stream (100 messages a second) to no reader measurably slows its own link
/// (round trips to the board 60 ms at 50 Hz, 46 ms with one stream at 5 Hz).
/// A demand raises it at once: `every`, or `latest@N` within the ceiling.
pub const UNDEMANDED_HZ: f64 = 10.0;

/// One stream as this robot publishes it.
#[derive(Debug, Default)]
struct Outlet {
    seq: u64,
    /// The demand revision installed (0: none, the default plan).
    revision: u64,
    plan: Option<Plan>,
    /// The plan's rate, or the `send` ceiling; `None` sends every sample.
    gate: Option<Gate>,
}

impl Outlet {
    /// Installs demand revision `revision` (`None`: no demand). Every plan
    /// is one a board can follow, since each only limits what is offered;
    /// a new kind of plan fails to compile here until it is decided.
    fn install(&mut self, revision: u64, plan: Option<Plan>, ceiling: Option<f64>, fixed: bool) {
        let rate = match plan {
            _ if fixed => None,
            // Compared by hand: `f64::min` is a libm call no firmware links.
            Some(Plan::Latest { hz }) => Some(ceiling.filter(|c| *c < hz).unwrap_or(hz)),
            Some(Plan::Every) => ceiling,
            None => Some(ceiling.unwrap_or(UNDEMANDED_HZ)),
        };
        if self.revision == revision && self.plan == plan && self.gate.is_some() == rate.is_some() {
            return;
        }
        self.revision = revision;
        self.plan = plan;
        self.gate = rate.map(Gate::new);
    }

    /// Whether a sample offered at `now_us` goes out: a coalesced one takes
    /// no seq, so a gap at a subscriber is still a loss.
    fn admit(&mut self, now_us: u64) -> bool {
        self.gate.as_mut().is_none_or(|g| g.admit(now_us))
    }

    fn next_seq(&mut self) -> u64 {
        self.seq += 1;
        self.seq
    }
}

/// One stream this robot publishes, worked out once when its parts are
/// declared: its spec takes a dozen allocations, and a tick needs two.
struct Published {
    spec: StreamSpec,
    key: String,
    class: u8,
}

impl Published {
    fn new(keys: &SessionKeys, me: &ParticipantId, spec: StreamSpec) -> Self {
        Self {
            key: keys.data(me, &spec.path),
            class: class_of(spec.priority),
            spec,
        }
    }
}

/// What the runtime sends and listens on for one session, worked out when it
/// enters it: each is a format on a path that runs many times a second.
struct Routes {
    view: String,
    sealed_view: String,
    echo: String,
    heartbeat: String,
}

impl Routes {
    fn new(keys: &SessionKeys, me: &ParticipantId) -> Self {
        Self {
            view: keys.host_view(me),
            sealed_view: keys.participant_view(me),
            echo: keys.heartbeat_echo(me),
            heartbeat: keys.heartbeat(me),
        }
    }
}

/// One command group's edge.
struct Edge {
    core: EdgeCore,
    /// The latch count it last saw: a latch cleared since still fences it.
    latches_seen: u64,
    /// Chunks it refused, and the last one's reason and when.
    refused: u64,
    last_refusal: Option<(Rejection, i64)>,
}

impl Edge {
    fn refused(&mut self, reason: Rejection, now: i64) {
        self.refused += 1;
        self.last_refusal = Some((reason, now));
    }
}

/// One control tick of a part's command group: session-core's
/// [`EdgeTick`], with the part it concerns as an index into the map.
#[derive(Debug, Clone)]
pub struct Tick {
    /// The edge's tick.
    pub tick: EdgeTick,
    /// The part the tick concerns (the active or last-driven member).
    pub part: Option<usize>,
}

/// The robot.
pub struct Robot {
    config: Config,
    /// The session it is in now: the configured one, the lobby, or a run.
    tenant: TenantId,
    session: SessionId,
    keys: SessionKeys,
    routes: Routes,
    /// Between runs: what its next wait reports.
    lobby: Option<Lobby>,
    /// The run it joined, while in one.
    run: Option<Join>,
    /// The lobby wait in flight: its query, request id and command.
    wait: Option<(Asked, u64, Command)>,
    /// Where it connects: one of the homes, or a run's address.
    target: (String, u16),
    /// The home it tries now, an index into `config.homes`.
    home_at: usize,
    /// How many of `config.homes` it was given: after them, at most the one
    /// the platform found last.
    stored: usize,
    /// The run's addresses, while joining one hosted elsewhere, and the one
    /// tried now.
    run_targets: Vec<(String, u16)>,
    run_at: usize,
    /// The handshake's keys now: the home's, or a run's host key.
    credential: Credential,
    /// Every home address failed in a row: the platform may look for it
    /// (mDNS).
    home_unreachable: bool,
    /// The pause before the next round of homes, doubling to 30 s.
    backoff_ms: u32,
    /// The pause after a refusal (the home won't let this device wait or
    /// attach under this name), doubling to 30 s until it joins a run.
    refused_ms: u32,
    /// Why it last left a run, and the view that said so (revision,
    /// generation listed for it, its own).
    left: Option<(Leave, u64, Option<u32>, u32)>,
    /// The map's part paths, in its order. `DeclareParts.seal` names
    /// them all: this runtime seals any part.
    paths: Vec<String>,
    phase: Phase,
    out: Vec<Action>,
    started: bool,
    next_query: u32,
    /// The join step's query (host state, handshake).
    waiting: Option<Asked>,
    initiator: Option<IkInitiator>,
    link: Option<Link>,
    /// The link is new, and a transport that verifies broadcasts on another
    /// task hasn't taken its checker.
    checker_due: bool,
    /// This link took a sealed view: signed ones are ignored from then on.
    sealed_seen: bool,
    /// Sealed views taken, signed views taken, and views refused (not
    /// opened, another generation's, or not newer than the one held), over
    /// every attach.
    sealed_views: u64,
    signed_views: u64,
    view_refusals: u64,
    pending: Option<Pending>,
    queue: VecDeque<Command>,
    /// An unsealed refusal of the `ctl` request in flight: a hint, kept in
    /// case no sealed answer comes.
    hint: Option<Fault>,
    /// What the host says about this robot: its own signed view, or the
    /// part of a ctl reply's state it would hold (the same).
    view: Option<DeviceView>,
    /// Signed views opened from `host/view/{id}`, over every attach.
    views: u64,
    /// Local time the view was observed at.
    observed_ns: i64,
    /// The host's parts of this robot, in the map's order.
    parts: Vec<PartSpec>,
    /// Each part's sample stream, in the same order (`None`: it has none it
    /// can publish), its applied stream, and its command key.
    samples: Vec<Option<Published>>,
    applied_streams: Vec<Option<Published>>,
    command_keys: Vec<String>,
    edges: Vec<Edge>,
    edge_of: Vec<Option<usize>>,
    /// Command subscriptions made (they outlive a restart).
    subscribed: Vec<bool>,
    /// Each stream's seq and delivery plan.
    seqs: BTreeMap<ChannelId, Outlet>,
    fault: Option<Fault>,
    retry_at: u64,
    next_heartbeat: u64,
    last_heard: u64,
    echoes: u64,
    heartbeat_seq: u64,
    previous: Option<TimingSample>,
    generation: u32,
    last_get_keys: Option<u64>,
    /// When each part was last ticked and its control period, in
    /// microseconds: a board's loop can sleep until the next tick is due.
    ticked: Vec<Option<u64>>,
    tick_periods: Vec<u64>,
    accepted: u64,
    applied: u64,
    /// This robot's e-stop latch: its own stop (`ESTOP_LOCAL`)
    /// or a member's signal (`ESTOP_SIGNAL`). It outlives a session, and the
    /// next attach carries it.
    estop: Option<Reason>,
    latches: u64,
    /// The code it last latched with, which a cleared latch still fences
    /// an edge with.
    last_code: ReasonCode,
}

impl core::fmt::Debug for Robot {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Robot")
            .field("name", &self.config.name)
            .field("phase", &self.phase)
            .field("fault", &self.fault)
            .finish_non_exhaustive()
    }
}

fn ns(now_us: u64) -> i64 {
    i64::try_from(now_us.saturating_mul(1000)).unwrap_or(i64::MAX)
}

/// The class a part's traffic rides.
fn class_of(priority: ChannelPriority) -> u8 {
    match priority {
        ChannelPriority::Control => 0,
        ChannelPriority::Realtime => 1,
        ChannelPriority::Data => 2,
        ChannelPriority::Bulk => 3,
    }
}

impl Robot {
    /// A robot that starts joining on the first [`Self::poll`], built in
    /// its allocation: it is too large for a board's stack.
    ///
    /// # Errors
    /// [`ConfigError`]: a map the host would refuse, checked here first.
    pub fn new(config: Config) -> Result<Box<Self>, ConfigError> {
        if config.heartbeat_ms == 0 {
            return Err(ConfigError::Heartbeat);
        }
        // The host normalises the map at `DeclareParts`: the board needs
        // only its paths, in authored order.
        let paths = crate::map::paths(&config.robot_json)
            .ok_or_else(|| ConfigError::Map(String::from("not a JSON object of parts")))?;
        let (tenant, session) = if config.lobby {
            (
                TenantId::new(crate::address::TENANT)
                    .map_err(|_| ConfigError::Map(String::from("tenant")))?,
                SessionId::named(LOBBY_SESSION),
            )
        } else {
            (config.tenant.clone(), config.session)
        };
        let keys = SessionKeys::new(&tenant, session);
        let routes = Routes::new(&keys, &config.name);
        let lobby = config.lobby.then(|| Lobby::new(config.name.clone()));
        let target = config.homes.first().cloned().ok_or(ConfigError::NoHome)?;
        let credential = config.credential.clone();
        let stored = config.homes.len();
        let robot = Box::<Self>::new_uninit();
        Ok(Box::write(
            robot,
            Self {
                subscribed: alloc::vec![false; paths.len()],
                edge_of: alloc::vec![None; paths.len()],
                ticked: alloc::vec![None; paths.len()],
                paths,
                config,
                tenant,
                session,
                keys,
                routes,
                lobby,
                run: None,
                wait: None,
                target,
                home_at: 0,
                stored,
                run_targets: Vec::new(),
                run_at: 0,
                credential,
                home_unreachable: false,
                backoff_ms: RETRY_MS,
                refused_ms: RETRY_MS,
                left: None,
                phase: Phase::Host,
                out: Vec::new(),
                started: false,
                next_query: 0,
                waiting: None,
                initiator: None,
                link: None,
                checker_due: false,
                sealed_seen: false,
                sealed_views: 0,
                signed_views: 0,
                view_refusals: 0,
                pending: None,
                queue: VecDeque::new(),
                hint: None,
                view: None,
                views: 0,
                observed_ns: 0,
                parts: Vec::new(),
                samples: Vec::new(),
                applied_streams: Vec::new(),
                command_keys: Vec::new(),
                edges: Vec::new(),
                seqs: BTreeMap::new(),
                fault: None,
                retry_at: 0,
                next_heartbeat: 0,
                last_heard: 0,
                echoes: 0,
                heartbeat_seq: 0,
                previous: None,
                generation: 0,
                last_get_keys: None,
                tick_periods: Vec::new(),
                accepted: 0,
                applied: 0,
                estop: None,
                latches: 0,
                last_code: ReasonCode::EstopLocal,
            },
        ))
    }

    /// Where it is.
    pub fn phase(&self) -> Phase {
        self.phase
    }

    /// Why it last started over, if it has.
    pub fn fault(&self) -> Option<&Fault> {
        self.fault.as_ref()
    }

    /// Where it connects now: a host and its first port (class `i` listens
    /// on the port plus `i`).
    pub fn host(&self) -> (&str, u16) {
        (&self.target.0, self.target.1)
    }

    /// The home's host id, empty when not known.
    pub fn home_id(&self) -> &str {
        &self.config.home_id
    }

    /// Whether every home address failed in a row: time to look for it.
    pub fn home_unreachable(&self) -> bool {
        self.home_unreachable
    }

    /// Another address of the home, found by the platform (mDNS): tried
    /// next, at once. It replaces the one found before.
    pub fn add_home(&mut self, host: &str, port: u16, now_us: u64) {
        let home = (String::from(host), port);
        let at = match self.config.homes.iter().position(|h| *h == home) {
            Some(at) => at,
            None => {
                self.config.homes.truncate(self.stored);
                self.config.homes.push(home);
                self.stored
            }
        };
        if self.home_unreachable && self.phase == Phase::Retry {
            self.home_unreachable = false;
            self.home_at = at;
            self.retry_at = now_us;
            self.retarget_home();
        }
    }

    /// Why it last left a run: closing, lost or superseded, with the view's
    /// revision, the generation the view listed for it, and its own.
    pub fn left(&self) -> Option<&(Leave, u64, Option<u32>, u32)> {
        self.left.as_ref()
    }

    /// Whether it is in a run it joined from the lobby.
    pub fn in_run(&self) -> bool {
        self.run.is_some()
    }

    /// Signed device views received, over every attach.
    pub fn views(&self) -> u64 {
        self.views
    }

    /// The newest verified device view.
    pub fn view(&self) -> Option<&DeviceView> {
        self.view.as_ref()
    }

    /// Heartbeat echoes received, over every attach.
    pub fn echoes(&self) -> u64 {
        self.echoes
    }

    /// This attach's generation (0 before the first).
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// The map's part paths, in its order: the indices every call takes.
    pub fn paths(&self) -> &[String] {
        &self.paths
    }

    /// The index of `path` in the map.
    pub fn part(&self, path: &str) -> Option<usize> {
        self.paths.iter().position(|p| p == path)
    }

    /// The index in the map of the part with `channel`, once running.
    pub fn part_of(&self, channel: ChannelId) -> Option<usize> {
        self.parts.iter().position(|p| p.channel == channel)
    }

    /// Latches this robot's e-stop for its own stop (`ESTOP_LOCAL`): an
    /// e-stop pin. Every edge stops at once, with or without a host, and
    /// the host is told. Only [`Self::clear_estop`] clears it.
    pub fn estop(&mut self, reason: &str, now_us: u64) {
        self.latch(Reason {
            code: ReasonCode::EstopLocal,
            by: self.config.name.to_string(),
            detail: reason.to_string(),
            at_ns: ns(now_us),
            ..Reason::default()
        });
    }

    /// Clears the e-stop latch: the only way to, since no wire path clears
    /// it. The host is told after the `Estop` for every latch before, so it
    /// ends in ESTOP, unlatched; leaving that still needs disarm and arm.
    pub fn clear_estop(&mut self) {
        if self.estop.take().is_some() {
            self.queue.push_back(Command::Clear);
        }
    }

    /// Why the e-stop is latched, while it is.
    pub fn estop_reason(&self) -> Option<&Reason> {
        self.estop.as_ref()
    }

    /// Chunks the edges accepted, over every attach.
    pub fn chunks_accepted(&self) -> u64 {
        self.accepted
    }

    /// Chunks the edges refused, this attach.
    pub fn chunks_refused(&self) -> u64 {
        self.edges.iter().map(|e| e.refused).sum()
    }

    /// Why the most recent chunk refused this attach was.
    pub fn last_refusal(&self) -> Option<Rejection> {
        self.edges
            .iter()
            .filter_map(|e| e.last_refusal)
            .max_by_key(|(_, at)| *at)
            .map(|(reason, _)| reason)
    }

    /// Chunks refused this attach for `reason`.
    pub fn refusals(&self, reason: Rejection) -> u64 {
        self.edges.iter().map(|e| e.core.rejected(reason)).sum()
    }

    /// The next moment, on this board's clock, the robot has something to
    /// do without new input: actions or a `ctl` request waiting, a retry, a
    /// heartbeat, the host's silence running out, a query's deadline, or a
    /// part's next control tick (from when it was last ticked, by its control
    /// period). A board's loop sleeps until then, or until the transport has
    /// input, instead of spinning. `None` when nothing is scheduled.
    pub fn next_due(&self) -> Option<u64> {
        if !self.out.is_empty() || !self.started {
            return Some(0);
        }
        if self.phase == Phase::Stopped {
            return None;
        }
        if self.pending.is_none() && self.link.is_some() && !self.queue.is_empty() {
            return Some(0);
        }
        let mut due: Option<u64> = None;
        let mut at = |t: u64| due = Some(due.map_or(t, |d| d.min(t)));
        match self.phase {
            Phase::Retry => at(self.retry_at),
            Phase::Running | Phase::Waiting => {
                at(self.next_heartbeat);
                at(self.last_heard + u64::from(HOST_LOST_MS) * 1000);
            }
            _ => {}
        }
        for asked in self
            .waiting
            .iter()
            .chain(self.pending.as_ref().map(|p| &p.query))
            .chain(self.wait.as_ref().map(|w| &w.0))
        {
            at(asked.deadline);
        }
        for (i, last) in self.ticked.iter().enumerate() {
            if let (Some(last), Some(period), Some(Some(_))) =
                (last, self.tick_periods.get(i), self.edge_of.get(i))
            {
                at(last.saturating_add(*period));
            }
        }
        due
    }

    /// Runs the timers and returns what the transport should do next, in
    /// order.
    pub fn poll(&mut self, now_us: u64) -> Vec<Action> {
        if self.phase == Phase::Stopped {
            return core::mem::take(&mut self.out);
        }
        self.expire_queries(now_us);
        if !self.started {
            self.started = true;
            let me = &self.config.name;
            self.out.extend([
                Action::Subscribe {
                    class: CLASS_CONTROL,
                    sub: SUB_VIEW,
                    key: self.keys.host_view(me),
                },
                Action::Subscribe {
                    class: CLASS_CONTROL,
                    sub: SUB_VIEW_SEALED,
                    key: self.routes.sealed_view.clone(),
                },
                Action::Subscribe {
                    class: CLASS_CONTROL,
                    sub: SUB_TICK,
                    key: self.keys.host_tick(),
                },
                Action::Subscribe {
                    class: CLASS_CONTROL,
                    sub: SUB_ECHO,
                    key: self.keys.heartbeat_echo(me),
                },
                Action::Subscribe {
                    class: CLASS_CONTROL,
                    sub: SUB_KEYS,
                    key: self.keys.participant_keys(me),
                },
                // A robot latches every member's e-stop signal at once,
                // with or without a host.
                Action::Subscribe {
                    class: CLASS_CONTROL,
                    sub: SUB_ESTOP,
                    key: self.keys.estops(),
                },
                // A multi-session host starts the session when it sees
                // this token.
                Action::Token {
                    class: CLASS_CONTROL,
                    key: self.keys.participant(me),
                },
            ]);
            // After a failure the retry timer asks, after its pause.
            if self.phase != Phase::Retry {
                self.ask_host(now_us);
            }
        }
        match self.phase {
            Phase::Retry if now_us >= self.retry_at => self.ask_host(now_us),
            // Nothing from the host for HOST_LOST_MS: it is gone, or
            // restarted under a new instance. Join again.
            Phase::Running | Phase::Waiting
                if now_us >= self.last_heard + u64::from(HOST_LOST_MS) * 1000 =>
            {
                self.fail(Fault::HostLost, now_us);
            }
            Phase::Running | Phase::Waiting if now_us >= self.next_heartbeat => {
                self.heartbeat(now_us);
            }
            _ => {}
        }
        self.next_ctl(now_us);
        core::mem::take(&mut self.out)
    }

    /// A reply to query `id`. `now_us` and `rng` drive the next step.
    pub fn reply(&mut self, id: u32, payload: &[u8], now_us: u64, rng: &mut dyn Rng) {
        if self.pending.as_ref().is_some_and(|p| p.query.id == id) {
            return self.ctl_reply(payload, now_us);
        }
        if self.wait.as_ref().is_some_and(|w| w.0.id == id) {
            return self.wait_reply(payload, now_us);
        }
        if self.waiting.is_none_or(|w| w.id != id) {
            return;
        }
        match self.phase {
            Phase::Host => {
                // Unverified: a wrong instance only fails the handshake. A
                // reply that isn't a signed state is some other node's: the
                // host's may follow, and the query's end says if none did.
                let Ok(peeked) = signed::peek(payload) else {
                    return;
                };
                self.last_heard = now_us;
                self.start(peeked.header.host_instance, rng, now_us);
            }
            Phase::Handshake => self.handshake_reply(payload, now_us),
            _ => {}
        }
    }

    /// The host's handshake answer: on to the attach, or a failure. Out of
    /// line, like [`Self::ctl_reply`]: inlined, each step's large temporaries
    /// add up in one frame of [`Self::reply`].
    #[inline(never)]
    fn handshake_reply(&mut self, payload: &[u8], now_us: u64) {
        let msg2 = match codec::decode_handshake_reply(payload) {
            Ok(Ok(msg2)) => msg2,
            Ok(Err(refusal)) => {
                self.last_heard = now_us;
                self.waiting = None;
                return self.fail(Fault::Refused(refusal.status, refusal.message), now_us);
            }
            Err(_) => return,
        };
        self.last_heard = now_us;
        self.waiting = None;
        let Some(initiator) = self.initiator.take() else {
            return self.fail(Fault::Internal, now_us);
        };
        let Ok(attached) = auth::finish(initiator, &msg2) else {
            return self.fail(Fault::Handshake, now_us);
        };
        self.generation = attached.accept.generation;
        self.link = Some(Link::new(self.config.name.clone(), attached));
        self.checker_due = true;
        self.sealed_seen = false;
        self.phase = Phase::Attach;
        // The host answers nothing before the attach, which carries
        // the latch: stop reports queued until now are in it.
        self.queue
            .retain(|c| !matches!(c, Command::Estop { .. } | Command::Clear));
        self.queue.push_front(Command::Attach {
            name: self.config.name.as_str().to_string(),
            roles: alloc::vec![Role::Embodiment],
            priority: PriorityClass::Scripted,
            protocol_minor: PROTOCOL_MINOR,
            // The device view instead of the full state.
            supports: alloc::vec![String::from(VIEW_DEVICE), String::from(VIEW_DEVICE_SEALED)],
            requires: Vec::new(),
            workflow_edition: String::new(),
            workflow_editions: Vec::new(),
            // A latch outlives the session: the host learns it at
            // once.
            estop: self.estop.clone(),
        });
        self.next_ctl(now_us);
    }

    /// Query `id` is over: every reply has arrived, or it timed out.
    pub fn query_done(&mut self, id: u32, now_us: u64) {
        if self.pending.as_ref().is_some_and(|p| p.query.id == id) {
            return self.ctl_unanswered(now_us);
        }
        if self.wait.as_ref().is_some_and(|w| w.0.id == id) {
            // No order within WAIT_RETRY: the same request again, which
            // joins the pending wait (or fetches its cached answer).
            return self.send_wait(now_us);
        }
        if self.waiting.is_none_or(|w| w.id != id) {
            return;
        }
        self.waiting = None;
        if matches!(self.phase, Phase::Host | Phase::Handshake) {
            self.fail(Fault::NoReply, now_us);
        }
    }

    /// Query `id` left for the network at `now_us`, later than [`Self::poll`]
    /// handed it out: its timeout runs from here.
    pub fn query_sent(&mut self, id: u32, now_us: u64) {
        let asked = self
            .waiting
            .iter_mut()
            .chain(self.pending.as_mut().map(|p| &mut p.query))
            .chain(self.wait.as_mut().map(|w| &mut w.0))
            .find(|a| a.id == id);
        if let Some(asked) = asked {
            *asked = Asked::new(id, asked.timeout_ms, now_us);
        }
    }

    /// Ends every query past its deadline, as if the transport had reported
    /// it: a reply or end that arrives for it later names no query in
    /// flight.
    fn expire_queries(&mut self, now_us: u64) {
        let due = |a: Asked| (now_us >= a.deadline).then_some(a.id);
        let expired = [
            self.waiting.and_then(due),
            self.pending.as_ref().and_then(|p| due(p.query)),
            self.wait.as_ref().and_then(|w| due(w.0)),
        ];
        for id in expired.into_iter().flatten() {
            self.query_done(id, now_us);
        }
    }

    /// A sample on subscription `sub`, received on `key`.
    pub fn sample(&mut self, sub: u8, key: &str, payload: &mut [u8], now_us: u64) {
        let now = ns(now_us);
        if self.link.is_none() {
            return;
        }
        // Each kind out of line: inlined, their temporaries add up in one
        // frame underneath the signature check, a board's deepest call.
        match sub {
            SUB_VIEW => self.sample_view(payload, now_us),
            SUB_VIEW_SEALED => self.sample_sealed_view(payload, now_us),
            SUB_TICK => self.sample_tick(payload),
            SUB_ECHO => self.sample_echo(payload, now_us),
            SUB_ESTOP => self.sample_estop(key, payload, now),
            SUB_KEYS => self.sample_keys(payload, now),
            _ => {
                if let Some(index) = sub.checked_sub(SUB_COMMANDS) {
                    self.chunk(usize::from(index), payload.to_vec(), now);
                }
            }
        }
    }

    #[inline(never)]
    fn sample_view(&mut self, payload: &mut [u8], now_us: u64) {
        let now = ns(now_us);
        if self.sealed_seen {
            return;
        }
        let Some(link) = self.link.as_mut() else {
            return;
        };
        if let Ok(view) = link.open_view(&self.routes.view, payload) {
            link.observe(&view, now);
            self.signed_views += 1;
            self.took_view(view, now_us);
        }
    }

    /// This robot's view sealed under its `k_down`: opened at a symmetric
    /// cipher's cost, and from then on this link ignores signed views.
    #[inline(never)]
    fn sample_sealed_view(&mut self, payload: &mut [u8], now_us: u64) {
        let now = ns(now_us);
        let Some(link) = self.link.as_mut() else {
            return;
        };
        let Ok(view) = link.open_sealed_view(&self.routes.sealed_view, payload) else {
            self.view_refusals += 1;
            return;
        };
        if self
            .view
            .as_ref()
            .is_some_and(|held| (held.revision, held.host_ns) >= (view.revision, view.host_ns))
        {
            self.view_refusals += 1;
            return;
        }
        link.observe(&view, now);
        self.sealed_seen = true;
        self.sealed_views += 1;
        self.took_view(view, now_us);
    }

    /// Whether a transport should still check signed views for this robot:
    /// not once this link has taken a sealed one, which saves a signature
    /// check per view.
    pub fn wants_signed_views(&self) -> bool {
        !self.sealed_seen
    }

    /// Sealed views taken, signed views taken, and views refused (not
    /// opened, another generation's, or not newer than the one held), over
    /// every attach.
    pub fn view_counts(&self) -> (u64, u64, u64) {
        (self.sealed_views, self.signed_views, self.view_refusals)
    }

    /// A verified signed view, from either path.
    fn took_view(&mut self, view: DeviceView, now_us: u64) {
        self.views += 1;
        self.last_heard = now_us;
        self.observe(view, ns(now_us));
        self.check_leave(now_us);
    }

    /// A verifier for this link's signed broadcasts, once per link: a
    /// transport with another task to spare checks them there, off the
    /// control loop, and hands them to [`Self::sample_checked`]. One that
    /// never takes it passes them to [`Self::sample`], which checks them
    /// itself.
    pub fn take_checker(&mut self) -> Option<SignedChecker> {
        if !core::mem::take(&mut self.checker_due) {
            return None;
        }
        self.link.as_ref().map(Link::checker)
    }

    /// A signed view on [`SUB_VIEW`] that a [`Self::take_checker`] checker
    /// verified. Only its signature check was skipped: it is still held to
    /// this link's key, instance and floor, and to this robot's key
    /// expression.
    pub fn sample_checked(&mut self, sub: u8, checked: &Checked, now_us: u64) {
        if sub != SUB_VIEW || self.sealed_seen {
            return;
        }
        let Some(link) = self.link.as_mut() else {
            return;
        };
        if let Ok(view) = link.take_view(&self.routes.view, checked) {
            link.observe(&view, ns(now_us));
            self.signed_views += 1;
            self.took_view(view, now_us);
        }
    }

    #[inline(never)]
    fn sample_tick(&mut self, payload: &mut [u8]) {
        let key = self.keys.host_tick();
        if let Some(link) = self.link.as_mut() {
            // Verified to keep the host's sequence floor; lockstep isn't
            // run on a board yet.
            let _ = link.tick(&key, payload);
        }
    }

    #[inline(never)]
    fn sample_echo(&mut self, payload: &mut [u8], now_us: u64) {
        let Some(link) = self.link.as_mut() else {
            return;
        };
        if let Ok(echo) = link.echo(&self.routes.echo, payload) {
            self.previous = Some(echo.complete(ns(now_us)));
            self.last_heard = now_us;
            self.echoes += 1;
        }
    }

    #[inline(never)]
    fn sample_estop(&mut self, key: &str, payload: &[u8], now: i64) {
        let Some(link) = self.link.as_mut() else {
            return;
        };
        if let Ok((by, signal)) = link.open_estop_signal(key, payload.to_vec()) {
            self.latch(Reason {
                code: ReasonCode::EstopSignal,
                by: by.to_string(),
                detail: signal.reason,
                at_ns: now,
                ..Reason::default()
            });
        }
    }

    #[inline(never)]
    fn sample_keys(&mut self, payload: &mut [u8], now: i64) {
        let key = self.keys.participant_keys(&self.config.name);
        let Some(link) = self.link.as_mut() else {
            return;
        };
        if let Ok(ack) = link.keys(&key, payload, now)
            && !ack.is_empty()
        {
            self.enqueue(Command::KeysAck(ack));
        }
    }

    /// One control tick of part `part`'s command group at `now_us`: what
    /// to dispatch, hold, or the safe behaviour to run. Publishes the
    /// group's applied samples, and tells the host when the edge went safe
    /// on its own account. `None` for a part that takes no commands, or
    /// before the robot runs.
    pub fn tick(&mut self, part: usize, now_us: u64) -> Option<Tick> {
        if let Some(at) = self.ticked.get_mut(part) {
            *at = Some(now_us);
        }
        let now = ns(now_us);
        let index = (*self.edge_of.get(part)?)?;
        let latch = self.latch_now();
        let edge = self.edges.get_mut(index)?;
        apply_latch(edge, latch);
        let tick = edge.core.tick(now, self.view.as_ref(), self.observed_ns);
        let epoch = self.view.as_ref().map_or(Epoch(0), |s| s.epoch);
        for sample in &tick.applied {
            let Some(stream) = self
                .parts
                .iter()
                .position(|p| p.channel == sample.part)
                .and_then(|i| self.applied_streams.get(i))
                .and_then(Option::as_ref)
            else {
                continue;
            };
            let outlet = self.seqs.entry(stream.spec.channel).or_default();
            if !outlet.admit(now_us) {
                continue;
            }
            let sealed = Sample {
                channel: stream.spec.channel,
                epoch,
                seq: outlet.next_seq(),
                capture_ns: now,
                publish_ns: now,
                tick: 0,
                payload: sample.value.clone(),
                applied: Some(sample.applied),
            };
            if let Some(link) = self.link.as_mut()
                && let Ok(payload) = link.seal_sample(&stream.key, &sealed, true)
            {
                self.out.push(Action::Publish {
                    class: stream.class,
                    key: stream.key.clone(),
                    payload,
                });
            }
        }
        if let Some(action) = &tick.action {
            self.applied = action.chunk.seq;
        }
        if let Some((_, cause)) = tick.report {
            let channel = tick.channel.map(|c| c.to_string()).unwrap_or_default();
            self.enqueue(Command::EnterSafe {
                reason: alloc::format!("{channel}: {cause}"),
                code: cause.code(),
            });
        }
        let part = tick.channel.and_then(|c| self.part_of(c));
        Some(Tick { tick, part })
    }

    /// The robot finished the handover of `part`'s group: it may dispatch
    /// again.
    pub fn handover_done(&mut self, part: usize) {
        if let Some(Some(index)) = self.edge_of.get(part)
            && let Some(edge) = self.edges.get_mut(*index)
        {
            edge.core.handover_done();
        }
    }

    /// Publishes a sample of part `part`: `value` in its dense layout,
    /// captured at `capture_us` (this board's clock). Under a delivery plan
    /// that coalesces it, nothing is sent and it still succeeds.
    ///
    /// # Errors
    /// [`PublishError`]: nothing was sent.
    pub fn publish(
        &mut self,
        part: usize,
        capture_us: u64,
        value: &[u8],
        now_us: u64,
    ) -> Result<(), PublishError> {
        if part >= self.paths.len() {
            return Err(PublishError::UnknownPart);
        }
        let (Some(stream), Some(link)) = (self.samples.get(part), self.link.as_mut()) else {
            return Err(PublishError::NotReady);
        };
        let stream = stream.as_ref().ok_or(PublishError::Invalid)?;
        stream
            .spec
            .check_payload(value)
            .map_err(|_| PublishError::Invalid)?;
        let outlet = self.seqs.entry(stream.spec.channel).or_default();
        if !outlet.admit(now_us) {
            return Ok(());
        }
        let sample = Sample {
            channel: stream.spec.channel,
            epoch: self.view.as_ref().map_or(Epoch(0), |s| s.epoch),
            seq: outlet.next_seq(),
            capture_ns: ns(capture_us),
            publish_ns: ns(now_us),
            tick: 0,
            payload: Bytes::copy_from_slice(value),
            applied: None,
        };
        match link.seal_sample(&stream.key, &sample, false) {
            Ok(payload) => {
                self.out.push(Action::Publish {
                    class: stream.class,
                    key: stream.key.clone(),
                    payload,
                });
                Ok(())
            }
            Err(err) => {
                self.missing_key(&err, now_us);
                Err(PublishError::NotReady)
            }
        }
    }

    #[inline(never)]
    fn chunk(&mut self, index: usize, msg: Vec<u8>, now: i64) {
        let (Some(key), Some(link)) = (self.command_keys.get(index), self.link.as_mut()) else {
            return;
        };
        let chunk = match link.open_chunk(key, msg) {
            Ok(chunk) => chunk,
            Err(err) => {
                if let Some(Some(edge)) = self
                    .edge_of
                    .get(index)
                    .map(|e| e.and_then(|i| self.edges.get_mut(i)))
                {
                    edge.core.count_rejection(Rejection::Unauthenticated);
                    edge.refused(Rejection::Unauthenticated, now);
                }
                let now_us = u64::try_from(now / 1000).unwrap_or(0);
                return self.missing_key(&err, now_us);
            }
        };
        let latch = self.latch_now();
        let Some(Some(edge)) = self
            .edge_of
            .get(index)
            .map(|e| e.and_then(|i| self.edges.get_mut(i)))
        else {
            return;
        };
        apply_latch(edge, latch);
        match edge
            .core
            .accept(chunk, now, self.view.as_ref(), self.observed_ns)
        {
            Ok(_) => self.accepted += 1,
            Err(reason) => edge.refused(reason, now),
        }
    }

    /// Queues `command` for `ctl`, bounded: the same request already
    /// waiting (or in flight) is not queued twice, a newer `EnterSafe`, key
    /// request or stream's demand acknowledgement replaces the one waiting,
    /// and past [`CTL_QUEUE_MAX`] the oldest
    /// request that is not a stop report is dropped.
    fn enqueue(&mut self, command: Command) {
        if self.pending.as_ref().is_some_and(|p| p.command == command)
            || self.queue.contains(&command)
        {
            return;
        }
        let same_kind = |c: &Command| match (c, &command) {
            (Command::EnterSafe { .. }, Command::EnterSafe { .. })
            | (Command::GetKeys, Command::GetKeys) => true,
            (Command::DemandAck { channel: a, .. }, Command::DemandAck { channel: b, .. }) => {
                a == b
            }
            _ => false,
        };
        self.queue.retain(|c| !same_kind(c));
        while self.queue.len() >= CTL_QUEUE_MAX {
            match self.queue.iter().position(|c| !is_stop(c)) {
                Some(at) => {
                    self.queue.remove(at);
                }
                None => break,
            }
        }
        self.queue.push_back(command);
    }

    /// A message under a key this robot doesn't hold: ask for every key of
    /// its generation, at most every [`GET_KEYS_MS`].
    fn missing_key(&mut self, err: &LinkError, now_us: u64) {
        if !matches!(err, LinkError::Key(KeyError::Unknown(_))) {
            return;
        }
        if self
            .last_get_keys
            .is_some_and(|at| now_us < at + GET_KEYS_MS * 1000)
        {
            return;
        }
        self.last_get_keys = Some(now_us);
        self.enqueue(Command::GetKeys);
    }

    /// Latches the e-stop for `reason`, and queues its report to the host
    /// (in order: before any `Clear` queued after it). Already latched, it
    /// keeps the first reason and queues nothing: a held button or a stream
    /// of e-stop signals never floods the control queue.
    /// Chunks its edges refused, by reason (nonzero only): a compact report
    /// riding each heartbeat.
    fn rejected(&self) -> Vec<(Rejection, u64)> {
        Rejection::ALL
            .iter()
            .map(|r| {
                let n = self.edges.iter().map(|e| e.core.rejected(*r)).sum::<u64>();
                (*r, n)
            })
            .filter(|(_, n)| *n > 0)
            .collect()
    }

    fn latch(&mut self, reason: Reason) {
        if self.estop.is_some() {
            return;
        }
        let origin = (reason.code == ReasonCode::EstopSignal)
            .then(|| ParticipantId::new(&reason.by).ok())
            .flatten();
        self.queue.push_back(Command::Estop {
            reason: reason.detail.clone(),
            code: reason.code,
            origin,
            at_ns: reason.at_ns,
        });
        self.last_code = reason.code;
        self.estop = Some(reason);
        self.latches += 1;
    }

    /// The latch as the edges take it.
    fn latch_now(&self) -> Latch {
        Latch {
            latches: self.latches,
            code: self.last_code,
            latched: self.estop.is_some(),
        }
    }

    /// Takes `view` if it is newer than the one held (a later revision, or
    /// a later republish of the same one): views and ctl replies arrive in
    /// any order.
    fn observe(&mut self, view: DeviceView, now: i64) {
        let order = |v: &DeviceView| (v.revision, v.host_ns);
        if self
            .view
            .as_ref()
            .is_some_and(|held| order(held) > order(&view))
        {
            return;
        }
        self.view = Some(view);
        self.observed_ns = now;
        self.sync_demands();
    }

    /// Installs the view's delivery plan on every stream this robot
    /// publishes, and acknowledges any revision the host hasn't heard about
    /// yet. Every view asks again, so an acknowledgement lost with its query
    /// is sent anew.
    fn sync_demands(&mut self) {
        let Some(view) = self.view.as_ref() else {
            return;
        };
        let wall = view.clock_mode == ClockMode::Wall;
        let mut acks = Vec::new();
        for part in &self.parts {
            let streams = [
                (Some(part.channel), part.send_policy().hz, part.event),
                (part.applied_channel, None, false),
            ];
            for (channel, ceiling, event) in streams {
                let Some(channel) = channel else {
                    continue;
                };
                let demand = view.demands.iter().find(|d| d.channel == channel);
                let fixed = event || !wall;
                self.seqs.entry(channel).or_default().install(
                    demand.map_or(0, |d| d.revision),
                    demand.map(|d| d.plan),
                    ceiling.filter(|_| !fixed),
                    fixed,
                );
                let Some(demand) = demand.filter(|d| d.acked < d.revision) else {
                    continue;
                };
                let ack = Command::DemandAck {
                    channel,
                    revision: demand.revision,
                };
                acks.push(ack);
            }
        }
        for ack in acks {
            self.enqueue(ack);
        }
    }

    fn query(&mut self, class: u8, key: String, payload: Option<Vec<u8>>) -> u32 {
        self.query_for(class, key, payload, QUERY_MS)
    }

    fn query_for(
        &mut self,
        class: u8,
        key: String,
        payload: Option<Vec<u8>>,
        timeout_ms: u32,
    ) -> u32 {
        self.next_query = self.next_query.wrapping_add(1);
        let id = self.next_query;
        self.out.push(Action::Query {
            class,
            id,
            timeout_ms,
            key,
            payload,
        });
        id
    }

    /// Queries `key` and starts its deadline.
    fn ask(&mut self, key: String, payload: Option<Vec<u8>>, timeout: u32, now_us: u64) -> Asked {
        let id = self.query_for(CLASS_CONTROL, key, payload, timeout);
        Asked::new(id, timeout, now_us)
    }

    fn ask_host(&mut self, now_us: u64) {
        self.phase = Phase::Host;
        let asked = self.ask(self.keys.host_state(), None, QUERY_MS, now_us);
        self.waiting = Some(asked);
    }

    #[inline(never)]
    fn start(&mut self, host_instance: u64, rng: &mut dyn Rng, now_us: u64) {
        let session = self.session.to_string();
        let started = self.credential.start(
            rng,
            self.tenant.as_str(),
            &session,
            host_instance,
            &self.config.name,
        );
        let Ok((initiator, hello)) = started else {
            return self.fail(Fault::Internal, now_us);
        };
        self.initiator = Some(initiator);
        self.phase = Phase::Handshake;
        let payload = codec::encode_handshake_request(&hello);
        let asked = self.ask(self.keys.handshake(), Some(payload), QUERY_MS, now_us);
        self.waiting = Some(asked);
    }

    fn epoch(&self) -> Option<Epoch> {
        self.view.as_ref().map(|s| s.epoch)
    }

    /// Sends the next queued `ctl` request when none is in flight.
    fn next_ctl(&mut self, now_us: u64) {
        while self.pending.is_none() && self.link.is_some() {
            let Some(command) = self.queue.pop_front() else {
                return;
            };
            // A stop the host already shows needs no report.
            if matches!(command, Command::EnterSafe { .. })
                && self
                    .view
                    .as_ref()
                    .is_some_and(|s| matches!(s.mode, Mode::Safe | Mode::Estop))
            {
                continue;
            }
            let epoch = self.epoch();
            let ctl = self.keys.ctl();
            let Some(link) = self.link.as_mut() else {
                return;
            };
            let Ok((request, sealed)) = link.request(&ctl, epoch, &command) else {
                return self.fail(Fault::Internal, now_us);
            };
            self.hint = None;
            let query = self.ask(ctl, Some(sealed), QUERY_MS, now_us);
            self.pending = Some(Pending {
                query,
                request,
                command,
                tries: 0,
            });
        }
    }

    /// Whether the request in flight is a join step: its failure starts
    /// over.
    fn joining(&self) -> bool {
        matches!(self.phase, Phase::Attach | Phase::Declare)
            && self.pending.as_ref().is_some_and(|p| {
                matches!(
                    p.command,
                    Command::Attach { .. } | Command::DeclareParts { .. }
                )
            })
    }

    fn ctl_unanswered(&mut self, now_us: u64) {
        let joining = self.joining();
        let Some(mut pending) = self.pending.take() else {
            return;
        };
        if let Some(hint) = self.hint.take() {
            if joining {
                return self.fail(hint, now_us);
            }
            return self.next_ctl(now_us);
        }
        pending.tries += 1;
        if pending.tries >= CTL_TRIES {
            if joining {
                return self.fail(Fault::NoReply, now_us);
            }
            // A stop report is retried until the host answers it.
            if is_stop(&pending.command) {
                self.queue.push_front(pending.command);
            }
            return self.next_ctl(now_us);
        }
        // The same request id: the host answers a retry from its reply
        // cache, never applying it twice.
        let epoch = self.epoch();
        let ctl = self.keys.ctl();
        let Some(link) = self.link.as_mut() else {
            return;
        };
        let Ok(sealed) = link.resend(&ctl, pending.request, epoch, &pending.command) else {
            return self.fail(Fault::Internal, now_us);
        };
        pending.query = self.ask(ctl, Some(sealed), QUERY_MS, now_us);
        self.pending = Some(pending);
    }

    #[inline(never)]
    fn ctl_reply(&mut self, payload: &[u8], now_us: u64) {
        let now = ns(now_us);
        let ctl = self.keys.ctl();
        let (Some(link), Some(pending)) = (self.link.as_mut(), self.pending.as_ref()) else {
            return;
        };
        let mut bytes = payload.to_vec();
        let reply = match link.reply_view(&ctl, &mut bytes, pending.request, now) {
            Ok(ReplyOutcome::Answer(reply)) => reply,
            Ok(ReplyOutcome::Hint(refusal)) => {
                self.hint = Some(Fault::Refused(refusal.status, refusal.message));
                return;
            }
            Ok(ReplyOutcome::Other(_)) | Err(_) => return,
        };
        self.last_heard = now_us;
        let joining = self.joining();
        let Some(pending) = self.pending.take() else {
            return;
        };
        self.hint = None;
        let reply = *reply;
        // A view participant's replies carry its view, not the state.
        if let Some(view) = reply.view {
            self.observe(view, now);
            self.check_leave(now_us);
            if self.phase == Phase::Stopped || self.pending.is_none() && self.link.is_none() {
                return;
            }
        }
        match reply.result {
            Err(refusal) if joining => {
                return self.fail(Fault::Refused(refusal.status, refusal.message), now_us);
            }
            // Its first outcome is unknown: send it again, anew.
            Err(refusal) if refusal.status == CtlStatus::Expired && is_stop(&pending.command) => {
                self.queue.push_front(pending.command.clone());
            }
            _ => {}
        }
        match (self.phase, &pending.command) {
            (Phase::Attach, Command::Attach { .. }) if self.in_lobby() => {
                // In the lobby a robot never declares: it waits.
                self.phase = Phase::Waiting;
                self.fault = None;
                self.reached_home();
                self.next_heartbeat = now_us;
                self.last_heard = now_us;
                self.wait = None;
                self.send_wait(now_us);
            }
            (Phase::Attach, Command::Attach { .. }) => {
                self.phase = Phase::Declare;
                self.queue.push_front(Command::DeclareParts {
                    spec_version: SPEC_VERSION,
                    robot_json: self.config.robot_json.clone(),
                    seal: self.paths.clone(),
                    // A networked board: no tethered board id behind it.
                    board_id: None,
                });
            }
            (Phase::Declare, Command::DeclareParts { .. }) => {
                if !self.build_edges() {
                    return self.fail(Fault::Internal, now_us);
                }
                self.sync_demands();
                self.phase = Phase::Running;
                self.fault = None;
                self.refused_ms = RETRY_MS;
                if self.run.is_none() {
                    self.reached_home();
                }
                self.next_heartbeat = now_us;
                self.last_heard = now_us;
            }
            _ => {}
        }
        self.next_ctl(now_us);
    }

    /// This robot's parts as the host declared them, an edge per command
    /// group, and a subscription to each commandable part's commands (as
    /// the host-side `Edge::attach` does).
    fn build_edges(&mut self) -> bool {
        let Some(view) = self.view.as_ref() else {
            return false;
        };
        let me = &self.config.name;
        let mut parts = Vec::with_capacity(self.paths.len());
        for path in &self.paths {
            match view
                .parts
                .iter()
                .find(|p| &p.owner == me && p.path.as_str() == path)
            {
                Some(part) => parts.push(part.clone()),
                None => return false,
            }
        }
        let mut edges: Vec<Edge> = Vec::new();
        let mut edge_of = alloc::vec![None; parts.len()];
        for (i, part) in parts.iter().enumerate() {
            let Some(command) = &part.command else {
                continue;
            };
            if edge_of[i].is_some() {
                continue;
            }
            let members: Vec<usize> = match &command.group {
                Some(group) => parts
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| {
                        p.command
                            .as_ref()
                            .is_some_and(|c| c.group.as_deref() == Some(group))
                    })
                    .map(|(j, _)| j)
                    .collect(),
                None => alloc::vec![i],
            };
            let specs = members.iter().map(|&j| parts[j].clone()).collect();
            let Ok(core) = EdgeCore::new(specs, me.clone(), EdgeConfig::default()) else {
                return false;
            };
            for &j in &members {
                edge_of[j] = Some(edges.len());
            }
            // Latches before it are no news to a new edge: one still held
            // stops it all the same.
            edges.push(Edge {
                core,
                latches_seen: self.latches,
                refused: 0,
                last_refusal: None,
            });
        }
        for (i, part) in parts.iter().enumerate() {
            if part.command.is_none() || self.subscribed[i] {
                continue;
            }
            let Ok(sub) = u8::try_from(usize::from(SUB_COMMANDS) + i) else {
                return false;
            };
            self.subscribed[i] = true;
            self.out.push(Action::Subscribe {
                class: class_of(part.priority),
                sub,
                key: self.keys.commands(me, part.path.as_str()),
            });
        }
        self.samples = parts
            .iter()
            .map(|p| Some(Published::new(&self.keys, me, p.stream().ok()?)))
            .collect();
        self.applied_streams = parts
            .iter()
            .map(|p| Some(Published::new(&self.keys, me, p.applied_stream()?.ok()?)))
            .collect();
        self.command_keys = parts
            .iter()
            .map(|p| self.keys.commands(me, p.path.as_str()))
            .collect();
        self.parts = parts;
        self.edges = edges;
        self.edge_of = edge_of;
        self.tick_periods = self
            .parts
            .iter()
            .map(|p| u64::try_from(p.control_period().as_micros()).unwrap_or(u64::MAX))
            .collect();
        self.seqs.clear();
        true
    }

    fn heartbeat(&mut self, now_us: u64) {
        self.next_heartbeat = now_us + u64::from(self.config.heartbeat_ms) * 1000;
        self.heartbeat_seq += 1;
        let (epoch, mode) = self
            .view
            .as_ref()
            .map_or((Epoch(0), Mode::Disarmed), |s| (s.epoch, s.mode));
        let queue_depth = self
            .edges
            .iter()
            .map(|e| u32::try_from(e.core.queue_depth()).unwrap_or(u32::MAX))
            .sum();
        let heartbeat = Heartbeat {
            participant: self.config.name.clone(),
            epoch,
            mode,
            seq: self.heartbeat_seq,
            t1_ns: ns(now_us),
            previous: self.previous.take(),
            last_seq_seen: Vec::new(),
            last_chunk_accepted: self.accepted,
            last_chunk_applied: self.applied,
            queue_depth,
            health: Health::Ok,
            rejected: self.rejected(),
            clock_refused: self
                .edges
                .iter()
                .filter_map(|e| e.core.clock_refused().cloned())
                .collect(),
        };
        let Some(link) = self.link.as_mut() else {
            return;
        };
        if let Ok(sealed) = link.heartbeat(&self.routes.heartbeat, &heartbeat) {
            self.out.push(Action::Publish {
                class: CLASS_CONTROL,
                key: self.routes.heartbeat.clone(),
                payload: sealed,
            });
        }
    }

    fn fail(&mut self, fault: Fault, now_us: u64) {
        // Silence this long means dead links, or a host with new ones to
        // make. Each failure from here on reopens them, at the pace of the
        // retries below, until something answers.
        if matches!(fault, Fault::NoReply | Fault::HostLost)
            && now_us.saturating_sub(self.last_heard) >= u64::from(LINK_LOST_MS) * 1000
        {
            self.out.push(Action::Reconnect);
        }
        let joining = self.phase != Phase::Running;
        // Nobody answered, or someone other than the host did: the address
        // is wrong or down.
        let unreachable = matches!(fault, Fault::NoReply | Fault::Handshake)
            && matches!(self.phase, Phase::Host | Phase::Handshake);
        if self.run.is_some() && joining && unreachable && self.run_at + 1 < self.run_targets.len()
        {
            // The run's next address.
            self.run_at += 1;
            self.target = self.run_targets[self.run_at].clone();
            self.fault = Some(fault);
            return self.rejoin_here();
        }
        if let Some(join) = self.run.take() {
            // A run's join that failed is reported in the next wait; a run
            // whose host went quiet is left. Either way, back to the lobby
            // at once.
            if joining && let Some(lobby) = self.lobby.as_mut() {
                lobby.failed(&join.run, fault_error(&fault));
            }
            self.fault = Some(fault);
            return self.enter_lobby(false);
        }
        // Refused (a board flashed under a name its device may not
        // embody): it would be refused again, so back off rather than spin.
        let refused = matches!(fault, Fault::Refused(..));
        self.fault = Some(fault);
        // Subscriptions and the token again, too: the transport may not have
        // reached the host to declare them.
        let (tenant, session) = (self.tenant.clone(), self.session);
        self.move_to(tenant, session);
        self.phase = Phase::Retry;
        let mut pause = RETRY_MS;
        if refused {
            pause = self.refused_ms;
            self.refused_ms = (self.refused_ms * 2).min(MAX_BACKOFF_MS);
        } else if matches!(self.fault, Some(Fault::Internal | Fault::HostLost)) {
            // A host that keeps going quiet, or a fault of its own: back off
            // (1 s doubling to 30 s) until it reaches its home again.
            pause = self.backoff_ms;
            self.backoff_ms = (self.backoff_ms * 2).min(MAX_BACKOFF_MS);
        }
        if unreachable && self.config.homes.len() > 1 {
            // The home's next address, at once; after a whole round,
            // back off (1 s doubling to 30 s) and let the platform look.
            self.home_at = (self.home_at + 1) % self.config.homes.len();
            pause = 0;
            if self.home_at == 0 {
                self.home_unreachable = true;
                pause = self.backoff_ms;
                self.backoff_ms = (self.backoff_ms * 2).min(MAX_BACKOFF_MS);
            }
            self.retarget_home();
            self.phase = Phase::Retry;
        } else if unreachable {
            self.home_unreachable = true;
            pause = self.backoff_ms;
            self.backoff_ms = (self.backoff_ms * 2).min(MAX_BACKOFF_MS);
        }
        self.retry_at = now_us + u64::from(pause) * 1000;
    }

    /// Points at the home it tries now, with the home's keys: a new host
    /// means new sessions to it.
    fn retarget_home(&mut self) {
        let home = self.config.homes[self.home_at % self.config.homes.len()].clone();
        self.credential = self.config.credential.clone();
        if home != self.target {
            self.target = home;
            let (tenant, session) = (self.tenant.clone(), self.session);
            self.move_to(tenant, session);
            self.phase = Phase::Retry;
        }
    }

    /// It reached its home: the next round starts from here, at once.
    fn reached_home(&mut self) {
        self.home_unreachable = false;
        self.backoff_ms = RETRY_MS;
    }

    /// Joins the run again at the address it now points at.
    fn rejoin_here(&mut self) {
        let (tenant, session) = (self.tenant.clone(), self.session);
        self.move_to(tenant, session);
    }

    /// Forgets everything of the session it was in; the e-stop latch and
    /// the lobby's report stay.
    fn reset_session(&mut self) {
        self.sealed_seen = false;
        self.waiting = None;
        self.initiator = None;
        self.link = None;
        self.pending = None;
        self.wait = None;
        self.queue.clear();
        self.hint = None;
        self.parts.clear();
        self.samples.clear();
        self.applied_streams.clear();
        self.command_keys.clear();
        self.edges.clear();
        self.edge_of = alloc::vec![None; self.paths.len()];
        self.view = None;
    }

    fn in_lobby(&self) -> bool {
        self.lobby.is_some() && self.run.is_none()
    }

    /// Sends the lobby wait: the one in flight again (the same request id,
    /// a fresh seq), or a new one.
    fn send_wait(&mut self, now_us: u64) {
        let ctl = self.keys.ctl();
        let epoch = self.epoch();
        let Some(link) = self.link.as_mut() else {
            return;
        };
        let sealed = match self.wait.take() {
            Some((_, request, command)) => link
                .resend(&ctl, request, epoch, &command)
                .map(|sealed| (request, command, sealed)),
            None => {
                let Some(lobby) = self.lobby.as_ref() else {
                    return;
                };
                let command = Command::LobbyWait(lobby.wait());
                link.request(&ctl, epoch, &command)
                    .map(|(request, sealed)| (request, command, sealed))
            }
        };
        let Ok((request, command, sealed)) = sealed else {
            return self.fail(Fault::Internal, now_us);
        };
        let timeout = u32::try_from(WAIT_RETRY.as_millis()).unwrap_or(u32::MAX);
        let query = self.ask(ctl, Some(sealed), timeout, now_us);
        self.wait = Some((query, request, command));
    }

    /// The lobby's answer to a wait: an order for this robot, or a refusal.
    #[inline(never)]
    fn wait_reply(&mut self, payload: &[u8], now_us: u64) {
        let now = ns(now_us);
        let ctl = self.keys.ctl();
        let (Some(link), Some((_, request, _))) = (self.link.as_mut(), self.wait.as_ref()) else {
            return;
        };
        let mut bytes = payload.to_vec();
        let Ok(ReplyOutcome::Answer(reply)) = link.reply_view(&ctl, &mut bytes, *request, now)
        else {
            return;
        };
        self.last_heard = now_us;
        self.wait = None;
        let reply = *reply;
        if let Some(view) = reply.view {
            self.observe(view, now);
        }
        if let Err(refusal) = reply.result {
            // The lobby dropped the wait: a new one.
            if refusal.status == CtlStatus::Expired {
                return self.send_wait(now_us);
            }
            // A host that hosts no lobby, or refuses this robot one: try
            // again later.
            return self.fail(Fault::Refused(refusal.status, refusal.message), now_us);
        }
        // The lobby takes it: a later refusal starts its backoff over.
        self.refused_ms = RETRY_MS;
        let taken = match (reply.assignment, self.lobby.as_mut()) {
            (Some(assignment), Some(lobby)) => lobby.take(assignment),
            _ => return self.send_wait(now_us),
        };
        match taken {
            Ok(join) => self.enter_run(join),
            // Someone else's order: keep waiting.
            Err(_) => self.send_wait(now_us),
        }
    }

    /// Leaves the run it is in when the view says so.
    fn check_leave(&mut self, now_us: u64) {
        if self.run.is_none() || self.phase != Phase::Running {
            return;
        }
        let Some(view) = self.view.as_ref() else {
            return;
        };
        let current = view.generation(&self.config.name);
        let leave = lobby::leave(view.closing(), current, self.generation);
        if let Some(why) = &leave {
            self.left = Some((why.clone(), view.revision, current, self.generation));
        }
        match leave {
            None => {}
            Some(Leave::Closing(_)) => self.enter_lobby(true),
            Some(Leave::Lost) => self.enter_lobby(false),
            Some(Leave::Superseded) => self.stop(),
        }
        let _ = now_us;
    }

    /// Detaches from the session it is in, without waiting for the answer
    /// (the host also sees its heartbeats stop).
    fn detach(&mut self) {
        let ctl = self.keys.ctl();
        let epoch = self.epoch();
        if let Some(link) = self.link.as_mut()
            && let Ok((_, sealed)) = link.request(&ctl, epoch, &Command::Detach)
        {
            self.query(CLASS_CONTROL, ctl, Some(sealed));
        }
    }

    /// Moves to `session` of `tenant` on the same host: a new handshake,
    /// subscriptions and generation.
    fn move_to(&mut self, tenant: TenantId, session: SessionId) {
        self.reset_session();
        self.out.push(Action::Reset);
        self.subscribed = alloc::vec![false; self.paths.len()];
        self.keys = SessionKeys::new(&tenant, session);
        self.routes = Routes::new(&self.keys, &self.config.name);
        self.tenant = tenant;
        self.session = session;
        self.seqs.clear();
        // The next poll subscribes there and asks for its host.
        self.started = false;
        self.phase = Phase::Host;
    }

    /// Back to the lobby: detaching first when the run is closing.
    fn enter_lobby(&mut self, detach: bool) {
        if detach {
            self.detach();
        }
        self.run = None;
        self.run_targets.clear();
        self.target = self.config.homes[self.home_at % self.config.homes.len()].clone();
        self.credential = self.config.credential.clone();
        if let Ok(tenant) = TenantId::new(crate::address::TENANT) {
            self.move_to(tenant, SessionId::named(LOBBY_SESSION));
        }
    }

    /// Joins the run `join` names: detaches from the lobby, then joins the
    /// run's session (hosted at home for now: the same host and key).
    fn enter_run(&mut self, join: Join) {
        self.detach();
        let Ok(tenant) = TenantId::new(&join.tenant) else {
            if let Some(lobby) = self.lobby.as_mut() {
                lobby.failed(
                    &join.run,
                    CtlError::new(CtlStatus::InvalidRequest, "the join's tenant isn't one"),
                );
            }
            return self.enter_lobby(false);
        };
        let session = join.session;
        // Hosted at home, or on a satellite: its addresses, its key.
        self.run_targets = join
            .addresses
            .iter()
            .filter_map(|a| crate::address::host_port(a))
            .collect();
        self.run_at = 0;
        if let Some(first) = self.run_targets.first() {
            self.target = first.clone();
            self.credential = self.config.credential.with_host(join.host_key);
        }
        self.run = Some(join);
        self.move_to(tenant, session);
    }

    /// Superseded: another process of this device holds its name. Safe,
    /// off the network, until reset.
    fn stop(&mut self) {
        self.reset_session();
        self.out.push(Action::Reset);
        self.run = None;
        self.phase = Phase::Stopped;
    }
}

/// Whether `command` is a stop report: sent until the host answers it.
fn is_stop(command: &Command) -> bool {
    matches!(
        command,
        Command::EnterSafe { .. } | Command::Estop { .. } | Command::Clear
    )
}

/// The robot's e-stop latch, as an edge takes it.
#[derive(Clone, Copy)]
struct Latch {
    /// How many times it latched.
    latches: u64,
    /// The code it last latched with.
    code: ReasonCode,
    latched: bool,
}

/// Gives `edge` the robot's latch: a latch it hasn't seen yet stops and
/// fences it even when cleared since (as the host-side edge does).
fn apply_latch(edge: &mut Edge, latch: Latch) {
    if edge.latches_seen != latch.latches {
        edge.latches_seen = latch.latches;
        edge.core.set_local_estop(Some(latch.code));
    }
    edge.core
        .set_local_estop(latch.latched.then_some(latch.code));
}

/// What a failed join reports to the lobby.
fn fault_error(fault: &Fault) -> CtlError {
    match fault {
        Fault::Refused(status, message) => CtlError::new(*status, message.clone()),
        Fault::Handshake => CtlError::new(
            CtlStatus::Unauthenticated,
            "the run's host failed the handshake",
        ),
        Fault::NoReply | Fault::HostLost => {
            CtlError::new(CtlStatus::NotPresent, "no reply from the run's host")
        }
        Fault::Internal => CtlError::new(CtlStatus::InvalidRequest, "the robot failed to join"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn robot() -> Box<Robot> {
        let keypair = |b: u8| {
            let mut private = zeroize::Zeroizing::new([b; 32]);
            rlmesh_session_core::crypto::StaticKeypair::from_private(&mut private)
        };
        Robot::new(Config {
            tenant: TenantId::new("test").unwrap(),
            session: SessionId::named("default"),
            name: ParticipantId::new("eyes").unwrap(),
            credential: Credential::new(keypair(1), *keypair(2).public()),
            robot_json: String::from(
                r#"{"eyes": {"kind": "joints", "names": ["pan"], "fields": ["pos"], "hz": 50}}"#,
            ),
            heartbeat_ms: 50,
            lobby: false,
            homes: alloc::vec![(String::from("127.0.0.1"), 7447)],
            home_id: String::new(),
        })
        .unwrap()
    }

    #[test]
    fn a_host_that_never_answers_never_grows_the_ctl_queue() {
        let mut robot = robot();
        for i in 0..10_000_u32 {
            robot.enqueue(Command::EnterSafe {
                reason: alloc::format!("fall {i}"),
                code: ReasonCode::Unspecified,
            });
            robot.enqueue(Command::GetKeys);
            robot.enqueue(Command::DemandAck {
                channel: ChannelId(i % 100),
                revision: u64::from(i),
            });
        }
        assert!(
            robot.queue.len() <= CTL_QUEUE_MAX,
            "{} queued",
            robot.queue.len()
        );
        // The newest stop report is the one kept, and a key request is still
        // there.
        assert!(
            robot
                .queue
                .iter()
                .any(|c| matches!(c, Command::EnterSafe { reason, .. } if reason == "fall 9999"))
        );
        assert_eq!(
            robot
                .queue
                .iter()
                .filter(|c| matches!(c, Command::EnterSafe { .. }))
                .count(),
            1
        );
    }

    #[test]
    fn a_gate_admits_its_rate_with_a_burst_of_two() {
        let mut gate = Gate::new(10.0);
        let admitted = (0..100_u64).filter(|i| gate.admit(i * 10_000)).count();
        assert!((10..=12).contains(&admitted), "{admitted} of 100 at 100 Hz");
        // Jitter around the declared rate is not decimation.
        let mut gate = Gate::new(30.0);
        let admitted = (0..300_u64)
            .filter(|i| gate.admit(i * 33_334 + if i % 2 == 0 { 2_000 } else { 0 }))
            .count();
        assert_eq!(admitted, 300);
    }

    #[test]
    fn an_outlet_follows_its_plan_within_the_send_ceiling() {
        let offered =
            |outlet: &mut Outlet| (0..50_u64).filter(|i| outlet.admit(i * 20_000)).count();
        // No demand and no ceiling: a board's default rate, from 50 Hz.
        let mut outlet = Outlet::default();
        outlet.install(0, None, None, false);
        assert!((10..=12).contains(&offered(&mut outlet)));
        // Demanded `every`: every sample.
        let mut outlet = Outlet::default();
        outlet.install(1, Some(Plan::Every), None, false);
        assert_eq!(offered(&mut outlet), 50);
        // Latest at 10 Hz, from 50 Hz.
        let mut outlet = Outlet::default();
        outlet.install(1, Some(Plan::Latest { hz: 10.0 }), None, false);
        assert!((10..=12).contains(&offered(&mut outlet)));
        assert_eq!(outlet.revision, 1);
        // A ceiling below the plan wins, and holds under `every`.
        let mut outlet = Outlet::default();
        outlet.install(1, Some(Plan::Latest { hz: 40.0 }), Some(10.0), false);
        assert!((10..=12).contains(&offered(&mut outlet)));
        let mut outlet = Outlet::default();
        outlet.install(2, Some(Plan::Every), Some(10.0), false);
        assert!((10..=12).contains(&offered(&mut outlet)));
        // Events and non-wall clocks send every sample, whatever the plan.
        let mut outlet = Outlet::default();
        outlet.install(3, Some(Plan::Latest { hz: 1.0 }), None, true);
        assert_eq!(offered(&mut outlet), 50);
    }
}
