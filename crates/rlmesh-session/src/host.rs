//! The session host: the single writer of authority state.
//!
//! One task owns the [`Authority`] and serialises everything that touches it:
//! handshakes, `ctl` queries, heartbeats, liveliness changes and the expiry
//! tick. Presence is heartbeats only: a liveliness delete is a hint that
//! makes the host check a participant's heartbeats sooner, never proof it
//! left. Heartbeat echoes are sent from the Zenoh callback itself, off the
//! task, so the host's hold time (t3 − t2) stays small and queueing cannot
//! inflate the measured round trip. Their publishers are declared when a
//! participant attaches, so the echo path never resolves a key per message.
//!
//! **Who is speaking** (design §8.3): a participant attaches through the
//! Noise IK handshake on `hs`, which gives each process (a generation) its
//! own `k_up` and `k_down`. Every `ctl` request and heartbeat must be
//! sealed under the `k_up` of the generation in effect, and every reply and
//! echo is sealed under its `k_down`; anything else is `UNAUTHENTICATED`.
//! A handshake makes a candidate, which takes effect when the host accepts
//! its sealed `Attach`; a newer generation of a participant supersedes the
//! older one. Request ids and a per-generation reply cache make a retried
//! verb idempotent (§8.6). `host/state` and `host/tick` are signed with the
//! host's Ed25519 key, under the host instance and one publication counter
//! (§8.7).
//!
//! **Keys** (§8.4, §8.5): the host draws every other key, delivers it on
//! `p/{participant}/keys` sealed under the recipient's `k_down`, and
//! retries until the recipient's `KeysAck` (its
//! [`KeyIssuer`](crate::auth::KeyIssuer)). Each part's
//! stream key goes to every member (and the host) and, once they installed
//! it, to its robot to send under; a robot's new generation rotates it. A
//! grant (or a rebind) that mints a token is prepared on a copy of the
//! authority, its `k_cmd` and `k_obs` go to the robots, and the verb is
//! applied, re-checked, only once every robot acknowledged: the holder gets
//! its keys in the reply. Meanwhile the host keeps dispatching; a retry of
//! the pending request joins it.
//!
//! The host also keeps the session's **history store** (§8.8): for every
//! declared stream (bulk streams only when configured) it subscribes to the
//! samples and keeps a bounded ring per stream, in the current epoch only,
//! served on `h/{robot}/{path}`. A controller queries it on attach, on a
//! standby swap and to recover a gap, never per tick: steady-state windows
//! are assembled from the live stream. A query names the epoch, an exclusive
//! `after_seq`, an inclusive `until_seq` watermark taken from the live
//! stream and a limit; the reply is one or more [`SampleBatch`] pages, the
//! first carrying `truncated` (retention ran out before the range) and the
//! gaps the store knows of. A query must be sealed under the requester's
//! `k_up`, and its pages are sealed under its `k_down` with the query's
//! request id (§8.6); the store opens every sample under its stream's key
//! and keeps only the session's current epoch.
//!
//! In a virtual-clock session the host also runs the tick barrier (§11.3)
//! and publishes its ticks on `host/tick`.
//!
//! The host is a library ([`SessionHost::start`] on any [`Transport`]);
//! `rlmesh host` runs one in the foreground.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use crate::wire::codec;
use crate::wire::part::{MAX_BATCH, MAX_CHUNK_BYTES};
use crate::wire::{
    Applied, AuthoritySnapshot, ChannelId, ChannelPriority, Command, CtlError, CtlReply, CtlStatus,
    Epoch, HeartbeatEcho, HistoryQuery, IssuedKey, Lease, LeaseToken, ParticipantId, Role,
    SampleBatch, SampleMeta, SeqRange, SessionId, SessionKeys, StreamSpec, TenantId,
};
use bytes::Bytes;
use rlmesh_session_core::auth::issuer::{
    GrantId, GrantPart, GrantRequest, Joining, Prepared, StreamSlot,
};
use rlmesh_session_core::auth::{
    Admit, Credential, Device, HostAuth, HostIdentity, Keyring, LOCAL_DEVICE, Pin, RequestWindow,
};
use rlmesh_session_core::crypto::sealed::Header;
use rlmesh_session_core::crypto::{
    Kind, Rng, RngError, SealReceiver, SealSender, SignedKind, SignedSender,
};
use tokio::sync::{broadcast, mpsc, oneshot, watch};
use tokio::task::JoinHandle;
use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};
use zenoh::Wait;
use zenoh::query::Query;
use zenoh::sample::SampleKind;

use crate::authority::{
    Access, Applied as Accepted, Authority, AuthorityConfig, AuthorityEvent, DeviceAccess,
    Identity, SealPolicy,
};
use crate::clock::{Clock, MonotonicClock, OffsetEstimator, OffsetEstimatorConfig};
use crate::error::SessionError;
use crate::metrics::{LinkStats, ParticipantHealth, Report};
use crate::transport::{
    CONTROL_CONGESTION, CONTROL_PRIORITY, HEARTBEAT_CONGESTION, TrafficClass, Transport,
};
use crate::virtual_clock::VirtualConfig;

/// Heartbeats queued between the Zenoh callback and the host task. Beyond
/// this the host is overloaded and drops them (the echo was already sent).
const HEARTBEAT_QUEUE: usize = 4096;
/// How long a server asks whether a participant's token is declared.
const PRESENCE_QUERY: Duration = Duration::from_millis(500);
/// The most gap ranges a stream's history remembers.
const MAX_GAPS: usize = 64;
/// How long a handshake's keys wait for their `Attach` before the next
/// handshake drops them.
const CANDIDATE_TTL: Duration = Duration::from_secs(10);
/// What the host answers a request it cannot authenticate.
const UNSEALED: &str = "this session requires authenticated participants (feature auth.sealed): \
                        attach through the handshake with the session's address";

/// The operating system's RNG: host keys, handshake ephemerals and
/// pairing nonces.
pub struct OsRng;

impl Rng for OsRng {
    fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), RngError> {
        use rand::TryRng;
        rand::rngs::SysRng
            .try_fill_bytes(dest)
            .map_err(|_| RngError)
    }
}

/// A lock that a panicked holder left behind is still the state to use.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// What the history store keeps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistoryConfig {
    /// Keep histories at all.
    pub enabled: bool,
    /// Keep bulk streams (cameras) too. Off by default: a raw camera ring is
    /// large, and windows on bulk streams are rare.
    pub bulk: bool,
    /// Samples kept per stream: the policy-side history cap (§3.4).
    pub depth: usize,
}

impl Default for HistoryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            bulk: false,
            depth: 1024,
        }
    }
}

/// How a host runs.
#[derive(Clone)]
pub struct HostConfig {
    /// The tenant namespace.
    pub tenant: TenantId,
    /// The session.
    pub session: SessionId,
    /// The session clock (wall mode: the host's monotonic clock).
    pub clock: Arc<dyn Clock>,
    /// How long a lease holder's heartbeats may stop before its lease lapses.
    pub lease_grace: Duration,
    /// How long any participant's heartbeats may stop before it is lost (at
    /// least `lease_grace`; a liveliness delete shortens it to that).
    pub presence_grace: Duration,
    /// How often lease expiry is checked: the lapse latency on top of grace.
    pub tick: Duration,
    /// Republish the snapshot this often, so clock mappings stay fresh.
    /// While any lease is held the host also republishes every quarter of
    /// `lease_grace`, so edges see renewed expiries in time.
    pub republish: Duration,
    /// Log every participant's link health this often (`None`: never).
    pub health_log: Option<Duration>,
    /// How participants' clock offsets are filtered.
    pub offset: OffsetEstimatorConfig,
    /// Features this session implements.
    pub supports: Vec<String>,
    /// The history store.
    pub history: HistoryConfig,
    /// Virtual clock mode (§11.3); `None` is wall mode.
    pub virtual_clock: Option<VirtualConfig>,
    /// The parts whose streams, chunks and lockstep traffic are encrypted
    /// rather than signed (§8.2): none by default.
    pub seal: SealPolicy,
    /// The host's keys, shared by every session a multi-session host runs:
    /// participants handshake with its static key, and it signs its
    /// broadcasts with its Ed25519 key. Holders of its local secret (the
    /// address's key) attach as device `local`.
    pub identity: Arc<HostIdentity>,
    /// The devices besides `local` it accepts, each with its entry (design
    /// §4); none by default. Change them at runtime with
    /// [`SessionHost::set_access`], [`SessionHost::admit`] and
    /// [`SessionHost::revoke`].
    pub devices: Vec<Device>,
    /// Robot name → its pin (rlmeshd's `[robots.*]`, design §4): the
    /// device, checked at the handshake, and a tethered board, checked at
    /// `DeclareParts`. None by default, so a name binds to its first
    /// embodiment's device for the host instance. Change one at runtime with
    /// [`SessionHost::pin`].
    pub pins: BTreeMap<ParticipantId, Pin>,
}

impl HostConfig {
    /// Defaults for `session` in `tenant`: wall mode, a 250 ms lease grace,
    /// a 1 s presence grace, a 10 ms expiry tick, a 500 ms republish, a
    /// 1 s health log, and fresh keys.
    ///
    /// # Panics
    /// If the operating system's random number generator fails.
    pub fn new(tenant: TenantId, session: SessionId) -> Self {
        Self {
            identity: Arc::new(
                HostIdentity::generate(&mut OsRng)
                    .expect("the operating system's random number generator failed"),
            ),
            tenant,
            session,
            clock: MonotonicClock::new().shared(),
            lease_grace: Duration::from_millis(250),
            presence_grace: Duration::from_secs(1),
            tick: Duration::from_millis(10),
            republish: Duration::from_millis(500),
            health_log: Some(Duration::from_secs(1)),
            offset: OffsetEstimatorConfig::default(),
            supports: crate::wire::version::FEATURES
                .iter()
                .map(ToString::to_string)
                .collect(),
            history: HistoryConfig::default(),
            virtual_clock: None,
            seal: SealPolicy::None,
            devices: Vec::new(),
            pins: BTreeMap::new(),
        }
    }

    fn validate(&self) -> Result<(), SessionError> {
        if self.lease_grace.is_zero() || self.presence_grace.is_zero() {
            return Err(SessionError::Config(
                "lease_grace and presence_grace must be positive".into(),
            ));
        }
        if self.tick.is_zero() || self.republish.is_zero() {
            return Err(SessionError::Config(
                "tick and republish must be positive".into(),
            ));
        }
        if self.virtual_clock.is_some_and(|v| v.tick_ns == 0) {
            return Err(SessionError::Config("tick_ns must be positive".into()));
        }
        for (i, device) in self.devices.iter().enumerate() {
            if device.id == LOCAL_DEVICE
                || self.devices[..i]
                    .iter()
                    .any(|d| d.id == device.id || d.public == device.public)
            {
                return Err(SessionError::Config(format!(
                    "device {} is local or listed twice",
                    device.id
                )));
            }
        }
        Ok(())
    }
}

impl std::fmt::Debug for HostConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostConfig")
            .field("tenant", &self.tenant)
            .field("session", &self.session)
            .field("lease_grace", &self.lease_grace)
            .field("presence_grace", &self.presence_grace)
            .field("tick", &self.tick)
            .field("republish", &self.republish)
            .field("health_log", &self.health_log)
            .field("offset", &self.offset)
            .field("supports", &self.supports)
            .field("history", &self.history)
            .field("virtual_clock", &self.virtual_clock)
            .field("seal", &self.seal)
            .field("devices", &self.devices)
            .field("pins", &self.pins)
            .finish_non_exhaustive()
    }
}

/// A running session host. Dropping it stops the host task; prefer
/// [`SessionHost::shutdown`], which waits for it.
pub struct SessionHost {
    keys: SessionKeys,
    identity: Arc<HostIdentity>,
    snapshot: watch::Receiver<AuthoritySnapshot>,
    health: watch::Receiver<Vec<ParticipantHealth>>,
    events: broadcast::Sender<AuthorityEvent>,
    access: mpsc::UnboundedSender<AccessChange>,
    cancel: CancellationToken,
    task: Option<JoinHandle<()>>,
}

/// A runtime change of the device table, and who is told it applied.
type AccessChange = (Change, oneshot::Sender<Result<(), SessionError>>);

/// One change of the device table (design §4, §8.5).
#[derive(Debug)]
enum Change {
    /// A device's new entry.
    Access(String, DeviceAccess),
    /// A device it accepts from now on.
    Admit(Device),
    /// A device it no longer accepts.
    Revoke(String),
    /// A robot name's new pin, or none.
    Pin(ParticipantId, Option<Pin>),
}

impl SessionHost {
    /// Declares the host's queryables, subscribers and liveliness token, and
    /// starts its task. The authority snapshot is queryable once this
    /// returns.
    pub async fn start(transport: Transport, config: HostConfig) -> Result<Self, SessionError> {
        config.validate()?;
        let keys = SessionKeys::new(&config.tenant, config.session);
        let mut access = Access::default();
        for device in &config.devices {
            access.set(device.id.clone(), device.access.clone());
        }
        let authority = Authority::new(
            config.session,
            AuthorityConfig {
                lease_grace_ns: u64::try_from(config.lease_grace.as_nanos()).unwrap_or(u64::MAX),
                presence_grace_ns: u64::try_from(config.presence_grace.as_nanos())
                    .unwrap_or(u64::MAX),
                supports: config.supports.clone(),
                virtual_clock: config.virtual_clock,
                seal: config.seal.clone(),
                access: access.clone(),
            },
        );
        let epoch = Arc::new(AtomicU64::new(authority.epoch().0));
        let control = transport.control().clone();
        let mut instance = [0u8; 8];
        OsRng
            .fill_bytes(&mut instance)
            .map_err(|err| SessionError::Config(err.to_string()))?;
        let host_instance = u64::from_le_bytes(instance);
        let mut auth = HostAuth::new(
            &config.identity,
            config.tenant.as_str(),
            &config.session.to_string(),
            host_instance,
        );
        for device in &config.devices {
            auth.enroll(device.public, device.id.clone());
        }
        for (robot, pin) in &config.pins {
            auth.pin(robot, Some(pin.clone()));
        }
        let signer = config.identity.signer(host_instance);

        let (hb_tx, hb_rx) = mpsc::channel(HEARTBEAT_QUEUE);
        let peers: SharedPeers = Arc::default();
        let heartbeats = control
            .declare_subscriber(keys.heartbeats())
            .callback({
                let (keys, clock, epoch, peers) = (
                    keys.clone(),
                    config.clock.clone(),
                    epoch.clone(),
                    peers.clone(),
                );
                move |sample| {
                    let t2 = clock.now_ns();
                    on_heartbeat_sample(&keys, &*clock, &epoch, &peers, &hb_tx, &sample, t2);
                }
            })
            .await?;
        let handshakes = control.declare_queryable(keys.handshake()).await?;
        let ctl = control.declare_queryable(keys.ctl()).await?;
        let state = control.declare_queryable(keys.host_state()).await?;
        let presence = control
            .liveliness()
            .declare_subscriber(keys.participants())
            .history(true)
            .await?;
        let publisher = control
            .declare_publisher(keys.host_state())
            .priority(CONTROL_PRIORITY)
            .congestion_control(CONTROL_CONGESTION)
            .express(true)
            .await?;
        let ticks = control
            .declare_publisher(keys.host_tick())
            .priority(CONTROL_PRIORITY)
            .congestion_control(CONTROL_CONGESTION)
            .express(true)
            .await?;
        let host_keys = Arc::new(Mutex::new(Keyring::new(None, 0)));
        let store = HistoryStore {
            depth: config.history.depth.max(1),
            clock: Some(config.clock.clone()),
            rings: Arc::default(),
            peers: peers.clone(),
            access: Arc::new(RwLock::new(access)),
        };
        let history = transport
            .session(TrafficClass::Data)
            .declare_queryable(keys.histories())
            .callback({
                let (store, keys) = (store.clone(), keys.clone());
                move |query| store.serve(&keys, &query)
            })
            .await?;
        // Declared last: once a participant sees the token, every queryable
        // above is reachable (Zenoh keeps declaration order per link).
        let token = control
            .liveliness()
            .declare_token(keys.host_liveliness())
            .await?;

        let initial = authority.snapshot(Vec::new(), config.clock.now_ns());
        let (snapshot_tx, snapshot) = watch::channel(initial);
        let (health_tx, health) = watch::channel(Vec::new());
        let (events, _) = broadcast::channel(256);
        let cancel = CancellationToken::new();
        let (access_tx, access_rx) = mpsc::unbounded_channel();

        let identity = config.identity.clone();
        let mut actor = HostActor {
            transport,
            peers,
            auth,
            signer,
            last_publish_ns: 0,
            keys: keys.clone(),
            config,
            authority,
            links: BTreeMap::new(),
            epoch,
            publisher,
            ticks,
            snapshot_tx,
            health_tx,
            events: events.clone(),
            store,
            recorded: HashMap::new(),
            host_keys,
            waiting: HashMap::new(),
            failed: Vec::new(),
        };
        actor.publish().await;
        info!(session = %actor.config.session, key = actor.keys.prefix(), "session host up");

        let task = tokio::spawn({
            let cancel = cancel.clone();
            async move {
                actor
                    .run(cancel, hb_rx, access_rx, (handshakes, ctl, state), presence)
                    .await;
                // Withdraw presence first, so participants stop talking to a
                // host that no longer answers, then the rest.
                drop(token);
                drop(heartbeats);
                drop(history);
                actor.recorded.clear();
                actor.store.clear();
            }
        });

        Ok(Self {
            keys,
            identity,
            snapshot,
            health,
            events,
            access: access_tx,
            cancel,
            task: Some(task),
        })
    }

    /// Changes `device`'s entry at runtime (design §4, §8.5): its
    /// participants that embody a robot lose every lease they hold, the
    /// others are detached (`ACCESS_CHANGED`) and re-attach under the new
    /// entry, and every stream key it can no longer read rotates.
    ///
    /// # Errors
    /// [`SessionError::Config`] for a device the host doesn't know (or
    /// `local`, which may do everything); [`SessionError::Closed`] once the
    /// host stopped.
    pub async fn set_access(&self, device: &str, access: DeviceAccess) -> Result<(), SessionError> {
        self.change(Change::Access(device.to_string(), access))
            .await
    }

    /// Accepts `device` from now on, with its entry: a device enrolled
    /// while the host runs.
    ///
    /// # Errors
    /// [`SessionError::Config`] for `local` or a device it knows already
    /// (change that one with [`Self::set_access`]); [`SessionError::Closed`]
    /// once the host stopped.
    pub async fn admit(&self, device: Device) -> Result<(), SessionError> {
        self.change(Change::Admit(device)).await
    }

    /// Revokes `device` (design §4, §8.5): its handshakes are refused from
    /// now on, every participant of it is detached (`REVOKED`), robots it
    /// embodies too, and every stream key it could read rotates. The names
    /// it is bound to stay bound.
    ///
    /// # Errors
    /// As [`Self::set_access`].
    pub async fn revoke(&self, device: &str) -> Result<(), SessionError> {
        self.change(Change::Revoke(device.to_string())).await
    }

    /// Pins robot name `robot`, or unpins it (`None`): its device is checked
    /// at every handshake and its board at every declare (design §4, tether
    /// spec §7). A moved or removed pin detaches the robot's current
    /// embodiment as lost (`ROBOT_LOST`: an ACTIVE session goes SAFE); a
    /// first pin to the device that embodies it changes nothing.
    ///
    /// # Errors
    /// [`SessionError::Closed`] once the host stopped.
    pub async fn pin(&self, robot: &ParticipantId, pin: Option<Pin>) -> Result<(), SessionError> {
        self.change(Change::Pin(robot.clone(), pin)).await
    }

    async fn change(&self, change: Change) -> Result<(), SessionError> {
        let (done, applied) = oneshot::channel();
        self.access
            .send((change, done))
            .map_err(|_| SessionError::Closed)?;
        applied.await.map_err(|_| SessionError::Closed)?
    }

    /// The session's keys.
    pub fn keys(&self) -> &SessionKeys {
        &self.keys
    }

    /// The credential of a participant holding this host's address (device
    /// `local`).
    pub fn credential(&self) -> Credential {
        self.identity.local_credential()
    }

    /// The latest authority snapshot.
    pub fn snapshot(&self) -> AuthoritySnapshot {
        self.snapshot.borrow().clone()
    }

    /// Every new snapshot.
    pub fn watch(&self) -> watch::Receiver<AuthoritySnapshot> {
        self.snapshot.clone()
    }

    /// Per-participant link health, as of the last health interval.
    pub fn health(&self) -> Vec<ParticipantHealth> {
        self.health.borrow().clone()
    }

    /// Every health interval.
    pub fn watch_health(&self) -> watch::Receiver<Vec<ParticipantHealth>> {
        self.health.clone()
    }

    /// Every authority event from now on.
    pub fn events(&self) -> broadcast::Receiver<AuthorityEvent> {
        self.events.subscribe()
    }

    /// Stops the host task and undeclares everything it declared.
    pub async fn shutdown(mut self) {
        self.cancel.cancel();
        if let Some(task) = self.task.take()
            && let Err(err) = task.await
        {
            warn!(%err, "session host task failed");
        }
    }
}

impl Drop for SessionHost {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

/// One participant generation's keys: a handshake's candidate, or the
/// generation in effect. Shared with the heartbeat callback, which opens
/// heartbeats and seals echoes off the host task.
struct Peer {
    participant: ParticipantId,
    identity: Identity,
    /// Host clock at the handshake.
    created_ns: i64,
    /// `k_up`: requests and heartbeats.
    up: Mutex<SealReceiver>,
    /// `k_down`: replies and echoes. One lock per key: a seq is allocated
    /// and its message sent under it.
    down: Mutex<SealSender>,
    /// Request ids and their replies (encoded `CtlReply`s, sealed afresh
    /// for a retry).
    requests: Mutex<RequestWindow<Vec<u8>>>,
    /// The echo publisher, once in effect.
    echo: std::sync::OnceLock<zenoh::pubsub::Publisher<'static>>,
}

/// Every generation with keys.
#[derive(Default)]
struct Peers {
    /// Candidates and generations in effect, by `k_up`'s id.
    by_key: HashMap<u32, Arc<Peer>>,
    /// The generation in effect, by participant.
    current: HashMap<ParticipantId, Arc<Peer>>,
}

type SharedPeers = Arc<RwLock<Peers>>;

type HeartbeatTx = mpsc::Sender<(ParticipantId, crate::wire::Heartbeat, i64)>;

/// The heartbeat callback: open it under the sender's `k_up` (anything else
/// is dropped: presence is authenticated heartbeats), echo at once (t3 as
/// late as possible), then queue the heartbeat for the host task.
fn on_heartbeat_sample(
    keys: &SessionKeys,
    clock: &dyn Clock,
    epoch: &AtomicU64,
    peers: &SharedPeers,
    queue: &HeartbeatTx,
    sample: &zenoh::sample::Sample,
    t2: i64,
) {
    let Some(id) = keys.participant_of(sample.key_expr().as_str()) else {
        return;
    };
    // Not attached (or superseded): no keys, no presence.
    let Some(peer) = peers.read().ok().and_then(|p| p.current.get(&id).cloned()) else {
        return;
    };
    let mut bytes = sample.payload().to_bytes().into_owned();
    let heartbeat = match lock(&peer.up).open(&keys.heartbeat(&id), &mut bytes) {
        Ok(opened) => codec::decode_heartbeat(opened.body),
        Err(err) => {
            debug!(%id, %err, "heartbeat dropped");
            return;
        }
    };
    let heartbeat = match heartbeat {
        Ok(hb) if hb.participant == id => hb,
        Ok(hb) => {
            warn!(%id, body = %hb.participant, "heartbeat key and body disagree");
            return;
        }
        Err(err) => {
            warn!(%id, %err, "undecodable heartbeat");
            return;
        }
    };
    if let Some(publisher) = peer.echo.get() {
        // t3 under the lock: waiting for it is the host's hold time, never
        // network time.
        let mut down = lock(&peer.down);
        let echo = HeartbeatEcho {
            seq: heartbeat.seq,
            t1_ns: heartbeat.t1_ns,
            t2_ns: t2,
            t3_ns: clock.now_ns(),
            epoch: Epoch(epoch.load(Ordering::Relaxed)),
        };
        let sent = down
            .seal(
                Kind::HeartbeatEcho,
                &keys.heartbeat_echo(&id),
                &codec::encode_echo(&echo),
            )
            .map_err(|err| err.to_string())
            .and_then(|sealed| publisher.put(sealed).wait().map_err(|err| err.to_string()));
        if let Err(err) = sent {
            warn!(%id, %err, "heartbeat echo failed");
        }
    }
    if queue.try_send((id, heartbeat, t2)).is_err() {
        warn!("host overloaded: heartbeat dropped");
    }
}

/// One stored sample.
#[derive(Clone)]
struct Stored {
    seq: u64,
    capture_ns: i64,
    tick: u64,
    applied: Option<Applied>,
    payload: Bytes,
}

/// One stream's ring, in its newest epoch.
struct RingState {
    epoch: Epoch,
    /// The first seq seen in this epoch: anything older was never here.
    first_seq: u64,
    samples: VecDeque<Stored>,
    gaps: VecDeque<SeqRange>,
}

struct Ring {
    spec: StreamSpec,
    state: Mutex<RingState>,
    /// The session's epoch: the only one a sample is kept in (§8.5).
    epoch: Arc<AtomicU64>,
    /// Samples kept.
    depth: usize,
}

impl Ring {
    /// Records a live sample, opened under its stream's key.
    fn record(&self, keys: &Mutex<Keyring>, keyexpr: &str, payload: &zenoh::bytes::ZBytes) {
        let bytes = payload.to_bytes().into_owned();
        let opened = lock(keys).open_sample(keyexpr, bytes, &self.spec.owner, self.spec.applied);
        match opened {
            Ok(sample) => self.record_one(sample),
            Err(err) => debug!(channel = %self.spec.channel, %err, "history sample dropped"),
        }
    }

    fn record_one(&self, sample: crate::wire::Sample) {
        if sample.channel != self.spec.channel || self.spec.check_payload(&sample.payload).is_err()
        {
            return;
        }
        // A forged future epoch can't reset the ring: only the session's.
        if sample.epoch != Epoch(self.epoch.load(Ordering::Relaxed)) {
            return;
        }
        let depth = self.depth;
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        if sample.epoch != state.epoch {
            // A new epoch never seeds from an older one.
            *state = RingState {
                epoch: sample.epoch,
                first_seq: sample.seq,
                samples: VecDeque::with_capacity(depth.min(1024)),
                gaps: VecDeque::new(),
            };
        }
        let last = state.samples.back().map(|s| s.seq);
        match last {
            Some(last) if sample.seq <= last => return,
            Some(last) if sample.seq > last + 1 => {
                if state.gaps.len() == MAX_GAPS {
                    state.gaps.pop_front();
                }
                state.gaps.push_back(SeqRange {
                    first: last + 1,
                    last: sample.seq - 1,
                });
            }
            _ => {}
        }
        if state.samples.len() >= depth {
            state.samples.pop_front();
        }
        state.samples.push_back(Stored {
            seq: sample.seq,
            capture_ns: sample.capture_ns,
            tick: sample.tick,
            applied: sample.applied,
            payload: sample.payload,
        });
        let oldest = state.samples.front().map_or(0, |s| s.seq);
        while state.gaps.front().is_some_and(|g| g.last < oldest) {
            state.gaps.pop_front();
        }
    }

    /// The pages answering `query`.
    fn pages(&self, query: &HistoryQuery, cap: usize, now_ns: i64) -> Vec<SampleBatch> {
        let Ok(state) = self.state.lock() else {
            return Vec::new();
        };
        let empty = |epoch| SampleBatch {
            channel: self.spec.channel,
            epoch,
            publish_ns: now_ns,
            samples: Vec::new(),
            payload: Bytes::new(),
            truncated: false,
            gaps: Vec::new(),
            request_id: 0,
        };
        if query.epoch != state.epoch {
            // Another epoch's samples are never served: an empty page says
            // which epoch the store is in.
            return vec![empty(state.epoch)];
        }
        let until = query.until_seq.unwrap_or(u64::MAX);
        let limit = query
            .limit
            .map_or(cap, |l| usize::try_from(l).unwrap_or(cap).min(cap));
        let selected: Vec<&Stored> = state
            .samples
            .iter()
            .filter(|s| s.seq > query.after_seq && s.seq <= until)
            .collect();
        let selected = &selected[selected.len().saturating_sub(limit)..];
        let oldest = state.samples.front().map_or(u64::MAX, |s| s.seq);
        // Retention ran out: the range starts before the oldest sample kept,
        // and that sample is not the epoch's first.
        let truncated = query.after_seq.saturating_add(1) < oldest && oldest > state.first_seq;
        let (low, high) = (
            selected.first().map_or(query.after_seq, |s| s.seq),
            selected.last().map_or(until, |s| s.seq),
        );
        let gaps: Vec<SeqRange> = state
            .gaps
            .iter()
            .filter(|g| g.last >= low && g.first <= high)
            .copied()
            .collect();
        let per_page = self
            .spec
            .layout
            .stride()
            .filter(|_| self.spec.batchable())
            .and_then(|stride| MAX_CHUNK_BYTES.checked_div(stride))
            .map_or(1, |n| n.clamp(1, MAX_BATCH));
        let mut pages: Vec<SampleBatch> = selected
            .chunks(per_page)
            .map(|chunk| {
                let mut payload = Vec::with_capacity(chunk.iter().map(|s| s.payload.len()).sum());
                for s in chunk {
                    payload.extend_from_slice(&s.payload);
                }
                SampleBatch {
                    samples: chunk
                        .iter()
                        .map(|s| SampleMeta {
                            seq: s.seq,
                            capture_ns: s.capture_ns,
                            tick: s.tick,
                            applied: s.applied,
                        })
                        .collect(),
                    payload: Bytes::from(payload),
                    ..empty(state.epoch)
                }
            })
            .collect();
        if pages.is_empty() {
            pages.push(empty(state.epoch));
        }
        pages[0].truncated = truncated;
        pages[0].gaps = gaps;
        pages
    }
}

/// History rings by `(robot, stream path)`, shared with the history
/// queryable.
/// A stream by `(robot, key path)`.
type StreamKey = (ParticipantId, String);

#[derive(Clone)]
struct HistoryStore {
    depth: usize,
    clock: Option<Arc<dyn Clock>>,
    rings: Arc<RwLock<HashMap<StreamKey, Arc<Ring>>>>,
    /// Requesters' keys: a query is sealed under its sender's `k_up`.
    peers: SharedPeers,
    /// Who may read what: the authority's, kept in step by the host.
    access: Arc<RwLock<Access>>,
}

impl HistoryStore {
    /// Replies to an `h/{robot}/{path}` query, sealed under the `k_up` of a
    /// generation in effect, with its pages sealed under that generation's
    /// `k_down` (§8.6). Anything else is dropped unanswered.
    fn serve(&self, keys: &SessionKeys, query: &Query) {
        let key = query.key_expr().as_str();
        let Some((robot, path)) = keys.history_of(key) else {
            return;
        };
        let ring = self
            .rings
            .read()
            .ok()
            .and_then(|r| r.get(&(robot.clone(), path.to_string())).cloned());
        let Some(ring) = ring else {
            return;
        };
        let mut raw = query
            .payload()
            .map(|p| p.to_bytes().into_owned())
            .unwrap_or_default();
        let Some(peer) = Header::parse(&raw).ok().and_then(|h| {
            let peers = self.peers.read().ok()?;
            let peer = peers.by_key.get(&h.key_id)?;
            peers
                .current
                .get(&peer.participant)
                .filter(|current| Arc::ptr_eq(current, peer))
                .cloned()
        }) else {
            debug!(%key, "a history query not sealed by a member refused");
            return;
        };
        // Read permission, before producing anything.
        let reads = self
            .access
            .read()
            .is_ok_and(|a| a.may_read(&peer.identity.device, &robot, path));
        if !reads {
            let device = &peer.identity.device;
            debug!(%key, %device, "a history query without read permission refused");
            return;
        }
        let request = match lock(&peer.up).open(key, &mut raw) {
            Ok(opened) if opened.header.kind == Kind::HistoryQuery => {
                codec::decode_history_query(opened.body)
            }
            Ok(opened) => {
                debug!(%key, kind = ?opened.header.kind, "not a history query");
                return;
            }
            Err(err) => {
                debug!(%key, %err, "history query dropped");
                return;
            }
        };
        let request = match request {
            Ok(request) => request,
            Err(err) => {
                warn!(%key, %err, "undecodable history query");
                return;
            }
        };
        let now = self.clock.as_ref().map_or(0, |c| c.now_ns());
        for page in ring.pages(&request, self.depth, now) {
            let page = SampleBatch {
                request_id: request.request_id,
                ..page
            };
            let sealed = codec::encode_batch(&page)
                .map_err(|err| err.to_string())
                .and_then(|encoded| {
                    lock(&peer.down)
                        .seal(Kind::HistoryPage, key, &encoded.to_vec())
                        .map_err(|err| err.to_string())
                });
            let sealed = match sealed {
                Ok(sealed) => sealed,
                Err(err) => {
                    warn!(%key, %err, "history page");
                    return;
                }
            };
            if let Err(err) = query.reply(key, sealed).wait() {
                warn!(%key, %err, "history reply failed");
                return;
            }
        }
    }

    fn insert(&self, key: StreamKey, ring: Arc<Ring>) {
        if let Ok(mut rings) = self.rings.write() {
            rings.insert(key, ring);
        }
    }

    fn remove(&self, key: &StreamKey) {
        if let Ok(mut rings) = self.rings.write() {
            rings.remove(key);
        }
    }

    fn clear(&self) {
        if let Ok(mut rings) = self.rings.write() {
            rings.clear();
        }
    }
}

/// A recorded stream: its ring and the subscription feeding it.
struct Recorded {
    channel: ChannelId,
    _subscriber: zenoh::pubsub::Subscriber<()>,
}

/// A participant's timing link, as the host measures it.
struct Link {
    /// The generation it measures: a new process starts a new link.
    generation: u32,
    estimator: OffsetEstimator,
    stats: LinkStats,
}

type Fifo<T> = zenoh::handlers::FifoChannelHandler<T>;
type Queries = zenoh::query::Queryable<Fifo<Query>>;

struct HostActor {
    transport: Transport,
    peers: SharedPeers,
    auth: HostAuth,
    /// Signs `host/state` and `host/tick`: one publication counter.
    signer: SignedSender,
    /// Host clock at the last snapshot publish.
    last_publish_ns: i64,
    keys: SessionKeys,
    config: HostConfig,
    authority: Authority,
    links: BTreeMap<ParticipantId, Link>,
    /// Shared with the echo callback.
    epoch: Arc<AtomicU64>,
    publisher: zenoh::pubsub::Publisher<'static>,
    ticks: zenoh::pubsub::Publisher<'static>,
    snapshot_tx: watch::Sender<AuthoritySnapshot>,
    health_tx: watch::Sender<Vec<ParticipantHealth>>,
    events: broadcast::Sender<AuthorityEvent>,
    store: HistoryStore,
    recorded: HashMap<StreamKey, Recorded>,
    /// The host's own receivers of every stream key: what the history
    /// store opens samples with.
    host_keys: Arc<Mutex<Keyring>>,
    /// Grants waiting for their robots' keys.
    waiting: HashMap<GrantId, Waiting>,
    /// Grants a participant's end failed, still to answer.
    failed: Vec<GrantId>,
}

/// A grant (or rebind) whose keys are on their way to its robots: the verb,
/// what it was prepared against, and every query waiting for its answer.
struct Waiting {
    peer: Arc<Peer>,
    request: crate::wire::Request,
    token: LeaseToken,
    epoch: Epoch,
    robots: Vec<(ParticipantId, u32)>,
    queries: Vec<Query>,
}

/// What the host does with a verb.
enum Outcome {
    /// Answer it now.
    Reply(Result<(), CtlError>, Vec<Lease>, Vec<IssuedKey>),
    /// A candidate sent something other than `Attach`: dropped unanswered.
    Drop,
    /// A grant waiting for its robots: answered once they acknowledged.
    Deferred(GrantId),
}

impl Outcome {
    fn refused(err: CtlError) -> Self {
        Self::Reply(Err(err), Vec::new(), Vec::new())
    }
}

impl HostActor {
    async fn run(
        &mut self,
        cancel: CancellationToken,
        mut heartbeats: mpsc::Receiver<(ParticipantId, crate::wire::Heartbeat, i64)>,
        mut access: mpsc::UnboundedReceiver<AccessChange>,
        (handshakes, ctl, state): (Queries, Queries, Queries),
        presence: zenoh::pubsub::Subscriber<Fifo<zenoh::sample::Sample>>,
    ) {
        let interval = |period: Duration| {
            let mut interval = tokio::time::interval(period.max(Duration::from_millis(1)));
            interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
            interval
        };
        let mut tick = interval(self.config.tick);
        let mut republish = interval(self.config.republish);
        let health_log = self.config.health_log;
        let mut health = interval(health_log.unwrap_or(Duration::from_secs(1)));

        loop {
            tokio::select! {
                () = cancel.cancelled() => break,
                query = handshakes.recv_async() => match query {
                    Ok(query) => self.on_handshake(query).await,
                    Err(_) => break,
                },
                query = ctl.recv_async() => match query {
                    Ok(query) => {
                        self.on_ctl(query).await;
                        self.sync_history().await;
                        self.sync_streams();
                    }
                    Err(_) => break,
                },
                query = state.recv_async() => match query {
                    Ok(query) => self.on_state_query(query).await,
                    Err(_) => break,
                },
                Some((id, heartbeat, t2)) = heartbeats.recv() => {
                    self.on_heartbeat(id, heartbeat, t2);
                }
                Some((change, done)) = access.recv() => {
                    let applied = match change {
                        Change::Access(device, entry) => self.set_access(&device, entry).await,
                        Change::Admit(device) => self.admit(device).await,
                        Change::Revoke(device) => self.revoke(&device).await,
                        Change::Pin(robot, pin) => {
                            self.pin(&robot, pin).await;
                            Ok(())
                        }
                    };
                    let _ = done.send(applied);
                }
                sample = presence.recv_async() => match sample {
                    Ok(sample) => self.on_presence(&sample),
                    Err(_) => break,
                },
                _ = tick.tick() => {
                    // A heartbeat counts from when it arrived (t2), so any
                    // still queued behind a busy host task goes in before
                    // leases and presence are judged.
                    while let Ok((id, heartbeat, t2)) = heartbeats.try_recv() {
                        self.on_heartbeat(id, heartbeat, t2);
                    }
                    let events = self.authority.expire(self.now());
                    self.record(events).await;
                    self.publish_ticks().await;
                    self.publish_renewals().await;
                }
                _ = republish.tick() => self.publish().await,
                _ = health.tick() => {
                    let rows = self.health_rows();
                    if health_log.is_some() {
                        for row in &rows {
                            info!(target: "rlmesh_session::health", "{row}");
                        }
                    }
                    self.health_tx.send_replace(rows);
                    // A missed acknowledgement never detaches anyone: the
                    // step waits, and says for whom (§8.5).
                    for step in self.auth.issuer().pending() {
                        info!(
                            robot = %step.robot,
                            version = step.version,
                            stage = step.stage,
                            waiting = ?step.waiting,
                            "stream key rotation pending",
                        );
                    }
                }
            }
            self.flush_keys().await;
        }
        debug!(session = %self.config.session, "session host stopped");
    }

    fn now(&self) -> i64 {
        self.config.clock.now_ns()
    }

    /// Keeps one history subscription per recordable stream, re-subscribing
    /// when a path's stream id changes.
    async fn sync_history(&mut self) {
        if !self.config.history.enabled {
            return;
        }
        let wanted: BTreeMap<StreamKey, StreamSpec> = self
            .authority
            .parts()
            .flat_map(|part| {
                [
                    part.stream().ok(),
                    part.applied_stream().and_then(Result::ok),
                ]
            })
            .flatten()
            .filter(|s| self.config.history.bulk || s.priority != ChannelPriority::Bulk)
            .map(|s| ((s.owner.clone(), s.path.clone()), s))
            .collect();
        let store = &self.store;
        self.recorded.retain(|key, recorded| {
            let keep = wanted
                .get(key)
                .is_some_and(|s| s.channel == recorded.channel);
            if !keep {
                store.remove(key);
            }
            keep
        });
        for (key, spec) in wanted {
            if self.recorded.contains_key(&key) {
                continue;
            }
            let ring = Arc::new(Ring {
                state: Mutex::new(RingState {
                    epoch: Epoch(0),
                    first_seq: 0,
                    samples: VecDeque::new(),
                    gaps: VecDeque::new(),
                }),
                spec: spec.clone(),
                epoch: self.epoch.clone(),
                depth: self.store.depth,
            });
            let data = self.keys.data(&spec.owner, &spec.path);
            let subscriber = self
                .transport
                .for_priority(spec.priority)
                .declare_subscriber(data.clone())
                .callback({
                    let (ring, keys) = (ring.clone(), self.host_keys.clone());
                    move |sample| ring.record(&keys, &data, sample.payload())
                })
                .await;
            match subscriber {
                Ok(subscriber) => {
                    debug!(channel = %spec.channel, path = %spec.path, "history ring");
                    self.store.insert(key.clone(), ring);
                    self.recorded.insert(
                        key,
                        Recorded {
                            channel: spec.channel,
                            _subscriber: subscriber,
                        },
                    );
                }
                Err(err) => warn!(channel = %spec.channel, %err, "history subscription failed"),
            }
        }
    }

    /// Answers msg 1 of a handshake: msg 2 and a candidate, or a refusal.
    async fn on_handshake(&mut self, query: Query) {
        let now = self.now();
        let answer = query
            .payload()
            .ok_or_else(|| CtlError::new(CtlStatus::InvalidRequest, "a handshake without msg 1"))
            .and_then(|payload| {
                codec::decode_handshake_request(&payload.to_bytes())
                    .map_err(|err| CtlError::new(CtlStatus::InvalidRequest, err))
            })
            .and_then(|request| {
                self.auth.handshake(
                    &mut OsRng,
                    &self.config.identity,
                    &request,
                    &self.authority.config().access,
                )
            });
        let reply = match answer {
            Ok((msg2, candidate)) => {
                debug!(participant = %candidate.participant, generation = candidate.generation, "handshake");
                let ttl = i64::try_from(CANDIDATE_TTL.as_nanos()).unwrap_or(i64::MAX);
                let peer = Arc::new(Peer {
                    participant: candidate.participant,
                    identity: Identity {
                        device: candidate.device,
                        generation: candidate.generation,
                    },
                    created_ns: now,
                    up: Mutex::new(candidate.up),
                    down: Mutex::new(candidate.down),
                    requests: Mutex::default(),
                    echo: std::sync::OnceLock::new(),
                });
                if let Ok(mut peers) = self.peers.write() {
                    let Peers { by_key, current } = &mut *peers;
                    by_key.retain(|_, p| {
                        current
                            .get(&p.participant)
                            .is_some_and(|c| Arc::ptr_eq(c, p))
                            || now.saturating_sub(p.created_ns) < ttl
                    });
                    by_key.insert(candidate.up_key_id, peer);
                }
                Ok(msg2)
            }
            Err(err) => {
                info!(status = ?err.status, "handshake refused: {}", err.message);
                Err(err)
            }
        };
        Self::reply(query, codec::encode_handshake_reply(&reply)).await;
    }

    /// Opens a `ctl` request under its sender's `k_up`.
    fn open_ctl(&self, mut raw: Vec<u8>) -> Result<(Arc<Peer>, Vec<u8>), CtlError> {
        let unauthenticated = |why: String| CtlError::new(CtlStatus::Unauthenticated, why);
        let header = Header::parse(&raw).map_err(|_| unauthenticated(UNSEALED.into()))?;
        let peer = self
            .peers
            .read()
            .ok()
            .and_then(|p| p.by_key.get(&header.key_id).cloned())
            .ok_or_else(|| {
                unauthenticated(format!(
                    "key {} is unknown or superseded: attach again",
                    header.key_id
                ))
            })?;
        let body = lock(&peer.up)
            .open(&self.keys.ctl(), &mut raw)
            .map_err(|err| unauthenticated(format!("the request failed to open: {err}")))?
            .body
            .to_vec();
        Ok((peer, body))
    }

    async fn on_ctl(&mut self, query: Query) {
        let raw = query
            .payload()
            .map(|p| p.to_bytes().into_owned())
            .unwrap_or_default();
        let (peer, body) = match self.open_ctl(raw) {
            Ok(opened) => opened,
            Err(err) => {
                // Unauthenticated: a plain refusal, a hint only.
                info!(status = ?err.status, "ctl refused: {}", err.message);
                let reply = CtlReply {
                    result: Err(err),
                    state: self.snapshot(),
                    leases: Vec::new(),
                    request_id: 0,
                    keys: Vec::new(),
                    assignment: None,
                    run: None,
                };
                Self::reply(query, codec::encode_reply(&reply)).await;
                return;
            }
        };
        let request = codec::decode_request(&body);
        let request_id = request.as_ref().map_or(0, |r| r.request_id);
        let admitted = match lock(&peer.requests).admit(request_id) {
            Admit::Done(reply) => Some(reply.clone()),
            Admit::Expired => Some(Vec::new()),
            Admit::Pending => None,
            Admit::Execute => None,
        };
        // A retry of a grant still waiting for its keys joins it.
        if admitted.is_none()
            && let Some(waiting) = self
                .waiting
                .values_mut()
                .find(|w| Arc::ptr_eq(&w.peer, &peer) && w.request.request_id == request_id)
        {
            waiting.queries.push(query);
            return;
        }
        let encoded = match admitted {
            // A retry of an answered request: the same reply, sealed afresh.
            Some(cached) if !cached.is_empty() => cached,
            Some(_) => self.answer(
                Err(CtlError::new(
                    CtlStatus::Expired,
                    format!("request {request_id} is below the reply window: re-read the state"),
                )),
                Vec::new(),
                Vec::new(),
                request_id,
            ),
            None => {
                let outcome = match request {
                    Ok(request) => self.execute(&peer, request).await,
                    Err(err) => Outcome::refused(CtlError::new(CtlStatus::InvalidRequest, err)),
                };
                let (result, leases, keys) = match outcome {
                    // A verb a candidate may not send yet is dropped
                    // unanswered.
                    Outcome::Drop => return,
                    Outcome::Deferred(id) => {
                        lock(&peer.requests).begin(request_id);
                        if let Some(waiting) = self.waiting.get_mut(&id) {
                            waiting.queries.push(query);
                        }
                        return;
                    }
                    Outcome::Reply(result, leases, keys) => (result, leases, keys),
                };
                let encoded = self.answer(result, leases, keys, request_id);
                lock(&peer.requests).complete(request_id, encoded.clone());
                encoded
            }
        };
        // Ticks go out before the reply: a `Ready` that completed a step
        // returns after the step is published.
        self.publish_ticks().await;
        let sealed = lock(&peer.down).seal(Kind::CtlReply, &self.keys.ctl(), &encoded);
        match sealed {
            Ok(sealed) => Self::reply(query, sealed).await,
            Err(err) => warn!(participant = %peer.participant, %err, "reply not sealed"),
        }
    }

    /// Applies a verb from `peer`.
    async fn execute(&mut self, peer: &Arc<Peer>, request: crate::wire::Request) -> Outcome {
        let now = self.now();
        if request.from != peer.participant {
            return Outcome::refused(CtlError::new(
                CtlStatus::Unauthenticated,
                format!(
                    "sealed under {}'s keys but sent as {}",
                    peer.participant, request.from
                ),
            ));
        }
        let in_effect = self
            .peers
            .read()
            .ok()
            .and_then(|p| p.current.get(&peer.participant).cloned())
            .is_some_and(|current| Arc::ptr_eq(&current, peer));
        if !in_effect {
            if !matches!(request.command, Command::Attach { .. }) {
                return Outcome::Drop;
            }
            let Identity { device, generation } = &peer.identity;
            if let Err(err) = self
                .auth
                .may_take_effect(&peer.participant, device, *generation)
            {
                return Outcome::refused(err);
            }
        }
        debug!(from = %request.from, command = ?request.command, "ctl");
        let generation = peer.identity.generation;
        match &request.command {
            Command::KeysAck(ack) => {
                self.auth
                    .issuer()
                    .ack(&peer.participant, generation, ack, now);
            }
            Command::GetKeys => {
                let keys = self.auth.issuer().get_keys(
                    &peer.participant,
                    generation,
                    &self.authority.config().access,
                );
                return Outcome::Reply(Ok(()), Vec::new(), keys);
            }
            Command::GrantLease { .. } | Command::DeclareBinding { .. } => {
                return self.mint(peer, request).await;
            }
            Command::DeclareParts { board_id, .. } => {
                if let Err(err) = self.auth.check_board(
                    &peer.participant,
                    &peer.identity.device,
                    board_id.as_ref(),
                ) {
                    info!(status = ?err.status, "ctl refused: {}", err.message);
                    return Outcome::refused(err);
                }
            }
            _ => {}
        }
        let attaching = matches!(request.command, Command::Attach { .. });
        let result = self.authority.apply_from(request, &peer.identity, now);
        match result {
            Ok(Accepted { leases, events }) => {
                if !in_effect {
                    self.take_effect(peer);
                }
                self.record(events).await;
                // An attach's reply carries the generation's `k_estop`.
                let keys = if attaching {
                    let estop = self.auth.issuer().estop_key(&peer.participant, generation);
                    estop.into_iter().collect()
                } else {
                    Vec::new()
                };
                Outcome::Reply(Ok(()), leases, keys)
            }
            Err(err) => {
                info!(status = ?err.status, "ctl refused: {}", err.message);
                Outcome::refused(err)
            }
        }
    }

    /// A verb that may mint a lease token (a grant, a rebind): tried on a
    /// copy of the authority with a reserved token. Minting nothing, it is
    /// applied at once; otherwise its keys go to its robots and it waits
    /// (§8.5 Grant).
    async fn mint(&mut self, peer: &Arc<Peer>, request: crate::wire::Request) -> Outcome {
        let now = self.now();
        let token = self.authority.reserve_token();
        let mut trial = self.authority.clone();
        let applied = match trial.apply_minting(request.clone(), &peer.identity, now, token) {
            Ok(applied) => applied,
            Err(err) => {
                info!(status = ?err.status, "ctl refused: {}", err.message);
                return Outcome::refused(err);
            }
        };
        let minted: Vec<Lease> = trial
            .leases()
            .filter(|l| l.token == token)
            .cloned()
            .collect();
        if minted.is_empty() {
            self.authority = trial;
            self.record(applied.events).await;
            return Outcome::Reply(Ok(()), applied.leases, Vec::new());
        }
        let parts: Vec<GrantPart> = minted
            .iter()
            .filter_map(|lease| {
                let part = self.authority.part(lease.channel)?;
                Some(GrantPart {
                    robot: part.owner.clone(),
                    channel: lease.channel,
                    priority: part.priority,
                    security: self.authority.security(lease.channel),
                })
            })
            .collect();
        let mut robots: Vec<(ParticipantId, u32)> = Vec::new();
        for part in &parts {
            if robots.iter().all(|(r, _)| r != &part.robot) {
                let generation = self
                    .authority
                    .participant(&part.robot)
                    .map_or(0, |p| p.generation);
                robots.push((part.robot.clone(), generation));
            }
        }
        let access = &self.authority.config().access;
        let reads_all = robots
            .iter()
            .map(|(robot, _)| robot)
            .filter(|robot| {
                self.authority
                    .parts()
                    .filter(|p| &p.owner == *robot)
                    .all(|p| access.may_read(&peer.identity.device, robot, p.path.as_str()))
            })
            .cloned()
            .collect();
        let grant = GrantRequest {
            holder: peer.participant.clone(),
            holder_generation: peer.identity.generation,
            device: peer.identity.device.clone(),
            class: minted[0].priority,
            token,
            parts,
            reads_all,
        };
        match self.auth.issuer().prepare_grant(&mut OsRng, grant, now) {
            Ok(Prepared { id, evicted }) => {
                for victim in evicted {
                    let preempted = CtlError::new(
                        CtlStatus::LeaseBusy,
                        "preempted: a higher class's grant took this robot's last pending slot",
                    );
                    self.finish_grant(victim, Err(preempted)).await;
                }
                self.waiting.insert(
                    id,
                    Waiting {
                        peer: peer.clone(),
                        request,
                        token,
                        epoch: self.authority.epoch(),
                        robots,
                        queries: Vec::new(),
                    },
                );
                Outcome::Deferred(id)
            }
            Err(err) => {
                info!(status = ?err.status, "grant refused: {}", err.message);
                Outcome::refused(err)
            }
        }
    }

    /// Grant `id`'s robots all installed its keys: re-check it against the
    /// current state (the holder's and robots' generations, the epoch, and
    /// every grant check) and commit it, or refuse it.
    async fn commit_grant(
        &mut self,
        id: GrantId,
    ) -> Result<(Vec<Lease>, Vec<IssuedKey>), CtlError> {
        let changed = |why: &str| {
            CtlError::new(
                CtlStatus::StaleEpoch,
                format!("{why} while the lease's keys were delivered"),
            )
        };
        let Some(waiting) = self.waiting.get(&id) else {
            return Err(changed("the grant ended"));
        };
        let holder_current = self
            .peers
            .read()
            .ok()
            .and_then(|p| p.current.get(&waiting.peer.participant).cloned())
            .is_some_and(|current| Arc::ptr_eq(&current, &waiting.peer));
        let robots_current = waiting.robots.iter().all(|(robot, generation)| {
            self.authority
                .participant(robot)
                .is_some_and(|p| p.generation == *generation)
        });
        if !holder_current || !robots_current {
            self.auth.issuer().refuse_grant(id);
            return Err(changed("a holder or robot restarted"));
        }
        if waiting.epoch != self.authority.epoch() {
            self.auth.issuer().refuse_grant(id);
            return Err(changed("the epoch changed"));
        }
        let (request, identity, token) = (
            waiting.request.clone(),
            waiting.peer.identity.clone(),
            waiting.token,
        );
        let mut trial = self.authority.clone();
        let applied = match trial.apply_minting(request, &identity, self.now(), token) {
            Ok(applied) => applied,
            Err(err) => {
                self.auth.issuer().refuse_grant(id);
                return Err(err);
            }
        };
        let minted: BTreeMap<ChannelId, ()> = trial
            .leases()
            .filter(|l| l.token == token)
            .map(|l| (l.channel, ()))
            .collect();
        let prepared: BTreeMap<ChannelId, ()> = self
            .auth
            .issuer()
            .grant(id)
            .map(|g| g.parts.iter().map(|p| (p.channel, ())).collect())
            .unwrap_or_default();
        if minted != prepared {
            self.auth.issuer().refuse_grant(id);
            return Err(changed("the parts it covers changed"));
        }
        let Some(committed) = self.auth.issuer().commit_grant(id) else {
            return Err(changed("the grant ended"));
        };
        for (channel, key_id) in &committed.key_ids {
            trial.set_key_id(*channel, *key_id);
        }
        let leases = applied
            .leases
            .iter()
            .map(|l| trial.lease(l.channel).cloned().unwrap_or_else(|| l.clone()))
            .collect();
        self.authority = trial;
        self.record(applied.events).await;
        Ok((leases, committed.keys))
    }

    /// Answers every query waiting on grant `id`, once.
    async fn finish_grant(
        &mut self,
        id: GrantId,
        outcome: Result<(Vec<Lease>, Vec<IssuedKey>), CtlError>,
    ) {
        let Some(waiting) = self.waiting.remove(&id) else {
            return;
        };
        let request_id = waiting.request.request_id;
        let encoded = match outcome {
            Ok((leases, keys)) => self.answer(Ok(()), leases, keys, request_id),
            Err(err) => {
                info!(status = ?err.status, "grant refused: {}", err.message);
                self.answer(Err(err), Vec::new(), Vec::new(), request_id)
            }
        };
        lock(&waiting.peer.requests).complete(request_id, encoded.clone());
        let key = self.keys.ctl();
        for query in waiting.queries {
            let sealed = lock(&waiting.peer.down).seal(Kind::CtlReply, &key, &encoded);
            match sealed {
                Ok(sealed) => Self::reply(query, sealed).await,
                Err(err) => warn!(%err, "grant reply not sealed"),
            }
        }
    }

    /// Keeps one stream key slot per declared part (§8.4).
    fn sync_streams(&mut self) {
        let streams: Vec<StreamSlot> = self
            .authority
            .parts()
            .map(|part| StreamSlot {
                robot: part.owner.clone(),
                path: part.path.to_string(),
                channels: [Some(part.channel), part.applied_channel]
                    .into_iter()
                    .flatten()
                    .collect(),
                security: self.authority.security(part.channel),
            })
            .collect();
        let now = self.now();
        let access = &self.authority.config().access;
        if let Err(err) = self
            .auth
            .issuer()
            .sync_streams(&mut OsRng, &streams, access, now)
        {
            warn!(%err, "stream keys");
        }
    }

    /// Sends what the key schedule has due: the host's own receivers first,
    /// then every member's delivery; then decides the grants whose robots
    /// answered (or ran out of time) and answers those a participant's end
    /// failed.
    async fn flush_keys(&mut self) {
        let now = self.now();
        while let Some(delivery) = self.auth.issuer().host_delivery() {
            let ack = lock(&self.host_keys).apply(&delivery, now);
            self.auth.issuer().host_ack(&ack, now);
        }
        for (who, delivery) in self.auth.issuer().deliveries(now) {
            let Some(peer) = self
                .peers
                .read()
                .ok()
                .and_then(|p| p.current.get(&who).cloned())
            else {
                continue;
            };
            let key = self.keys.participant_keys(&who);
            let sealed = lock(&peer.down).seal(
                Kind::KeyDelivery,
                &key,
                &codec::encode_key_delivery(&delivery),
            );
            let sent = match sealed {
                Ok(sealed) => self
                    .transport
                    .control()
                    .put(key, sealed)
                    .priority(CONTROL_PRIORITY)
                    .congestion_control(CONTROL_CONGESTION)
                    .express(true)
                    .await
                    .map_err(|err| err.to_string()),
                Err(err) => Err(err.to_string()),
            };
            if let Err(err) = sent {
                warn!(participant = %who, %err, "key delivery failed");
            }
        }
        for id in std::mem::take(&mut self.failed) {
            let gone = CtlError::new(CtlStatus::NotAttached, "the holder or a robot left");
            self.finish_grant(id, Err(gone)).await;
        }
        let decided = self.auth.issuer().ready_grants(now);
        for (id, ready) in decided {
            let outcome = match ready {
                Ok(()) => self.commit_grant(id).await,
                Err(err) => {
                    self.auth.issuer().refuse_grant(id);
                    Err(err)
                }
            };
            self.finish_grant(id, outcome).await;
        }
    }

    /// Changes `device`'s entry: see [`SessionHost::set_access`].
    async fn set_access(&mut self, device: &str, entry: DeviceAccess) -> Result<(), SessionError> {
        if self.authority.config().access.get(device).is_none() {
            return Err(SessionError::Config(format!(
                "device {device} has no entry on this host"
            )));
        }
        let (events, detached) = self.authority.set_access(device, entry, self.now());
        info!(%device, ?detached, "device entry changed");
        self.access_changed(events).await;
        Ok(())
    }

    /// Accepts a new device: see [`SessionHost::admit`].
    async fn admit(&mut self, device: Device) -> Result<(), SessionError> {
        if device.id == LOCAL_DEVICE || self.authority.config().access.get(&device.id).is_some() {
            return Err(SessionError::Config(format!(
                "device {} is local or known already",
                device.id
            )));
        }
        self.auth.enroll(device.public, device.id.clone());
        let (events, _) = self
            .authority
            .set_access(&device.id, device.access, self.now());
        info!(device = %device.id, "device admitted");
        self.access_changed(events).await;
        Ok(())
    }

    /// Revokes a device: see [`SessionHost::revoke`].
    async fn revoke(&mut self, device: &str) -> Result<(), SessionError> {
        if self.authority.config().access.get(device).is_none() {
            return Err(SessionError::Config(format!(
                "device {device} has no entry on this host"
            )));
        }
        self.auth.unenroll(device);
        let (events, detached) = self.authority.revoke(device, self.now());
        info!(%device, ?detached, "device revoked");
        self.access_changed(events).await;
        Ok(())
    }

    /// The device table changed: the stores see it, and every stream key a
    /// device can no longer read rotates.
    async fn access_changed(&mut self, events: Vec<AuthorityEvent>) {
        let access = self.authority.config().access.clone();
        if let Ok(mut shared) = self.store.access.write() {
            *shared = access.clone();
        }
        self.record(events).await;
        let now = self.now();
        if let Err(err) = self.auth.issuer().reauthorize(&mut OsRng, &access, now) {
            warn!(%err, "stream keys");
        }
    }

    /// Pins a robot name: see [`SessionHost::pin`].
    async fn pin(&mut self, robot: &ParticipantId, pin: Option<Pin>) {
        let device = pin.as_ref().map(|p| p.device.clone());
        let before = self.auth.pin(robot, pin);
        let embodied = self
            .authority
            .participant(robot)
            .filter(|p| p.present && p.has_role(Role::Embodiment))
            .map(|p| p.device.clone());
        info!(%robot, ?device, "pin changed");
        if let Some(embodied) = embodied
            && (before.is_some() || device.as_deref() != Some(embodied.as_str()))
        {
            let events = self.authority.participant_lost(robot, self.now());
            self.record(events).await;
        }
    }

    /// `peer`'s `Attach` was accepted: it is the generation in effect, and
    /// every older one of its participant is dropped. It gets the stream
    /// keys it reads, and a robot's streams rotate to it.
    fn take_effect(&mut self, peer: &Arc<Peer>) {
        let Identity { device, generation } = &peer.identity;
        self.auth
            .take_effect(&peer.participant, device, *generation);
        let now = self.now();
        let robot = self
            .authority
            .participant(&peer.participant)
            .is_some_and(|p| p.has_role(Role::Embodiment));
        let joining = Joining {
            participant: &peer.participant,
            generation: *generation,
            device,
            robot,
        };
        match self
            .auth
            .issuer()
            .attach(&mut OsRng, joining, &self.authority.config().access, now)
        {
            Ok(failed) => self.failed.extend(failed),
            Err(err) => warn!(participant = %peer.participant, %err, "keys"),
        }
        if let Ok(mut peers) = self.peers.write() {
            peers.by_key.retain(|_, p| {
                p.participant != peer.participant || p.identity.generation >= *generation
            });
            peers.current.insert(peer.participant.clone(), peer.clone());
        }
    }

    /// The encoded reply: the result, the snapshot right after it and the
    /// caller's keys.
    fn answer(
        &self,
        result: Result<(), CtlError>,
        leases: Vec<Lease>,
        keys: Vec<IssuedKey>,
        request_id: u64,
    ) -> Vec<u8> {
        codec::encode_reply(&CtlReply {
            result,
            state: self.snapshot(),
            leases,
            request_id,
            keys,
            assignment: None,
            run: None,
        })
    }

    async fn on_state_query(&mut self, query: Query) {
        if let Some(signed) = self.sign_state(&self.snapshot()) {
            Self::reply(query, signed).await;
        }
    }

    /// The snapshot as a `Signed` broadcast (§8.7).
    fn sign_state(&mut self, snapshot: &AuthoritySnapshot) -> Option<Vec<u8>> {
        self.signer
            .sign(
                SignedKind::AuthorityState,
                &self.keys.host_state(),
                &codec::encode_state(snapshot),
            )
            .map_err(|err| warn!(%err, "authority state not signed"))
            .ok()
    }

    /// Replies ride the query's priority and congestion control (Zenoh 1.x
    /// ignores per-reply QoS), so the caller's choice decides both legs.
    async fn reply(query: Query, payload: Vec<u8>) {
        if let Err(err) = query
            .reply(query.key_expr().clone(), payload)
            .express(true)
            .await
        {
            warn!(%err, "reply failed");
        }
    }

    fn on_heartbeat(&mut self, id: ParticipantId, heartbeat: crate::wire::Heartbeat, t2: i64) {
        // Only attached participants are tracked; attach precedes the first
        // heartbeat.
        let Some(link) = self.links.get_mut(&id) else {
            return;
        };
        link.stats.on_heartbeat(t2);
        link.stats.on_report(Report {
            last_seq_seen: heartbeat.last_seq_seen,
            last_chunk_accepted: heartbeat.last_chunk_accepted,
            last_chunk_applied: heartbeat.last_chunk_applied,
            queue_depth: heartbeat.queue_depth,
            health: heartbeat.health,
        });
        if let Some(sample) = heartbeat.previous
            && link.estimator.add(sample)
        {
            link.stats
                .on_round_trip(sample.rtt_ns().max(0).unsigned_abs());
        }
        self.authority.heartbeat(&id, heartbeat.epoch, t2);
    }

    /// Liveliness never makes anyone present (anyone on the network can
    /// declare a token); a delete only has the expiry tick check the
    /// participant's heartbeats sooner. A participant that rejoined under its
    /// id keeps heartbeating, so its old token's late delete loses nothing.
    fn on_presence(&mut self, sample: &zenoh::sample::Sample) {
        let Some(id) = self.keys.participant_of(sample.key_expr().as_str()) else {
            return;
        };
        if sample.kind() == SampleKind::Delete {
            debug!(%id, "participant liveliness withdrawn: checking its heartbeats");
            self.authority.presence_hint(&id, self.now());
        }
    }

    /// Logs and broadcasts events, keeps the link table in step with the
    /// roster, and publishes the new state.
    async fn record(&mut self, events: Vec<AuthorityEvent>) {
        if events.is_empty() {
            return;
        }
        for event in events {
            info!(session = %self.config.session, "{event}");
            match &event {
                AuthorityEvent::Attached { participant, .. } => {
                    let generation = self.declare_echo(participant).await;
                    let offset = self.config.offset;
                    // A new generation is a new process, on its own clock.
                    if self
                        .links
                        .get(participant)
                        .is_none_or(|link| link.generation != generation)
                    {
                        self.links.insert(
                            participant.clone(),
                            Link {
                                generation,
                                estimator: OffsetEstimator::new(offset),
                                stats: LinkStats::new(offset.window),
                            },
                        );
                    }
                }
                AuthorityEvent::Detached { participant, .. } => {
                    self.links.remove(participant);
                    let failed = self.auth.issuer().detach(participant);
                    self.failed.extend(failed);
                    // Detaching ends the generation: its keys go with it.
                    if let Ok(mut peers) = self.peers.write()
                        && let Some(peer) = peers.current.remove(participant)
                    {
                        peers.by_key.retain(|_, p| !Arc::ptr_eq(p, &peer));
                    }
                }
                _ => {}
            }
            // Having no subscriber is fine.
            let _ = self.events.send(event);
        }
        let authority = &self.authority;
        self.auth
            .issuer()
            .end_tokens(|token| authority.leases().any(|l| l.token == token));
        self.sync_streams();
        self.epoch
            .store(self.authority.epoch().0, Ordering::Relaxed);
        self.publish().await;
    }

    async fn publish_ticks(&mut self) {
        for tick in self.authority.take_ticks() {
            let signed = self.signer.sign(
                SignedKind::VirtualTick,
                &self.keys.host_tick(),
                &codec::encode_tick(&tick),
            );
            let sent = match signed {
                Ok(signed) => self.ticks.put(signed).await.map_err(|err| err.to_string()),
                Err(err) => Err(err.to_string()),
            };
            if let Err(err) = sent {
                warn!(%err, "tick publish failed");
            }
        }
    }

    fn snapshot(&self) -> AuthoritySnapshot {
        let now = self.now();
        let clocks = self
            .links
            .iter()
            .filter_map(|(id, link)| {
                link.estimator.estimate(now).map(|estimate| {
                    estimate.mapping(id.clone(), link.estimator.config().drift_bound_ppm)
                })
            })
            .collect();
        self.authority.snapshot(clocks, now)
    }

    fn health_rows(&self) -> Vec<ParticipantHealth> {
        let now = self.now();
        self.links
            .iter()
            .map(|(id, link)| link.stats.health(id, link.estimator.estimate(now), now))
            .collect()
    }

    /// Declares the echo publisher of `id`'s generation in effect, and
    /// returns that generation.
    async fn declare_echo(&self, id: &ParticipantId) -> u32 {
        let Some(peer) = self
            .peers
            .read()
            .ok()
            .and_then(|p| p.current.get(id).cloned())
        else {
            return 0;
        };
        if peer.echo.get().is_none() {
            match self
                .transport
                .control()
                .declare_publisher(self.keys.heartbeat_echo(id))
                .priority(CONTROL_PRIORITY)
                .congestion_control(HEARTBEAT_CONGESTION)
                .express(true)
                .await
            {
                Ok(publisher) => {
                    let _ = peer.echo.set(publisher);
                }
                Err(err) => warn!(%id, %err, "echo publisher"),
            }
        }
        peer.identity.generation
    }

    /// Heartbeats renew leases without a structural change; while any lease
    /// is held, republish often enough that an edge's local expiry never
    /// runs ahead of the host's renewals.
    async fn publish_renewals(&mut self) {
        if self.authority.lease_count() == 0 {
            return;
        }
        let quarter = i64::try_from(self.config.lease_grace.as_nanos() / 4).unwrap_or(i64::MAX);
        if self.now().saturating_sub(self.last_publish_ns) >= quarter {
            self.publish().await;
        }
    }

    async fn publish(&mut self) {
        self.last_publish_ns = self.now();
        let snapshot = self.snapshot();
        let signed = self.sign_state(&snapshot);
        self.snapshot_tx.send_replace(snapshot);
        if let Some(signed) = signed
            && let Err(err) = self.publisher.put(signed).await
        {
            warn!(%err, "authority publish failed");
        }
    }
}

/// Whether a liveliness token on `key` is declared: a participant that left
/// and joined again under its key declares its new token before its old
/// token's delete may arrive, so a delete alone does not mean it is gone.
pub(crate) async fn token_declared(
    control: &zenoh::Session,
    key: &zenoh::key_expr::KeyExpr<'_>,
) -> bool {
    let Ok(replies) = control.liveliness().get(key).timeout(PRESENCE_QUERY).await else {
        return false;
    };
    let mut present = false;
    while let Ok(reply) = replies.recv_async().await {
        present |= reply.result().is_ok();
    }
    present
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::{Encoding, FieldSpec, Layout, Sample};

    fn ring(fields: Vec<FieldSpec>) -> Ring {
        Ring {
            spec: StreamSpec {
                channel: ChannelId(3),
                owner: ParticipantId::new("robot").expect("id"),
                path: "ft".into(),
                layout: Layout::of(&fields).expect("layout"),
                fields,
                encoding: Encoding::Raw,
                priority: ChannelPriority::Realtime,
                event: false,
                max_age: None,
                applied: false,
            },
            state: Mutex::new(RingState {
                epoch: Epoch(0),
                first_seq: 0,
                samples: VecDeque::new(),
                gaps: VecDeque::new(),
            }),
            epoch: Arc::new(AtomicU64::new(2)),
            depth: 6,
        }
    }

    fn put(ring: &Ring, epoch: u64, seq: u64) {
        let sample = Sample {
            channel: ChannelId(3),
            epoch: Epoch(epoch),
            seq,
            capture_ns: 0,
            publish_ns: 0,
            tick: 0,
            payload: Bytes::from(seq.to_le_bytes().to_vec()),
            applied: None,
        };
        ring.record_one(sample);
    }

    fn seqs(pages: &[SampleBatch]) -> Vec<u64> {
        pages
            .iter()
            .flat_map(|p| p.samples.iter().map(|s| s.seq))
            .collect()
    }

    #[test]
    fn the_store_pages_by_epoch_watermark_and_limit_with_gaps_and_truncation() {
        let ring = ring(vec![FieldSpec::new("x", crate::wire::DType::U64, &[])]);
        for seq in [1, 2, 3, 6, 7, 8, 9, 10] {
            put(&ring, 2, seq);
        }
        put(&ring, 1, 11);
        put(&ring, 3, 11);
        let query = |after, until, limit| HistoryQuery {
            epoch: Epoch(2),
            after_seq: after,
            until_seq: until,
            limit,
            request_id: 0,
        };
        // Kept: 3, 6..=10 (depth 6), with 4..=5 known missing.
        let pages = ring.pages(&query(0, None, None), 1024, 0);
        assert_eq!(seqs(&pages), vec![3, 6, 7, 8, 9, 10]);
        assert!(pages[0].truncated, "1 and 2 were evicted");
        assert_eq!(pages[0].gaps, vec![SeqRange { first: 4, last: 5 }]);
        let split = pages[0].split(Some(8)).expect("split");
        assert_eq!(&split[1].payload[..], &6_u64.to_le_bytes());
        // A watermark and a limit: the newest of the range.
        let pages = ring.pages(&query(5, Some(9), Some(2)), 1024, 0);
        assert_eq!(seqs(&pages), vec![8, 9]);
        assert!(!pages[0].truncated && pages[0].gaps.is_empty());
        // Another epoch: an empty page naming the store's epoch.
        let pages = ring.pages(&HistoryQuery::default(), 1024, 0);
        assert!(pages[0].samples.is_empty());
        assert_eq!(pages[0].epoch, Epoch(2));
        // Only the session's epoch is kept: a new one empties the ring, and
        // it never seeds from the old one.
        ring.epoch.store(3, Ordering::Relaxed);
        put(&ring, 3, 1);
        let pages = ring.pages(
            &HistoryQuery {
                epoch: Epoch(3),
                ..HistoryQuery::default()
            },
            1024,
            0,
        );
        assert_eq!(seqs(&pages), vec![1]);
        assert!(!pages[0].truncated);
    }

    #[test]
    fn large_ranges_split_into_pages() {
        let mut ring = ring(vec![FieldSpec::new("x", crate::wire::DType::U64, &[])]);
        ring.depth = 1024;
        ring.epoch.store(1, Ordering::Relaxed);
        for seq in 1..=600 {
            put(&ring, 1, seq);
        }
        let pages = ring.pages(
            &HistoryQuery {
                epoch: Epoch(1),
                ..HistoryQuery::default()
            },
            1024,
            0,
        );
        assert_eq!(
            pages.iter().map(|p| p.samples.len()).collect::<Vec<_>>(),
            vec![MAX_BATCH, MAX_BATCH, 600 - 2 * MAX_BATCH]
        );
        assert_eq!(seqs(&pages), (1..=600).collect::<Vec<_>>());
    }
}
