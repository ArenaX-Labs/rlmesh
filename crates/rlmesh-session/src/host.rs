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
//! **Who is speaking**: a participant attaches through the
//! Noise IK handshake on `hs`, which gives each process (a generation) its
//! own `k_up` and `k_down`. Every `ctl` request and heartbeat must be
//! sealed under the `k_up` of the generation in effect, and every reply and
//! echo is sealed under its `k_down`; anything else is `UNAUTHENTICATED`.
//! A handshake makes a candidate, which takes effect when the host accepts
//! its sealed `Attach`; a newer generation of a participant supersedes the
//! older one. Request ids and a per-generation reply cache make a retried
//! verb idempotent. `host/state` and `host/tick` are signed with the
//! host's Ed25519 key, under the host instance and one publication counter.
//!
//! **Keys**: the host draws every other key, delivers it on
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
//! In a virtual-clock session the host also runs the tick barrier
//! and publishes its ticks on `host/tick`.
//!
//! **Slow peers**: a Zenoh send runs on the calling thread and can wait on
//! a link that stopped reading, so the host task never makes one that could.
//! What it publishes is dropped for a congested link and sent again; replies
//! and history subscriptions go out from other threads. Virtual ticks are
//! the exception: the barrier needs every one. Zenoh still routes other
//! participants' BLOCK traffic to such a peer from its own threads, so the
//! host's listening [`Transport`] cuts a link whose peer stopped reading
//! (see [`Transport::close`] and the transport's link guard).
//!
//! The host is a library ([`SessionHost::start`] on any [`Transport`]);
//! `rlmesh host` runs one in the foreground.

mod actor;
mod config;
mod datagram;
mod history;
mod peer;

pub use config::{HistoryConfig, HostConfig, KeepRecording};
pub use datagram::{BOARD_DATA_VAR, BoardData, DatagramRelay};

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use crate::wire::{AuthoritySnapshot, DeviceView, ParticipantId, SessionKeys};
use rlmesh_session_core::auth::issuer::GrantId;
use rlmesh_session_core::auth::{
    Credential, Device, HostAuth, HostIdentity, Keyring, LOCAL_DEVICE, Pin,
};
use rlmesh_session_core::crypto::{Rng, RngError, SignedSender};
use tokio::sync::{broadcast, mpsc, oneshot, watch};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::authority::{Access, Authority, AuthorityConfig, AuthorityEvent, DeviceAccess};
use crate::error::SessionError;
#[cfg(feature = "fleet")]
use crate::host::actor::LobbyDone;
#[cfg(feature = "fleet")]
use crate::host::actor::LobbyKey;
#[cfg(feature = "fleet")]
use crate::host::actor::LobbyWaiting;
use crate::host::actor::{Link, Waiting};
use crate::host::history::{HistoryStore, HistorySync};
use crate::host::peer::{SharedPeers, on_heartbeat_sample, on_report_sample};
use crate::metrics::ParticipantHealth;
use crate::transport::{
    CONTROL_CONGESTION, CONTROL_PRIORITY, RESENT_CONGESTION, TrafficClass, Transport,
};
#[cfg(feature = "fleet")]
use crate::wire::ReasonCode;

/// Heartbeats queued between the Zenoh callback and the host task. Beyond
/// this the host is overloaded and drops them (the echo was already sent).
const HEARTBEAT_QUEUE: usize = 4096;
/// How long a server asks whether a participant's token is declared.
const PRESENCE_QUERY: Duration = Duration::from_millis(500);

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

/// One change of the device table.
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
    /// The scene closes: the reason's code, who, and why.
    #[cfg(feature = "fleet")]
    Close(ReasonCode, String, String),
}

impl SessionHost {
    /// Declares the host's queryables, subscribers and liveliness token, and
    /// starts its task. The authority snapshot is queryable once this
    /// returns.
    #[allow(clippy::needless_update)]
    pub async fn start(transport: Transport, config: HostConfig) -> Result<Self, SessionError> {
        config.validate()?;
        let keys = SessionKeys::new(&config.tenant, config.session);
        let mut access = Access::default();
        for device in config.devices.iter().filter(|d| d.id != LOCAL_DEVICE) {
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
                rearm_robots: config.rearm_robots,
                places: config.places.clone(),
                #[cfg(feature = "fleet")]
                scene: config.scene.clone(),
                // The core's `scene`, when another crate turned its `fleet`
                // on (a board's) and this crate's is off.
                ..AuthorityConfig::default()
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
        // Wall-clock sessions only: a virtual clock's barrier counts on every
        // sample, which a datagram may lose.
        let datagram = config
            .datagram
            .clone()
            .filter(|_| config.virtual_clock.is_none());
        auth.offer_datagram(datagram.as_ref().map(|relay| relay.offer()));
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
        // Observer reports, sealed to the host: the newest per generation is
        // relayed in `host/health`.
        let reports = transport
            .session(TrafficClass::Data)
            .declare_subscriber(keys.reports())
            .callback({
                let (keys, peers) = (keys.clone(), peers.clone());
                move |sample| on_report_sample(&keys, &peers, &sample)
            })
            .await?;
        let handshakes = control.declare_queryable(keys.handshake()).await?;
        let ctl = control.declare_queryable(keys.ctl()).await?;
        let state = control.declare_queryable(keys.host_state()).await?;
        let views = control.declare_queryable(keys.host_views()).await?;
        let presence = control
            .liveliness()
            .declare_subscriber(keys.participants())
            .history(true)
            .await?;
        let publisher = control
            .declare_publisher(keys.host_state())
            .priority(CONTROL_PRIORITY)
            .congestion_control(RESENT_CONGESTION)
            .express(true)
            .await?;
        let ticks = control
            .declare_publisher(keys.host_tick())
            .priority(CONTROL_PRIORITY)
            .congestion_control(CONTROL_CONGESTION)
            .express(true)
            .await?;
        // Observers only: a dropped health message is replaced within a
        // second, so it never blocks control traffic.
        let health_out = control
            .declare_publisher(keys.host_health())
            .congestion_control(zenoh::qos::CongestionControl::Drop)
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
        #[cfg(feature = "fleet")]
        let (lobby_done, lobby_answers) = mpsc::unbounded_channel::<LobbyDone>();

        let identity = config.identity.clone();
        let history_sync = config
            .history
            .enabled
            .then(|| {
                HistorySync::start(
                    transport.clone(),
                    keys.clone(),
                    store.clone(),
                    epoch.clone(),
                    host_keys.clone(),
                )
            })
            .transpose()?;
        let mut actor = HostActor {
            transport,
            peers,
            auth,
            signer,
            views: BTreeMap::new(),
            last_publish_ns: 0,
            keys: keys.clone(),
            config,
            authority,
            links: BTreeMap::new(),
            epoch,
            publisher,
            ticks,
            health_out,
            snapshot_tx,
            health_tx,
            events: events.clone(),
            store,
            history: history_sync,
            host_keys,
            waiting: HashMap::new(),
            failed: Vec::new(),
            last_keep: None,
            #[cfg(feature = "fleet")]
            lobby: HashMap::new(),
            #[cfg(feature = "fleet")]
            lobby_done,
            #[cfg(feature = "fleet")]
            lobby_answers,
            quiet: crate::quiet::Quiet::default(),
            datagram: datagram.map(actor::DatagramSession::new),
        };
        actor.publish().await;
        info!(session = %actor.config.session, key = actor.keys.prefix(), "session host up");

        let task = tokio::spawn({
            let cancel = cancel.clone();
            async move {
                actor
                    .run(
                        cancel,
                        hb_rx,
                        access_rx,
                        (handshakes, ctl, state, views),
                        presence,
                    )
                    .await;
                // Withdraw presence first, so participants stop talking to a
                // host that no longer answers, then the rest.
                drop(token);
                drop(heartbeats);
                drop(reports);
                drop(history);
                if let Some(history) = actor.history.take() {
                    history.stop().await;
                }
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

    /// Changes `device`'s entry at runtime: its
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

    /// Revokes `device`: its handshakes are refused from
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
    /// at every handshake and its board at every declare. A moved or removed pin
    /// detaches the robot's current
    /// embodiment as lost (`ROBOT_LOST`: an ACTIVE session goes SAFE); a
    /// first pin to the device that embodies it changes nothing.
    ///
    /// # Errors
    /// [`SessionError::Closed`] once the host stopped.
    pub async fn pin(&self, robot: &ParticipantId, pin: Option<Pin>) -> Result<(), SessionError> {
        self.change(Change::Pin(robot.clone(), pin)).await
    }

    #[cfg(feature = "fleet")]
    /// Closes the scene this session runs, in one
    /// revision: DISARMED (ESTOP stays) with `code` (`SCENE_ENDED`, or
    /// `SCENE_FAILED` with the error in `detail`) and `by`, every lease
    /// ended with it, and `scene.closing` set in the state and every device
    /// view, so robots detach and return to the lobby.
    ///
    /// # Errors
    /// [`SessionError::Config`] outside a scene; [`SessionError::Closed`]
    /// once the host stopped.
    pub async fn close_scene(
        &self,
        code: ReasonCode,
        by: &str,
        detail: &str,
    ) -> Result<(), SessionError> {
        self.change(Change::Close(code, by.to_string(), detail.to_string()))
            .await
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

struct HostActor {
    transport: Transport,
    peers: SharedPeers,
    auth: HostAuth,
    /// Signs `host/state`, `host/tick` and `host/view/*`: one publication
    /// counter.
    signer: SignedSender,
    /// The device view last published to each robot that asked for one.
    views: BTreeMap<ParticipantId, DeviceView>,
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
    /// Signed `host/health`.
    health_out: zenoh::pubsub::Publisher<'static>,
    snapshot_tx: watch::Sender<AuthoritySnapshot>,
    health_tx: watch::Sender<Vec<ParticipantHealth>>,
    events: broadcast::Sender<AuthorityEvent>,
    store: HistoryStore,
    /// The history subscriptions' thread, when history is on.
    history: Option<HistorySync>,
    /// The host's own receivers of every stream key: what the history
    /// store opens samples with.
    host_keys: Arc<Mutex<Keyring>>,
    /// Grants waiting for their robots' keys.
    waiting: HashMap<GrantId, Waiting>,
    /// Grants a participant's end failed, still to answer.
    failed: Vec<GrantId>,
    /// Host clock at the last Keep.
    last_keep: Option<i64>,
    /// Lobby verbs waiting for their answer (a long poll), by sender
    /// generation and request id.
    #[cfg(feature = "fleet")]
    lobby: HashMap<LobbyKey, LobbyWaiting>,
    /// Where a waiting lobby verb's answer comes back to the host task.
    #[cfg(feature = "fleet")]
    lobby_done: mpsc::UnboundedSender<LobbyDone>,
    /// Those answers, for the host task's loop.
    #[cfg(feature = "fleet")]
    lobby_answers: mpsc::UnboundedReceiver<LobbyDone>,
    /// Warnings held back while their subject's last one is recent.
    quiet: crate::quiet::Quiet,
    /// Its robots on the host's datagram path, when it offers one.
    datagram: Option<actor::DatagramSession>,
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
