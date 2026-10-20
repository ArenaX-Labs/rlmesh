//! A participant: presence, heartbeats, the authority view and `ctl` verbs.
//!
//! [`Participant::attach`] announces presence (a liveliness token), waits
//! for the host, subscribes to authority updates, attaches through `ctl` and
//! starts heartbeating on the control class. From then on it keeps the
//! newest authority snapshot (from updates and from every `ctl` reply) and
//! hands out channel endpoints: sample publishers and subscribers, chunk
//! publishers, and the hooks an edge runtime needs ([`AuthorityView`],
//! [`EdgeReport`], [`Participant::report_safe`]).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rlmesh_wire::codec;
use rlmesh_wire::version::PROTOCOL_MINOR;
use rlmesh_wire::{
    AuthoritySnapshot, ChannelId, ChannelSpec, Command, CtlReply, Direction, Epoch, Health,
    Heartbeat, Lease, Mode, ParticipantId, PriorityClass, Request, Role, SessionId, SessionKeys,
    TenantId, TimingSample,
};
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};
use zenoh::query::ConsolidationMode;
use zenoh::sample::SampleKind;

use crate::chunk::ChunkPublisher;
use crate::clock::{Clock, MonotonicClock};
use crate::error::SessionError;
use crate::sample::{SamplePublisher, SampleSubscriber, SubscribeContext, SubscribeOptions};
use crate::transport::{CONTROL_CONGESTION, CONTROL_PRIORITY, HEARTBEAT_CONGESTION, Transport};

/// How a participant attaches.
#[derive(Clone)]
pub struct ParticipantConfig {
    /// The tenant namespace.
    pub tenant: TenantId,
    /// The session.
    pub session: SessionId,
    /// This participant's id.
    pub id: ParticipantId,
    /// A display name.
    pub name: String,
    /// Its roles.
    pub roles: Vec<Role>,
    /// The class lease requests are arbitrated at.
    pub priority: PriorityClass,
    /// This participant's clock domain.
    pub clock: Arc<dyn Clock>,
    /// How often to heartbeat.
    pub heartbeat_period: Duration,
    /// How long to wait for a `ctl` reply.
    pub ctl_timeout: Duration,
    /// How long to wait for the host to appear.
    pub host_timeout: Duration,
    /// Capabilities this participant requires of the session.
    pub capabilities: Vec<String>,
}

impl ParticipantConfig {
    /// Defaults: a 50 ms heartbeat, a 2 s `ctl` timeout, a 10 s wait for the
    /// host, this process's monotonic clock.
    pub fn new(
        tenant: TenantId,
        session: SessionId,
        id: ParticipantId,
        roles: Vec<Role>,
        priority: PriorityClass,
    ) -> Self {
        Self {
            tenant,
            session,
            name: id.to_string(),
            id,
            roles,
            priority,
            clock: MonotonicClock::new().shared(),
            heartbeat_period: Duration::from_millis(50),
            ctl_timeout: Duration::from_secs(2),
            host_timeout: Duration::from_secs(10),
            capabilities: Vec::new(),
        }
    }
}

impl std::fmt::Debug for ParticipantConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParticipantConfig")
            .field("tenant", &self.tenant)
            .field("session", &self.session)
            .field("id", &self.id)
            .field("roles", &self.roles)
            .field("priority", &self.priority)
            .field("heartbeat_period", &self.heartbeat_period)
            .field("ctl_timeout", &self.ctl_timeout)
            .finish_non_exhaustive()
    }
}

/// What an edge reports in its participant's heartbeats.
#[derive(Debug, Default)]
pub struct EdgeReport {
    last_accepted: AtomicU64,
    last_applied: AtomicU64,
    queue_depth: AtomicU32,
}

impl EdgeReport {
    /// The newest accepted chunk's seq.
    pub fn set_accepted(&self, seq: u64) {
        self.last_accepted.store(seq, Ordering::Relaxed);
    }

    /// The newest applied chunk's seq.
    pub fn set_applied(&self, seq: u64) {
        self.last_applied.store(seq, Ordering::Relaxed);
    }

    /// Queued actions.
    pub fn set_queue_depth(&self, depth: usize) {
        self.queue_depth
            .store(u32::try_from(depth).unwrap_or(u32::MAX), Ordering::Relaxed);
    }

    /// `(last accepted, last applied, queue depth)`.
    pub fn get(&self) -> (u64, u64, u32) {
        (
            self.last_accepted.load(Ordering::Relaxed),
            self.last_applied.load(Ordering::Relaxed),
            self.queue_depth.load(Ordering::Relaxed),
        )
    }
}

/// A participant's live view of the authority state, for code that runs off
/// the async path (an edge's control loop). Cheap to clone.
#[derive(Clone, Debug)]
pub struct AuthorityView {
    rx: watch::Receiver<Option<AuthoritySnapshot>>,
    observed_local_ns: Arc<AtomicI64>,
    epoch: Arc<AtomicU64>,
}

impl AuthorityView {
    /// Runs `f` on the newest snapshot without cloning it.
    pub fn with<R>(&self, f: impl FnOnce(Option<&AuthoritySnapshot>) -> R) -> R {
        f(self.rx.borrow().as_ref())
    }

    /// A copy of the newest snapshot.
    pub fn current(&self) -> Option<AuthoritySnapshot> {
        self.rx.borrow().clone()
    }

    /// This participant's clock when it took the newest snapshot.
    pub fn observed_local_ns(&self) -> i64 {
        self.observed_local_ns.load(Ordering::Relaxed)
    }

    /// The newest snapshot's epoch (0 before the first).
    pub fn epoch(&self) -> Epoch {
        Epoch(self.epoch.load(Ordering::Relaxed))
    }

    /// A receiver that wakes on every new snapshot.
    pub fn subscribe(&self) -> watch::Receiver<Option<AuthoritySnapshot>> {
        self.rx.clone()
    }

    /// A view fed by hand (tests and sans-IO users).
    pub fn manual() -> (ManualView, Self) {
        let (tx, rx) = watch::channel(None);
        let observed_local_ns = Arc::new(AtomicI64::new(0));
        let epoch = Arc::new(AtomicU64::new(0));
        (
            ManualView {
                tx,
                observed_local_ns: observed_local_ns.clone(),
                epoch: epoch.clone(),
            },
            Self {
                rx,
                observed_local_ns,
                epoch,
            },
        )
    }
}

/// The feeding end of [`AuthorityView::manual`].
#[derive(Debug)]
pub struct ManualView {
    tx: watch::Sender<Option<AuthoritySnapshot>>,
    observed_local_ns: Arc<AtomicI64>,
    epoch: Arc<AtomicU64>,
}

impl ManualView {
    /// Publishes `snapshot`, observed at `local_ns`.
    pub fn set(&self, snapshot: AuthoritySnapshot, local_ns: i64) {
        self.observed_local_ns.store(local_ns, Ordering::Relaxed);
        self.epoch.store(snapshot.epoch.0, Ordering::Relaxed);
        self.tx.send_replace(Some(snapshot));
    }
}

/// An edge's request that the session go SAFE.
#[derive(Clone, Debug, PartialEq, Eq)]
struct SafeReport {
    epoch: Epoch,
    reason: String,
}

struct Shared {
    transport: Transport,
    keys: SessionKeys,
    id: ParticipantId,
    ctl_timeout: Duration,
    history_timeout: Duration,
    clock: Arc<dyn Clock>,
    view: watch::Sender<Option<AuthoritySnapshot>>,
    /// The view's epoch, for publishers stamping samples off the async path.
    epoch: Arc<AtomicU64>,
    /// This participant's clock when it last took a newer snapshot.
    view_local_ns: Arc<AtomicI64>,
    heartbeats_enabled: AtomicBool,
    health: AtomicU8,
    host_present: watch::Sender<bool>,
    /// The last completed exchange, sent with the next heartbeat.
    completed: Mutex<Option<TimingSample>>,
    /// The newest seq received per subscribed channel.
    seen: Mutex<BTreeMap<ChannelId, Arc<AtomicU64>>>,
    edge: Arc<EdgeReport>,
    /// Edges' requests that the session go SAFE, sent by the reporter task.
    reports: mpsc::UnboundedSender<SafeReport>,
}

impl Shared {
    /// Keeps the newest snapshot, and notes leases this participant lost.
    fn observe(&self, snapshot: AuthoritySnapshot) {
        let id = &self.id;
        self.view.send_if_modified(|view| {
            if view
                .as_ref()
                .is_some_and(|old| !snapshot.is_newer_than(old))
            {
                return false;
            }
            if let Some(old) = view.as_ref() {
                for lease in old.leases.iter().filter(|l| &l.holder == id) {
                    match snapshot.lease(lease.channel) {
                        Some(now) if now.token == lease.token => {}
                        Some(now) => info!(
                            participant = %id,
                            "lease {} {} lost to {} ({})",
                            lease.channel, lease.token, now.holder, now.token
                        ),
                        None => info!(
                            participant = %id,
                            "lease {} {} ended", lease.channel, lease.token
                        ),
                    }
                }
                if old.mode != snapshot.mode {
                    debug!(participant = %id, "mode {} -> {}", old.mode, snapshot.mode);
                }
            }
            self.epoch.store(snapshot.epoch.0, Ordering::Relaxed);
            self.view_local_ns
                .store(self.clock.now_ns(), Ordering::Relaxed);
            *view = Some(snapshot);
            true
        });
    }

    fn seen(&self, channel: ChannelId) -> Arc<AtomicU64> {
        self.seen
            .lock()
            .map(|mut seen| seen.entry(channel).or_default().clone())
            .unwrap_or_default()
    }

    fn epoch_and_mode(&self) -> (u64, Mode) {
        self.view
            .borrow()
            .as_ref()
            .map_or((0, Mode::Disarmed), |s| (s.epoch.0, s.mode))
    }

    /// Sends one verb and waits for the host's reply. Every reply carries the
    /// host's snapshot, which updates this participant's view even when the
    /// verb was refused.
    async fn ctl(
        &self,
        command: Command,
    ) -> Result<(AuthoritySnapshot, Option<Lease>), SessionError> {
        let (epoch, _) = self.epoch_and_mode();
        let request = Request {
            from: self.id.clone(),
            epoch: (epoch > 0).then_some(Epoch(epoch)),
            command,
        };
        let timeout = self.ctl_timeout;
        let replies = self
            .transport
            .control()
            .get(self.keys.ctl())
            .payload(codec::encode_request(&request))
            .consolidation(ConsolidationMode::None)
            .priority(CONTROL_PRIORITY)
            .congestion_control(CONTROL_CONGESTION)
            .express(true)
            .timeout(timeout)
            .await?;
        let reply = replies
            .recv_async()
            .await
            .map_err(|_| SessionError::NoReply(timeout))?;
        let sample = reply.result().map_err(|_| SessionError::NoReply(timeout))?;
        let CtlReply {
            result,
            state,
            lease,
        } = codec::decode_reply(&sample.payload().to_bytes())?;
        self.observe(state.clone());
        result?;
        Ok((state, lease))
    }
}

/// An attached participant. Dropping it stops heartbeats and withdraws its
/// liveliness token, which the host treats as the participant leaving; prefer
/// [`Participant::detach`] or [`Participant::close`].
pub struct Participant {
    shared: Arc<Shared>,
    cancel: CancellationToken,
    tasks: Vec<JoinHandle<()>>,
    _liveliness: zenoh::liveliness::LivelinessToken,
    _state: zenoh::pubsub::Subscriber<()>,
    _echo: zenoh::pubsub::Subscriber<()>,
    _host: zenoh::pubsub::Subscriber<()>,
}

impl Participant {
    /// Announces presence, waits for the host, subscribes to authority
    /// updates, attaches through `ctl` and starts heartbeating.
    pub async fn attach(
        transport: Transport,
        config: ParticipantConfig,
    ) -> Result<Self, SessionError> {
        if config.heartbeat_period.is_zero() {
            return Err(SessionError::Config(
                "heartbeat_period must be positive".into(),
            ));
        }
        let keys = SessionKeys::new(&config.tenant, config.session);
        let control = transport.control().clone();
        let (view, _) = watch::channel(None);
        let (host_present, _) = watch::channel(false);
        let (reports, report_rx) = mpsc::unbounded_channel();
        let shared = Arc::new(Shared {
            transport: transport.clone(),
            keys: keys.clone(),
            id: config.id.clone(),
            ctl_timeout: config.ctl_timeout,
            history_timeout: config.ctl_timeout,
            clock: config.clock.clone(),
            view,
            epoch: Arc::new(AtomicU64::new(0)),
            view_local_ns: Arc::new(AtomicI64::new(0)),
            heartbeats_enabled: AtomicBool::new(true),
            health: AtomicU8::new(health_code(Health::Ok)),
            host_present,
            completed: Mutex::new(None),
            seen: Mutex::new(BTreeMap::new()),
            edge: Arc::new(EdgeReport::default()),
            reports,
        });

        let state = control
            .declare_subscriber(keys.host_state())
            .callback({
                let shared = shared.clone();
                move |sample| match codec::decode_state(&sample.payload().to_bytes()) {
                    Ok(snapshot) => shared.observe(snapshot),
                    Err(err) => warn!(%err, "undecodable authority update"),
                }
            })
            .await?;
        let echo = control
            .declare_subscriber(keys.heartbeat_echo(&config.id))
            .callback({
                let (shared, clock) = (shared.clone(), config.clock.clone());
                move |sample| {
                    let t4 = clock.now_ns();
                    match codec::decode_echo(&sample.payload().to_bytes()) {
                        Ok(echo) => {
                            if let Ok(mut slot) = shared.completed.lock() {
                                *slot = Some(echo.complete(t4));
                            }
                        }
                        Err(err) => warn!(%err, "undecodable heartbeat echo"),
                    }
                }
            })
            .await?;
        let host = control
            .liveliness()
            .declare_subscriber(keys.host_liveliness())
            .history(true)
            .callback({
                let shared = shared.clone();
                move |sample| {
                    let present = sample.kind() == SampleKind::Put;
                    if !present {
                        warn!(participant = %shared.id, "session host liveliness lost");
                    }
                    shared.host_present.send_replace(present);
                }
            })
            .await?;
        let liveliness = control
            .liveliness()
            .declare_token(keys.participant(&config.id))
            .await?;
        // The host declares its queryables before its liveliness token, and
        // Zenoh keeps declaration order per link, so once the token is seen
        // the first `ctl` query cannot miss the host.
        let mut present = shared.host_present.subscribe();
        tokio::time::timeout(config.host_timeout, present.wait_for(|p| *p))
            .await
            .map_err(|_| SessionError::Timeout(config.host_timeout, "the session host".into()))?
            .map_err(|_| SessionError::Closed)?;

        let cancel = CancellationToken::new();
        let mut participant = Self {
            shared: shared.clone(),
            tasks: vec![tokio::spawn(report_safe(
                shared.clone(),
                report_rx,
                cancel.clone(),
            ))],
            cancel,
            _liveliness: liveliness,
            _state: state,
            _echo: echo,
            _host: host,
        };
        shared
            .ctl(Command::Attach {
                name: config.name.clone(),
                roles: config.roles.clone(),
                priority: config.priority,
                protocol_minor: PROTOCOL_MINOR,
                capabilities: config.capabilities.clone(),
            })
            .await?;

        let publisher = control
            .declare_publisher(keys.heartbeat(&config.id))
            .priority(CONTROL_PRIORITY)
            .congestion_control(HEARTBEAT_CONGESTION)
            .express(true)
            .await?;
        participant.tasks.push(tokio::spawn(heartbeat_loop(
            shared,
            publisher,
            config.heartbeat_period,
            participant.cancel.clone(),
        )));
        info!(participant = %config.id, session = %config.session, "attached");
        Ok(participant)
    }

    /// This participant's id.
    pub fn id(&self) -> &ParticipantId {
        &self.shared.id
    }

    /// The session's keys.
    pub fn keys(&self) -> &SessionKeys {
        &self.shared.keys
    }

    /// The transport this participant rides.
    pub fn transport(&self) -> &Transport {
        &self.shared.transport
    }

    /// The newest authority snapshot seen (from updates or `ctl` replies).
    pub fn authority(&self) -> Option<AuthoritySnapshot> {
        self.shared.view.borrow().clone()
    }

    /// Every new snapshot.
    pub fn watch(&self) -> watch::Receiver<Option<AuthoritySnapshot>> {
        self.shared.view.subscribe()
    }

    /// The live view, for code off the async path.
    pub fn authority_view(&self) -> AuthorityView {
        AuthorityView {
            rx: self.shared.view.subscribe(),
            observed_local_ns: self.shared.view_local_ns.clone(),
            epoch: self.shared.epoch.clone(),
        }
    }

    /// Whether the host's liveliness token is visible.
    pub fn host_present(&self) -> watch::Receiver<bool> {
        self.shared.host_present.subscribe()
    }

    /// Waits until the authority view satisfies `predicate`.
    pub async fn wait_for(
        &self,
        what: &str,
        timeout: Duration,
        mut predicate: impl FnMut(&AuthoritySnapshot) -> bool,
    ) -> Result<AuthoritySnapshot, SessionError> {
        let mut watch = self.watch();
        let found = tokio::time::timeout(
            timeout,
            watch.wait_for(|view| view.as_ref().is_some_and(&mut predicate)),
        )
        .await
        .map_err(|_| SessionError::Timeout(timeout, what.to_string()))?
        .map_err(|_| SessionError::Closed)?;
        found.clone().ok_or(SessionError::Closed)
    }

    /// Queries the host for its current snapshot.
    pub async fn fetch_authority(&self) -> Result<AuthoritySnapshot, SessionError> {
        let snapshot = query_authority(
            &self.shared.transport,
            &self.shared.keys,
            self.shared.ctl_timeout,
        )
        .await?;
        self.shared.observe(snapshot.clone());
        Ok(snapshot)
    }

    /// DISARMED or SAFE to ARMED, in a fresh epoch.
    pub async fn arm(&self) -> Result<AuthoritySnapshot, SessionError> {
        Ok(self.shared.ctl(Command::Arm).await?.0)
    }

    /// Back to DISARMED.
    pub async fn disarm(&self) -> Result<AuthoritySnapshot, SessionError> {
        Ok(self.shared.ctl(Command::Disarm).await?.0)
    }

    /// Emergency stop (accepted from anyone).
    pub async fn estop(
        &self,
        reason: impl Into<String>,
    ) -> Result<AuthoritySnapshot, SessionError> {
        Ok(self
            .shared
            .ctl(Command::Estop {
                reason: reason.into(),
            })
            .await?
            .0)
    }

    /// Releases a latched e-stop: an embodiment's local clear.
    pub async fn clear(&self) -> Result<AuthoritySnapshot, SessionError> {
        Ok(self.shared.ctl(Command::Clear).await?.0)
    }

    /// Requests (or renews) the lease on `channel`, preempting a holder of a
    /// lower class. `ttl` defaults to the session's lease grace.
    pub async fn grant_lease(
        &self,
        channel: ChannelId,
        ttl: Option<Duration>,
    ) -> Result<Lease, SessionError> {
        let ttl_ns = ttl.map(|ttl| u64::try_from(ttl.as_nanos()).unwrap_or(u64::MAX));
        let (_, lease) = self
            .shared
            .ctl(Command::GrantLease { channel, ttl_ns })
            .await?;
        lease.ok_or_else(|| {
            SessionError::Wire(rlmesh_wire::WireError::Malformed {
                message: "ctl reply",
                reason: "a granted lease is missing".into(),
            })
        })
    }

    /// Gives a lease back.
    pub async fn release_lease(&self, lease: &Lease) -> Result<AuthoritySnapshot, SessionError> {
        Ok(self
            .shared
            .ctl(Command::ReleaseLease {
                channel: lease.channel,
                token: lease.token,
            })
            .await?
            .0)
    }

    /// Declares channels this participant owns (publishes, or for an
    /// embodiment, consumes actuation on). Re-declaring identically is a
    /// no-op; the table is part of the authority snapshot.
    pub async fn declare_channels(
        &self,
        channels: Vec<ChannelSpec>,
    ) -> Result<AuthoritySnapshot, SessionError> {
        Ok(self.shared.ctl(Command::DeclareChannels(channels)).await?.0)
    }

    /// A declared channel, from the current view.
    pub fn channel(&self, id: ChannelId) -> Result<ChannelSpec, SessionError> {
        self.authority()
            .and_then(|view| view.channel(id).cloned())
            .ok_or(SessionError::UnknownChannel(id))
    }

    /// This participant's clock.
    pub fn clock(&self) -> &Arc<dyn Clock> {
        &self.shared.clock
    }

    /// This participant's clock converted to session time, through its
    /// clock mapping (unconverted until the host has one).
    pub fn session_now_ns(&self) -> i64 {
        self.to_session_ns(self.shared.clock.now_ns())
    }

    /// One of this participant's timestamps in session time.
    pub fn to_session_ns(&self, local_ns: i64) -> i64 {
        local_ns.saturating_add(self.offset_ns())
    }

    /// A session time in this participant's clock.
    pub fn to_local_ns(&self, session_ns: i64) -> i64 {
        session_ns.saturating_sub(self.offset_ns())
    }

    fn offset_ns(&self) -> i64 {
        self.shared
            .view
            .borrow()
            .as_ref()
            .and_then(|v| v.clock(&self.shared.id).map(|c| c.offset_ns))
            .unwrap_or(0)
    }

    /// A publisher for a sense or aux channel this participant declared.
    pub async fn publisher(&self, channel: ChannelId) -> Result<SamplePublisher, SessionError> {
        let spec = self.channel(channel)?;
        if spec.owner.as_ref() != Some(&self.shared.id) || spec.direction == Direction::Actuate {
            return Err(SessionError::WrongChannel {
                channel,
                reason: format!("not a sense or aux channel of {}", self.shared.id),
            });
        }
        let key = self.shared.keys.samples(&self.shared.id, channel);
        SamplePublisher::declare(
            &self.shared.transport,
            key,
            spec,
            self.shared.epoch.clone(),
            self.shared.clock.clone(),
        )
        .await
    }

    /// Subscribes to a declared sense or aux channel.
    pub async fn subscribe(
        &self,
        channel: ChannelId,
        options: SubscribeOptions,
    ) -> Result<SampleSubscriber, SessionError> {
        let spec = self.channel(channel)?;
        let owner = spec
            .owner
            .clone()
            .ok_or_else(|| SessionError::WrongChannel {
                channel,
                reason: "it has no owner".into(),
            })?;
        if spec.direction == Direction::Actuate {
            return Err(SessionError::WrongChannel {
                channel,
                reason: "it is an actuation channel".into(),
            });
        }
        let ctx = SubscribeContext {
            key: self.shared.keys.samples(&owner, channel),
            history_key: self.shared.keys.history(channel),
            spec,
            publisher: owner,
            me: self.shared.id.clone(),
            clock: self.shared.clock.clone(),
            view: self.shared.view.subscribe(),
            seen: self.shared.seen(channel),
            history_timeout: self.shared.history_timeout,
        };
        SampleSubscriber::declare(&self.shared.transport, ctx, options).await
    }

    /// A chunk sender for an actuation channel under `lease`, fenced to the
    /// view's current epoch.
    pub async fn chunk_publisher(&self, lease: &Lease) -> Result<ChunkPublisher, SessionError> {
        let spec = self.channel(lease.channel)?;
        let owner = spec
            .owner
            .clone()
            .ok_or_else(|| SessionError::WrongChannel {
                channel: lease.channel,
                reason: "it has no owner".into(),
            })?;
        if spec.direction != Direction::Actuate {
            return Err(SessionError::WrongChannel {
                channel: lease.channel,
                reason: "not an actuation channel".into(),
            });
        }
        let key = self.shared.keys.actions(&owner, lease.channel);
        let epoch = Epoch(self.shared.epoch.load(Ordering::Relaxed));
        ChunkPublisher::declare(
            &self.shared.transport,
            key,
            spec,
            self.shared.id.clone(),
            lease,
            epoch,
            self.shared.clock.clone(),
        )
        .await
    }

    /// The report an edge on this participant updates; heartbeats carry it.
    pub fn edge_report(&self) -> Arc<EdgeReport> {
        self.shared.edge.clone()
    }

    /// Tells the host that an edge fell back on its own account (queue dry
    /// past `hold_max`, a lease lapsed by its clock), so the session goes
    /// SAFE. Never blocks: a background task sends `EnterSafe`, retrying
    /// every 100 ms until the host answers, the view leaves ACTIVE, or the
    /// epoch moves on.
    pub fn report_safe(&self, epoch: Epoch, reason: impl Into<String>) {
        let report = SafeReport {
            epoch,
            reason: reason.into(),
        };
        // The reporter is gone only after close.
        let _ = self.shared.reports.send(report);
    }

    /// The health this participant reports in its heartbeats.
    pub fn set_health(&self, health: Health) {
        self.shared
            .health
            .store(health_code(health), Ordering::Relaxed);
    }

    /// Fault injection for tests and demos: stop (or resume) heartbeats while
    /// staying present, as a hung controller would.
    pub fn set_heartbeats_enabled(&self, enabled: bool) {
        self.shared
            .heartbeats_enabled
            .store(enabled, Ordering::Relaxed);
    }

    /// Detaches through `ctl` (releasing leases and removing owned
    /// channels), then closes.
    pub async fn detach(self) -> Result<(), SessionError> {
        let result = self.shared.ctl(Command::Detach).await.map(|_| ());
        self.close().await;
        result
    }

    /// Stops heartbeats and withdraws presence without detaching: the host
    /// sees liveliness loss (leases lapse, channels stay for a reconnect).
    pub async fn close(mut self) {
        self.cancel.cancel();
        for task in self.tasks.drain(..) {
            if let Err(err) = task.await
                && !err.is_cancelled()
            {
                warn!(participant = %self.shared.id, %err, "participant task failed");
            }
        }
    }
}

impl Drop for Participant {
    fn drop(&mut self) {
        self.cancel.cancel();
        for task in &self.tasks {
            task.abort();
        }
    }
}

fn health_code(health: Health) -> u8 {
    match health {
        Health::Unknown => 0,
        Health::Ok => 1,
        Health::Degraded => 2,
        Health::Unhealthy => 3,
    }
}

fn health_of(code: u8) -> Health {
    match code {
        1 => Health::Ok,
        2 => Health::Degraded,
        3 => Health::Unhealthy,
        _ => Health::Unknown,
    }
}

async fn heartbeat_loop(
    shared: Arc<Shared>,
    publisher: zenoh::pubsub::Publisher<'static>,
    period: Duration,
    cancel: CancellationToken,
) {
    let mut interval = tokio::time::interval(period);
    interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let mut seq = 0_u64;
    loop {
        tokio::select! {
            () = cancel.cancelled() => break,
            _ = interval.tick() => {}
        }
        if !shared.heartbeats_enabled.load(Ordering::Relaxed) {
            continue;
        }
        seq += 1;
        let (epoch, mode) = shared.epoch_and_mode();
        let previous = shared
            .completed
            .lock()
            .ok()
            .and_then(|mut slot| slot.take());
        let last_seq_seen = shared
            .seen
            .lock()
            .map(|seen| {
                seen.iter()
                    .map(|(channel, seq)| (*channel, seq.load(Ordering::Relaxed)))
                    .collect()
            })
            .unwrap_or_default();
        let (last_chunk_accepted, last_chunk_applied, queue_depth) = shared.edge.get();
        let heartbeat = Heartbeat {
            participant: shared.id.clone(),
            epoch: Epoch(epoch),
            mode,
            seq,
            t1_ns: shared.clock.now_ns(),
            previous,
            last_seq_seen,
            last_chunk_accepted,
            last_chunk_applied,
            queue_depth,
            health: health_of(shared.health.load(Ordering::Relaxed)),
        };
        if let Err(err) = publisher.put(codec::encode_heartbeat(&heartbeat)).await {
            warn!(participant = %shared.id, %err, "heartbeat failed");
        }
    }
}

/// How often an unanswered `EnterSafe` is resent.
const SAFE_RETRY: Duration = Duration::from_millis(100);

/// Sends each edge's `EnterSafe` to the host until the host answers, the
/// view already shows the session out of ACTIVE in that epoch, or the epoch
/// moves on. The edge has fallen back already; this only tells the host.
async fn report_safe(
    shared: Arc<Shared>,
    mut reports: mpsc::UnboundedReceiver<SafeReport>,
    cancel: CancellationToken,
) {
    loop {
        let report = tokio::select! {
            () = cancel.cancelled() => return,
            report = reports.recv() => match report {
                Some(report) => report,
                None => return,
            },
        };
        loop {
            let (epoch, mode) = shared.epoch_and_mode();
            if epoch != report.epoch.0 || mode != Mode::Active {
                break;
            }
            match shared
                .ctl(Command::EnterSafe {
                    reason: report.reason.clone(),
                })
                .await
            {
                Ok(_) | Err(SessionError::Rejected(_)) => break,
                Err(err) => {
                    warn!(participant = %shared.id, %err, "EnterSafe unanswered; retrying");
                    tokio::select! {
                        () = cancel.cancelled() => return,
                        () = tokio::time::sleep(SAFE_RETRY) => {}
                    }
                }
            }
        }
    }
}

/// Waits until the session host's liveliness token is visible.
pub async fn wait_for_host(
    transport: &Transport,
    keys: &SessionKeys,
    timeout: Duration,
) -> Result<(), SessionError> {
    let host = transport
        .control()
        .liveliness()
        .declare_subscriber(keys.host_liveliness())
        .history(true)
        .await?;
    let present = async {
        while let Ok(sample) = host.recv_async().await {
            if sample.kind() == SampleKind::Put {
                return Ok(());
            }
        }
        Err(SessionError::Closed)
    };
    tokio::time::timeout(timeout, present)
        .await
        .map_err(|_| SessionError::Timeout(timeout, "the session host".into()))?
}

/// Queries a session host for its authority snapshot, without attaching: how
/// a late-joining observer (or anyone) gets state at once.
pub async fn query_authority(
    transport: &Transport,
    keys: &SessionKeys,
    timeout: Duration,
) -> Result<AuthoritySnapshot, SessionError> {
    let replies = transport
        .control()
        .get(keys.host_state())
        .consolidation(ConsolidationMode::None)
        .priority(CONTROL_PRIORITY)
        .timeout(timeout)
        .await?;
    let reply = replies
        .recv_async()
        .await
        .map_err(|_| SessionError::NoReply(timeout))?;
    let sample = reply.result().map_err(|_| SessionError::NoReply(timeout))?;
    Ok(codec::decode_state(&sample.payload().to_bytes())?)
}
