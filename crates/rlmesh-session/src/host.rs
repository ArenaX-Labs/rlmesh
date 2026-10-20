//! The session host: the single writer of authority state.
//!
//! One task owns the [`Authority`] and serialises everything that touches it:
//! `ctl` queries, heartbeats, liveliness changes and the lease-expiry tick.
//! Heartbeat echoes are sent from the Zenoh callback itself, off the task, so
//! the host's hold time (t3 − t2) stays small and queueing cannot inflate the
//! measured round trip. Their publishers are declared when a participant
//! attaches, so the echo path never resolves a key per message.
//!
//! The host also keeps the session's **history store**: for every declared
//! sense or aux channel it subscribes to the samples and keeps the newest
//! `history_depth` frames, served on `history/{channel}` so a late joiner or a
//! swapped-in controller warms its windows from the session instead of from
//! the publisher.
//!
//! The host is a library ([`SessionHost::start`] on any [`Transport`]) and a
//! binary (`rlmesh-session-host`).

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use rlmesh_wire::codec;
use rlmesh_wire::{
    AuthoritySnapshot, ChannelId, ChannelPriority, CtlError, CtlReply, CtlStatus, Direction, Epoch,
    HeartbeatEcho, ParticipantId, SessionId, SessionKeys, TenantId,
};
use tokio::sync::{broadcast, mpsc, watch};
use tokio::task::JoinHandle;
use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};
use zenoh::Wait;
use zenoh::bytes::ZBytes;
use zenoh::query::Query;
use zenoh::sample::SampleKind;

use crate::authority::{Applied, Authority, AuthorityConfig, AuthorityEvent};
use crate::clock::{Clock, MonotonicClock, OffsetEstimator, OffsetEstimatorConfig};
use crate::error::SessionError;
use crate::metrics::{LinkStats, ParticipantHealth, Report};
use crate::transport::{
    CONTROL_CONGESTION, CONTROL_PRIORITY, HEARTBEAT_CONGESTION, TrafficClass, Transport,
};

/// Heartbeats queued between the Zenoh callback and the host task. Beyond
/// this the host is overloaded and drops them (the echo was already sent).
const HEARTBEAT_QUEUE: usize = 4096;

/// What the history store keeps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistoryConfig {
    /// Keep windows at all.
    pub enabled: bool,
    /// Keep BULK channels (cameras) too. Off by default: a raw camera window
    /// is large, and BULK channels are latest-value by nature.
    pub bulk: bool,
    /// The deepest window kept, whatever a channel declares.
    pub max_depth: usize,
}

impl Default for HistoryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            bulk: false,
            max_depth: 1024,
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
    /// Capabilities this session supports.
    pub capabilities: Vec<String>,
    /// The history store.
    pub history: HistoryConfig,
}

impl HostConfig {
    /// Defaults for `session` in `tenant`: a 250 ms lease grace, a 10 ms
    /// expiry tick, a 500 ms republish and a 1 s health log.
    pub fn new(tenant: TenantId, session: SessionId) -> Self {
        Self {
            tenant,
            session,
            clock: MonotonicClock::new().shared(),
            lease_grace: Duration::from_millis(250),
            tick: Duration::from_millis(10),
            republish: Duration::from_millis(500),
            health_log: Some(Duration::from_secs(1)),
            offset: OffsetEstimatorConfig::default(),
            capabilities: rlmesh_wire::version::CAPABILITIES
                .iter()
                .map(ToString::to_string)
                .collect(),
            history: HistoryConfig::default(),
        }
    }

    fn validate(&self) -> Result<(), SessionError> {
        if self.lease_grace.is_zero() {
            return Err(SessionError::Config("lease_grace must be positive".into()));
        }
        if self.tick.is_zero() || self.republish.is_zero() {
            return Err(SessionError::Config(
                "tick and republish must be positive".into(),
            ));
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
            .field("tick", &self.tick)
            .field("republish", &self.republish)
            .field("health_log", &self.health_log)
            .field("offset", &self.offset)
            .field("capabilities", &self.capabilities)
            .field("history", &self.history)
            .finish_non_exhaustive()
    }
}

/// A running session host. Dropping it stops the host task; prefer
/// [`SessionHost::shutdown`], which waits for it.
pub struct SessionHost {
    keys: SessionKeys,
    snapshot: watch::Receiver<AuthoritySnapshot>,
    health: watch::Receiver<Vec<ParticipantHealth>>,
    events: broadcast::Sender<AuthorityEvent>,
    cancel: CancellationToken,
    task: Option<JoinHandle<()>>,
}

impl SessionHost {
    /// Declares the host's queryables, subscribers and liveliness token, and
    /// starts its task. The authority snapshot is queryable once this
    /// returns.
    pub async fn start(transport: Transport, config: HostConfig) -> Result<Self, SessionError> {
        config.validate()?;
        let keys = SessionKeys::new(&config.tenant, config.session);
        let authority = Authority::new(
            config.session,
            AuthorityConfig {
                lease_grace_ns: u64::try_from(config.lease_grace.as_nanos()).unwrap_or(u64::MAX),
                capabilities: config.capabilities.clone(),
            },
        );
        let epoch = Arc::new(AtomicU64::new(authority.epoch().0));
        let control = transport.control().clone();

        let (hb_tx, hb_rx) = mpsc::channel(HEARTBEAT_QUEUE);
        let echoes: Echoes = Arc::new(RwLock::new(HashMap::new()));
        let heartbeats = control
            .declare_subscriber(keys.heartbeats())
            .callback({
                let (keys, clock, epoch, echoes) = (
                    keys.clone(),
                    config.clock.clone(),
                    epoch.clone(),
                    echoes.clone(),
                );
                move |sample| {
                    let t2 = clock.now_ns();
                    on_heartbeat_sample(&keys, &*clock, &epoch, &echoes, &hb_tx, &sample, t2);
                }
            })
            .await?;
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
        let store = HistoryStore::default();
        let history = transport
            .session(TrafficClass::Data)
            .declare_queryable(format!("{}/history/*", keys.prefix()))
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

        let mut actor = HostActor {
            transport,
            echoes,
            last_publish_ns: 0,
            keys: keys.clone(),
            config,
            authority,
            links: BTreeMap::new(),
            epoch,
            publisher,
            snapshot_tx,
            health_tx,
            events: events.clone(),
            store,
        };
        actor.publish().await;
        info!(session = %actor.config.session, key = actor.keys.prefix(), "session host up");

        let task = tokio::spawn({
            let cancel = cancel.clone();
            async move {
                actor.run(cancel, hb_rx, ctl, state, presence).await;
                // Withdraw presence first, so participants stop talking to a
                // host that no longer answers, then the rest.
                drop(token);
                drop(heartbeats);
                drop(history);
                actor.store.clear();
            }
        });

        Ok(Self {
            keys,
            snapshot,
            health,
            events,
            cancel,
            task: Some(task),
        })
    }

    /// The session's keys.
    pub fn keys(&self) -> &SessionKeys {
        &self.keys
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

/// Echo publishers by participant, shared with the heartbeat callback.
type Echoes = Arc<RwLock<HashMap<ParticipantId, zenoh::pubsub::Publisher<'static>>>>;

type HeartbeatTx = mpsc::Sender<(ParticipantId, rlmesh_wire::Heartbeat, i64)>;

/// The heartbeat callback: echo at once (t3 as late as possible), then queue
/// the heartbeat for the host task.
fn on_heartbeat_sample(
    keys: &SessionKeys,
    clock: &dyn Clock,
    epoch: &AtomicU64,
    echoes: &Echoes,
    queue: &HeartbeatTx,
    sample: &zenoh::sample::Sample,
    t2: i64,
) {
    let Some(id) = keys.participant_of(sample.key_expr().as_str()) else {
        return;
    };
    let heartbeat = match codec::decode_heartbeat(&sample.payload().to_bytes()) {
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
    let sent = match echoes.read().ok().as_ref().and_then(|e| e.get(&id)) {
        Some(publisher) => {
            let echo = HeartbeatEcho {
                seq: heartbeat.seq,
                t1_ns: heartbeat.t1_ns,
                t2_ns: t2,
                t3_ns: clock.now_ns(),
                epoch: Epoch(epoch.load(Ordering::Relaxed)),
            };
            publisher.put(codec::encode_echo(&echo)).wait()
        }
        // Not attached yet: no echo; the first exchange completes after
        // attach.
        None => Ok(()),
    };
    if let Err(err) = sent {
        warn!(%id, %err, "heartbeat echo failed");
    }
    if queue.try_send((id, heartbeat, t2)).is_err() {
        warn!("host overloaded: heartbeat dropped");
    }
}

/// The newest frames of one channel.
struct Window {
    depth: usize,
    frames: Mutex<VecDeque<ZBytes>>,
}

struct Recorded {
    window: Arc<Window>,
    _subscriber: zenoh::pubsub::Subscriber<()>,
}

/// History windows by channel, shared with the history queryable.
#[derive(Clone, Default)]
struct HistoryStore {
    windows: Arc<RwLock<HashMap<ChannelId, Arc<Window>>>>,
}

impl HistoryStore {
    /// Replies to a `history/{channel}` query with the window, oldest first.
    fn serve(&self, keys: &SessionKeys, query: &Query) {
        let key = query.key_expr().as_str();
        let channel = key
            .strip_prefix(keys.prefix())
            .and_then(|rest| rest.strip_prefix("/history/"))
            .and_then(|id| id.parse::<u32>().ok())
            .map(ChannelId);
        let Some(channel) = channel else {
            return;
        };
        let window = self
            .windows
            .read()
            .ok()
            .and_then(|w| w.get(&channel).cloned());
        let frames: Vec<ZBytes> = window
            .and_then(|w| w.frames.lock().ok().map(|f| f.iter().cloned().collect()))
            .unwrap_or_default();
        for frame in frames {
            if let Err(err) = query.reply(key, frame).wait() {
                warn!(%channel, %err, "history reply failed");
                return;
            }
        }
    }

    fn insert(&self, channel: ChannelId, window: Arc<Window>) {
        if let Ok(mut windows) = self.windows.write() {
            windows.insert(channel, window);
        }
    }

    fn remove(&self, channel: ChannelId) {
        if let Ok(mut windows) = self.windows.write() {
            windows.remove(&channel);
        }
    }

    fn clear(&self) {
        if let Ok(mut windows) = self.windows.write() {
            windows.clear();
        }
    }
}

/// A participant's timing link, as the host measures it.
struct Link {
    estimator: OffsetEstimator,
    stats: LinkStats,
}

type Fifo<T> = zenoh::handlers::FifoChannelHandler<T>;

struct HostActor {
    transport: Transport,
    echoes: Echoes,
    /// Host clock at the last snapshot publish.
    last_publish_ns: i64,
    keys: SessionKeys,
    config: HostConfig,
    authority: Authority,
    links: BTreeMap<ParticipantId, Link>,
    /// Shared with the echo callback.
    epoch: Arc<AtomicU64>,
    publisher: zenoh::pubsub::Publisher<'static>,
    snapshot_tx: watch::Sender<AuthoritySnapshot>,
    health_tx: watch::Sender<Vec<ParticipantHealth>>,
    events: broadcast::Sender<AuthorityEvent>,
    store: HistoryStore,
}

impl HostActor {
    async fn run(
        &mut self,
        cancel: CancellationToken,
        mut heartbeats: mpsc::Receiver<(ParticipantId, rlmesh_wire::Heartbeat, i64)>,
        ctl: zenoh::query::Queryable<Fifo<Query>>,
        state: zenoh::query::Queryable<Fifo<Query>>,
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
        let mut recorded: HashMap<ChannelId, Recorded> = HashMap::new();

        loop {
            tokio::select! {
                () = cancel.cancelled() => break,
                query = ctl.recv_async() => match query {
                    Ok(query) => {
                        self.on_ctl(query).await;
                        self.sync_history(&mut recorded).await;
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
                sample = presence.recv_async() => match sample {
                    Ok(sample) => {
                        self.on_presence(&sample).await;
                        self.sync_history(&mut recorded).await;
                    }
                    Err(_) => break,
                },
                _ = tick.tick() => {
                    let events = self.authority.expire(self.now());
                    self.record(events).await;
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
                }
            }
        }
        debug!(session = %self.config.session, "session host stopped");
    }

    fn now(&self) -> i64 {
        self.config.clock.now_ns()
    }

    /// Keeps one history subscription per recordable declared channel.
    async fn sync_history(&self, recorded: &mut HashMap<ChannelId, Recorded>) {
        if !self.config.history.enabled {
            return;
        }
        let wanted: BTreeMap<ChannelId, (ParticipantId, ChannelPriority, usize)> = self
            .authority
            .channels()
            .filter(|c| c.direction != Direction::Actuate)
            .filter(|c| self.config.history.bulk || c.qos.priority != ChannelPriority::Bulk)
            .filter_map(|c| {
                let owner = c.owner.clone()?;
                let depth = c.qos.history.depth().min(self.config.history.max_depth);
                Some((c.id, (owner, c.qos.priority, depth)))
            })
            .collect();
        recorded.retain(|id, _| {
            let keep = wanted.contains_key(id);
            if !keep {
                self.store.remove(*id);
            }
            keep
        });
        for (id, (owner, priority, depth)) in wanted {
            if recorded.contains_key(&id) {
                continue;
            }
            let window = Arc::new(Window {
                depth: depth.max(1),
                frames: Mutex::new(VecDeque::with_capacity(depth.max(1))),
            });
            let subscriber = self
                .transport
                .for_priority(priority)
                .declare_subscriber(self.keys.samples(&owner, id))
                .callback({
                    let window = window.clone();
                    move |sample| {
                        if let Ok(mut frames) = window.frames.lock() {
                            if frames.len() >= window.depth {
                                frames.pop_front();
                            }
                            frames.push_back(sample.payload().clone());
                        }
                    }
                })
                .await;
            match subscriber {
                Ok(subscriber) => {
                    debug!(channel = %id, depth, "history window");
                    self.store.insert(id, window.clone());
                    recorded.insert(
                        id,
                        Recorded {
                            window,
                            _subscriber: subscriber,
                        },
                    );
                }
                Err(err) => warn!(channel = %id, %err, "history subscription failed"),
            }
        }
        debug_assert!(recorded.values().all(|r| r.window.depth > 0));
    }

    async fn on_ctl(&mut self, query: Query) {
        let now = self.now();
        let request = query
            .payload()
            .ok_or_else(|| CtlError::new(CtlStatus::InvalidRequest, "ctl query without a payload"))
            .and_then(|payload| {
                codec::decode_request(&payload.to_bytes())
                    .map_err(|err| CtlError::new(CtlStatus::InvalidRequest, err))
            });
        let result = match request {
            Ok(request) => {
                debug!(from = %request.from, command = ?request.command, "ctl");
                self.authority.apply(request, now)
            }
            Err(err) => Err(err),
        };
        let (result, lease) = match result {
            Ok(Applied { lease, events }) => {
                self.record(events).await;
                (Ok(()), lease)
            }
            Err(err) => {
                info!(status = ?err.status, "ctl refused: {}", err.message);
                (Err(err), None)
            }
        };
        let reply = CtlReply {
            result,
            state: self.snapshot(),
            lease,
        };
        Self::reply(query, codec::encode_reply(&reply)).await;
    }

    async fn on_state_query(&self, query: Query) {
        Self::reply(query, codec::encode_state(&self.snapshot())).await;
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

    fn on_heartbeat(&mut self, id: ParticipantId, heartbeat: rlmesh_wire::Heartbeat, t2: i64) {
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

    async fn on_presence(&mut self, sample: &zenoh::sample::Sample) {
        let Some(id) = self.keys.participant_of(sample.key_expr().as_str()) else {
            return;
        };
        match sample.kind() {
            SampleKind::Put => debug!(%id, "participant present"),
            SampleKind::Delete => {
                info!(%id, "participant liveliness lost");
                let events = self.authority.participant_lost(&id);
                self.record(events).await;
            }
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
                    self.declare_echo(participant).await;
                    let offset = self.config.offset;
                    self.links
                        .entry(participant.clone())
                        .or_insert_with(|| Link {
                            estimator: OffsetEstimator::new(offset),
                            stats: LinkStats::new(offset.window),
                        });
                }
                AuthorityEvent::Detached { participant, .. } => {
                    self.links.remove(participant);
                    if let Ok(mut echoes) = self.echoes.write() {
                        echoes.remove(participant);
                    }
                }
                _ => {}
            }
            // Having no subscriber is fine.
            let _ = self.events.send(event);
        }
        self.epoch
            .store(self.authority.epoch().0, Ordering::Relaxed);
        self.publish().await;
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

    async fn declare_echo(&self, id: &ParticipantId) {
        if self.echoes.read().is_ok_and(|e| e.contains_key(id)) {
            return;
        }
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
                if let Ok(mut echoes) = self.echoes.write() {
                    echoes.insert(id.clone(), publisher);
                }
            }
            Err(err) => warn!(%id, %err, "echo publisher"),
        }
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
        let bytes = codec::encode_state(&snapshot);
        self.snapshot_tx.send_replace(snapshot);
        if let Err(err) = self.publisher.put(bytes).await {
            warn!(%err, "authority publish failed");
        }
    }
}
