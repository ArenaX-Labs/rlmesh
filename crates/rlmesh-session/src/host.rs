//! The session host: the single writer of authority state.
//!
//! One task owns the [`Authority`] and serialises everything that touches it:
//! `ctl` queries, heartbeats, liveliness changes and the lease-expiry tick.
//! Heartbeat echoes are sent from the Zenoh callback itself, off the task, so
//! the host's hold time (t3 − t2) stays small and queueing cannot inflate the
//! measured round trip. Their publishers are declared when a participant
//! attaches, so the echo path never resolves a key per message.
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
//! gaps the store knows of.
//!
//! In a virtual-clock session the host also runs the tick barrier (§11.3)
//! and publishes its ticks on `host/tick`.
//!
//! The host is a library ([`SessionHost::start`] on any [`Transport`]) and a
//! binary (`rlmesh-session-host`).

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use crate::wire::codec;
use crate::wire::part::{MAX_BATCH, MAX_CHUNK_BYTES};
use crate::wire::{
    Applied, AuthoritySnapshot, ChannelId, ChannelPriority, CtlError, CtlReply, CtlStatus, Epoch,
    HeartbeatEcho, HistoryQuery, ParticipantId, SampleBatch, SampleMeta, SeqRange, SessionId,
    SessionKeys, StreamSpec, TenantId,
};
use bytes::Bytes;
use tokio::sync::{broadcast, mpsc, watch};
use tokio::task::JoinHandle;
use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};
use zenoh::Wait;
use zenoh::query::Query;
use zenoh::sample::SampleKind;

use crate::authority::{Applied as Accepted, Authority, AuthorityConfig, AuthorityEvent};
use crate::clock::{Clock, MonotonicClock, OffsetEstimator, OffsetEstimatorConfig};
use crate::error::SessionError;
use crate::metrics::{LinkStats, ParticipantHealth, Report};
use crate::sample::decode_data;
use crate::transport::{
    CONTROL_CONGESTION, CONTROL_PRIORITY, HEARTBEAT_CONGESTION, TrafficClass, Transport,
    frame_zbytes,
};
use crate::virtual_clock::VirtualConfig;

/// Heartbeats queued between the Zenoh callback and the host task. Beyond
/// this the host is overloaded and drops them (the echo was already sent).
const HEARTBEAT_QUEUE: usize = 4096;
/// The most gap ranges a stream's history remembers.
const MAX_GAPS: usize = 64;

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
}

impl HostConfig {
    /// Defaults for `session` in `tenant`: wall mode, a 250 ms lease grace,
    /// a 10 ms expiry tick, a 500 ms republish and a 1 s health log.
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
            supports: crate::wire::version::FEATURES
                .iter()
                .map(ToString::to_string)
                .collect(),
            history: HistoryConfig::default(),
            virtual_clock: None,
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
        if self.virtual_clock.is_some_and(|v| v.tick_ns == 0) {
            return Err(SessionError::Config("tick_ns must be positive".into()));
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
            .field("supports", &self.supports)
            .field("history", &self.history)
            .field("virtual_clock", &self.virtual_clock)
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
                supports: config.supports.clone(),
                virtual_clock: config.virtual_clock,
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
        let ticks = control
            .declare_publisher(keys.host_tick())
            .priority(CONTROL_PRIORITY)
            .congestion_control(CONTROL_CONGESTION)
            .express(true)
            .await?;
        let store = HistoryStore {
            depth: config.history.depth.max(1),
            clock: Some(config.clock.clone()),
            rings: Arc::default(),
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
            ticks,
            snapshot_tx,
            health_tx,
            events: events.clone(),
            store,
            recorded: HashMap::new(),
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
                actor.recorded.clear();
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

type HeartbeatTx = mpsc::Sender<(ParticipantId, crate::wire::Heartbeat, i64)>;

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
}

impl Ring {
    /// Records a live sample.
    fn record(&self, payload: &zenoh::bytes::ZBytes, depth: usize) {
        if let Ok(sample) = decode_data(payload) {
            self.record_one(sample, depth);
        }
    }

    fn record_one(&self, sample: crate::wire::Sample, depth: usize) {
        if sample.channel != self.spec.channel || self.spec.check_payload(&sample.payload).is_err()
        {
            return;
        }
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        if sample.epoch > state.epoch {
            // A new epoch never seeds from an older one.
            *state = RingState {
                epoch: sample.epoch,
                first_seq: sample.seq,
                samples: VecDeque::with_capacity(depth.min(1024)),
                gaps: VecDeque::new(),
            };
        } else if sample.epoch < state.epoch {
            return;
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
}

impl HistoryStore {
    /// Replies to an `h/{robot}/{path}` query with its pages.
    fn serve(&self, keys: &SessionKeys, query: &Query) {
        let key = query.key_expr().as_str();
        let Some((robot, path)) = keys.history_of(key) else {
            return;
        };
        let ring = self
            .rings
            .read()
            .ok()
            .and_then(|r| r.get(&(robot, path.to_string())).cloned());
        let Some(ring) = ring else {
            return;
        };
        let request = match query.payload() {
            Some(payload) => match codec::decode_history_query(&payload.to_bytes()) {
                Ok(request) => request,
                Err(err) => {
                    warn!(%key, %err, "undecodable history query");
                    return;
                }
            },
            None => HistoryQuery::default(),
        };
        let now = self.clock.as_ref().map_or(0, |c| c.now_ns());
        for page in ring.pages(&request, self.depth, now) {
            let frame = match codec::encode_batch(&page) {
                Ok(encoded) => frame_zbytes(encoded),
                Err(err) => {
                    warn!(%key, %err, "history page");
                    return;
                }
            };
            if let Err(err) = query.reply(key, frame).wait() {
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
    ticks: zenoh::pubsub::Publisher<'static>,
    snapshot_tx: watch::Sender<AuthoritySnapshot>,
    health_tx: watch::Sender<Vec<ParticipantHealth>>,
    events: broadcast::Sender<AuthorityEvent>,
    store: HistoryStore,
    recorded: HashMap<StreamKey, Recorded>,
}

impl HostActor {
    async fn run(
        &mut self,
        cancel: CancellationToken,
        mut heartbeats: mpsc::Receiver<(ParticipantId, crate::wire::Heartbeat, i64)>,
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

        loop {
            tokio::select! {
                () = cancel.cancelled() => break,
                query = ctl.recv_async() => match query {
                    Ok(query) => {
                        self.on_ctl(query).await;
                        self.sync_history().await;
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
                        self.sync_history().await;
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
            });
            let depth = self.store.depth;
            let subscriber = self
                .transport
                .for_priority(spec.priority)
                .declare_subscriber(self.keys.data(&spec.owner, &spec.path))
                .callback({
                    let ring = ring.clone();
                    move |sample| ring.record(sample.payload(), depth)
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
        let (result, leases) = match result {
            Ok(Accepted { leases, events }) => {
                self.record(events).await;
                (Ok(()), leases)
            }
            Err(err) => {
                info!(status = ?err.status, "ctl refused: {}", err.message);
                (Err(err), Vec::new())
            }
        };
        // Ticks go out before the reply: a `Ready` that completed a step
        // returns after the step is published.
        self.publish_ticks().await;
        let reply = CtlReply {
            result,
            state: self.snapshot(),
            leases,
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

    async fn on_presence(&mut self, sample: &zenoh::sample::Sample) {
        let Some(id) = self.keys.participant_of(sample.key_expr().as_str()) else {
            return;
        };
        match sample.kind() {
            SampleKind::Put => debug!(%id, "participant present"),
            SampleKind::Delete => {
                info!(%id, "participant liveliness lost");
                let events = self.authority.participant_lost(&id, self.now());
                self.record(events).await;
                self.publish_ticks().await;
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

    async fn publish_ticks(&mut self) {
        for tick in self.authority.take_ticks() {
            if let Err(err) = self.ticks.put(codec::encode_tick(&tick)).await {
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
        }
    }

    fn put(ring: &Ring, epoch: u64, seq: u64, depth: usize) {
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
        ring.record(&frame_zbytes(codec::encode_sample(&sample)), depth);
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
            put(&ring, 2, seq, 6);
        }
        put(&ring, 1, 11, 6);
        let query = |after, until, limit| HistoryQuery {
            epoch: Epoch(2),
            after_seq: after,
            until_seq: until,
            limit,
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
        // A new epoch empties the ring; it never seeds from the old one.
        put(&ring, 3, 1, 6);
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
        let ring = ring(vec![FieldSpec::new("x", crate::wire::DType::U64, &[])]);
        for seq in 1..=600 {
            put(&ring, 1, seq, 1024);
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
