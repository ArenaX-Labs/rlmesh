//! A participant: presence, heartbeats, the authority view and `ctl` verbs.
//!
//! [`Participant::attach`] announces presence (a liveliness token), waits
//! for the host, subscribes to authority updates, runs the handshake with
//! its [`Credential`] (design §8.3), attaches through `ctl` and starts
//! heartbeating on the control class. Every verb and heartbeat it sends is
//! sealed under its `k_up`, every reply and echo it takes is opened under
//! its `k_down`, and every `host/state` and `host/tick` it takes is
//! verified against the host's signing key: anything else is dropped.
//! Verbs carry rising request ids, so a retry is answered once.
//!
//! **Keys** (design §8.4, §8.5): the host delivers the participant's other
//! keys on `p/{id}/keys` (subscribed before the handshake), and a grant's
//! reply carries the holder's. The participant installs them in its link
//! and acknowledges each delivery (`KeysAck`); a message under a key it
//! doesn't hold makes it ask for every key of its generation (`GetKeys`).
//! Its chunk publishers seal under the lease's `k_cmd`, its sample
//! publishers under the stream's `k_stream`, and its subscribers, edges and
//! history queries open under them ([`Sealer`]).
//!
//! From then on it keeps the newest authority snapshot (from updates and
//! from every `ctl` reply) and hands out endpoints addressed by robot and
//! path: sample publishers and subscribers, chunk publishers, history
//! queries, the virtual-mode tick stream, and the hooks an edge runtime
//! needs ([`AuthorityView`], [`EdgeReport`], [`Participant::report_safe`],
//! the local e-stop latch).
//!
//! A robot process tells the host about its own stops through **one ordered
//! queue** (DESIGN §6): `EnterSafe`, a local `Estop` and its `Clear` are
//! sent in the order they happened, each retried until the host answers, so
//! a host that comes back after a partition sees `Estop` before `Clear` and
//! ends in ESTOP (unlatched), which still needs disarm and arm.
//!
//! An embodiment's publishers follow their streams' publication plans
//! (§8.2): a background task installs each `Demand` the host publishes on
//! the stream's publishers and acknowledges its revision (`DemandAck`), so
//! controllers bound to the stream can be granted leases.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use crate::spec::{BindError, Binding, Policy, RunOptions, Target};
use crate::wire::codec;
use crate::wire::version::{PROTOCOL_MINOR, SPEC_VERSION, VIEW_DEVICE};
use crate::wire::{
    ActionChunk, AuthoritySnapshot, ChannelId, ChunkRef, ClockMode, Command, CtlError, CtlReply,
    CtlStatus, EdgeMode, Epoch, EstopSignal, Fingerprint, Health, Heartbeat, HeartbeatEcho,
    HistoryQuery, KeysAck, Lease, LeaseToken, LockstepRequest, LockstepResponse, Mode, PartSpec,
    ParticipantId, PriorityClass, Reason, ReasonCode, Role, Sample, SampleBatch, SessionId,
    SessionKeys, StreamSpec, TenantId, TimingSample, ViewReply, VirtualTick, Watermark,
};
use rlmesh_session_core::auth::{self, Credential, KeyError, StreamWindows};
use rlmesh_session_core::auth::{Link, LinkError, ReplyOutcome};
use rlmesh_session_core::crypto::SignedReceiver;
use rlmesh_session_core::crypto::{sealed, signed};
use tokio::sync::{Notify, mpsc, watch};
use tokio::task::JoinHandle;
use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};
use zenoh::query::ConsolidationMode;
use zenoh::sample::SampleKind;

use crate::chunk::ChunkPublisher;
use crate::clock::{Clock, MonotonicClock, OffsetEstimate};
use crate::error::SessionError;
use crate::host::OsRng;
use crate::media::ImagePublisher;
use crate::report::{ReportInputs, RunSource, report_loop};
use crate::sample::{
    Outlet, SamplePublisher, SampleSubscriber, SubscribeContext, SubscribeOptions, fetch_history,
};
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
    /// Features this participant implements.
    pub supports: Vec<String>,
    /// Features it cannot run without (the attach is refused when the
    /// session lacks one).
    pub requires: Vec<String>,
    /// The workflow edition it declares (empty: none).
    pub workflow_edition: String,
    /// Every workflow edition it runs.
    pub workflow_editions: Vec<String>,
    /// An embodiment's e-stop latch carried from its previous session
    /// ([`Participant::local_estop`]): it attaches latched, and the session
    /// starts in ESTOP (design §6).
    pub estop: Option<Reason>,
    /// What it attaches with: its device's key and the host's static key
    /// (`SessionHost::credential` in-process; a session address's key
    /// otherwise). Required.
    pub credential: Option<Credential>,
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
            supports: Vec::new(),
            requires: Vec::new(),
            workflow_edition: String::new(),
            workflow_editions: Vec::new(),
            estop: None,
            credential: None,
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

/// A robot process's local e-stop latch: an atomic flag, and how many
/// times it was latched, so an edge that missed a latch cleared before its
/// next look still stops (and fences) for it.
#[derive(Debug, Default)]
pub(crate) struct LocalLatch {
    latched: AtomicBool,
    latches: AtomicU64,
    /// Why it latched last: `ESTOP_LOCAL` (its own stop) or
    /// `ESTOP_SIGNAL` (a member's signal), and by whom.
    reason: Mutex<Reason>,
}

impl LocalLatch {
    /// Latches it for `reason`: which latch this is. The reason, then the
    /// count, then the flag, so whoever sees the flag also sees both.
    fn latch(&self, reason: Reason) -> u64 {
        let mut last = lock(&self.reason);
        *last = reason;
        let seq = self.latches.fetch_add(1, Ordering::SeqCst) + 1;
        self.latched.store(true, Ordering::SeqCst);
        seq
    }

    /// Clears it: how many latches came before, and why the last one
    /// latched.
    fn clear(&self) -> (u64, Reason) {
        let last = lock(&self.reason);
        self.latched.store(false, Ordering::SeqCst);
        (self.latches.load(Ordering::SeqCst), last.clone())
    }

    /// Whether it is latched now.
    pub(crate) fn is_latched(&self) -> bool {
        self.latched.load(Ordering::SeqCst)
    }

    /// How many times it was latched.
    pub(crate) fn latches(&self) -> u64 {
        self.latches.load(Ordering::SeqCst)
    }

    /// The code of the last latch.
    pub(crate) fn code(&self) -> ReasonCode {
        lock(&self.reason).code
    }

    /// Why it is latched, while it is.
    fn reason(&self) -> Option<Reason> {
        self.is_latched().then(|| lock(&self.reason).clone())
    }
}

/// What a robot process tells the host about its own stops, in order.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Report {
    /// It fell back on its own account in `epoch`: the session goes SAFE.
    Safe {
        epoch: Epoch,
        code: ReasonCode,
        reason: String,
    },
    /// It latched an e-stop, its `seq`th: its own (`ESTOP_LOCAL`), or a
    /// member's signal (`ESTOP_SIGNAL`, from `origin`).
    Estop {
        reason: String,
        seq: u64,
        code: ReasonCode,
        origin: Option<ParticipantId>,
        /// When it latched, on this process's clock.
        at_ns: i64,
    },
    /// It cleared the latch, after `after` latches, the last for
    /// `reason`.
    Clear { after: u64, reason: Reason },
}

impl Report {
    /// The report of latch `seq`, for `reason`.
    fn estop(reason: &Reason, seq: u64) -> Self {
        let origin = (reason.code == ReasonCode::EstopSignal)
            .then(|| ParticipantId::new(&reason.by).ok())
            .flatten();
        Self::Estop {
            reason: reason.detail.clone(),
            seq,
            code: reason.code,
            origin,
            at_ns: reason.at_ns,
        }
    }
}

/// Keeps a robot process's stop reports in DESIGN §6's order whatever order
/// its threads queued them in: a `Clear` goes out only after an `Estop` for
/// every latch before it, and an `Estop` a clear already covered goes
/// nowhere.
#[derive(Debug, Default)]
struct StopOrder {
    /// The latches an `Estop` went out for.
    estops: u64,
}

impl StopOrder {
    /// What `report` sends, in order.
    fn plan(&mut self, report: Report) -> Vec<Report> {
        match report {
            Report::Estop { seq, .. } if seq <= self.estops => Vec::new(),
            Report::Estop {
                reason,
                seq,
                code,
                origin,
                at_ns,
            } => {
                self.estops = seq;
                vec![Report::Estop {
                    reason,
                    seq,
                    code,
                    origin,
                    at_ns,
                }]
            }
            // The latch's own report hasn't gone out: the clear sends it,
            // with its reason.
            Report::Clear { after, reason } if after > self.estops => {
                self.estops = after;
                vec![
                    Report::estop(&reason, after),
                    Report::Clear { after, reason },
                ]
            }
            other => vec![other],
        }
    }
}

/// What the reporter does with one attempt's outcome.
#[derive(Debug, PartialEq, Eq)]
enum Attempt {
    /// The host answered (or refused it, authenticated): next report.
    Done,
    /// Send it again under the same request id.
    Retry,
    /// Its outcome is unknown (`EXPIRED`): send it again, anew.
    Anew,
}

impl Attempt {
    /// Judges `result`. A sealed refusal is the host's own and ends a
    /// report as an answer does (`EXPIRED` sends it anew). An unsealed one
    /// ([`SessionError::Unverified`]) is only a hint (anyone can answer
    /// `ctl`, and a superseded key gets one too): it never ends a report
    /// while this generation lasts (`ended`).
    fn of(result: &Result<(), SessionError>, ended: bool) -> Self {
        match result {
            Err(SessionError::Rejected(err)) if err.status == CtlStatus::Expired => Self::Anew,
            Ok(()) | Err(SessionError::Rejected(_)) => Self::Done,
            Err(SessionError::Unverified(_)) if ended => Self::Done,
            Err(_) => Self::Retry,
        }
    }
}

/// A handle that tells the host an edge fell back on its own account, for
/// code off the async path. Cheap to clone; see
/// [`Participant::report_safe`].
#[derive(Clone, Debug)]
pub struct SafeReporter {
    tx: mpsc::UnboundedSender<Report>,
}

impl SafeReporter {
    /// Queues an `EnterSafe` for a fault in `epoch`; never blocks.
    pub fn report(&self, epoch: Epoch, code: ReasonCode, reason: impl Into<String>) {
        // The reporter task is gone only after the participant closed.
        let _ = self.tx.send(Report::Safe {
            epoch,
            code,
            reason: reason.into(),
        });
    }
}

/// A lock that a panicked holder left behind is still the state to use.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Work for the key task: acknowledge a delivery, recover missing keys,
/// or keep retrying a request whose caller timed out (its reply may carry
/// keys: a grant's).
#[derive(Debug)]
enum KeyWork {
    Ack(KeysAck),
    Missing,
    Unanswered(u64, Command, watch::Sender<Option<CtlReply>>),
}

/// A request the host did not answer before its call timed out (§8.6): its
/// outcome is unknown. The participant keeps sending it under its id, which
/// the host answers from its reply cache without applying the verb twice,
/// until the answer comes or this generation ends.
#[derive(Clone, Debug)]
pub struct Pending {
    id: u64,
    reply: watch::Receiver<Option<CtlReply>>,
}

impl Pending {
    /// The request's id.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// The host's answer (a refusal stays in it, keys are installed), once
    /// it comes; [`SessionError::Closed`] when this generation ended first.
    pub async fn reply(mut self) -> Result<CtlReply, SessionError> {
        let reply = self.reply.wait_for(Option::is_some).await;
        reply
            .ok()
            .and_then(|reply| reply.clone())
            .ok_or(SessionError::Closed)
    }
}

/// The link, once the handshake completed.
type SharedLink = Arc<std::sync::OnceLock<Mutex<Link>>>;

/// A participant's delivered keys, for its publishers, subscribers, edges
/// and lockstep links: seals and opens under what its link holds (§8.4).
/// A message under a key it doesn't hold makes the participant recover its
/// keys (`GetKeys`). Cheap to clone.
#[derive(Clone)]
pub struct Sealer {
    link: SharedLink,
    work: mpsc::UnboundedSender<KeyWork>,
}

impl std::fmt::Debug for Sealer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sealer").finish_non_exhaustive()
    }
}

impl Sealer {
    /// A sealer with no link: everything is refused.
    #[cfg(test)]
    pub(crate) fn detached() -> Self {
        Self {
            link: Arc::default(),
            work: mpsc::unbounded_channel().0,
        }
    }

    fn with<R>(
        &self,
        f: impl FnOnce(&mut Link) -> Result<R, LinkError>,
    ) -> Result<R, SessionError> {
        let link = self.link.get().ok_or(SessionError::Closed)?;
        let result = f(&mut lock(link));
        result.map_err(|err| {
            if matches!(err, LinkError::Key(KeyError::Unknown(_))) {
                let _ = self.work.send(KeyWork::Missing);
            }
            SessionError::Key(err.to_string())
        })
    }

    pub(crate) fn seal_chunk(
        &self,
        keyexpr: &str,
        robot: &ParticipantId,
        chunk: &ActionChunk,
    ) -> Result<Vec<u8>, SessionError> {
        self.with(|link| link.seal_chunk(keyexpr, robot, chunk))
    }

    pub(crate) fn open_chunk(
        &self,
        keyexpr: &str,
        msg: Vec<u8>,
    ) -> Result<ActionChunk, SessionError> {
        self.with(|link| link.open_chunk(keyexpr, msg))
    }

    /// Seals one of this robot's samples (an `_applied` entry when
    /// `applied`) on its data key `keyexpr`.
    ///
    /// # Errors
    /// [`SessionError::Key`] before the stream's key arrived.
    pub fn seal_sample(
        &self,
        keyexpr: &str,
        sample: &Sample,
        applied: bool,
    ) -> Result<Vec<u8>, SessionError> {
        // Only the seq is taken under the link's lock: a camera frame is
        // sealed without holding up the process's chunks.
        let sealing =
            self.with(|link| Ok(link.keyring_mut().begin_sample(sample.channel, applied)?))?;
        let body = codec::encode_sample(sample).to_vec();
        let mut out = Vec::with_capacity(sealed::OVERHEAD + body.len());
        sealing
            .finish_to(keyexpr, &body, &mut out)
            .map_err(|err| SessionError::Key(err.to_string()))?;
        Ok(out)
    }

    /// Opens a sample of `robot`'s stream `channel` received on `keyexpr`
    /// for one subscription (its replay `windows`), verifying it outside
    /// the link's lock: the sample, its encoded body, and whether it is the
    /// first under a robot's newer stream key (its `seq`s restarted).
    pub(crate) fn open_stream(
        &self,
        keyexpr: &str,
        mut msg: Vec<u8>,
        robot: &ParticipantId,
        channel: ChannelId,
        applied: bool,
        windows: &Mutex<StreamWindows>,
    ) -> Result<(Sample, bytes::Bytes, bool), SessionError> {
        let (opening, version) = self.with(|link| {
            Ok(link
                .keyring()
                .begin_open_stream(&msg, robot, channel, applied, &lock(windows))?)
        })?;
        let key = |err: String| SessionError::Key(err);
        opening
            .finish(keyexpr, &mut msg)
            .map_err(|err| key(err.to_string()))?;
        let restarted = lock(windows)
            .commit(opening.header(), version)
            .map_err(|err| key(err.to_string()))?;
        self.with(|link| Ok(link.keyring_mut().opened(opening.header())?))?;
        let end = msg.len() - sealed::TAG_LEN;
        let body = bytes::Bytes::from(msg).slice(sealed::HEADER_LEN..end);
        let sample = codec::decode_sample(body.clone())?;
        if sample.channel != channel {
            return Err(key(format!("a sample of {} on {channel}", sample.channel)));
        }
        Ok((sample, body, restarted))
    }

    pub(crate) fn history_query(
        &self,
        keyexpr: &str,
        query: HistoryQuery,
    ) -> Result<(u64, Vec<u8>), SessionError> {
        self.with(|link| link.history_query(keyexpr, query))
    }

    pub(crate) fn history_page(
        &self,
        history_key: &str,
        data_key: &str,
        msg: Vec<u8>,
        request_id: u64,
        robot: &ParticipantId,
    ) -> Result<SampleBatch, SessionError> {
        self.with(|link| link.history_page(history_key, data_key, msg, request_id, robot))
    }

    /// Seals a lockstep request to `robot` under token `token` on
    /// `keyexpr` (`l/{robot}`).
    ///
    /// # Errors
    /// [`SessionError::Key`] without the lease's keys.
    pub fn seal_lockstep_request(
        &self,
        keyexpr: &str,
        robot: &ParticipantId,
        token: LeaseToken,
        request: &LockstepRequest,
    ) -> Result<Vec<u8>, SessionError> {
        self.with(|link| link.seal_lockstep_request(keyexpr, robot, token, request))
    }

    /// Opens a lockstep request to this robot: it and its key's token.
    ///
    /// # Errors
    /// [`SessionError::Key`]: it is dropped.
    pub fn open_lockstep_request(
        &self,
        keyexpr: &str,
        msg: Vec<u8>,
    ) -> Result<(LockstepRequest, LeaseToken), SessionError> {
        self.with(|link| link.open_lockstep_request(keyexpr, msg))
    }

    /// Seals this robot's response under token `token`'s `k_obs`.
    ///
    /// # Errors
    /// [`SessionError::Key`] without it.
    pub fn seal_lockstep_response(
        &self,
        keyexpr: &str,
        token: LeaseToken,
        response: &LockstepResponse,
    ) -> Result<Vec<u8>, SessionError> {
        self.with(|link| link.seal_lockstep_response(keyexpr, token, response))
    }

    /// Opens `robot`'s response under token `token`'s `k_obs`.
    ///
    /// # Errors
    /// [`SessionError::Key`]: it is dropped.
    pub fn open_lockstep_response(
        &self,
        keyexpr: &str,
        msg: Vec<u8>,
        robot: &ParticipantId,
        token: LeaseToken,
    ) -> Result<LockstepResponse, SessionError> {
        self.with(|link| link.open_lockstep_response(keyexpr, msg, robot, token))
    }
}

struct Shared {
    transport: Transport,
    /// A robot that asked for its device view (`view.device`): the host
    /// answers its verbs with the view, not the state.
    takes_view: bool,
    /// This process's link to the host (its keys, request ids and the
    /// host's broadcast floor), set once the handshake completes. One lock:
    /// a seq is allocated and its message sealed under it.
    link: SharedLink,
    /// The key task's queue.
    key_work: mpsc::UnboundedSender<KeyWork>,
    /// Bumped on every key delivery, for waiters on a key.
    keys_changed: watch::Sender<u64>,
    /// The heartbeat publisher, once attached.
    heartbeat: std::sync::OnceLock<zenoh::pubsub::Publisher<'static>>,
    /// Where the host's echoes go while this participant relays another
    /// clock's heartbeats ([`Participant::relay_heartbeats`]).
    relay: Mutex<Option<mpsc::UnboundedSender<HeartbeatEcho>>>,
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
    /// This participant's clock when the host's liveliness went away
    /// (`i64::MIN` while it is present).
    host_lost_ns: AtomicI64,
    /// The last completed exchange, sent with the next heartbeat.
    completed: Mutex<Option<TimingSample>>,
    /// The newest seq received per subscribed channel.
    seen: Mutex<BTreeMap<ChannelId, Arc<AtomicU64>>>,
    edge: Arc<EdgeReport>,
    /// This robot process's stops, sent in order by the reporter task.
    reports: mpsc::UnboundedSender<Report>,
    /// The local e-stop latch every edge of this process reads each tick.
    estop: Arc<LocalLatch>,
    /// This controller's binding revisions.
    binding_revision: AtomicU32,
    /// This embodiment's publishers by stream, for installing demands.
    outlets: Mutex<BTreeMap<ChannelId, Vec<Weak<Outlet>>>>,
    /// Wakes the demand installer: a publisher was declared.
    demand_wake: Notify,
    /// A bound controller's run counters, for its reports.
    run: Arc<Mutex<Option<RunSource>>>,
}

impl Shared {
    /// Keeps the newest snapshot.
    fn observe(&self, snapshot: AuthoritySnapshot) {
        if let Some(link) = self.link.get() {
            lock(link).observe(&snapshot, self.clock.now_ns());
        }
        let id = &self.id;
        self.view.send_if_modified(|view| {
            if view
                .as_ref()
                .is_some_and(|old| !snapshot.is_newer_than(old))
            {
                return false;
            }
            if let Some(old) = view.as_ref()
                && old.mode != snapshot.mode
            {
                debug!(participant = %id, "mode {} -> {}", old.mode, snapshot.mode);
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

    /// Latches this robot process's e-stop for `reason`, and queues its
    /// report to the host.
    fn latch(&self, reason: Reason) {
        let seq = self.estop.latch(reason.clone());
        let _ = self.reports.send(Report::estop(&reason, seq));
    }

    /// A member's e-stop signal on `key`: latched at once (`ESTOP_SIGNAL`,
    /// `by` its sender), when its `k_estop` verifies it.
    fn on_estop_signal(&self, key: &str, payload: &zenoh::bytes::ZBytes) {
        let Some(link) = self.link.get() else {
            return;
        };
        let opened = lock(link).open_estop_signal(key, payload.to_bytes().into_owned());
        match opened {
            Ok((by, signal)) => {
                warn!(participant = %self.id, %by, reason = %signal.reason, "e-stop signal latched");
                self.latch(Reason {
                    code: ReasonCode::EstopSignal,
                    by: by.to_string(),
                    detail: signal.reason,
                    at_ns: self.clock.now_ns(),
                    ..Reason::default()
                });
            }
            Err(err) => debug!(participant = %self.id, %err, "e-stop signal dropped"),
        }
    }

    /// Whether this participant's generation ended: the view lists a newer
    /// one under its id, or none at all.
    fn generation_ended(&self) -> bool {
        let Some(link) = self.link.get() else {
            return true;
        };
        let mine = lock(link).accept().generation;
        self.view.borrow().as_ref().is_some_and(|view| {
            view.participant(&self.id)
                .is_none_or(|p| p.generation != mine)
        })
    }

    fn epoch_and_mode(&self) -> (u64, Mode) {
        self.view
            .borrow()
            .as_ref()
            .map_or((0, Mode::Disarmed), |s| (s.epoch.0, s.mode))
    }

    fn clock_mode(&self) -> ClockMode {
        self.view
            .borrow()
            .as_ref()
            .map_or(ClockMode::Wall, |s| s.clock_mode)
    }

    /// Sends one verb and waits for the host's reply. Every reply carries the
    /// host's snapshot, which updates this participant's view even when the
    /// verb was refused.
    async fn ctl(&self, command: Command) -> Result<(AuthoritySnapshot, Vec<Lease>), SessionError> {
        let CtlReply {
            result,
            state,
            leases,
            ..
        } = self.ctl_reply(command).await?;
        result?;
        Ok((state, leases))
    }

    /// [`Self::ctl`], with a refusal left in the reply. A request left
    /// unanswered is retried in the background under its id until the
    /// host's answer (its cached reply, keys included) arrives or this
    /// generation ends (§8.6); a caller that timed out gets
    /// [`SessionError::Pending`], whose handle resolves with that answer.
    async fn ctl_reply(&self, command: Command) -> Result<CtlReply, SessionError> {
        let (id, reply) = self.call(None, &command).await;
        let Some(id) = id.filter(|&id| {
            reply.is_err()
                && self
                    .link
                    .get()
                    .is_some_and(|link| lock(link).unanswered(id))
        }) else {
            return reply;
        };
        let (answer, pending) = watch::channel(None);
        let _ = self.key_work.send(KeyWork::Unanswered(id, command, answer));
        match reply {
            Err(SessionError::NoReply(timeout)) => Err(SessionError::Pending {
                timeout,
                pending: Pending { id, reply: pending },
            }),
            other => other,
        }
    }

    /// Sends `command` as a new request (`None`) or again as request `id`
    /// (a retry, which the host answers from its reply cache, never
    /// applying the verb twice, §8.6), and takes the first reply that
    /// answers it. An unsealed refusal is only a hint (anyone can answer
    /// `ctl`): it is returned, as [`SessionError::Unverified`], when no
    /// sealed reply came. Returns the id.
    async fn call(
        &self,
        request_id: Option<u64>,
        command: &Command,
    ) -> (Option<u64>, Result<CtlReply, SessionError>) {
        let Some(link) = self.link.get() else {
            return (request_id, Err(SessionError::Closed));
        };
        let (epoch, _) = self.epoch_and_mode();
        let epoch = (epoch > 0).then_some(Epoch(epoch));
        let key = self.keys.ctl();
        let sealed = {
            let mut link = lock(link);
            match request_id {
                Some(id) => link
                    .resend(&key, id, epoch, command)
                    .map(|sealed| (id, sealed)),
                None => link.request(&key, epoch, command),
            }
        };
        match sealed {
            Ok((id, sealed)) => (Some(id), self.exchange(link, &key, id, sealed).await),
            Err(err) => (
                request_id,
                Err(SessionError::Config(format!("ctl request: {err}"))),
            ),
        }
    }

    async fn exchange(
        &self,
        link: &Mutex<Link>,
        key: &str,
        request_id: u64,
        sealed: Vec<u8>,
    ) -> Result<CtlReply, SessionError> {
        let timeout = self.ctl_timeout;
        let replies = self
            .transport
            .control()
            .get(key)
            .payload(sealed)
            .consolidation(ConsolidationMode::None)
            .priority(CONTROL_PRIORITY)
            .congestion_control(CONTROL_CONGESTION)
            .express(true)
            .timeout(timeout)
            .await?;
        let mut hint = None;
        while let Ok(reply) = replies.recv_async().await {
            let Ok(sample) = reply.result() else {
                continue;
            };
            let mut bytes = sample.payload().to_bytes().into_owned();
            if self.takes_view {
                let outcome =
                    lock(link).reply_view(key, &mut bytes, request_id, self.clock.now_ns());
                match outcome {
                    Ok(ReplyOutcome::Answer(reply)) => {
                        // The reply carries the view, not the state: read
                        // the state the verb left.
                        let state = self.fetch_state().await?;
                        let ViewReply {
                            result,
                            leases,
                            request_id,
                            keys,
                            assignment,
                            run,
                            ..
                        } = *reply;
                        return Ok(CtlReply {
                            result,
                            state,
                            leases,
                            request_id,
                            keys,
                            assignment,
                            run,
                        });
                    }
                    Ok(ReplyOutcome::Hint(refusal)) => hint = Some(refusal),
                    Ok(ReplyOutcome::Other(_)) => {}
                    Err(err) => debug!(participant = %self.id, %err, "ctl reply dropped"),
                }
                continue;
            }
            let outcome = lock(link).reply(key, &mut bytes, request_id, self.clock.now_ns());
            match outcome {
                Ok(ReplyOutcome::Answer(reply)) => {
                    self.observe(reply.state.clone());
                    return Ok(*reply);
                }
                Ok(ReplyOutcome::Hint(refusal)) => hint = Some(refusal),
                Ok(ReplyOutcome::Other(_)) => {}
                Err(err) => debug!(participant = %self.id, %err, "ctl reply dropped"),
            }
        }
        Err(hint.map_or(SessionError::NoReply(timeout), SessionError::Unverified))
    }

    fn sealer(&self) -> Sealer {
        Sealer {
            link: self.link.clone(),
            work: self.key_work.clone(),
        }
    }

    /// The host's current snapshot, queried and verified (a newer
    /// publication that overtook it stands).
    async fn fetch_state(&self) -> Result<AuthoritySnapshot, SessionError> {
        let key = self.keys.host_state();
        let signed = ask(self.transport.control(), &key, None, self.ctl_timeout).await?;
        if let Some(snapshot) = self.verify_state(&key, &signed) {
            self.observe(snapshot);
        }
        self.view
            .borrow()
            .clone()
            .ok_or(SessionError::NoReply(self.ctl_timeout))
    }

    /// Verifies a `host/state` broadcast, received on `key`.
    fn verify_state(&self, key: &str, msg: &[u8]) -> Option<AuthoritySnapshot> {
        lock(self.link.get()?)
            .state(key, msg)
            .map_err(|err| debug!(participant = %self.id, %err, "authority update dropped"))
            .ok()
    }
}

/// What a participant without the address's key is told.
const NO_CREDENTIAL: &str = "no credential for this session: join with the address its host \
                             printed (it ends in #key), or set ParticipantConfig.credential";

/// Runs the handshake (§8.3): reads the host instance from the public
/// snapshot (a hint: a wrong one only fails the handshake), sends msg 1 on
/// `hs` and reads msg 2.
async fn handshake(
    control: &zenoh::Session,
    keys: &SessionKeys,
    config: &ParticipantConfig,
) -> Result<auth::Attached, SessionError> {
    let credential = config.credential.as_ref().ok_or_else(|| {
        SessionError::Rejected(CtlError::new(CtlStatus::Unauthenticated, NO_CREDENTIAL))
    })?;
    let unauthenticated =
        |why: String| SessionError::Rejected(CtlError::new(CtlStatus::Unauthenticated, why));
    let timeout = config.ctl_timeout;
    let state = ask(control, &keys.host_state(), None, timeout).await?;
    let host_instance = signed::peek(&state)
        .map_err(|err| unauthenticated(format!("the host's state is not signed ({err})")))?
        .header
        .host_instance;
    let (initiator, hello) = credential
        .start(
            &mut OsRng,
            config.tenant.as_str(),
            &config.session.to_string(),
            host_instance,
            &config.id,
        )
        .map_err(|err| unauthenticated(err.to_string()))?;
    let reply = ask(
        control,
        &keys.handshake(),
        Some(codec::encode_handshake_request(&hello)),
        timeout,
    )
    .await?;
    let msg2 = codec::decode_handshake_reply(&reply)??;
    auth::finish(initiator, &msg2).map_err(|err| {
        unauthenticated(format!(
            "the host's handshake reply failed ({err}): the address's key is not this host's"
        ))
    })
}

/// The first reply to a query on `key`.
async fn ask(
    control: &zenoh::Session,
    key: &str,
    payload: Option<Vec<u8>>,
    timeout: Duration,
) -> Result<Vec<u8>, SessionError> {
    let mut get = control
        .get(key)
        .consolidation(ConsolidationMode::None)
        .priority(CONTROL_PRIORITY)
        .congestion_control(CONTROL_CONGESTION)
        .express(true)
        .timeout(timeout);
    if let Some(payload) = payload {
        get = get.payload(payload);
    }
    let replies = get.await?;
    while let Ok(reply) = replies.recv_async().await {
        if let Ok(sample) = reply.result() {
            return Ok(sample.payload().to_bytes().into_owned());
        }
    }
    Err(SessionError::NoReply(timeout))
}

/// An attached participant. Dropping it stops heartbeats and withdraws its
/// liveliness token, which the host treats as the participant leaving; prefer
/// [`Participant::detach`] or [`Participant::close`].
pub struct Participant {
    shared: Arc<Shared>,
    cancel: CancellationToken,
    tasks: Vec<JoinHandle<()>>,
    // Dropped in order: the watch on the host goes before this token, whose
    // loss can stop the session (and with it the host's token).
    _host: zenoh::pubsub::Subscriber<()>,
    /// A robot process's e-stop signal subscription.
    _estops: Option<zenoh::pubsub::Subscriber<()>>,
    _liveliness: zenoh::liveliness::LivelinessToken,
    _state: zenoh::pubsub::Subscriber<()>,
    _echo: zenoh::pubsub::Subscriber<()>,
    _keys: zenoh::pubsub::Subscriber<()>,
}

/// Whether `transport` reaches another Zenoh session within `timeout`: at
/// once for one sharing a session, or riding a host of this process.
async fn linked(transport: &Transport, timeout: Duration) -> bool {
    let control = transport.control();
    if transport.is_shared() || transport.is_in_process() {
        return true;
    }
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if control.info().peers_zid().await.next().is_some()
            || control.info().routers_zid().await.next().is_some()
        {
            return true;
        }
        if tokio::time::Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Whether a live participant holds the liveliness token `key`.
async fn live(control: &zenoh::Session, key: String) -> bool {
    match control
        .liveliness()
        .get(key)
        .timeout(Duration::from_millis(500))
        .await
    {
        Ok(replies) => {
            let mut any = false;
            while let Ok(reply) = replies.recv_async().await {
                any |= reply.result().is_ok();
            }
            any
        }
        Err(_) => false,
    }
}

/// Resolves when the host this transport reaches can never serve
/// `session`: it hosts other sessions, and is no multi-session host of
/// `tenant` (which would start `session` for this participant). Pending
/// while no host answers at all, so a host still starting is waited for.
async fn not_served(
    control: &zenoh::Session,
    tenant: &TenantId,
    session: SessionId,
) -> SessionError {
    use crate::wire::keys::{session_hosts, session_server};
    // Whether any queryable answers `key` (queries reach every matching
    // queryable, where liveliness tokens reach only declared interests).
    let answers = |key: String| async move {
        match control.get(key).timeout(Duration::from_millis(500)).await {
            Ok(replies) => {
                let mut any = false;
                while let Ok(reply) = replies.recv_async().await {
                    any |= reply.result().is_ok();
                }
                any
            }
            Err(_) => false,
        }
    };
    let own = SessionKeys::new(tenant, session).host_state();
    loop {
        // Give a multi-session host time to see this participant first.
        tokio::time::sleep(Duration::from_millis(300)).await;
        if answers(session_server(tenant)).await || answers(own.clone()).await {
            return std::future::pending().await;
        }
        if answers(session_hosts()).await {
            return SessionError::NotServed(format!(
                "it hosts other sessions and starts none for tenant {tenant}"
            ));
        }
    }
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
        let (key_work, key_rx) = mpsc::unbounded_channel();
        let (keys_changed, _) = watch::channel(0);
        let shared = Arc::new(Shared {
            transport: transport.clone(),
            takes_view: config.roles.contains(&Role::Embodiment)
                && config.supports.iter().any(|f| f == VIEW_DEVICE),
            link: Arc::default(),
            key_work,
            keys_changed,
            heartbeat: std::sync::OnceLock::new(),
            relay: Mutex::new(None),
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
            host_lost_ns: AtomicI64::new(i64::MIN),
            completed: Mutex::new(None),
            seen: Mutex::new(BTreeMap::new()),
            edge: Arc::new(EdgeReport::default()),
            reports,
            estop: Arc::default(),
            binding_revision: AtomicU32::new(0),
            outlets: Mutex::new(BTreeMap::new()),
            demand_wake: Notify::new(),
            run: Arc::default(),
        });

        // Subscriptions first (§8.5): nothing the host sends after the
        // handshake can be missed. Until it completes, nothing verifies.
        let state = control
            .declare_subscriber(keys.host_state())
            .callback({
                let (shared, key) = (shared.clone(), keys.host_state());
                move |sample| {
                    if let Some(snapshot) = shared.verify_state(&key, &sample.payload().to_bytes())
                    {
                        shared.observe(snapshot);
                    }
                }
            })
            .await?;
        let echo = control
            .declare_subscriber(keys.heartbeat_echo(&config.id))
            .callback({
                let (shared, clock) = (shared.clone(), config.clock.clone());
                let key = keys.heartbeat_echo(&config.id);
                move |sample| {
                    let t4 = clock.now_ns();
                    let Some(link) = shared.link.get() else {
                        return;
                    };
                    let mut bytes = sample.payload().to_bytes().into_owned();
                    let echo = lock(link).echo(&key, &mut bytes);
                    match echo {
                        Ok(echo) => match &*lock(&shared.relay) {
                            Some(relay) => {
                                let _ = relay.send(echo);
                            }
                            None => *lock(&shared.completed) = Some(echo.complete(t4)),
                        },
                        Err(err) => debug!(%err, "heartbeat echo dropped"),
                    }
                }
            })
            .await?;
        let deliveries = control
            .declare_subscriber(keys.participant_keys(&config.id))
            .callback({
                let shared = shared.clone();
                let key = keys.participant_keys(&config.id);
                move |sample| {
                    let Some(link) = shared.link.get() else {
                        return;
                    };
                    let mut bytes = sample.payload().to_bytes().into_owned();
                    let now = shared.clock.now_ns();
                    let ack = lock(link).keys(&key, &mut bytes, now);
                    match ack {
                        Ok(ack) => {
                            shared.keys_changed.send_modify(|n| *n += 1);
                            let _ = shared.key_work.send(KeyWork::Ack(ack));
                        }
                        Err(err) => debug!(%err, "key delivery dropped"),
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
                    let lost_ns = if present {
                        i64::MIN
                    } else {
                        warn!(participant = %shared.id, "session host liveliness lost");
                        shared.clock.now_ns()
                    };
                    shared.host_lost_ns.store(lost_ns, Ordering::Relaxed);
                    shared.host_present.send_replace(present);
                }
            })
            .await?;
        // Once linked, this transport sees every live participant's token:
        // another one under this id is refused, never shared.
        let key = keys.participant(&config.id);
        if linked(&transport, config.host_timeout).await && live(&control, key.clone()).await {
            return Err(SessionError::Config(format!(
                "participant {} is already in this session: serve under another name, or stop \
                 the one there first",
                config.id
            )));
        }
        // A multi-session host starts the session when it sees this token.
        let liveliness = control.liveliness().declare_token(key).await?;
        // The host declares its queryables before its liveliness token, and
        // Zenoh keeps declaration order per link, so once the token is seen
        // the first `ctl` query cannot miss the host.
        let mut present = shared.host_present.subscribe();
        tokio::select! {
            found = tokio::time::timeout(config.host_timeout, present.wait_for(|p| *p)) => {
                found
                    .map_err(|_| {
                        SessionError::Timeout(config.host_timeout, "the session host".into())
                    })?
                    .map_err(|_| SessionError::Closed)?;
            }
            refusal = not_served(&control, &config.tenant, config.session) => return Err(refusal),
        }

        let attached = handshake(&control, &keys, &config).await?;
        let generation = attached.accept.generation;
        let _ = shared
            .link
            .set(Mutex::new(Link::new(config.id.clone(), attached)));

        let cancel = CancellationToken::new();
        let mut participant = Self {
            shared: shared.clone(),
            tasks: vec![
                tokio::spawn(report_stops(shared.clone(), report_rx, cancel.clone())),
                tokio::spawn(install_demands(shared.clone(), cancel.clone())),
                tokio::spawn(report_streams(shared.clone(), cancel.clone())),
                tokio::spawn(work_keys(shared.clone(), key_rx, cancel.clone())),
            ],
            cancel,
            _liveliness: liveliness,
            _state: state,
            _echo: echo,
            _keys: deliveries,
            _host: host,
            _estops: None,
        };
        shared
            .ctl(Command::Attach {
                name: config.name.clone(),
                roles: config.roles.clone(),
                priority: config.priority,
                protocol_minor: PROTOCOL_MINOR,
                supports: config.supports.clone(),
                requires: config.requires.clone(),
                workflow_edition: config.workflow_edition.clone(),
                workflow_editions: config.workflow_editions.clone(),
                estop: config.estop.clone(),
            })
            .await?;
        // A robot process latches every member's e-stop signal at once,
        // with or without a host.
        if config.roles.contains(&Role::Embodiment) {
            if let Some(reason) = config.estop.clone() {
                shared.latch(reason);
            }
            let subscriber = control
                .declare_subscriber(keys.estops())
                .callback({
                    let shared = shared.clone();
                    move |sample| {
                        shared.on_estop_signal(sample.key_expr().as_str(), sample.payload())
                    }
                })
                .await?;
            participant._estops = Some(subscriber);
        }

        let publisher = control
            .declare_publisher(keys.heartbeat(&config.id))
            .priority(CONTROL_PRIORITY)
            .congestion_control(HEARTBEAT_CONGESTION)
            .express(true)
            .await?;
        let _ = shared.heartbeat.set(publisher);
        participant.tasks.push(tokio::spawn(heartbeat_loop(
            shared,
            config.heartbeat_period,
            participant.cancel.clone(),
        )));
        info!(participant = %config.id, session = %config.session, generation, "attached");
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

    /// `HOST_LOST` (at this participant's clock) once the host's liveliness
    /// went away after attaching; `None` while it is visible. The edge keeps
    /// running on its own clock meanwhile.
    pub fn host_lost(&self) -> Option<Reason> {
        let at_ns = self.shared.host_lost_ns.load(Ordering::Relaxed);
        (at_ns != i64::MIN).then(|| Reason {
            at_ns,
            ..Reason::new(ReasonCode::HostLost)
        })
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

    /// Queries the host for its current snapshot, verified. A reply that
    /// a newer publication overtook leaves the newer view.
    pub async fn fetch_authority(&self) -> Result<AuthoritySnapshot, SessionError> {
        let key = self.shared.keys.host_state();
        let timeout = self.shared.ctl_timeout;
        let signed = ask(self.shared.transport.control(), &key, None, timeout).await?;
        if let Some(snapshot) = self.shared.verify_state(&key, &signed) {
            self.shared.observe(snapshot);
        }
        self.authority().ok_or(SessionError::NoReply(timeout))
    }

    /// A verifier of the host's broadcasts (`host/state`, `host/tick`), for
    /// a subscriber of this process's own (a recorder, the gateway's relay):
    /// the host's key and instance from this participant's handshake, and
    /// the floor of its own state updates.
    pub fn host_verifier(&self) -> Option<SignedReceiver> {
        self.shared
            .link
            .get()
            .map(|link| lock(link).host_verifier())
    }

    /// Relays another clock's heartbeats (a gateway's browser): this
    /// participant's own heartbeats stop, [`Self::send_heartbeat`] seals and
    /// sends each relayed one, and the host's echoes of them, opened, go to
    /// `echoes`.
    pub fn relay_heartbeats(&self, echoes: mpsc::UnboundedSender<HeartbeatEcho>) {
        *lock(&self.shared.relay) = Some(echoes);
    }

    /// Seals and sends one heartbeat under this participant's keys (a
    /// relayed one: see [`Self::relay_heartbeats`]).
    pub async fn send_heartbeat(&self, heartbeat: &Heartbeat) -> Result<(), SessionError> {
        self.shared.send_heartbeat(heartbeat.clone(), false).await
    }

    /// DISARMED or SAFE to ARMED, in a fresh epoch.
    pub async fn arm(&self) -> Result<AuthoritySnapshot, SessionError> {
        Ok(self.shared.ctl(Command::Arm).await?.0)
    }

    /// Back to DISARMED.
    pub async fn disarm(&self) -> Result<AuthoritySnapshot, SessionError> {
        Ok(self.shared.ctl(Command::Disarm).await?.0)
    }

    /// Emergency stop (design §6): publishes this member's e-stop signal,
    /// which every robot edge latches directly (`ESTOP_SIGNAL`), and tells
    /// the host (`Estop`) in the background. Returns once the signal is
    /// out; a missing host reply is only a warning. The confirmation is
    /// each robot's `_applied` mode: [`Self::wait_for_estop`].
    ///
    /// # Errors
    /// [`SessionError::Key`] once this process is no longer attached (its
    /// key ended): only the robot's own stop is left.
    pub async fn estop(&self, reason: impl Into<String>) -> Result<(), SessionError> {
        let shared = &self.shared;
        let reason = reason.into();
        let key = shared.keys.estop(&shared.id);
        let signal = EstopSignal {
            reason: reason.clone(),
            sent_ns: shared.clock.now_ns(),
        };
        let link = shared.link.get().ok_or(SessionError::Closed)?;
        let sealed = lock(link).estop_signal(&key, &signal).map_err(|err| {
            SessionError::Key(format!(
                "this process is no longer attached ({err}): use the robot's stop"
            ))
        })?;
        shared
            .transport
            .control()
            .put(&key, sealed)
            .priority(CONTROL_PRIORITY)
            .congestion_control(CONTROL_CONGESTION)
            .express(true)
            .await?;
        let shared = shared.clone();
        tokio::spawn(async move {
            let estop = Command::Estop {
                reason,
                code: ReasonCode::EstopRemote,
                origin: None,
                at_ns: 0,
            };
            if let Err(err) = shared.ctl(estop).await {
                warn!(participant = %shared.id, %err, "the host did not answer the e-stop; the signal stands");
            }
        });
        Ok(())
    }

    /// Waits until every robot's edge applies ESTOP (its first commandable
    /// part's `_applied` mode): what [`Self::estop`] confirms by.
    ///
    /// # Errors
    /// [`SessionError::Timeout`]; a subscription's error.
    pub async fn wait_for_estop(&self, timeout: Duration) -> Result<(), SessionError> {
        let view = self.authority().ok_or(SessionError::Closed)?;
        let mut robots: Vec<&ParticipantId> = Vec::new();
        let mut applied = Vec::new();
        for part in view.parts.iter().filter(|p| p.command.is_some()) {
            if !robots.contains(&&part.owner) {
                robots.push(&part.owner);
                let options = SubscribeOptions::default();
                let path = part.path.as_str();
                applied.push(self.subscribe_applied(&part.owner, path, options).await?);
            }
        }
        let stopped = |s: &SampleSubscriber| {
            s.latest()
                .and_then(|f| f.received.sample.applied)
                .is_some_and(|a| a.mode == EdgeMode::Estop)
        };
        let started = tokio::time::Instant::now();
        while !applied.iter().all(stopped) {
            if started.elapsed() >= timeout {
                return Err(SessionError::Timeout(timeout, "every robot's ESTOP".into()));
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        Ok(())
    }

    /// Releases a latched e-stop: an embodiment's local clear.
    pub async fn clear(&self) -> Result<AuthoritySnapshot, SessionError> {
        Ok(self.shared.ctl(Command::Clear).await?.0)
    }

    /// Declares this embodiment's robot map (the authored JSON, §3): the
    /// host normalises it and assigns channel ids. Re-declaring identically
    /// is a no-op; the parts are part of the authority snapshot.
    ///
    /// This runtime can seal every part, so the session's choice of level
    /// stands. Returns once the host's keys for every part reached this
    /// process (or after `ctl_timeout`, with a warning): samples published
    /// before then are dropped unsent.
    pub async fn declare_parts(&self, robot_json: &str) -> Result<AuthoritySnapshot, SessionError> {
        let seal = crate::spec::normalise(&self.shared.id, SPEC_VERSION, robot_json)
            .map(|parts| parts.iter().map(|p| p.path.to_string()).collect())
            .unwrap_or_default();
        let snapshot = self
            .shared
            .ctl(Command::DeclareParts {
                spec_version: SPEC_VERSION,
                robot_json: robot_json.to_string(),
                seal,
                board_id: None,
            })
            .await?
            .0;
        let channels: Vec<ChannelId> = snapshot
            .parts
            .iter()
            .filter(|p| p.owner == self.shared.id)
            .map(|p| p.channel)
            .collect();
        let Some(link) = self.shared.link.get() else {
            return Ok(snapshot);
        };
        let ready = || {
            let link = lock(link);
            channels.iter().all(|c| link.keyring().sends_stream(*c))
        };
        let mut changed = self.shared.keys_changed.subscribe();
        let waited = tokio::time::timeout(self.shared.ctl_timeout, async {
            while !ready() {
                if changed.changed().await.is_err() {
                    break;
                }
            }
        })
        .await;
        if waited.is_err() {
            warn!(participant = %self.shared.id, "the host's stream keys have not arrived yet");
        }
        Ok(snapshot)
    }

    /// Declares an identity binding for this controller: the parts it will
    /// command, a table and a fingerprint, without delivery contracts or
    /// action grids (a controller that commands paths directly, §6.5). A new
    /// fingerprint re-issues this controller's leases under a fresh token.
    /// [`Self::declare_bound`] declares what the binder produced.
    pub async fn declare_binding(
        &self,
        fingerprint: Fingerprint,
        table_json: &str,
        channels: Vec<ChannelId>,
        replan_hz: f64,
    ) -> Result<AuthoritySnapshot, SessionError> {
        let revision = self.shared.binding_revision.fetch_add(1, Ordering::Relaxed) + 1;
        Ok(self
            .shared
            .ctl(Command::DeclareBinding {
                fingerprint,
                revision,
                table_json: table_json.to_string(),
                channels,
                replan_hz,
                contracts: Vec::new(),
                grids: Vec::new(),
            })
            .await?
            .0)
    }

    /// Binds `policy` against every part in this participant's view (§6.5),
    /// or its scene's view when it carries places and `options` names no
    /// robot (scene design §5): the binding table, or the refusal listing
    /// every problem.
    pub fn bind(&self, policy: &Policy, options: &RunOptions) -> Result<Binding, BindError> {
        let view = self.shared.view.borrow();
        match view.as_ref() {
            Some(state) => crate::spec::bind_in(policy, &Target::of(state), options),
            None => crate::spec::bind(policy, &[], options),
        }
    }

    /// Declares a binding the binder produced: its fingerprint, its table,
    /// the parts it commands, its delivery contracts and its action grids. A
    /// lease under it is granted once every stream it reads has installed a
    /// publication plan that includes its contract (`NOT_READY` until
    /// then).
    pub async fn declare_bound(
        &self,
        binding: &Binding,
    ) -> Result<AuthoritySnapshot, SessionError> {
        let revision = self.shared.binding_revision.fetch_add(1, Ordering::Relaxed) + 1;
        Ok(self
            .shared
            .ctl(Command::DeclareBinding {
                fingerprint: binding.fingerprint,
                revision,
                table_json: binding.table_text(),
                channels: binding.channels(),
                replan_hz: binding.replan_hz,
                contracts: binding.contracts.clone(),
                grids: binding.grids(),
            })
            .await?
            .0)
    }

    /// Requests (or renews) leases on commandable parts, all or none, under
    /// this controller's current binding, releasing `release` in the same
    /// revision (a switch within a command group). `ttl` defaults to the
    /// session's lease grace.
    pub async fn grant_leases(
        &self,
        channels: Vec<ChannelId>,
        release: Vec<ChannelId>,
        ttl: Option<Duration>,
    ) -> Result<Vec<Lease>, SessionError> {
        let binding = self.binding_fingerprint();
        let ttl_ns = ttl.map(|ttl| u64::try_from(ttl.as_nanos()).unwrap_or(u64::MAX));
        let (_, leases) = self
            .shared
            .ctl(Command::GrantLease {
                channels,
                release,
                binding,
                ttl_ns,
            })
            .await?;
        Ok(leases)
    }

    /// The fingerprint of this participant's declared binding (zero: none).
    pub fn binding_fingerprint(&self) -> Fingerprint {
        self.authority()
            .and_then(|view| view.binding(&self.shared.id).map(|b| b.fingerprint))
            .unwrap_or_default()
    }

    /// Sends one verb and returns the host's reply, a refusal included (as
    /// its `result`); the reply's snapshot updates the view either way.
    pub async fn command(&self, command: Command) -> Result<CtlReply, SessionError> {
        self.shared.ctl_reply(command).await
    }

    /// The host's estimate of this participant's clock offset, from the
    /// current view.
    pub fn clock_estimate(&self) -> Option<OffsetEstimate> {
        self.shared.view.borrow().as_ref().and_then(|view| {
            view.clock(&self.shared.id)
                .map(OffsetEstimate::from_mapping)
        })
    }

    /// [`Self::grant_leases`] for one part.
    pub async fn grant_lease(
        &self,
        channel: ChannelId,
        ttl: Option<Duration>,
    ) -> Result<Lease, SessionError> {
        self.grant_leases(vec![channel], Vec::new(), ttl)
            .await?
            .pop()
            .ok_or_else(|| {
                SessionError::Wire(crate::wire::WireError::Malformed {
                    message: "ctl reply",
                    reason: "a granted lease is missing".into(),
                })
            })
    }

    /// Gives a lease back.
    pub async fn release_lease(&self, lease: &Lease) -> Result<AuthoritySnapshot, SessionError> {
        self.release_leases(vec![lease.channel], lease.token).await
    }

    /// Gives leases held under one token back.
    pub async fn release_leases(
        &self,
        channels: Vec<ChannelId>,
        token: LeaseToken,
    ) -> Result<AuthoritySnapshot, SessionError> {
        Ok(self
            .shared
            .ctl(Command::ReleaseLease { channels, token })
            .await?
            .0)
    }

    /// A part of `robot`, from the current view.
    pub fn part(&self, robot: &ParticipantId, path: &str) -> Result<PartSpec, SessionError> {
        self.authority()
            .and_then(|view| view.part_at(robot, path).cloned())
            .ok_or_else(|| SessionError::UnknownPart(format!("{robot}/{path}")))
    }

    /// One of this embodiment's own parts.
    pub fn own_part(&self, path: &str) -> Result<PartSpec, SessionError> {
        self.part(&self.shared.id, path)
    }

    /// A part by its stream id, from the current view.
    pub fn part_by_id(&self, channel: ChannelId) -> Result<PartSpec, SessionError> {
        self.authority()
            .and_then(|view| view.part(channel).cloned())
            .ok_or_else(|| SessionError::UnknownPart(channel.to_string()))
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

    /// Seals and opens under this participant's delivered keys (a lockstep
    /// link's requests and replies).
    pub fn sealer(&self) -> Sealer {
        self.shared.sealer()
    }

    /// A publisher for one of this embodiment's own parts.
    pub async fn publisher(&self, path: &str) -> Result<SamplePublisher, SessionError> {
        let part = self.own_part(path)?;
        let stream = part.stream().map_err(|reason| SessionError::WrongChannel {
            channel: part.channel,
            reason,
        })?;
        self.stream_publisher(stream).await
    }

    /// A publisher for one of this embodiment's streams (a part's, or a
    /// commandable part's `_applied`).
    pub async fn stream_publisher(
        &self,
        stream: StreamSpec,
    ) -> Result<SamplePublisher, SessionError> {
        if stream.owner != self.shared.id {
            return Err(SessionError::WrongChannel {
                channel: stream.channel,
                reason: format!("published by {}, not {}", stream.owner, self.shared.id),
            });
        }
        let key = self.shared.keys.data(&stream.owner, &stream.path);
        let history_key = self.shared.keys.history(&stream.owner, &stream.path);
        // An applied stream has no `send` ceiling; a part's is its rate cap.
        let ceiling_hz = if stream.applied {
            None
        } else {
            self.shared
                .view
                .borrow()
                .as_ref()
                .and_then(|v| v.part(stream.channel))
                .and_then(|p| p.send_policy().hz)
        };
        let channel = stream.channel;
        let publisher = SamplePublisher::declare(
            &self.shared.transport,
            key,
            history_key,
            stream,
            self.shared.clock_mode(),
            ceiling_hz,
            self.shared.epoch.clone(),
            self.shared.clock.clone(),
            self.shared.sealer(),
        )
        .await?;
        if let Ok(mut outlets) = self.shared.outlets.lock() {
            outlets
                .entry(channel)
                .or_default()
                .push(Arc::downgrade(publisher.outlet()));
        }
        self.shared.demand_wake.notify_one();
        Ok(publisher)
    }

    /// An image publisher for one of this embodiment's image parts, with the
    /// bitrate budget its `send` declares.
    pub async fn image_publisher(&self, path: &str) -> Result<ImagePublisher, SessionError> {
        let part = self.own_part(path)?;
        let budget = part.send_policy().max_mbps.map(|mbps| (mbps * 1e6) as u64);
        ImagePublisher::new(self.publisher(path).await?, budget)
    }

    /// Subscribes to a part of `robot`.
    pub async fn subscribe(
        &self,
        robot: &ParticipantId,
        path: &str,
        options: SubscribeOptions,
    ) -> Result<SampleSubscriber, SessionError> {
        let part = self.part(robot, path)?;
        let stream = part.stream().map_err(|reason| SessionError::WrongChannel {
            channel: part.channel,
            reason,
        })?;
        self.subscribe_stream(stream, options).await
    }

    /// Subscribes to what the edge dispatched to a commandable part of
    /// `robot` (its `_applied` stream, §9.4).
    pub async fn subscribe_applied(
        &self,
        robot: &ParticipantId,
        path: &str,
        options: SubscribeOptions,
    ) -> Result<SampleSubscriber, SessionError> {
        let part = self.part(robot, path)?;
        let stream = part
            .applied_stream()
            .ok_or_else(|| SessionError::WrongChannel {
                channel: part.channel,
                reason: "not a commandable part".into(),
            })?
            .map_err(|reason| SessionError::WrongChannel {
                channel: part.channel,
                reason,
            })?;
        self.subscribe_stream(stream, options).await
    }

    /// Subscribes to a stream.
    pub async fn subscribe_stream(
        &self,
        stream: StreamSpec,
        options: SubscribeOptions,
    ) -> Result<SampleSubscriber, SessionError> {
        let ctx = SubscribeContext {
            key: self.shared.keys.data(&stream.owner, &stream.path),
            keys: self.shared.keys.clone(),
            seen: self.shared.seen(stream.channel),
            spec: stream,
            me: self.shared.id.clone(),
            clock: self.shared.clock.clone(),
            view: self.shared.view.subscribe(),
            history_timeout: self.shared.history_timeout,
            sealer: self.shared.sealer(),
        };
        SampleSubscriber::declare(&self.shared.transport, ctx, options).await
    }

    /// Queries the host's history store for a stream (attach, swap and
    /// recovery only, §8.8): every page of the reply, in seq order.
    pub async fn query_history(
        &self,
        stream: &StreamSpec,
        query: HistoryQuery,
    ) -> Result<Vec<SampleBatch>, SessionError> {
        fetch_history(
            &self.shared.transport,
            &self.shared.sealer(),
            stream,
            &self.shared.keys,
            query,
            self.shared.history_timeout,
        )
        .await
    }

    /// A chunk sender for a commandable part under `lease`, fenced to the
    /// view's current epoch.
    pub async fn chunk_publisher(&self, lease: &Lease) -> Result<ChunkPublisher, SessionError> {
        let part = self.part_by_id(lease.channel)?;
        let key = self.shared.keys.commands(&part.owner, part.path.as_str());
        let epoch = Epoch(self.shared.epoch.load(Ordering::Relaxed));
        ChunkPublisher::declare(
            &self.shared.transport,
            key,
            part,
            self.shared.id.clone(),
            lease,
            epoch,
            self.shared.clock_mode(),
            self.shared.clock.clone(),
            self.shared.sealer(),
        )
        .await
    }

    /// Virtual mode: this embodiment published tick `tick`'s samples, up to
    /// `watermarks`.
    pub async fn ready(&self, tick: u64, watermarks: Vec<Watermark>) -> Result<(), SessionError> {
        self.shared
            .ctl(Command::Ready { tick, watermarks })
            .await
            .map(|_| ())
    }

    /// Virtual mode: this controller's decision for tick `tick` (the chunks
    /// it sent for it; none is "no action").
    pub async fn decided(&self, tick: u64, chunks: Vec<ChunkRef>) -> Result<(), SessionError> {
        self.shared
            .ctl(Command::Decided { tick, chunks })
            .await
            .map(|_| ())
    }

    /// Virtual mode: the host's ticks from now on (`host/tick`), verified
    /// and in the host's (epoch, tick, phase) order: a tick that is not
    /// signed by the host, or not after the last one taken, is dropped.
    pub async fn virtual_ticks(&self) -> Result<TickStream, SessionError> {
        let (tx, rx) = mpsc::unbounded_channel();
        let verifier = Mutex::new(self.host_verifier().ok_or(SessionError::Closed)?);
        let key = self.shared.keys.host_tick();
        let last = Mutex::new(None);
        let subscriber = self
            .shared
            .transport
            .control()
            .declare_subscriber(key.clone())
            .callback(move |sample| {
                let bytes = sample.payload().to_bytes();
                let tick = match auth::open_tick(&mut lock(&verifier), &key, &bytes) {
                    Ok(tick) => tick,
                    Err(err) => {
                        debug!(%err, "tick dropped");
                        return;
                    }
                };
                let mut last = lock(&last);
                if last.is_none_or(|last| tick.order() > last) {
                    *last = Some(tick.order());
                    let _ = tx.send(tick);
                } else {
                    debug!(order = ?tick.order(), "a tick out of order dropped");
                }
            })
            .await?;
        Ok(TickStream {
            rx,
            _subscriber: subscriber,
        })
    }

    /// The report an edge on this participant updates; heartbeats carry it.
    pub fn edge_report(&self) -> Arc<EdgeReport> {
        self.shared.edge.clone()
    }

    /// Tells the host that this embodiment fell back on its own account
    /// (an edge's queue ran dry past `hold_max`, a lease lapsed by its
    /// clock, a device watchdog fired in `epoch`), so the session goes SAFE.
    /// Never blocks: a background task sends `EnterSafe`, retrying every
    /// 100 ms, until the host is SAFE or ESTOP at an epoch at or after
    /// `epoch`.
    pub fn report_safe(&self, epoch: Epoch, code: ReasonCode, reason: impl Into<String>) {
        self.safe_reporter().report(epoch, code, reason);
    }

    /// Latches this robot process's local e-stop (DESIGN §6): an atomic
    /// store, safe from any thread (a hardware button's handler too), that
    /// needs no network. Every edge of this process refuses chunks and runs
    /// Estop from its next tick; a background task reports
    /// `Estop{ESTOP_LOCAL}` to the host, retrying, in order with the other
    /// reports.
    pub fn latch_estop(&self, reason: impl Into<String>) {
        self.shared.latch(Reason {
            code: ReasonCode::EstopLocal,
            by: self.shared.id.to_string(),
            detail: reason.into(),
            at_ns: self.shared.clock.now_ns(),
            ..Reason::default()
        });
    }

    /// Why this robot process's e-stop is latched, while it is: what a
    /// process carries into its next session ([`ParticipantConfig::estop`]).
    pub fn local_estop(&self) -> Option<Reason> {
        self.shared.estop.reason()
    }

    /// Clears the local e-stop latch: the only way to, since no wire path
    /// clears it. `Clear` is queued behind any unsent `Estop`, so the host
    /// ends in ESTOP, unlatched, and leaving it still needs disarm and arm.
    /// It also clears a host latch this process did not set (the local
    /// clear of §11.1).
    pub fn clear_estop(&self) {
        let (after, reason) = self.shared.estop.clear();
        let _ = self.shared.reports.send(Report::Clear { after, reason });
    }

    /// Whether the local e-stop is latched.
    pub fn estop_latched(&self) -> bool {
        self.shared.estop.is_latched()
    }

    /// The latch, for this process's edges.
    pub(crate) fn estop_latch(&self) -> Arc<LocalLatch> {
        self.shared.estop.clone()
    }

    /// A handle for [`Participant::report_safe`] that outlives borrows.
    pub fn safe_reporter(&self) -> SafeReporter {
        SafeReporter {
            tx: self.shared.reports.clone(),
        }
    }

    /// Reports a bound controller's run counters to observers (§8.10,
    /// [`crate::report`]); the [`Controller`](crate::Controller) installs
    /// its own. `None` stops them.
    pub fn set_run_source(&self, source: Option<RunSource>) {
        if let Ok(mut run) = self.shared.run.lock() {
            *run = source;
        }
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

impl Shared {
    /// Seals `heartbeat` under `k_up` and publishes it.
    /// Seals and sends `heartbeat`. With `stamp`, its `t1` is taken
    /// under the link's lock, so time spent waiting for the lock (another
    /// task sealing or opening) never counts as network time in the clock
    /// mapping.
    async fn send_heartbeat(
        &self,
        mut heartbeat: Heartbeat,
        stamp: bool,
    ) -> Result<(), SessionError> {
        let (Some(link), Some(publisher)) = (self.link.get(), self.heartbeat.get()) else {
            return Err(SessionError::Closed);
        };
        let sealed = {
            let mut link = lock(link);
            if stamp {
                heartbeat.t1_ns = self.clock.now_ns();
            }
            link.heartbeat(publisher.key_expr().as_str(), &heartbeat)
        }
        .map_err(|err| SessionError::Config(format!("heartbeat: {err}")))?;
        Ok(publisher.put(sealed).await?)
    }
}

/// How often a participant missing keys asks for them again.
const GET_KEYS_PERIOD: Duration = Duration::from_millis(250);

/// Acknowledges every key delivery and, when a message came under a key
/// this participant doesn't hold, asks for every key of its generation
/// (`GetKeys`, at most every [`GET_KEYS_PERIOD`]): the reply installs them.
async fn work_keys(
    shared: Arc<Shared>,
    mut work: mpsc::UnboundedReceiver<KeyWork>,
    cancel: CancellationToken,
) {
    let mut last_recovery: Option<tokio::time::Instant> = None;
    loop {
        let item = tokio::select! {
            () = cancel.cancelled() => return,
            item = work.recv() => match item {
                Some(item) => item,
                None => return,
            },
        };
        let command = match item {
            KeyWork::Ack(ack) if !ack.is_empty() => Command::KeysAck(ack),
            KeyWork::Ack(_) => continue,
            KeyWork::Missing => {
                if last_recovery.is_some_and(|at| at.elapsed() < GET_KEYS_PERIOD) {
                    continue;
                }
                last_recovery = Some(tokio::time::Instant::now());
                Command::GetKeys
            }
            KeyWork::Unanswered(id, command, answer) => {
                tokio::spawn(retry_unanswered(
                    shared.clone(),
                    id,
                    command,
                    answer,
                    cancel.clone(),
                ));
                continue;
            }
        };
        match shared.ctl(command).await {
            Ok(_) => {
                shared.keys_changed.send_modify(|n| *n += 1);
            }
            Err(err) => debug!(participant = %shared.id, %err, "keys"),
        }
    }
}

/// Resends request `id` until it is answered, and hands the answer to its
/// [`Pending`] handle. Stops when this generation ends, or when another
/// attempt's reply answered it and no handle waits.
async fn retry_unanswered(
    shared: Arc<Shared>,
    id: u64,
    command: Command,
    answer: watch::Sender<Option<CtlReply>>,
    cancel: CancellationToken,
) {
    loop {
        tokio::select! {
            () = cancel.cancelled() => return,
            () = tokio::time::sleep(SAFE_RETRY) => {}
        }
        let open = shared
            .link
            .get()
            .is_some_and(|link| lock(link).unanswered(id));
        if (!open && answer.is_closed()) || shared.generation_ended() {
            return;
        }
        if let Ok(reply) = shared.call(Some(id), &command).await.1 {
            shared.keys_changed.send_modify(|n| *n += 1);
            answer.send_replace(Some(reply));
            return;
        }
    }
}

async fn heartbeat_loop(shared: Arc<Shared>, period: Duration, cancel: CancellationToken) {
    let mut interval = tokio::time::interval(period);
    interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let mut seq = 0_u64;
    loop {
        tokio::select! {
            () = cancel.cancelled() => break,
            _ = interval.tick() => {}
        }
        if !shared.heartbeats_enabled.load(Ordering::Relaxed) || lock(&shared.relay).is_some() {
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
        if let Err(err) = shared.send_heartbeat(heartbeat, true).await {
            warn!(participant = %shared.id, %err, "heartbeat failed");
        }
    }
}

/// Installs the host's demands on this embodiment's publishers and
/// acknowledges each revision once installed (§8.2), on every new view and
/// every newly declared publisher.
async fn install_demands(shared: Arc<Shared>, cancel: CancellationToken) {
    let mut view = shared.view.subscribe();
    loop {
        sync_demands(&shared).await;
        tokio::select! {
            () = cancel.cancelled() => return,
            changed = view.changed() => if changed.is_err() { return },
            () = shared.demand_wake.notified() => {}
        }
    }
}

async fn sync_demands(shared: &Shared) {
    let live: Vec<(ChannelId, Vec<Arc<Outlet>>)> = {
        let Ok(mut outlets) = shared.outlets.lock() else {
            return;
        };
        outlets.retain(|_, list| {
            list.retain(|o| o.strong_count() > 0);
            !list.is_empty()
        });
        outlets
            .iter()
            .map(|(c, list)| (*c, list.iter().filter_map(Weak::upgrade).collect()))
            .collect()
    };
    if live.is_empty() {
        return;
    }
    let acks: Vec<(ChannelId, u64)> = {
        let view = shared.view.borrow();
        let Some(view) = view.as_ref() else {
            return;
        };
        live.iter()
            .filter_map(|(channel, outlets)| {
                let demand = view.demand(*channel);
                let (revision, plan) = demand.map_or((0, None), |d| (d.revision, Some(d.plan)));
                for outlet in outlets {
                    outlet.install(revision, plan);
                }
                demand
                    .filter(|d| {
                        d.acked < d.revision && outlets.iter().all(|o| o.revision() == d.revision)
                    })
                    .map(|d| (*channel, d.revision))
            })
            .collect()
    };
    for (channel, revision) in acks {
        match shared.ctl(Command::DemandAck { channel, revision }).await {
            Ok(_) => debug!(participant = %shared.id, %channel, revision, "demand installed"),
            Err(err) => warn!(participant = %shared.id, %channel, %err, "demand ack failed"),
        }
    }
}

/// Publishes this participant's observer report (§8.10): its publishers'
/// counters and a bound controller's run counters.
async fn report_streams(shared: Arc<Shared>, cancel: CancellationToken) {
    let inputs = ReportInputs {
        transport: shared.transport.clone(),
        key: shared.keys.report(&shared.id),
        id: shared.id.clone(),
        clock: shared.clock.clone(),
        epoch: shared.epoch.clone(),
        run: shared.run.clone(),
    };
    let streams = move || {
        let Ok(outlets) = shared.outlets.lock() else {
            return Vec::new();
        };
        outlets
            .values()
            .filter_map(|list| list.iter().find_map(Weak::upgrade))
            .map(|outlet| outlet.counters())
            .collect()
    };
    report_loop(inputs, streams, cancel).await;
}

/// How often an unanswered stop report is resent.
const SAFE_RETRY: Duration = Duration::from_millis(100);

/// Sends this robot process's stops to the host one at a time, in order,
/// each until the host answers (or refuses it). An `EnterSafe` also stops
/// once the view shows the host SAFE or ESTOP at an epoch at or after the
/// fault's (a re-armed host goes SAFE again: the fault may be a device's).
/// The edges have stopped already; this only tells the host.
async fn report_stops(
    shared: Arc<Shared>,
    mut reports: mpsc::UnboundedReceiver<Report>,
    cancel: CancellationToken,
) {
    let mut order = StopOrder::default();
    loop {
        let report = tokio::select! {
            () = cancel.cancelled() => return,
            report = reports.recv() => match report {
                Some(report) => report,
                None => return,
            },
        };
        for report in order.plan(report) {
            if !send_stop(&shared, &report, &cancel).await {
                return;
            }
        }
    }
}

/// Sends one stop report until it is done ([`Attempt`]); `false` once the
/// participant closed.
async fn send_stop(shared: &Shared, report: &Report, cancel: &CancellationToken) -> bool {
    // One request id per report: a retry is answered from the host's reply
    // cache, never applied twice.
    let mut request_id = None;
    loop {
        let command = match &report {
            Report::Safe {
                epoch,
                code,
                reason,
            } => {
                let (now, mode) = shared.epoch_and_mode();
                if now >= epoch.0 && matches!(mode, Mode::Safe | Mode::Estop) {
                    return true;
                }
                Command::EnterSafe {
                    reason: reason.clone(),
                    code: *code,
                }
            }
            Report::Estop {
                reason,
                code,
                origin,
                at_ns,
                ..
            } => Command::Estop {
                reason: reason.clone(),
                code: *code,
                origin: origin.clone(),
                at_ns: *at_ns,
            },
            Report::Clear { .. } => Command::Clear,
        };
        let (id, reply) = shared.call(request_id, &command).await;
        request_id = id;
        let result = reply.and_then(|reply| reply.result.map_err(SessionError::Rejected));
        match Attempt::of(&result, shared.generation_ended()) {
            Attempt::Done => return true,
            Attempt::Anew => request_id = None,
            Attempt::Retry => {
                if let Err(err) = &result {
                    warn!(participant = %shared.id, %err, ?report, "stop report unanswered; retrying");
                }
                tokio::select! {
                    () = cancel.cancelled() => return false,
                    () = tokio::time::sleep(SAFE_RETRY) => {}
                }
            }
        }
    }
}

/// The host's virtual-mode ticks, as they arrive.
pub struct TickStream {
    rx: mpsc::UnboundedReceiver<VirtualTick>,
    _subscriber: zenoh::pubsub::Subscriber<()>,
}

impl TickStream {
    /// The next tick (`None` once the subscription is gone).
    pub async fn next(&mut self) -> Option<VirtualTick> {
        self.rx.recv().await
    }

    /// Waits for the next tick matching `predicate`, up to `timeout`.
    pub async fn wait_for(
        &mut self,
        timeout: Duration,
        mut predicate: impl FnMut(&VirtualTick) -> bool,
    ) -> Result<VirtualTick, SessionError> {
        tokio::time::timeout(timeout, async {
            while let Some(tick) = self.rx.recv().await {
                if predicate(&tick) {
                    return Ok(tick);
                }
            }
            Err(SessionError::Closed)
        })
        .await
        .map_err(|_| SessionError::Timeout(timeout, "a virtual tick".into()))?
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

/// Queries a session host for its authority snapshot, without attaching:
/// how anyone gets state at once. **Unverified**: without a handshake there
/// is no host key to check its signature against (authority state is
/// public, §8.7), so act only on what [`Participant::fetch_authority`]
/// returns.
pub async fn query_authority(
    transport: &Transport,
    keys: &SessionKeys,
    timeout: Duration,
) -> Result<AuthoritySnapshot, SessionError> {
    let signed = ask(transport.control(), &keys.host_state(), None, timeout).await?;
    let body = signed::peek(&signed)
        .map_err(|err| {
            SessionError::Wire(crate::wire::WireError::Malformed {
                message: "host state",
                reason: err.to_string(),
            })
        })?
        .body;
    Ok(codec::decode_state(body)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_reports_keep_estop_before_clear_and_retry_through_spoofed_refusals() {
        // The latch is a member's signal.
        let latch = Reason {
            code: ReasonCode::EstopSignal,
            by: "eye".into(),
            detail: "button".into(),
            ..Reason::default()
        };
        let estop = |seq| Report::estop(&latch, seq);
        let clear = |after| Report::Clear {
            after,
            reason: latch.clone(),
        };
        // A latch whose report was queued after the clear that followed
        // it: the clear sends its Estop first, with the latch's own
        // reason, and the late one nothing.
        let mut order = StopOrder::default();
        let sent: Vec<Report> = [clear(1), estop(1), estop(2), clear(2)]
            .into_iter()
            .flat_map(|r| order.plan(r))
            .collect();
        assert!(
            matches!(&sent[0], Report::Estop {
                reason,
                code: ReasonCode::EstopSignal,
                origin: Some(by),
                ..
            } if reason == "button" && by.as_str() == "eye"),
            "{sent:?}"
        );
        let kinds: Vec<&str> = sent
            .iter()
            .map(|r| match r {
                Report::Estop { .. } => "estop",
                Report::Clear { .. } => "clear",
                Report::Safe { .. } => "safe",
            })
            .collect();
        assert_eq!(kinds, ["estop", "clear", "estop", "clear"]);

        // A plain UNAUTHENTICATED (a MITM's, while the host's reply is
        // suppressed) keeps the Estop queued; the restored link's answer
        // ends it, and only then does the Clear go. The host's own sealed
        // UNAUTHENTICATED is an answer.
        let spoofed = Err(SessionError::Unverified(CtlError::new(
            CtlStatus::Unauthenticated,
            "spoofed",
        )));
        assert_eq!(Attempt::of(&spoofed, false), Attempt::Retry);
        assert_eq!(
            Attempt::of(&Err(SessionError::NoReply(Duration::ZERO)), false),
            Attempt::Retry
        );
        assert_eq!(Attempt::of(&Ok(()), false), Attempt::Done);
        // An authenticated refusal, an expired id, a generation that ended.
        let refused = |status| Err(SessionError::Rejected(CtlError::new(status, "")));
        for status in [CtlStatus::InvalidTransition, CtlStatus::Unauthenticated] {
            assert_eq!(Attempt::of(&refused(status), false), Attempt::Done);
        }
        assert_eq!(
            Attempt::of(&refused(CtlStatus::Expired), false),
            Attempt::Anew
        );
        assert_eq!(Attempt::of(&spoofed, true), Attempt::Done);
    }
}
