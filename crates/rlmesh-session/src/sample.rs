//! Samples: publishing a stream under its publication plan, and consuming
//! it with explicit loss and staleness.
//!
//! A [`SamplePublisher`] numbers its samples per stream (`seq`), stamps the
//! epoch, the capture time and the publish time, checks the payload against
//! the stream's packed layout (§10.6) and sends it without copying, as its
//! stream's **publication plan** says (§8.2). The host derives the plan from
//! every admitted delivery contract on the stream and publishes it as a
//! `Demand`; the participant installs it here and acknowledges it:
//!
//! - **every** (the default, with no demand, and for any `every` or
//!   `history` contract): each sample at once;
//! - **latest**: the newest sample at the plan's rate, coalescing the rest.
//!
//! No plan exceeds the part's `send` rate ceiling (a token bucket with a
//! burst of two, so jitter around the declared rate is not decimation). A
//! coalesced sample takes no seq, so a gap at a subscriber is still a loss.
//! Events, and every stream in virtual or lockstep mode (whose barriers wait
//! on seqs), always send every sample; their plans are acknowledged all the
//! same.
//!
//! A [`SampleSubscriber`] takes each sample, keeps the latest value and a
//! bounded window, counts gaps in `seq` as losses, drops samples from an
//! older epoch and samples that fail the layout check, and reports a
//! sample's age against the stream's `max_age`, so a consumer never silently
//! freezes on an old value.
//!
//! Late joiners warm their window from the host's history store
//! ([`SubscribeOptions::warm_from_history`]): the live subscription starts
//! first and buffers, the store is queried up to the first live seq, and the
//! two are merged by seq (§8.8). The publisher keeps no retransmission
//! (which would hold newer samples behind missing ones, the head-of-line
//! blocking a real-time stream must not have).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicI64, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use rlmesh_wire::codec;
use rlmesh_wire::{
    Applied, AuthoritySnapshot, ClockMode, Epoch, HistoryQuery, ParticipantId, Plan, Sample,
    SampleBatch, StreamSpec,
};
use tokio::sync::{Notify, watch};
use tracing::{debug, warn};
use zenoh::Wait;
use zenoh::bytes::ZBytes;
use zenoh::query::ConsolidationMode;

use crate::clock::Clock;
use crate::error::SessionError;
use crate::transport::{
    TrafficClass, Transport, declare_publisher, frame_zbytes, payload_bytes, stream_qos,
};

/// Every sample at once, ungated.
const FREE: u8 = 0;
/// Every sample or the newest, under a token bucket.
const GATED: u8 = 1;

/// At most `hz` samples per second, with a burst of two.
#[derive(Debug)]
struct Gate {
    per_ns: f64,
    tokens: f64,
    last_ns: i64,
}

impl Gate {
    const BURST: f64 = 2.0;

    fn new(hz: f64) -> Self {
        Self {
            per_ns: hz / 1e9,
            tokens: Self::BURST,
            last_ns: i64::MIN,
        }
    }

    fn admit(&mut self, now_ns: i64) -> bool {
        if self.last_ns != i64::MIN {
            let elapsed = now_ns.saturating_sub(self.last_ns).max(0) as f64;
            self.tokens = (self.tokens + elapsed * self.per_ns).min(Self::BURST);
        }
        self.last_ns = now_ns;
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

/// The installed plan.
#[derive(Debug)]
struct PlanState {
    /// The demand revision installed (0: none, the default plan).
    revision: u64,
    plan: Option<Plan>,
    gate: Option<Gate>,
}

/// What publishing one stream shares with the plan installer.
pub(crate) struct Outlet {
    spec: StreamSpec,
    publisher: zenoh::pubsub::Publisher<'static>,
    seq: AtomicU64,
    epoch: Arc<AtomicU64>,
    clock: Arc<dyn Clock>,
    /// Every sample whatever the plan: events, virtual and lockstep modes.
    fixed: bool,
    /// The `send` rate ceiling.
    ceiling_hz: Option<f64>,
    mode: AtomicU8,
    state: Mutex<PlanState>,
    changed: Notify,
    coalesced: AtomicU64,
}

impl Outlet {
    fn put_sample(&self, sample: &Sample) -> Result<(), SessionError> {
        self.publisher
            .put(frame_zbytes(codec::encode_sample(sample)))
            .wait()?;
        Ok(())
    }

    /// Installs demand revision `revision` (`None`: no demand, the default
    /// plan).
    pub(crate) fn install(&self, revision: u64, plan: Option<Plan>) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        if state.revision == revision && state.plan == plan {
            return;
        }
        let rate = match plan {
            Some(Plan::Latest { hz }) => Some(self.ceiling_hz.map_or(hz, |c| c.min(hz))),
            Some(Plan::Every) | None => self.ceiling_hz,
        };
        state.gate = rate.map(Gate::new);
        let mode = if self.fixed {
            FREE
        } else if state.gate.is_some() {
            GATED
        } else {
            FREE
        };
        state.revision = revision;
        state.plan = plan;
        self.mode.store(mode, Ordering::Release);
        drop(state);
        self.changed.notify_waiters();
    }

    /// The installed demand revision (0: none).
    pub(crate) fn revision(&self) -> u64 {
        self.state.lock().map_or(0, |s| s.revision)
    }
}

/// Publishes one stream. Cheap to share: `publish` takes `&self`.
pub struct SamplePublisher {
    outlet: Arc<Outlet>,
}

impl SamplePublisher {
    pub(crate) async fn declare(
        transport: &Transport,
        key: String,
        spec: StreamSpec,
        clock_mode: ClockMode,
        ceiling_hz: Option<f64>,
        epoch: Arc<AtomicU64>,
        clock: Arc<dyn Clock>,
    ) -> Result<Self, SessionError> {
        let publisher = declare_publisher(
            transport,
            TrafficClass::of(spec.priority),
            key,
            stream_qos(&spec, clock_mode),
        )
        .await?;
        let fixed = spec.event || clock_mode != ClockMode::Wall;
        let ceiling_hz = ceiling_hz.filter(|_| !fixed);
        // Before any demand the default plan applies: every sample within
        // the `send` ceiling (§8.2).
        let gate = ceiling_hz.map(Gate::new);
        let outlet = Arc::new(Outlet {
            fixed,
            ceiling_hz,
            mode: AtomicU8::new(if gate.is_some() { GATED } else { FREE }),
            state: Mutex::new(PlanState {
                revision: 0,
                plan: None,
                gate,
            }),
            changed: Notify::new(),
            coalesced: AtomicU64::new(0),
            spec,
            publisher,
            seq: AtomicU64::new(0),
            epoch,
            clock,
        });
        Ok(Self { outlet })
    }

    pub(crate) fn outlet(&self) -> &Arc<Outlet> {
        &self.outlet
    }

    /// The stream.
    pub fn spec(&self) -> &StreamSpec {
        &self.outlet.spec
    }

    /// The seq of the last published sample (zero before the first).
    pub fn last_seq(&self) -> u64 {
        self.outlet.seq.load(Ordering::Relaxed)
    }

    /// The installed publication plan and its demand revision (`None`, 0:
    /// no demand, every sample within the `send` ceiling).
    pub fn plan(&self) -> (Option<Plan>, u64) {
        self.outlet
            .state
            .lock()
            .map_or((None, 0), |s| (s.plan, s.revision))
    }

    /// Samples the plan (or the `send` ceiling) coalesced.
    pub fn coalesced(&self) -> u64 {
        self.outlet.coalesced.load(Ordering::Relaxed)
    }

    /// Now on this publisher's clock (the clock [`Self::due`] and the
    /// rate gate read).
    pub fn now_ns(&self) -> i64 {
        self.outlet.clock.now_ns()
    }

    /// Whether a sample offered at `now_ns` (this publisher's clock,
    /// [`Self::now_ns`]) would be published rather than coalesced (an image
    /// publisher asks before encoding).
    pub fn due(&self, now_ns: i64) -> bool {
        if self.outlet.mode.load(Ordering::Acquire) != GATED {
            return true;
        }
        self.outlet.state.lock().is_ok_and(|s| {
            s.gate.as_ref().is_none_or(|g| {
                let elapsed = now_ns.saturating_sub(g.last_ns).max(0) as f64;
                g.last_ns == i64::MIN || g.tokens + elapsed * g.per_ns >= 1.0
            })
        })
    }

    /// Publishes one packed value, captured at `capture_ns` (this
    /// participant's clock): its seq, or `None` when the plan coalesced it.
    /// The payload is checked against the stream's layout and sent without
    /// a copy; never blocks on a DROP stream.
    pub fn publish(&self, capture_ns: i64, payload: Bytes) -> Result<Option<u64>, SessionError> {
        self.publish_sample(capture_ns, 0, payload, None)
    }

    /// [`Self::publish`] with a virtual or lockstep tick and, on an applied
    /// stream, the applied metadata (a tick that dispatched nothing has no
    /// payload).
    pub fn publish_sample(
        &self,
        capture_ns: i64,
        tick: u64,
        payload: Bytes,
        applied: Option<Applied>,
    ) -> Result<Option<u64>, SessionError> {
        let outlet = &*self.outlet;
        let invalid = |reason: String| SessionError::InvalidValue {
            channel: outlet.spec.channel,
            reason,
        };
        match (outlet.spec.applied, applied) {
            (false, None) => {}
            (true, Some(a)) if !a.present => {
                if !payload.is_empty() {
                    return Err(invalid("an absent applied value has no payload".into()));
                }
            }
            (true, Some(_)) => {}
            (true, None) => return Err(invalid("an applied sample needs its metadata".into())),
            (false, Some(_)) => return Err(invalid("not an applied stream".into())),
        }
        if applied.is_none_or(|a| a.present) {
            outlet.spec.check_payload(&payload).map_err(invalid)?;
        }
        let epoch = Epoch(outlet.epoch.load(Ordering::Relaxed));
        let sample = |seq| Sample {
            channel: outlet.spec.channel,
            epoch,
            seq,
            capture_ns,
            publish_ns: outlet.clock.now_ns(),
            tick,
            payload: payload.clone(),
            applied,
        };
        if outlet.mode.load(Ordering::Acquire) == GATED {
            let now = outlet.clock.now_ns();
            let admitted = outlet
                .state
                .lock()
                .map_or(true, |mut s| s.gate.as_mut().is_none_or(|g| g.admit(now)));
            if !admitted {
                outlet.coalesced.fetch_add(1, Ordering::Relaxed);
                return Ok(None);
            }
        }
        let seq = outlet.seq.fetch_add(1, Ordering::Relaxed) + 1;
        outlet.put_sample(&sample(seq))?;
        Ok(Some(seq))
    }
}

/// The sample a data payload carries.
pub(crate) fn decode_data(payload: &ZBytes) -> Result<Sample, String> {
    codec::decode_sample(payload_bytes(payload)).map_err(|e| e.to_string())
}

/// A sample as received.
#[derive(Clone, Debug)]
pub struct Received {
    /// The sample; its payload shares the received buffer.
    pub sample: Sample,
    /// Subscriber's clock at receipt.
    pub receive_ns: i64,
    /// The capture time converted to the subscriber's clock, through the
    /// host's clock mappings (unconverted when either mapping is unknown).
    pub capture_local_ns: i64,
}

impl Received {
    /// Receipt minus capture: the one-way delay estimate.
    pub fn delay_ns(&self) -> i64 {
        self.receive_ns.saturating_sub(self.capture_local_ns)
    }
}

/// The latest sample, judged against the stream's `max_age`.
#[derive(Clone, Debug)]
pub struct Fresh {
    /// The sample.
    pub received: Received,
    /// Now minus capture, in the subscriber's clock.
    pub age_ns: i64,
    /// Older than `max_age`: a consumer must not use it as current.
    pub stale: bool,
}

/// Loss and freshness counters for one subscription.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SampleStats {
    /// Samples accepted.
    pub received: u64,
    /// Samples skipped over by `seq` gaps.
    pub lost: u64,
    /// Arrived with a seq at or below one already seen (reordered,
    /// duplicated, or already served from history); dropped.
    pub out_of_order: u64,
    /// From an older epoch than the subscriber's view; dropped.
    pub stale_epoch: u64,
    /// Not a decodable sample.
    pub undecodable: u64,
    /// A sample that does not fit the stream (`malformed`: the layout
    /// check, a wrong channel).
    pub malformed: u64,
    /// Reads of [`SampleSubscriber::latest`] that found the value stale.
    pub stale_reads: u64,
    /// Samples served from the host's history store.
    pub from_history: u64,
    /// The newest seq accepted.
    pub last_seq: u64,
}

/// Called with every accepted sample.
pub type OnSample = Arc<dyn Fn(&Received) + Send + Sync>;

/// How to subscribe.
#[derive(Clone, Default)]
pub struct SubscribeOptions {
    /// Called on every accepted sample, on the transport's callback thread:
    /// keep it short.
    pub on_sample: Option<OnSample>,
    /// Samples kept in the window (0 and 1 keep the latest only).
    pub window: usize,
    /// Seed the window from the host's history store before live samples.
    pub warm_from_history: bool,
}

struct SubState {
    spec: StreamSpec,
    me: ParticipantId,
    clock: Arc<dyn Clock>,
    view: watch::Receiver<Option<AuthoritySnapshot>>,
    stats: Mutex<SampleStats>,
    latest: Mutex<Option<Received>>,
    depth: usize,
    window: Mutex<VecDeque<Received>>,
    /// While warming from history, live payloads wait here.
    warming: Mutex<Option<Vec<ZBytes>>>,
    /// Shared with the participant's heartbeat (`last_seq_seen`).
    seen: Arc<AtomicU64>,
    /// Publisher clock to subscriber clock, refreshed from the view.
    offset_ns: AtomicI64,
    on_sample: Option<OnSample>,
}

impl SubState {
    fn on_live(&self, payload: &ZBytes) {
        if let Ok(mut warming) = self.warming.lock()
            && let Some(buffer) = warming.as_mut()
        {
            buffer.push(payload.clone());
            return;
        }
        self.on_payload(payload);
    }

    fn on_payload(&self, payload: &ZBytes) {
        match decode_data(payload) {
            Ok(sample) => self.on_sample(sample, false),
            Err(err) => {
                debug!(channel = %self.spec.channel, %err, "undecodable sample");
                self.bump(|s| s.undecodable += 1);
            }
        }
    }

    fn on_sample(&self, sample: Sample, from_history: bool) {
        let receive_ns = self.clock.now_ns();
        // The key's stream, and the payload against the declared layout
        // (O(fields), before any view of the value).
        let fits = if sample.channel != self.spec.channel {
            Err(format!("{} on {}'s key", sample.channel, self.spec.channel))
        } else {
            match (self.spec.applied, sample.applied) {
                (true, Some(a)) if !a.present => {
                    if sample.payload.is_empty() {
                        Ok(())
                    } else {
                        Err("an absent applied value with a payload".into())
                    }
                }
                (true, None) => Err("an applied sample without metadata".into()),
                _ => self.spec.check_payload(&sample.payload),
            }
        };
        if let Err(reason) = fits {
            debug!(channel = %self.spec.channel, reason, "malformed sample");
            self.bump(|s| s.malformed += 1);
            return;
        }
        let view_epoch = self.view.borrow().as_ref().map_or(0, |v| v.epoch.0);
        let accepted = {
            let Ok(mut stats) = self.stats.lock() else {
                return;
            };
            if sample.epoch.0 < view_epoch {
                stats.stale_epoch += 1;
                false
            } else if stats.last_seq > 0 && sample.seq <= stats.last_seq {
                stats.out_of_order += 1;
                false
            } else {
                if stats.last_seq > 0 {
                    stats.lost += sample.seq - stats.last_seq - 1;
                }
                stats.last_seq = sample.seq;
                stats.received += 1;
                if from_history {
                    stats.from_history += 1;
                }
                true
            }
        };
        if !accepted {
            return;
        }
        self.seen.store(sample.seq, Ordering::Relaxed);
        let received = Received {
            capture_local_ns: sample
                .capture_ns
                .saturating_add(self.offset_ns.load(Ordering::Relaxed)),
            sample,
            receive_ns,
        };
        if let Some(on_sample) = &self.on_sample {
            on_sample(&received);
        }
        if self.depth > 1
            && let Ok(mut window) = self.window.lock()
        {
            if window.len() == self.depth {
                window.pop_front();
            }
            window.push_back(received.clone());
        }
        if let Ok(mut latest) = self.latest.lock() {
            *latest = Some(received);
        }
    }

    fn bump(&self, f: impl FnOnce(&mut SampleStats)) {
        if let Ok(mut stats) = self.stats.lock() {
            f(&mut stats);
        }
    }

    /// session = participant + offset, so publisher → subscriber is
    /// `offset(publisher) − offset(me)`.
    fn refresh_offset(&self) {
        let view = self.view.borrow();
        let Some(view) = view.as_ref() else {
            return;
        };
        let offset = |id: &ParticipantId| view.clock(id).map(|c| c.offset_ns);
        if let (Some(theirs), Some(mine)) = (offset(&self.spec.owner), offset(&self.me)) {
            self.offset_ns
                .store(theirs.saturating_sub(mine), Ordering::Relaxed);
        }
    }
}

/// Consumes one stream. Dropping it undeclares the subscription.
pub struct SampleSubscriber {
    state: Arc<SubState>,
    _subscriber: zenoh::pubsub::Subscriber<()>,
    offsets: tokio::task::JoinHandle<()>,
}

pub(crate) struct SubscribeContext {
    pub key: String,
    pub history_key: String,
    pub spec: StreamSpec,
    pub me: ParticipantId,
    pub clock: Arc<dyn Clock>,
    pub view: watch::Receiver<Option<AuthoritySnapshot>>,
    pub seen: Arc<AtomicU64>,
    pub history_timeout: Duration,
}

impl SampleSubscriber {
    pub(crate) async fn declare(
        transport: &Transport,
        ctx: SubscribeContext,
        options: SubscribeOptions,
    ) -> Result<Self, SessionError> {
        let depth = options.window.max(1);
        let priority = ctx.spec.priority;
        let state = Arc::new(SubState {
            spec: ctx.spec,
            me: ctx.me,
            clock: ctx.clock,
            view: ctx.view,
            stats: Mutex::new(SampleStats::default()),
            latest: Mutex::new(None),
            depth,
            window: Mutex::new(VecDeque::with_capacity(depth.min(1024))),
            warming: Mutex::new(options.warm_from_history.then(Vec::new)),
            seen: ctx.seen,
            offset_ns: AtomicI64::new(0),
            on_sample: options.on_sample,
        });
        state.refresh_offset();
        // Live first: what arrives while the store answers waits in the
        // warming buffer.
        let subscriber = transport
            .for_priority(priority)
            .declare_subscriber(ctx.key)
            .callback({
                let state = state.clone();
                move |sample| state.on_live(sample.payload())
            })
            .await?;
        if options.warm_from_history {
            let first_live = state
                .warming
                .lock()
                .ok()
                .and_then(|w| w.as_ref()?.first().cloned())
                .and_then(|p| decode_data(&p).ok())
                .map(|s| s.seq);
            let epoch = Epoch(state.view.borrow().as_ref().map_or(0, |v| v.epoch.0));
            let query = HistoryQuery {
                epoch,
                after_seq: 0,
                until_seq: first_live
                    .map(|seq| seq.saturating_sub(1))
                    .filter(|s| *s > 0),
                limit: u32::try_from(depth).ok(),
            };
            match fetch_history(transport, &ctx.history_key, &query, ctx.history_timeout).await {
                Ok(pages) => {
                    let stride = state
                        .spec
                        .layout
                        .stride()
                        .filter(|_| state.spec.batchable());
                    for page in pages.iter().filter(|p| p.epoch == epoch) {
                        match page.split(stride) {
                            Ok(samples) => {
                                for sample in samples {
                                    state.on_sample(sample, true);
                                }
                            }
                            Err(reason) => {
                                debug!(channel = %state.spec.channel, reason, "malformed page");
                                state.bump(|s| s.malformed += 1);
                            }
                        }
                    }
                }
                Err(err) => warn!(channel = %state.spec.channel, %err, "history unavailable"),
            }
            // Hold the warming lock while flushing, so no live sample can
            // overtake the buffered ones.
            if let Ok(mut warming) = state.warming.lock() {
                for payload in warming.take().unwrap_or_default() {
                    state.on_payload(&payload);
                }
            }
        }
        let offsets = tokio::spawn({
            let state = state.clone();
            let mut view = state.view.clone();
            async move {
                while view.changed().await.is_ok() {
                    state.refresh_offset();
                }
            }
        });
        Ok(Self {
            state,
            _subscriber: subscriber,
            offsets,
        })
    }

    /// The stream.
    pub fn spec(&self) -> &StreamSpec {
        &self.state.spec
    }

    /// The counters so far.
    pub fn stats(&self) -> SampleStats {
        self.state.stats.lock().map(|s| *s).unwrap_or_default()
    }

    /// The latest sample with its age; `stale` when older than the stream's
    /// `max_age` (never, without one).
    pub fn latest(&self) -> Option<Fresh> {
        let received = self.state.latest.lock().ok()?.clone()?;
        let age_ns = self
            .state
            .clock
            .now_ns()
            .saturating_sub(received.capture_local_ns);
        let stale = self
            .state
            .spec
            .max_age
            .is_some_and(|max| u128::try_from(age_ns).is_ok_and(|age| age > max.as_nanos()));
        if stale {
            self.state.bump(|s| s.stale_reads += 1);
        }
        Some(Fresh {
            received,
            age_ns,
            stale,
        })
    }

    /// The bounded window, oldest first (the latest only, for a window of
    /// one).
    pub fn window(&self) -> Vec<Received> {
        if self.state.depth <= 1 {
            return self
                .state
                .latest
                .lock()
                .ok()
                .and_then(|l| l.clone())
                .into_iter()
                .collect();
        }
        self.state
            .window
            .lock()
            .map(|w| w.iter().cloned().collect())
            .unwrap_or_default()
    }
}

impl Drop for SampleSubscriber {
    fn drop(&mut self) {
        self.offsets.abort();
    }
}

/// Queries the host's history store: every page of the reply, in order.
pub(crate) async fn fetch_history(
    transport: &Transport,
    key: &str,
    query: &HistoryQuery,
    timeout: Duration,
) -> Result<Vec<SampleBatch>, SessionError> {
    let replies = transport
        .session(TrafficClass::Data)
        .get(key)
        .payload(codec::encode_history_query(query))
        .consolidation(ConsolidationMode::None)
        .timeout(timeout)
        .await?;
    let mut pages = Vec::new();
    while let Ok(reply) = replies.recv_async().await {
        match reply.result() {
            Ok(sample) => match codec::decode_batch(payload_bytes(sample.payload())) {
                Ok(page) => pages.push(page),
                Err(err) => warn!(%err, "undecodable history page"),
            },
            Err(err) => warn!(?err, "history error reply"),
        }
    }
    Ok(pages)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::MonotonicClock;
    use rlmesh_wire::{ChannelId, ChannelPriority, DType, FieldSpec, Layout};
    use std::time::Instant;

    fn spec() -> StreamSpec {
        let fields = vec![FieldSpec::new("wrench", DType::F32, &[6])];
        StreamSpec {
            channel: ChannelId(1),
            owner: ParticipantId::new("robot").expect("id"),
            path: "ft".into(),
            layout: Layout::of(&fields).expect("layout"),
            fields,
            encoding: rlmesh_wire::Encoding::Raw,
            priority: ChannelPriority::Realtime,
            event: false,
            max_age: None,
            applied: false,
        }
    }

    fn state(depth: usize, view: watch::Receiver<Option<AuthoritySnapshot>>) -> SubState {
        SubState {
            spec: spec(),
            me: ParticipantId::new("ctl").expect("id"),
            clock: MonotonicClock::with_anchor(Instant::now(), 0).shared(),
            view,
            stats: Mutex::new(SampleStats::default()),
            latest: Mutex::new(None),
            depth,
            window: Mutex::new(VecDeque::new()),
            warming: Mutex::new(None),
            seen: Arc::new(AtomicU64::new(0)),
            offset_ns: AtomicI64::new(0),
            on_sample: None,
        }
    }

    fn payload(seq: u64, epoch: u64, len: usize) -> ZBytes {
        let sample = Sample {
            channel: ChannelId(1),
            epoch: Epoch(epoch),
            seq,
            capture_ns: 0,
            publish_ns: 0,
            tick: 0,
            payload: Bytes::from(vec![0_u8; len]),
            applied: None,
        };
        frame_zbytes(codec::encode_sample(&sample))
    }

    #[test]
    fn gaps_count_as_losses_and_reorders_are_dropped() {
        let (_tx, view) = watch::channel(None);
        let state = state(3, view);
        for seq in [1, 2, 5, 4, 6, 6, 10] {
            state.on_payload(&payload(seq, 1, 24));
        }
        let stats = *state.stats.lock().expect("stats");
        assert_eq!(stats.received, 5);
        assert_eq!(stats.lost, 2 + 3, "3,4 skipped by 5; 7,8,9 by 10");
        assert_eq!(stats.out_of_order, 2, "the late 4 and the duplicate 6");
        assert_eq!(stats.last_seq, 10);
        assert_eq!(state.seen.load(Ordering::Relaxed), 10);
        let window: Vec<u64> = state
            .window
            .lock()
            .expect("window")
            .iter()
            .map(|r| r.sample.seq)
            .collect();
        assert_eq!(window, vec![5, 6, 10], "bounded to the newest three");
    }

    #[test]
    fn old_epochs_and_misfit_payloads_are_dropped_and_counted() {
        let mut snapshot = crate::authority::Authority::new(
            rlmesh_wire::SessionId::new(),
            crate::authority::AuthorityConfig::default(),
        )
        .snapshot(Vec::new(), 0);
        snapshot.epoch = Epoch(3);
        let (_tx, view) = watch::channel(Some(snapshot));
        let state = state(1, view);
        state.on_payload(&payload(1, 2, 24));
        state.on_payload(&payload(2, 3, 24));
        state.on_payload(&payload(3, 3, 20));
        state.on_payload(&ZBytes::from(vec![1_u8, 2, 3]));
        let stats = *state.stats.lock().expect("stats");
        assert_eq!((stats.stale_epoch, stats.received), (1, 1));
        assert_eq!((stats.malformed, stats.undecodable), (1, 1));
    }

    #[test]
    fn the_send_ceiling_admits_its_rate_with_a_burst_of_two() {
        let mut gate = Gate::new(10.0);
        let admitted = (0..100_i64).filter(|i| gate.admit(i * 10_000_000)).count();
        assert!((10..=12).contains(&admitted), "{admitted} of 100 at 100 Hz");
        // Jitter around the declared rate is not decimation.
        let mut gate = Gate::new(30.0);
        let admitted = (0..300_i64)
            .filter(|i| gate.admit(i * 33_333_333 + if i % 2 == 0 { 2_000_000 } else { 0 }))
            .count();
        assert_eq!(admitted, 300);
    }

    #[test]
    fn live_samples_wait_while_warming_and_merge_by_seq() {
        let (_tx, view) = watch::channel(None);
        let state = state(8, view);
        *state.warming.lock().expect("lock") = Some(Vec::new());
        state.on_live(&payload(5, 1, 24));
        state.on_live(&payload(6, 1, 24));
        assert_eq!(state.stats.lock().expect("stats").received, 0, "buffered");
        // History up to the first live seq, overlapping it by one.
        for seq in [3, 4, 5] {
            if let Ok(sample) = codec::decode_sample(payload_bytes(&payload(seq, 1, 24))) {
                state.on_sample(sample, true);
            }
        }
        for p in state
            .warming
            .lock()
            .expect("lock")
            .take()
            .unwrap_or_default()
        {
            state.on_payload(&p);
        }
        let stats = *state.stats.lock().expect("stats");
        assert_eq!(
            (stats.received, stats.from_history, stats.out_of_order),
            (4, 3, 1)
        );
        assert_eq!(stats.lost, 0);
    }
}
