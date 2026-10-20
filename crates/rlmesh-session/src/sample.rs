//! Samples: publishing a channel and consuming it with explicit loss and
//! staleness.
//!
//! A [`SamplePublisher`] numbers its samples per channel (`seq`), stamps the
//! epoch, the capture time and the publish time, and sends the leaves as a
//! data frame without copying them. A [`SampleSubscriber`] keeps the latest
//! value and a bounded window, counts gaps in `seq` as losses, drops samples
//! from an older epoch, checks every frame's length against the declaration,
//! and reports a sample's age against the channel's `max_age`, so a consumer
//! never silently freezes on an old value.
//!
//! Late joiners warm their window from the host's history store
//! ([`SubscribeOptions::warm_from_history`]) rather than from a publisher
//! cache: the publisher stays a plain Zenoh publisher, with no retransmission
//! (which would hold newer samples behind missing ones, the head-of-line
//! blocking a real-time channel must not have).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use rlmesh_wire::codec;
use rlmesh_wire::{AuthoritySnapshot, ChannelSpec, ParticipantId, Sample};
use tokio::sync::watch;
use tracing::{debug, warn};
use zenoh::Wait;
use zenoh::bytes::ZBytes;
use zenoh::query::ConsolidationMode;

use crate::clock::Clock;
use crate::error::SessionError;
use crate::transport::{Transport, frame_zbytes, payload_bytes, zenoh_congestion, zenoh_priority};

/// Publishes one channel. Cheap to share: `publish` takes `&self`.
pub struct SamplePublisher {
    spec: ChannelSpec,
    publisher: zenoh::pubsub::Publisher<'static>,
    seq: AtomicU64,
    epoch: Arc<AtomicU64>,
    clock: Arc<dyn Clock>,
}

impl SamplePublisher {
    pub(crate) async fn declare(
        transport: &Transport,
        key: String,
        spec: ChannelSpec,
        epoch: Arc<AtomicU64>,
        clock: Arc<dyn Clock>,
    ) -> Result<Self, SessionError> {
        let qos = spec.qos;
        let publisher = transport
            .for_priority(qos.priority)
            .declare_publisher(key)
            .priority(zenoh_priority(qos.priority))
            .congestion_control(zenoh_congestion(&qos))
            .express(qos.express())
            .await?;
        Ok(Self {
            spec,
            publisher,
            seq: AtomicU64::new(0),
            epoch,
            clock,
        })
    }

    /// The channel.
    pub fn spec(&self) -> &ChannelSpec {
        &self.spec
    }

    /// The seq of the last published sample (zero before the first).
    pub fn last_seq(&self) -> u64 {
        self.seq.load(Ordering::Relaxed)
    }

    /// Publishes one value, captured at `capture_ns` (this participant's
    /// clock), as one buffer per leaf (one JPEG for a JPEG channel). Returns
    /// the sample's seq. The leaves are checked against the declaration and
    /// sent without a copy; never blocks on a DROP channel.
    pub fn publish(&self, capture_ns: i64, leaves: Vec<Bytes>) -> Result<u64, SessionError> {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed) + 1;
        let sample = Sample {
            channel: self.spec.id,
            seq,
            epoch: rlmesh_wire::Epoch(self.epoch.load(Ordering::Relaxed)),
            capture_ns,
            publish_ns: self.clock.now_ns(),
            tick: 0,
            leaves,
        };
        self.spec
            .check_leaf_lens(sample.leaves.iter().map(Bytes::len))
            .map_err(|reason| SessionError::InvalidValue {
                channel: self.spec.id,
                reason,
            })?;
        self.publisher
            .put(frame_zbytes(codec::encode_sample(&sample)))
            .wait()?;
        Ok(seq)
    }
}

/// A sample as received.
#[derive(Clone, Debug)]
pub struct Received {
    /// The sample; its leaves share the received buffer.
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

/// The latest sample, judged against the channel's `max_age`.
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
    /// Arrived with a seq at or below one already seen (reordered or
    /// duplicated); dropped.
    pub out_of_order: u64,
    /// From an older epoch than the subscriber's view; dropped.
    pub stale_epoch: u64,
    /// Not a decodable sample.
    pub undecodable: u64,
    /// A sample that does not fit the declaration (the level-2 length
    /// check, a wrong channel).
    pub invalid: u64,
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
    /// Seed the window from the host's history store before live samples.
    pub warm_from_history: bool,
}

struct SubState {
    spec: ChannelSpec,
    publisher: ParticipantId,
    me: ParticipantId,
    clock: Arc<dyn Clock>,
    view: watch::Receiver<Option<AuthoritySnapshot>>,
    stats: Mutex<SampleStats>,
    latest: Mutex<Option<Received>>,
    window: Mutex<VecDeque<Received>>,
    /// Shared with the participant's heartbeat (`last_seq_seen`).
    seen: Arc<AtomicU64>,
    /// Publisher clock to subscriber clock, refreshed from the view.
    offset_ns: AtomicI64,
    on_sample: Option<OnSample>,
}

impl SubState {
    fn on_payload(&self, payload: &ZBytes, from_history: bool) {
        let receive_ns = self.clock.now_ns();
        let sample = match codec::decode_sample(payload_bytes(payload)) {
            Ok(sample) => sample,
            Err(err) => {
                debug!(channel = %self.spec.id, %err, "undecodable sample");
                self.bump(|s| s.undecodable += 1);
                return;
            }
        };
        // Level 2, O(leaves): the key's channel, and each leaf's length
        // against the declared layout.
        let fits = if sample.channel == self.spec.id {
            self.spec
                .check_leaf_lens(sample.leaves.iter().map(Bytes::len))
        } else {
            Err(format!("{} on {}'s key", sample.channel, self.spec.id))
        };
        if let Err(reason) = fits {
            debug!(channel = %self.spec.id, reason, "invalid sample");
            self.bump(|s| s.invalid += 1);
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
        let depth = self.spec.qos.history.depth();
        if depth > 1
            && let Ok(mut window) = self.window.lock()
        {
            if window.len() == depth {
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
        if let (Some(theirs), Some(mine)) = (offset(&self.publisher), offset(&self.me)) {
            self.offset_ns
                .store(theirs.saturating_sub(mine), Ordering::Relaxed);
        }
    }
}

/// Consumes one channel. Dropping it undeclares the subscription.
pub struct SampleSubscriber {
    state: Arc<SubState>,
    _subscriber: zenoh::pubsub::Subscriber<()>,
    offsets: tokio::task::JoinHandle<()>,
}

pub(crate) struct SubscribeContext {
    pub key: String,
    pub history_key: String,
    pub spec: ChannelSpec,
    pub publisher: ParticipantId,
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
        let depth = ctx.spec.qos.history.depth();
        let priority = ctx.spec.qos.priority;
        let state = Arc::new(SubState {
            spec: ctx.spec,
            publisher: ctx.publisher,
            me: ctx.me,
            clock: ctx.clock,
            view: ctx.view,
            stats: Mutex::new(SampleStats::default()),
            latest: Mutex::new(None),
            window: Mutex::new(VecDeque::with_capacity(depth)),
            seen: ctx.seen,
            offset_ns: AtomicI64::new(0),
            on_sample: options.on_sample,
        });
        state.refresh_offset();
        if options.warm_from_history {
            let history = fetch_history(transport, &ctx.history_key, ctx.history_timeout).await;
            match history {
                Ok(frames) => {
                    for frame in &frames {
                        state.on_payload(frame, true);
                    }
                }
                Err(err) => warn!(channel = %state.spec.id, %err, "history unavailable"),
            }
        }
        let subscriber = transport
            .for_priority(priority)
            .declare_subscriber(ctx.key)
            .callback({
                let state = state.clone();
                move |sample| state.on_payload(sample.payload(), false)
            })
            .await?;
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

    /// The channel.
    pub fn spec(&self) -> &ChannelSpec {
        &self.state.spec
    }

    /// The counters so far.
    pub fn stats(&self) -> SampleStats {
        self.state.stats.lock().map(|s| *s).unwrap_or_default()
    }

    /// The latest sample with its age; `stale` when older than the channel's
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
            .qos
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

    /// The bounded window, oldest first (the latest only, for `LATEST`).
    pub fn window(&self) -> Vec<Received> {
        if self.state.spec.qos.history.depth() <= 1 {
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

/// Queries the host's history store for a channel's window, oldest first.
pub(crate) async fn fetch_history(
    transport: &Transport,
    key: &str,
    timeout: Duration,
) -> Result<Vec<ZBytes>, SessionError> {
    let replies = transport
        .session(crate::transport::TrafficClass::Data)
        .get(key)
        .consolidation(ConsolidationMode::None)
        .timeout(timeout)
        .await?;
    let mut frames = Vec::new();
    while let Ok(reply) = replies.recv_async().await {
        match reply.result() {
            Ok(sample) => frames.push(sample.payload().clone()),
            Err(err) => warn!(?err, "history error reply"),
        }
    }
    Ok(frames)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::MonotonicClock;
    use rlmesh_wire::{
        ChannelId, ChannelPriority, ChannelQos, Direction, Epoch, History, TensorLayout,
    };
    use std::time::Instant;

    fn spec(max_age: Option<Duration>, depth: u32) -> ChannelSpec {
        let mut qos = ChannelQos::realtime(ChannelPriority::Realtime);
        qos.max_age = max_age;
        qos.history = History::from_depth(depth);
        ChannelSpec::new(
            ChannelId(1),
            "ft",
            Direction::Sense,
            "proprio/eef_wrench",
            vec![TensorLayout::f32(&[6])],
            1000.0,
            qos,
        )
    }

    fn state(depth: u32, view: watch::Receiver<Option<AuthoritySnapshot>>) -> SubState {
        SubState {
            spec: spec(None, depth),
            publisher: ParticipantId::new("robot").expect("id"),
            me: ParticipantId::new("ctl").expect("id"),
            clock: MonotonicClock::with_anchor(Instant::now(), 0).shared(),
            view,
            stats: Mutex::new(SampleStats::default()),
            latest: Mutex::new(None),
            window: Mutex::new(VecDeque::new()),
            seen: Arc::new(AtomicU64::new(0)),
            offset_ns: AtomicI64::new(0),
            on_sample: None,
        }
    }

    fn payload(seq: u64, epoch: u64, len: usize) -> ZBytes {
        let sample = Sample {
            channel: ChannelId(1),
            seq,
            epoch: Epoch(epoch),
            capture_ns: 0,
            publish_ns: 0,
            tick: 0,
            leaves: vec![Bytes::from(vec![0_u8; len])],
        };
        frame_zbytes(codec::encode_sample(&sample))
    }

    #[test]
    fn gaps_count_as_losses_and_reorders_are_dropped() {
        let (_tx, view) = watch::channel(None);
        let state = state(3, view);
        for seq in [1, 2, 5, 4, 6, 6, 10] {
            state.on_payload(&payload(seq, 1, 24), false);
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
    fn old_epochs_and_misfit_frames_are_dropped_and_counted() {
        let mut snapshot = crate::authority::Authority::new(
            rlmesh_wire::SessionId::new(),
            crate::authority::AuthorityConfig::default(),
        )
        .snapshot(Vec::new(), 0);
        snapshot.epoch = Epoch(3);
        let (_tx, view) = watch::channel(Some(snapshot));
        let state = state(1, view);
        state.on_payload(&payload(1, 2, 24), false);
        state.on_payload(&payload(2, 3, 24), false);
        state.on_payload(&payload(3, 3, 20), false);
        state.on_payload(&ZBytes::from(vec![1_u8, 2, 3]), false);
        let stats = *state.stats.lock().expect("stats");
        assert_eq!((stats.stale_epoch, stats.received), (1, 1));
        assert_eq!((stats.invalid, stats.undecodable), (1, 1));
    }
}
