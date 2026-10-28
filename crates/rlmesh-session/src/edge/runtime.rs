//! The networked edge: an [`EdgeCore`] fed by its participant.
//!
//! [`Edge::attach`] covers one commandable part, or every member of its
//! command group. It subscribes to each member's command key on its traffic
//! class and splices every chunk on the transport's callback thread, against
//! the participant's live authority view, and it declares each member's
//! `_applied` publisher. The control loop calls [`Edge::tick`]: the edge
//! decides, publishes one applied sample per member (§9.4), and when it falls
//! back on its own account it tells the host through the participant
//! (`EnterSafe`, retried in the background); the heartbeat reports its
//! queue. Every chunk and tick reads the robot process's local e-stop latch
//! first ([`Participant::latch_estop`]).

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::participant::LocalLatch;
use crate::transport::payload_bytes;
use crate::wire::codec;
use crate::wire::{ActionChunk, ChannelId, ChunkRef, ClockMode, ReasonCode};
use crate::{
    AuthorityView, Clock, EdgeReport, Participant, SafeReporter, SamplePublisher, TrafficClass,
};
use tracing::{debug, info};

use crate::edge::core::{AppliedAction, EdgeConfig, EdgeCore, EdgeStats, EdgeTick, Rejection};
use crate::edge::error::EdgeError;

/// Called with every accepted chunk and its edge receive time, on the
/// transport's callback thread: keep it short.
pub type OnAccept = Arc<dyn Fn(&ActionChunk, i64) + Send + Sync>;

struct Shared {
    core: Mutex<EdgeCore>,
    view: AuthorityView,
    clock: Arc<dyn Clock>,
    report: Arc<EdgeReport>,
    reporter: SafeReporter,
    /// The robot process's local e-stop.
    latch: Arc<LocalLatch>,
    /// How many latches the core has seen.
    latches_seen: AtomicU64,
    on_accept: Option<OnAccept>,
    /// The time of the last tick: the edge's clock in virtual mode.
    tick_ns: AtomicI64,
}

impl Shared {
    /// The core, with the local latch as of now.
    fn core(&self) -> MutexGuard<'_, EdgeCore> {
        // A poisoned edge only means a panic elsewhere mid-update; the core
        // stays consistent between calls, so keep serving it.
        let mut core = self
            .core
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // A latch cleared since the last look still stops and fences.
        let latches = self.latch.latches();
        if self.latches_seen.swap(latches, Ordering::SeqCst) != latches {
            core.set_local_estop(Some(ReasonCode::EstopLocal));
        }
        core.set_local_estop(self.latch.is_latched().then_some(ReasonCode::EstopLocal));
        core
    }

    fn on_payload(&self, payload: &zenoh::bytes::ZBytes) {
        let chunk = match codec::decode_chunk(payload_bytes(payload)) {
            Ok(chunk) => chunk,
            Err(err) => {
                debug!(%err, "undecodable chunk");
                self.core().count_rejection(Rejection::Malformed);
                return;
            }
        };
        let accepted = self.view.with(|view| {
            // In virtual and lockstep sessions the edge's clock is the
            // session tick it was last driven at.
            let receive_ns = if view.is_some_and(|v| v.clock_mode != ClockMode::Wall) {
                self.tick_ns.load(Ordering::Relaxed)
            } else {
                self.clock.now_ns()
            };
            let observed = self.view.observed_local_ns();
            let mut core = self.core();
            let outcome = core.accept(chunk.clone(), receive_ns, view, observed);
            self.report.set_queue_depth(core.queue_depth());
            outcome.map(|_| receive_ns)
        });
        match accepted {
            Ok(receive_ns) => {
                self.report.set_accepted(chunk.seq);
                if let Some(on_accept) = &self.on_accept {
                    on_accept(&chunk, receive_ns);
                }
            }
            Err(reason) => debug!(channel = %chunk.channel, ?reason, "chunk rejected"),
        }
    }
}

/// The edge runtime for one command group. Dropping it undeclares the
/// intake and the applied publishers.
pub struct Edge {
    shared: Arc<Shared>,
    channels: Vec<ChannelId>,
    applied: Vec<SamplePublisher>,
    _subscribers: Vec<zenoh::pubsub::Subscriber<()>>,
}

impl Edge {
    /// The edge for the commandable part at `path` that `participant`
    /// declared, and every other member of its command group.
    pub async fn attach(
        participant: &Participant,
        path: &str,
        config: EdgeConfig,
        on_accept: Option<OnAccept>,
    ) -> Result<Self, EdgeError> {
        let part = participant.own_part(path)?;
        let group = part.command.as_ref().and_then(|c| c.group.clone());
        let parts = match &group {
            Some(group) => participant
                .authority()
                .map(|view| {
                    view.group_members(participant.id(), group)
                        .cloned()
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default(),
            None => vec![part],
        };
        let core = EdgeCore::new(parts.clone(), participant.id().clone(), config)?;
        let shared = Arc::new(Shared {
            core: Mutex::new(core),
            view: participant.authority_view(),
            clock: participant.clock().clone(),
            report: participant.edge_report(),
            reporter: participant.safe_reporter(),
            latch: participant.estop_latch(),
            latches_seen: AtomicU64::new(0),
            on_accept,
            tick_ns: AtomicI64::new(0),
        });
        let mut subscribers = Vec::with_capacity(parts.len());
        let mut applied = Vec::with_capacity(parts.len());
        for part in &parts {
            // A plain subscriber: chunks are never served from history, so
            // a restarted edge cannot act on old ones.
            subscribers.push(
                participant
                    .transport()
                    .session(TrafficClass::of(part.priority))
                    .declare_subscriber(
                        participant
                            .keys()
                            .commands(participant.id(), part.path.as_str()),
                    )
                    .callback({
                        let shared = shared.clone();
                        move |sample| shared.on_payload(sample.payload())
                    })
                    .await?,
            );
            let stream = part
                .applied_stream()
                .ok_or(EdgeError::NotCommandable(part.channel))?
                .map_err(|_| EdgeError::NotCommandable(part.channel))?;
            applied.push(participant.stream_publisher(stream).await?);
        }
        Ok(Self {
            shared,
            channels: parts.iter().map(|p| p.channel).collect(),
            applied,
            _subscribers: subscribers,
        })
    }

    /// The group's parts.
    pub fn channels(&self) -> &[ChannelId] {
        &self.channels
    }

    /// The counters so far.
    pub fn stats(&self) -> EdgeStats {
        self.shared.core().stats()
    }

    /// One control tick now: what to dispatch, hold, or the safe behaviour
    /// to run. Call it at the embodiment's control rate; every timer runs on
    /// this edge's clock alone.
    pub fn tick(&self) -> EdgeTick {
        self.tick_at(self.shared.clock.now_ns())
    }

    /// [`Self::tick`] at an explicit edge time (session time in virtual
    /// mode: the tick × `tick_ns`).
    pub fn tick_at(&self, now_ns: i64) -> EdgeTick {
        let shared = &self.shared;
        shared.tick_ns.store(now_ns, Ordering::Relaxed);
        let (tick, tick_number) = shared.view.with(|view| {
            let observed = shared.view.observed_local_ns();
            let mut core = shared.core();
            let tick = core.tick(now_ns, view, observed);
            shared.report.set_queue_depth(core.queue_depth());
            let tick_ns = view.map_or(0, |v| v.tick_ns);
            let number = (tick_ns > 0).then(|| now_ns.max(0).unsigned_abs() / tick_ns);
            (tick, number.unwrap_or(0))
        });
        for (sample, publisher) in tick.applied.iter().zip(&self.applied) {
            if let Err(err) = publisher.publish_sample(
                now_ns,
                tick_number,
                sample.value.clone(),
                Some(sample.applied),
            ) {
                debug!(stream = %sample.stream, %err, "applied publish failed");
            }
        }
        if let Some(action) = tick.action.as_ref() {
            shared.report.set_applied(action.chunk.seq);
        }
        if let Some((epoch, cause)) = tick.report {
            let channel = tick.channel.map(|c| c.to_string()).unwrap_or_default();
            info!(%channel, %cause, "edge fell back on its own account");
            shared
                .reporter
                .report(epoch, cause.code(), format!("{channel}: {cause}"));
        }
        tick
    }

    /// The robot finished a handover between group members (it switched its
    /// control mode and neutralised the old member's target): the group may
    /// dispatch again.
    pub fn handover_done(&self) {
        self.shared.core().handover_done();
    }

    /// Whether the edge has judged `chunk` (virtual mode: wait for every
    /// chunk a `Step` lists before stepping).
    pub fn has_judged(&self, chunk: &ChunkRef) -> bool {
        self.shared.core().has_judged(chunk)
    }

    /// Takes the applied-action log: one entry per tick, oldest first.
    pub fn drain_log(&self) -> Vec<AppliedAction> {
        self.shared.core().drain_log()
    }
}
