//! The networked edge: an [`EdgeCore`] fed by its participant.
//!
//! [`Edge::attach`] subscribes to the channel's action key on its traffic
//! class and splices every chunk on the transport's callback thread, against
//! the participant's live authority view. The control loop calls
//! [`Edge::tick`]; when the edge falls back on its own account, it tells the
//! host through the participant (`EnterSafe`, retried in the background), and
//! the heartbeat reports its queue.

use std::sync::{Arc, Mutex, MutexGuard};

use rlmesh_session::transport::payload_bytes;
use rlmesh_session::{AuthorityView, Clock, EdgeReport, Participant, SafeReporter};
use rlmesh_wire::codec;
use rlmesh_wire::{ActionChunk, ChannelId};
use tracing::{debug, info};

use crate::core::{AppliedAction, EdgeConfig, EdgeCore, EdgeStats, EdgeTick, Rejection};
use crate::error::EdgeError;

/// Called with every accepted chunk and its edge receive time, on the
/// transport's callback thread: keep it short.
pub type OnAccept = Arc<dyn Fn(&ActionChunk, i64) + Send + Sync>;

struct Shared {
    core: Mutex<EdgeCore>,
    view: AuthorityView,
    clock: Arc<dyn Clock>,
    report: Arc<EdgeReport>,
    reporter: SafeReporter,
    on_accept: Option<OnAccept>,
}

impl Shared {
    fn core(&self) -> MutexGuard<'_, EdgeCore> {
        // A poisoned edge only means a panic elsewhere mid-update; the core
        // stays consistent between calls, so keep serving it.
        self.core
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn on_payload(&self, payload: &zenoh::bytes::ZBytes) {
        let receive_ns = self.clock.now_ns();
        let chunk = match codec::decode_chunk(payload_bytes(payload)) {
            Ok(chunk) => chunk,
            Err(err) => {
                debug!(%err, "undecodable chunk");
                self.core().count_rejection(Rejection::Malformed);
                return;
            }
        };
        let accepted = self.view.with(|view| {
            let observed = self.view.observed_local_ns();
            let mut core = self.core();
            let outcome = core.accept(chunk.clone(), receive_ns, view, observed);
            self.report.set_queue_depth(core.queue_depth());
            outcome
        });
        match accepted {
            Ok(_) => {
                self.report.set_accepted(chunk.seq);
                if let Some(on_accept) = &self.on_accept {
                    on_accept(&chunk, receive_ns);
                }
            }
            Err(reason) => debug!(channel = %chunk.channel, ?reason, "chunk rejected"),
        }
    }
}

/// The edge runtime for one actuation channel. Dropping it undeclares the
/// intake.
pub struct Edge {
    shared: Arc<Shared>,
    channel: ChannelId,
    _subscriber: zenoh::pubsub::Subscriber<()>,
}

impl Edge {
    /// The edge for `channel`, an actuation channel `participant` declared.
    pub async fn attach(
        participant: &Participant,
        channel: ChannelId,
        config: EdgeConfig,
        on_accept: Option<OnAccept>,
    ) -> Result<Self, EdgeError> {
        let spec = participant.channel(channel)?;
        let priority = spec.qos.priority;
        let core = EdgeCore::new(spec, participant.id().clone(), config)?;
        let shared = Arc::new(Shared {
            core: Mutex::new(core),
            view: participant.authority_view(),
            clock: participant.clock().clone(),
            report: participant.edge_report(),
            reporter: participant.safe_reporter(),
            on_accept,
        });
        // A plain subscriber: chunks are never served from history, so a
        // restarted edge cannot act on old ones.
        let subscriber = participant
            .transport()
            .for_priority(priority)
            .declare_subscriber(participant.keys().actions(participant.id(), channel))
            .callback({
                let shared = shared.clone();
                move |sample| shared.on_payload(sample.payload())
            })
            .await?;
        Ok(Self {
            shared,
            channel,
            _subscriber: subscriber,
        })
    }

    /// The channel.
    pub fn channel(&self) -> ChannelId {
        self.channel
    }

    /// The counters so far.
    pub fn stats(&self) -> EdgeStats {
        self.shared.core().stats()
    }

    /// One control tick now: what to apply, hold, or the safe behaviour to
    /// run. Call it at the embodiment's control rate; every timer runs on
    /// this edge's clock alone.
    pub fn tick(&self) -> EdgeTick {
        self.tick_at(self.shared.clock.now_ns())
    }

    /// [`Self::tick`] at an explicit edge-clock time.
    pub fn tick_at(&self, now_ns: i64) -> EdgeTick {
        let shared = &self.shared;
        let tick = shared.view.with(|view| {
            let observed = shared.view.observed_local_ns();
            let mut core = shared.core();
            let tick = core.tick(now_ns, view, observed);
            shared.report.set_queue_depth(core.queue_depth());
            tick
        });
        if let Some(action) = tick.action.as_ref() {
            shared.report.set_applied(action.chunk.seq);
        }
        if let Some((epoch, cause)) = tick.report {
            info!(channel = %self.channel, %cause, "edge fell back on its own account");
            shared
                .reporter
                .report(epoch, format!("{}: {cause}", self.channel));
        }
        tick
    }

    /// Takes the applied-action log: one entry per tick, oldest first.
    pub fn drain_log(&self) -> Vec<AppliedAction> {
        self.shared.core().drain_log()
    }
}
