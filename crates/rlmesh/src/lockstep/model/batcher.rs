//! The coalescing batcher: predicts from every session a fusing model answers
//! in, assembled into one handler call while the forward before it runs.

use std::collections::HashSet;
use std::io::Write as _;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use rlmesh_session::wire::timing::ns;
use rlmesh_session::wire::{EndpointTiming, ModelGroup, ModelOutcome};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc, oneshot};
use tracing::warn;

use super::{ModelService, fault};
use crate::model::handler::ModelHandler;
use crate::serve_options::BatchOptions;

/// A served model's batcher: its options, the queue its loop reads (started
/// on the first predict, dropped at close), and where batches are recorded.
pub(super) struct Batching {
    options: BatchOptions,
    pub(super) queue: std::sync::Mutex<Option<mpsc::UnboundedSender<Queued>>>,
    records: Option<std::sync::Mutex<std::fs::File>>,
    /// The model's name and address, which each record names.
    pub(super) serving: std::sync::OnceLock<(String, String)>,
}

/// A predict waiting in the batcher.
pub(super) struct Queued {
    groups: Vec<ModelGroup>,
    bytes: usize,
    arrived: Instant,
    reply: oneshot::Sender<(Vec<ModelOutcome>, EndpointTiming)>,
}

/// The observation bytes a predict carries, history included.
fn observation_bytes(groups: &[ModelGroup]) -> usize {
    let leaves = |leaves: &[Bytes]| leaves.iter().map(Bytes::len).sum::<usize>();
    groups
        .iter()
        .map(|g| {
            leaves(&g.observation)
                + g.history
                    .iter()
                    .map(|h| leaves(&h.observation))
                    .sum::<usize>()
        })
        .sum()
}
impl<H: ModelHandler + 'static> ModelService<H> {
    /// Predicts `groups` in the next batch, or on their own when the model
    /// does not batch.
    pub(super) async fn coalesce(
        self: &Arc<Self>,
        groups: Vec<ModelGroup>,
        timing: &mut EndpointTiming,
    ) -> Vec<ModelOutcome> {
        let Some(batching) = &self.batching else {
            return self.predict(groups, timing).await;
        };
        let queue = {
            let mut queue = batching
                .queue
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            queue
                .get_or_insert_with(|| {
                    let (tx, rx) = mpsc::unbounded_channel();
                    tokio::spawn(Arc::clone(self).run_batches(rx));
                    tx
                })
                .clone()
        };
        let (reply, answered) = oneshot::channel();
        let count = groups.len();
        let queued = Queued {
            bytes: observation_bytes(&groups),
            groups,
            arrived: Instant::now(),
            reply,
        };
        if let Err(mpsc::error::SendError(queued)) = queue.send(queued) {
            // Closing: answer it on its own.
            return self.predict(queued.groups, timing).await;
        }
        match answered.await {
            Ok((outcomes, answered)) => {
                *timing = EndpointTiming {
                    in_flight: timing.in_flight,
                    ..answered
                };
                outcomes
            }
            Err(_) => (0..count)
                .map(|_| Err(fault("INTERNAL", "the model stopped batching", false)))
                .collect(),
        }
    }

    /// The batcher loop: take an in-flight slot, assemble a batch, hand it
    /// to the handler, and assemble the next while it runs. The slot is
    /// taken before assembly, so at the cap requests keep joining the batch
    /// that goes the moment a slot frees. While another batch is in flight
    /// the batch stays open until that one ends (the handler runs one at a
    /// time) or it fills: closing it on the window instead sent whatever
    /// trickled in behind a big batch as a small one paying a whole forward.
    /// Either way it closes once every resolved route has a request in it,
    /// as no other request can come.
    async fn run_batches(self: Arc<Self>, mut rx: mpsc::UnboundedReceiver<Queued>) {
        let Some(options) = self.batching.as_ref().map(|b| b.options.clone()) else {
            return;
        };
        let cap = options.max_in_flight_batches.max(1);
        let window = Duration::from_millis(options.batch_window_ms);
        let in_flight = Arc::new(Semaphore::new(cap));
        // A request that did not fit the batch being assembled opens the next.
        let mut carry = None;
        loop {
            let Ok(permit) = Arc::clone(&in_flight).acquire_owned().await else {
                return;
            };
            let first = match carry.take() {
                Some(first) => first,
                None => match rx.recv().await {
                    Some(first) => first,
                    None => return,
                },
            };
            let (mut size, mut bytes) = (first.groups.len(), first.bytes);
            let mut waiting: HashSet<String> =
                first.groups.iter().map(|g| g.route.clone()).collect();
            let mut batch = vec![first];
            let busy = in_flight.available_permits() < cap - 1;
            let deadline = tokio::time::Instant::now() + window;
            while size < options.max_batch_size && bytes < options.batch_byte_budget {
                // Registered before the check, so a release in between still
                // wakes this batch to check again.
                let released = self.released.notified();
                tokio::pin!(released);
                released.as_mut().enable();
                if self.every_route_in(&waiting).await {
                    break;
                }
                let next = if busy {
                    tokio::select! {
                        queued = rx.recv() => queued,
                        // A slot freeing is the running forward ending; the
                        // permit goes straight back so this batch takes it.
                        freed = Arc::clone(&in_flight).acquire_owned() => {
                            drop(freed);
                            break;
                        }
                        () = &mut released => continue,
                    }
                } else {
                    tokio::select! {
                        queued = tokio::time::timeout_at(deadline, rx.recv()) => match queued {
                            Ok(queued) => queued,
                            Err(_) => break,
                        },
                        () = &mut released => continue,
                    }
                };
                let Some(queued) = next else { break };
                if size + queued.groups.len() > options.max_batch_size
                    || bytes + queued.bytes > options.batch_byte_budget
                {
                    carry = Some(queued);
                    break;
                }
                size += queued.groups.len();
                bytes += queued.bytes;
                waiting.extend(queued.groups.iter().map(|g| g.route.clone()));
                batch.push(queued);
            }
            let running = cap - 1 - in_flight.available_permits();
            tokio::spawn(Arc::clone(&self).flush(batch, running, permit));
        }
    }

    /// Whether every resolved route has a request in `waiting`.
    async fn every_route_in(&self, waiting: &HashSet<String>) -> bool {
        self.routes.lock().await.keys().all(|r| waiting.contains(r))
    }

    /// Answers a batch with one handler call, each request with its own
    /// groups' outcomes, and records it. `permit` is the batch's in-flight
    /// slot, released before the record write.
    async fn flush(
        self: Arc<Self>,
        batch: Vec<Queued>,
        in_flight_at_dispatch: usize,
        permit: OwnedSemaphorePermit,
    ) {
        let dispatched = Instant::now();
        let mut waits = Vec::new();
        let mut groups = Vec::new();
        let mut members = Vec::with_capacity(batch.len());
        for queued in batch {
            let waited = dispatched.duration_since(queued.arrived);
            waits.extend(std::iter::repeat_n(
                waited.as_secs_f64() * 1e3,
                queued.groups.len(),
            ));
            members.push((queued.reply, queued.groups.len(), waited));
            groups.extend(queued.groups);
        }
        let size = groups.len();
        let mut timing = EndpointTiming::default();
        let outcomes = self.predict(groups, &mut timing).await;
        let forward_wall = dispatched.elapsed();
        drop(permit);
        let ok = outcomes.iter().any(Result::is_ok);
        let mut outcomes = outcomes.into_iter();
        for (reply, count, waited) in members {
            let mine = outcomes.by_ref().take(count).collect();
            let queued = ns(waited).unwrap_or(0) + timing.queue_ns.unwrap_or(0);
            // The batch's work is shared: each request is charged its groups'
            // share of a grouped reply.
            // Waits are each request's own and pass through whole.
            let share = |ns: Option<u64>| {
                ns.map(|ns| (u128::from(ns) * count as u128 / size.max(1) as u128) as u64)
                    .filter(|ns| *ns != 0)
            };
            let timing = EndpointTiming {
                queue_ns: Some(queued).filter(|ns| *ns != 0),
                decode_ns: share(timing.decode_ns),
                user_ns: share(timing.user_ns),
                encode_ns: share(timing.encode_ns),
                adapter_ns: share(timing.adapter_ns),
                ..timing
            };
            let _ = reply.send((mine, timing));
        }
        if let Some(batching) = &self.batching {
            batching.record(BatchRecord {
                size,
                in_flight_at_dispatch,
                waits,
                forward_wall,
                user_ns: timing.user_ns,
                ok,
            });
        }
    }
}

/// One handler call of the batcher.
struct BatchRecord {
    size: usize,
    in_flight_at_dispatch: usize,
    /// How long each group sat in the batcher before the call, in ms.
    waits: Vec<f64>,
    forward_wall: Duration,
    user_ns: Option<u64>,
    ok: bool,
}

impl Batching {
    pub(super) fn new(options: BatchOptions, records: Option<std::fs::File>) -> Self {
        Self {
            options,
            queue: std::sync::Mutex::default(),
            records: records.map(std::sync::Mutex::new),
            serving: std::sync::OnceLock::new(),
        }
    }

    /// Appends `batch` to the record file, in the shape of the
    /// runner's `telemetry/batches.jsonl` so one reader reads both:
    /// `forwardWallMs` is the batch's call to settle (decode, the handler
    /// lock, the forward, encode) and `endpointTotalMs` the handler call.
    fn record(&self, mut batch: BatchRecord) {
        let Some(records) = &self.records else {
            return;
        };
        batch.waits.sort_by(f64::total_cmp);
        let p50 = match batch.waits.len().checked_sub(1) {
            Some(last) => batch.waits[((last as f64 * 0.5).round() as usize).min(last)],
            None => 0.0,
        };
        let (name, address) = self.serving.get().cloned().unwrap_or_default();
        let max_size = self.options.max_batch_size.max(1);
        let line = serde_json::json!({
            "at": chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::AutoSi, true),
            "componentId": name,
            "endpoint": address,
            "size": batch.size,
            "maxSize": max_size,
            "fill": batch.size as f64 / max_size as f64,
            "inFlightAtDispatch": batch.in_flight_at_dispatch,
            "queueWaitMs": {
                "p50": p50,
                "max": batch.waits.last().copied().unwrap_or(0.0),
            },
            "forwardWallMs": batch.forward_wall.as_secs_f64() * 1e3,
            "endpointTotalMs": batch.user_ns.map(|ns| ns as f64 / 1e6),
            "ok": batch.ok,
        });
        let mut file = records
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Err(err) = writeln!(file, "{line}") {
            warn!(%err, "batch record not written");
        }
    }
}
