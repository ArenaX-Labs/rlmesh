//! The cockpit's network side: an observer [`Participant`] subscribed to
//! everything a session shows (authority, every stream and `_applied`
//! stream, every commandable part's chunks, heartbeats and observer
//! reports), turned into [`Input`]s for the [`Model`], plus the verbs the
//! operator sends.
//!
//! The observer is never in the loop: subscriptions run on the transport's
//! threads and only queue inputs. Samples go through a bounded queue the UI
//! drains each frame; when the UI falls behind, inputs are dropped and
//! counted ([`Input::Overrun`]), never buffered without bound. Authority,
//! presence and verb answers ride their own unbounded queue (they are rare)
//! and are taken in first, so a new part's spec is known before its
//! samples.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, SyncSender, TrySendError, channel, sync_channel};
use std::sync::{Arc, Mutex};

use rlmesh_session::transport::{TrafficClass, payload_bytes};
use rlmesh_session::{Participant, Received, SampleSubscriber, SessionError, SubscribeOptions};
use rlmesh_wire::codec;
use rlmesh_wire::{ChannelId, StreamSpec};
use tokio::sync::mpsc::{UnboundedSender, unbounded_channel};
use tokio::task::JoinHandle;
use tracing::{debug, warn};

use crate::model::{Input, Model, Verb};

/// How many data inputs may wait for the UI.
pub const QUEUE: usize = 1 << 16;

/// Called when something the operator should see at once arrived.
pub type Wake = Arc<dyn Fn() + Send + Sync>;

/// The data queue's sending side: drops and counts when full.
#[derive(Clone)]
struct Data {
    tx: SyncSender<Input>,
    dropped: Arc<AtomicU64>,
}

impl Data {
    fn send(&self, input: Input) {
        if let Err(TrySendError::Full(_)) = self.tx.try_send(input) {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// A running feed; dropping it stops its tasks and subscriptions.
pub struct Feed {
    participant: Arc<Participant>,
    control: Receiver<Input>,
    data: Receiver<Input>,
    dropped: Arc<AtomicU64>,
    verbs: UnboundedSender<Verb>,
    tasks: Vec<JoinHandle<()>>,
    _subscriptions: Arc<Mutex<Subscriptions>>,
}

#[derive(Default)]
struct Subscriptions {
    streams: Vec<SampleSubscriber>,
    zenoh: Vec<zenoh::pubsub::Subscriber<()>>,
}

impl Feed {
    /// Subscribes `participant` (attached as an observer) to the session.
    /// Call it on a Tokio runtime.
    pub async fn start(participant: Arc<Participant>, wake: Wake) -> Result<Self, SessionError> {
        let (control_tx, control) = channel();
        let (data_tx, data) = sync_channel(QUEUE);
        let dropped = Arc::new(AtomicU64::new(0));
        let data_tx = Data {
            tx: data_tx,
            dropped: dropped.clone(),
        };
        let subscriptions = Arc::new(Mutex::new(Subscriptions::default()));
        let keys = participant.keys().clone();
        let clock = participant.clock().clone();

        // Heartbeats (control class) and reports (data class).
        let heartbeats = participant
            .transport()
            .control()
            .declare_subscriber(keys.heartbeats())
            .callback({
                let (data, clock) = (data_tx.clone(), clock.clone());
                move |sample| match codec::decode_heartbeat(&sample.payload().to_bytes()) {
                    Ok(heartbeat) => data.send(Input::Heartbeat {
                        heartbeat,
                        receive_ns: clock.now_ns(),
                    }),
                    Err(err) => debug!(%err, "undecodable heartbeat"),
                }
            })
            .await?;
        let reports = participant
            .transport()
            .session(TrafficClass::Data)
            .declare_subscriber(keys.reports())
            .callback({
                let (data, clock) = (data_tx.clone(), clock.clone());
                move |sample| match codec::decode_report(&sample.payload().to_bytes()) {
                    Ok(report) => data.send(Input::Report {
                        report,
                        receive_ns: clock.now_ns(),
                    }),
                    Err(err) => debug!(%err, "undecodable report"),
                }
            })
            .await?;
        if let Ok(mut subs) = subscriptions.lock() {
            subs.zenoh.extend([heartbeats, reports]);
        }

        let (verbs, mut verb_rx) = unbounded_channel::<Verb>();
        let tasks = vec![
            tokio::spawn(watch_authority(
                participant.clone(),
                control_tx.clone(),
                wake.clone(),
            )),
            tokio::spawn(watch_parts(
                participant.clone(),
                data_tx,
                subscriptions.clone(),
            )),
            tokio::spawn({
                let (participant, control, wake) =
                    (participant.clone(), control_tx.clone(), wake.clone());
                async move {
                    while let Some(verb) = verb_rx.recv().await {
                        let result = match verb {
                            Verb::Arm => participant.arm().await.map(drop),
                            Verb::Disarm => participant.disarm().await.map(drop),
                            Verb::Estop => participant
                                .estop(format!("cockpit {}", participant.id()))
                                .await
                                .map(drop),
                        };
                        if let Err(err) = &result {
                            warn!(verb = verb.name(), %err, "verb refused");
                        }
                        let _ = control.send(Input::Outcome {
                            verb,
                            result: result.map_err(|err| err.to_string()),
                        });
                        wake();
                    }
                }
            }),
            tokio::spawn({
                let (participant, control, wake) = (participant.clone(), control_tx, wake);
                async move {
                    let mut present = participant.host_present();
                    loop {
                        let now = *present.borrow_and_update();
                        if control.send(Input::HostPresent(now)).is_err() {
                            return;
                        }
                        wake();
                        if present.changed().await.is_err() {
                            return;
                        }
                    }
                }
            }),
        ];
        Ok(Self {
            participant,
            control,
            data,
            dropped,
            verbs,
            tasks,
            _subscriptions: subscriptions,
        })
    }

    /// The observer.
    pub fn participant(&self) -> &Arc<Participant> {
        &self.participant
    }

    /// Now on the cockpit's clock.
    pub fn now_ns(&self) -> i64 {
        self.participant.clock().now_ns()
    }

    /// Sends a verb; its answer arrives as [`Input::Outcome`].
    pub fn send(&self, verb: Verb) {
        let _ = self.verbs.send(verb);
    }

    /// Takes every waiting input into `model`; returns how many.
    pub fn drain(&self, model: &mut Model) -> usize {
        let now = self.now_ns();
        let mut n = 0;
        for input in self.control.try_iter().chain(self.data.try_iter()) {
            model.ingest(input, now);
            n += 1;
        }
        let dropped = self.dropped.load(Ordering::Relaxed);
        if dropped > 0 {
            model.ingest(Input::Overrun(dropped), now);
        }
        n
    }
}

impl Drop for Feed {
    fn drop(&mut self) {
        for task in &self.tasks {
            task.abort();
        }
    }
}

async fn watch_authority(participant: Arc<Participant>, control: Sender<Input>, wake: Wake) {
    let mut view = participant.watch();
    loop {
        let snapshot = view.borrow_and_update().clone();
        if let Some(snapshot) = snapshot {
            if control.send(Input::Authority(snapshot)).is_err() {
                return;
            }
            wake();
        }
        if view.changed().await.is_err() {
            return;
        }
    }
}

/// Subscribes to every stream and every commandable part's chunks, as parts
/// appear.
async fn watch_parts(
    participant: Arc<Participant>,
    data: Data,
    subscriptions: Arc<Mutex<Subscriptions>>,
) {
    let mut streams: BTreeSet<ChannelId> = BTreeSet::new();
    let mut chunks: BTreeSet<ChannelId> = BTreeSet::new();
    let mut view = participant.watch();
    loop {
        let snapshot = view.borrow_and_update().clone();
        for part in snapshot.iter().flat_map(|s| &s.parts) {
            let wanted: Vec<StreamSpec> = part
                .stream()
                .ok()
                .into_iter()
                .chain(part.applied_stream().and_then(Result::ok))
                .collect();
            for stream in wanted {
                if !streams.insert(stream.channel) {
                    continue;
                }
                let on_sample = {
                    let data = data.clone();
                    Arc::new(move |received: &Received| {
                        data.send(Input::Sample {
                            sample: received.sample.clone(),
                            receive_ns: received.receive_ns,
                            capture_local_ns: received.capture_local_ns,
                        });
                    })
                };
                let options = SubscribeOptions {
                    on_sample: Some(on_sample),
                    window: 0,
                    warm_from_history: false,
                };
                match participant.subscribe_stream(stream, options).await {
                    Ok(sub) => {
                        if let Ok(mut subs) = subscriptions.lock() {
                            subs.streams.push(sub);
                        }
                    }
                    Err(err) => warn!(%err, "cockpit: subscription refused"),
                }
            }
            if part.command.is_some() && chunks.insert(part.channel) {
                let key = participant.keys().commands(&part.owner, part.path.as_str());
                let (data, clock) = (data.clone(), participant.clock().clone());
                let declared = participant
                    .transport()
                    .session(TrafficClass::of(part.priority))
                    .declare_subscriber(key)
                    .callback(move |sample| {
                        let receive_ns = clock.now_ns();
                        match codec::decode_chunk(payload_bytes(sample.payload())) {
                            Ok(chunk) => data.send(Input::Chunk { chunk, receive_ns }),
                            Err(err) => debug!(%err, "undecodable chunk"),
                        }
                    })
                    .await;
                match declared {
                    Ok(sub) => {
                        if let Ok(mut subs) = subscriptions.lock() {
                            subs.zenoh.push(sub);
                        }
                    }
                    Err(err) => warn!(%err, "cockpit: chunk subscription refused"),
                }
            }
        }
        if view.changed().await.is_err() {
            return;
        }
    }
}
