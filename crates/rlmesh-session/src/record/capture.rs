//! The [`Recorder`]: subscribing to a session's streams, chunks and authority, and
//! queueing each envelope for the writer thread.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::record::rolling::{Rolling, RollingSink};
use crate::record::writer::{Segment, Sink, write_loop};
use crate::record::{RecordError, RecordStats};
use crate::transport::{TrafficClass, payload_bytes};
use crate::wire::codec::{self, schema};
use crate::wire::{
    AuthoritySnapshot, ChannelId, Fingerprint, Mode, PartSpec, ParticipantId, Sample, SessionId,
    StreamSpec,
};
use crate::{HealthSubscription, Participant, SampleSubscriber, SessionError, SubscribeOptions};
use bytes::Bytes;
use rlmesh_session_core::crypto::sealed::unverified_body;
use tracing::{debug, warn};

/// One recorded topic, as the writer opens it.
#[derive(Debug)]
pub(super) struct Topic {
    /// The session channel id, or `None` for a `_session` topic.
    pub(super) key: TopicKey,
    pub(super) name: String,
    pub(super) schema: &'static str,
    pub(super) metadata: BTreeMap<String, String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum TopicKey {
    Stream(ChannelId),
    /// A teed stream, per lane (the lockstep path).
    Lane(ChannelId, u32),
    Chunks(ChannelId),
    Authority,
    Bindings,
    Health,
}

pub(super) enum Entry {
    Message {
        topic: Arc<Topic>,
        log_time: i64,
        publish_time: i64,
        sequence: u64,
        data: Vec<u8>,
    },
    /// Something worth keeping happened (rolling capture only).
    Incident {
        label: &'static str,
        log_time: i64,
    },
    Finish {
        metadata: Metadata,
    },
}

/// How many payload bytes may wait for the writer: past it, messages are
/// dropped and counted, so a disk slower than the session never grows the
/// host's memory without bound.
pub const QUEUE_BYTES: usize = 256 << 20;

/// The writer's queue (see [`QUEUE_BYTES`]).
#[derive(Clone)]
pub(super) struct Tx {
    tx: mpsc::Sender<Entry>,
    pub(super) queued: Arc<AtomicUsize>,
    pub(super) dropped: Arc<AtomicU64>,
}

impl Tx {
    fn send(&self, entry: Entry) -> Result<(), mpsc::SendError<Entry>> {
        if let Entry::Message { data, topic, .. } = &entry {
            let len = data.len();
            if self.queued.fetch_add(len, Ordering::Relaxed) + len > QUEUE_BYTES {
                self.queued.fetch_sub(len, Ordering::Relaxed);
                if self.dropped.fetch_add(1, Ordering::Relaxed) == 0 {
                    warn!(
                        topic = topic.name,
                        "recording fell behind: dropping messages"
                    );
                }
                return Ok(());
            }
        }
        self.tx.send(entry)
    }
}

/// Stamps recorder-side receipt times (session time).
#[derive(Clone)]
struct Stamp {
    participant: Arc<Participant>,
}

impl Stamp {
    fn now(&self) -> i64 {
        self.participant.session_now_ns()
    }

    /// `t_ns` on `participant`'s clock in session time (unchanged when the
    /// host has no mapping for it yet).
    fn to_session(&self, participant: &ParticipantId, t_ns: i64) -> i64 {
        let offset = self
            .participant
            .authority_view()
            .with(|view| view.and_then(|v| v.clock(participant)).map(|c| c.offset_ns))
            .unwrap_or(0);
        t_ns.saturating_add(offset)
    }
}

/// What a new authority snapshot shows that is worth keeping, against the
/// one before it.
fn incident(previous: &AuthoritySnapshot, now: &AuthoritySnapshot) -> Option<&'static str> {
    if (now.estop_latched && !previous.estop_latched)
        || (now.mode == Mode::Estop && previous.mode != Mode::Estop)
    {
        return Some("estop");
    }
    if now.mode == Mode::Safe && previous.mode != Mode::Safe {
        return Some("safe");
    }
    let preempted = now.leases.iter().any(|lease| {
        previous.leases.iter().any(|old| {
            old.channel == lease.channel
                && old.holder != lease.holder
                && lease.priority > old.priority
        })
    });
    preempted.then_some("preempted")
}

/// Missed edge deadlines within [`MISS_WINDOW`] that make an incident.
const MISS_BURST: usize = 5;
/// See [`MISS_BURST`].
const MISS_WINDOW: Duration = Duration::from_secs(10);

/// One `_applied` stream's missed deadlines.
#[derive(Default)]
struct Misses {
    /// The previous tick's seq and capture time.
    last: Option<(u64, i64)>,
    /// When each recent miss was recorded.
    recent: std::collections::VecDeque<i64>,
}

/// A session recording in progress: see the crate docs.
pub struct Recorder {
    participant: Arc<Participant>,
    tx: Tx,
    writer: Option<JoinHandle<Result<RecordStats, RecordError>>>,
    watcher: Option<tokio::task::JoinHandle<Subscriptions>>,
    cancel: tokio::sync::watch::Sender<bool>,
    /// Authority.
    subscribers: Vec<zenoh::pubsub::Subscriber<()>>,
    _health: HealthSubscription,
    /// Every robot declared while recording, as metadata records.
    robots: Arc<Mutex<Metadata>>,
}

/// MCAP metadata records, by name.
pub(super) type Metadata = BTreeMap<String, BTreeMap<String, String>>;

/// Everything the watcher subscribed to; dropped to stop recording.
#[derive(Default)]
struct Subscriptions {
    streams: Vec<SampleSubscriber>,
    chunks: Vec<zenoh::pubsub::Subscriber<()>>,
}

impl Recorder {
    /// Starts recording the session `participant` is attached to (an
    /// observer) into a new MCAP file at `path`. Call it on a Tokio
    /// runtime: the recorder's subscriptions run there.
    pub async fn start(
        participant: Arc<Participant>,
        path: impl AsRef<Path>,
    ) -> Result<Self, RecordError> {
        let segment = Segment::create(path.as_ref(), 0)?;
        Self::start_with(participant, Sink::File(Box::new(segment))).await
    }

    /// Starts always-on recording of the session `participant` is attached
    /// to (an observer): segments of [`Rolling::segment`] under
    /// `{dir}/rolling/{session}/`, each opening with the newest authority
    /// snapshot so it replays on its own; the oldest segments of every
    /// session go when the budgets are exceeded. Below the free-space floor
    /// it stops writing (counted in [`RecordStats::paused_drops`]), reports
    /// itself degraded in its heartbeats and retries every segment period.
    pub async fn start_rolling(
        participant: Arc<Participant>,
        rolling: Rolling,
        session: SessionId,
    ) -> Result<Self, RecordError> {
        let dir = rolling.session_dir(session);
        std::fs::create_dir_all(&dir).map_err(|source| RecordError::Io {
            path: dir.clone(),
            source,
        })?;
        let sink = RollingSink {
            rolling,
            dir,
            session,
            participant: participant.clone(),
            current: None,
            retry_at: i64::MIN,
            authority: None,
            paused: false,
            keeping: Vec::new(),
        };
        Self::start_with(participant, Sink::Rolling(Box::new(sink))).await
    }

    async fn start_with(participant: Arc<Participant>, sink: Sink) -> Result<Self, RecordError> {
        let (tx, rx) = mpsc::channel();
        let tx = Tx {
            tx,
            queued: Arc::default(),
            dropped: Arc::default(),
        };
        let writer = std::thread::Builder::new()
            .name("rlmesh-record".into())
            .spawn({
                let tx = tx.clone();
                move || write_loop(sink, &rx, &tx)
            })
            .map_err(|source| RecordError::Io {
                path: PathBuf::new(),
                source,
            })?;
        let stamp = Stamp {
            participant: participant.clone(),
        };

        // Authority: every published snapshot, as published.
        let robots: Arc<Mutex<Metadata>> = Arc::default();
        let authority = {
            let robots = robots.clone();
            let topic = Arc::new(Topic {
                key: TopicKey::Authority,
                name: "/_session/authority".into(),
                schema: schema::AUTHORITY_STATE,
                metadata: BTreeMap::new(),
            });
            let bindings = Arc::new(Topic {
                key: TopicKey::Bindings,
                name: "/_session/bindings".into(),
                schema: schema::BINDING_RECORD,
                metadata: BTreeMap::new(),
            });
            let seen: Mutex<BTreeMap<ParticipantId, (Fingerprint, u32)>> =
                Mutex::new(BTreeMap::new());
            let previous: Mutex<Option<AuthoritySnapshot>> = Mutex::new(None);
            // Records one snapshot: a repeat of the last (the seed below
            // and the update it raced) is skipped.
            let record = {
                let tx = tx.clone();
                Arc::new(
                    move |state: AuthoritySnapshot, data: Vec<u8>, log_time: i64| {
                        let mut previous = previous
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        if previous
                            .as_ref()
                            .is_some_and(|p| p.revision >= state.revision)
                        {
                            return;
                        }
                        let _ = tx.send(Entry::Message {
                            topic: topic.clone(),
                            log_time,
                            publish_time: state.host_ns,
                            sequence: state.revision,
                            data,
                        });
                        if let Some(label) = previous.as_ref().and_then(|p| incident(p, &state)) {
                            let _ = tx.send(Entry::Incident { label, log_time });
                        }
                        *previous = Some(state.clone());
                        drop(previous);
                        robots
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .extend(robot_metadata(&state));
                        let mut seen = seen
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        for binding in &state.bindings {
                            let current = (binding.fingerprint, binding.revision);
                            if seen.get(&binding.participant) == Some(&current) {
                                continue;
                            }
                            seen.insert(binding.participant.clone(), current);
                            let _ = tx.send(Entry::Message {
                                topic: bindings.clone(),
                                log_time,
                                publish_time: state.host_ns,
                                sequence: u64::from(binding.revision),
                                data: codec::encode_binding(binding),
                            });
                        }
                    },
                )
            };
            // Only what the host signed is recorded, as its plain snapshot.
            let verifier = Mutex::new(
                participant
                    .host_verifier()
                    .ok_or_else(|| RecordError::Session(SessionError::Closed))?,
            );
            let key = participant.keys().host_state();
            let subscriber = {
                let (record, stamp) = (record.clone(), stamp.clone());
                participant
                    .transport()
                    .control()
                    .declare_subscriber(key.clone())
                    .callback(move |sample| {
                        let log_time = stamp.now();
                        let signed = sample.payload().to_bytes();
                        let data = match verifier
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .verify(&key, &signed)
                        {
                            Ok(verified) => verified.body.to_vec(),
                            Err(err) => {
                                debug!(%err, "authority update dropped");
                                return;
                            }
                        };
                        let Ok(state) = codec::decode_state(&data) else {
                            debug!("undecodable authority update");
                            return;
                        };
                        record(state, data, log_time);
                    })
                    .await
                    .map_err(SessionError::from)?
            };
            // The snapshot this participant already holds (from its attach),
            // so a recording opens with the authority even when it starts
            // mid-session and nothing changes for a while.
            if let Some(state) = participant.authority() {
                let data = codec::encode_state(&state);
                record(state, data, stamp.now());
            }
            subscriber
        };
        // Health: what the host relays of every link and edge, verified.
        let health = {
            let topic = Arc::new(Topic {
                key: TopicKey::Health,
                name: "/_session/health".into(),
                schema: schema::HOST_HEALTH,
                metadata: BTreeMap::new(),
            });
            let (tx, stamp) = (tx.clone(), stamp.clone());
            let sequence = AtomicU64::new(0);
            participant
                .host_health(move |health| {
                    let _ = tx.send(Entry::Message {
                        topic: topic.clone(),
                        log_time: stamp.now(),
                        publish_time: health.host_ns,
                        sequence: sequence.fetch_add(1, Ordering::Relaxed),
                        data: codec::encode_health(&health),
                    });
                })
                .await?
        };

        let (cancel, cancelled) = tokio::sync::watch::channel(false);
        let watcher = tokio::spawn(watch_parts(
            participant.clone(),
            tx.clone(),
            stamp,
            cancelled,
        ));
        Ok(Self {
            participant,
            tx,
            writer: Some(writer),
            watcher: Some(watcher),
            cancel,
            subscribers: vec![authority],
            _health: health,
            robots,
        })
    }

    /// Stops recording, writes the session metadata and finishes the file.
    pub async fn finish(mut self) -> Result<RecordStats, RecordError> {
        let _ = self.cancel.send(true);
        if let Some(watcher) = self.watcher.take() {
            drop(watcher.await);
        }
        self.subscribers.clear();
        let mut metadata = session_metadata(self.participant.authority().as_ref());
        metadata.extend(
            self.robots
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone(),
        );
        let _ = self.tx.send(Entry::Finish { metadata });
        let writer = self.writer.take();
        match writer {
            Some(writer) => tokio::task::spawn_blocking(move || writer.join())
                .await
                .map_err(|e| RecordError::Writer(e.to_string()))?
                .map_err(|_| RecordError::Writer("the writer thread panicked".into()))?,
            None => Err(RecordError::Writer("already finished".into())),
        }
    }

    /// The observer participant the recorder attached as.
    pub fn participant(&self) -> &Arc<Participant> {
        &self.participant
    }

    /// A handle that writes samples into this recording from inside the
    /// process: the lockstep path publishes no streams, so its env tees
    /// each step's slabs here, per lane.
    pub fn tee(&self) -> SampleTee {
        SampleTee {
            tx: self.tx.clone(),
            stamp: Stamp {
                participant: self.participant.clone(),
            },
            topics: Arc::default(),
        }
    }
}

/// Writes `Sample` envelopes into a recording directly (see
/// [`Recorder::tee`]). Cheap to clone.
#[derive(Clone)]
pub struct SampleTee {
    tx: Tx,
    stamp: Stamp,
    topics: Arc<Mutex<LaneTopics>>,
}

/// The teed topics, by channel and lane.
type LaneTopics = BTreeMap<(ChannelId, u32), Arc<Topic>>;

impl SampleTee {
    /// Records `sample` of `part` on lane `lane` of `lanes`: on
    /// `/{robot}/{path}` (or `{path}/_applied` when `applied`), with a
    /// `lane{i}` segment after the robot when there are several lanes.
    pub fn record(&self, part: &PartSpec, lane: u32, lanes: u32, applied: bool, sample: &Sample) {
        let channel = sample.channel;
        let topic = {
            let mut topics = self
                .topics
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            topics
                .entry((channel, lane))
                .or_insert_with(|| {
                    let lane_segment = if lanes > 1 {
                        format!("lane{lane}/")
                    } else {
                        String::new()
                    };
                    let suffix = if applied { "/_applied" } else { "" };
                    Arc::new(Topic {
                        key: TopicKey::Lane(channel, lane),
                        name: format!("/{}/{lane_segment}{}{suffix}", part.owner, part.path),
                        schema: schema::SAMPLE,
                        metadata: BTreeMap::from([
                            ("channel".to_string(), channel.0.to_string()),
                            ("kind".to_string(), part.kind.clone()),
                            ("lane".to_string(), lane.to_string()),
                        ]),
                    })
                })
                .clone()
        };
        let _ = self.tx.send(Entry::Message {
            topic,
            log_time: self.stamp.now(),
            publish_time: sample.capture_ns,
            sequence: sample.seq,
            data: codec::encode_sample(sample).to_vec(),
        });
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        let _ = self.cancel.send(true);
        if self.writer.is_some() {
            let _ = self.tx.send(Entry::Finish {
                metadata: Metadata::new(),
            });
        }
    }
}

/// Subscribes to every stream and chunk key as parts appear, until
/// cancelled; returns the subscriptions so the caller drops them.
async fn watch_parts(
    participant: Arc<Participant>,
    tx: Tx,
    stamp: Stamp,
    mut cancelled: tokio::sync::watch::Receiver<bool>,
) -> Subscriptions {
    let mut subs = Subscriptions::default();
    let mut streams: BTreeSet<ChannelId> = BTreeSet::new();
    let mut chunks: BTreeSet<ChannelId> = BTreeSet::new();
    let mut view = participant.watch();
    loop {
        let snapshot = view.borrow_and_update().clone();
        if let Some(snapshot) = snapshot {
            for part in &snapshot.parts {
                let mut wanted: Vec<StreamSpec> = Vec::with_capacity(2);
                if let Ok(stream) = part.stream() {
                    wanted.push(stream);
                }
                if let Some(Ok(applied)) = part.applied_stream() {
                    wanted.push(applied);
                }
                for stream in wanted {
                    if !streams.insert(stream.channel) {
                        continue;
                    }
                    match subscribe_stream(&participant, &tx, &stamp, &snapshot, stream).await {
                        Ok(sub) => subs.streams.push(sub),
                        Err(err) => warn!(%err, "recorder: subscription refused"),
                    }
                }
                if part.command.is_some() && chunks.insert(part.channel) {
                    let topic = Arc::new(Topic {
                        key: TopicKey::Chunks(part.channel),
                        name: format!("/{}/{}/_chunks", part.owner, part.path),
                        schema: schema::ACTION_CHUNK,
                        metadata: BTreeMap::from([
                            ("channel".into(), part.channel.0.to_string()),
                            ("kind".into(), part.kind.clone()),
                        ]),
                    });
                    let tx = tx.clone();
                    let stamp = stamp.clone();
                    let key = participant.keys().commands(&part.owner, part.path.as_str());
                    let declared = participant
                        .transport()
                        .session(TrafficClass::of(part.priority))
                        .declare_subscriber(key)
                        .callback(move |sample| {
                            let log_time = stamp.now();
                            // The recorder holds no command key: it records
                            // a signed chunk's body as sent, unverified, and
                            // can't read a sealed one. Read permission
                            // doesn't change that: `k_cmd` goes only to the
                            // robot and the host.
                            // TODO: record verified chunks when the host
                            // relays what its edges accepted (the heartbeat
                            // relay above would carry it).
                            let data = payload_bytes(sample.payload());
                            if let Some(body) = unverified_body(&data) {
                                on_chunk(&tx, &stamp, &topic, log_time, data.slice(body));
                            }
                        })
                        .await;
                    match declared {
                        Ok(sub) => subs.chunks.push(sub),
                        Err(err) => warn!(%err, "recorder: chunk subscription refused"),
                    }
                }
            }
        }
        tokio::select! {
            changed = view.changed() => if changed.is_err() { return subs },
            _ = cancelled.changed() => return subs,
        }
    }
}

fn on_chunk(tx: &Tx, stamp: &Stamp, topic: &Arc<Topic>, log_time: i64, data: Bytes) {
    let Ok(chunk) = codec::decode_chunk(data.clone()) else {
        debug!(topic = topic.name, "undecodable chunk");
        return;
    };
    let _ = tx.send(Entry::Message {
        topic: topic.clone(),
        log_time,
        publish_time: stamp.to_session(&chunk.source, chunk.publish_ns),
        sequence: chunk.seq,
        data: data.to_vec(),
    });
}

async fn subscribe_stream(
    participant: &Arc<Participant>,
    tx: &Tx,
    stamp: &Stamp,
    snapshot: &AuthoritySnapshot,
    stream: StreamSpec,
) -> Result<SampleSubscriber, SessionError> {
    let part = snapshot
        .parts
        .iter()
        .find(|p| p.channel == stream.channel || p.applied_channel == Some(stream.channel));
    let mut metadata = BTreeMap::from([
        ("channel".to_string(), stream.channel.0.to_string()),
        (
            "encoding".to_string(),
            stream.encoding.wire_name().to_string(),
        ),
        (
            "fields".to_string(),
            stream
                .fields
                .iter()
                .map(|f| format!("{}:{}{:?}", f.name, f.dtype.name(), f.shape))
                .collect::<Vec<_>>()
                .join(","),
        ),
    ]);
    if let Some(part) = part {
        metadata.insert("kind".into(), part.kind.clone());
        metadata.insert("hz".into(), part.hz.to_string());
    }
    // An `_applied` stream's consecutive ticks more than two control
    // periods apart are missed deadlines (as the cockpit counts them).
    let deadline = part
        .filter(|p| p.applied_channel == Some(stream.channel))
        .map(|p| 2 * i64::try_from(p.control_period().as_nanos()).unwrap_or(i64::MAX));
    let misses: Mutex<Misses> = Mutex::default();
    let topic = Arc::new(Topic {
        key: TopicKey::Stream(stream.channel),
        name: format!("/{}/{}", stream.owner, stream.path),
        schema: schema::SAMPLE,
        metadata,
    });
    let on_sample = {
        let tx = tx.clone();
        let stamp = stamp.clone();
        Arc::new(move |received: &crate::Received| {
            let p = &stamp.participant;
            let log_time = p.to_session_ns(received.receive_ns);
            let _ = tx.send(Entry::Message {
                topic: topic.clone(),
                log_time,
                publish_time: p.to_session_ns(received.capture_local_ns),
                sequence: received.sample.seq,
                data: codec::encode_sample(&received.sample).to_vec(),
            });
            let Some(deadline) = deadline else {
                return;
            };
            let sample = &received.sample;
            let mut misses = misses
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let Misses { last, recent } = &mut *misses;
            if let Some((seq, at)) = *last
                && sample.seq == seq + 1
                && sample.capture_ns - at > deadline
            {
                recent.push_back(log_time);
            }
            *last = Some((sample.seq, sample.capture_ns));
            let window = i64::try_from(MISS_WINDOW.as_nanos()).unwrap_or(i64::MAX);
            while recent.front().is_some_and(|&t| t < log_time - window) {
                recent.pop_front();
            }
            if recent.len() >= MISS_BURST {
                recent.clear();
                let _ = tx.send(Entry::Incident {
                    label: "deadline-misses",
                    log_time,
                });
            }
        })
    };
    participant
        .subscribe_stream(
            stream,
            SubscribeOptions {
                on_sample: Some(on_sample),
                window: 0,
                warm_from_history: false,
            },
        )
        .await
}

fn session_metadata(snapshot: Option<&AuthoritySnapshot>) -> Metadata {
    let Some(snapshot) = snapshot else {
        return Metadata::new();
    };
    Metadata::from([(
        "rlmesh.session".to_string(),
        BTreeMap::from([
            ("session".to_string(), snapshot.session.to_string()),
            ("epoch".to_string(), snapshot.epoch.0.to_string()),
            (
                "protocol_minor".to_string(),
                snapshot.protocol_minor.to_string(),
            ),
        ]),
    )])
}

/// One `rlmesh.robot.{robot}` record per declared robot: its declaration
/// fingerprint and its parts (`path:kind:channel`).
fn robot_metadata(snapshot: &AuthoritySnapshot) -> Metadata {
    snapshot
        .contract
        .iter()
        .map(|(robot, fingerprint)| {
            let parts: Vec<String> = snapshot
                .parts
                .iter()
                .filter(|p| &p.owner == robot)
                .map(|p| format!("{}:{}:{}", p.path, p.kind, p.channel.0))
                .collect();
            (
                format!("rlmesh.robot.{robot}"),
                BTreeMap::from([
                    ("fingerprint".to_string(), fingerprint.to_string()),
                    ("parts".to_string(), parts.join(",")),
                ]),
            )
        })
        .collect()
}

pub(super) fn ns(t: i64) -> u64 {
    u64::try_from(t).unwrap_or(0)
}
