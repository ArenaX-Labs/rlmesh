//! MCAP recording of a real-time RLMesh session (protocol v2, the v0.2
//! spec §13).
//!
//! A [`Recorder`] is an observer participant: it subscribes to every stream
//! of every robot (parts and their `_applied` streams), to every commandable
//! part's action chunks and to the host's authority snapshots, and writes
//! each envelope unchanged into an MCAP file:
//!
//! | Topic | Schema | One message per |
//! |---|---|---|
//! | `/{robot}/{path}` | `rlmesh.session.v2.Sample` | sample (a batch is split) |
//! | `/{robot}/{path}/_applied` | `rlmesh.session.v2.Sample` | edge control tick |
//! | `/{robot}/{path}/_chunks` | `rlmesh.session.v2.ActionChunk` | chunk sent to the edge |
//! | `/_session/authority` | `rlmesh.session.v2.AuthorityState` | authority change |
//! | `/_session/bindings` | `rlmesh.session.v2.BindingRecord` | new or changed binding |
//!
//! Schemas are the `rlmesh.session.v2` protobuf descriptors. A new MCAP
//! channel opens for every session channel id. `publish_time` is the
//! capture (or send) time mapped to session time, `log_time` session time
//! at the recorder, and the MCAP sequence the low 32 bits of the envelope's
//! seq (the full u64 stays in the envelope).
//!
//! The recorder records what the session publishes: it declares no delivery
//! contract, so a stream a policy reads at `latest@R` is recorded at R
//! (§8.2), while `_applied` and chunks are recorded in full.
//!
//! [`read`] decodes a recording back into its messages in log-time order
//! (the cockpit's replay). [`summarize`] checks one: message counts per
//! topic, time ordering, envelopes that do not decode, and the lease
//! timeline.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::transport::{TrafficClass, payload_bytes};
use crate::wire::codec::{self, schema};
use crate::wire::{
    ActionChunk, AuthoritySnapshot, ChannelId, Fingerprint, PartSpec, ParticipantId, Sample,
    SessionId, StreamSpec,
};
use crate::{Participant, SampleSubscriber, SessionError, SubscribeOptions};
use bytes::Bytes;
use tracing::{debug, info, warn};

/// What can go wrong recording or reading a recording.
#[derive(Debug, thiserror::Error)]
pub enum RecordError {
    /// The file could not be created, written or read.
    #[error("recording {path}: {source}")]
    Io {
        /// The file.
        path: PathBuf,
        /// Why.
        source: std::io::Error,
    },
    /// The MCAP writer or reader failed.
    #[error("mcap: {0}")]
    Mcap(#[from] mcap::McapError),
    /// The session refused a subscription.
    #[error(transparent)]
    Session(#[from] SessionError),
    /// The writer thread is gone.
    #[error("the recording writer stopped: {0}")]
    Writer(String),
}

/// What a finished recording holds.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RecordStats {
    /// The file (rolling: the last segment).
    pub path: PathBuf,
    /// Files written (rolling: segments).
    pub segments: u32,
    /// Messages not written while rolling capture was paused for disk
    /// space.
    pub paused_drops: u64,
    /// MCAP messages written.
    pub messages: u64,
    /// MCAP channels opened.
    pub channels: u32,
    /// Payload bytes written (envelopes).
    pub bytes: u64,
    /// Messages whose `log_time` was raised to their channel's previous one
    /// (the recorder's session clock stepped back on a remapping).
    pub log_time_clamped: u64,
}

/// One recorded topic, as the writer opens it.
#[derive(Debug)]
struct Topic {
    /// The session channel id, or `None` for a `_session` topic.
    key: TopicKey,
    name: String,
    schema: &'static str,
    metadata: BTreeMap<String, String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum TopicKey {
    Stream(ChannelId),
    /// A teed stream, per lane (the lockstep path).
    Lane(ChannelId, u32),
    Chunks(ChannelId),
    Authority,
    Bindings,
}

enum Entry {
    Message {
        topic: Arc<Topic>,
        log_time: i64,
        publish_time: i64,
        sequence: u64,
        data: Vec<u8>,
    },
    Finish {
        metadata: Metadata,
    },
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
}

/// A session recording in progress: see the crate docs.
pub struct Recorder {
    participant: Arc<Participant>,
    tx: mpsc::Sender<Entry>,
    writer: Option<JoinHandle<Result<RecordStats, RecordError>>>,
    watcher: Option<tokio::task::JoinHandle<Subscriptions>>,
    cancel: tokio::sync::watch::Sender<bool>,
    authority: Option<zenoh::pubsub::Subscriber<()>>,
    /// Every robot declared while recording, as metadata records.
    robots: Arc<Mutex<Metadata>>,
}

/// MCAP metadata records, by name.
type Metadata = BTreeMap<String, BTreeMap<String, String>>;

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
        Self::start_with(participant, Sink::File(segment)).await
    }

    /// Starts always-on recording of the session `participant` is attached
    /// to (an observer): segments of [`Rolling::segment`] under
    /// `{dir}/rolling/{session}/`, each opening with the newest authority
    /// snapshot so it replays on its own; the oldest segments of every
    /// session go when the budgets are exceeded. Below the free-space floor
    /// it stops writing (counted in [`RecordStats::paused_drops`]) and
    /// retries every segment period.
    pub async fn start_rolling(
        participant: Arc<Participant>,
        rolling: Rolling,
        session: SessionId,
    ) -> Result<Self, RecordError> {
        let dir = rolling.dir.join("rolling").join(session.to_string());
        std::fs::create_dir_all(&dir).map_err(|source| RecordError::Io {
            path: dir.clone(),
            source,
        })?;
        let sink = RollingSink {
            rolling,
            dir,
            current: None,
            retry_at: i64::MIN,
            authority: None,
            paused: false,
        };
        Self::start_with(participant, Sink::Rolling(sink)).await
    }

    async fn start_with(participant: Arc<Participant>, sink: Sink) -> Result<Self, RecordError> {
        let (tx, rx) = mpsc::channel();
        let writer = std::thread::Builder::new()
            .name("rlmesh-record".into())
            .spawn(move || write_loop(sink, &rx))
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
            let tx = tx.clone();
            let stamp = stamp.clone();
            participant
                .transport()
                .control()
                .declare_subscriber(participant.keys().host_state())
                .callback(move |sample| {
                    let log_time = stamp.now();
                    let data = sample.payload().to_bytes().into_owned();
                    let Ok(state) = codec::decode_state(&data) else {
                        debug!("undecodable authority update");
                        return;
                    };
                    let _ = tx.send(Entry::Message {
                        topic: topic.clone(),
                        log_time,
                        publish_time: state.host_ns,
                        sequence: state.revision,
                        data,
                    });
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
                })
                .await
                .map_err(SessionError::from)?
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
            authority: Some(authority),
            robots,
        })
    }

    /// Stops recording, writes the session metadata and finishes the file.
    pub async fn finish(mut self) -> Result<RecordStats, RecordError> {
        let _ = self.cancel.send(true);
        if let Some(watcher) = self.watcher.take() {
            drop(watcher.await);
        }
        drop(self.authority.take());
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
    /// each step's slabs here, per lane (§13).
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
    tx: mpsc::Sender<Entry>,
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
    tx: mpsc::Sender<Entry>,
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
                            let data = payload_bytes(sample.payload());
                            on_chunk(&tx, &stamp, &topic, log_time, data);
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

fn on_chunk(
    tx: &mpsc::Sender<Entry>,
    stamp: &Stamp,
    topic: &Arc<Topic>,
    log_time: i64,
    data: Bytes,
) {
    let Ok(chunk) = codec::decode_chunk(data.clone()) else {
        debug!(topic = topic.name, "undecodable chunk");
        return;
    };
    let offset = stamp
        .participant
        .authority_view()
        .with(|view| {
            view.and_then(|v| v.clock(&chunk.source))
                .map(|c| c.offset_ns)
        })
        .unwrap_or(0);
    let _ = tx.send(Entry::Message {
        topic: topic.clone(),
        log_time,
        publish_time: chunk.publish_ns.saturating_add(offset),
        sequence: chunk.seq,
        data: data.to_vec(),
    });
}

async fn subscribe_stream(
    participant: &Arc<Participant>,
    tx: &mpsc::Sender<Entry>,
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
            let _ = tx.send(Entry::Message {
                topic: topic.clone(),
                log_time: p.to_session_ns(received.receive_ns),
                publish_time: p.to_session_ns(received.capture_local_ns),
                sequence: received.sample.seq,
                data: codec::encode_sample(&received.sample).to_vec(),
            });
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

fn ns(t: i64) -> u64 {
    u64::try_from(t).unwrap_or(0)
}

fn write_loop(mut sink: Sink, rx: &mpsc::Receiver<Entry>) -> Result<RecordStats, RecordError> {
    let mut stats = RecordStats::default();
    let mut metadata = Metadata::new();
    while let Ok(entry) = rx.recv() {
        match entry {
            Entry::Message {
                topic,
                log_time,
                publish_time,
                sequence,
                data,
            } => {
                let message = Message {
                    topic,
                    log_time,
                    publish_time,
                    sequence,
                    data,
                };
                match &mut sink {
                    Sink::File(segment) => segment.write(&message, &mut stats)?,
                    Sink::Rolling(rolling) => rolling.write(message, &mut stats)?,
                }
            }
            Entry::Finish { metadata: m } => {
                metadata = m;
                break;
            }
        }
    }
    match sink {
        Sink::File(segment) => segment.finish(&metadata, &mut stats)?,
        Sink::Rolling(rolling) => {
            if let Some(segment) = rolling.current {
                segment.finish(&metadata, &mut stats)?;
            }
        }
    }
    Ok(stats)
}

/// Where the writer thread puts messages.
enum Sink {
    File(Segment),
    Rolling(RollingSink),
}

/// A queued message.
struct Message {
    topic: Arc<Topic>,
    log_time: i64,
    publish_time: i64,
    sequence: u64,
    data: Vec<u8>,
}

/// One MCAP file being written.
struct Segment {
    writer: mcap::Writer<BufWriter<File>>,
    path: PathBuf,
    /// The log time it was opened at.
    started: i64,
    schemas: BTreeMap<&'static str, u16>,
    /// MCAP channel and its newest log time, by topic.
    channels: BTreeMap<TopicKey, (u16, u64)>,
}

impl Segment {
    fn create(path: &Path, started: i64) -> Result<Self, RecordError> {
        let file = File::create(path).map_err(|source| RecordError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let writer = mcap::WriteOptions::new()
            .compression(None)
            .profile("rlmesh")
            .library(concat!("rlmesh-session ", env!("CARGO_PKG_VERSION")))
            .create(BufWriter::new(file))?;
        Ok(Self {
            writer,
            path: path.to_path_buf(),
            started,
            schemas: BTreeMap::new(),
            channels: BTreeMap::new(),
        })
    }

    fn write(&mut self, m: &Message, stats: &mut RecordStats) -> Result<(), RecordError> {
        let (channel_id, last_log) = match self.channels.get(&m.topic.key) {
            Some(c) => *c,
            None => {
                let schema_id = match self.schemas.get(m.topic.schema) {
                    Some(id) => *id,
                    None => {
                        let id = self.writer.add_schema(
                            m.topic.schema,
                            "protobuf",
                            codec::FILE_DESCRIPTOR_SET,
                        )?;
                        self.schemas.insert(m.topic.schema, id);
                        id
                    }
                };
                let id = self.writer.add_channel(
                    schema_id,
                    &m.topic.name,
                    "protobuf",
                    &m.topic.metadata,
                )?;
                stats.channels += 1;
                (id, 0)
            }
        };
        let mut log_time = ns(m.log_time);
        if log_time < last_log {
            log_time = last_log;
            stats.log_time_clamped += 1;
        }
        self.channels.insert(m.topic.key, (channel_id, log_time));
        self.writer.write_to_known_channel(
            &mcap::records::MessageHeader {
                channel_id,
                // The low 32 bits; the envelope keeps the full seq.
                sequence: u32::try_from(m.sequence & u64::from(u32::MAX)).unwrap_or(0),
                log_time,
                publish_time: ns(m.publish_time),
            },
            &m.data,
        )?;
        stats.messages += 1;
        stats.bytes += m.data.len() as u64;
        Ok(())
    }

    fn finish(mut self, metadata: &Metadata, stats: &mut RecordStats) -> Result<(), RecordError> {
        for (name, metadata) in metadata {
            self.writer.write_metadata(&mcap::records::Metadata {
                name: name.clone(),
                metadata: metadata.clone(),
            })?;
        }
        self.writer.finish()?;
        stats.segments += 1;
        stats.path = self.path;
        Ok(())
    }
}

/// Budgets for always-on recording ([`Recorder::start_rolling`]).
#[derive(Clone, Debug)]
pub struct Rolling {
    /// The recordings root; segments go under `rolling/{session}/`.
    pub dir: PathBuf,
    /// How much session time one segment covers.
    pub segment: Duration,
    /// Segments older than this go.
    pub max_age: Duration,
    /// The oldest segments go while all of them together exceed this.
    pub max_bytes: u64,
    /// Below this much free space on the disk, capture pauses.
    pub min_free_bytes: u64,
}

impl Rolling {
    /// The defaults (newest 30 min or 20 GB, one-minute segments, pause
    /// below 2 GB free) under `dir`.
    // TODO: fixed budgets; add per-host profiles (smaller budgets, lower
    // camera rate) when a low-storage host needs them.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            segment: Duration::from_secs(60),
            max_age: Duration::from_secs(30 * 60),
            max_bytes: 20_000_000_000,
            min_free_bytes: 2_000_000_000,
        }
    }
}

/// Rolling capture's writer side.
struct RollingSink {
    rolling: Rolling,
    /// This session's segments.
    dir: PathBuf,
    current: Option<Segment>,
    /// While paused, the log time to try opening a segment again.
    retry_at: i64,
    /// The newest authority snapshot, written first into every segment.
    authority: Option<Message>,
    paused: bool,
}

impl RollingSink {
    fn write(&mut self, m: Message, stats: &mut RecordStats) -> Result<(), RecordError> {
        let segment = i64::try_from(self.rolling.segment.as_nanos()).unwrap_or(i64::MAX);
        let due = match &self.current {
            Some(current) => m.log_time >= current.started.saturating_add(segment),
            None => m.log_time >= self.retry_at,
        };
        if due {
            self.rotate(m.log_time, stats)?;
            if self.current.is_none() {
                self.retry_at = m.log_time.saturating_add(segment);
            }
        }
        match &mut self.current {
            Some(current) => current.write(&m, stats)?,
            None => stats.paused_drops += 1,
        }
        if m.topic.key == TopicKey::Authority {
            self.authority = Some(m);
        }
        Ok(())
    }

    /// Closes the current segment, applies the budgets and opens the next
    /// one, unless the disk is below the floor.
    fn rotate(&mut self, log_time: i64, stats: &mut RecordStats) -> Result<(), RecordError> {
        if let Some(current) = self.current.take() {
            current.finish(&Metadata::new(), stats)?;
        }
        purge(&self.rolling);
        if free_bytes(&self.dir).is_some_and(|free| free < self.rolling.min_free_bytes) {
            if !self.paused {
                warn!(dir = %self.dir.display(), "recording paused: less than {} bytes free", self.rolling.min_free_bytes);
            }
            self.paused = true;
            return Ok(());
        }
        if self.paused {
            info!(dir = %self.dir.display(), "recording resumed");
            self.paused = false;
        }
        let wall_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis());
        let path = self.dir.join(format!("{wall_ms}.mcap"));
        // The purge removes a session directory it empties, this one too.
        let created = std::fs::create_dir_all(&self.dir)
            .map_err(|source| RecordError::Io {
                path: self.dir.clone(),
                source,
            })
            .and_then(|()| Segment::create(&path, log_time));
        let mut segment = match created {
            Ok(segment) => segment,
            Err(err) => {
                warn!(%err, "recording paused: cannot open a segment");
                self.paused = true;
                return Ok(());
            }
        };
        if let Some(authority) = &self.authority {
            segment.write(authority, stats)?;
        }
        self.current = Some(segment);
        Ok(())
    }
}

/// Removes the oldest rolling segments, of every session under the root,
/// past the age and size budgets. Racing purgers (one per session) are
/// fine: a segment already gone is skipped.
fn purge(rolling: &Rolling) {
    let mut segments: Vec<(std::time::SystemTime, u64, PathBuf)> = Vec::new();
    let sessions = std::fs::read_dir(rolling.dir.join("rolling"))
        .into_iter()
        .flatten();
    for session in sessions.flatten() {
        for file in std::fs::read_dir(session.path())
            .into_iter()
            .flatten()
            .flatten()
        {
            let path = file.path();
            if path.extension().is_some_and(|e| e == "mcap")
                && let Ok(meta) = file.metadata()
                && let Ok(modified) = meta.modified()
            {
                segments.push((modified, meta.len(), path));
            }
        }
    }
    segments.sort();
    let mut total: u64 = segments.iter().map(|(_, len, _)| len).sum();
    let now = std::time::SystemTime::now();
    for (modified, len, path) in segments {
        let old = now
            .duration_since(modified)
            .is_ok_and(|age| age > rolling.max_age);
        if !old && total <= rolling.max_bytes {
            break;
        }
        match std::fs::remove_file(&path) {
            Ok(()) => total = total.saturating_sub(len),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => warn!(%err, path = %path.display(), "could not remove an old segment"),
        }
        if let Some(dir) = path.parent() {
            // Only succeeds once the session's last segment is gone.
            let _ = std::fs::remove_dir(dir);
        }
    }
}

/// Free bytes on the disk holding `dir`, for an unprivileged user.
#[cfg(unix)]
#[allow(unsafe_code, clippy::useless_conversion)]
fn free_bytes(dir: &Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let path = std::ffi::CString::new(dir.as_os_str().as_bytes()).ok()?;
    // SAFETY: an all-zero statvfs is a valid value to be overwritten.
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: `path` is NUL-terminated and `stat` is valid and writable.
    if unsafe { libc::statvfs(path.as_ptr(), &raw mut stat) } != 0 {
        return None;
    }
    Some(u64::from(stat.f_bavail).saturating_mul(u64::from(stat.f_frsize)))
}

// TODO: no free-space floor off unix; add GetDiskFreeSpaceExW when the
// recorder runs on Windows hosts.
#[cfg(not(unix))]
fn free_bytes(_: &Path) -> Option<u64> {
    None
}

// --- reading ------------------------------------------------------------------

/// One topic of a recording.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TopicSummary {
    /// The topic.
    pub topic: String,
    /// Its schema name.
    pub schema: String,
    /// MCAP channels on it (more than one after a schema change).
    pub channels: u32,
    /// Messages on it.
    pub messages: u64,
    /// First and last `log_time`.
    pub first_log_ns: u64,
    /// See `first_log_ns`.
    pub last_log_ns: u64,
    /// Whether `log_time` never decreases within each MCAP channel.
    pub log_time_monotonic: bool,
    /// Whether `publish_time` never decreases within each MCAP channel.
    pub publish_time_monotonic: bool,
    /// Messages whose envelope does not decode as the schema.
    pub undecodable: u64,
}

/// One lease change in the authority timeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeaseChange {
    /// When the recorder saw it (session time).
    pub log_ns: u64,
    /// The part, as `{robot}/{path}`.
    pub part: String,
    /// The new holder, `None` when the lease went away.
    pub holder: Option<String>,
    /// Its priority class (`operator`, `model`, ...).
    pub priority: Option<String>,
    /// Its lease token.
    pub token: Option<u64>,
}

/// What [`summarize`] found.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Summary {
    /// Every topic, by name.
    pub topics: Vec<TopicSummary>,
    /// Messages in the file.
    pub messages: u64,
    /// Metadata record names.
    pub metadata: Vec<String>,
    /// Lease changes, in order.
    pub leases: Vec<LeaseChange>,
}

/// Reads a recording back (the whole file, in memory).
pub fn summarize(path: impl AsRef<Path>) -> Result<Summary, RecordError> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|source| RecordError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut topics: BTreeMap<String, TopicSummary> = BTreeMap::new();
    let mut last: BTreeMap<u16, (u64, u64)> = BTreeMap::new();
    let mut summary = Summary::default();
    let mut held: BTreeMap<String, (String, String, u64)> = BTreeMap::new();
    for message in mcap::MessageStream::new(&bytes)? {
        let message = message?;
        let channel = &message.channel;
        let schema = channel
            .schema
            .as_ref()
            .map_or_else(String::new, |s| s.name.clone());
        let topic = topics
            .entry(channel.topic.clone())
            .or_insert_with(|| TopicSummary {
                topic: channel.topic.clone(),
                schema: schema.clone(),
                first_log_ns: message.log_time,
                log_time_monotonic: true,
                publish_time_monotonic: true,
                ..TopicSummary::default()
            });
        topic.messages += 1;
        topic.last_log_ns = topic.last_log_ns.max(message.log_time);
        topic.first_log_ns = topic.first_log_ns.min(message.log_time);
        match last.get(&channel.id) {
            Some((log, publish)) => {
                topic.log_time_monotonic &= message.log_time >= *log;
                topic.publish_time_monotonic &= message.publish_time >= *publish;
            }
            None => topic.channels += 1,
        }
        last.insert(channel.id, (message.log_time, message.publish_time));
        summary.messages += 1;
        let data = Bytes::copy_from_slice(&message.data);
        let decodes = match schema.as_str() {
            schema::SAMPLE => codec::decode_sample(data).is_ok(),
            schema::ACTION_CHUNK => codec::decode_chunk(data).is_ok(),
            schema::BINDING_RECORD => codec::decode_binding(&data).is_ok(),
            schema::AUTHORITY_STATE => match codec::decode_state(&data) {
                Ok(state) => {
                    lease_changes(&state, message.log_time, &mut held, &mut summary.leases);
                    true
                }
                Err(_) => false,
            },
            _ => false,
        };
        if !decodes {
            topic.undecodable += 1;
        }
    }
    summary.topics = topics.into_values().collect();
    if let Ok(Some(file)) = mcap::Summary::read(&bytes) {
        for index in &file.metadata_indexes {
            summary.metadata.push(index.name.clone());
        }
    }
    Ok(summary)
}

/// A decoded recorded envelope.
#[derive(Clone, Debug)]
pub enum Recorded {
    /// A sample of a part's stream or of an `_applied` stream.
    Sample(Sample),
    /// An action chunk on its way to an edge.
    Chunk(ActionChunk),
    /// An authority snapshot (it carries the bindings too).
    Authority(AuthoritySnapshot),
}

/// One recorded message with its MCAP times, both session time.
#[derive(Clone, Debug)]
pub struct RecordedMessage {
    /// When the recorder received it.
    pub log_ns: i64,
    /// Its capture (or send) time.
    pub publish_ns: i64,
    /// The envelope.
    pub message: Recorded,
}

/// Every sample, chunk and authority snapshot of a recording, ordered by
/// log time (stable, so each channel keeps its recorded order). Binding
/// records are skipped (the snapshots carry them), as are envelopes that do
/// not decode ([`summarize`] counts those). A file cut short (the recorder
/// crashed or lost power) yields what was written before the cut.
pub fn read(path: impl AsRef<Path>) -> Result<Vec<RecordedMessage>, RecordError> {
    let path = path.as_ref();
    // TODO: reads the whole file into memory; stream it chunk by chunk when
    // recordings outgrow RAM (long camera sessions, rolling segments).
    let bytes = std::fs::read(path).map_err(|source| RecordError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut out = Vec::new();
    for message in mcap::MessageStream::new(&bytes)? {
        let message = match message {
            Ok(message) => message,
            Err(err) if !out.is_empty() => {
                warn!(%err, path = %path.display(), "recording cut short; reading what was written");
                break;
            }
            Err(err) => return Err(err.into()),
        };
        let data = Bytes::copy_from_slice(&message.data);
        let schema = message.channel.schema.as_ref().map(|s| s.name.as_str());
        let decoded = match schema {
            Some(schema::SAMPLE) => codec::decode_sample(data).ok().map(Recorded::Sample),
            Some(schema::ACTION_CHUNK) => codec::decode_chunk(data).ok().map(Recorded::Chunk),
            Some(schema::AUTHORITY_STATE) => {
                codec::decode_state(&data).ok().map(Recorded::Authority)
            }
            _ => None,
        };
        if let Some(decoded) = decoded {
            out.push(RecordedMessage {
                log_ns: i64::try_from(message.log_time).unwrap_or(i64::MAX),
                publish_ns: i64::try_from(message.publish_time).unwrap_or(i64::MAX),
                message: decoded,
            });
        }
    }
    out.sort_by_key(|m| m.log_ns);
    Ok(out)
}

fn lease_changes(
    state: &AuthoritySnapshot,
    log_ns: u64,
    held: &mut BTreeMap<String, (String, String, u64)>,
    out: &mut Vec<LeaseChange>,
) {
    let mut now: BTreeMap<String, (String, String, u64)> = BTreeMap::new();
    for lease in &state.leases {
        let part = state.part(lease.channel).map_or_else(
            || lease.channel.to_string(),
            |p| format!("{}/{}", p.owner, p.path),
        );
        now.insert(
            part,
            (
                lease.holder.to_string(),
                format!("{:?}", lease.priority).to_lowercase(),
                lease.token.0,
            ),
        );
    }
    for (part, current) in &now {
        if held.get(part) != Some(current) {
            out.push(LeaseChange {
                log_ns,
                part: part.clone(),
                holder: Some(current.0.clone()),
                priority: Some(current.1.clone()),
                token: Some(current.2),
            });
        }
    }
    for part in held.keys() {
        if !now.contains_key(part) {
            out.push(LeaseChange {
                log_ns,
                part: part.clone(),
                holder: None,
                priority: None,
                token: None,
            });
        }
    }
    *held = now;
}
