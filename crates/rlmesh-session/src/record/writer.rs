//! The writer thread: queued messages into one MCAP file or rolling segments.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::sync::mpsc;

use crate::record::capture::{Entry, Metadata, Topic, TopicKey, Tx, ns};
use crate::record::rolling::RollingSink;
use crate::record::{RECORDING_FORMAT, RECORDING_METADATA, RecordError, RecordStats};
use crate::wire::codec;

pub(super) fn write_loop(
    mut sink: Sink,
    rx: &mpsc::Receiver<Entry>,
    queue: &Tx,
) -> Result<RecordStats, RecordError> {
    let mut stats = RecordStats::default();
    let mut metadata = Metadata::new();
    let mut authority_seen = false;
    // Each topic's last (log, publish) time: both come through clock
    // offsets the host keeps re-estimating, and a refinement can step one
    // back by its jitter, so a topic's times are held non-decreasing.
    let mut last: BTreeMap<String, (i64, i64)> = BTreeMap::new();
    while let Ok(entry) = rx.recv() {
        match entry {
            Entry::Message {
                topic,
                log_time,
                publish_time,
                sequence,
                data,
            } => {
                queue.queued.fetch_sub(data.len(), Ordering::Relaxed);
                // Anything before the first snapshot is skipped, so a
                // recording opens with the authority: a busy host can
                // deliver a sample ahead of it.
                authority_seen |= topic.key == TopicKey::Authority;
                if !authority_seen {
                    continue;
                }
                let (log_time, publish_time) = match last.get_mut(&topic.name) {
                    Some(prev) => {
                        *prev = (log_time.max(prev.0), publish_time.max(prev.1));
                        *prev
                    }
                    None => *last
                        .entry(topic.name.clone())
                        .or_insert((log_time, publish_time)),
                };
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
            Entry::Incident { label, log_time } => {
                if let Sink::Rolling(rolling) = &mut sink {
                    rolling.incident(label, log_time);
                }
            }
            Entry::Finish { metadata: m } => {
                metadata = m;
                break;
            }
        }
    }
    stats.overflow_drops = queue.dropped.load(Ordering::Relaxed);
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
pub(super) enum Sink {
    File(Box<Segment>),
    Rolling(Box<RollingSink>),
}

/// A queued message.
pub(super) struct Message {
    pub(super) topic: Arc<Topic>,
    pub(super) log_time: i64,
    publish_time: i64,
    sequence: u64,
    data: Vec<u8>,
}

/// One MCAP file being written.
pub(super) struct Segment {
    writer: mcap::Writer<BufWriter<File>>,
    path: PathBuf,
    /// The log time it was opened at.
    pub(super) started: i64,
    schemas: BTreeMap<&'static str, u16>,
    /// MCAP channel and its newest log time, by topic.
    channels: BTreeMap<TopicKey, (u16, u64)>,
    /// The first message's log time: none is written before it, so the
    /// snapshot a segment opens with reads back first. (A sample's log time
    /// is its receipt, which can precede the stamp the recorder's own
    /// snapshot subscriber gives the snapshot that declared its part.)
    floor: Option<u64>,
}

impl Segment {
    pub(super) fn create(path: &Path, started: i64) -> Result<Self, RecordError> {
        let file = File::create(path).map_err(|source| RecordError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        // zstd at level 1: numeric topics shrink several times over, and
        // JPEG frames, already compressed, cost little time to pass through.
        let mut writer = mcap::WriteOptions::new()
            .compression(Some(mcap::Compression::Zstd))
            .compression_level(1)
            .profile("rlmesh")
            .library(concat!("rlmesh-session ", env!("CARGO_PKG_VERSION")))
            .create(BufWriter::new(file))?;
        writer.write_metadata(&mcap::records::Metadata {
            name: RECORDING_METADATA.into(),
            metadata: BTreeMap::from([
                ("format".to_string(), RECORDING_FORMAT.to_string()),
                (
                    "protocol_generation".to_string(),
                    crate::wire::version::GENERATION.to_string(),
                ),
                (
                    "protocol_minor".to_string(),
                    crate::wire::version::PROTOCOL_MINOR.to_string(),
                ),
            ]),
        })?;
        Ok(Self {
            writer,
            path: path.to_path_buf(),
            started,
            schemas: BTreeMap::new(),
            channels: BTreeMap::new(),
            floor: None,
        })
    }

    pub(super) fn write(
        &mut self,
        m: &Message,
        stats: &mut RecordStats,
    ) -> Result<(), RecordError> {
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
        let floor = *self.floor.get_or_insert(log_time);
        if log_time < last_log.max(floor) {
            log_time = last_log.max(floor);
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

    pub(super) fn finish(
        mut self,
        metadata: &Metadata,
        stats: &mut RecordStats,
    ) -> Result<(), RecordError> {
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
