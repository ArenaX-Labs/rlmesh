//! Reading a recording back: its messages, a summary, and decoded samples.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::record::{RECORDING_FORMAT, RECORDING_METADATA, RecordError};
use crate::wire::codec::{self, schema};
use crate::wire::{
    ActionChunk, AuthoritySnapshot, ChannelId, DType, Encoding, HostHealth, Sample, StreamSpec,
};
use bytes::Bytes;
use tracing::{debug, warn};

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
            schema::HOST_HEALTH => codec::decode_health(&data).is_ok(),
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
    Authority(Box<AuthoritySnapshot>),
    /// The host's health message: every link, edge and observer report.
    Health(Box<HostHealth>),
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

/// Every sample, chunk, authority snapshot and health message of a
/// recording (one MCAP file, or a directory of them: rolling segments, an
/// incident, an episode), ordered by
/// log time (stable, so each channel keeps its recorded order). Binding
/// records are skipped (the snapshots carry them), as are envelopes that do
/// not decode ([`summarize`] counts those). A file cut short (the recorder
/// crashed or lost power) yields what was written before the cut.
pub fn read(path: impl AsRef<Path>) -> Result<Vec<RecordedMessage>, RecordError> {
    let path = path.as_ref();
    let mut out = Vec::new();
    for file in mcap_files(path)? {
        match read_file(&file) {
            Ok(messages) => out.extend(messages),
            // A segment still being written may hold no finished chunk yet.
            Err(err) if path.is_dir() => {
                warn!(%err, file = %file.display(), "segment skipped");
            }
            Err(err) => return Err(err),
        }
    }
    out.sort_by_key(|m| m.log_ns);
    Ok(out)
}

/// `path` itself, or the `*.mcap` files in it, in name (opening) order.
pub fn mcap_files(path: &Path) -> Result<Vec<PathBuf>, RecordError> {
    if !path.is_dir() {
        return Ok(vec![path.to_path_buf()]);
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(path)
        .map_err(|source| RecordError::Io {
            path: path.to_path_buf(),
            source,
        })?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "mcap"))
        .collect();
    files.sort();
    Ok(files)
}

/// Refuses a recording in a newer format than [`RECORDING_FORMAT`]: its
/// `rlmesh.recording` record comes before the first chunk. A file without
/// one (written before formats were marked) is format 1.
fn check_format(path: &Path, bytes: &[u8]) -> Result<(), RecordError> {
    for record in mcap::read::LinearReader::new(bytes)? {
        match record {
            Ok(mcap::records::Record::Metadata(m)) if m.name == RECORDING_METADATA => {
                let format = m.metadata.get("format").cloned().unwrap_or_default();
                return match format.parse::<u32>() {
                    Ok(n) if n <= RECORDING_FORMAT => Ok(()),
                    _ => Err(RecordError::Format {
                        path: path.to_path_buf(),
                        format,
                    }),
                };
            }
            Ok(mcap::records::Record::Chunk { .. } | mcap::records::Record::Message { .. })
            | Err(_) => return Ok(()),
            Ok(_) => {}
        }
    }
    Ok(())
}

fn read_file(path: &Path) -> Result<Vec<RecordedMessage>, RecordError> {
    // TODO: reads the whole file into memory; stream it chunk by chunk when
    // recordings outgrow RAM (long camera sessions, rolling segments).
    let bytes = std::fs::read(path).map_err(|source| RecordError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    check_format(path, &bytes)?;
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
            Some(schema::AUTHORITY_STATE) => codec::decode_state(&data)
                .ok()
                .map(|s| Recorded::Authority(Box::new(s))),
            Some(schema::HOST_HEALTH) => codec::decode_health(&data)
                .ok()
                .map(|h| Recorded::Health(Box::new(h))),
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
    Ok(out)
}

/// One field of a [`DecodedSample`]: little-endian values, or UTF-8 text.
#[derive(Clone, Debug, PartialEq)]
pub struct Field {
    /// Its declared name.
    pub name: String,
    /// Its element type.
    pub dtype: DType,
    /// Its shape (`[H, W, 3]` for a camera frame).
    pub shape: Vec<u32>,
    /// Its bytes.
    pub bytes: Vec<u8>,
}

/// A recorded sample split into its declared fields.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodedSample {
    /// `{robot}/{path}`, or `{robot}/{path}/_applied` for what an edge
    /// dispatched.
    pub topic: String,
    /// When the recorder received it (session time).
    pub log_ns: i64,
    /// When it was captured, mapped to session time (the MCAP
    /// `publish_time`): what frames align on.
    pub capture_ns: i64,
    /// Its fields, in declared order; a JPEG frame decoded to raw RGB.
    /// None for an `_applied` tick that dispatched nothing.
    pub fields: Vec<Field>,
}

/// Every sample of `messages` (as [`read`] returns them) split into its
/// fields, through the parts the authority snapshots before it declare.
/// A sample of a channel not declared yet, or that does not decode, is
/// skipped.
pub fn decode_samples(messages: &[RecordedMessage]) -> Vec<DecodedSample> {
    let mut decoder = SampleDecoder::default();
    messages.iter().filter_map(|m| decoder.decode(m)).collect()
}

/// Splits recorded samples into their fields, one message at a time, through
/// the parts the authority snapshots before them declare.
#[derive(Clone, Debug, Default)]
pub struct SampleDecoder {
    streams: BTreeMap<ChannelId, StreamSpec>,
    /// Leave camera frames encoded: their field comes with its shape and no
    /// bytes.
    pub skip_images: bool,
}

impl SampleDecoder {
    /// Takes `m` in log-time order: an authority snapshot declares parts, a
    /// sample decodes. `None` for anything else, a sample of a channel not
    /// declared yet, or one that does not decode.
    pub fn decode(&mut self, m: &RecordedMessage) -> Option<DecodedSample> {
        let sample = match &m.message {
            Recorded::Authority(snapshot) => {
                for part in &snapshot.parts {
                    for stream in [
                        part.stream().ok(),
                        part.applied_stream().and_then(Result::ok),
                    ]
                    .into_iter()
                    .flatten()
                    {
                        self.streams.insert(stream.channel, stream);
                    }
                }
                return None;
            }
            Recorded::Sample(sample) => sample,
            _ => return None,
        };
        let spec = self.streams.get(&sample.channel)?;
        let fields = if sample.applied.is_some_and(|a| !a.present) {
            // An edge tick that dispatched nothing (hold, safe, e-stop).
            Vec::new()
        } else if matches!(spec.encoding, Encoding::Jpeg { .. }) {
            if self.skip_images {
                vec![(0, Vec::new())]
            } else {
                let Ok(rgb) = crate::media::decode_payload(spec, &sample.payload) else {
                    debug!(channel = %spec.channel, "undecodable frame");
                    return None;
                };
                vec![(0, rgb)]
            }
        } else {
            match spec.layout.stride() {
                None => vec![(0, sample.payload.to_vec())],
                Some(stride) if stride == sample.payload.len() => (0..spec.fields.len())
                    .filter_map(|i| Some((i, sample.payload.get(spec.layout.field(i)?)?.to_vec())))
                    .collect(),
                Some(_) => return None,
            }
        };
        Some(DecodedSample {
            topic: format!("{}/{}", spec.owner, spec.path),
            log_ns: m.log_ns,
            capture_ns: m.publish_ns,
            fields: fields
                .into_iter()
                .map(|(i, bytes)| Field {
                    name: spec.fields[i].name.clone(),
                    dtype: spec.fields[i].dtype,
                    shape: spec.fields[i].shape.clone(),
                    bytes,
                })
                .collect(),
        })
    }
}

/// Every sample of a recording (one MCAP file, or a directory of them) split
/// into its fields, one at a time: only one file's messages, still encoded,
/// are held at once, and each camera frame is decoded as it is taken.
/// Within a file samples come in log-time order; files in name (opening)
/// order. A segment still being written that holds no finished chunk is
/// skipped, as [`read`] does.
pub struct Samples {
    files: std::collections::VecDeque<PathBuf>,
    messages: std::vec::IntoIter<RecordedMessage>,
    decoder: SampleDecoder,
    dir: bool,
}

/// [`Samples`] over the recording at `path`.
///
/// # Errors
/// [`RecordError::Io`] when `path` can't be listed.
pub fn samples(path: impl AsRef<Path>) -> Result<Samples, RecordError> {
    let path = path.as_ref();
    Ok(Samples {
        files: mcap_files(path)?.into(),
        messages: Vec::new().into_iter(),
        decoder: SampleDecoder::default(),
        dir: path.is_dir(),
    })
}

impl Samples {
    /// Leaves camera frames encoded: each comes with its shape and no bytes
    /// (a pass that only needs to know the streams).
    #[must_use]
    pub fn skip_images(mut self) -> Self {
        self.decoder.skip_images = true;
        self
    }
}

impl Iterator for Samples {
    type Item = Result<DecodedSample, RecordError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            for m in self.messages.by_ref() {
                if let Some(sample) = self.decoder.decode(&m) {
                    return Some(Ok(sample));
                }
            }
            let file = self.files.pop_front()?;
            match read_file(&file) {
                Ok(mut messages) => {
                    messages.sort_by_key(|m| m.log_ns);
                    self.messages = messages.into_iter();
                }
                Err(err) if self.dir => {
                    warn!(%err, file = %file.display(), "segment skipped");
                }
                Err(err) => return Some(Err(err)),
            }
        }
    }
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
