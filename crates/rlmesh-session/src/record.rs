//! MCAP recording of a real-time RLMesh session.
//!
//! A [`Recorder`] is an observer participant: it subscribes to every stream
//! of every robot (parts and their `_applied` streams), to every commandable
//! part's action chunks and to the host's signed authority snapshots and
//! health messages, and writes
//! each envelope unchanged into an MCAP file:
//!
//! | Topic | Schema | One message per |
//! |---|---|---|
//! | `/{robot}/{path}` | `rlmesh.session.v2.Sample` | sample (a batch is split) |
//! | `/{robot}/{path}/_applied` | `rlmesh.session.v2.Sample` | edge control tick |
//! | `/{robot}/{path}/_chunks` | `rlmesh.session.v2.ActionChunk` | chunk sent to the edge |
//! | `/_session/authority` | `rlmesh.session.v2.AuthorityState` | authority change |
//! | `/_session/bindings` | `rlmesh.session.v2.BindingRecord` | new or changed binding |
//! | `/_session/health` | `rlmesh.session.v2.HostHealth` | host health message (about 1 Hz) |
//!
//! Schemas are the `rlmesh.session.v2` protobuf descriptors. A new MCAP
//! channel opens for every session channel id. `publish_time` is the
//! capture (or send) time mapped to session time, `log_time` session time
//! at the recorder, and the MCAP sequence the low 32 bits of the envelope's
//! seq (the full u64 stays in the envelope).
//!
//! The recorder records what the session publishes: it declares no delivery
//! contract, so a stream a policy reads at `latest@R` is recorded at R, while
//! `_applied` and chunks are recorded in full.
//!
//! Always-on capture ([`Recorder::start_rolling`]) also keeps incidents:
//! an e-stop, safe mode, a lease preemption or a burst of missed edge
//! deadlines links the session's recent segments, and the ones written
//! for a while after, into `incidents/{session}/{unix_ms}-{label}/`, where
//! the rolling purge does not reach them. [`keep`] does the same on
//! request (the cockpit's Keep).
//!
//! [`read`] decodes a recording back into its messages in log-time order
//! (the cockpit's replay). [`summarize`] checks one: message counts per
//! topic, time ordering, envelopes that do not decode, and the lease
//! timeline.

mod capture;
mod reader;
mod rolling;
mod writer;

pub use capture::{QUEUE_BYTES, Recorder, SampleTee};
pub use reader::{
    DecodedSample, Field, LeaseChange, Recorded, RecordedMessage, SampleDecoder, Samples, Summary,
    TopicSummary, decode_samples, mcap_files, read, samples, summarize,
};
pub use rolling::{Episode, Rolling, collect, keep};

use std::path::PathBuf;

use crate::SessionError;

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
    /// A recording in a newer format than this build reads.
    #[error(
        "recording {path} is in format {format}; this rlmesh reads format {RECORDING_FORMAT} \
         (upgrade rlmesh)"
    )]
    Format {
        /// The file.
        path: PathBuf,
        /// Its format.
        format: String,
    },
}

/// The recording format this build writes and reads: an `rlmesh.recording`
/// metadata record at the start of every file carries it, with the protocol
/// generation and minor. A reader refuses a newer format; a file without
/// one is format 1.
pub const RECORDING_FORMAT: u32 = 1;

/// The name of the metadata record that carries [`RECORDING_FORMAT`].
pub const RECORDING_METADATA: &str = "rlmesh.recording";

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
    /// Messages dropped because the disk fell [`QUEUE_BYTES`] behind.
    pub overflow_drops: u64,
    /// MCAP messages written.
    pub messages: u64,
    /// MCAP channels opened.
    pub channels: u32,
    /// Payload bytes written (envelopes).
    pub bytes: u64,
    /// Messages whose `log_time` was raised to their channel's previous one
    /// (the recorder's session clock stepped back on a remapping) or to the
    /// file's first.
    pub log_time_clamped: u64,
}
