//! Real-time RLMesh sessions (protocol v2) on Zenoh.
//!
//! A session is a long-lived shared world that participants (embodiments,
//! controllers, observers) attach to. This crate holds its authority core
//! and its network ends; the wire format lives in [`rlmesh_wire`] and the
//! embodiment-side action queue in `rlmesh-edge`.
//!
//! - [`authority`]: epoch, roster, the DISARMED/ARMED/ACTIVE/SAFE/ESTOP mode
//!   machine with a latched e-stop, and the lease table with priority classes
//!   (`safety > operator > model > scripted`), TTLs, preemption and fencing
//!   tokens. Sans-IO; the [`SessionHost`] is its single writer.
//! - [`clock`]: clock domains and NTP-style offset estimation with an error
//!   bound, fed by heartbeats.
//! - [`host`]: the [`SessionHost`]: `ctl` verbs, the published authority
//!   snapshot, heartbeat echoes, lease expiry and the history store. A
//!   library, and a binary (`rlmesh-session-host`).
//! - [`participant`]: the [`Participant`]: presence, heartbeats, the
//!   authority view, verbs, and channel endpoints ([`sample`], [`chunk`],
//!   [`media`]).
//! - [`transport`]: one Zenoh session per traffic class, so a camera can
//!   only starve its own class.
//!
//! Every time is a monotonic nanosecond count in some participant's clock
//! domain; session time is the host's. Motion never depends on the clock
//! offsets: the edge converts validity windows into its own clock and checks
//! them locally.

#![warn(missing_docs)]

pub mod authority;
pub mod chunk;
pub mod clock;
pub mod error;
pub mod host;
pub mod media;
pub mod metrics;
pub mod participant;
pub mod sample;
pub mod transport;

pub use authority::{Authority, AuthorityConfig, AuthorityEvent, DetachReason, LeaseEnd};
pub use chunk::{ChunkDraft, ChunkPublisher};
pub use clock::{Clock, MonotonicClock, OffsetEstimate, SystemMonotonicClock};
pub use error::SessionError;
pub use host::{HistoryConfig, HostConfig, SessionHost};
pub use media::{ImagePublisher, MediaStats, Offer};
pub use metrics::ParticipantHealth;
pub use participant::{
    AuthorityView, EdgeReport, ManualView, Participant, ParticipantConfig, SafeReporter,
};
pub use sample::{
    Fresh, Received, SamplePublisher, SampleStats, SampleSubscriber, SubscribeOptions,
};
pub use transport::{TrafficClass, Transport, TransportConfig, TransportTuning};

/// The wire types, re-exported so users need one dependency.
pub use rlmesh_wire as wire;
