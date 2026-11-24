//! Real-time RLMesh sessions (protocol v2) on Zenoh.
//!
//! A session is a long-lived shared world that participants (embodiments,
//! controllers, observers) attach to. This crate is the network end over
//! Zenoh; the sans-IO core ([`wire`], [`spec`], [`authority`], [`lease`],
//! [`virtual_clock`], [`schedule`]) is re-exported from `rlmesh-session-core`.
//!
//! - [`address`]: the address grammar (`[host/][session/]participant`), the
//!   hosting rule, and joining a session from another process.
//! - [`auth`] and [`crypto`]: the Noise IK handshake, the `Sealed` envelopes
//!   every `ctl` verb, reply and heartbeat travels in, and the host's `Signed`
//!   broadcasts. A participant attaches with a
//!   [`Credential`](auth::Credential).
//! - [`authority`]: epoch, roster, the DISARMED/ARMED/ACTIVE/SAFE/ESTOP mode
//!   machine with a latched e-stop, parts, bindings and leases.
//! - [`lease`]: the holder's side of a lease.
//! - [`virtual_clock`]: the virtual-mode tick barrier.
//! - [`clock`]: clock domains and offset estimation.
//! - `gateway` (feature `gateway`): the browser gateway.
//! - [`host`]: the [`SessionHost`]: `ctl` verbs, the published authority
//!   snapshot, heartbeat echoes, lease expiry, virtual ticks and the history
//!   store. `rlmesh host` runs one in the foreground.
//! - [`server`]: the [`SessionServer`], a multi-session host that starts a
//!   session's host for its first participant.
//! - [`participant`]: the [`Participant`] and its endpoints ([`sample`],
//!   [`chunk`], [`media`]).
//! - [`cache`]: the controller's input cache and history windows.
//! - [`schedule`]: policy starts and slowdown reporting.
//! - [`controller`]: the [`Controller`] runtime: a bound policy's inputs, its
//!   leases (kept and re-requested in the background), its chunks and its run
//!   loop.
//! - [`edge`]: the edge runtime next to an embodiment.
//! - `record` (feature `record`): MCAP recording and playback.
//! - [`report`]: observer telemetry.
//! - [`pairing`]: machine pairing with the host.
//! - With the unreleased `fleet` feature: `discovery`, `lobby` and `scene`.
//! - [`sdk`]: what the language bindings share.
//! - [`transport`]: one Zenoh session per traffic class.
//! - [`logging`]: the Zenoh log target every tracing subscriber must cap at
//!   `INFO`, since it logs the link password at `DEBUG`.
//!
//! Every time is a monotonic nanosecond count in some participant's clock
//! domain; session time is the host's. Motion never depends on clock offsets:
//! the edge converts validity windows into its own clock and checks them
//! locally.

#![warn(missing_docs)]

pub mod address;
pub use rlmesh_session_core::auth;
pub use rlmesh_session_core::authority;
pub mod cache;
pub mod chunk;
pub mod clock;
pub mod controller;
pub use rlmesh_session_core::crypto;
#[cfg(feature = "fleet")]
pub mod discovery;
pub mod edge;
pub mod error;
#[cfg(feature = "gateway")]
pub mod gateway;
pub mod host;
#[cfg(feature = "fleet")]
pub mod lobby;
pub mod logging;
pub mod media;
pub mod metrics;
pub mod pairing;
pub mod participant;
mod quiet;
#[cfg(feature = "record")]
pub mod record;
pub mod report;
pub mod sample;
#[cfg(feature = "fleet")]
pub mod scene;
pub mod sdk;
pub mod server;
pub use rlmesh_session_core::lease;
pub use rlmesh_session_core::schedule;
pub use rlmesh_session_core::spec;
pub mod transport;
pub use rlmesh_session_core::virtual_clock;

/// The log filter a binary or binding starts with when `RUST_LOG` is unset: warnings, none of
/// Zenoh's. What of Zenoh's matters reaches the caller as a session error.
pub const DEFAULT_LOG_FILTER: &str = "warn,zenoh=off";

pub use address::SessionAddress;
pub use authority::{Authority, AuthorityConfig, AuthorityEvent, DetachReason};
pub use cache::{
    Cached, InputCache, InputSnapshot, Inputs, Observation, ObserveError, Skip, Snapshot,
};
pub use chunk::{ChunkDraft, ChunkPublisher};
pub use clock::{Clock, MonotonicClock, OffsetEstimate, SystemMonotonicClock};
pub use controller::{
    Controller, ControllerConfig, ControllerError, HandBack, Parked, Predictor, Resume, RunConfig,
    RunError, RunMode, SendError, Sent, Step, StepAction,
};
pub use error::SessionError;
pub use host::{HistoryConfig, HostConfig, KeepRecording, SessionHost};
pub use media::{ImagePublisher, MediaStats, Offer};
pub use metrics::ParticipantHealth;
pub use participant::{
    AuthorityView, EdgeReport, HealthSubscription, ManualView, Participant, ParticipantConfig,
    Pending, SafeReporter, Sealer, TickStream,
};
pub use sample::{
    Fresh, Received, SamplePublisher, SampleStats, SampleSubscriber, SubscribeOptions,
};
pub use schedule::{InputStats, PhaseAlign, RunEvent, RunStats, SkipReason, StartSchedule};
pub use server::SessionServer;
pub use transport::{ClassQos, TrafficClass, Transport, TransportConfig, TransportTuning};
pub use virtual_clock::VirtualConfig;

/// The wire types, re-exported so users need one dependency.
pub use rlmesh_session_core::wire;
