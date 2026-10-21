//! Real-time RLMesh sessions (protocol v2) on Zenoh.
//!
//! A session is a long-lived shared world that participants (embodiments,
//! controllers, observers) attach to. This crate is the one v2 dependency:
//! the network ends over Zenoh, the [`edge`] next to an embodiment, and,
//! re-exported from `rlmesh-session-core` (the sans-IO core the browser
//! build shares), the [`wire`], the [`spec`] and the state machines
//! ([`authority`], [`virtual_clock`], [`schedule`], the offset estimator in
//! [`clock`], the input cache in [`cache`] and the receive rules in
//! [`sample`]).
//!
//! - [`address`]: a session's `{ip}:{port}/{tenant}/{session}` address,
//!   and joining it from another process.
//! - [`authority`]: epoch, roster, the DISARMED/ARMED/ACTIVE/SAFE/ESTOP mode
//!   machine with a latched e-stop, the part table with host-assigned,
//!   never-reused ids (normalised by [`spec`]), bindings, and the
//!   lease table: all-or-none grants
//!   bound to a binding, priority classes
//!   (`safety > operator > model > scripted`), command groups, TTLs,
//!   preemption and fencing tokens. Sans-IO; the [`SessionHost`] is its
//!   single writer.
//! - [`virtual_clock`]: the virtual-mode tick barrier (§11.3), sans-IO.
//! - [`clock`]: clock domains and NTP-style offset estimation with an error
//!   bound, fed by heartbeats.
//! - [`host`]: the [`SessionHost`]: `ctl` verbs, the published authority
//!   snapshot, heartbeat echoes, lease expiry, virtual ticks and the history
//!   store. A library, and a binary (`rlmesh-session-host`).
//! - [`participant`]: the [`Participant`]: presence, heartbeats, the
//!   authority view, verbs, and endpoints by robot and path ([`sample`],
//!   [`chunk`], [`media`]); an embodiment's publishers follow their
//!   streams' publication plans.
//! - [`cache`]: the controller's input cache and history windows (§8.8):
//!   the snapshot at each policy start, freshness, hold, warmup, gaps and
//!   provenance, fed by [`Inputs`].
//! - [`schedule`]: policy starts (§8.9: one inference in flight, overdue
//!   starts skipped) and slowdown reporting (§8.10), sans-IO.
//! - [`controller`]: the [`Controller`] runtime language bindings wrap: a
//!   bound policy's inputs, its leases (kept and re-requested in the
//!   background), its chunks and its run loop.
//! - [`edge`]: the edge runtime next to an embodiment (§9): the timed
//!   action queue, command limits, the hold-then-safe ladder, the applied
//!   stream and log, and the [`Robot`](edge::Robot) bindings wrap.
//! - [`report`]: observer telemetry (§8.10), the publisher and controller
//!   counters an observer cannot measure itself.
//! - [`transport`]: one Zenoh session per traffic class, so a camera can
//!   only starve its own class.
//!
//! Every time is a monotonic nanosecond count in some participant's clock
//! domain; session time is the host's. Motion never depends on the clock
//! offsets: the edge converts validity windows into its own clock and checks
//! them locally.

#![warn(missing_docs)]

pub mod address;
pub use rlmesh_session_core::authority;
pub mod cache;
pub mod chunk;
pub mod clock;
pub mod controller;
pub mod edge;
pub mod error;
pub mod host;
pub mod media;
pub mod metrics;
pub mod participant;
pub mod report;
pub mod sample;
pub use rlmesh_session_core::schedule;
pub use rlmesh_session_core::spec;
pub mod transport;
pub use rlmesh_session_core::virtual_clock;

pub use address::SessionAddress;
pub use authority::{Authority, AuthorityConfig, AuthorityEvent, DetachReason, LeaseEnd};
pub use cache::{
    Cached, InputCache, InputSnapshot, Inputs, Observation, ObserveError, Skip, Snapshot,
};
pub use chunk::{ChunkDraft, ChunkPublisher};
pub use clock::{Clock, MonotonicClock, OffsetEstimate, SystemMonotonicClock};
pub use controller::{
    Controller, ControllerConfig, ControllerError, PhaseAlign, Predictor, RunError, SendError, Sent,
};
pub use error::SessionError;
pub use host::{HistoryConfig, HostConfig, SessionHost};
pub use media::{ImagePublisher, MediaStats, Offer};
pub use metrics::ParticipantHealth;
pub use participant::{
    AuthorityView, EdgeReport, ManualView, Participant, ParticipantConfig, SafeReporter, TickStream,
};
pub use sample::{
    Fresh, Received, SamplePublisher, SampleStats, SampleSubscriber, SubscribeOptions,
};
pub use schedule::{RunEvent, RunStats, SkipReason, StartSchedule};
pub use transport::{ClassQos, TrafficClass, Transport, TransportConfig, TransportTuning};
pub use virtual_clock::VirtualConfig;

/// The wire types, re-exported so users need one dependency.
pub use rlmesh_session_core::wire;
