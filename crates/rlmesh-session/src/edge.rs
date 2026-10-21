//! The edge runtime of a real-time RLMesh session: what runs next to an
//! embodiment and turns controllers' timed action chunks into one command
//! per control tick, safely, whatever the network does (the v0.2 spec §9).
//!
//! - [`queue`]: the timed action queue, splicing new chunks at their
//!   `replace_from` (validated before anything is removed), with
//!   [`Delivery::Once`] for delta commands (§9.3).
//! - [`command`]: command validation and limits (§9.6): finite values, valid
//!   rotations, clamping, norm bounds and the binary snap.
//! - [`fallback`]: the per-tick mode ladder of one command group (Idle,
//!   Driving, Hold, Safe, Estop) on edge-local deadlines (§9.5), with
//!   dispatch gated to the leased member and the handover rule (§9.7).
//! - [`core`]: [`EdgeCore`], the sans-IO edge of one group: chunk fencing
//!   (epoch, lease token, source, binding, seq, the §9.1 clock rules and the
//!   transit check), validation, the queues, the ladder, the applied stream
//!   and the applied-action log with source attribution.
//! - [`runtime`]: [`Edge`], an [`EdgeCore`] attached to a
//!   [`crate::Participant`]: chunks arrive from the transport,
//!   applied samples go out on `{path}/_applied`, falls are reported to the
//!   host (`EnterSafe`), heartbeats carry the queue report.
//! - [`robot`]: [`Robot`], the embodiment runtime bindings wrap: a robot map
//!   declared, a publisher per part and an edge per command group, ticked
//!   by the robot's own control loop ([`Robot::due`]).
//!
//! The fallback never waits for the host: it works with the host gone. A
//! full process stall, though, cannot be bounded from inside the process:
//! pair the edge with an external or device watchdog. The `blend` and
//! `rtc_prefix` merge policies are Later (§19) and refused at declaration.

pub mod command;
pub mod core;
pub mod error;
pub mod fallback;
pub mod queue;
pub mod robot;
pub mod runtime;

pub use crate::edge::core::{
    AppliedAction, AppliedSample, EdgeConfig, EdgeCore, EdgeStats, EdgeTick, GroupHandover,
    Rejection,
};
pub use crate::wire::EdgeMode;
pub use error::EdgeError;
pub use fallback::{ModeStats, SafeCause};
pub use queue::{Delivery, QueueStats, QueuedAction};
pub use robot::{Due, Handover, Robot};
pub use runtime::{Edge, OnAccept};
