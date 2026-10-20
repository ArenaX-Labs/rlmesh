//! The edge runtime of a real-time RLMesh session: what runs next to an
//! embodiment and turns controllers' timed action chunks into one command
//! per control tick, safely, whatever the network does.
//!
//! - [`queue`]: the timed action queue, splicing new chunks at their
//!   `replace_from` (validated before anything is removed), with
//!   [`Delivery::Once`] for delta commands.
//! - [`fallback`]: the per-tick mode ladder (Idle, Driving, Hold, Safe,
//!   Estop) on edge-local deadlines: a dry queue or a preempted holder waits
//!   at most `hold_max` (repeating the last action only when the channel
//!   permits it and the action is still valid), a lapsed lease goes Safe at
//!   once, and Safe latches until a new epoch.
//! - [`core`]: [`EdgeCore`], the sans-IO edge for one actuation channel:
//!   chunk fencing (epoch, lease token, source, seq, conservative clock
//!   conversion, layout and finiteness), the queue, the ladder and the
//!   applied-action log with source attribution.
//! - [`runtime`]: [`Edge`], an [`EdgeCore`] attached to a
//!   [`rlmesh_session::Participant`]: chunks arrive from the transport,
//!   falls are reported to the host (`EnterSafe`), heartbeats carry the
//!   queue report.
//!
//! The fallback never waits for the host: it works with the host gone. A
//! full process stall, though, cannot be bounded from inside the process:
//! pair the edge with an external or device watchdog.
//!
//! Not yet here (see the v0.2 spec review): command groups that dispatch one
//! endpoint per group with a handover rule, and the `blend` and `rtc_prefix`
//! merge policies (refused at declaration).

#![warn(missing_docs)]

pub mod core;
pub mod error;
pub mod fallback;
pub mod queue;
pub mod runtime;

pub use crate::core::{AppliedAction, EdgeConfig, EdgeCore, EdgeStats, EdgeTick, Rejection};
pub use error::EdgeError;
pub use fallback::{EdgeMode, ModeStats, SafeCause};
pub use queue::{Delivery, QueueStats, QueuedAction};
pub use runtime::{Edge, OnAccept};
