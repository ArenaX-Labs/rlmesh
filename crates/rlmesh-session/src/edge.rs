//! The edge runtime of a real-time RLMesh session: what runs next to an
//! embodiment and turns controllers' timed action chunks into one command
//! per control tick, safely, whatever the network does.
//!
//! - [`queue`]: the timed action queue, splicing new chunks at their
//!   `replace_from`, with [`Delivery::Once`] for delta commands.
//! - [`command`]: command validation and limits.
//! - [`fallback`]: the per-tick mode ladder of one command group (Idle,
//!   Driving, Hold, Safe, Estop) on edge-local deadlines.
//! - [`core`]: [`EdgeCore`], the sans-IO edge of one group: chunk fencing,
//!   validation, the queues, the ladder and the applied stream.
//! - [`runtime`]: [`Edge`], an [`EdgeCore`] attached to a
//!   [`crate::Participant`].
//! - [`robot`]: [`Robot`], a robot map declared with a publisher per part and an
//!   edge per command group, ticked by the robot's own control loop
//!   ([`Robot::due`]).
//!
//! The fallback never waits for the host: it works with the host gone. A full
//! process stall cannot be bounded from inside the process, so pair the edge
//! with an external or device watchdog. The `blend` and `rtc_prefix` merge
//! policies are refused at declaration.

pub use rlmesh_session_core::edge::{command, core, fallback, queue};
pub mod error;
pub mod robot;
pub mod runtime;

pub use crate::edge::core::{
    AppliedSample, EdgeConfig, EdgeCore, EdgeStats, EdgeTick, GroupHandover, Rejection,
};
pub use crate::wire::EdgeMode;
pub use error::EdgeError;
pub use fallback::{ModeStats, SafeCause};
pub use queue::{Delivery, QueueStats, QueuedAction};
pub use robot::{Due, Handover, Published, Robot};
pub use runtime::{Edge, OnAccept};
