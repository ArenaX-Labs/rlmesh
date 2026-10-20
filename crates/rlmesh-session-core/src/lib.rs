//! The sans-IO core of a real-time RLMesh session (protocol v2): every
//! state machine a session host, a participant or a browser runs, with no
//! runtime, sockets, threads, clock reads or randomness. Every time is an
//! injected nanosecond count (`now_ns`) in a named clock domain; session
//! time is the host's.
//!
//! - [`authority`]: epoch, roster, the mode machine, parts, bindings and the
//!   lease table (preemption, command groups, TTLs, fencing tokens); the
//!   session host is its single writer.
//! - [`virtual_clock`]: the virtual-mode tick barrier (§11.3).
//! - [`clock`]: NTP-style offset estimation with an error bound.
//! - [`cache`]: a controller's input cache and history windows (§8.8).
//! - [`schedule`]: policy starts and slowdown reporting (§8.9, §8.10).
//! - [`stream`]: a subscriber's receive rules: epochs, seq gaps as losses,
//!   layout checks and freshness against `max_age`.
//!
//! `rlmesh-session` runs these over Zenoh and re-exports them, so its API is
//! unchanged; `rlmesh-wasm` runs them in the browser.

#![warn(missing_docs)]

pub mod authority;
pub mod cache;
pub mod clock;
pub mod schedule;
pub mod stream;
pub mod virtual_clock;

pub use authority::{Authority, AuthorityConfig, AuthorityEvent, DetachReason, LeaseEnd};
pub use cache::{Cached, InputCache, InputSnapshot, Skip, Snapshot};
pub use clock::{OffsetEstimate, OffsetEstimator, OffsetEstimatorConfig};
pub use schedule::{RunEvent, RunStats, SkipReason, StartSchedule};
pub use stream::SampleStats;
pub use virtual_clock::VirtualConfig;
