//! The sans-IO core of a real-time RLMesh session (protocol v2): every
//! state machine a session host, a participant or a browser runs, with no
//! runtime, sockets, threads, clock reads or randomness. Every time is an
//! injected nanosecond count (`now_ns`) in a named clock domain; session
//! time is the host's.
//!
//! - [`authority`]: epoch, roster, the mode machine, parts, bindings and the
//!   lease table (preemption, command groups, TTLs, fencing tokens); the
//!   session host is its single writer.
//! - [`lease`]: the holder's side of a lease: the local deadline (a
//!   conservative lower bound of the host's expiry that never outlives
//!   it), renewal and loss from authority snapshots, and fenced chunks.
//! - [`virtual_clock`]: the virtual-mode tick barrier (§11.3).
//! - [`clock`]: NTP-style offset estimation with an error bound.
//! - [`crypto`]: the safety layer's envelopes and handshakes (`Sealed`,
//!   `Signed`, replay windows, the IK handshake and pairing), with the
//!   caller's randomness.
//! - [`cache`]: a controller's input cache and history windows (§8.8).
//! - [`schedule`]: policy starts and slowdown reporting (§8.9, §8.10), and
//!   the start grid's phase alignment.
//! - [`spec`]: the physical-AI layer (§2-§8): the kind table, robot and
//!   policy maps, binding with refuse-and-explain, conversions and delivery
//!   contracts.
//! - [`stream`]: a subscriber's receive rules: epochs, seq gaps as losses,
//!   layout checks and freshness against `max_age`.
//! - [`wire`]: the session wire (§10): ids, paths, the key layout, the
//!   versioning rule, the message types and the one byte codec.
//!
//! `rlmesh-session` runs these over Zenoh and re-exports them under the same
//! paths (`rlmesh_session::wire`, `rlmesh_session::spec`, ...);
//! `rlmesh-wasm` runs them in the browser.

#![warn(missing_docs)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

/// What `std`'s prelude gives every module, from `alloc` so the crate builds
/// without `std`.
mod prelude {
    // Unused where `std` is still in the graph (a host target through
    // `uuid`, or a test harness): its inherent methods win over the trait's.
    #[cfg(not(feature = "std"))]
    #[allow(unused_imports)]
    pub(crate) use crate::float::F64Ext;
    pub(crate) use alloc::borrow::ToOwned;
    pub(crate) use alloc::boxed::Box;
    pub(crate) use alloc::string::{String, ToString};
    pub(crate) use alloc::vec::Vec;
    pub(crate) use alloc::{format, vec};
}

/// Transcendental functions, from `libm` in every build. `std`'s versions
/// call the platform's maths library, whose last-bit rounding differs from
/// `libm`'s, and the host and a device must compute the same values
/// (a resampled pixel, a rotation). clippy.toml bans the `f64` methods.
pub(crate) mod math {
    pub(crate) fn sin(x: f64) -> f64 {
        libm::sin(x)
    }
    #[cfg(test)]
    pub(crate) fn cos(x: f64) -> f64 {
        libm::cos(x)
    }
    pub(crate) fn sin_cos(x: f64) -> (f64, f64) {
        libm::sincos(x)
    }
    pub(crate) fn atan2(y: f64, x: f64) -> f64 {
        libm::atan2(y, x)
    }
    pub(crate) fn hypot(x: f64, y: f64) -> f64 {
        libm::hypot(x, y)
    }
}

/// The exact `f64` methods `core` lacks (correctly rounded, so `libm` and
/// `std` agree), for builds without `std`.
#[cfg(not(feature = "std"))]
#[allow(dead_code)]
mod float {
    pub(crate) trait F64Ext: Sized {
        fn floor(self) -> Self;
        fn ceil(self) -> Self;
        fn round(self) -> Self;
        fn round_ties_even(self) -> Self;
        fn trunc(self) -> Self;
        fn fract(self) -> Self;
        fn sqrt(self) -> Self;
        fn powi(self, n: i32) -> Self;
    }

    impl F64Ext for f64 {
        fn floor(self) -> Self {
            libm::floor(self)
        }
        fn ceil(self) -> Self {
            libm::ceil(self)
        }
        fn round(self) -> Self {
            libm::round(self)
        }
        fn round_ties_even(self) -> Self {
            libm::rint(self)
        }
        fn trunc(self) -> Self {
            libm::trunc(self)
        }
        fn fract(self) -> Self {
            self - libm::trunc(self)
        }
        fn sqrt(self) -> Self {
            libm::sqrt(self)
        }
        fn powi(self, n: i32) -> Self {
            libm::pow(self, f64::from(n))
        }
    }
}

pub mod authority;
pub mod cache;
pub mod clock;
pub mod crypto;
pub mod edge;
pub mod lease;
pub mod schedule;
pub mod spec;
pub mod stream;
pub mod virtual_clock;
pub mod wire;

pub use authority::{Authority, AuthorityConfig, AuthorityEvent, DetachReason, LeaseEnd};
pub use cache::{Cached, InputCache, InputSnapshot, Skip, Snapshot};
pub use clock::{OffsetEstimate, OffsetEstimator, OffsetEstimatorConfig};
pub use lease::{LeaseChange, LeaseClient, LeaseError, LeaseGroup, LeaseState};
pub use schedule::{PhaseAlign, RunEvent, RunStats, SkipReason, StartSchedule};
pub use stream::SampleStats;
pub use virtual_clock::VirtualConfig;
