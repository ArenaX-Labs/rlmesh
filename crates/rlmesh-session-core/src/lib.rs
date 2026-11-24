//! The sans-IO core of a real-time RLMesh session (protocol v2): every state
//! machine a session host, a participant or a browser runs, with no runtime,
//! sockets, threads, clock reads or randomness. Every time is an injected
//! nanosecond count (`now_ns`) in a named clock domain; session time is the
//! host's.
//!
//! - [`authority`]: epoch, roster, the mode machine, parts, bindings and the
//!   lease table (preemption, command groups, TTLs, fencing tokens); the session
//!   host is its single writer.
//! - [`auth`]: attaching through the handshake: credentials, identity,
//!   generations, name to device bindings and the reply cache.
//! - [`lease`]: the holder's side of a lease: a local deadline that never
//!   outlives the host's expiry, renewal, loss and fenced chunks.
//! - [`virtual_clock`]: the virtual-mode tick barrier.
//! - [`clock`]: NTP-style offset estimation with an error bound.
//! - [`crypto`]: the safety layer's envelopes and handshakes (`Sealed`,
//!   `Signed`, replay windows, the IK handshake and pairing), with the caller's
//!   randomness.
//! - [`cache`]: a controller's input cache and history windows.
//! - [`schedule`]: policy starts, slowdown reporting and start-grid phase
//!   alignment.
//! - [`spec`]: the physical-AI layer: kinds, robot and policy maps, binding and
//!   delivery contracts.
//! - [`stream`]: a subscriber's receive rules: epochs, seq gaps as losses,
//!   layout checks and freshness against `max_age`.
//! - [`wire`]: ids, paths, key layout, versioning, message types and the byte
//!   codec.
//!
//! `rlmesh-session` runs these over Zenoh and re-exports them under the same
//! paths; `rlmesh-wasm` runs them in the browser.

#![warn(missing_docs)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

#[cfg(all(test, feature = "std"))]
mod test_alloc;

mod prelude {
    // Unused where `std` is in the graph: its inherent methods win over the trait's.
    #[cfg(not(feature = "std"))]
    #[allow(unused_imports)]
    pub(crate) use crate::float::F64Ext;
    pub(crate) use alloc::borrow::ToOwned;
    pub(crate) use alloc::boxed::Box;
    pub(crate) use alloc::string::{String, ToString};
    pub(crate) use alloc::vec::Vec;
    pub(crate) use alloc::{format, vec};
}

// `std`'s maths calls round the last bit differently from `libm`; host and device must agree bit for bit.
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

pub mod auth;
pub mod authority;
pub mod cache;
pub mod clock;
pub mod crypto;
pub mod datagram;
pub mod edge;
pub mod lease;
#[cfg(feature = "fleet")]
pub mod lobby;
pub mod schedule;
pub mod spec;
pub mod stream;
pub mod virtual_clock;
pub mod wire;

pub use authority::{Authority, AuthorityConfig, AuthorityEvent, DetachReason, Identity};
pub use cache::{Cached, InputCache, InputSnapshot, Skip, Snapshot};
pub use clock::{OffsetEstimate, OffsetEstimator, OffsetEstimatorConfig};
pub use lease::{LeaseChange, LeaseClient, LeaseError, LeaseGroup, LeaseState};
pub use schedule::{InputStats, PhaseAlign, RunEvent, RunStats, SkipReason, StartSchedule};
pub use stream::SampleStats;
pub use virtual_clock::VirtualConfig;
