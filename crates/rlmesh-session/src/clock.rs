//! Clock domains and NTP-style offset estimation (design §3.1-3.2).
//!
//! Every participant stamps with its own monotonic clock. In wall mode the
//! session clock is the host's monotonic clock, and the host keeps, per
//! participant, an offset estimate with a rigorous error bound:
//! `session_ns = participant_ns + offset_ns`, within `± error_ns`.
//!
//! Virtual and lockstep session clocks are later work (wave 5).

use std::sync::Arc;
use std::time::Instant;

/// The offset estimator is sans-IO and lives in `rlmesh-session-core`.
pub use rlmesh_session_core::clock::{OffsetEstimate, OffsetEstimator, OffsetEstimatorConfig};

/// A monotonic clock in nanoseconds. Each participant owns one clock domain.
pub trait Clock: Send + Sync + 'static {
    /// The current time in this clock domain, in nanoseconds.
    fn now_ns(&self) -> i64;
}

/// `std::time::Instant` measured from an anchor, plus a fixed shift.
///
/// The shift exists so one process can host several clock domains with known
/// offsets between them (tests and the demo); production participants use
/// [`MonotonicClock::new`].
#[derive(Clone, Debug)]
pub struct MonotonicClock {
    anchor: Instant,
    shift_ns: i64,
}

impl MonotonicClock {
    /// A clock anchored now, reading 0 at creation.
    pub fn new() -> Self {
        Self::with_anchor(Instant::now(), 0)
    }

    /// Reads `anchor.elapsed() + shift_ns`: two clocks sharing an anchor differ
    /// by exactly the difference of their shifts.
    pub fn with_anchor(anchor: Instant, shift_ns: i64) -> Self {
        Self { anchor, shift_ns }
    }

    /// As a shared trait object.
    pub fn shared(self) -> Arc<dyn Clock> {
        Arc::new(self)
    }
}

impl Default for MonotonicClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for MonotonicClock {
    fn now_ns(&self) -> i64 {
        let elapsed = i64::try_from(self.anchor.elapsed().as_nanos()).unwrap_or(i64::MAX);
        elapsed.saturating_add(self.shift_ns)
    }
}

/// The host's raw `CLOCK_MONOTONIC`: one clock domain shared by every
/// process on a machine, so cross-process latencies on one host need no
/// offset estimation (benchmarks, co-located edge and controller). Off Unix
/// it falls back to a per-process clock.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemMonotonicClock;

impl SystemMonotonicClock {
    /// As a shared trait object.
    pub fn shared(self) -> Arc<dyn Clock> {
        Arc::new(self)
    }
}

#[cfg(unix)]
impl Clock for SystemMonotonicClock {
    // clock_gettime into a local timespec; the field types differ by target
    // (`time_t` is 32 bits on some), hence the widening conversions.
    #[allow(unsafe_code, clippy::useless_conversion)]
    fn now_ns(&self) -> i64 {
        let mut ts = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: `ts` is a valid, writable timespec and CLOCK_MONOTONIC is
        // always available on the platforms this crate builds for.
        let rc = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &raw mut ts) };
        debug_assert_eq!(rc, 0);
        i64::from(ts.tv_sec)
            .saturating_mul(1_000_000_000)
            .saturating_add(i64::from(ts.tv_nsec))
    }
}

#[cfg(not(unix))]
impl Clock for SystemMonotonicClock {
    fn now_ns(&self) -> i64 {
        use std::sync::OnceLock;
        static ANCHOR: OnceLock<Instant> = OnceLock::new();
        let anchor = ANCHOR.get_or_init(Instant::now);
        i64::try_from(anchor.elapsed().as_nanos()).unwrap_or(i64::MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clocks_sharing_an_anchor_differ_by_their_shift() {
        let anchor = Instant::now();
        let a = MonotonicClock::with_anchor(anchor, 0);
        let b = MonotonicClock::with_anchor(anchor, -3_000_000_000);
        // b is read after a, so the difference can only fall short of 3 s.
        let (ta, tb) = (a.now_ns(), b.now_ns());
        let diff = ta - tb;
        assert!((2_990_000_000..=3_000_000_000).contains(&diff), "{diff}");
    }
}
