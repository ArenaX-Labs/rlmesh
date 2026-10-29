//! The C ABI (`include/rlmesh_device.h`).

#![allow(unsafe_code)] // C callbacks and out-pointers.

use core::ffi::c_int;

use rlmesh_session_core::crypto::{Rng, RngError};

use crate::timing::{self, TimingError};

mod provision;
mod robot;
pub use provision::*;
pub use robot::*;

/// Success.
pub const RLMESH_DEVICE_OK: c_int = 0;
/// A null argument.
pub const RLMESH_DEVICE_ERR_ARG: c_int = -1;
/// The platform's randomness failed.
pub const RLMESH_DEVICE_ERR_RNG: c_int = -2;
/// The crypto failed where it can't (a bug, or memory corrupted).
pub const RLMESH_DEVICE_ERR_CRYPTO: c_int = -3;
/// A config field is invalid: an address without its key, a name that
/// isn't one, a map the host would refuse, a zero heartbeat period.
pub const RLMESH_DEVICE_ERR_CONFIG: c_int = -4;
/// Not running yet, or the stream's key hasn't arrived: nothing was sent.
pub const RLMESH_DEVICE_ERR_NOT_READY: c_int = -5;
/// A value not in its part's dense layout.
pub const RLMESH_DEVICE_ERR_INVALID: c_int = -6;

/// The platform's microsecond clock.
pub type NowUs = unsafe extern "C" fn() -> u64;
/// The platform's cryptographic randomness: fills `len` bytes, 0 on
/// success.
pub type Random = unsafe extern "C" fn(buf: *mut u8, len: usize) -> c_int;

/// The crate's version: major, minor and patch, a byte each.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_device_version() -> u32 {
    let n = |s: &str| s.parse::<u32>().unwrap_or(0);
    (n(env!("CARGO_PKG_VERSION_MAJOR")) << 16)
        | (n(env!("CARGO_PKG_VERSION_MINOR")) << 8)
        | n(env!("CARGO_PKG_VERSION_PATCH"))
}

/// `rlmesh_device_timing_t`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RlmeshDeviceTiming {
    pub handshake_start_us: u32,
    pub handshake_finish_us: u32,
    pub state_verify_us: u32,
    pub chunk_open_sign_us: u32,
    pub chunk_open_seal_us: u32,
    pub sample_seal_sign_us: u32,
}

pub(crate) struct CRng(pub(crate) Random);

impl Rng for CRng {
    fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), RngError> {
        // SAFETY: the caller's callback, writing `dest.len()` bytes.
        match unsafe { (self.0)(dest.as_mut_ptr(), dest.len()) } {
            0 => Ok(()),
            _ => Err(RngError),
        }
    }
}

/// Runs [`timing::run`] on the platform's clock and randomness.
///
/// # Safety
/// `out` is null or valid for a write; the callbacks are sound to call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_device_timing(
    now_us: Option<NowUs>,
    random: Option<Random>,
    out: *mut RlmeshDeviceTiming,
) -> c_int {
    let (Some(now_us), Some(random)) = (now_us, random) else {
        return RLMESH_DEVICE_ERR_ARG;
    };
    if out.is_null() {
        return RLMESH_DEVICE_ERR_ARG;
    }
    // SAFETY: the caller's clock.
    let mut clock = || unsafe { now_us() };
    match timing::run(&mut clock, &mut CRng(random)) {
        Ok(t) => {
            // SAFETY: checked non-null; valid per the contract.
            unsafe {
                out.write(RlmeshDeviceTiming {
                    handshake_start_us: t.handshake_start_us,
                    handshake_finish_us: t.handshake_finish_us,
                    state_verify_us: t.state_verify_us,
                    chunk_open_sign_us: t.chunk_open_sign_us,
                    chunk_open_seal_us: t.chunk_open_seal_us,
                    sample_seal_sign_us: t.sample_seal_sign_us,
                });
            }
            RLMESH_DEVICE_OK
        }
        Err(TimingError::Rng) => RLMESH_DEVICE_ERR_RNG,
        Err(TimingError::Crypto) => RLMESH_DEVICE_ERR_CRYPTO,
    }
}

#[cfg(test)]
mod tests {
    use core::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static CLOCK: AtomicU64 = AtomicU64::new(0);

    unsafe extern "C" fn now_us() -> u64 {
        CLOCK.fetch_add(10, Ordering::Relaxed)
    }

    static DRAWN: AtomicU64 = AtomicU64::new(0);

    /// A different byte stream on every call: no two keys alike.
    unsafe extern "C" fn random(buf: *mut u8, len: usize) -> c_int {
        for i in 0..len {
            let n = DRAWN
                .fetch_add(1, Ordering::Relaxed)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15);
            unsafe { buf.add(i).write((n >> 56) as u8) };
        }
        0
    }

    unsafe extern "C" fn broken(_: *mut u8, _: usize) -> c_int {
        -1
    }

    #[test]
    fn timing_through_the_c_abi() {
        let mut out = RlmeshDeviceTiming::default();
        let rc = unsafe { rlmesh_device_timing(Some(now_us), Some(random), &mut out) };
        assert_eq!(rc, RLMESH_DEVICE_OK);
        assert_eq!(out.chunk_open_seal_us, 10);
        let rc = unsafe { rlmesh_device_timing(Some(now_us), Some(broken), &mut out) };
        assert_eq!(rc, RLMESH_DEVICE_ERR_RNG);
        let rc = unsafe { rlmesh_device_timing(None, Some(random), &mut out) };
        assert_eq!(rc, RLMESH_DEVICE_ERR_ARG);
        let rc = unsafe { rlmesh_device_timing(Some(now_us), Some(random), core::ptr::null_mut()) };
        assert_eq!(rc, RLMESH_DEVICE_ERR_ARG);
    }

    #[test]
    fn version_is_the_crate_version() {
        let v = rlmesh_device_version();
        assert_eq!(
            v >> 16,
            env!("CARGO_PKG_VERSION_MAJOR").parse::<u32>().unwrap()
        );
    }
}
