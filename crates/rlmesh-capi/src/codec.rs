//! `RlmeshBytes` — an owned byte buffer handed across the ABI (JSON payloads
//! from the adapter path). The observation/action wire codec never crosses the
//! ABI: the model path hands C decoded values and takes typed values back.
#![allow(unsafe_code)] // FFI: owned-buffer transfer.

/// An owned byte buffer produced by the capi. Free with `rlmesh_bytes_free`.
#[repr(C)]
pub struct RlmeshBytes {
    /// Buffer start, or NULL when empty.
    pub data: *mut u8,
    pub len: usize,
    /// Allocation capacity (do not modify).
    pub cap: usize,
}

impl RlmeshBytes {
    pub(crate) fn from_vec(mut bytes: Vec<u8>) -> Self {
        // An empty Vec's as_mut_ptr() is a dangling non-null sentinel; report the
        // documented NULL/empty form so a consumer can branch on `data` not `len`.
        if bytes.is_empty() {
            return Self {
                data: std::ptr::null_mut(),
                len: 0,
                cap: 0,
            };
        }
        let out = Self {
            data: bytes.as_mut_ptr(),
            len: bytes.len(),
            cap: bytes.capacity(),
        };
        std::mem::forget(bytes);
        out
    }
    /// # Safety
    /// `self` must originate from `from_vec` and not have been freed.
    pub(crate) unsafe fn into_vec(self) -> Vec<u8> {
        // A capi buffer always satisfies cap >= len > 0; a foreign or corrupted
        // buffer (cap < len, or empty) would make Vec::from_raw_parts UB, so it is
        // leaked rather than reclaimed into an invalid Vec.
        if self.data.is_null() || self.cap == 0 || self.cap < self.len {
            Vec::new()
        } else {
            unsafe { Vec::from_raw_parts(self.data, self.len, self.cap) }
        }
    }
}

/// Free a buffer produced by the capi.
///
/// # Safety
/// `bytes` must be a buffer this thread owns and has not freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_bytes_free(bytes: RlmeshBytes) {
    drop(unsafe { bytes.into_vec() });
}
