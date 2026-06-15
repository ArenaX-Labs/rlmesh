//! `SpaceValue` ↔ wire-bytes codec entry points. Wraps the core
//! `encode/decode_batched_partial_values` verbatim — the raw-vs-proto framing
//! choice stays in Rust, never reconstructed in C. Encode validates with
//! `contains` (the capi adds this; core's wire path only count-checks).
#![allow(unsafe_code)] // FFI: raw pointers, owned-buffer transfer.

use rlmesh_grpc::wire::{
    binary_to_bytes, decode_batched_partial_values, encode_batched_partial_values,
};
use rlmesh_spaces::{BinaryPayload, SpaceSpec, SpaceValue, contains};

use crate::abi::status::{CapiError, RlmeshStatus, guard};
use crate::spaces::RlmeshSpaceSpec;
use crate::value::handle::{RlmeshValue, into_handle};

/// An owned byte buffer produced by the capi (e.g. `rlmesh_encode_batch`). Free
/// with `rlmesh_bytes_free`.
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
            return Self { data: std::ptr::null_mut(), len: 0, cap: 0 };
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
        // out_action (cap < len, or empty) would make Vec::from_raw_parts UB, so
        // it is leaked rather than reclaimed into an invalid Vec.
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

fn spec_ref<'a>(spec: *const RlmeshSpaceSpec) -> Result<&'a SpaceSpec, CapiError> {
    unsafe { spec.cast::<SpaceSpec>().as_ref() }
        .ok_or_else(|| CapiError::invalid_arg("null space spec"))
}

/// Decode an observation payload against `spec` into an array of values (one per
/// sub-env). Writes the array pointer + count; free with `rlmesh_values_free`. A
/// NULL/empty payload decodes to zero values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_decode_batch(
    data: *const u8,
    len: usize,
    spec: *const RlmeshSpaceSpec,
    out_values: *mut *mut *mut RlmeshValue,
    out_n: *mut usize,
) -> RlmeshStatus {
    guard(|| {
        let spec = spec_ref(spec)?;
        let out_values = unsafe { out_values.as_mut() }
            .ok_or_else(|| CapiError::invalid_arg("null out_values"))?;
        let out_n =
            unsafe { out_n.as_mut() }.ok_or_else(|| CapiError::invalid_arg("null out_n"))?;
        let values = if data.is_null() || len == 0 {
            decode_batched_partial_values(None, spec)
        } else {
            let payload = BinaryPayload {
                data: unsafe { std::slice::from_raw_parts(data, len) }.to_vec(),
            };
            decode_batched_partial_values(Some(&binary_to_bytes(&payload)), spec)
        }
        .map_err(|err| CapiError::invalid_value(err.to_string()))?;
        let boxed: Vec<*mut RlmeshValue> = values.into_iter().map(into_handle).collect();
        let n = boxed.len();
        *out_values = Box::into_raw(boxed.into_boxed_slice()).cast::<*mut RlmeshValue>();
        *out_n = n;
        Ok(())
    })
}

/// Encode `n` action values against `spec` into `out`. Validates each value with
/// `contains` first; free `out` with `rlmesh_bytes_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_encode_batch(
    values: *const *const RlmeshValue,
    n: usize,
    spec: *const RlmeshSpaceSpec,
    out: *mut RlmeshBytes,
) -> RlmeshStatus {
    guard(|| {
        let spec = spec_ref(spec)?;
        let out = unsafe { out.as_mut() }.ok_or_else(|| CapiError::invalid_arg("null out"))?;
        if n != 0 && values.is_null() {
            return Err(CapiError::invalid_arg("null values"));
        }
        let slice = if n == 0 {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(values, n) }
        };
        let mut owned: Vec<SpaceValue> = Vec::with_capacity(n);
        for &ptr in slice {
            let value = unsafe { ptr.cast::<SpaceValue>().as_ref() }
                .ok_or_else(|| CapiError::invalid_arg("null value in batch"))?;
            contains(spec, value).map_err(|err| CapiError::invalid_value(err.to_string()))?;
            owned.push(value.clone());
        }
        let encoded = encode_batched_partial_values(&owned, spec)
            .map_err(|err| CapiError::invalid_value(err.to_string()))?;
        *out = RlmeshBytes::from_vec(encoded.data);
        Ok(())
    })
}

/// Free an array of values from `rlmesh_decode_batch` (frees each value and the
/// array).
///
/// # Safety
/// `values`/`n` must come from a single `rlmesh_decode_batch` call, unfreed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_values_free(values: *mut *mut RlmeshValue, n: usize) {
    if values.is_null() {
        return;
    }
    unsafe {
        let slice = std::slice::from_raw_parts_mut(values, n);
        for slot in slice.iter() {
            if !slot.is_null() {
                drop(Box::from_raw(*slot));
            }
        }
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(values, n)));
    }
}
