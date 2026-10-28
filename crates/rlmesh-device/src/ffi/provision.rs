//! USB provisioning's C ABI (tether spec §14): `rlmesh_device_key_public`
//! and `rlmesh_device_provision_*`.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::ffi::{CStr, c_char, c_int};

use super::{RLMESH_DEVICE_ERR_ARG, RLMESH_DEVICE_OK};
use crate::provision::{self, Record, Responder, Step};

/// Nothing to do.
pub const RLMESH_DEVICE_PROVISION_NONE: c_int = 0;
/// Write the bytes to the serial port.
pub const RLMESH_DEVICE_PROVISION_REPLY: c_int = 1;
/// Store the bytes as the record (one atomic write), then call
/// `rlmesh_device_provision_stored`.
pub const RLMESH_DEVICE_PROVISION_STORE: c_int = 2;

/// The public key of a device key: the one `DEVICE_INFO` sends, and the
/// daemon enrolls.
///
/// # Safety
/// `private` and `public` are null or valid for 32 bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_device_key_public(private: *const u8, public: *mut u8) -> c_int {
    if private.is_null() || public.is_null() {
        return RLMESH_DEVICE_ERR_ARG;
    }
    let mut key = [0_u8; 32];
    // SAFETY: 32 bytes, per the contract.
    key.copy_from_slice(unsafe { core::slice::from_raw_parts(private, 32) });
    let out = provision::public_key(&key);
    zeroize::Zeroize::zeroize(&mut key);
    // SAFETY: 32 bytes, per the contract.
    unsafe { core::ptr::copy_nonoverlapping(out.as_ptr(), public, 32) };
    RLMESH_DEVICE_OK
}

/// `rlmesh_device_provision_t`: the responder, the record it asked to store,
/// and the bytes the caller borrows.
pub struct DeviceProvision {
    responder: Responder,
    pending: Option<Record>,
    out: Vec<u8>,
}

/// A responder for robot `name` (NUL-terminated) with device key
/// `device_key` (32 bytes), holding the stored `record` (`record_len`
/// bytes; null or 0 when not provisioned, or when it isn't valid). `arch`
/// as in the tether's `HELLO`: 3 RISC-V, 4 Xtensa.
///
/// # Safety
/// `name` is null or NUL-terminated; `device_key` null or 32 bytes;
/// `record` null or `record_len` bytes; `out` null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_device_provision_new(
    name: *const c_char,
    device_key: *const u8,
    arch: u8,
    record: *const u8,
    record_len: usize,
    out: *mut *mut DeviceProvision,
) -> c_int {
    if name.is_null() || device_key.is_null() || out.is_null() {
        return RLMESH_DEVICE_ERR_ARG;
    }
    // SAFETY: per the contract.
    let Ok(name) = unsafe { CStr::from_ptr(name) }.to_str() else {
        return RLMESH_DEVICE_ERR_ARG;
    };
    let mut key = [0_u8; 32];
    // SAFETY: 32 bytes, per the contract.
    key.copy_from_slice(unsafe { core::slice::from_raw_parts(device_key, 32) });
    let record = (!record.is_null() && record_len > 0)
        // SAFETY: `record_len` bytes, per the contract.
        .then(|| unsafe { core::slice::from_raw_parts(record, record_len) })
        .and_then(Record::parse);
    let responder = Responder::new(name, &key, arch, record.as_ref());
    zeroize::Zeroize::zeroize(&mut key);
    let p = Box::new(DeviceProvision {
        responder,
        pending: None,
        out: Vec::new(),
    });
    // SAFETY: checked non-null; valid for a write per the contract.
    unsafe { out.write(Box::into_raw(p)) };
    RLMESH_DEVICE_OK
}

/// Frees a responder; null is a no-op.
///
/// # Safety
/// `p` is null or from `rlmesh_device_provision_new`, and not used after.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_device_provision_free(p: *mut DeviceProvision) {
    if !p.is_null() {
        // SAFETY: from `Box::into_raw`, per the contract.
        drop(unsafe { Box::from_raw(p) });
    }
}

/// # Safety
/// `data` and `len` are null or valid for a write.
unsafe fn hand_out(p: &mut DeviceProvision, bytes: Vec<u8>, data: *mut *const u8, len: *mut usize) {
    p.out = bytes;
    // SAFETY: per the contract.
    unsafe {
        if !data.is_null() {
            data.write(p.out.as_ptr());
        }
        if !len.is_null() {
            len.write(p.out.len());
        }
    }
}

/// One byte from the serial port: `RLMESH_DEVICE_PROVISION_NONE`, or
/// `_REPLY` / `_STORE` with `*data`/`*len` set (valid until the next call
/// on `p`).
///
/// # Safety
/// `p` is null or from `rlmesh_device_provision_new`; `data` and `len`
/// null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_device_provision_push(
    p: *mut DeviceProvision,
    byte: u8,
    data: *mut *const u8,
    len: *mut usize,
) -> c_int {
    // SAFETY: per the contract.
    let Some(p) = (unsafe { p.as_mut() }) else {
        return RLMESH_DEVICE_ERR_ARG;
    };
    match p.responder.push(byte) {
        None => RLMESH_DEVICE_PROVISION_NONE,
        Some(Step::Reply(bytes)) => {
            // SAFETY: per the contract.
            unsafe { hand_out(p, bytes, data, len) };
            RLMESH_DEVICE_PROVISION_REPLY
        }
        Some(Step::Store(record)) => {
            let bytes = record.payload.clone();
            p.pending = Some(record);
            // SAFETY: per the contract.
            unsafe { hand_out(p, bytes, data, len) };
            RLMESH_DEVICE_PROVISION_STORE
        }
    }
}

/// The record from `_STORE` was stored (`ok` non-zero) or not: `*data` and
/// `*len` are the `PROVISIONED` to write (valid until the next call).
///
/// # Safety
/// `p` is null or from `rlmesh_device_provision_new`; `data` and `len`
/// null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_device_provision_stored(
    p: *mut DeviceProvision,
    ok: c_int,
    data: *mut *const u8,
    len: *mut usize,
) -> c_int {
    // SAFETY: per the contract.
    let Some(p) = (unsafe { p.as_mut() }) else {
        return RLMESH_DEVICE_ERR_ARG;
    };
    let Some(record) = p.pending.take() else {
        return RLMESH_DEVICE_ERR_ARG;
    };
    let bytes = p.responder.stored(&record, ok != 0);
    // SAFETY: per the contract.
    unsafe { hand_out(p, bytes, data, len) };
    RLMESH_DEVICE_OK
}

/// Whether `record` (`len` bytes) is a valid provisioning record: 1 or 0.
///
/// # Safety
/// `record` is null or valid for `len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_device_record_valid(record: *const u8, len: usize) -> c_int {
    if record.is_null() || len == 0 {
        return 0;
    }
    // SAFETY: per the contract.
    c_int::from(Record::parse(unsafe { core::slice::from_raw_parts(record, len) }).is_some())
}

#[cfg(test)]
mod tests {
    use rlmesh_tether::provision::{self as wire, Addresses, Prov, Provision};

    use super::*;

    #[test]
    fn provisioning_through_the_c_abi() {
        let key = [7_u8; 32];
        let mut public = [0_u8; 32];
        assert_eq!(
            unsafe { rlmesh_device_key_public(key.as_ptr(), public.as_mut_ptr()) },
            RLMESH_DEVICE_OK
        );
        assert_eq!(public, provision::public_key(&key));

        let mut p = core::ptr::null_mut();
        let rc = unsafe {
            rlmesh_device_provision_new(
                c"eye-3".as_ptr(),
                key.as_ptr(),
                4,
                core::ptr::null(),
                0,
                &mut p,
            )
        };
        assert_eq!(rc, RLMESH_DEVICE_OK);
        let mut wire_bytes = Vec::new();
        wire::write(
            &Prov::Provision(Provision {
                flags: 0,
                host_key: [9; 32],
                host_id: b"rlm-3f9a2c",
                addresses: Addresses::new(&[b"192.168.1.20:7447"]).unwrap(),
            }),
            0,
            |b| wire_bytes.push(b),
        )
        .unwrap();
        let (mut data, mut len) = (core::ptr::null(), 0);
        let mut kinds = Vec::new();
        let mut stored = Vec::new();
        for b in wire_bytes {
            let kind = unsafe { rlmesh_device_provision_push(p, b, &mut data, &mut len) };
            if kind != RLMESH_DEVICE_PROVISION_NONE {
                kinds.push(kind);
                stored = unsafe { core::slice::from_raw_parts(data, len) }.to_vec();
            }
        }
        assert_eq!(kinds, [RLMESH_DEVICE_PROVISION_STORE]);
        assert_eq!(
            unsafe { rlmesh_device_record_valid(stored.as_ptr(), stored.len()) },
            1
        );
        assert_eq!(unsafe { rlmesh_device_record_valid(stored.as_ptr(), 3) }, 0);
        let rc = unsafe { rlmesh_device_provision_stored(p, 1, &mut data, &mut len) };
        assert_eq!(rc, RLMESH_DEVICE_OK);
        let reply = unsafe { core::slice::from_raw_parts(data, len) };
        assert_eq!(
            (reply[0], *reply.last().unwrap()),
            (0, 0),
            "a delimited frame"
        );
        // Nothing pending any more.
        let rc = unsafe { rlmesh_device_provision_stored(p, 1, &mut data, &mut len) };
        assert_eq!(rc, RLMESH_DEVICE_ERR_ARG);
        unsafe { rlmesh_device_provision_free(p) };
    }
}
