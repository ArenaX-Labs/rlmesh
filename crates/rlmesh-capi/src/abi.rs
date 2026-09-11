#![allow(unsafe_code)] // FFI: no_mangle exports.

pub(crate) mod status;

/// Binary ABI generation — a monotonic integer, decoupled from the 0.x package
/// semver (which can't express an ABI break: every 0.x shares major 0). Bump it
/// ONLY on a binary-incompatible change: a `repr(C)` layout change or
/// enum-discriminant reorder, an `extern "C"` signature retype, or removing a
/// symbol. Appending a field to a `struct_size`-guarded vtable (the model.rs
/// pattern) is NOT a break and must not bump this. A consumer gates on it via the
/// header's
/// `RLMESH_ABI_VERSION` macro + `rlmesh_abi_check()`; the versioned SONAME
/// (`librlmesh_capi.so.N`) makes the loader enforce the same generation.
pub const RLMESH_ABI_VERSION: u32 = 1;

/// The linked library's ABI generation (see [`RLMESH_ABI_VERSION`]). A consumer
/// compares this against the `RLMESH_ABI_VERSION` macro it compiled against.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_abi_version() -> u32 {
    RLMESH_ABI_VERSION
}

/// Package (marketing) semver major — informational only. Do NOT gate ABI
/// compatibility on this; use [`rlmesh_abi_version`].
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_abi_version_major() -> u32 {
    env!("CARGO_PKG_VERSION_MAJOR").parse::<u32>().unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_abi_version_minor() -> u32 {
    env!("CARGO_PKG_VERSION_MINOR").parse::<u32>().unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_abi_version_patch() -> u32 {
    env!("CARGO_PKG_VERSION_PATCH").parse::<u32>().unwrap_or(0)
}
