//! ABI version getters. The header carries matching `RLMESH_ABI_VERSION_*`
//! macros; a test asserts they agree with the crate version (13-packaging P-A1).
#![allow(unsafe_code)] // FFI: no_mangle exports.

/// ABI major version (the crate's major). A loaded plugin compares this to
/// refuse a too-old host (20-review-spec M2).
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_abi_version_major() -> u32 {
    env!("CARGO_PKG_VERSION_MAJOR").parse::<u32>().unwrap_or(0)
}

/// ABI minor version (the crate's minor).
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_abi_version_minor() -> u32 {
    env!("CARGO_PKG_VERSION_MINOR").parse::<u32>().unwrap_or(0)
}

/// ABI patch version (the crate's patch).
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_abi_version_patch() -> u32 {
    env!("CARGO_PKG_VERSION_PATCH").parse::<u32>().unwrap_or(0)
}
