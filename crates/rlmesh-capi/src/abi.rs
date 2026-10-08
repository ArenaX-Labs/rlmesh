#![allow(unsafe_code)] // FFI: no_mangle exports.

pub(crate) mod status;

/// Ignore SIGPIPE in the host process, unless the host installed its own
/// handler. Called when a model or env handle is created.
///
/// A Rust binary starts with SIGPIPE ignored, and so does Python, but a C/C++
/// host keeps the default action: terminate. The gRPC transport writes sockets
/// with vectored writes, which (unlike `send`) cannot opt out of the signal, so
/// a peer that hangs up mid-write would kill the host instead of failing that
/// one write with `EPIPE`.
pub(crate) fn ignore_sigpipe() {
    #[cfg(unix)]
    {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            // SAFETY: querying and then setting the process signal disposition;
            // both calls take valid pointers or constants.
            unsafe {
                let mut current: libc::sigaction = std::mem::zeroed();
                if libc::sigaction(libc::SIGPIPE, std::ptr::null(), &mut current) == 0
                    && current.sa_sigaction == libc::SIG_DFL
                {
                    libc::signal(libc::SIGPIPE, libc::SIG_IGN);
                }
            }
        });
    }
}

/// Binary ABI generation — a monotonic integer, decoupled from the 0.x package
/// semver (which can't express an ABI break: every 0.x shares major 0). Bump it
/// ONLY on a binary-incompatible change: a `repr(C)` layout change or
/// enum-discriminant reorder, an `extern "C"` signature retype, or removing a
/// symbol. Appending a field to a `struct_size`-guarded vtable (the model.rs
/// pattern) is NOT a break and must not bump this — but `RlmeshServeOptions`
/// carries no `struct_size`, so growing IT is a layout change and does bump this
/// (generation 4 added `RlmeshServeOptions.workflow_edition`). A consumer gates
/// on it via the header's
/// `RLMESH_ABI_VERSION` macro + `rlmesh_abi_check()`; the versioned SONAME
/// (`librlmesh_capi.so.N`) makes the loader enforce the same generation.
pub const RLMESH_ABI_VERSION: u32 = 4;

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
