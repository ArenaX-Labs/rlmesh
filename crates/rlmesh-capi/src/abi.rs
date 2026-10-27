#![allow(unsafe_code)] // FFI: no_mangle exports.

pub(crate) mod sizing;
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

/// Binary ABI generation, independent of the package semver. ABI 6 is the
/// language-parity surface. Its options and callback records follow R-19.
///
/// ABI 6 grew after it was cut (the parity additions of `d70b81ad`) without a
/// bump. The C ABI does not ship in 0.2, so it is not bumped now: the first
/// release that ships it publishes these additions as ABI 7.
pub const RLMESH_ABI_VERSION: u32 = 6;

/// The linked library's ABI generation (see [`RLMESH_ABI_VERSION`]). A consumer
/// compares this against the `RLMESH_ABI_VERSION` macro it compiled against.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_abi_version() -> u32 {
    RLMESH_ABI_VERSION
}
