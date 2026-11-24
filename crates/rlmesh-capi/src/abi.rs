#![allow(unsafe_code)]

pub(crate) mod sizing;
pub(crate) mod status;

pub(crate) fn ignore_sigpipe() {
    #[cfg(unix)]
    {
        // A C/C++ host keeps the default SIGPIPE action (terminate), and vectored socket writes cannot opt out,
        // so a peer hanging up mid-write would kill the host instead of failing the write with `EPIPE`.
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

/// Binary ABI generation, independent of the package semver.
pub const RLMESH_ABI_VERSION: u32 = 6;

/// The linked library's ABI generation (see [`RLMESH_ABI_VERSION`]). A consumer
/// compares this against the `RLMESH_ABI_VERSION` macro it compiled against.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_abi_version() -> u32 {
    RLMESH_ABI_VERSION
}
