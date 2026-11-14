//! Room on the heap for what one received message makes the robot allocate.
//!
//! Decoding a message is bounded, and its large or message-sized
//! allocations fail rather than abort. What taking it allocates besides (a
//! key installed, a view's bookkeeping, a queued action's slice) is a few
//! small blocks allocated the way Rust always does: on a board, a failure
//! there aborts it. So a board holds a reserve: when an allocation fails,
//! its allocator frees the reserve and tries once more. Each receive path
//! starts only with the reserve held (taken again if it went), and otherwise
//! drops the message as [`crate::robot::Dropped::NoMemory`]: whatever it
//! then allocates past what the heap has comes out of the reserve.

/// Bytes held back: more than any one message allocates on a 32-bit board,
/// but the answer to a robot's declaration (once per attach), which a fresh
/// link's heap has room for.
pub const RESERVE: usize = 4096;

/// Whether a received message may be taken now: the reserve is held, or
/// could be taken again.
pub fn room() -> bool {
    imp::room()
}

#[cfg(all(not(feature = "std"), target_os = "none"))]
mod imp {
    pub(super) use crate::sys::room;
}

#[cfg(not(all(not(feature = "std"), target_os = "none")))]
mod imp {
    // A test build links std even when the crate itself is built without it.
    #[cfg(test)]
    extern crate std;

    #[cfg(test)]
    std::thread_local! {
        /// A test's heap with no room: the reserve gone and not to be had.
        pub(crate) static NONE: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
    }

    /// Off a board an allocation that fails aborts the process anyway.
    pub(super) fn room() -> bool {
        #[cfg(test)]
        if NONE.with(core::cell::Cell::get) {
            return false;
        }
        true
    }
}

/// Runs `f` as if the heap had no room for a message.
#[cfg(test)]
pub(crate) fn without_room<R>(f: impl FnOnce() -> R) -> R {
    imp::NONE.with(|n| n.set(true));
    let out = f();
    imp::NONE.with(|n| n.set(false));
    out
}
