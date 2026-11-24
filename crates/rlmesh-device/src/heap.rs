/// Bytes held back: more than any one message allocates on a 32-bit board, but the answer to
/// a robot's declaration (once per attach), which a fresh link's heap has room for.
pub const RESERVE: usize = 4096;

/// Whether a received message may be taken now: the reserve is held, or could be taken again.
pub fn room() -> bool {
    imp::room()
}

#[cfg(all(not(feature = "std"), target_os = "none"))]
mod imp {
    pub(super) use crate::sys::room;
}

#[cfg(not(all(not(feature = "std"), target_os = "none")))]
mod imp {
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
