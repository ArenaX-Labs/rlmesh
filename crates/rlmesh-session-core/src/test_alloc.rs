//! The test binary's allocator: the system's, except that a thread may make
//! every allocation of at least some size fail, as a board's fragmented
//! heap does, to reach the paths that handle it.

// A global allocator is unsafe to implement; this one exists only in the
// test binary.
#![allow(unsafe_code)]

use core::alloc::{GlobalAlloc, Layout};
use core::cell::Cell;
use std::alloc::System;

std::thread_local! {
    /// Allocations of this many bytes or more fail on this thread.
    static FAIL_FROM: Cell<usize> = const { Cell::new(usize::MAX) };
}

fn fails(size: usize) -> bool {
    FAIL_FROM
        .try_with(|from| size >= from.get())
        .unwrap_or(false)
}

struct Failing;

// SAFETY: every call forwards to the system allocator, or fails (null) before
// allocating anything.
unsafe impl GlobalAlloc for Failing {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if fails(layout.size()) {
            return core::ptr::null_mut();
        }
        // SAFETY: the caller's contract, forwarded.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if fails(layout.size()) {
            return core::ptr::null_mut();
        }
        // SAFETY: the caller's contract, forwarded.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller's contract, forwarded.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if fails(new_size) {
            return core::ptr::null_mut();
        }
        // SAFETY: the caller's contract, forwarded.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Failing = Failing;

/// Runs `f` with every allocation of `bytes` or more on this thread failing.
pub(crate) fn failing_from<R>(bytes: usize, f: impl FnOnce() -> R) -> R {
    let before = FAIL_FROM.with(|from| from.replace(bytes));
    let out = f();
    FAIL_FROM.with(|from| from.set(before));
    out
}
