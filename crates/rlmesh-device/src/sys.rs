//! A board's allocator and panic: the C library's, which the firmware
//! already links (ESP-IDF's heap, newlib's `abort`).

#![allow(unsafe_code)] // The C library's allocator.

use core::alloc::{GlobalAlloc, Layout};
use core::ffi::c_void;

unsafe extern "C" {
    fn malloc(size: usize) -> *mut c_void;
    fn realloc(ptr: *mut c_void, size: usize) -> *mut c_void;
    fn memalign(align: usize, size: usize) -> *mut c_void;
    fn free(ptr: *mut c_void);
    fn abort() -> !;
}

/// `malloc` aligns to 4 on ESP-IDF; a wider alignment goes to `memalign`,
/// whose blocks `free` also releases.
const MALLOC_ALIGN: usize = 4;

struct CAlloc;

// SAFETY: the C library's allocator meets `GlobalAlloc`'s contract for
// these alignments; null means out of memory.
unsafe impl GlobalAlloc for CAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.align() <= MALLOC_ALIGN {
            unsafe { malloc(layout.size()).cast() }
        } else {
            unsafe { memalign(layout.align(), layout.size()).cast() }
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, _: Layout) {
        unsafe { free(ptr.cast()) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if layout.align() <= MALLOC_ALIGN {
            unsafe { realloc(ptr.cast(), new_size).cast() }
        } else {
            // Copies into a new `memalign` block.
            let Ok(new) = Layout::from_size_align(new_size, layout.align()) else {
                return core::ptr::null_mut();
            };
            let out = unsafe { self.alloc(new) };
            if !out.is_null() {
                unsafe {
                    core::ptr::copy_nonoverlapping(ptr, out, layout.size().min(new_size));
                    self.dealloc(ptr, layout);
                }
            }
            out
        }
    }
}

#[global_allocator]
static ALLOC: CAlloc = CAlloc;

/// A panic resets the board through the C library's `abort` (ESP-IDF prints
/// a backtrace, then restarts), and the restart stops every output.
#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    unsafe { abort() }
}
