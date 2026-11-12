//! A board's allocator and panic: the C library's, which the firmware
//! already links (ESP-IDF's heap, newlib's `abort`).

#![allow(unsafe_code)] // The C library's allocator.

use core::alloc::{GlobalAlloc, Layout};
use core::ffi::{c_char, c_int, c_void};
use core::sync::atomic::{AtomicPtr, Ordering};

unsafe extern "C" {
    fn malloc(size: usize) -> *mut c_void;
    fn realloc(ptr: *mut c_void, size: usize) -> *mut c_void;
    fn memalign(align: usize, size: usize) -> *mut c_void;
    fn free(ptr: *mut c_void);
    fn abort() -> !;
    fn printf(format: *const c_char, ...) -> c_int;
}

/// `malloc` aligns to 4 on ESP-IDF; a wider alignment goes to `memalign`,
/// whose blocks `free` also releases.
const MALLOC_ALIGN: usize = 4;

struct CAlloc;

// SAFETY: the C library's allocator meets `GlobalAlloc`'s contract for
// these alignments; null means out of memory.
unsafe impl GlobalAlloc for CAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let block = unsafe { raw_alloc(layout) };
        if block.is_null() && released() {
            return unsafe { raw_alloc(layout) };
        }
        block
    }

    unsafe fn dealloc(&self, ptr: *mut u8, _: Layout) {
        unsafe { free(ptr.cast()) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if layout.align() <= MALLOC_ALIGN {
            let block = unsafe { realloc(ptr.cast(), new_size) };
            if block.is_null() && released() {
                return unsafe { realloc(ptr.cast(), new_size).cast() };
            }
            block.cast()
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

unsafe fn raw_alloc(layout: Layout) -> *mut u8 {
    if layout.align() <= MALLOC_ALIGN {
        unsafe { malloc(layout.size()).cast() }
    } else {
        unsafe { memalign(layout.align(), layout.size()).cast() }
    }
}

/// The reserve ([`crate::heap`]): a block held back, freed when an
/// allocation fails so that it can succeed. Null while it is not held.
static RESERVE: AtomicPtr<c_void> = AtomicPtr::new(core::ptr::null_mut());

/// Frees the reserve, if held: whether there was one to free.
fn released() -> bool {
    let block = RESERVE.swap(core::ptr::null_mut(), Ordering::AcqRel);
    if block.is_null() {
        return false;
    }
    unsafe { free(block) };
    true
}

/// Whether the reserve is held, taking it again if the heap has room.
pub(crate) fn room() -> bool {
    if !RESERVE.load(Ordering::Acquire).is_null() {
        return true;
    }
    let block = unsafe { malloc(crate::heap::RESERVE) };
    if block.is_null() {
        return false;
    }
    if RESERVE
        .compare_exchange(
            core::ptr::null_mut(),
            block,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .is_err()
    {
        // Another task took it again meanwhile.
        unsafe { free(block) };
    }
    true
}

/// A panic prints where it happened (no message: formatting one would pull
/// `core::fmt` into flash), then resets the board through the C library's
/// `abort` (ESP-IDF prints a backtrace, then restarts), which stops every
/// output.
#[panic_handler]
fn panic(info: &core::panic::PanicInfo<'_>) -> ! {
    if let Some(at) = info.location() {
        let file = at.file();
        // SAFETY: `%.*s` reads exactly `len` bytes of `file`.
        unsafe {
            printf(
                c"rlmesh-device panicked at %.*s:%u\n".as_ptr(),
                c_int::try_from(file.len()).unwrap_or(0),
                file.as_ptr(),
                at.line(),
            );
        }
    }
    unsafe { abort() }
}
