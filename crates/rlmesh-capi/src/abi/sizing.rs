#![allow(unsafe_code)]
use super::status::CapiError;
mod inputs;

/// A C input whose restricted representations are fully described here.
///
/// # Safety
/// Every bool field (including nested fields) must be listed, every other field must accept all
/// initialized byte representations, and the all-zero byte pattern must be a valid `T` (it stands
/// in for fields a smaller `struct_size` leaves out).
pub(crate) unsafe trait SizedInput: Copy {
    const BOOL_FIELDS: &'static [(usize, &'static str)];
    const MIN_SIZE: usize = std::mem::size_of::<Self>();
    const APPENDED_ENDS: &'static [usize] = &[];
}

pub(crate) unsafe fn read<T: SizedInput>(
    ptr: *const T,
    name: &str,
) -> Result<Option<T>, CapiError> {
    if ptr.is_null() {
        return Ok(None);
    }
    let size = unsafe { ptr.cast::<usize>().read_unaligned() };
    let own = std::mem::size_of::<T>();
    if size < T::MIN_SIZE {
        return Err(CapiError::invalid_arg(format!(
            "{name}.struct_size {size} is below its minimum {}",
            T::MIN_SIZE
        )));
    }
    if size < own && size != T::MIN_SIZE && !T::APPENDED_ENDS.contains(&size) {
        return Err(CapiError::invalid_arg(format!(
            "{name}.struct_size {size} ends inside a field"
        )));
    }
    if size < own {
        let mut value = std::mem::MaybeUninit::<T>::zeroed();
        // SAFETY: the caller declared `size` readable bytes, fewer than T's.
        unsafe {
            std::ptr::copy_nonoverlapping(ptr.cast::<u8>(), value.as_mut_ptr().cast::<u8>(), size);
        }
        for &(offset, field) in T::BOOL_FIELDS {
            // SAFETY: inside the zeroed T.
            if unsafe { value.as_ptr().cast::<u8>().add(offset).read() } > 1 {
                return Err(CapiError::invalid_arg(format!(
                    "{name}.{field} has an invalid bool byte at offset {offset}"
                )));
            }
        }
        // SAFETY: zero is valid for every field (trait contract) and the
        // copied bools are checked.
        return Ok(Some(unsafe { value.assume_init() }));
    }
    if size > own {
        let tail = unsafe { std::slice::from_raw_parts(ptr.cast::<u8>().add(own), size - own) };
        if let Some(offset) = tail.iter().position(|&b| b != 0) {
            return Err(CapiError::type_error(format!(
                "{name} sets byte {} past the {own} this library knows",
                own + offset
            )));
        }
    }
    for &(offset, field) in T::BOOL_FIELDS {
        // SAFETY: the ABI minimum covers every declared bool, and the caller
        // supplies readable initialized bytes. Read u8, never an invalid bool.
        if unsafe { ptr.cast::<u8>().add(offset).read() } > 1 {
            return Err(CapiError::invalid_arg(format!(
                "{name}.{field} has an invalid bool byte at offset {offset}"
            )));
        }
    }
    // SAFETY: all bool representations have been checked before making a T.
    Ok(Some(unsafe { ptr.read_unaligned() }))
}
