//! R-19 sized inputs. ABI 5 introduces every current field at the minimum.
#![allow(unsafe_code)] // Read only caller-declared bytes, after checking the minimum.
use super::status::CapiError;
mod inputs;

/// A C input whose restricted representations are fully described here.
///
/// # Safety
/// Every bool field (including nested fields) must be listed, and every other
/// field must accept all initialized byte representations used by C callers.
pub(crate) unsafe trait SizedInput: Copy {
    const BOOL_FIELDS: &'static [(usize, &'static str)];
}

/// Validate the ABI-5 minimum and the zero unknown tail before reading `T`.
/// This deliberately supports no truncated current fields: their ABI-5 sizeof
/// is the minimum. Future fields must be read individually with whole-field bounds.
pub(crate) unsafe fn read<T: SizedInput>(
    ptr: *const T,
    name: &str,
) -> Result<Option<T>, CapiError> {
    if ptr.is_null() {
        return Ok(None);
    }
    let size = unsafe { ptr.cast::<usize>().read_unaligned() };
    let own = std::mem::size_of::<T>();
    if size < own {
        return Err(CapiError::invalid_arg(format!(
            "{name}.struct_size {size} is below its minimum {own}"
        )));
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
