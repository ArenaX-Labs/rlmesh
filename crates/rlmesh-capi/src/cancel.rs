//! Reusable cross-thread cancellation token (R-18).
#![allow(unsafe_code)]
use crate::abi::status::{CapiError, RLMeshStatus, guard, guard_value};
use crate::value::handle::write_out;
pub struct RLMeshExtCancelToken(pub(crate) rlmesh::ext::CancelToken);
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_ext_cancel_token_new(
    out: *mut *mut RLMeshExtCancelToken,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, std::ptr::null_mut())?;
        write_out(
            out,
            Box::into_raw(Box::new(RLMeshExtCancelToken(
                rlmesh::ext::CancelToken::new(),
            ))),
        )
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_ext_cancel_token_clone(
    token: *const RLMeshExtCancelToken,
    out: *mut *mut RLMeshExtCancelToken,
) -> RLMeshStatus {
    guard(|| {
        write_out(out, std::ptr::null_mut())?;
        let token =
            unsafe { token.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null cancel token"))?;
        write_out(
            out,
            Box::into_raw(Box::new(RLMeshExtCancelToken(token.0.clone()))),
        )
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_ext_cancel_token_cancel(
    token: *mut RLMeshExtCancelToken,
) -> RLMeshStatus {
    guard(|| {
        let token =
            unsafe { token.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null cancel token"))?;
        token.0.cancel();
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_ext_cancel_token_free(token: *mut RLMeshExtCancelToken) {
    guard_value((), || {
        if !token.is_null() {
            drop(unsafe { Box::from_raw(token) });
        }
    });
}
