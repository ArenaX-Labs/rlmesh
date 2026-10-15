//! Reusable cross-thread cancellation token (R-18).
#![allow(unsafe_code)]
use crate::abi::status::{guard_ptr, guard_value};
pub struct RLMeshExtCancelToken(pub(crate) rlmesh::ext::CancelToken);
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_ext_cancel_token_new() -> *mut RLMeshExtCancelToken {
    guard_ptr(|| {
        Ok(Box::into_raw(Box::new(RLMeshExtCancelToken(
            rlmesh::ext::CancelToken::new(),
        ))))
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_ext_cancel_token_cancel(token: *const RLMeshExtCancelToken) {
    guard_value((), || {
        if let Some(t) = unsafe { token.as_ref() } {
            t.0.cancel();
        }
    });
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_ext_cancel_token_free(token: *mut RLMeshExtCancelToken) {
    guard_value((), || {
        if !token.is_null() {
            drop(unsafe { Box::from_raw(token) });
        }
    });
}
