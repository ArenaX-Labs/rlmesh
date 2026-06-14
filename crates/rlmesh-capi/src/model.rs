//! The model path: a C callback vtable adapted into a core `ModelHandler`, plus
//! a handle that owns a tokio runtime and drives/serves it.
//!
//! Every C callback runs inside `spawn_blocking` so a blocking/CPU-bound callback
//! cannot starve the shared multi-thread runtime (20-review-spec B3). A callback's
//! error is read on its own thread and folded into an `Error` value that travels
//! back to the caller (B1).
#![allow(unsafe_code)] // FFI: raw callback pointers + repr(C) structs.

use std::ffi::{CStr, CString, c_char, c_int, c_void};

use async_trait::async_trait;
use rlmesh::spaces::BinaryPayload;
use rlmesh::{
    ConnectAddress, Error, ModelEpisodeEnd, ModelHandler, ModelLaneReset, ModelObservation,
    ModelWorker, RunLocalOptions,
};

use crate::abi::status::{
    CapiError, RlmeshStatus, clear_last_error, guard, last_error_message, last_error_recoverable,
};
use crate::codec::RlmeshBytes;
use crate::spaces::RlmeshContract;

/// Predict callback: decode `obs`, run the policy, and write the encoded action
/// to `out_action`. Return `RLMESH_OK` (0), or nonzero to decline (set a message
/// via `rlmesh_callback_set_error`).
///
/// The return is read as a plain `c_int` (not the `RlmeshStatus` enum) so an
/// out-of-range value from a C author is not undefined behavior. `out_action`,
/// when set, MUST be a buffer produced by `rlmesh_encode_batch` — the capi
/// reclaims it with its own allocator (do not `malloc` it).
pub type RlmeshPredictFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    obs: *const RlmeshObservation,
    out_action: *mut RlmeshBytes,
) -> c_int;
/// A no-argument lifecycle callback (`on_close`).
pub type RlmeshLifecycleFn = unsafe extern "C" fn(user_data: *mut c_void);
/// A per-lane lifecycle callback carrying `episode_id` + `env_index`.
pub type RlmeshLaneFn =
    unsafe extern "C" fn(user_data: *mut c_void, episode_id: *const c_char, env_index: i32);

/// One sub-environment's slot within a predict request.
#[repr(C)]
pub struct RlmeshRouteSlot {
    /// Episode this slot belongs to (NUL-terminated).
    pub episode_id: *const c_char,
    pub env_index: i32,
    pub step: i64,
    /// First step of a new episode.
    pub reset: bool,
}

/// What a predict callback receives. Pointers are valid only for the duration of
/// the call.
#[repr(C)]
pub struct RlmeshObservation {
    /// Encoded observation payload, or NULL.
    pub observation: *const u8,
    pub observation_len: usize,
    /// Spaces/metadata to decode/encode against, or NULL on an unconfigured route.
    pub contract: *const RlmeshContract,
    pub num_envs: u32,
    /// NUL-terminated.
    pub session_id: *const c_char,
    /// NUL-terminated.
    pub route_id: *const c_char,
    /// NUL-terminated.
    pub request_id: *const c_char,
    pub slots: *const RlmeshRouteSlot,
    pub num_slots: usize,
}

/// The model callback vtable. Set `struct_size = sizeof(RlmeshModelVtable)`;
/// fields beyond that are ignored (append-only). `predict` is required.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RlmeshModelVtable {
    /// Size of the struct the caller compiled against.
    pub struct_size: usize,
    /// Required: map an observation to an encoded action.
    pub predict: Option<RlmeshPredictFn>,
    pub on_lane_reset: Option<RlmeshLaneFn>,
    pub on_episode_end: Option<RlmeshLaneFn>,
    pub on_close: Option<RlmeshLifecycleFn>,
}

/// Set the current callback's error message + recoverability (read by the capi
/// on this thread when the callback returns nonzero).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_callback_set_error(message: *const c_char, recoverable: bool) {
    let message = if message.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(message) }
            .to_string_lossy()
            .into_owned()
    };
    crate::abi::status::store_last_error(&message, recoverable);
}

/// A `*mut c_void` the C author guarantees is safe to use from a tokio worker
/// thread (callbacks do not run on the creating thread; 20-review-spec M4).
#[derive(Clone, Copy)]
struct UserData(*mut c_void);
// SAFETY: the C author guarantees `user_data` is thread-migration-safe.
unsafe impl Send for UserData {}

/// An owned model handle: the callback vtable plus a tokio runtime.
pub struct RlmeshModel {
    vtable: RlmeshModelVtable,
    user_data: UserData,
    runtime: tokio::runtime::Runtime,
}

struct CModelHandler {
    vtable: RlmeshModelVtable,
    user_data: UserData,
}

#[async_trait]
impl ModelHandler for CModelHandler {
    async fn predict(&mut self, observation: ModelObservation) -> rlmesh::Result<BinaryPayload> {
        let Some(predict) = self.vtable.predict else {
            return Err(Error::model("model vtable has no predict function"));
        };
        let user_data = self.user_data;
        let payload = observation.observation.map(|payload| payload.data);
        let contract = observation.env_contract;
        let num_envs = observation.num_envs as u32;
        let session_id = observation.route.session_id;
        let route_id = observation.route.route_id;
        let request_id = observation.route.request_id;
        let slots = observation.route.slots;

        let bytes = tokio::task::spawn_blocking(move || -> Result<Vec<u8>, Error> {
            // Capture the whole (`Send`) `UserData`, not just the bare pointer field.
            let user_data = user_data;
            let session = cstring(&session_id);
            let route = cstring(&route_id);
            let request = cstring(&request_id);
            let slot_ids: Vec<CString> =
                slots.iter().map(|slot| cstring(&slot.episode_id)).collect();
            let slot_views: Vec<RlmeshRouteSlot> = slots
                .iter()
                .zip(&slot_ids)
                .map(|(slot, id)| RlmeshRouteSlot {
                    episode_id: id.as_ptr(),
                    env_index: slot.env_index,
                    step: slot.step,
                    reset: slot.reset,
                })
                .collect();
            let contract = contract.map(RlmeshContract);
            let (obs_ptr, obs_len) = payload
                .as_ref()
                .map_or((std::ptr::null(), 0), |bytes| (bytes.as_ptr(), bytes.len()));
            let view = RlmeshObservation {
                observation: obs_ptr,
                observation_len: obs_len,
                contract: contract
                    .as_ref()
                    .map_or(std::ptr::null(), |c| c as *const RlmeshContract),
                num_envs,
                session_id: session.as_ptr(),
                route_id: route.as_ptr(),
                request_id: request.as_ptr(),
                slots: slot_views.as_ptr(),
                num_slots: slot_views.len(),
            };

            let mut out = RlmeshBytes {
                data: std::ptr::null_mut(),
                len: 0,
                cap: 0,
            };
            // Clear first so a decline that doesn't set an error can't report a
            // stale message left on this reused pool thread (B3).
            clear_last_error();
            let status = unsafe { predict(user_data.0, &view, &mut out) };
            if status == 0 {
                Ok(unsafe { out.into_vec() })
            } else {
                let message = last_error_message();
                let message = if message.is_empty() {
                    "model predict declined".to_string()
                } else {
                    message
                };
                Err(if last_error_recoverable() {
                    Error::model_recoverable(message)
                } else {
                    Error::model(message)
                })
            }
        })
        .await
        .map_err(|err| Error::Internal(format!("predict task panicked: {err}")))??;

        Ok(BinaryPayload { data: bytes })
    }

    async fn on_lane_reset(&mut self, event: ModelLaneReset) -> rlmesh::Result<()> {
        lane_callback(
            self.vtable.on_lane_reset,
            self.user_data,
            event.episode_id,
            event.env_index,
            "on_lane_reset",
        )
        .await
    }

    async fn on_episode_end(&mut self, event: ModelEpisodeEnd) -> rlmesh::Result<()> {
        lane_callback(
            self.vtable.on_episode_end,
            self.user_data,
            event.episode_id,
            event.env_index,
            "on_episode_end",
        )
        .await
    }

    async fn on_close(&mut self) -> rlmesh::Result<()> {
        let Some(callback) = self.vtable.on_close else {
            return Ok(());
        };
        let user_data = self.user_data;
        tokio::task::spawn_blocking(move || {
            // Capture the whole (`Send`) `UserData`, not just the bare pointer field.
            let user_data = user_data;
            unsafe { callback(user_data.0) };
        })
        .await
        .map_err(|err| Error::Internal(format!("on_close task panicked: {err}")))
    }
}

async fn lane_callback(
    callback: Option<RlmeshLaneFn>,
    user_data: UserData,
    episode_id: String,
    env_index: i32,
    phase: &'static str,
) -> rlmesh::Result<()> {
    let Some(callback) = callback else {
        return Ok(());
    };
    tokio::task::spawn_blocking(move || {
        // Capture the whole (`Send`) `UserData`, not just the bare pointer field.
        let user_data = user_data;
        let id = cstring(&episode_id);
        unsafe { callback(user_data.0, id.as_ptr(), env_index) };
    })
    .await
    .map_err(|err| Error::Internal(format!("{phase} task panicked: {err}")))
}

/// Read field `T` at `offset` only when `struct_size` covers it (else None), so a
/// caller that compiled against a smaller header is never read past.
///
/// # Safety
/// `base` points at the caller's vtable allocation of at least `struct_size` bytes.
unsafe fn vtable_field<T: Copy>(base: *const u8, struct_size: usize, offset: usize) -> Option<T> {
    (struct_size >= offset.saturating_add(std::mem::size_of::<T>()))
        .then(|| unsafe { base.add(offset).cast::<T>().read_unaligned() })
}

/// Read the caller's vtable honoring its `struct_size` (append-only contract):
/// callbacks past `struct_size` are absent, not garbage read out of bounds.
///
/// # Safety
/// `ptr` is non-NULL and points at a vtable whose first `struct_size` bytes are valid.
unsafe fn read_vtable(ptr: *const RlmeshModelVtable) -> Result<RlmeshModelVtable, CapiError> {
    let base = ptr.cast::<u8>();
    // struct_size is the first repr(C) field (offset 0); the caller set it.
    let struct_size = unsafe { (*ptr).struct_size };
    if struct_size == 0 {
        return Err(CapiError::invalid_arg("vtable struct_size is 0"));
    }
    let predict = match unsafe {
        vtable_field::<Option<RlmeshPredictFn>>(
            base,
            struct_size,
            std::mem::offset_of!(RlmeshModelVtable, predict),
        )
    } {
        None => {
            return Err(CapiError::invalid_arg(
                "vtable struct_size too small for predict",
            ));
        }
        Some(None) => return Err(CapiError::invalid_arg("vtable predict is null")),
        some => some.flatten(),
    };
    Ok(RlmeshModelVtable {
        struct_size,
        predict,
        on_lane_reset: unsafe {
            vtable_field::<Option<RlmeshLaneFn>>(
                base,
                struct_size,
                std::mem::offset_of!(RlmeshModelVtable, on_lane_reset),
            )
        }
        .flatten(),
        on_episode_end: unsafe {
            vtable_field::<Option<RlmeshLaneFn>>(
                base,
                struct_size,
                std::mem::offset_of!(RlmeshModelVtable, on_episode_end),
            )
        }
        .flatten(),
        on_close: unsafe {
            vtable_field::<Option<RlmeshLifecycleFn>>(
                base,
                struct_size,
                std::mem::offset_of!(RlmeshModelVtable, on_close),
            )
        }
        .flatten(),
    })
}

/// Create a model from a callback vtable. `predict` and `struct_size` are
/// required.
///
/// # Safety
/// `vtable` must be valid; `user_data` is passed unchanged to every callback.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_model_new(
    vtable: *const RlmeshModelVtable,
    user_data: *mut c_void,
    out: *mut *mut RlmeshModel,
) -> RlmeshStatus {
    guard(|| {
        if vtable.is_null() {
            return Err(CapiError::invalid_arg("null vtable"));
        }
        let vtable = unsafe { read_vtable(vtable) }?;
        let out = unsafe { out.as_mut() }.ok_or_else(|| CapiError::invalid_arg("null out"))?;
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|err| CapiError::internal(format!("failed to build runtime: {err}")))?;
        let model = Box::new(RlmeshModel {
            vtable,
            user_data: UserData(user_data),
            runtime,
        });
        *out = Box::into_raw(model);
        Ok(())
    })
}

/// Drive the model against a remote environment until it ends. Blocking.
///
/// # Safety
/// `model` must be a live handle; `env_address` a valid C string or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_model_run_local(
    model: *mut RlmeshModel,
    env_address: *const c_char,
) -> RlmeshStatus {
    run_local(model, env_address, None)
}

/// Drive the model against a remote environment for `max_episodes`. Blocking.
///
/// # Safety
/// See `rlmesh_model_run_local`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_model_run_local_for_episodes(
    model: *mut RlmeshModel,
    env_address: *const c_char,
    max_episodes: u64,
) -> RlmeshStatus {
    run_local(model, env_address, Some(max_episodes))
}

fn run_local(
    model: *mut RlmeshModel,
    env_address: *const c_char,
    max_episodes: Option<u64>,
) -> RlmeshStatus {
    guard(|| {
        let model =
            unsafe { model.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null model"))?;
        let address = cstr_to_str(env_address)?;
        let address = ConnectAddress::parse(address)
            .map_err(|err| CapiError::invalid_arg(format!("invalid env address: {err}")))?;
        let handler = CModelHandler {
            vtable: model.vtable,
            user_data: model.user_data,
        };
        let mut options = RunLocalOptions::new(address);
        if let Some(max) = max_episodes {
            options = options.for_episodes(max);
        }
        model
            .runtime
            .block_on(async move { ModelWorker::new(handler).run_local_async(options).await })
            .map_err(CapiError::from)?;
        Ok(())
    })
}

/// Free a model handle.
///
/// # Safety
/// `model` must be NULL or a handle this thread owns and has not freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_model_free(model: *mut RlmeshModel) {
    if !model.is_null() {
        drop(unsafe { Box::from_raw(model) });
    }
}

fn cstring(text: &str) -> CString {
    let bytes: Vec<u8> = text.bytes().filter(|&byte| byte != 0).collect();
    CString::new(bytes).unwrap_or_default()
}

fn cstr_to_str<'a>(ptr: *const c_char) -> Result<&'a str, CapiError> {
    if ptr.is_null() {
        return Err(CapiError::invalid_arg("null string"));
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map_err(|_| CapiError::invalid_arg("string is not UTF-8"))
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" fn noop_predict(
        _: *mut c_void,
        _: *const RlmeshObservation,
        _: *mut RlmeshBytes,
    ) -> c_int {
        0
    }
    unsafe extern "C" fn noop_lane(_: *mut c_void, _: *const c_char, _: i32) {}
    unsafe extern "C" fn noop_close(_: *mut c_void) {}

    fn full_vtable() -> RlmeshModelVtable {
        RlmeshModelVtable {
            struct_size: std::mem::size_of::<RlmeshModelVtable>(),
            predict: Some(noop_predict),
            on_lane_reset: Some(noop_lane),
            on_episode_end: Some(noop_lane),
            on_close: Some(noop_close),
        }
    }

    #[test]
    fn read_vtable_honors_truncated_struct_size() {
        // A caller that compiled against only {struct_size, predict} reports a
        // smaller size; the later callbacks must read as None even though they are
        // non-null in this fully-allocated struct (the OOB-read regression guard).
        let mut vtable = full_vtable();
        vtable.struct_size = std::mem::offset_of!(RlmeshModelVtable, on_lane_reset);
        let Ok(read) = (unsafe { read_vtable(&vtable) }) else {
            panic!("predict must be covered");
        };
        assert!(read.predict.is_some());
        assert!(read.on_lane_reset.is_none());
        assert!(read.on_episode_end.is_none());
        assert!(read.on_close.is_none());
    }

    #[test]
    fn read_vtable_rejects_size_too_small_for_predict() {
        let mut vtable = full_vtable();
        vtable.struct_size = std::mem::offset_of!(RlmeshModelVtable, predict);
        assert!(unsafe { read_vtable(&vtable) }.is_err());
        vtable.struct_size = 0;
        assert!(unsafe { read_vtable(&vtable) }.is_err());
    }

    #[test]
    fn read_vtable_full_struct_reads_every_field() {
        let Ok(read) = (unsafe { read_vtable(&full_vtable()) }) else {
            panic!("full struct must read");
        };
        assert!(read.predict.is_some());
        assert!(read.on_lane_reset.is_some());
        assert!(read.on_episode_end.is_some());
        assert!(read.on_close.is_some());
    }
}
