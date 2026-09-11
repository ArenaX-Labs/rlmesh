//! The model path: a C callback vtable adapted into a core `ModelHandler`, plus
//! a handle that owns a tokio runtime and drives/serves it.
//!
//! Every C callback runs inside `spawn_blocking` so a blocking/CPU-bound callback
//! cannot starve the shared multi-thread runtime. A callback's error is read on
//! its own thread and folded into an `Error` value that travels back to the caller.
#![allow(unsafe_code)] // FFI: raw callback pointers + repr(C) structs.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::time::Duration;

use async_trait::async_trait;
use rlmesh::spaces::{SpaceValue, contains};
use rlmesh::{
    BindAddress, ConnectAddress, Error, ModelHandler, ModelObservation, ModelWorker,
    RunLocalOptions, ServeModelOptions, ServeOptions,
};

use crate::abi::status::{
    CapiError, RlmeshStatus, clear_last_error, guard, last_error_message, last_error_recoverable,
};
use crate::spaces::RlmeshContract;
use crate::value::handle::RlmeshValue;

/// Predict callback: read the decoded per-env observations in `obs`, run the
/// policy, and write one owned action value per env into `out_actions`
/// (`num_envs` slots, pre-zeroed). The capi takes ownership of every value
/// written. Return `RLMESH_OK` (0), or nonzero to decline (set a message via
/// `rlmesh_callback_set_error`); on a nonzero return any values already written
/// are freed by the capi.
///
/// The return is read as a plain `c_int` (not the `RlmeshStatus` enum) so an
/// out-of-range value from a C author is not undefined behavior.
pub type RlmeshPredictFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    obs: *const RlmeshObservation,
    out_actions: *mut *mut RlmeshValue,
) -> c_int;
/// A no-argument lifecycle callback (`on_close`).
pub type RlmeshLifecycleFn = unsafe extern "C" fn(user_data: *mut c_void);
/// Episode teardown callback: `episode_id` NULL means every episode of `env_id`.
pub type RlmeshEpisodeEndFn =
    unsafe extern "C" fn(user_data: *mut c_void, env_id: *const c_char, episode_id: *const c_char);

/// One row's episode identity within a predict request.
#[repr(C)]
pub struct RlmeshEpisode {
    /// Runtime-minted episode id (NUL-terminated, never repeats).
    pub id: *const c_char,
    /// Whether `seed` carries an explicit reset seed.
    pub seeded: bool,
    /// The explicit reset seed; only meaningful when `seeded`.
    pub seed: i64,
}

/// What a predict callback receives. Pointers are valid only for the duration of
/// the call.
#[repr(C)]
pub struct RlmeshObservation {
    /// `num_envs` decoded observation values (borrowed), or NULL when absent.
    pub observations: *const *const RlmeshValue,
    /// Rows in this batch; also the length of `episodes` and of `out_actions`.
    pub num_envs: usize,
    /// Spaces/metadata for the route, or NULL on an unconfigured route.
    pub contract: *const RlmeshContract,
    /// NUL-terminated.
    pub session_id: *const c_char,
    /// NUL-terminated.
    pub env_id: *const c_char,
    /// NUL-terminated.
    pub request_id: *const c_char,
    /// `num_envs` entries; row `i` of `observations` belongs to `episodes[i]`.
    pub episodes: *const RlmeshEpisode,
}

/// The model callback vtable. Set `struct_size = sizeof(RlmeshModelVtable)`;
/// fields beyond that are ignored (append-only). `predict` is required.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RlmeshModelVtable {
    /// Size of the struct the caller compiled against.
    pub struct_size: usize,
    /// Required: map a batch of observations to one action per row.
    pub predict: Option<RlmeshPredictFn>,
    /// Optional: drop per-episode state.
    pub on_episode_end: Option<RlmeshEpisodeEndFn>,
    /// Optional: the worker/session is shutting down.
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
/// thread (callbacks do not run on the creating thread).
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

/// Read the callback's error slot into an `Error` after a nonzero return.
fn callback_error(fallback: &str) -> Error {
    let message = last_error_message();
    let message = if message.is_empty() {
        fallback.to_string()
    } else {
        message
    };
    if last_error_recoverable() {
        Error::model_recoverable(message)
    } else {
        Error::model(message)
    }
}

#[async_trait]
impl ModelHandler for CModelHandler {
    async fn predict(&mut self, observation: ModelObservation) -> rlmesh::Result<Vec<SpaceValue>> {
        let Some(predict) = self.vtable.predict else {
            return Err(Error::model("model vtable has no predict function"));
        };
        let user_data = self.user_data;
        // Decode on the async side (pure Rust, cheap) so the C callback only ever
        // sees typed values — the wire framing never crosses the ABI.
        let lanes = if observation.observation.is_some() {
            Some(observation.decoded_lanes()?)
        } else {
            None
        };
        let num_envs = observation.num_envs;
        let contract = observation.env_contract;
        let route = observation.route;

        tokio::task::spawn_blocking(move || -> Result<Vec<SpaceValue>, Error> {
            // Capture the whole (`Send`) `UserData`, not just the bare pointer field.
            let user_data = user_data;
            let session = cstring(&route.session_id);
            let env = cstring(&route.env_id);
            let request = cstring(&route.request_id);
            let episode_ids: Vec<CString> = route
                .episodes
                .iter()
                .map(|episode| cstring(&episode.episode_id))
                .collect();
            let episodes: Vec<RlmeshEpisode> = route
                .episodes
                .iter()
                .zip(&episode_ids)
                .map(|(episode, id)| RlmeshEpisode {
                    id: id.as_ptr(),
                    seeded: episode.seed.is_some(),
                    seed: episode.seed.unwrap_or_default(),
                })
                .collect();
            let lane_ptrs: Option<Vec<*const RlmeshValue>> = lanes.as_ref().map(|lanes| {
                lanes
                    .iter()
                    .map(|value| std::ptr::from_ref(value).cast::<RlmeshValue>())
                    .collect()
            });
            // `RlmeshContract` is repr(transparent) over `EnvContract`: borrow the
            // shared Arc's contract for the call rather than cloning it per predict.
            let contract_ptr = contract.as_deref().map_or(std::ptr::null(), |c| {
                std::ptr::from_ref(c).cast::<RlmeshContract>()
            });
            let view = RlmeshObservation {
                observations: lane_ptrs
                    .as_ref()
                    .map_or(std::ptr::null(), |ptrs| ptrs.as_ptr()),
                num_envs,
                contract: contract_ptr,
                session_id: session.as_ptr(),
                env_id: env.as_ptr(),
                request_id: request.as_ptr(),
                episodes: episodes.as_ptr(),
            };

            let mut out: Vec<*mut RlmeshValue> = vec![std::ptr::null_mut(); num_envs];
            // Clear first so a decline that doesn't set an error can't report a
            // stale message left on this reused pool thread.
            clear_last_error();
            let status = unsafe { predict(user_data.0, &view, out.as_mut_ptr()) };
            // Reclaim every written handle whatever the status: on success they are
            // the actions, on failure they must not leak.
            let actions: Vec<Option<SpaceValue>> = out
                .into_iter()
                .map(|ptr| (!ptr.is_null()).then(|| unsafe { Box::from_raw(ptr) }.0))
                .collect();
            if status != 0 {
                return Err(callback_error("model predict declined"));
            }
            let action_space = contract.as_deref().and_then(|c| c.action_space.as_ref());
            actions
                .into_iter()
                .enumerate()
                .map(|(row, action)| {
                    let action = action.ok_or_else(|| {
                        Error::model(format!("predict returned OK but left action {row} unset"))
                    })?;
                    if let Some(space) = action_space {
                        contains(space, &action).map_err(|err| {
                            Error::model(format!("action {row} outside action space: {err}"))
                        })?;
                    }
                    Ok(action)
                })
                .collect()
        })
        .await
        .map_err(|err| Error::Internal(format!("predict task panicked: {err}")))?
    }

    async fn reset_adapter(
        &mut self,
        env_id: &str,
        episode_ids: Vec<String>,
    ) -> rlmesh::Result<()> {
        let Some(callback) = self.vtable.on_episode_end else {
            return Ok(());
        };
        let user_data = self.user_data;
        let env_id = env_id.to_string();
        tokio::task::spawn_blocking(move || {
            // Capture the whole (`Send`) `UserData`, not just the bare pointer field.
            let user_data = user_data;
            let env = cstring(&env_id);
            if episode_ids.is_empty() {
                unsafe { callback(user_data.0, env.as_ptr(), std::ptr::null()) };
            }
            for id in episode_ids {
                let id = cstring(&id);
                unsafe { callback(user_data.0, env.as_ptr(), id.as_ptr()) };
            }
        })
        .await
        .map_err(|err| Error::Internal(format!("on_episode_end task panicked: {err}")))
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
        on_episode_end: unsafe {
            vtable_field::<Option<RlmeshEpisodeEndFn>>(
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

/// Options for `rlmesh_model_run_local`. Pass NULL for defaults (run until the
/// environment ends, unseeded). A 0 `max_episodes` means "unlimited".
#[repr(C)]
pub struct RlmeshRunOptions {
    /// Stop after this many episodes (0 = until the env ends).
    pub max_episodes: u64,
    /// Whether `base_seed` is set.
    pub seeded: bool,
    /// Seed for episode 0; later episodes derive from it.
    pub base_seed: i64,
}

/// Drive the model against a remote environment. Blocking — returns when the
/// run ends. `options` may be NULL for defaults.
///
/// # Safety
/// `model` must be a live handle; `env_address` a valid C string; `options` NULL
/// or a valid `RlmeshRunOptions`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_model_run_local(
    model: *mut RlmeshModel,
    env_address: *const c_char,
    options: *const RlmeshRunOptions,
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
        let mut run = RunLocalOptions::new(address);
        if let Some(options) = unsafe { options.as_ref() } {
            if options.max_episodes != 0 {
                run = run.for_episodes(options.max_episodes);
            }
            if options.seeded {
                run = run.base_seed(options.base_seed);
            }
        }
        model
            .runtime
            .block_on(async move { ModelWorker::new(handler).run_local_async(run).await })
            .map_err(CapiError::from)?;
        Ok(())
    })
}

/// Serve options for `rlmesh_model_serve`. A NULL pointer means all defaults (no
/// auth, no remote shutdown, no timeouts — serves until the process is killed). A
/// 0 timeout / concurrency means "unset".
#[repr(C)]
pub struct RlmeshServeOptions {
    /// Bearer token required on requests; NULL or "" disables auth.
    pub token: *const c_char,
    /// Honor a client-issued shutdown request.
    pub allow_remote_shutdown: bool,
    /// Shut down after this many ms with no activity (0 = never).
    pub idle_timeout_ms: u64,
    /// Grace period for in-flight requests on shutdown (0 = unset).
    pub drain_timeout_ms: u64,
    /// Deadline for the handler's close hook (0 = unset).
    pub close_timeout_ms: u64,
    /// Max concurrent predicts (0 = default).
    pub predict_concurrency: usize,
}

fn serve_model_options(
    bind: BindAddress,
    options: *const RlmeshServeOptions,
) -> Result<ServeModelOptions, CapiError> {
    let mut model_options = ServeModelOptions::new(bind);
    let Some(options) = (unsafe { options.as_ref() }) else {
        return Ok(model_options);
    };
    if !options.token.is_null() {
        model_options = model_options.token(cstr_to_str(options.token)?);
    }
    let ms = |value: u64| (value != 0).then(|| Duration::from_millis(value));
    model_options = model_options.serve_options(ServeOptions {
        allow_remote_shutdown: options.allow_remote_shutdown,
        idle_timeout: ms(options.idle_timeout_ms),
        drain_timeout: ms(options.drain_timeout_ms),
        close_timeout: ms(options.close_timeout_ms),
        predict_concurrency: (options.predict_concurrency != 0)
            .then_some(options.predict_concurrency),
        ..ServeOptions::default()
    });
    Ok(model_options)
}

/// Serve the model as a `ModelService` endpoint at `bind_address` (e.g.
/// `tcp://127.0.0.1:50051` or `unix:///path.sock`). Blocking — returns when the
/// server stops (a remote shutdown request or an idle timeout). The same vtable
/// backs every predict, exactly as `rlmesh_model_run_local`. `options` may be
/// NULL for defaults.
///
/// # Safety
/// `model` must be a live handle; `bind_address` a valid C string; `options` NULL
/// or a valid `RlmeshServeOptions`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_model_serve(
    model: *mut RlmeshModel,
    bind_address: *const c_char,
    options: *const RlmeshServeOptions,
) -> RlmeshStatus {
    guard(|| {
        let model =
            unsafe { model.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null model"))?;
        let address = cstr_to_str(bind_address)?;
        let bind = BindAddress::parse(address)
            .map_err(|err| CapiError::invalid_arg(format!("invalid bind address: {err}")))?;
        let model_options = serve_model_options(bind, options)?;
        let handler = CModelHandler {
            vtable: model.vtable,
            user_data: model.user_data,
        };
        model
            .runtime
            .block_on(async move { ModelWorker::new(handler).serve_async(model_options).await })
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
    use std::sync::Arc;

    use rlmesh::spaces::{EnvContract, spaces::DiscreteBuilder};
    use rlmesh::{EpisodeInfo, ModelRouteContext};

    use super::*;
    use crate::value::handle::rlmesh_value_discrete;

    unsafe extern "C" fn noop_predict(
        _: *mut c_void,
        _: *const RlmeshObservation,
        _: *mut *mut RlmeshValue,
    ) -> c_int {
        0
    }
    unsafe extern "C" fn noop_episode_end(_: *mut c_void, _: *const c_char, _: *const c_char) {}
    unsafe extern "C" fn noop_close(_: *mut c_void) {}

    /// Echo policy: action = observation (Discrete), so the round trip is checkable.
    unsafe extern "C" fn echo_predict(
        _: *mut c_void,
        obs: *const RlmeshObservation,
        out: *mut *mut RlmeshValue,
    ) -> c_int {
        let obs = unsafe { &*obs };
        for row in 0..obs.num_envs {
            let value = unsafe { &**obs.observations.add(row) };
            let SpaceValue::Discrete(n) = &value.0 else {
                return RlmeshStatus::InvalidValue as c_int;
            };
            unsafe { *out.add(row) = rlmesh_value_discrete(*n) };
        }
        0
    }

    /// Writes row 0 then declines: the capi must free what was written.
    unsafe extern "C" fn half_then_fail(
        _: *mut c_void,
        _: *const RlmeshObservation,
        out: *mut *mut RlmeshValue,
    ) -> c_int {
        unsafe { *out = rlmesh_value_discrete(1) };
        unsafe { rlmesh_callback_set_error(c"nope".as_ptr(), true) };
        RlmeshStatus::Model as c_int
    }

    fn full_vtable() -> RlmeshModelVtable {
        RlmeshModelVtable {
            struct_size: std::mem::size_of::<RlmeshModelVtable>(),
            predict: Some(noop_predict),
            on_episode_end: Some(noop_episode_end),
            on_close: Some(noop_close),
        }
    }

    fn handler(predict: RlmeshPredictFn) -> CModelHandler {
        CModelHandler {
            vtable: RlmeshModelVtable {
                predict: Some(predict),
                ..full_vtable()
            },
            user_data: UserData(std::ptr::null_mut()),
        }
    }

    fn discrete_observation(values: &[i64]) -> ModelObservation {
        let space = DiscreteBuilder::new(8).build().expect("space");
        let owned: Vec<SpaceValue> = values.iter().map(|&n| SpaceValue::Discrete(n)).collect();
        let wire = rlmesh_grpc::wire::encode_batched_partial_values(&owned, &space).expect("wire");
        ModelObservation {
            observation: Some(wire.leaves),
            route: ModelRouteContext {
                episodes: values.iter().map(|_| EpisodeInfo::default()).collect(),
                ..Default::default()
            },
            num_envs: values.len(),
            env_contract: Some(Arc::new(EnvContract {
                id: "T".into(),
                observation_space: Some(space.clone()),
                action_space: Some(space),
                num_envs: values.len() as u32,
                ..Default::default()
            })),
        }
    }

    #[tokio::test]
    async fn predict_hands_decoded_rows_and_collects_typed_actions() {
        let actions = handler(echo_predict)
            .predict(discrete_observation(&[3, 5]))
            .await
            .expect("predict");
        assert_eq!(
            actions,
            vec![SpaceValue::Discrete(3), SpaceValue::Discrete(5)]
        );
    }

    #[tokio::test]
    async fn predict_ok_with_unset_row_is_an_error() {
        let err = handler(noop_predict)
            .predict(discrete_observation(&[1]))
            .await
            .expect_err("unset action must error");
        assert!(err.to_string().contains("left action 0 unset"), "{err}");
    }

    #[tokio::test]
    async fn predict_decline_surfaces_message_and_recoverability() {
        let err = handler(half_then_fail)
            .predict(discrete_observation(&[1, 2]))
            .await
            .expect_err("decline must error");
        assert!(err.to_string().contains("nope"), "{err}");
        assert!(err.is_recoverable());
    }

    #[test]
    fn read_vtable_honors_truncated_struct_size() {
        // A caller that compiled against only {struct_size, predict} reports a
        // smaller size; the later callbacks must read as None even though they are
        // non-null in this fully-allocated struct (the OOB-read regression guard).
        let mut vtable = full_vtable();
        vtable.struct_size = std::mem::offset_of!(RlmeshModelVtable, on_episode_end);
        let Ok(read) = (unsafe { read_vtable(&vtable) }) else {
            panic!("predict must be covered");
        };
        assert!(read.predict.is_some());
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
        assert!(read.on_episode_end.is_some());
        assert!(read.on_close.is_some());
    }

    #[test]
    fn model_serve_binds_handshakes_and_remote_shuts_down() {
        // Reserve a free port, then hand it to the blocking serve on its own
        // thread; the real ModelClient drives the handshake + remote shutdown that
        // run_local never exercises.
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .expect("reserve port")
            .local_addr()
            .expect("local addr")
            .port();
        let address = format!("tcp://127.0.0.1:{port}");

        let vtable = full_vtable();
        let mut model: *mut RlmeshModel = std::ptr::null_mut();
        assert_eq!(
            unsafe { rlmesh_model_new(&vtable, std::ptr::null_mut(), &mut model) },
            RlmeshStatus::Ok
        );
        let bind = CString::new(address.clone()).expect("bind cstr");
        let options = RlmeshServeOptions {
            token: std::ptr::null(),
            allow_remote_shutdown: true,
            idle_timeout_ms: 0,
            drain_timeout_ms: 0,
            close_timeout_ms: 0,
            predict_concurrency: 0,
        };

        struct ServeArgs {
            model: *mut RlmeshModel,
            bind: *const c_char,
            options: *const RlmeshServeOptions,
        }
        // SAFETY: the test joins the serve thread before dropping model/bind/options.
        unsafe impl Send for ServeArgs {}
        let args = ServeArgs {
            model,
            bind: bind.as_ptr(),
            options: std::ptr::from_ref(&options),
        };
        let server = std::thread::spawn(move || {
            let args = args;
            unsafe { rlmesh_model_serve(args.model, args.bind, args.options) }
        });

        let runtime = tokio::runtime::Runtime::new().expect("client runtime");
        runtime.block_on(async {
            let connect = rlmesh_grpc::ConnectOptions::with_deadline(Duration::from_secs(5))
                .backoff(Duration::from_millis(10));
            let mut client = rlmesh_grpc::ModelClient::connect_with_retry(&address, "", &connect)
                .await
                .expect("client connects to the C-served model");
            client.handshake().await.expect("handshake");
            let shutdown = client.shutdown("test complete").await.expect("shutdown");
            assert!(shutdown.accepted, "server honored remote shutdown");
        });

        assert_eq!(server.join().expect("serve thread"), RlmeshStatus::Ok);
        unsafe { rlmesh_model_free(model) };
    }
}
