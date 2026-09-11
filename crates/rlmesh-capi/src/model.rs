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
use rlmesh::spaces::SpaceValue;
use rlmesh::{
    BindAddress, CancellationToken, ConnectAddress, Error, ModelHandler, ModelObservation,
    ModelWorker, RunLocalOptions, RuntimeReport, ServeModelOptions, ServeOptions,
};

use crate::abi::status::{
    CapiError, RlmeshStatus, clear_last_error, guard, guard_value, last_error_message,
    last_error_recoverable,
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
    /// `num_envs` decoded observation values (borrowed), or NULL when the
    /// request carries no observation or the contract declares no observation
    /// space.
    pub observations: *const *const RlmeshValue,
    /// Rows in this batch; also the length of `episodes` and of `out_actions`.
    pub num_envs: usize,
    /// Spaces/metadata for the route. Never NULL on a predict the runtime
    /// delivers (the route pins a contract with an action space before the
    /// first predict); check it anyway if you like.
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
///
/// The handle is **not** a concurrency primitive: run at most one
/// `rlmesh_model_run_local` / `rlmesh_model_serve` on it at a time. The only
/// call that may overlap a running one is [`rlmesh_model_cancel`], which is
/// explicitly cross-thread.
pub struct RlmeshModel {
    vtable: RlmeshModelVtable,
    user_data: UserData,
    runtime: tokio::runtime::Runtime,
    cancel: CancellationToken,
}

struct CModelHandler {
    vtable: RlmeshModelVtable,
    user_data: UserData,
}

impl CModelHandler {
    /// A handler over `model`'s vtable. The vtable is `Copy` (the capi took its
    /// own copy at `rlmesh_model_new`), so this is free.
    fn new(model: &RlmeshModel) -> Self {
        Self {
            vtable: model.vtable,
            user_data: model.user_data,
        }
    }
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
        let num_envs = observation.num_envs;
        // The C side reads `episodes[i]` for every row it is handed and writes
        // `num_envs` action slots, so a route whose row count disagrees with the
        // batch would hand out pointers past the end.
        if observation.route.episodes.len() != num_envs {
            return Err(Error::model(format!(
                "predict request carries {} episode rows for {num_envs} envs",
                observation.route.episodes.len()
            )));
        }
        // Decode on the async side (pure Rust, cheap) so the C callback only ever
        // sees typed values — the wire framing never crosses the ABI. A contract
        // without an observation space is legal (core allows it), and so is an
        // absent observation: both reach C as `observations == NULL`.
        let decodable = observation.observation.is_some()
            && observation
                .env_contract
                .as_deref()
                .is_some_and(|contract| contract.observation_space.is_some());
        let lanes = if decodable {
            let lanes = observation.decoded_lanes()?;
            if lanes.len() != num_envs {
                return Err(Error::model(format!(
                    "observation decoded {} lanes for {num_envs} envs",
                    lanes.len()
                )));
            }
            Some(lanes)
        } else {
            None
        };
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
            // No action-space check here: both core paths (`local.rs` and
            // `server.rs`) run `check_actions_conform` on the returned actions
            // immediately after, which rejects structural mismatches with a
            // lane-named message and — unlike `contains` — tolerates the Range
            // deviations (a Box bound overshoot) the rest of the stack passes to
            // the env's own validation policy.
            actions
                .into_iter()
                .enumerate()
                .map(|(row, action)| {
                    action.ok_or_else(|| {
                        Error::model(format!("predict returned OK but left action {row} unset"))
                    })
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
/// required. The vtable is **copied** into the handle, so the caller's struct
/// need not outlive the model; `user_data` is retained by pointer and must.
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
            cancel: CancellationToken::new(),
        });
        *out = Box::into_raw(model);
        Ok(())
    })
}

/// Options for `rlmesh_model_run_local`. Pass NULL for defaults (one unbounded,
/// unseeded run until the environment ends). Every field is unset at 0 / NULL.
#[repr(C)]
pub struct RlmeshRunOptions {
    /// Stop after this many episodes (0 = until the env ends).
    pub max_episodes: u64,
    /// Whether `base_seed` is set.
    pub seeded: bool,
    /// Seed for episode 0; later episodes derive from it.
    pub base_seed: i64,
    /// Truncate any episode after this many steps (0 = unset).
    pub max_episode_steps: i64,
    /// Truncate any episode after this much wall-clock time (0 = unset).
    pub max_episode_seconds: f64,
    /// Actions of each predicted chunk to execute before re-planning
    /// (0 and 1 both mean no chunking).
    pub execution_horizon: u32,
    /// Ask the env to close when the run ends.
    pub close_env: bool,
    /// `num_episode_seeds` explicit per-episode reset seeds, consumed in
    /// episode-start order; overrides `base_seed`. NULL = unset.
    pub episode_seeds: *const i64,
    /// Length of `episode_seeds`; 0 = unset.
    pub num_episode_seeds: usize,
}

/// What a finished `rlmesh_model_run_local` reports: plain scalars, no handle.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct RlmeshRunReport {
    /// Episodes that completed during the run.
    pub total_episodes: i64,
    /// Environment steps taken across the whole run.
    pub total_steps: i64,
    /// Summed episode reward.
    pub total_reward: f64,
    /// Mean episode reward (0 when no episode completed).
    pub mean_reward: f64,
    /// Completed episodes that ended terminated.
    pub terminated_episodes: i64,
    /// Completed episodes that ended truncated (a step/time cap).
    pub truncated_episodes: i64,
}

fn run_report(report: &RuntimeReport) -> RlmeshRunReport {
    let total_reward: f64 = report
        .episodes
        .iter()
        .map(|episode| episode.cumulative_reward)
        .sum();
    RlmeshRunReport {
        total_episodes: report.total_episodes,
        total_steps: report.total_steps,
        total_reward,
        mean_reward: if report.episodes.is_empty() {
            0.0
        } else {
            total_reward / report.episodes.len() as f64
        },
        terminated_episodes: report.episodes.iter().filter(|e| e.terminated).count() as i64,
        truncated_episodes: report.episodes.iter().filter(|e| e.truncated).count() as i64,
    }
}

fn run_local_options(address: ConnectAddress, options: *const RlmeshRunOptions) -> RunLocalOptions {
    let mut run = RunLocalOptions::new(address);
    // SAFETY: the export's contract is that `options` is NULL or a valid
    // `RlmeshRunOptions` for the duration of the call.
    let Some(options) = (unsafe { options.as_ref() }) else {
        return run;
    };
    if options.max_episodes != 0 {
        run = run.for_episodes(options.max_episodes);
    }
    if options.seeded {
        run = run.base_seed(options.base_seed);
    }
    if options.max_episode_steps != 0 {
        run = run.max_episode_steps(options.max_episode_steps);
    }
    if options.max_episode_seconds != 0.0 {
        run = run.max_episode_seconds(options.max_episode_seconds);
    }
    if options.execution_horizon != 0 {
        run = run.execution_horizon(options.execution_horizon);
    }
    run = run.close_env(options.close_env);
    if !options.episode_seeds.is_null() && options.num_episode_seeds != 0 {
        // SAFETY: the caller states `episode_seeds` points at `num_episode_seeds`
        // readable `int64_t`s; the slice is copied before this call returns.
        let seeds =
            unsafe { std::slice::from_raw_parts(options.episode_seeds, options.num_episode_seeds) };
        run = run.episode_seeds(seeds.to_vec());
    }
    run
}

/// Drive the model against a remote environment. Blocking — returns when the
/// run ends. `options` may be NULL for defaults; `out_report` may be NULL, and
/// is written only on `RLMESH_OK`.
///
/// # Safety
/// `model` must be a live handle; `env_address` a valid C string; `options` NULL
/// or a valid `RlmeshRunOptions`; `out_report` NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_model_run_local(
    model: *mut RlmeshModel,
    env_address: *const c_char,
    options: *const RlmeshRunOptions,
    out_report: *mut RlmeshRunReport,
) -> RlmeshStatus {
    guard(|| {
        let model =
            unsafe { model.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null model"))?;
        let address = cstr_to_str(env_address)?;
        let address = ConnectAddress::parse(address)
            .map_err(|err| CapiError::invalid_arg(format!("invalid env address: {err}")))?;
        let handler = CModelHandler::new(model);
        let run = run_local_options(address, options);
        let cancel = model.cancel.clone();
        let report = model
            .runtime
            .block_on(async move {
                ModelWorker::new(handler)
                    .run_local_cancellable_async(run, cancel)
                    .await
            })
            .map_err(|err| {
                let mut err = CapiError::from(err);
                // The core flattens a cancelled run to `Error::Internal` (the
                // driver's typed `RouteCancelled` is gone by then), so the token
                // we handed it is the typed signal -- not the message text.
                if model.cancel.is_cancelled() {
                    err.status = RlmeshStatus::Cancelled;
                }
                err
            })?;
        if let Some(out) = unsafe { out_report.as_mut() } {
            *out = run_report(&report);
        }
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
/// server stops: a remote shutdown request, an idle timeout, or
/// `rlmesh_model_cancel`. The same vtable backs every predict, exactly as
/// `rlmesh_model_run_local`. `options` may be NULL for defaults.
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
        let handler = CModelHandler::new(model);
        // The cancel arm drops the server future, which skips the close hook the
        // normal shutdown path runs -- so fire it here instead, once, on the
        // branch that won.
        let mut closer = CModelHandler::new(model);
        let cancel = model.cancel.clone();
        model
            .runtime
            .block_on(async move {
                tokio::select! {
                    result = ModelWorker::new(handler).serve_async(model_options) => result,
                    () = cancel.cancelled() => closer.on_close().await,
                }
            })
            .map_err(CapiError::from)?;
        Ok(())
    })
}

/// Stop a blocking `rlmesh_model_run_local` / `rlmesh_model_serve` running on
/// `model`. Designed to be called from a thread other than the one blocked in
/// the run (a signal handler's worker, a UI thread); it returns immediately and
/// the blocked call unwinds shortly after. A NULL `model` is a no-op.
///
/// Cancellation is terminal for the handle: a cancelled model refuses further
/// runs (each returns at once). Create a new model to run again.
///
/// A cancelled `rlmesh_model_serve` returns `RLMESH_OK` after the close hook; a
/// cancelled `rlmesh_model_run_local` -- this one, or any later one on the
/// handle -- returns `RLMESH_ERR_CANCELLED`, since it has no report to give.
///
/// # Safety
/// `model` must be NULL or a live handle that is not being freed concurrently.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_model_cancel(model: *mut RlmeshModel) {
    guard_value((), || {
        if let Some(model) = unsafe { model.as_ref() } {
            model.cancel.cancel();
        }
    });
}

/// Free a model handle. NULL is a no-op.
///
/// Must NOT be called from inside one of the model's own callbacks: freeing the
/// handle drops its tokio runtime, and dropping a runtime from a thread that
/// runtime owns panics. The panic is contained here (the call returns, with the
/// message on `rlmesh_last_error_message`) rather than unwinding into C, but
/// the handle is then in an undefined state — free a model only from a thread
/// that is not executing it.
///
/// # Safety
/// `model` must be NULL or a handle this thread owns and has not freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_model_free(model: *mut RlmeshModel) {
    guard_value((), || {
        if !model.is_null() {
            drop(unsafe { Box::from_raw(model) });
        }
    });
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
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use rlmesh::spaces::{DType, EnvContract, Tensor, spaces::DiscreteBuilder};
    use rlmesh::{EpisodeInfo, ModelRouteContext};

    use super::*;
    use crate::abi::status::rlmesh_last_error_is_recoverable;
    use crate::value::handle::rlmesh_value_discrete;

    /// A distinctive allocation size nothing else in the test binary requests, so
    /// "the capi freed the row" is observable without sampling ambient churn:
    /// `PROBE_LIVE` returns to 0 only if the allocation was actually released.
    const PROBE: usize = 1_048_573;
    static PROBE_LIVE: AtomicUsize = AtomicUsize::new(0);

    struct ProbeAlloc;

    // SAFETY: every method forwards to `System` unchanged; the counter is bookkeeping.
    unsafe impl GlobalAlloc for ProbeAlloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            if layout.size() == PROBE {
                PROBE_LIVE.fetch_add(1, Ordering::SeqCst);
            }
            unsafe { System.alloc(layout) }
        }
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            if layout.size() == PROBE {
                PROBE_LIVE.fetch_add(1, Ordering::SeqCst);
            }
            unsafe { System.alloc_zeroed(layout) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            if layout.size() == PROBE {
                PROBE_LIVE.fetch_sub(1, Ordering::SeqCst);
            }
            unsafe { System.dealloc(ptr, layout) };
        }
    }

    #[global_allocator]
    static ALLOC: ProbeAlloc = ProbeAlloc;

    /// What the C callbacks record, reached through `user_data`.
    #[derive(Default)]
    struct Counters {
        predicts: AtomicUsize,
        closes: AtomicUsize,
        episode_ends: AtomicUsize,
        null_episode_ids: AtomicUsize,
        null_observations: AtomicUsize,
    }

    impl Counters {
        /// # Safety
        /// `user_data` is the `&Counters` a test handed to `rlmesh_model_new`.
        unsafe fn of<'a>(user_data: *mut c_void) -> &'a Self {
            unsafe { &*user_data.cast::<Self>() }
        }
        fn user_data(&self) -> *mut c_void {
            std::ptr::from_ref(self).cast_mut().cast::<c_void>()
        }
    }

    fn owned(value: SpaceValue) -> *mut RlmeshValue {
        Box::into_raw(Box::new(RlmeshValue(value)))
    }

    fn u8_box(byte: u8) -> SpaceValue {
        SpaceValue::Box(Tensor::from_vec(vec![byte], vec![1], DType::Uint8).expect("tensor"))
    }

    unsafe extern "C" fn noop_predict(
        _: *mut c_void,
        _: *const RlmeshObservation,
        _: *mut *mut RlmeshValue,
    ) -> c_int {
        0
    }
    unsafe extern "C" fn noop_episode_end(_: *mut c_void, _: *const c_char, _: *const c_char) {}
    unsafe extern "C" fn noop_close(_: *mut c_void) {}

    /// Counts its calls and answers with the Uint8 `Box[1]` action `SmokeEnv`
    /// expects; also records whether the observation row reached C as NULL.
    unsafe extern "C" fn counting_predict(
        user_data: *mut c_void,
        obs: *const RlmeshObservation,
        out: *mut *mut RlmeshValue,
    ) -> c_int {
        let counters = unsafe { Counters::of(user_data) };
        let obs = unsafe { &*obs };
        counters.predicts.fetch_add(1, Ordering::SeqCst);
        if obs.observations.is_null() {
            counters.null_observations.fetch_add(1, Ordering::SeqCst);
        }
        for row in 0..obs.num_envs {
            unsafe { *out.add(row) = owned(u8_box(7)) };
        }
        0
    }

    unsafe extern "C" fn counting_episode_end(
        user_data: *mut c_void,
        _: *const c_char,
        episode_id: *const c_char,
    ) {
        let counters = unsafe { Counters::of(user_data) };
        counters.episode_ends.fetch_add(1, Ordering::SeqCst);
        if episode_id.is_null() {
            counters.null_episode_ids.fetch_add(1, Ordering::SeqCst);
        }
    }

    unsafe extern "C" fn counting_close(user_data: *mut c_void) {
        unsafe { Counters::of(user_data) }
            .closes
            .fetch_add(1, Ordering::SeqCst);
    }

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

    /// Writes a probe-sized row 0 then declines: the capi must free what was
    /// written (the probe allocation must not outlive the call). Only the leak
    /// test uses this one -- `PROBE_LIVE` is process-wide, so a second test
    /// declining in parallel would land inside its before/after window.
    unsafe extern "C" fn half_then_fail(
        _: *mut c_void,
        _: *const RlmeshObservation,
        out: *mut *mut RlmeshValue,
    ) -> c_int {
        unsafe { decline_after_row(out, PROBE) }
    }

    /// The same decline with an ordinary-sized row, for the tests that assert the
    /// message and status rather than the free.
    unsafe extern "C" fn decline_with_a_row(
        _: *mut c_void,
        _: *const RlmeshObservation,
        out: *mut *mut RlmeshValue,
    ) -> c_int {
        unsafe { decline_after_row(out, 8) }
    }

    unsafe fn decline_after_row(out: *mut *mut RlmeshValue, size: usize) -> c_int {
        let tensor =
            Tensor::from_vec(vec![0u8; size], vec![size as i64], DType::Uint8).expect("row tensor");
        unsafe { *out = owned(SpaceValue::Box(tensor)) };
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

    /// The vtable a live-runtime test uses: every hook wired to `Counters`.
    fn counting_vtable() -> RlmeshModelVtable {
        RlmeshModelVtable {
            struct_size: std::mem::size_of::<RlmeshModelVtable>(),
            predict: Some(counting_predict),
            on_episode_end: Some(counting_episode_end),
            on_close: Some(counting_close),
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

    fn counting_handler(counters: &Counters) -> CModelHandler {
        CModelHandler {
            vtable: counting_vtable(),
            user_data: UserData(counters.user_data()),
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
        let err = handler(decline_with_a_row)
            .predict(discrete_observation(&[1, 2]))
            .await
            .expect_err("decline must error");
        assert!(err.to_string().contains("nope"), "{err}");
        assert!(err.is_recoverable());
    }

    #[tokio::test]
    async fn predict_decline_frees_the_half_written_row() {
        let before = PROBE_LIVE.load(Ordering::SeqCst);
        handler(half_then_fail)
            .predict(discrete_observation(&[1, 2]))
            .await
            .expect_err("decline must error");
        assert_eq!(
            PROBE_LIVE.load(Ordering::SeqCst),
            before,
            "the row written before the decline must be freed, not leaked"
        );
    }

    #[tokio::test]
    async fn predict_rejects_a_row_count_that_disagrees_with_the_batch() {
        // num_envs says two rows, the route carries one: handing C two
        // `episodes` entries would read past the end.
        let mut observation = discrete_observation(&[1]);
        observation.num_envs = 2;
        let err = handler(echo_predict)
            .predict(observation)
            .await
            .expect_err("mismatched row count must error");
        assert!(err.to_string().contains("episode rows"), "{err}");
    }

    #[tokio::test]
    async fn absent_observation_reaches_c_as_null() {
        let counters = Counters::default();

        // No observation at all.
        let mut observation = discrete_observation(&[1]);
        observation.observation = None;
        counting_handler(&counters)
            .predict(observation)
            .await
            .expect("predict without an observation");

        // An observation the contract declares no space for: core allows such a
        // route, so it must reach C as absent rather than failing the decode.
        let mut observation = discrete_observation(&[1]);
        observation.env_contract = Some(Arc::new(EnvContract {
            id: "T".into(),
            observation_space: None,
            action_space: None,
            num_envs: 1,
            ..Default::default()
        }));
        counting_handler(&counters)
            .predict(observation)
            .await
            .expect("predict with a space-less contract");

        assert_eq!(counters.predicts.load(Ordering::SeqCst), 2);
        assert_eq!(counters.null_observations.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn reset_adapter_maps_ids_onto_callbacks() {
        let counters = Counters::default();
        // No ids means "every episode of this env": exactly one call, NULL id.
        counting_handler(&counters)
            .reset_adapter("env-1", Vec::new())
            .await
            .expect("reset_adapter");
        assert_eq!(counters.episode_ends.load(Ordering::SeqCst), 1);
        assert_eq!(counters.null_episode_ids.load(Ordering::SeqCst), 1);

        // With ids: one call per id, none of them NULL.
        counting_handler(&counters)
            .reset_adapter("env-1", vec!["a".to_string(), "b".to_string()])
            .await
            .expect("reset_adapter");
        assert_eq!(counters.episode_ends.load(Ordering::SeqCst), 3);
        assert_eq!(counters.null_episode_ids.load(Ordering::SeqCst), 1);
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

    /// A minimal single environment: a Uint8 `Box[1]` obs/action, one step per
    /// episode (reset → 0, step → 1 then terminated, reward 1.0). Records the
    /// reset seeds it was given so a seeded run is checkable.
    struct SmokeEnv {
        obs_space: rlmesh::SpaceSpec,
        action_space: rlmesh::SpaceSpec,
        env_contract: EnvContract,
        reset_seeds: Arc<std::sync::Mutex<Vec<Option<i64>>>>,
    }

    impl SmokeEnv {
        fn new(reset_seeds: Arc<std::sync::Mutex<Vec<Option<i64>>>>) -> Self {
            let space = |high: f64| {
                rlmesh::spaces::spaces::BoxSpaceBuilder::scalar(0.0, high, vec![1])
                    .dtype(DType::Uint8)
                    .build()
                    .expect("space")
            };
            let (obs_space, action_space) = (space(255.0), space(1.0));
            Self {
                env_contract: EnvContract {
                    id: "SmokeEnv-capi".to_string(),
                    observation_space: Some(obs_space.clone()),
                    action_space: Some(action_space.clone()),
                    num_envs: 1,
                    ..Default::default()
                },
                obs_space,
                action_space,
                reset_seeds,
            }
        }
    }

    #[async_trait]
    impl rlmesh::Env for SmokeEnv {
        fn observation_space(&self) -> &rlmesh::SpaceSpec {
            &self.obs_space
        }
        fn action_space(&self) -> &rlmesh::SpaceSpec {
            &self.action_space
        }
        fn env_contract(&self) -> &EnvContract {
            &self.env_contract
        }

        async fn reset(
            &mut self,
            req: rlmesh::ResetRequest,
        ) -> Result<rlmesh::ResetResult, rlmesh::EnvRuntimeError> {
            self.reset_seeds.lock().expect("seed log").push(req.seed);
            Ok(rlmesh::ResetResult {
                observation: Some(u8_box(0)),
                info: None,
                episode_id: None,
            })
        }

        async fn step(
            &mut self,
            _req: rlmesh::StepRequest,
        ) -> Result<rlmesh::StepResult, rlmesh::EnvRuntimeError> {
            Ok(rlmesh::StepResult {
                observation: Some(u8_box(1)),
                reward: 1.0,
                terminated: true,
                truncated: false,
                info: None,
            })
        }

        async fn render(
            &mut self,
            _req: rlmesh::RenderRequest,
        ) -> Result<rlmesh::RenderResult, rlmesh::EnvRuntimeError> {
            Ok(rlmesh::RenderResult::default())
        }

        async fn close(
            &mut self,
            _req: rlmesh::CloseRequest,
        ) -> Result<rlmesh::CloseResult, rlmesh::EnvRuntimeError> {
            Ok(rlmesh::CloseResult)
        }
    }

    /// A `SmokeEnv` served on a loopback port by its own thread + runtime, so a
    /// blocking capi export can be driven against it from the test thread.
    struct EnvHarness {
        address: String,
        seeds: Arc<std::sync::Mutex<Vec<Option<i64>>>>,
        stop: CancellationToken,
        thread: Option<std::thread::JoinHandle<()>>,
    }

    impl EnvHarness {
        fn start() -> Self {
            let seeds = Arc::new(std::sync::Mutex::new(Vec::new()));
            let stop = CancellationToken::new();
            let (tx, rx) = std::sync::mpsc::channel();
            let (env_seeds, serve_stop) = (Arc::clone(&seeds), stop.clone());
            let thread = std::thread::spawn(move || {
                let runtime = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(1)
                    .enable_all()
                    .build()
                    .expect("env runtime");
                runtime.block_on(async move {
                    let bound = rlmesh::EnvServer::new(SmokeEnv::new(env_seeds))
                        .bind(BindAddress::Tcp {
                            host: "127.0.0.1".to_string(),
                            port: 0,
                        })
                        .await
                        .expect("bind env");
                    tx.send(bound.local_addr().to_string()).expect("send addr");
                    tokio::select! {
                        _ = bound.serve() => {}
                        () = serve_stop.cancelled() => {}
                    }
                });
            });
            Self {
                address: rx.recv().expect("env address"),
                seeds,
                stop,
                thread: Some(thread),
            }
        }
    }

    impl Drop for EnvHarness {
        fn drop(&mut self) {
            self.stop.cancel();
            if let Some(thread) = self.thread.take() {
                let _ = thread.join();
            }
        }
    }

    fn new_model(vtable: &RlmeshModelVtable, user_data: *mut c_void) -> *mut RlmeshModel {
        let mut model: *mut RlmeshModel = std::ptr::null_mut();
        assert_eq!(
            unsafe { rlmesh_model_new(vtable, user_data, &mut model) },
            RlmeshStatus::Ok
        );
        model
    }

    #[test]
    fn run_local_drives_a_seeded_run_and_reports_it() {
        let env = EnvHarness::start();
        let counters = Counters::default();
        let model = new_model(&counting_vtable(), counters.user_data());
        let address = CString::new(env.address.clone()).expect("address");
        let seeds = [11_i64, 22, 33];
        let options = RlmeshRunOptions {
            max_episodes: 3,
            seeded: false,
            base_seed: 0,
            max_episode_steps: 0,
            max_episode_seconds: 0.0,
            execution_horizon: 0,
            close_env: false,
            episode_seeds: seeds.as_ptr(),
            num_episode_seeds: seeds.len(),
        };
        let mut report = RlmeshRunReport::default();
        let status =
            unsafe { rlmesh_model_run_local(model, address.as_ptr(), &options, &mut report) };
        assert_eq!(status, RlmeshStatus::Ok, "{}", last_error_message());

        assert_eq!(report.total_episodes, 3);
        assert_eq!(report.total_steps, 3);
        assert_eq!(report.total_reward, 3.0);
        assert_eq!(report.mean_reward, 1.0);
        assert_eq!(report.terminated_episodes, 3);
        assert_eq!(report.truncated_episodes, 0);
        assert_eq!(counters.predicts.load(Ordering::SeqCst), 3);
        // The close hook fires exactly once at the end of the run.
        assert_eq!(counters.closes.load(Ordering::SeqCst), 1);
        // The explicit per-episode seeds reached the env's resets in order.
        assert_eq!(
            *env.seeds.lock().expect("seed log"),
            vec![Some(11), Some(22), Some(33)]
        );

        // Cancellation is terminal, and says so with its own status: the next run
        // on this handle stops at once rather than reporting a generic internal
        // error a caller would have to string-match.
        unsafe { rlmesh_model_cancel(model) };
        let cancelled = unsafe {
            rlmesh_model_run_local(
                model,
                address.as_ptr(),
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(
            cancelled,
            RlmeshStatus::Cancelled,
            "{}",
            last_error_message()
        );
        unsafe { rlmesh_model_free(model) };
    }

    #[test]
    fn run_local_surfaces_a_model_decline_as_a_recoverable_model_error() {
        // The driver used to flatten every failure to Internal/non-recoverable;
        // a C decline must arrive as RLMESH_ERR_MODEL with its own message.
        let env = EnvHarness::start();
        let vtable = RlmeshModelVtable {
            predict: Some(decline_with_a_row),
            ..full_vtable()
        };
        let model = new_model(&vtable, std::ptr::null_mut());
        let address = CString::new(env.address.clone()).expect("address");
        let status = unsafe {
            rlmesh_model_run_local(
                model,
                address.as_ptr(),
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        unsafe { rlmesh_model_free(model) };

        assert_eq!(status, RlmeshStatus::Model);
        assert!(
            last_error_message().contains("nope"),
            "{}",
            last_error_message()
        );
        assert_eq!(rlmesh_last_error_is_recoverable(), 1);
    }

    /// Hand a raw model/args tuple to a serve thread. The test joins it before
    /// dropping anything it points at.
    struct ServeArgs {
        model: *mut RlmeshModel,
        bind: *const c_char,
        options: *const RlmeshServeOptions,
    }
    // SAFETY: every test joins the serve thread before dropping model/bind/options.
    unsafe impl Send for ServeArgs {}

    fn reserve_port() -> u16 {
        std::net::TcpListener::bind("127.0.0.1:0")
            .expect("reserve port")
            .local_addr()
            .expect("local addr")
            .port()
    }

    fn serve_options(allow_remote_shutdown: bool) -> RlmeshServeOptions {
        RlmeshServeOptions {
            token: std::ptr::null(),
            allow_remote_shutdown,
            idle_timeout_ms: 0,
            drain_timeout_ms: 0,
            close_timeout_ms: 0,
            predict_concurrency: 0,
        }
    }

    fn connect_options() -> rlmesh_grpc::ConnectOptions {
        rlmesh_grpc::ConnectOptions::with_deadline(Duration::from_secs(5))
            .backoff(Duration::from_millis(10))
    }

    #[test]
    fn served_model_predicts_through_the_c_callback() {
        // The served path, driven the way core's own tests drive a served model:
        // connect a RemoteModel, reset, predict, and check the action the C
        // callback produced actually came back over the wire.
        let port = reserve_port();
        let address = format!("tcp://127.0.0.1:{port}");
        let counters = Counters::default();
        let model = new_model(&counting_vtable(), counters.user_data());
        let bind = CString::new(address.clone()).expect("bind cstr");
        let options = serve_options(true);
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
            let mut client =
                rlmesh_grpc::ModelClient::connect_with_retry(&address, "", &connect_options())
                    .await
                    .expect("client connects to the C-served model");
            client.handshake().await.expect("handshake");

            let env_contract = SmokeEnv::new(Arc::new(std::sync::Mutex::new(Vec::new())))
                .env_contract
                .clone();
            let mut remote = rlmesh::RemoteModel::connect(&address, env_contract)
                .await
                .expect("remote model");
            remote.reset(None);
            let action = remote.predict(u8_box(3)).await.expect("predict");
            // The C callback answered every row with the byte 7.
            let SpaceValue::Box(tensor) = &action else {
                panic!("expected a Box action, got {action:?}");
            };
            assert_eq!(tensor.storage().as_slice(), [7]);
            remote.close().await.expect("close route");
            drop(remote);

            let shutdown = client.shutdown("test complete").await.expect("shutdown");
            assert!(shutdown.accepted, "server honored remote shutdown");
        });

        assert_eq!(server.join().expect("serve thread"), RlmeshStatus::Ok);
        assert_eq!(counters.predicts.load(Ordering::SeqCst), 1);
        assert_eq!(counters.closes.load(Ordering::SeqCst), 1);
        unsafe { rlmesh_model_free(model) };
    }

    #[test]
    fn cancel_stops_a_blocking_serve() {
        let port = reserve_port();
        let address = format!("tcp://127.0.0.1:{port}");
        let counters = Counters::default();
        let model = new_model(&counting_vtable(), counters.user_data());
        let bind = CString::new(address.clone()).expect("bind cstr");
        let options = serve_options(false);
        let args = ServeArgs {
            model,
            bind: bind.as_ptr(),
            options: std::ptr::from_ref(&options),
        };
        let server = std::thread::spawn(move || {
            let args = args;
            unsafe { rlmesh_model_serve(args.model, args.bind, args.options) }
        });

        // Wait until the server is really accepting, so the cancel races nothing.
        let runtime = tokio::runtime::Runtime::new().expect("client runtime");
        runtime.block_on(async {
            rlmesh_grpc::ModelClient::connect_with_retry(&address, "", &connect_options())
                .await
                .expect("server is up");
        });

        // Remote shutdown is off and there is no idle timeout: only cancel ends it.
        unsafe { rlmesh_model_cancel(model) };
        assert_eq!(server.join().expect("serve thread"), RlmeshStatus::Ok);
        assert_eq!(counters.closes.load(Ordering::SeqCst), 1);
        unsafe { rlmesh_model_free(model) };
    }
}
