//! The model path: a C callback vtable adapted into a core `ModelHandler`, plus
//! a handle that owns a tokio runtime and drives/serves it.
//!
//! Every C callback runs inside `spawn_blocking` so a blocking/CPU-bound callback
//! cannot starve the shared multi-thread runtime. A callback's error is read on
//! its own thread and folded into an `Error` value that travels back to the caller.
#![allow(unsafe_code)] // FFI: raw callback pointers + repr(C) structs.

use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use rlmesh::binding::model::{
    EpisodeInfo, ModelHandler, ModelObservation, ModelWorker, RunLocalOptions, ServeModelOptions,
};
use rlmesh::binding::spaces::SpaceValue;
use rlmesh::binding::{CancellationToken, RuntimeReport};
use rlmesh::ext::{BindAddress, ConnectAddress};
use rlmesh::{Error, ServeOptions, predict_seed};

use crate::abi::status::{
    CapiError, RLMeshStatus, clear_last_error, guard, guard_value, last_error_message,
    last_error_recoverable,
};
use crate::spaces::RLMeshEnvContract;
use crate::value::handle::RLMeshValue;

/// Predict callback: read the decoded per-env observations in `obs`, run the
/// policy, and write one owned action value per env into `out_actions`
/// (`num_envs` slots, pre-zeroed). The capi takes ownership of every value
/// written. Return `RLMESH_OK` (0), or nonzero to decline (set a message via
/// `rlmesh_callback_set_error`); on a nonzero return any values already written
/// are freed by the capi.
///
/// The return is read as a plain `c_int` (not the `RLMeshStatus` enum) so an
/// out-of-range value from a C author is not undefined behavior.
pub type RLMeshExtLegacyPredictFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    obs: *const RLMeshExtLegacyObservation,
    out_actions: *mut *mut RLMeshValue,
) -> c_int;
/// A no-argument lifecycle callback (`on_close`).
pub type RLMeshExtLegacyLifecycleFn = unsafe extern "C" fn(user_data: *mut c_void);
/// Episode teardown callback: `episode_id` NULL means every episode of `env_id`.
pub type RLMeshExtLegacyEpisodeEndFn =
    unsafe extern "C" fn(user_data: *mut c_void, env_id: *const c_char, episode_id: *const c_char);

/// One row's episode identity within a predict request.
#[repr(C)]
pub struct RLMeshExtLegacyEpisode {
    /// Runtime-minted episode id (NUL-terminated, never repeats).
    pub id: *const c_char,
    /// Whether `seed` carries an explicit reset seed.
    pub seeded: bool,
    /// The explicit reset seed; only meaningful when `seeded`.
    pub seed: i64,
    /// This episode's re-plan ordinal: 0 for the first predict under `id`,
    /// then counting up by one per predict until the episode ends.
    pub predict_index: u64,
    /// `rlmesh::predict_seed(seed, predict_index)` — a reproducible per-predict
    /// sampling seed. Only meaningful when `seeded` (0 otherwise), mirroring the
    /// SDK's `None` for an unseeded episode.
    pub predict_seed: i64,
}

/// Live episodes a model tracks before a NEW episode is refused -- the Python
/// SDK's `EPISODE_STATE_CAPACITY`. A ceiling, not an eviction threshold: an
/// entry only leaves at its end edge. Far above the lanes any one model server
/// legitimately admits at once, so reaching it means episode ends are not
/// arriving.
const EPISODE_STORE_CAPACITY: usize = 65_536;

/// One held episode's bookkeeping: the env it belongs to (so an env-wide
/// `on_episode_end` can drop it), the seed it was first seen with, and how many
/// predicts it has had.
struct HeldEpisode {
    env_id: String,
    seed: Option<i64>,
    index: u64,
}

/// The capi's bounded per-episode store, keyed by episode id — the same keying
/// the Python SDK does once in `EpisodeStore` so a C model reads
/// `predict_index` / `predict_seed` off the row instead of counting itself.
///
/// Entries are dropped at the episode-end edge (`reset_adapter`, on both the
/// local and the served path). Live state is never evicted: a run that never
/// signals an end still cannot grow without bound, because past `capacity` a
/// NEW episode's predict fails instead of silently dropping another episode's
/// state, as in Python.
struct EpisodeStore {
    episodes: HashMap<String, HeldEpisode>,
    capacity: usize,
}

impl EpisodeStore {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            episodes: HashMap::new(),
            capacity: capacity.max(1),
        }
    }

    /// Hand out each row's next `(predict_index, predict_seed)` and advance it.
    ///
    /// A row with no identity (an anonymous spec-less lane) gets a throwaway
    /// `(0, None)` that is never stored: there is nothing to key it by. A batch
    /// whose new episodes would take the store past capacity is refused whole,
    /// before any row advances.
    fn next(
        &mut self,
        env_id: &str,
        episodes: &[EpisodeInfo],
    ) -> Result<Vec<(u64, Option<i64>)>, Error> {
        let mut new_ids: Vec<&str> = Vec::new();
        for episode in episodes {
            let id = episode.episode_id.as_str();
            if !id.is_empty() && !self.episodes.contains_key(id) && !new_ids.contains(&id) {
                if self.episodes.len() + new_ids.len() >= self.capacity {
                    // Python raises the builtin `RuntimeError` for this local
                    // refusal (`_models/_episodes.py`): `Error::Runtime` ->
                    // `RLMESH_ERR_RUNTIME`, not a model decline.
                    return Err(Error::Runtime(format!(
                        "model already tracks {} live episodes; refusing episode {id} rather \
                         than evicting another episode's live state. Episode ends are not \
                         reaching this model, or the deployment runs more concurrent \
                         episodes than it allows",
                        self.capacity
                    )));
                }
                new_ids.push(id);
            }
        }
        Ok(episodes
            .iter()
            .map(|episode| {
                if episode.episode_id.is_empty() {
                    return (0, None);
                }
                let held = self
                    .episodes
                    .entry(episode.episode_id.clone())
                    .or_insert_with(|| HeldEpisode {
                        env_id: env_id.to_string(),
                        seed: episode.seed,
                        index: 0,
                    });
                let index = held.index;
                held.index += 1;
                (index, held.seed.map(|seed| predict_seed(seed, index)))
            })
            .collect())
    }

    /// Drop the given episodes of `env_id`; an empty list means every episode
    /// of that env (the `episode_id == NULL` form of `on_episode_end`).
    fn end(&mut self, env_id: &str, episode_ids: &[String]) {
        if episode_ids.is_empty() {
            self.episodes.retain(|_, held| held.env_id != env_id);
            return;
        }
        for id in episode_ids {
            self.episodes.remove(id);
        }
    }
}

/// What a predict callback receives. Pointers are valid only for the duration of
/// the call.
#[repr(C)]
pub struct RLMeshExtLegacyObservation {
    /// `num_envs` decoded observation values (borrowed), or NULL when the
    /// request carries no observation or the contract declares no observation
    /// space.
    pub observations: *const *const RLMeshValue,
    /// Rows in this batch; also the length of `episodes` and of `out_actions`.
    pub num_envs: usize,
    /// Spaces/metadata for the route. Never NULL on a predict the runtime
    /// delivers (the route pins a contract with an action space before the
    /// first predict); check it anyway if you like.
    pub contract: *const RLMeshEnvContract,
    /// NUL-terminated.
    pub session_id: *const c_char,
    /// NUL-terminated.
    pub env_id: *const c_char,
    /// NUL-terminated.
    pub request_id: *const c_char,
    /// `num_envs` entries; row `i` of `observations` belongs to `episodes[i]`.
    pub episodes: *const RLMeshExtLegacyEpisode,
}

/// The model callback vtable. Set `struct_size = sizeof(RLMeshExtLegacyModelVtable)`;
/// unknown trailing bytes must be zero. `predict` is required.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshExtLegacyModelVtable {
    /// Size of the struct the caller compiled against.
    pub struct_size: usize,
    /// Required: map a batch of observations to one action per row.
    pub predict: Option<RLMeshExtLegacyPredictFn>,
    /// Optional: drop per-episode state.
    pub on_episode_end: Option<RLMeshExtLegacyEpisodeEndFn>,
    /// Optional: the worker/session is shutting down.
    pub on_close: Option<RLMeshExtLegacyLifecycleFn>,
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
    // A callback failure surfaces as `Error::Model`, so record the status the
    // outermost export would map it to (the callback has no status channel).
    crate::abi::status::store_last_error(&message, recoverable, RLMeshStatus::Model);
}

/// Set the current callback's error as an exact Python class (R-12), as a
/// Python author raises `ValueError` or `TypeError` from a hook. An
/// Environment or Model class carries code INTERNAL and is recoverable only as
/// `RLMESH_ERROR_CLASS_RECOVERABLE_ENVIRONMENT_EXCEPTION`; INVALID or an
/// unknown value records `RLMeshException`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_callback_set_error_class(error_class: i32, message: *const c_char) {
    let message = if message.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(message) }
            .to_string_lossy()
            .into_owned()
    };
    crate::abi::status::store_last_error_class(&message, error_class);
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
/// `rlmesh_ext_legacy_model_run_local` / `rlmesh_ext_legacy_model_serve` on it at a time. The only
/// call that may overlap a running one is [`rlmesh_ext_legacy_model_cancel`], which is
/// explicitly cross-thread.
pub struct RLMeshExtLegacyModel {
    vtable: RLMeshExtLegacyModelVtable,
    user_data: UserData,
    runtime: tokio::runtime::Runtime,
    cancel: CancellationToken,
    /// The handle's one `on_close` gate, shared by every handler built over it:
    /// a run borrows the model (never closes it), a serve closes it when it
    /// stops, and `rlmesh_ext_legacy_model_free` closes it if nothing did.
    close: Arc<CloseGate>,
}

impl Drop for RLMeshExtLegacyModel {
    /// Free is the C caller's close: fire `on_close` here unless a serve
    /// already did. Runs on the freeing thread; no run is active by contract.
    fn drop(&mut self) {
        if self.close.fired.swap(true, Ordering::SeqCst) {
            return;
        }
        if let Some(callback) = self.vtable.on_close {
            unsafe { callback(self.user_data.0) };
        }
    }
}

struct CModelHandler {
    vtable: RLMeshExtLegacyModelVtable,
    user_data: UserData,
    /// Per-episode predict ordinals, keyed by episode id (see [`EpisodeStore`]).
    episodes: EpisodeStore,
    /// Ordering + once-only seam for the C `on_close` (see [`CloseGate`]).
    close: Arc<CloseGate>,
}

/// Keeps the C `on_close` to one call per handle, ordered after any predict
/// still inside the C callback. `rlmesh_ext_legacy_model_serve`'s cancel arm shares the
/// gate with the handler the server owns: cancelling drops the server future,
/// so its own close path never runs, while the connection tasks tonic spawned
/// can still be in a callback.
#[derive(Default)]
struct CloseGate {
    /// Held for the length of a predict and of the close hook.
    busy: tokio::sync::Mutex<()>,
    fired: AtomicBool,
}

impl CModelHandler {
    /// A handler over `model`'s vtable. The vtable is `Copy` (the capi took its
    /// own copy at `rlmesh_ext_legacy_model_new`), so this is free. Every handler over one
    /// model shares its close gate, so only one of them runs the C `on_close`.
    fn new(model: &RLMeshExtLegacyModel) -> Self {
        Self {
            vtable: model.vtable,
            user_data: model.user_data,
            episodes: EpisodeStore::with_capacity(EPISODE_STORE_CAPACITY),
            close: Arc::clone(&model.close),
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
    // A class set with `rlmesh_callback_set_error_class` propagates as that
    // class; `rlmesh_callback_set_error` (and none) is the model's own failure.
    if let Some(class) = crate::abi::status::take_callback_class() {
        return crate::abi::status::error_of_class(class, &message);
    }
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
        // Held for the whole call: the close hook takes the same gate, so a C
        // `on_close` never runs beside a predict still inside the C callback.
        let gate = Arc::clone(&self.close);
        let _busy = gate.busy.lock().await;
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
        // Stamp each row's re-plan ordinal + derived seed here, on the handler's
        // own thread: the store is plain `&mut self` state, no lock needed. A
        // batch with a new episode past capacity fails here, before the C
        // predict runs.
        let contexts = self.episodes.next(&route.env_id, &route.episodes)?;

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
            let episodes: Vec<RLMeshExtLegacyEpisode> = route
                .episodes
                .iter()
                .zip(&episode_ids)
                .zip(&contexts)
                .map(
                    |((episode, id), &(predict_index, predict_seed))| RLMeshExtLegacyEpisode {
                        id: id.as_ptr(),
                        seeded: episode.seed.is_some(),
                        seed: episode.seed.unwrap_or_default(),
                        predict_index,
                        predict_seed: predict_seed.unwrap_or_default(),
                    },
                )
                .collect();
            let lanes: Option<Vec<RLMeshValue>> = lanes.map(|lanes| {
                let spec = contract
                    .as_ref()
                    .and_then(|c| c.observation_space.as_ref())
                    .expect("decoded lanes have a spec");
                lanes
                    .into_iter()
                    .map(|v| RLMeshValue::new(rlmesh::binding::value_from_space(spec, v)))
                    .collect()
            });
            let lane_ptrs: Option<Vec<*const RLMeshValue>> = lanes
                .as_ref()
                .map(|lanes| lanes.iter().map(std::ptr::from_ref).collect());
            let contract_handle = contract.as_deref().cloned().map(RLMeshEnvContract::new);
            let contract_ptr = contract_handle
                .as_ref()
                .map_or(std::ptr::null(), std::ptr::from_ref);
            let view = RLMeshExtLegacyObservation {
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

            let mut out: Vec<*mut RLMeshValue> = vec![std::ptr::null_mut(); num_envs];
            // Clear first so a decline that doesn't set an error can't report a
            // stale message left on this reused pool thread.
            clear_last_error();
            let status = unsafe { predict(user_data.0, &view, out.as_mut_ptr()) };
            // Reclaim every written handle whatever the status: on success they are
            // the actions, on failure they must not leak.
            let actions: Vec<Option<rlmesh::Value>> = out
                .into_iter()
                .map(|ptr| (!ptr.is_null()).then(|| unsafe { Box::from_raw(ptr) }.value().clone()))
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
                    let action = action.ok_or_else(|| {
                        Error::model(format!("predict returned OK but left action {row} unset"))
                    })?;
                    match contract.as_ref().and_then(|c| c.action_space.as_ref()) {
                        Some(spec) => rlmesh::binding::value_to_space(spec, &action),
                        // The legacy handler permits routes with no contract. A
                        // tensor already carries the wire layout in that case.
                        None => Ok(SpaceValue::Box(
                            rlmesh::binding::tensor_core(action.as_tensor()?).clone(),
                        )),
                    }
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
        // The store's drop edge is unconditional: a model without an
        // `on_episode_end` still gets its ordinals restarted per episode.
        self.episodes.end(env_id, &episode_ids);
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
        let gate = Arc::clone(&self.close);
        let _busy = gate.busy.lock().await;
        if gate.fired.swap(true, Ordering::SeqCst) {
            return Ok(());
        }
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

/// Read an ABI-5 input vtable with its full minimum and checked zero tail.
///
/// # Safety
/// `ptr` is non-NULL and points at a vtable whose first `struct_size` bytes are valid.
unsafe fn read_vtable(
    ptr: *const RLMeshExtLegacyModelVtable,
) -> Result<RLMeshExtLegacyModelVtable, CapiError> {
    let v = unsafe { crate::abi::sizing::read(ptr, "RLMeshExtLegacyModelVtable") }?
        .ok_or_else(|| CapiError::invalid_arg("null vtable"))?;
    if v.predict.is_none() {
        return Err(CapiError::invalid_arg("vtable predict is null"));
    }
    Ok(v)
}

/// Create a model from a callback vtable. `predict` and `struct_size` are
/// required. The vtable is **copied** into the handle, so the caller's struct
/// need not outlive the model; `user_data` is retained by pointer and must.
///
/// # Safety
/// `vtable` must be valid; `user_data` is passed unchanged to every callback.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_ext_legacy_model_new(
    vtable: *const RLMeshExtLegacyModelVtable,
    user_data: *mut c_void,
    out: *mut *mut RLMeshExtLegacyModel,
) -> RLMeshStatus {
    guard(|| {
        if vtable.is_null() {
            return Err(CapiError::invalid_arg("null vtable"));
        }
        let vtable = unsafe { read_vtable(vtable) }?;
        crate::abi::ignore_sigpipe();
        let out = unsafe { out.as_mut() }.ok_or_else(|| CapiError::invalid_arg("null out"))?;
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|err| CapiError::internal(format!("failed to build runtime: {err}")))?;
        let model = Box::new(RLMeshExtLegacyModel {
            vtable,
            user_data: UserData(user_data),
            runtime,
            cancel: CancellationToken::new(),
            close: Arc::default(),
        });
        *out = Box::into_raw(model);
        Ok(())
    })
}

/// Options for `rlmesh_ext_legacy_model_run_local`. Pass NULL for defaults (one unseeded
/// episode, Python's `run()` default). Every field is unset at 0 / NULL.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshExtLegacyRunOptions {
    pub struct_size: usize,
    /// The exact number of episodes to run, resolved like Python's
    /// `run(episodes=...)`. 0 = unset: one episode, or one per seed when
    /// `episode_seeds` is given. A non-zero count with `episode_seeds` must
    /// match the seed count (`RLMESH_ERR_INVALID_ARGUMENT` otherwise). There is
    /// no unbounded run.
    // TODO(docs/specs/language-parity/running.md): Python's `episodes=0` (run
    // nothing) is not expressible here, since 0 means unset; it needs the
    // planned `has_episodes` field rather than a sentinel.
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
    /// Whether `trial_index_base` is set.
    pub trial_indexed: bool,
    /// First trial ordinal this run's episodes walk (`base`, `base + 1`, ...);
    /// delivered as `reset(options={"trial_index": k})` to an env that declares
    /// that reset option. Only meaningful when `trial_indexed`.
    pub trial_index_base: u64,
}

/// What a finished `rlmesh_ext_legacy_model_run_local` reports: plain scalars, no handle.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct RLMeshExtLegacyRunReport {
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

fn run_report(report: &RuntimeReport) -> RLMeshExtLegacyRunReport {
    let total_reward: f64 = report
        .episodes
        .iter()
        .map(|episode| episode.cumulative_reward)
        .sum();
    RLMeshExtLegacyRunReport {
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

fn run_local_options(
    address: ConnectAddress,
    options: *const RLMeshExtLegacyRunOptions,
) -> RunLocalOptions {
    let mut run = RunLocalOptions::new(address);
    // SAFETY: the export's contract is that `options` is NULL or a valid
    // `RLMeshExtLegacyRunOptions` for the duration of the call.
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
    if options.trial_indexed {
        run = run.trial_index_base(options.trial_index_base);
    }
    run
}

/// Drive the model against a remote environment. Blocking — returns when the
/// run ends. `options` may be NULL for defaults; `out_report` may be NULL, and
/// is written only on `RLMESH_OK`.
///
/// # Safety
/// `model` must be a live handle; `env_address` a valid C string; `options` NULL
/// or a valid `RLMeshExtLegacyRunOptions`; `out_report` NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_ext_legacy_model_run_local(
    model: *mut RLMeshExtLegacyModel,
    env_address: *const c_char,
    options: *const RLMeshExtLegacyRunOptions,
    out_report: *mut RLMeshExtLegacyRunReport,
) -> RLMeshStatus {
    guard(|| {
        let model =
            unsafe { model.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null model"))?;
        let address = cstr_to_str(env_address)?;
        let address = ConnectAddress::parse(address)
            .map_err(|err| CapiError::invalid_arg(format!("invalid env address: {err}")))?;
        let handler = CModelHandler::new(model);
        unsafe { crate::abi::sizing::read(options, "RLMeshExtLegacyRunOptions") }?;
        let run = run_local_options(address, options);
        // Python raises ValueError for a seed count that disagrees with the
        // episode count; refuse it here as an argument error, before dialing.
        run.episode_budget()
            .map_err(|err| CapiError::invalid_arg(err.to_string()))?;
        let cancel = model.cancel.clone();
        let report = model
            .runtime
            .block_on(async move {
                ModelWorker::new(handler)
                    .run_local_cancellable_async(run, cancel)
                    .await
            })
            .map_err(|err| {
                let err = CapiError::from(err);
                // The core flattens a cancelled run to `Error::Internal` (the
                // driver's typed `RouteCancelled` is gone by then), so the token
                // we handed it is the typed signal -- not the message text.
                if model.cancel.is_cancelled() {
                    err.into_cancelled()
                } else {
                    err
                }
            })?;
        if let Some(out) = unsafe { out_report.as_mut() } {
            *out = run_report(&report);
        }
        Ok(())
    })
}

/// Serve options for `rlmesh_ext_legacy_model_serve`. A NULL pointer means all defaults (no
/// auth, no remote shutdown, no timeouts — serves until the process is killed). A
/// 0 timeout / concurrency means "unset".
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshExtLegacyServeOptions {
    pub struct_size: usize,
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
    /// Workflow edition this served model declares (its WANT), NULL or "" to
    /// declare none. The bare `YYYY.MM` base names the contract and selects
    /// whichever spelling of it both sides offer (a dev build's cohort
    /// included); a cohort spelling pins to that exact build. A value that
    /// admits nothing this build offers fails the serve call with
    /// `RLMESH_ERR_INVALID_ARGUMENT`.
    pub workflow_edition: *const c_char,
}

fn serve_model_options(
    bind: BindAddress,
    options: *const RLMeshExtLegacyServeOptions,
) -> Result<ServeModelOptions, CapiError> {
    let mut model_options = ServeModelOptions::new(bind);
    let Some(c_options) =
        unsafe { crate::abi::sizing::read(options, "RLMeshExtLegacyServeOptions") }?
    else {
        return Ok(model_options);
    };
    if !c_options.token.is_null() {
        model_options = model_options.token(cstr_to_str(c_options.token)?);
    }
    Ok(model_options
        .predict_concurrency(
            (c_options.predict_concurrency != 0).then_some(c_options.predict_concurrency),
        )
        .serve_options(serve_options(options)?))
}

/// The core [`ServeOptions`] an `RLMeshExtLegacyServeOptions` names (NULL = defaults),
/// shared by the model and environment servers. The token and the predict
/// concurrency are left to each caller: the model server carries them on
/// `ServeModelOptions`, the environment server its token on its `ext`.
pub(crate) fn serve_options(
    options: *const RLMeshExtLegacyServeOptions,
) -> Result<ServeOptions, CapiError> {
    // SAFETY: the exports' contract is that `options` is NULL or a valid
    // `RLMeshExtLegacyServeOptions` for the duration of the call.
    let options = unsafe { crate::abi::sizing::read(options, "RLMeshExtLegacyServeOptions") }?;
    let call = match &options {
        Some(options) if !options.workflow_edition.is_null() => {
            Some(cstr_to_str(options.workflow_edition)?)
        }
        _ => None,
    };
    // The serve call's own value, then RLMESH_WORKFLOW_EDITION (COR-6); a
    // name this build cannot run is refused where the host typed it. A served
    // peer that floats warns once per process.
    let workflow_edition = rlmesh::binding::resolve_workflow_edition(
        rlmesh::binding::EditionSources {
            call,
            ..Default::default()
        },
        true,
    )
    .map_err(CapiError::from)?;
    let Some(options) = options else {
        return Ok(ServeOptions {
            workflow_edition,
            ..ServeOptions::default()
        });
    };
    let ms = |value: u64| (value != 0).then(|| Duration::from_millis(value));
    Ok(ServeOptions {
        allow_remote_shutdown: options.allow_remote_shutdown,
        idle_timeout: ms(options.idle_timeout_ms),
        drain_timeout: ms(options.drain_timeout_ms),
        close_timeout: ms(options.close_timeout_ms),
        workflow_edition,
        ..ServeOptions::default()
    })
}

/// Serve the model as a `ModelService` endpoint at `bind_address` (e.g.
/// `tcp://127.0.0.1:50051` or `unix:///path.sock`). Blocking — returns when the
/// server stops: a remote shutdown request, an idle timeout, or
/// `rlmesh_ext_legacy_model_cancel`. The same vtable backs every predict, exactly as
/// `rlmesh_ext_legacy_model_run_local`. `options` may be NULL for defaults.
///
/// # Safety
/// `model` must be a live handle; `bind_address` a valid C string; `options` NULL
/// or a valid `RLMeshExtLegacyServeOptions`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_ext_legacy_model_serve(
    model: *mut RLMeshExtLegacyModel,
    bind_address: *const c_char,
    options: *const RLMeshExtLegacyServeOptions,
) -> RLMeshStatus {
    guard(|| {
        let model =
            unsafe { model.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null model"))?;
        let address = cstr_to_str(bind_address)?;
        let bind = BindAddress::parse(address)
            .map_err(|err| CapiError::invalid_arg(format!("invalid bind address: {err}")))?;
        let model_options = serve_model_options(bind, options)?;
        // The cancel arm drops the server future, which skips the close hook the
        // normal shutdown path runs -- so fire it here instead, through the gate
        // the served handler shares: once, and after any in-flight predict has
        // left the C callback.
        let handler = CModelHandler::new(model);
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

/// Stop a blocking `rlmesh_ext_legacy_model_run_local` / `rlmesh_ext_legacy_model_serve` running on
/// `model`. Designed to be called from a thread other than the one blocked in
/// the run (a signal handler's worker, a UI thread); it returns immediately and
/// the blocked call unwinds shortly after. A NULL `model` is a no-op.
///
/// Cancellation is terminal for the handle: a cancelled model refuses further
/// runs (each returns at once). Create a new model to run again.
///
/// A cancelled `rlmesh_ext_legacy_model_serve` returns `RLMESH_OK` after the close hook; a
/// cancelled `rlmesh_ext_legacy_model_run_local` -- this one, or any later one on the
/// handle -- returns `RLMESH_ERR_CANCELLED`, since it has no report to give.
///
/// # Safety
/// `model` must be NULL or a live handle that is not being freed concurrently.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_ext_legacy_model_cancel(model: *mut RLMeshExtLegacyModel) {
    guard_value((), || {
        if let Some(model) = unsafe { model.as_ref() } {
            model.cancel.cancel();
        }
    });
}

/// Free a model handle. NULL is a no-op.
///
/// Freeing closes the model: `on_close` fires here, on the calling thread,
/// unless a serve already fired it. `rlmesh_ext_legacy_model_run_local` borrows the model
/// and never closes it.
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
pub unsafe extern "C" fn rlmesh_ext_legacy_model_free(model: *mut RLMeshExtLegacyModel) {
    guard_value((), || {
        if !model.is_null() {
            drop(unsafe { Box::from_raw(model) });
        }
    });
}

pub(crate) fn cstring(text: &str) -> CString {
    let bytes: Vec<u8> = text.bytes().filter(|&byte| byte != 0).collect();
    CString::new(bytes).unwrap_or_default()
}

pub(crate) fn cstr_to_str<'a>(ptr: *const c_char) -> Result<&'a str, CapiError> {
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

    use rlmesh::binding::model::ModelRouteContext;
    use rlmesh::binding::spaces::{DType, EnvContract, Tensor, spaces::DiscreteBuilder};
    use rlmesh_spaces::MetaValue;

    use super::*;
    use crate::abi::status::rlmesh_last_error_is_recoverable;
    fn rlmesh_value_discrete(n: i64) -> *mut RLMeshValue {
        crate::value::handle::into_handle(rlmesh::Value::Int(n))
    }

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

    /// One `RLMeshExtLegacyEpisode` row as the C side read it, copied out of the call.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct SeenEpisode {
        id: String,
        seeded: bool,
        seed: i64,
        predict_index: u64,
        predict_seed: i64,
    }

    /// What the C callbacks record, reached through `user_data`.
    #[derive(Default)]
    struct Counters {
        predicts: AtomicUsize,
        closes: AtomicUsize,
        episode_ends: AtomicUsize,
        null_episode_ids: AtomicUsize,
        null_observations: AtomicUsize,
        /// Predicts currently inside the C callback.
        in_predict: AtomicUsize,
        /// Times `counting_close` ran while one was.
        closes_during_predict: AtomicUsize,
        /// Every episode row handed to `counting_predict`, in call order.
        rows: std::sync::Mutex<Vec<SeenEpisode>>,
        /// Every non-NULL id handed to `counting_episode_end`, in call order.
        ended_ids: std::sync::Mutex<Vec<String>>,
    }

    impl Counters {
        /// # Safety
        /// `user_data` is the `&Counters` a test handed to `rlmesh_ext_legacy_model_new`.
        unsafe fn of<'a>(user_data: *mut c_void) -> &'a Self {
            unsafe { &*user_data.cast::<Self>() }
        }
        fn user_data(&self) -> *mut c_void {
            std::ptr::from_ref(self).cast_mut().cast::<c_void>()
        }
    }

    fn owned(value: SpaceValue) -> *mut RLMeshValue {
        crate::value::handle::into_handle(rlmesh::binding::value_from_space(
            &rlmesh_spaces::SpaceSpec::default(),
            value,
        ))
    }

    fn u8_box(byte: u8) -> SpaceValue {
        SpaceValue::Box(Tensor::from_vec(vec![byte], vec![1], DType::Uint8).expect("tensor"))
    }

    unsafe extern "C" fn noop_predict(
        _: *mut c_void,
        _: *const RLMeshExtLegacyObservation,
        _: *mut *mut RLMeshValue,
    ) -> c_int {
        0
    }
    unsafe extern "C" fn noop_episode_end(_: *mut c_void, _: *const c_char, _: *const c_char) {}
    unsafe extern "C" fn noop_close(_: *mut c_void) {}

    /// Counts its calls and answers with the Uint8 `Box[1]` action `SmokeEnv`
    /// expects; also records whether the observation row reached C as NULL.
    unsafe extern "C" fn counting_predict(
        user_data: *mut c_void,
        obs: *const RLMeshExtLegacyObservation,
        out: *mut *mut RLMeshValue,
    ) -> c_int {
        let counters = unsafe { Counters::of(user_data) };
        let obs = unsafe { &*obs };
        counters.predicts.fetch_add(1, Ordering::SeqCst);
        if obs.observations.is_null() {
            counters.null_observations.fetch_add(1, Ordering::SeqCst);
        }
        let mut rows = counters.rows.lock().expect("rows");
        for row in 0..obs.num_envs {
            let episode = unsafe { &*obs.episodes.add(row) };
            rows.push(SeenEpisode {
                id: unsafe { CStr::from_ptr(episode.id) }
                    .to_string_lossy()
                    .into_owned(),
                seeded: episode.seeded,
                seed: episode.seed,
                predict_index: episode.predict_index,
                predict_seed: episode.predict_seed,
            });
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
        } else {
            counters.ended_ids.lock().expect("ended ids").push(
                unsafe { CStr::from_ptr(episode_id) }
                    .to_string_lossy()
                    .into_owned(),
            );
        }
    }

    unsafe extern "C" fn counting_close(user_data: *mut c_void) {
        let counters = unsafe { Counters::of(user_data) };
        if counters.in_predict.load(Ordering::SeqCst) != 0 {
            counters
                .closes_during_predict
                .fetch_add(1, Ordering::SeqCst);
        }
        counters.closes.fetch_add(1, Ordering::SeqCst);
    }

    /// `counting_predict` that stays inside the C callback long enough for a
    /// concurrent `rlmesh_ext_legacy_model_cancel` to land mid-predict.
    unsafe extern "C" fn slow_counting_predict(
        user_data: *mut c_void,
        obs: *const RLMeshExtLegacyObservation,
        out: *mut *mut RLMeshValue,
    ) -> c_int {
        let counters = unsafe { Counters::of(user_data) };
        counters.in_predict.fetch_add(1, Ordering::SeqCst);
        let code = unsafe { counting_predict(user_data, obs, out) };
        std::thread::sleep(Duration::from_millis(300));
        counters.in_predict.fetch_sub(1, Ordering::SeqCst);
        code
    }

    /// Echo policy: action = observation (Discrete), so the round trip is checkable.
    unsafe extern "C" fn echo_predict(
        _: *mut c_void,
        obs: *const RLMeshExtLegacyObservation,
        out: *mut *mut RLMeshValue,
    ) -> c_int {
        let obs = unsafe { &*obs };
        for row in 0..obs.num_envs {
            let value = unsafe { &**obs.observations.add(row) };
            let rlmesh::Value::Int(n) = value.value() else {
                return RLMeshStatus::Type as c_int;
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
        _: *const RLMeshExtLegacyObservation,
        out: *mut *mut RLMeshValue,
    ) -> c_int {
        unsafe { decline_after_row(out, PROBE) }
    }

    /// The same decline with an ordinary-sized row, for the tests that assert the
    /// message and status rather than the free.
    unsafe extern "C" fn decline_with_a_row(
        _: *mut c_void,
        _: *const RLMeshExtLegacyObservation,
        out: *mut *mut RLMeshValue,
    ) -> c_int {
        unsafe { decline_after_row(out, 8) }
    }

    unsafe fn decline_after_row(out: *mut *mut RLMeshValue, size: usize) -> c_int {
        let tensor =
            Tensor::from_vec(vec![0u8; size], vec![size as i64], DType::Uint8).expect("row tensor");
        unsafe { *out = owned(SpaceValue::Box(tensor)) };
        unsafe { rlmesh_callback_set_error(c"nope".as_ptr(), true) };
        RLMeshStatus::Model as c_int
    }

    fn full_vtable() -> RLMeshExtLegacyModelVtable {
        RLMeshExtLegacyModelVtable {
            struct_size: std::mem::size_of::<RLMeshExtLegacyModelVtable>(),
            predict: Some(noop_predict),
            on_episode_end: Some(noop_episode_end),
            on_close: Some(noop_close),
        }
    }

    /// The vtable a live-runtime test uses: every hook wired to `Counters`.
    fn counting_vtable() -> RLMeshExtLegacyModelVtable {
        RLMeshExtLegacyModelVtable {
            struct_size: std::mem::size_of::<RLMeshExtLegacyModelVtable>(),
            predict: Some(counting_predict),
            on_episode_end: Some(counting_episode_end),
            on_close: Some(counting_close),
        }
    }

    fn handler(predict: RLMeshExtLegacyPredictFn) -> CModelHandler {
        CModelHandler {
            vtable: RLMeshExtLegacyModelVtable {
                predict: Some(predict),
                ..full_vtable()
            },
            user_data: UserData(std::ptr::null_mut()),
            episodes: EpisodeStore::with_capacity(EPISODE_STORE_CAPACITY),
            close: Arc::default(),
        }
    }

    fn counting_handler(counters: &Counters) -> CModelHandler {
        CModelHandler {
            vtable: counting_vtable(),
            user_data: UserData(counters.user_data()),
            episodes: EpisodeStore::with_capacity(EPISODE_STORE_CAPACITY),
            close: Arc::default(),
        }
    }

    fn episode(id: &str, seed: Option<i64>) -> EpisodeInfo {
        EpisodeInfo {
            episode_id: id.to_string(),
            seed,
            env_id: String::new(),
        }
    }

    /// A one-row observation for `env-1` whose row belongs to `episode`.
    fn episode_observation(episode: EpisodeInfo) -> ModelObservation {
        let mut observation = discrete_observation(&[1]);
        observation.route.env_id = "env-1".to_string();
        observation.route.episodes = vec![episode];
        observation
    }

    /// The `(predict_index, predict_seed)` the C side read on the last row.
    fn last_context(counters: &Counters) -> (u64, i64) {
        let rows = counters.rows.lock().expect("rows");
        let row = rows.last().expect("a predict row was recorded");
        (row.predict_index, row.predict_seed)
    }

    /// Every field of `RLMeshExtLegacyRunOptions` unset, the way a C caller's `{0}` is.
    fn unset_run_options() -> RLMeshExtLegacyRunOptions {
        RLMeshExtLegacyRunOptions {
            struct_size: std::mem::size_of::<RLMeshExtLegacyRunOptions>(),
            max_episodes: 0,
            seeded: false,
            base_seed: 0,
            max_episode_steps: 0,
            max_episode_seconds: 0.0,
            execution_horizon: 0,
            close_env: false,
            episode_seeds: std::ptr::null(),
            num_episode_seeds: 0,
            trial_indexed: false,
            trial_index_base: 0,
        }
    }

    fn discrete_observation(values: &[i64]) -> ModelObservation {
        let space = DiscreteBuilder::new(8).build().expect("space");
        let owned: Vec<SpaceValue> = values.iter().map(|&n| SpaceValue::Discrete(n)).collect();
        let wire = rlmesh_grpc::wire::encode_batched_partial_values(&owned, &space).expect("wire");
        ModelObservation {
            history: Vec::new(),
            step: None,
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
    async fn predict_index_counts_per_episode_and_restarts_after_its_end() {
        let counters = Counters::default();
        let mut handler = counting_handler(&counters);
        let two_rows = || {
            let mut observation = discrete_observation(&[1, 2]);
            observation.route.env_id = "env-1".to_string();
            observation.route.episodes = vec![episode("ep-a", Some(7)), episode("ep-b", None)];
            observation
        };
        for _ in 0..3 {
            handler.predict(two_rows()).await.expect("predict");
        }
        let rows = counters.rows.lock().expect("rows").clone();
        // The seeded episode counts 0, 1, 2 and carries the core helper's seed
        // for each ordinal; the unseeded one counts too, with no seed to mix.
        let seen = |id: &str| -> Vec<SeenEpisode> {
            rows.iter().filter(|row| row.id == id).cloned().collect()
        };
        assert_eq!(
            seen("ep-a"),
            (0..3)
                .map(|index| SeenEpisode {
                    id: "ep-a".to_string(),
                    seeded: true,
                    seed: 7,
                    predict_index: index,
                    predict_seed: predict_seed(7, index),
                })
                .collect::<Vec<_>>()
        );
        assert_eq!(
            seen("ep-b"),
            (0..3)
                .map(|index| SeenEpisode {
                    id: "ep-b".to_string(),
                    seeded: false,
                    seed: 0,
                    predict_index: index,
                    predict_seed: 0,
                })
                .collect::<Vec<_>>()
        );

        // Ending one episode restarts it at 0 while the other keeps counting.
        handler
            .reset_adapter("env-1", vec!["ep-a".to_string()])
            .await
            .expect("reset_adapter");
        handler.predict(two_rows()).await.expect("predict");
        let rows = counters.rows.lock().expect("rows").clone();
        let tail: Vec<(u64, i64)> = rows[rows.len() - 2..]
            .iter()
            .map(|row| (row.predict_index, row.predict_seed))
            .collect();
        assert_eq!(tail, vec![(0, predict_seed(7, 0)), (3, 0)]);

        // An env-wide end (no ids) drops every episode of that env.
        handler
            .reset_adapter("env-1", Vec::new())
            .await
            .expect("reset_adapter");
        handler.predict(two_rows()).await.expect("predict");
        let rows = counters.rows.lock().expect("rows").clone();
        let tail: Vec<u64> = rows[rows.len() - 2..]
            .iter()
            .map(|row| row.predict_index)
            .collect();
        assert_eq!(tail, vec![0, 0]);
    }

    #[tokio::test]
    async fn an_anonymous_row_always_reads_predict_index_zero() {
        // A spec-less lane carries no episode id: nothing to key a counter by,
        // so it reads as a fresh first predict every time and is never stored.
        let counters = Counters::default();
        let mut handler = counting_handler(&counters);
        for _ in 0..2 {
            handler
                .predict(episode_observation(EpisodeInfo::default()))
                .await
                .expect("predict");
            assert_eq!(last_context(&counters), (0, 0));
        }
        assert!(handler.episodes.episodes.is_empty());
    }

    #[tokio::test]
    async fn episode_store_refuses_a_new_episode_past_capacity_without_evicting() {
        assert_eq!(EPISODE_STORE_CAPACITY, 65_536, "Python's default capacity");
        let counters = Counters::default();
        let mut handler = counting_handler(&counters);
        handler.episodes = EpisodeStore::with_capacity(2);
        let single = |id: &str| episode_observation(episode(id, Some(1)));

        handler.predict(single("ep-a")).await.expect("predict");
        handler.predict(single("ep-b")).await.expect("predict");
        // A third live episode is refused before the C predict runs ...
        let error = handler
            .predict(single("ep-c"))
            .await
            .expect_err("past capacity");
        assert!(matches!(error, Error::Runtime(_)), "{error:?}");
        assert!(
            error.to_string().contains("refusing episode ep-c"),
            "{error}"
        );
        assert_eq!(counters.predicts.load(Ordering::SeqCst), 2);
        // ... and nothing was evicted: no end hook, both counts intact.
        assert_eq!(counters.episode_ends.load(Ordering::SeqCst), 0);
        handler.predict(single("ep-a")).await.expect("predict");
        assert_eq!(last_context(&counters), (1, predict_seed(1, 1)));
        handler.predict(single("ep-b")).await.expect("predict");
        assert_eq!(last_context(&counters), (1, predict_seed(1, 1)));

        // A batch that would overflow is refused whole: its known row does not
        // advance either.
        let mut batch = discrete_observation(&[1, 1]);
        batch.route.env_id = "env-1".to_string();
        batch.route.episodes = vec![episode("ep-a", Some(1)), episode("ep-d", Some(1))];
        handler.predict(batch).await.expect_err("past capacity");

        // An episode end frees its slot for a new one.
        handler
            .reset_adapter("env-1", vec!["ep-b".to_string()])
            .await
            .expect("end ep-b");
        handler.predict(single("ep-c")).await.expect("predict");
        assert_eq!(last_context(&counters), (0, predict_seed(1, 0)));
        handler.predict(single("ep-a")).await.expect("predict");
        assert_eq!(last_context(&counters), (2, predict_seed(1, 2)));
        assert_eq!(handler.episodes.episodes.len(), 2);
    }

    #[test]
    fn run_local_options_map_the_trial_index_base_only_when_flagged() {
        let address = ConnectAddress::parse("tcp://127.0.0.1:1").expect("address");
        let mut options = unset_run_options();
        options.trial_index_base = 100;
        assert_eq!(
            run_local_options(address.clone(), &options).trial_index_base,
            None,
            "an unflagged base is unset, like an unflagged base_seed"
        );
        options.trial_indexed = true;
        assert_eq!(
            run_local_options(address.clone(), &options).trial_index_base,
            Some(100)
        );
        assert_eq!(
            run_local_options(address, std::ptr::null()).trial_index_base,
            None
        );
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
    fn read_vtable_rejects_abi4_prefix() {
        let mut vtable = full_vtable();
        vtable.struct_size = std::mem::offset_of!(RLMeshExtLegacyModelVtable, on_episode_end);
        assert!(unsafe { read_vtable(&vtable) }.is_err());
    }

    #[test]
    fn read_vtable_rejects_size_too_small_for_predict() {
        let mut vtable = full_vtable();
        vtable.struct_size = std::mem::offset_of!(RLMeshExtLegacyModelVtable, predict);
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

    /// What `SmokeEnv` records off each reset: the seed and, when the runtime
    /// delivered one, the `trial_index` reset option.
    #[derive(Default)]
    struct ResetLog {
        seeds: Vec<Option<i64>>,
        trials: Vec<Option<i64>>,
    }

    /// A minimal single environment: a Uint8 `Box[1]` obs/action, one step per
    /// episode (reset → 0, step → 1 then terminated, reward 1.0). Declares the
    /// `trial_index` reset option and records the reset seeds + trial ordinals
    /// it was given so a seeded, trial-indexed run is checkable.
    struct SmokeEnv {
        obs_space: rlmesh::binding::spaces::SpaceSpec,
        action_space: rlmesh::binding::spaces::SpaceSpec,
        env_contract: EnvContract,
        resets: Arc<std::sync::Mutex<ResetLog>>,
    }

    impl SmokeEnv {
        fn new(resets: Arc<std::sync::Mutex<ResetLog>>) -> Self {
            let space = |high: f64| {
                rlmesh::binding::spaces::spaces::BoxSpaceBuilder::scalar(0.0, high, vec![1])
                    .dtype(DType::Uint8)
                    .build()
                    .expect("space")
            };
            let (obs_space, action_space) = (space(255.0), space(1.0));
            let reset_options = MetaValue::List(vec![MetaValue::String("trial_index".to_string())]);
            Self {
                env_contract: EnvContract {
                    id: "SmokeEnv-capi".to_string(),
                    observation_space: Some(obs_space.clone()),
                    action_space: Some(action_space.clone()),
                    metadata: Some(
                        [(rlmesh::ENV_RESET_OPTIONS_KEY.to_string(), reset_options)].into(),
                    ),
                    num_envs: 1,
                    ..Default::default()
                },
                obs_space,
                action_space,
                resets,
            }
        }
    }

    #[async_trait]
    impl rlmesh::binding::env::Env for SmokeEnv {
        fn observation_space(&self) -> &rlmesh::binding::spaces::SpaceSpec {
            &self.obs_space
        }
        fn action_space(&self) -> &rlmesh::binding::spaces::SpaceSpec {
            &self.action_space
        }
        fn env_contract(&self) -> &EnvContract {
            &self.env_contract
        }

        async fn reset(
            &mut self,
            req: rlmesh::binding::env::ResetRequest,
        ) -> Result<rlmesh::binding::env::ResetResult, rlmesh::binding::spaces::EnvRuntimeError>
        {
            let trial = match req.options.as_ref().and_then(|o| o.get("trial_index")) {
                Some(MetaValue::Int(trial)) => Some(*trial),
                _ => None,
            };
            let mut resets = self.resets.lock().expect("reset log");
            resets.seeds.push(req.seed);
            resets.trials.push(trial);
            Ok(rlmesh::binding::env::ResetResult {
                observation: Some(u8_box(0)),
                info: None,
                episode_id: None,
            })
        }

        async fn step(
            &mut self,
            _req: rlmesh::binding::env::StepRequest,
        ) -> Result<rlmesh::binding::env::StepResult, rlmesh::binding::spaces::EnvRuntimeError>
        {
            Ok(rlmesh::binding::env::StepResult {
                observation: Some(u8_box(1)),
                reward: 1.0,
                terminated: true,
                truncated: false,
                info: None,
            })
        }

        async fn render(
            &mut self,
            _req: rlmesh::binding::env::RenderRequest,
        ) -> Result<rlmesh::binding::env::RenderResult, rlmesh::binding::spaces::EnvRuntimeError>
        {
            Ok(rlmesh::binding::env::RenderResult::default())
        }

        async fn close(
            &mut self,
            _req: rlmesh::binding::env::CloseRequest,
        ) -> Result<rlmesh::binding::env::CloseResult, rlmesh::binding::spaces::EnvRuntimeError>
        {
            Ok(rlmesh::binding::env::CloseResult)
        }
    }

    /// A `SmokeEnv` served on a loopback port by its own thread + runtime, so a
    /// blocking capi export can be driven against it from the test thread.
    struct EnvHarness {
        address: String,
        resets: Arc<std::sync::Mutex<ResetLog>>,
        stop: CancellationToken,
        thread: Option<std::thread::JoinHandle<()>>,
    }

    impl EnvHarness {
        fn start() -> Self {
            let resets = Arc::new(std::sync::Mutex::new(ResetLog::default()));
            let stop = CancellationToken::new();
            let (tx, rx) = std::sync::mpsc::channel();
            let (env_resets, serve_stop) = (Arc::clone(&resets), stop.clone());
            let thread = std::thread::spawn(move || {
                let runtime = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(1)
                    .enable_all()
                    .build()
                    .expect("env runtime");
                runtime.block_on(async move {
                    let bound = rlmesh::binding::env::WireEnvServer::new(SmokeEnv::new(env_resets))
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
                resets,
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

    fn new_model(
        vtable: &RLMeshExtLegacyModelVtable,
        user_data: *mut c_void,
    ) -> *mut RLMeshExtLegacyModel {
        let mut model: *mut RLMeshExtLegacyModel = std::ptr::null_mut();
        assert_eq!(
            unsafe { rlmesh_ext_legacy_model_new(vtable, user_data, &mut model) },
            RLMeshStatus::Ok
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
        let options = RLMeshExtLegacyRunOptions {
            struct_size: std::mem::size_of::<RLMeshExtLegacyRunOptions>(),
            max_episodes: 3,
            episode_seeds: seeds.as_ptr(),
            num_episode_seeds: seeds.len(),
            trial_indexed: true,
            trial_index_base: 100,
            ..unset_run_options()
        };
        let mut report = RLMeshExtLegacyRunReport::default();
        let status = unsafe {
            rlmesh_ext_legacy_model_run_local(model, address.as_ptr(), &options, &mut report)
        };
        assert_eq!(status, RLMeshStatus::Ok, "{}", last_error_message());

        assert_eq!(report.total_episodes, 3);
        assert_eq!(report.total_steps, 3);
        assert_eq!(report.total_reward, 3.0);
        assert_eq!(report.mean_reward, 1.0);
        assert_eq!(report.terminated_episodes, 3);
        assert_eq!(report.truncated_episodes, 0);
        assert_eq!(counters.predicts.load(Ordering::SeqCst), 3);
        // The run borrows the model: its close hook waits for the free.
        assert_eq!(counters.closes.load(Ordering::SeqCst), 0);
        // The explicit per-episode seeds reached the env's resets in order, and
        // the trial ordinals walked from the base — the env declared the option.
        let resets = env.resets.lock().expect("reset log");
        assert_eq!(resets.seeds, vec![Some(11), Some(22), Some(33)]);
        assert_eq!(resets.trials, vec![Some(100), Some(101), Some(102)]);
        drop(resets);
        // One-step episodes: every predict is that episode's first, seeded with
        // the core helper's mix of its reset seed and ordinal 0.
        let rows = counters.rows.lock().expect("rows").clone();
        assert_eq!(
            rows.iter()
                .map(|row| (row.seeded, row.seed, row.predict_index, row.predict_seed))
                .collect::<Vec<_>>(),
            seeds
                .iter()
                .map(|&seed| (true, seed, 0, predict_seed(seed, 0)))
                .collect::<Vec<_>>()
        );

        // Cancellation is terminal, and says so with its own status: the next run
        // on this handle stops at once rather than reporting a generic internal
        // error a caller would have to string-match.
        unsafe { rlmesh_ext_legacy_model_cancel(model) };
        let cancelled = unsafe {
            rlmesh_ext_legacy_model_run_local(
                model,
                address.as_ptr(),
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(
            cancelled,
            RLMeshStatus::Cancelled,
            "{}",
            last_error_message()
        );
        // A cancelled run does not close the model either; the free does, once.
        assert_eq!(counters.closes.load(Ordering::SeqCst), 0);
        unsafe { rlmesh_ext_legacy_model_free(model) };
        assert_eq!(counters.closes.load(Ordering::SeqCst), 1);
    }

    /// Runs borrow the model: two runs on one handle leave it open (no
    /// `on_close` between or after them), and freeing it closes it exactly once.
    #[test]
    fn run_local_borrows_the_model_and_free_closes_it() {
        let env = EnvHarness::start();
        let counters = Counters::default();
        let model = new_model(&counting_vtable(), counters.user_data());
        let address = CString::new(env.address.clone()).expect("address");
        for _ in 0..2 {
            let status = unsafe {
                rlmesh_ext_legacy_model_run_local(
                    model,
                    address.as_ptr(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                )
            };
            assert_eq!(status, RLMeshStatus::Ok, "{}", last_error_message());
            assert_eq!(counters.closes.load(Ordering::SeqCst), 0);
        }
        assert_eq!(counters.predicts.load(Ordering::SeqCst), 2);
        unsafe { rlmesh_ext_legacy_model_free(model) };
        assert_eq!(counters.closes.load(Ordering::SeqCst), 1);
    }

    /// A zeroed `max_episodes` is unset, not unbounded: Python's default budget
    /// of one episode, or one per seed; a count that disagrees with the seeds
    /// is an argument error, refused before the env is dialed.
    #[test]
    fn run_local_budget_defaults_like_python() {
        let env = EnvHarness::start();
        let counters = Counters::default();
        let model = new_model(&counting_vtable(), counters.user_data());
        let address = CString::new(env.address.clone()).expect("address");

        // NULL options and a zeroed struct both run exactly one episode.
        let mut report = RLMeshExtLegacyRunReport::default();
        let status = unsafe {
            rlmesh_ext_legacy_model_run_local(
                model,
                address.as_ptr(),
                std::ptr::null(),
                &mut report,
            )
        };
        assert_eq!(status, RLMeshStatus::Ok, "{}", last_error_message());
        assert_eq!(report.total_episodes, 1);
        let unset = unset_run_options();
        let status = unsafe {
            rlmesh_ext_legacy_model_run_local(model, address.as_ptr(), &unset, &mut report)
        };
        assert_eq!(status, RLMeshStatus::Ok, "{}", last_error_message());
        assert_eq!(report.total_episodes, 1);

        // Seeds alone set the count.
        let seeds = [5_i64, 6];
        let seeded = RLMeshExtLegacyRunOptions {
            struct_size: std::mem::size_of::<RLMeshExtLegacyRunOptions>(),
            episode_seeds: seeds.as_ptr(),
            num_episode_seeds: seeds.len(),
            ..unset_run_options()
        };
        let status = unsafe {
            rlmesh_ext_legacy_model_run_local(model, address.as_ptr(), &seeded, &mut report)
        };
        assert_eq!(status, RLMeshStatus::Ok, "{}", last_error_message());
        assert_eq!(report.total_episodes, 2);

        // A count that disagrees with the seeds is refused.
        let mismatched = RLMeshExtLegacyRunOptions {
            struct_size: std::mem::size_of::<RLMeshExtLegacyRunOptions>(),
            max_episodes: 3,
            ..seeded
        };
        let status = unsafe {
            rlmesh_ext_legacy_model_run_local(
                model,
                address.as_ptr(),
                &mismatched,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(status, RLMeshStatus::InvalidArgument);
        assert!(
            last_error_message().contains("does not match"),
            "{}",
            last_error_message()
        );
        unsafe { rlmesh_ext_legacy_model_free(model) };
    }

    #[test]
    fn run_local_surfaces_a_model_decline_as_a_recoverable_model_error() {
        // The driver used to flatten every failure to Internal/non-recoverable;
        // a C decline must arrive as RLMESH_ERR_MODEL with its own message.
        let env = EnvHarness::start();
        let vtable = RLMeshExtLegacyModelVtable {
            predict: Some(decline_with_a_row),
            ..full_vtable()
        };
        let model = new_model(&vtable, std::ptr::null_mut());
        let address = CString::new(env.address.clone()).expect("address");
        let status = unsafe {
            rlmesh_ext_legacy_model_run_local(
                model,
                address.as_ptr(),
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        unsafe { rlmesh_ext_legacy_model_free(model) };

        assert_eq!(status, RLMeshStatus::Model);
        assert!(
            last_error_message().contains("nope"),
            "{}",
            last_error_message()
        );
        assert_eq!(rlmesh_last_error_is_recoverable(), 1);
    }

    unsafe extern "C" fn fail_with_a_type_error(
        _: *mut c_void,
        _: *const RLMeshExtLegacyObservation,
        _: *mut *mut RLMeshValue,
    ) -> c_int {
        unsafe { rlmesh_callback_set_error_class(8, c"bad observation".as_ptr()) };
        RLMeshStatus::Type as c_int
    }

    #[test]
    fn a_callback_error_class_propagates_unchanged_in_process() {
        // R-12: a class set with rlmesh_callback_set_error_class is the run's
        // error class and text in process, as a Python author's TypeError.
        let env = EnvHarness::start();
        let vtable = RLMeshExtLegacyModelVtable {
            predict: Some(fail_with_a_type_error),
            ..full_vtable()
        };
        let model = new_model(&vtable, std::ptr::null_mut());
        let address = CString::new(env.address.clone()).expect("address");
        let status = unsafe {
            rlmesh_ext_legacy_model_run_local(
                model,
                address.as_ptr(),
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        unsafe { rlmesh_ext_legacy_model_free(model) };
        assert_eq!(status, RLMeshStatus::Type, "{}", last_error_message());
        assert_eq!(
            crate::abi::status::rlmesh_last_error_class(),
            crate::abi::status::RLMeshErrorClass::TypeError
        );
        assert_eq!(last_error_message(), "bad observation");
        assert_eq!(rlmesh_last_error_is_recoverable(), 0);
    }

    /// `RLMeshExtLegacyServeOptions.workflow_edition` is the served model's sticky
    /// declaration: it reaches `ServeOptions`, NULL/"" declares nothing, and a
    /// name this build cannot drive is refused where the host typed it, naming
    /// both the value and the retained list.
    #[test]
    fn serve_options_carry_the_declared_workflow_edition() {
        let bind = || BindAddress::parse("tcp://127.0.0.1:0").expect("bind address");

        // This build's own edition: the value `rlmesh` reports and the one
        // a host is told to paste. A bare sealed base is NOT interchangeable with
        // it on a prerelease build, which is why the literal is not hardcoded.
        let current = rlmesh::binding::CURRENT_WORKFLOW_EDITION;
        let declared = CString::new(current).expect("edition cstr");
        let options = RLMeshExtLegacyServeOptions {
            struct_size: std::mem::size_of::<RLMeshExtLegacyServeOptions>(),
            workflow_edition: declared.as_ptr(),
            ..serve_options(false)
        };
        let Ok(resolved) = serve_model_options(bind(), std::ptr::from_ref(&options)) else {
            panic!("{current} must resolve: {}", last_error_message())
        };
        assert_eq!(resolved.serve.workflow_edition.as_deref(), Some(current));

        for blank in [std::ptr::null(), c"".as_ptr(), c"   ".as_ptr()] {
            let options = RLMeshExtLegacyServeOptions {
                struct_size: std::mem::size_of::<RLMeshExtLegacyServeOptions>(),
                workflow_edition: blank,
                ..serve_options(false)
            };
            let Ok(resolved) = serve_model_options(bind(), std::ptr::from_ref(&options)) else {
                panic!("a blank edition must declare nothing")
            };
            assert_eq!(resolved.serve.workflow_edition, None);
        }

        let unknown = CString::new("2099.01").expect("edition cstr");
        let options = RLMeshExtLegacyServeOptions {
            struct_size: std::mem::size_of::<RLMeshExtLegacyServeOptions>(),
            workflow_edition: unknown.as_ptr(),
            ..serve_options(false)
        };
        let Err(error) = serve_model_options(bind(), std::ptr::from_ref(&options)) else {
            panic!("2099.01 is not a retained edition and must be refused")
        };
        assert_eq!(error.status, RLMeshStatus::InvalidArgument);
        assert!(error.message.contains("2099.01"), "{}", error.message);
        assert!(
            error
                .message
                .contains(rlmesh::binding::CURRENT_WORKFLOW_EDITION),
            "{}",
            error.message
        );
    }

    /// `RLMESH_WORKFLOW_EDITION` declares for a serve call that sets none, as
    /// in Python (COR-6), and a set-but-blank one floats deliberately.
    #[test]
    #[allow(unsafe_code)]
    fn serve_options_read_the_workflow_edition_variable() {
        let current = rlmesh::current_workflow_edition();
        // SAFETY: nextest runs each test in its own process.
        unsafe { std::env::set_var("RLMESH_WORKFLOW_EDITION", current) };
        let resolved = super::serve_options(std::ptr::null())
            .unwrap_or_else(|_| panic!("resolves: {}", last_error_message()));
        assert_eq!(resolved.workflow_edition.as_deref(), Some(current));
        let explicit = CString::new(rlmesh::binding::CURRENT_WORKFLOW_EDITION).expect("cstr");
        let options = RLMeshExtLegacyServeOptions {
            struct_size: std::mem::size_of::<RLMeshExtLegacyServeOptions>(),
            workflow_edition: explicit.as_ptr(),
            ..serve_options(false)
        };
        let resolved = super::serve_options(std::ptr::from_ref(&options))
            .unwrap_or_else(|_| panic!("resolves: {}", last_error_message()));
        assert_eq!(
            resolved.workflow_edition.as_deref(),
            Some(rlmesh::binding::CURRENT_WORKFLOW_EDITION),
            "the call's own value outranks the variable"
        );
        unsafe { std::env::set_var("RLMESH_WORKFLOW_EDITION", " ") };
        let resolved = super::serve_options(std::ptr::null())
            .unwrap_or_else(|_| panic!("resolves: {}", last_error_message()));
        assert_eq!(resolved.workflow_edition, None);
        unsafe { std::env::remove_var("RLMESH_WORKFLOW_EDITION") };
    }

    /// Hand a raw model/args tuple to a serve thread. The test joins it before
    /// dropping anything it points at.
    struct ServeArgs {
        model: *mut RLMeshExtLegacyModel,
        bind: *const c_char,
        options: *const RLMeshExtLegacyServeOptions,
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

    fn serve_options(allow_remote_shutdown: bool) -> RLMeshExtLegacyServeOptions {
        RLMeshExtLegacyServeOptions {
            struct_size: std::mem::size_of::<RLMeshExtLegacyServeOptions>(),
            token: std::ptr::null(),
            allow_remote_shutdown,
            idle_timeout_ms: 0,
            drain_timeout_ms: 0,
            close_timeout_ms: 0,
            predict_concurrency: 0,
            workflow_edition: std::ptr::null(),
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
            unsafe { rlmesh_ext_legacy_model_serve(args.model, args.bind, args.options) }
        });

        let runtime = tokio::runtime::Runtime::new().expect("client runtime");
        runtime.block_on(async {
            let mut client =
                rlmesh_grpc::ModelClient::connect_with_retry(&address, "", &connect_options())
                    .await
                    .expect("client connects to the C-served model");
            client.handshake().await.expect("handshake");

            let env_contract = SmokeEnv::new(Arc::new(std::sync::Mutex::new(ResetLog::default())))
                .env_contract
                .clone();
            let mut remote = rlmesh::binding::model::ModelClient::connect(&address, env_contract)
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

        assert_eq!(server.join().expect("serve thread"), RLMeshStatus::Ok);
        assert_eq!(counters.predicts.load(Ordering::SeqCst), 1);
        assert_eq!(counters.closes.load(Ordering::SeqCst), 1);
        unsafe { rlmesh_ext_legacy_model_free(model) };
        // The serve already closed the model; the free does not close it again.
        assert_eq!(counters.closes.load(Ordering::SeqCst), 1);
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
            unsafe { rlmesh_ext_legacy_model_serve(args.model, args.bind, args.options) }
        });

        // Wait until the server is really accepting, so the cancel races nothing.
        let runtime = tokio::runtime::Runtime::new().expect("client runtime");
        runtime.block_on(async {
            rlmesh_grpc::ModelClient::connect_with_retry(&address, "", &connect_options())
                .await
                .expect("server is up");
        });

        // Remote shutdown is off and there is no idle timeout: only cancel ends it.
        unsafe { rlmesh_ext_legacy_model_cancel(model) };
        assert_eq!(server.join().expect("serve thread"), RLMeshStatus::Ok);
        assert_eq!(counters.closes.load(Ordering::SeqCst), 1);
        unsafe { rlmesh_ext_legacy_model_free(model) };
    }

    #[test]
    fn cancel_mid_predict_closes_once_after_the_callback_returns() {
        // Cancelling a serve drops the server future, so the C `on_close` is
        // fired by the cancel arm -- it must still wait for the predict the C
        // side is inside (a connection task tonic spawned, not the dropped
        // future) and must not run a second time.
        let port = reserve_port();
        let address = format!("tcp://127.0.0.1:{port}");
        let counters = Counters::default();
        let vtable = RLMeshExtLegacyModelVtable {
            predict: Some(slow_counting_predict),
            ..counting_vtable()
        };
        let model = new_model(&vtable, counters.user_data());
        let bind = CString::new(address.clone()).expect("bind cstr");
        let options = serve_options(false);
        let args = ServeArgs {
            model,
            bind: bind.as_ptr(),
            options: std::ptr::from_ref(&options),
        };
        let server = std::thread::spawn(move || {
            let args = args;
            unsafe { rlmesh_ext_legacy_model_serve(args.model, args.bind, args.options) }
        });

        // A client that keeps one predict in flight while this thread cancels.
        let client_address = address.clone();
        let client = std::thread::spawn(move || {
            let runtime = tokio::runtime::Runtime::new().expect("client runtime");
            runtime.block_on(async move {
                rlmesh_grpc::ModelClient::connect_with_retry(
                    &client_address,
                    "",
                    &connect_options(),
                )
                .await
                .expect("server is up");
                let env_contract =
                    SmokeEnv::new(Arc::new(std::sync::Mutex::new(ResetLog::default())))
                        .env_contract
                        .clone();
                let mut remote =
                    rlmesh::binding::model::ModelClient::connect(&client_address, env_contract)
                        .await
                        .expect("remote model");
                remote.reset(None);
                // The server goes away mid-call, so the predict itself may fail.
                let _ = remote.predict(u8_box(3)).await;
            });
        });

        // Cancel only once the C callback is really running.
        for _ in 0..1_000 {
            if counters.in_predict.load(Ordering::SeqCst) != 0 {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(
            counters.in_predict.load(Ordering::SeqCst),
            1,
            "the C predict callback never started"
        );
        unsafe { rlmesh_ext_legacy_model_cancel(model) };

        assert_eq!(server.join().expect("serve thread"), RLMeshStatus::Ok);
        client.join().expect("client thread");
        assert_eq!(
            counters.closes.load(Ordering::SeqCst),
            1,
            "on_close ran exactly once"
        );
        assert_eq!(
            counters.closes_during_predict.load(Ordering::SeqCst),
            0,
            "on_close waited for the in-flight predict"
        );
        unsafe { rlmesh_ext_legacy_model_free(model) };
    }
}
