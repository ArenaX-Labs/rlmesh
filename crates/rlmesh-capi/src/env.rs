//! The environment path: a C callback vtable adapted into a core `rlmesh::Env`,
//! plus a handle that binds and serves it as a gRPC `EnvService` endpoint.
//!
//! The core serves a scalar env on one dedicated lane thread, so every C
//! callback (reset, step, render, close) runs on that same thread, one at a
//! time, never on the thread that created the env. A callback's error is read
//! back on that thread right after it returns and travels on as an
//! `EnvRuntimeError` value.
#![allow(unsafe_code)] // FFI: raw callback pointers + repr(C) structs.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::sync::Mutex;

use async_trait::async_trait;
use image::ExtendedColorType;
use image::ImageEncoder;
use image::codecs::png::PngEncoder;
use rlmesh::{
    BindAddress, BoundEnvServer, CancellationToken, CloseRequest, CloseResult,
    ENV_RESET_OPTIONS_KEY, Env, EnvServer, RenderRequest, RenderResult, ResetRequest, ResetResult,
    StepRequest, StepResult, TRIAL_INDEX_OPTION,
};
use rlmesh_adapters::v1::{ENV_METADATA_KEY, EnvTags, SpaceView, join, reject_unknowns_env};
use rlmesh_spaces::errors::EnvRuntimeError;
use rlmesh_spaces::{DType, EnvContract, MetaMap, MetaValue, RenderFrame, SpaceSpec, SpaceValue};

use crate::abi::status::{
    CapiError, RlmeshStatus, clear_last_error, guard, guard_value, last_error_message,
};
use crate::adapters::{json_to_meta, meta_to_json};
use crate::codec::RlmeshBytes;
use crate::model::{RlmeshServeOptions, cstr_to_str, serve_options, vtable_field};
use crate::spaces::{RlmeshSpaceSpec, spec_ref};
use crate::value::handle::RlmeshValue;

/// What `reset` receives. Every pointer is valid only for the duration of the
/// call.
#[repr(C)]
pub struct RlmeshResetArgs {
    /// Whether `seed` carries an explicit reset seed.
    pub seeded: bool,
    /// The reset seed; only meaningful when `seeded`.
    pub seed: i64,
    /// Whether `trial_index` is set: the runtime sends it only to an env that
    /// declared the `trial_index` reset option.
    pub has_trial_index: bool,
    /// The trial ordinal this episode runs; only meaningful when `has_trial_index`.
    pub trial_index: i64,
    /// Every reset option as a JSON object (NUL-terminated), or NULL for none.
    pub options_json: *const c_char,
}

/// What `reset` writes. `observation` is an OWNED value (the capi takes it);
/// `info_json` is a borrowed JSON object, or NULL for none.
#[repr(C)]
pub struct RlmeshResetResult {
    pub observation: *mut RlmeshValue,
    pub info_json: *const c_char,
}

/// What `step` writes. `observation` is an OWNED value (the capi takes it);
/// `info_json` is a borrowed JSON object, or NULL for none.
#[repr(C)]
pub struct RlmeshStepResult {
    pub observation: *mut RlmeshValue,
    pub reward: f64,
    pub terminated: bool,
    pub truncated: bool,
    pub info_json: *const c_char,
}

/// Reset callback: start a new episode and write its first observation.
pub type RlmeshEnvResetFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    args: *const RlmeshResetArgs,
    out: *mut RlmeshResetResult,
) -> c_int;
/// Step callback: apply `action` (borrowed; NULL when the request carries
/// none) and write the transition.
pub type RlmeshEnvStepFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    action: *const RlmeshValue,
    out: *mut RlmeshStepResult,
) -> c_int;
/// Render callback: write an OWNED uint8 image value (`[H, W, 3]`, `[H, W, 4]`
/// or `[H, W]`), or leave it NULL for "no frame". The capi PNG-encodes it.
pub type RlmeshEnvRenderFn =
    unsafe extern "C" fn(user_data: *mut c_void, out_frame: *mut *mut RlmeshValue) -> c_int;
/// Close callback: the server stopped; release the simulation.
pub type RlmeshEnvCloseFn = unsafe extern "C" fn(user_data: *mut c_void);

/// The environment callback vtable. Set `struct_size =
/// sizeof(RlmeshEnvVtable)`; fields beyond that are ignored (append-only).
/// `reset` and `step` are required.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RlmeshEnvVtable {
    pub struct_size: usize,
    pub reset: Option<RlmeshEnvResetFn>,
    pub step: Option<RlmeshEnvStepFn>,
    pub render: Option<RlmeshEnvRenderFn>,
    pub close: Option<RlmeshEnvCloseFn>,
}

/// What the env declares: its spaces and the contract metadata a model reads.
/// Set `struct_size = sizeof(RlmeshEnvConfig)`; fields beyond it are unset.
/// Everything is borrowed for the `rlmesh_env_new` call only (copied in).
#[repr(C)]
pub struct RlmeshEnvConfig {
    pub struct_size: usize,
    /// Contract id, e.g. "ChronoArmReach-v0". NULL or "" = "env".
    pub id: *const c_char,
    /// Required.
    pub observation_space: *const RlmeshSpaceSpec,
    /// Required.
    pub action_space: *const RlmeshSpaceSpec,
    /// Adapter `EnvTags` (the v1 JSON wire format), validated against the
    /// spaces at `rlmesh_env_new`. NULL = untagged.
    pub adapter_tags_json: *const c_char,
    /// Reset options the env understands (e.g. "trial_index"); the runtime sends
    /// only declared ones. NULL / 0 = none.
    pub reset_options: *const *const c_char,
    pub num_reset_options: usize,
    /// "rgb_array" when `render` produces frames; NULL = none.
    pub render_mode: *const c_char,
    /// Extra contract metadata as a JSON object. NULL = none.
    pub metadata_json: *const c_char,
}

/// A `*mut c_void` the C author guarantees may move to the lane thread.
#[derive(Clone, Copy)]
struct UserData(*mut c_void);
// SAFETY: the C author guarantees `user_data` is thread-migration-safe, and the
// lane runs one callback at a time, so it is never shared concurrently.
unsafe impl Send for UserData {}
unsafe impl Sync for UserData {}

/// The C vtable as a core scalar `Env`.
struct CEnv {
    vtable: RlmeshEnvVtable,
    user_data: UserData,
    observation_space: SpaceSpec,
    action_space: SpaceSpec,
    contract: EnvContract,
}

fn callback_error(op: &str) -> EnvRuntimeError {
    let message = last_error_message();
    EnvRuntimeError::Runtime(if message.is_empty() {
        format!("environment {op} failed")
    } else {
        message
    })
}

/// Reclaim an owned value the callback wrote (NULL = none).
fn take_value(ptr: *mut RlmeshValue) -> Option<SpaceValue> {
    (!ptr.is_null()).then(|| unsafe { Box::from_raw(ptr) }.0)
}

/// Parse a borrowed info JSON object into a metadata map (NULL = none).
fn info_map(ptr: *const c_char, op: &str) -> Result<Option<MetaMap>, EnvRuntimeError> {
    if ptr.is_null() {
        return Ok(None);
    }
    let text = unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map_err(|_| EnvRuntimeError::InvalidValue(format!("{op} info_json is not UTF-8")))?;
    match json_object(text) {
        Ok(map) => Ok(Some(map)),
        Err(message) => Err(EnvRuntimeError::InvalidValue(format!(
            "{op} info_json: {message}"
        ))),
    }
}

fn json_object(text: &str) -> Result<MetaMap, String> {
    let value: serde_json::Value = serde_json::from_str(text).map_err(|err| err.to_string())?;
    match json_to_meta(&value) {
        MetaValue::Map(map) => Ok(map),
        _ => Err("expected a JSON object".to_string()),
    }
}

/// PNG-encode a uint8 image value: `[H, W, 3]`, `[H, W, 4]`, or `[H, W]`.
fn encode_png(frame: &SpaceValue) -> Result<Vec<u8>, EnvRuntimeError> {
    let SpaceValue::Box(tensor) = frame else {
        return Err(EnvRuntimeError::InvalidValue(
            "render frame must be a Box value".to_string(),
        ));
    };
    if tensor.dtype() != DType::Uint8 {
        return Err(EnvRuntimeError::InvalidValue(
            "render frame must be uint8".to_string(),
        ));
    }
    let dim = |d: i64| u32::try_from(d).unwrap_or(0);
    let (width, height, color) = match tensor.shape() {
        [h, w, 3] => (dim(*w), dim(*h), ExtendedColorType::Rgb8),
        [h, w, 4] => (dim(*w), dim(*h), ExtendedColorType::Rgba8),
        [h, w] => (dim(*w), dim(*h), ExtendedColorType::L8),
        shape => {
            return Err(EnvRuntimeError::InvalidValue(format!(
                "render frame shape {shape:?} is not [H, W, 3], [H, W, 4] or [H, W]"
            )));
        }
    };
    let mut encoded = Vec::new();
    PngEncoder::new(&mut encoded)
        .write_image(&tensor.to_contiguous_bytes(), width, height, color)
        .map_err(|err| EnvRuntimeError::Runtime(format!("encode render frame: {err}")))?;
    Ok(encoded)
}

#[async_trait]
impl Env for CEnv {
    fn observation_space(&self) -> &SpaceSpec {
        &self.observation_space
    }

    fn action_space(&self) -> &SpaceSpec {
        &self.action_space
    }

    fn env_contract(&self) -> &EnvContract {
        &self.contract
    }

    async fn reset(&mut self, req: ResetRequest) -> Result<ResetResult, EnvRuntimeError> {
        let Some(reset) = self.vtable.reset else {
            return Err(EnvRuntimeError::Runtime("env vtable has no reset".into()));
        };
        let trial_index = req
            .options
            .as_ref()
            .and_then(|options| options.get(TRIAL_INDEX_OPTION))
            .and_then(|value| match value {
                MetaValue::Int(index) => Some(*index),
                _ => None,
            });
        let options_json = req.options.as_ref().map(|options| {
            let map = MetaValue::Map(options.clone());
            crate::model::cstring(&meta_to_json(&map).to_string())
        });
        let args = RlmeshResetArgs {
            seeded: req.seed.is_some(),
            seed: req.seed.unwrap_or_default(),
            has_trial_index: trial_index.is_some(),
            trial_index: trial_index.unwrap_or_default(),
            options_json: options_json
                .as_ref()
                .map_or(std::ptr::null(), |s| s.as_ptr()),
        };
        let mut out = RlmeshResetResult {
            observation: std::ptr::null_mut(),
            info_json: std::ptr::null(),
        };
        clear_last_error();
        let status = unsafe { reset(self.user_data.0, &args, &mut out) };
        let observation = take_value(out.observation);
        if status != 0 {
            return Err(callback_error("reset"));
        }
        Ok(ResetResult {
            observation,
            info: info_map(out.info_json, "reset")?,
            episode_id: None,
        })
    }

    async fn step(&mut self, req: StepRequest) -> Result<StepResult, EnvRuntimeError> {
        let Some(step) = self.vtable.step else {
            return Err(EnvRuntimeError::Runtime("env vtable has no step".into()));
        };
        // `RlmeshValue` is repr(transparent) over `SpaceValue`: lend the action.
        let action = req.action.as_ref().map_or(std::ptr::null(), |action| {
            std::ptr::from_ref(action).cast::<RlmeshValue>()
        });
        let mut out = RlmeshStepResult {
            observation: std::ptr::null_mut(),
            reward: 0.0,
            terminated: false,
            truncated: false,
            info_json: std::ptr::null(),
        };
        clear_last_error();
        let status = unsafe { step(self.user_data.0, action, &mut out) };
        let observation = take_value(out.observation);
        if status != 0 {
            return Err(callback_error("step"));
        }
        Ok(StepResult {
            observation,
            reward: out.reward,
            terminated: out.terminated,
            truncated: out.truncated,
            info: info_map(out.info_json, "step")?,
        })
    }

    async fn render(&mut self, _req: RenderRequest) -> Result<RenderResult, EnvRuntimeError> {
        let Some(render) = self.vtable.render else {
            return Ok(RenderResult::default());
        };
        let mut frame: *mut RlmeshValue = std::ptr::null_mut();
        clear_last_error();
        let status = unsafe { render(self.user_data.0, &mut frame) };
        let frame = take_value(frame);
        if status != 0 {
            return Err(callback_error("render"));
        }
        Ok(RenderResult {
            frame: frame
                .map(|frame| encode_png(&frame).map(|frame| RenderFrame { frame }))
                .transpose()?,
        })
    }

    async fn close(&mut self, _req: CloseRequest) -> Result<CloseResult, EnvRuntimeError> {
        // The core calls close once, when the server stops; take the callback so
        // a second call could never reach C.
        if let Some(close) = self.vtable.close.take() {
            unsafe { close(self.user_data.0) };
        }
        Ok(CloseResult)
    }
}

/// An owned environment handle: the C env plus the runtime that serves it.
///
/// Lifecycle: `rlmesh_env_new` -> `rlmesh_env_bind` (learn the address) ->
/// `rlmesh_env_serve` (blocks) -> `rlmesh_env_free`. `rlmesh_env_cancel` is the
/// one call that may overlap a blocking serve, from any thread.
pub struct RlmeshEnv {
    runtime: tokio::runtime::Runtime,
    env: Mutex<Option<CEnv>>,
    bound: Mutex<Option<BoundEnvServer>>,
    cancel: CancellationToken,
}

/// Read the caller's vtable honoring its `struct_size`.
///
/// # Safety
/// `ptr` is non-NULL and its first `struct_size` bytes are valid.
unsafe fn read_vtable(ptr: *const RlmeshEnvVtable) -> Result<RlmeshEnvVtable, CapiError> {
    let base = ptr.cast::<u8>();
    let struct_size = unsafe { (*ptr).struct_size };
    if struct_size == 0 {
        return Err(CapiError::invalid_arg("vtable struct_size is 0"));
    }
    macro_rules! field {
        ($name:ident, $ty:ty) => {
            unsafe {
                vtable_field::<Option<$ty>>(
                    base,
                    struct_size,
                    std::mem::offset_of!(RlmeshEnvVtable, $name),
                )
            }
            .flatten()
        };
    }
    let vtable = RlmeshEnvVtable {
        struct_size,
        reset: field!(reset, RlmeshEnvResetFn),
        step: field!(step, RlmeshEnvStepFn),
        render: field!(render, RlmeshEnvRenderFn),
        close: field!(close, RlmeshEnvCloseFn),
    };
    if vtable.reset.is_none() {
        return Err(CapiError::invalid_arg("env vtable reset is null"));
    }
    if vtable.step.is_none() {
        return Err(CapiError::invalid_arg("env vtable step is null"));
    }
    Ok(vtable)
}

/// The config fields the caller's `struct_size` covers; the rest read as unset.
struct EnvConfig {
    id: *const c_char,
    observation_space: *const RlmeshSpaceSpec,
    action_space: *const RlmeshSpaceSpec,
    adapter_tags_json: *const c_char,
    reset_options: *const *const c_char,
    num_reset_options: usize,
    render_mode: *const c_char,
    metadata_json: *const c_char,
}

/// # Safety
/// `ptr` is non-NULL and its first `struct_size` bytes are valid.
unsafe fn read_config(ptr: *const RlmeshEnvConfig) -> Result<EnvConfig, CapiError> {
    let base = ptr.cast::<u8>();
    let struct_size = unsafe { (*ptr).struct_size };
    if struct_size == 0 {
        return Err(CapiError::invalid_arg("config struct_size is 0"));
    }
    macro_rules! field {
        ($name:ident, $ty:ty, $default:expr) => {
            unsafe {
                vtable_field::<$ty>(
                    base,
                    struct_size,
                    std::mem::offset_of!(RlmeshEnvConfig, $name),
                )
            }
            .unwrap_or($default)
        };
    }
    Ok(EnvConfig {
        id: field!(id, *const c_char, std::ptr::null()),
        observation_space: field!(observation_space, *const RlmeshSpaceSpec, std::ptr::null()),
        action_space: field!(action_space, *const RlmeshSpaceSpec, std::ptr::null()),
        adapter_tags_json: field!(adapter_tags_json, *const c_char, std::ptr::null()),
        reset_options: field!(reset_options, *const *const c_char, std::ptr::null()),
        num_reset_options: field!(num_reset_options, usize, 0),
        render_mode: field!(render_mode, *const c_char, std::ptr::null()),
        metadata_json: field!(metadata_json, *const c_char, std::ptr::null()),
    })
}

fn opt_str<'a>(ptr: *const c_char, what: &str) -> Result<Option<&'a str>, CapiError> {
    if ptr.is_null() {
        return Ok(None);
    }
    cstr_to_str(ptr)
        .map(Some)
        .map_err(|_| CapiError::invalid_arg(format!("{what} is not UTF-8")))
}

/// Build the served contract from a config, validating the adapter tags.
fn build_contract(config: &EnvConfig) -> Result<(SpaceSpec, SpaceSpec, EnvContract), CapiError> {
    let observation_space = spec_ref(config.observation_space)
        .ok_or_else(|| CapiError::invalid_arg("config observation_space is null"))?
        .clone();
    let action_space = spec_ref(config.action_space)
        .ok_or_else(|| CapiError::invalid_arg("config action_space is null"))?
        .clone();

    let mut metadata = MetaMap::new();
    if let Some(text) = opt_str(config.metadata_json, "metadata_json")? {
        metadata = json_object(text)
            .map_err(|message| CapiError::invalid_arg(format!("metadata_json: {message}")))?;
    }
    if config.num_reset_options != 0 {
        if config.reset_options.is_null() {
            return Err(CapiError::invalid_arg("null reset_options"));
        }
        let names =
            unsafe { std::slice::from_raw_parts(config.reset_options, config.num_reset_options) };
        let names = names
            .iter()
            .map(|&name| {
                opt_str(name, "reset option")?
                    .map(|name| MetaValue::String(name.to_string()))
                    .ok_or_else(|| CapiError::invalid_arg("null reset option"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        metadata.insert(ENV_RESET_OPTIONS_KEY.to_string(), MetaValue::List(names));
    }
    if let Some(text) = opt_str(config.adapter_tags_json, "adapter_tags_json")? {
        // Publishing tags is a PUBLISH door, exactly as the Python `tag()`: a
        // typo or a tag that disagrees with the spaces fails here, not on the
        // model's side of the wire.
        let tags: EnvTags = serde_json::from_str(text)
            .map_err(|err| CapiError::invalid_arg(format!("invalid adapter tags: {err}")))?;
        reject_unknowns_env(&tags).map_err(|message| {
            CapiError::invalid_arg(format!("invalid adapter tags: {message}"))
        })?;
        join(
            &tags,
            &SpaceView::from(&observation_space),
            &SpaceView::from(&action_space),
        )
        .map_err(|err| {
            CapiError::invalid_arg(format!("adapter tags do not fit the spaces: {err}"))
        })?;
        let tags_json: serde_json::Value = serde_json::from_str(text)
            .map_err(|err| CapiError::invalid_arg(format!("invalid adapter tags: {err}")))?;
        metadata.insert(ENV_METADATA_KEY.to_string(), json_to_meta(&tags_json));
    }

    let id = opt_str(config.id, "id")?
        .filter(|id| !id.is_empty())
        .unwrap_or("env")
        .to_string();
    let render_mode = opt_str(config.render_mode, "render_mode")?
        .unwrap_or_default()
        .to_string();
    let contract = EnvContract {
        id,
        observation_space: Some(observation_space.clone()),
        action_space: Some(action_space.clone()),
        metadata: (!metadata.is_empty()).then_some(metadata),
        render_mode,
        num_envs: 1,
        ..Default::default()
    };
    Ok((observation_space, action_space, contract))
}

/// Create an environment from a callback vtable and a config. Both are copied
/// in (the spaces cloned), so neither need outlive this call; `user_data` is
/// kept by pointer, passed to every callback, and must outlive the env.
///
/// # Safety
/// `vtable` and `config` must be valid for the call; `out` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_env_new(
    vtable: *const RlmeshEnvVtable,
    config: *const RlmeshEnvConfig,
    user_data: *mut c_void,
    out: *mut *mut RlmeshEnv,
) -> RlmeshStatus {
    guard(|| {
        if vtable.is_null() {
            return Err(CapiError::invalid_arg("null vtable"));
        }
        if config.is_null() {
            return Err(CapiError::invalid_arg("null config"));
        }
        let out = unsafe { out.as_mut() }.ok_or_else(|| CapiError::invalid_arg("null out"))?;
        let vtable = unsafe { read_vtable(vtable) }?;
        let config = unsafe { read_config(config) }?;
        let (observation_space, action_space, contract) = build_contract(&config)?;
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|err| CapiError::internal(format!("failed to build runtime: {err}")))?;
        let env = CEnv {
            vtable,
            user_data: UserData(user_data),
            observation_space,
            action_space,
            contract,
        };
        *out = Box::into_raw(Box::new(RlmeshEnv {
            runtime,
            env: Mutex::new(Some(env)),
            bound: Mutex::new(None),
            cancel: CancellationToken::new(),
        }));
        Ok(())
    })
}

/// Bind the env server to `bind_address` (`tcp://host:port`, `host:port`, or
/// `unix:///path`) without serving yet, and write the resolved address (the
/// OS-assigned port for port 0) to `out_address` when it is non-NULL (free with
/// `rlmesh_bytes_free`; UTF-8, not NUL-terminated). Once per handle. `options`
/// may be NULL for defaults; `predict_concurrency` does not apply to an env.
///
/// # Safety
/// `env` must be a live handle; `bind_address` a valid C string; `options` NULL
/// or valid; `out_address` NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_env_bind(
    env: *mut RlmeshEnv,
    bind_address: *const c_char,
    options: *const RlmeshServeOptions,
    out_address: *mut RlmeshBytes,
) -> RlmeshStatus {
    guard(|| {
        let handle = unsafe { env.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null env"))?;
        let address = cstr_to_str(bind_address)?;
        let bind = BindAddress::parse(address)
            .map_err(|err| CapiError::invalid_arg(format!("invalid bind address: {err}")))?;
        let mut serve = serve_options(options)?;
        if let Some(c_options) = unsafe { options.as_ref() } {
            serve.token = opt_str(c_options.token, "token")?
                .filter(|token| !token.is_empty())
                .map(str::to_string);
        }
        let cenv = lock(&handle.env)
            .take()
            .ok_or_else(|| CapiError::invalid_arg("env is already bound"))?;
        let bound = handle
            .runtime
            .block_on(EnvServer::new(cenv).bind_with_options(bind, serve))
            .map_err(CapiError::from)?;
        if !out_address.is_null() {
            let address = bound.local_addr().to_string();
            unsafe { *out_address = RlmeshBytes::from_vec(address.into_bytes()) };
        }
        *lock(&handle.bound) = Some(bound);
        Ok(())
    })
}

/// Serve the bound env until it stops: a remote shutdown request, an idle
/// timeout, or `rlmesh_env_cancel`. Blocking. The env's `close` callback runs
/// once, on the lane thread, before this returns. Requires `rlmesh_env_bind`.
///
/// # Safety
/// `env` must be a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_env_serve(env: *mut RlmeshEnv) -> RlmeshStatus {
    guard(|| {
        let handle = unsafe { env.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null env"))?;
        let bound = lock(&handle.bound)
            .take()
            .ok_or_else(|| CapiError::invalid_arg("env is not bound (call rlmesh_env_bind)"))?;
        let trigger = bound.shutdown_trigger();
        let cancel = handle.cancel.clone();
        handle
            .runtime
            .block_on(async move {
                // Cancelling triggers the server's own graceful shutdown (drain,
                // then the close hook) rather than dropping the serve future.
                let watcher = tokio::spawn(async move {
                    cancel.cancelled().await;
                    trigger.trigger("cancelled by rlmesh_env_cancel");
                });
                let result = bound.serve().await;
                watcher.abort();
                result
            })
            .map_err(CapiError::from)
    })
}

/// Stop a blocking `rlmesh_env_serve` from another thread (a signal-handling
/// thread, a UI thread). Returns at once; the serve drains, closes the env and
/// returns `RLMESH_OK`. Terminal: a cancelled handle's later serve stops at
/// once. NULL is a no-op.
///
/// # Safety
/// `env` must be NULL or a live handle not being freed concurrently.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_env_cancel(env: *mut RlmeshEnv) {
    guard_value((), || {
        if let Some(handle) = unsafe { env.as_ref() } {
            handle.cancel.cancel();
        }
    });
}

/// Free an env handle. NULL is a no-op. Must NOT be called from inside one of
/// the env's own callbacks. A handle freed without serving never runs `close`.
///
/// # Safety
/// `env` must be NULL or a handle this thread owns and has not freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_env_free(env: *mut RlmeshEnv) {
    guard_value((), || {
        if !env.is_null() {
            drop(unsafe { Box::from_raw(env) });
        }
    });
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use rlmesh::RemoteEnv;
    use rlmesh_spaces::Tensor;

    use super::*;
    use crate::model::rlmesh_callback_set_error;
    use crate::spaces::{rlmesh_space_box, rlmesh_space_dict, rlmesh_space_free};
    use crate::value::dtype::RlmeshDType;
    use crate::value::handle::into_handle;

    const F32: RlmeshDType = RlmeshDType {
        code: 2,
        bits: 32,
        lanes: 1,
    };
    const U8: RlmeshDType = RlmeshDType {
        code: 1,
        bits: 8,
        lanes: 1,
    };

    const TAGS: &str = r#"{"observation": {"eef_pos": {"type": "state", "role": "proprio/eef_pos"}},
        "action": {"components": [{"role": "action/gripper", "dim": 1}]}}"#;

    /// What the C side saw, shared with the test through `user_data`.
    #[derive(Default)]
    struct Probe {
        resets: Mutex<Vec<(Option<i64>, Option<i64>)>>,
        actions: Mutex<Vec<f32>>,
        closes: AtomicUsize,
        steps: AtomicUsize,
        info: Mutex<CString>,
    }

    fn eef(value: f32) -> *mut RlmeshValue {
        let tensor = Tensor::from_vec(value.to_le_bytes().repeat(3), vec![3], DType::Float32)
            .expect("tensor");
        into_handle(SpaceValue::Dict(
            [("eef_pos".to_string(), SpaceValue::Box(tensor))].into(),
        ))
    }

    fn action(value: f32) -> SpaceValue {
        SpaceValue::Box(
            Tensor::from_vec(value.to_le_bytes().to_vec(), vec![1], DType::Float32)
                .expect("tensor"),
        )
    }

    unsafe extern "C" fn reset(
        user_data: *mut c_void,
        args: *const RlmeshResetArgs,
        out: *mut RlmeshResetResult,
    ) -> c_int {
        let probe = unsafe { &*user_data.cast::<Probe>() };
        let args = unsafe { &*args };
        probe.resets.lock().unwrap().push((
            args.seeded.then_some(args.seed),
            args.has_trial_index.then_some(args.trial_index),
        ));
        unsafe { (*out).observation = eef(0.0) };
        0
    }

    unsafe extern "C" fn step(
        user_data: *mut c_void,
        action: *const RlmeshValue,
        out: *mut RlmeshStepResult,
    ) -> c_int {
        let probe = unsafe { &*user_data.cast::<Probe>() };
        let mut tensor = std::mem::MaybeUninit::<crate::value::tensor::RlmeshTensor>::zeroed();
        if unsafe { crate::value::handle::rlmesh_value_as_tensor(action, tensor.as_mut_ptr()) }
            != RlmeshStatus::Ok
        {
            return 1;
        }
        let chosen = unsafe { *(*tensor.as_ptr()).data.cast::<f32>() };
        if chosen < 0.0 {
            let message = CString::new("negative action declined").unwrap();
            unsafe { rlmesh_callback_set_error(message.as_ptr(), false) };
            return 1;
        }
        probe.actions.lock().unwrap().push(chosen);
        let steps = probe.steps.fetch_add(1, Ordering::SeqCst) + 1;
        let mut info = probe.info.lock().unwrap();
        *info = CString::new(format!("{{\"steps\": {steps}}}")).unwrap();
        unsafe {
            (*out).observation = eef(steps as f32);
            (*out).reward = 0.5;
            (*out).terminated = steps % 2 == 0;
            (*out).info_json = info.as_ptr();
        }
        0
    }

    unsafe extern "C" fn render(_user_data: *mut c_void, out: *mut *mut RlmeshValue) -> c_int {
        let tensor =
            Tensor::from_vec(vec![200; 2 * 2 * 3], vec![2, 2, 3], DType::Uint8).expect("tensor");
        unsafe { *out = into_handle(SpaceValue::Box(tensor)) };
        0
    }

    unsafe extern "C" fn close(user_data: *mut c_void) {
        let probe = unsafe { &*user_data.cast::<Probe>() };
        probe.closes.fetch_add(1, Ordering::SeqCst);
    }

    fn vtable() -> RlmeshEnvVtable {
        RlmeshEnvVtable {
            struct_size: std::mem::size_of::<RlmeshEnvVtable>(),
            reset: Some(reset),
            step: Some(step),
            render: Some(render),
            close: Some(close),
        }
    }

    /// Owned obs (Dict{eef_pos: f32[3]}) and action (f32[1] in [-1, 1]) spaces.
    fn spaces() -> (*mut RlmeshSpaceSpec, *mut RlmeshSpaceSpec) {
        let shape = [3i64];
        let eef =
            unsafe { rlmesh_space_box(F32, shape.as_ptr(), 1, f64::NEG_INFINITY, f64::INFINITY) };
        let key = CString::new("eef_pos").unwrap();
        let keys = [key.as_ptr()];
        let children = [eef];
        let obs = unsafe { rlmesh_space_dict(keys.as_ptr(), children.as_ptr(), 1) };
        assert!(!obs.is_null());
        let one = [1i64];
        (obs, unsafe {
            rlmesh_space_box(F32, one.as_ptr(), 1, -1.0, 1.0)
        })
    }

    fn new_env(tags: Option<&str>, probe: &Probe) -> Result<*mut RlmeshEnv, String> {
        let (obs, act) = spaces();
        let id = CString::new("CapiEnv-test").unwrap();
        let tags = tags.map(|tags| CString::new(tags).unwrap());
        let trial = CString::new("trial_index").unwrap();
        let options = [trial.as_ptr()];
        let mode = CString::new("rgb_array").unwrap();
        let config = RlmeshEnvConfig {
            struct_size: std::mem::size_of::<RlmeshEnvConfig>(),
            id: id.as_ptr(),
            observation_space: obs,
            action_space: act,
            adapter_tags_json: tags.as_ref().map_or(std::ptr::null(), |t| t.as_ptr()),
            reset_options: options.as_ptr(),
            num_reset_options: 1,
            render_mode: mode.as_ptr(),
            metadata_json: std::ptr::null(),
        };
        let mut env = std::ptr::null_mut();
        let status = unsafe {
            rlmesh_env_new(
                &vtable(),
                &config,
                std::ptr::from_ref(probe).cast_mut().cast(),
                &mut env,
            )
        };
        unsafe {
            rlmesh_space_free(obs);
            rlmesh_space_free(act);
        }
        if status == RlmeshStatus::Ok {
            Ok(env)
        } else {
            Err(last_error_message())
        }
    }

    fn bind(env: *mut RlmeshEnv) -> String {
        let address = CString::new("127.0.0.1:0").unwrap();
        let mut out = RlmeshBytes::from_vec(Vec::new());
        let status = unsafe { rlmesh_env_bind(env, address.as_ptr(), std::ptr::null(), &mut out) };
        assert_eq!(status, RlmeshStatus::Ok, "{}", last_error_message());
        String::from_utf8(unsafe { out.into_vec() }).expect("utf-8 address")
    }

    /// The env pointer, movable to the serve thread (the handle is Sync by contract:
    /// serve and cancel may overlap).
    struct SendEnv(*mut RlmeshEnv);
    unsafe impl Send for SendEnv {}

    #[test]
    fn serves_a_c_env_end_to_end() {
        let probe = Probe::default();
        let env = new_env(Some(TAGS), &probe).expect("env");
        let address = bind(env);
        let served = SendEnv(env);
        let server = std::thread::spawn(move || {
            let served = served;
            unsafe { rlmesh_env_serve(served.0) }
        });

        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut client = RemoteEnv::connect(&address).await.expect("connect");
            let contract = client.env_contract().clone();
            assert_eq!(contract.id, "CapiEnv-test");
            assert_eq!(contract.render_mode, "rgb_array");
            let metadata = contract.metadata.expect("metadata");
            let Some(MetaValue::Map(tags)) = metadata.get(ENV_METADATA_KEY) else {
                panic!("tags not published as a map: {metadata:?}");
            };
            assert!(tags.contains_key("observation"));
            assert_eq!(
                metadata.get(ENV_RESET_OPTIONS_KEY),
                Some(&MetaValue::List(vec![MetaValue::String(
                    "trial_index".into()
                )]))
            );

            let options: MetaMap = [(TRIAL_INDEX_OPTION.to_string(), MetaValue::Int(4))].into();
            let reset = client
                .reset(ResetRequest {
                    seed: Some(11),
                    options: Some(options),
                    timeout_ms: 0,
                })
                .await
                .expect("reset");
            assert!(reset.observation.is_some());

            let step = client
                .step(StepRequest {
                    action: Some(action(0.25)),
                    timeout_ms: 0,
                })
                .await
                .expect("step");
            assert_eq!(step.reward, 0.5);
            assert!(!step.terminated);
            assert_eq!(
                step.info.and_then(|info| info.get("steps").cloned()),
                Some(MetaValue::Int(1))
            );

            let frame = client
                .render(RenderRequest::default())
                .await
                .expect("render")
                .frame
                .expect("a frame");
            assert_eq!(&frame.frame[..8], b"\x89PNG\r\n\x1a\n");
        });

        unsafe { rlmesh_env_cancel(env) };
        let status = server.join().expect("serve thread");
        assert_eq!(status, RlmeshStatus::Ok, "{}", last_error_message());
        assert_eq!(probe.closes.load(Ordering::SeqCst), 1, "close runs once");
        assert_eq!(*probe.resets.lock().unwrap(), vec![(Some(11), Some(4))]);
        assert_eq!(*probe.actions.lock().unwrap(), vec![0.25]);
        unsafe { rlmesh_env_free(env) };
    }

    #[test]
    fn a_failing_callback_fails_the_request_not_the_server() {
        let probe = Probe::default();
        let env = new_env(None, &probe).expect("env");
        let address = bind(env);
        let served = SendEnv(env);
        let server = std::thread::spawn(move || {
            let served = served;
            unsafe { rlmesh_env_serve(served.0) }
        });
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut client = RemoteEnv::connect(&address).await.expect("connect");
            client.reset(ResetRequest::default()).await.expect("reset");
            // The C step declines a negative action with a message.
            let err = client
                .step(StepRequest {
                    action: Some(action(-0.5)),
                    timeout_ms: 0,
                })
                .await
                .expect_err("declined step");
            assert!(
                err.to_string().contains("negative action declined"),
                "{err}"
            );
            // The core ends that session; the env keeps serving new ones.
            let mut client = RemoteEnv::connect(&address).await.expect("reconnect");
            client
                .reset(ResetRequest::default())
                .await
                .expect("reset on a new session");
            client
                .step(StepRequest {
                    action: Some(action(0.5)),
                    timeout_ms: 0,
                })
                .await
                .expect("step on a new session");
        });
        unsafe { rlmesh_env_cancel(env) };
        assert_eq!(server.join().unwrap(), RlmeshStatus::Ok);
        unsafe { rlmesh_env_free(env) };
    }

    #[test]
    fn tags_that_do_not_fit_the_spaces_fail_at_new() {
        let probe = Probe::default();
        // `camera` is not an observation key.
        let bad = r#"{"observation": {"camera": {"type": "image", "role": "image/primary"}},
            "action": {"components": [{"role": "action/gripper", "dim": 1}]}}"#;
        let err = new_env(Some(bad), &probe).expect_err("rejected");
        assert!(err.contains("adapter tags"), "{err}");
        // An unknown field is a publish-time error too.
        let typo = r#"{"observation": {"eef_pos": {"type": "state", "role": "proprio/eef_pos", "rnage": [0, 1]}},
            "action": {"components": [{"role": "action/gripper", "dim": 1}]}}"#;
        assert!(new_env(Some(typo), &probe).is_err());
    }

    #[test]
    fn vtable_without_step_is_rejected() {
        let mut table = vtable();
        table.step = None;
        let (obs, act) = spaces();
        let config = RlmeshEnvConfig {
            struct_size: std::mem::size_of::<RlmeshEnvConfig>(),
            id: std::ptr::null(),
            observation_space: obs,
            action_space: act,
            adapter_tags_json: std::ptr::null(),
            reset_options: std::ptr::null(),
            num_reset_options: 0,
            render_mode: std::ptr::null(),
            metadata_json: std::ptr::null(),
        };
        let mut env = std::ptr::null_mut();
        let status = unsafe { rlmesh_env_new(&table, &config, std::ptr::null_mut(), &mut env) };
        assert_eq!(status, RlmeshStatus::InvalidArgument);
        assert!(env.is_null());
        unsafe {
            rlmesh_space_free(obs);
            rlmesh_space_free(act);
        }
    }

    #[test]
    fn png_encoding_rejects_non_images() {
        let tensor = Tensor::from_vec(vec![0; 4 * 4 * 4], vec![4, 4], DType::Float32).unwrap();
        assert!(encode_png(&SpaceValue::Box(tensor)).is_err());
        assert!(encode_png(&SpaceValue::Discrete(1)).is_err());
        let gray = Tensor::from_vec(vec![9; 3 * 5], vec![3, 5], DType::Uint8).unwrap();
        assert!(encode_png(&SpaceValue::Box(gray)).is_ok());
    }

    #[test]
    fn integer_box_bounds_round_trip_through_the_builders() {
        let shape = [8i64, 8, 3];
        let image = unsafe { rlmesh_space_box(U8, shape.as_ptr(), 3, 0.0, 255.0) };
        assert!(!image.is_null(), "{}", last_error_message());
        let (mut low, mut high) = (0.0, 0.0);
        let status =
            unsafe { crate::spaces::rlmesh_space_box_bounds(image, 5, &mut low, &mut high) };
        assert_eq!(status, RlmeshStatus::Ok);
        assert_eq!((low, high), (0.0, 255.0));
        unsafe { rlmesh_space_free(image) };
        // An integer dtype has no infinite bound.
        let unbounded = unsafe { rlmesh_space_box(U8, shape.as_ptr(), 3, 0.0, f64::INFINITY) };
        assert!(unbounded.is_null());
    }
}
