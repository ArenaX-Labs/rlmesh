//! Additional native utilities and interoperability features.
//!
//! Nothing here is inherent on a shared type: these are extension traits and
//! free items, keeping the shared types focused on their core operations.

//! Native endpoint helpers are available through this module.
//!
//! ```compile_fail
//! use rlmesh::BindAddress;
//! ```
//! ```compile_fail
//! use rlmesh::ConnectAddress;
//! ```
//! ```compile_fail
//! use rlmesh::ENV_ENDPOINT_TOKEN_ENV;
//! ```
//! ```compile_fail
//! use rlmesh::MODEL_ENDPOINT_TOKEN_ENV;
//! ```
//! ```compile_fail
//! use rlmesh::serve_options::ENV_ENDPOINT_TOKEN_ENV;
//! ```
//! ```compile_fail
//! use rlmesh::serve_options::MODEL_ENDPOINT_TOKEN_ENV;
//! ```
//! On Unix, stale-socket cleanup is also available here.
//! ```compile_fail
//! use rlmesh::remove_stale_socket;
//! ```
//!
use crate::spaces::Space;
use crate::tensor::{dims_i64, tensor_error};
use crate::value::space_codec;
use crate::{DType, Error, Result, Tensor, Value};

/// Raw DLPack interoperability and owned exports.
pub mod dlpack {
    pub use crate::dlpack::{
        DLDataType, DLDevice, DLManagedTensorVersioned, DLPackType, DLPackVersion, DLTensor,
        DlpackTensor, dlpack_type, dtype_from_dlpack,
    };
}

/// Reference-counted byte storage shared by tensors.
pub use rlmesh_spaces::Storage;

/// The size of one element of `dtype`, in bytes.
pub fn dtype_size(dtype: DType) -> usize {
    rlmesh_spaces::dtype_size(dtype.to_core())
}

/// Additional tensor constructors and layout access.
pub trait TensorExt: Sized {
    /// Adopt `data` as a C-contiguous tensor without copying.
    fn from_vec(data: Vec<u8>, shape: &[usize], dtype: DType) -> Result<Self>;
    /// A zero-filled tensor in fresh aligned storage.
    fn zeros(shape: &[usize], dtype: DType) -> Result<Self>;
    /// A view over `storage` with element strides (`None`: C-contiguous) and
    /// a byte offset.
    fn from_storage(
        storage: Storage,
        dtype: DType,
        shape: &[usize],
        element_strides: Option<&[usize]>,
        byte_offset: usize,
    ) -> Result<Self>;
    /// Stack same-shaped tensors along a new leading dimension.
    fn stack(tensors: &[Self]) -> Result<Self>;
    /// Split along the leading dimension.
    fn unstack(&self) -> Result<Vec<Self>>;
    /// The shared backing storage.
    fn storage(&self) -> &Storage;
    /// The first element's offset into the storage, in bytes.
    fn byte_offset(&self) -> usize;
    /// Strides in elements.
    fn element_strides(&self) -> Vec<usize>;
}

impl TensorExt for Tensor {
    fn from_vec(data: Vec<u8>, shape: &[usize], dtype: DType) -> Result<Self> {
        rlmesh_spaces::Tensor::from_vec(data, dims_i64(shape)?, dtype.to_core())
            .map(Tensor::wrap)
            .map_err(tensor_error)
    }

    fn zeros(shape: &[usize], dtype: DType) -> Result<Self> {
        rlmesh_spaces::Tensor::zeros(&dims_i64(shape)?, dtype.to_core())
            .map(Tensor::wrap)
            .map_err(tensor_error)
    }

    fn from_storage(
        storage: Storage,
        dtype: DType,
        shape: &[usize],
        element_strides: Option<&[usize]>,
        byte_offset: usize,
    ) -> Result<Self> {
        let strides = element_strides.map(dims_i64).transpose()?;
        rlmesh_spaces::Tensor::from_storage(
            storage,
            dtype.to_core(),
            dims_i64(shape)?,
            strides,
            byte_offset,
        )
        .map(Tensor::wrap)
        .map_err(tensor_error)
    }

    fn stack(tensors: &[Self]) -> Result<Self> {
        let core: Vec<_> = tensors.iter().map(|t| t.core().clone()).collect();
        rlmesh_spaces::Tensor::stack(&core)
            .map(Tensor::wrap)
            .map_err(tensor_error)
    }

    fn unstack(&self) -> Result<Vec<Self>> {
        self.core()
            .unstack()
            .map(|parts| parts.into_iter().map(Tensor::wrap).collect())
            .map_err(tensor_error)
    }

    fn storage(&self) -> &Storage {
        self.core().storage()
    }

    fn byte_offset(&self) -> usize {
        self.core().byte_offset()
    }

    fn element_strides(&self) -> Vec<usize> {
        self.core()
            .effective_strides()
            .iter()
            .map(|&stride| stride as usize)
            .collect()
    }
}

/// Why a value is not a member of a space.
pub trait SpaceExt {
    /// `Ok(())` exactly when `contains(value)` is true; otherwise the reason,
    /// as the ValueError class.
    fn contains_detail(&self, value: &Value) -> Result<()>;
}

impl SpaceExt for Space {
    fn contains_detail(&self, value: &Value) -> Result<()> {
        if space_codec::space_contains(self.spec().wire(), value) {
            return Ok(());
        }
        let encoded = space_codec::value_to_space(self.spec().wire(), value)
            .map_err(|err| Error::invalid_argument(err.to_string()))?;
        rlmesh_spaces::contains(self.spec().wire(), &encoded)
            .map_err(|err| Error::invalid_argument(err.to_string()))?;
        Err(Error::invalid_argument(format!(
            "value shape does not match the space shape {:?}",
            self.shape()
        )))
    }
}

/// The Rust- and C-only fields of an env server's options, reached through
/// [`EnvServerOptions::ext`](crate::EnvServerOptions::ext).
#[derive(Clone, Debug, Default)]
pub struct EnvServerOptions {
    /// Bearer token every request must carry. `RLMESH_ENV_ENDPOINT_TOKEN`,
    /// when set, replaces it. `None` or empty disables authentication.
    pub token: Option<String>,
    /// Advisory entries this endpoint adds to its handshake `PeerInfo.extra`.
    pub peer_info_extra: std::collections::BTreeMap<String, String>,
    /// Publish the served env's describe envelope on the handshake, as a
    /// factory serve does.
    pub publish_describe: bool,
}

/// The Rust- and C-only fields of a model serve, reached through
/// [`ModelServeOptions::ext`](crate::ModelServeOptions::ext).
#[derive(Clone, Debug, Default)]
pub struct ModelServeOptions {
    /// Bearer token every model RPC must carry. `RLMESH_MODEL_ENDPOINT_TOKEN`,
    /// when set, replaces it. `None` or empty disables authentication.
    pub token: Option<String>,
    /// How many Join-stream requests the served model processes at once per
    /// connection; `None` is the build's default cap. `Some(0)` is the
    /// ValueError class. Corner calls stay serialized either way.
    pub predict_concurrency: Option<usize>,
}

/// `EnvTags` from its wire JSON (the form
/// [`EnvTags::to_dict`](crate::adapters::EnvTags::to_dict) writes), checked
/// for unknown fields. A convenience alias for
/// [`EnvTags::from_json`](crate::adapters::EnvTags::from_json).
pub fn env_tags_from_json(json: &str) -> Result<crate::adapters::EnvTags> {
    crate::adapters::EnvTags::from_json(json)
}

/// `ModelSpec` from its wire JSON (the form
/// [`ModelSpec::to_dict`](crate::adapters::ModelSpec::to_dict) writes),
/// checked for unknown fields. A convenience alias for
/// [`ModelSpec::from_json`](crate::adapters::ModelSpec::from_json).
pub fn model_spec_from_json(json: &str) -> Result<crate::adapters::ModelSpec> {
    crate::adapters::ModelSpec::from_json(json)
}

/// Partial reset: reset the listed lanes (an empty slice resets all of them;
/// a per-lane seed list aligns with `env_indices`) and return the batched
/// observation of every lane and the info with every lane's `episode_ids`.
/// An index above `u32::MAX` is the ValueError class before any I/O.
pub fn reset_lanes(
    env: &mut crate::RemoteVectorEnv,
    env_indices: &[u64],
    seed: Option<crate::VectorSeed>,
    options: Option<&crate::InfoDict>,
) -> Result<(Value, crate::InfoDict)> {
    crate::run::drive::refuse_in_async_context("reset_lanes")?;
    crate::run::drive::block_on_library(crate::env::client::reset_lanes_in(
        env,
        env_indices,
        seed,
        options,
    ))
}

/// The async twin of [`reset_lanes`].
pub async fn reset_lanes_async(
    env: &mut crate::RemoteVectorEnv,
    env_indices: &[u64],
    seed: Option<crate::VectorSeed>,
    options: Option<&crate::InfoDict>,
) -> Result<(Value, crate::InfoDict)> {
    crate::run::drive::in_library(crate::env::client::reset_lanes_in(
        env,
        env_indices,
        seed,
        options,
    ))
    .await
}

/// Cancellation reported as `KeyboardInterrupt`. Every clone is the same
/// token, so [`cancel`] on any copy fires it. One `cancel()` fails the call
/// in flight on the token or, when none is, the next call made with it, at
/// its start; the token then re-arms. Repeated `cancel()` before that call
/// fails counts once.
///
/// [`cancel`]: CancelToken::cancel
#[derive(Clone, Debug, Default)]
pub struct CancelToken(std::sync::Arc<std::sync::Mutex<CancelState>>);

#[derive(Debug, Default)]
struct CancelState {
    generation: u64,
    subscribers: usize,
    pending: bool,
}

pub(crate) struct CancelSubscription {
    token: CancelToken,
    generation: u64,
    pending: bool,
    active: std::sync::atomic::AtomicBool,
}

impl CancelToken {
    /// A token that has not fired (equal to `default()`).
    pub fn new() -> Self {
        Self::default()
    }

    /// Fire every call currently using the token, or the next call when idle.
    pub fn cancel(&self) {
        let mut state = self.0.lock().unwrap_or_else(|err| err.into_inner());
        state.generation = state.generation.wrapping_add(1);
        if state.subscribers == 0 {
            state.pending = true;
        }
    }

    pub(crate) fn subscribe(&self) -> CancelSubscription {
        let mut state = self.0.lock().unwrap_or_else(|err| err.into_inner());
        state.subscribers += 1;
        CancelSubscription {
            token: self.clone(),
            generation: state.generation,
            pending: std::mem::take(&mut state.pending),
            active: std::sync::atomic::AtomicBool::new(true),
        }
    }

    /// Consume one idle firing at a synchronous admission boundary.
    pub(crate) fn take(&self) -> bool {
        let mut state = self.0.lock().unwrap_or_else(|err| err.into_inner());
        std::mem::take(&mut state.pending)
    }
}

impl CancelSubscription {
    pub(crate) fn fired(&self) -> bool {
        self.active.load(std::sync::atomic::Ordering::SeqCst)
            && (self.pending
                || self
                    .token
                    .0
                    .lock()
                    .unwrap_or_else(|err| err.into_inner())
                    .generation
                    != self.generation)
    }

    /// Observe the final generation and detach under the same token lock.
    /// A firing after this boundary is idle (or reaches other subscribers).
    pub(crate) fn finish(&self) -> bool {
        let mut state = self.token.0.lock().unwrap_or_else(|err| err.into_inner());
        if self.active.swap(false, std::sync::atomic::Ordering::SeqCst) {
            state.subscribers -= 1;
            self.pending || state.generation != self.generation
        } else {
            false
        }
    }

    pub(crate) fn unsubscribe(&self) {
        self.finish();
    }
}

impl Drop for CancelSubscription {
    fn drop(&mut self) {
        self.unsubscribe();
    }
}

/// Register a trusted custom-input entrypoint for subsequent adapter resolves.
/// Re-registering replaces it; existing routes retain their transform.
pub fn register_entrypoint(name: &str, transform: crate::adapters::ObsTransform) -> Result<()> {
    crate::adapters::register(name, transform)
}

/// Remove a trusted entrypoint when its transform has `expected`'s construction
/// identity, or remove any registration when `expected` is `None`. Resolved
/// routes keep their clones. Empty names are refused; unknown names return false.
/// The removed transform is released outside the registry lock.
pub fn unregister_entrypoint(
    name: &str,
    expected: Option<&crate::adapters::ObsTransform>,
) -> Result<bool> {
    crate::adapters::unregister(name, expected)
}

#[cfg(unix)]
pub use crate::address::remove_stale_socket;
/// Native endpoint parsing and stale Unix socket cleanup.
pub use crate::address::{BindAddress, ConnectAddress};
/// Environment variable names for native endpoint authentication.
pub const ENV_ENDPOINT_TOKEN_ENV: &str = "RLMESH_ENV_ENDPOINT_TOKEN";
/// Environment variable name for model endpoint authentication.
pub const MODEL_ENDPOINT_TOKEN_ENV: &str = "RLMESH_MODEL_ENDPOINT_TOKEN";

#[cfg(test)]
mod cancellation_regressions {
    use super::*;

    #[test]
    fn completion_observes_a_firing_without_polling_and_detaches_atomically() {
        let token = CancelToken::new();
        let watch = crate::run::drive::CancelWatch::start(
            vec![token.clone()],
            tokio_util::sync::CancellationToken::new(),
        )
        .unwrap();
        token.cancel();
        assert_eq!(watch.finish(Ok(42)), Err(crate::Error::Cancelled));
        token.cancel();
        assert!(token.take());
        assert!(!token.take());
    }

    #[test]
    fn admission_subscription_retains_a_firing_before_watcher_creation() {
        let token = CancelToken::new();
        let subscription = token.subscribe();
        token.cancel();
        let result = crate::run::drive::CancelWatch::from_subscriptions(
            vec![subscription],
            tokio_util::sync::CancellationToken::new(),
        );
        assert!(matches!(result, Err(crate::Error::Cancelled)));
        assert_eq!(token.0.lock().unwrap().subscribers, 0);
        // A constructor that returns early must release even its fired input.
        token.cancel();
        assert!(token.take());
        assert!(!token.take());
    }

    #[test]
    fn idle_cancel_after_a_finished_call_is_pending_even_before_its_watcher_exits() {
        let token = CancelToken::new();
        let watch = crate::run::drive::CancelWatch::start(
            vec![token.clone()],
            tokio_util::sync::CancellationToken::new(),
        )
        .unwrap();
        drop(watch);
        token.cancel();
        assert!(token.take());
    }
}
