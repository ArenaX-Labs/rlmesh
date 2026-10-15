//! Rust-only extras beside the shared surface (no Python counterpart).
//!
//! Nothing here is inherent on a shared type: these are extension traits and
//! free items, so the shared types keep exactly Python's members.

use crate::spaces::Space;
use crate::tensor::{dims_i64, tensor_error};
use crate::value::space_codec;
use crate::{DType, Error, Result, Tensor, Value};

/// Reference-counted byte storage shared by tensors.
pub use rlmesh_spaces::Storage;

/// The size of one element of `dtype`, in bytes.
pub fn dtype_size(dtype: DType) -> usize {
    rlmesh_spaces::dtype_size(dtype.to_core())
}

/// Tensor constructors and layout access beyond Python's members.
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

/// The Rust- and C-only fields of an env server's options (ENV-12), reached
/// through [`EnvServerOptions::ext`](crate::EnvServerOptions::ext).
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

/// The Rust- and C-only fields of a model serve (MDL-20), reached through
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

/// Python's `EnvTags` from its wire JSON (the form `EnvTags.to_dict()`
/// writes), checked for unknown fields: the way Rust builds tags until the
/// adapters surface lands.
pub fn env_tags_from_json(json: &str) -> Result<crate::adapters::EnvTags> {
    crate::adapters::EnvTags::from_json(json)
}

/// Python's `ModelSpec` from its wire JSON (the form `ModelSpec.to_dict()`
/// writes), checked for unknown fields: the way Rust builds a spec until the
/// adapters surface lands.
pub fn model_spec_from_json(json: &str) -> Result<crate::adapters::ModelSpec> {
    crate::adapters::ModelSpec::from_json(json)
}

/// A [`RemoteEnv`](crate::RemoteEnv) that sends `token` on every request
/// (RMT-5). `""` sends none; a token a header cannot carry is the ValueError
/// class before any dial. Otherwise as [`RemoteEnv::new`](crate::RemoteEnv::new).
pub fn remote_env_with_token(
    options: crate::RemoteEnvOptions,
    token: &str,
) -> Result<crate::RemoteEnv> {
    crate::env::client::check_token(token)?;
    crate::run::drive::refuse_in_async_context("remote_env_with_token")?;
    crate::run::drive::block_on_library(crate::RemoteEnv::new_in(options, token))
}

/// The async twin of [`remote_env_with_token`].
pub async fn remote_env_with_token_async(
    options: crate::RemoteEnvOptions,
    token: &str,
) -> Result<crate::RemoteEnv> {
    crate::env::client::check_token(token)?;
    crate::run::drive::in_library(crate::RemoteEnv::new_in(options, token)).await
}

/// A [`RemoteVectorEnv`](crate::RemoteVectorEnv) that sends `token` on every
/// request (RMT-5).
pub fn remote_vector_env_with_token(
    options: crate::RemoteEnvOptions,
    token: &str,
) -> Result<crate::RemoteVectorEnv> {
    crate::env::client::check_token(token)?;
    crate::run::drive::refuse_in_async_context("remote_vector_env_with_token")?;
    crate::run::drive::block_on_library(crate::RemoteVectorEnv::new_in(options, token))
}

/// The async twin of [`remote_vector_env_with_token`].
pub async fn remote_vector_env_with_token_async(
    options: crate::RemoteEnvOptions,
    token: &str,
) -> Result<crate::RemoteVectorEnv> {
    crate::env::client::check_token(token)?;
    crate::run::drive::in_library(crate::RemoteVectorEnv::new_in(options, token)).await
}

/// Partial reset (RMT-14): reset the listed lanes (an empty slice resets
/// all of them; a per-lane seed list aligns with `env_indices`) and return
/// the batched observation of every lane and the info with every lane's
/// `episode_ids`. An index above `u32::MAX` is the ValueError class before
/// any I/O.
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

/// Cancellation (R-18): the native counterpart of Python's
/// `KeyboardInterrupt`. Every clone is the same token, so [`cancel`] on any
/// copy fires it. One `cancel()` fails the call in flight on the token or,
/// when none is, the next call made with it, at its start; the token then
/// re-arms. Repeated `cancel()` before that call fails counts once.
///
/// [`cancel`]: CancelToken::cancel
#[derive(Clone, Debug, Default)]
pub struct CancelToken(std::sync::Arc<std::sync::atomic::AtomicBool>);

impl CancelToken {
    /// A token that has not fired (equal to `default()`).
    pub fn new() -> Self {
        Self::default()
    }

    /// Fire the token, from any thread.
    pub fn cancel(&self) {
        self.0.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Consume a pending cancel: true once per firing (the re-arm).
    pub(crate) fn take(&self) -> bool {
        self.0.swap(false, std::sync::atomic::Ordering::SeqCst)
    }
}
