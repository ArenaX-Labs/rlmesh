//! Rust-only extras beside the shared surface (no Python counterpart).
//!
//! Nothing here is inherent on a shared type: these are extension traits and
//! free items, so the shared types keep exactly Python's members.

use crate::spaces::{Space, SpaceValue};
use crate::tensor::{dims_i64, tensor_error};
use crate::{DType, Error, Result, Tensor};

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
    fn contains_detail(&self, value: &SpaceValue) -> Result<()>;
}

impl SpaceExt for Space {
    fn contains_detail(&self, value: &SpaceValue) -> Result<()> {
        rlmesh_spaces::contains(self.spec().wire(), value)
            .map_err(|err| Error::invalid_argument(err.to_string()))
    }
}
