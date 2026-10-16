//! Immutable tensors with shared storage.

use std::fmt;

use rlmesh_spaces::{Tensor as CoreTensor, TensorError, dtype_size};

use crate::{DType, Device, Error, Result};

/// Options for [`Tensor::to_dlpack`].
#[derive(Clone, Debug, Default)]
pub struct ToDlpackOptions {
    /// Requested DLPack device; only CPU `(1, 0)` is supported.
    pub dl_device: Option<(i32, i32)>,
    /// Export a fresh writable copy instead of sharing read-only storage.
    pub copy: bool,
}

/// An immutable n-dimensional array with shared storage.
///
/// The constructor copies its buffer; [`Clone`], [`buffer`](Self::buffer)
/// and a contiguous [`reshape`](Self::reshape) share storage, and
/// [`copy`](Self::copy) makes a fresh one. Strides are in bytes.
///
/// `PartialEq` compares dtype, shape and element bytes.
#[derive(Clone)]
pub struct Tensor {
    inner: CoreTensor,
    shape: Vec<usize>,
}

impl Tensor {
    /// Copy `buffer` into a C-contiguous tensor of `shape` and `dtype`, as
    /// `Tensor(buffer, shape, dtype)`. `buffer` must hold exactly
    /// `size * itemsize` bytes.
    pub fn new(buffer: &[u8], shape: &[usize], dtype: DType) -> Result<Tensor> {
        let dims = dims_i64(shape)?;
        let inner = CoreTensor::from_slice(buffer, &dims, dtype.to_core()).map_err(tensor_error)?;
        Ok(Self::wrap(inner))
    }

    /// Wrap a core tensor whose dtype is specified (every core constructor
    /// refuses `Unspecified`) and whose dims are non-negative.
    pub(crate) fn wrap(inner: CoreTensor) -> Tensor {
        let shape = inner.shape().iter().map(|&dim| dim as usize).collect();
        Tensor { inner, shape }
    }

    pub(crate) fn core(&self) -> &CoreTensor {
        &self.inner
    }

    /// Dimension sizes, C order.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Element dtype.
    pub fn dtype(&self) -> DType {
        DType::from_core(self.inner.dtype())
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Number of elements.
    pub fn size(&self) -> usize {
        self.inner.numel()
    }

    /// Size of the elements in bytes (`size * itemsize`).
    pub fn nbytes(&self) -> usize {
        self.inner.nbytes()
    }

    /// Strides in bytes, one per dimension.
    pub fn strides(&self) -> Vec<usize> {
        let itemsize = dtype_size(self.inner.dtype());
        self.inner
            .effective_strides()
            .iter()
            .map(|&stride| stride as usize * itemsize)
            .collect()
    }

    /// The device the storage lives on.
    pub fn device(&self) -> Device {
        self.inner.device()
    }

    /// The bytes from the first element, read-only and shared: `nbytes` long
    /// when contiguous, else through the last byte the strides address.
    pub fn buffer(&self) -> &[u8] {
        let offset = self.inner.byte_offset();
        let storage = self.inner.storage().as_slice();
        let len = if self.inner.is_contiguous() || self.size() == 0 {
            self.nbytes()
        } else {
            let itemsize = dtype_size(self.inner.dtype());
            let last: usize = self
                .shape
                .iter()
                .zip(self.inner.effective_strides().iter())
                .map(|(&dim, &stride)| (dim - 1) * stride as usize)
                .sum();
            (last + 1) * itemsize
        };
        &storage[offset..offset + len]
    }

    /// Whether the elements are laid out C-contiguously.
    pub fn is_contiguous(&self) -> bool {
        self.inner.is_contiguous()
    }

    /// The same elements in a new shape; one dimension may be `-1`. Shares
    /// storage when contiguous.
    pub fn reshape(&self, shape: &[i64]) -> Result<Tensor> {
        self.inner
            .reshape(shape)
            .map(Self::wrap)
            .map_err(tensor_error)
    }

    /// A deep, contiguous copy.
    pub fn copy(&self) -> Tensor {
        let bytes = self.inner.to_contiguous_bytes();
        let inner = CoreTensor::from_slice(&bytes, self.inner.shape(), self.inner.dtype())
            .expect("a tensor's own bytes fit its shape");
        Self::wrap(inner)
    }

    /// The elements as C-ordered bytes (a copy).
    pub fn tobytes(&self) -> Vec<u8> {
        self.inner.to_contiguous_bytes().into_owned()
    }
}

/// Text representation: `Tensor(dtype="float32", shape=[2, 3], bytes=24)`.
impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor(dtype={:?}, shape={:?}, bytes={})",
            self.dtype().as_str(),
            self.shape,
            self.nbytes()
        )
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

pub(crate) fn dims_i64(shape: &[usize]) -> Result<Vec<i64>> {
    shape
        .iter()
        .map(|&dim| {
            i64::try_from(dim).map_err(|_| Error::invalid_argument("tensor element count overflow"))
        })
        .collect()
}

pub(crate) fn tensor_error(err: TensorError) -> Error {
    match err {
        TensorError::Overflow => Error::invalid_argument("tensor element count overflow"),
        other => Error::invalid_argument(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_matches_python_members() {
        let bytes: Vec<u8> = (0u8..24).collect();
        let tensor = Tensor::new(&bytes, &[2, 3], DType::Float32).unwrap();
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.dtype(), DType::Float32);
        assert_eq!((tensor.ndim(), tensor.size(), tensor.nbytes()), (2, 6, 24));
        assert_eq!(tensor.strides(), vec![12, 4]);
        assert_eq!(tensor.device(), Device::Cpu);
        assert!(tensor.is_contiguous());
        assert_eq!(tensor.buffer(), bytes.as_slice());
        assert_eq!(tensor.tobytes(), bytes);
        assert_eq!(
            tensor.to_string(),
            r#"Tensor(dtype="float32", shape=[2, 3], bytes=24)"#
        );
        let flat = tensor.reshape(&[-1]).unwrap();
        assert_eq!(flat.shape(), &[6]);
        assert_eq!(flat.buffer().as_ptr(), tensor.buffer().as_ptr(), "shares");
        let copy = tensor.copy();
        assert_eq!(copy, tensor);
        assert_ne!(copy.buffer().as_ptr(), tensor.buffer().as_ptr());
    }

    #[test]
    fn tensor_refuses_bad_arguments_with_value_error() {
        let err = Tensor::new(&[0; 3], &[2], DType::Float32).unwrap_err();
        assert_eq!(err.class(), crate::ErrorClass::ValueError);
        let err = Tensor::new(&[], &[1 << 62, 1 << 62], DType::Float32).unwrap_err();
        assert_eq!(err.class(), crate::ErrorClass::ValueError);
        assert!(err.to_string().contains("overflow"), "{err}");
    }
}
