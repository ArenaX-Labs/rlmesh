mod dlpack;
#[cfg(test)]
mod proptests;
mod storage;

pub use dlpack::{DLPackType, dlpack_type, dtype_from_dlpack};
pub use storage::Storage;

use std::borrow::Cow;

use thiserror::Error;

use crate::dtype::{DType, dtype_size};

/// Device a tensor's storage lives on, mirroring DLPack's `DLDeviceType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
#[non_exhaustive]
pub enum Device {
    /// Host CPU memory (`kDLCPU`).
    Cpu = 1,
}

impl From<Device> for i32 {
    fn from(value: Device) -> Self {
        value as i32
    }
}

/// Errors raised by [`Tensor`] constructors and transformations.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    #[error("tensor dtype must be specified")]
    UnspecifiedDtype,
    #[error("negative dimension {0} in shape")]
    NegativeDim(i64),
    #[error("negative stride {0}")]
    NegativeStride(i64),
    #[error("strides rank {strides} does not match shape rank {shape}")]
    StrideRankMismatch { strides: usize, shape: usize },
    #[error("data is {actual} bytes but shape and dtype require exactly {expected}")]
    ByteLengthMismatch { expected: usize, actual: usize },
    #[error(
        "view requires {required} bytes at byte offset {byte_offset} but storage holds {available}"
    )]
    OutOfBounds {
        required: usize,
        byte_offset: usize,
        available: usize,
    },
    #[error("cannot reshape {from} elements into {to}")]
    NumelMismatch { from: usize, to: usize },
    #[error("reshape shape may contain at most one -1")]
    AmbiguousReshape,
    #[error("stack requires at least one tensor")]
    EmptyStack,
    #[error("stack requires uniform dtype and shape; tensor {index} differs")]
    StackMismatch { index: usize },
    #[error("cannot unstack a 0-dimensional tensor")]
    UnstackScalar,
    #[error("tensor size overflows usize")]
    Overflow,
}

/// C-contiguous (row-major) strides for `shape`, in element units.
pub fn contiguous_strides(shape: &[i64]) -> Vec<i64> {
    let mut strides = vec![0i64; shape.len()];
    let mut stride = 1i64;
    for (slot, dim) in strides.iter_mut().zip(shape).rev() {
        *slot = stride;
        stride *= *dim;
    }
    strides
}

/// An n-dimensional, immutable tensor backed by shared [`Storage`].
///
/// The layout follows DLPack conventions: C-order shape, element-unit
/// strides (`None` means C-contiguous), and a byte offset into the backing
/// storage. Constructors validate that every addressable element falls
/// inside the storage, so accessors never fail.
///
/// Equality is logical: two tensors are equal when dtype, shape, and
/// element bytes (in C order) match, regardless of how they are laid out
/// in storage.
#[derive(Debug, Clone)]
pub struct Tensor {
    storage: Storage,
    dtype: DType,
    shape: Vec<i64>,
    strides: Option<Vec<i64>>,
    byte_offset: usize,
}

impl Tensor {
    /// Adopt `data` as a C-contiguous tensor without copying.
    ///
    /// `data.len()` must equal exactly `numel * dtype_size`. The buffer keeps
    /// its original allocation, so no alignment is guaranteed.
    pub fn from_vec(data: Vec<u8>, shape: Vec<i64>, dtype: DType) -> Result<Self, TensorError> {
        let expected = checked_nbytes(&shape, dtype)?;
        if data.len() != expected {
            return Err(TensorError::ByteLengthMismatch {
                expected,
                actual: data.len(),
            });
        }
        Self::from_storage(Storage::from_vec(data), dtype, shape, None, 0)
    }

    /// Copy `data` into a fresh 64-byte-aligned C-contiguous tensor.
    ///
    /// `data.len()` must equal exactly `numel * dtype_size`.
    pub fn from_slice(data: &[u8], shape: &[i64], dtype: DType) -> Result<Self, TensorError> {
        let expected = checked_nbytes(shape, dtype)?;
        if data.len() != expected {
            return Err(TensorError::ByteLengthMismatch {
                expected,
                actual: data.len(),
            });
        }
        Self::from_storage(Storage::from_slice(data), dtype, shape.to_vec(), None, 0)
    }

    /// A zero-filled, 64-byte-aligned C-contiguous tensor.
    pub fn zeros(shape: &[i64], dtype: DType) -> Result<Self, TensorError> {
        let nbytes = checked_nbytes(shape, dtype)?;
        Self::from_storage(Storage::zeroed(nbytes), dtype, shape.to_vec(), None, 0)
    }

    /// A tensor view over `storage` with full layout control.
    ///
    /// `strides` are in element units; `None` means C-contiguous. Dims and
    /// strides must be non-negative and every addressable element must fall
    /// inside the storage.
    pub fn from_storage(
        storage: Storage,
        dtype: DType,
        shape: Vec<i64>,
        strides: Option<Vec<i64>>,
        byte_offset: usize,
    ) -> Result<Self, TensorError> {
        if dtype == DType::Unspecified {
            return Err(TensorError::UnspecifiedDtype);
        }
        if let Some(strides) = &strides {
            if strides.len() != shape.len() {
                return Err(TensorError::StrideRankMismatch {
                    strides: strides.len(),
                    shape: shape.len(),
                });
            }
            for &stride in strides {
                if stride < 0 {
                    return Err(TensorError::NegativeStride(stride));
                }
            }
        }
        let required = required_bytes(&shape, strides.as_deref(), dtype)?;
        let available = storage.len().saturating_sub(byte_offset);
        if required > available {
            return Err(TensorError::OutOfBounds {
                required,
                byte_offset,
                available: storage.len(),
            });
        }
        Ok(Self {
            storage,
            dtype,
            shape,
            strides,
            byte_offset,
        })
    }

    /// Element data type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Dimension sizes in C order.
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    /// Explicit element-unit strides, or `None` when C-contiguous.
    pub fn strides(&self) -> Option<&[i64]> {
        self.strides.as_deref()
    }

    /// Element-unit strides, materializing the C-contiguous default.
    pub fn effective_strides(&self) -> Cow<'_, [i64]> {
        match &self.strides {
            Some(strides) => Cow::Borrowed(strides),
            None => Cow::Owned(contiguous_strides(&self.shape)),
        }
    }

    /// Offset in bytes from the start of the storage to the first element.
    pub fn byte_offset(&self) -> usize {
        self.byte_offset
    }

    /// Device the storage lives on. Always [`Device::Cpu`] today.
    pub fn device(&self) -> Device {
        Device::Cpu
    }

    /// The shared backing storage.
    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().map(|&dim| dim as usize).product()
    }

    /// Logical size of the element data in bytes (`numel * dtype_size`).
    pub fn nbytes(&self) -> usize {
        self.numel() * dtype_size(self.dtype)
    }

    /// Whether elements are laid out C-contiguously.
    pub fn is_contiguous(&self) -> bool {
        let Some(strides) = &self.strides else {
            return true;
        };
        let mut expected = 1i64;
        for (&dim, &stride) in self.shape.iter().zip(strides).rev() {
            if dim == 0 {
                return true;
            }
            if dim != 1 {
                if stride != expected {
                    return false;
                }
                expected *= dim;
            }
        }
        true
    }

    /// A tensor with the same elements and a new shape.
    ///
    /// At most one dimension may be `-1`, which is inferred from the
    /// element count. Returns a zero-copy view sharing this tensor's
    /// storage when the layout is contiguous, and a contiguous copy
    /// otherwise.
    pub fn reshape(&self, shape: &[i64]) -> Result<Self, TensorError> {
        let shape = self.resolve_reshape_dims(shape)?;
        let to = checked_numel(&shape)?;
        let from = self.numel();
        if from != to {
            return Err(TensorError::NumelMismatch { from, to });
        }
        if self.is_contiguous() {
            Self::from_storage(
                self.storage.clone(),
                self.dtype,
                shape,
                None,
                self.byte_offset,
            )
        } else {
            let storage = Storage::aligned_with(self.nbytes(), |buf| self.gather_into(buf));
            Self::from_storage(storage, self.dtype, shape, None, 0)
        }
    }

    /// Replace a single `-1` dimension with the size inferred from this
    /// tensor's element count.
    fn resolve_reshape_dims(&self, shape: &[i64]) -> Result<Vec<i64>, TensorError> {
        let wildcards = shape.iter().filter(|&&dim| dim == -1).count();
        if wildcards > 1 {
            return Err(TensorError::AmbiguousReshape);
        }
        if wildcards == 0 {
            return Ok(shape.to_vec());
        }
        let mut known = 1usize;
        for &dim in shape {
            if dim < -1 {
                return Err(TensorError::NegativeDim(dim));
            }
            if dim >= 0 {
                known = known
                    .checked_mul(dim as usize)
                    .ok_or(TensorError::Overflow)?;
            }
        }
        let from = self.numel();
        if known == 0 || !from.is_multiple_of(known) {
            return Err(TensorError::NumelMismatch { from, to: known });
        }
        let inferred = (from / known) as i64;
        Ok(shape
            .iter()
            .map(|&dim| if dim == -1 { inferred } else { dim })
            .collect())
    }

    /// The element bytes in C order.
    ///
    /// Borrows from storage when the layout is contiguous; gathers into a
    /// fresh buffer otherwise.
    pub fn to_contiguous_bytes(&self) -> Cow<'_, [u8]> {
        if self.is_contiguous() {
            let start = self.byte_offset;
            return Cow::Borrowed(&self.storage.as_slice()[start..start + self.nbytes()]);
        }
        let mut out = Vec::with_capacity(self.nbytes());
        self.gather_into(&mut out);
        Cow::Owned(out)
    }

    /// Stack tensors of identical dtype and shape along a new leading axis.
    ///
    /// The result is a fresh 64-byte-aligned contiguous tensor of shape
    /// `[tensors.len(), ..shape]`.
    pub fn stack(tensors: &[Tensor]) -> Result<Tensor, TensorError> {
        let Some(first) = tensors.first() else {
            return Err(TensorError::EmptyStack);
        };
        for (index, tensor) in tensors.iter().enumerate() {
            if tensor.dtype != first.dtype || tensor.shape != first.shape {
                return Err(TensorError::StackMismatch { index });
            }
        }
        let total = first
            .nbytes()
            .checked_mul(tensors.len())
            .ok_or(TensorError::Overflow)?;
        let mut shape = Vec::with_capacity(first.shape.len() + 1);
        shape.push(tensors.len() as i64);
        shape.extend_from_slice(&first.shape);
        let storage = Storage::aligned_with(total, |buf| {
            for tensor in tensors {
                match tensor.to_contiguous_bytes() {
                    Cow::Borrowed(bytes) => buf.extend_from_slice(bytes),
                    Cow::Owned(bytes) => buf.extend_from_slice(&bytes),
                }
            }
        });
        Self::from_storage(storage, first.dtype, shape, None, 0)
    }

    /// Split along axis 0 into zero-copy views sharing this storage.
    pub fn unstack(&self) -> Result<Vec<Tensor>, TensorError> {
        if self.shape.is_empty() {
            return Err(TensorError::UnstackScalar);
        }
        let count = self.shape[0] as usize;
        let inner_shape = &self.shape[1..];
        let strides = self.effective_strides();
        let outer_stride_bytes = strides[0] as usize * dtype_size(self.dtype);
        let inner_strides = self.strides.as_ref().map(|_| strides[1..].to_vec());
        (0..count)
            .map(|index| {
                Self::from_storage(
                    self.storage.clone(),
                    self.dtype,
                    inner_shape.to_vec(),
                    inner_strides.clone(),
                    self.byte_offset + index * outer_stride_bytes,
                )
            })
            .collect()
    }

    /// Copy the element bytes in C order into `out`.
    fn gather_into(&self, out: &mut Vec<u8>) {
        let itemsize = dtype_size(self.dtype);
        let strides = self.effective_strides();
        let data = &self.storage.as_slice()[self.byte_offset..];
        let mut index = vec![0usize; self.shape.len()];
        for _ in 0..self.numel() {
            let element: usize = index
                .iter()
                .zip(strides.iter())
                .map(|(&i, &stride)| i * stride as usize)
                .sum();
            let start = element * itemsize;
            out.extend_from_slice(&data[start..start + itemsize]);
            for axis in (0..index.len()).rev() {
                index[axis] += 1;
                if (index[axis] as i64) < self.shape[axis] {
                    break;
                }
                index[axis] = 0;
            }
        }
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.dtype == other.dtype
            && self.shape == other.shape
            && self.to_contiguous_bytes() == other.to_contiguous_bytes()
    }
}

fn checked_numel(shape: &[i64]) -> Result<usize, TensorError> {
    let mut numel = 1usize;
    for &dim in shape {
        if dim < 0 {
            return Err(TensorError::NegativeDim(dim));
        }
        numel = numel
            .checked_mul(dim as usize)
            .ok_or(TensorError::Overflow)?;
    }
    Ok(numel)
}

fn checked_nbytes(shape: &[i64], dtype: DType) -> Result<usize, TensorError> {
    if dtype == DType::Unspecified {
        return Err(TensorError::UnspecifiedDtype);
    }
    checked_numel(shape)?
        .checked_mul(dtype_size(dtype))
        .ok_or(TensorError::Overflow)
}

/// Bytes a view must be able to address past its byte offset: one item plus
/// the span reached by the largest in-bounds index on every axis.
fn required_bytes(
    shape: &[i64],
    strides: Option<&[i64]>,
    dtype: DType,
) -> Result<usize, TensorError> {
    let numel = checked_numel(shape)?;
    if numel == 0 {
        return Ok(0);
    }
    let itemsize = dtype_size(dtype);
    let Some(strides) = strides else {
        return numel.checked_mul(itemsize).ok_or(TensorError::Overflow);
    };
    let mut last_element = 0usize;
    for (&dim, &stride) in shape.iter().zip(strides) {
        let span = (dim as usize - 1)
            .checked_mul(stride as usize)
            .ok_or(TensorError::Overflow)?;
        last_element = last_element
            .checked_add(span)
            .ok_or(TensorError::Overflow)?;
    }
    last_element
        .checked_add(1)
        .ok_or(TensorError::Overflow)?
        .checked_mul(itemsize)
        .ok_or(TensorError::Overflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn test_from_vec_adopts_and_validates_length() {
        let tensor = Tensor::from_vec(f32_bytes(&[1.0, 2.0, 3.0]), vec![3], DType::Float32)
            .expect("valid tensor");
        assert_eq!(tensor.shape(), &[3]);
        assert_eq!(tensor.numel(), 3);
        assert_eq!(tensor.nbytes(), 12);
        assert!(tensor.is_contiguous());
        assert_eq!(tensor.strides(), None);
        assert_eq!(tensor.effective_strides().as_ref(), &[1]);
        assert_eq!(tensor.device(), Device::Cpu);

        assert_eq!(
            Tensor::from_vec(vec![0u8; 11], vec![3], DType::Float32),
            Err(TensorError::ByteLengthMismatch {
                expected: 12,
                actual: 11
            })
        );
    }

    #[test]
    fn test_constructor_rejects_invalid_inputs() {
        assert_eq!(
            Tensor::from_vec(vec![], vec![2], DType::Unspecified),
            Err(TensorError::UnspecifiedDtype)
        );
        assert_eq!(
            Tensor::from_vec(vec![], vec![-1], DType::Float32),
            Err(TensorError::NegativeDim(-1))
        );
        assert_eq!(
            Tensor::from_storage(
                Storage::zeroed(8),
                DType::Float32,
                vec![2],
                Some(vec![1, 1]),
                0
            ),
            Err(TensorError::StrideRankMismatch {
                strides: 2,
                shape: 1
            })
        );
        assert_eq!(
            Tensor::from_storage(
                Storage::zeroed(8),
                DType::Float32,
                vec![2],
                Some(vec![-1]),
                0
            ),
            Err(TensorError::NegativeStride(-1))
        );
        assert_eq!(
            Tensor::from_storage(Storage::zeroed(8), DType::Float32, vec![3], None, 0),
            Err(TensorError::OutOfBounds {
                required: 12,
                byte_offset: 0,
                available: 8
            })
        );
        // Strided view reaching past the storage end.
        assert_eq!(
            Tensor::from_storage(
                Storage::zeroed(12),
                DType::Float32,
                vec![2],
                Some(vec![3]),
                0
            ),
            Err(TensorError::OutOfBounds {
                required: 16,
                byte_offset: 0,
                available: 12
            })
        );
        assert_eq!(
            Tensor::from_vec(vec![], vec![i64::MAX, i64::MAX], DType::Float32),
            Err(TensorError::Overflow)
        );
    }

    #[test]
    fn test_zeros_is_aligned_and_zero_filled() {
        let tensor = Tensor::zeros(&[4, 4], DType::Int32).expect("valid tensor");
        assert_eq!(tensor.nbytes(), 64);
        assert!(tensor.to_contiguous_bytes().iter().all(|&b| b == 0));
        assert_eq!(tensor.storage().as_slice().as_ptr() as usize % 64, 0);
    }

    #[test]
    fn test_scalar_tensor() {
        let tensor =
            Tensor::from_slice(&1.0f64.to_le_bytes(), &[], DType::Float64).expect("valid tensor");
        assert_eq!(tensor.shape(), &[] as &[i64]);
        assert_eq!(tensor.numel(), 1);
        assert_eq!(tensor.to_contiguous_bytes().as_ref(), 1.0f64.to_le_bytes());
    }

    #[test]
    fn test_reshape_contiguous_is_view() {
        let tensor = Tensor::from_slice(
            &f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            &[2, 3],
            DType::Float32,
        )
        .expect("valid tensor");
        let reshaped = tensor.reshape(&[3, 2]).expect("valid reshape");
        assert!(reshaped.storage().ptr_eq(tensor.storage()));
        assert_eq!(reshaped.shape(), &[3, 2]);
        assert_eq!(reshaped.byte_offset(), tensor.byte_offset());
        assert_eq!(
            tensor.reshape(&[4, 2]),
            Err(TensorError::NumelMismatch { from: 6, to: 8 })
        );
    }

    #[test]
    fn test_reshape_infers_one_dimension() {
        let tensor = Tensor::zeros(&[2, 3, 4], DType::Uint8).expect("valid tensor");

        let inferred = tensor.reshape(&[2, -1, 3]).expect("valid reshape");
        assert_eq!(inferred.shape(), &[2, 4, 3]);
        assert!(inferred.storage().ptr_eq(tensor.storage()));

        let flat = tensor.reshape(&[-1]).expect("valid reshape");
        assert_eq!(flat.shape(), &[24]);

        assert_eq!(
            tensor.reshape(&[-1, -1]),
            Err(TensorError::AmbiguousReshape)
        );
        assert_eq!(
            tensor.reshape(&[-1, 5]),
            Err(TensorError::NumelMismatch { from: 24, to: 5 })
        );
        assert_eq!(tensor.reshape(&[-2, 4]), Err(TensorError::NegativeDim(-2)));
    }

    #[test]
    fn test_reshape_inference_on_empty_tensors() {
        let empty = Tensor::zeros(&[0, 3], DType::Float32).expect("valid tensor");

        // 0 elements / 3 known => inferred 0.
        let inferred = empty.reshape(&[-1, 3]).expect("valid reshape");
        assert_eq!(inferred.shape(), &[0, 3]);

        // A zero-sized known dimension leaves -1 ambiguous.
        assert_eq!(
            empty.reshape(&[0, -1]),
            Err(TensorError::NumelMismatch { from: 0, to: 0 })
        );
    }

    #[test]
    fn test_reshape_strided_copies() {
        // Column-major 2x2 layout: storage [1, 3, 2, 4] viewed with strides [1, 2]
        // reads as [[1, 2], [3, 4]].
        let storage = Storage::from_slice(&f32_bytes(&[1.0, 3.0, 2.0, 4.0]));
        let tensor = Tensor::from_storage(storage, DType::Float32, vec![2, 2], Some(vec![1, 2]), 0)
            .expect("valid tensor");
        assert!(!tensor.is_contiguous());

        let reshaped = tensor.reshape(&[4]).expect("valid reshape");
        assert!(!reshaped.storage().ptr_eq(tensor.storage()));
        assert!(reshaped.is_contiguous());
        assert_eq!(
            reshaped.to_contiguous_bytes().as_ref(),
            f32_bytes(&[1.0, 2.0, 3.0, 4.0]).as_slice()
        );
    }

    #[test]
    fn test_to_contiguous_bytes_borrows_when_contiguous() {
        let tensor =
            Tensor::from_slice(&f32_bytes(&[1.0, 2.0]), &[2], DType::Float32).expect("valid");
        assert!(matches!(tensor.to_contiguous_bytes(), Cow::Borrowed(_)));

        let storage = Storage::from_slice(&f32_bytes(&[1.0, 2.0, 3.0, 4.0]));
        let strided = Tensor::from_storage(storage, DType::Float32, vec![2], Some(vec![2]), 0)
            .expect("valid tensor");
        assert!(!strided.is_contiguous());
        let gathered = strided.to_contiguous_bytes();
        assert!(matches!(gathered, Cow::Owned(_)));
        assert_eq!(gathered.as_ref(), f32_bytes(&[1.0, 3.0]).as_slice());
    }

    #[test]
    fn test_strided_gather_multi_dimensional() {
        // 3x4 storage; view the 3x2 sub-tensor of even columns.
        let values: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let storage = Storage::from_slice(&f32_bytes(&values));
        let view = Tensor::from_storage(storage, DType::Float32, vec![3, 2], Some(vec![4, 2]), 0)
            .expect("valid tensor");
        assert_eq!(
            view.to_contiguous_bytes().as_ref(),
            f32_bytes(&[0.0, 2.0, 4.0, 6.0, 8.0, 10.0]).as_slice()
        );
    }

    #[test]
    fn test_stack_and_unstack_roundtrip() {
        let tensors: Vec<Tensor> = (0..3)
            .map(|i| {
                Tensor::from_slice(
                    &f32_bytes(&[i as f32, i as f32 + 0.5]),
                    &[2],
                    DType::Float32,
                )
                .expect("valid tensor")
            })
            .collect();

        let stacked = Tensor::stack(&tensors).expect("valid stack");
        assert_eq!(stacked.shape(), &[3, 2]);
        assert!(stacked.is_contiguous());
        assert_eq!(stacked.storage().as_slice().as_ptr() as usize % 64, 0);

        let views = stacked.unstack().expect("valid unstack");
        assert_eq!(views.len(), 3);
        for (index, (view, original)) in views.iter().zip(&tensors).enumerate() {
            assert!(view.storage().ptr_eq(stacked.storage()));
            assert_eq!(view.byte_offset(), index * 8);
            assert_eq!(view, original);
        }
    }

    #[test]
    fn test_stack_rejects_empty_and_mismatched() {
        assert_eq!(Tensor::stack(&[]), Err(TensorError::EmptyStack));

        let a = Tensor::zeros(&[2], DType::Float32).expect("valid tensor");
        let b = Tensor::zeros(&[3], DType::Float32).expect("valid tensor");
        let c = Tensor::zeros(&[2], DType::Int32).expect("valid tensor");
        assert_eq!(
            Tensor::stack(&[a.clone(), b]),
            Err(TensorError::StackMismatch { index: 1 })
        );
        assert_eq!(
            Tensor::stack(&[a, c]),
            Err(TensorError::StackMismatch { index: 1 })
        );
    }

    #[test]
    fn test_unstack_scalar_fails() {
        let scalar = Tensor::zeros(&[], DType::Float32).expect("valid tensor");
        assert_eq!(scalar.unstack(), Err(TensorError::UnstackScalar));
    }

    #[test]
    fn test_partial_eq_is_logical() {
        // Strided view of [1, 3] vs a contiguous [1, 3]: equal content,
        // different storage layout.
        let storage = Storage::from_slice(&f32_bytes(&[1.0, 2.0, 3.0, 4.0]));
        let strided = Tensor::from_storage(storage, DType::Float32, vec![2], Some(vec![2]), 0)
            .expect("valid tensor");
        let contiguous =
            Tensor::from_slice(&f32_bytes(&[1.0, 3.0]), &[2], DType::Float32).expect("valid");
        assert_eq!(strided, contiguous);

        let other_dtype = Tensor::from_slice(&[0u8; 2], &[2], DType::Uint8).expect("valid");
        let same_bytes = Tensor::from_slice(&[0u8; 2], &[2], DType::Int8).expect("valid");
        assert_ne!(other_dtype, same_bytes);

        let flat = Tensor::zeros(&[4], DType::Float32).expect("valid");
        let square = Tensor::zeros(&[2, 2], DType::Float32).expect("valid");
        assert_ne!(flat, square);
    }

    #[test]
    fn test_view_with_byte_offset() {
        let storage = Storage::from_slice(&f32_bytes(&[1.0, 2.0, 3.0, 4.0]));
        let tail =
            Tensor::from_storage(storage, DType::Float32, vec![2], None, 8).expect("valid tensor");
        assert_eq!(
            tail.to_contiguous_bytes().as_ref(),
            f32_bytes(&[3.0, 4.0]).as_slice()
        );
        assert_eq!(tail.byte_offset(), 8);
    }

    #[test]
    fn test_empty_tensor() {
        let tensor = Tensor::zeros(&[0, 3], DType::Float32).expect("valid tensor");
        assert_eq!(tensor.numel(), 0);
        assert_eq!(tensor.nbytes(), 0);
        assert!(tensor.is_contiguous());
        assert!(tensor.to_contiguous_bytes().is_empty());
        let views = tensor.unstack().expect("valid unstack");
        assert!(views.is_empty());
    }

    #[test]
    fn test_contiguous_strides_table() {
        assert_eq!(contiguous_strides(&[]), Vec::<i64>::new());
        assert_eq!(contiguous_strides(&[5]), vec![1]);
        assert_eq!(contiguous_strides(&[2, 3]), vec![3, 1]);
        assert_eq!(contiguous_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(contiguous_strides(&[0, 3]), vec![3, 1]);
    }

    #[test]
    fn test_explicit_contiguous_strides_detected() {
        let storage = Storage::from_slice(&f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let tensor = Tensor::from_storage(storage, DType::Float32, vec![2, 3], Some(vec![3, 1]), 0)
            .expect("valid tensor");
        assert!(tensor.is_contiguous());
        assert!(matches!(tensor.to_contiguous_bytes(), Cow::Borrowed(_)));
    }
}
