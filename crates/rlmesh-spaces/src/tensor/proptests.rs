use proptest::prelude::*;

use super::{Storage, Tensor, contiguous_strides};
use crate::dtype::{DType, dtype_size};

fn concrete_dtype() -> impl Strategy<Value = DType> {
    prop::sample::select(
        DType::ALL
            .into_iter()
            .filter(|&dtype| dtype != DType::Unspecified)
            .collect::<Vec<_>>(),
    )
}

fn shape() -> impl Strategy<Value = Vec<i64>> {
    prop::collection::vec(0i64..=6, 0..=4)
}

/// A dtype, a shape, and exactly the right number of element bytes.
fn tensor_parts() -> impl Strategy<Value = (DType, Vec<i64>, Vec<u8>)> {
    (concrete_dtype(), shape()).prop_flat_map(|(dtype, shape)| {
        let numel: usize = shape.iter().map(|&dim| dim as usize).product();
        let nbytes = numel * dtype_size(dtype);
        prop::collection::vec(any::<u8>(), nbytes)
            .prop_map(move |data| (dtype, shape.clone(), data))
    })
}

proptest! {
    /// Flattening via reshape is a zero-copy view with identical bytes, and
    /// -1 inference agrees with the explicit shape.
    #[test]
    fn prop_reshape_preserves_bytes((dtype, shape, data) in tensor_parts()) {
        let tensor = Tensor::from_vec(data.clone(), shape, dtype).expect("valid parts");
        let numel = tensor.numel() as i64;

        let flat = tensor.reshape(&[numel]).expect("flatten");
        prop_assert!(flat.storage().ptr_eq(tensor.storage()));
        let flat_bytes = flat.to_contiguous_bytes();
        prop_assert_eq!(flat_bytes.as_ref(), data.as_slice());

        if numel > 0 {
            let inferred = tensor.reshape(&[-1]).expect("infer");
            prop_assert_eq!(inferred.shape(), &[numel]);
            let inferred_bytes = inferred.to_contiguous_bytes();
            prop_assert_eq!(inferred_bytes.as_ref(), data.as_slice());
        }
    }

    /// stack followed by unstack returns tensors equal to the inputs, as
    /// zero-copy views of the stacked storage.
    #[test]
    fn prop_stack_unstack_roundtrip(
        (dtype, shape, data) in tensor_parts(),
        count in 1usize..=4,
    ) {
        let item = Tensor::from_vec(data, shape, dtype).expect("valid parts");
        let items: Vec<Tensor> = (0..count).map(|_| item.clone()).collect();

        let stacked = Tensor::stack(&items).expect("stack");
        prop_assert_eq!(stacked.numel(), item.numel() * count);

        let views = stacked.unstack().expect("unstack");
        prop_assert_eq!(views.len(), count);
        for view in &views {
            prop_assert!(view.storage().ptr_eq(stacked.storage()));
            prop_assert_eq!(view, &item);
        }
    }

    /// A strided prefix view gathers exactly the elements a naive
    /// index-arithmetic reference selects.
    #[test]
    fn prop_strided_view_matches_reference(
        dtype in concrete_dtype(),
        dims in prop::collection::vec(1i64..=5, 1..=3),
        // Fractions used to pick a non-empty prefix of each dimension.
        keep in prop::collection::vec(1u32..=100, 3),
    ) {
        let numel: usize = dims.iter().map(|&dim| dim as usize).product();
        let item = dtype_size(dtype);
        let data: Vec<u8> = (0..numel * item).map(|index| index as u8).collect();
        let storage = Storage::from_slice(&data);

        let base_strides = contiguous_strides(&dims);
        let view_shape: Vec<i64> = dims
            .iter()
            .zip(&keep)
            .map(|(&dim, &fraction)| 1 + (dim - 1) * i64::from(fraction) / 100)
            .collect();

        let view = Tensor::from_storage(
            storage,
            dtype,
            view_shape.clone(),
            Some(base_strides.clone()),
            0,
        )
        .expect("valid view");

        // Naive reference gather over the multi-index space.
        let mut expected = Vec::new();
        let mut index = vec![0i64; view_shape.len()];
        let count: usize = view_shape.iter().map(|&dim| dim as usize).product();
        for _ in 0..count {
            let element: i64 = index
                .iter()
                .zip(&base_strides)
                .map(|(&position, &stride)| position * stride)
                .sum();
            let start = element as usize * item;
            expected.extend_from_slice(&data[start..start + item]);
            for axis in (0..index.len()).rev() {
                index[axis] += 1;
                if index[axis] < view_shape[axis] {
                    break;
                }
                index[axis] = 0;
            }
        }

        let gathered = view.to_contiguous_bytes();
        prop_assert_eq!(gathered.as_ref(), expected.as_slice());
    }
}
