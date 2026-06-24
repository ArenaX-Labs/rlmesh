use proptest::prelude::*;
use rlmesh_spaces as native;
use rlmesh_spaces::{DType, Tensor, dtype_size};

use super::{
    decode_batched_partial_values, decode_leaves, encode_batched_partial_values, encode_leaves,
};

fn concrete_dtype() -> impl Strategy<Value = DType> {
    prop::sample::select(
        DType::ALL
            .into_iter()
            .filter(|&dtype| dtype != DType::Unspecified)
            .collect::<Vec<_>>(),
    )
}

fn box_space(shape: &[i64], dtype: DType) -> native::SpaceSpec {
    native::SpaceSpec {
        shape: shape.to_vec(),
        dtype,
        spec: Some(native::SpaceKind::Box(native::BoxSpec { bounds: None })),
    }
}

/// A Box-compatible dtype, a positive-rank shape, and exact element bytes.
fn box_parts() -> impl Strategy<Value = (DType, Vec<i64>, Vec<u8>)> {
    (concrete_dtype(), prop::collection::vec(1i64..=4, 1..=3)).prop_flat_map(|(dtype, shape)| {
        let numel: usize = shape.iter().map(|&dim| dim as usize).product();
        let nbytes = numel * dtype_size(dtype);
        prop::collection::vec(any::<u8>(), nbytes)
            .prop_map(move |data| (dtype, shape.clone(), data))
    })
}

proptest! {
    /// Box values cross the wire byte-exactly: the payload is the raw
    /// little-endian element bytes, and decode reproduces the value.
    #[test]
    fn prop_box_value_roundtrips_byte_exact((dtype, shape, data) in box_parts()) {
        let space = box_space(&shape, dtype);
        let value = native::SpaceValue::Box(
            Tensor::from_vec(data.clone(), shape, dtype).expect("valid tensor"),
        );

        let leaves = encode_leaves(&value, &space).expect("encode");
        prop_assert_eq!(leaves[0].as_ref(), data.as_slice());

        let decoded = decode_leaves(&leaves, &space).expect("decode");
        prop_assert_eq!(decoded, value);
    }

    /// Batched partial Box payloads concatenate per-value bytes and split
    /// back into the original values.
    #[test]
    fn prop_batched_partial_box_roundtrips(
        (dtype, shape, data) in box_parts(),
        count in 1usize..=4,
    ) {
        let space = box_space(&shape, dtype);
        let values: Vec<native::SpaceValue> = (0..count)
            .map(|index| {
                // Vary the leading byte so batch order is observable.
                let mut bytes = data.clone();
                bytes[0] = bytes[0].wrapping_add(index as u8);
                native::SpaceValue::Box(
                    Tensor::from_vec(bytes, shape.clone(), dtype).expect("valid tensor"),
                )
            })
            .collect();

        let payload = encode_batched_partial_values(&values, &space).expect("encode");
        prop_assert_eq!(payload.leaves[0].len(), data.len() * count);

        let decoded =
            decode_batched_partial_values(Some(&payload), &space, count).expect("decode");
        prop_assert_eq!(decoded, values);
    }
}
