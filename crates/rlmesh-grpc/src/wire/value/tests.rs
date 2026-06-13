use half::bf16;
use rlmesh_proto::common::v1::MessageBytes;
use rlmesh_spaces::spaces::{
    BoxSpaceBuilder, DictSpaceBuilder, DiscreteBuilder, TupleSpaceBuilder,
};
use rlmesh_spaces::{BinaryPayload, DType, RenderRequest, SpaceValue, Tensor};

use super::{
    binary_to_bytes, decode_batch_bytes, decode_batched_partial_values, decode_value_bytes,
    encode_batch_bytes, encode_batched_partial_values, encode_value_bytes,
    optional_bytes_to_binary, render_request_to_proto,
};

#[test]
fn render_request_without_env_index_uses_empty_mask() {
    let request = RenderRequest {
        env_index: None,
        timeout_ms: 17,
    };

    let proto = render_request_to_proto(&request);
    assert_eq!(proto.mask, Vec::<u8>::new());
    assert_eq!(proto.timeout_ms, 17);
}

#[test]
fn render_request_with_env_index_maps_to_single_bit_mask() {
    let request = RenderRequest {
        env_index: Some(2),
        timeout_ms: 0,
    };

    let proto = render_request_to_proto(&request);
    assert_eq!(proto.mask, vec![0, 0, 1]);
}

#[test]
fn composite_tensor_leaf_shares_storage_with_the_encoded_node() {
    use rlmesh_proto::spaces::v1::space_value_node::Kind as NodeKind;

    let space = DictSpaceBuilder::new()
        .insert(
            "obs",
            BoxSpaceBuilder::scalar(0.0, 255.0, vec![4])
                .dtype(DType::Uint8)
                .build()
                .unwrap(),
        )
        .build()
        .unwrap();
    let tensor = Tensor::from_vec(vec![1, 2, 3, 4], vec![4], DType::Uint8).unwrap();
    let storage_ptr = tensor.to_contiguous_bytes().as_ptr();
    let value = SpaceValue::Dict(
        [("obs".to_string(), SpaceValue::Box(tensor))]
            .into_iter()
            .collect(),
    );

    let node = super::codec::encode_value_node(&value, &space).unwrap();
    let Some(NodeKind::Dict(map)) = node.kind.as_ref() else {
        panic!("expected dict node, got {:?}", node.kind);
    };
    let Some(NodeKind::Tensor(raw)) = map.entries["obs"].kind.as_ref() else {
        panic!("expected tensor leaf, got {:?}", map.entries["obs"].kind);
    };

    // The leaf must view the tensor's refcounted storage, not a copy.
    assert_eq!(raw.as_ptr(), storage_ptr);
    assert_eq!(&raw[..], &[1, 2, 3, 4]);
}

#[test]
fn batched_dict_values_roundtrip_through_wire_helpers() {
    let space = DictSpaceBuilder::new()
        .insert("choice", DiscreteBuilder::new(4).build().unwrap())
        .build()
        .unwrap();
    let values = vec![
        SpaceValue::Dict(
            [("choice".to_string(), SpaceValue::Discrete(1))]
                .into_iter()
                .collect(),
        ),
        SpaceValue::Dict(
            [("choice".to_string(), SpaceValue::Discrete(3))]
                .into_iter()
                .collect(),
        ),
    ];

    let payload = encode_batch_bytes(&values, &space).unwrap();
    let decoded = decode_batch_bytes(Some(&payload), &space).unwrap();

    assert_eq!(decoded, values);
}

#[test]
fn batched_partial_box_values_use_raw_concatenated_payload() {
    let space = BoxSpaceBuilder::scalar(0.0, 255.0, vec![2])
        .dtype(DType::Uint8)
        .build()
        .unwrap();
    let values = vec![
        SpaceValue::Box(Tensor::from_vec(vec![1, 2], vec![2], DType::Uint8).unwrap()),
        SpaceValue::Box(Tensor::from_vec(vec![3, 4], vec![2], DType::Uint8).unwrap()),
    ];

    let payload = encode_batched_partial_values(&values, &space).unwrap();
    let decoded = decode_batched_partial_values(Some(&payload), &space).unwrap();

    assert_eq!(payload.data, vec![1, 2, 3, 4]);
    assert_eq!(decoded, values);
}

#[test]
fn batched_partial_raw_decode_rejects_misaligned_payload() {
    let space = BoxSpaceBuilder::scalar(0.0, 255.0, vec![2])
        .dtype(DType::Uint8)
        .build()
        .unwrap();
    let payload = MessageBytes {
        data: vec![1, 2, 3],
    };

    let error = decode_batched_partial_values(Some(&payload), &space).unwrap_err();

    assert!(error.to_string().contains("is not divisible"));
}

#[test]
fn binary_payload_roundtrips_through_message_bytes() {
    let payload = BinaryPayload {
        data: vec![1, 2, 3, 4],
    };

    let encoded = binary_to_bytes(&payload);
    let decoded = optional_bytes_to_binary(Some(&encoded))
        .unwrap()
        .expect("payload present");

    assert_eq!(decoded, payload);
}

#[test]
fn nested_image_box_roundtrips_byte_exact_without_base64_inflation() {
    // D1: a Dict{image: uint8 Box, choice: Discrete(2^53+1)} must round-trip
    // byte-exact, and a 100KB image leaf must not pay base64 inflation: the
    // encoded payload is the raw bytes plus modest proto framing (< 1.1x).
    let big_discrete = (1i64 << 53) + 1;
    let space = DictSpaceBuilder::new()
        .insert(
            "image",
            BoxSpaceBuilder::scalar(0.0, 255.0, vec![200, 200, 3])
                .dtype(DType::Uint8)
                .build()
                .unwrap(),
        )
        .insert(
            "choice",
            DiscreteBuilder::new(big_discrete).build().unwrap(),
        )
        .build()
        .unwrap();
    let raw: Vec<u8> = (0..200 * 200 * 3).map(|i| (i % 256) as u8).collect();
    assert!(raw.len() > 100_000, "image leaf must exceed 100KB");
    let value = SpaceValue::Dict(
        [
            ("choice".to_string(), SpaceValue::Discrete(big_discrete - 1)),
            (
                "image".to_string(),
                SpaceValue::Box(
                    Tensor::from_vec(raw.clone(), vec![200, 200, 3], DType::Uint8).unwrap(),
                ),
            ),
        ]
        .into_iter()
        .collect(),
    );

    let payload = encode_value_bytes(&value, &space).unwrap();
    let decoded = decode_value_bytes(Some(&payload), &space).unwrap().unwrap();

    assert_eq!(decoded, value);
    // No base64 (which would be ~1.33x); framing stays well under 1.1x.
    assert!(
        (payload.data.len() as f64) < (raw.len() as f64) * 1.1,
        "encoded {} bytes for a {}-byte image leaf inflated past 1.1x",
        payload.data.len(),
        raw.len(),
    );
}

#[test]
fn nested_tuple_values_roundtrip_byte_exact() {
    // D1: tuple nesting, including a nested dict, round-trips exactly.
    let inner_dict = DictSpaceBuilder::new()
        .insert("choice", DiscreteBuilder::new(4).build().unwrap())
        .build()
        .unwrap();
    let space = TupleSpaceBuilder::new()
        .with(
            BoxSpaceBuilder::scalar(0.0, 255.0, vec![2])
                .dtype(DType::Uint8)
                .build()
                .unwrap(),
        )
        .with(inner_dict)
        .build()
        .unwrap();
    let value = SpaceValue::Tuple(vec![
        SpaceValue::Box(Tensor::from_vec(vec![7, 9], vec![2], DType::Uint8).unwrap()),
        SpaceValue::Dict(
            [("choice".to_string(), SpaceValue::Discrete(3))]
                .into_iter()
                .collect(),
        ),
    ]);

    let payload = encode_value_bytes(&value, &space).unwrap();
    let decoded = decode_value_bytes(Some(&payload), &space).unwrap().unwrap();
    assert_eq!(decoded, value);
}

#[test]
fn int16_box_roundtrips_raw_and_nested() {
    let space = BoxSpaceBuilder::scalar(-1000.0, 1000.0, vec![3])
        .dtype(DType::Int16)
        .build()
        .unwrap();
    let data: Vec<u8> = [-5i16, 0, 999]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let value = SpaceValue::Box(Tensor::from_vec(data.clone(), vec![3], DType::Int16).unwrap());

    // Raw single-value payload.
    let payload = encode_value_bytes(&value, &space).unwrap();
    assert_eq!(payload.data, data);
    let decoded = decode_value_bytes(Some(&payload), &space).unwrap().unwrap();
    assert_eq!(decoded, value);

    // Raw concatenated batch payload.
    let values = vec![value.clone(), value.clone()];
    let batch = encode_batched_partial_values(&values, &space).unwrap();
    assert_eq!(batch.data.len(), data.len() * 2);
    let decoded = decode_batched_partial_values(Some(&batch), &space).unwrap();
    assert_eq!(decoded, values);

    // Nested in a Dict, the leaf carries the same raw little-endian bytes
    // verbatim (no base64), so round-trip stays byte-exact.
    let dict_space = DictSpaceBuilder::new()
        .insert(
            "reading",
            BoxSpaceBuilder::scalar(-1000.0, 1000.0, vec![3])
                .dtype(DType::Int16)
                .build()
                .unwrap(),
        )
        .build()
        .unwrap();
    let dict_value = SpaceValue::Dict([("reading".to_string(), value)].into_iter().collect());
    let payload = encode_value_bytes(&dict_value, &dict_space).unwrap();
    let decoded = decode_value_bytes(Some(&payload), &dict_space)
        .unwrap()
        .unwrap();
    assert_eq!(decoded, dict_value);
}

#[test]
fn bfloat16_box_roundtrips_raw() {
    let space = BoxSpaceBuilder::scalar(0.0, 10.0, vec![2])
        .dtype(DType::Bfloat16)
        .build()
        .unwrap();
    let data: Vec<u8> = [bf16::from_f32(0.5), bf16::from_f32(2.0)]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let value = SpaceValue::Box(Tensor::from_vec(data.clone(), vec![2], DType::Bfloat16).unwrap());

    // Raw payload is the little-endian bf16 bytes, unchanged.
    let payload = encode_value_bytes(&value, &space).unwrap();
    assert_eq!(payload.data, data);
    let decoded = decode_value_bytes(Some(&payload), &space).unwrap().unwrap();
    assert_eq!(decoded, value);
}

#[test]
fn strided_box_view_encodes_contiguously() {
    use rlmesh_spaces::Storage;

    let space = BoxSpaceBuilder::scalar(0.0, 255.0, vec![2])
        .dtype(DType::Uint8)
        .build()
        .unwrap();
    // Storage [1, 9, 3, 9]; stride-2 view sees [1, 3].
    let storage = Storage::from_slice(&[1, 9, 3, 9]);
    let view = Tensor::from_storage(storage, DType::Uint8, vec![2], Some(vec![2]), 0).unwrap();
    let value = SpaceValue::Box(view);

    let payload = encode_value_bytes(&value, &space).unwrap();
    assert_eq!(payload.data, vec![1, 3]);
    let decoded = decode_value_bytes(Some(&payload), &space).unwrap().unwrap();
    assert_eq!(decoded, value);
}

#[test]
fn nested_discrete_survives_beyond_two_pow_53_exactly() {
    let big = (1i64 << 53) + 1;
    let space = DictSpaceBuilder::new()
        .insert("choice", DiscreteBuilder::new(big + 1).build().unwrap())
        .build()
        .unwrap();
    let value = SpaceValue::Dict(
        [("choice".to_string(), SpaceValue::Discrete(big))]
            .into_iter()
            .collect(),
    );

    let payload = encode_value_bytes(&value, &space).unwrap();
    let decoded = decode_value_bytes(Some(&payload), &space).unwrap().unwrap();
    assert_eq!(decoded, value);
    let SpaceValue::Dict(map) = decoded else {
        panic!("expected dict");
    };
    assert_eq!(map.get("choice"), Some(&SpaceValue::Discrete(big)));
}

#[test]
fn float_to_int_rejects_two_pow_63_boundary() {
    use super::scalars::float_to_int;

    // `i64::MAX as f64` rounds up to 2^63; an exact 2^63 input must be
    // rejected rather than saturated to i64::MAX by `as i64`.
    let two_pow_63 = (i64::MAX as f64) + 1.0 - 1.0; // 2^63 exactly
    assert_eq!(two_pow_63, 9223372036854775808.0);
    let err = float_to_int(two_pow_63).expect_err("2^63 is out of i64 range");
    assert!(err.to_string().contains("out of range"));

    // The largest f64 strictly below 2^63 and i64::MIN itself both convert.
    let below = f64::from_bits(two_pow_63.to_bits() - 1);
    assert_eq!(float_to_int(below).unwrap(), below as i64);
    assert_eq!(float_to_int(i64::MIN as f64).unwrap(), i64::MIN);
}
