use half::bf16;
use prost_types::{ListValue, Value, value};
use rlmesh_proto::common::v1::MessageBytes;
use rlmesh_spaces::spaces::{BoxSpaceBuilder, DictSpaceBuilder, DiscreteBuilder};
use rlmesh_spaces::{BinaryPayload, DType, RenderRequest, SpaceValue, Tensor};

use super::codec::{decode_value_for_space, encode_value_for_space};
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
fn nested_image_box_roundtrips_and_stays_compact() {
    let space = DictSpaceBuilder::new()
        .insert(
            "image",
            BoxSpaceBuilder::scalar(0.0, 255.0, vec![16, 16, 3])
                .dtype(DType::Uint8)
                .build()
                .unwrap(),
        )
        .build()
        .unwrap();
    let raw: Vec<u8> = (0..16 * 16 * 3).map(|i| (i % 256) as u8).collect();
    let value = SpaceValue::Dict(
        [(
            "image".to_string(),
            SpaceValue::Box(Tensor::from_vec(raw.clone(), vec![16, 16, 3], DType::Uint8).unwrap()),
        )]
        .into_iter()
        .collect(),
    );

    let payload = encode_value_bytes(&value, &space).unwrap();
    let decoded = decode_value_bytes(Some(&payload), &space).unwrap().unwrap();

    assert_eq!(decoded, value);
    assert!(payload.data.len() < raw.len() * 2);
}

#[test]
fn legacy_scalar_list_box_payload_still_decodes() {
    let space = BoxSpaceBuilder::scalar(0.0, 255.0, vec![3])
        .dtype(DType::Uint8)
        .build()
        .unwrap();
    let value = Value {
        kind: Some(value::Kind::ListValue(ListValue {
            values: vec![1.0, 2.0, 3.0]
                .into_iter()
                .map(|number| Value {
                    kind: Some(value::Kind::NumberValue(number)),
                })
                .collect(),
        })),
    };

    let decoded = decode_value_for_space(&value, &space).unwrap();

    assert_eq!(
        decoded,
        SpaceValue::Box(Tensor::from_vec(vec![1, 2, 3], vec![3], DType::Uint8).unwrap())
    );
}

#[test]
fn int16_box_roundtrips_raw_and_base64() {
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

    // Base64 path (Box nested in a Dict goes through proto Value encoding).
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
fn bfloat16_box_roundtrips_raw_and_legacy_scalar_list() {
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

    // Legacy scalar-list payload packs through the bf16 scalar codec.
    let legacy = Value {
        kind: Some(value::Kind::ListValue(ListValue {
            values: vec![0.5, 2.0]
                .into_iter()
                .map(|number| Value {
                    kind: Some(value::Kind::NumberValue(number)),
                })
                .collect(),
        })),
    };
    let decoded = decode_value_for_space(&legacy, &space).unwrap();
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
fn discrete_decode_rejects_fractional_and_nonfinite_floats() {
    let space = DiscreteBuilder::new(8).build().unwrap();

    let fractional = Value {
        kind: Some(value::Kind::NumberValue(3.7)),
    };
    let err = decode_value_for_space(&fractional, &space)
        .expect_err("a fractional Discrete action must be rejected, not truncated to 3");
    assert!(err.to_string().contains("fractional"));

    let nan = Value {
        kind: Some(value::Kind::NumberValue(f64::NAN)),
    };
    let err = decode_value_for_space(&nan, &space)
        .expect_err("a NaN Discrete action must be rejected, not coerced to 0");
    assert!(err.to_string().contains("non-finite"));
}

#[test]
fn discrete_encode_rejects_precision_losing_integers() {
    let space = DiscreteBuilder::new(8).build().unwrap();

    // 2^53 + 1 cannot be represented exactly as an f64; encoding it as a JSON
    // number would silently corrupt the value.
    let value = SpaceValue::Discrete((1i64 << 53) + 1);
    let err = encode_value_for_space(&value, &space)
        .expect_err("encoding an integer beyond 2^53 must be rejected");
    assert!(err.to_string().contains("2^53"));

    // A value within the exact-float range round-trips fine.
    let value = SpaceValue::Discrete(5);
    let encoded = encode_value_for_space(&value, &space).unwrap();
    assert_eq!(
        decode_value_for_space(&encoded, &space).unwrap(),
        SpaceValue::Discrete(5)
    );
}

#[test]
fn int_to_proto_f64_rejects_i64_min_without_overflow() {
    use super::scalars::int_to_proto_f64;

    // i64::MIN.abs() overflows; the guard must reject the value, not panic
    // (debug) or wrap negative and accept a lossy encode (release).
    let err = int_to_proto_f64(i64::MIN)
        .expect_err("i64::MIN exceeds the exact-float range and must be rejected");
    assert!(err.to_string().contains("2^53"));

    assert_eq!(int_to_proto_f64(1 << 53).unwrap(), 9007199254740992.0);
    let err = int_to_proto_f64((1 << 53) + 1).expect_err("2^53 + 1 must be rejected");
    assert!(err.to_string().contains("2^53"));
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
