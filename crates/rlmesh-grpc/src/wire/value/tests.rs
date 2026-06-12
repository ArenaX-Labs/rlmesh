use half::bf16;
use prost_types::{ListValue, Value, value};
use rlmesh_proto::common::v1::MessageBytes;
use rlmesh_spaces::spaces::{BoxSpaceBuilder, DictSpaceBuilder, DiscreteBuilder};
use rlmesh_spaces::{BinaryPayload, DType, RenderRequest, SpaceValue, Tensor};

use super::codec::decode_value_for_space;
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
