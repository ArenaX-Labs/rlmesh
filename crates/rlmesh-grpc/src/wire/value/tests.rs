use prost_types::{ListValue, Value, value};
use rlmesh_proto::common::v1::MessageBytes;
use rlmesh_spaces::v1::spaces::{BoxSpaceBuilder, DictSpaceBuilder, DiscreteBuilder};
use rlmesh_spaces::v1::{BinaryPayload, BoxValue, DType, RenderRequest, SpaceValue};

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
        SpaceValue::Box(BoxValue {
            data: vec![1, 2],
            shape: vec![2],
            dtype: DType::Uint8,
        }),
        SpaceValue::Box(BoxValue {
            data: vec![3, 4],
            shape: vec![2],
            dtype: DType::Uint8,
        }),
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
            SpaceValue::Box(BoxValue {
                data: raw.clone(),
                shape: vec![16, 16, 3],
                dtype: DType::Uint8,
            }),
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
        SpaceValue::Box(BoxValue {
            data: vec![1, 2, 3],
            shape: vec![3],
            dtype: DType::Uint8,
        })
    );
}
