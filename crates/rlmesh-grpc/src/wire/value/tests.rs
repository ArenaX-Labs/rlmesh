use prost::bytes::Bytes;
use rlmesh_spaces::spaces::{
    BoxSpaceBuilder, DictSpaceBuilder, DiscreteBuilder, TextBuilder, TupleSpaceBuilder,
};
use rlmesh_spaces::{BinaryPayload, DType, RenderRequest, SpaceValue, Tensor};

use super::{
    binary_to_bytes, decode_batched_partial_values, decode_leaves, encode_batched_partial_values,
    encode_leaves, leaves_value, optional_bytes_to_binary, render_request_to_proto,
    render_result_from_proto,
};
use crate::error::ProtocolError;

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
fn render_result_passes_known_formats_and_drops_unknown() {
    use rlmesh_proto::env::v1::{RenderFormat, RenderResponse};

    // UNSPECIFIED (historical PNG default) and PNG surface the frame.
    for format in [RenderFormat::Unspecified, RenderFormat::Png] {
        let resp = RenderResponse {
            frame: Some(b"\x89PNG".to_vec()),
            format: format as i32,
        };
        let result = render_result_from_proto(resp).expect("decodes");
        assert!(
            result.frame.is_some(),
            "format {format:?} must surface the frame"
        );
    }

    // An unrecognized future format must be skipped (no frame), never surfaced
    // as if it were PNG.
    let resp = RenderResponse {
        frame: Some(b"future-codec".to_vec()),
        format: 999,
    };
    let result = render_result_from_proto(resp).expect("decodes without error");
    assert!(
        result.frame.is_none(),
        "an unknown render format must be dropped, not misread as PNG"
    );
}

#[test]
fn box_leaf_shares_storage_with_the_encoded_value() {
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

    let leaves = encode_leaves(&value, &space).unwrap();
    assert_eq!(leaves.len(), 1);
    // The leaf must view the tensor's refcounted storage, not a copy.
    assert_eq!(leaves[0].as_ptr(), storage_ptr);
    assert_eq!(&leaves[0][..], &[1, 2, 3, 4]);
}

#[test]
fn top_level_discrete_roundtrips_at_declared_dtype_width() {
    let space = DiscreteBuilder::new(1000).build().unwrap();
    for value in [0i64, 1, 999] {
        let leaves = encode_leaves(&SpaceValue::Discrete(value), &space).unwrap();
        assert_eq!(leaves.len(), 1);
        // gym Discrete defaults to int64, so the leaf is the dtype-wide 8 bytes.
        assert_eq!(
            leaves[0].len(),
            8,
            "discrete encodes at its declared dtype width"
        );
        let SpaceValue::Discrete(got) = decode_leaves(&leaves, &space).unwrap() else {
            panic!("expected a discrete value");
        };
        assert_eq!(got, value);
    }
    // A leaf that is not exactly the dtype width is rejected, not truncated.
    assert!(decode_leaves(&[Bytes::from_static(&[0u8; 4])], &space).is_err());
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

    let n = values.len();
    let payload = encode_batched_partial_values(&values, &space).unwrap();
    let decoded = decode_batched_partial_values(Some(&payload), &space, n).unwrap();

    assert_eq!(decoded, values);
}

#[test]
fn batched_partial_box_values_use_raw_concatenated_slab() {
    let space = BoxSpaceBuilder::scalar(0.0, 255.0, vec![2])
        .dtype(DType::Uint8)
        .build()
        .unwrap();
    let values = vec![
        SpaceValue::Box(Tensor::from_vec(vec![1, 2], vec![2], DType::Uint8).unwrap()),
        SpaceValue::Box(Tensor::from_vec(vec![3, 4], vec![2], DType::Uint8).unwrap()),
    ];

    let n = values.len();
    let payload = encode_batched_partial_values(&values, &space).unwrap();
    // One leaf, the row-major (N, *shape) slab = lane-contiguous concat.
    assert_eq!(payload.leaves.len(), 1);
    assert_eq!(payload.leaves[0].as_ref(), &[1, 2, 3, 4]);
    let decoded = decode_batched_partial_values(Some(&payload), &space, n).unwrap();

    assert_eq!(decoded, values);
}

#[test]
fn batched_partial_decode_rejects_wrong_lane_count() {
    let space = BoxSpaceBuilder::scalar(0.0, 255.0, vec![2])
        .dtype(DType::Uint8)
        .build()
        .unwrap();
    // 3 bytes can't split into N=2 lanes of stride 2.
    let bad = leaves_value(vec![Bytes::from_static(&[1, 2, 3])]);

    let error = decode_batched_partial_values(Some(&bad), &space, 2).unwrap_err();
    assert!(
        matches!(error, ProtocolError::LengthMismatch(_)),
        "unexpected error: {error}"
    );
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
fn nested_image_box_roundtrips_byte_exact_without_inflation() {
    // A Dict{image: uint8 Box, choice: Discrete(2^53+1)} round-trips byte-exact,
    // and the 100KB image leaf is raw bytes with zero per-leaf framing.
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

    let leaves = encode_leaves(&value, &space).unwrap();
    let decoded = decode_leaves(&leaves, &space).unwrap();
    assert_eq!(decoded, value);
    // Keys sort to (choice, image): the image leaf is leaves[1], raw and exact.
    assert_eq!(
        leaves[1].len(),
        raw.len(),
        "image leaf must carry raw bytes, no inflation"
    );
}

#[test]
fn nested_tuple_values_roundtrip_byte_exact() {
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

    let leaves = encode_leaves(&value, &space).unwrap();
    let decoded = decode_leaves(&leaves, &space).unwrap();
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

    // Raw single-value leaf.
    let leaves = encode_leaves(&value, &space).unwrap();
    assert_eq!(leaves[0].as_ref(), data.as_slice());
    let decoded = decode_leaves(&leaves, &space).unwrap();
    assert_eq!(decoded, value);

    // Raw concatenated batch slab.
    let values = vec![value.clone(), value.clone()];
    let n = values.len();
    let batch = encode_batched_partial_values(&values, &space).unwrap();
    assert_eq!(batch.leaves[0].len(), data.len() * 2);
    let decoded = decode_batched_partial_values(Some(&batch), &space, n).unwrap();
    assert_eq!(decoded, values);

    // Nested in a Dict, the leaf carries the same raw little-endian bytes.
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
    let leaves = encode_leaves(&dict_value, &dict_space).unwrap();
    let decoded = decode_leaves(&leaves, &dict_space).unwrap();
    assert_eq!(decoded, dict_value);
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

    let leaves = encode_leaves(&value, &space).unwrap();
    assert_eq!(leaves[0].as_ref(), &[1, 3]);
    let decoded = decode_leaves(&leaves, &space).unwrap();
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

    let leaves = encode_leaves(&value, &space).unwrap();
    let decoded = decode_leaves(&leaves, &space).unwrap();
    assert_eq!(decoded, value);
    let SpaceValue::Dict(map) = decoded else {
        panic!("expected dict");
    };
    assert_eq!(map.get("choice"), Some(&SpaceValue::Discrete(big)));
}

#[test]
fn top_level_text_is_raw_utf8() {
    // Text crosses as raw UTF-8, self-framed by the `repeated bytes` element
    // length -- no SpaceValueNode tree. A conformant non-Rust peer relies on this.
    let space = TextBuilder::new(8).build().unwrap();
    let value = SpaceValue::Text("hi".to_string());

    let leaves = encode_leaves(&value, &space).unwrap();
    assert_eq!(leaves[0].as_ref(), b"hi");
    let decoded = decode_leaves(&leaves, &space).unwrap();
    assert_eq!(decoded, value);

    // An empty Text round-trips as a single empty leaf (present, not absent).
    let empty = SpaceValue::Text(String::new());
    let empty_leaves = encode_leaves(&empty, &space).unwrap();
    assert_eq!(empty_leaves.len(), 1);
    assert!(empty_leaves[0].is_empty());
    assert_eq!(decode_leaves(&empty_leaves, &space).unwrap(), empty);
}

#[test]
fn wire_decode_rejects_malformed_specs() {
    use rlmesh_proto::spaces::v1 as proto;

    // A peer-supplied MultiBinary spec with a wide (non-1-byte) dtype would make
    // the leaf byte count disagree with the raw batch stride; reject at decode.
    let wide_multibinary = proto::SpaceSpec {
        shape: vec![3],
        dtype: proto::DType::Int32 as i32,
        spec: Some(proto::space_spec::Spec::MultiBinary(
            proto::MultiBinarySpec {},
        )),
    };
    assert!(crate::wire::space_spec_from_proto(wide_multibinary).is_err());

    // A MultiDiscrete spec with a float dtype would lose index precision.
    let float_multidiscrete = proto::SpaceSpec {
        shape: vec![2],
        dtype: proto::DType::Float32 as i32,
        spec: Some(proto::space_spec::Spec::MultiDiscrete(
            proto::MultiDiscreteSpec { nvec: vec![3, 4] },
        )),
    };
    assert!(crate::wire::space_spec_from_proto(float_multidiscrete).is_err());

    // A well-formed uint8 MultiBinary still decodes.
    let ok = proto::SpaceSpec {
        shape: vec![3],
        dtype: proto::DType::Uint8 as i32,
        spec: Some(proto::space_spec::Spec::MultiBinary(
            proto::MultiBinarySpec {},
        )),
    };
    assert!(crate::wire::space_spec_from_proto(ok).is_ok());
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
