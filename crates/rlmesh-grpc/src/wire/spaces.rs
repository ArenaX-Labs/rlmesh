use std::collections::BTreeMap;

use rlmesh_proto::env::v1 as env_proto;
use rlmesh_proto::spaces::v1 as proto;
use rlmesh_proto::spaces::v1::meta_value::Kind as MetaKind;
use rlmesh_spaces as native;

use crate::error::ProtocolError;

pub fn env_contract_to_proto(spec: &native::EnvContract) -> env_proto::EnvContract {
    env_proto::EnvContract {
        id: spec.id.clone(),
        action_space: spec.action_space.as_ref().map(space_spec_to_proto),
        observation_space: spec.observation_space.as_ref().map(space_spec_to_proto),
        metadata: spec.metadata.as_ref().map(meta_map_to_proto),
        render_mode: spec.render_mode.clone(),
        num_envs: spec.num_envs,
    }
}

pub fn env_contract_from_proto(
    spec: env_proto::EnvContract,
) -> Result<native::EnvContract, ProtocolError> {
    Ok(native::EnvContract {
        id: spec.id,
        action_space: spec.action_space.map(space_spec_from_proto).transpose()?,
        observation_space: spec
            .observation_space
            .map(space_spec_from_proto)
            .transpose()?,
        metadata: spec.metadata.map(meta_map_from_proto),
        render_mode: spec.render_mode,
        num_envs: spec.num_envs,
    })
}

pub fn space_spec_to_proto(spec: &native::SpaceSpec) -> proto::SpaceSpec {
    proto::SpaceSpec {
        shape: spec.shape.clone(),
        dtype: proto_dtype_from_native(spec.dtype) as i32,
        spec: spec.spec.as_ref().map(space_kind_to_proto),
    }
}

pub fn space_spec_from_proto(spec: proto::SpaceSpec) -> Result<native::SpaceSpec, ProtocolError> {
    Ok(native::SpaceSpec {
        shape: spec.shape,
        dtype: native_dtype_from_proto(spec.dtype)?,
        spec: spec.spec.map(space_kind_from_proto).transpose()?,
    })
}

fn space_kind_to_proto(kind: &native::SpaceKind) -> proto::space_spec::Spec {
    match kind {
        native::SpaceKind::Box(spec) => proto::space_spec::Spec::Box(proto::BoxSpec {
            bounds: spec.bounds.as_ref().map(box_bounds_to_proto),
        }),
        native::SpaceKind::Discrete(spec) => {
            proto::space_spec::Spec::Discrete(proto::DiscreteSpec {
                n: spec.n,
                start: spec.start,
            })
        }
        native::SpaceKind::MultiBinary(spec) => {
            proto::space_spec::Spec::MultiBinary(proto::MultiBinarySpec {
                n: spec.n.as_ref().map(multibinary_n_to_proto),
            })
        }
        native::SpaceKind::MultiDiscrete(spec) => {
            proto::space_spec::Spec::MultiDiscrete(proto::MultiDiscreteSpec {
                nvec: spec.nvec.as_ref().map(multidiscrete_nvec_to_proto),
            })
        }
        native::SpaceKind::Text(spec) => proto::space_spec::Spec::Text(proto::TextSpec {
            min_length: spec.min_length,
            max_length: spec.max_length,
            charset: spec.charset.clone(),
        }),
        native::SpaceKind::Dict(spec) => proto::space_spec::Spec::Dict(proto::DictSpec {
            keys: spec.keys.clone(),
            spaces: spec.spaces.iter().map(space_spec_to_proto).collect(),
        }),
        native::SpaceKind::Tuple(spec) => proto::space_spec::Spec::Tuple(proto::TupleSpec {
            spaces: spec.spaces.iter().map(space_spec_to_proto).collect(),
        }),
    }
}

fn space_kind_from_proto(
    kind: proto::space_spec::Spec,
) -> Result<native::SpaceKind, ProtocolError> {
    Ok(match kind {
        proto::space_spec::Spec::Box(spec) => native::SpaceKind::Box(native::BoxSpec {
            bounds: spec.bounds.map(box_bounds_from_proto).transpose()?,
        }),
        proto::space_spec::Spec::Discrete(spec) => {
            native::SpaceKind::Discrete(native::DiscreteSpec {
                n: spec.n,
                start: spec.start,
            })
        }
        proto::space_spec::Spec::MultiBinary(spec) => {
            native::SpaceKind::MultiBinary(native::MultiBinarySpec {
                n: spec.n.map(multibinary_n_from_proto).transpose()?,
            })
        }
        proto::space_spec::Spec::MultiDiscrete(spec) => {
            native::SpaceKind::MultiDiscrete(native::MultiDiscreteSpec {
                nvec: spec.nvec.map(multidiscrete_nvec_from_proto).transpose()?,
            })
        }
        proto::space_spec::Spec::Text(spec) => native::SpaceKind::Text(native::TextSpec {
            min_length: spec.min_length,
            max_length: spec.max_length,
            charset: spec.charset,
        }),
        proto::space_spec::Spec::Dict(spec) => native::SpaceKind::Dict(native::DictSpec {
            keys: spec.keys,
            spaces: spec
                .spaces
                .into_iter()
                .map(space_spec_from_proto)
                .collect::<Result<_, _>>()?,
        }),
        proto::space_spec::Spec::Tuple(spec) => native::SpaceKind::Tuple(native::TupleSpec {
            spaces: spec
                .spaces
                .into_iter()
                .map(space_spec_from_proto)
                .collect::<Result<_, _>>()?,
        }),
    })
}

fn box_bounds_to_proto(bounds: &native::BoxBounds) -> proto::box_spec::Bounds {
    match bounds {
        native::BoxBounds::Unbounded(value) => proto::box_spec::Bounds::Unbounded(*value),
        native::BoxBounds::Uniform(bounds) => {
            proto::box_spec::Bounds::Uniform(proto::UniformBounds {
                low: bounds.low,
                high: bounds.high,
            })
        }
        native::BoxBounds::Elementwise(bounds) => {
            proto::box_spec::Bounds::Elementwise(proto::ElementwiseBounds {
                low: bounds.low.clone(),
                high: bounds.high.clone(),
            })
        }
        native::BoxBounds::TypedUniform(bounds) => {
            proto::box_spec::Bounds::TypedUniform(proto::TypedUniformBounds {
                low: bounds.low.clone(),
                high: bounds.high.clone(),
            })
        }
        native::BoxBounds::TypedElementwise(bounds) => {
            proto::box_spec::Bounds::TypedElementwise(proto::TypedElementwiseBounds {
                low: bounds.low.clone(),
                high: bounds.high.clone(),
            })
        }
    }
}

fn box_bounds_from_proto(
    bounds: proto::box_spec::Bounds,
) -> Result<native::BoxBounds, ProtocolError> {
    Ok(match bounds {
        proto::box_spec::Bounds::Unbounded(value) => native::BoxBounds::Unbounded(value),
        proto::box_spec::Bounds::Uniform(bounds) => {
            native::BoxBounds::Uniform(native::UniformBounds {
                low: bounds.low,
                high: bounds.high,
            })
        }
        proto::box_spec::Bounds::Elementwise(bounds) => {
            native::BoxBounds::Elementwise(native::ElementwiseBounds {
                low: bounds.low,
                high: bounds.high,
            })
        }
        proto::box_spec::Bounds::TypedUniform(bounds) => {
            native::BoxBounds::TypedUniform(native::TypedUniformBounds {
                low: bounds.low,
                high: bounds.high,
            })
        }
        proto::box_spec::Bounds::TypedElementwise(bounds) => {
            native::BoxBounds::TypedElementwise(native::TypedElementwiseBounds {
                low: bounds.low,
                high: bounds.high,
            })
        }
    })
}

fn multibinary_n_to_proto(value: &native::MultiBinaryDims) -> proto::multi_binary_spec::N {
    match value {
        native::MultiBinaryDims::Size(size) => proto::multi_binary_spec::N::Size(*size),
        native::MultiBinaryDims::Dims(dims) => {
            proto::multi_binary_spec::N::Dims(proto::VectorInt { data: dims.clone() })
        }
    }
}

fn multibinary_n_from_proto(
    value: proto::multi_binary_spec::N,
) -> Result<native::MultiBinaryDims, ProtocolError> {
    Ok(match value {
        proto::multi_binary_spec::N::Size(size) => native::MultiBinaryDims::Size(size),
        proto::multi_binary_spec::N::Dims(dims) => native::MultiBinaryDims::Dims(dims.data),
    })
}

fn multidiscrete_nvec_to_proto(
    value: &native::MultiDiscreteNvec,
) -> proto::multi_discrete_spec::Nvec {
    match value {
        native::MultiDiscreteNvec::Flat(vector) => {
            proto::multi_discrete_spec::Nvec::Flat(proto::VectorInt {
                data: vector.clone(),
            })
        }
        native::MultiDiscreteNvec::Shaped(matrix) => {
            proto::multi_discrete_spec::Nvec::Shaped(proto::MatrixInt {
                data: matrix
                    .iter()
                    .map(|row| proto::VectorInt { data: row.clone() })
                    .collect(),
            })
        }
    }
}

fn multidiscrete_nvec_from_proto(
    value: proto::multi_discrete_spec::Nvec,
) -> Result<native::MultiDiscreteNvec, ProtocolError> {
    Ok(match value {
        proto::multi_discrete_spec::Nvec::Flat(vector) => {
            native::MultiDiscreteNvec::Flat(vector.data)
        }
        proto::multi_discrete_spec::Nvec::Shaped(matrix) => {
            native::MultiDiscreteNvec::Shaped(matrix.data.into_iter().map(|row| row.data).collect())
        }
    })
}

pub fn meta_map_to_proto(value: &native::MetaMap) -> proto::MetaMap {
    proto::MetaMap {
        entries: value
            .iter()
            .map(|(key, value)| (key.clone(), meta_value_to_proto(value)))
            .collect(),
    }
}

pub fn meta_map_from_proto(value: proto::MetaMap) -> native::MetaMap {
    value
        .entries
        .into_iter()
        .map(|(key, value)| (key, meta_value_from_proto(value)))
        .collect::<BTreeMap<_, _>>()
}

pub(crate) fn meta_value_to_proto(value: &native::MetaValue) -> proto::MetaValue {
    // A null/None value is encoded as a MetaValue with no oneof set.
    let kind = match value {
        native::MetaValue::Null => None,
        native::MetaValue::Bool(value) => Some(MetaKind::Bool(*value)),
        native::MetaValue::Int(value) => Some(MetaKind::Int(*value)),
        native::MetaValue::Float(value) => Some(MetaKind::Float(*value)),
        native::MetaValue::String(value) => Some(MetaKind::Str(value.clone())),
        native::MetaValue::Bytes(value) => Some(MetaKind::Bytes(value.clone())),
        native::MetaValue::List(value) => Some(MetaKind::List(proto::MetaList {
            items: value.iter().map(meta_value_to_proto).collect(),
        })),
        native::MetaValue::Map(value) => Some(MetaKind::Map(meta_map_to_proto(value))),
    };
    proto::MetaValue { kind }
}

pub(crate) fn meta_value_from_proto(value: proto::MetaValue) -> native::MetaValue {
    match value.kind {
        None => native::MetaValue::Null,
        Some(MetaKind::Bool(value)) => native::MetaValue::Bool(value),
        Some(MetaKind::Int(value)) => native::MetaValue::Int(value),
        Some(MetaKind::Float(value)) => native::MetaValue::Float(value),
        Some(MetaKind::Str(value)) => native::MetaValue::String(value),
        Some(MetaKind::Bytes(value)) => native::MetaValue::Bytes(value),
        Some(MetaKind::List(value)) => {
            native::MetaValue::List(value.items.into_iter().map(meta_value_from_proto).collect())
        }
        Some(MetaKind::Map(value)) => native::MetaValue::Map(meta_map_from_proto(value)),
    }
}

fn proto_dtype_from_native(dtype: native::DType) -> proto::DType {
    match dtype {
        native::DType::Unspecified => proto::DType::Unspecified,
        native::DType::Bool => proto::DType::Bool,
        native::DType::Uint8 => proto::DType::Uint8,
        native::DType::Int32 => proto::DType::Int32,
        native::DType::Int64 => proto::DType::Int64,
        native::DType::Float16 => proto::DType::Float16,
        native::DType::Float32 => proto::DType::Float32,
        native::DType::Float64 => proto::DType::Float64,
        native::DType::Int8 => proto::DType::Int8,
        native::DType::Int16 => proto::DType::Int16,
        native::DType::Uint16 => proto::DType::Uint16,
        native::DType::Uint32 => proto::DType::Uint32,
        native::DType::Uint64 => proto::DType::Uint64,
        native::DType::Bfloat16 => proto::DType::Bfloat16,
    }
}

fn native_dtype_from_proto(dtype: i32) -> Result<native::DType, ProtocolError> {
    let dtype = proto::DType::try_from(dtype)
        .map_err(|_| ProtocolError::DecodeError(format!("unknown proto dtype value: {dtype}")))?;
    Ok(match dtype {
        proto::DType::Unspecified => native::DType::Unspecified,
        proto::DType::Bool => native::DType::Bool,
        proto::DType::Uint8 => native::DType::Uint8,
        proto::DType::Int32 => native::DType::Int32,
        proto::DType::Int64 => native::DType::Int64,
        proto::DType::Float16 => native::DType::Float16,
        proto::DType::Float32 => native::DType::Float32,
        proto::DType::Float64 => native::DType::Float64,
        proto::DType::Int8 => native::DType::Int8,
        proto::DType::Int16 => native::DType::Int16,
        proto::DType::Uint16 => native::DType::Uint16,
        proto::DType::Uint32 => native::DType::Uint32,
        proto::DType::Uint64 => native::DType::Uint64,
        proto::DType::Bfloat16 => native::DType::Bfloat16,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_proto_roundtrip() {
        let all = [
            native::DType::Unspecified,
            native::DType::Bool,
            native::DType::Uint8,
            native::DType::Int32,
            native::DType::Int64,
            native::DType::Float16,
            native::DType::Float32,
            native::DType::Float64,
            native::DType::Int8,
            native::DType::Int16,
            native::DType::Uint16,
            native::DType::Uint32,
            native::DType::Uint64,
            native::DType::Bfloat16,
        ];
        for dtype in all {
            let wire = proto_dtype_from_native(dtype) as i32;
            assert_eq!(
                native_dtype_from_proto(wire).expect("known dtype"),
                dtype,
                "roundtrip mismatch for {dtype:?}"
            );
        }
    }

    #[test]
    fn test_dtype_from_proto_rejects_unknown_value() {
        assert!(native_dtype_from_proto(999).is_err());
    }

    fn box_spec(
        dtype: native::DType,
        bounds: native::BoxBounds,
        shape: Vec<i64>,
    ) -> native::SpaceSpec {
        native::SpaceSpec {
            shape,
            dtype,
            spec: Some(native::SpaceKind::Box(native::BoxSpec {
                bounds: Some(bounds),
            })),
        }
    }

    fn roundtrip(spec: &native::SpaceSpec) -> native::SpaceSpec {
        space_spec_from_proto(space_spec_to_proto(spec)).expect("decodes")
    }

    #[test]
    fn test_box_bounds_proto_roundtrip_all_variants() {
        let cases = [
            box_spec(
                native::DType::Float32,
                native::BoxBounds::Unbounded(true),
                vec![3],
            ),
            box_spec(
                native::DType::Float32,
                native::BoxBounds::Uniform(native::UniformBounds {
                    low: -1.0,
                    high: 1.0,
                }),
                vec![3],
            ),
            box_spec(
                native::DType::Float32,
                native::BoxBounds::Elementwise(native::ElementwiseBounds {
                    low: vec![0.0, 1.0],
                    high: vec![2.0, 3.0],
                }),
                vec![2],
            ),
            box_spec(
                native::DType::Int64,
                native::BoxBounds::TypedUniform(native::TypedUniformBounds {
                    low: i64::MIN.to_le_bytes().to_vec(),
                    high: i64::MAX.to_le_bytes().to_vec(),
                }),
                vec![4],
            ),
            box_spec(
                native::DType::Uint64,
                native::BoxBounds::TypedElementwise(native::TypedElementwiseBounds {
                    low: [0u64, 1].iter().flat_map(|v| v.to_le_bytes()).collect(),
                    high: [u64::MAX, 9].iter().flat_map(|v| v.to_le_bytes()).collect(),
                }),
                vec![2],
            ),
        ];
        for spec in cases {
            assert_eq!(roundtrip(&spec), spec, "roundtrip mismatch for {spec:?}");
        }
    }

    #[test]
    fn test_typed_bounds_bytes_survive_the_wire() {
        // The raw bytes of an i64::MAX bound must be preserved exactly; an f64
        // path would have rounded them.
        let spec = box_spec(
            native::DType::Int64,
            native::BoxBounds::TypedUniform(native::TypedUniformBounds {
                low: (i64::MAX - 1).to_le_bytes().to_vec(),
                high: i64::MAX.to_le_bytes().to_vec(),
            }),
            vec![1],
        );
        let decoded = roundtrip(&spec);
        let native::SpaceKind::Box(b) = decoded.spec.unwrap() else {
            panic!("expected Box");
        };
        let native::BoxBounds::TypedUniform(t) = b.bounds.unwrap() else {
            panic!("expected typed-uniform bounds");
        };
        assert_eq!(t.high, i64::MAX.to_le_bytes());
        assert_eq!(t.low, (i64::MAX - 1).to_le_bytes());
    }

    fn meta_roundtrip(value: native::MetaValue) -> native::MetaValue {
        meta_value_from_proto(meta_value_to_proto(&value))
    }

    #[test]
    fn meta_int_beyond_two_pow_53_survives_exactly() {
        // The old Struct path rode Int through f64 and corrupted |v| > 2^53.
        let value = native::MetaValue::Int((1i64 << 53) + 1);
        assert_eq!(meta_roundtrip(value.clone()), value);

        let value = native::MetaValue::Int(i64::MAX);
        assert_eq!(meta_roundtrip(value.clone()), value);
        let value = native::MetaValue::Int(i64::MIN);
        assert_eq!(meta_roundtrip(value.clone()), value);
    }

    #[test]
    fn meta_whole_number_float_stays_float() {
        // The old decode reclassified any whole-number Float as Int.
        let value = native::MetaValue::Float(2.0);
        assert_eq!(meta_roundtrip(value.clone()), value);
        assert!(matches!(
            meta_roundtrip(native::MetaValue::Float(2.0)),
            native::MetaValue::Float(_)
        ));
    }

    #[test]
    fn meta_preserves_bool_str_bytes_null() {
        for value in [
            native::MetaValue::Null,
            native::MetaValue::Bool(true),
            native::MetaValue::Bool(false),
            native::MetaValue::String("hello".to_string()),
            native::MetaValue::Bytes(vec![0, 1, 2, 255]),
        ] {
            assert_eq!(meta_roundtrip(value.clone()), value);
        }
    }

    #[test]
    fn meta_nested_list_and_map_roundtrip() {
        let mut map = native::MetaMap::new();
        map.insert("big".to_string(), native::MetaValue::Int((1i64 << 53) + 1));
        map.insert("flag".to_string(), native::MetaValue::Bool(true));
        map.insert(
            "list".to_string(),
            native::MetaValue::List(vec![
                native::MetaValue::Float(2.0),
                native::MetaValue::Bytes(vec![7, 8]),
                native::MetaValue::Null,
            ]),
        );
        let value = native::MetaValue::Map(map.clone());
        assert_eq!(meta_roundtrip(value), native::MetaValue::Map(map.clone()));

        // And as a whole MetaMap channel.
        assert_eq!(meta_map_from_proto(meta_map_to_proto(&map)), map);
    }
}
