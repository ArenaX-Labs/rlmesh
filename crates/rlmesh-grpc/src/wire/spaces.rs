use std::collections::BTreeMap;

use prost_types::{ListValue, Struct, Value, value};
use rlmesh_proto::env::v1 as env_proto;
use rlmesh_proto::spaces::v1 as proto;
use rlmesh_spaces::v1 as native;

use crate::error::ProtocolError;

pub fn env_contract_to_proto(spec: &native::EnvContract) -> env_proto::EnvContract {
    env_proto::EnvContract {
        id: spec.id.clone(),
        action_space: spec.action_space.as_ref().map(space_spec_to_proto),
        observation_space: spec.observation_space.as_ref().map(space_spec_to_proto),
        metadata: spec.metadata.as_ref().map(meta_map_to_struct),
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
        metadata: spec.metadata.map(meta_map_from_struct),
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

fn box_bounds_to_proto(bounds: &native::box_spec::Bounds) -> proto::box_spec::Bounds {
    match bounds {
        native::box_spec::Bounds::Unbounded(value) => proto::box_spec::Bounds::Unbounded(*value),
        native::box_spec::Bounds::Uniform(bounds) => {
            proto::box_spec::Bounds::Uniform(proto::UniformBounds {
                low: bounds.low,
                high: bounds.high,
            })
        }
        native::box_spec::Bounds::Axiswise(bounds) => {
            proto::box_spec::Bounds::Axiswise(proto::AxiswiseBounds {
                low: bounds.low.clone(),
                high: bounds.high.clone(),
            })
        }
        native::box_spec::Bounds::Elementwise(bounds) => {
            proto::box_spec::Bounds::Elementwise(proto::ElementwiseBounds {
                low: bounds.low.clone(),
                high: bounds.high.clone(),
            })
        }
    }
}

fn box_bounds_from_proto(
    bounds: proto::box_spec::Bounds,
) -> Result<native::box_spec::Bounds, ProtocolError> {
    Ok(match bounds {
        proto::box_spec::Bounds::Unbounded(value) => native::box_spec::Bounds::Unbounded(value),
        proto::box_spec::Bounds::Uniform(bounds) => {
            native::box_spec::Bounds::Uniform(native::UniformBounds {
                low: bounds.low,
                high: bounds.high,
            })
        }
        proto::box_spec::Bounds::Axiswise(bounds) => {
            native::box_spec::Bounds::Axiswise(native::AxiswiseBounds {
                low: bounds.low,
                high: bounds.high,
            })
        }
        proto::box_spec::Bounds::Elementwise(bounds) => {
            native::box_spec::Bounds::Elementwise(native::ElementwiseBounds {
                low: bounds.low,
                high: bounds.high,
            })
        }
    })
}

fn multibinary_n_to_proto(value: &native::multi_binary_spec::N) -> proto::multi_binary_spec::N {
    match value {
        native::multi_binary_spec::N::Size(size) => proto::multi_binary_spec::N::Size(*size),
        native::multi_binary_spec::N::Dims(dims) => {
            proto::multi_binary_spec::N::Dims(proto::VectorInt {
                data: dims.data.clone(),
            })
        }
    }
}

fn multibinary_n_from_proto(
    value: proto::multi_binary_spec::N,
) -> Result<native::multi_binary_spec::N, ProtocolError> {
    Ok(match value {
        proto::multi_binary_spec::N::Size(size) => native::multi_binary_spec::N::Size(size),
        proto::multi_binary_spec::N::Dims(dims) => {
            native::multi_binary_spec::N::Dims(native::VectorInt { data: dims.data })
        }
    })
}

fn multidiscrete_nvec_to_proto(
    value: &native::multi_discrete_spec::Nvec,
) -> proto::multi_discrete_spec::Nvec {
    match value {
        native::multi_discrete_spec::Nvec::Flat(vector) => {
            proto::multi_discrete_spec::Nvec::Flat(proto::VectorInt {
                data: vector.data.clone(),
            })
        }
        native::multi_discrete_spec::Nvec::Shaped(matrix) => {
            proto::multi_discrete_spec::Nvec::Shaped(proto::MatrixInt {
                data: matrix
                    .data
                    .iter()
                    .map(|row| proto::VectorInt {
                        data: row.data.clone(),
                    })
                    .collect(),
            })
        }
    }
}

fn multidiscrete_nvec_from_proto(
    value: proto::multi_discrete_spec::Nvec,
) -> Result<native::multi_discrete_spec::Nvec, ProtocolError> {
    Ok(match value {
        proto::multi_discrete_spec::Nvec::Flat(vector) => {
            native::multi_discrete_spec::Nvec::Flat(native::VectorInt { data: vector.data })
        }
        proto::multi_discrete_spec::Nvec::Shaped(matrix) => {
            native::multi_discrete_spec::Nvec::Shaped(native::MatrixInt {
                data: matrix
                    .data
                    .into_iter()
                    .map(|row| native::VectorInt { data: row.data })
                    .collect(),
            })
        }
    })
}

pub fn meta_map_to_struct(value: &native::MetaMap) -> Struct {
    Struct {
        fields: value
            .iter()
            .map(|(key, value)| (key.clone(), meta_value_to_proto(value)))
            .collect(),
    }
}

pub fn meta_map_from_struct(value: Struct) -> native::MetaMap {
    value
        .fields
        .into_iter()
        .map(|(key, value)| (key, meta_value_from_proto(value)))
        .collect::<BTreeMap<_, _>>()
}

pub(crate) fn meta_value_to_proto(value: &native::MetaValue) -> Value {
    let kind = match value {
        native::MetaValue::Null => value::Kind::NullValue(0),
        native::MetaValue::Bool(value) => value::Kind::BoolValue(*value),
        native::MetaValue::Int(value) => value::Kind::NumberValue(*value as f64),
        native::MetaValue::Float(value) => value::Kind::NumberValue(*value),
        native::MetaValue::String(value) => value::Kind::StringValue(value.clone()),
        native::MetaValue::List(value) => value::Kind::ListValue(ListValue {
            values: value.iter().map(meta_value_to_proto).collect(),
        }),
        native::MetaValue::Map(value) => value::Kind::StructValue(meta_map_to_struct(value)),
    };
    Value { kind: Some(kind) }
}

pub(crate) fn meta_value_from_proto(value: Value) -> native::MetaValue {
    match value.kind {
        Some(value::Kind::NullValue(_)) | None => native::MetaValue::Null,
        Some(value::Kind::BoolValue(value)) => native::MetaValue::Bool(value),
        Some(value::Kind::NumberValue(value)) => {
            if value.fract() == 0.0 && value.is_finite() {
                native::MetaValue::Int(value as i64)
            } else {
                native::MetaValue::Float(value)
            }
        }
        Some(value::Kind::StringValue(value)) => native::MetaValue::String(value),
        Some(value::Kind::ListValue(value)) => native::MetaValue::List(
            value
                .values
                .into_iter()
                .map(meta_value_from_proto)
                .collect(),
        ),
        Some(value::Kind::StructValue(value)) => {
            native::MetaValue::Map(meta_map_from_struct(value))
        }
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
}
