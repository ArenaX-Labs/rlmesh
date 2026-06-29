//! Conversions between native space/contract types and their proto form.
//!
//! Env contracts, space specs (with Box bounds), and metadata maps cross the
//! wire here. The native types stay flat; the nesting lives in this codec. A
//! peer-supplied space spec is validated at decode (handshake time) so a
//! malformed space fails fast rather than at first use.

use std::collections::BTreeMap;

use rlmesh_proto::core::v1 as core_proto;
use rlmesh_proto::spaces::v1 as proto;
use rlmesh_proto::spaces::v1::meta_value::Kind as MetaKind;
use rlmesh_spaces as native;

use crate::error::ProtocolError;

/// Map the stable spec slice (id + spaces + metadata) of a native FLAT
/// [`native::EnvContract`] into the nested proto [`core_proto::EnvSpec`].
pub fn env_spec_to_proto(spec: &native::EnvContract) -> core_proto::EnvSpec {
    core_proto::EnvSpec {
        id: spec.id.clone(),
        action_space: spec.action_space.as_ref().map(space_spec_to_proto),
        observation_space: spec.observation_space.as_ref().map(space_spec_to_proto),
        metadata: spec.metadata.as_ref().map(meta_map_to_proto),
    }
}

/// Map a proto [`core_proto::EnvSpec`] into a native FLAT [`native::EnvContract`].
///
/// `EnvSpec` carries only the stable observation/action interface; the
/// orchestration knobs the model ignores (`num_envs`/`render_mode`/
/// `autoreset_mode`) are left at their native defaults, matching the model's
/// existing behavior of ignoring them.
pub fn env_spec_from_proto(
    spec: core_proto::EnvSpec,
) -> Result<native::EnvContract, ProtocolError> {
    Ok(native::EnvContract {
        id: spec.id,
        action_space: spec.action_space.map(space_spec_from_proto).transpose()?,
        observation_space: spec
            .observation_space
            .map(space_spec_from_proto)
            .transpose()?,
        metadata: spec.metadata.map(meta_map_from_proto),
        // The model never reads these; default them.
        render_mode: String::new(),
        num_envs: 0,
        autoreset_mode: native::AutoresetMode::default(),
    })
}

/// Flatten a native FLAT [`native::EnvContract`] into the nested proto
/// [`core_proto::EnvContract`] (the nesting lives entirely in the wire codec;
/// the native type stays flat for all consumers).
pub fn env_contract_to_proto(spec: &native::EnvContract) -> core_proto::EnvContract {
    core_proto::EnvContract {
        spec: Some(env_spec_to_proto(spec)),
        num_envs: spec.num_envs,
        render_mode: spec.render_mode.clone(),
        autoreset_mode: i32::from(spec.autoreset_mode),
        // Native `EnvContract` has no tags channel yet; leave the optional proto
        // field unset until a native source exists.
        tags: None,
    }
}

/// Reassemble a native FLAT [`native::EnvContract`] from the nested proto
/// [`core_proto::EnvContract`].
pub fn env_contract_from_proto(
    contract: core_proto::EnvContract,
) -> Result<native::EnvContract, ProtocolError> {
    // `env_spec_from_proto` already builds the full contract (id + spaces +
    // metadata); overlay only the orchestration knobs the spec leaves at their
    // defaults, so a new `EnvSpec` field need not be re-listed here.
    Ok(native::EnvContract {
        render_mode: contract.render_mode,
        num_envs: contract.num_envs,
        // proto UNSPECIFIED/DISABLED decode to Disabled; an unknown mode is
        // rejected loudly rather than silently folded.
        autoreset_mode: native::AutoresetMode::try_from(contract.autoreset_mode)
            .map_err(|e| ProtocolError::DecodeError(e.to_string()))?,
        ..env_spec_from_proto(contract.spec.unwrap_or_default())?
    })
}

pub fn space_spec_to_proto(spec: &native::SpaceSpec) -> proto::SpaceSpec {
    proto::SpaceSpec {
        shape: spec.shape.clone(),
        dtype: proto_dtype_from_native(spec.dtype) as i32,
        spec: spec
            .spec
            .as_ref()
            .map(|kind| space_kind_to_proto(kind, spec.dtype)),
    }
}

pub fn space_spec_from_proto(spec: proto::SpaceSpec) -> Result<native::SpaceSpec, ProtocolError> {
    let dtype = native_dtype_from_proto(spec.dtype)?;
    let shape = spec.shape;
    let kind = spec
        .spec
        .map(|kind| space_kind_from_proto(kind, dtype, &shape))
        .transpose()?;
    let spec = native::SpaceSpec {
        shape,
        dtype,
        spec: kind,
    };
    // Hold a peer-supplied spec to the same invariants the local builders
    // enforce (positive shape dims, Dict key/space parity, MultiBinary/
    // MultiDiscrete dtype rules, Box bound integrality). Without this, a
    // malformed spec only surfaces at the first `contains()` -- or, worse,
    // panics on an unchecked shape product. Validating at wire decode fails
    // fast at handshake instead.
    native::validate_space(&spec).map_err(|err| ProtocolError::DecodeError(err.to_string()))?;
    Ok(spec)
}

/// Fail fast on malformed Box bounds at wire decode (handshake time) instead of
/// only erroring on the first `contains()`. Bounds bytes are little-endian
/// scalars in the space's dtype: one scalar for uniform, `numel` elementwise.
fn validate_bound_byte_lengths(
    low: &[u8],
    high: &[u8],
    count: usize,
    dtype: native::DType,
) -> Result<(), ProtocolError> {
    let elem = native::dtype_size(dtype);
    if elem == 0 {
        return Err(ProtocolError::DecodeError(
            "Box bounds require a concrete dtype".to_string(),
        ));
    }
    let expected = count.checked_mul(elem).ok_or_else(|| {
        ProtocolError::DecodeError("Box bounds byte length overflowed".to_string())
    })?;
    for (name, bytes) in [("low", low), ("high", high)] {
        if bytes.len() != expected {
            return Err(ProtocolError::DecodeError(format!(
                "Box bounds `{name}` carries {} bytes; expected {expected} \
                 ({count} {dtype:?} scalar(s))",
                bytes.len(),
            )));
        }
    }
    Ok(())
}

fn checked_box_shape_numel(shape: &[i64]) -> Result<usize, ProtocolError> {
    if shape.is_empty() {
        return Err(ProtocolError::DecodeError(
            "typed elementwise Box bounds require a non-empty shape".to_string(),
        ));
    }
    shape.iter().enumerate().try_fold(1usize, |acc, (i, &dim)| {
        if dim <= 0 {
            return Err(ProtocolError::DecodeError(format!(
                "typed elementwise Box bounds shape[{i}] must be > 0"
            )));
        }
        acc.checked_mul(dim as usize).ok_or_else(|| {
            ProtocolError::DecodeError(
                "typed elementwise Box bounds shape product overflowed".to_string(),
            )
        })
    })
}

fn space_kind_to_proto(kind: &native::SpaceKind, dtype: native::DType) -> proto::space_spec::Spec {
    match kind {
        native::SpaceKind::Box(spec) => proto::space_spec::Spec::Box(proto::BoxSpec {
            bounds: spec.bounds.as_ref().map(|b| box_bounds_to_proto(b, dtype)),
        }),
        native::SpaceKind::Discrete(spec) => {
            proto::space_spec::Spec::Discrete(proto::DiscreteSpec {
                n: spec.n,
                start: spec.start,
            })
        }
        native::SpaceKind::MultiBinary(_) => {
            proto::space_spec::Spec::MultiBinary(proto::MultiBinarySpec {})
        }
        native::SpaceKind::MultiDiscrete(spec) => {
            proto::space_spec::Spec::MultiDiscrete(proto::MultiDiscreteSpec {
                nvec: spec.nvec.clone(),
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
    dtype: native::DType,
    shape: &[i64],
) -> Result<native::SpaceKind, ProtocolError> {
    Ok(match kind {
        proto::space_spec::Spec::Box(spec) => native::SpaceKind::Box(native::BoxSpec {
            bounds: spec
                .bounds
                .map(|b| box_bounds_from_proto(b, dtype, shape))
                .transpose()?,
        }),
        proto::space_spec::Spec::Discrete(spec) => {
            native::SpaceKind::Discrete(native::DiscreteSpec {
                n: spec.n,
                start: spec.start,
            })
        }
        proto::space_spec::Spec::MultiBinary(_) => {
            native::SpaceKind::MultiBinary(native::MultiBinarySpec)
        }
        proto::space_spec::Spec::MultiDiscrete(spec) => {
            native::SpaceKind::MultiDiscrete(native::MultiDiscreteSpec { nvec: spec.nvec })
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

/// Map native [`BoxBounds`](native::BoxBounds) onto the wire `BoxSpec.bounds`.
///
/// The native type has FOUR bound variants but the wire has only TWO arms
/// (`Uniform`/`Elementwise`), both carrying raw little-endian dtype bytes:
/// - `Uniform`/`Elementwise` hold `f64` and are encoded into the dtype's byte
///   width here (`encode_float_bound`), losing nothing for float dtypes and
///   round-tripping integer dtypes whose float-form bound is representable
///   (out-of-range/fractional bounds are rejected at construction in
///   `rlmesh-spaces`, so they never reach this point).
/// - `TypedUniform`/`TypedElementwise` already hold exact dtype bytes and pass
///   through verbatim.
///
/// The decoder ([`box_bounds_from_proto`]) reverses this by `dtype.is_float()`:
/// a float dtype decodes back to `Uniform`/`Elementwise` (`f64`), an integer
/// dtype to `TypedUniform`/`TypedElementwise` (bytes). So a float-form bound on
/// an integer dtype intentionally round-trips as a `Typed*` variant, not its
/// original float variant — the bytes, not the Rust variant, are the contract.
fn box_bounds_to_proto(
    bounds: &native::BoxBounds,
    dtype: native::DType,
) -> proto::box_spec::Bounds {
    match bounds {
        native::BoxBounds::Unbounded(value) => proto::box_spec::Bounds::Unbounded(*value),
        // Float bounds are encoded into the space's dtype as little-endian bytes.
        native::BoxBounds::Uniform(b) => proto::box_spec::Bounds::Uniform(proto::UniformBounds {
            low: encode_float_bound(b.low, dtype),
            high: encode_float_bound(b.high, dtype),
        }),
        native::BoxBounds::Elementwise(b) => {
            proto::box_spec::Bounds::Elementwise(proto::ElementwiseBounds {
                low: encode_float_bounds(&b.low, dtype),
                high: encode_float_bounds(&b.high, dtype),
            })
        }
        // Integer bounds already carry exact dtype bytes; pass them through.
        native::BoxBounds::TypedUniform(b) => {
            proto::box_spec::Bounds::Uniform(proto::UniformBounds {
                low: b.low.clone(),
                high: b.high.clone(),
            })
        }
        native::BoxBounds::TypedElementwise(b) => {
            proto::box_spec::Bounds::Elementwise(proto::ElementwiseBounds {
                low: b.low.clone(),
                high: b.high.clone(),
            })
        }
    }
}

fn box_bounds_from_proto(
    bounds: proto::box_spec::Bounds,
    dtype: native::DType,
    shape: &[i64],
) -> Result<native::BoxBounds, ProtocolError> {
    Ok(match bounds {
        proto::box_spec::Bounds::Unbounded(value) => native::BoxBounds::Unbounded(value),
        proto::box_spec::Bounds::Uniform(b) => {
            validate_bound_byte_lengths(&b.low, &b.high, 1, dtype)?;
            if dtype.is_float() {
                native::BoxBounds::Uniform(native::UniformBounds {
                    low: decode_float_bound(&b.low, dtype),
                    high: decode_float_bound(&b.high, dtype),
                })
            } else {
                native::BoxBounds::TypedUniform(native::TypedUniformBounds {
                    low: b.low,
                    high: b.high,
                })
            }
        }
        proto::box_spec::Bounds::Elementwise(b) => {
            let numel = checked_box_shape_numel(shape)?;
            validate_bound_byte_lengths(&b.low, &b.high, numel, dtype)?;
            if dtype.is_float() {
                native::BoxBounds::Elementwise(native::ElementwiseBounds {
                    low: decode_float_bounds(&b.low, dtype),
                    high: decode_float_bounds(&b.high, dtype),
                })
            } else {
                native::BoxBounds::TypedElementwise(native::TypedElementwiseBounds {
                    low: b.low,
                    high: b.high,
                })
            }
        }
    })
}

/// Encode a single float `Uniform`/`Elementwise` bound into the space's dtype
/// as little-endian bytes, exactly `dtype_size(dtype)` wide.
///
/// Float dtypes carry the value in their own width. An integer/bool dtype Box
/// can still hold a float-form bound (e.g. `BoxSpaceBuilder::scalar(0.0, 1.0,
/// ..).dtype(Uint8)`); such a bound must be packed at the integer dtype's width
/// so the bytes-only wire round-trips it (the decoder reads it back as a
/// `TypedUniform`/`TypedElementwise`). Float→int casts saturate (Rust `as`), so
/// an infinite half-open side clamps to the dtype's min/max rather than panics.
fn encode_float_bound(value: f64, dtype: native::DType) -> Vec<u8> {
    use native::DType;
    match dtype {
        DType::Float16 => half::f16::from_f64(value).to_le_bytes().to_vec(),
        DType::Float32 => (value as f32).to_le_bytes().to_vec(),
        DType::Float64 | DType::Unspecified => value.to_le_bytes().to_vec(),
        DType::Bool => vec![u8::from(value != 0.0)],
        DType::Uint8 => (value as u8).to_le_bytes().to_vec(),
        DType::Uint16 => (value as u16).to_le_bytes().to_vec(),
        DType::Uint32 => (value as u32).to_le_bytes().to_vec(),
        DType::Uint64 => (value as u64).to_le_bytes().to_vec(),
        DType::Int8 => (value as i8).to_le_bytes().to_vec(),
        DType::Int16 => (value as i16).to_le_bytes().to_vec(),
        DType::Int32 => (value as i32).to_le_bytes().to_vec(),
        DType::Int64 => (value as i64).to_le_bytes().to_vec(),
    }
}

fn encode_float_bounds(values: &[f64], dtype: native::DType) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| encode_float_bound(*value, dtype))
        .collect()
}

/// Decode one float bound from `dtype_size(dtype)` little-endian bytes. Callers
/// must validate the byte length first (see `validate_bound_byte_lengths`).
fn decode_float_bound(bytes: &[u8], dtype: native::DType) -> f64 {
    match dtype {
        native::DType::Float16 => half::f16::from_le_bytes([bytes[0], bytes[1]]).to_f64(),
        native::DType::Float32 => {
            f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64
        }
        _ => f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]),
    }
}

fn decode_float_bounds(bytes: &[u8], dtype: native::DType) -> Vec<f64> {
    let elem = native::dtype_size(dtype);
    bytes
        .chunks_exact(elem)
        .map(|chunk| decode_float_bound(chunk, dtype))
        .collect()
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
        native::MetaValue::Int(value) => Some(MetaKind::Integer(*value)),
        native::MetaValue::Float(value) => Some(MetaKind::Number(*value)),
        native::MetaValue::String(value) => Some(MetaKind::Text(value.clone())),
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
        Some(MetaKind::Integer(value)) => native::MetaValue::Int(value),
        Some(MetaKind::Number(value)) => native::MetaValue::Float(value),
        Some(MetaKind::Text(value)) => native::MetaValue::String(value),
        Some(MetaKind::Bytes(value)) => native::MetaValue::Bytes(value),
        Some(MetaKind::List(value)) => {
            native::MetaValue::List(value.items.into_iter().map(meta_value_from_proto).collect())
        }
        Some(MetaKind::Map(value)) => native::MetaValue::Map(meta_map_from_proto(value)),
    }
}

// Compile-time guard that the native `DType` discriminants stay byte-identical
// to the generated proto `DType` enum. The bridge fns below map by variant, but
// every `as i32` cast on a dtype relies on the numbers agreeing; this block
// fails the build the instant they drift. `rlmesh-spaces` stays proto-free, so
// the assert lives here, where both enums are in scope.
const _: () = {
    assert!(native::DType::Unspecified as i32 == proto::DType::Unspecified as i32);
    assert!(native::DType::Bool as i32 == proto::DType::Bool as i32);
    assert!(native::DType::Uint8 as i32 == proto::DType::Uint8 as i32);
    assert!(native::DType::Uint16 as i32 == proto::DType::Uint16 as i32);
    assert!(native::DType::Uint32 as i32 == proto::DType::Uint32 as i32);
    assert!(native::DType::Uint64 as i32 == proto::DType::Uint64 as i32);
    assert!(native::DType::Int8 as i32 == proto::DType::Int8 as i32);
    assert!(native::DType::Int16 as i32 == proto::DType::Int16 as i32);
    assert!(native::DType::Int32 as i32 == proto::DType::Int32 as i32);
    assert!(native::DType::Int64 as i32 == proto::DType::Int64 as i32);
    assert!(native::DType::Float16 as i32 == proto::DType::Float16 as i32);
    assert!(native::DType::Float32 as i32 == proto::DType::Float32 as i32);
    assert!(native::DType::Float64 as i32 == proto::DType::Float64 as i32);
};

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
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn autoreset_mode_native_and_proto_agree() {
        use rlmesh_proto::core::v1::AutoresetMode as Proto;
        use rlmesh_spaces::AutoresetMode as Native;
        // Known values decode identically; proto UNSPECIFIED (0) and DISABLED (3)
        // both map to the native DISABLED safe default.
        assert_eq!(Native::try_from(0), Ok(Native::Disabled));
        assert_eq!(Native::try_from(1), Ok(Native::NextStep));
        assert_eq!(Native::try_from(2), Ok(Native::SameStep));
        assert_eq!(Native::try_from(3), Ok(Native::Disabled));
        assert_eq!(Proto::try_from(0), Ok(Proto::Unspecified));
        assert_eq!(Proto::try_from(1), Ok(Proto::NextStep));
        // Both reject the same unknown values loudly; no silent fold.
        for unknown in [4, 5, 99, -1] {
            assert!(Native::try_from(unknown).is_err());
            assert!(Proto::try_from(unknown).is_err());
        }
    }

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
    fn integer_dtype_float_uniform_bound_encodes_at_dtype_width() {
        // A float-form Uniform bound on an integer Box (e.g.
        // `BoxSpaceBuilder::scalar(0.0, 1.0, ..).dtype(Uint8)`) must pack at the
        // dtype's width (1 byte for Uint8), not as raw 8-byte f64, so the
        // bytes-only wire round-trips it instead of failing length validation.
        let spec = box_spec(
            native::DType::Uint8,
            native::BoxBounds::Uniform(native::UniformBounds {
                low: 0.0,
                high: 1.0,
            }),
            vec![1],
        );
        let decoded = roundtrip(&spec);
        let native::SpaceKind::Box(b) = decoded.spec.unwrap() else {
            panic!("expected Box");
        };
        // An integer dtype decodes the byte-form bound back as TypedUniform.
        let native::BoxBounds::TypedUniform(t) = b.bounds.unwrap() else {
            panic!("expected typed-uniform bounds for an integer dtype");
        };
        assert_eq!(t.low, vec![0u8]);
        assert_eq!(t.high, vec![1u8]);
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

    #[test]
    fn malformed_typed_bounds_fail_at_wire_decode() {
        // A typed bound with the wrong byte length must be rejected at spec
        // decode (handshake time), not deferred to the first contains().
        let spec = box_spec(
            native::DType::Int64,
            native::BoxBounds::TypedUniform(native::TypedUniformBounds {
                low: vec![0u8; 4], // 4 bytes, but Int64 needs 8
                high: i64::MAX.to_le_bytes().to_vec(),
            }),
            vec![2],
        );
        let err = space_spec_from_proto(space_spec_to_proto(&spec)).expect_err("must fail fast");
        assert!(err.to_string().contains("Box bounds"));
    }

    #[test]
    fn malformed_typed_elementwise_bounds_reject_negative_shape() {
        let spec = box_spec(
            native::DType::Int64,
            native::BoxBounds::TypedElementwise(native::TypedElementwiseBounds {
                low: 0i64.to_le_bytes().to_vec(),
                high: 1i64.to_le_bytes().to_vec(),
            }),
            vec![-1],
        );
        let err = space_spec_from_proto(space_spec_to_proto(&spec)).expect_err("must fail fast");
        assert!(err.to_string().contains("shape[0] must be > 0"));
    }

    #[test]
    fn malformed_typed_elementwise_bounds_reject_shape_product_overflow() {
        let spec = box_spec(
            native::DType::Int64,
            native::BoxBounds::TypedElementwise(native::TypedElementwiseBounds {
                low: Vec::new(),
                high: Vec::new(),
            }),
            vec![i64::MAX, 3],
        );
        let err = space_spec_from_proto(space_spec_to_proto(&spec)).expect_err("must fail fast");
        assert!(err.to_string().contains("shape product overflowed"));
    }

    #[test]
    fn malformed_typed_elementwise_bounds_reject_byte_length_overflow() {
        let spec = box_spec(
            native::DType::Int64,
            native::BoxBounds::TypedElementwise(native::TypedElementwiseBounds {
                low: Vec::new(),
                high: Vec::new(),
            }),
            vec![i64::MAX, 2],
        );
        let err = space_spec_from_proto(space_spec_to_proto(&spec)).expect_err("must fail fast");
        assert!(err.to_string().contains("byte length overflowed"));
    }

    fn meta_roundtrip(value: native::MetaValue) -> native::MetaValue {
        meta_value_from_proto(meta_value_to_proto(&value))
    }

    #[test]
    fn meta_int_beyond_two_pow_53_survives_exactly() {
        let value = native::MetaValue::Int((1i64 << 53) + 1);
        assert_eq!(meta_roundtrip(value.clone()), value);

        let value = native::MetaValue::Int(i64::MAX);
        assert_eq!(meta_roundtrip(value.clone()), value);
        let value = native::MetaValue::Int(i64::MIN);
        assert_eq!(meta_roundtrip(value.clone()), value);
    }

    #[test]
    fn meta_whole_number_float_stays_float() {
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
