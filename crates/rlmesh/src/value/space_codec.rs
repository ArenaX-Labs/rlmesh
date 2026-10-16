//! Space values as [`Value`]s (SPC-5, SPC-10): the wire `SpaceValue` a space
//! samples and checks, read and written against its spec.
//!
//! Box → `Tensor` (shape `()` too), Discrete → `Int`, Dict → dict in the
//! space's declaration order, MultiBinary / MultiDiscrete → `Tensor`, Text →
//! `Text`, Tuple → `Tuple`.

use rlmesh_spaces::spaces::SpaceKind as WireKind;
use rlmesh_spaces::{
    DType as CoreDType, Scalar, SpaceSpec as WireSpec, SpaceValue, Tensor as CoreTensor,
    check_int_in_dtype_range, decode_scalars, encode_i64_scalars, f64_to_f16_bits,
};

use super::{Metadata, Value};
use crate::{Error, Result, Tensor};

/// The [`Value`] for `value`, a value of the space `spec`.
pub(crate) fn value_from_space(spec: &WireSpec, value: SpaceValue) -> Result<Value> {
    if let rlmesh_spaces::Conformance::Structural(err) = rlmesh_spaces::conform(spec, &value) {
        return Err(Error::Protocol(err.to_string()));
    }
    Ok(value_from_space_unchecked(spec, value))
}

fn value_from_space_unchecked(spec: &WireSpec, value: SpaceValue) -> Value {
    match (spec.spec.as_ref(), value) {
        (Some(WireKind::MultiBinary(_)), SpaceValue::MultiBinary(bits)) => {
            let ints: Vec<i64> = bits.iter().map(|&bit| i64::from(bit)).collect();
            int_tensor(&ints, &spec.shape, resolved(spec.dtype, CoreDType::Int8))
        }
        (Some(WireKind::MultiDiscrete(_)), SpaceValue::MultiDiscrete(ints)) => {
            int_tensor(&ints, &spec.shape, resolved(spec.dtype, CoreDType::Int64))
        }
        (Some(WireKind::Dict(dict)), SpaceValue::Dict(mut values)) => {
            let mut map = Metadata::new();
            for (key, child) in dict.keys.iter().zip(&dict.spaces) {
                if let Some(value) = values.remove(key) {
                    map.insert(key.clone(), value_from_space_unchecked(child, value));
                }
            }
            // A key the spec does not declare keeps the wire's sorted order.
            for (key, value) in values {
                map.insert(key, value_from_unspecced(value));
            }
            Value::Dict(map)
        }
        (Some(WireKind::Tuple(tuple)), SpaceValue::Tuple(items))
            if items.len() == tuple.spaces.len() =>
        {
            Value::Tuple(
                items
                    .into_iter()
                    .zip(&tuple.spaces)
                    .map(|(item, child)| value_from_space_unchecked(child, item))
                    .collect(),
            )
        }
        (_, value) => value_from_unspecced(value),
    }
}

/// The [`Value`] for a space value read without its spec.
fn value_from_unspecced(value: SpaceValue) -> Value {
    match value {
        SpaceValue::Box(tensor) => Value::Tensor(Tensor::wrap(tensor)),
        SpaceValue::Discrete(number) => Value::Int(number),
        SpaceValue::MultiBinary(bits) => {
            let ints: Vec<i64> = bits.iter().map(|&bit| i64::from(bit)).collect();
            int_tensor(&ints, &[], CoreDType::Int8)
        }
        SpaceValue::MultiDiscrete(ints) => int_tensor(&ints, &[], CoreDType::Int64),
        SpaceValue::Text(text) => Value::Text(text),
        SpaceValue::Dict(values) => Value::Dict(
            values
                .into_iter()
                .map(|(key, value)| (key, value_from_unspecced(value)))
                .collect(),
        ),
        SpaceValue::Tuple(items) => {
            Value::Tuple(items.into_iter().map(value_from_unspecced).collect())
        }
    }
}

/// A tensor of `ints` in `dtype`, shaped `shape` when it holds that many
/// elements, flat otherwise.
fn int_tensor(ints: &[i64], shape: &[i64], dtype: CoreDType) -> Value {
    let numel: i64 = shape.iter().product();
    let shape = if !shape.is_empty() && usize::try_from(numel).ok() == Some(ints.len()) {
        shape.to_vec()
    } else {
        vec![ints.len() as i64]
    };
    let bytes = encode_i64_scalars(ints, dtype).unwrap_or_default();
    match CoreTensor::from_vec(bytes, shape, dtype) {
        Ok(tensor) => Value::Tensor(Tensor::wrap(tensor)),
        Err(_) => Value::List(ints.iter().map(|&int| Value::Int(int)).collect()),
    }
}

fn resolved(dtype: CoreDType, default: CoreDType) -> CoreDType {
    if dtype == CoreDType::Unspecified {
        default
    } else {
        dtype
    }
}

/// Whether `value` is a member of `spec` (Python's `Space.contains`): it
/// encodes as Python's native encoder does and then conforms, range included.
pub(crate) fn space_contains(spec: &WireSpec, value: &Value) -> bool {
    shapes_match(value, spec)
        && value_to_space(spec, value)
            .is_ok_and(|encoded| rlmesh_spaces::contains(spec, &encoded).is_ok())
}

/// Every Box value inside `value` has its space's exact shape: a `(6,)`
/// value is not in a `(2, 3)` Box, and a bare number is a member only of a
/// shape-`()` Box. Anything the encoder refuses anyway reads as matching.
fn shapes_match(value: &Value, spec: &WireSpec) -> bool {
    match (spec.spec.as_ref(), value) {
        (Some(WireKind::Box(_)), value) => {
            input_shape(value).is_some_and(|shape| shape == spec.shape)
        }
        (Some(WireKind::Dict(dict)), Value::Dict(map)) => dict
            .keys
            .iter()
            .zip(&dict.spaces)
            .all(|(key, child)| map.get(key).is_none_or(|item| shapes_match(item, child))),
        (Some(WireKind::Tuple(tuple)), Value::Tuple(items) | Value::List(items)) => items
            .iter()
            .zip(&tuple.spaces)
            .all(|(item, child)| shapes_match(item, child)),
        _ => true,
    }
}

/// A Box input's shape: a tensor's, the nesting of a list or tuple (`None`
/// when ragged), or `()` for anything else.
fn input_shape(value: &Value) -> Option<Vec<i64>> {
    match value {
        Value::Tensor(tensor) => Some(tensor.core().shape().to_vec()),
        Value::List(items) | Value::Tuple(items) => {
            let mut inner: Option<Vec<i64>> = None;
            for item in items {
                let shape = input_shape(item)?;
                match &inner {
                    Some(existing) if *existing != shape => return None,
                    Some(_) => {}
                    None => inner = Some(shape),
                }
            }
            let mut shape = vec![items.len() as i64];
            shape.extend(inner.unwrap_or_default());
            Some(shape)
        }
        _ => Some(Vec::new()),
    }
}

/// Encode `value` as a value of `spec`, rejecting structural deviations (SPC-5).
pub(crate) fn value_to_space(spec: &WireSpec, value: &Value) -> Result<SpaceValue> {
    let encoded = value_to_space_unchecked(spec, value)?;
    if let rlmesh_spaces::Conformance::Structural(err) = rlmesh_spaces::conform(spec, &encoded) {
        return Err(Error::invalid_argument(err.to_string()));
    }
    Ok(encoded)
}

fn value_to_space_unchecked(spec: &WireSpec, value: &Value) -> Result<SpaceValue> {
    Ok(match spec.spec.as_ref() {
        Some(WireKind::Box(_)) => {
            let dtype = resolved(spec.dtype, CoreDType::Float32);
            let bytes = array_bytes(value, spec, dtype)?;
            SpaceValue::Box(
                CoreTensor::from_vec(bytes, spec.shape.clone(), dtype)
                    .map_err(|err| Error::invalid_argument(format!("invalid box value: {err}")))?,
            )
        }
        Some(WireKind::Discrete(_)) => SpaceValue::Discrete(discrete(value)?),
        Some(WireKind::MultiBinary(_)) => {
            let dtype = resolved(spec.dtype, CoreDType::Int8);
            let bytes = array_bytes(value, spec, dtype)?;
            SpaceValue::MultiBinary(
                scalars(&bytes, dtype)?
                    .into_iter()
                    .map(|scalar| scalar.as_i64() != 0)
                    .collect(),
            )
        }
        Some(WireKind::MultiDiscrete(_)) => {
            let dtype = resolved(spec.dtype, CoreDType::Int64);
            let bytes = array_bytes(value, spec, dtype)?;
            SpaceValue::MultiDiscrete(
                scalars(&bytes, dtype)?
                    .into_iter()
                    .map(Scalar::as_i64)
                    .collect(),
            )
        }
        Some(WireKind::Text(_)) => SpaceValue::Text(value.as_text()?.to_string()),
        Some(WireKind::Dict(dict)) => {
            let map = value.as_dict()?;
            let mut values = std::collections::BTreeMap::new();
            for (key, child) in dict.keys.iter().zip(&dict.spaces) {
                if let Some(item) = map.get(key) {
                    values.insert(key.clone(), value_to_space_unchecked(child, item)?);
                }
            }
            SpaceValue::Dict(values)
        }
        Some(WireKind::Tuple(tuple)) => {
            let items = match value {
                Value::Tuple(items) | Value::List(items) => items,
                _ => {
                    return Err(Error::Type(
                        "Tuple space values must be a tuple or list".into(),
                    ));
                }
            };
            if items.len() != tuple.spaces.len() {
                return Err(Error::invalid_argument(format!(
                    "Tuple space expected {} items, got {}",
                    tuple.spaces.len(),
                    items.len()
                )));
            }
            SpaceValue::Tuple(
                items
                    .iter()
                    .zip(&tuple.spaces)
                    .map(|(item, child)| value_to_space_unchecked(child, item))
                    .collect::<Result<_>>()?,
            )
        }
        None => return Err(Error::invalid_argument("space spec is missing")),
    })
}

fn scalars(bytes: &[u8], dtype: CoreDType) -> Result<Vec<Scalar>> {
    decode_scalars(bytes, dtype).map_err(|err| Error::invalid_argument(err.to_string()))
}

/// A Discrete member: `Int`, `Bool` (0/1), an integral finite `Float`, or a
/// one-element `Tensor` of one of these.
fn discrete(value: &Value) -> Result<i64> {
    match value {
        Value::Int(number) => Ok(*number),
        Value::Bool(flag) => Ok(i64::from(*flag)),
        Value::Float(number) => integral(*number),
        Value::Tensor(tensor) if tensor.size() == 1 => {
            let core = tensor.core();
            match scalars(&core.to_contiguous_bytes(), core.dtype())?.first() {
                Some(Scalar::Int(number)) => Ok(*number),
                Some(Scalar::Bool(flag)) => Ok(i64::from(*flag)),
                Some(Scalar::Float(number)) => integral(*number),
                None => Err(Error::invalid_argument("empty tensor")),
            }
        }
        other => Err(Error::Type(format!(
            "Discrete value must be an integer, got {:?}",
            other.kind()
        ))),
    }
}

fn integral(number: f64) -> Result<i64> {
    if !number.is_finite() || number.fract() != 0.0 {
        return Err(Error::invalid_argument(format!(
            "Discrete value must be an integer, got {number}"
        )));
    }
    Ok(number as i64)
}

/// The element bytes of an array-like value in `dtype`: a tensor of exactly
/// that dtype (a whole number of space elements), or a number or nested list
/// or tuple of numbers.
fn array_bytes(value: &Value, spec: &WireSpec, dtype: CoreDType) -> Result<Vec<u8>> {
    if let Value::Tensor(tensor) = value {
        let core = tensor.core();
        if core.dtype() != dtype {
            return Err(Error::invalid_argument(format!(
                "tensor dtype {} does not match space dtype {}",
                crate::DType::from_core(core.dtype()).as_str(),
                crate::DType::from_core(dtype).as_str(),
            )));
        }
        let base: i64 = spec.shape.iter().product();
        let base = usize::try_from(base).unwrap_or(0);
        let numel = core.numel();
        if (base == 0 && numel != 0) || (base != 0 && !numel.is_multiple_of(base)) {
            return Err(Error::invalid_argument(format!(
                "tensor element count {numel} is not a multiple of the space element count {base}"
            )));
        }
        return Ok(core.to_contiguous_bytes().into_owned());
    }
    let mut flat = Vec::new();
    flatten(value, &mut flat)?;
    let mut bytes = Vec::with_capacity(flat.len() * rlmesh_spaces::dtype_size(dtype));
    for item in flat {
        pack(item, dtype, &mut bytes)?;
    }
    Ok(bytes)
}

fn flatten<'a>(value: &'a Value, out: &mut Vec<&'a Value>) -> Result<()> {
    match value {
        Value::List(items) | Value::Tuple(items) => {
            for item in items {
                flatten(item, out)?;
            }
            Ok(())
        }
        Value::Dict(_) => Err(Error::Type("array-like values cannot be dicts".into())),
        Value::Tensor(_) => Err(Error::Type(
            "a nested tensor is not an array element".into(),
        )),
        other => {
            out.push(other);
            Ok(())
        }
    }
}

/// One element in `dtype`, as Python's `pack_scalar_bytes`: integer dtypes
/// take `Int` or `Bool` in range, never `Float`; `bool` takes any number,
/// true when nonzero; float dtypes take any number, rounded to the dtype.
fn pack(value: &Value, dtype: CoreDType, out: &mut Vec<u8>) -> Result<()> {
    let refused = || {
        Error::Type(format!(
            "cannot encode a {:?} value as {}",
            value.kind(),
            crate::DType::from_core(dtype).as_str()
        ))
    };
    let number = |value: &Value| -> Option<f64> {
        match value {
            Value::Bool(flag) => Some(f64::from(u8::from(*flag))),
            Value::Int(number) => Some(*number as f64),
            Value::Float(number) => Some(*number),
            _ => None,
        }
    };
    match dtype {
        CoreDType::Bool => {
            let flag = match value {
                Value::Bool(flag) => *flag,
                Value::Int(number) => *number != 0,
                Value::Float(number) => *number != 0.0,
                _ => return Err(refused()),
            };
            out.push(u8::from(flag));
        }
        CoreDType::Float32 | CoreDType::Unspecified => {
            out.extend((number(value).ok_or_else(refused)? as f32).to_le_bytes());
        }
        CoreDType::Float64 => out.extend(number(value).ok_or_else(refused)?.to_le_bytes()),
        CoreDType::Float16 => {
            out.extend(f64_to_f16_bits(number(value).ok_or_else(refused)?).to_le_bytes());
        }
        int_dtype => {
            let int = match value {
                Value::Int(number) => *number,
                Value::Bool(flag) => i64::from(*flag),
                _ => return Err(refused()),
            };
            check_int_in_dtype_range(int, int_dtype)
                .map_err(|err| Error::invalid_argument(err.to_string()))?;
            out.extend(
                encode_i64_scalars(&[int], int_dtype)
                    .map_err(|err| Error::invalid_argument(err.to_string()))?,
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use rlmesh_spaces::types::{MultiBinarySpec, MultiDiscreteSpec, TupleSpec};

    use super::*;
    use crate::{DType, ValueKind};

    fn spec(shape: &[i64], dtype: CoreDType, kind: WireKind) -> WireSpec {
        WireSpec {
            shape: shape.to_vec(),
            dtype,
            spec: Some(kind),
        }
    }

    #[test]
    fn missing_dict_keys_match_conformance_on_encode_and_are_protocol_errors_on_decode() {
        use rlmesh_spaces::spaces::{DictSpaceBuilder, DiscreteBuilder};
        let child = DictSpaceBuilder::new()
            .insert("choice", DiscreteBuilder::new(3).build().unwrap())
            .build()
            .unwrap();
        let nested = DictSpaceBuilder::new()
            .insert("nested", child.clone())
            .build()
            .unwrap();
        for (space, value, wire) in [
            (
                child,
                Value::Dict(Metadata::new()),
                SpaceValue::Dict(Default::default()),
            ),
            (
                nested,
                Value::Dict(
                    [("nested", Value::Dict(Metadata::new()))]
                        .into_iter()
                        .collect(),
                ),
                SpaceValue::Dict(
                    [("nested".into(), SpaceValue::Dict(Default::default()))]
                        .into_iter()
                        .collect(),
                ),
            ),
        ] {
            let rlmesh_spaces::Conformance::Structural(expected) =
                rlmesh_spaces::conform(&space, &wire)
            else {
                panic!("missing key must be structural");
            };
            let err = value_to_space(&space, &value).unwrap_err();
            assert_eq!(err.class(), crate::ErrorClass::ValueError);
            assert_eq!(err.to_string(), expected.to_string());
            let err = value_from_space(&space, wire).unwrap_err();
            assert_eq!(err.class(), crate::ErrorClass::ProtocolException);
            assert_eq!(err.to_string(), expected.to_string());
        }
    }

    #[test]
    fn array_kinds_read_as_tensors_of_the_space_dtype() {
        let bits = spec(
            &[2, 2],
            CoreDType::Uint8,
            WireKind::MultiBinary(MultiBinarySpec),
        );
        let value = value_from_space(
            &bits,
            SpaceValue::MultiBinary(vec![true, false, false, true]),
        )
        .unwrap();
        let tensor = value.as_tensor().unwrap();
        assert_eq!(
            (tensor.shape(), tensor.dtype()),
            (&[2usize, 2][..], DType::Uint8)
        );
        assert_eq!(tensor.tobytes(), [1, 0, 0, 1]);
        assert!(space_contains(&bits, &value));
        assert!(!space_contains(&bits, &Value::List(vec![Value::Int(1)])));

        let counts = spec(
            &[2],
            CoreDType::Int64,
            WireKind::MultiDiscrete(MultiDiscreteSpec { nvec: vec![3, 4] }),
        );
        let value = value_from_space(&counts, SpaceValue::MultiDiscrete(vec![2, 3])).unwrap();
        assert_eq!(value.kind(), ValueKind::Tensor);
        assert!(space_contains(&counts, &value));
        assert!(space_contains(
            &counts,
            &Value::List(vec![Value::Int(0), Value::Int(3)])
        ));
        assert!(!space_contains(
            &counts,
            &Value::List(vec![Value::Int(3), Value::Int(0)])
        ));
    }

    #[test]
    fn tuples_take_a_tuple_or_a_list() {
        let child = spec(
            &[],
            CoreDType::Int64,
            WireKind::Discrete(rlmesh_spaces::types::DiscreteSpec { n: 2, start: 0 }),
        );
        let pair = spec(
            &[],
            CoreDType::Unspecified,
            WireKind::Tuple(TupleSpec {
                spaces: vec![child.clone(), child],
            }),
        );
        let value = value_from_space(
            &pair,
            SpaceValue::Tuple(vec![SpaceValue::Discrete(1), SpaceValue::Discrete(0)]),
        )
        .unwrap();
        assert_eq!(value, Value::Tuple(vec![Value::Int(1), Value::Int(0)]));
        assert!(space_contains(&pair, &value));
        assert!(space_contains(
            &pair,
            &Value::List(vec![Value::Int(1), Value::Int(1)])
        ));
        assert!(!space_contains(&pair, &Value::List(vec![Value::Int(1)])));
    }
}
