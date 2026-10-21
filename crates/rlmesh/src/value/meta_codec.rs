//! [`Metadata`] on the wire.
//!
//! `None` → Null; `Bool`, `Int`, `Float`, `Bytes` and `List` keep their
//! kind; `Text` → String; `Dict` → Map; `Tuple` → List; a `Tensor` → its
//! elements as nested Lists by shape in C order (a 0-d tensor as its bare
//! element), preserving element order. Decoding yields only the wire kinds,
//! with maps in the wire's sorted key order.

use rlmesh_spaces::{DType as CoreDType, MetaMap, MetaValue, Scalar, decode_scalars};

use super::{Metadata, Value};
use crate::{Error, Result, Tensor};

/// Encode a map for the wire.
pub(crate) fn metadata_to_wire(map: &Metadata) -> Result<MetaMap> {
    map.iter()
        .map(|(key, value)| Ok((key.to_string(), value_to_wire(value)?)))
        .collect()
}

/// Encode one value for the wire.
pub(crate) fn value_to_wire(value: &Value) -> Result<MetaValue> {
    Ok(match value {
        Value::None => MetaValue::Null,
        Value::Bool(flag) => MetaValue::Bool(*flag),
        Value::Int(number) => MetaValue::Int(*number),
        Value::Float(number) => MetaValue::Float(*number),
        Value::Text(text) => MetaValue::String(text.clone()),
        Value::Bytes(bytes) => MetaValue::Bytes(bytes.clone()),
        Value::List(items) | Value::Tuple(items) => {
            MetaValue::List(items.iter().map(value_to_wire).collect::<Result<_>>()?)
        }
        Value::Dict(map) => MetaValue::Map(metadata_to_wire(map)?),
        Value::Tensor(tensor) => tensor_to_wire(tensor)?,
    })
}

/// Decode a wire map.
pub(crate) fn metadata_from_wire(map: MetaMap) -> Metadata {
    map.into_iter()
        .map(|(key, value)| (key, value_from_wire(value)))
        .collect()
}

/// Decode one wire value.
pub(crate) fn value_from_wire(value: MetaValue) -> Value {
    match value {
        MetaValue::Null => Value::None,
        MetaValue::Bool(flag) => Value::Bool(flag),
        MetaValue::Int(number) => Value::Int(number),
        MetaValue::Float(number) => Value::Float(number),
        MetaValue::String(text) => Value::Text(text),
        MetaValue::Bytes(bytes) => Value::Bytes(bytes),
        MetaValue::List(items) => Value::List(items.into_iter().map(value_from_wire).collect()),
        MetaValue::Map(map) => Value::Dict(metadata_from_wire(map)),
        // The facade's metadata carries arrays as their nested lists.
        array @ (MetaValue::Tensor(_) | MetaValue::Array { .. }) => {
            value_from_wire(array.lowered())
        }
    }
}

fn tensor_to_wire(tensor: &Tensor) -> Result<MetaValue> {
    let core = tensor.core();
    let dtype = core.dtype();
    let bytes = core.to_contiguous_bytes();
    let elements: Vec<MetaValue> = if dtype == CoreDType::Uint64 {
        bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|raw| {
                let number = u64::from_le_bytes(*raw);
                i64::try_from(number).map(MetaValue::Int).map_err(|_| {
                    Error::invalid_argument(format!(
                        "metadata integer {number} does not fit int64; RLMesh metadata \
                         integers must be in [-2**63, 2**63) -- store a larger value as a \
                         string instead"
                    ))
                })
            })
            .collect::<Result<_>>()?
    } else {
        decode_scalars(&bytes, dtype)
            .map_err(|err| Error::invalid_argument(err.to_string()))?
            .into_iter()
            .map(|scalar| match scalar {
                Scalar::Bool(flag) => MetaValue::Bool(flag),
                Scalar::Int(number) => MetaValue::Int(number),
                Scalar::Float(number) => MetaValue::Float(number),
            })
            .collect()
    };
    let shape = tensor.shape();
    if shape.is_empty() {
        return Ok(elements.into_iter().next().unwrap_or(MetaValue::Null));
    }
    let mut elements = elements.into_iter();
    Ok(nest(&mut elements, shape))
}

/// The next `shape[0]` items, each nested by the rest of `shape`.
fn nest(elements: &mut impl Iterator<Item = MetaValue>, shape: &[usize]) -> MetaValue {
    let (&len, rest) = shape.split_first().unwrap_or((&0, &[]));
    MetaValue::List(
        (0..len)
            .map(|_| {
                if rest.is_empty() {
                    elements.next().unwrap_or(MetaValue::Null)
                } else {
                    nest(elements, rest)
                }
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;

    fn i32_tensor(values: &[i32], shape: &[usize]) -> Tensor {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Tensor::new(&bytes, shape, DType::Int32).expect("valid tensor")
    }

    #[test]
    fn kinds_encode_as_python_encodes_a_dict() {
        let map: Metadata = [
            ("none", Value::None),
            ("flag", Value::Bool(true)),
            ("n", Value::Int(3)),
            ("x", Value::Float(0.5)),
            ("s", Value::from("hi")),
            ("b", Value::from(vec![1u8, 2])),
            ("t", Value::Tuple(vec![Value::Int(1), Value::from("a")])),
            ("m", Value::Dict([("k", 1)].into_iter().collect())),
            (
                "grid",
                Value::Tensor(i32_tensor(&[1, 2, 3, 4, 5, 6], &[2, 3])),
            ),
            ("scalar", Value::Tensor(i32_tensor(&[7], &[]))),
        ]
        .into_iter()
        .collect();
        let wire = metadata_to_wire(&map).unwrap();
        assert_eq!(wire["none"], MetaValue::Null);
        assert_eq!(wire["s"], MetaValue::String("hi".into()));
        assert_eq!(
            wire["t"],
            MetaValue::List(vec![MetaValue::Int(1), MetaValue::String("a".into())])
        );
        let row = |a, b, c| {
            MetaValue::List(vec![
                MetaValue::Int(a),
                MetaValue::Int(b),
                MetaValue::Int(c),
            ])
        };
        assert_eq!(
            wire["grid"],
            MetaValue::List(vec![row(1, 2, 3), row(4, 5, 6)])
        );
        assert_eq!(wire["scalar"], MetaValue::Int(7));
        // Decoding yields the wire kinds, in sorted key order.
        let back = metadata_from_wire(wire);
        assert_eq!(back.keys().collect::<Vec<_>>()[0], "b");
        assert_eq!(
            back.get("t").unwrap(),
            &Value::List(vec![Value::Int(1), Value::from("a")])
        );
        assert_eq!(back.get("flag").unwrap(), &Value::Bool(true));
    }

    #[test]
    fn a_uint64_above_int64_is_a_value_error() {
        let bytes = u64::MAX.to_le_bytes();
        let big = Tensor::new(&bytes, &[1], DType::Uint64).unwrap();
        let err = value_to_wire(&Value::Tensor(big)).unwrap_err();
        assert_eq!(err.class(), crate::ErrorClass::ValueError);
        assert!(err.to_string().contains("does not fit int64"), "{err}");
    }

    #[test]
    fn bool_and_float_elements_keep_their_kind() {
        let flags = Tensor::new(&[1, 0], &[2], DType::Bool).unwrap();
        assert_eq!(
            value_to_wire(&Value::Tensor(flags)).unwrap(),
            MetaValue::List(vec![MetaValue::Bool(true), MetaValue::Bool(false)])
        );
        let bytes: Vec<u8> = [0.5f32, 1.5].iter().flat_map(|v| v.to_le_bytes()).collect();
        let floats = Tensor::new(&bytes, &[2], DType::Float32).unwrap();
        assert_eq!(
            value_to_wire(&Value::Tensor(floats)).unwrap(),
            MetaValue::List(vec![MetaValue::Float(0.5), MetaValue::Float(1.5)])
        );
    }
}
