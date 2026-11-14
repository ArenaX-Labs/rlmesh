//! [`Metadata`] on the wire.
//!
//! `None` → Null; `Bool`, `Int`, `Float`, `Bytes`, `List` and `Tensor` keep
//! their kind (an array travels as an array, as a NumPy array does); `Text`
//! → String; `Dict` → Map; `Tuple` → List. Decoding yields only the wire
//! kinds, with maps in the wire's sorted key order and an object array as
//! its nested lists.

use rlmesh_spaces::{MetaMap, MetaValue};

use super::{Metadata, Value};
use crate::{Result, Tensor};

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
        Value::Tensor(tensor) => MetaValue::Tensor(tensor.core().clone()),
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
        MetaValue::Tensor(tensor) => Value::Tensor(Tensor::wrap(tensor)),
        // The facade has no object array: its nested lists.
        array @ MetaValue::Array { .. } => value_from_wire(array.lowered()),
    }
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
        // Arrays travel as arrays, 0-d ones too.
        let grid = i32_tensor(&[1, 2, 3, 4, 5, 6], &[2, 3]);
        assert_eq!(wire["grid"], MetaValue::Tensor(grid.core().clone()));
        assert_eq!(
            wire["scalar"],
            MetaValue::Tensor(i32_tensor(&[7], &[]).core().clone())
        );
        // Decoding yields the wire kinds, in sorted key order.
        let back = metadata_from_wire(wire);
        assert_eq!(back.keys().collect::<Vec<_>>()[0], "b");
        assert_eq!(
            back.get("t").unwrap(),
            &Value::List(vec![Value::Int(1), Value::from("a")])
        );
        assert_eq!(back.get("flag").unwrap(), &Value::Bool(true));
        assert_eq!(back.get("grid").unwrap(), &Value::Tensor(grid));
    }
}
