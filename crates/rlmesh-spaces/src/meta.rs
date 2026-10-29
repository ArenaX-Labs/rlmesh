use std::collections::BTreeMap;

use crate::{DType, Scalar, Tensor, decode_scalars};

#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Bytes(Vec<u8>),
    List(Vec<MetaValue>),
    Map(MetaMap),
    /// A numeric or bool array (a NumPy array; shape `[]` a NumPy scalar).
    Tensor(Tensor),
    /// An array of other elements (a NumPy object array): its shape and its
    /// elements, row-major.
    Array {
        shape: Vec<i64>,
        items: Vec<MetaValue>,
    },
}

pub type MetaMap = BTreeMap<String, MetaValue>;

impl MetaValue {
    /// This value with every array as nested lists by shape (a shape-`[]`
    /// array as its bare element), for a reader that carries no arrays (the
    /// v1 wire, the Rust facade's `Value`). A `uint64` element above
    /// `i64::MAX` lowers to a float.
    pub fn lowered(&self) -> MetaValue {
        match self {
            Self::Tensor(tensor) => {
                let bytes = tensor.to_contiguous_bytes();
                let elements: Vec<MetaValue> = if tensor.dtype() == DType::Uint64 {
                    bytes
                        .as_chunks::<8>()
                        .0
                        .iter()
                        .map(|raw| {
                            let n = u64::from_le_bytes(*raw);
                            i64::try_from(n).map_or(Self::Float(n as f64), Self::Int)
                        })
                        .collect()
                } else {
                    decode_scalars(&bytes, tensor.dtype())
                        .unwrap_or_default()
                        .into_iter()
                        .map(|scalar| match scalar {
                            Scalar::Bool(flag) => Self::Bool(flag),
                            Scalar::Int(n) => Self::Int(n),
                            Scalar::Float(x) => Self::Float(x),
                        })
                        .collect()
                };
                nested(elements, tensor.shape())
            }
            Self::Array { shape, items } => {
                nested(items.iter().map(Self::lowered).collect(), shape)
            }
            Self::List(items) => Self::List(items.iter().map(Self::lowered).collect()),
            Self::Map(map) => Self::Map(
                map.iter()
                    .map(|(key, value)| (key.clone(), value.lowered()))
                    .collect(),
            ),
            other => other.clone(),
        }
    }
}

/// `elements` nested by `shape` (C order); shape `[]` is the bare element.
fn nested(elements: Vec<MetaValue>, shape: &[i64]) -> MetaValue {
    fn nest(elements: &mut impl Iterator<Item = MetaValue>, shape: &[i64]) -> MetaValue {
        let (&len, rest) = shape.split_first().unwrap_or((&0, &[]));
        MetaValue::List(
            (0..len.max(0))
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
    let mut elements = elements.into_iter();
    if shape.is_empty() {
        return elements.next().unwrap_or(MetaValue::Null);
    }
    nest(&mut elements, shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arrays_lower_to_nested_lists_and_a_scalar_to_its_element() {
        let bytes: Vec<u8> = [1_i16, 2, 3, 4, 5, 6]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let grid = Tensor::from_vec(bytes, vec![2, 3], DType::Int16).unwrap();
        let int = MetaValue::Int;
        assert_eq!(
            MetaValue::Tensor(grid).lowered(),
            MetaValue::List(vec![
                MetaValue::List(vec![int(1), int(2), int(3)]),
                MetaValue::List(vec![int(4), int(5), int(6)]),
            ])
        );
        let scalar = Tensor::from_vec(0.5_f32.to_le_bytes().to_vec(), vec![], DType::Float32);
        assert_eq!(
            MetaValue::Tensor(scalar.unwrap()).lowered(),
            MetaValue::Float(0.5)
        );
        let objects = MetaValue::Array {
            shape: vec![2],
            items: vec![MetaValue::String("a".into()), MetaValue::Null],
        };
        assert_eq!(
            objects.lowered(),
            MetaValue::List(vec![MetaValue::String("a".into()), MetaValue::Null])
        );
    }
}
