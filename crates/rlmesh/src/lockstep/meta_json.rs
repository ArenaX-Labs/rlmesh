//! A `MetaMap` (Gym `info`, reset `options`, env metadata) to and from JSON,
//! losslessly.
//!
//! The lockstep messages carry `info` and `options` as JSON (`info_json`,
//! `options_json`, §12.2). Plain values map to plain JSON, so a v2-native
//! reader sees the dict it expects; the three things JSON cannot say are
//! tagged, so the compatibility path round-trips a `MetaMap` bit for bit:
//!
//! - an integer is a JSON integer and a float always has a fraction or an
//!   exponent (`1.0`), so the two stay apart;
//! - a non-finite float is `{"$f64": "nan" | "inf" | "-inf"}`;
//! - bytes are `{"$bytes": "<base64>"}`;
//! - a numeric or bool array (a NumPy array or scalar) is
//!   `{"$tensor": {"dtype", "shape", "data": "<base64 of its little-endian
//!   bytes>"}}`, and an object array `{"$array": {"shape", "items"}}`;
//! - a map key starting with `$` is written with one more `$`.

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use rlmesh_spaces::{DType, MetaMap, MetaValue, Tensor};
use serde_json::{Map, Number, Value};

const F64: &str = "$f64";
const BYTES: &str = "$bytes";
const TENSOR: &str = "$tensor";
const ARRAY: &str = "$array";

/// A map as JSON text.
pub fn to_json(map: &MetaMap) -> String {
    map_value(map).to_string()
}

/// An optional map as JSON text: empty text is no map at all (an unset
/// `info`), `{}` an empty one, so the two stay apart.
pub fn optional_to_json(map: Option<&MetaMap>) -> String {
    map.map(to_json).unwrap_or_default()
}

/// The inverse of [`optional_to_json`].
pub fn optional_from_json(text: &str) -> Result<Option<MetaMap>, String> {
    if text.is_empty() {
        return Ok(None);
    }
    from_json(text).map(Some)
}

/// A map from JSON text (empty text is an empty map).
pub fn from_json(text: &str) -> Result<MetaMap, String> {
    if text.is_empty() {
        return Ok(MetaMap::default());
    }
    let value: Value = serde_json::from_str(text).map_err(|e| format!("info json: {e}"))?;
    map_from_value(value)
}

/// A map as a JSON object.
pub fn map_value(map: &MetaMap) -> Value {
    let mut object = Map::new();
    for (key, value) in map {
        let escaped = if key.starts_with('$') {
            format!("${key}")
        } else {
            key.clone()
        };
        object.insert(escaped, self::value(value));
    }
    Value::Object(object)
}

/// The inverse of [`map_value`].
pub fn map_from_value(value: Value) -> Result<MetaMap, String> {
    match value {
        Value::Object(object) => object_map(object),
        _ => Err("info json is not an object".into()),
    }
}

/// A `MetaValue` as JSON.
pub fn value(value: &MetaValue) -> Value {
    match value {
        MetaValue::Null => Value::Null,
        MetaValue::Bool(b) => Value::Bool(*b),
        MetaValue::Int(i) => Value::Number((*i).into()),
        MetaValue::Float(x) => match Number::from_f64(*x) {
            Some(n) => Value::Number(n),
            None => {
                let name = if x.is_nan() {
                    "nan"
                } else if *x > 0.0 {
                    "inf"
                } else {
                    "-inf"
                };
                tagged(F64, Value::String(name.into()))
            }
        },
        MetaValue::String(s) => Value::String(s.clone()),
        MetaValue::Bytes(b) => tagged(BYTES, Value::String(STANDARD.encode(b))),
        MetaValue::List(items) => Value::Array(items.iter().map(self::value).collect()),
        MetaValue::Map(map) => map_value(map),
        MetaValue::Tensor(tensor) => tagged(
            TENSOR,
            serde_json::json!({
                "dtype": tensor.dtype().name(),
                "shape": tensor.shape(),
                "data": STANDARD.encode(tensor.to_contiguous_bytes()),
            }),
        ),
        MetaValue::Array { shape, items } => tagged(
            ARRAY,
            serde_json::json!({
                "shape": shape,
                "items": items.iter().map(self::value).collect::<Vec<_>>(),
            }),
        ),
    }
}

fn tagged(tag: &str, value: Value) -> Value {
    let mut object = Map::new();
    object.insert(tag.into(), value);
    Value::Object(object)
}

fn object_map(object: Map<String, Value>) -> Result<MetaMap, String> {
    let mut map = MetaMap::default();
    for (key, value) in object {
        let key = match key.strip_prefix('$') {
            Some(rest) if rest.starts_with('$') => rest.to_string(),
            Some(_) => return Err(format!("info json: unescaped key {key:?}")),
            None => key,
        };
        map.insert(key, meta(value)?);
    }
    Ok(map)
}

/// JSON as a `MetaValue`.
pub fn meta(value: Value) -> Result<MetaValue, String> {
    Ok(match value {
        Value::Null => MetaValue::Null,
        Value::Bool(b) => MetaValue::Bool(b),
        Value::Number(n) => match n.as_i64() {
            Some(i) if !n.is_f64() => MetaValue::Int(i),
            _ => MetaValue::Float(n.as_f64().ok_or("info json: a number out of range")?),
        },
        Value::String(s) => MetaValue::String(s),
        Value::Array(items) => {
            MetaValue::List(items.into_iter().map(meta).collect::<Result<_, _>>()?)
        }
        Value::Object(mut object) => {
            if object.len() == 1 {
                if let Some(Value::String(name)) = object.get(F64) {
                    let x = match name.as_str() {
                        "nan" => f64::NAN,
                        "inf" => f64::INFINITY,
                        "-inf" => f64::NEG_INFINITY,
                        other => return Err(format!("info json: $f64 {other:?}")),
                    };
                    return Ok(MetaValue::Float(x));
                }
                if let Some(tensor) = object.get(TENSOR) {
                    return tensor_value(tensor).ok_or_else(|| "info json: $tensor".to_string());
                }
                if let Some(Value::Object(array)) = object.remove(ARRAY) {
                    let shape = array
                        .get("shape")
                        .and_then(|s| serde_json::from_value(s.clone()).ok())
                        .ok_or("info json: $array shape")?;
                    let items = match array.get("items") {
                        Some(Value::Array(items)) => {
                            items.iter().cloned().map(meta).collect::<Result<_, _>>()?
                        }
                        _ => return Err("info json: $array items".into()),
                    };
                    return Ok(MetaValue::Array { shape, items });
                }
                if let Some(Value::String(b64)) = object.remove(BYTES) {
                    let bytes = STANDARD
                        .decode(b64)
                        .map_err(|e| format!("info json: $bytes: {e}"))?;
                    return Ok(MetaValue::Bytes(bytes));
                }
            }
            MetaValue::Map(object_map(object)?)
        }
    })
}

fn tensor_value(value: &Value) -> Option<MetaValue> {
    let dtype = DType::from_name(value.get("dtype")?.as_str()?)?;
    let shape: Vec<i64> = serde_json::from_value(value.get("shape")?.clone()).ok()?;
    let data = STANDARD.decode(value.get("data")?.as_str()?).ok()?;
    Tensor::from_vec(data, shape, dtype)
        .ok()
        .map(MetaValue::Tensor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_meta_value_round_trips() {
        let mut inner = MetaMap::default();
        inner.insert("$weird".into(), MetaValue::Int(-3));
        inner.insert("x".into(), MetaValue::Float(1.0));
        let mut map = MetaMap::default();
        for (key, value) in [
            ("int", MetaValue::Int(i64::MIN)),
            ("float", MetaValue::Float(2.0)),
            ("neg_zero", MetaValue::Float(-0.0)),
            ("tiny", MetaValue::Float(5e-324)),
            // A shortest-repr float the default (fast) parser misrounds.
            ("third", MetaValue::Float(1.333_513_733_333_333_3)),
            ("nan", MetaValue::Float(f64::NAN)),
            ("inf", MetaValue::Float(f64::NEG_INFINITY)),
            ("text", MetaValue::String("hé".into())),
            ("bool", MetaValue::Bool(true)),
            ("bytes", MetaValue::Bytes(vec![0, 255, 7])),
            ("null", MetaValue::Null),
            (
                "list",
                MetaValue::List(vec![MetaValue::Int(1), MetaValue::Float(1.5)]),
            ),
            ("map", MetaValue::Map(inner)),
            (
                "tensor",
                MetaValue::Tensor(
                    Tensor::from_vec(vec![1, 0, 2, 0], vec![2], DType::Int16).unwrap(),
                ),
            ),
            (
                "scalar",
                MetaValue::Tensor(
                    Tensor::from_vec(vec![0, 0, 0, 63], vec![], DType::Float32).unwrap(),
                ),
            ),
            (
                "objects",
                MetaValue::Array {
                    shape: vec![1, 2],
                    items: vec![MetaValue::String("a".into()), MetaValue::Int(3)],
                },
            ),
        ] {
            map.insert(key.into(), value);
        }
        let text = to_json(&map);
        let back = from_json(&text).expect("decode");
        // NaN != NaN: compare encodings.
        assert_eq!(to_json(&back), text);
        let MetaValue::Float(z) = back["neg_zero"] else {
            panic!("number");
        };
        assert!(z.is_sign_negative());
        assert!(matches!(back["float"], MetaValue::Float(_)));
        for key in ["third", "tiny", "float"] {
            let (MetaValue::Float(a), MetaValue::Float(b)) = (&map[key], &back[key]) else {
                panic!("{key}");
            };
            assert_eq!(a.to_bits(), b.to_bits(), "{key} round-trips bit-exact");
        }
        assert!(matches!(back["int"], MetaValue::Int(i64::MIN)));
        for key in ["tensor", "scalar", "objects"] {
            assert_eq!(back[key], map[key], "{key}");
        }
        assert_eq!(optional_to_json(None), "");
        assert_eq!(optional_to_json(Some(&MetaMap::default())), "{}");
        assert_eq!(optional_from_json("").expect("none"), None);
        assert_eq!(
            optional_from_json("{}").expect("empty"),
            Some(MetaMap::default())
        );
        assert!(from_json("").expect("empty").is_empty());
        assert!(from_json("{\"$x\": 1}").is_err());
    }
}
