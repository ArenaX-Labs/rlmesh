//! v1 `MetaMap` (Gym `info`, reset `options`) to and from JSON, losslessly.
//!
//! The lockstep messages carry `info` and `options` as JSON (`info_json`,
//! `options_json`, §12.2). Plain values map to plain JSON, so a v2-native
//! reader sees the dict it expects; the three things JSON cannot say are
//! tagged, so the compatibility path round-trips a v1 `MetaMap` bit for bit:
//!
//! - an integer is a JSON integer and a float always has a fraction or an
//!   exponent (`1.0`), so the two stay apart;
//! - a non-finite float is `{"$f64": "nan" | "inf" | "-inf"}`;
//! - bytes are `{"$bytes": "<base64>"}`;
//! - a map key starting with `$` is written with one more `$`.

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use rlmesh_proto::spaces::v1::{MetaList, MetaMap, MetaNull, MetaValue, meta_value::Kind};
use serde_json::{Map, Number, Value};

const F64: &str = "$f64";
const BYTES: &str = "$bytes";

/// A map as JSON text.
pub fn to_json(map: &MetaMap) -> String {
    map_value(map).to_string()
}

/// An optional map as JSON text: empty text is no map at all (v1's unset
/// `infos`), `{}` an empty one, so the two stay apart.
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
    match value {
        Value::Object(object) => object_map(object),
        _ => Err("info json is not an object".into()),
    }
}

/// A `MetaValue` as JSON.
pub fn value(value: &MetaValue) -> Value {
    match &value.kind {
        None | Some(Kind::Null(_)) => Value::Null,
        Some(Kind::Bool(b)) => Value::Bool(*b),
        Some(Kind::Integer(i)) => Value::Number((*i).into()),
        Some(Kind::Number(x)) => match Number::from_f64(*x) {
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
        Some(Kind::Text(s)) => Value::String(s.clone()),
        Some(Kind::Bytes(b)) => tagged(BYTES, Value::String(STANDARD.encode(b))),
        Some(Kind::List(list)) => Value::Array(list.items.iter().map(self::value).collect()),
        Some(Kind::Map(map)) => map_value(map),
    }
}

fn tagged(tag: &str, value: Value) -> Value {
    let mut object = Map::new();
    object.insert(tag.into(), value);
    Value::Object(object)
}

fn map_value(map: &MetaMap) -> Value {
    let mut keys: Vec<&String> = map.entries.keys().collect();
    keys.sort();
    let mut object = Map::new();
    for key in keys {
        let escaped = if key.starts_with('$') {
            format!("${key}")
        } else {
            key.clone()
        };
        object.insert(escaped, value(&map.entries[key]));
    }
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
        map.entries.insert(key, meta(value)?);
    }
    Ok(map)
}

/// JSON as a `MetaValue`.
pub fn meta(value: Value) -> Result<MetaValue, String> {
    let kind = match value {
        Value::Null => Kind::Null(MetaNull {}),
        Value::Bool(b) => Kind::Bool(b),
        Value::Number(n) => match n.as_i64() {
            Some(i) if !n.is_f64() => Kind::Integer(i),
            _ => Kind::Number(n.as_f64().ok_or("info json: a number out of range")?),
        },
        Value::String(s) => Kind::Text(s),
        Value::Array(items) => Kind::List(MetaList {
            items: items.into_iter().map(meta).collect::<Result<_, _>>()?,
        }),
        Value::Object(mut object) => {
            if object.len() == 1 {
                if let Some(Value::String(name)) = object.get(F64) {
                    let x = match name.as_str() {
                        "nan" => f64::NAN,
                        "inf" => f64::INFINITY,
                        "-inf" => f64::NEG_INFINITY,
                        other => return Err(format!("info json: $f64 {other:?}")),
                    };
                    return Ok(MetaValue {
                        kind: Some(Kind::Number(x)),
                    });
                }
                if let Some(Value::String(b64)) = object.remove(BYTES) {
                    let bytes = STANDARD
                        .decode(b64)
                        .map_err(|e| format!("info json: $bytes: {e}"))?;
                    return Ok(MetaValue {
                        kind: Some(Kind::Bytes(bytes)),
                    });
                }
            }
            Kind::Map(object_map(object)?)
        }
    };
    Ok(MetaValue { kind: Some(kind) })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(kind: Kind) -> MetaValue {
        MetaValue { kind: Some(kind) }
    }

    #[test]
    fn every_meta_value_round_trips() {
        let mut inner = MetaMap::default();
        inner.entries.insert("$weird".into(), v(Kind::Integer(-3)));
        inner.entries.insert("x".into(), v(Kind::Number(1.0)));
        let mut map = MetaMap::default();
        for (key, value) in [
            ("int", v(Kind::Integer(i64::MIN))),
            ("float", v(Kind::Number(2.0))),
            ("neg_zero", v(Kind::Number(-0.0))),
            ("tiny", v(Kind::Number(5e-324))),
            // A shortest-repr float the default (fast) parser misrounds.
            ("third", v(Kind::Number(1.333_513_733_333_333_3))),
            ("nan", v(Kind::Number(f64::NAN))),
            ("inf", v(Kind::Number(f64::NEG_INFINITY))),
            ("text", v(Kind::Text("hé".into()))),
            ("bool", v(Kind::Bool(true))),
            ("bytes", v(Kind::Bytes(vec![0, 255, 7]))),
            ("null", v(Kind::Null(MetaNull {}))),
            (
                "list",
                v(Kind::List(MetaList {
                    items: vec![v(Kind::Integer(1)), v(Kind::Number(1.5))],
                })),
            ),
            ("map", v(Kind::Map(inner))),
        ] {
            map.entries.insert(key.into(), value);
        }
        let text = to_json(&map);
        let back = from_json(&text).expect("decode");
        // NaN != NaN: compare encodings.
        assert_eq!(to_json(&back), text);
        let Some(Kind::Number(z)) = back.entries["neg_zero"].kind else {
            panic!("number");
        };
        assert!(z.is_sign_negative());
        assert!(matches!(back.entries["float"].kind, Some(Kind::Number(_))));
        for key in ["third", "tiny", "float"] {
            let (Some(Kind::Number(a)), Some(Kind::Number(b))) =
                (&map.entries[key].kind, &back.entries[key].kind)
            else {
                panic!("{key}");
            };
            assert_eq!(a.to_bits(), b.to_bits(), "{key} round-trips bit-exact");
        }
        assert!(matches!(
            back.entries["int"].kind,
            Some(Kind::Integer(i64::MIN))
        ));
        assert_eq!(optional_to_json(None), "");
        assert_eq!(optional_to_json(Some(&MetaMap::default())), "{}");
        assert_eq!(optional_from_json("").expect("none"), None);
        assert_eq!(
            optional_from_json("{}").expect("empty"),
            Some(MetaMap::default())
        );
        assert!(from_json("").expect("empty").entries.is_empty());
        assert!(from_json("{\"$x\": 1}").is_err());
    }
}
