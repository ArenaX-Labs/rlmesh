//! The `SpaceSpec` JSON form a describe envelope embeds under
//! `env_spec.observation_space` / `env_spec.action_space`.
//!
//! `{"kind", "shape", "dtype", "details": {...}}`, the same structure the Python
//! `SpaceSpec._to_dict()` returns (the managed platform reads it from either
//! producer). The envelope is strict JSON, so a non-finite bound renders as
//! `null` (serde_json's own rendering of a non-finite `f64`): unbounded on that
//! edge, exactly as the Python gatherer maps it.

use rlmesh_spaces::scalar::{Scalar, decode_scalars, encode_scalars};
use rlmesh_spaces::spaces::{SpaceKind, SpaceSpec};
use rlmesh_spaces::{
    BoxBounds, BoxSpec, DType, DictSpec, DiscreteSpec, ElementwiseBounds, MultiBinarySpec,
    MultiDiscreteSpec, TextSpec, TupleSpec, TypedElementwiseBounds, TypedUniformBounds,
    UniformBounds,
};
use serde_json::{Map, Value, json};

/// A space that cannot be rendered (no spec at all, or typed Box bounds whose
/// bytes do not decode in the space's dtype), or JSON that is no space.
#[derive(Debug, thiserror::Error)]
pub enum SpaceJsonError {
    #[error("space spec is missing")]
    Missing,
    #[error("{0}")]
    Bounds(String),
    #[error("space json: {0}")]
    Json(String),
}

/// The canonical kind name (`"box"`, `"multi_discrete"`, ...).
pub fn space_kind_name(space: &SpaceSpec) -> &'static str {
    match space.spec.as_ref() {
        Some(SpaceKind::Box(_)) => "box",
        Some(SpaceKind::Discrete(_)) => "discrete",
        Some(SpaceKind::MultiBinary(_)) => "multi_binary",
        Some(SpaceKind::MultiDiscrete(_)) => "multi_discrete",
        Some(SpaceKind::Text(_)) => "text",
        Some(SpaceKind::Dict(_)) => "dict",
        Some(SpaceKind::Tuple(_)) => "tuple",
        None => "unknown",
    }
}

/// The dtype's display name; an unspecified dtype surfaces as `float32` (the
/// legacy display fallback the Python binding shares).
pub fn space_dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::Unspecified => "float32",
        dtype => dtype.name(),
    }
}

/// Render a space as `{"kind", "shape", "dtype", "details"}`, nested spaces
/// recursively in the same form.
pub fn space_spec_to_json(space: &SpaceSpec) -> Result<Value, SpaceJsonError> {
    Ok(json!({
        "kind": space_kind_name(space),
        "shape": space.shape,
        "dtype": space_dtype_name(space.dtype),
        "details": details(space)?,
    }))
}

/// The inverse of [`space_spec_to_json`], exact for every valid spec it
/// renders: integer bounds keep their dtype's bytes, float bounds their bits,
/// and a `null` bound is that edge's infinity. A composite's dtype is
/// unspecified. A JSON object keeps no key order, so a Dict takes its order
/// from `details.keys` when present (else the object's).
pub fn space_spec_from_json(value: &Value) -> Result<SpaceSpec, SpaceJsonError> {
    let field = |name: &str| value.get(name).ok_or_else(|| json_error(name));
    let kind = field("kind")?.as_str().ok_or_else(|| json_error("kind"))?;
    let shape = field("shape")?
        .as_array()
        .and_then(|dims| dims.iter().map(Value::as_i64).collect())
        .ok_or_else(|| json_error("shape"))?;
    let details = field("details")?;
    let detail = |name: &str| details.get(name).ok_or_else(|| json_error(name));
    let int = |name: &str| detail(name)?.as_i64().ok_or_else(|| json_error(name));
    let dtype = match kind {
        "dict" | "tuple" => DType::Unspecified,
        _ => field("dtype")?
            .as_str()
            .and_then(DType::from_name)
            .ok_or_else(|| json_error("dtype"))?,
    };
    let spec = match kind {
        "box" => SpaceKind::Box(BoxSpec {
            bounds: box_bounds(details, dtype)?,
        }),
        "discrete" => SpaceKind::Discrete(DiscreteSpec {
            n: int("n")?,
            start: int("start")?,
        }),
        "multi_binary" => SpaceKind::MultiBinary(MultiBinarySpec),
        "multi_discrete" => {
            let mut nvec = Vec::new();
            flat_ints(detail("nvec")?, &mut nvec).ok_or_else(|| json_error("nvec"))?;
            SpaceKind::MultiDiscrete(MultiDiscreteSpec { nvec })
        }
        "text" => SpaceKind::Text(TextSpec {
            min_length: int("min_length")?,
            max_length: int("max_length")?,
            charset: detail("charset")?
                .as_str()
                .ok_or_else(|| json_error("charset"))?
                .to_string(),
        }),
        "dict" => {
            let spaces = detail("spaces")?
                .as_object()
                .ok_or_else(|| json_error("spaces"))?;
            let keys: Vec<String> = match details.get("keys") {
                Some(keys) => keys
                    .as_array()
                    .and_then(|keys| keys.iter().map(|k| Some(k.as_str()?.to_string())).collect())
                    .ok_or_else(|| json_error("keys"))?,
                None => spaces.keys().cloned().collect(),
            };
            let spaces = keys
                .iter()
                .map(|key| space_spec_from_json(spaces.get(key).ok_or_else(|| json_error(key))?))
                .collect::<Result<_, _>>()?;
            SpaceKind::Dict(DictSpec { keys, spaces })
        }
        "tuple" => SpaceKind::Tuple(TupleSpec {
            spaces: detail("spaces")?
                .as_array()
                .ok_or_else(|| json_error("spaces"))?
                .iter()
                .map(space_spec_from_json)
                .collect::<Result<_, _>>()?,
        }),
        other => return Err(SpaceJsonError::Json(format!("no space kind {other:?}"))),
    };
    Ok(SpaceSpec {
        shape,
        dtype,
        spec: Some(spec),
    })
}

fn json_error(field: &str) -> SpaceJsonError {
    SpaceJsonError::Json(format!("no valid {field}"))
}

/// A flat or nested integer list, flattened row-major.
fn flat_ints(value: &Value, out: &mut Vec<i64>) -> Option<()> {
    match value {
        Value::Array(items) => items.iter().try_for_each(|item| flat_ints(item, out)),
        other => {
            out.push(other.as_i64()?);
            Some(())
        }
    }
}

fn box_bounds(details: &Value, dtype: DType) -> Result<Option<BoxBounds>, SpaceJsonError> {
    let kind = match details.get("bounds_kind") {
        None | Some(Value::Null) => return Ok(None),
        Some(kind) => kind.as_str().ok_or_else(|| json_error("bounds_kind"))?,
    };
    let edges = |name: &str, unbounded: f64| -> Result<Vec<Scalar>, SpaceJsonError> {
        let value = details.get(name).ok_or_else(|| json_error(name))?;
        let items = match value {
            Value::Array(items) => items.as_slice(),
            one => std::slice::from_ref(one),
        };
        items
            .iter()
            .map(|item| scalar(item, unbounded))
            .collect::<Option<_>>()
            .ok_or_else(|| json_error(name))
    };
    let floats = |name: &str, unbounded: f64| -> Result<Vec<f64>, SpaceJsonError> {
        Ok(edges(name, unbounded)?
            .into_iter()
            .map(|s| match s {
                Scalar::Float(x) => x,
                other => other.as_i64() as f64,
            })
            .collect())
    };
    let uniform = |name: &str, unbounded: f64| -> Result<f64, SpaceJsonError> {
        match floats(name, unbounded)?.as_slice() {
            [x] => Ok(*x),
            _ => Err(json_error(name)),
        }
    };
    let typed = |name: &str, unbounded: f64| -> Result<Vec<u8>, SpaceJsonError> {
        encode_scalars(&edges(name, unbounded)?, dtype)
            .map_err(|err| SpaceJsonError::Bounds(err.to_string()))
    };
    let (low, high) = (f64::NEG_INFINITY, f64::INFINITY);
    Ok(Some(match kind {
        "unbounded" => BoxBounds::Unbounded(true),
        "uniform" => BoxBounds::Uniform(UniformBounds {
            low: uniform("low", low)?,
            high: uniform("high", high)?,
        }),
        "elementwise" => BoxBounds::Elementwise(ElementwiseBounds {
            low: floats("low", low)?,
            high: floats("high", high)?,
        }),
        "typed_uniform" => BoxBounds::TypedUniform(TypedUniformBounds {
            low: typed("low", low)?,
            high: typed("high", high)?,
        }),
        "typed_elementwise" => BoxBounds::TypedElementwise(TypedElementwiseBounds {
            low: typed("low", low)?,
            high: typed("high", high)?,
        }),
        other => return Err(SpaceJsonError::Json(format!("no bounds kind {other:?}"))),
    }))
}

/// One bound as a scalar: an integer stays exact (a `Uint64` above
/// `i64::MAX` wraps, as its scalars do), `null` is `unbounded`.
fn scalar(value: &Value, unbounded: f64) -> Option<Scalar> {
    match value {
        Value::Null => Some(Scalar::Float(unbounded)),
        Value::Bool(b) => Some(Scalar::Bool(*b)),
        Value::Number(n) if !n.is_f64() => n
            .as_i64()
            .or_else(|| n.as_u64().map(|u| u as i64))
            .map(Scalar::Int),
        Value::Number(n) => n.as_f64().map(Scalar::Float),
        _ => None,
    }
}

fn details(space: &SpaceSpec) -> Result<Value, SpaceJsonError> {
    let mut details = Map::new();
    match space.spec.as_ref().ok_or(SpaceJsonError::Missing)? {
        SpaceKind::Box(spec) => box_details(&mut details, spec, space.dtype)?,
        SpaceKind::Discrete(spec) => {
            details.insert("n".into(), spec.n.into());
            details.insert("start".into(), spec.start.into());
        }
        // The marker carries no fields: a rank-1 shape surfaces as a scalar
        // `size`, higher ranks as `dims` (Gymnasium's scalar-vs-vector form).
        SpaceKind::MultiBinary(_) => {
            if space.shape.len() == 1 {
                details.insert("size".into(), space.shape[0].into());
            } else {
                details.insert("dims".into(), space.shape.clone().into());
            }
        }
        // `nvec` is stored flat; a rank-2 shape reshapes back to the nested
        // matrix form, lower ranks stay flat.
        SpaceKind::MultiDiscrete(spec) => {
            let nvec = if space.shape.len() == 2 {
                let cols = space.shape[1].max(0) as usize;
                let rows: Vec<Vec<i64>> = if cols == 0 {
                    Vec::new()
                } else {
                    spec.nvec.chunks(cols).map(<[i64]>::to_vec).collect()
                };
                json!(rows)
            } else {
                json!(spec.nvec)
            };
            details.insert("nvec".into(), nvec);
        }
        SpaceKind::Text(spec) => {
            details.insert("min_length".into(), spec.min_length.into());
            details.insert("max_length".into(), spec.max_length.into());
            details.insert("charset".into(), spec.charset.clone().into());
        }
        SpaceKind::Dict(spec) => {
            let mut spaces = Map::new();
            for (key, child) in spec.keys.iter().zip(&spec.spaces) {
                spaces.insert(key.clone(), space_spec_to_json(child)?);
            }
            details.insert("spaces".into(), Value::Object(spaces));
        }
        SpaceKind::Tuple(spec) => {
            let spaces = spec
                .spaces
                .iter()
                .map(space_spec_to_json)
                .collect::<Result<Vec<_>, _>>()?;
            details.insert("spaces".into(), Value::Array(spaces));
        }
    }
    Ok(Value::Object(details))
}

fn box_details(
    details: &mut Map<String, Value>,
    spec: &BoxSpec,
    dtype: DType,
) -> Result<(), SpaceJsonError> {
    let (kind, low, high) = match &spec.bounds {
        Some(BoxBounds::Unbounded(_)) => ("unbounded", None, None),
        Some(BoxBounds::Uniform(bounds)) => (
            "uniform",
            Some(Value::from(bounds.low)),
            Some(Value::from(bounds.high)),
        ),
        Some(BoxBounds::Elementwise(bounds)) => (
            "elementwise",
            Some(floats(&bounds.low)),
            Some(floats(&bounds.high)),
        ),
        // Typed byte bounds decode in the space's dtype, so integer values stay
        // exact (no f64 round-trip).
        Some(BoxBounds::TypedUniform(bounds)) => (
            "typed_uniform",
            Some(typed(&bounds.low, dtype)?),
            Some(typed(&bounds.high, dtype)?),
        ),
        Some(BoxBounds::TypedElementwise(bounds)) => (
            "typed_elementwise",
            Some(typed(&bounds.low, dtype)?),
            Some(typed(&bounds.high, dtype)?),
        ),
        None => {
            details.insert("bounds_kind".into(), Value::Null);
            return Ok(());
        }
    };
    details.insert("bounds_kind".into(), kind.into());
    if let (Some(low), Some(high)) = (low, high) {
        details.insert("low".into(), low);
        details.insert("high".into(), high);
    }
    Ok(())
}

fn floats(values: &[f64]) -> Value {
    Value::Array(values.iter().copied().map(Value::from).collect())
}

fn typed(bytes: &[u8], dtype: DType) -> Result<Value, SpaceJsonError> {
    let scalars =
        decode_scalars(bytes, dtype).map_err(|err| SpaceJsonError::Bounds(err.to_string()))?;
    Ok(Value::Array(
        scalars
            .into_iter()
            .map(|scalar| match scalar {
                Scalar::Bool(value) => Value::from(value),
                // `Uint64` decodes into a wrapped i64; reinterpret so values
                // above i64::MAX stay the correct positive integer.
                Scalar::Int(_) if dtype == DType::Uint64 => Value::from(scalar.as_u64()),
                Scalar::Int(value) => Value::from(value),
                Scalar::Float(value) => Value::from(value),
            })
            .collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn space(shape: &[i64], dtype: DType, kind: SpaceKind) -> SpaceSpec {
        SpaceSpec {
            shape: shape.to_vec(),
            dtype,
            spec: Some(kind),
        }
    }

    fn boxed(shape: &[i64], dtype: DType, bounds: BoxBounds) -> SpaceSpec {
        space(
            shape,
            dtype,
            SpaceKind::Box(BoxSpec {
                bounds: Some(bounds),
            }),
        )
    }

    fn render(space: &SpaceSpec) -> String {
        serde_json::to_string(&space_spec_to_json(space).expect("renders")).unwrap()
    }

    #[test]
    fn box_bounds_render_per_kind() {
        let uniform = boxed(
            &[3],
            DType::Float32,
            BoxBounds::Uniform(UniformBounds {
                low: -1.0,
                high: 1.0,
            }),
        );
        assert_eq!(
            render(&uniform),
            r#"{"details":{"bounds_kind":"uniform","high":1.0,"low":-1.0},"dtype":"float32","kind":"box","shape":[3]}"#
        );
        // A half-bounded Box: the infinite edges are null, as the envelope wants.
        let half = boxed(
            &[2],
            DType::Float64,
            BoxBounds::Elementwise(ElementwiseBounds {
                low: vec![f64::NEG_INFINITY, 0.0],
                high: vec![2.5, f64::INFINITY],
            }),
        );
        assert_eq!(
            render(&half),
            r#"{"details":{"bounds_kind":"elementwise","high":[2.5,null],"low":[null,0.0]},"dtype":"float64","kind":"box","shape":[2]}"#
        );
        let unbounded = boxed(&[1], DType::Unspecified, BoxBounds::Unbounded(true));
        assert_eq!(
            render(&unbounded),
            r#"{"details":{"bounds_kind":"unbounded"},"dtype":"float32","kind":"box","shape":[1]}"#
        );
        let none = space(
            &[1],
            DType::Float32,
            SpaceKind::Box(BoxSpec { bounds: None }),
        );
        assert_eq!(
            render(&none),
            r#"{"details":{"bounds_kind":null},"dtype":"float32","kind":"box","shape":[1]}"#
        );
    }

    #[test]
    fn typed_bounds_stay_exact_in_their_dtype() {
        let bytes = |values: &[Scalar], dtype| encode_scalars(values, dtype).unwrap();
        let image = boxed(
            &[2, 2, 3],
            DType::Uint8,
            BoxBounds::TypedUniform(TypedUniformBounds {
                low: bytes(&[Scalar::Int(0)], DType::Uint8),
                high: bytes(&[Scalar::Int(255)], DType::Uint8),
            }),
        );
        assert_eq!(
            render(&image),
            r#"{"details":{"bounds_kind":"typed_uniform","high":[255],"low":[0]},"dtype":"uint8","kind":"box","shape":[2,2,3]}"#
        );
        let wide = boxed(
            &[2],
            DType::Uint64,
            BoxBounds::TypedElementwise(TypedElementwiseBounds {
                low: bytes(&[Scalar::Int(0), Scalar::Int(1)], DType::Uint64),
                high: bytes(&[Scalar::Int(-1), Scalar::Int(i64::MAX)], DType::Uint64),
            }),
        );
        assert_eq!(
            render(&wide),
            r#"{"details":{"bounds_kind":"typed_elementwise","high":[18446744073709551615,9223372036854775807],"low":[0,1]},"dtype":"uint64","kind":"box","shape":[2]}"#
        );
        let truncated = boxed(
            &[1],
            DType::Int32,
            BoxBounds::TypedUniform(TypedUniformBounds {
                low: vec![0, 0, 0],
                high: vec![0, 0, 0, 0],
            }),
        );
        assert!(matches!(
            space_spec_to_json(&truncated).unwrap_err(),
            SpaceJsonError::Bounds(_)
        ));
    }

    #[test]
    fn every_rendered_space_reads_back_exactly() {
        let bytes = |values: &[Scalar], dtype| encode_scalars(values, dtype).unwrap();
        let spaces = [
            boxed(
                &[2],
                DType::Float64,
                BoxBounds::Elementwise(ElementwiseBounds {
                    low: vec![f64::NEG_INFINITY, 0.1],
                    high: vec![1.0 / 3.0, f64::INFINITY],
                }),
            ),
            boxed(
                &[2],
                DType::Uint64,
                BoxBounds::TypedElementwise(TypedElementwiseBounds {
                    low: bytes(&[Scalar::Int(0), Scalar::Int(1)], DType::Uint64),
                    high: bytes(&[Scalar::Int(-1), Scalar::Int(i64::MAX)], DType::Uint64),
                }),
            ),
            boxed(
                &[3],
                DType::Int8,
                BoxBounds::TypedUniform(TypedUniformBounds {
                    low: bytes(&[Scalar::Int(-128)], DType::Int8),
                    high: bytes(&[Scalar::Int(127)], DType::Int8),
                }),
            ),
            boxed(
                &[1],
                DType::Float32,
                BoxBounds::Uniform(UniformBounds {
                    low: f64::NEG_INFINITY,
                    high: 2.5,
                }),
            ),
            boxed(&[1], DType::Float32, BoxBounds::Unbounded(true)),
            space(
                &[2, 2],
                DType::Int64,
                SpaceKind::MultiDiscrete(MultiDiscreteSpec {
                    nvec: vec![2, 3, 4, 5],
                }),
            ),
            space(
                &[2, 3],
                DType::Int8,
                SpaceKind::MultiBinary(MultiBinarySpec),
            ),
            space(
                &[],
                DType::Uint8,
                SpaceKind::Text(TextSpec {
                    min_length: 1,
                    max_length: 8,
                    charset: "ab".into(),
                }),
            ),
            space(
                &[],
                DType::Unspecified,
                SpaceKind::Tuple(TupleSpec {
                    spaces: vec![space(
                        &[],
                        DType::Int32,
                        SpaceKind::Discrete(DiscreteSpec { n: 4, start: -1 }),
                    )],
                }),
            ),
        ];
        for spec in spaces {
            let json = space_spec_to_json(&spec).unwrap();
            assert_eq!(space_spec_from_json(&json).unwrap(), spec, "{json}");
        }
        // A Dict reads its order from `details.keys`.
        let discrete = space(
            &[],
            DType::Int64,
            SpaceKind::Discrete(DiscreteSpec { n: 2, start: 0 }),
        );
        let dict = space(
            &[],
            DType::Unspecified,
            SpaceKind::Dict(DictSpec {
                keys: vec!["b".into(), "a".into()],
                spaces: vec![discrete.clone(), discrete],
            }),
        );
        let mut json = space_spec_to_json(&dict).unwrap();
        json["details"]["keys"] = json!(["b", "a"]);
        assert_eq!(space_spec_from_json(&json).unwrap(), dict);
        assert!(space_spec_from_json(&json!({"kind": "box"})).is_err());
    }

    #[test]
    fn discrete_family_and_text() {
        let discrete = space(
            &[],
            DType::Int64,
            SpaceKind::Discrete(DiscreteSpec { n: 4, start: -1 }),
        );
        assert_eq!(
            render(&discrete),
            r#"{"details":{"n":4,"start":-1},"dtype":"int64","kind":"discrete","shape":[]}"#
        );
        let flat = space(&[5], DType::Int8, SpaceKind::MultiBinary(MultiBinarySpec));
        assert!(render(&flat).contains(r#""details":{"size":5}"#));
        let grid = space(
            &[2, 3],
            DType::Int8,
            SpaceKind::MultiBinary(MultiBinarySpec),
        );
        assert!(render(&grid).contains(r#""details":{"dims":[2,3]}"#));
        let matrix = space(
            &[2, 2],
            DType::Int64,
            SpaceKind::MultiDiscrete(MultiDiscreteSpec {
                nvec: vec![2, 3, 4, 5],
            }),
        );
        assert!(render(&matrix).contains(r#""details":{"nvec":[[2,3],[4,5]]}"#));
        let vector = space(
            &[2],
            DType::Int64,
            SpaceKind::MultiDiscrete(MultiDiscreteSpec { nvec: vec![2, 3] }),
        );
        assert!(render(&vector).contains(r#""details":{"nvec":[2,3]}"#));
        let text = space(
            &[],
            DType::Unspecified,
            SpaceKind::Text(TextSpec {
                min_length: 0,
                max_length: 8,
                charset: "ab".into(),
            }),
        );
        assert!(
            render(&text).contains(r#""details":{"charset":"ab","max_length":8,"min_length":0}"#)
        );
    }

    #[test]
    fn composites_nest_the_same_form() {
        let discrete = space(
            &[],
            DType::Int64,
            SpaceKind::Discrete(DiscreteSpec { n: 2, start: 0 }),
        );
        let tuple = space(
            &[],
            DType::Unspecified,
            SpaceKind::Tuple(TupleSpec {
                spaces: vec![discrete.clone()],
            }),
        );
        let dict = space(
            &[],
            DType::Unspecified,
            SpaceKind::Dict(DictSpec {
                keys: vec!["b".into(), "a".into()],
                spaces: vec![discrete, tuple],
            }),
        );
        assert_eq!(
            render(&dict),
            concat!(
                r#"{"details":{"spaces":{"#,
                r#""a":{"details":{"spaces":[{"details":{"n":2,"start":0},"dtype":"int64","kind":"discrete","shape":[]}]},"dtype":"float32","kind":"tuple","shape":[]},"#,
                r#""b":{"details":{"n":2,"start":0},"dtype":"int64","kind":"discrete","shape":[]}"#,
                r#"}},"dtype":"float32","kind":"dict","shape":[]}"#,
            )
        );
        let missing = SpaceSpec::default();
        assert!(matches!(
            space_spec_to_json(&missing).unwrap_err(),
            SpaceJsonError::Missing
        ));
    }
}
