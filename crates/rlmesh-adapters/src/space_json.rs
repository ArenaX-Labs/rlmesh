//! The `SpaceSpec` JSON form a describe envelope embeds under
//! `env_spec.observation_space` / `env_spec.action_space`.
//!
//! `{"kind", "shape", "dtype", "details": {...}}`, the same structure the Python
//! `SpaceSpec._to_dict()` returns (the managed platform reads it from either
//! producer). The envelope is strict JSON, so a non-finite bound renders as
//! `null` (serde_json's own rendering of a non-finite `f64`): unbounded on that
//! edge, exactly as the Python gatherer maps it.

use rlmesh_spaces::scalar::{Scalar, decode_scalars};
use rlmesh_spaces::spaces::{SpaceKind, SpaceSpec};
use rlmesh_spaces::{BoxBounds, BoxSpec, DType};
use serde_json::{Map, Value, json};

/// A space that cannot be rendered: no spec at all, or typed Box bounds whose
/// bytes do not decode in the space's dtype.
#[derive(Debug, thiserror::Error)]
pub enum SpaceJsonError {
    #[error("space spec is missing")]
    Missing,
    #[error("{0}")]
    Bounds(String),
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
    use rlmesh_spaces::scalar::encode_scalars;
    use rlmesh_spaces::{
        DictSpec, DiscreteSpec, ElementwiseBounds, MultiBinarySpec, MultiDiscreteSpec, TextSpec,
        TupleSpec, TypedElementwiseBounds, TypedUniformBounds, UniformBounds,
    };

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
