//! A compact, serializable projection of a gymnasium space.
//!
//! `rlmesh-adapters` resolves IO plans from the *structure* of an env's
//! observation/action spaces (keys, widths, dtypes, bounds) plus the
//! semantic tags layered on top. It does not depend on the proto wire
//! form of spaces; [`join`](crate::join::join) and the conformance vectors consume
//! this view, derived from [`rlmesh_spaces::SpaceSpec`] via [`From`].
//!
//! The view is intentionally lossy: it keeps only what role resolution and
//! validation need. Bounds are projected to `f64` (the byte-typed integer
//! bounds are decoded here, once), so downstream code never re-derives them.

use rlmesh_spaces::{BoxBounds, DType, SpaceKind, SpaceSpec, decode_scalars};
use serde::{Deserialize, Serialize};

/// The space family of a [`SpaceView`] node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpaceViewKind {
    Box,
    Discrete,
    MultiBinary,
    MultiDiscrete,
    Text,
    Dict,
    Tuple,
    Unspecified,
}

/// A structural view of one space (possibly nested via `Dict`/`Tuple`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpaceView {
    pub kind: SpaceViewKind,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub shape: Vec<i64>,
    pub dtype: String,
    /// Finite-or-infinite lower bounds in row-major order; `None` when the
    /// space declares no bounds. Callers filter for finiteness as needed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub low: Option<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub high: Option<Vec<f64>>,
    /// `Dict` child keys, parallel to `children`. Empty for non-dicts.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub keys: Vec<String>,
    /// `Dict` values (parallel to `keys`) or `Tuple` items.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<SpaceView>,
}

impl SpaceView {
    /// Number of scalar elements in this node's own shape (`1` for a scalar
    /// shape `[]`). Negative dims (unknown) are treated as `0`.
    pub fn numel(&self) -> usize {
        self.shape
            .iter()
            .map(|&dim| usize::try_from(dim).unwrap_or(0))
            .product()
    }

    /// The `Dict` child registered under `key`, if any.
    pub fn child(&self, key: &str) -> Option<&SpaceView> {
        if self.kind != SpaceViewKind::Dict {
            return None;
        }
        self.keys
            .iter()
            .position(|candidate| candidate == key)
            .and_then(|index| self.children.get(index))
    }

    /// The `Tuple` child at positional `index`, if any.
    ///
    /// The positional analogue of [`child`](Self::child): `Tuple` children
    /// carry no keys (they are addressed by order), so the recursive `join`
    /// walk descends a Tuple observation node by index.
    pub fn child_at(&self, index: usize) -> Option<&SpaceView> {
        if self.kind != SpaceViewKind::Tuple {
            return None;
        }
        self.children.get(index)
    }
}

/// Project the `f64` bounds out of a Box space, decoding byte-typed integer
/// bounds exactly. `None` means the space is unbounded on that side.
fn box_bounds_f64(
    bounds: Option<&BoxBounds>,
    dtype: DType,
) -> (Option<Vec<f64>>, Option<Vec<f64>>) {
    match bounds {
        None | Some(BoxBounds::Unbounded(_)) => (None, None),
        Some(BoxBounds::Uniform(uniform)) => (Some(vec![uniform.low]), Some(vec![uniform.high])),
        Some(BoxBounds::Elementwise(elementwise)) => (
            Some(elementwise.low.clone()),
            Some(elementwise.high.clone()),
        ),
        Some(BoxBounds::TypedUniform(typed)) => (
            decode_typed(&typed.low, dtype),
            decode_typed(&typed.high, dtype),
        ),
        Some(BoxBounds::TypedElementwise(typed)) => (
            decode_typed(&typed.low, dtype),
            decode_typed(&typed.high, dtype),
        ),
    }
}

/// Decode little-endian dtype bytes into `f64` bounds. A malformed buffer
/// yields `None` (treated as unbounded) rather than panicking.
fn decode_typed(bytes: &[u8], dtype: DType) -> Option<Vec<f64>> {
    decode_scalars(bytes, dtype)
        .ok()
        .map(|scalars| scalars.iter().map(|scalar| scalar.to_f64(dtype)).collect())
}

impl From<&SpaceSpec> for SpaceView {
    fn from(spec: &SpaceSpec) -> Self {
        let kind = match &spec.spec {
            Some(SpaceKind::Box(_)) => SpaceViewKind::Box,
            Some(SpaceKind::Discrete(_)) => SpaceViewKind::Discrete,
            Some(SpaceKind::MultiBinary(_)) => SpaceViewKind::MultiBinary,
            Some(SpaceKind::MultiDiscrete(_)) => SpaceViewKind::MultiDiscrete,
            Some(SpaceKind::Text(_)) => SpaceViewKind::Text,
            Some(SpaceKind::Dict(_)) => SpaceViewKind::Dict,
            Some(SpaceKind::Tuple(_)) => SpaceViewKind::Tuple,
            None => SpaceViewKind::Unspecified,
        };
        let (low, high) = match &spec.spec {
            Some(SpaceKind::Box(box_spec)) => box_bounds_f64(box_spec.bounds.as_ref(), spec.dtype),
            _ => (None, None),
        };
        let (keys, children) = match &spec.spec {
            Some(SpaceKind::Dict(dict)) => (
                dict.keys.clone(),
                dict.spaces.iter().map(SpaceView::from).collect(),
            ),
            Some(SpaceKind::Tuple(tuple)) => (
                Vec::new(),
                tuple.spaces.iter().map(SpaceView::from).collect(),
            ),
            _ => (Vec::new(), Vec::new()),
        };
        SpaceView {
            kind,
            shape: spec.shape.clone(),
            dtype: spec.dtype.name().to_owned(),
            low,
            high,
            keys,
            children,
        }
    }
}

#[cfg(test)]
mod tests {
    use rlmesh_spaces::{BoxSpec, DictSpec, TypedUniformBounds, UniformBounds};

    use super::*;

    fn box_spec(shape: Vec<i64>, dtype: DType, bounds: Option<BoxBounds>) -> SpaceSpec {
        SpaceSpec {
            shape,
            dtype,
            spec: Some(SpaceKind::Box(BoxSpec { bounds })),
        }
    }

    #[test]
    fn projects_uniform_box() {
        let spec = box_spec(
            vec![3],
            DType::Float32,
            Some(BoxBounds::Uniform(UniformBounds {
                low: -1.0,
                high: 1.0,
            })),
        );
        let view = SpaceView::from(&spec);
        assert_eq!(view.kind, SpaceViewKind::Box);
        assert_eq!(view.shape, vec![3]);
        assert_eq!(view.dtype, "float32");
        assert_eq!(view.low, Some(vec![-1.0]));
        assert_eq!(view.high, Some(vec![1.0]));
        assert_eq!(view.numel(), 3);
    }

    #[test]
    fn projects_unbounded_box_as_none() {
        let spec = box_spec(vec![2, 2], DType::Float64, Some(BoxBounds::Unbounded(true)));
        let view = SpaceView::from(&spec);
        assert_eq!(view.low, None);
        assert_eq!(view.high, None);
        assert_eq!(view.numel(), 4);
    }

    #[test]
    fn decodes_typed_integer_bounds_exactly() {
        // uint8 bounds carried as raw bytes: low 0, high 255.
        let spec = box_spec(
            vec![1],
            DType::Uint8,
            Some(BoxBounds::TypedUniform(TypedUniformBounds {
                low: vec![0],
                high: vec![255],
            })),
        );
        let view = SpaceView::from(&spec);
        assert_eq!(view.low, Some(vec![0.0]));
        assert_eq!(view.high, Some(vec![255.0]));
    }

    #[test]
    fn traverses_dict_children_by_key() {
        let spec = SpaceSpec {
            shape: Vec::new(),
            dtype: DType::Unspecified,
            spec: Some(SpaceKind::Dict(DictSpec {
                keys: vec!["image".to_owned(), "state".to_owned()],
                spaces: vec![
                    box_spec(vec![64, 64, 3], DType::Uint8, None),
                    box_spec(vec![7], DType::Float32, None),
                ],
            })),
        };
        let view = SpaceView::from(&spec);
        assert_eq!(view.kind, SpaceViewKind::Dict);
        assert_eq!(view.child("state").map(SpaceView::numel), Some(7));
        assert_eq!(
            view.child("image").map(|child| child.shape.clone()),
            Some(vec![64, 64, 3])
        );
        assert!(view.child("missing").is_none());
    }

    #[test]
    fn child_at_indexes_tuple_children() {
        let spec = SpaceSpec {
            shape: Vec::new(),
            dtype: DType::Unspecified,
            spec: Some(SpaceKind::Tuple(rlmesh_spaces::TupleSpec {
                spaces: vec![
                    box_spec(vec![64, 64, 3], DType::Uint8, None),
                    box_spec(vec![7], DType::Float32, None),
                ],
            })),
        };
        let view = SpaceView::from(&spec);
        assert_eq!(view.kind, SpaceViewKind::Tuple);
        assert_eq!(
            view.child_at(0).map(|child| child.shape.clone()),
            Some(vec![64, 64, 3])
        );
        assert_eq!(view.child_at(1).map(SpaceView::numel), Some(7));
        assert!(view.child_at(2).is_none());
        // A Dict-only accessor never descends a Tuple, and vice-versa.
        assert!(view.child("0").is_none());
    }
}
