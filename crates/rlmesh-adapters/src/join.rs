//! Derive the internal [`EnvFeatures`] from sparse [`EnvTags`] layered
//! over a gymnasium space.
//!
//! `join` is the single place env semantics meet env structure. It runs at two
//! seams with identical rules: at authoring time (when an env publishes
//! tags, so mistakes fail fast) and worker-side at resolve time (from
//! the untrusted handshake contract). Every failure names which side
//! disagreed — the tag or the space.

use crate::path::NodePath;
use crate::space_view::{SpaceView, SpaceViewKind};
use crate::spec::{
    Action, Actuator, EnvFeature, EnvFeatures, EnvImage, EnvState, EnvTags, EnvText, ImageLayout,
    ObsLeaf, ObsNode, SplitLayout,
};

/// A validation failure while joining tags against a space.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum JoinError {
    #[error("observation key {key:?} does not resolve to a leaf of the observation space")]
    KeyNotInSpace { key: String },
    #[error("observation {key:?} is tagged as {expected} but the space is {actual}")]
    ClassMismatch {
        key: String,
        expected: &'static str,
        actual: String,
    },
    #[error("observation tuple at {path:?} declares {actual} item(s) but the space has {expected}")]
    TupleArityMismatch {
        path: String,
        expected: usize,
        actual: usize,
    },
    #[error("observation tuple at {path:?} index [{index}] is out of range")]
    TupleIndexOutOfRange { path: String, index: usize },
    #[error(
        "state {key:?} declares encoding {encoding} ({dims} dims) but the space width is {width}"
    )]
    EncodingWidthMismatch {
        key: String,
        encoding: &'static str,
        dims: u32,
        width: u32,
    },
    #[error("state layout for {key:?} sums to {layout_sum} dims but the space width is {width}")]
    StateLayoutWidthMismatch {
        key: String,
        layout_sum: u32,
        width: u32,
    },
    #[error("state layout for {key:?} declares dims that overflow u32")]
    StateLayoutWidthOverflow { key: String },
    #[error("state layout for {key:?} declares role {role:?} more than once")]
    DuplicateLayoutRole { key: String, role: String },
    #[error(
        "{key:?} tag range {tag:?} disagrees with the space's finite bounds \
         {space:?}"
    )]
    RangeDisagreement {
        key: String,
        tag: (f64, f64),
        space: (f64, f64),
    },
    #[error("the action space must be a flat Box but it is {actual}")]
    ActionClass { actual: String },
    #[error(
        "action component dims sum to {component_sum} but the action space width is {action_width}"
    )]
    ActionWidthMismatch {
        action_width: u32,
        component_sum: u32,
    },
    #[error("action component dims overflow u32")]
    ActionWidthOverflow,
    #[error(
        "action component {role:?} declares encoding {encoding} ({dims} dims) but its dim is {dim}"
    )]
    ActionEncodingMismatch {
        role: String,
        encoding: &'static str,
        dims: u32,
        dim: u32,
    },
}

type Result<T> = std::result::Result<T, JoinError>;

/// Join the env observation tree and action layout against their spaces.
pub fn join(
    tags: &EnvTags,
    obs_space: &SpaceView,
    action_space: &SpaceView,
) -> Result<EnvFeatures> {
    let mut observation = Vec::new();
    join_node(
        &tags.observation,
        obs_space,
        &NodePath::root(),
        &mut observation,
    )?;
    let action = resolve_action(&tags.action, action_space)?;
    Ok(EnvFeatures {
        observation,
        action,
    })
}

/// Walk the observation tree in lockstep with the space, flattening each leaf
/// into the `Vec<EnvFeature>` the resolver consumes. A `Dict` node descends each
/// key (the space must be a `Dict` carrying that key); a `Tuple` node descends
/// each index positionally (the space must be a same-arity `Tuple`); a `Leaf`
/// joins against the space leaf at the current `source` path.
fn join_node(
    node: &ObsNode,
    view: &SpaceView,
    source: &NodePath,
    out: &mut Vec<EnvFeature>,
) -> Result<()> {
    match node {
        ObsNode::Leaf(leaf) => out.extend(join_feature(source, leaf, view)?),
        ObsNode::Dict(map) => {
            if view.kind != SpaceViewKind::Dict {
                return Err(JoinError::ClassMismatch {
                    key: source.to_string(),
                    expected: "a Dict space",
                    actual: describe_space(view),
                });
            }
            for (key, child) in map {
                let child_view = view.child(key).ok_or_else(|| JoinError::KeyNotInSpace {
                    key: source.push_key(key.clone()).to_string(),
                })?;
                join_node(child, child_view, &source.push_key(key.clone()), out)?;
            }
        }
        ObsNode::Tuple(items) => {
            if view.kind != SpaceViewKind::Tuple {
                return Err(JoinError::ClassMismatch {
                    key: source.to_string(),
                    expected: "a Tuple space",
                    actual: describe_space(view),
                });
            }
            if items.len() != view.children.len() {
                return Err(JoinError::TupleArityMismatch {
                    path: source.to_string(),
                    expected: view.children.len(),
                    actual: items.len(),
                });
            }
            for (index, item) in items.iter().enumerate() {
                let child_view =
                    view.child_at(index)
                        .ok_or_else(|| JoinError::TupleIndexOutOfRange {
                            path: source.to_string(),
                            index,
                        })?;
                join_node(item, child_view, &source.push_index(index), out)?;
            }
        }
    }
    Ok(())
}

fn join_feature(
    source: &NodePath,
    leaf_tag: &ObsLeaf,
    leaf: &SpaceView,
) -> Result<Vec<EnvFeature>> {
    let path = source.to_string();
    match leaf_tag {
        ObsLeaf::Image(image) => {
            if leaf.kind != SpaceViewKind::Box || leaf.shape.len() != 3 {
                return Err(JoinError::ClassMismatch {
                    key: path,
                    expected: "an image (3-D Box)",
                    actual: describe_space(leaf),
                });
            }
            let (height, width, channels) = image_hwc(&leaf.shape, image.layout);
            Ok(vec![EnvFeature::Image(EnvImage {
                source: source.clone(),
                role: image.role.clone(),
                layout: image.layout,
                upside_down: image.upside_down,
                height,
                width,
                channels,
                value_range: uniform_finite_range(leaf),
            })])
        }
        ObsLeaf::State(state) => {
            if !is_numeric(leaf) {
                return Err(JoinError::ClassMismatch {
                    key: path,
                    expected: "a numeric state",
                    actual: describe_space(leaf),
                });
            }
            let width = width_of(leaf);
            // Validate the *native* (raw) encoding's width against the space:
            // the first recognized entry is what the env actually produces.
            if let Some(native) = state.encoding.as_ref().and_then(|set| set.first_known())
                && width != native.dims()
            {
                return Err(JoinError::EncodingWidthMismatch {
                    key: path,
                    encoding: native.as_str(),
                    dims: native.dims(),
                    width,
                });
            }
            let range = reconcile_range(uniform_finite_range(leaf), state.range, &path)?;
            Ok(vec![EnvFeature::State(EnvState {
                source: source.clone(),
                role: state.role.clone(),
                slice_offset: None,
                dim: Some(width),
                encoding: state.encoding.clone(),
                range,
            })])
        }
        ObsLeaf::Split(layout) => join_split(source, layout, leaf),
        ObsLeaf::Text(text) => {
            if leaf.kind != SpaceViewKind::Text {
                return Err(JoinError::ClassMismatch {
                    key: path,
                    expected: "a text space",
                    actual: describe_space(leaf),
                });
            }
            Ok(vec![EnvFeature::Text(EnvText {
                source: source.clone(),
                role: text.role.clone(),
            })])
        }
    }
}

/// Split one flat numeric leaf into a [`EnvState`] per role field of a layout.
///
/// Fields are laid out in order; offsets accumulate; the field widths must sum
/// to the leaf width (mirroring the action width law). A role-less field is a
/// skip — it advances the offset but emits no feature. Each role field's range
/// is derived from its own slice of the leaf's bounds.
fn join_split(
    source: &NodePath,
    layout: &SplitLayout,
    leaf: &SpaceView,
) -> Result<Vec<EnvFeature>> {
    let path = source.to_string();
    if !is_numeric(leaf) {
        return Err(JoinError::ClassMismatch {
            key: path,
            expected: "a numeric state",
            actual: describe_space(leaf),
        });
    }
    let width = width_of(leaf);
    let layout_sum = layout
        .fields
        .iter()
        .try_fold(0u32, |acc, field| acc.checked_add(field.dim))
        .ok_or_else(|| JoinError::StateLayoutWidthOverflow { key: path.clone() })?;
    if layout_sum != width {
        return Err(JoinError::StateLayoutWidthMismatch {
            key: path,
            layout_sum,
            width,
        });
    }
    let mut features = Vec::new();
    let mut seen_roles: Vec<&str> = Vec::new();
    let mut offset: u32 = 0;
    for field in &layout.fields {
        if let Some(role) = &field.role {
            if let Some(native) = field.encoding.as_ref().and_then(|set| set.first_known())
                && field.dim != native.dims()
            {
                return Err(JoinError::EncodingWidthMismatch {
                    key: path,
                    encoding: native.as_str(),
                    dims: native.dims(),
                    width: field.dim,
                });
            }
            if seen_roles.contains(&role.as_str()) {
                return Err(JoinError::DuplicateLayoutRole {
                    key: path,
                    role: role.clone(),
                });
            }
            seen_roles.push(role.as_str());
            let space_range = slice_uniform_finite_range(leaf, offset, field.dim);
            let range = reconcile_range(space_range, field.range, &path)?;
            features.push(EnvFeature::State(EnvState {
                source: source.clone(),
                role: role.clone(),
                slice_offset: Some(offset),
                dim: Some(field.dim),
                encoding: field.encoding.clone(),
                range,
            }));
        }
        offset += field.dim;
    }
    Ok(features)
}

/// Reconcile a state's value range: take the space's finite bounds when
/// uniform, overridden by an explicit tag only where the space is unbounded.
/// A finite space range that disagrees with an explicit tag is an error.
/// Shared by state features, state-layout fields, and action components.
fn reconcile_range(
    space: Option<(f64, f64)>,
    tag: Option<(f64, f64)>,
    key: &str,
) -> Result<Option<(f64, f64)>> {
    match (space, tag) {
        (Some(space), Some(tag)) => {
            if ranges_agree(space, tag) {
                // The explicit tag is the author's stated intent; keep
                // it exactly (the space's bounds may be the same value at a
                // narrower precision).
                Ok(Some(tag))
            } else {
                Err(JoinError::RangeDisagreement {
                    key: key.to_owned(),
                    tag,
                    space,
                })
            }
        }
        (Some(space), None) => Ok(Some(space)),
        (None, tag) => Ok(tag),
    }
}

/// The uniform finite `(low, high)` of a field's `[offset, offset+dim)` slice of
/// a leaf's bounds, or `None` if unbounded, non-finite, or non-uniform within
/// the slice. Length-1 bounds are a uniform broadcast over the whole leaf.
fn slice_uniform_finite_range(leaf: &SpaceView, offset: u32, dim: u32) -> Option<(f64, f64)> {
    let lo = slice_uniform(leaf.low.as_deref()?, offset, dim)?;
    let hi = slice_uniform(leaf.high.as_deref()?, offset, dim)?;
    if !lo.is_finite() || !hi.is_finite() {
        return None;
    }
    Some((lo, hi))
}

/// The single bound value across a field's slice of one side's bounds, or
/// `None` if the slice is out of range or non-uniform. A length-1 vec is a
/// uniform broadcast and applies to every field.
fn slice_uniform(bounds: &[f64], offset: u32, dim: u32) -> Option<f64> {
    if bounds.len() == 1 {
        return Some(bounds[0]);
    }
    let start = offset as usize;
    let slice = bounds.get(start..start + dim as usize)?;
    let first = *slice.first()?;
    if slice.iter().any(|&value| value != first) {
        return None;
    }
    Some(first)
}

/// Whether two `(low, high)` ranges agree up to floating-point rounding.
///
/// A space stores its bounds at the dtype's precision, so a float32
/// gymnasium `Box(_, 0.08, _)` projects to `0.0799999982`; an exact compare
/// against a tag's `0.08` would reject ranges that are equal in
/// intent. The tolerance (~8x float32 epsilon, with an absolute floor) still
/// rejects genuine disagreements.
fn ranges_agree(a: (f64, f64), b: (f64, f64)) -> bool {
    fn close(x: f64, y: f64) -> bool {
        (x - y).abs() <= 1e-6 * x.abs().max(y.abs()).max(1.0)
    }
    close(a.0, b.0) && close(a.1, b.1)
}

/// The single finite `(low, high)` pair a space declares, or `None` if it is
/// unbounded, non-finite, or per-element non-uniform. The whole-leaf case is
/// just the `[0, width)` slice of [`slice_uniform_finite_range`].
fn uniform_finite_range(leaf: &SpaceView) -> Option<(f64, f64)> {
    slice_uniform_finite_range(leaf, 0, width_of(leaf))
}

/// Validate the action layout against the action space and derive each
/// component's value range from the space's finite bounds, the way
/// [`join_feature`] does for observation state. Without this a model output is
/// passed through unmapped even when the env's `action_space` declares finite
/// bounds the model's range should map into (e.g. model `[-1, 1]` into an env
/// `Box(0, 1)` action). Components are laid out in order; each derives its
/// range from its own `[offset, offset+dim)` slice of the action bounds.
fn resolve_action(action: &Action, action_space: &SpaceView) -> Result<Action> {
    if action_space.kind != SpaceViewKind::Box {
        return Err(JoinError::ActionClass {
            actual: describe_space(action_space),
        });
    }
    let action_width = width_of(action_space);
    let component_sum = action
        .components
        .iter()
        .try_fold(0u32, |acc, component| acc.checked_add(component.dim))
        .ok_or(JoinError::ActionWidthOverflow)?;
    if component_sum != action_width {
        return Err(JoinError::ActionWidthMismatch {
            action_width,
            component_sum,
        });
    }
    let mut components = Vec::with_capacity(action.components.len());
    let mut offset: u32 = 0;
    for component in &action.components {
        if let Some(encoding) = component.encoding
            && component.dim != encoding.dims()
        {
            return Err(JoinError::ActionEncodingMismatch {
                role: component.role.clone(),
                encoding: encoding.as_str(),
                dims: encoding.dims(),
                dim: component.dim,
            });
        }
        let space_range = slice_uniform_finite_range(action_space, offset, component.dim);
        let range = reconcile_range(space_range, component.range, &component.role)?;
        components.push(Actuator {
            range,
            ..component.clone()
        });
        offset += component.dim;
    }
    Ok(Action {
        components,
        clip: action.clip,
        // The env declaration drives this join; preserve its replay horizon
        // (always 1 env-side, but carried through rather than reset by hand).
        execute_horizon: action.execute_horizon,
    })
}

fn is_numeric(view: &SpaceView) -> bool {
    matches!(
        view.kind,
        SpaceViewKind::Box
            | SpaceViewKind::Discrete
            | SpaceViewKind::MultiBinary
            | SpaceViewKind::MultiDiscrete
    )
}

fn width_of(view: &SpaceView) -> u32 {
    u32::try_from(view.numel()).unwrap_or(u32::MAX)
}

/// Pixel `(height, width, channels)` of a 3-D image shape under its layout.
fn image_hwc(shape: &[i64], layout: ImageLayout) -> (u32, u32, u32) {
    let dim = |index: usize| {
        shape
            .get(index)
            .copied()
            .and_then(|value| u32::try_from(value).ok())
            .unwrap_or(0)
    };
    match layout {
        ImageLayout::Hwc => (dim(0), dim(1), dim(2)),
        ImageLayout::Chw => (dim(1), dim(2), dim(0)),
    }
}

fn describe_space(view: &SpaceView) -> String {
    match view.kind {
        SpaceViewKind::Box => format!("a {}-D Box", view.shape.len()),
        SpaceViewKind::Discrete => "a Discrete space".to_owned(),
        SpaceViewKind::MultiBinary => "a MultiBinary space".to_owned(),
        SpaceViewKind::MultiDiscrete => "a MultiDiscrete space".to_owned(),
        SpaceViewKind::Text => "a Text space".to_owned(),
        SpaceViewKind::Dict => "a Dict space".to_owned(),
        SpaceViewKind::Tuple => "a Tuple space".to_owned(),
        SpaceViewKind::Unspecified => "an unspecified space".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::spec::{AcceptSet, RotationEncoding};
    use crate::spec::{Actuator, Field, ImageTag, SplitLayout, StateTag, TextTag};

    fn box_view(shape: Vec<i64>, low: Option<Vec<f64>>, high: Option<Vec<f64>>) -> SpaceView {
        SpaceView {
            kind: SpaceViewKind::Box,
            shape,
            dtype: "float32".to_owned(),
            low,
            high,
            keys: Vec::new(),
            children: Vec::new(),
        }
    }

    fn text_view() -> SpaceView {
        SpaceView {
            kind: SpaceViewKind::Text,
            shape: Vec::new(),
            dtype: "unspecified".to_owned(),
            low: None,
            high: None,
            keys: Vec::new(),
            children: Vec::new(),
        }
    }

    fn dict_view(entries: Vec<(&str, SpaceView)>) -> SpaceView {
        SpaceView {
            kind: SpaceViewKind::Dict,
            shape: Vec::new(),
            dtype: "unspecified".to_owned(),
            low: None,
            high: None,
            keys: entries.iter().map(|(key, _)| (*key).to_owned()).collect(),
            children: entries.into_iter().map(|(_, view)| view).collect(),
        }
    }

    fn action_layout(components: Vec<Actuator>) -> Action {
        Action {
            components,
            clip: None,
            execute_horizon: 1,
        }
    }

    fn component(role: &str, dim: u32, encoding: Option<RotationEncoding>) -> Actuator {
        Actuator {
            role: role.to_owned(),
            dim,
            encoding,
            range: None,
            scale: None,
            invert: false,
            threshold: None,
            binary: false,
        }
    }

    /// A single-key Dict observation tree carrying one leaf at `key`.
    fn leaf_at(key: &str, leaf: ObsLeaf) -> ObsNode {
        let mut map = BTreeMap::new();
        map.insert(key.to_owned(), ObsNode::Leaf(leaf));
        ObsNode::Dict(map)
    }

    /// An empty Dict observation tree (no tagged leaves).
    fn empty_obs() -> ObsNode {
        ObsNode::Dict(BTreeMap::new())
    }

    /// Join a single observation entry against its leaf, with an empty action.
    /// The source path of the joined feature is `key`.
    fn join_obs(key: &str, view: SpaceView, leaf: ObsLeaf) -> Result<EnvFeatures> {
        let obs = dict_view(vec![(key, view)]);
        let tags = EnvTags {
            observation: leaf_at(key, leaf),
            action: action_layout(vec![]),
        };
        join(&tags, &obs, &box_view(vec![0], None, None))
    }

    #[test]
    fn joins_image_state_text_and_action() {
        let obs = dict_view(vec![
            (
                "camera",
                box_view(vec![64, 64, 3], Some(vec![0.0]), Some(vec![255.0])),
            ),
            (
                "eef_pos",
                box_view(vec![3], Some(vec![-1.0]), Some(vec![1.0])),
            ),
            ("instruction", text_view()),
        ]);
        let action = box_view(vec![4], None, None);
        let mut observation = BTreeMap::new();
        observation.insert(
            "camera".to_owned(),
            ObsNode::Leaf(ObsLeaf::Image(ImageTag {
                role: "image/primary".to_owned(),
                layout: Default::default(),
                upside_down: false,
            })),
        );
        observation.insert(
            "eef_pos".to_owned(),
            ObsNode::Leaf(ObsLeaf::State(StateTag {
                role: "proprio/eef_pos".to_owned(),
                encoding: None,
                range: None,
            })),
        );
        observation.insert(
            "instruction".to_owned(),
            ObsNode::Leaf(ObsLeaf::Text(TextTag {
                role: "instruction".to_owned(),
            })),
        );
        let tags = EnvTags {
            observation: ObsNode::Dict(observation),
            action: action_layout(vec![component("action/delta_pos", 4, None)]),
        };

        let features = join(&tags, &obs, &action).expect("join");
        assert_eq!(features.observation.len(), 3);
        // The state's width and range are derived from the space.
        let state = features
            .observation
            .iter()
            .find_map(|feature| match feature {
                EnvFeature::State(state) => Some(state),
                _ => None,
            })
            .expect("state feature");
        assert_eq!(state.dim, Some(3));
        assert_eq!(state.range, Some((-1.0, 1.0)));
        assert_eq!(state.source.to_string(), "eef_pos");
    }

    #[test]
    fn joins_a_nested_dict_observation() {
        // The tree shape that replaces dotted keys: robot.eef_pos lives in a
        // nested Dict, addressed structurally; its source path renders dotted.
        let obs = dict_view(vec![(
            "robot",
            dict_view(vec![(
                "eef_pos",
                box_view(vec![3], Some(vec![-1.0]), Some(vec![1.0])),
            )]),
        )]);
        let action = box_view(vec![0], None, None);
        let mut robot = BTreeMap::new();
        robot.insert(
            "eef_pos".to_owned(),
            ObsNode::Leaf(ObsLeaf::State(StateTag {
                role: "proprio/eef_pos".to_owned(),
                encoding: None,
                range: None,
            })),
        );
        let mut root = BTreeMap::new();
        root.insert("robot".to_owned(), ObsNode::Dict(robot));
        let tags = EnvTags {
            observation: ObsNode::Dict(root),
            action: action_layout(vec![]),
        };
        let features = join(&tags, &obs, &action).expect("join");
        let EnvFeature::State(state) = &features.observation[0] else {
            panic!("expected state");
        };
        assert_eq!(state.source.to_string(), "robot.eef_pos");
    }

    #[test]
    fn joins_a_tuple_observation_positionally() {
        // A Tuple obs: items descend by index, source paths render `[i]`.
        let obs = SpaceView {
            kind: SpaceViewKind::Tuple,
            shape: Vec::new(),
            dtype: "unspecified".to_owned(),
            low: None,
            high: None,
            keys: Vec::new(),
            children: vec![
                box_view(vec![3], Some(vec![-1.0]), Some(vec![1.0])),
                text_view(),
            ],
        };
        let action = box_view(vec![0], None, None);
        let tags = EnvTags {
            observation: ObsNode::Tuple(vec![
                ObsNode::Leaf(ObsLeaf::State(StateTag {
                    role: "proprio/eef_pos".to_owned(),
                    encoding: None,
                    range: None,
                })),
                ObsNode::Leaf(ObsLeaf::Text(TextTag {
                    role: "instruction".to_owned(),
                })),
            ]),
            action: action_layout(vec![]),
        };
        let features = join(&tags, &obs, &action).expect("join");
        assert_eq!(features.observation.len(), 2);
        let EnvFeature::State(state) = &features.observation[0] else {
            panic!("expected state");
        };
        assert_eq!(state.source.to_string(), "[0]");
    }

    #[test]
    fn rejects_tuple_arity_mismatch() {
        let obs = SpaceView {
            kind: SpaceViewKind::Tuple,
            shape: Vec::new(),
            dtype: "unspecified".to_owned(),
            low: None,
            high: None,
            keys: Vec::new(),
            children: vec![box_view(vec![3], None, None)],
        };
        let action = box_view(vec![0], None, None);
        let tags = EnvTags {
            observation: ObsNode::Tuple(vec![
                ObsNode::Leaf(ObsLeaf::State(StateTag {
                    role: "a".to_owned(),
                    encoding: None,
                    range: None,
                })),
                ObsNode::Leaf(ObsLeaf::State(StateTag {
                    role: "b".to_owned(),
                    encoding: None,
                    range: None,
                })),
            ]),
            action: action_layout(vec![]),
        };
        assert!(matches!(
            join(&tags, &obs, &action),
            Err(JoinError::TupleArityMismatch {
                expected: 1,
                actual: 2,
                ..
            })
        ));
    }

    #[test]
    fn rejects_dict_node_against_non_dict_space() {
        // A Dict observation node against a flat Box space is a class mismatch.
        let obs = box_view(vec![3], None, None);
        let action = box_view(vec![0], None, None);
        let tags = EnvTags {
            observation: leaf_at(
                "eef_pos",
                ObsLeaf::State(StateTag {
                    role: "a".to_owned(),
                    encoding: None,
                    range: None,
                }),
            ),
            action: action_layout(vec![]),
        };
        assert!(matches!(
            join(&tags, &obs, &action),
            Err(JoinError::ClassMismatch {
                expected: "a Dict space",
                ..
            })
        ));
    }

    #[test]
    fn joins_a_bare_single_leaf_at_root() {
        // A single-leaf observation is a bare Leaf at the root (no key), and its
        // source path is the empty root.
        let obs = box_view(vec![3], Some(vec![-1.0]), Some(vec![1.0]));
        let action = box_view(vec![0], None, None);
        let tags = EnvTags {
            observation: ObsNode::Leaf(ObsLeaf::State(StateTag {
                role: "proprio/eef_pos".to_owned(),
                encoding: None,
                range: None,
            })),
            action: action_layout(vec![]),
        };
        let features = join(&tags, &obs, &action).expect("join");
        let EnvFeature::State(state) = &features.observation[0] else {
            panic!("expected state");
        };
        assert!(state.source.is_root());
        assert_eq!(state.dim, Some(3));
    }

    #[test]
    fn rejects_key_not_in_space() {
        let obs = dict_view(vec![("camera", box_view(vec![8, 8, 3], None, None))]);
        let tags = EnvTags {
            observation: leaf_at(
                "missing",
                ObsLeaf::Text(TextTag {
                    role: "instruction".to_owned(),
                }),
            ),
            action: action_layout(vec![]),
        };
        let action = box_view(vec![0], None, None);
        assert!(matches!(
            join(&tags, &obs, &action),
            Err(JoinError::KeyNotInSpace { key }) if key == "missing"
        ));
    }

    #[test]
    fn rejects_action_dim_sum_overflow() {
        // Declared component dims that overflow u32 are a clean error, not a
        // silent wrap (which would fabricate a false width-match in release).
        let obs = dict_view(vec![]);
        let action = box_view(vec![3], None, None);
        let tags = EnvTags {
            observation: empty_obs(),
            action: action_layout(vec![
                component("action/a", u32::MAX, None),
                component("action/b", 2, None),
            ]),
        };
        assert!(matches!(
            join(&tags, &obs, &action),
            Err(JoinError::ActionWidthOverflow)
        ));
    }

    #[test]
    fn rejects_state_layout_dim_sum_overflow() {
        let result = join_obs(
            "state",
            box_view(vec![3], None, None),
            ObsLeaf::Split(SplitLayout {
                fields: vec![
                    Field {
                        role: Some("a".to_owned()),
                        dim: u32::MAX,
                        encoding: None,
                        range: None,
                    },
                    Field {
                        role: Some("b".to_owned()),
                        dim: 2,
                        encoding: None,
                        range: None,
                    },
                ],
            }),
        );
        assert!(matches!(
            result,
            Err(JoinError::StateLayoutWidthOverflow { key }) if key == "state"
        ));
    }

    #[test]
    fn rejects_image_on_non_3d_box() {
        let result = join_obs(
            "camera",
            box_view(vec![64, 64], None, None),
            ObsLeaf::Image(ImageTag {
                role: "image/primary".to_owned(),
                layout: Default::default(),
                upside_down: false,
            }),
        );
        assert!(matches!(
            result,
            Err(JoinError::ClassMismatch { key, .. }) if key == "camera"
        ));
    }

    #[test]
    fn enforces_rotation_width_law_unconditionally() {
        // A quaternion encoding (4 dims) on a width-3 space.
        let result = join_obs(
            "rot",
            box_view(vec![3], None, None),
            ObsLeaf::State(StateTag {
                role: "proprio/eef_rot".to_owned(),
                encoding: Some(AcceptSet::single(RotationEncoding::QuatXyzw)),
                range: None,
            }),
        );
        assert!(matches!(
            result,
            Err(JoinError::EncodingWidthMismatch {
                width: 3,
                dims: 4,
                ..
            })
        ));
    }

    #[test]
    fn rejects_finite_range_disagreement_but_allows_unbounded_override() {
        let gripper = || {
            ObsLeaf::State(StateTag {
                role: "proprio/gripper".to_owned(),
                encoding: None,
                range: Some((0.0, 2.0)),
            })
        };
        // Finite space bounds [0, 1] disagree with tag [0, 2] -> error.
        assert!(matches!(
            join_obs(
                "g",
                box_view(vec![1], Some(vec![0.0]), Some(vec![1.0])),
                gripper(),
            ),
            Err(JoinError::RangeDisagreement { .. })
        ));

        // An unbounded space lets the tag supply the range.
        let features = join_obs("g", box_view(vec![1], None, None), gripper()).expect("join");
        let EnvFeature::State(state) = &features.observation[0] else {
            panic!("expected state");
        };
        assert_eq!(state.range, Some((0.0, 2.0)));
    }

    #[test]
    fn finite_bounds_agree_with_tag_up_to_float32_rounding() {
        // A float32 gymnasium Box(0, 0.08) projects its high bound to the
        // nearest float32, 0.0799999982; the tag declares 0.08. These
        // must agree (rounding, not disagreement), and the exact tag
        // value is kept.
        let f32_high = f64::from(0.08_f32);
        assert_ne!(f32_high, 0.08);
        let features = join_obs(
            "g",
            box_view(vec![1], Some(vec![0.0]), Some(vec![f32_high])),
            ObsLeaf::State(StateTag {
                role: "proprio/gripper".to_owned(),
                encoding: None,
                range: Some((0.0, 0.08)),
            }),
        )
        .expect("join");
        let EnvFeature::State(state) = &features.observation[0] else {
            panic!("expected state");
        };
        assert_eq!(state.range, Some((0.0, 0.08)));
    }

    fn field(role: Option<&str>, dim: u32, encoding: Option<RotationEncoding>) -> Field {
        Field {
            role: role.map(str::to_owned),
            dim,
            encoding: encoding.map(AcceptSet::single),
            range: None,
        }
    }

    /// A split-leaf at the root (the flat-Box case the old `"."` sentinel keyed).
    fn layout_tags(fields: Vec<Field>) -> EnvTags {
        EnvTags {
            observation: ObsNode::Leaf(ObsLeaf::Split(SplitLayout { fields })),
            action: action_layout(vec![]),
        }
    }

    #[test]
    fn splits_flat_layout_into_role_fields_with_offsets() {
        // A flat width-8 root obs split into three role fields plus a skip.
        let obs = box_view(vec![8], None, None);
        let action = box_view(vec![0], None, None);
        let tags = layout_tags(vec![
            field(Some("proprio/eef_pos"), 3, None),
            field(Some("proprio/gripper"), 1, None),
            field(None, 1, None), // skip
            field(Some("proprio/obj_pos"), 3, None),
        ]);
        let features = join(&tags, &obs, &action).expect("join");
        // Three role fields emitted; the skip produces nothing.
        assert_eq!(features.observation.len(), 3);
        let states: Vec<&EnvState> = features
            .observation
            .iter()
            .filter_map(|feature| match feature {
                EnvFeature::State(state) => Some(state),
                _ => None,
            })
            .collect();
        assert_eq!(states[0].role, "proprio/eef_pos");
        assert_eq!((states[0].slice_offset, states[0].dim), (Some(0), Some(3)));
        assert_eq!(states[1].role, "proprio/gripper");
        assert_eq!((states[1].slice_offset, states[1].dim), (Some(3), Some(1)));
        assert_eq!(states[2].role, "proprio/obj_pos");
        // Offset skips past the role-less field (3 + 1 + 1 = 5).
        assert_eq!((states[2].slice_offset, states[2].dim), (Some(5), Some(3)));
    }

    #[test]
    fn derives_layout_field_range_from_its_own_bounds_slice() {
        // Elementwise bounds: only the gripper field (index 3) is [0, 1].
        let low = vec![-1.0, -1.0, -1.0, 0.0, -1.0];
        let high = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let obs = box_view(vec![5], Some(low), Some(high));
        let action = box_view(vec![0], None, None);
        let tags = layout_tags(vec![
            field(Some("proprio/eef_pos"), 3, None),
            field(Some("proprio/gripper"), 1, None),
            field(Some("proprio/extra"), 1, None),
        ]);
        let features = join(&tags, &obs, &action).expect("join");
        let by_role = |role: &str| {
            features
                .observation
                .iter()
                .find_map(|feature| match feature {
                    EnvFeature::State(state) if state.role == role => Some(state.range),
                    _ => None,
                })
        };
        assert_eq!(by_role("proprio/eef_pos"), Some(Some((-1.0, 1.0))));
        assert_eq!(by_role("proprio/gripper"), Some(Some((0.0, 1.0))));
    }

    #[test]
    fn rejects_state_layout_width_mismatch() {
        let obs = box_view(vec![5], None, None);
        let action = box_view(vec![0], None, None);
        let tags = layout_tags(vec![
            field(Some("proprio/eef_pos"), 3, None),
            field(Some("proprio/gripper"), 1, None),
        ]);
        assert!(matches!(
            join(&tags, &obs, &action),
            Err(JoinError::StateLayoutWidthMismatch {
                layout_sum: 4,
                width: 5,
                ..
            })
        ));
    }

    #[test]
    fn rejects_duplicate_role_in_layout() {
        let obs = box_view(vec![6], None, None);
        let action = box_view(vec![0], None, None);
        let tags = layout_tags(vec![
            field(Some("proprio/eef_pos"), 3, None),
            field(Some("proprio/eef_pos"), 3, None),
        ]);
        assert!(matches!(
            join(&tags, &obs, &action),
            Err(JoinError::DuplicateLayoutRole { role, .. }) if role == "proprio/eef_pos"
        ));
    }

    #[test]
    fn rejects_layout_field_encoding_width_mismatch() {
        // A quaternion field (4 dims) declared as width 3.
        let obs = box_view(vec![3], None, None);
        let action = box_view(vec![0], None, None);
        let tags = layout_tags(vec![field(
            Some("proprio/eef_rot"),
            3,
            Some(RotationEncoding::QuatXyzw),
        )]);
        assert!(matches!(
            join(&tags, &obs, &action),
            Err(JoinError::EncodingWidthMismatch {
                width: 3,
                dims: 4,
                ..
            })
        ));
    }

    #[test]
    fn rejects_layout_on_non_numeric_leaf() {
        let result = join_obs(
            "text",
            text_view(),
            ObsLeaf::Split(SplitLayout {
                fields: vec![field(Some("proprio/eef_pos"), 3, None)],
            }),
        );
        assert!(matches!(
            result,
            Err(JoinError::ClassMismatch { key, .. }) if key == "text"
        ));
    }

    #[test]
    fn rejects_action_width_mismatch() {
        let obs = dict_view(vec![]);
        let action = box_view(vec![7], None, None);
        let tags = EnvTags {
            observation: empty_obs(),
            action: action_layout(vec![component("action/delta_pos", 3, None)]),
        };
        assert!(matches!(
            join(&tags, &obs, &action),
            Err(JoinError::ActionWidthMismatch {
                action_width: 7,
                component_sum: 3,
            })
        ));
    }

    #[test]
    fn derives_action_component_ranges_from_bounded_space() {
        // A width-4 action Box(0, 1); neither component tags a range. Each
        // derives (0, 1) from its slice of the space bounds, so a model output
        // is later mapped into the env's accepted range instead of passed raw.
        let obs = dict_view(vec![]);
        let action = box_view(vec![4], Some(vec![0.0]), Some(vec![1.0]));
        let tags = EnvTags {
            observation: empty_obs(),
            action: action_layout(vec![
                component("action/delta_eef_pos", 3, None),
                component("action/gripper", 1, None),
            ]),
        };
        let features = join(&tags, &obs, &action).expect("join");
        assert_eq!(features.action.components[0].range, Some((0.0, 1.0)));
        assert_eq!(features.action.components[1].range, Some((0.0, 1.0)));
    }

    #[test]
    fn leaves_action_ranges_untouched_for_unbounded_space() {
        // An unbounded action space derives nothing; an explicit tag survives.
        let obs = dict_view(vec![]);
        let action = box_view(vec![1], None, None);
        let mut gripper = component("action/gripper", 1, None);
        gripper.range = Some((-1.0, 1.0));
        let tags = EnvTags {
            observation: empty_obs(),
            action: action_layout(vec![gripper]),
        };
        let features = join(&tags, &obs, &action).expect("join");
        assert_eq!(features.action.components[0].range, Some((-1.0, 1.0)));
    }

    #[test]
    fn rejects_action_range_disagreeing_with_bounded_space() {
        // A gripper tagged (-1, 1) against a Box(0, 1) action is a contradiction
        // (mirrors the state-side range-disagreement check).
        let obs = dict_view(vec![]);
        let action = box_view(vec![1], Some(vec![0.0]), Some(vec![1.0]));
        let mut gripper = component("action/gripper", 1, None);
        gripper.range = Some((-1.0, 1.0));
        let tags = EnvTags {
            observation: empty_obs(),
            action: action_layout(vec![gripper]),
        };
        assert!(matches!(
            join(&tags, &obs, &action),
            Err(JoinError::RangeDisagreement { .. })
        ));
    }
}
