//! Derive the internal [`EnvFeatures`] from sparse [`EnvTags`] layered over a gymnasium
//! space.
//!
//! The same rules run at authoring time (when an env publishes tags) and at resolve time
//! (from the untrusted handshake contract). Every failure names which side disagreed: the
//! tag or the space.

use crate::advisory::Advisory;
use crate::path::NodePath;
use crate::space_view::{SpaceView, SpaceViewKind};
use crate::spec::{
    Action, Actuator, EnvFeature, EnvFeatures, EnvImage, EnvState, EnvTags, EnvText, ImageLayout,
    ObsLeaf, ObsNode, SplitLayout, UnknownFeature,
};

fn quoted(value: &str) -> String {
    format!("{value:?}")
}

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
        "state {key:?} names {labels} labels but the space width is {width}; one label per element"
    )]
    LabelWidthMismatch {
        key: String,
        labels: usize,
        width: u32,
    },
    /// The role names a kind prefix this core does not define.
    #[error("{key:?}: {reason}")]
    InvalidRole { key: String, reason: String },
    /// A whole-leaf tag's labels break the codec rule a `Field` is held to:
    /// at least one, none repeated.
    #[error("{key:?}: {reason}")]
    InvalidLabels { key: String, reason: String },
    /// An observation tag under the `action/` kind: a command is an actuator's
    /// to carry, and a model reads its own previous command with
    /// `source="action"` (`Previous`), never from an env observation.
    #[error(
        "observation {key:?} declares role {role:?}, an action kind; a command is carried by an \
         actuator, and a model reads its previous action with source=\"action\" (Previous), not \
         from an env observation"
    )]
    ActionRoleOnObservation { key: String, role: String },
    #[error(
        "{key:?} tag range {tag:?} disagrees with the space's finite bounds \
         {space:?}"
    )]
    RangeDisagreement {
        key: String,
        tag: (f64, f64),
        space: (f64, f64),
    },
    #[error("the action space must be a Box but it is {actual}")]
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
    #[error(
        "{key:?} declares role {role:?}, which is {expected}-D by convention, but its width is {actual}"
    )]
    RoleDimMismatch {
        key: String,
        role: String,
        expected: u32,
        actual: u32,
    },
}

fn check_role_dim_law(role: &str, actual: u32, key: &str) -> Result<()> {
    if let Some(def) = crate::roles::registry::role_def(role)
        && let crate::roles::registry::DimLaw::Fixed(expected) = def.dim
        && actual != expected
    {
        return Err(JoinError::RoleDimMismatch {
            key: key.to_owned(),
            role: role.to_owned(),
            expected,
            actual,
        });
    }
    Ok(())
}

fn check_role_identity(role: &str, key: &str) -> Result<()> {
    crate::roles::registry::check_role(role).map_err(|reason| JoinError::InvalidRole {
        key: key.to_owned(),
        reason,
    })
}

fn check_observation_role(role: &str, key: &str) -> Result<()> {
    if role.starts_with("action/") {
        return Err(JoinError::ActionRoleOnObservation {
            key: key.to_owned(),
            role: role.to_owned(),
        });
    }
    Ok(())
}

type Result<T> = std::result::Result<T, JoinError>;

/// Join the env observation tree and action layout against their spaces.
pub fn join(
    tags: &EnvTags,
    obs_space: &SpaceView,
    action_space: &SpaceView,
) -> Result<EnvFeatures> {
    let mut observation = Vec::new();
    let mut unknown = Vec::new();
    join_node(
        &tags.observation,
        obs_space,
        &NodePath::root(),
        &mut observation,
        &mut unknown,
    )?;
    let action = resolve_action(&tags.action, action_space)?;
    let mut advisories: Vec<Advisory> = observation
        .iter()
        .filter_map(|feature| match feature {
            EnvFeature::Image(image) => image_layout_advisory(image),
            _ => None,
        })
        .collect();
    let observation_roles = observation.iter().map(|feature| match feature {
        EnvFeature::Image(image) => &image.role,
        EnvFeature::State(state) => &state.role,
        EnvFeature::Text(text) => &text.role,
    });
    let action_roles = action.components.iter().filter_map(|c| c.role.as_ref());
    advisories.extend(
        observation_roles
            .chain(action_roles)
            .filter_map(|role| role_registry_advisory(role)),
    );
    Ok(EnvFeatures {
        observation,
        action,
        unknown,
        advisories,
    })
}

fn role_registry_advisory(role: &str) -> Option<Advisory> {
    if crate::roles::registry::is_sanctioned_role(role) {
        return None;
    }
    let mut message = format!(
        "role {role:?} is not in the role registry; it resolves only when the env and \
         model agree on its exact string. Prefer a blessed role, mark it intentionally \
         non-standard with an `x/` prefix, or -- for dims no model reads -- use a \
         role-less (opaque) actuator"
    );
    if let Some(hint) = crate::roles::registry::kindless_hint(role) {
        message.push_str(". ");
        message.push_str(&hint);
    }
    Some(Advisory::info(message))
}

const PLAUSIBLE_CHANNELS: u32 = 4;

fn image_layout_advisory(image: &EnvImage) -> Option<Advisory> {
    let (declared, opposite_channels, other) = match image.layout {
        ImageLayout::Hwc => ("hwc", image.height, "chw"),
        ImageLayout::Chw => ("chw", image.width, "hwc"),
    };
    (image.channels > PLAUSIBLE_CHANNELS && (1..=PLAUSIBLE_CHANNELS).contains(&opposite_channels))
        .then(|| {
            Advisory::info(format!(
                "image {}: shape implies {} channels under layout={declared}; this looks like \
                 {other} — declare layout=\"{other}\" if so",
                quoted(&image.role),
                image.channels,
            ))
        })
}

fn join_node(
    node: &ObsNode,
    view: &SpaceView,
    source: &NodePath,
    out: &mut Vec<EnvFeature>,
    unknown: &mut Vec<UnknownFeature>,
) -> Result<()> {
    match node {
        ObsNode::Leaf(ObsLeaf::Unknown { kind, role, .. }) => unknown.push(UnknownFeature {
            source: source.clone(),
            role: role.clone(),
            kind: kind.clone(),
        }),
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
                let child_path = source.push_key(key.clone());
                let child_view = view.child(key).ok_or_else(|| JoinError::KeyNotInSpace {
                    key: child_path.to_string(),
                })?;
                join_node(child, child_view, &child_path, out, unknown)?;
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
                join_node(
                    item,
                    &view.children[index],
                    &source.push_index(index),
                    out,
                    unknown,
                )?;
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
        // Unknown leaves are intercepted in `join_node` (recorded, not joined);
        // this arm is unreachable but keeps the match total without a panic.
        ObsLeaf::Unknown { .. } => Ok(Vec::new()),
        ObsLeaf::Image(image) => {
            if leaf.kind != SpaceViewKind::Box || leaf.shape.len() != 3 {
                return Err(JoinError::ClassMismatch {
                    key: path,
                    expected: "an image (3-D Box)",
                    actual: describe_space(leaf),
                });
            }
            check_role_identity(&image.role, &path)?;
            check_observation_role(&image.role, &path)?;
            let (height, width, channels) = image_hwc(&leaf.shape, image.layout);
            Ok(vec![EnvFeature::Image(EnvImage {
                source: source.clone(),
                role: image.role.clone(),
                part: image.part.clone(),
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
            check_role_dim_law(&state.role, width, &path)?;
            check_role_identity(&state.role, &path)?;
            check_observation_role(&state.role, &path)?;
            if let Some(labels) = &state.labels {
                crate::spec::labels::check_labels(labels, &path).map_err(|reason| {
                    JoinError::InvalidLabels {
                        key: path.clone(),
                        reason,
                    }
                })?;
                if labels.len() != width as usize {
                    return Err(JoinError::LabelWidthMismatch {
                        key: path,
                        labels: labels.len(),
                        width,
                    });
                }
            }
            let range = reconcile_range(uniform_finite_range(leaf), state.range, &path)?;
            Ok(vec![EnvFeature::State(EnvState {
                source: source.clone(),
                role: state.role.clone(),
                part: state.part.clone(),
                slice_offset: None,
                dim: Some(width),
                encoding: state.encoding.clone(),
                range,
                frame: state.frame.clone(),
                provenance: state.provenance.clone(),
                labels: state.labels.clone(),
                polarity: state.polarity,
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
            check_role_identity(&text.role, &path)?;
            check_observation_role(&text.role, &path)?;
            Ok(vec![EnvFeature::Text(EnvText {
                source: source.clone(),
                role: text.role.clone(),
            })])
        }
    }
}

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
    // Keyed by `(role, part, provenance)`: one role may repeat across parts or
    // provenances, never within one key.
    let mut seen: Vec<(&str, Option<&str>, Option<&str>)> = Vec::new();
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
            check_role_dim_law(role, field.dim, &path)?;
            check_role_identity(role, &path)?;
            check_observation_role(role, &path)?;
            let key = (
                role.as_str(),
                field.part.as_deref(),
                field.provenance.as_ref().map(|value| value.as_str()),
            );
            if seen.contains(&key) {
                return Err(JoinError::DuplicateLayoutRole {
                    key: path,
                    role: role.clone(),
                });
            }
            seen.push(key);
            let space_range = slice_uniform_finite_range(leaf, offset, field.dim);
            let range = reconcile_range(space_range, field.range, &path)?;
            features.push(EnvFeature::State(EnvState {
                source: source.clone(),
                role: role.clone(),
                part: field.part.clone(),
                slice_offset: Some(offset),
                dim: Some(field.dim),
                encoding: field.encoding.clone(),
                range,
                frame: field.frame.clone(),
                provenance: field.provenance.clone(),
                labels: field.labels.clone(),
                polarity: field.polarity,
            }));
        }
        offset += field.dim;
    }
    Ok(features)
}

fn reconcile_range(
    space: Option<(f64, f64)>,
    tag: Option<(f64, f64)>,
    key: &str,
) -> Result<Option<(f64, f64)>> {
    match (space, tag) {
        (Some(space), Some(tag)) => {
            if ranges_agree(space, tag) {
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

fn slice_uniform_finite_range(leaf: &SpaceView, offset: u32, dim: u32) -> Option<(f64, f64)> {
    let lo = slice_uniform(leaf.low.as_deref()?, offset, dim)?;
    let hi = slice_uniform(leaf.high.as_deref()?, offset, dim)?;
    if !lo.is_finite() || !hi.is_finite() {
        return None;
    }
    Some((lo, hi))
}

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

fn ranges_agree(a: (f64, f64), b: (f64, f64)) -> bool {
    fn close(x: f64, y: f64) -> bool {
        (x - y).abs() <= 1e-6 * x.abs().max(y.abs()).max(1.0)
    }
    close(a.0, b.0) && close(a.1, b.1)
}

fn uniform_finite_range(leaf: &SpaceView) -> Option<(f64, f64)> {
    slice_uniform_finite_range(leaf, 0, width_of(leaf))
}

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
        let Some(role) = &component.role else {
            components.push(component.clone());
            offset += component.dim;
            continue;
        };
        if let Some(encoding) = component.encoding.as_ref().map(|encoding| encoding.base())
            && component.dim != encoding.dims()
        {
            return Err(JoinError::ActionEncodingMismatch {
                role: role.clone(),
                encoding: encoding.as_str(),
                dims: encoding.dims(),
                dim: component.dim,
            });
        }
        check_role_dim_law(role, component.dim, role)?;
        check_role_identity(role, role)?;
        let space_range = slice_uniform_finite_range(action_space, offset, component.dim);
        let range = reconcile_range(space_range, component.range, role)?;
        components.push(Actuator {
            range,
            ..component.clone()
        });
        offset += component.dim;
    }
    Ok(Action {
        components,
        clip: action.clip,
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
    use crate::spec::{AcceptSet, FrameRef, RotationEncoding};
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
        }
    }

    fn component(role: &str, dim: u32, encoding: Option<RotationEncoding>) -> Actuator {
        Actuator {
            role: Some(role.to_owned()),
            dim,
            encoding: encoding.map(crate::spec::ActionEncoding::Native),
            range: None,
            scale: None,
            offset: None,
            axis_scale: None,
            axis_offset: None,
            invert: false,
            threshold: None,
            binary: false,
            clip: false,
            fill: 0.0,
            optional: false,
            unknown: Default::default(),
            frame: None,
            part: None,
            labels: None,
            reference: None,
            polarity: None,
            unit: None,
        }
    }

    fn leaf_at(key: &str, leaf: ObsLeaf) -> ObsNode {
        let mut map = BTreeMap::new();
        map.insert(key.to_owned(), ObsNode::Leaf(leaf));
        ObsNode::Dict(map)
    }

    fn empty_obs() -> ObsNode {
        ObsNode::Dict(BTreeMap::new())
    }

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
                part: None,
                unknown: Default::default(),
            })),
        );
        observation.insert(
            "eef_pos".to_owned(),
            ObsNode::Leaf(ObsLeaf::State(StateTag {
                role: "proprio/eef_pos".to_owned(),
                encoding: None,
                range: None,
                unknown: Default::default(),
                frame: None,
                provenance: None,
                part: None,
                labels: None,
                polarity: None,
            })),
        );
        observation.insert(
            "instruction".to_owned(),
            ObsNode::Leaf(ObsLeaf::Text(TextTag {
                role: "instruction".to_owned(),
                unknown: Default::default(),
            })),
        );
        let tags = EnvTags {
            observation: ObsNode::Dict(observation),
            action: action_layout(vec![component("action/delta_pos", 4, None)]),
        };

        let features = join(&tags, &obs, &action).expect("join");
        assert_eq!(features.observation.len(), 3);
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
    fn action_role_dim_law_rejects_a_fixed_role_with_the_wrong_dim() {
        let action = action_layout(vec![component("action/delta_eef_pos", 4, None)]);
        let err = resolve_action(&action, &box_view(vec![4], None, None)).unwrap_err();
        assert!(
            matches!(
                err,
                JoinError::RoleDimMismatch {
                    expected: 3,
                    actual: 4,
                    ..
                }
            ),
            "{err:?}"
        );

        let adhoc = action_layout(vec![component("action/custom_thing", 4, None)]);
        assert!(resolve_action(&adhoc, &box_view(vec![4], None, None)).is_ok());
    }

    #[test]
    fn join_nudges_ad_hoc_roles_but_not_registered_or_escape() {
        let obs = dict_view(vec![("instruction", text_view())]);
        let action_space = box_view(vec![3], None, None);
        let mut observation = BTreeMap::new();
        observation.insert(
            "instruction".to_owned(),
            ObsNode::Leaf(ObsLeaf::Text(TextTag {
                role: "text/instruction".to_owned(),
                unknown: Default::default(),
            })),
        );
        let tags = EnvTags {
            observation: ObsNode::Dict(observation),
            action: action_layout(vec![
                component("action/gripper", 1, None),
                component("action/wiggle", 1, None),
                component("x/custom", 1, None),
            ]),
        };
        let features = join(&tags, &obs, &action_space).expect("join");
        let nudges: Vec<&Advisory> = features
            .advisories
            .iter()
            .filter(|note| note.message.contains("role registry"))
            .collect();
        assert_eq!(nudges.len(), 1, "{:?}", features.advisories);
        assert!(nudges[0].message.contains("action/wiggle"), "{}", nudges[0]);
        assert_eq!(nudges[0].severity, crate::advisory::AdvisorySeverity::Info);
        assert!(
            !nudges[0].message.contains("names no kind"),
            "{}",
            nudges[0]
        );
    }

    #[test]
    fn join_nudges_a_kindless_role_toward_a_kind_prefix() {
        let obs = dict_view(vec![("instruction", text_view())]);
        let action_space = box_view(vec![1], None, None);
        let mut observation = BTreeMap::new();
        observation.insert(
            "instruction".to_owned(),
            ObsNode::Leaf(ObsLeaf::Text(TextTag {
                role: "text/instruction".to_owned(),
                unknown: Default::default(),
            })),
        );
        let tags = EnvTags {
            observation: ObsNode::Dict(observation),
            action: action_layout(vec![component("wiggle", 1, None)]),
        };
        let features = join(&tags, &obs, &action_space).expect("join");
        let nudge = features
            .advisories
            .iter()
            .find(|note| note.message.contains("names no kind"))
            .unwrap_or_else(|| panic!("{:?}", features.advisories));
        assert!(nudge.message.contains("e.g. x/wiggle"), "{nudge}");
        assert!(!nudge.message.contains("  "), "{nudge}");
        assert_eq!(nudge.severity, crate::advisory::AdvisorySeverity::Info);
    }

    #[test]
    fn joins_a_nested_dict_observation() {
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
                unknown: Default::default(),
                frame: None,
                provenance: None,
                part: None,
                labels: None,
                polarity: None,
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
                    unknown: Default::default(),
                    frame: None,
                    provenance: None,
                    part: None,
                    labels: None,
                    polarity: None,
                })),
                ObsNode::Leaf(ObsLeaf::Text(TextTag {
                    role: "instruction".to_owned(),
                    unknown: Default::default(),
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
                    unknown: Default::default(),
                    frame: None,
                    provenance: None,
                    part: None,
                    labels: None,
                    polarity: None,
                })),
                ObsNode::Leaf(ObsLeaf::State(StateTag {
                    role: "b".to_owned(),
                    encoding: None,
                    range: None,
                    unknown: Default::default(),
                    frame: None,
                    provenance: None,
                    part: None,
                    labels: None,
                    polarity: None,
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
        let obs = box_view(vec![3], None, None);
        let action = box_view(vec![0], None, None);
        let tags = EnvTags {
            observation: leaf_at(
                "eef_pos",
                ObsLeaf::State(StateTag {
                    role: "a".to_owned(),
                    encoding: None,
                    range: None,
                    unknown: Default::default(),
                    frame: None,
                    provenance: None,
                    part: None,
                    labels: None,
                    polarity: None,
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
        let obs = box_view(vec![3], Some(vec![-1.0]), Some(vec![1.0]));
        let action = box_view(vec![0], None, None);
        let tags = EnvTags {
            observation: ObsNode::Leaf(ObsLeaf::State(StateTag {
                role: "proprio/eef_pos".to_owned(),
                encoding: None,
                range: None,
                unknown: Default::default(),
                frame: None,
                provenance: None,
                part: None,
                labels: None,
                polarity: None,
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
                    unknown: Default::default(),
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
                        unknown: Default::default(),
                        frame: None,
                        provenance: None,
                        part: None,
                        labels: None,
                        polarity: None,
                    },
                    Field {
                        role: Some("b".to_owned()),
                        dim: 2,
                        encoding: None,
                        range: None,
                        unknown: Default::default(),
                        frame: None,
                        provenance: None,
                        part: None,
                        labels: None,
                        polarity: None,
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
                part: None,
                unknown: Default::default(),
            }),
        );
        assert!(matches!(
            result,
            Err(JoinError::ClassMismatch { key, .. }) if key == "camera"
        ));
    }

    fn image_leaf(layout: ImageLayout) -> ObsLeaf {
        ObsLeaf::Image(ImageTag {
            role: "image/primary".to_owned(),
            layout,
            upside_down: false,
            part: None,
            unknown: Default::default(),
        })
    }

    fn cam_advisories(shape: Vec<i64>, layout: ImageLayout) -> Vec<String> {
        join_obs("camera", box_view(shape, None, None), image_leaf(layout))
            .expect("joins")
            .advisories
            .into_iter()
            .map(|advisory| advisory.message)
            .collect()
    }

    #[test]
    fn mislabeled_hwc_layout_advises_chw() {
        let advisories = cam_advisories(vec![3, 224, 224], ImageLayout::Hwc);
        assert!(
            advisories
                .iter()
                .any(|a| a.contains("layout=hwc") && a.contains("looks like chw")),
            "expected a chw layout hint, got: {advisories:?}"
        );
    }

    #[test]
    fn mislabeled_chw_layout_advises_hwc() {
        let advisories = cam_advisories(vec![224, 224, 3], ImageLayout::Chw);
        assert!(
            advisories
                .iter()
                .any(|a| a.contains("layout=chw") && a.contains("looks like hwc")),
            "expected an hwc layout hint, got: {advisories:?}"
        );
    }

    #[test]
    fn correctly_labeled_image_has_no_layout_advisory() {
        assert!(cam_advisories(vec![224, 224, 3], ImageLayout::Hwc).is_empty());
    }

    #[test]
    fn legit_many_channel_image_stays_silent() {
        assert!(cam_advisories(vec![224, 224, 6], ImageLayout::Hwc).is_empty());
    }

    #[test]
    fn enforces_rotation_width_law_unconditionally() {
        let result = join_obs(
            "rot",
            box_view(vec![3], None, None),
            ObsLeaf::State(StateTag {
                role: "proprio/eef_rot".to_owned(),
                encoding: Some(AcceptSet::single(RotationEncoding::QuatXyzw)),
                range: None,
                unknown: Default::default(),
                frame: None,
                provenance: None,
                part: None,
                labels: None,
                polarity: None,
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
    fn labels_must_name_every_element_of_the_leaf() {
        let joints = |labels: &[&str]| {
            ObsLeaf::State(StateTag {
                role: "proprio/joint_pos".to_owned(),
                encoding: None,
                range: None,
                unknown: Default::default(),
                frame: None,
                provenance: None,
                part: None,
                labels: Some(labels.iter().map(|label| (*label).to_owned()).collect()),
                polarity: None,
            })
        };
        assert!(matches!(
            join_obs("q", box_view(vec![3], None, None), joints(&["a", "b"])),
            Err(JoinError::LabelWidthMismatch {
                labels: 2,
                width: 3,
                ..
            })
        ));
        join_obs("q", box_view(vec![2], None, None), joints(&["a", "b"])).expect("joins");
        assert!(matches!(
            join_obs("q", box_view(vec![3], None, None), joints(&["a", "a", "b"])),
            Err(JoinError::InvalidLabels { .. })
        ));
        assert!(matches!(
            join_obs("q", box_view(vec![0], None, None), joints(&[])),
            Err(JoinError::InvalidLabels { .. })
        ));
    }

    #[test]
    fn rejects_finite_range_disagreement_but_allows_unbounded_override() {
        let gripper = || {
            ObsLeaf::State(StateTag {
                role: "proprio/gripper".to_owned(),
                encoding: None,
                range: Some((0.0, 2.0)),
                unknown: Default::default(),
                frame: None,
                provenance: None,
                part: None,
                labels: None,
                polarity: None,
            })
        };
        assert!(matches!(
            join_obs(
                "g",
                box_view(vec![1], Some(vec![0.0]), Some(vec![1.0])),
                gripper(),
            ),
            Err(JoinError::RangeDisagreement { .. })
        ));

        let features = join_obs("g", box_view(vec![1], None, None), gripper()).expect("join");
        let EnvFeature::State(state) = &features.observation[0] else {
            panic!("expected state");
        };
        assert_eq!(state.range, Some((0.0, 2.0)));
    }

    #[test]
    fn finite_bounds_agree_with_tag_up_to_float32_rounding() {
        let f32_high = f64::from(0.08_f32);
        assert_ne!(f32_high, 0.08);
        let features = join_obs(
            "g",
            box_view(vec![1], Some(vec![0.0]), Some(vec![f32_high])),
            ObsLeaf::State(StateTag {
                role: "proprio/gripper".to_owned(),
                encoding: None,
                range: Some((0.0, 0.08)),
                unknown: Default::default(),
                frame: None,
                provenance: None,
                part: None,
                labels: None,
                polarity: None,
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
            unknown: Default::default(),
            frame: None,
            provenance: None,
            part: None,
            labels: None,
            polarity: None,
        }
    }

    fn layout_tags(fields: Vec<Field>) -> EnvTags {
        EnvTags {
            observation: ObsNode::Leaf(ObsLeaf::Split(SplitLayout { fields })),
            action: action_layout(vec![]),
        }
    }

    #[test]
    fn splits_flat_layout_into_role_fields_with_offsets() {
        let obs = box_view(vec![8], None, None);
        let action = box_view(vec![0], None, None);
        let tags = layout_tags(vec![
            field(Some("proprio/eef_pos"), 3, None),
            field(Some("proprio/gripper"), 1, None),
            field(None, 1, None),
            field(Some("proprio/obj_pos"), 3, None),
        ]);
        let features = join(&tags, &obs, &action).expect("join");
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
        assert_eq!((states[2].slice_offset, states[2].dim), (Some(5), Some(3)));
    }

    #[test]
    fn a_split_may_publish_one_role_under_two_provenances() {
        let obs = box_view(vec![8], None, None);
        let action = box_view(vec![0], None, None);
        let mut truth = field(
            Some("proprio/base_rot"),
            4,
            Some(RotationEncoding::QuatWxyz),
        );
        truth.provenance = Some(FrameRef::from("privileged"));
        let mut estimate = truth.clone();
        estimate.provenance = Some(FrameRef::from("estimated"));
        let features = join(&layout_tags(vec![truth.clone(), estimate]), &obs, &action)
            .expect("two provenances are two leaves");
        let provenances: Vec<Option<&str>> = features
            .observation
            .iter()
            .filter_map(|feature| match feature {
                EnvFeature::State(state) => Some(state.provenance.as_ref().map(FrameRef::as_str)),
                _ => None,
            })
            .collect();
        assert_eq!(provenances, vec![Some("privileged"), Some("estimated")]);
        let err = join(&layout_tags(vec![truth.clone(), truth]), &obs, &action).unwrap_err();
        assert!(
            matches!(err, JoinError::DuplicateLayoutRole { .. }),
            "{err}"
        );
    }

    #[test]
    fn derives_layout_field_range_from_its_own_bounds_slice() {
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
    fn an_action_role_on_an_observation_leaf_fails_the_join() {
        let obs = box_view(vec![6], None, None);
        let action = box_view(vec![0], None, None);
        let mut last = field(Some("action/joint_pos"), 3, None);
        let fine = field(Some("proprio/eef_pos"), 3, None);
        assert!(matches!(
            join(&layout_tags(vec![last.clone(), fine.clone()]), &obs, &action),
            Err(JoinError::ActionRoleOnObservation { role, .. }) if role == "action/joint_pos"
        ));
        last.role = Some("proprio/joint_pos".to_owned());
        join(&layout_tags(vec![last, fine]), &obs, &action).expect("joins");
    }

    #[test]
    fn a_role_repeats_across_parts_and_an_unknown_kind_fails_the_join() {
        let obs = box_view(vec![6], None, None);
        let action = box_view(vec![0], None, None);
        let mut left = field(Some("proprio/eef_pos"), 3, None);
        left.part = Some("left_arm".to_owned());
        let mut right = field(Some("proprio/eef_pos"), 3, None);
        right.part = Some("right_arm".to_owned());
        let joined = join(&layout_tags(vec![left.clone(), right]), &obs, &action).expect("joins");
        assert_eq!(joined.observation.len(), 2);
        assert!(joined.advisories.is_empty(), "{:?}", joined.advisories);
        assert!(matches!(
            join(&layout_tags(vec![left.clone(), left.clone()]), &obs, &action),
            Err(JoinError::DuplicateLayoutRole { role, .. }) if role == "proprio/eef_pos"
        ));

        let mut odd = field(Some("audio/mic"), 3, None);
        odd.part = Some("franka".to_owned());
        assert!(matches!(
            join(&layout_tags(vec![odd.clone(), left.clone()]), &obs, &action),
            Err(JoinError::InvalidRole { reason, .. }) if reason.contains("kind this core does not define")
        ));
        odd.role = Some("x/mic".to_owned());
        let mut tail = field(Some("proprio/eef_pos"), 3, None);
        tail.part = Some("tail".to_owned());
        let obs = box_view(vec![9], None, None);
        let joined = join(&layout_tags(vec![odd, left, tail]), &obs, &action).expect("joins");
        assert!(joined.advisories.is_empty(), "{:?}", joined.advisories);
    }

    #[test]
    fn rejects_layout_field_encoding_width_mismatch() {
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
