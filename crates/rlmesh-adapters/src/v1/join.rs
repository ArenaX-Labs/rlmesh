//! Derive the internal [`EnvFeatures`] from sparse [`EnvAnnotations`] layered
//! over a gymnasium space.
//!
//! `join` is the single place env semantics meet env structure. It runs at two
//! seams with identical rules: at authoring time (when an env publishes
//! annotations, so mistakes fail fast) and worker-side at resolve time (from
//! the untrusted handshake contract). Every failure names which side
//! disagreed — the annotation or the space.

use super::space_view::{SpaceView, SpaceViewKind};
use super::spec::{
    ActionLayout, EnvAnnotations, EnvFeature, EnvFeatures, EnvImage, EnvState, EnvText,
    ImageLayout, ObsAnnotation,
};

/// A validation failure while joining annotations against a space.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum JoinError {
    #[error("observation key {key:?} does not resolve to a leaf of the observation space")]
    KeyNotInSpace { key: String },
    #[error("observation {key:?} is annotated as {expected} but the space is {actual}")]
    ClassMismatch {
        key: String,
        expected: &'static str,
        actual: String,
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
    #[error(
        "state {key:?} annotation range {annotation:?} disagrees with the space's finite bounds \
         {space:?}"
    )]
    RangeDisagreement {
        key: String,
        annotation: (f64, f64),
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

/// Join sparse annotations against the observation and action spaces.
pub fn join(
    annotations: &EnvAnnotations,
    obs_space: &SpaceView,
    action_space: &SpaceView,
) -> Result<EnvFeatures> {
    let mut observation = Vec::with_capacity(annotations.observation.len());
    for (path, annotation) in &annotations.observation {
        let leaf = resolve_leaf(obs_space, path)
            .ok_or_else(|| JoinError::KeyNotInSpace { key: path.clone() })?;
        observation.push(join_feature(path, annotation, leaf)?);
    }
    validate_action(&annotations.action, action_space)?;
    Ok(EnvFeatures {
        observation,
        action: annotations.action.clone(),
    })
}

/// Resolve a dotted observation key-path to a leaf of the space. The reserved
/// `"."` denotes the flat/root observation.
fn resolve_leaf<'view>(space: &'view SpaceView, path: &str) -> Option<&'view SpaceView> {
    if path == "." {
        return Some(space);
    }
    let mut node = space;
    for segment in path.split('.') {
        node = node.child(segment)?;
    }
    Some(node)
}

fn join_feature(path: &str, annotation: &ObsAnnotation, leaf: &SpaceView) -> Result<EnvFeature> {
    match annotation {
        ObsAnnotation::Image(image) => {
            if leaf.kind != SpaceViewKind::Box || leaf.shape.len() != 3 {
                return Err(JoinError::ClassMismatch {
                    key: path.to_owned(),
                    expected: "an image (3-D Box)",
                    actual: describe_space(leaf),
                });
            }
            let (height, width) = image_hw(&leaf.shape, image.layout);
            Ok(EnvFeature::Image(EnvImage {
                key: path.to_owned(),
                role: image.role.clone(),
                layout: image.layout,
                upside_down: image.upside_down,
                height,
                width,
            }))
        }
        ObsAnnotation::State(state) => {
            if !is_numeric(leaf) {
                return Err(JoinError::ClassMismatch {
                    key: path.to_owned(),
                    expected: "a numeric state",
                    actual: describe_space(leaf),
                });
            }
            let width = width_of(leaf);
            if let Some(encoding) = state.encoding
                && width != encoding.dims()
            {
                return Err(JoinError::EncodingWidthMismatch {
                    key: path.to_owned(),
                    encoding: encoding.as_str(),
                    dims: encoding.dims(),
                    width,
                });
            }
            let range = derive_state_range(leaf, state.range, path)?;
            Ok(EnvFeature::State(EnvState {
                key: path.to_owned(),
                role: state.role.clone(),
                dim: Some(width),
                encoding: state.encoding,
                range,
            }))
        }
        ObsAnnotation::Text(text) => {
            if leaf.kind != SpaceViewKind::Text {
                return Err(JoinError::ClassMismatch {
                    key: path.to_owned(),
                    expected: "a text space",
                    actual: describe_space(leaf),
                });
            }
            Ok(EnvFeature::Text(EnvText {
                key: path.to_owned(),
                role: text.role.clone(),
            }))
        }
    }
}

/// Derive a state's value range: from the space's finite bounds when uniform,
/// overridden by an explicit annotation only where the space is unbounded. A
/// finite space range that disagrees with an explicit annotation is an error.
fn derive_state_range(
    leaf: &SpaceView,
    annotation: Option<(f64, f64)>,
    key: &str,
) -> Result<Option<(f64, f64)>> {
    match (uniform_finite_range(leaf), annotation) {
        (Some(space), Some(annotation)) if space != annotation => {
            Err(JoinError::RangeDisagreement {
                key: key.to_owned(),
                annotation,
                space,
            })
        }
        (Some(space), _) => Ok(Some(space)),
        (None, annotation) => Ok(annotation),
    }
}

/// The single finite `(low, high)` pair a space declares, or `None` if it is
/// unbounded, non-finite, or per-element non-uniform.
fn uniform_finite_range(leaf: &SpaceView) -> Option<(f64, f64)> {
    let low = leaf.low.as_ref()?;
    let high = leaf.high.as_ref()?;
    let lo = *low.first()?;
    let hi = *high.first()?;
    if !lo.is_finite() || !hi.is_finite() {
        return None;
    }
    if low.iter().any(|&value| value != lo) || high.iter().any(|&value| value != hi) {
        return None;
    }
    Some((lo, hi))
}

fn validate_action(action: &ActionLayout, action_space: &SpaceView) -> Result<()> {
    if action_space.kind != SpaceViewKind::Box {
        return Err(JoinError::ActionClass {
            actual: describe_space(action_space),
        });
    }
    let action_width = width_of(action_space);
    let component_sum: u32 = action
        .components
        .iter()
        .map(|component| component.dim)
        .sum();
    if component_sum != action_width {
        return Err(JoinError::ActionWidthMismatch {
            action_width,
            component_sum,
        });
    }
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
    }
    Ok(())
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

/// Pixel `(height, width)` of a 3-D image shape under its layout.
fn image_hw(shape: &[i64], layout: ImageLayout) -> (u32, u32) {
    let dim = |index: usize| {
        shape
            .get(index)
            .copied()
            .and_then(|value| u32::try_from(value).ok())
            .unwrap_or(0)
    };
    match layout {
        ImageLayout::Hwc => (dim(0), dim(1)),
        ImageLayout::Chw => (dim(1), dim(2)),
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

    use super::super::spec::RotationEncoding;
    use super::super::spec::{ActionComponent, ImageAnnotation, StateAnnotation, TextAnnotation};
    use super::*;

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

    fn action_layout(components: Vec<ActionComponent>) -> ActionLayout {
        ActionLayout {
            components,
            clip: None,
        }
    }

    fn component(role: &str, dim: u32, encoding: Option<RotationEncoding>) -> ActionComponent {
        ActionComponent {
            role: role.to_owned(),
            dim,
            encoding,
            range: None,
            binary: false,
        }
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
            ObsAnnotation::Image(ImageAnnotation {
                role: "image/primary".to_owned(),
                layout: Default::default(),
                upside_down: false,
            }),
        );
        observation.insert(
            "eef_pos".to_owned(),
            ObsAnnotation::State(StateAnnotation {
                role: "proprio/eef_pos".to_owned(),
                encoding: None,
                range: None,
            }),
        );
        observation.insert(
            "instruction".to_owned(),
            ObsAnnotation::Text(TextAnnotation {
                role: "instruction".to_owned(),
            }),
        );
        let annotations = EnvAnnotations {
            observation,
            action: action_layout(vec![component("action/delta_pos", 4, None)]),
        };

        let features = join(&annotations, &obs, &action).expect("join");
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
    }

    #[test]
    fn rejects_key_not_in_space() {
        let obs = dict_view(vec![("camera", box_view(vec![8, 8, 3], None, None))]);
        let mut observation = BTreeMap::new();
        observation.insert(
            "missing".to_owned(),
            ObsAnnotation::Text(TextAnnotation {
                role: "instruction".to_owned(),
            }),
        );
        let annotations = EnvAnnotations {
            observation,
            action: action_layout(vec![]),
        };
        let action = box_view(vec![0], None, None);
        assert!(matches!(
            join(&annotations, &obs, &action),
            Err(JoinError::KeyNotInSpace { key }) if key == "missing"
        ));
    }

    #[test]
    fn rejects_image_on_non_3d_box() {
        let obs = dict_view(vec![("camera", box_view(vec![64, 64], None, None))]);
        let mut observation = BTreeMap::new();
        observation.insert(
            "camera".to_owned(),
            ObsAnnotation::Image(ImageAnnotation {
                role: "image/primary".to_owned(),
                layout: Default::default(),
                upside_down: false,
            }),
        );
        let annotations = EnvAnnotations {
            observation,
            action: action_layout(vec![]),
        };
        let action = box_view(vec![0], None, None);
        assert!(matches!(
            join(&annotations, &obs, &action),
            Err(JoinError::ClassMismatch { key, .. }) if key == "camera"
        ));
    }

    #[test]
    fn enforces_rotation_width_law_unconditionally() {
        // A quaternion encoding (4 dims) on a width-3 space.
        let obs = dict_view(vec![("rot", box_view(vec![3], None, None))]);
        let mut observation = BTreeMap::new();
        observation.insert(
            "rot".to_owned(),
            ObsAnnotation::State(StateAnnotation {
                role: "proprio/eef_rot".to_owned(),
                encoding: Some(RotationEncoding::QuatXyzw),
                range: None,
            }),
        );
        let annotations = EnvAnnotations {
            observation,
            action: action_layout(vec![]),
        };
        let action = box_view(vec![0], None, None);
        assert!(matches!(
            join(&annotations, &obs, &action),
            Err(JoinError::EncodingWidthMismatch {
                width: 3,
                dims: 4,
                ..
            })
        ));
    }

    #[test]
    fn rejects_finite_range_disagreement_but_allows_unbounded_override() {
        // Finite space bounds [0, 1] disagree with annotation [0, 2] -> error.
        let obs = dict_view(vec![(
            "g",
            box_view(vec![1], Some(vec![0.0]), Some(vec![1.0])),
        )]);
        let mut observation = BTreeMap::new();
        observation.insert(
            "g".to_owned(),
            ObsAnnotation::State(StateAnnotation {
                role: "proprio/gripper".to_owned(),
                encoding: None,
                range: Some((0.0, 2.0)),
            }),
        );
        let annotations = EnvAnnotations {
            observation,
            action: action_layout(vec![]),
        };
        let action = box_view(vec![0], None, None);
        assert!(matches!(
            join(&annotations, &obs, &action),
            Err(JoinError::RangeDisagreement { .. })
        ));

        // An unbounded space lets the annotation supply the range.
        let obs = dict_view(vec![("g", box_view(vec![1], None, None))]);
        let mut observation = BTreeMap::new();
        observation.insert(
            "g".to_owned(),
            ObsAnnotation::State(StateAnnotation {
                role: "proprio/gripper".to_owned(),
                encoding: None,
                range: Some((0.0, 2.0)),
            }),
        );
        let annotations = EnvAnnotations {
            observation,
            action: action_layout(vec![]),
        };
        let features = join(&annotations, &obs, &action).expect("join");
        let EnvFeature::State(state) = &features.observation[0] else {
            panic!("expected state");
        };
        assert_eq!(state.range, Some((0.0, 2.0)));
    }

    #[test]
    fn rejects_action_width_mismatch() {
        let obs = dict_view(vec![]);
        let action = box_view(vec![7], None, None);
        let annotations = EnvAnnotations {
            observation: BTreeMap::new(),
            action: action_layout(vec![component("action/delta_pos", 3, None)]),
        };
        assert!(matches!(
            join(&annotations, &obs, &action),
            Err(JoinError::ActionWidthMismatch {
                action_width: 7,
                component_sum: 3,
            })
        ));
    }
}
