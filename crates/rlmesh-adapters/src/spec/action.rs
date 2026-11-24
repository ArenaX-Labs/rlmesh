use std::collections::BTreeMap;

use super::conventions::Polarity;
use serde::{Deserialize, Serialize};

use super::FrameRef;
use super::custom_encoding::ActionEncoding;

/// One contiguous slice of an action vector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Actuator {
    /// A role-less actuator is *opaque*: it occupies `dim` dims of the action
    /// vector with the constant `fill`, matched by no model role -- the
    /// action-side mirror of a role-less `Field`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(deserialize_with = "crate::spec::num::de_count")]
    pub dim: u32,
    /// A single native encoding (bare string) or a custom-encoding object
    /// (`{base, ...}`) that shadows to its `base`.
    #[serde(default)]
    pub encoding: Option<ActionEncoding>,
    #[serde(default, deserialize_with = "crate::spec::num::de_opt_range")]
    pub range: Option<(f64, f64)>,
    #[serde(default)]
    pub binary: bool,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::spec::num::de_opt_number"
    )]
    pub scale: Option<f64>,
    /// Added after `scale` and before `invert`: `value * scale + offset`, the
    /// stand pose a joint target is expressed around.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::spec::num::de_opt_number"
    )]
    pub offset: Option<f64>,
    /// The per-axis forms of `scale`/`offset`, one value per axis in this
    /// actuator's own label order (its `dim` otherwise).
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::spec::num::de_opt_numbers"
    )]
    pub axis_scale: Option<Vec<f64>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::spec::num::de_opt_numbers"
    )]
    pub axis_offset: Option<Vec<f64>>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub invert: bool,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::spec::num::de_opt_number"
    )]
    pub threshold: Option<f64>,
    /// Env-side per-actuator safety clamp: clamp this component's mapped value to
    /// its declared `range` (the global `Action.clip` cannot, since it applies one
    /// bound to a whole mixed-range vector).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub clip: bool,
    /// Constant emitted for each dim of a role-less (opaque) actuator -- the env
    /// requires these dims but no model produces them.
    #[serde(default, skip_serializing_if = "is_default_fill")]
    pub fill: f64,
    /// On a *roled* actuator: the role is optional. If no model output declares it,
    /// resolution fills the actuator's `dim` dims with `fill` instead of failing; a
    /// model that outputs the role drives it normally. Rejected on a role-less
    /// actuator.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub optional: bool,
    /// The coordinate frame a Cartesian command's axes are expressed in. Required
    /// of an absolute pose (`action/eef_*`) under the require-frames tier; a delta
    /// may declare it too, and when both sides do they must agree.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frame: Option<FrameRef>,
    /// What a *delta* command is integrated against (`action/delta_eef_*`): the
    /// env declares what its controller adds the delta to, a model declares what
    /// it was trained against, and a disagreement is a hard resolve error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<FrameRef>,
    /// The body part this actuator drives, when the role repeats across a
    /// body (`left_arm`, `right_arm`, ...): an identity key, matched never
    /// checked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub part: Option<String>,
    /// The axis names this actuator drives, `dim` of them in this side's own
    /// order. When both sides carry them they must name the same set, and the
    /// model's output is permuted onto the env's axes by name. An opaque actuator
    /// may not carry them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub labels: Option<Vec<String>>,
    /// Which way a gripper moves as this command rises (`action/gripper`):
    /// see `Polarity`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub polarity: Option<Polarity>,
    /// The SI size (metres, or radians for a rotation) of one unit of this
    /// delta command (`action/delta_eef_*`).
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::spec::num::de_opt_number"
    )]
    pub unit: Option<f64>,
    /// Unrecognized additive fields, retained for round-trip and surfaced to the
    /// publish-door `reject_unknowns` guard.
    #[serde(flatten)]
    pub unknown: BTreeMap<String, serde_json::Value>,
}

fn is_default_fill(fill: &f64) -> bool {
    *fill == 0.0
}

/// Ordered action components plus optional clipping bounds.
///
/// Deserialization rejects a duplicate component `(role, part)`: one role may repeat
/// across parts, never within one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "ActionWire")]
pub struct Action {
    pub components: Vec<Actuator>,
    #[serde(default)]
    pub clip: Option<(f64, f64)>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ActionWire {
    components: Vec<Actuator>,
    #[serde(default, deserialize_with = "crate::spec::num::de_opt_range")]
    clip: Option<(f64, f64)>,
}

impl TryFrom<ActionWire> for Action {
    type Error = String;

    fn try_from(wire: ActionWire) -> Result<Self, Self::Error> {
        let mut seen = std::collections::BTreeSet::new();
        for component in &wire.components {
            if !component.fill.is_finite() {
                return Err(format!("actuator {:?} fill must be finite", component.role));
            }
            let locus = format!("actuator {:?}", component.role);
            crate::spec::conventions::check_unit(component.unit, &locus)?;
            if let Some(encoding) = component.encoding.as_ref().map(|encoding| encoding.base())
                && encoding.is_sink()
            {
                return Err(format!(
                    "{locus}: encoding {:?} is observation-only (a direction gravity points, \
                     not a rotation a controller can take); declare a rotation encoding",
                    encoding.as_str()
                ));
            }
            for (name, scalar, axis) in [
                (
                    "scale",
                    component.scale.is_some(),
                    component.axis_scale.as_deref(),
                ),
                (
                    "offset",
                    component.offset.is_some(),
                    component.axis_offset.as_deref(),
                ),
            ] {
                if scalar && axis.is_some() {
                    return Err(format!(
                        "{locus}: sets both {name} and axis_{name}; a quantity is a scalar \
                         or a per-axis vector, not both"
                    ));
                }
                crate::spec::labels::check_axis(axis, &format!("axis_{name}"), &locus)?;
            }
            if let Some(labels) = &component.labels {
                crate::spec::labels::check_labels(labels, &locus)?;
                if labels.len() != component.dim as usize {
                    return Err(format!(
                        "{locus}: declares dim {} but names {} labels; one label per axis",
                        component.dim,
                        labels.len()
                    ));
                }
            }
            match &component.role {
                Some(role) => {
                    if !seen.insert((role.as_str(), component.part.as_deref())) {
                        return Err(match &component.part {
                            Some(part) => format!(
                                "an action layout declares role {role:?} under part {part:?} \
                                 more than once"
                            ),
                            None => {
                                format!("an action layout declares role {role:?} more than once")
                            }
                        });
                    }
                    if component.fill != 0.0 && !component.optional {
                        return Err(format!(
                            "actuator {role:?}: fill applies only to a role-less (opaque) \
                             or optional actuator; a roled, non-optional actuator takes \
                             its values from the model"
                        ));
                    }
                }
                None => {
                    if component.encoding.is_some()
                        || component.range.is_some()
                        || component.scale.is_some()
                        || component.offset.is_some()
                        || component.axis_scale.is_some()
                        || component.axis_offset.is_some()
                        || component.invert
                        || component.threshold.is_some()
                        || component.binary
                        || component.clip
                        || component.optional
                        || component.frame.is_some()
                        || component.reference.is_some()
                        || component.part.is_some()
                        || component.labels.is_some()
                    {
                        return Err("a role-less (opaque) actuator carries only dim and \
                             fill; drop encoding/range/scale/offset/axis_scale/axis_offset/\
                             invert/threshold/binary/clip/optional/frame/reference/part/labels (an opaque actuator is already always filled)"
                            .to_owned());
                    }
                }
            }
        }
        Ok(Action {
            components: wire.components,
            clip: wire.clip,
        })
    }
}

#[cfg(test)]
mod finiteness_contract {
    use super::Action;

    #[test]
    fn rejects_infinity_token() {
        assert!(
            serde_json::from_str::<Action>(r#"{"components": [], "clip": [Infinity, 1.0]}"#)
                .is_err()
        );
    }

    #[test]
    fn rejects_overflow_literal() {
        assert!(
            serde_json::from_str::<Action>(r#"{"components": [], "clip": [1e400, 1.0]}"#).is_err()
        );
        assert!(
            serde_json::from_str::<Action>(
                r#"{"components": [{"role": "g", "dim": 1, "scale": 1e400}]}"#
            )
            .is_err()
        );
    }

    #[test]
    fn accepts_finite_bounds() {
        let ok: Action =
            serde_json::from_str(r#"{"components": [], "clip": [-1.0, 1.0]}"#).unwrap();
        assert_eq!(ok.clip, Some((-1.0, 1.0)));
    }
}

#[cfg(test)]
mod tolerant_field_contract {
    use super::{Action, Actuator};

    #[test]
    fn actuator_captures_unknown_field_for_round_trip() {
        let actuator: Actuator =
            serde_json::from_str(r#"{"role": "x", "dim": 1, "rnge": [0.0, 1.0]}"#).unwrap();
        assert_eq!(
            actuator.unknown.get("rnge"),
            Some(&serde_json::json!([0.0, 1.0]))
        );
        let json = serde_json::to_string(&actuator).unwrap();
        assert!(json.contains("rnge"), "unknown field dropped: {json}");
    }

    #[test]
    fn action_envelope_stays_strict() {
        let err =
            serde_json::from_str::<Action>(r#"{"components": [], "clipp": null}"#).unwrap_err();
        assert!(err.to_string().contains("unknown field"), "got: {err}");
    }
}

#[cfg(test)]
mod opaque_actuator_contract {
    use super::Action;

    #[test]
    fn role_less_actuator_carries_only_dim_and_fill() {
        let ok: Action =
            serde_json::from_str(r#"{"components": [{"dim": 2, "fill": 0.25}]}"#).unwrap();
        assert_eq!(ok.components[0].role, None);
        assert_eq!(ok.components[0].fill, 0.25);

        let err =
            serde_json::from_str::<Action>(r#"{"components": [{"dim": 2, "range": [-1.0, 1.0]}]}"#)
                .unwrap_err();
        assert!(err.to_string().contains("role-less"), "{err}");

        for doc in [
            r#"{"components": [{"dim": 2, "frame": "world"}]}"#,
            r#"{"components": [{"dim": 2, "reference": "current"}]}"#,
        ] {
            let err = serde_json::from_str::<Action>(doc).unwrap_err();
            assert!(err.to_string().contains("role-less"), "{err}");
        }
    }

    #[test]
    fn fill_on_a_roled_actuator_is_rejected() {
        let err = serde_json::from_str::<Action>(
            r#"{"components": [{"role": "g", "dim": 1, "fill": 0.5}]}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("fill applies only"), "{err}");
    }

    #[test]
    fn fill_on_an_optional_roled_actuator_is_accepted() {
        let ok: Action = serde_json::from_str(
            r#"{"components": [{"role": "g", "dim": 1, "optional": true, "fill": 0.5}]}"#,
        )
        .unwrap();
        assert_eq!(ok.components[0].role.as_deref(), Some("g"));
        assert!(ok.components[0].optional);
        assert_eq!(ok.components[0].fill, 0.5);
    }

    #[test]
    fn optional_on_a_role_less_actuator_is_rejected() {
        let err =
            serde_json::from_str::<Action>(r#"{"components": [{"dim": 1, "optional": true}]}"#)
                .unwrap_err();
        assert!(err.to_string().contains("role-less"), "{err}");
    }

    #[test]
    fn part_is_barred_from_an_opaque_actuator_and_keys_the_dup_check() {
        let err =
            serde_json::from_str::<Action>(r#"{"components": [{"dim": 2, "part": "left_arm"}]}"#)
                .unwrap_err();
        assert!(err.to_string().contains("role-less"), "{err}");

        let ok: Action = serde_json::from_str(
            r#"{"components": [{"role": "action/joint_pos", "dim": 7, "part": "left_arm"},
                               {"role": "action/joint_pos", "dim": 7, "part": "right_arm"}]}"#,
        )
        .expect("a role split across parts parses");
        assert_eq!(ok.components[0].part.as_deref(), Some("left_arm"));
        let json = serde_json::to_string(&ok).unwrap();
        assert!(json.contains(r#""part":"left_arm""#), "{json}");
        let err = serde_json::from_str::<Action>(
            r#"{"components": [{"role": "action/joint_pos", "dim": 7, "part": "left_arm"},
                               {"role": "action/joint_pos", "dim": 7, "part": "left_arm"}]}"#,
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains(r#"under part "left_arm" more than once"#),
            "{err}"
        );
        let bare: Action =
            serde_json::from_str(r#"{"components": [{"role": "g", "dim": 1}]}"#).unwrap();
        assert!(!serde_json::to_string(&bare).unwrap().contains("part"));
    }

    #[test]
    fn labels_match_dim_and_are_barred_from_an_opaque_actuator() {
        let ok: Action = serde_json::from_str(
            r#"{"components": [{"role": "action/joint_pos", "dim": 2, "labels": ["FR_hip_joint", "FR_thigh_joint"]}]}"#,
        )
        .unwrap();
        assert_eq!(
            ok.components[0].labels.as_deref(),
            Some(&["FR_hip_joint".to_owned(), "FR_thigh_joint".to_owned()][..])
        );
        let json = serde_json::to_string(&ok).unwrap();
        assert!(
            json.contains(r#""labels":["FR_hip_joint","FR_thigh_joint"]"#),
            "{json}"
        );
        for (doc, expect) in [
            (
                r#"{"components": [{"role": "a", "dim": 3, "labels": ["x", "y"]}]}"#,
                "one label per axis",
            ),
            (
                r#"{"components": [{"role": "a", "dim": 2, "labels": ["x", "x"]}]}"#,
                "is repeated",
            ),
            (
                r#"{"components": [{"dim": 1, "labels": ["x"]}]}"#,
                "role-less",
            ),
        ] {
            let err = serde_json::from_str::<Action>(doc).unwrap_err();
            assert!(err.to_string().contains(expect), "{doc}: {err}");
        }
    }

    #[test]
    fn offset_and_the_per_axis_forms_serialize_only_when_set() {
        let ok: Action = serde_json::from_str(
            r#"{"components": [{"role": "a", "dim": 2, "offset": 0.8, "axis_scale": [0.125, 0.25]}]}"#,
        )
        .unwrap();
        assert_eq!(ok.components[0].offset, Some(0.8));
        assert_eq!(ok.components[0].axis_scale, Some(vec![0.125, 0.25]));
        let json = serde_json::to_string(&ok).unwrap();
        assert!(
            json.contains(r#""offset":0.8,"axis_scale":[0.125,0.25]"#)
                && !json.contains("axis_offset"),
            "{json}"
        );
        for (doc, expect) in [
            (
                r#"{"components": [{"role": "a", "dim": 1, "scale": 2.0, "axis_scale": [1.0]}]}"#,
                "both scale and axis_scale",
            ),
            (
                r#"{"components": [{"role": "a", "dim": 1, "offset": 2.0, "axis_offset": [1.0]}]}"#,
                "both offset and axis_offset",
            ),
            (
                r#"{"components": [{"dim": 1, "offset": 1.0}]}"#,
                "role-less",
            ),
        ] {
            let err = serde_json::from_str::<Action>(doc).unwrap_err();
            assert!(err.to_string().contains(expect), "{doc}: {err}");
        }
    }

    #[test]
    fn role_less_actuators_do_not_collide_on_the_dup_check() {
        let ok: Action =
            serde_json::from_str(r#"{"components": [{"dim": 1}, {"dim": 2}]}"#).unwrap();
        assert_eq!(ok.components.len(), 2);
    }
}

#[cfg(test)]
mod no_action_accept_set_contract {
    use super::Action;
    use crate::spec::env_tags::EnvTags;
    use crate::spec::model::ModelSpec;

    const LIST_ENCODED_COMPONENTS: &str =
        r#"{"components": [{"role": "action/rot", "dim": 6, "encoding": ["rot6d", "quat_wxyz"]}]}"#;

    #[test]
    fn the_gravity_sink_is_refused_on_any_actuator() {
        for doc in [
            r#"{"components": [{"role": "action/rot", "dim": 3, "encoding": "gravity_xyz"}]}"#,
            r#"{"components": [{"role": "action/rot", "dim": 3, "encoding": {"base": "gravity_xyz", "to_base": "m:g"}}]}"#,
        ] {
            let err = serde_json::from_str::<Action>(doc).unwrap_err();
            assert!(err.to_string().contains("observation-only"), "{doc}: {err}");
        }
    }

    #[test]
    fn action_encoding_list_fails_to_parse() {
        assert!(serde_json::from_str::<Action>(LIST_ENCODED_COMPONENTS).is_err());

        let ok: Action = serde_json::from_str(
            r#"{"components": [{"role": "action/rot", "dim": 6, "encoding": "rot6d"}]}"#,
        )
        .expect("a single encoding value parses");
        assert_eq!(
            ok.components[0].encoding,
            Some(crate::spec::ActionEncoding::Native(
                crate::spec::rotations::RotationEncoding::Rot6d
            ))
        );

        let custom: Action = serde_json::from_str(
            r#"{"components": [{"role": "action/rot", "dim": 3, "encoding": {"base": "euler_xyz", "to_base": "m:g"}}]}"#,
        )
        .expect("a custom-encoding object parses");
        assert!(
            custom.components[0]
                .encoding
                .as_ref()
                .unwrap()
                .custom()
                .is_some()
        );
    }

    const SINGLE_ENCODED_COMPONENTS: &str =
        r#"{"components": [{"role": "action/rot", "dim": 6, "encoding": "rot6d"}]}"#;

    #[test]
    fn env_side_action_encoding_list_fails_to_parse() {
        let ok = format!(r#"{{"observation": {{}}, "action": {SINGLE_ENCODED_COMPONENTS}}}"#);
        serde_json::from_str::<EnvTags>(&ok).expect("the single-encoding control doc parses");

        let doc = format!(r#"{{"observation": {{}}, "action": {LIST_ENCODED_COMPONENTS}}}"#);
        assert!(serde_json::from_str::<EnvTags>(&doc).is_err());
    }

    #[test]
    fn model_side_action_encoding_list_fails_to_parse() {
        let ok = format!(r#"{{"input": {{}}, "output": {SINGLE_ENCODED_COMPONENTS}}}"#);
        serde_json::from_str::<ModelSpec>(&ok).expect("the single-encoding control doc parses");

        let doc = format!(r#"{{"input": {{}}, "output": {LIST_ENCODED_COMPONENTS}}}"#);
        assert!(serde_json::from_str::<ModelSpec>(&doc).is_err());
    }
}

#[cfg(test)]
mod dup_role_contract {
    use super::Action;

    #[test]
    fn rejects_duplicate_component_role() {
        let err = serde_json::from_str::<Action>(
            r#"{"components": [{"role": "g", "dim": 1}, {"role": "g", "dim": 1}]}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("more than once"), "got: {err}");

        let ok: Action = serde_json::from_str(r#"{"components": []}"#).expect("empty parses");
        assert!(ok.components.is_empty());
        let ok: Action = serde_json::from_str(
            r#"{"components": [{"role": "a", "dim": 1}, {"role": "b", "dim": 1}]}"#,
        )
        .expect("distinct roles parse");
        assert_eq!(ok.components.len(), 2);
    }
}
