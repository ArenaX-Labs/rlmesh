use std::collections::BTreeMap;
use std::fmt;

use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::spec::{AcceptSet, FrameRef, Provenance, RotationLiteral, StateEncoding};

fn default_float32() -> String {
    "float32".to_owned()
}

fn is_default_fill(fill: &f64) -> bool {
    *fill == 0.0
}

/// Where a [`ConcatPart`] reads its value from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PartSource {
    /// An env state feature matched by role (the default, omitted on the wire).
    #[default]
    Observation,
    /// The model's own output actuator of the same `(role, part)`: the raw
    /// action it emitted at the previous step, in model order, before any
    /// affine; `fill` before the episode's first action.
    Action,
}

fn is_observation(source: &PartSource) -> bool {
    *source == PartSource::Observation
}

/// One part of a [`State`] concat, sourced from an env state feature.
///
/// Deserializes from a bare string (a role-only part) or an object with the full field set; a
/// role-only part serializes back to a bare string. A part with no `role` is a constant that
/// contributes `dim` copies of `fill`. A part with `source: "action"` reads the model's own
/// previous output for the role and carries only `part`, `labels`, `dim` and `fill`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ConcatPart {
    /// The env state feature this part reads, or `None` for a constant part.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Where the value comes from: the env (default) or the model's own
    /// previous action for the role.
    #[serde(default, skip_serializing_if = "is_observation")]
    pub source: PartSource,
    /// Rotation encoding(s) the model accepts for this part, in preference order. The env's
    /// native encoding is used when listed, else it is converted into the first entry. A
    /// custom-encoding object (`{base, ...}`) resolves as its `base`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoding: Option<StateEncoding>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dim: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
    /// Target value range. Values are mapped affinely from the env feature's source range into
    /// this one; with no source range on the env side this is a no-op, not a clamp.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range: Option<(f64, f64)>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub optional: bool,
    /// The value this part contributes when it has no env source: the constant
    /// of a role-less part, or the fill of an `optional` role the env lacks.
    #[serde(default, skip_serializing_if = "is_default_fill")]
    pub fill: f64,
    /// A fixed rotation right-multiplied onto the env's rotation
    /// (`R_out = R_in @ R(post_rotate)`) before it is re-encoded -- the rigid
    /// re-frame a checkpoint was trained against.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub post_rotate: Option<RotationLiteral>,
    /// Model-side affine, applied after the range map: `value * scale + offset`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset: Option<f64>,
    /// The per-axis forms of `scale`/`offset`, one value per axis in this
    /// part's own label order (the resolved width otherwise).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub axis_scale: Option<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub axis_offset: Option<Vec<f64>>,
    /// The coordinate frame the checkpoint was trained to read this part in,
    /// when the role is an absolute pose.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frame: Option<FrameRef>,
    /// Where the checkpoint expects this part's numbers to come from: one
    /// provenance, or the set it accepts (a bare string on the wire when
    /// one). Against an env that publishes the role under several, the set
    /// picks the leaf; a value the env contradicts is a resolve error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<AcceptSet<Provenance>>,
    /// The body part this part reads, when the role repeats across a body
    /// (`left_arm`, ...): an identity key the resolver matches on. `None`
    /// binds the env's only leaf of the role; a constant part may not carry one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub part: Option<String>,
    /// The axis names this part reads, in the order the checkpoint was
    /// trained on. Fixes `dim`; must name the same set as the env leaf's
    /// labels, and a differing order is a permutation. A constant part may
    /// not carry them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub labels: Option<Vec<String>>,
    /// The gripper polarity the checkpoint reads (`proprio/gripper_pos`):
    /// openness (`open`) or closedness (`close`); see
    /// `Polarity`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub polarity: Option<crate::spec::Polarity>,
    /// Unrecognized additive fields, retained for round-trip and surfaced to the
    /// publish-door `reject_unknowns` guard.
    #[serde(flatten)]
    pub unknown: BTreeMap<String, serde_json::Value>,
}

#[derive(Deserialize)]
struct ConcatPartWire {
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    source: PartSource,
    #[serde(default)]
    encoding: Option<StateEncoding>,
    #[serde(default, deserialize_with = "crate::spec::num::de_opt_count")]
    dim: Option<u32>,
    #[serde(default, deserialize_with = "crate::spec::num::de_opt_count")]
    index: Option<u32>,
    #[serde(default, deserialize_with = "crate::spec::num::de_opt_range")]
    range: Option<(f64, f64)>,
    #[serde(default)]
    optional: bool,
    #[serde(default)]
    fill: f64,
    #[serde(default)]
    post_rotate: Option<RotationLiteral>,
    #[serde(default, deserialize_with = "crate::spec::num::de_opt_number")]
    scale: Option<f64>,
    #[serde(default, deserialize_with = "crate::spec::num::de_opt_number")]
    offset: Option<f64>,
    #[serde(default, deserialize_with = "crate::spec::num::de_opt_numbers")]
    axis_scale: Option<Vec<f64>>,
    #[serde(default, deserialize_with = "crate::spec::num::de_opt_numbers")]
    axis_offset: Option<Vec<f64>>,
    #[serde(default)]
    frame: Option<FrameRef>,
    #[serde(default)]
    provenance: Option<AcceptSet<Provenance>>,
    #[serde(default)]
    part: Option<String>,
    #[serde(default)]
    labels: Option<Vec<String>>,
    #[serde(default)]
    polarity: Option<crate::spec::Polarity>,
    #[serde(flatten)]
    unknown: BTreeMap<String, serde_json::Value>,
}

impl TryFrom<ConcatPartWire> for ConcatPart {
    type Error = String;

    fn try_from(wire: ConcatPartWire) -> Result<Self, Self::Error> {
        if wire.dim.is_some() && wire.index.is_some() {
            return Err(format!(
                "state part {:?} sets both dim and index; index selects one element \
                 and dim truncates to the leading N -- set one, not both",
                wire.role
            ));
        }
        if !wire.fill.is_finite() {
            return Err(format!("state part {:?} fill must be finite", wire.role));
        }
        let locus = format!("state part {:?}", wire.role);
        if wire.source == PartSource::Action {
            if wire.role.is_none() {
                return Err(
                    "a state part with source \"action\" needs the role of the actuator it \
                     reads"
                        .to_owned(),
                );
            }
            if wire.encoding.is_some()
                || wire.index.is_some()
                || wire.range.is_some()
                || wire.optional
                || wire.post_rotate.is_some()
                || wire.scale.is_some()
                || wire.offset.is_some()
                || wire.axis_scale.is_some()
                || wire.axis_offset.is_some()
                || wire.frame.is_some()
                || wire.provenance.is_some()
            {
                return Err(format!(
                    "{locus}: source \"action\" reads the model's own raw output and carries \
                     only part/labels/dim/fill; drop encoding/index/range/optional/\
                     post_rotate/scale/offset/axis_scale/axis_offset/frame/provenance"
                ));
            }
        }
        for (name, scalar, axis) in [
            ("scale", wire.scale.is_some(), wire.axis_scale.as_deref()),
            ("offset", wire.offset.is_some(), wire.axis_offset.as_deref()),
        ] {
            if scalar && axis.is_some() {
                return Err(format!(
                    "{locus}: sets both {name} and axis_{name}; a quantity is a scalar or a \
                     per-axis vector, not both"
                ));
            }
            crate::spec::labels::check_axis(axis, &format!("axis_{name}"), &locus)?;
        }
        if let Some(labels) = &wire.labels {
            crate::spec::labels::check_labels(labels, &locus)?;
            if wire.index.is_some() {
                return Err(format!(
                    "{locus}: sets both labels and index; labels name every axis of the \
                     leaf, so drop one of them"
                ));
            }
            if let Some(dim) = wire.dim
                && dim as usize != labels.len()
            {
                return Err(format!(
                    "{locus}: declares dim {dim} but names {} labels; labels fix the width, \
                     so drop dim or make them agree",
                    labels.len()
                ));
            }
        }
        match &wire.role {
            Some(role) => {
                if wire.fill != 0.0 && !wire.optional && wire.source == PartSource::Observation {
                    return Err(format!(
                        "state part {role:?}: fill applies only to a constant (role-less) \
                         or optional part; a roled, non-optional part takes its values \
                         from the env"
                    ));
                }
                if wire.post_rotate.is_some() {
                    match &wire.encoding {
                        None => {
                            return Err(format!(
                                "state part {role:?}: post_rotate needs a rotation encoding"
                            ));
                        }
                        Some(encoding) if encoding.custom().is_some() => {
                            return Err(format!(
                                "state part {role:?}: post_rotate cannot combine with a \
                                 custom encoding; fold the rotation into the repack"
                            ));
                        }
                        Some(_) => {}
                    }
                }
            }
            None => {
                if wire.dim.is_none() {
                    return Err(
                        "a constant (role-less) state part needs dim to size itself".to_owned()
                    );
                }
                if wire.encoding.is_some()
                    || wire.index.is_some()
                    || wire.range.is_some()
                    || wire.optional
                    || wire.post_rotate.is_some()
                    || wire.scale.is_some()
                    || wire.offset.is_some()
                    || wire.axis_scale.is_some()
                    || wire.axis_offset.is_some()
                    || wire.frame.is_some()
                    || wire.provenance.is_some()
                    || wire.part.is_some()
                    || wire.labels.is_some()
                    || wire.polarity.is_some()
                {
                    return Err("a constant (role-less) state part carries only dim and \
                         fill; drop encoding/index/range/optional/post_rotate/scale/offset/\
                         axis_scale/axis_offset/frame/provenance/part/labels/polarity"
                        .to_owned());
                }
            }
        }
        Ok(ConcatPart {
            role: wire.role,
            source: wire.source,
            encoding: wire.encoding,
            dim: wire.dim,
            index: wire.index,
            range: wire.range,
            optional: wire.optional,
            fill: wire.fill,
            post_rotate: wire.post_rotate,
            scale: wire.scale,
            offset: wire.offset,
            axis_scale: wire.axis_scale,
            axis_offset: wire.axis_offset,
            frame: wire.frame,
            provenance: wire.provenance,
            part: wire.part,
            labels: wire.labels,
            polarity: wire.polarity,
            unknown: wire.unknown,
        })
    }
}

impl<'de> Deserialize<'de> for ConcatPart {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct PartVisitor;

        impl<'de> Visitor<'de> for PartVisitor {
            type Value = ConcatPart;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a role name (string) or a state-part object")
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<ConcatPart, E> {
                Ok(ConcatPart {
                    role: Some(value.to_owned()),
                    source: PartSource::Observation,
                    encoding: None,
                    dim: None,
                    index: None,
                    range: None,
                    optional: false,
                    fill: 0.0,
                    post_rotate: None,
                    scale: None,
                    offset: None,
                    axis_scale: None,
                    axis_offset: None,
                    frame: None,
                    provenance: None,
                    part: None,
                    labels: None,
                    unknown: BTreeMap::new(),
                    polarity: None,
                })
            }

            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<ConcatPart, A::Error> {
                let wire = ConcatPartWire::deserialize(de::value::MapAccessDeserializer::new(map))?;
                ConcatPart::try_from(wire).map_err(de::Error::custom)
            }
        }

        deserializer.deserialize_any(PartVisitor)
    }
}

fn serialize_concat_part<S: Serializer>(
    part: &ConcatPart,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let role_only = part.role.is_some()
        && is_observation(&part.source)
        && part.encoding.is_none()
        && part.dim.is_none()
        && part.index.is_none()
        && part.range.is_none()
        && !part.optional
        && is_default_fill(&part.fill)
        && part.post_rotate.is_none()
        && part.scale.is_none()
        && part.offset.is_none()
        && part.axis_scale.is_none()
        && part.axis_offset.is_none()
        && part.frame.is_none()
        && part.provenance.is_none()
        && part.part.is_none()
        && part.labels.is_none()
        && part.unknown.is_empty();
    if let (true, Some(role)) = (role_only, &part.role) {
        serializer.serialize_str(role)
    } else {
        part.serialize(serializer)
    }
}

fn serialize_parts<S: Serializer>(parts: &[ConcatPart], serializer: S) -> Result<S::Ok, S::Error> {
    use serde::ser::SerializeSeq;
    let mut seq = serializer.serialize_seq(Some(parts.len()))?;
    for part in parts {
        seq.serialize_element(&PartWrapper(part))?;
    }
    seq.end()
}

struct PartWrapper<'a>(&'a ConcatPart);

impl Serialize for PartWrapper<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serialize_concat_part(self.0, serializer)
    }
}

/// Container kind for a resolved state value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StateContainer {
    #[default]
    Array,
    List,
}

/// A numeric state input expected by a model.
///
/// `components` must be non-empty; a single role is one role-only part. Placement is the tree
/// position the leaf sits at.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "StateWire")]
pub struct State {
    #[serde(serialize_with = "serialize_parts")]
    pub components: Vec<ConcatPart>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pad_to: Option<u32>,
    #[serde(default = "default_float32")]
    pub dtype: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reshape: Option<Vec<i64>>,
    #[serde(default, skip_serializing_if = "is_default_container")]
    pub container: StateContainer,
    /// Clamp the assembled vector to `(low, high)`, after every part's own
    /// transforms and before `pad_to` (legged_gym's `clip_obs`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clip: Option<(f64, f64)>,
    /// Unrecognized additive fields, retained for round-trip and surfaced to the
    /// publish-door `reject_unknowns` guard.
    #[serde(flatten)]
    pub unknown: BTreeMap<String, serde_json::Value>,
}

fn is_default_container(container: &StateContainer) -> bool {
    *container == StateContainer::Array
}

#[derive(Deserialize)]
struct StateWire {
    components: Vec<ConcatPart>,
    #[serde(default, deserialize_with = "crate::spec::num::de_opt_count")]
    pad_to: Option<u32>,
    #[serde(default = "default_float32")]
    dtype: String,
    #[serde(default, deserialize_with = "crate::spec::num::de_opt_dims")]
    reshape: Option<Vec<i64>>,
    #[serde(default)]
    container: StateContainer,
    #[serde(default, deserialize_with = "crate::spec::num::de_opt_range")]
    clip: Option<(f64, f64)>,
    #[serde(flatten)]
    unknown: BTreeMap<String, serde_json::Value>,
}

impl TryFrom<StateWire> for State {
    type Error = String;

    fn try_from(wire: StateWire) -> Result<Self, Self::Error> {
        if wire.components.is_empty() {
            return Err("a state input needs at least one component".to_owned());
        }
        if wire.components.iter().all(|part| part.role.is_none()) {
            return Err(
                "a state input of only constant parts reads nothing from the env; give it \
                 at least one roled part"
                    .to_owned(),
            );
        }
        Ok(State {
            components: wire.components,
            pad_to: wire.pad_to,
            dtype: wire.dtype,
            reshape: wire.reshape,
            container: wire.container,
            clip: wire.clip,
            unknown: wire.unknown,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{ConcatPart, PartSource, State};
    use crate::spec::Provenance;

    #[test]
    fn rejects_empty_components() {
        let err = serde_json::from_str::<State>(r#"{"components": []}"#).unwrap_err();
        assert!(
            err.to_string().contains("at least one component"),
            "got: {err}"
        );
        let ok: State =
            serde_json::from_str(r#"{"components": [{"role": "r"}]}"#).expect("non-empty parses");
        assert_eq!(ok.components.len(), 1);
    }

    #[test]
    fn part_parses_from_bare_role_string_or_object() {
        let state: State = serde_json::from_str(
            r#"{"components": ["proprio/eef_pos", {"role": "proprio/gripper", "dim": 1}]}"#,
        )
        .expect("parse");
        assert_eq!(state.components.len(), 2);
        assert_eq!(state.components[0].role.as_deref(), Some("proprio/eef_pos"));
        assert_eq!(state.components[0].dim, None);
        assert_eq!(state.components[1].dim, Some(1));
    }

    #[test]
    fn role_only_part_round_trips_to_a_bare_string() {
        let state: State = serde_json::from_str(r#"{"components": ["proprio/eef_pos"]}"#).unwrap();
        let json = serde_json::to_string(&state).unwrap();
        assert!(
            json.contains(r#""components":["proprio/eef_pos"]"#),
            "got: {json}"
        );
    }

    #[test]
    fn rejects_component_with_both_dim_and_index() {
        let err = serde_json::from_str::<State>(
            r#"{"components": [{"role": "r", "dim": 3, "index": 0}]}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("both dim and index"), "got: {err}");
    }

    #[test]
    fn rejects_invalid_reshape_dims() {
        let err =
            serde_json::from_str::<State>(r#"{"components": [{"role": "r"}], "reshape": [-5]}"#)
                .unwrap_err();
        assert!(err.to_string().contains("infer"), "got: {err}");
        let err = serde_json::from_str::<State>(
            r#"{"components": [{"role": "r"}], "reshape": [-1, -1]}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("at most one -1"), "got: {err}");
        let ok: State =
            serde_json::from_str(r#"{"components": [{"role": "r"}], "reshape": [-1, 4]}"#)
                .expect("one infer parses");
        assert_eq!(ok.reshape, Some(vec![-1, 4]));
    }

    #[test]
    fn constant_part_round_trips_to_a_dim_and_fill_object() {
        let state: State = serde_json::from_str(
            r#"{"components": ["proprio/eef_pos", {"dim": 1}, {"dim": 2, "fill": 0.5}]}"#,
        )
        .expect("parse");
        assert_eq!(state.components[1].role, None);
        assert_eq!(state.components[1].fill, 0.0);
        assert_eq!(state.components[2].fill, 0.5);
        let json = serde_json::to_string(&state).unwrap();
        assert!(
            json.contains(r#"["proprio/eef_pos",{"dim":1},{"dim":2,"fill":0.5}]"#),
            "got: {json}"
        );
    }

    #[test]
    fn rejects_a_state_of_only_constants() {
        let err = serde_json::from_str::<State>(r#"{"components": [{"dim": 3}]}"#).unwrap_err();
        assert!(err.to_string().contains("reads nothing"), "got: {err}");
    }

    #[test]
    fn rejects_a_constant_part_without_dim_or_with_source_fields() {
        let err =
            serde_json::from_str::<State>(r#"{"components": ["r", {"fill": 1.0}]}"#).unwrap_err();
        assert!(err.to_string().contains("needs dim"), "got: {err}");
        let err =
            serde_json::from_str::<State>(r#"{"components": ["r", {"dim": 1, "scale": 2.0}]}"#)
                .unwrap_err();
        assert!(err.to_string().contains("only dim and"), "got: {err}");
        let err =
            serde_json::from_str::<State>(r#"{"components": ["r", {"dim": 1, "frame": "world"}]}"#)
                .unwrap_err();
        assert!(err.to_string().contains("only dim and"), "got: {err}");
    }

    #[test]
    fn rejects_a_non_zero_fill_on_a_required_roled_part() {
        let err = serde_json::from_str::<State>(r#"{"components": [{"role": "r", "fill": 1.0}]}"#)
            .unwrap_err();
        assert!(err.to_string().contains("or optional part"), "got: {err}");
        let ok: State = serde_json::from_str(
            r#"{"components": [{"role": "r", "dim": 1, "optional": true, "fill": 1.0}]}"#,
        )
        .expect("optional parses");
        assert_eq!(ok.components[0].fill, 1.0);
    }

    #[test]
    fn rejects_post_rotate_without_an_encoding() {
        let err = serde_json::from_str::<State>(
            r#"{"components": [{"role": "r", "post_rotate": {"encoding": "rot6d",
               "value": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]}}]}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("needs a rotation"), "got: {err}");
    }

    #[test]
    fn part_round_trips_as_an_object_field_and_is_barred_from_a_constant() {
        let state: State = serde_json::from_str(
            r#"{"components": [{"role": "proprio/eef_pos", "part": "left_arm"}, "proprio/gripper"]}"#,
        )
        .unwrap();
        assert_eq!(state.components[0].part.as_deref(), Some("left_arm"));
        let json = serde_json::to_string(&state).unwrap();
        assert!(
            json.contains(r#"[{"role":"proprio/eef_pos","part":"left_arm"},"proprio/gripper"]"#),
            "got: {json}"
        );
        let err = serde_json::from_str::<State>(
            r#"{"components": ["r", {"dim": 1, "part": "left_arm"}]}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("only dim and"), "got: {err}");
    }

    #[test]
    fn labels_fix_dim_and_refuse_index_duplicates_and_a_constant() {
        let state: State = serde_json::from_str(
            r#"{"components": [{"role": "proprio/joint_pos", "labels": ["FR_hip_joint", "FR_thigh_joint"]}]}"#,
        )
        .unwrap();
        assert_eq!(state.components[0].dim, None);
        let json = serde_json::to_string(&state).unwrap();
        assert!(
            json.contains(
                r#"[{"role":"proprio/joint_pos","labels":["FR_hip_joint","FR_thigh_joint"]}]"#
            ),
            "got: {json}"
        );
        let ok: State = serde_json::from_str(
            r#"{"components": [{"role": "r", "dim": 2, "labels": ["a", "b"]}]}"#,
        )
        .expect("an agreeing dim parses");
        assert_eq!(ok.components[0].dim, Some(2));
        for (doc, expect) in [
            (
                r#"{"components": [{"role": "r", "dim": 3, "labels": ["a", "b"]}]}"#,
                "labels fix the width",
            ),
            (
                r#"{"components": [{"role": "r", "index": 0, "labels": ["a", "b"]}]}"#,
                "both labels and index",
            ),
            (
                r#"{"components": [{"role": "r", "labels": ["a", "a"]}]}"#,
                "is repeated",
            ),
            (
                r#"{"components": [{"role": "r", "labels": []}]}"#,
                "at least one axis",
            ),
            (
                r#"{"components": ["r", {"dim": 1, "labels": ["a"]}]}"#,
                "only dim and",
            ),
        ] {
            let err = serde_json::from_str::<State>(doc).unwrap_err();
            assert!(err.to_string().contains(expect), "{doc}: {err}");
        }
    }

    #[test]
    fn per_axis_affine_is_a_vector_key_and_never_doubles_the_scalar() {
        let state: State = serde_json::from_str(
            r#"{"components": [{"role": "r", "axis_scale": [0.5, 1], "axis_offset": [0.0, -1.5]}]}"#,
        )
        .unwrap();
        assert_eq!(state.components[0].axis_scale, Some(vec![0.5, 1.0]));
        assert_eq!(state.components[0].axis_offset, Some(vec![0.0, -1.5]));
        assert_eq!(state.components[0].scale, None);
        let json = serde_json::to_string(&state).unwrap();
        assert!(
            json.contains(r#""axis_scale":[0.5,1.0],"axis_offset":[0.0,-1.5]"#),
            "got: {json}"
        );
        for (doc, expect) in [
            (
                r#"{"components": [{"role": "r", "scale": 2.0, "axis_scale": [1.0]}]}"#,
                "both scale and axis_scale",
            ),
            (
                r#"{"components": [{"role": "r", "offset": 2.0, "axis_offset": [1.0]}]}"#,
                "both offset and axis_offset",
            ),
            (
                r#"{"components": [{"role": "r", "axis_offset": []}]}"#,
                "at least one value",
            ),
            (
                r#"{"components": [{"role": "r", "axis_scale": ["x"]}]}"#,
                "a number",
            ),
            (
                r#"{"components": ["r", {"dim": 1, "axis_scale": [1.0]}]}"#,
                "only dim and",
            ),
        ] {
            let err = serde_json::from_str::<State>(doc).unwrap_err();
            assert!(err.to_string().contains(expect), "{doc}: {err}");
        }
    }

    #[test]
    fn provenance_is_one_value_or_a_set_and_is_barred_from_a_constant() {
        let state: State = serde_json::from_str(
            r#"{"components": [{"role": "proprio/base_rot", "provenance": "estimated"},
                {"role": "proprio/joint_pos", "provenance": ["privileged", "estimated"]}]}"#,
        )
        .unwrap();
        let single = state.components[0].provenance.as_ref().unwrap();
        assert_eq!(single.first_known(), Some(Provenance::Estimated));
        let set = state.components[1].provenance.as_ref().unwrap();
        assert!(set.accepts(Provenance::Privileged) && set.accepts(Provenance::Estimated));
        let json = serde_json::to_string(&state).unwrap();
        assert!(
            json.contains(r#"{"role":"proprio/base_rot","provenance":"estimated"}"#)
                && json.contains(r#""provenance":["privileged","estimated"]"#),
            "got: {json}"
        );
        let state: State =
            serde_json::from_str(r#"{"components": [{"role": "r", "provenance": "guessed"}]}"#)
                .unwrap();
        assert_eq!(
            state.components[0]
                .provenance
                .as_ref()
                .unwrap()
                .first_known(),
            None
        );
        let err = serde_json::from_str::<State>(
            r#"{"components": ["r", {"dim": 1, "provenance": "sensed"}]}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("only dim and"), "got: {err}");
    }

    #[test]
    fn clip_is_a_container_range_omitted_when_unset() {
        let state: State =
            serde_json::from_str(r#"{"components": ["r"], "clip": [-100, 100]}"#).unwrap();
        assert_eq!(state.clip, Some((-100.0, 100.0)));
        let json = serde_json::to_string(&state).unwrap();
        assert!(json.contains(r#""clip":[-100.0,100.0]"#), "got: {json}");
        let bare: State = serde_json::from_str(r#"{"components": ["r"]}"#).unwrap();
        assert!(!serde_json::to_string(&bare).unwrap().contains("clip"));
        let err = serde_json::from_str::<State>(r#"{"components": ["r"], "clip": [1.0, -1.0]}"#)
            .unwrap_err();
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn an_action_source_part_carries_only_its_identity_labels_and_fill() {
        let state: State = serde_json::from_str(
            r#"{"components": ["proprio/joint_pos",
                {"role": "action/joint_pos", "source": "action"},
                {"role": "action/joint_pos", "source": "action", "part": "left_arm",
                 "labels": ["a", "b"], "dim": 2, "fill": 0.5}]}"#,
        )
        .unwrap();
        assert_eq!(state.components[0].source, PartSource::Observation);
        assert_eq!(state.components[1].source, PartSource::Action);
        assert_eq!(state.components[2].fill, 0.5);
        let json = serde_json::to_string(&state).unwrap();
        assert!(
            json.contains(r#"["proprio/joint_pos",{"role":"action/joint_pos","source":"action"},"#),
            "got: {json}"
        );
        assert_eq!(json.matches("source").count(), 2);
        for (doc, expect) in [
            (
                r#"{"components": ["r", {"dim": 1, "source": "action"}]}"#,
                "needs the role",
            ),
            (
                r#"{"components": [{"role": "r", "source": "action", "scale": 2.0}]}"#,
                "carries only part/labels/dim/fill",
            ),
            (
                r#"{"components": [{"role": "r", "source": "action", "optional": true}]}"#,
                "carries only part/labels/dim/fill",
            ),
            (
                r#"{"components": [{"role": "r", "source": "action", "provenance": "sensed"}]}"#,
                "carries only part/labels/dim/fill",
            ),
            (
                r#"{"components": [{"role": "r", "source": "sideways"}]}"#,
                "unknown variant",
            ),
        ] {
            let err = serde_json::from_str::<State>(doc).unwrap_err();
            assert!(err.to_string().contains(expect), "{doc}: {err}");
        }
    }

    #[test]
    fn bare_role_part_constructs() {
        let part: ConcatPart = serde_json::from_str(r#""only/role""#).unwrap();
        assert_eq!(part.role.as_deref(), Some("only/role"));
        assert!(part.dim.is_none() && !part.optional);
    }
}
