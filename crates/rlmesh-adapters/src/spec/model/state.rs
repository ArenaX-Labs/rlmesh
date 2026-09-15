//! A numeric state input expected by a model.

use std::collections::BTreeMap;
use std::fmt;

use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::spec::{FrameRef, RotationLiteral, StateEncoding};

fn default_float32() -> String {
    "float32".to_owned()
}

/// A fill of 0.0 is the default (a zero-filled absent role, a zero constant),
/// omitted on the wire so every pre-`fill` spec stays byte-identical.
fn is_default_fill(fill: &f64) -> bool {
    *fill == 0.0
}

/// One part of a [`State`] concat, sourced from an env state feature.
///
/// A part deserializes from **either** a bare JSON string (a role, sugar for a
/// part carrying only that role) **or** a JSON object with the full field set
/// (`role`, `encoding`, `dim`, `index`, `optional`, `range`, `fill`,
/// `post_rotate`, `scale`, `offset`, `frame`, `part`). On the wire a role-only part round-trips
/// back to a bare string; any other part to an object.
///
/// A part with **no** `role` is a constant: it reads nothing from the env and
/// contributes `dim` copies of `fill` (serialized `{"dim": N[, "fill": v]}`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ConcatPart {
    /// The env state feature this part reads, or `None` for a constant part.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Rotation encoding(s) the model accepts for this part. A bare string (the
    /// common single-encoding case) or a list, in preference order -- the
    /// resolver picks the env's native encoding when it appears here (no
    /// conversion), else converts the env's native into the first entry. Or a
    /// custom-encoding object (`{base, ...}`), a host-side re-packing that
    /// shadows to its `base` for resolution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoding: Option<StateEncoding>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dim: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
    /// Target value range. When set and the env feature declares a (derived
    /// or tagged) source range, values are affinely mapped from the env
    /// range into this one — the state-side analogue of action range mapping.
    /// When the env feature has no source range (an unbounded/non-uniform space
    /// with no `range` tag) there is nothing to map from, so this is a no-op —
    /// it does not clamp or rescale on its own.
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
    /// The coordinate frame the checkpoint was trained to read this part in,
    /// when the role is an absolute pose. Omitted when unset, so every
    /// pre-`frame` spec is byte-identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frame: Option<FrameRef>,
    /// The body part this part reads, when the role repeats across a body
    /// (`left_arm`, ...): an identity key the resolver matches on. Naming one
    /// binds only that leaf; naming none binds the env's only leaf of the role
    /// under any part. Omitted when unset; a constant part may not carry one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub part: Option<String>,
    /// Unrecognized additive fields, retained for round-trip and surfaced to the
    /// publish-door `reject_unknowns` guard. See the strict-v1 publish gate.
    #[serde(flatten)]
    pub unknown: BTreeMap<String, serde_json::Value>,
}

/// Wire form of a [`ConcatPart`]'s object branch, validated via [`TryFrom`] (the
/// `dim`/`index`/`range` deserializers and the `dim`+`index` conflict guard).
#[derive(Deserialize)]
struct ConcatPartWire {
    #[serde(default)]
    role: Option<String>,
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
    #[serde(default)]
    frame: Option<FrameRef>,
    #[serde(default)]
    part: Option<String>,
    // Captured instead of hard-erroring so a newer writer's field survives an
    // older reader; the publish gate rejects a bare one.
    #[serde(flatten)]
    unknown: BTreeMap<String, serde_json::Value>,
}

impl TryFrom<ConcatPartWire> for ConcatPart {
    type Error = String;

    fn try_from(wire: ConcatPartWire) -> Result<Self, Self::Error> {
        // `index` selects one element and `dim` truncates to the leading N;
        // apply applies `index` and silently ignores `dim` when both are set,
        // so reject the ambiguous pairing at the codec instead of picking one.
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
        match &wire.role {
            Some(role) => {
                // `fill` is what an absent part contributes; a roled part that
                // is not `optional` always has an env source, so a non-zero
                // fill there could never fire (the `Actuator` rule's twin).
                if wire.fill != 0.0 && !wire.optional {
                    return Err(format!(
                        "state part {role:?}: fill applies only to a constant (role-less) \
                         or optional part; a roled, non-optional part takes its values \
                         from the env"
                    ));
                }
                // A post-rotation is a rotation: it needs an encoding to decode
                // into a matrix and re-encode out of, and the host-side repack
                // of a custom encoding owns its own geometry.
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
            // A constant part emits `dim` copies of `fill` and reads nothing, so
            // every source-mapping field is meaningless on it (the state-side
            // mirror of a role-less actuator's rule).
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
                    || wire.frame.is_some()
                    || wire.part.is_some()
                {
                    return Err("a constant (role-less) state part carries only dim and \
                         fill; drop encoding/index/range/optional/post_rotate/scale/offset/\
                         frame/part"
                        .to_owned());
                }
            }
        }
        Ok(ConcatPart {
            role: wire.role,
            encoding: wire.encoding,
            dim: wire.dim,
            index: wire.index,
            range: wire.range,
            optional: wire.optional,
            fill: wire.fill,
            post_rotate: wire.post_rotate,
            scale: wire.scale,
            offset: wire.offset,
            frame: wire.frame,
            part: wire.part,
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
                    encoding: None,
                    dim: None,
                    index: None,
                    range: None,
                    optional: false,
                    fill: 0.0,
                    post_rotate: None,
                    scale: None,
                    offset: None,
                    frame: None,
                    part: None,
                    unknown: BTreeMap::new(),
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

/// Custom serialize so a role-only part round-trips to a bare string (matching
/// the str-or-map wire form), and any richer part to an object.
fn serialize_concat_part<S: Serializer>(
    part: &ConcatPart,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let role_only = part.role.is_some()
        && part.encoding.is_none()
        && part.dim.is_none()
        && part.index.is_none()
        && part.range.is_none()
        && !part.optional
        && is_default_fill(&part.fill)
        && part.post_rotate.is_none()
        && part.scale.is_none()
        && part.offset.is_none()
        && part.frame.is_none()
        && part.part.is_none()
        && part.unknown.is_empty();
    if let (true, Some(role)) = (role_only, &part.role) {
        serializer.serialize_str(role)
    } else {
        // Reuse the derived Serialize on the struct (the `#[derive(Serialize)]`
        // above), which skips the unset optionals.
        part.serialize(serializer)
    }
}

/// Serialize a `Vec<ConcatPart>` part-by-part through [`serialize_concat_part`].
fn serialize_parts<S: Serializer>(parts: &[ConcatPart], serializer: S) -> Result<S::Ok, S::Error> {
    use serde::ser::SerializeSeq;
    let mut seq = serializer.serialize_seq(Some(parts.len()))?;
    for part in parts {
        seq.serialize_element(&PartWrapper(part))?;
    }
    seq.end()
}

/// Newtype so a `ConcatPart` inside the parts list serializes through the
/// str-or-object policy rather than the derived struct form.
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
/// Deserialization goes through `StateWire` so an empty `components` list is
/// rejected by the authoritative codec — matching the Python mirror. A single
/// role can be authored as `Concat("role")` (one role-only part); a packed
/// state lists several parts. There is no `key` — placement is the tree
/// position the [`State`] leaf sits at.
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
    /// Unrecognized additive fields, retained for round-trip and surfaced to the
    /// publish-door `reject_unknowns` guard. See the strict-v1 publish gate. Threaded
    /// through `StateWire` (which previously dropped unknown fields silently).
    #[serde(flatten)]
    pub unknown: BTreeMap<String, serde_json::Value>,
}

fn is_default_container(container: &StateContainer) -> bool {
    *container == StateContainer::Array
}

/// Wire form of [`State`]; see its docs for the non-empty-components rule.
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
    // Retained verbatim instead of silently dropped (the pre-tolerance bug): the
    // single field rule is flatten-capture, threaded into `State` below.
    #[serde(flatten)]
    unknown: BTreeMap<String, serde_json::Value>,
}

impl TryFrom<StateWire> for State {
    type Error = String;

    fn try_from(wire: StateWire) -> Result<Self, Self::Error> {
        if wire.components.is_empty() {
            return Err("a state input needs at least one component".to_owned());
        }
        // Every part a constant means the input reads nothing from the env: a
        // fabricated tensor pretending to be an observation.
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
            unknown: wire.unknown,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{ConcatPart, State};

    #[test]
    fn rejects_empty_components() {
        // A state input with no components is rejected at the codec, so the
        // publish door never blesses a spec the read path cannot reconstruct.
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
        // A Concat part can be a bare role (sugar) or a full object.
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
        // A part with a `part` is no longer role-only, so it serializes as an
        // object; a role-only part still collapses to the bare string.
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
    fn bare_role_part_constructs() {
        let part: ConcatPart = serde_json::from_str(r#""only/role""#).unwrap();
        assert_eq!(part.role.as_deref(), Some("only/role"));
        assert!(part.dim.is_none() && !part.optional);
    }
}
