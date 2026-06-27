//! A numeric state input expected by a model.

use std::collections::BTreeMap;
use std::fmt;

use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::spec::AcceptSet;
use crate::spec::rotations::RotationEncoding;

fn default_float32() -> String {
    "float32".to_owned()
}

/// One part of a [`State`] concat, sourced from an env state feature.
///
/// A part deserializes from **either** a bare JSON string (a role, sugar for a
/// part carrying only that role) **or** a JSON object with the full field set
/// (`role`, `encoding`, `dim`, `index`, `optional`, `range`). On the wire a
/// role-only part round-trips back to a bare string; any other part to an
/// object. The field set is identical to the pre-redesign `StateComponent` so
/// `plan_state`/`StatePiece`/`apply_state` consume it unchanged.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ConcatPart {
    pub role: String,
    /// Rotation encoding(s) the model accepts for this part, in preference
    /// order (most-preferred first). The resolver picks the env's native
    /// encoding when it appears here (no conversion), else converts the env's
    /// native into the first entry. A bare string on the wire for the common
    /// single-encoding case.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoding: Option<AcceptSet<RotationEncoding>>,
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
}

/// Wire form of a [`ConcatPart`]'s object branch, validated via [`TryFrom`] (the
/// `dim`/`index`/`range` deserializers and the `dim`+`index` conflict guard).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ConcatPartWire {
    role: String,
    #[serde(default)]
    encoding: Option<AcceptSet<RotationEncoding>>,
    #[serde(default, deserialize_with = "crate::spec::num::de_opt_count")]
    dim: Option<u32>,
    #[serde(default, deserialize_with = "crate::spec::num::de_opt_count")]
    index: Option<u32>,
    #[serde(default, deserialize_with = "crate::spec::num::de_opt_range")]
    range: Option<(f64, f64)>,
    #[serde(default)]
    optional: bool,
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
        Ok(ConcatPart {
            role: wire.role,
            encoding: wire.encoding,
            dim: wire.dim,
            index: wire.index,
            range: wire.range,
            optional: wire.optional,
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
                    role: value.to_owned(),
                    encoding: None,
                    dim: None,
                    index: None,
                    range: None,
                    optional: false,
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
    let role_only = part.encoding.is_none()
        && part.dim.is_none()
        && part.index.is_none()
        && part.range.is_none()
        && !part.optional;
    if role_only {
        serializer.serialize_str(&part.role)
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
        assert_eq!(state.components[0].role, "proprio/eef_pos");
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
    fn bare_role_part_constructs() {
        let part: ConcatPart = serde_json::from_str(r#""only/role""#).unwrap();
        assert_eq!(part.role, "only/role");
        assert!(part.dim.is_none() && !part.optional);
    }
}
