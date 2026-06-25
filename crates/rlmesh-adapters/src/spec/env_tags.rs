//! The sparse env-side tags users author over a gymnasium space.
//!
//! Tags carry only *semantics* — the role each observation entry plays
//! and how to interpret it (image layout, rotation encoding, value range).
//! All *structure* (keys' widths, dtypes, bounds) lives in the gymnasium
//! space and is derived by [`join`](crate::join::join), which validates
//! the tags against the space and produces the internal
//! [`EnvFeatures`](super::env::EnvFeatures) the resolver consumes.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::AcceptSet;
use super::action::ActionLayout;
use super::layouts::ImageLayout;
use super::rotations::RotationEncoding;

/// A camera image entry's semantics. Width/height/channels are derived from
/// the space, so only the layout (genuinely underdetermined by shape) and the
/// upside-down flag are carried.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageTag {
    pub role: String,
    #[serde(default)]
    pub layout: ImageLayout,
    #[serde(default)]
    pub upside_down: bool,
}

/// A numeric proprioception entry's semantics. The width is derived from the
/// space; an `encoding` declares a rotation representation (and its width is
/// then checked against the space) and `range` overrides infinite space
/// bounds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateTag {
    pub role: String,
    #[serde(default)]
    pub encoding: Option<AcceptSet<RotationEncoding>>,
    #[serde(default, deserialize_with = "crate::spec::num::de_opt_range")]
    pub range: Option<(f64, f64)>,
}

/// A text entry's semantics (typically the task instruction).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextTag {
    pub role: String,
}

/// Wire form of a [`StateField`], deserialized before the cross-field
/// validation `StateField` enforces via [`TryFrom`].
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StateFieldWire {
    #[serde(default)]
    role: Option<String>,
    #[serde(deserialize_with = "crate::spec::num::de_count")]
    dim: u32,
    #[serde(default)]
    encoding: Option<AcceptSet<RotationEncoding>>,
    #[serde(default, deserialize_with = "crate::spec::num::de_opt_range")]
    range: Option<(f64, f64)>,
}

impl TryFrom<StateFieldWire> for StateField {
    type Error = String;

    fn try_from(wire: StateFieldWire) -> Result<Self, Self::Error> {
        if wire.dim < 1 {
            return Err(format!("state field dim must be >= 1, got {}", wire.dim));
        }
        if wire.role.is_none() && (wire.encoding.is_some() || wire.range.is_some()) {
            return Err("a role-less field (a skip) cannot carry an encoding or range".to_owned());
        }
        Ok(StateField {
            role: wire.role,
            dim: wire.dim,
            encoding: wire.encoding,
            range: wire.range,
        })
    }
}

/// One contiguous field of a flat numeric observation leaf.
///
/// The observation-side mirror of [`ActionComponent`](super::action::ActionComponent):
/// a slice of `dim` elements carrying a `role`, with offsets implied by order
/// within a [`StateLayout`]. A field with no `role` is a *skip* — it advances
/// the offset and contributes to the layout's width but produces no feature.
/// Deserialization goes through `StateFieldWire` so `dim >= 1` and the
/// role-less-skip rule are enforced by the authoritative Rust codec.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "StateFieldWire")]
pub struct StateField {
    #[serde(default)]
    pub role: Option<String>,
    pub dim: u32,
    #[serde(default)]
    pub encoding: Option<AcceptSet<RotationEncoding>>,
    #[serde(default)]
    pub range: Option<(f64, f64)>,
}

/// Wire form of a [`StateLayout`], validated via [`TryFrom`] so an empty layout
/// (zero fields) and a duplicate role are rejected by the authoritative Rust
/// codec — matching the Python `StateLayout` guard and Rust `join`
/// ([`JoinError::DuplicateLayoutRole`](crate::v1::JoinError)). Without this the
/// two engines disagree: Rust accepts `fields: []` (or a repeated role) and
/// Python's `from_dict` (which normalizes through Rust first, then reconstructs)
/// crashes in its own constructor on input the codec just called valid.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StateLayoutWire {
    fields: Vec<StateField>,
}

impl TryFrom<StateLayoutWire> for StateLayout {
    type Error = String;

    fn try_from(wire: StateLayoutWire) -> Result<Self, Self::Error> {
        if wire.fields.is_empty() {
            return Err("a state layout needs at least one field".to_owned());
        }
        let mut seen = std::collections::BTreeSet::new();
        for role in wire.fields.iter().filter_map(|field| field.role.as_deref()) {
            if !seen.insert(role) {
                return Err(format!(
                    "a state layout declares role {role:?} more than once"
                ));
            }
        }
        Ok(StateLayout {
            fields: wire.fields,
        })
    }
}

/// An ordered split of one flat numeric observation leaf into role fields.
///
/// The observation-side mirror of [`ActionLayout`](super::action::ActionLayout):
/// fields are laid out in order, offsets accumulate, and `join` requires the
/// field widths to sum to the leaf width. Use it when an env returns a flat
/// `Box` whose fixed index ranges carry distinct semantics (e.g. Metaworld).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "StateLayoutWire")]
pub struct StateLayout {
    pub fields: Vec<StateField>,
}

/// One observation tag, tagged by the kind of space leaf it describes.
///
/// **Strict v1 kind tag.** A new observation *kind* (a new variant here) is a
/// structural change = a v2 key bump, not an additive v1 value; an unknown
/// `type` is rejected at parse by design.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ObsTag {
    Image(ImageTag),
    State(StateTag),
    Layout(StateLayout),
    Text(TextTag),
}

/// The env-side tags: a sparse map from observation key-path to its
/// semantics, plus the action layout.
///
/// Observation keys are space key-paths: a dotted path traverses nested
/// `Dict` spaces (`"robot.eef_pos"`), and the reserved key `"."` denotes a
/// flat/root observation (valid only when it is the sole entry). Untagged
/// space keys are allowed; they simply carry no semantics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnvTags {
    pub observation: BTreeMap<String, ObsTag>,
    pub action: ActionLayout,
}

#[cfg(test)]
mod state_field_wire_tests {
    use super::StateField;

    #[test]
    fn rejects_zero_dim() {
        let err = serde_json::from_str::<StateField>(r#"{"role": "x", "dim": 0}"#).unwrap_err();
        assert!(err.to_string().contains("dim must be >= 1"), "got: {err}");
    }

    #[test]
    fn rejects_skip_carrying_encoding_or_range() {
        // A role-less field is a skip; it must not carry an encoding or range.
        let err =
            serde_json::from_str::<StateField>(r#"{"dim": 3, "encoding": "rot6d"}"#).unwrap_err();
        assert!(err.to_string().contains("role-less"), "got: {err}");
        let err =
            serde_json::from_str::<StateField>(r#"{"dim": 3, "range": [0.0, 1.0]}"#).unwrap_err();
        assert!(err.to_string().contains("role-less"), "got: {err}");
    }

    #[test]
    fn accepts_valid_field_and_pure_skip() {
        let field: StateField = serde_json::from_str(r#"{"role": "x", "dim": 3}"#).unwrap();
        assert_eq!(field.dim, 3);
        let skip: StateField = serde_json::from_str(r#"{"dim": 2}"#).unwrap();
        assert!(skip.role.is_none());
    }

    #[test]
    fn rejects_empty_layout() {
        // Parity with Python's StateLayout guard: a zero-field layout is
        // rejected here, so the codec never calls valid a doc Python can't read.
        use super::StateLayout;
        let err = serde_json::from_str::<StateLayout>(r#"{"fields": []}"#).unwrap_err();
        assert!(err.to_string().contains("at least one field"), "got: {err}");
        let ok: StateLayout = serde_json::from_str(r#"{"fields": [{"role": "x", "dim": 1}]}"#)
            .expect("non-empty layout parses");
        assert_eq!(ok.fields.len(), 1);
    }

    #[test]
    fn rejects_duplicate_layout_role() {
        // Parity with Python's StateLayout guard and Rust join's
        // DuplicateLayoutRole: a role repeated across fields is rejected at the
        // codec, so the normalize/publish door never blesses a layout the read
        // path (or join) rejects. A role-less skip can repeat freely.
        use super::StateLayout;
        let err = serde_json::from_str::<StateLayout>(
            r#"{"fields": [{"role": "r", "dim": 1}, {"role": "r", "dim": 1}]}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("more than once"), "got: {err}");
        let ok: StateLayout =
            serde_json::from_str(r#"{"fields": [{"dim": 1}, {"dim": 2}]}"#).expect("skips repeat");
        assert_eq!(ok.fields.len(), 2);
    }
}
