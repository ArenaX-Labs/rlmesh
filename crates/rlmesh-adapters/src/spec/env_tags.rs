//! The sparse env-side tags users author over a gymnasium space.
//!
//! Tags carry only *semantics* — the role each observation entry plays
//! and how to interpret it (image layout, rotation encoding, value range).
//! All *structure* (keys' widths, dtypes, bounds) lives in the gymnasium
//! space and is derived by [`join`](crate::join::join), which validates
//! the tags against the space and produces the internal
//! [`EnvFeatures`](super::env::EnvFeatures) the resolver consumes.

use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::AcceptSet;
use super::action::Action;
use super::layouts::ImageLayout;
use super::model::{NodeShape, TreeNode, deserialize_node};
use super::rotations::RotationEncoding;

/// A camera image entry's semantics. Width/height/channels are derived from
/// the space, so only the layout (genuinely underdetermined by shape) and the
/// upside-down flag are carried.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
pub struct StateTag {
    pub role: String,
    #[serde(default)]
    pub encoding: Option<AcceptSet<RotationEncoding>>,
    #[serde(default, deserialize_with = "crate::spec::num::de_opt_range")]
    pub range: Option<(f64, f64)>,
}

/// A text entry's semantics (typically the task instruction).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextTag {
    pub role: String,
}

/// Wire form of a [`Field`], deserialized before the cross-field
/// validation `Field` enforces via [`TryFrom`].
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FieldWire {
    #[serde(default)]
    role: Option<String>,
    #[serde(deserialize_with = "crate::spec::num::de_count")]
    dim: u32,
    #[serde(default)]
    encoding: Option<AcceptSet<RotationEncoding>>,
    #[serde(default, deserialize_with = "crate::spec::num::de_opt_range")]
    range: Option<(f64, f64)>,
}

impl TryFrom<FieldWire> for Field {
    type Error = String;

    fn try_from(wire: FieldWire) -> Result<Self, Self::Error> {
        if wire.dim < 1 {
            return Err(format!("state field dim must be >= 1, got {}", wire.dim));
        }
        if wire.role.is_none() && (wire.encoding.is_some() || wire.range.is_some()) {
            return Err("a role-less field (a skip) cannot carry an encoding or range".to_owned());
        }
        Ok(Field {
            role: wire.role,
            dim: wire.dim,
            encoding: wire.encoding,
            range: wire.range,
        })
    }
}

/// One contiguous field of a flat numeric observation leaf.
///
/// The observation-side mirror of [`Actuator`](super::action::Actuator):
/// a slice of `dim` elements carrying a `role`, with offsets implied by order
/// within a [`SplitLayout`]. A field with no `role` is a *skip* — it advances
/// the offset and contributes to the layout's width but produces no feature.
/// Deserialization goes through `FieldWire` so `dim >= 1` and the
/// role-less-skip rule are enforced by the authoritative Rust codec.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "FieldWire")]
pub struct Field {
    #[serde(default)]
    pub role: Option<String>,
    pub dim: u32,
    #[serde(default)]
    pub encoding: Option<AcceptSet<RotationEncoding>>,
    #[serde(default)]
    pub range: Option<(f64, f64)>,
}

/// Wire form of a [`SplitLayout`], validated via [`TryFrom`] so an empty layout
/// (zero fields) and a duplicate role are rejected by the authoritative Rust
/// codec — matching the Python `SplitLayout` guard and Rust `join`
/// ([`JoinError::DuplicateLayoutRole`](crate::v1::JoinError)). Without this the
/// two engines disagree: Rust accepts `fields: []` (or a repeated role) and
/// Python's `from_dict` (which normalizes through Rust first, then reconstructs)
/// crashes in its own constructor on input the codec just called valid.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SplitLayoutWire {
    fields: Vec<Field>,
}

impl TryFrom<SplitLayoutWire> for SplitLayout {
    type Error = String;

    fn try_from(wire: SplitLayoutWire) -> Result<Self, Self::Error> {
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
        Ok(SplitLayout {
            fields: wire.fields,
        })
    }
}

/// An ordered split of one flat numeric observation leaf into role fields.
///
/// The observation-side mirror of [`Action`](super::action::Action):
/// fields are laid out in order, offsets accumulate, and `join` requires the
/// field widths to sum to the leaf width. Use it when an env returns a flat
/// `Box` whose fixed index ranges carry distinct semantics (e.g. Metaworld).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "SplitLayoutWire")]
pub struct SplitLayout {
    pub fields: Vec<Field>,
}

/// One observation *leaf*: the semantics of a single space leaf, tagged by the
/// kind of leaf it describes.
///
/// **Strict v1 kind tag.** A new observation *kind* (a new variant here) is a
/// structural change = a v2 key bump, not an additive v1 value; an unknown
/// `type` is rejected at parse with a clear `unknown observation kind` error.
/// The `split` discriminant carries a [`SplitLayout`] — itself a *leaf* (one
/// tensor split into role fields), not a container.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ObsLeaf {
    Image(ImageTag),
    State(StateTag),
    Split(SplitLayout),
    Text(TextTag),
}

/// The leaf-vocabulary `type` discriminants that mark a JSON object as an
/// [`ObsLeaf`] rather than an [`ObsNode::Dict`].
pub const OBS_LEAF_TYPES: &[&str] = &["image", "state", "text", "split"];

/// A node in the recursive env observation tree: a leaf, a `Dict` of named
/// sub-nodes, or a `Tuple` of positional sub-nodes.
///
/// The container type written here **is** the runtime container type: a `Dict`
/// node maps a gymnasium `Dict` space, a `Tuple` node maps a `Tuple` space, and
/// a [`Leaf`](ObsNode::Leaf) tags a single space leaf. This replaces the old
/// flat dotted-key map (`{"robot.eef_pos": tag}`) and the magic `"."` root
/// sentinel — a single-leaf observation is a bare [`Leaf`](ObsNode::Leaf).
///
/// Discrimination on the wire is **structural** (the shared
/// [`TreeNode`](super::model::TreeNode) parser): a JSON array is a `Tuple`, a
/// JSON object whose `"type"` is in the leaf vocabulary
/// (`image`/`state`/`text`/`split`) is a `Leaf`, an object whose `"type"` is an
/// unrecognized string is a clear `unknown observation kind` error, and any
/// other JSON object is a `Dict`. `"type"` is therefore a **reserved key**: a
/// `Dict` child may not be named `"type"`.
#[derive(Debug, Clone, PartialEq)]
pub enum ObsNode {
    Leaf(ObsLeaf),
    Dict(BTreeMap<String, ObsNode>),
    Tuple(Vec<ObsNode>),
}

/// Wires [`ObsNode`] into the shared structural [`TreeNode`] parser; the only
/// observation-specific knowledge is its leaf type, leaf vocabulary, and the
/// `KIND` word used in the unknown-kind error.
impl TreeNode for ObsNode {
    type Leaf = ObsLeaf;

    const LEAF_TYPES: &'static [&'static str] = OBS_LEAF_TYPES;
    const KIND: &'static str = "observation";

    fn from_shape(shape: NodeShape<<Self as TreeNode>::Leaf, Self>) -> Self {
        match shape {
            NodeShape::Leaf(leaf) => ObsNode::Leaf(leaf),
            NodeShape::Dict(map) => ObsNode::Dict(map),
            NodeShape::Tuple(items) => ObsNode::Tuple(items),
        }
    }
}

impl Serialize for ObsNode {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            ObsNode::Leaf(leaf) => leaf.serialize(serializer),
            ObsNode::Dict(map) => map.serialize(serializer),
            ObsNode::Tuple(items) => items.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for ObsNode {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserialize_node(deserializer)
    }
}

/// The env-side tags: the recursive observation tree plus the action layout.
///
/// `observation` is an [`ObsNode`] whose container type is the runtime container
/// type (a `Dict` node maps a `Dict` space, a `Tuple` node maps a `Tuple` space,
/// a bare `Leaf` tags a single space leaf). Untagged space leaves are allowed
/// where the tree does not descend; they simply carry no semantics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnvTags {
    pub observation: ObsNode,
    pub action: Action,
}

#[cfg(test)]
mod tag_deny_unknown_tests {
    use super::ObsLeaf;

    #[test]
    fn obs_tag_rejects_typod_field_but_accepts_valid() {
        // serde 1.0.228 honors deny_unknown_fields on an internally-tagged
        // variant (the `type` tag is stripped before the variant deserializes),
        // so a typo'd authoring field on the trust boundary is rejected at parse
        // instead of silently defaulting.
        for typo in [
            r#"{"type": "image", "role": "x", "layuot": "chw"}"#,
            r#"{"type": "state", "role": "x", "rnge": [0.0, 1.0]}"#,
            r#"{"type": "text", "role": "x", "rol": "y"}"#,
        ] {
            let err = serde_json::from_str::<ObsLeaf>(typo).unwrap_err();
            assert!(err.to_string().contains("unknown field"), "got: {err}");
        }
        // Valid tags (including the `type` tag) still parse.
        let tag: ObsLeaf =
            serde_json::from_str(r#"{"type": "image", "role": "x", "layout": "chw"}"#).unwrap();
        assert!(matches!(tag, ObsLeaf::Image(_)));
    }
}

#[cfg(test)]
mod state_field_wire_tests {
    use super::Field;

    #[test]
    fn rejects_zero_dim() {
        let err = serde_json::from_str::<Field>(r#"{"role": "x", "dim": 0}"#).unwrap_err();
        assert!(err.to_string().contains("dim must be >= 1"), "got: {err}");
    }

    #[test]
    fn rejects_skip_carrying_encoding_or_range() {
        // A role-less field is a skip; it must not carry an encoding or range.
        let err = serde_json::from_str::<Field>(r#"{"dim": 3, "encoding": "rot6d"}"#).unwrap_err();
        assert!(err.to_string().contains("role-less"), "got: {err}");
        let err = serde_json::from_str::<Field>(r#"{"dim": 3, "range": [0.0, 1.0]}"#).unwrap_err();
        assert!(err.to_string().contains("role-less"), "got: {err}");
    }

    #[test]
    fn accepts_valid_field_and_pure_skip() {
        let field: Field = serde_json::from_str(r#"{"role": "x", "dim": 3}"#).unwrap();
        assert_eq!(field.dim, 3);
        let skip: Field = serde_json::from_str(r#"{"dim": 2}"#).unwrap();
        assert!(skip.role.is_none());
    }

    #[test]
    fn rejects_empty_layout() {
        // Parity with Python's SplitLayout guard: a zero-field layout is
        // rejected here, so the codec never calls valid a doc Python can't read.
        use super::SplitLayout;
        let err = serde_json::from_str::<SplitLayout>(r#"{"fields": []}"#).unwrap_err();
        assert!(err.to_string().contains("at least one field"), "got: {err}");
        let ok: SplitLayout = serde_json::from_str(r#"{"fields": [{"role": "x", "dim": 1}]}"#)
            .expect("non-empty layout parses");
        assert_eq!(ok.fields.len(), 1);
    }

    #[test]
    fn rejects_duplicate_layout_role() {
        // Parity with Python's SplitLayout guard and Rust join's
        // DuplicateLayoutRole: a role repeated across fields is rejected at the
        // codec, so the normalize/publish door never blesses a layout the read
        // path (or join) rejects. A role-less skip can repeat freely.
        use super::SplitLayout;
        let err = serde_json::from_str::<SplitLayout>(
            r#"{"fields": [{"role": "r", "dim": 1}, {"role": "r", "dim": 1}]}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("more than once"), "got: {err}");
        let ok: SplitLayout =
            serde_json::from_str(r#"{"fields": [{"dim": 1}, {"dim": 2}]}"#).expect("skips repeat");
        assert_eq!(ok.fields.len(), 2);
    }
}

#[cfg(test)]
mod obs_node_serde_tests {
    use super::{ObsLeaf, ObsNode};

    /// Parse, serialize, re-parse, and assert structural stability (the
    /// serializer fills leaf defaults, so we compare the parsed structs rather
    /// than byte-comparing to the minimal input).
    fn round_trip(json: &str) -> ObsNode {
        let node: ObsNode = serde_json::from_str(json).expect("parse node");
        let serialized = serde_json::to_string(&node).expect("serialize node");
        let reparsed: ObsNode = serde_json::from_str(&serialized).expect("re-parse node");
        assert_eq!(node, reparsed, "round-trip differs for {json}");
        node
    }

    #[test]
    fn single_leaf_is_bare() {
        // A single-leaf observation is a bare leaf object — no `"."` sentinel.
        let node = round_trip(r#"{"type": "state", "role": "proprio/eef_pos"}"#);
        assert!(matches!(node, ObsNode::Leaf(ObsLeaf::State(_))));
    }

    #[test]
    fn flat_dict_of_leaves() {
        let node = round_trip(
            r#"{"cam": {"type": "image", "role": "image/primary"}, "instruction": {"type": "text", "role": "instruction"}}"#,
        );
        let ObsNode::Dict(map) = node else {
            panic!("expected dict")
        };
        assert_eq!(map.len(), 2);
        assert!(matches!(map["cam"], ObsNode::Leaf(ObsLeaf::Image(_))));
    }

    #[test]
    fn nested_dict() {
        let node =
            round_trip(r#"{"robot": {"eef_pos": {"type": "state", "role": "proprio/eef_pos"}}}"#);
        let ObsNode::Dict(map) = node else {
            panic!("expected dict")
        };
        assert!(matches!(map["robot"], ObsNode::Dict(_)));
    }

    #[test]
    fn tuple_of_nodes() {
        let node = round_trip(
            r#"[{"type": "state", "role": "proprio/eef_pos"}, {"type": "text", "role": "instruction"}]"#,
        );
        let ObsNode::Tuple(items) = node else {
            panic!("expected tuple")
        };
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn split_leaf() {
        // The `split` discriminant (was `layout`) deserializes as a leaf, not a dict.
        let node = round_trip(
            r#"{"type": "split", "fields": [{"role": "proprio/eef_pos", "dim": 3}, {"dim": 1}]}"#,
        );
        assert!(matches!(node, ObsNode::Leaf(ObsLeaf::Split(_))));
    }

    #[test]
    fn unknown_type_is_a_clear_kind_error() {
        // An object whose `type` is a string outside the leaf vocabulary names
        // the unknown kind directly — `type` is reserved, so it is never
        // misparsed as a Dict child.
        let err = serde_json::from_str::<ObsNode>(r#"{"type": "audio", "role": "x"}"#)
            .expect_err("unknown kind rejected");
        assert!(
            err.to_string()
                .contains(r#"unknown observation kind "audio""#),
            "got: {err}"
        );
    }

    #[test]
    fn non_string_type_is_a_reserved_key_error() {
        // `type` is a reserved Dict key: a non-string `type` cannot be a leaf
        // discriminant, so it is rejected rather than misparsed as a Dict child.
        let err = serde_json::from_str::<ObsNode>(r#"{"type": 7}"#)
            .expect_err("non-string type rejected");
        assert!(
            err.to_string().contains(
                r#"the reserved key "type" may not name a dict child (observation tree)"#
            ),
            "got: {err}"
        );
    }
}
