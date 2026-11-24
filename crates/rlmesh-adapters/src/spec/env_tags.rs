//! The sparse env-side tags users author over a gymnasium space.
//!
//! Tags carry only semantics: the role each observation entry plays and how to read it
//! (image layout, rotation encoding, value range). Structure (widths, dtypes, bounds) comes
//! from the space; `join` validates the tags against it and derives the
//! [`EnvFeatures`](super::env::EnvFeatures) the resolver consumes.

use std::collections::BTreeMap;

use super::conventions::Polarity;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::AcceptSet;
use super::FrameRef;
use super::action::Action;
use super::layouts::ImageLayout;
use super::leaf_codec::leaf_codec;
use super::model::{NodeShape, TreeNode, deserialize_node};
use super::rotations::RotationEncoding;

/// A camera image entry's semantics. Width, height and channels come from the space;
/// only the layout and the upside-down flag are carried.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageTag {
    pub role: String,
    #[serde(default)]
    pub layout: ImageLayout,
    #[serde(default)]
    pub upside_down: bool,
    /// The body part this leaf sits on, when the role repeats across a body
    /// (`left_arm`, `head`, ...): an identity key the resolver matches on, not
    /// a value it checks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub part: Option<String>,
    /// Unrecognized additive fields, retained verbatim for round-trip and
    /// surfaced to the publish-door `reject_unknowns` guard.
    #[serde(flatten)]
    pub unknown: BTreeMap<String, serde_json::Value>,
}

/// A numeric proprioception entry's semantics. The width comes from the space; an
/// `encoding` declares a rotation representation (its width is checked against the
/// space) and `range` overrides infinite space bounds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateTag {
    pub role: String,
    #[serde(default)]
    pub encoding: Option<AcceptSet<RotationEncoding>>,
    #[serde(default, deserialize_with = "crate::spec::num::de_opt_range")]
    pub range: Option<(f64, f64)>,
    /// The coordinate frame this feature's values are expressed in, when the
    /// role is an absolute pose.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frame: Option<FrameRef>,
    /// Where this feature's numbers come from: `sensed`, `estimated` or
    /// `privileged` (see [`PROVENANCES`](super::PROVENANCES)).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<FrameRef>,
    /// The body part this feature belongs to; see [`ImageTag::part`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub part: Option<String>,
    /// The axis names of this feature in the order the env emits them, one
    /// per element of the space leaf (checked at `join`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub labels: Option<Vec<String>>,
    /// Openness (`open`) or closedness (`close`) for a gripper reading
    /// (`proprio/gripper_pos`); see `Polarity`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub polarity: Option<Polarity>,
    /// Unrecognized additive fields, retained for round-trip (see [`ImageTag`]).
    #[serde(flatten)]
    pub unknown: BTreeMap<String, serde_json::Value>,
}

/// A text entry's semantics (typically the task instruction).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextTag {
    pub role: String,
    /// Unrecognized additive fields, retained for round-trip (see [`ImageTag`]).
    #[serde(flatten)]
    pub unknown: BTreeMap<String, serde_json::Value>,
}

#[derive(Deserialize)]
struct FieldWire {
    #[serde(default)]
    role: Option<String>,
    #[serde(deserialize_with = "crate::spec::num::de_count")]
    dim: u32,
    #[serde(default)]
    encoding: Option<AcceptSet<RotationEncoding>>,
    #[serde(default, deserialize_with = "crate::spec::num::de_opt_range")]
    range: Option<(f64, f64)>,
    #[serde(default)]
    frame: Option<FrameRef>,
    #[serde(default)]
    provenance: Option<FrameRef>,
    #[serde(default)]
    part: Option<String>,
    #[serde(default)]
    labels: Option<Vec<String>>,
    #[serde(default)]
    polarity: Option<Polarity>,
    /// Unrecognized additive fields, captured instead of hard-erroring so a
    /// newer writer's field survives an older reader; the publish gate rejects
    /// a bare one.
    #[serde(flatten)]
    unknown: BTreeMap<String, serde_json::Value>,
}

impl TryFrom<FieldWire> for Field {
    type Error = String;

    fn try_from(wire: FieldWire) -> Result<Self, Self::Error> {
        if wire.dim < 1 {
            return Err(format!("state field dim must be >= 1, got {}", wire.dim));
        }
        if wire.role.is_none()
            && (wire.encoding.is_some()
                || wire.range.is_some()
                || wire.frame.is_some()
                || wire.provenance.is_some()
                || wire.part.is_some()
                || wire.labels.is_some()
                || wire.polarity.is_some())
        {
            return Err(
                "a role-less field (a skip) cannot carry an encoding, range, frame, \
                 provenance, part, labels or polarity"
                    .to_owned(),
            );
        }
        if let Some(labels) = &wire.labels {
            let locus = format!("state field {:?}", wire.role);
            crate::spec::labels::check_labels(labels, &locus)?;
            if labels.len() != wire.dim as usize {
                return Err(format!(
                    "{locus}: declares dim {} but names {} labels; one label per element",
                    wire.dim,
                    labels.len()
                ));
            }
        }
        Ok(Field {
            role: wire.role,
            dim: wire.dim,
            encoding: wire.encoding,
            range: wire.range,
            frame: wire.frame,
            provenance: wire.provenance,
            part: wire.part,
            labels: wire.labels,
            polarity: wire.polarity,
            unknown: wire.unknown,
        })
    }
}

/// One contiguous field of a flat numeric observation leaf.
///
/// The observation-side mirror of [`Actuator`](super::action::Actuator): `dim >= 1`
/// elements carrying a `role`, at an offset implied by order within a [`SplitLayout`]. A
/// field with no `role` is a skip: it takes up width but produces no feature.
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
    /// The coordinate frame this field's values are expressed in; see
    /// [`StateTag::frame`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frame: Option<FrameRef>,
    /// Where this field's numbers come from; see [`StateTag::provenance`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<FrameRef>,
    /// The body part this field belongs to; see [`ImageTag::part`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub part: Option<String>,
    /// The axis names of this field, `dim` of them; see [`StateTag::labels`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub labels: Option<Vec<String>>,
    /// Unrecognized additive fields, retained for round-trip (see [`ImageTag`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub polarity: Option<Polarity>,
    #[serde(flatten)]
    pub unknown: BTreeMap<String, serde_json::Value>,
}

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
        for field in &wire.fields {
            if let Some(role) = field.role.as_deref()
                && !seen.insert((
                    role,
                    field.part.as_deref(),
                    field.provenance.as_ref().map(FrameRef::as_str),
                ))
            {
                return Err(match &field.part {
                    Some(part) => format!(
                        "a state layout declares role {role:?} under part {part:?} more than once"
                    ),
                    None => format!("a state layout declares role {role:?} more than once"),
                });
            }
        }
        Ok(SplitLayout {
            fields: wire.fields,
        })
    }
}

/// An ordered split of one flat numeric observation leaf into role fields.
///
/// Field offsets accumulate in order, and the widths must sum to the leaf width. Use it
/// when an env returns a flat `Box` whose fixed index ranges mean different things.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "SplitLayoutWire")]
pub struct SplitLayout {
    pub fields: Vec<Field>,
}

/// One observation *leaf*: the semantics of a single space leaf, tagged by the
/// kind of leaf it describes.
///
/// A malformed payload of a known `type` is an error; an unknown `type` parses as
/// [`Unknown`](ObsLeaf::Unknown) and round-trips verbatim. It produces no feature, and
/// resolution fails only if a model input references its `role`.
#[derive(Debug, Clone, PartialEq)]
pub enum ObsLeaf {
    Image(ImageTag),
    State(StateTag),
    Split(SplitLayout),
    Text(TextTag),
    /// An observation kind this core does not define. `role` is the raw object's
    /// top-level `role` string, if any (without one it cannot be referenced); `raw`
    /// re-serializes unchanged.
    Unknown {
        kind: String,
        role: Option<String>,
        raw: serde_json::Value,
    },
}

leaf_codec! {
    leaf: ObsLeaf,
    known: ObsLeafKnown,
    known_ref: ObsLeafKnownRef,
    vocab: OBS_LEAF_TYPES = "The leaf-vocabulary `type` discriminants that mark a JSON object as a known\n[`ObsLeaf`] variant; any other string `type` is an `Unknown` leaf.",
    missing_type_msg: "an observation leaf needs a string \"type\"",
    lift_role: yes,
    variants: {
        Image(ImageTag) = "image",
        State(StateTag) = "state",
        Split(SplitLayout) = "split",
        Text(TextTag) = "text",
    }
}

/// A node in the recursive env observation tree: a leaf, a `Dict` of named
/// sub-nodes, or a `Tuple` of positional sub-nodes.
///
/// The container type is the space's: a `Dict` node maps a `Dict` space, a `Tuple` node a
/// `Tuple` space, and a single-leaf observation is a bare [`Leaf`](ObsNode::Leaf). On the
/// wire a JSON array is a `Tuple`, an object whose `"type"` is a string is a `Leaf`, and
/// any other object is a `Dict`, so a `Dict` child may not be named `"type"`.
#[derive(Debug, Clone, PartialEq)]
pub enum ObsNode {
    Leaf(ObsLeaf),
    Dict(BTreeMap<String, ObsNode>),
    Tuple(Vec<ObsNode>),
}

impl TreeNode for ObsNode {
    type Leaf = ObsLeaf;

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

/// The env-side tags: the recursive observation tree plus the action layout. Space leaves
/// the tree does not reach are allowed and carry no semantics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnvTags {
    pub observation: ObsNode,
    pub action: Action,
}

#[cfg(test)]
mod tag_tolerant_field_tests {
    use super::ObsLeaf;

    #[test]
    fn obs_tag_captures_unknown_field_and_never_leaks_type() {
        let tag: ObsLeaf =
            serde_json::from_str(r#"{"type": "image", "role": "x", "layuot": "chw"}"#).unwrap();
        let ObsLeaf::Image(image) = &tag else {
            panic!("expected image")
        };
        assert_eq!(image.unknown.get("layuot"), Some(&serde_json::json!("chw")));
        assert!(
            !image.unknown.contains_key("type"),
            "type leaked into capture"
        );
        let json = serde_json::to_string(&tag).unwrap();
        assert!(
            json.contains("layuot") && json.contains(r#""type":"image""#),
            "got: {json}"
        );

        let tag: ObsLeaf =
            serde_json::from_str(r#"{"type": "image", "role": "x", "layout": "chw"}"#).unwrap();
        let ObsLeaf::Image(image) = &tag else {
            panic!("expected image")
        };
        assert!(image.unknown.is_empty());
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
        let err = serde_json::from_str::<Field>(r#"{"dim": 3, "encoding": "rot6d"}"#).unwrap_err();
        assert!(err.to_string().contains("role-less"), "got: {err}");
        let err = serde_json::from_str::<Field>(r#"{"dim": 3, "range": [0.0, 1.0]}"#).unwrap_err();
        assert!(err.to_string().contains("role-less"), "got: {err}");
        let err = serde_json::from_str::<Field>(r#"{"dim": 3, "frame": "world"}"#).unwrap_err();
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
        use super::SplitLayout;
        let err = serde_json::from_str::<SplitLayout>(r#"{"fields": []}"#).unwrap_err();
        assert!(err.to_string().contains("at least one field"), "got: {err}");
        let ok: SplitLayout = serde_json::from_str(r#"{"fields": [{"role": "x", "dim": 1}]}"#)
            .expect("non-empty layout parses");
        assert_eq!(ok.fields.len(), 1);
    }

    #[test]
    fn part_is_optional_omitted_when_unset_and_barred_from_a_skip() {
        let field: Field = serde_json::from_str(r#"{"role": "x", "dim": 3}"#).unwrap();
        assert_eq!(field.part, None);
        assert!(!serde_json::to_string(&field).unwrap().contains("part"));
        let field: Field =
            serde_json::from_str(r#"{"role": "x", "dim": 3, "part": "left_arm"}"#).unwrap();
        assert_eq!(field.part.as_deref(), Some("left_arm"));
        assert!(
            serde_json::to_string(&field)
                .unwrap()
                .contains(r#""part":"left_arm""#)
        );
        let err = serde_json::from_str::<Field>(r#"{"dim": 3, "part": "left_arm"}"#).unwrap_err();
        assert!(err.to_string().contains("role-less"), "got: {err}");

        for (doc, expect) in [
            (
                r#"{"type": "state", "role": "proprio/eef_pos", "part": "left_arm"}"#,
                true,
            ),
            (r#"{"type": "state", "role": "proprio/eef_pos"}"#, false),
            (
                r#"{"type": "image", "role": "image/wrist", "part": "left_arm"}"#,
                true,
            ),
            (r#"{"type": "image", "role": "image/wrist"}"#, false),
        ] {
            let leaf: super::ObsLeaf = serde_json::from_str(doc).unwrap();
            let json = serde_json::to_string(&leaf).unwrap();
            assert_eq!(
                json.contains(r#""part":"left_arm""#),
                expect,
                "{doc} -> {json}"
            );
        }
    }

    #[test]
    fn labels_are_optional_match_dim_and_are_barred_from_a_skip() {
        let field: Field =
            serde_json::from_str(r#"{"role": "x", "dim": 2, "labels": ["a", "b"]}"#).unwrap();
        assert_eq!(field.labels.as_ref().map(Vec::len), Some(2));
        assert!(
            serde_json::to_string(&field)
                .unwrap()
                .contains(r#""labels":["a","b"]"#)
        );
        let bare: Field = serde_json::from_str(r#"{"role": "x", "dim": 2}"#).unwrap();
        assert!(!serde_json::to_string(&bare).unwrap().contains("labels"));
        for (doc, expect) in [
            (
                r#"{"role": "x", "dim": 3, "labels": ["a", "b"]}"#,
                "one label per element",
            ),
            (
                r#"{"role": "x", "dim": 2, "labels": ["a", "a"]}"#,
                "is repeated",
            ),
            (r#"{"dim": 1, "labels": ["a"]}"#, "role-less"),
        ] {
            let err = serde_json::from_str::<Field>(doc).unwrap_err();
            assert!(err.to_string().contains(expect), "{doc}: {err}");
        }
        let tag: super::ObsLeaf = serde_json::from_str(
            r#"{"type": "state", "role": "proprio/joint_pos", "labels": ["FR_hip_joint"]}"#,
        )
        .unwrap();
        let json = serde_json::to_string(&tag).unwrap();
        assert!(json.contains(r#""labels":["FR_hip_joint"]"#), "{json}");
    }

    #[test]
    fn provenance_is_one_value_omitted_when_unset_and_keys_a_split_field() {
        use super::SplitLayout;
        let tag: super::ObsLeaf = serde_json::from_str(
            r#"{"type": "state", "role": "proprio/base_rot", "provenance": "privileged"}"#,
        )
        .unwrap();
        let json = serde_json::to_string(&tag).unwrap();
        assert!(json.contains(r#""provenance":"privileged""#), "{json}");
        let bare: super::ObsLeaf =
            serde_json::from_str(r#"{"type": "state", "role": "proprio/base_rot"}"#).unwrap();
        assert!(!serde_json::to_string(&bare).unwrap().contains("provenance"));
        let err =
            serde_json::from_str::<Field>(r#"{"dim": 3, "provenance": "sensed"}"#).unwrap_err();
        assert!(err.to_string().contains("role-less"), "got: {err}");
        let ok: SplitLayout = serde_json::from_str(
            r#"{"fields": [{"role": "proprio/base_rot", "dim": 4, "provenance": "privileged"},
                           {"role": "proprio/base_rot", "dim": 4, "provenance": "estimated"}]}"#,
        )
        .expect("one role, two provenances");
        assert_eq!(
            ok.fields[1].provenance.as_ref().map(|p| p.as_str()),
            Some("estimated")
        );
        let err = serde_json::from_str::<SplitLayout>(
            r#"{"fields": [{"role": "r", "dim": 1, "provenance": "estimated"},
                           {"role": "r", "dim": 1, "provenance": "estimated"}]}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("more than once"), "got: {err}");
    }

    #[test]
    fn a_role_may_repeat_across_parts_but_not_within_one() {
        use super::SplitLayout;
        let ok: SplitLayout = serde_json::from_str(
            r#"{"fields": [{"role": "r", "dim": 1, "part": "left_arm"},
                           {"role": "r", "dim": 1, "part": "right_arm"},
                           {"role": "r", "dim": 1}]}"#,
        )
        .expect("one role, three parts");
        assert_eq!(ok.fields.len(), 3);
        let err = serde_json::from_str::<SplitLayout>(
            r#"{"fields": [{"role": "r", "dim": 1, "part": "left_arm"},
                           {"role": "r", "dim": 1, "part": "left_arm"}]}"#,
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains(r#"under part "left_arm" more than once"#),
            "got: {err}"
        );
    }

    #[test]
    fn rejects_duplicate_layout_role() {
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

    fn round_trip(json: &str) -> ObsNode {
        let node: ObsNode = serde_json::from_str(json).expect("parse node");
        let serialized = serde_json::to_string(&node).expect("serialize node");
        let reparsed: ObsNode = serde_json::from_str(&serialized).expect("re-parse node");
        assert_eq!(node, reparsed, "round-trip differs for {json}");
        node
    }

    #[test]
    fn single_leaf_is_bare() {
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
        let node = round_trip(
            r#"{"type": "split", "fields": [{"role": "proprio/eef_pos", "dim": 3}, {"dim": 1}]}"#,
        );
        assert!(matches!(node, ObsNode::Leaf(ObsLeaf::Split(_))));
    }

    #[test]
    fn unknown_type_parses_into_a_tolerant_unknown_leaf() {
        let node: ObsNode =
            serde_json::from_str(r#"{"type": "audio", "role": "x", "sample_rate": 16000}"#)
                .expect("unknown kind parses");
        let ObsNode::Leaf(super::ObsLeaf::Unknown { kind, role, .. }) = &node else {
            panic!("expected an unknown leaf, got {node:?}")
        };
        assert_eq!(kind, "audio");
        assert_eq!(role.as_deref(), Some("x"));
        let json = serde_json::to_string(&node).unwrap();
        assert!(
            json.contains(r#""type":"audio""#) && json.contains("sample_rate"),
            "got: {json}"
        );
    }

    #[test]
    fn non_string_type_is_a_reserved_key_error() {
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
