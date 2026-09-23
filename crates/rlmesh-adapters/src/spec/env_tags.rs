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
use super::frames::FrameRef;
use super::layouts::ImageLayout;
use super::leaf_codec::leaf_codec;
use super::model::{NodeShape, TreeNode, deserialize_node};
use super::rotations::RotationEncoding;

/// A camera image entry's semantics. Width/height/channels are derived from
/// the space, so only the layout (genuinely underdetermined by shape) and the
/// upside-down flag are carried.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageTag {
    pub role: String,
    #[serde(default)]
    pub layout: ImageLayout,
    // `bool` is right while 180° is the only non-trivial orientation (it alone
    // preserves H×W, so it composes with shape-derived dims). If a 2nd
    // orientation is ever needed (e.g. a 90° mount, which swaps H/W), model it
    // as a constrained string defaulting to "upright" — like `dtype`/`resample`
    // — so additive values degrade to a typed resolve error on old cores, not a
    // bool→enum wire break.
    #[serde(default)]
    pub upside_down: bool,
    /// The body part this leaf sits on, when the role repeats across a body
    /// (`left_arm`, `head`, ...): an identity key the resolver matches on, not
    /// a value it checks. Omitted when unset, so every pre-`part` spec is
    /// byte-identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub part: Option<String>,
    /// Unrecognized additive fields, retained verbatim for round-trip and
    /// surfaced to the publish-door `reject_unknowns` guard. See the strict-v1 publish gate.
    #[serde(flatten)]
    pub unknown: BTreeMap<String, serde_json::Value>,
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
    /// The coordinate frame this feature's values are expressed in, when the
    /// role is an absolute pose. Omitted when unset, so every pre-`frame` spec
    /// is byte-identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frame: Option<FrameRef>,
    /// Where this feature's numbers come from: `sensed`, `estimated` or
    /// `privileged` (see [`PROVENANCES`](super::frames::PROVENANCES)). Part
    /// of the leaf's identity, so an env may publish one role under two
    /// provenances. Omitted when unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<FrameRef>,
    /// The body part this feature belongs to; see [`ImageTag::part`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub part: Option<String>,
    /// The axis names of this feature in the order the env emits them, one
    /// per element of the space leaf (checked at `join`). A model that names
    /// the same labels in another order is gathered by name. Omitted when
    /// unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub labels: Option<Vec<String>>,
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

/// Wire form of a [`Field`], deserialized before the cross-field
/// validation `Field` enforces via [`TryFrom`].
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
    /// Unrecognized additive fields, captured instead of hard-erroring so a
    /// newer writer's field survives an older reader; the publish gate rejects
    /// a bare one. See [`ImageTag`].
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
                || wire.labels.is_some())
        {
            return Err(
                "a role-less field (a skip) cannot carry an encoding, range, frame, \
                 provenance, part or labels"
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
            unknown: wire.unknown,
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
    /// The coordinate frame this field's values are expressed in; see
    /// [`StateTag::frame`]. A role-less skip may not carry one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frame: Option<FrameRef>,
    /// Where this field's numbers come from; see [`StateTag::provenance`]. A
    /// role-less skip may not carry one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<FrameRef>,
    /// The body part this field belongs to; see [`ImageTag::part`]. A
    /// role-less skip may not carry one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub part: Option<String>,
    /// The axis names of this field, `dim` of them; see [`StateTag::labels`].
    /// A role-less skip may not carry them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub labels: Option<Vec<String>>,
    /// Unrecognized additive fields, retained for round-trip (see [`ImageTag`]).
    #[serde(flatten)]
    pub unknown: BTreeMap<String, serde_json::Value>,
}

/// Wire form of a [`SplitLayout`], validated via [`TryFrom`] so an empty layout
/// (zero fields) and a duplicate `(role, part)` are rejected by the authoritative Rust
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
        // Keyed by `(role, part, provenance)`: one role may repeat across
        // parts (a joint vector per arm) or provenances (a sim's truth beside
        // its estimate), never within the same key.
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
/// **Tolerant kind tag.** A `type` in `OBS_LEAF_TYPES` deserializes into the
/// strict variant (a malformed payload of a *recognized* kind still hard-errors).
/// An unrecognized `type` becomes [`Unknown`](ObsLeaf::Unknown), retained
/// verbatim — a newer env's new modality parses and relays without loss. It
/// produces no `EnvFeature` at [`join`](crate::join::join) (like an untagged
/// leaf); the resolver ignores it unless a model input references its `role`, in
/// which case resolution fails with a localized
/// [`UnsupportedKind`](crate::v1::ErrorCode). The `split` discriminant carries a
/// [`SplitLayout`] — itself a *leaf* (one tensor split into role fields).
#[derive(Debug, Clone, PartialEq)]
pub enum ObsLeaf {
    Image(ImageTag),
    State(StateTag),
    Split(SplitLayout),
    Text(TextTag),
    /// An observation kind this core does not define. `role` is lifted
    /// opportunistically from the raw object's top-level `role` string (absent ⇒
    /// unreferenceable ⇒ silently dropped); `raw` re-emits byte-faithfully.
    Unknown {
        kind: String,
        role: Option<String>,
        raw: serde_json::Value,
    },
}

// The fragile, internally-tagged + flatten-aware serde codec for this leaf —
// shared verbatim with `ModelLeaf` — lives in one place. The macro emits the
// `OBS_LEAF_TYPES` vocabulary, the owned/borrowed known-variant mirrors, the
// `From` lift, and the hand-written Serialize/Deserialize impls. `lift_role:
// yes` opportunistically lifts the raw object's top-level `role` into the
// `Unknown` arm (absent ⇒ unreferenceable ⇒ silently dropped at resolve).
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
/// The container type written here **is** the runtime container type: a `Dict`
/// node maps a gymnasium `Dict` space, a `Tuple` node maps a `Tuple` space, and
/// a [`Leaf`](ObsNode::Leaf) tags a single space leaf. This replaces the old
/// flat dotted-key map (`{"robot.eef_pos": tag}`) and the magic `"."` root
/// sentinel — a single-leaf observation is a bare [`Leaf`](ObsNode::Leaf).
///
/// Discrimination on the wire is **structural** (the shared
/// `TreeNode` parser): a JSON array is a `Tuple`, a JSON object whose `"type"` is
/// a string is a `Leaf` — a recognized leaf kind (`image`/`state`/`text`/`split`)
/// parses fully, an unrecognized one becomes a tolerant [`ObsLeaf::Unknown`]
/// (dropped with an advisory at resolve unless a model input references it) — and
/// any other JSON object is a `Dict`. `"type"` is therefore a **reserved key**: a
/// `Dict` child may not be named `"type"` (a non-string `"type"` is a clear error).
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
mod tag_tolerant_field_tests {
    use super::ObsLeaf;

    #[test]
    fn obs_tag_captures_unknown_field_and_never_leaks_type() {
        // Tolerant reader: an unknown (typo'd or future-additive) field on a
        // known leaf kind is captured verbatim in `unknown`, not rejected at the
        // serde layer. The publish-door `reject_unknowns` gate is where strictness
        // lives now. Critically, the internally-tagged `type` discriminant is
        // consumed before the variant deserializes, so it never pollutes the
        // capture map (the fragile flatten-on-tagged-variant edge this pins).
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
        // Round-trips verbatim, and the re-emitted object still carries `type`.
        let json = serde_json::to_string(&tag).unwrap();
        assert!(
            json.contains("layuot") && json.contains(r#""type":"image""#),
            "got: {json}"
        );

        // Valid tags (including the `type` tag) still parse, and a clean tag has
        // an empty capture map.
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
        // A role-less field is a skip; it must not carry an encoding or range.
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
    fn part_is_optional_omitted_when_unset_and_barred_from_a_skip() {
        // Additive: absent on the wire when unset (every pre-`part` spec is
        // byte-identical), carried verbatim when set, and meaningless on a
        // role-less skip (it names nothing to place on a body).
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
        // A sim publishes its truth beside its estimate: two fields, one
        // role, distinct provenances.
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
    fn unknown_type_parses_into_a_tolerant_unknown_leaf() {
        // Tolerant reader: an object whose string `type` is outside the leaf
        // vocabulary is no longer a parse error — it becomes an `Unknown` leaf
        // with its `role` lifted and the raw object retained for byte-faithful
        // relay. Whether it matters is decided at *resolve*, not at parse.
        let node: ObsNode =
            serde_json::from_str(r#"{"type": "audio", "role": "x", "sample_rate": 16000}"#)
                .expect("unknown kind parses");
        let ObsNode::Leaf(super::ObsLeaf::Unknown { kind, role, .. }) = &node else {
            panic!("expected an unknown leaf, got {node:?}")
        };
        assert_eq!(kind, "audio");
        assert_eq!(role.as_deref(), Some("x"));
        // Round-trips verbatim: kind and the unknown sub-field both survive.
        let json = serde_json::to_string(&node).unwrap();
        assert!(
            json.contains(r#""type":"audio""#) && json.contains("sample_rate"),
            "got: {json}"
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
