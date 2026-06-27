//! The model-side spec: expected input payload tree plus the action output.

mod custom;
mod image;
mod state;
mod text;

use std::collections::BTreeMap;

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::action::Action;

pub use custom::Custom;
pub use image::Image;
pub use state::{ConcatPart, State, StateContainer};
pub use text::{Text, TextContainer};

// ---------------------------------------------------------------------------
// Shared structural node parser
//
// Both spec trees — the model input tree ([`InputNode`]) and the env
// observation tree ([`super::env_tags::ObsNode`]) — share one structural
// discriminant: a JSON array is a `Tuple`, a JSON object whose `"type"` is in
// the leaf vocabulary is a `Leaf`, an object whose `"type"` is an unrecognized
// string is a clear unknown-kind error, and any other object is a `Dict`. This
// is the single Rust source of that discriminant; each tree supplies only its
// leaf type, leaf vocabulary, and the `KIND` word naming it in errors.
// ---------------------------------------------------------------------------

/// The three structural shapes a tree node can take, generic over the leaf
/// payload `L` and the child node `C`, so [`from_shape`](TreeNode::from_shape)
/// builds a node from the discriminant's output without duplicating the
/// leaf/dict/tuple match per tree.
pub(super) enum NodeShape<L, C> {
    Leaf(L),
    Dict(BTreeMap<String, C>),
    Tuple(Vec<C>),
}

/// A recursive spec tree (leaf / dict / tuple) parsed by the shared structural
/// discriminant. Implementors supply only their leaf type, leaf `type`
/// vocabulary, and the `KIND` word naming the tree in the unknown-kind error;
/// the [`Deserialize`] impl delegates to [`deserialize_node`], which is why that
/// trait is required here. (Serialization is the trivial inverse — a leaf
/// serializes as its tagged object, a dict as a JSON object, a tuple as a JSON
/// array — and stays a per-type three-arm match.)
pub(super) trait TreeNode: Sized + for<'de> Deserialize<'de> {
    /// The leaf payload. Its own [`Deserialize`] sorts a known `type` into the
    /// strict variant and an unrecognized `type` into the tolerant `Unknown`
    /// arm, so the visitor's job is purely structural (leaf / dict / tuple).
    type Leaf: Serialize + for<'de> Deserialize<'de>;

    /// Domain word naming this tree in the reserved-`type`-key error, e.g.
    /// `"model input"`.
    const KIND: &'static str;

    /// Build a node from one of the three owned structural shapes the shared
    /// discriminant produces.
    fn from_shape(shape: NodeShape<Self::Leaf, Self>) -> Self;
}

/// Deserialize any [`TreeNode`] via the shared structural discriminant.
pub(super) fn deserialize_node<'de, N: TreeNode, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<N, D::Error> {
    deserializer.deserialize_any(NodeVisitor::<N>(std::marker::PhantomData))
}

/// Hand-written visitor mirroring [`AcceptSet`](crate::spec::AcceptSet)'s
/// str-or-map pattern: it distinguishes a leaf object from a dict object
/// structurally (by the `"type"` key) so the leaf keeps its own
/// `#[serde(tag = "type")]` form intact.
struct NodeVisitor<N: TreeNode>(std::marker::PhantomData<N>);

impl<'de, N: TreeNode> Visitor<'de> for NodeVisitor<N> {
    type Value = N;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            formatter,
            "a {} leaf, a dict of nodes, or a tuple (array) of nodes",
            N::KIND
        )
    }

    fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<N, A::Error> {
        let mut items = Vec::new();
        while let Some(item) = seq.next_element::<N>()? {
            items.push(item);
        }
        Ok(N::from_shape(NodeShape::Tuple(items)))
    }

    fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<N, A::Error> {
        // Buffer the object into an ordered map of raw JSON so we can peek at the
        // `"type"` key, then re-interpret as either a leaf or a dict of nodes.
        let mut buffered: BTreeMap<String, serde_json::Value> = BTreeMap::new();
        while let Some((key, value)) = map.next_entry::<String, serde_json::Value>()? {
            buffered.insert(key, value);
        }
        // `"type"` is a reserved Dict key: a *string* `"type"` forces the leaf
        // branch (the leaf's own Deserialize then sorts known vs `Unknown` kind),
        // so the visitor stays purely structural. A non-string `"type"` cannot be
        // a leaf discriminant and is malformed — a clear error here rather than a
        // misparse-as-Dict that fails deep with a misleading message.
        if let Some(tag) = buffered.get("type") {
            if tag.is_string() {
                let object = serde_json::Value::Object(buffered.into_iter().collect());
                let leaf = N::Leaf::deserialize(object).map_err(serde::de::Error::custom)?;
                return Ok(N::from_shape(NodeShape::Leaf(leaf)));
            }
            return Err(serde::de::Error::custom(format!(
                "the reserved key \"type\" may not name a dict child ({} tree)",
                N::KIND
            )));
        }
        let mut children: BTreeMap<String, N> = BTreeMap::new();
        for (key, value) in buffered {
            let child = N::deserialize(value).map_err(serde::de::Error::custom)?;
            children.insert(key, child);
        }
        Ok(N::from_shape(NodeShape::Dict(children)))
    }
}

/// One input *leaf* expected by a model, tagged by the kind of payload it is.
///
/// **Tolerant kind tag.** A `type` in `MODEL_LEAF_TYPES` deserializes into the
/// strict variant (a malformed payload of a *recognized* kind still hard-errors).
/// An unrecognized `type` becomes [`Unknown`](ModelLeaf::Unknown), retained
/// verbatim for round-trip — a newer peer's new modality parses and relays
/// without loss, and surfaces only at *resolve*: a model input of an unknown
/// kind is always an [`UnsupportedKind`](crate::v1::ErrorCode) error (an old core
/// has no apply path for it), named by its placement.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelLeaf {
    Image(Image),
    State(State),
    Text(Text),
    Custom(Custom),
    /// A model input kind this core does not define. Carries the raw object
    /// verbatim (it already embeds `type`) so it re-emits byte-faithfully.
    Unknown {
        kind: String,
        raw: serde_json::Value,
    },
}

/// The leaf-vocabulary `type` discriminants that mark a JSON object as a known
/// [`ModelLeaf`] variant; any other string `type` is an `Unknown` leaf.
pub const MODEL_LEAF_TYPES: &[&str] = &["image", "state", "text", "custom"];

/// Owned mirror of the *known* [`ModelLeaf`] variants. Reuses serde's
/// internally-tagged derive (which strips `type` before the variant's flatten
/// capture sees it) so the fragile tagged+flatten interaction lives in one
/// derive, not hand-rolled dispatch.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum ModelLeafKnown {
    Image(Image),
    State(State),
    Text(Text),
    Custom(Custom),
}

impl From<ModelLeafKnown> for ModelLeaf {
    fn from(known: ModelLeafKnown) -> Self {
        match known {
            ModelLeafKnown::Image(input) => ModelLeaf::Image(input),
            ModelLeafKnown::State(input) => ModelLeaf::State(input),
            ModelLeafKnown::Text(input) => ModelLeaf::Text(input),
            ModelLeafKnown::Custom(input) => ModelLeaf::Custom(input),
        }
    }
}

/// Borrowed mirror for Serialize (re-emits the internally-tagged known form).
#[derive(Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum ModelLeafKnownRef<'a> {
    Image(&'a Image),
    State(&'a State),
    Text(&'a Text),
    Custom(&'a Custom),
}

impl Serialize for ModelLeaf {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            ModelLeaf::Image(input) => ModelLeafKnownRef::Image(input).serialize(serializer),
            ModelLeaf::State(input) => ModelLeafKnownRef::State(input).serialize(serializer),
            ModelLeaf::Text(input) => ModelLeafKnownRef::Text(input).serialize(serializer),
            ModelLeaf::Custom(input) => ModelLeafKnownRef::Custom(input).serialize(serializer),
            // The raw object already embeds `type`; emit it verbatim.
            ModelLeaf::Unknown { raw, .. } => raw.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for ModelLeaf {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        let kind = value
            .get("type")
            .and_then(|tag| tag.as_str())
            .ok_or_else(|| de::Error::custom("a model input leaf needs a string \"type\""))?;
        if MODEL_LEAF_TYPES.contains(&kind) {
            // Malformed payload of a recognized kind still hard-errors here.
            ModelLeafKnown::deserialize(value)
                .map(ModelLeaf::from)
                .map_err(de::Error::custom)
        } else {
            Ok(ModelLeaf::Unknown {
                kind: kind.to_owned(),
                raw: value,
            })
        }
    }
}

/// A node in the recursive model input tree: a leaf, a `Dict` of named
/// sub-nodes, or a `Tuple` of positional sub-nodes.
///
/// The container type written here **is** the payload container the model's
/// `predict` receives. Discrimination on the wire is **structural** (the shared
/// `TreeNode` parser): a JSON array is a `Tuple`, a JSON object whose `"type"`
/// is a string is a `Leaf` — a recognized leaf kind
/// (`image`/`state`/`text`/`custom`) parses fully, an unrecognized one becomes a
/// tolerant [`ModelLeaf::Unknown`] (rejected only at resolve, and only if a model
/// input references it) — and any other JSON object is a `Dict`. `"type"` is
/// therefore a **reserved key**: a `Dict` child may not be named `"type"` (a
/// non-string `"type"` is a clear error).
#[derive(Debug, Clone, PartialEq)]
pub enum InputNode {
    Leaf(ModelLeaf),
    Dict(BTreeMap<String, InputNode>),
    Tuple(Vec<InputNode>),
}

/// Wires [`InputNode`] into the shared structural [`TreeNode`] parser; the
/// only model-specific knowledge is its leaf type, leaf vocabulary, and the
/// `KIND` word used in the unknown-kind error.
impl TreeNode for InputNode {
    type Leaf = ModelLeaf;

    const KIND: &'static str = "model input";

    fn from_shape(shape: NodeShape<<Self as TreeNode>::Leaf, Self>) -> Self {
        match shape {
            NodeShape::Leaf(leaf) => InputNode::Leaf(leaf),
            NodeShape::Dict(map) => InputNode::Dict(map),
            NodeShape::Tuple(items) => InputNode::Tuple(items),
        }
    }
}

impl Serialize for InputNode {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            InputNode::Leaf(leaf) => leaf.serialize(serializer),
            InputNode::Dict(map) => map.serialize(serializer),
            InputNode::Tuple(items) => items.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for InputNode {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserialize_node(deserializer)
    }
}

/// Declarative description of a model's input payload tree and action output.
///
/// `input` is the recursive [`InputNode`] tree the model's `predict` receives;
/// `output` is the model's action layout. A model role may be reused across
/// leaves (one env camera can feed several input slots) — there is no
/// duplicate-key check, since placement (tree position) is unique by structure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSpec {
    pub input: InputNode,
    pub output: Action,
}

#[cfg(test)]
mod tests {
    use super::{InputNode, ModelLeaf, ModelSpec};

    /// Parse, serialize, re-parse, and assert structural stability (the
    /// serializer fills leaf defaults, so we compare the parsed structs rather
    /// than byte-comparing to the minimal input).
    fn round_trip(json: &str) -> InputNode {
        let node: InputNode = serde_json::from_str(json).expect("parse node");
        let serialized = serde_json::to_string(&node).expect("serialize node");
        let reparsed: InputNode = serde_json::from_str(&serialized).expect("re-parse node");
        assert_eq!(node, reparsed, "round-trip differs for {json}");
        node
    }

    #[test]
    fn single_bare_leaf() {
        let node = round_trip(r#"{"type": "image", "role": "image/primary", "size": 224}"#);
        assert!(matches!(node, InputNode::Leaf(ModelLeaf::Image(_))));
    }

    #[test]
    fn flat_dict_input() {
        let node = round_trip(
            r#"{"pixels": {"type": "image", "role": "image/primary"}, "prompt": {"type": "text", "role": "instruction"}}"#,
        );
        let InputNode::Dict(map) = node else {
            panic!("expected dict")
        };
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn nested_dict_input() {
        let node = round_trip(
            r#"{"obs": {"state": {"type": "state", "components": ["proprio/eef_pos"]}}}"#,
        );
        assert!(matches!(node, InputNode::Dict(_)));
    }

    #[test]
    fn tuple_input() {
        let node = round_trip(
            r#"[{"type": "image", "role": "image/primary"}, {"type": "text", "role": "instruction"}]"#,
        );
        let InputNode::Tuple(items) = node else {
            panic!("expected tuple")
        };
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn concat_leaf_with_two_parts() {
        let node = round_trip(
            r#"{"type": "state", "components": ["proprio/eef_pos", {"role": "proprio/gripper", "dim": 1}]}"#,
        );
        let InputNode::Leaf(ModelLeaf::State(state)) = node else {
            panic!("expected state leaf")
        };
        assert_eq!(state.components.len(), 2);
    }

    #[test]
    fn model_spec_round_trips() {
        let spec: ModelSpec = serde_json::from_str(
            r#"{"input": {"type": "image", "role": "image/primary"}, "output": {"components": [{"role": "action/delta", "dim": 7}]}}"#,
        )
        .expect("parse spec");
        assert!(matches!(spec.input, InputNode::Leaf(ModelLeaf::Image(_))));
        assert_eq!(spec.output.components.len(), 1);
    }

    #[test]
    fn unknown_type_parses_into_a_tolerant_unknown_leaf() {
        // Tolerant reader: an unrecognized string `type` becomes an `Unknown`
        // leaf retaining the raw object, not a parse error. A model input of an
        // unknown kind is rejected at *resolve* (UnsupportedKind), never here.
        let node: InputNode =
            serde_json::from_str(r#"{"type": "audio", "role": "x", "sample_rate": 16000}"#)
                .expect("unknown kind parses");
        let InputNode::Leaf(ModelLeaf::Unknown { kind, .. }) = &node else {
            panic!("expected an unknown leaf, got {node:?}")
        };
        assert_eq!(kind, "audio");
        // Round-trips verbatim.
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
        let err = serde_json::from_str::<InputNode>(r#"{"type": 7}"#)
            .expect_err("non-string type rejected");
        assert!(
            err.to_string().contains(
                r#"the reserved key "type" may not name a dict child (model input tree)"#
            ),
            "got: {err}"
        );
    }
}
