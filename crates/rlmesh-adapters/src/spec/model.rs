//! The model-side spec: expected input payload tree plus the action output.

mod custom;
mod image;
mod state;
mod text;

use std::collections::BTreeMap;

use serde::de::Visitor;
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
    /// The leaf payload, internally `#[serde(tag = "type")]`-tagged.
    type Leaf: Serialize + for<'de> Deserialize<'de>;

    /// The `type` discriminants that mark a JSON object as a leaf (not a dict).
    const LEAF_TYPES: &'static [&'static str];

    /// Domain word naming this tree in errors, e.g. `"model input"` →
    /// `unknown model input kind "audio"`.
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
        // `"type"` is a reserved Dict key: its presence forces the leaf branch,
        // so an unknown or non-string `"type"` is a clear error here rather than
        // a misparse-as-Dict that fails deep with a misleading message.
        if let Some(tag) = buffered.get("type") {
            match tag.as_str() {
                Some(tag) if N::LEAF_TYPES.contains(&tag) => {
                    let object = serde_json::Value::Object(buffered.into_iter().collect());
                    let leaf = N::Leaf::deserialize(object).map_err(serde::de::Error::custom)?;
                    return Ok(N::from_shape(NodeShape::Leaf(leaf)));
                }
                Some(unknown) => {
                    return Err(serde::de::Error::custom(format!(
                        "unknown {} kind {unknown:?}",
                        N::KIND
                    )));
                }
                None => {
                    return Err(serde::de::Error::custom(format!(
                        "the reserved key \"type\" may not name a dict child ({} tree)",
                        N::KIND
                    )));
                }
            }
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
/// **Strict v1 kind tag.** A new input *kind* (a new variant here) is a
/// structural change = a v2 key bump, not an additive v1 value; an unknown
/// `type` is rejected at parse with a clear `unknown model input kind` error
/// (the value-vocabulary degradation that applies to
/// [`crate::spec::RotationEncoding`] is deliberately NOT extended to node kinds
/// — a new kind has no defined structure for an old reader).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ModelLeaf {
    Image(Image),
    State(State),
    Text(Text),
    Custom(Custom),
}

/// The leaf-vocabulary `type` discriminants that mark a JSON object as a
/// [`ModelLeaf`] rather than an [`InputNode::Dict`].
pub const MODEL_LEAF_TYPES: &[&str] = &["image", "state", "text", "custom"];

/// A node in the recursive model input tree: a leaf, a `Dict` of named
/// sub-nodes, or a `Tuple` of positional sub-nodes.
///
/// The container type written here **is** the payload container the model's
/// `predict` receives. Discrimination on the wire is **structural** (the shared
/// `TreeNode` parser): a JSON array is a `Tuple`, a JSON object whose `"type"`
/// is in the leaf vocabulary (`image`/`state`/`text`/`custom`) is a `Leaf`, an
/// object whose `"type"` is an unrecognized string is a clear
/// `unknown model input kind` error, and any other JSON object is a `Dict`.
/// `"type"` is therefore a **reserved key**: a `Dict` child may not be named
/// `"type"`.
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

    const LEAF_TYPES: &'static [&'static str] = MODEL_LEAF_TYPES;
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
    fn unknown_type_is_a_clear_kind_error() {
        // An object whose `type` is a string outside the leaf vocabulary names
        // the unknown kind directly — `type` is reserved, so it is never
        // misparsed as a Dict child.
        let err = serde_json::from_str::<InputNode>(r#"{"type": "audio", "role": "x"}"#)
            .expect_err("unknown kind rejected");
        assert!(
            err.to_string()
                .contains(r#"unknown model input kind "audio""#),
            "got: {err}"
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
