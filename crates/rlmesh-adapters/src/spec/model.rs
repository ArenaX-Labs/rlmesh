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

/// One input *leaf* expected by a model, tagged by the kind of payload it is.
///
/// **Strict v1 kind tag.** A new input *kind* (a new variant here) is a
/// structural change = a v2 key bump, not an additive v1 value; an unknown
/// `type` is rejected at parse by design (the value-vocabulary degradation that
/// applies to [`crate::spec::RotationEncoding`] is deliberately NOT extended to
/// node kinds — a new kind has no defined structure for an old reader).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ModelLeaf {
    Image(Image),
    State(State),
    Text(Text),
    Custom(Custom),
}

/// A node in the recursive model input tree: a leaf, a `Dict` of named
/// sub-nodes, or a `Tuple` of positional sub-nodes.
///
/// The container type written here **is** the payload container the model's
/// `predict` receives. Discrimination on the wire is **structural** (see the
/// hand-written `Deserialize`): a JSON array is a `Tuple`, a JSON object whose
/// `"type"` is in the leaf vocabulary (`image`/`state`/`text`/`custom`) is a
/// `Leaf`, and any other JSON object is a `Dict`. `"type"` is therefore a
/// **reserved key**: a `Dict` child may not be named `"type"`.
#[derive(Debug, Clone, PartialEq)]
pub enum InputNode {
    Leaf(ModelLeaf),
    Dict(BTreeMap<String, InputNode>),
    Tuple(Vec<InputNode>),
}

/// The leaf-vocabulary `type` discriminants that mark a JSON object as a
/// [`ModelLeaf`] rather than an [`InputNode::Dict`].
const MODEL_LEAF_TYPES: &[&str] = &["image", "state", "text", "custom"];

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
        deserializer.deserialize_any(InputNodeVisitor)
    }
}

/// Hand-written visitor mirroring [`AcceptSet`](crate::spec::AcceptSet)'s
/// str-or-map pattern, distinguishing a leaf object from a dict object
/// structurally (by the `"type"` key) so the leaf keeps its own
/// `#[serde(tag = "type")]` form intact.
struct InputNodeVisitor;

impl<'de> Visitor<'de> for InputNodeVisitor {
    type Value = InputNode;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a model input leaf, a dict of nodes, or a tuple (array) of nodes")
    }

    fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<InputNode, A::Error> {
        let mut items = Vec::new();
        while let Some(item) = seq.next_element::<InputNode>()? {
            items.push(item);
        }
        Ok(InputNode::Tuple(items))
    }

    fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<InputNode, A::Error> {
        let mut buffered: BTreeMap<String, serde_json::Value> = BTreeMap::new();
        while let Some((key, value)) = map.next_entry::<String, serde_json::Value>()? {
            buffered.insert(key, value);
        }
        let is_leaf = buffered
            .get("type")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|tag| MODEL_LEAF_TYPES.contains(&tag));
        if is_leaf {
            let object = serde_json::Value::Object(buffered.into_iter().collect());
            let leaf = ModelLeaf::deserialize(object).map_err(serde::de::Error::custom)?;
            return Ok(InputNode::Leaf(leaf));
        }
        let mut children: BTreeMap<String, InputNode> = BTreeMap::new();
        for (key, value) in buffered {
            let child = InputNode::deserialize(value).map_err(serde::de::Error::custom)?;
            children.insert(key, child);
        }
        Ok(InputNode::Dict(children))
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
}
