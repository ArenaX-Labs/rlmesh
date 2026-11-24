mod custom;
mod image;
mod state;
mod text;

use std::collections::BTreeMap;

use serde::de::Visitor;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::action::Action;
use super::leaf_codec::leaf_codec;

pub use custom::Custom;
pub use image::{CHANNEL_ORDERS, CROP_MODES, Image, Normalize, StackPad};
pub use state::{ConcatPart, PartSource, State, StateContainer};
pub use text::{Text, TextContainer};

pub(super) enum NodeShape<L, C> {
    Leaf(L),
    Dict(BTreeMap<String, C>),
    Tuple(Vec<C>),
}

pub(super) trait TreeNode: Sized + for<'de> Deserialize<'de> {
    type Leaf: Serialize + for<'de> Deserialize<'de>;

    const KIND: &'static str;

    fn from_shape(shape: NodeShape<Self::Leaf, Self>) -> Self;
}

pub(super) fn deserialize_node<'de, N: TreeNode, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<N, D::Error> {
    deserializer.deserialize_any(NodeVisitor::<N>(std::marker::PhantomData))
}

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
        let mut buffered: BTreeMap<String, serde_json::Value> = BTreeMap::new();
        while let Some((key, value)) = map.next_entry::<String, serde_json::Value>()? {
            buffered.insert(key, value);
        }
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
/// A recognized `type` parses strictly. Any other `type` becomes
/// [`Unknown`](ModelLeaf::Unknown), kept verbatim for round-trip; a model input of an unknown
/// kind fails at resolve with `UnsupportedKind`, named by its placement.
#[derive(Debug, Clone, PartialEq)]
#[allow(
    clippy::large_enum_variant,
    reason = "the image leaf carries the whole pixel pipeline; a leaf is built once per \
              spec and read by placement, so boxing it would buy nothing on the hot path"
)]
pub enum ModelLeaf {
    Image(Image),
    State(State),
    Text(Text),
    Custom(Custom),
    /// A model input kind this core does not define.
    Unknown {
        kind: String,
        raw: serde_json::Value,
    },
}

leaf_codec! {
    leaf: ModelLeaf,
    known: ModelLeafKnown,
    known_ref: ModelLeafKnownRef,
    vocab: MODEL_LEAF_TYPES = "The leaf-vocabulary `type` discriminants that mark a JSON object as a known\n[`ModelLeaf`] variant; any other string `type` is an `Unknown` leaf.",
    missing_type_msg: "a model input leaf needs a string \"type\"",
    lift_role: no,
    variants: {
        Image(Image) = "image",
        State(State) = "state",
        Text(Text) = "text",
        Custom(Custom) = "custom",
    }
}

/// A node in the recursive model input tree: a leaf, a `Dict` of named
/// sub-nodes, or a `Tuple` of positional sub-nodes.
///
/// The container written here is the payload container `predict` receives. On the wire a
/// JSON array is a `Tuple`, an object whose `"type"` is a string is a `Leaf`, and any other
/// object is a `Dict`; `"type"` is therefore reserved and may not name a `Dict` child.
#[derive(Debug, Clone, PartialEq)]
#[allow(
    clippy::large_enum_variant,
    reason = "the leaf IS the payload here; boxing it would put an allocation \
              on every node of a tree that is built once at resolve"
)]
pub enum InputNode {
    Leaf(ModelLeaf),
    Dict(BTreeMap<String, InputNode>),
    Tuple(Vec<InputNode>),
}

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
/// A role may be reused across leaves (one env camera can feed several inputs); placement in
/// the tree keeps each leaf unique.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSpec {
    pub input: InputNode,
    pub output: Action,
    /// Named projections (lossy fits) the model accepts, by name
    /// (`drop_action`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allow: Vec<String>,
    /// How the model paces its actions (see [`Timing`]).
    #[serde(default, skip_serializing_if = "Timing::is_unset")]
    pub timing: Timing,
}

/// How a model paces its actions. `execution` is how many actions of each predicted chunk
/// a run executes before re-planning when neither the run nor the model's `load()` names one
/// (`1..=1024`); omitted when unset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Timing {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution: Option<u32>,
}

impl Timing {
    /// Whether nothing is declared (the field is then omitted from the wire).
    pub fn is_unset(&self) -> bool {
        self.execution.is_none()
    }

    /// Refuse an `execution` outside `1..=1024`.
    pub fn check(&self) -> Result<(), String> {
        match self.execution {
            Some(execution)
                if !(1..=crate::stateful::MAX_EXECUTION_HORIZON).contains(&execution) =>
            {
                Err(format!(
                    "timing.execution must be in 1..={}, got {execution}",
                    crate::stateful::MAX_EXECUTION_HORIZON
                ))
            }
            _ => Ok(()),
        }
    }
}

/// The camera resolutions this model's image inputs assert, in document order.
///
/// One `(role, (height, width))` per [`Image`] leaf that declares a `render`; a reused role
/// appears once per leaf. Document order is dict keys sorted, tuple entries by position, so two
/// walks of one spec agree. Feed them to the environment's camera dial before a run;
/// [`resolve`](crate::v1::resolve) asserts each against the camera it binds.
pub fn render_requests(spec: &ModelSpec) -> Vec<(String, (u32, u32))> {
    fn walk(node: &InputNode, out: &mut Vec<(String, (u32, u32))>) {
        match node {
            InputNode::Leaf(ModelLeaf::Image(image)) => {
                if let Some(size) = image.render {
                    out.push((image.role.clone(), size));
                }
            }
            InputNode::Leaf(_) => {}
            InputNode::Dict(children) => {
                for child in children.values() {
                    walk(child, out);
                }
            }
            InputNode::Tuple(items) => {
                for item in items {
                    walk(item, out);
                }
            }
        }
    }

    let mut requests = Vec::new();
    walk(&spec.input, &mut requests);
    requests
}

#[cfg(test)]
mod tests {
    use super::{InputNode, ModelLeaf, ModelSpec, render_requests};

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
        let node: InputNode =
            serde_json::from_str(r#"{"type": "audio", "role": "x", "sample_rate": 16000}"#)
                .expect("unknown kind parses");
        let InputNode::Leaf(ModelLeaf::Unknown { kind, .. }) = &node else {
            panic!("expected an unknown leaf, got {node:?}")
        };
        assert_eq!(kind, "audio");
        let json = serde_json::to_string(&node).unwrap();
        assert!(
            json.contains(r#""type":"audio""#) && json.contains("sample_rate"),
            "got: {json}"
        );
    }

    fn spec(input: &str) -> ModelSpec {
        let json = format!(
            r#"{{"input": {input}, "output": {{"components": [{{"role": "action/gripper", "dim": 1}}]}}}}"#
        );
        serde_json::from_str(&json).expect("parse spec")
    }

    #[test]
    fn render_requests_walks_the_tree_in_document_order() {
        let requests = render_requests(&spec(
            r#"{
                "zzz": {"type": "image", "role": "image/secondary", "render": [240, 320]},
                "obs": [
                    {"type": "image", "role": "image/primary", "render": 448},
                    {"type": "image", "role": "image/wrist"},
                    {"type": "text", "role": "instruction"},
                    {"type": "image", "role": "image/wrist", "render": [480, 480]}
                ]
            }"#,
        ));
        assert_eq!(
            requests,
            vec![
                ("image/primary".to_owned(), (448, 448)),
                ("image/wrist".to_owned(), (480, 480)),
                ("image/secondary".to_owned(), (240, 320)),
            ]
        );
        assert_eq!(
            requests,
            render_requests(&spec(
                r#"{
                "zzz": {"type": "image", "role": "image/secondary", "render": [240, 320]},
                "obs": [
                    {"type": "image", "role": "image/primary", "render": 448},
                    {"type": "image", "role": "image/wrist"},
                    {"type": "text", "role": "instruction"},
                    {"type": "image", "role": "image/wrist", "render": [480, 480]}
                ]
            }"#,
            ))
        );
    }

    #[test]
    fn render_requests_is_empty_when_no_input_asserts_one() {
        assert!(render_requests(&spec(r#"{"type": "image", "role": "image/primary"}"#)).is_empty());
    }

    #[test]
    fn non_string_type_is_a_reserved_key_error() {
        let err = serde_json::from_str::<InputNode>(r#"{"type": 7}"#)
            .expect_err("non-string type rejected");
        assert!(
            err.to_string().contains(
                r#"the reserved key "type" may not name a dict child (model input tree)"#
            ),
            "got: {err}"
        );
    }

    #[test]
    fn a_declared_execution_timing_round_trips_and_an_unset_one_keeps_the_old_bytes() {
        let mut timed = spec(r#"{"type": "image", "role": "image/primary"}"#);
        let bare = serde_json::to_string(&timed).unwrap();
        assert!(!bare.contains("timing"));
        timed.timing.execution = Some(8);
        let json = serde_json::to_string(&timed).unwrap();
        assert!(json.contains(r#""timing":{"execution":8}"#), "got: {json}");
        assert_eq!(serde_json::from_str::<ModelSpec>(&json).unwrap(), timed);
        assert!(crate::spec::reject_unknowns_model(&timed).is_ok());
        timed.timing.execution = Some(0);
        assert!(
            crate::spec::reject_unknowns_model(&timed)
                .unwrap_err()
                .contains("1..=1024")
        );
    }
}
