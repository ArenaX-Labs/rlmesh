//! A model probe's stand-in environment, built from the model's own spec.
//!
//! A probe needs no paired env: each model input becomes one observation leaf tagged with the
//! role it asks for, plus an action space the width of the declared output, and the binder
//! resolves the model against that as against a real env. The stand-in is an [`Env`] that
//! serves scripted observations and keeps the actions it receives.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard};

use rlmesh_adapters::roles::registry::{DimLaw, role_def};
use rlmesh_adapters::v1::{
    Action, ConcatPart, EnvTags, FrameRef, Image, ImageLayout, ImageTag, InputNode, ModelLeaf,
    ModelSpec, ObsLeaf, ObsNode, SpaceView, State, StateTag, Text, TextTag, join, render_requests,
};
use rlmesh_spaces::spaces::{BoxSpaceBuilder, DictSpaceBuilder, TextBuilder};
use rlmesh_spaces::{DType, SpaceSpec as WireSpec, SpaceValue};
use serde_json::Value as Json;

use super::report::RenderRequest;
use crate::spaces::{Space, SpaceSpec};
use crate::value::space_codec;
use crate::{Env, InfoDict, Metadata, Result};

/// Env id reported for a synthesized env, so a reader can tell a probe's
/// stand-in spaces from a real env's.
pub const SYNTHETIC_ENV_ID: &str = "rlmesh.probe.synthetic";

const DEFAULT_IMAGE_SIZE: u32 = 224;
const MAX_IMAGE_SIZE: u32 = 4096;
const DEFAULT_TEXT_LENGTH: i64 = 256;

#[derive(Debug, Clone)]
pub(super) struct SynthesizedEnv {
    pub(super) observation: WireSpec,
    pub(super) action: WireSpec,
    pub(super) tags: Option<Json>,
    pub(super) unsupported: Vec<String>,
    pub(super) render_requests: Vec<RenderRequest>,
}

pub(super) fn synthesize(model_spec: &Json) -> std::result::Result<SynthesizedEnv, String> {
    if let Some(error) = model_spec.get("error").and_then(Json::as_str) {
        return Err(format!(
            "the model's describe envelope carries no usable spec: {error}"
        ));
    }
    let spec: ModelSpec = serde_json::from_value(model_spec.clone()).map_err(|error| {
        format!("decode the model_spec block of the describe envelope: {error}")
    })?;

    let mut leaves = Vec::new();
    let mut unsupported = Vec::new();
    collect_leaves(&spec.input, &mut leaves, &mut unsupported);

    let mut spaces: Vec<(String, WireSpec)> = Vec::new();
    let mut tags: BTreeMap<String, ObsNode> = BTreeMap::new();
    for leaf in leaves {
        let key = match leaf.part() {
            Some(part) => feature_key(&format!("{}#{part}", leaf.role())),
            None => feature_key(leaf.role()),
        };
        if tags.contains_key(&key) {
            continue;
        }
        let Some((space, tag)) = observation_leaf(&leaf) else {
            unsupported.push(leaf.role().to_string());
            continue;
        };
        spaces.push((key.clone(), space));
        tags.insert(key, ObsNode::Leaf(tag));
    }
    if spaces.is_empty() {
        return Err(
            "the model declares no resolvable inputs to synthesize an observation from".into(),
        );
    }

    let observation = DictSpaceBuilder::new()
        .extend(spaces)
        .build()
        .map_err(|error| format!("build the synthesized observation space: {error}"))?;
    let action = action_space(&spec.output)
        .map_err(|error| format!("build the synthesized action space: {error}"))?;

    let tags = EnvTags {
        observation: ObsNode::Dict(tags),
        action: spec.output.clone(),
    };
    join(
        &tags,
        &SpaceView::from(&observation),
        &SpaceView::from(&action),
    )
    .map_err(|error| format!("synthesized env tags do not join their spaces: {error}"))?;

    Ok(SynthesizedEnv {
        observation,
        action,
        tags: Some(
            serde_json::to_value(&tags)
                .map_err(|error| format!("encode synthesized env tags: {error}"))?,
        ),
        unsupported,
        render_requests: render_requests(&spec)
            .into_iter()
            .map(|(role, (height, width))| RenderRequest {
                role,
                height,
                width,
            })
            .collect(),
    })
}

enum SynthLeaf {
    Image(Image),
    State(ConcatPart, State),
    Text(Text),
}

impl SynthLeaf {
    fn role(&self) -> &str {
        match self {
            Self::Image(image) => &image.role,
            Self::State(part, _) => part.role.as_deref().unwrap_or_default(),
            Self::Text(text) => &text.role,
        }
    }

    fn part(&self) -> Option<&str> {
        match self {
            Self::Image(image) => image.part.as_deref(),
            Self::State(part, _) => part.part.as_deref(),
            Self::Text(_) => None,
        }
    }
}

fn collect_leaves(node: &InputNode, out: &mut Vec<SynthLeaf>, unsupported: &mut Vec<String>) {
    match node {
        InputNode::Leaf(ModelLeaf::Image(image)) => out.push(SynthLeaf::Image(image.clone())),
        InputNode::Leaf(ModelLeaf::State(state)) => {
            for part in state.components.iter().filter(|part| part.role.is_some()) {
                out.push(SynthLeaf::State(part.clone(), state.clone()));
            }
        }
        InputNode::Leaf(ModelLeaf::Text(text)) => out.push(SynthLeaf::Text(text.clone())),
        InputNode::Leaf(ModelLeaf::Custom(_)) => unsupported.push("custom".to_string()),
        InputNode::Leaf(ModelLeaf::Unknown { kind, .. }) => unsupported.push(kind.clone()),
        InputNode::Dict(children) => {
            for child in children.values() {
                collect_leaves(child, out, unsupported);
            }
        }
        InputNode::Tuple(children) => {
            for child in children {
                collect_leaves(child, out, unsupported);
            }
        }
    }
}

fn observation_leaf(leaf: &SynthLeaf) -> Option<(WireSpec, ObsLeaf)> {
    match leaf {
        SynthLeaf::Image(image) => image_leaf(image),
        SynthLeaf::State(part, state) => {
            state_leaf(part, state_dim(part)?, &state.dtype, Some((-1.0, 1.0)))
        }
        SynthLeaf::Text(text) => text_leaf(text),
    }
}

fn image_leaf(image: &Image) -> Option<(WireSpec, ObsLeaf)> {
    let (height, width) = camera_size(image);
    let (height, width) = (i64::from(height), i64::from(width));
    let channels = i64::from(image.channels.unwrap_or(3).clamp(1, 4));
    let space = BoxSpaceBuilder::uint_scalar(0, 255, vec![height, width, channels])
        .dtype(DType::Uint8)
        .build()
        .ok()?;
    let tag = ObsLeaf::Image(ImageTag {
        role: image.role.clone(),
        layout: ImageLayout::Hwc,
        upside_down: false,
        part: image.part.clone(),
        unknown: BTreeMap::new(),
    });
    Some((space, tag))
}

fn state_leaf(
    part: &ConcatPart,
    dim: u32,
    dtype: &str,
    bounds: Option<(f64, f64)>,
) -> Option<(WireSpec, ObsLeaf)> {
    let shape = vec![i64::from(dim)];
    let builder = match bounds {
        Some((low, high)) => BoxSpaceBuilder::scalar(low, high, shape),
        None => BoxSpaceBuilder::unbounded(shape),
    };
    let space = builder.dtype(dtype_of(dtype)).build().ok()?;
    let tag = ObsLeaf::State(StateTag {
        role: part.role.clone()?,
        encoding: part.encoding.as_ref().map(|encoding| encoding.accept_set()),
        range: None,
        frame: part.frame.clone(),
        provenance: part
            .provenance
            .as_ref()
            .and_then(|accepted| accepted.first_known())
            .map(|provenance| FrameRef::from(provenance.as_str())),
        part: part.part.clone(),
        labels: part.labels.clone(),
        polarity: part.polarity,
        unknown: BTreeMap::new(),
    });
    Some((space, tag))
}

fn text_leaf(text: &Text) -> Option<(WireSpec, ObsLeaf)> {
    let space = TextBuilder::new(DEFAULT_TEXT_LENGTH).build().ok()?;
    let tag = ObsLeaf::Text(TextTag {
        role: text.role.clone(),
        unknown: BTreeMap::new(),
    });
    Some((space, tag))
}

fn camera_size(image: &Image) -> (u32, u32) {
    let (height, width) = match image.render {
        Some((height, width)) => (height, width),
        None => (
            image.height.unwrap_or(DEFAULT_IMAGE_SIZE),
            image.width.unwrap_or(DEFAULT_IMAGE_SIZE),
        ),
    };
    let headroom = crop_headroom(image);
    let axis = |size: u32| {
        let scaled = (f64::from(size.max(1)) * headroom).ceil();
        (scaled as u32).clamp(1, MAX_IMAGE_SIZE)
    };
    (axis(height), axis(width))
}

fn crop_headroom(image: &Image) -> f64 {
    let fraction = match (image.crop, image.crop_area) {
        (Some(side), _) => side,
        (None, Some(area)) => area.sqrt(),
        (None, None) => return 1.0,
    };
    if fraction <= 0.0 { 1.0 } else { 1.0 / fraction }
}

fn state_dim(part: &ConcatPart) -> Option<u32> {
    if let Some(dim) = part.dim {
        return Some(dim.max(1));
    }
    if let Some(def) = part.role.as_deref().and_then(role_def)
        && let DimLaw::Fixed(dim) = def.dim
    {
        return Some(dim);
    }
    part.encoding
        .as_ref()
        .and_then(|encoding| encoding.base())
        .map(|base| base.dims())
}

fn dtype_of(name: &str) -> DType {
    match name {
        "float64" => DType::Float64,
        "int32" => DType::Int32,
        "int64" => DType::Int64,
        "uint8" => DType::Uint8,
        _ => DType::Float32,
    }
}

fn feature_key(role: &str) -> String {
    let key: String = role
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect();
    if key.is_empty() {
        "input".to_string()
    } else {
        key
    }
}

fn action_space(action: &Action) -> std::result::Result<WireSpec, String> {
    let width: usize = action
        .components
        .iter()
        .map(|actuator| actuator.dim as usize)
        .sum();
    if width == 0 {
        return Err("the model declares a zero-width action".into());
    }
    let mut low = Vec::with_capacity(width);
    let mut high = Vec::with_capacity(width);
    for actuator in &action.components {
        let (lo, hi) = actuator.range.unwrap_or((-1.0, 1.0));
        for _ in 0..actuator.dim {
            low.push(lo);
            high.push(hi);
        }
    }
    BoxSpaceBuilder::tensor(low, high, vec![width as i64])
        .dtype(DType::Float32)
        .build()
        .map_err(|error| error.to_string())
}

#[derive(Debug, Default)]
pub(super) struct Script {
    pub(super) observations: Vec<u64>,
    pub(super) actions: Vec<SpaceValue>,
}

pub(super) struct StandIn {
    observation: Space,
    action: Space,
    metadata: Option<Metadata>,
    script: Arc<Mutex<Script>>,
    tick: usize,
}

impl StandIn {
    pub(super) fn new(env: &SynthesizedEnv, script: Arc<Mutex<Script>>) -> Self {
        let metadata = env.tags.as_ref().map(|tags| {
            let mut metadata = Metadata::new();
            metadata.insert(
                rlmesh_adapters::v1::ENV_METADATA_KEY,
                crate::specs::json_to_value(tags.clone()),
            );
            metadata
        });
        Self {
            observation: Space::new(SpaceSpec::from_wire(env.observation.clone())),
            action: Space::new(SpaceSpec::from_wire(env.action.clone())),
            metadata,
            script,
            tick: 0,
        }
    }

    fn script(&self) -> MutexGuard<'_, Script> {
        self.script.lock().unwrap_or_else(|err| err.into_inner())
    }

    fn observation_at(&self, tick: usize) -> Result<crate::Value> {
        let seed = {
            let script = self.script();
            let last = script.observations.len().saturating_sub(1);
            script
                .observations
                .get(tick.min(last))
                .copied()
                .unwrap_or_default()
        };
        let wire = self.observation.spec().wire();
        let sampled = rlmesh_spaces::sample_seeded(wire, seed)
            .map_err(|err| crate::Error::invalid_argument(err.to_string()))?;
        space_codec::value_from_space(wire, sampled)
    }
}

impl Env for StandIn {
    fn observation_space(&self) -> Space {
        self.observation.clone()
    }

    fn action_space(&self) -> Space {
        self.action.clone()
    }

    fn reset(
        &mut self,
        _seed: Option<i64>,
        _options: Option<&InfoDict>,
    ) -> Result<(crate::Value, InfoDict)> {
        self.tick = 0;
        self.script().actions.clear();
        Ok((self.observation_at(0)?, InfoDict::new()))
    }

    fn step(
        &mut self,
        action: Option<crate::Value>,
    ) -> Result<(crate::Value, f64, bool, bool, InfoDict)> {
        if let Some(action) = action {
            let action = space_codec::value_to_space(self.action.spec().wire(), &action)?;
            self.script().actions.push(action);
        }
        self.tick += 1;
        Ok((
            self.observation_at(self.tick)?,
            0.0,
            false,
            false,
            InfoDict::new(),
        ))
    }

    fn metadata(&self) -> Option<Metadata> {
        self.metadata.clone()
    }

    fn id(&self) -> Option<String> {
        Some(SYNTHETIC_ENV_ID.to_string())
    }
}

pub(super) fn serve_beside(model_address: &str, env: StandIn) -> Result<crate::EnvServer> {
    let base = model_address
        .rsplit_once('/')
        .map_or(model_address, |(base, _)| base);
    let run = uuid::Uuid::new_v4().simple().to_string();
    let server = crate::EnvServer::new(
        env,
        crate::EnvServerOptions {
            address: Some(format!("{base}/run-{}/env", &run[..12])),
            ..Default::default()
        },
    )?;
    server.start()?;
    Ok(server)
}

#[cfg(test)]
mod tests {
    use rlmesh_spaces::spaces::SpaceKind;

    use super::*;

    fn smolvla_spec() -> Json {
        serde_json::json!({
            "input": {
                "image": {"type": "image", "role": "image/primary", "height": 256, "width": 256,
                          "layout": "chw", "dtype": "float32", "normalize": true},
                "image2": {"type": "image", "role": "image/wrist", "height": 256, "width": 256,
                           "layout": "chw", "dtype": "float32", "normalize": true},
                "state": {"type": "state", "components": [
                    {"role": "proprio/eef_pos"},
                    {"role": "proprio/eef_rot", "encoding": "axis_angle"},
                    {"role": "proprio/gripper", "dim": 1}
                ]},
                "instruction": {"type": "text", "role": "instruction"}
            },
            "output": {"components": [
                {"role": "action/delta_eef_pos", "dim": 3},
                {"role": "action/delta_eef_rot", "dim": 3, "encoding": "axis_angle"},
                {"role": "action/gripper", "dim": 1}
            ]}
        })
    }

    fn dict_keys(space: &WireSpec) -> Vec<String> {
        match &space.spec {
            Some(SpaceKind::Dict(dict)) => dict.keys.clone(),
            other => panic!("expected a dict observation, got {other:?}"),
        }
    }

    fn image_input(fields: Json) -> Image {
        let mut input = serde_json::json!({"type": "image", "role": "image/primary"});
        input
            .as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        serde_json::from_value(input).expect("image input")
    }

    #[test]
    fn synthesizes_one_observation_leaf_per_model_input_role() {
        let env = synthesize(&smolvla_spec()).expect("synthesize");
        assert_eq!(
            dict_keys(&env.observation),
            [
                "image_primary",
                "image_wrist",
                "instruction",
                "proprio_eef_pos",
                "proprio_eef_rot",
                "proprio_gripper",
            ]
        );
        assert!(env.unsupported.is_empty());
        assert!(env.tags.as_ref().is_some_and(Json::is_object));
        assert_eq!(env.action.shape, [7]);
    }

    #[test]
    fn a_bimanual_models_parts_are_separate_leaves() {
        let spec = serde_json::json!({
            "input": {
                "left": {"type": "image", "role": "image/wrist", "part": "left_arm",
                         "height": 224, "width": 224},
                "right": {"type": "image", "role": "image/wrist", "part": "right_arm",
                          "height": 224, "width": 224},
                "state": {"type": "state", "components": [
                    {"role": "proprio/gripper", "part": "left_arm", "dim": 1},
                    {"role": "proprio/gripper", "part": "right_arm", "dim": 1}
                ]}
            },
            "output": {"components": [{"role": "action/gripper", "dim": 1}]}
        });
        let env = synthesize(&spec).expect("synthesize");
        assert_eq!(
            dict_keys(&env.observation),
            [
                "image_wrist_left_arm",
                "image_wrist_right_arm",
                "proprio_gripper_left_arm",
                "proprio_gripper_right_arm",
            ]
        );
    }

    #[test]
    fn the_stand_in_camera_honours_render_and_crop_headroom() {
        let cases = [
            (serde_json::json!({"height": 224, "width": 224}), (224, 224)),
            (
                serde_json::json!({"height": 224, "width": 224, "render": [256, 320]}),
                (256, 320),
            ),
            (
                serde_json::json!({"height": 224, "width": 224, "crop_area": 0.9}),
                (237, 237),
            ),
            (
                serde_json::json!({"height": 224, "width": 224, "crop": 0.5}),
                (448, 448),
            ),
            (
                serde_json::json!({}),
                (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
            ),
        ];
        for (fields, expected) in cases {
            assert_eq!(
                camera_size(&image_input(fields.clone())),
                expected,
                "{fields}"
            );
        }
    }

    #[test]
    fn state_widths_come_from_the_declaration_then_the_role_registry() {
        let env = synthesize(&smolvla_spec()).expect("synthesize");
        let SpaceKind::Dict(dict) = env.observation.spec.as_ref().expect("dict") else {
            panic!("expected dict")
        };
        let width = |key: &str| -> Vec<i64> {
            let index = dict.keys.iter().position(|k| k == key).expect("key");
            dict.spaces[index].shape.clone()
        };
        assert_eq!(width("proprio_eef_pos"), [3]);
        assert_eq!(width("proprio_eef_rot"), [3]);
        assert_eq!(width("proprio_gripper"), [1]);
    }

    #[test]
    fn custom_and_unknown_inputs_are_reported_not_guessed() {
        let spec = serde_json::json!({
            "input": {
                "state": {"type": "state", "components": [{"role": "proprio/eef_pos"}]},
                "weird": {"type": "audio", "role": "mic", "sample_rate": 16000}
            },
            "output": {"components": [{"role": "action/gripper", "dim": 1}]}
        });
        let env = synthesize(&spec).expect("synthesize");
        assert_eq!(env.unsupported, ["audio"]);
        assert_eq!(dict_keys(&env.observation), ["proprio_eef_pos"]);
    }

    #[test]
    fn an_unusable_describe_block_fails_with_a_legible_reason() {
        let error = synthesize(&serde_json::json!({"error": "spec is not publishable"}))
            .expect_err("must fail");
        assert!(error.contains("not publishable"), "{error}");

        let error = synthesize(&Json::Null).expect_err("must fail");
        assert!(error.contains("model_spec"), "{error}");

        let no_inputs = serde_json::json!({
            "input": {"weird": {"type": "audio", "role": "mic"}},
            "output": {"components": [{"role": "action/gripper", "dim": 1}]}
        });
        let error = synthesize(&no_inputs).expect_err("must fail");
        assert!(error.contains("no resolvable inputs"), "{error}");
    }
}
