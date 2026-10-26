//! A model probe's stand-in environment, built from the model's own spec.
//!
//! A probe exercises one container. For a model that means observations to
//! predict on, which normally come from a paired env -- but a probe must not
//! need one (its results say nothing about the env, and a model with no
//! compatible env is exactly the one worth probing). The model's `ModelSpec`
//! already declares every input it wants, by role, so this module turns that
//! declaration into the spaces an env *would* offer: one observation leaf per
//! model input, tagged with the role it asks for, plus an action space the
//! width of its declared output. The binder then resolves the model against
//! the stand-in exactly as it would against a real env.
//!
//! Ported from the managed probe (`probe/synth.rs`, and the stand-in leaves of
//! `rlmesh-transfer`'s `stand_in.rs`); [`StandIn`] is new: the 0.2 binder
//! drives an [`Env`], so the stand-in serves scripted observations and keeps
//! the actions it receives.

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

/// Pixel size for an image input that declares none. Large enough to survive a
/// model's own resize, small enough to keep payloads cheap.
const DEFAULT_IMAGE_SIZE: u32 = 224;
/// Upper bound on a synthesized camera axis, matching the spec's own
/// `render`/`height`/`width` bound.
const MAX_IMAGE_SIZE: u32 = 4096;
const DEFAULT_TEXT_LENGTH: i64 = 256;

/// The spaces a synthetic env offers, plus the tags that let the model's
/// adapter resolve against them.
#[derive(Debug, Clone)]
pub(super) struct SynthesizedEnv {
    pub(super) observation: WireSpec,
    pub(super) action: WireSpec,
    /// The env tags the stand-in publishes, as JSON; `None` passes a
    /// reference env's contract through untagged.
    pub(super) tags: Option<Json>,
    /// Model inputs that carry no resolvable role (custom transforms, kinds
    /// this runtime does not know); the probe reports them on `resolve`.
    pub(super) unsupported: Vec<String>,
    /// The camera resolutions the model's image inputs assert, in spec document
    /// order. Read off the spec, not off a resolve, so a model whose adapter
    /// fails to resolve still says what to set the env's camera dial to.
    pub(super) render_requests: Vec<RenderRequest>,
}

/// Builds a stand-in env from the model's `model_spec` describe block.
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
        // One env feature per (role, part): a model may read a role from several
        // input slots (one camera feeding two), but an env publishes it once, and
        // two features sharing a role and part is a hard Duplicate error at
        // resolve. The part stays in the key: a bimanual model's left and right
        // wrist cameras are two features, not one.
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
    // The same join the resolver runs env-side: a synthesized tree that cannot
    // join its own spaces would fail later inside the bind with a much worse
    // message.
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

/// A model input leaf that maps to one observation leaf.
enum SynthLeaf {
    Image(Image),
    /// One component of a state input: each pulls its own env role, so a
    /// concat of three roles becomes three observation leaves.
    State(ConcatPart, State),
    Text(Text),
}

impl SynthLeaf {
    fn role(&self) -> &str {
        match self {
            Self::Image(image) => &image.role,
            // `collect_leaves` never builds a State leaf for a role-less
            // (constant) part.
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
            // A role-less part is a declared constant: it contributes a fixed
            // value, reads no env feature, and so needs no synthesized leaf.
            for part in state.components.iter().filter(|part| part.role.is_some()) {
                out.push(SynthLeaf::State(part.clone(), state.clone()));
            }
        }
        InputNode::Leaf(ModelLeaf::Text(text)) => out.push(SynthLeaf::Text(text.clone())),
        // A custom transform reads whatever it likes from the env; there is no
        // declared role to synthesize, so the probe reports it rather than
        // guessing a shape the model would reject.
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
        // A probe's stand-in values are its own to choose: bounded, so a
        // model declaring a range exercises its remap.
        SynthLeaf::State(part, state) => {
            state_leaf(part, state_dim(part)?, &state.dtype, Some((-1.0, 1.0)))
        }
        SynthLeaf::Text(text) => text_leaf(text),
    }
}

/// The camera a model image input asks for.
fn image_leaf(image: &Image) -> Option<(WireSpec, ObsLeaf)> {
    let (height, width) = camera_size(image);
    let (height, width) = (i64::from(height), i64::from(width));
    // Envs publish frames as HWC uint8; the model's own layout and dtype are
    // its adapter's job, not the env's. The channel count is not: the resolver
    // refuses to convert between channel counts, so a model asking for
    // grayscale must be offered grayscale.
    let channels = i64::from(image.channels.unwrap_or(3).clamp(1, 4));
    let space = BoxSpaceBuilder::uint_scalar(0, 255, vec![height, width, channels])
        .dtype(DType::Uint8)
        .build()
        .ok()?;
    let tag = ObsLeaf::Image(ImageTag {
        role: image.role.clone(),
        layout: ImageLayout::Hwc,
        upside_down: false,
        // A model naming a body part binds only a leaf under that part, so the
        // stand-in publishes the part it declared.
        part: image.part.clone(),
        unknown: BTreeMap::new(),
    });
    Some((space, tag))
}

/// The state feature one component of a model state input reads, `dim` wide
/// in the input's `dtype`, its space bounded by `bounds` or unbounded.
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
    // The tag carries no width: the resolver derives it from the space this
    // leaf publishes, which is built from the same `dim`.
    let tag = ObsLeaf::State(StateTag {
        role: part.role.clone()?,
        encoding: part.encoding.as_ref().map(|encoding| encoding.accept_set()),
        range: None,
        // The stand-in speaks the frame the model declared it was trained in,
        // so it never reports a frame disagreement against spaces the model
        // itself described.
        frame: part.frame.clone(),
        // Likewise the provenance (the first the model accepts), body part,
        // axis labels and polarity: a labeled model against an unlabeled env
        // is a resolve error, and a stand-in is not a resolver test.
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

/// The text a model text input reads.
fn text_leaf(text: &Text) -> Option<(WireSpec, ObsLeaf)> {
    let space = TextBuilder::new(DEFAULT_TEXT_LENGTH).build().ok()?;
    let tag = ObsLeaf::Text(TextTag {
        role: text.role.clone(),
        unknown: BTreeMap::new(),
    });
    Some((space, tag))
}

/// The resolution the stand-in camera renders at.
///
/// A `render` assertion is authoritative: the model says what its camera was
/// trained at, and resolution *checks* that assertion, so a stand-in that
/// ignored it would fail every model that declares one. Absent it, the
/// declared input size, then a square default.
///
/// A crop keeps a `fraction` of each side, and the resolver's upscale guard
/// measures that box rather than the whole frame, in either crop mode. So the
/// camera must be large enough that the box still covers the target, hence
/// `ceil(size / fraction)` of headroom. Both axes are clamped to
/// `1..=MAX_IMAGE_SIZE`.
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

/// `1 / side fraction` for a cropping input, else 1.
fn crop_headroom(image: &Image) -> f64 {
    let fraction = match (image.crop, image.crop_area) {
        (Some(side), _) => side,
        (None, Some(area)) => area.sqrt(),
        (None, None) => return 1.0,
    };
    if fraction <= 0.0 { 1.0 } else { 1.0 / fraction }
}

/// The declared dim, else the role registry's fixed width, else the rotation
/// encoding's. `None` for a role whose width is the embodiment's business
/// (joint counts, gripper fingers) and that the model did not declare:
/// guessing a scalar there would hand the model a 1-wide joint vector and
/// surface as a shape failure of the model rather than of the stand-in.
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

/// A model input's dtype name as a space dtype; float32 for anything else.
fn dtype_of(name: &str) -> DType {
    match name {
        "float64" => DType::Float64,
        "int32" => DType::Int32,
        "int64" => DType::Int64,
        "uint8" => DType::Uint8,
        _ => DType::Float32,
    }
}

/// One observation key per role, sanitized for a dict key.
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

/// The action space an env would publish for this output: element-wise bounds
/// from each actuator's declared range, so the tags' ranges and the space agree
/// (a fabricated uniform clip would be reconciled against them and lose).
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

/// What the stand-in serves and what it has received, shared between the
/// probe and the env the binding drives.
#[derive(Debug, Default)]
pub(super) struct Script {
    /// The observation seed of each tick: the reset's first, then one per
    /// step; the last repeats once the script runs out.
    pub(super) observations: Vec<u64>,
    /// The actions stepped in the current episode, in order.
    pub(super) actions: Vec<SpaceValue>,
}

/// A stand-in env over synthesized spaces: it serves the observation the
/// script names for each tick and records every action, so a check compares
/// the model's answers to inputs it chose.
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
            // The binder reads the tags from this key as a mapping, as a live
            // env publishes them.
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

/// Serve `env` in a fresh session on the host of the model at
/// `model_address`, as the Python sandbox serves a local env for a served
/// model: a model in a container reaches its own session, not one this
/// process hosts.
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

    /// The SmolVLA libero spec: two cameras, a three-role state concat, an
    /// instruction, and a 7-wide action.
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
        // The resolver rejects a JSON string under the tags key: the tags
        // must be the mapping a live env sends.
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

    /// A `render` assertion wins over the input size; a crop gets headroom so
    /// the cropped box still covers the target, in either crop mode.
    #[test]
    fn the_stand_in_camera_honours_render_and_crop_headroom() {
        let cases = [
            (serde_json::json!({"height": 224, "width": 224}), (224, 224)),
            (
                serde_json::json!({"height": 224, "width": 224, "render": [256, 320]}),
                (256, 320),
            ),
            // OpenVLA's libero input: a 0.9-area crop resized to 224; a 224
            // camera would crop to 213 and trip the upscale guard.
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
        // eef_pos is DimLaw::Fixed(3), eef_rot comes from axis_angle, gripper
        // is declared dim=1.
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
