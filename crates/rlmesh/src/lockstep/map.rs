//! A Gym env as a robot (§12.5): its spaces and v1 tags become a robot map,
//! and its observation and action leaves become part slabs.
//!
//! Every leaf of the observation space lands in exactly one part, so the
//! v1 bytes of an observation are recoverable bit for bit from the parts:
//!
//! | Gym leaf | Part |
//! |---|---|
//! | `Dict` key nesting, `Tuple` item i | path segments ([`escape`](mod@crate::lockstep::escape)) |
//! | untagged `Box` | `gym/box`, field `value` |
//! | `Discrete`, `MultiDiscrete`, `MultiBinary` | `gym/discrete`, `gym/multi_discrete`, `gym/multi_binary`, field `value` |
//! | `Text` | `text` (per step, as v1) |
//! | `image/*` tag on a `uint8 [H, W, 3]` HWC `Box` | `camera` at `scene/cam`, `scene/cam_side`, `wrist/cam` |
//! | `proprio/joint_pos`, `joint_vel`, `x/joint_torque` | one `joints` part `arm`, fields `pos`, `vel`, `effort` |
//! | `proprio/gripper` | `gripper` (one value) or `joints` `gripper/fingers` (two) |
//! | `proprio/eef_wrench` with a `frame` | `wrench` at `wrist/ft` |
//! | `text/instruction` | `task` (text) |
//!
//! A tag `part` is a path prefix (`left_arm` → `left/arm`). A tagged leaf
//! whose role has no row here, or whose leaf does not fit the kind's
//! layout (a CHW image, a pose without a child frame, a split leaf), keeps
//! the custom kind of its space at its key path, with the tag in
//! `env_tags`: it travels bit-exact, only without a core kind.
//!
//! Actions keep the `action` part (`gym/box`, `command: "value"`; one part
//! per leaf below `action/` for a `Dict` or `Tuple` action space), so the
//! wrapper never has to reassemble or fill a flat vector: §12.5's "or
//! `action`" branch. Per-component action parts are deferred.
//!
//! Each part carries its original space (`gym_space`), keys (`gym_key`)
//! and tag (`env_tags`) in `attrs`; the robot's first part carries the
//! whole v1 contract and handshake (`gym_env`), from which the driver
//! rebuilds the exact spaces and this same map.

use std::collections::BTreeMap;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use bytes::Bytes;
use prost::Message as _;
use rlmesh_adapters::v1::{EnvTags, ObsLeaf, ObsNode, space_spec_to_json};
use rlmesh_session::wire::{ChannelId, DType, Layout, PartSpec, ParticipantId};
use rlmesh_spaces::{EnvContract, SpaceKind, SpaceSpec};
use serde_json::{Value, json};

use crate::lockstep::error::GymError;
use crate::lockstep::escape::{self, KeySeg};

/// What a driver needs to drive a Gym env it cannot handshake with: the v1
/// contract and the handshake answers, carried in `gym_env`.
#[derive(Clone, Debug, PartialEq)]
pub struct GymEnvInfo {
    /// The v1 env contract (spaces, metadata and tags, `num_envs`,
    /// autoreset).
    pub contract: EnvContract,
    /// The workflow edition the env runs.
    pub workflow_edition: String,
    /// Every edition it implements.
    pub supported_workflow_editions: Vec<String>,
    /// Its handshake capabilities (`subset_step`).
    pub capabilities: Vec<String>,
}

impl GymEnvInfo {
    /// Whether lanes step and reset individually (`subset_step`).
    pub fn subset_step(&self) -> bool {
        self.capabilities
            .iter()
            .any(|c| c == rlmesh_proto::capabilities::ENV_SUBSET_STEP)
    }

    fn to_json(&self) -> Value {
        let contract = rlmesh_grpc::wire::env_contract_to_proto(&self.contract).encode_to_vec();
        json!({
            "contract": STANDARD.encode(contract),
            "workflow_edition": self.workflow_edition,
            "supported_workflow_editions": self.supported_workflow_editions,
            "capabilities": self.capabilities,
        })
    }

    /// Parses a `gym_env` attribute.
    pub fn from_attr(text: &str) -> Result<Self, GymError> {
        let bad = |why: String| GymError::Map(format!("gym_env: {why}"));
        let value: Value = serde_json::from_str(text).map_err(|e| bad(e.to_string()))?;
        let field = |name: &str| value.get(name).ok_or_else(|| bad(format!("no {name}")));
        let strings = |v: &Value| -> Result<Vec<String>, GymError> {
            v.as_array()
                .ok_or_else(|| bad("not a list".into()))?
                .iter()
                .map(|s| {
                    s.as_str()
                        .map(str::to_string)
                        .ok_or_else(|| bad("not a string".into()))
                })
                .collect()
        };
        let contract = STANDARD
            .decode(
                field("contract")?
                    .as_str()
                    .ok_or_else(|| bad("contract".into()))?,
            )
            .map_err(|e| bad(e.to_string()))?;
        let contract = rlmesh_proto::core::v1::EnvContract::decode(contract.as_slice())
            .map_err(|e| bad(e.to_string()))?;
        Ok(Self {
            contract: rlmesh_grpc::wire::env_contract_from_proto(contract)
                .map_err(|e| bad(e.to_string()))?,
            workflow_edition: field("workflow_edition")?
                .as_str()
                .unwrap_or_default()
                .to_string(),
            supported_workflow_editions: strings(field("supported_workflow_editions")?)?,
            capabilities: strings(field("capabilities")?)?,
        })
    }
}

/// One fundamental leaf of a Gym space.
#[derive(Clone, Debug)]
pub struct Leaf {
    /// Its keys from the root.
    pub keys: Vec<KeySeg>,
    /// Its space.
    pub spec: SpaceSpec,
    /// Bytes per lane (`None`: text, variable).
    pub lane_bytes: Option<usize>,
}

/// Where a part field's bytes come from: a byte range of one leaf's lane.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldSource {
    /// The leaf.
    pub leaf: usize,
    /// Byte offset in the leaf's lane.
    pub offset: usize,
    /// Byte length.
    pub len: usize,
}

/// One part of a Gym robot.
#[derive(Clone, Debug)]
pub struct PartMap {
    /// The part as normalised (channel 0 until a host assigns one).
    pub spec: PartSpec,
    /// Its packed layout.
    pub layout: Layout,
    /// One source per field, in payload order.
    pub sources: Vec<FieldSource>,
    /// The slab is the leaf's own bytes (one whole leaf, 8-aligned lanes):
    /// no copy either way.
    pub zero_copy: bool,
}

/// The observation or the action side of a Gym robot.
#[derive(Clone, Debug, Default)]
pub struct Side {
    /// The space's leaves, in canonical (pre-) order.
    pub leaves: Vec<Leaf>,
    /// Its parts, in declaration order.
    pub parts: Vec<PartMap>,
}

/// A Gym env as a robot.
#[derive(Clone, Debug)]
pub struct GymMap {
    /// The authored robot map (§3), JSON in declaration order.
    pub robot_json: String,
    /// Observation leaves and parts.
    pub observation: Side,
    /// Action leaves and parts.
    pub action: Side,
    /// The contract and handshake.
    pub info: GymEnvInfo,
}

/// Authored keys of one part, in order, each with its JSON text.
type DeclKeys = Vec<(&'static str, String)>;

/// A part being assembled: its authored keys and its fields.
struct Draft {
    path: String,
    /// Authored keys in order, as JSON text.
    keys: DeclKeys,
    /// (field name, source), in any order: matched to the normalised fields.
    fields: Vec<(String, FieldSource)>,
    /// Leaves this part reads (for `gym_key` / `gym_space` / `env_tags`).
    leaves: Vec<usize>,
    /// The first leaf's keys, for messages.
    origin: String,
    tags: Vec<Value>,
    /// Joints merging: count and dtype.
    joints: Option<(usize, DType)>,
}

fn bad(message: impl Into<String>) -> GymError {
    GymError::Map(message.into())
}

fn json_text(value: &Value) -> String {
    value.to_string()
}

/// `[dtype, shape]`-ready names and shapes.
fn wire_dtype(dtype: rlmesh_spaces::DType) -> Result<DType, GymError> {
    DType::from_code(dtype as u32).ok_or_else(|| bad(format!("dtype {dtype:?} has no v2 code")))
}

fn shape_u32(shape: &[i64]) -> Result<Vec<u32>, GymError> {
    shape
        .iter()
        .map(|d| u32::try_from(*d).map_err(|_| bad(format!("shape {shape:?}"))))
        .collect()
}

/// Bytes per lane of a leaf, as v1 packs it (`None` for text).
fn lane_bytes(spec: &SpaceSpec) -> Result<Option<usize>, GymError> {
    let numel = |shape: &[i64]| -> Result<usize, GymError> {
        shape.iter().try_fold(1_usize, |n, d| {
            usize::try_from(*d)
                .ok()
                .and_then(|d| n.checked_mul(d))
                .ok_or_else(|| bad(format!("shape {shape:?}")))
        })
    };
    let size = rlmesh_spaces::dtype_size(spec.dtype);
    Ok(Some(match &spec.spec {
        Some(SpaceKind::Box(_)) => numel(&spec.shape)? * size,
        Some(SpaceKind::Discrete(_)) => size,
        Some(SpaceKind::MultiBinary(_)) => numel(&spec.shape)?,
        Some(SpaceKind::MultiDiscrete(m)) => m.nvec.len() * size,
        Some(SpaceKind::Text(_)) => return Ok(None),
        other => return Err(bad(format!("no leaf layout for {other:?}"))),
    }))
}

fn walk(spec: &SpaceSpec, keys: &mut Vec<KeySeg>, out: &mut Vec<Leaf>) -> Result<(), GymError> {
    match &spec.spec {
        Some(SpaceKind::Dict(d)) => {
            for (key, child) in d.keys.iter().zip(&d.spaces) {
                keys.push(KeySeg::Key(key.clone()));
                walk(child, keys, out)?;
                keys.pop();
            }
        }
        Some(SpaceKind::Tuple(t)) => {
            for (i, child) in t.spaces.iter().enumerate() {
                keys.push(KeySeg::Item(i));
                walk(child, keys, out)?;
                keys.pop();
            }
        }
        _ => out.push(Leaf {
            keys: keys.clone(),
            spec: spec.clone(),
            lane_bytes: lane_bytes(spec)?,
        }),
    }
    Ok(())
}

/// The leaves of a space in canonical order.
pub fn leaves(spec: &SpaceSpec) -> Result<Vec<Leaf>, GymError> {
    let mut out = Vec::new();
    walk(spec, &mut Vec::new(), &mut out)?;
    Ok(out)
}

/// The tag over `keys`, if the tag tree has one.
fn tag_at<'t>(node: &'t ObsNode, keys: &[KeySeg]) -> Option<&'t ObsLeaf> {
    match (node, keys.split_first()) {
        (ObsNode::Leaf(leaf), None) => Some(leaf),
        (ObsNode::Dict(map), Some((KeySeg::Key(k), rest))) => tag_at(map.get(k)?, rest),
        (ObsNode::Tuple(items), Some((KeySeg::Item(i), rest))) => tag_at(items.get(*i)?, rest),
        _ => None,
    }
}

/// The v1 tags a contract publishes (`None`: untagged, or unreadable).
pub fn contract_tags(contract: &EnvContract) -> Option<EnvTags> {
    let value = contract
        .metadata
        .as_ref()?
        .get(rlmesh_adapters::v1::ENV_METADATA_KEY)?;
    serde_json::from_value(native_json(value)).ok()
}

/// A native metadata value as plain JSON.
fn native_json(value: &rlmesh_spaces::MetaValue) -> Value {
    use rlmesh_spaces::MetaValue as M;
    match value {
        M::Null => Value::Null,
        M::Bool(b) => Value::Bool(*b),
        M::Int(i) => json!(i),
        M::Float(x) => json!(x),
        M::String(s) => Value::String(s.clone()),
        M::Bytes(b) => json!(b),
        M::List(items) => Value::Array(items.iter().map(native_json).collect()),
        M::Map(map) => Value::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), native_json(v)))
                .collect(),
        ),
    }
}

/// `left_arm` → `left/arm/`.
fn part_prefix(part: Option<&str>) -> String {
    match part {
        Some(part) if !part.is_empty() => {
            let segs: Vec<String> = part
                .split('_')
                .filter(|s| !s.is_empty())
                .map(|s| escape::segment(&KeySeg::Key(s.into())))
                .collect();
            format!("{}/", segs.join("/"))
        }
        _ => String::new(),
    }
}

/// The custom (or text) kind of an untagged leaf, its authored keys and
/// its one field.
fn custom_kind(leaf: &Leaf, hz: f64) -> Result<(String, DeclKeys), GymError> {
    let dtype = wire_dtype(leaf.spec.dtype)?;
    let (kind, dtype, shape) = match &leaf.spec.spec {
        Some(SpaceKind::Box(_)) => ("gym/box", dtype, shape_u32(&leaf.spec.shape)?),
        Some(SpaceKind::Discrete(_)) => ("gym/discrete", dtype, Vec::new()),
        Some(SpaceKind::MultiDiscrete(m)) => (
            "gym/multi_discrete",
            dtype,
            if leaf.spec.shape.is_empty() {
                vec![u32::try_from(m.nvec.len()).map_err(|_| bad("nvec"))?]
            } else {
                shape_u32(&leaf.spec.shape)?
            },
        ),
        // One byte per element on the v1 wire, whatever the dtype.
        Some(SpaceKind::MultiBinary(_)) => {
            ("gym/multi_binary", DType::I8, shape_u32(&leaf.spec.shape)?)
        }
        Some(SpaceKind::Text(_)) => {
            return Ok(("text".into(), vec![("kind", json_text(&json!("text")))]));
        }
        other => return Err(bad(format!("no kind for {other:?}"))),
    };
    let fields = format!(
        "{{\"value\":[{:?},{}]}}",
        dtype.name(),
        json_text(&json!(shape))
    );
    Ok((
        kind.into(),
        vec![
            ("kind", json_text(&json!(kind))),
            ("hz", json_text(&json!(hz))),
            ("fields", fields),
        ],
    ))
}

impl GymMap {
    /// The robot map of a Gym env with `info`'s contract, at a notional
    /// rate of `hz` (lockstep never paces: §11.2).
    pub fn new(info: GymEnvInfo, hz: f64) -> Result<Self, GymError> {
        let obs_space = info
            .contract
            .observation_space
            .as_ref()
            .ok_or_else(|| bad("the env has no observation space"))?;
        let act_space = info
            .contract
            .action_space
            .as_ref()
            .ok_or_else(|| bad("the env has no action space"))?;
        let obs_leaves = leaves(obs_space)?;
        let act_leaves = leaves(act_space)?;
        let tags = contract_tags(&info.contract);

        let mut drafts: Vec<Draft> = Vec::new();
        let mut by_path: BTreeMap<String, usize> = BTreeMap::new();
        for (i, leaf) in obs_leaves.iter().enumerate() {
            let tag = tags
                .as_ref()
                .and_then(|t| tag_at(&t.observation, &leaf.keys));
            if let Some(tag) = tag
                && Self::core_part(&mut drafts, &mut by_path, i, leaf, tag, hz)?
            {
                continue;
            }
            let path = if leaf.keys.is_empty() {
                "obs".to_string()
            } else {
                escape::path(&[], &leaf.keys)
            };
            let (kind, keys) = custom_kind(leaf, hz)?;
            let field = if kind == "text" { "text" } else { "value" };
            Self::push(
                &mut drafts,
                &mut by_path,
                Draft {
                    path,
                    keys,
                    fields: vec![(
                        field.into(),
                        FieldSource {
                            leaf: i,
                            offset: 0,
                            len: leaf.lane_bytes.unwrap_or(0),
                        },
                    )],
                    leaves: vec![i],
                    origin: escape::key_text(&leaf.keys),
                    tags: tag
                        .map(|t| serde_json::to_value(t).unwrap_or(Value::Null))
                        .into_iter()
                        .collect(),
                    joints: None,
                },
            )?;
        }
        let obs_count = drafts.len();
        for (i, leaf) in act_leaves.iter().enumerate() {
            let path = escape::path(&["action"], &leaf.keys);
            let (kind, mut keys) = custom_kind(leaf, hz)?;
            if kind == "text" {
                return Err(bad(format!(
                    "action leaf {} is Text, which a part cannot be commanded with",
                    escape::key_text(&leaf.keys)
                )));
            }
            keys.push(("command", json_text(&json!("value"))));
            Self::push(
                &mut drafts,
                &mut by_path,
                Draft {
                    path,
                    keys,
                    fields: vec![(
                        "value".into(),
                        FieldSource {
                            leaf: i,
                            offset: 0,
                            len: leaf.lane_bytes.unwrap_or(0),
                        },
                    )],
                    leaves: vec![i],
                    origin: escape::key_text(&leaf.keys),
                    tags: Vec::new(),
                    joints: None,
                },
            )?;
        }

        // The authored map, in declaration order (a JSON object's order is
        // the parts' order, so it is written by hand).
        let mut entries = Vec::with_capacity(drafts.len());
        for (n, draft) in drafts.iter().enumerate() {
            let side_leaves = if n < obs_count {
                &obs_leaves
            } else {
                &act_leaves
            };
            let mut keys = draft.keys.clone();
            let key_lists: Vec<&Vec<KeySeg>> =
                draft.leaves.iter().map(|l| &side_leaves[*l].keys).collect();
            let spaces: Vec<Value> = draft
                .leaves
                .iter()
                .map(|l| space_spec_to_json(&side_leaves[*l].spec).unwrap_or(Value::Null))
                .collect();
            keys.push(("gym_key", json_text(&json!(key_lists))));
            keys.push(("gym_space", json_text(&json!(spaces))));
            if !draft.tags.is_empty() {
                keys.push(("env_tags", json_text(&json!(draft.tags))));
            }
            if n == 0 {
                keys.push(("gym_env", json_text(&info.to_json())));
            }
            let body: Vec<String> = keys
                .iter()
                .map(|(k, v)| format!("{}:{v}", json_text(&json!(k))))
                .collect();
            entries.push(format!(
                "{}:{{{}}}",
                json_text(&json!(draft.path)),
                body.join(",")
            ));
        }
        let robot_json = format!("{{{}}}", entries.join(","));

        // Normalise here: the exact field layout, as the host will have it.
        let robot = ParticipantId::new("gym").map_err(|e| bad(e.to_string()))?;
        let specs = rlmesh_session::spec::normalise(
            &robot,
            rlmesh_session::wire::version::SPEC_VERSION,
            &robot_json,
        )
        .map_err(|e| {
            bad(format!(
                "the Gym robot map does not normalise: {}",
                e.message
            ))
        })?;
        let mut observation = Side {
            leaves: obs_leaves,
            parts: Vec::new(),
        };
        let mut action = Side {
            leaves: act_leaves,
            parts: Vec::new(),
        };
        for (n, (draft, spec)) in drafts.into_iter().zip(specs).enumerate() {
            let side = if n < obs_count {
                &mut observation
            } else {
                &mut action
            };
            side.parts.push(part_map(draft, spec, &side.leaves)?);
        }
        Ok(Self {
            robot_json,
            observation,
            action,
            info,
        })
    }

    fn push(
        drafts: &mut Vec<Draft>,
        by_path: &mut BTreeMap<String, usize>,
        mut draft: Draft,
    ) -> Result<(), GymError> {
        // Keys that escape alike (`Pos` and `pos`), or an observation key
        // under the action root, keep apart: the later one's last segment
        // takes the hash of its original keys.
        let commanded = draft.keys.iter().any(|(key, _)| *key == "command");
        let under_action = draft.path == "action" || draft.path.starts_with("action/");
        if by_path.contains_key(&draft.path) || (under_action && !commanded) {
            draft.path = escape::distinct(&draft.path, &draft.origin);
        }
        rlmesh_session::wire::PartPath::new(&draft.path).map_err(|e| {
            bad(format!(
                "leaf {} maps to the path {:?}, which is refused: {}",
                draft.origin, draft.path, e.reason
            ))
        })?;
        if let Some(&other) = by_path.get(&draft.path) {
            return Err(bad(format!(
                "Gym keys {} and {} both map to the path {:?}; rename one",
                drafts[other].origin, draft.origin, draft.path
            )));
        }
        by_path.insert(draft.path.clone(), drafts.len());
        drafts.push(draft);
        Ok(())
    }

    /// Maps a tagged leaf onto a core kind, merging joints; `false` when
    /// it does not fit (the leaf then keeps its custom kind).
    fn core_part(
        drafts: &mut Vec<Draft>,
        by_path: &mut BTreeMap<String, usize>,
        index: usize,
        leaf: &Leaf,
        tag: &ObsLeaf,
        hz: f64,
    ) -> Result<bool, GymError> {
        let is_box = matches!(leaf.spec.spec, Some(SpaceKind::Box(_)));
        let tag_json = serde_json::to_value(tag).unwrap_or(Value::Null);
        let whole = FieldSource {
            leaf: index,
            offset: 0,
            len: leaf.lane_bytes.unwrap_or(0),
        };
        let float = matches!(
            leaf.spec.dtype,
            rlmesh_spaces::DType::Float32 | rlmesh_spaces::DType::Float64
        );
        let dtype_key = || {
            json_text(&json!(
                if leaf.spec.dtype == rlmesh_spaces::DType::Float64 {
                    "float64"
                } else {
                    "float32"
                }
            ))
        };
        let vector = (leaf.spec.shape.len() == 1).then(|| leaf.spec.shape[0] as usize);
        let single = |path: String, keys: Vec<(&'static str, String)>, fields| Draft {
            path,
            keys,
            fields,
            leaves: vec![index],
            origin: escape::key_text(&leaf.keys),
            tags: vec![tag_json.clone()],
            joints: None,
        };
        let draft = match tag {
            ObsLeaf::Image(image) => {
                let (true, rlmesh_spaces::DType::Uint8, [h, w, 3]) =
                    (is_box, leaf.spec.dtype, leaf.spec.shape.as_slice())
                else {
                    return Ok(false);
                };
                if image.layout != rlmesh_adapters::v1::ImageLayout::Hwc {
                    return Ok(false);
                }
                let name = match image.role.as_str() {
                    "image/primary" => "scene/cam".to_string(),
                    "image/secondary" => "scene/cam_side".to_string(),
                    "image/wrist" => "wrist/cam".to_string(),
                    _ => return Ok(false),
                };
                let mut keys = vec![
                    ("kind", json_text(&json!("camera"))),
                    ("width", json_text(&json!(w))),
                    ("height", json_text(&json!(h))),
                    ("hz", json_text(&json!(hz))),
                ];
                if image.upside_down {
                    keys.push(("mount", json_text(&json!("upside_down"))));
                }
                single(
                    format!("{}{name}", part_prefix(image.part.as_deref())),
                    keys,
                    vec![("image".into(), whole)],
                )
            }
            ObsLeaf::Text(text) if text.role == "text/instruction" => {
                if !matches!(leaf.spec.spec, Some(SpaceKind::Text(_))) {
                    return Ok(false);
                }
                single(
                    "task".into(),
                    vec![("kind", json_text(&json!("text")))],
                    vec![("text".into(), whole)],
                )
            }
            ObsLeaf::State(state) => {
                let (Some(n), true, true) = (vector, is_box, float) else {
                    return Ok(false);
                };
                let prefix = part_prefix(state.part.as_deref());
                let sim = if state.provenance.as_ref().map(|p| p.as_str()) == Some("privileged") {
                    "sim/"
                } else {
                    ""
                };
                let dtype = wire_dtype(leaf.spec.dtype)?;
                match state.role.as_str() {
                    role @ ("proprio/joint_pos" | "proprio/joint_vel" | "x/joint_torque") => {
                        let field = match role {
                            "proprio/joint_pos" => "pos",
                            "proprio/joint_vel" => "vel",
                            _ => "effort",
                        };
                        let path = format!("{sim}{prefix}arm");
                        if let Some(&at) = by_path.get(&path) {
                            let existing = &mut drafts[at];
                            let fits = existing.joints == Some((n, dtype))
                                && !existing.fields.iter().any(|(f, _)| f == field);
                            if !fits {
                                return Ok(false);
                            }
                            existing.fields.push((field.into(), whole));
                            existing.leaves.push(index);
                            existing.tags.push(tag_json);
                            // `fields` lists what is published; the normaliser
                            // packs them in table order.
                            let names: Vec<&str> = ["pos", "vel", "effort"]
                                .into_iter()
                                .filter(|f| existing.fields.iter().any(|(g, _)| g == f))
                                .collect();
                            for (k, v) in &mut existing.keys {
                                if *k == "fields" {
                                    *v = json_text(&json!(names));
                                }
                            }
                            return Ok(true);
                        }
                        let mut keys = vec![
                            ("kind", json_text(&json!("joints"))),
                            ("hz", json_text(&json!(hz))),
                            ("dtype", dtype_key()),
                            ("fields", json_text(&json!([field]))),
                        ];
                        match &state.labels {
                            Some(labels) => keys.push(("names", json_text(&json!(labels)))),
                            None => keys.push(("count", json_text(&json!(n)))),
                        }
                        Draft {
                            path,
                            keys,
                            fields: vec![(field.into(), whole)],
                            leaves: vec![index],
                            origin: escape::key_text(&leaf.keys),
                            tags: vec![tag_json],
                            joints: Some((n, dtype)),
                        }
                    }
                    "proprio/gripper" if n == 1 => single(
                        format!("{sim}{prefix}gripper"),
                        vec![
                            ("kind", json_text(&json!("gripper"))),
                            ("hz", json_text(&json!(hz))),
                            ("dtype", dtype_key()),
                        ],
                        vec![("pos".into(), whole)],
                    ),
                    "proprio/gripper" if n == 2 => single(
                        format!("{sim}{prefix}gripper/fingers"),
                        vec![
                            ("kind", json_text(&json!("joints"))),
                            ("hz", json_text(&json!(hz))),
                            ("dtype", dtype_key()),
                            ("fields", json_text(&json!(["pos"]))),
                            ("count", json_text(&json!(2))),
                        ],
                        vec![("pos".into(), whole)],
                    ),
                    "proprio/eef_wrench" if n == 6 => {
                        let Some(frame) = &state.frame else {
                            return Ok(false);
                        };
                        let half = whole.len / 2;
                        single(
                            format!("{sim}{prefix}wrist/ft"),
                            vec![
                                ("kind", json_text(&json!("wrench"))),
                                ("frame", json_text(&json!(frame.as_str()))),
                                ("hz", json_text(&json!(hz))),
                                ("dtype", dtype_key()),
                            ],
                            vec![
                                (
                                    "force".into(),
                                    FieldSource {
                                        len: half,
                                        ..whole.clone()
                                    },
                                ),
                                (
                                    "torque".into(),
                                    FieldSource {
                                        offset: half,
                                        len: half,
                                        ..whole
                                    },
                                ),
                            ],
                        )
                    }
                    _ => return Ok(false),
                }
            }
            _ => return Ok(false),
        };
        if by_path.contains_key(&draft.path) {
            // Two leaves claim one core part: the later one keeps its key path.
            return Ok(false);
        }
        by_path.insert(draft.path.clone(), drafts.len());
        drafts.push(draft);
        Ok(true)
    }

    /// The part with `path`.
    pub fn part(&self, path: &str) -> Option<&PartMap> {
        self.observation
            .parts
            .iter()
            .chain(&self.action.parts)
            .find(|p| p.spec.path.as_str() == path)
    }

    /// Every part, observations first.
    pub fn parts(&self) -> impl Iterator<Item = &PartMap> {
        self.observation.parts.iter().chain(&self.action.parts)
    }

    /// Takes each part's channel id from the declared parts (by path).
    pub fn bind_channels(&mut self, declared: &[PartSpec]) -> Result<(), GymError> {
        for part in self
            .observation
            .parts
            .iter_mut()
            .chain(self.action.parts.iter_mut())
        {
            let found = declared
                .iter()
                .find(|d| d.path == part.spec.path)
                .ok_or_else(|| bad(format!("the robot declares no part {}", part.spec.path)))?;
            if found.fields != part.spec.fields {
                return Err(bad(format!(
                    "part {} is declared with other fields than this env's map",
                    part.spec.path
                )));
            }
            part.spec.channel = found.channel;
            part.spec.owner = found.owner.clone();
        }
        Ok(())
    }

    /// Numbers the parts 1, 2, ... (no host: the in-process path).
    pub fn number_channels(&mut self) {
        for (i, part) in self
            .observation
            .parts
            .iter_mut()
            .chain(self.action.parts.iter_mut())
            .enumerate()
        {
            part.spec.channel = ChannelId(i as u32 + 1);
        }
    }
}

fn part_map(draft: Draft, spec: PartSpec, leaves: &[Leaf]) -> Result<PartMap, GymError> {
    let layout = Layout::of(&spec.fields).map_err(|e| bad(format!("{}: {e}", spec.path)))?;
    let mut sources = Vec::with_capacity(spec.fields.len());
    for (i, field) in spec.fields.iter().enumerate() {
        let (_, source) = draft
            .fields
            .iter()
            .find(|(name, _)| *name == field.name)
            .ok_or_else(|| bad(format!("{}: no source for field {}", spec.path, field.name)))?;
        if let Some(range) = layout.field(i)
            && range.len() != source.len
        {
            return Err(bad(format!(
                "{}: field {} is {} bytes, its leaf gives {}",
                spec.path,
                field.name,
                range.len(),
                source.len
            )));
        }
        sources.push(source.clone());
    }
    let zero_copy = match (&layout, sources.as_slice()) {
        (Layout::Text, _) => true,
        (Layout::Packed { stride, .. }, [only]) => {
            only.offset == 0 && leaves[only.leaf].lane_bytes == Some(*stride)
        }
        _ => false,
    };
    Ok(PartMap {
        spec,
        layout,
        sources,
        zero_copy,
    })
}

/// Packs a side's leaf slabs (`lanes` lanes each) into one slab per part.
pub fn pack(side: &Side, leaves: &[Bytes], lanes: usize) -> Result<Vec<Bytes>, GymError> {
    if leaves.len() != side.leaves.len() {
        return Err(GymError::Malformed(format!(
            "{} leaves, the space has {}",
            leaves.len(),
            side.leaves.len()
        )));
    }
    for (leaf, bytes) in side.leaves.iter().zip(leaves) {
        if let Some(lb) = leaf.lane_bytes
            && Some(bytes.len()) != lb.checked_mul(lanes)
        {
            return Err(GymError::Malformed(format!(
                "leaf {} is {} bytes, {lanes} lanes of {lb} expected",
                escape::key_text(&leaf.keys),
                bytes.len()
            )));
        }
    }
    side.parts
        .iter()
        .map(|part| {
            if part.zero_copy {
                return Ok(leaves[part.sources[0].leaf].clone());
            }
            let Layout::Packed {
                offsets, stride, ..
            } = &part.layout
            else {
                return Ok(leaves[part.sources[0].leaf].clone());
            };
            let mut out = vec![0_u8; stride * lanes];
            for lane in 0..lanes {
                for (source, offset) in part.sources.iter().zip(offsets) {
                    let lb = side.leaves[source.leaf].lane_bytes.unwrap_or(0);
                    let from = lane * lb + source.offset;
                    let to = lane * stride + offset;
                    out[to..to + source.len]
                        .copy_from_slice(&leaves[source.leaf][from..from + source.len]);
                }
            }
            Ok(Bytes::from(out))
        })
        .collect()
}

/// Unpacks one slab per part (in part order) into the side's leaf slabs.
pub fn unpack(side: &Side, slabs: &[Bytes], lanes: usize) -> Result<Vec<Bytes>, GymError> {
    if slabs.len() != side.parts.len() {
        return Err(GymError::Malformed(format!(
            "{} part slabs, the robot has {}",
            slabs.len(),
            side.parts.len()
        )));
    }
    let mut out: Vec<Option<Bytes>> = vec![None; side.leaves.len()];
    let mut scratch: Vec<Option<Vec<u8>>> = vec![None; side.leaves.len()];
    for (part, slab) in side.parts.iter().zip(slabs) {
        match (&part.layout, part.zero_copy) {
            (Layout::Text, _) | (_, true) => {
                if let Some(stride) = part.layout.stride()
                    && slab.len() != stride * lanes
                {
                    return Err(GymError::Malformed(format!(
                        "part {} slab is {} bytes, {lanes} lanes of {stride} expected",
                        part.spec.path,
                        slab.len()
                    )));
                }
                out[part.sources[0].leaf] = Some(slab.clone());
            }
            (
                Layout::Packed {
                    offsets, stride, ..
                },
                false,
            ) => {
                if slab.len() != stride * lanes {
                    return Err(GymError::Malformed(format!(
                        "part {} slab is {} bytes, {lanes} lanes of {stride} expected",
                        part.spec.path,
                        slab.len()
                    )));
                }
                for (source, offset) in part.sources.iter().zip(offsets) {
                    let lb = side.leaves[source.leaf].lane_bytes.unwrap_or(0);
                    let buf = scratch[source.leaf].get_or_insert_with(|| vec![0_u8; lb * lanes]);
                    for lane in 0..lanes {
                        let to = lane * lb + source.offset;
                        let from = lane * stride + offset;
                        buf[to..to + source.len].copy_from_slice(&slab[from..from + source.len]);
                    }
                }
            }
        }
    }
    out.into_iter()
        .zip(scratch)
        .enumerate()
        .map(|(i, (whole, built))| {
            whole.or_else(|| built.map(Bytes::from)).ok_or_else(|| {
                GymError::Malformed(format!(
                    "leaf {} is in no part",
                    escape::key_text(&side.leaves[i].keys)
                ))
            })
        })
        .collect()
}
