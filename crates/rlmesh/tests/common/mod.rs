//! A catalog-style v1 env for the lockstep tests: every v1 leaf kind and
//! dtype, v1 tags a LIBERO-style `ModelSpec` resolves against, NEXT_STEP or
//! DISABLED autoreset, scalar, vector or lane endpoints, and deterministic
//! dynamics (a pure function of seed, lane, step and the last action), so
//! two runs fed the same requests answer the same bytes.

#![allow(dead_code, clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use bytes::Bytes;
use rlmesh_grpc::env::{
    CloseEnvsResponse, EndpointPhases, Environment, RenderRequest, RenderResponse, ResetRequest,
    ResetResponse, StepRequest, StepResponse,
};
use rlmesh_grpc::error::EnvError;
use rlmesh_proto::spaces::v1::{MetaList, MetaMap, MetaValue as ProtoMeta, meta_value::Kind};
use rlmesh_spaces::{
    AutoresetMode, BoxBounds, BoxSpec, DType, DictSpec, DiscreteSpec, EnvContract, MetaValue,
    MultiBinarySpec, MultiDiscreteSpec, SpaceKind, SpaceSpec, SpaceValue, Tensor, TextSpec,
    TupleSpec, UniformBounds,
};

pub fn boxed(dtype: DType, shape: &[i64]) -> SpaceSpec {
    bounded(dtype, shape, -1.0, 1.0)
}

pub fn bounded(dtype: DType, shape: &[i64], low: f64, high: f64) -> SpaceSpec {
    SpaceSpec {
        shape: shape.to_vec(),
        dtype,
        spec: Some(SpaceKind::Box(BoxSpec {
            bounds: Some(BoxBounds::Uniform(UniformBounds { low, high })),
        })),
    }
}

pub fn dict(items: Vec<(&str, SpaceSpec)>) -> SpaceSpec {
    SpaceSpec {
        shape: Vec::new(),
        dtype: DType::Unspecified,
        spec: Some(SpaceKind::Dict(DictSpec {
            keys: items.iter().map(|(k, _)| (*k).to_string()).collect(),
            spaces: items.into_iter().map(|(_, s)| s).collect(),
        })),
    }
}

pub fn text() -> SpaceSpec {
    SpaceSpec {
        shape: Vec::new(),
        dtype: DType::Uint8,
        spec: Some(SpaceKind::Text(TextSpec {
            min_length: 0,
            max_length: 64,
            charset: String::new(),
        })),
    }
}

pub fn native(value: &serde_json::Value) -> MetaValue {
    match value {
        serde_json::Value::Null => MetaValue::Null,
        serde_json::Value::Bool(b) => MetaValue::Bool(*b),
        serde_json::Value::Number(n) => match n.as_i64() {
            Some(i) => MetaValue::Int(i),
            None => MetaValue::Float(n.as_f64().unwrap_or(0.0)),
        },
        serde_json::Value::String(s) => MetaValue::String(s.clone()),
        serde_json::Value::Array(items) => MetaValue::List(items.iter().map(native).collect()),
        serde_json::Value::Object(map) => MetaValue::Map(
            map.iter()
                .map(|(k, v)| (k.clone(), native(v)))
                .collect::<BTreeMap<_, _>>(),
        ),
    }
}

/// The LIBERO-style tags of the conformance case `apply_basic_pairing`,
/// over a catalog env with every other v1 leaf kind beside them.
pub fn catalog_tags() -> serde_json::Value {
    serde_json::json!({
        "action": {"clip": [-1.0, 1.0], "components": [
            {"dim": 3, "role": "action/delta_eef_pos"},
            {"dim": 3, "encoding": "axis_angle", "role": "action/delta_eef_rot"},
            {"dim": 1, "range": [-1.0, 1.0], "role": "action/gripper"}]},
        "observation": {
            "cam": {"layout": "hwc", "role": "image/primary", "type": "image"},
            "eef_pos": {"role": "proprio/eef_pos", "type": "state"},
            "eef_quat": {"encoding": "quat_xyzw", "role": "proprio/eef_rot", "type": "state"},
            "gripper": {"role": "proprio/gripper", "type": "state"},
            "instruction": {"role": "text/instruction", "type": "text"},
        }
    })
}

/// The model side of the same case.
pub const CATALOG_MODEL_SPEC: &str = r#"{
  "input": {
    "image": {"type": "image", "role": "image/primary", "dtype": "float32", "layout": "chw",
              "normalize": true, "resample": "bilinear_aa", "height": 4, "width": 4},
    "instruction": {"type": "text", "role": "text/instruction", "container": "str"},
    "state": {"type": "state", "container": "list", "dtype": "float32", "components": [
      {"dim": 3, "role": "proprio/eef_pos"},
      {"encoding": "axis_angle", "role": "proprio/eef_rot"},
      {"dim": 1, "role": "proprio/gripper"}]}
  },
  "output": {"components": [
    {"dim": 3, "role": "action/delta_eef_pos"},
    {"dim": 3, "encoding": "axis_angle", "role": "action/delta_eef_rot"},
    {"binary": true, "dim": 1, "range": [0.0, 1.0], "role": "action/gripper"}]}
}"#;

pub fn catalog_observation() -> SpaceSpec {
    dict(vec![
        ("cam", bounded(DType::Uint8, &[6, 6, 3], 0.0, 255.0)),
        ("eef_pos", boxed(DType::Float32, &[3])),
        ("eef_quat", boxed(DType::Float32, &[4])),
        ("gripper", boxed(DType::Float32, &[2])),
        ("instruction", text()),
        (
            "mode",
            SpaceSpec {
                shape: Vec::new(),
                dtype: DType::Int64,
                spec: Some(SpaceKind::Discrete(DiscreteSpec { n: 4, start: 0 })),
            },
        ),
        (
            "md",
            SpaceSpec {
                shape: vec![3],
                dtype: DType::Int32,
                spec: Some(SpaceKind::MultiDiscrete(MultiDiscreteSpec {
                    nvec: vec![2, 3, 4],
                })),
            },
        ),
        (
            "mb",
            SpaceSpec {
                shape: vec![5],
                dtype: DType::Int8,
                spec: Some(SpaceKind::MultiBinary(MultiBinarySpec)),
            },
        ),
        (
            "pair",
            SpaceSpec {
                shape: Vec::new(),
                dtype: DType::Unspecified,
                spec: Some(SpaceKind::Tuple(TupleSpec {
                    spaces: vec![
                        boxed(DType::Int8, &[3]),
                        bounded(DType::Uint64, &[2], 0.0, 1.0),
                    ],
                })),
            },
        ),
        (
            "Nested",
            dict(vec![
                ("i16", boxed(DType::Int16, &[5])),
                ("u32", bounded(DType::Uint32, &[1], 0.0, 1.0)),
                ("u16", bounded(DType::Uint16, &[3], 0.0, 1.0)),
                ("f64", boxed(DType::Float64, &[2, 2])),
                ("half", boxed(DType::Float16, &[3])),
                ("flags", bounded(DType::Bool, &[2], 0.0, 1.0)),
                ("odd-f32", boxed(DType::Float32, &[3])),
            ]),
        ),
    ])
}

pub fn catalog_contract(num_envs: u32, autoreset: AutoresetMode) -> EnvContract {
    let mut metadata = BTreeMap::new();
    metadata.insert(
        rlmesh::lockstep::adapters::ENV_METADATA_KEY.to_string(),
        native(&catalog_tags()),
    );
    EnvContract {
        id: "Catalog-v0".into(),
        action_space: Some(boxed(DType::Float32, &[7])),
        observation_space: Some(catalog_observation()),
        metadata: Some(metadata),
        render_mode: String::new(),
        num_envs,
        autoreset_mode: autoreset,
    }
}

/// A mixing hash (splitmix64).
pub fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x100_0000_01b3)
    })
}

#[derive(Clone, Debug, Default)]
struct Lane {
    seed: u64,
    t: u64,
    /// Steps until the episode ends.
    length: u64,
    /// Terminate (else truncate) at the end.
    terminates: bool,
    pending_autoreset: bool,
    active: bool,
    last_action: u64,
}

impl Lane {
    fn start(&mut self, seed: u64) {
        let h = mix(seed);
        *self = Self {
            seed,
            t: 0,
            length: 3 + h % 5,
            terminates: h & 1 == 0,
            pending_autoreset: false,
            active: true,
            last_action: 0,
        };
    }

    fn value(&self, lane: usize) -> SpaceValue {
        // Lane-free: an episode's transcript depends on its seed and actions only,
        // so lane-racing runs compare as multisets of episodes.
        let _ = lane;
        let h = |salt: u64| mix(self.seed ^ (self.t << 20) ^ salt ^ self.last_action);
        let bytes = |n: usize, salt: u64| -> Vec<u8> {
            (0..n).map(|i| (h(salt + i as u64) & 0xff) as u8).collect()
        };
        // Little-endian integers in {-1, 0, 1} (signed) or {0, 1} (unsigned).
        let ints = |n: usize, width: usize, signed: bool, salt: u64| -> Vec<u8> {
            (0..n)
                .flat_map(|i| {
                    let v: i64 = if signed {
                        (h(salt + i as u64) % 3) as i64 - 1
                    } else {
                        (h(salt + i as u64) % 2) as i64
                    };
                    v.to_le_bytes()[..width].to_vec()
                })
                .collect()
        };
        // binary16 in {-1, -0.5, 0, 0.5, 1}.
        let halves = |n: usize, salt: u64| -> Vec<u8> {
            const H: [u16; 5] = [0xBC00, 0xB800, 0x0000, 0x3800, 0x3C00];
            (0..n)
                .flat_map(|i| H[(h(salt + i as u64) % 5) as usize].to_le_bytes())
                .collect()
        };
        let floats = |n: usize, salt: u64| -> Vec<u8> {
            (0..n)
                .flat_map(|i| {
                    let x = (h(salt + i as u64) % 2000) as f32 / 1000.0 - 1.0;
                    x.to_le_bytes()
                })
                .collect()
        };
        let tensor = |data: Vec<u8>, shape: &[i64], dtype: DType| {
            SpaceValue::Box(Tensor::from_vec(data, shape.to_vec(), dtype).unwrap())
        };
        let quat: Vec<u8> = {
            let q = [0.5_f32, 0.5, 0.5, 0.5];
            q.iter().flat_map(|x| x.to_le_bytes()).collect()
        };
        let mut nested = BTreeMap::new();
        nested.insert(
            "i16".into(),
            tensor(ints(5, 2, true, 1), &[5], DType::Int16),
        );
        nested.insert(
            "u32".into(),
            tensor(ints(1, 4, false, 2), &[1], DType::Uint32),
        );
        nested.insert(
            "u16".into(),
            tensor(ints(3, 2, false, 3), &[3], DType::Uint16),
        );
        nested.insert(
            "f64".into(),
            tensor(
                (0..4)
                    .flat_map(|i| ((h(40 + i) % 3000) as f64 / 1500.0 - 1.0).to_le_bytes())
                    .collect(),
                &[2, 2],
                DType::Float64,
            ),
        );
        nested.insert("half".into(), tensor(halves(3, 5), &[3], DType::Float16));
        nested.insert(
            "flags".into(),
            tensor(vec![(h(6) & 1) as u8, (h(7) & 1) as u8], &[2], DType::Bool),
        );
        nested.insert("odd-f32".into(), tensor(floats(3, 8), &[3], DType::Float32));
        let mut obs = BTreeMap::new();
        obs.insert(
            "cam".into(),
            tensor(bytes(108, 9), &[6, 6, 3], DType::Uint8),
        );
        obs.insert(
            "eef_pos".into(),
            tensor(floats(3, 10), &[3], DType::Float32),
        );
        obs.insert("eef_quat".into(), tensor(quat, &[4], DType::Float32));
        obs.insert(
            "gripper".into(),
            tensor(floats(2, 11), &[2], DType::Float32),
        );
        obs.insert(
            "instruction".into(),
            SpaceValue::Text(format!("stack cup {} on the plate", h(12) % 7)),
        );
        obs.insert("mode".into(), SpaceValue::Discrete((h(13) % 4) as i64));
        obs.insert(
            "md".into(),
            SpaceValue::MultiDiscrete(vec![
                (h(14) % 2) as i64,
                (h(15) % 3) as i64,
                (h(16) % 4) as i64,
            ]),
        );
        obs.insert(
            "mb".into(),
            SpaceValue::MultiBinary((0..5).map(|i| h(17 + i) & 1 == 1).collect()),
        );
        obs.insert(
            "pair".into(),
            SpaceValue::Tuple(vec![
                tensor(ints(3, 1, true, 22), &[3], DType::Int8),
                tensor(ints(2, 8, false, 23), &[2], DType::Uint64),
            ]),
        );
        obs.insert("Nested".into(), SpaceValue::Dict(nested));
        SpaceValue::Dict(obs)
    }

    fn reward(&self) -> f64 {
        (self.t as f64) / 3.0 + (self.last_action % 1000) as f64 * 1e-7
    }
}

fn meta(kind: Kind) -> ProtoMeta {
    ProtoMeta { kind: Some(kind) }
}

/// What the env saw and answered, per lane, in order.
pub type Trace = Arc<Mutex<BTreeMap<usize, Vec<String>>>>;

/// The catalog env.
pub struct CatalogEnv {
    contract: EnvContract,
    lanes: Mutex<Vec<Lane>>,
    lane_endpoint: bool,
    pub trace: Trace,
}

impl CatalogEnv {
    /// `num_envs` lanes; a lane endpoint resets and steps lanes one by one.
    pub fn new(num_envs: u32, autoreset: AutoresetMode, lane_endpoint: bool) -> Self {
        Self {
            contract: catalog_contract(num_envs, autoreset),
            lanes: Mutex::new(vec![Lane::default(); num_envs as usize]),
            lane_endpoint,
            trace: Arc::default(),
        }
    }

    fn observe(&self, lanes: &[usize], state: &[Lane]) -> rlmesh_proto::spaces::v1::SpaceValue {
        let values: Vec<SpaceValue> = lanes.iter().map(|l| state[*l].value(*l)).collect();
        rlmesh_grpc::wire::encode_batched_partial_values(
            &values,
            self.contract.observation_space.as_ref().unwrap(),
        )
        .unwrap()
    }

    fn log(&self, lane: usize, line: String) {
        self.trace
            .lock()
            .unwrap()
            .entry(lane)
            .or_default()
            .push(line);
    }
}

fn hex(bytes: &[u8]) -> String {
    format!("{:016x}:{}", fnv(bytes), bytes.len())
}

fn leaves_hex(value: &rlmesh_proto::spaces::v1::SpaceValue) -> String {
    value
        .leaves
        .iter()
        .map(|l| hex(l))
        .collect::<Vec<_>>()
        .join(",")
}

#[async_trait]
impl Environment for CatalogEnv {
    fn observation_space(&self) -> &SpaceSpec {
        self.contract.observation_space.as_ref().unwrap()
    }
    fn action_space(&self) -> &SpaceSpec {
        self.contract.action_space.as_ref().unwrap()
    }
    fn num_envs(&self) -> usize {
        self.contract.num_envs as usize
    }
    fn env_contract(&self) -> &EnvContract {
        &self.contract
    }
    fn supports_lanes(&self) -> bool {
        self.lane_endpoint
    }

    async fn reset(&self, req: ResetRequest) -> Result<(ResetResponse, EndpointPhases), EnvError> {
        let n = self.num_envs();
        let lanes: Vec<usize> = if req.env_indices.is_empty() {
            (0..n).collect()
        } else {
            req.env_indices.iter().map(|l| *l as usize).collect()
        };
        let mut state = self.lanes.lock().unwrap();
        for (pos, lane) in lanes.iter().enumerate() {
            let seed = req
                .seeds
                .get(pos)
                .map_or(1000 + *lane as u64, |s| *s as u64);
            state[*lane].start(seed);
        }
        let observation = self.observe(&lanes, &state);
        let options = req.options.as_ref().map(|o| {
            format!(
                "{:?}",
                o.entries.keys().collect::<std::collections::BTreeSet<_>>()
            )
        });
        let mut infos = MetaMap::default();
        infos.entries.insert(
            "seeds".into(),
            meta(Kind::List(MetaList {
                items: lanes
                    .iter()
                    .map(|l| meta(Kind::Integer(state[*l].seed as i64)))
                    .collect(),
            })),
        );
        infos
            .entries
            .insert("nan".into(), meta(Kind::Number(f64::NAN)));
        infos
            .entries
            .insert("blob".into(), meta(Kind::Bytes(vec![1, 2, 3])));
        for (pos, lane) in lanes.iter().enumerate() {
            self.log(
                *lane,
                format!(
                    "reset seed={:?} options={options:?} obs={}",
                    req.seeds.get(pos),
                    leaves_hex(&observation)
                ),
            );
        }
        Ok((
            ResetResponse {
                observation: Some(observation),
                infos: Some(infos),
            },
            EndpointPhases::default(),
        ))
    }

    async fn step(&self, req: StepRequest) -> Result<(StepResponse, EndpointPhases), EnvError> {
        let n = self.num_envs();
        let lanes: Vec<usize> = if req.env_indices.is_empty() {
            (0..n).collect()
        } else {
            req.env_indices.iter().map(|l| *l as usize).collect()
        };
        let action = req.action.as_ref().unwrap();
        let slab: &Bytes = &action.leaves[0];
        let stride = slab.len() / lanes.len();
        let next_step = self.contract.autoreset_mode == AutoresetMode::NextStep;
        let mut state = self.lanes.lock().unwrap();
        let (mut rewards, mut terminated, mut truncated) = (Vec::new(), Vec::new(), Vec::new());
        let mut done_flags = Vec::new();
        for (pos, lane) in lanes.iter().enumerate() {
            let bytes = &slab[pos * stride..(pos + 1) * stride];
            let s = &mut state[*lane];
            if s.pending_autoreset {
                let seed = mix(s.seed);
                s.start(seed);
                rewards.push(0.0);
                terminated.push(0);
                truncated.push(0);
                done_flags.push(false);
                continue;
            }
            s.t += 1;
            s.last_action = fnv(bytes);
            let done = s.t >= s.length;
            rewards.push(s.reward());
            terminated.push(u8::from(done && s.terminates));
            truncated.push(u8::from(done && !s.terminates));
            done_flags.push(done && s.terminates);
            if done {
                if next_step {
                    s.pending_autoreset = true;
                } else {
                    s.active = false;
                }
            }
        }
        let observation = self.observe(&lanes, &state);
        let mut infos = MetaMap::default();
        if lanes.len() == 1 {
            infos
                .entries
                .insert("is_success".into(), meta(Kind::Bool(done_flags[0])));
            infos
                .entries
                .insert("x".into(), meta(Kind::Number(rewards[0] * 2.0)));
        } else {
            infos.entries.insert(
                "t".into(),
                meta(Kind::List(MetaList {
                    items: lanes
                        .iter()
                        .map(|l| meta(Kind::Integer(state[*l].t as i64)))
                        .collect(),
                })),
            );
        }
        for (pos, lane) in lanes.iter().enumerate() {
            self.log(
                *lane,
                format!(
                    "step action={} reward={:016x} term={} trunc={} obs={}",
                    hex(&slab[pos * stride..(pos + 1) * stride]),
                    rewards[pos].to_bits(),
                    terminated[pos],
                    truncated[pos],
                    leaves_hex(&observation)
                ),
            );
        }
        Ok((
            StepResponse {
                observation: Some(observation),
                rewards,
                terminated_mask: terminated,
                truncated_mask: truncated,
                infos: Some(infos),
                completed_episodes: Vec::new(),
                env_indices: Vec::new(),
            },
            EndpointPhases::default(),
        ))
    }

    async fn render(
        &self,
        _req: RenderRequest,
    ) -> Result<(RenderResponse, EndpointPhases), EnvError> {
        Ok((RenderResponse::default(), EndpointPhases::default()))
    }

    async fn close(&self) -> Result<CloseEnvsResponse, EnvError> {
        Ok(CloseEnvsResponse::default())
    }
}
