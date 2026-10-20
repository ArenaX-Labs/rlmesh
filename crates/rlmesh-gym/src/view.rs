//! The Gym view over a v2-native robot in lockstep (§12.6).
//!
//! [`GymView`] drives a robot that serves the lockstep exchange (a
//! [`LockstepHandler`](crate::LockstepHandler) behind [`serve_robot`](crate::LockstepBinding::serve_robot))
//! and presents it as a Gymnasium env:
//!
//! - the observation space is keyed by the policy's input names, or by part
//!   path (`/` → `.`) without a policy: a single-field part is its field, a
//!   multi-field part a dict of fields;
//! - the action space is keyed by action names (`[horizon, ...]`, squeezed
//!   when the horizon is 1), or by commandable part path without a policy;
//! - one view step with horizon H is H lockstep steps (stopping at the end
//!   of the episode), rewards summed;
//! - `info` carries each input's age (in session time: steps × `tick_ns`),
//!   the stale inputs, and each action's edge mode (`driving` for a part
//!   commanded this step, else `idle`; lockstep has no hold or safe).
//!
//! With a policy, the binder's plans run on the replies' packed payloads
//! (conversions, history windows, blanks), exactly as for a v2 policy in
//! wall mode; a window shorter than its history is padded with its oldest
//! sample, as `allow: warmup` would.
//!
//! [`Frame`] is the robot side's helper: it packs `publish`ed fields into
//! the reply's part slabs.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use bytes::Bytes;
use rlmesh_spec::apply::{Tensor, Value};
use rlmesh_spec::bind::{ActionRow, InputRow};
use rlmesh_spec::plan::Delivery;
use rlmesh_spec::{Policy, RunOptions};
use rlmesh_wire::{
    ChannelId, DType, Epoch, FieldSpec, Layout, LeaseToken, LockstepReply, LockstepReset,
    LockstepStep, PartSpec, PartValues, ParticipantId,
};

use crate::error::GymError;
use crate::lockstep::{LockstepLink, Request};

/// What one observation or action entry is.
#[derive(Clone, Debug, PartialEq)]
pub enum Entry {
    /// One array.
    Array {
        /// Its dtype.
        dtype: DType,
        /// Its shape.
        shape: Vec<u32>,
    },
    /// Arrays by field name.
    Dict(Vec<(String, DType, Vec<u32>)>),
    /// A `str`.
    Text,
}

fn entry_of(fields: &[FieldSpec]) -> Entry {
    match fields {
        [field] if field.dtype == DType::Utf8 => Entry::Text,
        [field] => Entry::Array {
            dtype: field.dtype,
            shape: field.shape.clone(),
        },
        fields => Entry::Dict(
            fields
                .iter()
                .map(|f| (f.name.clone(), f.dtype, f.shape.clone()))
                .collect(),
        ),
    }
}

fn entry_of_delivery(delivery: &Delivery, squeeze: bool) -> Entry {
    let cut = |shape: &[u32]| -> Vec<u32> {
        if squeeze && shape.first() == Some(&1) {
            shape[1..].to_vec()
        } else {
            shape.to_vec()
        }
    };
    match delivery {
        Delivery::Array { dtype, shape, .. } => Entry::Array {
            dtype: *dtype,
            shape: cut(shape),
        },
        Delivery::Dict(fields) => Entry::Dict(
            fields
                .iter()
                .map(|d| (d.name.clone(), d.dtype, cut(&d.shape)))
                .collect(),
        ),
        Delivery::Text => Entry::Text,
    }
}

/// A part path as a Gym key.
pub fn dotted(path: &str) -> String {
    path.replace('/', ".")
}

/// The robot side: the reply a lockstep robot builds as it publishes.
#[derive(Debug)]
pub struct Frame {
    parts: Vec<PartSpec>,
    values: Vec<PartValues>,
    /// Reward since the previous step.
    pub reward: f32,
    /// The episode reached a terminal state.
    pub terminated: bool,
    /// The episode was cut short.
    pub truncated: bool,
    /// `info`, as JSON (empty: none).
    pub info_json: String,
}

impl Frame {
    /// An empty frame for a robot with `parts` (one lane).
    pub fn new(parts: Vec<PartSpec>) -> Self {
        Self {
            parts,
            values: Vec::new(),
            reward: 0.0,
            terminated: false,
            truncated: false,
            info_json: String::new(),
        }
    }

    /// The robot's parts.
    pub fn parts(&self) -> &[PartSpec] {
        &self.parts
    }

    /// Publishes `path`'s fields (raw little-endian bytes, in field order;
    /// UTF-8 for text) into the reply.
    pub fn publish(&mut self, path: &str, fields: &[&[u8]]) -> Result<(), GymError> {
        let part = self
            .parts
            .iter()
            .find(|p| p.path.as_str() == path)
            .ok_or_else(|| GymError::Map(format!("the robot has no part {path}")))?;
        let layout = Layout::of(&part.fields).map_err(GymError::Malformed)?;
        let payload = layout
            .pack(fields)
            .map_err(|e| GymError::Malformed(format!("{path}: {e}")))?;
        let channel = part.channel;
        self.values.retain(|v| v.channel != channel);
        self.values.push(PartValues {
            channel,
            payload: Bytes::from(payload),
        });
        Ok(())
    }

    /// The reply to `epoch`, `step`.
    pub fn reply(self, epoch: Epoch, step: u64) -> LockstepReply {
        LockstepReply {
            epoch,
            step,
            observations: self.values,
            reward: Bytes::from(self.reward.to_le_bytes().to_vec()),
            terminated: Bytes::from(vec![u8::from(self.terminated)]),
            truncated: Bytes::from(vec![u8::from(self.truncated)]),
            episode_ids: Vec::new(),
            info_json: self.info_json,
        }
    }
}

/// The commands of one lockstep step, by part path, unpacked into fields.
pub fn commands(
    parts: &[PartSpec],
    step: &LockstepStep,
) -> Result<Vec<(String, Vec<Bytes>)>, GymError> {
    step.commands
        .iter()
        .map(|values| {
            let part = parts
                .iter()
                .find(|p| p.channel == values.channel)
                .ok_or_else(|| GymError::Malformed(format!("no part {:?}", values.channel)))?;
            let command = part
                .command
                .as_ref()
                .ok_or_else(|| GymError::Malformed(format!("{} is not commandable", part.path)))?;
            let layout = command.layout().map_err(GymError::Malformed)?;
            let stride = layout.stride().unwrap_or(0);
            if values.payload.len() != stride {
                return Err(GymError::Malformed(format!(
                    "{}: a {} byte command, the part takes {stride}",
                    part.path,
                    values.payload.len()
                )));
            }
            let fields = (0..command.fields.len())
                .filter_map(|i| layout.field(i))
                .map(|range| values.payload.slice(range))
                .collect();
            Ok((part.path.to_string(), fields))
        })
        .collect()
}

enum Input {
    Raw { name: String, part: Box<PartSpec> },
    Bound(Box<InputRow>),
}

enum Output {
    Raw { name: String, part: Box<PartSpec> },
    Bound(Box<ActionRow>),
}

impl Output {
    fn name(&self) -> &str {
        match self {
            Self::Raw { name, .. } => name,
            Self::Bound(row) => &row.entry,
        }
    }

    fn part(&self) -> Option<&PartSpec> {
        match self {
            Self::Raw { part, .. } => Some(part),
            Self::Bound(row) => row.part.as_ref(),
        }
    }
}

/// One view step's outcome.
#[derive(Clone, Debug, PartialEq)]
pub struct ViewStep {
    /// The observation, by entry.
    pub observation: Vec<(String, Value)>,
    /// Summed reward of the lockstep steps taken.
    pub reward: f64,
    /// The episode reached a terminal state.
    pub terminated: bool,
    /// The episode was cut short.
    pub truncated: bool,
    /// Lockstep steps taken.
    pub steps: u32,
    /// Age of each input's newest sample, in ns of session time (`None`:
    /// never published).
    pub ages_ns: Vec<(String, Option<u64>)>,
    /// Inputs older than their freshness bound, or never published.
    pub stale: Vec<String>,
    /// Each action's edge mode this step.
    pub edge_modes: Vec<(String, &'static str)>,
    /// The robot's own `info` (JSON; empty: none).
    pub info_json: String,
}

struct History {
    /// Per channel, the newest samples (oldest first) and the step of the
    /// newest.
    samples: BTreeMap<ChannelId, (VecDeque<Bytes>, u64)>,
    depth: BTreeMap<ChannelId, usize>,
}

impl History {
    fn clear(&mut self) {
        self.samples.clear();
    }

    fn take(&mut self, reply: &LockstepReply, step: u64) {
        for values in &reply.observations {
            let depth = self.depth.get(&values.channel).copied().unwrap_or(1);
            let (ring, at) = self
                .samples
                .entry(values.channel)
                .or_insert_with(|| (VecDeque::new(), step));
            ring.push_back(values.payload.clone());
            while ring.len() > depth {
                ring.pop_front();
            }
            *at = step;
        }
    }
}

/// A Gym env over a v2-native lockstep robot.
pub struct GymView {
    link: Arc<dyn LockstepLink>,
    inputs: Vec<Input>,
    outputs: Vec<Output>,
    tick_ns: u64,
    epoch: AtomicU64,
    token: AtomicU64,
    step: AtomicU64,
    history: Mutex<History>,
    table: Option<String>,
    _keep: Option<Box<dyn std::any::Any + Send + Sync>>,
}

impl GymView {
    /// A view of the robot whose `parts` are behind `link`, through
    /// `policy` (`None`: every part, by path), in `epoch` under `token`.
    pub fn new(
        link: Arc<dyn LockstepLink>,
        parts: &[PartSpec],
        policy: Option<(&Policy, RunOptions)>,
        epoch: Epoch,
        token: LeaseToken,
        keep: Option<Box<dyn std::any::Any + Send + Sync>>,
    ) -> Result<Self, GymError> {
        let (inputs, outputs, table, depth) = match policy {
            None => {
                let inputs = parts
                    .iter()
                    .map(|p| Input::Raw {
                        name: dotted(p.path.as_str()),
                        part: Box::new(p.clone()),
                    })
                    .collect();
                let outputs = parts
                    .iter()
                    .filter(|p| p.command.is_some())
                    .map(|p| Output::Raw {
                        name: dotted(p.path.as_str()),
                        part: Box::new(p.clone()),
                    })
                    .collect();
                (inputs, outputs, None, BTreeMap::new())
            }
            Some((policy, options)) => {
                let binding = rlmesh_spec::bind(policy, parts, &options)
                    .map_err(|e| GymError::Map(e.to_string()))?;
                let mut depth = BTreeMap::new();
                for row in &binding.inputs {
                    if let Some(channel) = row.channel {
                        let d = depth.entry(channel).or_insert(1_usize);
                        *d = (*d).max(row.history.max(1) as usize);
                    }
                }
                let table = binding.table_text();
                (
                    binding
                        .inputs
                        .into_iter()
                        .map(|r| Input::Bound(Box::new(r)))
                        .collect(),
                    binding
                        .actions
                        .into_iter()
                        .map(|r| Output::Bound(Box::new(r)))
                        .collect(),
                    Some(table),
                    depth,
                )
            }
        };
        let tick_ns = 1_000_000;
        Ok(Self {
            link,
            inputs,
            outputs,
            tick_ns,
            epoch: AtomicU64::new(epoch.0),
            token: AtomicU64::new(token.0),
            step: AtomicU64::new(0),
            history: Mutex::new(History {
                samples: BTreeMap::new(),
                depth,
            }),
            table,
            _keep: keep,
        })
    }

    /// The binding table (with a policy).
    pub fn table(&self) -> Option<&str> {
        self.table.as_deref()
    }

    /// The observation entries, in order.
    pub fn observation_entries(&self) -> Vec<(String, Entry)> {
        self.inputs
            .iter()
            .map(|input| match input {
                Input::Raw { name, part } => (name.clone(), entry_of(&part.fields)),
                Input::Bound(row) => (row.entry.clone(), entry_of_delivery(&row.delivers, false)),
            })
            .collect()
    }

    /// The action entries, in order.
    pub fn action_entries(&self) -> Vec<(String, Entry)> {
        self.outputs
            .iter()
            .filter_map(|output| match output {
                Output::Raw { name, part } => part
                    .command
                    .as_ref()
                    .map(|c| (name.clone(), entry_of(&c.fields))),
                Output::Bound(row) => row.part.as_ref().map(|_| {
                    (
                        row.entry.clone(),
                        entry_of_delivery(&row.delivers, row.horizon == 1),
                    )
                }),
            })
            .collect()
    }

    /// Resets (`seed`: one seed; `options_json`: Gym options).
    pub async fn reset(&self, seed: Option<u64>, options_json: &str) -> Result<ViewStep, GymError> {
        let reset = LockstepReset {
            epoch: Epoch(self.epoch.load(Ordering::Acquire)),
            lanes: Vec::new(),
            seeds: seed.into_iter().collect(),
            options_json: options_json.to_string(),
            episode_ids: Vec::new(),
        };
        let reply = self.link.call(Request::Reset(reset), None).await?;
        let step = self.step.load(Ordering::Acquire);
        {
            let mut history = self.history.lock().unwrap_or_else(PoisonError::into_inner);
            history.clear();
            history.take(&reply, step);
        }
        self.observe(0.0, false, false, 0, Vec::new(), reply.info_json)
    }

    /// One view step: `actions` by action entry name.
    pub async fn step(&self, actions: &[(String, Value)]) -> Result<ViewStep, GymError> {
        // Every commanded part's actions, packed.
        let mut packed: Vec<(ChannelId, usize, Vec<u8>)> = Vec::new();
        let mut horizon = 1_usize;
        for output in &self.outputs {
            let Some(part) = output.part() else { continue };
            let Some(command) = &part.command else {
                continue;
            };
            let value = actions
                .iter()
                .find(|(name, _)| name == output.name())
                .map(|(_, v)| v)
                .ok_or_else(|| GymError::Malformed(format!("no action {}", output.name())))?;
            let stride = command
                .layout()
                .map_err(GymError::Malformed)?
                .stride()
                .unwrap_or(0);
            let bytes = match output {
                Output::Raw { .. } => pack_raw(&command.fields, value)?,
                Output::Bound(row) => {
                    let value = if row.horizon == 1 {
                        unsqueeze(value)
                    } else {
                        value.clone()
                    };
                    row.pack(&value).map_err(|e| GymError::Malformed(e.0))?
                }
            };
            let count = bytes.len() / stride.max(1);
            horizon = horizon.max(count);
            packed.push((part.channel, stride, bytes));
        }
        let mut reward = 0.0;
        let (mut terminated, mut truncated) = (false, false);
        let mut info_json = String::new();
        let mut taken = 0;
        for k in 0..horizon {
            let commands = packed
                .iter()
                .map(|(channel, stride, bytes)| {
                    // A shorter chunk holds its last action.
                    let count = bytes.len() / stride.max(&1);
                    let i = k.min(count.saturating_sub(1));
                    PartValues {
                        channel: *channel,
                        payload: Bytes::copy_from_slice(&bytes[i * stride..(i + 1) * stride]),
                    }
                })
                .collect();
            let n = self.step.fetch_add(1, Ordering::AcqRel) + 1;
            let step = LockstepStep {
                epoch: Epoch(self.epoch.load(Ordering::Acquire)),
                step: n,
                lease_token: LeaseToken(self.token.load(Ordering::Acquire)),
                lanes: Vec::new(),
                commands,
            };
            let reply = self.link.call(Request::Step(step), None).await?;
            reward += first_reward(&reply.reward);
            terminated = reply.terminated.first().is_some_and(|b| *b != 0);
            truncated = reply.truncated.first().is_some_and(|b| *b != 0);
            self.history
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .take(&reply, n);
            info_json = reply.info_json;
            taken += 1;
            if terminated || truncated {
                break;
            }
        }
        let modes = self
            .outputs
            .iter()
            .map(|o| {
                let mode = if o.part().is_some() {
                    "driving"
                } else {
                    "idle"
                };
                (o.name().to_string(), mode)
            })
            .collect();
        self.observe(reward, terminated, truncated, taken, modes, info_json)
    }

    fn observe(
        &self,
        reward: f64,
        terminated: bool,
        truncated: bool,
        steps: u32,
        edge_modes: Vec<(String, &'static str)>,
        info_json: String,
    ) -> Result<ViewStep, GymError> {
        let history = self.history.lock().unwrap_or_else(PoisonError::into_inner);
        let now = self.step.load(Ordering::Acquire);
        let mut observation = Vec::with_capacity(self.inputs.len());
        let mut ages = Vec::with_capacity(self.inputs.len());
        let mut stale = Vec::new();
        for input in &self.inputs {
            let (name, channel, max_age) = match input {
                Input::Raw { name, part } => (name.clone(), Some(part.channel), part.max_age),
                Input::Bound(row) => (row.entry.clone(), row.channel, row.max_age),
            };
            let held = channel.and_then(|c| history.samples.get(&c));
            let age = held.map(|(_, at)| now.saturating_sub(*at) * self.tick_ns);
            if let (Some(age), Some(bound)) = (age, max_age)
                && Duration::from_nanos(age) > bound
            {
                stale.push(name.clone());
            }
            let value = match (input, held) {
                (Input::Raw { part, .. }, Some((ring, _))) => {
                    let newest = ring.back().cloned().unwrap_or_default();
                    unpack_raw(&part.fields, &newest)?
                }
                (Input::Raw { part, .. }, None) => {
                    stale.push(name.clone());
                    zeros(&part.fields)
                }
                (Input::Bound(row), held) => {
                    let want = row.history.max(1) as usize;
                    let window: Vec<&[u8]> = match held {
                        Some((ring, _)) => {
                            let pad = want.saturating_sub(ring.len());
                            let first = ring.front().map(|b| b.as_ref()).unwrap_or_default();
                            std::iter::repeat_n(first, pad)
                                .chain(ring.iter().map(|b| b.as_ref()))
                                .collect()
                        }
                        None if row.part.is_none() => Vec::new(),
                        None => {
                            stale.push(name.clone());
                            return Err(GymError::Map(format!(
                                "input {name} has no sample yet (the robot never published {})",
                                row.part.as_ref().map_or("it", |p| p.path.as_str())
                            )));
                        }
                    };
                    row.apply(&window).map_err(|e| GymError::Malformed(e.0))?
                }
            };
            observation.push((name.clone(), value));
            ages.push((name, age));
        }
        Ok(ViewStep {
            observation,
            reward,
            terminated,
            truncated,
            steps,
            ages_ns: ages,
            stale,
            edge_modes,
            info_json,
        })
    }
}

fn first_reward(bytes: &[u8]) -> f64 {
    match bytes.len() {
        8 => f64::from_le_bytes(bytes.try_into().unwrap_or([0; 8])),
        4 => f64::from(f32::from_le_bytes(bytes.try_into().unwrap_or([0; 4]))),
        _ => 0.0,
    }
}

fn unsqueeze(value: &Value) -> Value {
    let lead = |t: &Tensor| Tensor {
        dtype: t.dtype,
        shape: std::iter::once(1).chain(t.shape.iter().copied()).collect(),
        data: t.data.clone(),
    };
    match value {
        Value::Tensor(t) => Value::Tensor(lead(t)),
        Value::Dict(fields) => {
            Value::Dict(fields.iter().map(|(n, t)| (n.clone(), lead(t))).collect())
        }
        Value::Text(t) => Value::Text(t.clone()),
    }
}

fn tensor_bytes<'v>(field: &FieldSpec, value: &'v Tensor) -> Result<&'v [u8], GymError> {
    let len = field.byte_len().unwrap_or(0) as usize;
    if value.dtype != field.dtype || value.data.len() != len {
        return Err(GymError::Malformed(format!(
            "{}: a {} {:?} array, the part takes {} {:?}",
            field.name,
            value.dtype.name(),
            value.shape,
            field.dtype.name(),
            field.shape
        )));
    }
    Ok(&value.data)
}

fn pack_raw(fields: &[FieldSpec], value: &Value) -> Result<Vec<u8>, GymError> {
    let layout = Layout::of(fields).map_err(GymError::Malformed)?;
    let parts: Vec<&[u8]> = match (fields, value) {
        ([field], Value::Tensor(t)) => vec![tensor_bytes(field, t)?],
        (fields, Value::Dict(values)) => fields
            .iter()
            .map(|f| {
                let t = values
                    .iter()
                    .find(|(n, _)| *n == f.name)
                    .map(|(_, t)| t)
                    .ok_or_else(|| GymError::Malformed(format!("no field {}", f.name)))?;
                tensor_bytes(f, t)
            })
            .collect::<Result<_, _>>()?,
        _ => return Err(GymError::Malformed("an action of the wrong form".into())),
    };
    layout.pack(&parts).map_err(GymError::Malformed)
}

fn unpack_raw(fields: &[FieldSpec], payload: &[u8]) -> Result<Value, GymError> {
    let layout = Layout::of(fields).map_err(GymError::Malformed)?;
    if matches!(layout, Layout::Text) {
        return Ok(Value::Text(String::from_utf8_lossy(payload).into_owned()));
    }
    if layout.stride() != Some(payload.len()) {
        return Err(GymError::Malformed(format!(
            "a {} byte sample, the part packs {:?}",
            payload.len(),
            layout.stride()
        )));
    }
    let tensor = |i: usize, f: &FieldSpec| Tensor {
        dtype: f.dtype,
        shape: f.shape.clone(),
        data: layout
            .field(i)
            .map(|r| payload[r].to_vec())
            .unwrap_or_default(),
    };
    Ok(match fields {
        [field] => Value::Tensor(tensor(0, field)),
        fields => Value::Dict(
            fields
                .iter()
                .enumerate()
                .map(|(i, f)| (f.name.clone(), tensor(i, f)))
                .collect(),
        ),
    })
}

fn zeros(fields: &[FieldSpec]) -> Value {
    let tensor = |f: &FieldSpec| Tensor {
        dtype: f.dtype,
        shape: f.shape.clone(),
        data: vec![0; f.byte_len().unwrap_or(0) as usize],
    };
    match fields {
        [field] if field.dtype == DType::Utf8 => Value::Text(String::new()),
        [field] => Value::Tensor(tensor(field)),
        fields => Value::Dict(fields.iter().map(|f| (f.name.clone(), tensor(f))).collect()),
    }
}

/// The view's binder options for `robot`.
pub fn run_options(name: &str, robot: &ParticipantId) -> RunOptions {
    RunOptions {
        name: name.to_string(),
        robot: Some(robot.clone()),
        ..RunOptions::default()
    }
}
