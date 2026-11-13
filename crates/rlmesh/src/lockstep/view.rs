//! The Gym view over a v2-native robot in lockstep.
//!
//! [`GymView`] drives a robot that serves the lockstep exchange (a
//! [`LockstepHandler`](crate::lockstep::LockstepHandler) behind [`serve_robot`](crate::lockstep::LockstepBinding::serve_robot))
//! and presents it as a Gymnasium env:
//!
//! - the observation space is keyed by the policy's input names, or by part
//!   path (`/` → `.`) without a policy: a single-field part is its field, a
//!   multi-field part a dict of fields;
//! - the action space is keyed by action names (`[chunk, ...]`, squeezed
//!   when the chunk is 1), or by commandable part path without a policy;
//! - one view step with chunk H is H lockstep steps (stopping at the end
//!   of the episode), rewards summed;
//! - `info` carries each input's age (in session time: steps × `tick_ns`,
//!   one action period of the policy per step, 1 ms without one),
//!   the stale inputs, each action's edge mode (`driving` for a part
//!   commanded this step, else `idle`; lockstep has no hold or safe), and
//!   how many of each action's commands the part's limits changed
//!   (`clamped`: the robot applies them, as a wall-clock edge would);
//! - without a policy, an action's `Box` carries its command field's
//!   declared limits, and an observation's its part's declared `range`.
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
use rlmesh_session::controller::StepAction;
use rlmesh_session::edge::command::CommandCheck;
use rlmesh_session::spec::apply::{Tensor, Value};
use rlmesh_session::spec::bind::{ActionRow, Binding, InputRow};
use rlmesh_session::spec::normalise::attrs;
use rlmesh_session::spec::plan::Delivery;
use rlmesh_session::spec::{Policy, RunOptions};
use rlmesh_session::wire::{
    ChannelId, DType, Epoch, FieldSpec, Layout, LeaseToken, LockstepReply, LockstepReset,
    LockstepStep, PartSpec, PartValues, ParticipantId,
};
use serde_json::Value as Json;

use rlmesh_spaces::{
    BoxBounds, BoxSpec, DictSpec, ElementwiseBounds, MetaMap, MetaValue, Scalar, SpaceKind,
    SpaceSpec, SpaceValue, TextSpec, TypedUniformBounds, UniformBounds,
};

use crate::lockstep::client::Keep;
use crate::lockstep::error::GymError;
use crate::lockstep::exchange::{LockstepLink, Request};
use crate::lockstep::meta_json;

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
    /// Reward since the previous step (sent as `f64`, as the Gym
    /// compatibility path sends it, so `1/3` stays `1/3`).
    pub reward: f64,
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
            timing: None,
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
    /// Per action, how many of the commands sent this step its part's
    /// limits, norm bounds or binary snap changed.
    pub clamped: Vec<(String, u32)>,
    /// Per input, the window slots padded with its oldest sample (only
    /// those padded).
    pub padded: Vec<(String, u32)>,
    /// The inputs bound to no part (blank).
    pub blank: Vec<String>,
    /// Per commanding action, the actions played this step, packed back to
    /// back at the command's stride.
    pub played: Vec<(String, Bytes)>,
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
    /// Each output's command checks, as the robot runs them: to count what
    /// its limits change.
    checks: Vec<Option<CommandCheck>>,
    tick_ns: u64,
    epoch: AtomicU64,
    token: AtomicU64,
    step: AtomicU64,
    history: Mutex<History>,
    binding: Option<Box<Binding>>,
    /// The driver's participant, released (its lease freed) on close or
    /// drop.
    keep: Mutex<Option<Box<dyn Keep>>>,
}

impl Drop for GymView {
    /// A view dropped without [`GymView::close`] still detaches, in the
    /// background, so the next driver can take the lease at once.
    fn drop(&mut self) {
        let keep = self
            .keep
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(keep) = keep {
            crate::run::drive::runtime().spawn(keep.release());
        }
    }
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
        keep: Option<Box<dyn Keep>>,
    ) -> Result<Self, GymError> {
        let (inputs, outputs, binding, depth): (Vec<Input>, Vec<Output>, _, _) = match policy {
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
                let binding = rlmesh_session::spec::bind(policy, parts, &options)
                    .map_err(|e| GymError::Map(e.to_string()))?;
                let mut depth = BTreeMap::new();
                for row in &binding.inputs {
                    if let Some(channel) = row.channel {
                        let d = depth.entry(channel).or_insert(1_usize);
                        *d = (*d).max(row.history.max(1) as usize);
                    }
                }
                (
                    binding
                        .inputs
                        .iter()
                        .map(|r| Input::Bound(Box::new(r.clone())))
                        .collect(),
                    binding
                        .actions
                        .iter()
                        .map(|r| Output::Bound(Box::new(r.clone())))
                        .collect(),
                    Some(Box::new(binding)),
                    depth,
                )
            }
        };
        let tick_ns = binding.as_deref().map_or(1_000_000, tick_of);
        let checks = outputs
            .iter()
            .map(|o| CommandCheck::new(o.part()?.command.as_ref()?))
            .collect();
        Ok(Self {
            link,
            inputs,
            outputs,
            checks,
            tick_ns,
            epoch: AtomicU64::new(epoch.0),
            token: AtomicU64::new(token.0),
            step: AtomicU64::new(0),
            history: Mutex::new(History {
                samples: BTreeMap::new(),
                depth,
            }),
            binding,
            keep: Mutex::new(keep),
        })
    }

    /// Releases the lease and detaches; later steps fail.
    pub async fn close(&self) {
        let keep = self
            .keep
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(keep) = keep {
            keep.release().await;
        }
    }

    /// The binding table (with a policy).
    pub fn table(&self) -> Option<String> {
        self.binding.as_deref().map(Binding::table_text)
    }

    /// The binding (with a policy).
    pub fn binding(&self) -> Option<&Binding> {
        self.binding.as_deref()
    }

    /// Session time per lockstep step, in ns.
    pub fn tick_ns(&self) -> u64 {
        self.tick_ns
    }

    /// Session time now: the steps taken so far, at [`Self::tick_ns`] each.
    pub fn time_ns(&self) -> i64 {
        let now = self
            .step
            .load(Ordering::Acquire)
            .saturating_mul(self.tick_ns);
        i64::try_from(now).unwrap_or(i64::MAX)
    }

    /// `step`'s actions as each part played them, for a [`StepAction`]
    /// report, the chunk's first action at `t0_ns`.
    pub fn played(&self, step: &ViewStep, t0_ns: i64) -> Vec<StepAction> {
        self.outputs
            .iter()
            .map(|output| {
                let command = output.part().and_then(|p| Some((p, p.command.as_ref()?)));
                let packed = step
                    .played
                    .iter()
                    .find(|(name, _)| name == output.name())
                    .map(|(_, bytes)| bytes.clone())
                    .unwrap_or_default();
                StepAction::played(
                    output.name().to_string(),
                    command.map(|(part, _)| part.qualified()),
                    t0_ns,
                    self.tick_ns,
                    packed,
                    command.map(|(_, c)| c.fields.clone()).unwrap_or_default(),
                )
            })
            .collect()
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
                        entry_of_delivery(&row.delivers, row.chunk == 1),
                    )
                }),
            })
            .collect()
    }

    /// The observation and action spaces, keyed by entry: an array entry
    /// a `Box` spanning its dtype (a float's unbounded), a text entry a
    /// `Text`, a multi-field entry a `Dict` of its fields. Without a
    /// policy, a float action field's `Box` is its declared limits and a
    /// float observation field's its part's declared `range`; a policy's
    /// entries may be converted from the part's, so they keep the dtype's
    /// span.
    pub fn spaces(&self) -> (SpaceSpec, SpaceSpec) {
        let observation = self.inputs.iter().map(|input| match input {
            Input::Raw { name, part } => {
                let ranges = ranges(part);
                let bounds = |field: &str| ranges.get(field).map(|r| vec![*r]);
                let entry = entry_of(&part.fields);
                (name.clone(), entry_space(&entry, &part.fields, bounds))
            }
            Input::Bound(row) => (
                row.entry.clone(),
                entry_space(&entry_of_delivery(&row.delivers, false), &[], |_| None),
            ),
        });
        let action = self.outputs.iter().filter_map(|output| match output {
            Output::Raw { name, part } => part.command.as_ref().map(|command| {
                let bounds = |field: &str| {
                    let limit = command.limit(field)?;
                    let pairs: Vec<(f64, f64)> = limit
                        .low
                        .iter()
                        .copied()
                        .zip(limit.high.iter().copied())
                        .collect();
                    (!pairs.is_empty()).then_some(pairs)
                };
                let entry = entry_of(&command.fields);
                (name.clone(), entry_space(&entry, &command.fields, bounds))
            }),
            Output::Bound(row) => row.part.as_ref().map(|_| {
                let entry = entry_of_delivery(&row.delivers, row.chunk == 1);
                (row.entry.clone(), entry_space(&entry, &[], |_| None))
            }),
        });
        let space = |entries: Vec<(String, SpaceSpec)>| {
            let (keys, spaces) = entries.into_iter().unzip();
            dict_space(keys, spaces)
        };
        (space(observation.collect()), space(action.collect()))
    }

    /// Lane 0's frame (PNG bytes), if the robot renders one.
    pub async fn render(&self) -> Result<Option<Bytes>, GymError> {
        self.link.render(0, None).await
    }

    /// Resets (`seed`: one seed; `options_json`: Gym options).
    pub async fn reset(&self, seed: Option<u64>, options_json: &str) -> Result<ViewStep, GymError> {
        let reset = LockstepReset {
            epoch: Epoch(self.epoch.load(Ordering::Acquire)),
            lanes: Vec::new(),
            seeds: seed.into_iter().collect(),
            options_json: options_json.to_string(),
            episode_ids: Vec::new(),
            lease_token: LeaseToken(self.token.load(Ordering::Acquire)),
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
        self.play(actions, false).await
    }

    /// [`Self::step`] with every bound action as a policy returns it:
    /// `[chunk, ...]`, a chunk of 1 included.
    pub async fn step_chunks(&self, actions: &[(String, Value)]) -> Result<ViewStep, GymError> {
        self.play(actions, true).await
    }

    async fn play(&self, actions: &[(String, Value)], chunked: bool) -> Result<ViewStep, GymError> {
        // Every commanded part's actions, packed, by action name.
        let mut packed: Vec<(ChannelId, usize, Vec<u8>, Option<&CommandCheck>)> = Vec::new();
        let mut names: Vec<&str> = Vec::new();
        let mut chunk = 1_usize;
        for (output, check) in self.outputs.iter().zip(&self.checks) {
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
                    let value = if row.chunk == 1 && !chunked {
                        unsqueeze(value)
                    } else {
                        value.clone()
                    };
                    row.pack(&value).map_err(|e| GymError::Malformed(e.0))?
                }
            };
            let count = bytes.len() / stride.max(1);
            chunk = chunk.max(count);
            packed.push((part.channel, stride, bytes, check.as_ref()));
            names.push(output.name());
        }
        let mut reward = 0.0;
        let (mut terminated, mut truncated) = (false, false);
        let mut info_json = String::new();
        let mut taken = 0;
        let mut clamped = vec![0_u32; packed.len()];
        for k in 0..chunk {
            let commands = packed
                .iter()
                .zip(&mut clamped)
                .map(|((channel, stride, bytes, check), clamped)| {
                    // A shorter chunk holds its last action.
                    let count = bytes.len() / stride.max(&1);
                    let i = k.min(count.saturating_sub(1));
                    let payload = Bytes::copy_from_slice(&bytes[i * stride..(i + 1) * stride]);
                    // The robot applies its limits; this only counts what
                    // they change (an invalid action is the robot's to
                    // refuse).
                    if let Some(Ok(checked)) = check.map(|c| c.check(&payload, 1)) {
                        *clamped += checked.clamped;
                    }
                    PartValues {
                        channel: *channel,
                        payload,
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
        // What each part played: its first `taken` actions (a shorter chunk
        // held its last).
        let played = names
            .iter()
            .zip(&packed)
            .map(|(name, (_, stride, bytes, _))| {
                let count = bytes.len() / stride.max(&1);
                let played: Vec<u8> = (0..taken as usize)
                    .flat_map(|k| {
                        let i = k.min(count.saturating_sub(1));
                        bytes[i * stride..(i + 1) * stride].iter().copied()
                    })
                    .collect();
                ((*name).to_string(), Bytes::from(played))
            })
            .collect();
        let clamped = names.into_iter().map(str::to_string).zip(clamped).collect();
        let mut step = self.observe(reward, terminated, truncated, taken, modes, info_json)?;
        step.clamped = clamped;
        step.played = played;
        Ok(step)
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
        let mut padded = Vec::new();
        let mut blank = Vec::new();
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
                            if pad > 0 {
                                padded.push((name.clone(), pad as u32));
                            }
                            let first = ring.front().map(|b| b.as_ref()).unwrap_or_default();
                            std::iter::repeat_n(first, pad)
                                .chain(ring.iter().map(|b| b.as_ref()))
                                .collect()
                        }
                        None if row.part.is_none() => {
                            blank.push(name.clone());
                            Vec::new()
                        }
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
            clamped: Vec::new(),
            padded,
            blank,
            played: Vec::new(),
            info_json,
        })
    }
}

/// A bound policy's session time per step: its shortest action period, or
/// its replan period when it commands nothing.
fn tick_of(binding: &Binding) -> u64 {
    let period = binding
        .actions
        .iter()
        .filter(|a| a.part.is_some())
        .map(|a| a.dt)
        .min()
        .unwrap_or_else(|| Duration::from_secs_f64(1.0 / binding.replan_hz.max(f64::MIN_POSITIVE)));
    u64::try_from(period.as_nanos()).unwrap_or(u64::MAX).max(1)
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

fn dict_space(keys: Vec<String>, spaces: Vec<SpaceSpec>) -> SpaceSpec {
    SpaceSpec {
        shape: Vec::new(),
        dtype: rlmesh_spaces::DType::Unspecified,
        spec: Some(SpaceKind::Dict(DictSpec { keys, spaces })),
    }
}

/// A part's declared `range`: `(low, high)` by field.
fn ranges(part: &PartSpec) -> BTreeMap<String, (f64, f64)> {
    let Some(Ok(Json::Object(ranges))) = part
        .attrs
        .get(attrs::RANGE)
        .map(|text| serde_json::from_str::<Json>(text))
    else {
        return BTreeMap::new();
    };
    ranges
        .into_iter()
        .filter_map(|(field, pair)| Some((field, (pair.get(0)?.as_f64()?, pair.get(1)?.as_f64()?))))
        .collect()
}

/// Float bounds of `numel` elements from `pairs`: one pair broadcasts,
/// else one per element (anything else is left unbounded).
fn float_bounds(pairs: &[(f64, f64)], numel: usize) -> Option<BoxBounds> {
    match pairs {
        [(low, high)] => Some(BoxBounds::Uniform(UniformBounds {
            low: *low,
            high: *high,
        })),
        pairs if pairs.len() == numel => Some(BoxBounds::Elementwise(ElementwiseBounds {
            low: pairs.iter().map(|p| p.0).collect(),
            high: pairs.iter().map(|p| p.1).collect(),
        })),
        _ => None,
    }
}

/// An array's `Box`: `bounds` (one pair, or one per element) on a
/// `float32`/`float64` field, else its dtype's span. A `float16` keeps its
/// span: the edge does not clamp it.
fn array_space(dtype: DType, shape: &[u32], bounds: Option<Vec<(f64, f64)>>) -> SpaceSpec {
    let native = rlmesh_spaces::DType::from_name(dtype.name()).unwrap_or_default();
    let numel = shape.iter().map(|d| *d as usize).product();
    let declared = bounds
        .filter(|_| matches!(dtype, DType::F32 | DType::F64))
        .and_then(|pairs| float_bounds(&pairs, numel));
    let typed = |low: Scalar, high: Scalar| {
        let bytes = |s| rlmesh_spaces::encode_scalars(&[s], native).unwrap_or_default();
        BoxBounds::TypedUniform(TypedUniformBounds {
            low: bytes(low),
            high: bytes(high),
        })
    };
    let bounds = match (declared, dtype) {
        (Some(declared), _) => declared,
        (None, DType::F16 | DType::F32 | DType::F64) => BoxBounds::Uniform(UniformBounds {
            low: f64::NEG_INFINITY,
            high: f64::INFINITY,
        }),
        (None, DType::Bool) => typed(Scalar::Bool(false), Scalar::Bool(true)),
        // The dtype's own range; `u64`'s top wraps to -1, as its scalars do.
        (None, DType::U64) => typed(Scalar::Int(0), Scalar::Int(-1)),
        (None, _) => {
            let bits = dtype.size() * 8;
            let signed = matches!(dtype, DType::I8 | DType::I16 | DType::I32 | DType::I64);
            let (low, high) = if signed {
                (-(1_i128 << (bits - 1)), (1_i128 << (bits - 1)) - 1)
            } else {
                (0, (1_i128 << bits) - 1)
            };
            typed(Scalar::Int(low as i64), Scalar::Int(high as i64))
        }
    };
    SpaceSpec {
        shape: shape.iter().map(|d| i64::from(*d)).collect(),
        dtype: native,
        spec: Some(SpaceKind::Box(BoxSpec {
            bounds: Some(bounds),
        })),
    }
}

/// An entry's space, its fields' `bounds` by field name (an array entry's
/// field is `fields`' only one).
fn entry_space(
    entry: &Entry,
    fields: &[FieldSpec],
    bounds: impl Fn(&str) -> Option<Vec<(f64, f64)>>,
) -> SpaceSpec {
    match entry {
        Entry::Array { dtype, shape } => {
            let field = match fields {
                [field] => bounds(&field.name),
                _ => None,
            };
            array_space(*dtype, shape, field)
        }
        Entry::Text => SpaceSpec {
            shape: Vec::new(),
            dtype: rlmesh_spaces::DType::Uint8,
            spec: Some(SpaceKind::Text(TextSpec {
                min_length: 1,
                max_length: 4096,
                charset: String::new(),
            })),
        },
        Entry::Dict(fields) => dict_space(
            fields.iter().map(|(name, _, _)| name.clone()).collect(),
            fields
                .iter()
                .map(|(name, dtype, shape)| array_space(*dtype, shape, bounds(name)))
                .collect(),
        ),
    }
}

fn tensor_value(tensor: Tensor) -> Result<SpaceValue, GymError> {
    let dtype = rlmesh_spaces::DType::from_name(tensor.dtype.name())
        .ok_or_else(|| GymError::Malformed(format!("dtype {}", tensor.dtype.name())))?;
    let shape = tensor.shape.iter().map(|d| i64::from(*d)).collect();
    rlmesh_spaces::Tensor::from_vec(tensor.data, shape, dtype)
        .map(SpaceValue::Box)
        .map_err(|e| GymError::Malformed(e.to_string()))
}

fn value_tensor(value: &SpaceValue, what: &str) -> Result<Tensor, GymError> {
    let SpaceValue::Box(tensor) = value else {
        return Err(GymError::Value(format!("{what}: not an array")));
    };
    Ok(Tensor {
        dtype: DType::from_name(tensor.dtype().name())
            .ok_or_else(|| GymError::Value(format!("{what}: dtype {}", tensor.dtype())))?,
        shape: tensor.shape().iter().map(|d| *d as u32).collect(),
        data: tensor.to_contiguous_bytes().into_owned(),
    })
}

/// A view step's observation as a value of [`GymView::spaces`]' observation
/// space.
pub fn observation_value(observation: Vec<(String, Value)>) -> Result<SpaceValue, GymError> {
    observation
        .into_iter()
        .map(|(name, value)| {
            let value = match value {
                Value::Tensor(tensor) => tensor_value(tensor)?,
                Value::Text(text) => SpaceValue::Text(text),
                Value::Dict(fields) => SpaceValue::Dict(
                    fields
                        .into_iter()
                        .map(|(field, tensor)| Ok((field, tensor_value(tensor)?)))
                        .collect::<Result<_, GymError>>()?,
                ),
            };
            Ok((name, value))
        })
        .collect::<Result<_, _>>()
        .map(SpaceValue::Dict)
}

/// A value of [`GymView::spaces`]' action space as [`GymView::step`]'s
/// actions.
pub fn action_values(action: &SpaceValue) -> Result<Vec<(String, Value)>, GymError> {
    let SpaceValue::Dict(entries) = action else {
        return Err(GymError::Value(
            "an action is a dict by action entry".into(),
        ));
    };
    entries
        .iter()
        .map(|(name, value)| {
            let value = match value {
                SpaceValue::Dict(fields) => Value::Dict(
                    fields
                        .iter()
                        .map(|(field, v)| Ok((field.clone(), value_tensor(v, field)?)))
                        .collect::<Result<_, GymError>>()?,
                ),
                SpaceValue::Text(text) => Value::Text(text.clone()),
                other => Value::Tensor(value_tensor(other, name)?),
            };
            Ok((name.clone(), value))
        })
        .collect()
}

/// A view step's `info`: `ages_ns`, `stale`, `edge_modes`, `clamped` and
/// `steps`, then the robot's own keys.
pub fn info(step: &ViewStep) -> Result<MetaMap, GymError> {
    let mut info = meta_json::from_json(&step.info_json).map_err(GymError::Malformed)?;
    let map = |pairs: Vec<(String, MetaValue)>| MetaValue::Map(pairs.into_iter().collect());
    info.insert(
        "ages_ns".into(),
        map(step
            .ages_ns
            .iter()
            .map(|(name, age)| {
                let age = age.map_or(MetaValue::Null, |a| MetaValue::Int(a as i64));
                (name.clone(), age)
            })
            .collect()),
    );
    info.insert(
        "stale".into(),
        MetaValue::List(step.stale.iter().cloned().map(MetaValue::String).collect()),
    );
    info.insert(
        "edge_modes".into(),
        map(step
            .edge_modes
            .iter()
            .map(|(name, mode)| (name.clone(), MetaValue::String((*mode).into())))
            .collect()),
    );
    info.insert(
        "clamped".into(),
        map(step
            .clamped
            .iter()
            .map(|(name, n)| (name.clone(), MetaValue::Int(i64::from(*n))))
            .collect()),
    );
    info.insert("steps".into(), MetaValue::Int(i64::from(step.steps)));
    Ok(info)
}

/// The view's binder options for `robot`.
pub fn run_options(name: &str, robot: &ParticipantId) -> RunOptions {
    RunOptions {
        name: name.to_string(),
        robot: Some(robot.clone()),
        ..RunOptions::default()
    }
}
