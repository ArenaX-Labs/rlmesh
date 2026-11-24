//! Per-episode host state around a Gym binding: episode-keyed frame-stacking and the per-lane
//! assemble/apply seam.
//!
//! Frame-stacking lives here keyed by `episode_id`, so a vectorized serve route stacks each lane
//! correctly and the in-process path shares the same ring. [`assemble_obs`] and [`apply_actions`]
//! are the per-lane seam the core handler drives once per lane; they never grow a batch axis.
//! The host-language holes cross via traits the binding implements: custom inputs via
//! [`CustomTransform`], and host-side custom encodings via [`EncodingTransform`].

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use rlmesh_spaces::scalar::encode_i64_scalars;
use rlmesh_spaces::{SpaceKind, SpaceSpec, SpaceValue, Tensor};

use crate::apply::value::{cast, to_f64_vec};
use crate::apply::{CustomTransform, Value};
use crate::error::ApplyError;
use crate::path::NodePath;
use crate::plans::{Binding as ResolvedAdapter, ImagePlan, ObsPlan, StackedPlacement};
use rlmesh_session_core::spec::role::input::StackPad;

// Mirrors the spec's `MAX_STACK`; bounds in-process callers so a window cannot grow without limit.
const MAX_STACK: usize = 64;

/// Host-language hook applying custom *encodings* the native core left in a base
/// encoding (e.g. a rotation re-packing the spec declared as a `CustomEncoding`,
/// which resolution shadows to a native base encoding).
///
/// Distinct from [`CustomTransform`], which fills whole custom inputs: an encoding transform
/// repacks an existing payload key (obs) or action segment in place, and owns which keys or
/// segments it touches.
pub trait EncodingTransform {
    fn repack_obs(&self, payload: &mut Value) -> Result<(), ApplyError>;

    fn repack_action(&self, action: &mut Value) -> Result<(), ApplyError>;
}

/// An [`EncodingTransform`] that touches nothing — for fully declarative routes
/// (no custom encodings).
pub struct NoEncodings;

impl EncodingTransform for NoEncodings {
    fn repack_obs(&self, _payload: &mut Value) -> Result<(), ApplyError> {
        Ok(())
    }

    fn repack_action(&self, _action: &mut Value) -> Result<(), ApplyError> {
        Ok(())
    }
}

/// Per-route, episode-keyed frame-history buffers:
/// `episode_id -> placement -> rolling window` (window `maxlen = depth`).
///
/// An episode's entry appears on first use and is dropped only at episode end
/// ([`evict`](Self::evict)) or on close ([`clear`](Self::clear)), never on absence. Keyed by
/// `episode_id`, not lane index, so autoreset reusing a lane and grouped predict moving an
/// episode between slots both keep each episode's window intact.
#[derive(Default)]
pub struct FrameBuffers {
    // Keyed by `episode_id`, not lane index: autoreset reuses a lane's index across episodes,
    // and grouped predict can move an episode between slots.
    inner: HashMap<String, EpisodeWindows>,
}

#[derive(Default)]
struct EpisodeWindows {
    windows: BTreeMap<NodePath, Window>,
    last_step: Option<i64>,
    /// The raw model action executed at each step, in model order, recorded
    /// by [`record_action`]; an observation at step `s` reads row `s - 1`.
    actions: BTreeMap<i64, Vec<f32>>,
    /// The step of the episode's first assembled observation, which reads the
    /// parts' `fill` (no action has executed yet).
    first_step: Option<i64>,
}

impl FrameBuffers {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Evict an episode's buffers at episode END or a close sweep.
    pub fn evict(&mut self, episode_id: &str) {
        self.inner.remove(episode_id);
    }

    /// Drop every episode's buffers (session shutdown / route close).
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Number of episodes currently holding frame windows.
    #[must_use]
    pub fn episodes(&self) -> usize {
        self.inner.len()
    }

    /// Whether `episode_id` already holds buffers here (its next frame does
    /// not open a new window).
    #[must_use]
    pub fn holds(&self, episode_id: &str) -> bool {
        self.inner.contains_key(episode_id)
    }

    /// Total bytes held across every episode's frame and action windows.
    #[must_use]
    pub fn state_bytes(&self) -> u64 {
        self.inner
            .values()
            .map(|episode| {
                let frames: u64 = episode
                    .windows
                    .values()
                    .flat_map(|window| window.frames.iter())
                    .map(|tensor| tensor.nbytes() as u64)
                    .sum();
                let actions: u64 = episode
                    .actions
                    .values()
                    .map(|row| (row.len() * std::mem::size_of::<f32>()) as u64)
                    .sum();
                frames + actions
            })
            .sum()
    }

    /// The step the episode's last frame carried (`None` before its first
    /// [`advance_step`](Self::advance_step)).
    #[must_use]
    pub fn last_step(&self, episode_id: &str) -> Option<i64> {
        self.inner
            .get(episode_id)
            .and_then(|episode| episode.last_step)
    }

    /// The step at which the next chunk this episode returns starts executing:
    /// one past its last recorded action, or the last observed step when none
    /// is recorded yet. The two differ under an async prefetch lead, where the request's
    /// observation is `lead` steps before the chunk it asks for begins.
    #[must_use]
    pub fn chunk_start(&self, episode_id: &str) -> i64 {
        let Some(episode) = self.inner.get(episode_id) else {
            return 0;
        };
        let observed = episode.last_step.unwrap_or(0);
        episode
            .actions
            .keys()
            .next_back()
            .map_or(observed, |last| observed.max(last + 1))
    }

    /// The step the episode's next frame carries when the caller counts
    /// steps itself: one past the last, or `0` for an episode not yet held.
    #[must_use]
    pub fn next_step(&self, episode_id: &str) -> i64 {
        self.last_step(episode_id).map_or(0, |last| last + 1)
    }

    fn previous_action(
        &mut self,
        episode_id: &str,
        step: i64,
    ) -> Result<Option<&[f32]>, ApplyError> {
        let episode = self.inner.entry(episode_id.to_owned()).or_default();
        let first = *episode.first_step.get_or_insert(step);
        if first == step {
            return Ok(None);
        }
        let wanted = step - 1;
        episode.actions.retain(|&recorded, _| recorded >= wanted);
        match episode.actions.get(&wanted) {
            Some(row) => Ok(Some(row.as_slice())),
            None => Err(ApplyError::new(format!(
                "episode {episode_id}: the observation at step {step} reads the previous action \
                 but no action was recorded for step {wanted}; every executed action of the \
                 episode must reach apply_actions (or transform_action) once, under its step"
            ))),
        }
    }

    /// Record that the episode's next frame carries `step`, holding a runtime
    /// that delivers history to consecutive steps. Call before pushing the frame, once per
    /// (episode, step).
    ///
    /// # Errors
    ///
    /// An episode already held must advance by exactly one; a gap or a repeat is an error. An
    /// episode not yet held accepts any step, so a cold start or a mid-episode re-attach is fine.
    pub fn advance_step(&mut self, episode_id: &str, step: i64) -> Result<(), ApplyError> {
        let episode = self.inner.entry(episode_id.to_owned()).or_default();
        if let Some(last) = episode.last_step
            && step != last + 1
        {
            return Err(ApplyError::new(format!(
                "observation history for episode {episode_id} is not consecutive: step {step} \
                 arrived after step {last} (expected {}); every env step of the episode must \
                 reach the model exactly once, in order",
                last + 1
            )));
        }
        episode.last_step = Some(step);
        Ok(())
    }

    fn episode(&mut self, episode_id: &str) -> &mut BTreeMap<NodePath, Window> {
        &mut self.inner.entry(episode_id.to_owned()).or_default().windows
    }
}

#[derive(Debug, Default)]
struct Window {
    frames: VecDeque<Tensor>,
}

impl Window {
    fn push(
        &mut self,
        frame: Tensor,
        entry: &StackedPlacement,
        plan: &ImagePlan,
    ) -> Result<(), ApplyError> {
        let span = entry.span as usize;
        debug_assert!(
            entry.depth as usize <= MAX_STACK,
            "stack depth {} exceeds {MAX_STACK}",
            entry.depth
        );
        if self.frames.is_empty() {
            let pad = match entry.stack_pad {
                StackPad::First => frame.clone(),
                StackPad::Black => black_frame(&frame, plan)?,
            };
            for _ in 0..span.saturating_sub(1) {
                self.frames.push_back(pad.clone());
            }
        }
        if self.frames.len() == span {
            self.frames.pop_front();
        }
        self.frames.push_back(frame);
        Ok(())
    }

    fn gather(&mut self, entry: &StackedPlacement) -> Result<Tensor, ApplyError> {
        let stacked = match &entry.offsets {
            None => Tensor::stack(self.frames.make_contiguous()),
            Some(offsets) => {
                let newest = self.frames.len().saturating_sub(1);
                let picked = offsets
                    .iter()
                    .map(|&offset| {
                        newest
                            .checked_sub(offset.unsigned_abs() as usize)
                            .and_then(|index| self.frames.get(index))
                            .cloned()
                            .ok_or_else(|| {
                                ApplyError::new(format!(
                                    "frame window for '{}' holds {} frame(s), too few to reach \
                                     offset {offset}",
                                    entry.key,
                                    self.frames.len()
                                ))
                            })
                    })
                    .collect::<Result<Vec<Tensor>, ApplyError>>()?;
                Tensor::stack(&picked)
            }
        };
        stacked.map_err(|err| ApplyError::new(format!("frame-stack failed: {err}")))
    }
}

#[cfg(test)]
impl<const N: usize> From<[Tensor; N]> for Window {
    fn from(frames: [Tensor; N]) -> Self {
        Self {
            frames: VecDeque::from(frames),
        }
    }
}

fn black_frame(frame: &Tensor, plan: &ImagePlan) -> Result<Tensor, ApplyError> {
    let level = plan.normalize.map_or(0.0, |(low, _)| f64::from(low as f32));
    crate::apply::value::encode_f64_to(
        vec![level; frame.numel()],
        frame.shape().to_vec(),
        frame.dtype(),
    )
}

/// Upper bound on the runtime's execution horizon `h` (how many actions of one
/// predicted chunk are executed before re-planning). Enforced at engine resolve and at the SDK
/// session seam; a larger pin is a configuration error.
pub const MAX_EXECUTION_HORIZON: u32 = 1024;

/// Split a chunked model action into its per-step actions (the leading axis is
/// the chunk axis).
///
/// A `Value::Tensor` of shape `[chunk, ..]` unstacks along axis 0; a 0-d tensor or any other
/// leaf is a single-step chunk; a `Value::List` is already per-step. A model that omits the chunk
/// axis is not always caught: a flat `[1]` against a dim-1 action space splits cleanly.
pub fn split_chunk(raw_action: Value) -> Result<Vec<Value>, ApplyError> {
    match raw_action {
        Value::Tensor(tensor) if tensor.shape().is_empty() => Ok(vec![Value::Tensor(tensor)]),
        Value::Tensor(tensor) => Ok(tensor
            .unstack()
            .map_err(|err| ApplyError::new(format!("action chunk split failed: {err}")))?
            .into_iter()
            .map(Value::Tensor)
            .collect()),
        Value::List(actions) => Ok(actions),
        Value::Map(map) => {
            let mut fields: Vec<(String, Vec<Value>)> = Vec::with_capacity(map.len());
            let mut horizon: Option<usize> = None;
            for (key, value) in map {
                let frames = split_chunk(value)?;
                match horizon {
                    Some(h) if h != frames.len() => {
                        return Err(ApplyError::new(format!(
                            "action chunk fields disagree on horizon: {h} vs {}",
                            frames.len()
                        )));
                    }
                    _ => horizon = Some(frames.len()),
                }
                fields.push((key, frames));
            }
            let horizon = horizon.unwrap_or(0);
            Ok((0..horizon)
                .map(|i| {
                    Value::Map(
                        fields
                            .iter()
                            .map(|(key, frames)| (key.clone(), frames[i].clone()))
                            .collect(),
                    )
                })
                .collect())
        }
        other => Ok(vec![other]),
    }
}

/// Assemble one lane's model-input payload from its raw observation at `step`.
///
/// In order: the declarative transform (customs dispatched inside it, an action-source part
/// reading the action recorded for `step - 1`, or its `fill` at the first step; a missing row is
/// an error), the observation encoding repack ([`EncodingTransform::repack_obs`]), then
/// per-episode frame-stacking. `step` is read only by an adapter that
/// [reads its previous action](ResolvedAdapter::reads_previous_action); otherwise a caller can
/// count steps with [`FrameBuffers::next_step`].
pub fn assemble_obs(
    adapter: &ResolvedAdapter,
    raw_obs: &BTreeMap<String, Value>,
    episode_id: &str,
    step: i64,
    buffers: &mut FrameBuffers,
    customs: &dyn CustomTransform,
    encodings: &dyn EncodingTransform,
) -> Result<Value, ApplyError> {
    let previous = if adapter.reads_previous_action() {
        buffers
            .previous_action(episode_id, step)?
            .map(<[f32]>::to_vec)
    } else {
        None
    };
    let mut payload = crate::apply::obs::transform_obs(
        &adapter.obs_plans,
        raw_obs,
        customs,
        previous.as_deref(),
    )?;
    encodings.repack_obs(&mut payload)?;
    let stacked = adapter.stacked_placements();
    if !stacked.is_empty() {
        let windows = buffers.episode(episode_id);
        for entry in stacked {
            let slot = crate::apply::lookup::resolve_source_mut(&mut payload, &entry.placement)?;
            if !matches!(slot, Value::Tensor(_)) {
                return Err(ApplyError::new(format!(
                    "frame-stacked input '{}' must be a tensor",
                    entry.key
                )));
            }
            let Value::Tensor(frame) = std::mem::replace(slot, Value::Number(0.0)) else {
                unreachable!("frame confirmed a tensor above")
            };
            let window = if windows.contains_key(&entry.placement) {
                windows
                    .get_mut(&entry.placement)
                    .expect("window present (just checked)")
            } else {
                windows.entry(entry.placement.clone()).or_default()
            };
            let ObsPlan::Image(image_plan) = &adapter.obs_plans[entry.plan_index] else {
                unreachable!("only image plans are ever stacked")
            };
            window.push(frame, entry, image_plan)?;
            *slot = Value::Tensor(window.gather(entry)?);
        }
    }
    Ok(payload)
}

/// Advance one lane's frame windows from a raw observation, assembling nothing.
///
/// Call it for an env step that replays a queued action, so the windows hold every step rather
/// than only the decision points. A route with no stacked input does nothing.
pub fn observe_obs(
    adapter: &ResolvedAdapter,
    raw_obs: &BTreeMap<String, Value>,
    episode_id: &str,
    buffers: &mut FrameBuffers,
) -> Result<(), ApplyError> {
    let stacked = adapter.stacked_placements();
    if stacked.is_empty() {
        return Ok(());
    }
    let frames = crate::apply::obs::observe_obs(&adapter.obs_plans, stacked, raw_obs)?;
    let windows = buffers.episode(episode_id);
    for (entry, frame) in stacked.iter().zip(frames) {
        let ObsPlan::Image(image_plan) = &adapter.obs_plans[entry.plan_index] else {
            unreachable!("only image plans are ever stacked")
        };
        windows
            .entry(entry.placement.clone())
            .or_default()
            .push(frame, entry, image_plan)?;
    }
    Ok(())
}

/// Convert one lane's model action into the env action [`SpaceValue`], and
/// record it as the action executed at `step`.
///
/// Runs the action encoding repack ([`EncodingTransform::repack_action`]) before the native
/// conversion. For chunk frame `k`, the caller passes `step + k`.
pub fn apply_actions(
    adapter: &ResolvedAdapter,
    raw_action: Value,
    action_space: &SpaceSpec,
    encodings: &dyn EncodingTransform,
    buffers: &mut FrameBuffers,
    episode_id: &str,
    step: i64,
) -> Result<SpaceValue, ApplyError> {
    let mut action = raw_action;
    encodings.repack_action(&mut action)?;
    let env_action = crate::apply::transform_action(adapter, &action)?;
    let value = tensor_to_space_value(env_action, action_space)?;
    record_action(adapter, &action, buffers, episode_id, step)?;
    Ok(value)
}

/// Record the raw model action executed at `step` for the episode's
/// action-source parts: the action as the action plan reads it (after any
/// host-side encoding repack, before any scale, offset, range or scatter), in
/// model order. A no-op for an adapter with no such part. The record half of
/// [`apply_actions`], for a caller that converts the action itself.
pub fn record_action(
    adapter: &ResolvedAdapter,
    raw_action: &Value,
    buffers: &mut FrameBuffers,
    episode_id: &str,
    step: i64,
) -> Result<(), ApplyError> {
    if !adapter.reads_previous_action() {
        return Ok(());
    }
    let row = crate::apply::lookup::numeric_vector(raw_action)?;
    if row.len() != adapter.action_plan.in_dim as usize {
        return Err(ApplyError::new(format!(
            "cannot record the action executed at step {step}: it has {} elements but the \
             action plan reads {}",
            row.len(),
            adapter.action_plan.in_dim
        )));
    }
    buffers
        .inner
        .entry(episode_id.to_owned())
        .or_default()
        .actions
        .insert(step, row);
    Ok(())
}

fn tensor_to_space_value(action: Tensor, space: &SpaceSpec) -> Result<SpaceValue, ApplyError> {
    let integral = |values: Vec<f64>, kind: &str| -> Result<Vec<i64>, ApplyError> {
        values
            .into_iter()
            .map(|value| {
                if value.is_finite() && value.fract() == 0.0 {
                    Ok(value as i64)
                } else {
                    Err(ApplyError::new(format!(
                        "{kind} action must be an integer, got {value}"
                    )))
                }
            })
            .collect()
    };
    match space.spec.as_ref() {
        Some(SpaceKind::Box(_)) => {
            let cast = cast(&action, space.dtype)?;
            let reshaped = cast
                .reshape(&space.shape)
                .map_err(|err| ApplyError::new(format!("action reshape: {err}")))?;
            Ok(SpaceValue::Box(reshaped))
        }
        Some(SpaceKind::Discrete(_)) => {
            let value = to_f64_vec(&action);
            let first = integral(value, "Discrete")?
                .into_iter()
                .next()
                .ok_or_else(|| {
                    ApplyError::new("Discrete action produced an empty vector".to_owned())
                })?;
            Ok(SpaceValue::Discrete(first))
        }
        Some(SpaceKind::MultiBinary(_)) => Ok(SpaceValue::MultiBinary(
            to_f64_vec(&action).into_iter().map(|v| v != 0.0).collect(),
        )),
        Some(SpaceKind::MultiDiscrete(_)) => Ok(SpaceValue::MultiDiscrete(integral(
            to_f64_vec(&action),
            "MultiDiscrete",
        )?)),
        Some(SpaceKind::Text(_) | SpaceKind::Dict(_) | SpaceKind::Tuple(_)) => {
            Err(ApplyError::new(
                "a resolved adapter cannot produce a text/dict/tuple action".to_owned(),
            ))
        }
        None => Err(ApplyError::new("action space spec is missing".to_owned())),
    }
}

/// Max absolute elementwise difference between two model action values, or
/// `None` if their shapes/types do not line up (the registration probe treats an
/// unmeasurable comparison conservatively).
pub fn value_max_abs_diff(left: &Value, right: &Value) -> Option<f64> {
    match (left, right) {
        (Value::Tensor(a), Value::Tensor(b)) => {
            if a.shape() != b.shape() {
                return None;
            }
            Some(
                to_f64_vec(a)
                    .iter()
                    .zip(to_f64_vec(b).iter())
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0_f64, f64::max),
            )
        }
        (Value::None, Value::None) => Some(0.0),
        (Value::Bool(a), Value::Bool(b)) => Some(if a == b { 0.0 } else { 1.0 }),
        (Value::Int(a), Value::Int(b)) => Some((i128::from(*a) - i128::from(*b)).abs() as f64),
        (Value::Number(a), Value::Number(b)) => Some((a - b).abs()),
        (Value::List(a), Value::List(b)) | (Value::Tuple(a), Value::Tuple(b))
            if a.len() == b.len() =>
        {
            let mut worst = 0.0_f64;
            for (x, y) in a.iter().zip(b.iter()) {
                worst = worst.max(value_max_abs_diff(x, y)?);
            }
            Some(worst)
        }
        (Value::Map(a), Value::Map(b)) if a.len() == b.len() => {
            let mut worst = 0.0_f64;
            for (key, x) in a {
                worst = worst.max(value_max_abs_diff(x, b.get(key)?)?);
            }
            Some(worst)
        }
        _ => None,
    }
}

/// Convert a decoded env-observation [`SpaceValue`] leaf into the adapter
/// [`Value`] payload model: a `Box` keeps its tensor, a `Discrete` becomes a number,
/// `MultiBinary`/`MultiDiscrete` become the space's dtyped tensor, and `Dict`/`Tuple` recurse
/// into a map/list.
pub fn space_value_to_value(value: &SpaceValue, space: &SpaceSpec) -> Result<Value, ApplyError> {
    match value {
        SpaceValue::Box(tensor) => Ok(Value::Tensor(tensor.clone())),
        SpaceValue::Discrete(value) => Ok(Value::Int(*value)),
        SpaceValue::Text(text) => Ok(Value::Text(text.clone())),
        SpaceValue::MultiBinary(values) => {
            let bytes: Vec<u8> = values.iter().map(|value| u8::from(*value)).collect();
            Tensor::from_vec(bytes, space.shape.clone(), space.dtype)
                .map(Value::Tensor)
                .map_err(|err| ApplyError::new(format!("multibinary observation leaf: {err}")))
        }
        SpaceValue::MultiDiscrete(values) => {
            let bytes = encode_i64_scalars(values, space.dtype)
                .map_err(|err| ApplyError::new(format!("multidiscrete observation leaf: {err}")))?;
            Tensor::from_vec(bytes, space.shape.clone(), space.dtype)
                .map(Value::Tensor)
                .map_err(|err| ApplyError::new(format!("multidiscrete observation leaf: {err}")))
        }
        SpaceValue::Dict(values) => {
            let Some(SpaceKind::Dict(spec)) = space.spec.as_ref() else {
                return Err(ApplyError::new(
                    "dict observation value without a dict space".to_owned(),
                ));
            };
            let mut out: BTreeMap<String, Value> = BTreeMap::new();
            for (key, child_space) in spec.keys.iter().zip(spec.spaces.iter()) {
                if let Some(child) = values.get(key) {
                    out.insert(key.clone(), space_value_to_value(child, child_space)?);
                }
            }
            Ok(Value::Map(out))
        }
        SpaceValue::Tuple(values) => {
            let Some(SpaceKind::Tuple(spec)) = space.spec.as_ref() else {
                return Err(ApplyError::new(
                    "tuple observation value without a tuple space".to_owned(),
                ));
            };
            let mut out: Vec<Value> = Vec::with_capacity(values.len());
            for (child, child_space) in values.iter().zip(spec.spaces.iter()) {
                out.push(space_value_to_value(child, child_space)?);
            }
            Ok(Value::Tuple(out))
        }
    }
}

/// Bridge a decoded env-observation [`SpaceValue`] into the adapter's raw-obs
/// envelope, keeping only the top-level entries `referenced` selects.
///
/// A `Dict` env yields the selected entries under their keys; any other env yields the whole
/// observation under the reserved root key. A declarative-only route passes
/// [`ResolvedAdapter::referenced_obs_keys`]; a route with custom inputs passes every top-level key
/// so the custom callback sees the full observation.
pub fn space_value_to_obs_map(
    value: &SpaceValue,
    space: &SpaceSpec,
    referenced: &BTreeSet<String>,
) -> Result<BTreeMap<String, Value>, ApplyError> {
    use crate::plans::OBS_ROOT_KEY;
    let mut out: BTreeMap<String, Value> = BTreeMap::new();
    match value {
        SpaceValue::Dict(values) => {
            let Some(SpaceKind::Dict(spec)) = space.spec.as_ref() else {
                return Err(ApplyError::new(
                    "dict observation value without a dict space".to_owned(),
                ));
            };
            for (key, child_space) in spec.keys.iter().zip(spec.spaces.iter()) {
                if referenced.contains(key.as_str())
                    && let Some(child) = values.get(key)
                {
                    // A referenced key equal to the reserved envelope key would collide with the root source's
                    // entry; unreferenced keys are dropped, never checked.
                    if key == OBS_ROOT_KEY {
                        return Err(ApplyError::new(format!(
                            "env observation has a top-level key '{OBS_ROOT_KEY}', which collides \
                             with the reserved whole-observation envelope key; rename the env key"
                        )));
                    }
                    out.insert(key.clone(), space_value_to_value(child, child_space)?);
                }
            }
        }
        _ => {
            out.insert(OBS_ROOT_KEY.to_owned(), space_value_to_value(value, space)?);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlmesh_spaces::{DType, DictSpec, SpaceKind, SpaceSpec};

    fn frame(base: u8) -> Tensor {
        Tensor::from_vec(vec![base, base + 1], vec![2], DType::Uint8).expect("frame")
    }

    fn flat(shape: Vec<i64>, dtype: DType) -> SpaceSpec {
        SpaceSpec {
            shape,
            dtype,
            spec: None,
        }
    }

    #[test]
    fn advance_step_holds_an_episode_to_consecutive_steps() {
        let mut buffers = FrameBuffers::new();
        buffers.advance_step("ep", 7).expect("first step");
        buffers.advance_step("ep", 8).expect("next step");
        let repeat = buffers.advance_step("ep", 8).unwrap_err().to_string();
        assert!(repeat.contains("step 8 arrived after step 8"), "{repeat}");
        let gap = buffers.advance_step("ep", 11).unwrap_err().to_string();
        assert!(gap.contains("expected 9"), "{gap}");
        buffers
            .advance_step("other", 0)
            .expect("independent episode");
        buffers.evict("ep");
        buffers.advance_step("ep", 0).expect("fresh after eviction");
    }

    #[test]
    fn frame_buffers_report_held_episodes_and_bytes() {
        let mut buffers = FrameBuffers::new();
        assert_eq!((buffers.episodes(), buffers.state_bytes()), (0, 0));

        buffers.episode("ep-1").insert(
            NodePath::root().push_key("image"),
            Window::from([frame(1), frame(2)]),
        );
        buffers
            .episode("ep-2")
            .insert(NodePath::root().push_key("image"), Window::from([frame(3)]));
        assert_eq!(buffers.episodes(), 2);
        assert_eq!(buffers.state_bytes(), 6);

        buffers.evict("ep-1");
        assert_eq!((buffers.episodes(), buffers.state_bytes()), (1, 2));
        buffers.clear();
        assert_eq!((buffers.episodes(), buffers.state_bytes()), (0, 0));
    }

    #[test]
    fn split_chunk_unstacks_a_tensor_leading_axis() {
        let chunk = Tensor::from_vec(vec![0, 1, 2, 3, 4, 5], vec![3, 2], DType::Uint8).unwrap();
        let steps = split_chunk(Value::Tensor(chunk)).expect("split");
        assert_eq!(steps.len(), 3);
        let bytes: Vec<Vec<u8>> = steps
            .iter()
            .map(|step| match step {
                Value::Tensor(t) => {
                    assert_eq!(t.shape(), &[2]);
                    assert_eq!(t.dtype(), DType::Uint8);
                    t.to_contiguous_bytes().into_owned()
                }
                other => panic!("expected tensor, got {other:?}"),
            })
            .collect();
        assert_eq!(bytes, vec![vec![0u8, 1], vec![2, 3], vec![4, 5]]);

        let listed =
            split_chunk(Value::List(vec![Value::Number(1.0), Value::Number(2.0)])).expect("list");
        assert_eq!(listed.len(), 2);
        assert_eq!(split_chunk(Value::Number(7.0)).expect("leaf").len(), 1);

        let scalar = Tensor::from_vec(vec![9], vec![], DType::Uint8).unwrap();
        assert_eq!(split_chunk(Value::Tensor(scalar)).expect("scalar").len(), 1);
    }

    #[test]
    fn split_chunk_unstacks_a_map_per_leaf() {
        let arm = Tensor::from_vec(vec![0, 1, 2, 3], vec![2, 2], DType::Uint8).unwrap();
        let grip = Tensor::from_vec(vec![7, 8], vec![2, 1], DType::Uint8).unwrap();
        let chunk = Value::Map(BTreeMap::from([
            ("arm".to_owned(), Value::Tensor(arm)),
            ("grip".to_owned(), Value::Tensor(grip)),
        ]));
        let steps = split_chunk(chunk).expect("split map");
        assert_eq!(steps.len(), 2);
        for (i, step) in steps.iter().enumerate() {
            let map = match step {
                Value::Map(m) => m,
                other => panic!("expected map, got {other:?}"),
            };
            match map.get("arm") {
                Some(Value::Tensor(t)) => {
                    assert_eq!(t.shape(), &[2]);
                    assert_eq!(
                        t.to_contiguous_bytes().into_owned(),
                        vec![i as u8 * 2, i as u8 * 2 + 1]
                    );
                }
                other => panic!("expected arm tensor, got {other:?}"),
            }
            match map.get("grip") {
                Some(Value::Tensor(t)) => {
                    assert_eq!(t.shape(), &[1]);
                    assert_eq!(t.to_contiguous_bytes().into_owned(), vec![7 + i as u8]);
                }
                other => panic!("expected grip tensor, got {other:?}"),
            }
        }

        let three = Tensor::from_vec(vec![0, 1, 2], vec![3], DType::Uint8).unwrap();
        let two = Tensor::from_vec(vec![0, 1], vec![2], DType::Uint8).unwrap();
        let ragged = Value::Map(BTreeMap::from([
            ("a".to_owned(), Value::Tensor(three)),
            ("b".to_owned(), Value::Tensor(two)),
        ]));
        assert!(split_chunk(ragged).is_err());
    }

    #[test]
    fn bridge_flat_box_keys_under_root_envelope() {
        use crate::plans::OBS_ROOT_KEY;
        let tensor = Tensor::from_vec(vec![1, 2, 3, 4], vec![4], DType::Uint8).expect("tensor");
        let value = SpaceValue::Box(tensor.clone());
        let space = flat(vec![4], DType::Uint8);
        let referenced: BTreeSet<String> = [OBS_ROOT_KEY.to_owned()].into_iter().collect();

        let map = space_value_to_obs_map(&value, &space, &referenced).expect("bridge");
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(OBS_ROOT_KEY), Some(&Value::Tensor(tensor)));
    }

    #[test]
    fn bridge_discrete_becomes_number() {
        use crate::plans::OBS_ROOT_KEY;
        let value = SpaceValue::Discrete(7);
        let space = flat(vec![], DType::Int64);
        let referenced: BTreeSet<String> = [OBS_ROOT_KEY.to_owned()].into_iter().collect();

        let map = space_value_to_obs_map(&value, &space, &referenced).expect("bridge");
        assert_eq!(map.get(OBS_ROOT_KEY), Some(&Value::Int(7)));
    }

    #[test]
    fn bridge_dict_selects_referenced_top_level_keys() {
        let cam = Tensor::from_vec(vec![9, 8], vec![2], DType::Uint8).expect("cam");
        let unused = Tensor::from_vec(vec![1], vec![1], DType::Uint8).expect("unused");
        let value = SpaceValue::Dict(
            [
                ("cam".to_owned(), SpaceValue::Box(cam.clone())),
                ("unused".to_owned(), SpaceValue::Box(unused)),
            ]
            .into_iter()
            .collect(),
        );
        let space = SpaceSpec {
            shape: vec![],
            dtype: DType::Uint8,
            spec: Some(SpaceKind::Dict(DictSpec {
                keys: vec!["cam".to_owned(), "unused".to_owned()],
                spaces: vec![flat(vec![2], DType::Uint8), flat(vec![1], DType::Uint8)],
            })),
        };
        let referenced: BTreeSet<String> = ["cam".to_owned()].into_iter().collect();

        let map = space_value_to_obs_map(&value, &space, &referenced).expect("bridge");
        assert_eq!(map.len(), 1);
        assert_eq!(map.get("cam"), Some(&Value::Tensor(cam)));
        assert!(!map.contains_key("unused"));
    }

    #[test]
    fn value_max_abs_diff_measures_numeric_drift() {
        let tensor = |values: &[f32]| {
            Value::Tensor(crate::apply::value::tensor_from_f32(
                vec![values.len() as i64],
                values,
            ))
        };
        assert_eq!(
            value_max_abs_diff(&Value::Number(1.0), &Value::Number(1.0)),
            Some(0.0)
        );
        assert_eq!(
            value_max_abs_diff(&Value::Number(1.0), &Value::Number(3.5)),
            Some(2.5)
        );
        assert_eq!(
            value_max_abs_diff(&tensor(&[1.0, 2.0]), &tensor(&[1.0, 2.0])),
            Some(0.0)
        );
        assert_eq!(
            value_max_abs_diff(&tensor(&[1.0, 2.0]), &tensor(&[1.0, 5.0])),
            Some(3.0)
        );
        assert!(value_max_abs_diff(&tensor(&[1.0]), &tensor(&[1.0, 2.0])).is_none());
        assert!(value_max_abs_diff(&Value::Number(1.0), &tensor(&[1.0])).is_none());
    }

    fn stacked_adapter(stack: u32) -> ResolvedAdapter {
        strided_adapter(stack, None, StackPad::First, None)
    }

    fn strided_adapter(
        stack: u32,
        offsets: Option<Vec<i32>>,
        stack_pad: StackPad,
        normalize: Option<(f64, f64)>,
    ) -> ResolvedAdapter {
        let mut image = serde_json::json!({
            "type": "image", "role": "image/primary", "stack": stack,
            "resample": "bilinear_aa", "dtype": "uint8",
        });
        if let Some(offsets) = offsets {
            image["offsets"] = serde_json::json!(offsets);
        }
        if stack_pad == StackPad::Black {
            image["stack_pad"] = serde_json::json!("black");
        }
        if let Some((low, high)) = normalize {
            image["normalize"] = serde_json::json!([low, high]);
            image["dtype"] = serde_json::json!("float32");
        }
        images_adapter(&[("cam", "cam", image)])
    }

    fn images_adapter(inputs: &[(&str, &str, serde_json::Value)]) -> ResolvedAdapter {
        let roles = ["image/primary", "image/secondary"];
        let mut tags = serde_json::Map::new();
        let mut keys = Vec::new();
        let mut model = serde_json::Map::new();
        for (i, (placement, source, input)) in inputs.iter().enumerate() {
            let mut input = input.clone();
            input["role"] = serde_json::json!(roles[i]);
            tags.insert(
                (*source).to_owned(),
                serde_json::json!({"type": "image", "role": roles[i]}),
            );
            keys.push(*source);
            match placement.split_once('/') {
                Some((outer, inner)) => {
                    model
                        .entry(outer.to_owned())
                        .or_insert_with(|| serde_json::json!({}))[inner] = input;
                }
                None => {
                    model.insert((*placement).to_owned(), input);
                }
            }
        }
        let tags: crate::spec::EnvTags = serde_json::from_value(serde_json::json!({
            "observation": tags, "action": {"components": [{"dim": 1}]},
        }))
        .expect("env tags");
        let spec: crate::spec::ModelSpec = serde_json::from_value(serde_json::json!({
            "input": model, "output": {"components": [{"dim": 1}]},
        }))
        .expect("model spec");
        let camera = serde_json::json!({"kind": "box", "shape": [1, 1, 3], "dtype": "uint8"});
        let obs: crate::space_view::SpaceView = serde_json::from_value(serde_json::json!({
            "kind": "dict", "dtype": "unspecified", "keys": keys,
            "children": vec![camera; keys.len()],
        }))
        .expect("obs space");
        let act: crate::space_view::SpaceView =
            serde_json::from_str(r#"{"kind":"box","shape":[1],"dtype":"float32"}"#)
                .expect("action space");
        crate::lower::resolve(&tags, &obs, &act, &spec, false).expect("resolves")
    }

    fn tagged_obs(tag: u8) -> BTreeMap<String, Value> {
        let image = Tensor::from_vec(vec![tag, tag, tag], vec![1, 1, 3], DType::Uint8)
            .expect("tagged frame");
        [("cam".to_owned(), Value::Tensor(image))]
            .into_iter()
            .collect()
    }

    fn cam_stack(payload: &Value) -> (Vec<i64>, Vec<u8>) {
        let Value::Map(map) = payload else {
            panic!("payload not a map: {payload:?}");
        };
        match map.get("cam").expect("cam present") {
            Value::Tensor(tensor) => (
                tensor.shape().to_vec(),
                tensor.to_contiguous_bytes().into_owned(),
            ),
            other => panic!("cam not a tensor: {other:?}"),
        }
    }

    #[test]
    fn a_dotted_key_and_the_nested_path_it_renders_like_keep_separate_windows() {
        use crate::apply::NoCustoms;
        use crate::path::NodePath;

        let dotted = NodePath::root().push_key("camera.rgb");
        let nested = NodePath::root().push_key("camera").push_key("rgb");
        assert_eq!(dotted.to_string(), nested.to_string());
        let image = serde_json::json!({"type": "image", "stack": 2});
        let adapter = images_adapter(&[
            ("camera.rgb", "primary", image.clone()),
            ("camera/rgb", "secondary", image),
        ]);
        let obs = |primary: u8, secondary: u8| -> BTreeMap<String, Value> {
            let frame = |tag| {
                Value::Tensor(Tensor::from_vec(vec![tag; 3], vec![1, 1, 3], DType::Uint8).unwrap())
            };
            [
                ("primary".to_owned(), frame(primary)),
                ("secondary".to_owned(), frame(secondary)),
            ]
            .into_iter()
            .collect()
        };
        let mut buffers = FrameBuffers::new();
        let mut payload = Value::Number(0.0);
        for (step, (primary, secondary)) in [(10, 100), (11, 101)].into_iter().enumerate() {
            payload = assemble_obs(
                &adapter,
                &obs(primary, secondary),
                "ep",
                step as i64,
                &mut buffers,
                &NoCustoms,
                &NoEncodings,
            )
            .expect("assemble");
        }
        let mut bytes = |path: &NodePath| {
            let Value::Tensor(tensor) =
                crate::apply::lookup::resolve_source_mut(&mut payload, path).unwrap()
            else {
                panic!("stacked input is not a tensor")
            };
            tensor.to_contiguous_bytes().into_owned()
        };
        assert_eq!(bytes(&dotted), vec![10, 10, 10, 11, 11, 11]);
        assert_eq!(bytes(&nested), vec![100, 100, 100, 101, 101, 101]);
    }

    #[test]
    fn contiguous_stack_bytes_are_frozen() {
        use crate::apply::NoCustoms;

        let adapter = stacked_adapter(3);
        let mut buffers = FrameBuffers::new();
        let step = |adapter: &ResolvedAdapter, tag: u8, buffers: &mut FrameBuffers| {
            cam_stack(
                &assemble_obs(
                    adapter,
                    &tagged_obs(tag),
                    "ep",
                    0,
                    buffers,
                    &NoCustoms,
                    &NoEncodings,
                )
                .expect("assemble"),
            )
        };

        assert_eq!(
            step(&adapter, 10, &mut buffers),
            (vec![3, 1, 1, 3], vec![10; 9])
        );
        assert_eq!(
            step(&adapter, 11, &mut buffers).1,
            vec![10, 10, 10, 10, 10, 10, 11, 11, 11]
        );
        assert_eq!(
            step(&adapter, 12, &mut buffers).1,
            vec![10, 10, 10, 11, 11, 11, 12, 12, 12]
        );
        assert_eq!(
            step(&adapter, 13, &mut buffers).1,
            vec![11, 11, 11, 12, 12, 12, 13, 13, 13]
        );
        assert_eq!(
            step(&adapter, 14, &mut buffers).1,
            vec![12, 12, 12, 13, 13, 13, 14, 14, 14]
        );
    }

    #[test]
    fn a_strided_window_gathers_the_rldx_frame_table() {
        use crate::apply::NoCustoms;

        let adapter = strided_adapter(4, Some(vec![-6, -4, -2, 0]), StackPad::First, None);
        let mut buffers = FrameBuffers::new();
        let step = |tag: u8, buffers: &mut FrameBuffers| {
            cam_stack(
                &assemble_obs(
                    &adapter,
                    &tagged_obs(tag),
                    "ep",
                    0,
                    buffers,
                    &NoCustoms,
                    &NoEncodings,
                )
                .expect("assemble"),
            )
        };
        let table: [(u8, [u8; 4]); 9] = [
            (0, [0, 0, 0, 0]),
            (1, [0, 0, 0, 1]),
            (2, [0, 0, 0, 2]),
            (3, [0, 0, 1, 3]),
            (4, [0, 0, 2, 4]),
            (5, [0, 1, 3, 5]),
            (6, [0, 2, 4, 6]),
            (7, [1, 3, 5, 7]),
            (8, [2, 4, 6, 8]),
        ];
        for (tag, expected) in table {
            let (shape, bytes) = step(tag, &mut buffers);
            assert_eq!(shape, vec![4, 1, 1, 3], "step {tag}: shape");
            let picked: Vec<u8> = bytes.chunks(3).map(|frame| frame[0]).collect();
            assert_eq!(picked, expected, "step {tag}");
        }
    }

    #[test]
    fn a_black_pad_lands_on_the_plan_s_normalized_floor() {
        use crate::apply::NoCustoms;

        let adapter = strided_adapter(
            6,
            Some(vec![-25, -20, -15, -10, -5, 0]),
            StackPad::Black,
            Some((-1.0, 1.0)),
        );
        let mut buffers = FrameBuffers::new();
        let first = assemble_obs(
            &adapter,
            &tagged_obs(255),
            "ep",
            0,
            &mut buffers,
            &NoCustoms,
            &NoEncodings,
        )
        .expect("assemble");
        let Value::Map(map) = &first else {
            panic!("payload not a map");
        };
        let Value::Tensor(stacked) = map.get("cam").expect("cam") else {
            panic!("cam not a tensor");
        };
        assert_eq!(stacked.shape(), &[6, 1, 1, 3]);
        let values = crate::apply::value::to_f64_vec(stacked);
        assert_eq!(&values[..15], &[-1.0f64; 15]);
        assert_eq!(&values[15..], &[1.0f64; 3]);
        assert_eq!(buffers.state_bytes(), 26 * 3 * 4);
    }

    #[test]
    fn observing_a_step_advances_the_window_exactly_like_assembling_one() {
        use crate::apply::NoCustoms;

        let adapter = strided_adapter(3, Some(vec![-4, -2, 0]), StackPad::First, None);
        let assembled = |buffers: &mut FrameBuffers, tag: u8| {
            cam_stack(
                &assemble_obs(
                    &adapter,
                    &tagged_obs(tag),
                    "ep",
                    0,
                    buffers,
                    &NoCustoms,
                    &NoEncodings,
                )
                .expect("assemble"),
            )
            .1
        };

        let mut every_step = FrameBuffers::new();
        let mut expected = Vec::new();
        for tag in 0..8u8 {
            expected.push(assembled(&mut every_step, tag));
        }

        let mut chunked = FrameBuffers::new();
        for tag in 0..8u8 {
            if tag % 3 == 0 {
                assert_eq!(assembled(&mut chunked, tag), expected[tag as usize]);
            } else {
                observe_obs(&adapter, &tagged_obs(tag), "ep", &mut chunked).expect("observe");
            }
        }
    }

    fn previous_adapter() -> ResolvedAdapter {
        let tags: crate::spec::EnvTags = serde_json::from_str(
            r#"{"observation":{"g":{"type":"state","role":"proprio/gripper"}},
                "action":{"components":[{"role":"action/gripper","dim":1},
                                        {"role":"action/joint_pos","dim":2}]}}"#,
        )
        .expect("env tags");
        let spec: crate::spec::ModelSpec = serde_json::from_str(
            r#"{"input":{"type":"state","components":["proprio/gripper",
                    {"role":"action/joint_pos","source":"action","fill":-1.0}]},
                "output":{"components":[{"role":"action/gripper","dim":1},
                                        {"role":"action/joint_pos","dim":2}]}}"#,
        )
        .expect("model spec");
        let obs: crate::space_view::SpaceView = serde_json::from_str(
            r#"{"kind":"dict","dtype":"unspecified","keys":["g"],
                "children":[{"kind":"box","shape":[1],"dtype":"float32"}]}"#,
        )
        .expect("obs space");
        let act: crate::space_view::SpaceView =
            serde_json::from_str(r#"{"kind":"box","shape":[3],"dtype":"float32"}"#)
                .expect("action space");
        crate::lower::resolve(&tags, &obs, &act, &spec, false).expect("resolves")
    }

    fn vector(values: &[f32]) -> Value {
        Value::Tensor(crate::apply::value::tensor_from_f32(
            vec![values.len() as i64],
            values,
        ))
    }

    fn assembled(adapter: &ResolvedAdapter, buffers: &mut FrameBuffers, step: i64) -> Vec<f32> {
        use crate::apply::NoCustoms;
        let raw = [("g".to_owned(), vector(&[9.0]))].into_iter().collect();
        match assemble_obs(adapter, &raw, "ep", step, buffers, &NoCustoms, &NoEncodings)
            .expect("assemble")
        {
            Value::Tensor(tensor) => crate::apply::value::to_f64_vec(&tensor)
                .into_iter()
                .map(|v| v as f32)
                .collect(),
            other => panic!("expected a tensor, got {other:?}"),
        }
    }

    fn executed(
        adapter: &ResolvedAdapter,
        buffers: &mut FrameBuffers,
        raw: &[f32],
        step: i64,
    ) -> SpaceValue {
        let space = SpaceSpec {
            shape: vec![3],
            dtype: DType::Float32,
            spec: Some(SpaceKind::Box(rlmesh_spaces::BoxSpec { bounds: None })),
        };
        apply_actions(
            adapter,
            vector(raw),
            &space,
            &NoEncodings,
            buffers,
            "ep",
            step,
        )
        .expect("apply")
    }

    #[test]
    fn a_previous_action_part_reads_the_fill_then_the_raw_action_of_the_step_before() {
        let adapter = previous_adapter();
        let mut buffers = FrameBuffers::new();
        assert_eq!(assembled(&adapter, &mut buffers, 0), [9.0, -1.0, -1.0]);
        let env_action = executed(&adapter, &mut buffers, &[0.5, 7.0, 8.0], 0);
        assert!(matches!(env_action, SpaceValue::Box(_)));
        assert_eq!(assembled(&adapter, &mut buffers, 1), [9.0, 7.0, 8.0]);
        executed(&adapter, &mut buffers, &[0.5, 9.0, 10.0], 1);
        assert_eq!(assembled(&adapter, &mut buffers, 2), [9.0, 9.0, 10.0]);
        assert_eq!(buffers.state_bytes(), 3 * 4);
        buffers.evict("ep");
        assert_eq!(assembled(&adapter, &mut buffers, 0), [9.0, -1.0, -1.0]);
    }

    #[test]
    fn a_missed_action_tick_is_loud() {
        let adapter = previous_adapter();
        let mut buffers = FrameBuffers::new();
        assembled(&adapter, &mut buffers, 0);
        let raw = [("g".to_owned(), vector(&[9.0]))].into_iter().collect();
        let err = assemble_obs(
            &adapter,
            &raw,
            "ep",
            1,
            &mut buffers,
            &crate::apply::NoCustoms,
            &NoEncodings,
        )
        .expect_err("no action was recorded for step 0")
        .to_string();
        assert!(
            err.contains("episode ep: the observation at step 1 reads the previous action but no action was recorded for step 0"),
            "{err}"
        );
        let err = record_action(&adapter, &vector(&[1.0]), &mut buffers, "ep", 0)
            .expect_err("width")
            .to_string();
        assert!(
            err.contains("has 1 elements but the action plan reads 3"),
            "{err}"
        );
    }

    #[test]
    fn chunk_frames_recorded_under_their_steps_feed_the_replan() {
        let adapter = previous_adapter();
        let mut buffers = FrameBuffers::new();
        assembled(&adapter, &mut buffers, 0);
        for frame in 0..4 {
            let k = frame as f32;
            executed(&adapter, &mut buffers, &[0.0, 10.0 + k, 20.0 + k], frame);
        }
        assert_eq!(buffers.state_bytes(), 4 * 3 * 4);
        assert_eq!(assembled(&adapter, &mut buffers, 4), [9.0, 13.0, 23.0]);
        for frame in 0..4 {
            let k = frame as f32;
            executed(
                &adapter,
                &mut buffers,
                &[0.0, 30.0 + k, 40.0 + k],
                4 + frame,
            );
        }
        assert_eq!(assembled(&adapter, &mut buffers, 6), [9.0, 31.0, 41.0]);
        executed(&adapter, &mut buffers, &[0.0, 50.0, 60.0], 6);
        assert_eq!(assembled(&adapter, &mut buffers, 7), [9.0, 50.0, 60.0]);
    }

    #[test]
    fn a_route_without_an_action_source_part_records_nothing() {
        let adapter = stacked_adapter(2);
        let mut buffers = FrameBuffers::new();
        record_action(&adapter, &Value::Number(1.0), &mut buffers, "ep", 0).expect("no-op");
        assert_eq!((buffers.episodes(), buffers.state_bytes()), (0, 0));
        assert!(!adapter.reads_previous_action());
    }

    #[test]
    fn assemble_obs_stacks_each_episode_independently_across_autoreset() {
        use crate::apply::NoCustoms;

        let adapter = stacked_adapter(2);
        let stack =
            |adapter: &ResolvedAdapter, tag: u8, episode: &str, buffers: &mut FrameBuffers| {
                let payload = assemble_obs(
                    adapter,
                    &tagged_obs(tag),
                    episode,
                    0,
                    buffers,
                    &NoCustoms,
                    &NoEncodings,
                )
                .unwrap();
                let (shape, bytes) = cam_stack(&payload);
                assert_eq!(shape, vec![2, 1, 1, 3]);
                bytes
            };

        let mut buffers = FrameBuffers::new();

        assert_eq!(stack(&adapter, 10, "ep-a", &mut buffers), vec![10; 6]);
        assert_eq!(stack(&adapter, 20, "ep-b", &mut buffers), vec![20; 6]);
        assert_eq!(
            stack(&adapter, 11, "ep-a", &mut buffers),
            vec![10, 10, 10, 11, 11, 11]
        );
        assert_eq!(
            stack(&adapter, 21, "ep-b", &mut buffers),
            vec![20, 20, 20, 21, 21, 21]
        );

        buffers.evict("ep-a");
        assert_eq!(stack(&adapter, 30, "ep-c", &mut buffers), vec![30; 6]);
        assert_eq!(
            stack(&adapter, 22, "ep-b", &mut buffers),
            vec![21, 21, 21, 22, 22, 22]
        );
    }
}
