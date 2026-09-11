//! Vectorized stateful adapter engine: episode-keyed frame-stacking and the
//! per-lane assemble/apply seam.
//!
//! Frame-stacking is per-episode state that used to live host-side in the
//! Python binding (`adapter.py` `_buffers`/`_stack_frames`). It relocates here,
//! keyed by `episode_id`, so a vectorized serve route frame-stacks each lane
//! correctly without the state ever crossing the network or living in Python.
//!
//! [`assemble_obs`] and [`apply_actions`] are the frozen per-lane seam the core
//! handler drives once per lane. They are deliberately single-sample: a future
//! fused forward pass stacks N `assemble_obs` outputs into one call, but the
//! per-lane transform is the ground-truth unit and never grows a batch axis.
//!
//! The two genuinely host-language holes cross via traits the binding
//! implements (so a pure-Rust handler inherits the same engine): custom inputs
//! via [`CustomTransform`](crate::apply::CustomTransform), and host-side custom
//! *encodings* (rotation re-packings the native core represents only as a base
//! encoding) via [`EncodingTransform`].

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use rlmesh_spaces::scalar::encode_i64_scalars;
use rlmesh_spaces::{SpaceKind, SpaceSpec, SpaceValue, Tensor};

use crate::apply::value::{cast, to_f64_vec};
use crate::apply::{CustomTransform, Value};
use crate::error::ApplyError;
use crate::plans::{ImagePlan, ObsPlan, ResolvedAdapter, StackedPlacement};
use crate::spec::StackPad;

/// Upper bound on frame-stack depth (mirrors the spec's `MAX_STACK`). A raw
/// native caller must not buffer an unbounded window and exhaust memory; the
/// spec deserializer already enforces this at the wire, the assert guards the
/// in-process path.
const MAX_STACK: usize = 64;

/// Host-language hook applying custom *encodings* the native core left in a base
/// encoding (e.g. a rotation re-packing the spec declared as a `CustomEncoding`,
/// which resolution shadows to a native base encoding). Distinct from
/// [`CustomTransform`](crate::apply::CustomTransform), which fills whole custom
/// *inputs*: an encoding transform repacks an existing payload key (obs) or
/// action segment in place. The implementor owns which keys/segments it touches.
pub trait EncodingTransform {
    /// Repack any custom-encoded observation payload entries, in place,
    /// **before** frame-stacking (so a stacked input stacks the model's
    /// representation). A no-op for routes with no observation encoding shims.
    /// The payload is a [`Value`] tree (`Map`/`List`/leaf); the implementor
    /// walks to whichever entries it owns.
    fn repack_obs(&self, payload: &mut Value) -> Result<(), ApplyError>;

    /// Repack the model action's custom-encoded segments back to their base
    /// encoding, in place, **before** the native action conversion. A no-op for
    /// routes with no action encoding shims.
    fn repack_action(&self, action: &mut Value) -> Result<(), ApplyError>;
}

/// An [`EncodingTransform`] that touches nothing — for fully declarative routes
/// (no custom encodings). Parallels [`NoCustoms`](crate::apply::NoCustoms).
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
/// `episode_id -> placement-string -> rolling window` (window `maxlen = depth`).
///
/// The handler holds one of these per `route_key`, with an edge-driven
/// lifecycle: an episode's entry appears lazily on first use and is dropped at
/// episode END ([`evict`](Self::evict)) or on close ([`clear`](Self::clear)) —
/// never on absence.
///
/// The key is the `episode_id` (a UUIDv4), **not** the lane index:
/// - autoreset reuses a lane's index across episodes, but `episode_id` is fresh
///   per episode, so old and new entries never collide;
/// - grouped predict can migrate an episode's slot index between groups, but
///   `episode_id` is stable, so the entry follows the episode, not the index.
#[derive(Default)]
pub struct FrameBuffers {
    inner: HashMap<String, BTreeMap<String, Window>>,
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

    /// Total bytes held across every episode's frame windows.
    #[must_use]
    pub fn state_bytes(&self) -> u64 {
        self.inner
            .values()
            .flat_map(|windows| windows.values())
            .flat_map(|window| window.frames.iter())
            .map(|tensor| tensor.nbytes() as u64)
            .sum()
    }

    /// The per-key window map for an episode, created lazily if absent.
    fn episode(&mut self, episode_id: &str) -> &mut BTreeMap<String, Window> {
        self.inner.entry(episode_id.to_owned()).or_default()
    }
}

/// One input's rolling frame window: the `span` most recent processed frames,
/// oldest first.
///
/// `span` is what the window *holds*; the stacked tensor is what the plan's
/// offsets *gather* out of it. For a contiguous stack the two coincide (hold 4,
/// stack all 4); a strided stack holds every frame in its reach and gathers a
/// few (hold 7, stack frames `-6, -4, -2, 0`). One ring, both cases.
#[derive(Debug, Default)]
struct Window {
    frames: VecDeque<Tensor>,
}

impl Window {
    /// Push one processed frame, padding the start of an episode so the window
    /// is full from step zero.
    ///
    /// The pad is the first observed frame ([`StackPad::First`], the default) or
    /// a black frame ([`StackPad::Black`]) — raw 8-bit `0` as the plan would
    /// have produced it, so under `normalize = [-1, 1]` a black pad really is
    /// `-1.0`.
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

    /// Stack the window's frames onto a new leading axis, oldest first.
    ///
    /// A contiguous stack takes the whole window; a strided one gathers the
    /// declared offsets back from the newest frame.
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
                                    "frame window for '{}' holds {} frame(s), too few to reach                                      offset {offset}",
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

/// A black frame the shape and dtype of a processed one.
///
/// Raw 8-bit `0` pushed through the plan is a constant frame, and every step
/// after the normalize map (channel swap, layout transpose, lead dims) either
/// permutes or reshapes it — so the level alone reconstructs it exactly:
/// `normalize`'s low end when the plan normalizes, `0` when it does not. The
/// `as f32` round trip matches `finalize_dtype`'s own arithmetic bit for bit.
fn black_frame(frame: &Tensor, plan: &ImagePlan) -> Result<Tensor, ApplyError> {
    let level = plan.normalize.map_or(0.0, |(low, _)| f64::from(low as f32));
    crate::apply::value::encode_f64_to(
        vec![level; frame.numel()],
        frame.shape().to_vec(),
        frame.dtype(),
    )
}

/// Upper bound on the runtime's execution horizon `h` (how many actions of one
/// predicted chunk are executed before re-planning). A pin above this is a
/// configuration error, not a capability question: the replay buffer is held in
/// memory per lane, and a four-digit horizon is always a mis-set knob rather than
/// a real open-loop plan. Enforced at engine resolve, at the SDK session seam,
/// and as the `maximum` on the platform's own horizon knobs.
pub const MAX_EXECUTION_HORIZON: u32 = 1024;

/// Split a chunked model action into its per-step actions (the leading axis is
/// the chunk axis). A `Value::Tensor` of shape `[chunk, ..]` unstacks along axis
/// 0 into per-step tensors; a scalar (0-d) tensor has no chunk axis and is a
/// degenerate single-step "chunk" (matching the run(env) path, which treats a 0-d
/// output as one step); a `Value::List` is already a list of per-step actions; any
/// other leaf is a single-step chunk. Called only when the runtime pins a replay
/// horizon > 1, so the leading axis is taken as the chunk. A mis-shaped multi-dim
/// output (e.g. a
/// flat `[dim]` action where `dim > 1`) splits into per-step scalars that fail the
/// action-space reshape downstream; a flat `[1]` against a dim-1 action space,
/// however, splits cleanly and is accepted — shape inference cannot catch that
/// degenerate case, so a model that forgets the chunk axis is the caller's bug.
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
        // A Dict-space action chunk carries the horizon as each leaf's leading axis
        // (e.g. {"arm": tensor[H, ...], "gripper": tensor[H, ...]}); split every
        // leaf and zip into H per-step Maps. Mirrors the Python `_first_frame`
        // Mapping recursion and the batched `_dechunk` tree_map, so a Dict action
        // replays its whole horizon instead of being treated as one frame.
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

/// Assemble one lane's model-input payload from its raw observation.
///
/// Pipeline order (matching the Python truth, `adapter.py:179-205`):
/// 1. declarative native transform (customs dispatched inside it),
/// 2. observation encoding-shim repack ([`EncodingTransform::repack_obs`]),
/// 3. per-episode frame-stacking for keys in [`ResolvedAdapter::stacks`].
///
/// Frozen so a future fused path can stack N `assemble_obs` outputs into one
/// forward pass without re-shaping callers.
pub fn assemble_obs(
    adapter: &ResolvedAdapter,
    raw_obs: &BTreeMap<String, Value>,
    episode_id: &str,
    buffers: &mut FrameBuffers,
    customs: &dyn CustomTransform,
    encodings: &dyn EncodingTransform,
) -> Result<Value, ApplyError> {
    let mut payload = adapter.transform_obs(raw_obs, customs)?;
    encodings.repack_obs(&mut payload)?;
    // Frame-stacking runs as a post-scatter pass over the single precomputed
    // stacking list ([`ResolvedAdapter::stacked_placements`]): walk the assembled
    // tree to each stacked placement and stack in place, keyed (in the
    // per-episode window) by the placement's precomputed canonical string.
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
            // Move the frame out of the payload slot (overwritten with the
            // stacked result below) so it lands in the window without a per-step
            // tensor copy.
            let Value::Tensor(frame) = std::mem::replace(slot, Value::Number(0.0)) else {
                unreachable!("frame confirmed a tensor above")
            };
            // After the first frame the window exists, so the steady-state path is a
            // borrow with no key clone; only the first-frame insert pays the clone.
            let window = if windows.contains_key(&entry.key) {
                windows
                    .get_mut(&entry.key)
                    .expect("window present (just checked)")
            } else {
                windows.entry(entry.key.clone()).or_default()
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
/// The observe tick: an env step that replays a queued action still happened, so
/// every frame window has to see its frame or the history the next predict
/// stacks would be the decision points rather than the steps. Pushes exactly the
/// frames [`ResolvedAdapter::history_frames`] produces and returns; a route with
/// no stacked input does nothing at all.
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
    let frames = adapter.history_frames(raw_obs)?;
    let windows = buffers.episode(episode_id);
    for (entry, frame) in stacked.iter().zip(frames) {
        let ObsPlan::Image(image_plan) = &adapter.obs_plans[entry.plan_index] else {
            unreachable!("only image plans are ever stacked")
        };
        windows
            .entry(entry.key.clone())
            .or_default()
            .push(frame, entry, image_plan)?;
    }
    Ok(())
}

/// Convert one lane's model action into the env action [`SpaceValue`].
///
/// Runs the action encoding-shim repack ([`EncodingTransform::repack_action`])
/// **before** the native conversion, matching `adapter.py:318-327`, then encodes
/// the resulting env-action tensor into the route's `action_space`. Frozen so a
/// future action-chunk replay queue / externalize-scatter inserts here without
/// re-shaping callers.
pub fn apply_actions(
    adapter: &ResolvedAdapter,
    raw_action: Value,
    action_space: &SpaceSpec,
    encodings: &dyn EncodingTransform,
) -> Result<SpaceValue, ApplyError> {
    let mut action = raw_action;
    encodings.repack_action(&mut action)?;
    let env_action = adapter.transform_action(&action)?;
    tensor_to_space_value(env_action, action_space)
}

/// Encode an env-action tensor into the action [`SpaceValue`] for `space`.
///
/// The inverse of [`space_value_to_value`] for the action side, mirroring the
/// binding's `py_any_to_space_value` semantics: a `Box` casts to the space dtype
/// and shape; a `Discrete`/`MultiDiscrete` rejects non-integral values rather
/// than truncating; a `MultiBinary` takes `!= 0`. The native action transform
/// always yields a float32 vector, so the cast restores the space's dtype.
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
/// unmeasurable comparison conservatively). Numeric leaves (`Tensor`, `Number`)
/// diff elementwise; `List`/`Map` take the max over children; other leaves and
/// shape mismatches return `None`.
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
        (Value::Number(a), Value::Number(b)) => Some((a - b).abs()),
        (Value::List(a), Value::List(b)) if a.len() == b.len() => {
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
/// [`Value`] payload model. Reproduces the binding's
/// `decode_value(space_value_to_py_neutral(..))` composition natively (no numpy,
/// no Python): a `Box` keeps its tensor; a `Discrete` becomes a number; a
/// `MultiBinary`/`MultiDiscrete` becomes the space's dtyped tensor; `Dict`/
/// `Tuple` recurse into a map/list.
pub fn space_value_to_value(value: &SpaceValue, space: &SpaceSpec) -> Result<Value, ApplyError> {
    match value {
        SpaceValue::Box(tensor) => Ok(Value::Tensor(tensor.clone())),
        SpaceValue::Discrete(value) => Ok(Value::Number(*value as f64)),
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
            Ok(Value::List(out))
        }
    }
}

/// Bridge a decoded env-observation [`SpaceValue`] into the adapter's raw-obs
/// envelope, keeping only the top-level entries `referenced` selects. Mirrors
/// the binding's `decode_referenced_obs`: a `Dict` env yields the selected
/// top-level entries (keyed by their Dict key); any flat (non-`Dict`) env yields
/// the single whole-obs leaf under the reserved `OBS_ROOT_KEY`.
///
/// `referenced` carries *envelope keys* (see `envelope_key`): a
/// Dict-rooted source's first path segment, or `OBS_ROOT_KEY` for a root/
/// Tuple-rooted source. The caller selects the keys: a declarative-only route
/// passes [`ResolvedAdapter::referenced_obs_keys`]; a route with custom holes
/// passes all top-level keys so the custom callback sees the full observation.
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
                    // A referenced Dict-space top-level key must not equal the
                    // reserved whole-observation envelope key, or it would
                    // silently collide with the root/Tuple-rooted source's
                    // envelope entry. Only referenced keys are checked: an unused
                    // env key never aborts a step (it is dropped, not inserted),
                    // matching `referenced_obs_keys`.
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
            // A flat (non-Dict) env is a single leaf, presented under the
            // reserved envelope key a root/Tuple-rooted source references.
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
        // A trivial 2-element uint8 "image" frame.
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
    fn frame_buffers_report_held_episodes_and_bytes() {
        let mut buffers = FrameBuffers::new();
        assert_eq!((buffers.episodes(), buffers.state_bytes()), (0, 0));

        buffers
            .episode("ep-1")
            .insert("image".to_owned(), Window::from([frame(1), frame(2)]));
        buffers
            .episode("ep-2")
            .insert("image".to_owned(), Window::from([frame(3)]));
        assert_eq!(buffers.episodes(), 2);
        // Three 2-byte uint8 frames held across the two episodes.
        assert_eq!(buffers.state_bytes(), 6);

        buffers.evict("ep-1");
        assert_eq!((buffers.episodes(), buffers.state_bytes()), (1, 2));
        buffers.clear();
        assert_eq!((buffers.episodes(), buffers.state_bytes()), (0, 0));
    }

    #[test]
    fn split_chunk_unstacks_a_tensor_leading_axis() {
        // A [3, 2] uint8 chunk splits into 3 per-step [2] actions, dtype + values
        // preserved (the zero-copy unstack view materializes correctly).
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

        // A List is already per-step; a bare leaf is a degenerate single-step chunk.
        let listed =
            split_chunk(Value::List(vec![Value::Number(1.0), Value::Number(2.0)])).expect("list");
        assert_eq!(listed.len(), 2);
        assert_eq!(split_chunk(Value::Number(7.0)).expect("leaf").len(), 1);

        // A scalar (0-d) tensor has no chunk axis: one step, not an unstack error
        // (matches the run(env) path, which treats a 0-d output as a single step).
        let scalar = Tensor::from_vec(vec![9], vec![], DType::Uint8).unwrap();
        assert_eq!(split_chunk(Value::Tensor(scalar)).expect("scalar").len(), 1);
    }

    #[test]
    fn split_chunk_unstacks_a_map_per_leaf() {
        // A Dict-space action chunk carries the horizon inside each leaf:
        // {"arm": [2, 2], "grip": [2, 1]} splits into 2 per-step Maps, each leaf
        // unstacked along its leading axis (not treated as a single frame).
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

        // Leaves that disagree on horizon are a clear error, not a silent zip.
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
        assert_eq!(map.get(OBS_ROOT_KEY), Some(&Value::Number(7.0)));
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
        // Reference only "cam": "unused" must be dropped (never materialized).
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
        // Equal values -> zero drift; differing values -> the max abs delta.
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
        // Shape and type mismatches are unmeasurable (the probe handles None).
        assert!(value_max_abs_diff(&tensor(&[1.0]), &tensor(&[1.0, 2.0])).is_none());
        assert!(value_max_abs_diff(&Value::Number(1.0), &tensor(&[1.0])).is_none());
    }

    /// A pass-through (no resize/normalize) single-image adapter that stacks
    /// `stack` consecutive frames. One literal, shared by the stacking tests, so
    /// a new [`ImagePlan`] field lands in one place.
    fn stacked_adapter(stack: u32) -> crate::plans::ResolvedAdapter {
        strided_adapter(stack, None, StackPad::First, None)
    }

    /// The same adapter with a declared window: `offsets` gathers, `stack_pad`
    /// fills the start of an episode, `normalize` (a `(low, high)` pair) is what
    /// a black pad frame has to land on.
    fn strided_adapter(
        stack: u32,
        offsets: Option<Vec<i32>>,
        stack_pad: StackPad,
        normalize: Option<(f64, f64)>,
    ) -> crate::plans::ResolvedAdapter {
        use crate::plans::{ActionPlan, ImagePlan, ObsPlan, ResolvedAdapter};
        use crate::spec::ImageLayout;

        ResolvedAdapter::new(
            vec![ObsPlan::Image(ImagePlan {
                placement: crate::path::NodePath::root().push_key("cam"),
                source: crate::path::NodePath::root().push_key("cam"),
                src_layout: ImageLayout::Hwc,
                dst_layout: ImageLayout::Hwc,
                flip: false,
                size: None,
                fit: crate::spec::FitMode::Stretch,
                resample: "bilinear_aa".to_owned(),
                dtype: if normalize.is_some() {
                    "float32".to_owned()
                } else {
                    "uint8".to_owned()
                },
                normalize,
                lead_dims: 0,
                src_range: Some((0.0, 255.0)),
                stack,
                zero_fill: None,
                fill: 0,
                crop: None,
                jpeg_quality: None,
                swap_rb: false,
                render: None,
                offsets,
                stack_pad,
                frame_bytes: 3,
                role_rebound: None,
            })],
            ActionPlan {
                segments: vec![],
                clip: None,
                in_dim: 0,
            },
            Vec::new(),
            Vec::new(),
        )
    }

    /// A 1x1x3 uint8 frame whose every byte is `tag` — a per-step fingerprint.
    fn tagged_obs(tag: u8) -> BTreeMap<String, Value> {
        let image = Tensor::from_vec(vec![tag, tag, tag], vec![1, 1, 3], DType::Uint8)
            .expect("tagged frame");
        [("cam".to_owned(), Value::Tensor(image))]
            .into_iter()
            .collect()
    }

    /// The stacked `cam` tensor's shape and bytes out of an assembled payload.
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
    fn contiguous_stack_bytes_are_frozen() {
        use crate::apply::NoCustoms;

        // GOLDEN LOCK. A contiguous stack (no declared offsets, first-frame pad)
        // is the behaviour every spec written before strided history depends on:
        // oldest-first on a new leading axis, the start of an episode padded with
        // copies of the first frame, a `maxlen = depth` sliding window. These
        // bytes are the contract — the window internals may be rebuilt around
        // them, but they may not move.
        let adapter = stacked_adapter(3);
        let mut buffers = FrameBuffers::new();
        let step = |adapter: &ResolvedAdapter, tag: u8, buffers: &mut FrameBuffers| {
            cam_stack(
                &assemble_obs(
                    adapter,
                    &tagged_obs(tag),
                    "ep",
                    buffers,
                    &NoCustoms,
                    &NoEncodings,
                )
                .expect("assemble"),
            )
        };

        // (depth, *frame) on a new leading axis.
        assert_eq!(
            step(&adapter, 10, &mut buffers),
            (vec![3, 1, 1, 3], vec![10; 9])
        );
        // [f0, f0, f1]
        assert_eq!(
            step(&adapter, 11, &mut buffers).1,
            vec![10, 10, 10, 10, 10, 10, 11, 11, 11]
        );
        // [f0, f1, f2]
        assert_eq!(
            step(&adapter, 12, &mut buffers).1,
            vec![10, 10, 10, 11, 11, 11, 12, 12, 12]
        );
        // [f1, f2, f3] — the oldest frame is evicted at depth.
        assert_eq!(
            step(&adapter, 13, &mut buffers).1,
            vec![11, 11, 11, 12, 12, 12, 13, 13, 13]
        );
        // [f2, f3, f4]
        assert_eq!(
            step(&adapter, 14, &mut buffers).1,
            vec![12, 12, 12, 13, 13, 13, 14, 14, 14]
        );
    }

    #[test]
    fn a_strided_window_gathers_the_rldx_frame_table() {
        use crate::apply::NoCustoms;

        // rldx's hand-rolled history: a 7-deep buffer sampled at
        // `FRAME_INDICES = [0, 2, 4, 6]`, i.e. offsets [-6, -4, -2, 0] off the
        // current step. The table below is that arithmetic, clamped at the start
        // of an episode exactly as its `max(end, 0)` does.
        let adapter = strided_adapter(4, Some(vec![-6, -4, -2, 0]), StackPad::First, None);
        let mut buffers = FrameBuffers::new();
        let step = |tag: u8, buffers: &mut FrameBuffers| {
            cam_stack(
                &assemble_obs(
                    &adapter,
                    &tagged_obs(tag),
                    "ep",
                    buffers,
                    &NoCustoms,
                    &NoEncodings,
                )
                .expect("assemble"),
            )
        };
        // Frames are tagged by their step number, so the expected table reads as
        // the frame indices upstream would have taken.
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

        // hy-vla's window: 6 frames at stride 5, and the slots whose un-clipped
        // index is negative are BLACK, not a replayed first frame. Under
        // `normalize = [-1, 1]` "black" is raw 8-bit 0 pushed through the plan,
        // so the pad frames read -1.0 — the name says what the pixels are, never
        // what the tensor holds.
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
        // Five black pad slots at the normalize floor, then the real frame at 1.0.
        assert_eq!(&values[..15], &[-1.0f64; 15]);
        assert_eq!(&values[15..], &[1.0f64; 3]);
        // The window holds the whole 26-frame span even though it stacks 6.
        assert_eq!(buffers.state_bytes(), 26 * 3 * 4);
    }

    #[test]
    fn observing_a_step_advances_the_window_exactly_like_assembling_one() {
        use crate::apply::NoCustoms;

        // The horizon invariant, at the seam: a step whose action was replayed
        // still pushes its frame. A window ticked by `observe_obs` on the steps
        // between predicts must therefore stack the same frames as one that
        // assembled every step.
        let adapter = strided_adapter(3, Some(vec![-4, -2, 0]), StackPad::First, None);
        let assembled = |buffers: &mut FrameBuffers, tag: u8| {
            cam_stack(
                &assemble_obs(
                    &adapter,
                    &tagged_obs(tag),
                    "ep",
                    buffers,
                    &NoCustoms,
                    &NoEncodings,
                )
                .expect("assemble"),
            )
            .1
        };

        // Every step assembles (execution_horizon = 1).
        let mut every_step = FrameBuffers::new();
        let mut expected = Vec::new();
        for tag in 0..8u8 {
            expected.push(assembled(&mut every_step, tag));
        }

        // Every third step assembles; the two in between only observe.
        let mut chunked = FrameBuffers::new();
        for tag in 0..8u8 {
            if tag % 3 == 0 {
                assert_eq!(assembled(&mut chunked, tag), expected[tag as usize]);
            } else {
                observe_obs(&adapter, &tagged_obs(tag), "ep", &mut chunked).expect("observe");
            }
        }
    }

    #[test]
    fn assemble_obs_stacks_each_episode_independently_across_autoreset() {
        use crate::apply::NoCustoms;

        // A frame-stacking (depth-2) single-image adapter that passes the image
        // through unchanged, so the stacked bytes are exactly the input frames.
        let adapter = stacked_adapter(2);
        let stack =
            |adapter: &ResolvedAdapter, tag: u8, episode: &str, buffers: &mut FrameBuffers| {
                let payload = assemble_obs(
                    adapter,
                    &tagged_obs(tag),
                    episode,
                    buffers,
                    &NoCustoms,
                    &NoEncodings,
                )
                .unwrap();
                let (shape, bytes) = cam_stack(&payload);
                assert_eq!(shape, vec![2, 1, 1, 3]); // (depth, *frame)
                bytes
            };

        // Two lanes (episodes ep-a, ep-b) share ONE FrameBuffers, keyed by
        // episode_id — the property that makes vectorized stateful correct.
        let mut buffers = FrameBuffers::new();

        // Step 0: each episode first-frame-pads independently.
        assert_eq!(stack(&adapter, 10, "ep-a", &mut buffers), vec![10; 6]);
        assert_eq!(stack(&adapter, 20, "ep-b", &mut buffers), vec![20; 6]);
        // Step 1: each episode slides independently (no cross-contamination).
        assert_eq!(
            stack(&adapter, 11, "ep-a", &mut buffers),
            vec![10, 10, 10, 11, 11, 11]
        );
        assert_eq!(
            stack(&adapter, 21, "ep-b", &mut buffers),
            vec![20, 20, 20, 21, 21, 21]
        );

        // Lane A autoresets: the old episode ends (evict) and a NEW episode_id
        // begins — the engine's per-lane reset edge.
        buffers.evict("ep-a");
        // The rolled lane first-frame-pads fresh: no leak from ep-a's history.
        assert_eq!(stack(&adapter, 30, "ep-c", &mut buffers), vec![30; 6]);
        // ...and the still-running lane B continues uncontaminated by either roll.
        assert_eq!(
            stack(&adapter, 22, "ep-b", &mut buffers),
            vec![21, 21, 21, 22, 22, 22]
        );
    }
}
