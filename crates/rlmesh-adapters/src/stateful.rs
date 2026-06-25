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
use crate::plans::ResolvedAdapter;

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
    /// Repack any custom-encoded observation payload keys, in place, **before**
    /// frame-stacking (so a stacked input stacks the model's representation).
    /// A no-op for routes with no observation encoding shims.
    fn repack_obs(&self, payload: &mut BTreeMap<String, Value>) -> Result<(), ApplyError>;

    /// Repack the model action's custom-encoded segments back to their base
    /// encoding, in place, **before** the native action conversion. A no-op for
    /// routes with no action encoding shims.
    fn repack_action(&self, action: &mut Value) -> Result<(), ApplyError>;
}

/// An [`EncodingTransform`] that touches nothing — for fully declarative routes
/// (no custom encodings). Parallels [`NoCustoms`](crate::apply::NoCustoms).
pub struct NoEncodings;

impl EncodingTransform for NoEncodings {
    fn repack_obs(&self, _payload: &mut BTreeMap<String, Value>) -> Result<(), ApplyError> {
        Ok(())
    }

    fn repack_action(&self, _action: &mut Value) -> Result<(), ApplyError> {
        Ok(())
    }
}

/// Per-route, episode-keyed frame-history buffers.
///
/// The handler holds one of these per `route_key`. The inner key is the
/// `episode_id` (a UUIDv4), **not** the lane index:
/// - autoreset reuses a lane's index across episodes, but `episode_id` is fresh
///   per episode, so old and new windows never collide;
/// - grouped predict can migrate an episode's slot index between groups, but
///   `episode_id` is stable, so the window follows the episode, not the index.
///
/// Eviction is edge-driven ([`seed`](Self::seed) at episode START,
/// [`evict`](Self::evict) at END, [`clear`](Self::clear) on close) — never on
/// absence.
#[derive(Default)]
pub struct FrameBuffers {
    /// `episode_id -> model_key -> rolling window` (each window `maxlen = depth`).
    inner: HashMap<String, BTreeMap<String, VecDeque<Tensor>>>,
}

impl FrameBuffers {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Seed an episode's (empty) buffer set at episode START. Returns `false` if
    /// the episode was already present — a missed END the handler asserts on
    /// rather than silently re-padding.
    pub fn seed(&mut self, episode_id: &str) -> bool {
        if self.inner.contains_key(episode_id) {
            return false;
        }
        self.inner.insert(episode_id.to_owned(), BTreeMap::new());
        true
    }

    /// Evict an episode's buffers at episode END or a close sweep.
    pub fn evict(&mut self, episode_id: &str) {
        self.inner.remove(episode_id);
    }

    /// Drop every episode's buffers (session shutdown / route close).
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Number of live episodes currently buffered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// The per-key window map for an episode, created lazily if absent.
    fn episode(&mut self, episode_id: &str) -> &mut BTreeMap<String, VecDeque<Tensor>> {
        self.inner.entry(episode_id.to_owned()).or_default()
    }
}

/// Push one frame into a rolling window and return the stacked tensor.
///
/// Reproduces the Python `_stack_frames` algorithm byte-for-byte: oldest-first
/// order on a new leading axis, first-frame replication padding at the start of
/// an episode (so the window is full from step zero), and a `maxlen = depth`
/// sliding window.
fn stack_frame(
    window: &mut VecDeque<Tensor>,
    frame: Tensor,
    depth: u32,
) -> Result<Tensor, ApplyError> {
    let depth = depth as usize;
    debug_assert!(
        depth <= MAX_STACK,
        "stack depth {depth} exceeds {MAX_STACK}"
    );
    // Start-of-episode pad: replicate the first observed frame so step 0 stacks
    // `[f0, f0, ..., f0]` rather than a short window.
    if window.is_empty() {
        for _ in 0..depth.saturating_sub(1) {
            window.push_back(frame.clone());
        }
    }
    if window.len() == depth {
        window.pop_front();
    }
    window.push_back(frame);
    Tensor::stack(window.make_contiguous())
        .map_err(|err| ApplyError::new(format!("frame-stack failed: {err}")))
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
) -> Result<BTreeMap<String, Value>, ApplyError> {
    let mut payload = adapter.transform_obs(raw_obs, customs)?;
    encodings.repack_obs(&mut payload)?;
    let stacks = adapter.stacks();
    if !stacks.is_empty() {
        let windows = buffers.episode(episode_id);
        for (model_key, depth) in &stacks {
            let Some(value) = payload.get_mut(model_key) else {
                continue;
            };
            let Value::Tensor(frame) = value else {
                return Err(ApplyError::new(format!(
                    "frame-stacked input '{model_key}' must be a tensor"
                )));
            };
            let window = windows.entry(model_key.clone()).or_default();
            let stacked = stack_frame(window, frame.clone(), *depth)?;
            *value = Value::Tensor(stacked);
        }
    }
    Ok(payload)
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
    raw_action: &Value,
    action_space: &SpaceSpec,
    encodings: &dyn EncodingTransform,
) -> Result<SpaceValue, ApplyError> {
    let mut action = raw_action.clone();
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

/// The top-level observation entry a (possibly dotted) plan key lives under.
/// The reserved `"."` denotes the flat/root observation, its own top-level key.
fn top_level_key(key: &str) -> &str {
    if key == "." {
        return ".";
    }
    key.split('.').next().unwrap_or(key)
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
/// map, keeping only the top-level entries `referenced` selects (their
/// `top_level_key`). Mirrors the binding's `decode_referenced_obs`: a `Dict`
/// env yields the selected top-level entries; any flat (non-`Dict`) env yields
/// the single leaf under the reserved `"."` key.
///
/// The caller selects the keys: a declarative-only route passes
/// [`ResolvedAdapter::referenced_obs_keys`]; a route with custom holes passes
/// all top-level keys so the custom callback sees the full observation
/// (materialized lazily, only when there are holes).
pub fn space_value_to_obs_map(
    value: &SpaceValue,
    space: &SpaceSpec,
    referenced: &BTreeSet<String>,
) -> Result<BTreeMap<String, Value>, ApplyError> {
    let mut out: BTreeMap<String, Value> = BTreeMap::new();
    match value {
        SpaceValue::Dict(values) => {
            let Some(SpaceKind::Dict(spec)) = space.spec.as_ref() else {
                return Err(ApplyError::new(
                    "dict observation value without a dict space".to_owned(),
                ));
            };
            let top: BTreeSet<&str> = referenced.iter().map(|key| top_level_key(key)).collect();
            for (key, child_space) in spec.keys.iter().zip(spec.spaces.iter()) {
                if top.contains(key.as_str())
                    && let Some(child) = values.get(key)
                {
                    out.insert(key.clone(), space_value_to_value(child, child_space)?);
                }
            }
        }
        _ => {
            // A flat (non-Dict) env is a single leaf, presented under the
            // reserved "." key the plan references for a StateLayout-tagged env.
            out.insert(".".to_owned(), space_value_to_value(value, space)?);
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
    fn stack_frame_first_frame_pads_then_slides() {
        // Reproduces Python `_stack_frames`: depth-1 copies of the first frame,
        // oldest-first, maxlen=depth sliding window.
        let mut window = VecDeque::new();

        let step0 = stack_frame(&mut window, frame(10), 3).expect("step0");
        assert_eq!(step0.shape(), &[3, 2]);
        // [f0, f0, f0]
        assert_eq!(
            step0.to_contiguous_bytes().as_ref(),
            &[10, 11, 10, 11, 10, 11]
        );

        // [f0, f0, f1]
        let step1 = stack_frame(&mut window, frame(20), 3).expect("step1");
        assert_eq!(
            step1.to_contiguous_bytes().as_ref(),
            &[10, 11, 10, 11, 20, 21]
        );

        // [f0, f1, f2]
        let step2 = stack_frame(&mut window, frame(30), 3).expect("step2");
        assert_eq!(
            step2.to_contiguous_bytes().as_ref(),
            &[10, 11, 20, 21, 30, 31]
        );

        // [f1, f2, f3] — oldest evicted (maxlen=3)
        let step3 = stack_frame(&mut window, frame(40), 3).expect("step3");
        assert_eq!(
            step3.to_contiguous_bytes().as_ref(),
            &[20, 21, 30, 31, 40, 41]
        );
    }

    #[test]
    fn frame_buffers_seed_and_evict_are_edge_driven() {
        let mut buffers = FrameBuffers::new();
        assert!(buffers.is_empty());
        assert!(buffers.seed("ep-a"));
        assert!(!buffers.seed("ep-a")); // already present -> missed END signal
        buffers.seed("ep-b");
        assert_eq!(buffers.len(), 2);
        buffers.evict("ep-a");
        assert_eq!(buffers.len(), 1);
        buffers.clear();
        assert!(buffers.is_empty());
    }

    #[test]
    fn bridge_flat_box_keys_under_dot() {
        let tensor = Tensor::from_vec(vec![1, 2, 3, 4], vec![4], DType::Uint8).expect("tensor");
        let value = SpaceValue::Box(tensor.clone());
        let space = flat(vec![4], DType::Uint8);
        let referenced: BTreeSet<String> = [".".to_owned()].into_iter().collect();

        let map = space_value_to_obs_map(&value, &space, &referenced).expect("bridge");
        assert_eq!(map.len(), 1);
        assert_eq!(map.get("."), Some(&Value::Tensor(tensor)));
    }

    #[test]
    fn bridge_discrete_becomes_number() {
        let value = SpaceValue::Discrete(7);
        let space = flat(vec![], DType::Int64);
        let referenced: BTreeSet<String> = [".".to_owned()].into_iter().collect();

        let map = space_value_to_obs_map(&value, &space, &referenced).expect("bridge");
        assert_eq!(map.get("."), Some(&Value::Number(7.0)));
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

    #[test]
    fn assemble_obs_stacks_each_episode_independently_across_autoreset() {
        use crate::apply::NoCustoms;
        use crate::plans::{ActionPlan, ImagePlan, ObsPlan, ResolvedAdapter};
        use crate::spec::ImageLayout;

        // A frame-stacking (depth-2) single-image adapter that passes the image
        // through unchanged (no resize/flip/normalize), so the stacked bytes are
        // exactly the input frames.
        let adapter = ResolvedAdapter {
            obs_plans: vec![ObsPlan::Image(ImagePlan {
                model_key: "cam".to_owned(),
                env_key: "cam".to_owned(),
                src_layout: ImageLayout::Hwc,
                dst_layout: ImageLayout::Hwc,
                flip: false,
                size: None,
                resample: "bilinear_aa".to_owned(),
                dtype: "uint8".to_owned(),
                normalize: false,
                lead_dims: 0,
                src_range: Some((0.0, 255.0)),
                stack: 2,
            })],
            action_plan: ActionPlan {
                segments: vec![],
                clip: None,
                in_dim: 0,
            },
        };
        // A 1x1x3 image whose every byte is `tag` — a per-frame fingerprint.
        let obs = |tag: u8| -> BTreeMap<String, Value> {
            let image = Tensor::from_vec(vec![tag, tag, tag], vec![1, 1, 3], DType::Uint8).unwrap();
            [("cam".to_owned(), Value::Tensor(image))]
                .into_iter()
                .collect()
        };
        let cam_bytes = |payload: &BTreeMap<String, Value>| -> Vec<u8> {
            match payload.get("cam").unwrap() {
                Value::Tensor(tensor) => {
                    assert_eq!(tensor.shape(), &[2, 1, 1, 3]); // (depth, *frame)
                    tensor.to_contiguous_bytes().into_owned()
                }
                other => panic!("cam not a tensor: {other:?}"),
            }
        };
        let stack =
            |adapter: &ResolvedAdapter, tag: u8, episode: &str, buffers: &mut FrameBuffers| {
                assemble_obs(
                    adapter,
                    &obs(tag),
                    episode,
                    buffers,
                    &NoCustoms,
                    &NoEncodings,
                )
                .unwrap()
            };

        // Two lanes (episodes ep-a, ep-b) share ONE FrameBuffers, keyed by
        // episode_id — the property that makes vectorized stateful correct.
        let mut buffers = FrameBuffers::new();

        // Step 0: each episode first-frame-pads independently.
        assert_eq!(
            cam_bytes(&stack(&adapter, 10, "ep-a", &mut buffers)),
            vec![10; 6]
        );
        assert_eq!(
            cam_bytes(&stack(&adapter, 20, "ep-b", &mut buffers)),
            vec![20; 6]
        );
        // Step 1: each episode slides independently (no cross-contamination).
        assert_eq!(
            cam_bytes(&stack(&adapter, 11, "ep-a", &mut buffers)),
            vec![10, 10, 10, 11, 11, 11]
        );
        assert_eq!(
            cam_bytes(&stack(&adapter, 21, "ep-b", &mut buffers)),
            vec![20, 20, 20, 21, 21, 21]
        );

        // Lane A autoresets: the old episode ends (evict) and a NEW episode_id
        // begins — the engine's per-lane reset edge.
        buffers.evict("ep-a");
        // The rolled lane first-frame-pads fresh: no leak from ep-a's history.
        assert_eq!(
            cam_bytes(&stack(&adapter, 30, "ep-c", &mut buffers)),
            vec![30; 6]
        );
        // ...and the still-running lane B continues uncontaminated by either roll.
        assert_eq!(
            cam_bytes(&stack(&adapter, 22, "ep-b", &mut buffers)),
            vec![21, 21, 21, 22, 22, 22]
        );
    }
}
