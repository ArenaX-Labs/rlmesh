//! Pure verdict and comparison helpers the checks share, ported unchanged from
//! the managed probe: action spreads, replay divergence, conformance, the
//! probe's actions and the contract-branch verdict.

use rlmesh_spaces::spaces::SpaceKind;
use rlmesh_spaces::{
    BoxBounds, BoxSpec, Conformance, DType, Scalar, SpaceSpec, SpaceValue, Tensor, assemble_value,
    conform, decode_scalars, encode_scalars, leaf_specs,
};
use serde_json::Value;

use super::report::{CheckStatus, ProbeCheck};

/// Renderer jitter tolerated on image leaves, in raw units (LSBs for uint8).
pub(super) const IMAGE_REPLAY_ATOL: f64 = 2.0;
/// Fraction of image elements allowed beyond the jitter tolerance: rasterizers
/// flip a few edge/shadow pixels hard between identical frames (SimplerEnv
/// widowx: ~0.5% of pixels at up to 149 LSBs with identical physics state).
pub(super) const IMAGE_REPLAY_MAX_FRACTION: f64 = 0.01;
/// Base seed for the action the latency loop steps with; `+ step` per step.
pub(super) const ENV_STEP_SEED: u64 = 0x5EED_0057;
/// The contract branch a tagged env stamps on its own metadata.
pub(super) const ENV_BRANCH_METADATA_KEY: &str = rlmesh_adapters::v1::ENV_BRANCH_METADATA_KEY;

/// The actions of one predict: frames (one per executed action) x lanes.
pub(super) type Frames = Vec<Vec<SpaceValue>>;

/// Structural JSON equality that ignores int/float representation and treats
/// a missing object key as null, returning the first differing path.
pub(super) fn first_difference(live: &Value, declared: &Value) -> Option<String> {
    diff_at(live, declared, "$")
}

fn diff_at(live: &Value, declared: &Value, path: &str) -> Option<String> {
    match (live, declared) {
        (Value::Number(a), Value::Number(b)) => {
            (a.as_f64() != b.as_f64()).then(|| format!("{path}: {a} vs declared {b}"))
        }
        (Value::Array(a), Value::Array(b)) => {
            if a.len() != b.len() {
                return Some(format!(
                    "{path}: {} item(s) vs declared {}",
                    a.len(),
                    b.len()
                ));
            }
            a.iter()
                .zip(b)
                .enumerate()
                .find_map(|(i, (x, y))| diff_at(x, y, &format!("{path}[{i}]")))
        }
        (Value::Object(a), Value::Object(b)) => {
            let mut keys: Vec<&String> = a.keys().chain(b.keys()).collect();
            keys.sort();
            keys.dedup();
            keys.into_iter().find_map(|key| {
                let x = a.get(key).unwrap_or(&Value::Null);
                let y = b.get(key).unwrap_or(&Value::Null);
                diff_at(x, y, &format!("{path}.{key}"))
            })
        }
        (a, b) if a == b => None,
        (a, b) => Some(format!("{path}: {a} vs declared {b}")),
    }
}

/// The `env_contract` verdict, as a pure function of the three things it
/// compares: the image's declared branch table, the branch the running env
/// stamped, and the tags it is serving (`Value::Null` for any of them means
/// "absent").
pub(super) fn env_contract_check(
    contracts: &Value,
    served_branch: &Value,
    served_tags: &Value,
) -> ProbeCheck {
    let name = "env_contract";
    if !contracts.is_object() {
        return ProbeCheck::skipped(
            name,
            "the image's describe envelope declares no contract branches (a factory with no tag_params serves one contract)",
        );
    }
    if let Some(error) = contracts.get("error").and_then(Value::as_str) {
        return ProbeCheck::new(
            name,
            CheckStatus::Fail,
            format!(
                "the image's contract-branch table is an error badge: {error}; nothing can say which protocol this env serves"
            ),
        )
        .tag("env_contracts_unusable");
    }
    if !served_branch.is_object() {
        return ProbeCheck::new(
            name,
            CheckStatus::Fail,
            format!(
                "the image declares contract branches but the running env stamps no {ENV_BRANCH_METADATA_KEY}; nothing can tell which branch it is serving"
            ),
        )
        .tag("env_branch_missing");
    }
    let Some(branches) = contracts.get("branches").and_then(Value::as_array) else {
        return ProbeCheck::new(
            name,
            CheckStatus::Fail,
            "the image's contract-branch table carries no branches list".to_string(),
        )
        .tag("env_contracts_unusable");
    };
    let Some(declared) = branches
        .iter()
        .find(|branch| branch.get("params") == Some(served_branch))
    else {
        let known: Vec<String> = branches
            .iter()
            .filter_map(|branch| branch.get("params"))
            .map(ToString::to_string)
            .collect();
        return ProbeCheck::new(
            name,
            CheckStatus::Fail,
            format!(
                "the env serves contract branch {served_branch}, which its own describe envelope does not declare (it lists {}); the image's label and its code disagree",
                known.join(", ")
            ),
        )
        .tag("env_branch_undeclared");
    };
    let declared_tags = declared.get("env_tags").cloned().unwrap_or(Value::Null);
    match first_difference(served_tags, &declared_tags) {
        Some(diff) => ProbeCheck::new(
            name,
            CheckStatus::Fail,
            format!(
                "the env serves contract branch {served_branch} but its tags differ from the ones that branch declares: {diff}"
            ),
        )
        .tag("env_contract_tags_mismatch"),
        None => ProbeCheck::new(
            name,
            CheckStatus::Pass,
            format!(
                "the env serves contract branch {served_branch}, and its live tags match the ones that branch declares"
            ),
        ),
    }
}

pub(super) enum Drift {
    Shape,
    Within(f64),
    Beyond(f64),
}

pub(super) fn replay_drift(delta: Option<f64>, tolerance: f64) -> Drift {
    match delta {
        None => Drift::Shape,
        Some(delta) if delta <= tolerance => Drift::Within(delta),
        Some(delta) => Drift::Beyond(delta),
    }
}

/// Verdict for a model whose A,B,A replay drifted: informational if a fresh
/// episode brings input A back to its first answer, a failure if the state
/// survives the episode end.
pub(super) fn stateful_check(
    replay: f64,
    tolerance: f64,
    after_reset: Result<Option<f64>, String>,
) -> ProbeCheck {
    let check = ProbeCheck::new("stateless", CheckStatus::Warn, String::new())
        .tag("stateful")
        .metric("replayMaxAbsDiff", replay)
        .metric("tolerance", tolerance);
    match after_reset {
        Ok(Some(after)) if after <= tolerance => ProbeCheck {
            detail: format!(
                "replaying input A around input B drifted {replay:.3e} (> tolerance {tolerance:.3e}); the model carries per-episode state that the episode end clears"
            ),
            ..check
        }
        .tag("resets_on_episode_end")
        .metric("resetMaxAbsDiff", after),
        Ok(Some(after)) => ProbeCheck {
            status: CheckStatus::Fail,
            detail: format!(
                "state survives the episode end: input A replayed in a fresh episode still drifted {after:.3e} (> tolerance {tolerance:.3e})"
            ),
            ..check
        }
        .metric("resetMaxAbsDiff", after),
        Ok(None) => ProbeCheck {
            status: CheckStatus::Fail,
            detail: "input A replayed in a fresh episode returned incomparable action shapes"
                .to_string(),
            ..check
        }
        .tag("nondeterministic_shape"),
        Err(error) => ProbeCheck {
            status: CheckStatus::Fail,
            detail: format!("after stateful drift {replay:.3e}: {error}"),
            ..check
        },
    }
}

pub(super) fn observation_conformance(
    name: &str,
    lanes: &[SpaceValue],
    space: &SpaceSpec,
) -> ProbeCheck {
    let mut range_warning = None;
    for (lane, observation) in lanes.iter().enumerate() {
        match conform(space, observation) {
            Conformance::Ok => {}
            Conformance::Structural(err) => {
                return ProbeCheck::new(
                    name,
                    CheckStatus::Fail,
                    format!("lane {lane} observation does not fit the observation space: {err}"),
                )
                .tag("observation_shape_mismatch");
            }
            Conformance::Range(err) => {
                range_warning.get_or_insert(format!("lane {lane}: {err}"));
            }
        }
    }
    match range_warning {
        Some(detail) => ProbeCheck::new(
            name,
            CheckStatus::Warn,
            format!("step observation out of range: {detail}"),
        )
        .tag("observation_out_of_range"),
        None => ProbeCheck::new(
            name,
            CheckStatus::Pass,
            format!(
                "step observation for {} lane(s) conforms to the observation space",
                lanes.len()
            ),
        ),
    }
}

pub(super) fn shape_check(
    name: &str,
    frames: &Frames,
    rows: usize,
    action: &SpaceSpec,
) -> ProbeCheck {
    let mut range_warning = None;
    for (frame_index, frame) in frames.iter().enumerate() {
        for (lane, value) in frame.iter().enumerate() {
            match conform(action, value) {
                Conformance::Ok => {}
                Conformance::Structural(err) => {
                    return ProbeCheck::new(
                        name,
                        CheckStatus::Fail,
                        format!(
                            "frame {frame_index} lane {lane} does not fit the action space: {err}"
                        ),
                    )
                    .tag("action_shape_mismatch");
                }
                Conformance::Range(err) => {
                    range_warning.get_or_insert(format!("frame {frame_index} lane {lane}: {err}"));
                }
            }
        }
    }
    let check = ProbeCheck::new(name, CheckStatus::Pass, String::new())
        .metric("frames", frames.len() as f64)
        .metric("rows", rows as f64);
    match range_warning {
        Some(detail) => ProbeCheck {
            status: CheckStatus::Warn,
            detail: format!("actions out of range: {detail}"),
            ..check
        }
        .tag("action_out_of_range"),
        None => ProbeCheck {
            detail: format!(
                "{} frame(s) x {rows} row(s) conform to the action space",
                frames.len()
            ),
            ..check
        },
    }
}

pub(super) struct Replay {
    /// The reset observation followed by one per step.
    pub(super) observations: Vec<SpaceValue>,
    pub(super) rewards: Vec<f64>,
    pub(super) dones: Vec<bool>,
    /// How long the reset took; episode setup that dominates a short run is
    /// worth reporting even when it is perfectly reproducible.
    pub(super) reset_ms: f64,
}

/// Where two replays part ways: index 0 is the reset, `n` the n-th step.
#[derive(Debug, Default, PartialEq)]
pub(super) struct ReplayDivergence {
    pub(super) steps: usize,
    pub(super) state_max: f64,
    pub(super) image_max: f64,
    /// Largest per-leaf fraction of image elements beyond `IMAGE_REPLAY_ATOL`.
    pub(super) image_over_fraction: f64,
    pub(super) state_first: Option<usize>,
    pub(super) image_first: Option<usize>,
}

impl ReplayDivergence {
    fn state(&mut self, index: usize, diff: f64) {
        self.state_max = self.state_max.max(diff);
        if diff > 0.0 {
            self.state_first.get_or_insert(index);
        }
    }

    fn image(&mut self, index: usize, stats: &DiffStats) {
        self.image_max = self.image_max.max(stats.max);
        let over = stats.over_atol_fraction();
        self.image_over_fraction = self.image_over_fraction.max(over);
        if over > IMAGE_REPLAY_MAX_FRACTION {
            self.image_first.get_or_insert(index);
        }
    }
}

/// `None` when any observation pair has incomparable shapes.
pub(super) fn compare_replays(
    a: &Replay,
    b: &Replay,
    space: &SpaceSpec,
    image_paths: &[Vec<String>],
) -> Option<ReplayDivergence> {
    let mut divergence = ReplayDivergence {
        steps: a
            .observations
            .len()
            .min(b.observations.len())
            .saturating_sub(1),
        ..ReplayDivergence::default()
    };
    for (index, (obs_a, obs_b)) in a.observations.iter().zip(&b.observations).enumerate() {
        let mut leaves = Vec::new();
        leaf_max_diffs(space, obs_a, obs_b, Vec::new(), &mut leaves)?;
        for (path, stats) in leaves {
            if image_paths.contains(&path) {
                divergence.image(index, &stats);
            } else {
                divergence.state(index, stats.max);
            }
        }
    }
    for (step, (reward_a, reward_b)) in a.rewards.iter().zip(&b.rewards).enumerate() {
        divergence.state(step + 1, (reward_a - reward_b).abs());
    }
    for (step, (done_a, done_b)) in a.dones.iter().zip(&b.dones).enumerate() {
        divergence.state(step + 1, f64::from(done_a != done_b));
    }
    // One run ending earlier than the other is itself a state divergence.
    if a.observations.len() != b.observations.len() {
        divergence.state(divergence.steps, 1.0);
    }
    Some(divergence)
}

/// Max |a-b| per leaf, keyed by path (tuple positions as digits), walking
/// `space`; `None` on an incomparable leaf.
fn leaf_max_diffs(
    space: &SpaceSpec,
    a: &SpaceValue,
    b: &SpaceValue,
    path: Vec<String>,
    out: &mut Vec<(Vec<String>, DiffStats)>,
) -> Option<()> {
    match (space.spec.as_ref(), a, b) {
        (Some(SpaceKind::Dict(spec)), SpaceValue::Dict(a), SpaceValue::Dict(b)) => {
            for (key, child) in spec.keys.iter().zip(&spec.spaces) {
                let mut child_path = path.clone();
                child_path.push(key.clone());
                leaf_max_diffs(child, a.get(key)?, b.get(key)?, child_path, out)?;
            }
            Some(())
        }
        (Some(SpaceKind::Tuple(spec)), SpaceValue::Tuple(a), SpaceValue::Tuple(b)) => {
            for (index, child) in spec.spaces.iter().enumerate() {
                let mut child_path = path.clone();
                child_path.push(index.to_string());
                leaf_max_diffs(child, a.get(index)?, b.get(index)?, child_path, out)?;
            }
            Some(())
        }
        _ => {
            out.push((path, value_diff_stats(a, b)?));
            Some(())
        }
    }
}

/// A Box leaf that reads as a camera frame: uint8, HW, HWC or CHW with 1, 3
/// or 4 channels.
pub(super) fn is_image_space(space: &SpaceSpec) -> bool {
    matches!(space.spec.as_ref(), Some(SpaceKind::Box(_)))
        && space.dtype == DType::Uint8
        && matches!(
            space.shape.as_slice(),
            [_, _] | [_, _, 1 | 3 | 4] | [1 | 3 | 4, _, _]
        )
}

/// Paths of the image-like leaves of `space`, in declaration order.
pub(super) fn image_leaves(space: &SpaceSpec, path: Vec<String>, out: &mut Vec<Vec<String>>) {
    match space.spec.as_ref() {
        Some(SpaceKind::Box(_)) if is_image_space(space) => out.push(path),
        Some(SpaceKind::Dict(spec)) => {
            for (key, child) in spec.keys.iter().zip(&spec.spaces) {
                let mut child_path = path.clone();
                child_path.push(key.clone());
                image_leaves(child, child_path, out);
            }
        }
        _ => {}
    }
}

/// The safe action the conformance step sends: zero pulled into range on every
/// Box element. `sample_seeded` draws N(0,1) for a `Box(-inf, inf)`, and
/// simpler_env, robotwin and absolute-pose LIBERO all declare one, so a random
/// draw can throw the arm out of the scene and fail a step the image serves
/// fine.
pub(super) fn neutral_action(space: &SpaceSpec) -> Result<SpaceValue, String> {
    probe_action(space, None)
}

/// The latency loop's action: a seeded draw wherever the Box element is
/// bounded (so the preview the loop captures shows a scene that moves), and
/// the neutral value wherever a bound is open (so an unbounded space is still
/// not a lottery).
pub(super) fn bounded_action(space: &SpaceSpec, seed: u64) -> Result<SpaceValue, String> {
    probe_action(space, Some(seed))
}

fn probe_action(space: &SpaceSpec, seed: Option<u64>) -> Result<SpaceValue, String> {
    let leaves = leaf_specs(space)
        .into_iter()
        .map(|spec| action_leaf(spec, seed))
        .collect::<Result<Vec<_>, _>>()?;
    assemble_value(space, leaves).map_err(|err| format!("build action: {err}"))
}

fn action_leaf(spec: &SpaceSpec, seed: Option<u64>) -> Result<SpaceValue, String> {
    let sample = |seed: u64| {
        rlmesh_spaces::sample_seeded(spec, seed).map_err(|err| format!("sample action: {err}"))
    };
    let Some(SpaceKind::Box(box_spec)) = spec.spec.as_ref() else {
        // Discrete and friends are finite by construction: nothing to blow up.
        return sample(seed.unwrap_or(ENV_STEP_SEED));
    };
    let dtype = match spec.dtype {
        DType::Unspecified => DType::Float32,
        dtype => dtype,
    };
    let numel = spec.shape.iter().product::<i64>().max(0) as usize;
    let (low, high) = box_bounds(box_spec, numel, dtype);
    let drawn = match seed {
        Some(seed) => match sample(seed)? {
            SpaceValue::Box(tensor) => decode_scalars(&tensor.to_contiguous_bytes(), dtype).ok(),
            _ => None,
        },
        None => None,
    };
    let scalars: Vec<Scalar> = low
        .iter()
        .zip(&high)
        .enumerate()
        .map(|(index, (low, high))| {
            match drawn.as_ref().and_then(|drawn| drawn.get(index)) {
                // A finite element's draw is already in range; an open bound
                // is the only place the sampler's tails can hurt.
                Some(scalar) if low.is_finite() && high.is_finite() => *scalar,
                _ => neutral_scalar(*low, *high, dtype),
            }
        })
        .collect();
    let bytes = encode_scalars(&scalars, dtype).map_err(|err| format!("encode action: {err}"))?;
    Tensor::from_slice(&bytes, &spec.shape, dtype)
        .map(SpaceValue::Box)
        .map_err(|err| format!("action tensor: {err}"))
}

/// The neutral value of one action element: the middle of a fully bounded
/// `[low, high]` -- a joint whose limits are `[2, 5]` rests at 3.5, not pinned
/// against its low stop -- and zero pulled into range when either end is open,
/// where there is no middle to take. `max` then `min` rather than `clamp`,
/// which panics on inverted or NaN bounds a probed image is free to declare.
fn neutral_scalar(low: f64, high: f64, dtype: DType) -> Scalar {
    let clamped_zero = 0.0_f64.max(low).min(high);
    let neutral = if low.is_finite() && high.is_finite() {
        low + (high - low) / 2.0
    } else {
        clamped_zero
    };
    match dtype {
        // A bool element has no meaningful midpoint: false stays the rest.
        DType::Bool => Scalar::Bool(clamped_zero != 0.0),
        DType::Float16 | DType::Float32 | DType::Float64 => Scalar::Float(neutral),
        // Integer Box bounds are inclusive over `[ceil(low), floor(high)]`,
        // as in the sampler.
        _ => Scalar::Int(neutral.max(low.ceil()).min(high.floor()) as i64),
    }
}

/// `low` and `high` per element, defaulting to the unbounded pair. The
/// sampler's own version of this is private to rlmesh-spaces.
fn box_bounds(spec: &BoxSpec, numel: usize, dtype: DType) -> (Vec<f64>, Vec<f64>) {
    // The sampler reads a TypedUniform bound as its first scalar whatever the
    // buffer holds, so a buffer sized for neither reading still bounds the
    // leaf here instead of falling back to the unbounded fill and drawing a
    // neutral the space rejects.
    let typed = |bytes: &[u8], fill: f64| -> Vec<f64> {
        match decode_scalars(bytes, dtype) {
            Ok(scalars) if scalars.len() == numel => {
                scalars.into_iter().map(|s| s.to_f64(dtype)).collect()
            }
            Ok(scalars) if !scalars.is_empty() => vec![scalars[0].to_f64(dtype); numel],
            _ => vec![fill; numel],
        }
    };
    let spread = |values: &[f64], fill: f64| -> Vec<f64> {
        if values.len() == numel {
            values.to_vec()
        } else {
            vec![fill; numel]
        }
    };
    match spec.bounds.as_ref() {
        Some(BoxBounds::Uniform(bounds)) => (vec![bounds.low; numel], vec![bounds.high; numel]),
        Some(BoxBounds::Elementwise(bounds)) => (
            spread(&bounds.low, f64::NEG_INFINITY),
            spread(&bounds.high, f64::INFINITY),
        ),
        Some(BoxBounds::TypedUniform(bounds)) => (
            typed(&bounds.low, f64::NEG_INFINITY),
            typed(&bounds.high, f64::INFINITY),
        ),
        Some(BoxBounds::TypedElementwise(bounds)) => (
            typed(&bounds.low, f64::NEG_INFINITY),
            typed(&bounds.high, f64::INFINITY),
        ),
        Some(BoxBounds::Unbounded(_)) | None => {
            (vec![f64::NEG_INFINITY; numel], vec![f64::INFINITY; numel])
        }
    }
}

/// Widest finite `high - low` over the Box leaves of `space`; `None` when
/// nothing is bounded, so a spread has no scale to be read against.
pub(super) fn action_range(space: &SpaceSpec) -> Option<f64> {
    let widest = |spaces: &[SpaceSpec]| spaces.iter().filter_map(action_range).reduce(f64::max);
    match space.spec.as_ref()? {
        SpaceKind::Dict(dict) => widest(&dict.spaces),
        SpaceKind::Tuple(tuple) => widest(&tuple.spaces),
        SpaceKind::Box(spec) => {
            let dtype = match space.dtype {
                DType::Unspecified => DType::Float32,
                dtype => dtype,
            };
            let typed = |bytes: &[u8]| -> Option<Vec<f64>> {
                let scalars = decode_scalars(bytes, dtype).ok()?;
                Some(scalars.into_iter().map(|s| s.to_f64(dtype)).collect())
            };
            let (low, high) = match spec.bounds.as_ref()? {
                BoxBounds::Uniform(bounds) => (vec![bounds.low], vec![bounds.high]),
                BoxBounds::Elementwise(bounds) => (bounds.low.clone(), bounds.high.clone()),
                BoxBounds::TypedUniform(bounds) => (typed(&bounds.low)?, typed(&bounds.high)?),
                BoxBounds::TypedElementwise(bounds) => (typed(&bounds.low)?, typed(&bounds.high)?),
                BoxBounds::Unbounded(_) => return None,
            };
            low.iter()
                .zip(&high)
                .map(|(low, high)| high - low)
                .filter(|span| span.is_finite() && *span > 0.0)
                .reduce(f64::max)
        }
        _ => None,
    }
}

/// " (12.3% of the action range)" when the range is known, else "".
pub(super) fn range_fraction_note(diff: f64, range: Option<f64>) -> String {
    range.map_or_else(String::new, |range| {
        format!(" ({:.1}% of the action range)", diff / range * 100.0)
    })
}

/// Element-wise |a-b| summary over one comparison (or several merged ones).
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub(super) struct DiffStats {
    pub(super) max: f64,
    sum: f64,
    pub(super) count: usize,
    differing: usize,
    over_atol: usize,
}

impl DiffStats {
    fn scalar(diff: f64) -> Self {
        Self {
            max: diff,
            sum: diff,
            count: 1,
            differing: usize::from(diff != 0.0),
            over_atol: usize::from(diff > IMAGE_REPLAY_ATOL),
        }
    }

    fn merge(self, other: Self) -> Self {
        Self {
            max: self.max.max(other.max),
            sum: self.sum + other.sum,
            count: self.count + other.count,
            differing: self.differing + other.differing,
            over_atol: self.over_atol + other.over_atol,
        }
    }

    fn over_atol_fraction(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.over_atol as f64 / self.count as f64
        }
    }

    fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    fn differing_fraction(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.differing as f64 / self.count as f64
        }
    }
}

/// Spread of every later trial against the first, plus the per-predict
/// element count; `None` when any trial has an incomparable shape.
pub(super) struct TrialSpread {
    pub(super) max: f64,
    mean_sum: f64,
    differing_sum: f64,
    comparisons: usize,
    pub(super) elements: usize,
}

impl TrialSpread {
    pub(super) fn mean(&self) -> f64 {
        if self.comparisons == 0 {
            0.0
        } else {
            self.mean_sum / self.comparisons as f64
        }
    }

    pub(super) fn differing_fraction(&self) -> f64 {
        if self.comparisons == 0 {
            0.0
        } else {
            self.differing_sum / self.comparisons as f64
        }
    }
}

pub(super) fn trials_spread(trials: &[Frames]) -> Option<TrialSpread> {
    let (first, rest) = trials.split_first()?;
    let mut spread = TrialSpread {
        max: 0.0,
        mean_sum: 0.0,
        differing_sum: 0.0,
        comparisons: 0,
        elements: frames_diff_stats(first, first)?.count,
    };
    for trial in rest {
        let stats = frames_diff_stats(first, trial)?;
        spread.max = spread.max.max(stats.max);
        spread.mean_sum += stats.mean();
        spread.differing_sum += stats.differing_fraction();
        spread.comparisons += 1;
    }
    Some(spread)
}

pub(super) fn frames_max_abs_diff(a: &Frames, b: &Frames) -> Option<f64> {
    frames_diff_stats(a, b).map(|stats| stats.max)
}

fn frames_diff_stats(a: &Frames, b: &Frames) -> Option<DiffStats> {
    if a.len() != b.len() {
        return None;
    }
    let mut stats = DiffStats::default();
    for (frame_a, frame_b) in a.iter().zip(b) {
        if frame_a.len() != frame_b.len() {
            return None;
        }
        for (x, y) in frame_a.iter().zip(frame_b) {
            stats = stats.merge(value_diff_stats(x, y)?);
        }
    }
    Some(stats)
}

fn value_diff_stats(left: &SpaceValue, right: &SpaceValue) -> Option<DiffStats> {
    let fold = |diffs: &mut dyn Iterator<Item = f64>| {
        diffs
            .map(DiffStats::scalar)
            .fold(DiffStats::default(), DiffStats::merge)
    };
    match (left, right) {
        (SpaceValue::Box(a), SpaceValue::Box(b)) => {
            if a.shape() != b.shape() || a.dtype() != b.dtype() {
                return None;
            }
            let xs = tensor_f64(a)?;
            let ys = tensor_f64(b)?;
            Some(fold(&mut xs.iter().zip(&ys).map(|(x, y)| (x - y).abs())))
        }
        (SpaceValue::Discrete(a), SpaceValue::Discrete(b)) => {
            Some(DiffStats::scalar((a - b).abs() as f64))
        }
        (SpaceValue::MultiBinary(a), SpaceValue::MultiBinary(b)) if a.len() == b.len() => {
            Some(fold(&mut a.iter().zip(b).map(|(x, y)| f64::from(x != y))))
        }
        (SpaceValue::MultiDiscrete(a), SpaceValue::MultiDiscrete(b)) if a.len() == b.len() => Some(
            fold(&mut a.iter().zip(b).map(|(x, y)| (x - y).abs() as f64)),
        ),
        (SpaceValue::Text(a), SpaceValue::Text(b)) => Some(DiffStats::scalar(f64::from(a != b))),
        (SpaceValue::Dict(a), SpaceValue::Dict(b)) if a.len() == b.len() => {
            let mut stats = DiffStats::default();
            for (key, x) in a {
                stats = stats.merge(value_diff_stats(x, b.get(key)?)?);
            }
            Some(stats)
        }
        (SpaceValue::Tuple(a), SpaceValue::Tuple(b)) if a.len() == b.len() => {
            let mut stats = DiffStats::default();
            for (x, y) in a.iter().zip(b) {
                stats = stats.merge(value_diff_stats(x, y)?);
            }
            Some(stats)
        }
        _ => None,
    }
}

fn tensor_f64(tensor: &Tensor) -> Option<Vec<f64>> {
    let dtype: DType = tensor.dtype();
    let scalars = decode_scalars(&tensor.to_contiguous_bytes(), dtype).ok()?;
    Some(scalars.into_iter().map(|s| s.to_f64(dtype)).collect())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use rlmesh_spaces::spaces::{BoxSpaceBuilder, DictSpaceBuilder};
    use serde_json::json;

    use super::*;

    // simpler_env, robotwin and absolute-pose LIBERO declare half-open action
    // boxes, and the blocking conformance step must not be a lottery on them.
    // The latency loop still has to move the scene it previews, so only its
    // open elements go neutral.
    #[test]
    fn probe_actions_stay_in_range_and_zero_only_the_open_elements() {
        let space = BoxSpaceBuilder::tensor(
            vec![2.0, f64::NEG_INFINITY, 2.0, f64::NEG_INFINITY],
            vec![5.0, 5.0, f64::INFINITY, f64::INFINITY],
            vec![4],
        )
        .dtype(DType::Float64)
        .build()
        .expect("box space");
        let values = |action: SpaceValue| -> Vec<f64> {
            assert!(
                matches!(conform(&space, &action), Conformance::Ok),
                "action must conform to its own space: {action:?}"
            );
            let SpaceValue::Box(tensor) = action else {
                panic!("expected a Box value, got {action:?}");
            };
            tensor_f64(&tensor).expect("decode")
        };

        // A bounded element rests at its midpoint; an element with an open end
        // has no midpoint, so zero pulled into range is the rest.
        const NEUTRAL: [f64; 4] = [3.5, 0.0, 2.0, 0.0];
        assert_eq!(values(neutral_action(&space).expect("neutral")), NEUTRAL);

        let first = values(bounded_action(&space, ENV_STEP_SEED).expect("bounded"));
        let second = values(bounded_action(&space, ENV_STEP_SEED + 1).expect("bounded"));
        for values in [&first, &second] {
            assert!((2.0..=5.0).contains(&values[0]), "{values:?}");
            assert_eq!(&values[1..], &NEUTRAL[1..], "open elements stay neutral");
        }
        assert_ne!(
            first[0], second[0],
            "the bounded element must move between steps, or the preview is a still"
        );
    }

    /// rlmesh-spaces reads a TypedUniform bound as its first scalar however
    /// long the buffer is, so a pair sized for neither one-per-element nor one
    /// scalar must still bound the leaf here.
    #[test]
    fn typed_bounds_broadcast_their_first_scalar_like_the_sampler() {
        let pair = |value: f64| {
            encode_scalars(
                &[Scalar::Float(value), Scalar::Float(value + 1.0)],
                DType::Float64,
            )
            .expect("encode bounds")
        };
        let spec = BoxSpec {
            bounds: Some(BoxBounds::TypedUniform(rlmesh_spaces::TypedUniformBounds {
                low: pair(2.0),
                high: pair(5.0),
            })),
        };
        assert_eq!(
            box_bounds(&spec, 3, DType::Float64),
            (vec![2.0; 3], vec![5.0; 3])
        );
    }

    /// Same-width protocol swaps pass every other check silently, because
    /// `join` never compares role *names* against bounds. Each arm here is one
    /// way the swap shows up.
    #[test]
    fn env_contract_gates_the_branch_the_image_actually_serves() {
        let delta_tags = json!({
            "observation": {"eef": {"kind": "state", "role": "proprio/eef_pos"}},
            "action": {"components": [{"role": "action/delta_eef_pos", "dim": 3}]},
        });
        let abs_tags = json!({
            "observation": {"eef": {"kind": "state", "role": "proprio/eef_pos"}},
            "action": {"components": [{"role": "action/eef_pos", "dim": 3}]},
        });
        let contracts = json!({
            "discriminants": ["action_type"],
            "branches": [
                {"params": {"action_type": "delta"}, "default": true, "env_tags": delta_tags},
                {"params": {"action_type": "abs"}, "default": false, "env_tags": abs_tags},
            ],
        });
        let delta = json!({"action_type": "delta"});
        let abs = json!({"action_type": "abs"});
        let cases = [
            // No discriminants: one contract, nothing to gate.
            (
                Value::Null,
                Value::Null,
                Value::Null,
                CheckStatus::Skip,
                None,
            ),
            (
                contracts.clone(),
                abs.clone(),
                abs_tags.clone(),
                CheckStatus::Pass,
                None,
            ),
            (
                contracts.clone(),
                delta.clone(),
                delta_tags.clone(),
                CheckStatus::Pass,
                None,
            ),
            (
                contracts.clone(),
                Value::Null,
                delta_tags.clone(),
                CheckStatus::Fail,
                Some("env_branch_missing"),
            ),
            (
                contracts.clone(),
                json!({"action_type": "joint"}),
                delta_tags.clone(),
                CheckStatus::Fail,
                Some("env_branch_undeclared"),
            ),
            // The swap the gate exists for: absolute targets served under a
            // label that declares deltas.
            (
                contracts.clone(),
                abs.clone(),
                delta_tags.clone(),
                CheckStatus::Fail,
                Some("env_contract_tags_mismatch"),
            ),
            (
                json!({"error": "make() needs a GPU"}),
                abs.clone(),
                abs_tags.clone(),
                CheckStatus::Fail,
                Some("env_contracts_unusable"),
            ),
        ];
        for (contracts, branch, tags, status, tag) in cases {
            let check = env_contract_check(&contracts, &branch, &tags);
            assert_eq!(check.status, status, "{branch}: {}", check.detail);
            assert_eq!(check.tags, tag.into_iter().collect::<Vec<_>>());
        }
    }

    #[test]
    fn differences_ignore_number_representation_and_name_the_path() {
        let live = json!({"kind": "box", "shape": [3], "details": {"low": -1.0}});
        assert_eq!(
            first_difference(
                &live,
                &json!({"kind": "box", "shape": [3], "details": {"low": -1}})
            ),
            None
        );
        assert_eq!(
            first_difference(
                &live,
                &json!({"kind": "box", "shape": [4], "details": {"low": -1}})
            )
            .as_deref(),
            Some("$.shape[0]: 3 vs declared 4")
        );
        assert!(
            first_difference(&live, &json!({"kind": "box", "shape": [3]}))
                .unwrap()
                .starts_with("$.details")
        );
    }

    #[test]
    fn max_abs_diff_covers_scalar_and_composite_values() {
        let a = SpaceValue::Tuple(vec![
            SpaceValue::Discrete(3),
            SpaceValue::MultiDiscrete(vec![1, 5]),
        ]);
        let b = SpaceValue::Tuple(vec![
            SpaceValue::Discrete(1),
            SpaceValue::MultiDiscrete(vec![1, 4]),
        ]);
        let stats = value_diff_stats(&a, &b).expect("comparable");
        assert_eq!(stats.max, 2.0);
        assert_eq!((stats.count, stats.differing), (3, 2));
        assert_eq!(stats.mean(), 1.0);
        assert_eq!(value_diff_stats(&a, &SpaceValue::Discrete(1)), None);
        let frames_a: Frames = vec![vec![a.clone()]];
        let frames_b: Frames = vec![vec![a.clone()], vec![b.clone()]];
        assert_eq!(frames_max_abs_diff(&frames_a, &frames_b), None);
        assert_eq!(frames_max_abs_diff(&frames_a, &frames_a), Some(0.0));

        let spread = trials_spread(&[frames_a.clone(), frames_a.clone(), vec![vec![b]]])
            .expect("comparable trials");
        assert_eq!(
            (spread.max, spread.elements, spread.comparisons),
            (2.0, 3, 2)
        );
        assert!((spread.mean() - 0.5).abs() < 1e-12);
        assert!((spread.differing_fraction() - 1.0 / 3.0).abs() < 1e-12);
        assert!(trials_spread(&[frames_a, frames_b]).is_none());
        assert!(trials_spread(&[]).is_none());
    }

    fn image_and_state_space() -> SpaceSpec {
        let image = BoxSpaceBuilder::scalar(0.0, 255.0, vec![10, 100, 1])
            .dtype(DType::Uint8)
            .build()
            .expect("image");
        let state = BoxSpaceBuilder::scalar(-1.0, 1.0, vec![2])
            .dtype(DType::Float32)
            .build()
            .expect("state");
        DictSpaceBuilder::new()
            .insert("cam", image)
            .insert("pos", state)
            .build()
            .expect("dict")
    }

    /// One observation of [`image_and_state_space`]: `flipped` of its 1000
    /// pixels at `10 + jump`, the rest at 10.
    fn observation(flipped: usize, jump: u8, pos: f32) -> SpaceValue {
        let mut pixels = vec![10u8; 1000];
        pixels[..flipped].fill(10 + jump);
        SpaceValue::Dict(BTreeMap::from([
            (
                "cam".to_string(),
                SpaceValue::Box(
                    Tensor::from_vec(pixels, vec![10, 100, 1], DType::Uint8).expect("tensor"),
                ),
            ),
            (
                "pos".to_string(),
                SpaceValue::Box(
                    Tensor::from_vec(
                        [pos, 0.0].iter().flat_map(|v| v.to_le_bytes()).collect(),
                        vec![2],
                        DType::Float32,
                    )
                    .expect("tensor"),
                ),
            ),
        ]))
    }

    fn replay(observations: Vec<SpaceValue>, reward: f64) -> Replay {
        Replay {
            rewards: vec![reward; observations.len() - 1],
            dones: vec![false; observations.len() - 1],
            observations,
            reset_ms: 0.0,
        }
    }

    #[test]
    fn replay_divergence_ranks_state_and_image_leaves_separately() {
        let space = image_and_state_space();
        let images = vec![vec!["cam".to_string()]];
        let base = replay(vec![observation(0, 0, 0.5); 3], 1.0);

        let same = compare_replays(&base, &base, &space, &images).expect("comparable");
        assert_eq!(same.steps, 2);
        assert_eq!((same.state_first, same.image_first), (None, None));

        // Every pixel off by the jitter tolerance is still the same frame.
        let jitter = replay(
            vec![
                observation(0, 0, 0.5),
                observation(1000, 2, 0.5),
                observation(0, 0, 0.5),
            ],
            1.0,
        );
        let within = compare_replays(&base, &jitter, &space, &images).expect("comparable");
        assert_eq!((within.state_first, within.image_first), (None, None));
        assert_eq!(within.image_max, 2.0);

        let render = replay(
            vec![
                observation(0, 0, 0.5),
                observation(0, 0, 0.5),
                observation(1000, 30, 0.5),
            ],
            1.0,
        );
        let image_only = compare_replays(&base, &render, &space, &images).expect("comparable");
        assert_eq!(
            (image_only.state_first, image_only.image_first),
            (None, Some(2))
        );

        let physics = replay(
            vec![
                observation(0, 0, 0.5),
                observation(0, 0, 0.75),
                observation(0, 0, 0.5),
            ],
            1.0,
        );
        let state = compare_replays(&base, &physics, &space, &images).expect("comparable");
        assert_eq!(state.state_first, Some(1));
        assert!((state.state_max - 0.25).abs() < 1e-6);

        let reward = replay(vec![observation(0, 0, 0.5); 3], 2.0);
        let rewards = compare_replays(&base, &reward, &space, &images).expect("comparable");
        assert_eq!((rewards.state_first, rewards.state_max), (Some(1), 1.0));

        let short = replay(vec![observation(0, 0, 0.5); 2], 1.0);
        let ended = compare_replays(&base, &short, &space, &images).expect("comparable");
        assert_eq!((ended.steps, ended.state_first), (1, Some(1)));

        // Untagged: the image leaf is still found by shape.
        let mut found = Vec::new();
        image_leaves(&space, Vec::new(), &mut found);
        assert_eq!(found, images);
    }

    #[test]
    fn replay_divergence_tolerates_a_few_hard_pixel_flips() {
        let space = image_and_state_space();
        let images = vec![vec!["cam".to_string()]];
        let base = replay(vec![observation(0, 0, 0.5); 2], 0.0);

        let flips = replay(vec![observation(5, 149, 0.5), observation(0, 0, 0.5)], 0.0);
        let within = compare_replays(&base, &flips, &space, &images).expect("comparable");
        assert_eq!(within.image_first, None);
        assert_eq!(within.image_max, 149.0);
        assert!((within.image_over_fraction - 0.005).abs() < 1e-12);

        let broken = replay(vec![observation(20, 149, 0.5), observation(0, 0, 0.5)], 0.0);
        let diverged = compare_replays(&base, &broken, &space, &images).expect("comparable");
        assert_eq!(diverged.image_first, Some(0));
        assert!((diverged.image_over_fraction - 0.02).abs() < 1e-12);
    }

    #[test]
    fn action_range_is_the_widest_bounded_box_span() {
        let action = BoxSpaceBuilder::scalar(-1.0, 1.0, vec![7])
            .dtype(DType::Float32)
            .build()
            .expect("box");
        assert_eq!(action_range(&action), Some(2.0));
        let unbounded = BoxSpaceBuilder::unbounded(vec![3])
            .dtype(DType::Float32)
            .build()
            .expect("box");
        assert_eq!(action_range(&unbounded), None);
        let dict = DictSpaceBuilder::new()
            .insert(
                "arm",
                BoxSpaceBuilder::scalar(0.0, 5.0, vec![2])
                    .build()
                    .expect("box"),
            )
            .insert("gripper", action)
            .build()
            .expect("dict");
        assert_eq!(action_range(&dict), Some(5.0));
        assert_eq!(
            range_fraction_note(0.25, Some(2.0)),
            " (12.5% of the action range)"
        );
        assert_eq!(range_fraction_note(0.25, None), "");
    }

    #[test]
    fn replay_drift_splits_on_tolerance_and_shape() {
        assert!(matches!(replay_drift(None, 1.0), Drift::Shape));
        assert!(matches!(replay_drift(Some(1.0), 1.0), Drift::Within(_)));
        assert!(matches!(replay_drift(Some(1.5), 1.0), Drift::Beyond(_)));
    }

    #[test]
    fn stateful_verdict_is_informational_only_when_a_fresh_episode_clears_the_state() {
        let cleared = stateful_check(2.0, 1.0, Ok(Some(0.0)));
        assert_eq!(cleared.status, CheckStatus::Warn);
        assert_eq!(cleared.tags, ["stateful", "resets_on_episode_end"]);
        assert_eq!(cleared.metrics["resetMaxAbsDiff"], 0.0);

        let survives = stateful_check(2.0, 1.0, Ok(Some(2.0)));
        assert_eq!(survives.status, CheckStatus::Fail);
        assert_eq!(survives.tags, ["stateful"]);
        assert!(
            survives
                .detail
                .starts_with("state survives the episode end")
        );

        let shape = stateful_check(2.0, 1.0, Ok(None));
        assert_eq!(shape.status, CheckStatus::Fail);
        assert!(shape.tags.contains(&"nondeterministic_shape".to_string()));

        let failed = stateful_check(2.0, 1.0, Err("boom".to_string()));
        assert_eq!(failed.status, CheckStatus::Fail);
        assert!(failed.detail.ends_with("boom"));
    }
}
