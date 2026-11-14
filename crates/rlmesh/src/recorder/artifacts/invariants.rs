//! Cheap per-episode sanity checks over what the recorder already sees: a
//! frozen render context, a policy that never moves, a predict path that is
//! orders of magnitude off its baseline. Each is a bounded-memory tracker fed
//! per step and folded into `EpisodeJson.flags` when the episode closes.

use rlmesh_spaces::{DType, SpaceKind, SpaceSpec, SpaceValue, Tensor};
use serde_json::Value;

use super::is_image_space;
use super::schema::EpisodeFlag;

pub const FROZEN_OBSERVATION: &str = "frozen_observation";
pub const STATIC_POLICY: &str = "static_policy";
pub const SLOW_PREDICT: &str = "slow_predict";
pub const HORIZON_CAP: &str = "horizon_cap";

/// Frames inside the warm-up are never compared: the first steps after a
/// reset legitimately hold still.
const WARMUP_FRAMES: usize = 10;
const MIN_COMPARED_FRAMES: usize = 10;
const FRAME_SAMPLE_BYTES: usize = 4096;
/// Mean absolute pixel difference (u8 units) under which two frames count
/// as identical; 0.5/255 in normalized units.
const FROZEN_MEAN_ABS_DIFF: f64 = 0.5;
const FROZEN_RATIO: f64 = 0.9;
const MIN_STEPS: usize = 10;
const STATIC_ACTION_VARIANCE: f64 = 1e-6;
const MOTION_EPSILON: f64 = 1e-4;
const STILL_RATIO: f64 = 0.95;
const MIN_PREDICTS: usize = 3;
/// No probe baseline reaches the recorder, so the slow-predict bar is absolute.
pub const SLOW_PREDICT_MS: f64 = 500.0;
/// Env info leaves that report end-effector motion, matched on the last
/// segment of a dotted key (`dense_reward.eef_speed`).
const MOTION_KEYS: [&str; 5] = [
    "eef_speed",
    "ee_speed",
    "eef_velocity",
    "ee_velocity",
    "end_effector_speed",
];

#[derive(Default)]
pub(super) struct EpisodeInvariants {
    frames: FrameTracker,
    actions: ActionTracker,
    motion: MotionTracker,
    predict_ms: Vec<f32>,
}

impl EpisodeInvariants {
    /// The primary camera's raw (pre-perturbation) frame for one step.
    pub(super) fn frame(&mut self, bytes: &[u8]) {
        self.frames.observe(bytes);
    }

    /// The model's raw action for one step and the observe-to-action latency.
    pub(super) fn action(
        &mut self,
        value: &SpaceValue,
        space: &SpaceSpec,
        predict_ms: Option<f64>,
    ) {
        let mut values = Vec::new();
        numeric_leaves(value, space, &mut values);
        self.actions.observe(&values);
        if let Some(ms) = predict_ms {
            self.predict_ms.push(ms as f32);
        }
    }

    pub(super) fn step_info(&mut self, info: &Value) {
        self.motion.observe(info);
    }

    pub(super) fn finish(
        mut self,
        steps: u64,
        truncated: bool,
    ) -> (Vec<EpisodeFlag>, Option<String>) {
        let mut flags = Vec::new();
        flags.extend(self.frames.flag());
        let still = [self.actions.detail(), self.motion.detail()]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        if !still.is_empty() {
            flags.push(EpisodeFlag::new(STATIC_POLICY, still.join("; ")));
        }
        flags.extend(slow_predict(&mut self.predict_ms));
        if truncated {
            flags.push(EpisodeFlag::new(
                HORIZON_CAP,
                format!("truncated at step {steps}"),
            ));
        }
        let hash = self.frames.first_hash.map(|hash| format!("{hash:016x}"));
        (flags, hash)
    }
}

#[derive(Default)]
struct FrameTracker {
    first_hash: Option<u64>,
    prev: Option<Vec<u8>>,
    seen: usize,
    compared: usize,
    frozen: usize,
}

impl FrameTracker {
    fn observe(&mut self, bytes: &[u8]) {
        if self.first_hash.is_none() {
            self.first_hash = Some(fnv1a64(bytes));
        }
        let sample = sample(bytes);
        if self.seen >= WARMUP_FRAMES
            && let Some(prev) = &self.prev
        {
            self.compared += 1;
            if frames_identical(prev, &sample) {
                self.frozen += 1;
            }
        }
        self.prev = Some(sample);
        self.seen += 1;
    }

    fn flag(&self) -> Option<EpisodeFlag> {
        (self.compared >= MIN_COMPARED_FRAMES
            && self.frozen as f64 >= FROZEN_RATIO * self.compared as f64)
            .then(|| {
                EpisodeFlag::new(
                    FROZEN_OBSERVATION,
                    format!(
                        "primary image unchanged on {}/{} steps after the first {WARMUP_FRAMES}",
                        self.frozen, self.compared
                    ),
                )
            })
    }
}

/// Every `stride`-th byte, capped at `FRAME_SAMPLE_BYTES`, so a frame
/// comparison costs the same whatever the resolution.
fn sample(bytes: &[u8]) -> Vec<u8> {
    let stride = bytes.len().div_ceil(FRAME_SAMPLE_BYTES).max(1);
    bytes.iter().step_by(stride).copied().collect()
}

fn frames_identical(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() || a.is_empty() {
        return false;
    }
    if a == b {
        return true;
    }
    let total: u64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| u64::from(x.abs_diff(y)))
        .sum();
    (total as f64) / (a.len() as f64) < FROZEN_MEAN_ABS_DIFF
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// Welford running variance per action dimension.
#[derive(Default)]
struct ActionTracker {
    count: usize,
    mean: Vec<f64>,
    m2: Vec<f64>,
}

impl ActionTracker {
    fn observe(&mut self, values: &[f64]) {
        if values.is_empty() {
            return;
        }
        if self.count == 0 {
            self.mean = vec![0.0; values.len()];
            self.m2 = vec![0.0; values.len()];
        }
        if values.len() != self.mean.len() {
            return;
        }
        self.count += 1;
        let n = self.count as f64;
        for ((mean, m2), &x) in self.mean.iter_mut().zip(self.m2.iter_mut()).zip(values) {
            let delta = x - *mean;
            *mean += delta / n;
            *m2 += delta * (x - *mean);
        }
    }

    fn detail(&self) -> Option<String> {
        if self.count < MIN_STEPS {
            return None;
        }
        let max_variance = self
            .m2
            .iter()
            .map(|m2| m2 / self.count as f64)
            .fold(0.0, f64::max);
        (max_variance < STATIC_ACTION_VARIANCE).then(|| {
            format!(
                "action variance {max_variance:.1e} across {} dims over {} steps",
                self.mean.len(),
                self.count
            )
        })
    }
}

#[derive(Default)]
struct MotionTracker {
    key: Option<String>,
    steps: usize,
    still: usize,
}

impl MotionTracker {
    fn observe(&mut self, info: &Value) {
        let Some((key, speed)) = motion_value(info, "") else {
            return;
        };
        self.key.get_or_insert(key);
        self.steps += 1;
        if speed.abs() < MOTION_EPSILON {
            self.still += 1;
        }
    }

    fn detail(&self) -> Option<String> {
        let key = self.key.as_ref()?;
        (self.steps >= MIN_STEPS && self.still as f64 >= STILL_RATIO * self.steps as f64).then(
            || {
                format!(
                    "{key} below {MOTION_EPSILON} on {}/{} steps",
                    self.still, self.steps
                )
            },
        )
    }
}

/// The first motion leaf in the info tree as `(dotted key, magnitude)`; a
/// vector leaf reports its L2 norm.
fn motion_value(info: &Value, prefix: &str) -> Option<(String, f64)> {
    let object = info.as_object()?;
    for (key, value) in object {
        let path = if prefix.is_empty() {
            key.clone()
        } else {
            format!("{prefix}.{key}")
        };
        let leaf = key.rsplit(['.', '/']).next().unwrap_or(key);
        if MOTION_KEYS.contains(&leaf) {
            if let Some(magnitude) = magnitude(value) {
                return Some((path, magnitude));
            }
            continue;
        }
        if value.is_object()
            && let Some(found) = motion_value(value, &path)
        {
            return Some(found);
        }
    }
    None
}

fn magnitude(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::Array(items) => {
            let squares = items
                .iter()
                .map(|item| item.as_f64().map(|x| x * x))
                .collect::<Option<Vec<_>>>()?;
            (!squares.is_empty()).then(|| squares.iter().sum::<f64>().sqrt())
        }
        _ => None,
    }
}

fn slow_predict(predict_ms: &mut [f32]) -> Option<EpisodeFlag> {
    if predict_ms.len() < MIN_PREDICTS {
        return None;
    }
    predict_ms.sort_by(f32::total_cmp);
    let median = f64::from(predict_ms[predict_ms.len() / 2]);
    (median > SLOW_PREDICT_MS).then(|| {
        EpisodeFlag::new(
            SLOW_PREDICT,
            format!(
                "median predict {median:.0} ms > {SLOW_PREDICT_MS:.0} ms over {} steps",
                predict_ms.len()
            ),
        )
    })
}

/// Flattens every non-image numeric leaf in declaration order.
fn numeric_leaves(value: &SpaceValue, space: &SpaceSpec, out: &mut Vec<f64>) {
    match (value, space.spec.as_ref()) {
        (SpaceValue::Box(_), _) if is_image_space(space) => {}
        (SpaceValue::Box(tensor), _) => out.extend(tensor_values(tensor)),
        (SpaceValue::Discrete(value), _) => out.push(*value as f64),
        (SpaceValue::MultiBinary(values), _) => out.extend(values.iter().map(|&b| f64::from(b))),
        (SpaceValue::MultiDiscrete(values), _) => out.extend(values.iter().map(|&v| v as f64)),
        (SpaceValue::Dict(values), Some(SpaceKind::Dict(spec))) => {
            for (key, child_space) in spec.keys.iter().zip(spec.spaces.iter()) {
                if let Some(child) = values.get(key) {
                    numeric_leaves(child, child_space, out);
                }
            }
        }
        (SpaceValue::Tuple(values), Some(SpaceKind::Tuple(spec))) => {
            for (child, child_space) in values.iter().zip(spec.spaces.iter()) {
                numeric_leaves(child, child_space, out);
            }
        }
        _ => {}
    }
}

fn tensor_values(tensor: &Tensor) -> Vec<f64> {
    fn chunks<const N: usize>(data: &[u8], decode: impl Fn([u8; N]) -> f64) -> Vec<f64> {
        data.as_chunks::<N>()
            .0
            .iter()
            .map(|chunk| decode(*chunk))
            .collect()
    }
    let data = tensor.to_contiguous_bytes();
    match tensor.dtype() {
        DType::Float32 => chunks::<4>(&data, |b| f64::from(f32::from_le_bytes(b))),
        DType::Float64 => chunks::<8>(&data, f64::from_le_bytes),
        DType::Float16 => chunks::<2>(&data, |b| half::f16::from_le_bytes(b).to_f64()),
        DType::Int8 => data.iter().map(|&b| f64::from(b as i8)).collect(),
        DType::Int16 => chunks::<2>(&data, |b| f64::from(i16::from_le_bytes(b))),
        DType::Int32 => chunks::<4>(&data, |b| f64::from(i32::from_le_bytes(b))),
        DType::Int64 => chunks::<8>(&data, |b| i64::from_le_bytes(b) as f64),
        DType::Uint8 | DType::Bool => data.iter().map(|&b| f64::from(b)).collect(),
        DType::Uint16 => chunks::<2>(&data, |b| f64::from(u16::from_le_bytes(b))),
        DType::Uint32 => chunks::<4>(&data, |b| f64::from(u32::from_le_bytes(b))),
        DType::Uint64 => chunks::<8>(&data, |b| u64::from_le_bytes(b) as f64),
        DType::Unspecified => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlmesh_spaces::spaces::BoxSpaceBuilder;
    use serde_json::json;

    fn action_space(dims: usize) -> SpaceSpec {
        BoxSpaceBuilder::scalar(-1.0, 1.0, vec![dims as i64])
            .dtype(DType::Float32)
            .build()
            .unwrap()
    }

    fn action(values: &[f32]) -> SpaceValue {
        let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        SpaceValue::Box(Tensor::from_vec(bytes, vec![values.len() as i64], DType::Float32).unwrap())
    }

    fn frame(fill: impl Fn(usize) -> u8) -> Vec<u8> {
        (0..64 * 64 * 3).map(fill).collect()
    }

    fn names(flags: &[EpisodeFlag]) -> Vec<&str> {
        flags.iter().map(|flag| flag.name.as_str()).collect()
    }

    #[test]
    fn frozen_frames_after_warmup_are_flagged_and_moving_frames_are_not() {
        let mut frozen = EpisodeInvariants::default();
        for step in 0..40 {
            // Physics keeps moving for the first frames, then the render dies.
            frozen.frame(&frame(|i| (i + step.min(5)) as u8));
        }
        let (flags, hash) = frozen.finish(40, false);
        assert_eq!(names(&flags), [FROZEN_OBSERVATION]);
        assert!(
            flags[0]
                .detail
                .starts_with("primary image unchanged on 30/30")
        );
        assert_eq!(hash.unwrap().len(), 16);

        let mut moving = EpisodeInvariants::default();
        for step in 0..40 {
            moving.frame(&frame(|i| (i + step) as u8));
        }
        assert!(moving.finish(40, false).0.is_empty());
    }

    #[test]
    fn sub_half_pixel_noise_still_counts_as_frozen() {
        let mut invariants = EpisodeInvariants::default();
        for step in 0..40 {
            // One pixel toggles per frame: mean abs diff far below 0.5.
            invariants.frame(&frame(|i| if i == 0 { step as u8 } else { 7 }));
        }
        assert_eq!(names(&invariants.finish(40, false).0), [FROZEN_OBSERVATION]);
    }

    #[test]
    fn short_episodes_never_trip_the_frozen_check() {
        let mut invariants = EpisodeInvariants::default();
        for _ in 0..19 {
            invariants.frame(&frame(|_| 3));
        }
        assert!(invariants.finish(19, false).0.is_empty());
    }

    #[test]
    fn first_frame_hash_is_stable_and_content_addressed() {
        let mut a = EpisodeInvariants::default();
        a.frame(&frame(|i| i as u8));
        let mut b = EpisodeInvariants::default();
        b.frame(&frame(|i| i as u8));
        b.frame(&frame(|_| 0));
        let mut c = EpisodeInvariants::default();
        c.frame(&frame(|i| (i + 1) as u8));
        let hash = |inv: EpisodeInvariants| inv.finish(1, false).1.unwrap();
        let first = hash(a);
        assert_eq!(first, hash(b));
        assert_ne!(first, hash(c));
        let mut none = EpisodeInvariants::default();
        none.action(&action(&[0.0]), &action_space(1), None);
        assert!(none.finish(1, false).1.is_none());
    }

    #[test]
    fn static_actions_are_flagged_and_varying_actions_are_not() {
        let space = action_space(7);
        let mut fixed = EpisodeInvariants::default();
        for _ in 0..30 {
            fixed.action(
                &action(&[0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0]),
                &space,
                Some(7.0),
            );
        }
        let (flags, _) = fixed.finish(30, false);
        assert_eq!(names(&flags), [STATIC_POLICY]);
        assert!(flags[0].detail.contains("across 7 dims over 30 steps"));

        let mut moving = EpisodeInvariants::default();
        for step in 0..30 {
            let mut values = [0.0f32; 7];
            values[step % 7] = 0.5;
            moving.action(&action(&values), &space, Some(7.0));
        }
        assert!(moving.finish(30, false).0.is_empty());
    }

    #[test]
    fn zero_end_effector_speed_in_info_flags_a_static_policy() {
        let space = action_space(2);
        let mut still = EpisodeInvariants::default();
        for step in 0..40 {
            still.action(&action(&[step as f32, 1.0]), &space, None);
            let speed = if step == 3 { 0.2 } else { 1e-6 };
            still.step_info(&json!({"dense_reward": {"eef_speed": speed, "reach": 0.5}}));
        }
        let (flags, _) = still.finish(40, false);
        assert_eq!(names(&flags), [STATIC_POLICY]);
        assert_eq!(
            flags[0].detail,
            "dense_reward.eef_speed below 0.0001 on 39/40 steps"
        );

        let mut moving = EpisodeInvariants::default();
        for step in 0..40 {
            moving.step_info(&json!({"eef_velocity": [0.0, 0.01 * (step % 2) as f64, 0.0]}));
        }
        assert!(moving.finish(40, false).0.is_empty());
    }

    #[test]
    fn slow_median_predict_is_flagged() {
        let space = action_space(1);
        let mut slow = EpisodeInvariants::default();
        for step in 0..9 {
            let ms = if step == 0 { 9000.0 } else { 964.0 };
            slow.action(&action(&[step as f32]), &space, Some(ms));
        }
        let (flags, _) = slow.finish(9, false);
        assert_eq!(names(&flags), [SLOW_PREDICT]);
        assert_eq!(
            flags[0].detail,
            "median predict 964 ms > 500 ms over 9 steps"
        );

        let mut warm = EpisodeInvariants::default();
        for step in 0..9 {
            let ms = if step == 0 { 9000.0 } else { 7.0 };
            warm.action(&action(&[step as f32]), &space, Some(ms));
        }
        assert!(warm.finish(9, false).0.is_empty());
    }

    #[test]
    fn truncation_is_reported_with_the_step_it_hit() {
        let (flags, _) = EpisodeInvariants::default().finish(530, true);
        assert_eq!(
            flags,
            [EpisodeFlag::new(HORIZON_CAP, "truncated at step 530")]
        );
    }
}
