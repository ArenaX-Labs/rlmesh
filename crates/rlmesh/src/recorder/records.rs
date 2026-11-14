//! Portable result bundle records.
use crate::env::factory::{py_float, py_repr, py_str};
use crate::{EpisodeResult, Metadata, Value};
use std::fmt;

/// A media manifest row; bytes live at a bundle-relative path.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MediaRef {
    /// Camera/source label.
    pub camera: String,
    /// Media kind (normally `video`).
    pub kind: String,
    /// Bundle-relative asset path.
    pub path: String,
    /// Container format, such as `mp4`.
    pub format: String,
    /// Recorded frames, when known.
    pub frame_count: Option<i64>,
    /// Pixel width, when known.
    pub width: Option<i64>,
    /// Pixel height, when known.
    pub height: Option<i64>,
    /// Playback frames per second, when known.
    pub fps: Option<f64>,
}
impl MediaRef {
    /// JSON-native manifest row, omitting unknown optional fields.
    pub fn to_dict(&self) -> Metadata {
        let mut out: Metadata = [
            ("camera", Value::Text(self.camera.clone())),
            ("kind", self.kind.clone().into()),
            ("path", self.path.clone().into()),
            ("format", self.format.clone().into()),
        ]
        .into_iter()
        .collect();
        for (key, value) in [
            ("frameCount", self.frame_count),
            ("width", self.width),
            ("height", self.height),
        ] {
            if let Some(value) = value {
                out.insert(key, Value::Int(value));
            }
        }
        if let Some(fps) = self.fps {
            out.insert("fps", Value::Float(fps));
        }
        out
    }
}
impl fmt::Display for MediaRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let opt = |v: Option<i64>| v.map_or_else(|| "None".into(), |v| v.to_string());
        write!(
            f,
            "MediaRef(camera={}, kind={}, path={}, format={}, frame_count={}, width={}, height={}, fps={})",
            py_str(&self.camera),
            py_str(&self.kind),
            py_str(&self.path),
            py_str(&self.format),
            opt(self.frame_count),
            opt(self.width),
            opt(self.height),
            self.fps.map_or_else(|| "None".into(), py_float)
        )
    }
}
/// One episode's outcome and captured media, in SDK names and seconds.
#[derive(Clone, Debug, PartialEq)]
pub struct EpisodeRecord {
    /// Workload-stable episode index.
    pub index: i64,
    /// Reset seed, if provided.
    pub seed: Option<i64>,
    /// Env steps taken.
    pub steps: i64,
    /// Total reward.
    pub reward: f64,
    /// Env terminal state.
    pub terminated: bool,
    /// Cut short by env, caps or viewer.
    pub truncated: bool,
    /// Env-reported success.
    pub success: Option<bool>,
    /// Episode wall time in seconds.
    pub duration_s: f64,
    /// Include this episode in reward and success aggregates.
    pub included_in_metrics: bool,
    /// Media captured for this episode.
    pub media: Vec<MediaRef>,
}
impl Default for EpisodeRecord {
    fn default() -> Self {
        Self {
            index: 0,
            seed: None,
            steps: 0,
            reward: 0.0,
            terminated: false,
            truncated: false,
            success: None,
            duration_s: 0.0,
            included_in_metrics: true,
            media: Vec::new(),
        }
    }
}
impl EpisodeRecord {
    pub(super) fn from_result(
        result: &EpisodeResult,
        index: i64,
        included: bool,
        media: Vec<MediaRef>,
    ) -> Self {
        Self {
            index,
            seed: result.seed,
            steps: result.steps,
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            success: result.success,
            duration_s: result.duration_s,
            included_in_metrics: included,
            media,
        }
    }
    /// JSON-native episode row, with media only when present.
    pub fn to_dict(&self) -> Metadata {
        let mut out: Metadata = [
            ("index", self.index.into()),
            ("seed", self.seed.map_or(Value::None, Value::Int)),
            ("steps", self.steps.into()),
            ("reward", self.reward.into()),
            ("terminated", self.terminated.into()),
            ("truncated", self.truncated.into()),
            ("success", self.success.map_or(Value::None, Value::Bool)),
            ("durationS", self.duration_s.into()),
            ("includedInMetrics", self.included_in_metrics.into()),
        ]
        .into_iter()
        .collect();
        if !self.media.is_empty() {
            out.insert(
                "media",
                Value::List(
                    self.media
                        .iter()
                        .map(|m| Value::Dict(m.to_dict()))
                        .collect(),
                ),
            );
        }
        out
    }
}
fn boolean(value: bool) -> &'static str {
    if value { "True" } else { "False" }
}
impl fmt::Display for EpisodeRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let media = self
            .media
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        write!(
            f,
            "EpisodeRecord(index={}, seed={}, steps={}, reward={}, terminated={}, truncated={}, success={}, duration_s={}, included_in_metrics={}, media=({}{}))",
            self.index,
            self.seed.map_or_else(|| "None".into(), |v| v.to_string()),
            self.steps,
            py_float(self.reward),
            boolean(self.terminated),
            boolean(self.truncated),
            self.success.map_or("None", boolean),
            py_float(self.duration_s),
            boolean(self.included_in_metrics),
            media,
            if self.media.len() == 1 { "," } else { "" }
        )
    }
}
/// A snapshot of one model × env × task cell.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WorkloadRecord {
    /// Model name.
    pub model: String,
    /// Env name.
    pub env: String,
    /// Task name (defaults to the env name at recording).
    pub task: String,
    /// Caller-supplied metadata.
    pub config: Metadata,
    /// Episodes accumulated into this workload.
    pub episodes: Vec<EpisodeRecord>,
}
impl WorkloadRecord {
    /// Mean reward of metric-counted episodes, or zero.
    pub fn mean_reward(&self) -> f64 {
        let counted: Vec<_> = self
            .episodes
            .iter()
            .filter(|e| e.included_in_metrics)
            .collect();
        if counted.is_empty() {
            0.0
        } else {
            counted.iter().map(|e| e.reward).sum::<f64>() / counted.len() as f64
        }
    }
    /// Success fraction of metric-counted episodes, by
    /// [`RunResult::success_rate`](crate::RunResult::success_rate)'s rule.
    pub fn success_rate(&self) -> Option<f64> {
        crate::run::success_rate_of(
            self.episodes
                .iter()
                .filter(|e| e.included_in_metrics)
                .map(|e| e.success),
        )
    }
    /// Steps over all episodes, including discarded ones.
    pub fn total_steps(&self) -> i64 {
        self.episodes.iter().map(|e| e.steps).sum()
    }
    /// JSON-native workload with computed aggregates.
    pub fn to_dict(&self) -> Metadata {
        let metrics: Metadata = [
            ("meanReward", self.mean_reward().into()),
            (
                "successRate",
                self.success_rate().map_or(Value::None, Value::Float),
            ),
            ("totalSteps", self.total_steps().into()),
        ]
        .into_iter()
        .collect();
        [
            ("model", self.model.clone().into()),
            ("env", self.env.clone().into()),
            ("task", self.task.clone().into()),
            ("config", Value::Dict(self.config.clone())),
            ("numEpisodes", (self.episodes.len() as i64).into()),
            ("metrics", Value::Dict(metrics)),
            (
                "episodes",
                Value::List(
                    self.episodes
                        .iter()
                        .map(|e| Value::Dict(e.to_dict()))
                        .collect(),
                ),
            ),
        ]
        .into_iter()
        .collect()
    }
}
impl fmt::Display for WorkloadRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "WorkloadRecord(model={}, env={}, task={}, config={}, episodes=[{}])",
            py_str(&self.model),
            py_str(&self.env),
            py_str(&self.task),
            py_repr(&Value::Dict(self.config.clone())),
            self.episodes
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}
