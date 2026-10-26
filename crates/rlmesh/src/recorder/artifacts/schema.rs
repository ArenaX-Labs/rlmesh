//! The per-episode artifact contract: the tree layout and the JSON records.
//! Names and field spellings are the managed platform's
//! `docs/evaluation-artifacts.md`; a change here is a change to that layout.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const SCHEMA_EPISODE_V1: &str = "rlmesh.episode.v1";

pub const EPISODES_PREFIX: &str = "episodes/";
pub const EPISODE_JSON: &str = "episode.json";
pub const EPISODE_TRAJECTORY: &str = "trajectory.parquet";
pub const EPISODE_PERTURBATIONS: &str = "perturbations.jsonl";
pub const EPISODE_VIDEO_DIR: &str = "video";
pub const TELEMETRY_WINDOWS: &str = "telemetry/windows.jsonl";

/// `episodes/ep000001`: episode indices are 1-based.
pub fn episode_dir(episode_index: u32) -> String {
    format!("{EPISODES_PREFIX}ep{episode_index:06}")
}

pub fn episode_video(episode_index: u32, camera: &str, stream: RecordStream) -> String {
    format!(
        "{}/{EPISODE_VIDEO_DIR}/{}.{}.mp4",
        episode_dir(episode_index),
        sanitize_path_part(camera),
        stream.as_str()
    )
}

pub fn sanitize_path_part(value: &str) -> String {
    let sanitized = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    if sanitized.is_empty() {
        "unnamed".to_string()
    } else {
        sanitized
    }
}

pub fn is_safe_relative_path(path: &str) -> bool {
    if path.is_empty() || path.starts_with('/') || path.contains('\\') {
        return false;
    }
    !path
        .split('/')
        .any(|part| part.is_empty() || part == "." || part == "..")
}

/// The kind an attempt-relative path is listed under in `episode.json`
/// `files` (and the platform manifest).
pub fn infer_artifact_kind(path: &str) -> &'static str {
    if path == "attempt.json" {
        return "attempt";
    }
    if path == "manifest.json" {
        return "manifest";
    }
    if path == TELEMETRY_WINDOWS {
        return "telemetry";
    }
    if path.starts_with("logs/") {
        return "log";
    }
    if let Some((_, file)) = path
        .strip_prefix(EPISODES_PREFIX)
        .and_then(|rest| rest.split_once('/'))
    {
        if file == EPISODE_JSON {
            return "episode";
        }
        if file == EPISODE_TRAJECTORY {
            return "trajectory";
        }
        if file == EPISODE_PERTURBATIONS {
            return "perturbations";
        }
        if file.starts_with(&format!("{EPISODE_VIDEO_DIR}/")) {
            return "recording";
        }
    }
    "artifact"
}

/// Hardware and software provenance of the machine that ran an evaluation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AttemptStamp {
    pub runner_version: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub oss_crate_versions: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub managed_abi_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runner_image: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_image: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env_image: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_sha: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nvidia_driver: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuda: Option<String>,
    /// The ROCm release installed where the evaluation ran (`/opt/rocm`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rocm: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernel: Option<String>,
}

/// `episodes/epNNNNNN/episode.json`, written once the episode dir is closed.
/// `files` and `recordings` paths are attempt-relative.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EpisodeJson {
    pub schema: String,
    pub task_id: String,
    pub workload_id: String,
    pub workload_artifact_id: String,
    pub attempt: i32,
    pub episode_index: u32,
    pub env_index: i32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// The benchmark trial ordinal this episode walked; `None` when the run
    /// named no trial base.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trial_index: Option<u64>,
    pub reward: f64,
    pub steps: u64,
    pub timing: EpisodeTiming,
    pub verdict: Verdict,
    pub terminated: bool,
    pub truncated: bool,
    pub files: Vec<EpisodeFile>,
    pub recordings: Vec<EpisodeRecording>,
    /// Per-episode invariant checks that tripped.
    #[serde(default)]
    pub flags: Vec<EpisodeFlag>,
    /// FNV-1a 64 of the primary camera's reset frame, so different tasks
    /// that reset into the same scene stand out.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_frame_hash: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpisodeFlag {
    pub name: String,
    pub detail: String,
}

impl EpisodeFlag {
    pub fn new(name: &str, detail: impl Into<String>) -> Self {
        Self {
            name: name.to_string(),
            detail: detail.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EpisodeTiming {
    pub started_at: DateTime<Utc>,
    pub completed_at: DateTime<Utc>,
    pub duration_ms: u64,
}

/// `unknown` when the env reports no verdict; never inferred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    Success,
    Failure,
    Unknown,
}

impl Verdict {
    pub fn from_success(success: Option<bool>) -> Self {
        match success {
            Some(true) => Self::Success,
            Some(false) => Self::Failure,
            None => Self::Unknown,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EpisodeFile {
    pub path: String,
    pub kind: String,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EpisodeRecording {
    pub camera: String,
    pub stream: RecordStream,
    pub path: String,
    pub frames: u64,
    pub fps: u32,
    pub width: u32,
    pub height: u32,
}

/// `raw` is what the env produced, `perturbed` what the model received
/// (after any observation transform), `extra` more raw cameras.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RecordStream {
    Raw,
    Perturbed,
    Extra,
}

impl RecordStream {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::Perturbed => "perturbed",
            Self::Extra => "extra",
        }
    }
}

/// The platform's `record` block: which videos and trajectory columns an
/// episode writes, and for which episodes.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct RecordSpec {
    pub video: VideoRecordSpec,
    pub trajectory: TrajectoryRecordSpec,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct VideoRecordSpec {
    pub raw: Vec<String>,
    pub perturbed: Vec<String>,
    pub extra: Vec<String>,
    pub fps: u32,
    pub when: RecordWhen,
    /// Every Nth episode, 1-based inclusive: 1, 1+N, 1+2N, ….
    pub every: u32,
    /// Encode every Nth frame of a recorded episode. Playback fps stays as
    /// configured, so a stride-k video plays k× faster than wall time.
    pub stride: u32,
}

impl Default for VideoRecordSpec {
    fn default() -> Self {
        Self {
            raw: Vec::new(),
            perturbed: Vec::new(),
            extra: Vec::new(),
            fps: 30,
            when: RecordWhen::All,
            every: 1,
            stride: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct TrajectoryRecordSpec {
    pub obs: TrajectoryStream,
    pub obs_images: bool,
    pub action: TrajectoryStream,
    pub reward: bool,
    pub info: bool,
    pub step_timing: bool,
    pub when: RecordWhen,
    pub every: u32,
}

impl TrajectoryRecordSpec {
    /// A trajectory is written when any column beyond `step`/`t_ns` is on.
    pub fn enabled(&self) -> bool {
        self.obs != TrajectoryStream::None
            || self.action != TrajectoryStream::None
            || self.reward
            || self.info
            || self.step_timing
    }
}

impl Default for TrajectoryRecordSpec {
    fn default() -> Self {
        Self {
            obs: TrajectoryStream::None,
            obs_images: true,
            action: TrajectoryStream::None,
            reward: true,
            info: true,
            step_timing: true,
            when: RecordWhen::All,
            every: 1,
        }
    }
}

/// `failures` keeps anything not proven a success; `successes` keeps only
/// proven successes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RecordWhen {
    #[default]
    All,
    Failures,
    Successes,
}

impl RecordWhen {
    pub fn keeps(self, verdict: Verdict) -> bool {
        match self {
            Self::All => true,
            Self::Failures => verdict != Verdict::Success,
            Self::Successes => verdict == Verdict::Success,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrajectoryStream {
    Raw,
    Perturbed,
    Both,
    #[default]
    None,
}

impl TrajectoryStream {
    pub(crate) fn raw(self) -> bool {
        matches!(self, Self::Raw | Self::Both)
    }
    pub(crate) fn perturbed(self) -> bool {
        matches!(self, Self::Perturbed | Self::Both)
    }
}

/// Episode indices are 1-based, so episode 1 is always selected.
pub(crate) fn selected(index: u32, every: u32) -> bool {
    every <= 1 || index.saturating_sub(1).is_multiple_of(every)
}
