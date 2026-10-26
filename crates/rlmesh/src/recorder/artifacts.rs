//! The per-episode artifact tree the managed platform's evaluations write
//! (`docs/evaluation-artifacts.md` there): `episodes/epNNNNNN/` with
//! `episode.json`, `trajectory.parquet` and `video/{camera}.{stream}.mp4`,
//! plus `telemetry/windows.jsonl`.
//!
//! [`EpisodeArtifacts`] is fed the runtime driver's events, so raw and
//! transformed streams come from `raw_observation` / `raw_action` with no
//! knowledge of the transform. [`Recorder::capture`](crate::Recorder::capture)
//! with a `record` spec drives it on a run; a runner with its own
//! [`RuntimeHooks`](crate::binding::RuntimeHooks) drives it directly.

// Moved from rlmesh-middleman, whose artifact field names are this contract.
#![allow(missing_docs)]

mod invariants;
pub mod schema;
pub mod stamp;
#[cfg(test)]
mod tests;
mod trajectory;
mod video;

use std::collections::{HashMap, VecDeque};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use rlmesh_grpc::wire::{
    decode_batched_partial_values, leaves_value, meta_map_from_proto, space_spec_from_proto,
};
use rlmesh_proto::spaces::v1::{MetaMap, SpaceSpec as ProtoSpaceSpec, meta_value};
use rlmesh_spaces::{DType, MetaValue, SpaceKind, SpaceSpec, SpaceValue};
use serde::Serialize;
use serde_json::{Value as Json, json};

use crate::recorder::video::Encoding;
use crate::run::runtime::telemetry::{Horizon, Kind};
use crate::run::runtime::{
    ActionReceivedEvent, EpisodeCompletedEvent, EpisodeStartedEvent, ObservationEmittedEvent,
    SessionStartedEvent, StepCompletedEvent, TelemetrySnapshotEvent,
};
use crate::{Error, Result};
use invariants::EpisodeInvariants;
use schema::{
    EPISODE_JSON, EPISODE_TRAJECTORY, EPISODE_VIDEO_DIR, EpisodeFile, EpisodeJson, EpisodeTiming,
    RecordSpec, SCHEMA_EPISODE_V1, TELEMETRY_WINDOWS, TrajectoryStream, Verdict, episode_dir,
    infer_artifact_kind, is_safe_relative_path, selected,
};
pub use stamp::collect as collect_stamp;
use trajectory::{StepRow, StepValue, TrajectoryWriter, step_value};
use video::{EpisodeVideo, Video};

/// The shared handle a run's hooks hand the native loop to feed (R-25).
pub type ArtifactTap = Arc<Mutex<EpisodeArtifacts>>;

/// What an [`EpisodeArtifacts`] writes and the identity its `episode.json`
/// records carry.
#[derive(Clone, Debug)]
pub struct ArtifactOptions {
    pub record: RecordSpec,
    pub task_id: String,
    pub workload_id: String,
    pub workload_artifact_id: String,
    pub attempt: i32,
    /// How videos are encoded (AV1 quality 60 by default).
    pub encoding: Encoding,
    /// The runtime's `telemetry_window`: the seconds a window row covers.
    pub telemetry_window: Duration,
}

impl Default for ArtifactOptions {
    fn default() -> Self {
        Self {
            record: RecordSpec::default(),
            task_id: String::new(),
            workload_id: String::new(),
            workload_artifact_id: String::new(),
            attempt: 1,
            encoding: Encoding::default(),
            telemetry_window: crate::run::runtime::RuntimeLimits::default().telemetry_window,
        }
    }
}

/// A `runtime.phase` timing mark: `warmup` runs from session start to the
/// first completed step.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PhaseMark {
    pub phase: String,
    pub started_at: DateTime<Utc>,
    pub ended_at: DateTime<Utc>,
    pub duration_ms: i64,
    pub env_id: String,
}

struct Episode {
    index: u32,
    env_index: i32,
    seed: Option<i64>,
    /// `seed` reported by the env in its reset infos; wins over the runtime's.
    info_seed: Option<i64>,
    trial_index: Option<u64>,
    /// The first observation is the reset one regardless of `is_reset`: under
    /// NEXT_STEP autoreset the reset infos ride the first post-roll observation.
    observed: bool,
    started_at: DateTime<Utc>,
    dir: PathBuf,
    rel: String,
    /// Present only for a selected episode of a trajectory-recording spec.
    trajectory: Option<TrajectoryWriter>,
    /// Present only for a selected episode of a video-recording spec.
    video: Option<EpisodeVideo>,
    open: Option<StepRow>,
    observed_at: Option<Instant>,
    action_at: Option<Instant>,
    invariants: EpisodeInvariants,
}

/// Writes one workload attempt's episode tree under `root`. Calls are
/// blocking (encoder close, parquet, `episode.json`): drive it from a
/// blocking context.
pub struct EpisodeArtifacts {
    root: PathBuf,
    options: ArtifactOptions,
    video: Video,
    spaces: HashMap<&'static str, (Arc<ProtoSpaceSpec>, Arc<SpaceSpec>)>,
    episodes: HashMap<String, Episode>,
    /// `env_index` -> open episode record id: step events carry only lane
    /// 0's record id, so lanes are resolved from their started events.
    lanes: HashMap<i32, String>,
    offset: u32,
    started: Instant,
    warmup: Option<(DateTime<Utc>, String)>,
    phases: Vec<PhaseMark>,
    finished: VecDeque<EpisodeJson>,
    windows: Option<std::fs::File>,
}

impl EpisodeArtifacts {
    pub fn new(root: impl Into<PathBuf>, options: ArtifactOptions) -> Self {
        Self {
            root: root.into(),
            options,
            video: Video::default(),
            spaces: HashMap::new(),
            episodes: HashMap::new(),
            lanes: HashMap::new(),
            offset: 0,
            started: Instant::now(),
            warmup: None,
            phases: Vec::new(),
            finished: VecDeque::new(),
            windows: None,
        }
    }

    /// The attempt root every recorded path is relative to.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Number later sessions' episodes after `offset`, so several runs into
    /// one tree never reuse an episode directory.
    pub fn set_episode_offset(&mut self, offset: u32) {
        self.offset = offset;
    }

    /// Closed episodes not yet taken, oldest first.
    pub fn take_finished(&mut self) -> Option<EpisodeJson> {
        self.finished.pop_front()
    }

    pub fn phases(&self) -> &[PhaseMark] {
        &self.phases
    }

    pub fn session_started(&mut self, event: &SessionStartedEvent) -> Result<()> {
        self.started = Instant::now();
        self.warmup = Some((Utc::now(), event.route.env_id.clone()));
        self.video = Video::resolve(
            &self.options.record.video,
            self.options.encoding.clone(),
            &event.env_contract,
        )?;
        Ok(())
    }

    pub fn episode_started(&mut self, event: &EpisodeStartedEvent) -> Result<()> {
        let index = self.offset + u32::try_from(event.episode_index).unwrap_or(0);
        let rel = episode_dir(index);
        let dir = self.root.join(&rel);
        std::fs::create_dir_all(&dir).map_err(os)?;
        let record = &self.options.record;
        let trajectory = (record.trajectory.enabled() && selected(index, record.trajectory.every))
            .then(|| {
                TrajectoryWriter::new(dir.join(EPISODE_TRAJECTORY), record.trajectory.clone())
            });
        let video = (self.video.records() && selected(index, record.video.every))
            .then(EpisodeVideo::default);
        self.lanes
            .insert(event.env_index, event.episode_record_id.clone());
        self.episodes.insert(
            event.episode_record_id.clone(),
            Episode {
                index,
                env_index: event.env_index,
                seed: event.seed,
                info_seed: None,
                trial_index: event.trial_index,
                observed: false,
                started_at: Utc::now(),
                dir,
                rel,
                trajectory,
                video,
                open: None,
                observed_at: None,
                action_at: None,
                invariants: EpisodeInvariants::default(),
            },
        );
        Ok(())
    }

    fn space(&mut self, key: &'static str, proto: &Arc<ProtoSpaceSpec>) -> Result<Arc<SpaceSpec>> {
        if let Some((raw, parsed)) = self.spaces.get(key)
            && Arc::ptr_eq(raw, proto)
        {
            return Ok(parsed.clone());
        }
        let parsed = Arc::new(space_spec_from_proto((**proto).clone()).map_err(os)?);
        self.spaces.insert(key, (proto.clone(), parsed.clone()));
        Ok(parsed)
    }

    pub fn observation(&mut self, event: &ObservationEmittedEvent) -> Result<()> {
        let Some(observation) = event.observation.as_ref() else {
            return Ok(());
        };
        let lanes = usize::try_from(event.num_envs).unwrap_or(1).max(1);
        let trajectory = self.options.record.trajectory.clone();
        let reset_seed = info_seed(event.infos.as_ref());
        let info = trajectory
            .info
            .then(|| {
                event
                    .infos
                    .as_ref()
                    .map(|i| meta_map_to_json(i).to_string())
            })
            .flatten();
        let space = self.space("obs", &event.observation_space)?;
        let raw_leaves = event.raw_observation.as_ref().unwrap_or(observation);
        let mut perturbed: Option<Vec<SpaceValue>> = None;
        let mut raw: Option<Vec<SpaceValue>> = None;
        let decode = |leaves: &Vec<bytes::Bytes>| -> Result<Vec<SpaceValue>> {
            decode_batched_partial_values(Some(&leaves_value(leaves.clone())), &space, lanes)
                .map_err(os)
        };
        let now = Instant::now();
        for lane in 0..lanes {
            let record_id =
                lane_record_id(&event.episode_record_ids, lane, &event.episode_record_id);
            let Some(episode) = self.episodes.get_mut(record_id) else {
                continue;
            };
            let reset = !std::mem::replace(&mut episode.observed, true);
            if reset && reset_seed.is_some() {
                episode.info_seed = reset_seed;
            }
            episode.observed_at = Some(now);
            if let Some(primary) = &self.video.primary
                && let Some(frame) = primary.frame(raw_leaves, lanes, lane)
            {
                episode.invariants.frame(frame);
            }
            let row = || StepRow {
                t_ns: Utc::now().timestamp_nanos_opt().unwrap_or_default(),
                info: reset.then(|| info.clone()).flatten(),
                ..StepRow::default()
            };
            let wants_obs =
                episode.trajectory.is_some() && trajectory.obs != TrajectoryStream::None;
            if episode.trajectory.is_some() && !wants_obs {
                episode.open = Some(row());
            }
            // Selection first: a sampled-out episode costs no decode or copy.
            if !wants_obs && episode.video.is_none() {
                continue;
            }
            if perturbed.is_none() {
                perturbed = Some(decode(observation)?);
                if event.raw_observation.is_some() {
                    raw = Some(decode(raw_leaves)?);
                }
            }
            let perturbed_values = perturbed.as_deref().expect("decoded above");
            let Some(perturbed_value) = perturbed_values.get(lane) else {
                continue;
            };
            let raw_value = raw
                .as_deref()
                .and_then(|values| values.get(lane))
                .unwrap_or(perturbed_value);
            if wants_obs {
                let value = |v: &SpaceValue| step_value(v, &space, trajectory.obs_images);
                let obs_perturbed = trajectory.obs.perturbed().then(|| value(perturbed_value));
                let obs_raw = match (&obs_perturbed, raw.is_none()) {
                    // No transform: the raw stream shares the perturbed pixels.
                    (Some(same), true) if trajectory.obs.raw() => Some(same.clone()),
                    _ => trajectory.obs.raw().then(|| value(raw_value)),
                };
                episode.open = Some(StepRow {
                    obs_raw,
                    obs_perturbed,
                    ..row()
                });
            }
            if let Some(video) = episode.video.as_mut() {
                self.video.observe(
                    video,
                    &episode.dir,
                    &space,
                    (raw_value, perturbed_value),
                    raw.is_none(),
                )?;
            }
        }
        Ok(())
    }

    /// The raw lanes feed the invariants whatever the spec says; their JSON
    /// is only built when the trajectory records actions.
    pub fn action(&mut self, event: &ActionReceivedEvent) -> Result<()> {
        let Some(action) = event.action.as_ref() else {
            return Ok(());
        };
        let trajectory = &self.options.record.trajectory;
        let stream = if trajectory.enabled() {
            trajectory.action
        } else {
            TrajectoryStream::None
        };
        let lanes = event.episode_record_ids.len().max(1);
        let space = self.space("act", &event.action_space)?;
        let decode = |leaves: &Vec<bytes::Bytes>| -> Result<Vec<SpaceValue>> {
            decode_batched_partial_values(Some(&leaves_value(leaves.clone())), &space, lanes)
                .map_err(os)
        };
        let raw_values = decode(event.raw_action.as_ref().unwrap_or(action))?;
        let lane_values = |values: &[SpaceValue]| -> Vec<StepValue> {
            values.iter().map(|v| step_value(v, &space, true)).collect()
        };
        let (mut raw, mut perturbed) = if stream == TrajectoryStream::None {
            (Vec::new(), Vec::new())
        } else {
            let raw = lane_values(&raw_values);
            let perturbed = match &event.raw_action {
                None => raw.clone(),
                Some(_) => lane_values(&decode(action)?),
            };
            (raw, perturbed)
        };
        let now = Instant::now();
        for lane in 0..lanes {
            let record_id =
                lane_record_id(&event.episode_record_ids, lane, &event.episode_record_id);
            let Some(episode) = self.episodes.get_mut(record_id) else {
                continue;
            };
            let predict_ms = episode
                .observed_at
                .map(|at| now.duration_since(at).as_secs_f64() * 1e3);
            if let Some(value) = raw_values.get(lane) {
                episode.invariants.action(value, &space, predict_ms);
            }
            let Some(open) = episode.open.as_mut() else {
                continue;
            };
            episode.action_at = Some(now);
            open.predict_ms = predict_ms;
            open.action_raw = take_lane(&mut raw, lane, stream.raw());
            open.action_perturbed = take_lane(&mut perturbed, lane, stream.perturbed());
        }
        Ok(())
    }

    /// Closes every lane's open step: lane `i` takes `rewards[i]`; infos are
    /// shared across lanes. A trajectory over its memory budget is written
    /// as a row group here.
    pub fn step(&mut self, event: &StepCompletedEvent) -> Result<()> {
        if let Some((started_at, env_id)) = self.warmup.take() {
            let ended_at = Utc::now();
            self.phases.push(PhaseMark {
                phase: "warmup".into(),
                started_at,
                ended_at,
                duration_ms: (ended_at - started_at).num_milliseconds().max(0),
                env_id,
            });
        }
        let trajectory = &self.options.record.trajectory;
        let info_value = event.infos.as_ref().map(meta_map_to_json);
        let info = trajectory
            .info
            .then(|| info_value.as_ref().map(Json::to_string))
            .flatten();
        let lanes = event.rewards.len().max(1);
        for lane in 0..lanes {
            let env_index = if lanes == 1 {
                event.env_index
            } else {
                lane as i32
            };
            let Some(episode) = self
                .lanes
                .get(&env_index)
                .and_then(|id| self.episodes.get_mut(id))
            else {
                continue;
            };
            if let Some(info) = &info_value {
                episode.invariants.step_info(info);
            }
            let Some(mut row) = episode.open.take() else {
                continue;
            };
            row.step = event.step;
            row.reward = trajectory
                .reward
                .then(|| event.rewards.get(lane).copied())
                .flatten();
            if info.is_some() {
                row.info = info.clone();
            }
            if trajectory.step_timing {
                row.step_ms = episode.action_at.map(|at| at.elapsed().as_secs_f64() * 1e3);
            } else {
                row.predict_ms = None;
            }
            let Some(writer) = episode.trajectory.as_mut() else {
                continue;
            };
            writer.push(row);
            if writer.over_budget()
                && let Err(err) = writer.flush()
            {
                // Abandoned rather than written short.
                episode.trajectory = None;
                warn(format!(
                    "recorder: dropping the trajectory of {}: {err}",
                    episode.rel
                ));
            }
        }
        Ok(())
    }

    /// Close the episode's encoders and trajectory, apply the `when`
    /// filters and write `episode.json`. Every tail failure is warned about
    /// rather than returned, so the record lists whatever succeeded.
    pub fn episode_completed(&mut self, event: &EpisodeCompletedEvent) -> Result<()> {
        let Some(episode) = self.episodes.remove(&event.episode_record_id) else {
            return Ok(());
        };
        if self.lanes.get(&episode.env_index) == Some(&event.episode_record_id) {
            self.lanes.remove(&episode.env_index);
        }
        let verdict = Verdict::from_success(event.success);
        let record = &self.options.record;
        let mut recordings = episode
            .video
            .map(|video| self.video.finish(video, &self.root))
            .unwrap_or_default();
        if !record.video.when.keeps(verdict) {
            let video_dir = episode.dir.join(EPISODE_VIDEO_DIR);
            if video_dir.exists()
                && let Err(err) = std::fs::remove_dir_all(&video_dir)
            {
                warn(format!(
                    "recorder: failed to remove the unwanted videos of {}: {err}",
                    episode.rel
                ));
            }
            recordings.clear();
        }
        if let Some(writer) = episode.trajectory
            && let Err(err) = writer.finish(record.trajectory.when.keeps(verdict))
        {
            warn(format!(
                "recorder: failed to write the trajectory of {}: {err}",
                episode.rel
            ));
        }
        let files = list_files(&episode.dir, &self.root).unwrap_or_else(|err| {
            warn(format!(
                "recorder: failed to list the files of {}: {err}",
                episode.rel
            ));
            Vec::new()
        });
        let steps = u64::try_from(event.step_count).unwrap_or(0);
        let (flags, first_frame_hash) = episode.invariants.finish(steps, event.truncated);
        let record = EpisodeJson {
            schema: SCHEMA_EPISODE_V1.to_string(),
            task_id: self.options.task_id.clone(),
            workload_id: self.options.workload_id.clone(),
            workload_artifact_id: self.options.workload_artifact_id.clone(),
            attempt: self.options.attempt,
            episode_index: episode.index,
            env_index: episode.env_index,
            seed: episode
                .info_seed
                .or_else(|| info_seed(event.final_info.as_ref()))
                .or(event.seed)
                .or(episode.seed),
            trial_index: event.trial_index.or(episode.trial_index),
            reward: event.cumulative_reward,
            steps,
            timing: EpisodeTiming {
                started_at: episode.started_at,
                completed_at: Utc::now(),
                duration_ms: u64::try_from(event.duration_ms).unwrap_or(0),
            },
            verdict,
            terminated: event.terminated,
            truncated: event.truncated,
            files,
            recordings,
            flags,
            first_frame_hash,
        };
        write_json(&episode.dir.join(EPISODE_JSON), &record)?;
        self.finished.push_back(record);
        Ok(())
    }

    /// One `telemetry/windows.jsonl` row per snapshot: the live windows and
    /// the final cumulative session.
    pub fn telemetry(&mut self, event: &TelemetrySnapshotEvent) -> Result<()> {
        let snapshot = &event.snapshot;
        // Session rows carry cumulative counts; pairing them with the
        // window's seconds would inflate derived rates.
        let (horizon, window_seconds) = match snapshot.horizon {
            Horizon::Window => ("window", self.options.telemetry_window.as_secs_f64()),
            Horizon::Session => ("session", self.started.elapsed().as_secs_f64().max(1.0)),
        };
        let rows = snapshot
            .rows
            .iter()
            .map(|row| {
                json!({
                    "source": { "op": row.source.op, "component": row.source.component },
                    "metric": { "name": row.metric.name, "kind": match row.metric.kind {
                        Kind::Duration => "duration",
                        Kind::Bytes => "bytes",
                        Kind::Count => "count",
                    } },
                    "count": row.count,
                    "avg": row.avg,
                    "p50": row.p50,
                    "p95": row.p95,
                    "p99": row.p99,
                })
            })
            .collect::<Vec<_>>();
        let mut line = serde_json::to_vec(&json!({
            "horizon": horizon,
            "sessionId": event.session_id,
            "envId": event.route.env_id,
            "taskId": self.options.task_id,
            "windowSeconds": window_seconds,
            "emittedAtMs": Utc::now().timestamp_millis(),
            "rows": rows,
        }))
        .map_err(os)?;
        line.push(b'\n');
        if self.windows.is_none() {
            let path = self.root.join(TELEMETRY_WINDOWS);
            std::fs::create_dir_all(path.parent().expect("telemetry has a parent")).map_err(os)?;
            self.windows = Some(
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .map_err(os)?,
            );
        }
        let file = self.windows.as_mut().expect("opened above");
        file.write_all(&line).map_err(os)?;
        file.flush().map_err(os)
    }

    /// Drop every unfinished episode of a failed session: its rows, its
    /// encoders and its directory, so nothing of it is listed as completed.
    // TODO: discard is the whole partial-artifact policy; finish the partial
    // videos as a flagged artifact if anyone needs to watch a failed episode.
    pub fn abandon(&mut self) {
        self.lanes.clear();
        for (_, episode) in self.episodes.drain() {
            drop(episode.video);
            drop(episode.trajectory);
            let _ = std::fs::remove_dir_all(&episode.dir);
        }
    }
}

pub(crate) fn os(err: impl std::fmt::Display) -> Error {
    Error::Os(err.to_string())
}

pub(crate) fn warn(message: impl Into<String>) {
    let _ = crate::warnings::warn(crate::WarningCategory::UserWarning, message);
}

/// A uint8 Box shaped like a grayscale, HWC or CHW image.
pub(crate) fn is_image_space(space: &SpaceSpec) -> bool {
    matches!(space.spec.as_ref(), Some(SpaceKind::Box(_)))
        && space.dtype == DType::Uint8
        && matches!(
            space.shape.as_slice(),
            [_, _] | [_, _, 1 | 3 | 4] | [1 | 3 | 4, _, _]
        )
}

fn lane_record_id<'a>(ids: &'a [String], index: usize, fallback: &'a str) -> &'a str {
    ids.get(index)
        .filter(|id| !id.is_empty())
        .map_or(fallback, String::as_str)
}

/// Each lane is claimed once per event, so the value moves out rather than
/// copying the lane's pixels.
fn take_lane(values: &mut [StepValue], index: usize, keep: bool) -> Option<StepValue> {
    keep.then(|| values.get_mut(index).map(std::mem::take))
        .flatten()
}

fn info_seed(info: Option<&MetaMap>) -> Option<i64> {
    match info?.entries.get("seed")?.kind {
        Some(meta_value::Kind::Integer(seed)) => Some(seed),
        _ => None,
    }
}

fn meta_map_to_json(map: &MetaMap) -> Json {
    fn value(v: &MetaValue) -> Json {
        match v {
            // Bytes and tensors never appear in step infos the JSON column
            // can carry; represented as null, as on 0.1.
            MetaValue::Null
            | MetaValue::Bytes(_)
            | MetaValue::Tensor(_)
            | MetaValue::Array { .. } => Json::Null,
            MetaValue::Bool(b) => (*b).into(),
            MetaValue::Int(i) => (*i).into(),
            MetaValue::Float(f) => {
                serde_json::Number::from_f64(*f).map_or(Json::Null, Json::Number)
            }
            MetaValue::String(s) => s.clone().into(),
            MetaValue::List(items) => items.iter().map(value).collect(),
            MetaValue::Map(map) => {
                Json::Object(map.iter().map(|(key, v)| (key.clone(), value(v))).collect())
            }
        }
    }
    Json::Object(
        meta_map_from_proto(map.clone())
            .iter()
            .map(|(key, v)| (key.clone(), value(v)))
            .collect(),
    )
}

/// Whole-file writes go through a `.rlmesh-tmp` sibling, so a reader never
/// sees a partial document.
fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("out");
    let tmp = path.with_file_name(format!("{name}.rlmesh-tmp"));
    std::fs::write(&tmp, serde_json::to_vec_pretty(value).map_err(os)?).map_err(os)?;
    std::fs::rename(&tmp, path).map_err(os)
}

fn list_files(dir: &Path, root: &Path) -> std::io::Result<Vec<EpisodeFile>> {
    let mut files = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(current) = pending.pop() {
        for entry in std::fs::read_dir(&current)? {
            let path = entry?.path();
            if path.is_dir() {
                pending.push(path);
                continue;
            }
            let rel = video::relative(&path, root);
            if !is_safe_relative_path(&rel) || rel.ends_with(".rlmesh-tmp") {
                continue;
            }
            files.push(EpisodeFile {
                kind: infer_artifact_kind(&rel).to_string(),
                size_bytes: path.metadata()?.len(),
                path: rel,
            });
        }
    }
    files.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(files)
}
