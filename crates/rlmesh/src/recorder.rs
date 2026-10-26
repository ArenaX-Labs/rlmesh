//! Portable evaluation bundles.
pub(crate) mod artifacts;
mod capture;
mod export;
mod records;
#[cfg(test)]
mod tests;
/// The video writer behind capture (R-25 binding plumbing: the
/// recorder and the Python binding drive it).
#[doc(hidden)]
pub mod video;
use crate::{Error, Metadata, Result, RunContext, RunHooks, RunResult, TelemetryRow, Value};
use artifacts::schema::{AttemptStamp, RecordSpec};
use artifacts::{ArtifactOptions, ArtifactTap, EpisodeArtifacts};
pub use records::{EpisodeRecord, MediaRef, WorkloadRecord};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

/// Recorded video codec.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum VideoCodec {
    /// AV1, encoded in process (rav1e); needs nothing installed.
    #[default]
    Av1,
    /// H.264 (libx264), encoded by the user's own ffmpeg; rlmesh ships no
    /// H.264 encoder.
    H264,
}
impl VideoCodec {
    /// The codec name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Av1 => "av1",
            Self::H264 => "h264",
        }
    }
}
impl std::fmt::Display for VideoCodec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
/// The portable result bundle schema tag.
pub const SCHEMA: &str = "rlmesh.result.v1";
/// Recorder construction keywords.
#[derive(Clone, Debug)]
pub struct RecorderOptions {
    /// Bundle identity; a fresh UUID hex when absent or empty.
    pub result_set_id: Option<String>,
    /// Playback rate, at least one frame per second.
    pub fps: u64,
    /// AV1 quality in 1..100.
    pub quality: u8,
    /// Video codec; H.264 runs `ffmpeg`.
    pub codec: VideoCodec,
    /// The ffmpeg H.264 runs; `ffmpeg` on `PATH` when absent.
    pub ffmpeg: Option<PathBuf>,
}
impl Default for RecorderOptions {
    fn default() -> Self {
        Self {
            result_set_id: None,
            fps: 30,
            quality: 60,
            codec: VideoCodec::Av1,
            ffmpeg: None,
        }
    }
}
/// Keywords for recording a completed run.
#[derive(Clone, Debug)]
pub struct RecorderAddOptions {
    /// Task label; defaults to the env name.
    pub task: Option<String>,
    /// Caller metadata, copied into the workload.
    pub config: Option<Metadata>,
    /// Include the episodes in reward and success aggregates.
    pub included_in_metrics: bool,
}
impl Default for RecorderAddOptions {
    fn default() -> Self {
        Self {
            task: None,
            config: None,
            included_in_metrics: true,
        }
    }
}
/// Keywords for live recording.
#[derive(Clone)]
pub struct RecorderCaptureOptions {
    /// Task label; defaults to the env name.
    pub task: Option<String>,
    /// Caller metadata, copied into the workload.
    pub config: Option<Metadata>,
    /// Sources to record; absent discovers all, empty records metrics only.
    pub cameras: Option<Vec<String>>,
    /// Explicit context for hand-driven hooks; otherwise adopted from the run.
    pub binding: Option<Arc<dyn RunContext>>,
    /// Step-info keys containing an env-produced video path.
    pub video_info_keys: Vec<String>,
    /// Include the episodes in reward and success aggregates.
    pub included_in_metrics: bool,
    /// Binding plumbing for Python's Experimental `Recorder.record` (R-25):
    /// write the per-episode artifact tree this spec selects (episode
    /// records, trajectories, sampled videos) instead of `cameras` media.
    #[doc(hidden)]
    pub record: Option<RecordSpec>,
}
impl Default for RecorderCaptureOptions {
    fn default() -> Self {
        Self {
            task: None,
            config: None,
            cameras: None,
            binding: None,
            video_info_keys: vec!["video_artifact_path".into(), "video_url".into()],
            included_in_metrics: true,
            record: None,
        }
    }
}
impl std::fmt::Debug for RecorderCaptureOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecorderCaptureOptions")
            .field("task", &self.task)
            .field("config", &self.config)
            .field("cameras", &self.cameras)
            .field("binding", &self.binding.as_ref().map(|_| "<context>"))
            .field("video_info_keys", &self.video_info_keys)
            .field("included_in_metrics", &self.included_in_metrics)
            .field("record", &self.record)
            .finish()
    }
}
/// Keywords for a recorder document snapshot.
#[derive(Clone, Debug, Default)]
pub struct RecorderToDictOptions {
    /// Timestamp; otherwise current UTC ISO-8601 time.
    pub recorded_at: Option<String>,
}
/// Keywords for exporting a bundle.
#[derive(Clone, Debug, Default)]
pub struct RecorderExportOptions {
    /// ZIP when true, folder when false; otherwise infer from `.zip`.
    pub archive: Option<bool>,
    /// Timestamp; otherwise current UTC ISO-8601 time.
    pub recorded_at: Option<String>,
}
struct State {
    workloads: Vec<WorkloadRecord>,
    /// Each workload's telemetry rows, one batch per recorded run.
    telemetry: Vec<Vec<TelemetryRow>>,
    /// Each workload's artifact tree: its bundle prefix and writer.
    trees: Vec<Option<(String, ArtifactTap)>>,
    /// The machine's stamp, taken with the first artifact tree.
    stamp: Option<AttemptStamp>,
    staging: Option<tempfile::TempDir>,
    assets: Vec<(PathBuf, String)>,
    used: std::collections::HashSet<String>,
    closed: bool,
    fps: u64,
    encoding: video::Encoding,
}
impl Drop for State {
    fn drop(&mut self) {
        // A live capture retains State. The last owner's cleanup must not do
        // recursive filesystem I/O on an async executor's thread (R-17).
        if crate::run::drive::in_async_context()
            && let Some(dir) = self.staging.take()
        {
            crate::run::drive::runtime().spawn_blocking(move || drop(dir));
        }
    }
}
impl State {
    fn staging(&mut self) -> Result<&Path> {
        if self.closed {
            return Err(Error::Runtime("Recorder is closed".into()));
        }
        if self.staging.is_none() {
            self.staging = Some(
                tempfile::Builder::new()
                    .prefix("rlmesh-recorder-")
                    .tempdir()
                    .map_err(os)?,
            );
        }
        Ok(self
            .staging
            .as_ref()
            .expect("staging was initialized above")
            .path())
    }
    fn stage(&mut self, prefix: &str, index: i64, filename: &str) -> Result<(PathBuf, String)> {
        self.staging()?;
        let mut rel = format!("media/{prefix}/ep{index:05}/{filename}");
        let (stem, ext) = filename.rsplit_once('.').unwrap_or((filename, ""));
        let mut n = 2;
        while !self.used.insert(rel.clone()) {
            rel = format!("media/{prefix}/ep{index:05}/{stem}-{n}.{ext}");
            n += 1;
        }
        let path = self
            .staging
            .as_ref()
            .expect("staging was initialized above")
            .path()
            .join(uuid::Uuid::new_v4().simple().to_string());
        Ok((path, rel))
    }
    fn workload(
        &mut self,
        model: &str,
        env: &str,
        task: Option<String>,
        config: Option<Metadata>,
    ) -> usize {
        let ordinal = self.workloads.len();
        self.workloads.push(WorkloadRecord {
            model: model.into(),
            env: env.into(),
            task: task.unwrap_or_else(|| env.into()),
            config: config.unwrap_or_default(),
            episodes: Vec::new(),
        });
        self.telemetry.push(Vec::new());
        self.trees.push(None);
        ordinal
    }
    /// Export a file a writer staged at `source` as `rel`, once.
    fn adopt(&mut self, source: PathBuf, rel: String) {
        if self.used.insert(rel.clone()) {
            self.assets.push((source, rel));
        }
    }
}
fn os(err: impl std::fmt::Display) -> Error {
    Error::Os(err.to_string())
}
fn lock<T>(state: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    state.lock().unwrap_or_else(|err| err.into_inner())
}
fn now() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Micros, false)
}
fn document(id: &str, state: &State, recorded_at: Option<String>) -> Metadata {
    [
        ("schema", SCHEMA.into()),
        ("resultSetId", id.into()),
        (
            "recordedAt",
            recorded_at
                .filter(|s| !s.is_empty())
                .unwrap_or_else(now)
                .into(),
        ),
        (
            "workloads",
            Value::List(
                state
                    .workloads
                    .iter()
                    .zip(&state.telemetry)
                    .zip(&state.trees)
                    .map(|((w, rows), tree)| {
                        let mut workload = w.to_dict();
                        if !rows.is_empty() {
                            workload.insert("telemetry", telemetry(rows));
                        }
                        if let Some((prefix, tap)) = tree {
                            workload.insert("artifactRoot", Value::Text(prefix.clone()));
                            let phases = lock(tap).phases().to_vec();
                            if !phases.is_empty() {
                                workload.insert("phases", json_value(&phases));
                            }
                        }
                        Value::Dict(workload)
                    })
                    .collect(),
            ),
        ),
    ]
    .into_iter()
    .chain(
        state
            .stamp
            .as_ref()
            .map(|stamp| ("stamp", json_value(stamp))),
    )
    .collect()
}
/// A serde record as a bundle value.
fn json_value(record: &impl serde::Serialize) -> Value {
    fn value(json: serde_json::Value) -> Value {
        match json {
            serde_json::Value::Null => Value::None,
            serde_json::Value::Bool(b) => b.into(),
            serde_json::Value::Number(n) => match n.as_i64() {
                Some(i) => i.into(),
                None => n.as_f64().unwrap_or(f64::NAN).into(),
            },
            serde_json::Value::String(s) => s.into(),
            serde_json::Value::Array(items) => Value::List(items.into_iter().map(value).collect()),
            serde_json::Value::Object(map) => {
                Value::Dict(map.into_iter().map(|(k, v)| (k, value(v))).collect())
            }
        }
    }
    value(serde_json::to_value(record).unwrap_or_default())
}
fn telemetry(rows: &[TelemetryRow]) -> Value {
    Value::List(
        rows.iter()
            .map(|r| {
                Value::Dict(
                    [
                        ("op", Value::Text(r.op.clone())),
                        ("component", r.component.clone().into()),
                        ("metric", r.metric.clone().into()),
                        ("unit", r.unit.clone().into()),
                        ("count", r.count.into()),
                        ("avg", r.avg.into()),
                        ("p50", r.p50.into()),
                        ("p95", r.p95.into()),
                        ("p99", r.p99.into()),
                    ]
                    .into_iter()
                    .collect(),
                )
            })
            .collect(),
    )
}
/// Accumulate workloads and export a portable result bundle.
///
/// Capture hooks retain shared state independently of this handle. Returned
/// workload values are snapshots; a hook can accumulate several runs.
/// Explicit `close` removes staged assets; dropping a handle leaves live
/// capture hooks valid and the last owner removes staging.
pub struct Recorder {
    id: String,
    state: Arc<Mutex<State>>,
}
impl std::fmt::Debug for Recorder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Recorder")
            .field("result_set_id", &self.id)
            .finish_non_exhaustive()
    }
}
impl Recorder {
    /// Construct a recorder, validating its capture and export options.
    pub fn new(options: RecorderOptions) -> Result<Self> {
        if options.fps == 0 {
            return Err(Error::invalid_argument(
                "Recorder.fps must be an integer >= 1; got 0",
            ));
        }
        if !(1..=100).contains(&options.quality) {
            return Err(Error::invalid_argument(format!(
                "Recorder.quality must be an integer in 1..=100; got {}",
                options.quality
            )));
        }
        let encoding =
            video::Encoding::resolve(options.codec, options.quality, options.ffmpeg.as_deref())
                .map_err(|err| Error::Runtime(err.to_string()))?;
        Ok(Self {
            id: options
                .result_set_id
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string()),
            state: Arc::new(Mutex::new(State {
                workloads: Vec::new(),
                telemetry: Vec::new(),
                trees: Vec::new(),
                stamp: None,
                staging: None,
                assets: Vec::new(),
                used: Default::default(),
                closed: false,
                fps: options.fps,
                encoding,
            })),
        })
    }
    /// Bundle identity (re-upload deduplication key).
    pub fn result_set_id(&self) -> &str {
        &self.id
    }
    /// Snapshot every workload in insertion order.
    pub fn workloads(&self) -> Vec<WorkloadRecord> {
        lock(&self.state).workloads.clone()
    }
    /// Record a completed run, without media.
    pub fn add(
        &mut self,
        result: &RunResult,
        model: &str,
        env: &str,
        options: RecorderAddOptions,
    ) -> Result<WorkloadRecord> {
        let mut state = lock(&self.state);
        let index = state.workload(model, env, options.task, options.config);
        state.telemetry[index].extend(result.telemetry.iter().cloned());
        let workload = &mut state.workloads[index];
        workload
            .episodes
            .extend(result.episodes.iter().enumerate().map(|(i, e)| {
                EpisodeRecord::from_result(e, i as i64, options.included_in_metrics, Vec::new())
            }));
        Ok(workload.clone())
    }
    /// Start a workload and return independent, shareable-lifetime capture hooks.
    pub fn capture(
        &mut self,
        model: &str,
        env: &str,
        mut options: RecorderCaptureOptions,
    ) -> Result<Box<dyn RunHooks>> {
        let mut state = lock(&self.state);
        // A tree stages from the start; a closed recorder refuses it here.
        let root = match options.record {
            Some(_) => Some(
                state
                    .staging()?
                    .join(uuid::Uuid::new_v4().simple().to_string()),
            ),
            None => None,
        };
        let index = state.workload(model, env, options.task.clone(), options.config.clone());
        let prefix = format!(
            "w{index:03}-{}/{}",
            capture::sanitize(model),
            capture::sanitize(&state.workloads[index].task)
        );
        let tap = match options.record.take().zip(root) {
            Some((record, root)) => {
                // The platform's attempt tree, positional as there: workload
                // `c0001` is the first, and a local run is its one attempt.
                let workload = format!("c{:04}", index + 1);
                let tap: ArtifactTap = Arc::new(Mutex::new(EpisodeArtifacts::new(
                    root,
                    ArtifactOptions {
                        record,
                        task_id: state.workloads[index].task.clone(),
                        workload_id: workload.clone(),
                        workload_artifact_id: workload.clone(),
                        encoding: state.encoding.clone(),
                        ..ArtifactOptions::default()
                    },
                )));
                state.trees[index] =
                    Some((format!("workloads/{workload}/attempts/a001"), tap.clone()));
                if state.stamp.is_none() {
                    state.stamp = Some(artifacts::collect_stamp());
                }
                // The tree records the videos; cameras= media stays off.
                options.cameras = Some(Vec::new());
                Some(tap)
            }
            None => None,
        };
        Ok(Box::new(capture::Capture::new(
            self.state.clone(),
            index,
            prefix,
            options,
            tap,
        )))
    }
    /// Snapshot the `rlmesh.result.v1` document.
    pub fn to_dict(&self, options: RecorderToDictOptions) -> Metadata {
        document(&self.id, &lock(&self.state), options.recorded_at)
    }
    /// Write a folder or ZIP bundle. Filesystem failures have the OSError class.
    pub fn export_(
        &self,
        path: impl AsRef<Path>,
        options: RecorderExportOptions,
    ) -> Result<PathBuf> {
        if crate::run::drive::in_async_context() {
            return Err(Error::Runtime(
                "Recorder::export_() blocks; call Recorder::export_async() from async code".into(),
            ));
        }
        crate::run::drive::block_on_library(self.export_async(path, options))
    }
    /// The async twin of `export_`. Dropping it stops further file operations.
    pub async fn export_async(
        &self,
        path: impl AsRef<Path>,
        options: RecorderExportOptions,
    ) -> Result<PathBuf> {
        let path = path.as_ref().to_path_buf();
        let state = self.state.clone();
        let id = self.id.clone();
        crate::run::drive::in_library(crate::warnings::blocking(move || {
            let state = lock(&state);
            export::write(
                &path,
                &document(&id, &state, options.recorded_at),
                &state,
                options.archive,
            )
        }))
        .await?
    }
    /// Remove staged media. Idempotent; metric-only exports still work afterwards.
    pub fn close(&mut self) -> Result<()> {
        crate::run::drive::refuse_in_async_context("Recorder::close")?;
        crate::run::drive::block_on_library(self.close_async())
    }
    /// The async twin of `close`.
    pub async fn close_async(&mut self) -> Result<()> {
        let state = self.state.clone();
        crate::run::drive::in_library(crate::warnings::blocking(move || {
            let mut state = lock(&state);
            crate::warnings::check_cancelled()?;
            state.closed = true;
            if let Some(dir) = state.staging.take() {
                dir.close().map_err(os)?;
            }
            Ok(())
        }))
        .await?
    }
}
