//! Best-effort live capture shared by both run drivers.
use super::{EpisodeRecord, MediaRef, RecorderCaptureOptions, State, lock, os};
use crate::env::factory::{py_repr, py_str};
use crate::{EpisodeResult, Error, Result, RunContext, RunHooks, RunResult, StepEvent, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
    sync::{Arc, Mutex},
};

pub(super) fn sanitize(part: &str) -> String {
    let mut out = String::new();
    let mut unsafe_run = false;
    for c in part.chars() {
        if c.is_ascii_alphanumeric() || "._-".contains(c) {
            out.push(c);
            unsafe_run = false;
        } else if !unsafe_run {
            out.push('_');
            unsafe_run = true;
        }
    }
    let out = out.trim_matches(['.', '_', '-']);
    if out.is_empty() {
        "unnamed".into()
    } else {
        out.into()
    }
}
fn warn(message: impl Into<String>) {
    let _ = crate::warnings::warn(crate::WarningCategory::UserWarning, message);
}
struct Camera {
    writer: super::video::VideoWriter,
    staged: PathBuf,
    rel: String,
}
pub(super) struct Capture {
    state: Arc<Mutex<State>>,
    index: usize,
    prefix: String,
    options: RecorderCaptureOptions,
    context: Option<Arc<dyn RunContext>>,
    explicit_context: bool,
    cameras: Option<Vec<String>>,
    explicit: bool,
    render_camera: Option<String>,
    writers: BTreeMap<String, Camera>,
    disabled: BTreeSet<String>,
    captured: BTreeSet<String>,
    video_path: Option<String>,
    /// The workload's artifact tree writer, when it records one.
    tap: Option<super::ArtifactTap>,
}
impl Capture {
    pub(super) fn new(
        state: Arc<Mutex<State>>,
        index: usize,
        prefix: String,
        options: RecorderCaptureOptions,
        tap: Option<super::ArtifactTap>,
    ) -> Self {
        Self {
            tap,
            state,
            index,
            prefix,
            context: options.binding.clone(),
            explicit_context: options.binding.is_some(),
            cameras: options.cameras.clone(),
            explicit: options.cameras.is_some(),
            render_camera: options.cameras.as_ref().map(|_| "render".into()),
            writers: BTreeMap::new(),
            disabled: BTreeSet::new(),
            captured: BTreeSet::new(),
            video_path: None,
            options,
        }
    }
    fn sources(&mut self) {
        if self.cameras.is_some() {
            return;
        }
        let sources = self
            .context
            .as_ref()
            .and_then(|ctx| ctx.frame_sources().ok());
        self.render_camera = sources.as_ref().and_then(|s| s.render_label.clone());
        let cameras = sources.map(|s| s.labels()).unwrap_or_default();
        if cameras.is_empty() {
            warn(
                "recorder: no render() or image roles discovered from the session; recording metrics only",
            );
        }
        self.cameras = Some(cameras);
    }
    fn read(&self, event: &StepEvent, camera: &str) -> Option<crate::view::Frame> {
        if Some(camera) == self.render_camera.as_deref() {
            if self.explicit
                && let Some(frame) = crate::view::read_frame(&event.read, camera)
            {
                return Some(frame);
            }
            // Resolve the lent render thunk within this dispatch window. Keeping a
            // borrowed callable in the hooks would outlive a local env's loan.
            let ctx = self.context.as_ref()?;
            let mut sources = ctx.frame_sources().ok()?;
            return crate::view::frame(&sources.render.as_mut()?.call().ok()?);
        }
        crate::view::read_frame(&event.read, camera)
    }
    fn write_frame(&mut self, camera: &str, frame: crate::view::Frame) -> Result<()> {
        if !self.writers.contains_key(camera) {
            let mut state = lock(&self.state);
            let episode = state.workloads[self.index].episodes.len() as i64;
            let (staged, rel) =
                state.stage(&self.prefix, episode, &format!("{}.mp4", sanitize(camera)))?;
            // The encoder uses a u32 MP4 timebase. Saturating an unbounded fps
            // only affects rates no physical stream can produce.
            let writer = super::video::VideoWriter::create(
                &staged,
                frame.width,
                frame.height,
                state.fps.min(u32::MAX as u64) as u32,
                state.quality,
            )
            .map_err(os)?;
            self.writers.insert(
                camera.into(),
                Camera {
                    writer,
                    staged,
                    rel,
                },
            );
        }
        let cam = self
            .writers
            .get_mut(camera)
            .expect("camera was just opened");
        cam.writer
            .push(&frame.data, frame.width, frame.height, frame.channels)
            .map_err(os)?;
        self.captured.insert(camera.into());
        Ok(())
    }
    fn tree_prefix(&self) -> String {
        lock(&self.state).trees[self.index]
            .as_ref()
            .map(|(prefix, _)| prefix.clone())
            .unwrap_or_default()
    }
    /// The artifact tree's record of the episode that just ended: its files
    /// join the bundle and its videos the episode's media.
    fn adopt_episode(&self) -> Vec<MediaRef> {
        let Some(tap) = &self.tap else {
            return Vec::new();
        };
        let (record, root) = {
            let mut artifacts = lock(tap);
            (artifacts.take_finished(), artifacts.root().to_path_buf())
        };
        let Some(record) = record else {
            return Vec::new();
        };
        let prefix = self.tree_prefix();
        let dir = super::artifacts::schema::episode_dir(record.episode_index);
        let mut state = lock(&self.state);
        for path in record
            .files
            .iter()
            .map(|file| file.path.clone())
            .chain([format!("{dir}/{}", super::artifacts::schema::EPISODE_JSON)])
        {
            state.adopt(root.join(&path), format!("{prefix}/{path}"));
        }
        record
            .recordings
            .iter()
            .map(|r| MediaRef {
                camera: format!("{}.{}", r.camera, r.stream.as_str()),
                kind: "video".into(),
                path: format!("{prefix}/{}", r.path),
                format: "mp4".into(),
                frame_count: Some(r.frames as i64),
                width: Some(i64::from(r.width)),
                height: Some(i64::from(r.height)),
                fps: Some(f64::from(r.fps)),
            })
            .collect()
    }
    fn carry(&self, source: &str, index: i64) -> Result<Option<MediaRef>> {
        let root = std::env::current_dir()
            .and_then(std::fs::canonicalize)
            .map_err(os)?;
        let Ok(source) = std::fs::canonicalize(root.join(source)) else {
            return Ok(None);
        };
        if !source.starts_with(&root) || !source.is_file() {
            return Ok(None);
        }
        let ext = source
            .extension()
            .and_then(|s| s.to_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("bin")
            .to_ascii_lowercase();
        let mut state = lock(&self.state);
        let (staged, rel) = state.stage(&self.prefix, index, &format!("default.{ext}"))?;
        std::fs::copy(source, &staged).map_err(os)?;
        state.assets.push((staged, rel.clone()));
        Ok(Some(MediaRef {
            camera: "default".into(),
            kind: "video".into(),
            path: rel,
            format: ext,
            ..Default::default()
        }))
    }
}
impl RunHooks for Capture {
    fn on_run_start(&mut self, context: Arc<dyn RunContext>) -> Result<()> {
        if !self.explicit_context {
            self.context = Some(context);
        }
        if let Some(tap) = &self.tap {
            // Several runs into one workload continue its episode numbering.
            let recorded = lock(&self.state).workloads[self.index].episodes.len();
            lock(tap).set_episode_offset(u32::try_from(recorded).unwrap_or(u32::MAX));
        }
        Ok(())
    }
    fn artifacts(&mut self) -> Option<super::ArtifactTap> {
        self.tap.clone()
    }
    fn on_episode_start(&mut self, _episode: i64, _seed: Option<i64>) -> Result<()> {
        if (self.cameras.as_ref().is_none_or(|c| !c.is_empty()))
            && let Some(ctx) = &self.context
            && ctx.num_envs()? > 1
        {
            return Err(Error::invalid_argument(
                "recorder captures frames from a single env: a vector env's episodes interleave. Pass cameras=[] to record metrics only.",
            ));
        }
        self.writers.clear();
        self.video_path = None;
        Ok(())
    }
    fn on_step(&mut self, event: &StepEvent) -> Result<()> {
        self.sources();
        for key in &self.options.video_info_keys {
            if let Some(Value::Text(path)) = event.info.get(key)
                && !path.is_empty()
            {
                self.video_path = Some(path.clone());
                break;
            }
        }
        for camera in self.cameras.clone().unwrap_or_default() {
            if self.disabled.contains(&camera) {
                continue;
            }
            if let Some(frame) = self.read(event, &camera)
                && let Err(err) = self.write_frame(&camera, frame)
            {
                self.disabled.insert(camera.clone());
                self.writers.remove(&camera);
                warn(format!(
                    "recorder: disabling camera {} after an error: {err}",
                    py_str(&camera)
                ));
            }
        }
        Ok(())
    }
    fn on_episode_end(&mut self, result: &EpisodeResult) -> Result<()> {
        let index = lock(&self.state).workloads[self.index].episodes.len() as i64;
        let mut media = self.adopt_episode();
        for (camera, cam) in std::mem::take(&mut self.writers) {
            match cam.writer.finish() {
                Ok(stats) => {
                    let mut state = lock(&self.state);
                    state.assets.push((cam.staged, cam.rel.clone()));
                    media.push(MediaRef {
                        camera,
                        kind: "video".into(),
                        path: cam.rel,
                        format: "mp4".into(),
                        frame_count: Some(stats.frame_count as i64),
                        width: Some(stats.width as i64),
                        height: Some(stats.height as i64),
                        fps: Some(state.fps as f64),
                    });
                }
                Err(err) => warn(format!(
                    "recorder: dropping video for {}: {err}",
                    py_str(&camera)
                )),
            }
        }
        if let Some(path) = self.video_path.take() {
            match self.carry(&path, index) {
                Ok(Some(reference)) => media.push(reference),
                Ok(None) => warn(format!(
                    "recorder: env video {} is not a readable local file under the run directory; skipping",
                    py_str(&path)
                )),
                Err(err) => warn(format!("recorder: dropping env video: {err}")),
            }
        }
        lock(&self.state).workloads[self.index]
            .episodes
            .push(EpisodeRecord::from_result(
                result,
                index,
                self.options.included_in_metrics,
                media,
            ));
        Ok(())
    }
    fn on_run_end(&mut self, result: &RunResult) -> Result<()> {
        self.writers.clear();
        if let Some(tap) = &self.tap {
            let root = lock(tap).root().to_path_buf();
            let windows = root.join(super::artifacts::schema::TELEMETRY_WINDOWS);
            if windows.is_file() {
                let rel = format!(
                    "{}/{}",
                    self.tree_prefix(),
                    super::artifacts::schema::TELEMETRY_WINDOWS
                );
                lock(&self.state).adopt(windows, rel);
            }
        }
        lock(&self.state).telemetry[self.index].extend(result.telemetry.iter().cloned());
        let missing: Vec<_> = self
            .cameras
            .as_ref()
            .into_iter()
            .flatten()
            .filter(|c| !self.captured.contains(*c) && !self.disabled.contains(*c))
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        if !missing.is_empty() {
            warn(format!(
                "recorder: no frames captured for {} (role absent or not a 1/3/4-channel image)",
                py_repr(&Value::List(missing.into_iter().map(Value::Text).collect()))
            ));
        }
        Ok(())
    }
}
