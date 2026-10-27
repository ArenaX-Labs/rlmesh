//! Layout parity with the managed runner: the names, columns and record keys
//! below are the ones `rlmesh-middleman` writes (`artifacts/layout.rs`,
//! `trajectory.rs`, `artifacts/schema/records.rs`).

use std::collections::BTreeSet;
use std::path::Path;

use arrow_schema::DataType;
use serde_json::{Value as Json, json};

use super::schema::*;
use crate::adapters::*;
use crate::recorder::{Recorder, RecorderCaptureOptions, RecorderExportOptions, RecorderOptions};
use crate::spaces::{Box as BoxSpace, Dict, Space};
use crate::{DType, Env, InfoDict, Model, ModelOptions, Result, RunHooks, RunOptions, Tensor};
use crate::{RunContext, RunResult, StepEvent, Value};

const SIDE: usize = 16;

/// A camera plus a state leaf; four steps an episode, odd episodes succeed.
struct Scene {
    episode: i64,
    step: u8,
}
impl Env for Scene {
    fn observation_space(&self) -> Space {
        let camera = BoxSpace::new(
            0.,
            255.,
            vec![SIDE as i64, SIDE as i64, 3],
            crate::spaces::BoxOptions {
                dtype: DType::Uint8,
            },
        )
        .unwrap();
        let state = BoxSpace::new(-1., 1., vec![3], Default::default()).unwrap();
        Dict::new([("agentview", Space::from(camera)), ("eef", state.into())])
            .unwrap()
            .into()
    }
    fn action_space(&self) -> Space {
        BoxSpace::new(-1., 1., vec![1], Default::default())
            .unwrap()
            .into()
    }
    fn reset(&mut self, _: Option<i64>, _: Option<&InfoDict>) -> Result<(Value, InfoDict)> {
        self.episode += 1;
        self.step = 0;
        Ok((self.observation(), InfoDict::new()))
    }
    fn step(&mut self, _: Option<Value>) -> Result<(Value, f64, bool, bool, InfoDict)> {
        self.step += 1;
        let done = self.step == 4;
        let mut info = InfoDict::new();
        if done {
            info.insert("is_success", self.episode % 2 == 1);
        }
        Ok((self.observation(), 0.5, done, false, info))
    }
}
impl Scene {
    fn observation(&self) -> Value {
        let pixels = vec![self.step * 10; SIDE * SIDE * 3];
        let state = [0.25f32, -0.5, 1.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        Value::Dict(
            [
                (
                    "agentview",
                    Value::Tensor(Tensor::new(&pixels, &[SIDE, SIDE, 3], DType::Uint8).unwrap()),
                ),
                (
                    "eef",
                    Value::Tensor(Tensor::new(&state, &[3], DType::Float32).unwrap()),
                ),
            ]
            .into_iter()
            .collect(),
        )
    }
}

/// The recorder's hooks under a perturbation that inverts the camera, so
/// the raw and perturbed streams differ.
struct Inverted(Box<dyn RunHooks>);
impl RunHooks for Inverted {
    fn on_run_start(&mut self, context: std::sync::Arc<dyn RunContext>) -> Result<()> {
        self.0.on_run_start(context)
    }
    fn on_episode_start(&mut self, episode: i64, seed: Option<i64>) -> Result<()> {
        self.0.on_episode_start(episode, seed)
    }
    fn on_step(&mut self, event: &StepEvent) -> Result<()> {
        self.0.on_step(event)
    }
    fn on_episode_end(&mut self, result: &crate::EpisodeResult) -> Result<()> {
        self.0.on_episode_end(result)
    }
    fn on_run_end(&mut self, result: &RunResult) -> Result<()> {
        self.0.on_run_end(result)
    }
    fn artifacts(&mut self) -> Option<super::ArtifactTap> {
        self.0.artifacts()
    }
    fn transform_observation(&mut self, _: i64, _: i64, obs: &Value) -> Result<Option<Value>> {
        let mut obs = obs.as_dict()?.clone();
        let Some(Value::Tensor(camera)) = obs.get("agentview") else {
            return Ok(None);
        };
        let inverted = camera.buffer().iter().map(|p| 255 - p).collect::<Vec<_>>();
        obs.insert(
            "agentview",
            Value::Tensor(Tensor::new(&inverted, &[SIDE, SIDE, 3], DType::Uint8)?),
        );
        Ok(Some(Value::Dict(obs)))
    }
}

fn files(root: &Path) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path);
            } else {
                out.insert(super::video::relative(&path, root));
            }
        }
    }
    out
}

fn keys(value: &Json) -> BTreeSet<&str> {
    value
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect()
}

fn read_json(path: &Path) -> Json {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

#[test]
fn a_recorded_run_writes_the_managed_episode_layout() {
    let mut recorder = Recorder::new(RecorderOptions::default()).unwrap();
    let record: RecordSpec = serde_json::from_value(json!({
        "video": {"raw": ["primary"], "perturbed": ["primary"], "every": 2},
        "trajectory": {"obs": "both", "action": "both", "when": "failures"},
    }))
    .unwrap();
    let capture = recorder
        .capture(
            "policy",
            "scene",
            RecorderCaptureOptions {
                task: Some("scene-0".into()),
                record: Some(record),
                ..Default::default()
            },
        )
        .unwrap();
    let mut hooks = Inverted(capture);
    let mut env = tag(
        Scene {
            episode: 0,
            step: 0,
        },
        &EnvTags::new(
            ObsNode::dict([
                ("agentview", ImageTag::new(IMAGE_PRIMARY).into()),
                ("eef", StateTag::new(EEF_POS).into()),
            ]),
            Action::new([Actuator {
                role: Some(ACTION_GRIPPER.into()),
                dim: 1,
                ..Default::default()
            }]),
        )
        .unwrap(),
        Default::default(),
    )
    .unwrap();
    let model = Model::new(
        |_: Value| Ok(Value::List(vec![Value::Float(0.)])),
        ModelOptions {
            spec: Some(crate::NO_ADAPTER),
            ..Default::default()
        },
    )
    .unwrap();
    model
        .run(
            &mut env,
            RunOptions {
                episodes: Some(3),
                seeds: Some(vec![11, 12, 13]),
                hooks: Some(&mut hooks),
                ..Default::default()
            },
        )
        .unwrap();
    let out = tempfile::tempdir().unwrap();
    let bundle = recorder
        .export_(out.path().join("bundle"), RecorderExportOptions::default())
        .unwrap();

    let result = read_json(&bundle.join("result.json"));
    let workload = &result["workloads"][0];
    assert_eq!(workload["artifactRoot"], "workloads/c0001/attempts/a001");
    assert_eq!(workload["phases"][0]["phase"], "warmup");
    assert_eq!(result["stamp"]["runnerVersion"], crate::VERSION);
    let root = bundle.join("workloads/c0001/attempts/a001");

    // every=2 records video for episodes 1 and 3; when=failures keeps only
    // episode 2's trajectory. A transformed camera is encoded twice.
    assert_eq!(
        files(&root),
        [
            "episodes/ep000001/episode.json",
            "episodes/ep000001/video/primary.perturbed.mp4",
            "episodes/ep000001/video/primary.raw.mp4",
            "episodes/ep000002/episode.json",
            "episodes/ep000002/trajectory.parquet",
            "episodes/ep000003/episode.json",
            "episodes/ep000003/video/primary.perturbed.mp4",
            "episodes/ep000003/video/primary.raw.mp4",
            "telemetry/windows.jsonl",
        ]
        .map(String::from)
        .into(),
    );

    // episode.json keys: middleman's `EpisodeJson` fields.
    let episode = read_json(&root.join("episodes/ep000001/episode.json"));
    assert_eq!(
        keys(&episode),
        keys(&json!({
            "schema": 0, "taskId": 0, "workloadId": 0, "workloadArtifactId": 0, "attempt": 0,
            "episodeIndex": 0, "envIndex": 0, "seed": 0, "trialIndex": 0, "reward": 0,
            "steps": 0, "timing": 0, "verdict": 0, "terminated": 0, "truncated": 0,
            "files": 0, "recordings": 0, "flags": 0, "firstFrameHash": 0,
        }))
    );
    assert_eq!(
        [
            &episode["schema"],
            &episode["taskId"],
            &episode["workloadId"],
            &episode["episodeIndex"],
            &episode["seed"],
            &episode["verdict"],
            &episode["steps"],
        ],
        [
            &json!("rlmesh.episode.v1"),
            &json!("scene-0"),
            &json!("c0001"),
            &json!(1),
            &json!(11),
            &json!("success"),
            &json!(4),
        ]
    );
    assert_eq!(
        keys(&episode["timing"]),
        keys(&json!({"startedAt": 0, "completedAt": 0, "durationMs": 0}))
    );
    let recording = |stream: &str| {
        episode["recordings"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["stream"] == stream)
            .unwrap()
            .clone()
    };
    // Frame N is the observation step N acted on (frame 0 the reset one),
    // so frames pair with trajectory rows; the terminal one is never seen.
    assert_eq!(
        recording("raw"),
        json!({"camera": "primary", "stream": "raw", "path": "episodes/ep000001/video/primary.raw.mp4", "frames": 4, "fps": 30, "width": 16, "height": 16})
    );
    assert_eq!(
        recording("perturbed")["path"],
        "episodes/ep000001/video/primary.perturbed.mp4"
    );
    for file in episode["files"].as_array().unwrap() {
        assert_eq!(
            keys(file),
            keys(&json!({"path": 0, "kind": 0, "sizeBytes": 0}))
        );
        assert_eq!(file["kind"], "recording");
    }
    let failed = read_json(&root.join("episodes/ep000002/episode.json"));
    assert_eq!(failed["verdict"], "failure");
    assert_eq!(failed["files"][0]["kind"], "trajectory");
    assert_eq!(failed["recordings"], json!([]));

    // trajectory.parquet: middleman's columns and types, in its order.
    let trajectory =
        std::fs::File::open(root.join("episodes/ep000002/trajectory.parquet")).unwrap();
    let reader =
        parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(trajectory).unwrap();
    assert_eq!(
        reader
            .schema()
            .fields()
            .iter()
            .map(|f| (f.name().as_str(), f.data_type().clone()))
            .collect::<Vec<_>>(),
        [
            ("step", DataType::Int64),
            ("t_ns", DataType::Int64),
            ("predict_ms", DataType::Float64),
            ("step_ms", DataType::Float64),
            ("reward", DataType::Float64),
            ("info", DataType::Utf8),
            ("obs_raw", DataType::Binary),
            ("obs_raw_images", DataType::Binary),
            ("obs_perturbed", DataType::Binary),
            ("obs_perturbed_images", DataType::Binary),
            ("action_raw", DataType::Binary),
            ("action_perturbed", DataType::Binary),
        ]
    );
    let batches = reader
        .build()
        .unwrap()
        .collect::<std::result::Result<Vec<_>, _>>()
        .unwrap();
    let rows = &batches[0];
    assert_eq!(rows.num_rows(), 4);
    let column = |name: &str| {
        rows.column_by_name(name)
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::BinaryArray>()
            .unwrap()
            .value(0)
            .to_vec()
    };
    let obs: Json = serde_json::from_slice(&column("obs_raw")).unwrap();
    assert_eq!(
        obs,
        json!({
            "agentview": {"shape": [16, 16, 3], "dtype": "uint8", "offset": 0, "len": 768},
            "eef": {"shape": [3], "data": [0.25, -0.5, 1.0]},
        })
    );
    // The raw stream is what the env rendered, the perturbed one what the
    // model saw.
    assert_eq!(column("obs_raw_images"), vec![0; 768]);
    assert_eq!(column("obs_perturbed_images"), vec![255; 768]);
    let action: Json = serde_json::from_slice(&column("action_raw")).unwrap();
    assert_eq!(action, json!({"shape": [1], "data": [0.0]}));

    // windows.jsonl rows are middleman's snapshot payload.
    let windows = std::fs::read_to_string(root.join("telemetry/windows.jsonl")).unwrap();
    let session = windows
        .lines()
        .map(|line| serde_json::from_str::<Json>(line).unwrap())
        .find(|row| row["horizon"] == "session")
        .unwrap();
    assert_eq!(
        keys(&session),
        keys(&json!({
            "horizon": 0, "sessionId": 0, "envId": 0, "taskId": 0, "windowSeconds": 0,
            "emittedAtMs": 0, "rows": 0,
        }))
    );

    // The bundle's episodes point at the tree's videos.
    let media = &workload["episodes"][0]["media"];
    assert_eq!(
        media[0]["path"],
        "workloads/c0001/attempts/a001/episodes/ep000001/video/primary.raw.mp4"
    );
    assert_eq!(media[0]["camera"], "primary.raw");
}

#[test]
fn selection_and_when_rules() {
    assert!(selected(1, 0) && selected(1, 1) && selected(1, 2) && selected(3, 2));
    assert!(!selected(2, 2) && selected(1, 5) && selected(6, 5) && !selected(5, 5));
    for verdict in [Verdict::Success, Verdict::Failure, Verdict::Unknown] {
        assert!(RecordWhen::All.keeps(verdict));
    }
    assert!(RecordWhen::Failures.keeps(Verdict::Failure));
    assert!(RecordWhen::Failures.keeps(Verdict::Unknown));
    assert!(!RecordWhen::Failures.keeps(Verdict::Success));
    assert!(RecordWhen::Successes.keeps(Verdict::Success));
    assert!(!RecordWhen::Successes.keeps(Verdict::Failure));
    assert!(!RecordWhen::Successes.keeps(Verdict::Unknown));
    assert_eq!(episode_dir(12), "episodes/ep000012");
    assert_eq!(
        episode_video(12, "wrist cam", RecordStream::Perturbed),
        "episodes/ep000012/video/wrist_cam.perturbed.mp4"
    );
}

/// Two lanes under `NEXT_STEP` autoreset: lane 0's episodes end on their
/// first step, lane 1's on their third, and each restarts on the next.
struct Staggered {
    counts: [u32; 2],
    ended: [bool; 2],
}
impl crate::VectorEnv for Staggered {
    fn num_envs(&self) -> u64 {
        2
    }
    fn single_observation_space(&self) -> Space {
        BoxSpace::new(-1., 1., vec![1], Default::default())
            .unwrap()
            .into()
    }
    fn single_action_space(&self) -> Space {
        BoxSpace::new(-1., 1., vec![1], Default::default())
            .unwrap()
            .into()
    }
    fn reset(
        &mut self,
        _: Option<crate::VectorSeed>,
        _: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        *self = Self {
            counts: [0; 2],
            ended: [false; 2],
        };
        Ok((lanes(2), InfoDict::new()))
    }
    fn step(&mut self, _: Value) -> Result<(Value, Vec<f64>, Vec<bool>, Vec<bool>, InfoDict)> {
        let mut terminated = vec![false; 2];
        for (lane, length) in [1, 3].into_iter().enumerate() {
            if self.ended[lane] {
                (self.counts[lane], self.ended[lane]) = (0, false);
            } else {
                self.counts[lane] += 1;
                self.ended[lane] = self.counts[lane] == length;
                terminated[lane] = self.ended[lane];
            }
        }
        Ok((
            lanes(2),
            vec![1.; 2],
            terminated,
            vec![false; 2],
            InfoDict::new(),
        ))
    }
    fn metadata(&self) -> Option<crate::Metadata> {
        let mut metadata = crate::Metadata::new();
        metadata.insert("autoreset_mode", Value::Text("NextStep".into()));
        Some(metadata)
    }
}

/// `count` lanes of a `float32 [1]` zero.
fn lanes(count: usize) -> Value {
    Value::Tensor(Tensor::new(&vec![0; 4 * count], &[count, 1], DType::Float32).unwrap())
}

#[test]
fn a_next_step_autoreset_roll_is_not_a_trajectory_row() {
    let mut recorder = Recorder::new(RecorderOptions::default()).unwrap();
    let record: RecordSpec =
        serde_json::from_value(json!({"trajectory": {"action": "raw"}})).unwrap();
    let mut capture = recorder
        .capture(
            "policy",
            "staggered",
            RecorderCaptureOptions {
                record: Some(record),
                ..Default::default()
            },
        )
        .unwrap();
    let model = Model::new(
        |_: Value| Ok(lanes(2)),
        ModelOptions {
            spec: Some(crate::NO_ADAPTER),
            ..Default::default()
        },
    )
    .unwrap();
    model
        .run(
            crate::EnvTarget::vector(&mut Staggered {
                counts: [0; 2],
                ended: [false; 2],
            }),
            RunOptions {
                episodes: Some(4),
                hooks: Some(&mut *capture),
                ..Default::default()
            },
        )
        .unwrap();
    let out = tempfile::tempdir().unwrap();
    let bundle = recorder
        .export_(out.path().join("bundle"), RecorderExportOptions::default())
        .unwrap();
    let root = bundle.join("workloads/c0001/attempts/a001/episodes");
    let mut rows = BTreeSet::new();
    assert_eq!(std::fs::read_dir(&root).unwrap().count(), 4);
    for dir in std::fs::read_dir(&root).unwrap() {
        let dir = dir.unwrap().path();
        let episode = read_json(&dir.join("episode.json"));
        let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
            std::fs::File::open(dir.join("trajectory.parquet")).unwrap(),
        )
        .unwrap()
        .build()
        .unwrap();
        let steps: Vec<i64> = reader
            .flat_map(|batch| {
                let batch = batch.unwrap();
                batch
                    .column_by_name("step")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<arrow_array::Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect();
        rows.insert((episode["envIndex"].as_i64().unwrap(), steps));
    }
    // One row per step, numbered within its episode.
    assert_eq!(
        rows,
        [(0, vec![1]), (1, vec![1, 2, 3])].into_iter().collect()
    );
}

/// `(envIndex, trajectory rows)` per recorded episode under `root`.
fn episode_rows(root: &Path) -> Vec<(i64, usize)> {
    let mut rows = Vec::new();
    for dir in std::fs::read_dir(root.join("episodes")).unwrap() {
        let dir = dir.unwrap().path();
        let episode = read_json(&dir.join("episode.json"));
        // An episode that closed no step writes no trajectory.
        let count = std::fs::File::open(dir.join("trajectory.parquet")).map_or(0, |file| {
            parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
                .unwrap()
                .build()
                .unwrap()
                .map(|batch| batch.unwrap().num_rows())
                .sum()
        });
        rows.push((episode["envIndex"].as_i64().unwrap(), count));
    }
    rows.sort_unstable();
    rows
}

#[test]
fn episodes_of_a_lane_endpoint_record_their_own_lane() {
    let server = crate::EnvServer::lanes(
        (0..2)
            .map(|_| Scene {
                episode: 0,
                step: 0,
            })
            .collect(),
        crate::EnvServerOptions {
            address: Some(crate::tests::ENV_ADDRESS.into()),
            options: Some(crate::tests::env_serve_options()),
            ..Default::default()
        },
    )
    .unwrap();
    server.start().unwrap();
    let mut recorder = Recorder::new(RecorderOptions::default()).unwrap();
    let record: RecordSpec =
        serde_json::from_value(json!({"trajectory": {"action": "raw"}})).unwrap();
    let mut capture = recorder
        .capture(
            "policy",
            "scene",
            RecorderCaptureOptions {
                record: Some(record),
                ..Default::default()
            },
        )
        .unwrap();
    let model = Model::new(
        |_: Value| Ok(Value::List(vec![Value::Float(0.)])),
        ModelOptions {
            spec: Some(crate::NO_ADAPTER),
            ..Default::default()
        },
    )
    .unwrap();
    model
        .run(
            server.address(),
            RunOptions {
                episodes: Some(4),
                hooks: Some(&mut *capture),
                ..Default::default()
            },
        )
        .unwrap();
    server.shutdown().unwrap();
    let out = tempfile::tempdir().unwrap();
    let bundle = recorder
        .export_(out.path().join("bundle"), RecorderExportOptions::default())
        .unwrap();
    // Four steps an episode, two episodes a lane.
    assert_eq!(
        episode_rows(&bundle.join("workloads/c0001/attempts/a001")),
        [(0, 4), (0, 4), (1, 4), (1, 4)]
    );
}
