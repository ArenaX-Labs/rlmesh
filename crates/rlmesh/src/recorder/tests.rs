use super::*;
use crate::spaces::{Discrete, Space};
use crate::{
    DType, Env, ErrorClass, FrameSources, InfoDict, Model, ModelOptions, RunOptions,
    SessionOptions, SessionRunOptions, Tensor,
};
use std::io::Read;
use std::sync::atomic::{AtomicUsize, Ordering};

fn result() -> RunResult {
    RunResult {
        episodes: vec![crate::EpisodeResult {
            index: 8,
            seed: Some(7),
            steps: 3,
            reward: 4.5,
            success: Some(true),
            terminated: true,
            duration_s: 0.125,
            predict_ms: Some(99.0),
            trial: Some(18),
            ..Default::default()
        }],
        ..Default::default()
    }
}
fn recorder() -> Recorder {
    Recorder::new(RecorderOptions {
        result_set_id: Some("fixed".into()),
        ..Default::default()
    })
    .unwrap()
}
fn timestamp() -> RecorderExportOptions {
    RecorderExportOptions {
        recorded_at: Some("2026-10-15T12:00:00+00:00".into()),
        ..Default::default()
    }
}
#[test]
fn records_defaults_metrics_and_document_match_python() {
    assert_eq!(RecorderOptions::default().fps, 30);
    assert_eq!(RecorderOptions::default().quality, 60);
    assert_eq!(
        RecorderCaptureOptions::default().video_info_keys,
        ["video_artifact_path", "video_url"]
    );
    assert!(EpisodeRecord::default().included_in_metrics);
    let mut rec = recorder();
    let mut snapshot = rec
        .add(&result(), "model", "env", Default::default())
        .unwrap();
    assert_eq!(snapshot.episodes[0].index, 0);
    assert_eq!(snapshot.mean_reward(), 4.5);
    assert_eq!(snapshot.success_rate(), Some(1.0));
    snapshot.episodes.push(EpisodeRecord {
        steps: 2,
        reward: 100.,
        included_in_metrics: false,
        ..Default::default()
    });
    assert_eq!(snapshot.mean_reward(), 4.5);
    assert_eq!(snapshot.success_rate(), Some(1.0));
    assert_eq!(snapshot.total_steps(), 5);
    assert_eq!(
        rec.workloads()[0].total_steps(),
        3,
        "returned workloads are snapshots"
    );
    let doc = crate::specs::value_to_json(&Value::Dict(rec.to_dict(RecorderToDictOptions {
        recorded_at: Some("stamp".into()),
    })))
    .unwrap();
    assert_eq!(
        doc,
        serde_json::json!({"schema": "rlmesh.result.v1", "resultSetId": "fixed", "recordedAt": "stamp", "workloads": [{"model":"model", "env":"env", "task":"env", "config": {}, "numEpisodes":1, "metrics":{"meanReward":4.5,"successRate":1.0,"totalSteps":3},"episodes":[{"index":0,"seed":7,"steps":3,"reward":4.5,"terminated":true,"truncated":false,"success":true,"durationS":0.125,"includedInMetrics":true}]}]})
    );
    let media = MediaRef {
        camera: "cam".into(),
        kind: "video".into(),
        path: "media/x.mp4".into(),
        format: "mp4".into(),
        frame_count: Some(2),
        ..Default::default()
    };
    assert_eq!(media.to_dict().get("frameCount"), Some(&Value::Int(2)));
    assert!(!media.to_dict().contains_key("fps"));
    assert_eq!(
        media.to_string(),
        "MediaRef(camera='cam', kind='video', path='media/x.mp4', format='mp4', frame_count=2, width=None, height=None, fps=None)"
    );
    assert_eq!(
        EpisodeRecord::default().to_string(),
        "EpisodeRecord(index=0, seed=None, steps=0, reward=0.0, terminated=False, truncated=False, success=None, duration_s=0.0, included_in_metrics=True, media=())"
    );
    assert_eq!(
        WorkloadRecord::default().to_string(),
        "WorkloadRecord(model='', env='', task='', config={}, episodes=[])"
    );
}
#[test]
fn folder_zip_refusal_and_closed_metric_export() {
    let root = tempfile::tempdir().unwrap();
    let mut rec = recorder();
    rec.add(&result(), "m", "e", Default::default()).unwrap();
    let folder = rec
        .export_(root.path().join("folder"), timestamp())
        .unwrap();
    let doc = std::fs::read(folder.join("result.json")).unwrap();
    let zip_path = rec
        .export_(root.path().join("bundle.ZIP"), timestamp())
        .unwrap();
    let mut zip = zip::ZipArchive::new(std::fs::File::open(zip_path).unwrap()).unwrap();
    let mut zipped = Vec::new();
    zip.by_name("result.json")
        .unwrap()
        .read_to_end(&mut zipped)
        .unwrap();
    assert_eq!(doc, zipped);
    let wrong = root.path().join("file");
    std::fs::write(&wrong, "keep").unwrap();
    assert_eq!(
        rec.export_(&wrong, Default::default()).unwrap_err().class(),
        ErrorClass::OSError
    );
    let unknown = root.path().join("unknown");
    std::fs::create_dir_all(unknown.join("media")).unwrap();
    assert_eq!(
        rec.export_(&unknown, Default::default())
            .unwrap_err()
            .class(),
        ErrorClass::OSError
    );
    assert!(!unknown.join("result.json").exists());
    rec.close().unwrap();
    rec.close().unwrap();
    rec.export_(root.path().join("after-close"), Default::default())
        .unwrap();
}
struct VideoEnv {
    renders: Arc<AtomicUsize>,
    fail: bool,
}
impl Env for VideoEnv {
    fn observation_space(&self) -> Space {
        Discrete::new(2, Default::default()).unwrap().into()
    }
    fn action_space(&self) -> Space {
        self.observation_space()
    }
    fn reset(&mut self, _: Option<i64>, _: Option<&InfoDict>) -> Result<(Value, InfoDict)> {
        Ok((Value::Int(0), InfoDict::new()))
    }
    fn step(&mut self, _: Option<Value>) -> Result<(Value, f64, bool, bool, InfoDict)> {
        Ok((Value::Int(1), 2., true, false, InfoDict::new()))
    }
    fn render_mode(&self) -> Option<String> {
        Some("rgb_array".into())
    }
    fn render(&mut self) -> Result<Option<Value>> {
        self.renders.fetch_add(1, Ordering::SeqCst);
        if self.fail {
            Err(Error::Runtime("render failed".into()))
        } else {
            Ok(Some(Value::Tensor(Tensor::new(
                &vec![128; 16 * 16 * 3],
                &[16, 16, 3],
                DType::Uint8,
            )?)))
        }
    }
}
fn model() -> Model {
    Model::new(|_: Value| Ok(Value::Int(0)), ModelOptions::default()).unwrap()
}
#[test]
fn capture_native_and_session_render_media_and_stable_episode_indices() {
    let mut rec = recorder();
    let mut capture = rec
        .capture("model/../name", "env", Default::default())
        .unwrap();
    let mut env = VideoEnv {
        renders: Arc::default(),
        fail: false,
    };
    let model = model();
    model
        .run(
            &mut env,
            RunOptions {
                hooks: Some(&mut *capture),
                ..Default::default()
            },
        )
        .unwrap();
    {
        let mut session = model.session(&mut env, SessionOptions::default()).unwrap();
        session
            .run(SessionRunOptions {
                hooks: Some(&mut *capture),
                ..Default::default()
            })
            .unwrap();
        session.close().unwrap();
    }
    let workloads = rec.workloads();
    assert_eq!(
        workloads[0]
            .episodes
            .iter()
            .map(|e| e.index)
            .collect::<Vec<_>>(),
        [0, 1]
    );
    let root = tempfile::tempdir().unwrap();
    let path = rec
        .export_(root.path().join("bundle"), Default::default())
        .unwrap();
    for episode in &workloads[0].episodes {
        let media = &episode.media[0];
        assert_eq!(media.camera, "render");
        assert_eq!(media.frame_count, Some(1));
        assert_eq!(media.fps, Some(30.));
        let bytes = std::fs::read(path.join(&media.path)).unwrap();
        assert_eq!(&bytes[4..8], b"ftyp");
    }
    rec.close().unwrap();
    let after = root.path().join("closed");
    assert_eq!(
        rec.export_(&after, Default::default()).unwrap_err().class(),
        ErrorClass::RuntimeError
    );
    assert!(!after.exists());
}
#[test]
fn capture_survives_recorder_drop_and_metrics_opt_out() {
    let mut capture = {
        let mut rec = recorder();
        rec.capture(
            "m",
            "e",
            RecorderCaptureOptions {
                cameras: Some(vec![]),
                ..Default::default()
            },
        )
        .unwrap()
    };
    let renders = Arc::default();
    let mut env = VideoEnv {
        renders: Arc::clone(&renders),
        fail: false,
    };
    model()
        .run(
            &mut env,
            RunOptions {
                hooks: Some(&mut *capture),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(renders.load(Ordering::SeqCst), 0);
}
struct Context {
    n: u64,
    fails: bool,
}
impl RunContext for Context {
    fn contract(&self) -> Result<Option<crate::EnvContract>> {
        Ok(None)
    }
    fn num_envs(&self) -> Result<u64> {
        Ok(self.n)
    }
    fn read(&self, _: &Value, _: &crate::ReadItem) -> Result<Value> {
        Err(Error::Runtime("no image".into()))
    }
    fn frame_sources(&self) -> Result<FrameSources<'_>> {
        Ok(FrameSources {
            roles: vec![],
            render_label: Some("render".into()),
            render: Some(crate::RenderFn::new(|| {
                if self.fails {
                    Err(Error::Runtime("render failed".into()))
                } else {
                    Ok(Value::None)
                }
            })),
        })
    }
}
#[test]
fn vector_capture_refusal_and_missing_frames_warning_never_aborts() {
    let _guard = crate::warnings::tests::HANDLER_LOCK.lock().unwrap();
    let warnings = Arc::new(Mutex::new(Vec::new()));
    let sink = warnings.clone();
    crate::set_warning_handler(Some(Box::new(move |w| {
        sink.lock().unwrap().push(w.message.clone());
        Err(Error::Runtime("warnings are errors".into()))
    })));
    let mut rec = recorder();
    let mut capture = rec
        .capture(
            "m",
            "e",
            RecorderCaptureOptions {
                session: Some(Arc::new(Context { n: 2, fails: true })),
                ..Default::default()
            },
        )
        .unwrap();
    let err = capture.on_episode_start(0, None).unwrap_err();
    assert_eq!(err.class(), ErrorClass::ValueError);
    assert_eq!(
        err.to_string(),
        "recorder captures frames from a single env: a vector env's episodes interleave. Pass cameras=[] to record metrics only."
    );
    let mut capture = rec
        .capture(
            "m",
            "e",
            RecorderCaptureOptions {
                session: Some(Arc::new(Context { n: 1, fails: true })),
                cameras: Some(vec!["render".into()]),
                ..Default::default()
            },
        )
        .unwrap();
    capture.on_episode_start(0, None).unwrap();
    let event = crate::StepEvent {
        episode: 0,
        seed: None,
        step: 0,
        observation: Value::None,
        action: Value::None,
        reward: 0.,
        terminated: true,
        truncated: false,
        info: InfoDict::new(),
        predict_ms: 0.,
        step_ms: 0.,
        read: Arc::new(|_| Ok(Value::None)),
    };
    capture.on_step(&event).unwrap();
    capture.on_episode_end(&result().episodes[0]).unwrap();
    capture.on_run_end(&result()).unwrap();
    crate::set_warning_handler(None);
    assert_eq!(warnings.lock().unwrap().len(), 1);
    assert!(warnings.lock().unwrap()[0].contains("no frames captured"));
    assert!(rec.workloads()[1].episodes[0].media.is_empty());
}
#[tokio::test]
async fn async_export_close_refusal_and_drop_before_lock_produces_no_files() {
    tokio::spawn(async {
        let rec = recorder();
        let root = tempfile::tempdir().unwrap();
        let err = rec
            .export_(root.path().join("refused"), Default::default())
            .unwrap_err();
        assert_eq!(err.class(), ErrorClass::RuntimeError);
        assert!(err.to_string().contains("export_async"));
        rec.export_async(root.path().join("async"), timestamp())
            .await
            .unwrap();
        let state = rec.state.clone();
        let dropped = root.path().join("dropped");
        {
            let guard = lock(&state);
            let mut export = Box::pin(rec.export_async(&dropped, timestamp()));
            use std::future::Future;
            let mut cx = std::task::Context::from_waker(futures::task::noop_waker_ref());
            assert!(export.as_mut().poll(&mut cx).is_pending());
            drop(export);
            drop(guard);
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(!dropped.exists());
        let mut rec = rec;
        assert!(rec.close().is_err());
        rec.close_async().await.unwrap();
    })
    .await
    .unwrap();
}

#[test]
fn export_preserves_python_json_kinds_and_refuses_binary_config_before_writing() {
    let root = tempfile::tempdir().unwrap();
    let mut rec = recorder();
    let mut config = Metadata::new();
    config.insert("binary", Value::Bytes(vec![1, 2, 3]));
    rec.add(
        &result(),
        "m",
        "e",
        RecorderAddOptions {
            config: Some(config),
            ..Default::default()
        },
    )
    .unwrap();
    let target = root.path().join("invalid");
    let err = rec.export_(&target, timestamp()).unwrap_err();
    assert_eq!(err.class(), ErrorClass::TypeError);
    assert_eq!(
        err.to_string(),
        "Object of type bytes is not JSON serializable"
    );
    assert!(!target.exists());
    let mut rec = recorder();
    let mut data = result();
    data.episodes[0].reward = f64::INFINITY;
    rec.add(&data, "m", "e", Default::default()).unwrap();
    let path = rec
        .export_(root.path().join("nonfinite"), timestamp())
        .unwrap();
    let json = std::fs::read_to_string(path.join("result.json")).unwrap();
    assert!(json.contains("\"reward\": Infinity"));
    assert!(json.contains("\"meanReward\": Infinity"));
}
