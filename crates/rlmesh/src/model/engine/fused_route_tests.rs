use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use rlmesh_adapters::v1::{
    EnvTags, HistoryWindow, MAX_EXECUTION_HORIZON, ModelSpec, NoCustoms, NoEncodings, SpaceView,
    Value, resolve,
};

use super::corners::take_prefix;
use super::route::{FRAME_HISTORY_LIMIT_DEFAULT, FRAME_HISTORY_LIMIT_ENV, check_history_budget};
use super::*;
use crate::Error;
use crate::binding::spaces::EnvContract;
use crate::binding::spaces::{self, DType, Tensor};
use crate::model::handler::{ResolveOptions, RouteNeeds};
use crate::model::predict_fn::RouteConfig;
use crate::model::types::ModelRouteContext;

const ENV_TAGS: &str = r#"{
        "observation": {"state": {"type": "state", "role": "proprio/gripper"}},
        "action": {"components": [{"role": "action/gripper", "dim": 1, "range": [0.0, 10.0]}]}
    }"#;
const MODEL_SPEC: &str = r#"{
        "input": {"state": {"type": "state", "container": "list", "dtype": "float32",
                            "components": [{"role": "proprio/gripper", "dim": 1}]}},
        "output": {"components": [{"role": "action/gripper", "dim": 1, "range": [0.0, 10.0]}]}
    }"#;

struct TagResolver;

#[async_trait]
impl RouteResolver for TagResolver {
    async fn resolve(
        &self,
        _route_key: &str,
        env_contract: &EnvContract,
    ) -> Result<Option<RouteConfig>> {
        let tags: EnvTags = serde_json::from_str(ENV_TAGS).expect("env tags parse");
        let spec: ModelSpec = serde_json::from_str(MODEL_SPEC).expect("model spec parse");
        let obs = env_contract
            .observation_space
            .clone()
            .expect("contract obs space");
        let action = env_contract
            .action_space
            .clone()
            .expect("contract action space");
        let adapter = resolve(
            &tags,
            &SpaceView::from(&obs),
            &SpaceView::from(&action),
            &spec,
            true,
        )
        .map_err(|err| Error::model(err.message))?;
        Ok(Some(RouteConfig::new(
            adapter,
            obs,
            action,
            Box::new(NoCustoms),
            Box::new(NoEncodings),
        )))
    }
}

struct EchoModel {
    chunk: bool,
    fail_recoverable: bool,
    has_batch: bool,
    predict_calls: AtomicUsize,
    chunk_calls: AtomicUsize,
    batch_calls: AtomicUsize,
    episodes_seen: std::sync::Mutex<Vec<(String, Option<i64>)>>,
}

const CHUNK_FRAMES: usize = 6;

impl EchoModel {
    fn new(chunk: bool, fail_recoverable: bool) -> Arc<Self> {
        Arc::new(Self {
            chunk,
            fail_recoverable,
            has_batch: true,
            predict_calls: AtomicUsize::new(0),
            chunk_calls: AtomicUsize::new(0),
            batch_calls: AtomicUsize::new(0),
            episodes_seen: std::sync::Mutex::new(Vec::new()),
        })
    }

    fn new_single_lane() -> Arc<Self> {
        Arc::new(Self {
            chunk: false,
            fail_recoverable: false,
            has_batch: false,
            predict_calls: AtomicUsize::new(0),
            chunk_calls: AtomicUsize::new(0),
            batch_calls: AtomicUsize::new(0),
            episodes_seen: std::sync::Mutex::new(Vec::new()),
        })
    }
}

fn state_number(input: &Value) -> f64 {
    match input {
        Value::Number(n) => *n,
        Value::List(items) => items.first().map(state_number).unwrap_or(0.0),
        Value::Map(map) => map.values().next().map(state_number).unwrap_or(0.0),
        _ => 0.0,
    }
}

fn action_value(state: f64) -> Value {
    Value::List(vec![Value::Number(state)])
}

impl PredictFn for EchoModel {
    fn predict(&self, model_input: Value, episode: Option<&EpisodeInfo>) -> Result<Value> {
        self.predict_calls.fetch_add(1, Ordering::SeqCst);
        if let Some(episode) = episode {
            self.episodes_seen
                .lock()
                .expect("episodes_seen poisoned")
                .push((episode.episode_id.clone(), episode.seed));
        }
        Ok(action_value(state_number(&model_input)))
    }

    fn predict_spec_less(&self, _observation: ModelObservation) -> Result<Vec<SpaceValue>> {
        Err(Error::model("EchoModel serves spec'd routes only"))
    }

    fn has_chunk(&self) -> bool {
        self.chunk
    }

    fn predict_chunk(
        &self,
        model_input: Value,
        _horizon: u32,
        _episode: Option<&EpisodeInfo>,
    ) -> Result<Option<Value>> {
        self.chunk_calls.fetch_add(1, Ordering::SeqCst);
        let state = state_number(&model_input);
        Ok(Some(Value::List(
            (0..CHUNK_FRAMES)
                .map(|frame| action_value(state + 0.125 * frame as f64))
                .collect(),
        )))
    }

    fn has_batch(&self) -> bool {
        self.has_batch
    }

    fn predict_batch(&self, inputs: Vec<Value>, episodes: &[EpisodeInfo]) -> Result<Vec<Value>> {
        self.batch_calls.fetch_add(1, Ordering::SeqCst);
        let mut seen = self.episodes_seen.lock().expect("episodes_seen poisoned");
        seen.extend(
            episodes
                .iter()
                .map(|episode| (episode.episode_id.clone(), episode.seed)),
        );
        drop(seen);
        if self.fail_recoverable {
            return Err(Error::model_recoverable("transient forward failure"));
        }
        Ok(inputs
            .iter()
            .map(|input| action_value(state_number(input)))
            .collect())
    }

    fn allow_fusion(&self) -> bool {
        true
    }
}

fn obs_space() -> spaces::SpaceSpec {
    spaces::spaces::DictSpaceBuilder::new()
        .insert(
            "state",
            spaces::spaces::BoxSpaceBuilder::scalar(0.0, 10.0, vec![1])
                .dtype(DType::Float32)
                .build()
                .unwrap(),
        )
        .build()
        .unwrap()
}

fn action_space() -> spaces::SpaceSpec {
    spaces::spaces::BoxSpaceBuilder::scalar(0.0, 10.0, vec![1])
        .dtype(DType::Float32)
        .build()
        .unwrap()
}

fn contract(env_id: &str, num_envs: u32) -> spaces::EnvContract {
    spaces::EnvContract {
        id: env_id.to_string(),
        observation_space: Some(obs_space()),
        action_space: Some(action_space()),
        metadata: None,
        render_mode: String::new(),
        num_envs,
        autoreset_mode: Default::default(),
    }
}

fn box_f32(value: f32) -> SpaceValue {
    SpaceValue::Box(
        Tensor::from_vec(value.to_le_bytes().to_vec(), vec![1], DType::Float32).unwrap(),
    )
}

fn grouped_obs(
    env_id: &str,
    values: &[f32],
    env_contract: &Arc<spaces::EnvContract>,
) -> ModelObservation {
    let lanes: Vec<SpaceValue> = values
        .iter()
        .map(|value| {
            SpaceValue::Dict(BTreeMap::from([(
                "state".to_string(),
                box_f32(*value).clone(),
            )]))
        })
        .collect();
    let wire = crate::data::encode_batched_partial_values(&lanes, &obs_space()).unwrap();
    ModelObservation {
        history: Vec::new(),
        step: None,
        observation: Some(wire.leaves),
        route: ModelRouteContext {
            env_id: env_id.to_string(),
            episodes: values
                .iter()
                .map(|value| EpisodeInfo {
                    episode_id: format!("{env_id}-ep-{value}"),
                    seed: None,
                    env_id: String::new(),
                })
                .collect(),
            ..Default::default()
        },
        num_envs: values.len(),
        env_contract: Some(Arc::clone(env_contract)),
    }
}

fn single_obs_with_episode(
    env_id: &str,
    value: f32,
    episode_id: &str,
    seed: Option<i64>,
    env_contract: &Arc<spaces::EnvContract>,
) -> ModelObservation {
    let lane = SpaceValue::Dict(BTreeMap::from([("state".to_string(), box_f32(value))]));
    let wire = crate::data::encode_batched_partial_values(&[lane], &obs_space()).unwrap();
    ModelObservation {
        history: Vec::new(),
        step: None,
        observation: Some(wire.leaves),
        route: ModelRouteContext {
            env_id: env_id.to_string(),
            episodes: vec![EpisodeInfo {
                episode_id: episode_id.to_string(),
                seed,
                env_id: String::new(),
            }],
            ..Default::default()
        },
        num_envs: 1,
        env_contract: Some(Arc::clone(env_contract)),
    }
}

async fn spec_handler(
    predict: Arc<dyn PredictFn>,
    envs: &[(&str, u32)],
    horizon: u32,
) -> (AdaptedModelHandler, Vec<Arc<spaces::EnvContract>>) {
    let handler = AdaptedModelHandler::new(predict, Some(Arc::new(TagResolver)));
    let setup = handler.route_setup().expect("resolver-backed route setup");
    let mut contracts = Vec::with_capacity(envs.len());
    for (env_id, lanes) in envs {
        let env_contract = contract(env_id, *lanes);
        setup
            .resolve_adapter(
                env_id,
                &env_contract,
                ResolveOptions {
                    execution_horizon: horizon,
                    delivers_history: false,
                },
            )
            .await
            .expect("route resolves");
        contracts.push(Arc::new(env_contract));
    }
    (handler, contracts)
}

#[tokio::test]
async fn direct_predict_carries_real_episode_id_and_seed() {
    let echo = EchoModel::new_single_lane();
    let (mut handler, contracts) =
        spec_handler(Arc::clone(&echo) as Arc<dyn PredictFn>, &[("env-a", 1)], 1).await;

    handler
        .predict(single_obs_with_episode(
            "env-a",
            1.0,
            "ep-linear",
            Some(42),
            &contracts[0],
        ))
        .await
        .expect("predict succeeds");

    assert_eq!(
        echo.episodes_seen.lock().expect("poisoned").as_slice(),
        [("ep-linear".to_string(), Some(42))],
    );
}

#[tokio::test]
async fn grouped_predict_concatenates_row_aligned_episode_identity() {
    let echo = EchoModel::new(false, false);
    let (mut handler, contracts) = spec_handler(
        Arc::clone(&echo) as Arc<dyn PredictFn>,
        &[("env-a", 1), ("env-b", 1)],
        1,
    )
    .await;

    let results = handler
        .predict_grouped(vec![
            grouped_obs("env-a", &[1.0], &contracts[0]),
            grouped_obs("env-b", &[2.0], &contracts[1]),
        ])
        .await;

    assert!(results.iter().all(Result::is_ok));
    assert_eq!(echo.batch_calls.load(Ordering::SeqCst), 1, "fusion ran");
    assert_eq!(
        echo.predict_calls.load(Ordering::SeqCst),
        0,
        "the per-lane predict() corner never runs when fused"
    );
    assert_eq!(
        echo.episodes_seen.lock().expect("poisoned").as_slice(),
        [
            ("env-a-ep-1".to_string(), None),
            ("env-b-ep-2".to_string(), None),
        ],
        "rows arrive in group order, aligned with the concatenated inputs"
    );
}

#[tokio::test]
async fn grouped_non_fused_predict_carries_each_routes_episode_identity() {
    let echo = EchoModel::new_single_lane();
    let (mut handler, contracts) = spec_handler(
        Arc::clone(&echo) as Arc<dyn PredictFn>,
        &[("env-a", 1), ("env-b", 1)],
        1,
    )
    .await;

    let results = handler
        .predict_grouped(vec![
            single_obs_with_episode("env-a", 1.0, "ep-a", Some(7), &contracts[0]),
            single_obs_with_episode("env-b", 2.0, "ep-b", None, &contracts[1]),
        ])
        .await;

    assert!(results.iter().all(Result::is_ok));
    assert_eq!(echo.batch_calls.load(Ordering::SeqCst), 0, "must not fuse");
    assert_eq!(
        echo.episodes_seen.lock().expect("poisoned").as_slice(),
        [("ep-a".to_string(), Some(7)), ("ep-b".to_string(), None)],
    );
}

#[tokio::test]
async fn fused_grouped_predict_runs_one_forward_and_splits_actions_per_group() {
    let echo = EchoModel::new(false, false);
    let (mut handler, contracts) = spec_handler(
        Arc::clone(&echo) as Arc<dyn PredictFn>,
        &[("env-a", 2), ("env-b", 3)],
        1,
    )
    .await;
    let batch_baseline = echo.batch_calls.load(Ordering::SeqCst);

    let results = handler
        .predict_grouped(vec![
            grouped_obs("env-a", &[1.0, 2.0], &contracts[0]),
            grouped_obs("env-b", &[3.0, 4.0, 5.0], &contracts[1]),
        ])
        .await;

    assert_eq!(
        echo.batch_calls.load(Ordering::SeqCst) - batch_baseline,
        1,
        "the whole group rides ONE fused forward"
    );
    assert_eq!(results.len(), 2);
    let a = results[0].as_ref().expect("env-a serves");
    assert_eq!(
        a.actions,
        vec![box_f32(1.0), box_f32(2.0)],
        "env-a gets its own lanes back"
    );
    assert!(a.replay.is_empty());
    let b = results[1].as_ref().expect("env-b serves");
    assert_eq!(
        b.actions,
        vec![box_f32(3.0), box_f32(4.0), box_f32(5.0)],
        "env-b gets its own lanes back"
    );
    assert!(b.replay.is_empty());
}

#[tokio::test]
async fn grouped_chunk_and_batch_model_keeps_chunking_when_grouped() {
    let echo = EchoModel::new(true, false);
    let (mut handler, contracts) = spec_handler(
        Arc::clone(&echo) as Arc<dyn PredictFn>,
        &[("env-a", 2), ("env-b", 3)],
        4,
    )
    .await;
    let batch_baseline = echo.batch_calls.load(Ordering::SeqCst);
    let chunk_baseline = echo.chunk_calls.load(Ordering::SeqCst);

    let results = handler
        .predict_grouped(vec![
            grouped_obs("env-a", &[1.0, 2.0], &contracts[0]),
            grouped_obs("env-b", &[3.0, 4.0, 5.0], &contracts[1]),
        ])
        .await;

    assert_eq!(
        echo.batch_calls.load(Ordering::SeqCst) - batch_baseline,
        0,
        "the parity gate declines fusion: batching would drop chunking"
    );
    assert_eq!(
        echo.chunk_calls.load(Ordering::SeqCst) - chunk_baseline,
        5,
        "every lane runs the per-lane chunk corner, exactly as ungrouped"
    );
    for (result, lane_states) in results.iter().zip([vec![1.0f32, 2.0], vec![3.0, 4.0, 5.0]]) {
        let frames = result.as_ref().expect("group serves chunked");
        let frame0: Vec<SpaceValue> = lane_states.iter().map(|s| box_f32(*s)).collect();
        assert_eq!(frames.actions, frame0);
        assert_eq!(frames.replay.len(), 3, "horizon 4 = frame 0 + 3 replays");
        for (step, row) in frames.replay.iter().enumerate() {
            let expected: Vec<SpaceValue> = lane_states
                .iter()
                .map(|s| box_f32(s + 0.125 * (step as f32 + 1.0)))
                .collect();
            assert_eq!(row, &expected, "replay step {step} keeps lane order");
        }
    }
}

#[tokio::test]
async fn wide_grouped_predict_keeps_every_group_in_request_order() {
    for chunk in [false, true] {
        let echo = EchoModel::new(chunk, false);
        let names: Vec<String> = (0..48).map(|i| format!("env-{i}")).collect();
        let specs: Vec<(&str, u32)> = names.iter().map(|n| (n.as_str(), 1)).collect();
        let horizon = if chunk { 3 } else { 1 };
        let (mut handler, contracts) =
            spec_handler(Arc::clone(&echo) as Arc<dyn PredictFn>, &specs, horizon).await;
        for round in 0..5 {
            let offset = 100.0 * round as f32;
            let observations = (0..48)
                .map(|i| grouped_obs(&names[i], &[offset + i as f32], &contracts[i]))
                .collect();
            let results = handler.predict_grouped(observations).await;
            assert_eq!(results.len(), 48);
            for (i, result) in results.iter().enumerate() {
                let frames = result.as_ref().expect("every group serves");
                let state = offset + i as f32;
                assert_eq!(frames.actions, vec![box_f32(state)], "group {i}");
                if chunk {
                    assert_eq!(frames.replay.len(), 2, "group {i} replays horizon-1 frames");
                    for (k, frame) in frames.replay.iter().enumerate() {
                        assert_eq!(
                            frame,
                            &vec![box_f32((f64::from(state) + 0.125 * (k + 1) as f64) as f32)]
                        );
                    }
                }
            }
        }
        assert!(
            echo.batch_calls.load(Ordering::SeqCst) + echo.chunk_calls.load(Ordering::SeqCst) > 0
        );
    }
}

const IMAGE_ENV_TAGS: &str = r#"{
        "observation": {"cam": {"type": "image", "role": "image/primary", "upside_down": true},
                        "state": {"type": "state", "role": "proprio/gripper"}},
        "action": {"components": [{"role": "action/gripper", "dim": 1, "range": [0.0, 10.0]}]}
    }"#;
const IMAGE_MODEL_SPEC: &str = r#"{
        "input": {"cam": {"type": "image", "role": "image/primary", "height": 256, "width": 256, "fit": "pad",
                          "layout": "chw", "dtype": "float32", "normalize": true},
                  "state": {"type": "state", "container": "list", "dtype": "float32",
                            "components": [{"role": "proprio/gripper", "dim": 1}]}},
        "output": {"components": [{"role": "action/gripper", "dim": 1, "range": [0.0, 10.0]}]}
    }"#;

struct ImageTagResolver;

#[async_trait]
impl RouteResolver for ImageTagResolver {
    async fn resolve(
        &self,
        _route_key: &str,
        env_contract: &EnvContract,
    ) -> Result<Option<RouteConfig>> {
        let tags: EnvTags = serde_json::from_str(IMAGE_ENV_TAGS).expect("image env tags parse");
        let spec: ModelSpec =
            serde_json::from_str(IMAGE_MODEL_SPEC).expect("image model spec parse");
        let obs = env_contract
            .observation_space
            .clone()
            .expect("contract obs space");
        let action = env_contract
            .action_space
            .clone()
            .expect("contract action space");
        let adapter = resolve(
            &tags,
            &SpaceView::from(&obs),
            &SpaceView::from(&action),
            &spec,
            true,
        )
        .map_err(|err| Error::model(err.message))?;
        Ok(Some(RouteConfig::new(
            adapter,
            obs,
            action,
            Box::new(NoCustoms),
            Box::new(NoEncodings),
        )))
    }
}

fn image_obs_space() -> spaces::SpaceSpec {
    spaces::spaces::DictSpaceBuilder::new()
        .insert(
            "cam",
            spaces::spaces::BoxSpaceBuilder::scalar(0.0, 255.0, vec![256, 256, 3])
                .dtype(DType::Uint8)
                .build()
                .unwrap(),
        )
        .insert(
            "state",
            spaces::spaces::BoxSpaceBuilder::scalar(0.0, 10.0, vec![1])
                .dtype(DType::Float32)
                .build()
                .unwrap(),
        )
        .build()
        .unwrap()
}

fn image_contract(env_id: &str) -> spaces::EnvContract {
    spaces::EnvContract {
        id: env_id.to_string(),
        observation_space: Some(image_obs_space()),
        action_space: Some(action_space()),
        metadata: None,
        render_mode: String::new(),
        num_envs: 1,
        autoreset_mode: Default::default(),
    }
}

fn grouped_image_obs(env_id: &str, seed: u32, value: f32) -> ModelObservation {
    let mut state = seed | 1;
    let pixels: Vec<u8> = (0..256 * 256 * 3)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state & 0xff) as u8
        })
        .collect();
    let lane = SpaceValue::Dict(BTreeMap::from([
        (
            "cam".to_string(),
            SpaceValue::Box(Tensor::from_vec(pixels, vec![256, 256, 3], DType::Uint8).unwrap()),
        ),
        ("state".to_string(), box_f32(value)),
    ]));
    let wire = crate::data::encode_batched_partial_values(&[lane], &image_obs_space()).unwrap();
    ModelObservation {
        history: Vec::new(),
        step: None,
        observation: Some(wire.leaves),
        route: ModelRouteContext {
            session_id: "session-1".to_string(),
            request_id: format!("req-{env_id}-{seed}"),
            env_id: env_id.to_string(),
            episodes: vec![EpisodeInfo {
                episode_id: format!("ep-{env_id}"),
                seed: None,
                env_id: String::new(),
            }],
        },
        num_envs: 1,
        env_contract: Some(Arc::new(image_contract(env_id))),
    }
}

#[tokio::test]
#[ignore = "timing, not correctness: run with --release --nocapture"]
#[allow(clippy::print_stdout, reason = "the point of the test is the report")]
async fn camera_grouped_predict_timing() {
    let echo = EchoModel::new(true, false);
    let handler_predict = Arc::clone(&echo) as Arc<dyn PredictFn>;
    let mut handler = AdaptedModelHandler::new(handler_predict, Some(Arc::new(ImageTagResolver)));
    let setup = handler.route_setup().expect("resolver-backed route setup");
    let names: Vec<String> = (0..32).map(|i| format!("cam-{i}")).collect();
    for name in &names {
        setup
            .resolve_adapter(
                name,
                &image_contract(name),
                ResolveOptions {
                    execution_horizon: 10,
                    delivers_history: false,
                },
            )
            .await
            .expect("camera route resolves");
    }
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    let pool = rayon::current_num_threads();
    for round in 0..6 {
        let observations: Vec<ModelObservation> = (0..32)
            .map(|i| grouped_image_obs(&names[i], 1000 * round + i as u32 + 1, i as f32))
            .collect();
        let started = std::time::Instant::now();
        let results = handler.predict_grouped(observations).await;
        let wall = started.elapsed();
        let adapter = handler.take_adapter_ns();
        assert!(
            results.iter().all(Result::is_ok),
            "every camera group serves"
        );
        println!(
            "round {round}: 32 camera groups in {:.1} ms wall, adapter span {:.1} ms (available_parallelism {threads}, rayon pool {pool})",
            wall.as_secs_f64() * 1e3,
            adapter as f64 / 1e6
        );
    }
}

#[tokio::test]
async fn grouped_predict_with_duplicate_env_completes() {
    let echo = EchoModel::new(false, false);
    let (mut handler, contracts) =
        spec_handler(Arc::clone(&echo) as Arc<dyn PredictFn>, &[("env-a", 2)], 1).await;

    let results = tokio::time::timeout(
        Duration::from_secs(10),
        handler.predict_grouped(vec![
            grouped_obs("env-a", &[1.0, 2.0], &contracts[0]),
            grouped_obs("env-a", &[3.0, 4.0], &contracts[0]),
        ]),
    )
    .await
    .expect("a grouped predict repeating an env must not deadlock");

    assert_eq!(results.len(), 2);
    let first = results[0].as_ref().expect("first duplicate serves");
    assert_eq!(first.actions, vec![box_f32(1.0), box_f32(2.0)]);
    let second = results[1].as_ref().expect("second duplicate serves");
    assert_eq!(second.actions, vec![box_f32(3.0), box_f32(4.0)]);
}

#[tokio::test]
async fn fused_failure_reports_recoverable_error_with_each_groups_own_signature() {
    let echo = EchoModel::new(false, true);
    let (mut handler, contracts) = spec_handler(
        Arc::clone(&echo) as Arc<dyn PredictFn>,
        &[("env-a", 2), ("env-b", 3)],
        1,
    )
    .await;

    let results = handler
        .predict_grouped(vec![
            grouped_obs("env-a", &[1.0, 2.0], &contracts[0]),
            grouped_obs("env-b", &[3.0, 4.0, 5.0], &contracts[1]),
        ])
        .await;

    for (result, lanes) in results.iter().zip([2usize, 3]) {
        let error = result
            .as_ref()
            .expect_err("fused failure reaches the group");
        assert!(
            error.is_recoverable(),
            "the model's recoverable flag survives the fused broadcast: {error}"
        );
        assert!(
            error.to_string().contains(&format!("lanes: {lanes}")),
            "each group is annotated with its OWN input signature, got: {error}"
        );
    }
}

struct DeclaredChunkModel {
    emit: usize,
    declared: Option<u32>,
    has_chunk: bool,
}

impl DeclaredChunkModel {
    fn new(emit: usize, declared: Option<u32>) -> Arc<Self> {
        Arc::new(Self {
            emit,
            declared,
            has_chunk: true,
        })
    }

    fn cornerless(declared: u32) -> Arc<Self> {
        Arc::new(Self {
            emit: 0,
            declared: Some(declared),
            has_chunk: false,
        })
    }
}

impl PredictFn for DeclaredChunkModel {
    fn predict(&self, model_input: Value, _episode: Option<&EpisodeInfo>) -> Result<Value> {
        Ok(action_value(state_number(&model_input)))
    }

    fn predict_spec_less(&self, _observation: ModelObservation) -> Result<Vec<SpaceValue>> {
        Err(Error::model("DeclaredChunkModel serves spec'd routes only"))
    }

    fn has_chunk(&self) -> bool {
        self.has_chunk
    }

    fn predict_chunk(
        &self,
        model_input: Value,
        _horizon: u32,
        _episode: Option<&EpisodeInfo>,
    ) -> Result<Option<Value>> {
        let state = state_number(&model_input);
        Ok(Some(Value::List(
            (0..self.emit)
                .map(|frame| action_value(state + 0.125 * frame as f64))
                .collect(),
        )))
    }

    fn native_chunk(&self) -> Option<u32> {
        self.declared
    }
}

struct SpecLessResolver;

#[async_trait]
impl RouteResolver for SpecLessResolver {
    async fn resolve(
        &self,
        _route_key: &str,
        _env_contract: &EnvContract,
    ) -> Result<Option<RouteConfig>> {
        Ok(None)
    }
}

mod spec_less_batching {
    use super::*;
    use crate::model::BatchPredictContext;
    use crate::model::bridge::{engine_handler, from_adapter, to_adapter};
    use crate::spaces::{SpaceKind, SpaceSpec};
    use crate::{Corners, Model, ModelOptions, Policy, Value as RawValue};

    struct MixedResolver(Arc<dyn RouteResolver>);

    #[async_trait]
    impl RouteResolver for MixedResolver {
        async fn resolve(&self, key: &str, contract: &EnvContract) -> Result<Option<RouteConfig>> {
            self.0.resolve(key, contract).await?;
            if key == "routed" {
                TagResolver.resolve(key, contract).await
            } else {
                Ok(None)
            }
        }
    }

    #[derive(Debug, PartialEq)]
    enum Call {
        Single(RawValue),
        Batch(Vec<RawValue>, Vec<(String, Option<SpaceKind>)>),
        Chunk(RawValue, u32),
    }

    #[derive(Clone, Copy)]
    enum BatchOutput {
        Actions,
        Rejected,
        Short,
        Long,
    }

    struct RecordingPolicy<const BATCH: bool> {
        calls: Arc<Mutex<Vec<Call>>>,
        allow_fusion: bool,
        batch_output: BatchOutput,
    }

    fn text(value: &str) -> RawValue {
        RawValue::Text(value.to_string())
    }

    fn action(input: &RawValue) -> Result<RawValue> {
        match input {
            RawValue::Text(value) if value == "5" => Ok(RawValue::Int(6)),
            RawValue::Text(value) => Ok(text(&format!("{value}-action"))),
            _ => from_adapter(action_value(state_number(&to_adapter(input)?))),
        }
    }

    impl<const BATCH: bool> Policy for RecordingPolicy<BATCH> {
        const CORNERS: Corners = Corners::PREDICT
            .union(Corners::PREDICT_CHUNK)
            .union(if BATCH {
                Corners::PREDICT_BATCH
            } else {
                Corners::PREDICT_CHUNK_BATCH
            });
        const CONTEXT: Corners = if BATCH {
            Corners::PREDICT_BATCH
        } else {
            Corners::NONE
        };

        fn allow_fusion(&self) -> bool {
            self.allow_fusion
        }

        fn predict(&mut self, input: RawValue) -> Result<RawValue> {
            let output = action(&input);
            self.calls.lock().unwrap().push(Call::Single(input));
            output
        }

        fn predict_chunk(&mut self, input: RawValue, horizon: u32) -> Result<RawValue> {
            let output = RawValue::List(vec![action(&input)?, text("replay")]);
            self.calls.lock().unwrap().push(Call::Chunk(input, horizon));
            Ok(output)
        }

        fn predict_chunk_batch(&mut self, _: Vec<RawValue>, _: u32) -> Result<Vec<RawValue>> {
            Err(Error::model(
                "unexpected chunk-batch call on a spec-less route",
            ))
        }

        fn predict_batch_with_context(
            &mut self,
            inputs: Vec<RawValue>,
            context: &mut BatchPredictContext<'_>,
        ) -> Result<Vec<RawValue>> {
            let mut actions = inputs.iter().map(action).collect::<Result<Vec<_>>>()?;
            let spans_spec_less_routes =
                inputs.contains(&text("alpha")) && inputs.contains(&text("5"));
            let rows = context
                .iter()
                .map(|row| {
                    (
                        row.episode_id.to_string(),
                        row.action_space.map(SpaceSpec::kind),
                    )
                })
                .collect();
            self.calls.lock().unwrap().push(Call::Batch(inputs, rows));
            if spans_spec_less_routes {
                match self.batch_output {
                    BatchOutput::Actions => (),
                    BatchOutput::Rejected => return Err(Error::model("policy rejected the batch")),
                    BatchOutput::Short => {
                        actions.pop();
                    }
                    BatchOutput::Long => actions.push(RawValue::Int(7)),
                }
            }
            Ok(actions)
        }
    }

    fn text_space() -> spaces::SpaceSpec {
        spaces::spaces::TextBuilder::new(32).build().unwrap()
    }

    fn text_observation(
        env: &str,
        inputs: &[&str],
        actions: spaces::SpaceSpec,
    ) -> ModelObservation {
        let mut env_contract = contract(env, inputs.len() as u32);
        env_contract.observation_space = Some(text_space());
        env_contract.action_space = Some(actions);
        let lanes: Vec<_> = inputs
            .iter()
            .map(|value| SpaceValue::Text((*value).into()))
            .collect();
        let wire = crate::data::encode_batched_partial_values(&lanes, &text_space()).unwrap();
        ModelObservation {
            observation: Some(wire.leaves),
            route: ModelRouteContext {
                env_id: env.into(),
                episodes: inputs
                    .iter()
                    .map(|value| EpisodeInfo {
                        episode_id: format!("{env}-{value}"),
                        env_id: env.into(),
                        seed: None,
                    })
                    .collect(),
                ..Default::default()
            },
            num_envs: lanes.len(),
            env_contract: Some(Arc::new(env_contract)),
            ..Default::default()
        }
    }

    fn observations() -> [ModelObservation; 3] {
        [
            text_observation("text", &["beta", "alpha"], text_space()),
            grouped_obs("routed", &[4.0], &Arc::new(contract("routed", 1))),
            text_observation(
                "discrete",
                &["5"],
                spaces::spaces::DiscreteBuilder::new(8).build().unwrap(),
            ),
        ]
    }

    async fn grouped<const BATCH: bool>(
        observations: Vec<ModelObservation>,
        horizon: u32,
        allow_fusion: bool,
        batch_output: BatchOutput,
    ) -> (Vec<Result<PredictFrames>>, Vec<Call>) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let model = Model::new(
            RecordingPolicy::<BATCH> {
                calls: Arc::clone(&calls),
                allow_fusion,
                batch_output,
            },
            ModelOptions::default(),
        )
        .unwrap();
        let mut handler = engine_handler(Arc::clone(model.shared()));
        handler.resolver = Some(Arc::new(MixedResolver(handler.resolver.take().unwrap())));
        let setup = handler.route_setup().unwrap();
        for observation in &observations {
            setup
                .resolve_adapter(
                    &observation.route.env_id,
                    observation.env_contract.as_ref().unwrap(),
                    ResolveOptions {
                        execution_horizon: horizon,
                        delivers_history: false,
                    },
                )
                .await
                .unwrap();
        }
        let results = handler.predict_grouped(observations).await;
        model.close_async().await.unwrap();
        let calls = std::mem::take(&mut *calls.lock().unwrap());
        (results, calls)
    }

    fn spec_less_calls(calls: Vec<Call>) -> Vec<Call> {
        calls
            .into_iter()
            .filter(|call| match call {
                Call::Batch(inputs, _) => inputs
                    .iter()
                    .any(|input| !matches!(input, RawValue::Dict(_))),
                _ => true,
            })
            .collect()
    }

    fn expected_spec_less_batch() -> Call {
        Call::Batch(
            vec![text("beta"), text("alpha"), text("5")],
            vec![
                ("text-beta".into(), Some(SpaceKind::Text)),
                ("text-alpha".into(), Some(SpaceKind::Text)),
                ("discrete-5".into(), Some(SpaceKind::Discrete)),
            ],
        )
    }

    fn assert_actions(result: &Result<PredictFrames>, actions: &[SpaceValue]) {
        let frames = result.as_ref().unwrap();
        assert_eq!(frames.actions, actions);
        assert!(frames.replay.is_empty());
    }

    #[tokio::test]
    async fn spec_less_grouped_predict_fuses_rows_and_splits_actions_by_route() {
        let (results, calls) =
            grouped::<true>(observations().into(), 1, true, BatchOutput::Actions).await;
        let [left, routed, right] = results.as_slice() else {
            panic!("expected one result per observation: {results:?}")
        };
        assert_actions(
            left,
            &[
                SpaceValue::Text("beta-action".into()),
                SpaceValue::Text("alpha-action".into()),
            ],
        );
        assert_actions(routed, &[box_f32(4.0)]);
        assert_actions(right, &[SpaceValue::Discrete(6)]);
        assert_eq!(spec_less_calls(calls), [expected_spec_less_batch()]);
    }

    #[tokio::test]
    async fn spec_less_batch_failure_reaches_every_observation_and_preserves_routed_results() {
        let mut failures = Vec::new();
        for (output, reason) in [
            (BatchOutput::Rejected, &["policy rejected the batch"][..]),
            (
                BatchOutput::Short,
                &["predict_batch", "actions", "2", "3"][..],
            ),
            (
                BatchOutput::Long,
                &["predict_batch", "actions", "4", "3"][..],
            ),
        ] {
            let (results, _) = grouped::<true>(observations().into(), 1, true, output).await;
            let [left, routed, right] = results.as_slice() else {
                panic!("expected one result per observation: {results:?}")
            };
            assert_actions(routed, &[box_f32(4.0)]);
            for result in [left, right] {
                if !result.as_ref().is_err_and(|error| {
                    let message = error.to_string();
                    reason.iter().all(|&part| message.contains(part))
                }) {
                    failures.push(format!(
                        "expected an error containing {reason:?}, got {result:?}"
                    ));
                }
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    #[tokio::test]
    async fn spec_less_batch_excludes_an_observation_missing_space_metadata() {
        let [left, _, right] = observations();
        let mut invalid = text_observation("invalid", &["ignored"], text_space());
        Arc::make_mut(invalid.env_contract.as_mut().unwrap()).action_space = None;
        let (results, calls) =
            grouped::<true>(vec![left, invalid, right], 1, true, BatchOutput::Actions).await;
        let [left, invalid, right] = results.as_slice() else {
            panic!("expected one result per observation: {results:?}")
        };
        assert_actions(
            left,
            &[
                SpaceValue::Text("beta-action".into()),
                SpaceValue::Text("alpha-action".into()),
            ],
        );
        let error = invalid.as_ref().unwrap_err().to_string();
        assert!(error.contains("action space"), "{error}");
        assert_actions(right, &[SpaceValue::Discrete(6)]);
        assert_eq!(calls, [expected_spec_less_batch()]);
    }

    #[tokio::test]
    async fn spec_less_grouped_predict_preserves_per_observation_corners_when_ineligible() {
        let one = text_observation("first", &["alpha"], text_space());
        let two = text_observation("second", &["beta"], text_space());
        let group = vec![one.clone(), two];
        for (results, calls) in [
            grouped::<false>(group.clone(), 1, true, BatchOutput::Actions).await,
            grouped::<true>(group.clone(), 1, false, BatchOutput::Actions).await,
        ] {
            assert_eq!(
                calls,
                [Call::Single(text("alpha")), Call::Single(text("beta"))]
            );
            assert_eq!(results.len(), 2);
            assert_actions(&results[0], &[SpaceValue::Text("alpha-action".into())]);
            assert_actions(&results[1], &[SpaceValue::Text("beta-action".into())]);
        }

        let (results, calls) = grouped::<true>(vec![one], 1, true, BatchOutput::Actions).await;
        assert_eq!(calls, [Call::Single(text("alpha"))]);
        assert_eq!(results.len(), 1);
        assert_actions(&results[0], &[SpaceValue::Text("alpha-action".into())]);

        let (results, calls) = grouped::<true>(group, 2, true, BatchOutput::Actions).await;
        assert_eq!(
            calls,
            [Call::Chunk(text("alpha"), 2), Call::Chunk(text("beta"), 2)]
        );
        assert_eq!(results.len(), 2);
        for (result, expected) in results.iter().zip(["alpha-action", "beta-action"]) {
            let frames = result.as_ref().unwrap();
            assert_eq!(frames.actions, [SpaceValue::Text(expected.into())]);
            assert_eq!(frames.replay, [vec![SpaceValue::Text("replay".into())]]);
        }
    }
}

async fn resolve_once(
    predict: Arc<dyn PredictFn>,
    resolver: Arc<dyn RouteResolver>,
    lanes: u32,
    horizon: u32,
) -> Result<RouteNeeds> {
    let handler = AdaptedModelHandler::new(predict, Some(resolver));
    let setup = handler.route_setup().expect("resolver-backed route setup");
    let env_contract = contract("env-h", lanes);
    setup
        .resolve_adapter(
            "env-h",
            &env_contract,
            ResolveOptions {
                execution_horizon: horizon,
                delivers_history: false,
            },
        )
        .await
}

async fn frame_count(handler: &mut AdaptedModelHandler, obs: ModelObservation) -> usize {
    let frames = handler
        .predict_chunked(obs)
        .await
        .expect("predict succeeds");
    1 + frames.replay.len()
}

#[tokio::test]
async fn resolve_refuses_a_horizon_over_the_bound() {
    let error = resolve_once(
        DeclaredChunkModel::new(4, None),
        Arc::new(TagResolver),
        1,
        MAX_EXECUTION_HORIZON + 1,
    )
    .await
    .expect_err("over the bound");
    assert!(
        error
            .to_string()
            .contains(&MAX_EXECUTION_HORIZON.to_string()),
        "the bound should name itself: {error}"
    );
}

#[tokio::test]
async fn resolve_accepts_the_horizon_bound_exactly() {
    resolve_once(
        DeclaredChunkModel::new(4, None),
        Arc::new(TagResolver),
        1,
        MAX_EXECUTION_HORIZON,
    )
    .await
    .expect("the bound itself resolves");
}

#[tokio::test]
async fn the_horizon_doors_guard_the_spec_less_branch_too() {
    resolve_once(
        DeclaredChunkModel::new(4, Some(6)),
        Arc::new(SpecLessResolver),
        1,
        8,
    )
    .await
    .expect_err("a spec-less route is bound by the same doors");
}

#[tokio::test]
async fn resolve_refuses_a_declared_chunk_with_no_chunk_corner() {
    let error = resolve_once(
        DeclaredChunkModel::cornerless(6),
        Arc::new(TagResolver),
        1,
        1,
    )
    .await
    .expect_err("declared K, no corner");
    assert!(
        error.to_string().contains("no chunk corner"),
        "unexpected error: {error}"
    );
}

#[tokio::test]
async fn resolve_refuses_a_horizon_over_the_declared_chunk() {
    let error = resolve_once(
        DeclaredChunkModel::new(6, Some(6)),
        Arc::new(TagResolver),
        1,
        8,
    )
    .await
    .expect_err("h > K");
    assert!(
        error.to_string().contains("native_chunk=6"),
        "unexpected error: {error}"
    );
}

#[tokio::test]
async fn resolve_answers_the_declared_native_chunk() {
    let needs = resolve_once(
        DeclaredChunkModel::new(6, Some(6)),
        Arc::new(TagResolver),
        1,
        6,
    )
    .await
    .expect("h == K resolves");
    assert_eq!(needs.native_chunk, Some(6));
}

#[tokio::test]
async fn resolve_answers_none_when_undeclared() {
    let needs = resolve_once(
        DeclaredChunkModel::new(6, None),
        Arc::new(TagResolver),
        1,
        4,
    )
    .await
    .expect("undeclared resolves");
    assert_eq!(needs.native_chunk, None);
}

#[tokio::test]
async fn served_resolve_sees_no_lanes() {
    resolve_once(
        DeclaredChunkModel::new(6, Some(6)),
        Arc::new(TagResolver),
        4,
        4,
    )
    .await
    .expect("the engine carries no lane check");
}

#[tokio::test]
async fn a_declared_model_replays_the_horizon_prefix_of_its_whole_chunk() {
    let model = DeclaredChunkModel::new(6, Some(6));
    let (mut handler, contracts) =
        spec_handler(model as Arc<dyn PredictFn>, &[("env-a", 1)], 4).await;
    let obs = single_obs_with_episode("env-a", 1.0, "ep", None, &contracts[0]);
    assert_eq!(frame_count(&mut handler, obs).await, 4);
}

#[tokio::test]
async fn a_declared_model_at_the_full_horizon_replays_every_frame() {
    let model = DeclaredChunkModel::new(6, Some(6));
    let (mut handler, contracts) =
        spec_handler(model as Arc<dyn PredictFn>, &[("env-a", 1)], 6).await;
    let obs = single_obs_with_episode("env-a", 1.0, "ep", None, &contracts[0]);
    assert_eq!(frame_count(&mut handler, obs).await, 6);
}

#[tokio::test]
async fn a_declared_model_that_slices_its_own_chunk_fails_the_predict() {
    let model = DeclaredChunkModel::new(4, Some(6));
    let (mut handler, contracts) =
        spec_handler(model as Arc<dyn PredictFn>, &[("env-a", 1)], 4).await;
    let obs = single_obs_with_episode("env-a", 1.0, "ep", None, &contracts[0]);
    let error = handler
        .predict_chunked(obs)
        .await
        .expect_err("a declared K is exact");
    let message = error.to_string();
    assert!(
        message.contains("native_chunk=6") && message.contains("4"),
        "the error should name both lengths: {message}"
    );
}

#[tokio::test]
async fn an_undeclared_short_chunk_replays_what_there_is() {
    let model = DeclaredChunkModel::new(3, None);
    let (mut handler, contracts) =
        spec_handler(model as Arc<dyn PredictFn>, &[("env-a", 1)], 6).await;
    let obs = single_obs_with_episode("env-a", 1.0, "ep", None, &contracts[0]);
    assert_eq!(frame_count(&mut handler, obs).await, 3);
}

#[test]
fn the_short_chunk_warning_latches_once_per_episode() {
    let warned = ShortChunkWarned::default();
    let frames = || vec![Value::Number(0.0), Value::Number(1.0)];
    assert_eq!(
        take_prefix(frames(), 6, None, &warned, "ep")
            .expect("elastic")
            .len(),
        2
    );
    assert!(!warned.first("ep"), "the first short chunk warns");
    assert_eq!(
        take_prefix(frames(), 6, None, &warned, "ep")
            .expect("elastic")
            .len(),
        2,
        "a latched flag must not change what is replayed"
    );
    assert!(warned.first("other"), "another episode warns on its own");
}

#[test]
fn chunk_refusals_are_value_errors() {
    let warned = ShortChunkWarned::default();
    let error = take_prefix(vec![Value::Number(0.0)], 2, Some(3), &warned, "ep").unwrap_err();
    assert_eq!(error.class(), crate::ErrorClass::ValueError);
    let error = take_prefix(Vec::new(), 2, None, &warned, "ep").unwrap_err();
    assert_eq!(error.class(), crate::ErrorClass::ValueError);
}

const STACK_ENV_TAGS: &str = r#"{
        "observation": {"cam": {"type": "image", "role": "image/primary", "layout": "hwc"}},
        "action": {"components": [{"role": "action/gripper", "dim": 1, "range": [0.0, 10.0]}]}
    }"#;
const STACK_MODEL_SPEC: &str = r#"{
        "input": {"type": "image", "role": "image/primary", "layout": "hwc", "dtype": "uint8",
                  "stack": 3},
        "output": {"components": [{"role": "action/gripper", "dim": 1, "range": [0.0, 10.0]}]}
    }"#;

struct StackResolver;

#[async_trait]
impl RouteResolver for StackResolver {
    async fn resolve(
        &self,
        _route_key: &str,
        env_contract: &EnvContract,
    ) -> Result<Option<RouteConfig>> {
        let tags: EnvTags = serde_json::from_str(STACK_ENV_TAGS).expect("env tags parse");
        let spec: ModelSpec = serde_json::from_str(STACK_MODEL_SPEC).expect("spec parse");
        let obs = env_contract
            .observation_space
            .clone()
            .expect("contract obs space");
        let action = env_contract
            .action_space
            .clone()
            .expect("contract action space");
        let adapter = resolve(
            &tags,
            &SpaceView::from(&obs),
            &SpaceView::from(&action),
            &spec,
            true,
        )
        .map_err(|err| Error::model(err.message))?;
        Ok(Some(RouteConfig::new(
            adapter,
            obs,
            action,
            Box::new(NoCustoms),
            Box::new(NoEncodings),
        )))
    }
}

fn cam_obs_space() -> spaces::SpaceSpec {
    spaces::spaces::DictSpaceBuilder::new()
        .insert(
            "cam",
            spaces::spaces::BoxSpaceBuilder::scalar(0.0, 255.0, vec![1, 1, 3])
                .dtype(DType::Uint8)
                .build()
                .unwrap(),
        )
        .build()
        .unwrap()
}

fn cam_contract(env_id: &str) -> Arc<spaces::EnvContract> {
    Arc::new(spaces::EnvContract {
        id: env_id.to_string(),
        observation_space: Some(cam_obs_space()),
        action_space: Some(action_space()),
        metadata: None,
        render_mode: String::new(),
        num_envs: 1,
        autoreset_mode: Default::default(),
    })
}

fn cam_leaves(tag: u8) -> Vec<bytes::Bytes> {
    let lane = SpaceValue::Dict(BTreeMap::from([(
        "cam".to_string(),
        SpaceValue::Box(Tensor::from_vec(vec![tag; 3], vec![1, 1, 3], DType::Uint8).unwrap()),
    )]));
    crate::data::encode_batched_partial_values(&[lane], &cam_obs_space())
        .unwrap()
        .leaves
}

fn episode(id: &str) -> Vec<EpisodeInfo> {
    vec![EpisodeInfo {
        episode_id: id.to_string(),
        seed: None,
        env_id: String::new(),
    }]
}

fn cam_predict(
    env_id: &str,
    episode_id: &str,
    step: Option<i64>,
    tag: u8,
    rows: &[(i64, u8)],
) -> ModelObservation {
    ModelObservation {
        observation: Some(cam_leaves(tag)),
        route: ModelRouteContext {
            env_id: env_id.to_string(),
            episodes: episode(episode_id),
            ..Default::default()
        },
        num_envs: 1,
        env_contract: Some(cam_contract(env_id)),
        history: rows
            .iter()
            .map(|&(step, tag)| crate::model::types::HistoryFrame {
                observation: Some(cam_leaves(tag)),
                episodes: episode(episode_id),
                step,
            })
            .collect(),
        step,
    }
}

struct WindowRecorder {
    stacks: Mutex<Vec<Vec<u8>>>,
}

impl PredictFn for WindowRecorder {
    fn predict(&self, model_input: Value, _episode: Option<&EpisodeInfo>) -> Result<Value> {
        fn first_tensor(value: &Value) -> Option<Vec<u8>> {
            match value {
                Value::Tensor(tensor) => Some(tensor.to_contiguous_bytes().into_owned()),
                Value::Map(map) => map.values().find_map(first_tensor),
                Value::List(items) => items.iter().find_map(first_tensor),
                _ => None,
            }
        }
        self.stacks
            .lock()
            .expect("stacks poisoned")
            .push(first_tensor(&model_input).expect("a stacked tensor input"));
        Ok(Value::Number(1.0))
    }

    fn predict_spec_less(&self, observation: ModelObservation) -> Result<Vec<SpaceValue>> {
        Ok((0..observation.num_envs)
            .map(|_| SpaceValue::Discrete(0))
            .collect())
    }
}

async fn stacked_handler(
    env_id: &str,
    delivers_history: bool,
    horizon: u32,
) -> Result<(AdaptedModelHandler, Arc<WindowRecorder>, RouteNeeds)> {
    let recorder = Arc::new(WindowRecorder {
        stacks: Mutex::new(Vec::new()),
    });
    let handler = AdaptedModelHandler::new(
        Arc::clone(&recorder) as Arc<dyn PredictFn>,
        Some(Arc::new(StackResolver) as Arc<dyn RouteResolver>),
    );
    let needs = handler
        .route_setup()
        .expect("resolver-backed route setup")
        .resolve_adapter(
            env_id,
            &cam_contract(env_id),
            ResolveOptions {
                execution_horizon: horizon,
                delivers_history,
            },
        )
        .await?;
    Ok((handler, recorder, needs))
}

#[tokio::test]
async fn a_stacked_route_needs_history_only_when_the_runtime_offers_it() {
    let (_, _, needs) = stacked_handler("env-offer", true, 4).await.unwrap();
    let history = needs.history.expect("a stacked route asks for history");
    assert_eq!(history.keys.len(), 1, "one stacked input: {history:?}");
    assert!(!history.prunable);

    let refused = stacked_handler("env-refuse", false, 4)
        .await
        .err()
        .expect("stack + chunk without history is refused")
        .to_string();
    assert!(
        refused.contains("did not offer observation history"),
        "{refused}"
    );
    let (_, _, needs) = stacked_handler("env-h1", false, 1).await.unwrap();
    assert!(needs.history.is_none());

    let handler = AdaptedModelHandler::new(
        EchoModel::new(false, false) as Arc<dyn PredictFn>,
        Some(Arc::new(TagResolver) as Arc<dyn RouteResolver>),
    );
    let needs = handler
        .route_setup()
        .unwrap()
        .resolve_adapter(
            "env-flat",
            &contract("env-flat", 1),
            ResolveOptions {
                execution_horizon: 4,
                delivers_history: true,
            },
        )
        .await
        .unwrap();
    assert!(needs.history.is_none());
}

#[tokio::test]
async fn history_rows_advance_the_window_exactly_like_sequential_predicts() {
    let (mut every_step, sequential, _) = stacked_handler("env-seq", true, 1).await.unwrap();
    for step in 0..8u8 {
        every_step
            .predict(cam_predict(
                "env-seq",
                "ep",
                Some(i64::from(step)),
                step,
                &[],
            ))
            .await
            .expect("sequential predict");
    }
    let expected = sequential.stacks.lock().unwrap().clone();
    assert_eq!(expected.len(), 8);

    let (mut chunked, recorder, _) = stacked_handler("env-hist", true, 4).await.unwrap();
    for (step, rows) in [
        (0i64, vec![]),
        (4, vec![(1, 1u8), (2, 2), (3, 3)]),
        (7, vec![(5, 5), (6, 6)]),
    ] {
        chunked
            .predict(cam_predict("env-hist", "ep", Some(step), step as u8, &rows))
            .await
            .expect("chunked predict");
    }
    let got = recorder.stacks.lock().unwrap().clone();
    assert_eq!(
        got,
        vec![
            expected[0].clone(),
            expected[4].clone(),
            expected[7].clone()
        ]
    );
}

#[tokio::test]
async fn held_state_is_an_endpoint_total_kept_by_route_deltas() {
    let (mut handler, _, _) = stacked_handler("env-a", true, 1).await.unwrap();
    let setup = handler.route_setup().expect("resolver-backed route setup");
    setup
        .resolve_adapter(
            "env-b",
            &cam_contract("env-b"),
            ResolveOptions {
                execution_horizon: 1,
                delivers_history: true,
            },
        )
        .await
        .unwrap();
    handler
        .predict(cam_predict("env-a", "ep-a", Some(0), 1, &[]))
        .await
        .unwrap();
    handler
        .predict(cam_predict("env-b", "ep-b", Some(0), 2, &[]))
        .await
        .unwrap();
    let both = handler
        .held_state()
        .expect("the engine accounts held state");
    assert_eq!(both.episodes, 2);
    assert!(both.bytes > 0, "{both:?}");

    let handler = Arc::new(handler);
    let roster = handler.routes.lock().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    let reader = Arc::clone(&handler);
    std::thread::spawn(move || tx.send(reader.held_state()));
    let read = rx
        .recv_timeout(std::time::Duration::from_millis(50))
        .expect("held_state must not take the routes map lock");
    drop(roster);
    assert_eq!(read, Some(both));

    setup
        .reset_adapter("env-a", &["ep-a".to_string()])
        .await
        .unwrap();
    let one = handler.held_state().unwrap();
    assert_eq!(one.episodes, 1);
    assert!(
        one.bytes > 0 && one.bytes < both.bytes,
        "{one:?} vs {both:?}"
    );
    setup.release_adapter("env-b").await.unwrap();
    assert_eq!(handler.held_state().unwrap(), HeldState::default());
}

const PREVIOUS_ENV_TAGS: &str = r#"{
        "observation": {"g": {"type": "state", "role": "proprio/gripper"}},
        "action": {"components": [{"role": "action/joint_pos", "dim": 2}]}
    }"#;
const PREVIOUS_MODEL_SPEC: &str = r#"{
        "input": {"type": "state", "components": ["proprio/gripper",
                  {"role": "action/joint_pos", "source": "action"}]},
        "output": {"components": [{"role": "action/joint_pos", "dim": 2}]}
    }"#;
const PLAIN_MODEL_SPEC: &str = r#"{
        "input": {"type": "state", "components": ["proprio/gripper"]},
        "output": {"components": [{"role": "action/joint_pos", "dim": 2}]}
    }"#;

struct PreviousResolver(&'static str);

#[async_trait]
impl RouteResolver for PreviousResolver {
    async fn resolve(
        &self,
        _route_key: &str,
        env_contract: &EnvContract,
    ) -> Result<Option<RouteConfig>> {
        let tags: EnvTags = serde_json::from_str(PREVIOUS_ENV_TAGS).expect("env tags parse");
        let spec: ModelSpec = serde_json::from_str(self.0).expect("spec parse");
        let obs = env_contract
            .observation_space
            .clone()
            .expect("contract obs space");
        let action = env_contract
            .action_space
            .clone()
            .expect("contract action space");
        let adapter = resolve(
            &tags,
            &SpaceView::from(&obs),
            &SpaceView::from(&action),
            &spec,
            true,
        )
        .map_err(|err| Error::model(err.message))?;
        Ok(Some(RouteConfig::new(
            adapter,
            obs,
            action,
            Box::new(NoCustoms),
            Box::new(NoEncodings),
        )))
    }
}

fn g_obs_space() -> spaces::SpaceSpec {
    spaces::spaces::DictSpaceBuilder::new()
        .insert(
            "g",
            spaces::spaces::BoxSpaceBuilder::scalar(0.0, 100.0, vec![1])
                .dtype(DType::Float32)
                .build()
                .unwrap(),
        )
        .build()
        .unwrap()
}

fn g_contract(env_id: &str) -> Arc<spaces::EnvContract> {
    Arc::new(spaces::EnvContract {
        id: env_id.to_string(),
        observation_space: Some(g_obs_space()),
        action_space: Some(
            spaces::spaces::BoxSpaceBuilder::scalar(-1000.0, 1000.0, vec![2])
                .dtype(DType::Float32)
                .build()
                .unwrap(),
        ),
        metadata: None,
        render_mode: String::new(),
        num_envs: 1,
        autoreset_mode: Default::default(),
    })
}

fn g_leaves(g: f32) -> Vec<bytes::Bytes> {
    let lane = SpaceValue::Dict(BTreeMap::from([(
        "g".to_string(),
        SpaceValue::Box(
            Tensor::from_vec(g.to_le_bytes().to_vec(), vec![1], DType::Float32).unwrap(),
        ),
    )]));
    crate::data::encode_batched_partial_values(&[lane], &g_obs_space())
        .unwrap()
        .leaves
}

fn g_predict(
    env_id: &str,
    episode_id: &str,
    step: Option<i64>,
    g: f32,
    rows: &[(i64, f32)],
) -> ModelObservation {
    ModelObservation {
        observation: Some(g_leaves(g)),
        route: ModelRouteContext {
            env_id: env_id.to_string(),
            episodes: episode(episode_id),
            ..Default::default()
        },
        num_envs: 1,
        env_contract: Some(g_contract(env_id)),
        history: rows
            .iter()
            .map(|&(step, g)| crate::model::types::HistoryFrame {
                observation: Some(g_leaves(g)),
                episodes: episode(episode_id),
                step,
            })
            .collect(),
        step,
    }
}

struct PreviousRecorder {
    inputs: Mutex<Vec<Vec<f32>>>,
    calls: AtomicUsize,
}

impl PreviousRecorder {
    fn record(&self, model_input: &Value) -> usize {
        let Value::Tensor(state) = model_input else {
            panic!("expected a bare state tensor, got {model_input:?}");
        };
        let values: Vec<f32> = state
            .to_contiguous_bytes()
            .chunks(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        self.inputs.lock().expect("inputs poisoned").push(values);
        self.calls.fetch_add(1, Ordering::SeqCst)
    }

    fn frame(call: usize, k: usize) -> Value {
        let (a, b) = ((100 * call + 10 + k) as f32, (100 * call + 20 + k) as f32);
        let mut bytes = a.to_le_bytes().to_vec();
        bytes.extend(b.to_le_bytes());
        Value::Tensor(Tensor::from_vec(bytes, vec![2], DType::Float32).unwrap())
    }
}

impl PredictFn for PreviousRecorder {
    fn predict(&self, model_input: Value, _episode: Option<&EpisodeInfo>) -> Result<Value> {
        let call = self.record(&model_input);
        Ok(Self::frame(call, 0))
    }

    fn has_chunk(&self) -> bool {
        true
    }

    fn predict_chunk(
        &self,
        model_input: Value,
        _horizon: u32,
        _episode: Option<&EpisodeInfo>,
    ) -> Result<Option<Value>> {
        let call = self.record(&model_input);
        Ok(Some(Value::List(
            (0..4).map(|k| Self::frame(call, k)).collect(),
        )))
    }

    fn predict_spec_less(&self, observation: ModelObservation) -> Result<Vec<SpaceValue>> {
        Ok((0..observation.num_envs)
            .map(|_| SpaceValue::Discrete(0))
            .collect())
    }
}

async fn previous_handler(
    env_id: &str,
    spec: &'static str,
    delivers_history: bool,
    horizon: u32,
) -> Result<(AdaptedModelHandler, Arc<PreviousRecorder>, RouteNeeds)> {
    let recorder = Arc::new(PreviousRecorder {
        inputs: Mutex::new(Vec::new()),
        calls: AtomicUsize::new(0),
    });
    let handler = AdaptedModelHandler::new(
        Arc::clone(&recorder) as Arc<dyn PredictFn>,
        Some(Arc::new(PreviousResolver(spec)) as Arc<dyn RouteResolver>),
    );
    let needs = handler
        .route_setup()
        .expect("resolver-backed route setup")
        .resolve_adapter(
            env_id,
            &g_contract(env_id),
            ResolveOptions {
                execution_horizon: horizon,
                delivers_history,
            },
        )
        .await?;
    Ok((handler, recorder, needs))
}

#[tokio::test]
async fn a_previous_action_part_reads_the_last_replayed_frame_at_the_replan() {
    let (mut handler, recorder, needs) = previous_handler("env-prev", PREVIOUS_MODEL_SPEC, true, 4)
        .await
        .unwrap();
    let history = needs
        .history
        .expect("a previous-action route asks for history");
    assert_eq!(history.keys, vec!["<root>".to_string()]);
    let frames = handler
        .predict_chunked(g_predict("env-prev", "ep", Some(0), 1.0, &[]))
        .await
        .expect("first predict");
    assert_eq!(frames.replay.len(), 3);
    handler
        .predict_chunked(g_predict(
            "env-prev",
            "ep",
            Some(4),
            5.0,
            &[(1, 2.0), (2, 3.0), (3, 4.0)],
        ))
        .await
        .expect("re-plan");
    assert_eq!(
        recorder.inputs.lock().unwrap().clone(),
        vec![vec![1.0, 0.0, 0.0], vec![5.0, 13.0, 23.0]]
    );
}

#[tokio::test]
async fn a_previous_action_part_under_a_prefetch_lead_reads_the_frame_that_executed() {
    let (mut handler, recorder, _) = previous_handler("env-lead", PREVIOUS_MODEL_SPEC, true, 4)
        .await
        .unwrap();
    handler
        .predict_chunked(g_predict("env-lead", "ep", Some(0), 1.0, &[]))
        .await
        .expect("first predict");
    handler
        .predict_chunked(g_predict(
            "env-lead",
            "ep",
            Some(3),
            4.0,
            &[(1, 2.0), (2, 3.0)],
        ))
        .await
        .expect("prefetched re-plan");
    handler
        .predict_chunked(g_predict(
            "env-lead",
            "ep",
            Some(7),
            8.0,
            &[(4, 5.0), (5, 6.0), (6, 7.0)],
        ))
        .await
        .expect("second prefetched re-plan");
    assert_eq!(
        recorder.inputs.lock().unwrap().clone(),
        vec![
            vec![1.0, 0.0, 0.0],
            vec![4.0, 12.0, 22.0],
            vec![8.0, 112.0, 122.0]
        ]
    );
}

#[tokio::test]
async fn a_previous_action_route_counts_its_own_steps_when_none_are_stamped() {
    let (mut handler, recorder, needs) =
        previous_handler("env-count", PREVIOUS_MODEL_SPEC, false, 1)
            .await
            .unwrap();
    assert!(needs.history.is_none());
    for g in [1.0f32, 2.0, 3.0] {
        handler
            .predict_chunked(g_predict("env-count", "ep", None, g, &[]))
            .await
            .expect("predict");
    }
    assert_eq!(
        recorder.inputs.lock().unwrap().clone(),
        vec![
            vec![1.0, 0.0, 0.0],
            vec![2.0, 10.0, 20.0],
            vec![3.0, 110.0, 120.0]
        ]
    );
    let refused = previous_handler("env-refuse", PREVIOUS_MODEL_SPEC, false, 4)
        .await
        .err()
        .expect("a chunked previous-action route without history is refused")
        .to_string();
    assert!(
        refused.contains("a previous-action part (input '<root>') needs every env step"),
        "{refused}"
    );
    let (mut plain, recorder, needs) = previous_handler("env-plain", PLAIN_MODEL_SPEC, true, 4)
        .await
        .unwrap();
    assert!(needs.history.is_none());
    plain
        .predict_chunked(g_predict("env-plain", "ep", None, 1.0, &[]))
        .await
        .expect("plain predict");
    assert_eq!(recorder.inputs.lock().unwrap().clone(), vec![vec![1.0]]);
}

#[tokio::test]
async fn a_negotiated_route_rejects_a_step_gap_repeat_or_omission() {
    let (mut handler, _, _) = stacked_handler("env-steps", true, 4).await.unwrap();
    handler
        .predict(cam_predict("env-steps", "ep", Some(0), 0, &[]))
        .await
        .expect("first predict");

    let gap = handler
        .predict(cam_predict("env-steps", "ep", Some(2), 2, &[]))
        .await
        .unwrap_err()
        .to_string();
    assert!(gap.contains("not consecutive"), "{gap}");

    let repeat = handler
        .predict(cam_predict("env-steps", "ep", Some(1), 1, &[(0, 0)]))
        .await
        .unwrap_err()
        .to_string();
    assert!(repeat.contains("step 0 arrived after step 0"), "{repeat}");

    let omitted = handler
        .predict(cam_predict("env-steps", "ep", None, 1, &[]))
        .await
        .unwrap_err()
        .to_string();
    assert!(omitted.contains("carries no step"), "{omitted}");

    handler
        .predict(cam_predict("env-steps", "ep-2", Some(40), 0, &[]))
        .await
        .expect("new episode, any step");
}

#[tokio::test]
async fn rows_sent_to_a_route_that_did_not_negotiate_history_are_refused() {
    let (mut handler, _, needs) = stacked_handler("env-none", false, 1).await.unwrap();
    assert!(needs.history.is_none());
    let refused = handler
        .predict(cam_predict("env-none", "ep", Some(1), 1, &[(0, 0)]))
        .await
        .unwrap_err()
        .to_string();
    assert!(refused.contains("did not negotiate history"), "{refused}");
}

#[test]
fn the_frame_history_budget_refuses_a_window_that_outgrows_it() {
    let window = |frame_bytes: u64| HistoryWindow {
        key: "cam".to_string(),
        span: 4,
        frame_bytes,
    };
    let limit = FRAME_HISTORY_LIMIT_DEFAULT;
    check_history_budget(8, &[window(224 * 224 * 3)], limit).expect("a few MiB per lane");
    let refused = check_history_budget(8, &[window(1 << 30)], limit)
        .unwrap_err()
        .to_string();
    assert!(refused.contains(FRAME_HISTORY_LIMIT_ENV), "{refused}");
}

fn cam_predict_lanes(env_id: &str, episode_ids: &[&str], step: i64) -> ModelObservation {
    let lanes: Vec<SpaceValue> = episode_ids
        .iter()
        .map(|_| {
            SpaceValue::Dict(BTreeMap::from([(
                "cam".to_string(),
                SpaceValue::Box(
                    Tensor::from_vec(vec![step as u8; 3], vec![1, 1, 3], DType::Uint8).unwrap(),
                ),
            )]))
        })
        .collect();
    ModelObservation {
        observation: Some(
            crate::data::encode_batched_partial_values(&lanes, &cam_obs_space())
                .unwrap()
                .leaves,
        ),
        route: ModelRouteContext {
            env_id: env_id.to_string(),
            episodes: episode_ids
                .iter()
                .map(|id| EpisodeInfo {
                    episode_id: id.to_string(),
                    seed: None,
                    env_id: String::new(),
                })
                .collect(),
            ..Default::default()
        },
        num_envs: episode_ids.len(),
        env_contract: Some(cam_contract(env_id)),
        history: Vec::new(),
        step: Some(step),
    }
}

#[tokio::test]
async fn fresh_lanes_are_admitted_against_the_endpoint_window_budget() {
    const LANE_BYTES: u64 = 9;
    let recorder = Arc::new(WindowRecorder {
        stacks: Mutex::new(Vec::new()),
    });
    let mut handler = AdaptedModelHandler::new(
        recorder as Arc<dyn PredictFn>,
        Some(Arc::new(StackResolver) as Arc<dyn RouteResolver>),
    );
    handler.history_limit = 2 * LANE_BYTES;
    let setup = handler.route_setup().expect("resolver-backed");
    for env_id in ["env-a", "env-b"] {
        setup
            .resolve_adapter(
                env_id,
                &cam_contract(env_id),
                ResolveOptions {
                    execution_horizon: 1,
                    delivers_history: true,
                },
            )
            .await
            .expect("one lane fits");
    }
    assert_eq!(
        handler.entry("env-a").unwrap().lock().unwrap().lane_bytes,
        LANE_BYTES
    );

    let refused = handler
        .predict(cam_predict_lanes("env-a", &["a1", "a2", "a3"], 0))
        .await
        .unwrap_err()
        .to_string();
    assert!(refused.contains(FRAME_HISTORY_LIMIT_ENV), "{refused}");
    assert_eq!(handler.held_state(), Some(HeldState::default()));

    for step in 0..2 {
        handler
            .predict(cam_predict_lanes("env-a", &["a1", "a2"], step))
            .await
            .expect("two lanes fit");
    }
    let refused = handler
        .predict(cam_predict_lanes("env-b", &["b1"], 0))
        .await
        .unwrap_err()
        .to_string();
    assert!(refused.contains("other routes"), "{refused}");

    setup
        .reset_adapter("env-a", &["a2".to_string()])
        .await
        .unwrap();
    handler
        .predict(cam_predict_lanes("env-b", &["b1"], 0))
        .await
        .expect("room again after an episode ends");
}
