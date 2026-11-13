//! The engine against real resolved routes: grouped and fused predicts, the
//! horizon doors, observation history, and previous-action parts.

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

/// Resolves every route through the one ENV_TAGS x MODEL_SPEC pairing — the
/// minimal real spec'd route (one gripper state in, one gripper action out),
/// so grouped predicts exercise assemble -> fuse -> split -> finish against
/// genuine adapter plans instead of hand-built frames.
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

/// Echoes each lane's state value back as its action (lane-identifying, so a
/// split-back misalignment returns the wrong route's actions and fails the
/// equality asserts). The chunk corner emits `CHUNK_FRAMES` frames of
/// `state + 0.125 * frame`; `fail_recoverable` makes the batched corner
/// return a recoverable model error instead.
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

    /// A single-lane model (no batched corner, so the engine always runs
    /// the per-lane `predict()` loop), for asserting real episode identity
    /// reaches it.
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

/// The first number reachable in the assembled input tree (the echoed state).
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

/// A grouped-member observation for `env_id`: one lane per value, each lane's
/// state = the value, episode ids derived from the values (unique per lane).
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

/// A single-lane observation for `env_id` carrying real episode id/seed
/// metadata, unlike `grouped_obs` (which leaves `episode_seeds` default).
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

/// Build a handler with real resolved routes (one per `(env_id, lanes)`),
/// pinning `horizon` on each at resolve.
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
                    workflow_edition: None,
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
    // Grouped predict serves multiple routes' episodes in ONE batched
    // forward. There is no single episode identity for the call, but every
    // ROW has one: the fused batch concatenates each group's inputs and its
    // episodes in the same order, so the batched corner can key per-episode
    // state by row.
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

/// Regression for the grouped-fusion deadlock: the fused path once held every
/// route's entry guard across the whole call, so a request repeating one
/// env_id re-locked the same mutex on one thread and hung forever. Locks are
/// now short-lived; a duplicate route must simply serve twice.
/// A wide fused batch — one route per lane, the fleet shape — assembles and
/// finishes its groups on the rayon pool; every result must still land at
/// its own group's index with its own lane's state, across the chunked and
/// un-chunked corners.
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

/// The camera route (the SmolVLA-on-LIBERO shape): one 256x256x3 uint8 frame
/// tagged upside-down, consumed as normalised float32 CHW, plus a gripper state.
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

/// `cargo test -p rlmesh --release camera_grouped_predict_timing -- --ignored --nocapture`
/// (and again with `RAYON_NUM_THREADS=1`): wall time of one fused grouped predict
/// over 32 camera routes and the adapter span it reports, so the lane fan-out's
/// gain is a number rather than a projection.
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
                    workflow_edition: None,
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

/// A chunk model that emits `emit` frames per chunk call and optionally
/// DECLARES a native chunk length, so the resolve doors and `take_prefix`
/// can be driven against every combination of (declared, emitted, horizon).
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

    /// Declares a chunk length but defines no chunk corner at all.
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

/// A resolver that declares every route spec-LESS, so the doors can be driven
/// down the branch that never builds a `RouteConfig`.
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

/// Resolve one route at `horizon` and hand back the setup's answer.
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
                workflow_edition: None,
            },
        )
        .await
}

/// Total per-step frames one predict emitted (frame 0 plus the replay tail).
async fn frame_count(handler: &mut AdaptedModelHandler, obs: ModelObservation) -> usize {
    let frames = handler
        .predict_chunked(obs)
        .await
        .expect("predict succeeds");
    1 + frames.replay.len()
}

// 1. A horizon past the bound is a mis-set knob, refused before any work.
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

// 2. The bound itself is legal (it is a ceiling, not an exclusive limit).
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

// 3. The same door guards the spec-LESS branch, which never builds a config.
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

// 4. Declaring K without a chunk corner could never be honored or checked.
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

// 5. h > K asks for actions the model cannot produce.
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

// 6. A resolved route answers the declaration back to the runtime.
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

// 7. An undeclared model answers nothing: the elastic contract.
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

// 8. The served path sees no lanes: `ResolveAdapterRequest` carries no
//    num_envs, so the engine must NOT try to guard lanes here. A vector
//    contract with a chunking horizon resolves; the refusal lives in the
//    layers that actually know the lane count.
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

// 9. Declared K, whole chunk returned, h < K: the runtime takes the prefix.
#[tokio::test]
async fn a_declared_model_replays_the_horizon_prefix_of_its_whole_chunk() {
    let model = DeclaredChunkModel::new(6, Some(6));
    let (mut handler, contracts) =
        spec_handler(model as Arc<dyn PredictFn>, &[("env-a", 1)], 4).await;
    let obs = single_obs_with_episode("env-a", 1.0, "ep", None, &contracts[0]);
    assert_eq!(frame_count(&mut handler, obs).await, 4);
}

// 10. Declared K at h == K: the whole chunk is executed.
#[tokio::test]
async fn a_declared_model_at_the_full_horizon_replays_every_frame() {
    let model = DeclaredChunkModel::new(6, Some(6));
    let (mut handler, contracts) =
        spec_handler(model as Arc<dyn PredictFn>, &[("env-a", 1)], 6).await;
    let obs = single_obs_with_episode("env-a", 1.0, "ep", None, &contracts[0]);
    assert_eq!(frame_count(&mut handler, obs).await, 6);
}

// 11. Declared K but the model still slices to the horizon (the glue this
//     contract removes): the predict fails instead of short-replaying.
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

// 12. Undeclared and short: elastic, so the run continues on what there is.
#[tokio::test]
async fn an_undeclared_short_chunk_replays_what_there_is() {
    let model = DeclaredChunkModel::new(3, None);
    let (mut handler, contracts) =
        spec_handler(model as Arc<dyn PredictFn>, &[("env-a", 1)], 6).await;
    let obs = single_obs_with_episode("env-a", 1.0, "ep", None, &contracts[0]);
    assert_eq!(frame_count(&mut handler, obs).await, 3);
}

// 13. That short-chunk warning is latched once per episode, not per predict.
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

// 13b. Declared-K and empty-chunk refusals are the ValueError class.
#[test]
fn chunk_refusals_are_value_errors() {
    let warned = ShortChunkWarned::default();
    let error = take_prefix(vec![Value::Number(0.0)], 2, Some(3), &warned, "ep").unwrap_err();
    assert_eq!(error.class(), crate::ErrorClass::ValueError);
    let error = take_prefix(Vec::new(), 2, None, &warned, "ep").unwrap_err();
    assert_eq!(error.class(), crate::ErrorClass::ValueError);
}

// ---- observation history (a stacked route fed replayed steps as rows) ----

const STACK_ENV_TAGS: &str = r#"{
        "observation": {"cam": {"type": "image", "role": "image/primary", "layout": "hwc"}},
        "action": {"components": [{"role": "action/gripper", "dim": 1, "range": [0.0, 10.0]}]}
    }"#;
const STACK_MODEL_SPEC: &str = r#"{
        "input": {"type": "image", "role": "image/primary", "layout": "hwc", "dtype": "uint8",
                  "stack": 3},
        "output": {"components": [{"role": "action/gripper", "dim": 1, "range": [0.0, 10.0]}]}
    }"#;

/// Resolves every route as one 1x1 RGB camera stacked 3 deep, so the
/// engine's frame windows are real and every predict's input is the stack.
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

/// The wire leaves of a one-lane camera frame whose every byte is `tag`.
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

/// A predict at `step` carrying the frame `tag`, preceded by `rows` of
/// `(step, tag)` replayed steps.
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

/// Records the stacked frame bytes each predict was handed.
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

/// A stacked handler resolved for `env_id`, with or without the runtime
/// offering history, at `horizon`.
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
                workflow_edition: None,
            },
        )
        .await?;
    Ok((handler, recorder, needs))
}

#[tokio::test]
async fn a_stacked_route_needs_history_only_when_the_runtime_offers_it() {
    // Offered: the route answers its window keys and the horizon is fine.
    let (_, _, needs) = stacked_handler("env-offer", true, 4).await.unwrap();
    let history = needs.history.expect("a stacked route asks for history");
    assert_eq!(history.keys.len(), 1, "one stacked input: {history:?}");
    assert!(!history.prunable);

    // Not offered: above horizon 1 the window would hold decision points
    // only, so the route is refused; at horizon 1 every step predicts.
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

    // An unstacked route never asks, offered or not.
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
                workflow_edition: None,
            },
        )
        .await
        .unwrap();
    assert!(needs.history.is_none());
}

#[tokio::test]
async fn history_rows_advance_the_window_exactly_like_sequential_predicts() {
    // The invariant the wire exists for: a predict carrying the replayed
    // steps as rows hands the model the same stack a predict-every-step
    // route would have at that step.
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
    // Predict at 0; steps 1..3 replay; predict at 4 with rows 1..3; steps
    // 5 and 6 replay; predict at 7 with rows 5 and 6.
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
    // Two stacked routes, each holding one episode's window.
    let (mut handler, _, _) = stacked_handler("env-a", true, 1).await.unwrap();
    let setup = handler.route_setup().expect("resolver-backed route setup");
    setup
        .resolve_adapter(
            "env-b",
            &cam_contract("env-b"),
            ResolveOptions {
                execution_horizon: 1,
                delivers_history: true,
                workflow_edition: None,
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

    // The total answers without the roster: read it from another thread
    // while this one holds the routes map lock.
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

    // Evicting one route's episode, then releasing the other route, each
    // take exactly their share back out of the total.
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

// ---- previous action (a route reading back its own executed frames) ----

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

/// Resolves every route as a scalar gripper reading plus, with
/// [`PREVIOUS_MODEL_SPEC`], the model's own previous 2-d joint command.
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

/// A predict at `step` reading `g`, preceded by `rows` of `(step, g)`
/// replayed steps.
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

/// Records the state each predict was handed and answers a chunk whose
/// frame `k` of call `c` is `[100c + 10 + k, 100c + 20 + k]`, so every
/// executed frame is telling.
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
                workflow_edition: None,
            },
        )
        .await?;
    Ok((handler, recorder, needs))
}

#[tokio::test]
async fn a_previous_action_part_reads_the_last_replayed_frame_at_the_replan() {
    // execution_horizon 4: the predict at step 0 emits frames 0..3, the
    // runtime replays 1..3 and re-plans at 4 carrying their rows. The
    // policy then reads frame 3 (the action executed at step 3), not
    // frame 0 (the one it saw itself emit), and the fill before any.
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
    // execution_horizon 4 with a prefetch lead of 1: the re-plan is
    // requested at step 3, one frame before the first chunk runs out, so
    // its own observation is step 3 but its frames execute from step 4.
    // The predict at step 7 must read call 1's frame 2 (executed at 6),
    // not frame 3, which the request's own step would have recorded there.
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
    // No history offered at execution_horizon 1: every predict is one env
    // step, so the engine numbers them itself and each predict reads the
    // frame the one before it emitted.
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
    // Above 1 without the offer the route is refused, like a stacked one.
    let refused = previous_handler("env-refuse", PREVIOUS_MODEL_SPEC, false, 4)
        .await
        .err()
        .expect("a chunked previous-action route without history is refused")
        .to_string();
    assert!(
        refused.contains("a previous-action part (input '<root>') needs every env step"),
        "{refused}"
    );
    // A route without the part negotiates nothing, offered or not, and
    // its predicts are the plain ones.
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

    // A fresh episode starts its own sequence at any step.
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

/// A predict for `env_id` at `step` carrying one frame per lane, each lane
/// its own episode.
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
    // The 1x1x3 uint8 camera stacked 3 deep: 9 bytes per lane once full.
    // Budget two lanes' worth for the whole endpoint.
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
        // Each route fits on its own: the served resolve sees one lane.
        setup
            .resolve_adapter(
                env_id,
                &cam_contract(env_id),
                ResolveOptions {
                    execution_horizon: 1,
                    delivers_history: true,
                    workflow_edition: None,
                },
            )
            .await
            .expect("one lane fits");
    }
    assert_eq!(
        handler.entry("env-a").unwrap().lock().unwrap().lane_bytes,
        LANE_BYTES
    );

    // Many lanes at once: the multiplier is the real lane count, not the
    // resolve-time placeholder, and a refusal seeds nothing.
    let refused = handler
        .predict(cam_predict_lanes("env-a", &["a1", "a2", "a3"], 0))
        .await
        .unwrap_err()
        .to_string();
    assert!(refused.contains(FRAME_HISTORY_LIMIT_ENV), "{refused}");
    assert_eq!(handler.held_state(), Some(HeldState::default()));

    // Two lanes fill the budget (their windows are charged at full size
    // from the first frame), and revisiting held lanes is never charged.
    for step in 0..2 {
        handler
            .predict(cam_predict_lanes("env-a", &["a1", "a2"], step))
            .await
            .expect("two lanes fit");
    }
    // A second route that fits on its own is refused against the total.
    let refused = handler
        .predict(cam_predict_lanes("env-b", &["b1"], 0))
        .await
        .unwrap_err()
        .to_string();
    assert!(refused.contains("other routes"), "{refused}");

    // An ended episode gives its window back (the route setup evicts the
    // window; the handler's reset is only the model's episode-end hook).
    setup
        .reset_adapter("env-a", &["a2".to_string()])
        .await
        .unwrap();
    handler
        .predict(cam_predict_lanes("env-b", &["b1"], 0))
        .await
        .expect("room again after an episode ends");
}
