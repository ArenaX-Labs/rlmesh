#![allow(clippy::unwrap_used)]

mod common;

use std::sync::{Arc, Mutex};

use rlmesh::binding::model::{
    EpisodeInfo, ModelHandler, ModelObservation, ModelRouteContext, ResolveOptions, model_handler,
};
use rlmesh::types::BatchPredictContext;
use rlmesh::{Corners, Model, ModelOptions, Value};
use rlmesh_spaces::leaves::encode_leaf_slab;
use rlmesh_spaces::{AutoresetMode, EnvContract, SpaceSpec, SpaceValue};

fn contract(action_space: SpaceSpec, num_envs: u32) -> EnvContract {
    EnvContract {
        id: "Chat-v0".into(),
        action_space: Some(action_space),
        observation_space: Some(common::text()),
        metadata: None,
        render_mode: String::new(),
        num_envs,
        autoreset_mode: AutoresetMode::Disabled,
    }
}

fn group(route: &str, texts: &[&str], action_space: SpaceSpec) -> ModelObservation {
    let lanes: Vec<SpaceValue> = texts
        .iter()
        .map(|text| SpaceValue::Text((*text).to_string()))
        .collect();
    let contract = contract(action_space, texts.len() as u32);
    ModelObservation {
        observation: Some(
            encode_leaf_slab(&lanes, contract.observation_space.as_ref().unwrap()).unwrap(),
        ),
        route: ModelRouteContext {
            env_id: route.into(),
            episodes: (0..texts.len())
                .map(|lane| EpisodeInfo {
                    episode_id: format!("{route}-{lane}"),
                    seed: Some(lane as i64),
                    env_id: route.into(),
                })
                .collect(),
            ..ModelRouteContext::default()
        },
        num_envs: texts.len(),
        env_contract: Some(Arc::new(contract)),
        ..ModelObservation::default()
    }
}

type Calls = Arc<Mutex<Vec<Vec<(String, u64, String)>>>>;

fn record(calls: &Calls, observations: &[Value], context: &BatchPredictContext<'_>) {
    calls.lock().unwrap().push(
        observations
            .iter()
            .zip(context.iter())
            .map(|(observation, row)| {
                let Value::Text(text) = observation else {
                    panic!("a text observation, got {observation:?}")
                };
                (row.episode_id.to_string(), row.predict_index, text.clone())
            })
            .collect(),
    );
}

struct Harness(Calls);

impl rlmesh::Policy for Harness {
    const CORNERS: Corners = Corners::PREDICT_BATCH;
    const CONTEXT: Corners = Corners::PREDICT_BATCH;

    fn predict_batch_with_context(
        &mut self,
        observations: Vec<Value>,
        context: &mut BatchPredictContext<'_>,
    ) -> rlmesh::Result<Vec<Value>> {
        record(&self.0, &observations, context);
        Ok(observations
            .iter()
            .map(|observation| match observation {
                Value::Text(text) => Value::Text(format!("re:{text}")),
                _ => Value::None,
            })
            .collect())
    }
}

fn texts(frames: &rlmesh::binding::model::PredictFrames) -> Vec<String> {
    frames
        .actions
        .iter()
        .map(|action| match action {
            SpaceValue::Text(text) => text.clone(),
            other => panic!("a text action, got {other:?}"),
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_spec_less_batched_model_serves_every_group_in_one_call() {
    let calls = Calls::default();
    let model = Model::new(Harness(Arc::clone(&calls)), ModelOptions::default()).unwrap();
    let mut handler = model_handler(&model);
    assert!(handler.fuses_groups());

    let outcomes = handler
        .predict_grouped(vec![
            group("a", &["a0"], common::text()),
            group("b", &["b0", "b1"], common::text()),
            group("c", &["c0"], common::text()),
        ])
        .await;
    let outcomes: Vec<Vec<String>> = outcomes
        .iter()
        .map(|o| texts(o.as_ref().unwrap()))
        .collect();
    assert_eq!(
        outcomes,
        vec![vec!["re:a0"], vec!["re:b0", "re:b1"], vec!["re:c0"]]
    );

    let outcomes = handler
        .predict_grouped(vec![
            group("c", &["c0'"], common::text()),
            group("a", &["a0'"], common::text()),
        ])
        .await;
    assert!(outcomes.iter().all(Result::is_ok));

    let calls = calls.lock().unwrap().clone();
    let row =
        |episode: &str, index: u64, text: &str| (episode.to_string(), index, text.to_string());
    assert_eq!(
        calls,
        vec![
            vec![
                row("a-0", 0, "a0"),
                row("b-0", 0, "b0"),
                row("b-1", 0, "b1"),
                row("c-0", 0, "c0"),
            ],
            vec![row("c-0", 1, "c0'"), row("a-0", 1, "a0'")],
        ],
        "one call per grouped predict, with each row's own episode context"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_group_whose_action_does_not_fit_fails_alone() {
    let calls = Calls::default();
    let model = Model::new(Harness(Arc::clone(&calls)), ModelOptions::default()).unwrap();
    let mut handler = model_handler(&model);
    let outcomes = handler
        .predict_grouped(vec![
            group("a", &["a0"], common::text()),
            group(
                "b",
                &["b0"],
                common::boxed(rlmesh_spaces::DType::Float32, &[2]),
            ),
            group("c", &["c0"], common::text()),
        ])
        .await;
    assert_eq!(calls.lock().unwrap().len(), 1);
    assert_eq!(texts(outcomes[0].as_ref().unwrap()), vec!["re:a0"]);
    assert!(outcomes[1].is_err(), "{:?}", outcomes[1]);
    assert_eq!(texts(outcomes[2].as_ref().unwrap()), vec!["re:c0"]);
}

struct Failing;

impl rlmesh::Policy for Failing {
    const CORNERS: Corners = Corners::PREDICT_BATCH;

    fn predict_batch(&mut self, _observations: Vec<Value>) -> rlmesh::Result<Vec<Value>> {
        Ok(vec![Value::Text("only one".into())])
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_short_batch_fails_every_group_in_the_call() {
    let model = Model::new(Failing, ModelOptions::default()).unwrap();
    let mut handler = model_handler(&model);
    let outcomes = handler
        .predict_grouped(vec![
            group("a", &["a0"], common::text()),
            group("b", &["b0"], common::text()),
        ])
        .await;
    for outcome in outcomes {
        let error = outcome.unwrap_err().to_string();
        assert!(error.contains("returned 1 results for 2 lanes"), "{error}");
    }
}

struct Chunker(Calls);

impl rlmesh::Policy for Chunker {
    const CORNERS: Corners = Corners::PREDICT.union(Corners::PREDICT_CHUNK_BATCH);
    const CONTEXT: Corners = Corners::PREDICT_CHUNK_BATCH;

    fn predict(&mut self, observation: Value) -> rlmesh::Result<Value> {
        Ok(observation)
    }

    fn predict_chunk_batch_with_context(
        &mut self,
        observations: Vec<Value>,
        execution_horizon: u32,
        context: &mut BatchPredictContext<'_>,
    ) -> rlmesh::Result<Vec<Value>> {
        record(&self.0, &observations, context);
        Ok(observations
            .iter()
            .map(|observation| {
                let Value::Text(text) = observation else {
                    return Value::None;
                };
                Value::List(
                    (0..=execution_horizon)
                        .map(|frame| Value::Text(format!("{text}@{frame}")))
                        .collect(),
                )
            })
            .collect())
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_spec_less_chunk_batch_model_fuses_groups_and_vector_lanes() {
    let calls = Calls::default();
    let model = Model::new(Chunker(Arc::clone(&calls)), ModelOptions::default()).unwrap();
    let mut handler = model_handler(&model);
    let setup = handler.route_setup().unwrap();
    for (route, lanes) in [("a", 1), ("b", 2)] {
        setup
            .resolve_adapter(
                route,
                &contract(common::text(), lanes),
                ResolveOptions {
                    execution_horizon: 2,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
    }

    let outcomes = handler
        .predict_grouped(vec![
            group("a", &["a0"], common::text()),
            group("b", &["b0", "b1"], common::text()),
        ])
        .await;
    let a = outcomes[0].as_ref().unwrap();
    let b = outcomes[1].as_ref().unwrap();
    assert_eq!(texts(a), vec!["a0@0"]);
    assert_eq!(texts(b), vec!["b0@0", "b1@0"]);
    assert_eq!(a.replay, vec![vec![SpaceValue::Text("a0@1".into())]]);
    assert_eq!(
        b.replay,
        vec![vec![
            SpaceValue::Text("b0@1".into()),
            SpaceValue::Text("b1@1".into())
        ]]
    );
    assert_eq!(calls.lock().unwrap().len(), 1, "one chunk-batch call");

    let frames = handler
        .predict_chunked(group("b", &["b0'", "b1'"], common::text()))
        .await
        .unwrap();
    assert_eq!(texts(&frames), vec!["b0'@0", "b1'@0"]);
    assert_eq!(frames.replay.len(), 1, "a lone vector group chunks too");
}

struct ChunkOrBatch(Calls);

impl rlmesh::Policy for ChunkOrBatch {
    const CORNERS: Corners = Corners::PREDICT
        .union(Corners::PREDICT_CHUNK)
        .union(Corners::PREDICT_BATCH);
    const CONTEXT: Corners = Corners::PREDICT_BATCH;

    fn predict(&mut self, observation: Value) -> rlmesh::Result<Value> {
        Ok(observation)
    }

    fn predict_chunk(&mut self, observation: Value, _horizon: u32) -> rlmesh::Result<Value> {
        Ok(Value::List(vec![observation]))
    }

    fn predict_batch_with_context(
        &mut self,
        observations: Vec<Value>,
        context: &mut BatchPredictContext<'_>,
    ) -> rlmesh::Result<Vec<Value>> {
        Harness(Arc::clone(&self.0)).predict_batch_with_context(observations, context)
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_vector_group_without_a_chunk_batch_corner_takes_the_batched_corner() {
    let calls = Calls::default();
    let model = Model::new(ChunkOrBatch(Arc::clone(&calls)), ModelOptions::default()).unwrap();
    let mut handler = model_handler(&model);
    handler
        .route_setup()
        .unwrap()
        .resolve_adapter(
            "v",
            &contract(common::text(), 2),
            ResolveOptions {
                execution_horizon: 2,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let frames = handler
        .predict_chunked(group("v", &["v0", "v1"], common::text()))
        .await
        .unwrap();
    assert_eq!(texts(&frames), vec!["re:v0", "re:v1"]);
    let calls = calls.lock().unwrap().clone();
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0]
            .iter()
            .map(|row| row.0.as_str())
            .collect::<Vec<_>>(),
        vec!["v-0", "v-1"]
    );
}
