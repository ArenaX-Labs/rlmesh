//! The facade `Model` over a `Policy`, end to end against a served env.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::*;
use crate::binding::model::{RunLocalOptions, run_model_local};
use crate::spaces::{Discrete, DiscreteOptions, Space};
use crate::{Env, EnvServer, EnvServerOptions, ErrorClass, InfoDict, Kwargs, Result, Value};

/// A three-step env; `reset` seeds with the given seed.
struct Steps {
    step: i64,
}

impl Env for Steps {
    fn observation_space(&self) -> Space {
        Discrete::new(5, DiscreteOptions::default()).unwrap().into()
    }
    fn action_space(&self) -> Space {
        Discrete::new(2, DiscreteOptions::default()).unwrap().into()
    }
    fn reset(
        &mut self,
        _seed: Option<i64>,
        _options: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        self.step = 0;
        Ok((Value::Int(0), InfoDict::new()))
    }
    fn step(&mut self, _action: Option<Value>) -> Result<(Value, f64, bool, bool, InfoDict)> {
        self.step += 1;
        Ok((
            Value::Int(self.step),
            1.0,
            self.step >= 3,
            false,
            InfoDict::new(),
        ))
    }
}

/// A served `Steps` env and its address.
fn served_env() -> (EnvServer, String) {
    let server = EnvServer::new(
        Steps { step: 0 },
        EnvServerOptions {
            address: Some("tcp://127.0.0.1:0".into()),
            options: Some(crate::ServeOptions {
                drain_timeout: Some(Duration::from_millis(100)),
                ..Default::default()
            }),
            ..Default::default()
        },
    )
    .unwrap();
    server.start().unwrap();
    let address = server.address().to_string();
    (server, address)
}

#[derive(Default)]
struct Log {
    indices: Vec<(String, u64, u64)>,
    ended: Vec<String>,
    closes: usize,
    batch_sizes: Vec<usize>,
}

/// Counts its predicts per episode in the episode state.
struct Counting {
    log: Arc<Mutex<Log>>,
}

impl Policy for Counting {
    const CONTEXT: Corners = Corners::PREDICT;

    fn predict_with_context(
        &mut self,
        _observation: Value,
        context: &mut PredictContext<'_>,
    ) -> Result<Value> {
        let count = context.state.get_or_insert_with("count", || 0u64)?;
        *count += 1;
        let count = *count;
        self.log.lock().unwrap().indices.push((
            context.episode_id.to_string(),
            context.predict_index,
            count,
        ));
        Ok(Value::Int(0))
    }

    fn on_episode_end(&mut self, episode_id: &str) -> Result<()> {
        self.log.lock().unwrap().ended.push(episode_id.to_string());
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        self.log.lock().unwrap().closes += 1;
        Ok(())
    }
}

#[test]
fn a_context_policy_counts_per_episode_and_hears_each_episode_end() {
    let (server, address) = served_env();
    let log = Arc::new(Mutex::new(Log::default()));
    let model = Model::new(
        Counting {
            log: Arc::clone(&log),
        },
        ModelOptions::default(),
    )
    .unwrap();
    run_model_local(
        &model,
        RunLocalOptions::parse(&address).unwrap().for_episodes(2),
    )
    .unwrap();
    {
        let log = log.lock().unwrap();
        assert_eq!(log.indices.len(), 6, "three predicts per episode");
        for (index, (_, predict_index, count)) in log.indices.iter().enumerate() {
            assert_eq!(*predict_index as usize, index % 3);
            assert_eq!(
                *count as usize,
                index % 3 + 1,
                "state lives for the episode"
            );
        }
        assert_eq!(log.ended.len(), 2, "one episode end per episode");
        assert_eq!(log.closes, 0, "a run borrows the model");
    }
    model.close().unwrap();
    model.close().unwrap();
    assert_eq!(log.lock().unwrap().closes, 1, "close fires once");
    let error = run_model_local(&model, RunLocalOptions::parse(&address).unwrap()).unwrap_err();
    assert_eq!(error.class(), ErrorClass::RuntimeError, "{error}");
    drop(server);
}

/// Only `predict_batch`: `predict` is a batch of one.
struct BatchOnly {
    log: Arc<Mutex<Log>>,
}

impl Policy for BatchOnly {
    const CORNERS: Corners = Corners::PREDICT_BATCH;

    fn predict_batch(&mut self, observations: Vec<Value>) -> Result<Vec<Value>> {
        self.log
            .lock()
            .unwrap()
            .batch_sizes
            .push(observations.len());
        Ok(observations.iter().map(|_| Value::Int(1)).collect())
    }
}

#[test]
fn predict_derives_from_predict_batch_as_a_batch_of_one() {
    let (server, address) = served_env();
    let log = Arc::new(Mutex::new(Log::default()));
    let model = Model::new(
        BatchOnly {
            log: Arc::clone(&log),
        },
        ModelOptions::default(),
    )
    .unwrap();
    run_model_local(&model, RunLocalOptions::parse(&address).unwrap()).unwrap();
    assert_eq!(log.lock().unwrap().batch_sizes, vec![1, 1, 1]);
    drop(server);
}

#[test]
fn a_closure_is_a_policy_and_with_context_reads_the_context() {
    let (server, address) = served_env();
    let greedy = Model::new(|_obs: Value| Ok(Value::Int(0)), ModelOptions::default()).unwrap();
    run_model_local(&greedy, RunLocalOptions::parse(&address).unwrap()).unwrap();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&seen);
    let model = Model::new(
        with_context(move |_obs: Value, context: &mut PredictContext<'_>| {
            sink.lock().unwrap().push(context.predict_index);
            Ok(Value::Int(1))
        }),
        ModelOptions::default(),
    )
    .unwrap();
    run_model_local(&model, RunLocalOptions::parse(&address).unwrap()).unwrap();
    assert_eq!(*seen.lock().unwrap(), vec![0, 1, 2]);
    drop(server);
}

#[test]
fn options_hooks_replace_the_policy_hooks() {
    let (server, address) = served_env();
    let log = Arc::new(Mutex::new(Log::default()));
    let ended = Arc::new(AtomicUsize::new(0));
    let closed = Arc::new(AtomicUsize::new(0));
    let (ended_hook, closed_hook) = (Arc::clone(&ended), Arc::clone(&closed));
    let model = Model::new(
        Counting {
            log: Arc::clone(&log),
        },
        ModelOptions {
            on_episode_end: Some(Box::new(move |_id: &str| {
                ended_hook.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })),
            on_close: Some(Box::new(move || {
                closed_hook.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })),
            ..Default::default()
        },
    )
    .unwrap();
    run_model_local(&model, RunLocalOptions::parse(&address).unwrap()).unwrap();
    drop(model);
    let log = log.lock().unwrap();
    assert!(
        log.ended.is_empty() && log.closes == 0,
        "the policy's hooks are replaced"
    );
    assert_eq!(ended.load(Ordering::SeqCst), 1);
    assert_eq!(closed.load(Ordering::SeqCst), 1, "the last drop closes");
    drop(server);
}

struct ChunkOnly;
impl Policy for ChunkOnly {
    const CORNERS: Corners = Corners::PREDICT_CHUNK;
}

struct ContextNotListed;
impl Policy for ContextNotListed {
    const CONTEXT: Corners = Corners::PREDICT_BATCH;
}

struct StaticChunkNoCorner;
impl Policy for StaticChunkNoCorner {
    const NATIVE_CHUNK: Option<u32> = Some(4);
}

struct ZeroChunk;
impl Policy for ZeroChunk {
    const CORNERS: Corners = Corners::PREDICT.union(Corners::PREDICT_CHUNK);
    const NATIVE_CHUNK: Option<u32> = Some(0);
}

/// Sets a K in `load` without a chunk corner; its `close` must never run.
struct LoadedChunk {
    closes: Arc<AtomicUsize>,
}
impl Policy for LoadedChunk {
    fn native_chunk(&self) -> Option<u32> {
        Some(2)
    }
    fn close(&mut self) -> Result<()> {
        self.closes.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[test]
fn invalid_declarations_are_refused_at_construction() {
    let class = |result: Result<Model>| result.unwrap_err().class();
    assert_eq!(
        class(Model::new(ChunkOnly, ModelOptions::default())),
        ErrorClass::TypeError
    );
    assert_eq!(
        class(Model::new(ContextNotListed, ModelOptions::default())),
        ErrorClass::TypeError
    );
    assert_eq!(
        class(Model::new(StaticChunkNoCorner, ModelOptions::default())),
        ErrorClass::TypeError
    );
    assert_eq!(
        class(Model::new(ZeroChunk, ModelOptions::default())),
        ErrorClass::ValueError
    );
    let closes = Arc::new(AtomicUsize::new(0));
    assert_eq!(
        class(Model::new(
            LoadedChunk {
                closes: Arc::clone(&closes)
            },
            ModelOptions::default()
        )),
        ErrorClass::TypeError
    );
    assert_eq!(
        closes.load(Ordering::SeqCst),
        0,
        "a failed constructor runs no hook"
    );
}

#[derive(Default)]
struct Keyed {
    checkpoint: Option<String>,
}

impl Policy for Keyed {
    const LOAD_KEYS: Option<&'static [LoadKey]> =
        Some(&[LoadKey::required("checkpoint"), LoadKey::optional("device")]);

    fn load(&mut self, config: &Kwargs) -> Result<()> {
        self.checkpoint = Some(config.get("checkpoint").unwrap().as_text()?.to_string());
        Ok(())
    }

    fn predict(&mut self, _observation: Value) -> Result<Value> {
        Ok(Value::Int(0))
    }
}

#[derive(Default)]
struct NoLoad;
impl Policy for NoLoad {}

fn config(pairs: &[(&str, &str)]) -> Kwargs {
    pairs
        .iter()
        .map(|(key, value)| (key.to_string(), Value::Text(value.to_string())))
        .collect()
}

#[test]
fn load_keys_are_checked_in_pythons_order() {
    let model = Model::from_config::<Keyed>(config(&[("checkpoint", "a.pt")])).unwrap();
    drop(model);
    let error = Model::from_config::<Keyed>(config(&[("zeta", "1"), ("alpha", "2")])).unwrap_err();
    assert_eq!(error.class(), ErrorClass::ValueError);
    assert_eq!(
        error.to_string(),
        "missing required parameter: 'checkpoint'"
    );
    let error = Model::from_config::<Keyed>(config(&[
        ("checkpoint", "a.pt"),
        ("zeta", "1"),
        ("alpha", "2"),
    ]))
    .unwrap_err();
    assert_eq!(error.class(), ErrorClass::ValueError);
    assert_eq!(
        error.to_string(),
        "extra='passthrough' but load has no **kwargs to forward: alpha, zeta"
    );
    let error = Model::new(Keyed::default(), ModelOptions::default()).unwrap_err();
    assert_eq!(error.class(), ErrorClass::TypeError);
    assert!(error.to_string().contains("'checkpoint'"), "{error}");
}

#[test]
fn the_default_load_accepts_only_an_empty_config() {
    Model::from_config::<NoLoad>(Kwargs::new()).unwrap();
    let error = Model::from_config::<NoLoad>(config(&[("b", "1"), ("a", "2")])).unwrap_err();
    assert_eq!(error.class(), ErrorClass::TypeError);
    assert_eq!(
        error.to_string(),
        "model received construction params (a, b) but does not override load(...) to apply \
         them; implement load(**kwargs) to consume the binding"
    );
}

#[test]
fn an_unwritten_listed_corner_fails_with_the_runtime_error_class() {
    let (server, address) = served_env();
    let model = Model::new(NoLoad, ModelOptions::default()).unwrap();
    let error = run_model_local(&model, RunLocalOptions::parse(&address).unwrap()).unwrap_err();
    assert_eq!(error.class(), ErrorClass::RuntimeError, "{error}");
    assert!(error.to_string().contains("Corners::PREDICT"), "{error}");
    drop(server);
}

#[test]
fn a_corner_that_re_enters_its_own_model_fails_instead_of_deadlocking() {
    let (server, address) = served_env();
    let slot: Arc<Mutex<Option<Model>>> = Arc::new(Mutex::new(None));
    let seen = Arc::new(Mutex::new(None));
    let (inner, sink) = (Arc::clone(&slot), Arc::clone(&seen));
    let model = Model::new(
        move |_obs: Value| {
            let model = inner.lock().unwrap().clone();
            if let Some(model) = model {
                *sink.lock().unwrap() = Some(model.close().unwrap_err().class());
                drop(model);
            }
            Ok(Value::Int(0))
        },
        ModelOptions::default(),
    )
    .unwrap();
    *slot.lock().unwrap() = Some(model.clone());
    run_model_local(&model, RunLocalOptions::parse(&address).unwrap()).unwrap();
    assert_eq!(*seen.lock().unwrap(), Some(ErrorClass::RuntimeError));
    slot.lock().unwrap().take();
    drop(server);
}

const GRIPPER_TAGS: &str = r#"{
    "observation": {"state": {"type": "state", "role": "proprio/gripper"}},
    "action": {"components": [{"role": "action/gripper", "dim": 1, "range": [0.0, 10.0]}]}
}"#;
const GRIPPER_SPEC: &str = r#"{
    "input": {"state": {"type": "state", "container": "list", "dtype": "float32",
                        "components": [{"role": "proprio/gripper", "dim": 1}]}},
    "output": {"components": [{"role": "action/gripper", "dim": 1, "range": [0.0, 10.0]}]}
}"#;

/// One gripper state in, one gripper action out, three steps.
struct Gripper {
    step: i64,
}

impl Env for Gripper {
    fn observation_space(&self) -> Space {
        let state =
            crate::spaces::Box::new(0.0, 10.0, [1], crate::spaces::BoxOptions::default()).unwrap();
        crate::spaces::Dict::new([("state", state)]).unwrap().into()
    }
    fn action_space(&self) -> Space {
        crate::spaces::Box::new(0.0, 10.0, [1], crate::spaces::BoxOptions::default())
            .unwrap()
            .into()
    }
    fn reset(
        &mut self,
        _seed: Option<i64>,
        _options: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        self.step = 0;
        Ok((self.observation_space().sample()?, InfoDict::new()))
    }
    fn step(&mut self, _action: Option<Value>) -> Result<(Value, f64, bool, bool, InfoDict)> {
        self.step += 1;
        let observation = self.observation_space().sample()?;
        Ok((observation, 1.0, self.step >= 4, false, InfoDict::new()))
    }
}

fn served_gripper() -> (EnvServer, String) {
    let server = EnvServer::new(
        Gripper { step: 0 },
        EnvServerOptions {
            address: Some("tcp://127.0.0.1:0".into()),
            tags: Some(crate::ext::env_tags_from_json(GRIPPER_TAGS).unwrap()),
            options: Some(crate::ServeOptions {
                drain_timeout: Some(Duration::from_millis(100)),
                ..Default::default()
            }),
            ..Default::default()
        },
    )
    .unwrap();
    server.start().unwrap();
    let address = server.address().to_string();
    (server, address)
}

/// A chunked spec'd policy: two frames per chunk, declared.
struct Planner {
    chunks: Arc<AtomicUsize>,
}

impl Policy for Planner {
    const CORNERS: Corners = Corners::PREDICT.union(Corners::PREDICT_CHUNK);
    const NATIVE_CHUNK: Option<u32> = Some(2);

    fn spec() -> Option<SpecArg> {
        Some(
            crate::ext::model_spec_from_json(GRIPPER_SPEC)
                .unwrap()
                .into(),
        )
    }

    fn predict(&mut self, _observation: Value) -> Result<Value> {
        Ok(Value::List(vec![Value::Float(1.0)]))
    }

    fn predict_chunk(&mut self, observation: Value, execution_horizon: u32) -> Result<Value> {
        assert_eq!(execution_horizon, 2);
        assert!(
            observation.as_dict().is_ok(),
            "the adapted input: {observation:?}"
        );
        self.chunks.fetch_add(1, Ordering::SeqCst);
        let frame = Value::List(vec![Value::Float(2.0)]);
        Ok(Value::List(vec![frame.clone(), frame]))
    }
}

#[test]
fn a_spec_model_resolves_its_adapter_and_replays_its_chunks() {
    let (server, address) = served_gripper();
    let chunks = Arc::new(AtomicUsize::new(0));
    let model = Model::new(
        Planner {
            chunks: Arc::clone(&chunks),
        },
        ModelOptions::default(),
    )
    .unwrap();
    assert!(matches!(model.spec(), Some(SpecArg::ModelSpec(_))));
    assert_eq!(model.native_chunk(), Some(2));
    let mut options = RunLocalOptions::parse(&address).unwrap();
    options = options.execution_horizon(2);
    run_model_local(&model, options).unwrap();
    // Four steps at two frames per chunk.
    assert_eq!(chunks.load(Ordering::SeqCst), 2);

    // A tagged env with no spec is refused at resolve.
    let plain = Model::new(
        |_obs: Value| Ok(Value::List(vec![Value::Float(0.0)])),
        ModelOptions::default(),
    )
    .unwrap();
    let error = run_model_local(&plain, RunLocalOptions::parse(&address).unwrap()).unwrap_err();
    assert!(
        error.class().is_subclass_of(ErrorClass::ValueError),
        "{error}"
    );
    assert!(error.to_string().contains("spec=None"), "{error}");
    // NO_ADAPTER skips resolution: the model sees the raw observation.
    let raw = Model::new(
        |obs: Value| {
            assert!(obs.as_dict().is_ok());
            Ok(Value::List(vec![Value::Float(0.0)]))
        },
        ModelOptions {
            spec: Some(NO_ADAPTER),
            ..Default::default()
        },
    )
    .unwrap();
    run_model_local(&raw, RunLocalOptions::parse(&address).unwrap()).unwrap();
    drop(server);
}

/// Returns three frames though it declares two.
struct Liar;

impl Policy for Liar {
    const CORNERS: Corners = Corners::PREDICT.union(Corners::PREDICT_CHUNK);
    const NATIVE_CHUNK: Option<u32> = Some(2);

    fn spec() -> Option<SpecArg> {
        Some(
            crate::ext::model_spec_from_json(GRIPPER_SPEC)
                .unwrap()
                .into(),
        )
    }

    fn predict(&mut self, _observation: Value) -> Result<Value> {
        Ok(Value::List(vec![Value::Float(1.0)]))
    }

    fn predict_chunk(&mut self, _observation: Value, _execution_horizon: u32) -> Result<Value> {
        let frame = Value::List(vec![Value::Float(2.0)]);
        Ok(Value::List(vec![frame.clone(), frame.clone(), frame]))
    }
}

#[test]
fn a_chunk_of_the_wrong_length_is_a_value_error() {
    let (server, address) = served_gripper();
    let model = Model::new(Liar, ModelOptions::default()).unwrap();
    let error = run_model_local(
        &model,
        RunLocalOptions::parse(&address)
            .unwrap()
            .execution_horizon(2),
    )
    .unwrap_err();
    assert!(error.to_string().contains("native_chunk=2"), "{error}");
    drop(server);
}

/// A free loopback port.
fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn quick_serve(cancel: Option<crate::ext::CancelToken>) -> ModelServeOptions {
    ModelServeOptions {
        options: Some(crate::ServeOptions {
            drain_timeout: Some(Duration::from_millis(100)),
            ..Default::default()
        }),
        cancel,
        ..Default::default()
    }
}

#[test]
fn a_cancelled_serve_closes_the_model_and_fails_with_the_cancellation_class() {
    let log = Arc::new(Mutex::new(Log::default()));
    let model = Model::new(
        Counting {
            log: Arc::clone(&log),
        },
        ModelOptions::default(),
    )
    .unwrap();
    // A token cancelled before serve fails it at once and closes nothing.
    let cancel = crate::ext::CancelToken::new();
    cancel.cancel();
    let address = format!("tcp://127.0.0.1:{}", free_port());
    let error = model
        .serve(&address, quick_serve(Some(cancel.clone())))
        .unwrap_err();
    assert_eq!(error, crate::Error::Cancelled);
    assert_eq!(log.lock().unwrap().closes, 0);

    let serving = model.clone();
    let token = cancel.clone();
    let thread = std::thread::spawn(move || serving.serve(&address, quick_serve(Some(token))));
    std::thread::sleep(Duration::from_millis(300));
    cancel.cancel();
    let result = thread.join().unwrap();
    assert_eq!(result.unwrap_err(), crate::Error::Cancelled);
    assert_eq!(log.lock().unwrap().closes, 1, "the serve-stop close ran");
    let error = model
        .serve(
            &format!("tcp://127.0.0.1:{}", free_port()),
            quick_serve(None),
        )
        .unwrap_err();
    assert_eq!(error.class(), ErrorClass::RuntimeError);
}

#[test]
fn close_stops_a_running_serve_which_then_returns_ok() {
    let log = Arc::new(Mutex::new(Log::default()));
    let model = Model::new(
        Counting {
            log: Arc::clone(&log),
        },
        ModelOptions::default(),
    )
    .unwrap();
    let serving = model.clone();
    let address = format!("tcp://127.0.0.1:{}", free_port());
    let thread = std::thread::spawn(move || serving.serve(&address, quick_serve(None)));
    std::thread::sleep(Duration::from_millis(300));
    model.close().unwrap();
    thread.join().unwrap().unwrap();
    assert_eq!(log.lock().unwrap().closes, 1);
}

#[test]
fn serve_refuses_a_zero_predict_concurrency() {
    let model = Model::new(|_obs: Value| Ok(Value::Int(0)), ModelOptions::default()).unwrap();
    let error = model
        .serve(
            "tcp://127.0.0.1:0",
            ModelServeOptions {
                ext: Some(crate::ext::ModelServeOptions {
                    predict_concurrency: Some(0),
                    ..Default::default()
                }),
                ..Default::default()
            },
        )
        .unwrap_err();
    assert_eq!(error.class(), ErrorClass::ValueError);
}

#[tokio::test(flavor = "multi_thread")]
async fn sync_calls_from_async_code_name_their_twin() {
    // A task (the test's root future is not one, R-16).
    tokio::spawn(async {
        let model = Model::new(|_obs: Value| Ok(Value::Int(0)), ModelOptions::default()).unwrap();
        let error = model
            .serve("tcp://127.0.0.1:0", ModelServeOptions::default())
            .unwrap_err();
        assert!(error.to_string().contains("serve_async"), "{error}");
        let error = model.close().unwrap_err();
        assert!(error.to_string().contains("close_async"), "{error}");
        model.close_async().await.unwrap();
    })
    .await
    .unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn dropping_a_polled_serve_async_future_closes_the_model() {
    let closed = Arc::new(AtomicUsize::new(0));
    let hook = Arc::clone(&closed);
    let model = Model::new(
        |_obs: Value| Ok(Value::Int(0)),
        ModelOptions {
            on_close: Some(Box::new(move || {
                hook.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })),
            ..Default::default()
        },
    )
    .unwrap();
    let address = format!("tcp://127.0.0.1:{}", free_port());
    let serve = model.serve_async(&address, quick_serve(None));
    let _ = tokio::time::timeout(Duration::from_millis(300), serve).await;
    model.close_async().await.unwrap();
    assert_eq!(closed.load(Ordering::SeqCst), 1);
}
