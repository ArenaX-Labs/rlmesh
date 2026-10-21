#![cfg(feature = "lockstep")]
//! v1 and v2 trace equivalence (§12.4, the wave 5 release gate's
//! "semantically equivalent" half).
//!
//! The same catalog-style env and the same model run once over v1 (the env
//! behind the gRPC Join stream) and once over v2 (the env as a lockstep
//! robot in a session, at its participant address), and every trace must match:
//!
//! - what the env received and answered, per lane: reset seeds and options,
//!   the exact action bytes, the observation leaf bytes, rewards (bit
//!   patterns) and the done flags;
//! - what the model's `predict` corners received after the v1 adapter
//!   engine (`ModelSpec` resolved against the env's v1 tags: image resize
//!   and layout, rotation conversion, text) and what they returned;
//! - every step event (action, reward bits, flags, info) and episode result
//!   (seed, steps, reward bits, termination, success).
//!
//! The shapes: a scalar env with chunking (execution horizon 3) and
//! prefetch; a 4-lane vector env with NEXT_STEP autoreset and one batched
//! predict per step; a 3-lane lane endpoint with DISABLED autoreset, so
//! the driver resets lanes one by one (partial resets, per-lane seeds) and
//! chunks per lane; and the managed shape, a served (gRPC) model server
//! driven by the v1 runtime over a lockstep env, unchanged.

#![allow(clippy::unwrap_used, clippy::print_stderr)]

mod common;

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::{CATALOG_MODEL_SPEC, CatalogEnv, Trace, mix};
use rlmesh::adapters::ModelSpec;
use rlmesh::prelude::*;
use rlmesh::{DType, EpisodeResult, Model, ModelOptions, RunHooks, RunOptions, StepEvent, Tensor};
use rlmesh_spaces::AutoresetMode;

fn fnv(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
    })
}

/// A deterministic model: its action is a hash of what it was given.
struct CatalogPolicy {
    log: Arc<Mutex<Vec<String>>>,
}

impl CatalogPolicy {
    fn action(&self, observation: &Value, frames: usize) -> rlmesh::Result<Value> {
        let seen = format!("{observation:?}");
        let h = fnv(&seen);
        let values: Vec<f32> = (0..frames * 7)
            .map(|i| ((mix(h ^ i as u64) % 2400) as f32 / 1000.0) - 1.2)
            .collect();
        self.log
            .lock()
            .unwrap()
            .push(format!("{frames}x {h:016x} -> {values:?}"));
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let shape: &[usize] = if frames == 1 { &[7] } else { &[frames, 7] };
        Ok(Value::Tensor(Tensor::new(&bytes, shape, DType::Float32)?))
    }
}

impl Policy for CatalogPolicy {
    const CORNERS: Corners = Corners::PREDICT
        .union(Corners::PREDICT_CHUNK)
        .union(Corners::PREDICT_BATCH);

    fn predict(&mut self, observation: Value) -> rlmesh::Result<Value> {
        self.action(&observation, 1)
    }

    fn predict_chunk(&mut self, observation: Value, horizon: u32) -> rlmesh::Result<Value> {
        self.action(&observation, horizon as usize)
    }

    fn predict_batch(&mut self, observations: Vec<Value>) -> rlmesh::Result<Vec<Value>> {
        self.log
            .lock()
            .unwrap()
            .push(format!("batch of {}", observations.len()));
        observations.iter().map(|o| self.action(o, 1)).collect()
    }
}

#[derive(Default)]
struct Steps {
    events: Vec<(Option<i64>, i64, String)>,
    episodes: Vec<String>,
}

struct Hooks(Arc<Mutex<Steps>>);

impl RunHooks for Hooks {
    fn on_step(&mut self, event: &StepEvent) -> rlmesh::Result<()> {
        self.0.lock().unwrap().events.push((
            event.seed,
            event.step,
            format!(
                "obs={:016x} action={:?} reward={:016x} term={} trunc={} info={:?}",
                fnv(&format!("{:?}", event.observation)),
                event.action,
                event.reward.to_bits(),
                event.terminated,
                event.truncated,
                event.info
            ),
        ));
        Ok(())
    }

    fn on_episode_end(&mut self, result: &EpisodeResult) -> rlmesh::Result<()> {
        self.0.lock().unwrap().episodes.push(format!(
            "seed={:?} steps={} reward={:016x} term={} trunc={} success={:?}",
            result.seed,
            result.steps,
            result.reward.to_bits(),
            result.terminated,
            result.truncated,
            result.success
        ));
        Ok(())
    }
}

/// Everything one run left behind.
#[derive(Debug, PartialEq)]
struct Traces {
    env: BTreeMap<usize, Vec<String>>,
    model: Vec<String>,
    steps: Vec<(Option<i64>, i64, String)>,
    episodes: Vec<String>,
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap()
}

/// Serves `env` over v1 (gRPC) or v2 (lockstep) and returns its address.
fn serve(rt: &tokio::runtime::Runtime, env: CatalogEnv, v2: bool) -> String {
    if v2 {
        let binding = rt.block_on(rlmesh::lockstep::bind(":0/catalog")).unwrap();
        let address = binding.address();
        rt.spawn(binding.serve(
            Arc::new(env),
            rlmesh::lockstep::ServeOptions::default(),
            tokio_util::sync::CancellationToken::new(),
        ));
        return address;
    }
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let addr: std::net::SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
    rt.spawn(rlmesh_grpc::env::serve(env, addr));
    std::thread::sleep(Duration::from_millis(200));
    format!("tcp://127.0.0.1:{port}")
}

fn options(env: &CatalogEnv) -> Trace {
    env.trace.clone()
}

struct Shape {
    num_envs: u32,
    autoreset: AutoresetMode,
    lanes: bool,
    episodes: u64,
    seeds: Option<Vec<i64>>,
    horizon: u32,
    prefetch: u64,
    /// Lane-order-free comparison (a lane endpoint's lanes race).
    unordered: bool,
}

fn run_once(shape: &Shape, v2: bool) -> Traces {
    let rt = runtime();
    let env = CatalogEnv::new(shape.num_envs, shape.autoreset, shape.lanes);
    let trace = options(&env);
    let address = serve(&rt, env, v2);
    let log = Arc::new(Mutex::new(Vec::new()));
    let model = Model::new(
        CatalogPolicy { log: log.clone() },
        ModelOptions {
            spec: Some(ModelSpec::from_json(CATALOG_MODEL_SPEC).unwrap().into()),
            ..Default::default()
        },
    )
    .unwrap();
    let steps = Arc::new(Mutex::new(Steps::default()));
    let mut hooks = Hooks(steps.clone());
    let result = model
        .run(
            address.as_str(),
            RunOptions {
                episodes: Some(shape.episodes),
                seeds: shape.seeds.clone(),
                execution_horizon: shape.horizon,
                prefetch_lead: shape.prefetch,
                hooks: Some(&mut hooks),
                workflow_edition: Some("2026.06".into()),
                ..Default::default()
            },
        )
        .unwrap();
    model.close().unwrap();
    assert_eq!(result.num_episodes() as u64, shape.episodes);
    drop(rt);
    let mut steps = std::mem::take(&mut *steps.lock().unwrap());
    let mut model_log = std::mem::take(&mut *log.lock().unwrap());
    if shape.unordered {
        steps.events.sort();
        steps.episodes.sort();
        model_log.retain(|l| !l.starts_with("batch of"));
        model_log.sort();
    }
    let mut env = std::mem::take(&mut *trace.lock().unwrap());
    if shape.unordered {
        // Which lane runs which seed is a race between lanes that end on
        // the same step: compare the multiset of episode transcripts.
        let mut episodes: Vec<String> = env
            .values()
            .flat_map(|lines| {
                let mut out: Vec<Vec<String>> = Vec::new();
                for line in lines {
                    if line.starts_with("reset") || out.is_empty() {
                        out.push(Vec::new());
                    }
                    out.last_mut().unwrap().push(line.clone());
                }
                out.into_iter().map(|e| e.join("\n"))
            })
            .collect();
        episodes.sort();
        env = episodes
            .into_iter()
            .enumerate()
            .map(|(i, e)| (i, vec![e]))
            .collect();
    }
    Traces {
        env,
        model: model_log,
        steps: steps.events,
        episodes: steps.episodes,
    }
}

fn assert_equivalent(shape: &Shape) {
    let v1 = run_once(shape, false);
    let v2 = run_once(shape, true);
    assert!(!v1.steps.is_empty() && !v1.model.is_empty());
    for (lane, lines) in &v1.env {
        assert_eq!(Some(lines), v2.env.get(lane), "env trace of lane {lane}");
    }
    assert_eq!(v1.env.len(), v2.env.len());
    assert_eq!(v1.model, v2.model, "model trace");
    assert_eq!(v1.steps, v2.steps, "step events");
    assert_eq!(v1.episodes, v2.episodes, "episode results");
}

#[test]
fn a_scalar_env_with_chunking_and_prefetch_runs_unchanged_on_lockstep() {
    assert_equivalent(&Shape {
        num_envs: 1,
        autoreset: AutoresetMode::Disabled,
        lanes: false,
        episodes: 3,
        seeds: Some(vec![11, -12, 13]),
        horizon: 3,
        prefetch: 1,
        unordered: false,
    });
}

#[test]
fn a_next_step_vector_env_with_batched_predict_runs_unchanged_on_lockstep() {
    assert_equivalent(&Shape {
        num_envs: 4,
        autoreset: AutoresetMode::NextStep,
        lanes: false,
        episodes: 10,
        seeds: None,
        horizon: 1,
        prefetch: 0,
        unordered: false,
    });
}

#[test]
fn a_lane_endpoint_with_partial_resets_and_per_lane_seeds_runs_unchanged_on_lockstep() {
    assert_equivalent(&Shape {
        num_envs: 3,
        autoreset: AutoresetMode::Disabled,
        lanes: true,
        episodes: 7,
        seeds: Some(vec![21, 22, 23, 24, 25, 26, 27]),
        horizon: 2,
        prefetch: 0,
        unordered: true,
    });
}

// --- the managed shape: a served model server, driven by the v1 runtime ----

use async_trait::async_trait;
use rlmesh_proto::model::v1::{
    AdapterContext, PredictRequest, ReleaseAdapterRequest, ResetAdapterRequest,
    ResolveAdapterRequest,
};
use rlmesh_runtime::{
    NoopRuntimeHooks, RuntimeDriver, RuntimeEnv, RuntimeError, RuntimeModel,
    RuntimeModelPrediction, RuntimeSessionSpec,
};

/// A served model's gRPC client as a runtime model (the managed runner's
/// unbatched route).
struct ServedModel {
    client: tokio::sync::Mutex<rlmesh_grpc::ModelClient>,
}

#[async_trait]
impl RuntimeModel for ServedModel {
    async fn predict(
        &self,
        request: PredictRequest,
    ) -> std::result::Result<RuntimeModelPrediction, RuntimeError> {
        let mut client = self.client.lock().await;
        let response = client
            .predict(request)
            .await
            .map_err(|source| RuntimeError::model_rpc("served-model", source))?;
        Ok(RuntimeModelPrediction {
            response,
            endpoint_total_ns: client.take_last_endpoint_total_ns(),
            phases: client.take_last_phases(),
            group_size: None,
        })
    }

    async fn reset_adapter(
        &self,
        request: ResetAdapterRequest,
    ) -> std::result::Result<(), RuntimeError> {
        self.client
            .lock()
            .await
            .reset_adapter(request)
            .await
            .map_err(|source| RuntimeError::model_rpc("served-model", source))
    }

    async fn release_adapter(
        &self,
        request: ReleaseAdapterRequest,
        _timeout: Duration,
    ) -> std::result::Result<(), String> {
        self.client
            .lock()
            .await
            .release_adapter(request)
            .await
            .map(|_| ())
            .map_err(|err| err.to_string())
    }
}

async fn drive_managed<E: RuntimeEnv + Clone + 'static>(
    env: E,
    handshake: rlmesh_grpc::EnvHandshake,
    mut model: rlmesh_grpc::ModelClient,
    selected: String,
    episodes: u64,
) -> rlmesh_runtime::RuntimeReport {
    model
        .resolve_adapter(ResolveAdapterRequest {
            context: Some(AdapterContext {
                session_id: "managed".into(),
                env_id: "catalog".into(),
                request_id: "managed:resolve_adapter".into(),
            }),
            env_spec: handshake.env_contract.spec.clone(),
            selected_workflow_edition: selected.clone(),
            execution_horizon: 1,
            delivers_history: true,
        })
        .await
        .unwrap();
    let spec = RuntimeSessionSpec {
        session_id: "managed".into(),
        env_id: "catalog".into(),
        env_component_id: "catalog-env".into(),
        model_component_id: "served-model".into(),
        workflow_edition: rlmesh_proto::parse_retained_edition(&selected).unwrap(),
        env_contract: handshake.env_contract.clone(),
        num_envs: handshake.num_envs,
        base_seed: Some(5),
        episode_seeds: Vec::new(),
        max_episodes: Some(episodes),
        trial_index_base: None,
        max_episode_steps: None,
        max_episode_seconds: None,
        close_env_on_end: false,
        limits: Default::default(),
        subset_step: false,
        env_ceiling: None,
        model_ceiling: None,
    };
    RuntimeDriver::new(
        spec,
        env,
        ServedModel {
            client: tokio::sync::Mutex::new(model),
        },
        Arc::new(NoopRuntimeHooks),
    )
    .run()
    .await
    .unwrap()
}

fn run_managed(v2: bool) -> Traces {
    let rt = runtime();
    let env = CatalogEnv::new(4, AutoresetMode::NextStep, false);
    let trace = options(&env);
    let env_address = serve(&rt, env, v2);
    let log = Arc::new(Mutex::new(Vec::new()));
    let model = Model::new(
        CatalogPolicy { log: log.clone() },
        ModelOptions {
            spec: Some(ModelSpec::from_json(CATALOG_MODEL_SPEC).unwrap().into()),
            ..Default::default()
        },
    )
    .unwrap();
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let model_address = format!("tcp://127.0.0.1:{port}");
    let cancel = rlmesh::ext::CancelToken::new();
    let server = {
        let (model, address, cancel) = (model.clone(), model_address.clone(), cancel.clone());
        std::thread::spawn(move || {
            let _ = model.serve(
                &address,
                rlmesh::ModelServeOptions {
                    cancel: Some(cancel),
                    ..Default::default()
                },
            );
        })
    };
    let report = rt.block_on(async {
        let connect = rlmesh_grpc::connect::ConnectOptions::default();
        let mut client = rlmesh_grpc::ModelClient::connect_with_retry(&model_address, "", &connect)
            .await
            .unwrap();
        client.handshake().await.unwrap();
        if v2 {
            let (env, handshake) =
                rlmesh::binding::model::LockstepRuntimeEnv::connect(&env_address, Some("2026.06"))
                    .await
                    .unwrap();
            let floor = rlmesh_grpc::env_floor(
                &handshake.session_offer(),
                &client.model_session_offer(),
                Some("2026.06"),
            )
            .unwrap();
            drive_managed(env, handshake, client, floor.selected_workflow_edition, 9).await
        } else {
            let mut env = rlmesh_grpc::EnvClient::connect_with_retry(&env_address, "", &connect)
                .await
                .unwrap();
            env.declare_workflow_edition(Some("2026.06".into()));
            let handshake = env.handshake().await.unwrap();
            let floor = rlmesh_grpc::env_floor(
                &handshake.session_offer(),
                &client.model_session_offer(),
                Some("2026.06"),
            )
            .unwrap();
            env.pin_workflow_edition(floor.selected_workflow_edition.clone());
            let env = rlmesh::binding::model::EnvClientRuntimeEnv::new(env);
            drive_managed(env, handshake, client, floor.selected_workflow_edition, 9).await
        }
    });
    cancel.cancel();
    let _ = server.join();
    model.close().unwrap();
    drop(rt);
    let env = std::mem::take(&mut *trace.lock().unwrap());
    let model_log = std::mem::take(&mut *log.lock().unwrap());
    Traces {
        env,
        model: model_log,
        steps: Vec::new(),
        episodes: report
            .episodes
            .iter()
            .map(|e| {
                format!(
                    "lane={} seed={:?} steps={} reward={:016x} term={} trunc={} success={:?}",
                    e.env_index,
                    e.seed,
                    e.step_count,
                    e.cumulative_reward.to_bits(),
                    e.terminated,
                    e.truncated,
                    e.success
                )
            })
            .collect(),
    }
}

#[test]
fn a_served_model_server_drives_a_lockstep_env_unchanged() {
    let v1 = run_managed(false);
    let v2 = run_managed(true);
    assert!(!v1.model.is_empty() && v1.episodes.len() >= 9);
    assert_eq!(v1.env, v2.env, "env trace");
    assert_eq!(v1.model, v2.model, "model trace");
    assert_eq!(v1.episodes, v2.episodes, "episode results");
}
