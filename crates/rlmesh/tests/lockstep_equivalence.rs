#![cfg(feature = "lockstep")]
//! v1 and v2 trace equivalence (§12.4, the wave 5 release gate's
//! "semantically equivalent" half).
//!
//! A catalog-style env and a model run over v2 (the env as a lockstep robot
//! in a session, at its participant address), and every trace must match
//! what the v1 runtime left for the same env and model (the env behind the
//! gRPC Join stream), kept as golden files in `tests/golden/lockstep`:
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
//! chunks per lane; and the managed shape, a model served on a session
//! driving a lockstep env as v1's gRPC model server did.

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

/// Serves `env` as a lockstep robot and returns its address.
fn serve(rt: &tokio::runtime::Runtime, env: CatalogEnv) -> String {
    let binding = rt.block_on(rlmesh::lockstep::bind(":0/catalog")).unwrap();
    let address = binding.address();
    rt.spawn(binding.serve(
        Arc::new(env),
        rlmesh::lockstep::ServeOptions::default(),
        tokio_util::sync::CancellationToken::new(),
    ));
    address
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

fn run_once(shape: &Shape) -> Traces {
    let rt = runtime();
    let env = CatalogEnv::new(shape.num_envs, shape.autoreset, shape.lanes);
    let trace = options(&env);
    let address = serve(&rt, env);
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

/// Checks `traces` against the golden file `name`: the traces the v1
/// runtime left before it was deleted.
fn golden(name: &str, traces: &Traces) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden/lockstep")
        .join(format!("{name}.txt"));
    let kept = std::fs::read_to_string(&path).unwrap();
    assert!(
        kept == format!("{traces:#?}\n"),
        "traces differ from {}",
        path.display()
    );
}

fn assert_equivalent(name: &str, shape: &Shape) {
    let traces = run_once(shape);
    assert!(!traces.steps.is_empty() && !traces.model.is_empty());
    golden(name, &traces);
}

#[test]
fn a_scalar_env_with_chunking_and_prefetch_runs_unchanged_on_lockstep() {
    assert_equivalent(
        "scalar_chunked",
        &Shape {
            num_envs: 1,
            autoreset: AutoresetMode::Disabled,
            lanes: false,
            episodes: 3,
            seeds: Some(vec![11, -12, 13]),
            horizon: 3,
            prefetch: 1,
            unordered: false,
        },
    );
}

#[test]
fn a_next_step_vector_env_with_batched_predict_runs_unchanged_on_lockstep() {
    assert_equivalent(
        "vector_next_step",
        &Shape {
            num_envs: 4,
            autoreset: AutoresetMode::NextStep,
            lanes: false,
            episodes: 10,
            seeds: None,
            horizon: 1,
            prefetch: 0,
            unordered: false,
        },
    );
}

#[test]
fn a_lane_endpoint_with_partial_resets_and_per_lane_seeds_runs_unchanged_on_lockstep() {
    assert_equivalent(
        "lane_endpoint",
        &Shape {
            num_envs: 3,
            autoreset: AutoresetMode::Disabled,
            lanes: true,
            episodes: 7,
            seeds: Some(vec![21, 22, 23, 24, 25, 26, 27]),
            horizon: 2,
            prefetch: 0,
            unordered: true,
        },
    );
}

// --- the managed shape: a model served on a session, driven by the runtime -

use rlmesh::binding::model::{LockstepEnv, ResolveOptions};
use rlmesh::binding::runtime::{NoopRuntimeHooks, RuntimeDriver, RuntimeSessionSpec};
use rlmesh::lockstep::{ModelBinding, ModelService, ServedModel};

/// The managed runner's route: the runtime drives the lockstep env at
/// `env_address` with the model served at `model_address`, under the
/// session id, env id and base seed the v1 runtime ran it with.
async fn drive_managed(
    env_address: &str,
    model_address: &str,
    episodes: u64,
) -> rlmesh::binding::runtime::RuntimeReport {
    let client = rlmesh::lockstep::connect(env_address, Default::default())
        .await
        .unwrap();
    let contract = client.info().contract.clone();
    let model = ServedModel::new(model_address, Duration::from_secs(10));
    let routes = rlmesh::binding::model::ModelHandler::route_setup(&model).unwrap();
    routes
        .resolve_adapter(
            "catalog",
            &contract,
            ResolveOptions {
                execution_horizon: 1,
                delivers_history: true,
                workflow_edition: None,
            },
        )
        .await
        .unwrap();
    let spec = RuntimeSessionSpec {
        session_id: "managed".into(),
        env_id: "catalog".into(),
        env_component_id: "catalog-env".into(),
        model_component_id: "served-model".into(),
        workflow_edition: rlmesh_proto::parse_retained_edition("2026.06").unwrap(),
        env_contract: rlmesh_grpc::wire::env_contract_to_proto(&contract),
        num_envs: contract.num_envs as usize,
        base_seed: Some(5),
        episode_seeds: Vec::new(),
        max_episodes: Some(episodes),
        trial_index_base: None,
        max_episode_steps: None,
        max_episode_seconds: None,
        close_env_on_end: false,
        limits: Default::default(),
        subset_step: false,
    };
    RuntimeDriver::new(
        spec,
        LockstepEnv::new(client),
        model,
        contract,
        Arc::new(NoopRuntimeHooks),
    )
    .with_history(true)
    .run()
    .await
    .unwrap()
}

fn run_managed() -> Traces {
    let rt = runtime();
    let env = CatalogEnv::new(4, AutoresetMode::NextStep, false);
    let trace = options(&env);
    let env_address = serve(&rt, env);
    let log = Arc::new(Mutex::new(Vec::new()));
    let model = Model::new(
        CatalogPolicy { log: log.clone() },
        ModelOptions {
            spec: Some(ModelSpec::from_json(CATALOG_MODEL_SPEC).unwrap().into()),
            ..Default::default()
        },
    )
    .unwrap();
    let cancel = tokio_util::sync::CancellationToken::new();
    let report = rt.block_on(async {
        let binding = ModelBinding::bind(":0/model").await.unwrap();
        let model_address = binding.address();
        let service = Arc::new(ModelService::new(
            rlmesh::binding::model::model_handler(&model),
            String::new(),
        ));
        let served = tokio::spawn(binding.serve(service, Duration::from_secs(10), cancel.clone()));
        let report = drive_managed(&env_address, &model_address, 9).await;
        cancel.cancel();
        served.await.unwrap().unwrap();
        report
    });
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
fn a_model_served_on_a_session_drives_a_lockstep_env_unchanged() {
    let traces = run_managed();
    assert!(!traces.model.is_empty() && traces.episodes.len() >= 9);
    golden("served_model", &traces);
}
