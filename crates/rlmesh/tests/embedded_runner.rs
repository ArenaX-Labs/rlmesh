//! A runner of its own, embedding the driver as a program outside this
//! repository does: public paths only (`rlmesh::binding`, `rlmesh::lockstep`,
//! `rlmesh_spaces`), a custom env and a custom model handler, then a served
//! env and a served model. A change that breaks that path fails here first.

#![allow(clippy::unwrap_used)]

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use rlmesh::binding::data::{env_contract_from_proto, env_contract_to_proto, meta_map_to_proto};
use rlmesh::binding::model::{
    HeldState, LockstepEnv, ModelHandler, ModelObservation, ModelRouteSetup, PredictFrames,
    ResolveOptions, RouteNeeds,
};
use rlmesh::binding::runtime::telemetry::Snapshot;
use rlmesh::binding::runtime::{
    EnvResetRequest, EnvStepRequest, EpisodeCompletedEvent, HookError, RuntimeDriver, RuntimeEnv,
    RuntimeEnvReset, RuntimeEnvStep, RuntimeError, RuntimeHooks, RuntimeLimits, RuntimeSessionSpec,
    StepCompletedEvent, TelemetrySnapshotEvent,
};
use rlmesh::lockstep::{ConnectOptions, ServedModel, is_v2_address};
use rlmesh_spaces::leaves::decode_leaf_slab;
use rlmesh_spaces::{
    AutoresetMode, DType, EnvContract, MetaMap, MetaValue, SpaceSpec, SpaceValue, Tensor,
};

/// A `[n]` box of `dtype` in `[-1, 1]`.
fn boxed(dtype: DType, n: i64) -> SpaceSpec {
    rlmesh_spaces::spaces::BoxSpaceBuilder::scalar(-1.0, 1.0, vec![n])
        .dtype(dtype)
        .build()
        .unwrap()
}

/// A counter env: its observation is the step it is at, and it ends after
/// three steps.
#[derive(Clone, Default)]
struct Counter {
    t: Arc<AtomicUsize>,
}

impl Counter {
    fn contract() -> EnvContract {
        EnvContract {
            id: "Counter-v0".into(),
            observation_space: Some(boxed(DType::Uint8, 2)),
            action_space: Some(boxed(DType::Float32, 1)),
            metadata: None,
            render_mode: String::new(),
            num_envs: 1,
            autoreset_mode: AutoresetMode::Disabled,
        }
    }

    fn observation(&self) -> Vec<Bytes> {
        vec![Bytes::from(vec![self.t.load(Ordering::SeqCst) as u8, 7])]
    }
}

#[async_trait]
impl RuntimeEnv for Counter {
    async fn reset(&mut self, _request: EnvResetRequest) -> Result<RuntimeEnvReset, RuntimeError> {
        self.t.store(0, Ordering::SeqCst);
        let info: MetaMap = [("phase".to_string(), MetaValue::String("reset".into()))].into();
        Ok(RuntimeEnvReset {
            observation: self.observation(),
            infos: Some(meta_map_to_proto(&info)),
            endpoint_total_ns: None,
            phases: Default::default(),
        })
    }

    async fn step(&mut self, request: EnvStepRequest) -> Result<RuntimeEnvStep, RuntimeError> {
        assert_eq!(request.action.len(), 1, "one action leaf");
        let t = self.t.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(RuntimeEnvStep {
            observation: self.observation(),
            rewards: vec![1.0],
            terminated: vec![t >= 3],
            truncated: vec![false],
            infos: None,
            endpoint_total_ns: None,
            phases: Default::default(),
        })
    }
}

/// A handler that reads every observation with the space its route resolved
/// and answers zeros of the action space.
#[derive(Clone, Default)]
struct Reader {
    spaces: Arc<Mutex<Option<(SpaceSpec, SpaceSpec)>>>,
    seen: Arc<Mutex<Vec<SpaceValue>>>,
    closed: Arc<AtomicUsize>,
}

#[async_trait]
impl ModelRouteSetup for Reader {
    async fn resolve_adapter(
        &self,
        _env_id: &str,
        contract: &EnvContract,
        options: ResolveOptions,
    ) -> rlmesh::Result<RouteNeeds> {
        assert_eq!(options.execution_horizon, 1);
        *self.spaces.lock().unwrap() = Some((
            contract.observation_space.clone().unwrap(),
            contract.action_space.clone().unwrap(),
        ));
        Ok(RouteNeeds::default())
    }
}

#[async_trait]
impl ModelHandler for Reader {
    async fn predict(&mut self, observation: ModelObservation) -> rlmesh::Result<Vec<SpaceValue>> {
        let (space, action) = self.spaces.lock().unwrap().clone().expect("resolved");
        let leaves = observation.observation.as_deref().unwrap_or_default();
        let lanes = decode_leaf_slab(leaves, &space, observation.num_envs)
            .map_err(|err| rlmesh::Error::model(err.to_string()))?;
        self.seen.lock().unwrap().extend(lanes);
        let zeros = Tensor::zeros(&action.shape, action.dtype)
            .map_err(|err| rlmesh::Error::model(err.to_string()))?;
        Ok(vec![SpaceValue::Box(zeros); observation.num_envs])
    }

    async fn predict_chunked(
        &mut self,
        observation: ModelObservation,
    ) -> rlmesh::Result<PredictFrames> {
        Ok(PredictFrames {
            actions: self.predict(observation).await?,
            replay: Vec::new(),
        })
    }

    fn held_state(&self) -> Option<HeldState> {
        Some(HeldState::default())
    }

    fn route_setup(&self) -> Option<Arc<dyn ModelRouteSetup>> {
        Some(Arc::new(self.clone()))
    }

    async fn on_close(&mut self) -> rlmesh::Result<()> {
        self.closed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

/// What the runner watches.
#[derive(Default)]
struct Watch {
    steps: AtomicUsize,
    episodes: Mutex<Vec<(i64, f64)>>,
    telemetry: Mutex<Option<Snapshot>>,
}

#[async_trait]
impl RuntimeHooks for Watch {
    async fn step_completed(&self, _event: StepCompletedEvent) -> Result<(), HookError> {
        self.steps.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn episode_completed(&self, event: EpisodeCompletedEvent) -> Result<(), HookError> {
        self.episodes
            .lock()
            .unwrap()
            .push((event.step_count, event.cumulative_reward));
        Ok(())
    }

    async fn on_telemetry(&self, event: TelemetrySnapshotEvent) -> Result<(), HookError> {
        *self.telemetry.lock().unwrap() = Some(event.snapshot);
        Ok(())
    }
}

/// A session spec for `contract`, as a runner fills it in.
fn session(
    contract: &EnvContract,
    num_envs: usize,
    subset_step: bool,
    workflow_edition: rlmesh_proto::Edition,
) -> RuntimeSessionSpec {
    RuntimeSessionSpec {
        session_id: "runner-1".into(),
        env_id: contract.id.clone(),
        env_component_id: "env".into(),
        model_component_id: "model".into(),
        workflow_edition,
        env_contract: env_contract_to_proto(contract),
        num_envs,
        base_seed: Some(7),
        seed_key: contract.id.clone(),
        episode_seeds: Vec::new(),
        max_episodes: Some(2),
        trial_index_base: None,
        max_episode_steps: Some(50),
        max_episode_seconds: None,
        close_env_on_end: false,
        close_model_on_end: false,
        limits: RuntimeLimits::default(),
        subset_step,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_runner_drives_its_own_env_and_model() {
    let contract = Counter::contract();
    // The contract round-trips through the driver's data form.
    let wire = env_contract_to_proto(&contract);
    assert_eq!(env_contract_from_proto(wire).unwrap().id, contract.id);

    let model = Reader::default();
    let watch = Arc::new(Watch::default());
    let spec = session(&contract, 1, false, rlmesh_proto::Edition::E2026_10);
    model
        .route_setup()
        .unwrap()
        .resolve_adapter(
            &spec.env_id,
            &contract,
            ResolveOptions {
                execution_horizon: 1,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let report = RuntimeDriver::new(
        spec,
        Counter::default(),
        model.clone(),
        contract,
        watch.clone(),
    )
    .run()
    .await
    .unwrap();

    assert_eq!(report.total_episodes, 2);
    assert_eq!(report.total_steps, 6);
    assert_eq!(watch.steps.load(Ordering::SeqCst), 6);
    assert_eq!(*watch.episodes.lock().unwrap(), vec![(3, 3.0), (3, 3.0)]);
    assert!(
        report
            .telemetry
            .rows
            .iter()
            .any(|row| row.source.op == "model.predict"),
        "the session telemetry records the predicts"
    );
    // Every observation decoded against the env's space: step 0..2, twice.
    {
        let seen = model.seen.lock().unwrap();
        assert_eq!(seen.len(), 6);
        assert!(matches!(&seen[0], SpaceValue::Box(t) if *t.to_contiguous_bytes() == [0, 7]));
    }
    // A model the run does not own is left open for the next run...
    assert_eq!(model.closed.load(Ordering::SeqCst), 0);

    // ...and one it owns is closed once, when the run ends.
    let contract = Counter::contract();
    let mut spec = session(&contract, 1, false, rlmesh_proto::Edition::E2026_10);
    spec.close_model_on_end = true;
    RuntimeDriver::new(
        spec,
        Counter::default(),
        model.clone(),
        contract,
        Arc::new(Watch::default()),
    )
    .run()
    .await
    .unwrap();
    assert_eq!(model.closed.load(Ordering::SeqCst), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_runner_drives_a_served_env_with_a_served_model() {
    use common::CatalogEnv;

    let env = rlmesh::lockstep::bind(":0/catalog").await.unwrap();
    let env_address = env.address();
    assert!(is_v2_address(&env_address));
    let cancel = tokio_util::sync::CancellationToken::new();
    let served_env = tokio::spawn(env.serve(
        Arc::new(CatalogEnv::new(1, AutoresetMode::Disabled, false)),
        rlmesh::lockstep::ServeOptions::default(),
        cancel.clone(),
    ));

    // The runner's model, served in a session of its own.
    let model = Reader::default();
    let binding = rlmesh::lockstep::ModelBinding::bind(":0/model")
        .await
        .unwrap();
    let model_address = binding.address();
    let service = Arc::new(rlmesh::lockstep::ModelService::new(
        model.clone(),
        String::new(),
    ));
    let served_model =
        tokio::spawn(binding.serve(service, Duration::from_secs(10), cancel.clone()));

    // Connected as Model.run connects: the edition and contract checked.
    let connected = LockstepEnv::connect(&env_address, ConnectOptions::default(), Some("2026.10"))
        .await
        .unwrap();
    let mut handler = ServedModel::new(model_address, Duration::from_secs(10));
    if let Some(session) = connected.session.as_deref() {
        handler.join(session).await.unwrap();
    }
    let edition = connected
        .workflow_edition(&mut handler, Some("2026.10"))
        .await
        .unwrap();
    let spec = session(
        &connected.contract,
        connected.num_envs,
        connected.subset_step,
        edition,
    );
    let needs = handler
        .route_setup()
        .unwrap()
        .resolve_adapter(
            &spec.env_id,
            &connected.contract,
            ResolveOptions {
                execution_horizon: 1,
                delivers_history: true,
                workflow_edition: Some(edition),
            },
        )
        .await
        .unwrap();
    let watch = Arc::new(Watch::default());
    let report = RuntimeDriver::new(
        spec,
        connected.env,
        handler,
        connected.contract,
        watch.clone(),
    )
    .with_history(needs.history.is_some())
    .run()
    .await
    .unwrap();

    assert_eq!(report.total_episodes, 2);
    assert_eq!(watch.episodes.lock().unwrap().len(), 2);
    assert!(report.total_steps > 0);
    // The served model read every observation the env sent, decoded.
    assert_eq!(model.seen.lock().unwrap().len() as i64, report.total_steps);
    // A bare connect drives the same env: the runner's own choice to skip the
    // checks above.
    let bare = rlmesh::lockstep::connect(&env_address, ConnectOptions::default())
        .await
        .unwrap();
    let _env = LockstepEnv::new(bare);

    cancel.cancel();
    served_env.await.unwrap().unwrap();
    served_model.await.unwrap().unwrap();
}
