use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use rlmesh_proto::common::v1::MessageBytes;
use rlmesh_proto::env::v1::{
    EnvContract, EpisodeMetadata, ResetRequest, ResetResponse, StepRequest, StepResponse,
};
use rlmesh_proto::model::v1::{CloseRouteRequest, PredictRequest, PredictResponse};
use rlmesh_proto::spaces::v1::{SpaceSpec, SpaceValue};
use rlmesh_runtime::{
    ActionReceivedEvent, HookError, RuntimeDriver, RuntimeEnv, RuntimeEnvReset, RuntimeEnvStep,
    RuntimeError, RuntimeHooks, RuntimeModel, RuntimeModelPrediction, RuntimeSessionSpec,
};

#[tokio::test]
async fn driver_runs_one_episode_and_closes_terminal_route() {
    let env = TestEnv::default();
    let model = TestModel::default();
    let hooks = Arc::new(RecordingHooks::default());

    let report = RuntimeDriver::new(
        one_episode_spec(),
        env.clone(),
        model.clone(),
        hooks.clone(),
    )
    .run()
    .await
    .unwrap();

    assert_eq!(report.total_steps, 1);
    assert_eq!(report.total_episodes, 1);
    assert!(env.closed.load(Ordering::SeqCst));
    assert!(model.closed.load(Ordering::SeqCst));
    assert_eq!(hooks.actions.load(Ordering::SeqCst), 1);
    assert_eq!(hooks.ended.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn model_predict_timeout_fails_session() {
    let env = TestEnv::default();
    let model = TestModel {
        predict_delay: Some(Duration::from_millis(50)),
        ..Default::default()
    };
    let hooks = Arc::new(RecordingHooks::default());
    let mut spec = one_episode_spec();
    spec.limits.model_predict_timeout = Duration::from_millis(5);

    let error = RuntimeDriver::new(spec, env, model, hooks.clone())
        .run()
        .await
        .unwrap_err();

    assert!(matches!(
        error,
        RuntimeError::OperationTimeout {
            operation: "model.predict",
            ..
        }
    ));
    assert_eq!(hooks.failed.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn driver_continues_after_non_terminal_step() {
    let env = TestEnv {
        terminal_after: 2,
        ..Default::default()
    };
    let model = TestModel::default();

    let report = RuntimeDriver::new(
        one_episode_spec(),
        env,
        model.clone(),
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .unwrap();

    assert_eq!(report.total_steps, 2);
    assert_eq!(report.total_episodes, 1);
    assert_eq!(model.predicts.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn fatal_transform_hook_failure_closes_route() {
    let env = TestEnv::default();
    let model = TestModel::default();
    let hooks = Arc::new(RecordingHooks {
        fail_action_transform: true,
        ..Default::default()
    });

    let error = RuntimeDriver::new(one_episode_spec(), env.clone(), model.clone(), hooks)
        .run()
        .await
        .unwrap_err();

    assert!(matches!(error, RuntimeError::Hook(_)));
    assert!(env.closed.load(Ordering::SeqCst));
    assert!(model.closed.load(Ordering::SeqCst));
}

#[tokio::test]
async fn driver_threads_deterministic_reset_seeds() {
    let mut spec = one_episode_spec();
    spec.base_seed = Some(1234);
    spec.max_episodes = Some(2);

    let first_env = TestEnv::default();
    let first_model = TestModel::default();
    RuntimeDriver::new(
        spec.clone(),
        first_env.clone(),
        first_model,
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .unwrap();

    let second_env = TestEnv::default();
    let second_model = TestModel::default();
    RuntimeDriver::new(
        spec,
        second_env.clone(),
        second_model,
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .unwrap();

    let first_seeds = first_env
        .reset_seeds
        .lock()
        .expect("reset seed recorder lock poisoned")
        .clone();
    let second_seeds = second_env
        .reset_seeds
        .lock()
        .expect("reset seed recorder lock poisoned")
        .clone();

    assert_eq!(first_seeds, second_seeds);
    assert_eq!(first_seeds.len(), 2);
    assert_eq!(first_seeds[0].len(), 1);
    assert_eq!(first_seeds[1].len(), 1);
    assert_ne!(first_seeds[0], first_seeds[1]);
}

#[derive(Clone)]
struct TestEnv {
    closed: Arc<AtomicBool>,
    step_count: Arc<AtomicUsize>,
    reset_seeds: Arc<Mutex<Vec<Vec<i64>>>>,
    terminal_after: usize,
}

impl Default for TestEnv {
    fn default() -> Self {
        Self {
            closed: Arc::new(AtomicBool::new(false)),
            step_count: Arc::new(AtomicUsize::new(0)),
            reset_seeds: Arc::new(Mutex::new(Vec::new())),
            terminal_after: 1,
        }
    }
}

#[async_trait]
impl RuntimeEnv for TestEnv {
    async fn reset(&mut self, request: ResetRequest) -> Result<RuntimeEnvReset, RuntimeError> {
        self.reset_seeds
            .lock()
            .expect("reset seed recorder lock poisoned")
            .push(request.seeds);
        self.step_count.store(0, Ordering::SeqCst);
        Ok(RuntimeEnvReset {
            response: ResetResponse {
                observation: Some(bytes_value(payload([1]))),
                infos: None,
                episode_ids: vec!["episode-1".to_string()],
            },
            telemetry: None,
        })
    }

    async fn step(&mut self, _request: StepRequest) -> Result<RuntimeEnvStep, RuntimeError> {
        let step = self.step_count.fetch_add(1, Ordering::SeqCst) + 1;
        let terminal = step >= self.terminal_after;
        Ok(RuntimeEnvStep {
            response: StepResponse {
                observation: Some(bytes_value(payload([step as u8]))),
                rewards: vec![1.0],
                terminated_mask: vec![u8::from(terminal)],
                truncated_mask: vec![0],
                infos: None,
                completed_episodes: terminal
                    .then(|| EpisodeMetadata {
                        episode_id: "episode-1".to_string(),
                        step_count: step as i64,
                        cumulative_reward: step as f64,
                        terminated: true,
                        ..Default::default()
                    })
                    .into_iter()
                    .collect(),
                episode_ids: vec![],
            },
            telemetry: None,
        })
    }

    async fn close(&mut self, _timeout: Duration) -> Result<(), String> {
        self.closed.store(true, Ordering::SeqCst);
        Ok(())
    }
}

#[derive(Clone, Default)]
struct TestModel {
    closed: Arc<AtomicBool>,
    predicts: Arc<AtomicUsize>,
    predict_delay: Option<Duration>,
}

#[async_trait]
impl RuntimeModel for TestModel {
    async fn predict(
        &mut self,
        request: PredictRequest,
    ) -> Result<RuntimeModelPrediction, RuntimeError> {
        if let Some(delay) = self.predict_delay {
            tokio::time::sleep(delay).await;
        }
        self.predicts.fetch_add(1, Ordering::SeqCst);
        Ok(RuntimeModelPrediction {
            response: PredictResponse {
                context: request.context,
                action: Some(bytes_value(payload([0]))),
            },
            telemetry: None,
        })
    }

    async fn close_route(
        &mut self,
        _request: CloseRouteRequest,
        _timeout: Duration,
    ) -> Result<(), String> {
        self.closed.store(true, Ordering::SeqCst);
        Ok(())
    }
}

#[derive(Default)]
struct RecordingHooks {
    actions: AtomicUsize,
    ended: AtomicUsize,
    failed: AtomicUsize,
    fail_action_transform: bool,
}

#[async_trait]
impl RuntimeHooks for RecordingHooks {
    async fn action_received(&self, _event: ActionReceivedEvent) -> Result<(), HookError> {
        self.actions.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn transform_action(
        &self,
        event: ActionReceivedEvent,
    ) -> Result<Option<MessageBytes>, HookError> {
        if self.fail_action_transform {
            return Err(HookError::Message("transform failed".to_string()));
        }
        Ok(event.action)
    }

    async fn session_ended(
        &self,
        _event: rlmesh_runtime::SessionEndedEvent,
    ) -> Result<(), HookError> {
        self.ended.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn session_failed(
        &self,
        _event: rlmesh_runtime::SessionFailedEvent,
    ) -> Result<(), HookError> {
        self.failed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

fn one_episode_spec() -> RuntimeSessionSpec {
    RuntimeSessionSpec {
        session_id: "session-1".to_string(),
        route_id: "route-1".to_string(),
        env_component_id: "env-1".to_string(),
        model_component_id: "model-1".to_string(),
        env_id: "TestEnv-v0".to_string(),
        env_contract: EnvContract {
            observation_space: Some(SpaceSpec::default()),
            action_space: Some(SpaceSpec::default()),
            num_envs: 1,
            ..Default::default()
        },
        num_envs: 1,
        base_seed: None,
        max_episodes: Some(1),
        close_env_on_end: true,
        limits: Default::default(),
    }
}

fn payload<const N: usize>(data: [u8; N]) -> MessageBytes {
    MessageBytes {
        data: data.to_vec(),
    }
}

fn bytes_value(value: MessageBytes) -> SpaceValue {
    SpaceValue { bytes: Some(value) }
}
