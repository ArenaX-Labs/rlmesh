use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use rlmesh_grpc::wire::{env_contract_from_proto, env_contract_to_proto};
use rlmesh_proto::model::v1::PredictRequest;
use rlmesh_runtime::{
    NoopRuntimeHooks, RuntimeDriver, RuntimeEnv, RuntimeEnvReset, RuntimeEnvStep, RuntimeModel,
    RuntimeModelPrediction, RuntimeSessionSpec,
};
use tokio::sync::Mutex;

use super::handler::ModelHandler;
use super::lifecycle::{finish_lifecycle, update_lifecycle};
use super::wire::{
    ModelAction, model_action_to_endpoint_response, model_observation_from_endpoint_request,
};
use crate::{ConnectAddress, Error, Result, spaces};

pub(super) async fn run_local<H>(
    handler: &mut H,
    env_address: ConnectAddress,
    max_episodes: Option<u64>,
    base_seed: Option<i64>,
) -> Result<()>
where
    H: ModelHandler + 'static,
{
    let mut env = rlmesh_grpc::EnvClient::connect(&env_address.to_string())
        .await
        .map_err(Error::from)?;
    let handshake = env.handshake().await.map_err(Error::from)?;
    let env_contract = env_contract_from_proto(handshake.env_contract)
        .map_err(|err| Error::Internal(format!("invalid spaces spec from env: {err}")))?;
    let env_id = if env_contract.id.is_empty() {
        "local-env".to_string()
    } else {
        env_contract.id.clone()
    };
    let num_envs = handshake.num_envs;
    let session_id = format!("local-{}", std::process::id());

    let spec = RuntimeSessionSpec {
        session_id,
        route_id: "local".to_string(),
        env_component_id: "local-env".to_string(),
        model_component_id: "local-model".to_string(),
        env_id,
        env_contract: env_contract_to_proto(&env_contract),
        num_envs,
        base_seed,
        max_episodes,
        close_env_on_end: false,
        limits: Default::default(),
    };
    let env = EnvClientRuntimeEnv::new(env);
    let active_episodes = Arc::new(Mutex::new(HashMap::new()));
    let model = ModelHandlerRuntimeModel::new(
        handler,
        env_contract,
        num_envs,
        Arc::clone(&active_episodes),
    );

    let result = RuntimeDriver::new(spec, env, model, Arc::new(NoopRuntimeHooks))
        .run()
        .await
        .map(|_| ())
        .map_err(|err| Error::Internal(err.to_string()));

    if result.is_ok() {
        let mut active_episodes = active_episodes.lock().await;
        finish_lifecycle(handler, &mut active_episodes).await?;
    }

    result
}

/// Adapts a connected [`rlmesh_grpc::EnvClient`] to the [`RuntimeEnv`] trait
/// expected by [`rlmesh_runtime::RuntimeDriver`].
///
/// Use this to drive a remote environment from your own `RuntimeDriver`
/// embedding without re-implementing the per-call telemetry choreography: the
/// adapter takes the client's last-operation telemetry after each `reset`/`step`
/// and attaches it to the runtime result for you, and it maps transport errors
/// onto recoverable/non-recoverable [`rlmesh_runtime::RuntimeError`]s.
pub struct EnvClientRuntimeEnv {
    inner: rlmesh_grpc::EnvClient,
}

impl EnvClientRuntimeEnv {
    /// Wrap a connected (and handshaked) env client.
    pub fn new(client: rlmesh_grpc::EnvClient) -> Self {
        Self { inner: client }
    }

    /// Borrow the underlying client.
    pub fn client(&self) -> &rlmesh_grpc::EnvClient {
        &self.inner
    }

    /// Consume the adapter and return the underlying client.
    pub fn into_inner(self) -> rlmesh_grpc::EnvClient {
        self.inner
    }
}

#[async_trait]
impl RuntimeEnv for EnvClientRuntimeEnv {
    async fn reset(
        &mut self,
        request: rlmesh_proto::env::v1::ResetRequest,
    ) -> std::result::Result<RuntimeEnvReset, rlmesh_runtime::RuntimeError> {
        let response = self.inner.reset(request).await.map_err(|err| {
            let recoverable = err.is_recoverable();
            rlmesh_runtime::RuntimeError::env_rpc_with_recoverability(
                "env.reset",
                0,
                recoverable,
                err,
            )
        })?;
        Ok(RuntimeEnvReset {
            response,
            telemetry: self.inner.take_last_telemetry(),
        })
    }

    async fn step(
        &mut self,
        request: rlmesh_proto::env::v1::StepRequest,
    ) -> std::result::Result<RuntimeEnvStep, rlmesh_runtime::RuntimeError> {
        let response = self.inner.step(request).await.map_err(|err| {
            let recoverable = err.is_recoverable();
            rlmesh_runtime::RuntimeError::env_rpc_with_recoverability(
                "env.step",
                0,
                recoverable,
                err,
            )
        })?;
        Ok(RuntimeEnvStep {
            response,
            telemetry: self.inner.take_last_telemetry(),
        })
    }

    async fn close(&mut self, timeout: Duration) -> std::result::Result<(), String> {
        let close = self.inner.close();
        tokio::time::timeout(timeout, close)
            .await
            .map_err(|err| err.to_string())?
            .map(|_| ())
            .map_err(|err| err.to_string())
    }
}

/// Adapts a [`ModelHandler`] to the [`RuntimeModel`] trait expected by
/// [`rlmesh_runtime::RuntimeDriver`].
///
/// Use this to drive your handler from your own `RuntimeDriver` embedding. The
/// adapter decodes the runtime's predict request into a
/// [`ModelObservation`](crate::ModelObservation),
/// runs the handler's episode lifecycle (`on_reset`/`on_episode_end`) before
/// each `predict`, and re-encodes the action — the same choreography the
/// in-process `run_local` path performs.
///
/// It borrows the handler mutably so the caller retains ownership (e.g. to run
/// the close hook afterward). `active_episodes` is shared lifecycle state; pass
/// a fresh `Arc<Mutex<_>>` and drain it with the model lifecycle helpers when
/// the session ends.
pub struct ModelHandlerRuntimeModel<'a, H> {
    handler: &'a mut H,
    env_contract: spaces::EnvContract,
    num_envs: usize,
    active_episodes: Arc<Mutex<HashMap<(String, i32), String>>>,
}

impl<'a, H> ModelHandlerRuntimeModel<'a, H> {
    /// Build an adapter for `handler` against the given env contract.
    pub fn new(
        handler: &'a mut H,
        env_contract: spaces::EnvContract,
        num_envs: usize,
        active_episodes: Arc<Mutex<HashMap<(String, i32), String>>>,
    ) -> Self {
        Self {
            handler,
            env_contract,
            num_envs,
            active_episodes,
        }
    }
}

#[async_trait]
impl<H> RuntimeModel for ModelHandlerRuntimeModel<'_, H>
where
    H: ModelHandler + 'static,
{
    async fn predict(
        &mut self,
        request: PredictRequest,
    ) -> std::result::Result<RuntimeModelPrediction, rlmesh_runtime::RuntimeError> {
        let mut observation = model_observation_from_endpoint_request(request)
            .map_err(|err| rlmesh_runtime::RuntimeError::model_rpc("local-model", err))?;
        let route = observation.route.clone();
        observation.env_contract = Some(self.env_contract.clone());
        observation.num_envs = self.num_envs;
        let mut active_episodes = self.active_episodes.lock().await;
        update_lifecycle(self.handler, &mut active_episodes, &observation)
            .await
            .map_err(|err| rlmesh_runtime::RuntimeError::model_rpc("local-model", err))?;
        let action = self
            .handler
            .predict(observation)
            .await
            .map_err(|err| rlmesh_runtime::RuntimeError::model_rpc("local-model", err))?;
        Ok(RuntimeModelPrediction {
            response: model_action_to_endpoint_response(ModelAction {
                action: Some(action),
                route,
                telemetry: None,
            }),
            telemetry: None,
        })
    }
}
