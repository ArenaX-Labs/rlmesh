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

pub(super) async fn run_local_to_async_with_max_episodes<H>(
    handler: &mut H,
    env_address: ConnectAddress,
    max_episodes: Option<u64>,
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
        base_seed: None,
        max_episodes,
        close_env_on_end: false,
        limits: Default::default(),
    };
    let env = LocalRuntimeEnv { inner: env };
    let active_episodes = Arc::new(Mutex::new(HashMap::new()));
    let model = LocalRuntimeModel {
        handler,
        env_contract,
        num_envs,
        active_episodes: Arc::clone(&active_episodes),
    };

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

struct LocalRuntimeEnv {
    inner: rlmesh_grpc::EnvClient,
}

#[async_trait]
impl RuntimeEnv for LocalRuntimeEnv {
    async fn reset(
        &mut self,
        request: rlmesh_proto::env::v1::ResetRequest,
    ) -> std::result::Result<RuntimeEnvReset, rlmesh_runtime::RuntimeError> {
        let response = self.inner.reset(request).await.map_err(|err| {
            rlmesh_runtime::RuntimeError::EnvRpc {
                operation: "env.reset",
                step: 0,
                message: err.to_string(),
            }
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
        let response =
            self.inner
                .step(request)
                .await
                .map_err(|err| rlmesh_runtime::RuntimeError::EnvRpc {
                    operation: "env.step",
                    step: 0,
                    message: err.to_string(),
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

struct LocalRuntimeModel<'a, H> {
    handler: &'a mut H,
    env_contract: spaces::EnvContract,
    num_envs: usize,
    active_episodes: Arc<Mutex<HashMap<(String, i32), String>>>,
}

#[async_trait]
impl<H> RuntimeModel for LocalRuntimeModel<'_, H>
where
    H: ModelHandler + 'static,
{
    async fn predict(
        &mut self,
        request: PredictRequest,
    ) -> std::result::Result<RuntimeModelPrediction, rlmesh_runtime::RuntimeError> {
        let mut observation = model_observation_from_endpoint_request(request).map_err(|err| {
            rlmesh_runtime::RuntimeError::ModelRpc {
                component_id: "local-model".to_string(),
                message: err.to_string(),
            }
        })?;
        let route = observation.route.clone();
        observation.env_contract = Some(self.env_contract.clone());
        observation.num_envs = self.num_envs;
        let mut active_episodes = self.active_episodes.lock().await;
        update_lifecycle(self.handler, &mut active_episodes, &observation)
            .await
            .map_err(|err| rlmesh_runtime::RuntimeError::ModelRpc {
                component_id: "local-model".to_string(),
                message: err.to_string(),
            })?;
        let action = self.handler.predict(observation).await.map_err(|err| {
            rlmesh_runtime::RuntimeError::ModelRpc {
                component_id: "local-model".to_string(),
                message: err.to_string(),
            }
        })?;
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
