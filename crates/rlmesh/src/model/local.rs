use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use rlmesh_grpc::wire::{
    encode_batched_partial_values, env_contract_from_proto, env_contract_to_proto,
};
use rlmesh_proto::model::v1::{PredictRequest, ResetAdapterRequest};
use rlmesh_runtime::{
    NoopRuntimeHooks, RuntimeDriver, RuntimeEnv, RuntimeEnvReset, RuntimeEnvStep, RuntimeError,
    RuntimeModel, RuntimeModelPrediction, RuntimeReport, RuntimeSessionSpec,
};

use super::handler::{ModelHandler, PredictFrames};
use super::wire::{
    ModelAction, check_actions_conform, encode_replay_frames, model_action_to_endpoint_response,
    model_observation_from_endpoint_request,
};
use crate::{ConnectAddress, Error, Result, spaces};

pub(super) async fn run_local<H>(
    handler: &mut H,
    env_address: ConnectAddress,
    max_episodes: Option<u64>,
    base_seed: Option<i64>,
) -> Result<RuntimeReport>
where
    H: ModelHandler + 'static,
{
    let mut env = rlmesh_grpc::EnvClient::connect(&env_address.to_string())
        .await
        .map_err(Error::from)?;
    let handshake = env.handshake().await.map_err(Error::from)?;
    let env_contract = env_contract_from_proto(handshake.env_contract)
        .map_err(|err| Error::Internal(format!("invalid spaces spec from env: {err}")))?;
    // The handler returns typed actions the runtime encodes against this space;
    // a missing one would fail every predict, so reject at connect, not mid-run.
    if env_contract.action_space.is_none() {
        return Err(Error::Internal(
            "env contract has no action_space; a model cannot encode actions without it"
                .to_string(),
        ));
    }
    // The runtime is the env-id authority (R1); for the in-process path mint a
    // UUIDv7 container id. The human env name lives on the SDK's own contract.
    let env_id = crate::mint_id();
    let num_envs = handshake.num_envs;
    let session_id = format!("local-{}", std::process::id());

    let spec = RuntimeSessionSpec {
        session_id,
        env_id,
        env_component_id: "local-env".to_string(),
        model_component_id: "local-model".to_string(),
        workflow_edition: handshake.workflow_edition,
        env_contract: env_contract_to_proto(&env_contract),
        num_envs,
        base_seed,
        max_episodes,
        close_env_on_end: false,
        limits: Default::default(),
    };
    let env = EnvClientRuntimeEnv::new(env);
    let model = ModelHandlerRuntimeModel::new(handler, env_contract, num_envs);

    RuntimeDriver::new(spec, env, model, Arc::new(NoopRuntimeHooks))
        .run()
        .await
        .map_err(|err| Error::Internal(err.to_string()))
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
            endpoint_total_ns: self.inner.take_last_endpoint_total_ns(),
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
            endpoint_total_ns: self.inner.take_last_endpoint_total_ns(),
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
/// [`ModelObservation`](crate::ModelObservation), runs `predict`, and re-encodes
/// the action, matching the choreography the in-process `run_local` path
/// performs. Per-episode lifecycle is explicit (see below) — there is no
/// episode-begin hook; the model's state is lazy-seeded on first predict.
///
/// It borrows the handler mutably so the caller retains ownership (e.g. to run
/// the close hook afterward). Per-episode lifecycle is explicit (R2): the runtime
/// driver emits `ResetAdapter` on episode end, routed here to the handler's
/// `reset_adapter`; there is no position-diff / active-episodes state.
pub struct ModelHandlerRuntimeModel<'a, H> {
    handler: &'a mut H,
    env_contract: Arc<spaces::EnvContract>,
    num_envs: usize,
}

impl<'a, H> ModelHandlerRuntimeModel<'a, H> {
    /// Build an adapter for `handler` against the given env contract.
    pub fn new(handler: &'a mut H, env_contract: spaces::EnvContract, num_envs: usize) -> Self {
        Self {
            handler,
            env_contract: Arc::new(env_contract),
            num_envs,
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
        observation.env_contract = Some(Arc::clone(&self.env_contract));
        observation.num_envs = self.num_envs;
        let num_envs = self.num_envs;
        let action_space = self.env_contract.action_space.clone().ok_or_else(|| {
            rlmesh_runtime::RuntimeError::model_rpc(
                "local-model",
                Error::model("model route contract missing action space"),
            )
        })?;
        let PredictFrames { actions, replay } = self
            .handler
            .predict_chunked(observation)
            .await
            .map_err(|err| rlmesh_runtime::RuntimeError::model_rpc("local-model", err))?;
        if actions.len() != num_envs {
            return Err(rlmesh_runtime::RuntimeError::model_rpc(
                "local-model",
                Error::model(format!(
                    "predict returned {} actions for {num_envs} lanes",
                    actions.len()
                )),
            ));
        }
        check_actions_conform(&action_space, &actions)
            .map_err(|err| rlmesh_runtime::RuntimeError::model_rpc("local-model", err))?;
        let frame0 = encode_batched_partial_values(&actions, &action_space).map_err(|err| {
            rlmesh_runtime::RuntimeError::model_rpc("local-model", Error::model(err.to_string()))
        })?;
        let replay_frames = encode_replay_frames(&replay, num_envs, &action_space)
            .map_err(|err| rlmesh_runtime::RuntimeError::model_rpc("local-model", err))?;
        let mut wire_actions = Vec::with_capacity(1 + replay_frames.len());
        wire_actions.push(frame0);
        wire_actions.extend(replay_frames);
        Ok(RuntimeModelPrediction {
            response: model_action_to_endpoint_response(ModelAction {
                actions: wire_actions,
                route,
            }),
            endpoint_total_ns: None,
        })
    }

    async fn reset_adapter(
        &mut self,
        request: ResetAdapterRequest,
    ) -> std::result::Result<(), RuntimeError> {
        // Route the driver's explicit episode-end GC to the handler's evict hook.
        let env_id = request
            .context
            .map(|context| context.env_id)
            .unwrap_or_default();
        self.handler
            .reset_adapter(&env_id, request.episode_ids)
            .await
            .map_err(|err| RuntimeError::model_rpc("local-model", err))
    }
}
