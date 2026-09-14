use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use rlmesh_grpc::wire::{
    encode_batched_partial_values, env_contract_from_proto, env_contract_to_proto,
};
use rlmesh_proto::model::v1::{PredictRequest, ResetAdapterRequest};
use rlmesh_proto::{EndpointPhases, elapsed_ns};
use rlmesh_runtime::{
    NoopRuntimeHooks, RuntimeDriver, RuntimeEnv, RuntimeEnvReset, RuntimeEnvStep, RuntimeError,
    RuntimeModel, RuntimeModelPrediction, RuntimeReport, RuntimeSessionSpec,
};

use super::handler::{ModelHandler, PredictFrames};
use super::wire::{
    ModelAction, check_actions_conform, encode_replay_frames, model_action_to_endpoint_response,
    model_observation_from_endpoint_request,
};
use crate::{Error, Result, spaces};

/// Connect to the env, resolve the handler's route, and drive the runtime
/// loop to completion. The route resolves before driving, exactly as the
/// served path does at `ResolveAdapter`: that arms the spec'd engine path
/// (adapter, per-episode frame buffers, batched/chunked predict corners) and
/// pins the execution horizon — without it every predict would fall into the
/// spec-less branch. The handler is dropped when the run ends, so there is no
/// release step.
pub(super) async fn run_local<H>(
    handler: &mut H,
    options: crate::RunLocalOptions,
    cancellation: tokio_util::sync::CancellationToken,
) -> Result<RuntimeReport>
where
    H: ModelHandler + 'static,
{
    let mut env = rlmesh_grpc::EnvClient::connect(&options.env_address.to_string())
        .await
        .map_err(Error::from)?;
    let handshake = env.handshake().await.map_err(Error::from)?;
    // A lane endpoint steps/resets lanes individually, so a num_envs > 1
    // session can run driver-owned (DISABLED) resets lane by lane.
    let subset_step = rlmesh_proto::has_capability(&handshake.capabilities, "subset_step");
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

    // Action chunking across a lockstep vector env would replay one whole-batch
    // chunk for every lane, and a lane that ends mid-chunk invalidates the buffer
    // for all of them. A lane endpoint replays per lane, so it is fine there.
    if num_envs > 1 && !subset_step && options.execution_horizon > 1 {
        return Err(Error::Internal(format!(
            "execution_horizon={} cannot be combined with a lockstep vector env (num_envs={num_envs}): \
             chunk replay is whole-batch, so one lane's episode end discards every lane's \
             buffered frames. Use num_envs=1, a lane endpoint, or execution_horizon=1.",
            options.execution_horizon,
        )));
    }

    if let Some(route_setup) = handler.route_setup() {
        route_setup
            .resolve_adapter(
                &env_id,
                &env_contract,
                crate::model::ResolveOptions {
                    execution_horizon: options.execution_horizon,
                    // Observation history is not delivered on this path (or any,
                    // yet); a history-needing model keeps its own window.
                    delivers_history: false,
                },
            )
            .await?;
    }

    let spec = RuntimeSessionSpec {
        session_id,
        env_id,
        env_component_id: "local-env".to_string(),
        model_component_id: "local-model".to_string(),
        workflow_edition: handshake.workflow_edition,
        env_contract: env_contract_to_proto(&env_contract),
        num_envs,
        base_seed: options.base_seed,
        episode_seeds: options.episode_seeds,
        max_episodes: options.max_episodes,
        trial_index_base: options.trial_index_base,
        max_episode_steps: options.max_episode_steps,
        max_episode_seconds: options.max_episode_seconds,
        close_env_on_end: options.close_env,
        // A lane endpoint is driven one group per lane; the grouped predicts
        // reach the handler's `predict_grouped` (one fused forward for a model
        // with a batched corner).
        subset_step,
        limits: Default::default(),
    };
    let env = EnvClientRuntimeEnv::new(env);
    let model = ModelHandlerRuntimeModel::new(handler, env_contract);
    RuntimeDriver::new(spec, env, model, Arc::new(NoopRuntimeHooks))
        .run_with_cancellation_reason(cancellation, "interrupted by the host (signal)")
        .await
        .map_err(run_error)
}

/// Wrap a facade [`Error`] as a driver model-RPC failure, keeping its
/// recoverable flag (the default `model_rpc` constructor drops it, so a
/// recoverable handler decline used to reach the caller as permanent).
fn model_rpc(error: Error) -> RuntimeError {
    RuntimeError::model_rpc_with_recoverability("local-model", error.is_recoverable(), error)
}

/// Map the driver's error back onto the facade taxonomy. Flattening every
/// failure to [`Error::Internal`] made a model decline, an env fault, a dropped
/// connection and a timeout indistinguishable to the caller (and always
/// non-recoverable); the structured `#[source]` each RPC variant carries is the
/// original error, so unwrap it when it is one of ours.
fn run_error(error: RuntimeError) -> Error {
    let recoverable = error.is_recoverable();
    let message = error.to_string();
    match error {
        RuntimeError::ModelRpc { source, .. } => source
            .and_then(|source| source.downcast::<Error>().ok())
            .map_or_else(
                || {
                    if recoverable {
                        Error::model_recoverable(message)
                    } else {
                        Error::model(message)
                    }
                },
                |error| *error,
            ),
        // Keep the driver's message (it names the op and the step) and take only
        // the classification from the structured source.
        RuntimeError::EnvRpc { source, .. } => match source
            .and_then(|source| source.downcast::<rlmesh_grpc::error::Error>().ok())
            .map(|error| Error::from(*error))
        {
            Some(Error::Connection(_)) => Error::Connection(message),
            Some(Error::Timeout(timeout)) => Error::Timeout(timeout),
            Some(Error::Environment(env)) => {
                Error::Environment(crate::EnvironmentError { message, ..env })
            }
            _ => Error::Environment(crate::EnvironmentError {
                code: crate::ErrorCode::Internal,
                message,
                is_recoverable: recoverable,
            }),
        },
        RuntimeError::OperationTimeout { timeout, .. } => Error::Timeout(timeout),
        _ => Error::Internal(message),
    }
}

/// Adapts a connected [`rlmesh_grpc::EnvClient`] to the [`RuntimeEnv`] trait
/// expected by [`rlmesh_runtime::RuntimeDriver`].
///
/// Use this to drive a remote environment from your own `RuntimeDriver`
/// embedding without re-implementing the per-call telemetry choreography: the
/// adapter takes the client's last-operation telemetry after each `reset`/`step`
/// and attaches it to the runtime result for you, and it maps transport errors
/// onto recoverable/non-recoverable [`rlmesh_runtime::RuntimeError`]s.
#[derive(Clone)]
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
            phases: self.inner.take_last_phases(),
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
            phases: self.inner.take_last_phases(),
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
    /// The handler, behind a lock: the driver may evict adapter state while a
    /// grouped predict is being prepared, and a served handler is shared the
    /// same way.
    handler: Arc<tokio::sync::Mutex<&'a mut H>>,
    env_contract: Arc<spaces::EnvContract>,
}

impl<'a, H> ModelHandlerRuntimeModel<'a, H> {
    /// Build an adapter for `handler` against the given env contract.
    pub fn new(handler: &'a mut H, env_contract: spaces::EnvContract) -> Self {
        Self {
            handler: Arc::new(tokio::sync::Mutex::new(handler)),
            env_contract: Arc::new(env_contract),
        }
    }
}

/// One group of a grouped predict, prepared for the handler: what encoding
/// its reply needs.
struct PreparedGroup {
    route: crate::model::types::ModelRouteContext,
    num_envs: usize,
}

#[async_trait]
impl<H> RuntimeModel for ModelHandlerRuntimeModel<'_, H>
where
    H: ModelHandler + 'static,
{
    async fn predict(
        &self,
        request: PredictRequest,
    ) -> std::result::Result<RuntimeModelPrediction, rlmesh_runtime::RuntimeError> {
        self.predict_group(vec![request])
            .await
            .pop()
            .unwrap_or_else(|| {
                Err(rlmesh_runtime::RuntimeError::model_rpc(
                    "local-model",
                    Error::model("predict_grouped returned no result"),
                ))
            })
    }

    /// Every request decodes into one `ModelObservation` and the batch goes to
    /// the handler's `predict_grouped` — the same seam the served endpoint
    /// uses, so a model with a batched corner runs one fused forward for all
    /// the lanes waiting on it. Results align 1:1 and in order.
    async fn predict_group(
        &self,
        requests: Vec<PredictRequest>,
    ) -> Vec<std::result::Result<RuntimeModelPrediction, rlmesh_runtime::RuntimeError>> {
        let started = Instant::now();
        let model_err = model_rpc;
        let Some(action_space) = self.env_contract.action_space.clone() else {
            return requests
                .iter()
                .map(|_| {
                    Err(model_err(Error::model(
                        "model route contract missing action space",
                    )))
                })
                .collect();
        };
        // Prepare every group; one that fails to decode reports its own error
        // and is left out of the batch.
        let mut batch = Vec::with_capacity(requests.len());
        let mut prepared: Vec<std::result::Result<PreparedGroup, rlmesh_runtime::RuntimeError>> =
            Vec::with_capacity(requests.len());
        for request in requests {
            match model_observation_from_endpoint_request(request) {
                Ok(mut observation) => {
                    // The request's row count is its own width (a lane group
                    // sends one row), never the route's.
                    let num_envs = observation.route.episodes.len().max(1);
                    observation.env_contract = Some(Arc::clone(&self.env_contract));
                    observation.num_envs = num_envs;
                    prepared.push(Ok(PreparedGroup {
                        route: observation.route.clone(),
                        num_envs,
                    }));
                    batch.push(observation);
                }
                Err(err) => prepared.push(Err(model_err(err))),
            }
        }
        let decode_ns = elapsed_ns(started);

        let mut handler = self.handler.lock().await;
        let call_started = Instant::now();
        let mut frames = handler.predict_grouped(batch).await.into_iter();
        let user_ns = elapsed_ns(call_started);
        // Drain the adapter share even for a failed forward, or its time leaks
        // into the next predict's `adapter_ns`.
        let adapter_ns = handler.take_adapter_ns();
        let held = handler.held_state();
        drop(handler);

        let encode_started = Instant::now();
        prepared
            .into_iter()
            .map(|group| {
                let PreparedGroup { route, num_envs } = group?;
                let PredictFrames { actions, replay } = frames
                    .next()
                    .ok_or_else(|| {
                        model_err(Error::model(
                            "predict_grouped returned fewer results than prepared groups",
                        ))
                    })?
                    .map_err(model_err)?;
                if actions.len() != num_envs {
                    return Err(model_err(Error::model(format!(
                        "predict returned {} actions for {num_envs} lanes",
                        actions.len()
                    ))));
                }
                check_actions_conform(&action_space, &actions).map_err(model_err)?;
                let frame0 = encode_batched_partial_values(&actions, &action_space)
                    .map_err(|err| model_err(Error::model(err.to_string())))?;
                let replay_frames =
                    encode_replay_frames(&replay, num_envs, &action_space).map_err(model_err)?;
                let mut wire_actions = Vec::with_capacity(1 + replay_frames.len());
                wire_actions.push(frame0);
                wire_actions.extend(replay_frames);
                Ok(RuntimeModelPrediction {
                    response: model_action_to_endpoint_response(ModelAction {
                        actions: wire_actions,
                        route,
                    }),
                    endpoint_total_ns: Some(elapsed_ns(started)),
                    phases: EndpointPhases {
                        decode_ns,
                        user_ns,
                        encode_ns: elapsed_ns(encode_started),
                        adapter_ns,
                        held_episodes: held
                            .map(|held| held.episodes.min(u64::from(u32::MAX)) as u32),
                        held_state_bytes: held.map(|held| held.bytes),
                        ..EndpointPhases::default()
                    },
                    group_size: None,
                })
            })
            .collect()
    }

    async fn reset_adapter(
        &self,
        request: ResetAdapterRequest,
    ) -> std::result::Result<(), RuntimeError> {
        // Route the driver's explicit episode-end GC to the handler's evict hook.
        let env_id = request
            .context
            .map(|context| context.env_id)
            .unwrap_or_default();
        self.handler
            .lock()
            .await
            .reset_adapter(&env_id, request.episode_ids)
            .await
            .map_err(model_rpc)
    }
}
