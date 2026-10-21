use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
#[cfg(feature = "lockstep")]
use rlmesh_grpc::wire::{decode_batched_partial_values, meta_map_from_proto, meta_map_to_proto};
use rlmesh_grpc::wire::{
    encode_batched_partial_values, env_contract_from_proto, env_contract_to_proto,
};
use rlmesh_proto::model::v1::{PredictRequest, ResetAdapterRequest};
use rlmesh_proto::{EndpointPhases, elapsed_ns};
use rlmesh_runtime::{
    PeerCeiling, RuntimeDriver, RuntimeEnv, RuntimeEnvReset, RuntimeEnvStep, RuntimeError,
    RuntimeHooks, RuntimeModel, RuntimeModelPrediction, RuntimeReport, RuntimeSessionSpec,
};

use super::handler::{ModelHandler, PredictFrames};
use super::wire::{
    ModelAction, check_actions_conform, encode_replay_frames, model_action_to_endpoint_response,
    model_observation_from_endpoint_request,
};
#[cfg(feature = "lockstep")]
use crate::lockstep::GymError;
use crate::{Error, Result, binding::spaces};

/// One native run's resolved inputs.
pub(crate) struct LocalRun {
    pub(crate) address: String,
    pub(crate) token: String,
    /// The resolved budget (at least 1).
    pub(crate) episodes: u64,
    pub(crate) base_seed: Option<i64>,
    pub(crate) seeds: Vec<i64>,
    pub(crate) trial_index_base: Option<u64>,
    pub(crate) max_episode_steps: Option<i64>,
    pub(crate) max_episode_seconds: Option<f64>,
    pub(crate) close_env: bool,
    pub(crate) execution_horizon: u32,
    pub(crate) prefetch_lead: u32,
    pub(crate) workflow_edition: Option<String>,
}

/// Connect to the env, resolve the handler's route, and drive the runtime
/// loop to completion (the binding's run path). The run borrows the handler;
/// a zero-episode budget never dials the env.
pub(super) async fn run_local<H>(
    handler: &mut H,
    options: super::worker::RunLocalOptions,
    cancellation: tokio_util::sync::CancellationToken,
    hooks: Arc<dyn RuntimeHooks>,
) -> Result<RuntimeReport>
where
    H: ModelHandler + 'static,
{
    // Python's budget (`resolve_episode_budget`): one episode by default, one
    // per seed when only seeds are given. Zero runs nothing, so the env is
    // never dialed -- exactly Python's `run(episodes=0)`.
    let episodes = options.episode_budget()?;
    if episodes == 0 {
        return Ok(empty_report());
    }
    let run = LocalRun {
        address: options.env_address.to_string(),
        token: options.token,
        episodes,
        base_seed: options.base_seed,
        seeds: options.episode_seeds,
        trial_index_base: options.trial_index_base,
        max_episode_steps: options.max_episode_steps,
        max_episode_seconds: options.max_episode_seconds,
        close_env: options.close_env,
        execution_horizon: options.execution_horizon,
        prefetch_lead: options.prefetch_lead,
        workflow_edition: options.workflow_edition,
    };
    drive_local(handler, run, cancellation, hooks, &|_| {}).await
}

/// Connect to the env, resolve the handler's route, and drive the runtime
/// loop to completion. The route resolves before driving, exactly as the
/// served path does at `ResolveAdapter`: that arms the spec'd engine path
/// (adapter, per-episode frame buffers, batched/chunked predict corners) and
/// pins the execution horizon -- without it every predict would fall into the
/// spec-less branch. `connected` sees the env's contract once the handshake
/// answered, before the route resolves.
pub(crate) async fn drive_local<H>(
    handler: &mut H,
    options: LocalRun,
    cancellation: tokio_util::sync::CancellationToken,
    hooks: Arc<dyn RuntimeHooks>,
    connected: &(dyn Fn(&spaces::EnvContract) + Send + Sync),
) -> Result<RuntimeReport>
where
    H: ModelHandler + 'static,
{
    let (env, handshake) = connect_env(&options).await?;
    let env_ceiling = env_ceiling(&handshake);
    // A lane endpoint steps/resets lanes individually, so a num_envs > 1
    // session can run driver-owned (DISABLED) resets lane by lane.
    let subset_step = rlmesh_proto::has_capability(
        &handshake.capabilities,
        rlmesh_proto::capabilities::ENV_SUBSET_STEP,
    );
    let env_contract = env_contract_from_proto(handshake.env_contract)
        .map_err(|err| Error::Protocol(format!("invalid spaces spec from env: {err}")))?;
    // The handler returns typed actions the runtime encodes against this space;
    // a missing one would fail every predict, so reject at connect, not mid-run.
    if env_contract.action_space.is_none() {
        return Err(Error::Internal(
            "env contract has no action_space; a model cannot encode actions without it"
                .to_string(),
        ));
    }
    connected(&env_contract);
    // The runtime is the env-id authority (R1); for the in-process path mint a
    // UUIDv7 container id. The human env name lives on the SDK's own contract.
    let env_id = crate::mint_id();
    let num_envs = handshake.num_envs;
    let session_id = format!("local-{}", std::process::id());

    // Action chunking across a lockstep vector env would replay one whole-batch
    // chunk for every lane, and a lane that ends mid-chunk invalidates the buffer
    // for all of them. A lane endpoint replays per lane, so it is fine there.
    // The options and the env decide it together: ValueError at connect (RUN-P1).
    if num_envs > 1 && !subset_step && options.execution_horizon > 1 {
        return Err(Error::invalid_argument(format!(
            "execution_horizon={} cannot be combined with a lockstep vector env (num_envs={num_envs}): \
             chunk replay is whole-batch, so one lane's episode end discards every lane's \
             buffered frames. Use num_envs=1, a lane endpoint, or execution_horizon=1.",
            options.execution_horizon,
        )));
    }

    // The driver delivers every replayed step as a history row, so a stacked
    // adapter's window advances on every step at any horizon; the route answers
    // whether it needs that, and the driver only buffers rows when it does. The
    // handler is in-process (this build's engine), so unlike the served path
    // there is no peer capability to gate the offer on.
    let mut wants_history = false;
    if let Some(route_setup) = handler.route_setup() {
        let needs = route_setup
            .resolve_adapter(
                &env_id,
                &env_contract,
                crate::model::ResolveOptions {
                    execution_horizon: options.execution_horizon,
                    delivers_history: true,
                },
            )
            .await?;
        wants_history = needs.history.is_some();
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
        episode_seeds: options.seeds,
        max_episodes: Some(options.episodes),
        trial_index_base: options.trial_index_base,
        max_episode_steps: options.max_episode_steps,
        max_episode_seconds: options.max_episode_seconds,
        close_env_on_end: options.close_env,
        // A lane endpoint is driven one group per lane; the grouped predicts
        // reach the handler's `predict_grouped` (one fused forward for a model
        // with a batched corner).
        subset_step,
        limits: Default::default(),
        env_ceiling: Some(env_ceiling),
        model_ceiling: None,
    };
    let model = ModelHandlerRuntimeModel::new(handler, env_contract).with_history(wants_history);
    RuntimeDriver::new(spec, env, model, hooks)
        .with_prefetch(options.prefetch_lead)
        .run_with_cancellation_reason(cancellation, "interrupted by the host (signal)")
        .await
        .map_err(run_error)
}

/// Dials the env and handshakes: over gRPC, or (with the `lockstep`
/// feature) as a v2 lockstep driver (`lockstep://...`, §12), whose
/// handshake the env's declaration answers.
async fn connect_env(options: &LocalRun) -> Result<(DrivenEnv, rlmesh_grpc::EnvHandshake)> {
    if options.address.starts_with("lockstep://") {
        #[cfg(feature = "lockstep")]
        {
            let (env, handshake) =
                LockstepRuntimeEnv::connect(&options.address, options.workflow_edition.as_deref())
                    .await?;
            return Ok((DrivenEnv::Lockstep(env), handshake));
        }
        #[cfg(not(feature = "lockstep"))]
        return Err(Error::invalid_argument(format!(
            "{}: this build has no v2 lockstep transport (the `lockstep` feature)",
            options.address
        )));
    }
    // An address carries what a RemoteEnv dialed with no api_key would
    // (RLMESH_API_KEY over https://, RMT-22); a loopback server's token wins.
    let auth = crate::credentials::resolve(&options.address, None, None, false)?;
    let credentials = rlmesh_grpc::Credentials {
        token: if options.token.is_empty() {
            auth.token().to_string()
        } else {
            options.token.clone()
        },
        trust: auth.trust().clone(),
    };
    let mut env = rlmesh_grpc::EnvClient::connect_with(&options.address, &credentials)
        .await
        .map_err(Error::from_grpc)?;
    // The runtime tier's WANT rides this leg's handshake and caps the env-leg
    // negotiation, so it has to be declared before the handshake goes out.
    env.declare_workflow_edition(options.workflow_edition.clone());
    let handshake = env.handshake().await.map_err(Error::from_grpc)?;
    Ok((
        DrivenEnv::Grpc(Box::new(EnvClientRuntimeEnv::new(env))),
        handshake,
    ))
}

/// The env a local run drives.
#[derive(Clone)]
enum DrivenEnv {
    Grpc(Box<EnvClientRuntimeEnv>),
    #[cfg(feature = "lockstep")]
    Lockstep(LockstepRuntimeEnv),
}

#[async_trait]
impl RuntimeEnv for DrivenEnv {
    async fn reset(
        &mut self,
        request: rlmesh_proto::env::v1::ResetRequest,
    ) -> std::result::Result<RuntimeEnvReset, rlmesh_runtime::RuntimeError> {
        match self {
            Self::Grpc(env) => env.reset(request).await,
            #[cfg(feature = "lockstep")]
            Self::Lockstep(env) => env.reset(request).await,
        }
    }

    async fn step(
        &mut self,
        request: rlmesh_proto::env::v1::StepRequest,
    ) -> std::result::Result<RuntimeEnvStep, rlmesh_runtime::RuntimeError> {
        match self {
            Self::Grpc(env) => env.step(request).await,
            #[cfg(feature = "lockstep")]
            Self::Lockstep(env) => env.step(request).await,
        }
    }

    async fn close(&mut self, timeout: Duration) -> std::result::Result<(), String> {
        match self {
            Self::Grpc(env) => env.close(timeout).await,
            #[cfg(feature = "lockstep")]
            Self::Lockstep(env) => env.close(timeout).await,
        }
    }
}

/// A lockstep env as a v1 [`RuntimeEnv`], so the v1 runtime's chunking,
/// prefetch, batching, autoreset handling and driver-minted episode ids run
/// unchanged over lockstep (§12.4). It holds the v1 handshake the env's
/// `gym_env` answers, and an [`EpisodeMirror`](rlmesh_grpc::env::EpisodeMirror)
/// that rebuilds the completions the env server would have reported (a
/// lockstep reply carries no episode metadata). Clones share one mirror (the
/// runtime holds one env per lane group).
///
/// TODO: the last v1 touchpoint on the lockstep path; delete it when
/// `Model.run` drives a `LockstepClient` without `RuntimeDriver`.
#[cfg(feature = "lockstep")]
#[derive(Clone)]
pub struct LockstepRuntimeEnv {
    client: crate::lockstep::LockstepClient,
    mirror: Arc<std::sync::Mutex<rlmesh_grpc::env::EpisodeMirror>>,
}

#[cfg(feature = "lockstep")]
impl LockstepRuntimeEnv {
    /// Connects to the lockstep env at `address` and answers the v1
    /// handshake for `declared` (the runtime's WANT) from its declaration.
    pub async fn connect(
        address: &str,
        declared: Option<&str>,
    ) -> Result<(Self, rlmesh_grpc::EnvHandshake)> {
        let client = crate::lockstep::connect(address, Default::default())
            .await
            .map_err(|err| Error::Connection(err.to_string()))?;
        let info = client.info();
        // The v1 runtime drives the env through its v1 contract.
        rlmesh_grpc::wire::check_proto_contract(&info.contract)
            .map_err(|err| Error::invalid_argument(err.to_string()))?;
        let (offer, selected, workflow_edition) =
            crate::editions::negotiate_env(info, declared).map_err(Error::Protocol)?;
        let capabilities: Vec<&str> = info.capabilities.iter().map(String::as_str).collect();
        let num_envs = (info.contract.num_envs as usize).max(1);
        let handshake = rlmesh_grpc::EnvHandshake {
            env_contract: env_contract_to_proto(&info.contract),
            num_envs,
            workflow_edition,
            selected_workflow_edition: selected,
            supported_workflow_editions: offer.editions,
            preferred_workflow_edition: offer.preferred,
            capabilities: rlmesh_proto::capability_map(&capabilities),
        };
        let mirror = rlmesh_grpc::env::EpisodeMirror::new(num_envs, info.contract.autoreset_mode);
        Ok((
            Self {
                client,
                mirror: Arc::new(std::sync::Mutex::new(mirror)),
            },
            handshake,
        ))
    }

    fn spaces(&self) -> std::result::Result<(&spaces::SpaceSpec, &spaces::SpaceSpec), GymError> {
        let contract = &self.client.info().contract;
        contract
            .observation_space
            .as_ref()
            .zip(contract.action_space.as_ref())
            .ok_or_else(|| GymError::Map("the env has no spaces".into()))
    }

    fn mirror(&self) -> std::sync::MutexGuard<'_, rlmesh_grpc::env::EpisodeMirror> {
        self.mirror
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    async fn reset_v1(
        &self,
        request: rlmesh_proto::env::v1::ResetRequest,
    ) -> std::result::Result<rlmesh_proto::env::v1::ResetResponse, GymError> {
        let options = request.options.clone().map(meta_map_from_proto);
        let reset = self
            .client
            .reset(
                &request.env_indices,
                &request.seeds,
                options.as_ref(),
                lockstep_timeout(request.timeout_ms),
            )
            .await?;
        let observation = encode_batched_partial_values(&reset.observations, self.spaces()?.0)
            .map_err(|err| GymError::Malformed(err.to_string()))?;
        self.mirror().on_reset(&request);
        Ok(rlmesh_proto::env::v1::ResetResponse {
            observation: Some(observation),
            infos: reset.info.as_ref().map(meta_map_to_proto),
        })
    }

    async fn step_v1(
        &self,
        request: rlmesh_proto::env::v1::StepRequest,
    ) -> std::result::Result<rlmesh_proto::env::v1::StepResponse, GymError> {
        let (observation_space, action_space) = self.spaces()?;
        let width = if request.env_indices.is_empty() {
            (self.client.info().contract.num_envs as usize).max(1)
        } else {
            request.env_indices.len()
        };
        let actions = decode_batched_partial_values(request.action.as_ref(), action_space, width)
            .map_err(|err| GymError::Malformed(err.to_string()))?;
        let step = self
            .client
            .step(
                &request.env_indices,
                &actions,
                lockstep_timeout(request.timeout_ms),
            )
            .await?;
        let mask = |flags: &[bool]| flags.iter().map(|f| u8::from(*f)).collect();
        let mut response = rlmesh_proto::env::v1::StepResponse {
            observation: Some(
                encode_batched_partial_values(&step.observations, observation_space)
                    .map_err(|err| GymError::Malformed(err.to_string()))?,
            ),
            rewards: step.rewards,
            terminated_mask: mask(&step.terminated),
            truncated_mask: mask(&step.truncated),
            infos: step.info.as_ref().map(meta_map_to_proto),
            completed_episodes: Vec::new(),
            env_indices: request.env_indices.clone(),
        };
        self.mirror().on_step(&request, &mut response);
        Ok(response)
    }
}

/// A lockstep call's bound: the request's own `timeout_ms`, when set.
#[cfg(feature = "lockstep")]
fn lockstep_timeout(ms: u64) -> Option<Duration> {
    (ms > 0).then(|| Duration::from_millis(ms))
}

/// Maps a lockstep failure onto the runtime's env error, classified as the
/// gRPC env client classifies the same failure.
#[cfg(feature = "lockstep")]
fn lockstep_runtime_error(operation: &'static str, error: GymError) -> RuntimeError {
    use rlmesh_grpc::error::{Error as GrpcError, ProtocolError, TransportError};
    let recoverable = error.is_recoverable();
    let source = match error {
        GymError::Env(error) => GrpcError::Environment(error.to_wire()),
        GymError::Timeout(t) => GrpcError::Timeout(t),
        GymError::Closed(why) => GrpcError::Transport(TransportError::ConnectFailed(why)),
        other => GrpcError::Protocol(ProtocolError::DecodeError(other.to_string())),
    };
    RuntimeError::env_rpc_with_recoverability(operation, 0, recoverable, source)
}

#[cfg(feature = "lockstep")]
#[async_trait]
impl RuntimeEnv for LockstepRuntimeEnv {
    async fn reset(
        &mut self,
        request: rlmesh_proto::env::v1::ResetRequest,
    ) -> std::result::Result<RuntimeEnvReset, RuntimeError> {
        let response = self
            .reset_v1(request)
            .await
            .map_err(|e| lockstep_runtime_error("env.reset", e))?;
        Ok(RuntimeEnvReset {
            response,
            endpoint_total_ns: None,
            phases: Default::default(),
        })
    }

    async fn step(
        &mut self,
        request: rlmesh_proto::env::v1::StepRequest,
    ) -> std::result::Result<RuntimeEnvStep, RuntimeError> {
        let response = self
            .step_v1(request)
            .await
            .map_err(|e| lockstep_runtime_error("env.step", e))?;
        Ok(RuntimeEnvStep {
            response,
            endpoint_total_ns: None,
            phases: Default::default(),
        })
    }

    async fn close(&mut self, timeout: Duration) -> std::result::Result<(), String> {
        tokio::time::timeout(timeout, self.client.close())
            .await
            .map_err(|_| format!("the lockstep driver did not detach within {timeout:?}"))
    }
}

/// The report of a zero-episode run: nothing ran, nothing was measured.
pub(crate) fn empty_report() -> RuntimeReport {
    RuntimeReport {
        session_id: format!("local-{}", std::process::id()),
        env_id: String::new(),
        total_steps: 0,
        total_episodes: 0,
        episodes: Vec::new(),
        telemetry: rlmesh_runtime::telemetry::Snapshot {
            horizon: rlmesh_runtime::telemetry::Horizon::Session,
            rows: Vec::new(),
        },
        advisories: Vec::new(),
    }
}

/// What the served env on the other end of `handshake` can decode.
pub(super) fn env_ceiling(handshake: &rlmesh_grpc::EnvHandshake) -> PeerCeiling {
    PeerCeiling::wire_v1(
        PeerCeiling::highest_shared_edition(&handshake.supported_workflow_editions)
            .unwrap_or(handshake.workflow_edition),
        handshake.capabilities.clone(),
        rlmesh_grpc::MAX_MESSAGE_SIZE,
    )
}

/// Wrap a facade [`Error`] as a driver model-RPC failure, keeping its
/// recoverable flag (the default `model_rpc` constructor drops it, so a
/// recoverable handler decline used to reach the caller as permanent).
fn model_rpc(error: Error) -> RuntimeError {
    RuntimeError::model_rpc_with_recoverability("local-model", error.is_transient(), error)
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
            .map(|error| Error::from_grpc(*error))
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
        RuntimeError::Protocol(_) => Error::Protocol(message),
        // A spec the options and the env cannot run together (autoreset,
        // vector-env refusals): ValueError at connect (RUN-P1).
        RuntimeError::InvalidSpec(_) => Error::invalid_argument(message),
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
        match tokio::time::timeout(timeout, self.inner.close()).await {
            Ok(result) => result.map(|_| ()).map_err(|err| err.to_string()),
            Err(err) => {
                self.inner.detach();
                Err(err.to_string())
            }
        }
    }
}

/// Adapts a [`ModelHandler`] to the [`RuntimeModel`] trait expected by
/// [`rlmesh_runtime::RuntimeDriver`].
///
/// Use this to drive your handler from your own `RuntimeDriver` embedding. The
/// adapter decodes the runtime's predict request into a
/// [`ModelObservation`](crate::binding::model::ModelObservation), runs `predict`, and re-encodes
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
    /// The route asked for observation history at resolve (see
    /// [`with_history`](Self::with_history)).
    wants_history: bool,
}

impl<'a, H> ModelHandlerRuntimeModel<'a, H> {
    /// Build an adapter for `handler` against the given env contract.
    pub fn new(handler: &'a mut H, env_contract: spaces::EnvContract) -> Self {
        Self {
            handler: Arc::new(tokio::sync::Mutex::new(handler)),
            env_contract: Arc::new(env_contract),
            wants_history: false,
        }
    }

    /// Tell the driver whether the route negotiated observation history
    /// (`RouteNeeds::history` was answered at resolve), so it buffers every
    /// replayed step as a history row on the next predict.
    pub fn with_history(mut self, wants_history: bool) -> Self {
        self.wants_history = wants_history;
        self
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
    fn wants_history(&self) -> bool {
        self.wants_history
    }

    /// `predict_group` below is one fused forward, so the driver batches the
    /// lanes waiting on it rather than predicting each on its own.
    fn fuses_predicts(&self) -> bool {
        true
    }

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
        // Route the driver's explicit episode-end GC to the route setup's
        // eviction and then the handler's own hook, as the served path does.
        let env_id = request
            .context
            .map(|context| context.env_id)
            .unwrap_or_default();
        let mut handler = self.handler.lock().await;
        if let Some(route_setup) = handler.route_setup() {
            route_setup
                .reset_adapter(&env_id, &request.episode_ids)
                .await
                .map_err(model_rpc)?;
        }
        handler
            .reset_adapter(&env_id, request.episode_ids)
            .await
            .map_err(model_rpc)
    }
}
