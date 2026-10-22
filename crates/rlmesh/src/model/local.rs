use std::sync::Arc;
use std::time::Duration;

use crate::run::runtime::{
    EnvResetRequest, EnvStepRequest, RuntimeDriver, RuntimeEnv, RuntimeEnvReset, RuntimeEnvStep,
    RuntimeError, RuntimeHooks, RuntimeReport, RuntimeSessionSpec,
};
use async_trait::async_trait;
use rlmesh_grpc::wire::{env_contract_from_proto, env_contract_to_proto};

use super::handler::ModelHandler;
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
    let Connected {
        env,
        contract: env_contract,
        num_envs,
        subset_step,
        workflow_edition,
    } = connect_env(&options).await?;
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
    // whether it needs that, and the driver only buffers rows when it does.
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
        workflow_edition,
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
    };
    RuntimeDriver::new(spec, env, handler, env_contract, hooks)
        .with_history(wants_history)
        .with_prefetch(options.prefetch_lead)
        .run_with_cancellation_reason(cancellation, "interrupted by the host (signal)")
        .await
        .map_err(run_error)
}

/// A connected env: what the driver steps, and what the connection learned.
struct Connected {
    env: DrivenEnv,
    contract: spaces::EnvContract,
    num_envs: usize,
    /// Lanes reset and step individually.
    subset_step: bool,
    /// The edition negotiated with the env.
    workflow_edition: rlmesh_proto::Edition,
}

/// Dials the env and handshakes: as a v2 lockstep driver (a participant
/// address, §12, with the `lockstep` feature), whose declaration answers the
/// handshake, or over gRPC.
async fn connect_env(options: &LocalRun) -> Result<Connected> {
    if crate::address::is_v2(&options.address) {
        #[cfg(feature = "lockstep")]
        return LockstepEnv::connect(&options.address, options.workflow_edition.as_deref()).await;
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
    Ok(Connected {
        subset_step: rlmesh_proto::has_capability(
            &handshake.capabilities,
            rlmesh_proto::capabilities::ENV_SUBSET_STEP,
        ),
        contract: env_contract_from_proto(handshake.env_contract)
            .map_err(|err| Error::Protocol(format!("invalid spaces spec from env: {err}")))?,
        num_envs: handshake.num_envs,
        workflow_edition: handshake.workflow_edition,
        env: DrivenEnv::Grpc(Box::new(EnvClientRuntimeEnv::new(env))),
    })
}

/// The env a local run drives.
#[derive(Clone)]
enum DrivenEnv {
    Grpc(Box<EnvClientRuntimeEnv>),
    #[cfg(feature = "lockstep")]
    Lockstep(LockstepEnv),
}

#[async_trait]
impl RuntimeEnv for DrivenEnv {
    async fn reset(
        &mut self,
        request: EnvResetRequest,
    ) -> std::result::Result<RuntimeEnvReset, RuntimeError> {
        match self {
            Self::Grpc(env) => env.reset(request).await,
            #[cfg(feature = "lockstep")]
            Self::Lockstep(env) => env.reset(request).await,
        }
    }

    async fn step(
        &mut self,
        request: EnvStepRequest,
    ) -> std::result::Result<RuntimeEnvStep, RuntimeError> {
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

/// A lockstep env (§12) as the driver's env: resets and steps on leaf slabs,
/// the env's own timing reported beside each answer.
#[cfg(feature = "lockstep")]
#[derive(Clone)]
pub struct LockstepEnv {
    client: crate::lockstep::LockstepClient,
}

#[cfg(feature = "lockstep")]
impl LockstepEnv {
    /// The env `client` drives.
    pub fn new(client: crate::lockstep::LockstepClient) -> Self {
        Self { client }
    }

    /// Connects to the lockstep env at `address`, negotiating the edition
    /// with `declared` (the runtime's WANT) from its declaration.
    async fn connect(address: &str, declared: Option<&str>) -> Result<Connected> {
        let client = crate::lockstep::connect(address, Default::default())
            .await
            .map_err(|err| Error::Connection(err.to_string()))?;
        let info = client.info();
        if info.contract.autoreset_mode == spaces::types::AutoresetMode::SameStep {
            return Err(Error::invalid_argument(
                "Model.run does not drive a SAME_STEP autoreset env; serve it with NEXT_STEP or \
                 DISABLED autoreset, or drive it with RemoteVectorEnv",
            ));
        }
        // The driver checks the env's contract as a v1 handshake did.
        rlmesh_grpc::wire::check_proto_contract(&info.contract)
            .map_err(|err| Error::invalid_argument(err.to_string()))?;
        let (_, _, workflow_edition) =
            crate::editions::negotiate_env(info, declared).map_err(Error::Protocol)?;
        Ok(Connected {
            contract: info.contract.clone(),
            num_envs: (info.contract.num_envs as usize).max(1),
            subset_step: info.subset_step(),
            workflow_edition,
            env: DrivenEnv::Lockstep(Self::new(client)),
        })
    }

    /// The env's timing of its last answer, as the driver records it.
    fn timing(&self) -> (Option<u64>, rlmesh_proto::EndpointPhases) {
        let Some(timing) = self.client.last_timing() else {
            return (None, rlmesh_proto::EndpointPhases::default());
        };
        (
            timing.total_ns,
            rlmesh_proto::EndpointPhases {
                decode_ns: timing.decode_ns.unwrap_or(0),
                user_ns: timing.user_ns.unwrap_or(0),
                encode_ns: timing.encode_ns.unwrap_or(0),
                queue_ns: timing.queue_ns.unwrap_or(0),
                lane_skew_ns: timing.lane_skew_ns,
                ..rlmesh_proto::EndpointPhases::default()
            },
        )
    }
}

#[cfg(feature = "lockstep")]
#[async_trait]
impl RuntimeEnv for LockstepEnv {
    async fn reset(
        &mut self,
        request: EnvResetRequest,
    ) -> std::result::Result<RuntimeEnvReset, RuntimeError> {
        let options = request.options.map(rlmesh_grpc::wire::meta_map_from_proto);
        let (observation, info) = self
            .client
            .reset_leaves(&request.env_indices, &request.seeds, options.as_ref(), None)
            .await
            .map_err(|e| lockstep_runtime_error("env.reset", e))?;
        let (endpoint_total_ns, phases) = self.timing();
        Ok(RuntimeEnvReset {
            observation,
            infos: info.as_ref().map(rlmesh_grpc::wire::meta_map_to_proto),
            endpoint_total_ns,
            phases,
        })
    }

    async fn step(
        &mut self,
        request: EnvStepRequest,
    ) -> std::result::Result<RuntimeEnvStep, RuntimeError> {
        let step = self
            .client
            .step_leaves(&request.env_indices, &request.action, None)
            .await
            .map_err(|e| lockstep_runtime_error("env.step", e))?;
        let (endpoint_total_ns, phases) = self.timing();
        Ok(RuntimeEnvStep {
            observation: step.observation,
            rewards: step.rewards,
            terminated: step.terminated,
            truncated: step.truncated,
            infos: step.info.as_ref().map(rlmesh_grpc::wire::meta_map_to_proto),
            endpoint_total_ns,
            phases,
        })
    }

    async fn close(&mut self, timeout: Duration) -> std::result::Result<(), String> {
        tokio::time::timeout(timeout, self.client.close())
            .await
            .map_err(|_| format!("the lockstep driver did not detach within {timeout:?}"))
    }
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

/// The report of a zero-episode run: nothing ran, nothing was measured.
pub(crate) fn empty_report() -> RuntimeReport {
    RuntimeReport {
        session_id: format!("local-{}", std::process::id()),
        env_id: String::new(),
        total_steps: 0,
        total_episodes: 0,
        episodes: Vec::new(),
        telemetry: crate::run::runtime::telemetry::Snapshot {
            horizon: crate::run::runtime::telemetry::Horizon::Session,
            rows: Vec::new(),
        },
    }
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
                class: None,
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

/// A connected [`rlmesh_grpc::EnvClient`] as the driver's env: the v1 wire
/// until v1 env serving goes.
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

fn env_rpc(operation: &'static str, error: rlmesh_grpc::error::Error) -> RuntimeError {
    let recoverable = error.is_recoverable();
    RuntimeError::env_rpc_with_recoverability(operation, 0, recoverable, error)
}

fn mask(bytes: &[u8], lanes: usize) -> Vec<bool> {
    (0..lanes)
        .map(|lane| bytes.get(lane).is_some_and(|bit| *bit != 0))
        .collect()
}

#[async_trait]
impl RuntimeEnv for EnvClientRuntimeEnv {
    async fn reset(
        &mut self,
        request: EnvResetRequest,
    ) -> std::result::Result<RuntimeEnvReset, RuntimeError> {
        let response = self
            .inner
            .reset(rlmesh_proto::env::v1::ResetRequest {
                seeds: request.seeds,
                options: request.options,
                timeout_ms: 0,
                env_indices: request.env_indices,
                episode_ids: request.episode_ids,
            })
            .await
            .map_err(|err| env_rpc("env.reset", err))?;
        Ok(RuntimeEnvReset {
            observation: response.observation.map(|v| v.leaves).unwrap_or_default(),
            infos: response.infos,
            endpoint_total_ns: self.inner.take_last_endpoint_total_ns(),
            phases: self.inner.take_last_phases(),
        })
    }

    async fn step(
        &mut self,
        request: EnvStepRequest,
    ) -> std::result::Result<RuntimeEnvStep, RuntimeError> {
        let response = self
            .inner
            .step(rlmesh_proto::env::v1::StepRequest {
                action: Some(rlmesh_grpc::wire::leaves_value(request.action)),
                timeout_ms: 0,
                env_indices: request.env_indices,
                episode_ids: request.episode_ids,
            })
            .await
            .map_err(|err| env_rpc("env.step", err))?;
        let lanes = response.rewards.len();
        Ok(RuntimeEnvStep {
            observation: response.observation.map(|v| v.leaves).unwrap_or_default(),
            terminated: mask(&response.terminated_mask, lanes),
            truncated: mask(&response.truncated_mask, lanes),
            rewards: response.rewards,
            infos: response.infos,
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
