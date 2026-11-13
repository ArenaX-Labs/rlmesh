use std::sync::Arc;
use std::time::Duration;

use crate::data::env_contract_to_proto;
use crate::run::runtime::{
    EnvResetRequest, EnvStepRequest, RuntimeDriver, RuntimeEnv, RuntimeEnvReset, RuntimeEnvStep,
    RuntimeError, RuntimeHooks, RuntimeReport, RuntimeSessionSpec,
};
use async_trait::async_trait;

use super::handler::ModelHandler;
use crate::lockstep::GymError;
use crate::{Error, Result, binding::spaces};

/// One native run's resolved inputs.
pub(crate) struct LocalRun {
    pub(crate) address: String,
    /// The resolved budget (at least 1).
    pub(crate) episodes: u64,
    pub(crate) base_seed: Option<i64>,
    /// What a `base_seed` keys on; `None` is the env's own id.
    pub(crate) seed_key: Option<String>,
    /// A multi-route run's start: connect retry and the start barrier.
    pub(crate) start: Option<crate::run::routes::RouteStart>,
    pub(crate) seeds: Vec<i64>,
    pub(crate) trial_index_base: Option<u64>,
    pub(crate) max_episode_steps: Option<i64>,
    pub(crate) max_episode_seconds: Option<f64>,
    pub(crate) close_env: bool,
    pub(crate) execution_horizon: u32,
    pub(crate) prefetch_lead: u32,
    pub(crate) workflow_edition: Option<String>,
    pub(crate) invite_at: Option<String>,
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
        address: options.env_address.as_str(),
        episodes,
        base_seed: options.base_seed,
        seed_key: None,
        start: None,
        seeds: options.episode_seeds,
        trial_index_base: options.trial_index_base,
        max_episode_steps: options.max_episode_steps,
        max_episode_seconds: options.max_episode_seconds,
        close_env: options.close_env,
        execution_horizon: options.execution_horizon,
        prefetch_lead: options.prefetch_lead,
        workflow_edition: options.workflow_edition,
        invite_at: options.invite_at,
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
    mut options: LocalRun,
    cancellation: tokio_util::sync::CancellationToken,
    hooks: Arc<dyn RuntimeHooks>,
    connected: &(dyn Fn(&spaces::EnvContract) + Send + Sync),
) -> Result<RuntimeReport>
where
    H: ModelHandler + 'static,
{
    // A route of a multi-route run retries its connect (its env may still
    // be coming up) and holds at the start barrier once it is ready; leaving
    // early, on any error below, counts it as arrived.
    let start = options.start.take();
    let ConnectedEnv {
        env,
        contract: env_contract,
        num_envs,
        subset_step,
        edition,
        session,
    } = match &start {
        Some(start) => {
            crate::run::routes::connect_with_retry(start.connect_timeout, &cancellation, || {
                connect_env(&options)
            })
            .await?
        }
        None => connect_env(&options).await?,
    };
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

    // A model served elsewhere answers in the env's session.
    if let Some(session) = options.invite_at.as_ref().or(session.as_ref()) {
        handler.join(session).await?;
    }
    let workflow_edition = edition
        .with_model(handler, options.workflow_edition.as_deref())
        .await?;

    // The driver delivers every replayed step as a history row, so a stacked
    // adapter's window advances on every step at any horizon; the route answers
    // whether it needs that, and the driver only buffers rows when it does.
    let mut wants_history = false;
    let mut advisories = Vec::new();
    if let Some(route_setup) = handler.route_setup() {
        let needs = route_setup
            .resolve_adapter(
                &env_id,
                &env_contract,
                crate::model::ResolveOptions {
                    execution_horizon: options.execution_horizon,
                    delivers_history: true,
                    workflow_edition: Some(workflow_edition),
                },
            )
            .await?;
        wants_history = needs.history.is_some();
        advisories = needs.advisories;
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
        // The env's own id unless the route names one: what no model can
        // change, so models pair.
        seed_key: options.seed_key.unwrap_or_else(|| env_contract.id.clone()),
        episode_seeds: options.seeds,
        max_episodes: Some(options.episodes),
        trial_index_base: options.trial_index_base,
        max_episode_steps: options.max_episode_steps,
        max_episode_seconds: options.max_episode_seconds,
        close_env_on_end: options.close_env,
        // `Model.run` closes its own model after the run.
        close_model_on_end: false,
        // A lane endpoint is driven one group per lane; the grouped predicts
        // reach the handler's `predict_grouped` (one fused forward for a model
        // with a batched corner).
        subset_step,
        limits: Default::default(),
    };
    if let Some(start) = start {
        start.arrival.arrive(&cancellation).await;
    }
    RuntimeDriver::new(spec, DrivenEnv::Lockstep(env), handler, env_contract, hooks)
        .with_history(wants_history)
        .with_prefetch(options.prefetch_lead)
        .run_with_cancellation_reason(cancellation, "interrupted by the host (signal)")
        .await
        .map(|report| RuntimeReport {
            advisories,
            ..report
        })
        .map_err(run_error)
}

/// A lockstep env connected for the runtime driver
/// ([`LockstepEnv::connect`]): the env, and what the connection learned.
pub struct ConnectedEnv {
    /// The env, for [`RuntimeDriver::new`].
    pub env: LockstepEnv,
    /// Its contract, for [`RuntimeDriver::new`] and (through
    /// `env_contract_to_proto`) [`RuntimeSessionSpec::env_contract`].
    pub contract: spaces::EnvContract,
    /// Its lanes (at least 1): [`RuntimeSessionSpec::num_envs`].
    pub num_envs: usize,
    /// Its lanes reset and step individually:
    /// [`RuntimeSessionSpec::subset_step`].
    pub subset_step: bool,
    /// The session the env is in, with its key (a credential): where a
    /// model served elsewhere is invited ([`ModelHandler::join`]).
    pub session: Option<String>,
    /// The edition negotiated with the env.
    edition: RunEdition,
}

impl ConnectedEnv {
    /// The edition a run with `handler` executes at, for
    /// [`RuntimeSessionSpec::workflow_edition`]: the one negotiated with the
    /// env, or with a model served elsewhere that declared its editions
    /// (after [`ModelHandler::join`]), the floor across the env, the model
    /// and `declared`, as `Model.run` settles it.
    ///
    /// # Errors
    /// The model and the env share no edition.
    pub async fn workflow_edition<H: ModelHandler + ?Sized>(
        &self,
        handler: &mut H,
        declared: Option<&str>,
    ) -> Result<rlmesh_proto::Edition> {
        self.edition.with_model(handler, declared).await
    }
}

/// Connect to a session env and negotiate its declared edition.
async fn connect_env(options: &LocalRun) -> Result<ConnectedEnv> {
    {
        if !crate::address::is_v2(&options.address) {
            return Err(Error::invalid_argument(format!(
                "Model.run({:?}): a v1 env endpoint, which rlmesh 0.2 no longer dials; \
                 serve the env at a session address (EnvServer(env, \":0/name\")) and run \
                 against that, or pass the env object itself",
                crate::address::redact(&options.address)
            )));
        }
        if options.close_env {
            return Err(Error::invalid_argument(
                "close_env=True on a session address is gone in rlmesh 0.2: the env belongs \
                 to its serving process; an owner shutdown is server.shutdown() there",
            ));
        }
        LockstepEnv::connect(
            &options.address,
            crate::lockstep::ConnectOptions::default(),
            options.workflow_edition.as_deref(),
        )
        .await
    }
}

/// The edition a connection to an env negotiated, and the env's offer.
pub(crate) struct RunEdition {
    /// The edition selected with the env, as spelled on the wire.
    selected: String,
    /// The env's offer, on the v2 path.
    env: Option<rlmesh_proto::SessionOffer>,
}

impl RunEdition {
    /// The edition a run executes at: the env's, or with a model served
    /// elsewhere that declared its editions, the floor across the env, the
    /// model and the driver's `declared` WANT (pinned on the model's route).
    pub(crate) async fn with_model<H: ModelHandler + ?Sized>(
        &self,
        handler: &mut H,
        declared: Option<&str>,
    ) -> Result<rlmesh_proto::Edition> {
        let selected = match (&self.env, handler.workflow_editions().await?) {
            // TODO: pin the env to the floor on reset once an edition changes
            // what an env does; while every retained edition's defaults row is
            // equal, the floor is the env's (guarded by rlmesh-proto's
            // `retained_edition_rows_stay_equal_until_an_env_is_pinned_to_the_floor`).
            (Some(env), Some(model)) => {
                crate::floor::env_floor(env, &model, declared)?.selected_workflow_edition
            }
            _ => self.selected.clone(),
        };
        rlmesh_proto::parse_retained_edition(&selected).map_err(Error::Protocol)
    }
}

/// The env a local run drives.
#[derive(Clone)]
pub(crate) enum DrivenEnv {
    Lockstep(LockstepEnv),
}

#[async_trait]
impl RuntimeEnv for DrivenEnv {
    async fn reset(
        &mut self,
        request: EnvResetRequest,
    ) -> std::result::Result<RuntimeEnvReset, RuntimeError> {
        match self {
            Self::Lockstep(env) => env.reset(request).await,
        }
    }

    async fn step(
        &mut self,
        request: EnvStepRequest,
    ) -> std::result::Result<RuntimeEnvStep, RuntimeError> {
        match self {
            Self::Lockstep(env) => env.step(request).await,
        }
    }

    async fn close(&mut self, timeout: Duration) -> std::result::Result<(), String> {
        match self {
            Self::Lockstep(env) => env.close(timeout).await,
        }
    }
}

/// A lockstep env as the driver's env: resets and steps on leaf slabs,
/// the env's own timing reported beside each answer.
#[derive(Clone)]
pub struct LockstepEnv {
    client: crate::lockstep::LockstepClient,
}

impl LockstepEnv {
    /// The env `client` drives.
    pub fn new(client: crate::lockstep::LockstepClient) -> Self {
        Self { client }
    }

    /// The client driving the env.
    pub(crate) fn client(&self) -> &crate::lockstep::LockstepClient {
        &self.client
    }

    /// Connects to the lockstep env at the participant address `address`
    /// for the runtime driver, checking what `Model.run` checks before it
    /// drives one: an autoreset mode and spaces the driver represents, and a
    /// workflow edition the env and this build share, negotiated against
    /// `declared` (the runner's own declaration; `None`: this build's
    /// current). The path for a runner that builds its own
    /// [`RuntimeDriver`]; [`LockstepEnv::new`] over a bare
    /// [`crate::lockstep::connect`] skips these checks.
    ///
    /// # Errors
    /// [`ErrorExt::is_transient`](crate::ext::ErrorExt::is_transient) says
    /// whether retrying can succeed: a host not
    /// listening yet or an env not joined yet is transient; a key the host
    /// refuses, an edition the env does not run, or a contract the driver
    /// cannot represent is final.
    pub async fn connect(
        address: &str,
        options: crate::lockstep::ConnectOptions,
        declared: Option<&str>,
    ) -> Result<ConnectedEnv> {
        let client = crate::lockstep::connect(address, options)
            .await
            .map_err(crate::env::client::lockstep_error)?;
        let info = client.info();
        if info.contract.autoreset_mode == spaces::types::AutoresetMode::SameStep {
            return Err(Error::invalid_argument(
                "the runtime driver (Model.run) does not drive a SAME_STEP autoreset env; serve \
                 it with NEXT_STEP or DISABLED autoreset, or drive it with RemoteVectorEnv",
            ));
        }
        // The driver checks the env's contract as a v1 handshake did.
        crate::data::validate_driver_contract(&info.contract)
            .map_err(|err| Error::invalid_argument(err.to_string()))?;
        let (env, selected, _) =
            crate::editions::negotiate_env(info, declared).map_err(Error::Protocol)?;
        Ok(ConnectedEnv {
            contract: info.contract.clone(),
            num_envs: (info.contract.num_envs as usize).max(1),
            subset_step: info.subset_step(),
            edition: RunEdition {
                selected,
                env: Some(env),
            },
            session: client.session().map(str::to_string),
            env: Self::new(client),
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

#[async_trait]
impl RuntimeEnv for LockstepEnv {
    async fn reset(
        &mut self,
        request: EnvResetRequest,
    ) -> std::result::Result<RuntimeEnvReset, RuntimeError> {
        let (observation, info) = self
            .client
            .reset_leaves(
                &request.env_indices,
                &request.seeds,
                request.options.as_ref(),
                None,
            )
            .await
            .map_err(|e| lockstep_runtime_error("env.reset", e))?;
        let (endpoint_total_ns, phases) = self.timing();
        Ok(RuntimeEnvReset {
            observation,
            infos: info.as_ref().map(crate::data::meta_map_to_proto),
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
            infos: step.info.as_ref().map(crate::data::meta_map_to_proto),
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

/// Preserve a session failure's class and recoverability in the driver.
fn lockstep_runtime_error(operation: &'static str, error: GymError) -> RuntimeError {
    let recoverable = error.is_recoverable();
    let source = match error {
        GymError::Env(error) => Error::Environment(error),
        GymError::Timeout(t) => Error::Timeout(t),
        GymError::Closed(why) => Error::Connection(why),
        other => Error::Protocol(other.to_string()),
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
        advisories: Vec::new(),
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
            .and_then(|source| source.downcast::<Error>().ok())
            .map(|error| *error)
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
