//! `Model::run` on the native runtime loop (RUN-8, RUN-9, RUN-25).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rlmesh_grpc::wire::{
    decode_batched_partial_values, leaves_value, meta_map_from_proto, space_spec_from_proto,
};
use rlmesh_runtime::{
    ActionReceivedEvent, EpisodeCompletedEvent, EpisodeStartedEvent, HookError,
    ObservationEmittedEvent, RuntimeHooks, StepCompletedEvent,
};
use tokio_util::sync::CancellationToken;

use super::drive::{
    self, CallerPort, CallerSide, CancelWatch, LocalEnv, block_on_caller, caller_channel,
};
use super::hooks::LibraryContext;
use super::lent::{LentEnv, LentVectorEnv};
use super::options::{check_caps, check_horizon, check_prefetch, episode_budget};
use super::results::{self, EpisodeResult, RunResult, StepEvent};
use super::targets::{EnvKind, EnvTarget};
use super::{LoopbackEnv, RunHooks, RunOptions};
use crate::editions::EditionSources;
use crate::model::local::{LocalRun, drive_local};
use crate::value::{meta_codec, space_codec};
use crate::{EnvServer, Error, MadeEnv, Model, Result, Value};

/// How a call waits: on the caller's thread in segments, or as a future.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    Sync,
    Async,
}

impl Mode {
    /// Drive `future`, running caller jobs as this mode does.
    pub(crate) async fn drive<F: std::future::Future>(
        self,
        future: F,
        queue: &mut drive::CallerQueue,
        side: &mut CallerSide<'_>,
    ) -> F::Output {
        match self {
            Mode::Sync => block_on_caller(future, queue, side),
            Mode::Async => drive::await_on_caller(future, queue, side).await,
        }
    }

    /// Run a caller-side callback (a hook, a local env call) as this mode
    /// does: an `_async` twin runs it with the driving flag set (R-16).
    pub(crate) fn call<R>(self, f: impl FnOnce() -> R) -> R {
        match self {
            Mode::Sync => f(),
            Mode::Async => drive::as_driving(f),
        }
    }
}

/// `Model::run` / `run_async`.
pub(crate) async fn run_model(
    model: &Model,
    env: EnvTarget<'_>,
    options: RunOptions<'_>,
    mode: Mode,
) -> Result<RunResult> {
    let shared = model.shared();
    shared.check_not_reentrant("run()")?;
    shared.admit()?;
    let RunOptions {
        episodes,
        seeds,
        max_episode_steps,
        max_episode,
        mut hooks,
        close_env,
        trust_entrypoints,
        instruction,
        execution_horizon,
        prefetch_lead,
        workflow_edition,
        trial_index_base,
        cancel,
    } = options;
    // RUN-5: what the options alone decide, before anything dials.
    let budget = episode_budget(episodes, seeds.as_deref())?;
    check_horizon(execution_horizon, shared.decl.native_chunk)?;
    let caps = check_caps(max_episode_steps, max_episode)?;
    let prefetch_lead = check_prefetch(prefetch_lead)?;
    let _slot = shared.claim_run()?;
    let driver = CancellationToken::new();
    let watch = CancelWatch::start(cancel.into_iter().collect(), driver.clone())?;
    // COR-6: the call, RLMESH_WORKFLOW_EDITION, then the model's type.
    let edition = crate::editions::resolve_native(
        EditionSources {
            call: workflow_edition.as_deref(),
            option: None,
            declared: shared.decl.workflow_edition.as_deref(),
        },
        shared.decl.authored,
    )?;
    if execution_horizon > 1 && !shared.decl.has_chunk() {
        crate::warnings::warn(
            crate::WarningCategory::UserWarning,
            format!(
                "execution_horizon={execution_horizon} was requested but the model defines no \
                 predict_chunk(); running un-chunked."
            ),
        )?;
    }
    let context = LibraryContext::new();
    let relay = hooks
        .is_some()
        .then(|| Arc::new(HookRelay::new(driver.clone(), context.clone())));
    let ended = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let plan = Plan {
        ended: ended.clone(),
        shared: Arc::clone(shared),
        trust: trust_entrypoints.unwrap_or(shared.decl.trust_entrypoints),
        instruction,
        budget,
        seeds: seeds.unwrap_or_default(),
        caps,
        close_env,
        execution_horizon,
        prefetch_lead,
        edition,
        trial_index_base,
    };
    let outcome = run_body(
        plan,
        env,
        hooks.as_deref_mut(),
        Arc::clone(&context),
        relay.clone(),
        driver,
        mode,
    )
    .await;
    // The first error wins: a hook's (kept by the relay), a token cancel,
    // then the run's own.
    let outcome = match outcome {
        Ok(result) => Ok(result),
        Err(err) => Err(relay
            .as_ref()
            .and_then(|relay| relay.take_error())
            .unwrap_or(if watch.fired() { Error::Cancelled } else { err })),
    };
    match (outcome, hooks) {
        (Ok(result), Some(hooks)) if !ended.load(std::sync::atomic::Ordering::SeqCst) => {
            mode.call(|| hooks.on_run_end(&result))?;
            Ok(result)
        }
        (Ok(result), _) => Ok(result),
        (Err(err), Some(hooks)) if !ended.load(std::sync::atomic::Ordering::SeqCst) => {
            let partial = RunResult {
                advisories: Vec::new(),
                episodes: relay.map(|relay| relay.results()).unwrap_or_default(),
                telemetry: Vec::new(),
            };
            let _ = mode.call(|| hooks.on_run_end(&partial));
            Err(err)
        }
        (Err(err), _) => Err(err),
    }
}

/// A factory-made env remains owned until its explicit close, including on a
/// dropped run future. Cleanup runs on a plain thread and outlives the future.
#[derive(Default)]
pub(crate) struct OwnedMadeEnv(pub(crate) Option<MadeEnv>, pub(crate) bool);

impl Drop for OwnedMadeEnv {
    fn drop(&mut self) {
        if self.1 {
            return;
        }
        if let Some(mut env) = self.0.take() {
            let _ = std::thread::Builder::new()
                .name("rlmesh-env-close".into())
                .spawn(move || {
                    let _ = match &mut env {
                        MadeEnv::Env(env) => env.close(),
                        MadeEnv::Vector(env) => env.close(),
                    };
                });
        }
    }
}

struct Plan {
    ended: Arc<std::sync::atomic::AtomicBool>,
    shared: Arc<crate::model::facade::Shared>,
    trust: bool,
    instruction: Option<String>,
    budget: u64,
    seeds: Vec<i64>,
    caps: super::options::Caps,
    close_env: bool,
    execution_horizon: u32,
    prefetch_lead: u32,
    edition: Option<String>,
    trial_index_base: u64,
}

/// Where the driver dials, and what the run releases afterwards.
struct Dial<'e> {
    address: String,
    token: String,
    loopback: Option<EnvServer>,
    /// A served env to shut down afterwards (`close_env` on an `EnvServer`).
    shutdown: Option<&'e EnvServer>,
}

async fn run_body(
    plan: Plan,
    env: EnvTarget<'_>,
    mut hooks: Option<&mut (dyn RunHooks + '_)>,
    context: Arc<LibraryContext>,
    relay: Option<Arc<HookRelay>>,
    driver: CancellationToken,
    mode: Mode,
) -> Result<RunResult> {
    if let Some(hooks) = hooks.as_deref_mut() {
        let context: Arc<dyn super::RunContext> = context.clone();
        mode.call(|| hooks.on_run_start(context))?;
    }
    if plan.budget == 0 {
        // Nothing runs, so the env is never dialed (or built).
        return Ok(RunResult::default());
    }
    let (port, mut queue) = caller_channel();
    let mut env = env.0;
    let over_wire = matches!(env, EnvKind::Address(_));
    let remote = match &env {
        EnvKind::Address(address) => Some(Dial {
            address: address.clone(),
            token: String::new(),
            loopback: None,
            shutdown: None,
        }),
        EnvKind::Server(server) => {
            let server: &EnvServer = server;
            Some(Dial {
                address: server.address().to_string(),
                token: server.token().unwrap_or_default().to_string(),
                loopback: None,
                shutdown: None,
            })
        }
        // The native loop dials a caller's client's address itself.
        EnvKind::Remote(handle) => Some(Dial {
            address: handle.remote_address().to_string(),
            token: String::new(),
            loopback: None,
            shutdown: None,
        }),
        _ => None,
    };
    let served = match &env {
        EnvKind::Server(server) if plan.close_env => Some(*server),
        _ => None,
    };
    // A factory-built env is this call's own: built only now that an
    // episode needs it, closed when the run ends.
    let mut made = OwnedMadeEnv::default();
    if let EnvKind::Factory(factory) = &mut env {
        made.0 = Some(mode.call(|| factory.make())?);
    }
    let built = made.0.is_some();
    let (mut local, remote_handle) = match (&mut env, &mut made.0) {
        (_, Some(MadeEnv::Env(env))) => (Some(LocalEnv::Env(&mut **env)), None),
        (_, Some(MadeEnv::Vector(env))) => (Some(LocalEnv::Vector(&mut **env)), None),
        (EnvKind::Env(env), None) => (Some(LocalEnv::Env(&mut **env)), None),
        (EnvKind::Vector(env), None) => (Some(LocalEnv::Vector(&mut **env)), None),
        (EnvKind::Remote(handle), None) => (None, Some(&mut **handle)),
        _ => (None, None),
    };
    let dial = match (remote, &local) {
        (Some(dial), _) => Dial {
            shutdown: served,
            ..dial
        },
        (None, Some(local)) => {
            let lent = match local {
                LocalEnv::Env(env) => {
                    LoopbackEnv::Env(Box::new(LentEnv::new(&**env, port.clone())))
                }
                LocalEnv::Vector(env) => {
                    LoopbackEnv::Vector(Box::new(LentVectorEnv::new(&**env, port.clone())))
                }
            };
            let server = EnvServer::loopback(lent, plan.edition.clone())?;
            server.start()?;
            Dial {
                address: server.address().to_string(),
                token: server.token().unwrap_or_default().to_string(),
                loopback: Some(server),
                shutdown: None,
            }
        }
        (None, None) => return Err(Error::Internal("env target without an env".to_string())),
    };
    let is_local = local.is_some();
    let run = LocalRun {
        address: dial.address.clone(),
        token: dial.token.clone(),
        episodes: plan.budget,
        base_seed: None,
        seeds: plan.seeds,
        trial_index_base: Some(plan.trial_index_base),
        max_episode_steps: plan.caps.max_episode_steps,
        max_episode_seconds: plan.caps.max_episode_seconds,
        // An address closes over the wire; a local env closes directly below.
        close_env: plan.close_env && over_wire,
        execution_horizon: plan.execution_horizon,
        prefetch_lead: plan.prefetch_lead,
        workflow_edition: plan.edition,
    };
    let runtime_hooks: Arc<dyn RuntimeHooks> = match &relay {
        Some(relay) => {
            relay.attach(port.clone());
            Arc::clone(relay) as Arc<dyn RuntimeHooks>
        }
        None => Arc::new(rlmesh_runtime::NoopRuntimeHooks),
    };
    let shared = Arc::clone(&plan.shared);
    let trust = plan.trust;
    let connected = Arc::clone(&context);
    let adapter_context = context.clone();
    let core = async move {
        let mut handler = crate::model::bridge::engine_handler_configured(
            shared,
            trust,
            plan.instruction,
            Some(adapter_context),
        );
        drive_local(&mut handler, run, driver, runtime_hooks, &move |contract| {
            if let Ok(contract) = crate::EnvContract::from_wire(contract) {
                connected.connect(contract);
            }
        })
        .await
    };
    let mut side = CallerSide {
        env: local.take(),
        hooks: drive::lend_hooks(hooks),
    };
    let report = mode
        .drive(core, &mut queue, &mut side)
        .await
        .map_err(|err| match err {
            Error::Runtime(message) | Error::Connection(message)
                if message.contains("active Join session") =>
            {
                Error::Runtime(
                    "the target env already has an active session: a RemoteEnv / \
                     RemoteVectorEnv handle that was driven with reset()/step() holds the \
                     env's single session slot. close() the handle (or its open session) \
                     before run(), or drive it interactively via session()."
                        .to_string(),
                )
            }
            other => other,
        });
    // Drop the port's queue first: a lane still waiting on a relayed call
    // fails at once instead of holding the loopback's drain.
    drop(queue);
    drop(port);
    let mut release: Result<()> = Ok(());
    if let Some(server) = &dial.loopback {
        release = match mode {
            Mode::Sync => server.shutdown(),
            Mode::Async => server.shutdown_async().await,
        };
    }
    if let Some(server) = dial.shutdown {
        let shut = match mode {
            Mode::Sync => server.shutdown(),
            Mode::Async => server.shutdown_async().await,
        };
        release = release.and(shut);
    }
    // A caller's local env closes on the opt-in; one built here always.
    if is_local && (plan.close_env || built) {
        let closed = match side.env.as_mut() {
            Some(LocalEnv::Env(env)) => mode.call(|| env.close()),
            Some(LocalEnv::Vector(env)) => mode.call(|| env.close()),
            None => Ok(()),
        };
        release = release.and(closed);
    }
    // Explicit close has run; disarm the cancellation cleanup.
    made.1 = true;
    // A caller's remote client shuts its env down on the opt-in.
    if plan.close_env
        && let Some(handle) = remote_handle
    {
        let shut = match handle.lend() {
            Ok(mut client) => {
                let out = match mode {
                    Mode::Sync => drive::block_on_library(client.shutdown(None)),
                    Mode::Async => client.shutdown(None).await,
                };
                handle.give_back(client);
                out.map(|_| ())
            }
            Err(err) => Err(err),
        };
        release = release.and(shut);
    }
    let outcome = report.and_then(|report| {
        release.map(|()| {
            let mut result = results::from_report(&report);
            result.advisories = context.advisories();
            result
        })
    });
    if side.hooks.is_some() {
        plan.ended.store(true, std::sync::atomic::Ordering::SeqCst);
        let partial = match &outcome {
            Ok(result) => result.clone(),
            Err(_) => RunResult {
                episodes: relay.as_ref().map(|r| r.results()).unwrap_or_default(),
                advisories: context.advisories(),
                ..Default::default()
            },
        };
        let callback = mode.call(|| side.hook(context.id, |hooks| hooks.on_run_end(&partial)));
        match outcome {
            Ok(result) => {
                callback?;
                Ok(result)
            }
            Err(error) => Err(error),
        }
    } else {
        outcome
    }
}

/// Per-episode bookkeeping the relay keeps between events.
struct Episode {
    index: i64,
    seed: Option<i64>,
    steps: i64,
}

/// One lane group's in-flight step: what it observed, then what it did.
struct Group {
    ids: Vec<String>,
    observations: Vec<Value>,
    actions: Option<Vec<Value>>,
}

#[derive(Default)]
struct RelayState {
    episodes: HashMap<String, Episode>,
    groups: HashMap<i32, Group>,
    results: Vec<EpisodeResult>,
}

/// Turns the runtime's events into [`RunHooks`] calls on the caller
/// (Python's `NativeHookRelay`): each is awaited, so hooks run serially and
/// the env is idle while one runs. The first hook error is kept and cancels
/// the run.
pub(crate) struct HookRelay {
    port: Mutex<Option<CallerPort>>,
    driver: CancellationToken,
    error: Mutex<Option<Error>>,
    state: Mutex<RelayState>,
    context: Arc<LibraryContext>,
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|err| err.into_inner())
}

/// One facade value per lane from a group's wire leaves.
fn decode_lanes(
    leaves: Vec<rlmesh_grpc::wire::Bytes>,
    space: &rlmesh_proto::spaces::v1::SpaceSpec,
    lanes: usize,
) -> Result<Vec<Value>> {
    let space =
        space_spec_from_proto(space.clone()).map_err(|err| Error::Protocol(err.to_string()))?;
    let values = decode_batched_partial_values(Some(&leaves_value(leaves)), &space, lanes)
        .map_err(|err| Error::Protocol(err.to_string()))?;
    Ok(values
        .into_iter()
        .map(|value| space_codec::value_from_space(&space, value))
        .collect())
}

impl HookRelay {
    fn new(driver: CancellationToken, context: Arc<LibraryContext>) -> HookRelay {
        HookRelay {
            port: Mutex::new(None),
            driver,
            error: Mutex::new(None),
            state: Mutex::new(RelayState::default()),
            context,
        }
    }

    fn attach(&self, port: CallerPort) {
        *lock(&self.port) = Some(port);
    }

    fn take_error(&self) -> Option<Error> {
        lock(&self.error).take()
    }

    fn results(&self) -> Vec<EpisodeResult> {
        std::mem::take(&mut lock(&self.state).results)
    }

    fn fail(&self, err: Error) {
        lock(&self.error).get_or_insert(err);
        self.driver.cancel();
    }

    /// Run `call` on the caller's hooks and wait for it.
    async fn dispatch(&self, call: impl FnOnce(&mut dyn RunHooks) -> Result<()> + Send + 'static) {
        if self.driver.is_cancelled() {
            return;
        }
        let Some(port) = lock(&self.port).clone() else {
            return;
        };
        let id = self.context.id;
        let result = port.call(move |side| side.hook(id, call).map(|_| ())).await;
        if let Err(err) = result {
            self.fail(err);
        }
    }
}

#[async_trait]
impl RuntimeHooks for HookRelay {
    async fn episode_started(
        &self,
        event: EpisodeStartedEvent,
    ) -> std::result::Result<(), HookError> {
        let index = event.episode_index - 1;
        lock(&self.state).episodes.insert(
            event.episode_id,
            Episode {
                index,
                seed: event.seed,
                steps: 0,
            },
        );
        let seed = event.seed;
        self.dispatch(move |hooks| hooks.on_episode_start(index, seed))
            .await;
        Ok(())
    }

    async fn observation_emitted(
        &self,
        event: ObservationEmittedEvent,
    ) -> std::result::Result<(), HookError> {
        let Some(leaves) = event.observation else {
            return Ok(());
        };
        match decode_lanes(leaves, &event.observation_space, event.episode_ids.len()) {
            Ok(observations) => {
                lock(&self.state).groups.insert(
                    event.env_index,
                    Group {
                        ids: event.episode_ids,
                        observations,
                        actions: None,
                    },
                );
            }
            Err(err) => self.fail(err),
        }
        Ok(())
    }

    async fn action_received(
        &self,
        event: ActionReceivedEvent,
    ) -> std::result::Result<(), HookError> {
        let Some(leaves) = event.action else {
            return Ok(());
        };
        match decode_lanes(leaves, &event.action_space, event.episode_ids.len()) {
            Ok(actions) => {
                if let Some(group) = lock(&self.state).groups.get_mut(&event.env_index) {
                    group.actions = Some(actions);
                }
            }
            Err(err) => self.fail(err),
        }
        Ok(())
    }

    async fn step_completed(
        &self,
        event: StepCompletedEvent,
    ) -> std::result::Result<(), HookError> {
        let info = event
            .infos
            .map(|infos| meta_codec::metadata_from_wire(meta_map_from_proto(infos)))
            .unwrap_or_default();
        let mut events = Vec::new();
        {
            let mut state = lock(&self.state);
            let RelayState {
                episodes, groups, ..
            } = &mut *state;
            let Some(group) = groups.get(&event.env_index) else {
                return Ok(());
            };
            for (lane, episode_id) in group.ids.iter().enumerate() {
                let Some(episode) = episodes.get_mut(episode_id) else {
                    continue;
                };
                if event.autoreset_roll.get(lane).copied().unwrap_or(false) {
                    continue;
                }
                // RUN-21: the runtime always sends an action.
                let Some(action) = group
                    .actions
                    .as_ref()
                    .and_then(|actions| actions.get(lane))
                    .cloned()
                else {
                    drop(state);
                    self.fail(Error::Internal(
                        "the runtime reported a step without its action".to_string(),
                    ));
                    return Ok(());
                };
                let observation = group.observations.get(lane).cloned().unwrap_or(Value::None);
                events.push(StepEvent {
                    read: super::read::read_fn(self.context.clone(), observation.clone()),
                    episode: episode.index,
                    seed: episode.seed,
                    step: episode.steps,
                    observation,
                    action,
                    reward: event.rewards.get(lane).copied().unwrap_or_default(),
                    terminated: event.terminated.get(lane).copied().unwrap_or_default(),
                    truncated: event.truncated.get(lane).copied().unwrap_or_default(),
                    info: info.clone(),
                    predict_ms: event.predict_ms,
                    step_ms: event.step_ms,
                });
                episode.steps += 1;
            }
        }
        for step in events {
            self.dispatch(move |hooks| hooks.on_step(&step)).await;
        }
        Ok(())
    }

    async fn episode_completed(
        &self,
        event: EpisodeCompletedEvent,
    ) -> std::result::Result<(), HookError> {
        let result = EpisodeResult {
            index: event.episode_index - 1,
            seed: event.seed,
            steps: event.step_count,
            reward: event.cumulative_reward,
            terminated: event.terminated,
            truncated: event.truncated,
            success: event.success,
            duration_s: event.duration_ms as f64 / 1000.0,
            predict_ms: event.predict_ms,
            step_ms: event.step_ms,
            trial: event.trial_index.map(|trial| trial as i64),
        };
        {
            let mut state = lock(&self.state);
            state.episodes.remove(&event.episode_id);
            state.results.push(result.clone());
        }
        self.dispatch(move |hooks| hooks.on_episode_end(&result))
            .await;
        Ok(())
    }
}
