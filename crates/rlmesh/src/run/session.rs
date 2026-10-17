//! `Session<'a>`: a model bound to one env, driven by hand or by
//! `Session::run`.

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rlmesh_grpc::wire::encode_batched_partial_values;
use tokio_util::sync::CancellationToken;

use super::drive::{
    CallerPort, CallerQueue, CallerSide, CancelWatch, LocalEnv, caller_channel,
    refuse_in_async_context, run_sync,
};
use super::hooks::LibraryContext;
use super::native::Mode;
use super::options::{check_caps, check_horizon, episode_budget};
use super::results::{EpisodeResult, RunResult, StepEvent, TelemetryRow};
use super::targets::{EnvKind, EnvTarget, ErasedFactory};
use super::{RunContext, RunHooks, SessionOptions, SessionRunOptions};
use crate::RemoteModel;
use crate::binding::env::{CoreConnect, EnvClientCore};
use crate::binding::model::{EpisodeInfo, HistoryFrame, ModelObservation, ModelRouteContext};
use crate::binding::model::{ModelClient, ModelConnect};
use crate::env::client::Lendable;
use crate::model::engine::AdaptedModelHandler;
use crate::model::handler::{ModelHandler, PredictFrames, ResolveOptions};
use crate::spaces::Space;
use crate::value::space_codec;
use crate::{
    EnvContract, EnvServer, Error, InfoDict, MadeEnv, Metadata, Model, Result, Value,
    WarningCategory,
};

/// The native session loop's per-episode step bound for this edition.
const MAX_STEPS_PER_EPISODE: i64 = 100_000;

/// How long Drop waits for owned cleanup outside an async runtime.
const DROP_BOUND: Duration = Duration::from_secs(5);

const SUCCESS_INFO_KEYS: [&str; 3] = ["is_success", "success", "task_success"];

fn closed_error() -> Error {
    Error::Runtime("session is closed".to_string())
}

/// One info value as a task outcome, or `None` when its kind carries none.
fn success_flag(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(flag) => Some(*flag),
        Value::Int(number) => Some(*number != 0),
        Value::Float(number) => Some(*number != 0.0),
        _ => None,
    }
}

/// The env-reported task outcome of a step's info.
fn episode_success(info: &InfoDict) -> Option<bool> {
    SUCCESS_INFO_KEYS
        .iter()
        .find_map(|key| info.get(key).and_then(success_flag))
}

/// The per-op samples one `Session::run` measures.
#[derive(Default)]
struct Telemetry {
    samples: HashMap<(&'static str, &'static str), Vec<f64>>,
}

impl Telemetry {
    const ORDER: [&'static str; 4] = ["model.predict", "env.reset", "env.step", "runner.round"];

    fn record(&mut self, op: &'static str, component: &'static str, ms: f64) {
        self.samples.entry((op, component)).or_default().push(ms);
    }

    fn rows(&self) -> Vec<TelemetryRow> {
        let rank = |op: &str| {
            Self::ORDER
                .iter()
                .position(|known| *known == op)
                .unwrap_or(Self::ORDER.len())
        };
        let mut keys: Vec<_> = self.samples.keys().copied().collect();
        keys.sort_by(|a, b| (rank(a.0), a.0).cmp(&(rank(b.0), b.0)));
        keys.into_iter()
            .map(|(op, component)| {
                let values = &self.samples[&(op, component)];
                let mut ordered = values.clone();
                ordered.sort_by(f64::total_cmp);
                let nearest = |percentile: f64| {
                    ordered[((ordered.len() - 1) as f64 * percentile).ceil() as usize]
                };
                TelemetryRow {
                    op: op.to_string(),
                    component: component.to_string(),
                    metric: "rpc.total".to_string(),
                    unit: "ms".to_string(),
                    count: values.len() as i64,
                    avg: values.iter().sum::<f64>() / values.len() as f64,
                    p50: nearest(0.50),
                    p95: nearest(0.95),
                    p99: nearest(0.99),
                }
            })
            .collect()
    }
}

/// A local model's per-session engine state.
struct LocalModel {
    shared: Arc<crate::model::facade::Shared>,
    handler: AdaptedModelHandler,
    env_id: String,
    session_id: String,
    /// The horizon the session asked for; engaged only with a chunk corner.
    requested_horizon: u32,
    horizon: u32,
    wants_history: bool,
    replay: VecDeque<Value>,
    history: Vec<HistoryFrame>,
}

enum ModelPort {
    Local(Box<LocalModel>),
    /// A served model's per-env client, dialed at session creation.
    Remote(Box<ModelClient>),
    Random(Option<Space>),
}

enum EnvPort {
    /// A local env (borrowed, or built from a factory) on the caller's side.
    Local,
    /// A connection the session dials itself (an address or an `EnvServer`),
    /// or a caller's remote client lent for the borrow (`borrowed`).
    Remote {
        address: String,
        token: String,
        client: Option<Box<EnvClientCore>>,
        borrowed: bool,
    },
}

/// Returns a borrowed connection even when creation is abandoned during a dial.
struct EnvLease<'a> {
    env: Option<EnvPort>,
    lent: Option<&'a mut (dyn Lendable + Send + 'a)>,
}

impl Drop for EnvLease<'_> {
    fn drop(&mut self) {
        if let Some(EnvPort::Remote {
            client: Some(mut core),
            borrowed,
            ..
        }) = self.env.take()
        {
            if borrowed {
                if let Some(handle) = self.lent.as_mut() {
                    handle.give_back(core);
                }
            } else {
                core.prepare_close();
                super::drive::runtime().spawn(async move {
                    let _ = core.close().await;
                });
            }
        }
    }
}

struct Connected {
    contract: EnvContract,
    wire: Arc<rlmesh_spaces::EnvContract>,
    observation: rlmesh_spaces::SpaceSpec,
    action: rlmesh_spaces::SpaceSpec,
    /// A local vector env with one lane, driven through its batch shapes.
    vector: bool,
    trial_option: bool,
}

#[derive(Default)]
struct Episode {
    open: bool,
    id: String,
    seed: Option<i64>,
    steps: i64,
    reward: f64,
    terminated: bool,
    truncated: bool,
}

/// The session's owned state: everything its calls drive, none of the
/// caller's borrows (those are reached through `port`).
struct Core {
    model: ModelPort,
    env: EnvPort,
    port: CallerPort,
    context: Arc<LibraryContext>,
    edition: Option<String>,
    connected: Option<Connected>,
    episode: Episode,
    telemetry: Option<Telemetry>,
    /// The call in flight's cancellation.
    cancel: CancellationToken,
    subscriptions: Arc<Vec<crate::ext::CancelSubscription>>,
    viewer: Option<Arc<std::sync::Mutex<crate::view::ViewerDriver>>>,
    episode_started: Instant,
    run_episode: i64,
    run_episodes: i64,
    last_predict_ms: f64,
    last_step_ms: f64,
    last_pace: Duration,
    last_view_failed: bool,
}

/// Fail `future` with the cancellation class once `cancel` fires.
async fn cancellable<T>(
    cancel: &CancellationToken,
    future: impl Future<Output = Result<T>>,
) -> Result<T> {
    tokio::select! {
        biased;
        () = cancel.cancelled() => Err(Error::Cancelled),
        out = future => out,
    }
}

fn ms_since(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn declares_trial_option(contract: &EnvContract) -> bool {
    match contract
        .metadata()
        .and_then(|metadata| metadata.get(crate::ENV_RESET_OPTIONS_KEY))
    {
        Some(Value::Text(key)) => key == "trial_index",
        Some(Value::List(keys) | Value::Tuple(keys)) => keys
            .iter()
            .any(|key| matches!(key, Value::Text(key) if key == "trial_index")),
        _ => false,
    }
}

fn reject_vector_env(num_envs: u64) -> Result<()> {
    if num_envs > 1 {
        return Err(Error::invalid_argument(format!(
            "the session loop drives a single env, but the env reports num_envs={num_envs}; \
             drive a lane endpoint with Model.run(address) (the native loop), or serve \
             num_envs=1 for session()."
        )));
    }
    Ok(())
}

impl Core {
    async fn feed_view(&mut self, observation: Value) -> Result<()> {
        let Some(viewer) = self.viewer.clone() else {
            return Ok(());
        };
        let context = self.context.clone();
        let elapsed = self.episode_started.elapsed().as_secs_f64();
        let hud = rlmesh_viewer::Hud {
            step: self.episode.steps,
            reward: self.episode.reward,
            outcome: if self.episode.terminated {
                "terminated"
            } else if self.episode.truncated {
                "truncated"
            } else {
                "running"
            }
            .into(),
            episode: self.run_episode,
            episodes: self.run_episodes,
            seed: self.episode.seed.unwrap_or(-1),
            elapsed_s: elapsed,
            sps: self.episode.steps as f64 / elapsed.max(f64::EPSILON),
            model_ms: self.last_predict_ms,
            env_ms: self.last_step_ms,
            ..Default::default()
        };
        let skip = self
            .port
            .call(move |side| {
                super::read::dispatch(context.id, side.env.as_mut(), || {
                    let mut driver = viewer.lock().unwrap_or_else(|e| e.into_inner());
                    match driver.feed(&*context, observation, hud) {
                        Ok(skip) => Ok(skip),
                        Err(err) => {
                            driver.disable();
                            crate::warnings::warn(
                                WarningCategory::UserWarning,
                                format!("rlmesh view: disabled after feed error: {err}"),
                            )?;
                            Ok(false)
                        }
                    }
                })
            })
            .await?;
        if skip {
            self.episode.truncated = true;
        }
        Ok(())
    }

    async fn close_view(&mut self) -> Result<()> {
        let Some(viewer) = self.viewer.clone() else {
            return Ok(());
        };
        let hold = viewer.lock().unwrap_or_else(|e| e.into_inner()).hold();
        if !self.last_view_failed && hold != crate::ViewHold::Off {
            let quittable = viewer.lock().unwrap_or_else(|e| e.into_inner()).quittable();
            if hold == crate::ViewHold::UntilQuit && !quittable {
                self.port.call(|_| crate::warnings::warn(WarningCategory::UserWarning,
                    "rlmesh view: hold=True needs an interactive terminal to quit from (pass a number of seconds to hold without one); closing the viewer instead.")).await?;
            } else {
                let started = Instant::now();
                loop {
                    if viewer.lock().unwrap_or_else(|e| e.into_inner()).quit {
                        break;
                    }
                    if let crate::ViewHold::For(duration) = hold
                        && started.elapsed() >= duration
                    {
                        break;
                    }
                    let viewer = viewer.clone();
                    let context = self.context.clone();
                    self.port
                        .call(move |side| {
                            super::read::dispatch(context.id, side.env.as_mut(), || {
                                viewer
                                    .lock()
                                    .unwrap_or_else(|e| e.into_inner())
                                    .held(&*context)
                            })
                        })
                        .await?;
                    cancellable(&self.cancel, async {
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        Ok(())
                    })
                    .await?;
                }
            }
        }
        viewer.lock().unwrap_or_else(|e| e.into_inner()).disable();
        Ok(())
    }

    async fn local_call<R: Send + 'static>(
        &self,
        call: impl for<'x, 'c> FnOnce(&'x mut CallerSide<'c>) -> Result<R> + Send + 'static,
    ) -> Result<R> {
        let subscriptions = Arc::clone(&self.subscriptions);
        cancellable(
            &self.cancel,
            self.port.call(move |side| {
                let cancelled = || {
                    subscriptions
                        .iter()
                        .any(crate::ext::CancelSubscription::fired)
                };
                if cancelled() {
                    return Err(Error::Cancelled);
                }
                let result = call(side);
                if cancelled() {
                    return Err(Error::Cancelled);
                }
                result
            }),
        )
        .await
    }

    fn check_model_open(&self) -> Result<()> {
        match &self.model {
            ModelPort::Local(model) => model.shared.admit(),
            ModelPort::Remote(_) | ModelPort::Random(_) => Ok(()),
        }
    }

    async fn ensure_connected(&mut self) -> Result<()> {
        if self.connected.is_some() {
            return Ok(());
        }
        let (contract, vector, action_space) = match &mut self.env {
            EnvPort::Local => {
                self.port
                    .call(|side| {
                        let (contract, vector, action) = match side.env.as_mut() {
                            Some(LocalEnv::Env(env)) => (
                                crate::env::sync_wire::env_contract_of(&**env)?,
                                false,
                                env.action_space(),
                            ),
                            Some(LocalEnv::Vector(env)) => (
                                crate::env::sync_wire::vector_contract_of(&**env)?,
                                true,
                                env.single_action_space(),
                            ),
                            None => {
                                return Err(Error::Internal("no local env to connect".into()));
                            }
                        };
                        Ok((contract, vector, Some(action)))
                    })
                    .await?
            }
            EnvPort::Remote {
                address,
                token,
                client,
                ..
            } => {
                if let Some(remote) = client.as_deref() {
                    let contract = EnvContract::from_wire(remote.env_contract())?;
                    (contract, false, None)
                } else {
                    let remote = cancellable(
                        &self.cancel,
                        EnvClientCore::connect_with(
                            address,
                            CoreConnect {
                                token,
                                declared: self.edition.as_deref(),
                                ..CoreConnect::default()
                            },
                        ),
                    )
                    .await?;
                    let contract = EnvContract::from_wire(remote.env_contract())?;
                    *client = Some(Box::new(remote));
                    (contract, false, None)
                }
            }
        };
        reject_vector_env(contract.num_envs())?;
        let wire = Arc::new(contract.to_wire()?);
        let observation = contract.observation_space().wire().clone();
        let action = contract.action_space().wire().clone();
        match &mut self.model {
            // The env leg runs at the three-way floor the model leg settled
            // on: pin it before the first reset (RMT-20).
            ModelPort::Remote(model) => {
                if let EnvPort::Remote {
                    client: Some(core), ..
                } = &mut self.env
                {
                    core.pin_workflow_edition(model.selected_workflow_edition());
                }
            }
            ModelPort::Random(space) => {
                *space = Some(
                    action_space.unwrap_or_else(|| Space::new(contract.action_space().clone())),
                );
            }
            ModelPort::Local(model) => {
                model.horizon = if model.shared.decl.has_chunk() {
                    model.requested_horizon
                } else {
                    1
                };
                if let Some(route_setup) = model.handler.route_setup() {
                    let needs = route_setup
                        .resolve_adapter(
                            &model.env_id,
                            &wire,
                            ResolveOptions {
                                execution_horizon: model.horizon,
                                delivers_history: true,
                                fill_missing: false,
                            },
                        )
                        .await?;
                    model.wants_history = needs.history.is_some();
                }
            }
        }
        if let EnvPort::Remote {
            client: Some(client),
            ..
        } = &self.env
        {
            self.context.remote_render(client.render_callback());
        }
        self.context.connect(contract.clone());
        self.connected = Some(Connected {
            trial_option: declares_trial_option(&contract),
            contract,
            wire,
            observation,
            action,
            vector,
        });
        Ok(())
    }

    fn connected(&self) -> Result<&Connected> {
        self.connected
            .as_ref()
            .ok_or_else(|| Error::Internal("the session is not connected".to_string()))
    }

    /// Fire the model's episode end once for the open episode.
    async fn end_episode(&mut self) -> Result<()> {
        if !self.episode.open {
            return Ok(());
        }
        self.episode.open = false;
        if let ModelPort::Local(model) = &mut self.model {
            let ids = vec![self.episode.id.clone()];
            if let Some(route_setup) = model.handler.route_setup() {
                route_setup.reset_adapter(&model.env_id, &ids).await?;
            }
            model.handler.reset_adapter(&model.env_id, ids).await?;
        }
        Ok(())
    }

    async fn env_reset(
        &mut self,
        seed: Option<i64>,
        options: Option<InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        let connected = self.connected()?;
        let vector = connected.vector;
        let observation = connected.observation.clone();
        match &mut self.env {
            EnvPort::Local => {
                self.local_call(move |side| {
                    if vector {
                        let (batch, info) = side
                            .vector()?
                            .reset(seed.map(crate::VectorSeed::One), options.as_ref())?;
                        let lane = crate::env::batch::unstack_lanes(&observation, batch, 1)?
                            .into_iter()
                            .next()
                            .unwrap_or(Value::None);
                        Ok((lane, info))
                    } else {
                        side.env()?.reset(seed, options.as_ref())
                    }
                })
                .await
            }
            EnvPort::Remote { client, .. } => {
                let client = client.as_mut().ok_or_else(closed_error)?;
                let (lanes, info) = cancellable(
                    &self.cancel,
                    client.reset_values(seed.into_iter().collect(), options.as_ref(), Vec::new()),
                )
                .await?;
                let value = lanes
                    .into_iter()
                    .next()
                    .ok_or_else(|| Error::Protocol("reset returned no observation".to_string()))?;
                Ok((value, info))
            }
        }
    }

    async fn env_step(&mut self, action: Value) -> Result<(Value, f64, bool, bool, InfoDict)> {
        let connected = self.connected()?;
        let vector = connected.vector;
        let observation = connected.observation.clone();
        let action_space = connected.action.clone();
        match &mut self.env {
            EnvPort::Local => {
                self.local_call(move |side| {
                    if vector {
                        let actions = crate::env::batch::stack_lanes(&action_space, vec![action])?;
                        let (batch, rewards, terminated, truncated, info) =
                            side.vector()?.step(actions)?;
                        let lane = crate::env::batch::unstack_lanes(&observation, batch, 1)?
                            .into_iter()
                            .next()
                            .unwrap_or(Value::None);
                        Ok((
                            lane,
                            rewards.first().copied().unwrap_or_default(),
                            terminated.first().copied().unwrap_or_default(),
                            truncated.first().copied().unwrap_or_default(),
                            info,
                        ))
                    } else {
                        side.env()?.step(Some(action))
                    }
                })
                .await
            }
            EnvPort::Remote { client, .. } => {
                let client = client.as_mut().ok_or_else(closed_error)?;
                let (lanes, rewards, terminated, truncated, info) =
                    cancellable(&self.cancel, client.step_values(vec![action])).await?;
                let value = lanes
                    .into_iter()
                    .next()
                    .ok_or_else(|| Error::Protocol("step returned no observation".to_string()))?;
                Ok((
                    value,
                    rewards.first().copied().unwrap_or_default(),
                    terminated.first().copied().unwrap_or_default(),
                    truncated.first().copied().unwrap_or_default(),
                    info,
                ))
            }
        }
    }

    async fn reset(
        &mut self,
        seed: Option<i64>,
        trial_index: Option<i64>,
    ) -> Result<(Value, InfoDict)> {
        self.ensure_connected().await?;
        self.end_episode().await?;
        let mut options = None;
        if let Some(trial) = trial_index {
            if self.connected()?.trial_option {
                let mut map = InfoDict::new();
                map.insert("trial_index", Value::Int(trial));
                options = Some(map);
            } else {
                crate::warnings::warn(
                    WarningCategory::UserWarning,
                    "trial_index was given but this env declares no \"trial_index\" reset option \
                     (EnvFactory.reset_options); resetting without it.",
                )?;
            }
        }
        let started = Instant::now();
        let (observation, info) = self.env_reset(seed, options).await?;
        if let Some(telemetry) = &mut self.telemetry {
            telemetry.record("env.reset", "env", ms_since(started));
        }
        match &mut self.model {
            ModelPort::Random(Some(space)) => {
                if let Some(seed) = seed {
                    space.seed(Some(seed as u64));
                }
            }
            ModelPort::Random(None) => {}
            ModelPort::Remote(model) => model.reset(seed),
            ModelPort::Local(model) => {
                model.replay.clear();
                model.history.clear();
            }
        }
        self.episode = Episode {
            open: true,
            id: crate::mint_id(),
            seed,
            ..Episode::default()
        };
        self.episode_started = Instant::now();
        self.feed_view(observation.clone()).await?;
        Ok((observation, info))
    }

    async fn predict(&mut self, observation: Value) -> Result<Value> {
        let started = Instant::now();
        let result = self.predict_inner(observation).await;
        self.last_predict_ms = ms_since(started);
        result
    }

    async fn predict_inner(&mut self, observation: Value) -> Result<Value> {
        self.check_model_open()?;
        self.ensure_connected().await?;
        if self.episode.id.is_empty() {
            self.episode.id = crate::mint_id();
        }
        let connected = self.connected.as_ref().ok_or_else(closed_error)?;
        let started = Instant::now();
        let model = match &mut self.model {
            ModelPort::Random(space) => {
                let space = space.as_ref().ok_or_else(closed_error)?;
                let action = space.sample()?;
                if let Some(telemetry) = &mut self.telemetry {
                    telemetry.record("model.predict", "model", ms_since(started));
                }
                return Ok(action);
            }
            ModelPort::Remote(model) => {
                let lane = space_codec::value_to_space(&connected.observation, &observation)?;
                let action = cancellable(&self.cancel, model.predict(lane)).await?;
                if let Some(telemetry) = &mut self.telemetry {
                    telemetry.record("model.predict", "model", ms_since(started));
                }
                return space_codec::value_from_space(&connected.action, action);
            }
            ModelPort::Local(model) => model,
        };
        let row = EpisodeInfo {
            episode_id: self.episode.id.clone(),
            seed: self.episode.seed,
            env_id: model.env_id.clone(),
        };
        let encode = |value: &Value| -> Result<Vec<rlmesh_grpc::wire::Bytes>> {
            let lane = space_codec::value_to_space(&connected.observation, value)?;
            Ok(
                encode_batched_partial_values(&[lane], &connected.observation)
                    .map_err(|err| Error::invalid_argument(err.to_string()))?
                    .leaves,
            )
        };
        if let Some(action) = model.replay.pop_front() {
            // A replayed step still consumed an env step: deliver it as
            // history so a stacked adapter's window advances.
            if model.wants_history {
                model.history.push(HistoryFrame {
                    observation: Some(encode(&observation)?),
                    episodes: vec![row],
                    step: self.episode.steps,
                });
            }
            return Ok(action);
        }
        let request = ModelObservation {
            observation: Some(encode(&observation)?),
            route: ModelRouteContext {
                session_id: model.session_id.clone(),
                env_id: model.env_id.clone(),
                request_id: crate::mint_id(),
                episodes: vec![row],
            },
            num_envs: 1,
            env_contract: Some(Arc::clone(&connected.wire)),
            history: std::mem::take(&mut model.history),
            step: model.wants_history.then_some(self.episode.steps),
        };
        let frames = cancellable(&self.cancel, async {
            model
                .handler
                .predict_grouped(vec![request])
                .await
                .pop()
                .unwrap_or_else(|| Err(Error::model("predict_grouped returned no result")))
        })
        .await?;
        if let Some(telemetry) = &mut self.telemetry {
            telemetry.record("model.predict", "model", ms_since(started));
        }
        let PredictFrames { actions, replay } = frames;
        let action = actions
            .into_iter()
            .next()
            .ok_or_else(|| Error::model("predict returned no action"))?;
        let horizon = model.horizon.max(1) as usize;
        for frame in replay.into_iter().take(horizon - 1) {
            if let Some(lane) = frame.into_iter().next() {
                model
                    .replay
                    .push_back(space_codec::value_from_space(&connected.action, lane)?);
            }
        }
        space_codec::value_from_space(&connected.action, action)
    }

    async fn step(&mut self, action: Value) -> Result<(Value, f64, bool, bool, InfoDict)> {
        self.check_model_open()?;
        self.ensure_connected().await?;
        let started = Instant::now();
        let (observation, reward, terminated, truncated, info) = self.env_step(action).await?;
        if let Some(telemetry) = &mut self.telemetry {
            telemetry.record("env.step", "env", ms_since(started));
        }
        self.last_step_ms = ms_since(started);
        self.episode.reward += reward;
        self.episode.steps += 1;
        self.episode.terminated = terminated;
        self.episode.truncated = truncated;
        self.feed_view(observation.clone()).await?;
        self.last_pace = Duration::ZERO;
        if let Some(viewer) = &self.viewer {
            let delay = viewer.lock().unwrap_or_else(|e| e.into_inner()).delay();
            if !delay.is_zero() {
                let started = Instant::now();
                cancellable(&self.cancel, async {
                    tokio::time::sleep(delay).await;
                    Ok(())
                })
                .await?;
                self.last_pace = started.elapsed();
            }
        }
        Ok((
            observation,
            reward,
            terminated,
            self.episode.truncated,
            info,
        ))
    }

    fn done(&self) -> bool {
        self.episode.terminated || self.episode.truncated
    }

    /// Run `call` on the caller's hooks.
    async fn hook(
        &self,
        call: impl FnOnce(&mut dyn RunHooks) -> Result<()> + Send + 'static,
    ) -> Result<()> {
        let id = self.context.id;
        self.port
            .call(move |side| side.hook(id, call).map(|_| ()))
            .await
    }

    async fn run(&mut self, plan: RunPlan) -> Result<RunResult> {
        self.check_model_open()?;
        self.ensure_connected().await?;
        self.run_episodes = plan.episodes.min(i64::MAX as u64) as i64;
        self.telemetry = Some(Telemetry::default());
        let mut results = Vec::new();
        let outcome = self.run_episodes(&plan, &mut results).await;
        let telemetry = self.telemetry.take().map(|t| t.rows()).unwrap_or_default();
        let result = RunResult {
            advisories: self.context.advisories(),
            episodes: results,
            telemetry,
        };
        // RUN-P11: the run's own error wins over `on_run_end`'s, which wins
        // over the last episode end's.
        let run_end = if plan.hooks {
            let partial = result.clone();
            self.hook(move |hooks| hooks.on_run_end(&partial)).await
        } else {
            Ok(())
        };
        let ended = self.end_episode().await;
        self.last_view_failed = outcome.is_err() || run_end.is_err() || ended.is_err();
        outcome?;
        run_end?;
        ended?;
        Ok(result)
    }

    async fn run_episodes(
        &mut self,
        plan: &RunPlan,
        results: &mut Vec<EpisodeResult>,
    ) -> Result<()> {
        if plan.hooks {
            let context: Arc<dyn RunContext> = self.context.clone();
            self.hook(move |hooks| hooks.on_run_start(context)).await?;
        }
        let walks_trials = self.connected()?.trial_option;
        let step_cap = plan.max_episode_steps.unwrap_or(MAX_STEPS_PER_EPISODE);
        for i in 0..plan.episodes {
            self.run_episode = i as i64 + 1;
            let index = i as i64;
            let seed = plan
                .seeds
                .as_ref()
                .and_then(|seeds| seeds.get(i as usize))
                .copied();
            let trial = plan.trial_index_base as i64 + index;
            let (mut observation, mut info) =
                self.reset(seed, walks_trials.then_some(trial)).await?;
            let episode_start = Instant::now();
            let mut round_start = episode_start;
            if plan.hooks {
                self.hook(move |hooks| hooks.on_episode_start(index, seed))
                    .await?;
            }
            let (mut predict_total, mut step_total) = (0.0, 0.0);
            while !self.done() && self.episode.steps < step_cap {
                if self.cancel.is_cancelled() {
                    return Err(Error::Cancelled);
                }
                if plan
                    .max_episode
                    .is_some_and(|cap| episode_start.elapsed() >= cap)
                {
                    break;
                }
                let step = self.episode.steps;
                let previous = observation.clone();
                let t0 = Instant::now();
                let action = self.predict(previous.clone()).await?;
                let t1 = Instant::now();
                let (next, reward, terminated, mut truncated, step_info) =
                    self.step(action.clone()).await?;
                let t2 = Instant::now();
                let predict_ms = (t1 - t0).as_secs_f64() * 1000.0;
                let step_ms = self.last_step_ms;
                self.last_predict_ms = predict_ms;
                if let Some(telemetry) = &mut self.telemetry {
                    telemetry.record(
                        "runner.round",
                        "runner",
                        (t2 - round_start)
                            .saturating_sub(self.last_pace)
                            .as_secs_f64()
                            * 1000.0,
                    );
                }
                round_start = t2;
                predict_total += predict_ms;
                step_total += step_ms;
                // The step cap truncates at the capped step itself.
                if !self.done() && self.episode.steps >= step_cap {
                    truncated = true;
                    self.episode.truncated = true;
                }
                observation = next;
                info = step_info;
                if plan.hooks {
                    let event = StepEvent {
                        read: super::read::read_fn(self.context.clone(), previous.clone()),
                        episode: index,
                        seed,
                        step,
                        observation: previous,
                        action,
                        reward,
                        terminated,
                        truncated,
                        info: info.clone(),
                        predict_ms,
                        step_ms,
                    };
                    self.hook(move |hooks| hooks.on_step(&event)).await?;
                }
            }
            if !self.done() {
                self.episode.truncated = true;
            }
            let steps = self.episode.steps;
            let episode = EpisodeResult {
                index,
                seed,
                steps,
                reward: self.episode.reward,
                terminated: self.episode.terminated,
                truncated: self.episode.truncated,
                success: episode_success(&info),
                duration_s: episode_start.elapsed().as_secs_f64(),
                predict_ms: (steps > 0).then(|| predict_total / steps as f64),
                step_ms: (steps > 0).then(|| step_total / steps as f64),
                trial: Some(trial),
            };
            results.push(episode.clone());
            if plan.hooks {
                self.hook(move |hooks| hooks.on_episode_end(&episode))
                    .await?;
            }
            if self
                .viewer
                .as_ref()
                .is_some_and(|v| v.lock().unwrap_or_else(|e| e.into_inner()).quit)
            {
                break;
            }
        }
        Ok(())
    }

    async fn selected_workflow_edition(&mut self) -> Result<String> {
        // A served model's session runs at the floor settled at creation.
        if let ModelPort::Remote(model) = &self.model {
            return Ok(model.selected_workflow_edition().to_string());
        }
        self.ensure_connected().await?;
        match &self.env {
            EnvPort::Remote {
                client: Some(client),
                ..
            } => Ok(client.selected_workflow_edition().to_string()),
            _ => Err(Error::Runtime(
                "a local env negotiates no workflow edition: it runs in this process at \
                 rlmesh.current_workflow_edition()"
                    .to_string(),
            )),
        }
    }

    /// Release what the session made: the open episode's end, the route,
    /// and a dialed connection (shut down first on `close_env`). The first
    /// error wins; every step runs.
    async fn release(&mut self, close_env: bool) -> Result<()> {
        let mut first: Result<()> = self.end_episode().await;
        // A served model's route closes with the session (its last episode
        // end flushes first).
        if let ModelPort::Remote(model) = &mut self.model {
            first = first.and(model.close().await);
        }
        if let ModelPort::Local(model) = &mut self.model
            && self.connected.is_some()
            && let Some(route_setup) = model.handler.route_setup()
        {
            first = first.and(route_setup.release_adapter(&model.env_id).await);
        }
        match &mut self.env {
            // A caller's client: shut down on the opt-in, never closed here.
            EnvPort::Remote {
                client: Some(client),
                borrowed: true,
                ..
            } => {
                if close_env && !client.is_closed() {
                    first = first.and(client.shutdown(None).await.map(|_| ()));
                }
            }
            EnvPort::Remote {
                client,
                borrowed: false,
                ..
            } => {
                if let Some(mut client) = client.take() {
                    if close_env {
                        first = first.and(client.shutdown(None).await.map(|_| ()));
                    }
                    first = first.and(client.close().await);
                }
            }
            _ => {}
        }
        self.connected = None;
        first
    }
}

/// `Session::run`'s resolved options.
struct RunPlan {
    episodes: u64,
    seeds: Option<Vec<i64>>,
    max_episode_steps: Option<i64>,
    max_episode: Option<Duration>,
    trial_index_base: u64,
    hooks: bool,
}

/// What the caller's side holds for the session's env.
enum CallerEnv<'a> {
    None,
    Borrowed(LocalEnv<'a>),
    /// Built from the factory at connect.
    Factory(Box<dyn ErasedFactory + Send + 'a>, Option<MadeEnv>),
}

impl CallerEnv<'_> {
    fn local(&mut self) -> Option<LocalEnv<'_>> {
        match self {
            CallerEnv::None | CallerEnv::Factory(_, None) => None,
            CallerEnv::Borrowed(LocalEnv::Env(env)) => Some(LocalEnv::Env(&mut **env)),
            CallerEnv::Borrowed(LocalEnv::Vector(env)) => Some(LocalEnv::Vector(&mut **env)),
            CallerEnv::Factory(_, Some(MadeEnv::Env(env))) => Some(LocalEnv::Env(&mut **env)),
            CallerEnv::Factory(_, Some(MadeEnv::Vector(env))) => Some(LocalEnv::Vector(&mut **env)),
        }
    }
}

/// A model bound to one env. Drive it with [`reset`](Session::reset) /
/// [`predict`](Session::predict) / [`step`](Session::step), or pump whole
/// episodes with [`run`](Session::run). It borrows the model and the env for
/// `'a`; [`close`](Session::close) releases what it made, and Drop acts as
/// `close` on an error (no viewer hold, errors swallowed).
pub struct Session<'a> {
    core: Core,
    env: CallerEnv<'a>,
    queue: CallerQueue,
    server: Option<&'a EnvServer>,
    /// A caller's remote client whose core the session holds for the borrow.
    lent: Option<&'a mut (dyn Lendable + Send + 'a)>,
    close_env: bool,
    closed: bool,
    cancel: crate::ext::CancelToken,
    _model: PhantomData<&'a Model>,
}

impl std::fmt::Debug for Session<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("closed", &self.closed)
            .field("done", &self.core.done())
            .finish_non_exhaustive()
    }
}

type CoreCall<'c, R> = Pin<Box<dyn Future<Output = Result<R>> + Send + 'c>>;

impl<'a> Session<'a> {
    /// Build a session; nothing dials until first use.
    pub(crate) fn create(
        model: Option<&'a Model>,
        env: EnvTarget<'a>,
        options: SessionOptions,
    ) -> Result<Session<'a>> {
        let SessionOptions {
            close_env,
            trust_entrypoints,
            instruction,
            view,
            execution_horizon,
            workflow_edition,
            cancel,
        } = options;
        if let Some(view) = &view {
            view.validate()?;
        }
        let context = LibraryContext::new();
        let (model_port, edition) = match model {
            Some(model) => {
                let shared = model.shared();
                shared.check_not_reentrant("session()")?;
                shared.admit()?;
                check_horizon(execution_horizon, shared.decl.native_chunk)?;
                let edition = crate::editions::resolve_native(
                    crate::editions::EditionSources {
                        call: workflow_edition.as_deref(),
                        option: None,
                        declared: shared.decl.workflow_edition.as_deref(),
                    },
                    shared.decl.authored,
                )?;
                if execution_horizon > 1 && !shared.decl.has_chunk() {
                    crate::warnings::warn(
                        WarningCategory::UserWarning,
                        format!(
                            "execution_horizon={execution_horizon} was requested but the model \
                             defines no predict_chunk(); running un-chunked."
                        ),
                    )?;
                }
                let trust = trust_entrypoints.unwrap_or(shared.decl.trust_entrypoints);
                let handler = crate::model::bridge::engine_handler_configured(
                    Arc::clone(shared),
                    trust,
                    instruction,
                    Some(context.clone()),
                );
                (
                    ModelPort::Local(Box::new(LocalModel {
                        shared: Arc::clone(shared),
                        handler,
                        env_id: crate::mint_id(),
                        session_id: format!("session-{}", crate::mint_id()),
                        requested_horizon: execution_horizon,
                        horizon: 1,
                        wants_history: false,
                        replay: VecDeque::new(),
                        history: Vec::new(),
                    })),
                    edition,
                )
            }
            None => {
                check_horizon(execution_horizon, None)?;
                let edition = crate::editions::resolve_native(
                    crate::editions::EditionSources {
                        call: workflow_edition.as_deref(),
                        option: None,
                        declared: None,
                    },
                    false,
                )?;
                if execution_horizon > 1 {
                    crate::warnings::warn(
                        WarningCategory::UserWarning,
                        format!(
                            "execution_horizon={execution_horizon} was requested but \
                             RANDOM_SAMPLE defines no predict_chunk(); running un-chunked."
                        ),
                    )?;
                }
                (ModelPort::Random(None), edition)
            }
        };
        let (port, queue) = caller_channel();
        let mut lent = None;
        let (env_port, caller_env, server) = match env.0 {
            EnvKind::Address(address) => (
                EnvPort::Remote {
                    address,
                    token: String::new(),
                    client: None,
                    borrowed: false,
                },
                CallerEnv::None,
                None,
            ),
            EnvKind::Server(server) => (
                EnvPort::Remote {
                    address: server.address().to_string(),
                    token: server.token().unwrap_or_default().to_string(),
                    client: None,
                    borrowed: false,
                },
                CallerEnv::None,
                Some(server),
            ),
            EnvKind::Remote(handle) => {
                let core = handle.lend()?;
                lent = Some(handle);
                (
                    EnvPort::Remote {
                        address: core.address().to_string(),
                        token: String::new(),
                        client: Some(core),
                        borrowed: true,
                    },
                    CallerEnv::None,
                    None,
                )
            }
            EnvKind::Env(env) => (
                EnvPort::Local,
                CallerEnv::Borrowed(LocalEnv::Env(env)),
                None,
            ),
            EnvKind::Vector(env) => (
                EnvPort::Local,
                CallerEnv::Borrowed(LocalEnv::Vector(env)),
                None,
            ),
            EnvKind::Factory(factory) => (EnvPort::Local, CallerEnv::Factory(factory, None), None),
        };
        Ok(Session {
            core: Core {
                model: model_port,
                env: env_port,
                port,
                context,
                edition,
                connected: None,
                episode: Episode::default(),
                telemetry: None,
                cancel: CancellationToken::new(),
                subscriptions: Arc::default(),
                viewer: view
                    .map(|v| Arc::new(std::sync::Mutex::new(crate::view::ViewerDriver::new(v)))),
                episode_started: Instant::now(),
                run_episode: 0,
                run_episodes: 0,
                last_predict_ms: 0.0,
                last_step_ms: 0.0,
                last_pace: Duration::ZERO,
                last_view_failed: false,
            },
            env: caller_env,
            queue,
            server,
            lent,
            close_env,
            closed: false,
            cancel: cancel.unwrap_or_default(),
            _model: PhantomData,
        })
    }

    /// Bind a served model to `env` and dial it: options and the target's
    /// kind and arity are checked before anything dials; an address is
    /// dialed before the model and closed again if the model dial fails.
    pub(crate) async fn create_remote(
        model: &'a RemoteModel,
        env: EnvTarget<'a>,
        options: SessionOptions,
    ) -> Result<Session<'a>> {
        let SessionOptions {
            close_env,
            trust_entrypoints,
            instruction,
            view,
            execution_horizon,
            workflow_edition,
            cancel,
        } = options;
        if let Some(view) = &view {
            view.validate()?;
        }
        if instruction.is_some() {
            return Err(Error::invalid_argument(
                "instruction= is not supported for RemoteModel sessions",
            ));
        }
        if trust_entrypoints.is_some() {
            return Err(Error::invalid_argument(
                "trust_entrypoints= applies to local model sources; it is not supported for \
                 RemoteModel sessions",
            ));
        }
        check_horizon(execution_horizon, None)?;
        let edition = crate::editions::resolve_native(
            crate::editions::EditionSources {
                call: workflow_edition.as_deref(),
                option: None,
                declared: None,
            },
            false,
        )?;
        let single_lane = |num_envs: u64| -> Result<()> {
            if num_envs > 1 {
                return Err(Error::invalid_argument(format!(
                    "RemoteModel drives a single env, but the env contract reports \
                     num_envs={num_envs}; use num_envs=1 (this client sends one observation \
                     per predict on lane 0)"
                )));
            }
            Ok(())
        };
        let cancel = cancel.unwrap_or_default();
        let dial_cancel = CancellationToken::new();
        let _watch = CancelWatch::start(vec![cancel.clone()], dial_cancel.clone())?;
        // What the session's env is, and the contract and offer the model is
        // bound with.
        let mut lent = None;
        let mut server = None;
        let (env_port, contract, env_offer) = match env.0 {
            EnvKind::Env(_) | EnvKind::Vector(_) | EnvKind::Factory(_) => {
                return Err(Error::Type(
                    "a served model runs against an env client exposing `env_contract` \
                     (RemoteEnv, an EnvServer, or an address to dial); got a local env"
                        .to_string(),
                ));
            }
            EnvKind::Server(served) => {
                let contract = served.env_contract();
                single_lane(contract.num_envs())?;
                server = Some(served);
                (
                    EnvPort::Remote {
                        address: served.address().to_string(),
                        token: served.token().unwrap_or_default().to_string(),
                        client: None,
                        borrowed: false,
                    },
                    contract.to_wire()?,
                    None,
                )
            }
            EnvKind::Remote(handle) => {
                let core = handle.lend()?;
                if let Err(err) = single_lane(core.num_envs() as u64) {
                    handle.give_back(core);
                    return Err(err);
                }
                let contract = core.env_contract().clone();
                let offer = core.session_offer().clone();
                lent = Some(handle);
                (
                    EnvPort::Remote {
                        address: core.address().to_string(),
                        token: String::new(),
                        client: Some(core),
                        borrowed: true,
                    },
                    contract,
                    Some(offer),
                )
            }
            EnvKind::Address(address) => {
                let connected = EnvClientCore::connect_with(
                    &address,
                    CoreConnect {
                        declared: edition.as_deref(),
                        ..CoreConnect::default()
                    },
                )
                .await;
                if _watch.fired() {
                    return Err(Error::Cancelled);
                }
                let mut core = connected?;
                if let Err(err) = core.require_single() {
                    let _ = core.close().await;
                    return Err(err);
                }
                let contract = core.env_contract().clone();
                let offer = core.session_offer().clone();
                (
                    EnvPort::Remote {
                        address,
                        token: String::new(),
                        client: Some(Box::new(core)),
                        borrowed: false,
                    },
                    contract,
                    Some(offer),
                )
            }
        };
        let mut lease = EnvLease {
            env: Some(env_port),
            lent,
        };
        let dialed = ModelClient::connect_with(
            model.address(),
            contract,
            ModelConnect {
                token: model.token(),
                env_offer,
                declared: edition.as_deref(),
                connect_timeout: model.connect_timeout(),
                request_timeout: model.request_timeout(),
            },
        )
        .await;
        // A cancel fired during the uncancellable dial fails this call once
        // it returns (RMT-6).
        let cancelled = dial_cancel.is_cancelled() || _watch.fired();
        // A firing wins over either dial outcome and is consumed by this call.
        if cancelled {
            return Err(Error::Cancelled);
        }
        let mut client = dialed?;
        let env_port = lease
            .env
            .take()
            .expect("the pending env is owned until transfer");
        let lent = lease.lent.take();
        client.set_execution_horizon(execution_horizon);
        let (port, queue) = caller_channel();
        Ok(Session {
            core: Core {
                model: ModelPort::Remote(Box::new(client)),
                env: env_port,
                port,
                context: LibraryContext::new(),
                edition,
                connected: None,
                episode: Episode::default(),
                telemetry: None,
                cancel: CancellationToken::new(),
                subscriptions: Arc::default(),
                viewer: view
                    .map(|v| Arc::new(std::sync::Mutex::new(crate::view::ViewerDriver::new(v)))),
                episode_started: Instant::now(),
                run_episode: 0,
                run_episodes: 0,
                last_predict_ms: 0.0,
                last_step_ms: 0.0,
                last_pace: Duration::ZERO,
                last_view_failed: false,
            },
            env: CallerEnv::None,
            queue,
            server,
            lent,
            close_env,
            closed: false,
            cancel,
            _model: PhantomData,
        })
    }

    /// Return a caller's lent remote client core (at close or drop).
    fn give_back(&mut self) {
        if let Some(handle) = self.lent.as_mut()
            && let EnvPort::Remote {
                client,
                borrowed: true,
                ..
            } = &mut self.core.env
            && let Some(core) = client.take()
        {
            handle.give_back(core);
        }
    }

    /// Run one call: its caller-side env and hooks lent, its tokens watched.
    async fn call<R: Send + 'static>(
        &mut self,
        mode: Mode,
        hooks: Option<&mut (dyn RunHooks + '_)>,
        cancel: Option<crate::ext::CancelToken>,
        op: impl for<'c> FnOnce(&'c mut Core) -> CoreCall<'c, R>,
    ) -> Result<R> {
        if self.closed {
            return Err(closed_error());
        }
        if let CallerEnv::Factory(factory, made @ None) = &mut self.env {
            *made = Some(mode.call(|| factory.make())?);
        }
        let _context_busy = self.core.context.busy()?;
        self.core.cancel = CancellationToken::new();
        let tokens = std::iter::once(self.cancel.clone()).chain(cancel).collect();
        let _watch = CancelWatch::start(tokens, self.core.cancel.clone())?;
        self.core.subscriptions = _watch.subscriptions();
        let Session {
            core, env, queue, ..
        } = self;
        let mut side = CallerSide {
            env: env.local(),
            hooks: super::drive::lend_hooks(hooks),
        };
        let out = mode.drive(op(core), queue, &mut side).await;
        // A job left queued by an abandoned call never runs later.
        while queue.try_recv().is_ok() {}
        out
    }

    /// Begin an episode: end the open one (the model's `on_episode_end`),
    /// then reset the env. `trial_index` reaches only an env that declares
    /// the option; another resets without it and warns.
    pub fn reset(
        &mut self,
        seed: Option<i64>,
        trial_index: Option<i64>,
    ) -> Result<(Value, InfoDict)> {
        refuse_in_async_context("Session::reset")?;
        run_sync(self.reset_async_in(Mode::Sync, seed, trial_index))
    }

    /// The async twin of [`reset`](Session::reset).
    pub async fn reset_async(
        &mut self,
        seed: Option<i64>,
        trial_index: Option<i64>,
    ) -> Result<(Value, InfoDict)> {
        self.reset_async_in(Mode::Async, seed, trial_index).await
    }

    async fn reset_async_in(
        &mut self,
        mode: Mode,
        seed: Option<i64>,
        trial_index: Option<i64>,
    ) -> Result<(Value, InfoDict)> {
        self.call(mode, None, None, move |core| {
            Box::pin(core.reset(seed, trial_index))
        })
        .await
    }

    /// Map one observation to an env-ready action (the model's adapter
    /// applied, a chunk replayed one action per step).
    pub fn predict(&mut self, observation: Value) -> Result<Value> {
        refuse_in_async_context("Session::predict")?;
        run_sync(self.call(Mode::Sync, None, None, move |core| {
            Box::pin(core.predict(observation))
        }))
    }

    /// The async twin of [`predict`](Session::predict).
    pub async fn predict_async(&mut self, observation: Value) -> Result<Value> {
        self.call(Mode::Async, None, None, move |core| {
            Box::pin(core.predict(observation))
        })
        .await
    }

    /// Apply one action to the env (`(observation, reward, terminated,
    /// truncated, info)`).
    #[allow(clippy::type_complexity)]
    pub fn step(&mut self, action: Value) -> Result<(Value, f64, bool, bool, InfoDict)> {
        refuse_in_async_context("Session::step")?;
        run_sync(self.call(Mode::Sync, None, None, move |core| {
            Box::pin(core.step(action))
        }))
    }

    /// The async twin of [`step`](Session::step).
    #[allow(clippy::type_complexity)]
    pub async fn step_async(
        &mut self,
        action: Value,
    ) -> Result<(Value, f64, bool, bool, InfoDict)> {
        self.call(Mode::Async, None, None, move |core| {
            Box::pin(core.step(action))
        })
        .await
    }

    /// Drive whole episodes and return their result; the session stays
    /// open. Hook errors abort the run, and `on_run_end` fires exactly once.
    pub fn run(&mut self, options: SessionRunOptions<'_>) -> Result<RunResult> {
        refuse_in_async_context("Session::run")?;
        run_sync(self.run_in(Mode::Sync, options))
    }

    /// The async twin of [`run`](Session::run). Its hooks run inside this
    /// future.
    pub async fn run_async(&mut self, options: SessionRunOptions<'_>) -> Result<RunResult> {
        self.run_in(Mode::Async, options).await
    }

    async fn run_in(&mut self, mode: Mode, options: SessionRunOptions<'_>) -> Result<RunResult> {
        let SessionRunOptions {
            episodes,
            seeds,
            max_episode_steps,
            max_episode,
            hooks,
            trial_index_base,
            cancel,
        } = options;
        // RUN-P1: the budget and caps resolve before connecting.
        let caps = check_caps(max_episode_steps, max_episode)?;
        let episodes = episode_budget(episodes, seeds.as_deref())?;
        let plan = RunPlan {
            episodes,
            seeds,
            max_episode_steps: caps.max_episode_steps,
            max_episode: max_episode.filter(|cap| *cap != Duration::MAX),
            trial_index_base,
            hooks: hooks.is_some(),
        };
        let result = self
            .call(mode, hooks, cancel, move |core| Box::pin(core.run(plan)))
            .await;
        self.core.last_view_failed = result.is_err();
        result
    }

    /// Whether the current episode has terminated or truncated.
    pub fn done(&self) -> bool {
        self.core.done()
    }

    /// The connected env's contract (connects on first use).
    pub fn contract(&mut self) -> Result<&EnvContract> {
        refuse_in_async_context("Session::contract")?;
        run_sync(self.call(Mode::Sync, None, None, |core| {
            Box::pin(core.ensure_connected())
        }))?;
        Ok(&self.core.connected()?.contract)
    }

    /// The async twin of [`contract`](Session::contract).
    pub async fn contract_async(&mut self) -> Result<&EnvContract> {
        self.call(Mode::Async, None, None, |core| {
            Box::pin(core.ensure_connected())
        })
        .await?;
        Ok(&self.core.connected()?.contract)
    }

    /// Lanes the connected env runs (a session drives one).
    pub fn num_envs(&mut self) -> Result<u64> {
        Ok(self.contract()?.num_envs().max(1))
    }

    /// The async twin of [`num_envs`](Session::num_envs).
    pub async fn num_envs_async(&mut self) -> Result<u64> {
        Ok(self.contract_async().await?.num_envs().max(1))
    }

    /// The workflow edition the session's env connection settled on; a
    /// local env negotiates none (the `RuntimeError` class).
    pub fn selected_workflow_edition(&mut self) -> Result<String> {
        refuse_in_async_context("Session::selected_workflow_edition")?;
        run_sync(self.call(Mode::Sync, None, None, |core| {
            Box::pin(core.selected_workflow_edition())
        }))
    }

    /// The async twin of
    /// [`selected_workflow_edition`](Session::selected_workflow_edition).
    pub async fn selected_workflow_edition_async(&mut self) -> Result<String> {
        self.call(Mode::Async, None, None, |core| {
            Box::pin(core.selected_workflow_edition())
        })
        .await
    }

    /// Resolve a reusable reader for semantic roles or single-role input leaves.
    pub fn reader(&mut self, items: &[super::ReadItem]) -> Result<super::Reader> {
        let contract = self.contract()?;
        super::read::reader(contract, items)
    }
    /// The async twin of `reader`.
    pub async fn reader_async(&mut self, items: &[super::ReadItem]) -> Result<super::Reader> {
        let contract = self.contract_async().await?;
        super::read::reader(contract, items)
    }
    /// Read a role or leaf from an observation.
    pub fn read(&mut self, observation: &Value, item: impl Into<super::ReadItem>) -> Result<Value> {
        self.contract()?;
        self.core.context.read(observation, &item.into())
    }
    /// The async twin of `read`.
    pub async fn read_async(
        &mut self,
        observation: &Value,
        item: impl Into<super::ReadItem>,
    ) -> Result<Value> {
        self.contract_async().await?;
        self.core.context.read(observation, &item.into())
    }
    /// Observation roles grouped by modality in authored declaration order.
    pub fn observation_roles(&mut self) -> Result<crate::adapters::ObservationRoles> {
        let contract = self.contract()?;
        Ok(crate::adapters::EnvTags::from_metadata(
            contract.metadata().unwrap_or(&Metadata::new()),
        )?
        .map_or_else(Default::default, |t| t.observation_roles()))
    }
    /// The async twin of `observation_roles`.
    pub async fn observation_roles_async(&mut self) -> Result<crate::adapters::ObservationRoles> {
        let contract = self.contract_async().await?;
        Ok(crate::adapters::EnvTags::from_metadata(
            contract.metadata().unwrap_or(&Metadata::new()),
        )?
        .map_or_else(Default::default, |t| t.observation_roles()))
    }
    /// Discover sources, borrowing this session for its render callable.
    pub fn frame_sources(&mut self) -> Result<super::FrameSources<'_>> {
        let contract = self.contract()?.clone();
        self.sources(contract)
    }
    /// The async twin of `frame_sources`.
    pub async fn frame_sources_async(&mut self) -> Result<super::FrameSources<'_>> {
        let contract = self.contract_async().await?.clone();
        self.sources(contract)
    }
    fn sources(&mut self, contract: EnvContract) -> Result<super::FrameSources<'_>> {
        let roles = super::read::image_roles(Some(&contract))?;
        let label = super::read::render_label(&contract, &roles);
        let render = label
            .as_ref()
            .map(|_| super::RenderFn::new(|| self.render_current()));
        Ok(super::FrameSources {
            roles,
            render_label: label,
            render,
        })
    }
    fn render_current(&mut self) -> Result<Value> {
        if self.closed {
            return Err(closed_error());
        }
        if let Some(mut env) = self.env.local() {
            return match &mut env {
                LocalEnv::Env(e) => e.render().map(|v| v.unwrap_or(Value::None)),
                LocalEnv::Vector(e) => e.render().map(|v| v.unwrap_or(Value::None)),
            };
        }
        self.core.context.render()
    }

    /// This session as its hooks see it (connects first). The context fails
    /// with the `RuntimeError` class once the session has closed.
    pub fn run_context(&mut self) -> Result<Arc<dyn RunContext>> {
        self.contract()?;
        Ok(self.core.context.clone())
    }

    /// The async twin of [`run_context`](Session::run_context).
    pub async fn run_context_async(&mut self) -> Result<Arc<dyn RunContext>> {
        self.contract_async().await?;
        Ok(self.core.context.clone())
    }

    /// The session's cancel token: the `SessionOptions::cancel` one, or one
    /// the session made; the same token on every call.
    pub fn cancel_handle(&self) -> crate::ext::CancelToken {
        self.cancel.clone()
    }

    /// Close the session: end an open episode (the model's
    /// `on_episode_end`), release the route and a connection the session
    /// dialed, and on `close_env` shut the env down. Idempotent; the first
    /// release error is returned. The model is never closed.
    pub fn close(&mut self) -> Result<()> {
        refuse_in_async_context("Session::close")?;
        run_sync(self.close_in(Mode::Sync))
    }

    /// The async twin of [`close`](Session::close).
    pub async fn close_async(&mut self) -> Result<()> {
        self.close_in(Mode::Async).await
    }

    async fn close_in(&mut self, mode: Mode) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        let hold = self
            .call(mode, None, None, |core| Box::pin(core.close_view()))
            .await;
        if let Some(viewer) = self.core.viewer.take() {
            viewer.lock().unwrap_or_else(|e| e.into_inner()).disable();
        }
        let close_env = self.close_env;
        let released = self
            .call(mode, None, None, move |core| {
                Box::pin(core.release(close_env))
            })
            .await;
        self.closed = true;
        self.core.context.close();
        self.give_back();
        let mut first = hold.and(released);
        if close_env && let Some(server) = self.server {
            first = first.and(match mode {
                Mode::Sync => server.shutdown(),
                Mode::Async => server.shutdown_async().await,
            });
        }
        // A caller's env closes on the opt-in; one built here always.
        let built = matches!(self.env, CallerEnv::Factory(_, Some(_)));
        if close_env || built {
            let closed = match self.env.local() {
                Some(LocalEnv::Env(env)) => mode.call(|| env.close()),
                Some(LocalEnv::Vector(env)) => mode.call(|| env.close()),
                None => Ok(()),
            };
            first = first.and(closed);
        }
        first
    }
}

impl Drop for Session<'_> {
    fn drop(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;
        if let Some(viewer) = self.core.viewer.take() {
            viewer.lock().unwrap_or_else(|e| e.into_inner()).disable();
        }
        self.core.context.close();
        if super::drive::in_async_context() {
            // Queue the owned episode end before a later model close can refuse it.
            if self.core.episode.open
                && let ModelPort::Local(model) = &self.core.model
            {
                let _ = model.shared.queue_episode_end(self.core.episode.id.clone());
                self.core.episode.open = false;
            }
            let made = match &mut self.env {
                CallerEnv::Factory(_, made) => made.take(),
                _ => None,
            };
            let _made = super::native::OwnedMadeEnv(made, false);
            // A borrowed client goes back to its owner at once: a borrow
            // cannot be detached (R-17).
            self.give_back();
            // Inside a runtime nothing may block (R-17): the owned cleanup
            // detaches; a borrowed env is not closed.
            let mut core = std::mem::replace(
                &mut self.core,
                Core {
                    model: ModelPort::Random(None),
                    env: EnvPort::Local,
                    port: caller_channel().0,
                    context: LibraryContext::new(),
                    edition: None,
                    connected: None,
                    episode: Episode::default(),
                    telemetry: None,
                    cancel: CancellationToken::new(),
                    subscriptions: Arc::default(),
                    viewer: None,
                    episode_started: Instant::now(),
                    run_episode: 0,
                    run_episodes: 0,
                    last_predict_ms: 0.0,
                    last_step_ms: 0.0,
                    last_pace: Duration::ZERO,
                    last_view_failed: false,
                },
            );
            let close_env = self.close_env;
            let detached = matches!(
                core.env,
                EnvPort::Remote {
                    borrowed: false,
                    ..
                }
            );
            if detached || core.episode.open || core.connected.is_some() {
                if let EnvPort::Remote {
                    client: Some(client),
                    borrowed: false,
                    ..
                } = &core.env
                {
                    client.prepare_close();
                }
                super::drive::runtime().spawn(async move {
                    let _ = core.release(close_env).await;
                });
            }
            return;
        }
        // Outside one, finish the owned cleanup with a bound, then close a
        // borrowed env on the opt-in (one built here always).
        let close_env = self.close_env;
        let built = matches!(self.env, CallerEnv::Factory(_, Some(_)));
        {
            let Session {
                core, env, queue, ..
            } = &mut *self;
            let mut side = CallerSide {
                env: env.local(),
                hooks: None,
            };
            let _ = super::drive::block_on_caller(
                async {
                    tokio::time::timeout(DROP_BOUND, core.release(close_env))
                        .await
                        .unwrap_or(Err(Error::Timeout(DROP_BOUND)))
                },
                queue,
                &mut side,
            );
        }
        self.give_back();
        if close_env && let Some(server) = self.server {
            let _ = server.shutdown();
        }
        if close_env || built {
            let _ = match self.env.local() {
                Some(LocalEnv::Env(env)) => env.close(),
                Some(LocalEnv::Vector(env)) => env.close(),
                None => Ok(()),
            };
        }
    }
}
