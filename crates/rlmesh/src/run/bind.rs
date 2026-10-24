//! `Binding<'a>`: a model bound to one env and stepped by hand, as Python's
//! `Model.bind(target)`.
//!
//! The engine's [`Binding`](crate::model::bind::Binding) drives the env over
//! its session. A local env (or a factory's) is served in this process
//! behind a loopback server whose lanes relay each call to the thread that
//! drives the binding, so the env only ever runs on its caller's thread.

use std::sync::Arc;
use std::time::Instant;

use super::drive::{
    CallerPort, CallerQueue, CallerSide, LocalEnv, block_on_caller, caller_channel,
    refuse_in_async_context,
};
use super::hooks::LibraryContext;
use super::lent::{LentEnv, LentVectorEnv};
use super::native::OwnedMadeEnv;
use super::options::{BindOptions, check_horizon};
use super::targets::{EnvKind, EnvTarget, ModelKind, ModelTarget};
use super::{EpisodeResult, LoopbackEnv, RunContext, RunHooks, RunResult, StepEvent};
use crate::model::handler::ModelHandler;
use crate::value::{meta_codec, space_codec};
use crate::{EnvContract, EnvServer, Error, InfoDict, MadeEnv, Metadata, Result, Value};

/// The env a binding serves itself, run on the caller's side.
enum Local<'a> {
    Borrowed(LocalEnv<'a>),
    /// Built from a factory at bind and closed with the binding.
    Made(OwnedMadeEnv),
}

enum HookSink<'a> {
    Borrowed(&'a mut dyn RunHooks),
    Owned(Box<dyn RunHooks>),
}

impl<'a> std::ops::Deref for HookSink<'a> {
    type Target = dyn RunHooks + 'a;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Borrowed(hooks) => &**hooks,
            Self::Owned(hooks) => &**hooks,
        }
    }
}
impl<'a> std::ops::DerefMut for HookSink<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Borrowed(hooks) => &mut **hooks,
            Self::Owned(hooks) => &mut **hooks,
        }
    }
}

/// The hooks a binding feeds, with the episodes it has seen.
struct Track<'a> {
    hooks: HookSink<'a>,
    /// The info keys the binding's edition reads an episode's success from.
    success_keys: &'static [&'static str],
    episodes: Vec<EpisodeResult>,
    index: i64,
    seed: Option<i64>,
    steps: i64,
    reward: f64,
    started: Instant,
    predict_ms: Vec<f64>,
    step_ms: Vec<f64>,
    /// The latest reset's or step's info.
    info: InfoDict,
    /// The observation the next action is predicted from.
    last: Value,
}

/// A model bound to one env lane; see [`Model::bind`](crate::Model::bind).
pub struct Binding<'a> {
    inner: Option<crate::model::bind::Binding>,
    /// Set once connected.
    contract: Option<EnvContract>,
    context: Arc<LibraryContext>,
    local: Option<Local<'a>>,
    queue: CallerQueue,
    loopback: Option<EnvServer>,
    /// A served env this binding shuts down on close (`close_env`).
    shutdown: Option<&'a EnvServer>,
    close_env: bool,
    done: Arc<std::sync::atomic::AtomicBool>,
    cancel: tokio_util::sync::CancellationToken,
    track: Option<Track<'a>>,
    adapted: Option<crate::binding::model::OwnedTarget>,
    workflow_edition: Option<String>,
}

impl std::fmt::Debug for Binding<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Binding")
            .field("done", &self.done)
            .field("closed", &self.inner.is_none())
            .finish_non_exhaustive()
    }
}

fn closed_error() -> Error {
    Error::Runtime("the binding is closed".into())
}

impl<'a> Binding<'a> {
    /// Bind `model` to `env`: connects now, serving a local env in this
    /// process for the binding's life. A vector env with more than one
    /// lane and an open `RemoteEnv` are refused.
    pub(crate) fn create(
        model: ModelTarget<'a>,
        env: EnvTarget<'a>,
        options: BindOptions<'a>,
    ) -> Result<Binding<'a>> {
        refuse_in_async_context("bind")?;
        let BindOptions {
            close_env,
            execution_horizon,
            workflow_edition,
            hooks,
            view,
        } = options;
        if view.is_some() {
            return Err(Error::Type(
                "view= is not supported by the native binding loop".into(),
            ));
        }
        let context = LibraryContext::new();
        let (handler, edition, execution_horizon): (Box<dyn ModelHandler>, _, _) = match model.0 {
            ModelKind::Model(model) => {
                let shared = model.shared();
                shared.check_not_reentrant("bind()")?;
                shared.admit()?;
                let execution_horizon = shared.decl.resolve_horizon(execution_horizon)?;
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
                        crate::WarningCategory::UserWarning,
                        format!(
                            "execution_horizon={execution_horizon} was requested but the model \
                             defines no predict_chunk(); running un-chunked."
                        ),
                    )?;
                }
                let handler = crate::model::bridge::engine_handler_configured(
                    Arc::clone(shared),
                    shared.decl.trust_entrypoints,
                    None,
                    Some(context.clone()),
                );
                (Box::new(handler), edition, execution_horizon)
            }
            ModelKind::Remote(model) => {
                if !crate::address::is_v2(model.address()) {
                    return Err(Error::invalid_argument(format!(
                        "bind: {:?} is a v1 model endpoint, which rlmesh 0.2 no longer dials; \
                         reach a served model by its participant address (e.g. \
                         'lab:7447/openpi')",
                        model.address()
                    )));
                }
                let timeout = model
                    .connect_timeout()
                    .unwrap_or(super::native::SERVED_CONNECT);
                if let Some(horizon) = execution_horizon {
                    check_horizon(horizon, None)?;
                }
                let execution_horizon = match execution_horizon {
                    Some(horizon) => horizon,
                    None => super::drive::block_on_library(
                        crate::lockstep::ServedModel::default_execution_horizon(
                            model.address(),
                            timeout,
                        ),
                    )?
                    .unwrap_or(1),
                };
                check_horizon(execution_horizon, None)?;
                let handler = crate::lockstep::ServedModel::new(model.address(), timeout);
                (
                    Box::new(handler),
                    unauthored(workflow_edition.as_deref())?,
                    execution_horizon,
                )
            }
            ModelKind::Prepared(model) => {
                let horizon = super::options::resolve_execution_horizon(
                    execution_horizon.map(i128::from),
                    model.execution_horizon.map(i128::from),
                    None,
                    model.native_chunk,
                )?;
                let edition = crate::editions::resolve_native(
                    crate::editions::EditionSources {
                        call: workflow_edition.as_deref(),
                        option: None,
                        declared: model.workflow_edition.as_deref(),
                    },
                    model.authored,
                )?;
                if horizon > 1 && model.has_chunk == Some(false) {
                    crate::warnings::warn(
                        crate::WarningCategory::UserWarning,
                        format!(
                            "execution_horizon={horizon} was requested but the model defines no predict_chunk(); running un-chunked."
                        ),
                    )?;
                }
                (model.handler, edition, horizon)
            }
            ModelKind::Random => {
                let execution_horizon = execution_horizon.unwrap_or(1);
                check_horizon(execution_horizon, None)?;
                (
                    Box::<crate::model::random::RandomSampleHandler>::default(),
                    unauthored(workflow_edition.as_deref())?,
                    execution_horizon,
                )
            }
        };
        let (port, queue) = caller_channel();
        let mut binding = Binding {
            inner: None,
            contract: None,
            context,
            local: None,
            queue,
            loopback: None,
            shutdown: None,
            close_env,
            done: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            cancel: tokio_util::sync::CancellationToken::new(),
            track: None,
            adapted: None,
            workflow_edition: edition.clone(),
        };
        let address = match env.0 {
            EnvKind::Address(address) => {
                super::options::check_address_close(&address, close_env)?;
                address
            }
            EnvKind::Adapted(adapter) => {
                binding.adapted = Some(crate::binding::model::OwnedTarget::new(adapter, close_env));
                let target = binding.adapted.as_mut().expect("installed target").open(
                    edition.as_deref(),
                    crate::binding::model::caller_port(port.clone()),
                )?;
                super::options::check_address_close(
                    &target.address,
                    close_env && target.close_over_wire,
                )?;
                target.address
            }

            EnvKind::Server(server) => {
                if close_env {
                    binding.shutdown = Some(server);
                }
                server.address().to_string()
            }
            EnvKind::Remote(_) => {
                return Err(Error::Type(
                    "bind(RemoteEnv(address)) is gone in rlmesh 0.2: an open RemoteEnv holds \
                     its env's lease, so close it and pass the address: bind(address)"
                        .into(),
                ));
            }
            EnvKind::Env(env) => {
                binding.serve(Local::Borrowed(LocalEnv::Env(env)), &port, &edition)?
            }
            EnvKind::Vector(env) => {
                binding.serve(Local::Borrowed(LocalEnv::Vector(env)), &port, &edition)?
            }
            EnvKind::Factory(mut factory) => {
                let made = OwnedMadeEnv(Some(factory.make()?), false);
                binding.serve(Local::Made(made), &port, &edition)?
            }
        };
        let inner = binding.drive(crate::model::bind::Binding::connect(
            handler,
            &address,
            execution_horizon,
            edition.as_deref(),
        ))?;
        let contract = EnvContract::from_wire(inner.contract())?;
        binding.context.connect(contract.clone());
        binding.contract = Some(contract);
        binding.inner = Some(inner);
        if let Some(hooks) = hooks {
            binding.attach_hooks(HookSink::Borrowed(hooks))?;
        }
        Ok(binding)
    }

    /// Install an owned binding observer, without a self-referential host borrow.
    #[doc(hidden)]
    pub(crate) fn observe(
        &mut self,
        hooks: Box<dyn RunHooks>,
        presentation: Box<dyn crate::binding::model::Presentation>,
    ) -> Result<()> {
        self.context.set_presentation(Some(presentation));
        self.attach_hooks(HookSink::Owned(hooks))
    }

    fn attach_hooks(&mut self, hooks: HookSink<'a>) -> Result<()> {
        let edition = self
            .workflow_edition
            .as_deref()
            .unwrap_or(rlmesh_proto::CURRENT_WORKFLOW_EDITION);
        let edition =
            rlmesh_proto::parse_retained_edition(edition).map_err(Error::invalid_argument)?;
        let track = self.track.insert(Track {
            hooks,
            success_keys: rlmesh_proto::defaults(edition).success_info_keys,
            episodes: Vec::new(),
            index: -1,
            seed: None,
            steps: 0,
            reward: 0.0,
            started: Instant::now(),
            predict_ms: Vec::new(),
            step_ms: Vec::new(),
            info: InfoDict::new(),
            last: Value::None,
        });
        track.hooks.on_run_start(self.context.clone())?;
        Ok(())
    }

    /// Serve `local` behind a loopback server relaying to this thread.
    fn serve(
        &mut self,
        local: Local<'a>,
        port: &CallerPort,
        edition: &Option<String>,
    ) -> Result<String> {
        self.local = Some(local);
        let lent = match self.side().env {
            Some(LocalEnv::Env(env)) => {
                LoopbackEnv::Env(Box::new(LentEnv::new(&*env, port.clone())))
            }
            Some(LocalEnv::Vector(env)) => {
                LoopbackEnv::Vector(Box::new(LentVectorEnv::new(&*env, port.clone())))
            }
            None => return Err(Error::Internal("no local env to serve".into())),
        };
        let server = EnvServer::loopback(lent, edition.clone())?;
        server.start()?;
        let address = server.address().to_string();
        self.loopback = Some(server);
        Ok(address)
    }

    /// What the loopback's lanes reach on this thread.
    fn side(&mut self) -> CallerSide<'_> {
        let env = match &mut self.local {
            Some(Local::Borrowed(LocalEnv::Env(env))) => Some(LocalEnv::Env(&mut **env)),
            Some(Local::Borrowed(LocalEnv::Vector(env))) => Some(LocalEnv::Vector(&mut **env)),
            Some(Local::Made(OwnedMadeEnv(Some(MadeEnv::Env(env)), _))) => {
                Some(LocalEnv::Env(&mut **env))
            }
            Some(Local::Made(OwnedMadeEnv(Some(MadeEnv::Vector(env)), _))) => {
                Some(LocalEnv::Vector(&mut **env))
            }
            _ => None,
        };
        CallerSide { env, hooks: None }
    }

    /// Start one host operation; its token spans every internal driver call
    /// (for example ending a live episode and then resetting its env).
    #[doc(hidden)]
    pub(crate) fn cancellation(&mut self) -> tokio_util::sync::CancellationToken {
        self.cancel = tokio_util::sync::CancellationToken::new();
        self.cancel.clone()
    }

    fn drive<F, T>(&mut self, future: F) -> Result<T>
    where
        F: std::future::Future<Output = Result<T>>,
    {
        let mut queue = std::mem::replace(&mut self.queue, caller_channel().1);
        let cancel = self.cancel.clone();
        let call = async {
            tokio::select! {
                result = future => result,
                _ = cancel.cancelled() => Err(Error::Cancelled),
            }
        };
        let out = block_on_caller(call, &mut queue, &mut self.side());
        self.queue = queue;
        out
    }

    fn engine(&mut self) -> Result<crate::model::bind::Binding> {
        refuse_in_async_context("Binding")?;
        self.inner.take().ok_or_else(closed_error)
    }

    /// Ends the live episode, if any, and starts the next: seeded with
    /// `seed`, with the model's reset options under `options`. Returns its
    /// first observation and the reset's info.
    pub fn reset(
        &mut self,
        seed: Option<i64>,
        options: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        let options = options.map(meta_codec::metadata_to_wire).transpose()?;
        if self.inner.is_none() {
            return Err(closed_error());
        }
        self.end_episode(false, true)?;
        let mut inner = self.engine()?;
        let out = self.drive(inner.reset(seed, options));
        self.inner = Some(inner);
        let (observation, info) = out?;
        self.done.store(false, std::sync::atomic::Ordering::Release);
        let observation =
            space_codec::value_from_space(self.contract().observation_space().wire(), observation)?;
        let info = info.map_or_else(Metadata::new, meta_codec::metadata_from_wire);
        if let Some(track) = &mut self.track {
            track.index += 1;
            track.seed = seed;
            track.steps = 0;
            track.reward = 0.0;
            track.started = Instant::now();
            track.predict_ms.clear();
            track.step_ms.clear();
            track.info = info.clone();
            track.last = observation.clone();
            track.hooks.on_episode_start(track.index, seed)?;
            self.context.frame(&super::RunFrame {
                observation: &observation,
                episode: track.index,
                seed,
                steps: 0,
                reward: 0.0,
                outcome: Some(""),
                predict_ms: 0.0,
                step_ms: 0.0,
                elapsed_s: track.started.elapsed().as_secs_f64(),
            })?;
        }
        Ok((observation, info))
    }

    /// One env step under `action`, or the model's when `None`:
    /// `(observation, reward, terminated, truncated, info)`.
    pub fn step(&mut self, action: Option<Value>) -> Result<(Value, f64, bool, bool, InfoDict)> {
        if self.done() {
            return Err(Error::Runtime(
                "the episode ended; reset to start the next".into(),
            ));
        }
        let action = action
            .map(|action| {
                space_codec::value_to_space(self.contract().action_space().wire(), &action)
            })
            .transpose()?;
        let mut inner = self.engine()?;
        let out = self.drive(inner.step(action));
        self.inner = Some(inner);
        let step = out?;
        let observation = space_codec::value_from_space(
            self.contract().observation_space().wire(),
            step.observation,
        )?;
        let info = step
            .info
            .map_or_else(Metadata::new, meta_codec::metadata_from_wire);
        let mut truncated = step.truncated;
        if let Some(track) = &mut self.track {
            let action = space_codec::value_from_space(
                self.contract
                    .as_ref()
                    .expect("a binding connects at creation")
                    .action_space()
                    .wire(),
                step.action,
            )?;
            let before = std::mem::replace(&mut track.last, observation.clone());
            let predict_ms = step.predict.as_secs_f64() * 1000.0;
            let step_ms = step.env.as_secs_f64() * 1000.0;
            track.steps += 1;
            track.reward += step.reward;
            track.predict_ms.push(predict_ms);
            track.step_ms.push(step_ms);
            track.info = info.clone();
            let outcome = if !step.terminated && !step.truncated {
                ""
            } else {
                match info_success(track.success_keys, &info) {
                    Some(true) => "success",
                    Some(false) if step.terminated => "failure",
                    Some(false) => "timeout",
                    None if step.terminated => "done",
                    None => "timeout",
                }
            };
            self.context.frame(&super::RunFrame {
                observation: &observation,
                episode: track.index,
                seed: track.seed,
                steps: track.steps,
                reward: track.reward,
                outcome: Some(outcome),
                predict_ms,
                step_ms,
                elapsed_s: track.started.elapsed().as_secs_f64(),
            })?;
            if self.context.directive()? != super::runtime::RunDirective::Continue {
                truncated = true;
            }
            let context: Arc<dyn RunContext> = self.context.clone();
            track.hooks.on_step(&StepEvent {
                read: super::read::read_fn(context, before.clone()),
                episode: track.index,
                seed: track.seed,
                step: track.steps - 1,
                observation: before,
                action,
                reward: step.reward,
                terminated: step.terminated,
                truncated,
                info: info.clone(),
                predict_ms,
                step_ms,
            })?;
        }
        if step.terminated || truncated {
            self.end_episode(step.terminated, truncated)?;
        }
        Ok((observation, step.reward, step.terminated, truncated, info))
    }

    /// Record the episode in flight as ended, once, for the hooks.
    fn end_episode(&mut self, terminated: bool, truncated: bool) -> Result<()> {
        if self.done.swap(true, std::sync::atomic::Ordering::AcqRel) {
            return Ok(());
        }
        // Directives and reset-over-live episodes evict model state at the
        // same edge as a natural terminal step; the engine's end is idempotent.
        let mut inner = self.engine()?;
        let ended = self.drive(inner.end());
        self.inner = Some(inner);
        ended?;
        let Some(track) = &mut self.track else {
            return Ok(());
        };
        let mean = |values: &[f64]| {
            (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
        };
        let result = EpisodeResult {
            index: track.index,
            seed: track.seed,
            steps: track.steps,
            reward: track.reward,
            terminated,
            truncated,
            success: info_success(track.success_keys, &track.info),
            duration_s: track.started.elapsed().as_secs_f64(),
            predict_ms: mean(&track.predict_ms),
            step_ms: mean(&track.step_ms),
            trial: None,
        };
        track.episodes.push(result.clone());
        track.hooks.on_episode_end(&result)
    }

    /// Whether the episode ended (or none started yet).
    pub fn done(&self) -> bool {
        self.done.load(std::sync::atomic::Ordering::Acquire)
    }

    #[doc(hidden)]
    pub(crate) fn done_flag(&self) -> Arc<std::sync::atomic::AtomicBool> {
        self.done.clone()
    }

    /// A renderer independent of a mutable binding borrow, for hook observers.
    #[doc(hidden)]
    pub(crate) fn renderer(&self) -> Result<Arc<dyn Fn() -> Result<Value> + Send + Sync>> {
        Ok(self.inner.as_ref().ok_or_else(closed_error)?.renderer())
    }

    /// The env's contract.
    pub fn contract(&self) -> &EnvContract {
        self.contract
            .as_ref()
            .expect("a binding connects at creation")
    }

    /// The env's render mode (`""` when it declares none).
    pub fn render_mode(&self) -> &str {
        self.contract().render_mode().unwrap_or_default()
    }

    /// Lanes a binding drives: one.
    pub fn num_envs(&self) -> u64 {
        1
    }

    /// Read a role or leaf from an observation.
    pub fn read(&self, observation: &Value, item: impl Into<super::ReadItem>) -> Result<Value> {
        self.context.read(observation, &item.into())
    }

    /// The env's image roles and its `render()` source.
    pub fn frame_sources(&mut self) -> Result<super::FrameSources<'_>> {
        let roles = super::read::image_roles(Some(self.contract()))?;
        let label = super::read::render_label(self.contract(), &roles);
        let render = label
            .as_ref()
            .map(|_| super::RenderFn::new(|| self.render()));
        Ok(super::FrameSources {
            roles,
            render_label: label,
            render,
        })
    }

    /// The env's rendered frame, or `Value::None`.
    pub fn render(&mut self) -> Result<Value> {
        let rendered = match self.side().env {
            Some(LocalEnv::Env(env)) => env.render(),
            Some(LocalEnv::Vector(env)) => env.render(),
            None => {
                let inner = self.engine()?;
                let png = self.drive(inner.render());
                self.inner = Some(inner);
                png?.map(|png| crate::env::client::decode_render_png(&png))
                    .transpose()
            }
        };
        Ok(rendered?.unwrap_or(Value::None))
    }

    /// Ends the episode and releases the env: closes a factory's env (and a
    /// local or served one under `close_env`). Never closes the model;
    /// idempotent.
    pub fn close(&mut self) -> Result<()> {
        let Some(inner) = self.inner.take() else {
            return Ok(());
        };
        refuse_in_async_context("Binding::close")?;
        self.inner = Some(inner);
        let ended = self.end_episode(false, true);
        let inner = self.inner.take().ok_or_else(closed_error)?;
        self.context.close();
        let mut result = ended.and(self.drive(inner.close()));
        // A lane still waiting on a relayed call fails at once.
        self.queue = caller_channel().1;
        if let Some(server) = self.loopback.take() {
            result = result.and(server.shutdown());
        }
        if let Some(server) = self.shutdown.take() {
            result = result.and(server.shutdown());
        }
        let closed = match self.local.take() {
            Some(Local::Made(mut made)) => {
                made.1 = true;
                match made.0.take() {
                    Some(MadeEnv::Env(mut env)) => env.close(),
                    Some(MadeEnv::Vector(mut env)) => env.close(),
                    None => Ok(()),
                }
            }
            Some(Local::Borrowed(LocalEnv::Env(env))) if self.close_env => env.close(),
            Some(Local::Borrowed(LocalEnv::Vector(env))) if self.close_env => env.close(),
            _ => Ok(()),
        };
        let mut result = result.and(closed);
        if let Some(mut target) = self.adapted.take() {
            result = result.and(target.release());
        }
        if let Some(mut track) = self.track.take() {
            let episodes = track.episodes;
            result = result.and(track.hooks.on_run_end(&RunResult {
                episodes,
                ..RunResult::default()
            }));
        }
        self.context.set_presentation(None);
        result
    }
}

impl Drop for Binding<'_> {
    fn drop(&mut self) {
        if self.inner.is_none() {
            return;
        }
        if super::drive::in_async_context() {
            // The engine's close needs this thread; the env's lane fails.
            if let Some(inner) = self.inner.take() {
                super::drive::runtime().spawn(async move {
                    let _ = inner.close().await;
                });
            }
            return;
        }
        let _ = self.close();
    }
}

/// The task outcome in an episode's last info, read as the native loop reads
/// it: the first of `keys` holding a bool, integer or number decides
/// (numbers by non-zero); `None` when none does.
fn info_success(keys: &[&str], info: &InfoDict) -> Option<bool> {
    keys.iter().find_map(|key| match info.get(key)? {
        Value::Bool(flag) => Some(*flag),
        Value::Int(value) => Some(*value != 0),
        Value::Float(value) => Some(*value != 0.0),
        _ => None,
    })
}

/// The edition a model with no declaration of its own (served, or the random
/// baseline) binds under.
fn unauthored(call: Option<&str>) -> Result<Option<String>> {
    crate::editions::resolve_native(
        crate::editions::EditionSources {
            call,
            option: None,
            declared: None,
        },
        false,
    )
}
