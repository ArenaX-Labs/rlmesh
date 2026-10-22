//! `Binding<'a>`: a model bound to one env and stepped by hand, as Python's
//! `Model.bind(target)`.
//!
//! The engine's [`Binding`](crate::model::bind::Binding) drives the env over
//! its session. A local env (or a factory's) is served in this process
//! behind a loopback server whose lanes relay each call to the thread that
//! drives the binding, so the env only ever runs on its caller's thread.

use std::sync::Arc;

use super::drive::{
    CallerPort, CallerQueue, CallerSide, LocalEnv, block_on_caller, caller_channel,
    refuse_in_async_context,
};
use super::hooks::LibraryContext;
use super::lent::{LentEnv, LentVectorEnv};
use super::native::OwnedMadeEnv;
use super::options::{BindOptions, check_horizon};
use super::targets::{EnvKind, EnvTarget, ModelKind, ModelTarget};
use super::{LoopbackEnv, RunContext};
use crate::model::handler::ModelHandler;
use crate::value::{meta_codec, space_codec};
use crate::{EnvContract, EnvServer, Error, InfoDict, MadeEnv, Metadata, Result, Value};

/// The env a binding serves itself, run on the caller's side.
enum Local<'a> {
    Borrowed(LocalEnv<'a>),
    /// Built from a factory at bind and closed with the binding.
    Made(OwnedMadeEnv),
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
    done: bool,
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
        options: BindOptions,
    ) -> Result<Binding<'a>> {
        refuse_in_async_context("bind")?;
        let BindOptions {
            close_env,
            execution_horizon,
            workflow_edition,
        } = options;
        let context = LibraryContext::new();
        let (handler, edition): (Box<dyn ModelHandler>, _) = match model.0 {
            ModelKind::Model(model) => {
                let shared = model.shared();
                shared.check_not_reentrant("bind()")?;
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
                (Box::new(handler), edition)
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
                check_horizon(execution_horizon, None)?;
                let handler = crate::lockstep::ServedModel::new(
                    model.address(),
                    model
                        .connect_timeout()
                        .unwrap_or(super::native::SERVED_CONNECT),
                );
                (Box::new(handler), unauthored(workflow_edition.as_deref())?)
            }
            ModelKind::Random => {
                check_horizon(execution_horizon, None)?;
                (
                    Box::<crate::model::random::RandomSampleHandler>::default(),
                    unauthored(workflow_edition.as_deref())?,
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
            done: true,
        };
        let address = match env.0 {
            EnvKind::Address(address) => address,
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
        Ok(binding)
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

    /// Drive `future` here, running the local env's calls between polls.
    fn drive<F: std::future::Future>(&mut self, future: F) -> F::Output {
        let mut queue = std::mem::replace(&mut self.queue, caller_channel().1);
        let out = block_on_caller(future, &mut queue, &mut self.side());
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
        let mut inner = self.engine()?;
        let out = self.drive(inner.reset(seed, options));
        self.inner = Some(inner);
        let (observation, info) = out?;
        self.done = false;
        Ok((
            space_codec::value_from_space(self.contract().observation_space().wire(), observation)?,
            info.map_or_else(Metadata::new, meta_codec::metadata_from_wire),
        ))
    }

    /// One env step under `action`, or the model's when `None`:
    /// `(observation, reward, terminated, truncated, info)`.
    pub fn step(&mut self, action: Option<Value>) -> Result<(Value, f64, bool, bool, InfoDict)> {
        let action = action
            .map(|action| {
                space_codec::value_to_space(self.contract().action_space().wire(), &action)
            })
            .transpose()?;
        let mut inner = self.engine()?;
        let out = self.drive(inner.step(action));
        self.inner = Some(inner);
        let step = out?;
        self.done = step.terminated || step.truncated;
        Ok((
            space_codec::value_from_space(
                self.contract().observation_space().wire(),
                step.observation,
            )?,
            step.reward,
            step.terminated,
            step.truncated,
            step.info
                .map_or_else(Metadata::new, meta_codec::metadata_from_wire),
        ))
    }

    /// Whether the episode ended (or none started yet).
    pub fn done(&self) -> bool {
        self.done
    }

    /// The env's contract.
    pub fn contract(&self) -> &EnvContract {
        self.contract
            .as_ref()
            .expect("a binding connects at creation")
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
        self.done = true;
        self.context.close();
        let mut result = self.drive(inner.close());
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
        result.and(closed)
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
