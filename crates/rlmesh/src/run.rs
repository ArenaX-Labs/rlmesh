//! Running a model against an env: runs, bindings, hooks and results.
//!
//! Every call is sync and blocks; a run has an `_async` twin. A run or a
//! binding borrows the model and the env: it never closes the model, and
//! closes the env only on `close_env` (or one it built from a factory).

pub(crate) mod drive;
pub(crate) mod hooks;
mod read;
pub use read::{FrameSources, ReadFn, ReadItem, Reader, RenderFn};
#[cfg(feature = "lockstep")]
mod bind;
mod lent;
mod native;
mod options;
pub(crate) mod plan;
pub(crate) mod report;
mod results;
pub(crate) mod routes;
pub mod runtime;
mod targets;

#[cfg(test)]
mod tests;

#[cfg(feature = "lockstep")]
pub use bind::Binding;
pub use hooks::{RunContext, RunFrame, RunHooks};
pub(crate) use options::check_horizon;
pub use options::{BindOptions, RunOptions, resolve_execution_horizon};
pub use results::{EpisodeResult, RunResult, RunResultSaveOptions, StepEvent, TelemetryRow};
pub use targets::{EnvTarget, ModelTarget, RANDOM_SAMPLE, RandomSample};

pub(crate) use native::Mode;
use targets::ModelKind;

use crate::{Env, Error, Model, Result, VectorEnv};

/// A local env behind a run's loopback server.
pub(crate) enum LoopbackEnv {
    Env(Box<dyn Env + Send>),
    Vector(Box<dyn VectorEnv + Send>),
}

impl Model {
    /// Drive this model against `env` on the native runtime loop and return
    /// the result. The run borrows the model (it stays open) and the env
    /// (closed only on `close_env`); hooks run on this thread. One run at a
    /// time per model.
    pub fn run<'e>(
        &self,
        env_or_address: impl Into<EnvTarget<'e>>,
        options: RunOptions<'_>,
    ) -> Result<RunResult> {
        drive::refuse_in_async_context("Model::run")?;
        drive::run_sync(native::run_model(
            native::Driven::Local(self),
            env_or_address.into(),
            options,
            Mode::Sync,
        ))
    }

    /// The async twin of [`run`](Model::run): hooks run inside this future,
    /// where a blocking rlmesh call fails with the `RuntimeError` class.
    /// Dropping it cancels the run and fires no hook.
    pub async fn run_async<'e>(
        &self,
        env_or_address: impl Into<EnvTarget<'e>>,
        options: RunOptions<'_>,
    ) -> Result<RunResult> {
        native::run_model(
            native::Driven::Local(self),
            env_or_address.into(),
            options,
            Mode::Async,
        )
        .await
    }

    /// Bind this model to `env` and step it by hand, as Python's
    /// `Model.bind`: connects now, serving a local env in this process for
    /// the binding's life. Closing the binding never closes the model.
    #[cfg(feature = "lockstep")]
    pub fn bind<'a>(
        &'a self,
        target: impl Into<EnvTarget<'a>>,
        options: BindOptions<'a>,
    ) -> Result<Binding<'a>> {
        Binding::create(self.into(), target.into(), options)
    }
}

/// Drive `model` against `env` to completion on the native loop: a
/// [`Model`], a model served on a session ([`RemoteModel`](crate::RemoteModel)
/// at its participant address) or [`RANDOM_SAMPLE`].
pub fn run<'a>(
    model: impl Into<ModelTarget<'a>>,
    env: impl Into<EnvTarget<'a>>,
    options: RunOptions<'_>,
) -> Result<RunResult> {
    let driven = driven(model.into())?;
    drive::refuse_in_async_context("run")?;
    drive::run_sync(native::run_model(driven, env.into(), options, Mode::Sync))
}

/// The async twin of [`run`].
pub async fn run_async<'a>(
    model: impl Into<ModelTarget<'a>>,
    env: impl Into<EnvTarget<'a>>,
    options: RunOptions<'_>,
) -> Result<RunResult> {
    native::run_model(driven(model.into())?, env.into(), options, Mode::Async).await
}

fn driven(model: ModelTarget<'_>) -> Result<native::Driven<'_>> {
    Ok(match model.0 {
        ModelKind::Model(model) => native::Driven::Local(model),
        #[cfg(feature = "lockstep")]
        ModelKind::Remote(model) if crate::address::is_v2(model.address()) => {
            native::Driven::Served(model)
        }
        ModelKind::Remote(model) => {
            return Err(Error::invalid_argument(format!(
                "run: {:?} is a v1 model endpoint, which rlmesh 0.2 no longer dials; reach a \
                 served model by its participant address (e.g. 'lab:7447/openpi')",
                model.address()
            )));
        }
        ModelKind::Random => native::Driven::Random,
        ModelKind::Prepared(model) => native::Driven::Prepared(model),
    })
}

/// Bind `model` to `env` (see [`Model::bind`]).
#[cfg(feature = "lockstep")]
pub fn bind<'a>(
    model: impl Into<ModelTarget<'a>>,
    env: impl Into<EnvTarget<'a>>,
    options: BindOptions<'a>,
) -> Result<Binding<'a>> {
    Binding::create(model.into(), env.into(), options)
}

// Host handlers and resources enter the same private implementation as the
// public facade, without exposing its modules as alternate API paths.
pub(crate) fn run_prepared(
    model: crate::binding::model::PreparedModel,
    env: EnvTarget<'_>,
    options: RunOptions<'_>,
) -> Result<RunResult> {
    drive::refuse_in_async_context("run")?;
    drive::run_sync(native::run_model(
        native::Driven::Prepared(model),
        env,
        options,
        Mode::Sync,
    ))
}

#[cfg(feature = "lockstep")]
pub(crate) fn bind_prepared<'a>(
    model: crate::binding::model::PreparedModel,
    env: EnvTarget<'a>,
    options: BindOptions<'a>,
) -> Result<Binding<'a>> {
    Binding::create(ModelTarget(ModelKind::Prepared(model)), env, options)
}

pub(crate) fn adapted_env(
    adapter: Box<dyn crate::binding::model::EnvAdapter>,
) -> EnvTarget<'static> {
    EnvTarget(targets::EnvKind::Adapted(adapter))
}
