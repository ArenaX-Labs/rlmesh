//! Running a model against an env: runs, sessions, hooks and results.
//!
//! Every call is sync and blocks; each has an `_async` twin. A run borrows
//! the model and the env: it never closes the model, and closes the env only
//! on `close_env`.

pub(crate) mod drive;
pub(crate) mod hooks;
mod read;
pub use read::{FrameSources, ReadFn, ReadItem, Reader, RenderFn};
mod lent;
mod native;
mod options;
mod results;
pub mod runtime;
mod session;
mod targets;

#[cfg(test)]
mod tests;

pub use hooks::{RunContext, RunHooks};
pub use options::{RunOptions, SessionOptions, SessionRunOptions};
pub use results::{EpisodeResult, RunResult, StepEvent, TelemetryRow};
pub use session::Session;
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

    /// Bind this model to `env` for driving by hand. Nothing dials until
    /// first use; closing the session never closes the model.
    pub fn session<'a>(
        &'a self,
        env_or_address: impl Into<EnvTarget<'a>>,
        options: SessionOptions,
    ) -> Result<Session<'a>> {
        Session::create(Some(self), env_or_address.into(), options)
    }

    /// The async twin of [`session`](Model::session) (nothing dials at
    /// creation, so it never waits).
    pub async fn session_async<'a>(
        &'a self,
        env_or_address: impl Into<EnvTarget<'a>>,
        options: SessionOptions,
    ) -> Result<Session<'a>> {
        self.session(env_or_address, options)
    }
}

/// Drive `model` against `env` to completion. A [`Model`] runs on the native
/// loop ([`Model::run`]); a [`RemoteModel`](crate::RemoteModel) and
/// [`RANDOM_SAMPLE`] run through a session it opens and closes.
pub fn run<'a>(
    model: impl Into<ModelTarget<'a>>,
    env: impl Into<EnvTarget<'a>>,
    options: RunOptions<'_>,
) -> Result<RunResult> {
    match model.into().0 {
        ModelKind::Model(model) => model.run(env, options),
        #[cfg(feature = "lockstep")]
        ModelKind::Remote(model) if crate::address::is_v2(model.address()) => {
            drive::refuse_in_async_context("run")?;
            drive::run_sync(native::run_model(
                native::Driven::Served(model),
                env.into(),
                options,
                Mode::Sync,
            ))
        }
        ModelKind::Remote(model) => {
            drive::refuse_in_async_context("run")?;
            drive::run_sync(run_session_loop(
                Some(model),
                env.into(),
                options,
                Mode::Sync,
            ))
        }
        ModelKind::Random => {
            drive::refuse_in_async_context("run")?;
            drive::run_sync(run_session_loop(None, env.into(), options, Mode::Sync))
        }
    }
}

/// The async twin of [`run`].
pub async fn run_async<'a>(
    model: impl Into<ModelTarget<'a>>,
    env: impl Into<EnvTarget<'a>>,
    options: RunOptions<'_>,
) -> Result<RunResult> {
    match model.into().0 {
        ModelKind::Model(model) => model.run_async(env, options).await,
        #[cfg(feature = "lockstep")]
        ModelKind::Remote(model) if crate::address::is_v2(model.address()) => {
            native::run_model(
                native::Driven::Served(model),
                env.into(),
                options,
                Mode::Async,
            )
            .await
        }
        ModelKind::Remote(model) => {
            run_session_loop(Some(model), env.into(), options, Mode::Async).await
        }
        ModelKind::Random => run_session_loop(None, env.into(), options, Mode::Async).await,
    }
}

/// Run on the session loop: validate, open, run, close.
async fn run_session_loop<'a>(
    remote: Option<&'a crate::RemoteModel>,
    env: EnvTarget<'a>,
    options: RunOptions<'_>,
    mode: Mode,
) -> Result<RunResult> {
    let RunOptions {
        episodes,
        seeds,
        max_episode_steps,
        max_episode,
        hooks,
        close_env,
        trust_entrypoints,
        instruction,
        view,
        execution_horizon,
        prefetch_lead,
        workflow_edition,
        trial_index_base,
        cancel,
    } = options;
    if prefetch_lead != 0 {
        return Err(Error::invalid_argument(format!(
            "prefetch_lead={prefetch_lead} drives the native runtime loop; a served model or \
             RANDOM_SAMPLE runs through the session loop, which does not prefetch. Remove \
             prefetch_lead or run a local Model."
        )));
    }
    // RUN-P6: what the options alone decide, before the session dials.
    options::episode_budget(episodes, seeds.as_deref())?;
    options::check_caps(max_episode_steps, max_episode)?;
    let session_options = SessionOptions {
        close_env,
        trust_entrypoints,
        instruction,
        view,
        execution_horizon,
        workflow_edition,
        cancel: None,
    };
    let mut session = match remote {
        Some(model) => match mode {
            Mode::Sync => model.session(env, session_options)?,
            Mode::Async => model.session_async(env, session_options).await?,
        },
        None => Session::create(None, env, session_options)?,
    };
    let run_options = SessionRunOptions {
        episodes,
        seeds,
        max_episode_steps,
        max_episode,
        hooks,
        trial_index_base,
        cancel,
    };
    let result = match mode {
        Mode::Sync => session.run(run_options),
        Mode::Async => session.run_async(run_options).await,
    };
    match result {
        // A failed run drops the session: `__exit__` on an exception.
        Err(err) => Err(err),
        Ok(result) => {
            match mode {
                Mode::Sync => session.close()?,
                Mode::Async => session.close_async().await?,
            }
            Ok(result)
        }
    }
}

/// Bind `model` to `env`.
pub fn session<'a>(
    model: impl Into<ModelTarget<'a>>,
    env: impl Into<EnvTarget<'a>>,
    options: SessionOptions,
) -> Result<Session<'a>> {
    match model.into().0 {
        ModelKind::Model(model) => model.session(env, options),
        ModelKind::Remote(model) => model.session(env, options),
        ModelKind::Random => Session::create(None, env.into(), options),
    }
}

/// The async twin of [`session`].
pub async fn session_async<'a>(
    model: impl Into<ModelTarget<'a>>,
    env: impl Into<EnvTarget<'a>>,
    options: SessionOptions,
) -> Result<Session<'a>> {
    match model.into().0 {
        ModelKind::Remote(model) => model.session_async(env, options).await,
        ModelKind::Model(model) => model.session(env, options),
        ModelKind::Random => Session::create(None, env.into(), options),
    }
}
