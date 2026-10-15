//! Running a model against an env: Python's `rlmesh.run`, `rlmesh.session`,
//! `Model.run`, `Model.session`, `Session`, the run hooks and the results.
//!
//! Every call is sync and blocks, as in Python; each has an `_async` twin
//! (R-16). A run borrows the model and the env (R-17): it never closes the
//! model, and closes the env only on `close_env`.

pub(crate) mod drive;
mod hooks;
mod lent;
mod native;
mod options;
mod results;
mod session;
mod targets;

#[cfg(test)]
mod tests;

pub use hooks::{RunContext, RunHooks};
pub use options::{RunOptions, SessionOptions, SessionRunOptions};
pub use results::{EpisodeResult, RunResult, StepEvent, TelemetryRow};
pub use session::Session;
pub use targets::{EnvTarget, ModelTarget, RANDOM_SAMPLE, RandomSample};

use native::Mode;
use targets::ModelKind;

use crate::{Env, Error, Model, Result, VectorEnv};

/// A local env behind a run's loopback server.
pub(crate) enum LoopbackEnv {
    Env(Box<dyn Env + Send>),
    Vector(Box<dyn VectorEnv + Send>),
}

impl Model {
    /// Drive this model against `env` on the native runtime loop and return
    /// the result: Python's `Model.run`. The run borrows the model (it stays
    /// open) and the env (closed only on `close_env`); hooks run on this
    /// thread (RUN-8). One run at a time per model (RUN-9).
    pub fn run<'e>(
        &self,
        env: impl Into<EnvTarget<'e>>,
        options: RunOptions<'_>,
    ) -> Result<RunResult> {
        drive::refuse_in_async_context("Model::run")?;
        drive::run_sync(native::run_model(self, env.into(), options, Mode::Sync))
    }

    /// The async twin of [`run`](Model::run): hooks run inside this future,
    /// where a blocking rlmesh call fails with the `RuntimeError` class.
    /// Dropping it cancels the run and fires no hook (R-18).
    pub async fn run_async<'e>(
        &self,
        env: impl Into<EnvTarget<'e>>,
        options: RunOptions<'_>,
    ) -> Result<RunResult> {
        native::run_model(self, env.into(), options, Mode::Async).await
    }

    /// Bind this model to `env` for driving by hand: Python's
    /// `Model.session`. Nothing dials until first use; closing the session
    /// never closes the model.
    pub fn session<'a>(
        &'a self,
        env: impl Into<EnvTarget<'a>>,
        options: SessionOptions,
    ) -> Result<Session<'a>> {
        Session::create(Some(self), env.into(), options)
    }

    /// The async twin of [`session`](Model::session) (nothing dials at
    /// creation, so it never waits).
    pub async fn session_async<'a>(
        &'a self,
        env: impl Into<EnvTarget<'a>>,
        options: SessionOptions,
    ) -> Result<Session<'a>> {
        self.session(env, options)
    }
}

/// Drive `model` against `env` to completion: Python's `rlmesh.run`. A
/// [`Model`] runs on the native loop ([`Model::run`]); a
/// [`RemoteModel`](crate::RemoteModel) and [`RANDOM_SAMPLE`] run through a
/// session it opens and closes.
pub fn run<'a>(
    model: impl Into<ModelTarget<'a>>,
    env: impl Into<EnvTarget<'a>>,
    options: RunOptions<'_>,
) -> Result<RunResult> {
    match model.into().0 {
        ModelKind::Model(model) => model.run(env, options),
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
        ModelKind::Remote(model) => {
            run_session_loop(Some(model), env.into(), options, Mode::Async).await
        }
        ModelKind::Random => run_session_loop(None, env.into(), options, Mode::Async).await,
    }
}

/// `rlmesh.run` on the session loop: validate, open, run, close.
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

/// Bind `model` to `env`: Python's `rlmesh.session`.
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
