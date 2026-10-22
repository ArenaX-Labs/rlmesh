//! The model engine plumbing the in-repo bindings drive: the async
//! `ModelHandler` seam, the `PredictFn` holes of the adapter engine, the
//! worker that runs or serves a handler, and the facade `Model`'s erased
//! construction for runtime corner declarations (R-25).

use std::sync::Arc;

pub use crate::model::engine::AdaptedModelHandler;
pub use crate::model::handler::{
    HeldState, HistoryNeeds, ModelHandler, ModelRouteSetup, PredictFrames, ResolveOptions,
    RouteNeeds,
};
#[cfg(feature = "lockstep")]
pub use crate::model::local::LockstepRuntimeEnv;
pub use crate::model::local::{EnvClientRuntimeEnv, ModelHandlerRuntimeModel};
pub use crate::model::predict_fn::{PredictFn, RouteConfig, RouteResolver};
pub use crate::model::remote::{ModelClient, ModelConnect};
pub use crate::model::server::BoundModelServer;
pub use crate::model::types::{EpisodeInfo, HistoryFrame, ModelObservation, ModelRouteContext};
pub use crate::model::worker::{ModelWorker, RunLocalOptions, ServeModelOptions};
pub use crate::model::{DynPolicy, PolicyDecl};

use crate::{Corners, Error, Kwargs, Model, ModelOptions, Result};

/// The [`Corners`] for C's `RLMESH_CORNER_*` bits; a bit outside the four
/// corners is the TypeError class.
pub fn corners_from_bits(bits: u32) -> Result<Corners> {
    if bits & !0b1111 != 0 {
        return Err(Error::Type(format!(
            "corner bits {bits:#x} hold bits outside the four RLMESH_CORNER_* values"
        )));
    }
    Ok(Corners::from_bits(bits))
}

/// A [`Model`] over a binding's policy with runtime declarations, built in
/// the native construction order (declarations, config keys, `load`, the
/// loaded native chunk). `config` is `Some` for `from_config`.
pub fn new_model(
    policy: Box<dyn DynPolicy>,
    decl: PolicyDecl,
    options: ModelOptions,
    config: Option<Kwargs>,
) -> Result<Model> {
    crate::model::construct(policy, decl, options, config)
}

/// Runs closures on the thread blocked in [`block_on_caller`]: a binding's
/// hooks and signal checks on the thread that called a sync run (RUN-8).
#[derive(Clone)]
pub struct CallerJobs(crate::run::drive::CallerPort);

/// The caller's end of [`CallerJobs`].
pub struct CallerQueue(crate::run::drive::CallerQueue);

/// A job port and its queue.
pub fn caller_jobs() -> (CallerJobs, CallerQueue) {
    let (port, queue) = crate::run::drive::caller_channel();
    (CallerJobs(port), CallerQueue(queue))
}

impl CallerJobs {
    /// Run `f` on the caller's thread and await its result; fails with the
    /// `RuntimeError` class once the caller has stopped driving.
    pub async fn run<R: Send + 'static>(
        &self,
        f: impl FnOnce() -> R + Send + 'static,
    ) -> Result<R> {
        self.0.call(move |_side| Ok(f())).await
    }
}

/// Poll `future` on this thread in segments (inside the library runtime's
/// context), running each job between segments outside any async execution
/// context: the facade's sync driver.
pub fn block_on_caller<F: std::future::Future>(future: F, queue: &mut CallerQueue) -> F::Output {
    let mut side = crate::run::drive::CallerSide {
        env: None,
        hooks: None,
    };
    crate::run::drive::block_on_caller(future, &mut queue.0, &mut side)
}

/// Drive `model` in-process against a remote env until the run's episode
/// budget is spent: the binding's run path until `run` lands. The run
/// borrows the model and never closes it.
pub async fn run_model_local_async(
    model: &Model,
    options: RunLocalOptions,
    cancellation: tokio_util::sync::CancellationToken,
    hooks: Arc<dyn crate::run::runtime::RuntimeHooks>,
) -> Result<crate::run::runtime::RuntimeReport> {
    crate::model::run_local(model, options, cancellation, hooks).await
}

/// The blocking form of [`run_model_local_async`], on a private runtime.
pub fn run_model_local(
    model: &Model,
    options: RunLocalOptions,
) -> Result<crate::run::runtime::RuntimeReport> {
    if crate::env::server::in_async_context() {
        return Err(Error::Runtime(
            "run_model_local() blocks; call run_model_local_async() from async code".to_string(),
        ));
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|err| Error::Internal(format!("failed to create tokio runtime: {err}")))?;
    runtime.block_on(run_model_local_async(
        model,
        options,
        tokio_util::sync::CancellationToken::new(),
        Arc::new(crate::run::runtime::NoopRuntimeHooks),
    ))
}

/// Serve a binding's model with explicit control over process signal handlers.
pub fn serve_model(
    model: &Model,
    address: &str,
    options: crate::ModelServeOptions,
    foreground: bool,
) -> Result<()> {
    model.serve_host(address, options, foreground)
}
