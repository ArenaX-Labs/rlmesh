//! The model engine plumbing the in-repo bindings drive: the async
//! `ModelHandler` seam, the `PredictFn` holes of the adapter engine, the
//! worker that runs a handler, and the facade `Model`'s erased
//! construction for runtime corner declarations (R-25).

use std::sync::Arc;

#[cfg(feature = "lockstep")]
pub use crate::model::bind::{Binding, BoundStep};
pub use crate::model::engine::AdaptedModelHandler;
pub use crate::model::handler::{
    HeldState, HistoryNeeds, ModelHandler, ModelResetOptions, ModelRouteSetup, PredictFrames,
    ResolveOptions, RouteNeeds,
};
pub use crate::model::local::EnvClientRuntimeEnv;
#[cfg(feature = "lockstep")]
pub use crate::model::local::LockstepEnv;
pub use crate::model::predict_fn::{PredictFn, RouteConfig, RouteResolver};
pub use crate::model::random::RandomSampleHandler;
pub use crate::model::types::{EpisodeInfo, HistoryFrame, ModelObservation, ModelRouteContext};
pub use crate::model::worker::{ModelWorker, RunLocalOptions};
pub use crate::model::{DynPolicy, PolicyDecl};

use crate::{Corners, Error, Kwargs, Model, ModelOptions, Result};

/// The engine handler that runs `model`: what a binding serves or drives.
pub fn model_handler(model: &Model) -> AdaptedModelHandler {
    crate::model::bridge::engine_handler(Arc::clone(model.shared()))
}

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

/// A model served elsewhere, reached by its participant address
/// (`"lab:7447/openpi"`), as Python's `Model(address)` takes it: a v1
/// endpoint or an address naming no model is the ValueError class.
#[cfg(feature = "lockstep")]
pub fn served_model(address: &str) -> Result<crate::RemoteModel> {
    if !crate::address::is_v2(address) {
        return Err(Error::invalid_argument(format!(
            "Model({address:?}): a served model is reached by its participant address (e.g. \
             'lab:7447/openpi'); 0.2 no longer dials a v1 model endpoint"
        )));
    }
    let parsed = rlmesh_session::address::Address::participant(address)
        .map_err(|err| Error::invalid_argument(err.to_string()))?;
    if parsed.participant.is_none() {
        return Err(Error::invalid_argument(format!(
            "Model({address:?}) names no model: end the address with the model's name, e.g. \
             'lab:7447/openpi'"
        )));
    }
    crate::RemoteModel::new(crate::RemoteModelOptions::default().address(address))
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

/// An already-loaded language handler entering the shared facade orchestration.
/// Bindings retain their language-specific predict/route conversion, while run
/// and bind own validation, hooks, episode tracking and resource lifetimes.
pub struct PreparedModel {
    pub handler: Box<dyn ModelHandler>,
    pub presentation: Option<Box<dyn Presentation>>,
    pub run_gate: Arc<std::sync::atomic::AtomicBool>,
    pub run_lease: Option<Arc<RunLease>>,
    pub native_chunk: Option<u32>,
    pub execution_horizon: Option<u32>,
    pub workflow_edition: Option<String>,
    pub authored: bool,
    pub has_chunk: Option<bool>,
}

/// Drive an already-loaded binding handler through the same facade as Model.
pub fn run_prepared(
    model: PreparedModel,
    env: crate::EnvTarget<'_>,
    options: crate::RunOptions<'_>,
) -> Result<crate::RunResult> {
    crate::run::run_prepared(model, env, options)
}

/// Bind an already-loaded language handler through the shared facade.
#[cfg(feature = "lockstep")]
pub fn bind_prepared<'a>(
    model: PreparedModel,
    env: crate::EnvTarget<'a>,
    options: crate::BindOptions<'a>,
) -> Result<crate::Binding<'a>> {
    crate::run::bind_prepared(model, env, options)
}

/// Language resource conversion for targets the facade cannot construct itself
/// (for example a Python env server or a lent Docker session).
pub trait EnvAdapter: Send {
    fn open(&mut self, edition: Option<&str>, caller: CallerJobs) -> Result<AdaptedEnv>;
    fn release(&mut self, close_env: bool) -> Result<()>;
    fn owns_env(&self) -> bool;
}

pub struct AdaptedEnv {
    pub address: String,
    pub invite_at: Option<String>,
    pub close_over_wire: bool,
}

/// An owned target conversion; the core, rather than the host, decides when
/// it is opened and released, including zero budgets and connection failures.
pub fn adapted_env(adapter: Box<dyn EnvAdapter>) -> crate::EnvTarget<'static> {
    crate::run::adapted_env(adapter)
}

pub(crate) struct OwnedTarget {
    adapter: Box<dyn EnvAdapter>,
    close: bool,
    released: bool,
}
impl OwnedTarget {
    pub(crate) fn new(adapter: Box<dyn EnvAdapter>, close: bool) -> Self {
        let close = close || adapter.owns_env();
        Self {
            adapter,
            close,
            released: false,
        }
    }
    pub(crate) fn open(&mut self, edition: Option<&str>, caller: CallerJobs) -> Result<AdaptedEnv> {
        self.adapter.open(edition, caller)
    }
    pub(crate) fn release(&mut self) -> Result<()> {
        if std::mem::replace(&mut self.released, true) {
            return Ok(());
        }
        self.adapter.release(self.close)
    }
}
impl Drop for OwnedTarget {
    fn drop(&mut self) {
        let _ = self.release();
    }
}

/// Poll language signals on the caller between facade driver segments. The
/// binding decides how a signal maps to its cancellation token and exception.
pub fn with_caller_poll<R>(poll: Arc<dyn Fn() + Send + Sync>, call: impl FnOnce() -> R) -> R {
    crate::run::drive::with_caller_poll(poll, call)
}

/// Presentation frames at the facade's binding seam.
pub use crate::run::RunFrame;

/// Preserve tensor ownership and scalar/metadata types when crossing an
/// existing language adapter's neutral-value codec.
pub fn adapter_value(value: &crate::Value) -> Result<rlmesh_adapters::v1::Value> {
    crate::model::bridge::to_adapter(value)
}

pub use crate::model::facade::RunSlot as RunLease;

/// Retain run admission through a language binding's exception conversion.
/// A nested/parallel run fails before it can replace the active run's errors.
pub fn claim_run(gate: Arc<std::sync::atomic::AtomicBool>) -> Result<Arc<RunLease>> {
    RunLease::claim(gate).map(Arc::new)
}

pub(crate) fn caller_port(port: crate::run::drive::CallerPort) -> CallerJobs {
    CallerJobs(port)
}

/// Attach an owned host observer to a manual binding.
#[cfg(feature = "lockstep")]
pub fn observe_binding(
    binding: &mut crate::Binding<'_>,
    hooks: Box<dyn crate::RunHooks>,
    presentation: Box<dyn Presentation>,
) -> Result<()> {
    binding.observe(hooks, presentation)
}

/// The next manual operation's signal cancellation.
#[cfg(feature = "lockstep")]
pub fn binding_cancellation(
    binding: &mut crate::Binding<'_>,
) -> tokio_util::sync::CancellationToken {
    binding.cancellation()
}

/// Read episode status from callbacks without borrowing the mutable binding.
#[cfg(feature = "lockstep")]
pub fn binding_done_flag(binding: &crate::Binding<'_>) -> Arc<std::sync::atomic::AtomicBool> {
    binding.done_flag()
}

/// A render callback usable while a manual binding is dispatching a hook.
#[cfg(feature = "lockstep")]
pub fn binding_renderer(
    binding: &crate::Binding<'_>,
) -> Result<Arc<dyn Fn() -> Result<crate::Value> + Send + Sync>> {
    binding.renderer()
}

/// A language's drawing observer, paired with its ordinary RunHooks. The
/// shared core supplies accounting and applies its truncation/quit requests.
/// Kept in the semver-exempt binding namespace, not the public hooks contract.
pub trait Presentation: Send {
    fn on_frame(&mut self, frame: &RunFrame<'_>) -> Result<()>;
    fn directive(&mut self) -> Result<crate::run::runtime::RunDirective>;
}
