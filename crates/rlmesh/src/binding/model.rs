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
pub use crate::model::local::{EnvClientRuntimeEnv, ModelHandlerRuntimeModel};
pub use crate::model::predict_fn::{PredictFn, RouteConfig, RouteResolver};
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

/// Drive `model` in-process against a remote env until the run's episode
/// budget is spent: the binding's run path until `run` lands. The run
/// borrows the model and never closes it.
pub async fn run_model_local_async(
    model: &Model,
    options: RunLocalOptions,
    cancellation: tokio_util::sync::CancellationToken,
    hooks: Arc<dyn rlmesh_runtime::RuntimeHooks>,
) -> Result<rlmesh_runtime::RuntimeReport> {
    crate::model::run_local(model, options, cancellation, hooks).await
}

/// The blocking form of [`run_model_local_async`], on a private runtime.
pub fn run_model_local(
    model: &Model,
    options: RunLocalOptions,
) -> Result<rlmesh_runtime::RuntimeReport> {
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
        Arc::new(rlmesh_runtime::NoopRuntimeHooks),
    ))
}
