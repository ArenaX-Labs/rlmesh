//! Model authoring and serving: the [`Policy`] trait, the concrete
//! [`Model`] handle, the per-episode context, and the engine plumbing the
//! bindings drive (re-exported through `binding`).

pub(crate) mod bridge;
mod context;
pub(crate) mod engine;
pub(crate) mod facade;
pub(crate) mod handler;
pub(crate) mod local;
mod policy;
pub(crate) mod predict_fn;
pub(crate) mod remote;
pub(crate) mod server;
pub(crate) mod types;
pub(crate) mod wire;
pub(crate) mod worker;

pub use context::{BatchPredictContext, EpisodeState, PredictContext};
pub use facade::{DynPolicy, Model, ModelOptions, ModelServeOptions, PolicyDecl};
pub(crate) use facade::{construct, run_local};
pub use policy::{Corners, LoadKey, NO_ADAPTER, Policy, SpecArg, WithContext, with_context};
pub use remote::{RemoteModel, RemoteModelOptions};
pub use types::predict_seed;

// The engine plumbing, public through `binding` (R-25); named here for the
// crate's own tests.
#[allow(unused_imports)]
pub(crate) use engine::AdaptedModelHandler;
#[allow(unused_imports)]
pub(crate) use handler::{
    HeldState, HistoryNeeds, ModelHandler, ModelResetOptions, ModelRouteSetup, PredictFrames,
    ResolveOptions, RouteNeeds,
};
#[allow(unused_imports)]
pub(crate) use predict_fn::{PredictFn, RouteConfig, RouteResolver};
#[allow(unused_imports)]
pub(crate) use server::BoundModelServer;
#[allow(unused_imports)]
pub(crate) use types::{EpisodeInfo, HistoryFrame, ModelObservation, ModelRouteContext};
#[allow(unused_imports)]
pub(crate) use worker::{ModelWorker, RunLocalOptions, ServeModelOptions};

#[cfg(test)]
mod facade_tests;
#[cfg(test)]
mod tests;
