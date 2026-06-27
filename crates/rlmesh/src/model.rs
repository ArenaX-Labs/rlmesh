//! Model-side API: the [`ModelHandler`] trait, the [`ModelWorker`] that drives
//! or serves it, and the observation/route/lifecycle types a handler receives.

mod engine;
mod handler;
mod local;
mod predict_fn;
mod remote;
mod server;
mod types;
mod wire;
mod worker;

pub use engine::AdaptedModelHandler;
pub use handler::{ModelHandler, ModelRouteSetup};
pub use local::{EnvClientRuntimeEnv, ModelHandlerRuntimeModel};
pub use predict_fn::{PredictFn, RouteConfig, RouteResolver};
pub use remote::RemoteModel;
pub use server::BoundModelServer;
pub use types::{ModelObservation, ModelRouteContext};
pub use worker::{ModelWorker, RunLocalOptions, ServeModelOptions};

#[cfg(test)]
mod tests;
