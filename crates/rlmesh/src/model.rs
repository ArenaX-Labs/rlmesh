//! Model-side API: the [`ModelHandler`] trait, the [`ModelWorker`] that drives
//! or serves it, and the observation/route/lifecycle types a handler receives.

mod handler;
mod lifecycle;
mod local;
mod remote;
mod server;
mod types;
mod wire;
mod worker;

pub use handler::{ModelHandler, ModelRouteSetup};
pub use local::{EnvClientRuntimeEnv, ModelHandlerRuntimeModel};
pub use remote::RemoteModel;
pub use server::BoundModelServer;
pub use types::{
    ModelEpisodeEnd, ModelLaneReset, ModelObservation, ModelRouteContext, ModelRouteSlot,
};
pub use worker::{ModelWorker, RunLocalOptions, ServeModelOptions};

#[cfg(test)]
mod tests;
