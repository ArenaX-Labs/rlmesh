//! Model-side API: the [`ModelHandler`] trait, the [`ModelWorker`] that drives
//! or serves it, and the observation/route/lifecycle types a handler receives.

mod handler;
mod lifecycle;
mod local;
mod server;
mod types;
mod wire;
mod worker;

pub use handler::ModelHandler;
pub use local::{EnvClientRuntimeEnv, ModelHandlerRuntimeModel};
pub use server::BoundModelServer;
pub use types::{ModelEpisodeEnd, ModelObservation, ModelRouteContext, ModelRouteSlot};
pub use worker::{ModelWorker, RunLocalOptions, ServeModelOptions};

#[cfg(test)]
mod tests;
