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

use crate::Result;
use crate::spaces::{BinaryPayload, SpaceSpec, SpaceValue};

/// Encode an action [`SpaceValue`] into the [`BinaryPayload`] a
/// [`ModelHandler::predict`] returns.
///
/// [`predict`](ModelHandler::predict) hands back *encoded* action bytes, so a
/// handler must turn the [`SpaceValue`] its policy chose into the same wire
/// encoding rlmesh uses on the env side. This is that bridge: pass the value and
/// the action [`SpaceSpec`] (from
/// [`Env::action_space`](crate::Env::action_space), reachable on the handler
/// side via
/// [`ModelObservation::env_contract`](crate::ModelObservation::env_contract)),
/// and get back the payload — no need to hunt through [`crate::spaces`] for the
/// per-space-kind codec.
///
/// Returns [`Error`](crate::Error) if `value` does not match the kind/dtype of
/// `space`.
///
/// # Example
///
/// ```
/// use rlmesh::prelude::*;
/// use rlmesh::spaces::DType;
/// use rlmesh::spaces::types::{DiscreteSpec, SpaceKind};
///
/// // A `Discrete(4)` action space and the action "pick option 2".
/// let space = SpaceSpec {
///     spec: Some(SpaceKind::Discrete(DiscreteSpec { n: 4, start: 0 })),
///     dtype: DType::Int64,
///     ..SpaceSpec::default()
/// };
/// let action = SpaceValue::Discrete(2);
///
/// let payload: BinaryPayload = rlmesh::encode_action(&action, &space)?;
/// assert!(!payload.data.is_empty());
/// # Ok::<(), rlmesh::Error>(())
/// ```
pub fn encode_action(value: &SpaceValue, space: &SpaceSpec) -> Result<BinaryPayload> {
    let data = rlmesh_grpc::wire::encode_space_value_bytes(value, space)?;
    Ok(BinaryPayload { data })
}

#[cfg(test)]
mod tests;
