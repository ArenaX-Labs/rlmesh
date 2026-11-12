//! A model served on a session: one request, one reply, on `m/{model}`.
//!
//! ```text
//! driver ──ModelRequest::Resolve{route, contract}──▶ model   (once per env)
//! driver ──ModelRequest::Predict{groups}──────────▶ model   (per decision)
//! driver ◀──ModelReply::Predicted{outcomes}──────── model
//! ```
//!
//! The model side is a [`ModelHandler`] (the engine a local run drives),
//! answered by [`ModelService`]: routes resolve off the predict lock, every
//! forward runs under it, and each group of a predict gets its own outcome.
//! [`ModelBinding::serve`] joins a session and answers on the lockstep
//! class session, as a lockstep env does. The driver side is a
//! [`ModelLink`].
//!
//! A driver invites the model into its own session (`ModelRequest::Join`,
//! sent in the model's): the model joins it under its name and answers
//! there too, so predicts stay in the session the driver runs. It leaves
//! when the driver says so, or when it is the session's last participant.
//!
//! A probe (the selector parameter `probe`) is answered with an empty
//! reply, so a driver knows the route to the model is up.
//!
//! A model that fuses coalesces predicts from every session it answers in
//! into one handler call ([`BatchOptions`]): a batch assembles while the
//! forward before it runs and goes to the handler as that one ends.
//!
//! A served model serves, then closes once: the first
//! [`ModelService::close`] refuses new requests and starts the one closer,
//! which waits for every accepted request to land (each runs to its end
//! holding a flight, even once its caller stops waiting), then releases
//! the routes and runs `on_close`. Every close waits for that closer, up
//! to its deadline.

use std::collections::HashMap;
use std::sync::Arc;

use rlmesh_session::wire::ModelFault;
use rlmesh_spaces::{EnvContract, SpaceSpec};
use tokio::sync::{Mutex, watch};

use crate::Error;
use crate::lockstep::error::GymError;
use crate::model::handler::{ModelResetOptions, ModelRouteSetup};
#[cfg(doc)]
use crate::{model::handler::ModelHandler, serve_options::BatchOptions};

mod batcher;
mod closer;
mod joins;
mod link;
mod served;
mod service;

#[cfg(all(test, target_os = "linux"))]
mod reach_tests;

pub use link::{DEFAULT_TIMEOUT, ModelBinding, ModelConnection, ModelLink};
pub use served::ServedModel;

use batcher::Batching;
use closer::Flights;

/// The selector parameter of a probe.
const PROBE: &str = "probe";

/// A fault with `code`.
fn fault(code: &str, message: impl Into<String>, recoverable: bool) -> ModelFault {
    ModelFault {
        code: code.into(),
        message: message.into(),
        recoverable,
    }
}

/// A handler's error as a fault: a model error's own message and
/// recoverability, anything else as an internal failure.
fn fault_of(error: &Error) -> ModelFault {
    match error {
        Error::Attached(attached) => {
            let mut wire = fault_of(attached.error());
            if let Some(message) = attached.authored_message() {
                wire.message = message.to_owned();
            }
            wire
        }
        Error::Model(model) => fault("INTERNAL", model.message.clone(), model.is_recoverable),
        Error::InvalidArgument(invalid) if invalid.adapter_resolution.is_some() => {
            fault("ADAPTER_RESOLUTION", invalid.message.clone(), false)
        }
        Error::InvalidArgument(invalid) => fault("INVALID_REQUEST", invalid.message.clone(), false),
        Error::Environment(env) => fault("INTERNAL", env.message.clone(), false),
        other => fault("INTERNAL", other.to_string(), false),
    }
}

/// The inverse of [`fault_of`]: the error class a fault names.
fn error_of(fault: ModelFault) -> Error {
    match fault.code.as_str() {
        "ADAPTER_RESOLUTION" => Error::adapter_resolution(fault.message),
        "INVALID_REQUEST" => Error::invalid_argument(fault.message),
        _ if fault.recoverable => Error::model_recoverable(fault.message),
        _ => Error::model(fault.message),
    }
}

/// A link failure as the caller's error.
fn link_error(err: GymError) -> Error {
    match err {
        GymError::Timeout(t) => Error::Timeout(t),
        // Retrying cannot help (a key the host does not accept): final.
        other if !other.is_transient() => Error::Protocol(other.to_string()),
        other => Error::Connection(other.to_string()),
    }
}

/// A resolved route.
struct Route {
    contract: Arc<EnvContract>,
    action_space: SpaceSpec,
}

/// Answers a served model's requests with a [`ModelHandler`].
pub struct ModelService<H> {
    handler: Arc<Mutex<H>>,
    setup: Option<Arc<dyn ModelRouteSetup>>,
    reset_options: Option<Arc<dyn ModelResetOptions>>,
    routes: Mutex<HashMap<String, Arc<Route>>>,
    /// Notified when a route is released: a batch waiting for every route
    /// may now hold them all.
    released: tokio::sync::Notify,
    describe: String,
    /// The workflow edition the model declares and every one it runs, shown
    /// in the roster of each session it answers in.
    editions: (String, Vec<String>),
    flights: Arc<Flights>,
    /// Set by the first `close`: requests from then on are refused.
    closing: std::sync::atomic::AtomicBool,
    /// The closer's outcome, once it ends.
    closed: watch::Sender<Option<Result<(), Error>>>,
    /// Whether the handler fuses a grouped predict, which batching needs.
    fuses: bool,
    batching: Option<Batching>,
}
