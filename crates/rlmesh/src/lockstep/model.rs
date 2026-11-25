//! A model served on a session: one request, one reply, on `m/{model}`.
//!
//! A [`ModelHandler`] is answered by [`ModelService`]: routes resolve off the predict lock,
//! every forward runs under it, and each group of a predict gets its own outcome.
//! [`ModelBinding::serve`] joins a session and answers there; the driver side is a
//! [`ModelLink`]. A model coalesces predicts from every session it answers in into one
//! handler call ([`BatchOptions`]).
//!
//! The first [`ModelService::close`] refuses new requests and waits for every accepted one to
//! land, then releases the routes and runs `on_close`; every close waits for that, up to its
//! deadline.

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

const PROBE: &str = "probe";

fn fault(code: &str, message: impl Into<String>, recoverable: bool) -> ModelFault {
    ModelFault {
        code: code.into(),
        message: message.into(),
        recoverable,
    }
}

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

fn error_of(fault: ModelFault) -> Error {
    match fault.code.as_str() {
        "ADAPTER_RESOLUTION" => Error::adapter_resolution(fault.message),
        "INVALID_REQUEST" => Error::invalid_argument(fault.message),
        _ if fault.recoverable => Error::model_recoverable(fault.message),
        _ => Error::model(fault.message),
    }
}

fn link_error(err: GymError) -> Error {
    match err {
        GymError::Timeout(t) => Error::Timeout(t),
        other if !other.is_transient() => Error::Protocol(other.to_string()),
        other => Error::Connection(other.to_string()),
    }
}

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
    released: tokio::sync::Notify,
    describe: String,
    workflow_edition: String,
    flights: Arc<Flights>,
    closing: std::sync::atomic::AtomicBool,
    closed: watch::Sender<Option<Result<(), Error>>>,
    fuses: bool,
    batching: Option<Batching>,
}
