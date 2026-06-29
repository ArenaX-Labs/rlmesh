//! Tonic gRPC transport for the RLMesh runtime contract.
//!
//! This crate carries env and model traffic over gRPC. It provides the
//! [`EnvClient`] and [`ModelClient`] clients, the [`mod@env`] server behind
//! EnvService (the ModelService server lives in the `rlmesh` facade, which
//! parameterizes it over the public model-handler family), the [`wire`] codec
//! that converts native space values and env contracts to and from their proto
//! form, and the [`lifecycle`] plumbing the servers run under (idle/drain/close
//! timeouts, shutdown triggers, and in-flight activity tracking).
//!
//! Most users reach this through the higher-level `rlmesh` facade. Depend on it
//! directly to embed a client or server, to reuse the [`wire`] helpers, or to
//! reconcile the three-way session floor across env, model, and runtime with
//! [`env_floor`].
//!
//! [`connect`] centralizes the deadline/backoff/cancellation retry policy, and
//! [`health`] wires the standard `grpc.health.v1` service.

pub mod connect;
pub mod env;
pub mod error;
pub mod floor;
pub mod health;
pub mod helpers;
pub mod lifecycle;
pub mod model;
pub mod states;
pub mod wire;

/// gRPC encode/decode limit used by RLMesh clients and servers.
///
/// The tonic default is 4 MiB. RLMesh raises it so vectorized observations such
/// as 64 float32 Atari frames fit without tripping transport limits.
pub const MAX_MESSAGE_SIZE: usize = 256 * 1024 * 1024;

/// Configure endpoint timeouts and keepalives.
pub(crate) fn configure_endpoint(
    endpoint: tonic::transport::Endpoint,
) -> tonic::transport::Endpoint {
    endpoint
        .connect_timeout(std::time::Duration::from_secs(10))
        .http2_keep_alive_interval(std::time::Duration::from_secs(30))
        .keep_alive_timeout(std::time::Duration::from_secs(10))
        .keep_alive_while_idle(true)
        .tcp_keepalive(Some(std::time::Duration::from_secs(60)))
}

pub use connect::{ConnectOptions, retry_connect};
pub use env::{EnvClient, EnvHandshake};
pub use floor::env_floor;
pub use lifecycle::{DEFAULT_PREDICT_CONCURRENCY, ServeOptions};
pub use model::ModelClient;
