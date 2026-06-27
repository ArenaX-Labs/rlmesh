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
