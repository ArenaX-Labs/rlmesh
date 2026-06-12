pub mod connect;
pub mod env;
pub mod error;
pub mod helpers;
pub mod lifecycle;
pub mod model;
pub mod states;
pub mod wire;

/// Maximum gRPC message size (encode and decode) configured on all rlmesh
/// clients and servers.
///
/// tonic defaults the decode limit to 4 MiB, which silently rejects realistic
/// vectorized observation payloads (e.g. 64 envs of float32 (4,84,84) is
/// ~7.2 MiB). We raise both the encode and decode limits to a generous bound so
/// large observations/actions are not capped at an undiagnosable 4 MiB.
pub const MAX_MESSAGE_SIZE: usize = 256 * 1024 * 1024;

/// Harden a client endpoint so dead peers are detected instead of hanging
/// forever: a bounded connect, HTTP/2 keepalive pings (also effective over
/// unix sockets), and TCP keepalive for tcp transports. Without these a silent
/// network partition leaves RPC futures pending indefinitely.
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
pub use lifecycle::ServeOptions;
pub use model::ModelClient;
