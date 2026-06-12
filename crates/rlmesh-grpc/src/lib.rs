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

pub use connect::{ConnectOptions, retry_connect};
pub use env::{EnvClient, EnvHandshake};
pub use lifecycle::ServeOptions;
pub use model::ModelClient;
