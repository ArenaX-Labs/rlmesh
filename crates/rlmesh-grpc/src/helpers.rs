//! Transport-agnostic helpers shared by the clients and servers: [`address`]
//! parsing/normalization for connect and bind targets, and [`auth`] bearer-token
//! comparison.

pub mod address;
pub mod auth;

pub use address::{
    BindTarget, EnvConnectTarget, normalize_endpoint, normalize_tcp_session_address,
    parse_bind_target, parse_env_connect_target,
};
pub use auth::{bearer_token_matches, constant_time_eq};
