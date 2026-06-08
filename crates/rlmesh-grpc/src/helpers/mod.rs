pub mod address;
pub(crate) mod handshake;

pub use address::{
    BindTarget, EnvConnectTarget, normalize_endpoint, normalize_tcp_session_address,
    parse_bind_target, parse_env_connect_target,
};

#[cfg(test)]
mod address_tests;
