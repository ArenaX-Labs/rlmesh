//! The wire-level env plumbing the Python binding serves through: the async
//! request/result env traits, the lane host, the transport adapters and the
//! request/result types. Authors use the sync [`crate::Env`] instead.

pub use crate::binding::spaces::request::{
    CloseResult, ResetRequest, ResetResult, StepRequest, StepResult,
};
pub use crate::binding::spaces::{CloseRequest, RenderFrame, RenderRequest, RenderResult};
pub use crate::env::lanes::LaneEnv;
pub use crate::env::legacy::{Env, VectorEnv};
pub use crate::env::legacy_server::{BoundWireEnvServer, WireEnvServer, WireVectorEnvServer};
pub use crate::env::types::{
    CloseResult as VectorCloseResult, EpisodeMetadata, ResetRequest as VectorResetRequest,
    ResetResult as VectorResetResult, StepRequest as VectorStepRequest,
    StepResult as VectorStepResult,
};
pub use crate::env::wire::{WireEnvAdapter, WireLaneAdapter};
pub use rlmesh_proto::{EndpointPhases, lane_skew_ns};

/// The any-arity remote env client both shared clients wrap (RMT-14), how
/// it dials, and its edition cell (RMT-20).
pub use crate::env::client::{CoreConnect, EditionCell, EnvClientCore};

/// The core behind a [`crate::RemoteEnv`] (the Runtime class while a session
/// borrows it).
pub fn env_core(env: &crate::RemoteEnv) -> crate::Result<&EnvClientCore> {
    env.core_ref()
        .ok_or_else(|| crate::Error::Runtime("RemoteEnv is in use by a session".to_string()))
}

/// The core behind a [`crate::RemoteVectorEnv`].
pub fn vector_env_core(env: &crate::RemoteVectorEnv) -> crate::Result<&EnvClientCore> {
    env.core_ref()
        .ok_or_else(|| crate::Error::Runtime("RemoteVectorEnv is in use by a session".to_string()))
}

/// Python's `normalize_connect_address` (one implementation for every
/// language): an address, or the `host` / `port` / `path` / `transport`
/// helpers, as one connect string; every refusal is the ValueError class.
pub fn normalize_connect_address(
    address: Option<&str>,
    host: Option<&str>,
    port: Option<u16>,
    path: Option<&str>,
    transport: Option<crate::Transport>,
) -> crate::Result<String> {
    crate::env::client::normalize_connect_address(address, host, port, path, transport)
}

/// A client timeout in seconds (RMT-4): `None` and zero are no timeout;
/// negative or non-finite is the ValueError class naming `field`.
pub fn client_timeout_from_secs(
    value: Option<f64>,
    field: &str,
) -> crate::Result<Option<std::time::Duration>> {
    crate::env::client::timeout_from_secs(value, field)
}

/// Decode a wire PNG render frame to `(shape, raw uint8)` (RMT-16); an
/// undecodable frame is the ProtocolException class.
pub fn decode_render_frame(png: &[u8]) -> crate::Result<(Vec<usize>, Vec<u8>)> {
    crate::env::client::decode_render_frame(png)
}
