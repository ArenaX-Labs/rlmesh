//! What can go wrong on the lockstep path.

use rlmesh_proto::env::v1::{EnvError as ProtoEnvError, EnvErrorCode};
use rlmesh_session::SessionError;

/// A lockstep failure.
#[derive(Debug, thiserror::Error)]
pub enum GymError {
    /// A Gym env cannot be mapped to a robot (§12.5), or a robot to a Gym
    /// view (§12.6).
    #[error("{0}")]
    Map(String),
    /// A message or payload does not fit the map.
    #[error("malformed lockstep message: {0}")]
    Malformed(String),
    /// The env answered with an error (a v1 env error, unchanged).
    #[error("env error: {}", .0.message)]
    Env(ProtoEnvError),
    /// The env refused the step: a stale epoch or lease token, or an e-stop.
    #[error("lockstep step refused: {0}")]
    Refused(String),
    /// No reply in time.
    #[error("no lockstep reply within {0:?}")]
    Timeout(std::time::Duration),
    /// The session.
    #[error(transparent)]
    Session(#[from] SessionError),
    /// The wire.
    #[error(transparent)]
    Wire(#[from] rlmesh_wire::WireError),
    /// The link is gone.
    #[error("the lockstep link is closed: {0}")]
    Closed(String),
    /// A bad address or configuration.
    #[error("{0}")]
    Config(String),
}

impl GymError {
    /// This error as a v1 env error (what a lockstep error reply carries).
    pub fn to_proto(&self) -> ProtoEnvError {
        match self {
            Self::Env(error) => error.clone(),
            Self::Refused(message) => proto(EnvErrorCode::NotReady, message.clone(), false),
            Self::Malformed(message) => proto(EnvErrorCode::InvalidAction, message.clone(), false),
            other => proto(EnvErrorCode::Internal, other.to_string(), false),
        }
    }

    /// Whether retrying can help.
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::Env(error) => error.is_recoverable,
            Self::Timeout(_) => true,
            _ => false,
        }
    }
}

fn proto(code: EnvErrorCode, message: String, recoverable: bool) -> ProtoEnvError {
    ProtoEnvError {
        code: code as i32,
        message,
        is_recoverable: recoverable,
        debug_info: String::new(),
        interrupted_episodes: Vec::new(),
    }
}
