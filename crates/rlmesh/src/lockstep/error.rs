//! What can go wrong on the lockstep path.

use rlmesh_session::SessionError;
use serde_json::{Value, json};

use crate::{EnvironmentError, ErrorCode};

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
    /// A value does not fit its space (an action the client cannot send).
    #[error("{0}")]
    Value(String),
    /// The env answered with an error.
    #[error("env error: {}", .0.message)]
    Env(EnvironmentError),
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
    Wire(#[from] rlmesh_session::wire::WireError),
    /// The link is gone.
    #[error("the lockstep link is closed: {0}")]
    Closed(String),
    /// A bad address or configuration.
    #[error("{0}")]
    Config(String),
}

impl GymError {
    /// An env failure with `code`.
    pub fn env(code: ErrorCode, message: impl Into<String>, recoverable: bool) -> Self {
        Self::Env(EnvironmentError {
            code,
            message: message.into(),
            is_recoverable: recoverable,
        })
    }

    /// This error as the env failure a lockstep error reply carries.
    pub fn to_failure(&self) -> EnvironmentError {
        let failure = |code, message: String| EnvironmentError {
            code,
            message,
            is_recoverable: false,
        };
        match self {
            Self::Env(error) => error.clone(),
            Self::Refused(message) => failure(ErrorCode::NotReady, message.clone()),
            Self::Malformed(message) => failure(ErrorCode::InvalidAction, message.clone()),
            other => failure(ErrorCode::Internal, other.to_string()),
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

const CODES: [ErrorCode; 10] = [
    ErrorCode::Unspecified,
    ErrorCode::Timeout,
    ErrorCode::InvalidAction,
    ErrorCode::NotReady,
    ErrorCode::Busy,
    ErrorCode::Unsupported,
    ErrorCode::Internal,
    ErrorCode::Crashed,
    ErrorCode::Cancelled,
    ErrorCode::Closed,
];

/// An env failure as a lockstep error reply: `{code, message, recoverable}`.
pub fn failure_to_json(failure: &EnvironmentError) -> Vec<u8> {
    json!({
        "code": failure.code.as_str(),
        "message": failure.message,
        "recoverable": failure.is_recoverable,
    })
    .to_string()
    .into_bytes()
}

/// The inverse of [`failure_to_json`]; `None` for anything else (a
/// transport-level error reply).
pub fn failure_from_json(bytes: &[u8]) -> Option<EnvironmentError> {
    let value: Value = serde_json::from_slice(bytes).ok()?;
    let code = value.get("code")?.as_str()?;
    Some(EnvironmentError {
        code: CODES.into_iter().find(|c| c.as_str() == code)?,
        message: value.get("message")?.as_str()?.to_string(),
        is_recoverable: value.get("recoverable")?.as_bool()?,
    })
}
