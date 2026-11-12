//! What can go wrong on the lockstep path.

use rlmesh_session::SessionError;
use serde_json::{Value, json};

use crate::{EnvironmentError, ErrorCode};

/// A lockstep failure.
#[derive(Debug, thiserror::Error)]
pub enum GymError {
    /// A Gym env cannot be mapped to a robot, or a robot to a Gym
    /// view.
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
            class: None,
        })
    }

    /// This error as the env failure a lockstep error reply carries.
    pub fn to_failure(&self) -> EnvironmentError {
        let failure = |code, message: String| EnvironmentError {
            code,
            message,
            is_recoverable: false,
            class: None,
        };
        match self {
            Self::Env(error) => error.clone(),
            Self::Refused(message) => failure(ErrorCode::NotReady, message.clone()),
            Self::Malformed(message) => failure(ErrorCode::InvalidAction, message.clone()),
            other => failure(ErrorCode::Internal, other.to_string()),
        }
    }

    /// Whether the env stays usable after this failure: its own
    /// recoverable flag, or a reply that did not come in time. What the
    /// driver records on an env call; see [`is_transient`](Self::is_transient)
    /// for whether to retry a call.
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::Env(error) => error.is_recoverable,
            Self::Timeout(_) => true,
            _ => false,
        }
    }

    /// Whether retrying the same call can succeed, as
    /// [`crate::Error::is_transient`] answers it: a peer not reachable or not
    /// there yet (no host listening, no reply in time, the robot not joined,
    /// its lease held by another driver), or the env's recoverable failure.
    /// A refusal that the same call meets again is not: a key the host does
    /// not accept, a malformed address or map, a protocol the session lacks.
    ///
    /// A driver that dials a peer that may still be starting retries
    /// [`connect`](crate::lockstep::connect) while this holds and stops at
    /// once when it does not. A robot name the session never sees reads as
    /// one not joined yet: the session cannot tell them apart.
    pub fn is_transient(&self) -> bool {
        match self {
            Self::Env(error) => error.is_recoverable,
            Self::Timeout(_) | Self::Closed(_) => true,
            Self::Session(error) => session_transient(error),
            Self::Map(_)
            | Self::Malformed(_)
            | Self::Value(_)
            | Self::Refused(_)
            | Self::Wire(_)
            | Self::Config(_) => false,
        }
    }
}

/// Whether a session failure can clear on its own: the transport, a
/// deadline, or a refusal of a state that moves (a lease, an epoch, a
/// member arriving).
fn session_transient(error: &SessionError) -> bool {
    use rlmesh_session::wire::CtlStatus;
    match error {
        SessionError::Zenoh(_)
        | SessionError::NoReply(_)
        | SessionError::Pending { .. }
        | SessionError::Timeout(..)
        | SessionError::Io(_)
        | SessionError::Closed => true,
        SessionError::Rejected(refusal) | SessionError::Unverified(refusal) => matches!(
            refusal.status,
            CtlStatus::StaleEpoch
                | CtlStatus::StaleToken
                | CtlStatus::InvalidTransition
                | CtlStatus::LeaseBusy
                | CtlStatus::GroupBusy
                | CtlStatus::RobotBusy
                | CtlStatus::NotReady
                | CtlStatus::NotPresent
        ),
        _ => false,
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

/// An env failure as a lockstep error reply: `{code, message, recoverable}`,
/// and `class` when the env raised a builtin exception class.
pub fn failure_to_json(failure: &EnvironmentError) -> Vec<u8> {
    let mut value = json!({
        "code": failure.code.as_str(),
        "message": failure.message,
        "recoverable": failure.is_recoverable,
    });
    if let Some(class) = &failure.class {
        value["class"] = json!(class);
    }
    value.to_string().into_bytes()
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
        class: value
            .get("class")
            .and_then(Value::as_str)
            .map(str::to_string),
    })
}
