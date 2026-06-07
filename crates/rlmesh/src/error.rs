use std::fmt;
use std::time::Duration;

use rlmesh_grpc::error::{EnvError, EnvErrorCode};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    Unspecified,
    Timeout,
    InvalidAction,
    NotReady,
    Busy,
    Internal,
    Crashed,
    Cancelled,
    Closed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnvironmentError {
    pub code: ErrorCode,
    pub message: String,
    pub is_recoverable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Address(String),
    Connection(String),
    Timeout(Duration),
    Environment(EnvironmentError),
    Server(String),
    Internal(String),
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Unspecified => "UNSPECIFIED",
            Self::Timeout => "TIMEOUT",
            Self::InvalidAction => "INVALID_ACTION",
            Self::NotReady => "NOT_READY",
            Self::Busy => "BUSY",
            Self::Internal => "INTERNAL",
            Self::Crashed => "CRASHED",
            Self::Cancelled => "CANCELLED",
            Self::Closed => "CLOSED",
        };
        f.write_str(label)
    }
}

impl fmt::Display for EnvironmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)
    }
}

impl std::error::Error for EnvironmentError {}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Address(message) => write!(f, "invalid address: {message}"),
            Self::Connection(message) => write!(f, "connection error: {message}"),
            Self::Timeout(duration) => write!(f, "timeout after {duration:?}"),
            Self::Environment(error) => write!(f, "environment error: {error}"),
            Self::Server(message) => write!(f, "server error: {message}"),
            Self::Internal(message) => write!(f, "internal error: {message}"),
        }
    }
}

impl std::error::Error for Error {}

fn map_env_error_code(code: EnvErrorCode) -> ErrorCode {
    match code {
        EnvErrorCode::Unspecified => ErrorCode::Unspecified,
        EnvErrorCode::Timeout => ErrorCode::Timeout,
        EnvErrorCode::InvalidAction => ErrorCode::InvalidAction,
        EnvErrorCode::NotReady => ErrorCode::NotReady,
        EnvErrorCode::Busy => ErrorCode::Busy,
        EnvErrorCode::Internal => ErrorCode::Internal,
        EnvErrorCode::Crashed => ErrorCode::Crashed,
        EnvErrorCode::Cancelled => ErrorCode::Cancelled,
        EnvErrorCode::Closed => ErrorCode::Closed,
    }
}

impl From<EnvError> for EnvironmentError {
    fn from(value: EnvError) -> Self {
        Self {
            code: map_env_error_code(value.code),
            message: value.message,
            is_recoverable: value.is_recoverable,
        }
    }
}

impl From<rlmesh_grpc::error::ProtocolError> for Error {
    fn from(value: rlmesh_grpc::error::ProtocolError) -> Self {
        Self::Internal(value.to_string())
    }
}

impl From<rlmesh_grpc::error::Error> for Error {
    fn from(value: rlmesh_grpc::error::Error) -> Self {
        match value {
            rlmesh_grpc::error::Error::Transport(error) => match error {
                rlmesh_grpc::error::TransportError::InvalidAddress(message) => {
                    Self::Address(message)
                }
                rlmesh_grpc::error::TransportError::BindFailed(message) => Self::Server(message),
                rlmesh_grpc::error::TransportError::ConnectFailed(message) => {
                    Self::Connection(message)
                }
                rlmesh_grpc::error::TransportError::ConnectionClosed => {
                    Self::Connection("connection closed".to_string())
                }
                rlmesh_grpc::error::TransportError::Io(error) => {
                    Self::Connection(error.to_string())
                }
                rlmesh_grpc::error::TransportError::MessageTooLarge { size, max } => {
                    Self::Connection(format!("message too large: {size} > {max}"))
                }
            },
            rlmesh_grpc::error::Error::Protocol(error) => Self::Internal(error.to_string()),
            rlmesh_grpc::error::Error::Environment(error) => {
                Self::Environment(EnvironmentError::from(error))
            }
            rlmesh_grpc::error::Error::Timeout(duration) => Self::Timeout(duration),
            rlmesh_grpc::error::Error::Cancelled(message) => Self::Internal(message),
            rlmesh_grpc::error::Error::Server(error) => Self::Server(error.to_string()),
            rlmesh_grpc::error::Error::Client(error) => Self::Connection(error.to_string()),
        }
    }
}
