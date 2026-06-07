use thiserror::Error;

#[derive(Error, Debug)]
pub enum SpaceError {
    #[error("{path}: {msg}")]
    Invalid { path: String, msg: String },
}

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum EnvRuntimeError {
    #[error("invalid space: {0}")]
    InvalidSpace(String),
    #[error("invalid value: {0}")]
    InvalidValue(String),
    #[error("runtime error: {0}")]
    Runtime(String),
}

macro_rules! err_space {
    // err_space!(path, msg)
    ($path:expr, $msg:expr) => {
        Err(SpaceError::Invalid {
            path: $path.into(),
            msg: $msg.into(),
        })
    };

    // err_space!(path, name, msg)
    ($path:expr, $name:expr, $msg:expr) => {
        Err(SpaceError::Invalid {
            path: $path.into(),
            msg: format!("[{}] {}", $name, $msg),
        })
    };
}
pub(crate) use err_space;
