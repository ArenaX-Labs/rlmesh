use thiserror::Error;

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum SpaceError {
    #[error("{path}: {msg}")]
    Invalid { path: String, msg: String },
}

impl SpaceError {
    /// Build an [`SpaceError::Invalid`] at `path` with `msg`.
    pub(crate) fn invalid(path: impl Into<String>, msg: impl Into<String>) -> Self {
        SpaceError::Invalid {
            path: path.into(),
            msg: msg.into(),
        }
    }

    /// The value path this error is anchored at (advisory; used to dedup
    /// conformance warnings per path).
    #[must_use]
    pub fn path(&self) -> &str {
        match self {
            SpaceError::Invalid { path, .. } => path,
        }
    }
}

#[derive(Error, Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
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
