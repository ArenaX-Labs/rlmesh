use std::fmt;
use std::path::PathBuf;

use rlmesh_grpc::helpers::{BindTarget, parse_bind_target, parse_env_connect_target};

use crate::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectAddress {
    Tcp(String),
    Unix(PathBuf),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindAddress {
    Tcp { host: String, port: u16 },
    Unix { path: PathBuf },
}

impl ConnectAddress {
    pub fn parse(value: impl AsRef<str>) -> Result<Self> {
        let target = parse_env_connect_target(value.as_ref()).map_err(Error::from)?;
        Ok(match target.unix_path() {
            Some(path) => Self::Unix(path.clone()),
            None => Self::Tcp(target.display_address().to_string()),
        })
    }

    pub fn as_str(&self) -> String {
        match self {
            Self::Tcp(value) => value.clone(),
            Self::Unix(path) => format!("unix://{}", path.display()),
        }
    }
}

impl fmt::Display for ConnectAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_str().fmt(f)
    }
}

impl BindAddress {
    pub fn parse(value: impl AsRef<str>) -> Result<Self> {
        Ok(
            match parse_bind_target(value.as_ref()).map_err(Error::from)? {
                BindTarget::Tcp { host, port } => Self::Tcp { host, port },
                BindTarget::Unix { path } => Self::Unix { path },
            },
        )
    }

    pub fn display_address(&self) -> String {
        match self {
            Self::Tcp { host, port } => format!("tcp://{host}:{port}"),
            Self::Unix { path } => format!("unix://{}", path.display()),
        }
    }
}

impl fmt::Display for BindAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.display_address().fmt(f)
    }
}

/// Remove a leftover Unix-domain socket file before binding.
///
/// `bind(2)` returns `EADDRINUSE` whenever the socket path already exists on
/// disk, regardless of whether anything is listening, so a clean shutdown that
/// leaves the socket file behind would otherwise make every subsequent serve on
/// the same path fail. We only unlink a path that is actually a socket, so we
/// never clobber an unrelated regular file (the bind will then surface a clear
/// error instead).
#[cfg(unix)]
pub(crate) fn remove_stale_socket(path: &std::path::Path) -> Result<()> {
    use std::os::unix::fs::FileTypeExt;

    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_socket() => {
            std::fs::remove_file(path).map_err(|err| {
                Error::Server(format!(
                    "failed to remove stale socket {}: {err}",
                    path.display()
                ))
            })
        }
        // Not a socket (or does not exist): leave it alone and let bind decide.
        _ => Ok(()),
    }
}
