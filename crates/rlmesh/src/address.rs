use std::fmt;
use std::path::PathBuf;

use rlmesh_grpc::helpers::{BindTarget, parse_bind_target, parse_env_connect_target};

use crate::{Error, Result};

/// A client-side address for connecting to a running server.
///
/// Build one with [`ConnectAddress::parse`], or pass a `&str` directly to the
/// connect helpers (e.g. [`crate::RemoteEnv::connect`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectAddress {
    /// A TCP endpoint, e.g. `tcp://host:port`.
    Tcp(String),
    /// A Unix-domain socket path (Unix only).
    Unix(PathBuf),
}

/// A server-side address to bind a listener to.
///
/// Build one with [`BindAddress::parse`]. Bind to TCP port `0` to let the OS
/// assign a free port and read the result back from the bound server's
/// `local_addr()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindAddress {
    /// A TCP host/port to bind. Port `0` requests an OS-assigned port.
    Tcp {
        /// Host or interface to bind (e.g. `127.0.0.1`, `0.0.0.0`).
        host: String,
        /// Port to bind; `0` requests an OS-assigned port.
        port: u16,
    },
    /// A Unix-domain socket path to bind (Unix only).
    Unix {
        /// Filesystem path for the socket.
        path: PathBuf,
    },
}

impl ConnectAddress {
    /// Parse a connect target such as `host:port`, `tcp://host:port`, or
    /// `unix:///path/to.sock`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Address`](crate::Error::Address) if `value` is not a
    /// recognized address (including a `unix://` path on Windows, where Unix
    /// sockets are unsupported).
    pub fn parse(value: impl AsRef<str>) -> Result<Self> {
        let target = parse_env_connect_target(value.as_ref()).map_err(Error::from)?;
        Ok(match target.unix_path() {
            Some(path) => Self::Unix(path.clone()),
            None => Self::Tcp(target.display_address().to_string()),
        })
    }

    /// The address rendered as a connect string (`tcp://...` or `unix://...`).
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
    /// Parse a bind target such as `port`, `host:port`, `tcp://host:port`, or
    /// `unix:///path/to.sock`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Address`](crate::Error::Address) if `value` is not a
    /// recognized bind target (including a `unix://` path on Windows).
    pub fn parse(value: impl AsRef<str>) -> Result<Self> {
        Ok(
            match parse_bind_target(value.as_ref()).map_err(Error::from)? {
                BindTarget::Tcp { host, port } => Self::Tcp { host, port },
                BindTarget::Unix { path } => Self::Unix { path },
            },
        )
    }

    /// The address rendered as a `tcp://host:port` or `unix://path` string.
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
