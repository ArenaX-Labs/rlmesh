use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;

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

impl FromStr for ConnectAddress {
    type Err = Error;

    /// Equivalent to [`ConnectAddress::parse`]; lets you write
    /// `"tcp://host:port".parse::<ConnectAddress>()`.
    fn from_str(value: &str) -> Result<Self> {
        Self::parse(value)
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

impl FromStr for BindAddress {
    type Err = Error;

    /// Equivalent to [`BindAddress::parse`]; lets you write
    /// `"tcp://host:port".parse::<BindAddress>()`.
    fn from_str(value: &str) -> Result<Self> {
        Self::parse(value)
    }
}

/// Remove a leftover Unix-domain socket file before binding.
///
/// `bind(2)` returns `EADDRINUSE` whenever the socket path already exists on
/// disk, regardless of whether anything is listening, so a socket left by a
/// crashed server must be unlinked before we rebind. We probe first to avoid
/// stealing the address from a *live* server: a successful connect means the
/// socket is live and we refuse to bind, and only a `ConnectionRefused` probe
/// is treated as stale and unlinked. Any other connect failure (e.g. permission
/// denied) cannot prove the socket is dead, so we report it rather than risk
/// unlinking a socket still in use. Non-socket files are left for `bind` to
/// reject.
#[cfg(unix)]
pub(crate) fn remove_stale_socket(path: &std::path::Path) -> Result<()> {
    use std::io::ErrorKind;
    use std::os::unix::fs::FileTypeExt;

    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_socket() => {
            match std::os::unix::net::UnixStream::connect(path) {
                Ok(_) => Err(Error::Server(format!(
                    "address already in use: a server is already listening on {}",
                    path.display()
                ))),
                // Nothing is accepting: the socket is stale and safe to unlink.
                Err(err) if err.kind() == ErrorKind::ConnectionRefused => {
                    std::fs::remove_file(path).map_err(|err| {
                        Error::Server(format!(
                            "failed to remove stale socket {}: {err}",
                            path.display()
                        ))
                    })
                }
                // The path vanished from under us; nothing left to remove.
                Err(err) if err.kind() == ErrorKind::NotFound => Ok(()),
                // Can't prove it's stale; fail safe rather than steal the path.
                Err(err) => Err(Error::Server(format!(
                    "cannot verify whether socket {} is stale: {err}",
                    path.display()
                ))),
            }
        }
        // Not a socket (or does not exist): leave it alone and let bind decide.
        _ => Ok(()),
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn refuses_to_unlink_a_live_socket() {
        let dir = std::env::temp_dir().join(format!("rlmesh-live-{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&dir);
        let _listener = std::os::unix::net::UnixListener::bind(&dir).unwrap();

        let result = remove_stale_socket(&dir);

        assert!(
            matches!(result, Err(Error::Server(_))),
            "live socket must not be unlinked, got: {result:?}"
        );
        assert!(dir.exists(), "live socket file must be left in place");
        let _ = std::fs::remove_file(&dir);
    }

    #[test]
    fn removes_a_stale_socket_with_no_listener() {
        let path = std::env::temp_dir().join(format!("rlmesh-stale-{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&path);
        // Bind then drop the listener, leaving the socket file behind with
        // nothing accepting on it.
        {
            let _listener = std::os::unix::net::UnixListener::bind(&path).unwrap();
        }
        assert!(path.exists(), "precondition: stale socket file present");

        remove_stale_socket(&path).expect("stale socket should be removed");

        assert!(!path.exists(), "stale socket file must be unlinked");
    }
}
