//! Shared bind-first listener plumbing for the env and model servers.
//!
//! Binding the socket up front (rather than inside `serve`) lets callers learn
//! the resolved address — most importantly when binding to port 0 — before the
//! server starts awaiting shutdown, removing the bind-drop-rebind races and
//! poll-connect loops that consumers otherwise reimplement.

use std::net::SocketAddr;

use tokio::net::TcpListener;
#[cfg(unix)]
use tokio::net::UnixListener;
#[cfg(unix)]
use tokio_stream::wrappers::UnixListenerStream;
use tonic::transport::server::TcpIncoming;

use crate::{BindAddress, Error, Result};

/// A bound-but-not-yet-serving listener for a [`BindAddress`].
pub(crate) enum BoundListener {
    Tcp(TcpListener),
    #[cfg(unix)]
    Unix {
        listener: UnixListener,
        path: std::path::PathBuf,
    },
}

impl BoundListener {
    /// Bind the listener for `addr`, performing stale Unix-socket cleanup.
    pub(crate) async fn bind(addr: BindAddress) -> Result<Self> {
        match addr {
            BindAddress::Tcp { host, port } => {
                let listener = TcpListener::bind((host.as_str(), port))
                    .await
                    .map_err(|err| Error::Server(err.to_string()))?;
                Ok(Self::Tcp(listener))
            }
            BindAddress::Unix { path } => {
                #[cfg(not(unix))]
                {
                    let _ = path;
                    Err(Error::Address(
                        "unix sockets are not supported on Windows; use tcp://host:port instead"
                            .to_string(),
                    ))
                }

                #[cfg(unix)]
                {
                    crate::address::remove_stale_socket(&path)?;
                    let listener =
                        UnixListener::bind(&path).map_err(|err| Error::Server(err.to_string()))?;
                    Ok(Self::Unix { listener, path })
                }
            }
        }
    }

    /// The resolved address the listener is bound to.
    ///
    /// For a TCP listener bound to port 0 this returns the OS-assigned port.
    pub(crate) fn local_addr(&self) -> Result<BindAddress> {
        match self {
            Self::Tcp(listener) => {
                let addr: SocketAddr = listener
                    .local_addr()
                    .map_err(|err| Error::Server(err.to_string()))?;
                Ok(BindAddress::Tcp {
                    host: addr.ip().to_string(),
                    port: addr.port(),
                })
            }
            #[cfg(unix)]
            Self::Unix { path, .. } => Ok(BindAddress::Unix { path: path.clone() }),
        }
    }

    /// Serve `router` over this listener until `shutdown` fires, draining for at
    /// most `drain_timeout`. Unlinks the Unix socket file after shutdown.
    pub(crate) async fn serve(
        self,
        router: tonic::transport::server::Router,
        shutdown: rlmesh_grpc::lifecycle::ShutdownTrigger,
        drain_timeout: Option<std::time::Duration>,
    ) -> Result<()> {
        match self {
            Self::Tcp(listener) => rlmesh_grpc::lifecycle::await_server_shutdown(
                // Nagle off on accepted connections. A step response is one
                // large write (an image observation is ~200 KB); with Nagle on,
                // its tail waits for the client's delayed ACK and a loopback
                // step stalls ~40 ms about one time in ten. tonic only warns
                // when the option cannot be set; the connection still serves.
                router.serve_with_incoming_shutdown(
                    TcpIncoming::from(listener).with_nodelay(Some(true)),
                    shutdown.cancelled_owned(),
                ),
                shutdown.clone(),
                drain_timeout,
            )
            .await
            .map_err(|err| Error::Server(err.to_string())),
            #[cfg(unix)]
            Self::Unix { listener, path } => {
                let result = rlmesh_grpc::lifecycle::await_server_shutdown(
                    router.serve_with_incoming_shutdown(
                        UnixListenerStream::new(listener),
                        shutdown.cancelled_owned(),
                    ),
                    shutdown.clone(),
                    drain_timeout,
                )
                .await
                .map_err(|err| Error::Server(err.to_string()));
                // Unlink the socket file on shutdown so a subsequent serve on
                // the same path does not fail with AddrInUse.
                let _ = std::fs::remove_file(&path);
                result
            }
        }
    }
}
