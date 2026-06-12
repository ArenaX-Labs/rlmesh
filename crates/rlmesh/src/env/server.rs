use std::sync::Arc;

use rlmesh_grpc::env::Environment;
use rlmesh_grpc::lifecycle::{
    await_close_with_timeout, await_server_shutdown, start_idle_shutdown,
};
use tokio::net::TcpListener;
#[cfg(unix)]
use tokio::net::UnixListener;
use tokio::sync::Mutex;
use tokio_stream::wrappers::TcpListenerStream;
#[cfg(unix)]
use tokio_stream::wrappers::UnixListenerStream;

use super::{Env, WireEnvAdapter};
use crate::{BindAddress, EnvironmentError, Error, Result, ServeOptions};

pub struct EnvServer<E: Env> {
    env: E,
}

impl<E: Env> EnvServer<E> {
    pub fn new(env: E) -> Self {
        Self { env }
    }
}

impl<E: Env + 'static> EnvServer<E> {
    pub async fn serve(self, addr: BindAddress) -> Result<()> {
        self.serve_with_options(addr, ServeOptions::default()).await
    }

    pub async fn serve_with_options(self, addr: BindAddress, options: ServeOptions) -> Result<()> {
        let shutdown = rlmesh_grpc::lifecycle::ShutdownTrigger::new();
        let activity_tx = start_idle_shutdown(options.idle_timeout, shutdown.clone());
        let grpc_options = rlmesh_grpc::ServeOptions::from(options);

        let env = Arc::new(Mutex::new(WireEnvAdapter::new(self.env)));
        let service = rlmesh_grpc::env::env_service_from_shared(
            Arc::clone(&env),
            shutdown.clone(),
            grpc_options,
            activity_tx,
        );
        let serve_result = match addr {
            BindAddress::Tcp { host, port } => {
                let listener = TcpListener::bind((host.as_str(), port))
                    .await
                    .map_err(|err| Error::Server(err.to_string()))?;
                await_server_shutdown(
                    tonic::transport::Server::builder()
                        .add_service(service)
                        .serve_with_incoming_shutdown(
                            TcpListenerStream::new(listener),
                            shutdown.cancelled_owned(),
                        ),
                    shutdown.clone(),
                    options.drain_timeout,
                )
                .await
                .map_err(|err| Error::Server(err.to_string()))
            }
            BindAddress::Unix { path } => {
                #[cfg(not(unix))]
                {
                    let _ = path;
                    return Err(Error::Address(
                        "unix sockets are not supported on Windows; use tcp://host:port instead"
                            .to_string(),
                    ));
                }

                #[cfg(unix)]
                {
                    crate::address::remove_stale_socket(&path)?;
                    let listener =
                        UnixListener::bind(&path).map_err(|err| Error::Server(err.to_string()))?;
                    let result = await_server_shutdown(
                        tonic::transport::Server::builder()
                            .add_service(service)
                            .serve_with_incoming_shutdown(
                                UnixListenerStream::new(listener),
                                shutdown.cancelled_owned(),
                            ),
                        shutdown.clone(),
                        options.drain_timeout,
                    )
                    .await
                    .map_err(|err| Error::Server(err.to_string()));
                    // Unlink the socket file on shutdown so a subsequent serve on
                    // the same path does not fail with AddrInUse.
                    let _ = std::fs::remove_file(&path);
                    result
                }
            }
        };
        let close_result = close_env(env, options.close_timeout).await;
        match (serve_result, close_result) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(err), Ok(())) => Err(err),
            (Ok(()), Err(err)) => Err(err),
            (Err(serve_err), Err(close_err)) => Err(Error::Internal(format!(
                "environment server failed: {serve_err}; close hook failed: {close_err}"
            ))),
        }
    }

    pub async fn serve_tcp(self, addr: impl Into<std::net::SocketAddr>) -> Result<()> {
        let addr = addr.into();
        self.serve(BindAddress::Tcp {
            host: addr.ip().to_string(),
            port: addr.port(),
        })
        .await
    }
}

async fn close_env<E: Environment>(
    env: Arc<Mutex<E>>,
    close_timeout: Option<std::time::Duration>,
) -> Result<()> {
    let close = async {
        env.lock()
            .await
            .close()
            .await
            .map(|_| ())
            .map_err(|err| Error::Environment(EnvironmentError::from(err)))
    };
    await_close_with_timeout(close, close_timeout)
        .await
        .map_err(Error::Timeout)?
}
