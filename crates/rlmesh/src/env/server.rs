use std::sync::Arc;

use rlmesh_grpc::env::Environment;
use rlmesh_grpc::lifecycle::{await_close_with_timeout, start_idle_shutdown};
use tokio::sync::Mutex;

use super::{Env, WireEnvAdapter};
use crate::bound::BoundListener;
use crate::{BindAddress, EnvironmentError, Error, Result, ServeOptions};

/// Hosts an [`Env`] as a gRPC environment server.
///
/// Construct with [`EnvServer::new`], then either [`bind`](EnvServer::bind) to
/// reserve the socket and learn the resolved address before serving, or
/// [`serve`](EnvServer::serve) to bind and run in one call.
pub struct EnvServer<E: Env> {
    env: E,
}

impl<E: Env> EnvServer<E> {
    /// Wrap an [`Env`] implementation to be served.
    pub fn new(env: E) -> Self {
        Self { env }
    }
}

impl<E: Env + 'static> EnvServer<E> {
    /// Bind the server to `addr` without yet serving.
    ///
    /// The returned [`BoundEnvServer`] exposes [`BoundEnvServer::local_addr`]
    /// so callers can learn the resolved address (e.g. the OS-assigned port
    /// when binding to port 0) before awaiting shutdown.
    pub async fn bind(self, addr: BindAddress) -> Result<BoundEnvServer> {
        self.bind_with_options(addr, ServeOptions::default()).await
    }

    /// Bind the server to `addr` with explicit [`ServeOptions`].
    pub async fn bind_with_options(
        self,
        addr: BindAddress,
        options: ServeOptions,
    ) -> Result<BoundEnvServer> {
        let shutdown = rlmesh_grpc::lifecycle::ShutdownTrigger::new();
        let activity_tx = start_idle_shutdown(options.idle_timeout, shutdown.clone());
        let drain_timeout = options.drain_timeout;
        let close_timeout = options.close_timeout;
        let grpc_options = rlmesh_grpc::ServeOptions::from(options);

        let listener = BoundListener::bind(addr).await?;
        let local_addr = listener.local_addr()?;

        let env = Arc::new(Mutex::new(WireEnvAdapter::new(self.env)));
        let service = rlmesh_grpc::env::env_service_from_shared(
            Arc::clone(&env),
            shutdown.clone(),
            grpc_options,
            activity_tx,
        );
        // Always-on standard gRPC health service (`grpc.health.v1`). The
        // listener is already bound (bind-first), so the overall server health
        // is marked SERVING immediately (review finding #57).
        let (_health_reporter, health_service) =
            rlmesh_grpc::health::serving_health_service().await;
        let router = tonic::transport::Server::builder()
            .add_service(health_service)
            .add_service(service);
        // Upcast to a trait object so the bound handle does not leak the env
        // generic; only the close hook needs the environment afterward.
        let env: Arc<Mutex<dyn Environment + Send + Sync>> = env;

        Ok(BoundEnvServer {
            listener,
            router,
            shutdown,
            env,
            local_addr,
            drain_timeout,
            close_timeout,
        })
    }

    /// Bind to `addr` and serve until shutdown, with default [`ServeOptions`].
    ///
    /// Equivalent to [`bind`](EnvServer::bind) followed by
    /// [`BoundEnvServer::serve`], for callers that do not need the resolved
    /// address up front.
    pub async fn serve(self, addr: BindAddress) -> Result<()> {
        self.serve_with_options(addr, ServeOptions::default()).await
    }

    /// Bind to `addr` and serve until shutdown, with explicit [`ServeOptions`].
    pub async fn serve_with_options(self, addr: BindAddress, options: ServeOptions) -> Result<()> {
        self.bind_with_options(addr, options).await?.serve().await
    }

    /// Convenience wrapper around [`serve`](EnvServer::serve) for a
    /// [`SocketAddr`](std::net::SocketAddr) TCP bind target.
    pub async fn serve_tcp(self, addr: impl Into<std::net::SocketAddr>) -> Result<()> {
        let addr = addr.into();
        self.serve(BindAddress::Tcp {
            host: addr.ip().to_string(),
            port: addr.port(),
        })
        .await
    }
}

/// An [`EnvServer`] that has bound its listener but not yet started serving.
///
/// Created by [`EnvServer::bind`] / [`EnvServer::bind_with_options`]. Use
/// [`BoundEnvServer::local_addr`] to read the resolved bind address, then
/// [`BoundEnvServer::serve`] to run until shutdown.
pub struct BoundEnvServer {
    listener: BoundListener,
    router: tonic::transport::server::Router,
    shutdown: rlmesh_grpc::lifecycle::ShutdownTrigger,
    env: Arc<Mutex<dyn Environment + Send + Sync>>,
    local_addr: BindAddress,
    drain_timeout: Option<std::time::Duration>,
    close_timeout: Option<std::time::Duration>,
}

impl BoundEnvServer {
    /// The resolved address the server is bound to (the OS-assigned port for
    /// TCP port 0).
    pub fn local_addr(&self) -> &BindAddress {
        &self.local_addr
    }

    /// Serve until the server shuts down (idle timeout, remote shutdown, or
    /// drain), then run the environment close hook.
    ///
    /// The served environment outlives individual client sessions: a client
    /// calling `close` detaches its session but leaves the env running for the
    /// next client to connect (review finding #81). The environment's own close
    /// hook runs only here, once the *server* stops — not on each client close.
    pub async fn serve(self) -> Result<()> {
        let serve_result = self
            .listener
            .serve(self.router, self.shutdown, self.drain_timeout)
            .await;
        let close_result = close_env(self.env, self.close_timeout).await;
        match (serve_result, close_result) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(err), Ok(())) => Err(err),
            (Ok(()), Err(err)) => Err(err),
            (Err(serve_err), Err(close_err)) => Err(Error::Internal(format!(
                "environment server failed: {serve_err}; close hook failed: {close_err}"
            ))),
        }
    }
}

async fn close_env(
    env: Arc<Mutex<dyn Environment + Send + Sync>>,
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
