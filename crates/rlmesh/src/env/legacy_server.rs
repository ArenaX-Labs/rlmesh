use std::sync::Arc;

use rlmesh_grpc::env::Environment;
use rlmesh_grpc::lifecycle::{await_close_with_timeout, start_idle_shutdown};

use super::legacy::{Env, VectorEnv};
use super::wire::{WireEnvAdapter, WireLaneAdapter};
use crate::bound::BoundListener;
use crate::{BindAddress, EnvironmentError, Error, Result, ServeOptions};

/// Hosts a scalar [`Env`] as a gRPC environment server.
///
/// Construct with [`WireEnvServer::new`], then either [`bind`](WireEnvServer::bind) to
/// reserve the socket and learn the resolved address before serving, or
/// [`serve`](WireEnvServer::serve) to bind and run in one call.
///
/// [`WireEnvServer::lanes`] serves several instances as the lanes of one
/// `num_envs = N` endpoint instead, each on its own thread.
pub struct WireEnvServer<E: Env> {
    envs: Vec<E>,
}

impl<E: Env> WireEnvServer<E> {
    /// Wrap an [`Env`] implementation to be served.
    pub fn new(env: E) -> Self {
        Self { envs: vec![env] }
    }

    /// Serve `envs` as lanes `0..envs.len()` of one `num_envs = N` endpoint
    /// with each lane owning its env on its own
    /// thread, so the lanes run concurrently. Every lane must carry the same
    /// contract; binding fails on an empty `envs` or on a lane that disagrees.
    pub fn lanes(envs: Vec<E>) -> Self {
        Self { envs }
    }
}

impl<E: Env + 'static> WireEnvServer<E> {
    /// Bind the server to `addr` without yet serving.
    ///
    /// The returned [`BoundWireEnvServer`] exposes [`BoundWireEnvServer::local_addr`]
    /// so callers can learn the resolved address (e.g. the OS-assigned port
    /// when binding to port 0) before awaiting shutdown.
    pub async fn bind(self, addr: BindAddress) -> Result<BoundWireEnvServer> {
        self.bind_with_options(addr, ServeOptions::default()).await
    }

    /// Bind the server to `addr` with explicit [`ServeOptions`].
    ///
    /// `RLMESH_ENV_ENDPOINT_TOKEN`, when set, is captured at bind time; an
    /// empty, whitespace-only, or non-Unicode value fails instead of
    /// disabling authentication.
    pub async fn bind_with_options(
        self,
        addr: BindAddress,
        options: ServeOptions,
    ) -> Result<BoundWireEnvServer> {
        self.bind_with_ext(addr, options, crate::ext::EnvServerOptions::default())
            .await
    }

    /// Bind with the fenced extras (token, peer-info entries).
    pub async fn bind_with_ext(
        self,
        addr: BindAddress,
        options: ServeOptions,
        ext: crate::ext::EnvServerOptions,
    ) -> Result<BoundWireEnvServer> {
        if self.envs.is_empty() {
            return Err(Error::server(
                "an env server needs at least one lane".to_string(),
            ));
        }
        // A scalar env is the one-lane case of the lane server: same wire,
        // same code path as `num_envs > 1`.
        let env =
            WireLaneAdapter::new(self.envs).map_err(|err| Error::Internal(err.to_string()))?;
        bind_environment(env, addr, options, ext).await
    }

    /// Bind to `addr` and serve until shutdown, with default [`ServeOptions`].
    ///
    /// Equivalent to [`bind`](WireEnvServer::bind) followed by
    /// [`BoundWireEnvServer::serve`], for callers that do not need the resolved
    /// address up front.
    pub async fn serve(self, addr: BindAddress) -> Result<()> {
        self.serve_with_options(addr, ServeOptions::default()).await
    }

    /// Bind to `addr` and serve until shutdown, with explicit [`ServeOptions`].
    pub async fn serve_with_options(self, addr: BindAddress, options: ServeOptions) -> Result<()> {
        self.bind_with_options(addr, options).await?.serve().await
    }
}

/// Hosts a [`VectorEnv`] as a gRPC environment server.
///
/// Use this only when the endpoint intentionally owns multiple environment
/// lanes in one process. The scalar [`WireEnvServer`] is the default.
pub struct WireVectorEnvServer<E: VectorEnv> {
    env: E,
}

impl<E: VectorEnv> WireVectorEnvServer<E> {
    /// Wrap a [`VectorEnv`] implementation to be served.
    pub fn new(env: E) -> Self {
        Self { env }
    }
}

impl<E: VectorEnv + 'static> WireVectorEnvServer<E> {
    /// Bind the server to `addr` without yet serving.
    pub async fn bind(self, addr: BindAddress) -> Result<BoundWireEnvServer> {
        self.bind_with_options(addr, ServeOptions::default()).await
    }

    /// Bind the server to `addr` with explicit [`ServeOptions`].
    ///
    /// `RLMESH_ENV_ENDPOINT_TOKEN`, when set, is captured at bind time; an
    /// empty, whitespace-only, or non-Unicode value fails instead of
    /// disabling authentication.
    pub async fn bind_with_options(
        self,
        addr: BindAddress,
        options: ServeOptions,
    ) -> Result<BoundWireEnvServer> {
        self.bind_with_ext(addr, options, crate::ext::EnvServerOptions::default())
            .await
    }

    /// Bind with the fenced extras (token, peer-info entries).
    pub async fn bind_with_ext(
        self,
        addr: BindAddress,
        options: ServeOptions,
        ext: crate::ext::EnvServerOptions,
    ) -> Result<BoundWireEnvServer> {
        bind_environment(WireEnvAdapter::new(self.env), addr, options, ext).await
    }

    /// Bind to `addr` and serve until shutdown, with default [`ServeOptions`].
    pub async fn serve(self, addr: BindAddress) -> Result<()> {
        self.serve_with_options(addr, ServeOptions::default()).await
    }

    /// Bind to `addr` and serve until shutdown, with explicit [`ServeOptions`].
    pub async fn serve_with_options(self, addr: BindAddress, options: ServeOptions) -> Result<()> {
        self.bind_with_options(addr, options).await?.serve().await
    }
}

/// Bind a wire environment: the listener, the EnvService and health services,
/// and the shutdown lifecycle every env server shares.
async fn bind_environment<T: Environment + Send + Sync + 'static>(
    env: T,
    addr: BindAddress,
    options: ServeOptions,
    ext: crate::ext::EnvServerOptions,
) -> Result<BoundWireEnvServer> {
    let token = crate::serve_options::env_endpoint_token(ext.token)?;
    let shutdown = rlmesh_grpc::lifecycle::ShutdownTrigger::new();
    let activity_tx = start_idle_shutdown(options.idle_timeout, shutdown.clone());
    let drain_timeout = options.drain_timeout;
    let close_timeout = options.close_timeout;
    let grpc_options = options.to_grpc(crate::serve_options::WireExtras {
        token,
        predict_concurrency: None,
        peer_info_extra: ext.peer_info_extra,
    });

    let listener = BoundListener::bind(addr).await?;
    let local_addr = listener.local_addr()?;

    let env = Arc::new(env);
    let service = rlmesh_grpc::env::env_service_from_shared(
        Arc::clone(&env),
        shutdown.clone(),
        grpc_options,
        activity_tx,
    );
    let (_health_reporter, health_service) = rlmesh_grpc::health::serving_health_service().await;
    let router = rlmesh_grpc::lifecycle::server_builder()
        .add_service(health_service)
        .add_service(service);
    let env: Arc<dyn Environment + Send + Sync> = env;

    Ok(BoundWireEnvServer {
        listener,
        router,
        shutdown,
        env,
        local_addr,
        drain_timeout,
        close_timeout,
    })
}

/// An [`WireEnvServer`] that has bound its listener but not yet started serving.
///
/// Created by [`WireEnvServer::bind`] / [`WireEnvServer::bind_with_options`]. Use
/// [`BoundWireEnvServer::local_addr`] to read the resolved bind address, then
/// [`BoundWireEnvServer::serve`] to run until shutdown.
pub struct BoundWireEnvServer {
    listener: BoundListener,
    router: tonic::transport::server::Router,
    shutdown: rlmesh_grpc::lifecycle::ShutdownTrigger,
    env: Arc<dyn Environment + Send + Sync>,
    local_addr: BindAddress,
    drain_timeout: Option<std::time::Duration>,
    close_timeout: Option<std::time::Duration>,
}

impl BoundWireEnvServer {
    /// The resolved address the server is bound to (the OS-assigned port for
    /// TCP port 0).
    pub fn local_addr(&self) -> &BindAddress {
        &self.local_addr
    }

    /// A handle that stops [`serve`](Self::serve) from outside (a signal, a host
    /// shutdown), draining in-flight requests first. The environment close hook
    /// still runs.
    pub fn shutdown_trigger(&self) -> rlmesh_grpc::lifecycle::ShutdownTrigger {
        self.shutdown.clone()
    }

    /// Serve until shutdown, then run the environment close hook.
    pub async fn serve(self) -> Result<()> {
        let serve_result = self
            .listener
            .serve(self.router, self.shutdown, self.drain_timeout)
            .await;
        let close_result = close_env(self.env, self.close_timeout).await;
        crate::error::join_results(serve_result, close_result, "environment server failed")
    }
}

async fn close_env(
    env: Arc<dyn Environment + Send + Sync>,
    close_timeout: Option<std::time::Duration>,
) -> Result<()> {
    let close = async {
        env.close()
            .await
            .map(|_| ())
            .map_err(|err| Error::Environment(EnvironmentError::from_wire(err)))
    };
    await_close_with_timeout(close, close_timeout)
        .await
        .map_err(Error::Timeout)?
}
