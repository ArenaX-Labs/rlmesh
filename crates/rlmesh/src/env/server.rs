use std::sync::Arc;

use rlmesh_grpc::env::Environment;
use rlmesh_grpc::lifecycle::{await_close_with_timeout, start_idle_shutdown};

use super::wire::{WireEnvAdapter, WireLaneAdapter};
use super::{Env, VectorEnv};
use crate::bound::BoundListener;
use crate::{BindAddress, EnvironmentError, Error, Result, ServeOptions};

/// Hosts a scalar [`Env`] as a gRPC environment server.
///
/// Construct with [`EnvServer::new`], then either [`bind`](EnvServer::bind) to
/// reserve the socket and learn the resolved address before serving, or
/// [`serve`](EnvServer::serve) to bind and run in one call.
///
/// [`EnvServer::lanes`] serves several instances as the lanes of one
/// `num_envs = N` endpoint instead, each on its own thread.
pub struct EnvServer<E: Env> {
    envs: Vec<E>,
}

impl<E: Env> EnvServer<E> {
    /// Wrap an [`Env`] implementation to be served.
    pub fn new(env: E) -> Self {
        Self { envs: vec![env] }
    }

    /// Serve `envs` as lanes `0..envs.len()` of one `num_envs = N` endpoint
    /// (see [`LaneEnv`](super::LaneEnv)): each lane owns its env on its own
    /// thread, so the lanes run concurrently. Every lane must carry the same
    /// contract; binding fails on an empty `envs` or on a lane that disagrees.
    pub fn lanes(envs: Vec<E>) -> Self {
        Self { envs }
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
    ///
    /// `RLMESH_ENV_ENDPOINT_TOKEN`, when set, overrides
    /// [`ServeOptions::token`] and is captured at bind time; an empty,
    /// whitespace-only, or non-Unicode value fails instead of disabling
    /// authentication.
    pub async fn bind_with_options(
        self,
        addr: BindAddress,
        options: ServeOptions,
    ) -> Result<BoundEnvServer> {
        if self.envs.is_empty() {
            return Err(Error::Server(
                "an env server needs at least one lane".to_string(),
            ));
        }
        // A scalar env is the one-lane case of the lane server: same wire,
        // same code path as `num_envs > 1`.
        let env =
            WireLaneAdapter::new(self.envs).map_err(|err| Error::Internal(err.to_string()))?;
        bind_environment(env, addr, options).await
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
}

/// Hosts a [`VectorEnv`] as a gRPC environment server.
///
/// Use this only when the endpoint intentionally owns multiple environment
/// lanes in one process. The scalar [`EnvServer`] is the default.
pub struct VectorEnvServer<E: VectorEnv> {
    env: E,
}

impl<E: VectorEnv> VectorEnvServer<E> {
    /// Wrap a [`VectorEnv`] implementation to be served.
    pub fn new(env: E) -> Self {
        Self { env }
    }
}

impl<E: VectorEnv + 'static> VectorEnvServer<E> {
    /// Bind the server to `addr` without yet serving.
    pub async fn bind(self, addr: BindAddress) -> Result<BoundEnvServer> {
        self.bind_with_options(addr, ServeOptions::default()).await
    }

    /// Bind the server to `addr` with explicit [`ServeOptions`].
    ///
    /// `RLMESH_ENV_ENDPOINT_TOKEN`, when set, overrides
    /// [`ServeOptions::token`] and is captured at bind time; an empty,
    /// whitespace-only, or non-Unicode value fails instead of disabling
    /// authentication.
    pub async fn bind_with_options(
        self,
        addr: BindAddress,
        options: ServeOptions,
    ) -> Result<BoundEnvServer> {
        bind_environment(WireEnvAdapter::new(self.env), addr, options).await
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
) -> Result<BoundEnvServer> {
    let options = options.with_env_endpoint_token()?;
    let shutdown = rlmesh_grpc::lifecycle::ShutdownTrigger::new();
    let activity_tx = start_idle_shutdown(options.idle_timeout, shutdown.clone());
    let drain_timeout = options.drain_timeout;
    let close_timeout = options.close_timeout;
    let grpc_options = rlmesh_grpc::ServeOptions::from(options);

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
    let router = tonic::transport::Server::builder()
        .add_service(health_service)
        .add_service(service);
    let env: Arc<dyn Environment + Send + Sync> = env;

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

/// An [`EnvServer`] that has bound its listener but not yet started serving.
///
/// Created by [`EnvServer::bind`] / [`EnvServer::bind_with_options`]. Use
/// [`BoundEnvServer::local_addr`] to read the resolved bind address, then
/// [`BoundEnvServer::serve`] to run until shutdown.
pub struct BoundEnvServer {
    listener: BoundListener,
    router: tonic::transport::server::Router,
    shutdown: rlmesh_grpc::lifecycle::ShutdownTrigger,
    env: Arc<dyn Environment + Send + Sync>,
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
            .map_err(|err| Error::Environment(EnvironmentError::from(err)))
    };
    await_close_with_timeout(close, close_timeout)
        .await
        .map_err(Error::Timeout)?
}
