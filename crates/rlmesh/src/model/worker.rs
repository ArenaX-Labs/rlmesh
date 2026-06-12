use super::handler::ModelHandler;
use super::server::BoundModelServer;
use super::{local, server};
use crate::{BindAddress, ConnectAddress, Error, Result, ServeOptions};

pub struct ModelWorker<H> {
    handler: H,
}

impl<H> ModelWorker<H> {
    pub fn new(handler: H) -> Self {
        Self { handler }
    }
}

/// Options for [`ModelWorker::run_local`] / [`ModelWorker::run_local_async`].
///
/// Build with [`RunLocalOptions::new`] (or `RunLocalOptions::parse` from a
/// string address) and the chaining setters covering the run axes:
/// `for_episodes` (run a bounded number of episodes) and `base_seed`
/// (deterministic env seeding).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunLocalOptions {
    /// Address of the environment server to connect to.
    pub env_address: ConnectAddress,
    /// Stop after this many episodes; `None` runs until the env ends.
    pub max_episodes: Option<u64>,
    /// Base seed threaded into the runtime session for deterministic env
    /// reset seeding; `None` leaves seeding to the env.
    pub base_seed: Option<i64>,
}

impl RunLocalOptions {
    /// Run against `env_address` until the environment ends.
    pub fn new(env_address: ConnectAddress) -> Self {
        Self {
            env_address,
            max_episodes: None,
            base_seed: None,
        }
    }

    /// Parse a string env address (e.g. `"tcp://host:50051"`).
    pub fn parse(env_address: &str) -> Result<Self> {
        Ok(Self::new(ConnectAddress::parse(env_address)?))
    }

    /// Stop after `max_episodes` episodes.
    pub fn for_episodes(mut self, max_episodes: u64) -> Self {
        self.max_episodes = Some(max_episodes);
        self
    }

    /// Set the base seed used for deterministic env reset seeding.
    pub fn base_seed(mut self, base_seed: i64) -> Self {
        self.base_seed = Some(base_seed);
        self
    }
}

impl From<ConnectAddress> for RunLocalOptions {
    fn from(env_address: ConnectAddress) -> Self {
        Self::new(env_address)
    }
}

/// Options for [`ModelWorker::serve`] / [`ModelWorker::serve_async`] /
/// [`ModelWorker::bind_async`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServeModelOptions {
    /// Address to bind the model server to.
    pub address: BindAddress,
    /// Bearer token required on requests; empty/`""` disables auth.
    pub token: String,
    /// Transport serve options (idle/drain/close timeouts, remote shutdown).
    pub serve: ServeOptions,
}

impl ServeModelOptions {
    /// Serve on `address` with no token and default serve options.
    pub fn new(address: BindAddress) -> Self {
        Self {
            address,
            token: String::new(),
            serve: ServeOptions::default(),
        }
    }

    /// Parse a string bind address (e.g. `"tcp://0.0.0.0:50061"`).
    pub fn parse(address: &str) -> Result<Self> {
        Ok(Self::new(BindAddress::parse(address)?))
    }

    /// Require `token` on the `authorization` header (empty disables auth).
    pub fn token(mut self, token: impl Into<String>) -> Self {
        self.token = token.into();
        self
    }

    /// Set the transport serve options.
    pub fn serve_options(mut self, serve: ServeOptions) -> Self {
        self.serve = serve;
        self
    }
}

impl From<BindAddress> for ServeModelOptions {
    fn from(address: BindAddress) -> Self {
        Self::new(address)
    }
}

impl<H: ModelHandler + 'static> ModelWorker<H> {
    /// Run the handler in-process against a remote environment (blocking).
    ///
    /// Drives the model/env loop on a private Tokio runtime until the env ends
    /// (or `options.max_episodes` episodes complete).
    pub fn run_local(self, options: impl Into<RunLocalOptions>) -> Result<()> {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|err| Error::Internal(format!("failed to create tokio runtime: {err}")))?;
        runtime.block_on(self.run_local_async(options))
    }

    /// Async variant of [`ModelWorker::run_local`].
    pub async fn run_local_async(mut self, options: impl Into<RunLocalOptions>) -> Result<()> {
        let options = options.into();
        let result = local::run_local(
            &mut self.handler,
            options.env_address,
            options.max_episodes,
            options.base_seed,
        )
        .await;
        let close_result = self.handler.on_close().await;
        match (result, close_result) {
            (Ok(_), Ok(())) => Ok(()),
            (Err(err), Ok(())) => Err(err),
            (Ok(_), Err(err)) => Err(err),
            (Err(run_err), Err(close_err)) => Err(Error::Internal(format!(
                "local model run failed: {run_err}; close hook failed: {close_err}"
            ))),
        }
    }

    /// Serve the handler as a model endpoint (blocking).
    pub fn serve(self, options: impl Into<ServeModelOptions>) -> Result<()> {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|err| Error::Internal(format!("failed to create tokio runtime: {err}")))?;
        runtime.block_on(self.serve_async(options))
    }

    /// Async variant of [`ModelWorker::serve`].
    pub async fn serve_async(self, options: impl Into<ServeModelOptions>) -> Result<()> {
        self.bind_async(options).await?.serve().await
    }

    /// Bind the model server without yet serving.
    ///
    /// The returned [`BoundModelServer`] exposes its resolved address via
    /// [`BoundModelServer::local_addr`] (e.g. the OS-assigned port for TCP port
    /// 0) before [`BoundModelServer::serve`] is awaited.
    pub async fn bind_async(
        self,
        options: impl Into<ServeModelOptions>,
    ) -> Result<BoundModelServer> {
        let options = options.into();
        server::bind_model_with_options(
            self.handler,
            options.address,
            &options.token,
            options.serve,
        )
        .await
    }
}
