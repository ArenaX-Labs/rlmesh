use std::sync::Arc;

use rlmesh_runtime::RuntimeReport;

use super::handler::ModelHandler;
use super::server::BoundModelServer;
use super::{local, server};
use crate::{BindAddress, ConnectAddress, Error, Result, ServeOptions};

/// Drives or serves a [`ModelHandler`].
///
/// Construct with [`ModelWorker::new`], then pick a mode:
///
/// - [`run_local`](ModelWorker::run_local) / [`run_local_async`](ModelWorker::run_local_async):
///   connect to a remote env and run the model/env loop in-process.
/// - [`serve`](ModelWorker::serve) / [`serve_async`](ModelWorker::serve_async) /
///   [`bind_async`](ModelWorker::bind_async): host the handler as a model
///   endpoint that an orchestrator joins.
pub struct ModelWorker<H> {
    handler: H,
}

impl<H> ModelWorker<H> {
    /// Wrap a [`ModelHandler`] to be driven or served.
    pub fn new(handler: H) -> Self {
        Self { handler }
    }
}

/// Options for [`ModelWorker::run_local`] / [`ModelWorker::run_local_async`].
///
/// Build with [`RunLocalOptions::new`] (or `RunLocalOptions::parse` from a
/// string address) and the chaining setters covering the run axes:
/// `for_episodes` (run a bounded number of episodes), `base_seed` /
/// `episode_seeds` (deterministic env seeding), the episode caps, and
/// `execution_horizon` (action chunking) with `prefetch_lead` (async
/// inference over the chunk replay).
#[derive(Debug, Clone, PartialEq)]
pub struct RunLocalOptions {
    /// Address of the environment server to connect to.
    pub env_address: ConnectAddress,
    /// Bearer token sent on every request to that environment server;
    /// empty/`""` connects unauthenticated.
    pub token: String,
    /// Stop after this many episodes; `None` runs until the env ends.
    pub max_episodes: Option<u64>,
    /// Base seed threaded into the runtime session for deterministic env
    /// reset seeding; `None` leaves seeding to the env.
    pub base_seed: Option<i64>,
    /// Explicit per-episode reset seeds, consumed in episode-start order.
    /// Overrides `base_seed` when non-empty; requires an env with autoreset
    /// disabled (see `RuntimeSessionSpec::episode_seeds`).
    pub episode_seeds: Vec<i64>,
    /// First trial ordinal this run's episodes walk; `None` walks from 0. The
    /// runtime mints one ordinal per episode and delivers it as
    /// `reset(options={"trial_index": k})` to an env that declared the option;
    /// a non-zero base requires an env with autoreset disabled (see
    /// `RuntimeSessionSpec::trial_index_base`).
    pub trial_index_base: Option<u64>,
    /// Truncate any episode after this many steps (reported `truncated`).
    /// Requires an env with autoreset disabled.
    pub max_episode_steps: Option<i64>,
    /// Truncate any episode after this wall-clock duration in seconds
    /// (reported `truncated`). Requires an env with autoreset disabled.
    pub max_episode_seconds: Option<f64>,
    /// Ask the env to close when the run ends.
    pub close_env: bool,
    /// Fire the handler's `on_close` when the run ends (the default: the
    /// worker is consumed, so the model is done). Off when the handler
    /// outlives the run -- a borrowed model the caller closes itself.
    pub close_model: bool,
    /// How many actions of each predicted chunk the runtime executes before
    /// re-planning (1 = no chunking). Pinned onto the route at resolve, exactly
    /// as the served path pins it via `ResolveAdapter`.
    pub execution_horizon: u32,
    /// Async inference: with this many (or fewer) replay frames of the current
    /// chunk left, the driver predicts the next chunk while they execute (see
    /// `RuntimeDriver::with_prefetch`). The prefetched chunk is conditioned on
    /// an observation up to `prefetch_lead` steps stale, and one prefetched
    /// across an episode boundary is discarded. 0 = synchronous (predict only
    /// when no frame is left to play).
    pub prefetch_lead: u32,
    /// The workflow edition this run **declares** (its WANT): the runtime tier's
    /// explicit pin, so the session runs at this edition even when this build
    /// and the env could both go higher. `None` (the default) declares nothing
    /// and wants this build's current edition, which is its own `max(can)` and
    /// therefore caps no env.
    ///
    /// Any cohort spelling of an edition this build retains is accepted; a pin
    /// no peer can run is refused by the env-leg negotiation before any Join
    /// stream opens, naming every tier's WANT and CAN.
    pub workflow_edition: Option<String>,
}

impl RunLocalOptions {
    /// Run against `env_address` until the environment ends.
    pub fn new(env_address: ConnectAddress) -> Self {
        Self {
            env_address,
            token: String::new(),
            max_episodes: None,
            base_seed: None,
            episode_seeds: Vec::new(),
            trial_index_base: None,
            max_episode_steps: None,
            max_episode_seconds: None,
            close_env: false,
            close_model: true,
            execution_horizon: 1,
            prefetch_lead: 0,
            workflow_edition: None,
        }
    }

    /// Parse a string env address (e.g. `"tcp://host:50051"`).
    pub fn parse(env_address: &str) -> Result<Self> {
        Ok(Self::new(ConnectAddress::parse(env_address)?))
    }

    /// Send `token` on the `authorization` header of every env request (empty
    /// connects unauthenticated).
    pub fn token(mut self, token: impl Into<String>) -> Self {
        self.token = token.into();
        self
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

    /// Execute `execution_horizon` actions of each predicted chunk before
    /// re-planning (1 = no chunking).
    pub fn execution_horizon(mut self, execution_horizon: u32) -> Self {
        self.execution_horizon = execution_horizon.max(1);
        self
    }

    /// Predict the next chunk while `prefetch_lead` (or fewer) replay frames
    /// of the current one remain (0 = synchronous). The chunk sees an
    /// observation up to `prefetch_lead` steps stale, so the run is not
    /// comparable to the synchronous loop.
    pub fn prefetch_lead(mut self, prefetch_lead: u32) -> Self {
        self.prefetch_lead = prefetch_lead;
        self
    }

    /// Reset each episode with the next seed from `episode_seeds`, in
    /// episode-start order (overrides `base_seed`; autoreset-disabled envs
    /// only).
    pub fn episode_seeds(mut self, episode_seeds: Vec<i64>) -> Self {
        self.episode_seeds = episode_seeds;
        self
    }

    /// Walk trial ordinals from `trial_index_base` instead of 0, one per
    /// episode (a non-zero base needs an autoreset-disabled env).
    pub fn trial_index_base(mut self, trial_index_base: u64) -> Self {
        self.trial_index_base = Some(trial_index_base);
        self
    }

    /// Truncate any episode after `max_episode_steps` steps.
    pub fn max_episode_steps(mut self, max_episode_steps: i64) -> Self {
        self.max_episode_steps = Some(max_episode_steps);
        self
    }

    /// Truncate any episode after `max_episode_seconds` of wall-clock time.
    pub fn max_episode_seconds(mut self, max_episode_seconds: f64) -> Self {
        self.max_episode_seconds = Some(max_episode_seconds);
        self
    }

    /// Ask the env to close when the run ends.
    pub fn close_env(mut self, close_env: bool) -> Self {
        self.close_env = close_env;
        self
    }

    /// Keep the handler open after the run (see [`close_model`](Self::close_model)).
    pub fn close_model(mut self, close_model: bool) -> Self {
        self.close_model = close_model;
        self
    }

    /// Declare the workflow edition this run is authored against (the runtime
    /// tier's WANT). See [`RunLocalOptions::workflow_edition`].
    pub fn workflow_edition(mut self, workflow_edition: impl Into<String>) -> Self {
        self.workflow_edition = Some(workflow_edition.into());
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
    ///
    /// A non-empty [`ServeOptions::token`](crate::ServeOptions::token) in
    /// `serve` takes precedence over this field; either one alone arms auth.
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
    ///
    /// A non-empty [`ServeOptions::token`](crate::ServeOptions::token) set
    /// through [`serve_options`](Self::serve_options) wins over this one.
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
    /// (or `options.max_episodes` episodes complete). Returns the session's
    /// [`RuntimeReport`].
    ///
    /// # Panics
    ///
    /// This is a blocking entrypoint: it builds its own Tokio runtime, so
    /// calling it from inside one panics. From async code call
    /// [`run_local_async`](ModelWorker::run_local_async) instead.
    pub fn run_local(self, options: impl Into<RunLocalOptions>) -> Result<RuntimeReport> {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|err| Error::Internal(format!("failed to create tokio runtime: {err}")))?;
        runtime.block_on(self.run_local_async(options))
    }

    /// Async variant of [`ModelWorker::run_local`].
    pub async fn run_local_async(
        self,
        options: impl Into<RunLocalOptions>,
    ) -> Result<RuntimeReport> {
        self.run_local_cancellable_async(options, tokio_util::sync::CancellationToken::new())
            .await
    }

    /// [`run_local_async`](ModelWorker::run_local_async) with an external
    /// cancellation token: cancelling it aborts the run between operations
    /// (the driver returns a cancellation error) with the close hook still
    /// fired -- the seam a host binding uses to deliver e.g. Ctrl-C.
    pub async fn run_local_cancellable_async(
        self,
        options: impl Into<RunLocalOptions>,
        cancellation: tokio_util::sync::CancellationToken,
    ) -> Result<RuntimeReport> {
        self.run_local_hooked_async(
            options,
            cancellation,
            Arc::new(rlmesh_runtime::NoopRuntimeHooks),
        )
        .await
    }

    /// [`run_local_cancellable_async`](ModelWorker::run_local_cancellable_async)
    /// with the runtime's hook seam: every per-episode event of the local
    /// session reaches `hooks`, awaited inline by the driver.
    pub async fn run_local_hooked_async(
        mut self,
        options: impl Into<RunLocalOptions>,
        cancellation: tokio_util::sync::CancellationToken,
        hooks: Arc<dyn rlmesh_runtime::RuntimeHooks>,
    ) -> Result<RuntimeReport> {
        let options = options.into();
        let close_model = options.close_model;
        let result = local::run_local(&mut self.handler, options, cancellation, hooks).await;
        if !close_model {
            return result;
        }
        let close_result = self.handler.on_close().await;
        crate::error::join_results(result, close_result, "local model run failed")
    }

    /// Serve the handler as a model endpoint (blocking).
    ///
    /// # Panics
    ///
    /// This is a blocking entrypoint: it builds its own Tokio runtime, so
    /// calling it from inside one panics. From async code call
    /// [`serve_async`](ModelWorker::serve_async) (or
    /// [`bind_async`](ModelWorker::bind_async)) instead.
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
    ///
    /// The bearer token is taken from `options.serve.token` when it is set and
    /// non-empty, otherwise from `options.token`, so auth configured either way
    /// is enforced.
    pub async fn bind_async(
        self,
        options: impl Into<ServeModelOptions>,
    ) -> Result<BoundModelServer> {
        let options = options.into();
        let effective_token = options
            .serve
            .token
            .clone()
            .filter(|token| !token.is_empty())
            .unwrap_or_else(|| options.token.clone());
        server::bind_model_with_options(
            self.handler,
            options.address,
            &effective_token,
            options.serve,
        )
        .await
    }
}
