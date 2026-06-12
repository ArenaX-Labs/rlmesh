use super::handler::ModelHandler;
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

impl<H: ModelHandler + 'static> ModelWorker<H> {
    pub fn run_local(self, env_address: &str) -> Result<()> {
        self.run_local_to(ConnectAddress::parse(env_address)?)
    }

    pub fn run_local_for_episodes(self, env_address: &str, max_episodes: u64) -> Result<()> {
        self.run_local_to_with_max_episodes(ConnectAddress::parse(env_address)?, Some(max_episodes))
    }

    pub fn run_local_to(self, env_address: ConnectAddress) -> Result<()> {
        self.run_local_to_with_max_episodes(env_address, None)
    }

    fn run_local_to_with_max_episodes(
        self,
        env_address: ConnectAddress,
        max_episodes: Option<u64>,
    ) -> Result<()> {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|err| Error::Internal(format!("failed to create tokio runtime: {err}")))?;
        runtime.block_on(self.run_local_to_async_with_max_episodes(env_address, max_episodes))
    }

    pub async fn run_local_async(self, env_address: &str) -> Result<()> {
        self.run_local_to_async(ConnectAddress::parse(env_address)?)
            .await
    }

    pub async fn run_local_to_async(self, env_address: ConnectAddress) -> Result<()> {
        self.run_local_to_async_with_max_episodes(env_address, None)
            .await
    }

    pub async fn run_local_async_for_episodes(
        self,
        env_address: &str,
        max_episodes: u64,
    ) -> Result<()> {
        self.run_local_to_async_for_episodes(ConnectAddress::parse(env_address)?, max_episodes)
            .await
    }

    pub async fn run_local_to_async_for_episodes(
        self,
        env_address: ConnectAddress,
        max_episodes: u64,
    ) -> Result<()> {
        self.run_local_to_async_with_max_episodes(env_address, Some(max_episodes))
            .await
    }

    async fn run_local_to_async_with_max_episodes(
        mut self,
        env_address: ConnectAddress,
        max_episodes: Option<u64>,
    ) -> Result<()> {
        let result = local::run_local_to_async_with_max_episodes(
            &mut self.handler,
            env_address,
            max_episodes,
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

    pub fn serve(self, address: &str, token: &str) -> Result<()> {
        self.serve_to(BindAddress::parse(address)?, token)
    }

    pub fn serve_to(self, address: BindAddress, token: &str) -> Result<()> {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|err| Error::Internal(format!("failed to create tokio runtime: {err}")))?;
        runtime.block_on(self.serve_to_async(address, token))
    }

    pub async fn serve_async(self, address: &str, token: &str) -> Result<()> {
        self.serve_to_async(BindAddress::parse(address)?, token)
            .await
    }

    pub async fn serve_to_async(self, address: BindAddress, token: &str) -> Result<()> {
        self.serve_to_async_with_options(address, token, ServeOptions::default())
            .await
    }

    pub fn serve_with_options(
        self,
        address: &str,
        token: &str,
        options: ServeOptions,
    ) -> Result<()> {
        self.serve_to_with_options(BindAddress::parse(address)?, token, options)
    }

    pub fn serve_to_with_options(
        self,
        address: BindAddress,
        token: &str,
        options: ServeOptions,
    ) -> Result<()> {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|err| Error::Internal(format!("failed to create tokio runtime: {err}")))?;
        runtime.block_on(self.serve_to_async_with_options(address, token, options))
    }

    pub async fn serve_async_with_options(
        self,
        address: &str,
        token: &str,
        options: ServeOptions,
    ) -> Result<()> {
        self.serve_to_async_with_options(BindAddress::parse(address)?, token, options)
            .await
    }

    pub async fn serve_to_async_with_options(
        self,
        address: BindAddress,
        token: &str,
        options: ServeOptions,
    ) -> Result<()> {
        server::serve_model_with_options(self.handler, address, token, options).await
    }
}
