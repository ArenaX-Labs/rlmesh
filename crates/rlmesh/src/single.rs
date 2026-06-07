use crate::{CloseResult, Env, ResetRequest, ResetResult, StepRequest, StepResult, spaces};

#[async_trait::async_trait]
pub trait SingleEnv: Send + Sync {
    fn observation_space(&self) -> &spaces::SpaceSpec;
    fn action_space(&self) -> &spaces::SpaceSpec;
    fn env_contract(&self) -> &spaces::EnvContract;

    async fn reset(
        &mut self,
        req: spaces::ResetRequest,
    ) -> std::result::Result<spaces::ResetResult, spaces::EnvRuntimeError>;

    async fn step(
        &mut self,
        req: spaces::StepRequest,
    ) -> std::result::Result<spaces::StepResult, spaces::EnvRuntimeError>;

    async fn render(
        &mut self,
        req: spaces::RenderRequest,
    ) -> std::result::Result<spaces::RenderResult, spaces::EnvRuntimeError>;

    async fn close(
        &mut self,
        req: spaces::CloseRequest,
    ) -> std::result::Result<spaces::CloseResult, spaces::EnvRuntimeError>;
}

pub struct SingleEnvAdapter<E> {
    inner: E,
}

impl<E> SingleEnvAdapter<E> {
    pub fn new(inner: E) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> E {
        self.inner
    }

    pub fn inner(&self) -> &E {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut E {
        &mut self.inner
    }
}

#[async_trait::async_trait]
impl<E: SingleEnv> Env for SingleEnvAdapter<E> {
    fn observation_space(&self) -> &spaces::SpaceSpec {
        self.inner.observation_space()
    }

    fn action_space(&self) -> &spaces::SpaceSpec {
        self.inner.action_space()
    }

    fn num_envs(&self) -> usize {
        1
    }

    fn env_contract(&self) -> &spaces::EnvContract {
        self.inner.env_contract()
    }

    async fn reset(
        &mut self,
        req: ResetRequest,
    ) -> std::result::Result<ResetResult, spaces::EnvRuntimeError> {
        let result = self
            .inner
            .reset(spaces::ResetRequest {
                seed: req.seeds.first().copied(),
                options: req.options,
                timeout_ms: req.timeout_ms,
            })
            .await?;

        Ok(ResetResult {
            observations: result.observation.into_iter().collect(),
            info: result.info,
            episode_ids: result.episode_id.into_iter().collect(),
        })
    }

    async fn step(
        &mut self,
        req: StepRequest,
    ) -> std::result::Result<StepResult, spaces::EnvRuntimeError> {
        let result = self
            .inner
            .step(spaces::StepRequest {
                action: req.actions.into_iter().next(),
                timeout_ms: req.timeout_ms,
            })
            .await?;

        Ok(StepResult {
            observations: result.observation.into_iter().collect(),
            rewards: vec![result.reward],
            terminated: vec![result.terminated],
            truncated: vec![result.truncated],
            info: result.info,
            completed_episodes: vec![],
            episode_ids: vec![],
        })
    }

    async fn render(
        &mut self,
        req: spaces::RenderRequest,
    ) -> std::result::Result<spaces::RenderResult, spaces::EnvRuntimeError> {
        self.inner.render(req).await
    }

    async fn close(
        &mut self,
        req: spaces::CloseRequest,
    ) -> std::result::Result<CloseResult, spaces::EnvRuntimeError> {
        let _ = self.inner.close(req).await?;
        Ok(CloseResult {
            final_episodes: vec![],
        })
    }
}
