use crate::{CloseResult, Env, ResetRequest, ResetResult, StepRequest, StepResult, spaces};

/// A single (non-vectorized) environment.
///
/// Implement this when your environment steps exactly one sub-environment at a
/// time — its `reset`/`step` use the scalar single-env request family under
/// [`spaces::request`](crate::spaces::request) (`seed: Option`, `action:
/// Option`) rather than the vectorized [`Env`] batches. Wrap an implementation
/// in [`SingleEnvAdapter`] to get an [`Env`] you can host with
/// [`EnvServer`](crate::EnvServer).
#[async_trait::async_trait]
pub trait SingleEnv: Send + Sync {
    /// The space a single observation belongs to.
    fn observation_space(&self) -> &spaces::SpaceSpec;
    /// The space a single action belongs to.
    fn action_space(&self) -> &spaces::SpaceSpec;
    /// The environment contract (spaces, id, render mode, metadata).
    fn env_contract(&self) -> &spaces::EnvContract;

    /// Reset the environment and return the initial observation.
    async fn reset(
        &mut self,
        req: spaces::request::ResetRequest,
    ) -> std::result::Result<spaces::request::ResetResult, spaces::EnvRuntimeError>;

    /// Apply one action and return the resulting transition.
    async fn step(
        &mut self,
        req: spaces::request::StepRequest,
    ) -> std::result::Result<spaces::request::StepResult, spaces::EnvRuntimeError>;

    /// Produce a render frame for the current state.
    async fn render(
        &mut self,
        req: spaces::RenderRequest,
    ) -> std::result::Result<spaces::RenderResult, spaces::EnvRuntimeError>;

    /// Release resources held by this environment.
    async fn close(
        &mut self,
        req: spaces::CloseRequest,
    ) -> std::result::Result<spaces::request::CloseResult, spaces::EnvRuntimeError>;
}

/// Adapts a [`SingleEnv`] into a vectorized [`Env`] with `num_envs() == 1`.
///
/// The adapter translates between the scalar single-env request family and the
/// batched env-layer one: it unwraps the first seed/action on the way in and
/// wraps the single observation/reward back into one-element batches on the way
/// out.
///
/// A vector of one is the canonical server-side truth — "single env" is just the
/// client-side un-batching view. A single/scalar env carries `DISABLED` autoreset
/// (it has no `metadata["autoreset_mode"]`), so the runtime resets it explicitly;
/// the per-lane `on_lane_reset` event covers its single lane. Collapsing the
/// remaining single/vector construction fork (`uses_single_env_api`) is a later,
/// non-breaking internal change.
pub struct SingleEnvAdapter<E> {
    inner: E,
}

impl<E> SingleEnvAdapter<E> {
    /// Wrap a [`SingleEnv`] implementation.
    pub fn new(inner: E) -> Self {
        Self { inner }
    }

    /// Consume the adapter and return the wrapped environment.
    pub fn into_inner(self) -> E {
        self.inner
    }

    /// Borrow the wrapped environment.
    pub fn inner(&self) -> &E {
        &self.inner
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
            .reset(spaces::request::ResetRequest {
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
            .step(spaces::request::StepRequest {
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
