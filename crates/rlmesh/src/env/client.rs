use rlmesh_grpc::env::{ResetRequest as ProtoResetRequest, StepRequest as ProtoStepRequest};
use rlmesh_grpc::wire::{
    decode_batched_partial_values, encode_batched_partial_values, meta_map_from_proto,
    meta_map_to_proto, render_request_to_proto, render_result_from_proto,
};

use super::types::{
    CloseResult as VectorCloseResult, RenderRequest, RenderResult,
    ResetRequest as VectorResetRequest, ResetResult as VectorResetResult,
    StepRequest as VectorStepRequest, StepResult as VectorStepResult,
};
use super::wire::{
    proto_episode_metadata_to_public, protocol_error_to_error, validate_action_count,
    validate_count,
};
use crate::{ConnectAddress, EnvironmentError, Error, ErrorCode, Result, spaces};

/// A client handle to a remote vector environment server.
///
/// Connect with [`RemoteVectorEnv::connect`] (or
/// [`RemoteVectorEnv::connect_with_token`] for an authenticated endpoint), then
/// drive the env by hand with [`reset`](RemoteVectorEnv::reset) /
/// [`step`](RemoteVectorEnv::step), the same vectorized request family a
/// [`VectorEnv`](crate::VectorEnv) implementor serves.
pub struct RemoteVectorEnv {
    inner: rlmesh_grpc::EnvClient,
    env_contract: spaces::EnvContract,
    observation_space: std::sync::Arc<spaces::SpaceSpec>,
    action_space: std::sync::Arc<spaces::SpaceSpec>,
    num_envs: usize,
    /// The env's negotiation offer (generations/editions/capabilities) captured
    /// at handshake, for three-way (relay) session-floor reconciliation.
    session_offer: rlmesh_proto::SessionOffer,
}

impl RemoteVectorEnv {
    /// Connect to an env server at `address` and perform the handshake.
    ///
    /// `address` accepts the same forms as
    /// [`ConnectAddress::parse`](crate::ConnectAddress::parse) (e.g.
    /// `tcp://host:port`, `unix:///path.sock`).
    ///
    /// # Errors
    ///
    /// Returns a [`crate::Error`] if the address is invalid, the connection
    /// fails, or the handshake reports an incompatible / malformed env contract.
    pub async fn connect(address: &str) -> Result<Self> {
        Self::connect_to(ConnectAddress::parse(address)?).await
    }

    /// Connect to an env server that requires a bearer token (see
    /// [`ServeOptions::token`](crate::ServeOptions::token)). An empty token
    /// behaves like [`RemoteVectorEnv::connect`].
    pub async fn connect_with_token(address: &str, token: &str) -> Result<Self> {
        Self::connect_to_with_token(ConnectAddress::parse(address)?, token).await
    }

    /// Connect to an already-parsed [`ConnectAddress`].
    pub async fn connect_to(address: ConnectAddress) -> Result<Self> {
        Self::connect_to_with_token(address, "").await
    }

    async fn connect_to_with_token(address: ConnectAddress, token: &str) -> Result<Self> {
        let mut inner = rlmesh_grpc::EnvClient::connect_with_token(&address.to_string(), token)
            .await
            .map_err(Error::from)?;
        let handshake = inner.handshake().await.map_err(Error::from)?;
        let session_offer = handshake.session_offer();
        let env_contract = rlmesh_grpc::wire::env_contract_from_proto(handshake.env_contract)
            .map_err(|error| {
                Error::Internal(format!("invalid spaces spec from remote env: {error}"))
            })?;
        validate_env_contract(&env_contract)?;
        let observation_space = std::sync::Arc::new(
            env_contract
                .observation_space
                .clone()
                .expect("remote env contract was validated during connect"),
        );
        let action_space = std::sync::Arc::new(
            env_contract
                .action_space
                .clone()
                .expect("remote env contract was validated during connect"),
        );
        Ok(Self {
            inner,
            env_contract,
            observation_space,
            action_space,
            num_envs: handshake.num_envs,
            session_offer,
        })
    }

    /// The address this client is connected to.
    pub fn address(&self) -> &str {
        self.inner.address()
    }

    /// The env's negotiation offer captured at handshake (supported protocol
    /// generations, workflow editions, and advertised capabilities).
    ///
    /// Pass this to [`RemoteModel::connect_with_env_offer`](crate::RemoteModel::connect_with_env_offer)
    /// so the runtime can reconcile the three-way (relay) session floor across
    /// env, model, and runtime (see versioning-governance §7).
    pub fn session_offer(&self) -> &rlmesh_proto::SessionOffer {
        &self.session_offer
    }

    /// Tear down the session locally without waiting for a Close round-trip.
    ///
    /// Drops the Join stream; the server completes this session's in-flight
    /// episodes as truncated and frees the exclusive session slot once it
    /// observes the stream end (after any still-draining operation finishes).
    /// Final episode metadata from the server is forfeited. Prefer
    /// [`RemoteVectorEnv::close`]; use this when close cannot complete (e.g. it
    /// timed out behind a long-draining server operation).
    pub fn detach(&mut self) {
        self.inner.detach();
    }

    /// The environment contract reported by the server at handshake (spaces,
    /// id, render mode, metadata).
    pub fn env_contract(&self) -> &spaces::EnvContract {
        &self.env_contract
    }

    /// The number of sub-environments the remote env steps in lockstep.
    pub fn num_envs(&self) -> usize {
        self.num_envs
    }

    /// Reset the remote environment and return the initial observations.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Environment`](crate::Error::Environment) if the env
    /// reports a failure or returns a batch whose size does not match
    /// [`num_envs`](RemoteVectorEnv::num_envs), or a transport error on RPC failure.
    pub async fn reset(&mut self, req: VectorResetRequest) -> Result<VectorResetResult> {
        let observation_space = std::sync::Arc::clone(&self.observation_space);

        let response = self
            .inner
            .reset(ProtoResetRequest {
                seeds: req.seeds,
                options: req.options.as_ref().map(meta_map_to_proto),
                // Native timeout_ms is i64 (>=0); proto field is uint64.
                timeout_ms: req.timeout_ms.max(0) as u64,
                // Forward the requested lanes: empty = whole-vector reset; a
                // non-empty list is a partial reset the server routes to
                // reset_subset (honored only by envs that support it). Lane
                // offsets must be non-negative array indices: reject a negative
                // lane loudly rather than clamping it to 0, which would silently
                // reset the wrong lane and misalign the paired seed.
                env_indices: req
                    .env_indices
                    .into_iter()
                    .map(|index| {
                        u32::try_from(index).map_err(|_| {
                            Error::Environment(EnvironmentError {
                                code: ErrorCode::InvalidAction,
                                message: format!(
                                    "partial reset: env_index {index} must be a \
                                     non-negative lane offset"
                                ),
                                is_recoverable: false,
                            })
                        })
                    })
                    .collect::<Result<Vec<u32>>>()?,
            })
            .await
            .map_err(Error::from)?;

        let observations = decode_batched_partial_values(
            response.observation.as_ref(),
            &observation_space,
            self.num_envs,
        )
        .map_err(protocol_error_to_error)?;
        validate_count(&observations, self.num_envs, "observations")
            .map_err(|error| Error::Environment(error.into()))?;

        Ok(VectorResetResult {
            observations,
            info: response.infos.map(meta_map_from_proto),
            episode_ids: response.episode_ids,
        })
    }

    /// Apply one action per sub-environment and return the batched transition.
    ///
    /// `req.actions` must contain exactly [`num_envs`](RemoteVectorEnv::num_envs)
    /// actions.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Environment`](crate::Error::Environment) on an env
    /// failure or batch-size mismatch, or a transport error on RPC failure.
    pub async fn step(&mut self, req: VectorStepRequest) -> Result<VectorStepResult> {
        let action_space = std::sync::Arc::clone(&self.action_space);
        let observation_space = std::sync::Arc::clone(&self.observation_space);

        validate_action_count(&req.actions, self.num_envs)
            .map_err(|error| Error::Environment(error.into()))?;
        let response = self
            .inner
            .step(ProtoStepRequest {
                action: Some(
                    encode_batched_partial_values(&req.actions, &action_space)
                        .map_err(protocol_error_to_error)?,
                ),
                // Native timeout_ms is i64 (>=0); proto field is uint64.
                timeout_ms: req.timeout_ms.max(0) as u64,
                // Full-width step; subset-stepping is reserved-but-deferred.
                env_indices: Vec::new(),
            })
            .await
            .map_err(Error::from)?;

        let observations = decode_batched_partial_values(
            response.observation.as_ref(),
            &observation_space,
            self.num_envs,
        )
        .map_err(protocol_error_to_error)?;
        let terminated = response
            .terminated_mask
            .iter()
            .map(|value| *value != 0)
            .collect::<Vec<_>>();
        let truncated = response
            .truncated_mask
            .iter()
            .map(|value| *value != 0)
            .collect::<Vec<_>>();
        let completed_episodes = response
            .completed_episodes
            .into_iter()
            .map(proto_episode_metadata_to_public)
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(protocol_error_to_error)?;

        let env_count = self.num_envs;
        validate_count(&observations, env_count, "observations")
            .map_err(|error| Error::Environment(error.into()))?;
        validate_count(&terminated, env_count, "terminated values")
            .map_err(|error| Error::Environment(error.into()))?;
        validate_count(&truncated, env_count, "truncated values")
            .map_err(|error| Error::Environment(error.into()))?;
        validate_count(&response.rewards, env_count, "rewards values")
            .map_err(|error| Error::Environment(error.into()))?;

        Ok(VectorStepResult {
            observations,
            rewards: response.rewards,
            terminated,
            truncated,
            info: response.infos.map(meta_map_from_proto),
            completed_episodes,
            episode_ids: response.episode_ids,
        })
    }

    /// Request a render frame from the remote environment.
    pub async fn render(&mut self, req: RenderRequest) -> Result<RenderResult> {
        let response = self
            .inner
            .render(render_request_to_proto(&req))
            .await
            .map_err(Error::from)?;
        render_result_from_proto(response).map_err(protocol_error_to_error)
    }

    /// Close this client session and return final episode metadata.
    ///
    /// This does not shut down the server. Use [`RemoteVectorEnv::shutdown`] when
    /// remote shutdown is enabled.
    pub async fn close(&mut self) -> Result<VectorCloseResult> {
        let response = self.inner.close().await.map_err(Error::from)?;
        Ok(VectorCloseResult {
            final_episodes: response
                .final_episodes
                .into_iter()
                .map(proto_episode_metadata_to_public)
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(protocol_error_to_error)?,
        })
    }

    /// Ask the server to shut itself down, returning whether it accepted.
    ///
    /// Unlike [`close`](RemoteVectorEnv::close), this stops the *server*, but only if
    /// it was started with
    /// [`ServeOptions::allow_remote_shutdown`](crate::ServeOptions::allow_remote_shutdown);
    /// otherwise the request is declined and this returns `Ok(false)`.
    pub async fn shutdown(&mut self, reason: impl Into<String>) -> Result<bool> {
        let response = self
            .inner
            .shutdown(reason.into())
            .await
            .map_err(Error::from)?;
        Ok(response.accepted)
    }
}

/// A client handle to one remote environment.
///
/// `RemoteEnv` is scalar-only. It rejects endpoints that report more than one
/// environment; use [`RemoteVectorEnv`] when the endpoint is intentionally
/// vectorized.
pub struct RemoteEnv {
    inner: RemoteVectorEnv,
}

impl RemoteEnv {
    /// Connect to a scalar env server at `address` and perform the handshake.
    pub async fn connect(address: &str) -> Result<Self> {
        Self::connect_to(ConnectAddress::parse(address)?).await
    }

    /// Connect to a scalar env server that requires a bearer token.
    pub async fn connect_with_token(address: &str, token: &str) -> Result<Self> {
        Self::connect_to_with_token(ConnectAddress::parse(address)?, token).await
    }

    /// Connect to an already-parsed [`ConnectAddress`].
    pub async fn connect_to(address: ConnectAddress) -> Result<Self> {
        Self::connect_to_with_token(address, "").await
    }

    async fn connect_to_with_token(address: ConnectAddress, token: &str) -> Result<Self> {
        let inner = RemoteVectorEnv::connect_to_with_token(address, token).await?;
        if inner.num_envs() != 1 {
            return Err(Error::Internal(format!(
                "RemoteEnv connects to one environment, but the endpoint reports num_envs={}; \
                 use RemoteVectorEnv instead",
                inner.num_envs()
            )));
        }
        Ok(Self { inner })
    }

    /// The address this client is connected to.
    pub fn address(&self) -> &str {
        self.inner.address()
    }

    /// The env's negotiation offer captured at handshake.
    pub fn session_offer(&self) -> &rlmesh_proto::SessionOffer {
        self.inner.session_offer()
    }

    /// Tear down the session locally without waiting for a Close round-trip.
    pub fn detach(&mut self) {
        self.inner.detach();
    }

    /// The environment contract reported by the server at handshake.
    pub fn env_contract(&self) -> &spaces::EnvContract {
        self.inner.env_contract()
    }

    /// Reset the remote environment and return the initial observation.
    pub async fn reset(
        &mut self,
        req: spaces::request::ResetRequest,
    ) -> Result<spaces::request::ResetResult> {
        let result = self
            .inner
            .reset(VectorResetRequest {
                seeds: req.seed.into_iter().collect(),
                options: req.options,
                timeout_ms: req.timeout_ms,
                env_indices: Vec::new(),
            })
            .await?;

        Ok(spaces::request::ResetResult {
            observation: result.observations.into_iter().next(),
            info: result.info,
            episode_id: result.episode_ids.into_iter().next(),
        })
    }

    /// Apply one action and return the transition.
    pub async fn step(
        &mut self,
        req: spaces::request::StepRequest,
    ) -> Result<spaces::request::StepResult> {
        let result = self
            .inner
            .step(VectorStepRequest {
                actions: req.action.into_iter().collect(),
                timeout_ms: req.timeout_ms,
            })
            .await?;

        Ok(spaces::request::StepResult {
            observation: result.observations.into_iter().next(),
            reward: result.rewards.into_iter().next().unwrap_or_default(),
            terminated: result.terminated.into_iter().next().unwrap_or_default(),
            truncated: result.truncated.into_iter().next().unwrap_or_default(),
            info: result.info,
        })
    }

    /// Request a render frame from the remote environment.
    pub async fn render(&mut self, req: RenderRequest) -> Result<RenderResult> {
        self.inner.render(req).await
    }

    /// Close this client session.
    pub async fn close(&mut self) -> Result<spaces::request::CloseResult> {
        let _ = self.inner.close().await?;
        Ok(spaces::request::CloseResult)
    }

    /// Ask the server to shut itself down, returning whether it accepted.
    pub async fn shutdown(&mut self, reason: impl Into<String>) -> Result<bool> {
        self.inner.shutdown(reason).await
    }
}

fn validate_env_contract(env_contract: &spaces::EnvContract) -> Result<()> {
    if env_contract.observation_space.is_none() {
        return Err(Error::Internal(
            "remote env contract missing observation_space".to_string(),
        ));
    }
    if env_contract.action_space.is_none() {
        return Err(Error::Internal(
            "remote env contract missing action_space".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_env_contract_requires_spaces() {
        let observation_space = spaces::spaces::BoxSpaceBuilder::scalar(-1.0, 1.0, vec![1])
            .build()
            .unwrap();
        let action_space = spaces::spaces::DiscreteBuilder::new(2).build().unwrap();
        let valid = spaces::EnvContract {
            observation_space: Some(observation_space.clone()),
            action_space: Some(action_space.clone()),
            ..Default::default()
        };
        assert!(validate_env_contract(&valid).is_ok());

        let missing_observation = spaces::EnvContract {
            action_space: Some(action_space),
            ..Default::default()
        };
        let err = validate_env_contract(&missing_observation).unwrap_err();
        assert!(err.to_string().contains("missing observation_space"));

        let missing_action = spaces::EnvContract {
            observation_space: Some(observation_space),
            ..Default::default()
        };
        let err = validate_env_contract(&missing_action).unwrap_err();
        assert!(err.to_string().contains("missing action_space"));
    }
}
