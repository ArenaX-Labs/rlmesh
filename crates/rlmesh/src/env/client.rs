use rlmesh_grpc::env::{ResetRequest as ProtoResetRequest, StepRequest as ProtoStepRequest};
use rlmesh_grpc::wire::{
    bytes_value, decode_batched_partial_values, encode_batched_partial_values,
    meta_map_from_struct, meta_map_to_struct, render_request_to_proto, render_result_from_proto,
    value_bytes,
};

use super::types::{
    CloseResult, RenderRequest, RenderResult, ResetRequest, ResetResult, StepRequest, StepResult,
};
use super::wire::{
    proto_episode_metadata_to_public, protocol_error_to_error, validate_action_count,
    validate_bool_count, validate_f64_count, validate_observation_count,
};
use crate::{ConnectAddress, Error, Result, spaces};

pub struct RemoteEnv {
    inner: rlmesh_grpc::EnvClient,
    env_contract: spaces::EnvContract,
    // Cheap-to-clone handles to the obs/action specs. Cloning the `Arc` for each
    // reset/step (to release the borrow of `self` before the `&mut self.inner`
    // RPC) is a refcount bump rather than a prost deep clone of the spec.
    observation_space: std::sync::Arc<spaces::SpaceSpec>,
    action_space: std::sync::Arc<spaces::SpaceSpec>,
    num_envs: usize,
}

impl RemoteEnv {
    pub async fn connect(address: &str) -> Result<Self> {
        Self::connect_to(ConnectAddress::parse(address)?).await
    }

    pub async fn connect_to(address: ConnectAddress) -> Result<Self> {
        let mut inner = rlmesh_grpc::EnvClient::connect(&address.to_string())
            .await
            .map_err(Error::from)?;
        let handshake = inner.handshake().await.map_err(Error::from)?;
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
        })
    }

    pub fn address(&self) -> &str {
        self.inner.address()
    }

    pub fn env_contract(&self) -> &spaces::EnvContract {
        &self.env_contract
    }

    pub fn num_envs(&self) -> usize {
        self.num_envs
    }

    pub async fn reset(&mut self, req: ResetRequest) -> Result<ResetResult> {
        let observation_space = std::sync::Arc::clone(&self.observation_space);

        let response = self
            .inner
            .reset(ProtoResetRequest {
                seeds: req.seeds,
                options: req.options.as_ref().map(meta_map_to_struct),
                timeout_ms: req.timeout_ms,
            })
            .await
            .map_err(Error::from)?;

        let observation_payload =
            value_bytes(response.observation.as_ref()).map_err(protocol_error_to_error)?;
        let observations =
            decode_batched_partial_values(observation_payload.as_ref(), &observation_space)
                .map_err(protocol_error_to_error)?;
        validate_observation_count(&observations, self.num_envs)
            .map_err(|error| Error::Environment(error.into()))?;

        Ok(ResetResult {
            observations,
            info: response.infos.map(meta_map_from_struct),
            episode_ids: response.episode_ids,
        })
    }

    pub async fn step(&mut self, req: StepRequest) -> Result<StepResult> {
        let action_space = std::sync::Arc::clone(&self.action_space);
        let observation_space = std::sync::Arc::clone(&self.observation_space);

        validate_action_count(&req.actions, self.num_envs)
            .map_err(|error| Error::Environment(error.into()))?;
        let response = self
            .inner
            .step(ProtoStepRequest {
                action: Some(bytes_value(
                    encode_batched_partial_values(&req.actions, &action_space)
                        .map_err(protocol_error_to_error)?,
                )),
                timeout_ms: req.timeout_ms,
            })
            .await
            .map_err(Error::from)?;

        let observation_payload =
            value_bytes(response.observation.as_ref()).map_err(protocol_error_to_error)?;
        let observations =
            decode_batched_partial_values(observation_payload.as_ref(), &observation_space)
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
        validate_observation_count(&observations, env_count)
            .map_err(|error| Error::Environment(error.into()))?;
        validate_bool_count(&terminated, env_count, "terminated")
            .map_err(|error| Error::Environment(error.into()))?;
        validate_bool_count(&truncated, env_count, "truncated")
            .map_err(|error| Error::Environment(error.into()))?;
        validate_f64_count(&response.rewards, env_count, "rewards")
            .map_err(|error| Error::Environment(error.into()))?;

        Ok(StepResult {
            observations,
            rewards: response.rewards,
            terminated,
            truncated,
            info: response.infos.map(meta_map_from_struct),
            completed_episodes,
            episode_ids: response.episode_ids,
        })
    }

    pub async fn render(&mut self, req: RenderRequest) -> Result<RenderResult> {
        let response = self
            .inner
            .render(render_request_to_proto(&req))
            .await
            .map_err(Error::from)?;
        render_result_from_proto(response).map_err(protocol_error_to_error)
    }

    /// Close this client's session and return its final episode metadata.
    ///
    /// # Session vs. server lifetime
    ///
    /// This detaches the **client session** only; it does **not** shut down the
    /// served environment. By design the environment is reusable across
    /// sessions, so a new [`RemoteEnv::connect`] to the same endpoint after
    /// `close` starts a fresh session against the same (still-running) env. To
    /// stop the server itself, use [`RemoteEnv::shutdown`] (when the server was
    /// started with remote shutdown allowed) or the server's own idle-timeout /
    /// drain policy. This is intentional, not a leak (review finding #81).
    pub async fn close(&mut self) -> Result<CloseResult> {
        let response = self.inner.close().await.map_err(Error::from)?;
        Ok(CloseResult {
            final_episodes: response
                .final_episodes
                .into_iter()
                .map(proto_episode_metadata_to_public)
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(protocol_error_to_error)?,
        })
    }

    pub async fn shutdown(&mut self, reason: impl Into<String>) -> Result<bool> {
        let response = self
            .inner
            .shutdown(reason.into())
            .await
            .map_err(Error::from)?;
        Ok(response.accepted)
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
