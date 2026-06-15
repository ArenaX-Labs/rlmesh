//! [`RemoteModel`]: the client side of a served model — the inverse of
//! [`RemoteEnv`](crate::RemoteEnv). Where `RemoteEnv` lets your code drive a
//! served *environment*, `RemoteModel` lets an eval loop drive a served *model*:
//! it connects to a `ModelService` endpoint, configures one route from an env
//! contract, then maps observations to actions. The model is served (e.g. a fast
//! C/C++ binary via `rlmesh_model_serve`) while the loop stays wherever you write
//! it — including a Python eval script.

use std::sync::Arc;

use rlmesh_grpc::wire::{
    bytes_value, decode_batched_partial_values, encode_batched_partial_values,
    env_contract_to_proto, value_bytes,
};
use rlmesh_proto::model::v1::{ConfigureRouteRequest, PredictContext, PredictRequest, PredictSlot};

use crate::{ConnectAddress, Error, Result, spaces};

// A single client drives a single route; the server keys routes by
// "session_id:route_id", so fixed ids are enough for one eval loop.
const SESSION_ID: &str = "rlmesh.remote_model";
const ROUTE_ID: &str = "default";

/// A client handle to a remote model server (single env / single route).
///
/// Connect with [`RemoteModel::connect`] (or
/// [`connect_with_token`](RemoteModel::connect_with_token)) passing the env
/// contract the model should answer against — unlike an env server, a model
/// publishes no contract of its own, so the caller supplies it. Then call
/// [`predict`](RemoteModel::predict) per step and
/// [`begin_episode`](RemoteModel::begin_episode) at each episode boundary.
pub struct RemoteModel {
    inner: rlmesh_grpc::ModelClient,
    observation_space: Arc<spaces::SpaceSpec>,
    action_space: Arc<spaces::SpaceSpec>,
    request_seq: u64,
    episode: u64,
    step: i64,
    pending_reset: bool,
}

impl RemoteModel {
    /// Connect to a model server at `address`, handshake, and open one route for
    /// `env_contract`. `address` takes the same forms as
    /// [`ConnectAddress::parse`](crate::ConnectAddress::parse).
    pub async fn connect(address: &str, env_contract: spaces::EnvContract) -> Result<Self> {
        Self::connect_to_with_token(ConnectAddress::parse(address)?, "", env_contract).await
    }

    /// [`connect`](RemoteModel::connect) for an endpoint that requires a bearer
    /// token (an empty token behaves like `connect`).
    pub async fn connect_with_token(
        address: &str,
        token: &str,
        env_contract: spaces::EnvContract,
    ) -> Result<Self> {
        Self::connect_to_with_token(ConnectAddress::parse(address)?, token, env_contract).await
    }

    async fn connect_to_with_token(
        address: ConnectAddress,
        token: &str,
        env_contract: spaces::EnvContract,
    ) -> Result<Self> {
        let observation_space = env_contract
            .observation_space
            .clone()
            .ok_or_else(|| Error::Internal("env contract missing observation_space".to_string()))?;
        let action_space = env_contract
            .action_space
            .clone()
            .ok_or_else(|| Error::Internal("env contract missing action_space".to_string()))?;

        let mut inner = rlmesh_grpc::ModelClient::connect(&address.to_string(), token)
            .await
            .map_err(Error::from)?;
        inner.handshake().await.map_err(Error::from)?;
        inner
            .configure_route(ConfigureRouteRequest {
                context: Some(route_context("configure".to_string(), Vec::new())),
                env_contract: Some(env_contract_to_proto(&env_contract)),
            })
            .await
            .map_err(Error::from)?;

        Ok(Self {
            inner,
            observation_space: Arc::new(observation_space),
            action_space: Arc::new(action_space),
            request_seq: 0,
            episode: 0,
            step: 0,
            // The first predict opens the model's first episode.
            pending_reset: true,
        })
    }

    /// The observation space the model decodes against (from the env contract).
    pub fn observation_space(&self) -> &spaces::SpaceSpec {
        &self.observation_space
    }

    /// The action space the model encodes against (from the env contract).
    pub fn action_space(&self) -> &spaces::SpaceSpec {
        &self.action_space
    }

    /// Mark the next [`predict`](RemoteModel::predict) as the first step of a new
    /// episode (sets the slot's `reset` flag and resets the step counter), so a
    /// stateful served model can re-init its per-episode state.
    pub fn begin_episode(&mut self) {
        self.episode += 1;
        self.step = 0;
        self.pending_reset = true;
    }

    /// Send one observation and return the model's action. Single env: exactly
    /// one observation in, one action out.
    pub async fn predict(&mut self, observation: spaces::SpaceValue) -> Result<spaces::SpaceValue> {
        let reset = self.pending_reset;
        self.pending_reset = false;
        self.request_seq += 1;

        let observation_space = Arc::clone(&self.observation_space);
        let slot = PredictSlot {
            env_index: 0,
            episode_id: format!("episode-{}", self.episode),
            step: self.step,
            reset,
        };
        let request = PredictRequest {
            context: Some(route_context(
                format!("predict-{}", self.request_seq),
                vec![slot],
            )),
            observation: Some(bytes_value(
                encode_batched_partial_values(
                    std::slice::from_ref(&observation),
                    &observation_space,
                )
                .map_err(wire_error)?,
            )),
        };

        let response = self.inner.predict(request).await.map_err(Error::from)?;
        self.step += 1;

        let action_payload = value_bytes(response.action.as_ref()).map_err(wire_error)?;
        let actions = decode_batched_partial_values(action_payload.as_ref(), &self.action_space)
            .map_err(wire_error)?;
        actions
            .into_iter()
            .next()
            .ok_or_else(|| Error::model("model returned no action"))
    }

    /// Close this client session (does not stop the server).
    pub async fn close(&mut self) -> Result<()> {
        self.inner
            .close("remote model session complete")
            .await
            .map_err(Error::from)
    }

    /// Ask the server to shut down (only honored when it was started with
    /// [`ServeOptions::allow_remote_shutdown`](crate::ServeOptions::allow_remote_shutdown)).
    pub async fn shutdown(&mut self, reason: impl Into<String>) -> Result<bool> {
        let response = self
            .inner
            .shutdown(reason.into())
            .await
            .map_err(Error::from)?;
        Ok(response.accepted)
    }
}

fn route_context(request_id: String, slots: Vec<PredictSlot>) -> PredictContext {
    PredictContext {
        session_id: SESSION_ID.to_string(),
        route_id: ROUTE_ID.to_string(),
        request_id,
        slots,
    }
}

fn wire_error(error: impl ToString) -> Error {
    Error::Internal(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BindAddress, ModelHandler, ModelObservation, ModelWorker, ServeModelOptions};
    use async_trait::async_trait;

    /// A model that ignores its observation and always answers `action`.
    struct ConstModel {
        action_space: spaces::SpaceSpec,
        action: i64,
    }

    #[async_trait]
    impl ModelHandler for ConstModel {
        async fn predict(
            &mut self,
            _observation: ModelObservation,
        ) -> Result<spaces::BinaryPayload> {
            let action = spaces::SpaceValue::Discrete(self.action);
            let encoded =
                encode_batched_partial_values(std::slice::from_ref(&action), &self.action_space)
                    .map_err(wire_error)?;
            Ok(spaces::BinaryPayload { data: encoded.data })
        }
    }

    #[tokio::test]
    async fn remote_model_predict_round_trips_against_a_served_model() {
        let observation_space = spaces::spaces::DiscreteBuilder::new(8).build().unwrap();
        let action_space = spaces::spaces::DiscreteBuilder::new(4).build().unwrap();
        let contract = spaces::EnvContract {
            observation_space: Some(observation_space),
            action_space: Some(action_space.clone()),
            num_envs: 1,
            ..Default::default()
        };

        let bound = ModelWorker::new(ConstModel {
            action_space,
            action: 2,
        })
        .bind_async(ServeModelOptions::new(
            BindAddress::parse("tcp://127.0.0.1:0").unwrap(),
        ))
        .await
        .expect("bind model server");
        let address = bound.local_addr().to_string();
        let server = tokio::spawn(async move { bound.serve().await });

        let mut model = RemoteModel::connect(&address, contract)
            .await
            .expect("connect remote model");

        let action = model
            .predict(spaces::SpaceValue::Discrete(3))
            .await
            .expect("predict");
        match action {
            spaces::SpaceValue::Discrete(value) => assert_eq!(value, 2),
            other => panic!("expected a discrete action, got {other:?}"),
        }

        // A new episode flags reset on the next predict; the constant model still
        // answers.
        model.begin_episode();
        let action = model
            .predict(spaces::SpaceValue::Discrete(5))
            .await
            .expect("predict after reset");
        assert!(matches!(action, spaces::SpaceValue::Discrete(2)));

        server.abort();
    }
}
