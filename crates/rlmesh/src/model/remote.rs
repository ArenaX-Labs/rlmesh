use std::sync::Arc;

use rlmesh_grpc::wire::{
    bytes_value, decode_batched_partial_values, encode_batched_partial_values,
    env_contract_to_proto, value_bytes,
};
use rlmesh_proto::model::v1::{
    CloseRouteRequest, ConfigureRouteRequest, PredictContext, PredictRequest, PredictSlot,
};
use uuid::Uuid;

use crate::{ConnectAddress, Error, Result, spaces};

/// A client handle to a remote model (policy) server.
///
/// The mirror image of [`RemoteEnv`](crate::RemoteEnv): connect with
/// [`RemoteModel::connect`], then drive the policy by hand with
/// [`reset`](RemoteModel::reset) / [`predict`](RemoteModel::predict). The same
/// hand-written loop steps a [`RemoteEnv`](crate::RemoteEnv) and a `RemoteModel`
/// in lockstep:
///
/// ```ignore
/// let mut obs = env.reset(reset_req).await?.observations.remove(0);
/// model.reset();
/// loop {
///     let action = model.predict(obs).await?;
///     let step = env.step(StepRequest { actions: vec![action], ..Default::default() }).await?;
///     if step.terminated[0] || step.truncated[0] { break; }
///     obs = step.observations.into_iter().next().unwrap();
/// }
/// ```
///
/// Use this when your code owns the loop; to hand the loop to rlmesh instead,
/// see [`ModelWorker::run_local`](crate::ModelWorker).
///
/// A `RemoteModel` drives a single env lane (`env_index = 0`): it configures one
/// route from the env contract and sends one observation per `predict`.
pub struct RemoteModel {
    inner: rlmesh_grpc::ModelClient,
    observation_space: Arc<spaces::SpaceSpec>,
    action_space: Arc<spaces::SpaceSpec>,
    env_contract: spaces::EnvContract,
    session_id: String,
    route_id: String,
    request_counter: u64,
    episode_counter: u64,
    configured: bool,
    episode_id: Option<String>,
    step: i64,
    /// Set by [`reset`](RemoteModel::reset), consumed by the first
    /// [`predict`](RemoteModel::predict) after it, which marks the wire reset
    /// boundary. Making the boundary explicit (rather than inferring it from
    /// `step == 0`) keeps it correct even though this client cannot observe an
    /// episode's end from the server.
    pending_reset: bool,
}

/// Per-instance session id. The served model caches route configs, the
/// resolved adapter, and active episodes keyed by `session_id:route_id`, so a
/// globally unique id keeps two clients — even in separate containers that both
/// run at PID 1 — from colliding on that key and clobbering each other's
/// contract/adapter/lifecycle.
fn new_session_id() -> String {
    format!("remote-model-{}", Uuid::new_v4())
}

/// Coerce an env contract to the single lane this client drives. A zero-width
/// contract clamps up to one lane (configure_route rejects num_envs=0); a vector
/// contract is rejected, since this client would silently evaluate only lane 0.
fn single_lane_contract(mut env_contract: spaces::EnvContract) -> Result<spaces::EnvContract> {
    if env_contract.num_envs == 0 {
        env_contract.num_envs = 1;
    }
    if env_contract.num_envs > 1 {
        return Err(Error::Internal(format!(
            "RemoteModel drives a single env, but the env contract reports num_envs={}; \
             use num_envs=1 (this client sends one observation per predict on lane 0)",
            env_contract.num_envs
        )));
    }
    Ok(env_contract)
}

impl RemoteModel {
    /// Connect to a model server at `address` and perform the handshake.
    ///
    /// `env_contract` is the contract of the env this policy will be driven
    /// against (take it from [`RemoteEnv::env_contract`](crate::RemoteEnv::env_contract));
    /// it pins the observation/action spaces this client encodes and decodes
    /// with, and is sent once to configure the model's route. It must carry both
    /// an observation and an action space.
    pub async fn connect(address: &str, env_contract: spaces::EnvContract) -> Result<Self> {
        Self::connect_with_token(address, "", env_contract).await
    }

    /// Connect to a model server that requires a bearer token. An empty token
    /// behaves like [`RemoteModel::connect`].
    pub async fn connect_with_token(
        address: &str,
        token: &str,
        env_contract: spaces::EnvContract,
    ) -> Result<Self> {
        let address = ConnectAddress::parse(address)?;
        let observation_space = Arc::new(
            env_contract
                .observation_space
                .clone()
                .ok_or_else(|| Error::Internal("env contract missing observation_space".into()))?,
        );
        let action_space = Arc::new(
            env_contract
                .action_space
                .clone()
                .ok_or_else(|| Error::Internal("env contract missing action_space".into()))?,
        );
        let env_contract = single_lane_contract(env_contract)?;

        let mut inner = rlmesh_grpc::ModelClient::connect(&address.to_string(), token)
            .await
            .map_err(Error::from)?;
        inner.handshake().await.map_err(Error::from)?;

        Ok(Self {
            inner,
            observation_space,
            action_space,
            env_contract,
            session_id: new_session_id(),
            route_id: "remote-model".to_string(),
            request_counter: 0,
            episode_counter: 0,
            configured: false,
            episode_id: None,
            step: 0,
            pending_reset: false,
        })
    }

    /// The address this client is connected to.
    pub fn address(&self) -> &str {
        self.inner.address()
    }

    /// Begin a new episode: the next [`predict`](RemoteModel::predict) marks a
    /// reset boundary so the policy starts a fresh trajectory.
    ///
    /// Call this once before each episode's first `predict`. The client cannot
    /// observe an episode's end from the server, so an episode boundary is only
    /// signalled to the policy when you call `reset()`; a `predict()` after a
    /// finished episode without an intervening `reset()` continues the prior
    /// episode rather than starting a new one.
    pub fn reset(&mut self) {
        self.episode_counter += 1;
        self.episode_id = Some(format!("{}-ep-{}", self.route_id, self.episode_counter));
        self.step = 0;
        self.pending_reset = true;
    }

    /// Ask the policy for an action given `observation`.
    ///
    /// You must call [`reset`](RemoteModel::reset) once before the first
    /// `predict` of each episode: it is the only signal that marks an episode
    /// boundary on the wire, since this client cannot detect an episode's end
    /// from the server. The observation is encoded with the env contract's
    /// observation space and the returned action is decoded with its action
    /// space.
    pub async fn predict(&mut self, observation: spaces::SpaceValue) -> Result<spaces::SpaceValue> {
        let episode_id = self
            .episode_id
            .clone()
            .ok_or_else(|| Error::Internal("call reset() before predict()".into()))?;

        if !self.configured {
            self.configure_route().await?;
            self.configured = true;
        }

        // Encode the observation the same way the env wire path does (a
        // one-lane batched-partial payload): the served model decodes it with
        // decode_batched_partial_values, so a plain single-value encoding would
        // be misread as carrying a batch dimension.
        let observation_bytes = encode_batched_partial_values(
            std::slice::from_ref(&observation),
            &self.observation_space,
        )
        .map_err(|error| Error::Internal(error.to_string()))?;
        // The reset boundary is the explicit pending flag set by reset(), not
        // an inference from step == 0. The two agree for correct usage, but the
        // flag stays correct even when this client cannot see an episode's end.
        // Peek it here; only clear it after the RPC succeeds, so a failed
        // predict leaves the reset edge intact for a retry.
        let reset = self.pending_reset;
        let request = PredictRequest {
            context: Some(PredictContext {
                session_id: self.session_id.clone(),
                route_id: self.route_id.clone(),
                request_id: self.next_request_id(),
                slots: vec![PredictSlot {
                    env_index: 0,
                    episode_id,
                    step: self.step,
                    reset,
                }],
            }),
            observation: Some(bytes_value(observation_bytes)),
        };

        let response = self.inner.predict(request).await.map_err(Error::from)?;
        self.step += 1;
        self.pending_reset = false;

        let action_bytes = value_bytes(response.action.as_ref())
            .map_err(|error| Error::Internal(error.to_string()))?;
        let mut actions = decode_batched_partial_values(action_bytes.as_ref(), &self.action_space)
            .map_err(|error| Error::Internal(error.to_string()))?;
        actions
            .drain(..)
            .next()
            .ok_or_else(|| Error::Internal("predict response missing action".into()))
    }

    /// Close this client's route. Does not shut down the server or drain other
    /// clients sharing it. No-op if no route was ever configured.
    pub async fn close(&mut self) -> Result<()> {
        if !self.configured {
            return Ok(());
        }
        self.inner
            .close_route(CloseRouteRequest {
                context: Some(PredictContext {
                    session_id: self.session_id.clone(),
                    route_id: self.route_id.clone(),
                    request_id: format!("{}:close_route", self.route_id),
                    slots: Vec::new(),
                }),
                reason: "remote model session complete".to_string(),
            })
            .await
            .map_err(Error::from)
    }

    async fn configure_route(&mut self) -> Result<()> {
        self.inner
            .configure_route(ConfigureRouteRequest {
                context: Some(PredictContext {
                    session_id: self.session_id.clone(),
                    route_id: self.route_id.clone(),
                    request_id: format!("{}:configure_route", self.route_id),
                    slots: Vec::new(),
                }),
                env_contract: Some(env_contract_to_proto(&self.env_contract)),
            })
            .await
            .map_err(Error::from)
    }

    fn next_request_id(&mut self) -> String {
        self.request_counter += 1;
        format!("{}:predict:{}", self.route_id, self.request_counter)
    }
}

#[cfg(test)]
mod tests {
    use super::{new_session_id, single_lane_contract};
    use crate::spaces;

    fn single_lane_test_contract(num_envs: u32) -> spaces::EnvContract {
        spaces::EnvContract {
            id: "remote-model-test".to_string(),
            autoreset_mode: Default::default(),
            observation_space: None,
            action_space: None,
            metadata: None,
            render_mode: String::new(),
            num_envs,
        }
    }

    #[test]
    fn session_ids_are_unique_and_globally_namespaced() {
        // Each RemoteModel must claim a distinct session id so the served
        // model's session_id:route_id-keyed caches (route config, adapter,
        // active episodes) never collide — including between two containers that
        // both run at PID 1, which the old process-local counter could not avoid.
        let first = new_session_id();
        let second = new_session_id();
        assert_ne!(first, second);
        assert!(first.starts_with("remote-model-"), "{first}");
        assert!(second.starts_with("remote-model-"), "{second}");
    }

    #[test]
    fn single_lane_contract_rejects_vector_env() {
        let err = single_lane_contract(single_lane_test_contract(4))
            .expect_err("num_envs > 1 must be rejected");
        assert!(err.to_string().contains("num_envs=4"), "{err}");
    }

    #[test]
    fn single_lane_contract_clamps_zero_and_accepts_one() {
        assert_eq!(
            single_lane_contract(single_lane_test_contract(0))
                .unwrap()
                .num_envs,
            1
        );
        assert_eq!(
            single_lane_contract(single_lane_test_contract(1))
                .unwrap()
                .num_envs,
            1
        );
    }

    // The predict() reset-edge guarantee (a failed predict RPC leaves
    // pending_reset set so a retry re-sends reset=true) cannot be unit-tested
    // without a live model server: predict() drives a real RPC. It is covered by
    // the integration suite. The unit-level invariant — pending_reset is peeked,
    // not consumed, before the RPC, and cleared only after success — is enforced
    // structurally in predict() (no std::mem::take before the send).
}
