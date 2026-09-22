use std::sync::Arc;

use rlmesh_grpc::wire::{
    decode_batched_partial_values, encode_batched_partial_values, env_spec_to_proto,
};
use rlmesh_proto::SessionOffer;
use rlmesh_proto::model::v1::{
    AdapterContext, EpisodeInfo, ObservationHistoryFrame, PredictRequest, ReleaseAdapterRequest,
    ResetAdapterRequest, ResolveAdapterRequest,
};
use rlmesh_runtime::PeerCeiling;
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
/// ```no_run
/// use rlmesh::prelude::*;
///
/// # async fn drive(mut env: RemoteEnv, mut model: rlmesh::RemoteModel) -> rlmesh::Result<()> {
/// let mut obs = env
///     .reset(ResetRequest::default())
///     .await?
///     .observation
///     .expect("env returned an observation");
/// model.reset(None);
/// loop {
///     let action = model.predict(obs).await?;
///     let step = env.step(StepRequest { action: Some(action), ..Default::default() }).await?;
///     if step.terminated || step.truncated { break; }
///     obs = step.observation.expect("env returned an observation");
/// }
/// # Ok(())
/// # }
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
    /// The env (adapter) routing key, a UUIDv7 minted by this client (the local
    /// id authority for the direct path). Replaces the old route_id + lane/slot.
    env_id: String,
    request_counter: u64,
    configured: bool,
    episode_id: Option<String>,
    /// Runtime-chosen execution horizon `h`, sent on `ResolveAdapter`. When `> 1` and
    /// the served model defines a chunk corner, each real `predict` returns up to
    /// `h` ordered frames in `PredictResponse.actions`; this client buffers and
    /// replays the frames open-loop (skipping the RPC after frame 0), so a
    /// `RemoteModel` is a single-lane replay mini-driver. `1` (the default) = no
    /// chunking.
    execution_horizon: u32,
    /// Decoded chunk frames awaiting replay (the whole `PredictResponse.actions`
    /// list — frame 0 plus frames 1..). Drained one per `predict`; the next real
    /// RPC fires only once it empties. Flushed on `reset` — the only episode
    /// boundary this client can observe.
    replay_buffer: std::collections::VecDeque<spaces::SpaceValue>,
    /// The explicit seed the current episode was reset with, if any — set by
    /// [`reset`](Self::reset), rides on every `predict` of this episode, and is
    /// advisory context for the served model (not consumed by this client).
    seed: Option<i64>,
    /// Ids of episodes this client has already left behind, waiting for a
    /// `ResetAdapter` to be flushed. `reset` cannot send one (it is not async), so
    /// the next `predict`/`close` does — the served engine then fires the model's
    /// `on_episode_end` for exactly those ids and evicts their frame buffers.
    pending_end: Vec<String>,
    /// The env edition this session runs at: the floor — the highest edition env,
    /// model, AND this runtime all support. Sent to the model in `ResolveAdapter`
    /// as AUTHORITATIVE over its own (pairwise) handshake result.
    selected_workflow_edition: String,
    /// What the served model can decode, learned at its handshake.
    ceiling: PeerCeiling,
    /// The served route asked for observation history at resolve: every replayed
    /// step's observation is then carried as a history row on the next real
    /// predict, so the model-side frame windows see every step.
    history_on: bool,
    /// Replayed steps' observations awaiting the next real predict.
    history: Vec<ObservationHistoryFrame>,
    /// This client's env-step counter: one per `predict` call (each call is one
    /// env step from the caller's side), stamped on rows and requests.
    step: i64,
}

/// Per-instance session id, kept for correlation only.
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
    ///
    /// Without an explicit env offer this assumes the env speaks this build's
    /// own generation/edition window (true while the window holds a single
    /// generation and edition). For a genuine three-way reconciliation against a
    /// remote env's advertised window, use
    /// [`connect_with_env_offer`](Self::connect_with_env_offer).
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
        // No explicit env offer: take the runtime's own offer (its retained list
        // plus the edition it declares) as the env offer. With a single-edition
        // build the mutual is unaffected; the fail-fast machinery still runs.
        let env_offer = SessionOffer::this_build(None);
        Self::connect_with_env_offer(address, token, env_contract, env_offer).await
    }

    /// Connect and reconcile the floor with an explicit runtime-tier
    /// declaration — the `workflow_edition` pin the caller was handed.
    ///
    /// Same as [`connect_with_env_offer`](Self::connect_with_env_offer), except
    /// that `declared` is the edition THIS runtime declares (its WANT): the
    /// session then runs there even when the env and the model could both go
    /// higher. `None` declares nothing, which is exactly
    /// [`connect_with_env_offer`](Self::connect_with_env_offer). A pin no peer
    /// can run is refused before any route is configured, naming every tier's
    /// WANT and CAN.
    pub async fn connect_declaring(
        address: &str,
        token: &str,
        env_contract: spaces::EnvContract,
        env_offer: SessionOffer,
        declared: Option<&str>,
    ) -> Result<Self> {
        Self::connect_inner(address, token, env_contract, env_offer, declared).await
    }

    /// Connect to a model server and reconcile the **route workflow edition**
    /// across the env, the model, and this runtime.
    ///
    /// The runtime re-frames env<->model traffic, so a session runs at the
    /// 3-way [`rlmesh_proto::negotiate_session_floor`] — the highest edition all three support —
    /// never the env<->model max, or an edition-gated field would be silently
    /// stripped crossing this runtime. This handshakes the **model first** to learn
    /// its window, then reconciles the floor. If this runtime is what holds the
    /// edition back it **warns** (still runs, safely, at the floor); with no mutual
    /// edition at all it **fails before any route is configured**, naming each
    /// tier's offer. The selected edition is sent to the model in `ResolveAdapter`
    /// as authoritative over its own handshake.
    ///
    /// `env_offer` is the env's advertised window; take it from
    /// [`RemoteEnv::session_offer`](crate::RemoteEnv::session_offer).
    pub async fn connect_with_env_offer(
        address: &str,
        token: &str,
        env_contract: spaces::EnvContract,
        env_offer: SessionOffer,
    ) -> Result<Self> {
        Self::connect_inner(address, token, env_contract, env_offer, None).await
    }

    async fn connect_inner(
        address: &str,
        token: &str,
        env_contract: spaces::EnvContract,
        env_offer: SessionOffer,
        declared: Option<&str>,
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

        // Model first: the runtime learns the model's window before it can pick
        // the edition. (The env was already handshaked by the caller; its offer
        // narrows the floor here.)
        let mut inner = rlmesh_grpc::ModelClient::connect(&address.to_string(), token)
            .await
            .map_err(Error::from)?;
        inner.declare_workflow_edition(declared.map(str::to_string));
        inner.handshake().await.map_err(Error::from)?;
        let model_offer = inner.model_session_offer();

        // Reconcile the 3-way floor (env + model + this runtime) via the shared
        // helper, which warns when this runtime is the limiting tier and errs when
        // the three share no edition (before any Join/ResolveAdapter is sent). The
        // helper lives in rlmesh-grpc so the production runtime computes the same
        // floor; here the facade just consumes it.
        let selected_workflow_edition = rlmesh_grpc::env_floor(&env_offer, &model_offer, declared)
            .map_err(Error::from)?
            .selected_workflow_edition;
        let session_edition = rlmesh_proto::parse_retained_edition(&selected_workflow_edition)
            .map_err(Error::Internal)?;
        let ceiling = PeerCeiling::wire_v1(
            PeerCeiling::highest_shared_edition(&model_offer.editions).unwrap_or(session_edition),
            inner.server_capabilities().clone(),
            rlmesh_grpc::MAX_MESSAGE_SIZE,
        );

        Ok(Self {
            inner,
            observation_space,
            action_space,
            env_contract,
            session_id: new_session_id(),
            env_id: crate::mint_id(),
            request_counter: 0,
            configured: false,
            episode_id: None,
            execution_horizon: 1,
            replay_buffer: std::collections::VecDeque::new(),
            history_on: false,
            history: Vec::new(),
            step: 0,
            seed: None,
            pending_end: Vec::new(),
            selected_workflow_edition,
            ceiling,
        })
    }

    /// Set the execution horizon `h` this client requests from the served model.
    ///
    /// `h > 1` opts a chunk-capable served model into action chunking: each real
    /// `predict` returns the chunk's future frames, which this client replays
    /// open-loop. Must be set **before the first `predict`** (the route is
    /// configured lazily there and the horizon is pinned on `ResolveAdapter`);
    /// `1` (the default) leaves chunking off.
    pub fn set_execution_horizon(&mut self, execution_horizon: u32) {
        self.execution_horizon = execution_horizon;
    }

    /// The workflow edition this session runs at (the floor across env, model, and runtime).
    pub fn selected_workflow_edition(&self) -> &str {
        &self.selected_workflow_edition
    }

    /// What the served model can decode: the `model_ceiling` a
    /// [`RuntimeSessionSpec`](rlmesh_runtime::RuntimeSessionSpec) relaying to
    /// it carries.
    pub fn ceiling(&self) -> &PeerCeiling {
        &self.ceiling
    }

    /// The env (adapter) routing key this client uses with the model — a UUIDv7
    /// minted at connect. The single model-facing identity for this connection.
    pub fn env_id(&self) -> &str {
        &self.env_id
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
    ///
    /// `seed` is the explicit seed this episode was reset with, if any — rides
    /// on every `predict` of this episode as advisory context for the served
    /// model (its `predict`/`predict_chunk` `context` argument), mirroring the
    /// seed you pass to the paired [`RemoteEnv::reset`](crate::RemoteEnv::reset).
    pub fn reset(&mut self, seed: Option<i64>) {
        // The client is the local id authority for the direct path: mint a fresh
        // UUIDv7 per episode (never repeats, time-ordered). The new id is itself
        // the reset boundary on the wire.
        //
        // The episode this replaces has ended: stash its id so the next RPC flushes
        // a `ResetAdapter` for it. Without that the served model never sees an
        // episode-end edge on this path (the server cannot observe the client's
        // boundary) and a stateful policy carries state across episodes.
        self.pending_end
            .extend(self.episode_id.replace(crate::mint_id()));
        self.seed = seed;
        // Drop any un-replayed chunk tail: a new episode re-plans from its first
        // observation. This is the only flush point — the client cannot observe the
        // server's episode end, so a stale tail would otherwise bleed across the
        // boundary.
        self.replay_buffer.clear();
        // The rows belonged to the ended episode's windows, which the flushed
        // `ResetAdapter` evicts; the new episode starts with none.
        self.history.clear();
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
            self.resolve_adapter().await?;
            self.configured = true;
        }
        self.flush_pending_end().await?;

        // Re-call predict only when the replay buffer is empty; otherwise replay a
        // buffered chunk frame without an RPC (and without consuming the
        // observation). The buffer is filled by a real predict below and flushed on
        // reset, so a replay step is never the reset edge (pending_reset is already
        // cleared by the predict that filled the buffer).
        // This call is one env step from the caller's side, whether it re-plans
        // or replays; the counter stamps rows and requests so the served engine
        // can hold this client to consecutive steps.
        let step = self.step;
        self.step += 1;
        // Encode the observation the same way the env wire path does (a
        // one-lane batched-partial payload): the served model decodes it with
        // decode_batched_partial_values, so a plain single-value encoding would
        // be misread as carrying a batch dimension.
        let encode = |observation: &spaces::SpaceValue| {
            encode_batched_partial_values(
                std::slice::from_ref(observation),
                &self.observation_space,
            )
            .map_err(|error| Error::Internal(error.to_string()))
        };
        if !self.replay_buffer.is_empty() && self.history_on {
            // A replayed step: the served windows still have to see its frame,
            // so it rides the next real predict as a history row.
            let frame = ObservationHistoryFrame {
                observation: Some(encode(&observation)?),
                episode_info: vec![EpisodeInfo {
                    episode_id: episode_id.clone(),
                    seed: self.seed,
                }],
                step,
            };
            self.history.push(frame);
        }

        if self.replay_buffer.is_empty() {
            let observation_value = encode(&observation)?;
            // The wire carries no per-row reset flag; the reset boundary is the
            // fresh episode id minted in reset(), which rides episode_info below,
            // alongside the seed that same reset() call was given (if any).
            let request = PredictRequest {
                context: Some(AdapterContext {
                    session_id: self.session_id.clone(),
                    env_id: self.env_id.clone(),
                    request_id: self.next_request_id(),
                }),
                observation: Some(observation_value),
                episode_info: vec![EpisodeInfo {
                    episode_id,
                    seed: self.seed,
                }],
                history: std::mem::take(&mut self.history),
                step: self.history_on.then_some(step),
            };

            let response = self.inner.predict(request).await.map_err(Error::from)?;

            if response.actions.is_empty() {
                return Err(Error::Internal(
                    "predict returned no actions for a single-env route".to_string(),
                ));
            }
            // Push every ordered frame (`actions[0]` is this step, `actions[1..]`
            // the open-loop replay frames) for the next steps; a non-chunking model
            // returns exactly one frame, so this is the unchanged single-action
            // path. Each frame is the batched (one-lane) action for its step;
            // decode with N=1 and assert exactly one action, never silently take
            // the first lane (§5).
            for frame in &response.actions {
                let mut frames = decode_batched_partial_values(Some(frame), &self.action_space, 1)
                    .map_err(|error| Error::Internal(error.to_string()))?;
                if frames.len() != 1 {
                    return Err(Error::Internal(format!(
                        "predict frame decoded to {} actions for a single-env route",
                        frames.len()
                    )));
                }
                self.replay_buffer.push_back(frames.remove(0));
            }
        }

        let action = self
            .replay_buffer
            .pop_front()
            .expect("replay buffer is non-empty after a refill");
        Ok(action)
    }

    /// Close this client's route. Does not shut down the server or drain other
    /// clients sharing it. No-op if no route was ever configured.
    pub async fn close(&mut self) -> Result<()> {
        if !self.configured {
            return Ok(());
        }
        // The last episode ends here too: flush its end edge before the route goes
        // away, so the served model's `on_episode_end` fires for every episode.
        self.pending_end.extend(self.episode_id.take());
        self.flush_pending_end().await?;
        self.inner
            .release_adapter(ReleaseAdapterRequest {
                context: Some(AdapterContext {
                    session_id: self.session_id.clone(),
                    env_id: self.env_id.clone(),
                    request_id: format!("{}:release_adapter", self.env_id),
                }),
                reason: "remote model session complete".to_string(),
            })
            .await
            .map_err(Error::from)
    }

    /// Send the stashed `ResetAdapter` for the episode `reset` left behind, if
    /// any. Fires before the next predict and at close; a failure surfaces to the
    /// caller rather than silently dropping the edge.
    async fn flush_pending_end(&mut self) -> Result<()> {
        if self.pending_end.is_empty() {
            return Ok(());
        }
        // Distinct per flush: the client demuxes stream responses by request_id.
        self.request_counter += 1;
        let request_id = format!("{}:reset_adapter:{}", self.env_id, self.request_counter);
        self.inner
            .reset_adapter(ResetAdapterRequest {
                context: Some(AdapterContext {
                    session_id: self.session_id.clone(),
                    env_id: self.env_id.clone(),
                    request_id,
                }),
                episode_ids: std::mem::take(&mut self.pending_end),
            })
            .await
            .map_err(Error::from)
    }

    async fn resolve_adapter(&mut self) -> Result<()> {
        let response = self
            .inner
            .resolve_adapter(ResolveAdapterRequest {
                context: Some(AdapterContext {
                    session_id: self.session_id.clone(),
                    env_id: self.env_id.clone(),
                    request_id: format!("{}:resolve_adapter", self.env_id),
                }),
                env_spec: Some(env_spec_to_proto(&self.env_contract)),
                // Pin the model to the runtime-selected env edition: authoritative
                // over the model's own (pairwise) handshake result.
                selected_workflow_edition: self.selected_workflow_edition.clone(),
                // Runtime-chosen execution horizon, pinned on the env (see
                // [`set_execution_horizon`](Self::set_execution_horizon)). 1 = no chunking.
                execution_horizon: self.execution_horizon,
                // This client replays chunk frames itself, so it can carry every
                // replayed step's observation as a history row; offered only to
                // a served model that advertised it ingests them.
                delivers_history: self.inner.server_ingests_history(),
            })
            .await
            .map_err(Error::from)?;
        // The declared native chunk needs nothing here (this client replays
        // exactly the frames the engine emitted); whether the route wants
        // history decides what the replay branch buffers.
        self.history_on = response.history.is_some();
        Ok(())
    }

    fn next_request_id(&mut self) -> String {
        self.request_counter += 1;
        format!("{}:predict:{}", self.env_id, self.request_counter)
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
