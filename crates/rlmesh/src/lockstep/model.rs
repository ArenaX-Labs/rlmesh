//! A model served on a session: one request, one reply, on `m/{model}`.
//!
//! ```text
//! driver ──ModelRequest::Resolve{route, contract}──▶ model   (once per env)
//! driver ──ModelRequest::Predict{groups}──────────▶ model   (per decision)
//! driver ◀──ModelReply::Predicted{outcomes}──────── model
//! ```
//!
//! The model side is a [`ModelHandler`] (the engine a local run drives),
//! answered by [`ModelService`]: routes resolve off the predict lock, every
//! forward runs under it, and each group of a predict gets its own outcome.
//! [`ModelBinding::serve`] joins a session and answers on the lockstep
//! class session, as a lockstep env does. The driver side is a
//! [`ModelLink`].
//!
//! A probe (the selector parameter `probe`) is answered with an empty
//! reply, so a driver knows the route to the model is up.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use rlmesh_session::address::{Address, LEASE_GRACE, Placed, SessionAddress, place};
use rlmesh_session::transport::payload_bytes;
use rlmesh_session::wire::codec::{
    decode_model_reply, decode_model_request, encode_model_reply, encode_model_request,
};
use rlmesh_session::wire::{
    ModelFault, ModelGroup, ModelOutcome, ModelReply, ModelRequest, ModelResolve, ModelResolved,
    ParticipantId, PriorityClass, Role,
};
use rlmesh_session::{Participant, SessionError};
use rlmesh_spaces::leaves::encode_leaf_slab;
use rlmesh_spaces::{EnvContract, SpaceSpec};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};
use zenoh::qos::{CongestionControl, Priority};
use zenoh::query::ConsolidationMode;

use crate::Error;
use crate::lockstep::error::GymError;
use crate::lockstep::map::contract_from_json;
use crate::lockstep::session::lockstep_session;
use crate::model::handler::{ModelHandler, ModelRouteSetup, PredictFrames, ResolveOptions};
use crate::model::types::{EpisodeInfo, HistoryFrame, ModelObservation, ModelRouteContext};

/// The selector parameter of a probe.
const PROBE: &str = "probe";

/// A fault with `code`.
fn fault(code: &str, message: impl Into<String>, recoverable: bool) -> ModelFault {
    ModelFault {
        code: code.into(),
        message: message.into(),
        recoverable,
    }
}

/// A handler's error as a fault: a model error's own message and
/// recoverability, anything else as an internal failure.
fn fault_of(error: &Error) -> ModelFault {
    match error {
        Error::Attached(attached) => {
            let mut wire = fault_of(attached.error());
            if let Some(message) = attached.authored_message() {
                wire.message = message.to_owned();
            }
            wire
        }
        Error::Model(model) => fault("INTERNAL", model.message.clone(), model.is_recoverable),
        Error::Environment(env) => fault("INTERNAL", env.message.clone(), false),
        other => fault("INTERNAL", other.to_string(), false),
    }
}

/// A resolved route.
struct Route {
    contract: Arc<EnvContract>,
    action_space: SpaceSpec,
}

/// Answers a served model's requests with a [`ModelHandler`].
pub struct ModelService<H> {
    handler: Arc<Mutex<H>>,
    setup: Option<Arc<dyn ModelRouteSetup>>,
    routes: Mutex<HashMap<String, Arc<Route>>>,
    describe: String,
}

impl<H: ModelHandler + 'static> ModelService<H> {
    /// Serves `handler`, describing it with `describe` (a describe
    /// envelope, JSON; empty for none).
    pub fn new(handler: H, describe: String) -> Self {
        let setup = handler.route_setup();
        Self {
            handler: Arc::new(Mutex::new(handler)),
            setup,
            routes: Mutex::default(),
            describe,
        }
    }

    /// Answers one request.
    pub async fn answer(&self, request: ModelRequest) -> ModelReply {
        let result = match request {
            ModelRequest::Resolve(resolve) => self.resolve(resolve).await.map(ModelReply::Resolved),
            ModelRequest::Predict(groups) => Ok(ModelReply::Predicted(self.predict(groups).await)),
            ModelRequest::Reset { route, episode_ids } => self
                .reset(&route, episode_ids)
                .await
                .map(|()| ModelReply::Done),
            ModelRequest::Release { route } => {
                self.release(&route).await.map(|()| ModelReply::Done)
            }
            ModelRequest::Describe => Ok(ModelReply::Described(self.describe.clone())),
        };
        result.unwrap_or_else(|error| ModelReply::Fault(fault_of(&error)))
    }

    async fn resolve(&self, request: ModelResolve) -> Result<ModelResolved, Error> {
        if request.route.is_empty() {
            return Err(Error::invalid_argument("a resolve names no route"));
        }
        if !request.workflow_edition.is_empty() {
            rlmesh_proto::parse_retained_edition(&request.workflow_edition).map_err(|why| {
                Error::invalid_argument(format!(
                    "the driver pinned workflow edition {:?}, which this model build does not \
                     implement: {why}",
                    request.workflow_edition
                ))
            })?;
        }
        let value: serde_json::Value = serde_json::from_str(&request.contract_json)
            .map_err(|e| Error::invalid_argument(format!("env contract: {e}")))?;
        let contract =
            contract_from_json(&value).map_err(|e| Error::invalid_argument(e.to_string()))?;
        let action_space = contract.action_space.clone().ok_or_else(|| {
            Error::invalid_argument("the env has no action space; a model cannot act on it")
        })?;
        let mut needs = crate::model::handler::RouteNeeds::default();
        if let Some(setup) = &self.setup {
            needs = setup
                .resolve_adapter(
                    &request.route,
                    &contract,
                    ResolveOptions {
                        execution_horizon: request.execution_horizon,
                        delivers_history: request.delivers_history,
                    },
                )
                .await?;
        }
        self.routes.lock().await.insert(
            request.route,
            Arc::new(Route {
                contract: Arc::new(contract),
                action_space,
            }),
        );
        Ok(ModelResolved {
            native_chunk: needs.native_chunk,
            history: needs.history.map(|h| (h.keys, h.prunable)),
        })
    }

    async fn route(&self, name: &str) -> Result<Arc<Route>, Error> {
        self.routes
            .lock()
            .await
            .get(name)
            .cloned()
            .ok_or_else(|| Error::model(format!("route {name:?} was not resolved")))
    }

    /// One group as the handler's observation, with its route.
    async fn observation(
        &self,
        group: ModelGroup,
    ) -> Result<(ModelObservation, Arc<Route>), Error> {
        if group.episodes.is_empty() {
            return Err(Error::model("a predict group has no rows"));
        }
        let route = self.route(&group.route).await?;
        let episodes = |rows: Vec<rlmesh_session::wire::ModelEpisode>| -> Vec<EpisodeInfo> {
            rows.into_iter()
                .map(|e| EpisodeInfo {
                    episode_id: e.id,
                    seed: e.seed,
                    env_id: group.route.clone(),
                })
                .collect()
        };
        let observation = ModelObservation {
            observation: Some(group.observation),
            num_envs: group.episodes.len(),
            route: ModelRouteContext {
                session_id: String::new(),
                env_id: group.route.clone(),
                request_id: String::new(),
                episodes: episodes(group.episodes),
            },
            env_contract: Some(route.contract.clone()),
            history: group
                .history
                .into_iter()
                .map(|row| HistoryFrame {
                    observation: Some(row.observation),
                    episodes: episodes(row.episodes),
                    step: row.step,
                })
                .collect(),
            step: group.step,
        };
        Ok((observation, route))
    }

    async fn predict(&self, groups: Vec<ModelGroup>) -> Vec<ModelOutcome> {
        let mut prepared = Vec::with_capacity(groups.len());
        let mut batch = Vec::with_capacity(groups.len());
        for group in groups {
            match self.observation(group).await {
                Ok((observation, route)) => {
                    prepared.push(Ok((observation.num_envs, route)));
                    batch.push(observation);
                }
                Err(error) => prepared.push(Err(error)),
            }
        }
        let mut frames = self
            .handler
            .lock()
            .await
            .predict_grouped(batch)
            .await
            .into_iter();
        prepared
            .into_iter()
            .map(|group| {
                let (rows, route) = group.map_err(|e| fault_of(&e))?;
                let frames = frames
                    .next()
                    .unwrap_or_else(|| {
                        Err(Error::model(
                            "predict_grouped returned fewer results than groups",
                        ))
                    })
                    .map_err(|e| fault_of(&e))?;
                encode_frames(frames, rows, &route.action_space).map_err(|e| fault_of(&e))
            })
            .collect()
    }

    async fn reset(&self, route: &str, episode_ids: Vec<String>) -> Result<(), Error> {
        if let Some(setup) = &self.setup {
            setup.reset_adapter(route, &episode_ids).await?;
        }
        self.handler
            .lock()
            .await
            .reset_adapter(route, episode_ids)
            .await
    }

    async fn release(&self, route: &str) -> Result<(), Error> {
        self.routes.lock().await.remove(route);
        if let Some(setup) = &self.setup {
            setup.release_adapter(route).await?;
        }
        Ok(())
    }

    /// Drops every route and closes the handler.
    pub async fn close(&self) -> Result<(), Error> {
        let routes: Vec<String> = self.routes.lock().await.drain().map(|(r, _)| r).collect();
        if let Some(setup) = &self.setup {
            for route in &routes {
                setup.release_adapter(route).await?;
            }
        }
        self.handler.lock().await.on_close().await
    }
}

/// A group's frames as leaf slabs: frame 0, then the replay frames, each
/// `rows` actions checked against the action space.
fn encode_frames(
    frames: PredictFrames,
    rows: usize,
    space: &SpaceSpec,
) -> Result<Vec<Vec<Bytes>>, Error> {
    std::iter::once(frames.actions)
        .chain(frames.replay)
        .map(|frame| {
            if frame.len() != rows {
                return Err(Error::model(format!(
                    "predict returned {} actions for {rows} rows",
                    frame.len()
                )));
            }
            crate::model::wire::check_actions_conform(space, &frame)?;
            encode_leaf_slab(&frame, space).map_err(|e| Error::model(e.to_string()))
        })
        .collect()
}

/// Answers `service` on `key` until `cancel` fires. Requests run side by
/// side; the handler's lock orders forwards.
pub async fn serve_zenoh<H: ModelHandler + 'static>(
    session: zenoh::Session,
    key: String,
    service: Arc<ModelService<H>>,
    cancel: CancellationToken,
) -> Result<(), GymError> {
    let queryable = session
        .declare_queryable(&key)
        .complete(true)
        .await
        .map_err(|e| GymError::Closed(e.to_string()))?;
    loop {
        let query = tokio::select! {
            () = cancel.cancelled() => break,
            query = queryable.recv_async() => match query {
                Ok(query) => query,
                Err(_) => break,
            },
        };
        let (service, key) = (service.clone(), key.clone());
        tokio::spawn(async move {
            let reply = if query.parameters().contains_key(PROBE) {
                ModelReply::Done
            } else {
                let payload = query.payload().map(payload_bytes).unwrap_or_default();
                match decode_model_request(payload) {
                    Ok(request) => service.answer(request).await,
                    Err(err) => ModelReply::Fault(fault("INVALID_REQUEST", err.to_string(), false)),
                }
            };
            if let Err(err) = query
                .reply(&key, encode_model_reply(&reply))
                .express(true)
                .await
            {
                warn!(%err, "model reply not sent");
            }
        });
    }
    Ok(())
}

/// How long a model query waits when the request sets no deadline.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(600);

/// The driver side's way to a served model: queries on `m/{model}`.
pub struct ModelLink {
    session: zenoh::Session,
    key: String,
}

impl ModelLink {
    /// A link to the model answering on `key`.
    pub fn new(session: zenoh::Session, key: String) -> Self {
        Self { session, key }
    }

    /// Waits until the model answers, probing every 20 ms for up to
    /// `timeout`.
    pub async fn wait_ready(&self, timeout: Duration) -> Result<(), GymError> {
        let deadline = tokio::time::Instant::now() + timeout;
        let selector = format!("{}?{PROBE}", self.key);
        loop {
            let replies = self
                .session
                .get(&selector)
                .consolidation(ConsolidationMode::None)
                .timeout(Duration::from_millis(500))
                .await
                .map_err(|e| GymError::Closed(e.to_string()))?;
            if let Ok(reply) = replies.recv_async().await
                && reply.result().is_ok()
            {
                return Ok(());
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(GymError::Timeout(timeout));
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// Sends one request and waits for its reply.
    pub async fn call(
        &self,
        request: &ModelRequest,
        timeout: Option<Duration>,
    ) -> Result<ModelReply, GymError> {
        let timeout = timeout.unwrap_or(DEFAULT_TIMEOUT);
        let replies = self
            .session
            .get(&self.key)
            .payload(encode_model_request(request))
            .consolidation(ConsolidationMode::None)
            .priority(Priority::RealTime)
            .congestion_control(CongestionControl::Block)
            .express(true)
            .timeout(timeout)
            .await
            .map_err(|e| GymError::Closed(e.to_string()))?;
        let reply = replies
            .recv_async()
            .await
            .map_err(|_| GymError::Timeout(timeout))?;
        match reply.result() {
            Ok(sample) => Ok(decode_model_reply(payload_bytes(sample.payload()))?),
            // A Zenoh-level error (the queryable is gone).
            Err(_) => Err(GymError::Closed(format!("{} did not answer", self.key))),
        }
    }
}

/// A served model's participant address, its session placed.
pub struct ModelBinding {
    session: SessionAddress,
    name: ParticipantId,
    placed: Placed,
}

impl ModelBinding {
    /// Places the session of the participant address `address` (the
    /// model's name is its participant) by the hosting rule.
    pub async fn bind(address: &str) -> Result<Self, GymError> {
        let parsed = Address::participant(address).map_err(|e| GymError::Config(e.to_string()))?;
        let name = ParticipantId::new(parsed.participant.clone().unwrap_or_default())
            .map_err(|e| GymError::Config(e.to_string()))?;
        let placed = place(&parsed, LEASE_GRACE).await?;
        Ok(Self {
            session: placed.address.clone(),
            name,
            placed,
        })
    }

    /// The full address drivers reach the model at.
    pub fn address(&self) -> String {
        self.session.participant(self.name.as_str())
    }

    /// Joins the session as the model and answers `service` until
    /// `shutdown` fires, then closes it.
    pub async fn serve<H: ModelHandler + 'static>(
        self,
        service: Arc<ModelService<H>>,
        timeout: Duration,
        shutdown: CancellationToken,
    ) -> Result<(), GymError> {
        let result = self.serve_inner(&service, timeout, shutdown).await;
        let closed = service.close().await;
        let mut placed = self.placed;
        let _ = placed.close().await;
        result?;
        closed.map_err(|e| GymError::Config(e.to_string()))
    }

    async fn serve_inner<H: ModelHandler + 'static>(
        &self,
        service: &Arc<ModelService<H>>,
        timeout: Duration,
        shutdown: CancellationToken,
    ) -> Result<(), GymError> {
        // A model drives nothing itself: it joins as a controller at model
        // priority, so the roster shows it, and never takes a lease.
        let participant: Participant = self
            .session
            .join(
                self.name.as_str(),
                vec![Role::Controller],
                PriorityClass::Model,
                timeout,
            )
            .await?;
        let lockstep = lockstep_session(participant.transport(), self.session.base).await?;
        let key = participant.keys().model(&self.name);
        info!(model = %self.name, %key, "model serving");
        let served = serve_zenoh(lockstep.clone(), key, service.clone(), shutdown).await;
        if !participant.transport().is_shared() {
            let _ = lockstep.close().await;
        }
        participant.close().await;
        served
    }
}

/// A driver's connection to a served model: its participant in the
/// model's session, and the link.
pub struct ModelConnection {
    /// The link to the model.
    pub link: ModelLink,
    participant: Participant,
    lockstep: Option<zenoh::Session>,
    placed: Placed,
}

impl ModelConnection {
    /// Connects to the model at the participant address `address`: joins
    /// its session as `id` and waits up to `timeout` for the model to
    /// answer.
    pub async fn connect(address: &str, id: &str, timeout: Duration) -> Result<Self, GymError> {
        let parsed = Address::participant(address).map_err(|e| GymError::Config(e.to_string()))?;
        let model = ParticipantId::new(parsed.participant.clone().unwrap_or_default())
            .map_err(|e| GymError::Config(e.to_string()))?;
        let placed = place(&parsed, LEASE_GRACE).await?;
        let participant = placed
            .address
            .join(id, vec![Role::Controller], PriorityClass::Model, timeout)
            .await?;
        let session = lockstep_session(participant.transport(), placed.address.base).await?;
        let link = ModelLink::new(session.clone(), participant.keys().model(&model));
        link.wait_ready(timeout).await.map_err(|err| match err {
            GymError::Timeout(_) => GymError::Session(SessionError::Config(format!(
                "no model {model} answered at {address} within {timeout:?}"
            ))),
            other => other,
        })?;
        let lockstep = (!participant.transport().is_shared()).then_some(session);
        Ok(Self {
            link,
            participant,
            lockstep,
            placed,
        })
    }

    /// Leaves the model's session.
    pub async fn close(self) {
        if let Err(err) = self.participant.detach().await {
            tracing::debug!(%err, "model client: detach");
        }
        if let Some(lockstep) = self.lockstep {
            let _ = lockstep.close().await;
        }
        let mut placed = self.placed;
        let _ = placed.close().await;
    }
}
