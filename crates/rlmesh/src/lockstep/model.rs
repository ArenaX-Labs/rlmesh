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
use std::time::{Duration, Instant};

use bytes::Bytes;
use rlmesh_session::address::{Address, LEASE_GRACE, Placed, SessionAddress, place};
use rlmesh_session::transport::payload_bytes;
use rlmesh_session::wire::codec::{
    decode_model_reply, decode_model_request, encode_model_reply, encode_model_request,
};
use rlmesh_session::wire::timing::ns;
use rlmesh_session::wire::{
    EndpointTiming, ModelFault, ModelGroup, ModelOutcome, ModelReply, ModelRequest, ModelResolve,
    ModelResolved, ParticipantId, PriorityClass, Role,
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
use crate::model::handler::{
    ModelHandler, ModelResetOptions, ModelRouteSetup, PredictFrames, ResolveOptions,
};
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
        Error::InvalidArgument(invalid) if invalid.adapter_resolution.is_some() => {
            fault("ADAPTER_RESOLUTION", invalid.message.clone(), false)
        }
        Error::InvalidArgument(invalid) => fault("INVALID_REQUEST", invalid.message.clone(), false),
        Error::Environment(env) => fault("INTERNAL", env.message.clone(), false),
        other => fault("INTERNAL", other.to_string(), false),
    }
}

/// The inverse of [`fault_of`]: the error class a fault names.
fn error_of(fault: ModelFault) -> Error {
    match fault.code.as_str() {
        "ADAPTER_RESOLUTION" => Error::adapter_resolution(fault.message),
        "INVALID_REQUEST" => Error::invalid_argument(fault.message),
        _ if fault.recoverable => Error::model_recoverable(fault.message),
        _ => Error::model(fault.message),
    }
}

/// A link failure as the caller's error.
fn link_error(err: GymError) -> Error {
    match err {
        GymError::Timeout(t) => Error::Timeout(t),
        other => Error::Connection(other.to_string()),
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
    reset_options: Option<Arc<dyn ModelResetOptions>>,
    routes: Mutex<HashMap<String, Arc<Route>>>,
    describe: String,
    in_flight: std::sync::atomic::AtomicU32,
}

impl<H: ModelHandler + 'static> ModelService<H> {
    /// Serves `handler`, describing it with `describe` (a describe
    /// envelope, JSON; empty for none).
    pub fn new(handler: H, describe: String) -> Self {
        let setup = handler.route_setup();
        let reset_options = handler.reset_options();
        Self {
            handler: Arc::new(Mutex::new(handler)),
            setup,
            reset_options,
            routes: Mutex::default(),
            describe,
            in_flight: std::sync::atomic::AtomicU32::new(0),
        }
    }

    /// Answers one request, with what answering it measured.
    pub async fn answer(&self, request: ModelRequest) -> (ModelReply, EndpointTiming) {
        use std::sync::atomic::Ordering;
        let in_flight = self.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        let mut timing = EndpointTiming {
            in_flight: Some(in_flight),
            ..EndpointTiming::default()
        };
        let result = match request {
            ModelRequest::Resolve(resolve) => self.resolve(resolve).await.map(ModelReply::Resolved),
            ModelRequest::Predict(groups) => Ok(ModelReply::Predicted(
                self.predict(groups, &mut timing).await,
            )),
            ModelRequest::Reset { route, episode_ids } => self
                .reset(&route, episode_ids, &mut timing)
                .await
                .map(|()| ModelReply::Done),
            ModelRequest::Release { route } => {
                self.release(&route).await.map(|()| ModelReply::Done)
            }
            ModelRequest::Describe => Ok(ModelReply::Described(self.describe.clone())),
            ModelRequest::ResetOptions {
                route,
                episode,
                trial_index,
            } => self
                .options(route, episode, trial_index)
                .await
                .map(ModelReply::Options),
        };
        self.in_flight.fetch_sub(1, Ordering::Relaxed);
        let reply = result.unwrap_or_else(|error| ModelReply::Fault(fault_of(&error)));
        (reply, timing)
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
            reset_options: self.reset_options.is_some(),
        })
    }

    /// The options the model adds to the reset that starts `episode`.
    async fn options(
        &self,
        route: String,
        episode: rlmesh_session::wire::ModelEpisode,
        trial_index: Option<u64>,
    ) -> Result<String, Error> {
        let Some(source) = &self.reset_options else {
            return Ok(String::new());
        };
        let episode = EpisodeInfo {
            episode_id: episode.id,
            seed: episode.seed,
            env_id: route,
        };
        let options = source.reset_options(&episode, trial_index).await?;
        Ok(crate::lockstep::meta_json::optional_to_json(
            options.as_ref(),
        ))
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

    async fn predict(
        &self,
        groups: Vec<ModelGroup>,
        timing: &mut EndpointTiming,
    ) -> Vec<ModelOutcome> {
        let started = Instant::now();
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
        timing.decode_ns = ns(started.elapsed());
        let waited = Instant::now();
        let mut handler = self.handler.lock().await;
        timing.queue_ns = ns(waited.elapsed());
        let called = Instant::now();
        let mut frames = handler.predict_grouped(batch).await.into_iter();
        timing.user_ns = ns(called.elapsed());
        timing.adapter_ns = Some(handler.take_adapter_ns()).filter(|ns| *ns != 0);
        if let Some(held) = handler.held_state() {
            timing.held_episodes = Some(u32::try_from(held.episodes).unwrap_or(u32::MAX));
            timing.held_state_bytes = Some(held.bytes);
        }
        drop(handler);
        let encoded = Instant::now();
        let outcomes = prepared
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
            .collect();
        timing.encode_ns = ns(encoded.elapsed());
        outcomes
    }

    async fn reset(
        &self,
        route: &str,
        episode_ids: Vec<String>,
        timing: &mut EndpointTiming,
    ) -> Result<(), Error> {
        let called = Instant::now();
        if let Some(setup) = &self.setup {
            setup.reset_adapter(route, &episode_ids).await?;
        }
        let waited = Instant::now();
        let mut handler = self.handler.lock().await;
        timing.queue_ns = ns(waited.elapsed());
        let reset = handler.reset_adapter(route, episode_ids).await;
        // The route's cleanup and the model's own hook, less the lock wait.
        let queued = timing.queue_ns.unwrap_or(0);
        timing.user_ns = ns(called.elapsed()).map(|total| total.saturating_sub(queued));
        reset
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
            let arrived = Instant::now();
            let (reply, timing) = if query.parameters().contains_key(PROBE) {
                (ModelReply::Done, None)
            } else {
                let payload = query.payload().map(payload_bytes).unwrap_or_default();
                match decode_model_request(payload) {
                    Ok(request) => {
                        let decoded = arrived.elapsed();
                        let (reply, mut timing) = service.answer(request).await;
                        timing.decode_ns =
                            ns(decoded).map(|wire| wire + timing.decode_ns.unwrap_or(0));
                        timing.total_ns = ns(arrived.elapsed());
                        (reply, Some(timing))
                    }
                    Err(err) => (
                        ModelReply::Fault(fault("INVALID_REQUEST", err.to_string(), false)),
                        None,
                    ),
                }
            };
            if let Err(err) = query
                .reply(&key, encode_model_reply(&reply, timing.as_ref()))
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
#[derive(Clone)]
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

    /// Sends one request and waits for its reply and what the model
    /// measured answering it.
    pub async fn call(
        &self,
        request: &ModelRequest,
        timeout: Option<Duration>,
    ) -> Result<(ModelReply, Option<EndpointTiming>), GymError> {
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

/// A served model as a [`ModelHandler`]: drives the model at a participant
/// address through `m/{model}`, connecting on first use. What a run, a
/// binding or a relay drives a `Model(address)` with.
pub struct ServedModel {
    remote: Arc<Remote>,
}

struct Remote {
    address: String,
    timeout: Duration,
    connection: Mutex<Option<ModelConnection>>,
    /// Each resolved route's action space, to decode its frames.
    spaces: std::sync::Mutex<HashMap<String, SpaceSpec>>,
    /// What the model measured answering the last predict.
    last: std::sync::Mutex<EndpointTiming>,
    /// A resolved route said the model adds reset options.
    reset_options: std::sync::atomic::AtomicBool,
}

impl Remote {
    async fn link(&self) -> Result<ModelLink, Error> {
        let mut connection = self.connection.lock().await;
        if connection.is_none() {
            let id = format!("driver-{}", crate::lockstep::session::uuid_suffix());
            *connection = Some(
                ModelConnection::connect(&self.address, &id, self.timeout)
                    .await
                    .map_err(link_error)?,
            );
        }
        Ok(connection
            .as_ref()
            .map(|c| c.link.clone())
            .unwrap_or_else(|| unreachable!()))
    }

    async fn call(&self, request: ModelRequest) -> Result<ModelReply, Error> {
        let predict = matches!(request, ModelRequest::Predict(_));
        let (reply, timing) = self
            .link()
            .await?
            .call(&request, None)
            .await
            .map_err(link_error)?;
        if predict {
            *self
                .last
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = timing.unwrap_or_default();
        }
        match reply {
            ModelReply::Fault(fault) => Err(error_of(fault)),
            reply => Ok(reply),
        }
    }

    async fn close(&self) {
        if let Some(connection) = self.connection.lock().await.take() {
            connection.close().await;
        }
    }
}

impl ServedModel {
    /// The model at the participant address `address`; connecting waits up
    /// to `timeout` for it.
    pub fn new(address: impl Into<String>, timeout: Duration) -> Self {
        Self {
            remote: Arc::new(Remote {
                address: address.into(),
                timeout,
                connection: Mutex::default(),
                spaces: std::sync::Mutex::default(),
                last: std::sync::Mutex::default(),
                reset_options: std::sync::atomic::AtomicBool::new(false),
            }),
        }
    }

    /// The model's describe envelope (JSON; empty when it has none).
    pub async fn describe(&self) -> Result<String, Error> {
        match self.remote.call(ModelRequest::Describe).await? {
            ModelReply::Described(json) => Ok(json),
            other => Err(unexpected(&other)),
        }
    }
}

fn unexpected(reply: &ModelReply) -> Error {
    Error::Protocol(format!("unexpected model reply: {reply:?}"))
}

/// One observation as a predict group.
fn group_of(observation: ModelObservation) -> ModelGroup {
    let episodes = |rows: Vec<EpisodeInfo>| -> Vec<rlmesh_session::wire::ModelEpisode> {
        rows.into_iter()
            .map(|e| rlmesh_session::wire::ModelEpisode {
                id: e.episode_id,
                seed: e.seed,
            })
            .collect()
    };
    ModelGroup {
        route: observation.route.env_id,
        episodes: episodes(observation.route.episodes),
        observation: observation.observation.unwrap_or_default(),
        history: observation
            .history
            .into_iter()
            .map(|frame| rlmesh_session::wire::ModelHistoryRow {
                observation: frame.observation.unwrap_or_default(),
                episodes: episodes(frame.episodes),
                step: frame.step,
            })
            .collect(),
        step: observation.step,
    }
}

#[async_trait::async_trait]
impl ModelHandler for ServedModel {
    async fn predict(
        &mut self,
        observation: ModelObservation,
    ) -> crate::Result<Vec<rlmesh_spaces::SpaceValue>> {
        Ok(self.predict_chunked(observation).await?.actions)
    }

    async fn predict_chunked(
        &mut self,
        observation: ModelObservation,
    ) -> crate::Result<PredictFrames> {
        self.predict_grouped(vec![observation])
            .await
            .pop()
            .unwrap_or_else(|| Err(Error::model("no outcome for the predict")))
    }

    async fn predict_grouped(
        &mut self,
        observations: Vec<ModelObservation>,
    ) -> Vec<crate::Result<PredictFrames>> {
        let rows: Vec<(String, usize)> = observations
            .iter()
            .map(|o| (o.route.env_id.clone(), o.num_envs))
            .collect();
        let groups = observations.into_iter().map(group_of).collect();
        let outcomes = match self.remote.call(ModelRequest::Predict(groups)).await {
            Ok(ModelReply::Predicted(outcomes)) if outcomes.len() == rows.len() => outcomes,
            Ok(other) => {
                let error = unexpected(&other);
                return rows
                    .iter()
                    .map(|_| Err(Error::Protocol(error.to_string())))
                    .collect();
            }
            Err(error) => {
                return rows
                    .iter()
                    .map(|_| Err(Error::Connection(error.to_string())))
                    .collect();
            }
        };
        let spaces = self
            .remote
            .spaces
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        outcomes
            .into_iter()
            .zip(rows)
            .map(|(outcome, (route, rows))| {
                let frames = outcome.map_err(error_of)?;
                let space = spaces
                    .get(&route)
                    .ok_or_else(|| Error::model(format!("route {route:?} was not resolved")))?;
                let mut frames = frames.into_iter().map(|leaves| {
                    rlmesh_spaces::leaves::decode_leaf_slab(&leaves, space, rows)
                        .map_err(|e| Error::model(e.to_string()))
                });
                let actions = frames
                    .next()
                    .ok_or_else(|| Error::model("the model returned no frames"))??;
                Ok(PredictFrames {
                    actions,
                    replay: frames.collect::<crate::Result<_>>()?,
                })
            })
            .collect()
    }

    /// The model's held frame windows, as its last predict reported them.
    fn held_state(&self) -> Option<crate::model::handler::HeldState> {
        let last = *self
            .remote
            .last
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Some(crate::model::handler::HeldState {
            episodes: u64::from(last.held_episodes?),
            bytes: last.held_state_bytes.unwrap_or(0),
        })
    }

    /// The adapter share of the last predict, as the model reported it.
    fn take_adapter_ns(&mut self) -> u64 {
        self.remote
            .last
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .adapter_ns
            .take()
            .unwrap_or(0)
    }

    fn route_setup(&self) -> Option<Arc<dyn ModelRouteSetup>> {
        Some(Arc::new(ServedRoutes(self.remote.clone())))
    }

    async fn on_close(&mut self) -> crate::Result<()> {
        self.remote.close().await;
        Ok(())
    }

    fn reset_options(&self) -> Option<Arc<dyn ModelResetOptions>> {
        self.remote
            .reset_options
            .load(std::sync::atomic::Ordering::Relaxed)
            .then(|| Arc::new(ServedRoutes(self.remote.clone())) as _)
    }
}

#[async_trait::async_trait]
impl ModelResetOptions for ServedRoutes {
    async fn reset_options(
        &self,
        episode: &EpisodeInfo,
        trial_index: Option<u64>,
    ) -> crate::Result<Option<rlmesh_spaces::MetaMap>> {
        let reply = self
            .0
            .call(ModelRequest::ResetOptions {
                route: episode.env_id.clone(),
                episode: rlmesh_session::wire::ModelEpisode {
                    id: episode.episode_id.clone(),
                    seed: episode.seed,
                },
                trial_index,
            })
            .await?;
        let ModelReply::Options(json) = reply else {
            return Err(unexpected(&reply));
        };
        crate::lockstep::meta_json::optional_from_json(&json).map_err(Error::Protocol)
    }
}

/// A served model's routes: resolved, reset and released on the model.
struct ServedRoutes(Arc<Remote>);

#[async_trait::async_trait]
impl ModelRouteSetup for ServedRoutes {
    async fn resolve_adapter(
        &self,
        env_id: &str,
        env_contract: &EnvContract,
        options: ResolveOptions,
    ) -> crate::Result<crate::model::handler::RouteNeeds> {
        let reply = self
            .0
            .call(ModelRequest::Resolve(ModelResolve {
                route: env_id.to_string(),
                contract_json: crate::lockstep::map::contract_to_json(env_contract).to_string(),
                execution_horizon: options.execution_horizon,
                delivers_history: options.delivers_history,
                workflow_edition: String::new(),
            }))
            .await?;
        let ModelReply::Resolved(resolved) = reply else {
            return Err(unexpected(&reply));
        };
        if resolved.reset_options {
            self.0
                .reset_options
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
        if let Some(space) = &env_contract.action_space {
            self.0
                .spaces
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .insert(env_id.to_string(), space.clone());
        }
        Ok(crate::model::handler::RouteNeeds {
            native_chunk: resolved.native_chunk,
            history: resolved
                .history
                .map(|(keys, prunable)| crate::model::handler::HistoryNeeds { keys, prunable }),
        })
    }

    async fn reset_adapter(&self, env_id: &str, episode_ids: &[String]) -> crate::Result<()> {
        self.0
            .call(ModelRequest::Reset {
                route: env_id.to_string(),
                episode_ids: episode_ids.to_vec(),
            })
            .await
            .map(|_| ())
    }

    async fn release_adapter(&self, env_id: &str) -> crate::Result<()> {
        self.0
            .spaces
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(env_id);
        self.0
            .call(ModelRequest::Release {
                route: env_id.to_string(),
            })
            .await
            .map(|_| ())
    }
}
