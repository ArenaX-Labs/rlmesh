//! A served model as a [`ModelHandler`] ([`ServedModel`]): drives a model at
//! a participant address, connecting on first use.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use rlmesh_session::wire::{EndpointTiming, ModelGroup, ModelReply, ModelRequest, ModelResolve};
use rlmesh_spaces::{EnvContract, SpaceSpec};
use tokio::sync::Mutex;

use super::link::{ModelConnection, ModelLink};
use super::{error_of, link_error};
use crate::Error;
use crate::lockstep::error::GymError;
use crate::model::handler::{
    ModelHandler, ModelResetOptions, ModelRouteSetup, PredictFrames, ResolveOptions,
};
use crate::model::types::{EpisodeInfo, ModelObservation};

/// A served model as a [`ModelHandler`]: drives the model at a participant
/// address through `m/{model}`, connecting on first use. What a run, a
/// binding or a relay drives a `Model(address)` with.
pub struct ServedModel {
    remote: Arc<Remote>,
}

struct Remote {
    address: String,
    /// How long joining the model's session may take.
    timeout: Duration,
    /// How long each request may wait for the model's answer once joined.
    deadline: std::sync::Mutex<Duration>,
    /// The session the run drives in, which the model is invited into.
    target: std::sync::Mutex<Option<String>>,
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
            let mut connected = ModelConnection::connect(&self.address, &id, self.timeout)
                .await
                .map_err(link_error)?;
            let target = self
                .target
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone();
            // A model already in the run's session (an env served into the
            // model's own) answers there as it is.
            if let Some(session) = target.filter(|session| !connected.answers_in(session)) {
                connected = connected
                    .invite(&session, &id, self.timeout)
                    .await
                    .map_err(link_error)?;
            }
            *connection = Some(connected);
        }
        Ok(connection
            .as_ref()
            .map(|c| c.link.clone())
            .unwrap_or_else(|| unreachable!()))
    }

    async fn workflow_editions(&self) -> Result<Option<rlmesh_proto::SessionOffer>, Error> {
        self.link().await?;
        Ok(self
            .connection
            .lock()
            .await
            .as_ref()
            .and_then(ModelConnection::workflow_editions))
    }

    async fn call(&self, request: ModelRequest) -> Result<ModelReply, Error> {
        let predict = matches!(request, ModelRequest::Predict(_));
        let deadline = *self
            .deadline
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let called = self.link().await?.call(&request, Some(deadline)).await;
        if let Err(GymError::Closed(_)) = &called {
            // The model or the link to it is gone: the next call joins again
            // rather than meet the same dead link, which is what makes the
            // failure worth retrying.
            self.connection.lock().await.take();
        }
        let (reply, timing) = called.map_err(link_error)?;
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
    /// The model at the participant address `address`; connecting (joining
    /// its session and waiting for it to answer) waits up to `timeout` for
    /// it. Each request after that waits up to [`DEFAULT_TIMEOUT`] for its
    /// answer ([`Self::with_request_timeout`]).
    ///
    /// [`DEFAULT_TIMEOUT`]: super::DEFAULT_TIMEOUT
    pub fn new(address: impl Into<String>, timeout: Duration) -> Self {
        Self {
            remote: Arc::new(Remote {
                address: address.into(),
                timeout,
                deadline: std::sync::Mutex::new(super::DEFAULT_TIMEOUT),
                target: std::sync::Mutex::default(),
                connection: Mutex::default(),
                spaces: std::sync::Mutex::default(),
                last: std::sync::Mutex::default(),
                reset_options: std::sync::atomic::AtomicBool::new(false),
            }),
        }
    }

    /// Bounds each request to the model once joined (a predict, a route's
    /// resolve, reset or release, a describe) by `deadline`, apart from the
    /// join's timeout: a model that has not answered by then fails the
    /// request with [`Error::Timeout`], which a retry may get past. A model
    /// or link that goes away fails it at once, with [`Error::Connection`].
    #[must_use]
    pub fn with_request_timeout(self, deadline: Duration) -> Self {
        *self
            .remote
            .deadline
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = deadline;
        self
    }

    /// The model's describe envelope (JSON; empty when it has none).
    pub async fn describe(&self) -> Result<String, Error> {
        match self.remote.call(ModelRequest::Describe).await? {
            ModelReply::Described(json) => Ok(json),
            other => Err(unexpected(&other)),
        }
    }

    /// The execution horizon the model at `address` runs at when a run names
    /// none, read once from its describe envelope over a connection of its
    /// own: the horizon its `load()` chose, else its spec's
    /// `timing.execution`; `None` when it declares neither.
    pub async fn default_execution_horizon(
        address: &str,
        timeout: Duration,
    ) -> Result<Option<u32>, Error> {
        let probe = Self::new(address, timeout);
        let described = probe.describe().await;
        probe.remote.close().await;
        envelope_execution_horizon(&described?)
    }
}

/// The horizon a describe envelope declares: its `execution_horizon`, else
/// its model spec's `timing.execution` (an empty envelope declares none).
pub(crate) fn envelope_execution_horizon(envelope: &str) -> Result<Option<u32>, Error> {
    if envelope.trim().is_empty() {
        return Ok(None);
    }
    let envelope: serde_json::Value = serde_json::from_str(envelope)
        .map_err(|why| Error::Protocol(format!("model describe envelope: {why}")))?;
    let value = envelope
        .get("execution_horizon")
        .filter(|v| !v.is_null())
        .or_else(|| {
            envelope
                .pointer("/model_spec/timing/execution")
                .filter(|v| !v.is_null())
        });
    let Some(value) = value else {
        return Ok(None);
    };
    let horizon = value
        .as_u64()
        .and_then(|h| u32::try_from(h).ok())
        .ok_or_else(|| {
            Error::invalid_argument(format!("invalid model execution_horizon: {value}"))
        })?;
    super::super::super::run::resolve_execution_horizon(Some(horizon.into()), None, None, None)
        .map(Some)
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
            // Each row fails as the call did: a refusal stays final, a lost
            // link or a deadline stays worth retrying.
            Err(error) => {
                return rows.iter().map(|_| Err(error.clone())).collect();
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

    /// The model's own timing of the last predict, as it reported it.
    fn take_endpoint_timing(&mut self) -> Option<(Option<u64>, rlmesh_proto::EndpointPhases)> {
        let timing = std::mem::take(
            &mut *self
                .remote
                .last
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        Some((
            timing.total_ns,
            rlmesh_proto::EndpointPhases {
                decode_ns: timing.decode_ns.unwrap_or(0),
                user_ns: timing.user_ns.unwrap_or(0),
                encode_ns: timing.encode_ns.unwrap_or(0),
                queue_ns: timing.queue_ns.unwrap_or(0),
                in_flight: timing.in_flight.unwrap_or(0),
                ..rlmesh_proto::EndpointPhases::default()
            },
        ))
    }

    fn route_setup(&self) -> Option<Arc<dyn ModelRouteSetup>> {
        Some(Arc::new(ServedRoutes(self.remote.clone())))
    }

    async fn on_close(&mut self) -> crate::Result<()> {
        self.remote.close().await;
        Ok(())
    }

    /// Invites the model into `session` at the next connect, leaving its
    /// own session (a connection made before closes).
    async fn join(&mut self, session: &str) -> crate::Result<()> {
        *self
            .remote
            .target
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(session.to_string());
        self.remote.close().await;
        Ok(())
    }

    async fn workflow_editions(&mut self) -> crate::Result<Option<rlmesh_proto::SessionOffer>> {
        self.remote.workflow_editions().await
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
                workflow_edition: options
                    .workflow_edition
                    .map(|edition| edition.to_string())
                    .unwrap_or_default(),
                reads_advisories: true,
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
            advisories: resolved
                .advisories
                .into_iter()
                .map(|(severity, message)| rlmesh_adapters::v1::Advisory {
                    severity: if severity == "info" {
                        rlmesh_adapters::v1::AdvisorySeverity::Info
                    } else {
                        rlmesh_adapters::v1::AdvisorySeverity::Caution
                    },
                    message,
                })
                .collect(),
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

#[cfg(test)]
mod horizon_envelope_tests {
    use super::envelope_execution_horizon as read;

    #[test]
    fn served_timing_is_validated_and_loaded_timing_wins() {
        assert_eq!(read("").unwrap(), None);
        assert_eq!(read("{}").unwrap(), None);
        assert_eq!(
            read(r#"{"model_spec":{"timing":{"execution":2}}}"#).unwrap(),
            Some(2)
        );
        assert_eq!(
            read(r#"{"execution_horizon":3,"model_spec":{"timing":{"execution":2}}}"#).unwrap(),
            Some(3)
        );
        assert_eq!(
            read(r#"{"execution_horizon":null,"model_spec":{"timing":{"execution":2}}}"#).unwrap(),
            Some(2)
        );
        for bad in ["-1", "0", "1025", "4294967296", "1.5", r#""3""#] {
            assert!(read(&format!(r#"{{"execution_horizon":{bad}}}"#)).is_err());
        }
        assert!(read("broken").is_err());
    }
}
