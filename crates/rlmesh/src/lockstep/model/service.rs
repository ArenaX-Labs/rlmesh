//! The served-model exchange: [`ModelService`] answering resolves, predicts,
//! resets and releases with its handler.

use std::sync::Arc;
use std::time::Instant;

use bytes::Bytes;
use rlmesh_session::wire::timing::ns;
use rlmesh_session::wire::{
    EndpointTiming, ModelGroup, ModelOutcome, ModelReply, ModelRequest, ModelResolve, ModelResolved,
};
use rlmesh_spaces::SpaceSpec;
use rlmesh_spaces::leaves::encode_leaf_slab;
use tokio::sync::{Mutex, watch};

use super::batcher::Batching;
use super::{ModelService, Route, fault, fault_of};
use crate::Error;
use crate::lockstep::map::contract_from_json;
use crate::model::handler::{ModelHandler, PredictFrames, ResolveOptions};
use crate::model::types::{EpisodeInfo, HistoryFrame, ModelObservation, ModelRouteContext};
use crate::serve_options::BatchOptions;

impl<H: ModelHandler + 'static> ModelService<H> {
    /// Serves `handler`, describing it with `describe` (a describe
    /// envelope, JSON; empty for none).
    pub fn new(handler: H, describe: String) -> Self {
        let setup = handler.route_setup();
        let reset_options = handler.reset_options();
        let fuses = handler.fuses_groups();
        Self {
            handler: Arc::new(Mutex::new(handler)),
            setup,
            reset_options,
            routes: Mutex::default(),
            released: tokio::sync::Notify::new(),
            describe,
            editions: crate::editions::offer(None),
            flights: Arc::default(),
            closing: std::sync::atomic::AtomicBool::new(false),
            closed: watch::channel(None).0,
            fuses,
            batching: fuses.then(|| Batching::new(BatchOptions::default(), None)),
        }
    }

    /// Coalesces predicts as `options` say (a model that does not fuse is
    /// never batched). Fails when the batch record file can't be opened.
    pub fn batching(mut self, options: BatchOptions) -> Result<Self, Error> {
        if !self.fuses {
            self.batching = None;
            return Ok(self);
        }
        let records = options
            .batch_records
            .as_ref()
            .map(|path| {
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .map_err(|e| {
                        Error::invalid_argument(format!(
                            "batch_records: can't open {}: {e}",
                            path.display()
                        ))
                    })
            })
            .transpose()?;
        self.batching = Some(Batching::new(options, records));
        Ok(self)
    }

    /// Declares the workflow edition the model was authored against (`None`:
    /// this build's current).
    pub fn declaring(mut self, workflow_edition: Option<&str>) -> Self {
        self.editions = crate::editions::offer(workflow_edition);
        self
    }

    /// Answers one request, with what answering it measured. An accepted
    /// request runs to its end, holding its flight, even when this is
    /// dropped: its forward may run on where a cancel can't reach it.
    pub async fn answer(self: &Arc<Self>, request: ModelRequest) -> (ModelReply, EndpointTiming) {
        // Taken before the check, so a close either sees it or refuses it.
        let (flight, in_flight) = self.flights.take();
        let timing = move || EndpointTiming {
            in_flight: Some(in_flight),
            ..EndpointTiming::default()
        };
        if self.closing.load(std::sync::atomic::Ordering::SeqCst) {
            let closing = fault("UNAVAILABLE", "the model is closing", false);
            return (ModelReply::Fault(closing), timing());
        }
        let service = Arc::clone(self);
        let accepted = tokio::spawn(async move {
            let _flight = flight;
            service.accepted(request, timing()).await
        });
        match accepted.await {
            Ok(answered) => answered,
            Err(e) if e.is_panic() => std::panic::resume_unwind(e.into_panic()),
            Err(e) => {
                let stopped = fault("UNAVAILABLE", format!("the request stopped: {e}"), false);
                (ModelReply::Fault(stopped), timing())
            }
        }
    }

    async fn accepted(
        self: &Arc<Self>,
        request: ModelRequest,
        mut timing: EndpointTiming,
    ) -> (ModelReply, EndpointTiming) {
        let result = match request {
            ModelRequest::Resolve(resolve) => self.resolve(resolve).await.map(ModelReply::Resolved),
            ModelRequest::Predict(groups) => Ok(ModelReply::Predicted(
                self.coalesce(groups, &mut timing).await,
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
            // The serving loop answers invitations.
            ModelRequest::Join { .. } | ModelRequest::Leave { .. } => Err(Error::invalid_argument(
                "an invitation reaches a model through its session",
            )),
        };
        let reply = result.unwrap_or_else(|error| ModelReply::Fault(fault_of(&error)));
        (reply, timing)
    }

    async fn resolve(&self, request: ModelResolve) -> Result<ModelResolved, Error> {
        if request.route.is_empty() {
            return Err(Error::invalid_argument("a resolve names no route"));
        }
        let workflow_edition = (!request.workflow_edition.is_empty())
            .then(|| {
                rlmesh_proto::parse_retained_edition(&request.workflow_edition).map_err(|why| {
                    Error::invalid_argument(format!(
                        "the driver pinned workflow edition {:?}, which this model build does \
                         not implement: {why}",
                        request.workflow_edition
                    ))
                })
            })
            .transpose()?;
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
                        workflow_edition,
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
            // Only to a driver that reads them.
            advisories: if request.reads_advisories {
                needs
                    .advisories
                    .into_iter()
                    .map(|note| (note.severity.as_str().to_owned(), note.message))
                    .collect()
            } else {
                Vec::new()
            },
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

    pub(super) async fn predict(
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
        // A batch of unresolved routes alone has nothing to predict.
        let mut frames = if batch.is_empty() {
            Vec::new()
        } else {
            handler.predict_grouped(batch).await
        }
        .into_iter();
        timing.user_ns = ns(called.elapsed());
        timing.adapter_ns = Some(handler.take_adapter_ns()).filter(|ns| *ns != 0);
        if let Some(held) = handler.held_state() {
            timing.held_episodes = Some(u32::try_from(held.episodes).unwrap_or(u32::MAX));
            timing.held_state_bytes = Some(held.bytes);
        }
        // A model served elsewhere (this one proxies it) reports its own
        // phases: its user time replaces the call's wall time, which holds
        // the hop, and its decode, encode and queue add to this side's.
        let backend = handler.take_endpoint_timing().map(|(_, phases)| phases);
        drop(handler);
        if let Some(backend) = &backend {
            timing.user_ns = Some(backend.user_ns).filter(|ns| *ns != 0);
            timing.decode_ns = sum_ns(timing.decode_ns, backend.decode_ns);
            timing.queue_ns = sum_ns(timing.queue_ns, backend.queue_ns);
        }
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
        if let Some(backend) = &backend {
            timing.encode_ns = sum_ns(timing.encode_ns, backend.encode_ns);
        }
        outcomes
    }

    /// Records batches under the model's `name`, reached at `address`.
    pub(super) fn serving_as(&self, name: &str, address: String) {
        if let Some(batching) = &self.batching {
            let _ = batching.serving.set((name.to_string(), address));
        }
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
        self.released.notify_waiters();
        if let Some(setup) = &self.setup {
            setup.release_adapter(route).await?;
        }
        Ok(())
    }
}

/// A phase measured here plus the backend's share of it; zero as `None`.
fn sum_ns(here: Option<u64>, backend: u64) -> Option<u64> {
    Some(here.unwrap_or(0).saturating_add(backend)).filter(|ns| *ns != 0)
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
            crate::model::types::check_actions_conform(space, &frame)?;
            encode_leaf_slab(&frame, space).map_err(|e| Error::model(e.to_string()))
        })
        .collect()
}
