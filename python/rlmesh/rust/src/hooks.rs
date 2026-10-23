//! The native run's observer seam: runtime events delivered to one Python relay.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rlmesh::binding::model::CallerJobs;
use rlmesh::binding::runtime::spec::EpisodeSummary;
use rlmesh::binding::runtime::{
    LogEvent, SessionEndedEvent, SessionFailedEvent, TelemetrySnapshotEvent,
};
use rlmesh::binding::{
    ActionReceivedEvent, CancellationToken, EpisodeCompletedEvent, EpisodeStartedEvent, HookError,
    ObservationEmittedEvent, RunDirective, RuntimeHooks, SessionStartedEvent, StepCompletedEvent,
};
use rlmesh_grpc::wire::{
    Bytes, decode_batched_partial_values, env_contract_from_proto, leaves_value,
    meta_map_from_proto, space_spec_from_proto,
};
use rlmesh_proto::spaces::v1::SpaceSpec;

use crate::spaces::{env_contract_to_py, meta_map_to_pydict, space_value_to_py_neutral};

type RelayCall = Box<dyn for<'py> FnOnce(Python<'py>, &Bound<'py, PyAny>) -> PyResult<()> + Send>;

/// Delivers the runtime's per-episode events to a Python relay object as plain
/// data, one method call per event, each awaited inline by the driver (so the
/// env is idle while a hook runs) and run on the thread that called
/// `Model.run` (RUN-8). The first exception a call raises is kept with its
/// original type and cancels the run; every later event is dropped. The
/// relay's `step` may return "skip" (end the live episodes) or "quit" (stop
/// the run), which the driver applies at the next step.
pub(crate) struct PyRunHooks {
    relay: Arc<Py<PyAny>>,
    cancellation: CancellationToken,
    jobs: CallerJobs,
    error: Mutex<Option<PyErr>>,
    directive: Arc<Mutex<RunDirective>>,
}

impl PyRunHooks {
    pub(crate) fn new(relay: Py<PyAny>, cancellation: CancellationToken, jobs: CallerJobs) -> Self {
        Self {
            relay: Arc::new(relay),
            cancellation,
            jobs,
            error: Mutex::new(None),
            directive: Arc::default(),
        }
    }

    /// The hook exception that cancelled the run, if one did.
    pub(crate) fn take_error(&self) -> Option<PyErr> {
        self.error.lock().unwrap_or_else(|e| e.into_inner()).take()
    }

    async fn deliver(&self, call: RelayCall) {
        if self.cancellation.is_cancelled() {
            return;
        }
        let relay = Arc::clone(&self.relay);
        let result = self
            .jobs
            .run(move || Python::attach(|py| call(py, relay.bind(py))))
            .await
            .unwrap_or_else(|err| Err(PyRuntimeError::new_err(err.to_string())));
        if let Err(err) = result {
            self.error
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get_or_insert(err);
            self.cancellation.cancel();
        }
    }
}

/// One neutral Python value per lane from a group's wire leaves.
fn decode_lanes<'py>(
    py: Python<'py>,
    leaves: Vec<Bytes>,
    space: &SpaceSpec,
    lanes: usize,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let space =
        space_spec_from_proto(space.clone()).map_err(|e| PyValueError::new_err(e.to_string()))?;
    decode_batched_partial_values(Some(&leaves_value(leaves)), &space, lanes)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .iter()
        .map(|value| space_value_to_py_neutral(py, value, &space))
        .collect()
}

#[async_trait]
impl RuntimeHooks for PyRunHooks {
    async fn session_started(&self, event: SessionStartedEvent) -> Result<(), HookError> {
        self.deliver(Box::new(move |py, relay| {
            let contract = env_contract_from_proto(event.env_contract.as_ref().clone())
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let contract = env_contract_to_py(py, &contract)?;
            relay.call_method1("session_started", (contract,)).map(drop)
        }))
        .await;
        Ok(())
    }

    async fn episode_started(&self, event: EpisodeStartedEvent) -> Result<(), HookError> {
        self.deliver(Box::new(move |_py, relay| {
            relay
                .call_method1(
                    "episode_started",
                    (
                        event.episode_id,
                        event.episode_index,
                        event.env_index,
                        event.seed,
                        event.trial_index,
                    ),
                )
                .map(drop)
        }))
        .await;
        Ok(())
    }

    async fn observation_emitted(&self, event: ObservationEmittedEvent) -> Result<(), HookError> {
        let Some(leaves) = event.observation else {
            return Ok(());
        };
        let (space, ids, env_index) = (event.observation_space, event.episode_ids, event.env_index);
        self.deliver(Box::new(move |py, relay| {
            let values = decode_lanes(py, leaves, &space, ids.len())?;
            relay
                .call_method1("observation", (env_index, ids, values))
                .map(drop)
        }))
        .await;
        Ok(())
    }

    async fn action_received(&self, event: ActionReceivedEvent) -> Result<(), HookError> {
        let Some(leaves) = event.action else {
            return Ok(());
        };
        let (space, lanes, env_index) =
            (event.action_space, event.episode_ids.len(), event.env_index);
        self.deliver(Box::new(move |py, relay| {
            let values = decode_lanes(py, leaves, &space, lanes)?;
            relay.call_method1("action", (env_index, values)).map(drop)
        }))
        .await;
        Ok(())
    }

    async fn step_completed(&self, event: StepCompletedEvent) -> Result<(), HookError> {
        let info = event.infos.map(meta_map_from_proto);
        let directive = Arc::clone(&self.directive);
        self.deliver(Box::new(move |py, relay| {
            let info = info.map(|info| meta_map_to_pydict(py, &info)).transpose()?;
            let asked = relay.call_method1(
                "step",
                (
                    event.env_index,
                    event.rewards,
                    event.terminated,
                    event.truncated,
                    event.autoreset_roll,
                    info,
                    event.predict_ms,
                    event.step_ms,
                ),
            )?;
            let asked = match asked.extract::<Option<String>>()?.as_deref() {
                Some("skip") => RunDirective::EndEpisode,
                Some("quit") => RunDirective::Stop,
                _ => return Ok(()),
            };
            let mut directive = directive.lock().unwrap_or_else(|e| e.into_inner());
            if *directive != RunDirective::Stop {
                *directive = asked;
            }
            Ok(())
        }))
        .await;
        Ok(())
    }

    fn directive(&self) -> RunDirective {
        let mut directive = self.directive.lock().unwrap_or_else(|e| e.into_inner());
        match *directive {
            RunDirective::EndEpisode => std::mem::take(&mut *directive),
            held => held,
        }
    }

    async fn episode_completed(&self, event: EpisodeCompletedEvent) -> Result<(), HookError> {
        self.deliver(Box::new(move |_py, relay| {
            relay
                .call_method1(
                    "episode_completed",
                    (
                        event.episode_id,
                        event.episode_index,
                        event.env_index,
                        event.seed,
                        event.trial_index,
                        event.step_count,
                        event.cumulative_reward,
                        event.terminated,
                        event.truncated,
                        event.success,
                        event.duration_ms as f64 / 1000.0,
                        (event.predict_ms, event.step_ms),
                    ),
                )
                .map(drop)
        }))
        .await;
        Ok(())
    }
}

/// Delegates every event to `inner`, keeping each completed episode's
/// summary, so a run cut short by Ctrl-C still reports the episodes it
/// finished.
pub(crate) struct FinishedEpisodes {
    inner: Arc<dyn RuntimeHooks>,
    episodes: Mutex<Vec<EpisodeSummary>>,
}

impl FinishedEpisodes {
    pub(crate) fn new(inner: Arc<dyn RuntimeHooks>) -> Self {
        Self {
            inner,
            episodes: Mutex::default(),
        }
    }

    /// The summaries kept so far, in completion order.
    pub(crate) fn take(&self) -> Vec<EpisodeSummary> {
        std::mem::take(&mut self.episodes.lock().expect("episodes lock poisoned"))
    }
}

#[async_trait]
impl RuntimeHooks for FinishedEpisodes {
    async fn session_started(&self, event: SessionStartedEvent) -> Result<(), HookError> {
        self.inner.session_started(event).await
    }

    async fn episode_started(&self, event: EpisodeStartedEvent) -> Result<(), HookError> {
        self.inner.episode_started(event).await
    }

    async fn episode_completed(&self, event: EpisodeCompletedEvent) -> Result<(), HookError> {
        self.episodes
            .lock()
            .expect("episodes lock poisoned")
            .push(EpisodeSummary {
                episode_index: event.episode_index,
                env_index: event.env_index,
                seed: event.seed,
                trial_index: event.trial_index,
                step_count: event.step_count,
                cumulative_reward: event.cumulative_reward,
                terminated: event.terminated,
                truncated: event.truncated,
                duration_ms: event.duration_ms,
                success: event.success,
                predict_ms: event.predict_ms,
                step_ms: event.step_ms,
            });
        self.inner.episode_completed(event).await
    }

    async fn action_received(&self, event: ActionReceivedEvent) -> Result<(), HookError> {
        self.inner.action_received(event).await
    }

    async fn step_completed(&self, event: StepCompletedEvent) -> Result<(), HookError> {
        self.inner.step_completed(event).await
    }

    async fn observation_emitted(&self, event: ObservationEmittedEvent) -> Result<(), HookError> {
        self.inner.observation_emitted(event).await
    }

    async fn session_ended(&self, event: SessionEndedEvent) -> Result<(), HookError> {
        self.inner.session_ended(event).await
    }

    async fn on_telemetry(&self, event: TelemetrySnapshotEvent) -> Result<(), HookError> {
        self.inner.on_telemetry(event).await
    }

    async fn session_failed(&self, event: SessionFailedEvent) -> Result<(), HookError> {
        self.inner.session_failed(event).await
    }

    async fn log(&self, event: LogEvent) -> Result<(), HookError> {
        self.inner.log(event).await
    }

    fn directive(&self) -> RunDirective {
        self.inner.directive()
    }
}
