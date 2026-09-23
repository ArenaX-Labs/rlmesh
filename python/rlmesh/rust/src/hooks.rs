//! The native run's observer seam: runtime events delivered to one Python relay.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rlmesh::{
    ActionReceivedEvent, CancellationToken, EpisodeCompletedEvent, EpisodeStartedEvent, HookError,
    ObservationEmittedEvent, RuntimeHooks, StepCompletedEvent,
};
use rlmesh_grpc::wire::{
    Bytes, decode_batched_partial_values, leaves_value, meta_map_from_proto, space_spec_from_proto,
};
use rlmesh_proto::spaces::v1::SpaceSpec;

use crate::spaces::{meta_map_to_pydict, space_value_to_py_neutral};

type RelayCall = Box<dyn for<'py> FnOnce(Python<'py>, &Bound<'py, PyAny>) -> PyResult<()> + Send>;

/// Delivers the runtime's per-episode events to a Python relay object as plain
/// data, one method call per event, each awaited inline by the driver (so the
/// env is idle while a hook runs). The first exception a call raises is kept
/// with its original type and cancels the run; every later event is dropped.
pub(crate) struct PyRunHooks {
    relay: Arc<Py<PyAny>>,
    cancellation: CancellationToken,
    error: Mutex<Option<PyErr>>,
}

impl PyRunHooks {
    pub(crate) fn new(relay: Py<PyAny>, cancellation: CancellationToken) -> Self {
        Self {
            relay: Arc::new(relay),
            cancellation,
            error: Mutex::new(None),
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
        let result =
            tokio::task::spawn_blocking(move || Python::attach(|py| call(py, relay.bind(py))))
                .await
                .unwrap_or_else(|err| {
                    Err(PyRuntimeError::new_err(format!(
                        "run hook task panicked: {err}"
                    )))
                });
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
        self.deliver(Box::new(move |py, relay| {
            let info = info.map(|info| meta_map_to_pydict(py, &info)).transpose()?;
            relay
                .call_method1(
                    "step",
                    (
                        event.env_index,
                        event.rewards,
                        event.terminated,
                        event.truncated,
                        event.autoreset_roll,
                        info,
                    ),
                )
                .map(drop)
        }))
        .await;
        Ok(())
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
                    ),
                )
                .map(drop)
        }))
        .await;
        Ok(())
    }
}
