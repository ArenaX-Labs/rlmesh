//! Python presentation and signal conversion at the shared facade boundary.
use crate::model::LastPyError;
use crate::types::to_py_err;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rlmesh::binding::RunDirective;
use rlmesh::{EpisodeResult, RunContext, RunHooks, RunResult, StepEvent, Value};
use std::sync::Arc;

pub(crate) fn value_to_py<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    let value = rlmesh::binding::model::adapter_value(value).map_err(to_py_err)?;
    crate::adapters::encode_value(py, &value)
}

pub(crate) fn episode_to_py<'py>(
    py: Python<'py>,
    e: &EpisodeResult,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("index", e.index)?;
    out.set_item("seed", e.seed)?;
    out.set_item("trial", e.trial)?;
    out.set_item("steps", e.steps)?;
    out.set_item("reward", e.reward)?;
    out.set_item("terminated", e.terminated)?;
    out.set_item("truncated", e.truncated)?;
    out.set_item("success", e.success)?;
    out.set_item("duration_s", e.duration_s)?;
    out.set_item("predict_ms", e.predict_ms)?;
    out.set_item("step_ms", e.step_ms)?;
    Ok(out)
}

pub(crate) fn result_to_py<'py>(
    py: Python<'py>,
    result: &RunResult,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let episodes = pyo3::types::PyList::empty(py);
    for e in &result.episodes {
        episodes.append(episode_to_py(py, e)?)?;
    }
    out.set_item("episodes", episodes)?;
    let telemetry = pyo3::types::PyList::empty(py);
    for row in &result.telemetry {
        let item = PyDict::new(py);
        item.set_item("op", &row.op)?;
        item.set_item("component", &row.component)?;
        item.set_item("metric", &row.metric)?;
        item.set_item("unit", &row.unit)?;
        item.set_item("count", row.count)?;
        item.set_item("avg", row.avg)?;
        item.set_item("p50", row.p50)?;
        item.set_item("p95", row.p95)?;
        item.set_item("p99", row.p99)?;
        telemetry.append(item)?;
    }
    out.set_item("telemetry", telemetry)?;
    // Advisory conversion uses the binding's existing JSON/schema wrapper.
    let advisories = pyo3::types::PyList::empty(py);
    for advisory in &result.advisories {
        advisories.append(crate::adapters::PyAdvisory::from(
            rlmesh_adapters::v1::Advisory {
                severity: if advisory.severity.as_str() == "info" {
                    rlmesh_adapters::v1::AdvisorySeverity::Info
                } else {
                    rlmesh_adapters::v1::AdvisorySeverity::Caution
                },
                message: advisory.message.clone(),
            },
        ))?;
    }
    out.set_item("advisories", advisories)?;
    Ok(out)
}

#[pyclass(module = "rlmesh._rlmesh")]
pub(crate) struct PyRunContext {
    inner: Arc<dyn RunContext>,
}
#[pymethods]
impl PyRunContext {
    #[getter]
    fn env_contract(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.inner
            .contract()
            .map_err(to_py_err)?
            .map(|c| {
                let wire = rlmesh::binding::env_contract_to_wire(&c).map_err(to_py_err)?;
                crate::spaces::env_contract_to_py(py, &wire)
            })
            .transpose()
    }
}

#[derive(Clone)]
pub(crate) struct PyFacadeHooks {
    relay: Option<Arc<Py<PyAny>>>,
    error: LastPyError,
}
impl PyFacadeHooks {
    pub(crate) fn new(relay: Py<PyAny>, error: LastPyError) -> Self {
        Self {
            relay: Some(Arc::new(relay)),
            error,
        }
    }
    pub(crate) fn optional(relay: Option<Py<PyAny>>, error: LastPyError) -> Self {
        Self {
            relay: relay.map(Arc::new),
            error,
        }
    }
    fn call(
        &self,
        call: impl for<'py> FnOnce(Python<'py>, &Bound<'py, PyAny>) -> PyResult<()>,
    ) -> rlmesh::Result<()> {
        let Some(relay) = &self.relay else {
            return Ok(());
        };
        Python::attach(|py| call(py, relay.bind(py))).map_err(|err| {
            let native = rlmesh::Error::Internal(err.to_string());
            self.error
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get_or_insert(err);
            native
        })
    }
}
impl RunHooks for PyFacadeHooks {
    fn on_run_start(&mut self, context: Arc<dyn RunContext>) -> rlmesh::Result<()> {
        self.call(|py, relay| {
            relay
                .call_method1("start", (Py::new(py, PyRunContext { inner: context })?,))
                .map(drop)
        })
    }
    fn on_episode_start(&mut self, episode: i64, seed: Option<i64>) -> rlmesh::Result<()> {
        self.call(|_, relay| {
            relay
                .call_method1("episode_start", (episode, seed))
                .map(drop)
        })
    }
    fn on_step(&mut self, e: &StepEvent) -> rlmesh::Result<()> {
        self.call(|py, relay| {
            if !relay.call_method0("wants_step")?.extract::<bool>()? {
                return Ok(());
            }
            let data = PyDict::new(py);
            data.set_item("episode", e.episode)?;
            data.set_item("seed", e.seed)?;
            data.set_item("step", e.step)?;
            data.set_item("observation", value_to_py(py, &e.observation)?)?;
            data.set_item("action", value_to_py(py, &e.action)?)?;
            data.set_item("reward", e.reward)?;
            data.set_item("terminated", e.terminated)?;
            data.set_item("truncated", e.truncated)?;
            data.set_item("info", value_to_py(py, &Value::Dict(e.info.clone()))?)?;
            data.set_item("predict_ms", e.predict_ms)?;
            data.set_item("step_ms", e.step_ms)?;
            relay.call_method1("step", (data,)).map(drop)
        })
    }
    fn on_episode_end(&mut self, result: &EpisodeResult) -> rlmesh::Result<()> {
        self.call(|py, relay| {
            relay
                .call_method1("episode_end", (episode_to_py(py, result)?,))
                .map(drop)
        })
    }
    fn on_run_end(&mut self, result: &RunResult) -> rlmesh::Result<()> {
        self.call(|py, relay| {
            relay
                .call_method1("end", (result_to_py(py, result)?,))
                .map(drop)
        })
    }
}

impl rlmesh::binding::model::Presentation for PyFacadeHooks {
    fn on_frame(&mut self, frame: &rlmesh::binding::model::RunFrame<'_>) -> rlmesh::Result<()> {
        self.call(|py, relay| {
            if !relay.call_method0("wants_frame")?.extract::<bool>()? {
                return Ok(());
            }
            relay
                .call_method1(
                    "frame",
                    (
                        value_to_py(py, frame.observation)?,
                        frame.episode,
                        frame.seed,
                        frame.steps,
                        frame.reward,
                        frame.outcome,
                        frame.predict_ms,
                        frame.step_ms,
                        frame.elapsed_s,
                    ),
                )
                .map(drop)
        })
    }
    fn directive(&mut self) -> rlmesh::Result<RunDirective> {
        let mut asked = RunDirective::Continue;
        self.call(|_, relay| {
            asked = match relay
                .call_method0("directive")?
                .extract::<Option<String>>()?
                .as_deref()
            {
                Some("skip") => RunDirective::EndEpisode,
                Some("quit") => RunDirective::Stop,
                _ => RunDirective::Continue,
            };
            Ok(())
        })?;
        Ok(asked)
    }
}

pub(crate) struct PyEnvTarget {
    target: Py<PyAny>,
    owned: bool,
    error: LastPyError,
}
impl PyEnvTarget {
    pub(crate) fn new(py: Python<'_>, target: Py<PyAny>, error: LastPyError) -> PyResult<Self> {
        let owned = target.bind(py).getattr("owned")?.extract()?;
        Ok(Self {
            target,
            owned,
            error,
        })
    }
    fn failed(&self, err: PyErr) -> rlmesh::Error {
        let error = rlmesh::Error::Internal(err.to_string());
        self.error
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_or_insert(err);
        error
    }
}
impl rlmesh::binding::model::EnvAdapter for PyEnvTarget {
    fn owns_env(&self) -> bool {
        self.owned
    }
    fn open(
        &mut self,
        edition: Option<&str>,
        caller: rlmesh::binding::model::CallerJobs,
    ) -> rlmesh::Result<rlmesh::binding::model::AdaptedEnv> {
        crate::server::py_environment::with_facade_caller(caller, self.error.clone(), || {
            Python::attach(|py| {
                let out = self.target.bind(py).call_method1("open", (edition,))?;
                let (address, invite_at, close_over_wire) = out.extract()?;
                Ok(rlmesh::binding::model::AdaptedEnv {
                    address,
                    invite_at,
                    close_over_wire,
                })
            })
        })
        .map_err(|err| self.failed(err))
    }
    fn release(&mut self, close_env: bool) -> rlmesh::Result<()> {
        Python::attach(|py| {
            self.target
                .call_method1(py, "release", (close_env,))
                .map(drop)
        })
        .map_err(|err| self.failed(err))
    }
}

pub(crate) fn signal_poll(
    error: LastPyError,
    cancel: impl Fn() + Send + Sync + 'static,
) -> Arc<dyn Fn() + Send + Sync> {
    Arc::new(move || {
        if let Err(err) = crate::client::check_signals() {
            error
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get_or_insert(err);
            cancel();
        }
    })
}

pub(crate) struct AddressTarget {
    pub address: String,
    pub invite_at: Option<String>,
}
impl rlmesh::binding::model::EnvAdapter for AddressTarget {
    fn owns_env(&self) -> bool {
        false
    }
    fn open(
        &mut self,
        _: Option<&str>,
        _: rlmesh::binding::model::CallerJobs,
    ) -> rlmesh::Result<rlmesh::binding::model::AdaptedEnv> {
        Ok(rlmesh::binding::model::AdaptedEnv {
            address: self.address.clone(),
            invite_at: self.invite_at.clone(),
            close_over_wire: true,
        })
    }
    fn release(&mut self, _: bool) -> rlmesh::Result<()> {
        Ok(())
    }
}
