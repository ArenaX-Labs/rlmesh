use async_trait::async_trait;
use pyo3::prelude::*;
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::derive::{gen_methods_from_python, gen_stub_pyclass};
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::inventory::submit;
use rlmesh::{
    AdaptedModelHandler, BindAddress, ConnectAddress, EpisodeInfo, Error as RLMeshError,
    ModelObservation, ModelWorker, PredictFn, RemoteModel, RouteConfig, RouteResolver,
    RunLocalOptions, ServeModelOptions,
};
use rlmesh_adapters::v1::Value;
use rlmesh_spaces::{EnvContract, SpaceValue, spaces::SpaceSpec};
use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;

use crate::adapters::{PyCustomTransform, PyEncodings, decode_value, encode_value};
use crate::lifecycle::PyServeOptions;
use crate::spaces::{
    ValueBackend, batched_space_values_to_py_neutral, env_contract_to_py, extract_space_spec,
    make_space, py_any_to_batched_space_values_with_backend, py_any_to_meta_map,
    py_any_to_space_value_with_backend, space_value_to_py_neutral,
};
use crate::telemetry::{ProfileCollector, init_tracing};
use crate::types::to_py_err;

/// Process-wide multi-threaded runtime shared by Python model clients AND
/// served model workers. The Join response pump spawned during handshake lives
/// here, so it must outlive any single client; a process-wide runtime is
/// simplest and matches the env client. Workers share it too: a per-`PyModel`
/// runtime would cost a full worker-thread pool (plus its kqueue/pipe fds) per
/// constructed model, and Python's cycle collector cannot reclaim a pyclass
/// (no `__traverse__`), so those runtimes leaked for the process lifetime —
/// exhausting the default macOS 256-fd limit in model-heavy test suites.
fn model_runtime() -> &'static tokio::runtime::Runtime {
    static RUNTIME: std::sync::OnceLock<tokio::runtime::Runtime> = std::sync::OnceLock::new();
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("failed to build shared rlmesh model runtime")
    })
}

/// The model's predict hole, backed by the Python predict callable and its
/// discovered lifecycle callbacks. The engine ([`AdaptedModelHandler`]) calls
/// these back from a blocking worker thread; the framework-bridge (numpy/torch)
/// round-trip lives in the Python `predict_neutral` callable, so this layer only
/// converts between the adapter [`Value`] model and the neutral Python tree.
struct PyPredict {
    predict_fn: Py<PyAny>,
    /// Optional chunk corner (the model's `predict_chunk`): one assembled input ->
    /// a chunk of actions (leading axis = chunk). Absent for a single-action model;
    /// its presence is what [`has_chunk`](PredictFn::has_chunk) reports so the
    /// runtime can pin a execution horizon > 1.
    predict_chunk_fn: Option<Py<PyAny>>,
    /// Optional batched corner (`predict_batch`): a list of N assembled lane inputs
    /// -> N actions, in one call (one forward for the vector).
    predict_batch_fn: Option<Py<PyAny>>,
    /// Optional batched chunk corner (`predict_chunk_batch`): a list of N inputs ->
    /// N action chunks, in one call.
    predict_chunk_batch_fn: Option<Py<PyAny>>,
    /// The author's fusion permission (`Model.allow_fusion`, default true): the
    /// SDK's batched-corner glue (`tree_stack` over the batch axis) fuses
    /// independent lanes by construction, but the forward body it wraps is the
    /// author's own — one written against same-route, fixed-size batches (a
    /// shape-pinned jit trace, per-batch statistics) opts out here.
    allow_fusion: bool,
    on_episode_end: Option<Py<PyAny>>,
    on_close: Option<Py<PyAny>>,
}

impl PyPredict {
    fn fire(callback: &Option<Py<PyAny>>) -> rlmesh::Result<()> {
        let Some(callback) = callback else {
            return Ok(());
        };
        Python::attach(|py| callback.call0(py))
            .map(|_| ())
            .map_err(|err| RLMeshError::Internal(err.to_string()))
    }
}

impl PredictFn for PyPredict {
    fn predict(&self, model_input: Value, episode: Option<&EpisodeInfo>) -> rlmesh::Result<Value> {
        Python::attach(|py| -> PyResult<Value> {
            // The assembled input is now a Value tree (a nested dict/list/leaf
            // matching the model spec's InputNode shape).
            let input = encode_value(py, &model_input)?;
            let context = episode_context_dict(py, episode)?;
            let action = self.predict_fn.call1(py, (input, context))?;
            decode_value(action.bind(py))
        })
        .map_err(|err| RLMeshError::Internal(err.to_string()))
    }

    fn predict_chunk(
        &self,
        model_input: Value,
        horizon: u32,
        episode: Option<&EpisodeInfo>,
    ) -> rlmesh::Result<Option<Value>> {
        let Some(predict_chunk_fn) = self.predict_chunk_fn.as_ref() else {
            return Ok(None);
        };
        Python::attach(|py| -> PyResult<Value> {
            // The assembled input is a Value tree (dict/list/leaf matching the
            // model spec's InputNode shape), encoded as one Python argument.
            let input = encode_value(py, &model_input)?;
            let context = episode_context_dict(py, episode)?;
            // `predict_chunk(observation, horizon)`: the model returns up to
            // `horizon` actions; its chunk's leading axis is the chunk axis, which
            // the native engine's `split_chunk` unstacks into per-step frames.
            let chunk = predict_chunk_fn.call1(py, (input, horizon, context))?;
            decode_value(chunk.bind(py))
        })
        .map(Some)
        .map_err(|err| RLMeshError::Internal(err.to_string()))
    }

    fn has_chunk(&self) -> bool {
        self.predict_chunk_fn.is_some()
    }

    fn predict_batch(&self, inputs: Vec<Value>) -> rlmesh::Result<Vec<Value>> {
        match self.predict_batch_fn.as_ref() {
            Some(f) => call_batched(f, inputs, None),
            None => Err(RLMeshError::Internal(
                "predict_batch not implemented".to_string(),
            )),
        }
    }

    fn has_batch(&self) -> bool {
        self.predict_batch_fn.is_some()
    }

    fn predict_chunk_batch(&self, inputs: Vec<Value>, horizon: u32) -> rlmesh::Result<Vec<Value>> {
        match self.predict_chunk_batch_fn.as_ref() {
            Some(f) => call_batched(f, inputs, Some(horizon)),
            None => Err(RLMeshError::Internal(
                "predict_chunk_batch not implemented".to_string(),
            )),
        }
    }

    fn has_chunk_batch(&self) -> bool {
        self.predict_chunk_batch_fn.is_some()
    }

    fn allow_fusion(&self) -> bool {
        self.allow_fusion
            && (self.predict_batch_fn.is_some() || self.predict_chunk_batch_fn.is_some())
    }

    fn predict_spec_less(&self, observation: ModelObservation) -> rlmesh::Result<Vec<SpaceValue>> {
        // A spec-less route hands the raw observation straight to the model,
        // batched, preserving the pre-relocation path exactly (no adapter).
        let lanes = if observation.observation.is_some() {
            observation
                .decoded_lanes()
                .map_err(|err| RLMeshError::Internal(err.to_string()))?
        } else {
            Vec::new()
        };
        Python::attach(|py| -> PyResult<Vec<SpaceValue>> {
            let observation_space = observation
                .env_contract
                .as_ref()
                .and_then(|contract| contract.observation_space.as_ref());
            let obs = match (observation_space, lanes.len()) {
                (_, 0) => py.None().bind(py).clone(),
                (Some(space), 1) => space_value_to_py_neutral(py, &lanes[0], space)?,
                (Some(space), _) => batched_space_values_to_py_neutral(py, &lanes, space)?,
                (None, _) => {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "model worker requires observation space metadata",
                    ));
                }
            };
            // Real identity only for the single-episode case: a spec-less call
            // with more than one lane fuses N episodes into one forward pass,
            // same as the batched corners (see `PredictFn::predict_batch`), so
            // it carries the empty-identity context — the argument itself is
            // always delivered, matching every other path.
            let episode = (observation.num_envs == 1)
                .then(|| observation.route.episodes.first())
                .flatten();
            let context = episode_context_dict(py, episode)?;
            let action = self.predict_fn.call1(py, (obs, context))?;
            let action_space = observation
                .env_contract
                .as_ref()
                .and_then(|contract| contract.action_space.as_ref())
                .ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err(
                        "model worker requires action space metadata",
                    )
                })?;
            if observation.num_envs == 1 {
                Ok(vec![py_any_to_space_value_with_backend(
                    py,
                    action.bind(py),
                    action_space,
                    ValueBackend::Native,
                )?])
            } else {
                py_any_to_batched_space_values_with_backend(
                    py,
                    action.bind(py),
                    action_space,
                    observation.num_envs,
                    ValueBackend::Native,
                )
            }
        })
        .map_err(|err| RLMeshError::Internal(err.to_string()))
    }

    fn predict_spec_less_chunked(
        &self,
        observation: ModelObservation,
        execution_horizon: u32,
    ) -> rlmesh::Result<rlmesh::PredictFrames> {
        let horizon = execution_horizon.max(1) as usize;
        if horizon <= 1 || self.predict_chunk_fn.is_none() || observation.num_envs != 1 {
            return Ok(rlmesh::PredictFrames {
                actions: self.predict_spec_less(observation)?,
                replay: Vec::new(),
            });
        }
        let predict_chunk_fn = self
            .predict_chunk_fn
            .as_ref()
            .expect("checked predict_chunk_fn above");
        let lanes = observation
            .decoded_lanes()
            .map_err(|err| RLMeshError::Internal(err.to_string()))?;
        Python::attach(|py| -> PyResult<rlmesh::PredictFrames> {
            let contract = observation.env_contract.as_ref();
            let observation_space = contract
                .and_then(|contract| contract.observation_space.as_ref())
                .ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err(
                        "model worker requires observation space metadata",
                    )
                })?;
            let action_space = contract
                .and_then(|contract| contract.action_space.as_ref())
                .ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err(
                        "model worker requires action space metadata",
                    )
                })?;
            let lane = lanes.first().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err(
                    "spec-less chunked predict received no observation lane",
                )
            })?;
            let obs = space_value_to_py_neutral(py, lane, observation_space)?;
            // Guaranteed exactly one lane by the num_envs == 1 check above.
            let context = episode_context_dict(py, observation.route.episodes.first())?;
            let chunk = predict_chunk_fn.call1(py, (obs, horizon as u32, context))?;
            let chunk = chunk.bind(py);
            let frames_len = leading_axis_len(chunk).unwrap_or(horizon);
            let frames = py_any_to_batched_space_values_with_backend(
                py,
                chunk,
                action_space,
                frames_len,
                ValueBackend::Native,
            )?;
            let mut frames = frames.into_iter().take(horizon);
            let first = frames.next().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "predict_chunk returned an empty chunk; return at least one action",
                )
            })?;
            Ok(rlmesh::PredictFrames {
                actions: vec![first],
                replay: frames.map(|frame| vec![frame]).collect(),
            })
        })
        .map_err(|err| RLMeshError::Internal(err.to_string()))
    }

    fn on_episode_end(&self) -> rlmesh::Result<()> {
        Self::fire(&self.on_episode_end)
    }

    fn on_close(&self) -> rlmesh::Result<()> {
        Self::fire(&self.on_close)
    }
}

/// The leading-axis length of a neutral chunk value: a tensor-like exposes
/// `shape` (native `Tensor`, numpy array), a Dict chunk carries the axis
/// inside each leaf (probe the first value), a plain sequence its `len`.
/// `None` only for shapes this cannot introspect, where the caller falls back
/// to the pinned horizon.
fn leading_axis_len(value: &Bound<'_, PyAny>) -> Option<usize> {
    if let Ok(shape) = value.getattr("shape")
        && let Ok(first) = shape.get_item(0)
        && let Ok(len) = first.extract::<usize>()
    {
        return Some(len);
    }
    if let Ok(dict) = value.cast::<pyo3::types::PyDict>() {
        let (_, first) = dict.iter().next()?;
        return leading_axis_len(&first);
    }
    if value.is_instance_of::<pyo3::types::PyList>()
        || value.is_instance_of::<pyo3::types::PyTuple>()
    {
        return value.len().ok();
    }
    None
}

/// The predict context handed to the single-episode corners (`predict`,
/// `predict_chunk`, and the spec-less corners at `num_envs == 1`) as their
/// trailing positional argument: `{"episode_id": str, "episode_seed": int |
/// None}`. `predict_neutral` and friends (the Python SDK's native-worker glue)
/// only forward it to the author's own callback when that callback's
/// signature accepts a trailing context argument, so this is additive — an
/// author who never asked for it never sees it. Never built for the batched
/// corners (`predict_batch`/`predict_chunk_batch`, see [`call_batched`]) or a
/// multi-lane spec-less call.
fn episode_context_dict<'py>(
    py: Python<'py>,
    episode: Option<&EpisodeInfo>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item(
        "episode_id",
        episode
            .map(|episode| episode.episode_id.as_str())
            .unwrap_or(""),
    )?;
    dict.set_item("episode_seed", episode.and_then(|episode| episode.seed))?;
    Ok(dict)
}

/// Call a Python batched corner (`predict_batch` / `predict_chunk_batch`) with N
/// assembled lane inputs and decode its N returned actions/chunks. The Python side
/// receives a list of N neutral input dicts and returns a sequence of N actions
/// (one per lane, in order); the model owns how it batches the forward pass. When
/// `horizon` is `Some(h)` (the chunk-batch corner) it is passed as a second
/// argument, so the model can size each chunk to the execution horizon. No
/// episode context: N lanes fused into one forward pass (possibly from
/// independent episodes) is the opposite of the single-episode case episode
/// context is for.
fn call_batched(
    fn_obj: &Py<PyAny>,
    inputs: Vec<Value>,
    horizon: Option<u32>,
) -> rlmesh::Result<Vec<Value>> {
    Python::attach(|py| -> PyResult<Vec<Value>> {
        let list = pyo3::types::PyList::empty(py);
        for model_input in &inputs {
            // Each lane's assembled input is a Value tree, encoded as one element.
            list.append(encode_value(py, model_input)?)?;
        }
        let result = match horizon {
            Some(h) => fn_obj.call1(py, (list, h))?,
            None => fn_obj.call1(py, (list,))?,
        };
        let mut out = Vec::with_capacity(inputs.len());
        for item in result.bind(py).try_iter()? {
            out.push(decode_value(&item?)?);
        }
        Ok(out)
    })
    .map_err(|err| RLMeshError::Internal(err.to_string()))
}

/// The in-process run's report as a Python dict: `episodes` (the per-episode
/// summaries) and `telemetry` (the session metric aggregate).
fn report_to_py(py: Python<'_>, report: &rlmesh::RuntimeReport) -> PyResult<Py<PyAny>> {
    let out = pyo3::types::PyDict::new(py);
    out.set_item("episodes", report_episodes_to_py(py, report)?)?;
    out.set_item("telemetry", report_telemetry_to_py(py, report)?)?;
    Ok(out.into_any().unbind())
}

/// The session telemetry aggregate as a `list[dict]`: one row per
/// (op, component, metric) with count/avg/p50/p95/p99 in the metric's unit
/// (`ms` for durations), keys matching the SDK's `TelemetryRow` fields.
fn report_telemetry_to_py(py: Python<'_>, report: &rlmesh::RuntimeReport) -> PyResult<Py<PyAny>> {
    use rlmesh::telemetry::Kind;
    let rows = pyo3::types::PyList::empty(py);
    for row in &report.telemetry.rows {
        let entry = pyo3::types::PyDict::new(py);
        entry.set_item("op", row.source.op)?;
        entry.set_item("component", row.source.component)?;
        entry.set_item("metric", row.metric.name)?;
        entry.set_item(
            "unit",
            match row.metric.kind {
                Kind::Duration => "ms",
                Kind::Bytes => "bytes",
                Kind::Count => "count",
            },
        )?;
        entry.set_item("count", row.count)?;
        entry.set_item("avg", row.avg)?;
        entry.set_item("p50", row.p50)?;
        entry.set_item("p95", row.p95)?;
        entry.set_item("p99", row.p99)?;
        rows.append(entry)?;
    }
    Ok(rows.into_any().unbind())
}

/// The in-process run's per-episode results as a Python `list[dict]` (keys
/// matching the SDK's `EpisodeResult` fields), from the runtime report's
/// episode summaries. `predict_ms`/`step_ms` carry the report's SESSION-mean
/// op latencies (the runtime aggregates timing per op, not per episode), so
/// every episode reports the same mean -- real numbers rather than silent
/// zeros, with the per-episode distinction documented on `Model.run`.
fn report_episodes_to_py(py: Python<'_>, report: &rlmesh::RuntimeReport) -> PyResult<Py<PyAny>> {
    let predict_ms = session_mean_ms(report, "model.predict");
    let step_ms = session_mean_ms(report, "env.step");
    let episodes = pyo3::types::PyList::empty(py);
    for episode in &report.episodes {
        let entry = pyo3::types::PyDict::new(py);
        entry.set_item("index", episode.episode_index)?;
        entry.set_item("env_index", episode.env_index)?;
        entry.set_item("seed", episode.seed)?;
        entry.set_item("steps", episode.step_count)?;
        entry.set_item("reward", episode.cumulative_reward)?;
        entry.set_item("terminated", episode.terminated)?;
        entry.set_item("truncated", episode.truncated)?;
        entry.set_item("success", episode.success)?;
        entry.set_item("duration_s", episode.duration_ms as f64 / 1000.0)?;
        entry.set_item("predict_ms", predict_ms)?;
        entry.set_item("step_ms", step_ms)?;
        episodes.append(entry)?;
    }
    Ok(episodes.into_any().unbind())
}

/// The session-mean `rpc.total` latency (ms) for `op` from the report's
/// telemetry aggregate, `0.0` when the op never ran.
fn session_mean_ms(report: &rlmesh::RuntimeReport, op: &str) -> f64 {
    report
        .telemetry
        .rows
        .iter()
        .find(|row| row.source.op == op && row.metric.name == "rpc.total")
        .map(|row| row.avg)
        .unwrap_or(0.0)
}

/// Run the in-process worker loop with Python signal delivery: the run future
/// races a poll interval that re-attaches only to check for pending signals
/// (Ctrl-C), cancelling the runtime session and re-raising the signal's
/// exception -- the same pattern the remote-env client uses (`run_rpc`), so a
/// long eval aborts between operations instead of blocking SIGINT until the
/// final episode.
fn run_local_blocking(
    py: Python<'_>,
    handler: AdaptedModelHandler,
    options: RunLocalOptions,
) -> PyResult<rlmesh::RuntimeReport> {
    enum RunOutcome {
        Done(rlmesh::Result<rlmesh::RuntimeReport>),
        Signal(PyErr),
    }
    let cancellation = rlmesh::CancellationToken::new();
    let run_token = cancellation.clone();
    let outcome = py.detach(|| {
        model_runtime().block_on(async move {
            let run = ModelWorker::new(handler).run_local_cancellable_async(options, run_token);
            let mut run = std::pin::pin!(run);
            let mut poll = tokio::time::interval(crate::client::SIGNAL_POLL_INTERVAL);
            poll.tick().await;
            loop {
                tokio::select! {
                    result = &mut run => return RunOutcome::Done(result),
                    _ = poll.tick() => {
                        if let Err(err) = Python::attach(|py| py.check_signals()) {
                            cancellation.cancel();
                            let _ = (&mut run).await;
                            return RunOutcome::Signal(err);
                        }
                    }
                }
            }
        })
    });
    match outcome {
        RunOutcome::Done(result) => result.map_err(to_py_err),
        RunOutcome::Signal(err) => Err(err),
    }
}

/// The per-route resolver the served model exposes via `route_setup`. Runs the
/// Python `configure_fn` (which resolves the spec into a native plan + neutral
/// host holes) off the predict lock and hands the engine a [`RouteConfig`];
/// `None` is a spec-less / `NO_ADAPTER` route.
struct PyRouteResolver {
    configure_fn: Py<PyAny>,
}

#[async_trait]
impl RouteResolver for PyRouteResolver {
    async fn resolve(
        &self,
        _route_key: &str,
        env_contract: &EnvContract,
    ) -> rlmesh::Result<Option<RouteConfig>> {
        let configure_fn = Python::attach(|py| self.configure_fn.clone_ref(py));
        let contract = env_contract.clone();
        let observation_space = contract.observation_space.clone();
        let action_space = contract.action_space.clone();
        tokio::task::spawn_blocking(move || {
            Python::attach(|py| -> PyResult<Option<RouteConfig>> {
                let contract_py = env_contract_to_py(py, &contract)?;
                let resolved = configure_fn.call1(py, (contract_py,))?;
                let resolved = resolved.bind(py);
                if resolved.is_none() {
                    return Ok(None);
                }
                let served = resolved.cast::<pyo3::types::PyDict>()?;
                let plan_obj = served.get_item("plan")?.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("served route is missing 'plan'")
                })?;
                let plan = plan_obj
                    .cast::<crate::adapters::PyAdapterPlan>()
                    .map_err(|_| {
                        pyo3::exceptions::PyTypeError::new_err(
                            "served route 'plan' is not an AdapterPlan",
                        )
                    })?;
                let adapter = plan.borrow().adapter().clone();

                let mut customs: HashMap<String, Py<PyAny>> = HashMap::new();
                if let Some(customs_obj) = served.get_item("customs")? {
                    for (key, value) in customs_obj.cast::<pyo3::types::PyDict>()?.iter() {
                        customs.insert(key.extract()?, value.unbind());
                    }
                }
                let obs_encodings = optional_callable(served.get_item("obs_encodings")?);
                let action_encodings = optional_callable(served.get_item("action_encodings")?);

                let observation_space = observation_space.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "env contract missing observation_space",
                    )
                })?;
                let action_space = action_space.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("env contract missing action_space")
                })?;

                Ok(Some(RouteConfig::new(
                    adapter,
                    observation_space,
                    action_space,
                    Box::new(PyCustomTransform::new(customs)),
                    Box::new(PyEncodings::new(obs_encodings, action_encodings)),
                )))
            })
        })
        .await
        .map_err(|err| RLMeshError::Internal(format!("configure task panicked: {err}")))?
        .map_err(|err| RLMeshError::Internal(err.to_string()))
    }
}

/// A present, non-`None` Python value as an owned callable handle.
fn optional_callable(value: Option<Bound<'_, PyAny>>) -> Option<Py<PyAny>> {
    match value {
        Some(value) if !value.is_none() => Some(value.unbind()),
        _ => None,
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(module = "rlmesh._rlmesh")]
pub struct PyModel {
    predict_fn: Py<PyAny>,
    predict_chunk_fn: Option<Py<PyAny>>,
    predict_batch_fn: Option<Py<PyAny>>,
    predict_chunk_batch_fn: Option<Py<PyAny>>,
    allow_fusion: bool,
    configure_fn: Option<Py<PyAny>>,
    on_episode_end: Option<Py<PyAny>>,
    on_close: Option<Py<PyAny>>,
    profiler: Arc<ProfileCollector>,
}

impl PyModel {
    /// Build the served engine handler: the predict hole plus, for a spec'd
    /// model, the route resolver. The engine owns the per-route adapter state and
    /// frame buffers; this layer only supplies the host (Python) callbacks.
    fn build_handler(&self) -> AdaptedModelHandler {
        let predict = Python::attach(|py| PyPredict {
            predict_fn: self.predict_fn.clone_ref(py),
            predict_chunk_fn: self.predict_chunk_fn.as_ref().map(|cb| cb.clone_ref(py)),
            predict_batch_fn: self.predict_batch_fn.as_ref().map(|cb| cb.clone_ref(py)),
            predict_chunk_batch_fn: self
                .predict_chunk_batch_fn
                .as_ref()
                .map(|cb| cb.clone_ref(py)),
            allow_fusion: self.allow_fusion,
            on_episode_end: self.on_episode_end.as_ref().map(|cb| cb.clone_ref(py)),
            on_close: self.on_close.as_ref().map(|cb| cb.clone_ref(py)),
        });
        let resolver: Option<Arc<dyn RouteResolver>> =
            self.configure_fn.as_ref().map(|configure| {
                let configure_fn = Python::attach(|py| configure.clone_ref(py));
                Arc::new(PyRouteResolver { configure_fn }) as Arc<dyn RouteResolver>
            });
        AdaptedModelHandler::new(Arc::new(predict), resolver)
    }
}

#[pymethods]
impl PyModel {
    #[new]
    #[pyo3(signature = (predict_fn, configure_fn=None, on_episode_end=None, on_close=None, predict_chunk_fn=None, predict_batch_fn=None, predict_chunk_batch_fn=None, allow_fusion=true))]
    #[allow(clippy::too_many_arguments)] // a PyO3 #[new] ctor maps each arg to a Python kwarg
    fn new(
        predict_fn: Py<PyAny>,
        configure_fn: Option<Py<PyAny>>,
        on_episode_end: Option<Py<PyAny>>,
        on_close: Option<Py<PyAny>>,
        predict_chunk_fn: Option<Py<PyAny>>,
        predict_batch_fn: Option<Py<PyAny>>,
        predict_chunk_batch_fn: Option<Py<PyAny>>,
        allow_fusion: bool,
    ) -> PyResult<Self> {
        init_tracing("model_worker");
        let profiler = ProfileCollector::new("model_worker");

        Ok(Self {
            predict_fn,
            predict_chunk_fn,
            predict_batch_fn,
            predict_chunk_batch_fn,
            allow_fusion,
            configure_fn,
            on_episode_end,
            on_close,
            profiler,
        })
    }

    #[pyo3(signature = (env_address, execution_horizon=1))]
    fn run_local(
        &self,
        py: Python<'_>,
        env_address: &str,
        execution_horizon: u32,
    ) -> PyResult<Py<PyAny>> {
        let run_span = tracing::info_span!("rlmesh.model.run_local", env_address = env_address);
        let _run_enter = run_span.enter();
        let total_guard = self.profiler.start("model.run_local.total");

        let env_address = ConnectAddress::parse(env_address).map_err(to_py_err)?;
        let handler = self.build_handler();
        let options = RunLocalOptions::new(env_address).execution_horizon(execution_horizon);

        let report = run_local_blocking(py, handler, options)?;

        let _ = total_guard.finish(0);
        self.profiler.log_summary_once();
        report_to_py(py, &report)
    }

    #[pyo3(signature = (env_address, max_episodes, execution_horizon=1, seeds=None, max_episode_steps=None, max_episode_seconds=None, close_env=false))]
    #[allow(clippy::too_many_arguments)]
    fn run_local_for_episodes(
        &self,
        py: Python<'_>,
        env_address: &str,
        max_episodes: u64,
        execution_horizon: u32,
        seeds: Option<Vec<i64>>,
        max_episode_steps: Option<i64>,
        max_episode_seconds: Option<f64>,
        close_env: bool,
    ) -> PyResult<Py<PyAny>> {
        let run_span = tracing::info_span!(
            "rlmesh.model.run_local_for_episodes",
            env_address = env_address,
            max_episodes
        );
        let _run_enter = run_span.enter();
        let total_guard = self.profiler.start("model.run_local.total");

        let env_address = ConnectAddress::parse(env_address).map_err(to_py_err)?;
        let handler = self.build_handler();
        let mut options = RunLocalOptions::new(env_address)
            .for_episodes(max_episodes)
            .execution_horizon(execution_horizon)
            .episode_seeds(seeds.unwrap_or_default())
            .close_env(close_env);
        if let Some(cap) = max_episode_steps {
            options = options.max_episode_steps(cap);
        }
        if let Some(cap) = max_episode_seconds {
            options = options.max_episode_seconds(cap);
        }

        let report = run_local_blocking(py, handler, options)?;

        let _ = total_guard.finish(0);
        self.profiler.log_summary_once();
        report_to_py(py, &report)
    }

    #[pyo3(signature = (address, options=None))]
    fn serve(
        &self,
        py: Python<'_>,
        address: &str,
        options: Option<PyServeOptions>,
    ) -> PyResult<()> {
        let run_span = tracing::info_span!("rlmesh.model.serve", address = address);
        let _run_enter = run_span.enter();
        let total_guard = self.profiler.start("model.serve.total");

        let address = BindAddress::parse(address).map_err(to_py_err)?;
        let options = options.map(PyServeOptions::into_rust).unwrap_or_default();
        let handler = self.build_handler();

        py.detach(|| {
            model_runtime().block_on(async move {
                ModelWorker::new(handler)
                    .serve_async(ServeModelOptions::new(address).serve_options(options))
                    .await
            })
        })
        .map_err(to_py_err)?;

        let _ = total_guard.finish(0);
        self.profiler.log_summary_once();
        Ok(())
    }
}

#[cfg(feature = "stub-gen")]
submit! {
    gen_methods_from_python! {
        r#"
import collections.abc
import typing

class PyModel:
    def __init__(self, predict_fn: collections.abc.Callable[[Value], Value], configure_fn: collections.abc.Callable[[EnvContract], object] | None = None, on_episode_end: collections.abc.Callable[[], None] | None = None, on_close: collections.abc.Callable[[], None] | None = None, predict_chunk_fn: collections.abc.Callable[[Value, int], Value] | None = None, predict_batch_fn: collections.abc.Callable[[list[Value]], list[Value]] | None = None, predict_chunk_batch_fn: collections.abc.Callable[[list[Value], int], list[Value]] | None = None, allow_fusion: bool = True) -> None: ...
    def run_local(self, env_address: str, execution_horizon: int = 1) -> dict[str, typing.Any]: ...
    def run_local_for_episodes(self, env_address: str, max_episodes: int, execution_horizon: int = 1, seeds: list[int] | None = None, max_episode_steps: int | None = None, max_episode_seconds: float | None = None, close_env: bool = False) -> dict[str, typing.Any]: ...
    def serve(self, address: str, options: ServeOptions | None = None) -> None: ...
"#
    }
}

#[cfg(feature = "stub-gen")]
submit! {
    gen_methods_from_python! {
        r#"
class PyModelClient:
    def __init__(self, address: str, env_contract: EnvContract, execution_horizon: int = 1, *, connect_timeout_seconds: float | None = None, request_timeout_seconds: float | None = None) -> None: ...
    def address(self) -> str: ...
    def env_id(self) -> str: ...
    def observation_space(self) -> Space: ...
    def action_space(self) -> Space: ...
    def reset(self, seed: int | None = None) -> None: ...
    def predict(self, observation: Value) -> Value: ...
    def close(self) -> None: ...
"#
    }
}

impl Drop for PyModel {
    fn drop(&mut self) {
        self.profiler.log_summary_once();
    }
}

/// Client handle to a model (policy) server: the model-side mirror of
/// `PyEnvClient`. Bound to one env contract (one route) for its lifetime; the
/// Python layer creates one per `rlmesh.session(model, env)`.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(module = "rlmesh._rlmesh")]
pub struct PyModelClient {
    inner: RemoteModel,
    runtime: &'static tokio::runtime::Runtime,
    observation_space: SpaceSpec,
    action_space: SpaceSpec,
    address: String,
    default_timeout: Option<std::time::Duration>,
}

#[pymethods]
impl PyModelClient {
    #[new]
    #[pyo3(signature = (address, env_contract, execution_horizon=1, *, connect_timeout_seconds=None, request_timeout_seconds=None))]
    fn new(
        py: Python<'_>,
        address: &str,
        env_contract: &Bound<'_, PyAny>,
        execution_horizon: u32,
        connect_timeout_seconds: Option<f64>,
        request_timeout_seconds: Option<f64>,
    ) -> PyResult<Self> {
        init_tracing("model_client");
        let contract = native_env_contract_from_py(env_contract)?;
        let observation_space = contract.observation_space.clone().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("env contract missing observation_space")
        })?;
        let action_space = contract.action_space.clone().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("env contract missing action_space")
        })?;
        let connect_timeout =
            crate::client::optional_timeout(connect_timeout_seconds, "connect_timeout_seconds")?;
        let default_timeout =
            crate::client::optional_timeout(request_timeout_seconds, "request_timeout_seconds")?;
        let runtime = model_runtime();
        let address = address.to_string();
        let mut inner = py.detach(|| {
            let connect = RemoteModel::connect(&address, contract);
            match connect_timeout {
                Some(timeout) => {
                    match runtime.block_on(async { tokio::time::timeout(timeout, connect).await }) {
                        Ok(result) => result.map_err(to_py_err),
                        Err(_) => Err(pyo3::exceptions::PyTimeoutError::new_err(format!(
                            "remote model connect timed out after {:.3}s",
                            timeout.as_secs_f64()
                        ))),
                    }
                }
                None => runtime.block_on(connect).map_err(to_py_err),
            }
        })?;
        // Opt the served model into chunking (h > 1): pinned on ConfigureRoute and
        // replayed open-loop by RemoteModel. 1 = no chunking.
        inner.set_execution_horizon(execution_horizon);
        let address = inner.address().to_string();
        Ok(Self {
            inner,
            runtime,
            observation_space,
            action_space,
            address,
            default_timeout,
        })
    }

    fn address(&self) -> String {
        self.address.clone()
    }

    /// The env (adapter) routing key this client uses with the model — a UUIDv7
    /// minted at connect.
    fn env_id(&self) -> String {
        self.inner.env_id().to_string()
    }

    fn observation_space(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(make_space(py, &self.observation_space)?.into_any().unbind())
    }

    fn action_space(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(make_space(py, &self.action_space)?.into_any().unbind())
    }

    /// Begin a new episode (next predict marks a reset boundary). `seed` (the
    /// explicit reset seed, if any) rides on every predict of the episode as
    /// the served model's `context["episode_seed"]`.
    #[pyo3(signature = (seed=None))]
    fn reset(&mut self, seed: Option<i64>) {
        self.inner.reset(seed);
    }

    fn predict(&mut self, py: Python<'_>, observation: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let value = py_any_to_space_value_with_backend(
            py,
            observation.bind(py),
            &self.observation_space,
            ValueBackend::Native,
        )?;
        let runtime = self.runtime;
        let timeout = self.default_timeout;
        let inner = &mut self.inner;
        let action = py.detach(|| block_on_with_timeout(runtime, timeout, inner.predict(value)))?;
        Ok(space_value_to_py_neutral(py, &action, &self.action_space)?.unbind())
    }

    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        let runtime = self.runtime;
        let timeout = self.default_timeout;
        let inner = &mut self.inner;
        py.detach(|| block_on_with_timeout(runtime, timeout, inner.close()))
    }
}

/// Run one model RPC on `runtime` with the GIL released, honoring the client's
/// optional request timeout; `None` blocks indefinitely (the pre-timeout
/// behavior).
fn block_on_with_timeout<T>(
    runtime: &tokio::runtime::Runtime,
    timeout: Option<std::time::Duration>,
    rpc: impl Future<Output = rlmesh::Result<T>>,
) -> PyResult<T> {
    match timeout {
        Some(timeout) => {
            match runtime.block_on(async { tokio::time::timeout(timeout, rpc).await }) {
                Ok(result) => result.map_err(to_py_err),
                Err(_) => Err(pyo3::exceptions::PyTimeoutError::new_err(format!(
                    "remote model request timed out after {:.3}s",
                    timeout.as_secs_f64()
                ))),
            }
        }
        None => runtime.block_on(rpc).map_err(to_py_err),
    }
}

/// Reconstruct a native `EnvContract` from a Python contract object (the value
/// `RemoteEnv.env_contract` returns). Duck-typed via getattr because the pyclass
/// `PyEnvContract` is `skip_from_py_object` and its native `inner` cannot be
/// extracted directly. Carries `metadata` (the env's adapter tags) so the
/// served model can resolve its adapter from the route's contract.
fn native_env_contract_from_py(contract: &Bound<'_, PyAny>) -> PyResult<EnvContract> {
    let id: String = contract.getattr("id")?.extract()?;
    let num_envs: u32 = contract.getattr("num_envs")?.extract()?;
    let render_mode: String = contract
        .getattr("render_mode")?
        .extract::<Option<String>>()?
        .unwrap_or_default();
    let observation_space = extract_space_spec(&contract.getattr("observation_space")?)
        .ok_or_else(|| {
            pyo3::exceptions::PyTypeError::new_err(
                "env contract observation_space is not an RLMesh space spec",
            )
        })?;
    let action_space = extract_space_spec(&contract.getattr("action_space")?).ok_or_else(|| {
        pyo3::exceptions::PyTypeError::new_err(
            "env contract action_space is not an RLMesh space spec",
        )
    })?;
    let metadata_obj = contract.getattr("metadata")?;
    let metadata = if metadata_obj.is_none() {
        None
    } else {
        Some(py_any_to_meta_map(&metadata_obj)?)
    };
    Ok(EnvContract {
        id,
        action_space: Some(action_space),
        observation_space: Some(observation_space),
        metadata,
        render_mode,
        num_envs,
        // The driving loop is user-owned and single-env; autoreset mode does not
        // affect the client route, so the contract default is fine here.
        autoreset_mode: Default::default(),
    })
}
