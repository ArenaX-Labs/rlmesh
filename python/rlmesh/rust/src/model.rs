use async_trait::async_trait;
use pyo3::prelude::*;
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::derive::{gen_methods_from_python, gen_stub_pyclass, gen_stub_pyfunction};
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::inventory::submit;
use rlmesh::Error as RLMeshError;
use rlmesh::binding::model::{
    AdaptedModelHandler, EpisodeInfo, ModelHandler, ModelObservation, PredictFn, PredictFrames,
    RouteConfig, RouteResolver,
};
use rlmesh_adapters::v1::Value;
use rlmesh_spaces::{EnvContract, SpaceValue};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use crate::adapters::{PyCustomTransform, PyEncodings, decode_value, encode_value};
use crate::lifecycle::PyServeOptions;
use crate::spaces::{
    ValueBackend, batched_space_values_to_py_neutral, env_contract_to_py,
    py_any_to_batched_space_values_with_backend, py_any_to_meta_map,
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
pub(crate) fn model_runtime() -> &'static tokio::runtime::Runtime {
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
    /// The author's declared native chunk K (`Model.native_chunk`), read off the
    /// loaded policy when the worker was built. `None` = undeclared (elastic).
    native_chunk: Option<u32>,
    on_episode_end: Option<Py<PyAny>>,
    on_close: Option<Py<PyAny>>,
    /// Drops an env's per-episode entries (an evict-all `ResetAdapter` or a
    /// `ReleaseAdapter`), firing no episode-end hook.
    on_route_drop: Option<Py<PyAny>>,
    /// The episodes a spec-less route already warned a short chunk for.
    short_warned: std::sync::Mutex<std::collections::HashSet<String>>,
    /// The last exception a host callback raised, kept so the in-process run
    /// re-raises the user's own exception instead of its stringified copy.
    last_error: LastPyError,
    /// The model's `reset_options(episode)`, when it defines one.
    reset_options_fn: Option<Py<PyAny>>,
    /// The models this model calls from its predict (`self.inner.predict`).
    nested: Option<Arc<crate::nested::Nested>>,
}

pub(crate) type LastPyError = Arc<std::sync::Mutex<Option<PyErr>>>;

/// A facade call's parsed run options, and what a route adds to them.
struct ParsedRunOptions {
    options: rlmesh::RunOptions<'static>,
    default_horizon: Option<u32>,
    base_seed: Option<i64>,
    seed_key: Option<String>,
}

impl PyPredict {
    /// A raised `ModelException` becomes a model failure carrying its
    /// recoverable flag (code `INTERNAL`); any other exception stays internal.
    fn internal(&self, err: PyErr) -> RLMeshError {
        let error = crate::types::model_callback_error(&err);
        *self.last_error.lock().expect("last_error lock poisoned") = Some(err);
        error
    }

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
        rlmesh::binding::check_model_call()?;
        Python::attach(|py| -> PyResult<Value> {
            // The assembled input is now a Value tree (a nested dict/list/leaf
            // matching the model spec's InputNode shape).
            let input = encode_value(py, &model_input)?;
            let context = episode_context_dict(py, episode)?;
            let action = self.predict_fn.call1(py, (input, context))?;
            decode_value(action.bind(py))
        })
        .map_err(|err| self.internal(err))
    }

    fn predict_chunk(
        &self,
        model_input: Value,
        horizon: u32,
        episode: Option<&EpisodeInfo>,
    ) -> rlmesh::Result<Option<Value>> {
        rlmesh::binding::check_model_call()?;
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
        .map_err(|err| self.internal(err))
    }

    fn has_chunk(&self) -> bool {
        self.predict_chunk_fn.is_some()
    }

    fn predict_batch(
        &self,
        inputs: Vec<Value>,
        episodes: &[EpisodeInfo],
    ) -> rlmesh::Result<Vec<Value>> {
        rlmesh::binding::check_model_call()?;
        match self.predict_batch_fn.as_ref() {
            Some(f) => call_batched(f, inputs, None, episodes).map_err(|err| self.internal(err)),
            None => Err(RLMeshError::Internal(
                "predict_batch not implemented".to_string(),
            )),
        }
    }

    fn has_batch(&self) -> bool {
        self.predict_batch_fn.is_some()
    }

    fn predict_chunk_batch(
        &self,
        inputs: Vec<Value>,
        horizon: u32,
        episodes: &[EpisodeInfo],
    ) -> rlmesh::Result<Vec<Value>> {
        rlmesh::binding::check_model_call()?;
        match self.predict_chunk_batch_fn.as_ref() {
            Some(f) => {
                call_batched(f, inputs, Some(horizon), episodes).map_err(|err| self.internal(err))
            }
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

    fn native_chunk(&self) -> Option<u32> {
        self.native_chunk
    }

    fn predict_spec_less(&self, observation: ModelObservation) -> rlmesh::Result<Vec<SpaceValue>> {
        rlmesh::binding::check_model_call()?;
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
            if lanes.len() > 1
                && let (Some(batch_fn), Some(space)) =
                    (self.predict_batch_fn.as_ref(), observation_space)
            {
                // The batched corner: per-lane inputs the SDK glue stacks into
                // one forward and splits back, the same contract as an adapted
                // route (see `call_batched`).
                let action_space = observation
                    .env_contract
                    .as_ref()
                    .and_then(|contract| contract.action_space.as_ref())
                    .ok_or_else(|| {
                        pyo3::exceptions::PyRuntimeError::new_err(
                            "model worker requires action space metadata",
                        )
                    })?;
                let inputs = pyo3::types::PyList::empty(py);
                let contexts = pyo3::types::PyList::empty(py);
                for (index, lane) in lanes.iter().enumerate() {
                    inputs.append(space_value_to_py_neutral(py, lane, space)?)?;
                    contexts.append(episode_context_dict(
                        py,
                        observation.route.episodes.get(index),
                    )?)?;
                }
                let result = batch_fn.call1(py, (inputs, contexts))?;
                let mut actions = Vec::with_capacity(lanes.len());
                for item in result.bind(py).try_iter()? {
                    actions.push(py_any_to_space_value_with_backend(
                        py,
                        &item?,
                        action_space,
                        ValueBackend::Native,
                    )?);
                }
                return Ok(actions);
            }
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
            // An inner model's predict answers this one episode.
            let contract = observation.env_contract.as_ref();
            let action = crate::nested::scoped(self.nested.as_ref(), episode, contract, || {
                self.predict_fn.call1(py, (obs, context))
            })?;
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
                .map_err(|err| {
                    pyo3::exceptions::PyTypeError::new_err(format!(
                        "predict received a batch of {} observations from a vector env and \
                         must return {} actions with a leading batch axis; define \
                         predict_batch for a batched forward, or serve the env as lanes \
                         for one observation per predict ({err})",
                        observation.num_envs, observation.num_envs
                    ))
                })
            }
        })
        .map_err(|err| self.internal(err))
    }

    fn predict_spec_less_chunked(
        &self,
        observation: ModelObservation,
        execution_horizon: u32,
    ) -> rlmesh::Result<PredictFrames> {
        rlmesh::binding::check_model_call()?;
        let horizon = execution_horizon.max(1) as usize;
        if horizon <= 1 || self.predict_chunk_fn.is_none() || observation.num_envs != 1 {
            return Ok(PredictFrames {
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
        Python::attach(|py| -> PyResult<PredictFrames> {
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
            let episode = observation.route.episodes.first();
            let context = episode_context_dict(py, episode)?;
            let chunk = crate::nested::scoped(self.nested.as_ref(), episode, contract, || {
                predict_chunk_fn.call1(py, (obs, horizon as u32, context))
            })?;
            let chunk = chunk.bind(py);
            let frames_len = leading_axis_len(chunk).unwrap_or(horizon);
            let frames = py_any_to_batched_space_values_with_backend(
                py,
                chunk,
                action_space,
                frames_len,
                ValueBackend::Native,
            )?;
            if let Some(native) = self.native_chunk
                && frames.len() != native as usize
            {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "model declares native_chunk={native} but its chunk corner returned {} \
                     frames at execution_horizon={horizon}: return the WHOLE native chunk and \
                     let the runtime execute its prefix (do not slice to the horizon), or drop \
                     the declaration",
                    frames.len()
                )));
            }
            if self.native_chunk.is_none() && frames.len() < horizon {
                // Python's Session warning, once per episode (MDL-P11).
                let key = observation
                    .route
                    .episodes
                    .first()
                    .map(|episode| episode.episode_id.clone())
                    .filter(|id| !id.is_empty())
                    .unwrap_or_else(|| format!("route:{}", observation.route.env_id));
                let first = self
                    .short_warned
                    .lock()
                    .expect("short-chunk set poisoned")
                    .insert(key);
                if first {
                    let message = format!(
                        "the model returned {} actions for execution_horizon={horizon}; \
                         replaying what there is and re-planning early. Declare \
                         Model.native_chunk so the horizon can be validated up front.",
                        frames.len()
                    );
                    py.detach(|| {
                        rlmesh::binding::emit_warning(
                            rlmesh::WarningCategory::RuntimeWarning,
                            message,
                        )
                    })
                    .map_err(to_py_err)?;
                }
            }
            let mut frames = frames.into_iter().take(horizon);
            let first = frames.next().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "predict_chunk returned an empty chunk; return at least one action",
                )
            })?;
            Ok(PredictFrames {
                actions: vec![first],
                replay: frames.map(|frame| vec![frame]).collect(),
            })
        })
        .map_err(|err| self.internal(err))
    }

    fn on_episode_end(&self, episode_id: &str) -> rlmesh::Result<()> {
        self.short_warned
            .lock()
            .expect("short-chunk set poisoned")
            .remove(episode_id);
        let Some(callback) = self.on_episode_end.as_ref() else {
            return Ok(());
        };
        Python::attach(|py| callback.call1(py, (episode_id,)))
            .map(|_| ())
            .map_err(|err| self.internal(err))
    }

    fn on_close(&self) -> rlmesh::Result<()> {
        Self::fire(&self.on_close)
    }

    fn on_route_reset(&self, env_id: &str) -> rlmesh::Result<()> {
        let Some(callback) = self.on_route_drop.as_ref() else {
            return Ok(());
        };
        Python::attach(|py| callback.call1(py, (env_id,)))
            .map(|_| ())
            .map_err(|err| self.internal(err))
    }

    fn on_route_release(&self, env_id: &str) -> rlmesh::Result<()> {
        self.on_route_reset(env_id)
    }

    fn has_reset_options(&self) -> bool {
        self.reset_options_fn.is_some()
    }

    fn reset_options(
        &self,
        episode: &EpisodeInfo,
        trial_index: Option<u64>,
    ) -> rlmesh::Result<Option<rlmesh_spaces::MetaMap>> {
        let Some(reset_options) = self.reset_options_fn.as_ref() else {
            return Ok(None);
        };
        Python::attach(|py| -> PyResult<Option<rlmesh_spaces::MetaMap>> {
            let start = pyo3::types::PyDict::new(py);
            start.set_item("episode_id", &episode.episode_id)?;
            start.set_item("seed", episode.seed)?;
            start.set_item("trial_index", trial_index)?;
            let options = reset_options.call1(py, (start,))?;
            if options.is_none(py) {
                return Ok(None);
            }
            py_any_to_meta_map(options.bind(py)).map(Some)
        })
        .map_err(|err| self.internal(err))
    }
}

/// The per-predict sampling seed for an episode's `predict_index`-th re-plan.
///
/// Exported so a model can derive the same seed the SDK stamps on
/// `context["predict_seed"]` (e.g. for a sampler that takes a generator rather
/// than a global seed). The mixing law lives in the runtime crate, so local and
/// served paths cannot drift.
#[cfg_attr(
    feature = "stub-gen",
    gen_stub_pyfunction(
        module = "rlmesh._rlmesh",
        python = r#"
def predict_seed(episode_seed: int, predict_index: int) -> int: ...
"#
    )
)]
#[pyfunction]
pub fn predict_seed(episode_seed: i64, predict_index: u64) -> i64 {
    rlmesh::predict_seed(episode_seed, predict_index)
}

/// The paired reset seed of episode `slot` under `base_seed` and `seed_key`.
///
/// Private: exported so the cross-language seed fixture checks the runtime's
/// own derivation from Python.
#[cfg_attr(
    feature = "stub-gen",
    gen_stub_pyfunction(
        module = "rlmesh._rlmesh",
        python = r#"
def episode_seed(base_seed: int, seed_key: str, slot: int) -> int: ...
"#
    )
)]
#[pyfunction]
pub fn episode_seed(base_seed: i64, seed_key: &str, slot: u64) -> i64 {
    rlmesh::binding::runtime::episode_seed(base_seed, seed_key, slot)
}

/// Waits up to `timeout_s` seconds for the model at the participant address
/// `address` to answer: ConnectionError when it does not. A sandbox's
/// readiness check.
#[cfg_attr(
    feature = "stub-gen",
    gen_stub_pyfunction(
        module = "rlmesh._rlmesh",
        python = r#"
def model_answers(address: str, timeout_s: float) -> None: ...
"#
    )
)]
#[pyfunction]
pub fn model_answers(py: Python<'_>, address: &str, timeout_s: f64) -> PyResult<()> {
    let timeout = crate::session::timeout(timeout_s)?;
    let id = format!("probe-{}", uuid::Uuid::new_v4().simple());
    py.detach(|| {
        model_runtime().block_on(async {
            let connection = rlmesh::lockstep::ModelConnection::connect(address, &id, timeout)
                .await
                .map_err(|err| pyo3::exceptions::PyConnectionError::new_err(err.to_string()))?;
            connection.close().await;
            Ok(())
        })
    })
}

/// The describe envelope (JSON; empty when it has none) of the model at the
/// participant address `address`, connecting for up to `timeout_s` seconds.
/// How a sandbox env learns a served model's spec before the env is made.
#[cfg_attr(
    feature = "stub-gen",
    gen_stub_pyfunction(
        module = "rlmesh._rlmesh",
        python = r#"
def model_describe(address: str, timeout_s: float) -> str: ...
"#
    )
)]
#[pyfunction]
pub fn model_describe(py: Python<'_>, address: &str, timeout_s: f64) -> PyResult<String> {
    let timeout = crate::session::timeout(timeout_s)?;
    py.detach(|| {
        model_runtime().block_on(async {
            let mut served = rlmesh::lockstep::ServedModel::new(address, timeout);
            let described = served.describe().await;
            let _ = served.on_close().await;
            described.map_err(|err| pyo3::exceptions::PyConnectionError::new_err(err.to_string()))
        })
    })
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

/// The raw predict context handed to a corner as its trailing positional
/// argument: `{"episode_id": str, "episode_seed": int | None}` — one dict for the
/// single-episode corners (`predict`, `predict_chunk`, and the spec-less corners
/// at `num_envs == 1`), one dict PER ROW in the list the batched corners get (see
/// [`call_batched`]). The SDK enriches it into the full `PredictContext`
/// (`predict_index` / `predict_seed` / `state`) from its own episode store, and
/// only forwards it to the author's callback when that callback's signature
/// accepts a trailing context argument, so this is additive — an author who never
/// asked for it never sees it.
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
    // The route the row arrived on: the store tags its entry with it, so the
    // env's teardown drops it (MDL-P4).
    dict.set_item(
        "env_id",
        episode.map(|episode| episode.env_id.as_str()).unwrap_or(""),
    )?;
    Ok(dict)
}

/// Call a Python batched corner (`predict_batch` / `predict_chunk_batch`) with N
/// assembled lane inputs and decode its N returned actions/chunks. The Python side
/// receives a list of N neutral input dicts and returns a sequence of N actions
/// (one per lane, in order); the model owns how it batches the forward pass. When
/// `horizon` is `Some(h)` (the chunk-batch corner) it is passed as a second
/// argument, so the model can size each chunk to the execution horizon.
///
/// `episodes` is row-aligned with `inputs` and rides as the trailing argument: a
/// LIST of the same per-lane context dicts the single-episode corners get. N lanes
/// fused into one forward may come from independent episodes, so identity is per
/// row — never one dict for the call.
fn call_batched(
    fn_obj: &Py<PyAny>,
    inputs: Vec<Value>,
    horizon: Option<u32>,
    episodes: &[EpisodeInfo],
) -> PyResult<Vec<Value>> {
    Python::attach(|py| -> PyResult<Vec<Value>> {
        let list = pyo3::types::PyList::empty(py);
        for model_input in &inputs {
            // Each lane's assembled input is a Value tree, encoded as one element.
            list.append(encode_value(py, model_input)?)?;
        }
        let contexts = pyo3::types::PyList::empty(py);
        for episode in episodes {
            contexts.append(episode_context_dict(py, Some(episode))?)?;
        }
        let result = match horizon {
            Some(h) => fn_obj.call1(py, (list, h, contexts))?,
            None => fn_obj.call1(py, (list, contexts))?,
        };
        let mut out = Vec::with_capacity(inputs.len());
        for item in result.bind(py).try_iter()? {
            out.push(decode_value(&item?)?);
        }
        Ok(out)
    })
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
        .map_err(|err| crate::types::resolve_callback_error(&err))
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
    native_chunk: Option<u32>,
    configure_fn: Option<Py<PyAny>>,
    on_episode_end: Option<Py<PyAny>>,
    on_close: Option<Py<PyAny>>,
    on_route_drop: Option<Py<PyAny>>,
    profiler: Arc<ProfileCollector>,
    last_error: LastPyError,
    /// The running serves' shutdown triggers, and whether `stop_serving`
    /// fired (sticky: the model is closing, so no serve starts again).
    serving: Arc<std::sync::Mutex<Serving>>,
    /// A served model's participant address and connect timeout: the
    /// worker then drives that model instead of the callbacks.
    served: Option<(String, Duration)>,
    /// The model's `reset_options(episode)`, when it defines one.
    reset_options_fn: Option<Py<PyAny>>,
    /// The workers of the models this model calls from its predict, by the
    /// Python `id` of each.
    inners: std::sync::Mutex<Vec<(u64, Py<PyModel>)>>,
    /// The random baseline (`RANDOM_SAMPLE`) rather than the callbacks.
    random: bool,
    run_gate: Arc<std::sync::atomic::AtomicBool>,
}

#[derive(Default)]
struct Serving {
    stopped: bool,
    triggers: Vec<rlmesh_grpc::lifecycle::ShutdownTrigger>,
}

impl PyModel {
    /// Build the served engine handler: the predict hole plus, for a spec'd
    /// model, the route resolver. The engine owns the per-route adapter state and
    /// frame buffers; this layer only supplies the host (Python) callbacks.
    fn build_handler(&self) -> Box<dyn ModelHandler> {
        if self.random {
            return Box::<rlmesh::binding::model::RandomSampleHandler>::default();
        }
        if let Some((address, timeout)) = &self.served {
            return Box::new(rlmesh::lockstep::ServedModel::new(
                address.clone(),
                *timeout,
            ));
        }
        let nested = Python::attach(|py| {
            let inners = self.inners.lock().expect("inner models lock poisoned");
            (!inners.is_empty()).then(|| {
                crate::nested::Nested::new(
                    inners
                        .iter()
                        .map(|(key, inner)| (*key, inner.borrow(py).build_handler()))
                        .collect(),
                )
            })
        });
        let predict = Python::attach(|py| PyPredict {
            predict_fn: self.predict_fn.clone_ref(py),
            predict_chunk_fn: self.predict_chunk_fn.as_ref().map(|cb| cb.clone_ref(py)),
            predict_batch_fn: self.predict_batch_fn.as_ref().map(|cb| cb.clone_ref(py)),
            predict_chunk_batch_fn: self
                .predict_chunk_batch_fn
                .as_ref()
                .map(|cb| cb.clone_ref(py)),
            allow_fusion: self.allow_fusion,
            native_chunk: self.native_chunk,
            on_episode_end: self.on_episode_end.as_ref().map(|cb| cb.clone_ref(py)),
            on_close: self.on_close.as_ref().map(|cb| cb.clone_ref(py)),
            on_route_drop: self.on_route_drop.as_ref().map(|cb| cb.clone_ref(py)),
            short_warned: Default::default(),
            last_error: Arc::clone(&self.last_error),
            reset_options_fn: self.reset_options_fn.as_ref().map(|cb| cb.clone_ref(py)),
            nested: nested.clone(),
        });
        let resolver: Option<Arc<dyn RouteResolver>> =
            self.configure_fn.as_ref().map(|configure| {
                let configure_fn = Python::attach(|py| configure.clone_ref(py));
                Arc::new(PyRouteResolver { configure_fn }) as Arc<dyn RouteResolver>
            });
        let handler = Box::new(AdaptedModelHandler::new(Arc::new(predict), resolver));
        match nested {
            Some(nested) => Box::new(crate::nested::NestedHandler {
                outer: handler,
                nested,
            }),
            None => handler,
        }
    }

    /// The run options a facade call's `options` dict names, checked as
    /// Python's keywords are.
    fn run_options(&self, options: &Bound<'_, pyo3::types::PyDict>) -> PyResult<ParsedRunOptions> {
        macro_rules! option {
            ($key:literal, $ty:ty) => {
                options
                    .get_item($key)?
                    .filter(|v| !v.is_none())
                    .map(|v| v.extract::<$ty>())
                    .transpose()?
            };
        }
        let nonnegative = |name: &str, v: i128| -> PyResult<u64> {
            u64::try_from(v).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "{name} must be >= 0 and <= {}, got {v}",
                    u64::MAX
                ))
            })
        };
        let episodes = option!("episodes", i128)
            .map(|v| nonnegative("episodes", v))
            .transpose()?;
        let seeds = option!("seeds", Vec<i64>);
        let max_episode_steps = option!("max_episode_steps", i128)
            .map(|v| nonnegative("max_episode_steps", v))
            .transpose()?;
        let max_episode = option!("max_episode_seconds", f64)
            .map(|v| {
                if v == f64::INFINITY {
                    return Ok(std::time::Duration::MAX);
                }
                std::time::Duration::try_from_secs_f64(v).map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "max_episode_seconds must be > 0, got {}",
                        if v.is_nan() {
                            "nan".to_owned()
                        } else {
                            v.to_string()
                        }
                    ))
                })
            })
            .transpose()?;
        let execution_horizon = option!("execution_horizon", i128)
            .map(|v| {
                rlmesh::binding::resolve_execution_horizon(Some(v), None, None, self.native_chunk)
                    .map_err(to_py_err)
            })
            .transpose()?;
        let prefetch_lead =
            nonnegative("prefetch_lead", option!("prefetch_lead", i128).unwrap_or(0))?;
        let trial_index_base = nonnegative(
            "trial_index_base",
            option!("trial_index_base", i128).unwrap_or(0),
        )?;
        Ok(ParsedRunOptions {
            options: rlmesh::RunOptions {
                episodes,
                seeds,
                max_episode_steps,
                max_episode,
                execution_horizon,
                prefetch_lead,
                trial_index_base,
                close_env: option!("close_env", bool).unwrap_or(false),
                workflow_edition: option!("workflow_edition", String),
                ..Default::default()
            },
            default_horizon: option!("default_horizon", u32),
            base_seed: option!("base_seed", i64),
            seed_key: option!("seed_key", String),
        })
    }

    /// A handler for one run or binding; a run that does not hold the model
    /// (`exclusive` false) gets a gate of its own.
    fn prepared(&self, exclusive: bool) -> rlmesh::binding::model::PreparedModel {
        rlmesh::binding::model::PreparedModel {
            handler: self.build_handler(),
            presentation: None,
            run_gate: if exclusive {
                self.run_gate.clone()
            } else {
                Arc::default()
            },
            run_lease: None,
            native_chunk: self.native_chunk,
            execution_horizon: None,
            workflow_edition: None,
            authored: false,
            has_chunk: (self.served.is_none() && !self.random)
                .then_some(self.predict_chunk_fn.is_some()),
        }
    }

    /// Serves the model as the participant `name` in the session at
    /// `address`, declaring `workflow_edition`, until a signal or
    /// [`stop_serving`](Self::stop_serving).
    #[allow(clippy::too_many_arguments)] // the serve call's settings, passed through
    fn serve_on_session(
        &self,
        py: Python<'_>,
        address: &str,
        name: &str,
        workflow_edition: Option<&str>,
        batching: rlmesh::BatchOptions,
        close: Duration,
        envelope: Option<&str>,
    ) -> PyResult<()> {
        let participant = if address.trim().is_empty() {
            name.to_string()
        } else {
            format!("{}/{name}", address.trim().trim_end_matches('/'))
        };
        // The describe envelope `rlmesh.serve` stamped for the handshake.
        let describe = envelope.map(str::to_owned).unwrap_or_else(|| {
            rlmesh::binding::peer_info_override()
                .and_then(|info| info.extra.get(rlmesh::DESCRIBE_METADATA_KEY).cloned())
                .unwrap_or_default()
        });
        let service = Arc::new(
            rlmesh::lockstep::ModelService::new(self.build_handler(), describe)
                .declaring(workflow_edition)
                .batching(batching)
                .map_err(to_py_err)?,
        );
        let serving = Arc::clone(&self.serving);
        let shutdown = rlmesh_grpc::lifecycle::ShutdownTrigger::new();
        // SIGTERM stops the serve gracefully, as an env server's (`docker
        // stop`); restored when the serve returns. SIGINT stays Python's.
        let _sigterm = rlmesh::binding::foreground_sigterm_scope(shutdown.clone());
        py.detach(|| {
            model_runtime().block_on(async move {
                let binding = rlmesh::lockstep::ModelBinding::bind(&participant)
                    .await
                    .map_err(crate::lockstep::gym_err)?;
                {
                    let mut serving = serving.lock().expect("serving lock poisoned");
                    if serving.stopped {
                        shutdown.trigger("model closed");
                    }
                    serving.triggers.push(shutdown.clone());
                }
                crate::server::py_env_server::spawn_signal_shutdown(shutdown.clone());
                let cancel = tokio_util::sync::CancellationToken::new();
                let stop = {
                    let (shutdown, cancel) = (shutdown.clone(), cancel.clone());
                    tokio::spawn(async move {
                        shutdown.cancelled_owned().await;
                        cancel.cancel();
                    })
                };
                tracing::info!(address = %binding.address(), "model serving");
                let mut serve = std::pin::pin!(binding.serve(service, close, cancel.clone()));
                let mut poll = crate::client::signal_poll();
                let result = loop {
                    tokio::select! {
                        result = &mut serve => break result.map_err(crate::lockstep::gym_err),
                        _ = poll.tick() => {
                            if let Err(err) = crate::client::check_signals() {
                                cancel.cancel();
                                let _ = (&mut serve).await;
                                break Err(err);
                            }
                        }
                    }
                };
                stop.abort();
                result
            })
        })
    }
}

#[cfg_attr(feature = "stub-gen", gen_stub_pyfunction(module = "rlmesh._rlmesh"))]
#[pyfunction]
#[pyo3(signature = (explicit=None, loaded=None, timing=None, native_chunk=None))]
pub fn resolve_execution_horizon(
    explicit: Option<i128>,
    loaded: Option<i128>,
    timing: Option<i128>,
    native_chunk: Option<u32>,
) -> PyResult<u32> {
    rlmesh::binding::resolve_execution_horizon(explicit, loaded, timing, native_chunk)
        .map_err(to_py_err)
}

#[pymethods]
impl PyModel {
    #[new]
    #[pyo3(signature = (predict_fn, configure_fn=None, on_episode_end=None, on_close=None, predict_chunk_fn=None, predict_batch_fn=None, predict_chunk_batch_fn=None, allow_fusion=true, native_chunk=None, on_route_drop=None, reset_options_fn=None))]
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
        native_chunk: Option<u32>,
        on_route_drop: Option<Py<PyAny>>,
        reset_options_fn: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        init_tracing("model_worker");
        let profiler = ProfileCollector::new("model_worker");

        Ok(Self {
            predict_fn,
            predict_chunk_fn,
            predict_batch_fn,
            predict_chunk_batch_fn,
            allow_fusion,
            native_chunk,
            configure_fn,
            on_episode_end,
            on_close,
            on_route_drop,
            profiler,
            last_error: LastPyError::default(),
            serving: Arc::default(),
            served: None,
            reset_options_fn,
            inners: Default::default(),
            random: false,
            run_gate: Arc::default(),
        })
    }

    /// A worker that drives the model served at the participant address
    /// `address`, waiting up to `timeout_s` seconds to reach it.
    #[staticmethod]
    #[pyo3(signature = (address, timeout_s=30.0))]
    fn served(py: Python<'_>, address: &str, timeout_s: f64) -> PyResult<Self> {
        init_tracing("model_worker");
        Ok(Self {
            predict_fn: py.None(),
            predict_chunk_fn: None,
            predict_batch_fn: None,
            predict_chunk_batch_fn: None,
            allow_fusion: true,
            native_chunk: None,
            configure_fn: None,
            on_episode_end: None,
            on_close: None,
            on_route_drop: None,
            profiler: ProfileCollector::new("model_worker"),
            last_error: LastPyError::default(),
            serving: Arc::default(),
            served: Some((address.to_string(), crate::session::timeout(timeout_s)?)),
            reset_options_fn: None,
            inners: Default::default(),
            random: false,
            run_gate: Arc::default(),
        })
    }

    /// The random baseline: samples the env's action space.
    #[staticmethod]
    fn random_sample(py: Python<'_>) -> Self {
        init_tracing("model_worker");
        Self {
            predict_fn: py.None(),
            predict_chunk_fn: None,
            predict_batch_fn: None,
            predict_chunk_batch_fn: None,
            allow_fusion: true,
            native_chunk: None,
            configure_fn: None,
            on_episode_end: None,
            on_close: None,
            on_route_drop: None,
            profiler: ProfileCollector::new("model_worker"),
            last_error: LastPyError::default(),
            serving: Arc::default(),
            served: None,
            reset_options_fn: None,
            inners: Default::default(),
            random: true,
            run_gate: Arc::default(),
        }
    }

    /// Language target and presentation conversion over the shared run facade.
    fn run_facade(
        &self,
        py: Python<'_>,
        target: Py<PyAny>,
        relay: Py<PyAny>,
        options: &Bound<'_, pyo3::types::PyDict>,
    ) -> PyResult<Py<PyAny>> {
        // A served model answers each run in a session of its own and the
        // random baseline builds a handler per run, so only an in-process
        // model is held for the run (and its errors are the run's).
        let exclusive = self.served.is_none() && !self.random;
        let lease = exclusive
            .then(|| rlmesh::binding::model::claim_run(self.run_gate.clone()))
            .transpose()
            .map_err(to_py_err)?;
        let last_error = if exclusive {
            self.last_error.clone()
        } else {
            LastPyError::default()
        };
        let parsed = self.run_options(options)?;
        let cancel = rlmesh::ext::CancelToken::new();
        let signal_cancel = cancel.clone();
        last_error.lock().unwrap_or_else(|e| e.into_inner()).take();
        let target = crate::facade::PyEnvTarget::new(py, target, last_error.clone())?;
        let mut hooks = crate::facade::PyFacadeHooks::new(relay, last_error.clone());
        let mut prepared = self.prepared(exclusive);
        prepared.presentation = Some(Box::new(hooks.clone()));
        prepared.run_lease = lease;
        prepared.execution_horizon = parsed.default_horizon;
        let options = rlmesh::RunOptions {
            cancel: Some(cancel),
            hooks: Some(&mut hooks),
            ..parsed.options
        };
        let poll = crate::facade::signal_poll(last_error.clone(), move || signal_cancel.cancel());
        let result = py.detach(|| {
            rlmesh::binding::model::with_caller_poll(poll, || {
                rlmesh::binding::model::run_prepared(
                    prepared,
                    rlmesh::binding::model::adapted_env(Box::new(target)),
                    options,
                )
            })
        });
        match result {
            Ok(result) => Ok(crate::facade::result_to_py(py, &result)?
                .into_any()
                .unbind()),
            Err(error) => Err(last_error
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take()
                .unwrap_or_else(|| to_py_err(error))),
        }
    }

    /// `Model.run_routes`: drives every `(target, options)` route at once
    /// and returns, per route, its result dict or the exception it failed
    /// with. A signal interrupts every route and raises.
    fn run_routes_facade(
        &self,
        py: Python<'_>,
        routes: Vec<(Py<PyAny>, Bound<'_, pyo3::types::PyDict>)>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let exclusive = self.served.is_none() && !self.random;
        if exclusive && routes.len() > 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "run_routes: an in-process Model drives one route at a time, got {}; serve it \
                 (or use a SandboxModel) and run the served model across routes",
                routes.len()
            )));
        }
        let lease = exclusive
            .then(|| rlmesh::binding::model::claim_run(self.run_gate.clone()))
            .transpose()
            .map_err(to_py_err)?;
        let cancel = rlmesh::ext::CancelToken::new();
        let mut errors = Vec::with_capacity(routes.len());
        let mut prepared = Vec::with_capacity(routes.len());
        for (target, options) in routes {
            let parsed = self.run_options(&options)?;
            let error = LastPyError::default();
            let target = crate::facade::PyEnvTarget::new(py, target, error.clone())?;
            let mut model = self.prepared(exclusive);
            model.run_lease = lease.clone();
            model.execution_horizon = parsed.default_horizon;
            prepared.push((
                model,
                rlmesh::binding::routes::Route {
                    env: rlmesh::binding::model::adapted_env(Box::new(target)),
                    options: rlmesh::RunOptions {
                        cancel: Some(cancel.clone()),
                        ..parsed.options
                    },
                    base_seed: parsed.base_seed,
                    seed_key: parsed.seed_key,
                },
            ));
            errors.push(error);
        }
        let interrupted = LastPyError::default();
        let signal_cancel = cancel.clone();
        let poll = crate::facade::signal_poll(interrupted.clone(), move || signal_cancel.cancel());
        let results = py.detach(|| {
            rlmesh::binding::model::with_caller_poll(poll, || {
                rlmesh::binding::routes::run_routes_prepared(prepared)
            })
        });
        // Ctrl-C cancels every route: those it cut short hold the interrupt,
        // so the caller keeps the ones that finished before raising it.
        let interrupt = interrupted.lock().unwrap_or_else(|e| e.into_inner()).take();
        results
            .into_iter()
            .zip(errors)
            .map(|(result, error)| match (result, &interrupt) {
                (Ok(result), _) => Ok(crate::facade::result_to_py(py, &result)?
                    .into_any()
                    .unbind()),
                (Err(rlmesh::Error::Cancelled), Some(interrupt)) => {
                    Ok(interrupt.clone_ref(py).into_value(py).into_any())
                }
                (Err(failure), _) => Ok(error
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .take()
                    .unwrap_or_else(|| to_py_err(failure))
                    .into_value(py)
                    .into_any()),
            })
            .collect()
    }

    #[pyo3(signature = (target, execution_horizon=1, workflow_edition=None, close_env=false))]
    fn bind_facade(
        &self,
        py: Python<'_>,
        target: Py<PyAny>,
        execution_horizon: u32,
        workflow_edition: Option<String>,
        close_env: bool,
    ) -> PyResult<crate::bind::PyBinding> {
        self.last_error
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        let target = crate::facade::PyEnvTarget::new(py, target, self.last_error.clone())?;
        let model = self.prepared(true);
        let result = py.detach(|| {
            rlmesh::binding::model::bind_prepared(
                model,
                rlmesh::binding::model::adapted_env(Box::new(target)),
                rlmesh::BindOptions {
                    execution_horizon: Some(execution_horizon),
                    workflow_edition,
                    close_env,
                    ..Default::default()
                },
            )
        });
        let binding = result.map_err(|error| {
            self.last_error
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take()
                .unwrap_or_else(|| to_py_err(error))
        })?;
        crate::bind::PyBinding::from_binding(py, binding, self.last_error.clone())
    }

    /// Binds the model to the env at the session address `env_address`, to
    /// step by hand.
    #[pyo3(signature = (env_address, execution_horizon=1, workflow_edition=None))]
    fn bind(
        &self,
        py: Python<'_>,
        env_address: &str,
        execution_horizon: u32,
        workflow_edition: Option<String>,
    ) -> PyResult<crate::bind::PyBinding> {
        crate::bind::PyBinding::connect(
            py,
            self.prepared(true),
            Arc::clone(&self.last_error),
            env_address,
            execution_horizon,
            workflow_edition,
        )
    }

    /// The workers of the models this model calls from its predict, by the
    /// Python `id` of each: a run, bind or serve of this model drives them
    /// on its episodes.
    /// Read this served model's default through its describe envelope.
    fn default_execution_horizon(&self, py: Python<'_>) -> PyResult<Option<u32>> {
        let Some((address, timeout)) = &self.served else {
            return Ok(None);
        };
        py.detach(|| {
            model_runtime().block_on(rlmesh::lockstep::ServedModel::default_execution_horizon(
                address, *timeout,
            ))
        })
        .map_err(to_py_err)
    }

    fn set_inner_models(&self, inners: Vec<(u64, Py<PyModel>)>) {
        *self.inners.lock().expect("inner models lock poisoned") = inners;
    }

    /// Stop every serve running on this worker (drain, then return), and any
    /// serve that starts later: the model is closing.
    fn stop_serving(&self) {
        let mut serving = self.serving.lock().expect("serving lock poisoned");
        serving.stopped = true;
        for trigger in &serving.triggers {
            trigger.trigger("model closed");
        }
    }

    #[pyo3(signature = (env_address, execution_horizon=1, prefetch_lead=0, workflow_edition=None))]
    fn run_local(
        &self,
        py: Python<'_>,
        env_address: &str,
        execution_horizon: u32,
        prefetch_lead: u32,
        workflow_edition: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        self.run_local_for_episodes(
            py,
            env_address,
            1,
            execution_horizon,
            None,
            None,
            None,
            false,
            None,
            prefetch_lead,
            workflow_edition,
            None,
            None,
        )
    }

    /// `hooks` is a relay object receiving the runtime's per-episode events as
    /// plain data: `episode_started(episode_id, index, env_index, seed, trial)`,
    /// `observation(env_index, episode_ids, values)`, `action(env_index, values)`,
    /// `step(env_index, rewards, terminated, truncated, autoreset_roll, info)`,
    /// and `episode_completed(episode_id, index, env_index, seed, trial, steps,
    /// reward, terminated, truncated, success, duration_s)`. An exception it
    /// raises aborts the run and is re-raised as-is. `invite_at` is the env's
    /// session as a model served elsewhere reaches it, when that differs from
    /// `env_address`'s.
    #[pyo3(signature = (env_address, episodes, execution_horizon=1, seeds=None, max_episode_steps=None, max_episode_seconds=None, close_env=false, trial_index_base=None, prefetch_lead=0, workflow_edition=None, hooks=None, invite_at=None))]
    #[allow(clippy::too_many_arguments)]
    fn run_local_for_episodes(
        &self,
        py: Python<'_>,
        env_address: &str,
        episodes: u64,
        execution_horizon: u32,
        seeds: Option<Vec<i64>>,
        max_episode_steps: Option<i64>,
        max_episode_seconds: Option<f64>,
        close_env: bool,
        trial_index_base: Option<u64>,
        prefetch_lead: u32,
        workflow_edition: Option<String>,
        hooks: Option<Py<PyAny>>,
        invite_at: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        let lease = rlmesh::binding::model::claim_run(self.run_gate.clone()).map_err(to_py_err)?;
        let mut hooks = crate::facade::PyFacadeHooks::optional(hooks, self.last_error.clone());
        let cancel = rlmesh::ext::CancelToken::new();
        let signal = cancel.clone();
        let max_episode = max_episode_seconds
            .map(|seconds| {
                if seconds == f64::INFINITY {
                    return Ok(Duration::MAX);
                }
                Duration::try_from_secs_f64(seconds).map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err("max_episode_seconds must be > 0")
                })
            })
            .transpose()?;
        let max_episode_steps = max_episode_steps
            .map(|steps| {
                u64::try_from(steps).map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err("max_episode_steps must be >= 1")
                })
            })
            .transpose()?;
        let mut prepared = self.prepared(true);
        prepared.presentation = Some(Box::new(hooks.clone()));
        prepared.run_lease = Some(lease.clone());
        let options = rlmesh::RunOptions {
            episodes: Some(episodes),
            seeds,
            max_episode_steps,
            max_episode,
            execution_horizon: Some(execution_horizon),
            prefetch_lead: prefetch_lead.into(),
            trial_index_base: trial_index_base.unwrap_or(0),
            workflow_edition,
            close_env,
            hooks: Some(&mut hooks),
            cancel: Some(cancel),
            ..Default::default()
        };
        let target = crate::facade::AddressTarget {
            address: env_address.to_owned(),
            invite_at,
        };
        let poll = crate::facade::signal_poll(self.last_error.clone(), move || signal.cancel());
        self.last_error
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        let result = py
            .detach(|| {
                rlmesh::binding::model::with_caller_poll(poll, || {
                    rlmesh::binding::model::run_prepared(
                        prepared,
                        rlmesh::binding::model::adapted_env(Box::new(target)),
                        options,
                    )
                })
            })
            .map_err(|error| {
                self.last_error
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .take()
                    .unwrap_or_else(|| to_py_err(error))
            })?;
        let out = crate::facade::result_to_py(py, &result)?;
        // Compatibility of this private, low-level report: runtime ordinals
        // were one-based. Public run/hooks use the facade's zero-based records.
        for episode in out
            .get_item("episodes")?
            .expect("report episodes")
            .try_iter()?
        {
            let episode = episode?;
            let index: i64 = episode.get_item("index")?.extract()?;
            episode.set_item("index", index + 1)?;
        }
        Ok(out.into_any().unbind())
    }

    /// Serves the model as the participant `name` in the session at
    /// `address`, until a signal or [`stop_serving`](Self::stop_serving).
    ///
    /// The batch settings are `rlmesh.BatchOptions`' fields; `None` takes
    /// that field's default.
    #[pyo3(signature = (address, name, options=None, envelope=None, max_batch_size=None, batch_window_ms=None, batch_byte_budget=None, max_in_flight_batches=None, batch_records=None))]
    #[allow(clippy::too_many_arguments)] // each batch setting is a Python kwarg
    fn serve(
        &self,
        py: Python<'_>,
        address: &str,
        name: &str,
        options: Option<PyServeOptions>,
        envelope: Option<&str>,
        max_batch_size: Option<usize>,
        batch_window_ms: Option<u64>,
        batch_byte_budget: Option<usize>,
        max_in_flight_batches: Option<usize>,
        batch_records: Option<std::path::PathBuf>,
    ) -> PyResult<()> {
        let defaults = rlmesh::BatchOptions::default();
        let batching = rlmesh::BatchOptions {
            max_batch_size: max_batch_size.unwrap_or(defaults.max_batch_size),
            batch_window_ms: batch_window_ms.unwrap_or(defaults.batch_window_ms),
            batch_byte_budget: batch_byte_budget.unwrap_or(defaults.batch_byte_budget),
            max_in_flight_batches: max_in_flight_batches.unwrap_or(defaults.max_in_flight_batches),
            batch_records,
        };
        let run_span = tracing::info_span!("rlmesh.model.serve", address = address);
        let _run_enter = run_span.enter();
        let total_guard = self.profiler.start("model.serve.total");
        let options = options.map(PyServeOptions::into_rust).unwrap_or_default();
        rlmesh::binding::check_session_model_serve(&options).map_err(to_py_err)?;
        let close = options.close_timeout.unwrap_or(Duration::from_secs(30));
        self.serve_on_session(
            py,
            address,
            name,
            options.workflow_edition.as_deref(),
            batching,
            close,
            envelope,
        )?;
        let _ = total_guard.finish(0);
        Ok(())
    }
}

#[cfg(feature = "stub-gen")]
submit! {
    gen_methods_from_python! {
        r#"
import collections.abc
import os
import typing

class PyModel:
    def __init__(self, predict_fn: collections.abc.Callable[[Value], Value], configure_fn: collections.abc.Callable[[EnvContract], object] | None = None, on_episode_end: collections.abc.Callable[[str], None] | None = None, on_close: collections.abc.Callable[[], None] | None = None, predict_chunk_fn: collections.abc.Callable[[Value, int], Value] | None = None, predict_batch_fn: collections.abc.Callable[[list[Value], list[dict[str, typing.Any]]], list[Value]] | None = None, predict_chunk_batch_fn: collections.abc.Callable[[list[Value], int, list[dict[str, typing.Any]]], list[Value]] | None = None, allow_fusion: bool = True, native_chunk: int | None = None, on_route_drop: collections.abc.Callable[[str], None] | None = None, reset_options_fn: collections.abc.Callable[[dict[str, typing.Any]], typing.Any] | None = None) -> None: ...
    @staticmethod
    def served(address: str, timeout_s: float = 30.0) -> PyModel: ...
    def default_execution_horizon(self) -> int | None: ...
    def set_inner_models(self, inners: list[tuple[int, PyModel]]) -> None: ...
    @staticmethod
    def random_sample() -> PyModel: ...
    def bind(self, env_address: str, execution_horizon: int = 1, workflow_edition: str | None = None) -> PyBinding: ...
    def run_facade(self, target: object, relay: object, options: dict[str, typing.Any]) -> dict[str, typing.Any]: ...
    def run_routes_facade(self, routes: list[tuple[object, dict[str, typing.Any]]]) -> list[typing.Any]: ...
    def bind_facade(self, target: object, execution_horizon: int = 1, workflow_edition: str | None = None, close_env: bool = False) -> PyBinding: ...
    def stop_serving(self) -> None: ...
    def run_local(self, env_address: str, execution_horizon: int = 1, prefetch_lead: int = 0, workflow_edition: str | None = None) -> dict[str, typing.Any]: ...
    def run_local_for_episodes(self, env_address: str, episodes: int, execution_horizon: int = 1, seeds: list[int] | None = None, max_episode_steps: int | None = None, max_episode_seconds: float | None = None, close_env: bool = False, trial_index_base: int | None = None, prefetch_lead: int = 0, workflow_edition: str | None = None, hooks: object | None = None, invite_at: str | None = None) -> dict[str, typing.Any]: ...
    def serve(self, address: str, name: str, options: ServeOptions | None = None, envelope: str | None = None, max_batch_size: int | None = None, batch_window_ms: int | None = None, batch_byte_budget: int | None = None, max_in_flight_batches: int | None = None, batch_records: str | os.PathLike[str] | None = None) -> None: ...
"#
    }
}

impl Drop for PyModel {
    fn drop(&mut self) {
        self.profiler.log_summary_once();
    }
}

#[cfg(test)]
mod review_regressions {
    use super::*;
    use pyo3::types::PyDictMethods;
    use rlmesh::binding::model::PredictFn;

    fn batch_corner_error_regression(chunked: bool) {
        Python::attach(|py| {
            let exception = crate::types::ModelException::new_err("batch refused");
            exception.value(py).setattr("is_recoverable", true).unwrap();
            let globals = pyo3::types::PyDict::new(py);
            globals.set_item("failure", exception.value(py)).unwrap();
            py.run(c"def fail(*args):\n    raise failure", Some(&globals), None)
                .unwrap();
            let fail = globals.get_item("fail").unwrap().unwrap().unbind();
            let predict = PyPredict {
                predict_fn: fail.clone_ref(py),
                predict_chunk_fn: None,
                predict_batch_fn: Some(fail.clone_ref(py)),
                predict_chunk_batch_fn: Some(fail),
                allow_fusion: true,
                native_chunk: None,
                on_episode_end: None,
                on_close: None,
                on_route_drop: None,
                short_warned: Default::default(),
                reset_options_fn: None,
                nested: None,
                last_error: Default::default(),
            };
            {
                let error = if chunked {
                    predict.predict_chunk_batch(vec![Value::Number(0.)], 2, &[])
                } else {
                    predict.predict_batch(vec![Value::Number(0.)], &[])
                }
                .unwrap_err();
                assert_eq!(error.class(), rlmesh::ErrorClass::ModelException);
                assert!(error.is_recoverable());
                let RLMeshError::Model(error) = error else {
                    panic!()
                };
                assert_eq!(error.message, "batch refused");
                let original = predict.last_error.lock().unwrap().take().unwrap();
                assert!(original.value(py).is(exception.value(py)));
            }
        });
    }
    #[test]
    fn batch_preserves_model_error_identity_and_wire_text() {
        batch_corner_error_regression(false);
    }

    #[test]
    fn chunk_batch_preserves_model_error_identity_and_wire_text() {
        batch_corner_error_regression(true);
    }
}
