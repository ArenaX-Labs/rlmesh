//! PyEnvironment - Rust adapter for Python gymnasium environments.

use async_trait::async_trait;
use image::ColorType;
use image::ImageEncoder;
use image::codecs::png::PngEncoder;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use rlmesh::{
    CloseResult as EnvCloseResult, Env as RLMeshEnv, ResetRequest as EnvResetRequest,
    ResetResult as EnvResetResult, StepRequest as EnvStepRequest, StepResult as EnvStepResult,
};
use rlmesh_grpc::error::{EnvError, EnvErrorCode};
use rlmesh_spaces::errors::EnvRuntimeError;
use rlmesh_spaces::v1::spaces::SpaceSpec;
use rlmesh_spaces::v1::{
    CloseRequest, CloseRequest as SingleCloseRequest, CloseResult as SingleCloseResult,
    EnvContract, MetaMap, RenderFrame as NativeRenderFrame, RenderRequest,
    RenderRequest as SingleRenderRequest, RenderResult, RenderResult as SingleRenderResult,
    ResetRequest as SingleResetRequest, ResetResult as SingleResetResult,
    StepRequest as SingleStepRequest, StepResult as SingleStepResult,
};
use std::sync::Arc;

use crate::spaces::{
    ValueBackend, batched_space_values_to_py_with_backend, meta_map_to_pydict,
    py_any_to_batched_space_values_with_backend, py_any_to_meta_map,
    py_any_to_space_value_with_backend, space_value_to_py_with_backend,
};
use crate::telemetry::ProfileCollector;

/// A Rust wrapper around a Python gymnasium environment.
///
/// Wraps a Python gymnasium environment for the RLMesh protocol.
/// Manages GIL acquisition, numpy array encoding, and async execution.
pub struct PyEnvironment {
    /// The Python environment object (cached)
    env: Py<PyAny>,
    /// Cached observation space (parsed once at construction)
    observation_space: SpaceSpec,
    /// Cached action space (parsed once at construction)
    action_space: SpaceSpec,
    /// Cached spaces spec
    env_contract: EnvContract,
    /// Number of parallel environments (1 for single env)
    num_envs: usize,
    /// Whether the wrapped Python object uses the Gymnasium vector-env API.
    uses_vector_api: bool,
    /// Per-process profiling summary
    profiler: Arc<ProfileCollector>,
}

pub struct PySingleEnv(PyEnvironment);

pub struct PyVectorEnv(PyEnvironment);

pub enum PyServerEnv {
    Single(PySingleEnv),
    Vector(PyVectorEnv),
}

impl PyServerEnv {
    pub fn env_contract(&self) -> &EnvContract {
        match self {
            PyServerEnv::Single(env) => &env.0.env_contract,
            PyServerEnv::Vector(env) => &env.0.env_contract,
        }
    }

    pub async fn close(&mut self) -> Result<(), EnvRuntimeError> {
        match self {
            PyServerEnv::Single(env) => env.close(CloseRequest::default()).await.map(|_| ()),
            PyServerEnv::Vector(env) => env.close(CloseRequest::default()).await.map(|_| ()),
        }
    }
}

impl PyEnvironment {
    /// Create a new PyEnvironment from a Python env object.
    ///
    /// # Arguments
    /// * `env` - Python gymnasium.Env or gymnasium.vector.VectorEnv object
    ///
    /// # Errors
    /// Returns error if:
    /// - `env` doesn't have required methods (reset, step, close)
    /// - Space parsing fails
    pub fn new(env: Py<PyAny>) -> PyResult<Self> {
        let profiler = ProfileCollector::new("env_server");
        Python::attach(|py| {
            let env_ref = env.bind(py);
            if !env_ref.hasattr("reset")? {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "env must have a 'reset' method",
                ));
            }
            if !env_ref.hasattr("step")? {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "env must have a 'step' method",
                ));
            }
            if !env_ref.hasattr("close")? {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "env must have a 'close' method",
                ));
            }

            let num_envs = if env_ref.hasattr("num_envs")? {
                env_ref.getattr("num_envs")?.extract::<usize>()?
            } else {
                1
            };
            let uses_vector_api = env_ref.hasattr("single_observation_space")?
                || env_ref.hasattr("single_action_space")?
                || env_ref.hasattr("num_envs")?;

            let obs_space_attr =
                if uses_vector_api && env_ref.hasattr("single_observation_space")? {
                    "single_observation_space"
                } else {
                    "observation_space"
                };

            let action_space_attr = if uses_vector_api && env_ref.hasattr("single_action_space")? {
                "single_action_space"
            } else {
                "action_space"
            };

            let obs_space_py = env_ref.getattr(obs_space_attr)?;
            let action_space_py = env_ref.getattr(action_space_attr)?;

            let observation_space = crate::spaces::parse_space(&obs_space_py)?;
            let action_space = crate::spaces::parse_space(&action_space_py)?;

            // Build spaces spec
            let env_id = if env_ref.hasattr("spec")? {
                let spec = env_ref.getattr("spec")?;
                if !spec.is_none() && spec.hasattr("id")? {
                    spec.getattr("id")?.extract::<String>().unwrap_or_default()
                } else {
                    String::from("UnknownEnv-v1")
                }
            } else {
                String::from("UnknownEnv-v1")
            };

            let env_contract = EnvContract {
                id: env_id,
                observation_space: Some(observation_space.clone()),
                action_space: Some(action_space.clone()),
                metadata: extract_optional_meta_attr(env_ref, "metadata")?,
                render_mode: extract_render_mode(env_ref)?,
                num_envs: num_envs as u32,
            };

            Ok(Self {
                env,
                observation_space,
                action_space,
                env_contract,
                num_envs,
                uses_vector_api,
                profiler,
            })
        })
    }

    fn ensure_single_env(&self, action: &str) -> Result<(), EnvError> {
        if self.uses_single_env_api() {
            return Ok(());
        }

        Err(EnvError::new(
            EnvErrorCode::Internal,
            format!("{action} is only supported for single-env native adaptation"),
        ))
    }

    fn uses_single_env_api(&self) -> bool {
        self.num_envs == 1 && !self.uses_vector_api
    }
}

pub fn build_server_env(env: Py<PyAny>) -> PyResult<PyServerEnv> {
    let env = PyEnvironment::new(env)?;
    if env.uses_single_env_api() {
        Ok(PyServerEnv::Single(PySingleEnv(env)))
    } else {
        Ok(PyServerEnv::Vector(PyVectorEnv(env)))
    }
}

fn extract_optional_meta_attr(
    obj: &Bound<'_, PyAny>,
    attr_name: &str,
) -> PyResult<Option<MetaMap>> {
    if !obj.hasattr(attr_name)? {
        return Ok(None);
    }

    let value = obj.getattr(attr_name)?;
    if value.is_none() {
        return Ok(None);
    }

    Ok(Some(py_any_to_meta_map(&value)?))
}

fn extract_render_mode(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    if !obj.hasattr("render_mode")? {
        return Ok(String::new());
    }

    let value = obj.getattr("render_mode")?;
    if value.is_none() {
        return Ok(String::new());
    }

    value.extract::<String>()
}

fn env_error_to_runtime_error(error: EnvError) -> EnvRuntimeError {
    match error.code {
        EnvErrorCode::InvalidAction => EnvRuntimeError::InvalidValue(error.message),
        _ => EnvRuntimeError::Runtime(error.message),
    }
}

type SingleStepResultParts<'py> = (Bound<'py, PyAny>, f64, bool, bool, Bound<'py, PyAny>);
type VectorStepResultParts<'py> = (
    Bound<'py, PyAny>,
    Vec<f64>,
    Vec<bool>,
    Vec<bool>,
    Bound<'py, PyAny>,
);

fn normalize_reset_result<'py>(
    py: Python<'py>,
    result: Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    if let Ok(tuple) = result.cast::<PyTuple>()
        && tuple.len() == 2
    {
        let info = tuple.get_item(1)?;
        if info.is_none() || info.cast::<PyDict>().is_ok() {
            return Ok((tuple.get_item(0)?, info));
        }
    }

    Ok((result, PyDict::new(py).into_any()))
}

fn normalize_single_step_result<'py>(
    result: Bound<'py, PyAny>,
) -> PyResult<SingleStepResultParts<'py>> {
    let tuple = result.cast::<PyTuple>()?;
    match tuple.len() {
        5 => Ok((
            tuple.get_item(0)?,
            tuple.get_item(1)?.extract::<f64>()?,
            tuple.get_item(2)?.extract::<bool>()?,
            tuple.get_item(3)?.extract::<bool>()?,
            tuple.get_item(4)?,
        )),
        4 => {
            let done = tuple.get_item(2)?.extract::<bool>()?;
            let info = tuple.get_item(3)?;
            let truncated = done && legacy_single_truncated(&info)?;
            Ok((
                tuple.get_item(0)?,
                tuple.get_item(1)?.extract::<f64>()?,
                done && !truncated,
                truncated,
                info,
            ))
        }
        len => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "env.step() must return 4 legacy Gym values or 5 Gymnasium values, got {len}"
        ))),
    }
}

fn normalize_vector_step_result<'py>(
    result: Bound<'py, PyAny>,
    num_envs: usize,
) -> PyResult<VectorStepResultParts<'py>> {
    let tuple = result.cast::<PyTuple>()?;
    match tuple.len() {
        5 => Ok((
            tuple.get_item(0)?,
            tuple.get_item(1)?.extract::<Vec<f64>>()?,
            tuple.get_item(2)?.extract::<Vec<bool>>()?,
            tuple.get_item(3)?.extract::<Vec<bool>>()?,
            tuple.get_item(4)?,
        )),
        4 => {
            let done = tuple.get_item(2)?.extract::<Vec<bool>>()?;
            let info = tuple.get_item(3)?;
            let truncated = legacy_vector_truncated(&info, num_envs)?;
            let terminated = done
                .iter()
                .zip(truncated.iter())
                .map(|(done, truncated)| *done && !*truncated)
                .collect();
            Ok((
                tuple.get_item(0)?,
                tuple.get_item(1)?.extract::<Vec<f64>>()?,
                terminated,
                truncated,
                info,
            ))
        }
        len => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "env.step() must return 4 legacy Gym values or 5 Gymnasium values, got {len}"
        ))),
    }
}

fn legacy_single_truncated(info: &Bound<'_, PyAny>) -> PyResult<bool> {
    if let Ok(dict) = info.cast::<PyDict>()
        && let Some(value) = dict.get_item("TimeLimit.truncated")?
    {
        return value.extract::<bool>();
    }
    Ok(false)
}

fn legacy_vector_truncated(info: &Bound<'_, PyAny>, num_envs: usize) -> PyResult<Vec<bool>> {
    if let Ok(dict) = info.cast::<PyDict>()
        && let Some(value) = dict.get_item("TimeLimit.truncated")?
    {
        if let Ok(values) = value.extract::<Vec<bool>>() {
            return Ok(values);
        }
        if value.hasattr("tolist")? {
            return value.call_method0("tolist")?.extract::<Vec<bool>>();
        }
        if let Ok(value) = value.extract::<bool>() {
            return Ok(vec![value; num_envs]);
        }
    }
    Ok(vec![false; num_envs])
}

fn wrap_single_vector_action<'py>(
    py: Python<'py>,
    action: Bound<'py, PyAny>,
    action_space: &SpaceSpec,
) -> PyResult<Bound<'py, PyAny>> {
    if matches!(
        action_space.spec.as_ref(),
        Some(rlmesh_spaces::v1::spaces::space_spec::Spec::Box(_))
            | Some(rlmesh_spaces::v1::spaces::space_spec::Spec::Discrete(_))
            | Some(rlmesh_spaces::v1::spaces::space_spec::Spec::MultiBinary(_))
            | Some(rlmesh_spaces::v1::spaces::space_spec::Spec::MultiDiscrete(
                _
            ))
    ) && let Ok(numpy) = py.import("numpy")
    {
        if action.hasattr("shape")? {
            return numpy.getattr("expand_dims")?.call1((action, 0));
        }
        return numpy.getattr("array")?.call1((vec![action.unbind()],));
    }

    Ok(PyList::new(py, [action.unbind()])?.into_any())
}

fn encode_render_png(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Option<Vec<u8>>> {
    if value.is_none() {
        return Ok(None);
    }

    let numpy = py.import("numpy")?;
    let mut array = if value.hasattr("tobytes")? && value.hasattr("shape")? {
        value.clone()
    } else {
        numpy.call_method1("array", (value,))?
    };

    array = array.call_method1("astype", ("uint8",))?;
    let shape = array.getattr("shape")?.extract::<Vec<usize>>()?;
    let bytes = array.call_method0("tobytes")?.extract::<Vec<u8>>()?;

    let (width, height, color_type, data) = match shape.as_slice() {
        [height, width, 3] => (*width as u32, *height as u32, ColorType::Rgb8, bytes),
        [height, width, 4] => (*width as u32, *height as u32, ColorType::Rgba8, bytes),
        [3, height, width] => (
            *width as u32,
            *height as u32,
            ColorType::Rgb8,
            chw_to_hwc(bytes, 3, *height, *width),
        ),
        [4, height, width] => (
            *width as u32,
            *height as u32,
            ColorType::Rgba8,
            chw_to_hwc(bytes, 4, *height, *width),
        ),
        [height, width] => (*width as u32, *height as u32, ColorType::L8, bytes),
        _ => return Ok(None),
    };

    let mut encoded = Vec::new();
    PngEncoder::new(&mut encoded)
        .write_image(&data, width, height, color_type.into())
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;
    Ok(Some(encoded))
}

fn chw_to_hwc(bytes: Vec<u8>, channels: usize, height: usize, width: usize) -> Vec<u8> {
    let plane = height * width;
    let mut out = Vec::with_capacity(bytes.len());
    for index in 0..plane {
        for channel in 0..channels {
            out.push(bytes[channel * plane + index]);
        }
    }
    out
}

impl PyEnvironment {
    async fn reset_single(
        &mut self,
        req: SingleResetRequest,
    ) -> Result<SingleResetResult, EnvError> {
        self.ensure_single_env("reset")?;

        let span = tracing::info_span!("rlmesh.server.reset", num_envs = self.num_envs);
        let _enter = span.enter();
        let total_guard = self.profiler.start("server.reset.total");
        let env = Python::attach(|py| self.env.clone_ref(py));
        let observation_space = self.observation_space.clone();
        let seed = req.seed;
        let options = req.options;
        let profiler = Arc::clone(&self.profiler);

        let result = tokio::task::spawn_blocking(move || {
            Python::attach(|py| {
                let env_ref = env.bind(py);
                let kwargs = PyDict::new(py);
                if let Some(seed) = seed {
                    kwargs.set_item("seed", seed)?;
                }
                if let Some(options) = options.as_ref() {
                    kwargs.set_item("options", meta_map_to_pydict(py, options)?)?;
                }

                let call_guard = profiler.start("server.reset.python_call");
                let result = env_ref.call_method("reset", (), Some(&kwargs))?;
                let _ = call_guard.finish(0);

                let (obs, info) = normalize_reset_result(py, result)?;

                let encode_guard = profiler.start("server.reset.encode_obs");
                let observation = py_any_to_space_value_with_backend(
                    py,
                    &obs,
                    &observation_space,
                    ValueBackend::Auto,
                )?;
                let _ = encode_guard.finish(native_value_size(&observation));

                let info = if info.is_none() {
                    None
                } else {
                    Some(py_any_to_meta_map(&info)?)
                };

                Ok::<_, PyErr>((observation, info))
            })
        })
        .await
        .map_err(|e| {
            EnvError::new(
                EnvErrorCode::Internal,
                format!("reset task panicked: {}", e),
            )
        })?
        .map_err(|e: PyErr| {
            EnvError::new(EnvErrorCode::Internal, format!("reset failed: {}", e))
        })?;

        let (observation, info) = result;
        let _ = total_guard
            .finish(native_value_size(&observation) + info.as_ref().map(|m| m.len()).unwrap_or(0));

        Ok(SingleResetResult {
            observation: Some(observation),
            info,
            episode_id: None,
        })
    }

    async fn step_single(&mut self, req: SingleStepRequest) -> Result<SingleStepResult, EnvError> {
        self.ensure_single_env("step")?;

        let span = tracing::info_span!("rlmesh.server.step", num_envs = self.num_envs);
        let _enter = span.enter();
        let total_guard = self.profiler.start("server.step.total");
        let env = Python::attach(|py| self.env.clone_ref(py));
        let observation_space = self.observation_space.clone();
        let action_space = self.action_space.clone();
        let action = req.action;
        let action_size = action.as_ref().map(native_value_size).unwrap_or(0);
        let profiler = Arc::clone(&self.profiler);

        let result = tokio::task::spawn_blocking(move || {
            Python::attach(|py| {
                let env_ref = env.bind(py);

                let decode_guard = profiler.start("server.step.decode_action");
                let action = match action.as_ref() {
                    Some(action) => space_value_to_py_with_backend(
                        py,
                        action,
                        &action_space,
                        ValueBackend::Auto,
                    )?
                    .unbind(),
                    None => py.None(),
                };
                let _ = decode_guard.finish(action_size);

                let call_guard = profiler.start("server.step.python_call");
                let result = env_ref.call_method1("step", (action.bind(py),))?;
                let _ = call_guard.finish(0);

                let (obs, reward, terminated, truncated, info) =
                    normalize_single_step_result(result)?;

                let encode_guard = profiler.start("server.step.encode_obs");
                let observation = py_any_to_space_value_with_backend(
                    py,
                    &obs,
                    &observation_space,
                    ValueBackend::Auto,
                )?;
                let _ = encode_guard.finish(native_value_size(&observation));

                let info = if info.is_none() {
                    None
                } else {
                    Some(py_any_to_meta_map(&info)?)
                };

                Ok::<_, PyErr>((observation, reward, terminated, truncated, info))
            })
        })
        .await
        .map_err(|e| EnvError::new(EnvErrorCode::Internal, format!("step task panicked: {}", e)))?
        .map_err(|e: PyErr| EnvError::new(EnvErrorCode::Internal, format!("step failed: {}", e)))?;

        let (observation, reward, terminated, truncated, info) = result;
        let _ = total_guard
            .finish(native_value_size(&observation) + info.as_ref().map(|m| m.len()).unwrap_or(0));

        Ok(SingleStepResult {
            observation: Some(observation),
            reward,
            terminated,
            truncated,
            info,
        })
    }

    async fn render_single(
        &mut self,
        req: SingleRenderRequest,
    ) -> Result<SingleRenderResult, EnvError> {
        self.ensure_single_env("render")?;
        if matches!(req.env_index, Some(index) if index != 0) {
            return Err(EnvError::new(
                EnvErrorCode::InvalidAction,
                format!(
                    "single-env render only supports env_index 0, got {}",
                    req.env_index.unwrap_or_default()
                ),
            ));
        }

        let span = tracing::info_span!("rlmesh.server.render", num_envs = self.num_envs);
        let _enter = span.enter();
        let total_guard = self.profiler.start("server.render.total");
        let env = Python::attach(|py| self.env.clone_ref(py));
        let profiler = Arc::clone(&self.profiler);

        let result = tokio::task::spawn_blocking(move || {
            Python::attach(|py| {
                let env_ref = env.bind(py);
                if env_ref.hasattr("render")? {
                    let call_guard = profiler.start("server.render.python_call");
                    let frame = env_ref.call_method0("render")?;
                    let _ = call_guard.finish(0);

                    let encode_guard = profiler.start("server.render.encode_frame");
                    let encoded = encode_render_png(py, &frame)?;
                    let encoded_len = encoded.as_ref().map(|raw| raw.len()).unwrap_or(0);
                    let _ = encode_guard.finish(encoded_len);
                    return Ok::<_, PyErr>(encoded);
                }

                Ok::<_, PyErr>(None)
            })
        })
        .await
        .map_err(|e| {
            EnvError::new(
                EnvErrorCode::Internal,
                format!("render task panicked: {}", e),
            )
        })?
        .map_err(|e: PyErr| {
            EnvError::new(EnvErrorCode::Internal, format!("render failed: {}", e))
        })?;

        let frame_bytes = result.as_ref().map(|raw| raw.len()).unwrap_or(0);
        let _ = total_guard.finish(frame_bytes);

        Ok(SingleRenderResult {
            frame: result.map(|raw| NativeRenderFrame { png_frame: raw }),
        })
    }

    async fn close_single(
        &mut self,
        _req: SingleCloseRequest,
    ) -> Result<SingleCloseResult, EnvError> {
        self.ensure_single_env("close")?;

        let span = tracing::info_span!("rlmesh.server.close", num_envs = self.num_envs);
        let _enter = span.enter();
        let total_guard = self.profiler.start("server.close.total");
        let env = Python::attach(|py| self.env.clone_ref(py));
        let profiler = Arc::clone(&self.profiler);

        tokio::task::spawn_blocking(move || {
            Python::attach(|py| {
                let env_ref = env.bind(py);
                let call_guard = profiler.start("server.close.python_call");
                env_ref.call_method0("close")?;
                let _ = call_guard.finish(0);
                Ok::<_, PyErr>(())
            })
        })
        .await
        .map_err(|e| {
            EnvError::new(
                EnvErrorCode::Internal,
                format!("close task panicked: {}", e),
            )
        })?
        .map_err(|e: PyErr| {
            EnvError::new(EnvErrorCode::Internal, format!("close failed: {}", e))
        })?;

        let _ = total_guard.finish(0);
        self.profiler.log_summary_once();
        Ok(SingleCloseResult)
    }

    async fn reset_vector(
        &mut self,
        req: EnvResetRequest,
    ) -> Result<EnvResetResult, EnvRuntimeError> {
        let span = tracing::info_span!("rlmesh.server.reset", num_envs = self.num_envs);
        let _enter = span.enter();
        let total_guard = self.profiler.start("server.reset.total");
        let env = Python::attach(|py| self.env.clone_ref(py));
        let observation_space = self.observation_space.clone();
        let seeds = req.seeds;
        let options = req.options;
        let num_envs = self.num_envs;
        let profiler = Arc::clone(&self.profiler);

        let result = tokio::task::spawn_blocking(move || {
            Python::attach(|py| {
                let env_ref = env.bind(py);
                let kwargs = PyDict::new(py);
                if !seeds.is_empty() {
                    if seeds.len() == 1 {
                        kwargs.set_item("seed", seeds[0])?;
                    } else {
                        kwargs.set_item("seed", seeds.clone())?;
                    }
                }
                if let Some(options) = options.as_ref() {
                    kwargs.set_item("options", meta_map_to_pydict(py, options)?)?;
                }

                let call_guard = profiler.start("server.reset.python_call");
                let result = env_ref.call_method("reset", (), Some(&kwargs))?;
                let _ = call_guard.finish(0);

                let (obs, info) = normalize_reset_result(py, result)?;

                let encode_guard = profiler.start("server.reset.encode_obs");
                let observations = py_any_to_batched_space_values_with_backend(
                    py,
                    &obs,
                    &observation_space,
                    num_envs,
                    ValueBackend::Auto,
                )?;
                let obs_bytes = observations.iter().map(native_value_size).sum::<usize>();
                let _ = encode_guard.finish(obs_bytes);

                let info = if info.is_none() {
                    None
                } else {
                    Some(py_any_to_meta_map(&info)?)
                };

                Ok::<_, PyErr>((observations, info, obs_bytes))
            })
        })
        .await
        .map_err(|e| EnvRuntimeError::Runtime(format!("reset task panicked: {e}")))?
        .map_err(|e: PyErr| EnvRuntimeError::Runtime(format!("reset failed: {e}")))?;

        let (observations, info, obs_bytes) = result;
        let _ = total_guard.finish(obs_bytes);

        Ok(EnvResetResult {
            observations,
            info,
            episode_ids: vec![],
        })
    }

    async fn step_vector(&mut self, req: EnvStepRequest) -> Result<EnvStepResult, EnvRuntimeError> {
        if req.actions.len() != self.num_envs {
            return Err(EnvRuntimeError::InvalidValue(format!(
                "expected {} actions, got {}",
                self.num_envs,
                req.actions.len()
            )));
        }

        let span = tracing::info_span!("rlmesh.server.step", num_envs = self.num_envs);
        let _enter = span.enter();
        let total_guard = self.profiler.start("server.step.total");
        let env = Python::attach(|py| self.env.clone_ref(py));
        let action_space = self.action_space.clone();
        let observation_space = self.observation_space.clone();
        let actions = req.actions;
        let num_envs = self.num_envs;
        let uses_vector_api = self.uses_vector_api;
        let profiler = Arc::clone(&self.profiler);

        let result = tokio::task::spawn_blocking(move || {
            Python::attach(|py| {
                let env_ref = env.bind(py);

                let decode_guard = profiler.start("server.step.decode_action");
                let action = batched_space_values_to_py_with_backend(
                    py,
                    &actions,
                    &action_space,
                    ValueBackend::Auto,
                )?;
                let action = if uses_vector_api && num_envs == 1 {
                    wrap_single_vector_action(py, action, &action_space)?
                } else {
                    action
                };
                let action_bytes = actions.iter().map(native_value_size).sum::<usize>();
                let _ = decode_guard.finish(action_bytes);

                let call_guard = profiler.start("server.step.python_call");
                let result = env_ref.call_method1("step", (&action,))?;
                let _ = call_guard.finish(0);

                let (obs, rewards, terminated, truncated, info) =
                    normalize_vector_step_result(result, num_envs)?;

                let encode_guard = profiler.start("server.step.encode_obs");
                let observations = py_any_to_batched_space_values_with_backend(
                    py,
                    &obs,
                    &observation_space,
                    num_envs,
                    ValueBackend::Auto,
                )?;
                let obs_bytes = observations.iter().map(native_value_size).sum::<usize>();
                let _ = encode_guard.finish(obs_bytes);

                let info = if info.is_none() {
                    None
                } else {
                    Some(py_any_to_meta_map(&info)?)
                };

                Ok::<_, PyErr>((
                    observations,
                    rewards,
                    terminated,
                    truncated,
                    info,
                    action_bytes,
                    obs_bytes,
                ))
            })
        })
        .await
        .map_err(|e| EnvRuntimeError::Runtime(format!("step task panicked: {e}")))?
        .map_err(|e: PyErr| EnvRuntimeError::Runtime(format!("step failed: {e}")))?;

        let (observations, rewards, terminated, truncated, info, action_bytes, obs_bytes) = result;
        let _ = total_guard.finish(action_bytes + obs_bytes);

        Ok(EnvStepResult {
            observations,
            rewards,
            terminated,
            truncated,
            info,
            completed_episodes: vec![],
            episode_ids: vec![],
        })
    }

    async fn render_vector(
        &mut self,
        _req: RenderRequest,
    ) -> Result<RenderResult, EnvRuntimeError> {
        let span = tracing::info_span!("rlmesh.server.render", num_envs = self.num_envs);
        let _enter = span.enter();
        let total_guard = self.profiler.start("server.render.total");
        let env = Python::attach(|py| self.env.clone_ref(py));
        let profiler = Arc::clone(&self.profiler);

        let result = tokio::task::spawn_blocking(move || {
            Python::attach(|py| {
                let env_ref = env.bind(py);

                if env_ref.hasattr("render")? {
                    let call_guard = profiler.start("server.render.python_call");
                    let frame = env_ref.call_method0("render")?;
                    let _ = call_guard.finish(0);

                    let encode_guard = profiler.start("server.render.encode_frame");
                    let encoded = encode_render_png(py, &frame)?;
                    let encoded_len = encoded.as_ref().map(|raw| raw.len()).unwrap_or(0);
                    let _ = encode_guard.finish(encoded_len);
                    return Ok::<_, PyErr>(encoded);
                }

                Ok::<_, PyErr>(None)
            })
        })
        .await
        .map_err(|e| EnvRuntimeError::Runtime(format!("render task panicked: {e}")))?
        .map_err(|e: PyErr| EnvRuntimeError::Runtime(format!("render failed: {e}")))?;

        let frame_bytes = result.as_ref().map(|raw| raw.len()).unwrap_or(0);
        let _ = total_guard.finish(frame_bytes);

        Ok(RenderResult {
            frame: result.map(|raw| NativeRenderFrame { png_frame: raw }),
        })
    }

    async fn close_vector(
        &mut self,
        _req: CloseRequest,
    ) -> Result<EnvCloseResult, EnvRuntimeError> {
        let span = tracing::info_span!("rlmesh.server.close", num_envs = self.num_envs);
        let _enter = span.enter();
        let total_guard = self.profiler.start("server.close.total");
        let env = Python::attach(|py| self.env.clone_ref(py));
        let profiler = Arc::clone(&self.profiler);

        tokio::task::spawn_blocking(move || {
            Python::attach(|py| {
                let env_ref = env.bind(py);
                let call_guard = profiler.start("server.close.python_call");
                env_ref.call_method0("close")?;
                let _ = call_guard.finish(0);
                Ok::<_, PyErr>(())
            })
        })
        .await
        .map_err(|e| EnvRuntimeError::Runtime(format!("close task panicked: {e}")))?
        .map_err(|e: PyErr| EnvRuntimeError::Runtime(format!("close failed: {e}")))?;

        let _ = total_guard.finish(0);
        self.profiler.log_summary_once();

        Ok(EnvCloseResult {
            final_episodes: vec![],
        })
    }
}

impl Drop for PyEnvironment {
    fn drop(&mut self) {
        self.profiler.log_summary_once();
    }
}

#[async_trait]
impl RLMeshEnv for PySingleEnv {
    fn observation_space(&self) -> &SpaceSpec {
        &self.0.observation_space
    }

    fn action_space(&self) -> &SpaceSpec {
        &self.0.action_space
    }

    fn num_envs(&self) -> usize {
        1
    }

    fn env_contract(&self) -> &EnvContract {
        &self.0.env_contract
    }

    async fn reset(&mut self, req: EnvResetRequest) -> Result<EnvResetResult, EnvRuntimeError> {
        let result = self
            .0
            .reset_single(SingleResetRequest {
                seed: req.seeds.first().copied(),
                options: req.options,
                timeout_ms: req.timeout_ms,
            })
            .await
            .map_err(env_error_to_runtime_error)?;

        Ok(EnvResetResult {
            observations: result.observation.into_iter().collect(),
            info: result.info,
            episode_ids: result.episode_id.into_iter().collect(),
        })
    }

    async fn step(&mut self, req: EnvStepRequest) -> Result<EnvStepResult, EnvRuntimeError> {
        let action = req.actions.into_iter().next();
        let result = self
            .0
            .step_single(SingleStepRequest {
                action,
                timeout_ms: req.timeout_ms,
            })
            .await
            .map_err(env_error_to_runtime_error)?;

        Ok(EnvStepResult {
            observations: result.observation.into_iter().collect(),
            rewards: vec![result.reward],
            terminated: vec![result.terminated],
            truncated: vec![result.truncated],
            info: result.info,
            completed_episodes: vec![],
            episode_ids: vec![],
        })
    }

    async fn render(&mut self, req: RenderRequest) -> Result<RenderResult, EnvRuntimeError> {
        self.0
            .render_single(req)
            .await
            .map_err(env_error_to_runtime_error)
    }

    async fn close(&mut self, req: CloseRequest) -> Result<EnvCloseResult, EnvRuntimeError> {
        let _ = self
            .0
            .close_single(req)
            .await
            .map_err(env_error_to_runtime_error)?;
        Ok(EnvCloseResult {
            final_episodes: vec![],
        })
    }
}

#[async_trait]
impl RLMeshEnv for PyVectorEnv {
    fn observation_space(&self) -> &SpaceSpec {
        &self.0.observation_space
    }

    fn action_space(&self) -> &SpaceSpec {
        &self.0.action_space
    }

    fn num_envs(&self) -> usize {
        self.0.num_envs
    }

    fn env_contract(&self) -> &EnvContract {
        &self.0.env_contract
    }

    async fn reset(&mut self, req: EnvResetRequest) -> Result<EnvResetResult, EnvRuntimeError> {
        self.0.reset_vector(req).await
    }

    async fn step(&mut self, req: EnvStepRequest) -> Result<EnvStepResult, EnvRuntimeError> {
        self.0.step_vector(req).await
    }

    async fn render(&mut self, req: RenderRequest) -> Result<RenderResult, EnvRuntimeError> {
        self.0.render_vector(req).await
    }

    async fn close(&mut self, req: CloseRequest) -> Result<EnvCloseResult, EnvRuntimeError> {
        self.0.close_vector(req).await
    }
}

fn native_value_size(value: &rlmesh_spaces::v1::SpaceValue) -> usize {
    match value {
        rlmesh_spaces::v1::SpaceValue::Box(value) => value.data.len(),
        rlmesh_spaces::v1::SpaceValue::Discrete(_) => std::mem::size_of::<i64>(),
        rlmesh_spaces::v1::SpaceValue::MultiBinary(values) => values.len(),
        rlmesh_spaces::v1::SpaceValue::MultiDiscrete(values) => {
            values.len() * std::mem::size_of::<i64>()
        }
        rlmesh_spaces::v1::SpaceValue::Text(value) => value.len(),
        rlmesh_spaces::v1::SpaceValue::Dict(values) => values.values().map(native_value_size).sum(),
        rlmesh_spaces::v1::SpaceValue::Tuple(values) => values.iter().map(native_value_size).sum(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_reset_obs_only_gets_empty_info() {
        Python::attach(|py| {
            let result = 7i64.into_pyobject(py).unwrap().into_any();
            let (obs, info) = normalize_reset_result(py, result).unwrap();

            assert_eq!(obs.extract::<i64>().unwrap(), 7);
            assert!(info.cast::<PyDict>().unwrap().is_empty());
        });
    }

    #[test]
    fn modern_reset_tuple_preserves_info() {
        Python::attach(|py| {
            let info = PyDict::new(py);
            info.set_item("seed", 123).unwrap();
            let result = (7i64, info).into_pyobject(py).unwrap().into_any();
            let (obs, info) = normalize_reset_result(py, result).unwrap();

            assert_eq!(obs.extract::<i64>().unwrap(), 7);
            assert_eq!(
                info.cast::<PyDict>()
                    .unwrap()
                    .get_item("seed")
                    .unwrap()
                    .unwrap()
                    .extract::<i64>()
                    .unwrap(),
                123
            );
        });
    }

    #[test]
    fn legacy_single_step_done_can_map_to_truncated() {
        Python::attach(|py| {
            let info = PyDict::new(py);
            info.set_item("TimeLimit.truncated", true).unwrap();
            let result = (1i64, 1.0f64, true, info)
                .into_pyobject(py)
                .unwrap()
                .into_any();
            let (obs, reward, terminated, truncated, _info) =
                normalize_single_step_result(result).unwrap();

            assert_eq!(obs.extract::<i64>().unwrap(), 1);
            assert_eq!(reward, 1.0);
            assert!(!terminated);
            assert!(truncated);
        });
    }

    #[test]
    fn legacy_vector_step_done_maps_each_time_limit_flag() {
        Python::attach(|py| {
            let info = PyDict::new(py);
            info.set_item("TimeLimit.truncated", vec![true, false])
                .unwrap();
            let result = (vec![1i64, 1], vec![1.0f64, 2.0], vec![true, true], info)
                .into_pyobject(py)
                .unwrap()
                .into_any();
            let (_obs, rewards, terminated, truncated, _info) =
                normalize_vector_step_result(result, 2).unwrap();

            assert_eq!(rewards, vec![1.0, 2.0]);
            assert_eq!(terminated, vec![false, true]);
            assert_eq!(truncated, vec![true, false]);
        });
    }
}
