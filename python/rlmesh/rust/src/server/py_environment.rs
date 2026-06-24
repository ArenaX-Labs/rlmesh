//! PyEnvironment - Rust adapter for Python gymnasium environments.

use async_trait::async_trait;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rlmesh::{
    Env as RLMeshEnv, VectorCloseResult as EnvCloseResult, VectorEnv as RLMeshVectorEnv,
    VectorResetRequest as EnvResetRequest, VectorResetResult as EnvResetResult,
    VectorStepRequest as EnvStepRequest, VectorStepResult as EnvStepResult,
};
use rlmesh_grpc::error::{EnvError, EnvErrorCode};
use rlmesh_spaces::errors::EnvRuntimeError;
use rlmesh_spaces::spaces::{PolicyOutcome, SpaceSpec, ValidationPolicy};
use rlmesh_spaces::{
    CloseRequest, CloseRequest as SingleCloseRequest, CloseResult as SingleCloseResult,
    EnvContract, MetaMap, MetaValue, RenderFrame as NativeRenderFrame, RenderRequest,
    RenderRequest as SingleRenderRequest, RenderResult, RenderResult as SingleRenderResult,
    ResetRequest as SingleResetRequest, ResetResult as SingleResetResult, SpaceValue,
    StepRequest as SingleStepRequest, StepResult as SingleStepResult,
};
use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use super::conversion::{
    derive_autoreset_mode, encode_render_png, extract_optional_meta_attr, extract_render_mode,
    normalize_reset_result, normalize_single_step_result, normalize_vector_step_result,
};
use crate::spaces::{
    ValueBackend, batched_space_values_to_py_with_backend, meta_map_to_pydict,
    py_any_to_batched_space_values_with_backend, py_any_to_meta_map,
    py_any_to_space_value_with_backend, space_value_to_py_with_backend,
};
use crate::telemetry::ProfileCollector;
use crate::types::space_value_size as native_value_size;

/// Reserved info-map key carrying value-conformance warnings (2026.06 edition).
const CONFORMANCE_WARNING_KEY: &str = "rlmesh.conformance.warning";

/// One value-conformance warning surfaced in the info map.
struct ConformanceWarning {
    kind: String,
    path: String,
    detail: String,
}

/// Resolve the serving-side validation policy. `RLMESH_VALIDATION_POLICY` may
/// select `strict` or `off`; the default (and any other value) is `warn`.
fn validation_policy_from_env() -> ValidationPolicy {
    let raw = match std::env::var("RLMESH_VALIDATION_POLICY") {
        Ok(raw) => raw,
        Err(_) => return ValidationPolicy::Warn,
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "strict" => ValidationPolicy::Strict,
        "off" => ValidationPolicy::Off,
        // An unrecognized value defaults to warn, matching the env wire path
        // (crates/rlmesh/src/env/wire.rs); no stderr print (clippy::print_stderr
        // is denied workspace-wide).
        _ => ValidationPolicy::Warn,
    }
}

/// Merge conformance warnings into an info map under the reserved key, once per
/// `(kind, path)` per session.
fn inject_conformance_warnings(info: &mut Option<MetaMap>, warnings: Vec<ConformanceWarning>) {
    if warnings.is_empty() {
        return;
    }
    let entries = warnings
        .into_iter()
        .map(|warning| {
            MetaValue::Map(BTreeMap::from([
                ("kind".to_string(), MetaValue::String(warning.kind)),
                ("path".to_string(), MetaValue::String(warning.path)),
                ("detail".to_string(), MetaValue::String(warning.detail)),
            ]))
        })
        .collect();
    info.get_or_insert_with(BTreeMap::new).insert(
        CONFORMANCE_WARNING_KEY.to_string(),
        MetaValue::List(entries),
    );
}

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
    /// Serving-side validation policy for observation/action range deviations.
    policy: ValidationPolicy,
    /// Conformance-warning dedup: `(kind, path)` already reported this session.
    warned: HashSet<(String, String)>,
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

            let env_id = if env_ref.hasattr("spec")? {
                let spec = env_ref.getattr("spec")?;
                if !spec.is_none() && spec.hasattr("id")? {
                    spec.getattr("id")?
                        .extract::<String>()
                        .unwrap_or_else(|_| String::from("UnknownEnv-v1"))
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
                autoreset_mode: derive_autoreset_mode(env_ref)?,
            };

            Ok(Self {
                env,
                observation_space,
                action_space,
                env_contract,
                num_envs,
                uses_vector_api,
                profiler,
                policy: validation_policy_from_env(),
                warned: HashSet::new(),
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

    /// Validate one observation or action against `space` under the active
    /// policy. Returns an `EnvError` to reject (structural always; a range
    /// deviation under `strict`); records a deduped conformance warning to
    /// `warnings` for a range deviation under `warn`; otherwise accepts.
    fn enforce(
        policy: ValidationPolicy,
        warned: &mut HashSet<(String, String)>,
        space: &SpaceSpec,
        value: &SpaceValue,
        kind: &str,
        warnings: &mut Vec<ConformanceWarning>,
    ) -> Result<(), EnvError> {
        match policy.check(space, value) {
            PolicyOutcome::Accept => Ok(()),
            PolicyOutcome::Reject(err) => {
                let code = if kind == "action" {
                    EnvErrorCode::InvalidAction
                } else {
                    EnvErrorCode::Internal
                };
                Err(EnvError::new(code, err.to_string()))
            }
            PolicyOutcome::Warn(err) => {
                let path = err.path().to_string();
                if warned.insert((kind.to_string(), path.clone())) {
                    warnings.push(ConformanceWarning {
                        kind: kind.to_string(),
                        path,
                        detail: err.to_string(),
                    });
                }
                Ok(())
            }
        }
    }
}

pub fn build_scalar_server_env(env: Py<PyAny>) -> PyResult<PyServerEnv> {
    let env = PyEnvironment::new(env)?;
    if env.uses_single_env_api() {
        Ok(PyServerEnv::Single(PySingleEnv(env)))
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "EnvServer serves one environment. Use VectorEnvServer for vectorized environments.",
        ))
    }
}

pub fn build_vector_server_env(env: Py<PyAny>) -> PyResult<PyServerEnv> {
    let env = PyEnvironment::new(env)?;
    if env.uses_single_env_api() {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "VectorEnvServer requires a vectorized environment. Use EnvServer for one environment.",
        ))
    } else if env.num_envs < 2 {
        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "VectorEnvServer requires num_envs >= 2",
        ))
    } else {
        Ok(PyServerEnv::Vector(PyVectorEnv(env)))
    }
}

fn env_error_to_runtime_error(error: EnvError) -> EnvRuntimeError {
    match error.code {
        EnvErrorCode::InvalidAction => EnvRuntimeError::InvalidValue(error.message),
        _ => EnvRuntimeError::Runtime(error.message),
    }
}

/// Run `work` on a blocking thread under the GIL, mapping a join panic or a
/// `PyErr` through `into_err`. Wraps the `spawn_blocking` + `Python::attach` +
/// double `map_err` boilerplate every reset/step/render/close body shares.
async fn spawn_py<T, E, W, M>(phase: &str, work: W, into_err: M) -> Result<T, E>
where
    T: Send + 'static,
    W: FnOnce(Python<'_>) -> PyResult<T> + Send + 'static,
    M: Fn(String) -> E,
{
    tokio::task::spawn_blocking(move || Python::attach(work))
        .await
        .map_err(|e| into_err(format!("{phase} task panicked: {e}")))?
        .map_err(|e: PyErr| into_err(format!("{phase} failed: {e}")))
}

fn internal_env_err(message: String) -> EnvError {
    EnvError::new(EnvErrorCode::Internal, message)
}

impl PyEnvironment {
    #[tracing::instrument(name = "rlmesh.server.reset", skip_all, fields(num_envs = self.num_envs))]
    async fn reset_single(
        &mut self,
        req: SingleResetRequest,
    ) -> Result<SingleResetResult, EnvError> {
        self.ensure_single_env("reset")?;

        let total_guard = self.profiler.start("server.reset.total");
        let env = Python::attach(|py| self.env.clone_ref(py));
        let observation_space = self.observation_space.clone();
        let seed = req.seed;
        let options = req.options;
        let profiler = Arc::clone(&self.profiler);

        let (observation, mut info) = spawn_py(
            "reset",
            move |py| {
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

                let (obs, info) = normalize_reset_result(py, result, &observation_space)?;

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

                Ok((observation, info))
            },
            internal_env_err,
        )
        .await?;
        let _ = total_guard
            .finish(native_value_size(&observation) + info.as_ref().map(|m| m.len()).unwrap_or(0));

        let mut warnings = Vec::new();
        Self::enforce(
            self.policy,
            &mut self.warned,
            &self.observation_space,
            &observation,
            "observation",
            &mut warnings,
        )?;
        inject_conformance_warnings(&mut info, warnings);

        Ok(SingleResetResult {
            observation: Some(observation),
            info,
            episode_id: None,
        })
    }

    #[tracing::instrument(name = "rlmesh.server.step", skip_all, fields(num_envs = self.num_envs))]
    async fn step_single(&mut self, req: SingleStepRequest) -> Result<SingleStepResult, EnvError> {
        self.ensure_single_env("step")?;

        let mut warnings = Vec::new();
        if let Some(action) = req.action.as_ref() {
            Self::enforce(
                self.policy,
                &mut self.warned,
                &self.action_space,
                action,
                "action",
                &mut warnings,
            )?;
        }

        let total_guard = self.profiler.start("server.step.total");
        let env = Python::attach(|py| self.env.clone_ref(py));
        let observation_space = self.observation_space.clone();
        let action_space = self.action_space.clone();
        let action = req.action;
        let action_size = action.as_ref().map(native_value_size).unwrap_or(0);
        let profiler = Arc::clone(&self.profiler);

        let (observation, reward, terminated, truncated, mut info) = spawn_py(
            "step",
            move |py| {
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

                Ok((observation, reward, terminated, truncated, info))
            },
            internal_env_err,
        )
        .await?;
        let _ = total_guard
            .finish(native_value_size(&observation) + info.as_ref().map(|m| m.len()).unwrap_or(0));

        Self::enforce(
            self.policy,
            &mut self.warned,
            &self.observation_space,
            &observation,
            "observation",
            &mut warnings,
        )?;
        inject_conformance_warnings(&mut info, warnings);

        Ok(SingleStepResult {
            observation: Some(observation),
            reward,
            terminated,
            truncated,
            info,
        })
    }

    #[tracing::instrument(name = "rlmesh.server.render", skip_all, fields(num_envs = self.num_envs))]
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

        let total_guard = self.profiler.start("server.render.total");
        let env = Python::attach(|py| self.env.clone_ref(py));
        let profiler = Arc::clone(&self.profiler);

        let result = spawn_py(
            "render",
            move |py| {
                let env_ref = env.bind(py);
                if env_ref.hasattr("render")? {
                    let call_guard = profiler.start("server.render.python_call");
                    let frame = env_ref.call_method0("render")?;
                    let _ = call_guard.finish(0);

                    let encode_guard = profiler.start("server.render.encode_frame");
                    let encoded = encode_render_png(py, &frame)?;
                    let encoded_len = encoded.as_ref().map(|raw| raw.len()).unwrap_or(0);
                    let _ = encode_guard.finish(encoded_len);
                    return Ok(encoded);
                }

                Ok(None)
            },
            internal_env_err,
        )
        .await?;

        let frame_bytes = result.as_ref().map(|raw| raw.len()).unwrap_or(0);
        let _ = total_guard.finish(frame_bytes);

        Ok(SingleRenderResult {
            frame: result.map(|raw| NativeRenderFrame { frame: raw }),
        })
    }

    #[tracing::instrument(name = "rlmesh.server.close", skip_all, fields(num_envs = self.num_envs))]
    async fn close_single(
        &mut self,
        _req: SingleCloseRequest,
    ) -> Result<SingleCloseResult, EnvError> {
        self.ensure_single_env("close")?;

        let total_guard = self.profiler.start("server.close.total");
        let env = Python::attach(|py| self.env.clone_ref(py));
        let profiler = Arc::clone(&self.profiler);

        spawn_py(
            "close",
            move |py| {
                let env_ref = env.bind(py);
                let call_guard = profiler.start("server.close.python_call");
                env_ref.call_method0("close")?;
                let _ = call_guard.finish(0);
                Ok(())
            },
            internal_env_err,
        )
        .await?;

        let _ = total_guard.finish(0);
        self.profiler.log_summary_once();
        Ok(SingleCloseResult)
    }

    #[tracing::instrument(name = "rlmesh.server.reset", skip_all, fields(num_envs = self.num_envs))]
    async fn reset_vector(
        &mut self,
        req: EnvResetRequest,
    ) -> Result<EnvResetResult, EnvRuntimeError> {
        let total_guard = self.profiler.start("server.reset.total");
        let env = Python::attach(|py| self.env.clone_ref(py));
        let observation_space = self.observation_space.clone();
        let seeds = req.seeds;
        let options = req.options;
        let num_envs = self.num_envs;
        let profiler = Arc::clone(&self.profiler);

        let (observations, mut info, obs_bytes) = spawn_py(
            "reset",
            move |py| {
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

                let (obs, info) = normalize_reset_result(py, result, &observation_space)?;

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

                Ok((observations, info, obs_bytes))
            },
            EnvRuntimeError::Runtime,
        )
        .await?;
        let _ = total_guard.finish(obs_bytes);

        let mut warnings = Vec::new();
        for observation in &observations {
            Self::enforce(
                self.policy,
                &mut self.warned,
                &self.observation_space,
                observation,
                "observation",
                &mut warnings,
            )
            .map_err(env_error_to_runtime_error)?;
        }
        inject_conformance_warnings(&mut info, warnings);

        Ok(EnvResetResult {
            observations,
            info,
            episode_ids: vec![],
        })
    }

    #[tracing::instrument(name = "rlmesh.server.step", skip_all, fields(num_envs = self.num_envs))]
    async fn step_vector(&mut self, req: EnvStepRequest) -> Result<EnvStepResult, EnvRuntimeError> {
        if req.actions.len() != self.num_envs {
            return Err(EnvRuntimeError::InvalidValue(format!(
                "expected {} actions, got {}",
                self.num_envs,
                req.actions.len()
            )));
        }

        let mut warnings = Vec::new();
        for action in &req.actions {
            Self::enforce(
                self.policy,
                &mut self.warned,
                &self.action_space,
                action,
                "action",
                &mut warnings,
            )
            .map_err(env_error_to_runtime_error)?;
        }

        let total_guard = self.profiler.start("server.step.total");
        let env = Python::attach(|py| self.env.clone_ref(py));
        let action_space = self.action_space.clone();
        let observation_space = self.observation_space.clone();
        let actions = req.actions;
        let num_envs = self.num_envs;
        let profiler = Arc::clone(&self.profiler);

        let (observations, rewards, terminated, truncated, mut info, action_bytes, obs_bytes) =
            spawn_py(
                "step",
                move |py| {
                    let env_ref = env.bind(py);

                    let decode_guard = profiler.start("server.step.decode_action");
                    let action = batched_space_values_to_py_with_backend(
                        py,
                        &actions,
                        &action_space,
                        ValueBackend::Auto,
                    )?;
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

                    Ok((
                        observations,
                        rewards,
                        terminated,
                        truncated,
                        info,
                        action_bytes,
                        obs_bytes,
                    ))
                },
                EnvRuntimeError::Runtime,
            )
            .await?;
        let _ = total_guard.finish(action_bytes + obs_bytes);

        for observation in &observations {
            Self::enforce(
                self.policy,
                &mut self.warned,
                &self.observation_space,
                observation,
                "observation",
                &mut warnings,
            )
            .map_err(env_error_to_runtime_error)?;
        }
        inject_conformance_warnings(&mut info, warnings);

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

    #[tracing::instrument(name = "rlmesh.server.render", skip_all, fields(num_envs = self.num_envs))]
    async fn render_vector(
        &mut self,
        _req: RenderRequest,
    ) -> Result<RenderResult, EnvRuntimeError> {
        let total_guard = self.profiler.start("server.render.total");
        let env = Python::attach(|py| self.env.clone_ref(py));
        let profiler = Arc::clone(&self.profiler);

        let result = spawn_py(
            "render",
            move |py| {
                let env_ref = env.bind(py);

                if env_ref.hasattr("render")? {
                    let call_guard = profiler.start("server.render.python_call");
                    let frame = env_ref.call_method0("render")?;
                    let _ = call_guard.finish(0);

                    let encode_guard = profiler.start("server.render.encode_frame");
                    let encoded = encode_render_png(py, &frame)?;
                    let encoded_len = encoded.as_ref().map(|raw| raw.len()).unwrap_or(0);
                    let _ = encode_guard.finish(encoded_len);
                    return Ok(encoded);
                }

                Ok(None)
            },
            EnvRuntimeError::Runtime,
        )
        .await?;

        let frame_bytes = result.as_ref().map(|raw| raw.len()).unwrap_or(0);
        let _ = total_guard.finish(frame_bytes);

        Ok(RenderResult {
            frame: result.map(|raw| NativeRenderFrame { frame: raw }),
        })
    }

    #[tracing::instrument(name = "rlmesh.server.close", skip_all, fields(num_envs = self.num_envs))]
    async fn close_vector(
        &mut self,
        _req: CloseRequest,
    ) -> Result<EnvCloseResult, EnvRuntimeError> {
        let total_guard = self.profiler.start("server.close.total");
        let env = Python::attach(|py| self.env.clone_ref(py));
        let profiler = Arc::clone(&self.profiler);

        spawn_py(
            "close",
            move |py| {
                let env_ref = env.bind(py);
                let call_guard = profiler.start("server.close.python_call");
                env_ref.call_method0("close")?;
                let _ = call_guard.finish(0);
                Ok(())
            },
            EnvRuntimeError::Runtime,
        )
        .await?;

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

    fn env_contract(&self) -> &EnvContract {
        &self.0.env_contract
    }

    async fn reset(
        &mut self,
        req: SingleResetRequest,
    ) -> Result<SingleResetResult, EnvRuntimeError> {
        self.0
            .reset_single(req)
            .await
            .map_err(env_error_to_runtime_error)
    }

    async fn step(&mut self, req: SingleStepRequest) -> Result<SingleStepResult, EnvRuntimeError> {
        self.0
            .step_single(req)
            .await
            .map_err(env_error_to_runtime_error)
    }

    async fn render(
        &mut self,
        req: SingleRenderRequest,
    ) -> Result<SingleRenderResult, EnvRuntimeError> {
        self.0
            .render_single(req)
            .await
            .map_err(env_error_to_runtime_error)
    }

    async fn close(
        &mut self,
        req: SingleCloseRequest,
    ) -> Result<SingleCloseResult, EnvRuntimeError> {
        self.0
            .close_single(req)
            .await
            .map_err(env_error_to_runtime_error)
    }
}

#[async_trait]
impl RLMeshVectorEnv for PyVectorEnv {
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
